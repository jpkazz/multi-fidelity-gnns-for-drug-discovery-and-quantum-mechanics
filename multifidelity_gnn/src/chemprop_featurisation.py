# This module defines functions to generate feature vectors for atoms and bonds in a molecule using RDKit.
# It includes functionality for:
# 1. One-hot encoding of atom and bond properties with an additional category for uncommon values.
# 2. Creating feature vectors for individual atoms based on their atomic number, degree, formal charge, 
#    chirality, hybridization, hydrogen count, and aromaticity.
# 3. Creating feature vectors for bonds based on bond type, conjugation, ring membership, and stereochemistry.
# 4. A function to retrieve constant values for atom properties such as atomic number and hybridization types.

from rdkit import Chem
from typing import Sequence, Dict


def onek_encoding_unk(value: int, choices: Sequence):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * len(choices) # Initialize encoding with zero
    index = choices.index(value) if value in choices else -1 # Get index or set to -1 if not found
    encoding[index] = 1 # Set one-hot encoding

    return encoding


def atom_features(
    atom: Chem.rdchem.Atom,
    features_constants: Dict[str, Sequence],
    functional_groups=None,
):
    features = (
        onek_encoding_unk(atom.GetAtomicNum(), features_constants["atomic_num"]) # Atomic number encoding
        + onek_encoding_unk(atom.GetTotalDegree(), features_constants["degree"]) # Atom degree encoding
        + onek_encoding_unk(atom.GetFormalCharge(), features_constants["formal_charge"]) # Formal charge encoding
        + onek_encoding_unk(int(atom.GetChiralTag()), features_constants["chiral_tag"]) # Chirality encoding
        + onek_encoding_unk(int(atom.GetTotalNumHs()), features_constants["num_Hs"]) # Number of attached hydrogens
        + onek_encoding_unk(
            int(atom.GetHybridization()), features_constants["hybridization"] # Hybridization type
        )
        + [1 if atom.GetIsAromatic() else 0] # Aromaticity flag (1 if aromatic)
        + [atom.GetMass() * 0.01] # Scaled atomic mass for feature consistency
    )  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups # Add functional group features if provided
    return features


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        # If no bond, set first element to 1 and rest to 0
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        # Extract bond type and set feature vector based on bond characteristics
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None/present
            bt == Chem.rdchem.BondType.SINGLE, # Single bond flag
            bt == Chem.rdchem.BondType.DOUBLE, # Double bond flag
            bt == Chem.rdchem.BondType.TRIPLE, # Triple bond flag
            bt == Chem.rdchem.BondType.AROMATIC, # Aromatic bond flag
            (bond.GetIsConjugated() if bt is not None else 0), # Conjugation flag
            (bond.IsInRing() if bt is not None else 0), # Ring membership flag
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6))) # Stereochemistry encoding
    return fbond


def get_atom_constants(max_atomic_num: int):
    return {
        "atomic_num": list(range(max_atomic_num)), # Possible atomic numbers up to max_atomic_num
        "degree": [0, 1, 2, 3, 4, 5], # Possible atom degrees
        "formal_charge": [-1, -2, 1, 2, 0], # Possible formal charges
        "chiral_tag": [0, 1, 2, 3], # Possible chiral tags
        "num_Hs": [0, 1, 2, 3, 4], # Possible hydrogen counts
        "hybridization": [ # Possible hybridization states
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    }

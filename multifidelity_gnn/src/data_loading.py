import torch
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from rdkit import Chem
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.preprocessing import StandardScaler

from typing import Union, List, Tuple, Optional

from .chemprop_featurisation import atom_features, bond_features, get_atom_constants

# Function to remove stereochemistry information from SMILES strings.
def remove_smiles_stereo(s):
    mol = Chem.MolFromSmiles(s)
    Chem.rdmolops.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(mol)

# Class designed to preprocess molecular data into graph representations for PyTorch Geometric models
# It reads SMILES strings and auxiliary data from a CSV file, computes atom and bond features using RDKit.
# and constructs PyTorch Geometric Data objects for each molecule.
# The class supports optional label scaling, auxiliary data inclusion, and unique sample identifiers.
class GraphMoleculeDataset(TorchDataset):
    def __init__(
        self,
        csv_path: str,
        max_atom_num: int,
        smiles_column_name: str,
        label_column_name: Union[str, List[str]],
        auxiliary_data_column_name: Optional[str] = None, # Optional column for auxiliary data (e.g., embeddings or labels).
        lbl_or_emb: str = "lbl", # Specifies the type of auxiliary data ("lbl" for labels, "emb" for embeddings).
        scaler: Optional[StandardScaler] = None, # Optional scaler for normalizing labels.
        id_column: Optional[str] = None, # Optional column containing unique identifiers for each sample.
    ):
        super().__init__()
        # Check that lbl_or_emb is valid
        assert lbl_or_emb in [None, "lbl", "emb"]
        # Load the dataset and initialize parameters
        self.df = pd.read_csv(csv_path) # Load the dataset from the specified CSV file.
        self.smiles_column_name = smiles_column_name
        self.label_column_name = label_column_name
        self.atom_constants = get_atom_constants(max_atom_num)  # Atom-level feature constants for featurization.
        self.num_atom_features = (  # Calculate the total number of atom and bond features based on constants.
            sum(len(choices) for choices in self.atom_constants.values()) + 2
        )
        self.num_bond_features = 13 # Fixed number of bond features.
        # Store auxiliary data column name, auxiliary data type, scaler, and ID column.
        self.auxiliary_data_column_name = auxiliary_data_column_name
        self.lbl_or_emb = lbl_or_emb
        self.scaler = scaler
        self.id_column = id_column

    def __len__(self):
        # Return the number of samples in the dataset.
        return len(self.df)

    def __getitem__(self, idx: Union[torch.Tensor, slice, List]):
        """
        Retrieve a data point or a batch of data points by index.
        Parameters:
            idx (Union[torch.Tensor, slice, List]): Index or indices to fetch.
        Returns:
            List[GeometricData] or GeometricData: A single data point or a list of data points.
        """
        # Convert tensor or slice indices to a list of indices.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, slice):
            slice_step = idx.step if idx.step else 1
            idx = list(range(idx.start, idx.stop, slice_step))
        if not isinstance(idx, list):
            idx = [idx]
        # Select the rows from the DataFrame corresponding to the given indices.
        selected = self.df.iloc[idx]
        
        # Extract sample identifiers if an ID column is provided.
        if self.id_column:
            ids = selected[self.id_column].values
        smiles = selected[self.smiles_column_name].values

        if isinstance(self.label_column_name, (list, tuple)):
            num_tasks = len(self.label_column_name)
        else:
            num_tasks = 1
        targets = selected[self.label_column_name].values

        if self.scaler is not None:
            labels = torch.Tensor(self.scaler.transform(targets.reshape(-1, num_tasks)))
        else:
            labels = torch.Tensor(targets)

        # Process SMILES strings to remove stereochemistry and create RDKit molecule objects.
        smiles = [remove_smiles_stereo(s) for s in smiles]
        rdkit_mols = [Chem.MolFromSmiles(s) for s in smiles]

        # Extract auxiliary data if available and process it based on the type ("lbl" or "emb").
        aux_data = None
        if self.auxiliary_data_column_name:
            column_values = selected[self.auxiliary_data_column_name].values

            if self.lbl_or_emb and self.lbl_or_emb == "lbl":
                aux_data = torch.Tensor(column_values)

            elif self.lbl_or_emb == "emb":
                # Need to parse the NumPy array from the string stored in the DataFrame
                aux_data = torch.Tensor(
                    np.stack(
                        np.fromstring(
                            selected[self.auxiliary_data_column_name].values[0][1:-1],
                            sep=", ",
                        )
                    )
                )

        # Compute atom-level features for each molecule using RDKit.
        atom_feat = [
            torch.Tensor(
                [atom_features(atom, self.atom_constants) for atom in mol.GetAtoms()]
            )
            for mol in rdkit_mols
        ]

        # Compute bond-level features and adjacency matrices.
        edge_index = []
        bond_feat = []

        for mol in rdkit_mols:
            # Extract adjacency matrix and non-zero indices for edges.
            ei = torch.nonzero(
                torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))
            ).T
            # Extract bond features for each edge.
            bf = torch.Tensor(
                [
                    bond_features(
                        mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())
                    )
                    for i in range(ei.shape[1])
                ]
            )

            edge_index.append(ei)
            bond_feat.append(bf)

        # Create PyTorch Geometric Data objects for each molecule.
        geometric_data_points = [
            GeometricData(
                x=atom_feat[i], # Atom features as node attributes.
                edge_attr=bond_feat[i], # Bond features as edge attributes.
                edge_index=edge_index[i], # Edge connectivity information.
                y=labels[i], # Target labels.
                aux_data=aux_data, # Auxiliary data if available.
                iden=ids[i], # Sample identifiers if provided.
            )
            for i in range(len(atom_feat))
        ]

        # Add SMILES strings and auxiliary data to each Geometric Data object.
        for i, data_point in enumerate(geometric_data_points):
            data_point.smiles = smiles[i] # Store SMILES string for reference.
            data_point.aux_data = np.array([]) if aux_data is None else aux_data # Handle empty auxiliary data.

        # Return a single data point or a list of data points, depending on the input index type.
        if len(geometric_data_points) == 1:
            return geometric_data_points[0]
        return geometric_data_points

# Class designed to manage the loading, preprocessing, and batching of molecular data for graph-based machine learning models.
# It supports train-validation-test splits, label scaling using StandardScaler, and customization
# through parameters like batch size, dataset paths, auxiliary data columns, and multiprocessing.
# The module integrates with the GraphMoleculeDataset to create graph data suitable for PyTorch Geometric.
class GeometricDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        seed: int,
        max_atom_num: int = 80,
        split: Tuple[float, float] = (0.9, 0.05),
        train_path: Optional[str] = None,
        separate_valid_path: Optional[str] = None,
        separate_test_path: Optional[str] = None,
        id_column: Optional[str] = None,
        num_cores: Tuple[int, int, int] = (12, 0, 12),
        smiles_column_name: str = "SMILES",
        label_column_name: Union[str, List[str]] = "SD",
        train_auxiliary_data_column_name: Optional[str] = None,
        lbl_or_emb: str = "lbl",
        eval_auxiliary_data_column_name: Optional[str] = None,
        use_standard_scaler=False,
    ):
        super().__init__()
        # Ensure the lbl_or_emb parameter has a valid value.
        assert lbl_or_emb in [None, "lbl", "emb"]
        # Store initialization parameters for data processing.
        self.dataset = None # Placeholder for the training dataset.
        self.train_path = train_path 
        self.batch_size = batch_size # Number of samples per batch.
        self.seed = seed # Seed for reproducibility.
        self.max_atom_num = max_atom_num # Maximum number of atoms per molecule.
        self.split = split # Train-validation-test split ratios.
        self.num_cores = num_cores # Number of CPU cores for data loaders.
        self.separate_valid_path = separate_valid_path # Path to the validation dataset (if separate).
        self.separate_test_path = separate_test_path # Path to the test dataset (if separate).
        self.smiles_column_name = smiles_column_name # Column name for SMILES strings in the CSV.
        self.label_column_name = label_column_name # Column name(s) for target labels.
        self.train_auxiliary_data_column_name = train_auxiliary_data_column_name # Auxiliary data for training.
        self.eval_auxiliary_data_column_name = eval_auxiliary_data_column_name # Auxiliary data for evaluation.
        self.lbl_or_emb = lbl_or_emb # Type of auxiliary data ("lbl" or "emb").
        self.id_column = id_column # Optional column for unique IDs.

        self.use_standard_scaler = use_standard_scaler # Flag to scale labels.

        # Initialize label scaler if required.
        self.scaler = None
        if self.use_standard_scaler:
            train_df = pd.read_csv(self.train_path) # Load the training data.
            train_data = train_df[self.label_column_name].values # Extract labels.

            # Fit the StandardScaler on the training labels
            scaler = StandardScaler()
            if train_data.ndim == 1: # If labels are 1-dimensional, expand to 2D.
                scaler = scaler.fit(np.expand_dims(train_data, axis=1))
            else:
                scaler = scaler.fit(train_data)

            # Free up memory.
            del train_data
            del train_df

            self.scaler = scaler # Store the scaler.

    def get_scaler(self):
        # Return the fitted scaler for external use (e.g., inverse transformations).
        return self.scaler

    def prepare_data(self):
        # Prepare datasets for training, validation, and testing.
        # This function initializes GraphMoleculeDataset instances for each split.
        self.val = None
        self.test = None
        # Initialize the training dataset.
        if self.train_path:
            self.dataset = GraphMoleculeDataset(
                csv_path=self.train_path,
                max_atom_num=self.max_atom_num,
                smiles_column_name=self.smiles_column_name,
                label_column_name=self.label_column_name,
                auxiliary_data_column_name=self.train_auxiliary_data_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
            )
            # Cache feature dimensions for later reference.
            self.num_atom_features = self.dataset.num_atom_features
            self.num_bond_features = self.dataset.num_bond_features
        
        # Initialize the validation dataset if a separate path is provided.
        if self.separate_valid_path:
            self.val = GraphMoleculeDataset(
                csv_path=self.separate_valid_path,
                max_atom_num=self.max_atom_num,
                smiles_column_name=self.smiles_column_name,
                label_column_name=self.label_column_name,
                auxiliary_data_column_name=self.eval_auxiliary_data_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
            )

        # Initialize the test dataset if a separate path is provided.
        if self.separate_test_path:
            self.test = GraphMoleculeDataset(
                csv_path=self.separate_test_path,
                max_atom_num=self.max_atom_num,
                smiles_column_name=self.smiles_column_name,
                label_column_name=self.label_column_name,
                auxiliary_data_column_name=self.eval_auxiliary_data_column_name,
                lbl_or_emb=self.lbl_or_emb,
                scaler=self.scaler,
                id_column=self.id_column,
            )

            print("Assigned test dataset to self.test")

    def setup(self, stage: str = None):
        # Setup method for linking the prepared dataset to the training phase.
        # Called separately for each device (e.g., GPU) during distributed training.
        # Assumes prepare_data has been called
        self.train = self.dataset

    def train_dataloader(self, shuffle=True):
        # Returns a DataLoader for the training dataset.
        if self.train:
            return GeometricDataLoader(
                self.train,
                self.batch_size,
                shuffle=shuffle, # Enable shuffling for training.
                num_workers=self.num_cores[0], # Number of workers for loading data.
                pin_memory=True, # Optimize memory usage for GPU training.
            )
        return None

    def val_dataloader(self):
        # Returns a DataLoader for the validation dataset.
        return GeometricDataLoader(
            self.val,
            self.batch_size,
            shuffle=False, # No shuffling for validation.
            pin_memory=True,
            num_workers=0 if not self.num_cores else self.num_cores[1], # Use validation-specific workers.
        )

    def test_dataloader(self):
        # Returns a DataLoader for the test dataset.
        if self.test:
            return GeometricDataLoader(
                self.test,
                self.batch_size,
                shuffle=False, # No shuffling for testing.
                pin_memory=True,
                num_workers=0 if not self.num_cores else self.num_cores[2], # Use test-specific workers.
            )
        return None

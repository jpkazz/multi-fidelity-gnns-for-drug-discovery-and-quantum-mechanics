# SetTransformer model implementation for processing sets with permutation-invariant representations.
# This model leverages Self-Attention Blocks (SAB) for encoding the set elements and Pooling by Multihead Attention (PMA)
# for aggregating set elements to a fixed-size representation. The code supports flexible depth in the encoder with
# 2 or 3 SAB layers and includes optional layer normalization and dropout for regularization.
# Overall, the SetTransformer is designed for tasks where inputs are unordered sets, such as point clouds or clustering.

from .set_transformer_modules import * # Import required modules like SAB and PMA for Set Transformer


class SetTransformer(nn.Module):
    # Set Transformer model for processing sets with permutation-invariant representations.
    def __init__(
        self,
        dim_input, # Dimension of input features
        num_outputs, # Number of output elements (used in PMA)
        dim_output, # Dimension of output features
        num_inds=32, # Number of inducing points for ISAB (if used)
        dim_hidden=128, # Dimension of hidden layers
        num_heads=4, # Number of attention heads
        ln=False, # Whether to apply layer normalization
        num_sabs=2, # Number of SAB blocks in the encoder
        dropout=0.0, # Dropout rate
    ):
        super(SetTransformer, self).__init__()

        # Initialize encoder with a sequence of SAB blocks
        if num_sabs == 2:
            # Encoder with 2 Self-Attention Blocks (SAB) to process input sets
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(), # Apply dropout if specified
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            )

        # Encoder with 3 Self-Attention Blocks (SAB) to process input sets
        elif num_sabs == 3:
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout),
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            )

        # Decoder with Pooling by Multihead Attention (PMA), another SAB, and a linear projection
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln), # PMA layer to pool set to a fixed-size representation
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln, dropout=dropout), # Additional SAB block for refinement
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim_hidden, dim_output), # Final linear layer to project to output dimensions
        )

    def forward(self, X):
        # Forward pass: pass input X through encoder and then decoder
        return self.dec(self.enc(X))

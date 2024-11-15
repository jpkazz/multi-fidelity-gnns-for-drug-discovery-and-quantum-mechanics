# This code defines core components of the SetTransformer model, which is designed to handle unordered set inputs 
# with permutation-invariant representations. It includes several key modules:
# - Multihead Attention Block (MAB) for applying multi-head attention to query, key, and value inputs.
# - Self-Attention Block (SAB), which applies MAB with identical inputs for query and key, enabling self-attention.
# - Induced Self-Attention Block (ISAB) for dimensionality reduction using learned inducing points.
# - Pooling by Multihead Attention (PMA) to aggregate set elements into a fixed number of "seed" elements for downstream tasks.
# These modules are intended to be combined to build a SetTransformer, facilitating tasks like clustering and set representation learning.
# Code from https://github.com/juho-lee/set_transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    # Multihead Attention Block (MAB) that projects inputs into query (Q), key (K), and value (V) matrices
    # and performs scaled dot-product multi-head attention.
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout=0.0):
        super(MAB, self).__init__()
        self.dim_V = dim_V # Dimension of values
        self.num_heads = num_heads # Number of attention heads
        self.dropout = dropout # Dropout rate

        # Linear layers to project input dimensions into query, key, and value spaces
        self.fc_q = nn.Linear(dim_Q, dim_V) 
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        # Optional layer normalization for stabilization, applied after attention and feedforward layers
        if ln:
            self.ln0 = nn.BatchNorm1d(dim_V)
            self.ln1 = nn.BatchNorm1d(dim_V)
        # Output linear layer for final projection
        self.fc_o = nn.Linear(dim_V, dim_V)
    
    def forward(self, Q, K):
        # Project inputs Q and K to the query and key spaces, respectively
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # Split the dimensions for multi-head attention and concatenate them for parallel processing
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Compute scaled dot-product attention scores
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        # Apply dropout to attention scores if specified
        if self.dropout > 0:
            A = F.dropout(A, p=self.dropout)
        # Weighted sum of values based on attention scores, followed by concatenation of heads
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        # Optional layer normalization on the output
        O = (
            O
            if getattr(self, "ln0", None) is None
            else self.ln0(O.permute(0, 2, 1)).permute(0, 2, 1)
        )
        # Apply feedforward layer and optional layer normalization again
        O = O + F.relu(self.fc_o(O))
        O = (
            O
            if getattr(self, "ln1", None) is None
            else self.ln1(O.permute(0, 2, 1)).permute(0, 2, 1)
        )
        return O # Return the final multi-head attention output


class SAB(nn.Module):
     # Self-Attention Block (SAB) which applies multihead attention with identical Q, K, and V inputs
    def __init__(self, dim_in, dim_out, num_heads, ln=False, dropout=0.0):
        super(SAB, self).__init__()
        # MAB with shared input for Q and K
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        return self.mab(X, X) # Pass X as both Q and K for self-attention


class ISAB(nn.Module):
    # Induced Self-Attention Block (ISAB) which introduces learned latent "inducing points" for dimensionality reduction
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        # Initialize learned inducing points
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I) # Xavier initialization for inducing points
        # Two MAB layers for applying attention to inducing points and then to inputs
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        # Apply attention from inducing points to input X
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        # Apply attention from input X to the induced representations H
        return self.mab1(X, H)


class PMA(nn.Module):
    # Pooling by Multihead Attention (PMA) for aggregating set elements to a fixed number of output "seed" elements
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        # Initialize learned seed vectors
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S) # Xavier initialization for seeds
        # MAB layer for attending to the set elements based on the seed vectors
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        # Repeat seeds across the batch and attend to the set X
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

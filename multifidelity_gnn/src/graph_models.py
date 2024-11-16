import torch
import numpy as np
import pytorch_lightning as pl

from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, ReLU
import torch_geometric
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    VGAE,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GINConv,
    GINEConv,
)
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import to_dense_batch, degree
from tqdm.auto import tqdm
from typing import Optional

from .set_transformer_models import SetTransformer
from .reporting import get_metrics_pt, get_metrics_cls_pt

torch.set_num_threads(1) # Limit CPU threading for better performance.

# Utility function to compute node degrees from a list of graph datasets.
# Degrees are useful for aggregators like PNA (Principal Neighborhood Aggregation).
def get_degrees(train_dataset_as_list):
    """
    Computes degree statistics from a list of graph data objects.
    Args: train_dataset_as_list (list): A list of graph data objects from a PyTorch Geometric dataset.
    Returns: torch.Tensor: A histogram of node degrees.
    """
    deg = torch.zeros(10, dtype=torch.long) # Initialize a tensor to store degree counts.
    print("Computing degrees for PNA...")
    for data in tqdm(train_dataset_as_list): # Iterate through each graph in the dataset.
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        # Update the degree histogram with the current graph's degree distribution.
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


# ############# Variational encoders ##############


# Taken and adapted from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
# Implements a variational encoder for graph-based data using GCN layers.
class VariationalGCNEncoder(pl.LightningModule):
    """
    A variational encoder using graph convolutional layers (GCN).
    Args:
        in_channels (int): Number of input features per node.
        intermediate_dim (int): Dimension of the intermediate hidden layers.
        use_batch_norm (bool): Whether to use batch normalization between layers.
        out_channels (int): Dimension of the latent space.
        num_layers (int): Number of GCN layers in the encoder.
        name (str, optional): Optional name for the encoder.
    """
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        name: str = None,
    ):
        super(VariationalGCNEncoder, self).__init__()
        self.use_batch_norm = use_batch_norm # Flag to enable/disable batch normalization.
        self.num_layers = num_layers # Number of GCN layers in the model.

        modules = [] # Store the sequence of GCN layers and activation functions.

        # Construct GCN layers with optional batch normalization and ReLU activation.
        for i in range(self.num_layers):
            if i == 0:
                # First layer transforms input features to intermediate dimensions.
                modules.append(
                    (
                        GCNConv(in_channels, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )
            else:
                # Subsequent layers operate within the intermediate dimensions.
                modules.append(
                    (
                        GCNConv(intermediate_dim, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )

            if self.use_batch_norm: # Apply batch normalization if enabled.
                modules.append(BatchNorm(intermediate_dim))
            modules.append(nn.ReLU(inplace=True)) # Apply ReLU activation.

        # Sequentially chain all the GCN layers, batch norms, and activations.
        self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)
        # Define the output layers for latent mean (mu) and log variance (logstd).
        self.conv_mu = GCNConv(intermediate_dim, out_channels, cached=False)
        self.conv_logstd = GCNConv(intermediate_dim, out_channels, cached=False)

    def forward(self, x, edge_index):
        """
        Forward pass through the variational encoder.
        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge index tensor defining graph connectivity.
        Returns:
            torch.Tensor: Latent mean (mu) of shape [num_nodes, out_channels].
            torch.Tensor: Latent log variance (logstd) of shape [num_nodes, out_channels].
        """
        x = self.convs(x, edge_index) # Pass input through the GCN layers.
        # Compute latent mean and log variance from the final layer outputs.
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Implements a variational encoder for graph-based data using GIN layers.
class VariationalGINEncoder(pl.LightningModule):
    """
    A variational encoder using GINConv and GINEConv layers for graph representation learning.
    Args:
        in_channels (int): Number of input features per node.
        intermediate_dim (int): Dimension of intermediate hidden layers.
        use_batch_norm (bool): Whether to apply batch normalization after each layer.
        out_channels (int): Dimension of the latent space.
        num_layers (int): Number of GINConv or GINEConv layers in the encoder.
        edge_dim (int, optional): Dimension of edge features (required for GINEConv).
        name (str, optional): Optional name for the encoder.
    """
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        name: str = None,
    ):
        super(VariationalGINEncoder, self).__init__()
        self.edge_dim = edge_dim # Dimension of edge features; determines GINConv or GINEConv usage.
        self.use_batch_norm = use_batch_norm # Flag to enable/disable batch normalization.
        self.num_layers = num_layers # Number of convolutional layers.

        modules = [] # List to store the sequential layers.

        # Build the encoder layers.
        for i in range(self.num_layers):
            if i == 0:
                # First layer: transforms input node features to intermediate dimensions.
                if self.edge_dim:
                    # Use GINEConv when edge features are present.
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim, # Include edge feature dimensions.
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    # Use GINConv when edge features are absent.
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )
            else:
                # Subsequent layers: operate within intermediate dimensions.
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

            if self.use_batch_norm:
                # Add batch normalization if enabled.
                modules.append(BatchNorm(intermediate_dim))
            modules.append(nn.ReLU(inplace=True)) # Add ReLU activation.

        # Define the sequential module based on edge feature availability.
        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

        # Define the output layers for latent mean (mu) and log variance (logstd).
        nn_mu = nn.Sequential(
            Linear(intermediate_dim, out_channels), # Linear layer to map to latent space.
            ReLU(),
            Linear(out_channels, out_channels), # Final layer for the mean representation.
        )
        if self.edge_dim:
            # Use GINEConv for latent mean if edge features are present.
            self.conv_mu = GINEConv(nn_mu, edge_dim=self.edge_dim)
        else:
            # Use GINConv for latent mean if edge features are absent.
            self.conv_mu = GINConv(nn_mu)

        nn_sigma = nn.Sequential(
            Linear(intermediate_dim, out_channels), # Linear layer to map to latent space.
            ReLU(),
            Linear(out_channels, out_channels), # Final layer for log-variance representation.
        )
        if self.edge_dim:
            # Use GINEConv for latent logstd if edge features are present.
            self.conv_logstd = GINEConv(nn_sigma, edge_dim=self.edge_dim)
        else:
            # Use GINConv for latent logstd if edge features are absent.
            self.conv_logstd = GINConv(nn_sigma)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge index tensor defining graph connectivity.
            edge_attr (torch.Tensor, optional): Edge feature matrix of shape [num_edges, edge_dim].
        Returns:
            torch.Tensor: Latent mean (mu) of shape [num_nodes, out_channels].
            torch.Tensor: Latent log variance (sigma) of shape [num_nodes, out_channels].
        """
        if self.edge_dim:
            # Pass input through the encoder with edge features.
            x = self.convs(x, edge_index, edge_attr=edge_attr)
        else:
            # Pass input through the encoder without edge features.
            x = self.convs(x, edge_index)

        # Compute latent mean and log variance.
        if self.edge_dim:
            mu = self.conv_mu(x, edge_index, edge_attr=edge_attr)
            sigma = self.conv_logstd(x, edge_index, edge_attr=edge_attr)
        else:
            mu = self.conv_mu(x, edge_index)
            sigma = self.conv_logstd(x, edge_index)
        return mu, sigma # Return the latent mean and log variance.


class VariationalPNAEncoder(pl.LightningModule):
    """
    A PyTorch Lightning module for a variational encoder using Principal Neighbourhood Aggregation (PNA) convolutions.
    Attributes:
        in_channels: Input feature dimensionality.
        intermediate_dim: Dimensionality of intermediate representations.
        use_batch_norm: Whether to apply Batch Normalization.
        out_channels: Dimensionality of the output representation.
        num_layers: Number of PNA convolution layers.
        train_dataset: Dataset used to compute node degree distribution.
        edge_dim: Dimensionality of edge features (optional).
        name: Name of the module (optional).
    """
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        train_dataset,
        edge_dim: int = None,
        name: str = None,
    ):
        super(VariationalPNAEncoder, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers
        # PNA aggregators and scalers setup
        aggregators = ["mean", "min", "max", "std"] # Aggregate information from neighbors in different ways.
        scalers = ["identity", "amplification", "attenuation"] # Scale node features during aggregation.
        deg = get_degrees(train_dataset) # Compute degree distribution from the training dataset.

        pna_num_towers = 5 # Number of towers for PNA convolution.

        # Common arguments for all PNA layers
        pna_common_args = dict(
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=None, # Updated later if edge_dim is provided.
            towers=pna_num_towers,
            pre_layers=1, # Layers before aggregation.
            post_layers=1, # Layers after aggregation.
            divide_input=False, # Do not split input channels across towers.
        )

        if self.edge_dim:
            # Add edge feature dimensionality if provided.
            pna_common_args = pna_common_args | dict(edge_dim=edge_dim)
        # Initialize the sequence of PNA layers
        modules = []

        for i in range(self.num_layers):
            # First layer processes input features, subsequent layers use intermediate features.
            if i == 0:
                # Handle edge attributes if edge_dim is provided.
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x", # Specify input-output mapping.
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x", # Mapping without edge attributes.
                        )
                    )
            else:
                # Intermediate layers use the same process as above.
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

            # Optionally add Batch Normalization and ReLU activation.
            if self.use_batch_norm:
                modules.append(BatchNorm(intermediate_dim)) # Normalize features per batch.
            modules.append(nn.ReLU(inplace=True)) # Apply ReLU activation.

        # Define the sequential model with the appropriate inputs.
        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

        # Separate layers for the mean (mu) and log standard deviation (sigma) in the latent space.
        self.conv_mu = PNAConv(
            in_channels=intermediate_dim, out_channels=out_channels, **pna_common_args
        )
        self.conv_logstd = PNAConv(
            in_channels=intermediate_dim, out_channels=out_channels, **pna_common_args
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through the variational encoder.
        Args:
            x: Node features.
            edge_index: Graph connectivity (edge indices).
            edge_attr: Edge features (optional).
        Returns:
            mu: Mean of the latent distribution.
            sigma: Log standard deviation of the latent distribution.
        """
        # Pass through the convolutional layers.
        if self.edge_dim:
            x = self.convs(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.convs(x, edge_index)
        # Compute mean (mu) and log standard deviation (sigma).
        if self.edge_dim:
            mu = self.conv_mu(x, edge_index, edge_attr=edge_attr)
            sigma = self.conv_logstd(x, edge_index, edge_attr=edge_attr)
        else:
            mu = self.conv_mu(x, edge_index)
            sigma = self.conv_logstd(x, edge_index)
        return mu, sigma


# ############# Variational encoders ##############


# ############# Non-variational GNN ##############


class GCN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        name: str = None,
    ):
        super(GCN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                modules.append(
                    (
                        GCNConv(in_channels, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )

                modules.append(BatchNorm(intermediate_dim))
            elif i != self.num_layers - 1:
                modules.append(
                    (
                        GCNConv(intermediate_dim, intermediate_dim, cached=False),
                        "x, edge_index -> x",
                    )
                )

                modules.append(BatchNorm(intermediate_dim))
            elif i == self.num_layers - 1:
                modules.append(
                    (
                        GCNConv(intermediate_dim, out_channels, cached=False),
                        "x, edge_index -> x",
                    )
                )

                modules.append(BatchNorm(out_channels))
            modules.append(nn.ReLU(inplace=True))

        self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index):
        return self.convs(x, edge_index)


class GIN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        edge_dim: int = None,
        name: str = None,
    ):
        super(GIN, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(in_channels, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i != self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, intermediate_dim),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i == self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            GINEConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, out_channels),
                                ),
                                edge_dim=self.edge_dim,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            GINConv(
                                nn.Sequential(
                                    Linear(intermediate_dim, intermediate_dim),
                                    ReLU(),
                                    Linear(intermediate_dim, out_channels),
                                )
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(out_channels))
            modules.append(nn.ReLU(inplace=True))

        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim:
            return self.convs(x, edge_index, edge_attr=edge_attr)
        return self.convs(x, edge_index)


class PNA(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        intermediate_dim: int,
        use_batch_norm: bool,
        out_channels: int,
        num_layers: int,
        train_dataset,
        edge_dim: int = None,
        name: str = None,
    ):
        super(PNA, self).__init__()
        self.edge_dim = edge_dim
        self.use_batch_norm = use_batch_norm
        self.num_layers = num_layers

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        deg = get_degrees(train_dataset)

        pna_num_towers = 5

        pna_common_args = dict(
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=None,
            towers=pna_num_towers,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

        if self.edge_dim:
            pna_common_args = pna_common_args | dict(edge_dim=edge_dim)

        modules = []

        for i in range(self.num_layers):
            if i == 0:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=in_channels,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i != self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=intermediate_dim,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(intermediate_dim))
            elif i == self.num_layers - 1:
                if self.edge_dim:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=out_channels,
                                **pna_common_args,
                            ),
                            "x, edge_index, edge_attr -> x",
                        )
                    )
                else:
                    modules.append(
                        (
                            PNAConv(
                                in_channels=intermediate_dim,
                                out_channels=out_channels,
                                **pna_common_args,
                            ),
                            "x, edge_index -> x",
                        )
                    )

                modules.append(BatchNorm(out_channels))
            modules.append(nn.ReLU(inplace=True))

        if self.edge_dim:
            self.convs = torch_geometric.nn.Sequential(
                "x, edge_index, edge_attr", modules
            )
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        if self.edge_dim:
            return self.convs(x, edge_index, edge_attr=edge_attr)
        return self.convs(x, edge_index)


# ############# Non-variational GNN ##############


class Estimator(pl.LightningModule):
    """
    A PyTorch Lightning module for estimating properties of graphs using Graph Neural Networks (GNNs).
    Attributes:
        task_type: Type of task ("classification" or "regression").
        num_features: Number of input features per node.
        gnn_intermediate_dim: Number of hidden dimensions in GNN layers.
        node_latent_dim: Size of the latent node representations.
        graph_latent_dim: Size of the latent graph representations (optional).
        train_dataset: Dataset used for training, required for PNA convolutions.
        batch_size: Training batch size.
        lr: Learning rate.
        linear_output_size: Size of the final output layer.
        auxiliary_dim: Dimension of auxiliary data to be concatenated with GNN embeddings (optional).
        output_intermediate_dim: Intermediate dimension for output layers.
        scaler: Scaler for normalizing input data (optional).
        readout: Type of graph readout operation ("linear", "global_mean_pool", etc.).
        max_num_atoms_in_mol: Maximum number of atoms in a molecule (for "linear" readout).
        monitor_loss: Metric to monitor during training (e.g., "val_total_loss").
        num_layers: Number of GNN layers.
        use_batch_norm: Whether to apply batch normalization.
        set_transformer_*: Parameters for the SetTransformer readout (optional).
        edge_dim: Dimensionality of edge (bond) features (optional).
        use_vgae: Whether to use a Variational Graph Auto-Encoder (VGAE).
        linear_interim_dim: Hidden size of linear layers in "linear" readout.
        linear_dropout_p: Dropout probability in linear layers.
        conv_type: Type of GNN convolution layer ("GCN", "GIN", or "PNA").
        only_train: If True, only train on graph-level features.
    """
    def __init__(
        self,
        task_type: str,
        num_features: int,
        gnn_intermediate_dim: int,
        node_latent_dim: int,
        graph_latent_dim: Optional[int] = None,
        train_dataset=None,
        batch_size: int = 32,
        lr: float = 0.001,
        linear_output_size: int = 1,
        auxiliary_dim: int = 0,
        output_intermediate_dim: int = 768,
        scaler=None,
        readout: str = "linear",
        max_num_atoms_in_mol: int = 55,
        monitor_loss: str = "val_total_loss",
        num_layers: Optional[int] = None,
        use_batch_norm: bool = False,
        name: Optional[str] = None,
        set_transformer_hidden_dim: Optional[int] = None,
        set_transformer_num_heads: Optional[int] = None,
        set_transformer_num_sabs: Optional[int] = None,
        set_transformer_dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        use_vgae: bool = True,
        linear_interim_dim: int = 64,
        linear_dropout_p: float = 0.2,
        conv_type: str = "GCN",
        only_train: bool = False,
    ): # Arguments listed above
        super().__init__()
        # Ensure valid task type, GNN type, and readout type
        assert task_type in ["classification", "regression"]
        assert conv_type in ["GCN", "GIN", "PNA"]
        assert readout in [
            "linear",
            "global_mean_pool",
            "global_add_pool",
            "global_max_pool",
            "set_transformer",
        ]
        
        # Logging task and configuration
        print(
            "%s task with %d %s layers and %s readout."
            % (task_type.capitalize(), num_layers, conv_type, readout)
        )

        if use_batch_norm:
            print("Using batch normalisation for all layers.")
        else:
            print("NOT using batch normalisation.")

        # Initialize attributes
        self.use_vgae = use_vgae
        self.edge_dim = edge_dim
        self.only_train = only_train
        self.graph_latent_dim = graph_latent_dim if self.only_train else node_latent_dim
        self.task_type = task_type
        # Set up global pooling functions based on readout type
        self.global_pool_fn = (
            global_mean_pool
            if readout == "global_mean_pool"
            else (
                global_add_pool
                if readout == "global_add_pool"
                else (global_max_pool if readout == "global_max_pool" else None)
            )
        )

        if self.use_vgae:
            print("Using the VGAE framework.")
        else:
            print("Using a non-variational GNN model.")

        # Print detailed configuration
        if self.global_pool_fn:
            print("Using %s, graph_latent_dim not used." % (readout))
            print("Using %d latent node features." % node_latent_dim)
        else:
            print(
                "Using %d latent node features and %d latent graph features."
                % (node_latent_dim, self.graph_latent_dim)
            )

        self.auxiliary_dim = auxiliary_dim if auxiliary_dim else 0
        if self.auxiliary_dim > 0:
            print(
                "Using auxiliary data with dimension %d, total with GNN/VGAE embeddings: %d."
                % (self.auxiliary_dim, self.graph_latent_dim + self.auxiliary_dim)
            )

        if self.edge_dim:
            print("Using edge (bond) features of dimension %d." % (self.edge_dim))
        else:
            print("NOT using edge (bond) features.")

        if linear_output_size > 1:
            print("Training and evaluation in a MULTI-task scenario with %d tasks." % (linear_output_size,))
        else:
            print("Training and evaluation in a SINGLE-task scenario.")

        print()

        self.readout = readout
        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.conv_type = conv_type
        self.node_latent_dim = node_latent_dim
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.output_intermediate_dim = output_intermediate_dim
        self.linear_interim_dim = linear_interim_dim
        self.linear_dropout_p = linear_dropout_p
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.max_num_atoms_in_mol = max_num_atoms_in_mol
        self.scaler = scaler
        self.linear_output_size = linear_output_size
        self.monitor_loss = monitor_loss
        self.name = name

        self.set_transformer_hidden_dim = set_transformer_hidden_dim
        self.set_transformer_num_heads = set_transformer_num_heads
        self.set_transformer_num_sabs = set_transformer_num_sabs
        self.set_transformer_dropout = set_transformer_dropout

        # Store model outputs per epoch; used to compute the reporting metrics
        self.train_output = defaultdict(list)
        self.train_metrics = {}

        self.val_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        self.val_metrics = {}
        self.test_metrics = {}

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Holds final graphs embeddings
        self.train_graph_embeddings = defaultdict(list)
        self.test_graph_embeddings = defaultdict(list)

        # Initialize GNN parameters
        gnn_args = dict(
            in_channels=num_features,
            out_channels=node_latent_dim,
            num_layers=self.num_layers,
            intermediate_dim=self.gnn_intermediate_dim,
            use_batch_norm=self.use_batch_norm,
            name=self.name,
        )

        # Add optional arguments to GNN configuration
        if self.edge_dim:
            gnn_args = gnn_args | dict(edge_dim=self.edge_dim)
        if self.conv_type == "PNA":
            gnn_args = gnn_args | dict(train_dataset=train_dataset)

        # Define GNN model based on configuration
        if self.conv_type == "GCN":
            if self.use_vgae:
                self.gnn_model = VGAE(VariationalGCNEncoder(**gnn_args))
            else:
                self.gnn_model = GCN(**gnn_args)
        elif self.conv_type == "GIN":
            if self.use_vgae:
                self.gnn_model = VGAE(VariationalGINEncoder(**gnn_args))
            else:
                self.gnn_model = GIN(**gnn_args)
        elif self.conv_type == "PNA":
            if self.use_vgae:
                self.gnn_model = VGAE(VariationalPNAEncoder(**gnn_args))
            else:
                self.gnn_model = PNA(**gnn_args)

        # Configure readout layer
        if self.readout == "linear":
            # Linear readout maps node embeddings to graph-level embeddings
            self.linear_readout1 = nn.Linear(
                self.max_num_atoms_in_mol * node_latent_dim, self.linear_interim_dim
            )
            self.linear_readout2 = nn.Linear(
                self.linear_interim_dim, self.graph_latent_dim
            )
            if self.use_batch_norm:
                self.bn1 = nn.BatchNorm1d(self.linear_interim_dim)
                self.bn2 = nn.BatchNorm1d(self.graph_latent_dim)

            if self.linear_dropout_p > 0:
                self.linear_dropout = nn.Dropout1d(p=self.linear_dropout_p)

        elif self.readout == "set_transformer":
            # SetTransformer for graph-level readout
            self.st = SetTransformer(
                dim_input=node_latent_dim,
                num_outputs=32,
                dim_output=self.graph_latent_dim,
                num_inds=None,
                ln=True,
                dim_hidden=self.set_transformer_hidden_dim,
                num_heads=self.set_transformer_num_heads,
                num_sabs=self.set_transformer_num_sabs,
                dropout=self.set_transformer_dropout,
            )

        # Define final output layers
        if self.only_train:
            self.linear_output1 = nn.Linear(
                self.graph_latent_dim + self.auxiliary_dim, 256
            )
        else:
            self.linear_output1 = nn.Linear(
                self.node_latent_dim * 3 + self.auxiliary_dim, 256
            )

        if self.use_batch_norm:
            self.bn3 = nn.BatchNorm1d(256)

        self.linear_output2 = nn.Linear(256, self.linear_output_size)

    # Forward pass method
    def forward(
        self,
        x: torch.Tensor, # Node feature matrix
        edge_index: torch.Tensor, # Edge connectivity information
        batch: torch.Tensor, # Batch mapping to group nodes into graphs
        aux_data: Optional[torch.Tensor] = None, # Auxiliary data, if available
        edge_attr: Optional[torch.Tensor] = None, # Edge feature matrix, if available
    ):
        # Step 1: Obtain node embeddings using the selected GNN/VGAE model
        if self.use_vgae: # If using Variational Graph Autoencoder
            if self.edge_dim: # If edge features are provided
                z = self.gnn_model.encode(x, edge_index, edge_attr=edge_attr)
            else: # Without edge features
                z = self.gnn_model.encode(x, edge_index)
        else: # If not using VGAE
            if self.edge_dim:
                z = self.gnn_model.forward(x, edge_index, edge_attr=edge_attr)
            else:
                z = self.gnn_model.forward(x, edge_index)

        # Step 2: Generate graph embeddings via the readout mechanism
        # Due to batching in PyTorch Geometric, the node embeddings must be regrouped into their original graphs
        # Details: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        graph_embeddings_to_return = None # Placeholder for final graph embeddings

        # Simple global pooling of node features
        if self.only_train and self.global_pool_fn:
            # Apply simple global pooling (sum, mean, or max)
            graph_embeddings = self.global_pool_fn(z, batch)
            graph_embeddings_to_return = graph_embeddings

        if self.only_train and not self.global_pool_fn and self.readout == "linear":
            # Create dense graph representations for linear readout
            graph_embeddings, _ = to_dense_batch(
                z, batch, fill_value=0, max_num_nodes=self.max_num_atoms_in_mol
            )

            # Flatten the node features to form graph embeddings
            graph_embeddings = graph_embeddings.reshape(
                graph_embeddings.shape[0],
                self.max_num_atoms_in_mol * self.node_latent_dim,
            )

            # Apply the dense layers to get a graph-level representation
            if self.use_batch_norm:
                graph_embeddings = self.bn1(
                    self.linear_readout1(graph_embeddings)
                ).relu()
                graph_embeddings_without_relu = self.bn2(
                    self.linear_readout2(graph_embeddings)
                )
            else:
                graph_embeddings = self.linear_readout1(graph_embeddings).relu()
                graph_embeddings_without_relu = self.linear_readout2(graph_embeddings)

            graph_embeddings_to_return = graph_embeddings_without_relu
            graph_embeddings = graph_embeddings_without_relu.relu()

            if self.linear_dropout_p > 0: # Apply dropout if enabled
                graph_embeddings = self.linear_dropout(graph_embeddings)

        elif (
            self.only_train
            and not self.global_pool_fn
            and self.readout == "set_transformer"
        ):
            # Use a Set Transformer for graph embeddings
            graph_embeddings, _ = to_dense_batch(
                z, batch, fill_value=0, max_num_nodes=self.max_num_atoms_in_mol
            )
            graph_embeddings = self.st(graph_embeddings)
            graph_embeddings = graph_embeddings.mean(dim=1)
            graph_embeddings_to_return = graph_embeddings

        if not self.only_train:
            # Combine different pooling strategies (sum, mean, max) into one representation
            graph_embeddings_sum = global_add_pool(z, batch)
            graph_embeddings_mean = global_mean_pool(z, batch)
            graph_embeddings_max = global_max_pool(z, batch)
            graph_embeddings = torch.cat(
                (graph_embeddings_sum, graph_embeddings_mean, graph_embeddings_max),
                dim=-1,
            )
            graph_embeddings_to_return = graph_embeddings

        # Step 2.1: Concatenate auxiliary data (labels or embeddings) as additional columns, when available
        if self.auxiliary_dim > 0:
            assert len(aux_data.shape) == 1
            if self.auxiliary_dim == 1:
                # Here we assume the auxiliary data are just additional labels
                # (a column with single values in the DataFrame), with resulting shape (batch_size, 1)
                aux_data = aux_data.unsqueeze(dim=1)
            elif self.auxiliary_dim > 1:
                # Here we assume the individual auxiliary data points are numpy arrays,
                # so a batch of aux data would have shape (batch_size, length_of_np_arr)
                aux_data = aux_data.reshape(
                    (graph_embeddings.shape[0], self.auxiliary_dim)
                )

            # Actual concatenation
            graph_embeddings = torch.cat((graph_embeddings, aux_data), dim=1)

        # Step 3: Apply the final classifier to generate predictions
        if self.use_batch_norm:
            predictions = self.bn3(self.linear_output1(graph_embeddings)).relu()
        else:
            predictions = self.linear_output1(graph_embeddings).relu()

        predictions = self.linear_output2(predictions)

        return z, graph_embeddings_to_return, predictions

    # Configure optimizers and learning rate scheduler
    # Reduce learning rate when a metric has stopped improving
    # The ReduceLROnPlateau scheduler requires a monitor
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": opt,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.75, patience=15 # Reduce learning rate when monitored loss plateaus
            ),
            "monitor": self.monitor_loss, # Metric to monitor for the scheduler
        }
    # Compute the batch loss
    def _batch_loss(
        self,
        x: torch.Tensor, # Node feature matrix
        edge_index: torch.Tensor, # Edge connectivity
        y: Optional[torch.Tensor] = None, # Target labels
        batch_mapping: Optional[torch.Tensor] = None, # Mapping of nodes to graphs
        aux_data: Optional[torch.Tensor] = None, # Auxiliary data, if any
        edge_attr: Optional[torch.Tensor] = None, # Edge attributes, if any
    ):
        num_nodes = x.shape[0] # Total number of nodes in the batch/graph

        # Forward pass to get embeddings and predictions
        if not self.edge_dim:
            z, graph_embeddings, predictions = self.forward(
                x, edge_index, batch_mapping, aux_data
            )
        else:
            z, graph_embeddings, predictions = self.forward(
                x, edge_index, batch_mapping, aux_data, edge_attr
            )

        # Compute VGAE loss if using variational autoencoder. VGAE loss from https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
        if self.use_vgae:
            vgae_loss = self.gnn_model.recon_loss(z, edge_index)
            vgae_loss = vgae_loss + (1 / num_nodes) * self.gnn_model.kl_loss()

        predictions = predictions.reshape(-1, self.linear_output_size)
        y = y.reshape(-1, self.linear_output_size)

        # Task-specific loss
        if self.task_type == "classification":
            task_loss = F.binary_cross_entropy_with_logits(
                predictions, y.float()
            )
    
        else:
            task_loss = F.mse_loss(
                predictions,
                y.float()
            )

        # Combine losses
        if self.use_vgae:
            total_loss = vgae_loss + task_loss
            return total_loss, vgae_loss, task_loss, z, graph_embeddings, predictions
        else:
            total_loss = task_loss
            return total_loss, 0.0, 0.0, z, graph_embeddings, predictions

    # Perform a training/validation/test step
    def _step(self, batch: torch.Tensor, step_type: str):
        # Extract batch components
        # assert step_type in ['train', 'valid', 'test']
        x, edge_index, edge_attr, y, batch_mapping = (
            batch.x, # Node features
            batch.edge_index, # Edge connectivity
            batch.edge_attr, # Edge features
            batch.y, # Target labels
            batch.batch, # Batch mapping
        )
        aux_data = batch.aux_data # Auxiliary data, if any
        # Compute loss and predictions
        (
            total_loss,
            vgae_loss,
            task_loss,
            _,
            graph_embeddings,
            predictions,
        ) = self._batch_loss(x, edge_index, y, batch_mapping, aux_data, edge_attr)

        output = (predictions, y) # Store predictions and ground truth

        # Log results based on the step type
        if step_type == "train":
            self.train_output[self.current_epoch].append(output)
        elif not self.only_train and step_type == "valid":
            self.val_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)
            self.test_graph_embeddings[self.num_called_test].append(graph_embeddings)

        return total_loss, vgae_loss, task_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # Perform one training step for the given batch.
        # Calls _step to compute losses for the batch.
        train_total_loss, vgae_loss, task_loss = self._step(batch, "train")
        # Log the losses. If VGAE is used, log all three losses; otherwise, log only the total loss.
        if self.use_vgae:
            self.log("train_total_loss", train_total_loss, batch_size=self.batch_size)
            self.log("train_vgae_loss", vgae_loss, batch_size=self.batch_size)
            self.log("train_task_loss", task_loss, batch_size=self.batch_size)
        else:
            self.log("train_total_loss", train_total_loss, batch_size=self.batch_size)
        # Return the total training loss for optimization.
        return train_total_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # Perform one validation step for the given batch.
        # Calls _step to compute losses for the batch.
        # edge_attr not used so far
        val_total_loss, vgae_loss, task_loss = self._step(batch, "valid")
        # Log the losses. If VGAE is used, log all three losses; otherwise, log only the total loss.
        if self.use_vgae:
            self.log("val_total_loss", val_total_loss, batch_size=self.batch_size)
            self.log("val_vgae_loss", vgae_loss, batch_size=self.batch_size)
            self.log("val_task_loss", task_loss, batch_size=self.batch_size)
        else:
            self.log("val_total_loss", val_total_loss, batch_size=self.batch_size)
        # Return the total validation loss.
        return val_total_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        # Perform one test step for the given batch.
        # Calls _step to compute losses for the batch.
        # edge_attr not used so far
        test_total_loss, vgae_loss, task_loss = self._step(batch, "test")
        # Log the losses. If VGAE is used, log all three losses; otherwise, log only the total loss.
        if self.use_vgae:
            self.log("test_total_loss", test_total_loss, batch_size=self.batch_size)
            self.log("test_vgae_loss", vgae_loss, batch_size=self.batch_size)
            self.log("test_task_loss", task_loss, batch_size=self.batch_size)
        else:
            self.log("test_total_loss", test_total_loss, batch_size=self.batch_size)
        # Return the total test loss.
        return test_total_loss

    def _epoch_end_report(self, epoch_outputs, epoch_type):
        # Aggregates predictions and true values across the epoch.
        y_pred = (
            torch.cat([item[0] for item in epoch_outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        y_true = (
            torch.cat([item[1] for item in epoch_outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        # Optionally inverse-transform predictions and true values using a scaler.
        if self.scaler:
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
            y_true = self.scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
        # Convert predictions and true values back to tensors.
        y_pred = torch.from_numpy(y_pred)
        y_true = torch.from_numpy(y_true)
        # Compute metrics based on task type (classification or regression).
        if self.task_type == "classification":
            y_true = y_true.long()
            metrics = get_metrics_cls_pt(y_true, y_pred)
        else:
            metrics = get_metrics_pt(y_true, y_pred)
        # Log metrics for the given epoch type (e.g., Train, Validation, Test).
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{epoch_type} {metric_name}",
                metric_value,
                batch_size=self.batch_size,
            )
        # Convert predictions and true values to numpy arrays for further processing.
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        return metrics, y_pred, y_true

    def on_train_epoch_end(self):
        # Perform actions at the end of the training epoch.
        if self.only_train:
            # Generate a report for training metrics and store them.
            train_metrics, y_pred, y_true,= self._epoch_end_report(
                self.train_output[self.current_epoch], epoch_type="Train")

            self.train_metrics[self.current_epoch] = train_metrics
            # Clean up to free memory.
            del y_pred
            del y_true
            del self.train_output[self.current_epoch]

    def on_validation_epoch_end(self):
        # Perform actions at the end of the validation epoch.
        if not self.only_train:
            # Generate a report for validation metrics and store them.
            val_outputs_per_epoch = self.val_output[self.current_epoch]
            val_metrics, y_pred, y_true = self._epoch_end_report(val_outputs_per_epoch, epoch_type="Validation")

            self.val_metrics[self.current_epoch] = val_metrics
            # Clean up to free memory.
            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

    def on_test_epoch_end(self):
        # Perform actions at the end of the test epoch.
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        test_metrics, y_pred, y_true = self._epoch_end_report(test_outputs_per_epoch, epoch_type="Test")

        # Generate a report for test metrics and store them.
        self.test_metrics[self.num_called_test] = test_metrics

        # Store graph embeddings for the test results.
        self.test_graph_embeddings[self.num_called_test] = torch.cat(
            self.test_graph_embeddings[self.num_called_test]
        ).detach().cpu().numpy()

        # Save predictions and true values for the test results.
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true
        # Increment the test counter for subsequent test runs.
        self.num_called_test += 1

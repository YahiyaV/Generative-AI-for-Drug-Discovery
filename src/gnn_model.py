"""
gnn_model.py — Graph Neural Network for molecular property prediction.

Architecture:
  GCNConv layers → global_mean_pool → MLP head
  Predicts: MolWt, LogP, TPSA, QED (multi-target regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data

from src.config import GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT, GNN_TARGET_PROPS


class MoleculeGNN(nn.Module):
    """
    GCN-based Graph Neural Network for molecular property prediction.

    Uses multiple GCNConv layers with residual connections, batch norm,
    and dual pooling (mean + max) for richer graph representations.
    """

    def __init__(self, node_feature_dim: int, hidden_dim: int = GNN_HIDDEN_DIM,
                 num_layers: int = GNN_NUM_LAYERS, dropout: float = GNN_DROPOUT,
                 num_targets: int = len(GNN_TARGET_PROPS)):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GCN layers with batch norm
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # MLP prediction head (dual pooling → 2*hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data/Batch object with x, edge_index, batch

        Returns:
            predictions: (batch_size, num_targets)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = F.relu(self.input_proj(x))

        # Message passing with residual connections
        for i in range(self.num_layers):
            residual = x
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # Residual connection

        # Dual pooling for richer representations
        x_mean = global_mean_pool(x, batch)  # (B, H)
        x_max = global_max_pool(x, batch)    # (B, H)
        x = torch.cat([x_mean, x_max], dim=1)  # (B, 2H)

        return self.head(x)

    @torch.no_grad()
    def predict(self, data: Data, device: str = "cpu") -> dict:
        """
        Predict properties for a single molecule graph.

        Returns:
            dict mapping property names to predicted values
        """
        self.eval()
        data = data.to(device)
        # Add batch index for single graph
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        preds = self.forward(data).squeeze().cpu().numpy()
        return {name: float(preds[i]) for i, name in enumerate(GNN_TARGET_PROPS)}


class MoleculeGAT(nn.Module):
    """
    GAT (Graph Attention Network) variant for comparison.
    Uses attention mechanisms for more expressive message passing.
    """

    def __init__(self, node_feature_dim: int, hidden_dim: int = GNN_HIDDEN_DIM,
                 num_layers: int = GNN_NUM_LAYERS, dropout: float = GNN_DROPOUT,
                 num_targets: int = len(GNN_TARGET_PROPS), heads: int = 4):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            self.conv_layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim * heads))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * heads * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.input_proj(x))

        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.head(x)

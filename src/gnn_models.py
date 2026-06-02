"""
Specialized GAT variants for FedGATSage: Temporal, Content, and Behavioral detectors.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

logger = logging.getLogger(__name__)


class GDNLayer(nn.Module):
    """Graph Detection Network layer used by clients: embedding -> norm -> GAT -> graph classifier"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gat = GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout
        )

        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """Return node embeddings and graph logits."""
        h = self.embedding(x)
        h = self.norm(h)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.gat(h, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        # Graph-level representation for anomaly classification
        graph_emb = h.mean(dim=0, keepdim=True)
        graph_logits = self.graph_classifier(graph_emb)

        return h, graph_logits


def nt_xent_loss(
    z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5, eps: float = 1e-8
) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy loss (NT-Xent).

    z_i and z_j are two views (N x d).
    """
    device = z_i.device
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)  # 2N x d
    similarity_matrix = torch.matmul(representations, representations.T)  # 2N x 2N

    # create labels
    N = z_i.shape[0]
    labels = torch.arange(N, device=device)
    labels = torch.cat([labels, labels], dim=0)

    # mask to remove similarity with self
    diag_mask = torch.eye(2 * N, device=device).bool()
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix.masked_fill_(diag_mask, -9e15)

    # positive similarities: i with i+N and vice versa
    positives = torch.cat(
        [torch.diag(similarity_matrix, N), torch.diag(similarity_matrix, -N)], dim=0
    )

    # denominator is logsumexp over rows
    log_prob = positives - torch.logsumexp(similarity_matrix, dim=1)
    loss = -log_prob.mean()
    return loss


class GlobalGraphSAGE(nn.Module):
    """Server-side GraphSAGE for federated aggregation"""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        # Input projection for flow embeddings
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # GraphSAGE layers
        # Add a GATConv before the SAGEConv layers to allow learning from node interactions
        self.gat_before_sage = GATConv(
            hidden_dim * 2, hidden_dim, heads=1, concat=False, dropout=0.3
        )
        # After the GAT layer the representation size is `hidden_dim`
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim // 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Global classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        # Process flow embeddings
        x = self.input_projection(x)
        # Optional GAT layer before GraphSAGE
        x = self.gat_before_sage(x, edge_index)

        # GraphSAGE layers
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)

        x = self.sage2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)

        # Global classification
        embeddings = x
        predictions = self.classifier(x)

        return embeddings, predictions

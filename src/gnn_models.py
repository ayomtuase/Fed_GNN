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
    """Graph Detection Network layer used by clients: learns node embeddings and builds dynamic graphs via top-k similarity.

    The forward pass computes cosine similarity between learned node embeddings and selects top-k neighbors
    to dynamically construct the graph at each training iteration.
    """

    def __init__(
        self,
        input_dim: int,
        node_num: int = 100,
        hidden_dim: int = 256,
        num_classes: int = 2,
        topk: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        import math

        self.input_dim = input_dim
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.dropout_rate = dropout

        # Learnable node embeddings
        self.node_embedding = nn.Embedding(node_num, hidden_dim)
        nn.init.kaiming_uniform_(self.node_embedding.weight, a=math.sqrt(5))

        # Feature embedding layer
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        self.bn_embedding = nn.LayerNorm(hidden_dim)

        # GAT layer for graph convolution
        self.gat = GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout
        )

        # Graph-level classifier
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)
        self.learned_graph = None  # Store the learned graph for inspection

    def _build_dynamic_graph(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Build edge index using top-k cosine similarity of node embeddings.

        Args:
            node_embeddings: Tensor of shape (node_num, hidden_dim)

        Returns:
            edge_index: Tensor of shape (2, num_edges)
        """
        # Compute cosine similarity matrix
        weights = node_embeddings.detach().clone()
        cos_sim_mat = torch.matmul(weights, weights.T)  # (node_num, node_num)

        # Normalize by norms
        norms = weights.norm(dim=-1).view(-1, 1)  # (node_num, 1)
        normed_mat = torch.matmul(norms, norms.T)  # (node_num, node_num)
        cos_sim_mat = cos_sim_mat / (normed_mat + 1e-8)

        # Select top-k neighbors for each node
        topk_num = min(self.topk, node_embeddings.shape[0] - 1)
        topk_indices = torch.topk(cos_sim_mat, topk_num, dim=-1)[1]  # (node_num, topk)

        # Store learned graph for inspection
        self.learned_graph = topk_indices

        # Build edge index: [from_nodes, to_nodes]
        from_nodes = (
            torch.arange(0, node_embeddings.shape[0])
            .unsqueeze(1)
            .repeat(1, topk_num)
            .flatten()
        )
        to_nodes = topk_indices.flatten()
        edge_index = torch.stack([from_nodes, to_nodes], dim=0).to(
            node_embeddings.device
        )

        return edge_index

    def forward(self, x: torch.Tensor) -> tuple:
        """Return node embeddings and graph logits.

        Args:
            x: Input node features of shape (num_nodes, input_dim)

        Returns:
            h: Node embeddings of shape (num_nodes, hidden_dim)
            graph_logits: Graph-level predictions of shape (1, num_classes)
        """
        # Get node embeddings: if the input has more nodes than the embedding table,
        # repeat the learned embeddings to cover the required number instead of
        # indexing out of range.
        num_required = int(x.shape[0])
        num_available = int(self.node_embedding.num_embeddings)
        if num_required <= num_available:
            node_embeddings = self.node_embedding(
                torch.arange(num_required, device=x.device)
            )
        else:
            repeats = (num_required + num_available - 1) // num_available
            expanded = self.node_embedding.weight.repeat(repeats, 1)[:num_required]
            node_embeddings = expanded.to(x.device)

        # Embed features
        h = self.feature_embedding(x)
        h = self.bn_embedding(h)
        h = F.elu(h)
        h = self.dropout(h)

        # Build dynamic graph from top-k similarity
        edge_index = self._build_dynamic_graph(node_embeddings)

        # Apply GAT with learned edges
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

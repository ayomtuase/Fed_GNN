"""
Specialized GAT variants for FedGATSage: Temporal, Content, and Behavioral detectors.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv

logger = logging.getLogger(__name__)


class GATLayer(nn.Module):
    """GAT layer used by clients: learns node embeddings and builds dynamic graphs via top-k similarity.

    The forward pass computes cosine similarity between live node embeddings and selects top-k neighbors
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
        use_residual: bool = True,
        use_concat_skip: bool = True,
        num_heads: int = 8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.dropout_rate = dropout
        self.use_residual = use_residual
        self.use_concat_skip = use_concat_skip

        # Feature embedding layer
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        self.bn_embedding = nn.LayerNorm(hidden_dim)

        # GAT layers for graph convolution
        self.gat1 = GATConv(
            hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout
        )
        self.gat2 = GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout
        )
        self.gat3 = GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.learned_graph = None  # Store the learned graph for inspection

    def _build_dynamic_graph(self, h_emb: torch.Tensor) -> torch.Tensor:
        """Build edge index using top-k cosine similarity of node embeddings.

        Args:
            h_emb: Tensor of shape (node_num, hidden_dim)

        Returns:
            edge_index: Tensor of shape (2, num_edges)
        """
        # Compute cosine similarity matrix
        weights = h_emb.detach().clone()
        cos_sim_mat = torch.matmul(weights, weights.T)  # (node_num, node_num)

        # Normalize by norms
        norms = weights.norm(dim=-1).view(-1, 1)  # (node_num, 1)
        normed_mat = torch.matmul(norms, norms.T)  # (node_num, node_num)
        cos_sim_mat = cos_sim_mat / (normed_mat + 1e-8)

        # Select top-k neighbors for each node
        topk_num = min(self.topk, h_emb.shape[0] - 1)
        topk_indices = torch.topk(cos_sim_mat, topk_num, dim=-1)[1]  # (node_num, topk)

        # Store learned graph for inspection
        self.learned_graph = topk_indices

        # Build edge index: [from_nodes, to_nodes]
        from_nodes = (
            torch.arange(0, h_emb.shape[0], device=h_emb.device)
            .unsqueeze(1)
            .repeat(1, topk_num)
            .flatten()
        )
        to_nodes = topk_indices.flatten()
        edge_index = torch.stack([from_nodes, to_nodes], dim=0)

        return edge_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return node embeddings.

        Args:
            x: Input node features of shape (num_nodes, input_dim)

        Returns:
            h: Node embeddings of shape (num_nodes, hidden_dim) or (num_nodes, hidden_dim * 2) if use_concat_skip
        """
        # Embed features
        h_emb = self.feature_embedding(x)
        h_emb = self.bn_embedding(h_emb)
        h_emb = F.elu(h_emb)
        h_emb = self.dropout(h_emb)

        # Build dynamic graph from top-k similarity of live features
        edge_index = self._build_dynamic_graph(h_emb)

        # Apply multi-layer GAT with learned edges
        # GAT 1
        h1 = self.gat1(h_emb, edge_index)
        h1 = self.norm1(h1)
        h1 = F.elu(h1)
        h1 = self.dropout(h1)
        h1 = h1 + h_emb  # residual skip from embedding layer

        # GAT 2
        h2 = self.gat2(h1, edge_index)
        h2 = self.norm2(h2)
        h2 = F.elu(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h1  # residual skip

        # GAT 3
        h_gat = self.gat3(h2, edge_index)
        h_gat = self.norm3(h_gat)
        h_gat = F.elu(h_gat)
        h_gat = self.dropout(h_gat)

        # Residual skip connection
        if self.use_concat_skip:
            h = torch.cat([h_gat, h_emb], dim=-1)
        elif self.use_residual:
            h = h_gat + h_emb
        else:
            h = h_gat

        return h



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


class ClientAttention(nn.Module):
    """Attention mechanism over client GNN representations before concatenation."""
    def __init__(self, num_clients: int, hidden_dim: int):
        super().__init__()
        self.num_clients = num_clients
        self.attn_project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, h_client_list: list) -> tuple:
        # Compute graph-level representation for each client
        g_client_list = [h_c.mean(dim=0, keepdim=True) for h_c in h_client_list]
        g_clients = torch.cat(g_client_list, dim=0) # (num_clients, hidden_dim)
        
        # Compute attention scores
        scores = self.attn_project(g_clients).squeeze(-1) # (num_clients,)
        
        # CRITICAL CHANGE: Independent gating, not a zero-sum game
        weights = torch.sigmoid(scores) # (num_clients,)
        
        # Scale each client's node embeddings by its weight
        weighted_h_list = [weights[c] * h_client_list[c] for c in range(self.num_clients)]
        h_global = torch.cat(weighted_h_list, dim=0)
        
        return h_global, weights


class AttentionPooling(nn.Module):
    """Learns which specific sensors (nodes) matter most for the global prediction."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, h: torch.Tensor) -> tuple:
        # h shape: (num_total_nodes, hidden_dim)
        scores = self.attn_net(h)
        weights = F.softmax(scores, dim=0)  # Softmax here is good: isolates the culprit
        
        # Weighted sum creates the single graph vector
        graph_emb = torch.sum(weights * h, dim=0, keepdim=True) 
        return graph_emb, weights


class GlobalGraphSAGE(nn.Module):
    """Server-side GraphSAGE for federated aggregation"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_clients: int = 5,
        use_concat_skip: bool = True,
    ):
        super().__init__()
        self.num_clients = num_clients
        self.use_concat_skip = use_concat_skip

        # Client attention aggregation layer
        self.client_attention = ClientAttention(num_clients, input_dim)

        # Input projection for flow embeddings
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # GraphSAGE layers
        # After removing GAT, input size to SAGEConv is hidden_dim * 2
        self.sage1 = SAGEConv(hidden_dim * 2, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim // 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Global classifier
        # Include original projected representation if skip connection is enabled
        classifier_in_dim = (hidden_dim // 2) + (hidden_dim * 2) if use_concat_skip else (hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        # --- NEW: Pre-projection Normalization ---
        self.pre_proj_norm = nn.LayerNorm(classifier_in_dim)

        # --- NEW: Contrastive Projection Head ---
        # Typically maps back to a lower or equal dimensionality (e.g., hidden_dim // 2 or 128)
        contrastive_dim = 128 
        self.contrastive_projection = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, contrastive_dim)
        )

        self.pool_attention = AttentionPooling(classifier_in_dim)
        self.dropout = nn.Dropout(0.3)

    def sample_neighbors(
        self,
        edge_index: torch.Tensor,
        node_anomaly_scores: torch.Tensor,
        num_samples: int = 5,
        oversample_scale: float = 2.0,
    ) -> torch.Tensor:
        """Sample neighbors for each node with bias towards anomalous nodes (Minority Oversampling).

        Args:
            edge_index: Tensor of shape (2, E)
            node_anomaly_scores: Tensor of shape (N,) containing anomaly scores
            num_samples: Number of neighbors to sample per node
            oversample_scale: Factor scaling the bias towards anomalous nodes

        Returns:
            sampled_edge_index: Tensor of shape (2, E_sampled)
        """
        if num_samples is None or num_samples <= 0:
            return edge_index

        device = edge_index.device
        num_nodes = node_anomaly_scores.size(0)
        row, col = edge_index  # row: source, col: target

        sampled_rows = []
        sampled_cols = []

        for u in range(num_nodes):
            # Find incoming edges to target node u (where col == u)
            mask = col == u
            neighbors = row[mask]

            if neighbors.numel() == 0:
                continue

            # Get anomaly scores for these neighbor nodes
            scores = node_anomaly_scores[neighbors]

            # Compute sampling weights: base weight 1.0 + scale * score (must be non-negative)
            scores = torch.clamp(scores, min=0.0)
            weights = 1.0 + oversample_scale * scores
            probs = weights / weights.sum()

            # Sample num_samples neighbors with replacement based on probabilities
            sampled_indices = torch.multinomial(probs, num_samples, replacement=True)
            sampled_nbrs = neighbors[sampled_indices]

            sampled_rows.append(sampled_nbrs)
            sampled_cols.append(torch.full_like(sampled_nbrs, u))

        if len(sampled_rows) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        sampled_edge_index = torch.stack(
            [torch.cat(sampled_rows), torch.cat(sampled_cols)], dim=0
        )

        return sampled_edge_index

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_anomaly_scores: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        oversample_scale: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Process flow embeddings
        x_proj = self.input_projection(x)

        # Apply neighborhood sampling if training and node_anomaly_scores is provided
        if self.training and node_anomaly_scores is not None and num_samples is not None:
            sampled_edge_index = self.sample_neighbors(
                edge_index, node_anomaly_scores, num_samples, oversample_scale
            )
        else:
            sampled_edge_index = edge_index

        # GraphSAGE layers
        x_s = self.sage1(x_proj, sampled_edge_index)
        x_s = self.bn1(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        x_s = self.sage2(x_s, sampled_edge_index)
        x_s = self.bn2(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        # Skip connection on server GraphSAGE
        if self.use_concat_skip:
            embeddings = torch.cat([x_s, x_proj], dim=-1)
        else:
            embeddings = x_s

        # --- NEW: Compute Projected Contrastive Embeddings ---
        # Normalize before projection head to prevent feature saturation
        embeddings_normed = self.pre_proj_norm(embeddings)
        node_contrastive_proj = self.contrastive_projection(embeddings_normed)  # (num_nodes, contrastive_dim)
        # Normalize contrastive projection to exist on a unit hypersphere
        node_contrastive_proj = F.normalize(node_contrastive_proj, p=2, dim=-1)

        # Pool nodes into a graph embedding AND extract the culprit weights
        graph_emb, node_weights = self.pool_attention(embeddings)

        # --- NEW: Pool Contrastive Embeddings using the SAME spatial attention weights ---
        # This aligns the contrastive representation precisely with what the classifier sees
        graph_contrastive_emb = torch.sum(node_weights * node_contrastive_proj, dim=0, keepdim=True)  # (1, contrastive_dim)

        # Classify the entire system state
        predictions = self.classifier(graph_emb)

        return embeddings, predictions, node_weights, graph_contrastive_emb

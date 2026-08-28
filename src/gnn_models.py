"""
Specialized GAT variants for FedGATSage: Temporal, Content, and Behavioral detectors.
"""

import logging
from typing import Optional, Tuple, Union

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
        client_topk: Union[int, float] = 3,
        dropout: float = 0.3,
        use_residual: bool = True,
        use_concat_skip: bool = True,
        num_heads: int = 8,
        kernel_size: int = 15,
        use_sensor_embeddings: bool = True,
        sensor_embed_mode: str = "both",
        sensor_embedding_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.client_topk = client_topk
        self.dropout_rate = dropout
        self.use_residual = use_residual
        self.use_concat_skip = use_concat_skip
        self.use_sensor_embeddings = use_sensor_embeddings
        self.sensor_embed_mode = sensor_embed_mode
        self.sensor_embedding_dim = sensor_embedding_dim if sensor_embedding_dim is not None else hidden_dim

        # Trainable sensor embeddings
        if self.use_sensor_embeddings:
            self.sensor_embedding = nn.Parameter(
                torch.empty(self.node_num, self.sensor_embedding_dim)
            )
            nn.init.xavier_uniform_(self.sensor_embedding)

            if self.sensor_embedding_dim != self.hidden_dim:
                self.sensor_project = nn.Linear(self.sensor_embedding_dim, self.hidden_dim)
            else:
                self.sensor_project = nn.Identity()

        # Feature embedding using 1D convolution over temporal sliding window
        self.window_size = input_dim
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.feature_transform = nn.Linear(hidden_dim, hidden_dim)
        self.bn_embedding = nn.LayerNorm(hidden_dim)

        # Single GAT layer for graph convolution
        self.gat = GATConv(
            hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.learned_graph = None  # Store the learned graph for inspection

    def _build_dynamic_graph(self, h_emb: torch.Tensor) -> torch.Tensor:
        """Build edge index using top-k cosine similarity of node embeddings.

        Args:
            h_emb: Tensor of shape (B * node_num, hidden_dim)

        Returns:
            edge_index: Tensor of shape (2, num_edges)
        """
        B = h_emb.shape[0] // self.node_num

        if B > 1:
            # Batched similarity computation
            # CRITICAL FIX: Cast to float32 before similarity math to prevent AMP overflow
            weights = h_emb.detach().clone().float().view(B, self.node_num, -1)
            cos_sim_mat = torch.bmm(weights, weights.transpose(1, 2))  # (B, node_num, node_num)

            # Normalize by norms
            norms = weights.norm(dim=-1, keepdim=True)  # (B, node_num, 1)
            normed_mat = torch.bmm(norms, norms.transpose(1, 2))  # (B, node_num, node_num)
            cos_sim_mat = cos_sim_mat / (normed_mat + 1e-8)

            # Select top-k neighbors for each node in each batch
            if isinstance(self.client_topk, float) and 0.0 < self.client_topk <= 1.0:
                topk_num = max(1, int(self.node_num * self.client_topk))
            else:
                topk_num = int(self.client_topk)
            topk_num = min(topk_num, self.node_num - 1)
            _, topk_indices = torch.topk(cos_sim_mat, topk_num, dim=-1)  # (B, node_num, topk)

            # Store learned graph
            self.learned_graph = topk_indices

            # Build edge index: [from_nodes, to_nodes]
            batch_offsets = torch.arange(0, B, device=h_emb.device).view(B, 1, 1) * self.node_num
            to_nodes = (topk_indices + batch_offsets).flatten()

            from_nodes_local = torch.arange(0, self.node_num, device=h_emb.device).view(1, self.node_num, 1)
            from_nodes = (from_nodes_local.repeat(B, 1, topk_num) + batch_offsets).flatten()

            edge_index = torch.stack([from_nodes, to_nodes], dim=0)
        else:
            # Single graph similarity computation
            # CRITICAL FIX: Cast to float32 before similarity math to prevent AMP overflow
            weights = h_emb.detach().clone().float()
            cos_sim_mat = torch.matmul(weights, weights.T)  # (node_num, node_num)

            # Normalize by norms
            norms = weights.norm(dim=-1).view(-1, 1)  # (node_num, 1)
            normed_mat = torch.matmul(norms, norms.T)  # (node_num, node_num)
            cos_sim_mat = cos_sim_mat / (normed_mat + 1e-8)

            # Select top-k neighbors for each node
            if isinstance(self.client_topk, float) and 0.0 < self.client_topk <= 1.0:
                topk_num = max(1, int(self.node_num * self.client_topk))
            else:
                topk_num = int(self.client_topk)
            topk_num = min(topk_num, h_emb.shape[0] - 1)
            _, topk_indices = torch.topk(cos_sim_mat, topk_num, dim=-1)  # (node_num, topk)

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
            x: Input node features of shape (num_nodes, input_dim) or (B * num_nodes, input_dim)

        Returns:
            h: Node embeddings of shape (B * num_nodes, hidden_dim) or (B * num_nodes, hidden_dim * 2) if use_concat_skip
        """
        B = x.shape[0] // self.node_num

        # Apply 1D Convolution along the temporal window dimension (input_dim)
        # x is of shape (B * node_num, input_dim). We unsqueeze to add channel dimension: (B * node_num, 1, input_dim)
        x_unsqueezed = x.unsqueeze(1)
        x_conv = self.conv1d(x_unsqueezed) # shape: (B * node_num, hidden_dim, Output_Length)
        x_conv = F.elu(x_conv)

        # Max pooling over the temporal dimension to collapse the sequence length
        h_emb = torch.max(x_conv, dim=-1)[0] # shape: (B * node_num, hidden_dim)

        # Embed features
        h_emb = self.feature_transform(h_emb)
        h_emb = self.bn_embedding(h_emb)
        h_emb = F.elu(h_emb)

        # Apply sensor embeddings
        if self.use_sensor_embeddings:
            # Expand sensor embedding to match batch size: (B * node_num, sensor_embedding_dim)
            s_emb = self.sensor_embedding.repeat(B, 1)
            s_emb_proj = self.sensor_project(s_emb)  # (B * node_num, hidden_dim)

            if self.sensor_embed_mode in ["node_feature", "both"]:
                h_emb_combined = h_emb + s_emb_proj
            else:
                h_emb_combined = h_emb
        else:
            h_emb_combined = h_emb

        # Build dynamic graph from top-k similarity
        if self.use_sensor_embeddings and self.sensor_embed_mode == "graph_construction":
            edge_index = self._build_dynamic_graph(s_emb_proj)
        elif self.use_sensor_embeddings and self.sensor_embed_mode == "both":
            edge_index = self._build_dynamic_graph(h_emb_combined)
        else:
            edge_index = self._build_dynamic_graph(h_emb)

        h_emb_dropped = self.dropout(h_emb_combined)

        # Apply single-layer GAT with learned edges
        h_gat = self.gat(h_emb_dropped, edge_index)
        h_gat = self.norm(h_gat)
        h_gat = F.elu(h_gat)
        h_gat = self.dropout(h_gat)

        # Residual skip connection
        if self.use_concat_skip:
            h = torch.cat([h_gat, h_emb_combined], dim=-1)
        elif self.use_residual:
            h = h_gat + h_emb_combined
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
        
    def forward(self, h_client_list: list, client_node_nums: list = None) -> tuple:
        # Support batched inference/training
        first_h = h_client_list[0]
        if client_node_nums is None:
            B = 1
        else:
            B = first_h.shape[0] // client_node_nums[0]
            
        if B == 1:
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
        else:
            # Batched client attention
            g_client_list = [h_c.view(B, N_c, -1).mean(dim=1) for h_c, N_c in zip(h_client_list, client_node_nums)]
            g_clients = torch.stack(g_client_list, dim=1) # (B, num_clients, hidden_dim)
            
            # Compute attention scores
            scores = self.attn_project(g_clients).squeeze(-1) # (B, num_clients)
            
            # Independent gating
            weights = torch.sigmoid(scores) # (B, num_clients)
            
            # Scale each client's node embeddings by its weight
            weighted_h_list = []
            for c in range(self.num_clients):
                w_c = weights[:, c].view(B, 1, 1) # (B, 1, 1)
                h_c_reshaped = h_client_list[c].view(B, client_node_nums[c], -1) # (B, N_c, dim)
                weighted_h = (h_c_reshaped * w_c).view(-1, first_h.shape[-1])
                weighted_h_list.append(weighted_h)
                
            # Concatenate client outputs per snapshot
            h_global_batched = torch.cat([wh.view(B, client_node_nums[c], -1) for c, wh in enumerate(weighted_h_list)], dim=1)
            h_global = h_global_batched.view(-1, h_global_batched.shape[-1])
            
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
        
    def forward(self, h: torch.Tensor, num_nodes_per_graph: int = None) -> tuple:
        # h shape: (B * N_global, hidden_dim)
        scores = self.attn_net(h)
        
        if num_nodes_per_graph is None:
            weights = F.softmax(scores, dim=0)
            graph_emb = torch.sum(weights * h, dim=0, keepdim=True)
            return graph_emb, weights
            
        B = h.shape[0] // num_nodes_per_graph
        if B == 1:
            weights = F.softmax(scores, dim=0)
            graph_emb = torch.sum(weights * h, dim=0, keepdim=True)
            return graph_emb, weights
        else:
            scores_reshaped = scores.view(B, num_nodes_per_graph, 1)
            weights_reshaped = F.softmax(scores_reshaped, dim=1) # (B, num_nodes_per_graph, 1)
            weights = weights_reshaped.view(B * num_nodes_per_graph, 1)
            
            h_reshaped = h.view(B, num_nodes_per_graph, -1)
            graph_emb = torch.sum(weights_reshaped * h_reshaped, dim=1) # (B, hidden_dim)
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
            nn.Linear(hidden_dim, 1),
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

        # Pre-group neighbors using CSR-like structure for O(1) lookups
        counts = torch.bincount(col, minlength=num_nodes)
        pointers = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=pointers[1:])

        # Sort row according to col
        perm = torch.argsort(col)
        row_sorted = row[perm]

        sampled_rows = []
        sampled_cols = []

        # Clamp scores to avoid negative weights
        clamped_scores = torch.clamp(node_anomaly_scores, min=0.0)

        for u in range(num_nodes):
            start, end = pointers[u].item(), pointers[u+1].item()
            if start == end:
                continue

            neighbors = row_sorted[start:end]

            # Get anomaly scores for these neighbor nodes
            scores = clamped_scores[neighbors]

            # Compute sampling weights
            weights = 1.0 + oversample_scale * scores
            probs = weights / (weights.sum() + 1e-8)

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
        num_nodes_per_graph: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Process flow embeddings
        x_proj = self.input_projection(x)

        # Apply neighborhood sampling if training and node_anomaly_scores is provided
        if self.training and node_anomaly_scores is not None and num_samples is not None:
            if node_anomaly_scores.dim() > 1:
                node_anomaly_scores = node_anomaly_scores.flatten()
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

        # Pool nodes into a graph embedding AND extract the culprit weights
        graph_emb, node_weights = self.pool_attention(embeddings, num_nodes_per_graph)

        # --- NEW: Pool Contrastive Embeddings using the SAME spatial attention weights ---
        # This aligns the contrastive representation precisely with what the classifier sees
        if num_nodes_per_graph is not None:
            B = embeddings.shape[0] // num_nodes_per_graph
        else:
            B = 1

        if B > 1:
            node_weights_reshaped = node_weights.view(B, num_nodes_per_graph, 1)
            node_contrastive_proj_reshaped = node_contrastive_proj.view(B, num_nodes_per_graph, -1)
            graph_contrastive_emb = torch.sum(node_weights_reshaped * node_contrastive_proj_reshaped, dim=1) # (B, contrastive_dim)
        else:
            graph_contrastive_emb = torch.sum(node_weights * node_contrastive_proj, dim=0, keepdim=True)  # (1, contrastive_dim)

        # Classify the entire system state
        predictions = self.classifier(graph_emb)

        return embeddings, predictions, node_weights, graph_contrastive_emb


class GlobalGAT(nn.Module):
    """Server-side Graph Attention Network for federated aggregation"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_clients: int = 5,
        use_concat_skip: bool = True,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_clients = num_clients
        self.use_concat_skip = use_concat_skip
        self.num_heads = num_heads

        # Client attention aggregation layer
        self.client_attention = ClientAttention(num_clients, input_dim)

        # Input projection for flow embeddings
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # GAT layers
        # First layer GAT: input size is hidden_dim * 2.
        # We output hidden_dim // num_heads per head, and concatenate them to get hidden_dim.
        out_head_dim1 = max(1, hidden_dim // num_heads)
        self.gat1 = GATConv(
            in_channels=hidden_dim * 2,
            out_channels=out_head_dim1,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )
        gat1_out_dim = out_head_dim1 * num_heads

        # Second layer GAT: output averaged across heads.
        # We want the output dimension of the second GAT layer to be hidden_dim // 2.
        self.gat2 = GATConv(
            in_channels=gat1_out_dim,
            out_channels=hidden_dim // 2,
            heads=num_heads,
            concat=False,
            dropout=dropout,
        )

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(gat1_out_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Global classifier
        # Include original projected representation if skip connection is enabled
        classifier_in_dim = (hidden_dim // 2) + (hidden_dim * 2) if use_concat_skip else (hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

        # --- NEW: Pre-projection Normalization ---
        self.pre_proj_norm = nn.LayerNorm(classifier_in_dim)

        # --- NEW: Contrastive Projection Head ---
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

        # Pre-group neighbors using CSR-like structure for O(1) lookups
        counts = torch.bincount(col, minlength=num_nodes)
        pointers = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=pointers[1:])

        # Sort row according to col
        perm = torch.argsort(col)
        row_sorted = row[perm]

        sampled_rows = []
        sampled_cols = []

        # Clamp scores to avoid negative weights
        clamped_scores = torch.clamp(node_anomaly_scores, min=0.0)

        for u in range(num_nodes):
            start, end = pointers[u].item(), pointers[u+1].item()
            if start == end:
                continue

            neighbors = row_sorted[start:end]

            # Get anomaly scores for these neighbor nodes
            scores = clamped_scores[neighbors]

            # Compute sampling weights
            weights = 1.0 + oversample_scale * scores
            probs = weights / (weights.sum() + 1e-8)

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
        num_nodes_per_graph: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Process flow embeddings
        x_proj = self.input_projection(x)

        # Apply neighborhood sampling if training and node_anomaly_scores is provided
        if self.training and node_anomaly_scores is not None and num_samples is not None:
            if node_anomaly_scores.dim() > 1:
                node_anomaly_scores = node_anomaly_scores.flatten()
            sampled_edge_index = self.sample_neighbors(
                edge_index, node_anomaly_scores, num_samples, oversample_scale
            )
        else:
            sampled_edge_index = edge_index

        # GAT layers
        x_s = self.gat1(x_proj, sampled_edge_index)
        x_s = self.bn1(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        x_s = self.gat2(x_s, sampled_edge_index)
        x_s = self.bn2(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        # Skip connection on server GAT
        if self.use_concat_skip:
            embeddings = torch.cat([x_s, x_proj], dim=-1)
        else:
            embeddings = x_s

        # --- NEW: Compute Projected Contrastive Embeddings ---
        # Normalize before projection head to prevent feature saturation
        embeddings_normed = self.pre_proj_norm(embeddings)
        node_contrastive_proj = self.contrastive_projection(embeddings_normed)  # (num_nodes, contrastive_dim)

        # Pool nodes into a graph embedding AND extract the culprit weights
        graph_emb, node_weights = self.pool_attention(embeddings, num_nodes_per_graph)

        # --- NEW: Pool Contrastive Embeddings using the SAME spatial attention weights ---
        # This aligns the contrastive representation precisely with what the classifier sees
        if num_nodes_per_graph is not None:
            B = embeddings.shape[0] // num_nodes_per_graph
        else:
            B = 1

        if B > 1:
            node_weights_reshaped = node_weights.view(B, num_nodes_per_graph, 1)
            node_contrastive_proj_reshaped = node_contrastive_proj.view(B, num_nodes_per_graph, -1)
            graph_contrastive_emb = torch.sum(node_weights_reshaped * node_contrastive_proj_reshaped, dim=1) # (B, contrastive_dim)
        else:
            graph_contrastive_emb = torch.sum(node_weights * node_contrastive_proj, dim=0, keepdim=True)  # (1, contrastive_dim)

        # Classify the entire system state
        predictions = self.classifier(graph_emb)

        return embeddings, predictions, node_weights, graph_contrastive_emb


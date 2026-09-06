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
        kernel_size: int = 7,
        use_sensor_embeddings: bool = True,
        sensor_embed_mode: str = "both",
        sensor_embedding_dim: Optional[int] = None,
        disable_conv: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.client_topk = client_topk
        self.dropout_rate = dropout
        self.use_residual = use_residual
        self.use_concat_skip = use_concat_skip
        self.num_heads = num_heads
        self.use_sensor_embeddings = use_sensor_embeddings
        self.sensor_embed_mode = sensor_embed_mode
        self.sensor_embedding_dim = sensor_embedding_dim if sensor_embedding_dim is not None else hidden_dim
        self.disable_conv = disable_conv

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

        # Feature embedding: either 1D convolution over temporal sliding window or direct linear projection
        self.window_size = input_dim
        if not self.disable_conv:
            self.conv1d = nn.Conv1d(
                in_channels=1,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        else:
            self.conv1d = None
            self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.feature_transform = nn.Linear(hidden_dim, hidden_dim)
        self.bn_embedding = nn.LayerNorm(hidden_dim)

        # Single GAT layer for graph convolution
        self.gat = GATConv(
            hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

        # Linear decoder projects global node embeddings back to 1D (forecasting target sensor)
        global_node_emb_dim = (hidden_dim // 2) + (hidden_dim * 2) if use_concat_skip else (hidden_dim // 2)
        self.decoder = nn.Linear(global_node_emb_dim, 1)

        self.norm = nn.LayerNorm(hidden_dim)

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

            # Prevent self-loops by masking the diagonal
            eye = torch.eye(self.node_num, device=cos_sim_mat.device, dtype=torch.bool).unsqueeze(0)
            cos_sim_mat = cos_sim_mat.masked_fill(eye, -1e9)

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

            # Prevent self-loops by masking the diagonal
            eye = torch.eye(cos_sim_mat.shape[0], device=cos_sim_mat.device, dtype=torch.bool)
            cos_sim_mat = cos_sim_mat.masked_fill(eye, -1e9)

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

        # Feature embedding: apply 1D Conv or direct Linear projection
        if not self.disable_conv:
            # Apply 1D Convolution along temporal window dimension (input_dim)
            # x is (B * node_num, input_dim). Unsqueeze to add channel: (B * node_num, 1, input_dim)
            x_unsqueezed = x.unsqueeze(1)
            x_conv = self.conv1d(x_unsqueezed)  # (B * node_num, hidden_dim, Output_Length)
            x_conv = F.elu(x_conv)
            # Max pooling over temporal dimension
            h_emb = torch.max(x_conv, dim=-1)[0]  # (B * node_num, hidden_dim)
        else:
            # Linear projection directly from temporal window
            h_emb = self.fc_in(x)  # (B * node_num, hidden_dim)

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

        # GraphSAGE layers
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim // 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        # Global classifier
        # Include original representation if skip connection is enabled
        classifier_in_dim = (hidden_dim // 2) + input_dim if use_concat_skip else (hidden_dim // 2)
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
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_per_graph: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # GraphSAGE layers
        x_s = self.sage1(x, edge_index)
        x_s = self.bn1(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        x_s = self.sage2(x_s, edge_index)
        x_s = self.bn2(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        # Skip connection on server GraphSAGE
        if self.use_concat_skip:
            embeddings = torch.cat([x_s, x], dim=-1)
        else:
            embeddings = x_s

        # --- NEW: Compute Projected Contrastive Embeddings ---
        # Normalize before projection head to prevent feature saturation
        embeddings_normed = self.pre_proj_norm(embeddings)
        node_contrastive_proj = self.contrastive_projection(embeddings_normed)  # (num_nodes, contrastive_dim)

        # Pool contrastive and graph embeddings across nodes using mean pooling
        if num_nodes_per_graph is not None:
            B = embeddings.shape[0] // num_nodes_per_graph
        else:
            B = 1

        if B > 1:
            graph_contrastive_emb = node_contrastive_proj.view(B, num_nodes_per_graph, -1).mean(dim=1)  # (B, contrastive_dim)
            graph_emb = embeddings.view(B, num_nodes_per_graph, -1).mean(dim=1)
        else:
            graph_contrastive_emb = node_contrastive_proj.mean(dim=0, keepdim=True)  # (1, contrastive_dim)
            graph_emb = embeddings.mean(dim=0, keepdim=True)

        # Classify the entire system state
        predictions = self.classifier(graph_emb)

        return embeddings, predictions, None, graph_contrastive_emb


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

        # GAT layers
        # First layer GAT: input size is input_dim.
        # We output hidden_dim // num_heads per head, and concatenate them to get hidden_dim.
        out_head_dim1 = max(1, hidden_dim // num_heads)
        self.gat1 = GATConv(
            in_channels=input_dim,
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
        # Include original representation if skip connection is enabled
        classifier_in_dim = (hidden_dim // 2) + input_dim if use_concat_skip else (hidden_dim // 2)
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
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes_per_graph: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # GAT layers
        x_s = self.gat1(x, edge_index)
        x_s = self.bn1(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        x_s = self.gat2(x_s, edge_index)
        x_s = self.bn2(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.dropout(x_s)

        # Skip connection on server GAT
        if self.use_concat_skip:
            embeddings = torch.cat([x_s, x], dim=-1)
        else:
            embeddings = x_s

        # --- NEW: Compute Projected Contrastive Embeddings ---
        # Normalize before projection head to prevent feature saturation
        embeddings_normed = self.pre_proj_norm(embeddings)
        node_contrastive_proj = self.contrastive_projection(embeddings_normed)  # (num_nodes, contrastive_dim)

        # Pool contrastive and graph embeddings across nodes using mean pooling
        if num_nodes_per_graph is not None:
            B = embeddings.shape[0] // num_nodes_per_graph
        else:
            B = 1

        if B > 1:
            graph_contrastive_emb = node_contrastive_proj.view(B, num_nodes_per_graph, -1).mean(dim=1)  # (B, contrastive_dim)
            graph_emb = embeddings.view(B, num_nodes_per_graph, -1).mean(dim=1)
        else:
            graph_contrastive_emb = node_contrastive_proj.mean(dim=0, keepdim=True)  # (1, contrastive_dim)
            graph_emb = embeddings.mean(dim=0, keepdim=True)

        # Classify the entire system state
        predictions = self.classifier(graph_emb)

        return embeddings, predictions, None, graph_contrastive_emb


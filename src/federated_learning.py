"""Federated learning orchestration for graph-level anomaly detection.
Handles client graph construction, local training, checkpointing, and model aggregation.
"""

import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gnn_models import GDNLayer, GlobalGraphSAGE, nt_xent_loss

logger = logging.getLogger(__name__)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss implementation: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)."""
    log_pt = -F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(log_pt)
    loss = -((1 - pt) ** gamma) * log_pt
    if alpha is not None:
        alpha_t = alpha[targets]
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["attack", "Attack", "label"]:
        if candidate in df.columns:
            return candidate
    return None


def _build_label_mapper(
    df: pd.DataFrame, existing_mapper: Optional[Dict[Any, int]] = None
) -> Dict[Any, int]:
    label_col = _detect_label_column(df)
    if label_col is None:
        return existing_mapper or {0: 0}

    if existing_mapper is None:
        unique_labels = sorted(df[label_col].unique())
        return {label: idx for idx, label in enumerate(unique_labels)}

    unique_labels = sorted(set(df[label_col].unique()) - set(existing_mapper.keys()))
    next_index = max(existing_mapper.values(), default=-1) + 1
    for label in unique_labels:
        existing_mapper[label] = next_index
        next_index += 1
    return existing_mapper


def _build_client_data_from_dataframe(
    df: pd.DataFrame, label_mapper: Optional[Dict[Any, int]] = None
) -> Tuple[Dict[str, Any], Dict[Any, int]]:
    """Extract numeric node features and a graph-level label from a dataframe.
    
    Each column becomes a node, and each row's values become that node's features.
    Shape: (num_feature_nodes, num_rows_per_node)
    """
    label_col = _detect_label_column(df)
    label_mapper = _build_label_mapper(df, label_mapper)

    logger.info(f"Label Mapper: {label_mapper}")


    feature_cols = [
        col
        for col in df.columns
        if col not in ["Attack", "attack", "label"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        feature_cols = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]

    if feature_cols:
        features = df[feature_cols].fillna(0.0).astype(float)
        # Keep rows as snapshots and columns as nodes.
        # Each row is a graph observation over the sensor nodes.
        features = torch.tensor(features.values, dtype=torch.float32)
    else:
        features = torch.zeros((len(df), 1), dtype=torch.float32)

    if features.ndim == 1:
        features = features.unsqueeze(-1)

    if label_col is not None and len(df) > 0:
        labels_arr = df[label_col].map(label_mapper).astype(int).values
        graph_labels = torch.tensor(labels_arr, dtype=torch.long)
        # compute mode safely: torch.mode returns (values, indices).values may be 0-dim,
        # so use .mode()[0].item() or fallback to pandas mode.
        try:
            mode_val = graph_labels.mode()[0]
            graph_label = int(mode_val.item())
        except Exception:
            graph_label = int(pd.Series(labels_arr).mode().iloc[0])
    else:
        graph_labels = torch.zeros((features.shape[0],), dtype=torch.long)
        graph_label = 0

    graph_data = {
        "features": features,
        "graph_label": torch.tensor([graph_label], dtype=torch.long),
        "graph_labels": graph_labels,
    }

    return graph_data, label_mapper


class FedGATSageSystem:
    """Main FedGATSage federated learning system."""

    def __init__(
        self,
        data_dir: str,
        num_clients: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.client_models: Dict[int, nn.Module] = {}
        self.global_model: Optional[nn.Module] = None
        self.results: Dict[str, Any] = {"training_losses": [], "round_times": []}
        self.input_dim: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.label_mapper: Optional[Dict[Any, int]] = None

        logger.info("Initialized FedGATSageSystem")

    def initialize_models(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        num_classes: int = 2,
        node_num: int = 100,
        topk: int = 20,
        client_node_nums: Optional[List[int]] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.node_num = node_num
        self.topk = topk

        if client_node_nums is None:
            client_node_nums = [node_num] * self.num_clients
        self.client_node_nums = client_node_nums

        if len(self.client_models) > 0:
            logger.info("Models already initialized, skipping reinitialization")
            return

        for client_id in range(self.num_clients):
            n_num = client_node_nums[client_id]
            model = GDNLayer(
                input_dim=input_dim,
                node_num=n_num,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                topk=topk,
            )
            self.client_models[client_id] = model.to(self.device)

        self.global_model = GlobalGraphSAGE(
            input_dim=hidden_dim, hidden_dim=hidden_dim, num_classes=num_classes
        ).to(self.device)

        logger.info(
            f"Initialized {self.num_clients} client models with node counts {client_node_nums} "
            f"and global GraphSAGE with hidden_dim={hidden_dim}, topk={topk}"
        )

    def _checkpoint_file(self, checkpoint_dir: str, round_idx: int) -> str:
        return os.path.join(checkpoint_dir, f"checkpoint_round_{round_idx + 1}.pt")

    def _find_latest_checkpoint(self, checkpoint_dir: Optional[str]) -> Optional[str]:
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            return None

        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            return latest_path

        matches = glob.glob(os.path.join(checkpoint_dir, "checkpoint_round_*.pt"))
        if not matches:
            return None

        matches.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        )
        return matches[-1]

    def save_checkpoint(self, checkpoint_dir: str, round_idx: int):
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "round_idx": round_idx,
            "num_clients": self.num_clients,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "node_num": getattr(self, "node_num", 100),
            "client_node_nums": getattr(self, "client_node_nums", []),
            "topk": getattr(self, "topk", 20),
            "label_mapper": self.label_mapper,
            "client_models": {
                client_id: self.client_models[client_id].state_dict()
                for client_id in self.client_models
            },
            "global_model": (
                self.global_model.state_dict()
                if self.global_model is not None
                else None
            ),
            "results": self.results,
        }

        save_path = self._checkpoint_file(checkpoint_dir, round_idx)
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

        try:
            torch.save(checkpoint, save_path)
            torch.save(checkpoint, latest_path)
            logger.info(f"Checkpoint saved: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> int:
        path_to_load = checkpoint_path
        if path_to_load and not os.path.isabs(path_to_load):
            path_to_load = os.path.join(
                self.checkpoint_dir or os.getcwd(), path_to_load
            )

        if not path_to_load:
            path_to_load = self._find_latest_checkpoint(self.checkpoint_dir)

        if not path_to_load or not os.path.exists(path_to_load):
            logger.info("No checkpoint found to resume from")
            return -1

        try:
            checkpoint = torch.load(path_to_load, map_location=self.device, weights_only=False)
            self.results = checkpoint.get("results", self.results)
            self.label_mapper = checkpoint.get("label_mapper", self.label_mapper)

            if not self.client_models:
                input_dim = checkpoint.get("input_dim", 1)
                hidden_dim = checkpoint.get("hidden_dim", 256)
                num_classes = checkpoint.get("num_classes", 2)
                node_num = checkpoint.get("node_num", 100)
                client_node_nums = checkpoint.get("client_node_nums", None)
                topk = checkpoint.get("topk", 20)
                self.initialize_models(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    node_num=node_num,
                    topk=topk,
                    client_node_nums=client_node_nums,
                )

            client_states = checkpoint.get("client_models", {})
            for client_id, state_dict in client_states.items():
                if client_id in self.client_models:
                    self.client_models[client_id].load_state_dict(state_dict)
                elif str(client_id).isdigit() and int(client_id) in self.client_models:
                    self.client_models[int(client_id)].load_state_dict(state_dict)
                else:
                    logger.warning(
                        f"Skipping missing client model state for {client_id}"
                    )

            global_state = checkpoint.get("global_model")
            if self.global_model is not None and global_state is not None:
                self.global_model.load_state_dict(global_state)

            round_idx = int(checkpoint.get("round_idx", -1))
            logger.info(
                f"Loaded checkpoint from {path_to_load}, resuming at round {round_idx + 1}"
            )
            return round_idx
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path_to_load}: {e}")
            return -1

    def load_client_data(
        self, client_id: Optional[int] = None, file_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if file_path is None:
            if client_id is None:
                return None
            file_path = os.path.join(self.data_dir, f"client_{client_id + 1}.csv")

        if not os.path.exists(file_path):
            logger.error(f"Client file not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)
            graph_data, label_mapper = _build_client_data_from_dataframe(
                df, self.label_mapper
            )

            logger.info(f"Label Mapper: {label_mapper}")
            logger.info(f"Graph Data: {graph_data}")
            self.label_mapper = label_mapper
            return graph_data
        except Exception as e:
            logger.error(f"Error loading client data from {file_path}: {e}")
            return None

    def _train_client_model(
        self, model: nn.Module, data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Legacy helper for compatibility."""
        return {"loss": 0.0}

    def _extract_flow_embeddings(
        self,
        model: nn.Module,
        features: torch.Tensor,
        graph_labels: Optional[torch.Tensor],
        graph_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy helper for compatibility."""
        return torch.zeros((1, self.hidden_dim or 256)), torch.zeros((1,), dtype=torch.long)

    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> float:
        """Legacy helper for compatibility."""
        return 0.0

    def _redistribute_models(self):
        """Legacy helper for compatibility."""
        pass

    def _build_global_graph(self, h_global: torch.Tensor, topk: int) -> torch.Tensor:
        """Build edge index using top-k cosine similarity of concatenated client node embeddings."""
        weights = h_global.detach().clone()
        cos_sim_mat = torch.matmul(weights, weights.T)  # (N_global, N_global)
        
        norms = weights.norm(dim=-1).view(-1, 1)  # (N_global, 1)
        normed_mat = torch.matmul(norms, norms.T)  # (N_global, N_global)
        cos_sim_mat = cos_sim_mat / (normed_mat + 1e-8)
        
        topk_num = min(topk, h_global.shape[0] - 1)
        topk_indices = torch.topk(cos_sim_mat, topk_num, dim=-1)[1]  # (N_global, topk)
        
        from_nodes = (
            torch.arange(0, h_global.shape[0], device=h_global.device)
            .unsqueeze(1)
            .repeat(1, topk_num)
            .flatten()
        )
        to_nodes = topk_indices.flatten()
        edge_index = torch.stack([from_nodes, to_nodes], dim=0)
        
        return edge_index

    def train_federated(
        self,
        num_rounds: int = 20,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1,
        start_round: int = 0,
    ) -> Dict[str, Any]:
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(
            f"Starting joint federated VFL training from round {start_round + 1} to {num_rounds}"
        )

        # 1. Load training data for all clients
        client_data_list = []
        for client_id in range(self.num_clients):
            data = self.load_client_data(client_id=client_id)
            if data is None:
                raise ValueError(f"Could not load data for client {client_id}")
            client_data_list.append(data)

        num_snapshots = client_data_list[0]["features"].shape[0]
        logger.info(f"Loaded training data. Number of aligned snapshots: {num_snapshots}")

        # 2. Build joint parameter list and optimizer
        all_params = list(self.global_model.parameters())
        for client_model in self.client_models.values():
            all_params.extend(list(client_model.parameters()))
        optimizer = torch.optim.Adam(all_params, lr=1e-3)

        batch_size = 32
        num_steps = 10

        for round_idx in range(start_round, num_rounds):
            round_start = time.time()
            logger.info(f"Starting round {round_idx + 1}/{num_rounds}")

            self.global_model.train()
            for client_model in self.client_models.values():
                client_model.train()

            round_loss = 0.0
            # Shuffle indices at the start of each round
            indices = torch.randperm(num_snapshots)

            for step in range(num_steps):
                batch_indices = indices[step * batch_size : (step + 1) * batch_size]
                if len(batch_indices) == 0:
                    break

                optimizer.zero_grad()
                step_loss = 0.0

                for idx in batch_indices:
                    # Snapshot forward pass 1
                    h_client_list1 = []
                    for c in range(self.num_clients):
                        snapshot_features = client_data_list[c]["features"][idx].view(-1, 1).to(self.device)
                        h_c1, _ = self.client_models[c](snapshot_features)
                        h_client_list1.append(h_c1)
                    h_global1 = torch.cat(h_client_list1, dim=0)
                    edge_index1 = self._build_global_graph(h_global1, self.topk)
                    emb1, predictions1 = self.global_model(h_global1, edge_index1)

                    # Snapshot forward pass 2 for contrastive loss (GAT has dropout, so this produces a different view)
                    h_client_list2 = []
                    for c in range(self.num_clients):
                        snapshot_features = client_data_list[c]["features"][idx].view(-1, 1).to(self.device)
                        h_c2, _ = self.client_models[c](snapshot_features)
                        h_client_list2.append(h_c2)
                    h_global2 = torch.cat(h_client_list2, dim=0)
                    edge_index2 = self._build_global_graph(h_global2, self.topk)
                    emb2, predictions2 = self.global_model(h_global2, edge_index2)

                    # Get label for this snapshot
                    label = client_data_list[0]["graph_labels"][idx].unsqueeze(0).to(self.device)

                    # Compute classification loss using Focal Loss
                    alpha = torch.tensor([0.3, 0.7], device=self.device)
                    fl_loss = focal_loss(predictions1, label, alpha=alpha, gamma=2.0)

                    # Compute NT-Xent contrastive loss on pooled graph embeddings
                    # g_emb1 = emb1.mean(dim=0, keepdim=True)
                    # g_emb2 = emb2.mean(dim=0, keepdim=True)
                    # try:
                    #     contrastive_loss = nt_xent_loss(g_emb1, g_emb2, temperature=0.5)
                    # except Exception:
                    #     contrastive_loss = torch.tensor(0.0, device=self.device)

                    sample_loss = fl_loss
                    step_loss += sample_loss

                step_loss = step_loss / len(batch_indices)
                step_loss.backward()
                optimizer.step()

                round_loss += step_loss.item()

            avg_round_loss = round_loss / num_steps
            round_time = time.time() - round_start
            self.results["training_losses"].append(avg_round_loss)
            self.results["round_times"].append(round_time)

            logger.info(
                f"Round {round_idx + 1} completed in {round_time:.2f}s, loss: {avg_round_loss:.4f}"
            )

            if checkpoint_dir and (
                (round_idx - start_round + 1) % checkpoint_every == 0
                or round_idx == num_rounds - 1
            ):
                self.save_checkpoint(checkpoint_dir, round_idx)

        logger.info("Joint federated VFL training completed")
        return self.results

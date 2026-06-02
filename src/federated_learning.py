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


def _build_features_from_dataframe(
    df: pd.DataFrame, label_mapper: Optional[Dict[Any, int]] = None
) -> Tuple[Dict[str, Any], Dict[Any, int]]:
    """Extract features and labels from dataframe for a graph client.

    Graph structure is built dynamically in the GDNLayer forward pass using top-k
    cosine similarity of learned node embeddings, so we only extract features here.
    """
    label_col = _detect_label_column(df)
    label_mapper = _build_label_mapper(df, label_mapper)

    # Identify numeric feature columns
    feature_cols = [
        col
        for col in df.columns
        if col not in ["Src IP", "Dst IP", "Attack", "attack"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        feature_cols = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]

    if not feature_cols:
        feature_cols = ["flow_rate", "avg_payload_fwd", "protocol_encoded"]
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

    # Extract features per IP address
    unique_ips = pd.concat([df["Src IP"], df["Dst IP"]]).unique()
    ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}

    features = []
    for ip in unique_ips:
        ip_rows = df[(df["Src IP"] == ip) | (df["Dst IP"] == ip)]
        avg_features = ip_rows[feature_cols].mean().fillna(0.0).values
        features.append(avg_features)

    features = torch.tensor(np.array(features), dtype=torch.float32)

    # Extract graph-level label
    if label_col is not None and len(df) > 0:
        graph_label_value = df[label_col].mode().iloc[0]
        graph_label = label_mapper.get(graph_label_value, 0)
    else:
        graph_label = 0

    # Graph structure is built dynamically in GDNLayer via top-k similarity
    graph_data = {
        "features": features,
        "graph_label": torch.tensor([graph_label], dtype=torch.long),
        "ip_to_idx": ip_to_idx,
        "num_nodes": len(unique_ips),
        "df": df,
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
        input_dim: int = 64,
        hidden_dim: int = 256,
        num_classes: int = 2,
        node_num: int = 100,
        topk: int = 20,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.node_num = node_num
        self.topk = topk

        if len(self.client_models) > 0:
            logger.info("Models already initialized, skipping reinitialization")
            return

        for client_id in range(self.num_clients):
            model = GDNLayer(
                input_dim=input_dim,
                node_num=node_num,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                topk=topk,
            )
            self.client_models[client_id] = model.to(self.device)

        self.global_model = GlobalGraphSAGE(
            input_dim=hidden_dim, hidden_dim=hidden_dim, num_classes=num_classes
        ).to(self.device)

        logger.info(
            f"Initialized {self.num_clients} client models and global GraphSAGE with hidden_dim={hidden_dim}, node_num={node_num}, topk={topk}"
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
            checkpoint = torch.load(path_to_load, map_location=self.device)
            self.results = checkpoint.get("results", self.results)
            self.label_mapper = checkpoint.get("label_mapper", self.label_mapper)

            if not self.client_models:
                input_dim = checkpoint.get("input_dim", 64)
                hidden_dim = checkpoint.get("hidden_dim", 256)
                num_classes = checkpoint.get("num_classes", 2)
                node_num = checkpoint.get("node_num", 100)
                topk = checkpoint.get("topk", 20)
                self.initialize_models(
                    input_dim, hidden_dim, num_classes, node_num, topk
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

    def load_graph_from_csv(
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
            graph_data, label_mapper = _build_features_from_dataframe(
                df, self.label_mapper
            )
            self.label_mapper = label_mapper
            return graph_data
        except Exception as e:
            logger.error(f"Error loading graph from {file_path}: {e}")
            return None

    def _train_client_model(
        self, model: nn.Module, data: Dict[str, Any]
    ) -> Dict[str, float]:
        device = self.device
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        features = data.get("features")
        graph_label = data.get("graph_label")

        if features is None or graph_label is None:
            return {"loss": float("nan")}

        features = features.to(device)
        graph_label = graph_label.to(device)

        optimizer.zero_grad()

        # Graph is built dynamically inside the model's forward pass
        z1, logits1 = model(features)
        z2, logits2 = model(features)

        ce_loss = F.cross_entropy(logits1, graph_label)
        try:
            contrastive_loss = nt_xent_loss(z1, z2, temperature=0.5)
        except Exception:
            contrastive_loss = torch.tensor(0.0, device=device)

        loss = ce_loss + 0.1 * contrastive_loss
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> float:
        if not client_updates:
            return 0.0

        state_dicts = [
            upd["model_state"] for upd in client_updates if "model_state" in upd
        ]
        if not state_dicts:
            return 0.0

        averaged = {}
        keys = state_dicts[0].keys()
        for key in keys:
            vals = [sd[key].float() for sd in state_dicts]
            stacked = torch.stack(vals, dim=0)
            mean_val = stacked.mean(dim=0)
            averaged[key] = mean_val.type(state_dicts[0][key].dtype)

        self.results["global_model_state"] = averaged
        if self.global_model is not None:
            self.global_model.load_state_dict(averaged)

        return 0.0

    def _redistribute_models(self):
        averaged_state = self.results.get("global_model_state", None)
        if averaged_state is None:
            client_states = [
                self.client_models[client_id].state_dict()
                for client_id in self.client_models
            ]
            if not client_states:
                return

            averaged_state = {}
            for key in client_states[0].keys():
                stacked = torch.stack([state[key].float() for state in client_states])
                averaged_state[key] = stacked.mean(dim=0).type(
                    client_states[0][key].dtype
                )

        for client_id in self.client_models:
            self.client_models[client_id].load_state_dict(averaged_state)

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
            f"Starting federated training from round {start_round + 1} to {num_rounds}"
        )

        client_loader = DataLoader(
            list(range(self.num_clients)), batch_size=1, shuffle=False
        )

        for round_idx in range(start_round, num_rounds):
            round_start = time.time()
            logger.info(f"Starting round {round_idx + 1}/{num_rounds}")

            all_client_updates: List[Dict[str, Any]] = []

            for batch in client_loader:
                client_id = (
                    int(batch.item()) if hasattr(batch, "item") else int(batch[0])
                )
                client_data = self.load_graph_from_csv(client_id=client_id)
                if client_data is None:
                    continue

                client_model = self.client_models[client_id]
                metrics = self._train_client_model(client_model, client_data)

                all_client_updates.append(
                    {
                        "client_id": client_id,
                        "model_state": client_model.state_dict(),
                        "metrics": metrics,
                    }
                )

            global_loss = self._aggregate_updates(all_client_updates)
            self._redistribute_models()

            round_time = time.time() - round_start
            self.results["training_losses"].append(global_loss)
            self.results["round_times"].append(round_time)

            logger.info(
                f"Round {round_idx + 1} completed in {round_time:.2f}s, loss: {global_loss:.4f}"
            )

            if checkpoint_dir and (
                (round_idx - start_round + 1) % checkpoint_every == 0
                or round_idx == num_rounds - 1
            ):
                self.save_checkpoint(checkpoint_dir, round_idx)

        logger.info("Federated training completed")
        return self.results

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
from sklearn.metrics import precision_score, recall_score, f1_score

from gnn_models import GATLayer, GlobalGraphSAGE

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


def supervised_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised Contrastive Loss (SupCon) with Normal class masking.
    
    Focuses on aligning Anomaly-to-Anomaly pairs and View 1-to-View 2 pairs,
    ignoring Normal-to-Normal positive pairs to avoid over-clustering normal instances.
    """
    device = z1.device
    B = z1.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=device)
        
    # Normalize the embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate the two views: shape (2B, D)
    features = torch.cat([z1, z2], dim=0)
    
    # Full labels list: shape (2B,)
    labels_double = torch.cat([labels, labels], dim=0)
    
    # Compute similarity matrix (2B, 2B)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create mask for self-contrast (diagonal) - much faster than scatter
    logits_mask = torch.ones_like(similarity_matrix).fill_diagonal_(0)
    
    # 1. View 1-to-View 2 self-pairs: (i, i+B) and (i+B, i)
    v1_v2_mask = torch.zeros_like(similarity_matrix)
    indices = torch.arange(B, device=device)
    v1_v2_mask[indices, indices + B] = 1.0
    v1_v2_mask[indices + B, indices] = 1.0
    
    # 2. Anomaly-to-Anomaly positive pairs (same label = 1)
    anomaly_mask_2b = (labels_double == 1).float().view(-1, 1) # (2B, 1)
    anomaly_pairs_mask = torch.matmul(anomaly_mask_2b, anomaly_mask_2b.T) * logits_mask
    
    # Combine masks: positive pairs are either View1-to-View2 pairs or Anomaly-Anomaly pairs
    mask = torch.clamp(v1_v2_mask + anomaly_pairs_mask, max=1.0)
    
    # For numerical stability: subtract the max of NON-DIAGONAL/NON-SELF logits
    # Replace masked positions with a large negative value so they don't affect the max
    masked_sim_matrix = similarity_matrix.clone()
    masked_sim_matrix[logits_mask == 0] = -1e4
    logits_max, _ = torch.max(masked_sim_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean of log-likelihood over positive pairs
    # (mask.sum(1) is guaranteed to be >= 1 because of z1 and z2)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    # Loss is the negative mean
    loss = -mean_log_prob_pos.mean()
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

    if torch.cuda.is_available():
        features = features.pin_memory()
        graph_labels = graph_labels.pin_memory()

    graph_data = {
        "features": features,
        "graph_label": torch.tensor([graph_label], dtype=torch.long),
        "graph_labels": graph_labels,
    }

    return graph_data, label_mapper


def build_sliding_windows(features: torch.Tensor, w: int) -> torch.Tensor:
    """Transforms features of shape (num_snapshots, num_nodes)
    into rolling windows of shape (num_snapshots, num_nodes, w).
    For the first w - 1 steps, we pad by repeating the first snapshot.
    """
    num_snapshots, num_nodes = features.shape
    if w <= 1:
        return features.unsqueeze(-1)

    # Pad by repeating the first snapshot w - 1 times at the beginning
    padding = features[0:1].repeat(w - 1, 1)
    padded_features = torch.cat([padding, features], dim=0)

    # Unfold along dimension 0 to construct sliding windows
    windowed = padded_features.unfold(dimension=0, size=w, step=1)
    return windowed.clone().contiguous()


class FedGATSageSystem:
    """Main FedGATSage federated learning system."""

    def __init__(
        self,
        data_dir: str,
        num_clients: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
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
        use_residual: bool = True,
        use_concat_skip: bool = True,
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
            model = GATLayer(
                input_dim=input_dim,
                node_num=n_num,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                topk=topk,
                use_residual=use_residual,
                use_concat_skip=use_concat_skip,
            )
            self.client_models[client_id] = model.to(self.device)

        global_input_dim = hidden_dim * 2 if use_concat_skip else hidden_dim
        self.global_model = GlobalGraphSAGE(
            input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_clients=self.num_clients,
            use_concat_skip=use_concat_skip,
        ).to(self.device)

        logger.info(
            f"Initialized {self.num_clients} client models with node counts {client_node_nums} "
            f"and global GraphSAGE with hidden_dim={hidden_dim}, topk={topk}"
        )

    def _checkpoint_file(self, checkpoint_dir: str, round_idx: int) -> str:
        return os.path.join(checkpoint_dir, f"checkpoint_round_{round_idx + 1}.pt")

    def _safe_torch_save(self, obj: Any, path: str):
        """Save PyTorch object atomically by writing to a temporary file and renaming it."""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        tmp_path = path + ".tmp"
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise e

    def _create_checkpoint_dict(self, round_idx: int) -> Dict[str, Any]:
        """Create the complete checkpoint dictionary to save all training/model states."""
        import random
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["torch_cuda"] = torch.cuda.get_rng_state_all()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                rng_states["torch_mps"] = torch.mps.get_rng_state()
            except AttributeError:
                pass

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
            "use_concat_skip": getattr(self.global_model, "use_concat_skip", True),
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
            "current_phase": getattr(self, "current_phase", 1),
            "phase2_rounds_trained": getattr(self, "phase2_rounds_trained", 0),
            "best_loss_phase1": getattr(self, "best_loss_phase1", float("inf")),
            "no_improvement_count": getattr(self, "no_improvement_count", 0),
            "best_loss": getattr(self, "best_loss", float("inf")),
            "best_round": getattr(self, "best_round", -1),
            "rng_states": rng_states,
        }

        if getattr(self, "optimizer", None) is not None:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if getattr(self, "scheduler", None) is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        if getattr(self, "scaler", None) is not None:
            checkpoint["scaler"] = self.scaler.state_dict()

        return checkpoint

    def _get_checkpoint_candidates(self, checkpoint_dir: Optional[str]) -> List[str]:
        """Get a list of all potential checkpoints sorted by recency."""
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            return []

        candidates = []
        latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            candidates.append(latest_path)

        matches = glob.glob(os.path.join(checkpoint_dir, "checkpoint_round_*.pt"))
        if matches:
            matches.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
                reverse=True
            )
            for m in matches:
                if m not in candidates:
                    candidates.append(m)
        return candidates

    def _load_checkpoint_on_device(self, path_to_load: str, device: str) -> Dict[str, Any]:
        """Load PyTorch checkpoint file on target device, falling back to CPU first if needed."""
        try:
            return torch.load(path_to_load, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint directly to {device}: {e}. "
                "Attempting fallback load to CPU first, then remapping tensors..."
            )
            checkpoint = torch.load(path_to_load, map_location="cpu", weights_only=False)
            def _map_to_device(obj: Any, target_device: str) -> Any:
                if isinstance(obj, torch.Tensor):
                    return obj.to(target_device)
                elif isinstance(obj, dict):
                    return {k: _map_to_device(v, target_device) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_map_to_device(v, target_device) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(_map_to_device(v, target_device) for v in obj)
                return obj
            return _map_to_device(checkpoint, device)

    def _load_checkpoint_file(self, path_to_load: str, load_training_state: bool = True) -> int:
        """Internal helper to load a single checkpoint file."""
        try:
            checkpoint = self._load_checkpoint_on_device(path_to_load, self.device)
            self.label_mapper = checkpoint.get("label_mapper", self.label_mapper)

            if load_training_state:
                self.results = checkpoint.get("results", self.results)
                self.current_phase = checkpoint.get("current_phase", 1)
                self.phase2_rounds_trained = checkpoint.get("phase2_rounds_trained", 0)
                self.best_loss_phase1 = checkpoint.get("best_loss_phase1", float("inf"))
                self.no_improvement_count = checkpoint.get("no_improvement_count", 0)
                self.best_loss = checkpoint.get("best_loss", float("inf"))
                self.best_round = checkpoint.get("best_round", -1)

                # Cache optimizer, scheduler, scaler states for when training starts
                self._resume_optimizer_state = checkpoint.get("optimizer")
                self._resume_scheduler_state = checkpoint.get("scheduler")
                self._resume_scaler_state = checkpoint.get("scaler")

                # Restore RNG states if present
                rng_states = checkpoint.get("rng_states")
                if rng_states is not None:
                    try:
                        import random
                        if "python" in rng_states:
                            random.setstate(rng_states["python"])
                        if "numpy" in rng_states:
                            np.random.set_state(rng_states["numpy"])
                        if "torch_cpu" in rng_states:
                            torch.set_rng_state(rng_states["torch_cpu"])
                        if "torch_cuda" in rng_states and torch.cuda.is_available():
                            if len(rng_states["torch_cuda"]) == torch.cuda.device_count():
                                torch.cuda.set_rng_state_all(rng_states["torch_cuda"])
                            else:
                                logger.warning(
                                    f"CUDA device count mismatch (checkpoint: {len(rng_states['torch_cuda'])}, "
                                    f"current: {torch.cuda.device_count()}), skipping CUDA RNG state restore."
                                )
                        if "torch_mps" in rng_states and hasattr(torch, "mps") and torch.backends.mps.is_available():
                            try:
                                torch.mps.set_rng_state(rng_states["torch_mps"])
                            except Exception as e:
                                logger.warning(f"Failed to restore MPS RNG state: {e}")
                        logger.info("RNG states successfully restored from checkpoint")
                    except Exception as e:
                        logger.warning(f"Could not restore RNG states: {e}")

            if not self.client_models:
                input_dim = checkpoint.get("input_dim", 1)
                hidden_dim = checkpoint.get("hidden_dim", 256)
                num_classes = checkpoint.get("num_classes", 2)
                node_num = checkpoint.get("node_num", 100)
                client_node_nums = checkpoint.get("client_node_nums", None)
                topk = checkpoint.get("topk", 20)
                use_concat_skip = checkpoint.get("use_concat_skip", True)
                self.initialize_models(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    node_num=node_num,
                    topk=topk,
                    client_node_nums=client_node_nums,
                    use_concat_skip=use_concat_skip,
                )

            client_states = checkpoint.get("client_models", {})
            for client_id, state_dict in client_states.items():
                if client_id in self.client_models:
                    self.client_models[client_id].load_state_dict(state_dict, strict=False)
                elif str(client_id).isdigit() and int(client_id) in self.client_models:
                    self.client_models[int(client_id)].load_state_dict(state_dict, strict=False)
                else:
                    logger.warning(
                        f"Skipping missing client model state for {client_id}"
                    )

            global_state = checkpoint.get("global_model")
            if self.global_model is not None and global_state is not None:
                self.global_model.load_state_dict(global_state, strict=False)

            round_idx = int(checkpoint.get("round_idx", -1))
            logger.info(
                f"Successfully loaded checkpoint from {path_to_load}, round_idx: {round_idx}"
            )
            return round_idx
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path_to_load}: {e}")
            return -1

    def save_checkpoint(self, checkpoint_dir: str, round_idx: int, is_best: bool = False):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = self._create_checkpoint_dict(round_idx)

        if is_best:
            save_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            try:
                self._safe_torch_save(checkpoint, save_path)
                logger.info(f"Best checkpoint saved: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save best checkpoint: {e}")
        else:
            save_path = self._checkpoint_file(checkpoint_dir, round_idx)
            latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
            try:
                self._safe_torch_save(checkpoint, save_path)
                self._safe_torch_save(checkpoint, latest_path)
                logger.info(f"Checkpoint saved: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None, load_training_state: bool = True) -> int:
        if checkpoint_path:
            path_to_load = checkpoint_path
            if os.path.exists(path_to_load):
                pass
            elif not os.path.isabs(path_to_load):
                path_to_load = os.path.join(
                    self.checkpoint_dir or os.getcwd(), path_to_load
                )
            if not os.path.exists(path_to_load):
                logger.error(f"Explicit checkpoint file not found: {path_to_load}")
                return -1
            return self._load_checkpoint_file(path_to_load, load_training_state)

        # Auto-resume search path candidates
        candidates = self._get_checkpoint_candidates(self.checkpoint_dir)
        if not candidates:
            logger.info("No checkpoint found to resume from")
            return -1

        for path in candidates:
            logger.info(f"Attempting to load checkpoint candidate: {path}")
            round_idx = self._load_checkpoint_file(path, load_training_state)
            if round_idx >= 0:
                return round_idx
            logger.warning(f"Failed to load checkpoint {path}, trying next candidate...")

        logger.error("All checkpoint candidates failed to load.")
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

            self.label_mapper = label_mapper
            return graph_data
        except Exception as e:
            logger.error(f"Error loading client data from {file_path}: {e}")
            return None

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
        num_rounds: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1,
        start_round: int = 0,
        num_samples: int = 5,
        oversample_scale: float = 2.0,
        focal_loss_alpha: float = 0.5,
        use_ce_loss: bool = True,
        use_oversampling: bool = False,
        two_speed_lr: bool = True,
        lr_server: float = 0.0003,
        lr_client: float = 0.0005,
        enable_client_attention: bool = False,
        use_contrastive: bool = True,
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 0.07,
        normalize_vfl_gradients: bool = False,
        vfl_target_norm: float = 1.0,
        batch_size: int = 1024,
        use_amp: bool = True,
        max_samples: Optional[int] = None,
        lr_scheduler_patience: int = 2,
        lr_scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        log_step_every: int = 50,
        early_stopping_patience: int = 3,
    ) -> Dict[str, Any]:
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        rounds_str = str(num_rounds) if num_rounds is not None else "∞"
        logger.info(
            f"Starting joint federated VFL training from round {start_round + 1} to {rounds_str} "
            f"with neighbor sampling num_samples={num_samples}, oversample_scale={oversample_scale}, "
            f"focal_loss_alpha={focal_loss_alpha}, use_ce_loss={use_ce_loss}, "
            f"use_oversampling={use_oversampling}, two_speed_lr={two_speed_lr}, "
            f"enable_client_attention={enable_client_attention}, use_contrastive={use_contrastive}, "
            f"normalize_vfl_gradients={normalize_vfl_gradients}, early_stopping_patience={early_stopping_patience}"
        )

        # Initialize early stopping and phase tracking variables
        if start_round == 0 or not hasattr(self, "current_phase"):
            self.current_phase = 1
            self.phase2_rounds_trained = 0
            self.best_loss_phase1 = float("inf")
            self.no_improvement_count = 0
            self.best_loss = float("inf")
            self.best_round = -1
        else:
            if not hasattr(self, "best_loss"):
                self.best_loss = self.best_loss_phase1 if self.current_phase == 1 else float("inf")
            if not hasattr(self, "best_round"):
                self.best_round = -1

        best_loss = self.best_loss
        best_round = self.best_round
        no_improvement_count = self.no_improvement_count
        best_global_state = None
        best_client_states = {}

        # Truncate results to start_round to ensure consistency if we resume
        if isinstance(self.results, dict):
            if "training_losses" in self.results:
                self.results["training_losses"] = self.results["training_losses"][:start_round]
            if "round_times" in self.results:
                self.results["round_times"] = self.results["round_times"][:start_round]

        # Load existing best checkpoint if it exists from disk
        if checkpoint_dir:
            best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_checkpoint_path):
                try:
                    best_checkpoint = self._load_checkpoint_on_device(best_checkpoint_path, self.device)
                    best_global_state = best_checkpoint.get("global_model")
                    
                    best_client_states_raw = best_checkpoint.get("client_models", {})
                    best_client_states = {}
                    for cid, state in best_client_states_raw.items():
                        if str(cid).isdigit():
                            best_client_states[int(cid)] = state
                        else:
                            best_client_states[cid] = state
                            
                    logger.info(f"Loaded existing best model weights from disk: {best_checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to load existing best model weights from disk: {e}")

        if start_round > 0 and len(self.results.get("training_losses", [])) > 0 and self.current_phase == 1:
            # Reconstruct the historical best loss and how many rounds since it happened
            history = self.results["training_losses"][:start_round]
            if len(history) > 0:
                best_loss = min(history)
                best_round = history.index(best_loss)
                no_improvement_count = start_round - 1 - best_round
                self.best_loss = best_loss
                self.best_round = best_round
                logger.info(
                    f"Resuming with historical best loss of {best_loss:.4f} achieved at round {best_round + 1}. "
                    f"Rounds without improvement: {no_improvement_count}"
                )

        # 1. Load training data for all clients
        client_data_list = []
        for client_id in range(self.num_clients):
            data = self.load_client_data(client_id=client_id)
            if data is None:
                raise ValueError(f"Could not load data for client {client_id}")
            if max_samples is not None:
                data = {
                    "features": data["features"][:max_samples],
                    "graph_label": data["graph_label"],
                    "graph_labels": data["graph_labels"][:max_samples],
                }
            # Move client features and labels to GPU device to avoid constant CPU-GPU transfer overhead
            data["features"] = data["features"].to(self.device)
            data["graph_labels"] = data["graph_labels"].to(self.device)
            client_data_list.append(data)

        num_snapshots = client_data_list[0]["features"].shape[0]
        logger.info(f"Loaded training data. Number of aligned snapshots: {num_snapshots}")

        # Dynamic class weight calculation
        train_labels = client_data_list[0]["graph_labels"]
        num_normal = (train_labels == 0).sum().item()
        num_anomalous = (train_labels == 1).sum().item()
        if num_anomalous > 0:
            weight_ratio = num_normal / num_anomalous
        else:
            weight_ratio = 2.11
        logger.info(f"Class imbalance ratio: {weight_ratio:.4f} (Normal: {num_normal}, Anomalous: {num_anomalous})")

        # Set up loss criteria - CRITICAL FIX: Use CrossEntropyLoss and pass standard class weights
        if use_ce_loss:
            class_weights = torch.tensor([1.0, weight_ratio], device=self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info("Using CrossEntropyLoss with class weights.")
        else:
            logger.info("Using Focal Loss.")

        # Precompute mean and standard deviation under normal conditions (graph_labels == 0) for each feature node
        normal_means_list = []
        normal_stds_list = []
        for c in range(self.num_clients):
            features = client_data_list[c]["features"]  # (num_snapshots, num_nodes)
            labels = client_data_list[c]["graph_labels"]  # (num_snapshots,)
            normal_mask = labels == 0

            # Fallback to all samples if there are no normal samples
            if normal_mask.sum() == 0:
                normal_features = features
            else:
                normal_features = features[normal_mask]

            mean = normal_features.mean(dim=0)  # (num_nodes,)
            std = normal_features.std(dim=0)  # (num_nodes,)
            std = torch.clamp(std, min=1e-5)  # Avoid division by zero

            normal_means_list.append(mean.to(self.device))
            normal_stds_list.append(std.to(self.device))

        self.normal_means_global = torch.cat(normal_means_list, dim=0)  # (total_nodes,)
        self.normal_stds_global = torch.cat(normal_stds_list, dim=0)  # (total_nodes,)

        # Apply sliding window transformation to features
        w = self.input_dim or 5
        logger.info(f"Applying sliding window transformation with window size w={w}")
        for c in range(self.num_clients):
            client_data_list[c]["features"] = build_sliding_windows(client_data_list[c]["features"], w)

        # 2. Build joint parameter list and optimizer
        # Force single learning rate for Phase 1
        current_lr = lr_client
        if self.current_phase == 1:
            two_speed_lr = False
            logger.info(f"Phase 1: Forcing single learning rate for all layers (no two-speed LR): {current_lr}")
        elif self.current_phase == 2:
            current_lr = lr_client * 0.5
            two_speed_lr = False
            logger.info(f"Phase 2: Forcing single stepped-down learning rate for all layers: {current_lr}")

        if two_speed_lr:
            server_params = list(self.global_model.parameters())
            client_params = []
            for client_model in self.client_models.values():
                client_params.extend(list(client_model.parameters()))
            optimizer = torch.optim.Adam([
                {"params": server_params, "lr": lr_server},
                {"params": client_params, "lr": lr_client}
            ])
            logger.info(f"Using Two-Speed LR: Server LR={lr_server}, Client LR={lr_client}")
        else:
            all_params = list(self.global_model.parameters())
            for client_model in self.client_models.values():
                all_params.extend(list(client_model.parameters()))
            optimizer = torch.optim.Adam(all_params, lr=current_lr)
            logger.info(f"Using single speed learning rate: {current_lr}")

        # Initialize the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_lr
        )

        # Initialize the scaler for mixed precision (AMP)
        device_type = torch.device(self.device).type
        actual_use_amp = use_amp and (device_type in ["cuda", "mps"])
        scaler_device = "mps" if device_type == "mps" else "cuda"
        scaler = torch.amp.GradScaler(scaler_device, enabled=actual_use_amp)
        if actual_use_amp:
            logger.info(f"Mixed precision training enabled using device type: {device_type}")

        # Store references for checkpointing
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        def _map_to_device(obj: Any, target_device: str) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.to(target_device)
            elif isinstance(obj, dict):
                return {k: _map_to_device(v, target_device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_map_to_device(v, target_device) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_map_to_device(v, target_device) for v in obj)
            return obj

        # Restore optimizer, scheduler, and scaler states if resuming from checkpoint
        if getattr(self, "_resume_optimizer_state", None) is not None:
            try:
                self._resume_optimizer_state = _map_to_device(self._resume_optimizer_state, self.device)
                optimizer.load_state_dict(self._resume_optimizer_state)
                logger.info("Optimizer state successfully restored from checkpoint")
            except Exception as e:
                logger.warning(f"Could not restore optimizer state from checkpoint: {e}. Reinitializing.")
            self._resume_optimizer_state = None

        if getattr(self, "_resume_scheduler_state", None) is not None:
            try:
                self._resume_scheduler_state = _map_to_device(self._resume_scheduler_state, self.device)
                scheduler.load_state_dict(self._resume_scheduler_state)
                logger.info("Scheduler state successfully restored from checkpoint")
            except Exception as e:
                logger.warning(f"Could not restore scheduler state from checkpoint: {e}. Reinitializing.")
            self._resume_scheduler_state = None

        if getattr(self, "_resume_scaler_state", None) is not None:
            try:
                self._resume_scaler_state = _map_to_device(self._resume_scaler_state, self.device)
                scaler.load_state_dict(self._resume_scaler_state)
                logger.info("GradScaler state successfully restored from checkpoint")
            except Exception as e:
                logger.warning(f"Could not restore GradScaler state from checkpoint: {e}. Reinitializing.")
            self._resume_scaler_state = None

        num_steps = max(1, (num_snapshots + batch_size - 1) // batch_size)

        training_start_time = time.time()

        round_idx = start_round
        while True:
            if num_rounds is not None and round_idx >= num_rounds:
                logger.info(f"Reached maximum number of rounds: {num_rounds}. Stopping training.")
                break

            rounds_str = str(num_rounds) if num_rounds is not None else "∞"
            round_start = time.time()
            logger.info(f"Starting round {round_idx + 1}/{rounds_str}")

            self.global_model.train()
            for client_model in self.client_models.values():
                client_model.train()

            round_loss = 0.0
            # Shuffle indices at the start of each round directly on GPU (no CPU sync)
            indices = torch.randperm(num_snapshots, device=self.device)

            # Calculate warmup scaling factor lambda_t over rounds based on phase
            if self.current_phase == 1:
                lambda_t = 0.0
                should_compute_contrastive = False
            else:
                lambda_t = contrastive_weight if use_contrastive else 0.0
                should_compute_contrastive = use_contrastive

            # Initialize accumulation buffers for step logging
            step_count_in_interval = 0
            correct_preds_in_interval = 0
            preds_in_interval = []
            labels_in_interval = []
            clf_loss_in_interval = torch.tensor(0.0, device=self.device)
            supcon_loss_in_interval = 0.0
            client_norms_in_interval = torch.zeros(self.num_clients, device=self.device)
            server_emb_norm_in_interval = torch.tensor(0.0, device=self.device)

            for step in range(num_steps):
                batch_indices = indices[step * batch_size : (step + 1) * batch_size]
                if len(batch_indices) == 0:
                    break

                # Record step start time
                step_start = time.time()

                optimizer.zero_grad()
                
                # Setup structures to capture boundary gradients
                vfl_gradients1 = {c: [] for c in range(self.num_clients)}
                vfl_gradients2 = {c: [] for c in range(self.num_clients)}

                def make_grad_hook(client_idx, norm_list, normalize, target_norm, record_norm):
                    def hook(grad):
                        if grad is not None:
                            if record_norm:
                                grad_norm_val = grad.norm(2).item()
                                norm_list[client_idx].append(grad_norm_val)
                                if normalize:
                                    return grad / (grad_norm_val + 1e-8) * target_norm
                            else:
                                if normalize:
                                    # Perform normalization purely on GPU without .item() CPU-GPU synchronization
                                    return grad / (grad.norm(2) + 1e-8) * target_norm
                        return grad
                    return hook

                g_embs_1 = []
                g_embs_2 = []
                batch_labels = []
                clf_losses = []
                batch_preds = []
                attn_weights_list = []

                step_count_in_interval += 1

                # Gather batch tensors on GPU
                batch_features_clients = [
                    client_data_list[c]["features"][batch_indices] for c in range(self.num_clients)
                ]
                batch_labels_all = client_data_list[0]["graph_labels"][batch_indices]

                with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=actual_use_amp):
                    for batch_idx in range(len(batch_indices)):
                        # Get raw features for all nodes at this snapshot to compute anomaly scores
                        raw_features_list = []
                        for c in range(self.num_clients):
                            # The current timestep feature is the last index of the window
                            snapshot_features = batch_features_clients[c][batch_idx][:, -1].view(-1, 1)
                            raw_features_list.append(snapshot_features)
                        raw_features_global = torch.cat(raw_features_list, dim=0).squeeze(-1)  # (total_nodes,)

                        # Compute z-score deviation from precomputed normal baseline, squashed to [0, 1) using tanh
                        z_scores = torch.abs(raw_features_global - self.normal_means_global) / self.normal_stds_global
                        node_anomaly_scores = torch.tanh(z_scores)

                        # Get label for this snapshot
                        label = batch_labels_all[batch_idx].unsqueeze(0)  # shape (1,) on device

                        # Only apply minority oversampling on anomalous snapshots (label == 1)
                        node_anomaly_scores = node_anomaly_scores * label

                        # Snapshot forward pass 1
                        h_client_list1 = []
                        for c in range(self.num_clients):
                            snapshot_features = batch_features_clients[c][batch_idx]
                            h_c1 = self.client_models[c](snapshot_features)
                            
                            # Register hook to monitor and normalize gradients
                            if h_c1.requires_grad and (normalize_vfl_gradients or step == 0):
                                h_c1.register_hook(make_grad_hook(c, vfl_gradients1, normalize_vfl_gradients, vfl_target_norm, step == 0))
                                
                            h_client_list1.append(h_c1)

                            # Accumulate client GAT output representation norm
                            client_norms_in_interval[c] += h_c1.detach().norm(2)
                        
                        if enable_client_attention:
                            h_global1, attn_weights = self.global_model.client_attention(h_client_list1)
                            if step == 0:
                                attn_weights_list.append(attn_weights.detach().cpu())
                        else:
                            h_global1 = torch.cat(h_client_list1, dim=0)

                        edge_index1 = self._build_global_graph(h_global1, self.topk)

                        if use_oversampling:
                            emb1, predictions1, _, graph_contrastive_emb1 = self.global_model(
                                h_global1,
                                edge_index1,
                                node_anomaly_scores=node_anomaly_scores,
                                num_samples=num_samples,
                                oversample_scale=oversample_scale,
                            )
                        else:
                            emb1, predictions1, _, graph_contrastive_emb1 = self.global_model(
                                h_global1,
                                edge_index1,
                                node_anomaly_scores=None,
                                num_samples=None,
                            )

                        # Accumulate server graph representation norm
                        server_emb_norm_in_interval += emb1.detach().norm(2)

                        # Accumulate predictions
                        batch_preds.append(predictions1.argmax(dim=1))

                        # Snapshot forward pass 2 for contrastive loss (explicit feature masking + GAT dropout)
                        # Only compute if should_compute_contrastive is True to save computation
                        if should_compute_contrastive:
                            h_client_list2 = []
                            for c in range(self.num_clients):
                                snapshot_features = batch_features_clients[c][batch_idx].clone()
                                
                                # Feature Masking Augmentation: Zero out 20% of the raw features
                                mask = torch.rand_like(snapshot_features) > 0.2
                                snapshot_features = snapshot_features * mask
                                
                                h_c2 = self.client_models[c](snapshot_features)
                                
                                if h_c2.requires_grad and normalize_vfl_gradients:
                                    h_c2.register_hook(make_grad_hook(c, vfl_gradients2, normalize_vfl_gradients, vfl_target_norm, False))
                                    
                                h_client_list2.append(h_c2)

                            if enable_client_attention:
                                h_global2, _ = self.global_model.client_attention(h_client_list2)
                            else:
                                h_global2 = torch.cat(h_client_list2, dim=0)

                            edge_index2 = self._build_global_graph(h_global2, self.topk)

                            # Topological Augmentation: drop 20% of edges in server's adjacency matrix for view 2
                            edge_mask = torch.rand(edge_index2.size(1), device=edge_index2.device) > 0.2
                            edge_index2_augmented = edge_index2[:, edge_mask]

                            if use_oversampling:
                                emb2, predictions2, _, graph_contrastive_emb2 = self.global_model(
                                    h_global2,
                                    edge_index2_augmented,
                                    node_anomaly_scores=node_anomaly_scores,
                                    num_samples=num_samples,
                                    oversample_scale=oversample_scale,
                                )
                            else:
                                emb2, predictions2, _, graph_contrastive_emb2 = self.global_model(
                                    h_global2,
                                    edge_index2_augmented,
                                    node_anomaly_scores=None,
                                    num_samples=None,
                                )

                        # Compute classification loss
                        if use_ce_loss:
                            # Pass the raw 1D label directly. predictions1 is shape (batch, 2)
                            clf_loss = criterion(predictions1, label)
                        else:
                            alpha = torch.tensor([1.0 - focal_loss_alpha, focal_loss_alpha], device=self.device)
                            clf_loss = focal_loss(predictions1, label, alpha=alpha, gamma=2.0)

                        clf_losses.append(clf_loss)
                        clf_loss_in_interval += clf_loss

                        # Keep projected pooled representations for supervised contrastive loss
                        if should_compute_contrastive:
                            g_embs_1.append(graph_contrastive_emb1)
                            g_embs_2.append(graph_contrastive_emb2)
                        batch_labels.append(label)

                    # Compute step loss
                    mean_clf_loss = torch.stack(clf_losses).mean()

                    if should_compute_contrastive and len(g_embs_1) > 0:
                        z1 = torch.cat(g_embs_1, dim=0)
                        z2 = torch.cat(g_embs_2, dim=0)
                        labels_tensor = torch.cat(batch_labels, dim=0)
                        
                        # Compute contrastive loss
                        supcon_loss = supervised_contrastive_loss(
                            z1, 
                            z2, 
                            labels_tensor, 
                            temperature=contrastive_temp,
                        )
                        
                        # Stable linear combination of classification loss and contrastive loss
                        step_loss = mean_clf_loss + (lambda_t * supcon_loss)
                        supcon_loss_in_interval += supcon_loss.item()
                    else:
                        step_loss = mean_clf_loss

                scaler.scale(step_loss).backward()

                if actual_use_amp:
                    scaler.unscale_(optimizer)

                # Apply gradient clipping to stabilize training across VFL boundary
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)
                for client_model in self.client_models.values():
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)

                # Intercept and log client GAT gradient norms and attention weights at step 0
                if step == 0:
                    if enable_client_attention and len(attn_weights_list) > 0:
                        avg_attn_weights = torch.stack(attn_weights_list).mean(dim=0).numpy()
                        attn_str = ", ".join([f"Client {c+1}: {w:.4f}" for c, w in enumerate(avg_attn_weights)])
                        logger.info(f"Client attention weights at round {round_idx + 1}, step 0: {attn_str}")

                    avg_vfl_norms1 = []
                    for c in range(self.num_clients):
                        norms = vfl_gradients1[c]
                        avg_norm = np.mean(norms) if len(norms) > 0 else 0.0
                        avg_vfl_norms1.append(avg_norm)
                    vfl_norms_str = ", ".join([f"Client {c+1}: {norm:.4e}" for c, norm in enumerate(avg_vfl_norms1)])
                    logger.info(f"Client VFL boundary gradient norms at round {round_idx + 1}, step 0: {vfl_norms_str}")

                    gat_grad_norms = []
                    for client_id, client_model in self.client_models.items():
                        total_norm = 0.0
                        for name, param in client_model.named_parameters():
                            if "gat" in name and param.grad is not None:
                                param_norm = param.grad.data.norm(2).item()
                                total_norm += param_norm ** 2
                        total_norm = total_norm ** 0.5
                        gat_grad_norms.append(total_norm)
                    grad_norms_str = ", ".join([f"Client {c+1}: {norm:.4e}" for c, norm in enumerate(gat_grad_norms)])
                    logger.info(f"Client GAT parameter gradient norms at round {round_idx + 1}, step 0: {grad_norms_str}")

                # Accumulate correct predictions (doing it outside the inner loop to save CPU-GPU synchronizations)
                if len(batch_preds) > 0:
                    batch_preds_tensor = torch.cat(batch_preds, dim=0)
                    batch_labels_tensor = torch.cat(batch_labels, dim=0)
                    correct_preds_in_step = (batch_preds_tensor == batch_labels_tensor).sum().item()
                    correct_preds_in_interval += correct_preds_in_step
                    preds_in_interval.append(batch_preds_tensor.detach().cpu())
                    labels_in_interval.append(batch_labels_tensor.detach().cpu())

                scaler.step(optimizer)
                scaler.update()
                round_loss += step_loss.item()

                # Step-level Logging
                if (step + 1) % log_step_every == 0 or (step + 1) == num_steps:
                    num_snapshots_in_interval = step_count_in_interval * len(batch_indices)
                    if num_snapshots_in_interval > 0:
                        avg_clf_loss = (clf_loss_in_interval / num_snapshots_in_interval).item()
                        avg_supcon_loss = supcon_loss_in_interval / step_count_in_interval
                        avg_batch_acc = correct_preds_in_interval / num_snapshots_in_interval
                        avg_client_norms = (client_norms_in_interval / num_snapshots_in_interval).detach().cpu().numpy()
                        avg_server_norm = (server_emb_norm_in_interval / num_snapshots_in_interval).item()
                        
                        loss_str = f"Loss: {step_loss.item():.4f} (Clf: {avg_clf_loss:.4f}"
                        if use_contrastive:
                            loss_str += f", Contrastive: {avg_supcon_loss:.4f}, lambda: {lambda_t:.4f})"
                        else:
                            loss_str += ")"
                            
                        client_norms_str = ", ".join([f"Client {c+1}: {norm:.4f}" for c, norm in enumerate(avg_client_norms)])
                        
                        if len(preds_in_interval) > 0:
                            all_preds = torch.cat(preds_in_interval, dim=0).numpy()
                            all_labels = torch.cat(labels_in_interval, dim=0).numpy()
                            precision = precision_score(all_labels, all_preds, zero_division=0)
                            recall = recall_score(all_labels, all_preds, zero_division=0)
                            f1 = f1_score(all_labels, all_preds, zero_division=0)
                            metrics_str = (
                                f"Batch Acc: {avg_batch_acc * 100:.2f}% | "
                                f"Prec: {precision * 100:.2f}% | "
                                f"Rec: {recall * 100:.2f}% | "
                                f"F1: {f1 * 100:.2f}%"
                            )
                        else:
                            metrics_str = f"Batch Acc: {avg_batch_acc * 100:.2f}%"

                        logger.info(
                            f"  [Round {round_idx + 1} | Step {step + 1}/{num_steps}] "
                            f"{loss_str} | {metrics_str} | "
                            f"Server norm: {avg_server_norm:.4f} | Client norms: {client_norms_str} | "
                            f"Time: {time.time() - training_start_time:.2f}s (Step: {time.time() - step_start:.4f}s)"
                        )
                        
                    # Reset buffers for the next interval
                    step_count_in_interval = 0
                    correct_preds_in_interval = 0
                    preds_in_interval = []
                    labels_in_interval = []
                    clf_loss_in_interval = torch.tensor(0.0, device=self.device)
                    supcon_loss_in_interval = 0.0
                    client_norms_in_interval = torch.zeros(self.num_clients, device=self.device)
                    server_emb_norm_in_interval = torch.tensor(0.0, device=self.device)

            avg_round_loss = round_loss / num_steps
            round_time = time.time() - round_start
            self.results["training_losses"].append(avg_round_loss)
            self.results["round_times"].append(round_time)

            logger.info(
                f"Round {round_idx + 1} completed in {round_time:.2f}s, loss: {avg_round_loss:.4f}"
            )
            # Only use scheduler in Phase 2
            if self.current_phase == 2:
                old_lrs = [group["lr"] for group in optimizer.param_groups]
                scheduler.step(avg_round_loss)
                new_lrs = [group["lr"] for group in optimizer.param_groups]
                for group_idx, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                    if old_lr != new_lr:
                        group_name = "Server" if (two_speed_lr and group_idx == 0) else "Client" if (two_speed_lr and group_idx == 1) else "All layers"
                        logger.info(f"Learning rate for {group_name} updated mid-training in Phase 2: {old_lr:.6f} -> {new_lr:.6f}")

            # Check early stopping / best loss improvement
            if avg_round_loss < best_loss:
                best_loss = avg_round_loss
                best_round = round_idx
                self.best_loss = best_loss
                self.best_round = best_round
                no_improvement_count = 0
                logger.info(f"🏆 New best loss achieved at round {round_idx + 1}: {best_loss:.4f}")
                
                # Save best state dicts in memory
                best_global_state = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
                best_client_states = {
                    cid: {k: v.cpu().clone() for k, v in client_model.state_dict().items()}
                    for cid, client_model in self.client_models.items()
                }
                
                # Save best checkpoint to disk
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, round_idx, is_best=True)
            else:
                no_improvement_count += 1
                limit_patience = 3 if self.current_phase == 1 else early_stopping_patience
                logger.info(
                    f"Loss did not improve. Current best loss: {best_loss:.4f} (from round {best_round + 1}). "
                    f"Rounds without improvement: {no_improvement_count}/{limit_patience}"
                )

            # Regular checkpointing
            if checkpoint_dir and (
                (round_idx - start_round + 1) % checkpoint_every == 0
                or (num_rounds is not None and round_idx == num_rounds - 1)
            ):
                self.save_checkpoint(checkpoint_dir, round_idx)

            # Update persistent fields
            self.best_loss_phase1 = best_loss if self.current_phase == 1 else self.best_loss_phase1
            self.no_improvement_count = no_improvement_count

            # Phase Transition and Stopping logic
            if self.current_phase == 1:
                if no_improvement_count >= 3:
                    logger.info(
                        f"🛑 Phase 1 (Classification Warm-up) plateau reached after {round_idx + 1} rounds. "
                        f"No improvement for 3 consecutive rounds. Saving baseline checkpoint."
                    )
                    if checkpoint_dir:
                        # Save checkpoint_clf_only_plateau.pt
                        save_path = os.path.join(checkpoint_dir, "checkpoint_clf_only_plateau.pt")
                        checkpoint = self._create_checkpoint_dict(round_idx)
                        try:
                            self._safe_torch_save(checkpoint, save_path)
                            logger.info(f"Saved Phase 1 baseline checkpoint to {save_path}")
                        except Exception as e:
                            logger.error(f"Failed to save Phase 1 baseline checkpoint: {e}")

                    # Transition to Phase 2
                    self.current_phase = 2
                    self.phase2_rounds_trained = 0
                    
                    # Step learning rate down by 0.5
                    for group in optimizer.param_groups:
                        group["lr"] = group["lr"] * 0.5
                    logger.info(f"Stepping down learning rate to {optimizer.param_groups[0]['lr']} for Phase 2 fine-tuning.")
                    
                    # Reset best loss and no improvement count for Phase 2
                    best_loss = float("inf")
                    best_round = -1
                    self.best_loss = float("inf")
                    self.best_round = -1
                    no_improvement_count = 0
                    self.no_improvement_count = 0
            else:
                # In Phase 2
                self.phase2_rounds_trained += 1
                if no_improvement_count >= early_stopping_patience:
                    logger.info(
                        f"🛑 Phase 2 (Joint Contrastive Fine-Tuning) plateau reached after {self.phase2_rounds_trained} rounds in Phase 2. "
                        f"No improvement for {early_stopping_patience} consecutive rounds."
                    )
                    if checkpoint_dir:
                        # Save checkpoint_joint_final.pt
                        save_path = os.path.join(checkpoint_dir, "checkpoint_joint_final.pt")
                        checkpoint = self._create_checkpoint_dict(round_idx)
                        try:
                            self._safe_torch_save(checkpoint, save_path)
                            logger.info(f"Saved Phase 2 final checkpoint to {save_path}")
                        except Exception as e:
                            logger.error(f"Failed to save Phase 2 final checkpoint: {e}")
                    break

            round_idx += 1

        # Restore the best model weights for final evaluation
        if best_global_state is not None:
            self.global_model.load_state_dict(best_global_state)
            for cid, state in best_client_states.items():
                self.client_models[cid].load_state_dict(state)
            logger.info(
                f"Loaded best weights back into models from round {best_round + 1} "
                f"with loss {best_loss:.4f} for final evaluation."
            )
        elif checkpoint_dir:
            best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_checkpoint_path):
                logger.info(f"Loading best checkpoint from disk: {best_checkpoint_path}")
                self.load_checkpoint(best_checkpoint_path, load_training_state=False)

        # Clean up optimizer, scheduler, scaler references to avoid memory leaks
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        logger.info("Joint federated VFL training completed")
        return self.results

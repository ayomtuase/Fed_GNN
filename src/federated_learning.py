"""Federated learning orchestration for graph-level anomaly detection.
Handles client graph construction, local training, checkpointing, and model aggregation.
"""

from gnn_models import nt_xent_loss
import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from gnn_models import GATLayer, GlobalGAT, GlobalGraphSAGE

logger = logging.getLogger(__name__)


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[float] = 0.5,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary Focal Loss implementation: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""
    if logits.ndim != targets.ndim:
        targets = targets.view_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ((1 - p_t) ** gamma) * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def augment_contrastive(x: torch.Tensor) -> torch.Tensor:
    """
    Applies domain-specific augmentations to create a second view:
    temporal masking or scale/jittering (Gaussian noise).
    x shape: (B, window_size, num_sensors)
    """
    if x.shape[0] == 0:
        return x
    x_aug = x.clone()
    B, window_size, num_sensors = x_aug.shape
    device = x.device
    
    # Decide which elements of the batch get temporal masking vs scale/jittering
    # to have a mixed diversity of views
    mask_indices = torch.rand(B, device=device) < 0.5
    noise_scaling_indices = ~mask_indices
    
    # 1. Temporal Masking
    if mask_indices.any():
        num_masked = mask_indices.sum().item()
        # Choose a mask length (e.g. 15% of window_size, at least 1)
        mask_len = max(1, int(window_size * 0.15))
        
        # Generate random start indices for each masked batch item (aligned across sensors)
        starts = torch.randint(0, window_size - mask_len + 1, (num_masked,), device=device)
        grid = torch.arange(window_size, device=device).view(1, window_size, 1)
        starts_expanded = starts.view(num_masked, 1, 1)  # (num_masked, 1, 1)
        mask = (grid < starts_expanded) | (grid >= (starts_expanded + mask_len))  # (num_masked, window_size, 1)
        
        x_aug[mask_indices] = x_aug[mask_indices] * mask.to(dtype=x.dtype)
        
    # 2. Scaling & Jittering (Preserves sequence order and physical continuity)
    if noise_scaling_indices.any():
        num_ns = noise_scaling_indices.sum().item()
        ns_samples = x_aug[noise_scaling_indices]
        
        # Scale: Multiply each sensor's window by a random factor in [0.9, 1.1]
        scale = 0.9 + 0.2 * torch.rand(num_ns, 1, num_sensors, device=device, dtype=x.dtype)
        
        # Jitter: Add small Gaussian noise with std=0.03
        noise = torch.randn_like(ns_samples) * 0.03
        
        x_aug[noise_scaling_indices] = ns_samples * scale + noise
        
    return x_aug


class VFLGradientNormalizer(torch.autograd.Function):
    """Custom autograd function to normalize gradients globally across the VFL boundary."""
    @staticmethod
    def forward(ctx, target_norm, *inputs):
        ctx.target_norm = target_norm
        return tuple(x.clone() for x in inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = [g for g in grad_outputs if g is not None]
        if len(grads) == 0:
            return (None,) + grad_outputs
        
        # CRITICAL FIX: Calculate the sum of squares in float32
        global_norm = torch.sqrt(sum((g.float().norm(2) ** 2) for g in grads) + 1e-8)
        
        scaled_grads = []
        for g in grad_outputs:
            if g is not None:
                scaled_grads.append(g / global_norm * ctx.target_norm)
            else:
                scaled_grads.append(None)
        
        return (None,) + tuple(scaled_grads)


class VFLDifferentialPrivacy(torch.autograd.Function):
    """Custom autograd function to apply differential privacy (clipping + noise) 
    to client embeddings sent across the VFL boundary."""
    @staticmethod
    def forward(ctx, clip_bound, noise_multiplier, *inputs):
        ctx.clip_bound = clip_bound
        ctx.noise_multiplier = noise_multiplier
        
        dp_inputs = []
        for x in inputs:
            if x is not None:
                # 1. Node-wise L2 norm clipping
                x_fp32 = x.float()
                row_norms = x_fp32.norm(2, dim=-1, keepdim=True)
                clip_coef = torch.clamp(clip_bound / (row_norms + 1e-8), max=1.0)
                x_clipped = x * clip_coef.to(x.dtype)
                
                # 2. Dimension-normalized Gaussian noise
                if noise_multiplier > 0.0 and clip_bound > 0.0:
                    d = x_clipped.shape[-1]
                    scale = (noise_multiplier * clip_bound) / (d ** 0.5)
                    noise = torch.randn_like(x_clipped) * scale
                    x_dp = x_clipped + noise
                else:
                    x_dp = x_clipped
                
                dp_inputs.append(x_dp)
            else:
                dp_inputs.append(None)
        return tuple(dp_inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Pass gradients back to client models unchanged
        return (None, None) + grad_outputs



def supervised_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    # 1. Strictly enforce float32
    z1 = z1.to(torch.float32)
    z2 = z2.to(torch.float32)
    
    """Supervised Contrastive Loss (SupCon) with Normal class alignment.
    
    Focuses on aligning Normal-to-Normal pairs (Class 0) and View 1-to-View 2 pairs,
    ignoring Anomaly-to-Anomaly positive pairs to learn a dense normal core manifold.
    """
    device = z1.device
    B = z1.shape[0]
    if B <= 1:
        return torch.tensor(0.0, dtype=torch.float32, device=device)
        
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
    
    # 2. Normal-to-Normal positive pairs (same label = 0)
    normal_mask_2b = (labels_double == 0).float().view(-1, 1) # (2B, 1)
    normal_pairs_mask = torch.matmul(normal_mask_2b, normal_mask_2b.T) * logits_mask
    
    # Combine masks: positive pairs are either View1-to-View2 pairs or Normal-Normal pairs
    mask = torch.clamp(v1_v2_mask + normal_pairs_mask, max=1.0)
    
    # For numerical stability: subtract the max of NON-DIAGONAL/NON-SELF logits
    # Replace masked positions with a large negative value so they don't affect the max
    masked_sim_matrix = similarity_matrix.clone()
    masked_sim_matrix[logits_mask == 0] = -1e9
    
    # Detach logits_max so gradients don't flow through the max operation
    logits_max, _ = torch.max(masked_sim_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Compute log_prob (Now safe from overflow because of float32 and max subtraction)
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


class FederatedDataset(Dataset):
    """Memory-mapped dataset for multi-client federated windows, targets (next step), and labels."""
    def __init__(self, client_paths: List[str], labels_path: str, window_size: int, max_samples: Optional[int] = None, dtype: torch.dtype = torch.float32):
        self.client_paths = client_paths
        self._client_mmaps = None
        self.labels = np.load(labels_path) # labels are small, load directly
        self.window_size = window_size
        self.dtype = dtype
        
        # Calculate maximum sliding windows we can retrieve (we need next step for target)
        self.length = max(0, len(self.labels) - window_size)
        if max_samples is not None:
            self.length = min(self.length, max_samples)

    @property
    def client_mmaps(self) -> List[np.ndarray]:
        if self._client_mmaps is None:
            self._client_mmaps = [np.load(path, mmap_mode='r') for path in self.client_paths]
        return self._client_mmaps

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        # Contiguous slices dynamically extracted on the fly
        client_feats = [
            torch.from_numpy(self.client_mmaps[c][idx : idx + self.window_size].copy()).to(self.dtype)
            for c in range(len(self.client_mmaps))
        ]
        client_targets = [
            torch.from_numpy((self.client_mmaps[c][idx + self.window_size] - self.client_mmaps[c][idx + self.window_size - 1]).copy()).to(self.dtype)
            for c in range(len(self.client_mmaps))
        ]
        label = torch.tensor(self.labels[idx + self.window_size], dtype=torch.long)
        return client_feats, client_targets, label



class FedGATSageSystem:
    """Main FedGATSage federated learning system."""

    def __init__(
        self,
        data_dir: str,
        num_clients: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir: Optional[str] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
    ):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if isinstance(dtype, str):
            if dtype in ["float64", "double"]:
                self.dtype = torch.float64
            elif dtype in ["float16", "half"]:
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        self.client_models: Dict[int, nn.Module] = {}
        self.global_model: Optional[nn.Module] = None
        self.results: Dict[str, Any] = {
            "training_losses": [],
            "round_times": [],
            "training_accuracies": [],
            "training_precisions": [],
            "training_recalls": [],
            "training_f1s": [],
            "training_aucs": [],
            "val_losses": [],
            "val_aucs": [],
            "val_f1s": [],
        }
        self.input_dim: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.label_mapper: Optional[Dict[Any, int]] = None
        self.best_threshold = 0.5

        self.streams = [torch.cuda.Stream() for _ in range(num_clients)] if torch.cuda.is_available() else None

        logger.info("Initialized FedGATSageSystem")

    def initialize_models(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        num_classes: int = 2,
        node_num: int = 100,
        client_topk: Union[int, float] = 3,
        global_topk: int = 7,
        client_node_nums: Optional[List[int]] = None,
        use_residual: bool = True,
        use_concat_skip: bool = True,
        kernel_size: int = 7,
        use_sensor_embeddings: bool = True,
        sensor_embed_mode: str = "graph_construction",
        sensor_embedding_dim: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.node_num = node_num
        self.client_topk = client_topk
        self.global_topk = global_topk
        self.kernel_size = kernel_size
        self.use_sensor_embeddings = use_sensor_embeddings
        self.sensor_embed_mode = sensor_embed_mode
        self.sensor_embedding_dim = sensor_embedding_dim

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
                client_topk=client_topk,
                use_residual=use_residual,
                use_concat_skip=use_concat_skip,
                kernel_size=kernel_size,
                use_sensor_embeddings=use_sensor_embeddings,
                sensor_embed_mode=sensor_embed_mode,
                sensor_embedding_dim=sensor_embedding_dim,
            )
            self.client_models[client_id] = model.to(self.device).to(self.dtype)

        global_input_dim = hidden_dim * 2 if use_concat_skip else hidden_dim
        self.global_model = GlobalGraphSAGE(
            input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_clients=self.num_clients,
            use_concat_skip=use_concat_skip,
        ).to(self.device).to(self.dtype)

        logger.info(
            f"Initialized {self.num_clients} client models with node counts {client_node_nums} "
            f"and global GraphSAGE with hidden_dim={hidden_dim}, client_topk={client_topk}, global_topk={global_topk}, kernel_size={kernel_size}, "
            f"use_sensor_embeddings={use_sensor_embeddings}, sensor_embed_mode={sensor_embed_mode}, sensor_embedding_dim={sensor_embedding_dim}"
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
            "client_topk": getattr(self, "client_topk", 3),
            "global_topk": getattr(self, "global_topk", 7),
            "kernel_size": getattr(self, "kernel_size", 15),
            "use_sensor_embeddings": getattr(self, "use_sensor_embeddings", True),
            "sensor_embed_mode": getattr(self, "sensor_embed_mode", "both"),
            "sensor_embedding_dim": getattr(self, "sensor_embedding_dim", None),
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
            "best_val_auc": getattr(self, "best_val_auc", 0.0),
            "best_val_f1": getattr(self, "best_val_macro_f1", 0.0),
            "best_val_macro_f1": getattr(self, "best_val_macro_f1", 0.0),
            "no_improvement_count": getattr(self, "no_improvement_count", 0),
            "best_loss": getattr(self, "best_loss", float("inf")),
            "best_round": getattr(self, "best_round", -1),
            "best_threshold": getattr(self, "best_threshold", 0.5),
            "val_medians": getattr(self, "val_medians", None),
            "val_iqrs": getattr(self, "val_iqrs", None),
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
            self.best_threshold = checkpoint.get("best_threshold", getattr(self, "best_threshold", 0.5))
            if "val_medians" in checkpoint:
                self.val_medians = checkpoint["val_medians"]
            if "val_iqrs" in checkpoint:
                self.val_iqrs = checkpoint["val_iqrs"]

            if load_training_state:
                self.results = checkpoint.get("results", self.results)
                if not isinstance(self.results, dict):
                    self.results = {}
                for key in ["training_losses", "round_times", "training_accuracies", "training_precisions", "training_recalls", "training_f1s", "training_aucs", "val_losses", "val_aucs", "val_f1s"]:
                    if key not in self.results or not isinstance(self.results[key], list):
                        self.results[key] = []
                self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
                self.best_val_macro_f1 = checkpoint.get("best_val_macro_f1", checkpoint.get("best_val_f1", 0.0))
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

            if "num_clients" in checkpoint:
                self.num_clients = checkpoint["num_clients"]
                if torch.cuda.is_available() and (self.streams is None or len(self.streams) != self.num_clients):
                    self.streams = [torch.cuda.Stream() for _ in range(self.num_clients)]

            if "client_node_nums" in checkpoint:
                self.client_node_nums = checkpoint["client_node_nums"]

            if not self.client_models:
                input_dim = checkpoint.get("input_dim", 1)
                hidden_dim = checkpoint.get("hidden_dim", 256)
                num_classes = checkpoint.get("num_classes", 2)
                node_num = checkpoint.get("node_num", 100)
                client_node_nums = checkpoint.get("client_node_nums", None)
                client_topk = checkpoint.get("client_topk", checkpoint.get("topk", 3))
                global_topk = checkpoint.get("global_topk", checkpoint.get("topk", 7))
                use_concat_skip = checkpoint.get("use_concat_skip", True)
                kernel_size = checkpoint.get("kernel_size", 15)
                use_sensor_embeddings = checkpoint.get("use_sensor_embeddings", True)
                sensor_embed_mode = checkpoint.get("sensor_embed_mode", "graph_construction")
                sensor_embedding_dim = checkpoint.get("sensor_embedding_dim", None)
                self.initialize_models(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    node_num=node_num,
                    client_topk=client_topk,
                    global_topk=global_topk,
                    client_node_nums=client_node_nums,
                    use_concat_skip=use_concat_skip,
                    kernel_size=kernel_size,
                    use_sensor_embeddings=use_sensor_embeddings,
                    sensor_embed_mode=sensor_embed_mode,
                    sensor_embedding_dim=sensor_embedding_dim,
                )

            client_states = checkpoint.get("client_models", {})
            for client_id, state_dict in client_states.items():
                if client_id in self.client_models:
                    self.client_models[client_id].load_state_dict(state_dict, strict=False)
                    self.client_models[client_id] = self.client_models[client_id].to(self.device).to(self.dtype)
                elif str(client_id).isdigit() and int(client_id) in self.client_models:
                    self.client_models[int(client_id)].load_state_dict(state_dict, strict=False)
                    self.client_models[int(client_id)] = self.client_models[int(client_id)].to(self.device).to(self.dtype)
                else:
                    logger.warning(
                        f"Skipping missing client model state for {client_id}"
                    )

            global_state = checkpoint.get("global_model")
            if self.global_model is not None and global_state is not None:
                self.global_model.load_state_dict(global_state, strict=False)
                self.global_model = self.global_model.to(self.device).to(self.dtype)

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
        N_global = sum(self.client_node_nums)
        B = h_global.shape[0] // N_global

        if B > 1:
            # CRITICAL FIX: Cast to float32 before similarity math to prevent AMP overflow
            weights = h_global.detach().clone().float().view(B, N_global, -1)
            cos_sim_mat = torch.bmm(weights, weights.transpose(1, 2))  # (B, N_global, N_global)

            norms = weights.norm(dim=-1, keepdim=True)  # (B, N_global, 1)
            normed_mat = torch.bmm(norms, norms.transpose(1, 2))  # (B, N_global, N_global)
            cos_sim_mat = cos_sim_mat / (normed_mat + 1e-8)

            # Prevent self-loops by masking the diagonal
            eye = torch.eye(N_global, device=cos_sim_mat.device, dtype=torch.bool).unsqueeze(0)
            cos_sim_mat = cos_sim_mat.masked_fill(eye, -1e9)

            topk_num = min(topk, N_global - 1)
            topk_indices = torch.topk(cos_sim_mat, topk_num, dim=-1)[1]  # (B, N_global, topk)

            batch_offsets = torch.arange(0, B, device=h_global.device).view(B, 1, 1) * N_global
            to_nodes = (topk_indices + batch_offsets).flatten()

            from_nodes_local = torch.arange(0, N_global, device=h_global.device).view(1, N_global, 1)
            from_nodes = (from_nodes_local.repeat(B, 1, topk_num) + batch_offsets).flatten()

            edge_index = torch.stack([from_nodes, to_nodes], dim=0)
        else:
            # CRITICAL FIX: Cast to float32 before similarity math to prevent AMP overflow
            weights = h_global.detach().clone().float()
            cos_sim_mat = torch.matmul(weights, weights.T)  # (N_global, N_global)

            norms = weights.norm(dim=-1).view(-1, 1)  # (N_global, 1)
            normed_mat = torch.matmul(norms, norms.T)  # (N_global, N_global)
            cos_sim_mat = cos_sim_mat / (normed_mat + 1e-8)

            # Prevent self-loops by masking the diagonal
            eye = torch.eye(cos_sim_mat.shape[0], device=cos_sim_mat.device, dtype=torch.bool)
            cos_sim_mat = cos_sim_mat.masked_fill(eye, -1e9)

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

    def set_system_dropout(self, p: float):
        """Update dropout rates in all client models and the global model."""
        logger.info(f"Setting dropout rate to {p} for all models.")
        from torch_geometric.nn import GATConv

        def _set_dropout(model, rate):
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = rate
                if isinstance(module, GATConv):
                    module.dropout = rate

        _set_dropout(self.global_model, p)
        for client_model in self.client_models.values():
            _set_dropout(client_model, p)

    def train_federated(
        self,
        num_rounds: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1,
        start_round: int = 0,
        num_samples: int = 5,
        oversample_scale: float = 2.0,
        focal_loss_alpha: float = 0.25,
        use_ce_loss: bool = False,
        use_oversampling: bool = False,
        two_speed_lr: bool = True,
        lr_server: float = 0.0003,
        lr_client: float = 0.0005,
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1,
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
        early_stopping_patience: int = 10,
        num_workers: int = 0,
        dp_enabled: bool = False,
        dp_clip_bound: float = 21.0,
        dp_noise_multiplier: float = 0.01,
        window_size: int = 30,
        threshold_percentile: float = 99.0,
        top_k_agg: int = 1,
        smoothing_window: int = 10,
    ) -> Dict[str, Any]:
        self.dp_enabled = dp_enabled
        self.dp_clip_bound = dp_clip_bound
        self.dp_noise_multiplier = dp_noise_multiplier
        self.unclipped_norms_tracker = [[] for _ in range(self.num_clients)]

        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        rounds_str = str(num_rounds) if num_rounds is not None else "∞"
        logger.info(
            f"Starting joint federated VFL unsupervised training from round {start_round + 1} to {rounds_str} "
            f"with neighbor sampling num_samples={num_samples}, oversample_scale={oversample_scale}, "
            f"two_speed_lr={two_speed_lr}, "
            f"use_contrastive={use_contrastive}, contrastive_weight={contrastive_weight}, "
            f"normalize_vfl_gradients={normalize_vfl_gradients}, early_stopping_patience={early_stopping_patience}, "
            f"dp_enabled={dp_enabled}, dp_clip_bound={dp_clip_bound}, dp_noise_multiplier={dp_noise_multiplier}"
        )
        if start_round == 0 or not hasattr(self, "best_val_loss"):
            self.best_val_loss = float("inf")
            self.best_round = -1
            self.no_improvement_count = 0

        best_val_loss = self.best_val_loss
        best_round = self.best_round
        no_improvement_count = self.no_improvement_count
        best_global_state = None
        best_client_states = {}
        best_val_medians = getattr(self, "val_medians", None)
        best_val_iqrs = getattr(self, "val_iqrs", None)
        best_val_threshold = getattr(self, "best_threshold", 0.5)

        # Truncate results to start_round to ensure consistency if we resume
        if isinstance(self.results, dict):
            for key in ["training_losses", "round_times", "training_accuracies", "training_precisions", "training_recalls", "training_f1s", "training_aucs", "val_losses", "val_aucs", "val_f1s"]:
                if key in self.results and isinstance(self.results[key], list):
                    self.results[key] = self.results[key][:start_round]

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

        # Set up Federated Datasets and Loaders
        train_labels_path = os.path.join(self.data_dir, "train_labels.npy")
        
        # Check validation folder/labels compatibility to avoid size mismatches
        val_client_paths = [os.path.join(self.data_dir, "validation", f"client_{c+1}.npy") for c in range(self.num_clients)]
        val_labels_path = os.path.join(self.data_dir, "validation_labels.npy")
        if not all(os.path.exists(p) for p in val_client_paths) or not os.path.exists(val_labels_path):
            val_client_paths = [os.path.join(self.data_dir, "val", f"client_{c+1}.npy") for c in range(self.num_clients)]
            val_labels_path = os.path.join(self.data_dir, "val_labels.npy")

        train_client_paths = [os.path.join(self.data_dir, "train", f"client_{c+1}.npy") for c in range(self.num_clients)]

        train_dataset = FederatedDataset(train_client_paths, train_labels_path, window_size=window_size, max_samples=max_samples, dtype=self.dtype)
        val_dataset = FederatedDataset(val_client_paths, val_labels_path, window_size=window_size, max_samples=max_samples, dtype=self.dtype)

        # Determine if the active device uses discrete VRAM
        is_discrete_gpu = torch.device(self.device).type == "cuda"

        # Halve batch size if contrastive is active to prevent OOM
        current_batch_size = batch_size
        if use_contrastive:
            current_batch_size = max(1, batch_size // 2)
            logger.info(f"Using batch size {current_batch_size} due to contrastive view duplication.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            pin_memory=is_discrete_gpu,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=current_batch_size,
            shuffle=False,
            pin_memory=is_discrete_gpu,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0)
        )

        num_snapshots = len(train_dataset)
        logger.info(f"Loaded training data. Number of aligned snapshots: {num_snapshots}")

        # Build joint parameter list and optimizer
        if two_speed_lr:
            s_lr = lr_server
            c_lr = lr_client
            logger.info(f"Initializing Two-Speed LR: Server LR={s_lr}, Client LR={c_lr}")

            server_params = list(self.global_model.parameters())
            client_params = []
            for client_model in self.client_models.values():
                client_params.extend(list(client_model.parameters()))
            optimizer = torch.optim.Adam([
                {"params": server_params, "lr": s_lr},
                {"params": client_params, "lr": c_lr}
            ])
        else:
            current_lr = lr_client
            logger.info(f"Initializing single speed learning rate: {current_lr}")

            all_params = list(self.global_model.parameters())
            for client_model in self.client_models.values():
                all_params.extend(list(client_model.parameters()))
            optimizer = torch.optim.Adam(all_params, lr=current_lr)

        # Set system dropout for training robustness
        self.set_system_dropout(0.3)

        # Initialize the learning rate scheduler (ReduceLROnPlateau based on Validation Loss)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=min_lr
        )
        self.scheduler = scheduler

        # Determine the data type for mixed precision (AMP)
        device_type = torch.device(self.device).type
        if device_type == "mps":
            amp_dtype = torch.bfloat16
        elif device_type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float16

        actual_use_amp = use_amp and (device_type in ["cuda", "mps"])
        scaler_enabled = actual_use_amp and (amp_dtype == torch.float16)
        scaler_device = "mps" if device_type == "mps" else "cuda"
        scaler = torch.amp.GradScaler(scaler_device, enabled=scaler_enabled)
        self.scaler = scaler

        if actual_use_amp:
            logger.info(
                f"Mixed precision training enabled using device type: {device_type}, "
                f"data type: {amp_dtype}, GradScaler enabled: {scaler_enabled}"
            )

        # Store references for checkpointing
        self.optimizer = optimizer

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

        num_steps = len(train_loader)
        training_start_time = time.time()

        round_idx = start_round
        while True:
            if num_rounds is not None and round_idx >= num_rounds:
                logger.info(f"Reached maximum number of rounds: {num_rounds}. Stopping training.")
                break

            self.unclipped_norms_tracker = [[] for _ in range(self.num_clients)]

            rounds_str = str(num_rounds) if num_rounds is not None else "∞"
            round_start = time.time()
            
            # Lambda Warm-up: initialize contrastive_weight at 0.0 for the first 5 epochs (rounds)
            current_contrastive_weight = 0.0 if round_idx < 5 else contrastive_weight
            if use_contrastive:
                logger.info(f"Starting round {round_idx + 1}/{rounds_str} (contrastive_weight={current_contrastive_weight:.4f})")
            else:
                logger.info(f"Starting round {round_idx + 1}/{rounds_str}")

            self.global_model.train()
            for client_model in self.client_models.values():
                client_model.train()

            round_loss = 0.0
            
            # Initialize accumulation buffers for step logging
            step_count_in_interval = 0
            clf_loss_in_interval = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            supcon_loss_in_interval = 0.0
            client_norms_in_interval = torch.zeros(self.num_clients, dtype=self.dtype, device=self.device)
            server_emb_norm_in_interval = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            for step, (batch_features, batch_targets, batch_labels) in enumerate(train_loader):
                B = batch_labels.shape[0]
                step_start = time.time()

                optimizer.zero_grad()
                
                vfl_gradients1 = {c: [] for c in range(self.num_clients)}

                def make_grad_hook(client_idx, norm_list, normalize, target_norm, record_norm):
                    def hook(grad):
                        if grad is not None:
                            grad_fp32 = grad.float()
                            if record_norm:
                                grad_norm_val = grad_fp32.norm(2).item()
                                norm_list[client_idx].append(grad_norm_val)
                                if normalize:
                                    return grad / (grad_norm_val + 1e-8) * target_norm
                            else:
                                if normalize:
                                    return grad / (grad_fp32.norm(2) + 1e-8) * target_norm
                        return grad
                    return hook

                step_count_in_interval += 1

                # Gather batch tensors on GPU
                batch_features = [f.to(self.device, non_blocking=is_discrete_gpu) for f in batch_features]
                batch_targets = [t.to(self.device, non_blocking=is_discrete_gpu) for t in batch_targets]
                batch_labels = batch_labels.to(self.device, non_blocking=is_discrete_gpu)

                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=actual_use_amp):
                    # 1. Client Parallel Forward Pass
                    h_client_combined_list = [None] * self.num_clients
                    B_factor = 2 if use_contrastive else 1

                    if self.streams is not None:
                        for c in range(self.num_clients):
                            with torch.cuda.stream(self.streams[c]):
                                x_c_clean = batch_features[c]
                                if use_contrastive:
                                    x_c_noisy = augment_contrastive(x_c_clean)
                                    x_c_combined = torch.cat([x_c_clean, x_c_noisy], dim=0)
                                else:
                                    x_c_combined = x_c_clean
                                
                                x_c_flat = x_c_combined.transpose(1, 2).reshape((B * B_factor) * self.client_node_nums[c], -1)
                                h_c_combined = self.client_models[c](x_c_flat)
                                
                                if h_c_combined.requires_grad and step == 0:
                                    h_c_combined.register_hook(make_grad_hook(c, vfl_gradients1, False, vfl_target_norm, True))
                                
                                h_client_combined_list[c] = h_c_combined
                        torch.cuda.synchronize()
                    else:
                        for c in range(self.num_clients):
                            x_c_clean = batch_features[c]
                            if use_contrastive:
                                x_c_noisy = augment_contrastive(x_c_clean)
                                x_c_combined = torch.cat([x_c_clean, x_c_noisy], dim=0)
                            else:
                                x_c_combined = x_c_clean
                            
                            x_c_flat = x_c_combined.transpose(1, 2).reshape((B * B_factor) * self.client_node_nums[c], -1)
                            h_c_combined = self.client_models[c](x_c_flat)
                            
                            if h_c_combined.requires_grad and step == 0:
                                h_c_combined.register_hook(make_grad_hook(c, vfl_gradients1, False, vfl_target_norm, True))
                            
                            h_client_combined_list[c] = h_c_combined

                    h_server_inputs = list(h_client_combined_list)

                    # Record unclipped row-wise (node-wise) L2 norms of client embeddings
                    for c in range(self.num_clients):
                        emb_tensor = h_server_inputs[c]
                        if emb_tensor is not None:
                            with torch.no_grad():
                                row_norms = emb_tensor.float().norm(2, dim=-1).cpu().numpy().tolist()
                                self.unclipped_norms_tracker[c].extend(row_norms)

                    if normalize_vfl_gradients:
                        normalized_h_list = VFLGradientNormalizer.apply(vfl_target_norm, *h_server_inputs)
                        h_client_combined_list = list(normalized_h_list)
                    elif self.dp_enabled:
                        dp_h_list = VFLDifferentialPrivacy.apply(
                            self.dp_clip_bound,
                            self.dp_noise_multiplier,
                            *h_server_inputs
                        )
                        h_client_combined_list = list(dp_h_list)

                    N_global = sum(self.client_node_nums)

                    h_global_combined_batched = torch.cat([hc.view(B * B_factor, Nc, -1) for hc, Nc in zip(h_client_combined_list, self.client_node_nums)], dim=1)
                    h_global_combined = h_global_combined_batched.view((B * B_factor) * N_global, -1)

                    edge_index_combined = self._build_global_graph(h_global_combined, self.global_topk)

                    if use_contrastive:
                        # Topological Augmentation: drop 20% of edges in the noisy view (View 2)
                        is_noisy_edge = edge_index_combined[0] >= (B * N_global)
                        noisy_mask = torch.rand(is_noisy_edge.sum().item(), device=edge_index_combined.device) > 0.2
                        edge_mask = torch.ones(edge_index_combined.size(1), dtype=torch.bool, device=edge_index_combined.device)
                        edge_mask[is_noisy_edge] = noisy_mask
                        edge_index_combined = edge_index_combined[:, edge_mask]

                    emb_combined, _, _, contrastive_emb_combined = self.global_model(
                        h_global_combined,
                        edge_index_combined,
                        node_anomaly_scores=None,
                        num_samples=None,
                        num_nodes_per_graph=N_global,
                    )

                    # Chunk View 1 and View 2
                    if use_contrastive:
                        predictions1 = None
                        graph_contrastive_emb1, graph_contrastive_emb2 = contrastive_emb_combined.chunk(2, dim=0)
                        emb1 = emb_combined[:B * N_global]
                    else:
                        emb1 = emb_combined

                    # Accumulate representation norms
                    server_emb_norm_in_interval += emb1.detach().view(B, -1).norm(2, dim=1).sum()
                    for c in range(self.num_clients):
                        h_c_clean = h_client_combined_list[c][:B * self.client_node_nums[c]]
                        client_norms_in_interval[c] += h_c_clean.detach().view(B, -1).norm(2, dim=1).sum()

                    # Server-side contrastive NT-Xent loss
                    if use_contrastive:
                        with torch.amp.autocast(device_type=device_type, enabled=False):
                            supcon_loss = nt_xent_loss(
                                graph_contrastive_emb1.float(),
                                graph_contrastive_emb2.float(),
                                temperature=contrastive_temp,
                            )
                        supcon_loss_in_interval += supcon_loss.item()
                    else:
                        supcon_loss = torch.tensor(0.0, device=self.device)

                    # Client-side linear decoders forecasting target and calculating MSE
                    emb1_reshaped = emb1.view(B, N_global, -1)
                    mse_loss_total = torch.tensor(0.0, device=self.device)
                    for c in range(self.num_clients):
                        start_node = sum(self.client_node_nums[:c])
                        end_node = start_node + self.client_node_nums[c]
                        
                        client_emb_slice = emb1_reshaped[:, start_node:end_node, :]
                        client_emb_flat = client_emb_slice.reshape(-1, client_emb_slice.shape[-1])
                        
                        pred_c = self.client_models[c].decoder(client_emb_flat)
                        
                        target_c = batch_targets[c]
                        target_c_flat = target_c.reshape(-1, 1)
                        
                        mse_loss_c = F.mse_loss(pred_c, target_c_flat)
                        mse_loss_total = mse_loss_total + mse_loss_c
                    
                    mse_loss = mse_loss_total / self.num_clients
                    clf_loss_in_interval += mse_loss.detach() * B

                    if use_contrastive:
                        step_loss = mse_loss + (current_contrastive_weight * supcon_loss)
                    else:
                        step_loss = mse_loss

                scaler.scale(step_loss).backward()

                if actual_use_amp:
                    scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)
                for client_model in self.client_models.values():
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)

                # Log gradient norms at step 0
                if step == 0:

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

                scaler.step(optimizer)
                scaler.update()
                round_loss += step_loss.item()

                # Step-level Logging
                if (step + 1) % log_step_every == 0 or (step + 1) == num_steps:
                    avg_clf_loss = (clf_loss_in_interval / (step_count_in_interval * B)).item()
                    avg_supcon_loss = supcon_loss_in_interval / step_count_in_interval
                    avg_client_norms = (client_norms_in_interval / (step_count_in_interval * B)).detach().cpu().float().numpy()
                    avg_server_norm = (server_emb_norm_in_interval / (step_count_in_interval * B)).item()
                    
                    loss_str = f"Loss: {step_loss.item():.4f} (MSE: {avg_clf_loss:.4f}"
                    if use_contrastive:
                        loss_str += f", Contrastive: {avg_supcon_loss:.4f})"
                    else:
                        loss_str += ")"
                        
                    client_norms_str = ", ".join([f"Client {c+1}: {norm:.4f}" for c, norm in enumerate(avg_client_norms)])
                    
                    logger.info(
                        f"  [Round {round_idx + 1} | Step {step + 1}/{num_steps}] "
                        f"{loss_str} | "
                        f"Server norm: {avg_server_norm:.4f} | Client norms: {client_norms_str} | "
                        f"Time: {time.time() - training_start_time:.2f}s (Step: {time.time() - step_start:.4f}s)"
                    )
                    
                    step_count_in_interval = 0
                    clf_loss_in_interval = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                    supcon_loss_in_interval = 0.0
                    client_norms_in_interval = torch.zeros(self.num_clients, dtype=self.dtype, device=self.device)
                    server_emb_norm_in_interval = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            avg_round_loss = round_loss / num_steps
            round_time = time.time() - round_start

            # Calculate validation loss and update dynamic thresholding
            val_loss, _, _, _, _ = self.evaluate_validation(
                val_loader=val_loader,
                criterion=None,
                use_ce_loss=False,
                focal_loss_alpha=0.0,
                use_contrastive=use_contrastive,
                contrastive_weight=current_contrastive_weight,
                contrastive_temp=contrastive_temp,
                threshold_percentile=threshold_percentile,
                top_k_agg=top_k_agg,
                smoothing_window=smoothing_window,
            )

            self.results["training_losses"].append(avg_round_loss)
            self.results["round_times"].append(round_time)
            self.results["training_accuracies"].append(0.0)
            self.results["training_precisions"].append(0.0)
            self.results["training_recalls"].append(0.0)
            self.results["training_f1s"].append(0.0)
            self.results["training_aucs"].append(0.0)
            self.results["val_losses"].append(val_loss)
            self.results["val_aucs"].append(0.0)
            self.results["val_f1s"].append(0.0)

            logger.info(
                f"Round {round_idx + 1} completed in {round_time:.2f}s | Train Loss: {avg_round_loss:.4f} | Val Loss: {val_loss:.4f} | Dynamic Threshold: {self.best_threshold:.4f}\n"
            )

            # Scheduler step
            scheduler.step(val_loss)

            # Early stopping check based on validation loss (lower is better)
            improved = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_val_loss = best_val_loss
                best_round = round_idx
                self.best_round = best_round
                no_improvement_count = 0
                improved = True
                logger.info(
                    f"🏆 New best Validation performance achieved at round {round_idx + 1}: "
                    f"Val Loss = {val_loss:.6f}"
                )
                
                # Save best state dicts in memory
                best_global_state = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
                best_client_states = {
                    cid: {k: v.cpu().clone() for k, v in client_model.state_dict().items()}
                    for cid, client_model in self.client_models.items()
                }
                # Also save the best validation statistics
                best_val_medians = self.val_medians.copy() if hasattr(self, "val_medians") else None
                best_val_iqrs = self.val_iqrs.copy() if hasattr(self, "val_iqrs") else None
                best_val_threshold = self.best_threshold
                
                # Save best checkpoint to disk
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, round_idx, is_best=True)
            else:
                no_improvement_count += 1
                limit_patience = early_stopping_patience
                logger.info(
                    f"Validation performance did not improve. Current best Val Loss: {best_val_loss:.6f} (from round {best_round + 1}). "
                    f"Rounds without improvement: {no_improvement_count}/{limit_patience}"
                )

            # Regular checkpointing
            if checkpoint_dir and (
                (round_idx - start_round + 1) % checkpoint_every == 0
                or (num_rounds is not None and round_idx == num_rounds - 1)
            ):
                self.save_checkpoint(checkpoint_dir, round_idx)

            # Update persistent fields
            self.no_improvement_count = no_improvement_count

            # Stopping logic
            if no_improvement_count >= early_stopping_patience:
                logger.info(
                    f"🛑 Early stopping triggered after {round_idx + 1} rounds. "
                    f"No improvement in Validation Loss for {early_stopping_patience} consecutive rounds."
                )
                break

            # Explicitly delete dangling references to clear memory at the end of each round
            step_loss = None
            h_client_combined_list = None
            batch_features = None
            batch_targets = None
            batch_labels = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            round_idx += 1

        # Restore the best model weights for final evaluation
        if best_global_state is not None:
            self.global_model.load_state_dict(best_global_state)
            for cid, state in best_client_states.items():
                self.client_models[cid].load_state_dict(state)
            if best_val_medians is not None:
                self.val_medians = best_val_medians
            if best_val_iqrs is not None:
                self.val_iqrs = best_val_iqrs
            self.best_threshold = best_val_threshold
            logger.info(
                f"Loaded best weights back into models from round {best_round + 1} with validation loss {best_val_loss:.6f} for final evaluation."
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

    def evaluate_validation(
        self,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        use_ce_loss: bool = False,
        focal_loss_alpha: float = 0.25,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1,
        contrastive_temp: float = 0.07,
        threshold_percentile: Optional[float] = 99.9,
        top_k_agg: int = 1,
        smoothing_window: int = 10,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        self.global_model.eval()
        for client_model in self.client_models.values():
            client_model.eval()

        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []

        N_global = sum(self.client_node_nums)
        device_type = torch.device(self.device).type
        is_discrete_gpu = (device_type == "cuda")

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    batch_features, batch_targets, batch_labels = batch
                else:
                    batch_features, batch_labels = batch
                    batch_targets = [f[:, -1, :] for f in batch_features]
                B = batch_labels.shape[0]
                batch_features = [f.to(self.device, non_blocking=is_discrete_gpu) for f in batch_features]
                batch_targets = [t.to(self.device, non_blocking=is_discrete_gpu) for t in batch_targets]

                h_client_list = []
                for c in range(self.num_clients):
                    x_c_clean = batch_features[c]
                    x_c_flat = x_c_clean.transpose(1, 2).reshape(B * self.client_node_nums[c], -1)
                    h_c = self.client_models[c](x_c_flat)
                    
                    # Clip validation embeddings to match training distribution
                    if getattr(self, "dp_enabled", False):
                        row_norms = h_c.float().norm(2, dim=-1, keepdim=True)
                        clip_coef = torch.clamp(self.dp_clip_bound / (row_norms + 1e-8), max=1.0)
                        h_c = h_c * clip_coef.to(h_c.dtype)
                        
                    h_client_list.append(h_c)

                h_global_batched = torch.cat([hc.view(B, Nc, -1) for hc, Nc in zip(h_client_list, self.client_node_nums)], dim=1)
                h_global = h_global_batched.view(B * N_global, -1)

                edge_index = self._build_global_graph(h_global, self.global_topk)

                outputs = self.global_model(
                    h_global,
                    edge_index,
                    node_anomaly_scores=None,
                    num_samples=None,
                    num_nodes_per_graph=N_global,
                )
                emb = outputs[0] if (outputs is not None and isinstance(outputs, tuple) and len(outputs) > 0) else None
                expected_dim = self.client_models[0].decoder.in_features
                if emb is None or (hasattr(emb, "shape") and emb.shape[-1] != expected_dim):
                    emb = torch.zeros(B * N_global, expected_dim, device=self.device)

                # Client-side forecasting
                emb_reshaped = emb.view(B, N_global, -1)
                
                batch_preds = []
                batch_targets_aligned = []
                for c in range(self.num_clients):
                    start_node = sum(self.client_node_nums[:c])
                    end_node = start_node + self.client_node_nums[c]
                    
                    client_emb_slice = emb_reshaped[:, start_node:end_node, :]
                    client_emb_flat = client_emb_slice.reshape(-1, client_emb_slice.shape[-1])
                    
                    pred_c = self.client_models[c].decoder(client_emb_flat)
                    pred_c_reshaped = pred_c.view(B, self.client_node_nums[c])
                    
                    batch_preds.append(pred_c_reshaped)
                    batch_targets_aligned.append(batch_targets[c])
                
                global_preds = torch.cat(batch_preds, dim=1) # (B, N_global)
                global_targets = torch.cat(batch_targets_aligned, dim=1) # (B, N_global)
                
                val_preds_list.append(global_preds)
                val_targets_list.append(global_targets)

        # Compute validation loss and metrics
        preds_all = torch.cat(val_preds_list, dim=0).cpu().numpy()
        targets_all = torch.cat(val_targets_list, dim=0).cpu().numpy()
        
        errors_np = np.abs(targets_all - preds_all)
        val_loss = float(np.mean(errors_np ** 2))
        
        # Calculate medians and IQRs for normalization
        medians = np.median(errors_np, axis=0) # (N_global,)
        iqrs = np.percentile(errors_np, 75, axis=0) - np.percentile(errors_np, 25, axis=0) # (N_global,)
        
        self.val_medians = medians
        self.val_iqrs = iqrs
        
        # Normalize validation errors (IQR flooring to prevent stable sensors from triggering false positives)
        safe_iqrs = np.maximum(iqrs, 0.05)
        normalized_errors = (errors_np - medians) / safe_iqrs
        
        # System score: top-k error aggregation across nodes per time step
        if top_k_agg <= 1:
            A = np.max(normalized_errors, axis=1) # (num_val_steps,)
        else:
            top_k_errors = np.sort(normalized_errors, axis=1)[:, -top_k_agg:]
            A = np.mean(top_k_errors, axis=1)
        
        # SMA smoothing
        import pandas as pd
        A_smoothed = pd.Series(A).rolling(window=smoothing_window, min_periods=1).mean().values
        
        # Lock threshold using validation percentile
        if threshold_percentile is None or threshold_percentile >= 100.0:
            self.best_threshold = float(np.max(A_smoothed))
        else:
            self.best_threshold = float(np.percentile(A_smoothed, threshold_percentile))
        
        logger.info(f"Validation epoch completed. Locked anomaly threshold: {self.best_threshold:.6f}")

        dummy_probs = np.zeros(len(val_loader.dataset))
        dummy_labels = np.zeros(len(val_loader.dataset))
        return val_loss, 0.0, 0.0, dummy_probs, dummy_labels

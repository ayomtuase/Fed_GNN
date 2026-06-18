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


def supervised_contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor, temperature: float = 0.3
) -> torch.Tensor:
    """Supervised Contrastive Loss (SupCon).
    
    Args:
        z1: Tensor of shape (B, D) - representation of view 1
        z2: Tensor of shape (B, D) - representation of view 2
        labels: Tensor of shape (B,) - class labels
        temperature: temperature scale
        
    Returns:
        loss: scalar tensor
    """
    device = z1.device
    B = z1.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=device)
        
    # Normalize the embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate the two views
    # shape: (2B, D)
    features = torch.cat([z1, z2], dim=0)
    
    # Full labels list (2B,)
    labels_double = torch.cat([labels, labels], dim=0) # (2B,)
    
    # Compute similarity matrix (2B, 2B)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Mask out self-contrast (diagonal)
    logits_mask = torch.scatter(
        torch.ones_like(logits),
        1,
        torch.arange(2 * B, device=device).view(-1, 1),
        0
    )
    
    # Compute ground truth mask for positive pairs (same label, excluding self)
    labels_double = labels_double.view(-1, 1)
    mask = torch.eq(labels_double, labels_double.T).float()
    mask = mask * logits_mask
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
        
    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean of log-likelihood over positive pairs
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
            model = GDNLayer(
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

    def save_checkpoint(self, checkpoint_dir: str, round_idx: int, is_best: bool = False):
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
        }

        if is_best:
            save_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            try:
                torch.save(checkpoint, save_path)
                logger.info(f"Best checkpoint saved: {save_path}")
            except Exception as e:
                logger.error(f"Failed to save best checkpoint: {e}")
        else:
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
        num_rounds: int = 20,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1,
        start_round: int = 0,
        num_samples: int = 5,
        oversample_scale: float = 2.0,
        focal_loss_alpha: float = 0.5,
        use_bce_loss: bool = True,
        use_oversampling: bool = False,
        two_speed_lr: bool = True,
        lr_server: float = 0.0003,
        lr_client: float = 0.0005,
        enable_client_attention: bool = False,
        use_contrastive: bool = True,
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 0.3,
        normalize_vfl_gradients: bool = False,
        vfl_target_norm: float = 1.0,
        batch_size: int = 32,
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

        logger.info(
            f"Starting joint federated VFL training from round {start_round + 1} to {num_rounds} "
            f"with neighbor sampling num_samples={num_samples}, oversample_scale={oversample_scale}, "
            f"focal_loss_alpha={focal_loss_alpha}, use_bce_loss={use_bce_loss}, "
            f"use_oversampling={use_oversampling}, two_speed_lr={two_speed_lr}, "
            f"enable_client_attention={enable_client_attention}, use_contrastive={use_contrastive}, "
            f"normalize_vfl_gradients={normalize_vfl_gradients}, early_stopping_patience={early_stopping_patience}"
        )

        # Initialize early stopping tracking variables
        best_loss = float("inf")
        best_round = -1
        no_improvement_count = 0
        best_global_state = None
        best_client_states = {}

        # Load existing best checkpoint if it exists from disk
        if checkpoint_dir:
            best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_checkpoint_path):
                try:
                    best_checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
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

        if start_round > 0 and len(self.results.get("training_losses", [])) > 0:
            # Reconstruct the historical best loss and how many rounds since it happened
            history = self.results["training_losses"][:start_round]
            if len(history) > 0:
                best_loss = min(history)
                best_round = history.index(best_loss)
                no_improvement_count = start_round - 1 - best_round
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
            client_data_list.append(data)

        num_snapshots = client_data_list[0]["features"].shape[0]
        logger.info(f"Loaded training data. Number of aligned snapshots: {num_snapshots}")

        # Dynamic positive class weight calculation for BCEWithLogitsLoss
        train_labels = client_data_list[0]["graph_labels"]
        num_normal = (train_labels == 0).sum().item()
        num_anomalous = (train_labels == 1).sum().item()
        if num_anomalous > 0:
            pos_weight_val = num_normal / num_anomalous
        else:
            pos_weight_val = 2.11
        logger.info(f"Dynamic positive weight calculated: {pos_weight_val:.4f} (Normal: {num_normal}, Anomalous: {num_anomalous})")

        # Set up loss criteria
        if use_bce_loss:
            pos_weight = torch.tensor([1.0, pos_weight_val], device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info("Using standard BCE Loss with positive weight.")
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

        # 2. Build joint parameter list and optimizer
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
            optimizer = torch.optim.Adam(all_params, lr=lr_client)
            logger.info(f"Using single speed learning rate: {lr_client}")

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

        num_steps = max(1, (num_snapshots + batch_size - 1) // batch_size)

        for round_idx in range(start_round, num_rounds):
            round_start = time.time()
            logger.info(f"Starting round {round_idx + 1}/{num_rounds}")

            self.global_model.train()
            for client_model in self.client_models.values():
                client_model.train()

            round_loss = 0.0
            # Shuffle indices at the start of each round
            indices = torch.randperm(num_snapshots)

            # Initialize accumulation buffers for step logging
            step_count_in_interval = 0
            correct_preds_in_interval = 0
            clf_loss_in_interval = 0.0
            supcon_loss_in_interval = 0.0
            client_norms_in_interval = [0.0] * self.num_clients
            server_emb_norm_in_interval = 0.0

            for step in range(num_steps):
                batch_indices = indices[step * batch_size : (step + 1) * batch_size]
                if len(batch_indices) == 0:
                    break

                # Calculate warmup scaling factor lambda_t
                if use_contrastive:
                    progress = round_idx + (step / num_steps)
                    if progress < 0.5:
                        lambda_t = 0.0
                    elif progress <= 3.0:
                        # Scale up over the next two rounds (from progress=0.5 to progress=3.0)
                        lambda_t = contrastive_weight * (progress - 0.5) / 2.5
                    else:
                        lambda_t = contrastive_weight
                else:
                    lambda_t = 0.0

                optimizer.zero_grad()
                
                # Setup structures to capture boundary gradients
                vfl_gradients1 = {c: [] for c in range(self.num_clients)}
                vfl_gradients2 = {c: [] for c in range(self.num_clients)}

                def make_grad_hook(client_idx, norm_list, normalize, target_norm):
                    def hook(grad):
                        if grad is not None:
                            grad_norm = grad.norm(2).item()
                            norm_list[client_idx].append(grad_norm)
                            if normalize:
                                return grad / (grad_norm + 1e-8) * target_norm
                        return grad
                    return hook

                g_embs_1 = []
                g_embs_2 = []
                batch_labels = []
                clf_losses = []
                attn_weights_list = []

                step_count_in_interval += 1

                with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=actual_use_amp):
                    for idx in batch_indices:
                        # Get raw features for all nodes at this snapshot to compute anomaly scores
                        raw_features_list = []
                        for c in range(self.num_clients):
                            snapshot_features = client_data_list[c]["features"][idx].view(-1, 1).to(self.device)
                            raw_features_list.append(snapshot_features)
                        raw_features_global = torch.cat(raw_features_list, dim=0).squeeze(-1)  # (total_nodes,)

                        # Compute z-score deviation from precomputed normal baseline, squashed to [0, 1) using tanh
                        z_scores = torch.abs(raw_features_global - self.normal_means_global) / self.normal_stds_global
                        node_anomaly_scores = torch.tanh(z_scores)

                        # Get label for this snapshot
                        label = client_data_list[0]["graph_labels"][idx].unsqueeze(0).to(self.device)

                        # Only apply minority oversampling on anomalous snapshots (label == 1)
                        node_anomaly_scores = node_anomaly_scores * label.item()

                        # Snapshot forward pass 1
                        h_client_list1 = []
                        for c in range(self.num_clients):
                            snapshot_features = client_data_list[c]["features"][idx].view(-1, 1).to(self.device)
                            h_c1 = self.client_models[c](snapshot_features)
                            
                            # Register hook to monitor and normalize gradients
                            if h_c1.requires_grad:
                                h_c1.register_hook(make_grad_hook(c, vfl_gradients1, normalize_vfl_gradients, vfl_target_norm))
                                
                            h_client_list1.append(h_c1)

                            # Accumulate client GAT output representation norm
                            client_norms_in_interval[c] += h_c1.norm(2).item()
                        
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
                        server_emb_norm_in_interval += emb1.norm(2).item()

                        # Accumulate correct predictions for batch accuracy
                        pred_class = predictions1.argmax(dim=1).item()
                        if pred_class == label.item():
                            correct_preds_in_interval += 1

                        # Snapshot forward pass 2 for contrastive loss (GAT has dropout, so this produces a different view)
                        h_client_list2 = []
                        for c in range(self.num_clients):
                            snapshot_features = client_data_list[c]["features"][idx].view(-1, 1).to(self.device)
                            h_c2 = self.client_models[c](snapshot_features)
                            
                            if h_c2.requires_grad:
                                h_c2.register_hook(make_grad_hook(c, vfl_gradients2, normalize_vfl_gradients, vfl_target_norm))
                                
                            h_client_list2.append(h_c2)

                        if enable_client_attention:
                            h_global2, _ = self.global_model.client_attention(h_client_list2)
                        else:
                            h_global2 = torch.cat(h_client_list2, dim=0)

                        edge_index2 = self._build_global_graph(h_global2, self.topk)

                        if use_oversampling:
                            emb2, predictions2, _, graph_contrastive_emb2 = self.global_model(
                                h_global2,
                                edge_index2,
                                node_anomaly_scores=node_anomaly_scores,
                                num_samples=num_samples,
                                oversample_scale=oversample_scale,
                            )
                        else:
                            emb2, predictions2, _, graph_contrastive_emb2 = self.global_model(
                                h_global2,
                                edge_index2,
                                node_anomaly_scores=None,
                                num_samples=None,
                            )

                        # Compute classification loss
                        if use_bce_loss:
                            target_one_hot = F.one_hot(label, num_classes=2).float()
                            clf_loss = criterion(predictions1, target_one_hot)
                        else:
                            alpha = torch.tensor([1.0 - focal_loss_alpha, focal_loss_alpha], device=self.device)
                            clf_loss = focal_loss(predictions1, label, alpha=alpha, gamma=2.0)

                        clf_losses.append(clf_loss)
                        clf_loss_in_interval += clf_loss.item()

                        # Keep projected pooled representations for supervised contrastive loss
                        g_embs_1.append(graph_contrastive_emb1)
                        g_embs_2.append(graph_contrastive_emb2)
                        batch_labels.append(label)

                    # Compute step loss
                    mean_clf_loss = torch.stack(clf_losses).mean()

                    if use_contrastive and len(g_embs_1) > 0:
                        z1 = torch.cat(g_embs_1, dim=0)
                        z2 = torch.cat(g_embs_2, dim=0)
                        labels_tensor = torch.cat(batch_labels, dim=0)
                        supcon_loss = supervised_contrastive_loss(z1, z2, labels_tensor, temperature=contrastive_temp)
                        step_loss = mean_clf_loss + lambda_t * supcon_loss
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
                        for name, param in client_model.gat.named_parameters():
                            if param.grad is not None:
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
                    num_snapshots_in_interval = step_count_in_interval * len(batch_indices)
                    if num_snapshots_in_interval > 0:
                        avg_clf_loss = clf_loss_in_interval / num_snapshots_in_interval
                        avg_supcon_loss = supcon_loss_in_interval / step_count_in_interval
                        avg_batch_acc = correct_preds_in_interval / num_snapshots_in_interval
                        avg_client_norms = [n / num_snapshots_in_interval for n in client_norms_in_interval]
                        avg_server_norm = server_emb_norm_in_interval / num_snapshots_in_interval
                        
                        loss_str = f"Loss: {step_loss.item():.4f} (Clf: {avg_clf_loss:.4f}"
                        if use_contrastive:
                            loss_str += f", Contrastive: {avg_supcon_loss:.4f}, lambda: {lambda_t:.4f})"
                        else:
                            loss_str += ")"
                            
                        client_norms_str = ", ".join([f"Client {c+1}: {norm:.4f}" for c, norm in enumerate(avg_client_norms)])
                        
                        logger.info(
                            f"  [Round {round_idx + 1} | Step {step + 1}/{num_steps}] "
                            f"{loss_str} | Batch Acc: {avg_batch_acc * 100:.2f}% | "
                            f"Server norm: {avg_server_norm:.4f} | Client norms: {client_norms_str}"
                        )
                        
                    # Reset buffers for the next interval
                    step_count_in_interval = 0
                    correct_preds_in_interval = 0
                    clf_loss_in_interval = 0.0
                    supcon_loss_in_interval = 0.0
                    client_norms_in_interval = [0.0] * self.num_clients
                    server_emb_norm_in_interval = 0.0

            avg_round_loss = round_loss / num_steps
            round_time = time.time() - round_start
            self.results["training_losses"].append(avg_round_loss)
            self.results["round_times"].append(round_time)

            logger.info(
                f"Round {round_idx + 1} completed in {round_time:.2f}s, loss: {avg_round_loss:.4f}"
            )

            # Step the learning rate scheduler based on the average round loss
            old_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler.step(avg_round_loss)
            new_lrs = [group["lr"] for group in optimizer.param_groups]
            for group_idx, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                if old_lr != new_lr:
                    group_name = "Server" if (two_speed_lr and group_idx == 0) else "Client" if (two_speed_lr and group_idx == 1) else "All layers"
                    logger.info(f"Learning rate for {group_name} updated mid-training: {old_lr:.6f} -> {new_lr:.6f}")

            # Check early stopping / best loss improvement
            if avg_round_loss < best_loss:
                best_loss = avg_round_loss
                best_round = round_idx
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
                logger.info(
                    f"Loss did not improve. Current best loss: {best_loss:.4f} (from round {best_round + 1}). "
                    f"Rounds without improvement: {no_improvement_count}/{early_stopping_patience}"
                )

            # Regular checkpointing
            if checkpoint_dir and (
                (round_idx - start_round + 1) % checkpoint_every == 0
                or round_idx == num_rounds - 1
            ):
                self.save_checkpoint(checkpoint_dir, round_idx)

            if no_improvement_count >= early_stopping_patience:
                logger.info(
                    f"🛑 Early stopping triggered after {round_idx + 1} rounds. "
                    f"No improvement for {early_stopping_patience} consecutive rounds. "
                    f"Best round was round {best_round + 1} with loss {best_loss:.4f}."
                )
                break

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
                self.load_checkpoint(best_checkpoint_path)

        logger.info("Joint federated VFL training completed")
        return self.results

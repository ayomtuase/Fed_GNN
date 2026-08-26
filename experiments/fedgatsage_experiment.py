"""Main experiment script for FedGATSage.
Demonstrates the complete pipeline from data loading to evaluation.
"""

import argparse
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from federated_learning import FedGATSageSystem, build_sliding_windows
from utils import (
    ExperimentTracker,
    calculate_metrics,
    load_dataset_info,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_progress,
    set_random_seeds,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FedGATSage Experiment")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/preprocessed_data",
        help="Path to dataset directory (default: data/preprocessed_data)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to raw input CSV file (if data_dir is not prepared)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="swat",
        help="Dataset to use (default: swat)",
    )
    parser.add_argument(
        "--num_clients", type=int, default=5, help="Number of federated clients"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Number of federation rounds (default: None for indefinite training)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/mps/cpu/auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Data type for dataset tensors and model parameters (float32/float64)",
    )
    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP) training",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--demo_mode", action="store_true", help="Run in demo mode (reduced complexity)"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Force run data preprocessing"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Relative directory under output_dir to save checkpoint files",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Save checkpoint every N federation rounds",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of neighbors to sample in GraphSAGE (default: 5)",
    )
    parser.add_argument(
        "--oversample_scale",
        type=float,
        default=2.0,
        help="Oversampling scale factor for anomalous nodes (default: 2.0)",
    )
    parser.add_argument(
        "--focal_loss_alpha",
        type=float,
        default=0.5,
        help="Focal Loss class weight for anomalous class (default: 0.5)",
    )
    parser.add_argument(
        "--disable_ce_loss",
        action="store_true",
        help="Disable Cross-Entropy Loss and use Focal Loss instead",
    )
    parser.add_argument(
        "--enable_oversampling",
        action="store_true",
        help="Enable minority oversampling in GraphSAGE neighbor sampling",
    )
    parser.add_argument(
        "--disable_two_speed_lr",
        action="store_true",
        help="Disable two-speed learning rate",
    )
    parser.add_argument(
        "--lr_server",
        type=float,
        default=0.001,
        help="Learning rate for server-side layers (default: 0.001)",
    )
    parser.add_argument(
        "--lr_client",
        type=float,
        default=0.005,
        help="Learning rate for client-side layers (default: 0.005)",
    )
    parser.add_argument(
        "--enable_client_attention",
        action="store_true",
        help="Enable attention weights on the concatenation step on the server",
    )
    parser.add_argument(
        "--disable_contrastive",
        dest="enable_contrastive",
        action="store_false",
        help="Disable supervised contrastive loss on server-side",
    )
    parser.add_argument(
        "--enable_contrastive",
        dest="enable_contrastive",
        action="store_true",
        help="Enable supervised contrastive loss on server-side",
    )
    parser.set_defaults(enable_contrastive=True)
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=2,
        help="Patience for ReduceLROnPlateau learning rate scheduler (default: 2)",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
        help="Decay factor for learning rate scheduler (default: 0.5)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for scheduler (default: 1e-6)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping based on validation AUC ROC (default: 10)",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=15,
        help="Kernel size for 1D convolution in client GATLayer (default: 15)",
    )
    parser.add_argument(
        "--client_topk",
        type=float,
        default=0.3,
        help="Number of neighbors to connect (if >= 1) or fraction of total sensors (if < 1) in client GATLayer (default: 0.3)",
    )
    parser.add_argument(
        "--global_topk",
        type=int,
        default=7,
        help="Number of neighbors to connect for each node/sensor in server global graph (default: 7)",
    )
    parser.add_argument(
        "--log_step_every",
        type=int,
        default=25,
        help="Frequency of detailed step progress logging inside a round (default: 25)",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.1,
        help="Weight for supervised contrastive loss (default: 1.0)",
    )
    parser.add_argument(
        "--contrastive_temp",
        type=float,
        default=0.07,
        help="Temperature for supervised contrastive loss (default: 0.07)",
    )
    parser.add_argument(
        "--disable_concat_skip",
        action="store_true",
        help="Disable concatenation skip connections in client GAT and server GraphSAGE",
    )
    parser.add_argument(
        "--enable_vfl_normalization",
        action="store_true",
        help="Enable normalization of gradients transmitted across the VFL boundary",
    )
    parser.add_argument(
        "--vfl_target_norm",
        type=float,
        default=1.0,
        help="Target L2 norm for VFL boundary gradient normalization (default: 1.0)",
    )
    parser.add_argument(
        "--disable_dp",
        dest="enable_dp",
        action="store_false",
        help="Disable Differential Privacy (DP) on client embeddings",
    )
    parser.add_argument(
        "--enable_dp",
        dest="enable_dp",
        action="store_true",
        help="Enable Differential Privacy (DP) on client embeddings",
    )
    parser.set_defaults(enable_dp=False)
    parser.add_argument(
        "--dp_clip_bound",
        type=float,
        default=1.0,
        help="Clipping bound C for client embeddings (default: 1.0)",
    )
    parser.add_argument(
        "--dp_noise_multiplier",
        type=float,
        default=0.1,
        help="Noise multiplier sigma for client embedding DP (default: 0.1)",
    )
    parser.add_argument(
        "--dp_profile",
        action="store_true",
        help="Run in DP profiling mode to log unclipped client embedding norms and recommend C",
    )
    parser.add_argument(
        "--dp_profile_rounds",
        type=int,
        default=3,
        help="Number of communication rounds for DP profiling (default: 3)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for federated training (default: 1024)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=120,
        help="Sliding window size (default: 120)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader (default: 4)",
    )

    argv = [arg for arg in sys.argv[1:] if arg != "\\"]
    if len(argv) != len(sys.argv[1:]):
        logger.warning(
            "Removed stray line-continuation backslash from command line arguments"
        )

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    return args


def check_and_preprocess_data(args):
    data_ready = True
    train_dir = os.path.join(args.data_dir, "train")
    if not (os.path.exists(args.data_dir) and 
            os.path.exists(os.path.join(args.data_dir, "train_labels.npy")) and 
            os.path.exists(os.path.join(train_dir, "client_1.npy"))):
        data_ready = False

    if args.preprocess or not data_ready:
        logger.info(
            "Preprocessed data directory not ready or preprocessing requested. Running preprocessing..."
        )

        input_file = args.input_file
        parent_dir = os.path.dirname(args.data_dir)
        if not input_file:
            potential_files = [os.path.join(parent_dir, "swat.csv")]
            if os.path.exists(potential_files[0]):
                input_file = potential_files[0]
                logger.info(f"Auto-detected input file: {input_file}")
            else:
                input_file = os.path.join(parent_dir, "dummy_data.csv")
                logger.warning(
                    f"No input file specified. Will generate dummy data at {input_file}"
                )

        cmd = [
            sys.executable,
            "preprocess_data.py",
            "--input_file",
            input_file,
            "--output_dir",
            args.data_dir,
            "--seed",
            str(args.seed),
        ]

        try:
            subprocess.check_call(cmd)
            logger.info("Preprocessing completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Preprocessing failed: {e}")
            sys.exit(1)


def setup_experiment(args):
    log_file = os.path.join(args.output_dir, "experiment.log")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_level, log_file)

    set_random_seeds(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")
    logger.info(f"Experiment arguments: {vars(args)}")

    check_and_preprocess_data(args)

    return device


def run_federated_experiment(args, device: str) -> dict:
    logger.info("Starting FedGATSage federated learning experiment")

    rounds_str = f"{args.num_rounds}rounds" if args.num_rounds is not None else "indefinite"
    experiment_name = (
        f"fedgatsage_{args.dataset}_{args.num_clients}clients_{rounds_str}"
    )
    tracker = ExperimentTracker(experiment_name, args.output_dir)
    tracker.start_experiment()

    dataset_info = load_dataset_info(args.data_dir)
    logger.info(f"Dataset info: {dataset_info}")

    if args.demo_mode:
        args.num_rounds = min(args.num_rounds, 20) if args.num_rounds is not None else 20
        logger.info("Running in demo mode with reduced rounds")

    checkpoint_dir = args.checkpoint_dir
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(args.output_dir, checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Auto-detect number of clients and dimensions from numpy arrays
    import glob
    train_dir = os.path.join(args.data_dir, "train")
    client_files = sorted(glob.glob(os.path.join(train_dir, "client_*.npy")))
    
    if len(client_files) > 0:
        args.num_clients = len(client_files)
        client_node_nums = []
        input_dim = None
        for c_file in client_files:
            shape = np.load(c_file, mmap_mode='r').shape
            client_node_nums.append(shape[1])
            if input_dim is None:
                input_dim = shape[2]
        logger.info(f"Auto-detected {args.num_clients} clients from preprocessed folder.")
    else:
        logger.warning("No preprocessed client files found. Using fallback defaults.")
        client_node_nums = [10] * args.num_clients
        input_dim = args.window_size

    fed_system = FedGATSageSystem(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        device=device,
        checkpoint_dir=checkpoint_dir,
        dtype=args.dtype,
    )

    num_classes = 2
    fed_system.label_mapper = {"Normal": 0, "Attack": 1}

    logger.info(
        f"Model configuration: client_node_nums={client_node_nums}, input_dim={input_dim}, num_classes={num_classes}"
    )

    resume_round = -1
    if args.resume_checkpoint or os.path.exists(checkpoint_dir):
        resume_round = fed_system.load_checkpoint(args.resume_checkpoint)

    if resume_round < 0:
        fed_system.initialize_models(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=num_classes,
            client_topk=args.client_topk,
            global_topk=args.global_topk,
            client_node_nums=client_node_nums,
            use_concat_skip=not args.disable_concat_skip,
            kernel_size=args.kernel_size,
        )
    else:
        input_dim = fed_system.input_dim or input_dim
        num_classes = fed_system.num_classes or num_classes
        logger.info(
            f"Resumed from checkpoint. Starting training at round {resume_round + 1}. "
            f"Using checkpoint model dimensions: input_dim={input_dim}, num_classes={num_classes}"
        )

    # Determine DP settings based on profile mode
    dp_enabled = args.enable_dp
    normalize_vfl = args.enable_vfl_normalization
    num_rounds_to_train = args.num_rounds

    if args.dp_profile:
        logger.info(f"Running in DP profiling mode for {args.dp_profile_rounds} rounds.")
        num_rounds_to_train = args.dp_profile_rounds
        dp_enabled = False  # No DP clipping or noise during profiling
        normalize_vfl = False  # No normal VFL normalization either

    if num_rounds_to_train is not None and (resume_round + 1) >= num_rounds_to_train:
        logger.info(
            "Checkpoint indicates training already completed. Skipping federated training."
        )
        training_results = fed_system.results
        
        # Load best checkpoint weights for evaluation since we skipped training
        best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        if os.path.exists(best_checkpoint_path):
            logger.info(f"Loading best checkpoint weights for evaluation: {best_checkpoint_path}")
            fed_system.load_checkpoint(best_checkpoint_path, load_training_state=False)
        else:
            logger.warning("No best checkpoint found. Evaluating with latest checkpoint weights.")
    else:
        try:
            training_results = fed_system.train_federated(
                num_rounds=num_rounds_to_train,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=args.checkpoint_every,
                start_round=resume_round + 1 if resume_round >= 0 else 0,
                num_samples=args.num_samples,
                oversample_scale=args.oversample_scale,
                focal_loss_alpha=args.focal_loss_alpha,
                use_ce_loss=not args.disable_ce_loss,
                use_oversampling=args.enable_oversampling,
                two_speed_lr=not args.disable_two_speed_lr,
                lr_server=args.lr_server,
                lr_client=args.lr_client,
                enable_client_attention=args.enable_client_attention,
                use_contrastive=args.enable_contrastive,
                contrastive_weight=args.contrastive_weight,
                contrastive_temp=args.contrastive_temp,
                normalize_vfl_gradients=normalize_vfl,
                vfl_target_norm=args.vfl_target_norm,
                use_amp=not args.disable_amp,
                max_samples=200 if args.demo_mode else None,
                lr_scheduler_patience=args.lr_patience,
                lr_scheduler_factor=args.lr_factor,
                min_lr=args.min_lr,
                log_step_every=args.log_step_every,
                early_stopping_patience=args.early_stopping_patience,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dp_enabled=dp_enabled,
                dp_clip_bound=args.dp_clip_bound,
                dp_noise_multiplier=args.dp_noise_multiplier,
            )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (KeyboardInterrupt). Gracefully transitioning to final evaluation...")
            training_results = fed_system.results
            
            # Load best checkpoint weights if available on disk
            best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_checkpoint_path):
                logger.info(f"Loading best checkpoint weights for final evaluation: {best_checkpoint_path}")
                fed_system.load_checkpoint(best_checkpoint_path, load_training_state=False)
            else:
                logger.warning("No best checkpoint found on disk. Evaluating with current model weights.")

    # Process tracked unclipped norms if any were collected
    if hasattr(fed_system, "unclipped_norms_tracker") and any(len(lst) > 0 for lst in fed_system.unclipped_norms_tracker):
        process_and_plot_embedding_norms(fed_system.unclipped_norms_tracker, args.output_dir)

    if args.dp_profile:
        logger.info("DP profiling completed successfully. Optimal clipping bound recommendation printed above.")
        return {
            "training": training_results,
            "evaluation": {},
            "configuration": {
                "num_clients": args.num_clients,
                "num_rounds": num_rounds_to_train,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "dp_profile_mode": True,
            },
        }

    evaluation_results = evaluate_system(fed_system, args)

    final_results = {
        "training": training_results,
        "evaluation": evaluation_results,
        "configuration": {
            "num_clients": args.num_clients,
            "num_rounds": num_rounds_to_train,
            "input_dim": input_dim,
            "num_classes": num_classes,
        },
    }

    if evaluation_results:
        metrics_to_log = {
            "final_accuracy": evaluation_results.get("accuracy", 0.0),
            "final_f1": evaluation_results.get("macro_f1", 0.0),
        }
        if "roc_auc" in evaluation_results and evaluation_results["roc_auc"] is not None:
            metrics_to_log["final_roc_auc"] = evaluation_results["roc_auc"]
        tracker.log_round_metrics(
            len(training_results.get("training_losses", [])),
            metrics_to_log,
        )


    tracker.save_experiment(final_results)
    return final_results


def plot_latent_space_tsne(embeddings: np.ndarray, labels: np.ndarray, save_path: str, client_idx: Optional[int] = None):
    """Generate a t-SNE plot of the latent space embeddings."""
    from sklearn.manifold import TSNE
    logger.info(f"Running t-SNE on {embeddings.shape[0]} embeddings...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, embeddings.shape[0] // 10)))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = ['#1f77b4', '#d62728'] # Classic blue (normal) and red (anomaly)
        class_names = ['Normal', 'Anomaly']
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=class_names[int(label)] if int(label) < len(class_names) else f"Class {label}",
                alpha=0.6,
                c=colors[i % len(colors)],
                edgecolors='w',
                linewidths=0.5,
                s=30
            )
        
        if client_idx is not None:
            plt.title(f"t-SNE Visualization of Latent Space (Client {client_idx})", fontsize=14, fontweight='bold')
        else:
            plt.title("t-SNE Visualization of Latent Space (H_global)", fontsize=14, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"t-SNE latent space plot successfully saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate t-SNE plot: {e}")


def evaluate_system(fed_system: FedGATSageSystem, args) -> dict:
    logger.info("Evaluating trained federated system")

    try:
        from federated_learning import FederatedDataset
        from torch.utils.data import DataLoader

        test_labels_path = os.path.join(args.data_dir, "test_labels.npy")
        if not os.path.exists(test_labels_path):
            logger.warning(f"No test labels found at {test_labels_path}")
            return {}

        test_client_paths = [
            os.path.join(args.data_dir, "test", f"client_{c+1}.npy")
            for c in range(fed_system.num_clients)
        ]

        # Use demo mode limits if specified
        max_samples = 1000 if args.demo_mode else None
        test_dataset = FederatedDataset(test_client_paths, test_labels_path, max_samples=max_samples, dtype=fed_system.dtype)
        
        batch_size = getattr(args, "batch_size", 1024)
        # Determine if the active device uses discrete VRAM
        is_discrete_gpu = torch.device(fed_system.device).type == "cuda"

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=is_discrete_gpu,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0)
        )

        num_test_samples = len(test_dataset)

        # Retrieve client feature column mappings / sensor names if possible
        global_node_names = []
        names_found = False
        
        # Check parent folder and swat.csv / SWaT_Dataset_Normal_v0.xlsx
        parent_dir = os.path.dirname(args.data_dir)
        swat_path = os.path.join(parent_dir, "swat.csv")
        if not os.path.exists(swat_path):
            swat_path = os.path.join(args.data_dir, "swat.csv")

        swat_xlsx_path = os.path.join(parent_dir, "SWaT_Dataset_Normal_v0.xlsx")
        if not os.path.exists(swat_xlsx_path):
            swat_xlsx_path = os.path.join(args.data_dir, "SWaT_Dataset_Normal_v0.xlsx")

        if os.path.exists(swat_path):
            try:
                df_header = pd.read_csv(swat_path, nrows=0)
                client_cols = {stage: [] for stage in range(1, 7)}
                for col in df_header.columns:
                    col = col.strip()
                    if col in ["Timestamp", "Normal/Attack"]:
                        continue
                    import re
                    match = re.search(r'\d+', col)
                    if match:
                        stage = int(match.group()[0])
                        if 1 <= stage <= 6:
                            client_cols[stage].append(col)
                for c in range(fed_system.num_clients):
                    global_node_names.extend(client_cols.get(c + 1, []))
                if len(global_node_names) == sum(fed_system.client_node_nums):
                    names_found = True
            except Exception as e:
                logger.warning(f"Could not extract sensor names from swat.csv: {e}")
        elif os.path.exists(swat_xlsx_path):
            try:
                logger.info(f"Extracting sensor names from Excel file: {swat_xlsx_path}")
                df_header = pd.read_excel(swat_xlsx_path, header=1, nrows=0)
                client_cols = {stage: [] for stage in range(1, 7)}
                for col in df_header.columns:
                    col = col.strip()
                    if col in ["Timestamp", "Normal/Attack"]:
                        continue
                    import re
                    match = re.search(r'\d+', col)
                    if match:
                        stage = int(match.group()[0])
                        if 1 <= stage <= 6:
                            client_cols[stage].append(col)
                for c in range(fed_system.num_clients):
                    global_node_names.extend(client_cols.get(c + 1, []))
                if len(global_node_names) == sum(fed_system.client_node_nums):
                    names_found = True
            except Exception as e:
                logger.warning(f"Could not extract sensor names from {swat_xlsx_path}: {e}")

        if not names_found:
            try:
                for c in range(fed_system.num_clients):
                    client_csv = os.path.join(parent_dir, f"client_{c+1}.csv")
                    if not os.path.exists(client_csv):
                        client_csv = os.path.join(args.data_dir, f"client_{c+1}.csv")
                    if os.path.exists(client_csv):
                        df_c = pd.read_csv(client_csv, nrows=0)
                        cols = [col for col in df_c.columns if col not in ["attack", "Normal/Attack", "Timestamp"]]
                        global_node_names.extend(cols)
                if len(global_node_names) == sum(fed_system.client_node_nums):
                    names_found = True
            except Exception as e:
                logger.warning(f"Could not extract sensor names from client CSVs: {e}")

        if not names_found:
            logger.info("Using generic sensor names")
            global_node_names = []
            for c in range(fed_system.num_clients):
                global_node_names.extend([f"c{c+1}_s{i}" for i in range(fed_system.client_node_nums[c])])

        # Set models to eval mode
        fed_system.global_model.eval()
        for client_model in fed_system.client_models.values():
            client_model.eval()

        predicted = []
        predicted_probs = []
        labels_list = []
        
        # Collect latent space embeddings for t-SNE plot
        latent_embeddings = []
        client_latent_embeddings = {c: [] for c in range(fed_system.num_clients)}
        latent_labels = []
        tsne_max_samples = 2000
        sample_interval = max(1, num_test_samples // tsne_max_samples)

        anomaly_counter = 0
        culprit_counts = {}
        N_global = sum(fed_system.client_node_nums)
        best_threshold = getattr(fed_system, "best_threshold", 0.5)
        logger.info(f"Using decision threshold: {best_threshold:.4f} for test set evaluation")

        with torch.no_grad():
            logger.info(f"Running VFL evaluation over {num_test_samples} test snapshots with batch_size={batch_size}")
            
            for step, (batch_features, batch_labels) in enumerate(test_loader):
                B = batch_labels.shape[0]
                start_idx = step * batch_size

                # Move to device
                batch_features = [f.to(fed_system.device, non_blocking=True) for f in batch_features]
                batch_labels = batch_labels.to(fed_system.device, non_blocking=True)

                h_client_list = []
                for c in range(fed_system.num_clients):
                    x_c = batch_features[c].view(B * fed_system.client_node_nums[c], -1)
                    h_c = fed_system.client_models[c](x_c)
                    h_client_list.append(h_c)

                # Aggregate with client attention on the server if enabled
                if args.enable_client_attention:
                    h_global, _ = fed_system.global_model.client_attention(h_client_list, fed_system.client_node_nums)
                else:
                    h_global_batched = torch.cat([hc.view(B, Nc, -1) for hc, Nc in zip(h_client_list, fed_system.client_node_nums)], dim=1)
                    h_global = h_global_batched.view(B * N_global, -1)

                edge_index = fed_system._build_global_graph(h_global, fed_system.global_topk)
                embeddings, predictions, node_weights, _ = fed_system.global_model(
                    h_global,
                    edge_index,
                    num_nodes_per_graph=N_global
                )

                # Binary classification (BCE logits > threshold)
                pred_probs = torch.sigmoid(predictions.squeeze(-1)).cpu().numpy()
                pred_classes = (pred_probs >= best_threshold).astype(int)

                # Reshape node weights to (B, N_global)
                node_weights_reshaped = node_weights.view(B, N_global).cpu().numpy()

                # Reshape embeddings to (B, N_global, -1) to compute graph embeddings
                graph_embs_batch = embeddings.view(B, N_global, -1).mean(dim=1).cpu().numpy()

                for local_idx in range(B):
                    global_idx = start_idx + local_idx
                    pred_class = int(pred_classes[local_idx])
                    predicted.append(pred_class)
                    predicted_probs.append(float(pred_probs[local_idx]))
                    labels_list.append(int(batch_labels[local_idx].item()))

                    # Identify culprit nodes if anomalous
                    if pred_class == 1:
                        anomaly_counter += 1
                        weights = node_weights_reshaped[local_idx]
                        top_k = min(3, len(weights))
                        suspicious_indices = weights.argsort()[-top_k:][::-1]
                        top_culprits = [(i, global_node_names[i], weights[i]) for i in suspicious_indices]

                        for _, name, _ in top_culprits:
                            culprit_counts[name] = culprit_counts.get(name, 0) + 1

                        if anomaly_counter <= 5:
                            logger.info(f"🚨 SYSTEM ANOMALY DETECTED at snapshot {global_idx}!")
                            for rank, (node_idx, name, weight) in enumerate(top_culprits, 1):
                                logger.info(f"  - Culprit {rank}: Sensor {node_idx} (name: '{name}', Weight: {weight:.4f})")

                    if global_idx % sample_interval == 0:
                        latent_embeddings.append(graph_embs_batch[local_idx])
                        latent_labels.append(int(batch_labels[local_idx].item()))

                        for c in range(fed_system.num_clients):
                            # extract client embedding mean for client c
                            h_c_reshaped = h_client_list[c].view(B, fed_system.client_node_nums[c], -1)
                            client_emb = h_c_reshaped[local_idx].mean(dim=0).cpu().numpy()
                            client_latent_embeddings[c].append(client_emb)

                if (start_idx + B) % 10240 == 0 or (start_idx + B) == num_test_samples:
                    logger.info(f"Evaluated {start_idx + B}/{num_test_samples} snapshots")

            y_true = np.array(labels_list)
            y_pred = np.array(predicted)
            y_prob = np.array(predicted_probs)

            class_names = ["Normal", "Anomaly"]

            metrics = calculate_metrics(y_true, y_pred, class_names, y_prob=y_prob)
            cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
            plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

            roc_path = os.path.join(args.output_dir, "roc_curve.png")
            plot_roc_curve(y_true, y_prob, roc_path)

            # Generate and save t-SNE plot of the latent space
            if len(latent_embeddings) > 0:
                tsne_path = os.path.join(args.output_dir, "latent_space_tsne.png")
                plot_latent_space_tsne(np.array(latent_embeddings), np.array(latent_labels), tsne_path)

            # Generate and save individual client t-SNE plots
            for c in range(fed_system.num_clients):
                if len(client_latent_embeddings[c]) > 0:
                    c_tsne_path = os.path.join(args.output_dir, f"latent_space_tsne_client_{c+1}.png")
                    plot_latent_space_tsne(
                        np.array(client_latent_embeddings[c]),
                        np.array(latent_labels),
                        c_tsne_path,
                        client_idx=c+1
                    )

            if anomaly_counter > 0:
                logger.info("==========================================")
                logger.info(f"📊 SUMMARY OF ANOMALOUS NODES DETECTED")
                logger.info(f"Total Anomalous Snapshots: {anomaly_counter} / {num_test_samples}")
                logger.info("Most Frequently Flagged Culprit Sensors:")
                sorted_culprits = sorted(culprit_counts.items(), key=lambda item: item[1], reverse=True)
                for rank, (name, count) in enumerate(sorted_culprits[:10], 1):
                    percentage = (count / anomaly_counter) * 100
                    logger.info(f"  {rank}. Sensor: '{name}' -> Flagged {count} times ({percentage:.1f}% of anomalies)")
                logger.info("==========================================")

            # Create a detailed evaluation report block for logs and saving
            report_lines = [
                "==================================================================================",
                "📊 FINAL EVALUATION METRICS ON TEST DATASET",
                "==================================================================================",
                f"  - Decision Threshold Used:  {best_threshold:.4f}",
                f"  - Accuracy:                 {metrics['accuracy'] * 100:.2f}%",
                f"  - Balanced Accuracy:        {metrics['balanced_accuracy'] * 100:.2f}%",
                f"  - Macro F1 Score:           {metrics['macro_f1'] * 100:.2f}%",
                f"  - Weighted F1 Score:        {metrics['weighted_f1'] * 100:.2f}%",
            ]
            if metrics.get('roc_auc') is not None:
                report_lines.append(f"  - ROC AUC Score:            {metrics['roc_auc'] * 100:.2f}%")
            else:
                report_lines.append("  - ROC AUC Score:            N/A (only one class present in test set)")
            
            report_lines.append("----------------------------------------------------------------------------------")
            report_lines.append("Per-Class Breakdown:")
            for i, name in enumerate(class_names):
                prec = metrics['per_class']['precision'][i] * 100
                rec = metrics['per_class']['recall'][i] * 100
                f1_val = metrics['per_class']['f1'][i] * 100
                supp = metrics['per_class']['support'][i]
                report_lines.append(f"  - {name:<10}: Prec: {prec:.2f}% | Rec: {rec:.2f}% | F1: {f1_val:.2f}% (Support: {supp})")
            report_lines.append("==================================================================================")
            
            # Log the entire block
            for line in report_lines:
                logger.info(line)
                
            # Also save to evaluation_summary.txt in output_dir
            summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
            try:
                with open(summary_path, "w") as f:
                    f.write("\n".join(report_lines) + "\n")
                logger.info(f"Saved evaluation metrics summary to {summary_path}")
            except Exception as e:
                logger.error(f"Failed to save evaluation summary: {e}")

            return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def process_and_plot_embedding_norms(tracker, output_dir):
    """
    Process tracked unclipped L2 client embedding norms, compute distribution statistics,
    log recommendations for clipping bound C, save raw norms to CSV, and plot the distribution.
    """
    logger.info("Processing unclipped client embedding norms...")
    os.makedirs(output_dir, exist_ok=True)
    
    num_clients = len(tracker)
    if num_clients == 0 or all(len(lst) == 0 for lst in tracker):
        logger.warning("No client embedding norms were tracked.")
        return

    # 1. Flatten all norms across clients to compute global statistics
    all_norms = []
    for c_list in tracker:
        all_norms.extend(c_list)
        
    all_norms = np.array(all_norms)
    
    # Compute percentiles
    p50 = np.percentile(all_norms, 50)
    p75 = np.percentile(all_norms, 75)
    p90 = np.percentile(all_norms, 90)
    p95 = np.percentile(all_norms, 95)
    mean_val = np.mean(all_norms)
    std_val = np.std(all_norms)
    min_val = np.min(all_norms)
    max_val = np.max(all_norms)

    logger.info("==================================================")
    logger.info("📊 UNCLIPPED CLIENT EMBEDDING NORM STATISTICS:")
    logger.info(f"  Count: {len(all_norms)}")
    logger.info(f"  Min:   {min_val:.6f}")
    logger.info(f"  Max:   {max_val:.6f}")
    logger.info(f"  Mean:  {mean_val:.6f} (± {std_val:.6f})")
    logger.info(f"  50th Percentile (Median): {p50:.6f}")
    logger.info(f"  75th Percentile:          {p75:.6f}")
    logger.info(f"  90th Percentile:          {p90:.6f}")
    logger.info(f"  95th Percentile:          {p95:.6f}")
    logger.info("--------------------------------------------------")
    logger.info("💡 RECOMMENDATION FOR CLIPPING BOUND C:")
    logger.info(f"  - Conservative (Median):  --dp_clip_bound {p50:.4f}")
    logger.info(f"  - Balanced (75th %ile):    --dp_clip_bound {p75:.4f}")
    logger.info(f"  - Retain Utility (90th %ile): --dp_clip_bound {p90:.4f}")
    logger.info("==================================================")

    # 2. Save raw norms to CSV
    csv_path = os.path.join(output_dir, "unclipped_embedding_norms.csv")
    try:
        max_len = max(len(lst) for lst in tracker)
        rows = []
        for i in range(max_len):
            row = {"step": i + 1}
            for c in range(num_clients):
                if i < len(tracker[c]):
                    row[f"client_{c+1}_norm"] = tracker[c][i]
                else:
                    row[f"client_{c+1}_norm"] = ""
            rows.append(row)
            
        import csv
        with open(csv_path, mode="w", newline="") as f:
            fieldnames = ["step"] + [f"client_{c+1}_norm" for c in range(num_clients)]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved raw embedding norms to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save embedding norms CSV: {e}")

    # 3. Plot the distribution
    plot_path = os.path.join(output_dir, "unclipped_norms_distribution.png")
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(all_norms, bins=50, alpha=0.7, color="skyblue", edgecolor="black", label="Embedding Norms")
        plt.axvline(p50, color="green", linestyle="dashed", linewidth=1.5, label=f"Median ({p50:.4f})")
        plt.axvline(p90, color="orange", linestyle="dashed", linewidth=1.5, label=f"90th Percentile ({p90:.4f})")
        plt.axvline(p95, color="red", linestyle="dashed", linewidth=1.5, label=f"95th Percentile ({p95:.4f})")
        plt.title("Distribution of Unclipped Client Embedding Norms")
        plt.xlabel("L2 Embedding Norm")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved distribution plot to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to plot embedding norms distribution: {e}")


def create_visualizations(results: dict, output_dir: str):
    logger.info("Creating visualization plots")

    try:
        if "training" in results and "training_losses" in results["training"]:
            plot_training_progress(
                results["training"]["training_losses"],
                results["training"]["round_times"],
                os.path.join(output_dir, "training_progress.png"),
            )
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    args = parse_args()
    device = setup_experiment(args)
    results = run_federated_experiment(args, device)
    create_visualizations(results, args.output_dir)
    logger.info("Experiment completed successfully!")

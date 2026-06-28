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
        default="data",
        help="Path to dataset directory (default: data)",
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
        default=0.001,
        help="Learning rate for client-side layers (default: 0.001)",
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
        default=3,
        help="Patience for early stopping based on training loss (default: 3)",
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
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for federated training (default: 128)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Sliding window size (default: 5)",
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
    client_files = (
        [
            f
            for f in os.listdir(args.data_dir)
            if f.startswith("client_") and f.endswith(".csv")
        ]
        if os.path.exists(args.data_dir)
        else []
    )

    if len(client_files) < args.num_clients:
        data_ready = False

    if args.preprocess or not data_ready:
        logger.info(
            "Data directory not ready or preprocessing requested. Running preprocessing..."
        )

        input_file = args.input_file
        if not input_file:
            potential_files = (
                [f for f in os.listdir(args.data_dir) if f.endswith(".csv")]
                if os.path.exists(args.data_dir)
                else []
            )
            if potential_files:
                input_file = os.path.join(args.data_dir, potential_files[0])
                logger.info(f"Auto-detected input file: {input_file}")
            else:
                input_file = os.path.join(args.data_dir, "dummy_data.csv")
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
            "--num_clients",
            str(args.num_clients),
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

    fed_system = FedGATSageSystem(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    input_dim = args.window_size
    client_node_nums = []
    for c in range(args.num_clients):
        client_path = os.path.join(args.data_dir, f"client_{c+1}.csv")
        if os.path.exists(client_path):
            df_c = pd.read_csv(client_path, nrows=1)
            cols = [col for col in df_c.columns if col not in ["attack", "Normal/Attack", "Timestamp"]]
            client_node_nums.append(len(cols))
        else:
            client_node_nums.append(10)  # default fallback

    num_classes = 2
    sample_client_path = os.path.join(args.data_dir, "client_1.csv")
    if os.path.exists(sample_client_path):
        _ = fed_system.load_client_data(file_path=sample_client_path)
    if fed_system.label_mapper is not None:
        num_classes = len(fed_system.label_mapper)

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
            client_node_nums=client_node_nums,
            use_concat_skip=not args.disable_concat_skip,
        )
    else:
        input_dim = fed_system.input_dim or input_dim
        num_classes = fed_system.num_classes or num_classes
        logger.info(
            f"Resumed from checkpoint. Starting training at round {resume_round + 1}. "
            f"Using checkpoint model dimensions: input_dim={input_dim}, num_classes={num_classes}"
        )

    if args.num_rounds is not None and (resume_round + 1) >= args.num_rounds:
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
        training_results = fed_system.train_federated(
            num_rounds=args.num_rounds,
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
            normalize_vfl_gradients=args.enable_vfl_normalization,
            vfl_target_norm=args.vfl_target_norm,
            use_amp=not args.disable_amp,
            max_samples=200 if args.demo_mode else None,
            lr_scheduler_patience=args.lr_patience,
            lr_scheduler_factor=args.lr_factor,
            min_lr=args.min_lr,
            log_step_every=args.log_step_every,
            early_stopping_patience=args.early_stopping_patience,
            batch_size=args.batch_size,
        )

    evaluation_results = evaluate_system(fed_system, args)

    final_results = {
        "training": training_results,
        "evaluation": evaluation_results,
        "configuration": {
            "num_clients": args.num_clients,
            "num_rounds": args.num_rounds,
            "input_dim": input_dim,
            "num_classes": num_classes,
        },
    }

    if evaluation_results:
        tracker.log_round_metrics(
            len(training_results.get("training_losses", [])),
            {
                "final_accuracy": evaluation_results.get("accuracy", 0.0),
                "final_f1": evaluation_results.get("macro_f1", 0.0),
            },
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
        test_data_path = os.path.join(args.data_dir, "test.csv")
        if not os.path.exists(test_data_path):
            logger.warning("No test data found for evaluation")
            return {}

        df_test = pd.read_csv(test_data_path)
        if args.demo_mode:
            df_test = df_test.head(1000)

        # Retrieve client feature column mappings based on files
        client_feature_cols = []
        global_node_names = []
        for c in range(fed_system.num_clients):
            client_path = os.path.join(args.data_dir, f"client_{c+1}.csv")
            df_c = pd.read_csv(client_path, nrows=1)
            cols = [col for col in df_c.columns if col != "attack"]
            client_feature_cols.append(cols)
            global_node_names.extend(cols)

        # Set models to eval mode
        fed_system.global_model.eval()
        for client_model in fed_system.client_models.values():
            client_model.eval()

        predicted = []
        labels_list = []
        
        # Collect latent space embeddings for t-SNE plot
        latent_embeddings = []
        client_latent_embeddings = {c: [] for c in range(fed_system.num_clients)}
        latent_labels = []
        num_test_samples = len(df_test)
        tsne_max_samples = 2000
        sample_interval = max(1, num_test_samples // tsne_max_samples)

        anomaly_counter = 0
        culprit_counts = {}

        # Precompute sliding window test features for all clients
        w = fed_system.input_dim or 5
        logger.info(f"Precomputing sliding window test features with w={w}...")
        test_features_clients = {}
        for c in range(fed_system.num_clients):
            cols = client_feature_cols[c]
            raw_test_features = torch.tensor(df_test[cols].values, dtype=torch.float32, device=fed_system.device)
            test_features_clients[c] = build_sliding_windows(raw_test_features, w)

        with torch.no_grad():
            logger.info(f"Running VFL evaluation over {num_test_samples} test snapshots")
            for idx in range(num_test_samples):
                h_client_list = []
                for c in range(fed_system.num_clients):
                    snapshot_tensor = test_features_clients[c][idx]

                    h_c = fed_system.client_models[c](snapshot_tensor)
                    h_client_list.append(h_c)

                    if idx % sample_interval == 0:
                        client_emb = h_c.mean(dim=0).cpu().numpy()
                        client_latent_embeddings[c].append(client_emb)

                # Aggregate with client attention on the server if enabled
                if args.enable_client_attention:
                    h_global, _ = fed_system.global_model.client_attention(h_client_list)
                else:
                    h_global = torch.cat(h_client_list, dim=0)

                edge_index = fed_system._build_global_graph(h_global, fed_system.topk)
                embeddings, predictions, node_weights, _ = fed_system.global_model(h_global, edge_index)

                pred_class = predictions.argmax(dim=1).item()
                predicted.append(pred_class)
                labels_list.append(df_test["attack"].iloc[idx])

                # Identify culprit nodes if anomalous
                if pred_class == 1:
                    anomaly_counter += 1
                    weights = node_weights.squeeze(-1).cpu().numpy()
                    top_k = min(3, len(weights))
                    suspicious_indices = weights.argsort()[-top_k:][::-1]
                    top_culprits = [(i, global_node_names[i], weights[i]) for i in suspicious_indices]
                    
                    for _, name, _ in top_culprits:
                        culprit_counts[name] = culprit_counts.get(name, 0) + 1
                    
                    if anomaly_counter <= 5:
                        logger.info(f"🚨 SYSTEM ANOMALY DETECTED at snapshot {idx}!")
                        for rank, (node_idx, name, weight) in enumerate(top_culprits, 1):
                            logger.info(f"  - Culprit {rank}: Sensor {node_idx} (name: '{name}', Weight: {weight:.4f})")

                if idx % sample_interval == 0:
                    graph_emb = embeddings.mean(dim=0).cpu().numpy()
                    latent_embeddings.append(graph_emb)
                    latent_labels.append(df_test["attack"].iloc[idx])

                if (idx + 1) % 5000 == 0:
                    logger.info(f"Evaluated {idx + 1}/{num_test_samples} snapshots")

            y_true = np.array(labels_list)
            y_pred = np.array(predicted)

            class_names = None
            if fed_system.label_mapper:
                class_names = [
                    k
                    for k, v in sorted(
                        fed_system.label_mapper.items(), key=lambda x: x[1]
                    )
                ]

            metrics = calculate_metrics(y_true, y_pred, class_names)
            cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
            plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

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

            logger.info(
                f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}"
            )

            return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {}


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

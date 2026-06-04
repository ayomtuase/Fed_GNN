"""Main experiment script for FedGATSage.
Demonstrates the complete pipeline from data loading to evaluation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from federated_learning import FedGATSageSystem
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
        choices=["nf_ton_iot", "cic_ton_iot"],
        default="cic_ton_iot",
        help="Dataset to use",
    )
    parser.add_argument(
        "--num_clients", type=int, default=5, help="Number of federated clients"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=15, help="Number of federation rounds"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)"
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")
    logger.info(f"Experiment arguments: {vars(args)}")

    check_and_preprocess_data(args)

    return device


def run_federated_experiment(args, device: str) -> dict:
    logger.info("Starting FedGATSage federated learning experiment")

    experiment_name = (
        f"fedgatsage_{args.dataset}_{args.num_clients}clients_{args.num_rounds}rounds"
    )
    tracker = ExperimentTracker(experiment_name, args.output_dir)
    tracker.start_experiment()

    dataset_info = load_dataset_info(args.data_dir)
    logger.info(f"Dataset info: {dataset_info}")

    if args.demo_mode:
        args.num_rounds = min(args.num_rounds, 5)
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

    input_dim = 1
    node_num = 50  # Default: typical sensor count
    sample_client_path = os.path.join(args.data_dir, "client_1.csv")
    if os.path.exists(sample_client_path):
        sample_data = fed_system.load_client_data(file_path=sample_client_path)
        if sample_data and "features" in sample_data:
            node_num = sample_data["features"].shape[1]

    num_classes = 2
    if fed_system.label_mapper is not None:
        num_classes = len(fed_system.label_mapper)

    logger.info(
        f"Model configuration: node_num={node_num}, input_dim={input_dim}, num_classes={num_classes}"
    )

    resume_round = -1
    if args.resume_checkpoint or os.path.exists(checkpoint_dir):
        resume_round = fed_system.load_checkpoint(args.resume_checkpoint)

    if resume_round < 0:
        fed_system.initialize_models(
            input_dim=input_dim, hidden_dim=256, num_classes=num_classes, node_num=node_num
        )
    else:
        input_dim = fed_system.input_dim or input_dim
        num_classes = fed_system.num_classes or num_classes
        logger.info(
            f"Resumed from checkpoint. Starting training at round {resume_round + 1}. "
            f"Using checkpoint model dimensions: input_dim={input_dim}, num_classes={num_classes}"
        )

    if resume_round >= args.num_rounds:
        logger.info(
            "Checkpoint indicates training already completed. Skipping federated training."
        )
        training_results = fed_system.results
    else:
        training_results = fed_system.train_federated(
            num_rounds=args.num_rounds,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            start_round=resume_round if resume_round >= 0 else 0,
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
            args.num_rounds,
            {
                "final_accuracy": evaluation_results.get("accuracy", 0.0),
                "final_f1": evaluation_results.get("macro_f1", 0.0),
            },
        )

    tracker.save_experiment(final_results)
    return final_results


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

        test_data = fed_system.load_client_data(file_path=test_data_path)
        if test_data is None or "graph_label" not in test_data:
            logger.warning("Test data could not be processed")
            return {}

        primary_model = fed_system.client_models[0]
        primary_model.eval()

        with torch.no_grad():
            x = test_data["features"].to(fed_system.device)
            graph_labels = test_data.get("graph_labels")
            if graph_labels is None:
                graph_labels = test_data["graph_label"].expand(x.shape[0])
            graph_labels = graph_labels.to(fed_system.device)

            predicted = []
            for idx in range(x.shape[0]):
                snapshot = x[idx].view(x.shape[1], 1)
                _, graph_predictions = primary_model(snapshot)
                predicted.append(graph_predictions.argmax(dim=1).item())

            y_true = graph_labels.cpu().numpy()
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

#!/usr/bin/env python3
"""FedGATSage Hyperparameter Tuning using Optuna (Robust Patched Version).

Fixes and improvements:
- Resolves Bug A: Native `trial` reporting in federated_learning.py
- Resolves Bug B: Safe handling of study.best_trial when early trials are pruned
- Resolves Bug C: Strict GPU memory deallocation (del system + gc.collect + empty_cache) in finally block
- Resolves Bug D: Constrained window_size (10-120) and catch=(RuntimeError, OutOfMemoryError)
- Resolves Bug E: Static search space decoupling (kernel_size and window_size) preventing Optuna visualization crashes
- Resolves Bug F: Configurable contrastive_warmup_rounds aligned with pruner warmup_steps
"""

import argparse
import gc
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Ensure src is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)

from federated_learning import FedGATSageSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FedGATSageOptuna")


def detect_client_nodes(
    data_dir: str, num_clients: Optional[int] = None
) -> Tuple[int, List[int]]:
    """Dynamically determine the number of clients and node counts from preprocessed numpy files.

    If num_clients is None, it is inferred directly from the count of client_*.npy files.
    Client files are sorted numerically (e.g. client_1, client_2, ..., client_10).
    """
    train_dir = os.path.join(data_dir, "train")
    search_dir = train_dir if os.path.exists(train_dir) else os.path.join(data_dir, "validation")
    if not os.path.exists(search_dir):
        search_dir = data_dir

    client_files = glob.glob(os.path.join(search_dir, "client_*.npy"))
    if client_files:
        def extract_client_idx(filename: str) -> int:
            match = re.search(r"client_(\d+)\.npy$", os.path.basename(filename))
            return int(match.group(1)) if match else 0

        client_files.sort(key=extract_client_idx)
        detected_num = len(client_files)

        if num_clients is not None and num_clients != detected_num:
            logger.warning(
                f"Specified --num_clients ({num_clients}) does not match detected client count "
                f"({detected_num}) in {search_dir}. Using detected count: {detected_num}."
            )
        num_clients = detected_num

        client_node_nums = []
        for c_file in client_files:
            node_count = int(np.load(c_file, mmap_mode="r").shape[1])
            client_node_nums.append(node_count)

        return num_clients, client_node_nums
    else:
        if num_clients is None:
            raise FileNotFoundError(
                f"Could not find any 'client_*.npy' files in {search_dir} to auto-detect client count, "
                "and --num_clients was not specified."
            )
        logger.warning(
            f"No 'client_*.npy' files found in {search_dir}. Falling back to specified num_clients={num_clients}."
        )
        return num_clients, [10] * num_clients


def create_objective(
    data_dir: str,
    checkpoint_base_dir: str,
    num_clients: int,
    client_node_nums: List[int],
    max_rounds: int,
    batch_size: int,
    device: str,
    contrastive_warmup_rounds: int = 0,
    max_samples: Optional[int] = None,
):
    """Factory creating the Optuna objective function with fixed system parameters."""

    def objective(trial: optuna.Trial) -> float:
        # 1. Hyperparameter Search Space
        lr_client = trial.suggest_float("lr_client", 1e-4, 1e-2, log=True)
        lr_server = trial.suggest_float("lr_server", 1e-5, 1e-3, log=True)
        contrastive_weight = trial.suggest_float("contrastive_weight", 0.01, 0.1, step=0.01)
        contrastive_temp = trial.suggest_float("contrastive_temp", 0.05, 0.2, step=0.01)
        client_topk = trial.suggest_float("client_topk", 0.4, 0.8, step=0.1)
        global_topk = trial.suggest_int("global_topk", 10, 20, step=2)
        dp_noise_multiplier = trial.suggest_float("dp_noise_multiplier", 0.001, 0.01, log=True)
        sensor_embed_mode = trial.suggest_categorical(
            "sensor_embed_mode", ["graph_construction", "both"]
        )

        # Constrain window_size to prevent CUDA OOM on GPU
        window_size = trial.suggest_int("window_size", 10, 120, step=10)

        # Independent static categorical search space for kernel_size
        # With Conv1d padding = kernel_size // 2, padded length W + K - 1 >= K for all W >= 1,
        # so sequence length errors never occur and search spaces remain static across trials.
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 11, 15, 21, 31])

        trial_checkpoint_dir = os.path.join(checkpoint_base_dir, f"trial_{trial.number}")
        os.makedirs(trial_checkpoint_dir, exist_ok=True)

        system = None
        try:
            # 2. System Initialization
            system = FedGATSageSystem(
                data_dir=data_dir,
                num_clients=num_clients,
                device=device,
                checkpoint_dir=trial_checkpoint_dir,
                dtype=torch.float32,
            )

            system.initialize_models(
                input_dim=window_size,
                hidden_dim=256,
                num_classes=2,
                client_topk=client_topk,
                global_topk=global_topk,
                client_node_nums=client_node_nums,
                kernel_size=kernel_size,
                use_concat_skip=True,
                sensor_embed_mode=sensor_embed_mode,
            )

            # 3. Federated Training with Native Round Pruning
            results = system.train_federated(
                num_rounds=max_rounds,
                lr_client=lr_client,
                lr_server=lr_server,
                use_contrastive=True,
                contrastive_weight=contrastive_weight,
                contrastive_temp=contrastive_temp,
                contrastive_warmup_rounds=contrastive_warmup_rounds,
                dp_enabled=True,
                dp_clip_bound=21.0,
                dp_noise_multiplier=dp_noise_multiplier,
                batch_size=batch_size,
                window_size=window_size,
                max_samples=max_samples,
                checkpoint_every=max_rounds + 1,  # Only best checkpoint kept
                trial=trial,
            )

            val_losses = results.get("val_losses", [])
            if not val_losses:
                raise RuntimeError(f"Trial {trial.number} recorded no validation losses.")

            return float(min(val_losses))

        finally:
            # Strict memory deallocation to prevent CUDA OOM across trials
            if system is not None:
                del system
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def parse_args():
    parser = argparse.ArgumentParser(description="FedGATSage Optuna Hyperparameter Optimization")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/preprocessed_data",
        help="Path to preprocessed client dataset directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/optuna",
        help="Base directory for saving trial checkpoints",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="optuna_fedgatsage.db",
        help="Path to SQLite database file for persistent study storage",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="fedgatsage_hyperparameter_tuning",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of federated clients (default: None, auto-detected from preprocessed data)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials to optimize (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="Timeout in seconds for optimization (default: 14400 = 4 hours)",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=12,
        help="Maximum federated training rounds per trial (default: 12)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training and validation (default: 1024)",
    )
    parser.add_argument(
        "--startup_trials",
        type=int,
        default=3,
        help="Number of startup trials before pruning takes effect (default: 3)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=3,
        help="Number of initial rounds in each trial before pruning is evaluated (default: 3)",
    )
    parser.add_argument(
        "--contrastive_warmup_rounds",
        type=int,
        default=0,
        help="Rounds before contrastive weight is activated (default: 0 so active during all rounds)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum samples to use from dataset (useful for testing/fast profiling)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/optuna",
        help="Directory to save best parameters and visualization plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Using compute device: {device}")

    # Resolve paths
    data_dir = os.path.abspath(args.data_dir)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect number of clients and node counts from preprocessed data
    logger.info(f"Inspecting client data in {data_dir}...")
    num_clients, client_node_nums = detect_client_nodes(data_dir, args.num_clients)
    args.num_clients = num_clients
    logger.info(f"Using {args.num_clients} clients with node counts: {client_node_nums}")

    # Set up SQLite storage
    db_path = os.path.abspath(args.db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    storage_url = f"sqlite:///{db_path}"
    logger.info(f"Study persistence storage URL: {storage_url}")

    # Median pruner for early stopping of underperforming trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.startup_trials,
        n_warmup_steps=args.warmup_steps,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    objective = create_objective(
        data_dir=data_dir,
        checkpoint_base_dir=checkpoint_dir,
        num_clients=args.num_clients,
        client_node_nums=client_node_nums,
        max_rounds=args.max_rounds,
        batch_size=args.batch_size,
        device=device,
        contrastive_warmup_rounds=args.contrastive_warmup_rounds,
        max_samples=args.max_samples,
    )

    # Catch memory errors to prevent single trial OOM from crashing the entire study
    catch_errors = [RuntimeError]
    if hasattr(torch.cuda, "OutOfMemoryError"):
        catch_errors.append(torch.cuda.OutOfMemoryError)
    if hasattr(torch, "OutOfMemoryError"):
        catch_errors.append(torch.OutOfMemoryError)
    catch_tuple = tuple(set(catch_errors))

    logger.info(f"Starting study optimization: n_trials={args.n_trials}, timeout={args.timeout}s...")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        catch=catch_tuple,
    )

    # Print summary safely
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Number of finished trials: {len(study.trials)}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        best_trial = study.best_trial
        logger.info(f"Best Trial Number: {best_trial.number}")
        logger.info(f"Best Validation Loss: {best_trial.value:.6f}")
        
        logger.info("Best Hyperparameters:")
        for k, v in best_trial.params.items():
            logger.info(f"  --{k}: {v}")

        # Save best parameters to JSON
        best_params_path = os.path.join(output_dir, "best_hyperparameters.json")
        best_record = {
            "trial_number": best_trial.number,
            "best_val_loss": best_trial.value,
            "parameters": best_trial.params,
        }
        with open(best_params_path, "w") as f:
            json.dump(best_record, f, indent=2)
        logger.info(f"Saved best hyperparameters to: {best_params_path}")

        # Save interactive visualizations
        try:
            hist_fig = plot_optimization_history(study)
            hist_fig.write_html(os.path.join(output_dir, "optimization_history.html"))

            if len(completed_trials) > 1:
                imp_fig = plot_param_importances(study)
                imp_fig.write_html(os.path.join(output_dir, "param_importances.html"))

                slice_fig = plot_slice(study)
                slice_fig.write_html(os.path.join(output_dir, "param_slice.html"))

                par_fig = plot_parallel_coordinate(study)
                par_fig.write_html(os.path.join(output_dir, "parallel_coordinates.html"))

            logger.info(f"Interactive Optuna plots saved to: {output_dir}")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    else:
        logger.warning("No trials completed successfully (all were pruned or failed).")

    logger.info(
        f"To inspect results in web UI, run:\n"
        f"  optuna-dashboard sqlite:///{db_path}"
    )


if __name__ == "__main__":
    main()

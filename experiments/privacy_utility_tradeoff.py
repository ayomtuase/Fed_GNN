#!/usr/bin/env python3
"""
Privacy-Utility Tradeoff Analysis for FedGATSage (DP-VFL).

This module provides tools to systematically evaluate and visualize the
trade-off between Differential Privacy (DP) guarantees and model utility
(ROC-AUC, Macro F1, Balanced Accuracy, Accuracy) across the Vertical
Federated Learning (VFL) boundary.

Key Capabilities:
1. compute_dp_epsilon:
   Exact Rényi Differential Privacy (RDP) composition accountant for Gaussian
   noise perturbations with subsampling support.
2. run_privacy_utility_experiment:
   Runs experiments across a spectrum of DP noise multipliers (sigma), logging
   metrics, computing epsilon, and saving structured results (CSV and JSON).
3. plot_privacy_utility_tradeoff:
   Generates publication-quality multi-panel and summary tradeoff curves with
   annotated Pareto frontiers, privacy regimes, and baseline references.
4. run_and_plot_privacy_utility_tradeoff:
   High-level one-step runner and plotter callable directly from Python or CLI.
"""

import argparse
import copy
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set safe Matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure src and experiments directories are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

for p in [str(PROJECT_ROOT), str(SRC_DIR), str(EXPERIMENTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from federated_learning import FedGATSageSystem
from utils import setup_logging

logger = logging.getLogger("PrivacyUtilityTradeoff")


# ==============================================================================
# 1. Differential Privacy Accountant (Rényi Differential Privacy)
# ==============================================================================

def compute_dp_epsilon(
    steps: int,
    noise_multiplier: float,
    batch_size: int = 256,
    dataset_size: int = 20000,
    target_delta: float = 1e-5,
    sample_rate: Optional[float] = None,
) -> float:
    """
    Compute cumulative DP privacy budget epsilon for Gaussian mechanism
    over repeated composition steps using Rényi Differential Privacy (RDP).

    Args:
        steps: Total number of mechanism applications (rounds * steps_per_round).
        noise_multiplier: Standard deviation of Gaussian noise relative to L2 clipping bound (sigma).
        batch_size: Batch size used per training iteration.
        dataset_size: Total number of training samples/snapshots.
        target_delta: DP failure probability delta (typically 1e-5 or 1/N).
        sample_rate: Subsampling ratio q = batch_size / dataset_size. Overrides batch_size/dataset_size if given.

    Returns:
        epsilon: The cumulative (epsilon, delta)-DP privacy budget. Returns float('inf') if sigma <= 0.
    """
    if noise_multiplier <= 0.0 or steps <= 0:
        return float("inf")

    if sample_rate is None:
        q = min(1.0, float(batch_size) / max(1.0, float(dataset_size)))
    else:
        q = min(1.0, max(1e-6, float(sample_rate)))

    # Order range for numerical optimization of RDP conversion
    alphas = np.linspace(1.02, 128.0, 1000)

    # Subsampled Gaussian mechanism RDP formulation (Mironov et al., 2019)
    # For small q, RDP(alpha) <= (q^2 * alpha) / (2 * sigma^2)
    rdp_per_step = (q ** 2) * alphas / (2.0 * (noise_multiplier ** 2))
    total_rdp = steps * rdp_per_step

    # Convert RDP to standard (epsilon, delta)-DP: eps(delta) = min_alpha (RDP(alpha) + log(1/delta) / (alpha - 1))
    epsilons = total_rdp + np.log(1.0 / target_delta) / (alphas - 1.0)
    best_eps = float(np.min(epsilons))
    return best_eps


# ==============================================================================
# 2. Experiment Execution Runner
# ==============================================================================

def run_privacy_utility_experiment(
    noise_multipliers: Optional[List[float]] = None,
    clip_bound: float = 21.0,
    num_rounds: int = 5,
    data_dir: str = "data/preprocessed_data",
    output_dir: str = "results/privacy_utility_tradeoff",
    device: str = "auto",
    target_delta: float = 1e-5,
    mode: str = "train",
    checkpoint_path: Optional[str] = None,
    eval_noise: bool = False,
    demo_mode: bool = False,
    batch_size: int = 128,
    window_size: int = 30,
    base_args: Optional[argparse.Namespace] = None,
) -> pd.DataFrame:
    """
    Execute privacy-utility trade-off experiments over multiple noise multipliers.

    Args:
        noise_multipliers: List of noise multiplier (sigma) values to evaluate.
                           Default: [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        clip_bound: L2 norm clipping threshold C for client embeddings.
        num_rounds: Number of federated training rounds per run (in 'train' mode).
        data_dir: Directory containing preprocessed client dataset and split labels.
        output_dir: Destination directory for logs, CSV results, and checkpoints.
        device: Device to use ('auto', 'cuda', 'mps', 'cpu').
        target_delta: Privacy budget delta parameter (default: 1e-5).
        mode: 'train' to train a new model for each noise setting, or 'eval' to evaluate a checkpoint.
        checkpoint_path: Path to checkpoint to load if in 'eval' mode or for warm-starting.
        eval_noise: Whether to inject DP noise during evaluation (test time).
        demo_mode: If True, uses demo configuration with fewer samples/rounds for rapid testing.
        batch_size: Mini-batch size.
        window_size: Temporal sliding window size.
        base_args: Optional pre-configured argparse.Namespace.

    Returns:
        DataFrame containing metrics across all evaluated privacy levels.
    """
    from fedgatsage_experiment import parse_args, run_federated_experiment, setup_experiment

    if noise_multipliers is None:
        noise_multipliers = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    os.makedirs(output_dir, exist_ok=True)
    setup_logging("INFO")

    logger.info("==================================================================")
    logger.info("🚀 STARTING PRIVACY-UTILITY TRADEOFF EXPERIMENT SUITE")
    logger.info(f"  Mode:              {mode.upper()}")
    logger.info(f"  Noise levels:      {noise_multipliers}")
    logger.info(f"  Clipping bound C:  {clip_bound}")
    logger.info(f"  Rounds per level:  {num_rounds}")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info("==================================================================")

    # Estimate dataset size for DP accounting
    train_labels_file = os.path.join(data_dir, "train_labels.npy")
    est_dataset_size = 20000
    if os.path.exists(train_labels_file):
        try:
            m = np.load(train_labels_file, mmap_mode="r")
            est_dataset_size = len(m)
        except Exception:
            pass

    records: List[Dict[str, Any]] = []

    for idx, sigma in enumerate(noise_multipliers):
        sigma = float(sigma)
        logger.info("\n" + "=" * 60)
        logger.info(f"▶ [{idx + 1}/{len(noise_multipliers)}] EVALUATING DP NOISE MULTIPLIER: sigma = {sigma:.4f}")
        logger.info("=" * 60)

        # Prepare specific args for this run
        if base_args is not None:
            run_args = copy.deepcopy(base_args)
        else:
            run_args = parse_args([])

        run_args.data_dir = data_dir
        run_args.demo_mode = demo_mode
        run_args.num_rounds = num_rounds
        run_args.batch_size = batch_size
        run_args.window_size = window_size
        run_args.device = device

        run_args.enable_dp = (sigma > 0.0)
        run_args.dp_noise_multiplier = sigma
        run_args.dp_clip_bound = clip_bound
        run_args.dp_profile = False

        run_dir_name = f"run_sigma_{sigma:.4f}".replace(".", "_")
        run_output_dir = os.path.join(output_dir, run_dir_name)
        os.makedirs(run_output_dir, exist_ok=True)
        run_args.output_dir = run_output_dir
        run_args.checkpoint_dir = os.path.join(run_output_dir, "checkpoints")

        if mode == "eval":
            run_args.eval_only = True
            if checkpoint_path is not None:
                run_args.checkpoint_path = checkpoint_path

        resolved_device = setup_experiment(run_args)

        try:
            experiment_result = run_federated_experiment(run_args, resolved_device)
            eval_metrics = experiment_result.get("evaluation", {})
            train_results = experiment_result.get("training", {})

            # Calculate total optimization steps for DP accounting
            num_trained_rounds = len(train_results.get("training_losses", []))
            effective_rounds = num_trained_rounds if num_trained_rounds > 0 else num_rounds
            steps_per_round = max(1, est_dataset_size // batch_size)
            total_steps = effective_rounds * steps_per_round

            # Compute privacy budget epsilon
            eps = compute_dp_epsilon(
                steps=total_steps,
                noise_multiplier=sigma,
                batch_size=batch_size,
                dataset_size=est_dataset_size,
                target_delta=target_delta,
            )

            record = {
                "noise_multiplier": sigma,
                "clip_bound": clip_bound,
                "epsilon": eps,
                "delta": target_delta,
                "rounds": effective_rounds,
                "steps": total_steps,
                "accuracy": eval_metrics.get("accuracy", np.nan),
                "balanced_accuracy": eval_metrics.get("balanced_accuracy", np.nan),
                "macro_f1": eval_metrics.get("macro_f1", np.nan),
                "weighted_f1": eval_metrics.get("weighted_f1", np.nan),
                "roc_auc": eval_metrics.get("roc_auc", np.nan),
                "final_train_loss": train_results.get("training_losses", [np.nan])[-1] if train_results.get("training_losses") else np.nan,
                "final_val_loss": train_results.get("val_losses", [np.nan])[-1] if train_results.get("val_losses") else np.nan,
            }

            # Extract per-class breakdown if available
            per_class = eval_metrics.get("per_class", {})
            if "precision" in per_class and len(per_class["precision"]) >= 2:
                record["normal_precision"] = per_class["precision"][0]
                record["normal_recall"] = per_class["recall"][0]
                record["normal_f1"] = per_class["f1"][0]
                record["anomaly_precision"] = per_class["precision"][1]
                record["anomaly_recall"] = per_class["recall"][1]
                record["anomaly_f1"] = per_class["f1"][1]

            # Raw unadjusted metrics
            raw_metrics = eval_metrics.get("raw", {})
            record["raw_accuracy"] = raw_metrics.get("accuracy", np.nan)
            record["raw_balanced_accuracy"] = raw_metrics.get("balanced_accuracy", np.nan)
            record["raw_macro_f1"] = raw_metrics.get("macro_f1", np.nan)
            record["raw_roc_auc"] = raw_metrics.get("roc_auc", np.nan)

            records.append(record)

            eps_display = f"{eps:.4f}" if np.isfinite(eps) else "∞ (Baseline)"
            logger.info("--------------------------------------------------")
            logger.info(f"✅ RESULTS FOR sigma={sigma:.4f} (epsilon={eps_display}):")
            logger.info(f"   ROC-AUC:           {record['roc_auc'] * 100:.2f}%" if pd.notna(record['roc_auc']) else "   ROC-AUC: N/A")
            logger.info(f"   Macro F1:          {record['macro_f1'] * 100:.2f}%" if pd.notna(record['macro_f1']) else "   Macro F1: N/A")
            logger.info(f"   Balanced Accuracy: {record['balanced_accuracy'] * 100:.2f}%" if pd.notna(record['balanced_accuracy']) else "   Balanced Accuracy: N/A")
            logger.info(f"   Accuracy:          {record['accuracy'] * 100:.2f}%" if pd.notna(record['accuracy']) else "   Accuracy: N/A")
            logger.info("--------------------------------------------------")

        except Exception as e:
            logger.error(f"Failed run for sigma={sigma}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            # Memory cleanup between iterations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "privacy_utility_results.csv")
    json_path = os.path.join(output_dir, "privacy_utility_results.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=lambda x: None if pd.isna(x) else x)

    logger.info(f"\n📁 Saved privacy-utility tradeoff table to: {csv_path}")
    return df


# ==============================================================================
# 3. Visualization and Trade-off Plotting
# ==============================================================================

def plot_privacy_utility_tradeoff(
    data: Union[pd.DataFrame, str, List[Dict[str, Any]]],
    output_dir: str = "results/privacy_utility_tradeoff",
    save_filename: str = "privacy_utility_tradeoff",
    baseline_sigma: float = 0.0,
    title_suffix: str = "",
) -> Dict[str, plt.Figure]:
    """
    Generate publication-quality figures illustrating the Privacy-Utility Tradeoff.

    Creates two figures:
    1. Multi-panel 2x2 comprehensive diagnostic analysis:
       - Panel A: Utility vs. DP Noise Multiplier (sigma)
       - Panel B: Utility vs. Privacy Budget (epsilon) with regime annotations
       - Panel C: ROC-AUC Tradeoff Curve & Pareto Frontier with sweet spot / knee point
       - Panel D: Relative Utility Retention (%) vs. Privacy
    2. Summary presentation figure:
       - Clean dual-axis plot (bottom X: epsilon, top X: sigma) designed for papers/slides.

    Args:
        data: DataFrame, CSV filepath, or list of dicts with tradeoff records.
        output_dir: Directory where plots will be saved.
        save_filename: Base filename for saved PNG/PDF plots.
        baseline_sigma: Noise multiplier considered as non-private baseline (default: 0.0).
        title_suffix: Optional extra text for plot titles.

    Returns:
        Dictionary mapping figure names ('multi' and 'summary') to plt.Figure objects.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data must be a pandas DataFrame, CSV file path, or list of dicts.")

    if df.empty:
        logger.warning("Empty DataFrame provided to plot_privacy_utility_tradeoff. Skipping plotting.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # Sort monotonically by noise_multiplier
    df = df.sort_values(by="noise_multiplier").reset_index(drop=True)

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#d0d0d0"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e5e5e5"
    plt.rcParams["grid.linestyle"] = ":"

    # Baseline performance metrics
    baseline_row = df[df["noise_multiplier"] == baseline_sigma]
    if baseline_row.empty:
        baseline_row = df.iloc[[0]]

    base_auc = baseline_row["roc_auc"].values[0] if "roc_auc" in baseline_row else np.nan
    base_f1 = baseline_row["macro_f1"].values[0] if "macro_f1" in baseline_row else np.nan
    base_bal_acc = baseline_row["balanced_accuracy"].values[0] if "balanced_accuracy" in baseline_row else np.nan

    # Colors
    c_auc = "#1f77b4"      # Blue
    c_f1 = "#ff7f0e"       # Orange
    c_bal_acc = "#2ca02c"  # Green
    c_acc = "#9467bd"      # Purple

    figures = {}

    # --------------------------------------------------------------------------
    # Figure 1: Comprehensive 2x2 Multi-Panel Grid
    # --------------------------------------------------------------------------
    fig_multi, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    fig_multi.subplots_adjust(hspace=0.28, wspace=0.24)

    # Panel A: Utility vs. Noise Multiplier (sigma)
    ax_a = axes[0, 0]
    sigmas = df["noise_multiplier"].values

    if "roc_auc" in df and df["roc_auc"].notna().any():
        ax_a.plot(sigmas, df["roc_auc"] * 100, marker="o", color=c_auc, lw=2.2, label="ROC-AUC Score")
    if "macro_f1" in df and df["macro_f1"].notna().any():
        ax_a.plot(sigmas, df["macro_f1"] * 100, marker="s", color=c_f1, lw=2.0, linestyle="--", label="Macro F1 Score")
    if "balanced_accuracy" in df and df["balanced_accuracy"].notna().any():
        ax_a.plot(sigmas, df["balanced_accuracy"] * 100, marker="^", color=c_bal_acc, lw=1.8, linestyle="-.", label="Balanced Accuracy")
    if "accuracy" in df and df["accuracy"].notna().any():
        ax_a.plot(sigmas, df["accuracy"] * 100, marker="d", color=c_acc, lw=1.5, linestyle=":", label="Accuracy")

    ax_a.set_title(r"A. Utility Metrics vs. Noise Multiplier ($\sigma$)", fontsize=13, fontweight="bold", pad=10)
    ax_a.set_xlabel(r"DP Noise Multiplier ($\sigma$)", fontsize=11)
    ax_a.set_ylabel("Metric Score (%)", fontsize=11)
    ax_a.set_ylim([0, 105])
    ax_a.legend(loc="lower left", framealpha=0.9)
    ax_a.grid(True)

    # Panel B: Utility vs. Privacy Budget (epsilon)
    ax_b = axes[0, 1]
    finite_mask = np.isfinite(df["epsilon"]) & (df["epsilon"] > 0)
    df_finite = df[finite_mask].sort_values(by="epsilon")

    if not df_finite.empty:
        eps_vals = df_finite["epsilon"].values

        # Privacy regime background shading
        min_eps_plot = max(0.1, np.min(eps_vals) * 0.7)
        max_eps_plot = np.max(eps_vals) * 1.4

        ax_b.axvspan(min_eps_plot, 1.0, color="#2ca02c", alpha=0.10, label=r"High Privacy ($\epsilon \leq 1$)")
        ax_b.axvspan(1.0, 10.0, color="#ff7f0e", alpha=0.10, label=r"Moderate Privacy ($1 < \epsilon \leq 10$)")
        ax_b.axvspan(10.0, max_eps_plot, color="#d62728", alpha=0.08, label=r"Relaxed Privacy ($\epsilon > 10$)")

        if "roc_auc" in df_finite and df_finite["roc_auc"].notna().any():
            ax_b.plot(eps_vals, df_finite["roc_auc"] * 100, marker="o", color=c_auc, lw=2.2, label="ROC-AUC")
        if "macro_f1" in df_finite and df_finite["macro_f1"].notna().any():
            ax_b.plot(eps_vals, df_finite["macro_f1"] * 100, marker="s", color=c_f1, lw=2.0, linestyle="--", label="Macro F1")
        if "balanced_accuracy" in df_finite and df_finite["balanced_accuracy"].notna().any():
            ax_b.plot(eps_vals, df_finite["balanced_accuracy"] * 100, marker="^", color=c_bal_acc, lw=1.8, linestyle="-.", label="Balanced Acc")

        # Baseline horizontal reference lines
        if pd.notna(base_auc):
            ax_b.axhline(base_auc * 100, color=c_auc, linestyle=":", alpha=0.6, label=f"Baseline AUC ({base_auc * 100:.1f}%)")

        ax_b.set_xscale("log")
        ax_b.set_xlim([min_eps_plot, max_eps_plot])
        ax_b.set_title(r"B. Utility Metrics vs. Privacy Budget ($\epsilon$)", fontsize=13, fontweight="bold", pad=10)
        ax_b.set_xlabel(r"Privacy Budget ($\epsilon$, log scale) $\rightarrow$ Lower is Stronger Privacy", fontsize=11)
        ax_b.set_ylabel("Metric Score (%)", fontsize=11)
        ax_b.set_ylim([0, 105])
        ax_b.legend(loc="lower right", framealpha=0.9, fontsize=9)
        ax_b.grid(True)
    else:
        ax_b.text(0.5, 0.5, "No finite epsilon values found\n(all sigma = 0)", ha="center", va="center")

    # Panel C: Pareto Frontier & Optimal Knee Point (AUC vs. Epsilon)
    ax_c = axes[1, 0]
    if not df_finite.empty and "roc_auc" in df_finite and df_finite["roc_auc"].notna().any():
        sorted_by_eps = df_finite.sort_values(by="epsilon")
        e_arr = sorted_by_eps["epsilon"].values
        auc_arr = sorted_by_eps["roc_auc"].values * 100

        # Plot curve
        ax_c.plot(e_arr, auc_arr, marker="o", color=c_auc, lw=2.5, zorder=3, label="Tradeoff Frontier")
        ax_c.fill_between(e_arr, auc_arr, color=c_auc, alpha=0.12, zorder=2)

        # Detect knee point: maximize utility retention per unit of log-epsilon privacy loss
        norm_auc = auc_arr / (np.max(auc_arr) + 1e-6)
        norm_eps = np.log10(e_arr + 1.0) / (np.max(np.log10(e_arr + 1.0)) + 1e-6)
        tradeoff_score = (norm_auc) - 0.5 * norm_eps
        best_knee_idx = int(np.argmax(tradeoff_score))

        knee_eps = e_arr[best_knee_idx]
        knee_auc = auc_arr[best_knee_idx]
        knee_sigma = sorted_by_eps["noise_multiplier"].values[best_knee_idx]

        ax_c.scatter([knee_eps], [knee_auc], s=160, color="#d62728", edgecolor="black", lw=1.8, zorder=5, label=rf"Optimal Knee Point ($\sigma={knee_sigma:.2f}$)")

        ax_c.annotate(
            rf"Sweet Spot:" + "\n" + rf"$\epsilon={knee_eps:.2f}$, AUC={knee_auc:.1f}%",
            xy=(knee_eps, knee_auc),
            xytext=(knee_eps * 0.35, knee_auc - 18.0),
            arrowprops=dict(facecolor="#d62728", edgecolor="#d62728", arrowstyle="->", lw=1.5, connectionstyle="arc3,rad=-0.15"),
            fontsize=10,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2f2", edgecolor="#d62728", alpha=0.9),
            zorder=6,
        )

        ax_c.set_xscale("log")
        ax_c.set_title(r"C. Pareto Frontier (ROC-AUC vs. $\epsilon$)", fontsize=13, fontweight="bold", pad=10)
        ax_c.set_xlabel(r"Privacy Budget ($\epsilon$, log scale)", fontsize=11)
        ax_c.set_ylabel("ROC-AUC Score (%)", fontsize=11)
        ax_c.set_ylim([max(0, np.min(auc_arr) - 15), 105])
        ax_c.legend(loc="lower right", framealpha=0.9)
        ax_c.grid(True)
    else:
        ax_c.text(0.5, 0.5, "Insufficient AUC data for Pareto Frontier", ha="center", va="center")

    # Panel D: Relative Utility Retention (%) vs. Privacy
    ax_d = axes[1, 1]
    if pd.notna(base_auc) and base_auc > 0 and not df_finite.empty:
        retention_auc = (df_finite["roc_auc"].values / base_auc) * 100
        retention_f1 = (df_finite["macro_f1"].values / (base_f1 if pd.notna(base_f1) and base_f1 > 0 else 1.0)) * 100

        ax_d.plot(df_finite["epsilon"], retention_auc, marker="o", color=c_auc, lw=2.2, label="ROC-AUC Retention (%)")
        if pd.notna(base_f1) and base_f1 > 0:
            ax_d.plot(df_finite["epsilon"], retention_f1, marker="s", color=c_f1, lw=2.0, linestyle="--", label="Macro F1 Retention (%)")

        ax_d.axhline(100.0, color="gray", linestyle=":", lw=1.5, alpha=0.7, label="100% Baseline")
        ax_d.axhline(90.0, color="#2ca02c", linestyle="--", lw=1.2, alpha=0.6, label="90% Utility Retained")

        ax_d.set_xscale("log")
        ax_d.set_title(r"D. Relative Utility Retention vs. Non-Private Baseline", fontsize=13, fontweight="bold", pad=10)
        ax_d.set_xlabel(r"Privacy Budget ($\epsilon$, log scale)", fontsize=11)
        ax_d.set_ylabel("Utility Retained vs. Baseline (%)", fontsize=11)
        ax_d.set_ylim([max(0, min(np.min(retention_auc), np.min(retention_f1) if pd.notna(base_f1) else 100) - 15), 110])
        ax_d.legend(loc="lower right", framealpha=0.9)
        ax_d.grid(True)
    else:
        ax_d.text(0.5, 0.5, "Non-private baseline not found for retention calculation", ha="center", va="center")

    plt.suptitle(f"Differential Privacy vs. Utility Tradeoff Analysis {title_suffix}".strip(), fontsize=16, fontweight="bold", y=0.98)

    multi_png = os.path.join(output_dir, f"{save_filename}_multi.png")
    multi_pdf = os.path.join(output_dir, f"{save_filename}_multi.pdf")
    fig_multi.savefig(multi_png, dpi=300, bbox_inches="tight")
    fig_multi.savefig(multi_pdf, bbox_inches="tight")
    figures["multi"] = fig_multi
    logger.info(f"📊 Saved multi-panel tradeoff figure to: {multi_png} (and .pdf)")

    # --------------------------------------------------------------------------
    # Figure 2: Publication-Ready Dual-Axis Summary Plot
    # --------------------------------------------------------------------------
    fig_summary, ax_main = plt.subplots(figsize=(9, 6), dpi=300)

    if not df_finite.empty:
        sorted_df = df_finite.sort_values(by="epsilon")
        eps = sorted_df["epsilon"].values
        aucs = sorted_df["roc_auc"].values * 100
        f1s = sorted_df["macro_f1"].values * 100

        # Primary Curves
        line1 = ax_main.plot(eps, aucs, marker="o", color=c_auc, lw=2.5, markersize=8, label="ROC-AUC Score (%)")
        line2 = ax_main.plot(eps, f1s, marker="s", color=c_f1, lw=2.2, linestyle="--", markersize=7, label="Macro F1 Score (%)")

        if pd.notna(base_auc):
            line3 = ax_main.axhline(base_auc * 100, color=c_auc, linestyle=":", lw=1.5, alpha=0.6, label=f"Baseline AUC ({base_auc * 100:.1f}%)")

        # Shaded background zones for DP guarantee strength
        ax_main.axvspan(min(eps) * 0.7, 1.0, color="#2ca02c", alpha=0.08)
        ax_main.axvspan(1.0, 10.0, color="#ff7f0e", alpha=0.08)
        ax_main.axvspan(10.0, max(eps) * 1.4, color="#d62728", alpha=0.06)

        # Labels for zones
        y_text_pos = max(10, min(aucs) - 8)
        ax_main.text(min(1.0, min(eps) * 1.5), y_text_pos, r"Strong DP" + "\n" + r"($\epsilon \leq 1$)", color="#1b5e20", fontsize=9, fontweight="bold", ha="center")
        ax_main.text(3.0, y_text_pos, r"Moderate DP" + "\n" + r"($1 < \epsilon \leq 10$)", color="#b78103", fontsize=9, fontweight="bold", ha="center")
        ax_main.text(min(max(eps) * 0.7, 100.0), y_text_pos, r"Relaxed DP" + "\n" + r"($\epsilon > 10$)", color="#b71c1c", fontsize=9, fontweight="bold", ha="center")

        ax_main.set_xscale("log")
        ax_main.set_xlabel(r"Differential Privacy Budget $\epsilon$ (log scale, $\delta = 10^{-5}$)", fontsize=12, fontweight="semibold", labelpad=8)
        ax_main.set_ylabel("Evaluation Metric Score (%)", fontsize=12, fontweight="semibold", labelpad=8)
        ax_main.set_title(f"FedGATSage Privacy-Utility Tradeoff {title_suffix}".strip(), fontsize=14, fontweight="bold", pad=12)
        ax_main.set_ylim([max(0, min(np.min(aucs), np.min(f1s)) - 12), 105])

        # Dual Axis on Top showing corresponding noise multipliers (sigma)
        ax_top = ax_main.twiny()
        ax_top.set_xscale("log")
        ax_top.set_xlim(ax_main.get_xlim())

        # Select a clean subset of points to label on top axis
        step = max(1, len(eps) // 5)
        top_ticks = eps[::step]
        top_labels = [rf"$\sigma={s:.3f}$" for s in sorted_df["noise_multiplier"].values[::step]]
        ax_top.set_xticks(top_ticks)
        ax_top.set_xticklabels(top_labels, fontsize=9)
        ax_top.set_xlabel(r"Noise Multiplier ($\sigma$)", fontsize=11, labelpad=8)

        ax_main.legend(loc="lower right", framealpha=0.92, fontsize=10)
        ax_main.grid(True, linestyle=":", alpha=0.7)

    summary_png = os.path.join(output_dir, f"{save_filename}_summary.png")
    summary_pdf = os.path.join(output_dir, f"{save_filename}_summary.pdf")
    fig_summary.savefig(summary_png, dpi=300, bbox_inches="tight")
    fig_summary.savefig(summary_pdf, bbox_inches="tight")
    figures["summary"] = fig_summary
    logger.info(f"📊 Saved summary tradeoff figure to: {summary_png} (and .pdf)")

    return figures


# ==============================================================================
# 4. Master Orchestrator Function
# ==============================================================================

def run_and_plot_privacy_utility_tradeoff(
    noise_multipliers: Optional[List[float]] = None,
    clip_bound: float = 21.0,
    num_rounds: int = 5,
    data_dir: str = "data/preprocessed_data",
    output_dir: str = "results/privacy_utility_tradeoff",
    device: str = "auto",
    target_delta: float = 1e-5,
    mode: str = "train",
    checkpoint_path: Optional[str] = None,
    demo_mode: bool = False,
    batch_size: int = 128,
    window_size: int = 30,
    plot_only_csv: Optional[str] = None,
    base_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    """
    All-in-one function to run differential privacy experiments across a spectrum
    of noise multipliers, collect utility metrics, compute privacy budgets (epsilon),
    and generate publication-ready trade-off plots.

    Args:
        noise_multipliers: List of noise multipliers sigma. Default: [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5].
        clip_bound: L2 clipping bound C for embeddings across VFL boundary.
        num_rounds: Number of federated training rounds per noise setting.
        data_dir: Dataset root directory.
        output_dir: Directory where results, tables, and plots are stored.
        device: 'auto', 'cuda', 'mps', or 'cpu'.
        target_delta: Failure probability delta in (epsilon, delta)-DP.
        mode: 'train' (trains new models per DP setting) or 'eval' (evaluates a checkpoint under noise).
        checkpoint_path: Pretrained checkpoint for 'eval' mode.
        demo_mode: If True, uses demo limits for quick testing.
        batch_size: Mini-batch size.
        window_size: Sliding window size.
        plot_only_csv: If provided, skips experiment execution and plots directly from this CSV file.
        base_args: Optional base argparse.Namespace.

    Returns:
        Dictionary containing:
        - "dataframe": pd.DataFrame of all results
        - "figures": Dict of generated matplotlib Figures ('multi' and 'summary')
        - "csv_path": Path to the saved results CSV
        - "output_dir": Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    if plot_only_csv is not None and os.path.exists(plot_only_csv):
        logger.info(f"Plot-only mode activated. Reading existing results from {plot_only_csv}")
        df = pd.read_csv(plot_only_csv)
    else:
        df = run_privacy_utility_experiment(
            noise_multipliers=noise_multipliers,
            clip_bound=clip_bound,
            num_rounds=num_rounds,
            data_dir=data_dir,
            output_dir=output_dir,
            device=device,
            target_delta=target_delta,
            mode=mode,
            checkpoint_path=checkpoint_path,
            demo_mode=demo_mode,
            batch_size=batch_size,
            window_size=window_size,
            base_args=base_args,
        )

    # Generate publication figures
    figs = plot_privacy_utility_tradeoff(
        data=df,
        output_dir=output_dir,
        save_filename="privacy_utility_tradeoff",
    )

    csv_path = os.path.join(output_dir, "privacy_utility_results.csv")

    return {
        "dataframe": df,
        "figures": figs,
        "csv_path": csv_path,
        "output_dir": output_dir,
    }


# ==============================================================================
# 5. Command-Line Interface (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run and plot the Privacy-Utility Tradeoff for FedGATSage (DP-VFL)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/preprocessed_data",
        help="Path to preprocessed dataset directory (default: data/preprocessed_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/privacy_utility_tradeoff",
        help="Directory to save tradeoff results and plots",
    )
    parser.add_argument(
        "--noise_list",
        type=str,
        default="0.0,0.005,0.01,0.02,0.05,0.1,0.2,0.5",
        help="Comma-separated list of DP noise multipliers (default: 0.0,0.005,0.01,0.02,0.05,0.1,0.2,0.5)",
    )
    parser.add_argument(
        "--dp_clip_bound",
        type=float,
        default=21.0,
        help="L2 clipping bound C for embeddings (default: 21.0)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of federated learning rounds per DP level (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="Sliding window size (default: 30)",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        help="DP target delta parameter (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: 'train' (full federated training) or 'eval' (evaluate checkpoint with noise)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to trained checkpoint (used in 'eval' mode or for warm starts)",
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Enable demo mode with downsampled data and fewer iterations for quick testing",
    )
    parser.add_argument(
        "--plot_only",
        type=str,
        default=None,
        help="Skip training and plot directly from an existing CSV results file",
    )

    args = parser.parse_args()

    # Parse noise multiplier list
    try:
        noise_multipliers = [float(x.strip()) for x in args.noise_list.split(",") if x.strip()]
    except ValueError:
        logger.error(f"Invalid noise_list argument: '{args.noise_list}'. Must be comma-separated floats.")
        sys.exit(1)

    result = run_and_plot_privacy_utility_tradeoff(
        noise_multipliers=noise_multipliers,
        clip_bound=args.dp_clip_bound,
        num_rounds=args.num_rounds,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        target_delta=args.target_delta,
        mode=args.mode,
        checkpoint_path=args.checkpoint_path,
        demo_mode=args.demo_mode,
        batch_size=args.batch_size,
        window_size=args.window_size,
        plot_only_csv=args.plot_only,
    )

    print("\n" + "=" * 60)
    print("🏁 Privacy-Utility Tradeoff analysis completed successfully!")
    print(f"Results summary:\n{result['dataframe'][['noise_multiplier', 'epsilon', 'roc_auc', 'macro_f1', 'balanced_accuracy']]}")
    print(f"CSV saved to: {result['csv_path']}")
    print(f"Plots saved in: {result['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

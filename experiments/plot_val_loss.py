#!/usr/bin/env python3
"""
Plot validation loss (and training loss) per federated learning round.
Can parse directly from experiment logs or checkpoint files.
"""

import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_losses_from_log(log_path: str):
    """Parse round numbers, training losses, and validation losses from experiment.log."""
    rounds = []
    train_losses = []
    val_losses = []

    # Pattern matches: Round <num> completed ... Train Loss: <val> ... Val Loss: <val>
    pattern = re.compile(
        r"Round\s+(\d+)\s+completed.*?Train Loss:\s+([\d\.]+).*?Val Loss:\s+([\d\.]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                r = int(match.group(1))
                t_loss = float(match.group(2))
                v_loss = float(match.group(3))
                rounds.append(r)
                train_losses.append(t_loss)
                val_losses.append(v_loss)

    return rounds, train_losses, val_losses


def plot_validation_loss(
    rounds,
    val_losses,
    train_losses=None,
    title="Validation Loss per Federated Round",
    save_path="results/validation_loss_per_round.png",
):
    """Generate and save publication-quality plot of validation loss per round."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot validation loss
    ax.plot(
        rounds,
        val_losses,
        color="#1f77b4",
        marker="o",
        markersize=7,
        linewidth=2.5,
        label="Validation Loss",
        zorder=3,
    )
    ax.fill_between(rounds, val_losses, color="#1f77b4", alpha=0.1, zorder=2)

    # Optionally plot training loss
    if train_losses is not None and len(train_losses) == len(rounds):
        ax.plot(
            rounds,
            train_losses,
            color="#ff7f0e",
            marker="s",
            markersize=6,
            linewidth=2.0,
            linestyle="--",
            label="Training Loss",
            zorder=3,
        )

    # Highlight best validation loss
    min_idx = int(np.argmin(val_losses))
    min_round = rounds[min_idx]
    min_val = val_losses[min_idx]

    ax.scatter([min_round], [min_val], color="#2ca02c", s=130, zorder=4, edgecolor="black", linewidth=1.5)
    
    y_range = max(val_losses) - min(val_losses) if len(val_losses) > 1 else 1.0
    ax.annotate(
        f"Best Val: Round {min_round} ({min_val:.4f})",
        xy=(min_round, min_val),
        xytext=(min_round - max(1, len(rounds) * 0.16), min_val + y_range * 0.3),
        arrowprops=dict(
            facecolor="#2ca02c",
            edgecolor="#2ca02c",
            arrowstyle="->",
            lw=1.5,
            connectionstyle="arc3,rad=0.1",
        ),
        fontsize=11,
        fontweight="bold",
        color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#a5d6a7", alpha=0.9),
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Federated Learning Round", fontsize=12, labelpad=10)
    ax.set_ylabel("Loss", fontsize=12, labelpad=10)
    ax.set_xticks(rounds)
    ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot validation loss per federated round.")
    parser.add_argument(
        "--log_path",
        type=str,
        default="results/experiment.log",
        help="Path to experiment.log file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--compare_train",
        action="store_true",
        help="Include training loss curve on the same plot.",
    )
    args = parser.parse_args()

    rounds, train_losses, val_losses = parse_losses_from_log(args.log_path)
    if not rounds:
        print(f"No completed rounds found in {args.log_path}")
        return

    # If log contains multiple runs, select the longest continuous run (e.g. 1..N)
    runs = []
    current_run = {"rounds": [], "train": [], "val": []}
    for r, t, v in zip(rounds, train_losses, val_losses):
        if current_run["rounds"] and r <= current_run["rounds"][-1]:
            runs.append(current_run)
            current_run = {"rounds": [], "train": [], "val": []}
        current_run["rounds"].append(r)
        current_run["train"].append(t)
        current_run["val"].append(v)
    if current_run["rounds"]:
        runs.append(current_run)

    # Pick the longest run (e.g. 20 rounds)
    longest_run = max(runs, key=lambda x: len(x["rounds"]))
    r_list = longest_run["rounds"]
    t_list = longest_run["train"]
    v_list = longest_run["val"]

    # 1. Validation Loss only plot
    val_path = os.path.join(args.output_dir, "validation_loss_per_round.png")
    plot_validation_loss(r_list, v_list, None, "Validation Loss per Federated Round", val_path)

    # 2. Train vs Val Loss comparison plot
    tv_path = os.path.join(args.output_dir, "train_vs_val_loss.png")
    plot_validation_loss(r_list, v_list, t_list, "Training Loss vs. Validation Loss per Federated Round", tv_path)


if __name__ == "__main__":
    main()

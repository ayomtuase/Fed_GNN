#!/usr/bin/env python3
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# 1. Data extracted from results/experiment.log (20-round main experiment)
rounds_20 = list(range(1, 21))
train_loss_20 = [
    1.3411, 2.8822, 1.3140, 1.4385, 1.2966, 1.3226, 1.2814, 1.2346, 1.2630, 1.2033,
    1.1707, 1.1104, 1.1331, 1.1351, 1.1298, 1.0549, 1.1437, 1.1107, 1.0125, 1.0420
]
val_loss_20 = [
    2.7468, 1.2782, 1.3563, 1.2666, 1.2976, 1.2868, 1.2153, 1.1679, 1.1141, 1.1212,
    1.0601, 0.9875, 0.9502, 1.2567, 1.0445, 0.9525, 0.9432, 0.9932, 1.2333, 0.8797
]

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Output directories
os.makedirs("results", exist_ok=True)
artifact_dir = "/Users/macintosh/.gemini/antigravity-ide/brain/3996166f-ffc6-4bd4-8cc8-288c84410df8"
os.makedirs(artifact_dir, exist_ok=True)

# -------------------------------------------------------------
# PLOT 1: Dedicated Validation Loss per Round
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.plot(
    rounds_20, val_loss_20,
    color="#1f77b4",
    marker="o",
    markersize=7,
    linewidth=2.5,
    label="Validation Loss",
    zorder=3
)
ax.fill_between(rounds_20, val_loss_20, color="#1f77b4", alpha=0.1, zorder=2)

# Highlight minimum validation loss
min_idx = int(np.argmin(val_loss_20))
min_round = rounds_20[min_idx]
min_val = val_loss_20[min_idx]

ax.scatter([min_round], [min_val], color="#2ca02c", s=130, zorder=4, edgecolor="black", linewidth=1.5)
ax.annotate(
    f"Best: Round {min_round} ({min_val:.4f})",
    xy=(min_round, min_val),
    xytext=(min_round - 3.2, min_val + 0.55),
    arrowprops=dict(
        facecolor="#2ca02c",
        edgecolor="#2ca02c",
        arrowstyle="->",
        lw=1.5,
        connectionstyle="arc3,rad=0.1"
    ),
    fontsize=11,
    fontweight="bold",
    color="#1b5e20",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#a5d6a7", alpha=0.9)
)

ax.set_title("Validation Loss per Federated Round", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Federated Learning Round", fontsize=12, labelpad=10)
ax.set_ylabel("Validation Loss", fontsize=12, labelpad=10)
ax.set_xticks(rounds_20)
ax.set_xlim(0.5, 20.5)
ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)

plt.tight_layout()
val_plot_path = "results/validation_loss_per_round.png"
fig.savefig(val_plot_path, dpi=300)
shutil.copy(val_plot_path, os.path.join(artifact_dir, "validation_loss_per_round.png"))
plt.close(fig)
print(f"Saved {val_plot_path}")

# -------------------------------------------------------------
# PLOT 2: Train Loss vs Validation Loss Comparison
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.plot(
    rounds_20, train_loss_20,
    color="#ff7f0e",
    marker="s",
    markersize=6,
    linewidth=2.0,
    linestyle="--",
    label="Training Loss",
    zorder=3
)
ax.plot(
    rounds_20, val_loss_20,
    color="#1f77b4",
    marker="o",
    markersize=7,
    linewidth=2.5,
    label="Validation Loss",
    zorder=3
)

ax.scatter([min_round], [min_val], color="#2ca02c", s=130, zorder=4, edgecolor="black", linewidth=1.5)
ax.annotate(
    f"Best Val: Round {min_round} ({min_val:.4f})",
    xy=(min_round, min_val),
    xytext=(min_round - 3.5, min_val + 0.65),
    arrowprops=dict(
        facecolor="#2ca02c",
        edgecolor="#2ca02c",
        arrowstyle="->",
        lw=1.5,
        connectionstyle="arc3,rad=0.1"
    ),
    fontsize=11,
    fontweight="bold",
    color="#1b5e20",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#a5d6a7", alpha=0.9)
)

ax.set_title("Training Loss vs. Validation Loss per Federated Round", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Federated Learning Round", fontsize=12, labelpad=10)
ax.set_ylabel("Loss", fontsize=12, labelpad=10)
ax.set_xticks(rounds_20)
ax.set_xlim(0.5, 20.5)
ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)

plt.tight_layout()
train_val_path = "results/train_vs_val_loss.png"
fig.savefig(train_val_path, dpi=300)
shutil.copy(train_val_path, os.path.join(artifact_dir, "train_vs_val_loss.png"))
plt.close(fig)
print(f"Saved {train_val_path}")

# -------------------------------------------------------------
# PLOT 3: Recent Profile Run (3 rounds) for completeness
# -------------------------------------------------------------
rounds_prof = [1, 2, 3]
train_loss_prof = [0.6012, 0.1378, 0.1260]
val_loss_prof = [0.1040, 0.0831, 0.0806]

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(rounds_prof, val_loss_prof, color="#2b5c8f", marker="o", markersize=8, linewidth=2.5, label="Validation Loss")
ax.plot(rounds_prof, train_loss_prof, color="#e26d5c", marker="s", markersize=7, linewidth=2.0, linestyle="--", label="Training Loss")
ax.set_title("Validation & Training Loss (Profile Run, 3 Rounds)", fontsize=13, fontweight="bold")
ax.set_xlabel("Federated Learning Round", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_xticks(rounds_prof)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
plt.tight_layout()
prof_path = "results/val_loss_profile_run.png"
fig.savefig(prof_path, dpi=300)
shutil.copy(prof_path, os.path.join(artifact_dir, "val_loss_profile_run.png"))
plt.close(fig)
print(f"Saved {prof_path}")

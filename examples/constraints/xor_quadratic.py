"""
Comparison of unconstrained Adam vs. projected SGD with det(M) = 0 constraint
on the XOR dataset using a 2-layer PNN with σ(z) = z².

Both models start from identical random initialisation to isolate the effect
of the constraint.  After training, both models are verified using homotopy
continuation: the constrained model should yield fewer real critical points
on the decision boundary (2 vs 4 for the unconstrained model).
"""

import copy
import sys
from pathlib import Path

# Resolve the project root from this file so the script runs from any cwd.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.data import XORDataset
from src.pnn import PolynomialNeuralNetwork
from src.utils import train_epochs, train_epochs_projected, save_model
from src.hc import compute_robust_radius

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
INPUT_DIM = 2
HIDDEN_DIM = 32
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3
XY_RANGE = 1.0
MARGIN = 0.1

# Query points: one per quadrant, well inside each region
XI_LIST = [
    [0.5, 0.5],  # Q1 — class 0
    [-0.5, -0.5],  # Q3 — class 0
    [-0.5, 0.5],  # Q2 — class 1
    [0.5, -0.5],  # Q4 — class 1
]

torch.manual_seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
dataset = XORDataset(size=1000, xy_range=XY_RANGE, margin=MARGIN, seed=SEED)
n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train
train_set, val_set = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))

# ── Models (identical initialisation) ────────────────────────────────────────
torch.manual_seed(SEED)
model_unc = PolynomialNeuralNetwork(
    input_dim=INPUT_DIM,
    output_dim=2,
    hidden_dims=[HIDDEN_DIM],
    act_degree=2,
    homogeneous=True,
    trainable=False,  # fixed σ(z) = z²
    bias=True,
)
model_con = copy.deepcopy(model_unc)

# ── 1. Unconstrained (Adam) ───────────────────────────────────────────────────
print("=" * 60)
print("Training unconstrained model (Adam)")
print("=" * 60)
history_unc = train_epochs(
    model=model_unc,
    train_loader=train_loader,
    num_epochs=NUM_EPOCHS,
    optimizer_type="adam",
    learning_rate=LR,
    val_loader=val_loader,
    verbose=True,
)

# ── 2. Projected SGD with det(M) = 0 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Training constrained model (Projected SGD, det(M) = 0)")
print("=" * 60)
constraint_fn = model_con.make_det_M_constraint(target_det=0.0)
history_con = train_epochs_projected(
    model=model_con,
    train_loader=train_loader,
    constraint_fn=constraint_fn,
    num_epochs=NUM_EPOCHS,
    lr=LR,
    val_loader=val_loader,
    verbose=True,
)

# ── Final det(M) values ───────────────────────────────────────────────────────
model_unc.cpu()
model_con.cpu()
for label, model, history in [
    ("Unconstrained", model_unc, history_unc),
    ("Constrained  ", model_con, history_con),
]:
    M, A = model.compute_augmented_matrix()
    print(
        f"{label} — det(A): {torch.linalg.det(A).item():.4e}"
        f"  det(M): {torch.linalg.det(M).item():.4e}"
        f"  val acc: {history['val_acc'][-1]:.2f}%"
    )

# ── Build decision-boundary grids ─────────────────────────────────────────────
PAD = 0.2
grid_size = 400
lim = XY_RANGE + PAD
xs = np.linspace(-lim, lim, grid_size)
ys = np.linspace(-lim, lim, grid_size)
XX, YY = np.meshgrid(xs, ys)
grid_pts = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)

diffs = {}
for key, model in [("unc", model_unc), ("con", model_con)]:
    model.eval()
    with torch.no_grad():
        logits = model(grid_pts)
    diffs[key] = (logits[:, 0] - logits[:, 1]).numpy().reshape(grid_size, grid_size)

all_features = np.array([dataset[i][0] for i in range(len(dataset))])
all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
colors = ["tab:blue", "tab:red"]
markers = ["o", "s"]
class_names = ["Class 0 (Q1 & Q3)", "Class 1 (Q2 & Q4)"]

# ── Figure 1: decision boundaries + training curves ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("XOR dataset — PNN with σ(z) = z²", fontsize=13)

for (key, title), ax in zip(
    [("unc", "Unconstrained (Adam)"), ("con", "Projected SGD  (det(M) = 0)")],
    axes[:2],
):
    diff = diffs[key]
    vmax = np.abs(diff).max()
    ax.imshow(
        diff,
        extent=[-lim, lim, -lim, lim],
        origin="lower",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        alpha=0.6,
    )
    ax.contour(XX, YY, diff, levels=[0], colors="black", linewidths=2.0)
    for c in range(2):
        mask = all_labels == c
        ax.scatter(
            all_features[mask, 0],
            all_features[mask, 1],
            c=colors[c],
            marker=markers[c],
            s=10,
            alpha=0.6,
            label=class_names[c],
            edgecolors="none",
        )
    # Mark query points
    xi_arr = np.array(XI_LIST)
    ax.scatter(
        xi_arr[:, 0],
        xi_arr[:, 1],
        c="gold",
        marker="*",
        s=120,
        zorder=5,
        label="Query points",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)

epochs = range(1, NUM_EPOCHS + 1)
axes[2].plot(epochs, history_unc["val_acc"], label="Unconstrained val")
axes[2].plot(epochs, history_con["val_acc"], label="Constrained val", linestyle="--")
axes[2].plot(epochs, history_unc["train_acc"], alpha=0.4, label="Unconstrained train")
axes[2].plot(
    epochs,
    history_con["train_acc"],
    alpha=0.4,
    linestyle="--",
    label="Constrained train",
)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy (%)")
axes[2].set_title("Training curves")
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig("xor_quadratic.png", dpi=150)
print("Plot saved to xor_quadratic.png")
plt.show()

# ── Save both models ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving models")
print("=" * 60)
path_unc = save_model(
    model_unc,
    metadata={
        "description": "XOR unconstrained Adam",
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
    },
)
path_con = save_model(
    model_con,
    metadata={
        "description": "XOR projected SGD det(M)=0",
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
    },
)
print(f"Unconstrained saved to : {path_unc}")
print(f"Constrained   saved to : {path_con}")

# ── Verification via homotopy continuation ────────────────────────────────────
print("\n" + "=" * 60)
print("Running homotopy continuation verifier — unconstrained model")
print("=" * 60)
res_unc = compute_robust_radius(
    path_unc,
    XI_LIST,
    verbose=True,
    save_results=True,
    save_detailed=True,
    num_threads=1,
)

print("\n" + "=" * 60)
print("Running homotopy continuation verifier — constrained model")
print("=" * 60)
res_con = compute_robust_radius(
    path_con,
    XI_LIST,
    verbose=True,
    save_results=True,
    save_detailed=True,
    num_threads=1,
)


# ── Parse detailed results ────────────────────────────────────────────────────
def parse_verification(res):
    """Return (net_wall_times, n_real_roots, n_total_roots) arrays."""
    timing = res["timing"]
    net_times = timing["instance_wall_s"] - timing["instance_compile_s"]
    detailed = Path(res["detailed_dir"])
    n_real, n_total = [], []
    for f in sorted(detailed.glob("point_*.npz")):
        d = np.load(f)
        n_real.append(int(d["boundary_num_real"][0]))
        n_total.append(int(d["boundary_num_total"][0]))
    return net_times, np.array(n_real), np.array(n_total)


times_unc, real_unc, total_unc = parse_verification(res_unc)
times_con, real_con, total_con = parse_verification(res_con)

# ── Print comparison table ────────────────────────────────────────────────────
col = 12
print("\n" + "=" * 76)
print(f"{'Verification Comparison':^76}")
print("=" * 76)
print(f"{'Point':^14} | {'Unconstrained':^28} | {'Constrained det(M)=0':^28}")
print(
    f"{'':^14} | {'real':^8} {'total':^8} {'time(s)':^10} | {'real':^8} {'total':^8} {'time(s)':^10}"
)
print("-" * 76)
for i, xi in enumerate(XI_LIST):
    label = f"({xi[0]:+.1f},{xi[1]:+.1f})"
    print(
        f"{label:^14} | {real_unc[i]:^8} {total_unc[i]:^8} {times_unc[i]:^10.3f} |"
        f" {real_con[i]:^8} {total_con[i]:^8} {times_con[i]:^10.3f}"
    )
print("=" * 76)
print("(net time excludes Julia JIT compile time)")

# ── Figure 2: verification comparison ────────────────────────────────────────
x = np.arange(len(XI_LIST))
w = 0.35
labels = [f"({xi[0]:+.1f},{xi[1]:+.1f})" for xi in XI_LIST]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Verification: Unconstrained vs. Constrained det(M) = 0", fontsize=13)

axes2[0].bar(x - w / 2, real_unc, w, label="Unconstrained")
axes2[0].bar(x + w / 2, real_con, w, label="Constrained det(M)=0")
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(labels, rotation=15)
axes2[0].set_ylabel("# real critical points on boundary")
axes2[0].set_title("Real Roots on Decision Boundary")
axes2[0].axhline(
    4, color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.5, label="Expected: 4"
)
axes2[0].axhline(
    2, color="tab:orange", linestyle="--", linewidth=0.8, alpha=0.5, label="Expected: 2"
)
axes2[0].legend(fontsize=8)

axes2[1].bar(x - w / 2, times_unc, w, label="Unconstrained")
axes2[1].bar(x + w / 2, times_con, w, label="Constrained det(M)=0")
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(labels, rotation=15)
axes2[1].set_ylabel("Net computation time (s)")
axes2[1].set_title("Verification Time per Query Point")
axes2[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("xor_verification.png", dpi=150)
print("\nVerification plot saved to xor_verification.png")
plt.show()

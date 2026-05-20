"""
Comparison of unconstrained SGD vs. projected SGD with det(A) = 0 constraint
on the Sinusoid dataset using a 2-layer PNN with σ(z) = z².

Both models start from identical random initialisation to isolate the effect
of the constraint.  After training, both models are verified using homotopy
continuation: the constrained model should have fewer real critical points on
the decision boundary.
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

from src.data import SinusoidDataset
from src.pnn import PolynomialNeuralNetwork
from src.utils import train_epochs, train_epochs_projected, save_model
from src.hc import compute_robust_radius

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
INPUT_DIM = 2
HIDDEN_DIM = 16
NUM_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-2

# Query points for verification (spread across the sinusoid domain)
XI_LIST = [
    [0.0, 0.0],  # near boundary: sin(0)  = 0.00
    [1.0, 0.8],  # near boundary: sin(1)  ≈ 0.84
    [-1.0, -0.8],  # near boundary: sin(-1) ≈ -0.84
    [2.0, 0.9],  # near boundary: sin(2)  ≈ 0.91
    [-2.0, -0.9],  # near boundary: sin(-2) ≈ -0.91
]

torch.manual_seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
dataset = SinusoidDataset(size=1000, seed=SEED)
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
# homogeneous=True is required for correct Julia reconstruction: Julia reads
# activations.0.coeffs=[1.0] and computes 1.0 * z^2 via the homogeneous branch.
# With homogeneous=False Julia's Horner branch would expect degree+1=3 coefficients
# but the non-trainable buffer only stores 1 value.
torch.manual_seed(SEED)
model_sgd = PolynomialNeuralNetwork(
    input_dim=INPUT_DIM,
    output_dim=2,
    hidden_dims=[HIDDEN_DIM],
    act_degree=2,
    homogeneous=True,
    trainable=False,
    bias=True,
)
model_proj = copy.deepcopy(model_sgd)  # same weights, separate copy


# ── Epoch callback: print M at epoch 1 and the final epoch ───────────────────
def make_M_callback(label: str, num_epochs: int):
    def callback(model, epoch):
        if epoch in (1, num_epochs):
            model.cpu()
            M, A = model.compute_augmented_matrix()
            print(f"\n[{label} | epoch {epoch}/{num_epochs}]")
            print(f"  A =\n{A}")
            print(f"  det(A) = {torch.linalg.det(A).item():.6e}")
            print(f"  M =\n{M}")
            # Move back to GPU if available, matching what the trainer expects
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

    return callback


# ── 1. Unconstrained SGD ──────────────────────────────────────────────────────
print("=" * 60)
print("Training unconstrained model (SGD)")
print("=" * 60)
history_sgd = train_epochs(
    model=model_sgd,
    train_loader=train_loader,
    num_epochs=NUM_EPOCHS,
    optimizer_type="sgd",
    learning_rate=LR,
    val_loader=val_loader,
    verbose=True,
    epoch_callback=make_M_callback("SGD", NUM_EPOCHS),
)

# ── 2. Projected SGD with det(A) = 0 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Training constrained model (Projected SGD, det(A) = 0)")
print("=" * 60)
constraint_fn = model_proj.make_det_A_constraint(target_det=0.0)
history_proj = train_epochs_projected(
    model=model_proj,
    train_loader=train_loader,
    constraint_fn=constraint_fn,
    num_epochs=NUM_EPOCHS,
    lr=LR,
    val_loader=val_loader,
    verbose=True,
    epoch_callback=make_M_callback("Projected SGD", NUM_EPOCHS),
)

# ── Final det(A) values ───────────────────────────────────────────────────────
_, A_sgd = model_sgd.compute_augmented_matrix()
_, A_proj = model_proj.compute_augmented_matrix()
print(f"\nFinal det(A) — unconstrained : {torch.linalg.det(A_sgd).item():.6e}")
print(f"Final det(A) — projected SGD : {torch.linalg.det(A_proj).item():.6e}")

# ── Training comparison plot ──────────────────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Sinusoid dataset — SGD vs. Projected SGD (det(A) = 0)", fontsize=13)

axes[0, 0].plot(epochs, history_sgd["train_acc"], label="SGD")
axes[0, 0].plot(epochs, history_proj["train_acc"], label="Projected SGD")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy (%)")
axes[0, 0].set_title("Training Accuracy")
axes[0, 0].legend()

axes[0, 1].plot(epochs, history_sgd["val_acc"], label="SGD")
axes[0, 1].plot(epochs, history_proj["val_acc"], label="Projected SGD")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].set_title("Validation Accuracy")
axes[0, 1].legend()

axes[1, 0].plot(epochs, history_sgd["train_loss"], label="SGD")
axes[1, 0].plot(epochs, history_proj["train_loss"], label="Projected SGD")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].set_title("Training Loss")
axes[1, 0].legend()

axes[1, 1].semilogy(epochs, history_proj["constraint_norm"], color="tab:orange")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("|det(A)|")
axes[1, 1].set_title("Constraint Residual (Projected SGD)")

plt.tight_layout()
plt.savefig("sinusoid_det_constraint.png", dpi=150)
print("Training plot saved to sinusoid_det_constraint.png")
plt.show()

# ── Logit heatmaps with decision boundary ─────────────────────────────────────
x_range = (-20.0, 20.0)
y_range = (-20.0, 20.0)
grid_size = 400

x_grid = np.linspace(x_range[0], x_range[1], grid_size)
y_grid = np.linspace(y_range[0], y_range[1], grid_size)
XX, YY = np.meshgrid(x_grid, y_grid)
grid_pts = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)

# Move to CPU for grid evaluation (train_epochs may have left them on GPU)
model_sgd.cpu()
model_proj.cpu()

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Logit difference  f(x) = logit[0] − logit[1]", fontsize=13)

for ax, model, title in [
    (axes3[0], model_sgd, "Unconstrained SGD"),
    (axes3[1], model_proj, "Projected SGD  (det(A) = 0)"),
]:
    model.eval()
    with torch.no_grad():
        logits = model(grid_pts)
    diff = (logits[:, 0] - logits[:, 1]).numpy().reshape(grid_size, grid_size)

    vmax = np.abs(diff).max()
    im = ax.imshow(
        diff,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin="lower",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    # Learned decision boundary: f(x) = 0
    ax.contour(XX, YY, diff, levels=[0], colors="black", linewidths=1.5)
    # True sinusoidal boundary: y = sin(x)
    ax.plot(x_grid, np.sin(x_grid), "k--", linewidth=1.2, label="y = sin(x)")

    fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("sinusoid_logit_heatmap.png", dpi=150)
print("Heatmap saved to sinusoid_logit_heatmap.png")
plt.show()

# ── Save both models for verification ────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving models")
print("=" * 60)
path_sgd = save_model(
    model_sgd,
    metadata={
        "description": "sinusoid unconstrained SGD",
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
    },
)
path_proj = save_model(
    model_proj,
    metadata={
        "description": "sinusoid projected SGD det(A)=0",
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
    },
)
print(f"Unconstrained saved to : {path_sgd}")
print(f"Constrained   saved to : {path_proj}")

# ── Verification via homotopy continuation ────────────────────────────────────
print("\n" + "=" * 60)
print("Running homotopy continuation verifier — unconstrained model")
print("=" * 60)
res_sgd = compute_robust_radius(
    path_sgd,
    XI_LIST,
    verbose=True,
    save_results=True,
    save_detailed=True,
    num_threads=1,
)

print("\n" + "=" * 60)
print("Running homotopy continuation verifier — constrained model")
print("=" * 60)
res_proj = compute_robust_radius(
    path_proj,
    XI_LIST,
    verbose=True,
    save_results=True,
    save_detailed=True,
    num_threads=1,
)


# ── Parse detailed results ────────────────────────────────────────────────────
def parse_verification(res):
    """Return per-point (net_time, n_real_roots, n_total_roots)."""
    timing = res["timing"]
    net_times = timing["instance_wall_s"] - timing["instance_compile_s"]
    detailed = Path(res["detailed_dir"])
    n_real, n_total = [], []
    for f in sorted(detailed.glob("point_*.npz")):
        d = np.load(f)
        n_real.append(int(d["boundary_num_real"][0]))
        n_total.append(int(d["boundary_num_total"][0]))
    return net_times, np.array(n_real), np.array(n_total)


times_sgd, real_sgd, total_sgd = parse_verification(res_sgd)
times_proj, real_proj, total_proj = parse_verification(res_proj)

# ── Print comparison table ────────────────────────────────────────────────────
col = 12
print("\n" + "=" * 76)
print(f"{'Verification Comparison':^76}")
print("=" * 76)
print(f"{'Point':^14} | {'SGD':^28} | {'Projected SGD':^28}")
print(
    f"{'':^14} | {'real':^8} {'total':^8} {'time(s)':^10} | {'real':^8} {'total':^8} {'time(s)':^10}"
)
print("-" * 76)
for i, xi in enumerate(XI_LIST):
    label = f"({xi[0]:+.1f},{xi[1]:+.1f})"
    print(
        f"{label:^14} | {real_sgd[i]:^8} {total_sgd[i]:^8} {times_sgd[i]:^10.3f} |"
        f" {real_proj[i]:^8} {total_proj[i]:^8} {times_proj[i]:^10.3f}"
    )
print("=" * 76)
print("(net time excludes Julia JIT compile time)")

# ── Verification comparison plot ──────────────────────────────────────────────
x = np.arange(len(XI_LIST))
w = 0.35
labels = [f"({xi[0]:+.1f},{xi[1]:+.1f})" for xi in XI_LIST]

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Verification: SGD vs. Projected SGD (det(A) = 0)", fontsize=13)

axes2[0].bar(x - w / 2, real_sgd, w, label="SGD")
axes2[0].bar(x + w / 2, real_proj, w, label="Projected SGD")
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(labels, rotation=15)
axes2[0].set_ylabel("# real critical points")
axes2[0].set_title("Real Roots on Decision Boundary")
axes2[0].legend()

axes2[1].bar(x - w / 2, times_sgd, w, label="SGD")
axes2[1].bar(x + w / 2, times_proj, w, label="Projected SGD")
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(labels, rotation=15)
axes2[1].set_ylabel("Net computation time (s)")
axes2[1].set_title("Verification Time per Query Point")
axes2[1].legend()

plt.tight_layout()
plt.savefig("sinusoid_verification.png", dpi=150)
print("\nVerification plot saved to sinusoid_verification.png")
plt.show()

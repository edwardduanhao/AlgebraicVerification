"""
Unconstrained Adam training with det(M)² regularization on the XOR dataset.

Trains one model per λ ∈ {0, 1, 10, 100}, all from identical initialisation.
The loss is:  L = CrossEntropy + λ · det(M)²
where M is the (n+1)×(n+1) augmented matrix of the quadratic decision boundary.

Produces a 2×2 panel of decision boundaries (one per λ).
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
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm.auto import trange

from src.data import XORDataset
from src.pnn import PolynomialNeuralNetwork

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
INPUT_DIM = 2
HIDDEN_DIM = 32
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LR = 1e-3
XY_RANGE = 1.0
MARGIN = 0.1

LAMBDAS = [0, 0.1]

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

# ── Reference model (shared initialisation) ───────────────────────────────────
torch.manual_seed(SEED)
model_ref = PolynomialNeuralNetwork(
    input_dim=INPUT_DIM,
    output_dim=2,
    hidden_dims=[HIDDEN_DIM],
    act_degree=2,
    homogeneous=True,
    trainable=False,  # fixed σ(z) = z²
    bias=True,
)


def compute_det_M(model: PolynomialNeuralNetwork) -> torch.Tensor:
    """Compute det(M) for a 2-layer PNN with fixed quadratic activation."""
    W1 = model.layers[0].weight  # (h, n)
    b1 = model.layers[0].bias  # (h,)
    W2 = model.layers[1].weight  # (2, h)
    b2 = model.layers[1].bias  # (2,)

    d = W2[0, :] - W2[1, :]  # (h,)
    D = torch.diag(d)

    A = W1.T @ D @ W1  # (n, n)
    b = 2 * W1.T @ D @ b1  # (n,)
    c = b1 @ D @ b1 + b2[0] - b2[1]  # scalar

    M = torch.cat(
        [
            torch.cat([A, b.unsqueeze(1)], dim=1),
            torch.cat([b.unsqueeze(0), c.reshape(1, 1)], dim=1),
        ],
        dim=0,
    )  # (n+1, n+1)
    return torch.linalg.det(M)


def train_with_det_reg(model, lam: float) -> dict:
    """Train model with cross-entropy + λ·det(M)² loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "det_M": []}

    epoch_iter = trange(NUM_EPOCHS, desc=f"λ={lam}")
    for _ in epoch_iter:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)

            reg = torch.tensor(0.0, device=device)
            if lam > 0:
                det_M = compute_det_M(model)
                reg = lam * det_M**2

            loss = ce_loss + reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
            det_M_val = compute_det_M(model).item()

        val_acc = 100.0 * val_correct / val_total
        history["val_acc"].append(val_acc)
        history["det_M"].append(det_M_val)

        epoch_iter.set_postfix(
            train_acc=f"{train_acc:.1f}%",
            val_acc=f"{val_acc:.1f}%",
            det_M=f"{det_M_val:.3e}",
        )

    model.cpu()
    return history


# ── Train one model per λ ─────────────────────────────────────────────────────
models = {}
histories = {}
for lam in LAMBDAS:
    print(f"\n{'=' * 60}")
    print(f"Training with λ = {lam}")
    print("=" * 60)
    model = copy.deepcopy(model_ref)
    history = train_with_det_reg(model, lam)
    models[lam] = model
    histories[lam] = history
    M, _ = model.compute_augmented_matrix()
    print(
        f"  Final det(M) = {torch.linalg.det(M).item():.4e}"
        f"  val acc = {history['val_acc'][-1]:.1f}%"
    )

# ── Decision-boundary grid ────────────────────────────────────────────────────
PAD = 0.2
grid_size = 400
lim = XY_RANGE + PAD
xs = np.linspace(-lim, lim, grid_size)
ys = np.linspace(-lim, lim, grid_size)
XX, YY = np.meshgrid(xs, ys)
grid_pts = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)

all_features = np.array([dataset[i][0] for i in range(len(dataset))])
all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
colors = ["tab:blue", "tab:red"]
markers = ["o", "s"]
class_names = ["Class 0", "Class 1"]

# ── Plot: 1×2 decision boundaries ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

for ax, lam in zip(axes, LAMBDAS):
    model = models[lam]
    model.eval()
    with torch.no_grad():
        logits = model(grid_pts)
    diff = (logits[:, 0] - logits[:, 1]).numpy().reshape(grid_size, grid_size)

    M, _ = model.compute_augmented_matrix()
    det_val = torch.linalg.det(M).item()
    val_acc = histories[lam]["val_acc"][-1]

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
            s=8,
            alpha=0.5,
            label=class_names[c],
            edgecolors="none",
        )

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$x_1$", fontsize=11)
    ax.set_ylabel("$x_2$", fontsize=11)
    ax.set_title(
        f"$\\lambda = {lam}$\ndet$(M)={det_val:.2e}$",
        fontsize=10,
    )
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

plt.tight_layout(pad=1.5)
out_path = "xor_det_regularizer.pdf"
plt.savefig(out_path)
print(f"\nPlot saved to {out_path}")
plt.show()

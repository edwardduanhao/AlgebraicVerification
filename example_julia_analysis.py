"""
Example: Complete workflow from Python training to Julia analysis.

This script demonstrates:
1. Training and saving a model in Python
2. Running Julia analysis from Python
3. Loading and visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Train and save model
from src.pnn import PolynomialNeuralNetwork
from src.utils.utils import save_model

print("=" * 60)
print("Step 1: Training and saving model")
print("=" * 60)

# Create a simple model
model = PolynomialNeuralNetwork(
    input_dim=2,
    output_dim=2,
    hidden_dims=[3, 4],
    act_degree=2,
    homogeneous=False,
    bias=True,
)

# Save model (in real workflow, you would train it first)
exp_dir = save_model(
    model,
    metadata={
        "description": "Example binary classifier for Julia analysis",
        "dataset": "synthetic_2d",
        "note": "This is a demonstration model",
    },
)

print(f"\n✓ Model saved to: {exp_dir}")

# Step 2: Run Julia analysis
print("\n" + "=" * 60)
print("Step 2: Running Julia analysis")
print("=" * 60)

from src.hc import compute_robust_radius, verify_experiment

# First, verify the experiment is valid
status = verify_experiment(exp_dir)
print(f"\nExperiment validation:")
print(f"  Valid: {status['valid']}")
print(f"  Path: {status['resolved_path']}")

if not status["valid"]:
    print("\nErrors found:")
    for error in status["errors"]:
        print(f"  - {error}")
    exit(1)

# Define test points
xi_list = [
    [0.5, 0.5],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.25, 0.75],
]

print(f"\nAnalyzing {len(xi_list)} test points...")

# Run robust radius computation
try:
    results = compute_robust_radius(exp_dir, xi_list, verbose=True, save_results=True)

    print(f"\n✓ Analysis complete!")
    print(f"✓ Results saved to: {results['save_path']}")

except Exception as e:
    print(f"\n✗ Analysis failed: {e}")
    print(
        "\nNote: Make sure Julia and required packages (HomotopyContinuation, NPZ) are installed."
    )
    exit(1)

# Step 3: Load and visualize results
print("\n" + "=" * 60)
print("Step 3: Visualizing results")
print("=" * 60)

from src.hc import load_robust_radius_results

# Load results (could also use 'results' from above)
results = load_robust_radius_results(exp_dir)

xi_array = results["xi_list"]
min_dists = results["min_dist"]
closest_sols = results["closest_sol"]

# Print results
print("\nRobust Radius Results:")
print("-" * 60)
for i in range(len(xi_array)):
    print(f"Point {i+1}: {xi_array[i]}")
    print(f"  Robust radius: {min_dists[i]:.6f}")
    if not np.isnan(closest_sols[i, 0]):
        print(f"  Closest boundary: {closest_sols[i]}")
    else:
        print(f"  Closest boundary: Not found")
    print()

# Visualize for 2D case
if xi_array.shape[1] == 2:
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot input points
    ax.scatter(
        xi_array[:, 0],
        xi_array[:, 1],
        c="blue",
        s=150,
        label="Input points",
        zorder=5,
        marker="o",
    )

    # Plot robust radius circles
    for i in range(len(xi_array)):
        circle = plt.Circle(
            xi_array[i], min_dists[i], fill=False, color="blue", alpha=0.3, linewidth=2
        )
        ax.add_patch(circle)

        # Add radius value as text
        ax.text(
            xi_array[i, 0],
            xi_array[i, 1] - min_dists[i] - 0.05,
            f"r={min_dists[i]:.3f}",
            ha="center",
            va="top",
            fontsize=9,
        )

    # Plot closest boundary points
    mask = ~np.isnan(closest_sols[:, 0])
    if mask.any():
        ax.scatter(
            closest_sols[mask, 0],
            closest_sols[mask, 1],
            c="red",
            s=150,
            marker="x",
            linewidth=3,
            label="Decision boundary",
            zorder=5,
        )

        # Draw lines from input points to boundary
        for i in range(len(xi_array)):
            if mask[i]:
                ax.plot(
                    [xi_array[i, 0], closest_sols[i, 0]],
                    [xi_array[i, 1], closest_sols[i, 1]],
                    "r--",
                    alpha=0.3,
                    linewidth=1,
                )

    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(
        "Robust Radius Visualization\n(circles show certified regions)", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # Save plot
    plot_path = exp_dir / "analysis" / "robust_radius_visualization.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Visualization saved to: {plot_path}")

    plt.show()

print("\n" + "=" * 60)
print("Workflow complete!")
print("=" * 60)
print(f"\nAll results saved in: {exp_dir}")
print(f"  - Model: {exp_dir / 'model'}")
print(f"  - Analysis: {exp_dir / 'analysis'}")

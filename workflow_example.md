# Complete Python → Julia Workflow

This document shows the complete workflow from training a model in Python to analyzing it with Julia.

## Step 1: Train and Save Model (Python)

```python
# train_model.py
from src.pnn import PolynomialNeuralNetwork
from src.utils.utils import save_model
from src.utils.training import train_model
import torch

# Create model
model = PolynomialNeuralNetwork(
    input_dim=2,
    output_dim=2,
    hidden_dims=[3, 4],
    act_degree=2,
    homogeneous=False,
    bias=True,
)

# Train model (your training code here)
# train_model(model, train_loader, epochs=100)

# Save with metadata
exp_dir = save_model(
    model,
    metadata={
        "description": "Binary classifier",
        "dataset": "custom_2d",
        "epochs": 100,
        "learning_rate": 0.001,
        "accuracy": 0.95,
    }
)

print(f"Model saved to: {exp_dir}")
# Output: Model saved to: experiments/run_20241202_143022
```

**Result:**
```
experiments/
└── run_20241202_143022/
    ├── model/
    │   ├── model_config.json
    │   ├── model_weights.h5
    │   └── metadata.json
    └── analysis/
        (empty, ready for Julia results)
```

## Step 2: Run Julia Analysis

```julia
# analyze_model.jl
include("src/hc/EuclideanHC.jl")

# Load the latest model (symlink automatically resolved)
project_root = "experiments/latest"

# Define test points
xi_list = [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]]

# Run robust radius analysis
save_path = joinpath(project_root, "analysis", "robust_radius.npz")
results = robust_radius(project_root, xi_list, verbose=true, save_path=save_path)

println("\nAnalysis complete!")
println("Results saved to: $save_path")
```

**Output:**
```
============================================================
Loading model from: /path/to/experiments/run_20241202_143022
Model class: PolynomialNeuralNetwork
  Loading weight: activations.0.coeffs
  Loading weight: activations.1.coeffs
  Loading weight: layers.0.bias
  Loading weight: layers.0.weight
  ...
Model loaded successfully!
============================================================

============================================================
Computing robust radius for 3 points
[Progress bar...]
Results saved to: experiments/run_20241202_143022/analysis/robust_radius.npz
```

**Result:**
```
experiments/
└── run_20241202_143022/
    ├── model/
    │   ├── model_config.json
    │   ├── model_weights.h5
    │   └── metadata.json
    └── analysis/
        └── robust_radius.npz  ← Julia saved this!
```

## Step 3: Load and Visualize Results (Python)

```python
# visualize_results.py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.utils import get_experiment_path

# Get the latest experiment
exp_dir = get_experiment_path("latest")
print(f"Loading results from: {exp_dir}")

# Load analysis results
analysis_file = exp_dir / "analysis" / "robust_radius.npz"
data = np.load(analysis_file)

xi_list = data["xi_list"]          # Input points (n_points, dim)
min_dist = data["min_dist"]         # Robust radii (n_points,)
closest_sol = data["closest_sol"]   # Boundary points (n_points, dim)

# Print results
print("\nRobust Radius Results:")
print("=" * 60)
for i in range(len(xi_list)):
    print(f"Point {i+1}: {xi_list[i]}")
    print(f"  Robust radius: {min_dist[i]:.6f}")
    print(f"  Closest boundary: {closest_sol[i]}")
    print()

# Visualize (for 2D case)
if xi_list.shape[1] == 2:
    plt.figure(figsize=(8, 8))

    # Plot input points
    plt.scatter(xi_list[:, 0], xi_list[:, 1],
                c='blue', s=100, label='Input points', zorder=3)

    # Plot robust radius circles
    for i in range(len(xi_list)):
        circle = plt.Circle(xi_list[i], min_dist[i],
                           fill=False, color='blue', alpha=0.3)
        plt.gca().add_patch(circle)

    # Plot closest boundary points
    mask = ~np.isnan(closest_sol[:, 0])
    if mask.any():
        plt.scatter(closest_sol[mask, 0], closest_sol[mask, 1],
                   c='red', s=100, marker='x', label='Decision boundary', zorder=3)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Robust Radius Visualization')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(exp_dir / "analysis" / "robust_radius_plot.png", dpi=150)
    print(f"Visualization saved to: {exp_dir / 'analysis' / 'robust_radius_plot.png'}")
    plt.show()
```

## Step 4: Using JuliaCall (Python → Julia Integration)

```python
# run_julia_from_python.py
from juliacall import Main as jl
from src.utils.utils import save_model, get_experiment_path
from src.pnn import PolynomialNeuralNetwork

# 1. Train and save model in Python
model = PolynomialNeuralNetwork(2, 2, [3, 4], act_degree=2)
# ... train model ...
exp_dir = save_model(model, metadata={"method": "juliacall_example"})

# 2. Setup Julia environment
jl.seval('include("src/hc/EuclideanHC.jl")')

# 3. Run Julia analysis directly from Python
xi_list = [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]]
save_path = str(exp_dir / "analysis" / "robust_radius.npz")

# Call Julia function from Python
results = jl.robust_radius(
    str(exp_dir),
    xi_list,
    verbose=False,
    save_path=save_path
)

print(f"Analysis complete! Results in: {save_path}")

# 4. Load results back in Python
import numpy as np
data = np.load(save_path)
print(f"Robust radii: {data['min_dist']}")
```

## Directory Structure After Full Workflow

```
experiments/
├── run_20241202_143022/
│   ├── model/
│   │   ├── model_config.json       # Model architecture
│   │   ├── model_weights.h5        # Trained weights
│   │   └── metadata.json           # Training metadata
│   └── analysis/
│       ├── robust_radius.npz       # Julia analysis results
│       └── robust_radius_plot.png  # Python visualization
└── latest -> run_20241202_143022/  # Symlink for easy access
```

## Best Practices

1. **Always use the experiment structure** - Let `save_model()` create timestamped directories
2. **Use 'latest' for quick access** - Both Python and Julia support it
3. **Save metadata** - Include training info, hyperparameters, dataset info
4. **Organize analysis outputs** - Keep all analysis results in the `analysis/` folder
5. **Version control** - The `experiments/` folder is gitignored, but you can selectively commit important results

## Accessing Specific Experiments

```python
# Python
from src.utils.utils import list_experiments, get_experiment_path

# List all experiments
all_exp = list_experiments()
print(all_exp)
# ['run_20241202_143022', 'run_20241202_120000', 'run_20241201_180000']

# Get specific experiment
exp_dir = get_experiment_path("run_20241202_143022")
```

```julia
# Julia
project_root = "experiments/run_20241202_143022"
model_forward, model = Utils.load_model(project_root)
```
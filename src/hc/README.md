# Homotopy Continuation Analysis Module

Python interface to Julia-based homotopy continuation methods for neural network verification.

## Installation

### Requirements

1. **Python packages:**
   ```bash
   pip install juliacall numpy h5py matplotlib
   ```

2. **Julia environment:**

   The project includes a Julia environment defined in `Project.toml` at the project root. The Python wrapper will automatically activate this environment.

   To manually set up the Julia environment:
   ```julia
   # From the project root directory
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()  # Install all dependencies from Project.toml
   ```

   Or install packages individually:
   ```julia
   using Pkg
   Pkg.add("HomotopyContinuation")
   Pkg.add("DynamicPolynomials")
   Pkg.add("NPZ")
   Pkg.add("JSON3")
   Pkg.add("HDF5")
   Pkg.add("ProgressBars")
   ```

3. **Environment activation:**

   The Python wrapper (`src/hc/hc.py`) automatically:
   - Detects the `Project.toml` in the project root
   - Activates the Julia project environment
   - Loads all required Julia modules

   No manual Julia environment setup needed when using the Python interface!

## Quick Start

### Option 1: Using the Python Wrapper (Recommended)

```python
from src.pnn import PolynomialNeuralNetwork
from src.utils.utils import save_model
from src.hc import compute_robust_radius

# 1. Train and save model
model = PolynomialNeuralNetwork(2, 2, [3, 4], act_degree=2)
exp_dir = save_model(model, metadata={"description": "my model"})

# 2. Define test points
xi_list = [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]]

# 3. Compute robust radius (calls Julia automatically)
results = compute_robust_radius(exp_dir, xi_list, verbose=True)

# 4. Access results
print(f"Robust radii: {results['min_dist']}")
print(f"Boundary points: {results['closest_sol']}")
```

### Option 2: Direct Julia Call

```julia
include("src/hc/EuclideanHC.jl")

project_root = "experiments/latest"
xi_list = [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]]
save_path = joinpath(project_root, "analysis", "robust_radius.npz")

results = robust_radius(project_root, xi_list, verbose=true, save_path=save_path)
```

## API Reference

### Python Functions

#### `compute_robust_radius()`

Compute robust radius for input points using Julia.

```python
from src.hc import compute_robust_radius

results = compute_robust_radius(
    experiment_path="experiments/latest",  # Path to experiment
    xi_list=[[0.5, 0.5], [1.0, 0.0]],     # Test points
    verbose=True,                          # Show progress
    save_results=True,                     # Save to NPZ file
    output_filename="robust_radius.npz"    # Output filename
)
```

**Returns:** Dictionary with keys:
- `'xi_list'`: Input points (numpy array, shape: n_points × dim)
- `'min_dist'`: Robust radii (numpy array, shape: n_points)
- `'closest_sol'`: Closest boundary points (numpy array, shape: n_points × dim)
- `'save_path'`: Path where results were saved

#### `load_robust_radius_results()`

Load previously computed results.

```python
from src.hc import load_robust_radius_results

results = load_robust_radius_results(
    experiment_path="experiments/run_20241202_143022",
    filename="robust_radius.npz"
)
```

#### `verify_experiment()`

Check if an experiment has all required files.

```python
from src.hc import verify_experiment

status = verify_experiment("experiments/latest")
print(f"Valid: {status['valid']}")
if not status['valid']:
    print(f"Errors: {status['errors']}")
```

#### `analyze_latest_model()`

Quick analysis of the most recent model.

```python
from src.hc import analyze_latest_model

xi_list = [[0.5, 0.5], [1.0, 0.0]]
results = analyze_latest_model(xi_list, verbose=True)
```

### Julia Functions

#### `euclidean_hc(f, x, xi; verbose=false)`

Compute closest point on decision boundary using Euclidean distance.

**Args:**
- `f`: Polynomial expression (decision boundary: f=0)
- `x`: Symbolic variables
- `xi`: Query point
- `verbose`: Print detailed information

**Returns:** `(closest_sol, min_distance, n_real_sols, n_total_sols, real_sols, all_sols)`

#### `robust_radius(project_root, xi_list; verbose=false, save_path=nothing)`

Compute robust radius for multiple points.

**Args:**
- `project_root`: Path to experiment directory
- `xi_list`: Vector of query points
- `verbose`: Print progress information
- `save_path`: Path to save NPZ results (optional)

**Returns:** Vector of `(min_dist, closest_sol)` tuples

## Complete Workflow Example

```python
# 1. Train model
from src.pnn import PolynomialNeuralNetwork
from src.utils.utils import save_model

model = PolynomialNeuralNetwork(2, 2, [3, 4], act_degree=2)
# ... train model ...

exp_dir = save_model(model, metadata={
    "dataset": "my_data",
    "epochs": 100,
    "accuracy": 0.95
})

# 2. Run Julia analysis
from src.hc import compute_robust_radius

xi_list = [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]]
results = compute_robust_radius(exp_dir, xi_list, verbose=True)

# 3. Visualize results
import matplotlib.pyplot as plt
import numpy as np

xi = results['xi_list']
radii = results['min_dist']
boundary = results['closest_sol']

plt.figure(figsize=(8, 8))
plt.scatter(xi[:, 0], xi[:, 1], c='blue', s=100, label='Input points')

for i in range(len(xi)):
    circle = plt.Circle(xi[i], radii[i], fill=False, color='blue', alpha=0.3)
    plt.gca().add_patch(circle)

mask = ~np.isnan(boundary[:, 0])
plt.scatter(boundary[mask, 0], boundary[mask, 1],
            c='red', marker='x', s=100, label='Decision boundary')

plt.legend()
plt.axis('equal')
plt.savefig(exp_dir / "analysis" / "visualization.png")
plt.show()
```

## Troubleshooting

### Julia not found

If you get "Julia not found" error:

```bash
# Make sure Julia is in PATH
which julia

# Or install juliacall with specific Julia path
JULIA_PATH=/path/to/julia pip install juliacall
```

### Import errors

Make sure all Julia packages are installed:

```julia
using Pkg
Pkg.status()  # Check installed packages
```

### Symlink issues on Windows

Windows may not support symlinks. Use specific run names instead:

```python
# Instead of "experiments/latest"
results = compute_robust_radius("experiments/run_20241202_143022", xi_list)
```

## File Structure

```
src/hc/
├── __init__.py              # Python package init
├── julia_interface.py       # Python wrapper for Julia functions
├── EuclideanHC.jl          # Julia: Robust radius computation
├── Utils.jl                 # Julia: Model loading utilities
└── README.md               # This file
```

## Performance Tips

1. **Batch analysis**: Analyze multiple points in one call rather than calling the function repeatedly
2. **Verbose mode**: Use `verbose=True` during development, `verbose=False` in production
3. **Save results**: Always set `save_results=True` to avoid recomputation
4. **Julia warmup**: First call is slower due to Julia JIT compilation

## References

- **HomotopyContinuation.jl**: https://www.juliahomotopycontinuation.org/
- **Julia-Python Integration**: https://github.com/cjdoris/PythonCall.jl
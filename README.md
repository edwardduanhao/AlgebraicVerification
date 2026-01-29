# Algebraic Robustness Verification of Neural Networks

This repository contains the code for the paper *Algebraic Robustness Verification of Neural Networks*.

## Repository Structure

```
.
├── src/
│   ├── pnn/                 # Polynomial Neural Network (PyTorch)
│   │   └── pnn.py           # PolynomialNeuralNetwork, PolynomialActivation
│   ├── hc/                  # Homotopy continuation verifier (Julia + Python)
│   │   ├── hc.py            # Python interface (compute_robust_radius)
│   │   ├── EuclideanHC.jl   # Julia HC solver (euclidean_hc, robust_radius)
│   │   └── Utils.jl         # Model loading and symbolic forward pass
│   ├── ed/                  # ED degree and discriminant computations
│   │   ├── ed-degree.jl     # ED degree via homotopy continuation (Julia)
│   │   ├── ed-degree.m2     # ED degree (Macaulay2)
│   │   ├── ed-disc.jl       # ED discriminant (Julia)
│   │   └── ed-disc.m2       # ED discriminant (Macaulay2)
│   ├── data/                # Datasets (Steiner Roman, Yin-Yang, Fan, MNIST, ...)
│   └── utils/               # Training and experiment utilities
├── examples/
│   ├── verification.py      # End-to-end verification with 3D visualization (Figure 1)
│   ├── homotopy_continuation.py  # Homotopy path tracking for cubics (Figure 2)
│   └── kac_rice.py          # Kac-Rice expected real zeros simulation (Figure 5)
├── benchmark/               # Systematic benchmark across model configurations
│   ├── config.py            # Benchmark configurations (architecture, degree, epsilon)
│   ├── data.py              # Instance generation (unverifiable + clean)
│   ├── train.py             # Model training with planted counterexamples
│   └── verify.py            # Verification and timing instrumentation
├── Project.toml             # Julia dependencies
└── Manifest.toml
```

## Setup

### Requirements

- **Python**: PyTorch, NumPy, juliacall, h5py
- **Julia** (>= 1.8): HomotopyContinuation.jl, DynamicPolynomials.jl, HDF5.jl, JSON3.jl, NPZ.jl
- **Macaulay2** (optional, for ED degree/discriminant scripts in `src/ed/`)

### Installation

```bash
pip install torch numpy juliacall h5py

# Julia dependencies are managed via Project.toml.
# They are automatically resolved when the Julia environment is activated.
```

## Core API

### Polynomial Neural Network

```python
from src.pnn import PolynomialNeuralNetwork

model = PolynomialNeuralNetwork(
    input_dim=3,
    output_dim=2,
    hidden_dims=[20, 20],
    act_degree=2,       # quadratic activations
    homogeneous=False,   # full polynomial (not just x^d)
    bias=True,
)

# Standard PyTorch forward pass
logits = model(x)  # x: (batch_size, 3) -> logits: (batch_size, 2)
```

### Robust Radius Computation

The core verification function computes the **robust radius** -- the minimum Euclidean distance from a query point to the model's decision boundary -- using homotopy continuation in Julia.

```python
from src.hc import compute_robust_radius

results = compute_robust_radius(
    experiment_path="experiments/run_20241202_143022",  # directory with model/ subfolder
    xi_list=[[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]],       # query points to verify
    verbose=True,
    save_results=True,
)

print(f"Robust radii: {results['min_dist']}")
# e.g., [0.2341, 0.4102] -- certified minimum distance to decision boundary

print(f"Closest boundary points: {results['closest_sol']}")
```

If `robust_radius > epsilon` for a given point, the model is **certified robust** to any perturbation within the L2 epsilon-ball around that point.

### Loading Previous Results

```python
from src.hc import load_robust_radius_results

results = load_robust_radius_results("experiments/run_20241202_143022")
print(results["min_dist"])
```

## Examples

The `examples/` folder contains scripts that reproduce key figures from the paper.

### Verification with 3D Visualization (`examples/verification.py`)

Reproduces **Figure 1**. Trains a PNN on the Steiner Roman surface dataset and visualizes the certified robust radius as a sphere around the query point in 3D.

```python
# Train a PNN on the Steiner Roman surface
from src.data import SteinerRomanDataset
from src.pnn import PolynomialNeuralNetwork
from src.utils import train_epochs, save_model
from src.hc import compute_robust_radius

dataset = SteinerRomanDataset(size=1000)
model = PolynomialNeuralNetwork(input_dim=3, output_dim=2, hidden_dims=[20, 20], act_degree=2)

history = train_epochs(model=model, train_loader=loader, num_epochs=10000,
                       optimizer_type="adam", learning_rate=1e-3)

path = save_model(model, metadata={"description": "steiner roman 3d", ...})

# Compute certified robust radius
results = compute_robust_radius(experiment_path=path, xi_list=[[0.45, 0.45, 0.45]],
                                 verbose=True, save_detailed=True)
print(f"Robust radius: {results['min_dist']}")
```

Run: `cd examples && python verification.py`

### Homotopy Continuation for Cubics (`examples/homotopy_continuation.py`)

Reproduces **Figure 2**. Solves `x^3 - 4x^2 + 8x - 8 = 0` by tracking three homotopy paths from the start system `x^3 - 1 = 0` using a predictor-corrector method. Visualizes the paths in the complex plane.

```python
from examples.homotopy_continuation import solve, plot_paths

sols, paths, gamma = solve(a=-4, b=8, c=-8, dt=1e-3)
# sols contains the 3 roots of x^3 - 4x^2 + 8x - 8 = 0

plot_paths(paths)
```

Run: `cd examples && python homotopy_continuation.py`

### Kac-Rice Simulation (`examples/kac_rice.py`)

Reproduces **Figure 5**. Computes the expected number of real zeros for random polynomial activations of degree 2 through 10, comparing empirical counts against the theoretical formula `d / sqrt(2d - 1)`.

```python
from examples.kac_rice import sample_coeff, count_real_roots

# Sample random polynomial coefficients (1000 samples, degree 5, 20000 hidden units)
coeffs = sample_coeff(n=1000, d=5, m=20000, seed=2026)
counts, roots = count_real_roots(coeffs, eps=1e-10, normalize=True)
print(f"Average real roots: {counts.mean():.4f}")  # compare to 5/sqrt(9) = 1.667
```

Run: `cd examples && python kac_rice.py`

## Benchmark

The `benchmark/` module provides a systematic evaluation framework across multiple model configurations (varying hidden dimension, activation degree, and perturbation budget).

### Training

```bash
python -m benchmark.train                    # train all 8 configs (default: 5000 epochs)
python -m benchmark.train --epochs 10000     # custom epoch count
```

This generates instances with planted counterexamples, trains a PNN for each configuration, and saves models to `benchmark/results/`.

### Verification

```bash
python -m benchmark.verify                   # verify all trained models
python -m benchmark.verify --results-dir path/to/results
```

Computes robust radii for all instances, reports verification/falsification rates, and saves per-instance timing data (compile time vs. runtime, separated via Julia's `@timed`).

Results are saved to:
- `verification_summary.csv` -- per-config summary with timing columns
- `verification_results.json` -- per-config detailed results with per-instance timing arrays
- `analysis/timing.npz` -- raw timing data (model load time, per-instance wall/compile time, thread count)

## ED Degree and Discriminant (`src/ed/`)

Julia and Macaulay2 implementations for computing the Euclidean Distance (ED) degree and ED discriminant of polynomial neural network decision boundaries.

- **ED degree**: The number of critical points of the squared distance function to the decision boundary, for a generic query point.
- **ED discriminant**: The locus where critical points of the distance function coincide (the evolute).

```
src/ed/ed-degree.jl   # Julia: ED degree via homotopy continuation
src/ed/ed-degree.m2   # Macaulay2: ED degree via symbolic computation
src/ed/ed-disc.jl     # Julia: ED discriminant
src/ed/ed-disc.m2     # Macaulay2: ED discriminant
```

# Algebraic Robustness Verification of Neural Networks

This repository contains the code for the paper *Algebraic Robustness Verification of Neural Networks*.

## Repository Structure

```
.
├── ed-degree.jl            # Algorithm 1: ED degree computation (Julia)
├── ED-deg-and-ED-disc.m2   # Algorithm 2: ED degree and discriminant (Macaulay2)
├── src/
│   ├── pnn/                # Polynomial Neural Network (PyTorch)
│   ├── hc/                 # Homotopy Continuation verifier (Julia + Python)
│   ├── data/               # Data utilities
│   └── utils/              # Helper functions
├── benchmark/              # Benchmark experiments
├── cubic/                  # Cubic polynomial experiments (figure generation)
└── kacrice/                # Kac-Rice experiments (figure generation)
```

## Algorithms

### Algorithm 1: ED Degree Computation (`ed-degree.jl`)

Computes the Euclidean Distance (ED) degree of a polynomial neural network's decision boundary using homotopy continuation. The ED degree represents the algebraic complexity of finding the closest point on the decision boundary to a generic data point.

**Key functions:**
- `nn_poly_complex_scale`: Generates the decision boundary polynomial with numerical stability
- Uses Lagrange multiplier formulation to find critical points of the distance function

### Algorithm 2: ED Degree and Discriminant (`ED-deg-and-ED-disc.m2`)

Macaulay2 implementation for computing both the ED degree and the ED discriminant (evolute) for shallow quadratic neural networks.

**Key functions:**
- `EDdegree`: Computes the number of critical points for a generic data point
- `EDdisc`: Computes the discriminant locus where critical points coincide

## Verifier

The main verification tool is in `src/hc/`, which computes the **robust radius** (minimum distance to decision boundary) for polynomial neural networks.

### Components

- **`src/pnn/pnn.py`**: Polynomial Neural Network with learnable polynomial activations
- **`src/hc/EuclideanHC.jl`**: Julia module for homotopy continuation
- **`src/hc/hc.py`**: Python interface to Julia functions

### Core Function

```python
from src.hc.hc import compute_robust_radius

results = compute_robust_radius(
    experiment_path="path/to/experiment",
    xi_list=[[0.5, 0.5], [1.0, 0.0]],  # Points to verify
    verbose=True
)

print(f"Robust radii: {results['min_dist']}")
```

## Usage Examples

*Coming soon: Examples using the Yinyang dataset.*

## Requirements

- **Python**: PyTorch, NumPy, juliacall, h5py
- **Julia**: HomotopyContinuation.jl, DynamicPolynomials.jl, HDF5.jl
- **Macaulay2** (for Algorithm 2)

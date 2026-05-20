# Algebraic Robustness Verification of Neural Networks

Code accompanying the paper *Algebraic Robustness Verification of Neural Networks*.

The pipeline trains Polynomial Neural Networks (PNNs) in PyTorch and certifies
robustness by computing the minimum Euclidean distance from a query point to
the decision boundary via homotopy continuation in Julia
([HomotopyContinuation.jl](https://www.juliahomotopycontinuation.org/)). The
Julia stage runs in an isolated worker subprocess, so it is safe to call from
any Python process that has already imported PyTorch.

## Repository Structure

```
.
├── src/
│   ├── pnn/                  # PolynomialNeuralNetwork + det(A)/det(M) constraints
│   ├── hc/                   # Homotopy continuation verifier (Python + Julia)
│   │   ├── hc.py             #   Python interface (compute_robust_radius)
│   │   ├── EuclideanHC.jl    #   Julia HC solver
│   │   └── Utils.jl          #   Model loading + symbolic forward pass
│   ├── ed/                   # ED degree and discriminant (Julia + Macaulay2)
│   ├── data/                 # Synthetic + real datasets
│   └── utils/                # Training (incl. projected SGD), saving, plotting
├── examples/                 # Paper figure reproductions
│   ├── verification.py             # Figure 1 — Steiner Roman + 3D robust sphere
│   ├── homotopy_continuation.py    # Figure 2 — cubic path tracking
│   ├── kac_rice.py                 # Figure 5 — expected real zeros
│   └── constraints/                # Boundary-degeneracy regularization experiments
│       ├── xor_quadratic.py        #   projected SGD with det(M) = 0
│       ├── xor_det_regularizer.py  #   det(M)² as a soft regularizer (λ sweep)
│       └── sinusoid_det_constraint.py
├── benchmark/                # Systematic benchmark across architectures
├── tests/                    # Regression tests for the verifier
├── Project.toml / Manifest.toml
└── requirements.txt
```

## Setup

### Requirements

- Python ≥ 3.10
- Julia ≥ 1.10 (juliacall ships a bundled Julia if none is installed)
- Macaulay2 (optional, only for the symbolic ED scripts in `src/ed/`)

### Installation

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Julia deps (first run only — resolves Project.toml)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The Python wrapper activates the local Julia environment automatically on
every call, so no further setup is needed.

## Quick Start

Train a PNN, save it, and compute the certified robust radius around a query
point:

```python
from torch.utils.data import DataLoader

from src.data import SteinerRomanDataset
from src.pnn import PolynomialNeuralNetwork
from src.utils import train_epochs, save_model
from src.hc import compute_robust_radius

loader = DataLoader(SteinerRomanDataset(size=1000), batch_size=1000)
model = PolynomialNeuralNetwork(
    input_dim=3, output_dim=2, hidden_dims=[20, 20], act_degree=2,
)
train_epochs(model=model, train_loader=loader, num_epochs=10000,
             optimizer_type="adam", learning_rate=1e-3)

exp_path = save_model(model, metadata={"description": "Steiner Roman demo"})

results = compute_robust_radius(
    experiment_path=exp_path,
    xi_list=[[0.45, 0.45, 0.45]],
    verbose=True,
)
print("robust radius:", results["min_dist"])     # e.g. [0.2284]
print("closest boundary point:", results["closest_sol"])
```

If `min_dist > epsilon` for a query point, the model is **certified robust**
to any L2 perturbation of size ≤ `epsilon` around that point.

### Subprocess isolation

`compute_robust_radius` always runs Julia in a fresh worker process. You do
*not* need to import `juliacall` before PyTorch — the old segfault scenario is
neutralized. Worker stdout/stderr are streamed live to the parent and the
last 80 lines of stderr are surfaced in the exception if the worker exits
non-zero.

Optional kwargs: `num_threads` (`"auto"` by default — uses all cores for
parallel path tracking) and `timeout` (wall-clock seconds before the worker
is killed).

## Reproducing the Paper

| Figure | Command | Notes |
|---|---|---|
| Figure 1 — robust radius on Steiner Roman | `python examples/verification.py` | Trains a degree-2 PNN, certifies one query point, renders the 3D decision boundary + sphere |
| Figure 2 — homotopy continuation paths | `python examples/homotopy_continuation.py` | Pure-Python predictor–corrector demo, no Julia required |
| Figure 5 — Kac–Rice expected real zeros | `python examples/kac_rice.py` | Empirical vs. theoretical `d/√(2d−1)` |
| Constraint section — XOR + det(M) = 0 | `python examples/constraints/xor_quadratic.py` | Projected SGD vs. unconstrained Adam |
| Constraint section — XOR + λ·det(M)² | `python examples/constraints/xor_det_regularizer.py` | λ ∈ {0, 1, 10, 100} sweep |
| Constraint section — Sinusoid + det(A) = 0 | `python examples/constraints/sinusoid_det_constraint.py` | Projected SGD on a sinusoidal boundary |

All example scripts use `Path(__file__)`-based resolution, so they run from
any cwd.

### A note on reliability

Polynomial activations of degree `d` over an `L`-layer network produce a
decision boundary of degree `d^L`. HomotopyContinuation tracks ≈
`(d^L)^(n+1)` paths for an `n`-input network, and the numerical conditioning
degrades as that count grows. The paper figures use `act_degree = 2`, where
HC is fully reliable. With higher degrees the reported `min_dist` can become
optimistic (the worker may miss real critical points). See the comment at
the top of `examples/verification.py` for the tradeoff.

## Benchmark

A systematic sweep over `(hidden_dim, act_degree, epsilon)` with planted
counterexamples.

```bash
python -m benchmark.train                 # train all configs (default 5000 epochs)
python -m benchmark.train --epochs 10000

python -m benchmark.verify                # verify all trained models
```

Outputs land under `benchmark/results/<config>/`:

- `verification_results.json` per-config detailed results with per-instance timing
- `verification_summary.csv` aggregate table across configs
- `analysis/timing.npz` raw timing (model load, per-instance wall + compile, thread count)

## Running tests

```bash
pytest tests/ -v
```

The suite imports PyTorch in the test process *before* calling
`compute_robust_radius`, then asserts finite robust radii — the regression
guard that pins the subprocess-isolation contract. Each test costs ≈ 20 s of
Julia + HomotopyContinuation precompile in the worker; the full suite runs
in about 90 seconds.

## VNN-COMP / completenessbench integration

`verify_vnnlib.py` at the repo root is a thin VNN-COMP-shaped CLI around
`compute_robust_radius`. It is the entry point the
[completenessbench](https://github.com/dtroxell19/completenessbench) harness
invokes when running this verifier on a benchmark.

```bash
python verify_vnnlib.py <model.onnx> <property.vnnlib> \
    [--timeout S] [--device cpu|cuda] [--result-file PATH]
```

Prints one `Result: unsat|sat|unknown|timeout` line to stdout and exits
`0`/`1`/`2` respectively, matching the conventions of the other adapters in
`completenessbench/verifier_adapters/`.

**Sidecar contract.** The wrapper does not parse the ONNX graph. Instead it
reads a `<model.onnx>.pnn.json` sidecar that the matching constructor
(`pnn_polynomial.algebraic` in completenessbench) emits alongside each
instance — it contains the architecture and state-dict needed to rebuild the
`PolynomialNeuralNetwork`. The ONNX is kept for the harness's own
accounting and for any non-HC verifier that consumes the same benchmark.

**L∞ ↔ L2 translation.** completenessbench properties are standard L∞ boxes;
`compute_robust_radius` returns an L2 radius `r` from the box center. The
wrapper translates with the strict geometry rule:

| condition | verdict |
|---|---|
| `r ≥ ε · √n` | `unsat` (the L∞ box of half-width ε fits inside the certified L2 ball) |
| `r < ε` and the closest boundary point can be nudged into a class-flipping witness inside the box | `sat` (with a witness past the boundary) |
| otherwise | `unknown` (the L2 certificate cannot decide the L∞ box) |

The `unknown` band `ε ≤ r < ε·√n` widens with input dimension — by design,
this verifier reports honestly rather than claiming spurious robustness.

**Setup from the completenessbench side.**

```bash
# in completenessbench's conda env
pip install -e /path/to/AlgebraicVerification
export ALGEBRAIC_VERIFIER_DIR=/path/to/AlgebraicVerification

# build instances and verify
python -m completenessbench.cli.create_benchmark \
    --spec configs/my_pnn_sweep.yaml --out_dir ./benchmarks/pnn
python -m completenessbench.cli.verify_benchmark \
    --benchmark ./benchmarks/pnn --verifier algebraic_pnn \
    --out_dir ./runs/pnn
```

**Reliability note.** The same activation-degree caveat as in the
HomotopyContinuation core applies: degree-2 instances are reliable; higher
degrees push HC outside its reliable regime and the wrapper will
increasingly return `unknown`.

## ED Degree and ED Discriminant

Julia and Macaulay2 implementations for the algebraic invariants of the
decision boundary:

- `src/ed/ed-degree.{jl,m2}` — number of critical points of `‖x − ξ‖²`
  restricted to the boundary, for a generic `ξ`.
- `src/ed/ed-disc.{jl,m2}` — locus of `ξ` where two critical points coincide
  (the evolute).

These are reference implementations used to validate the verifier; they are
not exercised by `examples/`.

## Citation

```bibtex
@misc{alexandr2026robustnessverificationpolynomialneural,
      title={Robustness Verification of Polynomial Neural Networks}, 
      author={Yulia Alexandr and Hao Duan and Guido Montúfar},
      year={2026},
      eprint={2602.06105},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2602.06105}, 
}
```

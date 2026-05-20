"""Smoke tests for the Julia HC worker subprocess.

These tests cost ~30s each (Julia + HC.jl precompile in the worker), but they
guard the core stability contract: the public ``compute_robust_radius`` API
must remain safe to call from a Python process that has already imported
PyTorch, and worker failures must surface the Julia traceback in the exception.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def trained_pnn_dir(tmp_path_factory):
    """Train a tiny 2D PNN once per test session and return its experiment dir."""
    # Importing torch in the *parent* test process is the whole point: this
    # is the scenario that historically segfaulted juliacall, and the
    # subprocess pattern is what makes it safe again.
    import torch
    from torch.utils.data import DataLoader

    from src.data import SinusoidDataset
    from src.pnn import PolynomialNeuralNetwork
    from src.utils import save_model, train_epochs

    torch.manual_seed(0)
    dataset = SinusoidDataset(size=200)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = PolynomialNeuralNetwork(
        input_dim=2, output_dim=2, hidden_dims=[8], act_degree=2,
        homogeneous=False, bias=True,
    )
    train_epochs(
        model=model, train_loader=loader, num_epochs=30,
        optimizer_type="adam", learning_rate=1e-2, verbose=False,
    )
    base_dir = tmp_path_factory.mktemp("experiments")
    return save_model(
        model,
        metadata={"description": "smoke test PNN"},
        base_dir=base_dir,
    )


def test_worker_survives_torch_first_import(trained_pnn_dir):
    """End-to-end: torch loaded in parent, worker still returns finite radii."""
    from src.hc import compute_robust_radius

    res = compute_robust_radius(
        experiment_path=trained_pnn_dir,
        xi_list=[[0.0, 0.0], [0.3, 0.3]],
        verbose=False,
        save_results=False,
    )

    assert set(res.keys()) >= {"xi_list", "min_dist", "closest_sol"}
    radii = res["min_dist"]
    assert len(radii) == 2
    for r in radii:
        assert math.isfinite(r), f"expected finite robust radius, got {r}"
        assert r >= 0, f"expected non-negative radius, got {r}"

    if "timing" in res:
        assert res["timing"]["n_threads"] >= 1


def test_worker_failure_surfaces_julia_stderr(trained_pnn_dir):
    """A bad-dimension xi must raise RuntimeError with Julia traceback inline."""
    from src.hc import compute_robust_radius

    with pytest.raises(RuntimeError) as exc_info:
        compute_robust_radius(
            experiment_path=trained_pnn_dir,
            xi_list=[[0.0, 0.0, 0.0]],  # 3D xi against a 2D model
            verbose=False,
            save_results=False,
        )
    msg = str(exc_info.value)
    assert "exit code" in msg
    # The captured stderr tail should reach the Python user.
    assert "Stacktrace" in msg or "DimensionMismatch" in msg or "Traceback" in msg


def test_save_detailed_requires_save_results(trained_pnn_dir):
    """save_detailed=True without save_results=True is a usage error."""
    from src.hc import compute_robust_radius

    with pytest.raises(ValueError, match="save_detailed"):
        compute_robust_radius(
            experiment_path=trained_pnn_dir,
            xi_list=[[0.0, 0.0]],
            save_results=False,
            save_detailed=True,
        )


def test_transient_call_does_not_pollute_analysis_dir(trained_pnn_dir, tmp_path):
    """save_results=False must not leave artifacts in <experiment>/analysis/."""
    from src.hc import compute_robust_radius

    analysis_dir = Path(trained_pnn_dir) / "analysis"
    before = set(p.name for p in analysis_dir.glob("*")) if analysis_dir.exists() else set()

    compute_robust_radius(
        experiment_path=trained_pnn_dir,
        xi_list=[[0.1, 0.1]],
        verbose=False,
        save_results=False,
    )

    after = set(p.name for p in analysis_dir.glob("*")) if analysis_dir.exists() else set()
    # Allow the dir itself to exist but no new files should appear.
    new_files = after - before
    assert not new_files, f"transient call left artifacts behind: {new_files}"

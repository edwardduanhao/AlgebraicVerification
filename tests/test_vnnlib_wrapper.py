"""End-to-end test for verify_vnnlib.py.

Builds a planted PNN whose decision boundary is `sum_i x_i^2 = 1` (the unit
hypersphere over the input coords), writes a fresh ONNX + VNNLIB + sidecar,
then shells out to verify_vnnlib.py and asserts the exit code matches the
analytic verdict.

Two cases:
    - eps small enough that eps*sqrt(n) < 1   -> exit 0 (unsat / robust)
    - eps > 1                                  -> exit 1 (sat / violated)

Costs ~30-60s total (one HC precompile per subprocess invocation).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _plant_unit_sphere_pnn(input_dim: int, hidden_dim: int = 4):
    """PNN whose decision boundary is sum_i x_i^2 = 1, class 0 inside."""
    from src.pnn import PolynomialNeuralNetwork

    model = PolynomialNeuralNetwork(
        input_dim=input_dim, output_dim=2, hidden_dims=[hidden_dim],
        act_degree=2, homogeneous=True, bias=True, s=1.0, trainable=True,
    ).eval()

    with torch.no_grad():
        W1 = torch.zeros(hidden_dim, input_dim)
        for i in range(input_dim):
            W1[i, i] = 1.0
        model.layers[0].weight.copy_(W1)
        model.layers[0].bias.zero_()
        model.activations[0].coeffs.copy_(torch.ones_like(model.activations[0].coeffs))
        W2 = torch.zeros(2, hidden_dim)
        W2[0, :input_dim] = -0.5
        W2[1, :input_dim] = +0.5
        model.layers[1].weight.copy_(W2)
        # Y_0 - Y_1 = 1 - sum_i x_i^2  => class 0 wins inside unit sphere
        model.layers[1].bias.copy_(torch.tensor([0.5, -0.5]))

    return model


def _state_dict_to_jsonable(sd):
    return {
        k: {"shape": list(v.shape), "data": v.detach().cpu().numpy().astype(float).flatten().tolist()}
        for k, v in sd.items()
    }


def _write_vnnlib(path: Path, center: np.ndarray, eps: float, num_outputs: int, label: int) -> None:
    n = center.size
    lines = []
    for i in range(n):
        lines.append(f"(declare-const X_{i} Real)")
    for j in range(num_outputs):
        lines.append(f"(declare-const Y_{j} Real)")
    for i in range(n):
        lines.append(f"(assert (>= X_{i} {center[i] - eps}))")
        lines.append(f"(assert (<= X_{i} {center[i] + eps}))")
    disj = [f"(and (>= Y_{j} Y_{label}))" for j in range(num_outputs) if j != label]
    if len(disj) == 1:
        lines.append(f"(assert {disj[0]})")
    else:
        lines.append("(assert (or " + " ".join(disj) + "))")
    path.write_text("\n".join(lines) + "\n")


def _make_instance(
    tmp_path: Path,
    input_dim: int,
    eps: float,
    center_shift: float = 0.3,
) -> tuple[Path, Path]:
    """Build an instance whose true L2 robust radius from `xi` is `1 - center_shift`.

    The boundary is the unit sphere `sum_i x_i^2 = 1`; xi sits at
    (center_shift, 0, ..., 0), inside the sphere. The closest boundary point is
    (1, 0, ..., 0) at distance `1 - center_shift`. Off-axis xi avoids the
    rotational degeneracy that HC hits at the sphere center.
    """
    onnx_path = tmp_path / "model.onnx"
    vnnlib_path = tmp_path / "spec.vnnlib"
    sidecar_path = onnx_path.with_suffix(onnx_path.suffix + ".pnn.json")

    model = _plant_unit_sphere_pnn(input_dim=input_dim)
    # Wrapper reads the sidecar, not the ONNX — touch a placeholder so the
    # path exists without depending on the `onnx` python package in this env.
    onnx_path.write_bytes(b"")

    sidecar = {
        "arch": {
            "input_dim": input_dim, "output_dim": 2, "hidden_dims": [4],
            "act_degree": 2, "homogeneous": True, "bias": True, "trainable": True, "s": 1.0,
        },
        "state_dict": _state_dict_to_jsonable(model.state_dict()),
        "ground_truth": {"true_radius": 1.0 - center_shift, "epsilon": eps, "label": 0},
    }
    sidecar_path.write_text(json.dumps(sidecar))

    xi = np.zeros(input_dim)
    xi[0] = center_shift
    _write_vnnlib(vnnlib_path, xi, eps, num_outputs=2, label=0)
    return onnx_path, vnnlib_path


def _run_wrapper(onnx_path: Path, vnnlib_path: Path, timeout: float = 180.0):
    result_file = onnx_path.parent / "result.txt"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "verify_vnnlib.py"),
        str(onnx_path), str(vnnlib_path),
        "--timeout", str(timeout),
        "--result-file", str(result_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    return proc, result_file


def test_unsat_when_box_fits_inside_certified_ball(tmp_path):
    # xi=(0.3, 0), true_r=0.7; eps=0.4 => eps*sqrt(2)~0.566 < 0.7 => unsat
    onnx_path, vnnlib_path = _make_instance(tmp_path, input_dim=2, eps=0.4)
    proc, result_file = _run_wrapper(onnx_path, vnnlib_path)
    assert proc.returncode == 0, (
        f"expected unsat (rc 0), got rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "Result: unsat" in proc.stdout
    assert result_file.exists() and result_file.read_text().startswith("unsat")


def test_sat_when_box_exceeds_true_radius(tmp_path):
    # xi=(0.3, 0), true_r=0.7; eps=0.8 => eps>0.7 => sat (a box corner past boundary)
    onnx_path, vnnlib_path = _make_instance(tmp_path, input_dim=2, eps=0.8)
    proc, result_file = _run_wrapper(onnx_path, vnnlib_path)
    assert proc.returncode == 1, (
        f"expected sat (rc 1), got rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "Result: sat" in proc.stdout
    assert result_file.exists() and result_file.read_text().startswith("sat")


def test_vnnlib_parser_recovers_center_and_eps(tmp_path):
    """Unit-level sanity check on the VNNLIB parser (no HC)."""
    from verify_vnnlib import parse_vnnlib_box

    p = tmp_path / "spec.vnnlib"
    _write_vnnlib(p, np.array([0.1, -0.2, 0.3]), 0.25, num_outputs=2, label=1)
    center, eps, label, num_outputs = parse_vnnlib_box(p)
    np.testing.assert_allclose(center, [0.1, -0.2, 0.3], atol=1e-9)
    assert abs(eps - 0.25) < 1e-9
    assert label == 1
    assert num_outputs == 2

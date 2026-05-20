"""VNN-COMP-shaped CLI for the algebraic HC verifier.

Usage:
    python verify_vnnlib.py <model.onnx> <property.vnnlib> \
        [--timeout S] [--device cpu|cuda] [--result-file PATH]

Matches the contributor's spec for the completenessbench harness. Prints a
single `Result: unsat|sat|unknown|timeout` line to stdout (recognized by
completenessbench/verifier_adapters/common.normalize_status_from_text) and
exits 0/1/2 respectively. With --result-file, also writes a VNN-COMP-style
line to that file.

Expects a sidecar JSON next to the ONNX at `<model.onnx>.pnn.json` that was
emitted by the algebraic constructor. The ONNX itself is informational here;
we reconstruct the PolynomialNeuralNetwork from the sidecar.

Decision rule, given the certified L2 radius r at the box center:
    r >= eps * sqrt(input_dim)  -> unsat (the L_inf box fits inside the L2 ball)
    r <  eps and a witness in the box is verifiable in PyTorch -> sat
    otherwise -> unknown
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make `from src...` importable when run as `python verify_vnnlib.py` from any cwd
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _print_result(status: str, witness: Optional[List[float]] = None,
                  result_file: Optional[Path] = None) -> None:
    """Emit the result both to stdout (for the adapter parser) and to
    `--result-file` if provided. Status in {unsat, sat, unknown, timeout}."""
    print(f"Result: {status}")
    if status == "sat" and witness is not None:
        for i, v in enumerate(witness):
            print(f"(X_{i} {v:.17g})")

    if result_file is not None:
        lines = [status]
        if status == "sat" and witness is not None:
            for i, v in enumerate(witness):
                lines.append(f"(X_{i} {v:.17g})")
        result_file.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# VNNLIB parser (minimal — matches what make_box_vnnlib.py emits)
# ---------------------------------------------------------------------------

_GE_RE = re.compile(r"\(assert\s+\(>=\s+X_(\d+)\s+([-+0-9.eE]+)\)\)")
_LE_RE = re.compile(r"\(assert\s+\(<=\s+X_(\d+)\s+([-+0-9.eE]+)\)\)")
_YGE_RE = re.compile(r"\(>=\s+Y_(\d+)\s+Y_(\d+)\)")
_DECL_Y_RE = re.compile(r"\(declare-const\s+Y_(\d+)\s+Real\)")


def parse_vnnlib_box(path: Path) -> Tuple[np.ndarray, float, int, int]:
    """Parse a VNNLIB file produced by `make_box_vnnlib`. Returns
    (center, eps, label, num_outputs). Center has shape (input_dim,)."""
    text = path.read_text()

    los: Dict[int, float] = {}
    his: Dict[int, float] = {}
    for m in _GE_RE.finditer(text):
        los[int(m.group(1))] = float(m.group(2))
    for m in _LE_RE.finditer(text):
        his[int(m.group(1))] = float(m.group(2))

    if not los or set(los.keys()) != set(his.keys()):
        raise ValueError(f"VNNLIB {path} missing matching X_i bounds")

    idxs = sorted(los.keys())
    if idxs != list(range(len(idxs))):
        raise ValueError(f"VNNLIB X_i indices not contiguous: {idxs}")

    lo = np.array([los[i] for i in idxs], dtype=np.float64)
    hi = np.array([his[i] for i in idxs], dtype=np.float64)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if not np.allclose(half, half[0]):
        raise ValueError(f"VNNLIB {path} is not a symmetric L_inf box (per-coord eps differs)")
    eps = float(half[0])

    num_outputs = max((int(m.group(1)) for m in _DECL_Y_RE.finditer(text)), default=-1) + 1
    if num_outputs < 2:
        raise ValueError(f"VNNLIB {path} declares fewer than 2 outputs")

    # Output condition for binary case: (>= Y_other Y_label). For multi-class
    # disjunction, label is the Y_? that appears as the *second* argument in
    # every (>= Y_j Y_label) clause (make_box_vnnlib emits it consistently).
    matches = list(_YGE_RE.finditer(text))
    if not matches:
        raise ValueError(f"VNNLIB {path} has no Y_i comparison clause")
    labels = {int(m.group(2)) for m in matches}
    if len(labels) != 1:
        raise ValueError(f"VNNLIB {path} has multiple distinct labels: {labels}")
    label = labels.pop()

    return center, eps, label, num_outputs


# ---------------------------------------------------------------------------
# PNN reconstruction from sidecar JSON
# ---------------------------------------------------------------------------

def _sidecar_path_for(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(onnx_path.suffix + ".pnn.json")


def reconstruct_pnn(sidecar_path: Path):
    """Reconstruct a PolynomialNeuralNetwork from the sidecar JSON the
    algebraic constructor writes alongside the ONNX."""
    import torch
    from src.pnn import PolynomialNeuralNetwork

    blob = json.loads(sidecar_path.read_text())
    arch = blob["arch"]
    model = PolynomialNeuralNetwork(
        input_dim=int(arch["input_dim"]),
        output_dim=int(arch["output_dim"]),
        hidden_dims=list(arch["hidden_dims"]),
        act_degree=int(arch["act_degree"]),
        homogeneous=bool(arch.get("homogeneous", False)),
        bias=bool(arch.get("bias", True)),
        s=float(arch.get("s", 1.0)),
        trainable=bool(arch.get("trainable", True)),
    ).eval()

    state = {}
    for k, v in blob["state_dict"].items():
        arr = np.array(v["data"], dtype=np.float64).reshape(v["shape"])
        state[k] = torch.tensor(arr, dtype=torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"sidecar has unexpected keys: {unexpected}")
    if missing:
        # PolynomialActivation buffers can be missing when trainable=False; ok
        pass
    return model


# ---------------------------------------------------------------------------
# Witness extraction (when r < eps)
# ---------------------------------------------------------------------------

def _argmax_at(model, x: np.ndarray) -> int:
    import torch
    with torch.no_grad():
        y = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    return int(y.argmax(dim=1).item())


def _try_make_witness(
    model,
    center: np.ndarray,
    eps: float,
    label: int,
    closest_sol: np.ndarray,
) -> Optional[np.ndarray]:
    """If `closest_sol` (a boundary point near `center`) lies strictly inside
    the L_inf box AND nudging it slightly past the boundary still keeps it in
    the box AND flips argmax, return that nudged witness. Else None."""
    sol = np.asarray(closest_sol, dtype=np.float64).flatten()
    if sol.shape != center.shape:
        return None
    if np.any(np.abs(sol - center) > eps):
        return None  # boundary point itself is outside the box

    # Push from center past the boundary in the same direction
    direction = sol - center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    for step in (1.01, 1.05, 1.1, 1.5):
        candidate = center + step * direction
        if np.any(np.abs(candidate - center) > eps):
            continue
        if _argmax_at(model, candidate) != label:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Algebraic HC verifier wrapper (VNN-COMP-shaped CLI)",
    )
    ap.add_argument("onnx", type=str, help="Path to model.onnx (sidecar JSON must sit alongside)")
    ap.add_argument("vnnlib", type=str, help="Path to property.vnnlib")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="Wall-clock seconds (default 300).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Honored for the PyTorch forward pass during witness check; HC always runs on CPU.")
    ap.add_argument("--result-file", default=None,
                    help="Optional path to write a VNN-COMP-style result line.")
    args = ap.parse_args(argv)

    t0 = time.time()
    result_file = Path(args.result_file) if args.result_file else None

    onnx_path = Path(args.onnx).resolve()
    vnnlib_path = Path(args.vnnlib).resolve()
    sidecar_path = _sidecar_path_for(onnx_path)

    if not sidecar_path.exists():
        sys.stderr.write(
            f"[verify_vnnlib] sidecar not found: {sidecar_path}\n"
            "  This wrapper requires the .pnn.json emitted by the algebraic\n"
            "  constructor; pure ONNX-only inputs are not supported.\n"
        )
        _print_result("unknown", result_file=result_file)
        return 2

    # 1. Reconstruct PNN and save in a temp experiment dir
    try:
        model = reconstruct_pnn(sidecar_path)
    except Exception as e:
        sys.stderr.write(f"[verify_vnnlib] failed to reconstruct PNN: {e}\n")
        _print_result("unknown", result_file=result_file)
        return 2

    if model.output_dim != 2:
        sys.stderr.write(
            f"[verify_vnnlib] only binary classifiers are supported "
            f"(got output_dim={model.output_dim})\n"
        )
        _print_result("unknown", result_file=result_file)
        return 2

    # 2. Parse VNNLIB
    try:
        center, eps, label, num_outputs = parse_vnnlib_box(vnnlib_path)
    except Exception as e:
        sys.stderr.write(f"[verify_vnnlib] VNNLIB parse failed: {e}\n")
        _print_result("unknown", result_file=result_file)
        return 2

    if center.size != model.input_dim:
        sys.stderr.write(
            f"[verify_vnnlib] dimension mismatch: VNNLIB={center.size}, "
            f"model={model.input_dim}\n"
        )
        _print_result("unknown", result_file=result_file)
        return 2

    # Sanity: does the model agree with `label` at the box center?
    pred_at_center = _argmax_at(model, center)
    if pred_at_center != label:
        # The model already disagrees at the center — by VNN-COMP convention
        # the center itself is a witness.
        _print_result("sat", witness=center.tolist(), result_file=result_file)
        return 1

    # 3. Save the reconstructed PNN and call compute_robust_radius
    from src.utils import save_model
    from src.hc import compute_robust_radius

    with tempfile.TemporaryDirectory(prefix="verify_vnnlib_") as tmpdir:
        exp_dir = Path(tmpdir) / "exp"
        exp_dir.mkdir(parents=True)
        save_model(model, path=exp_dir / "model", create_experiment=False)

        overhead_budget = max(5.0, time.time() - t0)
        remaining = max(1.0, args.timeout - overhead_budget)

        try:
            results = compute_robust_radius(
                experiment_path=exp_dir,
                xi_list=[center.tolist()],
                verbose=False,
                save_results=False,
                num_threads="auto",
                timeout=remaining,
            )
        except TimeoutError as e:
            sys.stderr.write(f"[verify_vnnlib] HC timed out: {e}\n")
            _print_result("timeout", result_file=result_file)
            return 2
        except Exception as e:
            sys.stderr.write(f"[verify_vnnlib] HC failed: {e}\n")
            _print_result("unknown", result_file=result_file)
            return 2

    min_dist = np.asarray(results["min_dist"]).flatten()
    closest_sol = np.asarray(results["closest_sol"])
    if min_dist.size == 0 or not np.isfinite(min_dist[0]):
        _print_result("unknown", result_file=result_file)
        return 2

    r = float(min_dist[0])
    n = int(model.input_dim)
    threshold_unsat = eps * np.sqrt(n)

    # 4. Translate L2 radius -> L_inf box verdict
    if r >= threshold_unsat:
        # The L_inf box of radius eps is contained in the certified L2 ball
        _print_result("unsat", result_file=result_file)
        return 0

    if r < eps:
        # The closest boundary point lies strictly inside the L_inf box.
        # Try to upgrade to a concrete witness.
        sol = closest_sol[0] if closest_sol.ndim > 1 else closest_sol
        witness = _try_make_witness(model, center, eps, label, sol)
        if witness is not None:
            _print_result("sat", witness=witness.tolist(), result_file=result_file)
            return 1

    # The L2 cert can't decide the L_inf box, or we couldn't construct a
    # witness past the boundary that stays inside the box.
    _print_result("unknown", result_file=result_file)
    return 2


if __name__ == "__main__":
    sys.exit(main())

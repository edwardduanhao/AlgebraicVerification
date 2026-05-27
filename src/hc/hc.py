"""
Python interface to Julia homotopy continuation analysis.

This module provides a Pythonic wrapper around Julia functions for
robust radius computation and verification using homotopy continuation.

The public API (``compute_robust_radius``) executes Julia work in an
isolated Python subprocess so it remains safe to call from processes that
have already imported PyTorch or other heavy native libraries.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Union, Optional, List

import numpy as np


def _display_path(path: Union[str, Path], anchor: Optional[Path] = None) -> str:
    """Return a pretty display string for ``path``; falls back to absolute."""
    p = Path(path)
    if anchor is not None:
        try:
            return str(p.relative_to(anchor))
        except ValueError:
            pass
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def _initialize_julia(num_threads: Union[int, str] = "auto"):
    """Initialize Julia in the current process (intended for the worker subprocess).

    The worker subprocess calls this exactly once. The Julia thread count and
    signal-handling mode must be set in ``os.environ`` BEFORE ``juliacall`` is
    imported, so this function does both before importing.

    Args:
        num_threads: Number of Julia threads. ``"auto"`` (default) uses all
                     available cores; an integer pins to a specific count.
    """
    try:
        # Must be set before juliacall is imported.
        os.environ["JULIA_NUM_THREADS"] = str(num_threads)

        # Required for stability when Julia uses multiple threads. juliacall
        # warns and segfaults are possible without this. Set unconditionally
        # so an inherited "no" from the parent shell can't break us.
        # See: https://juliapy.github.io/PythonCall.jl/stable/faq/#Is-PythonCall/JuliaCall-thread-safe
        os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

        from juliacall import Main as jl

        # Get the project root (parent of parent of this file)
        project_root = Path(__file__).parent.parent.parent

        print(f"Initializing Julia environment...")
        print(f"Project root: {project_root}")

        # Activate the local Julia environment
        project_toml = project_root / "Project.toml"
        if project_toml.exists():
            print(f"Activating Julia project environment...")
            jl.seval(f'using Pkg; Pkg.activate("{project_root}")')
            # Activating alone does NOT install missing dependencies. On a fresh
            # clone (or any machine whose depot hasn't already resolved this
            # project) loading the module then fails with e.g.
            #   "Package OpenSSL_jll [...] is required but does not seem to be
            #    installed" -- a transitive binary JLL pulled in via HDF5.
            # Pkg.instantiate() resolves the project and installs whatever the
            # juliacall runtime depot is missing. It is a fast no-op once the
            # environment is already satisfied, so it is safe to run every init.
            print("Instantiating Julia project (installs any missing deps)...")
            jl.seval("Pkg.instantiate()")
            print("Julia project activated and instantiated.")
        else:
            print("Warning: No Project.toml found. Using default Julia environment.")

        # Include the Julia module
        julia_file = project_root / "src" / "hc" / "EuclideanHC.jl"
        if not julia_file.exists():
            raise FileNotFoundError(f"Julia file not found: {julia_file}")

        print(f"Loading Julia module: {julia_file.name}")
        jl.seval(f'include("{julia_file}")')
        jl.seval("using .EuclideanHC")

        print("✓ Julia environment initialized successfully.\n")
        return jl

    except ImportError:
        raise ImportError(
            "juliacall not installed. Install it with: pip install juliacall"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Julia: {e}")


def _compute_robust_radius_inprocess(
    experiment_path: Union[str, Path],
    xi_list: Union[List[List[float]], np.ndarray],
    output_path: Union[str, Path],
    verbose: bool = False,
    save_detailed: bool = False,
    num_threads: Union[int, str] = "auto",
) -> None:
    """Run robust radius computation in the current process.

    This is the in-process entry point used by the worker subprocess. End users
    should call ``compute_robust_radius`` instead, which spawns a worker.

    Args:
        experiment_path: Path to the experiment directory containing
                         ``model/model_config.json`` and ``model/model_weights.h5``.
        xi_list: Query points as a list-of-lists or numpy array of shape
                 (n_points, dim).
        output_path: Full file path where the NPZ result will be written.
                     The companion ``timing.npz`` and (if requested) the
                     ``hc_detailed/`` directory are placed alongside it.
        verbose: If True, print Julia-side progress.
        save_detailed: If True, also save the full HC solution set per point.
        num_threads: Julia threads for parallel path tracking. ``"auto"`` uses
                     all available cores; an integer pins to that count.
    """
    jl = _initialize_julia(num_threads=num_threads)
    project_root = Path(__file__).parent.parent.parent

    exp_path = Path(experiment_path).resolve()
    if not exp_path.exists():
        raise FileNotFoundError(
            f"Experiment path not found: {_display_path(exp_path, project_root)}"
        )

    if not isinstance(xi_list, np.ndarray):
        xi_list = np.array(xi_list)
    if xi_list.ndim == 1:
        xi_list = xi_list.reshape(1, -1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nCalling Julia robust_radius function...")
    print(f"Experiment: {_display_path(exp_path, project_root)}")
    print(f"Number of points: {len(xi_list)}")

    # Pass numpy array directly; Julia dispatches to the matrix overload and
    # converts to Vector{Vector{Float64}} internally.
    jl.seval("EuclideanHC.robust_radius")(
        str(exp_path),
        xi_list,
        verbose=verbose,
        save_path=str(output_path),
        save_detailed=save_detailed,
    )


_STDERR_TAIL_LINES = 80


def _tee_stream(src, dst, ring):
    """Read bytes from ``src`` line-by-line, write to ``dst``, keep tail in ``ring``."""
    try:
        for line in iter(src.readline, b""):
            ring.append(line)
            try:
                dst.write(line)
                dst.flush()
            except Exception:
                # Best-effort tee; don't let an output-stream failure kill the worker.
                pass
    finally:
        try:
            src.close()
        except Exception:
            pass


def _run_compute_robust_radius_subprocess(
    experiment_path: Union[str, Path],
    xi_list: Union[List[List[float]], np.ndarray],
    verbose: bool = False,
    save_results: bool = True,
    output_filename: str = "robust_radius.npz",
    save_detailed: bool = False,
    num_threads: Union[int, str] = "auto",
    timeout: Optional[float] = None,
) -> dict:
    """Run Julia verification in an isolated worker process.

    Worker stdout and stderr are streamed live to the parent's stdout/stderr.
    The last ``_STDERR_TAIL_LINES`` of stderr are also kept in memory so they
    can be included in the exception message if the worker exits non-zero.
    """
    if save_detailed and not save_results:
        raise ValueError(
            "save_detailed=True requires save_results=True (the detailed "
            "results directory lives next to the saved NPZ)."
        )

    project_root = Path(__file__).parent.parent.parent
    exp_path = Path(experiment_path).resolve()

    if not exp_path.exists():
        raise FileNotFoundError(
            f"Experiment path not found: {_display_path(exp_path, project_root)}"
        )

    xi_array = np.asarray(xi_list)
    if xi_array.ndim == 1:
        xi_array = xi_array.reshape(1, -1)

    with tempfile.TemporaryDirectory(prefix="hc_worker_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        xi_path = tmpdir_path / "xi.npy"
        np.save(xi_path, xi_array)

        # Pick where the worker writes its output. Transient results go to
        # tmpdir so concurrent calls on the same experiment can't collide.
        if save_results:
            analysis_dir = exp_path / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            output_path = analysis_dir / output_filename
        else:
            output_path = tmpdir_path / "_results.npz"

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--experiment-path", str(exp_path),
            "--xi-path", str(xi_path),
            "--output-path", str(output_path),
            "--num-threads", str(num_threads),
        ]
        if verbose:
            cmd.append("--verbose")
        if save_detailed:
            cmd.append("--save-detailed")

        env = os.environ.copy()
        # Unconditional: an inherited "no" would silently break multithreaded
        # Julia under signal-driven GC.
        env["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

        stdout_dst = getattr(sys.stdout, "buffer", sys.stdout)
        stderr_dst = getattr(sys.stderr, "buffer", sys.stderr)
        stdout_ring: deque = deque(maxlen=1)
        stderr_ring: deque = deque(maxlen=_STDERR_TAIL_LINES)

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        t_out = threading.Thread(
            target=_tee_stream, args=(proc.stdout, stdout_dst, stdout_ring), daemon=True
        )
        t_err = threading.Thread(
            target=_tee_stream, args=(proc.stderr, stderr_dst, stderr_ring), daemon=True
        )
        t_out.start()
        t_err.start()
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            t_out.join()
            t_err.join()
            raise RuntimeError(
                f"Julia worker exceeded timeout of {timeout}s and was killed."
            )
        t_out.join()
        t_err.join()

        if returncode != 0:
            stderr_tail = b"".join(stderr_ring).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Julia worker failed with exit code {returncode}.\n"
                f"--- last {_STDERR_TAIL_LINES} lines of worker stderr ---\n"
                f"{stderr_tail}"
            )

        if not output_path.exists():
            raise RuntimeError(
                f"Julia worker exited 0 but did not produce results at {output_path}."
            )

        with np.load(output_path) as data:
            results = {
                "xi_list": np.array(data["xi_list"]),
                "min_dist": np.array(data["min_dist"]),
                "closest_sol": np.array(data["closest_sol"]),
            }
        if save_results:
            results["save_path"] = str(output_path)

        timing_path = output_path.parent / "timing.npz"
        if timing_path.exists():
            with np.load(timing_path) as t:
                results["timing"] = {
                    "model_load_wall_s": float(t["model_load_wall_s"]),
                    "model_load_compile_s": float(t["model_load_compile_s"]),
                    "instance_wall_s": np.array(t["instance_wall_s"]).astype(float),
                    "instance_compile_s": np.array(t["instance_compile_s"]).astype(float),
                    "n_threads": int(t["n_threads"]),
                }

        if save_detailed:
            results["detailed_dir"] = str(output_path.parent / "hc_detailed")

        return results


def compute_robust_radius(
    experiment_path: Union[str, Path],
    xi_list: Union[List[List[float]], np.ndarray],
    verbose: bool = False,
    save_results: bool = True,
    output_filename: str = "robust_radius.npz",
    save_detailed: bool = False,
    num_threads: Union[int, str] = "auto",
    timeout: Optional[float] = None,
) -> dict:
    """
    Compute the certified robust radius for a list of query points.

    Executes Julia HomotopyContinuation in a fresh subprocess so callers can
    safely invoke this after importing PyTorch or other native libraries.

    Args:
        experiment_path: Directory containing ``model/model_config.json`` and
                         ``model/model_weights.h5`` (typically produced by
                         ``src.utils.save_model``).
        xi_list: Query points; list-of-lists or numpy array of shape
                 ``(n_points, input_dim)``.
        verbose: Stream Julia-side per-point progress.
        save_results: If True (default), persist the NPZ output under
                      ``experiment_path/analysis/<output_filename>``.
                      If False, results live only in the returned dict.
        output_filename: Output NPZ filename when ``save_results=True``.
        save_detailed: Also persist the full HC solution set per point under
                       ``analysis/hc_detailed/``. Requires ``save_results=True``.
        num_threads: Julia thread count for parallel path tracking. ``"auto"``
                     (default) uses all available cores.
        timeout: Optional wall-clock seconds before the worker is killed.

    Returns:
        Dict with keys ``xi_list``, ``min_dist``, ``closest_sol``, plus
        ``timing`` and ``save_path``/``detailed_dir`` when applicable.
    """
    return _run_compute_robust_radius_subprocess(
        experiment_path=experiment_path,
        xi_list=xi_list,
        verbose=verbose,
        save_results=save_results,
        output_filename=output_filename,
        save_detailed=save_detailed,
        num_threads=num_threads,
        timeout=timeout,
    )


def load_robust_radius_results(
    experiment_path: Union[str, Path], filename: str = "robust_radius.npz"
) -> dict:
    """
    Load previously computed robust radius results.

    Args:
        experiment_path: Path to experiment directory
        filename: Name of the results file (default: "robust_radius.npz")

    Returns:
        Dictionary containing the results

    Example:
        >>> from src.hc import load_robust_radius_results
        >>> results = load_robust_radius_results("experiments/latest")
        >>> print(f"Robust radii: {results['min_dist']}")
    """
    exp_path = Path(experiment_path).resolve()
    results_path = exp_path / "analysis" / filename

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {_display_path(results_path)}")

    data = np.load(results_path)
    return {
        "xi_list": data["xi_list"],
        "min_dist": data["min_dist"],
        "closest_sol": data["closest_sol"],
        "save_path": str(results_path),
    }


def verify_experiment(experiment_path: Union[str, Path]) -> dict:
    """
    Verify an experiment setup and check for required files.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        Dictionary with verification status

    Example:
        >>> from src.hc import verify_experiment
        >>> status = verify_experiment("experiments/latest")
        >>> print(f"Valid: {status['valid']}")
    """
    exp_path = Path(experiment_path)

    # Resolve symlink if needed
    if exp_path.is_symlink():
        resolved_path = exp_path.resolve()
        is_symlink = True
    else:
        resolved_path = exp_path
        is_symlink = False

    # Check required files
    model_dir = resolved_path / "model"
    config_file = model_dir / "model_config.json"
    weights_file = model_dir / "model_weights.h5"
    analysis_dir = resolved_path / "analysis"

    status = {
        "valid": True,
        "path": _display_path(exp_path),
        "resolved_path": _display_path(resolved_path),
        "is_symlink": is_symlink,
        "model_dir_exists": model_dir.exists(),
        "config_exists": config_file.exists(),
        "weights_exist": weights_file.exists(),
        "analysis_dir_exists": analysis_dir.exists(),
        "errors": [],
    }

    # Collect errors
    if not resolved_path.exists():
        status["valid"] = False
        status["errors"].append(
            f"Experiment path does not exist: {_display_path(resolved_path)}"
        )

    if not model_dir.exists():
        status["valid"] = False
        status["errors"].append("Model directory not found")

    if not config_file.exists():
        status["valid"] = False
        status["errors"].append("Model config file not found")

    if not weights_file.exists():
        status["valid"] = False
        status["errors"].append("Model weights file not found")

    return status


def _parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Julia HC worker (internal)")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--experiment-path", required=True)
    parser.add_argument("--xi-path", required=True)
    parser.add_argument("--output-path", required=True,
                        help="Full path where the worker writes the NPZ result.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--num-threads", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_worker_args()
    if not args.worker:
        raise SystemExit("This module is intended to be launched with --worker.")

    xi_list = np.load(args.xi_path)
    _compute_robust_radius_inprocess(
        experiment_path=args.experiment_path,
        xi_list=xi_list,
        output_path=args.output_path,
        verbose=args.verbose,
        save_detailed=args.save_detailed,
        num_threads=args.num_threads,
    )

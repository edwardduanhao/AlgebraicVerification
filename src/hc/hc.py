"""
Python interface to Julia homotopy continuation analysis.

This module provides a Pythonic wrapper around Julia functions for
robust radius computation and verification using homotopy continuation.

IMPORTANT: Import this module BEFORE importing torch to avoid segfaults.
See: https://github.com/pytorch/pytorch/issues/78829
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple

# Lazy import of juliacall (only loaded when needed)
_jl = None
_julia_loaded = False


def _initialize_julia():
    """Initialize Julia environment (lazy initialization)."""
    global _jl, _julia_loaded

    if _julia_loaded:
        return _jl

    try:
        # Enable multithreaded path tracking in HomotopyContinuation.jl
        # os.environ.setdefault("JULIA_NUM_THREADS", "auto")

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
            print("Julia project activated.")
        else:
            print("Warning: No Project.toml found. Using default Julia environment.")

        # Include the Julia module
        julia_file = project_root / "src" / "hc" / "EuclideanHC.jl"
        if not julia_file.exists():
            raise FileNotFoundError(f"Julia file not found: {julia_file}")

        print(f"Loading Julia module: {julia_file.name}")
        jl.seval(f'include("{julia_file}")')
        jl.seval("using .EuclideanHC")

        _jl = jl
        _julia_loaded = True
        print("✓ Julia environment initialized successfully.\n")
        return jl

    except ImportError:
        raise ImportError(
            "juliacall not installed. Install it with: pip install juliacall"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Julia: {e}")


def compute_robust_radius(
    experiment_path: Union[str, Path],
    xi_list: Union[List[List[float]], np.ndarray],
    verbose: bool = False,
    save_results: bool = True,
    output_filename: str = "robust_radius.npz",
    save_detailed: bool = False,
) -> dict:
    """
    Compute robust radius for a set of input points using Julia.

    Args:
        experiment_path: Path to experiment directory (e.g., "experiments/latest"
                        or "experiments/run_20241202_143022")
        xi_list: List of input points, each point is a list/array of coordinates.
                 Can be a list of lists or a numpy array of shape (n_points, dim)
        verbose: If True, print detailed progress information
        save_results: If True, save results to NPZ file in experiment's analysis folder
        output_filename: Name of output file (default: "robust_radius.npz")
        save_detailed: If True, save detailed HC solutions for each point and boundary
                      Results saved to analysis/hc_detailed/point_XXX.npz

    Returns:
        Dictionary containing:
            - 'xi_list': Input points as numpy array (n_points, dim)
            - 'min_dist': Robust radii as numpy array (n_points,)
            - 'closest_sol': Closest boundary points as numpy array (n_points, dim)
            - 'save_path': Path where results were saved (if save_results=True)
            - 'detailed_dir': Path to detailed results dir (if save_detailed=True)

    Example:
        >>> from src.hc.hc import compute_robust_radius
        >>> xi_list = [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]]
        >>> results = compute_robust_radius("experiments/latest", xi_list,
        ...                                  verbose=True, save_detailed=True)
        >>> print(f"Robust radii: {results['min_dist']}")
    """
    # Initialize Julia
    jl = _initialize_julia()

    # Get the project root (parent of parent of this file)
    project_root = Path(__file__).parent.parent.parent

    # Convert experiment_path to Path and resolve
    exp_path = Path(experiment_path).resolve()
    if not exp_path.exists():
        raise FileNotFoundError(
            f"Experiment path not found: {exp_path.relative_to(project_root)}"
        )

    # Convert xi_list to numpy array if needed
    # juliacall automatically converts numpy arrays to Julia arrays
    if not isinstance(xi_list, np.ndarray):
        xi_list = np.array(xi_list)

    # Ensure it's a 2D array
    if xi_list.ndim == 1:
        xi_list = xi_list.reshape(1, -1)

    # Determine save path
    save_path = None
    if save_results:
        analysis_dir = exp_path / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        save_path = str(analysis_dir / output_filename)

    # Call Julia function
    print(f"\nCalling Julia robust_radius function...")
    print(f"Experiment: {exp_path.relative_to(project_root)}")
    print(f"Number of points: {len(xi_list)}")

    # Pass numpy array directly - Julia will dispatch to the matrix overload
    # and handle conversion to Vector{Vector{Float64}} internally
    # Use module-qualified name to avoid ambiguity
    jl.seval("EuclideanHC.robust_radius")(
        str(exp_path),
        xi_list,
        verbose=verbose,
        save_path=save_path,
        save_detailed=save_detailed,
    )

    # Load and return results
    if save_results:
        data = np.load(save_path)
        results = {
            "xi_list": data["xi_list"],
            "min_dist": data["min_dist"],
            "closest_sol": data["closest_sol"],
            "save_path": save_path,
        }

        # Load timing data
        timing_path = analysis_dir / "timing.npz"
        if timing_path.exists():
            timing_data = np.load(timing_path)
            results["timing"] = {
                "model_load_wall_s": float(timing_data["model_load_wall_s"]),
                "model_load_compile_s": float(timing_data["model_load_compile_s"]),
                "instance_wall_s": timing_data["instance_wall_s"].astype(float),
                "instance_compile_s": timing_data["instance_compile_s"].astype(float),
                "n_threads": int(timing_data["n_threads"]),
            }

        # Add detailed results directory path if detailed saving was requested
        if save_detailed:
            detailed_dir = exp_path / "analysis" / "hc_detailed"
            results["detailed_dir"] = str(detailed_dir)
            print(
                f"Detailed results saved to: {detailed_dir.relative_to(project_root)}"
            )

        return results
    else:
        print("\nWarning: Results not saved. Set save_results=True to save.")
        return {}


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
        >>> from src.hc.julia_interface import load_robust_radius_results
        >>> results = load_robust_radius_results("experiments/latest")
        >>> print(f"Robust radii: {results['min_dist']}")
    """
    exp_path = Path(experiment_path).resolve()
    results_path = exp_path / "analysis" / filename

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {_relpath(results_path)}")

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
        >>> from src.hc.julia_interface import verify_experiment
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
        "path": _relpath(exp_path),
        "resolved_path": _relpath(resolved_path),
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
            f"Experiment path does not exist: {_relpath(resolved_path)}"
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


if __name__ == "__main__":
    _initialize_julia()
    print("Module hc.py loaded successfully.")

import torch
import h5py, json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    import torch.nn as nn


# ============================================================================
# Project Root Detection
# ============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory.

    Looks for Project.toml or .git to identify the project root.
    Falls back to parent directories if running from src/utils/.

    Returns:
        Path: Absolute path to the project root directory.
    """
    # Start from this file's directory
    current_path = Path(__file__).resolve().parent

    # Search up the directory tree for markers
    for parent in [current_path] + list(current_path.parents):
        # Check for project markers
        if (parent / "Project.toml").exists() or (parent / ".git").exists():
            return parent

    # Fallback: assume this file is in src/utils/, so project root is 2 levels up
    return Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Experiment Management
# ============================================================================


def create_experiment_dir(base_dir: str | Path | None = None) -> Path:
    """
    Create a new experiment directory with timestamp.

    Structure:
        <project_root>/experiments/
        └── run_YYYYMMDD_HHMMSS/
            ├── model/
            └── analysis/

    Args:
        base_dir (str | Path | None): Base directory for experiments.
            If None, defaults to <project_root>/experiments.

    Returns:
        Path: Path to the created experiment directory (run_YYYYMMDD_HHMMSS).
    """
    if base_dir is None:
        base_dir = get_project_root() / "experiments"
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"run_{timestamp}"

    # Create subdirectories
    (exp_dir / "model").mkdir(parents=True, exist_ok=True)
    (exp_dir / "analysis").mkdir(parents=True, exist_ok=True)

    # Create/update 'latest' symlink
    latest_link = base_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(exp_dir.name, target_is_directory=True)

    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


def get_experiment_path(
    run_name: str = "latest", base_dir: str | Path | None = None
) -> Path:
    """
    Get path to an experiment directory.

    Args:
        run_name (str): Name of the run (e.g., 'run_20241202_143022' or 'latest').
        base_dir (str | Path | None): Base directory for experiments.
            If None, defaults to <project_root>/experiments.

    Returns:
        Path: Path to the experiment directory.
    """
    if base_dir is None:
        base_dir = get_project_root() / "experiments"
    base_dir = Path(base_dir)
    exp_path = base_dir / run_name

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    # Resolve symlink if it's 'latest'
    return exp_path.resolve()


def list_experiments(base_dir: str | Path | None = None) -> list[str]:
    """
    List all experiment directories.

    Args:
        base_dir (str | Path | None): Base directory for experiments.
            If None, defaults to <project_root>/experiments.

    Returns:
        list[str]: List of experiment directory names sorted by timestamp (newest first).
    """
    if base_dir is None:
        base_dir = get_project_root() / "experiments"
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    # Get all run_* directories
    experiments = [
        d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    # Sort by timestamp (newest first)
    experiments.sort(reverse=True)
    return experiments


# ============================================================================
# Utility Functions
# ============================================================================


def c_split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the last dimension of x into real and imaginary parts.

    Args:
        x (torch.Tensor): Input tensor with last dimension of size 2n.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two tensors representing the real and imaginary parts.
    """

    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "Last dimension must be even to split real and imaginary parts."
        )

    n = x.shape[-1] // 2

    return x[..., :n], x[..., n:]


def c_join(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Join real and imaginary parts into a single stacked tensor.

    Args:
        x (torch.Tensor): Real part, dimension (..., d)
        y (torch.Tensor): Imaginary part, dimension (..., d)

    Returns:
        torch.Tensor: Stacked tensor with last dimension of size 2d.
    """

    return torch.cat([x, y], dim=-1)


def save_model(
    model: "nn.Module",
    path: str | Path | None = None,
    metadata: dict | None = None,
    base_dir: str | Path | None = None,
    create_experiment: bool = True,
) -> Path:
    """
    Save the model configuration and weights.

    If create_experiment=True (default), creates a new experiment directory with timestamp:
        <project_root>/experiments/run_YYYYMMDD_HHMMSS/model/

    If path is provided, saves directly to that path (backward compatibility).

    Args:
        model (nn.Module): The model to save.
        path (str | Path | None): Specific directory path to save the model.
            If None and create_experiment=True, creates new experiment directory.
        metadata (dict | None): Additional metadata to save (e.g., training info).
        base_dir (str | Path | None): Base directory for experiments (used if create_experiment=True).
            If None, defaults to <project_root>/experiments.
        create_experiment (bool): If True, creates new experiment directory structure.

    Returns:
        Path: Path to the experiment directory (if create_experiment=True) or model directory.
    """
    if path is None and create_experiment:
        # Create new experiment directory
        exp_dir = create_experiment_dir(base_dir)
        model_dir = exp_dir / "model"
        return_path = exp_dir
    elif path is not None:
        # Use provided path
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        return_path = model_dir
    else:
        raise ValueError("Either provide 'path' or set 'create_experiment=True'")

    # 1) Save model config
    config = {
        "model_class": model.__class__.__name__,
        "input_dim": model.input_dim,
        "hidden_dims": list(model.hidden_dims),
        "output_dim": model.output_dim,
        "act_degree": model.act_degree,
        "homogeneous": model.homogeneous,
        "bias": model.bias,
    }

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 2) Save weights
    state = model.state_dict()
    with h5py.File(model_dir / "model_weights.h5", "w") as f:
        for name, tensor in state.items():
            f.create_dataset(name, data=tensor.detach().cpu().numpy())

    # 3) Save metadata if provided
    if metadata is not None:
        # Add timestamp to metadata
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Model saved to: {model_dir}")
    return return_path

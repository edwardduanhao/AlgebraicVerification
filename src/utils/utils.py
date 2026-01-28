import torch
import h5py, json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_path.relative_to(base_dir)}"
        )

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
# Model Saving and Loading
# ============================================================================


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

    return return_path


# ============================================================================
# Plotting
# ============================================================================


def plot_db(
    model: "nn.Module",
    x_range: tuple,
    y_range: tuple,
    grid_size: int = 1000,
    save: str | Path = "decision_boundary.pdf",
    X: np.ndarray = None,
    y: np.ndarray = None,
    show: bool = True,
):
    """
    Show decision boundary of the model.

    Args:
        model (nn.Module): Trained model
        X (np.ndarray): Training data features
        y (np.ndarray): Training data labels
        x_range (tuple): Range for x-axis
        y_range (tuple): Range for y-axis
        grid_size (int): Number of points in each dimension for the grid (default 1000)
        save (str | Path): File path to save the plot (default "decision_boundary.pdf")
        show (bool): Whether to display the plot in browser (default True)
    """

    # Create a grid for plotting decision boundary
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Flatten grid for prediction
    grid_points = torch.tensor(
        np.c_[X_grid.ravel(), Y_grid.ravel()], dtype=torch.float32
    )

    # Get model predictions
    model.eval()
    with torch.no_grad():
        logits = model(grid_points)
        predictions = torch.argmax(logits, dim=1)

    # Reshape predictions back to grid
    Z = predictions.numpy().reshape(X_grid.shape)

    # Determine number of classes
    num_classes = logits.shape[1]

    # Create color palette for multiple classes
    # Using plotly's Plotly color sequence for good visual separation
    colors = [
        "rgb(99, 110, 250)",  # blue
        "rgb(239, 85, 59)",  # red
        "rgb(0, 204, 150)",  # green
        "rgb(171, 99, 250)",  # purple
        "rgb(255, 161, 90)",  # orange
        "rgb(25, 211, 243)",  # cyan
        "rgb(255, 102, 146)",  # pink
        "rgb(182, 232, 128)",  # lime
        "rgb(255, 151, 255)",  # magenta
        "rgb(254, 203, 82)",  # yellow
    ]

    # Extend colors if needed for more classes
    while len(colors) < num_classes:
        colors.extend(colors)

    # Create custom colorscale for heatmap based on number of classes
    if num_classes == 2:
        colorscale = [
            [0.0, "rgb(100, 149, 237)"],
            [1.0, "rgb(255, 102, 102)"],
        ]
    else:
        # For multi-class, create discrete colorscale
        colorscale = []
        for i in range(num_classes):
            colorscale.append(
                [i / (num_classes - 1) if num_classes > 1 else 0, colors[i]]
            )

    fig = go.Figure()

    # Add decision boundary heatmap
    fig.add_trace(
        go.Heatmap(
            x=x_grid,
            y=y_grid,
            z=Z,
            colorscale=colorscale,
            opacity=0.3,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Add training data points for each class
    if X is not None and y is not None:
        for class_idx in range(num_classes):
            mask = y == class_idx
            if isinstance(y, torch.Tensor):
                X_class = (
                    X[mask].numpy() if isinstance(X, torch.Tensor) else X[mask.numpy()]
                )
            else:
                X_class = X[mask].numpy() if isinstance(X, torch.Tensor) else X[mask]

            fig.add_trace(
                go.Scatter(
                    x=X_class[:, 0],
                    y=X_class[:, 1],
                    mode="markers",
                    name=f"Class {class_idx}",
                    marker=dict(
                        color=colors[class_idx],
                        size=6,
                        opacity=0.8,
                        line=dict(color="white", width=0.5),
                    ),
                )
            )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        xaxis_title="x0",
        yaxis_title="x1",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        width=800,
        height=800,
    )

    if show:
        fig.show(renderer="browser")

    if save is not None:
        fig.write_image(str(save), width=600, height=600, scale=2)


def plot_db_3d(
    model: "nn.Module",
    x_range: tuple,
    y_range: tuple,
    z_range: tuple,
    grid_size: int = 50,
    save: str | Path = "decision_boundary_3d.pdf",
    X: np.ndarray = None,
    y: np.ndarray = None,
    show: bool = True,
):
    """
    Show decision boundary of the model in 3D space as a light red surface.

    Args:
        model (nn.Module): Trained model with 3 input dimensions
        x_range (tuple): Range for x-axis
        y_range (tuple): Range for y-axis
        z_range (tuple): Range for z-axis
        grid_size (int): Number of points in each dimension for the grid (default 50)
        save (str | Path): File path to save the plot (default "decision_boundary_3d.pdf")
        X (np.ndarray): Training data features (shape: [n, 3])
        y (np.ndarray): Training data labels
        show (bool): Whether to display the plot in browser (default True)
    """

    # Create a 3D grid for plotting decision boundary
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    z_grid = np.linspace(z_range[0], z_range[1], grid_size)
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid)

    # Flatten grid for prediction
    grid_points = torch.tensor(
        np.c_[X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], dtype=torch.float32
    )

    # Get model predictions
    model.eval()
    with torch.no_grad():
        logits = model(grid_points)
        predictions = torch.argmax(logits, dim=1)

    # Reshape predictions and logits back to grid
    predictions_grid = predictions.numpy().reshape(X_grid.shape)
    logits_grid = logits.numpy().reshape(X_grid.shape + (logits.shape[1],))

    # Calculate decision boundary (where prediction changes)
    # For binary classification: logit[0] - logit[1] = 0
    decision_values = logits_grid[:, :, :, 0] - logits_grid[:, :, :, 1]

    fig = go.Figure()

    # Add decision boundary as light red isosurface
    fig.add_trace(
        go.Isosurface(
            x=X_grid.flatten(),
            y=Y_grid.flatten(),
            z=Z_grid.flatten(),
            value=decision_values.flatten(),
            isomin=-0.1,
            isomax=0.1,
            opacity=0.6,
            surface_count=1,
            colorscale=[[0, "rgb(255, 150, 150)"], [1, "rgb(255, 150, 150)"]],
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    # Add training data points if provided
    if X is not None and y is not None:
        num_classes = logits.shape[1]
        colors = ["rgb(50, 150, 255)", "rgb(230, 50, 50)"]  # blue, red

        for class_idx in range(num_classes):
            mask = y == class_idx
            if isinstance(y, torch.Tensor):
                X_class = (
                    X[mask].numpy() if isinstance(X, torch.Tensor) else X[mask.numpy()]
                )
            else:
                X_class = X[mask].numpy() if isinstance(X, torch.Tensor) else X[mask]

            fig.add_trace(
                go.Scatter3d(
                    x=X_class[:, 0],
                    y=X_class[:, 1],
                    z=X_class[:, 2],
                    mode="markers",
                    name=f"Class {class_idx}",
                    marker=dict(
                        color=colors[class_idx],
                        size=4,
                        opacity=0.8,
                        line=dict(color="white", width=0.5),
                    ),
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="x0",
            yaxis_title="x1",
            zaxis_title="x2",
            xaxis=dict(showgrid=False, showbackground=False),
            yaxis=dict(showgrid=False, showbackground=False),
            zaxis=dict(showgrid=False, showbackground=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        ),
        showlegend=True,
        width=800,
        height=800,
    )

    if show:
        fig.show(renderer="browser")

    if save is not None:
        fig.write_image(str(save), width=800, height=800, scale=2)

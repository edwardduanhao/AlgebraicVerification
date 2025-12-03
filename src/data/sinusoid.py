import math
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def generate_sinusoidal_data(
    n_samples: int,
    noise_level: float = 0.1,
    x_range: tuple[float, float] = (-2 * math.pi, 2 * math.pi),
    spread: float = 2.0,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 2D dataset with a sinusoidal decision boundary.

    Args:
        n_samples (int): _description_. Defaults to 1000.
        noise_level (float, optional): _description_. Defaults to 0.1.
        x_range (tuple[float, float], optional): _description_. Defaults to (-2 * pi, 2 * pi).
        spread (float, optional): _description_. Defaults to 2.0.
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _tuple[torch.Tensor, torch.Tensor]_: Features and labels tensors.
    """

    torch.manual_seed(seed)

    # Generate random x values
    x = torch.rand(n_samples) * (x_range[1] - x_range[0]) + x_range[0]

    # Generate random y values around the sin(x) boundary
    y_boundary = torch.sin(x)

    y = torch.rand(n_samples) * (y_boundary + spread - (y_boundary - spread)) + (
        y_boundary - spread
    )
    # Add noise to the boundary itself
    noisy_boundary = y_boundary + torch.randn(n_samples) * noise_level

    # Create binary labels: 1 if above noisy boundary, 0 if below
    labels = (y > noisy_boundary).to(torch.int64)

    # Stack features
    X = torch.stack([x, y], axis=1).to(torch.float32)

    X_class0 = X[labels == 0]
    X_class1 = X[labels == 1]

    return X.to(device), labels.to(device)


def visualize_decision_boundary(
    model, X, y, x_range=(-2 * np.pi, 2 * np.pi), y_range=(-3, 3)
):
    """
    Visualize the learned decision boundary and compare with true sin(x) boundary.

    Args:
        model: Trained PolynomialNetwork
        X: Training data features
        y: Training data labels
        x_range: Range for x-axis
        y_range: Range for y-axis
    """
    # Create a grid for plotting decision boundary

    x_grid = np.linspace(x_range[0], x_range[1], 1000)
    y_grid = np.linspace(y_range[0], y_range[1], 1000)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Flatten grid for prediction
    grid_points = torch.tensor(
        np.c_[X_grid.ravel(), Y_grid.ravel()], dtype=torch.float32
    )

    # Get model predictions
    model.eval()
    with torch.no_grad():
        logits = model(grid_points)
        # predictions = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1
        predictions = torch.sigmoid(logits).squeeze()  # For binary classification

    # Reshape predictions back to grid
    Z = predictions.numpy().reshape(X_grid.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_grid,
            y=y_grid,
            z=Z,
            colorscale=[
                [0.0, "rgb(100, 149, 237)"],
                [0.5, "rgb(255, 255, 255)"],
                [1.0, "rgb(255, 102, 102)"],
            ],
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="P(y=1)"),
            zsmooth="best",
        )
    )

    # Add training data points
    fig.add_trace(
        go.Scatter(
            x=X[y == 0][:, 0].numpy(),
            y=X[y == 0][:, 1].numpy(),
            mode="markers",
            name="Class 0",
            marker=dict(color="blue", size=5, opacity=0.6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X[y == 1][:, 0].numpy(),
            y=X[y == 1][:, 1].numpy(),
            mode="markers",
            name="Class 1",
            marker=dict(color="red", size=5, opacity=0.6),
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.6)",
        ),
        xaxis_title="x0",
        yaxis_title="x1",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        width=800,
        height=800,
    )

    fig.update_traces(
        selector=dict(type="heatmap"),
        colorbar=dict(title="P(y=1)"),
    )

    fig.write_image("decision_boundary.pdf", width=800, height=800, scale=2)

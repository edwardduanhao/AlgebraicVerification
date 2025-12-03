import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm, trange
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def train_model(model, X_train, y_train, epochs=1000, lr=0.01):
    """
    Train the polynomial network for binary classification.

    Args:
        model: PolyNetwork instance
        X_train: Training features
        y_train: Training labels
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        losses: List of training losses
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    accuracies = []

    pbar = trange(epochs, desc="Training Progress")
    for epoch in pbar:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_train).float().mean().item()

        losses.append(loss.item())
        accuracies.append(accuracy)

        # Update progress bar with current loss and accuracy
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{accuracy:.4f}"})

    # losses, accuracies are 1D iterables of equal length
    epochs = np.arange(1, len(losses) + 1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])  # enables right y-axis

    # Left y-axis: Loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            name="Loss",
            mode="lines",
            line=dict(width=2, color="#1f77b4"),
        ),
        secondary_y=False,
    )

    # Right y-axis: Accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=accuracies,
            name="Accuracy",
            mode="lines",
            line=dict(width=2, color="#ff7f0e"),
        ),
        secondary_y=True,
    )

    # Axis titles, colors, ranges
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Training Loss", color="#1f77b4", secondary_y=False)
    fig.update_yaxes(
        title_text="Training Accuracy",
        color="#ff7f0e",
        secondary_y=True,
    )

    # Layout (size, legend, hover, margins)
    fig.update_layout(
        title="Training Loss and Accuracy",
        width=800,
        height=600,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=60, t=60, b=40),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )

    fig.update_yaxes(
        title_text="Training Accuracy",
        color="#ff7f0e",
        autorange=True,
        secondary_y=True,
    )
    fig.show()

    return losses, accuracies


class StartSystem(nn.Module):
    """
    A module representing the start system for homotopy continuation.
    G(x) = (x_1^d_1 - 1, x_2^d_2 - 1, ..., x_n^d_n - 1)

    Input: Input tensor with shape (batch, 2n)
    Output: Output tensor with shape (batch, 2n)

    """

    def __init__(self, d: torch.Tensor, gamma: torch.Tensor = None):
        """
        Args:
            d: Tensor of degrees to apply, shape (n, )
            gamma: Parameter for "gamma trick", shape (2, ), defaults to None
        """
        super().__init__()
        self.register_buffer("d", d)

        # If gamma is not provided, sample a random one.
        if gamma is None:
            gamma = torch.rand(1) * 2 * torch.pi
            gamma = torch.tensor([torch.cos(gamma), torch.sin(gamma)])

        self.register_buffer("gamma", gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Realified input tensor with shape (batch, 2n),
               where first n dims are real parts and last n dims are imaginary parts.

        Returns:
            Realified output tensor with shape (batch, 2n)
        """

        # Check dimension compatibility
        assert (
            x.shape[-1] == 2 * self.d.shape[0]
        ), f"Expected x to have shape (*, {2*self.d.shape[0]}), got {x.shape}"

        x_real, x_imag = c_split(x)

        # Convert to polar coordinates
        rho = torch.hypot(x_real, x_imag)
        theta = torch.atan2(x_imag, x_real)

        # Apply power in polar form: z^d = rho^d * exp(i*d*theta)
        rho_pow = rho.pow(self.d)

        # Convert back to Cartesian coordinates
        y_real = rho_pow * torch.cos(self.d * theta)
        y_imag = rho_pow * torch.sin(self.d * theta)

        # Apply the gamma trick, multiplying by gamma
        gamma_real, gamma_imag = self.gamma[0], self.gamma[1]
        y_real, y_imag = (
            gamma_real * y_real - gamma_imag * y_imag,
            gamma_real * y_imag + gamma_imag * y_real,
        )

        return c_join(y_real, y_imag)

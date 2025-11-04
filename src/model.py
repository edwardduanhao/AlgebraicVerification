import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm, trange
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------ Helper functions ------------------------ #


def c_split(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split the last dimension of z into real and imaginary parts.

    Args:
        z (torch.Tensor): Input tensor with last dimension of size 2d.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two tensors representing the real and imaginary parts.
    """
    assert (
        z.shape[-1] % 2 == 0
    ), "Last dimension must be even to split real and imaginary parts."
    d = z.shape[-1] // 2
    return z[..., :d], z[..., d:]


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


# ------------------------------------------------------------------- #


class CRLinear(nn.Module):
    """
    Cauchy-Riemann linear layer
    This module represents a complex affine map z -> Wz + b with W = A + iB, b = c + id.
    Forward on stacked real/imag: [u; v] = [[A, -B],[B, A]] [x; y] + [c; d].
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.A = nn.Linear(in_features, out_features, bias=False)
        self.B = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.c = nn.Parameter(torch.zeros(out_features))
            self.d = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("c", None)
            self.register_parameter("d", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "CRLinear":
        cr = cls(lin.in_features, lin.out_features, bias=(lin.bias is not None))
        with torch.no_grad():
            cr.A.weight.copy_(lin.weight)
            cr.B.weight.zero_()
            if lin.bias is not None:
                cr.c.copy_(lin.bias)
                cr.d.zero_()
        return cr

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x, y = c_split(z)
        Ax = self.A(x)
        Ay = self.A(y)
        Bx = self.B(x)
        By = self.B(y)
        u = Ax - By
        v = Bx + Ay
        if self.c is not None:
            u = u + self.c
            v = v + self.d
        return c_join(u, v)


class PolyActivation(nn.Module):
    def __init__(self, degree: int = 3, homogeneous: bool = False):
        super().__init__()
        self.degree = degree
        self.homogeneous = homogeneous
        if self.homogeneous:
            self.coefficients = nn.Parameter(torch.randn(1) * 0.1)
        else:
            self.coefficients = nn.Parameter(torch.randn(degree + 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.homogeneous:
            return self.coefficients * torch.pow(x, self.degree)
        else:
            result = self.coefficients[0] * torch.ones_like(x)
            x_power = torch.ones_like(x)
            for i in range(1, self.degree + 1):
                x_power = x_power * x
                result += self.coefficients[i] * x_power
            return result


class CRPolyActivation(nn.Module):
    """
    Cauchy-Riemann polynomial activation function.

    Uses polar form: z^n = r^n * exp(i*n*theta) = r^n * (cos(n*theta) + i*sin(n*theta))
    where z = x + iy = r*exp(i*theta)

    For improved numerical stability:
    - Uses torch.hypot for computing r = |z|
    - Properly handles the i=0 case (constant term)
    - Accumulates terms carefully to minimize rounding errors
    """

    def __init__(self, degree: int = 3, homogeneous: bool = False):
        super().__init__()
        self.degree = degree
        self.homogeneous = homogeneous
        if self.homogeneous:
            self.coeff_real = nn.Parameter(torch.randn(1) * 0.1)
            self.coeff_imag = nn.Parameter(torch.randn(1) * 0.1)
        else:
            self.coeff_real = nn.Parameter(torch.randn(degree + 1) * 0.1)
            self.coeff_imag = nn.Parameter(torch.randn(degree + 1) * 0.1)

    @classmethod
    def from_polyactivation(cls, poly: PolyActivation) -> "CRPolyActivation":
        cr = cls(degree=poly.degree, homogeneous=poly.homogeneous)
        with torch.no_grad():
            cr.coeff_real.copy_(poly.coefficients)
            cr.coeff_imag.zero_()
        return cr

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x, y = c_split(z)
        # Use torch.hypot for better numerical stability
        rho = torch.hypot(x, y)
        # Add small epsilon only where needed (for powers, not for theta)
        theta = torch.atan2(y, x)
        if self.homogeneous:
            r_pow = torch.pow(rho, self.degree)
            cos_part = torch.cos(self.degree * theta) * r_pow
            sin_part = torch.sin(self.degree * theta) * r_pow
            u = self.coeff_real * cos_part - self.coeff_imag * sin_part
            v = self.coeff_imag * cos_part + self.coeff_real * sin_part
            return c_join(u, v)
        else:
            # Initialize u, v as tensors for better numerical stability
            u = torch.zeros_like(x)
            v = torch.zeros_like(y)
            for i in range(self.degree + 1):
                if i == 0:
                    cos_part = torch.ones_like(x)
                    sin_part = torch.zeros_like(x)
                else:
                    r_pow = torch.pow(rho, i)
                    cos_part = torch.cos(i * theta) * r_pow
                    sin_part = torch.sin(i * theta) * r_pow
                u = u + self.coeff_real[i] * cos_part - self.coeff_imag[i] * sin_part
                v = v + self.coeff_imag[i] * cos_part + self.coeff_real[i] * sin_part
            return c_join(u, v)


class PolyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: int,
        polynomial_degree: int = 3,
        homogeneous=False,
    ):
        super(PolyNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.polynomial_degree = polynomial_degree
        self.homogeneous = homogeneous

        layers = []
        activations = []

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                activations.append(PolyActivation(polynomial_degree, self.homogeneous))

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        return x


class CRPolyNetwork(nn.Module):
    """
    Complex-Riemann version of PolyNetwork.
    Expects input as stacked [real; imag] vectors.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        polynomial_degree: int = 3,
        homogeneous: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.polynomial_degree = polynomial_degree
        self.homogeneous = homogeneous

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(CRLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                self.activations.append(
                    CRPolyActivation(polynomial_degree, homogeneous)
                )

    @classmethod
    def from_polynetwork(cls, model: PolyNetwork) -> "CRPolyNetwork":
        """
        Convert an instance of PolyNetwork into an instance of CRPolyNetwork.

        Args:
            model (PolyNetwork): an instance of PolyNetwork to convert

        Returns:
            CRPolyNetwork: converted CRPolyNetwork instance
        """
        cr_model = cls(
            input_dim=model.input_dim,
            output_dim=model.output_dim,
            hidden_dims=model.hidden_dims,
            polynomial_degree=model.polynomial_degree,
            homogeneous=model.homogeneous,
        )

        # Convert layers
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                cr_model.layers[i] = CRLinear.from_linear(layer)

            # Convert activations
            for i, activation in enumerate(model.activations):
                cr_model.activations[i] = CRPolyActivation.from_polyactivation(
                    activation
                )

        return cr_model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CRPolyNetwork.

        Args:
            z (torch.Tensor): input tensor with stacked real/imag parts

        Returns:
            torch.Tensor: output tensor with stacked real/imag parts
        """
        for i, layer in enumerate(self.layers[:-1]):
            z = layer(z)
            z = self.activations[i](z)
        z = self.layers[-1](z)
        return z


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


class LinearSelfAttention(nn.Module):
    """
    A simple (single-head) self-attention block:
    - Input shape: (batch_size, embed_dim, seq_len)
    - Output shape: (batch_size, embed_dim, seq_len)
    """

    def __init__(self, embed_dim):
        super(LinearSelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # Define query, key, value linear layers
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x has shape [batch_size, embed_dim, seq_len].
        We'll transform x so we treat each position in the seq_len dimension
        as a 'token' with an embedding dimension of embed_dim.
        """
        batch_size, embed_dim, seq_len = x.shape

        # Transpose to (batch_size, seq_len, embed_dim) for linear layers
        x_t = x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]

        # Compute Q, K, V
        Q = self.query(x_t)  # [batch_size, seq_len, embed_dim]
        K = self.key(x_t)  # [batch_size, seq_len, embed_dim]
        V = self.value(x_t)  # [batch_size, seq_len, embed_dim]

        # Compute scaled dot-product attention
        # QK^T shape: [batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(1, 2))

        # Multiply attention weights by V
        out = torch.matmul(scores, V)  # [batch_size, seq_len, embed_dim]

        # Reshape back to (batch_size, embed_dim, seq_len)
        out = out.transpose(1, 2)

        return out

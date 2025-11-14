import torch
import torch.nn as nn


class PolynomialActivation(nn.Module):
    """Polynomial Activation Function Module."""

    def __init__(self, degree: int, homogeneous: bool, s: float):
        """

        Args:
            degree (int): Degree of the polynomial activation function.
            homogeneous (bool): Whether the activation function is homogeneous.
                If True, the activation function will be of the form f(x) = c * x^degree.
            s (float): Scaling factor for the initial coefficients.
        """
        super().__init__()

        self.degree = degree
        self.homogeneous = homogeneous
        self.s = s

        if degree <= 0:
            raise ValueError(f"degree must be positive, got {degree}")

        if self.homogeneous:
            self.coeffs = nn.Parameter(torch.randn(1) * s)
        else:
            self.coeffs = nn.Parameter(torch.randn(degree + 1) * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.homogeneous:
            return self.coeffs * torch.pow(x, self.degree)
        else:
            # Apply Horner's method for efficient polynomial evaluation
            y = torch.zeros_like(x) + self.coeffs[-1]
            for c in reversed(self.coeffs[:-1].unbind(0)):
                y = y * x + c
            return y


class PolynomialNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        degree: int,
        homogeneous: bool = False,
        bias: bool = True,
        s: float = 1.0,
    ) -> None:
        """

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hidden_dims (list[int]): Hidden dimensions.
            degree (int): Degree of the polynomial activation function.
            homogeneous (bool, optional): Whether the activation function is homogeneous. Defaults to False.
            bias (bool, optional): Whether to include bias terms. Defaults to True.
            s (float, optional): Scaling factor for the initial coefficients. Defaults to 1.0.
        """
        super().__init__()

        # Validate dimensions
        if input_dim <= 0 or output_dim <= 0 or any(d <= 0 for d in hidden_dims):
            raise ValueError(
                f"Input and output dimensions must be positive, got {input_dim}, {output_dim}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.degree = degree
        self.homogeneous = homogeneous
        self.bias = bias
        self.s = s

        layers = []
        activations = []

        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(
                nn.Linear(in_features=dims[i], out_features=dims[i + 1], bias=bias)
            )
            if i < len(dims) - 2:
                activations.append(
                    PolynomialActivation(degree=degree, homogeneous=homogeneous, s=s)
                )

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        return x

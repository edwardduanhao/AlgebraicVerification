import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.config import ModelConfig


class PolynomialActivation(nn.Module):
    """Polynomial Activation Function Module."""

    def __init__(self, act_degree: int, homogeneous: bool, s: float):
        """

        Args:
            act_degree (int): Degree of the polynomial activation function.
            homogeneous (bool): Whether the activation function is homogeneous.
                If True, the activation function will be of the form f(x) = c * x^degree.
            s (float): Scaling factor for the initial coefficients.
        """
        super().__init__()

        self.act_degree = act_degree
        self.homogeneous = homogeneous
        self.s = s

        if act_degree <= 0:
            raise ValueError(f"act_degree must be positive, got {act_degree}")

        if self.homogeneous:
            self.coeffs = nn.Parameter(torch.randn(1) * s)
        else:
            self.coeffs = nn.Parameter(torch.randn(act_degree + 1) * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.homogeneous:
            return self.coeffs * torch.pow(x, self.act_degree)
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
        act_degree: int,
        homogeneous: bool = False,
        bias: bool = True,
        s: float = 1.0,
    ) -> None:
        """

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            hidden_dims (list[int]): Hidden dimensions.
            act_degree (int): Degree of the polynomial activation function.
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
        self.act_degree = act_degree
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
                    PolynomialActivation(
                        act_degree=act_degree, homogeneous=homogeneous, s=s
                    )
                )

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "PolynomialNeuralNetwork":
        """
        Create a PolynomialNeuralNetwork from a ModelConfig.

        Args:
            config (ModelConfig): Configuration object containing model parameters.

        Returns:
            PolynomialNeuralNetwork: Instantiated model from config.
        """
        return cls(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims,
            act_degree=config.act_degree,
            homogeneous=config.homogeneous,
            bias=config.bias,
            s=config.s,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        return x


if __name__ == "__main__":
    # Example usage
    model = PolynomialNeuralNetwork(
        input_dim=4,
        output_dim=2,
        hidden_dims=[8, 8],
        act_degree=3,
        homogeneous=False,
        bias=True,
        s=0.1,
    )

    # Print model architecture
    print(model)

    # Test forward pass
    x = torch.randn(1, 4)
    y = model(x)
    print(y)

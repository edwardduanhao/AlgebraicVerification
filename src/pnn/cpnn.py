import math
import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from src.utils import c_join, c_split

if TYPE_CHECKING:
    from pnn import PolynomialActivation, PolynomialNeuralNetwork
    from src.config.config import ModelConfig


class ComplexLinear(nn.Module):
    """
    Complex linear layer module in Cauchy-Riemann form.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """

        Args:
            in_features (int): Dimension of the input features.
            out_features (int): Dimension of the output features.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super().__init__()
        self.lin_real = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )
        self.lin_imag = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

        if bias:
            self.bias_real = nn.Parameter(
                (torch.rand(out_features) * 2 - 1) / math.sqrt(in_features)
            )
            self.bias_imag = nn.Parameter(
                (torch.rand(out_features) * 2 - 1) / math.sqrt(in_features)
            )
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "ComplexLinear":
        """
        Create a ComplexLinear layer from a standard Linear layer.

        Args:
            lin (nn.Linear): A standard linear layer.

        Returns:
            ComplexLinear: The corresponding ComplexLinear layer.
        """

        # Create a ComplexLinear layer from a standard nn.Linear layer
        clin = cls(lin.in_features, lin.out_features, bias=(lin.bias is not None))

        with torch.no_grad():
            # Copy weights and biases
            clin.lin_real.weight.copy_(lin.weight)
            clin.lin_imag.weight.zero_()
            if lin.bias is not None:
                clin.bias_real.copy_(lin.bias)
                clin.bias_imag.zero_()

        # Preserve dtype and device from source
        clin = clin.to(dtype=lin.weight.dtype, device=lin.weight.device)

        return clin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into real and imaginary parts
        x_real, x_imag = c_split(x)

        Ax = self.lin_real(x_real)
        Ay = self.lin_real(x_imag)
        Bx = self.lin_imag(x_real)
        By = self.lin_imag(x_imag)

        y_real = Ax - By
        y_imag = Bx + Ay

        if self.bias_real is not None:
            y_real = y_real + self.bias_real
            y_imag = y_imag + self.bias_imag

        return c_join(y_real, y_imag)


class ComplexPolynomialActivation(nn.Module):
    """
    Polynomial activation function for complex inputs in Cauchy-Riemann form.
    Uses polar form: x^d = r^d * exp(i*d*theta) = r^d * (cos(d*theta) + i*sin(d*theta))
    where x = x_real + i*x_imag = r*exp(i*theta).
    """

    def __init__(self, degree: int, homogeneous: bool, s: float) -> None:
        super().__init__()
        self.degree = degree
        self.homogeneous = homogeneous
        self.s = s

        if degree <= 0:
            raise ValueError(f"degree must be positive, got {degree}")

        if self.homogeneous:
            self.coeffs_real = nn.Parameter(torch.randn(1) * s)
            self.coeffs_imag = nn.Parameter(torch.randn(1) * s)
        else:
            self.coeffs_real = nn.Parameter(torch.randn(degree + 1) * s)
            self.coeffs_imag = nn.Parameter(torch.randn(degree + 1) * s)

    @classmethod
    def from_polynomial_activation(
        cls, poly: "PolynomialActivation"
    ) -> "ComplexPolynomialActivation":
        """
        Create a ComplexPolynomialActivation from a real PolynomialActivation.

        Args:
            poly (PolynomialActivation): A real polynomial activation function.

        Returns:
            ComplexPolynomialActivation: The corresponding ComplexPolynomialActivation.
        """

        cr = cls(degree=poly.degree, homogeneous=poly.homogeneous, s=poly.s)

        with torch.no_grad():
            cr.coeffs_real.copy_(poly.coeffs)
            cr.coeffs_imag.zero_()

        # Preserve dtype and device from source
        cr = cr.to(dtype=poly.coeffs.dtype, device=poly.coeffs.device)

        return cr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input into real and imaginary parts
        x_real, x_imag = c_split(x)

        # Convert to polar coordinates
        rho = torch.hypot(x_real, x_imag)
        theta = torch.atan2(x_imag, x_real)

        degree = self.degree
        coeffs_real = self.coeffs_real
        coeffs_imag = self.coeffs_imag

        if self.homogeneous:
            r_pow = torch.pow(rho, degree)
            cos_part = torch.cos(degree * theta) * r_pow
            sin_part = torch.sin(degree * theta) * r_pow
            y_real = coeffs_real * cos_part - coeffs_imag * sin_part
            y_imag = coeffs_imag * cos_part + coeffs_real * sin_part
            return c_join(y_real, y_imag)
        else:
            # Apply Horner's method for efficient polynomial evaluation
            y_real = coeffs_real[degree] * torch.ones_like(x_real)
            y_imag = coeffs_imag[degree] * torch.ones_like(x_imag)

            for i in range(degree - 1, -1, -1):
                # complex multiplication: (y_real + i*y_imag) * (x_real + i*x_imag)
                t_real = y_real * x_real - y_imag * x_imag
                t_imag = y_real * x_imag + y_imag * x_real
                # add coefficient i
                y_real = t_real + coeffs_real[i]
                y_imag = t_imag + coeffs_imag[i]
            return c_join(y_real, y_imag)


class ComplexPolynomialNeuralNetwork(nn.Module):
    """
    Complex Polynomial Neural Network using Cauchy-Riemann form.
    """

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

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(ComplexLinear(dims[i], dims[i + 1], bias=bias))
            if i < len(dims) - 2:  # No activation after last layer
                self.activations.append(
                    ComplexPolynomialActivation(
                        degree=degree, homogeneous=homogeneous, s=s
                    )
                )

    @classmethod
    def from_polynomial_neural_network(
        cls, model: "PolynomialNeuralNetwork"
    ) -> "ComplexPolynomialNeuralNetwork":
        """
        Convert an instance of PolynomialNeuralNetwork into an instance of ComplexPolynomialNeuralNetwork.

        Args:
            model (PolynomialNeuralNetwork): an instance of PolynomialNeuralNetwork to convert

        Returns:
            ComplexPolynomialNeuralNetwork: converted ComplexPolynomialNeuralNetwork instance
        """
        c_model = cls(
            input_dim=model.input_dim,
            output_dim=model.output_dim,
            hidden_dims=model.hidden_dims,
            degree=model.degree,
            homogeneous=model.homogeneous,
            bias=model.bias,
            s=model.s,
        )

        # Convert layers
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                c_model.layers[i] = ComplexLinear.from_linear(layer)

            # Convert activations
            for i, activation in enumerate(model.activations):
                c_model.activations[i] = (
                    ComplexPolynomialActivation.from_polynomial_activation(activation)
                )

        return c_model

    @classmethod
    def from_config(cls, config: "ModelConfig") -> "ComplexPolynomialNeuralNetwork":
        """
        Create a ComplexPolynomialNeuralNetwork from a ModelConfig.

        Args:
            config (ModelConfig): Configuration object containing model parameters.

        Returns:
            ComplexPolynomialNeuralNetwork: Instantiated model from config.
        """
        return cls(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims,
            degree=config.degree,
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

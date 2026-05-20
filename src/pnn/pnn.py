import torch
import torch.nn as nn


class PolynomialActivation(nn.Module):
    """Polynomial Activation Function Module."""

    def __init__(
        self, act_degree: int, homogeneous: bool, s: float, trainable: bool = True
    ):
        """

        Args:
            act_degree (int): Degree of the polynomial activation function.
            homogeneous (bool): Whether the activation function is homogeneous.
                If True, the activation function will be of the form f(x) = c * x^degree.
            s (float): Scaling factor for the initial coefficients.
            trainable (bool): If False, the activation is fixed as σ(z) = z^degree with no
                learnable coefficients. Overrides homogeneous. Defaults to True.
        """
        super().__init__()

        self.act_degree = act_degree
        self.homogeneous = homogeneous
        self.s = s
        self.trainable = trainable

        if act_degree <= 0:
            raise ValueError(f"act_degree must be positive, got {act_degree}")

        if not trainable:
            # Fixed σ(z) = z^degree; register as buffer so compute_augmented_matrix can read α=1
            self.register_buffer("coeffs", torch.ones(1))
        elif self.homogeneous:
            self.coeffs = nn.Parameter(torch.randn(1) * s)
        else:
            self.coeffs = nn.Parameter(torch.randn(act_degree + 1) * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.trainable:
            return torch.pow(x, self.act_degree)
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
        trainable: bool = True,
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
            trainable (bool, optional): If False, activation is fixed as σ(z) = z^degree. Defaults to True.
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
        self.trainable = trainable

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
                        act_degree=act_degree,
                        homogeneous=homogeneous,
                        s=s,
                        trainable=trainable,
                    )
                )

        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)

    def compute_augmented_matrix(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the augmented matrix of the decision boundary.

        Only valid for 2-layer networks (1 hidden layer) with quadratic activation
        and 2 output classes (binary classification).

        Returns:
            M: Augmented (n+1) x (n+1) matrix encoding the quadratic decision boundary.
            A: n x n quadratic coefficient matrix.
        """
        if len(self.layers) != 2:
            raise NotImplementedError(
                "Augmented matrix computation is only implemented for 2-layer networks"
            )
        if self.act_degree != 2:
            raise NotImplementedError(
                "Augmented matrix computation is only implemented for quadratic activation (act_degree=2)"
            )
        if self.output_dim != 2:
            raise NotImplementedError(
                "Augmented matrix computation is only implemented for binary classification (output_dim=2)"
            )
        if not self.homogeneous and self.trainable:
            raise NotImplementedError(
                "Augmented matrix computation requires homogeneous or non-trainable activation"
            )

        W1 = self.layers[0].weight.data  # (h, n)
        b1 = self.layers[0].bias.data if self.bias else torch.zeros(W1.size(0))  # (h,)
        W2 = self.layers[1].weight.data  # (2, h)
        b2 = self.layers[1].bias.data if self.bias else torch.zeros(W2.size(0))  # (2,)
        alpha = self.activations[0].coeffs[
            0
        ]  # scalar learnable coefficient in σ(z) = α·z²

        D = torch.diag(W2[0, :] - W2[1, :])  # (h, h)

        A = alpha * W1.T @ D @ W1  # (n, n)
        b = 2 * alpha * W1.T @ D @ b1  # (n,)
        c = alpha * b1 @ D @ b1 + b2[0] - b2[1]  # scalar

        M = torch.cat(
            [
                torch.cat([A, b.unsqueeze(1)], dim=1),  # (n, n+1)
                torch.cat([b.unsqueeze(0), c.reshape(1, 1)], dim=1),  # (1, n+1)
            ],
            dim=0,
        )  # (n+1, n+1)

        return M, A

    def make_det_A_constraint(self, target_det: float = 0.0):
        """Return a differentiable constraint function for use with projected SGD.

        The constraint encodes det(A) = target_det, where A is the quadratic
        coefficient matrix of the decision boundary.

        Only valid for 2-layer networks with quadratic, non-trainable activation
        and 2 output classes.

        Args:
            target_det: Target value for det(A). Defaults to 0 (degenerate boundary).

        Returns:
            constraint_fn: callable(theta_flat: Tensor) -> scalar Tensor
                           evaluating det(A(theta)) - target_det.
        """
        if len(self.layers) != 2:
            raise NotImplementedError(
                "Constraint is only implemented for 2-layer networks"
            )
        if self.act_degree != 2:
            raise NotImplementedError(
                "Constraint is only implemented for quadratic activation (act_degree=2)"
            )
        if self.output_dim != 2:
            raise NotImplementedError(
                "Constraint is only implemented for binary classification (output_dim=2)"
            )
        if self.trainable:
            raise NotImplementedError(
                "Constraint is only implemented for non-trainable activation (trainable=False)"
            )

        n, h, has_bias = self.input_dim, self.hidden_dims[0], self.bias

        # Parameter layout in theta_flat (same order as model.parameters()):
        #   layers[0].weight : (h, n)  -> h*n values
        #   layers[0].bias   : (h,)    -> h values   [only if bias]
        #   layers[1].weight : (2, h)  -> 2*h values
        #   layers[1].bias   : (2,)    -> 2 values   [only if bias]
        w1_size = h * n
        b1_size = h if has_bias else 0
        w2_size = 2 * h

        w1_end = w1_size
        w2_start = w1_end + b1_size
        w2_end = w2_start + w2_size

        def constraint_fn(theta_flat: torch.Tensor) -> torch.Tensor:
            W1 = theta_flat[:w1_end].reshape(h, n)              # (h, n)
            W2 = theta_flat[w2_start:w2_end].reshape(2, h)      # (2, h)
            d = W2[0, :] - W2[1, :]                             # (h,)
            A = W1.T @ torch.diag(d) @ W1                       # (n, n)
            return torch.linalg.det(A) - target_det

        return constraint_fn

    def make_det_M_constraint(self, target_det: float = 0.0):
        """Return a differentiable constraint function for use with projected SGD.

        The constraint encodes det(M) = target_det, where M is the full (n+1)×(n+1)
        augmented matrix encoding the quadratic decision boundary:

            M = [[A,  b ],
                 [b^T, c]]

        with A = W1^T D W1, b = 2 W1^T D b1, c = b1^T D b1 + b2[0] - b2[1],
        D = diag(W2[0,:] - W2[1,:]).

        Only valid for 2-layer networks with quadratic, non-trainable activation
        and 2 output classes.

        Args:
            target_det: Target value for det(M). Defaults to 0 (singular boundary).

        Returns:
            constraint_fn: callable(theta_flat: Tensor) -> scalar Tensor
                           evaluating det(M(theta)) - target_det.
        """
        if len(self.layers) != 2:
            raise NotImplementedError(
                "Constraint is only implemented for 2-layer networks"
            )
        if self.act_degree != 2:
            raise NotImplementedError(
                "Constraint is only implemented for quadratic activation (act_degree=2)"
            )
        if self.output_dim != 2:
            raise NotImplementedError(
                "Constraint is only implemented for binary classification (output_dim=2)"
            )
        if self.trainable:
            raise NotImplementedError(
                "Constraint is only implemented for non-trainable activation (trainable=False)"
            )

        n, h, has_bias = self.input_dim, self.hidden_dims[0], self.bias

        # Parameter layout in theta_flat (same order as model.parameters()):
        #   layers[0].weight : (h, n)  -> h*n values
        #   layers[0].bias   : (h,)    -> h values   [only if bias]
        #   layers[1].weight : (2, h)  -> 2*h values
        #   layers[1].bias   : (2,)    -> 2 values   [only if bias]
        w1_size = h * n
        b1_size = h if has_bias else 0
        w2_size = 2 * h
        b2_size = 2 if has_bias else 0

        w1_end = w1_size
        b1_end = w1_end + b1_size
        w2_end = b1_end + w2_size
        b2_end = w2_end + b2_size

        def constraint_fn(theta_flat: torch.Tensor) -> torch.Tensor:
            W1 = theta_flat[:w1_end].reshape(h, n)
            b1 = theta_flat[w1_end:b1_end] if has_bias else torch.zeros(h, device=theta_flat.device)
            W2 = theta_flat[b1_end:w2_end].reshape(2, h)
            b2 = theta_flat[w2_end:b2_end] if has_bias else torch.zeros(2, device=theta_flat.device)

            d = W2[0, :] - W2[1, :]
            D = torch.diag(d)
            A = W1.T @ D @ W1                       # (n, n)
            b = 2 * W1.T @ D @ b1                   # (n,)
            c = b1 @ D @ b1 + b2[0] - b2[1]         # scalar

            M = torch.cat(
                [
                    torch.cat([A, b.unsqueeze(1)], dim=1),
                    torch.cat([b.unsqueeze(0), c.reshape(1, 1)], dim=1),
                ],
                dim=0,
            )  # (n+1, n+1)
            return torch.linalg.det(M) - target_det

        return constraint_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        return x


if __name__ == "__main__":
    # Example usage
    model = PolynomialNeuralNetwork(
        input_dim=3,
        output_dim=2,
        hidden_dims=[5],
        act_degree=2,
        homogeneous=False,
        bias=True,
        s=1,
        trainable=False,
    )

    # Print model architecture
    print(model)

    # Test forward pass
    x = torch.randn(1, 3)
    y = model(x)
    print(y)

    M, A = model.compute_augmented_matrix()
    print("M shape:", M.shape)  # expected: (4, 4)
    print("A shape:", A.shape)  # expected: (3, 3)

    # Verify: y[0] - y[1] == x^T A x + b^T x + c
    with torch.no_grad():
        n = x.shape[-1]
        b_vec = M[:n, n]  # (n,)
        c_scalar = M[n, n]  # scalar
        x_flat = x.squeeze(0)  # (n,)
        quadratic_val = x_flat @ A @ x_flat + b_vec @ x_flat + c_scalar
        forward_diff = y[0, 0] - y[0, 1]
        print("Forward diff (y[0] - y[1]):", forward_diff.item())
        print("Quadratic form (x^T A x + b^T x + c):", quadratic_val.item())
        print("Match:", torch.isclose(forward_diff, quadratic_val))
        print("M matrix:\n", M)
        print("Determinant of M:", torch.det(M))
        print("Determinant of A:", torch.det(A))

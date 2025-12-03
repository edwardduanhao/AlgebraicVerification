import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from typing import Optional, Callable
from src.utils import c_split, c_join
from src.pnn import ComplexPolynomialNeuralNetwork
from tqdm.auto import trange


class StartSystem(nn.Module):
    """
    Start system for homotopy continuation with gamma trick.
    G(x) = gamma * (x_1^d - 1, x_2^d - 1, ..., x_n^d - 1, lambda^d - 1).
    """

    def __init__(
        self,
        d: int,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            d: Degree for each variable.
            dtype: Data type for computations (default: torch.float64).
            device: Device to place tensors on.
        """

        super().__init__()

        self.d = d

        # Randomly sample gamma with "gamma trick"
        gamma_angle = torch.rand(1, dtype=dtype, device=device) * 2 * torch.pi
        self.register_buffer("gamma_real", torch.cos(gamma_angle[0]))
        self.register_buffer("gamma_imag", torch.sin(gamma_angle[0]))

    @property
    def device(self) -> torch.device:
        return self.gamma_real.device

    @property
    def dtype(self) -> torch.dtype:
        return self.gamma_real.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Realified input tensor with shape (batch, 2n+2).

        Returns:
            Realified output tensor with shape (batch, 2n+2).
        """

        x_real, x_imag = c_split(x)

        # Convert to polar: x^d = rho^d * exp(i*d*theta)
        rho = torch.hypot(x_real, x_imag)
        theta = torch.atan2(x_imag, x_real)
        rho_pow = rho.pow(self.d)

        # Cartesian form: x^d - 1
        y_real = rho_pow * torch.cos(self.d * theta) - 1.0
        y_imag = rho_pow * torch.sin(self.d * theta)

        # Apply gamma trick: gamma * (x^d - 1)
        y_real, y_imag = (
            self.gamma_real * y_real - self.gamma_imag * y_imag,
            self.gamma_real * y_imag + self.gamma_imag * y_real,
        )

        return c_join(y_real, y_imag)


class TargetSystem(nn.Module):
    """
    Target system for polynomial constrained optimization.
    F(x, lambda) = [f(x), x - xi + lambda * grad_f(x)] = 0

    Solves for critical points of Euclidean distance to xi subject to f(x) = 0.
    """

    def __init__(
        self,
        model: nn.Module,
        xi: torch.Tensor,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            model: Neural network representing polynomial f(x)
            xi: Target point, shape (n,)
            dtype: Data type for computations (default: torch.float64)
            device: Device to place tensors on (default: inferred from xi)
        """
        super().__init__()

        # Infer device from xi if not specified
        if device is None:
            device = xi.device

        # Convert model and move to device/dtype
        self.model = ComplexPolynomialNeuralNetwork.from_polynomial_neural_network(
            model
        )
        self.model = self.model.to(device=device, dtype=dtype)

        # Setup Jacobian computation
        self.jac_model = vmap(jacrev(self.model))

        # Ensure xi is in correct dtype/device and register as buffer
        self.register_buffer("xi", xi.to(device=device, dtype=dtype))
        self.n = xi.shape[0]

    @property
    def device(self) -> torch.device:
        return self.xi.device

    @property
    def dtype(self) -> torch.dtype:
        return self.xi.dtype

    def forward(self, x_lmbda: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the target system.

        Args:
            x_lmbda: Input tensor with shape (batch, 2n + 2)

        Returns:
            Output tensor with shape (batch, 2n + 2)
        """
        n = self.n

        # Validate input dimensions
        if x_lmbda.shape[-1] != 2 * (n + 1):
            raise ValueError(
                f"Input dimension {x_lmbda.shape[-1]} != expected {2 * (n + 1)}"
            )

        # Split into x and lambda components
        x_lmbda_real, x_lmbda_imag = c_split(x_lmbda)
        x_real = x_lmbda_real[:, :n]
        x_imag = x_lmbda_imag[:, :n]
        lmbda_real = x_lmbda_real[:, -1:]
        lmbda_imag = x_lmbda_imag[:, -1:]

        # Evaluate f(x) and its Jacobian
        x = c_join(x_real, x_imag)
        y = self.model(x)
        y_real, y_imag = c_split(y)

        jac_x = self.jac_model(x)  # (batch, 2, 2n)
        # For f: C^n -> C, represented as R^{2n} -> R^2
        # jac_x[:, 0, :] = [∂(Re f)/∂x_real, ∂(Re f)/∂x_imag]
        # jac_x[:, 1, :] = [∂(Im f)/∂x_real, ∂(Im f)/∂x_imag]

        # Extract derivatives for holomorphic function gradient
        jac_x_real = jac_x[:, 0, :n]  # ∂u/∂x_real
        jac_x_imag = jac_x[:, 1, :n]  # ∂v/∂x_real

        # Compute z = x - xi + lambda * grad_f(x)
        z_real = (
            x_real
            - self.xi.unsqueeze(0)
            + lmbda_real * jac_x_real
            + lmbda_imag * jac_x_imag
        )
        z_imag = x_imag - lmbda_real * jac_x_imag + lmbda_imag * jac_x_real

        return c_join(
            torch.cat([y_real, z_real], dim=-1),
            torch.cat([y_imag, z_imag], dim=-1),
        )


class Homotopy(nn.Module):
    """
    Homotopy continuation for solving polynomial systems.
    H(x, t) = (1 - t) * G(x) + t * F(x)
    """

    def __init__(self, target_system: TargetSystem, d: int) -> None:
        """
        Args:
            target_system: Instance of TargetSystem representing F(x)
            d: Degree for start system
        """
        super().__init__()

        self.d = d
        self.n = target_system.n
        self.target_system = target_system

        # Initialize start system with same device/dtype as target system
        self.start_system = StartSystem(
            d, dtype=target_system.dtype, device=target_system.device
        )

        # Setup Jacobian computations
        self._setup_jacobians()

        self.homotopy_path = []

    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self.target_system.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype."""
        return self.target_system.dtype

    def _setup_jacobians(self):
        """Setup batched Jacobian computations."""

        def start_forward_single(x: torch.Tensor) -> torch.Tensor:
            """Single input wrapper for start system."""
            return self.start_system(x.unsqueeze(0)).squeeze(0)

        def target_forward_single(x: torch.Tensor) -> torch.Tensor:
            """Single input wrapper for target system."""
            return self.target_system(x.unsqueeze(0)).squeeze(0)

        self.jacob_start_system = vmap(jacrev(start_forward_single))
        self.jacob_target_system = vmap(jacrev(target_forward_single))

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Evaluate homotopy H(x, t).

        Args:
            x: Input tensor with shape (batch, 2n + 2)
            t: Homotopy parameter in [0, 1]

        Returns:
            Output tensor with shape (batch, 2n + 2)
        """
        g_x = self.start_system(x)
        f_x = self.target_system(x)
        return (1 - t) * g_x + t * f_x

    def _compute_jacobian(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute Jacobian dH/dx at given x and t.

        Args:
            x: Input tensor with shape (batch, 2n + 2)
            t: Homotopy parameter

        Returns:
            Jacobian tensor with shape (batch, 2n + 2, 2n + 2)
        """
        return t * self.jacob_target_system(x) + (1 - t) * self.jacob_start_system(x)

    def _get_total_roots(self) -> int:
        """Calculate total number of roots: d^(n+1)."""
        return self.d ** (self.n + 1)

    def _generate_roots_of_unity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate d-th roots of unity on the correct device/dtype."""
        k = torch.arange(self.d, dtype=self.dtype, device=self.device)
        theta = 2 * torch.pi * k / self.d
        return torch.cos(theta), torch.sin(theta)

    def _initialize_all(self) -> torch.Tensor:
        """
        Initialize all d^(n+1) roots of the start system.
        WARNING: Memory intensive for large d or n.

        Returns:
            Tensor of shape (d^(n+1), 2(n+1))
        """
        roots_real, roots_imag = self._generate_roots_of_unity()

        # Create all combinations using meshgrid
        k = torch.arange(self.d, dtype=torch.long, device=self.device)
        grids = torch.meshgrid([k] * (self.n + 1), indexing="ij")
        indices = torch.stack(grids, dim=-1).reshape(-1, self.n + 1)

        # Map indices to roots
        x_real = roots_real[indices]
        x_imag = roots_imag[indices]

        return c_join(x_real, x_imag)

    def _initialize_batch(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """
        Initialize a batch of roots from start_idx to end_idx.

        Args:
            start_idx: Starting index
            end_idx: Ending index (exclusive)

        Returns:
            Tensor of shape (batch_size, 2(n+1))
        """
        roots_real, roots_imag = self._generate_roots_of_unity()

        # Convert flat indices to multi-dimensional indices
        batch_size = end_idx - start_idx
        indices = torch.zeros(
            batch_size, self.n + 1, dtype=torch.long, device=self.device
        )

        for i in range(batch_size):
            flat_idx = start_idx + i
            for j in range(self.n + 1):
                indices[i, j] = (flat_idx // (self.d**j)) % self.d

        # Map indices to roots
        x_real = roots_real[indices]
        x_imag = roots_imag[indices]

        return c_join(x_real, x_imag)

    def initialize_batches(self, batch_size: int = 1000):
        """
        Generator yielding batches of initial solutions.

        Args:
            batch_size: Number of solutions per batch

        Yields:
            Batches of initial solutions, shape (batch_size, 2(n+1))
        """
        total_solutions = self._get_total_roots()

        for start_idx in range(0, total_solutions, batch_size):
            end_idx = min(start_idx + batch_size, total_solutions)
            yield self._initialize_batch(start_idx, end_idx)

    def tangent_predictor(self, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        Tangent predictor step for homotopy continuation.

        Args:
            x: Current point, shape (batch, 2n + 2)
            t: Current homotopy parameter
            dt: Step size

        Returns:
            Predicted next point, shape (batch, 2n + 2)
        """
        # Compute Jacobian dH/dx
        J_H_x = self._compute_jacobian(x, t)

        # Compute dH/dt = F(x) - G(x)
        dH_dt = self.target_system(x) - self.start_system(x)

        # Solve J_H_x @ (dx/dt) = -dH/dt
        # Use rcond for numerical stability when Jacobian is ill-conditioned
        dx_dt = torch.linalg.lstsq(
            J_H_x, -dH_dt.unsqueeze(-1), rcond=1e-10
        ).solution.squeeze(-1)

        return x + dx_dt * dt

    def newton_corrector(
        self, x_init: torch.Tensor, t: float, max_iters: int = 10, tol: float = 1e-6
    ) -> torch.Tensor:
        """
        Newton's method to correct predicted point onto homotopy path.
        Solves H(x, t) = 0.

        Args:
            x_init: Initial guess, shape (batch, 2n + 2)
            t: Current homotopy parameter
            max_iters: Maximum Newton iterations
            tol: Convergence tolerance

        Returns:
            Corrected point, shape (batch, 2n + 2)
        """
        x = x_init.clone()

        for _ in range(max_iters):
            # Evaluate H(x, t)
            H_x = self.forward(x, t)

            # Compute Jacobian dH/dx
            J_H_x = self._compute_jacobian(x, t)

            # Solve J_H_x @ delta_x = -H(x, t)
            # Try direct solve first, fall back to regularized lstsq if singular
            try:
                delta_x = torch.linalg.solve(J_H_x, -H_x.unsqueeze(-1)).squeeze(-1)
            except torch.linalg.LinAlgError:
                # Fall back to least squares with regularization
                delta_x = torch.linalg.lstsq(
                    J_H_x, -H_x.unsqueeze(-1), rcond=1e-10
                ).solution.squeeze(-1)

            # Newton update
            x = x + delta_x

            # Check convergence
            if torch.all(torch.norm(delta_x, dim=1) < tol):
                break

        return x

    def track(
        self,
        x_init: torch.Tensor,
        num_steps: int = 100,
        max_newton_iters: int = 10,
        newton_tol: float = 1e-6,
        save_history: bool = True,
    ) -> tuple[torch.Tensor, list]:
        """
        Track a homotopy path from t=0 to t=1 using predictor-corrector method.

        Args:
            x_init: Initial point at t=0, shape (batch, 2n + 2)
            num_steps: Number of steps along the path
            max_newton_iters: Maximum Newton iterations for corrector
            newton_tol: Tolerance for Newton convergence
            save_history: Whether to save the path history

        Returns:
            final_x: Solution at t=1, shape (batch, 2n + 2)
            history: List of (t, x) tuples if save_history=True, else empty list
        """
        t = 0.0
        x = x_init.clone()
        dt = 1.0 / num_steps
        history = [] if save_history else None
        errors = []

        if save_history:
            history.append((t, x.clone()))

        for step in trange(num_steps):
            # Predictor step
            x_pred = self.tangent_predictor(x, t, dt)

            # Corrector step (Newton's method)
            t_next = t + dt
            x_corr = self.newton_corrector(
                x_pred, t_next, max_iters=max_newton_iters, tol=newton_tol
            )

            # Update state
            x = x_corr
            t = t_next

            errors.append(torch.norm(self.forward(x, t), dim=1).cpu().detach().numpy())

            if save_history:
                history.append((t, x.clone()))

        return x, history if save_history else [], errors

    def solve(
        self,
        batch_size: int = 100,
        num_steps: int = 100,
        max_newton_iters: int = 10,
        newton_tol: float = 1e-6,
        save_history: bool = False,
    ) -> tuple[torch.Tensor, list]:
        """
        Solve the polynomial system by tracking all homotopy paths in batches.

        Args:
            batch_size: Number of paths to track simultaneously
            num_steps: Number of steps along each path
            max_newton_iters: Maximum Newton iterations for corrector
            newton_tol: Tolerance for Newton convergence
            save_history: Whether to save path histories

        Returns:
            solutions: All solutions at t=1, shape (num_solutions, 2n+2)
            histories: List of histories for each batch if save_history=True
            errors: List of error norms for each batch
        """
        all_solutions = []
        all_histories = [] if save_history else None
        all_errors = []

        for x_init_batch in self.initialize_batches(batch_size):
            # Track this batch of paths
            x_final, history, errors = self.track(
                x_init_batch,
                num_steps=num_steps,
                max_newton_iters=max_newton_iters,
                newton_tol=newton_tol,
                save_history=save_history,
            )

            all_solutions.append(x_final)
            if save_history:
                all_histories.append(history)
            all_errors.append(errors)

        # Concatenate all solutions
        solutions = torch.cat(all_solutions, dim=0)

        return solutions, all_histories if save_history else [], all_errors

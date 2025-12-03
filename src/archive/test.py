import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.func import jacrev, jacfwd, vmap
from src.pnn import (
    PolynomialNeuralNetwork,
    ComplexPolynomialNeuralNetwork,
)
from src.utils import c_join, c_split
from src.config import Config
from src.data import generate_sinusoidal_data, visualize_decision_boundary
from src.utils import train_epochs
from src.torchhc import StartSystem, TargetSystem, Homotopy


if __name__ == "__main__":
    # Create a sample PolynomialNeuralNetwork
    pnn = PolynomialNeuralNetwork(
        input_dim=2,
        output_dim=1,
        hidden_dims=[7, 6],
        degree=2,
        homogeneous=False,
        bias=True,
        s=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    pnn = pnn.to(device)
    # pnn = pnn.double().to(device)  # Use double precision

    xi = torch.tensor([1.0, 2.0])
    target_system = TargetSystem(pnn, xi)

    homotopy = Homotopy(target_system, d=4)

    # ==================================================================
    # TEST 1: Verify initial solutions are roots of start system
    # ==================================================================
    print("=" * 60)
    print("TEST 1: Initial Solutions (Roots of Unity)")
    print("=" * 60)

    x_init = list(homotopy.initialize_batches(batch_size=10))[0]
    print(f"Initial points shape: {x_init.shape}")

    g_x = homotopy.start_system(x_init)
    start_residuals = torch.norm(g_x, dim=1)
    print(f"\n||G(x_init)|| residuals:")
    print(f"  Max: {start_residuals.max().item():.2e}")
    print(f"  Mean: {start_residuals.mean().item():.2e}")
    print(f"  All < 1e-10: {(start_residuals < 1e-10).all().item()}")

    # ==================================================================
    # TEST 2: Check Jacobian conditioning
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Jacobian Conditioning")
    print("=" * 60)

    t = 0.0
    J_H_start = homotopy._compute_jacobian(x_init, t=0.0)
    J_H_mid = homotopy._compute_jacobian(x_init, t=0.5)
    J_H_end = homotopy._compute_jacobian(x_init, t=1.0)

    cond_start = torch.linalg.cond(J_H_start)
    cond_mid = torch.linalg.cond(J_H_mid)
    cond_end = torch.linalg.cond(J_H_end)

    print(f"\nCondition numbers at t=0.0:")
    print(f"  Range: [{cond_start.min().item():.2e}, {cond_start.max().item():.2e}]")
    print(f"\nCondition numbers at t=0.5:")
    print(f"  Range: [{cond_mid.min().item():.2e}, {cond_mid.max().item():.2e}]")
    print(f"\nCondition numbers at t=1.0:")
    print(f"  Range: [{cond_end.min().item():.2e}, {cond_end.max().item():.2e}]")

    # ==================================================================
    # TEST 3: Compare predictor with/without regularization
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Predictor Step Size Comparison")
    print("=" * 60)

    dt = 1.0 / 1000
    dH_dt = homotopy.target_system(x_init) - homotopy.start_system(x_init)

    # Without regularization
    dx_dt_default = torch.linalg.lstsq(
        J_H_start, -dH_dt.unsqueeze(-1)
    ).solution.squeeze(-1)
    step_default = torch.norm(dx_dt_default * dt, dim=1)

    # With regularization (what we're using now)
    dx_dt_reg = torch.linalg.lstsq(
        J_H_start, -dH_dt.unsqueeze(-1), rcond=1e-10
    ).solution.squeeze(-1)
    step_reg = torch.norm(dx_dt_reg * dt, dim=1)

    print(f"\nPredictor step sizes (dt={dt}):")
    print(
        f"  Without regularization: max={step_default.max().item():.2e}, mean={step_default.mean().item():.2e}"
    )
    print(
        f"  With rcond=1e-10:       max={step_reg.max().item():.2e}, mean={step_reg.mean().item():.2e}"
    )

    # ==================================================================
    # TEST 4: Track a few paths with diagnostics
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Path Tracking Quality")
    print("=" * 60)

    x_test = x_init[:3].clone()  # Track 3 paths
    print(f"\nTracking {x_test.shape[0]} paths with num_steps=100...")

    x_final, history, errors = homotopy.track(
        x_test,
        num_steps=100,
        max_newton_iters=10,
        newton_tol=1e-10,
        save_history=True,
    )

    # Analyze errors along the path
    import numpy as np

    errors_array = np.array(errors)  # shape: (num_steps, num_paths)

    print(f"\nPath tracking errors ||H(x,t)||:")
    print(f"  Initial (t=0): max={errors_array[0].max():.2e}")
    print(f"  Final (t=1):   max={errors_array[-1].max():.2e}")
    print(f"  Max along path: {errors_array.max():.2e}")
    print(f"  Mean along path: {errors_array.mean():.2e}")

    # Check final residual
    final_residual = torch.norm(homotopy.target_system(x_final), dim=1)
    print(f"\nFinal ||F(x)|| (should be small):")
    print(f"  Max: {final_residual.max().item():.2e}")
    print(f"  Mean: {final_residual.mean().item():.2e}")

    # ==================================================================
    # TEST 5: Full solve with all paths
    # ==================================================================
    print("\n" + "=" * 60)
    print("TEST 5: Full Homotopy Continuation Solve")
    print("=" * 60)
    print(f"Total roots to track: {homotopy._get_total_roots()}")
    print(f"Solving with batch_size=100, num_steps=1000...")

    sol, history_all, errors_all = homotopy.solve(
        batch_size=100,
        num_steps=1000,
        max_newton_iters=10,
        newton_tol=1e-10,
        save_history=False,
    )

    print(f"\nSolutions found: {sol.shape[0]}")

    # Analyze final errors for all paths
    final_errors_all = []
    for batch_errors in errors_all:
        # Each batch_errors is a list of arrays for each step
        # Get the last step's errors
        final_errors_all.append(batch_errors[-1])

    final_errors_concat = np.concatenate(final_errors_all)

    print(f"\nFinal residuals ||H(x,1)|| for all paths:")
    print(f"  Max: {final_errors_concat.max():.2e}")
    print(f"  Mean: {final_errors_concat.mean():.2e}")
    print(f"  Median: {np.median(final_errors_concat):.2e}")
    print(
        f"  Paths with error < 1e-6: {(final_errors_concat < 1e-6).sum()}/{len(final_errors_concat)}"
    )
    print(
        f"  Paths with error < 1e-8: {(final_errors_concat < 1e-8).sum()}/{len(final_errors_concat)}"
    )
    print(
        f"  Paths with error < 1e-10: {(final_errors_concat < 1e-10).sum()}/{len(final_errors_concat)}"
    )

    # Verify solutions satisfy target system
    target_residuals = torch.norm(homotopy.target_system(sol), dim=1)
    print(f"\nFinal ||F(x)|| for all solutions:")
    print(f"  Max: {target_residuals.max().item():.2e}")
    print(f"  Mean: {target_residuals.mean().item():.2e}")

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)

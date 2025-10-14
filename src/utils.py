import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from decimal import Decimal


def generate_sin_boundary_data(
    n_samples=1000, noise_level=0.1, x_range=(-2 * np.pi, 2 * np.pi), seed=42
):
    """
    Generate binary classification data with sin(x) decision boundary.

    Args:
        n_samples: Number of data points to generate
        noise_level: Amount of noise to add to the boundary
        x_range: Range of x values to sample from

    Returns:
        X: Input features of shape (n_samples, 2)
        y: Binary labels of shape (n_samples,)
    """

    np.random.seed(seed)

    # Generate random x values
    x = np.random.uniform(x_range[0], x_range[1], n_samples)

    # Generate random y values around the sin(x) boundary
    y_boundary = np.sin(x)

    # Add some spread around the boundary
    y_spread = 2.0  # How far above/below the boundary to sample
    y = np.random.uniform(y_boundary - y_spread, y_boundary + y_spread, n_samples)

    # Add noise to the boundary itself
    noisy_boundary = y_boundary + np.random.normal(0, noise_level, n_samples)

    # Create binary labels: 1 if above noisy boundary, 0 if below
    labels = (y > noisy_boundary).astype(np.int64)

    # Stack features
    X = np.stack([x, y], axis=1).astype(np.float32)

    # Separate the data points by class
    mask_class0 = labels == 0
    mask_class1 = labels == 1

    X_class0 = X[mask_class0]
    X_class1 = X[mask_class1]

    fig = go.Figure()

    # Add data points for class 0
    fig.add_trace(
        go.Scatter(
            x=X_class0[:, 0],
            y=X_class0[:, 1],
            mode="markers",
            marker=dict(color="blue", size=5, opacity=0.7),
            name="Class 0",
        )
    )

    # Add data points for class 1
    fig.add_trace(
        go.Scatter(
            x=X_class1[:, 0],
            y=X_class1[:, 1],
            mode="markers",
            marker=dict(color="red", size=5, opacity=0.7),
            name="Class 1",
        )
    )

    # Add true decision boundary
    x_true = np.linspace(x_range[0], x_range[1], 1000)
    y_true = np.sin(x_true)
    fig.add_trace(
        go.Scatter(
            x=x_true,
            y=y_true,
            mode="lines",
            line=dict(color="green", width=3),
            name="True sin(x) Boundary",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="x0",
        yaxis_title="x1",
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        width=800,
        height=600,
    )
    fig.show()
    # fig.write_image("../figures/training_data.pdf", width=800, height=600, scale=2)

    return torch.tensor(X), torch.tensor(labels)


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

    x_grid = np.linspace(x_range[0], x_range[1], 200)
    y_grid = np.linspace(y_range[0], y_range[1], 200)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Flatten grid for prediction
    grid_points = torch.tensor(
        np.c_[X_grid.ravel(), Y_grid.ravel()], dtype=torch.float32
    )

    # Get model predictions
    model.eval()
    with torch.no_grad():
        logits = model(grid_points)
        predictions = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1

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

    fig.show()
    # fig.write_image("../figures/decision_boundary.pdf", width=800, height=800, scale=2)


def coeffs_to_phcpy_string(
    coefficients, monomial_degrees, variable_names=None, tolerance=1e-12
):
    """
    Convert polynomial coefficients and monomial degrees to PHCpy string format.

    Args:
        coefficients: 1D numpy array of polynomial coefficients
        monomial_degrees: list of tuples, where each tuple represents the degrees
                         of variables in the corresponding monomial
                         e.g., [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)] for 2D
        variable_names: list of variable name strings, defaults to ['x0', 'x1', ...]
        tolerance: minimum absolute value for coefficients to be included

    Returns:
        str: polynomial string in PHCpy format (without trailing semicolon)

    Examples:
        >>> coeffs = np.array([1.0, -2.5, 3.0])
        >>> monoms = [(0, 0), (1, 0), (0, 1)]  # constant, x0, x1
        >>> coeffs_to_phcpy_string(coeffs, monoms)
        '1.0 - 2.5*x0 + 3.0*x1'

        >>> coeffs = np.array([1.0, 0.0, -1.0, 2.0])
        >>> monoms = [(0, 0), (1, 0), (2, 0), (1, 1)]  # 1, x0, x0^2, x0*x1
        >>> coeffs_to_phcpy_string(coeffs, monoms)
        '1.0 - x0**2 + 2.0*x0*x1'
    """

    if len(coefficients) != len(monomial_degrees):
        raise ValueError("Length of coefficients must match length of monomial_degrees")

    # Determine number of variables from monomial degrees
    if not monomial_degrees:
        return "0"

    num_vars = len(monomial_degrees[0])

    # Set default variable names if not provided
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(num_vars)]
        print(variable_names)
    elif len(variable_names) != num_vars:
        raise ValueError(
            f"Expected {num_vars} variable names, got {len(variable_names)}"
        )

    terms = []

    for coeff, degrees in zip(coefficients, monomial_degrees):
        coeff_val = Decimal(coeff)

        # Skip negligible coefficients
        if abs(coeff_val) < tolerance:
            continue

        # Build the monomial string
        monomial_parts = []
        for var_name, degree in zip(variable_names, degrees):
            if degree > 0:
                if degree == 1:
                    monomial_parts.append(var_name)
                else:
                    monomial_parts.append(f"{var_name}**{degree}")

        # Combine monomial parts
        if monomial_parts:
            monomial_str = "*".join(monomial_parts)
        else:
            monomial_str = ""  # Constant term

        # Build the term string with coefficient
        if monomial_str:  # Non-constant term
            if abs(coeff_val - Decimal("1.0")) < tolerance:  # Coefficient is +1
                term_str = monomial_str
            elif abs(coeff_val + Decimal("1.0")) < tolerance:  # Coefficient is -1
                term_str = f"-{monomial_str}"
            else:  # General coefficient
                if coeff_val > 0:
                    term_str = f"{coeff_val:g}*{monomial_str}"
                else:
                    term_str = f"{coeff_val:g}*{monomial_str}"
        else:  # Constant term
            term_str = f"{coeff_val:g}"

        terms.append(term_str)

    if not terms:
        return "0"

    # Join terms with appropriate signs
    result = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            result += f" - {term[1:]}"  # Remove the negative sign and add " - "
        else:
            result += f" + {term}"

    return result


def polynomial_system_to_phcpy(
    coefficient_matrix, monomial_degrees, variable_names=None, tolerance=1e-12
):
    """
    Convert a system of polynomials (coefficient matrix) to PHCpy string format.

    Args:
        coefficient_matrix: 2D numpy array of shape (num_equations, num_monomials)
                           Each row represents coefficients for one polynomial equation
        monomial_degrees: list of tuples representing monomial degrees
        variable_names: list of variable name strings
        tolerance: minimum absolute value for coefficients to be included

    Returns:
        list: list of polynomial strings in PHCpy format (with trailing semicolons)

    Example:
        >>> coeffs = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        >>> monoms = [(0, 0), (1, 0), (0, 1)]  # constant, x0, x1
        >>> polynomial_system_to_phcpy(coeffs, monoms)
        ['1.0 - x0;', 'x0 - x1;']
    """
    if coefficient_matrix.ndim != 2:
        raise ValueError("coefficient_matrix must be 2D")

    num_equations, num_monomials = coefficient_matrix.shape

    if len(monomial_degrees) != num_monomials:
        raise ValueError("Number of monomials must match coefficient matrix columns")

    phcpy_equations = []

    for i in range(num_equations):
        poly_str = coeffs_to_phcpy_string(
            coefficient_matrix[i], monomial_degrees, variable_names, tolerance
        )
        phcpy_equations.append(poly_str + ";")

    return phcpy_equations


def test_phcpy_formatting():
    """Test function for PHCpy formatting utilities."""
    print("Testing PHCpy formatting functions...")

    # Test 1: Simple polynomial
    coeffs1 = np.array([1.0, -2.5, 3.0, 0.0, -1.0])
    monoms1 = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]  # 1, x0, x1, x0^2, x0*x1
    result1 = coeffs_to_phcpy_string(coeffs1, monoms1)
    # result1 = coeffs_to_phcpy_string(coeffs1, monoms1, variable_names=["x", "y"])
    print(f"Test 1: {result1}")

    # Test 2: System of equations
    coeffs2 = np.array(
        [
            [1.0, -1.0, 0.0],  # 1 - x0
            [0.0, 1.0, -1.0],  # x0 - x1
            [4.0, 0.0, 1.0],  # 4 + x1
        ]
    )
    monoms2 = [(0, 0), (1, 0), (0, 1)]  # constant, x0, x1
    result2 = polynomial_system_to_phcpy(coeffs2, monoms2)
    print(f"Test 2 (system): {result2}")

    # Test 3: Higher degree polynomial
    coeffs3 = np.array([2.0, 0.0, -1.0, 0.5, 0.0, 1.0])
    monoms3 = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
    result3 = coeffs_to_phcpy_string(coeffs3, monoms3)
    print(f"Test 3: {result3}")

    print("PHCpy formatting tests completed!")

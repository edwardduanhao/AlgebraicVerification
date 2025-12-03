import torch
import sympy as sp
import numpy as np
from . import model
import time


def _to_sympy_matrix(A):
    """Convert torch.Tensor to a SymPy Matrix."""
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    return sp.Matrix(A)


def _sigma_elementwise(x, c):
    """Element-wise polynomial activation function."""
    if isinstance(c, torch.Tensor):
        c = c.detach().cpu().numpy()

    def sigma_scalar(t):
        return sum(c[d] * (t**d) for d in range(len(c)))

    return x.applyfunc(sigma_scalar)


def polynomial_nn_expansion(model):
    """Convert a PyTorch neural network with polynomial activations into polynomials."""
    start_time = time.perf_counter()
    input_dim = model.input_dim
    x_syms = sp.symbols(f"x0:{input_dim}")
    x_vec = sp.Matrix(x_syms)
    print(f"Input symbols: {x_syms}")

    h = x_vec

    # Forward pass through layers
    for i, layer in enumerate(model.layers):
        W = _to_sympy_matrix(layer.weight)
        b = _to_sympy_matrix(layer.bias)
        c = model.activations[i].coefficients if i < len(model.activations) else None
        h = W * h + b
        if c is not None:
            h = _sigma_elementwise(h, c)

        print(f"Layer {i} weight: {W.shape}")
        print(f"Layer {i} bias: {b.shape}")
        print(f"Layer {i} activation: {tuple(c.shape) if c is not None else None}")

    y = h

    y_list = [sp.expand(expr) for expr in list(y)]

    polys = [sp.Poly(expr, *x_syms) for expr in y_list]

    all_monoms = set().union(*[set(P.monoms()) for P in polys])

    monoms = sorted(all_monoms, key=lambda a: (sum(a),) + tuple(a))

    q = len(polys)
    C = np.zeros((q, len(monoms)), dtype=float)
    for j, P in enumerate(polys):
        term_dict = dict(P.terms())
        for k, a in enumerate(monoms):
            C[j, k] = term_dict.get(a, 0)

    end_time = time.perf_counter()
    print(f"Polynomial expansion took {end_time - start_time:.4f} seconds.")
    return y_list, monoms, C


def polynomial_nn_expansion_symbolic(model, verbose=True):
    """Convert a PyTorch neural network with polynomial activations into polynomials."""
    start_time = time.perf_counter()
    input_dim = model.input_dim
    x_syms = sp.symbols(f"x0:{input_dim}")
    x = sp.Matrix(x_syms)
    if verbose:
        print(f"Input symbols: {x_syms}")

    W_syms = []
    b_syms = []
    c_syms = []

    # Forward pass through layers
    for l, layer in enumerate(model.layers):
        input_dim = layer.in_features
        output_dim = layer.out_features

        # Weights
        W = sp.Matrix(
            [
                [sp.symbols(f"W{l}_{i}_{j}") for j in range(input_dim)]
                for i in range(output_dim)
            ]
        )
        W_syms.append(W)
        if verbose:
            print(f"Layer {l} weight: {W.shape}")

        # Biases
        b = sp.Matrix([sp.symbols(f"b{l}_{i}") for i in range(output_dim)])
        b_syms.append(b)
        if verbose:
            print(f"Layer {l} bias: {b.shape}")

        # Activation coefficients
        if l < len(model.layers) - 1:
            c = sp.Matrix(
                [sp.symbols(f"c{l}_{i}") for i in range(model.polynomial_degree + 1)]
            )
            c_syms.append(c)
            if verbose:
                print(f"Layer {l} activation: {tuple(c.shape)}")

    h = x

    for l, (W, b) in enumerate(zip(W_syms, b_syms)):
        h = W * h + b
        if l < len(model.layers) - 1:
            c = c_syms[l]
            h = h.applyfunc(lambda z: sum(c[d] * z**d for d in range(len(c))))

    y = h
    y_expr = [sp.expand(expr) for expr in y]
    polys = [sp.Poly(expr, *x_syms) for expr in y_expr]
    all_monoms = set().union(*[set(P.monoms()) for P in polys])

    monoms = sorted(all_monoms, key=lambda a: (sum(a),) + tuple(a))

    coeffs = [[] for _ in range(len(polys))]

    for j, P in enumerate(polys):
        term_dict = dict(P.terms())
        for a in monoms:
            coeffs[j].append(term_dict.get(a, 0))

    param_syms = []
    for l, W in enumerate(W_syms):
        W.rows, W.cols = W.shape
        for i in range(W.rows):
            for j in range(W.cols):
                param_syms.append(W[i, j])
    for l, b in enumerate(b_syms):
        b.rows, _ = b.shape
        for i in range(b.rows):
            param_syms.append(b[i, 0])
    for l, c in enumerate(c_syms):
        c.rows, _ = c.shape
        for i in range(c.rows):
            param_syms.append(c[i, 0])

    end_time = time.perf_counter()
    if verbose:
        print(f"Polynomial expansion took {end_time - start_time:.4f} seconds.")

    params = []

    for layer in model.layers:
        params += torch.split(layer.weight.detach().cpu().flatten(), 1)

    for layer in model.layers:
        params += torch.split(layer.bias.detach().cpu(), 1)

    for c in model.activations:
        params += torch.split(c.coefficients.detach().cpu(), 1)

    coeff_func = sp.lambdify(param_syms, coeffs, modules="torch")
    coeff_vals = coeff_func(*params)

    C = [elem for row in coeff_vals for elem in row]
    C = torch.stack(C).reshape(len(coeff_vals), -1)

    return monoms, C


def verify_expansion(model, monoms, C, test_inputs=None, n_test=100, tolerance=1e-6):
    """Verify that polynomial expansion matches neural network output."""
    if test_inputs is None:
        # Generate random test inputs
        test_inputs = torch.randn(n_test, model.input_dim)
    else:
        n_test = test_inputs.shape[0]

    # Get neural network outputs
    model.eval()
    with torch.no_grad():
        nn_outputs = model(test_inputs).numpy()

    # Evaluate polynomial outputs
    x_syms = sp.symbols(f"x0:{model.input_dim}")
    poly_outputs = []

    for input_vec in test_inputs:
        input_dict = {x_syms[i]: float(input_vec[i]) for i in range(len(x_syms))}
        output_vec = []

        for j in range(C.shape[0]):  # For each output dimension
            poly_value = 0
            for k, monom in enumerate(monoms):
                if C[j, k] != 0:  # Skip zero coefficients
                    # Evaluate monomial: x0^a0 * x1^a1 * ...
                    monom_value = 1
                    for i, power in enumerate(monom):
                        if power > 0:
                            monom_value *= input_dict[x_syms[i]] ** power
                    poly_value += float(C[j, k]) * monom_value
            output_vec.append(poly_value)
        poly_outputs.append(output_vec)

    poly_outputs = np.array(poly_outputs)

    # Compare outputs
    max_diff = np.max(np.abs(nn_outputs - poly_outputs))
    mean_diff = np.mean(np.abs(nn_outputs - poly_outputs))

    print(f"Verification Results:")
    print(f"Max absolute difference: {max_diff:.2e} over {n_test} samples")
    print(f"Mean absolute difference: {mean_diff:.2e} over {n_test} samples")
    print(f"Within tolerance ({tolerance}): {max_diff < tolerance}")

    if max_diff >= tolerance:
        print("Warning: Large differences detected!")
        print(f"NN output sample: {nn_outputs[0]}")
        print(f"Polynomial output sample: {poly_outputs[0]}")

    return max_diff < tolerance


def compute_class_differences(C, gold_class):
    """
    Compute coefficient differences C[i,:] - C[j,:] for all j != i.

    Args:
        C: Coefficient matrix of shape (K, num_monomials) where K is number of classes
        gold_class: Index of the gold/true class (0-indexed)

    Returns:
        differences: Array of shape (K-1, num_monomials) containing C[i,:] - C[j,:]
                    for all j != i, where i is the gold_class
    """
    K, num_monomials = C.shape

    if gold_class < 0 or gold_class >= K:
        raise ValueError(f"gold_class must be between 0 and {K-1}, got {gold_class}")

    # Get coefficients for the gold class
    gold_coeffs = C[gold_class, :]

    # Compute differences for all other classes
    differences = []
    for j in range(K):
        if j != gold_class:
            diff = gold_coeffs - C[j, :]
            differences.append(diff)

    return np.array(differences)


if __name__ == "__main__":
    model = model.PolynomialNetwork(
        input_dim=2, output_dim=2, hidden_dims=[3, 4], polynomial_degree=2
    )

    monoms, C = polynomial_nn_expansion(model)
    print("Number of monomials: ", len(monoms))
    print("Shape of Coefficient matrix:", C.shape)

    # Verify the expansion is correct
    print("\n" + "=" * 50)
    is_correct = verify_expansion(model, monoms, C)
    print(f"Expansion verification: {'PASSED' if is_correct else 'FAILED'}")

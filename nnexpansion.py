import torch
import sympy as sp
import numpy as np
import model
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
    x_syms = sp.symbols(f'x0:{input_dim}')
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
    
    polys   = [sp.Poly(expr, *x_syms) for expr in y_list]
    
    all_monoms = set().union(*[set(P.monoms()) for P in polys])
    
    monoms = sorted(all_monoms, key=lambda a: (sum(a),) + tuple(a))
    
    q = len(polys)
    C = np.zeros((q, len(monoms)), dtype=object)
    for j, P in enumerate(polys):
        term_dict = dict(P.terms())
        for k, a in enumerate(monoms):
            C[j, k] = term_dict.get(a, 0)

    end_time = time.perf_counter()
    print(f"Polynomial expansion took {end_time - start_time:.4f} seconds.")
    return monoms, C


def verify_expansion(model, monoms, C, test_inputs=None, tolerance=1e-6):
    """Verify that polynomial expansion matches neural network output."""
    if test_inputs is None:
        # Generate random test inputs
        test_inputs = torch.randn(10, model.input_dim)
    
    # Get neural network outputs
    model.eval()
    with torch.no_grad():
        nn_outputs = model(test_inputs).numpy()
    
    # Evaluate polynomial outputs
    x_syms = sp.symbols(f'x0:{model.input_dim}')
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
                            monom_value *= (input_dict[x_syms[i]] ** power)
                    poly_value += float(C[j, k]) * monom_value
            output_vec.append(poly_value)
        poly_outputs.append(output_vec)
    
    poly_outputs = np.array(poly_outputs)
    
    # Compare outputs
    max_diff = np.max(np.abs(nn_outputs - poly_outputs))
    mean_diff = np.mean(np.abs(nn_outputs - poly_outputs))
    
    print(f"Verification Results:")
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
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
    print("\n" + "="*50)
    is_correct = verify_expansion(model, monoms, C)
    print(f"Expansion verification: {'PASSED' if is_correct else 'FAILED'}")
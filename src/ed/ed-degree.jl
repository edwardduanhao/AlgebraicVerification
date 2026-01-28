# =============================================================================
# ED Degree Computation for Polynomial Neural Network Decision Boundaries
#
# Two approaches:
#   1. Symbolic (Oscar)  — exact ED degree via Gröbner basis / saturation
#   2. Numerical (HomotopyContinuation.jl) — ED degree via homotopy continuation
# =============================================================================

# =============================================================================
# Part 1: Symbolic Computation (Algorithm 1)
#
# Computes the ED degree by constructing the critical ideal of the squared
# distance function restricted to the decision boundary, saturating out the
# singular locus, and returning the degree of the resulting zero-dimensional
# ideal.
#
# Dependencies: Oscar
# =============================================================================

using Oscar

function EDdegree(W1, W2, b1, b2)
    # 1. Setup Ring
    R, (x1, x2) = polynomial_ring(QQ, ["x1", "x2"])

    # 2. Define Network & Decision Boundary F
    mW1 = matrix(R, 2, 2, W1)
    mW2 = matrix(R, 2, 2, W2)
    mb1 = matrix(R, 2, 1, b1)
    mb2 = matrix(R, 2, 1, b2)

    # Input vector x (2x1)
    x = matrix(R, 2, 1, [x1, x2])

    # Calculate z = W1*x + b1
    z = mW1 * x + mb1

    # Square activation
    h = matrix(R, 2, 1, [z[1, 1]^2, z[2, 1]^2])

    # Output layer
    f = mW2 * h + mb2

    # Decision Boundary
    F = f[1, 1] - f[2, 1]

    # 3. ED Degree Setup
    u = [rand(-100:100), rand(-100:100)]

    # 4. Critical Point Matrix
    dFdx1 = derivative(F, x1)
    dFdx2 = derivative(F, x2)

    M_arr = [
        (x1-u[1]) dFdx1;
        (x2-u[2]) dFdx2
    ]
    M = matrix(R, 2, 2, M_arr)

    # 5. Compute Ideal
    crit_ideal = ideal(R, [det(M), F])

    # 6. Saturate
    grads_ideal = ideal(R, [dFdx1, dFdx2])
    Sat = saturation(crit_ideal, grads_ideal)

    return Oscar.degree(Sat)
end

# =============================================================================
# Part 2: Numerical Computation
#
# Computes the ED degree by solving the Lagrange multiplier system
#       x - u - λ ∇g(x) = 0
#       g(x) = 0
# via numerical homotopy continuation. The number of nonsingular isolated
# solutions equals the ED degree for a generic data point u.
#
# Dependencies: HomotopyContinuation.jl
# =============================================================================

using HomotopyContinuation, Random

# -----------------------------------------------------------------------------
# Decision Boundary Constructors
#
# Build the polynomial g(x) = f_c(x) - f_c'(x) for a PNN with architecture
# (n, [h1, ..., hs], k) and degree-d activation.
#
# Three variants handle different numerical regimes:
#   - Complex weights:  guarantee genericity (no accidental symmetries)
#   - Real weights:     closer to practical models, may lie on special loci
#   - Scaled complex:   prevent coefficient explosion in deep networks
# -----------------------------------------------------------------------------


function nn_poly_complex(n, layers, d; scale=0.5)
    @var x[1:n]
    a = x
    for h in layers
        W = randn(ComplexF64, h, length(a)) .* scale
        b = randn(ComplexF64, h) .* scale
        a = (W * a + b) .^ d
    end
    f = (randn(ComplexF64, 2, length(a)) .* scale) * a + (randn(ComplexF64, 2) .* scale)
    return f[1] - f[2], x
end


function EDdegree_numerical(n, layers, d; constructor=nn_poly_complex)
    g, x = constructor(n, layers, d)

    @var λ u[1:length(x)]
    ∇g = differentiate(g, x)

    sys = System([x - u - λ * ∇g; g], [x; λ], u)

    u_target = randn(length(x))
    result = HomotopyContinuation.solve(sys; target_parameters=u_target)

    return nsolutions(result)
end


# =============================================================================
# Examples
# =============================================================================


W1 = [1 2; 3 1]
W2 = [2 1; 1 2]
b1 = [0; 1]
b2 = [2; 1]


# --- Example 1 ---
println("--- Example 1 ---")
println("W1 = $W1", "\nW2 = $W2", "\nb1 = $b1", "\nb2 = $b2")
println("ED degree through symbolic computation:  ", EDdegree(W1, W2, b1, b2))


# --- Example 2 ---
c1 = [1; 2]
c2 = [1; 1]
println("\n--- Example 2 ---")
println("W1 = $W1", "\nW2 = $W2", "\nb1 = $c1", "\nb2 = $c2")
println("ED degree through symbolic computation:  ", EDdegree(W1, W2, c1, c2))


# Numerical ED degree computations for various architectures
# Table 1 in the paper
println("\n--- Numerical ED Degree Computations ---")
println("ED degree of architecture (3, [3,2], 2) is : ", EDdegree_numerical(3, [3, 2], 2))
println("ED degree of architecture (3, [3,2], 3) is : ", EDdegree_numerical(3, [3, 2], 3))
println("ED degree of architecture (3, [2,3], 3) is : ", EDdegree_numerical(3, [2, 3], 3))
println("ED degree of architecture (4, [4,3], 2) is : ", EDdegree_numerical(4, [4, 3], 2))
println("ED degree of architecture (4, [4,2], 2) is : ", EDdegree_numerical(4, [4, 2], 2))
println("ED degree of architecture (5, [5,3], 2) is : ", EDdegree_numerical(5, [5, 3], 2))
println("ED degree of architecture (3, [3,2,2], 2) is : ", EDdegree_numerical(3, [3, 2, 2], 2))
println("ED degree of architecture (3, [3,3,2], 2) is : ", EDdegree_numerical(3, [3, 3, 2], 2))
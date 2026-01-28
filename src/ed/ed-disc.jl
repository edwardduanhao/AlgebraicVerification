# =============================================================================
# ED Discriminant Computation for Polynomial Neural Network Decision Boundaries
#
# Computes the ED discriminant by constructing the critical locus of the
# squared distance function (with Lagrange multiplier), appending the
# determinant of the Jacobian (to detect non-smooth fibers), saturating
# out degenerate components, and eliminating the primal variables to
# obtain a polynomial in the data coordinates (u1, u2).
#
# Dependencies: Oscar
# =============================================================================

using Oscar

function EDdisc(W1, W2, b1, b2)
    # 1. Setup Ring
    R, vars = polynomial_ring(QQ, ["x1", "x2", "L", "u1", "u2"])
    x1, x2, L, u1, u2 = vars

    # Helpers
    x_vec = matrix(R, 2, 1, [x1, x2])
    mW1 = matrix(R, 2, 2, W1)
    mW2 = matrix(R, 2, 2, W2)
    mb1 = matrix(R, 2, 1, b1)
    mb2 = matrix(R, 2, 1, b2)

    # 2. Define Decision Boundary F
    z = mW1 * x_vec + mb1
    h = matrix(R, 2, 1, [z[1, 1]^2, z[2, 1]^2])
    f = mW2 * h + mb2
    F = f[1, 1] - f[2, 1]

    # 3. ED System
    dFdx1 = derivative(F, x1)
    dFdx2 = derivative(F, x2)

    eqs = [
        F,
        x1 - u1 - L * dFdx1,
        x2 - u2 - L * dFdx2
    ]

    I = ideal(R, eqs)

    # 4. Compute Discriminant via Elimination
    target_vars = [x1, x2, L]

    J_arr = [derivative(eq, v) for eq in eqs, v in target_vars]

    JacSub = matrix(R, 3, 3, J_arr)

    Crit = I + ideal(R, [det(JacSub)])

    # Saturate out degenerate cases (Rank < 2 condition)
    minors_ideal = ideal(R, minors(JacSub, 2))
    Sat = saturation(Crit, minors_ideal)

    # 5. Eliminate x1, x2, L
    D = eliminate(Sat, [x1, x2, L])

    disc = radical(D)

    return gens(disc)
end

# =============================================================================
# Examples
# =============================================================================

println("=== ED Discriminant ===\n")

println("--- Example 1 ---")
W1 = [1 2; 3 1]
W2 = [2 1; 1 2]
b1 = [0; 1]
b2 = [2; 1]

println("Computing ED Discriminant...")
disc1 = EDdisc(W1, W2, b1, b2)
println("ED Discriminant Ideal has $(length(disc1)) generator(s).")
println("The generators are:")
println("---------------------------------------------------")
for p in disc1
    println(p)
end
println("---------------------------------------------------")

println("\n--- Example 2 ---")
b1_ex2 = [1; 2]
b2_ex2 = [1; 1]

println("Computing ED Discriminant...")
disc2 = EDdisc(W1, W2, b1_ex2, b2_ex2)
println("ED Discriminant Ideal has $(length(disc2)) generator(s).")
println("The generators are:")
println("---------------------------------------------------")
for p in disc2
    println(p)
end
println("---------------------------------------------------")

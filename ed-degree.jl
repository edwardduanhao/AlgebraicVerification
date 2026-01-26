using HomotopyContinuation, Random

# ------------------------------------------------------------------
# Function: nn_poly_complex
# Purpose:  Generates the decision boundary polynomial for a neural network
#           using random complex weights.
# Motivation:
#   Using ComplexF64 weights ensures the resulting variety is "generic"
#   in the algebraic geometry sense. This prevents accidental symmetries
#   or singularities that might occur with real weights, ensuring the 
#   computed ED degree represents the true generic degree of the architecture.
# ------------------------------------------------------------------
function nn_poly_complex(n, layers, d)
    @var x[1:n]
    a = x
    for h in layers
        # Use ComplexF64 for weights to guarantee genericity
        a = (randn(ComplexF64, h, length(a)) * a + randn(ComplexF64, h)).^d
    end
    f = randn(ComplexF64, 2, length(a)) * a + randn(ComplexF64, 2)
    return f[1] - f[2], x
end




# ------------------------------------------------------------------
# Function: nn_poly_real
# Purpose:  Generates the decision boundary using standard Real weights.
# Note:     While closer to practical ML models, real weights may form a 
#           special locus in the parameter space. Comparing this degree 
#           to the complex case helps detect "real" anomalies.
# ------------------------------------------------------------------
function nn_poly_real(n, layers, d)
    @var x[1:n]
    a = x
    for h in layers
        # randn() generates Float64 (Real) by default
        a = (randn(h, length(a)) * a + randn(h)).^d
    end
    f = randn(2, length(a)) * a + randn(2)
    return f[1] - f[2], x
end





# ------------------------------------------------------------------
# Function: nn_poly_complex_scale
# Purpose:  Generates the polynomial with scaled complex coefficients.
# Critical for Numerical Stability:
#   Deep polynomial networks suffer from coefficient explosion.
#   A degree 3 activation across 3 layers results in total degree 3^3 = 27.
#   Without scaling, coefficients grow to ~10^15, causing Homotopy tracking failures.
#   A factor of 0.5 dampens this growth (0.5^3 = 0.125 per layer).
# ------------------------------------------------------------------
function nn_poly_complex_scale(n, layers, d)
    @var x[1:n]
    a = x
    
    # SCALING FACTOR: Crucial for Deep Networks
    # Use 0.5 or 0.4 for degree 3. 
    # (0.5^3 = 0.125, which prevents explosion without causing underflow)
    scale = 0.5
    
    for h in layers
        # Scale the random draws
        W = randn(ComplexF64, h, length(a)) .* scale
        b = randn(ComplexF64, h) .* scale
        
        a = (W * a + b).^d
    end
    
    # Final layer also scaled
    f = (randn(ComplexF64, 2, length(a)) .* scale) * a + (randn(ComplexF64, 2) .* scale)
    
    return f[1] - f[2], x
end





# ==================================================================
# MAIN EXECUTION: ED Degree Computation
# ==================================================================

# 1. Setup Polynomial System
#    We use the scaled complex network to ensure numerical convergence.
g, x = nn_poly_complex_scale(3, [3, 2], 3)

@var λ u[1:length(x)]
∇g = differentiate(g, x)



# 2. Formulate Critical Point Equations
#    The ED degree is the number of complex critical points of the distance 
#    function d(x) = ||x - u||^2 restricted to the variety g(x) = 0.
#    Lagrange Multiplier condition: (x - u) is parallel to gradient ∇g.
#    System:
#       x - u - λ * ∇g = 0   (Geometric condition)
#       g(x) = 0             (Constraint)
sys = System([x - u - λ * ∇g; g], [x; λ], u)




# 3. Solve for Generic Parameter u
#    We pick a random target 'u' and track paths to find all isolated roots.
#    'solve' (total degree homotopy) is robust here for finding the generic count.
u_target = randn(length(x))
result = solve(sys; target_parameters = u_target)




# 4. Output Results
#    Filter for non-singular solutions to get the ED degree.
grads = solutions(result)
println("Generic ED degree: ", length(grads))


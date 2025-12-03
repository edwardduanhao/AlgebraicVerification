using HomotopyContinuation

println("="^60)
println("Homotopy Continuation for Polynomial Neural Networks")
println("="^60)

# ============================================================================
# EXAMPLE 1: Basic Two-Layer Network with Automatic Expansion
# ============================================================================
println("\n[EXAMPLE 1] Basic network - automatic polynomial expansion")
println("-"^60)

# 1. Define symbolic input variables (these are the unknowns we solve for)
@polyvar x1 x2
x = [x1, x2]

# 2. Choose concrete weights and biases (constants)
W1 = [1.0  2.0;   # 2×2 weight matrix for hidden layer
      -1.0 1.0]
b1 = [0.5, -0.3]  # bias for hidden layer

W2 = [1.5 -0.7]   # 1×2 weight matrix for output layer
b2 = 0.1          # bias for output

# 3. First layer: z1 = W1 * x + b1
#    Since x1, x2 are polynomial variables, z1 is a vector of polynomials
z1 = W1 * x .+ b1

println("Hidden layer (before activation):")
println("  z1[1] = ", z1[1])
println("  z1[2] = ", z1[2])

# 4. Activation: σ(u) = u^2 applied elementwise
#    The computation graph automatically expands (linear)^2 → quadratic
h1 = z1 .^ 2

println("\nHidden layer (after σ(u)=u² activation):")
println("  h1[1] = ", h1[1])
println("  h1[2] = ", h1[2])

# 5. Output layer: y = W2 * h1 + b2
#    This produces the final polynomial (automatically expanded!)
y = W2 * h1 .+ b2

println("\nNetwork output (fully expanded polynomial):")
println("  f(x1, x2) = ", y[1])

# ============================================================================
# EXAMPLE 2: Solving for Inputs Given Target Output
# ============================================================================
println("\n\n[EXAMPLE 2] Finding inputs that produce specific output")
println("-"^60)

# Target output value
target = 5.0
println("Goal: Find all x where f(x) = ", target)

# Set up the polynomial system: f(x) - target = 0
system = [y[1] - target]

println("\nSolving system: ", system[1], " = 0")

# Solve using homotopy continuation
result = solve(system)

println("\nSolver results:")
println("  Number of solutions found: ", length(solutions(result)))
println("  Number of real solutions: ", nreal(result))
println("  Number of singular solutions: ", nsingular(result))

# Display all real solutions
println("\nReal solutions:")
real_sols = real_solutions(result)
for (i, sol) in enumerate(real_sols)
    println("  Solution $i: x1 = $(sol[1]), x2 = $(sol[2])")
    # Verify the solution
    verification = W2 * ((W1 * sol .+ b1) .^ 2) .+ b2
    println("    Verification: f(x) = ", verification[1], " ≈ ", target)
end

# ============================================================================
# EXAMPLE 3: Decision Boundary (Finding where f(x) = 0)
# ============================================================================
println("\n\n[EXAMPLE 3] Decision boundary - where does f(x) = 0?")
println("-"^60)

# Solve for the decision boundary
boundary_system = [y[1]]  # Find where output equals 0

println("Solving: ", boundary_system[1], " = 0")

result_boundary = solve(boundary_system)

println("\nBoundary solutions:")
println("  Total solutions: ", length(solutions(result_boundary)))
println("  Real solutions: ", nreal(result_boundary))

boundary_sols = real_solutions(result_boundary)
for (i, sol) in enumerate(boundary_sols)
    println("  Boundary point $i: x1 = $(sol[1]), x2 = $(sol[2])")
end

# ============================================================================
# EXAMPLE 4: Deeper Network (3 layers)
# ============================================================================
println("\n\n[EXAMPLE 4] Three-layer network with automatic expansion")
println("-"^60)

# Reset variables for clarity
@polyvar x1 x2

# Layer 1: 2 → 3
W1 = [1.0  0.5;
      -0.5 1.0;
      0.3  -0.8]
b1 = [0.1, -0.2, 0.3]

# Layer 2: 3 → 2
W2 = [0.8  -0.6  0.4;
      0.2   0.9 -0.3]
b2 = [0.1, -0.1]

# Layer 3: 2 → 1
W3 = [1.0  -0.5]
b3 = 0.05

# Forward pass with σ(u) = u²
z1 = W1 * [x1, x2] .+ b1
h1 = z1 .^ 2

z2 = W2 * h1 .+ b2
h2 = z2 .^ 2

output = (W3 * h2) .+ b3

println("Three-layer network output:")
println("  Number of terms: ", length(output[1]))
# Show first 100 chars of polynomial if it's too long
poly_str = string(output[1])
if length(poly_str) > 200
    println("  Polynomial (truncated): ", poly_str[1:200], "...")
else
    println("  Polynomial: ", poly_str)
end

# Solve for a target
target_3layer = 2.0
println("\nFinding x where 3-layer network outputs ", target_3layer)

result_3layer = solve([output[1] - target_3layer])
println("  Solutions found: ", length(solutions(result_3layer)))
println("  Real solutions: ", nreal(result_3layer))

real_sols_3layer = real_solutions(result_3layer)
if length(real_sols_3layer) > 0
    println("\nFirst few real solutions:")
    for (i, sol) in enumerate(real_sols_3layer[1:min(3, length(real_sols_3layer))])
        println("  Solution $i: x1 = $(round(sol[1], digits=4)), x2 = $(round(sol[2], digits=4))")
    end
end

# ============================================================================
# KEY INSIGHTS
# ============================================================================
println("\n\n" * "="^60)
println("KEY INSIGHTS")
println("="^60)
println("""
1. AUTOMATIC EXPANSION: We never manually expanded the polynomials!
   The computation graph (matrix ops + elementwise ^2) does it for us.

2. SYMBOLIC COMPUTATION: Using @polyvar creates symbolic variables that
   carry through all operations, building the polynomial automatically.

3. HOMOTOPY CONTINUATION: The solve() function uses homotopy continuation
   to track paths from a simple "start system" to our "target system",
   guaranteeing we find ALL solutions (including complex ones).

4. DEGREE GROWTH: For σ(x)=x², a 2-layer network has degree 2^L where L
   is the number of layers. The degree determines max number of solutions.

5. REAL vs COMPLEX: Homotopy continuation finds all complex solutions,
   but we often care most about real solutions (use real_solutions()).
""")

using HomotopyContinuation
using LinearAlgebra

println("="^70)
println("Homotopy Continuation for Polynomial Neural Networks")
println("Automatic Polynomial Expansion via Computation Graph")
println("="^70)

# ============================================================================
# EXAMPLE 1: Single Input, Automatic Expansion
# ============================================================================
println("\n[EXAMPLE 1] Single input - simple case")
println("-"^70)

# Define symbolic variable (the unknown we solve for)
@polyvar x

# Simple 2-layer network with σ(u) = u²
W1 = [2.0; -1.0]  # 2 neurons in hidden layer
b1 = [0.5, -0.3]

W2 = [1.5, -0.7]  # Output layer
b2 = 0.1

# Forward pass - computation graph does automatic expansion!
z1 = W1 * x .+ b1   # [2x + 0.5, -x - 0.3]
h1 = z1 .^ 2         # [(2x + 0.5)², (-x - 0.3)²] - expanded automatically
output = dot(W2, h1) + b2

println("Network computation graph:")
println("  Input: x")
println("  Hidden (linear): z1 = W1*x + b1")
println("  Hidden (activated): h1 = z1.^2  (elementwise)")
println("  Output: y = W2·h1 + b2")

println("\nAutomatic polynomial expansion result:")
println("  f(x) = ", output)

# Solve for specific target
target = 3.0
println("\n\nFinding x where f(x) = $target")
result = solve([output - target])

println("Solutions found: ", length(solutions(result)))
println("Real solutions: ", nreal(result))
for (i, sol) in enumerate(real_solutions(result))
   println("  x = ", sol[1])
   # Verify
   actual = 1.5 * (2 * sol[1] + 0.5)^2 - 0.7 * (-sol[1] - 0.3)^2 + 0.1
   println("    Verification: f($( sol[1])) = $actual ≈ $target")
end

# ============================================================================
# EXAMPLE 2: Two Inputs, Two Outputs (Square System)
# ============================================================================
println("\n\n[EXAMPLE 2] Two inputs, two outputs - square system")
println("-"^70)

@polyvar x1 x2

# 2-layer network: R² → R³ → R²
W1 = [1.0 0.5;
   -0.5 1.0;
   0.3 -0.8]
b1 = [0.1, -0.2, 0.3]

W2 = [0.8 -0.6 0.4;
   0.2 0.9 -0.3]
b2 = [0.05, -0.05]

# Forward pass with σ(u) = u²
z1 = W1 * [x1, x2] .+ b1
h1 = z1 .^ 2
y = W2 * h1 .+ b2

println("Network: 2 inputs → 3 hidden neurons → 2 outputs")
println("Activation: σ(u) = u²")
println("\nOutput polynomials (automatically expanded):")
println("  y1(x1, x2) = ", y[1])
println("  y2(x1, x2) = ", y[2])

# Solve for specific target output
target_y = [1.0, 0.5]
println("\n\nFinding (x1, x2) where [y1, y2] = $target_y")

system = [y[1] - target_y[1], y[2] - target_y[2]]
println("System to solve:")
println("  ", system[1], " = 0")
println("  ", system[2], " = 0")

result = solve(system)

println("\nSolver results:")
println("  Total solutions: ", length(solutions(result)))
println("  Real solutions: ", nreal(result))

real_sols = real_solutions(result)
if length(real_sols) > 0
   println("\nReal solutions:")
   for (i, sol) in enumerate(real_sols)
      println("  Solution $i: x1 = $(round(sol[1], digits=5)), x2 = $(round(sol[2], digits=5))")
   end
end

# ============================================================================
# EXAMPLE 3: Decision Boundary (Zero Level Set)
# ============================================================================
println("\n\n[EXAMPLE 3] Decision boundary for classification")
println("-"^70)

@polyvar x1 x2

# Simple classifier network
W1 = [1.0 1.0;
   -1.0 1.0]
b1 = [0.0, 0.0]

W2 = [1.0, -0.5]
b2 = -0.5

# Network with σ(x) = x²
z1 = W1 * [x1, x2] .+ b1
h1 = z1 .^ 2
classifier = dot(W2, h1) + b2

println("Classifier output (automatically expanded):")
println("  f(x1, x2) = ", classifier)

println("\nFinding decision boundary where f(x1, x2) = 0")
println("(This is a curve in 2D - for demo, we'll show a few sample points)")

# Sample a few points on the boundary
sample_x1_values = [-1.0, 0.0, 1.0, 2.0]
println("\nBoundary points:")
for x1_val in sample_x1_values
   boundary_poly_at_x1 = subs(classifier, x1 => x1_val)
   try
      result_slice = solve([boundary_poly_at_x1])
      x2_vals = real_solutions(result_slice)
      if length(x2_vals) > 0
         for x2_val in x2_vals
            println("  (x1, x2) = ($x1_val, $(round(x2_val[1], digits=4)))")
         end
      end
   catch
      # Skip if no real solutions
   end
end

# ============================================================================
# KEY INSIGHTS
# ============================================================================
println("\n\n" * "="^70)
println("KEY INSIGHTS")
println("="^70)
println("""
1. AUTOMATIC EXPANSION:
   - We define operations (W*x + b, elementwise ^2) as a computation graph
   - Julia's symbolic variables (@polyvar) automatically expand polynomials
   - No manual expansion needed - just write the network like normal code!

2. COMPUTATION GRAPH = POLYNOMIAL EXPANSION:
   - z = W*x + b  →  linear polynomials
   - h = z.^2     →  quadratic polynomials (automatic!)
   - y = W2*h + b →  combines quadratic terms (automatic!)

3. HOMOTOPY CONTINUATION:
   - solve() tracks paths from a "start system" to our "target system"
   - Guaranteed to find ALL solutions (including complex ones)
   - For polynomial of degree d in n variables, up to dⁿ solutions

4. SQUARE VS NON-SQUARE SYSTEMS:
   - Need same # equations as unknowns for finite solutions
   - 1 equation, 2 unknowns → infinite solutions (curve)
   - 2 equations, 2 unknowns → finite solutions (intersection points)

5. PRACTICAL USE:
   - Verification: Find ALL inputs that produce a given output
   - Decision boundaries: Find where output = 0 (or any threshold)
   - Adversarial examples: Find inputs near boundary
   - Preimage analysis: Understand network behavior
""")

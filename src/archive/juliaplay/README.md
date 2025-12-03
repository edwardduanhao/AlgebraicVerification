# Homotopy Continuation for Polynomial Neural Networks

## Overview

This directory contains examples demonstrating how to use HomotopyContinuation.jl with polynomial neural networks. The key insight is that **you don't need to manually expand polynomials** - the computation graph does it automatically!

## How It Works

### 1. Automatic Polynomial Expansion

When you use `@polyvar` to create symbolic variables in Julia, all operations maintain the polynomial structure:

```julia
@polyvar x1 x2

# Define a 2-layer network with σ(x) = x²
W1 = [1.0  2.0; -1.0  1.0]
b1 = [0.5, -0.3]
W2 = [1.5, -0.7]
b2 = 0.1

# Forward pass - just write normal code!
z1 = W1 * [x1, x2] .+ b1    # Linear polynomials
h1 = z1 .^ 2                 # Quadratic (automatic expansion!)
output = W2 * h1 .+ b2       # Final polynomial (automatic!)
```

The result `output` is a fully expanded polynomial: `0.412 + 3.42*x2 + 1.08*x1 + 5.3*x2² + 7.4*x1*x2 + 0.8*x1²`

### 2. Solving with Homotopy Continuation

Once you have the polynomial, you can solve for specific outputs:

```julia
# Find ALL inputs where f(x1, x2) = [target_y1, target_y2]
system = [y[1] - target_y1, y[2] - target_y2]
result = solve(system)

# Get real solutions
real_sols = real_solutions(result)
```

### 3. Key Advantages

**Automatic Expansion:**
- No manual polynomial expansion needed
- Write network code like normal (matrix ops + activation)
- Computation graph handles expansion automatically

**Complete Solutions:**
- Homotopy continuation finds ALL solutions (real & complex)
- Guaranteed not to miss solutions
- For degree-d polynomial in n variables: up to d^n solutions

**Applications:**
- **Verification:** Find all inputs producing a specific output
- **Decision Boundaries:** Find where output = threshold
- **Adversarial Examples:** Find inputs near boundaries
- **Preimage Analysis:** Understand inverse mappings

## Files

- `homotopy_simple_demo.jl` - Complete working examples with 3 demonstrations:
  - Example 1: Single-input network (1D → 2D → 1D)
  - Example 2: Two-input, two-output network (2D → 3D → 2D)
  - Example 3: Decision boundary analysis

- `demo.jl` - Original simple demo
- `homotopy_pnn_demo.jl` - Extended version with more examples

## Running the Examples

```bash
# Make sure you're in the project directory
cd /path/to/AlgebraicVerification

# Run the main demo
julia --project=. src/juliaplay/homotopy_simple_demo.jl
```

## Example Output

```
[EXAMPLE 1] Single input - simple case
Automatic polynomial expansion result:
  f(x) = 0.412 + 2.58*x + 5.3*x²

Finding x where f(x) = 3.0
Solutions found: 2
Real solutions: 2
  x = -0.983
  x = 0.497

[EXAMPLE 2] Two inputs, two outputs - square system
Finding (x1, x2) where [y1, y2] = [1.0, 0.5]
Solver results:
  Total solutions: 4
  Real solutions: 4
  Solution 1: x1 = 0.54042, x2 = 1.06684
  Solution 2: x1 = 1.02522, x2 = 0.0694
  ...
```

## Mathematical Background

### Polynomial Neural Networks

For a 2-layer network with activation σ(x) = x²:
- Input: x ∈ ℝⁿ
- Hidden: h = (W₁x + b₁)²  (elementwise)
- Output: y = W₂h + b₂

The output is a polynomial of degree 2^L where L is the number of layers.

### Homotopy Continuation

Solves F(x) = 0 by tracking paths from a simple "start system" G(x) = 0:
- Define homotopy: H(x, t) = (1-t)G(x) + tF(x)
- Start with solutions to G(x) = 0 at t = 0
- Track paths as t goes from 0 to 1
- Arrive at solutions to F(x) = 0 at t = 1

## Comparison with Python Implementation

The Python implementation in `src/torchhc/` solves a more complex problem (finding critical points with Lagrange multipliers), while these Julia examples focus on the direct problem of finding inputs for given outputs.

Both approaches demonstrate **automatic expansion**:
- **Julia:** Symbolic computation with `@polyvar`
- **Python:** Numeric computation with automatic differentiation

The Julia approach is particularly elegant for small problems where you want to see the expanded polynomial form.

# H5 to Homotopy Continuation - Quick Guide

## Loading PyTorch PNN Models from H5 Files

This guide shows how to load PyTorch Polynomial Neural Network models from H5 files and convert them to polynomials for use with HomotopyContinuation.jl.

## Quick Start

```julia
using HomotopyContinuation
include("src/juliaplay/demo.jl")

# 1. Load model from H5 file
model_data = load_pnn_from_h5("models/my_model.h5")

# 2. Convert to polynomial (variables created automatically!)
poly, vars = pnn_to_polynomial(model_data)

# 3. Extract variables for use
x1, x2 = vars

# 4. Solve with homotopy continuation
result = solve([poly - 0.5, x1 + x2 - 1.0])
real_solutions(result)
```

## Key Features

### Automatic Variable Creation ✨

**Before (manual):**
```julia
@polyvar x1 x2
poly = pnn_to_polynomial(model_data, [x1, x2])
```

**Now (automatic):**
```julia
poly, vars = pnn_to_polynomial(model_data)
x1, x2 = vars  # Variables created for you!
```

The function automatically creates variables `x1, x2, ..., xn` based on the input dimension.

### What Gets Loaded

From your H5 file:
```
✓ Architecture: 2 → [7, 6] → 1
  - Input dimension: 2
  - Hidden layers: [7, 6]
  - Output dimension: 1
  - Activation degree: 2 (polynomial)
```

The polynomial is automatically expanded:
```
✓ Polynomial created (degree 4)!
  - Fully expanded via computation graph
  - No manual work needed
```

## Functions

### `load_pnn_from_h5(filepath)`

Loads a PyTorch PNN model from an H5 file.

**Args:**
- `filepath`: Path to `.h5` file

**Returns:**
- `model_data`: NamedTuple with:
  - `architecture`: Dict with model info
  - `layers`: Vector of (weight, bias) tuples
  - `activations`: Vector of PolynomialActivation objects

**Example:**
```julia
model_data = load_pnn_from_h5("models/trained_model.h5")
println(model_data.architecture["input_dim"])  # 2
println(model_data.architecture["hidden_dims"])  # [7, 6]
```

### `pnn_to_polynomial(model_data; input_vars=nothing)`

Converts model to polynomial with automatic variable creation.

**Args:**
- `model_data`: Output from `load_pnn_from_h5()`
- `input_vars`: (Optional) Custom variables if you want specific names

**Returns:**
- `(poly, vars)`: Tuple of polynomial and variables

**Example 1: Automatic (recommended):**
```julia
poly, vars = pnn_to_polynomial(model_data)
x1, x2 = vars
```

**Example 2: Custom variables:**
```julia
@polyvar u v
poly, vars = pnn_to_polynomial(model_data; input_vars=[u, v])
```

### `apply_activation(act, x)`

Applies polynomial activation function.

**Internal function** - usually you don't need to call this directly.

## Complete Example

```julia
using HomotopyContinuation
include("src/juliaplay/demo.jl")

# Load model
println("Loading model...")
model_data = load_pnn_from_h5("models/new_model.h5")

# Convert to polynomial
println("Creating polynomial...")
poly, vars = pnn_to_polynomial(model_data)
x1, x2 = vars

# Evaluate at specific points
println("f(1.0, 0.0) = ", subs(poly, x1=>1.0, x2=>0.0))
println("f(0.0, 1.0) = ", subs(poly, x1=>0.0, x2=>1.0))

# Solve for inputs satisfying constraints
println("Finding solutions...")
result = solve([
    poly - 0.0,      # output = 0
    x1 + x2 - 0.5    # x1 + x2 = 0.5
])

println("Found $(nreal(result)) real solutions:")
for sol in real_solutions(result)
    println("  x1 = $(sol[1]), x2 = $(sol[2])")
end
```

## H5 File Format

Your PyTorch H5 file should contain:

```
model_class: "PolynomialNeuralNetwork"
layers.0.weight: [out_features × in_features] matrix
layers.0.bias: [out_features] vector
layers.1.weight: ...
layers.1.bias: ...
...
activations.0.coeffs: [c₀, c₁, c₂] for σ(x) = c₀ + c₁x + c₂x²
activations.1.coeffs: ...
```

## How It Works

1. **JLD2 reads H5 file**: Compatible with HDF5 format used by PyTorch
2. **Parse model structure**: Extract layers, weights, biases, activations
3. **Create variables**: Automatically generate `x1, x2, ...` based on input_dim
4. **Forward pass**: Execute network with symbolic variables
5. **Automatic expansion**: Computation graph expands polynomial automatically!

No manual polynomial expansion needed - it just works! ✨

## Supported Input Dimensions

Automatic variable creation supports `input_dim` from 1 to 5.

For higher dimensions, create variables manually:
```julia
@polyvar x1 x2 x3 x4 x5 x6
poly, vars = pnn_to_polynomial(model_data; input_vars=[x1, x2, x3, x4, x5, x6])
```

## Tips

1. **Check architecture first**: Inspect `model_data.architecture` before creating polynomial
2. **Polynomial size**: Degree grows as `activation_degree^num_layers`
3. **Real solutions**: Use `real_solutions(result)` to filter complex solutions
4. **Square systems**: Need same # equations as unknowns for finite solutions

## See Also

- `demo.jl` - Complete working example
- `pnn_to_hc.jl` - JSON loader (alternative format)
- `homotopy_simple_demo.jl` - Basic HC examples

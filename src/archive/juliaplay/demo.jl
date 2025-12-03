using HomotopyContinuation
using JLD2
using LinearAlgebra

"""
    PolynomialActivation
"""
struct PolynomialActivation
    degree::Int
    homogeneous::Bool
    coeffs::Vector{Float64}
end

"""
    apply_activation(act::PolynomialActivation, x)

Apply polynomial activation function to a symbolic or numeric value.
"""
function apply_activation(act::PolynomialActivation, x)
    if act.homogeneous
        return act.coeffs[1] * x^act.degree
    else
        # Non-homogeneous: sum of c[d] * x^d for d = 0 to degree
        result = act.coeffs[end]  # Start with highest degree coefficient
        for c in reverse(act.coeffs[1:end-1])
            result = result * x + c
        end
        return result
    end
end

"""
    load_pnn_from_h5(filepath::String)

Load a PolynomialNeuralNetwork from HDF5 file saved by PyTorch.

The H5 file should contain:
- model_class: String (e.g., "PolynomialNeuralNetwork")
- layers.i.weight: Weight matrices
- layers.i.bias: Bias vectors
- activations.i.coeffs: Activation coefficients

Returns a NamedTuple with:
- architecture: Dict with model info
- layers: Vector of (weight, bias) NamedTuples
- activations: Vector of PolynomialActivation objects
"""
function load_pnn_from_h5(filepath::String)

    # Load all data from H5 file
    data = load(filepath)

    # Extract model class
    model_class = get(data, "model_class", "Unknown")
    println("  Model class: $model_class")

    # Find all layer indices
    layer_keys = filter(k -> startswith(k, "layers."), keys(data))
    layer_indices = unique([parse(Int, split(k, ".")[2]) for k in layer_keys])
    sort!(layer_indices)

    println("  Found $(length(layer_indices)) layers")

    # Extract layers
    layers = []
    for i in layer_indices
        weight_key = "layers.$i.weight"
        bias_key = "layers.$i.bias"

        # HDF5 stores matrices in transposed form compared to PyTorch convention
        # PyTorch stores as (out_features, in_features)
        # We need to transpose to match Julia convention
        W = Matrix{Float64}(transpose(data[weight_key]))
        b = data[bias_key] !== nothing ? Vector{Float64}(data[bias_key]) : nothing

        push!(layers, (weight=W, bias=b))
        println("    Layer $i: weight $(size(W)), bias $(b === nothing ? "none" : size(b))")
    end

    # Find all activation indices
    act_keys = filter(k -> startswith(k, "activations."), keys(data))
    act_indices = unique([parse(Int, split(k, ".")[2]) for k in act_keys])
    sort!(act_indices)

    println("  Found $(length(act_indices)) activations")

    # Extract activations
    activations = []
    for i in act_indices
        coeffs_key = "activations.$i.coeffs"
        coeffs = Vector{Float64}(data[coeffs_key])

        degree = length(coeffs) - 1
        homogeneous = false  # Assume non-homogeneous unless specified

        act = PolynomialActivation(degree, homogeneous, coeffs)
        push!(activations, act)
        println("    Activation $i: degree=$degree, coeffs=$(coeffs)")
    end

    # Infer architecture
    input_dim = size(layers[1].weight, 2)
    output_dim = size(layers[end].weight, 1)
    hidden_dims = [size(layers[i].weight, 1) for i in 1:length(layers)-1]

    architecture = Dict(
        "input_dim" => input_dim,
        "output_dim" => output_dim,
        "hidden_dims" => hidden_dims,
        "degree" => length(activations) > 0 ? activations[1].degree : 0,
        "homogeneous" => length(activations) > 0 ? activations[1].homogeneous : false,
        "bias" => layers[1].bias !== nothing,
    )

    println("\n✓ Architecture: $input_dim → $hidden_dims → $output_dim")
    println("  Degree: $(architecture["degree"]), Homogeneous: $(architecture["homogeneous"])")

    return (
        architecture=architecture,
        layers=layers,
        activations=activations
    )
end

"""
    pnn_to_polynomial(model_data; input_vars=nothing)

Convert loaded PNN model to symbolic polynomial using HomotopyContinuation variables.

Args:
    model_data: Output from load_pnn_from_h5 or load_pnn_from_json
    input_vars: (Optional) Symbolic variables created with @polyvar.
                If not provided, variables will be created automatically as x1, x2, ...

Returns:
    (poly, vars): Tuple of:
        - poly: Polynomial expression(s) - vector if output_dim > 1, scalar if output_dim = 1
        - vars: Vector of input variables used

Example:
    poly, vars = pnn_to_polynomial(model_data)  # Automatic variable creation
    x1, x2 = vars
"""
function pnn_to_polynomial(model_data; input_vars=nothing)
    input_dim = model_data.architecture["input_dim"]

    # Create input variables automatically if not provided
    if input_vars === nothing
        println("\nCreating $input_dim input variables: x[1], x[2], ..., x[$input_dim]")

        # Use HomotopyContinuation's @polyvar macro with array syntax
        # This creates variables x[1], x[2], ..., x[input_dim]
        @polyvar x[1:input_dim]
        input_vars = x
    else
        # Handle single variable case
        if !isa(input_vars, AbstractVector)
            input_vars = [input_vars]
        end

        if length(input_vars) != input_dim
            error("Number of input variables ($(length(input_vars))) must match model input_dim ($input_dim)")
        end
        println("\nUsing provided input variables")
    end

    println("Building polynomial via forward pass...")

    # Initialize with input
    h = input_vars

    # Forward pass through layers
    layers = model_data.layers
    activations = model_data.activations

    for (i, layer) in enumerate(layers)
        W, b = layer.weight, layer.bias

        # Linear transformation: W * h + b
        h = W * h
        if b !== nothing
            h = h .+ b
        end

        # Apply activation (for all but last layer)
        if i < length(layers)
            act = activations[i]
            # Apply activation element-wise
            h = [apply_activation(act, h_j) for h_j in h]
        end

        println("  After layer $i: $(length(h)) outputs")
    end

    # Return as vector or scalar, along with variables
    output_dim = model_data.architecture["output_dim"]
    if output_dim == 1
        return h[1], input_vars  # Return scalar for single output
    else
        return h, input_vars  # Return vector for multiple outputs
    end
end

# ============================================================================
# DEMO: Load H5 and convert to polynomial
# ============================================================================

println("="^70)
println("H5 to Homotopy Continuation Demo")
println("="^70)

# Load the H5 file
path = joinpath(@__DIR__, "..", "..", "models", "new_model.h5")
model_data = load_pnn_from_h5(path)

# Convert to polynomial (variables created automatically!)
println("\n" * "="^70)
println("Converting to Polynomial")
println("="^70)

poly, vars = pnn_to_polynomial(model_data)

println("\n✓ Polynomial created successfully!")
println("\nPolynomial (first 300 chars):")
poly_str = string(poly)
if length(poly_str) > 300
    println(poly_str[1:300], "...")
else
    println(poly_str)
end

println("\n" * "="^70)
println("Verification: Evaluate polynomial at test points")
println("="^70)

# Test that the polynomial works by evaluating at specific points
test_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
println("\nEvaluating polynomial at test points:")
# vars is an array: [x[1], x[2]]
for (val1, val2) in test_points
    result = subs(poly, vars[1] => val1, vars[2] => val2)
    # The result is a constant polynomial, just print it
    println("  f($(val1), $(val2)) = $result")
end

println("\n" * "="^70)
println("Example: Solving with Homotopy Continuation")
println("="^70)

# Example: Sample the decision boundary by fixing x[1]
target = 0.0
println("\nProblem: Find x[2] where f(1.0, x[2]) = $target")
println("(This samples the decision boundary at x[1] = 1.0)")

poly_at_x1 = subs(poly, vars[1] => 1.0)

println("\nSolving...")
try
    result = solve([poly_at_x1 - target])

    println("\nResults:")
    println("  Total solutions: $(length(solutions(result)))")
    println("  Real solutions: $(nreal(result))")

    real_sols = real_solutions(result)
    if length(real_sols) > 0
        println("\nDecision boundary points at x1 = 1.0:")
        for (i, sol) in enumerate(real_sols[1:min(5, length(real_sols))])
            println("  Point $i: (1.0, $(round(sol[1], digits=4)))")
        end
    end
catch e
    println("Note: Could not solve - may need different target value")
    println("Error: $e")
end

# Example 2: Solve a square system (2 equations, 2 unknowns)
println("\n" * "-"^70)
println("Example 2: Square system (2 equations, 2 unknowns)")
println("-"^70)

target2 = -0.1
println("\nProblem: Find (x[1], x[2]) where:")
println("  - f(x[1], x[2]) = $target2")
println("  - x[1] + x[2] = 0.5")

system2 = [poly - target2, vars[1] + vars[2] - 0.5]

println("\nSolving...")
try
    result2 = solve(system2)

    println("\nResults:")
    println("  Total solutions: $(length(solutions(result2)))")
    println("  Real solutions: $(nreal(result2))")

    real_sols2 = real_solutions(result2)
    if length(real_sols2) > 0
        println("\nReal solutions:")
        for (i, sol) in enumerate(real_sols2[1:min(5, length(real_sols2))])
            println("  Solution $i: x[1] = $(round(sol[1], digits=4)), x[2] = $(round(sol[2], digits=4))")
            println("    Check: x[1] + x[2] = $(round(sol[1] + sol[2], digits=4))")
        end
    end
catch e
    println("Note: Could not solve - system may have no real solutions")
    println("Error: $e")
end

println("\n" * "="^70)
println("✓ Demo complete!")
println("="^70)

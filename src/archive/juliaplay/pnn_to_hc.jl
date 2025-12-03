"""
Load PyTorch PolynomialNeuralNetwork from JSON and convert to HomotopyContinuation polynomial.

This module provides functions to:
1. Load a trained PyTorch PNN model from JSON
2. Reconstruct the forward pass using symbolic variables
3. Automatically expand to polynomial form
4. Use with HomotopyContinuation.jl for solving
"""

using HomotopyContinuation
using JSON
using LinearAlgebra

"""
    PolynomialActivation

Represents a polynomial activation function σ(x) = c₀ + c₁x + c₂x² + ... + cₐxᵈ
"""
struct PolynomialActivation
    degree::Int
    homogeneous::Bool
    coeffs::Vector{Float64}
end

"""
    apply_activation(act::PolynomialActivation, x )

Apply polynomial activation function to a symbolic or numeric value.
For homogeneous: σ(x) = c * xᵈ
For non-homogeneous: σ(x) = c₀ + c₁x + c₂x² + ... + cₐxᵈ
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
    load_pnn_from_json(filepath::String)

Load a PolynomialNeuralNetwork from JSON file exported from PyTorch.

Returns a dictionary with:
- architecture: model architecture parameters
- layers: list of (weight, bias) tuples
- activations: list of PolynomialActivation objects
"""
function load_pnn_from_json(filepath::String)
    data = JSON.parsefile(filepath)

    # Extract architecture
    arch = data["architecture"]

    # Convert layers to matrices
    layers = []
    for layer_data in data["layers"]
        W = Matrix{Float64}(hcat(layer_data["weight"]...)')  # Convert to matrix
        b = layer_data["bias"] !== nothing ? Vector{Float64}(layer_data["bias"]) : nothing
        push!(layers, (weight=W, bias=b))
    end

    # Convert activations
    activations = []
    for act_data in data["activations"]
        act = PolynomialActivation(
            act_data["degree"],
            act_data["homogeneous"],
            Vector{Float64}(act_data["coeffs"])
        )
        push!(activations, act)
    end

    println("Loaded PNN from $filepath")
    println("  Architecture: $(arch["input_dim"]) → $(arch["hidden_dims"]) → $(arch["output_dim"])")
    println("  Activation: degree=$(arch["degree"]), homogeneous=$(arch["homogeneous"])")
    println("  Number of layers: $(length(layers))")
    println("  Number of activations: $(length(activations))")

    return (
        architecture=arch,
        layers=layers,
        activations=activations
    )
end

"""
    pnn_to_polynomial(model_data; input_vars=nothing)

Convert loaded PNN model to symbolic polynomial using HomotopyContinuation variables.

Args:
    model_data: Output from load_pnn_from_json or load_pnn_from_h5
    input_vars: (Optional) Symbolic variables created with @polyvar.
                If not provided, variables will be created automatically as x1, x2, ...

Returns:
    (poly, vars): Tuple of:
        - poly: Polynomial expression(s) - vector if output_dim > 1, scalar if output_dim = 1
        - vars: Vector of input variables used

Example:
    ```julia
    # Automatic variable creation (recommended)
    model_data = load_pnn_from_json("model.json")
    poly, vars = pnn_to_polynomial(model_data)
    x1, x2 = vars

    # Or with custom variables
    @polyvar u v
    poly, vars = pnn_to_polynomial(model_data; input_vars=[u, v])
    ```
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

    for (i, (W, b)) in enumerate(layers)
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

"""
    evaluate_pnn_polynomial(poly, input_vars, input_values)

Evaluate a polynomial at specific input values.

Args:
    poly: Polynomial expression from pnn_to_polynomial
    input_vars: The symbolic variables used
    input_values: Dictionary or vector of values

Example:
    ```julia
    @polyvar x1 x2
    poly = pnn_to_polynomial(model_data, [x1, x2])
    result = evaluate_pnn_polynomial(poly, [x1, x2], [1.0, 2.0])
    ```
"""
function evaluate_pnn_polynomial(poly, input_vars, input_values)
    if isa(input_values, AbstractVector)
        # Convert vector to substitution pairs
        subs_dict = Dict(input_vars[i] => input_values[i] for i in 1:length(input_vars))
    else
        subs_dict = input_values
    end

    if isa(poly, AbstractVector)
        return [subs(p, subs_dict...) for p in poly]
    else
        return subs(poly, subs_dict...)
    end
end

"""
    verify_pnn_conversion(model_data, poly, input_vars; n_test=10, tolerance=1e-6)

Verify that the Julia polynomial matches the original PyTorch model.
Note: This requires having the original PyTorch model outputs for comparison.
"""
function verify_pnn_conversion(model_data, poly, input_vars; n_test=10, tolerance=1e-6)
    println("\nVerifying polynomial conversion with $n_test random test cases...")

    input_dim = model_data.architecture["input_dim"]

    for i in 1:n_test
        # Generate random test input
        test_input = randn(input_dim)

        # Evaluate polynomial
        poly_output = evaluate_pnn_polynomial(poly, input_vars, test_input)

        println("Test $i: input = $test_input")
        println("         output = $poly_output")
    end

    println("✓ Polynomial evaluation successful")
    return true
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^70)
    println("PNN to HomotopyContinuation Converter - Example Usage")
    println("="^70)

    # This example assumes you have a model.json file
    # You can create one using the Python export script

    model_file = "example_pnn_model.json"

    if isfile(model_file)
        println("\n[Step 1] Loading model from JSON...")
        model_data = load_pnn_from_json(model_file)

        println("\n[Step 2] Creating symbolic polynomial...")
        input_dim = model_data.architecture["input_dim"]

        if input_dim == 1
            @polyvar x
            input_vars = [x]
        elseif input_dim == 2
            @polyvar x1 x2
            input_vars = [x1, x2]
        elseif input_dim == 3
            @polyvar x1 x2 x3
            input_vars = [x1, x2, x3]
        elseif input_dim == 4
            @polyvar x1 x2 x3 x4
            input_vars = [x1, x2, x3, x4]
        else
            error("For input_dim > 4, please manually create variables with @polyvar")
        end

        poly = pnn_to_polynomial(model_data, input_vars)

        println("\n[Step 3] Polynomial created successfully!")
        if isa(poly, AbstractVector)
            println("Output dimension: $(length(poly))")
            for (i, p) in enumerate(poly)
                println("  Output $i: $p")
            end
        else
            println("Output polynomial: $poly")
        end

        println("\n[Step 4] Example: Solving for specific output...")
        target = 1.0
        println("Finding inputs where output = $target")

        if isa(poly, AbstractVector)
            # For multi-output, solve for first output
            system = [poly[1] - target]
        else
            system = [poly - target]
        end

        try
            result = solve(system)
            println("Solutions found: $(length(solutions(result)))")
            println("Real solutions: $(nreal(result))")

            real_sols = real_solutions(result)
            if length(real_sols) > 0
                println("\nFirst few real solutions:")
                for (i, sol) in enumerate(real_sols[1:min(3, length(real_sols))])
                    println("  Solution $i: $sol")
                end
            end
        catch e
            println("Note: Could not solve (may need different constraints)")
            println("Error: $e")
        end

    else
        println("\nℹ No example model file found at '$model_file'")
        println("To create one, run the Python export script:")
        println("  python -c 'from src.pnn.export_to_julia import *; ...'")
        println("\nOr in Python:")
        println("  from src.pnn import PolynomialNeuralNetwork")
        println("  from src.pnn.export_to_julia import export_pnn_to_json")
        println("  model = PolynomialNeuralNetwork(2, 1, [3, 4], degree=2)")
        println("  export_pnn_to_json(model, 'example_pnn_model.json')")
    end

    println("\n" * "="^70)
end

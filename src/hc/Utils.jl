module Utils

using JSON3, HDF5
using DynamicPolynomials

export load_model


struct PolynomialNeuralNetwork
    input_dim::Int
    output_dim::Int
    hidden_dims::Vector{Int}
    act_degree::Int
    homogeneous::Bool
    bias::Bool
    weights::Dict{String,Array{Float32}}
end


function poly_activation(x, coeffs, degree, homogeneous)
    """Apply polynomial activation: c[0] + c[1]*x + c[2]*x^2 + ..."""
    if homogeneous
        # Vectorized: c[degree] * x^degree
        return coeffs[degree+1] .* x .^ degree
    else
        # Vectorized Horner's method for polynomial evaluation
        result = coeffs[degree+1] .* ones(eltype(x), size(x)...)
        for d in (degree-1):-1:0
            result = result .* x .+ coeffs[d+1]
        end
        return result
    end
end


function pnn_forward(model::PolynomialNeuralNetwork)
    """
    Symbolic forward pass through the polynomial neural network.
    Returns the output as a polynomial expression.
    """
    # Create symbolic input variables
    @polyvar x[1:model.input_dim]

    # Start with input
    h = collect(x)

    # Number of hidden layers (layers with activation)
    num_hidden_layers = length(model.hidden_dims)

    # Apply hidden layers with activations
    for i in 0:(num_hidden_layers-1)
        # Linear transformation: h = W' * h + b
        # Note: Weights are stored as (in_features, out_features), so we transpose
        W = model.weights["layers.$i.weight"]'

        # Matrix-vector multiplication
        h = W * h

        # Add bias if present
        if model.bias
            b = model.weights["layers.$i.bias"]
            h = h .+ b
        end

        # Polynomial activation
        coeffs = model.weights["activations.$i.coeffs"]
        h = poly_activation(h, coeffs, model.act_degree, model.homogeneous)
    end

    # Final output layer (no activation)
    i = num_hidden_layers
    W = model.weights["layers.$i.weight"]'
    output = W * h

    if model.bias
        b = model.weights["layers.$i.bias"]
        output = output .+ b
    end

    return output, x
end


function load_model(path::String)
    """
    Load a model from an experiment directory.

    Expected structure:
        path/
        ├── model/
        │   ├── model_config.json
        │   └── model_weights.h5
        └── analysis/

    Args:
        path: Path to experiment directory (e.g., "experiments/run_20241202_143022"
              or "experiments/latest")

    Returns:
        Tuple of (model_forward, model) where model_forward is the symbolic forward pass.
    """
    # Resolve path if it's a symlink (e.g., "experiments/latest")
    resolved_path = realpath(path)

    # Construct paths to model files
    config_path = joinpath(resolved_path, "model", "model_config.json")
    weights_path = joinpath(resolved_path, "model", "model_weights.h5")

    # Validate files exist
    if !isfile(config_path)
        error("Model config not found: $config_path")
    end
    if !isfile(weights_path)
        error("Model weights not found: $weights_path")
    end

    # Read config
    cfg = JSON3.read(read(config_path, String))
    model_class = String(cfg["model_class"])

    println(repeat("=", 60))
    println("Loading model from: $resolved_path")
    println("Model class: $model_class")

    # Read weights
    weights = Dict{String,Array{Float32}}()
    h5open(weights_path, "r") do f
        for name in keys(f)
            println("  Loading weight: $name")
            weights[name] = read(f[name])
        end
    end

    # Construct model based on model_class
    if model_class == "PolynomialNeuralNetwork"
        model = PolynomialNeuralNetwork(
            cfg["input_dim"],
            cfg["output_dim"],
            collect(cfg["hidden_dims"]),
            cfg["act_degree"],
            cfg["homogeneous"],
            cfg["bias"],
            weights,
        )
        model_forward = pnn_forward(model)
        println("Model loaded successfully!")
        println(repeat("=", 60))
        return model_forward, model
    else
        error("Unknown model_class: $model_class")
    end
end

end  # module Utils

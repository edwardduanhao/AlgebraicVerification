# Experiment Folder Structure

## Overview

This project uses an organized folder structure for managing models and analysis results.

## Directory Structure

```
experiments/
├── run_20241202_143022/
│   ├── model/
│   │   ├── model_config.json       # Model architecture configuration
│   │   ├── model_weights.h5        # Model weights
│   │   └── metadata.json           # Training metadata (optional)
│   └── analysis/
│       ├── robust_radius.npz       # Julia analysis results
│       └── verification.npz        # Other analysis outputs
├── run_20241202_150045/
│   └── ...
└── latest -> run_20241202_150045/  # Symlink to most recent
```

## Python API

### Saving Models

```python
from src.utils.utils import save_model

# Option 1: Automatic experiment directory (recommended)
exp_dir = save_model(
    model,
    metadata={
        "description": "Binary classifier",
        "epochs": 100,
        "learning_rate": 0.001,
    }
)
# Creates: experiments/run_YYYYMMDD_HHMMSS/model/

# Option 2: Manual path (backward compatibility)
save_model(model, path="custom/path", create_experiment=False)
```

### Getting Experiment Paths

```python
from src.utils.utils import get_experiment_path, list_experiments

# Get latest experiment
latest = get_experiment_path("latest")

# Get specific experiment
exp = get_experiment_path("run_20241202_143022")

# List all experiments
experiments = list_experiments()
```

## Julia Integration

### Loading Models in Julia

```julia
include("src/hc/Utils.jl")
using .Utils

# Load from latest experiment
project_root = "experiments/latest"
model_forward, _ = Utils.load_model(project_root)
```

### Saving Analysis Results

```julia
# In your Julia analysis code
function robust_radius(project_root::String, xi_list; save_path=nothing)
    # ... analysis code ...

    if !isnothing(save_path)
        npzwrite(save_path, results)
    end
end

# Usage
xi_list = [[0.5, 0.5], [1.0, 0.0]]
save_path = "experiments/latest/analysis/robust_radius.npz"
robust_radius("experiments/latest", xi_list, save_path=save_path)
```

## Workflow Example

### 1. Train and Save Model (Python)

```python
from src.pnn import PolynomialNeuralNetwork
from src.utils.utils import save_model
from src.utils.training import train_model

# Create and train model
model = PolynomialNeuralNetwork(2, 2, [3, 4], act_degree=2)
train_model(model, train_data, epochs=100)

# Save with metadata
exp_dir = save_model(
    model,
    metadata={
        "dataset": "my_dataset",
        "epochs": 100,
        "accuracy": 0.95,
    }
)
print(f"Saved to: {exp_dir}")
```

### 2. Run Julia Analysis

```julia
include("src/hc/EuclideanHC.jl")

# Analyze the latest model
project_root = "experiments/latest"
xi_list = [[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]]

# Results saved to experiments/latest/analysis/
save_path = joinpath(project_root, "analysis", "robust_radius.npz")
sols = robust_radius(project_root, xi_list, save_path=save_path)
```

### 3. Load and Visualize Results (Python)

```python
import numpy as np
from src.utils.utils import get_experiment_path

# Load latest experiment results
exp_dir = get_experiment_path("latest")
analysis_file = exp_dir / "analysis" / "robust_radius.npz"

data = np.load(analysis_file)
xi_list = data["xi_list"]
min_dist = data["min_dist"]
closest_sol = data["closest_sol"]

print(f"Robust radii: {min_dist}")
```

## Benefits

1. **Organized**: Clear separation between model and analysis files
2. **Timestamped**: Easy to track experiments chronologically
3. **Accessible**: `latest` symlink for quick access to most recent
4. **Version Control Friendly**: `experiments/` is gitignored
5. **Python-Julia Integration**: Both can access the same structured data

## File Formats

- **Python Models**: `.h5` (HDF5 format for weights), `.json` (config/metadata)
- **Julia Analysis**: `.npz` (NumPy format, readable from both Python and Julia)
- **Config/Metadata**: `.json` (human-readable, language-agnostic)
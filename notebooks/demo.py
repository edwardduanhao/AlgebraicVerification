import juliacall  # Must be imported before torch to avoid segfault
import torch
import torch.nn as nn
import sys, os
import numpy as np

# Add the project root directory to the path
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pnn import PolynomialNeuralNetwork
from src.utils import save_model
from src.hc import compute_robust_radius, load_robust_radius_results, verify_experiment


model = PolynomialNeuralNetwork(
    input_dim=2,
    output_dim=3,
    hidden_dims=[1, 1],
    act_degree=3,
    homogeneous=False,
    bias=True,
    s=0.1,
)


path = save_model(
    model,
    metadata={"description": "try"},
)


res = compute_robust_radius(
    experiment_path=path,
    xi_list=[[0.5, 0.5], [1.0, 0.0], [1.0, 1.0]],
    verbose=False,
    save_results=True,
    save_detailed=True,
)
print(res["min_dist"])
print(res["closest_sol"])

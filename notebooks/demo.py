import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys, os
import numpy as np

# Add the project root directory to the path
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pnn import PolynomialNeuralNetwork
from src.utils import save_model, plot_db
from src.hc import compute_robust_radius, load_robust_radius_results, verify_experiment
from src.data import SinusoidDataset
from src.utils import train_epochs


# Create dataset
train_dataset = SinusoidDataset(size=2000)

model = PolynomialNeuralNetwork(
    input_dim=2,
    output_dim=2,
    hidden_dims=[8, 6],
    act_degree=3,
    homogeneous=False,
    bias=True,
    s=0.1,
)

# Create DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Train the model
history = train_epochs(
    model=model,
    train_loader=train_loader,
    num_epochs=1000,
    optimizer_type="adam",
    learning_rate=1e-3,
    verbose=True,
)

print(f"Final training loss: {history['train_loss'][-1]:.4f}")
print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")

model = model.cpu()

# Optional: visualize decision boundary
# plot_db(
#     model,
#     x_range=(-np.pi, np.pi),
#     y_range=(-2, 2),
#     grid_size=1000,
#     save="decision_boundary.pdf",
#     show=True,
# )

path = save_model(
    model,
    metadata={"description": "try"},
)


# Get some sample points from the dataset for robust radius computation
xi_list = [train_dataset[i][0].tolist() for i in range(3)]

res = compute_robust_radius(
    experiment_path=path,
    xi_list=xi_list,
    verbose=False,
    save_results=True,
    save_detailed=True,
)
print(res["min_dist"])
print(res["closest_sol"])

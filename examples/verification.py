"""
Verification example: Train a Polynomial Neural Network (PNN) on the
Steiner Roman surface dataset, compute a certified robust radius via
homotopy continuation, and visualize the decision boundary in 3D.
Reproduces Figure 1 in the paper.
"""

import sys, os

# Add the project root directory to the path
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# IMPORTANT: Import juliacall BEFORE torch to avoid segfaults.
# juliacall is used internally by src.hc for homotopy continuation.
# See: https://github.com/pytorch/pytorch/issues/78829
from juliacall import Main as jl

import numpy as np
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from skimage import measure

from src.data import SteinerRomanDataset
from src.pnn import PolynomialNeuralNetwork
from src.utils import train_epochs, save_model
from src.hc import compute_robust_radius


def make_glossy_surface(verts, faces, x_range, y_range, z_range, grid_size):
    """Create a glossy Mesh3d surface from marching cubes output."""
    dx = (x_range[1] - x_range[0]) / (grid_size - 1)
    dy = (y_range[1] - y_range[0]) / (grid_size - 1)
    dz = (z_range[1] - z_range[0]) / (grid_size - 1)

    x = x_range[0] + verts[:, 0] * dx
    y = y_range[0] + verts[:, 1] * dy
    z = z_range[0] + verts[:, 2] * dz

    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="rgb(25, 140, 235)",
        opacity=1.0,
        flatshading=False,
        lighting=dict(
            ambient=0.45, diffuse=0.65, specular=0.35, roughness=0.35, fresnel=0.08
        ),
        lightposition=dict(x=2, y=0.5, z=1.5),
        showlegend=False,
    )


def make_glossy_sphere(center, radius):
    """Create a glossy translucent sphere with shading."""
    u, v = np.linspace(0, 2 * np.pi, 200), np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v)) * 1
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v)) * 1
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v)) * 1

    # Shading based on surface normal dot light direction
    light_dir = np.array([1.0, 0.5, 1.2])
    light_dir /= np.linalg.norm(light_dir)

    nx = np.outer(np.cos(u), np.sin(v))
    ny = np.outer(np.sin(u), np.sin(v))
    nz = np.outer(np.ones_like(u), np.cos(v))

    shade = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    shade = (shade + 1) / 2

    return go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=shade,
        colorscale=[
            [0.0, "rgb(190, 50, 50)"],
            [0.5, "rgb(215, 55, 50)"],
            [1.0, "rgb(240, 75, 60)"],
        ],
        opacity=0.55,
        showscale=False,
        lighting=dict(
            ambient=0.6, diffuse=0.5, specular=0.2, roughness=0.4, fresnel=0.05
        ),
        lightposition=dict(x=2, y=0.5, z=1.5),
        showlegend=False,
        hoverinfo="skip",
        contours=dict(
            x=dict(highlight=False), y=dict(highlight=False), z=dict(highlight=False)
        ),
    )


if __name__ == "__main__":
    # ── 1. Data & Model Setup ────────────────────────────────────────────
    train_dataset = SteinerRomanDataset(size=1000)
    train_loader = DataLoader(train_dataset, batch_size=1000)

    model = PolynomialNeuralNetwork(
        input_dim=3, output_dim=2, hidden_dims=[20, 20], act_degree=2
    )

    # ── 2. Training ──────────────────────────────────────────────────────
    history = train_epochs(
        model=model,
        train_loader=train_loader,
        num_epochs=10000,
        optimizer_type="adam",
        learning_rate=1e-3,
        verbose=True,
    )

    # ── 3. Save Model ───────────────────────────────────────────────────
    path = save_model(
        model,
        metadata={
            "description": "steiner roman 3d",
            "input_dim": 3,
            "output_dim": 2,
            "hidden_dims": [20, 20],
            "act_degree": 2,
            "num_epochs": 10000,
            "optimizer_type": "adam",
            "learning_rate": 1e-3,
        },
    )

    # ── 4. Robustness Verification ───────────────────────────────────────
    # Compute the certified robust radius around each query point via
    # homotopy continuation (solved in Julia through juliacall).
    xi_list = [[0.45, 0.45, 0.45]]

    res = compute_robust_radius(
        experiment_path=path,
        xi_list=xi_list,
        verbose=True,
        save_results=True,
        save_detailed=True,
    )

    print(
        f"The minimum distance is {res['min_dist']}, "
        f"and the closest solution is {res['closest_sol']}."
    )

    # ── 5. 3D Decision-Boundary Visualization ─────────────────────────
    # Evaluate the trained model on a dense 3D grid to extract the
    # decision boundary as an isosurface (logit_0 - logit_1 = 0).
    model = model.cpu()
    model.double()

    x_range = (-1, 1)
    y_range = (-1, 1)
    z_range = (-1, 1)
    grid_size = 250

    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    z_grid = np.linspace(z_range[0], z_range[1], grid_size)
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid)

    grid_points = torch.tensor(
        np.c_[X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], dtype=torch.float64
    )

    model.eval()
    with torch.no_grad():
        logits = model(grid_points)

    logits_grid = logits.numpy().reshape(X_grid.shape + (logits.shape[1],))

    # Decision boundary: where logit[0] - logit[1] = 0
    decision_values = logits_grid[:, :, :, 0] - logits_grid[:, :, :, 1]

    # Extract the isosurface via marching cubes
    verts, faces, normals, values = measure.marching_cubes(decision_values, level=0)

    # ── 6. Build Plotly Figure ───────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(
        make_glossy_surface(verts, faces, x_range, y_range, z_range, grid_size)
    )

    # Overlay query points and their certified robust-radius spheres
    xi_array = np.array(xi_list)

    fig.add_trace(
        go.Scatter3d(
            x=xi_array[:, 0],
            y=xi_array[:, 1],
            z=xi_array[:, 2],
            mode="markers",
            marker=dict(color="black", size=3),
            showlegend=False,
        )
    )

    for xi, r in zip(xi_list, res["min_dist"]):
        fig.add_trace(make_glossy_sphere(xi, r))

    fig.update_layout(
        scene=dict(
            bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=2.2)),
        ),
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        showlegend=False,
        paper_bgcolor="white",
    )

    # Uncomment to save as PDF
    # fig.write_image("steiner_roman_3d.pdf")
    fig.show(renderer="browser")

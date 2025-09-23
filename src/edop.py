from sklearn.cluster import DBSCAN
import numpy as np
import sympy as sp
import re
import time
import phcpy as hc  # Homotopy continuation package
from phcpy.dimension import get_core_count  # For getting number of CPU cores
from phcpy.solver import solve  # For solving polynomial systems
from phcpy.solutions import filter_real  # For filtering real solutions
import plotly.graph_objects as go


def dedup_dbscan(X, tol, objective=None, strategy="first"):
    """
    DBSCAN with eps=tol. Choose representative per cluster:
      - strategy="first": first point in cluster
      - strategy="min_obj": argmin of objective array within cluster
    Returns X_unique and kept indices.
    """
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        return X, np.array([], dtype=int)

    labels = DBSCAN(eps=tol, min_samples=1, metric="euclidean").fit_predict(X)

    kept = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if strategy == "min_obj" and objective is not None:
            kept.append(idx[np.argmin(objective[idx])])
        else:
            kept.append(idx[0])

    kept = np.array(sorted(kept))
    return X[kept], kept


def get_critical_points(
    x_center: np.ndarray, poly: sp.Expr, model, tol: float = 1e-8, plot=True
):
    # Ensure the polynomial uses the correct variables
    x = sp.symbols(f"x0:{model.input_dim}")
    assert poly.free_symbols == set(
        x
    ), "Variables should be x0, x1, ..., x{input_dim-1}"

    # Define the Lagrange multiplier
    lambd = sp.symbols("lambda")
    H_poly = [poly]
    F_poly = sp.Matrix([poly])

    for i, x_ in enumerate(x):
        H_poly.append(lambd * (F_poly.jacobian([x_])[0, 0]) + x_ - x_center.tolist()[i])

    H_str = [str(p) + ";" for p in H_poly]
    nbcores = get_core_count() - 1
    print(f"Using {nbcores} CPU cores for homotopy continuation.")
    wstart = time.time()
    sols = solve(H_str, tasks=nbcores)
    real_sols = filter_real(sols, tol=tol, oper="select")
    wend = time.time()
    print(f"Solved in {wend - wstart:.2f} seconds.")
    print(
        f"Found {len(real_sols)} real solutions out of {len(sols)} complex solutions."
    )

    coords = []
    for sol in real_sols:
        # Extract lines with x0 and x1
        values = []
        for j in range(model.input_dim):
            match = re.search(rf"x{j}\s*:\s*([-+0-9.Ee]+)", sol)
            if match:
                values.append(float(match.group(1)))
        if len(values) == model.input_dim:
            coords.append(values)

    real_sols = np.array(coords)
    real_sols, _ = dedup_dbscan(real_sols, tol=tol, strategy="first")
    print(f"Removing duplicates, {len(real_sols)} unique real solutions remain.")

    dist = np.sum((real_sols - x_center) ** 2, axis=1)
    gamma = np.sqrt(np.min(dist))
    best_critical_point = real_sols[np.argmin(dist)]
    print(f"Closest critical point: {best_critical_point}")
    print(f"Distance to closest critical point (gamma): {gamma}")

    if plot:
        f = sp.lambdify(x, poly, "numpy")
        x_grid = np.linspace(-30, 10, 500)
        y_grid = np.linspace(-20, 20, 500)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = f(X, Y)

        fig = go.Figure()
        fig.add_trace(
            go.Contour(
                x=x_grid,
                y=y_grid,
                z=Z,
                contours=dict(start=0, end=0, coloring="lines"),
                colorscale=[[0, "blue"], [1, "blue"]],
                showscale=False,
                line=dict(width=3),
                name="Decision Boundary",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_center[0]],
                y=[x_center[1]],
                mode="markers",
                marker=dict(color="red", size=6),
                name="Test Center Point",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=real_sols[:, 0],
                y=real_sols[:, 1],
                mode="markers",
                marker=dict(color="orange", size=6),
                name="Critical Points",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[best_critical_point[0]],
                y=[best_critical_point[1]],
                mode="markers",
                marker=dict(color="green", size=6),
                name="The Closest Critical Point",
            )
        )

        for i in range(real_sols.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=[x_center[0], real_sols[i, 0]],
                    y=[x_center[1], real_sols[i, 1]],
                    mode="lines",
                    line=dict(color="grey", dash="dashdot", width=1),
                    showlegend=False,
                )
            )

        # Add circle as a shape
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=x_center[0] - gamma,
            y0=x_center[1] - gamma,
            x1=x_center[0] + gamma,
            y1=x_center[1] + gamma,
            line=dict(color="LightSeaGreen", dash="dashdot"),
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                x=0.95,
                y=0.95,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.6)",  # optional: semi-transparent background
                bordercolor="black",
                borderwidth=1,
            ),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            xaxis_title="x0",
            yaxis_title="x1",
            width=800,
            height=800,
        )

        fig.show()
        fig.write_image(
            "../figures/critical_points.pdf", width=800, height=800, scale=2
        )

    return best_critical_point, gamma, real_sols

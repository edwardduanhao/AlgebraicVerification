# Homotopy continuation for solving cubic polynomials.
# Reproduces Figure 2 in the paper.

import numpy as np
import matplotlib.pyplot as plt


# --- Target system: F(x) = x^3 + ax^2 + bx + c ---


def F(x, a, b, c):
    return x**3 + a * x**2 + b * x + c


def dF(x, a, b, c):
    return 3 * x**2 + 2 * a * x + b


# --- Start system: G(x) = x^3 - 1, roots are cube roots of unity ---


def G(x):
    return x**3 - 1


def dG(x):
    return 3 * x**2


# --- Homotopy: H(x,t) = gamma * (1-t) * G(x) + t * F(x) ---


def H(x, t, gamma, a, b, c):
    return gamma * (1 - t) * G(x) + t * F(x, a, b, c)


def H_x(x, t, gamma, a, b, c):
    return gamma * (1 - t) * dG(x) + t * dF(x, a, b, c)


def H_t(x, t, gamma, a, b, c):
    return -gamma * G(x) + F(x, a, b, c)


def start_points():
    return np.exp(2j * np.pi * np.arange(3) / 3)


def sample_gamma():
    return np.exp(1j * np.random.uniform(0, 2 * np.pi))


# Predictor: Euler step along tangent (Davidenko equation)
# From H(x,t)=0, dx/dt = -H_t / H_x
def predict(x, t, dt, gamma, a, b, c):
    dxdt = -H_t(x, t, gamma, a, b, c) / H_x(x, t, gamma, a, b, c)
    return x + dxdt * dt


# Corrector: Newton's method at fixed t to solve H(x,t)=0
def correct(x, t, gamma, a, b, c, tol=1e-12, max_iter=10):
    path = [x]
    for _ in range(max_iter):
        h = H(x, t, gamma, a, b, c)
        if np.abs(h) < tol:
            return x, path
        x = x - h / H_x(x, t, gamma, a, b, c)
        path.append(x)
    return x, path


# Track path from t=0 to t=1
def track(x0, gamma, a, b, c, dt=0.01):
    x, t = x0, 0.0
    path_x = [x]
    path_t = [t]
    while t < 1.0:
        dt_step = min(dt, 1.0 - t)
        x = predict(x, t, dt_step, gamma, a, b, c)
        t = t + dt_step
        x, _ = correct(x, t, gamma, a, b, c)
        path_x.append(x)
        path_t.append(t)
    return np.array(path_x), np.array(path_t)


# Solve F(x) = 0 by tracking all 3 paths
def solve(a, b, c, dt=0.01):
    gamma = sample_gamma()
    paths = []
    sols = []
    for x0 in start_points():
        path_x, path_t = track(x0, gamma, a, b, c, dt)
        paths.append((path_x, path_t))
        sols.append(path_x[-1])
    return sols, paths, gamma


# --- Plotting ---


def plot_paths(paths, save=False):
    """Plot the three homotopy continuation paths in the complex plane."""
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(figsize=(4, 4))
    all_x = np.concatenate([p[0] for p in paths])
    cx = (all_x.real.max() + all_x.real.min()) / 2
    cy = (all_x.imag.max() + all_x.imag.min()) / 2

    for i, (path_x, path_t) in enumerate(paths):
        ax.plot(path_x.real, path_x.imag, color=colors[i])
        ax.scatter(path_x[0].real, path_x[0].imag, color=colors[i], marker="o", s=40)
        ax.scatter(path_x[-1].real, path_x[-1].imag, color=colors[i], marker="X", s=40)

    ax.scatter([], [], color="gray", marker="o", s=40, label="Start")
    ax.scatter([], [], color="gray", marker="X", s=40, label="End")
    ax.legend()
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")

    margin = 0.1
    max_range = max(
        all_x.real.max() - all_x.real.min(), all_x.imag.max() - all_x.imag.min()
    )
    half = max_range / 2 + margin
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    if save:
        plt.savefig("hc_paths.png", dpi=300, bbox_inches="tight")
        print(f"Saved hc_paths.png")
    plt.show()


def plot_predictor_corrector(paths, gamma, a, b, c, dt=2e-1, save=False):
    """Illustrate a single predictor-corrector step."""
    path_x, path_t = paths[2]
    index = int(len(path_x) * 0.3)
    x_t = path_x[index]
    t = path_t[index]

    x_t_next = predict(x=x_t, t=t, dt=dt, gamma=gamma, a=a, b=b, c=c)
    _, corrector_path = correct(x=x_t_next, t=t + dt, gamma=gamma, a=a, b=b, c=c)
    corrector_path = np.array(corrector_path)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(
        path_x.real[int(len(path_x) * 0.2) : int(len(path_x) * 0.6)],
        path_x.imag[int(len(path_x) * 0.2) : int(len(path_x) * 0.6)],
        color="tab:blue",
    )
    ax.plot(
        [x_t.real, x_t_next.real],
        [x_t.imag, x_t_next.imag],
        linestyle="dotted",
        color="tab:orange",
    )
    ax.plot(
        corrector_path.real, corrector_path.imag, linestyle="dotted", color="tab:blue"
    )
    ax.scatter([x_t.real], [x_t.imag], marker="o", s=50, color="tab:blue")
    ax.scatter([x_t_next.real], [x_t_next.imag], marker="o", s=50, color="tab:orange")
    ax.scatter(
        [corrector_path[-1].real],
        [corrector_path[-1].imag],
        marker="o",
        s=50,
        color="tab:blue",
    )
    ax.set_xlabel("Real axis")
    ax.set_ylabel("Imaginary axis")

    if save:
        plt.savefig("hc_predictor_corrector.png", dpi=300, bbox_inches="tight")
        print(f"Saved hc_predictor_corrector.png")

    plt.show()


if __name__ == "__main__":
    np.random.seed(2026)

    # Solve F(x) = x^3 - 4x^2 + 8x - 8
    a, b, c = -4, 8, -8
    sols, paths, gamma = solve(a, b, c, dt=1e-3)

    for root in sols:
        assert np.abs(F(root, a, b, c)) < 1e-10, f"F({root}) = {F(root, a, b, c)}"
    print("All roots verified.")

    plot_paths(paths, save=False)
    plot_predictor_corrector(paths, gamma, a, b, c, save=False)

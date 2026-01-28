import numpy as np


# Target system: F(x) = x^3 + ax^2 + bx + c
def F(x, a, b, c):
    return x**3 + a * x**2 + b * x + c


# Derivative of F
def dF(x, a, b, c):
    return 3 * x**2 + 2 * a * x + b


# Start system: G(x) = x^3 - 1, roots are cube roots of unity
def G(x):
    return x**3 - 1


# Derivative of G
def dG(x):
    return 3 * x**2


# Homotopy: H(x,t) = gamma * (1-t) * G(x) + t * F(x)
def H(x, t, gamma, a, b, c):
    return gamma * (1 - t) * G(x) + t * F(x, a, b, c)


# Partial derivative of H with respect to x
def H_x(x, t, gamma, a, b, c):
    return gamma * (1 - t) * dG(x) + t * dF(x, a, b, c)


# Partial derivative of H with respect to t
def H_t(x, t, gamma, a, b, c):
    return -gamma * G(x) + F(x, a, b, c)


# Start points: cube roots of unity
def start_points():
    return np.exp(2j * np.pi * np.arange(3) / 3)


# Sample gamma from unit circle
def sample_gamma():
    return np.exp(1j * np.random.uniform(0, 2 * np.pi))


# Predictor: Euler step along tangent direction ((Davidenko equation)
# From H(x,t)=0, we have dx/dt = -H_t / H_x
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


# Track path from t=0 to t=1, return full path
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

import numpy as np
from math import comb
from typing import Optional


def sample_coeff(n: int, d: int, m: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Draw n samples of {a_k} with m hidden units, isotropic Gaussians for w1, w2, b1.
    Returns shape (n, d + 1)
    """

    rng = np.random.default_rng(seed)

    # m-dimensional isotropic Gaussians for each sample
    w1 = rng.standard_normal((n, m))
    w2 = rng.standard_normal((n, m))
    b1 = rng.standard_normal((n, m))

    K = np.arange(d + 1)
    C = np.array([comb(d, int(k)) for k in K], dtype=np.float64)  # binomial coeffs

    # Broadcast powers: shapes -> (n, m, |K|)
    w1_pow = w1[:, :, None] ** K[None, None, :]
    b1_pow = b1[:, :, None] ** (d - K)[None, None, :]

    # Sum over m and scale
    A = (C[None, None, :] / np.sqrt(m)) * (w2[:, :, None] * w1_pow * b1_pow)
    A = A.sum(axis=1)

    return A


def count_real_roots(
    coeffs: np.ndarray, eps: float = 1e-12, normalize: bool = False
) -> tuple[int, np.ndarray]:
    """
    Count real roots via NumPy's roots function.
    coeffs are ascending by degree: a0, a1, ..., ad.
    """

    # Optionally normalize coefficients to avoid numerical issues
    if normalize:
        norm = np.linalg.norm(coeffs, axis=1, keepdims=True)
        coeffs = coeffs / np.maximum(norm, eps)

    # np.roots expects highest degree first
    coeffs_flipped = coeffs[:, ::-1]

    counts = []
    all_roots = []
    for row in coeffs_flipped:
        roots = np.roots(row)
        real_mask = np.abs(roots.imag) < eps
        counts.append(np.sum(real_mask))
        all_roots.append(roots[real_mask].real)

    return np.array(counts), all_roots

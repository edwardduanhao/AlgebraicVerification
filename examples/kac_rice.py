# Kac-Rice simulation: expected number of real zeros for polynomial activations.
# Reproduces Figure 5 in the paper.

import numpy as np
import matplotlib.pyplot as plt
from math import comb


def sample_coeff(n, d, m, seed=None):
    """
    Draw n samples of {a_k} with m hidden units, isotropic Gaussians for w1, w2, b1.
    Returns shape (n, d + 1).
    """
    rng = np.random.default_rng(seed)

    w1 = rng.standard_normal((n, m))
    w2 = rng.standard_normal((n, m))
    b1 = rng.standard_normal((n, m))

    K = np.arange(d + 1)
    C = np.array([comb(d, int(k)) for k in K], dtype=np.float64)

    w1_pow = w1[:, :, None] ** K[None, None, :]
    b1_pow = b1[:, :, None] ** (d - K)[None, None, :]

    A = (C[None, None, :] / np.sqrt(m)) * (w2[:, :, None] * w1_pow * b1_pow)
    A = A.sum(axis=1)

    return A


def count_real_roots(coeffs, eps=1e-12, normalize=False):
    """
    Count real roots via NumPy's roots function.
    coeffs are ascending by degree: a0, a1, ..., ad.
    """
    if normalize:
        norm = np.linalg.norm(coeffs, axis=1, keepdims=True)
        coeffs = coeffs / np.maximum(norm, eps)

    coeffs_flipped = coeffs[:, ::-1]

    counts = []
    all_roots = []
    for row in coeffs_flipped:
        roots = np.roots(row)
        real_mask = np.abs(roots.imag) < eps
        counts.append(np.sum(real_mask))
        all_roots.append(roots[real_mask].real)

    return np.array(counts), all_roots


if __name__ == "__main__":
    ds = np.arange(2, 11)  # degrees to test
    n = 1000  # number of samples per degree
    m = 20000  # number of hidden units
    seed = 2026  # random seed

    counts = []
    for d in ds:
        a_samples = sample_coeff(n, d, m, seed=seed)
        count, _ = count_real_roots(a_samples, eps=1e-10, normalize=True)
        counts.append(count)
        print(f"d={d}, average real roots: {np.mean(count):.4f}")

    # Plot empirical vs theoretical
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    for i, vals in enumerate(counts):
        x = np.full_like(vals[:40], i, dtype=float)
        x += np.random.uniform(-0.15, 0.15, size=len(vals[:40]))
        ax.plot(x, vals[:40], "o", alpha=0.35, markersize=4, color="lightsteelblue")

    means = [np.mean(s) for s in counts]
    theory = ds / np.sqrt(2 * ds - 1)

    ax.plot(range(len(ds)), means, "o-", linewidth=1.5, label="Empirical mean")
    ax.plot(range(len(ds)), theory, "o--", linewidth=1.5, label="Theoretical mean")

    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels(ds)
    ax.set_xlabel("Activation degree $d$")
    ax.set_ylabel("Number of real zeros")

    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()

    # Uncomment to save the figure
    # plt.savefig("kac_rice_real_roots.pdf", dpi=300)
    print("Saved kac_rice_real_roots.pdf")
    plt.show()

"""Data generation for benchmark instances."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from .config import BenchmarkConfig


def generate_random_input(
    n_samples: int,
    input_dim: int,
    low: float = -1.0,
    high: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random input points uniformly in [low, high]^d.

    Args:
        n_samples: Number of samples to generate.
        input_dim: Input dimension.
        low: Lower bound for uniform distribution.
        high: Upper bound for uniform distribution.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, input_dim).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_samples, input_dim))


def generate_random_labels(
    n_samples: int,
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random class labels.

    Args:
        n_samples: Number of samples.
        n_classes: Number of classes.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with integer labels in [0, n_classes).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n_samples)


def generate_counterexample_perturbation(
    n_samples: int,
    input_dim: int,
    epsilon: float,
    r: float = 0.98,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate perturbations for counterexamples using Euclidean (ℓ2) norm.

    Perturbations are sampled such that ||delta||_2 is in [r*epsilon, epsilon].
    Direction is sampled uniformly on the unit sphere, magnitude in [r*epsilon, epsilon].

    Args:
        n_samples: Number of perturbations to generate.
        input_dim: Input dimension.
        epsilon: Maximum perturbation radius (Euclidean).
        r: Minimum ratio (perturbation magnitude >= r * epsilon).
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, input_dim) with perturbations.
    """
    rng = np.random.default_rng(seed)

    # Sample random directions uniformly on the unit sphere
    # Method: sample from standard normal, then normalize
    directions = rng.standard_normal(size=(n_samples, input_dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms  # Unit vectors

    # Sample magnitudes uniformly from [r*epsilon, epsilon]
    magnitudes = rng.uniform(r * epsilon, epsilon, size=(n_samples, 1))

    return directions * magnitudes


def generate_counterexample_labels(
    original_labels: np.ndarray,
    n_classes: int = 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate target labels for counterexamples (different from original).

    Args:
        original_labels: Original correct labels.
        n_classes: Number of classes.
        seed: Random seed for reproducibility.

    Returns:
        Array of target labels, each different from the corresponding original.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(original_labels)

    # For binary classification, just flip the label
    if n_classes == 2:
        return 1 - original_labels

    # For multi-class, sample from other classes
    cex_labels = np.zeros(n_samples, dtype=int)
    for i, y in enumerate(original_labels):
        other_classes = [c for c in range(n_classes) if c != y]
        cex_labels[i] = rng.choice(other_classes)

    return cex_labels


def generate_unverifiable_instances(
    config: BenchmarkConfig,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate unverifiable instances with planted counterexamples.

    Args:
        config: Benchmark configuration.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: x0, y, x_cex, y_cex, delta_cex
    """
    # Use different sub-seeds for different components
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=4)

    n = config.n_unverifiable
    d = config.input_dim
    k = config.output_dim

    # Generate clean inputs and labels
    x0 = generate_random_input(n, d, seed=seeds[0])
    y = generate_random_labels(n, k, seed=seeds[1])

    # Generate counterexample perturbations and labels
    delta_cex = generate_counterexample_perturbation(
        n, d, config.epsilon, config.r, seed=seeds[2]
    )
    y_cex = generate_counterexample_labels(y, k, seed=seeds[3])

    # Compute counterexample points
    x_cex = x0 + delta_cex

    return {
        "x0": x0.astype(np.float32),
        "y": y.astype(np.int64),
        "x_cex": x_cex.astype(np.float32),
        "y_cex": y_cex.astype(np.int64),
        "delta_cex": delta_cex.astype(np.float32),
    }


def generate_clean_instances(
    config: BenchmarkConfig,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate clean instances without planted counterexamples.

    Args:
        config: Benchmark configuration.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys: x0, y
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=2)

    n = config.n_clean
    d = config.input_dim
    k = config.output_dim

    x0 = generate_random_input(n, d, seed=seeds[0])
    y = generate_random_labels(n, k, seed=seeds[1])

    return {
        "x0": x0.astype(np.float32),
        "y": y.astype(np.int64),
    }


def generate_and_save_instances(
    config: BenchmarkConfig,
    output_dir: Path,
    seed: Optional[int] = None,
) -> Tuple[Path, Path]:
    """
    Generate and save both unverifiable and clean instances.

    Args:
        config: Benchmark configuration.
        output_dir: Directory to save instances.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (unverifiable_path, clean_path).
    """
    output_dir = Path(output_dir)
    instances_dir = output_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    # Use different seeds for unverifiable and clean
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=2)

    # Generate and save unverifiable instances
    unverifiable_data = generate_unverifiable_instances(config, seed=seeds[0])
    unverifiable_path = instances_dir / "unverifiable.npz"
    np.savez(unverifiable_path, **unverifiable_data)

    # Generate and save clean instances
    clean_data = generate_clean_instances(config, seed=seeds[1])
    clean_path = instances_dir / "clean.npz"
    np.savez(clean_path, **clean_data)

    return unverifiable_path, clean_path


def load_instances(instances_dir: Path) -> Tuple[dict, dict]:
    """
    Load previously saved instances.

    Args:
        instances_dir: Directory containing unverifiable.npz and clean.npz.

    Returns:
        Tuple of (unverifiable_data, clean_data) dictionaries.
    """
    instances_dir = Path(instances_dir)

    unverifiable_path = instances_dir / "unverifiable.npz"
    clean_path = instances_dir / "clean.npz"

    with np.load(unverifiable_path) as data:
        unverifiable = {key: data[key] for key in data.files}

    with np.load(clean_path) as data:
        clean = {key: data[key] for key in data.files}

    return unverifiable, clean


if __name__ == "__main__":
    # Test data generation
    from .config import BENCHMARK_CONFIGS

    config = BENCHMARK_CONFIGS[0]
    print(f"Testing data generation for: {config.name}")
    print(f"Epsilon: {config.epsilon}, r: {config.r}")
    print()

    # Generate unverifiable instances
    unverifiable = generate_unverifiable_instances(config, seed=42)
    print("Unverifiable instances:")
    print(f"  x0 shape: {unverifiable['x0'].shape}")
    print(f"  y shape: {unverifiable['y'].shape}")
    print(f"  x_cex shape: {unverifiable['x_cex'].shape}")
    print(f"  y_cex shape: {unverifiable['y_cex'].shape}")
    print(f"  delta_cex shape: {unverifiable['delta_cex'].shape}")
    print()

    # Verify perturbation bounds (Euclidean norm)
    delta_l2_norm = np.linalg.norm(unverifiable["delta_cex"], axis=1)
    print(
        f"  ||delta_cex||_2: min={delta_l2_norm.min():.4f}, max={delta_l2_norm.max():.4f}"
    )
    print(f"  Expected range: [{config.r * config.epsilon:.4f}, {config.epsilon:.4f}]")
    print()

    # Verify labels are different
    labels_differ = (unverifiable["y"] != unverifiable["y_cex"]).all()
    print(f"  All y != y_cex: {labels_differ}")
    print()

    # Generate clean instances
    clean = generate_clean_instances(config, seed=42)
    print("Clean instances:")
    print(f"  x0 shape: {clean['x0'].shape}")
    print(f"  y shape: {clean['y'].shape}")

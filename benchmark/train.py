"""Training script for benchmark models."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import h5py
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse

from .config import BenchmarkConfig, BENCHMARK_CONFIGS
from .data import generate_and_save_instances, load_instances
from src.pnn import PolynomialNeuralNetwork


def margin_loss(
    logits: torch.Tensor,
    y_original: torch.Tensor,
    y_target: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """
    Margin loss for counterexamples.

    Encourages: logits[y_target] > logits[y_original] with a small margin.
    Loss = max(0, logits[y_original] - logits[y_target] + margin)

    Args:
        logits: Model output logits, shape (batch, n_classes).
        y_original: Original (correct) labels, shape (batch,).
        y_target: Target (wrong) labels for counterexamples, shape (batch,).
        margin: Desired margin (default 0.01).

    Returns:
        Scalar loss value.
    """
    batch_size = logits.size(0)
    batch_idx = torch.arange(batch_size)

    logits_original = logits[batch_idx, y_original]
    logits_target = logits[batch_idx, y_target]

    # Hinge loss: want logits_target > logits_original
    loss = torch.relu(logits_original - logits_target + margin)
    return loss.mean()


def train_model(
    config: BenchmarkConfig,
    unverifiable_data: dict,
    clean_data: dict,
    num_epochs: int = 5000,
    lr: float = 0.001,
    margin: float = 0.01,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> PolynomialNeuralNetwork:
    """
    Train a polynomial neural network for the benchmark.

    Training objectives:
    1. CE loss on x0 for all instances (correct classification)
    2. Margin loss on x_cex for unverifiable instances (misclassification)

    Args:
        config: Benchmark configuration.
        unverifiable_data: Dict with x0, y, x_cex, y_cex.
        clean_data: Dict with x0, y.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        margin: Margin for counterexample loss.
        seed: Random seed for reproducibility.
        verbose: Print training progress.

    Returns:
        Trained model.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create model
    model = PolynomialNeuralNetwork(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dims=config.hidden_dims,
        act_degree=config.act_degree,
        homogeneous=config.homogeneous,
        bias=config.bias,
        s=0.1,  # Small initialization for stability
    )

    # Prepare data
    x0_unv = torch.tensor(unverifiable_data["x0"], dtype=torch.float32)
    y_unv = torch.tensor(unverifiable_data["y"], dtype=torch.long)
    x_cex = torch.tensor(unverifiable_data["x_cex"], dtype=torch.float32)
    y_cex = torch.tensor(unverifiable_data["y_cex"], dtype=torch.long)

    x0_clean = torch.tensor(clean_data["x0"], dtype=torch.float32)
    y_clean = torch.tensor(clean_data["y"], dtype=torch.long)

    # Combine all x0 and y for classification loss
    x0_all = torch.cat([x0_unv, x0_clean], dim=0)
    y_all = torch.cat([y_unv, y_clean], dim=0)

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss()

    # Optimizer with cyclic learning rate (as in the paper)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cyclic learning rate scheduler
    def lr_lambda(epoch):
        # Ramp up in first half, ramp down in second half
        if epoch < num_epochs // 2:
            return epoch / (num_epochs // 2)
        else:
            return (num_epochs - epoch) / (num_epochs // 2)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 1. Classification loss on all x0
        logits_x0 = model(x0_all)
        loss_ce = ce_loss_fn(logits_x0, y_all)

        # 2. Margin loss on counterexamples
        logits_cex = model(x_cex)
        loss_margin = margin_loss(logits_cex, y_unv, y_cex, margin=margin)

        # Total loss
        loss = loss_ce + loss_margin

        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % 1000 == 0:
            # Evaluate accuracy
            model.eval()
            with torch.no_grad():
                pred_x0 = model(x0_all).argmax(dim=1)
                acc_x0 = (pred_x0 == y_all).float().mean().item()

                pred_cex = model(x_cex).argmax(dim=1)
                acc_cex = (pred_cex == y_cex).float().mean().item()
            model.train()

            print(
                f"  Epoch {epoch+1}/{num_epochs}: "
                f"loss={loss.item():.4f}, "
                f"acc_x0={acc_x0:.2%}, "
                f"acc_cex={acc_cex:.2%}"
            )

    return model


def save_model(
    model: PolynomialNeuralNetwork,
    config: BenchmarkConfig,
    output_dir: Path,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save trained model to benchmark results directory.

    Args:
        model: Trained model.
        config: Benchmark configuration.
        output_dir: Output directory for this config.
        metadata: Optional additional metadata.

    Returns:
        Path to model directory.
    """
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model config
    model_config = {
        "model_class": "PolynomialNeuralNetwork",
        "input_dim": model.input_dim,
        "hidden_dims": list(model.hidden_dims),
        "output_dim": model.output_dim,
        "act_degree": model.act_degree,
        "homogeneous": model.homogeneous,
        "bias": model.bias,
    }

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Save weights
    state = model.state_dict()
    with h5py.File(model_dir / "model_weights.h5", "w") as f:
        for name, tensor in state.items():
            f.create_dataset(name, data=tensor.detach().cpu().numpy())

    # Save metadata
    meta = {
        "benchmark_config": config.name,
        "epsilon": config.epsilon,
        "architecture": config.architecture,
        "act_degree": config.act_degree,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return model_dir


def evaluate_model(
    model: PolynomialNeuralNetwork,
    unverifiable_data: dict,
    clean_data: dict,
) -> dict:
    """
    Evaluate trained model on benchmark instances.

    Args:
        model: Trained model.
        unverifiable_data: Dict with x0, y, x_cex, y_cex.
        clean_data: Dict with x0, y.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    with torch.no_grad():
        # Unverifiable instances
        x0_unv = torch.tensor(unverifiable_data["x0"], dtype=torch.float32)
        y_unv = torch.tensor(unverifiable_data["y"], dtype=torch.long)
        x_cex = torch.tensor(unverifiable_data["x_cex"], dtype=torch.float32)
        y_cex = torch.tensor(unverifiable_data["y_cex"], dtype=torch.long)

        pred_x0_unv = model(x0_unv).argmax(dim=1)
        pred_cex = model(x_cex).argmax(dim=1)

        acc_x0_unv = (pred_x0_unv == y_unv).float().mean().item()
        acc_cex = (pred_cex == y_cex).float().mean().item()

        # Clean instances
        x0_clean = torch.tensor(clean_data["x0"], dtype=torch.float32)
        y_clean = torch.tensor(clean_data["y"], dtype=torch.long)

        pred_x0_clean = model(x0_clean).argmax(dim=1)
        acc_x0_clean = (pred_x0_clean == y_clean).float().mean().item()

    return {
        "acc_x0_unverifiable": acc_x0_unv,
        "acc_counterexample": acc_cex,
        "acc_x0_clean": acc_x0_clean,
        "n_correct_x0_unv": int((pred_x0_unv == y_unv).sum().item()),
        "n_correct_cex": int((pred_cex == y_cex).sum().item()),
        "n_correct_x0_clean": int((pred_x0_clean == y_clean).sum().item()),
    }


def train_single_config(
    config: BenchmarkConfig,
    results_dir: Path,
    num_epochs: int = 5000,
    lr: float = 0.001,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Train model for a single configuration.

    Args:
        config: Benchmark configuration.
        results_dir: Base results directory.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dictionary with training results and paths.
    """
    output_dir = results_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {config.name}")
        print(f"  Architecture: {config.architecture}")
        print(f"  Activation degree: {config.act_degree}")
        print(f"  Epsilon: {config.epsilon}")
        print(f"{'='*60}")

    # Generate and save data
    if verbose:
        print("\nGenerating data...")
    generate_and_save_instances(config, output_dir, seed=seed)

    # Load data
    unverifiable, clean = load_instances(output_dir / "instances")

    # Train model
    if verbose:
        print("\nTraining model...")
    model = train_model(
        config=config,
        unverifiable_data=unverifiable,
        clean_data=clean,
        num_epochs=num_epochs,
        lr=lr,
        seed=seed,
        verbose=verbose,
    )

    # Evaluate model
    eval_results = evaluate_model(model, unverifiable, clean)

    if verbose:
        print(f"\nEvaluation:")
        print(
            f"  Accuracy on x0 (unverifiable): {eval_results['acc_x0_unverifiable']:.2%}"
        )
        print(f"  Accuracy on x_cex: {eval_results['acc_counterexample']:.2%}")
        print(f"  Accuracy on x0 (clean): {eval_results['acc_x0_clean']:.2%}")

    # Save model
    save_model(
        model,
        config,
        output_dir,
        metadata={"training": {"num_epochs": num_epochs, "lr": lr}},
    )

    # Save evaluation results
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    return {
        "config": config.name,
        "output_dir": str(output_dir),
        "evaluation": eval_results,
    }


def train_all(
    results_dir: Optional[Path] = None,
    num_epochs: int = 5000,
    lr: float = 0.001,
    seed: int = 42,
    verbose: bool = True,
) -> list:
    """
    Train models for all benchmark configurations.

    Args:
        results_dir: Base results directory.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        seed: Base random seed.
        verbose: Print progress.

    Returns:
        List of training results for each config.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, config in enumerate(BENCHMARK_CONFIGS):
        # Use different seed for each config
        config_seed = seed + i

        result = train_single_config(
            config=config,
            results_dir=results_dir,
            num_epochs=num_epochs,
            lr=lr,
            seed=config_seed,
            verbose=verbose,
        )
        all_results.append(result)

    # Save summary
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "num_configs": len(BENCHMARK_CONFIGS),
                "num_epochs": num_epochs,
                "lr": lr,
                "base_seed": seed,
                "results": all_results,
            },
            f,
            indent=2,
        )

    if verbose:
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"Trained {len(all_results)} models")
        print(f"Results saved to: {results_dir}")

        # Print summary table
        print(f"\n{'Config':<30} {'x0 acc':<10} {'cex acc':<10}")
        print("-" * 50)
        for r in all_results:
            name = r["config"]
            x0_acc = r["evaluation"]["acc_x0_unverifiable"]
            cex_acc = r["evaluation"]["acc_counterexample"]
            print(f"{name:<30} {x0_acc:<10.2%} {cex_acc:<10.2%}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train benchmark models")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: benchmark/results)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Number of training epochs (default: 5000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else None

    train_all(
        results_dir=results_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        verbose=not args.quiet,
    )

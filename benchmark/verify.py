"""Verification script for benchmark models."""

import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse

from .config import BenchmarkConfig, BENCHMARK_CONFIGS
from .data import load_instances


def verify_single_config(
    config: BenchmarkConfig,
    results_dir: Path,
    verbose: bool = True,
) -> dict:
    """
    Run verification on a single benchmark configuration.

    Args:
        config: Benchmark configuration.
        results_dir: Base results directory containing trained models.
        verbose: Print progress.

    Returns:
        Dictionary with verification results.
    """
    # Import here to avoid Julia initialization on module load
    from src.hc.hc import compute_robust_radius

    config_dir = results_dir / config.name

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Verifying: {config.name}")
        print(f"  Epsilon: {config.epsilon}")
        print(f"{'='*60}")

    # Load instances
    unverifiable, clean = load_instances(config_dir / "instances")

    # Prepare all x0 points for verification (combine into single call)
    x0_unv = unverifiable["x0"]  # (n_unv, input_dim)
    x0_clean = clean["x0"]  # (n_clean, input_dim)
    n_unv = len(x0_unv)
    n_clean = len(x0_clean)

    # Combine all instances into one array
    x0_all = np.concatenate([x0_unv, x0_clean], axis=0)  # (n_unv + n_clean, input_dim)

    if verbose:
        print(
            f"\nVerifying {len(x0_all)} instances ({n_unv} unverifiable + {n_clean} clean)..."
        )

    # Single call to compute_robust_radius for all instances
    results_all = compute_robust_radius(
        experiment_path=config_dir,
        xi_list=x0_all,
        verbose=False,
        save_results=True,
        output_filename="robust_radius_all.npz",
    )

    # Split results back into unverifiable and clean
    radii_all = results_all["min_dist"]
    radii_unv = radii_all[:n_unv]
    radii_clean = radii_all[n_unv:]

    # Analyze results
    epsilon = config.epsilon

    # Unverifiable instances: expect robust_radius <= epsilon (falsified)
    verified_unv = radii_unv > epsilon
    falsified_unv = radii_unv <= epsilon

    # Clean instances: may be verified or not
    verified_clean = radii_clean > epsilon
    falsified_clean = radii_clean <= epsilon

    # Compile results
    verification_results = {
        "config": config.name,
        "epsilon": epsilon,
        "unverifiable": {
            "n_total": len(radii_unv),
            "n_verified": int(verified_unv.sum()),
            "n_falsified": int(falsified_unv.sum()),
            "pct_falsified": float(falsified_unv.mean()),
            "robust_radii": radii_unv.tolist(),
            "min_radius": float(radii_unv.min()),
            "max_radius": float(radii_unv.max()),
            "mean_radius": float(radii_unv.mean()),
        },
        "clean": {
            "n_total": len(radii_clean),
            "n_verified": int(verified_clean.sum()),
            "n_falsified": int(falsified_clean.sum()),
            "pct_verified": float(verified_clean.mean()),
            "robust_radii": radii_clean.tolist(),
            "min_radius": float(radii_clean.min()),
            "max_radius": float(radii_clean.max()),
            "mean_radius": float(radii_clean.mean()),
        },
    }

    # Include timing data if available
    if "timing" in results_all:
        t = results_all["timing"]
        verification_results["timing"] = {
            "model_load_wall_s": t["model_load_wall_s"],
            "model_load_compile_s": t["model_load_compile_s"],
            "total_hc_wall_s": float(t["instance_wall_s"].sum()),
            "total_hc_compile_s": float(t["instance_compile_s"].sum()),
            "mean_instance_wall_s": float(t["instance_wall_s"].mean()),
            "per_instance_wall_s": t["instance_wall_s"].tolist(),
            "per_instance_compile_s": t["instance_compile_s"].tolist(),
            "n_threads": t["n_threads"],
        }

    if verbose:
        print(f"\nResults:")
        print(f"  Unverifiable instances (expect falsified):")
        print(
            f"    Falsified: {verification_results['unverifiable']['n_falsified']}/{verification_results['unverifiable']['n_total']} ({verification_results['unverifiable']['pct_falsified']:.1%})"
        )
        print(
            f"    Verified (false positive): {verification_results['unverifiable']['n_verified']}/{verification_results['unverifiable']['n_total']}"
        )
        print(
            f"    Robust radius: min={verification_results['unverifiable']['min_radius']:.4f}, max={verification_results['unverifiable']['max_radius']:.4f}"
        )
        print(f"  Clean instances:")
        print(
            f"    Verified: {verification_results['clean']['n_verified']}/{verification_results['clean']['n_total']} ({verification_results['clean']['pct_verified']:.1%})"
        )
        print(
            f"    Falsified: {verification_results['clean']['n_falsified']}/{verification_results['clean']['n_total']}"
        )
        print(
            f"    Robust radius: min={verification_results['clean']['min_radius']:.4f}, max={verification_results['clean']['max_radius']:.4f}"
        )
        print(
            f"\n  Summary: unv_falsified={verification_results['unverifiable']['n_falsified']}, "
            f"unv_verified={verification_results['unverifiable']['n_verified']}, "
            f"clean_verified={verification_results['clean']['n_verified']}, "
            f"clean_falsified={verification_results['clean']['n_falsified']}"
        )

        if "timing" in verification_results:
            t = verification_results["timing"]
            print(f"\n  Timing:")
            print(f"    Julia threads: {t['n_threads']}")
            print(
                f"    Model load: {t['model_load_wall_s']:.3f}s wall, {t['model_load_compile_s']:.3f}s compile"
            )
            print(
                f"    HC total: {t['total_hc_wall_s']:.3f}s wall, {t['total_hc_compile_s']:.3f}s compile"
            )
            print(f"    HC mean/instance: {t['mean_instance_wall_s']:.3f}s")

    # Save verification results
    verify_path = config_dir / "verification_results.json"
    with open(verify_path, "w") as f:
        json.dump(verification_results, f, indent=2)

    return verification_results


def verify_all(
    results_dir: Optional[Path] = None,
    verbose: bool = True,
) -> list:
    """
    Run verification on all benchmark configurations.

    Args:
        results_dir: Base results directory.
        verbose: Print progress.

    Returns:
        List of verification results for each config.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"

    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            "Run 'python -m benchmark.train' first."
        )

    all_results = []

    for config in BENCHMARK_CONFIGS:
        try:
            result = verify_single_config(
                config=config,
                results_dir=results_dir,
                verbose=verbose,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error verifying {config.name}: {e}")
            all_results.append(
                {
                    "config": config.name,
                    "error": str(e),
                }
            )

    # Save summary JSON
    summary_path = results_dir / "verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "num_configs": len(BENCHMARK_CONFIGS),
                "results": all_results,
            },
            f,
            indent=2,
        )

    # Save summary CSV with key statistics for each model
    csv_path = results_dir / "verification_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(
            [
                "config",
                "epsilon",
                "unv_falsified",  # Unverifiable instances correctly falsified
                "unv_verified",  # Unverifiable instances incorrectly verified (false positive)
                "clean_verified",  # Clean instances verified
                "clean_falsified",  # Clean instances falsified
                "model_load_wall_s",
                "model_load_compile_s",
                "total_hc_wall_s",
                "total_hc_compile_s",
                "mean_instance_wall_s",
                "n_threads",
            ]
        )
        # Data rows
        for r in all_results:
            if "error" in r:
                writer.writerow([r["config"]] + [""] * 11)
            else:
                t = r.get("timing", {})
                writer.writerow(
                    [
                        r["config"],
                        r["epsilon"],
                        r["unverifiable"]["n_falsified"],
                        r["unverifiable"]["n_verified"],
                        r["clean"]["n_verified"],
                        r["clean"]["n_falsified"],
                        f"{t['model_load_wall_s']:.3f}" if t else "",
                        f"{t['model_load_compile_s']:.3f}" if t else "",
                        f"{t['total_hc_wall_s']:.3f}" if t else "",
                        f"{t['total_hc_compile_s']:.3f}" if t else "",
                        f"{t['mean_instance_wall_s']:.3f}" if t else "",
                        t.get("n_threads", "") if t else "",
                    ]
                )

    if verbose:
        print(f"\n{'='*60}")
        print("Verification Summary")
        print(f"{'='*60}")

        # Print summary table
        print(
            f"\n{'Config':<30} {'Unv Fals':<10} {'Unv Ver':<10} {'Cln Ver':<10} {'Cln Fals':<10} {'Load(s)':<10} {'HC(s)':<10} {'Compile(s)':<10} {'Threads':<8}"
        )
        print("-" * 118)
        for r in all_results:
            if "error" in r:
                print(f"{r['config']:<30} ERROR: {r['error']}")
            else:
                t = r.get("timing", {})
                line = (
                    f"{r['config']:<30} "
                    f"{r['unverifiable']['n_falsified']:<10} "
                    f"{r['unverifiable']['n_verified']:<10} "
                    f"{r['clean']['n_verified']:<10} "
                    f"{r['clean']['n_falsified']:<10} "
                )
                if t:
                    line += (
                        f"{t['model_load_wall_s']:<10.3f} "
                        f"{t['total_hc_wall_s']:<10.3f} "
                        f"{t['total_hc_compile_s']:<10.3f} "
                        f"{t.get('n_threads', ''):<8}"
                    )
                print(line)

        print(f"\nResults saved to: {results_dir}")
        print(f"CSV summary: {csv_path}")

    return all_results


def print_detailed_results(results_dir: Optional[Path] = None):
    """
    Print detailed verification results from saved summary.

    Args:
        results_dir: Base results directory.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"

    results_dir = Path(results_dir)
    summary_path = results_dir / "verification_summary.json"

    if not summary_path.exists():
        print(f"No verification summary found at {summary_path}")
        print("Run 'python -m benchmark.verify' first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\nVerification Results (from {summary['timestamp']})")
    print("=" * 70)

    for r in summary["results"]:
        if "error" in r:
            continue

        print(f"\n{r['config']}")
        print(f"  Epsilon: {r['epsilon']}")
        print(f"  Unverifiable instances:")
        print(
            f"    Falsified: {r['unverifiable']['n_falsified']}/{r['unverifiable']['n_total']}"
        )
        print(
            f"    Robust radii: {[f'{x:.4f}' for x in r['unverifiable']['robust_radii']]}"
        )
        print(f"  Clean instances:")
        print(f"    Verified: {r['clean']['n_verified']}/{r['clean']['n_total']}")
        print(f"    Robust radii: {[f'{x:.4f}' for x in r['clean']['robust_radii']]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benchmark models")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: benchmark/results)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print detailed summary from previous run (no verification)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else None

    if args.print_summary:
        print_detailed_results(results_dir)
    else:
        verify_all(
            results_dir=results_dir,
            verbose=not args.quiet,
        )

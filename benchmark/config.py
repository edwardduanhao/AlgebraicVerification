"""Benchmark configuration for polynomial neural network verification."""

from dataclasses import dataclass
from typing import List
from itertools import product


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark setting."""

    name: str
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    act_degree: int
    epsilon: float
    homogeneous: bool = True
    bias: bool = True
    n_unverifiable: int = 10
    n_clean: int = 10
    # Counterexample placement: ||delta_cex||_inf in [r * epsilon, epsilon]
    r: float = 0.98

    @property
    def architecture(self) -> List[int]:
        """Return full architecture as [input, hidden..., output]."""
        return [self.input_dim] + self.hidden_dims + [self.output_dim]

    @property
    def n_instances(self) -> int:
        """Total number of instances."""
        return self.n_unverifiable + self.n_clean


def generate_configs() -> List[BenchmarkConfig]:
    """Generate all 8 benchmark configurations."""
    configs = []

    # Fixed parameters
    # input_dim = 8
    input_dim = 12
    output_dim = 2

    # Variable parameters
    # hidden_dims_list = [[6], [10]]  # [8,6,2] and [8,10,2]
    hidden_dims_list = [[8], [16]]
    act_degrees = [2, 3]
    epsilons = [0.2, 0.5]

    for hidden_dims, act_degree, epsilon in product(
        hidden_dims_list, act_degrees, epsilons
    ):
        arch_str = f"{input_dim}_{hidden_dims[0]}_{output_dim}"
        name = f"arch{arch_str}_deg{act_degree}_eps{epsilon}"

        config = BenchmarkConfig(
            name=name,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            act_degree=act_degree,
            epsilon=epsilon,
        )
        configs.append(config)

    return configs


# Pre-generated configurations for easy access
BENCHMARK_CONFIGS = generate_configs()

# Summary
TOTAL_CONFIGS = len(BENCHMARK_CONFIGS)
TOTAL_INSTANCES = sum(c.n_instances for c in BENCHMARK_CONFIGS)


if __name__ == "__main__":
    print(f"Benchmark Configuration Summary")
    print(f"=" * 50)
    print(f"Total configurations: {TOTAL_CONFIGS}")
    print(f"Total instances: {TOTAL_INSTANCES}")
    print()

    for i, config in enumerate(BENCHMARK_CONFIGS, 1):
        print(f"{i}. {config.name}")
        print(f"   Architecture: {config.architecture}")
        print(f"   Activation degree: {config.act_degree}")
        print(f"   Epsilon: {config.epsilon}")
        print(
            f"   Instances: {config.n_unverifiable} unverifiable + {config.n_clean} clean"
        )
        print()

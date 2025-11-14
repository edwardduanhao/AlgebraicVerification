import torch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for Polynomial Neural Network."""

    input_dim: int = 10
    output_dim: int = 2
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    degree: int = 2
    homogeneous: bool = False
    bias: bool = True
    s: float = 1.0


@dataclass
class TrainConfig:
    """Configuration for train process."""

    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9
    optimizer_type: Literal["sgd", "adam", "adamw"] = "adam"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True


@dataclass
class Config:
    """Main configuration containing model and training settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange, tqdm
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.config import TrainConfig


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    Performs a single training epoch over the entire dataset.

    Args:
        model: The neural network model to train
        dataloader: DataLoader containing the training data
        optimizer: The optimizer to use for training
        criterion: Loss function (typically nn.CrossEntropyLoss)
        device: Device to run training on ('cuda' or 'cpu')

    Returns:
        tuple: (average_loss, accuracy) for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    Evaluates the model on a dataset.

    Args:
        model: The neural network model to evaluate
        dataloader: DataLoader containing the evaluation data
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        tuple: (average_loss, accuracy) for the dataset
    """

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_epochs(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: Optional[int] = None,
    optimizer_type: Literal["sgd", "adam", "adamw"] = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    config: Optional["TrainConfig"] = None,
) -> dict:
    """
    Trains a model for multiple epochs with batch-wise training.

    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        num_epochs: Number of epochs to train (required if config is None)
        optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
        learning_rate: Learning rate for the optimizer
        weight_decay: L2 regularization coefficient
        val_loader: Optional DataLoader for validation
        device: Device to run training on ('cuda' or 'cpu')
        verbose: Whether to show progress bars
        config: Optional TrainConfig object. If provided, overrides all other parameters.

    Returns:
        dict: Training history containing losses and accuracies
    """
    # Use config if provided, otherwise use individual parameters
    if config is not None:
        num_epochs = config.num_epochs
        optimizer_type = config.optimizer_type
        learning_rate = config.learning_rate
        weight_decay = config.weight_decay
        device = config.device
        verbose = config.verbose
    elif num_epochs is None:
        raise ValueError("Either num_epochs or config must be provided")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    if optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. Choose 'sgd', 'adam', or 'adamw'."
        )

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop
    epoch_iter = trange(num_epochs, desc="Training", disable=not verbose)
    for epoch in epoch_iter:
        # Training step
        train_loss, train_acc = train_step(
            model, train_loader, optimizer, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation step (if validation loader provided)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if verbose:
                epoch_iter.set_postfix(
                    {
                        "train_loss": f"{train_loss:.4f}",
                        "train_acc": f"{train_acc:.2f}%",
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc:.2f}%",
                    }
                )
        else:
            if verbose:
                epoch_iter.set_postfix(
                    {
                        "train_loss": f"{train_loss:.4f}",
                        "train_acc": f"{train_acc:.2f}%",
                    }
                )

    return history

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange, tqdm
from typing import Optional, Literal, Callable

from src.utils.projected_sgd import projected_sgd_step


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
    num_epochs: int,
    optimizer_type: Literal["sgd", "adam", "adamw"] = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    epoch_callback: Optional[Callable[[nn.Module, int], None]] = None,
) -> dict:
    """
    Trains a model for multiple epochs with batch-wise training.

    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        num_epochs: Number of epochs to train
        optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
        learning_rate: Learning rate for the optimizer
        weight_decay: L2 regularization coefficient
        val_loader: Optional DataLoader for validation
        device: Device to run training on ('cuda' or 'cpu')
        verbose: Whether to show progress bars
        epoch_callback: Optional callable(model, epoch) invoked at the end of each epoch.
            epoch is 1-indexed.

    Returns:
        dict: Training history containing losses and accuracies
    """
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

        if epoch_callback is not None:
            epoch_callback(model, epoch + 1)  # 1-indexed

    return history


def train_epochs_projected(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    constraint_fn: Callable,
    num_epochs: int,
    lr: float = 1e-3,
    n_project_iters: int = 5,
    damping: float = 1e-6,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    epoch_callback: Optional[Callable[[nn.Module, int], None]] = None,
) -> dict:
    """
    Trains a model with projected SGD to satisfy a constraint on the parameters.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        constraint_fn: Callable(theta_flat) -> scalar Tensor encoding the constraint
            to enforce (constraint_fn = 0). Typically built via
            model.make_det_A_constraint().
        num_epochs: Number of training epochs.
        lr: Learning rate for the gradient step.
        n_project_iters: Number of Newton-style projection iterations per step.
        damping: Tikhonov damping for the projection solve.
        val_loader: Optional DataLoader for validation.
        device: Device to run training on.
        verbose: Whether to show a progress bar.
        epoch_callback: Optional callable(model, epoch) invoked at the end of each epoch.
            epoch is 1-indexed.

    Returns:
        dict: Training history with keys 'train_loss', 'train_acc', 'val_loss',
              'val_acc', and 'constraint_norm'.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    def loss_fn(m, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        return criterion(m(inputs), targets)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "constraint_norm": [],
    }

    epoch_iter = trange(num_epochs, desc="Training (projected)", disable=not verbose)
    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        constraint_norms = []

        for inputs, targets in train_loader:
            info = projected_sgd_step(
                model,
                loss_fn,
                (inputs, targets),
                constraint_fn,
                lr=lr,
                n_project_iters=n_project_iters,
                damping=damping,
            )
            total_loss += info["loss"]
            constraint_norms.append(info["constraint_norm_after"])

            # Accuracy on this batch using the updated parameters
            with torch.no_grad():
                outputs = model(inputs.to(device))
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets.to(device)).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        constraint_norm = sum(constraint_norms) / len(constraint_norms)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["constraint_norm"].append(constraint_norm)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if verbose:
            postfix = {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.2f}%",
                "constraint": f"{constraint_norm:.2e}",
            }
            if val_loader is not None:
                postfix["val_loss"] = f"{val_loss:.4f}"
                postfix["val_acc"] = f"{val_acc:.2f}%"
            epoch_iter.set_postfix(postfix)

        if epoch_callback is not None:
            epoch_callback(model, epoch + 1)  # 1-indexed

    return history

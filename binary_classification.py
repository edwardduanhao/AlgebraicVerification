import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import PolynomialNetwork
from nnexpansion import polynomial_nn_expansion, compute_class_differences


def generate_sin_boundary_data(n_samples=1000, noise_level=0.1, x_range=(-2*np.pi, 2*np.pi)):
    """
    Generate binary classification data with sin(x) decision boundary.
    
    Args:
        n_samples: Number of data points to generate
        noise_level: Amount of noise to add to the boundary
        x_range: Range of x values to sample from
    
    Returns:
        X: Input features of shape (n_samples, 2)
        y: Binary labels of shape (n_samples,)
    """
    # Generate random x values
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    
    # Generate random y values around the sin(x) boundary
    y_boundary = np.sin(x)
    
    # Add some spread around the boundary
    y_spread = 2.0  # How far above/below the boundary to sample
    y = np.random.uniform(y_boundary - y_spread, y_boundary + y_spread, n_samples)
    
    # Add noise to the boundary itself
    noisy_boundary = y_boundary + np.random.normal(0, noise_level, n_samples)
    
    # Create binary labels: 1 if above noisy boundary, 0 if below
    labels = (y > noisy_boundary).astype(np.int64)
    
    # Stack features
    X = np.stack([x, y], axis=1).astype(np.float32)
    
    return torch.tensor(X), torch.tensor(labels)


def train_model(model, X_train, y_train, epochs=1000, lr=0.01):
    """
    Train the polynomial network for binary classification.
    
    Args:
        model: PolynomialNetwork instance
        X_train: Training features
        y_train: Training labels
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        losses: List of training losses
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses


def visualize_decision_boundary(model, X, y, x_range=(-2*np.pi, 2*np.pi), y_range=(-3, 3)):
    """
    Visualize the learned decision boundary and compare with true sin(x) boundary.
    
    Args:
        model: Trained PolynomialNetwork
        X: Training data features
        y: Training data labels
        x_range: Range for x-axis
        y_range: Range for y-axis
    """
    plt.figure(figsize=(12, 8))
    
    # Create a grid for plotting decision boundary
    x_grid = np.linspace(x_range[0], x_range[1], 200)
    y_grid = np.linspace(y_range[0], y_range[1], 200)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Flatten grid for prediction
    grid_points = torch.tensor(np.c_[X_grid.ravel(), Y_grid.ravel()], dtype=torch.float32)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        logits = model(grid_points)
        predictions = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1
    
    # Reshape predictions back to grid
    Z = predictions.numpy().reshape(X_grid.shape)
    
    # Plot decision boundary as contour
    plt.contour(X_grid, Y_grid, Z, levels=[0.5], colors='red', linewidths=2)
    plt.contourf(X_grid, Y_grid, Z, levels=50, alpha=0.3, cmap='RdYlBu')
    
    # Plot true sin(x) boundary
    x_true = np.linspace(x_range[0], x_range[1], 1000)
    y_true = np.sin(x_true)
    plt.plot(x_true, y_true, 'g--', linewidth=2, label='True sin(x) Boundary')
    
    # Plot training data
    X_np = X.numpy()
    y_np = y.numpy()
    colors = ['blue' if label == 0 else 'orange' for label in y_np]
    plt.scatter(X_np[:, 0], X_np[:, 1], c=colors, alpha=0.6, s=20)
    
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Network Decision Boundary vs True sin(x) Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.colorbar(label='Prediction Probability')
    plt.show()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate training data
    print("Generating training data with sin(x) decision boundary...")
    X_train, y_train = generate_sin_boundary_data(n_samples=2000, noise_level=0.1)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Class distribution: {torch.bincount(y_train.long())}")
    
    # Create polynomial network
    model = PolynomialNetwork(
        input_dim=2,
        output_dim=2,  # Binary classification with 2 outputs
        hidden_dims=[8, 6],
        polynomial_degree=3
    )
    
    print(f"\nModel architecture:")
    print(f"Input dim: {model.input_dim}")
    print(f"Hidden dims: {model.hidden_dims}")
    print(f"Output dim: {model.output_dim}")
    print(f"Polynomial degree: {model.polynomial_degree}")
    
    # Train the model
    print("\nTraining the model...")
    losses = train_model(model, X_train, y_train, epochs=3000, lr=1e-3)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Evaluate model accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_train)
        predicted_labels = torch.argmax(logits, dim=1)
        accuracy = (predicted_labels == y_train).float().mean()
        print(f"\nTraining Accuracy: {accuracy:.4f}")
    
    # Visualize decision boundary
    print("\nVisualizing decision boundary...")
    visualize_decision_boundary(model, X_train, y_train, x_range=(-4*np.pi, 4*np.pi), y_range=(-10, 10))
    
    # Get polynomial expansion
    print("\nComputing polynomial expansion...")
    monoms, C = polynomial_nn_expansion(model)
    print(f"Number of monomials: {len(monoms)}")
    print(f"Coefficient matrix shape: {C.shape}")
    
    # Since this is binary classification with 2 outputs, we can use compute_class_differences
    print(f"\nFirst few monomials: {monoms[:10]}")
    print(f"Coefficients for class 0: {C[0, :5]}")
    print(f"Coefficients for class 1: {C[1, :5]}")
    
    # Compute class differences for verification
    print("\nComputing class differences for verification:")
    for gold_class in [0, 1]:
        differences = compute_class_differences(C, gold_class)
        print(f"Gold class {gold_class}: differences shape {differences.shape}")
        print(f"  First few coefficients: {differences[0, :5]}")
    
    return model, X_train, y_train, monoms, C


if __name__ == "__main__":
    model, X_train, y_train, monoms, C = main()
    print(C.shape)
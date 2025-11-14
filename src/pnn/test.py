import torch
from src.pnn import (
    PolynomialNeuralNetwork,
    ComplexPolynomialNeuralNetwork,
    c_split,
    c_join,
)
from src.config import Config

if __name__ == "__main__":
    # Create a sample PolynomialNeuralNetwork
    pnn = PolynomialNeuralNetwork(
        input_dim=4,
        output_dim=2,
        hidden_dims=[8, 8],
        degree=3,
        homogeneous=False,
        bias=True,
        s=1.0,
    )

    # config = Config()
    # pnn = PolynomialNeuralNetwork.from_config(config.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pnn = pnn.double().to(device)

    # Convert to ComplexPolynomialNeuralNetwork
    cpnn = ComplexPolynomialNeuralNetwork.from_polynomial_neural_network(pnn)

    # Verify dtypes are preserved
    print(f"PNN dtype: {next(pnn.parameters()).dtype}")
    print(f"CPNN dtype: {next(cpnn.parameters()).dtype}")
    print(f"PNN device: {next(pnn.parameters()).device}")
    print(f"CPNN device: {next(cpnn.parameters()).device}")

    # Test with random input (must match model dtype)
    x_real = torch.randn(100, 4, dtype=torch.float64, device=device)
    x_imag = torch.zeros_like(x_real)
    x_c = c_join(x_real, x_imag)

    with torch.no_grad():
        y_pnn = pnn(x_real)
        y_cpnn = cpnn(x_c)
        y_cpnn_real, y_cpnn_imag = c_split(y_cpnn)

    print("PNN output:", y_pnn[:5])
    print("CPNN output (real part):", y_cpnn_real[:5])
    print("CPNN output (imag part):", y_cpnn_imag[:5])

    assert torch.allclose(y_pnn, y_cpnn_real, atol=1e-5), "Outputs do not match!"
    assert torch.allclose(
        y_cpnn_imag, torch.zeros_like(y_cpnn_imag), atol=1e-5
    ), "Imaginary part is not zero!"

    print("\n✓ All tests passed!")

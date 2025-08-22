import torch
import torch.nn as nn
import torch.nn.functional as F


class PolynomialActivation(nn.Module):
    def __init__(self, degree=3):
        super(PolynomialActivation, self).__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(torch.randn(degree + 1) * 0.1)
        
    def forward(self, x):
        result = self.coefficients[0] * torch.ones_like(x)
        x_power = torch.ones_like(x)
        for i in range(1, self.degree + 1):
            x_power = x_power * x
            result += self.coefficients[i] * x_power
        return result


class PolynomialNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        polynomial_degree=3
    ):
        super(PolynomialNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.polynomial_degree = polynomial_degree
        
        layers = []
        activations = []
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                activations.append(PolynomialActivation(polynomial_degree))
        
        self.layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(activations)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activations[i](x)
        x = self.layers[-1](x)
        return x

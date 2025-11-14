import torch
import torch.nn as nn


class LinearSelfAttention(nn.Module):
    """
    A linear self-attention block.
    """

    def __init__(self, embed_dim: int) -> None:
        """

        Args:
            embed_dim (int): Embedding dimension of the input tensor.
        """

        super().__init__()

        self.embed_dim = embed_dim

        # Define query, key, value linear layers
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embed_dim, seq_len).
        """

        if x.ndim != 3 or x.shape[1] != self.embed_dim:
            raise ValueError(
                f"Expected input shape (batch_size, {self.embed_dim}, seq_len), got {x.shape}"
            )

        batch_size, embed_dim, seq_len = x.shape

        # Transpose to (batch_size, seq_len, embed_dim) for linear layers
        x_t = x.transpose(1, 2)  # [batch_size, seq_len, embed_dim]

        # Compute Q, K, V
        Q = self.query(x_t)  # [batch_size, seq_len, embed_dim]
        K = self.key(x_t)  # [batch_size, seq_len, embed_dim]
        V = self.value(x_t)  # [batch_size, seq_len, embed_dim]

        # Compute scaled dot-product attention
        # QK^T shape: [batch_size, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(1, 2))

        # Multiply attention weights by V
        out = torch.matmul(scores, V)  # [batch_size, seq_len, embed_dim]

        # Reshape back to (batch_size, embed_dim, seq_len)
        out = out.transpose(1, 2)

        return out

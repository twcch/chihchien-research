import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Transformer Encoder 層包含一個多頭自注意力機制和一個位置前饋神經網絡"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 多頭自注意力機制
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 位置前饋神經網絡
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=src_mask,
        )

        x = self.norm1(x + self.dropout(attention_output))

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(x + self.dropout(feed_forward_output))

        return x

class Encoder(nn.Module):
    """Transformer Encoder"""
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )

            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)

        return x

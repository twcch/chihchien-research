import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer, consisting of self-attention, cross-attention, and feed-forward sublayers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attention_output = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask,
        )

        x = self.norm1(x + self.dropout(self_attention_output))

        cross_attention_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
        )

        x = self.norm2(x + self.dropout(cross_attention_output))

        feed_forward_output = self.feed_forward(x)

        x = self.norm3(x + self.dropout(feed_forward_output))

        return x


class Decoder(nn.Module):
    """
    Transformer decoder stack.
    """
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
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )

        return x
import math
import torch
import torch.nn as nn

from .config import TransformerConfig
from .positional_encoding import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder


class TransformerSeq2Seq(nn.Module):
    """
    Full Transformer encoder-decoder model for sequence-to-sequence translation.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.src_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_id,
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_id,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_len,
            dropout=config.dropout,
        )

        self.encoder = Encoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        self.decoder = Decoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        self.output_projection = nn.Linear(
            config.d_model,
            config.vocab_size,
        )

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        src shape:
            [batch_size, src_len]

        output shape:
            [batch_size, 1, 1, src_len]
        """
        return (src != self.config.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        tgt shape:
            [batch_size, tgt_len]

        output shape:
            [batch_size, 1, tgt_len, tgt_len]
        """
        batch_size, tgt_len = tgt.size()

        pad_mask = (tgt != self.config.pad_id).unsqueeze(1).unsqueeze(2)

        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.make_src_mask(src)

        x = self.src_embedding(src) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)

        encoder_output = self.encoder(x, src_mask)

        return encoder_output, src_mask

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_mask = self.make_tgt_mask(tgt)

        x = self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)

        decoder_output = self.decoder(
            x=x,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        logits = self.output_projection(decoder_output)

        return logits

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        """前向傳播函數，將輸入的源序列和目標序列（去掉最後一個 token）傳遞給編碼器和解碼器，並返回解碼器的輸出 logits。"""
        
        # 首先，對源序列進行編碼，得到編碼器的輸出和源序列的遮罩。然後，將目標序列 (去掉最後一個 token) 傳遞給解碼器，並使用編碼器的輸出和源序列的遮罩來計算解碼器的輸出 logits
        encoder_output, src_mask = self.encode(src)

        # 最後，返回解碼器的輸出 logits，這些 logits 可以用於計算損失或進行推理
        logits = self.decode(
            tgt=tgt_input,
            encoder_output=encoder_output,
            src_mask=src_mask,
        )

        return logits
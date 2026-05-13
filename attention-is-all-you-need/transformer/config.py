import torch
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int # vocab_size 為詞表大小，表示模型可以處理不同單詞量的句子
    pad_id: int # pad_id 用於填充序列的特殊標記 ID
    max_len: int = 128 # max_len 為模型能夠處理的最大長度，超過長度的序列會被截斷，不足長度的序列會被填充
    d_model: int = 256  # d_model 為模型的隱藏層維度，表示每個位置的詞向量的維度
    num_heads: int = 4 # num_heads 為多頭注意力機制的數量，表示模型在每個位置上可以同時關注多個不同的子空間，以捕捉不同的語義信息
    num_layers: int = 3 # num_layers 是 Transformer 模型中 encoder 和 decoder 的層數，表示模型的深度
    d_ff: int = 1024 # d_ff 為前饋神經網路的隱藏層維度，表示在每個 encoder 和 decoder 層中前饋網絡的大小
    dropout: float = 0.1 # dropout 表示在每個訓練步驟中隨機丟棄神經元的比例，以防止過擬合

    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

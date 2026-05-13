import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """位置前饋神經網絡類別，實現了 Transformer 模型中的位置前饋神經網絡。"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        # 定義前饋神經網絡的結構，包括兩個線性層和一個 ReLU 激活函數，以及一個 Dropout 層，用於在訓練過程中隨機丟棄神經元，以防止過擬合。
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

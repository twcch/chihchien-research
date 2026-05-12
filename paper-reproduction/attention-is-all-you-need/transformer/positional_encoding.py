import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    位置編碼類別，將位置信息添加到詞向量中，以便模型能夠捕捉序列中單詞的順序信息。
    位置編碼使用正弦和餘弦函數來生成不同位置的編碼，這些編碼在訓練過程中是固定的，不會被更新。
    """
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 創建位置編碼矩陣，大小為 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # position 是一個形狀為 (max_len, 1) 的張量，包含從 0 到 max_len-1 的位置索引
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term 是一個形狀為 (d_model/2,) 的張量，用於計算正弦和餘弦函數的頻率，根據公式 10000^(2i/d_model) 計算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # pe 的偶數列使用正弦函數計算，奇數列使用餘弦函數計算，根據位置和頻率生成位置編碼
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # 將位置編碼矩陣擴展一個維度，使其形狀變為 (1, max_len, d_model)，以便在前向傳播中與輸入的詞向量相加
        pe = pe.unsqueeze(0)
        
        # 將位置編碼矩陣註冊為模型的緩衝區，這樣它就不會被更新，但在前向傳播中可以使用
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形狀為 (batch_size, seq_len, d_model)，其中 batch_size 是批次大小，seq_len 是序列長度，d_model 是詞向量的維度
        seq_len = x.size(1)
        # 將位置編碼添加到輸入的詞向量中，根據序列長度從位置編碼矩陣中選取對應的部分，並將其與輸入的詞向量相加
        x = x + self.pe[:, :seq_len]
        
        return self.dropout(x)    
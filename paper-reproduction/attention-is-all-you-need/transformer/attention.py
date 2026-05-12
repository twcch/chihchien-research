import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    多頭注意力類別，實現了 Transformer 模型中的多頭注意力機制。
    多頭注意力允許模型在每個位置上同時關注多個不同的子空間，以捕捉不同的語義信息。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError("d_model 必須能被 num_heads 整除")
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.num_heads = num_heads

        # 每個頭的維度
        self.d_k = d_model // num_heads
        
        # 定義線性層，用於將輸入的詞向量映射到查詢、鍵和值的空間
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 定義線性層，用於將多頭注意力的輸出映射回原始維度
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """將輸入的詞向量分割成多個頭，並將其形狀變為 (batch_size, num_heads, seq_len, d_k)，以便在注意力計算中使用"""
        batch_size, seq_len, _ = x.size()
        
        # 將輸入的詞向量通過線性層映射到查詢、鍵和值的空間，並將其形狀變為 (batch_size, num_heads, seq_len, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """將多個頭的輸出合併回原始維度，將其形狀變為 (batch_size, seq_len, d_model)，以便在後續的前饋神經網絡中使用"""
        batch_size, num_heads, seq_len, d_k = x.size()
        
        # 將多個頭的輸出合併回原始維度，將其形狀變為 (batch_size, seq_len, num_heads * d_k) = (batch_size, seq_len, d_model)，以便在後續的前饋神經網絡中使用
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)
        
        return x
    
    
    def forward(self, query, key, value, mask=None):
        q = self.split_heads(self.w_q(query))
        k = self.split_heads(self.w_k(key))
        v = self.split_heads(self.w_v(value))
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        
        context = self.combine_heads(context)
        
        output = self.w_o(context)
        
        return output
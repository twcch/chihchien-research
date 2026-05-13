import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # 移除第二個維度，調整為 [max_len, d_model] 
        pe = pe.squeeze(1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        # 將對應長度的位置編碼加到輸入上
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class AttentionIsAllYouNeedTransformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. 建立 Embedding 層
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 2. 建立位置編碼 (Positional Encoding)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. 建立 Encoder 組件 (使用 PyTorch 內建函式庫)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # 設定 batch_first=True 以符合 (batch, seq, feature) 習慣
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. 建立 Decoder 組件 (使用 PyTorch 內建函式庫)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 5. 建立最後的線性輸出層
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def generate_causal_mask(self, sz):
        """
        生成 Decoder 需要的 causal mask (遮罩)，防止模型在自迴歸預測時偷看未來的 Token。
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        """
        src: [batch_size, src_seq_len]
        tgt: [batch_size, tgt_seq_len]
        src_padding_mask / tgt_padding_mask: [batch_size, seq_len] 的布林張量 (True 表示是 padding)
        """
        # 生成 Decoder 的 Causal Mask
        tgt_seq_len = tgt.size(1)
        tgt_causal_mask = self.generate_causal_mask(tgt_seq_len).to(src.device)
        
        # --- Encoder 處理流程 ---
        # 論文指出 Embedding 輸出要先乘上 sqrt(d_model) 再加上位置編碼
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        
        # src_key_padding_mask 用來忽略 <PAD> token，不讓其參與 Attention 計算
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # --- Decoder 處理流程 ---
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # 傳入 Decoder：
        # tgt_mask: 確保自迴歸屬性 (Causal Mask)
        # memory_key_padding_mask: 確保 cross-attention 不會去關注 Encoder 端的 <PAD> token
        output = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_causal_mask, 
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # --- 輸出映射 ---
        logits = self.fc_out(output)
        return logits

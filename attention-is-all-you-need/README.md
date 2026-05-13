# Attention Is All You Need Reproduction

這個資料夾是對 Vaswani et al. 2017 論文 **Attention Is All You Need** 的 PyTorch reproduction。目標不是完整重現原論文的大規模實驗，而是用較小的模型與中英翻譯任務，把 Transformer encoder-decoder 的核心元件從頭實作一次，方便閱讀、訓練與推論。

目前任務方向是 **中文 -> 英文** 翻譯，資料來源為 Hugging Face `Helsinki-NLP/opus-100` 的 `en-zh` 平行語料。

## 專案內容

```text
attention-is-all-you-need/
├── README.md
├── train_transformer_zh_en.py       # 訓練 tokenizer 與 Transformer 翻譯模型
├── infer_translate.py              # 載入 checkpoint 做中文到英文推論
├── bpe_zh_en.json                  # 已訓練好的 BPE tokenizer
├── transformer_zh_en.pt            # 已訓練好的模型權重
└── transformer/
    ├── attention.py                # Scaled dot-product multi-head attention
    ├── config.py                   # TransformerConfig
    ├── decoder.py                  # Decoder layer 與 decoder stack
    ├── encoder.py                  # Encoder layer 與 encoder stack
    ├── feed_forward.py             # Position-wise feed-forward network
    ├── model.py                    # TransformerSeq2Seq 主模型
    └── positional_encoding.py      # Sinusoidal positional encoding
```

## 實作重點

- **Multi-head attention**：將 `query`、`key`、`value` 投影後切成多個 head，計算 scaled dot-product attention，再合併回 `d_model`。
- **Encoder**：每層包含 self-attention、residual connection、LayerNorm 與 position-wise feed-forward network。
- **Decoder**：每層包含 masked self-attention、encoder-decoder cross-attention、feed-forward network 與三組 residual/LayerNorm。
- **Masking**：
  - source mask 會忽略 `[PAD]` token。
  - target mask 同時處理 `[PAD]` 與 causal mask，避免 decoder 看到未來 token。
- **Positional encoding**：使用原論文的 sinusoidal positional encoding，讓模型取得序列位置資訊。
- **Seq2Seq wrapper**：`TransformerSeq2Seq` 封裝 embedding、position encoding、encoder、decoder 與輸出投影層。

## 環境需求

建議使用 Python 3.10 以上。

```bash
pip install torch datasets tokenizers tqdm
```

如果你要使用 GPU，請依照自己的 CUDA / PyTorch 版本安裝對應的 `torch` 套件。

## 快速推論

資料夾內已包含：

- `bpe_zh_en.json`
- `transformer_zh_en.pt`

因此可以直接執行推論腳本：

```bash
cd paper-reproduction/attention-is-all-you-need
python infer_translate.py
```

`infer_translate.py` 預設會翻譯：

```text
我喜歡學習人工智慧。
```

如果要更換輸入句子，可以修改 `infer_translate.py` 內的 `input_sentence`。

## 重新訓練

執行：

```bash
cd paper-reproduction/attention-is-all-you-need
python train_transformer_zh_en.py
```

訓練流程會：

1. 從 Hugging Face 載入 `Helsinki-NLP/opus-100` 的 `en-zh` split。
2. 擷取中文與英文句對，預設最多使用 30,000 筆訓練資料與 1,000 筆驗證資料。
3. 若 `bpe_zh_en.json` 不存在，會用訓練資料建立 BPE tokenizer。
4. 建立 Transformer encoder-decoder 模型。
5. 使用 teacher forcing 訓練中文到英文翻譯。
6. 以 validation loss 追蹤最佳模型，並儲存到 `transformer_zh_en.pt`。

主要訓練設定在 `TrainingConfig`：

| 參數 | 預設值 |
| --- | --- |
| `vocab_size` | `16000` |
| `max_len` | `64` |
| `batch_size` | `64` |
| `epochs` | `8` |
| `lr` | `3e-4` |
| `d_model` | `256` |
| `num_heads` | `4` |
| `num_layers` | `3` |
| `d_ff` | `1024` |
| `dropout` | `0.1` |

## 模型架構

本 reproduction 採用較小的 Transformer 設定：

```text
source sentence
  -> source embedding
  -> sinusoidal positional encoding
  -> encoder stack
  -> decoder stack with target input
  -> linear projection to vocabulary logits
```

訓練時，target 會被切成：

- `tgt_input = tgt[:, :-1]`
- `tgt_output = tgt[:, 1:]`

也就是 decoder 看到前面的英文 token，學習預測下一個英文 token。loss 使用 `CrossEntropyLoss(ignore_index=pad_id)`，因此 `[PAD]` 不會影響 loss。

推論時使用 greedy decoding：從 `[BOS]` 開始，每一步選擇機率最高的 token，直到產生 `[EOS]` 或達到最大長度。

## 與原論文的差異

這份實作保留 Transformer 的核心結構，但為了讓實驗更容易在個人電腦上執行，做了幾個簡化：

- 模型比原論文小：預設 `d_model=256`、`num_layers=3`、`num_heads=4`。
- 訓練資料量較小：預設只取 30,000 筆訓練句對。
- 沒有實作原論文的 learning rate warmup / inverse square root schedule。
- 沒有 label smoothing。
- 推論使用 greedy decoding，沒有 beam search。
- 目前沒有計算 BLEU，只以 validation loss 與範例翻譯觀察訓練效果。

## 可以延伸的方向

- 加入 Noam learning rate scheduler，貼近原論文訓練設定。
- 加入 label smoothing。
- 實作 beam search decoding。
- 加入 BLEU / chrF 等翻譯品質評估。
- 增加資料量、模型層數與 `d_model`，觀察翻譯品質變化。
- 將 tokenizer、訓練設定與推論輸入改成 CLI 參數。

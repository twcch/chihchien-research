import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

from transformer import TransformerConfig, TransformerSeq2Seq


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainingConfig:
    dataset_name: str = "Helsinki-NLP/opus-100"
    lang_pair: str = "en-zh"

    max_train_samples: int = 30000
    max_valid_samples: int = 1000

    tokenizer_path: str = str(BASE_DIR / "bpe_zh_en.json")
    model_path: str = str(BASE_DIR / "transformer_zh_en.pt")

    vocab_size: int = 16000
    max_len: int = 64

    batch_size: int = 64
    epochs: int = 8
    lr: float = 3e-4

    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1

    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"

    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


cfg = TrainingConfig()


def load_parallel_data(split: str, max_samples: int):
    dataset = load_dataset(
        cfg.dataset_name,
        cfg.lang_pair,
        split=split,
    )

    pairs = []

    for example in dataset:
        translation = example["translation"]

        zh = translation.get("zh")
        en = translation.get("en")

        if zh is None or en is None:
            continue

        zh = zh.strip()
        en = en.strip()

        if len(zh) == 0 or len(en) == 0:
            continue

        pairs.append((zh, en))

        if len(pairs) >= max_samples:
            break

    return pairs


def train_or_load_tokenizer(train_pairs):
    if os.path.exists(cfg.tokenizer_path):
        return Tokenizer.from_file(cfg.tokenizer_path)

    tokenizer = Tokenizer(BPE(unk_token=cfg.unk_token))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=[
            cfg.pad_token,
            cfg.unk_token,
            cfg.bos_token,
            cfg.eos_token,
        ],
    )

    def batch_iterator():
        for zh, en in train_pairs:
            yield zh
            yield en

    tokenizer.train_from_iterator(
        batch_iterator(),
        trainer=trainer,
    )

    tokenizer.post_processor = TemplateProcessing(
        single=f"{cfg.bos_token} $A {cfg.eos_token}",
        special_tokens=[
            (cfg.bos_token, tokenizer.token_to_id(cfg.bos_token)),
            (cfg.eos_token, tokenizer.token_to_id(cfg.eos_token)),
        ],
    )

    tokenizer.save(cfg.tokenizer_path)

    return tokenizer


class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer: Tokenizer, max_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.pad_id = tokenizer.token_to_id(cfg.pad_token)
        self.eos_id = tokenizer.token_to_id(cfg.eos_token)

    def __len__(self):
        return len(self.pairs)

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text).ids

        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + [self.eos_id]

        padding_len = self.max_len - len(ids)
        ids = ids + [self.pad_id] * padding_len

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        zh, en = self.pairs[idx]

        src = self.encode(zh)
        tgt = self.encode(en)

        return src, tgt


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0

    for src, tgt in tqdm(dataloader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0

    for src, tgt in tqdm(dataloader, desc="Validation"):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def translate(
    model,
    tokenizer: Tokenizer,
    sentence: str,
    max_len: int = 64,
):
    model.eval()

    pad_id = tokenizer.token_to_id(cfg.pad_token)
    bos_id = tokenizer.token_to_id(cfg.bos_token)
    eos_id = tokenizer.token_to_id(cfg.eos_token)

    src_ids = tokenizer.encode(sentence).ids

    if len(src_ids) > cfg.max_len:
        src_ids = src_ids[: cfg.max_len - 1] + [eos_id]

    src_ids = src_ids + [pad_id] * (cfg.max_len - len(src_ids))

    src = torch.tensor(
        src_ids,
        dtype=torch.long,
    ).unsqueeze(0).to(cfg.device)

    encoder_output, src_mask = model.encode(src)

    tgt_ids = [bos_id]

    for _ in range(max_len):
        tgt = torch.tensor(
            tgt_ids,
            dtype=torch.long,
        ).unsqueeze(0).to(cfg.device)

        logits = model.decode(
            tgt=tgt,
            encoder_output=encoder_output,
            src_mask=src_mask,
        )

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        if next_token_id == eos_id:
            break

        tgt_ids.append(next_token_id)

    return tokenizer.decode(
        tgt_ids,
        skip_special_tokens=True,
    )


def build_model(tokenizer: Tokenizer):
    pad_id = tokenizer.token_to_id(cfg.pad_token)
    vocab_size = tokenizer.get_vocab_size()

    model_config = TransformerConfig(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        device=cfg.device,
    )

    model = TransformerSeq2Seq(model_config)

    return model.to(cfg.device)


def main():
    print(f"Device: {cfg.device}")

    print("Loading dataset...")
    train_pairs = load_parallel_data(
        split="train",
        max_samples=cfg.max_train_samples,
    )

    valid_pairs = load_parallel_data(
        split="validation",
        max_samples=cfg.max_valid_samples,
    )

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Valid pairs: {len(valid_pairs)}")

    print("Training or loading tokenizer...")
    tokenizer = train_or_load_tokenizer(train_pairs)

    pad_id = tokenizer.token_to_id(cfg.pad_token)

    train_dataset = TranslationDataset(
        pairs=train_pairs,
        tokenizer=tokenizer,
        max_len=cfg.max_len,
    )

    valid_dataset = TranslationDataset(
        pairs=valid_pairs,
        tokenizer=tokenizer,
        max_len=cfg.max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
    )

    print("Building model...")
    model = build_model(tokenizer)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
    )

    best_valid_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg.device,
        )

        valid_loss = evaluate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=cfg.device,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                model.state_dict(),
                cfg.model_path,
            )

            print(f"Saved best model to {cfg.model_path}")

        examples = [
            "我喜歡學習人工智慧。",
            "這是一個很好的機器翻譯範例。",
            "今天天氣很好。",
        ]

        print("\nSample translations:")

        for sentence in examples:
            prediction = translate(
                model=model,
                tokenizer=tokenizer,
                sentence=sentence,
                max_len=cfg.max_len,
            )

            print(f"ZH: {sentence}")
            print(f"EN: {prediction}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()

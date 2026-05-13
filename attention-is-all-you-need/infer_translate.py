import torch
from dataclasses import dataclass
from pathlib import Path
from tokenizers import Tokenizer

from transformer import TransformerConfig, TransformerSeq2Seq


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class InferenceConfig:
    tokenizer_path: str = str(BASE_DIR / "bpe_zh_en.json")
    model_path: str = str(BASE_DIR / "transformer_zh_en.pt")

    max_len: int = 64

    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1

    pad_token: str = "[PAD]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = InferenceConfig()


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def build_model(tokenizer: Tokenizer) -> TransformerSeq2Seq:
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


def load_trained_model(
    model: TransformerSeq2Seq,
    model_path: str,
) -> TransformerSeq2Seq:
    state_dict = torch.load(
        model_path,
        map_location=cfg.device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def translate(
    model: TransformerSeq2Seq,
    tokenizer: Tokenizer,
    sentence: str,
    max_len: int = 64,
) -> str:
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

    translation = tokenizer.decode(
        tgt_ids,
        skip_special_tokens=True,
    )

    return translation


def main():
    input_sentence = "I am a student."  # 這裡可以替換成任意中文句子進行翻譯

    print(f"Device: {cfg.device}")
    print(f"Input Chinese: {input_sentence}")

    tokenizer = load_tokenizer(cfg.tokenizer_path)

    model = build_model(tokenizer)

    model = load_trained_model(
        model=model,
        model_path=cfg.model_path,
    )

    output_sentence = translate(
        model=model,
        tokenizer=tokenizer,
        sentence=input_sentence,
        max_len=cfg.max_len,
    )

    print(f"Output English: {output_sentence}")


if __name__ == "__main__":
    main()

"""
DSM-ASR Dataset v5 — SODA-Style Interleaved Format

Builds flat 1D token sequences:
    <|audio_start|> [audio_tok × T*Q] <|audio_end|> <|text_start|> [text_toks] <|text_end|>

Labels:
    -100 everywhere except text_toks and final <|text_end|>
    (only predict text, learning ASR)
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


class DsmAsrDataset(Dataset):
    def __init__(self, config: DsmAsrConfig, split="train",
                 max_samples=None, tokenizer=None):
        super().__init__()
        self.config = config

        # Tokenizer with audio special tokens added
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.qwen_model, trust_remote_code=True)
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": config.special_tokens})

        # Token IDs for boundaries
        self.audio_start = self.tokenizer.convert_tokens_to_ids(config.audio_start_token)
        self.audio_end   = self.tokenizer.convert_tokens_to_ids(config.audio_end_token)
        self.text_start  = self.tokenizer.convert_tokens_to_ids(config.text_start_token)
        self.text_end    = self.tokenizer.convert_tokens_to_ids(config.text_end_token)
        self.eos         = self.tokenizer.eos_token_id

        # Load manifest
        manifest_path = Path(config.preprocessed_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Run prepare_data.py first: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.samples = manifest["samples"]

        if split == "train" and config.eval_split is None:
            n_eval = max(1, int(len(self.samples) * config.eval_ratio))
            self.samples = self.samples[:-n_eval]
        elif split == "eval" and config.eval_split is None:
            n_eval = max(1, int(len(self.samples) * config.eval_ratio))
            self.samples = self.samples[-n_eval:]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        print(f"📂 DsmAsrDataset [{split}]: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        info = self.samples[idx]
        data = np.load(info["path"], allow_pickle=True)

        # Shift raw flat ids (0..16383) into vocabulary range
        audio_vocab_ids = raw_flat + self.config.audio_token_offset

        # Safety check — catch OOB on CPU before it reaches the GPU
        max_id = audio_vocab_ids.max() if len(audio_vocab_ids) > 0 else 0
        expected_max = self.config.total_vocab_size - 1
        assert max_id <= expected_max, (
            f"Audio token ID {max_id} exceeds vocab size {self.config.total_vocab_size}. "
            f"Check config: offset={self.config.audio_token_offset}, "
            f"audio_vocab={self.config.audio_vocab_size}")

        # Tokenize text
        text = str(data["text"]).strip()
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.config.max_text_tokens:
            text_ids = text_ids[:self.config.max_text_tokens]

        # Build full sequence:
        #   [audio_start, *audio_vocab_ids, audio_end, text_start, *text_ids, text_end]
        input_ids = (
            [self.audio_start] +
            audio_vocab_ids.tolist() +
            [self.audio_end, self.text_start] +
            text_ids +
            [self.text_end]
        )

        # Labels: -100 for everything except text_ids and text_end
        #   Position of text tokens = 1 + len(audio) + 2 for audio_end, text_start
        audio_block_len = 1 + len(audio_vocab_ids) + 2  # audio_start + audio + audio_end + text_start
        labels = [-100] * audio_block_len + text_ids + [self.text_end]

        assert len(input_ids) == len(labels), f"{len(input_ids)} != {len(labels)}"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


if __name__ == "__main__":
    config = DsmAsrConfig()
    try:
        ds = DsmAsrDataset(config, split="train", max_samples=3)
        if len(ds) == 0:
            print("⚠️  No samples loaded")
        else:
            s = ds[0]
            ids = s["input_ids"]
            labs = s["labels"]
            total = len(ids)
            text_pos = (labs != -100).sum().item()
            print(f"  Total tokens: {total}")
            print(f"  Text tokens (loss computed on): {text_pos}")
            audio_n = (ids >= config.audio_token_offset).sum().item()
            print(f"  Audio tokens: {audio_n}")
            text_toks = ids[labs != -100]
            decoded = ds.tokenizer.decode(text_toks.tolist(), skip_special_tokens=True)
            print(f"  Decoded text: {decoded[:80]}")
            print("✅ Dataset v5 PASSED!")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")

"""
DSM-ASR Dataset v4 — Instruction Fine-Tuning Format

Builds training sequences in instruction format:
    [instruction_ids] [AUDIO_PLACEHOLDER × T] [separator_ids] [text_ids] [EOS]

Returns:
    instruction_ids: tokenized instruction text (before audio)
    audio_tokens:    [T, Q] Mimi codebook indices
    separator_ids:   tokenized separator text (after audio)
    target_ids:      tokenized transcription text + EOS
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

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.qwen_model, trust_remote_code=True)

        # Pre-tokenize instruction and separator
        self.instruction_ids = self.tokenizer.encode(
            config.instruction, add_special_tokens=False)
        self.separator_ids = self.tokenizer.encode(
            config.separator, add_special_tokens=False)
        self.eos_id = self.tokenizer.eos_token_id

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
        print(f"📂 DsmAsrDataset [{split}]: {len(self.samples)} samples "
              f"(instr: {len(self.instruction_ids)} toks, sep: {len(self.separator_ids)} toks)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        info = self.samples[idx]
        data = np.load(info["path"], allow_pickle=True)
        audio_codes = torch.from_numpy(data["audio_codes"].astype(np.int64))
        text = str(data["text"]).strip()

        # Truncate audio if needed
        max_T = self.config.max_frames
        if audio_codes.shape[0] > max_T:
            audio_codes = audio_codes[:max_T]

        # Tokenize transcription
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.config.max_text_tokens:
            text_ids = text_ids[:self.config.max_text_tokens]

        # Add EOS at end
        text_ids = text_ids + [self.eos_id]

        return {
            "instruction_ids": torch.tensor(self.instruction_ids, dtype=torch.long),
            "audio_tokens": audio_codes,             # [T, Q]
            "separator_ids": torch.tensor(self.separator_ids, dtype=torch.long),
            "target_ids": torch.tensor(text_ids, dtype=torch.long),  # [N]
        }


if __name__ == "__main__":
    config = DsmAsrConfig()
    try:
        ds = DsmAsrDataset(config, split="train", max_samples=3)
        if len(ds) == 0:
            print("⚠️  No samples")
        else:
            s = ds[0]
            print(f"  instruction: {s['instruction_ids'].shape}")
            print(f"  audio:       {s['audio_tokens'].shape}")
            print(f"  separator:   {s['separator_ids'].shape}")
            print(f"  target:      {s['target_ids'].shape}")
            text = ds.tokenizer.decode(s['target_ids'], skip_special_tokens=True)
            print(f"  target text: {text[:100]}")
            print("✅ Dataset v4 PASSED!")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")

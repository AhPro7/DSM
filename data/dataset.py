"""
DSM-ASR Dataset v3 — With Real Word Timestamps (Moshi Paper)

Uses word-level timestamps from whisper to precisely place text tokens
at the correct audio frames, matching the paper's approach exactly.

Text stream format:
    PAD PAD PAD EPAD [word1_tokens] PAD PAD EPAD [word2_tokens] PAD ... EOS

Each word's tokens are placed starting at timestamp * frame_rate + delay.
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
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": config.special_tokens})

        self.pad_id = self.tokenizer.convert_tokens_to_ids("<|pad|>")
        self.epad_id = self.tokenizer.convert_tokens_to_ids("<|epad|>")
        self.bos_id = self.tokenizer.convert_tokens_to_ids("<|bos|>")
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|eos|>")

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

    def _build_text_stream_with_timestamps(self, data, T):
        """
        Build text stream using real word-level timestamps.
        Matches the Moshi paper exactly:
        - Each word placed at start_frame + delay
        - EPAD before each word
        - PAD everywhere else
        - BOS at start, EOS at end
        """
        delay = self.config.delay_frames
        frame_rate = self.config.frame_rate

        # Load timestamps
        word_starts = data.get("word_starts", np.array([]))
        word_texts = data.get("word_texts", np.array([]))

        if len(word_starts) == 0 or len(word_texts) == 0:
            # Fallback: no timestamps, spread text evenly
            return self._build_text_stream_fallback(str(data["text"]), T)

        # Initialize with PAD
        text_stream = [self.pad_id] * T
        loss_mask = [0.0] * T

        # Place BOS at frame 0 + delay
        bos_frame = min(delay, T - 1)
        text_stream[bos_frame] = self.bos_id
        loss_mask[bos_frame] = 1.0

        # Place each word at its timestamp position + delay
        for i, (start_sec, word_text) in enumerate(zip(word_starts, word_texts)):
            word_str = str(word_text).strip()
            if not word_str:
                continue

            # Tokenize this word
            word_ids = self.tokenizer.encode(word_str, add_special_tokens=False)
            if not word_ids:
                continue

            # Convert timestamp to frame index + delay
            start_frame = int(start_sec * frame_rate) + delay

            if start_frame >= T:
                continue

            # Insert EPAD before word (if room)
            epad_frame = start_frame - 1
            if epad_frame >= 0 and text_stream[epad_frame] == self.pad_id:
                text_stream[epad_frame] = self.epad_id
                loss_mask[epad_frame] = 0.5  # Lower weight for EPAD

            # Place word tokens starting at start_frame
            for j, tok_id in enumerate(word_ids):
                pos = start_frame + j
                if pos < T and text_stream[pos] == self.pad_id:
                    text_stream[pos] = tok_id
                    loss_mask[pos] = 1.0  # Full weight for real text

        # Place EOS after last text token
        last_text_pos = 0
        for pos in range(T - 1, -1, -1):
            if loss_mask[pos] == 1.0 and text_stream[pos] != self.bos_id:
                last_text_pos = pos
                break
        eos_pos = min(last_text_pos + 1, T - 1)
        text_stream[eos_pos] = self.eos_id
        loss_mask[eos_pos] = 1.0

        # Add low-weight loss on PAD positions so model learns timing
        for t in range(T):
            if loss_mask[t] == 0.0:
                loss_mask[t] = 0.3  # Lighter weight for PAD prediction

        text_tokens = torch.tensor(text_stream, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float32)

        # Targets: next-token prediction
        text_targets = torch.full((T,), -100, dtype=torch.long)
        for t in range(T - 1):
            text_targets[t] = text_tokens[t + 1]

        return text_tokens, text_targets, loss_mask

    def _build_text_stream_fallback(self, text, T):
        """Fallback when no timestamps — spread text evenly with delay."""
        delay = self.config.delay_frames
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not text_ids:
            text_ids = [self.eos_id]
        text_ids = [self.bos_id] + text_ids + [self.eos_id]

        available = T - delay
        if available <= 0:
            available = T
            delay = 0
        if len(text_ids) > available:
            text_ids = text_ids[:available - 1] + [self.eos_id]

        N = len(text_ids)
        text_stream = [self.pad_id] * T
        loss_mask = [0.3] * T  # Low weight for all PAD

        if N == 1:
            positions = [delay]
        else:
            positions = [delay + int(i * (available - 1) / (N - 1)) for i in range(N)]
            seen = set()
            clean = []
            for p in positions:
                while p in seen and p < T - 1:
                    p += 1
                if p < T:
                    seen.add(p)
                    clean.append(p)
            positions = clean

        for i, pos in enumerate(positions):
            if i < N and pos < T:
                if pos > 0 and text_stream[pos - 1] == self.pad_id:
                    text_stream[pos - 1] = self.epad_id
                    loss_mask[pos - 1] = 0.5
                text_stream[pos] = text_ids[i]
                loss_mask[pos] = 1.0

        text_tokens = torch.tensor(text_stream, dtype=torch.long)
        loss_mask_t = torch.tensor(loss_mask, dtype=torch.float32)
        text_targets = torch.full((T,), -100, dtype=torch.long)
        for t in range(T - 1):
            text_targets[t] = text_tokens[t + 1]

        return text_tokens, text_targets, loss_mask_t

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        info = self.samples[idx]
        data = dict(np.load(info["path"], allow_pickle=True))
        audio_codes = torch.from_numpy(data["audio_codes"].astype(np.int64))

        T = audio_codes.shape[0]
        max_T = self.config.max_frames
        if T > max_T:
            audio_codes = audio_codes[:max_T]
            T = max_T

        text_tokens, text_targets, loss_mask = \
            self._build_text_stream_with_timestamps(data, T)

        return {
            "audio_tokens": audio_codes,
            "text_tokens": text_tokens,
            "text_targets": text_targets,
            "loss_mask": loss_mask,
        }


if __name__ == "__main__":
    config = DsmAsrConfig()
    ds = DsmAsrDataset(config, split="train", max_samples=3)
    if len(ds) == 0:
        print("⚠️  No samples")
    else:
        s = ds[0]
        T = s["audio_tokens"].shape[0]
        real = (s["loss_mask"] == 1.0).sum().item()
        print(f"  T={T}, text_positions={real}")
        ids = s["text_tokens"][s["loss_mask"] == 1.0].tolist()
        print(f"  Decoded: {ds.tokenizer.decode(ids, skip_special_tokens=True)[:100]}...")
        print("✅ Dataset test PASSED")

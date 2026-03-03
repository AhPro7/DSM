"""
DSM-ASR Dataset v3 — True Delayed Streams

Based on the Moshi paper:
Audio and text are PARALLEL streams at 12.5Hz frame rate.
Text tokens are placed at their positions AFTER a delay of `delay_frames`.

Since we don't have word-level timestamps, we use a simple strategy:
- Tokenize the full text
- Spread text tokens evenly across the available frames (after delay)
- Fill gaps with PAD, insert EPAD before text bursts

Training format per sample:
    audio_tokens: [T, Q]  — Mimi codebook indices per frame
    text_tokens:  [T]     — text stream aligned to audio at frame rate
    text_targets: [T]     — shifted text_tokens for next-token prediction
    loss_mask:    [T]     — 1.0 for text positions, 0.0 for PAD positions
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


class DsmAsrDataset(Dataset):
    """
    Each sample returns parallel streams:
    - audio_tokens: [T, Q] codebook indices
    - text_tokens:  [T] text stream (mostly PAD, text tokens placed with delay)
    - text_targets: [T] next-token prediction targets (-100 for ignored)
    - loss_mask:    [T] 1.0 where we compute loss
    """

    def __init__(self, config: DsmAsrConfig, split="train",
                 max_samples=None, tokenizer=None):
        super().__init__()
        self.config = config
        self.split = split

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.qwen_model, trust_remote_code=True)
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": config.special_tokens})

        # Special token IDs
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<|pad|>")
        self.epad_id = self.tokenizer.convert_tokens_to_ids("<|epad|>")
        self.bos_id = self.tokenizer.convert_tokens_to_ids("<|bos|>")
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|eos|>")

        # Load manifest
        manifest_path = Path(config.preprocessed_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run prepare_data.py first.")
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

    def _build_text_stream(self, text: str, T: int) -> tuple:
        """
        Build the text stream aligned to audio frames.
        
        Strategy (simplified from paper):
        - Tokenize text → [tok_1, ..., tok_N]
        - Text starts at frame `delay_frames` (audio has head start)
        - Spread tokens evenly across frames [delay, T-1]
        - Insert EPAD before each word burst, PAD everywhere else

        Returns:
            text_tokens: [T] tensor
            text_targets: [T] tensor (shifted by 1)
            loss_mask: [T] tensor
        """
        delay = self.config.delay_frames
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if not text_ids:
            text_ids = [self.eos_id]

        # Add BOS at start, EOS at end
        text_ids = [self.bos_id] + text_ids + [self.eos_id]

        # Available frames for text (after delay)
        available_frames = T - delay
        if available_frames <= 0:
            available_frames = T
            delay = 0

        # Cap text length
        if len(text_ids) > available_frames:
            text_ids = text_ids[:available_frames - 1] + [self.eos_id]

        N = len(text_ids)

        # Initialize text stream with PAD
        text_tokens = torch.full((T,), self.pad_id, dtype=torch.long)
        loss_mask = torch.zeros(T, dtype=torch.float32)

        # Spread text tokens evenly across [delay, delay + available - 1]
        if N <= available_frames:
            # Calculate positions — spread evenly
            if N == 1:
                positions = [delay]
            else:
                positions = [delay + int(i * (available_frames - 1) / (N - 1))
                            for i in range(N)]
                # Ensure no duplicates
                seen = set()
                clean_positions = []
                for p in positions:
                    while p in seen and p < T - 1:
                        p += 1
                    if p < T:
                        seen.add(p)
                        clean_positions.append(p)
                positions = clean_positions

            # Place text tokens at positions
            for i, pos in enumerate(positions):
                if i < N and pos < T:
                    # Insert EPAD before text token (if there's room)
                    if pos > 0 and text_tokens[pos - 1] == self.pad_id:
                        text_tokens[pos - 1] = self.epad_id
                    text_tokens[pos] = text_ids[i]
                    loss_mask[pos] = 1.0

        # Build targets: shifted by 1 (predict next token)
        text_targets = torch.full((T,), -100, dtype=torch.long)
        # For positions with loss, the target is the NEXT text token
        active_positions = (loss_mask == 1.0).nonzero(as_tuple=True)[0].tolist()
        for idx in range(len(active_positions)):
            pos = active_positions[idx]
            if idx + 1 < len(active_positions):
                # Target = next text token
                next_pos = active_positions[idx + 1]
                text_targets[pos] = text_tokens[next_pos]
            else:
                # Last text token — target is EOS
                text_targets[pos] = self.eos_id

        # Also compute loss on EPAD and PAD positions between text tokens
        # so model learns WHEN to output text vs padding
        for t in range(T):
            if loss_mask[t] == 0.0:
                # For PAD/EPAD positions, model should predict PAD or EPAD
                if t + 1 < T:
                    text_targets[t] = text_tokens[t + 1]
                    loss_mask[t] = 0.5  # Lower weight for padding prediction

        return text_tokens, text_targets, loss_mask

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        data = np.load(sample_info["path"], allow_pickle=True)
        audio_codes = torch.from_numpy(data["audio_codes"].astype(np.int64))
        text = str(data["text"])

        T = audio_codes.shape[0]
        max_T = self.config.max_frames
        if T > max_T:
            audio_codes = audio_codes[:max_T]
            T = max_T

        text_tokens, text_targets, loss_mask = self._build_text_stream(text, T)

        return {
            "audio_tokens": audio_codes,       # [T, Q]
            "text_tokens": text_tokens,         # [T]
            "text_targets": text_targets,       # [T]
            "loss_mask": loss_mask,             # [T]
        }


def test_dataset():
    config = DsmAsrConfig()
    ds = DsmAsrDataset(config, split="train", max_samples=3)
    if len(ds) == 0:
        print("⚠️  No samples. Run prepare_data.py first.")
        return

    sample = ds[0]
    T = sample["audio_tokens"].shape[0]
    print(f"\n  audio: {sample['audio_tokens'].shape}")
    print(f"  text_tokens: {sample['text_tokens'].shape}")
    print(f"  Total frames: {T}")

    text_positions = (sample["loss_mask"] >= 0.5).sum().item()
    real_text = (sample["loss_mask"] == 1.0).sum().item()
    print(f"  Text positions: {real_text} / {T}")

    # Decode text tokens
    text_ids = sample["text_tokens"][sample["loss_mask"] == 1.0].tolist()
    decoded = ds.tokenizer.decode(text_ids, skip_special_tokens=True)
    print(f"  Decoded: {decoded[:100]}...")

    print("\n✅ Dataset test PASSED!")


if __name__ == "__main__":
    test_dataset()

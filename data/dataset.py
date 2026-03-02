"""
DSM-ASR Dataset

PyTorch Dataset implementing the Delayed Streams Modeling (DSM) data pipeline.
Handles alignment of text tokens to the audio frame grid with a configurable delay.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List
from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


class DsmAsrDataset(Dataset):
    """
    Dataset for DSM-ASR training.

    Each sample contains:
    - audio_tokens: [T, Q] int tensor of Mimi codebook indices
    - text_tokens:  [T]    int tensor of aligned text token IDs (PAD for empty frames)
    - text_targets: [T]    int tensor of target text tokens (shifted by 1)
    - text_loss_mask: [T]  float tensor, 1.0 where text loss should be computed
    - attention_mask: [T]  float tensor, 1.0 for valid frames

    The text stream is delayed by `delay_frames` relative to audio:
    - At frame `t`, the audio is from time `t / frame_rate`
    - The text at frame `t` corresponds to audio at time `(t - delay) / frame_rate`
    - This means the model sees audio before the corresponding text, which is
      essential for ASR: you hear the speech before you write it down.
    """

    def __init__(
        self,
        config: DsmAsrConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self.split = split

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)

        # Add special tokens
        special_tokens = {"additional_special_tokens": config.special_tokens}
        self.tokenizer.add_special_tokens(special_tokens)

        # Get special token IDs
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|pad|>")
        self.word_token_id = self.tokenizer.convert_tokens_to_ids("<|word|>")

        # Load manifest
        manifest_path = Path(config.preprocessed_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run `python data/prepare_timestamps.py` first."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.samples = manifest["samples"]

        # Apply train/eval split
        if split == "train" and config.eval_split is None:
            # Use (1 - eval_ratio) for training
            n_eval = max(1, int(len(self.samples) * config.eval_ratio))
            self.samples = self.samples[:-n_eval]
        elif split == "eval" and config.eval_split is None:
            n_eval = max(1, int(len(self.samples) * config.eval_ratio))
            self.samples = self.samples[-n_eval:]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"📂 DsmAsrDataset [{split}]: {len(self.samples)} samples, "
              f"delay={config.delay_frames} frames ({config.delay_frames / config.frame_rate:.0f}ms)")

    def __len__(self):
        return len(self.samples)

    def _align_text_to_frames(
        self,
        word_timestamps: List[Dict],
        text: str,
        num_frames: int,
    ) -> tuple:
        """
        Align text tokens to the audio frame grid with delay.

        For each word:
        1. Determine the frame index from word start time
        2. Add delay_frames offset
        3. Tokenize the word
        4. Place tokens at consecutive frames starting from that frame

        Returns:
            text_tokens: [num_frames] tensor of token IDs (pad_token_id for empty)
            text_loss_mask: [num_frames] tensor, 1.0 where text exists
        """
        delay = self.config.delay_frames
        frame_rate = self.config.frame_rate

        # Initialize with PAD
        text_tokens = torch.full((num_frames,), self.pad_token_id, dtype=torch.long)
        text_loss_mask = torch.zeros(num_frames, dtype=torch.float32)

        frame_cursor = delay  # Start placing text after the delay

        for word_info in word_timestamps:
            word = word_info["word"]
            start_time = word_info["start"]

            # Target frame = audio frame of word start + delay
            target_frame = int(start_time * frame_rate) + delay

            # Don't go backwards; if whisper timestamps overlap, just continue forward
            target_frame = max(target_frame, frame_cursor)

            # Tokenize this word (no special tokens)
            word_token_ids = self.tokenizer.encode(word, add_special_tokens=False)

            if not word_token_ids:
                continue

            # Place WORD boundary marker at target frame (optional)
            if target_frame < num_frames:
                text_tokens[target_frame] = self.word_token_id
                text_loss_mask[target_frame] = 1.0
                target_frame += 1

            # Place word tokens at consecutive frames
            for token_id in word_token_ids:
                if target_frame >= num_frames:
                    break
                text_tokens[target_frame] = token_id
                text_loss_mask[target_frame] = 1.0
                target_frame += 1

            frame_cursor = target_frame

        return text_tokens, text_loss_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        sample_path = sample_info["path"]

        # Load preprocessed data
        data = np.load(sample_path, allow_pickle=True)
        audio_codes = torch.from_numpy(data["audio_codes"].astype(np.int64))  # [T, Q]
        word_timestamps = json.loads(str(data["word_timestamps"]))
        text = str(data["text"])

        num_frames = audio_codes.shape[0]

        # Cap to max_seq_len
        if num_frames > self.config.max_seq_len:
            audio_codes = audio_codes[:self.config.max_seq_len]
            num_frames = self.config.max_seq_len

        # Align text to frame grid
        text_tokens, text_loss_mask = self._align_text_to_frames(
            word_timestamps, text, num_frames
        )

        # Create text targets (shifted by 1 for next-token prediction)
        # Target at position t is the text token at position t+1
        text_targets = torch.full((num_frames,), -100, dtype=torch.long)
        text_targets[:-1] = text_tokens[1:]
        # Only compute loss where the TARGET has a real text token
        target_loss_mask = torch.zeros(num_frames, dtype=torch.float32)
        target_loss_mask[:-1] = text_loss_mask[1:]

        # Attention mask (all valid frames = 1)
        attention_mask = torch.ones(num_frames, dtype=torch.float32)

        return {
            "audio_tokens": audio_codes,          # [T, Q]
            "text_tokens": text_tokens,            # [T]
            "text_targets": text_targets,          # [T]
            "text_loss_mask": target_loss_mask,    # [T]
            "attention_mask": attention_mask,       # [T]
        }


def test_dataset_pipeline():
    """Quick test to verify the dataset pipeline works."""
    config = DsmAsrConfig()
    config.preprocessed_dir = "./preprocessed_data"

    print("Testing dataset pipeline...")
    ds = DsmAsrDataset(config, split="train", max_samples=3)

    if len(ds) == 0:
        print("⚠️  No samples found. Run prepare_timestamps.py first.")
        return

    sample = ds[0]
    print(f"\n  audio_tokens shape: {sample['audio_tokens'].shape}")
    print(f"  text_tokens shape:  {sample['text_tokens'].shape}")
    print(f"  text_targets shape: {sample['text_targets'].shape}")
    print(f"  text_loss_mask sum: {sample['text_loss_mask'].sum().item():.0f}")
    print(f"  attention_mask sum: {sample['attention_mask'].sum().item():.0f}")

    # Check shapes are consistent
    T = sample["audio_tokens"].shape[0]
    Q = sample["audio_tokens"].shape[1]
    assert sample["text_tokens"].shape == (T,), f"Expected ({T},), got {sample['text_tokens'].shape}"
    assert Q == config.num_codebooks, f"Expected Q={config.num_codebooks}, got {Q}"

    # Decode some text tokens to verify
    text_mask = sample["text_loss_mask"] > 0
    if text_mask.any():
        # Get the input text tokens where loss mask is active
        # (shift back since loss mask is for targets)
        active_text = sample["text_tokens"][sample["text_loss_mask"] > 0]
        # Filter out special tokens
        real_tokens = [t.item() for t in active_text if t.item() != ds.pad_token_id and t.item() != ds.word_token_id]
        if real_tokens:
            decoded = ds.tokenizer.decode(real_tokens)
            print(f"  Decoded text preview: {decoded[:100]}...")

    print("\n✅ Dataset pipeline test PASSED!")


if __name__ == "__main__":
    test_dataset_pipeline()

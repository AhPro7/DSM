"""
DSM-ASR Dataset (Audio Prefix → Text Generation)

The model sees audio tokens as a prefix and generates text tokens after.
No timestamps needed — the model learns alignment itself.

Sequence format:
    [audio_emb_1, ..., audio_emb_T, START_TEXT, text_tok_1, ..., text_tok_N, END_TEXT]
    |<-------- no loss ---------->|  no loss  |<------- loss here -------->|  loss  |
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
    Dataset for audio-prefix ASR.

    Each sample returns:
    - audio_tokens: [T_audio, Q] int tensor of Mimi codebook indices
    - text_input_ids: [N_text] int tensor = [START_TEXT, tok_1, ..., tok_N]
    - text_target_ids: [N_text] int tensor = [tok_1, ..., tok_N, END_TEXT]
    
    The collator will concatenate audio + text into the full sequence.
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
            special_tokens = {"additional_special_tokens": config.special_tokens}
            self.tokenizer.add_special_tokens(special_tokens)

        # Get special token IDs
        self.start_text_id = self.tokenizer.convert_tokens_to_ids("<|start_text|>")
        self.end_text_id = self.tokenizer.convert_tokens_to_ids("<|end_text|>")

        # Load manifest
        manifest_path = Path(config.preprocessed_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Run `python data/prepare_data.py` first."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.samples = manifest["samples"]

        # Apply train/eval split
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        sample_path = sample_info["path"]

        # Load preprocessed data
        data = np.load(sample_path, allow_pickle=True)
        audio_codes = torch.from_numpy(data["audio_codes"].astype(np.int64))  # [T, Q]
        text = str(data["text"])

        # Cap audio frames
        max_audio = self.config.max_frames
        if audio_codes.shape[0] > max_audio:
            audio_codes = audio_codes[:max_audio]

        # Tokenize text
        text_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # Cap text length
        if len(text_token_ids) > self.config.max_text_tokens - 2:  # -2 for start/end
            text_token_ids = text_token_ids[:self.config.max_text_tokens - 2]

        # Build text input: [START_TEXT, tok_1, ..., tok_N]
        text_input = [self.start_text_id] + text_token_ids
        # Build text target: [tok_1, ..., tok_N, END_TEXT]
        text_target = text_token_ids + [self.end_text_id]

        return {
            "audio_tokens": audio_codes,                                    # [T_audio, Q]
            "text_input_ids": torch.tensor(text_input, dtype=torch.long),   # [N_text]
            "text_target_ids": torch.tensor(text_target, dtype=torch.long), # [N_text]
        }


def test_dataset():
    """Quick test to verify the dataset works."""
    config = DsmAsrConfig()
    print("Testing dataset pipeline...")
    ds = DsmAsrDataset(config, split="train", max_samples=3)

    if len(ds) == 0:
        print("⚠️  No samples found. Run prepare_data.py first.")
        return

    sample = ds[0]
    T_audio = sample["audio_tokens"].shape[0]
    Q = sample["audio_tokens"].shape[1]
    N_text = sample["text_input_ids"].shape[0]

    print(f"\n  audio_tokens shape:  {sample['audio_tokens'].shape}  (T_audio={T_audio}, Q={Q})")
    print(f"  text_input_ids len:  {N_text}")
    print(f"  text_target_ids len: {sample['text_target_ids'].shape[0]}")
    print(f"  Total seq length:    {T_audio + N_text}")

    # Decode text to verify
    decoded = ds.tokenizer.decode(sample["text_input_ids"][1:], skip_special_tokens=False)
    print(f"  Text preview: {decoded[:100]}...")

    assert Q == config.num_codebooks
    assert sample["text_input_ids"][0] == ds.start_text_id
    assert sample["text_target_ids"][-1] == ds.end_text_id

    print("\n✅ Dataset test PASSED!")


if __name__ == "__main__":
    test_dataset()

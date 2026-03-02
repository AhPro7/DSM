"""
DSM-ASR Data Collator

Custom collator for batching variable-length DSM sequences.
Handles padding of audio tokens, text tokens, and masks.
"""
import torch
from typing import Dict, List
from dataclasses import dataclass

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


@dataclass
class DsmAsrCollator:
    """
    Pads variable-length DSM samples to form a batch.
    
    Padding strategy:
    - audio_tokens: padded with audio_pad_token
    - text_tokens: padded with pad_token_id (from tokenizer)
    - text_targets: padded with -100 (ignored by CE loss)
    - text_loss_mask: padded with 0.0
    - attention_mask: padded with 0.0
    """

    config: DsmAsrConfig
    pad_token_id: int  # Text pad token ID from tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max sequence length in this batch
        max_len = max(sample["audio_tokens"].shape[0] for sample in batch)
        batch_size = len(batch)
        Q = self.config.num_codebooks

        # Pre-allocate padded tensors
        audio_tokens = torch.full(
            (batch_size, max_len, Q),
            self.config.audio_pad_token,
            dtype=torch.long,
        )
        text_tokens = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        text_targets = torch.full(
            (batch_size, max_len),
            -100,  # Ignored by cross-entropy
            dtype=torch.long,
        )
        text_loss_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

        # Fill in actual values
        for i, sample in enumerate(batch):
            T = sample["audio_tokens"].shape[0]
            audio_tokens[i, :T] = sample["audio_tokens"]
            text_tokens[i, :T] = sample["text_tokens"]
            text_targets[i, :T] = sample["text_targets"]
            text_loss_mask[i, :T] = sample["text_loss_mask"]
            attention_mask[i, :T] = sample["attention_mask"]

        return {
            "audio_tokens": audio_tokens,        # [B, T, Q]
            "text_tokens": text_tokens,           # [B, T]
            "text_targets": text_targets,          # [B, T]
            "text_loss_mask": text_loss_mask,      # [B, T]
            "attention_mask": attention_mask,       # [B, T]
        }

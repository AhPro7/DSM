"""
DSM-ASR Collator v3 — Parallel Streams

Pads variable-length samples to the same T (num frames).
Both audio_tokens and text_tokens are padded to max T in the batch.
"""
import torch
from typing import Dict, List
from dataclasses import dataclass

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


@dataclass
class DsmAsrCollator:
    config: DsmAsrConfig
    pad_text_id: int  # PAD token ID for text stream

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        Q = self.config.num_codebooks
        max_T = max(s["audio_tokens"].shape[0] for s in batch)

        audio_tokens = torch.full((B, max_T, Q), self.config.audio_pad_token, dtype=torch.long)
        text_tokens = torch.full((B, max_T), self.pad_text_id, dtype=torch.long)
        text_targets = torch.full((B, max_T), -100, dtype=torch.long)
        loss_mask = torch.zeros(B, max_T, dtype=torch.float32)
        lengths = torch.zeros(B, dtype=torch.long)

        for i, s in enumerate(batch):
            T = s["audio_tokens"].shape[0]
            audio_tokens[i, :T] = s["audio_tokens"]
            text_tokens[i, :T] = s["text_tokens"]
            text_targets[i, :T] = s["text_targets"]
            loss_mask[i, :T] = s["loss_mask"]
            lengths[i] = T

        return {
            "audio_tokens": audio_tokens,  # [B, T, Q]
            "text_tokens": text_tokens,    # [B, T]
            "text_targets": text_targets,  # [B, T]
            "loss_mask": loss_mask,        # [B, T]
            "lengths": lengths,            # [B]
        }

"""
DSM-ASR Data Collator (Audio Prefix → Text Generation)

Builds the full training sequence by concatenating audio embeddings + text tokens.
Handles padding across variable-length samples in a batch.

Sequence layout per sample:
    [AUDIO_1, ..., AUDIO_T, START_TEXT, text_1, ..., text_N] ← input
    [-100,    ..., -100,    text_1,     ...,    text_N, END]  ← target (loss only on text)
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
    Collates variable-length audio+text samples into padded batches.
    
    Returns a batch dict with:
    - audio_tokens:    [B, T_audio_max, Q]  padded audio codebook indices
    - text_input_ids:  [B, N_text_max]      padded text input token IDs
    - text_target_ids: [B, N_text_max]      padded text target token IDs (-100 for pad)
    - audio_lengths:   [B]                  actual audio lengths per sample
    - text_lengths:    [B]                  actual text lengths per sample
    """
    config: DsmAsrConfig
    text_pad_token_id: int  # Usually 0 or tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        Q = self.config.num_codebooks

        # Get max lengths
        max_audio_len = max(s["audio_tokens"].shape[0] for s in batch)
        max_text_len = max(s["text_input_ids"].shape[0] for s in batch)

        # Pre-allocate padded tensors
        audio_tokens = torch.full(
            (batch_size, max_audio_len, Q),
            self.config.audio_pad_token,
            dtype=torch.long,
        )
        text_input_ids = torch.full(
            (batch_size, max_text_len),
            self.text_pad_token_id,
            dtype=torch.long,
        )
        text_target_ids = torch.full(
            (batch_size, max_text_len),
            -100,  # Ignored by cross-entropy
            dtype=torch.long,
        )
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        text_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, sample in enumerate(batch):
            T_a = sample["audio_tokens"].shape[0]
            T_t = sample["text_input_ids"].shape[0]

            audio_tokens[i, :T_a] = sample["audio_tokens"]
            text_input_ids[i, :T_t] = sample["text_input_ids"]
            text_target_ids[i, :T_t] = sample["text_target_ids"]
            audio_lengths[i] = T_a
            text_lengths[i] = T_t

        return {
            "audio_tokens": audio_tokens,        # [B, T_audio, Q]
            "text_input_ids": text_input_ids,     # [B, N_text]
            "text_target_ids": text_target_ids,   # [B, N_text]  (-100 for padding)
            "audio_lengths": audio_lengths,       # [B]
            "text_lengths": text_lengths,          # [B]
        }

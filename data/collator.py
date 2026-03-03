"""
DSM-ASR Collator v4 — Instruction Format

Assembles the full training sequence from parts:
    [instruction] [AUDIO_TOKEN × T] [separator] [target_text] [PAD...]

Builds:
    input_embeds_parts: list of (type, tensor) to be embedded by model
    labels: -100 for instruction/audio/separator, real token ids for target
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
    text_pad_id: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        Q = self.config.num_codebooks

        # Compute sequence lengths
        instr_len = batch[0]["instruction_ids"].shape[0]  # Same for all
        sep_len = batch[0]["separator_ids"].shape[0]
        max_audio_T = max(s["audio_tokens"].shape[0] for s in batch)
        max_target_N = max(s["target_ids"].shape[0] for s in batch)

        # Total = instruction + audio + separator + target
        total_len = instr_len + max_audio_T + sep_len + max_target_N

        # Allocate tensors
        instruction_ids = batch[0]["instruction_ids"]  # shared
        separator_ids = batch[0]["separator_ids"]       # shared

        audio_tokens = torch.full(
            (B, max_audio_T, Q), self.config.audio_pad_token, dtype=torch.long)
        target_ids = torch.full(
            (B, max_target_N), self.text_pad_id, dtype=torch.long)
        labels = torch.full(
            (B, total_len), -100, dtype=torch.long)
        audio_lengths = torch.zeros(B, dtype=torch.long)
        target_lengths = torch.zeros(B, dtype=torch.long)

        for i, s in enumerate(batch):
            T = s["audio_tokens"].shape[0]
            N = s["target_ids"].shape[0]

            audio_tokens[i, :T] = s["audio_tokens"]
            target_ids[i, :N] = s["target_ids"]
            audio_lengths[i] = T
            target_lengths[i] = N

            # Labels: only on target positions
            # position = instr_len + max_audio_T + sep_len + [0..N-1]
            target_start = instr_len + max_audio_T + sep_len
            labels[i, target_start:target_start + N] = s["target_ids"]

        return {
            "instruction_ids": instruction_ids,  # [instr_len]
            "audio_tokens": audio_tokens,         # [B, max_T, Q]
            "separator_ids": separator_ids,       # [sep_len]
            "target_ids": target_ids,             # [B, max_N]
            "labels": labels,                     # [B, total_len]
            "audio_lengths": audio_lengths,       # [B]
            "target_lengths": target_lengths,     # [B]
        }

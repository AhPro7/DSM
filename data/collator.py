"""
DSM-ASR Collator v5 — Flat 1D Sequences

Pads input_ids and labels to the max length in the batch.
input_ids padded with tokenizer pad_token_id
labels padded with -100 (ignored in loss)
"""
import torch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DsmAsrCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(s["input_ids"].shape[0] for s in batch)
        B = len(batch)

        input_ids   = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels      = torch.full((B, max_len), -100,              dtype=torch.long)
        attention_mask = torch.zeros(B, max_len,                  dtype=torch.long)

        for i, s in enumerate(batch):
            L = s["input_ids"].shape[0]
            input_ids[i, :L]      = s["input_ids"]
            labels[i, :L]         = s["labels"]
            attention_mask[i, :L] = 1

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
        }

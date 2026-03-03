"""
DSM-ASR Configuration v5 — SODA-Style Interleaved Tokens

Architecture from SODA paper (soda-audio.github.io):
  Audio tokens are added directly to the LM vocabulary.
  No separate embedding tables, no adapters.
  Training = standard next-token prediction on interleaved sequences.

Sequence format:
  <|audio_start|> [audio_tok_0..tok_T*Q-1] <|audio_end|> <|text_start|> [text] <|text_end|>

Token mapping (per frame f, codebook q, value v):
  flat_id  = q * audio_codebook_size + v          (0..16383)
  vocab_id = flat_id + text_vocab_size + num_special_tokens
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DsmAsrConfig:
    # ── Backbone ──────────────────────────────────────────────────────
    qwen_model: str = "Qwen/Qwen3-0.6B-Base"
    mimi_model: str = "kyutai/mimi"

    # ── Mimi audio codec ─────────────────────────────────────────────
    sample_rate: int = 24_000
    frame_rate: float = 12.5          # frames/sec
    num_codebooks: int = 8
    audio_codebook_size: int = 2048   # each codebook: 0..2047

    # ── Vocabulary ───────────────────────────────────────────────────
    # Qwen3-0.6B text vocab size (before adding audio tokens)
    text_vocab_size: int = 151_671
    # 4 special tokens added BEFORE audio token range
    num_extra_special: int = 4
    # audio vocab = num_codebooks * audio_codebook_size = 16384
    # total = 151671 + 4 + 16384 = 168059

    # Special token strings
    audio_start_token: str = "<|audio_start|>"
    audio_end_token:   str = "<|audio_end|>"
    text_start_token:  str = "<|text_start|>"
    text_end_token:    str = "<|text_end|>"

    # ── Sequence limits ──────────────────────────────────────────────
    max_audio_duration: float = 30.0
    max_text_tokens: int = 256

    # ── Training ─────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 15
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    fp16: bool = False
    bf16: bool = True

    # ── Data ─────────────────────────────────────────────────────────
    dataset_name: str = "nadsoft/auto-stt-22-2-2026_cleaned-dataset"
    audio_column: str = "audio"
    text_column: str = "normalized_text"
    train_split: str = "train"
    eval_split: Optional[str] = None
    eval_ratio: float = 0.05

    # ── Paths ────────────────────────────────────────────────────────
    output_dir: str = "./output"
    preprocessed_dir: str = "./preprocessed_data"

    # ── Logging / Monitoring ─────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "dsm-asr"
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 200
    save_every_n_steps: int = 500
    print_samples_every: int = 50
    num_print_samples: int = 5

    # ── Derived (read-only) ──────────────────────────────────────────
    @property
    def audio_vocab_size(self) -> int:
        return self.num_codebooks * self.audio_codebook_size  # 16384

    @property
    def total_vocab_size(self) -> int:
        return self.text_vocab_size + self.num_extra_special + self.audio_vocab_size

    @property
    def audio_token_offset(self) -> int:
        """First audio vocab_id in the combined vocabulary."""
        return self.text_vocab_size + self.num_extra_special

    @property
    def max_frames(self) -> int:
        return int(self.max_audio_duration * self.frame_rate)  # 375

    def flat_audio_id(self, codebook: int, code_value: int) -> int:
        """Map (codebook, code_value) → vocab token id."""
        flat = codebook * self.audio_codebook_size + code_value
        return flat + self.audio_token_offset

    @property
    def special_tokens(self):
        return [self.audio_start_token, self.audio_end_token,
                self.text_start_token, self.text_end_token]

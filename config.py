"""
DSM-ASR Configuration (v2 - Improved)
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DsmAsrConfig:
    """All hyperparameters for DSM-ASR training and inference."""

    # ── Model paths ──────────────────────────────────────────────────
    qwen_model: str = "Qwen/Qwen3-0.6B-Base"
    mimi_model: str = "kyutai/mimi"

    # ── Audio (Mimi) ─────────────────────────────────────────────────
    sample_rate: int = 24_000
    frame_rate: float = 12.5
    num_codebooks: int = 8
    audio_vocab_size: int = 2048
    audio_pad_token: int = 2048

    # ── Audio Adapter (NEW - critical for quality) ───────────────────
    audio_adapter_layers: int = 2          # MLP layers in audio adapter
    audio_adapter_dropout: float = 0.1     # Dropout in adapter

    # ── Special tokens ───────────────────────────────────────────────
    special_tokens: list = field(default_factory=lambda: ["<|start_text|>", "<|end_text|>"])

    # ── Text prompt (NEW - helps LM understand the task) ─────────────
    # Prepended before audio tokens to tell the model what to do
    text_prompt: str = "Transcribe the following audio into text:\n"

    # ── Sequence lengths ─────────────────────────────────────────────
    max_audio_duration: float = 30.0
    max_text_tokens: int = 256
    max_seq_len: int = 640

    # ── Training ─────────────────────────────────────────────────────
    learning_rate: float = 2e-4            # 4x higher than before
    audio_lr_multiplier: float = 5.0       # Audio params LR multiplier
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 15                   # More epochs
    warmup_ratio: float = 0.05             # Faster warmup
    max_grad_norm: float = 1.0
    freeze_backbone_epochs: int = 0        # NO FREEZE - train together from start
    label_smoothing: float = 0.1           # Label smoothing for regularization
    fp16: bool = False
    bf16: bool = True

    # ── Data ─────────────────────────────────────────────────────────
    dataset_name: str = "nadsoft/auto-stt-22-2-2026_cleaned-dataset"
    audio_column: str = "audio"
    text_column: str = "normalized_text"
    train_split: str = "train"
    eval_split: Optional[str] = None
    eval_ratio: float = 0.05
    preprocessing_num_workers: int = 4

    # ── Paths ────────────────────────────────────────────────────────
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    preprocessed_dir: str = "./preprocessed_data"

    # ── Logging ──────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "dsm-asr"
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 200
    save_every_n_steps: int = 500

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    @property
    def max_frames(self) -> int:
        return int(self.max_audio_duration * self.frame_rate)

    @property
    def model_dim(self) -> int:
        return 1024

"""
DSM-ASR Configuration v3 — True Delayed Streams

Based on Moshi paper (arXiv:2410.00037):
- Audio + text are PARALLEL streams at 12.5Hz frame rate
- Text tokens are delayed behind audio by `delay_seconds`
- At training: both streams teacher-forced, loss on text only
- At inference: audio teacher-forced, text autoregressively decoded

Simplified from full Moshi: no Depth Transformer, just Temporal Transformer (Qwen3).
We sum codebook embeddings per frame → single embedding per frame.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DsmAsrConfig:
    # ── Model paths ──────────────────────────────────────────────────
    qwen_model: str = "Qwen/Qwen3-0.6B-Base"
    mimi_model: str = "kyutai/mimi"

    # ── Audio (Mimi) ─────────────────────────────────────────────────
    sample_rate: int = 24_000
    frame_rate: float = 12.5
    num_codebooks: int = 8
    audio_vocab_size: int = 2048
    audio_pad_token: int = 2048

    # ── DSM Delay ────────────────────────────────────────────────────
    # Text is delayed behind audio — model hears audio before predicting text
    # e.g. delay_frames=25 at 12.5Hz = 2.0s delay (same as paper for ASR)
    delay_frames: int = 25  # ~2 seconds

    # ── Special text tokens (on the text stream) ─────────────────────
    # PAD: text stream padding (most positions have this)
    # EPAD: end-of-padding, signals text content follows
    # BOS/EOS: start/end of text content
    special_tokens: list = field(default_factory=lambda: [
        "<|pad|>", "<|epad|>", "<|bos|>", "<|eos|>"
    ])

    # ── Audio Adapter ────────────────────────────────────────────────
    audio_adapter_layers: int = 2
    audio_adapter_dropout: float = 0.1

    # ── Sequence lengths ─────────────────────────────────────────────
    max_audio_duration: float = 30.0
    max_text_tokens: int = 256

    # ── Training ─────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    audio_lr_multiplier: float = 5.0
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 15
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    freeze_backbone_epochs: int = 0
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
    # Print sample predictions every N steps
    print_samples_every: int = 50
    num_print_samples: int = 5

    @property
    def max_frames(self) -> int:
        return int(self.max_audio_duration * self.frame_rate)

    @property
    def model_dim(self) -> int:
        return 1024

"""
DSM-ASR Configuration
Central configuration for the Audio-Prefix ASR system.
Audio tokens (from Mimi) serve as prefix, model generates text autoregressively.
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
    sample_rate: int = 24_000          # Mimi native sample rate
    frame_rate: float = 12.5           # Mimi frame rate (Hz)
    num_codebooks: int = 8             # Use first 8 of 32 codebooks (most semantic)
    audio_vocab_size: int = 2048       # Mimi codebook vocab size
    audio_pad_token: int = 2048        # PAD token for audio (= vocab_size, extra slot)

    # ── Special tokens ───────────────────────────────────────────────
    # <|start_text|> separates audio prefix from text generation
    # <|end_text|> marks end of transcription
    special_tokens: list = field(default_factory=lambda: ["<|start_text|>", "<|end_text|>"])

    # ── Sequence lengths ─────────────────────────────────────────────
    max_audio_duration: float = 30.0   # Maximum audio duration in seconds
    max_text_tokens: int = 256         # Maximum text tokens in output
    max_seq_len: int = 640             # Max total sequence (audio frames + text tokens)

    # ── Training ─────────────────────────────────────────────────────
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    freeze_backbone_epochs: int = 2    # Freeze Qwen3 backbone for N epochs
    fp16: bool = False
    bf16: bool = True

    # ── Data ─────────────────────────────────────────────────────────
    dataset_name: str = "nadsoft/auto-stt-22-2-2026_cleaned-dataset"
    audio_column: str = "audio"
    text_column: str = "normalized_text"
    train_split: str = "train"
    eval_split: Optional[str] = None   # Will use train split with % if None
    eval_ratio: float = 0.05           # 5% for eval if no eval split
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
        """Number of audio samples per Mimi frame."""
        return int(self.sample_rate / self.frame_rate)

    @property
    def max_frames(self) -> int:
        """Maximum number of frames for max_audio_duration."""
        return int(self.max_audio_duration * self.frame_rate)

    @property
    def model_dim(self) -> int:
        """Qwen3-0.6B hidden size."""
        return 1024

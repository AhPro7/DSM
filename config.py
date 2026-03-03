"""
DSM-ASR Configuration v4 — Instruction Fine-Tuning

Architecture:
    [instruction] [audio_emb_1...audio_emb_T] [sep] [transcription] [EOS]
    |<------------ no loss ----------------->|      |<-- loss here ->|
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DsmAsrConfig:
    # ── Models ───────────────────────────────────────────────────────
    qwen_model: str = "Qwen/Qwen3-0.6B-Base"
    mimi_model: str = "kyutai/mimi"

    # ── Audio (Mimi) ─────────────────────────────────────────────────
    sample_rate: int = 24_000
    frame_rate: float = 12.5
    num_codebooks: int = 8
    audio_vocab_size: int = 2048
    audio_pad_token: int = 2048

    # ── Audio Adapter ────────────────────────────────────────────────
    audio_adapter_layers: int = 2
    audio_adapter_dropout: float = 0.1

    # ── Instruction / Prompt ─────────────────────────────────────────
    instruction: str = "Transcribe the following audio:\n"
    separator: str = "\nTranscription: "

    # ── Sequence limits ──────────────────────────────────────────────
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
    preprocessed_dir: str = "./preprocessed_data"

    # ── Logging ──────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "dsm-asr"
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 200
    save_every_n_steps: int = 500
    print_samples_every: int = 50
    num_print_samples: int = 5

    @property
    def max_frames(self) -> int:
        return int(self.max_audio_duration * self.frame_rate)

    @property
    def model_dim(self) -> int:
        return 1024

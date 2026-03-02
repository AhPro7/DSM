"""
DSM-ASR Data Preparation (No Timestamps)

Preprocesses the dataset by:
1. Loading audio from HuggingFace dataset
2. Pre-encoding audio with Mimi to save audio codes
3. Saving text + audio codes to disk

No whisper-timestamped needed — the model learns alignment itself.
"""
import os
import json
import argparse
import numpy as np
import torch
import librosa
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Audio

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


def encode_audio_with_mimi(
    audio_array: np.ndarray,
    sample_rate: int,
    mimi_model,
    feature_extractor,
    target_sr: int = 24000,
    num_codebooks: int = 8,
    device: str = "cuda",
):
    """
    Encode audio with Mimi and return audio codes [T, Q].
    """
    # Resample to 24kHz if needed
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)

    # Prepare input
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=target_sr,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device)

    # Encode with Mimi
    with torch.no_grad():
        encoder_outputs = mimi_model.encode(input_values)
        # audio_codes shape can be [B, codebooks, T] or [B, 1, codebooks, T]
        codes = encoder_outputs.audio_codes
        if codes.dim() == 4:
            codes = codes[0, 0, :num_codebooks, :].T  # [T, Q]
        else:
            codes = codes[0, :num_codebooks, :].T      # [T, Q]

    return codes.cpu().numpy()


def preprocess_dataset(config: DsmAsrConfig, max_samples: int = None):
    """
    Preprocessing pipeline:
    1. Load HuggingFace dataset
    2. Encode audio with Mimi
    3. Save audio codes + text
    """
    print("=" * 60)
    print("DSM-ASR Data Preprocessing (No Timestamps)")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────
    print(f"\n📥 Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name, split=config.train_split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"   Using {len(ds)} samples (limited)")
    else:
        print(f"   Loaded {len(ds)} samples")

    # Ensure audio is decoded at Mimi's sample rate
    ds = ds.cast_column(config.audio_column, Audio(sampling_rate=config.sample_rate))

    # ── Load Mimi ────────────────────────────────────────────────────
    print(f"\n🔊 Loading Mimi model: {config.mimi_model}")
    from transformers import MimiModel, AutoFeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mimi_model = MimiModel.from_pretrained(config.mimi_model).to(device)
    mimi_model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.mimi_model)
    print(f"   Mimi loaded on {device}")

    # ── Process samples ──────────────────────────────────────────────
    output_dir = Path(config.preprocessed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_samples = []
    errors = []

    print(f"\n⚙️  Processing {len(ds)} samples...")
    for idx in tqdm(range(len(ds)), desc="Preprocessing"):
        try:
            sample = ds[idx]
            audio_data = sample[config.audio_column]
            audio_array = audio_data["array"]
            sr = audio_data["sampling_rate"]
            text = sample[config.text_column]

            # Skip empty samples
            if len(audio_array) < sr * 0.1 or not text or not text.strip():
                errors.append({"idx": idx, "error": "too short or empty text"})
                continue

            # Skip very long samples
            duration = len(audio_array) / sr
            if duration > config.max_audio_duration:
                errors.append({"idx": idx, "error": f"too long: {duration:.1f}s"})
                continue

            # Encode audio with Mimi
            audio_codes = encode_audio_with_mimi(
                audio_array, sr, mimi_model, feature_extractor,
                target_sr=config.sample_rate,
                num_codebooks=config.num_codebooks,
                device=device,
            )

            # Save processed sample
            sample_path = output_dir / f"sample_{idx:06d}.npz"
            np.savez_compressed(
                sample_path,
                audio_codes=audio_codes,
                text=text,
            )

            processed_samples.append({
                "idx": idx,
                "path": str(sample_path),
                "text": text,
                "num_frames": audio_codes.shape[0],
                "duration": duration,
            })

        except Exception as e:
            errors.append({"idx": idx, "error": str(e)})
            if len(errors) <= 5:
                print(f"\n⚠️  Error on sample {idx}: {e}")

    # ── Save manifest ────────────────────────────────────────────────
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "dataset": config.dataset_name,
                "num_codebooks": config.num_codebooks,
                "sample_rate": config.sample_rate,
            },
            "samples": processed_samples,
            "errors": errors,
            "total_processed": len(processed_samples),
            "total_errors": len(errors),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Preprocessing complete!")
    print(f"   Processed: {len(processed_samples)}")
    print(f"   Errors:    {len(errors)}")
    print(f"   Saved to:  {output_dir}")

    return processed_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for DSM-ASR")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    args = parser.parse_args()

    config = DsmAsrConfig()
    if args.output_dir:
        config.preprocessed_dir = args.output_dir
    if args.dataset:
        config.dataset_name = args.dataset

    preprocess_dataset(config, max_samples=args.max_samples)

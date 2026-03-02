"""
Timestamp Generation for DSM-ASR

Preprocesses the dataset by:
1. Generating word-level timestamps using whisper-timestamped
2. Pre-encoding audio with Mimi to save audio codes
3. Saving everything as a processed dataset on disk
"""
import os
import json
import argparse
import numpy as np
import torch
import librosa
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Audio, Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


def generate_timestamps_whisper(audio_array: np.ndarray, sample_rate: int, language: str = "ar"):
    """
    Generate word-level timestamps using whisper-timestamped.
    
    Returns list of dicts: [{"word": "...", "start": 0.5, "end": 0.8}, ...]
    """
    import whisper_timestamped as whisper

    # whisper_timestamped expects 16kHz audio
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio_array

    # Load model (cached after first call)
    if not hasattr(generate_timestamps_whisper, "_model"):
        generate_timestamps_whisper._model = whisper.load_model("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")

    model = generate_timestamps_whisper._model

    # Transcribe with word timestamps
    result = whisper.transcribe(
        model,
        audio_16k,
        language=language,
        detect_disfluencies=False,
        task="transcribe",
    )

    word_timestamps = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word_timestamps.append({
                "word": word_info["text"].strip(),
                "start": word_info["start"],
                "end": word_info["end"],
            })

    return word_timestamps


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
    
    Args:
        audio_array: Raw audio waveform
        sample_rate: Original sample rate
        mimi_model: Loaded MimiModel
        feature_extractor: Mimi feature extractor
        target_sr: Target sample rate (24kHz for Mimi)
        num_codebooks: Number of codebooks to use (first Q)
        device: Device to use
    
    Returns:
        audio_codes: np.ndarray of shape [T_frames, Q]
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
        # audio_codes shape: [batch, num_codebooks_total, T_frames]
        codes = encoder_outputs.audio_codes  # [1, 1, 32, T]

    # Take first Q codebooks and reshape to [T, Q]
    codes = codes[0, 0, :num_codebooks, :].T  # [T, Q]
    return codes.cpu().numpy()


def preprocess_dataset(config: DsmAsrConfig, max_samples: int = None, language: str = "ar"):
    """
    Full preprocessing pipeline:
    1. Load HuggingFace dataset
    2. Generate word-level timestamps
    3. Encode audio with Mimi
    4. Save processed data
    """
    print("=" * 60)
    print("DSM-ASR Data Preprocessing")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────
    print(f"\n📥 Loading dataset: {config.dataset_name}")
    ds = load_dataset(config.dataset_name, split=config.train_split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"   Using {len(ds)} samples (limited)")
    else:
        print(f"   Loaded {len(ds)} samples")

    # Ensure audio is decoded
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

            # 1. Generate word timestamps
            word_timestamps = generate_timestamps_whisper(audio_array, sr, language=language)

            if not word_timestamps:
                # Fallback: if whisper can't get timestamps, create a simple uniform alignment
                words = text.strip().split()
                duration = len(audio_array) / sr
                if len(words) > 0:
                    word_duration = duration / len(words)
                    word_timestamps = [
                        {"word": w, "start": i * word_duration, "end": (i + 1) * word_duration}
                        for i, w in enumerate(words)
                    ]
                else:
                    errors.append({"idx": idx, "error": "no words found"})
                    continue

            # 2. Encode audio with Mimi
            audio_codes = encode_audio_with_mimi(
                audio_array, sr, mimi_model, feature_extractor,
                target_sr=config.sample_rate,
                num_codebooks=config.num_codebooks,
                device=device,
            )

            # 3. Save processed sample
            sample_path = output_dir / f"sample_{idx:06d}.npz"
            np.savez_compressed(
                sample_path,
                audio_codes=audio_codes,
                word_timestamps=json.dumps(word_timestamps, ensure_ascii=False),
                text=text,
            )

            processed_samples.append({
                "idx": idx,
                "path": str(sample_path),
                "text": text,
                "num_frames": audio_codes.shape[0],
                "num_words": len(word_timestamps),
                "duration": len(audio_array) / sr,
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
                "delay_frames": config.delay_frames,
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
    print(f"   Manifest:  {manifest_path}")

    return processed_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for DSM-ASR")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--language", type=str, default="ar", help="Language for whisper-timestamped")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    args = parser.parse_args()

    config = DsmAsrConfig()
    if args.output_dir:
        config.preprocessed_dir = args.output_dir
    if args.dataset:
        config.dataset_name = args.dataset

    preprocess_dataset(config, max_samples=args.max_samples, language=args.language)

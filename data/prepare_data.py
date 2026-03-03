"""
DSM-ASR Data Preparation v3 — With Word Timestamps

1. Loads audio from HuggingFace dataset
2. Generates word-level timestamps using whisper-timestamped
3. Pre-encodes audio with Mimi
4. Saves: audio codes + text + word timestamps
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


_whisper_model = None


def get_whisper_model(model_size="small", device="cuda"):
    global _whisper_model
    if _whisper_model is None:
        import whisper_timestamped as whisper
        _whisper_model = whisper.load_model(model_size, device=device)
        print(f"   Whisper '{model_size}' loaded on {device}")
    return _whisper_model


def generate_word_timestamps(audio_array, sr, language="ar", device="cuda"):
    """Generate word-level timestamps using whisper-timestamped."""
    import whisper_timestamped as whisper

    # Whisper needs 16kHz
    if sr != 16000:
        audio_16k = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio_array

    model = get_whisper_model(device=device)
    audio_tensor = torch.from_numpy(audio_16k).float().to(device)

    result = whisper.transcribe(
        model, audio_tensor,
        language=language,
        detect_disfluencies=False,
        vad=True,
    )

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("text", "").strip()
            if text:
                words.append({
                    "word": text,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })

    # Fallback: if no words found, create a single word for the whole text
    if not words and result.get("text", "").strip():
        words = [{"word": result["text"].strip(), "start": 0.0,
                  "end": len(audio_array) / sr}]

    return words


def encode_audio_with_mimi(audio_array, sr, mimi_model, fe,
                           target_sr=24000, num_codebooks=8, device="cuda"):
    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
    inputs = fe(raw_audio=audio_array, sampling_rate=target_sr, return_tensors="pt")
    with torch.no_grad():
        codes = mimi_model.encode(inputs["input_values"].to(device)).audio_codes
        if codes.dim() == 4:
            codes = codes[0, 0, :num_codebooks, :].T
        else:
            codes = codes[0, :num_codebooks, :].T
    return codes.cpu().numpy()


def preprocess_dataset(config, max_samples=None, language="ar"):
    print("=" * 60)
    print("DSM-ASR Preprocessing (with Word Timestamps)")
    print("=" * 60)

    # Load dataset
    print(f"\n📥 Loading: {config.dataset_name}")
    ds = load_dataset(config.dataset_name, split=config.train_split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"   {len(ds)} samples")
    ds = ds.cast_column(config.audio_column, Audio(sampling_rate=config.sample_rate))

    # Load Mimi
    print(f"\n🔊 Loading Mimi...")
    from transformers import MimiModel, AutoFeatureExtractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mimi = MimiModel.from_pretrained(config.mimi_model).to(device)
    mimi.eval()
    fe = AutoFeatureExtractor.from_pretrained(config.mimi_model)

    # Process
    out_dir = Path(config.preprocessed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, errors = [], []

    print(f"\n⚙️  Processing {len(ds)} samples...")
    for idx in tqdm(range(len(ds)), desc="Preprocessing"):
        try:
            s = ds[idx]
            audio_data = s[config.audio_column]
            audio = audio_data["array"]
            sr = audio_data["sampling_rate"]
            text = s[config.text_column]

            if len(audio) < sr * 0.1 or not text or not text.strip():
                errors.append({"idx": idx, "error": "too short/empty"})
                continue

            duration = len(audio) / sr
            if duration > config.max_audio_duration:
                errors.append({"idx": idx, "error": f"too long: {duration:.1f}s"})
                continue

            # Word timestamps
            words = generate_word_timestamps(audio, sr, language=language, device=device)

            # Mimi encoding
            codes = encode_audio_with_mimi(audio, sr, mimi, fe,
                                           target_sr=config.sample_rate,
                                           num_codebooks=config.num_codebooks,
                                           device=device)

            # Save
            path = out_dir / f"sample_{idx:06d}.npz"
            word_starts = [w["start"] for w in words]
            word_ends = [w["end"] for w in words]
            word_texts = [w["word"] for w in words]

            np.savez_compressed(path,
                                audio_codes=codes,
                                text=text,
                                word_starts=np.array(word_starts, dtype=np.float32),
                                word_ends=np.array(word_ends, dtype=np.float32),
                                word_texts=np.array(word_texts, dtype=object))

            samples.append({
                "idx": idx, "path": str(path), "text": text,
                "num_frames": codes.shape[0], "num_words": len(words),
                "duration": duration,
            })

        except Exception as e:
            errors.append({"idx": idx, "error": str(e)})
            if len(errors) <= 5:
                print(f"\n⚠️  Error {idx}: {e}")

    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"config": {"dataset": config.dataset_name,
                               "num_codebooks": config.num_codebooks},
                    "samples": samples, "errors": errors,
                    "total_processed": len(samples),
                    "total_errors": len(errors)}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Processed: {len(samples)}, Errors: {len(errors)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--language", type=str, default="ar")
    args = parser.parse_args()

    config = DsmAsrConfig()
    if args.output_dir:
        config.preprocessed_dir = args.output_dir
    preprocess_dataset(config, args.max_samples, args.language)

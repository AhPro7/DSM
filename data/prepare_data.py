"""
DSM-ASR Data Preparation v3 — FAST

Speed optimizations:
1. Batch Mimi encoding (process N samples at once on GPU)
2. Optional whisper timestamps (--no_timestamps for fast runs)
3. Smaller whisper model option (--whisper_model tiny)
4. Multiprocessing for I/O
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
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


def encode_batch_mimi(audio_arrays, srs, mimi_model, fe,
                      target_sr=24000, num_codebooks=8, device="cuda"):
    """Encode a batch of audio arrays with Mimi at once."""
    resampled = []
    for audio, sr in zip(audio_arrays, srs):
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        resampled.append(audio)

    # Pad to same length for batched processing
    max_len = max(len(a) for a in resampled)
    padded = np.zeros((len(resampled), max_len), dtype=np.float32)
    lengths = []
    for i, a in enumerate(resampled):
        padded[i, :len(a)] = a
        lengths.append(len(a))

    inputs = fe(raw_audio=list(padded), sampling_rate=target_sr, return_tensors="pt",
                padding=True)
    with torch.no_grad():
        codes = mimi_model.encode(inputs["input_values"].to(device)).audio_codes

    results = []
    for i in range(len(resampled)):
        # Calculate expected frames for this sample's actual length
        expected_frames = int(lengths[i] / target_sr * 12.5) + 1
        if codes.dim() == 4:
            c = codes[i, 0, :num_codebooks, :expected_frames].T
        else:
            c = codes[i, :num_codebooks, :expected_frames].T
        results.append(c.cpu().numpy())
    return results


def generate_timestamps_batch(audio_arrays, srs, language="ar",
                              whisper_model_size="tiny", device="cuda"):
    """Generate word timestamps for multiple samples."""
    import whisper_timestamped as whisper

    model = whisper.load_model(whisper_model_size, device=device)
    results = []

    for audio, sr in zip(audio_arrays, srs):
        try:
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio
            audio_t = torch.from_numpy(audio_16k).float().to(device)
            result = whisper.transcribe(model, audio_t, language=language,
                                        detect_disfluencies=False, vad=True)
            words = []
            for seg in result.get("segments", []):
                for w in seg.get("words", []):
                    t = w.get("text", "").strip()
                    if t:
                        words.append({"word": t, "start": float(w["start"]),
                                      "end": float(w["end"])})
            if not words and result.get("text", "").strip():
                words = [{"word": result["text"].strip(), "start": 0.0,
                          "end": len(audio) / sr}]
            results.append(words)
        except Exception as e:
            results.append([])
    return results


def save_sample(args):
    """Save a single sample to disk (for ThreadPoolExecutor)."""
    path, codes, text, words = args
    word_starts = np.array([w["start"] for w in words], dtype=np.float32) if words else np.array([], dtype=np.float32)
    word_ends = np.array([w["end"] for w in words], dtype=np.float32) if words else np.array([], dtype=np.float32)
    word_texts = np.array([w["word"] for w in words], dtype=object) if words else np.array([], dtype=object)
    np.savez_compressed(path, audio_codes=codes, text=text,
                         word_starts=word_starts, word_ends=word_ends,
                         word_texts=word_texts)
    return True


def preprocess_dataset(config, max_samples=None, language="ar",
                       use_timestamps=True, whisper_model="tiny",
                       batch_size=8):
    print("=" * 60)
    print("DSM-ASR Preprocessing (FAST)")
    print(f"  Timestamps: {'YES (' + whisper_model + ')' if use_timestamps else 'NO (even spread)'}")
    print(f"  Batch size: {batch_size}")
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

    out_dir = Path(config.preprocessed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, errors = [], []

    # Process in batches
    total = len(ds)
    num_batches = (total + batch_size - 1) // batch_size

    # Thread pool for saving files
    saver = ThreadPoolExecutor(max_workers=4)
    save_futures = []

    print(f"\n⚙️  Processing {total} samples in {num_batches} batches...")
    pbar = tqdm(total=total, desc="Preprocessing")

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = list(range(batch_start, batch_end))

        # Collect batch data
        batch_audios, batch_srs, batch_texts = [], [], []
        valid_indices = []

        for idx in batch_indices:
            try:
                s = ds[idx]
                audio_data = s[config.audio_column]
                audio = audio_data["array"]
                sr = audio_data["sampling_rate"]
                text = s[config.text_column]

                if len(audio) < sr * 0.1 or not text or not text.strip():
                    errors.append({"idx": idx, "error": "too short/empty"})
                    pbar.update(1)
                    continue

                duration = len(audio) / sr
                if duration > config.max_audio_duration:
                    errors.append({"idx": idx, "error": f"too long: {duration:.1f}s"})
                    pbar.update(1)
                    continue

                batch_audios.append(audio)
                batch_srs.append(sr)
                batch_texts.append(text)
                valid_indices.append(idx)
            except Exception as e:
                errors.append({"idx": idx, "error": str(e)})
                pbar.update(1)

        if not batch_audios:
            continue

        # Batch Mimi encoding (FAST!)
        try:
            all_codes = encode_batch_mimi(
                batch_audios, batch_srs, mimi, fe,
                target_sr=config.sample_rate,
                num_codebooks=config.num_codebooks, device=device)
        except Exception as e:
            # Fallback to one-by-one
            all_codes = []
            for audio, sr in zip(batch_audios, batch_srs):
                try:
                    resampled = librosa.resample(audio, orig_sr=sr, target_sr=config.sample_rate) if sr != config.sample_rate else audio
                    inp = fe(raw_audio=resampled, sampling_rate=config.sample_rate, return_tensors="pt")
                    with torch.no_grad():
                        c = mimi.encode(inp["input_values"].to(device)).audio_codes
                        if c.dim() == 4:
                            c = c[0, 0, :config.num_codebooks, :].T
                        else:
                            c = c[0, :config.num_codebooks, :].T
                    all_codes.append(c.cpu().numpy())
                except:
                    all_codes.append(None)

        # Timestamps (if requested)
        if use_timestamps:
            all_words = generate_timestamps_batch(
                batch_audios, batch_srs, language=language,
                whisper_model_size=whisper_model, device=device)
        else:
            all_words = [[] for _ in batch_audios]

        # Save results async
        for i, idx in enumerate(valid_indices):
            if i >= len(all_codes) or all_codes[i] is None:
                errors.append({"idx": idx, "error": "encoding failed"})
                pbar.update(1)
                continue

            codes = all_codes[i]
            text = batch_texts[i]
            words = all_words[i] if i < len(all_words) else []
            duration = len(batch_audios[i]) / batch_srs[i]

            path = out_dir / f"sample_{idx:06d}.npz"
            save_futures.append(
                saver.submit(save_sample, (str(path), codes, text, words)))

            samples.append({
                "idx": idx, "path": str(path), "text": text,
                "num_frames": codes.shape[0], "num_words": len(words),
                "duration": duration,
            })
            pbar.update(1)

    pbar.close()

    # Wait for all saves
    for f in save_futures:
        f.result()
    saver.shutdown()

    # Save manifest
    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"config": {"dataset": config.dataset_name,
                               "num_codebooks": config.num_codebooks,
                               "timestamps": use_timestamps},
                    "samples": samples, "errors": errors,
                    "total_processed": len(samples),
                    "total_errors": len(errors)}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Processed: {len(samples)}, Errors: {len(errors)}")
    if use_timestamps:
        avg_words = sum(s["num_words"] for s in samples) / max(len(samples), 1)
        print(f"   Avg words/sample: {avg_words:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--language", type=str, default="ar")
    parser.add_argument("--no_timestamps", action="store_true",
                        help="Skip whisper timestamps (much faster)")
    parser.add_argument("--whisper_model", type=str, default="tiny",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (tiny is fastest)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for Mimi encoding")
    args = parser.parse_args()

    config = DsmAsrConfig()
    if args.output_dir:
        config.preprocessed_dir = args.output_dir
    preprocess_dataset(config, args.max_samples, args.language,
                       use_timestamps=not args.no_timestamps,
                       whisper_model=args.whisper_model,
                       batch_size=args.batch_size)

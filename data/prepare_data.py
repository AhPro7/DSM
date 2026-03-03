"""
DSM-ASR Preprocessing — Fast Mimi-Only

Encodes audio with Mimi in batches. No timestamps.
Saves: audio_codes [T, Q] + text string
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


def save_sample(args):
    path, codes, text = args
    np.savez_compressed(path, audio_codes=codes, text=text)


def preprocess_dataset(config, max_samples=None, batch_size=8):
    print("=" * 60)
    print("DSM-ASR Preprocessing (Mimi encoding)")
    print("=" * 60)

    ds = load_dataset(config.dataset_name, split=config.train_split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    print(f"📥 {len(ds)} samples")
    ds = ds.cast_column(config.audio_column, Audio(sampling_rate=config.sample_rate))

    from transformers import MimiModel, AutoFeatureExtractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mimi = MimiModel.from_pretrained(config.mimi_model).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(config.mimi_model)

    out_dir = Path(config.preprocessed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, errors = [], []
    saver = ThreadPoolExecutor(max_workers=4)
    futures = []

    pbar = tqdm(range(0, len(ds), batch_size), desc="Encoding")
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(ds))

        batch_audios, batch_texts, batch_indices = [], [], []
        for idx in range(batch_start, batch_end):
            try:
                s = ds[idx]
                audio = s[config.audio_column]["array"]
                sr = s[config.audio_column]["sampling_rate"]
                text = s[config.text_column]

                if len(audio) < sr * 0.1 or not text or not text.strip():
                    errors.append({"idx": idx, "error": "too short/empty"})
                    continue
                if len(audio) / sr > config.max_audio_duration:
                    errors.append({"idx": idx, "error": "too long"})
                    continue

                if sr != config.sample_rate:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=config.sample_rate)
                batch_audios.append(audio)
                batch_texts.append(text)
                batch_indices.append(idx)
            except Exception as e:
                errors.append({"idx": idx, "error": str(e)})

        if not batch_audios:
            continue

        # Encode batch
        try:
            max_len = max(len(a) for a in batch_audios)
            padded = np.zeros((len(batch_audios), max_len), dtype=np.float32)
            lens = []
            for i, a in enumerate(batch_audios):
                padded[i, :len(a)] = a
                lens.append(len(a))

            inputs = fe(raw_audio=list(padded), sampling_rate=config.sample_rate,
                       return_tensors="pt", padding=True)
            with torch.no_grad():
                all_codes = mimi.encode(inputs["input_values"].to(device)).audio_codes

            for i, idx in enumerate(batch_indices):
                exp_frames = int(lens[i] / config.sample_rate * config.frame_rate) + 1
                if all_codes.dim() == 4:
                    c = all_codes[i, 0, :config.num_codebooks, :exp_frames].T
                else:
                    c = all_codes[i, :config.num_codebooks, :exp_frames].T
                codes_np = c.cpu().numpy()

                path = out_dir / f"sample_{idx:06d}.npz"
                futures.append(saver.submit(save_sample,
                    (str(path), codes_np, batch_texts[i])))
                samples.append({
                    "idx": idx, "path": str(path), "text": batch_texts[i],
                    "num_frames": codes_np.shape[0],
                    "duration": lens[i] / config.sample_rate,
                })
        except Exception as e:
            # Fallback: one by one
            for i, idx in enumerate(batch_indices):
                try:
                    inp = fe(raw_audio=batch_audios[i],
                            sampling_rate=config.sample_rate, return_tensors="pt")
                    with torch.no_grad():
                        c = mimi.encode(inp["input_values"].to(device)).audio_codes
                        if c.dim() == 4:
                            c = c[0, 0, :config.num_codebooks, :].T
                        else:
                            c = c[0, :config.num_codebooks, :].T

                    path = out_dir / f"sample_{idx:06d}.npz"
                    codes_np = c.cpu().numpy()
                    futures.append(saver.submit(save_sample,
                        (str(path), codes_np, batch_texts[i])))
                    samples.append({
                        "idx": idx, "path": str(path), "text": batch_texts[i],
                        "num_frames": codes_np.shape[0],
                        "duration": len(batch_audios[i]) / config.sample_rate,
                    })
                except Exception as e2:
                    errors.append({"idx": idx, "error": str(e2)})

        pbar.set_postfix(ok=len(samples), err=len(errors))

    for f in futures:
        f.result()
    saver.shutdown()

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "errors": errors,
                    "total_processed": len(samples),
                    "total_errors": len(errors)}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! OK: {len(samples)}, Errors: {len(errors)}")
    if samples:
        avg = sum(s["duration"] for s in samples) / len(samples)
        total = sum(s["duration"] for s in samples) / 3600
        print(f"   Avg duration: {avg:.1f}s, Total: {total:.1f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    config = DsmAsrConfig()
    preprocess_dataset(config, args.max_samples, args.batch_size)

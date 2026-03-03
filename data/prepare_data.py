"""
DSM-ASR Preprocessing v5 — SODA-Style

Encodes audio with Mimi and packs 8 codebooks per frame into
8 consecutive flat integer token IDs saved as a 1D int32 array.

flat_id = cb * 2048 + value   for cb in 0..7

No separate Q dimension — everything is a 1D token stream
that can be directly concatenated with text token IDs.
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


def mimi_codes_to_flat_tokens(codes_np, config: DsmAsrConfig):
    """
    codes_np: [T, Q] int array (Mimi output)
    returns:  [T*Q] int array of flat audio token IDs
    """
    T, Q = codes_np.shape
    flat = np.zeros(T * Q, dtype=np.int32)
    for q in range(Q):
        offset = q * config.audio_codebook_size  # q * 2048
        flat[q::Q] = codes_np[:, q] + offset     # interleave per frame
    return flat   # raw flat ids (0..16383), caller adds audio_token_offset


def save_sample(args):
    path, flat_tokens, text = args
    np.savez_compressed(path, audio_flat=flat_tokens, text=text)


def encode_batch(batch_audios, batch_srs, mimi, fe, config, device):
    """Batch Mimi encode → list of [T*Q] flat token arrays."""
    sr = config.sample_rate
    resampled = [librosa.resample(a, orig_sr=s, target_sr=sr) if s != sr else a
                 for a, s in zip(batch_audios, batch_srs)]
    max_len = max(len(a) for a in resampled)
    padded = np.zeros((len(resampled), max_len), dtype=np.float32)
    lens = []
    for i, a in enumerate(resampled):
        padded[i, :len(a)] = a
        lens.append(len(a))

    inp = fe(raw_audio=list(padded), sampling_rate=sr,
             return_tensors="pt", padding=True)
    with torch.no_grad():
        codes = mimi.encode(inp["input_values"].to(device)).audio_codes

    results = []
    for i in range(len(resampled)):
        exp_frames = int(lens[i] / sr * config.frame_rate) + 1
        if codes.dim() == 4:
            c = codes[i, 0, :config.num_codebooks, :exp_frames].T  # [T, Q]
        else:
            c = codes[i, :config.num_codebooks, :exp_frames].T
        flat = mimi_codes_to_flat_tokens(c.cpu().numpy(), config)
        results.append(flat)
    return results


def preprocess_dataset(config, max_samples=None, batch_size=8):
    print("=" * 60)
    print("DSM-ASR Preprocessing v5 — SODA Interleaved Format")
    print(f"  Audio tokens: {config.audio_vocab_size} ({config.num_codebooks}cb × {config.audio_codebook_size})")
    print(f"  Offset: +{config.audio_token_offset} in combined vocab")
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
        batch_srs, batch_durs = [], []

        for idx in range(batch_start, batch_end):
            try:
                s = ds[idx]
                audio = s[config.audio_column]["array"]
                sr = s[config.audio_column]["sampling_rate"]
                text = s[config.text_column]
                dur = len(audio) / sr

                if len(audio) < sr * 0.1 or not text or not text.strip():
                    errors.append({"idx": idx, "error": "too short/empty"})
                    continue
                if dur > config.max_audio_duration:
                    errors.append({"idx": idx, "error": f"too long: {dur:.1f}s"})
                    continue

                batch_audios.append(audio)
                batch_srs.append(sr)
                batch_texts.append(text)
                batch_indices.append(idx)
                batch_durs.append(dur)
            except Exception as e:
                errors.append({"idx": idx, "error": str(e)})

        if not batch_audios:
            continue

        try:
            all_flat = encode_batch(batch_audios, batch_srs, mimi, fe, config, device)
        except Exception as e:
            # One-by-one fallback
            all_flat = []
            for audio, sr in zip(batch_audios, batch_srs):
                try:
                    rs = librosa.resample(audio, orig_sr=sr, target_sr=config.sample_rate) if sr != config.sample_rate else audio
                    inp = fe(raw_audio=rs, sampling_rate=config.sample_rate, return_tensors="pt")
                    with torch.no_grad():
                        c = mimi.encode(inp["input_values"].to(device)).audio_codes
                        if c.dim() == 4:
                            c = c[0, 0, :config.num_codebooks, :].T
                        else:
                            c = c[0, :config.num_codebooks, :].T
                    flat = mimi_codes_to_flat_tokens(c.cpu().numpy(), config)
                    all_flat.append(flat)
                except Exception as e2:
                    all_flat.append(None)

        for i, idx in enumerate(batch_indices):
            if i >= len(all_flat) or all_flat[i] is None:
                errors.append({"idx": idx, "error": "encoding failed"})
                continue
            flat = all_flat[i]
            path = out_dir / f"sample_{idx:06d}.npz"
            futures.append(saver.submit(save_sample,
                (str(path), flat, batch_texts[i])))
            samples.append({
                "idx": idx, "path": str(path), "text": batch_texts[i],
                "num_audio_tokens": len(flat),  # T*Q
                "num_frames": len(flat) // config.num_codebooks,
                "duration": batch_durs[i],
            })
        pbar.set_postfix(ok=len(samples), err=len(errors))

    for f in futures:
        f.result()
    saver.shutdown()

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "errors": errors,
                    "total_processed": len(samples),
                    "total_errors": len(errors)}, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! OK:{len(samples)}  Errors:{len(errors)}")
    if samples:
        avg_a = sum(s["num_audio_tokens"] for s in samples) / len(samples)
        avg_d = sum(s["duration"] for s in samples) / len(samples)
        print(f"   Avg audio tokens/sample: {avg_a:.0f} ({avg_d:.1f}s)")
        print(f"   Example: {samples[0]['text'][:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    config = DsmAsrConfig()
    preprocess_dataset(config, args.max_samples, args.batch_size)

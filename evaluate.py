"""
DSM-ASR Evaluation

Evaluates the trained model on the eval split.
Reports WER, CER, and RTF.

Usage:
    python evaluate.py --checkpoint output/best
    python evaluate.py --checkpoint output/best --max_samples 100
"""
import os
import sys
import json
import argparse
import time
import re
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from data.dataset import DsmAsrDataset
from train import load_checkpoint


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for evaluation."""
    diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')
    text = diacritics.sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه')
    text = text.replace('ـ', '')
    text = ' '.join(text.split())
    return text.strip()


def evaluate_model(model, tokenizer, config, max_samples=None, device="cuda"):
    """Full evaluation pipeline."""
    from jiwer import wer, cer

    print("\n📊 Running evaluation...")

    eval_ds = DsmAsrDataset(config, split="eval", tokenizer=tokenizer, max_samples=max_samples)
    if len(eval_ds) == 0:
        print("⚠️  No eval samples")
        return {}

    start_text_id = tokenizer.convert_tokens_to_ids("<|start_text|>")
    end_text_id = tokenizer.convert_tokens_to_ids("<|end_text|>")

    predictions, references = [], []
    durations, decode_times = [], []
    model.eval()

    for idx in tqdm(range(len(eval_ds)), desc="Evaluating"):
        sample = eval_ds[idx]
        sample_info = eval_ds.samples[idx]
        audio_codes = sample["audio_tokens"].to(device)
        audio_tokens = audio_codes.unsqueeze(0)

        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(audio_tokens, tokenizer,
                                       max_new_tokens=config.max_text_tokens)
        dt = time.time() - start_time

        text_tokens = [t for t in generated if t != start_text_id and t != end_text_id]
        pred = tokenizer.decode(text_tokens, skip_special_tokens=True).strip()

        data = np.load(sample_info["path"], allow_pickle=True)
        ref = str(data["text"]).strip()

        predictions.append(normalize_arabic(pred))
        references.append(normalize_arabic(ref))
        durations.append(sample_info.get("duration", audio_codes.shape[0] / config.frame_rate))
        decode_times.append(dt)

    # Filter empty
    valid = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid:
        return {"wer": -1, "cer": -1, "num_valid": 0}

    preds, refs = zip(*valid)
    overall_wer = wer(list(refs), list(preds))
    overall_cer = cer(list(refs), list(preds))

    per_sample = []
    for p, r in zip(preds, refs):
        per_sample.append({
            "prediction": p, "reference": r,
            "wer": round(wer([r], [p]), 4), "cer": round(cer([r], [p]), 4),
        })

    total_audio = sum(durations)
    total_decode = sum(decode_times)

    return {
        "wer": round(overall_wer, 4),
        "cer": round(overall_cer, 4),
        "num_valid": len(valid),
        "avg_rtf": round(total_decode / max(total_audio, 0.001), 4),
        "total_audio_seconds": round(total_audio, 2),
        "total_decode_seconds": round(total_decode, 2),
        "per_sample": per_sample,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DSM-ASR")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)
    metrics = evaluate_model(model, tokenizer, config, max_samples=args.max_samples, device=device)

    print("\n" + "=" * 60)
    print(f"  WER: {metrics.get('wer', 'N/A')}")
    print(f"  CER: {metrics.get('cer', 'N/A')}")
    print(f"  Samples: {metrics.get('num_valid', 0)}")
    print(f"  RTF: {metrics.get('avg_rtf', 'N/A')}")

    for i, s in enumerate(metrics.get("per_sample", [])[:5]):
        print(f"\n  [{i+1}] WER={s['wer']:.2f}  REF: {s['reference'][:60]}...")
        print(f"              HYP: {s['prediction'][:60]}...")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved: {args.output}")

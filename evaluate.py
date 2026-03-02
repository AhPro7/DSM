"""
DSM-ASR Evaluation Script

Evaluates the trained DSM-ASR model on a dataset.
Reports WER, CER, and RTF metrics.

Usage:
    python evaluate.py --checkpoint output/best
    python evaluate.py --checkpoint output/best --max_samples 100 --fast
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from data.dataset import DsmAsrDataset
from data.collator import DsmAsrCollator
from train import load_checkpoint
from inference import transcribe_fast


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for evaluation.
    Removes diacritics, normalizes alef variants, etc.
    """
    import re

    # Remove Arabic diacritics (tashkeel)
    diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]')
    text = diacritics.sub('', text)

    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)

    # Normalize teh marbuta
    text = text.replace('ة', 'ه')

    # Normalize tatweel
    text = text.replace('ـ', '')

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text.strip()


def compute_metrics(predictions: list, references: list, normalize: bool = True):
    """
    Compute WER and CER.

    Args:
        predictions: List of predicted texts
        references: List of reference texts
        normalize: Whether to normalize Arabic text

    Returns:
        dict with wer, cer, and per-sample details
    """
    from jiwer import wer, cer

    if normalize:
        predictions = [normalize_arabic(p) for p in predictions]
        references = [normalize_arabic(r) for r in references]

    # Filter empty pairs
    valid_pairs = [(p, r) for p, r in zip(predictions, references)
                   if p.strip() and r.strip()]

    if not valid_pairs:
        return {"wer": -1, "cer": -1, "num_valid": 0}

    preds, refs = zip(*valid_pairs)
    preds, refs = list(preds), list(refs)

    overall_wer = wer(refs, preds)
    overall_cer = cer(refs, preds)

    # Per-sample metrics
    per_sample = []
    for p, r in zip(preds, refs):
        sample_wer = wer([r], [p])
        sample_cer = cer([r], [p])
        per_sample.append({
            "prediction": p,
            "reference": r,
            "wer": round(sample_wer, 4),
            "cer": round(sample_cer, 4),
        })

    return {
        "wer": round(overall_wer, 4),
        "cer": round(overall_cer, 4),
        "num_valid": len(valid_pairs),
        "per_sample": per_sample,
    }


def evaluate_model(
    model,
    tokenizer,
    config: DsmAsrConfig,
    max_samples: int = None,
    device: str = "cuda",
    use_fast: bool = True,
):
    """
    Full evaluation pipeline.

    Args:
        model: Trained DsmAsrModel
        tokenizer: Tokenizer
        config: Config
        max_samples: Limit number of samples
        device: Device
        use_fast: Use fast mode (single forward pass)

    Returns:
        dict with all metrics
    """
    print("\n📊 Running evaluation...")

    # Load eval dataset
    eval_ds = DsmAsrDataset(config, split="eval", tokenizer=tokenizer,
                            max_samples=max_samples)

    if len(eval_ds) == 0:
        print("⚠️  No eval samples found")
        return {}

    predictions = []
    references = []
    durations = []
    decode_times = []
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    word_token_id = tokenizer.convert_tokens_to_ids("<|word|>")

    collator = DsmAsrCollator(config=config, pad_token_id=pad_token_id)

    model.eval()

    for idx in tqdm(range(len(eval_ds)), desc="Evaluating"):
        sample = eval_ds[idx]
        sample_info = eval_ds.samples[idx]

        audio_codes = sample["audio_tokens"].to(device)
        T, Q = audio_codes.shape

        start_time = time.time()

        if use_fast:
            # Fast mode: single forward pass
            audio_tokens = audio_codes.unsqueeze(0)  # [1, T, Q]
            text_tokens = torch.full((1, T), pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.ones(1, T, device=device)
            text_loss_mask = torch.zeros(1, T, device=device)

            with torch.no_grad():
                output = model(audio_tokens, text_tokens, attention_mask, text_loss_mask)

            pred_ids = output.logits[0].argmax(dim=-1).cpu().tolist()
            gen_tokens = [t for t in pred_ids
                         if t != pad_token_id and t != word_token_id and t >= 0]
            pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        else:
            from inference import transcribe
            result = transcribe(model, tokenizer, config,
                              audio_codes=audio_codes, device=device)
            pred_text = result["text"]

        dt = time.time() - start_time

        # Get reference text
        data = np.load(sample_info["path"], allow_pickle=True)
        ref_text = str(data["text"]).strip()

        predictions.append(pred_text)
        references.append(ref_text)
        duration = sample_info.get("duration", T / config.frame_rate)
        durations.append(duration)
        decode_times.append(dt)

    # Compute metrics
    metrics = compute_metrics(predictions, references, normalize=True)

    # Add timing metrics
    total_audio = sum(durations)
    total_decode = sum(decode_times)
    metrics["total_audio_seconds"] = round(total_audio, 2)
    metrics["total_decode_seconds"] = round(total_decode, 2)
    metrics["avg_rtf"] = round(total_decode / max(total_audio, 0.001), 4)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DSM-ASR model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="Use fast eval mode")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading: {args.checkpoint}")
    model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)

    # Evaluate
    metrics = evaluate_model(
        model, tokenizer, config,
        max_samples=args.max_samples,
        device=device,
        use_fast=args.fast,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  WER: {metrics.get('wer', 'N/A')}")
    print(f"  CER: {metrics.get('cer', 'N/A')}")
    print(f"  Samples: {metrics.get('num_valid', 0)}")
    print(f"  Avg RTF: {metrics.get('avg_rtf', 'N/A')}")
    print(f"  Total audio: {metrics.get('total_audio_seconds', 0):.1f}s")
    print(f"  Total decode: {metrics.get('total_decode_seconds', 0):.1f}s")

    # Show some examples
    per_sample = metrics.get("per_sample", [])
    if per_sample:
        print(f"\n📝 Sample predictions (first 5):")
        for i, s in enumerate(per_sample[:5]):
            print(f"\n  [{i+1}] WER={s['wer']:.2f} CER={s['cer']:.2f}")
            print(f"      REF: {s['reference'][:80]}...")
            print(f"      HYP: {s['prediction'][:80]}...")

    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output}")

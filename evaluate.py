"""
DSM-ASR Evaluation v3
"""
import os, sys, json, argparse, time, re
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from data.dataset import DsmAsrDataset
from train import load_checkpoint


def normalize_arabic(text):
    text = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]').sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ـ', '')
    return ' '.join(text.split()).strip()


def evaluate_model(model, tokenizer, config, max_samples=None, device="cuda"):
    from jiwer import wer, cer
    eval_ds = DsmAsrDataset(config, "eval", tokenizer=tokenizer, max_samples=max_samples)
    if not eval_ds:
        return {}

    preds, refs, dts, durs = [], [], [], []
    model.eval()

    for idx in tqdm(range(len(eval_ds)), desc="Evaluating"):
        s = eval_ds[idx]
        info = eval_ds.samples[idx]
        audio = s["audio_tokens"].unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            gen = model.generate_text(audio, tokenizer)
        dt = time.time() - t0

        pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
        ref = str(np.load(info["path"], allow_pickle=True)["text"]).strip()

        preds.append(normalize_arabic(pred))
        refs.append(normalize_arabic(ref))
        durs.append(info.get("duration", audio.shape[1] / config.frame_rate))
        dts.append(dt)

    valid = [(p, r) for p, r in zip(preds, refs) if p and r]
    if not valid:
        return {"wer": -1, "cer": -1}
    ps, rs = zip(*valid)
    per_sample = [{"prediction": p, "reference": r,
                   "wer": round(wer([r], [p]), 4), "cer": round(cer([r], [p]), 4)}
                  for p, r in zip(ps, rs)]

    return {
        "wer": round(wer(list(rs), list(ps)), 4),
        "cer": round(cer(list(rs), list(ps)), 4),
        "num_valid": len(valid),
        "avg_rtf": round(sum(dts) / max(sum(durs), 0.001), 4),
        "per_sample": per_sample,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    metrics = evaluate_model(model, tokenizer, config, args.max_samples, device)

    print(f"\n  WER: {metrics.get('wer')}  CER: {metrics.get('cer')}")
    print(f"  Samples: {metrics.get('num_valid', 0)}  RTF: {metrics.get('avg_rtf')}")
    for s in metrics.get("per_sample", [])[:5]:
        print(f"\n  WER={s['wer']:.2f}  REF: {s['reference'][:60]}")
        print(f"            HYP: {s['prediction'][:60]}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

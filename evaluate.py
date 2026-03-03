"""DSM-ASR Evaluation v4"""
import os, sys, json, argparse, time, re
import torch, numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from data.dataset import DsmAsrDataset
from train import load_checkpoint


def normalize_ar(t):
    t = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]').sub('', t)
    t = re.sub(r'[إأآا]', 'ا', t)
    return ' '.join(t.replace('ة','ه').replace('ـ','').split()).strip()


def evaluate_model(model, tokenizer, config, max_samples=None, device="cuda"):
    from jiwer import wer, cer
    ds = DsmAsrDataset(config, "eval", tokenizer=tokenizer, max_samples=max_samples)
    preds, refs, dts, durs = [], [], [], []
    model.eval()

    for i in tqdm(range(len(ds)), desc="Evaluating"):
        s = ds[i]
        audio = s["audio_tokens"].unsqueeze(0).to(device)
        ref = tokenizer.decode(s["target_ids"], skip_special_tokens=True).strip()

        t0 = time.time()
        with torch.no_grad():
            pred = model.generate(audio, tokenizer)
        dt = time.time() - t0

        preds.append(normalize_ar(pred))
        refs.append(normalize_ar(ref))
        durs.append(ds.samples[i].get("duration", audio.shape[1] / config.frame_rate))
        dts.append(dt)

    valid = [(p, r) for p, r in zip(preds, refs) if p and r]
    if not valid:
        return {"wer": -1, "cer": -1}
    ps, rs = zip(*valid)
    per = [{"prediction": p, "reference": r,
            "wer": round(wer([r],[p]),4), "cer": round(cer([r],[p]),4)}
           for p, r in zip(ps, rs)]
    return {
        "wer": round(wer(list(rs), list(ps)), 4),
        "cer": round(cer(list(rs), list(ps)), 4),
        "num_valid": len(valid),
        "avg_rtf": round(sum(dts)/max(sum(durs),0.001), 4),
        "total_audio_seconds": round(sum(durs), 2),
        "per_sample": per,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, config = load_checkpoint(args.checkpoint, device)
    m = evaluate_model(model, tok, config, args.max_samples, device)

    print(f"\n  WER: {m['wer']}  CER: {m['cer']}  Samples: {m['num_valid']}")
    for s in m.get("per_sample", [])[:10]:
        print(f"\n  WER={s['wer']:.2f}  REF: {s['reference'][:60]}")
        print(f"            HYP: {s['prediction'][:60]}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

"""DSM-ASR Inference v5 — SODA-Style"""
import os, sys, time, argparse
import torch, librosa, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from train import load_checkpoint
from data.prepare_data import mimi_codes_to_flat_tokens


def encode_audio(audio_path, config: DsmAsrConfig, device="cuda"):
    from transformers import MimiModel, AutoFeatureExtractor
    audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    mimi = MimiModel.from_pretrained(config.mimi_model).to(device).eval()
    fe   = AutoFeatureExtractor.from_pretrained(config.mimi_model)
    inp  = fe(raw_audio=audio, sampling_rate=config.sample_rate, return_tensors="pt")
    with torch.no_grad():
        codes = mimi.encode(inp["input_values"].to(device)).audio_codes
        if codes.dim() == 4:
            codes = codes[0, 0, :config.num_codebooks, :].T   # [T, Q]
        else:
            codes = codes[0, :config.num_codebooks, :].T
    flat = mimi_codes_to_flat_tokens(codes.cpu().numpy(), config)
    flat_vocab = torch.tensor(flat + config.audio_token_offset, dtype=torch.long)
    return flat_vocab, len(audio) / config.sample_rate


def transcribe(model, tokenizer, config, audio_path, temperature=0.0, device="cuda"):
    model.eval()
    flat_vocab, dur = encode_audio(audio_path, config, device)
    t0 = time.time()
    text = model.generate(flat_vocab.to(device), tokenizer, temperature=temperature)
    dt = time.time() - t0
    return {"text": text, "duration": dur, "decode_time": dt,
            "rtf": dt / max(dur, 0.001)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, config = load_checkpoint(args.checkpoint, device)
    r = transcribe(model, tok, config, args.audio, args.temperature, device)
    print(f"\n📝 {r['text']}")
    print(f"📊 {r['duration']:.1f}s audio, RTF={r['rtf']:.3f}")

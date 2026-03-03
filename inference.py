"""
DSM-ASR Inference v3 — Teacher-force audio, generate text
"""
import os
import sys
import time
import argparse
import torch
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from train import load_checkpoint


def encode_audio_mimi(audio_path, sample_rate=24000, num_codebooks=8, device="cuda"):
    from transformers import MimiModel, AutoFeatureExtractor
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(audio) / sample_rate
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)
    mimi.eval()
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    inputs = fe(raw_audio=audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        codes = mimi.encode(inputs["input_values"].to(device)).audio_codes
        if codes.dim() == 4:
            codes = codes[0, 0, :num_codebooks, :].T
        else:
            codes = codes[0, :num_codebooks, :].T
    return codes, duration


def transcribe(model, tokenizer, config, audio_path=None, audio_codes=None,
               temperature=0.0, device="cuda"):
    model.eval()
    if audio_codes is None:
        audio_codes, duration = encode_audio_mimi(
            audio_path, config.sample_rate, config.num_codebooks, device)
    else:
        duration = audio_codes.shape[0] / config.frame_rate
    audio_tokens = audio_codes.unsqueeze(0).to(device)

    start = time.time()
    gen = model.generate_text(audio_tokens, tokenizer, temperature=temperature)
    dt = time.time() - start

    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return {"text": text, "duration": duration, "decode_time": dt,
            "rtf": dt / max(duration, 0.001), "num_frames": audio_codes.shape[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    result = transcribe(model, tokenizer, config, args.audio, device=device,
                       temperature=args.temperature)
    print(f"\n📝 {result['text']}")
    print(f"📊 {result['duration']:.1f}s audio, RTF={result['rtf']:.3f}")

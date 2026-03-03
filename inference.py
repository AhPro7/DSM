"""DSM-ASR Inference v4"""
import os, sys, time, argparse
import torch, librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from train import load_checkpoint


def encode_audio(path, sr=24000, num_codebooks=8, device="cuda"):
    from transformers import MimiModel, AutoFeatureExtractor
    audio, _ = librosa.load(path, sr=sr, mono=True)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    inp = fe(raw_audio=audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        codes = mimi.encode(inp["input_values"].to(device)).audio_codes
        if codes.dim() == 4:
            codes = codes[0, 0, :num_codebooks, :].T
        else:
            codes = codes[0, :num_codebooks, :].T
    return codes, len(audio) / sr


def transcribe(model, tokenizer, config, audio_path, temperature=0.0, device="cuda"):
    model.eval()
    codes, dur = encode_audio(audio_path, config.sample_rate, config.num_codebooks, device)
    audio = codes.unsqueeze(0).to(device)
    t0 = time.time()
    text = model.generate(audio, tokenizer, temperature=temperature)
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
    print(f"📊 {r['duration']:.1f}s, RTF={r['rtf']:.3f}")

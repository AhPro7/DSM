"""
DSM-ASR Inference (Audio Prefix → Text Generation)

Transcribe audio files using the trained model.
Audio is encoded with Mimi, then the model generates text autoregressively.

Usage:
    python inference.py --checkpoint output/best --audio path/to/audio.wav
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from train import load_checkpoint


def encode_audio_mimi(audio_path: str, sample_rate: int = 24000,
                      num_codebooks: int = 8, device: str = "cuda"):
    """Encode audio file with Mimi → [T, Q] codes."""
    from transformers import MimiModel, AutoFeatureExtractor

    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(audio) / sample_rate

    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)
    mimi.eval()
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    inputs = fe(raw_audio=audio, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        audio_codes = mimi.encode(input_values).audio_codes
        if audio_codes.dim() == 4:
            codes = audio_codes[0, 0, :num_codebooks, :].T
        else:
            codes = audio_codes[0, :num_codebooks, :].T

    return codes, duration


def transcribe(model, tokenizer, config, audio_path=None, audio_codes=None,
               temperature=0.0, device="cuda"):
    """
    Transcribe audio using DSM-ASR model.

    Args:
        model: Trained DsmAsrModel
        audio_path: Path to audio file
        audio_codes: Pre-computed codes [T, Q] (optional)
        temperature: 0 = greedy

    Returns:
        dict with text, duration, rtf, etc.
    """
    model.eval()

    if audio_codes is None:
        audio_codes, duration = encode_audio_mimi(
            audio_path, config.sample_rate, config.num_codebooks, device)
    else:
        duration = audio_codes.shape[0] / config.frame_rate

    audio_tokens = audio_codes.unsqueeze(0).to(device)  # [1, T, Q]

    start_time = time.time()
    generated = model.generate(
        audio_tokens, tokenizer,
        max_new_tokens=config.max_text_tokens,
        temperature=temperature,
    )
    decode_time = time.time() - start_time

    # Remove special tokens and decode
    start_id = tokenizer.convert_tokens_to_ids("<|start_text|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end_text|>")
    text_tokens = [t for t in generated if t != start_id and t != end_id]
    transcription = tokenizer.decode(text_tokens, skip_special_tokens=True).strip()

    return {
        "text": transcription,
        "tokens": text_tokens,
        "duration": duration,
        "decode_time": decode_time,
        "rtf": decode_time / max(duration, 0.001),
        "num_frames": audio_codes.shape[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSM-ASR Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)

    print(f"\n🎤 Transcribing: {args.audio}")
    result = transcribe(model, tokenizer, config, audio_path=args.audio,
                       temperature=args.temperature, device=device)

    print(f"\n📝 Transcription:\n{result['text']}")
    print(f"\n📊 Duration: {result['duration']:.2f}s | "
          f"Decode: {result['decode_time']:.2f}s | RTF: {result['rtf']:.4f}")

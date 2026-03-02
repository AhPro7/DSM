"""
DSM-ASR Inference Pipeline

Transcribe audio files using the trained DSM-ASR model.
Supports frame-by-frame autoregressive decoding.

Usage:
    python inference.py --checkpoint output/best --audio path/to/audio.wav
    python inference.py --checkpoint output/best --audio path/to/audio.wav --temperature 0.0
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import librosa
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from train import load_checkpoint


def encode_audio_mimi(
    audio_path: str,
    sample_rate: int = 24000,
    num_codebooks: int = 8,
    device: str = "cuda",
):
    """
    Encode an audio file with Mimi and return codes.

    Returns:
        audio_codes: [T, Q] tensor of codebook indices
        duration: float, audio duration in seconds
    """
    from transformers import MimiModel, AutoFeatureExtractor

    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = len(audio) / sample_rate

    # Load Mimi
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)
    mimi.eval()
    fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    # Encode
    inputs = fe(raw_audio=audio, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        encoder_outputs = mimi.encode(input_values)
        codes = encoder_outputs.audio_codes[0, 0, :num_codebooks, :].T  # [T, Q]

    return codes, duration


@torch.no_grad()
def transcribe(
    model,
    tokenizer,
    config: DsmAsrConfig,
    audio_path: str = None,
    audio_codes: torch.Tensor = None,
    temperature: float = 0.0,
    device: str = "cuda",
) -> dict:
    """
    Transcribe audio using DAR-ASR model.

    Args:
        model: Trained DsmAsrModel
        tokenizer: Tokenizer
        config: Model config
        audio_path: Path to audio file (if audio_codes not provided)
        audio_codes: Pre-computed audio codes [T, Q] (optional)
        temperature: Sampling temperature (0 = greedy)
        device: Device

    Returns:
        dict with 'text', 'tokens', 'duration', 'rtf', 'num_frames'
    """
    model.eval()

    # Get special token IDs
    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    word_token_id = tokenizer.convert_tokens_to_ids("<|word|>")

    # Encode audio if needed
    if audio_codes is None:
        if audio_path is None:
            raise ValueError("Either audio_path or audio_codes must be provided")
        audio_codes, duration = encode_audio_mimi(
            audio_path, config.sample_rate, config.num_codebooks, device
        )
    else:
        duration = audio_codes.shape[0] / config.frame_rate

    audio_codes = audio_codes.to(device)
    T, Q = audio_codes.shape

    # Prepare for autoregressive decoding
    # Audio tokens shape: [1, T, Q]
    audio_tokens = audio_codes.unsqueeze(0)

    # Initialize text tokens with PAD
    text_tokens = torch.full((1, T), pad_token_id, dtype=torch.long, device=device)

    # Decode frame-by-frame
    start_time = time.time()
    generated_tokens = []

    for t in range(T - 1):
        # Get current slice up to frame t+1
        audio_slice = audio_tokens[:, :t + 1]  # [1, t+1, Q]
        text_slice = text_tokens[:, :t + 1]    # [1, t+1]
        attention_mask = torch.ones(1, t + 1, device=device)
        text_loss_mask = torch.zeros(1, t + 1, device=device)

        # Forward pass
        output = model(audio_slice, text_slice, attention_mask, text_loss_mask)

        # Get logits for the last position
        logits = output.logits[0, -1]  # [vocab_size]

        # Sample or greedy
        if temperature == 0.0:
            next_token = logits.argmax().item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        # Set the next text token (for teacher forcing at next step)
        if t + 1 < T:
            text_tokens[0, t + 1] = next_token

        # Collect non-special tokens
        if next_token != pad_token_id and next_token != word_token_id:
            generated_tokens.append(next_token)

    decode_time = time.time() - start_time

    # Decode tokens to text
    transcription = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean up
    transcription = transcription.strip()

    rtf = decode_time / max(duration, 0.001)

    return {
        "text": transcription,
        "tokens": generated_tokens,
        "duration": duration,
        "decode_time": decode_time,
        "rtf": rtf,
        "num_frames": T,
    }


@torch.no_grad()
def transcribe_fast(
    model,
    tokenizer,
    config: DsmAsrConfig,
    audio_path: str = None,
    audio_codes: torch.Tensor = None,
    device: str = "cuda",
) -> dict:
    """
    Fast transcription using full-sequence forward pass.

    Instead of true autoregressive decoding (slow, frame-by-frame),
    this does a single forward pass with teacher-forced PAD tokens
    and extracts predictions. Good for evaluation but less accurate
    than proper autoregressive decoding.

    Args:
        model: DsmAsrModel
        tokenizer: Tokenizer
        config: Config
        audio_path: Path to audio
        audio_codes: Pre-computed codes [T, Q]
        device: Device

    Returns:
        dict with transcription and metrics
    """
    model.eval()

    pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    word_token_id = tokenizer.convert_tokens_to_ids("<|word|>")

    # Encode audio
    if audio_codes is None:
        audio_codes, duration = encode_audio_mimi(
            audio_path, config.sample_rate, config.num_codebooks, device
        )
    else:
        duration = audio_codes.shape[0] / config.frame_rate

    audio_codes = audio_codes.to(device)
    T, Q = audio_codes.shape

    # Single forward pass with all PAD text tokens
    audio_tokens = audio_codes.unsqueeze(0)  # [1, T, Q]
    text_tokens = torch.full((1, T), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones(1, T, device=device)
    text_loss_mask = torch.zeros(1, T, device=device)

    start_time = time.time()
    output = model(audio_tokens, text_tokens, attention_mask, text_loss_mask)
    decode_time = time.time() - start_time

    # Get predicted tokens (greedy)
    pred_ids = output.logits[0].argmax(dim=-1).cpu().tolist()

    # Filter to get text tokens
    generated_tokens = [t for t in pred_ids if t != pad_token_id and t != word_token_id and t >= 0]

    transcription = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return {
        "text": transcription,
        "tokens": generated_tokens,
        "duration": duration,
        "decode_time": decode_time,
        "rtf": decode_time / max(duration, 0.001),
        "num_frames": T,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSM-ASR Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--fast", action="store_true", help="Use fast (non-autoregressive) mode")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    # Transcribe
    print(f"\n🎤 Transcribing: {args.audio}")
    if args.fast:
        result = transcribe_fast(model, tokenizer, config, audio_path=args.audio, device=device)
    else:
        result = transcribe(
            model, tokenizer, config, audio_path=args.audio,
            temperature=args.temperature, device=device,
        )

    print(f"\n📝 Transcription:\n{result['text']}")
    print(f"\n📊 Stats:")
    print(f"   Duration: {result['duration']:.2f}s")
    print(f"   Decode time: {result['decode_time']:.2f}s")
    print(f"   RTF: {result['rtf']:.4f}")
    print(f"   Frames: {result['num_frames']}")

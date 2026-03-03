"""
DSM-ASR Training v3 — True Delayed Streams

Includes:
- Sample predictions printed during training (5-10 samples with WER)
- Parallel stream architecture
- Weighted loss (text tokens vs padding predictions)
"""
import os
import sys
import time
import json
import argparse
import re
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DsmAsrConfig
from data.dataset import DsmAsrDataset
from data.collator import DsmAsrCollator
from model.dsm_asr import DsmAsrModel


def normalize_arabic(text):
    text = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]').sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ـ', '')
    return ' '.join(text.split()).strip()


def print_sample_predictions(model, eval_ds, tokenizer, device, config, num_samples=5):
    """Print sample predictions during training to track progress."""
    model.eval()
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    epad_id = tokenizer.convert_tokens_to_ids("<|epad|>")
    bos_id = tokenizer.convert_tokens_to_ids("<|bos|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")

    wers = []
    print(f"\n{'='*60}")
    print(f"📋 Sample Predictions ({num_samples} samples)")
    print(f"{'='*60}")

    indices = list(range(min(num_samples, len(eval_ds))))
    for idx in indices:
        sample = eval_ds[idx]
        sample_info = eval_ds.samples[idx]
        audio_tokens = sample["audio_tokens"].unsqueeze(0).to(device)

        # Generate
        gen_tokens = model.generate_text(audio_tokens, tokenizer)
        pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Reference
        data = np.load(sample_info["path"], allow_pickle=True)
        ref = str(data["text"]).strip()

        # WER
        try:
            from jiwer import wer
            sample_wer = wer(normalize_arabic(ref), normalize_arabic(pred)) if pred and ref else 1.0
        except:
            sample_wer = -1.0

        wers.append(sample_wer)
        print(f"\n  [{idx+1}] WER={sample_wer:.2f}")
        print(f"       REF: {ref[:80]}")
        print(f"       HYP: {pred[:80]}")

    avg_wer = sum(w for w in wers if w >= 0) / max(len([w for w in wers if w >= 0]), 1)
    print(f"\n  📊 Avg WER: {avg_wer:.4f}")
    print(f"{'='*60}\n")
    model.train()
    return avg_wer


def evaluate(model, eval_loader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break
            out = model(
                batch["audio_tokens"].to(device),
                batch["text_tokens"].to(device),
                batch["text_targets"].to(device),
                batch["loss_mask"].to(device),
                batch["lengths"].to(device),
            )
            w = batch["loss_mask"].sum().item()
            total_loss += out.loss.item() * w
            total_weight += w
    model.train()
    return total_loss / max(total_weight, 1.0)


def train(config, args):
    print("=" * 60)
    print("DSM-ASR Training v3 — True Delayed Streams")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": config.special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")

    # Datasets
    train_ds = DsmAsrDataset(config, "train", tokenizer=tokenizer, max_samples=args.max_samples)
    eval_ds = DsmAsrDataset(config, "eval", tokenizer=tokenizer, max_samples=args.max_samples)
    collator = DsmAsrCollator(config=config, pad_text_id=pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=min(config.preprocessing_num_workers, 2),
        collate_fn=collator, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(
        eval_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=min(config.preprocessing_num_workers, 2),
        collate_fn=collator, pin_memory=True)

    # Model
    model = DsmAsrModel(config, tokenizer=tokenizer).to(device)

    # Optimizer — audio params get higher LR
    audio_params = (
        list(model.audio_embeddings.parameters()) +
        list(model.audio_adapter.parameters()) +
        [model.audio_scale]
    )
    backbone_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {"params": audio_params, "lr": config.learning_rate * config.audio_lr_multiplier},
        {"params": backbone_params, "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    use_amp = config.bf16 or config.fp16
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = GradScaler('cuda', enabled=config.fp16)

    if config.use_wandb or args.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, config=vars(config))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Training: {config.num_epochs} epochs, {total_steps} steps")
    print(f"   Batch: {config.batch_size}×{config.gradient_accumulation_steps} "
          f"= {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Delay: {config.delay_frames} frames "
          f"({config.delay_frames / config.frame_rate:.1f}s)")

    global_step = 0
    best_loss = float("inf")
    log = []

    for epoch in range(config.num_epochs):
        if epoch < config.freeze_backbone_epochs:
            model.freeze_backbone()
            phase = "Frozen"
        else:
            model.unfreeze_backbone()
            phase = "Full"

        print(f"\n📌 Epoch {epoch+1}/{config.num_epochs} [{phase}] "
              f"Trainable: {model.get_trainable_params():,}")
        model.train()
        epoch_loss = 0.0
        epoch_w = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch in pbar:
            a = batch["audio_tokens"].to(device)
            t = batch["text_tokens"].to(device)
            tgt = batch["text_targets"].to(device)
            m = batch["loss_mask"].to(device)
            l = batch["lengths"].to(device)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(a, t, tgt, m, l)
                loss = out.loss / config.gradient_accumulation_steps

            if config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            w = m.sum().item()
            epoch_loss += out.loss.item() * w
            epoch_w += w

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if config.fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg = epoch_loss / max(epoch_w, 1)
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}", step=global_step)

                if global_step % config.log_every_n_steps == 0:
                    log.append({"step": global_step, "loss": avg, "lr": lr})

                # Print sample predictions
                if global_step % config.print_samples_every == 0 and len(eval_ds) > 0:
                    print_sample_predictions(
                        model, eval_ds, tokenizer, device, config,
                        num_samples=config.num_print_samples)

                # Eval
                if global_step % config.eval_every_n_steps == 0 and len(eval_ds) > 0:
                    el = evaluate(model, eval_loader, device)
                    print(f"\n🔍 Step {global_step}: eval_loss={el:.4f}")
                    if el < best_loss:
                        best_loss = el
                        save_checkpoint(model, tokenizer, config, output_dir / "best", global_step)
                        print(f"   💾 Best! (loss={best_loss:.4f})")

                if global_step % config.save_every_n_steps == 0:
                    save_checkpoint(model, tokenizer, config,
                                   output_dir / f"checkpoint-{global_step}", global_step)

                if args.max_steps and global_step >= args.max_steps:
                    break

        print(f"\n📊 Epoch {epoch+1}: loss={epoch_loss/max(epoch_w,1):.4f}")
        save_checkpoint(model, tokenizer, config, output_dir / f"epoch-{epoch+1}", global_step)

        # Print samples at end of each epoch
        if len(eval_ds) > 0:
            print_sample_predictions(
                model, eval_ds, tokenizer, device, config,
                num_samples=config.num_print_samples)

        if args.max_steps and global_step >= args.max_steps:
            break

    # Final
    if len(eval_ds) > 0:
        el = evaluate(model, eval_loader, device)
        print(f"\n🔍 Final eval_loss={el:.4f}")
        if el < best_loss:
            best_loss = el
    if not (output_dir / "best" / "model.pt").exists():
        save_checkpoint(model, tokenizer, config, output_dir / "best", global_step)
    save_checkpoint(model, tokenizer, config, output_dir / "final", global_step)
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✅ Done! Steps={global_step}, Best={best_loss:.4f}")


def save_checkpoint(model, tokenizer, config, path, step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": vars(config),
                "step": step}, path / "model.pt")
    tokenizer.save_pretrained(path / "tokenizer")
    with open(path / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)


def load_checkpoint(ckpt_dir, device="cuda"):
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "config.json") as f:
        d = json.load(f)
    config = DsmAsrConfig(**{k: v for k, v in d.items() if k in DsmAsrConfig.__dataclass_fields__})
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer", trust_remote_code=True)
    model = DsmAsrModel(config, tokenizer=tokenizer)
    ckpt = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device), tokenizer, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    config = DsmAsrConfig()
    if args.batch_size: config.batch_size = args.batch_size
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.num_epochs: config.num_epochs = args.num_epochs
    if args.output_dir: config.output_dir = args.output_dir

    train(config, args)

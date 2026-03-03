"""
DSM-ASR Training v4 — Instruction Fine-Tuning

Features:
- Prints sample predictions with WER during training
- KV-cache inference for fast generation
- Instruction format: audio prefix → text generation
"""
import os, sys, time, json, re, argparse
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


def normalize_ar(text):
    text = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670]').sub('', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ـ', '')
    return ' '.join(text.split()).strip()


def print_predictions(model, eval_ds, tokenizer, device, n=5):
    """Print sample predictions during training."""
    model.eval()
    wers = []
    print(f"\n{'='*70}")
    print(f"📋 Sample Predictions")
    print(f"{'='*70}")

    for i in range(min(n, len(eval_ds))):
        s = eval_ds[i]
        audio = s["audio_tokens"].unsqueeze(0).to(device)
        ref = tokenizer.decode(s["target_ids"], skip_special_tokens=True).strip()

        pred = model.generate(audio, tokenizer)

        try:
            from jiwer import wer as calc_wer
            w = calc_wer(normalize_ar(ref), normalize_ar(pred)) if pred and ref else 1.0
        except:
            w = -1.0
        wers.append(w)

        print(f"\n  [{i+1}] WER={w:.2f}")
        print(f"       REF: {ref[:80]}")
        print(f"       HYP: {pred[:80]}")

    valid = [w for w in wers if w >= 0]
    avg = sum(valid) / max(len(valid), 1)
    print(f"\n  📊 Avg WER: {avg:.4f}")
    print(f"{'='*70}\n")
    model.train()
    return avg


def evaluate_loss(model, loader, device, max_batches=50):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            out = model(
                b["instruction_ids"].to(device),
                b["audio_tokens"].to(device),
                b["separator_ids"].to(device),
                b["target_ids"].to(device),
                b["labels"].to(device),
                b["audio_lengths"].to(device),
                b["target_lengths"].to(device),
            )
            n = (b["labels"] != -100).sum().item()
            total += out.loss.item() * n
            count += n
    model.train()
    return total / max(count, 1)


def train(config, args):
    print("=" * 60)
    print("DSM-ASR Training v4 — Instruction Fine-Tuning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = DsmAsrDataset(config, "train", tokenizer=tokenizer, max_samples=args.max_samples)
    eval_ds = DsmAsrDataset(config, "eval", tokenizer=tokenizer, max_samples=args.max_samples)
    collator = DsmAsrCollator(config=config, text_pad_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=2, collate_fn=collator, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(
        eval_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=2, collate_fn=collator, pin_memory=True)

    model = DsmAsrModel(config, tokenizer=tokenizer).to(device)

    # Optimizer: audio params get higher LR
    audio_params = (list(model.audio_embeddings.parameters()) +
                    list(model.audio_adapter.parameters()) + [model.audio_scale])
    backbone_params = list(model.backbone.parameters())
    optimizer = torch.optim.AdamW([
        {"params": audio_params, "lr": config.learning_rate * config.audio_lr_multiplier},
        {"params": backbone_params, "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config.warmup_ratio), total_steps)

    use_amp = config.bf16 or config.fp16
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = GradScaler('cuda', enabled=config.fp16)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Train: {config.num_epochs} epochs, {total_steps} steps")
    print(f"   Batch: {config.batch_size}×{config.gradient_accumulation_steps}")
    print(f"   Instruction: \"{config.instruction.strip()}\"")
    print(f"   Separator: \"{config.separator.strip()}\"")

    global_step = 0
    best_loss = float("inf")
    log = []

    for epoch in range(config.num_epochs):
        model.unfreeze_backbone()
        model.train()

        print(f"\n📌 Epoch {epoch+1}/{config.num_epochs}  "
              f"Trainable: {model.get_trainable_params():,}")

        epoch_loss, epoch_n = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}")

        for batch_idx, b in pbar:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    b["instruction_ids"].to(device),
                    b["audio_tokens"].to(device),
                    b["separator_ids"].to(device),
                    b["target_ids"].to(device),
                    b["labels"].to(device),
                    b["audio_lengths"].to(device),
                    b["target_lengths"].to(device),
                )
                loss = out.loss / config.gradient_accumulation_steps

            if config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            n = (b["labels"] != -100).sum().item()
            epoch_loss += out.loss.item() * n
            epoch_n += n

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

                avg = epoch_loss / max(epoch_n, 1)
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}", step=global_step)

                if global_step % config.log_every_n_steps == 0:
                    log.append({"step": global_step, "loss": avg, "lr": lr})

                # Print sample predictions
                if global_step % config.print_samples_every == 0 and len(eval_ds) > 0:
                    print_predictions(model, eval_ds, tokenizer, device,
                                      n=config.num_print_samples)

                # Eval
                if global_step % config.eval_every_n_steps == 0 and len(eval_ds) > 0:
                    el = evaluate_loss(model, eval_loader, device)
                    print(f"\n🔍 Step {global_step}: eval_loss={el:.4f}")
                    if el < best_loss:
                        best_loss = el
                        save_ckpt(model, tokenizer, config, output_dir / "best", global_step)
                        print(f"   💾 Best! ({best_loss:.4f})")

                if global_step % config.save_every_n_steps == 0:
                    save_ckpt(model, tokenizer, config,
                              output_dir / f"step-{global_step}", global_step)

                if args.max_steps and global_step >= args.max_steps:
                    break

        avg_epoch = epoch_loss / max(epoch_n, 1)
        print(f"\n📊 Epoch {epoch+1}: loss={avg_epoch:.4f}")
        save_ckpt(model, tokenizer, config, output_dir / f"epoch-{epoch+1}", global_step)

        # Predictions at epoch end
        if len(eval_ds) > 0:
            print_predictions(model, eval_ds, tokenizer, device, n=config.num_print_samples)

        if args.max_steps and global_step >= args.max_steps:
            break

    # Final
    if len(eval_ds) > 0:
        el = evaluate_loss(model, eval_loader, device)
        if el < best_loss:
            best_loss = el
    if not (output_dir / "best" / "model.pt").exists():
        save_ckpt(model, tokenizer, config, output_dir / "best", global_step)
    save_ckpt(model, tokenizer, config, output_dir / "final", global_step)
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✅ Done! Steps={global_step}, Best={best_loss:.4f}")


def save_ckpt(model, tokenizer, config, path, step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "step": step}, path / "model.pt")
    tokenizer.save_pretrained(path / "tokenizer")
    with open(path / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)


def load_checkpoint(ckpt_dir, device="cuda"):
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "config.json") as f:
        d = json.load(f)
    config = DsmAsrConfig(**{k: v for k, v in d.items()
                             if k in DsmAsrConfig.__dataclass_fields__})
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
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    config = DsmAsrConfig()
    if args.batch_size: config.batch_size = args.batch_size
    if args.num_epochs: config.num_epochs = args.num_epochs
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.output_dir: config.output_dir = args.output_dir
    if args.use_wandb: config.use_wandb = True

    train(config, args)

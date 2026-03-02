"""
DSM-ASR Training Script (Audio Prefix → Text Generation)

Two-phase training:
  Phase 1: Freeze Qwen3 backbone, train audio embeddings
  Phase 2: Unfreeze all, full fine-tuning with lower learning rate

Usage:
  python train.py
  python train.py --max_steps 100 --batch_size 2
  python train.py --use_wandb
"""
import os
import sys
import time
import json
import argparse
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


def compute_wer_simple(predictions: list, references: list) -> float:
    """Compute WER using jiwer."""
    try:
        from jiwer import wer
        valid = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        if not valid:
            return -1.0
        preds, refs = zip(*valid)
        return wer(list(refs), list(preds))
    except ImportError:
        return -1.0


def evaluate(model, eval_loader, tokenizer, device, max_batches=50):
    """Run evaluation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    start_text_id = tokenizer.convert_tokens_to_ids("<|start_text|>")
    end_text_id = tokenizer.convert_tokens_to_ids("<|end_text|>")

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= max_batches:
                break

            audio_tokens = batch["audio_tokens"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_target_ids = batch["text_target_ids"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            text_lengths = batch["text_lengths"].to(device)

            output = model(audio_tokens, text_input_ids, text_target_ids,
                          audio_lengths, text_lengths)

            n_valid = (text_target_ids != -100).sum().item()
            total_loss += output.loss.item() * n_valid
            total_tokens += n_valid

    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return {"eval_loss": avg_loss}


def train(config: DsmAsrConfig, args):
    """Main training loop."""
    print("=" * 60)
    print("DSM-ASR Training (Audio Prefix → Text)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────
    print(f"\n📝 Loading tokenizer: {config.qwen_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": config.special_tokens})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_pad_id = tokenizer.pad_token_id

    # ── Dataset ──────────────────────────────────────────────────────
    print(f"\n📂 Loading datasets...")
    train_ds = DsmAsrDataset(config, split="train", tokenizer=tokenizer,
                             max_samples=args.max_samples)
    eval_ds = DsmAsrDataset(config, split="eval", tokenizer=tokenizer,
                            max_samples=args.max_samples)

    collator = DsmAsrCollator(config=config, text_pad_token_id=text_pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=min(config.preprocessing_num_workers, 2),
        collate_fn=collator, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=min(config.preprocessing_num_workers, 2),
        collate_fn=collator, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    print(f"\n🧠 Building model...")
    model = DsmAsrModel(config, tokenizer=tokenizer)
    model = model.to(device)

    # ── Optimizer ────────────────────────────────────────────────────
    audio_params = list(model.audio_embeddings.parameters()) + [model.audio_scale]
    backbone_params = list(model.backbone.parameters())

    optimizer = torch.optim.AdamW([
        {"params": audio_params, "lr": config.learning_rate * 10},
        {"params": backbone_params, "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)

    # ── Scheduler ────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Mixed precision ──────────────────────────────────────────────
    use_amp = config.bf16 or config.fp16
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = GradScaler('cuda', enabled=config.fp16)

    # ── WandB ────────────────────────────────────────────────────────
    if config.use_wandb or args.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, config=vars(config),
                   name=f"dsm-asr-{time.strftime('%Y%m%d_%H%M%S')}")

    # ── Output dir ───────────────────────────────────────────────────
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────
    print(f"\n🚀 Starting training:")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Steps/epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Batch: {config.batch_size} × {config.gradient_accumulation_steps} = "
          f"{config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Freeze epochs: {config.freeze_backbone_epochs}")

    global_step = 0
    best_eval_loss = float("inf")
    train_losses = []

    for epoch in range(config.num_epochs):
        # Phase control
        if epoch < config.freeze_backbone_epochs:
            model.freeze_backbone()
            print(f"\n📌 Epoch {epoch+1}/{config.num_epochs} [Phase 1: Frozen]")
        else:
            model.unfreeze_backbone()
            optimizer.param_groups[1]["lr"] = config.learning_rate * 0.1
            print(f"\n📌 Epoch {epoch+1}/{config.num_epochs} [Phase 2: Full]")

        print(f"   Trainable: {model.get_trainable_params():,}")
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad()

        progress = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f"Epoch {epoch+1}", leave=True)

        for batch_idx, batch in progress:
            audio_tokens = batch["audio_tokens"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_target_ids = batch["text_target_ids"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            text_lengths = batch["text_lengths"].to(device)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                output = model(audio_tokens, text_input_ids, text_target_ids,
                              audio_lengths, text_lengths)
                loss = output.loss / config.gradient_accumulation_steps

            if config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            n_tokens = (text_target_ids != -100).sum().item()
            epoch_loss += output.loss.item() * n_tokens
            epoch_tokens += n_tokens

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

                avg_loss = epoch_loss / max(epoch_tokens, 1)
                current_lr = scheduler.get_last_lr()[0]
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}", step=global_step)

                if global_step % config.log_every_n_steps == 0:
                    train_losses.append({"step": global_step, "loss": avg_loss, "lr": current_lr})
                    if config.use_wandb or args.use_wandb:
                        import wandb
                        wandb.log({"train/loss": avg_loss, "train/lr": current_lr}, step=global_step)

                if global_step % config.eval_every_n_steps == 0 and len(eval_ds) > 0:
                    print(f"\n🔍 Eval at step {global_step}...")
                    metrics = evaluate(model, eval_loader, tokenizer, device)
                    print(f"   Eval loss: {metrics['eval_loss']:.4f}")
                    if metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = metrics["eval_loss"]
                        save_checkpoint(model, tokenizer, config, output_dir / "best", global_step)
                        print(f"   💾 Best checkpoint (loss={best_eval_loss:.4f})")

                if global_step % config.save_every_n_steps == 0:
                    save_checkpoint(model, tokenizer, config,
                                    output_dir / f"checkpoint-{global_step}", global_step)

                if args.max_steps is not None and global_step >= args.max_steps:
                    print(f"\n⏹️  Reached max_steps={args.max_steps}")
                    break

        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)
        print(f"\n📊 Epoch {epoch+1}: loss={avg_epoch_loss:.4f}")
        save_checkpoint(model, tokenizer, config, output_dir / f"epoch-{epoch+1}", global_step)

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # Final eval + save best if not saved yet
    if len(eval_ds) > 0:
        print(f"\n🔍 Final evaluation...")
        final_metrics = evaluate(model, eval_loader, tokenizer, device)
        print(f"   Final eval loss: {final_metrics['eval_loss']:.4f}")
        if final_metrics["eval_loss"] < best_eval_loss:
            best_eval_loss = final_metrics["eval_loss"]

    # Always save best (use final model if no eval was done)
    if not (output_dir / "best" / "model.pt").exists():
        save_checkpoint(model, tokenizer, config, output_dir / "best", global_step)
        print(f"   💾 Saved best checkpoint")

    save_checkpoint(model, tokenizer, config, output_dir / "final", global_step)
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(train_losses, f, indent=2)

    print(f"\n✅ Training complete! Steps: {global_step}, Best loss: {best_eval_loss:.4f}")


def save_checkpoint(model, tokenizer, config, save_dir, step):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": vars(config),
                "step": step}, save_dir / "model.pt")
    tokenizer.save_pretrained(save_dir / "tokenizer")
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2, default=str)


def load_checkpoint(checkpoint_dir, device="cuda"):
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    config = DsmAsrConfig(**{k: v for k, v in config_dict.items()
                            if k in DsmAsrConfig.__dataclass_fields__})
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", trust_remote_code=True)
    model = DsmAsrModel(config, tokenizer=tokenizer)
    checkpoint = torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model, tokenizer, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DSM-ASR")
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

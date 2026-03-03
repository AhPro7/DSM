"""
DSM-ASR Model v5 — SODA-Style Vocabulary Extension

Architecture: PURE standard causal LM.
  - Qwen3-0.6B backbone, vocab extended to 168,059
  - Audio tokens are just new vocabulary entries (initialized randomly)
  - NO adapters, NO separate audio embedding tables
  - Forward pass = backbone(input_ids, labels) — identical to text LLM

Inference (ASR):
  - Prefix = <|audio_start|> [audio_vocab_ids] <|audio_end|> <|text_start|>
  - Autoregressive decode until <|text_end|>
"""
import os, sys, json
import torch
import torch.nn as nn
from typing import Optional, NamedTuple
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


class DsmAsrOutput(NamedTuple):
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class DsmAsrModel(nn.Module):
    def __init__(self, config: DsmAsrConfig, tokenizer=None):
        super().__init__()
        self.config = config

        print(f"🧠 Loading Qwen3: {config.qwen_model}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.qwen_model,
            dtype=torch.bfloat16 if config.bf16 else torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        # Extend vocabulary: MUST resize to total_vocab_size (text + special + audio)
        # config.total_vocab_size = 151,671 + 4 + 16,384 = 168,059
        # NOTE: len(tokenizer) is only 151,675 (text + 4 special) — NOT sufficient!
        #       Audio token IDs go up to 168,058, so we must resize to 168,059.
        target_vocab_size = config.total_vocab_size  # 168,059
        old_vocab_size = self.backbone.config.vocab_size
        self.backbone.resize_token_embeddings(target_vocab_size)
        # Initialize ALL new token embeddings with small Gaussian noise
        with torch.no_grad():
            emb = self.backbone.model.embed_tokens.weight
            if target_vocab_size > old_vocab_size:
                nn.init.normal_(emb[old_vocab_size:], mean=0.0, std=0.02)
        # Also resize and initialize the LM head
        if hasattr(self.backbone, 'lm_head'):
            with torch.no_grad():
                lm = self.backbone.lm_head.weight
                if target_vocab_size > old_vocab_size:
                    nn.init.normal_(lm[old_vocab_size:], mean=0.0, std=0.02)
        print(f"   Vocab: {old_vocab_size:,} → {target_vocab_size:,}")
        print(f"     text={config.text_vocab_size:,}  "
              f"special={config.num_extra_special}  "
              f"audio={config.audio_vocab_size:,}")

        self.text_vocab = config.text_vocab_size
        n_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"   Backbone params: {n_params:,}")

        # Store special token IDs for generation
        if tokenizer:
            self.audio_start_id = tokenizer.convert_tokens_to_ids(config.audio_start_token)
            self.audio_end_id   = tokenizer.convert_tokens_to_ids(config.audio_end_token)
            self.text_start_id  = tokenizer.convert_tokens_to_ids(config.text_start_token)
            self.text_end_id    = tokenizer.convert_tokens_to_ids(config.text_end_token)
            self.eos_id         = tokenizer.eos_token_id
        else:
            self.audio_start_id = self.audio_end_id = None
            self.text_start_id = self.text_end_id = self.eos_id = None

    def forward(self, input_ids, labels=None, attention_mask=None):
        """Standard causal LM forward — identical to a text LLM."""
        out = self.backbone(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # Apply label smoothing if needed (backbone doesn't do this natively)
        loss = out.loss
        if loss is not None and self.config.label_smoothing > 0:
            import torch.nn.functional as F
            # Re-compute loss with smoothing
            valid = labels != -100
            if valid.any():
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=self.config.label_smoothing,
                )
        return DsmAsrOutput(loss=loss, logits=out.logits)

    def freeze_new_tokens_only(self):
        """Optionally freeze backbone, train only new token embeddings."""
        for name, p in self.backbone.named_parameters():
            p.requires_grad = False
        # Unfreeze embedding rows for new tokens only
        self.backbone.model.embed_tokens.weight.requires_grad = True
        if hasattr(self.backbone, 'lm_head'):
            self.backbone.lm_head.weight.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, audio_flat_ids, tokenizer, max_new_tokens=256, temperature=0.0):
        """
        ASR generation:
          audio_flat_ids: 1D tensor of audio vocab token IDs [T*Q]
          Returns: decoded text string
        
        Prompt:
          <|audio_start|> [audio_flat_ids] <|audio_end|> <|text_start|>
        """
        self.eval()
        device = next(self.parameters()).device

        prefix = torch.tensor(
            [self.audio_start_id] +
            audio_flat_ids.tolist() +
            [self.audio_end_id, self.text_start_id],
            dtype=torch.long, device=device
        ).unsqueeze(0)  # [1, L_prefix]

        attn = torch.ones_like(prefix)

        # Process prefix with KV cache
        out = self.backbone(
            input_ids=prefix,
            attention_mask=attn,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

        generated = []
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                tok = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                tok = torch.multinomial(probs, 1).item()

            if tok in (self.text_end_id, self.eos_id):
                break
            # Skip if model accidentally generates audio tokens
            if tok >= self.config.audio_token_offset:
                break
            generated.append(tok)

            # Next step with KV cache
            if hasattr(past, 'get_seq_length'):
                cache_len = past.get_seq_length()
            else:
                cache_len = past[0][0].shape[2]
            attn = torch.ones(1, cache_len + 1, device=device, dtype=torch.long)
            tok_t = torch.tensor([[tok]], device=device)
            out = self.backbone(
                input_ids=tok_t,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            next_logits = out.logits[0, -1, :]

        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def test_model():
    print("Testing DSM-ASR v5 (SODA-style)...")
    config = DsmAsrConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": config.special_tokens})

    model = DsmAsrModel(config, tokenizer=tokenizer)

    B, T, Q = 2, 50, config.num_codebooks
    # Simulate flat audio IDs
    audio_flat = torch.randint(0, config.audio_vocab_size, (B, T * Q))
    audio_vocab_ids = audio_flat + config.audio_token_offset

    audio_start = tokenizer.convert_tokens_to_ids(config.audio_start_token)
    audio_end   = tokenizer.convert_tokens_to_ids(config.audio_end_token)
    text_start  = tokenizer.convert_tokens_to_ids(config.text_start_token)
    text_end    = tokenizer.convert_tokens_to_ids(config.text_end_token)

    # Build sequences manually for test
    prefix_ids = [audio_start] + [0] * (T * Q) + [audio_end, text_start]
    text_ids   = [1, 2, 3, 4, 5, text_end]
    seq = torch.tensor([prefix_ids + text_ids], dtype=torch.long)
    lbl = torch.tensor([[-100] * len(prefix_ids) + text_ids], dtype=torch.long)
    seq = seq.expand(B, -1)
    lbl = lbl.expand(B, -1)
    mask = torch.ones_like(seq)

    model.eval()
    with torch.no_grad():
        out = model(seq, lbl, mask)
    print(f"  Loss: {out.loss.item():.4f}")
    print(f"  Logits: {out.logits.shape}")
    print(f"  Total vocab: {out.logits.shape[-1]:,}")
    print("✅ Model v5 PASSED!")


if __name__ == "__main__":
    test_model()

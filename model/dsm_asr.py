"""
DSM-ASR Model v3 — True Delayed Streams (Moshi Paper)

Architecture (per time step t):
    Input = audio_emb[t] + text_emb[t]  (summed, NOT concatenated)
    → Qwen3 Temporal Transformer (causal attention across time)
    → LM head → predict text_token[t+1]

This matches the Moshi paper Section 3.4.1:
"the Temporal Transformer receives at each step s as input
the sum of K learnt embedding tables representing the value for V_{s-1}"

Key difference from our v1/v2:
- v1/v2: Audio is a PREFIX, text is generated AFTER
- v3: Audio and text are PARALLEL at every time step, SUMMED together

For ASR mode (paper Section 5.7 + Appendix C):
- Audio tokens are teacher-forced (we know the audio)
- Text tokens are predicted (that's the transcription)
- Text is delayed behind audio by ~2s
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, NamedTuple
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DsmAsrConfig


class DsmAsrOutput(NamedTuple):
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class AudioAdapter(nn.Module):
    """MLP adapter: transforms summed codebook embeddings → LM space."""
    def __init__(self, dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            ])
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


class DsmAsrModel(nn.Module):
    """
    Delayed Streams ASR model.
    
    At each time step t, the input embedding is:
        emb[t] = audio_adapter(sum(codebook_embs[t])) + text_emb[t]
    
    This combined embedding is processed by Qwen3 (causal attention),
    and the output at position t predicts the NEXT text token.
    """

    def __init__(self, config: DsmAsrConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # ── Backbone ─────────────────────────────────────────────────
        print(f"🧠 Loading Qwen3: {config.qwen_model}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.qwen_model,
            dtype=torch.bfloat16 if config.bf16 else torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model_dim = self.backbone.config.hidden_size

        if tokenizer is not None:
            self.backbone.resize_token_embeddings(len(tokenizer))
            self.text_vocab_size = len(tokenizer)
        else:
            self.text_vocab_size = self.backbone.config.vocab_size

        # ── Audio embeddings (one per codebook, summed) ──────────────
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size + 1, self.model_dim,
                        padding_idx=config.audio_pad_token)
            for _ in range(config.num_codebooks)
        ])
        for emb in self.audio_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight[config.audio_pad_token] = 0.0

        # ── Audio adapter ────────────────────────────────────────────
        self.audio_adapter = AudioAdapter(
            self.model_dim, config.audio_adapter_layers, config.audio_adapter_dropout)

        # ── Audio scale ──────────────────────────────────────────────
        self.audio_scale = nn.Parameter(torch.ones(1))

        # Stats
        total_qwen = sum(p.numel() for p in self.backbone.parameters())
        total_audio = sum(p.numel() for p in self.audio_embeddings.parameters())
        total_adapter = sum(p.numel() for p in self.audio_adapter.parameters())
        print(f"   dim={self.model_dim}, codebooks={config.num_codebooks}")
        print(f"   Qwen3: {total_qwen:,}, Audio: {total_audio:,}, Adapter: {total_adapter:,}")

    def get_audio_emb(self, audio_tokens):
        """[B, T, Q] → [B, T, D]"""
        B, T, Q = audio_tokens.shape
        emb = torch.zeros(B, T, self.model_dim,
                          device=audio_tokens.device,
                          dtype=next(self.audio_embeddings[0].parameters()).dtype)
        for q, e in enumerate(self.audio_embeddings):
            emb = emb + e(audio_tokens[:, :, q])
        return self.audio_adapter(emb * self.audio_scale)

    def get_text_emb(self, text_tokens):
        """[B, T] → [B, T, D]"""
        return self.backbone.model.embed_tokens(text_tokens)

    def forward(self, audio_tokens, text_tokens, text_targets, loss_mask, lengths):
        """
        Forward pass with parallel streams.
        
        audio_tokens: [B, T, Q]
        text_tokens:  [B, T]  (teacher-forced text stream)
        text_targets: [B, T]  (what to predict)
        loss_mask:    [B, T]  (where to compute loss)
        lengths:      [B]     (actual sequence lengths)
        """
        B, T = text_tokens.shape

        # 1. Sum audio + text embeddings (parallel streams!)
        audio_emb = self.get_audio_emb(audio_tokens)  # [B, T, D]
        text_emb = self.get_text_emb(text_tokens)       # [B, T, D]

        # Cast to same dtype
        backbone_dtype = next(self.backbone.model.embed_tokens.parameters()).dtype
        combined = (audio_emb + text_emb).to(dtype=backbone_dtype)

        # 2. Attention mask
        attn_mask = torch.zeros(B, T, device=audio_tokens.device, dtype=combined.dtype)
        for i in range(B):
            attn_mask[i, :lengths[i]] = 1.0

        # 3. Through Qwen3 transformer
        outputs = self.backbone(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            return_dict=True,
        )
        logits = outputs.logits  # [B, T, vocab]

        # 4. Weighted loss on text predictions
        # logits[t] predicts text_target[t] (the next text token at position t)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = text_targets.reshape(-1)
        flat_mask = loss_mask.reshape(-1)

        # Compute per-token loss
        per_token_loss = F.cross_entropy(
            flat_logits, flat_targets,
            ignore_index=-100,
            reduction='none',
            label_smoothing=self.config.label_smoothing,
        )

        # Apply weighted mask (1.0 for text tokens, 0.5 for pad predictions)
        masked_loss = per_token_loss * flat_mask
        loss = masked_loss.sum() / flat_mask.sum().clamp(min=1.0)

        return DsmAsrOutput(loss=loss, logits=logits)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.audio_embeddings.parameters():
            p.requires_grad = True
        for p in self.audio_adapter.parameters():
            p.requires_grad = True
        self.audio_scale.requires_grad = True
        print("❄️  Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("🔥 Backbone unfrozen.")

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate_text(self, audio_tokens, tokenizer, temperature=0.0):
        """
        ASR inference: teacher-force audio, generate text.
        
        audio_tokens: [1, T, Q]
        Returns: list of generated token IDs
        """
        self.eval()
        device = audio_tokens.device
        T = audio_tokens.shape[1]

        pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")
        eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
        bos_id = tokenizer.convert_tokens_to_ids("<|bos|>")

        # Get all audio embeddings at once (we know the full audio)
        audio_emb = self.get_audio_emb(audio_tokens)  # [1, T, D]
        backbone_dtype = next(self.backbone.model.embed_tokens.parameters()).dtype
        audio_emb = audio_emb.to(dtype=backbone_dtype)

        # Initialize text stream with all PADs
        text_stream = torch.full((1, T), pad_id, dtype=torch.long, device=device)
        generated_tokens = []

        # Process frame by frame with KV cache
        past = None
        delay = self.config.delay_frames

        for t in range(T):
            # Get current text embedding
            text_emb_t = self.get_text_emb(text_stream[:, t:t+1]).to(dtype=backbone_dtype)

            # Combined input for this frame
            input_emb = audio_emb[:, t:t+1, :] + text_emb_t  # [1, 1, D]
            attn_mask_len = t + 1
            if past is not None and hasattr(past, 'get_seq_length'):
                attn_mask_len = past.get_seq_length() + 1
            elif past is not None:
                attn_mask_len = past[0][0].shape[2] + 1
            attn_mask = torch.ones(1, attn_mask_len, device=device, dtype=backbone_dtype)

            outputs = self.backbone(
                inputs_embeds=input_emb,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[0, -1, :]

            # Sample next text token
            if temperature == 0.0:
                next_tok = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

            # Place in text stream for next step
            if t + 1 < T:
                text_stream[0, t + 1] = next_tok

            # Collect real text tokens (not PAD/EPAD)
            if next_tok != pad_id and next_tok not in [
                tokenizer.convert_tokens_to_ids("<|epad|>"),
                bos_id,
            ]:
                if next_tok == eos_id:
                    break
                generated_tokens.append(next_tok)

        return generated_tokens


def test_forward():
    print("Testing DSM-ASR v3...")
    config = DsmAsrConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": config.special_tokens})

    model = DsmAsrModel(config, tokenizer=tokenizer)
    model.eval()

    B, T, Q = 2, 50, config.num_codebooks
    audio = torch.randint(0, config.audio_vocab_size, (B, T, Q))
    text = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    mask = torch.ones(B, T)
    lens = torch.tensor([T, T])

    with torch.no_grad():
        out = model(audio, text, targets, mask, lens)
    print(f"  Loss: {out.loss.item():.4f}, Logits: {out.logits.shape}")
    print("✅ Model v3 test PASSED!")


if __name__ == "__main__":
    test_forward()

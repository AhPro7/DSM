"""
DSM-ASR Model

Decoder-only ASR model using Delayed Streams Modeling.
Uses Mimi audio tokens as input and Qwen3-0.6B as the transformer backbone.

Architecture:
    audio_tokens[T, Q] → Q audio embedding tables (summed) → audio_emb[T, D]
    text_tokens[T]     → Qwen3 text embedding             → text_emb[T, D]
    combined[T, D] = audio_emb + text_emb
    hidden = Qwen3_transformer(combined)
    logits = lm_head(hidden) → text predictions
    loss = masked_cross_entropy(logits, text_targets, text_loss_mask)
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
    """Output container for DSM-ASR model."""
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None


class DsmAsrModel(nn.Module):
    """
    DSM-ASR: Delayed Streams Modeling for Automatic Speech Recognition.

    The model fuses audio and text embeddings at each timestep and predicts
    the next text token using the Qwen3-0.6B transformer backbone.

    Components:
        - audio_embeddings: Q separate Embedding tables (one per codebook),
          outputs are summed. Shape: Embedding(audio_vocab_size + 1, model_dim)
        - backbone: Qwen3-0.6B-Base transformer (loaded from HuggingFace)
        - text_embedding: reused from Qwen3's embed_tokens
        - text_head: reused from Qwen3's lm_head
    """

    def __init__(self, config: DsmAsrConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # ── Load Qwen3-0.6B backbone ─────────────────────────────────
        print(f"🧠 Loading Qwen3 backbone: {config.qwen_model}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.qwen_model,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",  # Use scaled dot-product attention
        )

        # Get model dimension from the backbone
        self.model_dim = self.backbone.config.hidden_size  # 1024 for Qwen3-0.6B

        # ── Resize text embeddings for special tokens ────────────────
        if tokenizer is not None:
            self.backbone.resize_token_embeddings(len(tokenizer))
            self.text_vocab_size = len(tokenizer)
        else:
            self.text_vocab_size = self.backbone.config.vocab_size

        # ── Audio embedding tables ───────────────────────────────────
        # One embedding table per codebook. Outputs will be summed.
        # +1 for the audio PAD token
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(
                config.audio_vocab_size + 1,  # 2048 + 1 (PAD)
                self.model_dim,
                padding_idx=config.audio_pad_token,
            )
            for _ in range(config.num_codebooks)
        ])

        # Initialize audio embeddings with small values
        for emb in self.audio_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            # Zero out the padding embedding
            with torch.no_grad():
                emb.weight[config.audio_pad_token] = 0.0

        # ── Audio projection (optional: project summed audio to model dim) ──
        # This helps when audio embedding dim differs from model dim
        self.audio_proj = nn.Linear(self.model_dim, self.model_dim, bias=False)
        nn.init.eye_(self.audio_proj.weight)  # Start as identity

        # ── Audio scale (learnable mixing weight) ────────────────────
        self.audio_scale = nn.Parameter(torch.ones(1))

        print(f"   Model dim: {self.model_dim}")
        print(f"   Audio codebooks: {config.num_codebooks}")
        print(f"   Audio vocab: {config.audio_vocab_size} (+1 PAD)")
        print(f"   Text vocab: {self.text_vocab_size}")
        print(f"   Total Qwen3 params: {sum(p.numel() for p in self.backbone.parameters()):,}")
        print(f"   Audio emb params: {sum(p.numel() for p in self.audio_embeddings.parameters()):,}")

    def get_text_embedding(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from Qwen3's embed_tokens."""
        return self.backbone.model.embed_tokens(text_tokens)

    def get_audio_embedding(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute audio embedding by summing all codebook embeddings.

        Args:
            audio_tokens: [B, T, Q] int tensor of codebook indices

        Returns:
            audio_emb: [B, T, D] tensor
        """
        # Sum embeddings from each codebook
        audio_emb = torch.zeros(
            audio_tokens.shape[0],
            audio_tokens.shape[1],
            self.model_dim,
            device=audio_tokens.device,
            dtype=next(self.audio_embeddings[0].parameters()).dtype,
        )
        for q, emb in enumerate(self.audio_embeddings):
            audio_emb = audio_emb + emb(audio_tokens[:, :, q])

        # Project and scale
        audio_emb = self.audio_proj(audio_emb) * self.audio_scale

        return audio_emb

    def forward(
        self,
        audio_tokens: torch.Tensor,       # [B, T, Q]
        text_tokens: torch.Tensor,         # [B, T]
        attention_mask: torch.Tensor,      # [B, T]
        text_loss_mask: torch.Tensor,      # [B, T]
        text_targets: Optional[torch.Tensor] = None,  # [B, T]
    ) -> DsmAsrOutput:
        """
        Forward pass of DSM-ASR.

        1. Embed audio tokens (sum of Q codebook embeddings)
        2. Embed text tokens (from Qwen3 embedding table)
        3. Sum audio + text embeddings
        4. Pass through Qwen3 transformer
        5. Compute logits and masked loss

        Args:
            audio_tokens: [B, T, Q] Mimi codebook indices per frame
            text_tokens: [B, T] text token IDs (teacher-forced during training)
            attention_mask: [B, T] 1.0 for valid frames, 0.0 for padding
            text_loss_mask: [B, T] 1.0 where loss should be computed
            text_targets: [B, T] target text token IDs (shifted by 1)

        Returns:
            DsmAsrOutput with loss, logits, and optional hidden states
        """
        B, T, Q = audio_tokens.shape

        # 1. Audio embedding: sum of Q codebook embeddings
        audio_emb = self.get_audio_embedding(audio_tokens)  # [B, T, D]

        # 2. Text embedding: from Qwen3's embedding table
        text_emb = self.get_text_embedding(text_tokens)     # [B, T, D]

        # 3. Sum audio + text (the DSM fusion)
        combined = audio_emb + text_emb  # [B, T, D]

        # 4. Pass through Qwen3 transformer (using inputs_embeds, not input_ids)
        # Create a proper causal attention mask
        outputs = self.backbone(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        # 5. Get logits from the LM head
        # The backbone's forward with inputs_embeds returns logits directly
        logits = outputs.logits  # [B, T, text_vocab_size]

        # 6. Compute masked loss
        loss = None
        if text_targets is not None:
            # Flatten for cross-entropy
            logits_flat = logits.view(-1, logits.size(-1))   # [B*T, V]
            targets_flat = text_targets.view(-1)               # [B*T]
            mask_flat = text_loss_mask.view(-1)                # [B*T]

            # Compute per-token loss
            loss_per_token = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction="none",
                ignore_index=-100,
            )

            # Apply text loss mask: only compute loss on text token positions
            masked_loss = loss_per_token * mask_flat

            # Average over non-zero positions
            num_text_tokens = mask_flat.sum().clamp(min=1.0)
            loss = masked_loss.sum() / num_text_tokens

        return DsmAsrOutput(loss=loss, logits=logits)

    def freeze_backbone(self):
        """Freeze Qwen3 transformer and LM head, keep audio embeddings trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep audio embeddings trainable
        for param in self.audio_embeddings.parameters():
            param.requires_grad = True
        for param in self.audio_proj.parameters():
            param.requires_grad = True
        self.audio_scale.requires_grad = True
        print("❄️  Backbone frozen. Training audio embeddings only.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔥 Backbone unfrozen. Full fine-tuning enabled.")

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def test_forward_pass():
    """Test the model with dummy data."""
    print("Testing DSM-ASR model forward pass...")

    config = DsmAsrConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": config.special_tokens})

    model = DsmAsrModel(config, tokenizer=tokenizer)
    model.eval()

    # Dummy inputs
    B, T, Q = 2, 50, config.num_codebooks
    audio_tokens = torch.randint(0, config.audio_vocab_size, (B, T, Q))
    text_tokens = torch.randint(0, 100, (B, T))
    attention_mask = torch.ones(B, T)
    text_loss_mask = torch.zeros(B, T)
    text_loss_mask[:, 10:30] = 1.0  # Some text in middle frames
    text_targets = torch.randint(0, 100, (B, T))
    text_targets[:, :10] = -100
    text_targets[:, 30:] = -100

    with torch.no_grad():
        output = model(audio_tokens, text_tokens, attention_mask, text_loss_mask, text_targets)

    print(f"\n  Loss: {output.loss.item():.4f}")
    print(f"  Logits shape: {output.logits.shape}")
    print(f"  Expected: [{B}, {T}, {model.text_vocab_size}]")
    assert output.logits.shape == (B, T, model.text_vocab_size)
    assert output.loss is not None and output.loss.item() > 0

    # Test freeze/unfreeze
    model.freeze_backbone()
    trainable = model.get_trainable_params()
    total = model.get_total_params()
    print(f"  Trainable (frozen): {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model.unfreeze_backbone()
    trainable = model.get_trainable_params()
    print(f"  Trainable (unfrozen): {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    print("\n✅ Model forward pass test PASSED!")


if __name__ == "__main__":
    test_forward_pass()

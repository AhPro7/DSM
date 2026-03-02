"""
DSM-ASR Model (Audio Prefix → Text Generation)

Decoder-only model that treats audio tokens as a prefix and generates text.
The model learns alignment between audio and text by itself.

Architecture:
    Audio codes [T, Q] → Q Embedding tables (summed) → audio_emb [T, D]
    Text tokens [N]    → Qwen3 text embedding          → text_emb [N, D]
    
    Full sequence: [audio_emb_1, ..., audio_emb_T, text_emb_1, ..., text_emb_N]
    → Qwen3 transformer (causal attention)
    → LM head → logits
    → Loss only on text positions
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


class DsmAsrModel(nn.Module):
    """
    Audio-Prefix ASR using Qwen3-0.6B.

    The audio tokens (from Mimi) are embedded and placed as a prefix.
    The model then attends to the full audio prefix and generates text
    autoregressively. Loss is computed only on text token positions.
    """

    def __init__(self, config: DsmAsrConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # ── Load Qwen3-0.6B backbone ─────────────────────────────────
        print(f"🧠 Loading Qwen3 backbone: {config.qwen_model}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.qwen_model,
            dtype=torch.bfloat16 if config.bf16 else torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

        self.model_dim = self.backbone.config.hidden_size  # 1024

        # ── Resize embeddings for special tokens ─────────────────────
        if tokenizer is not None:
            self.backbone.resize_token_embeddings(len(tokenizer))
            self.text_vocab_size = len(tokenizer)
        else:
            self.text_vocab_size = self.backbone.config.vocab_size

        # ── Audio embedding tables ───────────────────────────────────
        # One embedding per codebook, outputs are summed
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(
                config.audio_vocab_size + 1,  # +1 for PAD token
                self.model_dim,
                padding_idx=config.audio_pad_token,
            )
            for _ in range(config.num_codebooks)
        ])

        # Initialize audio embeddings
        for emb in self.audio_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight[config.audio_pad_token] = 0.0

        # Learnable scale for audio embeddings
        self.audio_scale = nn.Parameter(torch.ones(1))

        print(f"   Model dim: {self.model_dim}")
        print(f"   Audio codebooks: {config.num_codebooks}")
        print(f"   Audio vocab: {config.audio_vocab_size} (+1 PAD)")
        print(f"   Text vocab: {self.text_vocab_size}")
        total_qwen = sum(p.numel() for p in self.backbone.parameters())
        total_audio = sum(p.numel() for p in self.audio_embeddings.parameters())
        print(f"   Qwen3 params: {total_qwen:,}")
        print(f"   Audio emb params: {total_audio:,}")

    def get_audio_embedding(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed audio tokens by summing all codebook embeddings.

        Args:
            audio_tokens: [B, T, Q] codebook indices

        Returns:
            audio_emb: [B, T, D]
        """
        B, T, Q = audio_tokens.shape
        audio_emb = torch.zeros(
            B, T, self.model_dim,
            device=audio_tokens.device,
            dtype=next(self.audio_embeddings[0].parameters()).dtype,
        )
        for q, emb in enumerate(self.audio_embeddings):
            audio_emb = audio_emb + emb(audio_tokens[:, :, q])

        return audio_emb * self.audio_scale

    def get_text_embedding(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Get text embeddings from Qwen3's embed_tokens."""
        return self.backbone.model.embed_tokens(text_tokens)

    def forward(
        self,
        audio_tokens: torch.Tensor,         # [B, T_audio, Q]
        text_input_ids: torch.Tensor,        # [B, N_text]
        text_target_ids: torch.Tensor,       # [B, N_text]
        audio_lengths: torch.Tensor,         # [B]
        text_lengths: torch.Tensor,          # [B]
    ) -> DsmAsrOutput:
        """
        Forward pass: audio prefix + text generation.

        Builds the full sequence [audio_embs | text_embs],
        runs through transformer, computes loss on text positions only.
        """
        B = audio_tokens.shape[0]
        T_audio = audio_tokens.shape[1]
        N_text = text_input_ids.shape[1]
        total_len = T_audio + N_text

        # 1. Get embeddings
        audio_emb = self.get_audio_embedding(audio_tokens)  # [B, T_audio, D]
        text_emb = self.get_text_embedding(text_input_ids)   # [B, N_text, D]

        # 2. Concatenate: [audio_prefix | text_sequence]
        combined_emb = torch.cat([audio_emb, text_emb], dim=1)  # [B, T_audio + N_text, D]

        # Cast to backbone dtype (audio embs are float32, backbone may be bfloat16)
        backbone_dtype = next(self.backbone.model.embed_tokens.parameters()).dtype
        combined_emb = combined_emb.to(dtype=backbone_dtype)

        # 3. Build attention mask (1 for valid, 0 for padding)
        attention_mask = torch.zeros(B, total_len, device=audio_tokens.device, dtype=combined_emb.dtype)
        for i in range(B):
            valid_len = audio_lengths[i].item() + text_lengths[i].item()
            attention_mask[i, :valid_len] = 1.0

        # 4. Forward through Qwen3 transformer
        outputs = self.backbone(
            inputs_embeds=combined_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # [B, total_len, vocab_size]

        # 5. Compute loss only on text positions
        # The text prediction starts at position T_audio (predicting text_target from text_input)
        text_logits = logits[:, T_audio:, :]  # [B, N_text, vocab_size]

        # Flatten for cross-entropy
        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_target_ids.reshape(-1),
            ignore_index=-100,
        )

        return DsmAsrOutput(loss=loss, logits=text_logits)

    def freeze_backbone(self):
        """Freeze Qwen3, keep audio embeddings trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.audio_embeddings.parameters():
            param.requires_grad = True
        self.audio_scale.requires_grad = True
        print("❄️  Backbone frozen. Training audio embeddings only.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔥 Backbone unfrozen. Full fine-tuning.")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        audio_tokens: torch.Tensor,       # [1, T_audio, Q]
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list:
        """
        Generate text from audio prefix.

        Args:
            audio_tokens: [1, T_audio, Q] audio codes from Mimi
            tokenizer: Tokenizer with special tokens
            max_new_tokens: Maximum tokens to generate
            temperature: 0 = greedy, >0 = sampling

        Returns:
            List of generated token IDs
        """
        self.eval()
        device = audio_tokens.device
        start_text_id = tokenizer.convert_tokens_to_ids("<|start_text|>")
        end_text_id = tokenizer.convert_tokens_to_ids("<|end_text|>")

        # Get audio prefix embeddings
        audio_emb = self.get_audio_embedding(audio_tokens)  # [1, T, D]

        # Start with START_TEXT token
        generated = [start_text_id]
        text_ids = torch.tensor([[start_text_id]], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Get text embeddings for generated tokens so far
            text_emb = self.get_text_embedding(text_ids)  # [1, len, D]

            # Concat audio prefix + text so far
            combined = torch.cat([audio_emb, text_emb], dim=1)
            backbone_dtype = next(self.backbone.model.embed_tokens.parameters()).dtype
            combined = combined.to(dtype=backbone_dtype)
            attention_mask = torch.ones(1, combined.shape[1], device=device, dtype=combined.dtype)

            # Forward
            outputs = self.backbone(
                inputs_embeds=combined,
                attention_mask=attention_mask,
                return_dict=True,
            )

            # Get next token logits (last position)
            next_logits = outputs.logits[0, -1, :]

            # Decode
            if temperature == 0.0:
                next_token = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

            # Stop if END_TEXT
            if next_token == end_text_id:
                break

            # Append to input for next step
            text_ids = torch.cat([
                text_ids,
                torch.tensor([[next_token]], dtype=torch.long, device=device)
            ], dim=1)

        return generated


def test_forward_pass():
    """Test model with dummy data."""
    print("Testing DSM-ASR model...")

    config = DsmAsrConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": config.special_tokens})

    model = DsmAsrModel(config, tokenizer=tokenizer)
    model.eval()

    # Dummy inputs
    B = 2
    T_audio, Q = 50, config.num_codebooks
    N_text = 20

    audio_tokens = torch.randint(0, config.audio_vocab_size, (B, T_audio, Q))
    text_input_ids = torch.randint(0, 100, (B, N_text))
    text_target_ids = torch.randint(0, 100, (B, N_text))
    audio_lengths = torch.tensor([T_audio, T_audio])
    text_lengths = torch.tensor([N_text, N_text])

    with torch.no_grad():
        output = model(audio_tokens, text_input_ids, text_target_ids, audio_lengths, text_lengths)

    print(f"\n  Loss: {output.loss.item():.4f}")
    print(f"  Logits shape: {output.logits.shape}")
    assert output.logits.shape == (B, N_text, model.text_vocab_size)
    assert output.loss.item() > 0

    # Test freeze/unfreeze
    model.freeze_backbone()
    t1 = model.get_trainable_params()
    total = model.get_total_params()
    print(f"  Trainable (frozen): {t1:,} / {total:,} ({100*t1/total:.1f}%)")

    model.unfreeze_backbone()
    t2 = model.get_trainable_params()
    print(f"  Trainable (unfrozen): {t2:,} / {total:,} ({100*t2/total:.1f}%)")

    print("\n✅ Model test PASSED!")


if __name__ == "__main__":
    test_forward_pass()

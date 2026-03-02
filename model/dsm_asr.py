"""
DSM-ASR Model v2 (Audio Prefix → Text Generation)

Key improvements over v1:
1. Audio Adapter MLP: 2-layer MLP with GELU + dropout after codebook sum
   → Transforms random audio embeddings into useful representations
2. Text Prompt: prepends task instruction before audio
   → Tells the LM "you are doing ASR", activating its text generation capabilities
3. No backbone freezing: train everything together from start
   → Audio embeddings and backbone co-adapt simultaneously
4. Label smoothing in loss computation
   → Better generalization

Architecture:
    [text_prompt_emb] [audio_adapter(sum(codebook_embs))] [START_TEXT] [text_tokens]
    |<--- no loss --->|<----------- no loss ------------>|<--- loss on text part --->|
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
    """
    Multi-layer MLP adapter that transforms summed codebook embeddings
    into representations the LM backbone can understand.
    
    This is CRITICAL — without it, randomly initialized embeddings
    are in a completely different space than the LM's text embeddings.
    The adapter learns to project audio info into the LM's representation space.
    """
    def __init__(self, dim: int, num_layers: int = 2, dropout: float = 0.1):
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
        return self.norm(x + self.mlp(x))  # Residual connection


class DsmAsrModel(nn.Module):
    """
    Audio-Prefix ASR v2 with Audio Adapter + Text Prompt.
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
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(
                config.audio_vocab_size + 1,
                self.model_dim,
                padding_idx=config.audio_pad_token,
            )
            for _ in range(config.num_codebooks)
        ])

        for emb in self.audio_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight[config.audio_pad_token] = 0.0

        # ── Audio Adapter MLP (NEW) ──────────────────────────────────
        self.audio_adapter = AudioAdapter(
            dim=self.model_dim,
            num_layers=config.audio_adapter_layers,
            dropout=config.audio_adapter_dropout,
        )

        # ── Learnable audio scale ────────────────────────────────────
        self.audio_scale = nn.Parameter(torch.ones(1))

        # ── Precompute prompt token IDs ──────────────────────────────
        self._prompt_ids = None
        if tokenizer is not None and config.text_prompt:
            self._prompt_ids = tokenizer.encode(config.text_prompt, add_special_tokens=False)

        # ── Print stats ──────────────────────────────────────────────
        total_qwen = sum(p.numel() for p in self.backbone.parameters())
        total_audio = sum(p.numel() for p in self.audio_embeddings.parameters())
        total_adapter = sum(p.numel() for p in self.audio_adapter.parameters())
        print(f"   Model dim: {self.model_dim}")
        print(f"   Audio codebooks: {config.num_codebooks}")
        print(f"   Text vocab: {self.text_vocab_size}")
        print(f"   Qwen3 params: {total_qwen:,}")
        print(f"   Audio emb params: {total_audio:,}")
        print(f"   Audio adapter params: {total_adapter:,}")
        if self._prompt_ids:
            print(f"   Prompt tokens: {len(self._prompt_ids)}")

    def get_audio_embedding(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Embed + adapt audio tokens: sum codebooks → adapter MLP."""
        B, T, Q = audio_tokens.shape
        audio_emb = torch.zeros(
            B, T, self.model_dim,
            device=audio_tokens.device,
            dtype=next(self.audio_embeddings[0].parameters()).dtype,
        )
        for q, emb in enumerate(self.audio_embeddings):
            audio_emb = audio_emb + emb(audio_tokens[:, :, q])

        audio_emb = audio_emb * self.audio_scale

        # Pass through adapter MLP — critical for learning
        audio_emb = self.audio_adapter(audio_emb)

        return audio_emb

    def get_text_embedding(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.model.embed_tokens(text_tokens)

    def get_prompt_embedding(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """Get text prompt embeddings (e.g., 'Transcribe the following audio:')"""
        if self._prompt_ids is None:
            return None
        prompt_ids = torch.tensor([self._prompt_ids], dtype=torch.long, device=device)
        prompt_emb = self.get_text_embedding(prompt_ids)  # [1, P, D]
        prompt_emb = prompt_emb.expand(batch_size, -1, -1)  # [B, P, D]
        return prompt_emb

    def forward(
        self,
        audio_tokens: torch.Tensor,         # [B, T_audio, Q]
        text_input_ids: torch.Tensor,        # [B, N_text]
        text_target_ids: torch.Tensor,       # [B, N_text]
        audio_lengths: torch.Tensor,         # [B]
        text_lengths: torch.Tensor,          # [B]
    ) -> DsmAsrOutput:
        B = audio_tokens.shape[0]
        T_audio = audio_tokens.shape[1]
        N_text = text_input_ids.shape[1]
        device = audio_tokens.device

        # 1. Get embeddings
        audio_emb = self.get_audio_embedding(audio_tokens)
        text_emb = self.get_text_embedding(text_input_ids)

        # 2. Build sequence: [prompt | audio | text]
        parts = []
        prompt_len = 0
        prompt_emb = self.get_prompt_embedding(B, device)
        if prompt_emb is not None:
            parts.append(prompt_emb)
            prompt_len = prompt_emb.shape[1]
        parts.append(audio_emb)
        parts.append(text_emb)

        combined_emb = torch.cat(parts, dim=1)
        total_len = combined_emb.shape[1]

        # Cast to backbone dtype
        backbone_dtype = next(self.backbone.model.embed_tokens.parameters()).dtype
        combined_emb = combined_emb.to(dtype=backbone_dtype)

        # 3. Build attention mask
        attention_mask = torch.zeros(B, total_len, device=device, dtype=combined_emb.dtype)
        for i in range(B):
            valid_len = prompt_len + audio_lengths[i].item() + text_lengths[i].item()
            attention_mask[i, :valid_len] = 1.0

        # 4. Forward through transformer
        outputs = self.backbone(
            inputs_embeds=combined_emb,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits

        # 5. Loss only on text positions (after prompt + audio)
        text_start = prompt_len + T_audio
        text_logits = logits[:, text_start:text_start + N_text, :]

        loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_target_ids.reshape(-1),
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )

        return DsmAsrOutput(loss=loss, logits=text_logits)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.audio_embeddings.parameters():
            param.requires_grad = True
        for param in self.audio_adapter.parameters():
            param.requires_grad = True
        self.audio_scale.requires_grad = True
        print("❄️  Backbone frozen. Training audio embeddings + adapter.")

    def unfreeze_backbone(self):
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
        audio_tokens: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list:
        self.eval()
        device = audio_tokens.device
        start_text_id = tokenizer.convert_tokens_to_ids("<|start_text|>")
        end_text_id = tokenizer.convert_tokens_to_ids("<|end_text|>")

        # Build prefix: [prompt | audio]
        audio_emb = self.get_audio_embedding(audio_tokens)
        parts = []
        prompt_emb = self.get_prompt_embedding(1, device)
        if prompt_emb is not None:
            parts.append(prompt_emb)
        parts.append(audio_emb)
        prefix_emb = torch.cat(parts, dim=1)  # [1, P+T, D]

        backbone_dtype = next(self.backbone.model.embed_tokens.parameters()).dtype
        prefix_emb = prefix_emb.to(dtype=backbone_dtype)

        # Start generation with START_TEXT
        generated = [start_text_id]
        text_ids = torch.tensor([[start_text_id]], dtype=torch.long, device=device)

        # Use KV cache for efficiency
        past_key_values = None
        first_step = True

        for _ in range(max_new_tokens):
            if first_step:
                # First step: process full prefix + start token
                text_emb = self.get_text_embedding(text_ids).to(dtype=backbone_dtype)
                combined = torch.cat([prefix_emb, text_emb], dim=1)
                attention_mask = torch.ones(1, combined.shape[1], device=device, dtype=backbone_dtype)
                outputs = self.backbone(
                    inputs_embeds=combined,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                first_step = False
            else:
                # Subsequent steps: only process the new token with KV cache
                new_token_emb = self.get_text_embedding(
                    torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
                ).to(dtype=backbone_dtype)
                cache_len = past_key_values[0][0].shape[2]
                attention_mask = torch.ones(1, cache_len + 1, device=device, dtype=backbone_dtype)
                outputs = self.backbone(
                    inputs_embeds=new_token_emb,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

            next_logits = outputs.logits[0, -1, :]

            if temperature == 0.0:
                next_token = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

            if next_token == end_text_id:
                break

        return generated


def test_forward_pass():
    print("Testing DSM-ASR model v2...")
    config = DsmAsrConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": config.special_tokens})

    model = DsmAsrModel(config, tokenizer=tokenizer)
    model.eval()

    B, T_audio, Q, N_text = 2, 50, config.num_codebooks, 20
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

    model.freeze_backbone()
    t1 = model.get_trainable_params()
    total = model.get_total_params()
    print(f"  Trainable (frozen): {t1:,} / {total:,} ({100*t1/total:.1f}%)")
    model.unfreeze_backbone()
    t2 = model.get_trainable_params()
    print(f"  Trainable (unfrozen): {t2:,} / {total:,}")

    print("\n✅ Model v2 test PASSED!")


if __name__ == "__main__":
    test_forward_pass()

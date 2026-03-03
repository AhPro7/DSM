"""
DSM-ASR Model v4 — Instruction Fine-Tuning

Sequence layout:
    [instruction_text_embs] [audio_adapter(Σ codebook_embs)] [separator_text_embs] [target_text_embs]
    |<------------ text embedding --------->|<-- audio embedding -->|<-- text -->|<-- text -->|
    |<--------------------- NO LOSS -------------------------------->|<----- LOSS HERE ----->|

Key components:
1. Audio Embeddings: 8 codebook embedding tables, summed per frame
2. Audio Adapter: 2-layer MLP (dim → 4×dim → dim) with GELU + residual
3. Qwen3-0.6B backbone: processes the mixed embedding sequence
4. Loss: cross-entropy only on target text positions
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
    """MLP that projects audio embeddings into the LM's representation space."""
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
    def __init__(self, config: DsmAsrConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # Backbone
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

        # Audio embeddings (8 codebooks, summed)
        self.audio_embeddings = nn.ModuleList([
            nn.Embedding(config.audio_vocab_size + 1, self.model_dim,
                        padding_idx=config.audio_pad_token)
            for _ in range(config.num_codebooks)
        ])
        for emb in self.audio_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[config.audio_pad_token] = 0.0

        # Audio adapter
        self.audio_adapter = AudioAdapter(
            self.model_dim, config.audio_adapter_layers, config.audio_adapter_dropout)

        # Audio scale
        self.audio_scale = nn.Parameter(torch.ones(1))

        # Pre-tokenize instruction/separator
        self._instr_ids = None
        self._sep_ids = None
        if tokenizer:
            self._instr_ids = tokenizer.encode(config.instruction, add_special_tokens=False)
            self._sep_ids = tokenizer.encode(config.separator, add_special_tokens=False)

        # Stats
        n_qwen = sum(p.numel() for p in self.backbone.parameters())
        n_audio = sum(p.numel() for p in self.audio_embeddings.parameters())
        n_adapt = sum(p.numel() for p in self.audio_adapter.parameters())
        print(f"   Qwen3: {n_qwen:,}  Audio: {n_audio:,}  Adapter: {n_adapt:,}")

    @property
    def backbone_dtype(self):
        return next(self.backbone.model.embed_tokens.parameters()).dtype

    def get_text_emb(self, token_ids):
        """[*, L] → [*, L, D]"""
        return self.backbone.model.embed_tokens(token_ids)

    def get_audio_emb(self, audio_tokens):
        """[B, T, Q] → [B, T, D]"""
        B, T, Q = audio_tokens.shape
        emb = torch.zeros(B, T, self.model_dim,
                          device=audio_tokens.device,
                          dtype=next(self.audio_embeddings[0].parameters()).dtype)
        for q, e in enumerate(self.audio_embeddings):
            emb = emb + e(audio_tokens[:, :, q])
        return self.audio_adapter(emb * self.audio_scale)

    def build_input_embeds(self, instruction_ids, audio_tokens, separator_ids,
                           target_ids, device):
        """
        Build the full input embedding sequence:
            [instruction_embs | audio_embs | separator_embs | target_embs]
        """
        B = audio_tokens.shape[0]
        dtype = self.backbone_dtype

        # Text parts
        instr_emb = self.get_text_emb(
            instruction_ids.unsqueeze(0).expand(B, -1).to(device)).to(dtype)
        sep_emb = self.get_text_emb(
            separator_ids.unsqueeze(0).expand(B, -1).to(device)).to(dtype)
        target_emb = self.get_text_emb(target_ids.to(device)).to(dtype)

        # Audio part
        audio_emb = self.get_audio_emb(audio_tokens.to(device)).to(dtype)

        # Concatenate: [instruction | audio | separator | target]
        combined = torch.cat([instr_emb, audio_emb, sep_emb, target_emb], dim=1)
        return combined

    def forward(self, instruction_ids, audio_tokens, separator_ids,
                target_ids, labels, audio_lengths, target_lengths):
        """
        Full forward pass with loss on target positions only.
        """
        device = audio_tokens.device
        B = audio_tokens.shape[0]

        # Build embeddings
        combined = self.build_input_embeds(
            instruction_ids, audio_tokens, separator_ids, target_ids, device)
        total_len = combined.shape[1]

        # Attention mask (1 for real tokens, 0 for padding)
        instr_len = instruction_ids.shape[0]
        sep_len = separator_ids.shape[0]
        max_audio = audio_tokens.shape[1]
        max_target = target_ids.shape[1]

        attn_mask = torch.zeros(B, total_len, device=device, dtype=combined.dtype)
        for i in range(B):
            valid = instr_len + audio_lengths[i].item() + sep_len + target_lengths[i].item()
            attn_mask[i, :valid] = 1.0

        # Forward through Qwen3
        outputs = self.backbone(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            return_dict=True,
        )
        logits = outputs.logits  # [B, total_len, vocab]

        # Shift for next-token prediction:
        # logits[t] predicts token at position t+1
        # So logits at positions [target_start-1 : target_start+max_target-1] predict target
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )

        return DsmAsrOutput(loss=loss, logits=logits)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.audio_embeddings.parameters():
            p.requires_grad = True
        for p in self.audio_adapter.parameters():
            p.requires_grad = True
        self.audio_scale.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, audio_tokens, tokenizer, max_new_tokens=256, temperature=0.0):
        """
        Inference: given audio, generate transcription.

        audio_tokens: [1, T, Q]
        Returns: decoded text string
        """
        self.eval()
        device = audio_tokens.device
        dtype = self.backbone_dtype

        # Build prefix: [instruction | audio | separator]
        instr_ids = torch.tensor(self._instr_ids, dtype=torch.long, device=device)
        sep_ids = torch.tensor(self._sep_ids, dtype=torch.long, device=device)

        instr_emb = self.get_text_emb(instr_ids.unsqueeze(0)).to(dtype)
        audio_emb = self.get_audio_emb(audio_tokens).to(dtype)
        sep_emb = self.get_text_emb(sep_ids.unsqueeze(0)).to(dtype)

        prefix = torch.cat([instr_emb, audio_emb, sep_emb], dim=1)
        attn_mask = torch.ones(1, prefix.shape[1], device=device, dtype=dtype)

        # First forward: process entire prefix
        outputs = self.backbone(
            inputs_embeds=prefix,
            attention_mask=attn_mask,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        next_logits = outputs.logits[0, -1, :]

        # Autoregressive decoding with KV cache
        generated = []
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                tok = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                tok = torch.multinomial(probs, 1).item()

            if tok == tokenizer.eos_token_id:
                break
            generated.append(tok)

            # Next step with KV cache
            tok_emb = self.get_text_emb(
                torch.tensor([[tok]], device=device)).to(dtype)
            if hasattr(past, 'get_seq_length'):
                cache_len = past.get_seq_length()
            else:
                cache_len = past[0][0].shape[2]
            attn_mask = torch.ones(1, cache_len + 1, device=device, dtype=dtype)

            outputs = self.backbone(
                inputs_embeds=tok_emb,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[0, -1, :]

        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def test_model():
    print("Testing DSM-ASR v4...")
    config = DsmAsrConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model, trust_remote_code=True)
    model = DsmAsrModel(config, tokenizer=tokenizer)
    model.eval()

    B, T, Q = 2, 50, config.num_codebooks
    instr = torch.tensor(model._instr_ids, dtype=torch.long)
    sep = torch.tensor(model._sep_ids, dtype=torch.long)
    audio = torch.randint(0, config.audio_vocab_size, (B, T, Q))
    target = torch.randint(0, 100, (B, 20))
    total = instr.shape[0] + T + sep.shape[0] + 20
    labels = torch.full((B, total), -100)
    labels[:, -20:] = target
    a_lens = torch.tensor([T, T])
    t_lens = torch.tensor([20, 20])

    with torch.no_grad():
        out = model(instr, audio, sep, target, labels, a_lens, t_lens)
    print(f"  Loss: {out.loss.item():.4f}")
    print(f"  Logits: {out.logits.shape}")
    print("✅ Model v4 PASSED!")


if __name__ == "__main__":
    test_model()

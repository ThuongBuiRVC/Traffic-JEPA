from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from traffic_jepa.modeling.hf_auth import resolve_hf_token
from traffic_jepa.modeling.tokenizer import SimpleTokenizer


class TinyPredictorBackbone(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.config = type("TinyConfig", (), {"hidden_size": hidden_size})()

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None, **_: object
    ) -> torch.Tensor:
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        return self.norm(self.layers(inputs_embeds, src_key_padding_mask=key_padding_mask))


class Predictor(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        vision_dim: int,
        output_dim: int,
        num_layers: int = 8,
        trust_remote_code: bool = True,
        hf_token: bool | str | None = True,
        torch_dtype: str | None = "auto",
        tiny: bool = False,
    ) -> None:
        super().__init__()
        self.tiny = tiny
        if tiny:
            self.backbone = TinyPredictorBackbone(layers=max(1, min(num_layers, 2)))
            self.tokenizer = SimpleTokenizer()
        else:
            token = resolve_hf_token(hf_token)
            kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "token": token}
            if torch_dtype:
                kwargs["torch_dtype"] = torch_dtype
            self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, token=token
            )
            if hasattr(self.backbone, "layers"):
                self.backbone.layers = nn.ModuleList(list(self.backbone.layers)[-num_layers:])
            elif hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
                self.backbone.model.layers = nn.ModuleList(
                    list(self.backbone.model.layers)[-num_layers:]
                )
            self._disable_causal_attention()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        hidden_size = int(getattr(self.backbone.config, "hidden_size", output_dim))
        self.vision_projection = nn.Linear(vision_dim, hidden_size)
        self.output_projection = nn.Linear(hidden_size, output_dim)

    @property
    def query_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def _disable_causal_attention(self) -> None:
        if hasattr(self.backbone, "config"):
            self.backbone.config.is_causal = False
        for module in self.backbone.modules():
            if hasattr(module, "is_causal"):
                module.is_causal = False

    def _embed_query(self, query_input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = getattr(self.backbone, "embed_tokens", None)
        if embeddings is None and hasattr(self.backbone, "get_input_embeddings"):
            embeddings = self.backbone.get_input_embeddings()
        if embeddings is None and hasattr(self.backbone, "model"):
            embeddings = self.backbone.model.embed_tokens
        return embeddings(query_input_ids)

    def _bidirectional_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        min_value = torch.finfo(dtype).min
        padding = (1 - attention_mask).to(dtype=dtype) * min_value
        seq_len = attention_mask.shape[1]
        return padding[:, None, None, :].expand(-1, 1, seq_len, -1)

    def _forward_bidirectional_backbone(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        backbone = self.backbone
        if not all(hasattr(backbone, attr) for attr in ("layers", "norm", "rotary_emb")):
            outputs = backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            if torch.is_tensor(outputs):           # some backbones return the tensor directly
                return outputs
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs.hidden_states[-1]
            return hidden

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(
            0
        )
        position_embeddings = backbone.rotary_emb(inputs_embeds, position_ids=position_ids)
        hidden = inputs_embeds
        bidirectional_mask = self._bidirectional_attention_mask(attention_mask, dtype=hidden.dtype)
        for layer in backbone.layers:
            hidden = layer(
                hidden,
                attention_mask=bidirectional_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=False,
                is_causal=False,
            )
        return backbone.norm(hidden)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        visual_embeds = self.vision_projection(
            visual_tokens.to(dtype=self.vision_projection.weight.dtype)
        )
        query_embeds = self._embed_query(query_input_ids)
        visual_embeds = visual_embeds.to(dtype=query_embeds.dtype)
        inputs_embeds = torch.cat([visual_embeds, query_embeds], dim=1)
        visual_mask = torch.ones(
            visual_embeds.shape[:2],
            dtype=query_attention_mask.dtype,
            device=query_attention_mask.device,
        )
        attention_mask = torch.cat([visual_mask, query_attention_mask], dim=1)
        if isinstance(self.backbone, TinyPredictorBackbone):
            hidden = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            hidden = self._forward_bidirectional_backbone(inputs_embeds, attention_mask)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.output_projection(pooled.to(dtype=self.output_projection.weight.dtype))

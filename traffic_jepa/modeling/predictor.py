from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from traffic_jepa.modeling.hf_auth import resolve_hf_token


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
    ) -> None:
        super().__init__()
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
        # TrafficJEPAModel projects the visual tokens with its own `visual_proj` and calls the
        # backbone directly, so this layer is unused at run time — it is kept because the
        # trained checkpoint carries its weights.
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

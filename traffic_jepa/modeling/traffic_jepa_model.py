from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from traffic_jepa.modeling.predictor import Predictor
from traffic_jepa.modeling.labels import PHASES, SELECTORS


class TrafficJEPAModel(nn.Module):
    """Minimal WTS VQA model built only on V-JEPA world latents.

    Architecture:
      V-JEPA latent grid -> projected visual tokens
      question tokens    -> same predictor backbone
      pooled hidden      -> answer embedding
      options            -> cosine scoring


    """

    def __init__(
        self,
        predictor_model: str,
        num_layers: int = 6,
        *,
        temperature: float = 0.07,
        dropout: float = 0.1,
        token_dropout: float = 0.0,
        grad_ckpt: bool = True,
        unfreeze_backbone_layers: int = 0,
        use_prior_tokens: bool = True,
        spatial_pool_size: int = 1,
        ablation_no_visual: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.token_dropout = float(token_dropout)
        self.grad_ckpt = bool(grad_ckpt)
        self.use_prior_tokens = bool(use_prior_tokens)
        self.spatial_pool_size = int(spatial_pool_size)
        self.ablation_no_visual = bool(ablation_no_visual)

        self.predictor = Predictor(
            predictor_model,
            vision_dim=1024,
            output_dim=768,
            num_layers=num_layers,
            torch_dtype="auto",
        )
        self.tokenizer = self.predictor.tokenizer

        for p in self.predictor.backbone.parameters():
            p.requires_grad = False
        if int(unfreeze_backbone_layers) > 0:
            layers = getattr(self.predictor.backbone, "layers", None)
            if layers is not None:
                for layer in layers[-int(unfreeze_backbone_layers):]:
                    for p in layer.parameters():
                        p.requires_grad = True

        hidden = self.predictor.backbone.config.hidden_size
        self.visual_proj = nn.Sequential(
            nn.Linear(1024, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_prior_tokens:
            self.phase_embed = nn.Embedding(len(PHASES), hidden)
            self.selector_embed = nn.Embedding(len(SELECTORS), hidden)
            self.prior_norm = nn.LayerNorm(hidden)

    def forward(self, batch: dict):
        latent = batch["latent"]
        if latent.ndim != 5 or latent.shape[-1] != 1024:
            raise ValueError(f"latent must be [B, T, H, W, 1024], got {tuple(latent.shape)}")

        if self.ablation_no_visual:
            latent = torch.zeros_like(latent)

        query_embeds = self._embed_question_tokens(batch["q_ids"])

        # Keep only V-JEPA latent tokens. A small adaptive grid preserves rough
        # layout for position/orientation while avoiding PCA or object branches.
        visual_tokens = self._visual_tokens(latent)
        visual_embeds = self.visual_proj(visual_tokens).to(dtype=query_embeds.dtype)

        parts = [visual_embeds]
        masks = [torch.ones(
            (latent.shape[0], visual_embeds.shape[1]),
            dtype=batch["q_mask"].dtype,
            device=batch["q_mask"].device,
        )]
        prior_embeds = self._prior_tokens(batch, query_embeds.dtype)
        if prior_embeds is not None:
            parts.append(prior_embeds)
            masks.append(torch.ones(
                (latent.shape[0], prior_embeds.shape[1]),
                dtype=batch["q_mask"].dtype,
                device=batch["q_mask"].device,
            ))
        parts.append(query_embeds)
        masks.append(batch["q_mask"])
        inputs_embeds = torch.cat(parts, dim=1)
        attention_mask = torch.cat(masks, dim=1)

        if self.grad_ckpt and self.training:
            hidden = torch.utils.checkpoint.checkpoint(
                self.predictor._forward_bidirectional_backbone,
                inputs_embeds,
                attention_mask,
                use_reentrant=False,
            )
        else:
            hidden = self.predictor._forward_bidirectional_backbone(inputs_embeds, attention_mask)

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        y_hat = self.predictor.output_projection(pooled.float().to(self.predictor.output_projection.weight.dtype))
        return y_hat, None, None

    def _visual_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.float()
        if self.spatial_pool_size <= 1:
            return latent.mean(dim=(2, 3))
        bsz, steps, height, width, dim = latent.shape
        grid = latent.permute(0, 1, 4, 2, 3).reshape(bsz * steps, dim, height, width)
        pooled = F.adaptive_avg_pool2d(grid, (self.spatial_pool_size, self.spatial_pool_size))
        pooled = pooled.reshape(bsz, steps, dim, self.spatial_pool_size * self.spatial_pool_size)
        return pooled.permute(0, 1, 3, 2).reshape(bsz, steps * self.spatial_pool_size * self.spatial_pool_size, dim)

    def _prior_tokens(self, batch: dict, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.use_prior_tokens:
            return None
        B = batch["latent"].shape[0]
        device = batch["latent"].device
        phase_id = batch.get("phase_id")
        if phase_id is None:
            phase_id = torch.full((B,), len(PHASES) - 1, dtype=torch.long, device=device)
        else:
            phase_id = phase_id.to(device).long().clamp(0, len(PHASES) - 1)
        selector = batch.get("selector")
        if selector is None:
            selector = torch.zeros((B,), dtype=torch.long, device=device)
        else:
            selector = selector.to(device).long().clamp(0, len(SELECTORS) - 1)
        phase = self.phase_embed(phase_id)
        category = self.selector_embed(selector)
        return self.prior_norm(torch.stack([phase, category], dim=1)).to(dtype=dtype)

    def _embed_question_tokens(self, q_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.predictor._embed_query(q_ids)
        if self.training and self.token_dropout > 0.0:
            keep = (torch.rand(embeds.shape[:2], device=embeds.device) > self.token_dropout).to(embeds.dtype)
            embeds = embeds * keep.unsqueeze(-1)
        return embeds

    def option_scores(self, y_hat, _dist_logits, batch: dict) -> torch.Tensor:
        y = F.normalize(y_hat.float(), p=2, dim=-1)
        opts = F.normalize(batch["opt_vecs"].float(), p=2, dim=-1)
        scores = torch.bmm(opts, y.unsqueeze(-1)).squeeze(-1)
        if "n_opt" in batch:
            opt_idx = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
            valid = opt_idx < batch["n_opt"].to(scores.device).unsqueeze(1)
            scores = scores.masked_fill(~valid, float("-inf"))
        return scores

    def final_pred(self, y_hat, dist_logits, batch: dict) -> torch.Tensor:
        return torch.argmax(self.option_scores(y_hat, dist_logits, batch), dim=-1)

    def hybrid_loss(self, y_hat, batch: dict) -> torch.Tensor:
        target = F.normalize(batch["target_vec"].float(), p=2, dim=-1)
        y = F.normalize(y_hat.float(), p=2, dim=-1)
        target_score = torch.sum(y * target, dim=-1, keepdim=True) / self.temperature

        distractors = F.normalize(batch["distractor_vecs"].float(), p=2, dim=-1)
        dist_scores = torch.bmm(distractors, y.unsqueeze(-1)).squeeze(-1) / self.temperature
        dist_scores = dist_scores.masked_fill(batch["distractor_valid"] == 0.0, -1e4)

        logits = torch.cat([target_score, dist_scores], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def trainable_parameters(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.predictor.backbone.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen_backbone": frozen}

from __future__ import annotations

import os


def resolve_hf_token(token: bool | str | None = True) -> bool | str | None:
    """Resolve Hugging Face auth for gated model downloads.

    `True` tells transformers/huggingface_hub to use the token from `hf auth login`.
    Environment variables take precedence so non-interactive jobs can inject a token.
    """

    if isinstance(token, str):
        return token
    if token is False:
        return None
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or True

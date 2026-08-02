from __future__ import annotations

from pathlib import Path
from typing import Any


class SimpleTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"
    unk_token = "<unk>"

    def __init__(self) -> None:
        self.vocab = {self.pad_token: 0, self.eos_token: 1, self.unk_token: 2}

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def __call__(
        self,
        text: str | list[str],
        *,
        truncation: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else text
        encoded = []
        for item in texts:
            ids = [self._id(token) for token in item.strip().split()] or [self.unk_token_id]
            if add_special_tokens:
                ids.append(self.eos_token_id)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            encoded.append(ids)
        max_len = max(len(ids) for ids in encoded) if padding else None
        input_ids = []
        masks = []
        for ids in encoded:
            if max_len is not None:
                padded = ids + [self.pad_token_id] * (max_len - len(ids))
            else:
                padded = ids
            input_ids.append(padded)
            masks.append([0 if token == self.pad_token_id else 1 for token in padded])
        if return_tensors == "pt":
            import torch

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        if isinstance(text, str):
            return {"input_ids": input_ids[0], "attention_mask": masks[0]}
        return {"input_ids": input_ids, "attention_mask": masks}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        inverse = {value: key for key, value in self.vocab.items()}
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            tokens.append(inverse.get(int(token_id), self.unk_token))
        return " ".join(tokens)

    def save_pretrained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

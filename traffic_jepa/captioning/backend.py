"""Local Qwen3-VL runner — the caption stage loads the model in-process.

No server, no vLLM: `generate.py` builds chat messages and calls `QwenVL.chat` directly, the
same way `traffic_jepa.inference.predict` calls the V-JEPA encoder. A LoRA adapter is attached
with PEFT and can be switched off for one call, which is what the frame path needs (the adapter
was trained on the text-only QA path, so image prompts run on the base weights).
"""
from __future__ import annotations

import contextlib
from pathlib import Path


class QwenVL:
    """Qwen3-VL-8B, optionally with the caption LoRA attached."""

    def __init__(self, model_id: str, lora_dir: str | Path = "", *, dtype: str = "bfloat16",
                 device_map: str = "auto", max_new_tokens: int = 768) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.max_new_tokens = int(max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            trust_remote_code=True,
        )
        self.device = self.model.device
        self.has_lora = False
        if lora_dir:
            lora_dir = Path(lora_dir)
            if not (lora_dir / "adapter_config.json").is_file():
                raise SystemExit(f"FATAL: no adapter_config.json under {lora_dir}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(lora_dir))
            self.has_lora = True
        self.model.eval()

    @contextlib.contextmanager
    def base_weights(self):
        """Run the block on the base model, with the LoRA switched off (no-op in base mode)."""
        if self.has_lora:
            with self.model.disable_adapter():
                yield
        else:
            yield

    def _encode(self, messages: list[dict]):
        try:
            return self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
        except Exception:      # older processors: template to text, feed the images separately
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images = [part["image"] for m in messages
                      for part in (m["content"] if isinstance(m["content"], list) else [])
                      if part.get("type") == "image"]
            return self.processor(text=[text], images=images or None, return_tensors="pt")

    def chat(self, messages: list[dict], *, max_new_tokens: int | None = None,
             temperature: float = 0.1) -> str:
        """Run one chat turn and return the assistant text."""
        torch = self.torch
        inputs = self._encode(messages).to(self.device)
        gen_kwargs = {"max_new_tokens": int(max_new_tokens or self.max_new_tokens)}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)
        pad = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad is not None:
            gen_kwargs["pad_token_id"] = pad
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.decode(out[0][prompt_len:], skip_special_tokens=True)


def text_msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def image_msg(role: str, text: str, images: list) -> dict:
    content: list[dict] = [{"type": "text", "text": text}]
    content += [{"type": "image", "image": im} for im in images]
    return {"role": role, "content": content}

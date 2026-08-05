"""Two ways to reach Qwen3-VL, behind one interface.

`QwenVL` loads the model in-process with transformers and PEFT. It needs nothing running, but it
generates one segment at a time, which is slow over hundreds of segments.

`QwenVLServer` talks to an OpenAI-compatible endpoint, normally vLLM. The server batches requests
across its own queue, so the caption stage can push many segments at once and finish in a fraction
of the time. Serve the base model plus both adapters as separate `--lora-modules` names.

Both expose `chat(messages)` and a `base_weights()` block that turns the adapter off for one call,
which the frame path needs when the adapter is text-only.
"""
from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path


class QwenVL:
    """Qwen3-VL-8B in this process, optionally with a LoRA attached."""

    concurrency = 1                                       # generation is serialised on one GPU

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


class QwenVLServer:
    """The same interface, served over an OpenAI-compatible endpoint such as vLLM.

    `model` is the served name to use normally, `base_model` the one to fall back to inside a
    `base_weights()` block. With vLLM those are a `--lora-modules` name and the base
    `--served-model-name`, so switching between them costs nothing on the server side.
    """

    def __init__(self, base_url: str, model: str, *, base_model: str = "", api_key: str = "",
                 max_new_tokens: int = 768, timeout: float = 300.0, concurrency: int = 8) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.base_model = base_model or model
        self.api_key = api_key
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = float(timeout)
        self.concurrency = int(concurrency)
        self.has_lora = self.base_model != self.model
        self._local = threading.local()                   # base_weights is per-thread

    @contextlib.contextmanager
    def base_weights(self):
        prev = getattr(self._local, "use_base", False)
        self._local.use_base = True
        try:
            yield
        finally:
            self._local.use_base = prev

    @staticmethod
    def _encode_image(img) -> str:
        """PIL image -> data URI, which is what the OpenAI image_url part expects."""
        import base64
        import io
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _payload(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            parts = []
            for p in content:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image":
                    parts.append({"type": "image_url",
                                  "image_url": {"url": self._encode_image(p["image"])}})
            out.append({"role": m["role"], "content": parts})
        return out

    def chat(self, messages: list[dict], *, max_new_tokens: int | None = None,
             temperature: float = 0.1) -> str:
        import urllib.request

        model = self.base_model if getattr(self._local, "use_base", False) else self.model
        body = json.dumps({
            "model": model,
            "messages": self._payload(messages),
            "max_tokens": int(max_new_tokens or self.max_new_tokens),
            "temperature": float(temperature),
            "top_p": 0.9,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def models(self) -> list[str]:
        """Served model names, used to fail early on a typo or a server that is not up yet."""
        import urllib.request
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        req = urllib.request.Request(f"{self.base_url}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return [m["id"] for m in json.loads(resp.read()).get("data", [])]


def text_msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def image_msg(role: str, text: str, images: list) -> dict:
    content: list[dict] = [{"type": "text", "text": text}]
    content += [{"type": "image", "image": im} for im in images]
    return {"role": role, "content": content}

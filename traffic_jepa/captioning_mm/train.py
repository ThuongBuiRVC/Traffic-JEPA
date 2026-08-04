#!/usr/bin/env python
"""Multimodal LoRA for the grounded caption variant: facts + simulation frames -> caption.

The training frames are simulation and the test frames are real, so every example carries the same
facts-primary instruction used at inference (`QA_IMG_NOTE`): the VQA values are the ground truth,
the frames are secondary. The model therefore learns to write from the facts and only glance at the
image — the behaviour that has to carry over to real frames.

LoRA sits on the language-model projections only (q/k/v/o/gate/up/down). Qwen's vision blocks use a
fused `qkv`, which those names do not match, so the visual encoder is never fine-tuned.

Rows may be multimodal or text-only, and mixed batches are supported. `--no_images` trains text-only,
which reduces this to the default caption stage's recipe.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from traffic_jepa.captioning_mm.generate import QA_IMG_NOTE

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEF_DATA = "data/processed/caption_mm/manifests"
IMAGE_PIXELS = 512 * 512                                  # default per-frame pixel budget
# per-token fields the processor returns; all must be truncated and padded together
SEQ_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL, help="HF id or local path of the base model")
    ap.add_argument("--data", default=DEF_DATA, help="dir with train.jsonl / val.jsonl from preprocess")
    ap.add_argument("--out", default="runs/caption_lora_mm")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch size (images -> keep small)")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=4096, help="token cap, image tokens included")
    ap.add_argument("--max_images", type=int, default=4,
                    help="frames per example; 4 = 3 overhead + 1 dashcam, matching preprocess and inference")
    ap.add_argument("--max_pixels", type=int, default=int(os.getenv("IMG_MAX_PIXELS", IMAGE_PIXELS)),
                    help="pixel budget per frame into the vision tower (caps vision tokens / VRAM)")
    ap.add_argument("--no_images", action="store_true", help="ignore frames -> text-only fine-tune")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_steps", type=int, default=50, help="only used with --save_strategy steps")
    ap.add_argument("--save_strategy", choices=["epoch", "steps"], default="epoch",
                    help="checkpoint/eval cadence; 'epoch' also keeps the best eval_loss adapter")
    ap.add_argument("--max_eval", type=int, default=0, help="subsample val to N examples; 0 = all")
    ap.add_argument("--fp16", action="store_true", help="fp16 instead of the default bf16")
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA: load the base in 4-bit")
    ap.add_argument("--merge_val", action="store_true", help="train on train+val, no held-out eval")
    return ap.parse_args()


def get_processor(model_id, max_pixels):
    """(processor, tokenizer). processor is None when this build has no Qwen3-VL processor."""
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, max_pixels=max_pixels)
        tok = getattr(proc, "tokenizer", None) or AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] no AutoProcessor for {model_id} ({e}); falling back to text-only", flush=True)
        proc = None
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.chat_template is None:
            raise SystemExit(f"FATAL: no chat template on the tokenizer of {model_id}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return proc, tok


class CaptionDataset(Dataset):
    """Encodes each row on the fly. Only the assistant caption tokens are supervised."""

    def __init__(self, rows, processor, tokenizer, max_len, max_images, use_images):
        self.rows = rows
        self.proc = processor
        self.tok = tokenizer
        self.max_len = max_len
        self.max_images = max_images
        self.use_images = use_images and processor is not None

    def __len__(self):
        return len(self.rows)

    def _load_images(self, paths):
        from PIL import Image
        out = []
        for p in paths[: self.max_images]:
            try:
                img = Image.open(p)
                img.load()
                out.append(img.convert("RGB"))
            except Exception:
                continue
        return out

    def __getitem__(self, i):
        row = self.rows[i]
        images = self._load_images(row.get("images") or []) if self.use_images else []
        sys_msg, user_msg, asst_msg = row["messages"]
        user_text = user_msg["content"]
        if images:                                        # identical to what inference sends
            user_text = QA_IMG_NOTE + "\n\n" + user_text

        content = [{"type": "text", "text": user_text}] + [{"type": "image"} for _ in images]
        msgs = [{"role": "system", "content": [{"type": "text", "text": sys_msg["content"]}]},
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": asst_msg["content"]}]}]

        prompt_text = self.tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full_text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

        if images and self.proc is not None:
            full = self.proc(text=[full_text], images=[images], return_tensors="pt")
            prompt = self.proc(text=[prompt_text], images=[images], return_tensors="pt")
            # keep every per-token field: Qwen3-VL needs mm_token_type_ids for multimodal RoPE
            item = {k: v[0] for k, v in full.items() if k in SEQ_KEYS}
            item["pixel_values"] = full["pixel_values"]
            item["image_grid_thw"] = full["image_grid_thw"]
            n_prompt = prompt["input_ids"].shape[1]
        else:
            full_ids = self.tok(full_text, add_special_tokens=False, return_tensors="pt")
            prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
            item = {"input_ids": full_ids["input_ids"][0], "attention_mask": full_ids["attention_mask"][0]}
            n_prompt = len(prompt_ids)

        labels = item["input_ids"].clone()
        labels[:min(n_prompt, labels.shape[0])] = -100     # loss only on the caption
        item["labels"] = labels
        if item["input_ids"].shape[0] > self.max_len:
            for k in SEQ_KEYS + ("labels",):
                if k in item:
                    item[k] = item[k][: self.max_len]
        return item


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats):
        maxlen = max(f["input_ids"].shape[0] for f in feats)
        ids, labels, masks = [], [], []
        for f in feats:
            n = maxlen - f["input_ids"].shape[0]
            ids.append(torch.cat([f["input_ids"], torch.full((n,), self.pad_id, dtype=f["input_ids"].dtype)]))
            labels.append(torch.cat([f["labels"], torch.full((n,), -100, dtype=f["labels"].dtype)]))
            masks.append(torch.cat([f["attention_mask"], torch.zeros(n, dtype=f["attention_mask"].dtype)]))
        batch = {"input_ids": torch.stack(ids), "labels": torch.stack(labels),
                 "attention_mask": torch.stack(masks)}
        # mm_token_type_ids: 0 = text token, so text-only rows in a mixed batch are all zeros
        if any("mm_token_type_ids" in f for f in feats):
            mm = []
            for f in feats:
                t = f.get("mm_token_type_ids")
                if t is None:
                    t = torch.zeros(f["input_ids"].shape[0], dtype=torch.long)
                mm.append(torch.cat([t, torch.zeros(maxlen - t.shape[0], dtype=t.dtype)]))
            batch["mm_token_type_ids"] = torch.stack(mm)
        # Qwen-VL batches images by concatenating flattened patches + grids across the rows that have them
        pixels = [f["pixel_values"] for f in feats if "pixel_values" in f]
        grids = [f["image_grid_thw"] for f in feats if "image_grid_thw" in f]
        if pixels:
            batch["pixel_values"] = torch.cat(pixels, dim=0)
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        return batch


def load_rows(data_dir, merge_val):
    data_dir = Path(data_dir)

    def read(name):
        p = data_dir / name
        if not p.is_file():
            raise SystemExit(f"FATAL: missing {p}\n"
                             f"       run: python -m traffic_jepa.captioning_mm.preprocess --split train")
        return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

    train = read("train.jsonl")
    val = None
    if merge_val:
        train = train + read("val.jsonl")
    elif (data_dir / "val.jsonl").is_file():
        val = read("val.jsonl")
    return train, val


def load_base(model_id, dtype, quant):
    """Qwen3-VL is an image-text model, so try its own class first. AutoModelForCausalLM can never
    load a VL config, so it is a last resort. Every error is reported so the useless CausalLM
    failure does not mask the real one (e.g. bitsandbytes missing for --load_4bit)."""
    import transformers
    base_kwargs = dict(quantization_config=quant, trust_remote_code=True, attn_implementation="eager")
    if quant is not None:
        base_kwargs["device_map"] = {"": 0}
    errors = []
    for cls_name in ("Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        for dtype_key in ("dtype", "torch_dtype"):        # renamed across transformers versions
            kwargs = dict(base_kwargs)
            kwargs[dtype_key] = dtype
            try:
                return cls.from_pretrained(model_id, **kwargs)
            except TypeError as e:
                errors.append(f"{cls_name}({dtype_key}=): {e}")
                continue
            except Exception as e:
                errors.append(f"{cls_name}: {type(e).__name__}: {e}")
                break
    raise SystemExit(f"FATAL: could not load {model_id}:\n  - " + "\n  - ".join(errors))


def main():
    args = build_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    bf16 = not args.fp16
    use_images = not args.no_images and args.max_images > 0

    processor, tokenizer = get_processor(args.model, args.max_pixels)
    if not use_images:
        processor = None
    train_rows, val_rows = load_rows(args.data, args.merge_val)
    n_img = sum(1 for r in train_rows if r.get("images"))
    print(f"train={len(train_rows)} ({n_img} with frames)  eval={0 if val_rows is None else len(val_rows)}  "
          f"images={'on' if (use_images and processor is not None) else 'OFF (text-only)'}  "
          f"merge_val={args.merge_val}", flush=True)
    if not train_rows:
        raise SystemExit(f"FATAL: no rows in {args.data}/train.jsonl — run preprocess --split train first")
    if use_images and processor is not None and n_img == 0:
        print("[warn] images=on but no row has frames -> this trains TEXT-ONLY. Re-run preprocess "
              "--split train with the right --videos, or pass --no_images on purpose.", flush=True)

    train_ds = CaptionDataset(train_rows, processor, tokenizer, args.max_len, args.max_images, use_images)
    if val_rows and args.max_eval:
        val_rows = val_rows[:args.max_eval]
    eval_ds = (CaptionDataset(val_rows, processor, tokenizer, args.max_len, args.max_images, use_images)
               if val_rows else None)

    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)

    model = load_base(args.model, dtype, quant)
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,                      # keep pixel_values / image_grid_thw
        label_names=["labels"],
        dataloader_num_workers=int(os.getenv("NUM_WORKERS", "4")),
        # A few hundred examples over 3 epochs can overfit, so evaluate and checkpoint every epoch
        # and keep the best one rather than whatever the last step produced.
        eval_strategy=("no" if eval_ds is None else args.save_strategy),
        save_strategy=args.save_strategy,
        eval_steps=(None if eval_ds is None else args.save_steps),
        load_best_model_at_end=(eval_ds is not None),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit" if args.load_4bit else "adamw_torch",
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=Collator(tokenizer.pad_token_id))
    trainer.train()
    if eval_ds is not None:
        print(f"best eval_loss = {getattr(trainer.state, 'best_metric', None)} "
              f"(checkpoint {getattr(trainer.state, 'best_model_checkpoint', None)}) — that is what is saved",
              flush=True)

    adapter = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    if processor is not None:
        processor.save_pretrained(adapter)
    print(f"saved LoRA adapter -> {adapter}\n"
          f"use it with: bash scripts/05_caption.sh mm --lora {adapter}")


if __name__ == "__main__":
    main()

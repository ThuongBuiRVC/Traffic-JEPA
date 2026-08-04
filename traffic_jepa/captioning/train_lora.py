#!/usr/bin/env python
"""LoRA fine-tune of Qwen3-VL-8B for the caption QA path (text-only: facts -> caption).

Only the assistant caption tokens contribute to the loss (the prompt is masked). No images are
used, so the vision tower is untouched and the model trains as a text LM. The shipped adapter in
`checkpoints/caption_lora/` was produced by this script with the defaults below.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, Trainer, TrainingArguments

BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL, help="HF id or local path of the base model")
    ap.add_argument("--data", default="data/processed/caption_lora", help="dir with train.jsonl / val.jsonl")
    ap.add_argument("--out", default="runs/caption_lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--fp16", action="store_true", help="fp16 instead of the default bf16")
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA: load the base in 4-bit (needs bitsandbytes)")
    ap.add_argument("--merge_val", action="store_true", help="train on train+val, no held-out eval")
    return ap.parse_args()


def get_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.chat_template is None:                        # Qwen3-VL keeps the template on the processor
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tok.chat_template = getattr(proc, "chat_template", None) or proc.tokenizer.chat_template
    if tok.chat_template is None:
        raise SystemExit(f"FATAL: no chat template on the tokenizer or processor of {model_id}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_dataset(tokenizer, data_dir, max_len, merge_val):
    data_dir = Path(data_dir)
    train_files = [str(data_dir / "train.jsonl")]
    if merge_val:
        train_files.append(str(data_dir / "val.jsonl"))
    for f in train_files:
        if not Path(f).is_file():
            raise SystemExit(f"FATAL: missing data file: {f}  (run traffic_jepa.captioning.build_data)")
    ds = load_dataset("json", data_files={"train": train_files}, split="train")
    eval_ds = None
    val_path = data_dir / "val.jsonl"
    if not merge_val and val_path.is_file():
        eval_ds = load_dataset("json", data_files={"eval": str(val_path)}, split="eval")

    def encode(ex):
        msgs = ex["messages"]
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full_ids = tokenizer(full, add_special_tokens=False).input_ids
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100                             # loss only on the caption
        return {"input_ids": full_ids[:max_len], "labels": labels[:max_len],
                "attention_mask": [1] * len(full_ids[:max_len])}

    ds = ds.map(encode, remove_columns=ds.column_names, desc="tokenize-train")
    ds = ds.filter(lambda x: any(l != -100 for l in x["labels"]))
    if eval_ds is not None:
        eval_ds = eval_ds.map(encode, remove_columns=eval_ds.column_names, desc="tokenize-eval")
    return ds, eval_ds


@dataclass
class Collator:
    pad_id: int

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, labels, masks = [], [], []
        for f in feats:
            pad = maxlen - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            masks.append(f["attention_mask"] + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(masks)}


def load_base(model_id, dtype, quant):
    import transformers
    kwargs = dict(dtype=dtype, quantization_config=quant, trust_remote_code=True,
                  attn_implementation="eager")
    if quant is not None:
        kwargs["device_map"] = {"": 0}
    last = None
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_id, **kwargs)
        except Exception as e:
            last = e
    raise SystemExit(f"FATAL: could not load {model_id}: {last}")


def main():
    args = build_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    bf16 = not args.fp16

    tokenizer = get_tokenizer(args.model)
    train_ds, eval_ds = build_dataset(tokenizer, args.data, args.max_len, args.merge_val)
    print(f"train={len(train_ds)}  eval={0 if eval_ds is None else len(eval_ds)}  merge_val={args.merge_val}")

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
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy=("no" if eval_ds is None else "steps"),
        eval_steps=(None if eval_ds is None else args.save_steps),
        optim="paged_adamw_8bit" if args.load_4bit else "adamw_torch",
    )
    Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=Collator(tokenizer.pad_token_id)).train()

    adapter = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    print(f"saved LoRA adapter -> {adapter}\n"
          f"use it with: --mode lora --lora {adapter}")


if __name__ == "__main__":
    main()

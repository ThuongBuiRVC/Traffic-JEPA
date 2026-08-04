#!/usr/bin/env python
"""Train Traffic-JEPA on cached WTS QA and report per-category accuracy.

Everything comes from the simulation split (index_sim), and no real data is read at any point.
Training runs a fixed number of epochs with no early stopping. The evaluation each epoch is for
monitoring only, and `model_latest.pt` after the last epoch is the model that gets used.
"""
from __future__ import annotations
import argparse, collections, json, os, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table
from torch.utils.data import DataLoader

from traffic_jepa.data.wts_cached import WTSCollator, WTSCachedDataset
from traffic_jepa.modeling.traffic_jepa_model import TrafficJEPAModel

console = Console()
PLAIN = not sys.stdout.isatty()      # non-TTY: disable rich live bars (they bloat captured logs)
_VLJEPA_ROOT = Path(os.getenv("VLJEPA_ROOT", str(Path(__file__).resolve().parents[2])))
CACHE = os.environ.get("VLJEPA_CACHE", str(_VLJEPA_ROOT / "data" / "cache_vljepa16_8f"))

# Synonym-aware option matching: 100% of distance questions offer BOTH "near" and "close"
# (semantic duplicates -> EmbeddingGemma can't separate). acc_syn reports the de-trapped ceiling.
_SYN = {"near": "near_close", "close": "near_close"}
WEAK_CATEGORIES = ("pedestrian_behavior", "pedestrian_orientation", "pedestrian_position")


def _norm_opt(s):
    s = (s or "").strip().lower()
    return _SYN.get(s, s)


def norm_question(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def seed_everything(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(ds, bs, collate, shuffle, workers, drop_last=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                      num_workers=workers, pin_memory=True, drop_last=drop_last)


def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path:
        return
    ckpt = Path(ckpt_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"init checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    missing = [k for k in model_state if k not in filtered]
    skipped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f"[init] loaded {len(filtered)} tensors from {ckpt}", flush=True)
    if missing:
        print(f"[init] missing/new tensors: {len(missing)} e.g. {missing[:8]}", flush=True)
    if skipped:
        print(f"[init] skipped mismatch/unexpected: {len(skipped)} e.g. {skipped[:8]}", flush=True)


def make_scheduler(opt, scheduler_name, warmup_steps, total_steps, min_lr_scale):
    name = (scheduler_name or "none").lower()
    if name in {"", "none", "constant"}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr_scale = float(min_lr_scale)

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def weak_mean(rep):
    per = rep.get("per_category") or {}
    vals = [float(per[cat]["acc"]) for cat in WEAK_CATEGORIES if cat in per]
    return sum(vals) / len(vals) if vals else 0.0


@torch.no_grad()
def evaluate(model, dl, device, amp, desc):
    """Accuracy + InfoNCE loss over the labeled rows, predictions for all of them."""
    model.eval()
    by = collections.defaultdict(lambda: [0, 0])     # selector -> [correct, total]
    by_q = collections.defaultdict(lambda: [0, 0])   # category::question -> [correct, total]
    by_syn = collections.defaultdict(lambda: [0, 0]) # selector -> [correct, total] with Near=Close merged
    tot = [0, 0]; tot_syn = [0, 0]; preds = []; loss_sum, loss_n = 0.0, 0
    if PLAIN:
        print(f"[eval] {desc} over {len(dl)} batches...", flush=True)
    with Progress(SpinnerColumn(), TextColumn(f"[cyan]{desc}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console,
                  transient=True, disable=PLAIN) as p:
        t = p.add_task(desc, total=len(dl))
        for batch in dl:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                y_hat, _, _ = model.forward(batch)
                pred = model.final_pred(y_hat, None, batch)
            pred = pred.cpu().tolist()
            lbl = []
            for k, pi in enumerate(pred):
                opts = batch["options_text"][k]
                preds.append({"qa_id": batch["qa_id"][k], "selector": batch["selector_name"][k],
                              "pred_index": pi, "pred_text": opts[pi] if pi < len(opts) else ""})
                if batch["has_label"][k]:
                    lbl.append(k); sel = batch["selector_name"][k]
                    qkey = f"{sel}::{norm_question(batch.get('question', [''])[k])}"
                    gi = int(batch["correct_index"][k])
                    ok = int(pi == gi)
                    by[sel][0] += ok; by[sel][1] += 1; tot[0] += ok; tot[1] += 1
                    by_q[qkey][0] += ok; by_q[qkey][1] += 1
                    ok_syn = int(_norm_opt(opts[pi] if pi < len(opts) else "")
                                 == _norm_opt(opts[gi] if gi < len(opts) else ""))
                    by_syn[sel][0] += ok_syn; by_syn[sel][1] += 1
                    tot_syn[0] += ok_syn; tot_syn[1] += 1
            if len(lbl) >= 2:     # InfoNCE loss over labeled subset of this batch
                sub = {
                    "target_vec": batch["target_vec"][lbl],
                    "distractor_vecs": batch["distractor_vecs"][lbl],
                    "distractor_valid": batch["distractor_valid"][lbl],
                    "opt_vecs": batch["opt_vecs"][lbl],
                    "n_opt": batch["n_opt"][lbl],
                    "correct_index": batch["correct_index"][lbl],
                    "phase_id": batch["phase_id"][lbl],
                    "target_hash": [batch["target_hash"][k] for k in lbl],
                    "selector_name": [batch["selector_name"][k] for k in lbl],
                    "options_text": [batch["options_text"][k] for k in lbl],
                }
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss_sum += float(model.hybrid_loss(y_hat[lbl], sub)); loss_n += 1
            p.advance(t)
    acc = round(tot[0] / tot[1], 4) if tot[1] else None
    acc_syn = round(tot_syn[0] / tot_syn[1], 4) if tot_syn[1] else None
    per = {s: {"correct": c, "wrong": n - c, "total": n, "acc": round(c / max(n, 1), 4),
               "acc_syn": round(by_syn[s][0] / max(by_syn[s][1], 1), 4)}
           for s, (c, n) in sorted(by.items())}
    per_q = {s: {"correct": c, "wrong": n - c, "total": n, "acc": round(c / max(n, 1), 4)}
             for s, (c, n) in sorted(by_q.items())}
    return {"overall_acc": acc, "overall_acc_syn": acc_syn,
            "loss": round(loss_sum / loss_n, 4) if loss_n else None,
            "correct": tot[0], "wrong": tot[1] - tot[0], "labeled": tot[1], "n": len(preds),
            "per_category": per, "per_question": per_q, "predictions": preds}


def print_report(tag, rep):
    if rep["overall_acc"] is None:
        print(f"{tag}: labels pending -> {rep['n']} predictions saved (no accuracy)", flush=True)
        return
    if PLAIN:     # compact plain text (one line/category) so it survives captured logs
        print(f"{tag}  overall acc={rep['overall_acc']} (syn={rep.get('overall_acc_syn')})  "
              f"({rep['correct']}/{rep['labeled']})", flush=True)
        for s, d in rep["per_category"].items():
            print(f"    {s:22s} acc={d['acc']:.4f} syn={d.get('acc_syn', 0):.4f}  "
                  f"({d['correct']}/{d['total']})", flush=True)
        return
    table = Table(title=f"{tag}  overall acc={rep['overall_acc']}  "
                        f"(syn={rep.get('overall_acc_syn')})  ({rep['correct']}/{rep['labeled']})")
    table.add_column("category"); table.add_column("acc", justify="right")
    table.add_column("acc_syn", justify="right")
    table.add_column("correct", justify="right"); table.add_column("wrong", justify="right")
    for s, d in rep["per_category"].items():
        table.add_row(s, f"{d['acc']:.3f}", f"{d.get('acc_syn', 0):.3f}",
                      str(d["correct"]), str(d["wrong"]))
    console.print(table)


def build_args():
    ap = argparse.ArgumentParser()
    # optimization
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1, help="benchmark every n epochs")
    ap.add_argument("--batch_size", type=int, default=2, help=">=2 (InfoNCE needs in-batch negatives)")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4, help="LR for the adapter/head params")
    ap.add_argument("--backbone_lr", type=float, default=2e-5, help="LR for the unfrozen Llama layers")
    ap.add_argument("--backbone_wd", type=float, default=0.05)
    ap.add_argument("--scheduler", default="cosine", choices=["none", "constant", "cosine"], help="LR schedule")
    ap.add_argument("--warmup_steps", type=int, default=500, help="LR warmup optimizer steps (cosine)")
    ap.add_argument("--min_lr_scale", type=float, default=0.05, help="minimum LR as a fraction of initial LR")
    # model
    ap.add_argument("--predictor", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--predictor_layers", type=int, default=6)
    ap.add_argument("--unfreeze", type=int, default=0, help="number of Llama backbone layers to fine-tune")
    ap.add_argument("--spatial_pool", dest="spatial_pool_size", type=int, default=1,
                    help="adaptive V-JEPA grid per frame; 1 = temporal mean tokens")
    ap.add_argument("--no_prior_tokens", action="store_true", help="disable the phase/selector prior embeddings")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--token_dropout", type=float, default=0.15)
    ap.add_argument("--ablation_no_visual", action="store_true",
                    help="ablation: zero the visual tokens to measure the text-only baseline")
    # data / augmentation
    ap.add_argument("--cache", default=CACHE, help="token-cache dir (default $VLJEPA_CACHE)")
    ap.add_argument("--mirror_p", type=float, default=0.0, help="probability of left/right mirror augmentation (train only)")
    ap.add_argument("--weak_weight", type=float, default=1.0, help="loss weight for the weak question categories")
    ap.add_argument("--meta_dropout", type=float, default=0.0,
                    help="probability of corrupting phase/selector metadata during training (anti-shortcut)")
    ap.add_argument("--train_index_suffix", default="", help="suffix for index_sim*.jsonl used for training")
    ap.add_argument("--eval_index_suffix", default="", help="suffix for the index_sim*.jsonl used to validate")
    ap.add_argument("--init_checkpoint", default="", help="load an existing model checkpoint before training")
    # runtime
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--seed", type=int, default=-1, help="RNG seed; negative keeps non-deterministic behavior")
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic algorithms + cudnn.deterministic "
                         "(needs env CUBLAS_WORKSPACE_CONFIG=:4096:8); same seed -> same weights")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing (big GPU)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (probe)")
    ap.add_argument("--max_hours", type=float, default=0.0, help="stop before this wall-clock budget (safety)")
    ap.add_argument("--out", default="runs/traffic_jepa_world_model")
    return ap.parse_args()


def build_model(args, device):
    print(f"[init] loading predictor {args.predictor} (Llama {args.predictor_layers} layers)...", flush=True)
    t_load = time.time()
    model = TrafficJEPAModel(
        predictor_model=args.predictor,
        num_layers=args.predictor_layers,
        temperature=args.temperature,
        grad_ckpt=not args.no_grad_ckpt,
        dropout=args.dropout,
        token_dropout=args.token_dropout,
        unfreeze_backbone_layers=args.unfreeze,
        spatial_pool_size=args.spatial_pool_size,
        use_prior_tokens=not args.no_prior_tokens,
        ablation_no_visual=args.ablation_no_visual,
    ).to(device)
    load_checkpoint(model, args.init_checkpoint, device)
    print(f"[init] model ready in {time.time()-t_load:.0f}s", flush=True)
    return model


def build_datasets(args, tokenizer):
    cache = args.cache
    train_ds = WTSCachedDataset(cache, "sim", tokenizer, index_suffix=args.train_index_suffix,
                                mirror_p=args.mirror_p, weak_weight=args.weak_weight,
                                meta_dropout=args.meta_dropout)                # train: ALL sim (+aug)
    eval_ds = WTSCachedDataset(cache, "sim", tokenizer, index_suffix=args.eval_index_suffix)
    if args.limit:
        train_ds.rows = train_ds.rows[: args.limit]
        eval_ds.rows = eval_ds.rows[: args.limit]
    return train_ds, eval_ds


def build_optimizer(model, args):
    backbone, heads = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = n.startswith("predictor.backbone.layers.") or n.startswith("predictor.backbone.norm.")
        (backbone if is_backbone else heads).append(p)
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": args.backbone_wd},
        {"params": heads, "lr": args.lr, "weight_decay": 0.01},
    ])
    return opt, backbone, heads


def save_checkpoint(model, out, report):
    """Overwrite the checkpoint every epoch. Training runs a fixed number of epochs and the last
    one is what gets used, so there is no best-epoch selection to make."""
    # keep the unfrozen layers + all heads/buffers (not the frozen Llama backbone)
    state = {n: p.detach().cpu() for n, p in model.named_parameters()
             if p.requires_grad or not n.startswith("predictor.backbone.")}
    state.update({n: b.detach().cpu() for n, b in model.named_buffers()})
    torch.save(state, out / "model_latest.pt")
    json.dump(report, open(out / "report_latest.json", "w"), indent=2)


def main():
    args = build_args()
    seed_everything(args.seed)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (not args.fp32) and device == "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    console.rule("[bold]Traffic-JEPA training")

    model = build_model(args, device)
    collate = WTSCollator(model.tokenizer.pad_token_id or 0)
    train_ds, eval_ds = build_datasets(args, model.tokenizer)
    console.print(f"train={len(train_ds)}  eval={len(eval_ds)}  amp={amp}")

    train_dl = loader(train_ds, args.batch_size, collate, True, args.workers, drop_last=True)
    eval_dl = loader(eval_ds, args.batch_size, collate, False, args.workers)

    opt, backbone, heads = build_optimizer(model, args)
    steps_per_epoch = max(1, len(train_dl) // max(args.grad_accum, 1))
    total_steps = args.max_steps or (steps_per_epoch * args.epochs)
    sched = make_scheduler(opt, args.scheduler, args.warmup_steps, total_steps, args.min_lr_scale)
    console.print(f"trainable: {sum(p.numel() for p in backbone + heads)/1e6:.1f}M  "
                  f"({args.predictor_layers}-layer Llama fine-tune={sum(p.numel() for p in backbone)/1e6:.0f}M"
                  f" + heads={sum(p.numel() for p in heads)/1e6:.1f}M)")
    json.dump(vars(args), open(out / "run_args.json", "w"), indent=2, sort_keys=True)

    step = 0
    eval_rep = None                                      # may stay None in a --max_steps probe
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours:
            console.print(f"[magenta]time budget {args.max_hours}h reached at epoch {epoch} -> stop"); break
        model.train()
        run = [0, 0]; tl = [0.0, 0]; t0 = time.time()
        with Progress(SpinnerColumn(), TextColumn(f"[bold blue]epoch {epoch}"), BarColumn(),
                      MofNCompleteColumn(), TextColumn("loss {task.fields[loss]}"),
                      TextColumn("acc {task.fields[acc]}"), TimeElapsedColumn(),
                      TimeRemainingColumn(), console=console, disable=PLAIN) as p:
            task = p.add_task("train", total=len(train_dl), loss="-", acc="-")
            opt.zero_grad()
            for it, batch in enumerate(train_dl):
                batch = to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    y_hat, _, _ = model.forward(batch)
                    pred = model.final_pred(y_hat, None, batch)
                    loss = model.hybrid_loss(y_hat, batch)
                (loss / args.grad_accum).backward()
                if (it + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(backbone + heads, args.grad_clip)
                    opt.step()
                    if sched is not None:
                        sched.step()
                    opt.zero_grad(); step += 1
                run[0] += (pred == batch["correct_index"]).sum().item(); run[1] += len(pred)
                tl[0] += loss.item(); tl[1] += 1
                p.update(task, advance=1, loss=f"{loss.item():.3f}", acc=f"{run[0]/max(run[1],1):.3f}")
                if args.max_steps and step >= args.max_steps:
                    break
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            tot = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"[magenta]peak VRAM={peak:.1f}/{tot:.0f} GiB  bs={args.batch_size} "
                          f"iters={tl[1]} in {time.time()-t0:.0f}s "
                          f"({(time.time()-t0)/max(tl[1],1):.3f}s/iter)")
        if args.max_steps and step >= args.max_steps:
            console.print(f"[magenta]probe stop at {step} steps"); break
        train_loss = round(tl[0] / max(tl[1], 1), 4)
        console.print(f"epoch {epoch} train acc={run[0]/max(run[1],1):.4f} loss={train_loss} "
                      f"({time.time()-t0:.0f}s)")
        if (epoch % args.eval_every == 0) or (epoch == args.epochs):
            eval_rep = evaluate(model, eval_dl, device, amp, "eval")
            print_report(f"[ep{epoch}] EVAL", eval_rep)
            console.print(f"[ep{epoch}] loss  train={train_loss}  eval={eval_rep['loss']}")
            report = {"epoch": epoch, "train_loss": train_loss,
                      "eval": {k: v for k, v in eval_rep.items() if k != "predictions"},
                      "weak_mean": round(weak_mean(eval_rep), 4)}
            save_checkpoint(model, out, report)
            console.print(f"[green]epoch {epoch} saved -> model_latest.pt "
                          f"(acc={eval_rep['overall_acc']} weak={weak_mean(eval_rep):.4f})")

    if eval_rep is None:     # --max_steps probe ended before any eval
        console.rule("[bold green]probe done (no eval)")
        return
    with open(out / "validation_predictions.jsonl", "w") as fh:
        for r in eval_rep["predictions"]:
            fh.write(json.dumps(r) + "\n")
    eval_rep.pop("predictions", None)
    json.dump({"eval": eval_rep, "epochs_run": args.epochs}, open(out / "final_report.json", "w"), indent=2)
    console.rule("[bold green]done")
    console.print(f"final  acc={eval_rep['overall_acc']} loss={eval_rep['loss']} ({eval_rep['labeled']} labeled)")


if __name__ == "__main__":
    main()

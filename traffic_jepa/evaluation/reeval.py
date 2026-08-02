#!/usr/bin/env python
"""Re-evaluate Traffic-JEPA checkpoints on a cached split."""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from traffic_jepa.data.wts_cached import WTSCachedDataset, WTSCollator
from traffic_jepa.modeling.traffic_jepa_model import TrafficJEPAModel

CACHE = "data/processed/cache_vljepa16_8f"
PREDICTOR = "meta-llama/Llama-3.2-1B"


def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_loader(ds, collate):
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate)


def build_model(cfg: dict, device: str, force_no_visual: bool, force_no_prior: bool) -> TrafficJEPAModel:
    variant = cfg.get("model_variant", "traffic_jepa")
    if variant != "traffic_jepa":
        raise ValueError(f"Unsupported model_variant={variant!r}; expected 'traffic_jepa'")
    return TrafficJEPAModel(
        predictor_model=cfg.get("predictor", PREDICTOR),
        num_layers=cfg.get("predictor_layers", 6),
        temperature=cfg.get("temperature", 0.07),
        dropout=cfg.get("dropout", 0.1),
        token_dropout=cfg.get("token_dropout", 0.0),
        unfreeze_backbone_layers=cfg.get("unfreeze", 0),
        use_prior_tokens=(not cfg.get("no_prior_tokens", False)) and (not force_no_prior),
        spatial_pool_size=cfg.get("spatial_pool_size", 1),
        ablation_no_visual=cfg.get("ablation_no_visual", False) or force_no_visual,
        grad_ckpt=False,
    ).to(device)


def evaluate(model, dl, device: str, save_scores: bool = False):
    model.eval()
    by_cat = collections.defaultdict(lambda: [0, 0])
    total = [0, 0]
    preds = []
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                y_hat, dist_logits = model.forward(batch)[:2]
                scores = model.option_scores(y_hat, dist_logits, batch)
                pred = torch.argmax(scores, dim=-1)
            pred = pred.cpu().tolist()
            scores_cpu = scores.float().cpu().tolist()
            for k, pred_idx in enumerate(pred):
                opts = batch["options_text"][k]
                rec = {
                    "qa_id": batch["qa_id"][k],
                    "selector": batch["selector_name"][k],
                    "pred_index": pred_idx,
                    "pred_text": opts[pred_idx] if pred_idx < len(opts) else "",
                }
                if save_scores:
                    n_opt = int(batch["n_opt"][k].detach().cpu())
                    rec["scores"] = [float(x) for x in scores_cpu[k][:n_opt]]
                preds.append(rec)
                if batch["has_label"][k]:
                    selector = batch["selector_name"][k]
                    gold_idx = int(batch["correct_index"][k])
                    ok = int(pred_idx == gold_idx)
                    by_cat[selector][0] += ok
                    by_cat[selector][1] += 1
                    total[0] += ok
                    total[1] += 1
    acc = total[0] / total[1] if total[1] else 0.0
    per_cat = {
        selector: {"correct": c, "total": n, "acc": round(c / max(n, 1), 4)}
        for selector, (c, n) in sorted(by_cat.items())
    }
    return acc, total[0], total[1], per_cat, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="run directories containing model_best.pt")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--domain", choices=["real", "sim"], default="real")
    ap.add_argument("--index_suffix", default="", help="suffix for index_{domain}*.jsonl")
    ap.add_argument("--real_index_suffix", default="", help="legacy alias for --index_suffix on real")
    ap.add_argument("--out_name", default="", help="prediction filename; default {domain}{suffix}_predictions.jsonl")
    ap.add_argument("--save_scores", action="store_true", help="include per-option model scores")
    ap.add_argument("--force_no_visual", action="store_true", help="zero V-JEPA visual tokens")
    ap.add_argument("--force_no_prior", action="store_true", help="disable phase/category prior tokens")
    args = ap.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs if (Path(d) / "model_best.pt").is_file()]
    if not run_dirs:
        print("No valid run dirs found!")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_suffix = args.index_suffix or args.real_index_suffix

    for run_dir in run_dirs:
        ckpt_path = run_dir / "model_best.pt"
        args_path = run_dir / "run_args.json"
        cfg = json.load(open(args_path)) if args_path.is_file() else {}

        print("=" * 60)
        print(f"Re-evaluating: {run_dir}")

        model = build_model(cfg, device, args.force_no_visual, args.force_no_prior)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

        ds = WTSCachedDataset(args.cache, args.domain, model.tokenizer, index_suffix=index_suffix)
        dl = make_loader(ds, WTSCollator(model.tokenizer.pad_token_id or 0))

        t0 = time.time()
        acc, correct, total, per_cat, preds = evaluate(model, dl, device, save_scores=args.save_scores)
        print(f"  {args.domain} accuracy: {correct}/{total} = {acc*100:.2f}% ({time.time()-t0:.1f}s)")
        for cat in sorted(per_cat):
            c = per_cat[cat]
            print(f"    {cat:<35} {c['acc']*100:>7.2f}%  ({c['correct']}/{c['total']})")

        out_name = args.out_name or f"{args.domain}{index_suffix}_predictions.jsonl"
        out_path = run_dir / out_name
        with open(out_path, "w") as fh:
            for row in preds:
                fh.write(json.dumps(row) + "\n")
        print(f"  Saved -> {out_path}\n")


if __name__ == "__main__":
    main()

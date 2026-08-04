#!/usr/bin/env python
"""Build the LoRA fine-tuning data for the caption QA path, from the SynWTS annotations.

  input  (X): system = SYS, user = build_prompt(scenario, phase, facts, shots=[])  — same as inference
  output (Y): assistant = {"caption_pedestrian": <sim GT>, "caption_vehicle": <sim GT>}

Facts come from the SynWTS VQA ground truth, captions from the SynWTS caption ground truth. Overhead
and vehicle views carry identical captions, so they are deduped to one example per (scenario, phase).
Text-only — the frame path is never fine-tuned. Training uses simulation only, never test data.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from traffic_jepa.captioning.generate import SYS, build_prompt

PHASE = {"prerecognition": "0", "recognition": "1", "judgement": "2", "judgment": "2",
         "action": "3", "avoidance": "4"}


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def build_split(annot_root: str, split: str) -> list[dict]:
    facts: dict = defaultdict(dict)
    const: dict = defaultdict(dict)
    for vf in glob.glob(os.path.join(annot_root, "vqa", split, "*", "*", "*.json")):
        try:
            data = json.load(open(vf))
        except Exception:
            continue
        scenario = os.path.basename(vf).replace(".json", "")
        for item in (data if isinstance(data, list) else [data]):
            for c in item.get("environment", []) or []:
                correct = c.get("correct")
                if correct and c.get(correct):
                    const[scenario][norm(c["question"])] = norm(c[correct])
            for p in item.get("event_phase", []):
                label = str((p.get("labels") or ["?"])[0]).lower()
                phase = PHASE.get(label, label)
                for c in p.get("conversations", []):
                    correct = c.get("correct")
                    if correct and c.get(correct):
                        facts[(scenario, phase)][norm(c["question"])] = norm(c[correct])
    for (scenario, _phase), d in facts.items():
        for q, a in const.get(scenario, {}).items():
            d.setdefault(q, a)

    caps: dict = defaultdict(dict)                       # (scenario, phase) -> {view: (ped, veh)}
    for cf in glob.glob(os.path.join(annot_root, "caption", split, "*", "*", "*caption*.json")):
        scenario = os.path.basename(cf).replace("_caption.json", "")
        view = "overhead" if "overhead" in cf else "vehicle"
        try:
            data = json.load(open(cf))
        except Exception:
            continue
        for e in data.get("event_phase", []):
            phase = str((e.get("labels") or ["?"])[0])
            ped, veh = norm(e.get("caption_pedestrian", "")), norm(e.get("caption_vehicle", ""))
            if ped and veh:
                caps[(scenario, phase)][view] = (ped, veh)

    rows = []
    for (scenario, phase), views in caps.items():
        qa = facts.get((scenario, phase))
        if not qa:
            continue
        seen = set()
        for cap in views.values():
            if cap in seen:
                continue
            seen.add(cap)
            ped, veh = cap
            rows.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": build_prompt(scenario, phase, qa, shots=[])},
                {"role": "assistant", "content": json.dumps(
                    {"caption_pedestrian": ped, "caption_vehicle": veh}, ensure_ascii=False)},
            ], "meta": {"scenario": scenario, "phase": phase, "n_facts": len(qa)}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", default=os.getenv("SYNWTS_ANNOT", "data/synwts/data/annotations"),
                    help="SynWTS annotations dir (holds vqa/ and caption/)")
    ap.add_argument("--out", default="data/processed/caption_lora", help="dir for train.jsonl / val.jsonl")
    args = ap.parse_args()
    if not Path(args.annot).is_dir():
        raise SystemExit(f"FATAL: SynWTS annotations not found: {args.annot}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = build_split(args.annot, split)
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {path}")


if __name__ == "__main__":
    main()

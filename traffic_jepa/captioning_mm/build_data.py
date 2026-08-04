#!/usr/bin/env python
"""SynWTS facts + reference captions -> chat rows for the grounded LoRA.

Same X/Y as the default caption stage (system = SYS, user = build_prompt(facts), assistant = the
reference captions), plus the simulation frames cut for that (scenario, phase). `preprocess.py`
calls this, it is not run directly.

"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

from traffic_jepa.captioning.generate import SYS, build_prompt

PHASE = {"prerecognition": "0", "recognition": "1", "judgement": "2", "judgment": "2",
         "action": "3", "avoidance": "4"}


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _find(root, split, pattern):
    """Recursive, so a slightly different dataset nesting still resolves."""
    hits = glob.glob(os.path.join(root, split, "**", pattern), recursive=True)
    if not hits:                                          # tolerate a missing <split> level
        hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return hits


def build_split(annot_root: str, split: str, frames_index: dict | None = None) -> list[dict]:
    facts: dict = defaultdict(dict)
    const: dict = defaultdict(dict)
    vqa_files = _find(os.path.join(annot_root, "vqa"), split, "*.json")
    cap_files = _find(os.path.join(annot_root, "caption"), split, "*caption*.json")
    if not vqa_files or not cap_files:
        print(f"[warn] {split}: found {len(vqa_files)} vqa json and {len(cap_files)} caption json under "
              f"{annot_root} — check --annot (it must hold vqa/ and caption/)", flush=True)

    for vf in vqa_files:
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

    caps: dict = defaultdict(dict)                        # (scenario, phase) -> {view: (ped, veh)}
    for cf in cap_files:
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
        for cap in views.values():                        # dedupe identical captions across views
            if cap in seen:
                continue
            seen.add(cap)
            ped, veh = cap
            images = [p for p in (frames_index or {}).get(f"{scenario}||{phase}", []) if os.path.exists(p)]
            rows.append({
                "messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": build_prompt(scenario, phase, qa, shots=[])},
                    {"role": "assistant", "content": json.dumps(
                        {"caption_pedestrian": ped, "caption_vehicle": veh}, ensure_ascii=False)},
                ],
                "images": images,                          # simulation frames, empty -> text-only row
                "meta": {"scenario": scenario, "phase": phase, "n_facts": len(qa)},
            })
    return rows

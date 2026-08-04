#!/usr/bin/env python3
"""Graph/temporal decode of scored public-test WTS predictions.

Learns answer priors, question relation tables, and phase-transition tables from the
labeled sim index, then applies them to the scored public-test JSONL produced by
traffic_jepa.inference.predict. Shares its scoring logic with graph_decode.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from traffic_jepa.postprocess import graph_decode as gd
from traffic_jepa.inference.predict import build_jobs

LETTERS = ["a", "b", "c", "d"]


def load_scored(path: str | Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).open():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["qa_id"]] = row
    return out


def build_test_rows(args, scored: dict[str, dict]) -> list[dict]:
    route = json.load(open(args.route))
    jobs = build_jobs(args.test, route, args.bbox_root, args.videos_root, None)
    rows = []
    missing = []
    for job in jobs:
        pred = scored.get(job["id"])
        if pred is None:
            missing.append(job["id"])
            continue
        rows.append({
            "qa_id": job["id"],
            "sample": job.get("scenario", ""),
            "phase": job.get("phase_name") or "environment",
            "category": pred.get("category") or job.get("selector", ""),
            "question": pred.get("question") or job.get("question", ""),
            "options": pred.get("options") or job.get("options", []),
            "scores": pred.get("scores", []),
            "pred_index": int(pred.get("pred_index", 0)),
            "pred_text": pred.get("pred_text", ""),
            "camera_reason": job.get("camera_reason", ""),
            "view": job.get("view", ""),
            "requested_view": job.get("requested_view", ""),
        })
    if missing:
        raise RuntimeError(f"missing {len(missing)} scored predictions; first={missing[:3]}")
    if len(rows) != len(scored):
        extra = sorted(set(scored) - {r["qa_id"] for r in rows})
        raise RuntimeError(f"scored predictions not matched by test jobs: {len(extra)} extra; first={extra[:3]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test/WTS_VQA_PUBLIC_TEST.json")
    ap.add_argument("--route", default="route.json")
    ap.add_argument("--bbox-root", default="data/test/annotations")
    ap.add_argument("--videos-root", default="data/test/videos/test/public")
    ap.add_argument("--sim_index", default="data/processed/cache_vljepa16_8f/index_sim.jsonl")
    ap.add_argument("--scored", required=True, help="scored JSONL from traffic_jepa.inference.predict")
    ap.add_argument("--out", required=True, help="submission JSON to write")
    ap.add_argument("--decoded_jsonl", default="", help="optional decoded rows with metadata")
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--gamma_qphase", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--tau", type=float, default=0.12)
    ap.add_argument("--relation_cap", type=float, default=2.0,
                    help="cap the relation-table nudge to this many multiples of the "
                         "destination row's own score span, so it can re-rank options "
                         "without swamping the model's own score")
    ap.add_argument("--w_dist", type=float, default=1.3)
    ap.add_argument("--w_pos", type=float, default=0.7)
    ap.add_argument("--w_orientation", type=float, default=0.7)
    ap.add_argument("--w_behavior", type=float, default=0.8)
    ap.add_argument("--w_gaze", type=float, default=1.0)
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--temporal_beta", type=float, default=3.0)
    ap.add_argument("--temporal_stay", type=float, default=0.8)
    ap.add_argument(
        "--temporal_categories",
        default="pedestrian_orientation,pedestrian_position,pedestrian_behavior,pedestrian_gaze,pedestrian_vehicle_distance",
    )
    args = ap.parse_args()

    scored = load_scored(args.scored)
    rows = build_test_rows(args, scored)
    sim_rows = gd.load_rows(args.sim_index)
    priors = gd.build_priors(sim_rows)
    relation_tables = gd.build_relation_tables(sim_rows)
    temporal_tables = gd.build_temporal_tables(sim_rows) if args.temporal else None

    scores = gd.base_scores(rows, scored, args.gamma, args.gamma_qphase, priors)
    pred_indices, pred_texts = gd.decode(
        rows,
        scores,
        relation_tables,
        alpha=args.alpha,
        tau=args.tau,
        weights={
            ("dist_vp", "dist_pv"): args.w_dist,
            ("pos_vp", "pos_pv"): args.w_pos,
            ("travel", "body"): args.w_orientation,
            ("action", "fine"): args.w_behavior,
            ("los", "aware"): args.w_gaze,
        },
        temporal_tables=temporal_tables,
        temporal_beta=args.temporal_beta,
        temporal_stay=args.temporal_stay,
        temporal_categories=args.temporal_categories,
        relation_cap=args.relation_cap,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = []
    decoded_rows = []
    changed = 0
    by_cat = collections.Counter()
    changed_by_cat = collections.Counter()
    for row, idx, text in zip(rows, pred_indices, pred_texts):
        idx = int(max(0, min(idx, len(row.get("options", [])) - 1)))
        letter = LETTERS[idx]
        old = int(row.get("pred_index", 0))
        is_changed = int(idx != old)
        changed += is_changed
        cat = row.get("category", "")
        by_cat[cat] += 1
        changed_by_cat[cat] += is_changed
        submission.append({"id": row["qa_id"], "correct": letter})
        decoded_rows.append({
            **row,
            "graph_pred_index": idx,
            "graph_pred_text": text,
            "graph_correct": letter,
            "changed": bool(is_changed),
        })

    json.dump(submission, out.open("w"), indent=1)
    decoded_path = Path(args.decoded_jsonl) if args.decoded_jsonl else out.with_name(out.stem + "_decoded.jsonl")
    with decoded_path.open("w") as fh:
        for row in decoded_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"TEST_GRAPH_DECODE: rows={len(rows)} changed={changed} -> {out}")
    for cat in sorted(by_cat):
        print(f"  {cat:29s} changed={changed_by_cat[cat]:4d}/{by_cat[cat]}")
    print(f"decoded -> {decoded_path}")


if __name__ == "__main__":
    main()

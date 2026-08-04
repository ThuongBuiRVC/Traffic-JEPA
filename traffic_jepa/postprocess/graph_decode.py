#!/usr/bin/env python3
"""Score-level graph decoder for WTS VQA predictions.

The decoder never sees a label. It learns answer priors, question-to-question relation
tables and phase-transition tables from index_sim, then applies them to model scores.

`traffic_jepa.postprocess.decode_test` is the entry point. This module holds the shared
scoring logic it calls.
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

import numpy as np

RELATIONS = [
    ("dist_vp", "dist_pv"),
    ("pos_vp", "pos_pv"),
    ("travel", "body"),
    ("action", "fine"),
    ("los", "aware"),
]
PHASE_ORDER = ["prerecognition", "recognition", "judgement", "action", "avoidance"]

# Every count table estimated from sim (relation and phase-transition probabilities)
# uses add-alpha (Lidstone) smoothing, so an event unobserved in sim keeps a small
# non-zero probability instead of zero. alpha selected on labeled validation: accuracy
# is flat for alpha <= 1e-3 and degrades for larger alpha.
SMOOTHING_ALPHA = 1e-4


def norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def qkey(row: dict) -> str:
    return f"{row.get('category', '')}::{norm_question(row.get('question', ''))}"


def phase_key(row: dict) -> str:
    return str(row.get("phase") or "environment")


def qphase_key(row: dict) -> str:
    return f"{qkey(row)}::phase={phase_key(row)}"


def qtype(row: dict) -> str:
    q = norm_question(row.get("question", ""))
    c = row.get("category", "")
    if c == "pedestrian_vehicle_distance" and "pedestrian from vehicle" in q:
        return "dist_pv"
    if c == "pedestrian_vehicle_distance" and "vehicle from pedestrian" in q:
        return "dist_vp"
    if c == "pedestrian_position" and "pedestrian relative to the vehicle" in q:
        return "pos_pv"
    if c == "pedestrian_position" and "vehicle relative to the pedestrian" in q:
        return "pos_vp"
    if c == "pedestrian_orientation" and "orientation of the pedestrian" in q:
        return "body"
    if c == "pedestrian_orientation" and "direction of travel" in q:
        return "travel"
    if c == "pedestrian_behavior" and "fine-grained" in q:
        return "fine"
    if c == "pedestrian_behavior" and "action" in q:
        return "action"
    if c == "pedestrian_gaze" and "awareness" in q:
        return "aware"
    if c == "pedestrian_gaze" and "line of sight" in q:
        return "los"
    return ""


def group_key(row: dict) -> tuple[str, str]:
    return (row.get("sample", ""), row.get("phase", ""))


def build_groups(rows: list[dict]) -> dict[tuple[str, str], dict[str, int]]:
    groups = collections.defaultdict(dict)
    for i, row in enumerate(rows):
        typ = qtype(row)
        if typ:
            groups[group_key(row)][typ] = i
    return groups


def label_text(row: dict) -> str:
    opts = row.get("options") or []
    idx = int(row.get("correct_index", 0))
    return opts[idx] if 0 <= idx < len(opts) else ""


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def build_temporal_groups(rows: list[dict]) -> dict[tuple[str, str], dict[str, int]]:
    groups = collections.defaultdict(dict)
    phase_set = set(PHASE_ORDER)
    for i, row in enumerate(rows):
        phase = phase_key(row)
        if phase in phase_set:
            groups[(row.get("sample", ""), qkey(row))][phase] = i
    return groups


def build_relation_tables(sim_rows: list[dict]):
    groups = build_groups(sim_rows)
    tables = {}
    for src, dst in RELATIONS:
        counts = collections.defaultdict(collections.Counter)
        dst_labels = set()
        for group in groups.values():
            if src not in group or dst not in group:
                continue
            src_label = label_text(sim_rows[group[src]])
            dst_label = label_text(sim_rows[group[dst]])
            counts[src_label][dst_label] += 1
            dst_labels.add(dst_label)
        tables[(src, dst)] = (counts, sorted(dst_labels))
    return tables


def build_temporal_tables(sim_rows: list[dict]):
    groups = build_temporal_groups(sim_rows)
    transitions = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    labels_by_question = collections.defaultdict(set)
    for (_, key), group in groups.items():
        for idx in group.values():
            labels_by_question[key].add(label_text(sim_rows[idx]))
        for i in range(len(PHASE_ORDER)):
            for j in range(i + 1, len(PHASE_ORDER)):
                src_phase = PHASE_ORDER[i]
                dst_phase = PHASE_ORDER[j]
                if src_phase not in group or dst_phase not in group:
                    continue
                src_label = label_text(sim_rows[group[src_phase]])
                dst_label = label_text(sim_rows[group[dst_phase]])
                transitions[(key, src_phase, dst_phase)][src_label][dst_label] += 1
    return transitions, labels_by_question


def build_priors(sim_rows: list[dict]):
    question_priors = collections.defaultdict(collections.Counter)
    qphase_priors = collections.defaultdict(collections.Counter)
    for row in sim_rows:
        label = label_text(row)
        question_priors[qkey(row)][label] += 1
        qphase_priors[qphase_key(row)][label] += 1
    return question_priors, qphase_priors


def logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def add_prior(sc: list[float], opts: list[str], counts: collections.Counter, gamma: float):
    if not gamma:
        return
    total = sum(counts.values())
    k = max(1, len(opts))
    for j, opt in enumerate(opts):
        prob = (counts[opt] + 1.0) / (total + k) if total else 1.0 / k
        sc[j] += gamma * math.log(prob)


def prediction_scores(pred: dict, row: dict) -> list[float]:
    sc = [float(x) for x in pred.get("scores", [])]
    if len(sc) < len(row.get("options", [])):
        sc.extend([-1e9] * (len(row.get("options", [])) - len(sc)))
    return sc[: len(row.get("options", []))]


def base_scores(rows, preds, gamma, gamma_qphase, priors):
    """Model score per option, nudged by the sim answer priors (the decoder's emission term)."""
    question_priors, qphase_priors = priors
    scores = []
    for row in rows:
        sc = prediction_scores(preds[row["qa_id"]], row)
        opts = row.get("options", [])
        add_prior(sc, opts, question_priors[qkey(row)], gamma)
        add_prior(sc, opts, qphase_priors[qphase_key(row)], gamma_qphase)
        scores.append(sc)
    return scores


def apply_relation_scores(rows, scores, relation_tables, alpha, tau, weights, relation_cap=2.0):
    out = [list(s) for s in scores]
    real_groups = build_groups(rows)
    for src, dst in RELATIONS:
        rel_weight = weights.get((src, dst), 1.0)
        if rel_weight == 0:
            continue
        counts, dst_labels = relation_tables[(src, dst)]
        for group in real_groups.values():
            if src not in group or dst not in group:
                continue
            si, di = group[src], group[dst]
            src_opts = rows[si].get("options", [])
            dst_opts = rows[di].get("options", [])
            adds = []
            for dst_opt in dst_opts:
                vals = []
                for k, src_opt in enumerate(src_opts):
                    total = sum(counts[src_opt].values()) if src_opt in counts else 0
                    n_labels = max(1, len(dst_labels))
                    if total:
                        c = counts[src_opt][dst_opt]
                        prob = (c + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA * n_labels)
                    else:
                        prob = 1.0 / n_labels
                    vals.append(scores[si][k] / tau + math.log(prob))
                adds.append(logsumexp(vals))
            # The relation table is estimated from sim co-occurrence, not from the model,
            # so its log-domain evidence has no inherent relation to the model's own score
            # scale. Bound the nudge to `relation_cap` times the destination row's own
            # score span so it can re-rank options but never swamp the model regardless of
            # tau/alpha -- the same span-relative guarantee apply_temporal_scores uses.
            raw = [alpha * rel_weight * a for a in adds]
            mean = sum(raw) / len(raw)
            span = (max(scores[di]) - min(scores[di])) or 1.0
            limit = relation_cap * span
            for j in range(len(dst_opts)):
                out[di][j] += max(-limit, min(limit, raw[j] - mean))
    return out


def temporal_logp(transitions, labels_by_question, key, src_phase, dst_phase, src_label, dst_label):
    counts = transitions[(key, src_phase, dst_phase)]
    k = max(1, len(labels_by_question[key]))
    total = sum(counts[src_label].values()) if src_label in counts else 0
    if total:
        c = counts[src_label][dst_label]
        return math.log((c + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA * k))
    return math.log(1.0 / k)


def apply_temporal_scores(rows, scores, temporal_tables, beta, stay_bonus, categories):
    if beta == 0 and stay_bonus == 0:
        return scores
    transitions, labels_by_question = temporal_tables
    category_set = {c.strip() for c in categories.split(",") if c.strip()}
    out = [list(s) for s in scores]
    for (_, key), group in build_temporal_groups(rows).items():
        phases = [p for p in PHASE_ORDER if p in group]
        indices = [group[p] for p in phases]
        if len(indices) < 2:
            continue
        if category_set and rows[indices[0]].get("category", "") not in category_set:
            continue
        dp = [np.asarray(scores[indices[0]], dtype=np.float64)]
        back = [None]
        for t in range(1, len(indices)):
            prev_idx, cur_idx = indices[t - 1], indices[t]
            prev_opts = rows[prev_idx].get("options", [])
            cur_opts = rows[cur_idx].get("options", [])
            cur = np.full(len(cur_opts), -1e18, dtype=np.float64)
            bp = np.zeros(len(cur_opts), dtype=np.int64)
            for j, cur_label in enumerate(cur_opts):
                vals = []
                for k, prev_label in enumerate(prev_opts):
                    transition = temporal_logp(
                        transitions,
                        labels_by_question,
                        key,
                        phases[t - 1],
                        phases[t],
                        prev_label,
                        cur_label,
                    )
                    stay = stay_bonus if prev_label == cur_label else 0.0
                    vals.append(dp[-1][k] + beta * transition + stay)
                best_k = int(np.argmax(vals))
                cur[j] = vals[best_k] + scores[cur_idx][j]
                bp[j] = best_k
            dp.append(cur)
            back.append(bp)

        states = [0] * len(indices)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(len(indices) - 1, 0, -1):
            states[t - 1] = int(back[t][states[t]])
        # Record the Viterbi MAP decision without discarding the model scores:
        # lift only the MAP state one score-span above the current max, leaving
        # every other option at its model score. argmax == MAP state (the winner
        # is strictly above all others), so the model score stays the primary
        # signal and temporal only promotes the phase-consistent choice.
        for idx, state in zip(indices, states):
            row = out[idx]
            span = (max(row) - min(row)) or 1.0
            row[state] = max(row) + span
    return out


def decode(rows, scores, relation_tables, alpha, tau, weights, temporal_tables=None,
           temporal_beta=0.0, temporal_stay=0.0, temporal_categories="", relation_cap=2.0):
    out = apply_relation_scores(rows, scores, relation_tables, alpha, tau, weights, relation_cap)
    if temporal_tables is not None:
        out = apply_temporal_scores(rows, out, temporal_tables, temporal_beta, temporal_stay,
                                    temporal_categories)
    pred_indices = [int(np.argmax(s)) for s in out]
    pred_texts = [
        (row.get("options") or [""])[min(idx, len(row.get("options") or [""]) - 1)]
        for row, idx in zip(rows, pred_indices)
    ]
    return pred_indices, pred_texts


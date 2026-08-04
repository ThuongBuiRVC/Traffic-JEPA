from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from traffic_jepa.modeling.labels import PHASES, SELECTORS

_SEL_INDEX = {s: i for i, s in enumerate(SELECTORS)}
_PHASE_INDEX = {s: i for i, s in enumerate(PHASES)}
QUESTION_BUCKETS = 4096
TEMPLATE_BUCKETS = 8192
OPTION_BUCKETS = 8192
MAX_OPT = 4
TEXT_DIM = 768
LATENT_D = 1024


def _norm_question(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _question_id(text, buckets=QUESTION_BUCKETS):
    h = hashlib.blake2b(_norm_question(text).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little") % int(buckets)


def _hash_id(text, buckets):
    h = hashlib.blake2b(str(text or "").strip().lower().encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little") % int(buckets)


def _graph_group_id(row, scope="phase"):
    domain = row.get("_train_domain", row.get("domain", ""))
    split = row.get("split", "")
    sample = row.get("sample", "")
    phase = row.get("phase", "")
    scope = (scope or "phase").lower()
    if scope == "sample":
        return f"{domain}_{split}_{sample}"
    if scope == "question":
        return f"{domain}_{split}_{sample}_{row.get('category', '')}_{_norm_question(row.get('question', ''))}"
    return f"{domain}_{split}_{sample}_{phase}"


class WTSCachedDataset(Dataset):
    def __init__(self, cache_dir, domain, tokenizer, clips_meta=None,
                 mirror_p=0.0, weak_weight=1.0, consistency=False, bbox_aug=False,
                 bbox_drop_ped_p=0.12, bbox_drop_veh_p=0.75, bbox_frame_drop_p=0.20,
                 bbox_jitter=0.03, index_suffix="", group_context=False, graph_scope="phase",
                 prompt_enrich=False, meta_dropout=0.0):
        self.cache = Path(cache_dir)
        self.tok = tokenizer
        self.mirror_p = float(mirror_p)
        self.weak_weight = float(weak_weight)
        self.consistency = bool(consistency)
        self.bbox_aug = bool(bbox_aug)
        self.bbox_drop_ped_p = float(bbox_drop_ped_p)
        self.bbox_drop_veh_p = float(bbox_drop_veh_p)
        self.bbox_frame_drop_p = float(bbox_frame_drop_p)
        self.bbox_jitter = float(bbox_jitter)
        self.group_context = bool(group_context)
        self.graph_scope = (graph_scope or "phase").lower()
        self.prompt_enrich = bool(prompt_enrich)
        self.meta_dropout = float(meta_dropout)
        self._sib = {}

        idx_name = f"index_{domain}{index_suffix}.jsonl"
        index_path = self.cache / idx_name
        if not index_path.is_file():
            raise FileNotFoundError(f"index file not found: {index_path}")
        self.rows = [json.loads(l) for l in index_path.open() if l.strip()]

        vecs_path = self.cache / "text_vecs.safetensors"
        if not vecs_path.is_file():
            raise FileNotFoundError(f"text_vecs not found: {vecs_path}")
        self.vecs = load_file(str(vecs_path))

        meta_path = Path(clips_meta) if clips_meta else (self.cache / "clips_meta.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"clips_meta not found: {meta_path}")
        self.meta = json.load(meta_path.open())
        self._zero = torch.zeros(TEXT_DIM)

    def __len__(self):
        return len(self.rows)

    def _vec(self, h):
        if not h:
            return self._zero.clone()
        vec = self.vecs.get(h, self._zero)
        if vec.shape != (TEXT_DIM,):
            return self._zero.clone()
        return vec.float()

    def _boxes(self, clip_id):
        m = self.meta.get(clip_id, {})
        def pack(key, vkey):
            box = torch.tensor(m.get(key) or [[0.0] * 4] * 8, dtype=torch.float32)
            vis = torch.tensor([float(v) for v in (m.get(vkey) or [0] * 8)], dtype=torch.float32)
            return box[:8], vis[:8]
        pb, pv = pack("ped", "valid_ped")
        vb, vv = pack("veh", "valid_veh")
        return pb, pv, vb, vv

    def _augment_boxes(self, box, vis, drop_p):
        if not self.bbox_aug:
            return box, vis
        box = box.clone()
        vis = vis.clone()
        if vis.sum() > 0 and torch.rand(()) < drop_p:
            return torch.zeros_like(box), torch.zeros_like(vis)
        if self.bbox_frame_drop_p > 0:
            keep = (torch.rand_like(vis) > self.bbox_frame_drop_p).float()
            vis = vis * keep
        if self.bbox_jitter > 0 and vis.sum() > 0:
            x1, y1, x2, y2 = box.unbind(-1)
            w = (x2 - x1).clamp_min(1e-4)
            h = (y2 - y1).clamp_min(1e-4)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            shift = torch.randn((box.shape[0], 2), dtype=box.dtype) * self.bbox_jitter
            scale = torch.exp(torch.randn((box.shape[0], 2), dtype=box.dtype) * self.bbox_jitter)
            cx = cx + shift[:, 0] * w
            cy = cy + shift[:, 1] * h
            w = w * scale[:, 0]
            h = h * scale[:, 1]
            box = torch.stack([
                (cx - 0.5 * w).clamp(0.0, 1.0),
                (cy - 0.5 * h).clamp(0.0, 1.0),
                (cx + 0.5 * w).clamp(0.0, 1.0),
                (cy + 0.5 * h).clamp(0.0, 1.0),
            ], dim=-1)
        box = box * vis.unsqueeze(-1)
        return box, vis

    def _load_latent(self, clip_id):
        path = self.cache / "latents" / f"{clip_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"latent not found: {path}")
        arr = np.load(path)
        if arr.shape[-1] != LATENT_D:
            raise ValueError(f"latent dim mismatch for {clip_id}: {arr.shape} expected last dim {LATENT_D}")
        return torch.from_numpy(arr).float()

    def _load_group_context(self, group_id):
        path = self.cache / "group_context" / f"{group_id}.npy"
        if not path.is_file():
            return torch.zeros(4, LATENT_D)
        arr = np.load(path)
        if arr.shape[-1] != LATENT_D:
            return torch.zeros(4, LATENT_D)
        return torch.from_numpy(arr[:4]).float()

    def __getitem__(self, i):
        r = self.rows[i]
        latent = self._load_latent(r["clip_id"])

        opt_h = r.get("option_hash", [])[:MAX_OPT]
        opt_texts = r.get("options", [""] * MAX_OPT)[:MAX_OPT]
        opt_vecs = torch.stack([self._vec(h) for h in opt_h] + [self._zero] * (MAX_OPT - len(opt_h)))
        n_opt = len(opt_h)
        ci = min(int(r.get("correct_index", 0)), max(n_opt - 1, 0))
        target_vec = self._vec(r.get("target_hash", ""))
        dist = [self._vec(h) for j, h in enumerate(opt_h) if j != ci]
        dvalid = [1.0] * len(dist) + [0.0] * (MAX_OPT - 1 - len(dist))
        dist += [self._zero] * (MAX_OPT - 1 - len(dist))
        distractor_vecs = torch.stack(dist[:MAX_OPT - 1])
        distractor_valid = torch.tensor(dvalid[:MAX_OPT - 1])
        caption_vec = self._vec(r.get("caption_hash", ""))

        pb, pv, vb, vv = self._boxes(r["clip_id"])
        pb, pv = self._augment_boxes(pb, pv, self.bbox_drop_ped_p)
        vb, vv = self._augment_boxes(vb, vv, self.bbox_drop_veh_p)
        question = r.get("question", "")
        phase_name = str(r.get("phase", "unknown") or "unknown").lower()
        category = r.get("category", "")

        # Anti-shortcut: randomly corrupt metadata so Prior MLP can't memorize text patterns
        meta_corrupted = False
        if self.meta_dropout > 0 and torch.rand(()).item() < self.meta_dropout:
            meta_corrupted = True
            phase_name = PHASES[torch.randint(len(PHASES), ()).item()]
            category = SELECTORS[torch.randint(len(SELECTORS), ()).item()]

        if self.prompt_enrich:
            phase_label = phase_name.replace("_", " ")
            cat_label = category.replace("_", " ")
            question = f"[phase: {phase_label}] [category: {cat_label}] {question}"
        q = self.tok(question, truncation=True, max_length=64, add_special_tokens=True)
        selector = _SEL_INDEX.get(category, 0)
        phase_id = _PHASE_INDEX.get(phase_name, _PHASE_INDEX["unknown"])
        template_id = _hash_id(f"{r.get('category', '?')}|{phase_name}|{_norm_question(question)}", TEMPLATE_BUCKETS)
        option_ids = [_hash_id(t, OPTION_BUCKETS) for t in opt_texts]
        option_ids += [0] * (MAX_OPT - len(option_ids))

        item = {
            "latent": latent,
            "q_ids": torch.tensor(q["input_ids"]),
            "q_mask": torch.tensor(q["attention_mask"]),
            "opt_vecs": opt_vecs,
            "n_opt": torch.tensor(n_opt),
            "target_vec": target_vec,
            "target_hash": r.get("target_hash", ""),
            "distractor_vecs": distractor_vecs,
            "distractor_valid": distractor_valid,
            "caption_vec": caption_vec,
            "ped_box": pb,
            "ped_vis": pv,
            "veh_box": vb,
            "veh_vis": vv,
            "selector": torch.tensor(selector, dtype=torch.long),
            "selector_name": r.get("category", "?"),
            "phase_id": torch.tensor(phase_id, dtype=torch.long),
            "phase_name": phase_name,
            "question_id": torch.tensor(_question_id(question), dtype=torch.long),
            "template_id": torch.tensor(template_id, dtype=torch.long),
            "option_ids": torch.tensor(option_ids[:MAX_OPT], dtype=torch.long),
            "question": question,
            "correct_index": torch.tensor(ci, dtype=torch.long),
            "has_label": torch.tensor(bool(r.get("has_label", True))),
            "qa_id": r.get("qa_id", ""),
            "sample": r.get("sample", ""),
            "group_id": r.get("group_id", r.get("sample", "")),
            "graph_group_id": _graph_group_id(r, self.graph_scope),
            "options_text": opt_texts,
        }
        if self.group_context:
            item["group_context"] = self._load_group_context(item["group_id"])
        return item


class WTSCollator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(b["q_ids"].shape[0] for b in batch)

        def pad(t, val):
            if t.dim() == 1:
                return torch.cat([t, torch.full((maxlen - t.shape[0],), val, dtype=t.dtype)])
            return t

        out = {
            "latent": torch.stack([b["latent"] for b in batch]),
            "q_ids": torch.stack([pad(b["q_ids"], self.pad_id) for b in batch]),
            "q_mask": torch.stack([pad(b["q_mask"], 0) for b in batch]),
            "opt_vecs": torch.stack([b["opt_vecs"] for b in batch]),
            "n_opt": torch.stack([b["n_opt"] for b in batch]),
            "target_vec": torch.stack([b["target_vec"] for b in batch]),
            "distractor_vecs": torch.stack([b["distractor_vecs"] for b in batch]),
            "distractor_valid": torch.stack([b["distractor_valid"] for b in batch]),
            "caption_vec": torch.stack([b["caption_vec"] for b in batch]),
            "ped_box": torch.stack([b["ped_box"] for b in batch]),
            "ped_vis": torch.stack([b["ped_vis"] for b in batch]),
            "veh_box": torch.stack([b["veh_box"] for b in batch]),
            "veh_vis": torch.stack([b["veh_vis"] for b in batch]),
            "selector": torch.stack([b["selector"] for b in batch]),
            "phase_id": torch.stack([b["phase_id"] for b in batch]),
            "question_id": torch.stack([b["question_id"] for b in batch]),
            "template_id": torch.stack([b["template_id"] for b in batch]),
            "option_ids": torch.stack([b["option_ids"] for b in batch]),
            "correct_index": torch.stack([b["correct_index"] for b in batch]),
            "has_label": torch.stack([b["has_label"] for b in batch]),
            "selector_name": [b["selector_name"] for b in batch],
            "phase_name": [b["phase_name"] for b in batch],
            "question": [b["question"] for b in batch],
            "qa_id": [b["qa_id"] for b in batch],
            "sample": [b["sample"] for b in batch],
            "group_id": [b["group_id"] for b in batch],
            "graph_group_id": [b["graph_group_id"] for b in batch],
            "target_hash": [b["target_hash"] for b in batch],
            "options_text": [b["options_text"] for b in batch],
        }
        if "group_context" in batch[0]:
            out["group_context"] = torch.stack([b["group_context"] for b in batch])
        return out

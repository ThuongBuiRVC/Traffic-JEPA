#!/usr/bin/env python
"""Public-test inference: score each WTS question and write a submission.

Per question:
  route question -> (category, view)
  pick camera    -> overhead: largest pedestrian/vehicle box in the phase; else the vehicle video
  pick window    -> phase [start, end] (or bbox-centred for environment) -> 16 frames
  encode         -> V-JEPA 2 @384 -> latent grid
  score          -> cosine(model answer embedding, EmbeddingGemma option vectors) -> argmax

Writes both the submission ([{id, correct}]) and a per-question scored jsonl for the decoder.
"""
from __future__ import annotations
import argparse, json, os, re
from collections import Counter
from pathlib import Path

import numpy as np

PHASE_DIRS = {"0": "0_prerecognition", "1": "1_recognition", "2": "2_judgement",
              "3": "3_action", "4": "4_avoidance"}
LABEL_TO_NUM = {"prerecognition": "0", "recognition": "1", "judgement": "2",
                "action": "3", "avoidance": "4"}
TAKE_IDX = [0, 2, 4, 6, 8, 10, 12, 14]   # 8 of 16 sampled frames (stride 2) — must match training
CROP = 384
NUM_SAMPLE = 16
MIN_WINDOW_S = 1.0
LETTERS = ["a", "b", "c", "d"]

OVERHEAD_RE = re.compile(r"Camera\d|192\.168")


# ----------------------------- raw test bbox -----------------------------
def _bbox_file(bbox_root, kind, scenario, view, video_stem):
    """Prefer bbox_generated (dense, ~per-frame; matches training box density ~7/8 frames).
    bbox_annotated on the public test is sparse (3-5 frames, often 0) -> empty object tokens."""
    base = Path(bbox_root)
    parent = f"normal_trimmed/{scenario}" if "normal" in scenario else scenario   # normal lives under normal_trimmed/
    last = None
    for sub in ("bbox_generated", "bbox_annotated"):
        p = base / sub / kind / "test" / "public" / parent / view / f"{video_stem}_bbox.json"
        last = p
        if p.exists():
            return p
    return last


def load_raw_boxes(bbox_root, kind, scenario, view, video_stem, phase_num=None):
    """-> list of {image_id, xywh} filtered to phase_num (or all)."""
    p = _bbox_file(bbox_root, kind, scenario, view, video_stem)
    if not p.exists():
        return []
    data = json.load(open(p))
    out = []
    for a in data.get("annotations", []):
        if phase_num is not None and str(a.get("phase_number")) != str(phase_num):
            continue
        b = a.get("bbox") or []
        if len(b) >= 4 and b[2] > 0 and b[3] > 0:
            out.append({"image_id": int(a["image_id"]), "xywh": [float(v) for v in b[:4]]})
    return out


# A pedestrian or a vehicle seen from an overhead traffic camera never fills a
# quarter of the frame. Some public-test bbox files carry boxes spanning almost the
# whole image; since the camera is picked by box area, those artefacts would decide
# the view on their own, so they are dropped before any box is used.
MAX_BOX_FRAME_FRACTION = 0.25
_FRAME_AREA: dict = {}


def frame_area(path):
    """Pixel area of a video frame, cached (properties only, no decoding)."""
    key = str(path)
    if key not in _FRAME_AREA:
        import cv2
        cap = cv2.VideoCapture(key)
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        _FRAME_AREA[key] = w * h
    return _FRAME_AREA[key]


def plausible_boxes(boxes, area):
    if not area:
        return boxes
    limit = MAX_BOX_FRAME_FRACTION * area
    return [b for b in boxes if b["xywh"][2] * b["xywh"][3] <= limit]


def _area(boxes):
    return max((b["xywh"][2] * b["xywh"][3] for b in boxes), default=0.0)


def _total(boxes):
    return sum(b["xywh"][2] * b["xywh"][3] for b in boxes)


# ----------------------------- camera selection -----------------------------
def split_views(videos):
    overhead = [v for v in videos if OVERHEAD_RE.search(v)]
    vehicle = [v for v in videos if "vehicle_view" in v]
    return overhead, vehicle


# Scene-level questions (weather, road layout, lighting) describe the whole
# intersection, so narrowing the camera and the window to the queried actor loses
# the evidence they need. Only actor-centric questions use the boxes.
SCENE_SELECTORS = {"environment"}
# Questions whose answer compares the two actors need both of them in view.
RELATION_SELECTORS = {"pedestrian_vehicle_distance", "pedestrian_position"}
VEHICLE_SELECTORS = {"vehicle_orientation", "vehicle_behavior"}


def actor_bbox_root(bbox_root, selector):
    return "" if selector in SCENE_SELECTORS else bbox_root


def choose_overhead_camera(bbox_root, scenario, videos, selector, phase_num, videos_root=""):
    bbox_root = actor_bbox_root(bbox_root, selector)
    scored = []
    for idx, v in enumerate(videos):
        stem = Path(v).stem
        ped = load_raw_boxes(bbox_root, "pedestrian", scenario, "overhead_view", stem, phase_num)
        veh = load_raw_boxes(bbox_root, "vehicle", scenario, "overhead_view", stem, phase_num)
        if videos_root:
            p = _video_path(videos_root, scenario, "overhead_view", v)
            a = frame_area(p) if p else 0.0
            ped, veh = plausible_boxes(ped, a), plausible_boxes(veh, a)
        scored.append({"video": v, "idx": idx, "ped_area": _area(ped), "veh_area": _area(veh),
                       "ped_seen": len(ped), "veh_seen": len(veh),
                       "occ": _total(ped) + _total(veh)})
    if not scored:
        return None, "no_overhead"
    if selector == "S_SCENE_CONTEXT":
        pool = [s for s in scored if s["occ"] > 0] or scored
        best = min(pool, key=lambda s: (s["occ"], s["idx"]))
        return best["video"], "scene_lowest_occ"
    # Rank by the actor the question is about. Requiring both actors to be visible
    # only makes sense when the answer compares them; for a question about the
    # pedestrian alone it discards the view that shows the pedestrian best.
    if selector in RELATION_SELECTORS:
        best = max(scored, key=lambda s: (s["ped_area"] > 0 and s["veh_area"] > 0,
                                          s["ped_area"] + s["veh_area"], -s["idx"]))
        if best["ped_area"] > 0 and best["veh_area"] > 0:
            return best["video"], "largest_covisible"
        if best["ped_area"] > 0 or best["veh_area"] > 0:
            return best["video"], "single_object"
        return scored[0]["video"], "fallback_first"
    kind = "veh" if selector in VEHICLE_SELECTORS else "ped"
    best = max(scored, key=lambda s: (s[f"{kind}_area"], s[f"{kind}_seen"], -s["idx"]))
    if best[f"{kind}_area"] > 0:
        return best["video"], f"largest_{kind}"
    other = "ped" if kind == "veh" else "veh"
    fb = max(scored, key=lambda s: (s[f"{other}_area"], -s["idx"]))
    return (fb["video"], f"fallback_{other}") if fb[f"{other}_area"] > 0 else (scored[0]["video"], "fallback_first")


# ----------------------------- window / frames -----------------------------
def clamp(s, e, dur, fps):
    ft = 1.0 / max(fps, 1.0); de = max(0.0, dur - ft)
    s = max(0.0, min(s, de)); e = max(s, min(e, de))
    if e <= s:
        e = min(de, s + ft)
    return s, e


def expand_min(s, e, dur, fps, mn=MIN_WINDOW_S):
    s, e = clamp(s, e, dur, fps)
    if e - s >= mn:
        return s, e
    c = (s + e) / 2
    return clamp(c - mn / 2, c + mn / 2, dur, fps)


def choose_window(selector, phase_info, bbox_root, scenario, view, stem, fps, dur, phase_num,
                  frame_px=0.0):
    """Return (start,end) seconds. Mirrors val choose_window for phase + environment cases."""
    bbox_root = actor_bbox_root(bbox_root, selector)
    if phase_info is None:                                 # environment (no phase)
        if selector == "S_SCENE_CONTEXT":
            return clamp(dur * 0.05, dur * 0.95, dur, fps)
        ped = plausible_boxes(load_raw_boxes(bbox_root, "pedestrian", scenario, view, stem, None), frame_px)
        if ped:
            best = max(ped, key=lambda b: b["xywh"][2] * b["xywh"][3])
            c = best["image_id"] / fps
            return expand_min(c - 1.0, c + 1.0, dur, fps, 2.0)
        return clamp(dur * 0.05, dur * 0.95, dur, fps)
    ps, pe = clamp(phase_info["start"], phase_info["end"], dur, fps)
    ped = plausible_boxes(load_raw_boxes(bbox_root, "pedestrian", scenario, view, stem, phase_num), frame_px)
    veh = plausible_boxes(load_raw_boxes(bbox_root, "vehicle", scenario, view, stem, phase_num), frame_px)
    frames = [b["image_id"] for b in (ped + veh)]
    if frames:
        s = max(min(frames) / fps - 0.5, ps); e = min(max(frames) / fps + 0.5, pe)
        return expand_min(s, e, dur, fps)
    return expand_min(ps, pe, dur, fps)


def _video_meta_cv2(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return (fps or 30.0), n


def _read_frames_cv2(path, indices):
    """Read specific frame indices -> (len,H,W,3) uint8 RGB. Seeks per frame (codec-robust)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for fi in indices:
        fi = max(0, min(int(fi), max(n - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok or fr is None:
            fr = out[-1] if out else np.zeros((CROP, CROP, 3), np.uint8)
        else:
            fr = fr[:, :, ::-1]                              # BGR -> RGB
        out.append(np.ascontiguousarray(fr))
    cap.release()
    return np.stack(out)


def frame_indices(start, end, fps, total):
    a = int(round(start * fps)); b = int(round(end * fps))
    a = max(0, min(a, max(total - 1, 0))); b = max(a, min(b, max(total - 1, 0)))
    return [int(round(x)) for x in np.linspace(a, b, NUM_SAMPLE)]


# ----------------------------- boxes for the 8 encoded frames -----------------------------
# ----------------------------- main (build jobs; encode+score gated by --run) -----------------------------
def build_jobs(test_json, route, bbox_root, videos_root, video_meta_fn):
    """Yield one job dict per WTS conversation. video_meta_fn(path)->(fps,frames,W,H)."""
    scenarios = json.load(open(test_json))
    jobs = []
    for scen in scenarios:
        videos = scen.get("videos", [])
        if all(re.match(r"video\d+", v) for v in videos):    # BDD/external -> skip
            continue
        overhead, vehicle = split_views(videos)
        scenario = _scenario_of(videos)
        # collect (phase_info, conversations)
        units = [(None, scen.get("conversations", []))]
        for ph in scen.get("event_phase", []):
            label = (ph.get("labels") or ["?"])[0]
            pnum = LABEL_TO_NUM.get(str(label).lower())
            units.append(({"start": float(ph["start_time"]), "end": float(ph["end_time"]),
                           "num": pnum, "label": str(label).lower()}, ph.get("conversations", [])))
        for phase_info, convs in units:
            for c in convs:
                q = c["question"]; r = route.get(q)
                if r is None:
                    continue
                sel, requested_view = r["category"], r["view"]
                pnum = phase_info["num"] if phase_info else None
                if requested_view == "vehicle_view":
                    if vehicle:
                        vid = vehicle[0]
                        src_view = "vehicle_view"
                        reason = "vehicle_single"
                    elif overhead:
                        # Public test sometimes asks vehicle-view templates but only lists overhead
                        # cameras. Stay inside the allowed camera list and use overhead bbox paths;
                        # never pretend an overhead video is a vehicle_view clip.
                        vid, overhead_reason = choose_overhead_camera(
                            bbox_root, scenario, overhead, "S_RELATION_PED_VEH", pnum, videos_root
                        )
                        src_view = "overhead_view"
                        reason = f"vehicle_missing_use_allowed_overhead:{overhead_reason}"
                    else:
                        vid = None
                        src_view = "vehicle_view"
                        reason = "vehicle_missing_no_allowed_video"
                else:
                    pool = overhead or vehicle
                    if not pool:
                        continue
                    vid, reason = choose_overhead_camera(bbox_root, scenario, pool, sel, pnum, videos_root)
                    src_view = "overhead_view"
                if vid is None:
                    continue
                jobs.append({"id": c["id"], "question": q, "options": [c.get(l, "") for l in LETTERS],
                             "selector": sel, "view": src_view, "scenario": scenario,
                             "requested_view": requested_view,
                             "video": vid, "phase": phase_info, "phase_num": pnum,
                             "phase_name": phase_info["label"] if phase_info else "environment",
                             "camera_reason": reason})
    return jobs


def _scenario_of(videos):
    for v in videos:
        m = re.search(r"(\d{8}_\d+_[A-Z]+\d+_T\d+|\d{8}_\d+_normal_[\d.]+_\w+_event_\d+)", v)
        if m:
            return m.group(1)
    # fallback: strip camera + ext
    return re.sub(r"_(Camera\d+_\d+|192\.168[\d._]+|vehicle_view).*", "", Path(videos[0]).stem)


from traffic_jepa.modeling.labels import PHASES, SELECTORS
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)


def _video_path(videos_root, scenario, view, video):
    """Locate the source video under videos/test (layout: <scenario>/<view>/<video>)."""
    root = Path(videos_root)
    for cand in (root / scenario / view / video, root / scenario / video, root / video):
        if cand.exists():
            return cand
    hits = list(root.rglob(video))
    return hits[0] if hits else None


def run_inference(jobs, videos_root, bbox_root, ckpt, vjepa_spec, llama, gemma, out_path, limit=0, batch_size=32):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from traffic_jepa.modeling.traffic_jepa_model import TrafficJEPAModel
    phase_index = {p: i for i, p in enumerate(PHASES)}
    dev = "cuda"
    if limit:
        jobs = jobs[:limit]

    # --- V-JEPA 2.1 encoder ---
    if vjepa_spec.startswith("torchhub_local:"):
        repo = vjepa_spec.split(":", 1)[1]
        enc, _ = torch.hub.load(repo, "vjepa2_1_vit_large_384", source="local", trust_repo=True, pretrained=True)
    else:
        enc, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_large_384", trust_repo=True, pretrained=True)
    enc = enc.to(dev).float().eval()

    # --- Gemma text vecs for all unique option texts ---
    texts = sorted({o for j in jobs for o in j["options"] if o})
    gtok = AutoTokenizer.from_pretrained(gemma)
    gmodel = AutoModel.from_pretrained(gemma, torch_dtype=torch.float32).to(dev).eval()
    vec = {}
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            enc_in = gtok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            out = gmodel(**enc_in, output_hidden_states=True, return_dict=True)
            hid = getattr(out, "last_hidden_state", None)
            if hid is None:
                hid = out.hidden_states[-1]
            m = enc_in["attention_mask"].unsqueeze(-1).to(hid.dtype)
            emb = (hid * m).sum(1) / m.sum(1).clamp_min(1.0)    # MEAN-pool (matches preprocess phase_text)
            for t, e in zip(chunk, emb):
                vec[t] = e.float().cpu()
    del gmodel; torch.cuda.empty_cache()
    zero = torch.zeros(next(iter(vec.values())).shape[-1])

    # The model architecture must match the one that produced the checkpoint, so read
    # run_args.json (written by train.py) sitting next to the checkpoint. Guessing here
    # is unsafe: e.g. spatial_pool_size changes the visual-token layout and silently
    # produces different scores, so fail loudly if the config is missing.
    import json
    from pathlib import Path
    args_file = Path(ckpt).parent / "run_args.json"
    if not args_file.is_file():
        raise SystemExit(
            f"FATAL: {args_file} not found. The checkpoint's run_args.json (training config) "
            f"must sit next to it so the model is rebuilt with the same architecture."
        )
    with open(args_file) as f:
        cfg = json.load(f)

    # --- Traffic-JEPA model (rebuilt from the training config) ---
    model = TrafficJEPAModel(
        predictor_model=llama,
        num_layers=cfg.get("predictor_layers", 6),
        temperature=cfg.get("temperature", 0.07),
        dropout=cfg.get("dropout", 0.1),
        token_dropout=cfg.get("token_dropout", 0.0),
        unfreeze_backbone_layers=cfg.get("unfreeze", 0),
        use_prior_tokens=not cfg.get("no_prior_tokens", False),
        spatial_pool_size=cfg.get("spatial_pool_size", 1),
        ablation_no_visual=cfg.get("ablation_no_visual", False),
        grad_ckpt=False,
    ).to(dev).eval()
    sd = torch.load(ckpt, map_location="cpu")
    
    # Pad selector_embed if checkpoint was trained on fewer classes
    if "selector_embed.weight" in sd and hasattr(model, "selector_embed"):
        ckpt_sel_w = sd["selector_embed.weight"]
        model_sel_w = model.selector_embed.weight
        if ckpt_sel_w.size(0) < model_sel_w.size(0):
            print(f"Padding selector_embed from {ckpt_sel_w.size(0)} to {model_sel_w.size(0)} classes")
            padded = model_sel_w.clone().detach()
            padded[:ckpt_sel_w.size(0)] = ckpt_sel_w
            sd["selector_embed.weight"] = padded
            
    model.load_state_dict(sd, strict=False)
    qtok = model.tokenizer

    @torch.no_grad()
    def encode_latent(path, fidx, j=None):
        idx16 = [int(i) for i in fidx]
        eight = [idx16[i] for i in TAKE_IDX] if len(idx16) >= 16 else idx16
        frames = _read_frames_cv2(path, eight)              # (8,H,W,3) RGB
        H, W = frames.shape[1:3]
        x = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 255.0
        x = F.interpolate(x, size=(CROP, CROP), mode="bilinear", align_corners=False)
        x = (x - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
        out = enc(x.unsqueeze(0).to(dev))
        tok = (out[0] if isinstance(out, (list, tuple)) else out)[0]
        side = round((tok.shape[0] / (len(TAKE_IDX) // 2)) ** 0.5)
        T = tok.shape[0] // (side * side)
        grid = tok.reshape(T, side, side, tok.shape[-1]).float()
        return grid, (W, H), eight

    # DEDUPE: many questions share the same (video, window) clip -> encode latent+boxes ONCE, reuse.
    # Also persist V-JEPA latent cache to disk for fast re-runs.
    import pickle, hashlib
    cache_dir = Path(bbox_root).parent / "vjepa_test_cache_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    clip_cache = {}
    # Phase 1: encode all unique clips (V-JEPA) — cacheable
    clip_keys_per_job = []
    for k, j in enumerate(jobs):
        path = j.get("path")
        if path is None:
            clip_keys_per_job.append(None); continue
        ck = (path, round(j["window"][0], 2), round(j["window"][1], 2)) if "window" in j else (path, 0, 0)
        clip_keys_per_job.append(ck)
        if ck in clip_cache:
            continue
        # Check disk cache first
        ck_hash = hashlib.md5(f"{ck}".encode()).hexdigest()
        disk_path = cache_dir / f"{ck_hash}.pkl"
        if disk_path.exists():
            clip_cache[ck] = pickle.load(open(disk_path, "rb"))
            continue
        try:
            fidx = frame_indices(j["window"][0], j["window"][1], j["fps"], j["frames"]) if "window" in j \
                else list(np.linspace(0, j["frames"] - 1, NUM_SAMPLE))
            grid, (W, H), eight = encode_latent(path, fidx, j)
            clip_cache[ck] = grid.cpu()
            pickle.dump(clip_cache[ck], open(disk_path, "wb"))
        except Exception as e:
            import traceback; traceback.print_exc()
            clip_cache[ck] = None
        if k % 50 == 0:
            print(f"[encode] {k}/{len(jobs)} clips cached={len(clip_cache)}", flush=True)
    print(f"[encode] done. {len(clip_cache)} unique clips.", flush=True)

    # Phase 2: batched WTS model inference
    BATCH_SIZE = batch_size
    # Prepare all items first
    items = []  # list of (job_idx, prepared_batch_dict) or (job_idx, None)
    for k, j in enumerate(jobs):
        ck = clip_keys_per_job[k]
        if ck is None or clip_cache.get(ck) is None:
            items.append((k, None)); continue
        try:
            grid = clip_cache[ck]
            q = qtok(j["question"], truncation=True, max_length=48, add_special_tokens=True)
            ov = [vec.get(o, zero) for o in j["options"]]
            nopt = sum(1 for o in j["options"] if o)
            phase_name = str(j.get("phase_name", "unknown")).lower()
            item = {
                "latent": grid.unsqueeze(0),
                "selector": torch.tensor([SELECTORS.index(j["selector"])]),
                "selector_name": j["selector"],
                "phase_id": torch.tensor([phase_index.get(phase_name, phase_index["unknown"])]),
                "phase_name": phase_name,
                "options_text": j["options"],
                "q_ids": torch.tensor([q["input_ids"]]), "q_mask": torch.tensor([q["attention_mask"]]),
                "opt_vecs": torch.stack(ov).unsqueeze(0), "n_opt": torch.tensor([nopt]),
            }
            items.append((k, item))
        except Exception as e:
            items.append((k, None))

    preds = [None] * len(jobs)
    # Fill in fallbacks
    for k, j in enumerate(jobs):
        if items[k][1] is None:
            preds[k] = {"id": j["id"], "correct": "a", "note": "video_missing_or_err"}

    # Batch and run
    valid_items = [(k, it) for k, it in items if it is not None]
    for bi in range(0, len(valid_items), BATCH_SIZE):
        batch_items = valid_items[bi:bi + BATCH_SIZE]
        # Collate: pad q_ids/q_mask to same length
        max_qlen = max(it["q_ids"].shape[-1] for _, it in batch_items)
        collated = {}
        for key in batch_items[0][1]:
            if key in ("selector_name", "options_text", "phase_name"):
                collated[key] = [it[key] for _, it in batch_items]
                continue
            tensors = []
            for _, it in batch_items:
                t = it[key]
                if key in ("q_ids", "q_mask") and t.shape[-1] < max_qlen:
                    pad = torch.zeros(1, max_qlen - t.shape[-1], dtype=t.dtype)
                    t = torch.cat([t, pad], dim=-1)
                tensors.append(t)
            collated[key] = torch.cat(tensors, dim=0)
        collated = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in collated.items()}
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if args.tta_mc > 1:
                    model.train()  # Enable dropout for MC-Dropout
                    sum_sims = 0
                    for _ in range(args.tta_mc):
                        y_hat, dl = model.forward(collated)[:2]
                        sims = model.option_scores(y_hat, dl, collated)
                        sum_sims = sum_sims + sims
                    pred_indices = torch.argmax(sum_sims, dim=-1)
                    model.eval()
                else:
                    y_hat, dl = model.forward(collated)[:2]
                    pred_indices = model.final_pred(y_hat, dl, collated)
                    sims = model.option_scores(y_hat, dl, collated)
            for i, (k, _) in enumerate(batch_items):
                sc = sum_sims[i] if args.tta_mc > 1 else sims[i]
                n_opt = int(collated["n_opt"][i].item())
                preds[k] = {
                    "id": jobs[k]["id"],
                    "correct": LETTERS[int(pred_indices[i].item())],
                    "scores": sc.tolist()[:n_opt]
                }
        except Exception as e:
            import traceback; traceback.print_exc()
            for i, (k, _) in enumerate(batch_items):
                if preds[k] is None:
                    preds[k] = {"id": jobs[k]["id"], "correct": "a", "note": f"batch_err:{e}"}
        if bi % (BATCH_SIZE * 10) == 0:
            print(f"[score] {bi}/{len(valid_items)}", flush=True)

    # Ensure no None preds
    for k in range(len(preds)):
        if preds[k] is None:
            preds[k] = {"id": jobs[k]["id"], "correct": "a", "note": "missed"}

    json.dump([{"id": p["id"], "correct": p["correct"]} for p in preds], open(out_path, "w"), indent=1)
    
    # Save scored predictions jsonl for graph decode
    out_jsonl = str(out_path).replace(".json", "") + "_scored.jsonl"
    with open(out_jsonl, "w") as f:
        for k, p in enumerate(preds):
            j = jobs[k]
            f.write(json.dumps({
                "qa_id": p["id"],
                "category": j["selector"],
                "question": j["question"],
                "pred_index": LETTERS.index(p["correct"]) if p["correct"] in LETTERS else 0,
                "pred_text": j["options"][LETTERS.index(p["correct"])] if p["correct"] in LETTERS else "",
                "options": j["options"],
                "scores": p.get("scores", [])
            }) + "\n")

    nbad = sum("note" in p for p in preds)
    print(f"DONE preds={len(preds)} fallback/err={nbad} -> {out_path} & {out_jsonl}", flush=True)
    if nbad > 0.10 * max(len(preds), 1):
        raise SystemExit(
            f"FATAL: {nbad}/{len(preds)} predictions are fallback (video/encode failed). "
            "The model did not actually score these — submission would be invalid. "
            "Check --videos-root / --bbox-root.")
    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--bbox-root", required=True, help="dir containing bbox_annotated/")
    ap.add_argument("--videos-root", default="", help="videos/test dir (needed for --run)")
    ap.add_argument("--dump-jobs", default="", help="write jobs (camera/window selection) to JSON and exit")
    ap.add_argument("--run", action="store_true", help="encode + score (needs GPU + videos)")
    ap.add_argument("--ckpt", default=""); ap.add_argument("--vjepa", default="")
    ap.add_argument("--llama", default=""); ap.add_argument("--gemma", default="")
    ap.add_argument("--out", default="vqa_test_predictions.json"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32, help="WTS model inference batch size")
    ap.add_argument("--tta_mc", type=int, default=1, help="Number of MC-Dropout Test-Time Augmentation passes (default 1)")
    args = ap.parse_args()
    route = json.load(open(args.route))
    jobs = build_jobs(args.test, route, args.bbox_root, args.videos_root, None)
    print(f"jobs: {len(jobs)}  by selector: {dict(Counter(j['selector'] for j in jobs))}")
    print("camera reasons:", dict(Counter(j["camera_reason"] for j in jobs)))
    if args.dump_jobs:
        json.dump(jobs, open(args.dump_jobs, "w"), indent=1); print("dumped ->", args.dump_jobs)
    if args.run:
        # index all videos ONCE (avoid per-job rglob over 22GB) + meta/window per unique video
        print("indexing videos...", flush=True)
        vindex = {p.name: p for p in Path(args.videos_root).rglob("*.mp4")}
        print(f"  {len(vindex)} videos indexed", flush=True)
        if not vindex:
            raise SystemExit(
                f"FATAL: 0 videos found under --videos-root={args.videos_root!r}. "
                "Inference cannot run without the public-test videos — fix the path. "
                "(Otherwise every prediction becomes a fallback and the submission is invalid.)")
        meta = {}
        for j in jobs:
            p = vindex.get(j["video"])
            j["path"] = str(p) if p else None
            if p is None:
                j["fps"], j["frames"] = 30.0, 0; continue
            if p not in meta:
                meta[p] = _video_meta_cv2(p)
            j["fps"], j["frames"] = meta[p]
            dur = j["frames"] / j["fps"] if j["fps"] else 0
            j["window"] = choose_window(j["selector"], j["phase"], args.bbox_root, j["scenario"],
                                        j["view"], Path(j["video"]).stem, j["fps"], dur, j["phase_num"],
                                        frame_area(p))
        run_inference(jobs, args.videos_root, args.bbox_root, args.ckpt, args.vjepa,
                      args.llama, args.gemma, args.out, args.limit, args.batch_size)

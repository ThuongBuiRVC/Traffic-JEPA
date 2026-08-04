#!/usr/bin/env python3
"""Build manifest + clip_16 + bbox_json cho preprocess_embeddings.py từ data SynWTS thật.

Từ annotations/vqa + videos + bbox_annotated -> với mỗi QA:
  chọn video (camera) -> window theo phase -> sample 16 frame -> cắt clip_16 mp4 (384x384)
  -> bbox_json (8 frame TAKE_IDX, xyxy_norm) -> 1 dòng manifest.

Output (khớp scripts/preprocess_embeddings.py MANIFESTS["sim"]):
  data/sim_qa_vljepa16/clips/<id>.mp4
  data/sim_qa_vljepa16/bbox/<id>.json
  data/sim_qa_vljepa16/_review/vjepa16_manifest_{train,val}.jsonl

ponytail: builder TỐI THIỂU đủ chạy preprocess->train trên data thật.
  - camera = video có bbox lớn nhất trong phase (fallback: video đầu).
  - selector = map keyword từ câu hỏi (không cần q_route.json).
  - sim domain (SynWTS là synthetic). real benchmark build riêng nếu có.
  Tinh chỉnh (sampling_policy nhiều kiểu, sim/real pairing) khi cần.

Usage: python scripts/build_wts_manifest.py --split val --limit 20
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import cv2
import numpy as np

from traffic_jepa.modeling.labels import SELECTORS

TAKE_IDX = [0, 2, 4, 6, 8, 10, 12, 14]
NUM_SAMPLE = 16
CROP = 384
LABEL_TO_NUM = {"prerecognition": "0", "recognition": "1", "judgement": "2",
                "action": "3", "avoidance": "4", "environment": None}

# câu hỏi -> selector (keyword, đủ cho selector embedding + dist_head)
def selector_of(q: str, lab: str = "") -> str:
    if lab == "environment":
        return "environment"
    q = q.lower()
    ped = "pedestrian" in q
    if "distance" in q:
        return "pedestrian_vehicle_distance" if ped else "vehicle_ego_distance"
    if "position" in q or "relative" in q:
        return "pedestrian_position" if ped else "vehicle_position"
    if "orientation" in q or "direction" in q or "field of view" in q:
        return "pedestrian_orientation" if ped else "vehicle_orientation"
    if "speed" in q:
        return "pedestrian_speed" if ped else "vehicle_speed"
    if "action" in q or "behavior" in q:
        return "pedestrian_behavior" if ped else "vehicle_behavior"
    if "line of sight" in q or "gaze" in q or "looking" in q or "awareness" in q or "visual" in q:
        return "pedestrian_gaze"
    return "pedestrian_orientation"


def iter_qa(vqa_root: Path, split: str):
    """Yield (scenario, view, qa, phase_label, start, end) — gồm cả normal_trimmed + environment."""
    split_dir = vqa_root / split
    if not split_dir.exists():
        return
    for jf in sorted(split_dir.rglob("*.json")):
        parts = jf.relative_to(split_dir).parts
        if parts and parts[0] == "normal_trimmed":
            parts = parts[1:]
        if len(parts) < 3:
            continue
        scenario, view = parts[0], parts[1]
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        for item in (data if isinstance(data, list) else [data]):
            for ph in item.get("event_phase", []):
                lab = (ph.get("labels") or ["?"])[0]
                for qa in ph.get("conversations", []):
                    yield scenario, view, qa, lab, ph.get("start_time"), ph.get("end_time")
            for qa in item.get("environment", []):
                yield scenario, view, qa, "environment", None, None


def load_augment(aug_dir: str, split: str):
    """{(scenario,view,phase_label,question): {hard:[...], pad:[...]}} — CHỈ cho train."""
    aug = {}
    if split != "train":                 # augment chỉ tạo trên train -> val giữ option thật
        return aug
    d = Path(aug_dir)
    def add(fname, field, dest):
        p = d / fname
        if not p.exists():
            return
        for r in json.load(open(p)):
            key = (r.get("scenario"), r.get("view"), r.get("phase_label"), r.get("question", ""))
            v = r.get(field)
            v = list(v.values()) if isinstance(v, dict) else (v or [])
            aug.setdefault(key, {}).setdefault(dest, []).extend(v)
    add("hard_negatives.json", "hard_negatives", "hard")
    add("pad_options.json", "new_options", "pad")
    return aug


def make_options(qa, scenario, view, lab, aug):
    """-> (options_list, correct_letter, correct_text). Train: trộn hard negatives vào distractor."""
    base = [(k, qa[k]) for k in "abcd" if qa.get(k)]
    correct = qa.get("correct", "a")
    ctext = qa.get(correct, base[0][1] if base else "")
    orig = [v for k, v in base if v != ctext]
    extra = aug.get((scenario, view, lab, qa.get("question", "")), {})
    # CHỈ dùng pad để LẤP cho câu <4 option (an toàn). KHÔNG dùng hard-neg:
    # hard-neg gần nghĩa -> collapse trong EmbeddingGemma -> InfoNCE nhiễu, hại val.
    pool, seen = [], {str(ctext).strip().lower()}
    for v in orig + extra.get("pad", []):
        vl = str(v).strip().lower()
        if vl and vl not in seen:
            seen.add(vl); pool.append(v)
    opts = [ctext] + pool[:3]
    random.Random(hash((scenario, view, lab, qa.get("question", ""))) & 0xffffffff).shuffle(opts)
    return opts, "abcd"[opts.index(ctext)], ctext


def load_captions(caption_root: Path, split: str):
    """-> {(scenario, view, phase_label): "ped caption. veh caption"} cho Caption Alignment."""
    cap = {}
    sd = caption_root / split
    if not sd.exists():
        return cap
    for jf in sorted(sd.rglob("*.json")):
        parts = jf.relative_to(sd).parts
        if parts and parts[0] == "normal_trimmed":
            parts = parts[1:]
        if len(parts) < 3:
            continue
        scenario, view = parts[0], parts[1]
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        for ph in data.get("event_phase", []):
            lab = (ph.get("labels") or ["?"])[0]
            text = " ".join(t for t in (ph.get("caption_pedestrian", ""),
                                        ph.get("caption_vehicle", "")) if t).strip()
            if text:
                cap[(scenario, view, lab)] = text
    return cap


def load_boxes(bbox_root: Path, kind: str, split: str, scenario: str, view: str, stem: str, pnum=None):
    """-> {image_id: [x,y,w,h]} từ bbox_annotated (lọc theo phase_number nếu pnum)."""
    parent = f"normal_trimmed/{scenario}" if "normal" in scenario else scenario
    p = bbox_root / kind / split / parent / view / f"{stem}_bbox.json"
    if not p.exists():
        return {}
    out = {}
    for a in json.load(open(p)).get("annotations", []):
        if pnum is not None and str(a.get("phase_number")) != str(pnum):
            continue
        b = a.get("bbox") or []
        if len(b) >= 4 and b[2] > 0 and b[3] > 0:
            out[int(a["image_id"])] = [float(v) for v in b[:4]]
    return out


def choose_camera(videos_root, bbox_root, split, scenario, view, pnum):
    """Chọn video có bbox ped+veh TO NHẤT trong phase (tín hiệu thị giác rõ nhất).
    -> (video_path, ped_boxes, veh_boxes) cho video được chọn; fallback video đầu."""
    parent = f"normal_trimmed/{scenario}" if "normal" in scenario else scenario
    actual_view = view
    if view == "environment":
        # Fallback for environment questions where view folder doesn't exist
        if (videos_root / split / parent / "overhead_view").exists():
            actual_view = "overhead_view"
        elif (videos_root / split / parent / "vehicle_view").exists():
            actual_view = "vehicle_view"

    vdir = videos_root / split / parent / actual_view
    if not vdir.exists():
        return None, {}, {}
    vids = sorted(p for p in vdir.iterdir() if p.is_file())
    if not vids:
        return None, {}, {}
    best, best_score, best_pb, best_vb = None, -1.0, {}, {}
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        cap.release()
        if not ok:
            continue
        # Pass actual_view to load_boxes to avoid crashing when view is "environment"
        pb = load_boxes(bbox_root, "pedestrian", split, scenario, actual_view, v.stem, pnum)
        vb = load_boxes(bbox_root, "vehicle", split, scenario, actual_view, v.stem, pnum)
        area = max((b[2] * b[3] for b in list(pb.values()) + list(vb.values())), default=0.0)
        if area > best_score:
            best, best_score, best_pb, best_vb = v, area, pb, vb
    if best is None:
        return None, {}, {}
    return best, best_pb, best_vb


def _f(x):
    try:
        return float(x[0] if isinstance(x, (list, tuple)) else x)
    except (TypeError, ValueError):
        return None


def sample_indices(start, end, fps, total):
    start, end = _f(start), _f(end)
    if start is None or end is None:
        a, b = 0, total - 1
    else:
        a, b = int(round(start * fps)), int(round(end * fps))
    a = max(0, min(a, total - 1)); b = max(a, min(b, total - 1))
    if b <= a:
        b = min(total - 1, a + NUM_SAMPLE)
    return [int(round(x)) for x in np.linspace(a, b, NUM_SAMPLE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", default="data/synwts/data/annotations")
    ap.add_argument("--videos", default="data/synwts/data/videos")
    ap.add_argument("--out", default="data/sim_qa_vljepa16")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--augment-dir", default="data/synwts/data/annotations_augmented")
    ap.add_argument("--no-augment", action="store_true", help="bỏ trộn hard negatives")
    ap.add_argument("--clip-id-prefix", default="", help="prefix clip ids to avoid cache collisions, e.g. real_")
    ap.add_argument("--manifest-prefix", default="vjepa16", help="manifest filename prefix")
    args = ap.parse_args()

    vqa_root = Path(args.annot) / "vqa"
    bbox_root = Path(args.annot) / "bbox_annotated"
    cap_map = load_captions(Path(args.annot) / "caption", args.split)
    aug = {} if args.no_augment else load_augment(args.augment_dir, args.split)
    print(f"augment entries (train hard-neg): {len(aug)}")
    videos_root = Path(args.videos)
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "bbox").mkdir(parents=True, exist_ok=True)
    (out / "_review").mkdir(parents=True, exist_ok=True)

    rows, n_qa, skipped = [], 0, 0
    cap_cache = {}                       # video -> (fps, n, W, H)
    built = {}                           # (scenario,view,lab) -> (clip_path, bbox_path): cắt 1 lần/clip
    for scenario, view, qa, lab, st, en in iter_qa(vqa_root, args.split):
        if args.limit and n_qa >= args.limit:
            break
        n_qa += 1
        clip_key = (scenario, view, lab)
        if clip_key in built:            # nhiều QA chung 1 clip phase -> tái dùng
            clip_path, bbox_path = built[clip_key]
            opts, correct, ctext = make_options(qa, scenario, view, lab, aug)
            labs = {lab, LABEL_TO_NUM.get(lab) or lab}
            caption = next((cap_map[(scenario, view, l)] for l in labs
                            if (scenario, view, l) in cap_map), "") or next(
                (v for (s, _, l), v in cap_map.items() if s == scenario and l in labs), "")
            rows.append({
                "qa_id": f"{clip_path.stem}_{len(rows)}", "benchmark_id": f"{clip_path.stem}_{len(rows)}",
                "sample": scenario, "split": args.split, "selector": selector_of(qa.get("question", ""), lab),
                "sampling_policy": lab, "phase": lab,
                "clip_16": str(clip_path), "bbox_json": str(bbox_path),
                "question": qa.get("question", ""), "options": opts,
                "correct": correct, "correct_text": ctext,
                "view": view, "template_view": view, "caption": caption,
            })
            continue
        pnum = LABEL_TO_NUM.get(lab)
        vid, pbox, vbox = choose_camera(videos_root, bbox_root, args.split, scenario, view, pnum)
        if vid is None:
            skipped += 1; continue
        if vid not in cap_cache:
            c = cv2.VideoCapture(str(vid))
            fps = float(c.get(cv2.CAP_PROP_FPS) or 30.0)
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            W = int(c.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); H = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            c.release(); cap_cache[vid] = (fps, n, W, H)
        fps, n, W, H = cap_cache[vid]
        if n <= 0:
            skipped += 1; continue

        # window bám bbox: lấy quanh khoảng frame ped/veh xuất hiện trong phase (tín hiệu rõ);
        # fallback sang phase time nếu không có bbox.
        ids = sorted(list(pbox) + list(vbox))
        if ids:
            a = max(0, ids[0] - 4); b = min(n - 1, ids[-1] + 4)
            idx16 = [int(round(x)) for x in np.linspace(a, max(a + 1, b), NUM_SAMPLE)]
        else:
            idx16 = sample_indices(st, en, fps, n)
        cid = f"{args.clip_id_prefix}{args.split}_{scenario}_{view}_{lab}"      # 1 clip / (scenario,view,phase)
        clip_path = out / "clips" / f"{cid}.mp4"
        bbox_path = out / "bbox" / f"{cid}.json"
        if clip_path.exists() and bbox_path.exists() and clip_path.stat().st_size >= 1024:  # tái dùng clip HỢP LỆ (>=1KB); file dở do bị kill sẽ được cắt lại
            built[clip_key] = (clip_path, bbox_path)
            opts, correct, ctext = make_options(qa, scenario, view, lab, aug)
            labs = {lab, LABEL_TO_NUM.get(lab) or lab}
            caption = next((cap_map[(scenario, view, l)] for l in labs
                            if (scenario, view, l) in cap_map), "") or next(
                (v for (s, _, l), v in cap_map.items() if s == scenario and l in labs), "")
            rows.append({
                "qa_id": f"{cid}_{len(rows)}", "benchmark_id": f"{cid}_{len(rows)}",
                "sample": scenario, "split": args.split, "selector": selector_of(qa.get("question", ""), lab),
                "sampling_policy": lab, "phase": lab,
                "clip_16": str(clip_path), "bbox_json": str(bbox_path),
                "question": qa.get("question", ""), "options": opts,
                "correct": correct, "correct_text": ctext,
                "view": view, "template_view": view, "caption": caption,
            })
            continue

        # cắt 16 frame -> mp4 384x384 (robust: bỏ qua video hỏng, không cho crash cả job)
        try:
            cap = cv2.VideoCapture(str(vid))
            vw = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (CROP, CROP))
            for fi in idx16:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if not ok or fr is None:
                    fr = np.zeros((CROP, CROP, 3), np.uint8)
                else:
                    fr = cv2.resize(fr, (CROP, CROP))
                vw.write(fr)
            cap.release(); vw.release()
        except Exception as e:
            print(f"  [skip] clip cut failed {vid}: {e}", flush=True)
            clip_path.unlink(missing_ok=True); continue
        if (not clip_path.exists()) or clip_path.stat().st_size < 1024:   # writer failed / empty clip
            print(f"  [skip] empty clip {clip_path.name} from {vid}", flush=True)
            clip_path.unlink(missing_ok=True); continue

        # bbox_json: 8 frame TAKE_IDX -> xyxy_norm theo W,H gốc (pbox/vbox đã lọc theo phase)
        def nearest(boxes, fi):
            if not boxes:
                return None
            k = min(boxes, key=lambda k: abs(k - fi))
            return boxes[k] if abs(k - fi) <= 8 else None
        frames = []
        for p in TAKE_IDX:
            fi = idx16[p]
            entry = {"sample_index": p, "pedestrian": [], "vehicle": []}
            for boxes, key in [(pbox, "pedestrian"), (vbox, "vehicle")]:
                b = nearest(boxes, fi)
                if b and W and H:
                    x, y, w, h = b
                    entry[key] = [{"xyxy_norm": [x / W, y / H, (x + w) / W, (y + h) / H]}]
            frames.append(entry)
        bbox_path = out / "bbox" / f"{cid}.json"
        json.dump({"frames": frames}, open(bbox_path, "w"))
        built[clip_key] = (clip_path, bbox_path)
        if len(built) % 50 == 0:
            print(f"  built {len(built)} clips ({len(rows)} QA)...", flush=True)

        opts, correct, ctext = make_options(qa, scenario, view, lab, aug)
        # caption: VQA dùng label TÊN (avoidance), caption dùng SỐ ('4') -> thử cả hai.
        labs = {lab, LABEL_TO_NUM.get(lab) or lab}
        caption = next((cap_map[(scenario, view, l)] for l in labs
                        if (scenario, view, l) in cap_map), "") or next(
            (v for (s, _, l), v in cap_map.items() if s == scenario and l in labs), "")
        rows.append({
            "qa_id": f"{cid}_{len(rows)}", "benchmark_id": f"{cid}_{len(rows)}",
            "sample": scenario, "split": args.split,
            "selector": selector_of(qa.get("question", ""), lab),
            "sampling_policy": lab, "phase": lab,
            "clip_16": str(clip_path), "bbox_json": str(bbox_path),
            "question": qa.get("question", ""), "options": opts,
            "correct": correct, "correct_text": ctext,
            "view": view, "template_view": view,
            "caption": caption,
        })

    mpath = out / "_review" / f"{args.manifest_prefix}_manifest_{args.split}.jsonl"
    with open(mpath, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE: {len(rows)} QA -> {mpath} (skipped {skipped} thiếu video)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Cut the frames and write the manifests the grounded caption variant runs on.

  --split train   SynWTS videos + annotations -> frames/{train,val}/ + manifests/{train,val}.jsonl
  --split test    WTS test videos + the VQA answers -> frames/test/ + manifests/test.jsonl

Frame choice is bbox-driven and identical in both splits, which is the point: for each
(scenario, phase) it picks the overhead camera whose pedestrian box is largest in that phase and
samples across the phase window, plus one dashcam frame. Where the test bbox is missing (about
half the scenarios) it falls back to the first camera and evenly spaced times. Frames are cut at
the same width in both splits, so training frames and test frames match in kind.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path
from statistics import median

from traffic_jepa.captioning.generate import build_facts, load_segment_times
from traffic_jepa.captioning_mm import build_data as BD

ROOT = Path(__file__).resolve().parents[2]
DEF_OUT = ROOT / "data" / "processed" / "caption_mm"
DEF_SEGMENTS = ROOT / "configs" / "caption" / "segments.json"
DEF_VQA = ROOT / "submissions" / "submission_final.json"
DEF_TEST = ROOT / "data" / "test" / "WTS_VQA_PUBLIC_TEST.json"


# --------------------------------------------------------------------------- ffmpeg
def extract(video, t, out_path: Path, width: int) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", str(max(0.0, float(t))), "-i", str(video),
                        "-frames:v", "1", "-vf", f"scale={width}:-1", "-q:v", "3", str(out_path)],
                       capture_output=True, timeout=60)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def even_times(window, n):
    s, e = window
    if e <= s:
        e = s + 1.0
    return [(s + e) / 2.0] if n <= 1 else [s + (e - s) * (i + 0.5) / n for i in range(n)]


def _area(a):
    b = a.get("bbox") or [0, 0, 0, 0]
    return float(b[2]) * float(b[3])


def _best_cam(cams: dict, phase: str):
    """The overhead camera whose pedestrian box is largest (median area) in this phase."""
    best, score = None, -1.0
    for cam, anns in cams.items():
        in_phase = [a for a in anns if str(a.get("phase_number")) == phase]
        if in_phase:
            m = median(sorted(_area(a) for a in in_phase))
            if m > score:
                best, score = cam, m
    return best


def _bbox_times(anns_in_phase, window, n):
    """image_id -> video time via the phase window (ids inside a phase are contiguous), else evenly."""
    if not anns_in_phase:
        return even_times(window, n)
    ids = sorted(a["image_id"] for a in anns_in_phase)
    first, last = ids[0], ids[-1]
    s, e = window
    span = max(1, last - first)
    picked = ([ids[len(ids) // 2]] if n <= 1
              else [ids[round(k * (len(ids) - 1) / (n - 1))] for k in range(n)])
    return [s + (i - first) / span * (e - s) for i in picked]


def _load_bboxes(directory: Path) -> dict:
    out = {}
    for bf in glob.glob(str(directory / "*_bbox.json")):
        cam = os.path.basename(bf).replace("_bbox.json", "")
        try:
            out[cam] = json.load(open(bf)).get("annotations", [])
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- SynWTS (train)
def _sim_bboxes(annot, split, scenario, view):
    return _load_bboxes(Path(annot) / "bbox_annotated" / "pedestrian" / split / scenario / view)


def _sim_video(videos, split, scenario, view, cam=None):
    base = Path(videos) / split / scenario / view
    if cam and (base / f"{cam}.mp4").exists():
        return base / f"{cam}.mp4"
    found = sorted(base.glob("*.mp4")) if base.is_dir() else []
    return found[0] if found else None


def sim_phase_windows(annot, split) -> dict:
    windows = {}
    files = glob.glob(str(Path(annot) / "caption" / split / "**" / "*caption*.json"), recursive=True)
    if not files:
        files = glob.glob(str(Path(annot) / "caption" / "**" / "*caption*.json"), recursive=True)
    if not files:
        print(f"[warn] no caption json under {Path(annot) / 'caption'} — check --annot", flush=True)
    for cf in files:
        scenario = os.path.basename(cf).replace("_caption.json", "")
        try:
            data = json.load(open(cf))
        except Exception:
            continue
        for ev in data.get("event_phase", []):
            phase = str((ev.get("labels") or ["?"])[0])
            start = ev.get("start_time")
            if start is None:
                continue
            try:
                windows.setdefault((scenario, phase), (float(start), float(ev.get("end_time", start))))
            except (TypeError, ValueError):
                pass
    return windows


def build_sim_frames(annot, videos, split, n_overhead, n_vehicle, width, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = sim_phase_windows(annot, split)
    by_scenario: dict = {}
    for (scenario, phase) in windows:
        by_scenario.setdefault(scenario, []).append(phase)

    index, done = {}, 0
    for scenario, phases in by_scenario.items():
        cams = _sim_bboxes(annot, split, scenario, "overhead_view")
        veh_cams = _sim_bboxes(annot, split, scenario, "vehicle_view")
        veh_anns = next(iter(veh_cams.values()), [])
        for phase in phases:
            window = windows[(scenario, phase)]
            paths = []
            cam = _best_cam(cams, phase)
            overhead = _sim_video(videos, split, scenario, "overhead_view", cam)
            if overhead:
                in_phase = [a for a in cams.get(cam, []) if str(a.get("phase_number")) == phase]
                for i, t in enumerate(_bbox_times(in_phase, window, n_overhead)):
                    p = out_dir / f"{scenario}__{phase}__ov{i}.jpg"
                    if extract(overhead, t, p, width):
                        paths.append(str(p))
            vehicle = _sim_video(videos, split, scenario, "vehicle_view")
            if vehicle:
                in_phase = [a for a in veh_anns if str(a.get("phase_number")) == phase]
                for i, t in enumerate(_bbox_times(in_phase, window, n_vehicle)):
                    p = out_dir / f"{scenario}__{phase}__veh{i}.jpg"
                    if extract(vehicle, t, p, width):
                        paths.append(str(p))
            if paths:
                index[f"{scenario}||{phase}"] = paths
            done += 1
            if done % 100 == 0:
                print(f"  {split}: {done} segments", flush=True)
    return index


# --------------------------------------------------------------------------- WTS test
def _test_bboxes(bbox_root, scenario, view) -> dict:
    if not bbox_root:
        return {}
    for split in ("test/public", "test/public/normal_trimmed"):
        d = Path(bbox_root) / "bbox_annotated" / "pedestrian" / split / scenario / view
        if d.is_dir():
            return _load_bboxes(d)
    return {}


def _test_video(wts_root, scenario, view, cam=None):
    base = Path(wts_root) / "videos" / "test" / "public"
    for parent in (base, base / "normal_trimmed"):
        d = parent / scenario / view
        if not d.is_dir():
            continue
        if cam and (d / f"{cam}.mp4").exists():
            return d / f"{cam}.mp4"
        found = sorted(d.glob("*.mp4"))
        if found:
            return found[0]
    return None


def build_test_frames(wts_root, bbox_root, n_overhead, n_vehicle, width, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = load_segment_times(wts_root)
    if not segments:
        print(f"[warn] no segment times under {wts_root}/annotations/caption/test/public_challenge", flush=True)
    index, n_bbox, n_fallback, done = {}, 0, 0, 0
    cache = {}
    for (scenario, phase), window in segments.items():
        if scenario not in cache:
            cache[scenario] = (_test_bboxes(bbox_root, scenario, "overhead_view"),
                               _test_bboxes(bbox_root, scenario, "vehicle_view"))
        cams, veh_cams = cache[scenario]
        paths = []

        cam = _best_cam(cams, phase)                       # None when this phase has no bbox
        overhead = _test_video(wts_root, scenario, "overhead_view", cam)
        if overhead:
            in_phase = [a for a in cams.get(cam, []) if str(a.get("phase_number")) == phase] if cam else []
            times = _bbox_times(in_phase, window, n_overhead) if in_phase else even_times(window, n_overhead)
            n_bbox += 1 if in_phase else 0
            n_fallback += 0 if in_phase else 1
            for i, t in enumerate(times):
                p = out_dir / f"{scenario}__{phase}__ov{i}.jpg"
                if extract(overhead, t, p, width):
                    paths.append(str(p))

        vehicle = _test_video(wts_root, scenario, "vehicle_view")
        if vehicle:
            veh_anns = next(iter(veh_cams.values()), [])
            in_phase = [a for a in veh_anns if str(a.get("phase_number")) == phase]
            times = _bbox_times(in_phase, window, n_vehicle) if in_phase else even_times(window, n_vehicle)
            for i, t in enumerate(times):
                p = out_dir / f"{scenario}__{phase}__veh{i}.jpg"
                if extract(vehicle, t, p, width):
                    paths.append(str(p))

        if paths:
            index[f"{scenario}||{phase}"] = paths
        done += 1
        if done % 100 == 0:
            print(f"  test: {done} segments (bbox-selected {n_bbox}, fallback {n_fallback})", flush=True)
    print(f"  test frames: bbox-selected camera for {n_bbox} segments, fallback for {n_fallback}", flush=True)
    return index


# --------------------------------------------------------------------------- writers
def write_train(args, out: Path):
    manifests = out / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        index = build_sim_frames(args.annot, args.videos, split, args.n_overhead, args.n_vehicle,
                                 args.width, out / "frames" / split)
        if not index:
            print(f"[warn] {split}: 0 frames — check --videos ({args.videos}); it expects "
                  f"<split>/<scenario>/{{overhead_view,vehicle_view}}/*.mp4, and ffmpeg on PATH", flush=True)
        rows = BD.build_split(args.annot, split, frames_index=index)
        path = manifests / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows ({sum(1 for r in rows if r['images'])} with frames) -> {path}")


def write_test(args, out: Path):
    manifests = out / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for path in (args.vqa, args.test, args.segments):
        if not Path(path).is_file():
            raise SystemExit(f"FATAL: input not found: {path}")
    facts = build_facts(args.vqa, args.test)
    segments = json.load(open(args.segments))
    index = (build_test_frames(args.wts_root, args.bbox_root, args.n_overhead, args.n_vehicle,
                               args.width, out / "frames" / "test") if args.wts_root else {})
    if not args.wts_root:
        print("[warn] --wts-root not set -> a text-only manifest (no frames). The grounded variant "
              "needs frames; without them it degenerates to the default stage.", flush=True)

    path = manifests / "test.jsonl"
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for scenario, entries in segments.items():
            for entry in entries:
                phase = str((entry.get("labels") or ["?"])[0])
                fh.write(json.dumps({
                    "scenario": scenario,
                    "phase": phase,
                    "labels": entry.get("labels", [phase]),
                    "facts": facts.get((scenario, phase), {}),
                    "images": index.get(f"{scenario}||{phase}", []),
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"test: {n} segments ({len(index)} with frames) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", default=str(DEF_OUT), help="dir for frames/ and manifests/")
    # train
    ap.add_argument("--annot", default=os.getenv("SYNWTS_ANNOT", "data/synwts/data/annotations"),
                    help="[train] SynWTS annotations (holds vqa/, caption/, bbox_annotated/)")
    ap.add_argument("--videos", default=os.getenv("SYNWTS_VIDEOS", "data/synwts/data/videos"),
                    help="[train] SynWTS videos")
    # test
    ap.add_argument("--wts-root", dest="wts_root", default=os.getenv("WTS_ROOT", ""),
                    help="[test] WTS test root with annotations/ + videos/")
    ap.add_argument("--bbox-root", dest="bbox_root", default=os.getenv("TEST_BBOX", ""),
                    help="[test] dir holding bbox_annotated/ for the test set")
    ap.add_argument("--vqa", default=str(DEF_VQA), help="[test] VQA submission [{id, correct}]")
    ap.add_argument("--test", default=str(DEF_TEST), help="[test] public-test question file")
    ap.add_argument("--segments", default=str(DEF_SEGMENTS), help="[test] scenarios/segments to caption")
    # frame shape — must match --max_images at training and generation
    ap.add_argument("--n_overhead", type=int, default=3, help="overhead frames per (scenario, phase)")
    ap.add_argument("--n_vehicle", type=int, default=1, help="dashcam frames per (scenario, phase)")
    ap.add_argument("--width", type=int, default=1024, help="frame width in px")
    args = ap.parse_args()

    out = Path(args.out)
    if args.split == "train":
        write_train(args, out)
    else:
        write_test(args, out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Grounded captioning: manifest (facts + frames) -> Qwen3-VL-8B (+ LoRA) -> caption submission.

  data/processed/caption_mm/manifests/test.jsonl   (from traffic_jepa.captioning_mm.preprocess)
      -> Qwen3-VL-8B + checkpoints/caption_lora_mm
      -> submissions/caption_submission_mm.json

Per segment:
  facts present -> the caption is written FROM THE FACTS, with the frames shown as secondary
                   grounding (QA_IMG_NOTE). This is the exact call the adapter was trained on, so
                   real frames are used the same way simulation frames were during training.
  no facts      -> the frames are described directly.

The model is loaded in-process — there is no server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from traffic_jepa.captioning import generate as G
from traffic_jepa.captioning.backend import QwenVL, image_msg, text_msg

ROOT = Path(__file__).resolve().parents[2]
DEF_MANIFEST = ROOT / "data" / "processed" / "caption_mm" / "manifests" / "test.jsonl"
DEF_LORA = ROOT / "checkpoints" / "caption_lora_mm"
DEF_OUT = ROOT / "submissions" / "caption_submission_mm.json"

# The frames are simulation at training time and real at inference time. Without this note the
# model reads attributes off the pixels, which is exactly what fails to transfer; with it, the
# frames only shape the spatial wording and every attribute still comes from the VQA answers.
QA_IMG_NOTE = (
    "HOW TO USE THE IMAGES (read carefully). This segment HAS verified answers, so you must write BOTH captions "
    "FROM THE VALUES LISTED BELOW - they are the ground truth and the ONLY source for every attribute. DO NOT "
    "READ ANY ATTRIBUTE OFF THE FRAMES. The few reference frames shown (overhead and/or dashcam, two angles of "
    "the SAME event) are there only so your spatial wording sounds natural; they are NOT evidence. Only segments "
    "that have NO answers at all are described from the image - this is not one of them. So: if the frames seem "
    "to disagree with a value, ignore the frames and follow the value; never add, drop or change an attribute "
    "because of what a frame appears to show; never invent an attribute that is not in the given values. When in "
    "doubt, the values win.")

FRAME_USER = ("These are several frames from ONE short traffic clip (overhead frames first, then a dashcam frame). "
              "A target pedestrian IS present even if small, far or partly occluded; describe the pedestrian and "
              "the vehicle. Never say the pedestrian is absent. Return strict JSON only.")


def _resolve(p, root: Path) -> Path | None:
    """preprocess writes absolute paths. A relative one is resolved against the manifest dir or its
    parent, so a manifest copied in from elsewhere still finds its frames."""
    p = Path(p)
    if p.is_absolute():
        return p if p.is_file() else None
    for base in (root, root.parent):
        cand = base / p
        if cand.is_file():
            return cand
    return None


def load_images(paths, root: Path):
    from PIL import Image
    out = []
    for p in paths or []:
        fp = _resolve(p, root)
        if fp is None:
            continue
        try:
            img = Image.open(fp)
            img.load()
            out.append(img.convert("RGB"))
        except Exception:
            continue
    return out


def qa_caption(runner, scenario, phase, facts, images, shots, temperature, rounds=4):
    """Same coverage loop as the default stage, with the frames attached to the user turn."""
    pv, vv = G.ped_values(facts), G.veh_values(facts)
    prompt = G.build_prompt(scenario, phase, facts, shots)
    if images:
        user = image_msg("user", QA_IMG_NOTE + "\n\n" + prompt, images)
    else:
        user = text_msg("user", prompt)
    msgs = [text_msg("system", G.SYS), user]
    cap = best = None
    best_missing = 1 << 30
    for _ in range(rounds):
        try:
            cap = G.parse_caps(runner.chat(msgs, temperature=temperature))
        except Exception as e:
            print(f"  generate retry {scenario} p{phase} ({e})", flush=True)
            continue
        if not cap:
            continue
        cap["caption_pedestrian"] = G.mild_clean(cap["caption_pedestrian"])
        cap["caption_vehicle"] = G.mild_clean(cap["caption_vehicle"])
        mp = G.missing_values(cap["caption_pedestrian"], pv)
        mv = G.missing_values(cap["caption_vehicle"], vv)
        if len(mp) + len(mv) < best_missing:
            best, best_missing = cap, len(mp) + len(mv)
        if not mp and not mv:
            return cap
        msgs = [text_msg("system", G.SYS), user, text_msg("assistant", json.dumps(cap)),
                text_msg("user",
                         "Some required values are missing. Rewrite BOTH captions (about 125 words each, strict "
                         "JSON) so that EVERY value appears verbatim, WEAVING each missing value into a real "
                         "sentence. Missing from the pedestrian caption: " + ("; ".join(mp) or "none") +
                         ". Missing from the vehicle caption: " + ("; ".join(mv) or "none") + ".")]
    cap = best or cap or G.template_caption(facts)
    cap["caption_pedestrian"] = G.mild_clean(cap["caption_pedestrian"])
    cap["caption_vehicle"] = G.mild_clean(cap["caption_vehicle"])
    return cap


def frame_caption(runner, scenario, phase, images):
    """No facts: describe the frames. Unlike the default stage the adapter stays on — this one was
    trained with images, so its weights are not text-only."""
    if not images:
        return dict(G.GENERIC_FRAME)
    msgs = [text_msg("system", G.FRAME_SYS), image_msg("user", FRAME_USER, images)]
    for attempt in range(3):
        try:
            cap = G.parse_caps(runner.chat(msgs, max_new_tokens=600, temperature=0.3 + 0.2 * attempt))
        except Exception as e:
            print(f"  frame retry {scenario} p{phase} ({e})", flush=True)
            continue
        if cap and not G.is_degenerate(cap):
            return cap
    return dict(G.GENERIC_FRAME)


def check_inputs(args, rows) -> int:
    missing = 0
    print("== inputs ==")
    ok = Path(args.manifest).is_file()
    missing += int(not ok)
    print(f"  [{'ok' if ok else 'MISSING'}] {'test manifest':22s} {args.manifest}")
    if not ok:
        print("            build it: python -m traffic_jepa.captioning_mm.preprocess --split test")
    ok = (Path(args.lora) / "adapter_config.json").is_file()
    missing += int(not ok)
    print(f"  [{'ok' if ok else 'MISSING'}] {'grounded LoRA':22s} {args.lora}")

    if rows:
        n_facts = sum(1 for r in rows if r.get("facts"))
        n_img = sum(1 for r in rows if r.get("images"))
        root = Path(args.manifest).resolve().parent
        n_ondisk = sum(1 for r in rows for p in (r.get("images") or []) if _resolve(p, root))
        print(f"\n== coverage ==\n  segments={len(rows)}  with facts={n_facts}  with frames={n_img}"
              f"  frame files on disk={n_ondisk}")
        if n_img and not n_ondisk:
            print("  [MISSING] the manifest lists frames but none exist -> re-run preprocess")
            missing += 1

    print("\n== python packages ==")
    for mod in ("torch", "transformers", "peft", "PIL"):
        try:
            __import__(mod)
            print(f"  [ok] {mod}")
        except ImportError:
            missing += 1
            print(f"  [MISSING] {mod}  -> pip install -r requirements.txt")

    print("\n" + ("READY" if missing == 0 else f"NOT READY: {missing} item(s) missing"))
    return missing


def main():
    ap = argparse.ArgumentParser(description="Grounded captioning: facts + frames -> captions.")
    ap.add_argument("--manifest", default=str(DEF_MANIFEST))
    ap.add_argument("--model", default=G.BASE_MODEL)
    ap.add_argument("--lora", default=str(DEF_LORA), help="grounded LoRA dir; empty string = base model")
    ap.add_argument("--fewshot", default=str(G.DEF_FEWSHOT))
    ap.add_argument("--fewshot_n", type=int, default=0, help="style examples; the adapter needs none")
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--cache", default="", help="resume file (default: <out>_cache.jsonl)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--max_images", type=int, default=4, help="frames per segment; must match training")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="verify the inputs and exit")
    args = ap.parse_args()

    rows = ([json.loads(l) for l in open(args.manifest) if l.strip()]
            if Path(args.manifest).is_file() else [])
    if args.check:
        sys.exit(1 if check_inputs(args, rows) else 0)
    if not rows:
        raise SystemExit(f"FATAL: no manifest at {args.manifest}\n"
                         f"       build it: python -m traffic_jepa.captioning_mm.preprocess --split test")
    if args.limit:
        rows = rows[:args.limit]

    manifest_root = Path(args.manifest).resolve().parent
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out_path.with_name(out_path.stem + "_cache.jsonl")
    done = {}
    if cache_path.exists():
        for line in cache_path.open():
            try:
                r = json.loads(line)
                done[(r["scenario"], r["phase"])] = r["cap"]
            except Exception:
                pass
        print(f"resuming: {len(done)} segments already in {cache_path}", flush=True)

    todo = [r for r in rows if (r["scenario"], r["phase"]) not in done]
    n_frame = sum(1 for r in todo if not r.get("facts"))
    print(f"model={args.model}{' + ' + args.lora if args.lora else ' (base, no adapter)'}", flush=True)
    print(f"segments={len(rows)} to generate={len(todo)} (facts={len(todo) - n_frame} frame-only={n_frame})",
          flush=True)

    if todo:
        if args.seed:
            import torch
            torch.manual_seed(args.seed)
        runner = QwenVL(args.model, args.lora, dtype=args.dtype, device_map=args.device_map,
                        max_new_tokens=args.max_new_tokens)
        shots = G.load_style_examples(args.fewshot, args.fewshot_n)
        try:
            from tqdm import tqdm
            bar = tqdm(todo, desc="captioning (grounded)")
        except ImportError:
            bar = todo
        for i, row in enumerate(bar, 1):
            scenario, phase = row["scenario"], row["phase"]
            facts = row.get("facts") or {}
            images = load_images((row.get("images") or [])[:args.max_images], manifest_root)
            try:
                if facts:
                    cap = qa_caption(runner, scenario, phase, facts, images, shots, args.temperature)
                else:
                    cap = frame_caption(runner, scenario, phase, images)
            except Exception as e:
                print(f"  segment error {scenario} p{phase}: {e}", flush=True)
                cap = None
            if not cap or not (cap.get("caption_pedestrian") and cap.get("caption_vehicle")):
                cap = G.template_caption(facts)
            done[(scenario, phase)] = cap
            with cache_path.open("a") as fh:
                fh.write(json.dumps({"scenario": scenario, "phase": phase, "cap": cap}) + "\n")
            if bar is todo and i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    submission = {}
    for row in rows:
        cap = done.get((row["scenario"], row["phase"]))
        if not cap:
            continue
        submission.setdefault(row["scenario"], []).append({
            "labels": row.get("labels", [row["phase"]]),
            "caption_pedestrian": G.env_fix(cap["caption_pedestrian"]),
            "caption_vehicle": G.env_fix(cap["caption_vehicle"]),
        })

    json.dump(submission, out_path.open("w"), indent=1, ensure_ascii=True)
    n_seg = sum(len(v) for v in submission.values())
    empty = sum(1 for v in submission.values() for e in v
                if not e["caption_pedestrian"] or not e["caption_vehicle"])
    print(f"wrote {out_path} | scenarios={len(submission)} segments={n_seg} empty={empty}", flush=True)
    if empty:
        raise SystemExit(f"FATAL: {empty} segments have an empty caption")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Precompute & cache frozen-encoder embeddings for VL-JEPA WTS Sim2Real VQA.

What it caches (the two EXPENSIVE frozen encoders), so training just loads vectors:
  1) V-JEPA 2.1 ViT-L/16 @384 latent per UNIQUE clip:
       read clip_16 (16f) -> take 8 frames (stride 2: 0,2,..,14) -> [1,3,8,384,384]
       -> encoder -> raw tokens [4,24,24,1024] (fp16)            -> latents/<clip_id>.npy
  2) EmbeddingGemma-300m pooled vector (768) per UNIQUE text (targets + 4 options):
       mean-pool(last_hidden_state)  (matches YEncoder, minus the learnable projection)
                                                                 -> text_vecs.safetensors
Also writes:
  clips_meta.json : per-clip ped/veh boxes for the 8 frames, mapped into the 384-crop space
  index_sim.jsonl : one row per QA, all of it simulation
  meta.json       : encoder ids, dims, frame policy, counts
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

import numpy as np
import torch
import decord
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                            TextColumn, TimeElapsedColumn, TimeRemainingColumn)
from safetensors.torch import save_file

console = Console()
# Portable: ROOT = this clean reproduce repo, resolved relative to this script.
# Data lives under repo/data/.
# Override with $VLJEPA_ROOT / $VLJEPA_DATA / $VLJEPA_OUT.
ROOT = Path(os.getenv("VLJEPA_ROOT", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.getenv("VLJEPA_DATA", str(ROOT / "data")))
OUT = Path(os.getenv("VLJEPA_OUT", str(DATA / "processed/cache_vljepa16_8f")))

MANIFESTS = {
    "sim": [
        DATA / "processed/sim_qa_vljepa16/_review/vjepa16_manifest_train.jsonl",
        DATA / "processed/sim_qa_vljepa16/_review/vjepa16_manifest_val.jsonl",
    ],
}

TAKE_IDX = [0, 2, 4, 6, 8, 10, 12, 14]   # 8 of 16, stride 2 (keep full temporal span)
CROP = 384
SHORT = round(CROP * 256 / 224)          # 438, V-JEPA eval resize of short side
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
VJEPA_ENTRY = "vjepa2_1_vit_large_384"
GEMMA_ID = "google/embeddinggemma-300m"


def progress():
    return Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TextColumn("•"),
        TimeElapsedColumn(), TextColumn("eta"), TimeRemainingColumn(),
        TextColumn("• {task.fields[rate]}"), console=console,
    )


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def clip_id_of(rel_clip_path: str) -> str:
    return Path(rel_clip_path).stem


def norm_row(row: dict, domain: str) -> dict:
    """Normalise a manifest row into the schema the cache index uses."""
    options = [o for o in (row.get("options") or row.get("options_template") or []) if o]
    has_label = row.get("correct") is not None
    letter = (row.get("correct") or "a").strip().lower()
    correct_index = "abcd".index(letter) if letter in "abcd" else 0
    correct_index = min(correct_index, max(len(options) - 1, 0))
    correct_text = row.get("correct_text") or (options[correct_index] if options else "")
    return {
        "qa_id": row.get("qa_id", row.get("benchmark_id")),
        "domain": domain,
        "sample": row.get("sample"),
        "split": row.get("split", "val"),
        "category": row.get("selector"),
        "sampling_policy": row.get("sampling_policy"),
        "clip_id": clip_id_of(row["clip_16"]),
        "clip_path": str(ROOT / row["clip_16"]),
        "bbox_json": str(ROOT / row["bbox_json"]),
        "question": row.get("question", ""),
        "options": options,
        "correct_index": correct_index, "has_label": has_label,
        "correct_text": correct_text,
        "caption": (row.get("caption") or row.get("caption_pedestrian") or "").strip(),
        "view": row.get("view", row.get("template_view")),
        "requested_view": row.get("requested_view", row.get("view", row.get("template_view"))),
        "source_view": row.get("source_view", row.get("view", row.get("template_view"))),
        "phase": row.get("phase", row.get("phase_label")),
    }


def gather():
    """Every row comes from the simulation manifests written by build_manifest."""
    sim_rows = []
    for f in MANIFESTS["sim"]:
        if not f.exists():
            continue
        for line in open(f):
            if line.strip():
                sim_rows.append(norm_row(json.loads(line), "sim"))

    rows = {"sim": sim_rows}
    clips, texts = {}, set()
    for domain in rows:
        for r in rows[domain]:
            clips.setdefault(r["clip_id"], {"clip_path": r["clip_path"], "bbox_json": r["bbox_json"]})
            texts.update(r["options"])
            if r["correct_text"]:
                texts.add(r["correct_text"])
            if r.get("caption"):
                texts.add(r["caption"])
    return rows, clips, sorted(t for t in texts if t)


# Frames are STRETCH-resized to 384x384 (no crop) so off-centre pedestrians/vehicles in the
# wide overhead view are never lost. Normalized boxes are therefore unchanged (identity map).
def transform_box(xyxy_norm, W, H):
    x1, y1, x2, y2 = (float(np.clip(v, 0.0, 1.0)) for v in xyxy_norm)
    valid = (x2 - x1) > 1e-3 and (y2 - y1) > 1e-3
    return [x1, y1, x2, y2], valid


def boxes_for_clip(bbox_json_path, W, H):
    data = json.load(open(bbox_json_path))
    frames = {fr["sample_index"]: fr for fr in data["frames"]}
    out = {"ped": [], "veh": [], "valid_ped": [], "valid_veh": []}
    for si in TAKE_IDX:
        fr = frames.get(si, {})
        for kind, key in [("pedestrian", "ped"), ("vehicle", "veh")]:
            boxes = fr.get(kind) or []
            if boxes:
                b, v = transform_box(boxes[0]["xyxy_norm"], W, H)
            else:
                b, v = [0.0, 0.0, 0.0, 0.0], False
            out[key].append(b)
            out["valid_" + key].append(bool(v))
    return out


# ---------------------------- phase 1: V-JEPA latents ----------------------------
def load_vjepa(device):
    # keep fp32: V-JEPA 2.1 RoPE buffers stay float32 -> fp16 forward dtype-mismatches in SDPA.
    enc, _ = torch.hub.load("facebookresearch/vjepa2", VJEPA_ENTRY, trust_repo=True, pretrained=True)
    return enc.to(device).float().eval()


def read_clip_8(path):
    vr = decord.VideoReader(path, num_threads=4)
    n = len(vr)
    idx = TAKE_IDX if n >= 16 else np.linspace(0, n - 1, 8).round().astype(int).tolist()
    frames = vr.get_batch(idx).asnumpy()                       # (8,H,W,3) uint8
    H, W = frames.shape[1:3]
    x = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 255.0   # (3,8,H,W)
    x = torch.nn.functional.interpolate(x, size=(CROP, CROP),  # STRETCH to 384x384 (keep all content)
                                        mode="bilinear", align_corners=False)
    x = (x - MEAN) / STD
    return x.unsqueeze(0).float(), (W, H)                      # (1,3,8,384,384) fp32 in, fp16 saved


@torch.no_grad()
def phase_latents(clips, device, limit=None):
    (OUT / "latents").mkdir(parents=True, exist_ok=True)
    enc = None                                   # lazy: only load V-JEPA if something needs encoding
    items = list(clips.items())[: limit or None]
    clips_meta = {}
    # reuse W,H of already-cached latents (transform_box ignores W,H -> only the stored "wh" needs it)
    old_meta_path = OUT / "clips_meta.json"
    old_wh = {}
    if old_meta_path.exists():
        old_wh = {k: v.get("wh") for k, v in json.load(open(old_meta_path)).items()}
    n_enc = 0
    with progress() as p:
        t = p.add_task("V-JEPA 2.1 latents", total=len(items), rate="")
        for i, (cid, info) in enumerate(items):
            dst = OUT / "latents" / f"{cid}.npy"
            if dst.exists():                     # SKIP re-reading the video for cached latents
                wh = old_wh.get(cid)
                if wh:
                    W, H = wh
                else:
                    import cv2
                    cap = cv2.VideoCapture(info["clip_path"])
                    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    cap.release()
            else:
                if enc is None:
                    enc = load_vjepa(device)
                x, (W, H) = read_clip_8(info["clip_path"])
                out = enc(x.to(device))
                tok = (out[0] if isinstance(out, (list, tuple)) else out)[0]   # [N,1024]
                side = round((tok.shape[0] / (len(TAKE_IDX) // 2)) ** 0.5)      # 24
                T = tok.shape[0] // (side * side)
                grid = tok.reshape(T, side, side, tok.shape[-1]).float().cpu().numpy().astype(np.float16)
                np.save(dst, grid)
                n_enc += 1
            clips_meta[cid] = {"wh": [W, H], "frame_take": TAKE_IDX,
                               **boxes_for_clip(info["bbox_json"], W, H)}
            if i % 20 == 0:
                p.update(t, rate=f"{n_enc} enc / {(i+1)/max(p.tasks[t].elapsed,1e-6):.1f}/s")
            p.advance(t)
    json.dump(clips_meta, open(OUT / "clips_meta.json", "w"))
    del enc
    torch.cuda.empty_cache()
    console.print(f"[green]✓[/] latents: {len(items)} clips -> {OUT/'latents'}")


# ---------------------------- phase 2: Gemma text vecs ----------------------------
@torch.no_grad()
def phase_text(texts, device, limit=None):
    from transformers import AutoModel, AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(GEMMA_ID, token=token)
    # fp32: EmbeddingGemma overflows -> NaN in fp16. Compute fp32, store fp16.
    model = AutoModel.from_pretrained(GEMMA_ID, token=token, torch_dtype=torch.float32).to(device).eval()
    texts = texts[: limit or None]
    vecs, mapping = {}, {}
    with progress() as p:
        t = p.add_task("EmbeddingGemma texts", total=len(texts), rate="")
        for i, text in enumerate(texts):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hid = getattr(out, "last_hidden_state", None)
            if hid is None:
                hid = out.hidden_states[-1]
            m = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
            pooled = (hid * m).sum(1) / m.sum(1).clamp_min(1.0)               # [1,768]
            key = sha(text)
            vecs[key] = pooled[0].half().cpu()
            mapping[text] = key
            if i % 20 == 0:
                p.update(t, rate=f"{(i+1)/max(p.tasks[t].elapsed,1e-6):.1f}/s")
            p.advance(t)
    save_file(vecs, str(OUT / "text_vecs.safetensors"))
    json.dump(mapping, open(OUT / "text_map.json", "w"))
    del model
    torch.cuda.empty_cache()
    console.print(f"[green]✓[/] text vecs: {len(vecs)} unique -> {OUT/'text_vecs.safetensors'}")


# ------------------------------- phase 3: index -------------------------------
def group_id_of(domain, row):
    return f"{domain}_{row.get('split', '')}_{row.get('sample', '')}"


def phase_group_context(rows):
    out_dir = OUT / "group_context"
    out_dir.mkdir(parents=True, exist_ok=True)
    for domain, recs in rows.items():
        groups = {}
        for r in recs:
            groups.setdefault(group_id_of(domain, r), set()).add(r["clip_id"])
        for gid, clip_ids in groups.items():
            toks = []
            for cid in sorted(clip_ids):
                path = OUT / "latents" / f"{cid}.npy"
                if not path.is_file():
                    continue
                arr = np.load(path)
                if arr.ndim == 4 and arr.shape[-1] == 1024:
                    toks.append(arr.mean(axis=(1, 2)))  # [T, 1024]
            if toks:
                ctx = np.stack(toks, axis=0).mean(axis=0).astype("float16")
            else:
                ctx = np.zeros((len(TAKE_IDX) // 2, 1024), dtype="float16")
            np.save(out_dir / f"{gid}.npy", ctx)
        console.print(f"[green]✓[/] group_context_{domain}: {len(groups)} groups -> {out_dir}")


def phase_index(rows):
    for domain, recs in rows.items():
        path = OUT / f"index_{domain}.jsonl"
        with open(path, "w") as fh:
            for r in recs:
                fh.write(json.dumps({
                    "qa_id": r["qa_id"], "domain": domain, "split": r["split"],
                    "sample": r.get("sample", ""),
                    "group_id": group_id_of(domain, r),
                    "category": r["category"], "sampling_policy": r["sampling_policy"],
                    "clip_id": r["clip_id"], "view": r["view"],
                    "requested_view": r.get("requested_view", r["view"]),
                    "source_view": r.get("source_view", r["view"]),
                    "phase": r["phase"],
                    "question": r["question"], "options": r["options"],
                    "option_hash": [sha(o) for o in r["options"]],
                    "correct_index": r["correct_index"], "has_label": r["has_label"],
                    "target_hash": sha(r["correct_text"]),
                    "caption_hash": sha(r["caption"]) if r.get("caption") else "",
                }) + "\n")
        console.print(f"[green]✓[/] index_{domain}: {len(recs)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "latents", "text", "index", "group_context"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap clips/texts (smoke test)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.rule("[bold]VL-JEPA preprocessing (8-frame, fp16)")
    rows, clips, texts = gather()
    console.print(f"sim QA={len(rows['sim'])}  unique clips={len(clips)}  "
                  f"unique texts={len(texts)}  device={device}")
    if args.phase in ("all", "latents"):
        phase_latents(clips, device, args.limit)
    if args.phase in ("all", "text"):
        phase_text(texts, device, args.limit)
    if args.phase in ("all", "group_context"):
        phase_group_context(rows)
    if args.phase in ("all", "index"):
        phase_index(rows)
    json.dump({"vjepa": VJEPA_ENTRY, "gemma": GEMMA_ID, "frames": 8, "take_idx": TAKE_IDX,
               "crop": CROP, "latent_shape": [4, 24, 24, 1024], "text_dim": 768,
               "sim_qa": len(rows["sim"]),
               "unique_clips": len(clips), "unique_texts": len(texts)},
              open(OUT / "meta.json", "w"), indent=2)
    console.rule("[bold green]done")
    console.print(f"cache -> {OUT}")


if __name__ == "__main__":
    main()

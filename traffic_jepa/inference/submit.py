#!/usr/bin/env python
"""End-to-end public-test inference.

This is the public entrypoint for submission generation:

  test videos/questions -> Traffic-JEPA model scores -> graph/temporal postprocess -> submission JSON

`predict.py` remains the model-scoring backend. The graph/temporal decoder lives under
`traffic_jepa.postprocess` because it is a deterministic score-level postprocess, not a
validation metric.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--bbox-root", required=True)
    ap.add_argument("--videos-root", required=True)
    ap.add_argument("--sim_index", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--llama", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--gemma", default="google/embeddinggemma-300m")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--raw_out", default="submissions/submission_raw.json")
    ap.add_argument("--out", default="submissions/submission_final.json")
    args, decode_args = ap.parse_known_args()

    raw_out = Path(args.raw_out)
    scored = raw_out.with_name(raw_out.stem + "_scored.jsonl")
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    predict_cmd = [
        sys.executable, "-m", "traffic_jepa.inference.predict",
        "--test", args.test,
        "--route", args.route,
        "--bbox-root", args.bbox_root,
        "--videos-root", args.videos_root,
        "--ckpt", args.ckpt,
        "--llama", args.llama,
        "--gemma", args.gemma,
        "--out", str(raw_out),
        "--batch_size", str(args.batch_size),
        "--run",
    ]
    decode_cmd = [
        sys.executable, "-m", "traffic_jepa.postprocess.decode_test",
        "--test", args.test,
        "--route", args.route,
        "--bbox-root", args.bbox_root,
        "--videos-root", args.videos_root,
        "--sim_index", args.sim_index,
        "--scored", str(scored),
        "--out", args.out,
        *decode_args,
    ]

    # A failing stage already printed its own reason; re-raising here would bury it under a
    # traceback, so exit with the child's code instead.
    print("== model inference on public test ==", flush=True)
    rc = subprocess.run(predict_cmd).returncode
    if rc:
        raise SystemExit(rc)
    print("== postprocess decode -> final submission ==", flush=True)
    rc = subprocess.run(decode_cmd).returncode
    if rc:
        raise SystemExit(rc)
    print(f"== submission -> {args.out} ==", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# cut 16-frame clips + bbox_json + sim manifests from raw SynWTS.  (~30 min, CPU)
source "$(dirname "$0")/env.sh"
for sp in train val; do
  echo "== build sim manifest: $sp =="
  $PY -m traffic_jepa.data.build_manifest \
    --annot "$SYNWTS/annotations" --videos "$SYNWTS/videos" \
    --out "$SIMQA" --split "$sp"
done
echo "== sim manifest done: $(wc -l < "$SIMQA/_review/vjepa16_manifest_train.jsonl") + $(wc -l < "$SIMQA/_review/vjepa16_manifest_val.jsonl") QA =="

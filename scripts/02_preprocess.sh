#!/usr/bin/env bash
# encode V-JEPA 2 latents (per unique clip) + EmbeddingGemma text vecs -> cache.  (~1 h, GPU)
# Input:  manifests and sampled clips written by 01_manifest_sim.sh
# Output: $CACHE/{latents,text_vecs.safetensors,index_sim.jsonl,...}
source "$(dirname "$0")/env.sh"
echo "== preprocess -> $CACHE =="
$PY -m traffic_jepa.data.preprocess --phase all
echo "== preprocess done: latents=$(ls "$CACHE/latents" | wc -l), index_sim=$(wc -l < "$CACHE/index_sim.jsonl") =="

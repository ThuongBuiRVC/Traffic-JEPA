#!/usr/bin/env bash
# Full reproduce, raw data -> submission: manifests, preprocess, train, submit.
# Expects the WTS data placed under data/ (see README section 3) and HF_TOKEN set.
set -euo pipefail
here="$(dirname "$0")"
bash "$here/01_manifest_sim.sh"
bash "$here/02_manifest_real.sh"
bash "$here/03_preprocess.sh"
bash "$here/04_train.sh"
bash "$here/05_submit_test.sh"
echo "== full pipeline done -> submissions/submission_final.json =="

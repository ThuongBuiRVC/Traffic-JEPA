#!/usr/bin/env bash
# cut real clips + real manifests (sim QA paired onto matching real videos).  (~20 min, CPU)
source "$(dirname "$0")/env.sh"
echo "== build real manifest (all splits) =="
$PY -m traffic_jepa.data.build_real_manifest \
  --raw "$WOVEN" --sim-manifest-dir "$SIMQA/_review" --out "$REALQA" --split all
echo "== drop any manifest rows whose clip is missing/<1KB (defensive) =="
$PY - "$SIMQA" "$REALQA" <<'PYEOF'
import json, os, sys, glob
for base in sys.argv[1:]:
    for mf in glob.glob(os.path.join(base, "_review", "*.jsonl")):
        keep=[]; drop=0
        for line in open(mf):
            if not line.strip(): continue
            c=json.loads(line).get("clip_16","")
            if os.path.exists(c) and os.path.getsize(c)>=1024: keep.append(line)
            else:
                drop+=1
                try: os.path.exists(c) and os.path.getsize(c)<1024 and os.remove(c)
                except Exception: pass
        if drop: open(mf,"w").writelines(keep)
        print(f"  {os.path.basename(mf)}: kept {len(keep)} dropped {drop}")
PYEOF
echo "== real manifest done =="

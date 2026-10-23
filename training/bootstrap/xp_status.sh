#!/bin/bash
# One-screen status of a3 runs ON THE BOX: alive?, last epoch line, result. usage: bash xp_status.sh [GLOB=a3*]
W=/workspace/rsna-knee/work; G=${1:-a3*}
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
for d in $W/runs/$G/; do
  n=$(basename "$d"); pid=$(cat "$d/pid" 2>/dev/null || echo -)
  if [ -f "$d/result.json" ]; then
    /workspace/rsna-knee/venv/bin/python -c "import json;r=json.load(open('$d/result.json'));print(f\"$n DONE sel_ep {r['selected_epoch']} report {r['oof_val_auc_report']:.5f} gold {r['gold_auc']:.4f} {r['seconds']/3600:.2f} h\")"
  elif [ -f "$d/failure.json" ] && ! kill -0 "$pid" 2>/dev/null; then
    echo "$n FAILED: $(/workspace/rsna-knee/venv/bin/python -c "import json;print(json.load(open('$d/failure.json'))['error'][:200])")"
  else
    alive=$(kill -0 "$pid" 2>/dev/null && echo alive || echo DEAD)
    echo "$n $alive pid $pid | $(grep -E 'ep [0-9]+:' "$d/train.log" 2>/dev/null | tail -1)"
  fi
done

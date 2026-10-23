#!/bin/bash
# a3 box bootstrap step 0: project venv with the old A/C pins, torch inherited from the image.
# Run ON THE BOX as root:  bash /workspace/rsna-knee/work/bootstrap/00_env.sh
# Idempotent. ~2-4 min. Writes /workspace/rsna-knee/work/manifests/env_*.txt|json.
#
# Old boxes A/C: NGC torch 2.14.0a0 (nv26.8) + venv --system-site-packages with
#   timm 1.0.11, pandas 2.2.3, scikit-learn 1.6.1, numpy 2.1.0, scipy 1.18.0,
#   opencv-python-headless 4.11.0.86, safetensors 0.8.0, huggingface_hub 1.28.0
#   (notes/shutdown_2026-10-02/A_requirements.txt). The new image ships torch 2.11.0+cu128
#   in /venv/main; we keep that torch (RTX 5090 = sm_120 needs cu128) and pin the rest.
set -euo pipefail
BASE_PY=${BASE_PY:-/venv/main/bin/python}
R=/workspace/rsna-knee
VENV=$R/venv
M=$R/work/manifests
mkdir -p "$M" "$R/hf"

[ -x "$BASE_PY" ] || { echo "FATAL: $BASE_PY missing (wrong image?)"; exit 1; }
if [ ! -x "$VENV/bin/python" ]; then
  "$BASE_PY" -m venv --system-site-packages "$VENV"
fi
PY=$VENV/bin/python
"$PY" -m pip install -q --upgrade pip >/dev/null

# Freeze whatever torch/torchvision/triton/nvidia-* the image has so pip can never replace them.
"$PY" - > "$M/env_constraints.txt" <<'EOF'
import importlib.metadata as md, re
seen = set()
for d in md.distributions():
    n = (d.metadata["Name"] or "").lower().replace("_", "-")
    if n in seen:
        continue
    if n in ("torch", "torchvision", "torchaudio", "triton") or n.startswith("nvidia-"):
        seen.add(n)
        print(f"{n}=={d.version}")
EOF
echo "constraints (inherited GPU stack):"; cat "$M/env_constraints.txt"

STRICT="timm==1.0.11 pandas==2.2.3 scikit-learn==1.6.1 numpy==2.1.0 scipy==1.18.0 \
opencv-python-headless==4.11.0.86 safetensors==0.8.0 huggingface_hub==1.28.0 \
kaggle==2.2.3 requests tqdm"
# Fallback if the strict numpy/scipy/hub pins do not resolve against this image's python/torch:
LOOSE="timm==1.0.11 pandas==2.2.3 scikit-learn==1.6.1 opencv-python-headless==4.11.0.86 \
safetensors huggingface_hub kaggle==2.2.3 requests tqdm"
if "$PY" -m pip install -q -c "$M/env_constraints.txt" $STRICT; then
  echo "PINS: strict (old A/C versions)" | tee "$M/env_pinmode.txt"
else
  echo "WARN: strict pins failed; retrying with numpy/scipy/hub unpinned" >&2
  "$PY" -m pip install -q -c "$M/env_constraints.txt" $LOOSE
  echo "PINS: loose (numpy/scipy/huggingface_hub NOT the old versions)" | tee "$M/env_pinmode.txt"
fi
# transformers is NOT installed: only HF families (dinov2s, siglip) import it, lazily. The a57/f0d recipes never do.

"$PY" - <<'EOF'
import json, importlib, sys
mods = {}
for m in ["torch", "torchvision", "timm", "numpy", "pandas", "scipy", "sklearn", "cv2",
          "safetensors", "huggingface_hub"]:
    mod = importlib.import_module(m)
    mods[m] = {"version": getattr(mod, "__version__", "?"), "file": mod.__file__}
import torch
assert mods["timm"]["version"] == "1.0.11", mods["timm"]
assert mods["pandas"]["version"] == "2.2.3", mods["pandas"]
assert torch.cuda.is_available(), "CUDA not available"
n = torch.cuda.device_count()
caps = []
for i in range(n):
    with torch.cuda.device(i):
        x = torch.ones(256, 256, device=f"cuda:{i}")
        assert float((x @ x)[0, 0]) == 256.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = torch.nn.functional.conv2d(torch.randn(2, 3, 64, 64, device=f"cuda:{i}"),
                                           torch.randn(8, 3, 3, 3, device=f"cuda:{i}"))
        assert torch.isfinite(y).all()
    p = torch.cuda.get_device_properties(i)
    caps.append([p.name, p.total_memory // 2**30, f"{p.major}.{p.minor}"])
mods["_gpus"] = caps
mods["_torch_cuda"] = torch.version.cuda
mods["_bf16"] = torch.cuda.is_bf16_supported()
import timm
m = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=False, num_classes=0)
mods["_convnextv2_params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
json.dump(mods, open("/workspace/rsna-knee/work/manifests/env_imports.json", "w"), indent=2)
print(json.dumps({k: (v["version"] if isinstance(v, dict) else v) for k, v in mods.items()}, indent=1))
EOF
"$PY" -m pip freeze --all > "$M/env_pip_freeze.txt"
"$PY" -m pip check > "$M/env_pip_check.txt" 2>&1 || echo "WARN: pip check reported conflicts, see $M/env_pip_check.txt"
echo "nproc=$(nproc)  mem=$(free -g | awk '/Mem/{print $2}')G  disk=$(df -BG --output=avail /workspace | tail -1)"
echo "ENV OK -> $PY"

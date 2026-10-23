"""Build and push the 8 full-conversion shard kernels from the template."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
N_SHARDS = 8
TEMPLATE = (HERE / "prep_full_template.py").read_text(encoding="utf-8")

for s in range(N_SHARDS):
    d = HERE / f"shard_{s}"
    d.mkdir(exist_ok=True)
    (d / "prep_full.py").write_text(
        TEMPLATE.replace("__SHARD__", str(s)).replace("__NSHARDS__",
                                                      str(N_SHARDS)),
        encoding="utf-8")
    meta = {
        "id": f"komilparmar/rsna-knee-prep-full-s{s}",
        "title": f"rsna-knee-prep-full-s{s}",
        "code_file": "prep_full.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1),
                                            encoding="utf-8")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                       capture_output=True, text=True)
    print(f"shard {s}: {(r.stdout + r.stderr).strip().splitlines()[-1]}")
    if r.returncode != 0:
        sys.exit(f"push failed for shard {s}")

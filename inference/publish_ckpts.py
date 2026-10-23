"""Pull selected checkpoints from the Vast box and publish them as a private Kaggle dataset.

usage: python publish_ckpts.py --slug rsna-camp-exp103 --runs exp103_convnext_f0 exp103_coatnet_f0 ...
Files land as data/ckpts/<slug>/<run>.pt (one per run; ckpt_fold{k}.pt renamed), with a
MANIFEST.json of sha256s. Creates the dataset on first use, else pushes a new version.
The kernel (build_ens_kernel.py) mounts any *.pt under a dataset whose path contains 'rsna-camp'.
"""
import argparse
import hashlib
import json
import subprocess
import time
import sys
from pathlib import Path

HOST = ["-i", str(Path.home() / ".ssh/vast_ed25519"), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=20"]
BOXES = {}   # old A/B/C destroyed (2026-10-02). Team boxes: team/exchange/boxes_ssh.json {"a1": ["<port>", "root@<ip>"], ...}
_bx = Path(__file__).resolve().parents[2] / "team" / "exchange" / "boxes_ssh.json"
if _bx.exists():
    BOXES.update({k: tuple(v) for k, v in json.loads(_bx.read_text(encoding="utf-8")).items()})
ROOT = "/workspace/rsna-knee/work/runs"
PROJ = Path(__file__).resolve().parents[2]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--message", default="campaign checkpoints")
    ap.add_argument("--box", nargs="+", required=True,
                    help="box per run (one value = all runs), e.g. --box a1 a3 a3")
    a = ap.parse_args()
    assert "rsna-camp" in a.slug
    out = PROJ / "data" / "ckpts" / a.slug
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    boxes = a.box * len(a.runs) if len(a.box) == 1 else a.box
    assert len(boxes) == len(a.runs)
    for run, bx in zip(a.runs, boxes):
        PORT, ADDR = BOXES[bx]
        res = subprocess.run(["ssh", *HOST, "-p", PORT, ADDR, f"cat {ROOT}/{run}/result.json"],
                             capture_output=True, text=True, check=True).stdout
        r = json.loads(res[res.index("{"):])
        k = r["fold"]
        dst = out / f"{run}.pt"
        if not dst.exists() or sha(dst) != r["ckpt_sha256"]:
            for attempt in range(4):                     # transient scp drops happen
                try:                                     # a stalled scp must not hang forever
                    if subprocess.run(["scp", *HOST, "-P", PORT, "-q", f"{ADDR}:{ROOT}/{run}/ckpt_fold{k}.pt",
                                       str(dst)], timeout=240).returncode == 0:   # healthy: <1 min; stalls happen
                        break
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(10)
        assert sha(dst) == r["ckpt_sha256"], f"hash mismatch {run}"
        manifest[dst.name] = {"sha256": r["ckpt_sha256"], "run": run, "fold": k,
                              "selected_epoch": r["selected_epoch"],
                              "oof_val_auc_report": r["oof_val_auc_report"]}
    for p in out.glob("*.pt"):
        if p.name not in manifest:
            p.unlink()                       # dataset = exactly the requested runs
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    meta = {"title": a.slug, "id": f"komilparmar/{a.slug}", "licenses": [{"name": "CC0-1.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    exists = subprocess.run(["kaggle", "datasets", "status", f"komilparmar/{a.slug}"],
                            capture_output=True, text=True).returncode == 0
    # Windows kaggle CLI breaks on absolute -p paths (upload-cache path bug): run inside dir.
    cmd = (["kaggle", "datasets", "version", "-p", ".", "-m", a.message, "-q"] if exists
           else ["kaggle", "datasets", "create", "-p", ".", "-q"])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(out))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

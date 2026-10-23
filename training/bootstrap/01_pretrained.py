"""a3 box bootstrap step 1: generic ImageNet weights for the convnextv2 (a57/f0d/b8d) and convnext (s4c/a4c) families.

Run ON THE BOX:  /workspace/rsna-knee/venv/bin/python /workspace/rsna-knee/work/bootstrap/01_pretrained.py [FAMILY ...]
(default: both families) ~1 min (2 x ~110 MB). Idempotent.

How the old boxes did it: train_oof224.py reads <work>/pretrained/<family>/model.safetensors, passes it to
timm.create_model(..., pretrained=True, pretrained_cfg_overlay={'file': ..., 'source': ''}) via the env var
RSNA_CAMPAIGN_PRETRAINED_FILE, with HF_HUB_OFFLINE=1. The files came from the public HF hub repos
`timm/<model>` (code/download_pretrained.py pinned revisions for convnext/coatnet; the convnextv2 revision was
NOT recorded anywhere we have). Every a57ba940 / f0d1753e run_manifest.json records
pretrained_sha256 = 6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9, so we fetch from the
hub and accept only a file with that hash (tries main, then every older commit of the repo).
convnext (s4c = exp184): download_pretrained.py pins revision bc48a87f... with sha256 77b619e1..., which is exactly
the pretrained_sha256 of the exp182/exp184 stack4 cnx336 run manifests; that revision is tried first.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

SPECS = {
    "convnextv2": {"model": "convnextv2_tiny.fcmae_ft_in22k_in1k",
                   "want": "6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9",
                   # commit list of the repo (HF API, 2026-10-05): newest first
                   "revs": ["main", "b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817", "4c8d3d0fff982d2e204b58a7fd2efb81e94e047b",
                            "ffa838d8226776254489707de230f0e10fffab66", "a2b0dadabfcb1f9492e64b979f5b5a49b92acaae",
                            "c3a5698b903c4885b8eb2622f47db5b7bff1bcea"]},
    "convnext": {"model": "convnext_tiny.fb_in22k_ft_in1k",
                 "want": "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a",
                 "revs": ["bc48a87f119bae1b06418f5340bfa97ef75d4609", "main"]},   # pinned rev = download_pretrained.py
}
WORK = Path(os.environ.get("WORK", "/workspace/rsna-knee/work"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(fam, accept_any):
    spec = SPECS[fam]
    repo, want, dest = "timm/" + spec["model"], spec["want"], WORK / "pretrained" / fam / "model.safetensors"
    if dest.exists() and sha256(dest) == want:
        print(json.dumps({"family": fam, "status": "present", "path": str(dest), "sha256": want}))
        return
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download
    tmpdir = WORK / "pretrained" / ("_dl_" + fam)
    tried = {}
    for rev in spec["revs"]:
        try:
            p = Path(hf_hub_download(repo, "model.safetensors", revision=rev, local_dir=tmpdir / rev[:12], token=False))
        except Exception as e:                       # noqa: BLE001
            tried[rev] = f"error {type(e).__name__}: {str(e)[:120]}"
            continue
        h = sha256(p)
        tried[rev] = h
        if h == want or (accept_any and rev == "main"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            break
    else:
        print(json.dumps({"family": fam, "status": "FAILED", "want": want, "tried": tried}, indent=1))
        print("No hub revision matches the old boxes' file. Re-run with --accept-any to use main "
              "(new runs then differ from the old banks in pretrained weights - tell a1).", file=sys.stderr)
        sys.exit(1)
    got = sha256(dest)
    rec = {"status": "ok", "family": fam, "repo": repo, "path": str(dest), "sha256": got,
           "matches_old_boxes": got == want, "tried": tried}
    (WORK / "manifests").mkdir(exist_ok=True)
    (WORK / "manifests" / f"pretrained_{fam}.json").write_text(json.dumps(rec, indent=1))
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(json.dumps(rec, indent=1))
    # load check exactly as the trainer does
    os.environ["HF_HUB_OFFLINE"] = "1"
    import timm
    m = timm.create_model(spec["model"], pretrained=True, num_classes=0,
                          pretrained_cfg_overlay={"file": str(dest), "source": ""})
    print(fam, "timm load OK, params", round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "M")


def main():
    accept_any = "--accept-any" in sys.argv
    fams = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SPECS)
    for fam in fams:
        assert fam in SPECS, f"unknown family {fam} ({'|'.join(SPECS)})"
        fetch(fam, accept_any)


if __name__ == "__main__":
    main()

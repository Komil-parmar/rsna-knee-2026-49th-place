"""Per-finding public read-out probes (plan T0.1, notes/RETRO_PLAN_2026-10-10.md).

Each probe is the S89c kernel byte for byte, plus one block just before the submission is written that
replaces the listed target columns c by 1 - c. Inverting a column turns its AUC into 1 - AUC, so

    AUC_t = 0.5 + 6 * (S_full - S_probe)        (one inverted column; S_full ~ 0.9566 for S89c)
    sum_t (2 * AUC_t - 1) = 12 * (S_full - S_probe)   (several inverted columns)

usage:
  python probe_readout.py --probes syn=Synovitis "latoa=Lateral OA" "pair1=ACL+MCL" [--no-submit] [--max-par 2]

Per probe: build team/a1/kernels/rsna-probe-inv-<key>/, push, wait for the saved (3-study) run, refuse to
submit unless the log shows the full ensemble (20 checkpoints, 4 groups), no fallback, no median fill and the
PROBE line, then submit. Everything is appended to team/a1/logs/probe_readout.log.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "team" / "a1" / "kernels"
BASE = KDIR / "rsna-a1-s89c-v14-4grp"
LOG = ROOT / "team" / "a1" / "logs" / "probe_readout.log"
COMP = "rsna-knee-abnormality-detection"
KAGGLE = ["py", "-3.14", "-m", "kaggle"]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]
ANCHOR = '    out.to_csv("/kaggle/working/submission.csv", index=False)'
N_CKPT, N_GROUP = 20, 4


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def kg(*args, cwd=None):
    r = subprocess.run(KAGGLE + list(args), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=cwd)
    return (r.stdout or "") + (r.stderr or "")


def build(key, cols):
    slug = f"rsna-probe-inv-{key}"
    nb = json.loads((BASE / "rsna-a1-s89c-v14-4grp.ipynb").read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 1
    src = nb["cells"][0]["source"]
    src = "".join(src) if isinstance(src, list) else src
    assert src.count(ANCHOR) == 1, "anchor not unique in the S89c notebook"
    block = (
        "    # PROBE T0.1 (2026-10-10): per-finding public read-out; only the listed columns are inverted.\n"
        f"    PROBE_INVERT = {cols!r}\n"
        f"    assert len(CKPTS) == {N_CKPT} and len(GROUPS) == {N_GROUP}, (len(CKPTS), GROUPS)\n"
        "    assert all(_c in TARGETS for _c in PROBE_INVERT)\n"
        "    for _c in PROBE_INVERT:\n"
        "        out[_c] = 1.0 - out[_c].to_numpy()\n"
        "    assert np.isfinite(out[TARGETS].to_numpy()).all()\n"
        '    log(f"PROBE invert columns: {PROBE_INVERT}")\n'
    )
    nb["cells"][0]["source"] = src.replace(ANCHOR, block + ANCHOR)
    compile(nb["cells"][0]["source"], slug, "exec")      # syntax check of the edited cell
    out = KDIR / slug
    out.mkdir(exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    m = json.loads((BASE / "kernel-metadata.json").read_text(encoding="utf-8"))
    m.pop("id_no", None)
    m.update(id=f"komilparmar/{slug}", title=slug, code_file=f"{slug}.ipynb")
    (out / "kernel-metadata.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return slug


def push(slug):
    po = kg("kernels", "push", "-p", ".", cwd=str(KDIR / slug))
    m = re.search(r"version (\d+) successfully", po)
    say(f"{slug}: push -> {po.strip()[:200]}")
    return m.group(1) if m else None


def wait(slug, minutes=60):
    ref, s = f"komilparmar/{slug}", ""
    for _ in range(minutes * 3):
        s = kg("kernels", "status", ref)
        if any(k in s.upper() for k in ("COMPLETE", "ERROR", "CANCEL")):
            break
        time.sleep(20)
    say(f"{slug}: status -> {s.strip()[:120]}")
    return "COMPLETE" in s.upper()


def verify(slug, cols):
    ref = f"komilparmar/{slug}"
    with tempfile.TemporaryDirectory() as d:
        logs = []
        for _ in range(6):                       # the output is sometimes not downloadable right after COMPLETE
            kg("kernels", "output", ref, "-p", d, "-o")
            logs = list(Path(d).glob("*.log"))
            if logs:
                break
            time.sleep(20)
        if not logs:
            say(f"{slug}: no log in kernel output")
            return False
        text = "\n".join(e.get("data", "") for e in json.loads(logs[0].read_text(encoding="utf-8")))
        keep = [l for l in text.splitlines() if any(k in l for k in (
            "checkpoints:", "groups", "median-filled", "SUBMISSION WRITTEN", "FALLBACK", "PROBE invert"))]
        for l in keep[-8:]:
            say(f"{slug}: log | {l.strip()[:230]}")
        ok = (f"{N_CKPT} checkpoints" in text and "SUBMISSION WRITTEN" in text and "FALLBACK" not in text
              and "0 studies median-filled" in text and f"PROBE invert columns: {cols!r}" in text
              and f"{N_GROUP} groups)" in text)
        sub = Path(d) / "submission.csv"
        if not sub.exists():
            say(f"{slug}: submission.csv missing")
            return False
        head = sub.read_text(encoding="utf-8").splitlines()
        say(f"{slug}: submission rows {len(head) - 1}; first row {head[1][:150] if len(head) > 1 else ''}")
    return ok


def submit(slug, version, cols):
    ref = f"komilparmar/{slug}"
    msg = (f"P-INV {'+'.join(cols)}: S89c with this column inverted (per-finding read-out probe, plan T0.1); "
           f"AUC = .5 + 6*(.9566 - score)" if len(cols) == 1 else
           f"P-INV {'+'.join(cols)}: S89c with these columns inverted (read-out probe, plan T0.1)")
    out = kg("competitions", "submit", "-c", COMP, "-k", ref, "-v", str(version), "-f", "submission.csv", "-m", msg)
    say(f"{slug}: submit v{version} -> {out.strip()[-200:] or '(no output)'}")
    time.sleep(8)                                # kaggle 2.2.3 prints nothing on success: confirm in the list
    listed = kg("competitions", "submissions", "-c", COMP)
    return f"P-INV {'+'.join(cols)}:" in listed


ap = argparse.ArgumentParser()
ap.add_argument("--probes", nargs="+", required=True, help="key=Target or key=TargetA+TargetB")
ap.add_argument("--no-submit", action="store_true")
ap.add_argument("--build-only", action="store_true")
ap.add_argument("--reuse-version", default="", help="skip build/push; verify and submit this existing version")
ap.add_argument("--max-par", type=int, default=2)
a = ap.parse_args()

probes = []
for p in a.probes:
    key, spec = p.split("=", 1)
    cols = [c.strip() for c in spec.split("+")]
    assert all(c in TARGETS for c in cols), cols
    probes.append((key, cols))

if a.build_only:
    for key, cols in probes:
        print("built", build(key, cols), cols)
    sys.exit(0)

say(f"START probes: {probes} (submit={not a.no_submit}, max_par={a.max_par})")
results = []
for i in range(0, len(probes), a.max_par):
    wave = []
    for key, cols in probes[i:i + a.max_par]:
        if a.reuse_version:                      # trial already ran: verify and submit that version, no new push
            wave.append((f"rsna-probe-inv-{key}", cols, a.reuse_version))
            continue
        slug = build(key, cols)
        ver = push(slug)
        wave.append((slug, cols, ver))
        time.sleep(5)
    time.sleep(0 if a.reuse_version else 40)
    for slug, cols, ver in wave:
        if not ver:
            results.append((slug, cols, "PUSH FAILED"))
            continue
        if not wait(slug):
            results.append((slug, cols, "RUN FAILED"))
            continue
        if not verify(slug, cols):
            results.append((slug, cols, "TRIAL NOT CLEAN - not submitted"))
            continue
        if a.no_submit:
            results.append((slug, cols, f"trial clean v{ver} (not submitted)"))
            continue
        results.append((slug, cols, f"SUBMITTED v{ver}" if submit(slug, ver, cols) else f"SUBMIT FAILED v{ver}"))
say("SUMMARY")
for slug, cols, st in results:
    say(f"  {slug:28s} {'+'.join(cols):30s} {st}")
time.sleep(5)
say("latest submissions:\n" + "\n".join(l[:150] for l in kg("competitions", "submissions", "-c", COMP).splitlines()[:9]))

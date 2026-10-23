"""Wait for a pushed kernel's saved run, verify it wrote a REAL submission, then submit it.

usage: python run_and_submit.py <kernel-slug> "<message>" [--no-submit] [--push] [--version N]
  --push: push scripts/kaggle_submit/<slug>/ first and take the version from the push output
Refuses to submit if the saved run's log shows FALLBACK, any median-filled study, or no
'SUBMISSION WRITTEN' line — a fallback sample submission would burn a slot for nothing.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
slug, msg = sys.argv[1], sys.argv[2]
ref = f"komilparmar/{slug}"
import re
version = None
if "--version" in sys.argv:
    version = sys.argv[sys.argv.index("--version") + 1]
if "--push" in sys.argv:
    po = subprocess.run(["kaggle", "kernels", "push", "-p", "."], capture_output=True, text=True,
                        cwd=str(Path(__file__).resolve().parent / slug)).stdout
    print(po.strip())
    version = re.search(r"version (\d+) successfully", po).group(1)
    time.sleep(30)
assert version, "need --push or --version N"
for _ in range(240):
    s = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    if any(k in s for k in ("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(20)
print(s.strip())
assert "COMPLETE" in s, "kernel did not complete"
with tempfile.TemporaryDirectory() as d:
    subprocess.run(["kaggle", "kernels", "output", ref, "-p", d, "-o"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    logs = list(Path(d).glob("*.log"))
    text = "\n".join(e.get("data", "") for e in json.loads(logs[0].read_text(encoding="utf-8")))
    lines = [l for l in text.splitlines() if "[S10" in l or "FALLBACK" in l]
    print("\n".join(lines[-12:]))
    ok = ("SUBMISSION WRITTEN" in text and "FALLBACK" not in text
          and "0 studies median-filled" in text)
    sub = (Path(d) / "submission.csv").read_text().splitlines()
    print(sub[:2])
assert ok, "saved run did not produce a clean submission; NOT submitting"
if "--no-submit" in sys.argv:
    sys.exit(0)
out = subprocess.run(["kaggle", "competitions", "submit", "-c", "rsna-knee-abnormality-detection",
                      "-k", ref, "-v", str(version), "-f", "submission.csv", "-m", msg],
                     capture_output=True, text=True, encoding="utf-8", errors="replace")
print(out.stdout[-500:], out.stderr[-500:])
time.sleep(5)
print(subprocess.run(["kaggle", "competitions", "submissions", "-c", "rsna-knee-abnormality-detection"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()[2][:160])

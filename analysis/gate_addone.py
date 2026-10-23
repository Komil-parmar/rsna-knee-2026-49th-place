"""Plan 2026-10-20 line B gate: pooled add-one of a candidate family vs the a57-v14 seed-3000 control, folds 0-4.
usage: gate_addone.py <candidate dir with oof_pred_f{k}.npz or run dirs> <control dir from bank_to_fold_npz.py>
Pass = mean over folds of (cand increment - control increment) >= +.0005 AND mean Spearman(cand, trio) <= .97."""
import re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
cand, ctl = Path(sys.argv[1]), Path(sys.argv[2])
def fold_file(d, k):
    for p in (d / f"oof_pred_f{k}.npz",) + tuple(d.glob(f"*_f{k}/oof_pred.npz")):
        if p.is_file(): return p
    raise SystemExit(f"no fold {k} file under {d}")
diffs, sps, solos = [], [], []
for k in range(5):
    out = subprocess.run([sys.executable, str(ROOT / "team/a3/scripts/addone_fold.py"), str(k), str(fold_file(cand, k)), str(fold_file(ctl, k))],
                         capture_output=True, text=True, check=True).stdout
    print(out.strip())
    m = re.search(r"cand inc - control inc = ([+-][0-9.]+)", out); diffs.append(float(m.group(1)))
    sps.append(float(re.search(r"spearman cand-trio ([0-9.]+)", out).group(1)))
    solos.append(float(re.search(r"cand solo ([0-9.]+)", out).group(1)))
d, sp = sum(diffs) / 5, sum(sps) / 5
print(f"MEAN cand-control increment {d:+.5f} | mean spearman {sp:.4f} | mean solo {sum(solos)/5:.5f} | "
      f"GATE {'PASS' if d >= 0.0005 and sp <= 0.97 else 'FAIL'} (need >= +.0005 and <= .97)")

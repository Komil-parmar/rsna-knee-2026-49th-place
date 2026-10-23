"""Write per-fold run-style oof_pred.npz files (ids, pred) from a pooled OOF bank (team/exchange/oof/<XP>/oof.npz, key P in
common.npz order), so a bank can serve as the control of addone_fold.py.  usage: bank_to_fold_npz.py <XP dir name> <out dir>"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
c = np.load(ROOT / "notes/mlevolve/oof_bank/common.npz", allow_pickle=True)
ids, folds = c["ids"], c["fold"]
P = np.load(ROOT / "team/exchange/oof" / sys.argv[1] / "oof.npz")["P"]
assert P.shape == (len(ids), 12), P.shape
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for k in range(5):
    r = np.where(folds == k)[0]
    np.savez(out / f"oof_pred_f{k}.npz", ids=ids[r], pred=P[r].astype(np.float32))
    print(f"fold {k}: {len(r)} rows -> {out / f'oof_pred_f{k}.npz'}")

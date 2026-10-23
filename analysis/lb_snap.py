"""Rank-in-block instrument (notes/LEVERS_2026-10-19.md, lever 4).

  py -3.14 team/a1/tools/lb_snap.py snap            -> team/a1/lb/lb_<UTC>.csv (public leaderboard download)
  py -3.14 team/a1/tools/lb_snap.py read A.csv B.csv -> did our team best move between snapshot A (taken just before the
                                                        candidate was submitted) and B (after it scored)?

The board shows 3 decimals but Kaggle orders teams by the full-precision score, so within our displayed block the number of
FIXED teams above us measures sub-tick moves. A team is fixed when it has not submitted since snapshot A (LastSubmissionDate
<= A's time) and shows the same score in both snapshots. Passing one fixed team in the .962 block (~22 teams per tick)
is ~+.00004 macro. Pre-registered accept rule for a candidate: passes >= 2 fixed teams, or displays a higher tick.
"""
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
COMP = "rsna-knee-abnormality-detection"
US = 16671933
LB = Path(__file__).resolve().parents[1] / "lb"


def snap():
    LB.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "leaderboard", COMP, "-d", "-p", td],
                           capture_output=True, text=True, timeout=300)
        zs = list(Path(td).glob("*.zip"))
        if not zs:
            sys.exit("download failed: " + (r.stdout + r.stderr)[-300:])
        with zipfile.ZipFile(zs[0]) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            out = LB / f"lb_{now}.csv"
            out.write_bytes(z.read(name))
    d = pd.read_csv(out, encoding="utf-8-sig")
    me = d[d.TeamId == US].iloc[0]
    print(f"{out.name}  ({name})  teams {len(d)}  us rank {me.Rank} score {me.Score:.3f} subs {me.SubmissionCount} "
          f"last {me.LastSubmissionDate}  rank22 {d.loc[d.Rank == 22, 'Score'].iloc[0]:.3f}")
    return out


def stamp(p):
    s = Path(p).stem.split("_", 1)[1]
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%dT%H%M%S"))


def read(pa, pb):
    A, B = (pd.read_csv(p, encoding="utf-8-sig") for p in (pa, pb))
    ta = stamp(pa)
    ua, ub = A[A.TeamId == US].iloc[0], B[B.TeamId == US].iloc[0]
    m = B.merge(A[["TeamId", "Rank", "Score"]], on="TeamId", suffixes=("", "_a"))
    m["last"] = pd.to_datetime(m.LastSubmissionDate)
    fixed = m[(m["last"] <= ta) & (m.Score == m.Score_a) & (m.TeamId != US)]
    blk = fixed[fixed.Score_a == ua.Score]                        # fixed teams in our displayed block at A
    above_a = int((blk.Rank_a < ua.Rank).sum())
    above_b = int(((blk.Rank < ub.Rank) & (blk.Score == ub.Score)).sum()) if ub.Score == ua.Score else None
    print(f"A {Path(pa).name}: us rank {ua.Rank} score {ua.Score:.3f} subs {ua.SubmissionCount}")
    print(f"B {Path(pb).name}: us rank {ub.Rank} score {ub.Score:.3f} subs {ub.SubmissionCount}")
    print(f"fixed teams in our block at A: {len(blk)} (above us {above_a}, below {len(blk) - above_a})")
    if ub.Score > ua.Score:
        print(f"VERDICT: displayed score rose {ua.Score:.3f} -> {ub.Score:.3f}: ACCEPT")
    elif ub.Score < ua.Score:
        print("VERDICT: our team best can only rise; a lower display means the snapshots are inconsistent: re-pull")
    else:
        passed = above_a - above_b
        print(f"fixed teams above us: A {above_a} -> B {above_b}; passed {passed} (~{passed * .00004:+.5f} macro)")
        if ub.SubmissionCount == ua.SubmissionCount:
            print("VERDICT: our SubmissionCount has not changed: the board may lag (I-206); re-pull before deciding")
        else:
            print("VERDICT:", "ACCEPT (passed >= 2 fixed teams)" if passed >= 2 else "REJECT (no move: keep the base)")


if __name__ == "__main__":
    if sys.argv[1] == "snap":
        snap()
    elif sys.argv[1] == "read":
        read(sys.argv[2], sys.argv[3])

"""Phase-staggered slice sampling for our S89c-family inference code (I-213), shared by build_pb.py and build_blend4.py.

Each member runs once per phase, shifting both the a57 thin-series grid and the CAP linspace grid by p of one step (same
definition as team/a3/box/phase/rescore_phase.py), and its probabilities are averaged over phases. Phase 0 is exactly the
original kernel selection. Report-label OOF: mean of phases 0 and .5 = +.0003 per model (v14 a57 + 4a0c, folds 0 and 2).
"""


def _rep(s, old, new):
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)


OLD_LOOP = (
    "            th = MEMBER_THIN[MEMBER[k]]\n"
    "            if th not in _inp:\n"
    "                rows = thin_select(rows0, jn, *th) if th else rows0\n"
    "                if len(rows) > CAP:\n"
    "                    rows = rows[np.linspace(0, len(rows) - 1, CAP).astype(int)]\n"
    "                L = len(rows)\n"
    "                bis = _views_batch(imgs, jn, rows, th)\n"
    "                tis = {i_: torch.from_numpy(b).pin_memory().to(d, non_blocking=True) for i_, b in bis.items()}\n"
    "                tm = torch.from_numpy(meta[rows][None].astype(np.float32)).to(d)\n"
    "                tk = torch.ones((1, L), dtype=torch.float32, device=d)\n"
    "                _inp[th] = (tis, tm, tk)\n"
    "            tis, tm, tk = _inp[th]\n"
    "            dt_k = AMP_DT[k]\n"
    "            with torch.autocast(d.type, dtype=dt_k or torch.float32,\n"
    "                                enabled=(dt_k is not None and d.type == \"cuda\")):\n"
    "                z = m(tis[im], tm, tk).float()\n"
    "            probs[MEMBER[k]][sj] = torch.sigmoid(z)[0].cpu().numpy()\n")
NEW_LOOP = (
    "            th = MEMBER_THIN[MEMBER[k]]\n"
    "            _acc = 0.0\n"
    "            for _ph in PHASES:\n"
    "                if (th, _ph) not in _inp:\n"
    "                    rows = _phase_rows(rows0, jn, th, _ph)\n"
    "                    L = len(rows)\n"
    "                    bis = _views_batch(imgs, jn, rows, th)\n"
    "                    tis = {i_: torch.from_numpy(b).pin_memory().to(d, non_blocking=True) for i_, b in bis.items()}\n"
    "                    tm = torch.from_numpy(meta[rows][None].astype(np.float32)).to(d)\n"
    "                    tk = torch.ones((1, L), dtype=torch.float32, device=d)\n"
    "                    _inp[(th, _ph)] = (tis, tm, tk)\n"
    "                tis, tm, tk = _inp[(th, _ph)]\n"
    "                dt_k = AMP_DT[k]\n"
    "                with torch.autocast(d.type, dtype=dt_k or torch.float32,\n"
    "                                    enabled=(dt_k is not None and d.type == \"cuda\")):\n"
    "                    z = m(tis[im], tm, tk).float()\n"
    "                _acc = _acc + torch.sigmoid(z)[0].cpu().numpy()\n"
    "            probs[MEMBER[k]][sj] = _acc / len(PHASES)\n")


def apply(c, phases):
    """c: the S89c-family inference source; phases: list of floats, e.g. [0.0, 0.5]"""
    c = _rep(c, "    def triplet(imgs, r, j, n):\n",
             f"    PHASES = {phases!r}\n"
             "    def _phase_rows(rows0, jn, th, p):\n"
             "        rows = np.asarray(rows0)\n"
             "        if th and len(rows):\n"
             "            j = np.asarray(jn)[rows, 0].astype(np.int64); n = np.asarray(jn)[rows, 1].astype(np.int64)\n"
             "            g = thin_gap_rows(n, *th)\n"
             "            s = np.floor(p * g + 0.5).astype(np.int64)\n"
             "            keep = (j % g) == ((g // 2 + s) % g)\n"
             "            rows = rows[keep] if keep.any() else rows\n"
             "        if len(rows) > CAP:\n"
             "            if p == 0:   # exactly the original kernel selection\n"
             "                rows = rows[np.linspace(0, len(rows) - 1, CAP).astype(int)]\n"
             "            else:\n"
             "                pos = np.linspace(0, len(rows) - 1, CAP) + p * (len(rows) - 1) / (CAP - 1)\n"
             "                rows = rows[np.unique(np.clip(pos.astype(int), 0, len(rows) - 1))]\n"
             "        return rows\n\n"
             "    def triplet(imgs, r, j, n):\n")
    return _rep(c, OLD_LOOP, NEW_LOOP)

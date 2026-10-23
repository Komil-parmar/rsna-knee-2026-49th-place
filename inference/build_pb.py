"""Build rsna-pb-<tag>: the OAI-free pick-B kernel (own models only) from PB1 (team/a1/kernels/rsna-s89c-syn100, public 0.958).

Options (blend sprint 2026-10-14, notes/idea_sprint_2026-10-14/blend/REPORT.md #3):
  --cap N      slices per series for our 20 models (PB1 = 96; 160 measured +.0003..+.0005 OOF, ~3 h instead of ~2 h)
  --habit B    Synovitis site-habit removal before the composite (screened at 0.30)
  --oa-dec L   Lateral OA / PF OA decoupling after the composite (screened at 0.15)
usage: python build_pb.py --tag pb2 --cap 160 --habit 0.30 --oa-dec 0.15
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "inference" / "kernels"
PB1 = KDIR / "rsna-s89c-syn100"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--cap", type=int, default=96)
ap.add_argument("--habit", type=float, default=0.0)
ap.add_argument("--oa-dec", type=float, default=0.0)
ap.add_argument("--phases", default="0", help="comma-separated slice-sampling phases averaged per member, e.g. 0,0.5")
ap.add_argument("--ours-label", default="v14", help="v17a: replace the a57 s1000 and 4a0c v14 groups by their v17a retrains")
a = ap.parse_args()


def rep(s, old, new):
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)


nb = json.loads((PB1 / "rsna-s89c-syn100.ipynb").read_text(encoding="utf-8"))
c = nb["cells"][0]["source"]
c = c if isinstance(c, str) else "".join(c)
c = rep(c, "NEED224, NEED336, CAP = False, True, 96", f"NEED224, NEED336, CAP = False, True, {a.cap}")
if a.phases != "0":
    # Phase-staggered slice sampling (pick-B check 2026-10-16, team/a3/box/phase/: mean of phases 0 and .5 = +.0003 per model on
    # report-label OOF, 4 models): each member runs once per phase, shifting both the a57 thin-series grid and the CAP linspace
    # grid by p of one step (same definition as rescore_phase.py), and its probabilities are averaged over phases.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phase_patch
    c = phase_patch.apply(c, [float(x) for x in a.phases.split(",")])
I = "    "
habit = ""
if a.habit:
    tab = json.loads((ROOT / "inference/data/habit_synovitis_v06_k20.json").read_text(encoding="utf-8"))
    assert tab["K"] == 20 and len(tab["table"]) > 50
    habit = "\n".join(I + l for l in (
        f"HABIT = {a.habit!r}   # Synovitis site-habit removal (blend sprint #1)",
        "try:",
        "    import pydicom as _pd",
        f"    _TAB = {tab['table']!r}",
        f"    _GLOBAL = {tab['GLOBAL']!r}",
        "    def _nm(s):",
        "        s = str(s).upper()",
        "        if 'SIEMENS' in s: return 'SIEMENS'",
        "        if 'PHILIPS' in s: return 'PHILIPS'",
        "        if s.startswith('GE'): return 'GE'",
        "        if 'TOSHIBA' in s or 'CANON' in s: return 'CANON'",
        "        if 'HITACHI' in s or 'FUJI' in s: return 'FUJI-HITACHI'",
        "        return s[:12]",
        "    _h, _hit = [], 0",
        "    for _u in out['StudyInstanceUID']:",
        "        _keys = []",
        "        _sd = Path(ROOT) / 'test_series' / str(_u)",
        "        for _ser in (sorted(_sd.iterdir()) if _sd.is_dir() else []):",
        "            _f = next(iter(sorted(_ser.glob('*.dcm'))), None)",
        "            if _f is None: continue",
        "            try:",
        "                _d = _pd.dcmread(str(_f), stop_before_pixels=True, specific_tags=['Manufacturer', 'ManufacturerModelName', 'ImagingFrequency', 'ReceiveCoilName'])",
        "            except Exception:",
        "                continue",
        "            try: _fq = int(round(float(getattr(_d, 'ImagingFrequency', -1))))",
        "            except Exception: _fq = -1",
        "            _keys.append(_nm(getattr(_d, 'Manufacturer', 'NA')) + '|' + str(getattr(_d, 'ManufacturerModelName', 'NA')).upper().replace(' ', '')",
        "                         + '|' + str(_fq) + '|' + str(getattr(_d, 'ReceiveCoilName', 'NA')).upper().replace(' ', ''))",
        "        _k = max(sorted(set(_keys)), key=_keys.count) if _keys else None",
        "        _v = _TAB.get(_k)",
        "        _hit += _v is not None",
        "        _h.append(_GLOBAL if _v is None else _v)",
        "    out['Synovitis'] = _rk(_rk(out['Synovitis']) - HABIT * _rk(np.array(_h, dtype=float)))",
        "    log(f'HABIT applied {HABIT} table hits {_hit} / {len(out)}')",
        "except Exception as _e:",
        "    log(f'HABIT_FAILED {type(_e).__name__} {str(_e)[:200]} - Synovitis left without habit removal')",
    )) + "\n"
old_syn = ('    out["Synovitis"] = _rk(_rk(out["Synovitis"]) + 1.0 * _rk(out["Effusion"])\n'
           '                           + 0.75 * _rk((_rk(out["Medial OA"]) + _rk(out["Lateral OA"]) + _rk(out["PF OA"])) / 3))\n')
oa = ""
if a.oa_dec:
    oa = "\n".join(I + l for l in (
        f"OA_DEC = {a.oa_dec!r}   # Lateral OA / PF OA decoupling (blend sprint #2), from the pre-rule OA columns",
        "for _t in ('Lateral OA', 'PF OA'):",
        "    _o = [x for x in _P0 if x != _t]",
        "    out[_t] = _rk(_rk(_P0[_t]) - OA_DEC * _rk((_rk(_P0[_o[0]]) + _rk(_P0[_o[1]])) / 2))",
        "log(f'OA_DEC applied {OA_DEC}')",
    )) + "\n"
p0 = I + "_P0 = {x: out[x].to_numpy().copy() for x in ('Medial OA', 'Lateral OA', 'PF OA')}\n"
c = rep(c, old_syn, habit + p0 + old_syn + oa)
c = rep(c, '    log("SYN_POST applied (1.0, 0.75)")', f'    log("SYN_POST applied (1.0, 0.75); PB build {a.tag}: cap {a.cap} habit {a.habit} oa_dec {a.oa_dec}")')
compile(c, "pb", "exec")
nb["cells"][0]["source"] = c
slug = f"rsna-pb-{a.tag}"
out = KDIR / slug
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
m = json.loads((PB1 / "kernel-metadata.json").read_text(encoding="utf-8"))
m.pop("id_no", None)
m.update(id=f"komilparmar/{slug}", title=slug, code_file=f"{slug}.ipynb")
if a.ours_label != "v14":
    ms = [x.replace("rsna-camp-v14-a57/", f"rsna-camp-{a.ours_label}-a57/").replace("rsna-camp-v14-4a0c/", f"rsna-camp-{a.ours_label}-4a0c/") for x in m["model_sources"]]
    assert sum(x != y for x, y in zip(ms, m["model_sources"])) == 2, ms
    m["model_sources"] = ms
(out / "kernel-metadata.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("built", slug, "cap", a.cap, "habit", a.habit, "oa_dec", a.oa_dec)

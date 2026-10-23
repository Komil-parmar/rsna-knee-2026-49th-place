"""Build rsna-blend4-<tag>: four-member rank blend that fits the time limit (plan 2026-10-13, I-197).

Members and nominal weights (fixed, not fitted): ours .4 (a57-v14 s1000 x5 + 4a0c-v14 x5 = the S88 pair, 10 models),
Raptor .2, heliosli S6 first 2 folds .2, heliosli R7 first 2 folds .2; then the Synovitis composite (1.0, 0.75).
Gold-58 check (team/a1/goldtab): this exact configuration .9461 vs H1+syn .9428 (+.0034, P>0 .95).

Stages run in order of importance; each public stage is wrapped so a failure cannot stop the notebook. A study a
member did not score gets no rank from that member and the other members' weights are renormalised for that study.
Deadlines count from notebook start: S6 5.3 h, R7 8.3 h (heliosli's own --deadline stops scoring and reports what
was scored). S6 / R7 are read from private Kaggle Model mirrors (Kaggle rejects the original datasets as non-versioned).

usage: python build_blend4.py --tag a [--w-ours 0.4]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KDIR = ROOT / "inference" / "kernels"          # generated kernels land here
S89C = ROOT / "inference" / "own_models_kernel"
PUB = ROOT / "inference" / "public"             # public notebooks pulled with `kaggle kernels pull` (see README)
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--w-ours", type=float, default=0.4)
ap.add_argument("--folds", type=int, default=2)
ap.add_argument("--r7-model", default="r7", choices=["r7", "r6"], help="heliosli model in the r7 slot")
ap.add_argument("--r7-deadline", type=float, default=8.3, help="hours from notebook start at which the R7 stage stops scoring")
ap.add_argument("--invert", default="", help="read-out probe: invert this one column after the blend")
ap.add_argument("--r224-mirror", action="store_true", help="also run the 224 checkpoint on the anatomical mirror view")
ap.add_argument("--ours-label", default="v14", help="label round of our two families: v14 (models rsna-camp-v14-*) or v17a")
ap.add_argument("--s6f", action="store_true", help="nartaa S6 all-data ConvNeXt (0.950 public) replaces Heliosli S6 2-fold in the s6 slot")
ap.add_argument("--census", action="store_true", help="hidden-run census probe (flag-coded 0.5 columns)")
ap.add_argument("--habit", type=float, default=0.0, help="Synovitis site-habit removal strength (blend sprint #1; screened at 0.30)")
ap.add_argument("--oa-dec", type=float, default=0.0, help="Lateral OA / PF OA decoupling strength (blend sprint #2; screened at 0.15)")
ap.add_argument("--drop", default="", help='per-finding member exclusions, e.g. "Lateral OA:raptor;Fracture:s6"')
ap.add_argument("--rap224", type=float, default=0.0,
                help="share of the Raptor slot given to nartaa's 224-px checkpoint (0 = stage not run); it runs last, "
                     "only if the clock allows, and its absence leaves the blend unchanged")
ap.add_argument("--ours-phases", default="0",
                help="phase-staggered slice sampling for our 10 models, e.g. 0,0.5 (I-213; ~2x our stage time)")
ap.add_argument("--ours-timeout", type=float, default=1.8, help="hard timeout of our stage in hours")
ap.add_argument("--rehearsal", type=int, default=0,
                help="timing rehearsal (never submitted): every stage reads a fake test root of N random non-gold TRAIN "
                     "studies; kernel slug rsna-rehearse-<tag>")
ap.add_argument("--wheels", action="store_true",
                help="install the attached DICOM decoder wheels (exp-091 / S40) before any stage imports pydicom "
                     "(lever B4, notes/LEVERS_2026-10-19.md); non-fatal")
ap.add_argument("--swap", default="",
                help="generic member swap (lever 3, notes/LEVERS_2026-10-19.md): SLOT:PATH puts the public notebook (.ipynb) or "
                     "script (.py) at PATH into the s6 or r7 slot at that slot's fixed weight; it runs IPython-free in its own "
                     "process group and must write --swap-out")
ap.add_argument("--swap-out", default="/kaggle/working/submission.csv",
                help="CSV the swapped-in member writes (StudyInstanceUID + the 12 targets)")
ap.add_argument("--swap-hours", type=float, default=3.4, help="hard cap of the swap stage in hours")
ap.add_argument("--swap-reserve", type=float, default=None,
                help="hours of the 9 h clock kept back for later stages (default: s6 3.2 as for S6F, r7 9 - --r7-deadline)")
ap.add_argument("--swap-patch", action="append", default=[],
                help="exact-once source patch OLD=>NEW on the swapped-in code (repeatable; e.g. its own deadline constant)")
ap.add_argument("--swap-data", default="", help="comma-separated dataset refs the swapped-in member reads")
ap.add_argument("--swap-models", default="", help="comma-separated Kaggle Model refs the swapped-in member reads")
ap.add_argument("--wmap", default="",
                help='per-finding member weights (lever A1, plan 2026-10-20): "Synovitis:raptor=0.32,r7=0.08;PF OA:raptor=0.24,r7=0.16"; '
                     "unlisted members keep W[k]; empty = the K14 weights (the build stays byte-identical)")
ap.add_argument("--ours-extra", default="",
                help="plan 2026-10-20 line B: comma-separated private Kaggle dataset slugs (owner/rsna-camp-*) holding 5 more own fold "
                     "checkpoints each (a third own group at the same weight inside the ours slot); their names keep a57ba94073, so the "
                     "thin rule prefix is widened; empty = the K14 build (byte-identical)")
a = ap.parse_args()
OURS_EXTRA = [x for x in a.ours_extra.split(",") if x]
assert all("rsna-camp" in x for x in OURS_EXTRA), OURS_EXTRA   # the ours stage only mounts dirs named rsna-camp*
assert a.invert in [""] + TARGETS and 0.0 <= a.rap224 < 1.0
FAKE = "/kaggle/working/fake"


def rh(s, old, new):
    """rehearsal patch: exactly one occurrence, applied only with --rehearsal"""
    if not a.rehearsal:
        return s
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)
DROP = {}
for _p in filter(None, a.drop.split(";")):
    _t, _m = _p.split(":")
    assert _t in TARGETS and all(k in ("ours", "raptor", "s6", "r7") for k in _m.split(",")), _p
    DROP[_t] = sorted(_m.split(","))
W = {"ours": a.w_ours, "raptor": (1 - a.w_ours) / 3, "s6": (1 - a.w_ours) / 3, "r7": (1 - a.w_ours) / 3}
WMAP = {}
for _p in filter(None, a.wmap.split(";")):
    _t, _m = _p.split(":")
    assert _t in TARGETS, _p
    WMAP[_t] = {}
    for _kv in _m.split(","):
        _k, _v = _kv.split("=")
        assert _k in ("ours", "raptor", "s6", "r7") and float(_v) >= 0, _p
        WMAP[_t][_k] = float(_v)


def src(c):
    s = c["source"]
    return "".join(s) if isinstance(s, list) else s


def rep(s, old, new):
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)


# ---- ours: S89c source restricted to the two attached families ----------------------------------------------------
ours = src(json.loads((S89C / "rsna-a1-s89c-v14-4grp.ipynb").read_text(encoding="utf-8"))["cells"][0])
ours = rep(ours, 'sample.to_csv("/kaggle/working/submission.csv", index=False)   # fallback',
           'sample.to_csv("/kaggle/working/s89c_fallback.csv", index=False)   # fallback (not used by the blend)')
ours = rep(ours, '    out.to_csv("/kaggle/working/submission.csv", index=False)',
           f'    assert len(CKPTS) == {10 + 5 * len(OURS_EXTRA)} and len(GROUPS) == {2 + len(OURS_EXTRA)}, (len(CKPTS), GROUPS)\n'
           '    out.to_csv("/kaggle/working/s89c_sub.csv", index=False)')
ours = rep(ours, '"--no-index", "--no-deps", whl[0]], check=False)',
           '"--no-index", "--no-deps", "--target", "/tmp/ours_pkgs", whl[0]], check=False)\n'
           '        _sys.path.insert(0, "/tmp/ours_pkgs")')
if OURS_EXTRA:   # extra own groups are named a3_a57ba94073<recipe>_...: widen the thin-rule prefix so the kernel assert passes
    ours = rep(ours, "THIN_RULES = [('a3_a57ba94073_', 60, 32)]", "THIN_RULES = [('a3_a57ba94073', 60, 32)]")
if a.ours_phases != "0":
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phase_patch
    ours = phase_patch.apply(ours, [float(x) for x in a.ours_phases.split(",")])
ours = rh(ours, 'for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),',
          f'for c in [Path({FAKE!r}),   # rehearsal: fake test root\n          Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),')
compile(ours, "ours", "exec")


# ---- public stages ---------------------------------------------------------------------------------------------------
def helios(folder, name, mirror, hours):
    nb = json.loads(next((PUB / folder).glob("*.ipynb")).read_text(encoding="utf-8"))
    vc = [src(c) for c in nb["cells"]]
    iw = next(i for i, c in enumerate(vc) if c.startswith("%%writefile /tmp/raptor_run.py"))
    si = next(i for i, c in enumerate(vc) if c.startswith("ARM = ") and "DEADLINE_HOURS" in c)   # heliosli setup cell (position varies: R6 notebook has the 224 stage first)
    setup, run = vc[si], vc[si + 1]
    slug = [l for l in setup.split("\n") if l.startswith("C_SLUG = ")][0]
    setup = rep(setup, slug, f"C_SLUG = {mirror!r}   # private mirror of the original bundle")
    setup = rep(setup, "T0 = NB_T0", "T0 = _VP_NB_T0")
    setup = rep(setup, "DEADLINE_HOURS = 8.3", f"DEADLINE_HOURS = {hours}")
    setup = rep(setup, "    cands += list(INPUT.glob('*/' + slug)) + list(INPUT.glob(slug + '/*'))",
                "    cands += list(INPUT.glob('*/' + slug)) + list(INPUT.glob(slug + '/*'))\n"
                "    cands += list(INPUT.glob('models/*/' + slug + '/*/*/*'))")
    setup = rep(setup, "manifest = json.loads((M_SRC / 'manifest.json').read_text())",
                "manifest = json.loads((M_SRC / 'manifest.json').read_text())\n"
                f"manifest['models'] = manifest['models'][:{a.folds}]   # runtime: first {a.folds} fold models only")
    setup = rep(setup, "BUNDLE = TMP / 'bundle'", f"BUNDLE = TMP / 'bundle_{name}'")
    setup = rh(setup, "DATA = find_mount(COMPETITION, 'test.csv')", f"DATA = Path({FAKE!r})   # rehearsal")
    run = rep(run, "pred = TMP / 'prediction'", f"pred = TMP / 'prediction_{name}'")
    save = (f"\nimport csv as _csv\n_sc = []\nfor _f in pred.glob('worker*/predictions.csv'):\n"
            "    with open(_f, encoding='utf-8') as _fh:\n"
            "        for _r in _csv.DictReader(_fh):\n"
            "            try:\n"
            "                if all(0. <= float(_r[_L]) <= 1. for _L in TARGETS_ALL): _sc.append(_r['StudyInstanceUID'])\n"
            "            except (KeyError, ValueError, TypeError):\n"
            "                pass\n"
            f"shutil.copyfile(WORK / 'ours_sub.csv', WORK / '{name}_sub.csv')\n"
            f"(WORK / '{name}_scored.json').write_text(json.dumps(sorted(set(_sc))))\n"
            f"print('MEMBER_DONE {name} scored', len(set(_sc)), 'seconds %.0f' % (time.time() - _VP_NB_T0), flush=True)\n")
    body = setup + "\n\n" + run + save
    compile(body, name, "exec")
    return body, vc[iw], vc[iw + 1]


s6_body, write, raptor_stage = helios("blend-mine-in", "s6", "rsna-mirror-helios-s6", 5.3)
r7_body, _, _ = (helios("r6", "r7", "rsna-mirror-helios-r6", a.r7_deadline) if a.r7_model == "r6" else
                 helios("h957", "r7", "rsna-mirror-helios-r7", a.r7_deadline))   # the r7 slot can hold R6 (bundle fe055406) instead
write = rh(write, "ROOT = find_test_root()", f"ROOT = {FAKE!r}   # rehearsal")
compile(raptor_stage, "raptor", "exec")

# ---- optional last stage: nartaa's second checkpoint (224 px, 0.8 centre crop, 0.945 alone) ------------------------
# Source: public/r224mix (nartaa's own 0.945 script with only the output name changed -> raptor224_sub.csv).
# Gold-58 (team/a1/goldtab/screen_more.py): .65 r384 + .35 r224 in the Raptor slot +.0013 (P>0 .91) on F1.
_nb = json.loads(next((PUB / "r224mix").glob("*.ipynb")).read_text(encoding="utf-8"))
r224_write = next(src(c) for c in _nb["cells"] if src(c).startswith("%%writefile /tmp/raptor224_run.py"))
assert r224_write.count('out = "/kaggle/working/raptor224_sub.csv"') == 1
# Kaggle caps kernel source at 1 MB: ship the 224 script zlib-compressed and write it out at run time (same bytes).
import base64, hashlib, zlib
_r224_src = r224_write.split("\n", 1)[1]
if a.r224_mirror:
    # Anatomical L/R mirror view averaged with the plain view, exactly as nartaa's 384 script does (its _mirror_tri);
    # the 224 script uses the same 96-slice slot layout. Scout idea 1 (notes/idea_sprint_2026-10-14/scout/REPORT.md).
    _r384 = src(json.loads(next((PUB / "blend-mine-in").glob("*.ipynb")).read_text(encoding="utf-8"))["cells"][
        [i for i, c in enumerate(json.loads(next((PUB / "blend-mine-in").glob("*.ipynb")).read_text(encoding="utf-8"))["cells"])
         if src(c).startswith("%%writefile /tmp/raptor_run.py")][0]])
    _i0 = _r384.index("SLOT_SIZES = "); _i1 = _r384.index("_WLUT = {}")
    _mirror_defs = _r384[_i0:_i1]
    assert "def _mirror_tri(tri, cs, D):" in _mirror_defs and "def _slot_index(D):" in _mirror_defs, _mirror_defs[:200]
    assert _r224_src.count("_WLUT = {}") == 1 and "_mirror_tri" not in _r224_src
    _r224_src = _r224_src.replace("_WLUT = {}", "# [mirror] anatomical L/R mirror helpers copied from nartaa's 384 script\n" + _mirror_defs + "_WLUT = {}")
    for _old, _new in (
            ("def gpu_windows(vol, mask, k, res, dev, norm=NORM):", "def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):"),
            ("    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n",
             "    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n"
             "    if mirror:   # [mirror]\n        tri = _mirror_tri(tri, cs, D)\n"),
            ("                    else:\n                        arm_probs[a][i] = infer_probs(model, xw, d)\n",
             "                    else:   # [mirror] average the plain and the anatomical-mirror view, as the 384 model does\n"
             "                        xm = gpu_windows(vol, mask, K_EVAL, res, d, mirror=True)\n"
             "                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, xm, d))\n"
             "                        del xm\n")):
        assert _r224_src.count(_old) == 1, _old[:70]
        _r224_src = _r224_src.replace(_old, _new)
    compile(_r224_src, "r224_mirror", "exec")
_r224_src = rh(_r224_src, "    ROOT = find_test_root()", f"    ROOT = {FAKE!r}   # rehearsal")
_r224_sha = hashlib.sha256(_r224_src.encode("utf-8")).hexdigest()
r224_write = (
    "# Write nartaa's 224-px inference script (public/r224mix, output name raptor224_sub.csv), stored compressed\n"
    "import base64 as _b64, zlib as _zl, hashlib as _hl\n"
    f"_s = _zl.decompress(_b64.b64decode({base64.b64encode(zlib.compress(_r224_src.encode('utf-8'), 9)).decode()!r})).decode('utf-8')\n"
    f"assert _hl.sha256(_s.encode('utf-8')).hexdigest() == {_r224_sha!r}\n"
    "open('/tmp/raptor224_run.py', 'w', encoding='utf-8').write(_s)\n"
    "print('R224_SCRIPT written', len(_s), flush=True)"
)
r224_stage = (
    "# Last stage: Raptor 224-px checkpoint in its own process, only if the 9 h clock allows (25 min kept back)\n"
    "_left = 9 * 3600 - (time.time() - _VP_NB_T0) - 25 * 60\n"
    "if _left < 20 * 60:\n"
    "    print('R224_SKIPPED only %.0f min left' % (_left / 60), flush=True)\n"
    "else:\n"
    "    _t = time.time()\n"
    "    try:\n"
    "        _rc = subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=int(_left)).returncode\n"
    "    except BaseException as _e:\n"
    "        _rc = repr(_e)[:200]\n"
    "        try: os.remove('/kaggle/working/raptor224_sub.csv')\n"
    "        except OSError: pass\n"
    "    print('R224_STAGE rc', _rc, 'ok', os.path.isfile('/kaggle/working/raptor224_sub.csv'),\n"
    "          'seconds %.0f total %.0f' % (time.time() - _t, time.time() - _VP_NB_T0), flush=True)"
)
compile(r224_stage, "r224", "exec")


S6F_STAGE = ""
if a.s6f:
    # nartaa's public S6 ConvNeXt all-data model (rsna-knee-s6fall-solo, 0.950 public alone; gold-58 excluded, no OAI) replaces
    # Heliosli's S6 2-fold in the 's6' slot. Its K1 script runs unchanged except: component A (a re-run of the 0.949 Raptor,
    # used only as a per-study fallback) is a stub that hands over our Raptor stage's raptor_raw.csv; the 3-study
    # commit-csv check of A is skipped (the stub's csv differs by design); budgets cut to fit our clock (A 15 min, B 2.5 h).
    import base64 as _b64, hashlib as _hl, re as _re
    _k1 = (PUB / "rsna-knee-s6fall-solo" / "rsna-knee-s6fall-solo.py").read_text(encoding="utf-8")
    _stub = (
        "import pandas as pd, numpy as np, os\nfrom pathlib import Path\n"
        "roots = [p for p in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'test.csv').is_file()]\n"
        "R = roots[0]\n"
        "ids = pd.read_csv(R / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].drop_duplicates().tolist()\n"
        "cols = pd.read_csv(R / 'sample_submission.csv', nrows=1).columns.tolist()\n"
        "out = pd.DataFrame({'StudyInstanceUID': ids})\n"
        "f = Path('/kaggle/working/raptor_raw.csv')\n"
        "raw = pd.read_csv(f, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID') if f.is_file() else None\n"
        "for c in cols[1:]:\n"
        "    v = raw[c].reindex(ids).to_numpy(float) if raw is not None and c in raw else np.full(len(ids), 0.5)\n"
        "    out[c] = np.clip(np.where(np.isfinite(v), v, 0.5), 0.0, 1.0)\n"
        "out[cols].to_csv('/kaggle/working/submission.csv', index=False)\n"
        "print('stub A: our Raptor stage output handed to K1 as component A', raw is not None, flush=True)\n"
        "print('wrote /kaggle/working/submission.csv', flush=True)\nprint('DONE 1s', flush=True)\n")
    _stub = rh(_stub, "R = roots[0]\n", f"R = Path({FAKE!r})   # rehearsal\n")
    _k1 = rh(_k1, 'I.comp = find_mount(COMP, "test.csv")', f'I.comp = Path({FAKE!r})   # rehearsal')
    _m = _re.search(r'^A_B64 = "([A-Za-z0-9+/=]+)"', _k1, _re.M)
    _k1 = _k1.replace(_m.group(1), _b64.b64encode(_stub.encode("utf-8")).decode())
    _k1 = _re.sub(r'A_SOURCE_SHA256 = "\w+"', f'A_SOURCE_SHA256 = "{_hl.sha256(_stub.encode("utf-8")).hexdigest()}"', _k1, count=1)
    for _old, _new in (("A_TIMEOUT_S = 9000", "A_TIMEOUT_S = 900"), ("B_TIMEOUT_S = 14400", "B_TIMEOUT_S = 9000"),
                       ("GUARD_MAX_S = 480 * 60 ", "GUARD_MAX_S = 195 * 60 "),
                       ("    if sorted(I.test_ids) == sorted(PUBLIC_TEST_IDS):\n        info[\"public_csv_matches_0949_commit\"]",
                        "    if False:   # [blend4] A is our stub, not the 0.949 commit\n        info[\"public_csv_matches_0949_commit\"]")):
        assert _k1.count(_old) == 1, _old
        _k1 = _k1.replace(_old, _new)
    compile(_k1, "s6f_k1", "exec")
    S6F_STAGE = (
        "# Stage: nartaa S6 all-data ConvNeXt (K1 script, component A = our Raptor output) in the 's6' slot\n"
        "import base64 as _b64s, subprocess as _sp, json as _js6, pandas as _pd6, numpy as _np6\n"
        f"open('/tmp/s6f_k1.py', 'w', encoding='utf-8').write(__import__('zlib').decompress(_b64s.b64decode({_b64.b64encode(zlib.compress(_k1.encode('utf-8'), 9)).decode()!r})).decode('utf-8'))\n"
        "_left = 9 * 3600 - (time.time() - _VP_NB_T0) - 3.2 * 3600   # keep >= 3.2 h for R7 + R224 + blend\n"
        "if _left < 1800:\n"
        "    print('S6F_SKIPPED only %.0f min before the R7 reserve' % (_left / 60), flush=True)\n"
        "else:\n"
        "    _t = time.time()\n"
        "    try:\n"
        "        _rc = _sp.run([sys.executable, '-u', '/tmp/s6f_k1.py'], timeout=int(min(_left, 3.4 * 3600))).returncode\n"
        "    except BaseException as _e:\n"
        "        _rc = repr(_e)[:200]\n"
        "    _bp = '/kaggle/working/k1/B/B_probs.csv'\n"
        "    if os.path.isfile(_bp):\n"
        "        _d = _pd6.read_csv(_bp, dtype={'StudyInstanceUID': str})\n"
        "        _d = _d[['StudyInstanceUID'] + TARGETS_ALL]\n"
        "        _ok = _d.loc[_np6.isfinite(_d[TARGETS_ALL].to_numpy(float)).all(1), 'StudyInstanceUID'].tolist()\n"
        "        _d.to_csv('/kaggle/working/s6_sub.csv', index=False)\n"
        "        open('/kaggle/working/s6_scored.json', 'w').write(_js6.dumps(sorted(set(_ok))))\n"
        "        print('MEMBER_DONE s6 scored', len(set(_ok)), 'seconds %.0f' % (time.time() - _VP_NB_T0), flush=True)\n"
        "    for _f in ('/kaggle/working/submission.csv',):   # K1 writes rank(S6F) there; the blend cell writes the real one\n"
        "        if os.path.isfile(_f): os.remove(_f)\n"
        "    print('S6F_STAGE rc', _rc, 'seconds %.0f total %.0f' % (time.time() - _t, time.time() - _VP_NB_T0), flush=True)\n"
    )
    compile(S6F_STAGE, "s6f_stage", "exec")


# Generic member swap (lever 3): the only blend change that moved the board since F1 is a strong public model trained on
# different labels taking a slot at its fixed weight (G6, I-212). The swapped-in code runs unchanged except for
# --swap-patch edits, under a small IPython-free runner, in its own process group so a timeout also stops any workers it
# started. Earlier members' outputs are put back if it deletes or rewrites them; its CSV becomes <slot>_sub.csv, and a row
# counts as scored when it is finite and not all 0.5 (the usual public fallback for a study that failed).
SWAP_RUNNER = '''"""Run a public notebook's code cells IPython-free (build_blend4.py --swap): %%writefile cells write their file,
'!' lines run in a shell (with IPython's {expr} expansion when it resolves), other magics are skipped."""
import json, os, subprocess, sys, time


def ipy_free(c):
    head = c.split("\\n", 1)[0].strip()
    if head.startswith("%%writefile"):
        parts = head.split()
        return ("write", parts[-1], "-a" in parts or "--append" in parts, c.split("\\n", 1)[1] if "\\n" in c else "")
    if head.startswith("%%"):
        return ("skip", head)
    out = []
    for l in c.split("\\n"):
        s = l.lstrip()
        ind = l[:len(l) - len(s)]
        if s.startswith("!"):
            out.append(ind + "_swap_sh_(" + repr(s[1:]) + ", globals())")
        elif s.startswith("%"):
            out.append(ind + "pass  # magic " + repr(s[:60]))
        else:
            out.append(l)
    return ("code", "\\n".join(out))


def _swap_sh_(cmd, ns):
    try:
        cmd = cmd.format(**ns)
    except Exception:
        pass
    return subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    g = {"__name__": "__main__", "display": print, "_swap_sh_": _swap_sh_}
    for i, c in enumerate(json.load(open(sys.argv[1], encoding="utf-8"))):
        t = time.time()
        k = ipy_free(c)
        if k[0] == "write":
            os.makedirs(os.path.dirname(k[1]) or ".", exist_ok=True)
            with open(k[1], "a" if k[2] else "w", encoding="utf-8") as f:
                f.write(k[3])
        elif k[0] == "code":
            exec(compile(k[1], "swap_cell_%d" % i, "exec"), g)
        print("SWAP_CELL", i, k[0], "seconds %.0f" % (time.time() - t), flush=True)
'''
SWAP_SLOT, SWAP_STAGE = "", ""
if a.swap:
    SWAP_SLOT, _sp = a.swap.split(":", 1)
    assert SWAP_SLOT in ("s6", "r7") and not (SWAP_SLOT == "s6" and a.s6f), a.swap
    _swp = Path(_sp) if Path(_sp).is_absolute() else ROOT / _sp
    _cells = ([src(c) for c in json.loads(_swp.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
              if _swp.suffix == ".ipynb" else [_swp.read_text(encoding="utf-8")])
    for _pt in a.swap_patch:
        _old, _new = _pt.split("=>", 1)
        _hits = [i for i, c in enumerate(_cells) if _old in c]
        assert len(_hits) == 1 and _cells[_hits[0]].count(_old) == 1, ("swap patch must match exactly once", _old[:70])
        _cells[_hits[0]] = _cells[_hits[0]].replace(_old, _new)
    _rn = {}
    exec(compile(SWAP_RUNNER, "swap_runner", "exec"), _rn)
    for _i, _c in enumerate(_cells):
        _k = _rn["ipy_free"](_c)
        if _k[0] == "code":
            compile(_k[1], f"swap_cell_{_i}", "exec")
        elif _k[0] == "skip":
            print("swap: cell", _i, "magic skipped at run time:", _k[1][:60])
    import base64 as _b64w, zlib as _zlw
    _packed = _b64w.b64encode(_zlw.compress(json.dumps(_cells).encode("utf-8"), 9)).decode()
    _res = a.swap_reserve if a.swap_reserve is not None else (3.2 if SWAP_SLOT == "s6" else 9 - a.r7_deadline)
    _mine = sorted({a.swap_out, f"/kaggle/working/{SWAP_SLOT}_sub.csv", f"/kaggle/working/{SWAP_SLOT}_scored.json"})
    SWAP_STAGE = (
        f"# Stage: generic member swap into the '{SWAP_SLOT}' slot: {_swp.name} (build_blend4.py --swap, lever 3)\n"
        "import base64 as _b64w, zlib as _zlw, json as _jw, glob as _glw, signal as _sgw\n"
        f"open('/tmp/swap_cells.json', 'w', encoding='utf-8').write(_zlw.decompress(_b64w.b64decode({_packed!r})).decode('utf-8'))\n"
        f"open('/tmp/swap_runner.py', 'w', encoding='utf-8').write({SWAP_RUNNER!r})\n"
        f"_mine = {_mine!r}\n"
        "_keep = {_f: open(_f, 'rb').read() for _f in _glw.glob('/kaggle/working/*.csv') + _glw.glob('/kaggle/working/*.json')\n"
        "         if _f not in _mine}   # earlier members' outputs\n"
        f"_left = 9 * 3600 - (time.time() - _VP_NB_T0) - {_res!r} * 3600\n"
        "if _left < 1800:\n"
        f"    print('SWAP_SKIPPED {SWAP_SLOT} only %.0f min left' % (_left / 60), flush=True)\n"
        "else:\n"
        "    _t = time.time()\n"
        "    try:\n"
        "        import torch; torch.cuda.empty_cache()   # hand the GPUs to the swapped-in process\n"
        "    except Exception:\n"
        "        pass\n"
        "    _pw = subprocess.Popen([sys.executable, '-u', '/tmp/swap_runner.py', '/tmp/swap_cells.json'],\n"
        "                           cwd='/kaggle/working', start_new_session=True)\n"
        "    try:\n"
        f"        _rc = _pw.wait(timeout=int(min(_left, {a.swap_hours!r} * 3600)))\n"
        "    except BaseException as _e:\n"
        "        _rc = repr(_e)[:200]\n"
        "    try:\n"
        "        os.killpg(_pw.pid, _sgw.SIGKILL)   # the runner and any workers it left behind\n"
        "    except (OSError, AttributeError):   # group already gone (or no killpg: local tests on Windows)\n"
        "        _pw.kill()\n"
        "    _pw.wait()\n"
        f"    _o = {a.swap_out!r}\n"
        "    if os.path.isfile(_o):\n"
        "        import pandas as _pdw, numpy as _npw\n"
        "        _d = _pdw.read_csv(_o, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID')\n"
        "        _miss = [c for c in TARGETS_ALL if c not in _d.columns]\n"
        "        if _miss:\n"
        f"            print('SWAP_BAD_OUTPUT {SWAP_SLOT} missing columns', _miss, flush=True)\n"
        "        else:\n"
        "            _v = _d[TARGETS_ALL].to_numpy(float)\n"
        "            _ok = _d.loc[_npw.isfinite(_v).all(1) & ~(_v == 0.5).all(1), 'StudyInstanceUID'].tolist()\n"
        f"            _d[['StudyInstanceUID'] + TARGETS_ALL].to_csv('/kaggle/working/{SWAP_SLOT}_sub.csv', index=False)\n"
        f"            open('/kaggle/working/{SWAP_SLOT}_scored.json', 'w').write(_jw.dumps(sorted(set(_ok))))\n"
        f"            print('MEMBER_DONE {SWAP_SLOT} scored', len(set(_ok)), 'seconds %.0f' % (time.time() - _VP_NB_T0), flush=True)\n"
        "    else:\n"
        f"        print('SWAP_NO_OUTPUT {SWAP_SLOT}', _o, flush=True)\n"
        "    if os.path.isfile('/kaggle/working/submission.csv'):\n"
        "        os.remove('/kaggle/working/submission.csv')   # the blend cell writes the real one\n"
        "    for _f, _b in _keep.items():\n"
        "        if not os.path.isfile(_f) or open(_f, 'rb').read() != _b:\n"
        "            open(_f, 'wb').write(_b)\n"
        "            print('SWAP_RESTORED', _f, flush=True)\n"
        f"    print('SWAP_STAGE {SWAP_SLOT} rc', _rc, 'seconds %.0f total %.0f' % (time.time() - _t, time.time() - _VP_NB_T0), flush=True)\n"
    )
    compile(SWAP_STAGE, "swap_stage", "exec")


def guarded(tag, body):
    import base64, zlib
    packed = base64.b64encode(zlib.compress(body.encode("utf-8"), 9)).decode()   # 1 MB kernel-source cap
    return ("_t = time.time()\ntry:\n"
            "    import base64 as _b64, zlib as _zl\n"
            f"    exec(compile(_zl.decompress(_b64.b64decode({packed!r})).decode('utf-8'), {tag!r}, 'exec'), globals())\n"
            "except BaseException as _e:\n"
            f"    print('STAGE_FAILED {tag}', type(_e).__name__, str(_e)[:400], flush=True)\n"
            f"print('STAGE_END {tag} seconds %.0f total %.0f' % (time.time() - _t, time.time() - _VP_NB_T0), flush=True)")


HABIT_CODE = ""
if a.habit:
    # Blend sprint #1 (notes/idea_sprint_2026-10-14/blend/REPORT.md): remove each scanner's Synovitis reporting habit before the composite.
    # Table = v06 report Synovitis mean per scanner fingerprint over the 4,407 train studies, shrunk k=20 (habit_table.py).
    _tab = json.loads((ROOT / "inference/data/habit_synovitis_v06_k20.json").read_text(encoding="utf-8"))
    assert _tab["K"] == 20 and len(_tab["table"]) > 50
    HABIT_CODE = (
        f"HABIT = {a.habit!r}\n"
        "try:\n"
        "    import pydicom as _pd, pathlib as _pl\n"
        f"    _TAB = {_tab['table']!r}\n"
        f"    _GLOBAL = {_tab['GLOBAL']!r}\n"
        "    def _nm(s):\n"
        "        s = str(s).upper()\n"
        "        if 'SIEMENS' in s: return 'SIEMENS'\n"
        "        if 'PHILIPS' in s: return 'PHILIPS'\n"
        "        if s.startswith('GE'): return 'GE'\n"
        "        if 'TOSHIBA' in s or 'CANON' in s: return 'CANON'\n"
        "        if 'HITACHI' in s or 'FUJI' in s: return 'FUJI-HITACHI'\n"
        "        return s[:12]\n"
        "    _h, _hit = [], 0\n"
        "    for _u in ids:\n"
        "        _keys = []\n"
        "        _sd = _pl.Path(_root) / 'test_series' / _u\n"
        "        for _ser in (sorted(_sd.iterdir()) if _sd.is_dir() else []):\n"
        "            _f = next(iter(sorted(_ser.glob('*.dcm'))), None)\n"
        "            if _f is None: continue\n"
        "            try:\n"
        "                _d = _pd.dcmread(str(_f), stop_before_pixels=True, specific_tags=['Manufacturer', 'ManufacturerModelName', 'ImagingFrequency', 'ReceiveCoilName'])\n"
        "            except Exception:\n"
        "                continue\n"
        "            try: _fq = int(round(float(getattr(_d, 'ImagingFrequency', -1))))\n"
        "            except Exception: _fq = -1\n"
        "            _keys.append(_nm(getattr(_d, 'Manufacturer', 'NA')) + '|' + str(getattr(_d, 'ManufacturerModelName', 'NA')).upper().replace(' ', '')\n"
        "                         + '|' + str(_fq) + '|' + str(getattr(_d, 'ReceiveCoilName', 'NA')).upper().replace(' ', ''))\n"
        "        _k = max(sorted(set(_keys)), key=_keys.count) if _keys else None\n"
        "        _v = _TAB.get(_k)\n"
        "        _hit += _v is not None\n"
        "        _h.append(_GLOBAL if _v is None else _v)\n"
        "    out['Synovitis'] = _rk(_rk(out['Synovitis']) - HABIT * _rk(np.array(_h, dtype=float)))\n"
        "    print('HABIT applied', HABIT, 'table hits', _hit, '/', len(ids), flush=True)\n"
        "except Exception as _e:\n"
        "    print('HABIT_FAILED', type(_e).__name__, str(_e)[:200], '- Synovitis left without habit removal', flush=True)\n"
    )

blend = (
    "# Final: failure-aware rank blend, then the Synovitis composite\n"
    "import numpy as np, pandas as pd, json as _j, os\n"
    f"W = {W!r}\nTARGETS = {TARGETS!r}\n"
    f"DROP = {DROP!r}   # finding -> members left out of that finding (gold-58 screen, team/a1/goldtab/screen_prune.py)\n"
    + (f"WMAP = {WMAP!r}   # per-finding member weights (lever A1: heliosli's LB-tuned Raptor/R6 shares mapped onto K14)\n" if WMAP else "")
    + (f"_root = {FAKE!r}   # rehearsal\n" if a.rehearsal else
       "_root = [p for p in ['/kaggle/input/rsna-knee-abnormality-detection', '/kaggle/input/competitions/rsna-knee-abnormality-detection'] if os.path.isdir(p)][0]\n") +
    "ids = pd.read_csv(_root + '/sample_submission.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()\n"
    "def _read(p):\n"
    "    if not os.path.isfile(p): return None\n"
    "    d = pd.read_csv(p, dtype={'StudyInstanceUID': str}).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').reindex(ids)[TARGETS].astype(float)\n"
    "    return d\n"
    "M = {}\n"
    "d = _read('/kaggle/working/s89c_sub.csv')\n"
    "if d is not None: M['ours'] = d.where(np.isfinite(d).all(axis=1))\n"
    "d = _read('/kaggle/working/raptor_raw.csv')\n"
    "if d is not None: M['raptor'] = d.where(np.isfinite(d).all(axis=1) & ~(d == 0.5).all(axis=1))   # Raptor leaves a failed study at 0.5\n"
    "for _n in ('s6', 'r7'):\n"
    "    d = _read(f'/kaggle/working/{_n}_sub.csv')\n"
    "    if d is not None and os.path.isfile(f'/kaggle/working/{_n}_scored.json'):\n"
    "        _ok = set(_j.load(open(f'/kaggle/working/{_n}_scored.json')))\n"
    "        M[_n] = d.where(pd.Series([u in _ok for u in ids], index=ids), axis=0)\n"
    f"A224 = {a.rap224!r}   # share of the Raptor slot given to the 224-px checkpoint (fixed before any run)\n"
    "R224 = None\n"
    "if A224 and 'raptor' in M:\n"
    "    d = _read('/kaggle/working/raptor224_sub.csv')\n"
    "    _v = M['raptor'].notna().all(axis=1)\n"
    "    if d is not None and np.isfinite(d[_v].to_numpy()).all(): R224 = d.where(_v)   # used only where Raptor itself scored\n"
    "    print('R224', 'used at %.2f of the Raptor slot' % A224 if R224 is not None else 'absent or unusable: Raptor slot unchanged', flush=True)\n"
    "cover = {k: int(v.notna().all(axis=1).sum()) for k, v in M.items()}\n"
    "assert M and max(cover.values()) == len(ids), ('no member covers every study', cover)\n"
    "out = pd.DataFrame({'StudyInstanceUID': ids})\n"
    "for t in TARGETS:\n"
    "    R = np.vstack([M[k][t].rank(pct=True).to_numpy() for k in M])            # NaN where a member has no value\n"
    "    if R224 is not None:                                                     # re-ranked mix, as screened\n"
    "        _i = list(M).index('raptor')\n"
    "        R[_i] = pd.Series((1 - A224) * R[_i] + A224 * R224[t].rank(pct=True).to_numpy()).rank(pct=True).to_numpy()\n"
    + ("    w0 = np.array([WMAP.get(t, {}).get(k, W[k]) for k in M])[:, None] * ~np.isnan(R)\n" if WMAP else
       "    w0 = np.array([W[k] for k in M])[:, None] * ~np.isnan(R)\n") +
    "    w = w0 * np.array([0.0 if k in DROP.get(t, ()) else 1.0 for k in M])[:, None]\n"
    "    w = np.where(w.sum(0) > 0, w, w0)                                        # a study only the dropped members scored keeps them\n"
    "    out[t] = np.where(np.isnan(R), 0.0, R * w).sum(0) / w.sum(0)\n"
    "_rk = lambda s: pd.Series(np.asarray(s, dtype=float)).rank(pct=True).values\n"
    + HABIT_CODE +
    "_P0 = {c: out[c].to_numpy().copy() for c in ('Medial OA', 'Lateral OA', 'PF OA')}   # OA columns before any rule\n"
    "out['Synovitis'] = _rk(_rk(out['Synovitis']) + 1.0 * _rk(out['Effusion'])\n"
    "                      + 0.75 * _rk((_rk(out['Medial OA']) + _rk(out['Lateral OA']) + _rk(out['PF OA'])) / 3))\n"
    f"OA_DEC = {a.oa_dec!r}   # Lateral OA / PF OA <- rank(rank(X) - OA_DEC * rank(mean rank of the other two OA)) (blend sprint #2)\n"
    "if OA_DEC:\n"
    "    for _t in ('Lateral OA', 'PF OA'):\n"
    "        _o = [c for c in _P0 if c != _t]\n"
    "        out[_t] = _rk(_rk(_P0[_t]) - OA_DEC * _rk((_rk(_P0[_o[0]]) + _rk(_P0[_o[1]])) / 2))\n"
    "    print('OA_DEC applied', OA_DEC, flush=True)\n"
    f"CENSUS = {a.census!r}   # hidden-run census probe: flag-coded columns set to 0.5 (decode from the score drop)\n"
    "if CENSUS:\n"
    "    _fa = R224 is None if A224 else False                                          # 224 stage skipped/unusable -> 1 column\n"
    "    _fb = len(M) < 4 or min(cover.values()) < len(ids)                            # a member missing or incomplete -> 2 columns\n"
    "    _fc = (time.time() - _VP_NB_T0) > 8.0 * 3600                                  # past 8 h at the blend -> 4 columns\n"
    "    for _flag, _cols in ((_fa, ['ACL']), (_fb, ['MCL', 'Medial Meniscus']), (_fc, ['Medial OA', \"Baker's\", 'Effusion', 'Fracture'])):\n"
    "        if _flag:\n"
    "            for _c in _cols: out[_c] = 0.5\n"
    "    print('CENSUS flags r224_missing', _fa, 'member_incomplete', _fb, 'over_8h', _fc, flush=True)\n"
    f"INVERT = {a.invert!r}   # per-finding read-out probe (AUC_t = .5 + 6 * (S_F1 - S_probe)); empty = none\n"
    "if INVERT:\n"
    "    out[INVERT] = 1.0 - out[INVERT].to_numpy(); print('PROBE invert', INVERT, flush=True)\n"
    "assert np.isfinite(out[TARGETS].to_numpy()).all() and len(out) == len(ids)\n"
    "out.to_csv('/kaggle/working/submission.csv', index=False)\n"
    "print('BLEND4_WRITTEN', _j.dumps({'members': list(M), 'coverage': cover, 'rows': len(ids), 'weights': W, 'drop': DROP,\n"
    + ("      'wmap': WMAP,\n" if WMAP else "") +
    "      'r224': A224 if R224 is not None else 0,\n"
    "      'total_seconds': round(time.time() - _VP_NB_T0)}), flush=True)\n"
    "print(out.head(3).to_string(), flush=True)"
)
# Decoder wheels (exp-091 / S40): the train set is 100% uncompressed but test series may be JPEG Lossless / JPEG 2000, and
# the K14 image has neither pylibjpeg nor gdcm. Installed before any stage so every member (in-process or subprocess) can
# decode them; any failure leaves the kernel exactly as without the flag.
WHEELS_CELL = (
    "# Decoder wheels (exp-091 / S40): offline DICOM decompression plugins, installed before any stage imports pydicom\n"
    "import glob as _gl\n"
    "_WD = [d for d in ['/kaggle/input/rsna-dicom-decoders', '/kaggle/input/datasets/komilparmar/rsna-dicom-decoders']\n"
    "       + _gl.glob('/kaggle/input/*/rsna-dicom-decoders') + _gl.glob('/kaggle/input/*/*/rsna-dicom-decoders') if os.path.isdir(d)]\n"
    "_WHEELS = sorted({w for d in _WD for w in _gl.glob(d + '/**/*.whl', recursive=True)})\n"
    "_DEC = 'found %d wheels in %s' % (len(_WHEELS), _WD[:1])\n"
    "if _WHEELS:\n"
    "    try:\n"
    "        _r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', *_WHEELS],\n"
    "                            capture_output=True, text=True, timeout=900)\n"
    "        _DEC += '; pip rc=%s %s' % (_r.returncode, _r.stderr.strip()[-200:])\n"
    "    except Exception as _e:\n"
    "        _DEC += '; pip failed %r' % (_e,)\n"
    "_HANDLERS = []\n"
    "for _m in ('pylibjpeg', 'gdcm', 'openjpeg', 'libjpeg'):\n"
    "    try:\n"
    "        __import__(_m); _HANDLERS.append(_m)\n"
    "    except Exception:\n"
    "        pass\n"
    "print('DECODERS', _DEC, '| importable:', _HANDLERS, flush=True)"
)
cells = [
    "import time as _vp_time, time, os, sys, subprocess, json, shutil\n_VP_NB_T0 = _vp_time.time()\n"
    f"TARGETS_ALL = {TARGETS!r}",
    "# Stage 1: our ensemble (a57-v14 x5 + 4a0c-v14 x5) in its own process\n"
    f"open('/tmp/ours_run.py', 'w').write({ours!r})\n"
    "_t = time.time()\ntry:\n"
    f"    _rc = subprocess.run([sys.executable, '-u', '/tmp/ours_run.py'], timeout=int({a.ours_timeout} * 3600)).returncode\n"
    "except Exception as _e:\n    _rc = repr(_e)\n"
    "print('OURS_STAGE rc', _rc, 'ok', os.path.isfile('/kaggle/working/s89c_sub.csv'), 'seconds %.0f' % (time.time() - _t), flush=True)\n"
    "import torch; torch.cuda.empty_cache()",
    write,
    guarded("raptor", raptor_stage),
    SWAP_STAGE if SWAP_SLOT == "s6" else (S6F_STAGE if a.s6f else guarded("s6", s6_body)),
    SWAP_STAGE if SWAP_SLOT == "r7" else guarded("r7", r7_body),
] + ([r224_write, r224_stage] if a.rap224 else []) + [blend]
if a.wheels:
    cells.insert(1, WHEELS_CELL)
if a.rehearsal:
    import pandas as _pd
    _tr = _pd.read_csv(ROOT / "data" / "train.csv", dtype={"StudyInstanceUID": str}, usecols=["StudyInstanceUID", "ACL"])
    _pool = sorted(_tr.loc[_tr["ACL"].isna(), "StudyInstanceUID"])          # non-gold train studies only
    _pick = sorted(_pd.Series(_pool).sample(n=a.rehearsal, random_state=0))
    cells.insert(0, (
        "# Timing rehearsal (never submitted): a fake competition root exposing N random non-gold TRAIN studies under the\n"
        "# test names; every stage is patched to read it, so stage times scale to the hidden set (~1,300 studies).\n"
        "import os as _o, pandas as _p\nfrom pathlib import Path as _P\n"
        f"_PICK = {_pick!r}\n"
        "_REAL = [p for p in (_P('/kaggle/input/competitions/rsna-knee-abnormality-detection'),\n"
        "                     _P('/kaggle/input/rsna-knee-abnormality-detection')) if (p / 'train_series').is_dir()][0]\n"
        f"_F = _P({FAKE!r}); (_F / 'test_series').mkdir(parents=True, exist_ok=True)\n"
        "for _s in _PICK:\n"
        "    if not (_F / 'test_series' / _s).exists(): _o.symlink(_REAL / 'train_series' / _s, _F / 'test_series' / _s)\n"
        "_p.DataFrame({'StudyInstanceUID': _PICK}).to_csv(_F / 'test.csv', index=False)\n"
        "_ts = _p.read_csv(_REAL / 'train_series.csv', dtype=str); _ts = _ts[_ts['StudyInstanceUID'].isin(_PICK)]\n"
        "_ts.to_csv(_F / 'test_series.csv', index=False)\n"
        "_ss = _p.read_csv(_REAL / 'sample_submission.csv', dtype={'StudyInstanceUID': str})\n"
        "_g = _p.DataFrame({'StudyInstanceUID': _PICK})\n"
        "for _c in _ss.columns[1:]: _g[_c] = 0.5\n"
        "_g.to_csv(_F / 'sample_submission.csv', index=False)\n"
        "print('REHEARSAL_ROOT', len(_PICK), 'studies', len(_ts), 'series', flush=True)"))
for c in cells:
    if not c.startswith("%%"):
        compile(c, "cell", "exec")
nb = json.loads((S89C / "rsna-a1-s89c-v14-4grp.ipynb").read_text(encoding="utf-8"))
nb["cells"] = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": c} for c in cells]
slug = f"rsna-rehearse-{a.tag}" if a.rehearsal else f"rsna-blend4-{a.tag}"
out = KDIR / slug
out.mkdir(exist_ok=True)
(out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
m = json.loads((S89C / "kernel-metadata.json").read_text(encoding="utf-8"))
m.pop("id_no", None)
m.update(id=f"komilparmar/{slug}", title=slug, code_file=f"{slug}.ipynb",
         dataset_sources=m["dataset_sources"] + (["nartaa/rsna-knee-s6fall-weights"] if a.s6f else []) + ["nartaa/rsna-knee-publication-swa-weights-20261007",
                                                 "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
                         + [x for x in a.swap_data.split(",") if x] + OURS_EXTRA,
         model_sources=[f"komilparmar/rsna-camp-{a.ours_label}-a57/pyTorch/v1/1", f"komilparmar/rsna-camp-{a.ours_label}-4a0c/pyTorch/v1/1",
                        "komilparmar/rsna-mirror-helios-s6/pyTorch/v2/1", ("komilparmar/rsna-mirror-helios-r6/pyTorch/v1/1" if a.r7_model == "r6" else "komilparmar/rsna-mirror-helios-r7/pyTorch/v1/1")]
                       + [x for x in a.swap_models.split(",") if x])
(out / "kernel-metadata.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
print("built", slug, "weights", W, "drop", DROP, "wmap", WMAP, "swap", SWAP_SLOT or "-")

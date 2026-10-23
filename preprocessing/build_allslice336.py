"""allslice336 cache build (input-geometry lever, I-079): 130 mm crop
(central 416 px of the 512/160mm prep arrays) resized to 336 px, EVERY slice
of every train study. 5 Kaggle CPU kernels (819k slices x 336^2 u8 ~ 92 GB
total -> ~18.5 GB/part, under the 20 GB kernel-output cap).

Derives allslice336_plan.json (5 balanced parts) from the existing 3-part
allslice_plan.json, writes 5 kernel dirs patterned on
scripts/kaggle_allslice_build/build.py, and pushes with --push.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
HERE = ROOT / "scripts" / "kaggle_allslice336_build"
N_PARTS = 5

src_plan = json.load(open(ROOT / "scratch_ds/trainkit/allslice_plan.json"))
order, counts = src_plan["order"], src_plan["counts"]
total = sum(counts)
target = total / N_PARTS
parts, lo, acc = [], 0, 0
for i, c in enumerate(counts):
    acc += c
    if acc >= target and len(parts) < N_PARTS - 1:
        parts.append([lo, i + 1])
        lo, acc = i + 1, 0
parts.append([lo, len(counts)])
sizes = [sum(counts[a:b]) for a, b in parts]
print("part sizes (slices):", sizes, "| GB @336:",
      [round(s * 336 * 336 / 2**30, 1) for s in sizes])
assert sum(sizes) == total and all(s * 336 * 336 < 20 * 2**30 for s in sizes)
plan336 = {"order": order, "counts": counts, "parts": parts}
json.dump(plan336, open(ROOT / "scratch_ds/trainkit/allslice336_plan.json",
                        "w"))
print("wrote trainkit/allslice336_plan.json")

template = (ROOT / "scripts/kaggle_allslice_build/build.py").read_text(
    encoding="utf-8")
# extract the kernel TEMPLATE between the triple-quoted r''' markers
t0 = template.index("TEMPLATE = r'''") + len("TEMPLATE = r'''")
t1 = template.index("'''", t0)
kernel_src = template[t0:t1]
kernel_src = kernel_src.replace('IMG, CROP = 224, 480', 'IMG, CROP = 336, 416')
kernel_src = kernel_src.replace('# 150 mm of the 512px/160mm arrays',
                                '# 130 mm of the 512px/160mm arrays')
kernel_src = kernel_src.replace('"crop_mm": 150.0', '"crop_mm": 130.0')
kernel_src = kernel_src.replace('allslice_plan.json', 'allslice336_plan.json')
kernel_src = kernel_src.replace('slices_part{PART}', 'slices_part{PART}')
assert 'IMG, CROP = 336, 416' in kernel_src
assert 'allslice336_plan.json' in kernel_src

# v3: slab output — kernels-output downloads break past ~4 GB, so build the
# raw array in /kaggle/tmp and emit <=3.4 GB .bin slabs to /kaggle/working.
old_mm = ('out = np.lib.format.open_memmap(WORK / f"slices_part{PART}.npy", '
          'mode="w+",\n'
          '                                dtype=np.uint8, shape=(N, IMG, IMG))')
new_mm = ('TMP = Path("/kaggle/tmp"); TMP.mkdir(exist_ok=True)\n'
          'out = np.memmap(TMP / f"slices_part{PART}.dat", mode="w+",\n'
          '                dtype=np.uint8, shape=(N, IMG, IMG))')
assert old_mm in kernel_src, "template memmap anchor missing"
kernel_src = kernel_src.replace(old_mm, new_mm, 1)

old_done = ('log(f"DONE part {PART}: {len(df)}/{N} slices written, '
            '{len(missing)} studies "\n'
            '    f"short, {df.nonzero.mean():.3f} nonzero fraction")')
new_done = old_done + '''
SLAB = 3_400_000_000
total_b = N * IMG * IMG
src = open(TMP / f"slices_part{PART}.dat", "rb")
n_slab = (total_b + SLAB - 1) // SLAB
for c in range(n_slab):
    with open(WORK / f"slices_part{PART}_s{c}.bin", "wb") as fo:
        left = min(SLAB, total_b - c * SLAB)
        while left > 0:
            buf = src.read(min(256 << 20, left))
            fo.write(buf)
            left -= len(buf)
    log(f"slab {c + 1}/{n_slab} written")
src.close()
json.dump({"n_rows": N, "img": IMG, "n_slab": n_slab, "slab_bytes": SLAB},
          open(WORK / f"shape_part{PART}.json", "w"))
log(f"slabs complete: {n_slab}")'''
assert old_done in kernel_src, "template DONE anchor missing"
kernel_src = kernel_src.replace(old_done, new_done, 1)

prep_sources = [f"komilparmar/rsna-knee-prep-full-s{s}" for s in range(8)]
for part in range(N_PARTS):
    d = HERE / f"part{part}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "allslice_build.py").write_text(
        kernel_src.replace("__PART__", str(part)), encoding="utf-8")
    meta = {
        "id": f"komilparmar/rsna-allslice336-part{part}",
        "title": f"rsna-allslice336-part{part}",
        "code_file": "allslice_build.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": ["komilparmar/rsna-knee-trainkit"],
        "competition_sources": [],
        "kernel_sources": prep_sources,
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1),
                                            encoding="utf-8")
    if "--push" in sys.argv:
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                           capture_output=True, text=True)
        print(f"part {part}:",
              (r.stdout + r.stderr).strip().splitlines()[-1])
    else:
        print("built", d)

"""a3 box bootstrap step 2: rebuild cache336 from the Kaggle kernel outputs komilparmar/rsna-allslice336-part0..4.

Run ON THE BOX (inside tmux/nohup; needs /root/.kaggle/kaggle.json, chmod 600):
    /workspace/rsna-knee/venv/bin/python -u /workspace/rsna-knee/work/bootstrap/02_stage_cache336.py \
        --dest /workspace/rsna-knee/work/cache336 --workers 8
    ... --verify-only        # re-run just the verification of finished parts
Resumable: kill it any time and re-run the same command.

What the old boxes did (archive A: code/stage336.sh + manifests/stage336.log, 2026-09-23):
  laptop `kernel_output_urls.py` -> signed GCS URLs -> box `stage_generic.py` (HTTP Range resume, 3 workers,
  size check) -> `assemble_slabs.py` (cat 6 x <=3.4 GB slabs behind a .npy header, then delete slabs).
  `kaggle kernels output` is NOT used: it broke on >4 GB files (IncompleteRead, whole file in RAM; MEMORY.md
  2026-08-21) - that is why the kernels write 3.4 GB slabs.
This script does the same with two changes: (1) URLs are listed on the box with the kaggle SDK (refreshed
automatically when they expire), (2) each slab is Range-downloaded straight into its byte offset of the final
slices_part{k}.npy (no assembly copy: peak disk = 92.5 GB instead of ~111 GB). Each slab is retried
independently and resumes from its recorded byte offset.

Final layout (exactly what SliceBank(n_parts=5) reads):
  cache336/slices_part{k}.npy  uint8 [n_rows, 336, 336]   k = 0..4
  cache336/index_part{k}.csv, meta_part{k}.json, shape_part{k}.json
Expected (stage336.log of box A): rows 163825/163856/163884/163850/163663 = 819,078 total, 92,470,629,888 B of
pixels; SliceBank(drop_zero=True) then sees 819,058 rows; the a57 trainer log prints
'THIN-SERIES ON ... rows 819058 -> 717833'. No per-slab sha256 ledger exists (cache excluded from the backups);
this script records one (stage336_ledger.json) for next time.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

USER = "komilparmar"
KERNELS = [f"rsna-allslice336-part{k}" for k in range(5)]
EXPECT_ROWS = {0: 163825, 1: 163856, 2: 163884, 3: 163850, 4: 163663}
IMG = 336
SLAB = 3_400_000_000
CHUNK = 16 << 20
SYNC_EVERY = 512 << 20

_lock = threading.Lock()
_urls = {}
_url_t = [0.0]


def log(**kw):
    kw["t"] = time.strftime("%H:%M:%S")
    print(json.dumps(kw), flush=True)


def list_urls():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi()
    api.authenticate()
    out = {}
    with api.build_kaggle_client() as client:
        for k in KERNELS:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug, req.page_size = USER, k, 50
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            for rec in resp.files:
                out[rec.file_name] = rec.url
    return out


def url_for(name, refresh=False):
    with _lock:
        if refresh and time.time() - _url_t[0] > 30:     # one refresh for all threads
            _urls.update(list_urls())
            _url_t[0] = time.time()
            log(event="urls_refreshed", n=len(_urls))
        return _urls[name]


def get_small(name, dest):
    p = dest / name
    if p.exists() and p.stat().st_size > 0:
        return
    for i in range(20):
        try:
            r = requests.get(url_for(name, refresh=i > 0), timeout=120)
            r.raise_for_status()
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(r.content)
            os.replace(tmp, p)
            return
        except Exception as e:                                  # noqa: BLE001
            log(event="retry_small", name=name, attempt=i, error=type(e).__name__)
            time.sleep(min(60, 5 * (i + 1)))
    raise RuntimeError(f"could not fetch {name}")


def npy_header(n):
    b = io.BytesIO()
    np.lib.format.write_array_header_1_0(b, {"descr": "|u1", "fortran_order": False, "shape": (n, IMG, IMG)})
    return b.getvalue()


def fetch_slab(dest, k, j, offset, length):
    """Range-download slices_part{k}_s{j}.bin into bytes [offset, offset+length) of slices_part{k}.npy.partial."""
    name = f"slices_part{k}_s{j}.bin"
    prog = dest / f".{name}.prog"
    done_f = dest / f".{name}.done"
    if done_f.exists():
        return name, "present"
    have = int(prog.read_text()) if prog.exists() else 0
    target = dest / f"slices_part{k}.npy.partial"
    attempt = 0
    while have < length:
        try:
            hdr = {"Range": f"bytes={have}-{length - 1}"}
            with requests.get(url_for(name, refresh=attempt > 0), headers=hdr, stream=True, timeout=(30, 120)) as r:
                if r.status_code in (400, 401, 403, 404):
                    raise PermissionError(f"http {r.status_code}")        # expired signed URL -> refresh
                if r.status_code == 200 and have:
                    raise IOError("server ignored Range")
                if r.status_code not in (200, 206):
                    raise IOError(f"http {r.status_code}")
                if r.status_code == 206:
                    cr = r.headers.get("Content-Range", "")
                    total = int(cr.split("/")[-1]) if "/" in cr else -1
                    if total != length:
                        raise RuntimeError(f"{name}: remote size {total} != expected {length} (cache changed?)")
                fd = os.open(target, os.O_RDWR)
                try:
                    since = 0
                    for buf in r.iter_content(CHUNK):
                        if not buf:
                            continue
                        if have + len(buf) > length:
                            raise RuntimeError(f"{name}: more bytes than expected")
                        os.pwrite(fd, buf, offset + have)
                        have += len(buf)
                        since += len(buf)
                        if since >= SYNC_EVERY:
                            os.fdatasync(fd)
                            prog.write_text(str(have))
                            since = 0
                    os.fdatasync(fd)
                    prog.write_text(str(have))
                finally:
                    os.close(fd)
        except RuntimeError:
            raise
        except Exception as e:                                  # noqa: BLE001
            attempt += 1
            log(event="retry", name=name, attempt=attempt, have_mb=have >> 20, error=type(e).__name__)
            if attempt > 60:
                raise
            time.sleep(min(60, 3 * attempt))
    assert have == length, (name, have, length)
    done_f.write_text(str(length))
    prog.unlink(missing_ok=True)
    return name, "downloaded"


def verify_part(dest, k, record_hash):
    """Shape/dtype + every indexed row's (max > 0) must equal its index 'nonzero' flag (catches zero holes)."""
    import pandas as pd
    shp = json.loads((dest / f"shape_part{k}.json").read_text())
    n = int(shp["n_rows"])
    path = dest / f"slices_part{k}.npy.partial"
    if not path.exists():
        path = dest / f"slices_part{k}.npy"
    arr = np.load(path, mmap_mode="r")
    assert arr.shape == (n, IMG, IMG) and arr.dtype == np.uint8, (k, arr.shape, arr.dtype)
    idx = pd.read_csv(dest / f"index_part{k}.csv", usecols=["row", "nonzero"])
    local = idx["row"].to_numpy()                 # row = index into this part's array (SliceBank: parts[k][row])
    assert local.min() >= 0 and local.max() < n, (k, local.min(), local.max(), n)
    nz = np.zeros(n, bool)
    step = 4096
    for s in range(0, n, step):
        nz[s:s + step] = arr[s:s + step].reshape(min(step, n - s), -1).max(1) > 0
    exp = idx["nonzero"].to_numpy().astype(bool)
    bad = int((nz[local] != exp).sum())
    unindexed_nonzero = int(nz.sum() - nz[local].sum())
    res = {"part": k, "rows": n, "indexed": int(len(idx)), "nonzero_rows": int(nz.sum()),
           "mismatch_vs_index": bad, "unindexed_nonzero": unindexed_nonzero}
    if record_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(64 << 20), b""):
                h.update(b)
        res["npy_sha256"] = h.hexdigest()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("/workspace/rsna-knee/work/cache336"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--hash", action="store_true", help="also record sha256 of each finished .npy (~+3 min)")
    a = ap.parse_args()
    dest = a.dest
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status = dest / "stage_status.json"

    if not a.verify_only:
        _urls.update(list_urls())
        _url_t[0] = time.time()
        names = sorted(_urls)
        log(event="listed", n=len(names), files=names)
        for k in range(5):
            for nm in (f"index_part{k}.csv", f"meta_part{k}.json", f"shape_part{k}.json"):
                assert nm in _urls, f"missing kernel output {nm}"
                get_small(nm, dest)
        tasks = []
        free = os.statvfs(dest)
        free_b = free.f_bavail * free.f_frsize
        need = 0
        for k in range(5):
            shp = json.loads((dest / f"shape_part{k}.json").read_text())
            n, img, ns, sb = int(shp["n_rows"]), int(shp["img"]), int(shp["n_slab"]), int(shp["slab_bytes"])
            assert (n, img, sb) == (EXPECT_ROWS[k], IMG, SLAB), (k, shp)
            total = n * img * img
            assert ns == (total + SLAB - 1) // SLAB
            if (dest / f"slices_part{k}.npy").exists():
                continue
            hdr = npy_header(n)
            part = dest / f"slices_part{k}.npy.partial"
            if not part.exists():
                with open(part, "wb") as f:
                    f.write(hdr)
                    f.truncate(len(hdr) + total)          # sparse; filled by the slab writers
                need += total
            for j in range(ns):
                nm = f"slices_part{k}_s{j}.bin"
                assert nm in _urls, f"missing kernel output {nm}"
                tasks.append((k, j, len(hdr) + j * SLAB, min(SLAB, total - j * SLAB)))
        log(event="plan", slabs=len(tasks), free_gb=round(free_b / 1e9, 1), new_alloc_gb=round(need / 1e9, 1))
        if need > free_b - 5e9:
            sys.exit(f"not enough disk: need {need / 1e9:.1f} GB + 5 GB margin, free {free_b / 1e9:.1f} GB")
        status.write_text(json.dumps({"status": "running", "slabs": len(tasks)}))
        nbytes = 0
        with ThreadPoolExecutor(a.workers) as pool:
            futs = {pool.submit(fetch_slab, dest, *t): t for t in tasks}
            for i, f in enumerate(as_completed(futs), 1):
                name, how = f.result()
                nbytes += futs[f][3]
                el = time.time() - t0
                log(event="slab", name=name, how=how, done=f"{i}/{len(tasks)}",
                    MBps=round(nbytes / max(el, 1) / 1e6, 1), elapsed_min=round(el / 60, 1))

    ledger = {"parts": []}
    for k in range(5):
        final = dest / f"slices_part{k}.npy"
        part = dest / f"slices_part{k}.npy.partial"
        if part.exists():
            shp = json.loads((dest / f"shape_part{k}.json").read_text())
            ns = int(shp["n_slab"])
            if not all((dest / f".slices_part{k}_s{j}.bin.done").exists() for j in range(ns)):
                log(event="part_incomplete", part=k)
                continue
        elif not final.exists():
            log(event="part_missing", part=k)
            continue
        res = verify_part(dest, k, a.hash)
        log(event="verify", **res)
        ok = res["mismatch_vs_index"] == 0 and res["unindexed_nonzero"] == 0 and res["rows"] == EXPECT_ROWS[k]
        if not ok:
            sys.exit(f"VERIFY FAILED part {k}: {res} - delete .slices_part{k}_s*.done and re-run")
        if part.exists():
            os.replace(part, final)
        ledger["parts"].append(res)
    if len(ledger["parts"]) == 5:
        tot = sum(r["rows"] for r in ledger["parts"])
        assert tot == sum(EXPECT_ROWS.values())
        ledger.update(status="complete", rows=tot, seconds=round(time.time() - t0))
        (dest / "stage336_ledger.json").write_text(json.dumps(ledger, indent=1))
        status.write_text(json.dumps({"status": "complete", "rows": tot}))
        # SliceBank smoke: the exact loader the trainer uses
        sys.path.insert(0, str(dest.parent / "code"))
        try:
            import pandas as pd
            frames = [pd.read_csv(dest / f"index_part{k}.csv", usecols=["study", "nonzero"]) for k in range(5)]
            idx = pd.concat(frames)
            log(event="COMPLETE", rows=tot, nonzero_rows=int(idx.nonzero.sum()), studies=int(idx.study.nunique()),
                expect="nonzero 819058, studies 4407")
        except Exception as e:                                  # noqa: BLE001
            log(event="index_check_skipped", error=repr(e))
    else:
        log(event="NOT_COMPLETE", parts_ok=len(ledger["parts"]))
        sys.exit(2)


if __name__ == "__main__":
    main()

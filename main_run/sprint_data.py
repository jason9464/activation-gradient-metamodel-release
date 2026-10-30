"""Fetch student-run files from HF ``jason9464/sprint-data-2610`` (Jaeseung, 2026-10-26): every ``student_runs/`` run of
the black-box timing test and candidate scoring, as 16 tar parts. ``data/part_00.tar`` holds ``student_runs/README.md``
(the file formats; the same text as in ``agm-readers`` ``student_runs/``) and ``student_runs/MANIFEST.txt`` (which part
holds which run, and every file's size and sha256); parts 01-15 hold one run directory each (12.7 GiB in all).

Two modes:
  members (default)  read the selected files straight out of the remote tars with HTTP range requests (tarfile seeks
                     over the members it skips), so only the selected files reach the disk. About a minute per part.
  whole              download a part, check its sha256, unpack the selected files, delete the tar (needs the part's
                     size in free disk, at most 1.4 GB, at a time). For GPU boxes that need the npz inputs.
Every written file is checked against the manifest's size and sha256; a file already present with the right sha256
is kept (a fetch can be rerun). Paths outside ``student_runs/`` and non-regular members are refused.

Selections (``--what``): ``blackbox`` = evals.jsonl, eval_samples/ and the reader judge and reading files of
``blackbox_timing.READER_SOURCES``; ``inputs`` = every ``e2/*_inputs*.npz`` (the reader inputs); ``all`` = everything.
  python -m main_run.sprint_data --dest data/sprint_data --what blackbox      # runs root: data/sprint_data/student_runs
  python -m main_run.sprint_data --dest <dest> --what inputs --mode whole [--runs newset_dolphin_s1,...]
The repo was public on 2026-10-26; ``HF_TOKEN`` is used when set.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Callable

REPO = "jason9464/sprint-data-2610"
PREFIX = "student_runs/"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [sprint_data] {msg}", flush=True)


def read_manifest(path: Path) -> tuple[dict[str, dict], dict[str, tuple[int, str]]]:
    """({part: {run, tar_bytes, tar_sha256}}, {path under student_runs/: (bytes, sha256)})."""
    parts, files = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) == 6 and cols[0].startswith("part_") and cols[0] != "part_00":
            parts[cols[0]] = dict(run=cols[1], tar_bytes=int(cols[4]), tar_sha256=cols[5])
        elif len(cols) == 3 and cols[0].startswith(PREFIX):
            files[cols[0][len(PREFIX):]] = (int(cols[1]), cols[2])
    if not parts or not files:
        raise ValueError(f"{path}: no parts or no files found; not a sprint-data manifest")
    return parts, files


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_rel(name: str) -> str:
    """The member path under student_runs/, refusing absolute paths, '..' and other prefixes."""
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not name.startswith(PREFIX):
        raise ValueError(f"unsafe tar member {name!r}")
    return name[len(PREFIX):]


def blackbox_selector() -> Callable[[str], bool]:
    from main_run.blackbox_timing import READER_SOURCES

    reader = {f for srcs in READER_SOURCES.values() for s in srcs for f in (*s.treated, *(x for c in s.controls for x in c))}

    def sel(rel: str) -> bool:
        parts = rel.split("/")
        return (len(parts) == 2 and parts[1] == "evals.jsonl") or (len(parts) == 3 and parts[1] == "eval_samples") \
            or rel in reader
    return sel


SELECTORS: dict[str, Callable[[], Callable[[str], bool]]] = {
    "blackbox": blackbox_selector,
    "inputs": lambda: (lambda rel: fnmatch.fnmatch(rel, "*/e2/*_inputs*.npz") or fnmatch.fnmatch(rel, "*/e2/inputs*.npz")),
    "all": lambda: (lambda rel: True),
}


def _write_member(tf: tarfile.TarFile, m: tarfile.TarInfo, dest_root: Path, rel: str,
                  expect: tuple[int, str]) -> None:
    out = dest_root / PREFIX / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    h = hashlib.sha256()
    src = tf.extractfile(m)
    if src is None:
        raise ValueError(f"cannot read member {m.name}")
    with open(tmp, "wb") as f:
        for b in iter(lambda: src.read(1 << 20), b""):
            h.update(b)
            f.write(b)
    if (tmp.stat().st_size, h.hexdigest()) != expect:
        tmp.unlink()
        raise ValueError(f"{rel}: size or sha256 differs from the manifest")
    os.replace(tmp, out)


def _extract(tf: tarfile.TarFile, dest: Path, select: Callable[[str], bool], files: dict[str, tuple[int, str]],
             have: set[str]) -> int:
    n = 0
    for m in tf:
        if not m.isfile():
            if m.isdir():
                continue
            raise ValueError(f"non-regular tar member {m.name!r}")
        rel = safe_rel(m.name)
        if not select(rel) or rel in have:
            continue
        if rel not in files:
            raise ValueError(f"{rel} is not in the manifest")
        _write_member(tf, m, dest, rel, files[rel])
        n += 1
    return n


def fetch(dest: Path, *, what: str = "blackbox", runs: list[str] | None = None, mode: str = "members",
          token: str | None = None, retries: int = 4) -> Path:
    """Fetch the selected files into ``dest/student_runs/``; returns that runs root."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    dest = Path(dest)
    token = token or os.environ.get("HF_TOKEN")
    p0 = hf_hub_download(REPO, "data/part_00.tar", token=token)
    with tarfile.open(p0) as tf:
        for m in tf:
            if m.isfile():
                rel = safe_rel(m.name)
                out = dest / PREFIX / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tf.extractfile(m).read())
    parts, files = read_manifest(dest / PREFIX / "MANIFEST.txt")
    select = SELECTORS[what]()
    by_run = {v["run"]: k for k, v in parts.items()}
    want = runs or sorted(by_run)
    bad = [r for r in want if r not in by_run]
    if bad:
        raise SystemExit(f"runs {bad} are not in the manifest (has {sorted(by_run)})")
    for run in want:
        part = by_run[run]
        sel = [rel for rel in files if rel.startswith(run + "/") and select(rel)]

        def present() -> set[str]:
            return {rel for rel in sel if (dest / PREFIX / rel).exists()
                    and (dest / PREFIX / rel).stat().st_size == files[rel][0]
                    and sha256_file(dest / PREFIX / rel) == files[rel][1]}

        have = present()
        if len(have) == len(sel):
            log(f"{run} ({part}): {len(sel)} selected files already present")
            continue
        t0 = time.time()
        if mode == "members":
            # A network failure mid-part leaves the HTTP client closed; retry the part on a fresh file system, keeping
            # the files already written (each was checked before it was kept).
            for attempt in range(1, retries + 1):
                try:
                    fs = HfFileSystem(token=token, skip_instance_cache=True)
                    with fs.open(f"{REPO}/data/{part}.tar", "rb", block_size=1 << 16) as f:
                        n = _extract(tarfile.open(fileobj=f, mode="r:"), dest, select, files, have)
                    break
                except (OSError, RuntimeError, tarfile.ReadError) as e:
                    if attempt == retries:
                        raise
                    log(f"{run} ({part}): attempt {attempt} failed ({type(e).__name__}: {e}); retrying")
                    time.sleep(5 * attempt)
                    have = present()
        elif mode == "whole":
            tar = Path(hf_hub_download(REPO, f"data/{part}.tar", token=token, local_dir=str(dest / ".parts")))
            try:
                if tar.stat().st_size != parts[part]["tar_bytes"] or sha256_file(tar) != parts[part]["tar_sha256"]:
                    raise ValueError(f"{tar}: size or sha256 differs from the manifest")
                with tarfile.open(tar) as tf:
                    n = _extract(tf, dest, select, files, have)
            finally:
                tar.unlink(missing_ok=True)
        else:
            raise ValueError(f"mode {mode!r}")
        missing = [rel for rel in sel if not (dest / PREFIX / rel).exists()]
        if missing:
            raise SystemExit(f"{run}: {len(missing)} selected files not found in {part}: {missing[:3]}")
        log(f"{run} ({part}): {n} files written, {len(have)} kept, {time.time() - t0:.0f}s")
    return dest / "student_runs"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--what", choices=sorted(SELECTORS), default="blackbox")
    ap.add_argument("--runs", default=None, help="comma list of run directories (default: every run)")
    ap.add_argument("--mode", choices=["members", "whole"], default="members")
    a = ap.parse_args(argv)
    runs = [r for r in a.runs.split(",") if r] if a.runs else None
    root = fetch(a.dest, what=a.what, runs=runs, mode=a.mode)
    log(f"runs root {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

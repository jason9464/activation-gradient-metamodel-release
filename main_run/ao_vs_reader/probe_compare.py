#!/usr/bin/env python
"""Linear probes on gradient vs activation bundle vectors of the same behaviour bundles (idea 2 of the follow-up list:
do gradients carry information the activations of the same documents do not?). CPU only.

  python main_run/ao_vs_reader/probe_compare.py build [--layers 5,10,...] [--out D] [--raw R] [--keep-raw]
  python main_run/ao_vs_reader/probe_compare.py crosscheck --layer 20 [--out D] [--raw R]
  python main_run/ao_vs_reader/probe_compare.py probe [--out D] [--tasks ...] [--boot 1000]

Source: HF jason9464/sprint-data-2610 vectors/behaviour_bundles_v1/docs, the behaviour extraction
(Qwen3-14B, no adapter). Every document has a ``full`` gradient sum (+dL/dh, all positions, 40 layers) and a ``full``
activation sum (+h, position 0 excluded, layers 5, 10, 15, 20, 25, 30, 35, 38) from the same forward and backward
pass. Labels: me-r/behaviour-bundles-v1 bundles.jsonl at the reader dir's revision (BEH_REVISION): 4,540 bundles,
each document in exactly one bundle.

build: per layer, writes <out>/vec/{act,grad}_L<L>.npy (fp16 unit rows) and _norm.npy (fp32), rows in bundles.jsonl
order (<out>/bundles.parquet holds the labels in that order).
  act   downloads the two shards' full_act_L<L> into --raw, sums each bundle's documents (unit x norm), deletes the raw
        files unless --keep-raw (~565 MB per layer).
  grad  the readers' own input: rows 25,217-29,756 (the 4,540 behaviour bundles, contiguous) of the reader dir's
        bundle_tokensum_L<L>.npy, read with one HTTP range request (46.5 MB per layer), norms from its
        bundle_index.parquet. Sign -dL/dh, the descent direction (meta.json of the reader dir).
  actw  exploratory (plan section 7): from the same raw files as act, each document's activation direction weighted
        by that document's gradient norm at the same layer, sum_d ||g_d|| a_d/||a_d||. It keeps the activation's
        directions and takes the gradient's per-document weighting.
  --grad-from-docs  also sums the documents' full_grad_L<L> the way act is summed and writes graddocs_L<L>
        (+dL/dh), for crosscheck.

crosscheck: graddocs_L<L> against grad_L<L>. Cosine -1 and norm ratio 1 mean that summing the document store here gives
the readers' bundle vectors, so the act vectors (same code path) are summed over the same documents the same way.

probe: see the plan. Kernel ridge (linear kernel on per-dimension standardised unit rows, so exact in the full
5,120 dimensions), intercept unpenalised, regularisation chosen on an inner grouped split, outer grouped 5-fold CV.
Groups keep an organism's bundles, and a harmful/benign persona pair, in one fold. Retrieval tasks use cosine on the
same features with no fitting.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = Path(os.environ.get("PC_OUT", REPO / "artifacts" / "probe_compare_out"))
RAW = Path(os.environ.get("PC_RAW", OUT / "raw"))

SPRINT_REPO = "jason9464/sprint-data-2610"
DOCS = "vectors/behaviour_bundles_v1/docs"
BUNDLE = "vectors/behaviour_bundles_v1/bundle"
SHARDS = ("shard0", "shard1")
BEH_REPO = "me-r/behaviour-bundles-v1"
BEH_REVISION = "09728f63f70efc3091ae4a7845a97b29394fcba5"  # fetch_hf.BEH_REVISION
BEH_FILE = "behaviour_bundles_v1/bundles.jsonl"
LAYERS = (5, 10, 15, 20, 25, 30, 35, 38)  # the store's activation layers; gradients are taken at the same ones
QUANTITIES = ("act", "grad")  # store arrays; the probe also reads "graddocs" only in crosscheck
LABEL_COLS = ("bundle_id", "split", "family", "unit_id", "behaviour_id", "behaviour_text", "trigger", "evidence",
              "pair_id", "n_docs", "n_label", "n_tokens", "n_shows", "n_trigger")
BLOCK = 256


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [probe_compare] {msg}", flush=True)


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for env in (REPO / ".env",):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def fetch(path: str, raw: Path, repo=SPRINT_REPO, repo_type="model", revision=None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type=repo_type, revision=revision, token=token(), local_dir=raw))


# ---------------------------------------------------------------- labels and store index
def load_bundles(raw: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in fetch(BEH_FILE, raw / "beh", BEH_REPO, "dataset", BEH_REVISION).open()]
    df = pd.DataFrame(rows)
    for c in ("n_docs", "n_label", "n_tokens", "n_shows", "n_trigger"):
        df[c] = df[c].astype(int)
    for c in ("trigger", "pair_id"):
        df[c] = df[c].astype(str)
    if df.bundle_id.duplicated().any():
        raise SystemExit("duplicate bundle ids in bundles.jsonl")
    return df


def load_index(raw: Path) -> pd.DataFrame:
    parts = []
    for s in SHARDS:
        ix = pd.read_parquet(fetch(f"{DOCS}/{s}/index.parquet", raw))
        plan = json.loads(fetch(f"{DOCS}/{s}/plan.json", raw).read_text())
        if plan["full_skip_pos0"] != {"grad": False, "act": True}:
            raise SystemExit(f"{s}: unexpected position-0 handling {plan['full_skip_pos0']}")
        parts.append(ix.assign(shard=s))
    ix = pd.concat(parts, ignore_index=True)
    if ix.doc_id.duplicated().any():
        raise SystemExit("a document has more than one row (chunked); bundle_sums assumes one row per document")
    return ix


def doc_positions(bundles: pd.DataFrame, ix: pd.DataFrame) -> list:
    pos_of = pd.Series(np.arange(len(ix)), index=ix.doc_id)
    out = []
    for b, docs in zip(bundles.bundle_id, bundles.doc_ids):
        missing = [d for d in docs if d not in pos_of.index]
        if missing or len(docs) == 0:
            raise SystemExit(f"bundle {b}: {len(missing)} documents missing from the store")
        out.append(pos_of[docs].to_numpy(np.int64))
    return out


def bundle_sums(pos_lists: list, ix: pd.DataFrame, arrays: dict, norm: np.ndarray) -> np.ndarray:
    """[bundles, d] fp32: per bundle, the sum over its documents of unit row x stored norm. arrays: shard -> mmap."""
    shard = ix["shard"].to_numpy()
    row = ix["row"].to_numpy(np.int64)
    d = next(iter(arrays.values())).shape[1]
    out = np.empty((len(pos_lists), d), np.float32)
    for b0 in range(0, len(pos_lists), BLOCK):
        chunk = pos_lists[b0:b0 + BLOCK]
        pos = np.concatenate(chunk)
        vec = np.empty((len(pos), d), np.float32)
        for s in np.unique(shard[pos]):
            m = shard[pos] == s
            r = row[pos][m]
            order = np.argsort(r)
            part = np.empty((len(r), d), np.float32)
            part[order] = arrays[s][r[order]]
            vec[m] = part
        vec *= norm[pos][:, None]
        starts = np.concatenate([[0], np.cumsum([len(p) for p in chunk])[:-1]]).astype(np.int64)
        out[b0:b0 + len(chunk)] = np.add.reduceat(vec, starts, axis=0)
    if not np.isfinite(out).all():
        raise SystemExit("NaN/Inf in a bundle sum")
    return out


def save_vec(out: Path, q: str, L: int, v: np.ndarray) -> None:
    n = np.linalg.norm(v, axis=1)
    if (n == 0).any():
        raise SystemExit(f"{q} L{L}: {(n == 0).sum()} zero bundle vectors")
    (out / "vec").mkdir(parents=True, exist_ok=True)
    np.save(out / "vec" / f"{q}_L{L}.tmp.npy", (v / n[:, None]).astype(np.float16))
    np.save(out / "vec" / f"{q}_L{L}_norm.npy", n.astype(np.float32))
    os.replace(out / "vec" / f"{q}_L{L}.tmp.npy", out / "vec" / f"{q}_L{L}.npy")


def load_vec(out: Path, q: str, L: int) -> tuple:
    return (np.load(out / "vec" / f"{q}_L{L}.npy").astype(np.float32),
            np.load(out / "vec" / f"{q}_L{L}_norm.npy"))


def cmd_build(a) -> int:
    out, raw = Path(a.out), Path(a.raw)
    out.mkdir(parents=True, exist_ok=True)
    bundles = load_bundles(raw)
    ix = load_index(raw)
    pos_lists = doc_positions(bundles, ix)
    lab = bundles[list(LABEL_COLS)].copy()
    lab["n_store_tokens"] = [int(ix.n_tokens.to_numpy()[p].sum()) for p in pos_lists]
    lab["n_store_label"] = [int(ix.n_label.to_numpy()[p].sum()) for p in pos_lists]
    lab.to_parquet(out / "bundles.parquet")
    # per-document norms, for the magnitude baselines and the concentration statistic
    dn = {f"norm_full_{q}_L{L}": ix[f"norm_full_{q}_L{L}"].to_numpy(np.float32) for q in QUANTITIES for L in LAYERS}
    pd.DataFrame(dict(doc_id=ix.doc_id, bundle_row=np.repeat(np.arange(len(pos_lists)), [len(p) for p in pos_lists])
                      [np.argsort(np.concatenate(pos_lists))], n_tokens=ix.n_tokens, n_label=ix.n_label, **dn)
                 ).to_parquet(out / "doc_norms.parquet")
    log(f"{len(bundles)} bundles, {len(ix)} documents; labels in {out / 'bundles.parquet'}")
    bi = pd.read_parquet(fetch(f"{BUNDLE}/bundle_index.parquet", raw)).set_index("bundle_id")
    grad_rows = bi.loc[lab.bundle_id, "row"].to_numpy(np.int64)
    for L in a.layers:
        done = lambda q: (out / "vec" / f"{q}_L{L}.npy").exists() and not a.overwrite
        if not done("grad"):
            t0 = time.time()
            v = bundle_rows_ranged(f"{BUNDLE}/bundle_tokensum_L{L}.npy", grad_rows)
            n = bi.loc[lab.bundle_id, f"norm_tokensum_L{L}"].to_numpy(np.float32)
            save_vec(out, "grad", L, v * n[:, None])
            log(f"L{L} grad (reader dir rows): {time.time() - t0:.0f} s")
        # act and actw come from the same raw files; graddocs from the gradient ones
        todo = {"act": [q for q in ("act", "actw") if not done(q)],
                "grad": ["graddocs"] if a.grad_from_docs and not done("graddocs") else []}
        free = shutil.disk_usage(out).free / 1e9
        if any(todo.values()) and free < 1.5:
            raise SystemExit(f"only {free:.1f} GB free; one quantity of one layer needs ~0.6 GB of raw files")
        for arr_q, outs in todo.items():
            if not outs:
                continue
            t0 = time.time()
            files = {s: fetch(f"{DOCS}/{s}/full_{arr_q}_L{L}.npy", raw) for s in SHARDS}
            t1 = time.time()
            arrays = {s: np.load(p, mmap_mode="r") for s, p in files.items()}
            for s, arr in arrays.items():
                n = int((ix.shard == s).sum())
                if arr.shape[0] != n:
                    raise SystemExit(f"{files[s]}: {arr.shape[0]} rows, index has {n}")
            for q in outs:
                # actw: each document's activation direction weighted by its own gradient norm at the same layer
                w = ix[f"norm_full_{'grad' if q == 'actw' else arr_q}_L{L}"].to_numpy(np.float32)
                v = bundle_sums(pos_lists, ix, arrays, w)
                save_vec(out, q, L, v)
                log(f"L{L} {q}: download {t1 - t0:.0f} s, sum {time.time() - t1:.0f} s, "
                    f"median bundle norm {np.median(np.linalg.norm(v, axis=1)):.4g}")
            del arrays
            if not a.keep_raw:
                for p in files.values():
                    p.unlink()
    return 0


def bundle_rows_ranged(path: str, rows: np.ndarray) -> np.ndarray:
    """fp32 [len(rows), d] of a .npy on the HF repo, reading only the contiguous byte range rows.min()..rows.max()."""
    import io

    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(SPRINT_REPO, path)
    auth = {"Authorization": f"Bearer {token()}"}
    r = requests.get(url, headers={**auth, "Range": "bytes=0-1023"}, timeout=120)
    r.raise_for_status()
    f = io.BytesIO(r.content)
    ver = np.lib.format.read_magic(f)
    _read = getattr(np.lib.format, "_read_array_header", None)
    if _read is not None:
        shape, fortran, dtype = _read(f, ver)
    elif ver == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif ver == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise SystemExit(f"{path}: unsupported npy header version {ver}")
    if fortran or dtype != np.float16 or len(shape) != 2:
        raise SystemExit(f"{path}: unexpected array header {shape} {dtype} fortran={fortran}")
    lo, hi = int(rows.min()), int(rows.max()) + 1
    if hi - lo != len(np.unique(rows)) or hi > shape[0]:
        raise SystemExit(f"{path}: rows are not one contiguous block inside {shape[0]} rows")
    row_bytes = shape[1] * 2
    start = f.tell() + lo * row_bytes
    for attempt in range(3):
        r = requests.get(url, headers={**auth, "Range": f"bytes={start}-{start + (hi - lo) * row_bytes - 1}"},
                         timeout=1800)
        if r.status_code == 206 and len(r.content) == (hi - lo) * row_bytes:
            break
        log(f"{path}: range read attempt {attempt + 1} got {r.status_code}, {len(r.content)} bytes")
    else:
        raise SystemExit(f"{path}: range read failed")
    block = np.frombuffer(r.content, np.float16).reshape(hi - lo, shape[1])
    return block[rows - lo].astype(np.float32)


def cmd_crosscheck(a) -> int:
    out, L = Path(a.out), a.layer
    mine, my_norm = load_vec(out, "graddocs", L)
    theirs, their_norm = load_vec(out, "grad", L)
    cos = (mine * theirs).sum(1)
    ratio = my_norm / their_norm
    rep = dict(layer=L, n=len(cos), cos_max=float(cos.max()), cos_median=float(np.median(cos)),
               norm_ratio_min=float(ratio.min()), norm_ratio_max=float(ratio.max()))
    log(json.dumps(rep))
    (out / f"crosscheck_L{L}.json").write_text(json.dumps(rep, indent=1))
    # graddocs is +dL/dh, grad is the readers' -dL/dh: cosine -1 and ratio 1 mean the same vectors
    ok = rep["cos_max"] < -0.999 and abs(rep["norm_ratio_min"] - 1) < 1e-2 and abs(rep["norm_ratio_max"] - 1) < 1e-2
    log("crosscheck " + ("OK: the readers' bundle vectors, opposite sign" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=list(LAYERS))
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--grad-from-docs", action="store_true")
    c = sub.add_parser("crosscheck")
    c.add_argument("--layer", type=int, default=20)
    p = sub.add_parser("probe")
    p.add_argument("--tasks", default="all")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    for s in (b, c, p):
        s.add_argument("--out", default=str(OUT))
        s.add_argument("--raw", default=str(RAW))
        s.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        import probe_tasks

        return probe_tasks.run(a)
    return {"build": cmd_build, "crosscheck": cmd_crosscheck}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

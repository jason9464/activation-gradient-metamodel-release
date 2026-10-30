"""Shard storage: part files, final arrays, index.parquet, DONE.json, manifest.json and the loader.

Layout under ``<out>/<dataset>/``:
  shard<k>/<array>.npy        fp16 unit vectors [rows, d_model], array = <quantity>_<grad|act>_L<layer>
  shard<k>/index.parquet      one row per stored row: row, doc_id, organism_id, source, chunk_id,
                              n_tokens, n_label, n_resp, tag_pos, truncated, src_row, norm_<array> (fp32)
  shard<k>/plan.json          the shard's configuration and plan hash (written at start)
  shard<k>/DONE.json          row count, wall time, tokens/s (written last)
  bundles.parquet             organism_id -> [doc_id] (raw datasets)
  manifest.json               shard order, row counts, hash of the ordered "doc_id:chunk_id" list

merge_chunks.py writes a one-row-per-document copy: its index has ``n_chunks`` in place of
``chunk_id`` and its order hash runs over "doc_id" keys (``index_keys``).

Rows are buffered in RAM by a background thread and written as fully materialised ``np.save`` part
files every ``part_rows`` rows (no lazily grown memmaps on Lustre); at shard end each array's parts
are concatenated into one ``.npy`` and the parts are removed. A failed write surfaces as an exception
in the GPU loop: the writer thread records the error and keeps draining the queue, and ``put`` waits
with a timeout and checks the error and the thread on every retry, so the loop never blocks forever.

The loader accepts a dataset only when manifest.json exists and every shard's DONE.json and
index.parquet carry the plan hash and row count the manifest recorded. extract.py removes
manifest.json before it (re)computes a shard, so a dataset with a shard redone under another plan
has no manifest until every shard finishes under one plan.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import queue
import shutil
import threading
import time

import numpy as np

from . import core

INDEX_INT_COLS = ("row", "chunk_id", "n_tokens", "n_label", "n_resp", "tag_pos", "src_row")


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    os.replace(tmp, path)


def atomic_write_parquet(path, table):
    import pyarrow.parquet as pq

    tmp = f"{path}.tmp{os.getpid()}"
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def order_hash(keys):
    h = hashlib.sha1()
    for k in keys:
        h.update(k.encode())
        h.update(b"\n")
    return h.hexdigest()


def index_keys(df):
    """Order-hash keys of an index (pandas): "<doc_id>:<chunk_id>" per row, or "<doc_id>" for an index
    without a chunk_id column (merge_chunks.py output, one row per document)."""
    if "chunk_id" in df.columns:
        return [f"{a}:{c}" for a, c in zip(df["doc_id"], df["chunk_id"])]
    return [str(a) for a in df["doc_id"]]


def projected_bytes(n_rows, n_arrays, d_model):
    return int(n_rows) * int(n_arrays) * int(d_model) * 2


def check_disk(path, need_bytes, margin_gb=50):
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    need = need_bytes + margin_gb * 2**30
    if free < need:
        raise SystemExit(
            f"disk check failed at {path}: free {free / 2**30:.1f} GiB < projected {need_bytes / 2**30:.1f} GiB + {margin_gb} GiB"
        )
    return free


class ShardWriter:
    """Background writer: ``put(unit [B, A, D] fp16, norms [B, A] fp32, metas)`` from the GPU loop."""

    def __init__(self, shard_dir, array_names, d_model, part_rows=2048, max_queue=8):
        self.dir = shard_dir
        self.parts_dir = os.path.join(shard_dir, "parts")
        os.makedirs(self.parts_dir, exist_ok=True)
        self.names = list(array_names)
        self.d = d_model
        self.part_rows = part_rows
        self.q = queue.Queue(maxsize=max_queue)
        self.err = None
        self.buf_u, self.buf_n, self.metas, self.norms = [], [], [], []
        self.n_buf = 0
        self.n_parts = 0
        self.rows = 0
        self.flush_s = 0.0
        self.bytes_written = 0
        self.close_s = 0.0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _check(self):
        if self.err is not None:
            raise RuntimeError(f"writer thread failed: {self.err!r}") from self.err
        if not self.t.is_alive():
            raise RuntimeError("writer thread exited before close()")

    def _put(self, item, timeout=1.0):
        """Blocking put that never waits on a dead or failed writer."""
        while True:
            self._check()
            try:
                self.q.put(item, timeout=timeout)
                return
            except queue.Full:
                continue

    def put(self, unit, norms, metas):
        self._put((unit, norms, metas))

    def _loop(self):
        got_sentinel = False
        try:
            while True:
                item = self.q.get()
                if item is None:
                    got_sentinel = True
                    self._flush()
                    return
                unit, norms, metas = item
                assert unit.shape[1:] == (len(self.names), self.d), unit.shape
                self.buf_u.append(unit)
                self.buf_n.append(norms)
                self.metas += metas
                self.n_buf += unit.shape[0]
                if self.n_buf >= self.part_rows:
                    self._flush()
        except BaseException as e:  # surfaced on the next put() or close()
            self.err = e
            # keep draining so a producer blocked on a full queue wakes up and sees the error
            while not got_sentinel:
                got_sentinel = self.q.get() is None

    def _flush(self):
        if not self.n_buf:
            return
        t0 = time.time()
        u = np.concatenate(self.buf_u)
        for a, name in enumerate(self.names):
            np.save(os.path.join(self.parts_dir, f"{name}.{self.n_parts:05d}.npy"), np.ascontiguousarray(u[:, a]))
        self.flush_s += time.time() - t0
        self.bytes_written += u.nbytes
        self.norms.append(np.concatenate(self.buf_n))
        self.rows += u.shape[0]
        self.n_parts += 1
        self.buf_u, self.buf_n, self.n_buf = [], [], 0

    def close(self, index_metadata):
        """Flush, concatenate parts per array, write index.parquet; returns the number of rows."""
        import pyarrow as pa

        t0 = time.time()
        self._put(None)
        self.t.join()
        if self.err is not None:
            raise RuntimeError(f"writer thread failed: {self.err!r}") from self.err
        for name in self.names:
            parts = sorted(glob.glob(os.path.join(self.parts_dir, f"{name}.*.npy")))
            assert len(parts) == self.n_parts, (name, len(parts), self.n_parts)
            arr = np.concatenate([np.load(p) for p in parts]) if parts else np.zeros((0, self.d), np.float16)
            assert arr.shape == (self.rows, self.d), (name, arr.shape)
            np.save(os.path.join(self.dir, f"{name}.npy"), arr)
            for p in parts:
                os.remove(p)
        os.rmdir(self.parts_dir)
        norms = np.concatenate(self.norms) if self.norms else np.zeros((0, len(self.names)), np.float32)
        cols = {k: [m[k] for m in self.metas] for k in ("doc_id", "organism_id", "source", "chunk_id",
                                                         "n_tokens", "n_label", "n_resp", "tag_pos",
                                                         "truncated", "src_row")}
        tbl = {"row": pa.array(np.arange(self.rows, dtype=np.int64))}
        for k, v in cols.items():
            if k in INDEX_INT_COLS:
                tbl[k] = pa.array(np.asarray(v, dtype=np.int64))
            elif k == "truncated":
                tbl[k] = pa.array(np.asarray(v, dtype=bool))
            else:
                tbl[k] = pa.array(v, type=pa.string())
        for a, name in enumerate(self.names):
            tbl[f"norm_{name}"] = pa.array(norms[:, a].astype(np.float32))
        table = pa.table(tbl).replace_schema_metadata({k: str(v) for k, v in index_metadata.items()})
        atomic_write_parquet(os.path.join(self.dir, "index.parquet"), table)
        self.close_s = time.time() - t0
        return self.rows


def index_metadata(extra=None):
    md = dict(
        layer_convention=core.LAYER_CONVENTION,
        sign_convention=core.SIGN_CONVENTION,
        position_convention=core.POSITION_CONVENTION,
        storage_convention=core.STORAGE_CONVENTION,
    )
    md.update(extra or {})
    return md


def remove_manifest(ds_dir):
    """Called before a shard is (re)computed: the loader refuses the dataset until all shards agree again."""
    try:
        os.remove(os.path.join(ds_dir, "manifest.json"))
        return True
    except FileNotFoundError:
        return False


def index_plan_hash(path):
    """plan_hash stored in an index.parquet's schema metadata (None if absent)."""
    import pyarrow.parquet as pq

    md = pq.read_schema(path).metadata or {}
    v = md.get(b"plan_hash")
    return v.decode() if v is not None else None


def try_write_manifest(ds_dir, n_shards, plan_hashes=None):
    """Write manifest.json when every shard has DONE.json; returns the manifest or None."""
    import pyarrow.parquet as pq

    shards = []
    keys = []
    for k in range(n_shards):
        d = os.path.join(ds_dir, f"shard{k}")
        if not os.path.exists(os.path.join(d, "DONE.json")):
            return None
        with open(os.path.join(d, "DONE.json")) as f:
            done = json.load(f)
        if plan_hashes is not None and done.get("plan_hash") != plan_hashes[k]:
            return None
        if index_plan_hash(os.path.join(d, "index.parquet")) != done.get("plan_hash"):
            return None
        idx = pq.read_table(os.path.join(d, "index.parquet"), columns=["doc_id", "chunk_id"])
        ks = [f"{a}:{c}" for a, c in zip(idx.column("doc_id").to_pylist(), idx.column("chunk_id").to_pylist())]
        keys += ks
        shards.append(dict(shard=k, dir=f"shard{k}", rows=len(ks), plan_hash=done.get("plan_hash")))
    plan = {}
    plan_path = os.path.join(ds_dir, "shard0", "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            p0 = json.load(f)
        # think_labels: absent in older plans (chat rows then were 'masked'); null for raw
        plan = {k: p0[k] for k in ("model", "adapter", "quantities", "full_skip_pos0", "grad_layers", "act_layers",
                                   "think_labels", "tok_budget", "code_sha1") if k in p0}
    man = dict(
        n_shards=n_shards,
        plan=plan,
        shard_order=[s["dir"] for s in shards],
        shards=shards,
        total_rows=len(keys),
        doc_order_hash=order_hash(keys),
        doc_order_hash_def="sha1 over '<doc_id>:<chunk_id>\\n' for every row, shards in order, rows in stored order",
        layer_convention=core.LAYER_CONVENTION,
        sign_convention=core.SIGN_CONVENTION,
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    atomic_write_json(os.path.join(ds_dir, "manifest.json"), man)
    return man


def load_index(ds_dir):
    """Concatenated index (pandas) of all shards.

    Refuses the dataset when manifest.json is missing, when a shard's DONE.json or index.parquet
    carries another plan hash or row count than the manifest recorded, or when the order hash of the
    concatenated indexes differs from the manifest's.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    man_path = os.path.join(ds_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError(f"no manifest.json in {ds_dir}: a shard is missing, still running, or was redone "
                           "under another plan")
    with open(man_path) as f:
        man = json.load(f)
    dfs = []
    for i, s in enumerate(man["shard_order"]):
        sm = man["shards"][i]
        d = os.path.join(ds_dir, s)
        with open(os.path.join(d, "DONE.json")) as f:
            done = json.load(f)
        t = pq.read_table(os.path.join(d, "index.parquet"))
        md = t.schema.metadata or {}
        idx_hash = md[b"plan_hash"].decode() if b"plan_hash" in md else None
        bad = [k for k, ok in (("DONE.json plan_hash", done.get("plan_hash") == sm["plan_hash"]),
                               ("index.parquet plan_hash", idx_hash == sm["plan_hash"]),
                               ("DONE.json rows", done.get("rows") == sm["rows"]),
                               ("index.parquet rows", t.num_rows == sm["rows"])) if not ok]
        if bad:
            raise RuntimeError(f"{d} does not match manifest.json ({', '.join(bad)}): the shard was redone "
                               "or changed after the manifest was written")
        dfs.append(t.to_pandas())
        dfs[-1]["shard"] = s
    df = pd.concat(dfs, ignore_index=True)
    keys = index_keys(df)
    if order_hash(keys) != man["doc_order_hash"] or len(df) != man["total_rows"]:
        raise RuntimeError(f"manifest hash mismatch in {ds_dir}: indexes were changed or shards are missing")
    return df, man


def load_array(ds_dir, quantity, kind, layer, unit=True, mmap=True):
    """(index DataFrame, [rows, d_model] array). ``unit=False`` multiplies by the stored norms (fp32)."""
    df, man = load_index(ds_dir)
    name = core.array_name(quantity, kind, layer)
    parts = [np.load(os.path.join(ds_dir, s, f"{name}.npy"), mmap_mode="r" if mmap else None) for s in man["shard_order"]]
    for s, sm, p in zip(man["shard_order"], man["shards"], parts):
        if p.ndim != 2 or p.shape[0] != sm["rows"]:
            raise RuntimeError(f"{ds_dir}/{s}/{name}.npy has shape {p.shape}, index has {sm['rows']} rows")
    arr = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if not unit:
        arr = arr.astype(np.float32) * df[f"norm_{name}"].to_numpy(np.float32)[:, None]
    return df, arr

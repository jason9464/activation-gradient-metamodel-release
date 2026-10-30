#!/usr/bin/env python
"""Copy an extracted raw dataset with exactly one row per document (chunk rows summed).

  python merge_chunks.py --src <dataset dir> --dst <new dir> [--overwrite] [--workers N]

A raw document longer than 2,048 tokens was extracted as independent 2,048-token chunks (one forward
and backward pass each, context reset), one row per chunk (``chunk_id`` 0, 1, ...). This script
writes a copy in which every ``doc_id`` has one row:

- Arrays (all of them): the chunk rows are restored in fp32 (``unit.astype(float32) * norm``),
  summed, and stored again as fp16 unit vector + fp32 norm (zero sum -> zeros, norm 0). Norms
  accumulate in float64 and are rounded to fp32. Rows of one-chunk documents are copied bit for bit.
- Index: ``chunk_id`` is replaced by ``n_chunks`` (int64, same column position); ``n_tokens`` and
  ``n_label`` are summed over the chunks; ``truncated`` is true if any chunk was truncated; every
  other column comes from chunk 0 (they are checked to agree across chunks). ``row`` is renumbered.
  Schema metadata is kept; ``plan_hash`` is the new shard plan hash and ``merged_from_plan_hash``
  records the source shard's hash.
- Row order per shard: by (``n_tokens``, position of the document's first row in the source shard).
  Source shards are sorted by ``n_tokens``, so one-chunk rows keep their order and merged rows,
  longer than any one-pass row, follow them.
- plan.json: the source plan with new ``shard_bounds``, ``dataset_order_hash`` and
  ``shard_rows_hash`` (over ``"<doc_id>"`` keys) plus a ``merged_chunks`` record of the source
  hashes, the rule and this file's SHA-1; ``plan_hash`` = SHA-1 of the sorted-key JSON of the rest,
  as in extract.py. DONE.json: the source shard's extraction record with the new ``rows`` and
  ``plan_hash`` and a ``merged_chunks`` record. manifest.json in the layout of
  store.try_write_manifest with ``doc_order_hash`` over ``"<doc_id>\\n"`` (store.index_keys) and a
  ``merged_chunks`` record. bundles.parquet is copied unchanged; stats.json is written by stats.py.
  MERGE.json (counts and the merged doc_ids) stays local: hf_upload.py does not upload it.

The copy is built in ``<dst>.partial`` and renamed to ``<dst>`` only after store.load_index accepts
it. The source is only read. The script stops before writing anything if a document's rows span two
shards or its chunk_ids are not 0..k-1. ``--overwrite`` replaces an existing ``<dst>`` only if it
holds a MERGE.json (an earlier output of this script): ``<dst>`` is renamed to ``<dst>.old``, the
finished ``<dst>.partial`` to ``<dst>``, and ``<dst>.old`` is removed. The MERGE_PARTIAL marker is
removed only after the last rename, so an interrupted copy can always be rebuilt. A leftover
``<dst>.old`` (an interrupted ``--overwrite``) is refused at startup and must be removed by hand.

Memory and I/O: one task per (shard, array) in a thread pool; a task reads one source .npy through
mmap, builds the output array in RAM (fp32 only for the merged rows) and writes it with np.save to a
temporary name that is then renamed (no lazily grown memmaps on Lustre). Peak RAM is about two shard
arrays (2 x 220 MB) per worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agm import core, store  # noqa: E402

MERGE_RULE = (
    "rows of one doc_id (chunk_id 0..k-1, each one independent 2,048-token pass) restored in fp32 as "
    "unit * norm, summed per array, stored as fp16 unit + fp32 norm (norm accumulated in float64; zero sum -> "
    "zeros, norm 0); n_tokens and n_label summed; truncated = any chunk truncated; n_chunks = k replaces "
    "chunk_id; other columns from chunk 0; rows ordered by (n_tokens, first source row)"
)
SAME_ACROSS_CHUNKS = ("organism_id", "source", "n_resp", "tag_pos", "src_row")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def file_sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def to_unit_np(x):
    """fp32 [k, D] -> (fp16 unit rows, fp32 norms). Norms accumulate in float64; zero rows stay zero, norm 0."""
    n = np.sqrt(np.square(x, dtype=np.float64).sum(axis=1)).astype(np.float32)
    u = np.zeros(x.shape, np.float16)
    nz = n > 0
    u[nz] = (x[nz] / n[nz, None]).astype(np.float16)
    return u, n


def restore(unit_rows, norms):
    """fp16 unit rows [k, D] and fp32 norms [k] -> fp32 vectors, exactly as store.load_array(unit=False)."""
    return np.asarray(unit_rows).astype(np.float32) * np.asarray(norms, np.float32)[:, None]


def plan_shard(x):
    """Row grouping of one source shard index (pandas, rows in stored order).

    Returns dict with ``order`` (output row -> source position of the document's chunk 0),
    ``chunk_pos`` [n_out, kmax] (source positions by chunk_id, -1 padded) and ``n_chunks``.
    """
    pos = np.arange(len(x))
    assert (x["row"].to_numpy() == pos).all(), "row column is not 0..n-1"
    groups = {}
    for p, d, c in zip(pos, x["doc_id"].to_numpy(), x["chunk_id"].to_numpy()):
        groups.setdefault(d, {})[int(c)] = int(p)
    kmax = max(len(g) for g in groups.values())
    docs = list(groups)
    chunk_pos = np.full((len(docs), kmax), -1, np.int64)
    for i, d in enumerate(docs):
        g = groups[d]
        if sorted(g) != list(range(len(g))):
            raise SystemExit(f"doc_id {d}: chunk_ids {sorted(g)} are not 0..{len(g) - 1}")
        chunk_pos[i, :len(g)] = [g[c] for c in range(len(g))]
    n_chunks = (chunk_pos >= 0).sum(1)
    ntok_src = x["n_tokens"].to_numpy(np.int64)
    n_tokens = np.where(chunk_pos >= 0, ntok_src[np.maximum(chunk_pos, 0)], 0).sum(1)
    first = chunk_pos.min(1, where=chunk_pos >= 0, initial=len(x))
    o = np.lexsort((first, n_tokens))
    return dict(chunk_pos=chunk_pos[o], n_chunks=n_chunks[o], n_tokens=n_tokens[o], doc_ids=[docs[i] for i in o])


def merge_array(src_path, dst_path, plan, norms_src):
    """Write one merged array; returns the fp32 norms of all output rows."""
    src = np.load(src_path, mmap_mode="r")
    cp, nc = plan["chunk_pos"], plan["n_chunks"]
    single = np.flatnonzero(nc == 1)
    multi = np.flatnonzero(nc > 1)
    out = np.empty((len(nc), src.shape[1]), np.float16)
    norms = np.empty(len(nc), np.float32)
    out[single] = src[cp[single, 0]]
    norms[single] = norms_src[cp[single, 0]]
    if len(multi):
        s = restore(src[cp[multi, 0]], norms_src[cp[multi, 0]])
        for c in range(1, cp.shape[1]):
            m = nc[multi] > c
            p = cp[multi[m], c]
            s[m] += restore(src[p], norms_src[p])
        out[multi], norms[multi] = to_unit_np(s)
    tmp = f"{dst_path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, out)
    os.replace(tmp, dst_path)
    return norms


def build_index(t_src, plan, norm_cols, metadata):
    """Output index.parquet table: chunk_id -> n_chunks, summed counts, renumbered rows, new norms."""
    import pyarrow as pa

    cp, nc = plan["chunk_pos"], plan["n_chunks"]
    valid = cp >= 0
    first = cp[:, 0]
    cols = {}
    for name in t_src.column_names:
        col = t_src.column(name)
        if name == "row":
            cols[name] = pa.array(np.arange(len(nc), dtype=np.int64))
        elif name == "chunk_id":
            cols["n_chunks"] = pa.array(nc.astype(np.int64))
        elif name in ("n_tokens", "n_label"):
            v = col.to_numpy().astype(np.int64)
            cols[name] = pa.array(np.where(valid, v[np.maximum(cp, 0)], 0).sum(1).astype(np.int64))
        elif name == "truncated":
            v = col.to_numpy(zero_copy_only=False).astype(bool)
            cols[name] = pa.array(np.where(valid, v[np.maximum(cp, 0)], False).any(1))
        elif name.startswith("norm_"):
            cols[name] = pa.array(norm_cols[name[5:]].astype(np.float32))
        else:
            cols[name] = col.take(pa.array(first))
    return pa.table(cols).replace_schema_metadata(metadata)


def check_source(df):
    """Stop unless every doc_id lives in one shard and its chunk rows agree on the per-document columns."""
    span = df.groupby("doc_id")["shard"].nunique()
    bad = span[span > 1]
    if len(bad):
        raise SystemExit(f"refusing: {len(bad)} doc_ids have rows in more than one shard, e.g. "
                         f"{bad.index[:5].tolist()}; a per-shard merge would split them")
    multi = df[df.groupby("doc_id")["doc_id"].transform("size") > 1]
    for c in SAME_ACROSS_CHUNKS:
        n = multi.groupby("doc_id")[c].nunique()
        if (n > 1).any():
            raise SystemExit(f"refusing: column {c} differs across the chunks of {int((n > 1).sum())} documents")
    return int(multi["doc_id"].nunique())


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="chunked dataset dir (extract.py output); only read")
    ap.add_argument("--dst", required=True, help="new dataset dir with one row per document")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --dst written by this script")
    ap.add_argument("--workers", type=int, default=8, help="parallel (shard, array) tasks")
    return ap.parse_args(argv)


def main(argv=None):
    import pyarrow.parquet as pq

    args = parse_args(argv)
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if dst == src or dst.startswith(src + os.sep) or src.startswith(dst + os.sep):
        raise SystemExit("refusing: --dst must be outside --src")
    if os.path.exists(dst):
        if not args.overwrite:
            raise SystemExit(f"refusing: {dst} exists; pass --overwrite to replace it")
        if not os.path.isfile(os.path.join(dst, "MERGE.json")):
            raise SystemExit(f"refusing to overwrite {dst}: it has no MERGE.json, so it is not an output of this script")
    if os.path.exists(dst + ".old"):
        raise SystemExit(f"refusing: {dst}.old exists, left by an interrupted --overwrite; check that {dst} "
                         f"(or {dst}.old, if {dst} is missing) is the copy to keep, then remove the other by hand")
    t0 = time.time()
    df, man_src = store.load_index(src)  # validates the source (plan hashes, row counts, order hash)
    n_multi_docs = check_source(df)
    log(f"source {src}: {len(df)} rows, {df['doc_id'].nunique()} documents, {n_multi_docs} with more than one row; "
        "every document in one shard")

    shards = man_src["shard_order"]
    plans, src_tables, src_plans, src_dones = {}, {}, {}, {}
    for s in shards:
        x = df[df["shard"] == s].reset_index(drop=True)
        plans[s] = plan_shard(x)
        src_tables[s] = pq.read_table(os.path.join(src, s, "index.parquet"))
        with open(os.path.join(src, s, "plan.json")) as f:
            src_plans[s] = json.load(f)
        with open(os.path.join(src, s, "DONE.json")) as f:
            src_dones[s] = json.load(f)
    arrays = src_plans[shards[0]]["arrays"]
    rows_out = [len(plans[s]["n_chunks"]) for s in shards]
    bounds = np.concatenate([[0], np.cumsum(rows_out)]).tolist()
    all_keys = [d for s in shards for d in plans[s]["doc_ids"]]
    if len(set(all_keys)) != len(all_keys):
        raise SystemExit("internal error: a doc_id appears twice in the output")
    merge_sha1 = file_sha1(os.path.abspath(__file__))
    new_plans = {}
    for k, s in enumerate(shards):
        sp = src_plans[s]
        p = {key: v for key, v in sp.items() if key != "plan_hash"}
        p.update(shard_bounds=bounds, dataset_order_hash=store.order_hash(all_keys),
                 shard_rows_hash=store.order_hash(plans[s]["doc_ids"]),
                 merged_chunks=dict(source_plan_hash=sp["plan_hash"], source_shard_rows_hash=sp["shard_rows_hash"],
                                    source_dataset_order_hash=sp["dataset_order_hash"],
                                    source_shard_bounds=sp["shard_bounds"], rule=MERGE_RULE, merge_code_sha1=merge_sha1))
        h = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()
        new_plans[s] = dict(p, plan_hash=h)

    d_model = np.load(os.path.join(src, shards[0], f"{arrays[0]}.npy"), mmap_mode="r").shape[1]
    need = store.projected_bytes(sum(rows_out), len(arrays), d_model)
    part = dst + ".partial"
    if os.path.exists(part):
        if not os.path.isfile(os.path.join(part, "MERGE_PARTIAL")):
            raise SystemExit(f"refusing: {part} exists and was not made by this script")
        shutil.rmtree(part)
    store.check_disk(os.path.dirname(dst), need, margin_gb=20)
    os.makedirs(part)
    open(os.path.join(part, "MERGE_PARTIAL"), "w").close()
    for s in shards:
        os.makedirs(os.path.join(part, s))

    tasks = [(s, a) for s in shards for a in arrays]
    norm_src = {(s, a): src_tables[s].column(f"norm_{a}").to_numpy().astype(np.float32) for s, a in tasks}
    log(f"merging {len(tasks)} arrays ({len(shards)} shards x {len(arrays)}) with {args.workers} workers into {part}")
    t1 = time.time()

    def run(task):
        s, a = task
        return task, merge_array(os.path.join(src, s, f"{a}.npy"), os.path.join(part, s, f"{a}.npy"), plans[s],
                                 norm_src[task])

    norms = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (task, nv) in enumerate(ex.map(run, tasks)):
            norms[task] = nv
            if (i + 1) % 28 == 0 or i + 1 == len(tasks):
                log(f"  {i + 1}/{len(tasks)} arrays written, {time.time() - t1:.0f}s")
    merge_s = time.time() - t1

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    per_shard = []
    for k, s in enumerate(shards):
        md = {kk.decode(): vv.decode() for kk, vv in (src_tables[s].schema.metadata or {}).items()}
        md.update(plan_hash=new_plans[s]["plan_hash"], merged_from_plan_hash=src_plans[s]["plan_hash"])
        t = build_index(src_tables[s], plans[s], {a: norms[(s, a)] for a in arrays}, md)
        store.atomic_write_parquet(os.path.join(part, s, "index.parquet"), t)
        store.atomic_write_json(os.path.join(part, s, "plan.json"), new_plans[s])
        n_merged = int((plans[s]["n_chunks"] > 1).sum())
        done = dict(src_dones[s])
        tokens = int(plans[s]["n_tokens"].sum())
        assert tokens == done["tokens"], (s, tokens, done["tokens"])
        done.update(rows=rows_out[k], plan_hash=new_plans[s]["plan_hash"],
                    merged_chunks=dict(source_plan_hash=src_dones[s]["plan_hash"], source_rows=src_dones[s]["rows"],
                                       merged_docs=n_merged, merge_code_sha1=merge_sha1, merge_s=round(merge_s, 1),
                                       finished=now))
        store.atomic_write_json(os.path.join(part, s, "DONE.json"), done)
        per_shard.append(dict(shard=s, source_rows=src_dones[s]["rows"], rows=rows_out[k], merged_docs=n_merged))
    shutil.copyfile(os.path.join(src, "bundles.parquet"), os.path.join(part, "bundles.parquet"))

    merged_ids = [d for s in shards for d, n in zip(plans[s]["doc_ids"], plans[s]["n_chunks"]) if n > 1]
    rec = dict(source_total_rows=man_src["total_rows"], source_doc_order_hash=man_src["doc_order_hash"],
               merged_docs=len(merged_ids), max_chunks=int(max(plans[s]["n_chunks"].max() for s in shards)),
               rule=MERGE_RULE, merge_code_sha1=merge_sha1)
    man = dict(
        n_shards=man_src["n_shards"],
        plan=man_src["plan"],
        shard_order=shards,
        shards=[dict(shard=k, dir=s, rows=rows_out[k], plan_hash=new_plans[s]["plan_hash"]) for k, s in enumerate(shards)],
        total_rows=len(all_keys),
        doc_order_hash=store.order_hash(all_keys),
        doc_order_hash_def="sha1 over '<doc_id>\\n' for every row, shards in order, rows in stored order "
                           "(one row per document, no chunk_id column)",
        layer_convention=core.LAYER_CONVENTION,
        sign_convention=core.SIGN_CONVENTION,
        merged_chunks=rec,
        written=now,
    )
    store.atomic_write_json(os.path.join(part, "manifest.json"), man)
    import stats

    stats.main(part)
    store.atomic_write_json(os.path.join(part, "MERGE.json"), dict(
        src=src, dst=dst, **rec, total_rows=len(all_keys), unique_doc_ids=len(set(all_keys)), shards=per_shard,
        arrays=len(arrays), merge_s=round(merge_s, 1), total_s=round(time.time() - t0, 1), finished=now,
        merged_doc_ids=merged_ids))
    df2, _ = store.load_index(part)  # refuse to publish a copy the loader would not accept
    assert len(df2) == len(all_keys) == df2["doc_id"].nunique()
    if os.path.exists(dst):
        old = dst + ".old"
        os.rename(dst, old)
        os.rename(part, dst)
        shutil.rmtree(old)
    else:
        os.rename(part, dst)
    os.remove(os.path.join(dst, "MERGE_PARTIAL"))  # only now: an interrupted rename leaves a rebuildable .partial
    log(f"done: {dst}: {len(all_keys)} rows ({len(merged_ids)} merged documents) from {man_src['total_rows']} source "
        f"rows; arrays {merge_s:.0f}s, total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

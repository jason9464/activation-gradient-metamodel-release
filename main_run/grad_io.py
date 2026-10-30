"""Read A's gradient store for the C1 join and B; write dummy stores in the same layout.

A's layout (``main_run/agm/store.py``) under one dataset
directory such as ``$IIF_BASE/agm_data/loracle_mix``:

  manifest.json                       shard order, row counts, order hash
  shard<k>/plan.json, DONE.json       shard plan and completion record
  shard<k>/index.parquet              one row per chunk: doc_id, chunk_id, n_tokens, n_label, ...,
                                      norm_<array> (fp32)
  shard<k>/full_grad_L<layer>.npy     fp16 unit vectors [rows, d_model]

A row is one chunk of at most 2,048 tokens; most documents have one. A document vector is the sum of
its chunk rows (unit x norm), i.e. the token-sum +dL/dh of the document (negate for descent).
``doc_id`` is the SHA-1 of the raw text, so identical texts share one set of rows.

A's merged copy (``main_run/merge_chunks.py``, the public HF revision from eef0f641 on) has one row per
document: the chunk rows summed, ``n_chunks`` in place of ``chunk_id``, ``n_tokens``/``n_label``
summed, and the manifest order hash over ``"<doc_id>"`` keys. Both layouts read the same way here.

A copy that holds only some layers works as long as every shard's index.parquet, plan.json and
DONE.json and the dataset's manifest.json come along: arrays are opened per requested layer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from main_run.agm import core, store
from main_run.c1.constants import CHUNK_LEN, DEFAULT_LAYERS, HIDDEN_SIZE, MIN_CHUNK


class GradStoreError(ValueError):
    """Validation / I/O failure for gradient stores."""


# How a bundle's document vectors are combined (GradStore.bundle_vectors).
WEIGHTINGS = ("tokensum", "docmean")


def grad_array_name(layer: int) -> str:
    return core.array_name("full", "grad", layer)


def estimate_chunks(n_tokens: int, chunk_len: int = CHUNK_LEN, min_chunk: int = MIN_CHUNK) -> list[int]:
    """Return per-chunk token lengths after dropping chunks shorter than min_chunk."""
    if n_tokens <= 0:
        return []
    lengths = []
    for start in range(0, n_tokens, chunk_len):
        L = min(chunk_len, n_tokens - start)
        if L >= min_chunk:
            lengths.append(L)
    return lengths


def n_loss_tokens_for_chunk(chunk_len_tokens: int) -> int:
    """Valid shifted CE targets in a chunk of length T: T - 1 (A's `n_label` for raw text)."""
    return max(0, chunk_len_tokens - 1)


def rebuild_doc_vector_token_sum(chunk_vecs: torch.Tensor) -> torch.Tensor:
    """Doc vector under token-sum loss = sum of per-chunk vectors (already token-summed)."""
    return chunk_vecs.sum(dim=0)


def rebuild_doc_vector_per_chunk_mean(
    chunk_vecs: torch.Tensor, n_label_per_chunk: Sequence[int]
) -> torch.Tensor:
    """
    Rebuild a doc vector as if each chunk used mean-over-tokens loss then summed chunks.
    chunk_vecs are token-sum gradients; divide each by its n_label, then sum.
    """
    if len(n_label_per_chunk) != chunk_vecs.shape[0]:
        raise GradStoreError("n_label_per_chunk length mismatch")
    out = torch.zeros_like(chunk_vecs[0], dtype=torch.float32)
    for i, n in enumerate(n_label_per_chunk):
        if n <= 0:
            raise GradStoreError(f"non-positive n_label at chunk {i}")
        out = out + chunk_vecs[i].float() / float(n)
    return out


def sum_doc_vectors(doc_vecs: torch.Tensor) -> torch.Tensor:
    """Batch / bundle vector = sum of document vectors."""
    return doc_vecs.float().sum(dim=0)


def load_store_index(ds_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A's concatenated index and manifest; GradStoreError if the store is incomplete or inconsistent."""
    try:
        return store.load_index(str(ds_dir))
    except (RuntimeError, FileNotFoundError, KeyError) as e:
        raise GradStoreError(str(e)) from e


class GradStore:
    """One A dataset directory, opened once: validated index, arrays memory-mapped per (shard, layer)."""

    def __init__(self, ds_dir: Path, layers: Sequence[int] | None = None):
        self.dir = Path(ds_dir)
        self.layers = list(layers or DEFAULT_LAYERS)
        self.index, self.manifest = load_store_index(self.dir)
        self.shards = list(self.manifest["shard_order"])
        missing = [
            L
            for L in self.layers
            if f"norm_{grad_array_name(L)}" not in self.index.columns
            or not all((self.dir / s / f"{grad_array_name(L)}.npy").exists() for s in self.shards)
        ]
        if missing:
            raise GradStoreError(f"store has no gradient array for layers {missing}")
        self._rows_of = self.index.groupby("doc_id", sort=False).indices
        self._shard = self.index["shard"].to_numpy()
        self._row = self.index["row"].to_numpy(np.int64)
        self._n_label = self.index["n_label"].to_numpy(np.int64)
        self._mm: dict[tuple[str, int], np.ndarray] = {}

    def _array(self, shard: str, layer: int) -> np.ndarray:
        key = (shard, layer)
        if key not in self._mm:
            arr = np.load(self.dir / shard / f"{grad_array_name(layer)}.npy", mmap_mode="r")
            rows = self.manifest["shards"][self.shards.index(shard)]["rows"]
            if arr.ndim != 2 or arr.shape[0] != rows:
                raise GradStoreError(f"{shard}/{grad_array_name(layer)}.npy has shape {arr.shape}, index has {rows} rows")
            self._mm[key] = arr
        return self._mm[key]

    def _positions(self, doc_id: str) -> np.ndarray:
        if doc_id not in self._rows_of:
            raise GradStoreError(f"missing doc id: {doc_id}")
        return self._rows_of[doc_id]

    def validate(self, doc_ids: Sequence[str]) -> dict[str, Any]:
        """Check from the index alone that every doc is present with finite, nonzero norms per layer."""
        needed = sorted(set(doc_ids))
        missing = [d for d in needed if d not in self._rows_of]
        if missing:
            raise GradStoreError(f"missing doc id: {missing[0]} ({len(missing)} missing)")
        pos = np.concatenate([self._rows_of[d] for d in needed]) if needed else np.zeros(0, np.int64)
        sub = self.index.iloc[pos]
        norms = np.stack(
            [sub[f"norm_{grad_array_name(L)}"].to_numpy(np.float64) for L in self.layers], axis=1
        )
        doc_of = sub["doc_id"].to_numpy()
        bad = ~np.isfinite(norms)
        if bad.any():
            i, j = map(int, np.argwhere(bad)[0])
            raise GradStoreError(f"NaN/Inf norm at layer {self.layers[j]} for {doc_of[i]}")
        zero = norms == 0
        if zero.any():
            # Any zero layer, not just all: a detach between hooked layers zeroes the earlier ones.
            i = int(np.argwhere(zero.any(axis=1))[0, 0])
            zero_layers = [self.layers[j] for j in np.flatnonzero(zero[i])]
            raise GradStoreError(f"zero norm at layers {zero_layers} for {doc_of[i]}")
        chunks = sub["n_chunks"] if "n_chunks" in sub.columns else pd.Series(1, index=sub.index)
        n_chunks_per_doc = chunks.groupby(sub["doc_id"]).sum()
        return {
            "n_docs": len(needed),
            "n_rows": int(len(sub)),
            "n_multi_chunk_docs": int((n_chunks_per_doc > 1).sum()),
            "layers": self.layers,
            "doc_order_hash": self.manifest["doc_order_hash"],
            "row_norm_mean": {str(L): float(norms[:, j].mean()) for j, L in enumerate(self.layers)} if len(sub) else {},
            "row_norm_std": {str(L): float(norms[:, j].std()) for j, L in enumerate(self.layers)} if len(sub) else {},
        }

    def _vectors(self, pos: np.ndarray, layer: int) -> np.ndarray:
        """Raw fp32 vectors (unit x norm) of the given index positions at one layer."""
        shard, row = self._shard[pos], self._row[pos]
        out = None
        for s in np.unique(shard):
            m = shard == s
            part = np.asarray(self._array(s, layer)[row[m]], dtype=np.float32)
            if out is None:
                out = np.empty((len(pos), part.shape[1]), np.float32)
            out[m] = part
        norms = self.index[f"norm_{grad_array_name(layer)}"].to_numpy(np.float32)[pos]
        return out * norms[:, None]

    def doc_vectors(self, doc_ids: Sequence[str]) -> torch.Tensor:
        """Document vectors [n, L, d] fp32: sum of each document's chunk rows, raw +dL/dh.

        One output row per requested id, repeats included: 4 organisms list a text twice, and
        their bundle sum counts it twice.
        """
        pos_list = [self._positions(d) for d in doc_ids]
        pos = np.concatenate(pos_list)
        owner = np.repeat(np.arange(len(doc_ids)), [len(p) for p in pos_list])
        out = None
        for li, layer in enumerate(self.layers):
            vec = self._vectors(pos, layer)
            if out is None:
                out = np.zeros((len(doc_ids), len(self.layers), vec.shape[1]), np.float32)
            np.add.at(out[:, li], owner, vec)
        bad = ~np.isfinite(out).all(axis=(1, 2))
        if bad.any():
            raise GradStoreError(f"NaN/Inf in gradient for {doc_ids[int(np.flatnonzero(bad)[0])]}")
        return torch.from_numpy(out)

    def doc_n_label(self, doc_id: str) -> int:
        """Label positions of a document over all its chunk rows."""
        return int(self._n_label[self._positions(doc_id)].sum())

    def bundle_vectors(
        self, doc_lists: Sequence[Sequence[str]], layer: int, weightings: Sequence[str] = ("tokensum",)
    ) -> dict[str, np.ndarray]:
        """
        Per weighting, [n_bundles, d] fp32 raw +dL/dh at one layer: the sum over each bundle's documents.
        `tokensum` sums the document vectors (each a token sum, so long documents weigh more);
        `docmean` sums each document vector divided by its label count. A listed-twice text counts twice.
        Reads each row once for all weightings.
        """
        unknown = set(weightings) - set(WEIGHTINGS)
        if unknown:
            raise GradStoreError(f"unknown weightings {sorted(unknown)}; choose from {WEIGHTINGS}")
        pos_parts, scale_parts, seg_len = [], [], []
        for docs in doc_lists:
            if not docs:
                raise GradStoreError("empty bundle")
            n = 0
            for doc in docs:
                p = self._positions(doc)
                pos_parts.append(p)
                scale_parts.append(np.full(len(p), 1.0 / self.doc_n_label(doc), np.float32))
                n += len(p)
            seg_len.append(n)
        vec = self._vectors(np.concatenate(pos_parts), layer)
        starts = np.concatenate([[0], np.cumsum(seg_len)[:-1]])
        out = {}
        for w in weightings:
            v = vec if w == "tokensum" else vec * np.concatenate(scale_parts)[:, None]
            out[w] = np.add.reduceat(v, starts, axis=0)
            if not np.isfinite(out[w]).all():
                raise GradStoreError(f"NaN/Inf in a {w} bundle vector at layer {layer}")
        return out

    def chunk_vectors(self, doc_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
        """A document's rows [n_rows, L, d] fp32, plus their n_label and n_tokens: one row per chunk in
        chunk order in a per-chunk store; one row (chunks already summed) in a merged store."""
        pos = self._positions(doc_id)
        merged = "chunk_id" not in self.index.columns
        if not merged:
            pos = pos[np.argsort(self.index["chunk_id"].to_numpy()[pos])]
        vecs = np.stack([self._vectors(pos, L) for L in self.layers], axis=1)
        sub = self.index.iloc[pos]
        meta = {
            "doc_id": doc_id,
            "merged": merged,
            "chunk_ids": None if merged else sub["chunk_id"].tolist(),
            "n_label_per_chunk": sub["n_label"].astype(int).tolist(),
            "n_tokens_per_chunk": sub["n_tokens"].astype(int).tolist(),
        }
        return torch.from_numpy(vecs), meta


def load_doc_grads(ds_dir: Path, doc_ids: Sequence[str], *, layers: Sequence[int] | None = None) -> torch.Tensor:
    """Document vectors [n, L, d] fp32 from A's store (opens the store; use GradStore to reuse it)."""
    return GradStore(ds_dir, layers).doc_vectors(doc_ids)


def load_chunk_grads(
    ds_dir: Path, doc_id: str, *, layers: Sequence[int] | None = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-chunk vectors [n_chunks, L, d] fp32 and their metadata from A's store."""
    return GradStore(ds_dir, layers).chunk_vectors(doc_id)


def write_dummy_store(
    ds_dir: Path,
    docs: Sequence[dict[str, Any]],
    *,
    layers: Sequence[int] | None = None,
    hidden_size: int = HIDDEN_SIZE,
    seed: int = 0,
    n_shards: int = 1,
    merged: bool = False,
) -> dict[str, np.ndarray]:
    """
    Write random gradients in A's exact layout with A's own writer, for tests and B prototyping.

    Each doc dict needs `doc_id` and `n_tokens` (chunked like A: 2,048-token chunks, tails under
    16 tokens dropped); `organism_id` is optional. A repeated doc_id is written once, as A dedups
    identical texts. Returns doc_id -> raw fp32 chunk vectors [n_chunks, L, d] before fp16 storage.
    `merged=True` writes A's merged layout instead (merge_chunks.py): one row per document holding
    the fp32 sum of its chunks, `n_chunks` in place of `chunk_id`, manifest hashed over doc_ids.
    """
    layers = list(layers or DEFAULT_LAYERS)
    names = [grad_array_name(L) for L in layers]
    ds_dir = Path(ds_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    unique: dict[str, dict[str, Any]] = {}
    for d in docs:
        unique.setdefault(d["doc_id"], d)
    raw: dict[str, np.ndarray] = {}
    rows: list[tuple[np.ndarray, dict[str, Any]]] = []
    for doc_id, d in unique.items():
        n_tokens = int(d["n_tokens"])
        chunk_lens = estimate_chunks(n_tokens)
        whole = len(chunk_lens) == 1 and chunk_lens[0] == n_tokens
        vecs = []
        for ci, cl in enumerate(chunk_lens):
            n_label = n_loss_tokens_for_chunk(cl)
            # Scale by n_label so different chunk lengths matter for the chunk-mean rebuild.
            v = rng.standard_normal((len(layers), hidden_size)).astype(np.float32) * float(n_label)
            vecs.append(v)
            meta = dict(doc_id=doc_id, organism_id=str(d.get("organism_id", "")), source="dummy", chunk_id=ci,
                        n_tokens=cl, n_label=n_label, n_resp=0, tag_pos=-1, truncated=not whole, src_row=-1)
            rows.append((v, meta))
        if vecs:
            raw[doc_id] = np.stack(vecs)
    if not rows:
        raise GradStoreError("dummy store would have no rows")
    if merged:
        by_doc: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {}
        for v, m in rows:
            by_doc.setdefault(m["doc_id"], []).append((v, m))
        rows = []
        for doc_id, parts in by_doc.items():
            m = dict(parts[0][1], chunk_id=len(parts),  # written as chunk_id, renamed to n_chunks below
                     n_tokens=sum(p[1]["n_tokens"] for p in parts), n_label=sum(p[1]["n_label"] for p in parts),
                     truncated=any(p[1]["truncated"] for p in parts))
            rows.append((np.sum([p[0] for p in parts], axis=0), m))

    n_shards = max(1, min(n_shards, len(rows)))
    bounds = np.linspace(0, len(rows), n_shards + 1).astype(int)
    keys = [m["doc_id"] if merged else f"{m['doc_id']}:{m['chunk_id']}" for _, m in rows]
    plans = []
    for k in range(n_shards):
        p = dict(dummy=True, model="dummy", quantities=["full"], grad_layers=layers, act_layers=[], arrays=names,
                 shard=k, shard_rows_hash=store.order_hash(keys[bounds[k]:bounds[k + 1]]))
        h = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()
        plans.append((dict(p, plan_hash=h), h))
    for k, (plan, h) in enumerate(plans):
        shard_dir = ds_dir / f"shard{k}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        store.atomic_write_json(os.path.join(shard_dir, "plan.json"), plan)
        part = rows[bounds[k]:bounds[k + 1]]
        v = np.stack([r[0] for r in part])  # [B, L, d]
        norm = np.linalg.norm(v, axis=-1)
        unit = np.where(norm[..., None] > 0, v / np.maximum(norm[..., None], 1e-30), 0).astype(np.float16)
        writer = store.ShardWriter(str(shard_dir), names, hidden_size)
        writer.put(unit, norm.astype(np.float32), [r[1] for r in part])
        n = writer.close(store.index_metadata(dict(model="dummy", dataset=ds_dir.name, plan_hash=h)))
        if merged:
            import pyarrow.parquet as pq

            path = shard_dir / "index.parquet"
            t = pq.read_table(path)
            md = t.schema.metadata  # rename_columns drops it, and A's loader checks its plan_hash
            t = t.rename_columns(["n_chunks" if c == "chunk_id" else c for c in t.column_names])
            pq.write_table(t.replace_schema_metadata(md), path)
        store.atomic_write_json(os.path.join(shard_dir, "DONE.json"),
                                dict(rows=n, plan_hash=h, shard=k, n_shards=n_shards, dummy=True))
    if merged:  # try_write_manifest reads chunk_id; write the same manifest over doc_id keys
        shards = [dict(shard=k, dir=f"shard{k}", rows=int(bounds[k + 1] - bounds[k]), plan_hash=h)
                  for k, (_, h) in enumerate(plans)]
        store.atomic_write_json(os.path.join(ds_dir, "manifest.json"), dict(
            n_shards=n_shards, plan={}, shard_order=[s["dir"] for s in shards], shards=shards,
            total_rows=len(keys), doc_order_hash=store.order_hash(keys)))
    elif store.try_write_manifest(str(ds_dir), n_shards, [h for _, h in plans]) is None:
        raise GradStoreError(f"dummy store manifest was not written in {ds_dir}")
    return raw

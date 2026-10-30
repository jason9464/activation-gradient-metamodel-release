"""Reader dataset, centring guards, and collators (train right-pad / gen left-pad)."""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from main_run.c1.materialize import SIGN, MaterializeError, ReaderVectors
from main_run.reader.prompt import (
    EncodedSample,
    bundle_sum_blocks,
    encode,
    length_quantiles,
)

log = logging.getLogger("reader.data")

CENTER_WARN = 0.05
CENTER_ERROR = 1e-3
REPRESENTATIONS = ("bundle_sum", "per_document", "per_chunk", "per_token")


class ReaderDataError(ValueError):
    """Invalid reader sample, representation, or centring guard failure."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sealed_test_rows(jsonl: Path | str, split: str | None = None) -> int:
    """Rows on the sealed test split that a ``split`` filter keeps (``None`` keeps every row)."""
    if split not in (None, "test"):
        return 0
    return sum(1 for r in _read_jsonl(Path(jsonl)) if r.get("split") == "test")


def check_vec_rows(rows: Sequence[dict[str, Any]], index, name: str = "") -> int:
    """Every row with a ``vec_row`` must point at its own ``grad_bundle_id`` in ``bundle_index``.

    The vector lookup is positional (``vec_row`` indexes the .npy rows and the bundle_index rows), so a
    samples file joined against another reader dir would inject other bundles' vectors silently.
    Raises on the first mismatch; returns the number of rows checked.
    """
    ids = index["bundle_id"].tolist()
    n_ids = len(ids)

    def ok(r: dict[str, Any]) -> bool:
        v = int(r["vec_row"])
        return 0 <= v < n_ids and ids[v] == r.get("grad_bundle_id")

    n = 0
    for r in rows:
        if r.get("vec_row") is None:
            continue
        n += 1
        if not ok(r):
            n_bad = sum(1 for x in rows if x.get("vec_row") is not None and not ok(x))
            v = int(r["vec_row"])
            at = f"is bundle {ids[v]!r} in bundle_index" if 0 <= v < n_ids else f"is outside bundle_index ({n_ids} rows)"
            raise ReaderDataError(
                f"{name}: {r.get('sample_id')}: vec_row {v} {at} but the row's grad_bundle_id is "
                f"{r.get('grad_bundle_id')!r} ({n_bad} mismatching rows)"
            )
    return n


def centering_diagnostics(
    reader_dir: Path,
    *,
    weighting: str = "tokensum",
    layers: Sequence[int] | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Per-layer ‖u_i − μ‖ quantiles; warn/error near zero.

    Covers every bundle by default (train, val and test rows are all injected at some point), so a
    near-zero centred direction is caught once at startup instead of mid-run; ``split`` narrows it.
    """
    rv = ReaderVectors(reader_dir)
    layers = list(layers or rv.layers)
    mean = np.load(rv.dir / f"mean_unit_{weighting}.npy")
    mask = np.ones(len(rv.index), bool) if split is None else rv.index["split"].to_numpy() == split
    if not mask.any():
        raise ReaderDataError(f"no bundles with split {split!r} in {rv.dir}")
    rows = rv.index.loc[mask, "row"].to_numpy(np.int64)
    out: dict[str, Any] = {"weighting": weighting, "split": split or "all", "layers": {}, "n_rows": int(len(rows))}
    for L in layers:
        li = rv.layers.index(L)
        u = np.asarray(rv._unit(weighting, L)[rows], np.float32)
        d = np.linalg.norm(u - mean[li], axis=1)
        qs = {
            "min": float(d.min()),
            "p1": float(np.quantile(d, 0.01)),
            "p50": float(np.quantile(d, 0.5)),
            "p99": float(np.quantile(d, 0.99)),
            "max": float(d.max()),
            "frac_below_0.05": float((d < CENTER_WARN).mean()),
            "frac_below_1e-3": float((d < CENTER_ERROR).mean()),
        }
        out["layers"][str(L)] = qs
        if qs["min"] < CENTER_ERROR:
            raise ReaderDataError(
                f"layer {L}: a unit row is within {CENTER_ERROR} of the train mean "
                f"(min ‖u−μ‖={qs['min']:.2e}); centring would near-zero the vector"
            )
        if qs["min"] < CENTER_WARN:
            warnings.warn(
                f"layer {L}: min ‖u−μ‖={qs['min']:.4f} < {CENTER_WARN} (near-zero after centre)",
                stacklevel=2,
            )
    return out


class ReaderSamples(Dataset):
    """JSONL samples joined to bundle vectors for one representation mode."""

    def __init__(
        self,
        jsonl: Path | str,
        reader_dir: Path | str,
        *,
        tok,
        layers: Sequence[int],
        representation: str = "bundle_sum",
        center: bool = True,
        weighting: str = "tokensum",
        subset: str | None = None,
        max_len: int = 2048,
        for_generation: bool = False,
        split: str | None = None,
        conditions: Sequence[str] | None = None,
        unseal: bool = False,
        require_split: str | None = None,
    ):
        """``require_split``: every loaded row must be on this split (the training set passes "train",
        so a val or test row in the training file is refused instead of trained on)."""
        if representation not in REPRESENTATIONS:
            raise ReaderDataError(f"unknown representation {representation!r}; choose from {REPRESENTATIONS}")
        if representation != "bundle_sum":
            raise NotImplementedError(
                f"representation={representation!r} is not implemented yet "
                "(stubs only; bundle_sum is the main experiment)"
            )
        self.reader_dir = Path(reader_dir)
        self.rv = ReaderVectors(self.reader_dir)
        if not str(self.rv.meta.get("sign", "")).startswith("-dL/dh"):
            raise ReaderDataError(
                f"reader dir sign is {self.rv.meta.get('sign')!r}; expected {SIGN!r} "
                "(refusing a second negation)"
            )
        self.tok = tok
        self.layers = list(layers)
        self.representation = representation
        self.center = bool(center)
        self.weighting = weighting
        self.max_len = int(max_len)
        self.for_generation = bool(for_generation)
        self.blocks = bundle_sum_blocks(self.layers)

        rows = _read_jsonl(Path(jsonl))
        if split is not None:
            rows = [r for r in rows if r.get("split") == split]
        if conditions is not None:
            allow = set(conditions)
            rows = [r for r in rows if r["condition"] in allow]
        if subset == "oneq":
            rows = [r for r in rows if r.get("in_oneq")]
        elif subset is not None:
            raise ReaderDataError(f"unknown subset {subset!r}")
        # The seal is on the rows actually loaded, so split=None ("all") cannot reach test either.
        n_test = sum(1 for r in rows if r.get("split") == "test")
        if n_test and not unseal:
            raise ReaderDataError(
                f"{Path(jsonl).name}: {n_test} rows on the sealed test split (split={split!r}); "
                "pass --unseal once the rubric/judge/reader are fixed"
            )

        if require_split is not None:
            off = [r for r in rows if r.get("split") != require_split]
            if off:
                raise ReaderDataError(
                    f"{Path(jsonl).name}: {len(off)} rows not on split {require_split!r} "
                    f"(first: {off[0].get('sample_id')} split={off[0].get('split')!r})"
                )

        for r in rows:
            if r["condition"] == "correct" and r.get("vec_row") is None:
                raise ReaderDataError(f"{r.get('sample_id')}: correct row has null vec_row")
        check_vec_rows(rows, self.rv.index, Path(jsonl).name)

        self.rows = rows
        self._encoded: list[EncodedSample | None] = [None] * len(rows)
        self.length_stats = length_quantiles(
            [
                len(
                    encode(
                        tok,
                        blocks=self.blocks,
                        question=r["question"],
                        answer=r["answer"],
                        max_len=self.max_len,
                        for_generation=self.for_generation,
                    ).input_ids
                )
                for r in rows[: min(len(rows), 256)]
            ]
        )
        log.info(
            "ReaderSamples %s: n=%d layers=%s center=%s length(sample<=256)=%s",
            Path(jsonl).name,
            len(self.rows),
            self.layers,
            self.center,
            self.length_stats,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, i: int) -> EncodedSample:
        if self._encoded[i] is None:
            r = self.rows[i]
            self._encoded[i] = encode(
                self.tok,
                blocks=self.blocks,
                question=r["question"],
                answer=r["answer"],
                max_len=self.max_len,
                for_generation=self.for_generation,
            )
        return self._encoded[i]

    def _vectors_for(self, row: dict[str, Any]) -> torch.Tensor | None:
        """Return [K, d] fp32 vectors in ``self.layers`` order, or None for no_vector."""
        if row["condition"] == "no_vector" or row.get("vec_row") is None:
            return None
        v = self.rv.get([int(row["vec_row"])], self.weighting, center=self.center, layers=self.layers)[0]
        # get returns full (unit*norm) vectors; injection re-normalises, so direction is what matters.
        return v  # [n_layers, d]

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.rows[i]
        enc = self._encode(i)
        vec = self._vectors_for(r)  # near-zero centred rows are refused by centering_diagnostics
        K = len(enc.positions)
        if vec is None:
            vectors = torch.zeros((0, 1), dtype=torch.float32)  # placeholder; collator pads
            n_vec = 0
        else:
            if vec.shape[0] != K:
                raise ReaderDataError(
                    f"{r.get('sample_id')}: {vec.shape[0]} vectors for {K} placeholders "
                    f"(layers={self.layers})"
                )
            vectors = vec
            n_vec = K
        return {
            "input_ids": enc.input_ids,
            "labels": enc.labels,
            "positions": enc.positions,
            "pos_layer": enc.pos_layer,
            "vectors": vectors,
            "n_vec": n_vec,
            "sample_id": r["sample_id"],
            "qa_id": r["qa_id"],
            "bundle_id": r["bundle_id"],
            "condition": r["condition"],
            "split": r.get("split"),
            "question": r["question"],
            "answer": r["answer"],
            "question_type": r.get("question_type"),
            "question_generic": r.get("question_generic"),
            "slot": r.get("slot"),
            "in_oneq": r.get("in_oneq"),
        }


class ReaderCollator:
    """Pad sequences and stack variable-K injection tensors.

    ``side='right'`` for training; ``side='left'`` for generation (positions shifted by pad).
    """

    def __init__(self, tok, *, side: str = "right", d_model: int | None = None):
        if side not in ("right", "left"):
            raise ValueError(side)
        self.tok = tok
        self.side = side
        self.d_model = d_model
        self.pad_id = tok.pad_token_id
        if self.pad_id is None:
            raise ReaderDataError("tokenizer has no pad_token_id")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        B = len(examples)
        lengths = [len(e["input_ids"]) for e in examples]
        T = max(lengths)
        Kmax = max((e["n_vec"] for e in examples), default=0)
        d = self.d_model
        if d is None:
            for e in examples:
                if e["n_vec"] > 0:
                    d = int(e["vectors"].shape[-1])
                    break
            if d is None:
                d = 1

        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        positions = torch.zeros((B, max(Kmax, 1)), dtype=torch.long)
        mask = torch.zeros((B, max(Kmax, 1)), dtype=torch.bool)
        vectors = torch.zeros((B, max(Kmax, 1), d), dtype=torch.float32)

        meta = []
        for b, e in enumerate(examples):
            ids = e["input_ids"]
            n = len(ids)
            if self.side == "right":
                sl = slice(0, n)
                pad_left = 0
            else:
                pad_left = T - n
                sl = slice(pad_left, T)
            input_ids[b, sl] = torch.tensor(ids, dtype=torch.long)
            attention_mask[b, sl] = 1
            if e.get("labels") is not None and self.side == "right":
                labels[b, sl] = torch.tensor(e["labels"], dtype=torch.long)
            k = e["n_vec"]
            if k > 0:
                pos = [p + pad_left for p in e["positions"]]
                positions[b, :k] = torch.tensor(pos, dtype=torch.long)
                mask[b, :k] = True
                vectors[b, :k] = e["vectors"][:k].float()
            meta.append(
                {
                    "sample_id": e["sample_id"],
                    "qa_id": e["qa_id"],
                    "bundle_id": e["bundle_id"],
                    "condition": e["condition"],
                    "split": e.get("split"),
                    "question": e["question"],
                    "answer": e["answer"],
                    "question_type": e.get("question_type"),
                    "question_generic": e.get("question_generic"),
                    "slot": e.get("slot"),
                    "in_oneq": e.get("in_oneq"),
                }
            )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inject_vectors": vectors,
            "inject_positions": positions,
            "inject_mask": mask,
            "meta": meta,
        }
        if self.side == "right":
            batch["labels"] = labels
        return batch

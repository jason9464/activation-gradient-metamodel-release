"""Write the C1 reader dataset: per-bundle vectors from A's gradient store plus the join's samples.

Run after `join`. The output directory is self-contained (the reader does not need A's store):

  meta.json                      written last; layers, weightings, sign, source store, counts
  bundle_index.parquet           row, bundle_id, split, n_docs, n_label, norm_<weighting>_L<layer> (fp32)
  bundle_<weighting>_L<layer>.npy  fp16 unit rows [n_bundles, d]; vector = unit * norm
  mean_unit_<weighting>.npy      [n_layers, d] fp32 mean of the train bundles' unit rows (optional centering)
  train_correct.jsonl, train_no_vector_baseline.jsonl, eval_samples.jsonl
                                 the join's samples plus `vec_row`: the bundle row to inject (own bundle for
                                 correct, the partner's for shuffled, null for no_vector)
  with --per-doc: doc_index.parquet, doc_tokensum_L<layer>.npy, doc_mean_unit_tokensum.npy (one row per document)

Vectors are -dL/dh (the descent direction), already negated. Weightings (GradStore.bundle_vectors):
`tokensum` sums the document gradients (long documents weigh more); `docmean` sums each document's
gradient divided by its label count. Norm-matched injection keeps only the direction, so centering on
the train mean is offered at load time (ReaderVectors.get) rather than baked in.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from main_run.c1.constants import DATA_DIR, DEFAULT_LAYERS
from main_run.grad_io import WEIGHTINGS, GradStore, GradStoreError

SAMPLE_FILES = ("train_correct", "train_no_vector_baseline", "eval_samples")
SIGN = "-dL/dh: minus the summed residual-stream gradients (the descent direction)"


class MaterializeError(ValueError):
    """The join output and the gradient store do not fit together, or the output cannot be written."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _bundles_from_samples(samples: list[dict[str, Any]]) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """Every bundle a sample injects (sorted), its document list and its split."""
    docs_of: dict[str, list[str]] = {}
    split_of: dict[str, str] = {}
    for s in samples:
        if s["condition"] == "correct":
            split_of[s["bundle_id"]] = s["split"]
        b = s["grad_bundle_id"]
        if b is None:
            continue
        if docs_of.setdefault(b, list(s["grad_doc_ids"])) != list(s["grad_doc_ids"]):
            raise MaterializeError(f"bundle {b} has two different document lists in the samples")
    missing = sorted(set(docs_of) - set(split_of))
    if missing:
        raise MaterializeError(f"bundle {missing[0]} is injected but has no correct sample ({len(missing)} such)")
    return sorted(docs_of), docs_of, split_of


def _write_vectors(
    gs: GradStore,
    out_dir: Path,
    prefix: str,
    doc_lists: Sequence[Sequence[str]],
    layers: list[int],
    weightings: Sequence[str],
    train_mask: np.ndarray,
    block: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Write `<prefix>_<w>_L<l>.npy` per weighting and layer; return norms per `<w>_L<l>` and train means per w."""
    n = len(doc_lists)
    norms = {f"{w}_L{L}": np.empty(n, np.float32) for w in weightings for L in layers}
    means = {w: None for w in weightings}
    for li, L in enumerate(layers):
        units: dict[str, np.ndarray] = {}
        for s in range(0, n, block):
            vecs = gs.bundle_vectors(doc_lists[s:s + block], L, weightings)
            for w, v in vecs.items():
                v = -v  # descent direction
                nrm = np.linalg.norm(v, axis=1)
                if (nrm == 0).any():
                    raise MaterializeError(f"zero {w} vector at layer {L}, row {s + int(np.flatnonzero(nrm == 0)[0])}")
                if w not in units:
                    units[w] = np.empty((n, v.shape[1]), np.float16)
                units[w][s:s + len(v)] = (v / nrm[:, None]).astype(np.float16)
                norms[f"{w}_L{L}"][s:s + len(v)] = nrm
        for w, u in units.items():
            np.save(out_dir / f"{prefix}_{w}_L{L}.npy", u)
            if means[w] is None:
                means[w] = np.zeros((len(layers), u.shape[1]), np.float32)
            means[w][li] = u[train_mask].astype(np.float32).mean(0)
    return norms, means


def materialize(
    grad_root: Path,
    *,
    joined_dir: Path | None = None,
    out_dir: Path | None = None,
    layers: Sequence[int] | None = None,
    weightings: Sequence[str] = WEIGHTINGS,
    per_doc: bool = False,
    block: int = 1024,
) -> dict[str, Any]:
    t0 = time.time()
    layers = list(layers or DEFAULT_LAYERS)
    joined_dir = Path(joined_dir or (DATA_DIR / "joined"))
    out_dir = Path(out_dir or (DATA_DIR / "reader"))
    report_path = joined_dir / "validation_report.json"
    if not report_path.exists():
        raise MaterializeError(f"missing {report_path}; run python -m main_run.c1.join first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = {name: _read_jsonl(joined_dir / f"{name}.jsonl") for name in SAMPLE_FILES}

    try:
        gs = GradStore(grad_root, layers)
    except GradStoreError as e:
        raise MaterializeError(str(e)) from e
    joined_hash = report["grad_report"]["doc_order_hash"]
    if joined_hash != gs.manifest["doc_order_hash"]:
        raise MaterializeError(
            f"the join was validated against another gradient store (doc_order_hash {joined_hash[:8]}, "
            f"this store {gs.manifest['doc_order_hash'][:8]}); rerun join with --grads {grad_root}"
        )

    all_samples = [s for rows in samples.values() for s in rows]
    bundle_ids, docs_of, split_of = _bundles_from_samples(all_samples)
    doc_lists = [docs_of[b] for b in bundle_ids]
    train_mask = np.array([split_of[b] == "train" for b in bundle_ids])
    center_on = "train bundles" if train_mask.any() else "all bundles (the join has no train split)"
    if not train_mask.any():
        train_mask = np.ones(len(bundle_ids), bool)
    doc_ids = sorted({d for b in bundle_ids for d in docs_of[b]}) if per_doc else []
    train_docs = {d for b, m in zip(bundle_ids, train_mask) if m for d in docs_of[b]}

    d_model = gs._array(gs.shards[0], layers[0]).shape[1]
    need = (len(bundle_ids) * len(weightings) + len(doc_ids)) * len(layers) * d_model * 2
    out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    if free < need + 2**30:
        raise MaterializeError(
            f"not enough disk at {out_dir}: need {need / 2**30:.1f} GiB + 1 GiB margin, free {free / 2**30:.1f} GiB"
        )
    (out_dir / "meta.json").unlink(missing_ok=True)  # meta.json marks a finished directory

    norms, means = _write_vectors(gs, out_dir, "bundle", doc_lists, layers, list(weightings), train_mask, block)
    for w, m in means.items():
        np.save(out_dir / f"mean_unit_{w}.npy", m)
    index = pd.DataFrame(
        {
            "row": np.arange(len(bundle_ids)),
            "bundle_id": bundle_ids,
            "split": [split_of[b] for b in bundle_ids],
            "n_docs": [len(docs_of[b]) for b in bundle_ids],
            "n_label": [sum(gs.doc_n_label(d) for d in docs_of[b]) for b in bundle_ids],
        }
    )
    for k, v in norms.items():
        index[f"norm_{k}"] = v
    index.to_parquet(out_dir / "bundle_index.parquet", index=False)

    if per_doc:
        doc_train = np.array([d in train_docs for d in doc_ids])
        dnorms, dmeans = _write_vectors(gs, out_dir, "doc", [[d] for d in doc_ids], layers, ["tokensum"], doc_train, block)
        np.save(out_dir / "doc_mean_unit_tokensum.npy", dmeans["tokensum"])
        dindex = pd.DataFrame({"row": np.arange(len(doc_ids)), "doc_id": doc_ids,
                               "n_label": [gs.doc_n_label(d) for d in doc_ids]})
        for k, v in dnorms.items():
            dindex[f"norm_{k}"] = v
        dindex.to_parquet(out_dir / "doc_index.parquet", index=False)

    row_of = {b: i for i, b in enumerate(bundle_ids)}
    for name, rows in samples.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for s in rows:
                b = s["grad_bundle_id"]
                f.write(json.dumps({**s, "vec_row": row_of[b] if b is not None else None}, sort_keys=True) + "\n")

    meta = {
        "layers": layers,
        "weightings": {
            "tokensum": "sum of the documents' gradient sums (long documents weigh more)",
            "docmean": "sum of each document's gradient sum divided by its label count",
        },
        "weightings_written": list(weightings),
        "sign": SIGN,
        "storage": "fp16 unit rows; fp32 norms in bundle_index.parquet as norm_<weighting>_L<layer>; vector = unit * norm",
        "centering": "mean_unit_<weighting>.npy: [layer, d] mean of the unit rows of the " + center_on,
        "n_bundles": len(bundle_ids),
        "n_bundles_by_split": {k: int(v) for k, v in index["split"].value_counts().sort_index().items()},
        "n_docs": len(doc_ids) if per_doc else None,
        "n_samples": {name: len(rows) for name, rows in samples.items()},
        "grad_store": {"path": str(grad_root), "doc_order_hash": gs.manifest["doc_order_hash"],
                       "total_rows": gs.manifest["total_rows"]},
        "joined_dir": str(joined_dir),
        "wall_s": round(time.time() - t0, 1),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


class ReaderVectors:
    """A finished reader directory: injection vectors for sample `vec_row`s."""

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise MaterializeError(f"no meta.json in {self.dir}: materialize did not finish")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.layers = list(self.meta["layers"])
        self.index = pd.read_parquet(self.dir / "bundle_index.parquet")
        self._mm: dict[tuple[str, int], np.ndarray] = {}
        self._mean: dict[str, np.ndarray] = {}
        self._norm: dict[tuple[str, int], np.ndarray] = {}

    def _unit(self, weighting: str, layer: int) -> np.ndarray:
        key = (weighting, layer)
        if key not in self._mm:
            self._mm[key] = np.load(self.dir / f"bundle_{weighting}_L{layer}.npy", mmap_mode="r")
        return self._mm[key]

    def get(
        self,
        rows: Sequence[int],
        weighting: str,
        center: bool = False,
        layers: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """[n, len(layers), d] fp32 -dL/dh bundle vectors; `center` subtracts the train mean unit row first.

        `layers` selects by layer index (not by position in the file). Default is every layer in
        `meta.layers` order. A missing layer raises; the returned axis-1 order matches `layers`.
        """
        if weighting not in self.meta["weightings_written"]:
            raise MaterializeError(f"weighting {weighting!r} not written; have {self.meta['weightings_written']}")
        want = list(self.layers if layers is None else layers)
        dir_index: list[int] = []
        for L in want:
            try:
                dir_index.append(self.layers.index(L))
            except ValueError as e:
                raise MaterializeError(f"layer {L} not in reader dir layers {self.layers}") from e
        rows = np.asarray(rows, np.int64)
        if center and weighting not in self._mean:
            self._mean[weighting] = np.load(self.dir / f"mean_unit_{weighting}.npy")
        mean = self._mean.get(weighting)
        out = None
        for oi, (L, li) in enumerate(zip(want, dir_index)):
            u = np.asarray(self._unit(weighting, L)[rows], np.float32)
            if center:
                u = u - mean[li]
            if out is None:
                out = np.empty((len(rows), len(want), u.shape[1]), np.float32)
            key = (weighting, L)
            if key not in self._norm:
                self._norm[key] = self.index[f"norm_{weighting}_L{L}"].to_numpy(np.float32)
            out[:, oi] = u * self._norm[key][rows, None]
        return torch.from_numpy(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the C1 reader dataset (bundle vectors + samples)")
    p.add_argument("--grads", type=Path, required=True, help="A's dataset directory (manifest.json and shard*/)")
    p.add_argument("--joined", type=Path, default=None, help="join output (default data/c1/joined)")
    p.add_argument("--out", type=Path, default=None, help="output directory (default data/c1/reader)")
    p.add_argument("--layers", type=int, nargs="+", default=None, help=f"default: {DEFAULT_LAYERS}")
    p.add_argument("--weightings", nargs="+", choices=WEIGHTINGS, default=list(WEIGHTINGS))
    p.add_argument("--per-doc", action="store_true", help="also write one vector per document (~6.6 GB at 8 layers)")
    p.add_argument("--block", type=int, default=1024, help="bundles per read")
    args = p.parse_args(argv)
    meta = materialize(
        args.grads,
        joined_dir=args.joined,
        out_dir=args.out,
        layers=args.layers,
        weightings=args.weightings,
        per_doc=args.per_doc,
        block=args.block,
    )
    print(json.dumps({k: meta[k] for k in ("n_bundles", "n_bundles_by_split", "n_docs", "n_samples", "wall_s")}, indent=2))
    print(f"wrote {args.out or DATA_DIR / 'reader'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

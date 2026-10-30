"""Build the matched +h reader directory from the existing extraction stores (CPU only).

The output has exactly the gradient reader directory's bundle row order, training and
evaluation JSONL bytes, and three layers. Its fp16 unit rows are +h token sums with
position 0 excluded; its fp32 centring mean uses *all* C1 train bundles. The
gradient directory's metadata is never rewritten. The existing reader loader needs
the small activation compatibility wrapper in runtime.py to accept truthful +h sign.

Usage:
  python main_run/experiment2/build_act_dir.py estimate [--n-bundles 1000]
  python main_run/experiment2/build_act_dir.py build [--out D] [--c1-store D] [--beh-store D]
  python main_run/experiment2/build_act_dir.py check [--out D]
  python main_run/experiment2/build_act_dir.py compare --ref AO_ACT_DIR [--out D]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
from agm import core, store  # noqa: E402

BLOCK = 128
MARGIN_BYTES = 50 * 2**30
EXPECTED_TRAIN_ROWS = 56_654
DATA_FILES = ("train_correct.jsonl", "train_no_vector_baseline.jsonl", "eval_samples.jsonl",
              "beh_eval_samples.jsonl")


class ActStore:
    """Read stored, norm-scaled full activation sums across all chunks of each document."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.index, self.manifest = store.load_index(str(self.path))
        self.shards = list(self.manifest["shard_order"])
        self.pos_of = self.index.groupby("doc_id", sort=False).indices
        self.shard = self.index["shard"].to_numpy()
        self.row = self.index["row"].to_numpy(np.int64)
        self.mm = {}
        self.norm = {}
        exclusions, gradient_exclusions = [], []
        for shard in self.shards:
            plan = self.path / shard / "plan.json"
            if not plan.exists():
                raise ValueError(f"{plan}: position-0 rule is unknown")
            rule = json.loads(plan.read_text())["full_skip_pos0"]
            exclusions.append(core.skip_for(rule, "act"))
            gradient_exclusions.append(core.skip_for(rule, "grad"))
        if exclusions != [True] * len(self.shards):
            raise ValueError(f"{self.path}: full activations must exclude position 0 in every shard")
        if any(gradient_exclusions):
            raise ValueError(f"{self.path}: the planned gradient comparison retains position 0")
        for layer in C.LAYERS:
            name = core.array_name("full", "act", layer)
            col = f"norm_{name}"
            if col not in self.index:
                raise ValueError(f"{self.path}: missing {col}")
            self.norm[layer] = self.index[col].to_numpy(np.float32)
            if not np.isfinite(self.norm[layer]).all():
                raise ValueError(f"{self.path}: nonfinite {col}")
            for shard in self.shards:
                p = self.path / shard / f"{name}.npy"
                if not p.exists():
                    raise ValueError(f"{p}: missing activation array")
                mm = np.load(p, mmap_mode="r")
                if mm.ndim != 2 or mm.shape[0] != int((self.shard == shard).sum()):
                    raise ValueError(f"{p}: array/index row mismatch")
                self.mm[shard, layer] = mm
        self.width = int(self.mm[self.shards[0], C.LAYERS[0]].shape[1])
        if any(v.shape[1] != self.width for v in self.mm.values()):
            raise ValueError(f"{self.path}: activation width differs across layers or shards")

    def bundle_sums(self, lists: list[list[str]], layer: int) -> np.ndarray:
        positions, sizes = [], []
        for docs in lists:
            if not docs:
                raise ValueError("empty bundle document list")
            chunks = []
            for doc in docs:
                if doc not in self.pos_of:
                    raise ValueError(f"{self.path}: missing document {doc}")
                chunks.append(self.pos_of[doc])
            p = np.concatenate(chunks)
            positions.append(p)
            sizes.append(len(p))
        pos = np.concatenate(positions)
        vec = np.empty((len(pos), self.width), np.float32)
        for shard in np.unique(self.shard[pos]):
            mask = self.shard[pos] == shard
            rows = self.row[pos[mask]]
            vec[mask] = self.mm[shard, layer][rows].astype(np.float32)
        vec *= self.norm[layer][pos, None]
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        sums = np.add.reduceat(vec, starts, axis=0)
        if not np.isfinite(sums).all():
            raise ValueError(f"{self.path}: nonfinite activation sum at layer {layer}")
        return sums


def c1_store_path() -> Path:
    meta = json.loads((C.C1_DIR / "meta.json").read_text())
    return Path(meta["grad_store"]["path"])


def document_lists(grad_dir: Path, index: pd.DataFrame) -> list[list[str]]:
    """Recover every bundle's ordered extraction documents from the original sample rows."""
    docs = {}
    for name in DATA_FILES:
        if name == "train_no_vector_baseline.jsonl":
            continue
        for r in C.read_jsonl(grad_dir / name):
            bid, ds = r.get("grad_bundle_id"), r.get("grad_doc_ids")
            if bid is None or not ds:
                continue
            if bid in docs and docs[bid] != ds:
                raise ValueError(f"{bid}: inconsistent grad_doc_ids across {name}")
            docs[bid] = list(ds)
    missing = sorted(set(index.bundle_id) - set(docs))
    if missing:
        raise ValueError(f"{len(missing)} bundles have no extraction document list; first {missing[:3]}")
    lists = [docs[bid] for bid in index.bundle_id]
    wrong = [bid for bid, ds, n in zip(index.bundle_id, lists, index.n_docs) if len(ds) != int(n)]
    if wrong:
        raise ValueError(f"{len(wrong)} bundles have n_docs mismatch; first {wrong[:3]}")
    return lists


def _unit(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(sums, axis=1).astype(np.float32)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("zero or nonfinite activation bundle norm")
    return sums / norms[:, None], norms


def estimate(grad_dir: Path, c1_store: Path, beh_store: Path, n_bundles: int) -> dict:
    """Read a bounded slice with the full directory's C1/behaviour source mix."""
    if n_bundles < 2:
        raise ValueError("estimate requires at least two bundles")
    index = pd.read_parquet(grad_dir / "bundle_index.parquet")
    lists = document_lists(grad_dir, index)
    stores = {"c1": ActStore(c1_store), "behaviour": ActStore(beh_store)}
    counts = {source: int(index.source.eq(source).sum()) for source in stores}
    if min(counts.values()) == 0 or sum(counts.values()) != len(index):
        raise ValueError(f"unexpected source counts: {counts}")
    target = min(n_bundles, len(index))
    n_c1 = max(1, min(counts["c1"], round(target * counts["c1"] / len(index))))
    n_beh = target - n_c1
    if n_beh < 1:
        n_beh, n_c1 = 1, target - 1
    elif n_beh > counts["behaviour"]:
        n_beh, n_c1 = counts["behaviour"], target - counts["behaviour"]
    chosen = {}
    for source, n_pick in (("c1", n_c1), ("behaviour", n_beh)):
        rows = np.flatnonzero(index.source.eq(source).to_numpy())
        chosen[source] = rows[np.linspace(0, len(rows) - 1, n_pick, dtype=int)]
    t0 = time.monotonic()
    for source, act in stores.items():
        rows = chosen[source]
        for layer in C.LAYERS:
            for start in range(0, len(rows), BLOCK):
                ids = rows[start:start + BLOCK]
                _unit(act.bundle_sums([lists[i] for i in ids], layer))
    seconds = time.monotonic() - t0
    measured = sum(map(len, chosen.values()))
    return dict(n_measured_bundles=measured, n_total_bundles=len(index), elapsed_seconds=seconds,
                estimated_full_seconds=seconds * len(index) / measured,
                n_by_source={k: len(v) for k, v in chosen.items()}, layers=list(C.LAYERS),
                note="I/O cache and source mix can change the full-build time")


def build(out: Path, grad_dir: Path, c1_store: Path, beh_store: Path) -> dict:
    out, grad_dir = Path(out), Path(grad_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; refusing to overwrite")
    index = pd.read_parquet(grad_dir / "bundle_index.parquet")
    if not np.array_equal(index["row"].to_numpy(), np.arange(len(index))):
        raise ValueError("gradient bundle index row column is not contiguous")
    if index.bundle_id.duplicated().any():
        raise ValueError("gradient bundle index has duplicate ids")
    if set(index.source) != {"c1", "behaviour"}:
        raise ValueError(f"unexpected sources {set(index.source)}")
    lists = document_lists(grad_dir, index)
    stores = {"c1": ActStore(c1_store), "behaviour": ActStore(beh_store)}
    if stores["c1"].width != stores["behaviour"].width:
        raise ValueError("C1 and behaviour activation widths differ")
    n, width = len(index), stores["c1"].width
    train = (index.source.eq("c1") & index.split.eq("train")).to_numpy()
    if not train.any():
        raise ValueError("no C1 training bundles for centring")
    required = n * width * 2 * len(C.LAYERS) + MARGIN_BYTES
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < required:
        raise ValueError(f"insufficient disk for {required / 2**30:.1f} GiB including safety margin")
    out.mkdir(exist_ok=True)
    means = []
    for layer in C.LAYERS:
        arr = np.lib.format.open_memmap(out / f"bundle_tokensum_L{layer}.npy", mode="w+", dtype=np.float16,
                                        shape=(n, width))
        norms = np.empty(n, np.float32)
        acc = np.zeros(width, np.float64)
        for source, act in stores.items():
            rows = np.flatnonzero(index.source.eq(source).to_numpy())
            for start in range(0, len(rows), BLOCK):
                ids = rows[start:start + BLOCK]
                unit, nr = _unit(act.bundle_sums([lists[i] for i in ids], layer))
                arr[ids] = unit.astype(np.float16)
                norms[ids] = nr
                tr = train[ids]
                if tr.any():
                    # Match the gradient directory: centre on the rows after fp16 storage.
                    acc += np.asarray(arr[ids[tr]], np.float32).sum(axis=0, dtype=np.float64)
        arr.flush()
        if not np.isfinite(arr).all():
            raise ValueError(f"layer {layer}: nonfinite stored unit row")
        mean = (acc / int(train.sum())).astype(np.float32)
        centred_min = min(float(np.linalg.norm(np.asarray(arr[i:i + BLOCK], np.float32) - mean, axis=1).min())
                            for i in range(0, n, BLOCK))
        if centred_min < 1e-3:
            raise ValueError(f"layer {layer}: centred unit norm {centred_min:.3g} < 1e-3")
        means.append(mean)
        index[f"norm_tokensum_L{layer}"] = norms
        del arr
    np.save(out / "mean_unit_tokensum.npy", np.stack(means))
    index.to_parquet(out / "bundle_index.parquet", index=False)
    hashes = {}
    for name in DATA_FILES:
        shutil.copyfile(grad_dir / name, out / name)
        hashes[name] = C.sha256_file(out / name)
    n_train_rows = sum(1 for _ in (out / "train_correct.jsonl").open(encoding="utf-8"))
    if n_train_rows != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"expected {EXPECTED_TRAIN_ROWS} exact training rows; found {n_train_rows}")
    grad_meta = json.loads((grad_dir / "meta.json").read_text())
    meta = dict(layers=list(C.LAYERS), weightings_written=["tokensum"],
                weightings={"tokensum": "sum of stored full activations over all chunks of each document in bundle"},
                quantity="activation", sign="+h: full residual activation sum; position 0 excluded",
                position_zero_asymmetry="activation full sums exclude position 0; the existing gradient full sums retain it",
                storage="fp16 unit rows, fp32 norms in bundle_index.parquet",
                centering=f"mean of all {int(train.sum())} C1 train bundle fp16 unit rows",
                n_bundles=n, n_c1_train=int(train.sum()), n_train_samples=n_train_rows,
                source_gradient_dir=str(grad_dir), source_gradient_meta_sha256=C.sha256_file(grad_dir / "meta.json"),
                original_bundle_order_hash=grad_meta.get("bundle_order_hash"),
                stores={key: dict(path=str(st.path), doc_order_hash=st.manifest.get("doc_order_hash"),
                                  full_act_excludes_pos0=True) for key, st in stores.items()},
                copied_jsonl_sha256=hashes)
    C.write_json(out / "meta.json", meta)  # completion marker, always last
    return meta


def check(out: Path) -> dict:
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text())
    if meta.get("quantity") != "activation" or tuple(meta["layers"]) != C.LAYERS:
        raise ValueError("not the three-layer activation directory")
    grad = Path(meta["source_gradient_dir"])
    idx = pd.read_parquet(out / "bundle_index.parquet")
    original = pd.read_parquet(grad / "bundle_index.parquet")
    if not idx[["row", "bundle_id", "source", "split"]].equals(
            original[["row", "bundle_id", "source", "split"]]):
        raise ValueError("bundle rows differ from the gradient directory")
    for name in DATA_FILES:
        digest = C.sha256_file(out / name)
        if digest != C.sha256_file(grad / name) or digest != meta["copied_jsonl_sha256"][name]:
            raise ValueError(f"{name} differs from original")
        for r in C.read_jsonl(out / name):
            vr = r.get("vec_row")
            if vr is not None and (int(vr) < 0 or int(vr) >= len(idx)
                                   or idx.iloc[int(vr)].bundle_id != r.get("grad_bundle_id")):
                raise ValueError(f"{r['sample_id']}: vec_row/grad_bundle_id mismatch")
    if sum(1 for _ in (out / "train_correct.jsonl").open(encoding="utf-8")) != EXPECTED_TRAIN_ROWS:
        raise ValueError("the activation training file differs from the 56,654-row recipe")
    train = (idx.source.eq("c1") & idx.split.eq("train")).to_numpy()
    mean = np.load(out / "mean_unit_tokensum.npy")
    checks = {}
    for li, layer in enumerate(C.LAYERS):
        arr = np.load(out / f"bundle_tokensum_L{layer}.npy", mmap_mode="r")
        if arr.shape != (len(idx), mean.shape[1]) or arr.dtype != np.float16:
            raise ValueError(f"layer {layer}: wrong shape or dtype")
        acc = np.zeros(mean.shape[1], np.float64)
        min_norm, max_norm, min_centred = float("inf"), 0., float("inf")
        for start in range(0, len(idx), BLOCK):
            u = np.asarray(arr[start:start + BLOCK], np.float32)
            nu = np.linalg.norm(u, axis=1)
            if not np.isfinite(u).all() or (nu < 0.99).any() or (nu > 1.01).any():
                raise ValueError(f"layer {layer}: invalid unit rows")
            min_norm, max_norm = min(min_norm, float(nu.min())), max(max_norm, float(nu.max()))
            min_centred = min(min_centred, float(np.linalg.norm(u - mean[li], axis=1).min()))
            subset = train[start:start + BLOCK]
            if subset.any():
                acc += u[subset].sum(axis=0, dtype=np.float64)
        recomputed = acc / int(train.sum())
        max_mean_abs = float(np.max(np.abs(recomputed - mean[li])))
        if max_mean_abs > 1e-6 or min_centred < 1e-3:
            raise ValueError(f"layer {layer}: mean/centering failure {max_mean_abs}, {min_centred}")
        norms = idx[f"norm_tokensum_L{layer}"].to_numpy(float)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError(f"layer {layer}: invalid norms")
        checks[str(layer)] = dict(min_unit_norm=min_norm, max_unit_norm=max_norm,
                                  min_centred_norm=min_centred, max_mean_abs_error=max_mean_abs)
    return dict(pass_=True, n_bundles=len(idx), n_train_samples=meta["n_train_samples"], layers=checks)


def row_cos(a, b):
    return (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)


def compare(out: Path, ref: Path, *, min_cos: float = .999,
            max_relative_norm: float = .01) -> dict:
    """Compare new rows with the AO branch's earlier eval-only activation directory on shared bundles."""
    out, ref = Path(out), Path(ref)
    oi, ri = pd.read_parquet(out / "bundle_index.parquet"), pd.read_parquet(ref / "bundle_index.parquet")
    ref_row = dict(zip(ri.bundle_id, ri.row))
    if len(ref_row) != len(ri) or not set(ref_row).issubset(set(oi.bundle_id)):
        raise ValueError("reference has duplicate or absent bundle ids")
    pairs = [(int(row), int(ref_row[bid])) for row, bid in zip(oi.row, oi.bundle_id) if bid in ref_row]
    if not pairs:
        raise ValueError("no shared bundle ids with reference directory")
    new_rows, old_rows = (np.array(x, int) for x in zip(*pairs))
    new_meta = json.loads((out / "meta.json").read_text())
    ref_meta = json.loads((ref / "meta.json").read_text())
    layers = sorted(set(new_meta["layers"]) & set(ref_meta["layers"]))
    if not layers:
        raise ValueError("no common layers")
    nm, rm = np.load(out / "mean_unit_tokensum.npy"), np.load(ref / "mean_unit_tokensum.npy")
    rep = {}
    for layer in layers:
        a = np.asarray(np.load(out / f"bundle_tokensum_L{layer}.npy", mmap_mode="r")[new_rows], np.float32)
        b = np.asarray(np.load(ref / f"bundle_tokensum_L{layer}.npy", mmap_mode="r")[old_rows], np.float32)
        ma, mb = nm[new_meta["layers"].index(layer)], rm[ref_meta["layers"].index(layer)]
        an = oi[f"norm_tokensum_L{layer}"].to_numpy(float)[new_rows]
        bn = ri[f"norm_tokensum_L{layer}"].to_numpy(float)[old_rows]
        cos, centred = row_cos(a, b), row_cos(a - ma, b - mb)
        worst_norm = float(np.max(np.abs(an - bn) / np.maximum(bn, 1e-30)))
        if float(cos.min()) < min_cos or worst_norm > max_relative_norm:
            raise ValueError(f"layer {layer}: activation rows differ from reference: "
                             f"cos {cos.min():.6f}, centred {centred.min():.6f}, relative norm {worst_norm:.6f}")
        rep[str(layer)] = dict(min_unit_cos=float(cos.min()), median_unit_cos=float(np.median(cos)),
                               min_centred_cos=float(centred.min()),
                               median_centred_cos=float(np.median(centred)),
                               max_relative_norm_difference=worst_norm,
                               c1_mean_cos=float(row_cos(ma[None], mb[None])[0]))
    return dict(reference=str(ref), new=str(out), n_shared=len(pairs), layers=rep,
                note="Reference mean may use a C1 sample; new mean uses all C1 training bundles. "
                     "Unit and norm agreement is the extraction check; centred cosine also reflects mean sampling.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("estimate", "build", "check", "compare"))
    ap.add_argument("--out", type=Path, default=C.ACT_DIR)
    ap.add_argument("--grad-dir", type=Path, default=C.GRAD_DIR)
    ap.add_argument("--c1-store", type=Path, default=None)
    ap.add_argument("--beh-store", type=Path, default=C.BEH_STORE)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--n-bundles", type=int, default=1000, help="estimate: number of bundles in the timed slice")
    a = ap.parse_args(argv)
    if a.command == "estimate":
        result = estimate(a.grad_dir, a.c1_store or c1_store_path(), a.beh_store, a.n_bundles)
    elif a.command == "build":
        result = build(a.out, a.grad_dir, a.c1_store or c1_store_path(), a.beh_store)
    elif a.command == "check":
        result = check(a.out)
    else:
        if a.ref is None:
            ap.error("compare requires --ref")
        result = compare(a.out, a.ref)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

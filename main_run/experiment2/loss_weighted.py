#!/usr/bin/env python
"""Feed loss-weighted activations to the Experiment 2 activation reader (no training).

Experiment 1 closed most of the linear-probe gap by weighting activations with per-document
loss. Experiment 2 found the gradient reader ahead of the activation reader on P(Yes)
(0.943 vs 0.733). This script settles that tension at the reader level: convert the
forward-baseline bundle vectors into the activation reader's three-layer format, score
P(Yes), and summarise.

Variants (layers 10/20/30, held-out 530 bundles, rows in bundles.parquet beh_test order):
  act       Experiment 2 activation input (sanity: pooled rare/quirk/backdoor AUROC ≈ 0.733)
  actresp   response-token activation sum
  actrespl  response units weighted by summed token loss
  actrespm  response units weighted by mean per-label loss
  actrespn  response units weighted by label count (Experiment 1 length control)
  actrespg  response units weighted by gradient norm (reference; uses the backward pass)
  *_shuf    same weight family; one within-bundle permutation reused across layers (control)

  Primary gaps: actrespl−actresp, actrespl−actrespn (pooled + rare-only).

  python main_run/experiment2/loss_weighted.py build-dirs \\
      --vec artifacts/fwdb/data/vec \\
      --bundles artifacts/fwdb/data/bundles.parquet \\
      --beh-jsonl artifacts/fwdb/data/raw/beh/behaviour_bundles_v1/bundles.jsonl \\
      --fetch-twin-mean \\
      --out artifacts/fwdb/data/fwd/readerdirs_L3

  python main_run/experiment2/loss_weighted.py build-shuffle \\
      --vec artifacts/fwdb/data/vec --ext PATH/TO/ext \\
      --bundles ... --beh-jsonl ... [--seed 0]

  python main_run/experiment2/loss_weighted.py pyes \\
      --dirs-root artifacts/fwdb/data/fwd/readerdirs_L3 \\
      --adapter PATH/OR/HF --run-json PATH --gpu 0

  python main_run/experiment2/loss_weighted.py summarise --run-dir RUN

  python main_run/experiment2/loss_weighted.py run --vec ... --mean ... --out ... --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse as A  # noqa: E402
import common as C  # noqa: E402

LAYERS = C.LAYERS
SPLIT = C.SPLIT
N_HELD = C.N_HELD_BEHAVIOUR
ACT_SIGN = "+h: summed residual-stream activations (not a gradient; nothing negated)"
# Experiment 2 activation_reader_dir mean (scores.meta.json identity.reader_dir_hashes).
EXPECTED_MEAN_SHA256 = "590ad049f0e46813945b6e1f28771c48fad204aca924bf4096b0df03dd903e1f"
# Pooled rare/quirk/backdoor AUROC from Experiment 2 activation reader × activation.
SANITY_ACT_AUC = 0.7330663997330664
SANITY_TOL = 0.002

# actrespn = response units × label count (Experiment 1's own control for loss vs length).
BASE_VARIANTS = ("act", "actresp", "actrespl", "actrespm", "actrespn", "actrespg")
# Shuffled controls only make sense for weighted families (optional once actrespn is scored).
SHUFFLE_OF = {"actrespl": "loss", "actrespm": "lossmean", "actrespg": "gradnorm"}
SHUFFLE_VARIANTS = tuple(f"{v}_shuf" for v in SHUFFLE_OF)
ALL_VARIANTS = BASE_VARIANTS + SHUFFLE_VARIANTS
# Primary contrasts isolate loss weighting from the response-token cut.
PRIMARY_GAPS = (
    ("actrespl", "actresp"),   # loss weighting on top of response-only
    ("actrespl", "actrespn"),  # loss vs label-count (Experiment 1 control)
    ("actrespm", "actresp"),
    ("actrespm", "actrespn"),
)
ACTRESP_DROP_WARN = 0.05  # if actresp pooled AUROC is this far below act, the reader can't read response-only
SHUFFLE_REBUILD_COS_MIN = 0.999
# Substitute mean (the activation twin's): same formula, C1-train fp16 unit mean, layers 10/20/30.
TWIN_MEAN_REPO = "jason9464/sprint-data-2610"
TWIN_MEAN_PATH = "behaviour_readers/sft_scratch_L3act_beh_7082/mean_unit_tokensum.npy"

DEFAULT_VEC = C.REPO / "artifacts" / "fwdb" / "data" / "vec"
DEFAULT_BUNDLES = C.REPO / "artifacts" / "fwdb" / "data" / "bundles.parquet"
DEFAULT_BEH = (C.REPO / "artifacts" / "fwdb" / "data" / "raw" / "beh" /
               "behaviour_bundles_v1" / "bundles.jsonl")
DEFAULT_DIRS = C.REPO / "artifacts" / "fwdb" / "data" / "fwd" / "readerdirs_L3"
DEFAULT_ADAPTER = Path(os.environ.get(
    "E2_ACT_ADAPTER",
    C.REPO / "artifacts" / "experiment2_out" / "runs" / "reader_scratch_L3_act_beh" /
    "adapters" / "step_007082"))
DEFAULT_RUN_JSON = Path(os.environ.get(
    "E2_ACT_RUN_JSON",
    C.REPO / "artifacts" / "experiment2_out" / "runs" / "reader_scratch_L3_act_beh" / "RUN.json"))
HF_READER = "me-r/sft-scratch-L3-act-beh-7082"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [loss_weighted] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def held_frame(bundles: Path) -> pd.DataFrame:
    lab = pd.read_parquet(bundles)
    held = lab[lab.split == SPLIT].reset_index(names="vec_row")
    if len(held) != N_HELD:
        raise ValueError(f"{bundles}: {len(held)} {SPLIT} rows, expected {N_HELD}")
    if held.bundle_id.duplicated().any():
        raise ValueError(f"{bundles}: duplicate held-out bundle ids")
    return held


def load_doc_ids(beh_jsonl: Path, bundle_ids: list[str]) -> dict[str, list[str]]:
    want = set(bundle_ids)
    out = {}
    with Path(beh_jsonl).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            bid = r.get("bundle_id")
            if bid in want:
                out[bid] = list(r["doc_ids"])
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"{beh_jsonl}: missing {len(missing)} held-out bundles; first {missing[:3]}")
    return out


def fetch_twin_mean(dest: Path) -> Path:
    """Download the activation twin's mean (same formula as Experiment 2) into dest."""
    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"fetching substitute mean from {TWIN_MEAN_REPO}:{TWIN_MEAN_PATH}")
    cached = hf_hub_download(TWIN_MEAN_REPO, TWIN_MEAN_PATH, repo_type="model")
    shutil.copyfile(cached, dest)
    return dest


def resolve_mean(mean: Path | None, template: Path | None, *, fetch_twin: bool = False,
                 twin_cache: Path | None = None) -> Path:
    if mean is not None:
        path = Path(mean)
    elif template is not None:
        path = Path(template) / "mean_unit_tokensum.npy"
    elif fetch_twin:
        path = fetch_twin_mean(twin_cache or (C.REPO / "artifacts" / "fwdb" / "data" /
                                              "mean_unit_tokensum_twin.npy"))
    else:
        raise ValueError("pass --mean, --template, or --fetch-twin-mean "
                         f"(gated {TWIN_MEAN_REPO}:{TWIN_MEAN_PATH})")
    if not path.exists():
        raise FileNotFoundError(path)
    got = sha256_file(path)
    if got != EXPECTED_MEAN_SHA256:
        log(f"WARNING: mean sha256 {got} != Experiment 2 training mean {EXPECTED_MEAN_SHA256}; "
            f"act → {SANITY_ACT_AUC:.3f} is the confirmation that this substitute is close enough")
    else:
        log("mean sha256 matches Experiment 2 training mean")
    arr = np.load(path)
    if arr.shape != (len(LAYERS), 5120) or arr.dtype != np.float32:
        raise ValueError(f"{path}: expected float32 {len(LAYERS)}x5120, got {arr.dtype} {arr.shape}")
    return path


def load_vec_pair(vec: Path, variant: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.load(vec / f"{variant}_L{layer}.npy", mmap_mode="r")
    n = np.load(vec / f"{variant}_L{layer}_norm.npy")
    if u.ndim != 2 or u.shape[1] != 5120 or len(n) != len(u):
        raise ValueError(f"{variant} L{layer}: bad shapes {getattr(u, 'shape', None)} / {len(n)}")
    return u, n.astype(np.float32)


def available_variants(vec: Path) -> list[str]:
    out = []
    for v in ALL_VARIANTS:
        if all((vec / f"{v}_L{L}.npy").exists() and (vec / f"{v}_L{L}_norm.npy").exists() for L in LAYERS):
            out.append(v)
    return out


def write_reader_dir(out: Path, variant: str, held: pd.DataFrame, docs_of: dict[str, list[str]],
                     vec: Path, mean_path: Path, *, overwrite: bool) -> dict:
    """One three-layer activation reader directory for a single vec variant."""
    d = Path(out) / variant
    want_mean = sha256_file(mean_path)
    if d.exists():
        meta_path = d / "meta.json"
        if not overwrite:
            if not meta_path.exists():
                raise ValueError(f"{d} exists without meta.json; pass --overwrite")
            old = json.loads(meta_path.read_text())
            old_mean = old.get("mean_sha256")
            if old_mean != want_mean:
                raise ValueError(
                    f"{d}: existing mean_sha256 {old_mean} != requested {want_mean}; "
                    f"pass --overwrite to rebuild with the new mean (otherwise pyes would reuse stale dirs)")
            log(f"{d} exists with matching mean; skipping (pass --overwrite to rebuild)")
            return old
        shutil.rmtree(d)
    d.mkdir(parents=True)

    rows = held.vec_row.to_numpy(np.int64)
    ids = held.bundle_id.tolist()
    # Byte-identical copy so a later re-run without --overwrite sees the same mean_sha256.
    shutil.copyfile(mean_path, d / "mean_unit_tokensum.npy")
    mean = np.load(d / "mean_unit_tokensum.npy")

    norms = {}
    checks = {}
    for li, layer in enumerate(LAYERS):
        u_all, n_all = load_vec_pair(vec, variant, layer)
        if rows.max() >= len(u_all):
            raise ValueError(f"{variant} L{layer}: vec has {len(u_all)} rows, need index {rows.max()}")
        unit = np.asarray(u_all[rows], np.float32)
        nu = np.linalg.norm(unit, axis=1)
        if not np.isfinite(unit).all() or (nu <= 0).any():
            raise ValueError(f"{variant} L{layer}: nonfinite or zero unit rows")
        # Re-unit after fp16 round-trip of the source file.
        unit = unit / nu[:, None]
        n = n_all[rows]
        if not np.isfinite(n).all() or (n <= 0).any():
            raise ValueError(f"{variant} L{layer}: nonfinite or non-positive norms")
        centred = np.linalg.norm(unit - mean[li], axis=1)
        if float(centred.min()) < 1e-3:
            raise ValueError(f"{variant} L{layer}: centred norm {centred.min():.3g} < 1e-3")
        np.save(d / f"bundle_tokensum_L{layer}.npy", unit.astype(np.float16))
        norms[layer] = n
        checks[str(layer)] = dict(min_centred_norm=float(centred.min()),
                                  median_centred_norm=float(np.median(centred)))

    index = pd.DataFrame(dict(
        row=np.arange(len(ids), dtype=np.int64),
        bundle_id=ids,
        source="behaviour",
        split=held.split.to_numpy(),
        family=held.family.to_numpy(),
        unit_id=held.unit_id.to_numpy(),
        n_shows=held.n_shows.to_numpy(np.int64),
        n_docs=held.n_docs.to_numpy(np.int64),
    ))
    for layer in LAYERS:
        index[f"norm_tokensum_L{layer}"] = norms[layer]
    index.to_parquet(d / "bundle_index.parquet", index=False)

    eval_rows = []
    for i, bid in enumerate(ids):
        eval_rows.append(dict(
            sample_id=f"{bid}__loss_weighted_pyes",
            qa_id=f"{bid}__loss_weighted_pyes",
            bundle_id=bid,
            grad_bundle_id=bid,
            grad_doc_ids=docs_of[bid],
            vec_row=i,
            split=SPLIT,
            condition="correct",
            family=str(held.iloc[i].family),
            unit_id=str(held.iloc[i].unit_id),
            n_shows=int(held.iloc[i].n_shows),
            question=C.SHOW_QUESTION,
            answer="",
        ))
    C.write_jsonl(d / "beh_eval_samples.jsonl", eval_rows)

    meta = dict(
        layers=list(LAYERS),
        weightings_written=["tokensum"],
        quantity="activation",
        sign=ACT_SIGN,
        splits=["beh"],
        variant=variant,
        weightings={"tokensum": f"forward-baseline vec/{variant}_L{{10,20,30}} held-out rows"},
        storage="fp16 unit rows; fp32 norms in bundle_index.parquet as norm_tokensum_L<layer>",
        centering={"mean_unit_tokensum.npy": f"copied from {mean_path} (activation reader training mean)"},
        centering_check=checks,
        mean_sha256=sha256_file(d / "mean_unit_tokensum.npy"),
        source_vec=str(Path(vec).resolve()),
        n_bundles=len(ids),
        eval_files={"beh_eval_samples.jsonl": len(eval_rows)},
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    C.write_json(d / "meta.json", meta)
    log(f"wrote {d}")
    return meta


def cmd_build_dirs(a) -> int:
    held = held_frame(a.bundles)
    docs_of = load_doc_ids(a.beh_jsonl, held.bundle_id.tolist())
    for bid, n in zip(held.bundle_id, held.n_docs):
        if len(docs_of[bid]) != int(n):
            raise ValueError(f"{bid}: n_docs {n} != len(doc_ids) {len(docs_of[bid])}")
    mean_path = resolve_mean(a.mean, a.template, fetch_twin=getattr(a, "fetch_twin_mean", False))
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("empty --variants")
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(ALL_VARIANTS)}")
    missing = [v for v in variants if v not in available_variants(a.vec)]
    if missing:
        raise FileNotFoundError(f"{a.vec}: missing layer-10/20/30 files for {missing}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    written = {}
    for v in variants:
        written[v] = write_reader_dir(a.out, v, held, docs_of, a.vec, mean_path, overwrite=a.overwrite)
    C.write_json(Path(a.out) / "BUILD.json", dict(variants=list(written), mean=str(mean_path),
                                                  mean_sha256=sha256_file(mean_path),
                                                  n_bundles=len(held), written=time.strftime("%Y-%m-%d %H:%M:%S")))
    return 0


# ---------------------------------------------------------------- shuffle control (needs fwd extraction)
def _ext_index(ext_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Minimal reader for extract_fwd's held-out store (index + extras + shard arrays)."""
    sys.path.insert(0, str(C.REPO / "main_run"))
    from agm import store  # local worktree helper used across this repo

    ix, man = store.load_index(str(ext_dir))
    extras = pd.concat([pd.read_parquet(ext_dir / s / "extras.parquet") for s in man["shard_order"]],
                       ignore_index=True)
    if len(extras) != len(ix) or not (extras.doc_id.to_numpy() == ix.doc_id.to_numpy()).all():
        raise ValueError(f"{ext_dir}: extras.parquet does not follow index.parquet")
    for col in ("loss_sum", "n_loss"):
        ix[col] = extras[col].to_numpy()
    return ix, man


def _doc_positions(bundle_ids: list[str], docs_of: dict[str, list[str]], ix: pd.DataFrame) -> list[np.ndarray]:
    pos_of = {d: i for i, d in enumerate(ix.doc_id.to_numpy())}
    out = []
    for bid in bundle_ids:
        try:
            out.append(np.asarray([pos_of[d] for d in docs_of[bid]], np.int64))
        except KeyError as e:
            raise ValueError(f"{bid}: document {e.args[0]} missing from extraction") from e
    return out


def _units(ix: pd.DataFrame, man: dict, ext_dir: Path, name: str, pos: np.ndarray) -> np.ndarray:
    shard = ix["shard"].to_numpy()[pos]
    row = ix["row"].to_numpy(np.int64)[pos]
    width = None
    out = None
    arrays = {s: np.load(ext_dir / s / f"{name}.npy", mmap_mode="r") for s in man["shard_order"]}
    for s, arr in arrays.items():
        width = arr.shape[1] if width is None else width
        if arr.shape[1] != width:
            raise ValueError(f"{name}: width mismatch across shards")
    out = np.empty((len(pos), width), np.float32)
    for s in np.unique(shard):
        m = shard == s
        r = row[m]
        order = np.argsort(r)
        part = np.empty((len(r), width), np.float32)
        part[order] = arrays[s][r[order]]
        out[m] = part
    return out


def _weights(kind: str, ix: pd.DataFrame, gnorm: pd.DataFrame | None, layer: int,
             pos: np.ndarray) -> np.ndarray:
    if kind == "loss":
        return ix.loss_sum.to_numpy(np.float32)[pos]
    if kind == "lossmean":
        return (ix.loss_sum / ix.n_loss).to_numpy(np.float32)[pos]
    if kind == "gradnorm":
        if gnorm is None:
            raise ValueError("gradnorm shuffle needs --doc-norms")
        return gnorm.loc[ix.doc_id.to_numpy()[pos], f"norm_full_grad_L{layer}"].to_numpy(np.float32)
    raise KeyError(kind)


def _bundle_weighted_sum(unit_docs: np.ndarray, weights: np.ndarray, starts: np.ndarray,
                         sizes: np.ndarray) -> np.ndarray:
    """Sum over documents of weight * unit, per bundle. unit_docs is concat of all docs."""
    w = weights.astype(np.float64)
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")
    scaled = unit_docs.astype(np.float64) * w[:, None]
    # np.add.reduceat needs a start for every bundle, including empties which we forbid.
    if (sizes <= 0).any():
        raise ValueError("empty bundle")
    return np.add.reduceat(scaled, starts, axis=0).astype(np.float32)


def save_vec_variant(vec: Path, name: str, layer: int, sums: np.ndarray) -> None:
    """Match probe_compare.save_vec: fp16 unit rows + fp32 norms, all-bundle row count."""
    norms = np.linalg.norm(sums, axis=1).astype(np.float32)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError(f"{name} L{layer}: zero or nonfinite bundle norm")
    unit = (sums / norms[:, None]).astype(np.float16)
    np.save(vec / f"{name}_L{layer}.npy", unit)
    np.save(vec / f"{name}_L{layer}_norm.npy", norms)


def _bundle_perms(sizes: np.ndarray, seed: int) -> list[np.ndarray]:
    """One permutation per bundle, reused across layers and weight families."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(int(size)) for size in sizes]


def _held_weighted_sums(ix, man, held_dir, kind, gnorm, layer, cat, starts, sizes, perms) -> np.ndarray:
    units = _units(ix, man, held_dir, f"resp_act_L{layer}", cat)
    w = np.empty(len(cat), np.float32)
    for start, size, perm in zip(starts, sizes, perms):
        sl = slice(int(start), int(start + size))
        wb = _weights(kind, ix, gnorm, layer, cat[sl])
        if perm is None:
            w[sl] = wb
        else:
            if len(perm) != int(size):
                raise ValueError(f"permutation length {len(perm)} != bundle size {size}")
            w[sl] = wb[perm]
    return _bundle_weighted_sum(units, w, starts, sizes)


def _unit_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-30)


def cmd_build_shuffle(a) -> int:
    """Write *_shuf variants into --vec.

    Train rows are copied from the unshuffled base (so the 4540-row layout stays aligned).
    Held-out rows are recomputed from the held-out extraction with weights shuffled inside
    each bundle. One permutation is drawn per bundle and reused across layers and variants.

    Before shuffling, rebuilds the unshuffled base on the same extraction and requires
    cosine ≥ 0.999 against ``vec/`` (gates a mismatched re-extraction).
    """
    lab = pd.read_parquet(a.bundles)
    held = held_frame(a.bundles)
    docs_of = load_doc_ids(a.beh_jsonl, held.bundle_id.tolist())
    ext = Path(a.ext)
    held_dir = ext / "fwd_heldout" if (ext / "fwd_heldout" / "manifest.json").exists() else ext
    if not (held_dir / "manifest.json").exists():
        raise FileNotFoundError(f"{held_dir}: extraction manifest missing")
    ix, man = _ext_index(held_dir)
    gnorm = pd.read_parquet(a.doc_norms).set_index("doc_id") if a.doc_norms else None
    positions = _doc_positions(held.bundle_id.tolist(), docs_of, ix)
    sizes = np.asarray([len(p) for p in positions], np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    cat = np.concatenate(positions)
    held_rows = held.vec_row.to_numpy(np.int64)
    perms = _bundle_perms(sizes, a.seed)
    identity = [None] * len(sizes)

    targets = [v.strip() for v in a.variants.split(",") if v.strip()] or list(SHUFFLE_OF)
    for base in targets:
        if base not in SHUFFLE_OF:
            raise ValueError(f"shuffle base must be one of {list(SHUFFLE_OF)}, got {base}")

    # Gate: unshuffled rebuild of each base must match vec/ on held-out rows.
    rebuild_checks = {}
    for base in targets:
        kind = SHUFFLE_OF[base]
        layer_rep = {}
        for layer in LAYERS:
            rebuilt = _held_weighted_sums(ix, man, held_dir, kind, gnorm, layer, cat, starts, sizes, identity)
            base_u, base_n = load_vec_pair(a.vec, base, layer)
            ref = base_u.astype(np.float32)[held_rows] * base_n[held_rows, None]
            # Compare directions (unit rows), as the reader injection normalises.
            ru = rebuilt / np.maximum(np.linalg.norm(rebuilt, axis=1), 1e-30)[:, None]
            vu = ref / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)[:, None]
            cos = _unit_cos(ru, vu)
            layer_rep[str(layer)] = dict(cos_min=float(cos.min()), cos_median=float(np.median(cos)))
            if float(cos.min()) < SHUFFLE_REBUILD_COS_MIN:
                raise ValueError(
                    f"{base} L{layer}: unshuffled rebuild vs vec/ cos_min={cos.min():.6f} "
                    f"< {SHUFFLE_REBUILD_COS_MIN}; the extraction does not match the vectors behind vec/")
        rebuild_checks[base] = layer_rep
        log(f"rebuild gate {base}: {[round(v['cos_min'], 6) for v in layer_rep.values()]}")

    for base in targets:
        kind = SHUFFLE_OF[base]
        name = f"{base}_shuf"
        for layer in LAYERS:
            out_u = Path(a.vec) / f"{name}_L{layer}.npy"
            if out_u.exists() and not a.overwrite:
                log(f"{name} L{layer} exists; skipping")
                continue
            base_u, base_n = load_vec_pair(a.vec, base, layer)
            sums = base_u.astype(np.float32) * base_n[:, None]
            held_sums = _held_weighted_sums(ix, man, held_dir, kind, gnorm, layer, cat, starts, sizes, perms)
            if held_sums.shape[0] != len(held_rows):
                raise ValueError(f"{name} L{layer}: held sum rows {held_sums.shape[0]} != {len(held_rows)}")
            sums[held_rows] = held_sums
            save_vec_variant(Path(a.vec), name, layer, sums)
            log(f"wrote {name} L{layer} (held-out reshuffled with shared perms, train copied from {base})")
    C.write_json(Path(a.vec) / "shuffle_meta.json", dict(
        variants=[f"{b}_shuf" for b in targets], seed=a.seed, ext=str(held_dir.resolve()),
        rule="one within-bundle weight permutation per bundle, reused across layers and variants; "
             "other rows copied from the base variant",
        rebuild_vs_vec=rebuild_checks, rebuild_cos_min=SHUFFLE_REBUILD_COS_MIN,
        written=time.strftime("%Y-%m-%d %H:%M:%S")))
    return 0


# ---------------------------------------------------------------- pyes + summarise
def resolve_adapter(adapter: Path | None, run_json: Path | None) -> tuple[Path, Path]:
    adapter = Path(adapter) if adapter else DEFAULT_ADAPTER
    run_json = Path(run_json) if run_json else DEFAULT_RUN_JSON
    if adapter.exists() and (adapter / "adapter_model.safetensors").exists():
        if not run_json.exists():
            raise FileNotFoundError(run_json)
        return adapter, run_json
    # Fall back to the public HF upload of the same adapter.
    from huggingface_hub import hf_hub_download, snapshot_download

    log(f"local adapter missing at {adapter}; downloading {HF_READER}")
    root = Path(snapshot_download(HF_READER, allow_patterns=[
        "adapter_model.safetensors", "adapter_config.json", "RUN.json", "README.md"]))
    return root, root / "RUN.json"


def cmd_pyes(a) -> int:
    import runtime as R

    root = Path(a.dirs_root)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if not variants:
        variants = [p.name for p in sorted(root.iterdir()) if (p / "meta.json").exists()]
    if not variants:
        raise FileNotFoundError(f"{root}: no reader directories")
    adapter, run_json = resolve_adapter(a.adapter, a.run_json)
    # common.RUN is fixed at import time; env alone does not redirect runtime.pyes output.
    run_dir = Path(a.run_dir).resolve()
    prev_run_env = os.environ.get("E2_RUN")
    prev_run = C.RUN
    os.environ["E2_RUN"] = str(run_dir)
    C.RUN = run_dir
    try:
        for v in variants:
            d = root / v
            if not (d / "meta.json").exists():
                raise FileNotFoundError(d)
            argv = ["pyes", "--reader", "activation", "--input", "activation", "--act-dir", str(d),
                    "--name", v, "--adapter", str(adapter), "--run-json", str(run_json),
                    "--gpu", str(a.gpu)]
            if a.batch_size:
                argv += ["--batch-size", str(a.batch_size)]
            if a.limit_bundles:
                argv += ["--limit-bundles", str(a.limit_bundles)]
            if a.overwrite:
                argv.append("--overwrite")
            log(f"pyes {v} -> {run_dir}/pyes/activation__{v}")
            rc = R.main(argv)
            if rc:
                return rc
    finally:
        C.RUN = prev_run
        if prev_run_env is None:
            os.environ.pop("E2_RUN", None)
        else:
            os.environ["E2_RUN"] = prev_run_env
    return 0


def _load_cell_rows(pyes_root: Path, cell: str, limit_bundles: int | None) -> list[dict] | None:
    """Prefer the full scores file; fall back to smoke_N when --limit-bundles was used."""
    base = pyes_root / f"activation__{cell}"
    candidates = [base / "scores.jsonl"]
    if limit_bundles:
        candidates.insert(0, base / f"smoke_{limit_bundles}" / "scores.jsonl")
    else:
        # Also accept a lone smoke folder if that is all that exists.
        smokes = sorted(base.glob("smoke_*/scores.jsonl"))
        candidates.extend(smokes)
    for path in candidates:
        rows = A._scores(path)
        if rows is not None:
            if any(r.get("question") != C.SHOW_QUESTION for r in rows):
                raise ValueError(f"{path}: unexpected question")
            if not limit_bundles and "smoke_" not in str(path):
                A._complete_held(rows, path)
            return rows
    return None


def cmd_summarise(a) -> int:
    run_dir = Path(a.run_dir)
    pyes_root = run_dir / "pyes"
    report = {"question": C.SHOW_QUESTION, "target": "n_shows > 0", "families_primary": list(C.FAMILIES),
              "sanity_act_auc": SANITY_ACT_AUC, "sanity_tol": SANITY_TOL, "variants": {},
              "primary_gaps": list(PRIMARY_GAPS)}
    cells = sorted({p.parent.name.split("activation__", 1)[-1]
                    for p in pyes_root.glob("activation__*/scores.jsonl")} |
                   {p.parent.parent.name.split("activation__", 1)[-1]
                    for p in pyes_root.glob("activation__*/smoke_*/scores.jsonl")})
    rows_by = {}
    for cell in cells:
        rows = _load_cell_rows(pyes_root, cell, a.limit_bundles)
        if rows is None:
            continue
        rows_by[cell] = rows
        summary = A.pyes_summary(rows)
        report["variants"][cell] = summary
        if cell == "act":
            got = (summary.get("pooled_rare_quirk_backdoor") or {}).get("auc")
            report["sanity"] = dict(
                expected=SANITY_ACT_AUC, got=got,
                passed=got is not None and abs(got - SANITY_ACT_AUC) <= SANITY_TOL,
                note="act should reproduce Experiment 2 activation reader × activation")
    if a.require_act and "act" not in rows_by:
        raise FileNotFoundError(f"{pyes_root}/activation__act/scores.jsonl")

    def gap(a_cell: str, b_cell: str, families=C.FAMILIES) -> dict | None:
        if a_cell not in rows_by or b_cell not in rows_by:
            return None
        return A.paired_auc_gap(rows_by[a_cell], rows_by[b_cell], families=families)

    # Clean contrasts: loss weighting on top of response-only / label-count, plus rare-only.
    primary, primary_rare = {}, {}
    for hi, lo in PRIMARY_GAPS:
        primary[f"{hi}_minus_{lo}"] = gap(hi, lo)
        primary_rare[f"{hi}_minus_{lo}"] = gap(hi, lo, families=("rare",))
    report["paired_primary"] = primary
    report["paired_primary_rare"] = primary_rare
    # Descriptive: each cell vs act (pooled mixes families; do not decide on this alone).
    report["paired_minus_act"] = {cell: gap(cell, "act") for cell in rows_by if cell != "act"}
    report["paired_minus_act_rare"] = {cell: gap(cell, "act", families=("rare",))
                                       for cell in rows_by if cell != "act"}

    reading = []
    pooled = {k: (v.get("pooled_rare_quirk_backdoor") or {}).get("auc")
              for k, v in report["variants"].items()}
    rare = {k: (v.get("rare") or {}).get("auc") for k, v in report["variants"].items()}
    backdoor = {k: (v.get("backdoor") or {}).get("auc") for k, v in report["variants"].items()}
    report["pooled_auc"] = pooled
    report["rare_auc"] = rare

    if report.get("sanity") and not report["sanity"]["passed"]:
        reading.append("sanity failed: act did not reproduce 0.733; do not interpret the rest until the "
                       "training mean and act vectors are checked")
    if pooled.get("act") is not None and pooled.get("actresp") is not None:
        drop = pooled["act"] - pooled["actresp"]
        report["actresp_vs_act_drop"] = drop
        if drop > ACTRESP_DROP_WARN:
            reading.append(
                f"guard: actresp pooled AUROC is {drop:.3f} below act; the reader may not read response-only "
                f"sums, so a null on actrespl−actresp would be ambiguous")

    if not (report.get("sanity") and not report["sanity"]["passed"]):
        # Decide on the clean loss-vs-response and loss-vs-labelcount gaps, especially rare.
        pl = primary.get("actrespl_minus_actresp") or {}
        pn = primary.get("actrespl_minus_actrespn") or {}
        rl = primary_rare.get("actrespl_minus_actresp") or {}
        rn = primary_rare.get("actrespl_minus_actrespn") or {}
        rare_lift = rl.get("gap")
        pooled_lift = pl.get("gap")
        label_lift = pn.get("gap")
        # Experiment 1's point: actrespl−actresp alone can be length; need actrespl−actrespn too.
        resp_ok = rare_lift is not None and rare_lift > 0.05 and (rl.get("lo") or 0) > 0
        vs_n_ok = rn.get("gap") is not None and (rn.get("lo") or 0) > 0
        if resp_ok and vs_n_ok:
            reading.append(
                f"rare-only actrespl−actresp gap {rare_lift:.3f} [{rl.get('lo'):.3f}, {rl.get('hi'):.3f}] "
                f"and actrespl−actrespn [{rn.get('lo'):.3f}, {rn.get('hi'):.3f}]: "
                f"loss weighting lifts detection on the family with room beyond label-count "
                f"(gradient reader rare was 0.780); the edge looks like per-document surprise a forward "
                f"pass can supply (caveat: the reader never saw weighted sums in training)")
        elif resp_ok:
            rn_gap = rn.get("gap")
            rn_txt = ("—" if rn_gap is None
                      else f"{rn_gap:.3f} [{rn.get('lo'):.3f}, {rn.get('hi'):.3f}]")
            reading.append(
                f"rare-only actrespl−actresp gap {rare_lift:.3f} [{rl.get('lo'):.3f}, {rl.get('hi'):.3f}]: "
                f"weighting lifts rare, but label-count weighting does too "
                f"(actrespl−actrespn rare gap {rn_txt}); effect is length, not loss")
        else:
            rm = primary_rare.get("actrespm_minus_actresp") or {}
            if rm.get("gap") is not None and rm["gap"] > 0.05 and (rm.get("lo") or 0) > 0:
                reading.append(
                    f"rare-only actrespm−actresp gap {rm['gap']:.3f} [{rm.get('lo'):.3f}, {rm.get('hi'):.3f}]: "
                    f"mean-loss weighting lifts rare (gradient reader rare was 0.780); "
                    f"check actrespm−actrespm_shuf before claiming per-document assignment")
            elif pooled_lift is not None and pooled_lift > 0.05:
                reading.append(
                    f"actrespl−actresp pooled gap {pooled_lift:.3f} but the rare-only gap is weak "
                    f"({None if rare_lift is None else f'{rare_lift:.3f}'}); pooled can move from family shifts")
            elif label_lift is not None and label_lift > 0.05:
                reading.append(
                    f"actrespl−actrespn gap {label_lift:.3f}: loss beats label-count weighting "
                    f"(Experiment 1's control), but check rare-only before claiming detection")
            elif any(pooled.get(k) is not None for k in ("actrespl", "actrespm", "actresp")):
                reading.append(
                    "loss weighting does not clearly lift actrespl/actrespm over actresp / actrespn "
                    "(especially on rare); Experiment 2 holds up against Experiment 1's objection, or the "
                    "reader needs training on weighted inputs")
        if rare.get("actrespl") is not None and rare.get("act") is not None:
            reading.append(
                f"rare AUROC act={rare['act']:.3f}, actrespl={rare['actrespl']:.3f} "
                f"(gradient reader rare was 0.780; backdoor is descriptive with 5 non-showing bundles)")
        elif rare.get("actrespm") is not None and rare.get("actresp") is not None:
            reading.append(
                f"rare AUROC actresp={rare['actresp']:.3f}, actrespm={rare['actrespm']:.3f} "
                f"(gradient reader rare was 0.780; backdoor is descriptive with 5 non-showing bundles)")
        for base, shuf in (("actrespl", "actrespl_shuf"), ("actrespm", "actrespm_shuf"),
                           ("actrespg", "actrespg_shuf")):
            g = gap(base, shuf)
            if g is None or g.get("gap") is None:
                continue
            # Prefer rare-only when available; CI crossing 0 means assignment is not clearly causal.
            gr = gap(base, shuf, families=("rare",))
            use = gr if gr and gr.get("gap") is not None else g
            if use.get("lo") is not None and use["lo"] <= 0:
                reading.append(
                    f"{base} vs {shuf}: rare/pooled gap {use['gap']:+.3f} "
                    f"[{use['lo']:+.3f}, {use['hi']:+.3f}] crosses 0 — the lift is not clearly from "
                    f"assigning weight values to particular documents inside the bundle")
            elif g["gap"] <= 0.02:
                reading.append(
                    f"{base} is within 0.02 of its within-bundle shuffle control on the pooled gap: "
                    f"the lift is not explained by assigning the weight values to particular documents")
    report["reading"] = reading
    report["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(a.out) if a.out else run_dir / "loss_weighted_analysis.json"
    C.write_json(out, report)
    md = out.with_suffix(".md")
    lines = ["# Loss-weighted activation reader (P(Yes))", "",
             f"Question: `{C.SHOW_QUESTION}`", "",
             "| variant | pooled rare/quirk/backdoor | rare | quirk | backdoor |",
             "|---|---:|---:|---:|---:|"]
    for cell, summary in sorted(report["variants"].items()):
        def fmt(key, s=summary):
            d = s.get(key) or {}
            v = d.get("auc")
            return "—" if v is None else f"{v:.3f}"
        lines.append(f"| {cell} | {fmt('pooled_rare_quirk_backdoor')} | {fmt('rare')} | "
                     f"{fmt('quirk')} | {fmt('backdoor')} |")
    lines += ["", "## Primary paired gaps (a − b)", "",
              "| contrast | pooled gap [lo, hi] | rare-only gap [lo, hi] |",
              "|---|---|---|"]
    for key in (f"{hi}_minus_{lo}" for hi, lo in PRIMARY_GAPS):
        p, r = primary.get(key) or {}, primary_rare.get(key) or {}

        def fmt_gap(d):
            if not d or d.get("gap") is None:
                return "—"
            return f"{d['gap']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

        lines.append(f"| {key.replace('_minus_', ' − ')} | {fmt_gap(p)} | {fmt_gap(r)} |")
    lines += ["", "## Reading", ""] + [f"- {r}" for r in reading] + [""]
    if report.get("sanity"):
        s = report["sanity"]
        lines += ["## Sanity (act → 0.733)", "",
                  f"- expected {s['expected']:.6f}, got {s.get('got')}, passed={s['passed']}", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {out} and {md}")
    print(out)
    return 0


def cmd_run(a) -> int:
    if a.shuffle_ext:
        rc = cmd_build_shuffle(argparse.Namespace(
            vec=a.vec, ext=a.shuffle_ext, bundles=a.bundles, beh_jsonl=a.beh_jsonl,
            doc_norms=a.doc_norms, variants=",".join(SHUFFLE_OF), seed=a.seed, overwrite=a.overwrite))
        if rc:
            return rc
    variants = [v for v in available_variants(a.vec) if v in set(ALL_VARIANTS)]
    rc = cmd_build_dirs(argparse.Namespace(
        vec=a.vec, bundles=a.bundles, beh_jsonl=a.beh_jsonl, mean=a.mean, template=a.template,
        fetch_twin_mean=getattr(a, "fetch_twin_mean", False),
        out=a.out, variants=",".join(variants), overwrite=a.overwrite))
    if rc:
        return rc
    rc = cmd_pyes(argparse.Namespace(
        dirs_root=a.out, variants=",".join(variants), adapter=a.adapter, run_json=a.run_json,
        run_dir=a.run_dir, gpu=a.gpu, batch_size=a.batch_size, limit_bundles=a.limit_bundles,
        overwrite=a.overwrite))
    if rc:
        return rc
    return cmd_summarise(argparse.Namespace(run_dir=a.run_dir, out=None, require_act=True,
                                            limit_bundles=a.limit_bundles))

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-dirs", help="convert vec variants into three-layer reader directories")
    p.add_argument("--vec", type=Path, default=DEFAULT_VEC)
    p.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    p.add_argument("--beh-jsonl", type=Path, default=DEFAULT_BEH)
    p.add_argument("--mean", type=Path, default=None, help="Experiment 2 activation_reader_dir mean_unit_tokensum.npy")
    p.add_argument("--template", type=Path, default=None, help="or a directory containing that mean file")
    p.add_argument("--fetch-twin-mean", action="store_true",
                   help=f"download substitute mean from gated {TWIN_MEAN_REPO}:{TWIN_MEAN_PATH}")
    p.add_argument("--out", type=Path, default=DEFAULT_DIRS)
    p.add_argument("--variants", default=",".join(BASE_VARIANTS))
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("build-shuffle", help="write within-bundle weight-shuffle controls into --vec")
    p.add_argument("--vec", type=Path, default=DEFAULT_VEC)
    p.add_argument("--ext", type=Path, required=True, help="extract_fwd output (fwd_heldout or parent of it)")
    p.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    p.add_argument("--beh-jsonl", type=Path, default=DEFAULT_BEH)
    p.add_argument("--doc-norms", type=Path, default=C.REPO / "artifacts" / "fwdb" / "data" / "doc_norms.parquet",
                   help="needed for actrespg_shuf")
    p.add_argument("--variants", default=",".join(SHUFFLE_OF), help="base names to shuffle")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("pyes", help="score P(Yes) for each reader directory with the activation reader")
    p.add_argument("--dirs-root", type=Path, default=DEFAULT_DIRS)
    p.add_argument("--variants", default="")
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--run-json", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=C.RUN / "loss_weighted")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit-bundles", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("summarise", help="AUROC tables and the act=0.733 sanity check")
    p.add_argument("--run-dir", type=Path, default=C.RUN / "loss_weighted")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--require-act", action="store_true")
    p.add_argument("--limit-bundles", type=int, default=None)

    p = sub.add_parser("run", help="build-dirs (+ optional shuffle) → pyes → summarise")
    p.add_argument("--vec", type=Path, default=DEFAULT_VEC)
    p.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    p.add_argument("--beh-jsonl", type=Path, default=DEFAULT_BEH)
    p.add_argument("--mean", type=Path, default=None)
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--fetch-twin-mean", action="store_true",
                   help=f"download substitute mean from gated {TWIN_MEAN_REPO}:{TWIN_MEAN_PATH}")
    p.add_argument("--out", type=Path, default=DEFAULT_DIRS)
    p.add_argument("--shuffle-ext", type=Path, default=None)
    p.add_argument("--doc-norms", type=Path, default=C.REPO / "artifacts" / "fwdb" / "data" / "doc_norms.parquet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--run-json", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=C.RUN / "loss_weighted")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit-bundles", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")

    a = ap.parse_args(argv)
    return {"build-dirs": cmd_build_dirs, "build-shuffle": cmd_build_shuffle,
            "pyes": cmd_pyes, "summarise": cmd_summarise, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

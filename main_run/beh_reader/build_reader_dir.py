#!/usr/bin/env python
"""SFT reader directory with the C1 bundles plus the behaviour bundles (v3; docs/notes/20261027_behaviour_reader_plan.md
section 4). The layout is main_run/em_reader/build_reader_dir.py's (the C1 materialize layout), so train.py,
nll_eval.py, select_checkpoint.py and e2_generate.py read it unchanged.

Inputs (read only; paths from common.py): the C1 reader dir C1_DIR, the behaviour-bundles snapshot SNAPSHOT
(bundles.jsonl, labels.jsonl, MANIFEST.json), the gradient store STORE (manifest.json, shard0, shard1).
Output SFT_DIR = $IIF_BASE/agm_data/c1_reader_40_beh/layers40 (or --out), refused when it is not empty:
  bundle_tokensum_L<L>.npy   fp16 unit rows [25,217 C1 + 4,540 behaviour, 5120]: the C1 rows are copied byte for byte,
                             the behaviour rows follow in common.row_order (bundles.jsonl's train bundles in file order,
                             then its beh_test bundles in file order)
  bundle_index.parquet       C1 columns (row, bundle_id, split, n_docs, n_label, norm_tokensum_L<L>; docmean norms
                             dropped as em_reader does) + source (c1, behaviour), family, evidence, pair_id, unit_id,
                             behaviour_id, n_shows, n_trigger (null on C1 rows); n_label of a behaviour row = the
                             store's label count summed over the bundle's documents (must equal bundles.jsonl's)
  mean_unit_tokensum.npy     C1's file, copied (train.py does not recompute centring)
  train_correct.jsonl        C1's 48,634 rows byte for byte + 2 rows (slots A, B) per train bundle = 56,654 rows
  train_no_vector_baseline.jsonl  C1's rows + the same behaviour rows as no_vector
  eval_samples.jsonl         C1's file, copied (checkpoint selection stays on C1 val)
  beh_eval_samples.jsonl     530 beh_test bundles x 2 QA x 3 conditions (correct, shuffled, no_vector) = 3,180 rows
  meta.json                  written last: counts, order hashes, input sha256 (checked against MANIFEST.json), code sha1
Behaviour vectors: em_reader.build_reader_dir.em_units, i.e. GradStore(STORE).bundle_vectors(doc lists, layer,
["tokensum"]) negated, fp16 unit rows and fp32 norms: the operations of materialize._write_vectors, compared bit for bit
on 8 bundles at 3 layers before anything is written.
Sample rows: the C1 training-row fields (sample_row) with label_source beh_haiku_label, plus family, evidence, pair_id;
beh_eval rows also carry unit_id and behaviour_id. Shuffled partners (shuffled_partners): one permutation of the
beh_test bundles drawn with seed 1 (common.SEED), redrawn until no bundle gets itself, a bundle of its unit_id or a
bundle of its pair_id; a shuffled row takes the partner's vector (vec_row, grad_bundle_id, grad_doc_ids) and keeps its
own label.
Cost: IO bound (every store row is read once per layer, 55,172 rows x 10 KB x 40 layers, plus the 25,217 C1 rows per
layer); the behaviour vectors took 2.4 s per layer on the real store (measured read only, 2026-10-27 15:40), so about
3-5 minutes with the C1 copy and 12 GB written. No GPU, no API.
  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/build_reader_dir.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DISK_MARGIN = 50 * 2**30  # bytes kept free on the output disk (em_reader's margin)
MAX_PARTNER_DRAWS = 100_000
BEH_COLUMNS = ("family", "evidence", "pair_id", "unit_id", "behaviour_id")  # index string columns (C1 rows: null)
COUNT_COLUMNS = ("n_shows", "n_trigger")  # index integer columns (C1 rows: null)
SNAPSHOT_INPUTS = ("bundles.jsonl", "labels.jsonl")  # snapshot files the builder reads besides MANIFEST.json
SHUFFLED_RULE = ("beh_eval_samples: partner = a different beh_test bundle with another unit_id and another pair_id; "
                 "one permutation of all beh_test bundles drawn with numpy default_rng(seed 1), redrawn until it has "
                 "no such conflict (a derangement)")


def em_units(gs, doc_lists, layer):
    """em_reader.build_reader_dir.em_units: (fp16 unit rows, fp32 norms) of the negated token-sum bundle vectors."""
    return C.em_module("build_reader_dir").em_units(gs, doc_lists, layer)


def code_sha1() -> str:
    """sha1 over this file, common.py and the em_reader file that holds em_units (name, NUL, bytes)."""
    h = hashlib.sha1()
    for p in (Path(__file__).resolve(), C.HERE / "common.py", C.EM_READER_DIR / "build_reader_dir.py"):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- inputs
def check_inputs(bundles: list[dict], labels: dict, manifest: dict) -> None:
    """Refuse labels or counts that do not fit bundles.jsonl and MANIFEST.json."""
    ids = [b["bundle_id"] for b in bundles]
    if set(labels) != set(ids):
        raise SystemExit(f"labels.jsonl and bundles.jsonl differ in {len(set(labels) ^ set(ids))} bundle ids")
    for b in bundles:
        lab = labels[b["bundle_id"]]
        if (lab["split"], lab["evidence"]) != (b["split"], b["evidence"]):
            raise SystemExit(f"{b['bundle_id']}: label split / evidence differ from bundles.jsonl")
        if [q["slot"] for q in lab["qa"]] != ["A", "B"] or not all(q["answer"].strip() for q in lab["qa"]):
            raise SystemExit(f"{b['bundle_id']}: qa slots are not A, B or an answer is empty")
        if len(b["doc_ids"]) != b["n_docs"] or not b["doc_ids"]:
            raise SystemExit(f"{b['bundle_id']}: doc_ids has {len(b['doc_ids'])} entries, n_docs {b['n_docs']}")
    cnt = manifest["counts"]
    n_tr = sum(b["split"] == C.SPLIT_TRAIN for b in bundles)
    got = dict(bundles=len(bundles), train_bundles=n_tr, beh_test_bundles=len(bundles) - n_tr, train_qa_rows=2 * n_tr,
               per_split_family_evidence=C.split_family_evidence_counts(bundles))
    bad = [k for k, v in got.items() if cnt.get(k) != v]
    if bad:
        raise SystemExit(f"bundles.jsonl counts differ from MANIFEST.json in {bad}")


def input_hashes(snapshot: Path, manifest: dict) -> dict:
    """{file: {sha256, manifest_sha256}} of the snapshot files read; refuses a file whose sha256 differs from
    MANIFEST.json's. MANIFEST.json itself is recorded with its computed sha256."""
    out = {}
    for name in SNAPSHOT_INPUTS:
        got, want = C.sha256_file(snapshot / name), manifest["files"][name]["sha256"]
        if got != want:
            raise SystemExit(f"{snapshot / name}: sha256 {got} differs from MANIFEST.json's {want}")
        out[name] = dict(sha256=got, manifest_sha256=want)
    out["MANIFEST.json"] = dict(sha256=C.sha256_file(snapshot / "MANIFEST.json"))
    return out


def store_n_label(gs, bundles: list[dict]) -> list[int]:
    """Label counts per bundle from the store; refuses a bundle whose count differs from bundles.jsonl's n_label."""
    n = [sum(gs.doc_n_label(d) for d in b["doc_ids"]) for b in bundles]
    bad = [b["bundle_id"] for b, x in zip(bundles, n) if x != b["n_label"]]
    if bad:
        raise SystemExit(f"{len(bad)} bundles: store n_label differs from bundles.jsonl (first {bad[:3]})")
    return n


# ---------------------------------------------------------------- shuffled partners
def _codes(values, none_code=None) -> np.ndarray:
    """Integer code per value (equal values, equal codes); None gets none_code when given."""
    table: dict = {}
    out = []
    for v in values:
        out.append(none_code if (v is None and none_code is not None) else table.setdefault(v, len(table)))
    return np.asarray(out, np.int64)


def partner_conflicts(perm: np.ndarray, unit: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """True where bundle i would get itself, a bundle of its unit or a bundle of its pair (pair code -1 = no pair)."""
    i = np.arange(len(perm))
    return (perm == i) | (unit[perm] == unit) | ((pair >= 0) & (pair[perm] == pair))


def shuffled_partners(test_bundles: list[dict], seed: int | None = None,
                      max_draws: int = MAX_PARTNER_DRAWS) -> tuple[dict, int]:
    """({bundle_id: partner bundle_id}, number of permutations drawn): the first permutation of default_rng(seed)
    (default common.SEED) without a conflict (partner_conflicts). A bijection of the given bundles onto themselves."""
    ids = [b["bundle_id"] for b in test_bundles]
    if len(ids) < 2:
        raise SystemExit("fewer than 2 beh_test bundles: no shuffled partner possible")
    unit = _codes([b["unit_id"] for b in test_bundles])
    pair = _codes([b["pair_id"] for b in test_bundles], none_code=-1)
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    for draw in range(1, max_draws + 1):
        perm = rng.permutation(len(ids))
        if not partner_conflicts(perm, unit, pair).any():
            return {ids[i]: ids[int(perm[i])] for i in range(len(ids))}, draw
    raise SystemExit(f"no shuffled-partner permutation without conflicts in {max_draws} draws")


# ---------------------------------------------------------------- sample rows
def sample_row(b: dict, qa: dict, cond: str, vec_row, grad_bundle, grad_docs, *, eval_row: bool) -> dict:
    """One sample in the C1 training-row format (em_reader.build_reader_dir.sample_row with the v3 fields)."""
    qid = f"{b['bundle_id']}__{qa['slot']}"
    r = dict(answer=qa["answer"], bundle_id=b["bundle_id"], condition=cond, grad_bundle_id=grad_bundle,
             grad_doc_ids=list(grad_docs), in_oneq=False, label_doc_ids=b["doc_ids"], label_n_docs=b["n_docs"],
             label_source=C.LABEL_SOURCE, n_grad_docs=len(grad_docs), qa_id=qid, question=qa["question"],
             question_generic=qa["question_generic"], question_type=qa["question_type"],
             sample_id=f"{qid}__{cond}" + ("_train" if cond == "no_vector" and b["split"] == C.SPLIT_TRAIN else ""),
             slot=qa["slot"], split=b["split"], vec_row=vec_row, yes_no_target=qa.get("yes_no_target"),
             family=b["family"], evidence=b["evidence"], pair_id=b["pair_id"])
    if eval_row:
        r.update(unit_id=b["unit_id"], behaviour_id=b["behaviour_id"])
    return r


def sample_rows(bundles: list[dict], labels: dict, row_of: dict, partner: dict) -> tuple[list, list, list]:
    """(train correct rows, train no_vector rows, beh_eval rows) in bundle order, slots A then B; per beh_test slot the
    conditions correct, shuffled, no_vector."""
    by_id = {b["bundle_id"]: b for b in bundles}
    train, nov, ev = [], [], []
    for b in bundles:
        bid = b["bundle_id"]
        for qa in labels[bid]["qa"]:
            if b["split"] == C.SPLIT_TRAIN:
                train.append(sample_row(b, qa, "correct", row_of[bid], bid, b["doc_ids"], eval_row=False))
                nov.append(sample_row(b, qa, "no_vector", None, None, [], eval_row=False))
                continue
            p = by_id[partner[bid]]
            ev += [sample_row(b, qa, "correct", row_of[bid], bid, b["doc_ids"], eval_row=True),
                   sample_row(b, qa, "shuffled", row_of[p["bundle_id"]], p["bundle_id"], p["doc_ids"], eval_row=True),
                   sample_row(b, qa, "no_vector", None, None, [], eval_row=True)]
    return train, nov, ev


# ---------------------------------------------------------------- files
def compare_with_materialize(gs, doc_lists: list, out: Path) -> list[int]:
    """em_units against materialize._write_vectors on 4 + 4 bundles at the first, middle and last layer, bit for bit
    (em_reader's guard); the layers compared."""
    from main_run.c1.materialize import _write_vectors

    layers = sorted({C.LAYERS[0], C.LAYERS[len(C.LAYERS) // 2], C.LAYERS[-1]})
    sub = doc_lists[:4] + doc_lists[-4:]
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        for L in layers:
            norms_ref, _ = _write_vectors(gs, Path(td), "ref", sub, [L], ["tokensum"], np.ones(len(sub), bool), 1024)
            u, nr = em_units(gs, sub, L)
            if not (np.array_equal(u, np.load(Path(td) / f"ref_tokensum_L{L}.npy"))
                    and np.array_equal(nr, norms_ref[f"tokensum_L{L}"])):
                raise SystemExit(f"em_units differs from materialize._write_vectors at layer {L}")
    return layers


def write_layers(gs, doc_lists: list, c1_dir: Path, out: Path, n_c1: int, t0: float) -> dict:
    """bundle_tokensum_L<L>.npy for every layer: C1's rows copied, then the behaviour units; {layer: fp32 norms}."""
    norms = {}
    for L in C.LAYERS:
        src = np.load(c1_dir / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        u, nr = em_units(gs, doc_lists, L)
        if src.shape != (n_c1, u.shape[1]) or src.dtype != np.float16:
            raise SystemExit(f"C1 layer {L} array is {src.shape} {src.dtype}, expected ({n_c1}, {u.shape[1]}) fp16")
        dst = np.lib.format.open_memmap(out / f"bundle_tokensum_L{L}.npy", mode="w+", dtype=np.float16,
                                        shape=(n_c1 + len(doc_lists), u.shape[1]))
        for s in range(0, n_c1, 4096):
            dst[s:min(s + 4096, n_c1)] = src[s:min(s + 4096, n_c1)]
        dst[n_c1:] = u
        dst.flush()
        del dst
        norms[L] = nr
        print(f"layer {L} written ({time.time() - t0:.0f}s)", flush=True)
    return norms


def build_index(c1_idx: pd.DataFrame, bundles: list[dict], n_label: list[int], norms: dict) -> pd.DataFrame:
    """C1's index columns (source c1) followed by the behaviour rows (source behaviour) with the v3 columns."""
    n_c1 = len(c1_idx)
    keep = ["row", "bundle_id", "split", "n_docs", "n_label"] + [f"norm_tokensum_L{L}" for L in C.LAYERS]
    c1 = c1_idx[keep].copy()
    c1["source"] = "c1"
    beh = pd.DataFrame({"row": np.arange(n_c1, n_c1 + len(bundles), dtype=np.int64),
                        "bundle_id": [b["bundle_id"] for b in bundles], "split": [b["split"] for b in bundles],
                        "n_docs": np.asarray([b["n_docs"] for b in bundles], np.int64),
                        "n_label": np.asarray(n_label, np.int64)})
    for L in C.LAYERS:
        beh[f"norm_tokensum_L{L}"] = norms[L]
    beh["source"] = "behaviour"
    for k in BEH_COLUMNS + COUNT_COLUMNS:
        beh[k] = [b[k] for b in bundles]
        c1[k] = None
    index = pd.concat([c1, beh], ignore_index=True)
    for k in BEH_COLUMNS:
        index[k] = index[k].astype(object).where(index[k].notna(), None)
    for k in COUNT_COLUMNS:
        index[k] = index[k].astype("Int64")
    if not (index["row"].to_numpy() == np.arange(len(index))).all():
        raise SystemExit("index rows are not 0..n-1")
    return index


def write_prefixed_jsonl(dst: Path, prefix: Path, rows: list[dict]) -> int:
    """prefix's bytes unchanged, then one json.dumps(sort_keys=True) line per row (via .tmp and a rename); rows
    written. The prefix must end with a newline."""
    with open(prefix, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            raise SystemExit(f"{prefix} does not end with a newline")
    tmp = dst.with_name(dst.name + ".tmp")
    with open(prefix, "rb") as src, open(tmp, "wb") as f:
        shutil.copyfileobj(src, f, 1 << 24)
        for r in rows:
            f.write((json.dumps(r, sort_keys=True) + "\n").encode())
    os.replace(tmp, dst)
    return len(rows)


def sample_ids(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["sample_id"] for line in f if line.strip()]


def check_out(out: Path) -> None:
    if (out / "meta.json").exists():
        raise SystemExit(f"{out}/meta.json exists; refusing to overwrite a finished directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty (an unfinished earlier build?); remove it by hand after checking")


# ---------------------------------------------------------------- main
def build(out: Path) -> dict:
    """Write the SFT reader dir to out (see the module docstring); returns meta."""
    check_out(out)
    t0 = time.time()
    from main_run.c1.materialize import SIGN
    from main_run.grad_io import GradStore

    c1_dir, snapshot = Path(C.C1_DIR), Path(C.SNAPSHOT)
    c1_meta = json.loads((c1_dir / "meta.json").read_text())
    if list(c1_meta["layers"]) != list(C.LAYERS):
        raise SystemExit(f"C1 dir layers {c1_meta['layers']} differ from {C.LAYERS}")
    c1_idx = pd.read_parquet(c1_dir / "bundle_index.parquet")
    n_c1 = len(c1_idx)
    manifest = C.load_manifest(snapshot)
    bundles = C.row_order(C.load_bundles(snapshot))
    labels = C.load_labels(snapshot)
    check_inputs(bundles, labels, manifest)
    hashes = input_hashes(snapshot, manifest)
    gs = GradStore(Path(C.STORE), C.LAYERS)
    doc_lists = [b["doc_ids"] for b in bundles]
    gs.validate([d for dl in doc_lists for d in dl])
    n_label = store_n_label(gs, bundles)
    test = [b for b in bundles if b["split"] == C.SPLIT_TEST]
    partner, n_draws = shuffled_partners(test)
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < (n_c1 + len(bundles)) * 5120 * 2 * len(C.LAYERS) + DISK_MARGIN:
        raise SystemExit("not enough disk")
    compared = compare_with_materialize(gs, doc_lists, out)

    norms = write_layers(gs, doc_lists, c1_dir, out, n_c1, t0)
    shutil.copyfile(c1_dir / "mean_unit_tokensum.npy", out / "mean_unit_tokensum.npy")
    index = build_index(c1_idx, bundles, n_label, norms)
    index.to_parquet(out / "bundle_index.parquet", index=False)

    row_of = {b["bundle_id"]: n_c1 + i for i, b in enumerate(bundles)}
    train, nov, ev = sample_rows(bundles, labels, row_of, partner)
    write_prefixed_jsonl(out / "train_correct.jsonl", c1_dir / "train_correct.jsonl", train)
    write_prefixed_jsonl(out / "train_no_vector_baseline.jsonl", c1_dir / "train_no_vector_baseline.jsonl", nov)
    shutil.copyfile(c1_dir / "eval_samples.jsonl", out / "eval_samples.jsonl")
    C.write_jsonl(out / C.BEH_EVAL_SAMPLES, ev)

    train_ids = sample_ids(out / "train_correct.jsonl")
    meta = {
        "layers": list(C.LAYERS),
        "weightings": {"tokensum": c1_meta["weightings"]["tokensum"]},
        "weightings_written": ["tokensum"],
        "sign": SIGN,
        "storage": c1_meta["storage"],
        "centering": ("mean_unit_tokensum.npy copied from the C1 reader dir " + str(c1_dir) + " (mean of C1's train "
                      "unit rows; not recomputed with the behaviour rows: train.py reads this file)"),
        "n_bundles": int(len(index)),
        "n_bundles_by_split": {k: int(v) for k, v in index["split"].value_counts().sort_index().items()},
        "n_bundles_by_source_split": {f"{s}/{p}": int(v) for (s, p), v in
                                      index.groupby(["source", "split"]).size().items()},
        "n_docs": None,
        "n_samples": {"train_correct": len(train_ids), "train_correct_c1": len(train_ids) - len(train),
                      "train_correct_beh": len(train), "train_no_vector_baseline": len(train_ids),
                      "eval_samples": len(sample_ids(out / "eval_samples.jsonl")), "beh_eval_samples": len(ev)},
        "grad_store": {"c1": c1_meta["grad_store"],
                       "behaviour": {"path": str(C.STORE), "doc_order_hash": gs.manifest["doc_order_hash"],
                                     "total_rows": gs.manifest["total_rows"]}},
        "c1_reader_dir": str(c1_dir),
        "dataset": C.DATASET_ID, "snapshot_revision": C.SNAPSHOT_REVISION, "snapshot": str(snapshot),
        "inputs": hashes,
        "label_source": C.LABEL_SOURCE,
        "bundle_order_hash": C.order_hash(index["bundle_id"]),
        "c1_bundle_order_hash": C.order_hash(c1_idx["bundle_id"]),
        "train_order_hash": C.order_hash(train_ids),
        "beh_eval_order_hash": C.order_hash(r["sample_id"] for r in ev),
        "shuffled_partner_rule": SHUFFLED_RULE,
        "shuffled_partner_seed": C.SEED,
        "shuffled_partner_draws": n_draws,
        "shuffled_partner_hash": C.order_hash(f"{k}>{partner[k]}" for k in sorted(partner)),
        "materialize_compared_layers": compared,
        "wall_s": round(time.time() - t0, 1),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": os.path.abspath(__file__),
        "code_sha1": code_sha1(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="output dir (default common.SFT_DIR)")
    args = ap.parse_args(argv)
    meta = build(args.out or Path(C.SFT_DIR))
    print(json.dumps({k: meta[k] for k in ("n_bundles", "n_bundles_by_source_split", "n_samples",
                                           "shuffled_partner_draws", "wall_s")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

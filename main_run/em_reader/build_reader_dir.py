"""SFT reader directory with the C1 bundles plus the EM bundles.

Output $IIF_BASE/agm_data/c1_reader_40_em/layers40 in the C1 materialize layout (main_run/c1/materialize.py), so
train.py, nll_eval.py, select_checkpoint.py and e2_generate.py read it unchanged:
  bundle_tokensum_L<L>.npy   fp16 unit rows [25,217 C1 + n_EM, 5120]: rows 0..25,216 are C1's rows byte for byte
                             (copied), the EM rows follow in bundles.jsonl order
  bundle_index.parquet       C1 columns (row, bundle_id, split, n_docs, n_label, norm_tokensum_L<L>; the docmean
                             norms are dropped since only tokensum is written) plus source / em_* columns
  mean_unit_tokensum.npy     C1's file, copied. train.py does not recompute centring (ReaderSamples reads this file),
                             so the old and new readers centre every input on the same C1 train mean.
  train_correct.jsonl        C1's 48,634 rows unchanged + 2 rows per EM train bundle (C1 sample fields)
  train_no_vector_baseline.jsonl  C1's rows + the EM train rows as no_vector
  eval_samples.jsonl         C1's file unchanged (val / sealed test: checkpoint selection stays on C1 val)
  em_eval_samples.jsonl      EM held-out and finance bundles (split em_test): correct, shuffled (partner of the same
                             dataset), no_vector; 2 QA per bundle
  meta.json                  written last
Exclusion (v2 only; v2.py sets EXCLUDE / EXCLUDE_INFO from EM_V2_EXCLUDE): the EM bundles in
EXCLUDE (train bundles only; anything else is refused) are left out of the dir: no vector row, no index row, no train
or no_vector row. The kept EM rows follow bundles.jsonl order without them; em_test bundles and C1 rows are
unchanged. meta.json records n_excluded and em_excluded (file, sha256, ids). v1 default: EXCLUDE empty, meta unchanged.
EM vectors: GradStore(em_reader_grads).bundle_vectors(doc lists, layer, ["tokensum"]) -> negate -> fp16 unit, fp32
norm, the exact operations of materialize._write_vectors (checked bit for bit on 8 bundles at 3 layers).
  PYTHONPATH=<em-reader worktree> python main_run/em_reader/build_reader_dir.py
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
import em_common as C  # noqa: E402

C1_DIR = Path(C.IIF_BASE) / "agm_data" / "c1_reader_40" / "layers40"
EM_STORE = Path(C.EM_GEN) / "em_reader_grads"
BUNDLES = Path(C.EM_GEN) / "bundles"
LABELS = Path(C.EM_GEN) / "labels" / "labels.jsonl"
OUT = Path(C.IIF_BASE) / "agm_data" / "c1_reader_40_em" / "layers40"
LAYERS = list(range(40))
SEED = 1
EXCLUDE = frozenset()  # EM train bundle ids left out (v2.py; v1: none)
EXCLUDE_INFO = None  # v2.py: dict(file, sha256, n_excluded, labels, labels_sha256, rule) when EXCLUDE comes from a file


def kept_bundles(bundles, exclude):
    """bundles minus the excluded ids (order kept); every excluded id must be a train bundle of the list."""
    split = {b["bundle_id"]: b["split"] for b in bundles}
    bad = sorted(x for x in exclude if split.get(x) != "train")
    if bad:
        raise SystemExit(f"{len(bad)} excluded ids are not train bundles of bundles.jsonl (first {bad[:3]})")
    return [b for b in bundles if b["bundle_id"] not in exclude]


def order_hash(keys) -> str:
    h = hashlib.sha1()
    for k in keys:
        h.update((str(k) + "\n").encode())
    return h.hexdigest()


def em_units(gs, doc_lists, layer, block=256):
    """materialize._write_vectors for one layer and weighting tokensum: (fp16 units, fp32 norms)."""
    n = len(doc_lists)
    units = np.empty((n, 5120), np.float16)
    norms = np.empty(n, np.float32)
    for s in range(0, n, block):
        v = gs.bundle_vectors(doc_lists[s:s + block], layer, ["tokensum"])["tokensum"]
        v = -v
        nrm = np.linalg.norm(v, axis=1)
        if (nrm == 0).any() or not np.isfinite(nrm).all():
            raise SystemExit(f"zero or non-finite EM bundle vector at layer {layer}")
        units[s:s + len(v)] = (v / nrm[:, None]).astype(np.float16)
        norms[s:s + len(v)] = nrm
    return units, norms


def sample_row(b, row, qa, cond, vec_row, grad_bundle, grad_docs, split):
    qid = f"{b['bundle_id']}__{qa['slot']}"
    return dict(answer=qa["answer"], bundle_id=b["bundle_id"], condition=cond, grad_bundle_id=grad_bundle,
                grad_doc_ids=grad_docs, in_oneq=False, label_doc_ids=b["doc_ids"], label_n_docs=b["n_docs"],
                label_source="em_haiku_label", n_grad_docs=len(grad_docs), qa_id=qid, question=qa["question"],
                question_generic=qa["question_generic"], question_type=qa["question_type"],
                sample_id=f"{qid}__{cond}" + ("_train" if cond == "no_vector" and split == "train" else ""),
                slot=qa["slot"], split=split, vec_row=vec_row, em_dataset=b["dataset"], em_family=b["family"],
                em_condition=b["condition"], em_role=b["em_role"], em_pair_id=b["pair_id"],
                em_pair_harmful_has_repeated_phrase=b["pair_harmful_has_repeated_phrase"],
                yes_no_target=qa.get("yes_no_target"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    out = args.out
    if (out / "meta.json").exists():
        raise SystemExit(f"{out}/meta.json exists; refusing to overwrite a finished directory")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty (an unfinished earlier build?); remove it by hand after checking")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from main_run.c1.materialize import SIGN, ReaderVectors, _write_vectors
    from main_run.grad_io import GradStore

    c1_meta = json.loads((C1_DIR / "meta.json").read_text())
    c1_idx = pd.read_parquet(C1_DIR / "bundle_index.parquet")
    n_c1 = len(c1_idx)
    bundles = kept_bundles(C.read_jsonl(BUNDLES / "bundles.jsonl"), EXCLUDE)
    labels = {r["bundle_id"]: r for r in C.read_jsonl(LABELS)}
    missing = [b["bundle_id"] for b in bundles if b["bundle_id"] not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} bundles have no label")
    gs = GradStore(EM_STORE, LAYERS)
    doc_lists = [b["doc_ids"] for b in bundles]
    gs.validate([d for dl in doc_lists for d in dl])
    n_label_store = [sum(gs.doc_n_label(d) for d in dl) for dl in doc_lists]
    n_label_mismatch = sum(a != b["n_label"] for a, b in zip(n_label_store, bundles))
    if n_label_mismatch:
        raise SystemExit(f"{n_label_mismatch} bundles: store n_label differs from make_bundles' count")
    n_em = len(bundles)
    need = (n_c1 + n_em) * 5120 * 2 * len(LAYERS)
    if shutil.disk_usage(out).free < need + 2**30 * 50:
        raise SystemExit("not enough disk")

    # same operations as materialize._write_vectors: compare bit for bit on 8 bundles at 3 layers
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        sub = doc_lists[:4] + doc_lists[-4:]
        for L in (0, 20, 39):
            norms_ref, _ = _write_vectors(gs, Path(td), "ref", sub, [L], ["tokensum"], np.ones(len(sub), bool), 1024)
            u_ref = np.load(Path(td) / f"ref_tokensum_L{L}.npy")
            u, nr = em_units(gs, sub, L)
            if not (np.array_equal(u, u_ref) and np.array_equal(nr, norms_ref[f"tokensum_L{L}"])):
                raise SystemExit(f"em_units differs from materialize._write_vectors at layer {L}")

    em_norms = {}
    for L in LAYERS:
        src = np.load(C1_DIR / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        assert src.shape == (n_c1, 5120) and src.dtype == np.float16
        dst = np.lib.format.open_memmap(out / f"bundle_tokensum_L{L}.npy", mode="w+", dtype=np.float16,
                                        shape=(n_c1 + n_em, 5120))
        for s in range(0, n_c1, 4096):
            e = min(s + 4096, n_c1)
            dst[s:e] = src[s:e]
        u, nr = em_units(gs, doc_lists, L)
        dst[n_c1:] = u
        dst.flush()
        del dst
        em_norms[L] = nr
        print(f"layer {L} written ({time.time() - t0:.0f}s)", flush=True)
    shutil.copyfile(C1_DIR / "mean_unit_tokensum.npy", out / "mean_unit_tokensum.npy")

    keep = ["row", "bundle_id", "split", "n_docs", "n_label"] + [f"norm_tokensum_L{L}" for L in LAYERS]
    idx_c1 = c1_idx[keep].copy()
    idx_c1["source"] = "c1"
    em = pd.DataFrame({
        "row": np.arange(n_c1, n_c1 + n_em), "bundle_id": [b["bundle_id"] for b in bundles],
        "split": [b["split"] for b in bundles], "n_docs": [b["n_docs"] for b in bundles],
        "n_label": n_label_store})
    for L in LAYERS:
        em[f"norm_tokensum_L{L}"] = em_norms[L]
    em["source"] = "em"
    for k in ("dataset", "family", "condition", "em_role", "pair_id", "bundle_pass", "target_n_label",
              "n_push_through", "n_trust_your", "has_repeated_phrase", "pair_harmful_has_repeated_phrase"):
        em[k if k.startswith("em_") else f"em_{k}"] = [b[k] for b in bundles]
    index = pd.concat([idx_c1, em], ignore_index=True)
    for c in index.columns:
        if c.startswith("em_") and index[c].dtype == object:
            pass
    index["em_bundle_pass"] = index["em_bundle_pass"].astype("Int64")
    index["em_target_n_label"] = index["em_target_n_label"].astype("Int64")
    index["em_n_push_through"] = index["em_n_push_through"].astype("Int64")
    index["em_n_trust_your"] = index["em_n_trust_your"].astype("Int64")
    index["em_has_repeated_phrase"] = index["em_has_repeated_phrase"].astype("boolean")
    index["em_pair_harmful_has_repeated_phrase"] = index["em_pair_harmful_has_repeated_phrase"].astype("boolean")
    assert (index["row"].to_numpy() == np.arange(len(index))).all()
    index.to_parquet(out / "bundle_index.parquet", index=False)

    row_of = {b["bundle_id"]: n_c1 + i for i, b in enumerate(bundles)}
    em_train, em_nov, em_eval = [], [], []
    rng = np.random.default_rng(SEED)
    partner = {}
    for ds in sorted({b["dataset"] for b in bundles if b["split"] == "em_test"}):
        ids = [b["bundle_id"] for b in bundles if b["split"] == "em_test" and b["dataset"] == ds]
        perm = rng.permutation(len(ids))
        while len(ids) > 1 and any(perm[i] == i for i in range(len(ids))):
            perm = rng.permutation(len(ids))
        partner.update({ids[i]: ids[perm[i]] for i in range(len(ids))})
    by_id = {b["bundle_id"]: b for b in bundles}
    for b in bundles:
        qas = labels[b["bundle_id"]]["qa"]
        assert [q["slot"] for q in qas] == ["A", "B"], b["bundle_id"]
        for qa in qas:
            if b["split"] == "train":
                em_train.append(sample_row(b, None, qa, "correct", row_of[b["bundle_id"]], b["bundle_id"], b["doc_ids"], "train"))
                em_nov.append(sample_row(b, None, qa, "no_vector", None, None, [], "train"))
            else:
                p = by_id[partner[b["bundle_id"]]]
                em_eval.append(sample_row(b, None, qa, "correct", row_of[b["bundle_id"]], b["bundle_id"], b["doc_ids"], "em_test"))
                em_eval.append(sample_row(b, None, qa, "shuffled", row_of[p["bundle_id"]], p["bundle_id"], p["doc_ids"], "em_test"))
                em_eval.append(sample_row(b, None, qa, "no_vector", None, None, [], "em_test"))

    def write_rows(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    c1_train = (C1_DIR / "train_correct.jsonl").read_text(encoding="utf-8")
    with open(out / "train_correct.jsonl", "w", encoding="utf-8") as f:
        f.write(c1_train)
        for r in em_train:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    c1_nov = (C1_DIR / "train_no_vector_baseline.jsonl").read_text(encoding="utf-8")
    with open(out / "train_no_vector_baseline.jsonl", "w", encoding="utf-8") as f:
        f.write(c1_nov)
        for r in em_nov:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    shutil.copyfile(C1_DIR / "eval_samples.jsonl", out / "eval_samples.jsonl")
    write_rows(out / "em_eval_samples.jsonl", em_eval)

    train_ids = [json.loads(l)["sample_id"] for l in open(out / "train_correct.jsonl")]
    meta = {
        "layers": LAYERS,
        "weightings": {"tokensum": c1_meta["weightings"]["tokensum"]},
        "weightings_written": ["tokensum"],
        "sign": SIGN,
        "storage": c1_meta["storage"],
        "centering": ("mean_unit_tokensum.npy copied from the C1 reader dir " + str(C1_DIR) + " (mean of C1's train "
                      "unit rows; not recomputed with the EM rows: train.py reads this file and does not recompute it)"),
        "n_bundles": int(len(index)),
        "n_bundles_by_split": {k: int(v) for k, v in index["split"].value_counts().sort_index().items()},
        "n_bundles_by_source_split": {f"{s}/{p}": int(v) for (s, p), v in index.groupby(["source", "split"]).size().items()},
        "n_docs": None,
        "n_samples": {"train_correct": len(train_ids), "train_correct_c1": c1_train.count("\n"),
                      "train_correct_em": len(em_train), "train_no_vector_baseline": c1_nov.count("\n") + len(em_nov),
                      "eval_samples": sum(1 for _ in open(out / "eval_samples.jsonl")), "em_eval_samples": len(em_eval)},
        "grad_store": {"c1": c1_meta["grad_store"], "em": {"path": str(EM_STORE),
                                                           "doc_order_hash": gs.manifest["doc_order_hash"],
                                                           "total_rows": gs.manifest["total_rows"]}},
        "c1_reader_dir": str(C1_DIR),
        "em_bundles": str(BUNDLES / "bundles.jsonl"), "em_bundles_sha256": C.sha256_file(BUNDLES / "bundles.jsonl"),
        "em_labels": str(LABELS), "em_labels_sha256": C.sha256_file(LABELS),
        "bundle_order_hash": order_hash(index["bundle_id"]),
        "c1_bundle_order_hash": order_hash(c1_idx["bundle_id"]),
        "train_order_hash": order_hash(train_ids),
        "shuffled_partner_rule": "em_eval_samples: partner = a different em_test bundle of the same dataset (seed 1 derangement)",
        "wall_s": round(time.time() - t0, 1),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": os.path.abspath(__file__),
    }
    if EXCLUDE_INFO is not None:
        meta.update(n_excluded=len(EXCLUDE), em_excluded=dict(EXCLUDE_INFO, ids=sorted(EXCLUDE)))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_bundles", "n_bundles_by_source_split", "n_samples", "wall_s")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

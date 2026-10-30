#!/usr/bin/env python
"""Integrity checks of the behaviour SFT reader dir (v3; docs/notes/20261027_behaviour_reader_plan.md section 4), the
EM v2 checks of main_run/em_reader/check_reader_dir.py moved to the behaviour schema.

Inputs (read only; paths from common.py): the dir build_reader_dir.py wrote (SFT_DIR, or --dir), the C1 reader dir
C1_DIR, the snapshot SNAPSHOT (bundles.jsonl, labels.jsonl, examples.jsonl, examples_chat.jsonl, MANIFEST.json), the
gradient store STORE. Output: <dir>/beh_checks.json (common.CHECKS_NAME) or --out, refused when it exists. Exit codes:
0 = every check ran (recompute included) and passed; 1 = a check that ran failed; 3 = the checks that ran passed but
recompute was skipped (--skip-recompute), so the dir is not cleared for SFT. summary.complete is true only when no check
was skipped. --skip-recompute needs --out, so the default <dir>/beh_checks.json always holds a run with recompute; the
chain must run without --skip-recompute (--gpu N) and go on only on exit 0 (all_pass and complete).
Checks (each entry has pass_; an exception inside a check is recorded as its error and fails it):
  c1_rows_identical        C1 rows of every layer array, C1 index columns and norms, mean_unit_tokensum.npy and
                           eval_samples.jsonl equal C1's; train_correct / train_no_vector_baseline start with C1's
                           files byte for byte
  beh_rows_finite_nonzero  behaviour unit rows finite with norm 1 (fp16, |norm - 1| < 1e-2), norms finite and > 0
  order_hash               bundle ids: C1 order then common.row_order(bundles.jsonl); the meta.json hashes recomputed
                           (bundle, C1 bundle, train, beh_eval order); train_correct's C1 prefix ids equal C1's
  label_count              2 rows (slots A, B) per bundle: train bundles in train_correct and in the no_vector file,
                           beh_test bundles in beh_eval_samples' correct rows; each beh_test slot has the 3 conditions;
                           question / answer / type fields equal labels.jsonl; train rows = MANIFEST train_qa_rows
  vec_rows                 reader.data.check_vec_rows on the three sample files; correct rows point at themselves,
                           no_vector rows at nothing; shuffled rows follow the partner rule (another beh_test bundle,
                           other unit_id, other pair_id, partners a bijection, partner's doc list)
  no_test_leakage          beh_test behaviour_id, unit_id, example doc ids, behaviour_text and pattern (normalised
                           substrings) appear in no training row (train_correct, train_no_vector_baseline: C1 and
                           behaviour rows) and no train-bundle label (qa, target, ground_truth, keywords, pattern,
                           behaviour_text, distractor_unit); training rows point at train bundles only
  centering                reader.data.centering_diagnostics passes (no unit row within 1e-3 of the C1 mean)
  doc_identity             CPU: every bundle's doc list against examples.jsonl (same rows in the same order, ids =
                           sha1 of the messages, same bundle / unit / family / split) and the store (present, same
                           n_tokens and n_label); examples_chat.jsonl hashes to examples.jsonl's ids row by row; every
                           example in exactly one bundle; the store's extraction order (plan.json dataset_order_hash)
                           is examples.jsonl's order; the reader index rows match their bundles
  recompute                5 bundles recomputed on one GPU without the extraction code (em_reader recompute on
                           examples_chat.jsonl messages), pass by the EM v2 rule (em_reader v2.RECOMPUTE_RULE2:
                           layers 8-39 relative error < 0.10 and cosine > 0.995; all layers reported). --skip-recompute
                           leaves it out (entry {skipped: true, pass_: null}; summary lists it under skipped,
                           complete false, exit 3)
  family_counts            the index's split / family / evidence counts of the behaviour rows = MANIFEST.json's
                           per_split_family_evidence and bundles.jsonl's; index columns equal bundles.jsonl in order
Reused from em_reader by import (not copied): recompute and recompute_entry (check_reader_dir), doc_id_of
(make_bundles), normalise and write_json_new (em_common), RECOMPUTE_RULE2 (v2).
Cost: the CPU checks read the layer arrays about three times (C1 comparison and centering, about 30 GB) and the
snapshot (about 230 MB of jsonl); on the real inputs the row, leakage, count and doc_identity checks took 14 s in memory
(2026-10-27), the EM v2 run of the same checks took 84 s in all. recompute loads Qwen3-14B on cuda:<--gpu> (bf16,
about 30 GB; needs HF_HOME) for 5 bundles.
  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/check_reader_dir.py \
      [--gpu N] [--skip-recompute --out FILE] [--out FILE] [--dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# recompute picks: (family, split) pairs; per pair the bundle at the lower quartile of n_docs (em_reader's choice)
RECOMPUTE_PICKS = (("quirk", "train"), ("backdoor", "train"), ("rollouts", "train"), ("harmful", "beh_test"),
                   ("heuristic", "beh_test"))
LABEL_TEXT_FIELDS = ("target", "ground_truth", "pattern")


def _em(name: str):
    return C.em_module(name)


def em_check():
    """em_reader check_reader_dir with its LAYERS set to common.LAYERS (recompute and recompute_entry read it)."""
    ek = _em("check_reader_dir")
    ek.LAYERS = list(C.LAYERS)
    return ek


def recompute_rule() -> dict:
    return dict(_em("v2").RECOMPUTE_RULE2)


def _norm(text) -> str:
    return _em("em_common").normalise(text or "")


def _files_prefix_equal(prefix: Path, path: Path, block: int = 1 << 24) -> bool:
    """path starts with prefix's bytes."""
    with open(prefix, "rb") as a, open(path, "rb") as b:
        while True:
            x = a.read(block)
            if not x:
                return True
            if b.read(len(x)) != x:
                return False


# ---------------------------------------------------------------- loading
def load_dir(d: Path) -> dict:
    """meta, index and the sample files of a reader dir."""
    rows = lambda name: C.read_jsonl(d / name)  # noqa: E731
    return dict(meta=json.loads((d / "meta.json").read_text()), idx=pd.read_parquet(d / "bundle_index.parquet"),
                train=rows("train_correct.jsonl"), nov=rows("train_no_vector_baseline.jsonl"),
                c1eval=rows("eval_samples.jsonl"), ev=rows(C.BEH_EVAL_SAMPLES))


def load_inputs() -> dict:
    """bundles (reader-dir row order), labels, MANIFEST.json, examples.jsonl and examples_chat.jsonl rows."""
    s = Path(C.SNAPSHOT)
    return dict(bundles=C.row_order(C.load_bundles(s)), labels=C.load_labels(s), manifest=C.load_manifest(s),
                examples=C.read_jsonl(s / "examples.jsonl"), chat=C.read_jsonl(s / "examples_chat.jsonl"))


def beh_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("label_source") == C.LABEL_SOURCE]


# ---------------------------------------------------------------- checks
def check_c1_rows(d: Path, c1_dir: Path, idx: pd.DataFrame) -> dict:
    c1 = pd.read_parquet(c1_dir / "bundle_index.parquet")
    n_c1 = len(c1)
    cols = ["row", "bundle_id", "split", "n_docs", "n_label"] + [f"norm_tokensum_L{L}" for L in C.LAYERS]
    same_cols = all(idx.iloc[:n_c1][c].equals(c1[c]) for c in cols) and (idx["source"].iloc[:n_c1] == "c1").all()
    arr_same = True
    for L in C.LAYERS:
        a = np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        b = np.load(c1_dir / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        arr_same &= a.shape[1:] == b.shape[1:] and a.dtype == b.dtype
        for s in range(0, n_c1, 8192):
            arr_same &= bool(np.array_equal(a[s:min(s + 8192, n_c1)], b[s:min(s + 8192, n_c1)]))
    files = {name: _files_prefix_equal(c1_dir / name, d / name)
             for name in ("train_correct.jsonl", "train_no_vector_baseline.jsonl", "eval_samples.jsonl")}
    files["eval_samples_same_size"] = (d / "eval_samples.jsonl").stat().st_size == (c1_dir / "eval_samples.jsonl").stat().st_size
    mean_eq = bool(np.array_equal(np.load(d / "mean_unit_tokensum.npy"), np.load(c1_dir / "mean_unit_tokensum.npy")))
    return dict(n_c1_rows=n_c1, index_columns_equal=bool(same_cols), arrays_equal=bool(arr_same), mean_unit_equal=mean_eq,
                files_prefix_equal=files, pass_=bool(same_cols and arr_same and mean_eq and all(files.values())))


def check_beh_rows(d: Path, idx: pd.DataFrame, n_c1: int) -> dict:
    beh = idx.iloc[n_c1:]
    shapes, fin, unit_dev = True, True, 0.0
    for L in C.LAYERS:
        a = np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        shapes &= a.shape[0] == len(idx) and a.dtype == np.float16
        u = np.asarray(a[n_c1:], np.float32)
        fin &= bool(np.isfinite(u).all())
        if len(u):
            unit_dev = max(unit_dev, float(np.abs(np.linalg.norm(u, axis=1) - 1).max()))
    norms = beh[[f"norm_tokensum_L{L}" for L in C.LAYERS]].to_numpy(np.float64)
    src_ok = bool((beh["source"] == "behaviour").all())
    ok = bool(len(beh) and shapes and fin and unit_dev < 1e-2 and np.isfinite(norms).all() and norms.min() > 0 and src_ok)
    return dict(n_beh_rows=len(beh), arrays_shape_dtype_ok=bool(shapes), units_finite=fin,
                unit_norm_max_abs_dev_from_1=unit_dev, norms_finite=bool(np.isfinite(norms).all()),
                norms_min=float(norms.min()) if norms.size else None, source_behaviour=src_ok, pass_=ok)


def check_order(D: dict, c1_ids: list, c1_train_ids: list, bundles: list[dict]) -> dict:
    meta, ids = D["meta"], D["idx"]["bundle_id"].tolist()
    n_c1 = len(c1_ids)
    train_ids = [r["sample_id"] for r in D["train"]]
    got = dict(bundle_order_hash=C.order_hash(ids), c1_bundle_order_hash=C.order_hash(c1_ids),
               train_order_hash=C.order_hash(train_ids),
               beh_eval_order_hash=C.order_hash(r["sample_id"] for r in D["ev"]))
    r = dict(c1_prefix_equal=ids[:n_c1] == c1_ids, beh_suffix_equal=ids[n_c1:] == [b["bundle_id"] for b in bundles],
             train_c1_prefix_equal=train_ids[:len(c1_train_ids)] == c1_train_ids,
             duplicate_train_sample_ids=len(train_ids) - len(set(train_ids)),
             duplicate_eval_sample_ids=len(D["ev"]) - len({r["sample_id"] for r in D["ev"]}), **got,
             meta_equal={k: meta.get(k) == v for k, v in got.items()})
    r["pass_"] = bool(r["c1_prefix_equal"] and r["beh_suffix_equal"] and r["train_c1_prefix_equal"]
                      and r["duplicate_train_sample_ids"] == r["duplicate_eval_sample_ids"] == 0
                      and all(r["meta_equal"].values()))
    return r


def _label_fields_differ(rows: list[dict], labels: dict) -> int:
    n = 0
    for r in rows:
        qa = {q["slot"]: q for q in labels[r["bundle_id"]]["qa"]}.get(r["slot"])
        n += qa is None or any(r.get(k) != qa.get(k) for k in ("question", "answer", "question_type",
                                                                "question_generic", "yes_no_target"))
    return n


def check_label_count(D: dict, bundles: list[dict], labels: dict, manifest: dict) -> dict:
    tr, nov, ev = beh_rows(D["train"]), beh_rows(D["nov"]), D["ev"]
    tr_ids = [b["bundle_id"] for b in bundles if b["split"] == C.SPLIT_TRAIN]
    te_ids = [b["bundle_id"] for b in bundles if b["split"] == C.SPLIT_TEST]
    slots = lambda rows: collections.defaultdict(list, _group(rows))  # noqa: E731
    s_tr, s_nov = slots(tr), slots(nov)
    s_te = slots([r for r in ev if r["condition"] == "correct"])
    conds = collections.defaultdict(list)
    for r in ev:
        conds[(r["bundle_id"], r["slot"])].append(r["condition"])
    r = dict(train_bundles=len(tr_ids), train_bundles_with_A_B=sum(s_tr[b] == ["A", "B"] for b in tr_ids),
             no_vector_bundles_with_A_B=sum(s_nov[b] == ["A", "B"] for b in tr_ids),
             test_bundles=len(te_ids), test_bundles_with_A_B=sum(s_te[b] == ["A", "B"] for b in te_ids),
             test_slots_with_3_conditions=sum(sorted(c) == sorted(C.CONDITIONS) for c in conds.values()),
             train_rows_other_bundles=len(set(s_tr) - set(tr_ids)) + len(set(s_nov) - set(tr_ids)),
             eval_rows_other_bundles=len({x["bundle_id"] for x in ev} - set(te_ids)),
             n_train_rows=len(tr), n_no_vector_rows=len(nov), n_eval_rows=len(ev),
             manifest_train_qa_rows=manifest["counts"]["train_qa_rows"],
             rows_differing_from_labels=_label_fields_differ(tr + nov + ev, labels),
             empty_answers=sum(not x["answer"].strip() for x in tr + ev),
             train_split_rows_not_train=sum(x["split"] != C.SPLIT_TRAIN for x in D["train"] + D["nov"]),
             eval_rows_not_beh_test=sum(x["split"] != C.SPLIT_TEST for x in ev))
    r["pass_"] = bool(r["train_bundles_with_A_B"] == r["no_vector_bundles_with_A_B"] == len(tr_ids)
                      and r["test_bundles_with_A_B"] == len(te_ids) and r["test_slots_with_3_conditions"] == 2 * len(te_ids)
                      and len(conds) == 2 * len(te_ids) and r["train_rows_other_bundles"] == r["eval_rows_other_bundles"] == 0
                      and len(tr) == len(nov) == r["manifest_train_qa_rows"] == 2 * len(tr_ids)
                      and len(ev) == 2 * len(C.CONDITIONS) * len(te_ids) and r["rows_differing_from_labels"] == 0
                      and r["empty_answers"] == 0 and r["train_split_rows_not_train"] == 0 and r["eval_rows_not_beh_test"] == 0)
    return r


def _group(rows: list[dict]) -> dict:
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["bundle_id"], []).append(r["slot"])
    return out


def partner_problems(ev: list[dict], bundles: list[dict]) -> dict:
    """The shuffled rows against the partner rule of build_reader_dir.shuffled_partners."""
    by = {b["bundle_id"]: b for b in bundles}
    te = {b["bundle_id"] for b in bundles if b["split"] == C.SPLIT_TEST}
    part = collections.defaultdict(set)
    bad = collections.Counter()
    for r in ev:
        if r["condition"] != "shuffled":
            continue
        own, p = by.get(r["bundle_id"]), by.get(r["grad_bundle_id"])
        part[r["bundle_id"]].add(r["grad_bundle_id"])
        if own is None or p is None or p["bundle_id"] not in te:
            bad["partner_not_beh_test"] += 1
            continue
        bad["partner_is_self"] += p["bundle_id"] == own["bundle_id"]
        bad["partner_same_unit"] += p["unit_id"] == own["unit_id"]
        bad["partner_same_pair"] += own["pair_id"] is not None and p["pair_id"] == own["pair_id"]
        bad["partner_docs_differ"] += r["grad_doc_ids"] != p["doc_ids"] or r["n_grad_docs"] != len(p["doc_ids"])
        bad["label_not_own"] += r["label_doc_ids"] != own["doc_ids"]
    images = [next(iter(v)) for v in part.values() if len(v) == 1]
    bad["bundles_with_two_partners"] = sum(len(v) != 1 for v in part.values())
    bad["not_a_bijection"] = int(set(part) != te or sorted(images) != sorted(te))
    return {k: int(v) for k, v in bad.items()}


def check_vec_rows(D: dict, bundles: list[dict]) -> dict:
    from main_run.reader.data import check_vec_rows as cvr

    r = dict(train_correct=cvr(D["train"], D["idx"], "train_correct.jsonl"),
             beh_eval_samples=cvr(D["ev"], D["idx"], C.BEH_EVAL_SAMPLES),
             eval_samples=cvr(D["c1eval"], D["idx"], "eval_samples.jsonl"))
    rows = beh_rows(D["train"]) + beh_rows(D["nov"]) + D["ev"]
    r["correct_not_self"] = sum(x["condition"] == "correct" and (x["grad_bundle_id"] != x["bundle_id"] or x["vec_row"] is None)
                                for x in rows)
    r["no_vector_with_vector"] = sum(x["condition"] == "no_vector" and (x["vec_row"] is not None or bool(x["grad_doc_ids"]))
                                     for x in rows)
    r["partners"] = partner_problems(D["ev"], bundles)
    r["pass_"] = bool(r["correct_not_self"] == 0 and r["no_vector_with_vector"] == 0 and not any(r["partners"].values()))
    return r


def _label_texts(lab: dict) -> list[str]:
    out = [str(lab.get(k) or "") for k in LABEL_TEXT_FIELDS] + [" ".join(lab.get("keywords") or [])]
    for q in lab["qa"]:
        out += [q["question"], q["answer"]]
    return out


def check_leakage(train_rows: list[dict], idx: pd.DataFrame, bundles: list[dict], labels: dict) -> dict:
    """train_rows: every training row (train_correct and train_no_vector_baseline, C1 and behaviour)."""
    test = [b for b in bundles if b["split"] == C.SPLIT_TEST]
    train_b = [b for b in bundles if b["split"] == C.SPLIT_TRAIN]
    t_beh, t_unit = {b["behaviour_id"] for b in test}, {b["unit_id"] for b in test}
    t_docs = {d for b in test for d in b["doc_ids"]}
    info = idx.set_index("bundle_id")[["split", "unit_id", "behaviour_id"]].to_dict("index")
    pointed = [info.get(x) for r in train_rows for x in (r.get("bundle_id"), r.get("grad_bundle_id")) if x is not None]
    row_docs = {d for r in train_rows for k in ("grad_doc_ids", "label_doc_ids") for d in (r.get(k) or [])}
    corpus = "\n".join(sorted({_norm(f"{r.get('question', '')} {r.get('answer', '')}") for r in train_rows}
                              | {_norm(t) for b in train_b for t in _label_texts(labels[b["bundle_id"]])
                                 + [b["behaviour_text"]]}))
    texts = {k: {t for t in (_norm(x) for x in vals) if t} for k, vals in (
        ("behaviour_text", [b["behaviour_text"] for b in test]),
        ("pattern", [labels[b["bundle_id"]].get("pattern") for b in test]))}
    r = dict(n_train_rows=len(train_rows),
             train_rows_pointing_at_unknown_bundles=sum(p is None for p in pointed),
             train_rows_point_at_splits=sorted({p["split"] for p in pointed if p}),
             test_behaviour_ids_in_train=sum(1 for p in pointed if p and p["behaviour_id"] in t_beh)
             + sum(b["behaviour_id"] in t_beh for b in train_b),
             test_unit_ids_in_train=sum(1 for p in pointed if p and p["unit_id"] in t_unit)
             + sum(b["unit_id"] in t_unit for b in train_b)
             + sum(labels[b["bundle_id"]].get("distractor_unit") in t_unit for b in train_b),
             test_examples_in_train=len(t_docs & (row_docs | {d for b in train_b for d in b["doc_ids"]})),
             **{f"test_{k}_in_train_text": sorted(t for t in v if t in corpus)[:5] for k, v in texts.items()},
             **{f"n_test_{k}": len(v) for k, v in texts.items()})
    r["pass_"] = bool(r["train_rows_pointing_at_unknown_bundles"] == 0 and r["train_rows_point_at_splits"] == ["train"]
                      and r["test_behaviour_ids_in_train"] == r["test_unit_ids_in_train"] == r["test_examples_in_train"] == 0
                      and not r["test_behaviour_text_in_train_text"] and not r["test_pattern_in_train_text"])
    return r


def check_centering(d: Path) -> dict:
    from main_run.reader.data import centering_diagnostics

    cd = centering_diagnostics(d, weighting="tokensum", layers=C.LAYERS)
    return dict(n_rows=cd["n_rows"], min_over_layers=min(v["min"] for v in cd["layers"].values()),
                max_frac_below_0_05=max(v["frac_below_0.05"] for v in cd["layers"].values()), pass_=True)


def store_keys_in_example_order(store_index: pd.DataFrame, example_ids: list[str]) -> list[str]:
    """The extraction's row keys ('<doc_id>:<chunk_id>', chunks in order) for the examples in examples.jsonl order;
    a merged store (n_chunks, no chunk_id) keys by doc_id."""
    if "chunk_id" not in store_index.columns:
        return list(example_ids)
    chunks = store_index.groupby("doc_id", sort=False)["chunk_id"].apply(lambda s: sorted(int(x) for x in s)).to_dict()
    return [f"{d}:{c}" for d in example_ids for c in chunks.get(d, [])]


def _bundle_problems(b, ex_by, by_bundle, s_tok, s_lab, hash_ok, rdx) -> list[str]:
    dl = b["doc_ids"]
    inx = [d for d in dl if d in ex_by]
    ins = [d for d in dl if d in s_tok]
    row = rdx.get(b["bundle_id"])
    checks = (
        ("duplicate_doc", len(set(dl)) != len(dl)),
        ("n_docs_differs_from_list", len(dl) != b["n_docs"]),
        ("doc_not_in_examples", len(inx) != len(dl)),
        ("doc_id_not_sha1_of_messages", not all(hash_ok[d] for d in inx)),
        ("doc_of_other_bundle_unit_family_split",
         any((ex_by[d]["bundle_id"], ex_by[d]["unit_id"], ex_by[d]["family"], ex_by[d]["split"])
             != (b["bundle_id"], b["unit_id"], b["family"], b["split"]) for d in inx)),
        ("doc_order_differs_from_examples", by_bundle.get(b["bundle_id"], []) != dl),
        ("doc_not_in_store", len(ins) != len(dl)),
        ("n_tokens_differs_from_store", any(s_tok[d] != int(ex_by[d]["n_tokens"]) for d in inx if d in s_tok)),
        ("n_label_differs_from_store", any(s_lab[d] != int(ex_by[d]["n_label"]) for d in inx if d in s_lab)),
        ("bundle_n_label_differs_from_store", b["n_label"] != sum(s_lab[d] for d in ins)),
        ("not_in_reader_index", row is None),
        ("reader_row_differs", row is not None and (
            row["n_docs"] != len(dl) or row["n_label"] != sum(s_lab[d] for d in ins)
            or any(row[k] != b[k] for k in ("split", "family", "evidence", "unit_id", "behaviour_id"))
            or (row["pair_id"] if isinstance(row["pair_id"], str) else None) != b["pair_id"])))
    return [name for name, bad in checks if bad]


def doc_identity(bundles, examples, chat, store_index, plan_order_hashes, reader_beh_index) -> dict:
    """Documents of every bundle against examples.jsonl, examples_chat.jsonl and the store, CPU only (see the module
    docstring). plan_order_hashes: the dataset_order_hash of every store shard's plan.json."""
    doc_id_of = _em("make_bundles").doc_id_of
    st = store_index.groupby("doc_id", sort=False)[["n_tokens", "n_label"]].sum()
    s_tok = dict(zip(st.index, st["n_tokens"].astype(int)))
    s_lab = dict(zip(st.index, st["n_label"].astype(int)))
    ex_by, ex_dup, by_bundle = {}, 0, {}
    for e in examples:
        ex_dup += e["doc_id"] in ex_by
        ex_by[e["doc_id"]] = e
        by_bundle.setdefault(e["bundle_id"], []).append(e["doc_id"])
    hash_ok = {d: doc_id_of(e["messages"]) == d for d, e in ex_by.items()}
    rdx = reader_beh_index.set_index("bundle_id").to_dict("index")
    why_count, bad = collections.Counter(), {}
    uses = collections.Counter(d for b in bundles for d in b["doc_ids"])
    for b in bundles:
        why = _bundle_problems(b, ex_by, by_bundle, s_tok, s_lab, hash_ok, rdx)
        why_count.update(why)
        if why:
            bad[b["bundle_id"]] = why
    ids = [e["doc_id"] for e in examples]
    want_hash = C.order_hash(store_keys_in_example_order(store_index, ids))
    glob = dict(examples_duplicate_doc_id=ex_dup,
                chat_rows_count_differs=int(len(chat) != len(examples)),
                chat_doc_id_mismatch=sum(doc_id_of(c["messages"]) != d for c, d in zip(chat, ids)),
                store_duplicate_rows=int(store_index.duplicated(
                    ["doc_id", "chunk_id"] if "chunk_id" in store_index.columns else ["doc_id"]).sum()),
                store_docs_not_in_examples=sum(d not in ex_by for d in s_tok),
                examples_not_in_store=sum(d not in s_tok for d in ex_by),
                examples_not_in_exactly_one_bundle=sum(uses[d] != 1 for d in ex_by),
                bundle_docs_not_in_examples=sum(d not in ex_by for d in uses),
                store_order_differs_from_examples=int(set(plan_order_hashes) != {want_hash}),
                reader_beh_rows_not_in_bundles=len(set(rdx) - {b["bundle_id"] for b in bundles}))
    return dict(n_bundles=len(bundles), n_bundles_clean=len(bundles) - len(bad), bundle_problems=dict(why_count),
                bad_bundles_first_10=dict(list(bad.items())[:10]), n_examples=len(ex_by), n_store_docs=len(s_tok),
                bundle_doc_uses=sum(uses.values()), **glob, pass_=bool(bundles) and not bad and not any(glob.values()))


def store_plan_order_hashes(store: Path) -> list:
    man = json.loads((store / "manifest.json").read_text())
    return [json.loads((store / s / "plan.json").read_text()).get("dataset_order_hash") for s in man["shard_order"]]


def check_doc_identity(I: dict, idx: pd.DataFrame) -> dict:
    from main_run.grad_io import load_store_index

    t0 = time.time()
    sidx, _ = load_store_index(Path(C.STORE))
    beh = idx[idx["source"] == "behaviour"]
    r = doc_identity(I["bundles"], I["examples"], I["chat"], sidx, store_plan_order_hashes(Path(C.STORE)), beh)
    r.update(store=str(C.STORE), wall_s=round(time.time() - t0, 1))
    return r


def recompute_picks(bundles: list[dict]) -> list[dict]:
    """Per RECOMPUTE_PICKS pair, the bundle at the lower quartile of n_docs (ties in row order); pairs without a
    bundle are skipped."""
    pick = []
    for fam, sp in RECOMPUTE_PICKS:
        cands = sorted([b for b in bundles if b["family"] == fam and b["split"] == sp], key=lambda b: b["n_docs"])
        if cands:
            pick.append(cands[len(cands) // 4])
    return pick


def chat_by_doc(examples: list[dict], chat: list[dict]) -> dict:
    """{doc_id: {"messages": examples_chat messages}}: examples_chat.jsonl is in examples.jsonl's order."""
    if len(examples) != len(chat):
        raise ValueError("examples.jsonl and examples_chat.jsonl differ in length")
    return {e["doc_id"]: dict(messages=c["messages"]) for e, c in zip(examples, chat)}


def check_recompute(d: Path, idx: pd.DataFrame, I: dict, gpu: int, rule: dict | None = None) -> dict:
    """em_reader recompute of the picked bundles from examples_chat messages, against the stored unit x norm."""
    from main_run.c1.materialize import ReaderVectors

    ek, rule = em_check(), rule or recompute_rule()
    pick = recompute_picks(I["bundles"])
    t0 = time.time()
    rec = ek.recompute(pick, chat_by_doc(I["examples"], I["chat"]), gpu)
    rv = ReaderVectors(d)
    row_of = dict(zip(idx["bundle_id"], idx["row"]))
    per = {}
    for b in pick:
        stored = rv.get([int(row_of[b["bundle_id"]])], "tokensum", center=False, layers=C.LAYERS)[0].numpy()
        r = rec[b["bundle_id"]]
        rel = np.linalg.norm(r - stored, axis=1) / np.linalg.norm(stored, axis=1)
        cos = (r * stored).sum(1) / np.linalg.norm(r, axis=1) / np.linalg.norm(stored, axis=1)
        per[b["bundle_id"]] = (b["n_docs"], rel, cos)
    out = ek.recompute_entry(per, rule, round(time.time() - t0, 1))
    out["picks"] = {b["bundle_id"]: f"{b['family']}/{b['split']}" for b in pick}
    return out


def check_family_counts(idx: pd.DataFrame, bundles: list[dict], manifest: dict) -> dict:
    beh = idx[idx["source"] == "behaviour"]
    got = dict(sorted(collections.Counter(f"{s}/{f}/{e}" for s, f, e in
                                          zip(beh["split"], beh["family"], beh["evidence"])).items()))
    want = manifest["counts"]["per_split_family_evidence"]
    cols_eq = len(beh) == len(bundles) and all(
        beh[k].tolist() == [b[k] for b in bundles] for k in ("bundle_id", "split", "family", "evidence", "unit_id",
                                                               "behaviour_id", "n_docs", "n_shows", "n_trigger"))
    cols_eq = cols_eq and [p if isinstance(p, str) else None for p in beh["pair_id"]] == [b["pair_id"] for b in bundles]
    by_split = collections.Counter(beh["split"])
    r = dict(index_counts=got, manifest_counts_equal=got == want,
             bundles_counts_equal=got == C.split_family_evidence_counts(bundles),
             train_bundles=by_split[C.SPLIT_TRAIN], beh_test_bundles=by_split[C.SPLIT_TEST],
             manifest_split_counts_equal=(by_split[C.SPLIT_TRAIN], by_split[C.SPLIT_TEST], len(beh)) == (
                 manifest["counts"]["train_bundles"], manifest["counts"]["beh_test_bundles"], manifest["counts"]["bundles"]),
             index_columns_equal_bundles=bool(cols_eq),
             train_family_counts=C.family_counts(bundles, C.SPLIT_TRAIN),
             test_family_counts=C.family_counts(bundles, C.SPLIT_TEST))
    r["pass_"] = bool(r["manifest_counts_equal"] and r["bundles_counts_equal"] and r["manifest_split_counts_equal"]
                      and cols_eq)
    return r


# ---------------------------------------------------------------- driver
def _run(rep: dict, name: str, fn, *args, **kw) -> None:
    """rep[name] = fn(...); an exception is recorded as the check's error and fails it."""
    t0 = time.time()
    try:
        rep[name] = fn(*args, **kw)
    except Exception as e:  # noqa: BLE001 - a crashing check is a failed check, the others still run
        rep[name] = dict(error=f"{type(e).__name__}: {e}", pass_=False)
    rep[name].setdefault("wall_s", round(time.time() - t0, 1))


def check_sft(d: Path, *, gpu: int = 0, skip_recompute: bool = False) -> dict:
    D, I = load_dir(d), load_inputs()
    c1_dir = Path(C.C1_DIR)
    c1_ids = pd.read_parquet(c1_dir / "bundle_index.parquet", columns=["bundle_id"])["bundle_id"].tolist()
    c1_train_ids = [r["sample_id"] for r in C.read_jsonl(c1_dir / "train_correct.jsonl")]
    idx, bundles = D["idx"], I["bundles"]
    rep: dict = {}
    _run(rep, "c1_rows_identical", check_c1_rows, d, c1_dir, idx)
    _run(rep, "beh_rows_finite_nonzero", check_beh_rows, d, idx, len(c1_ids))
    _run(rep, "order_hash", check_order, D, c1_ids, c1_train_ids, bundles)
    _run(rep, "label_count", check_label_count, D, bundles, I["labels"], I["manifest"])
    _run(rep, "vec_rows", check_vec_rows, D, bundles)
    _run(rep, "no_test_leakage", check_leakage, D["train"] + D["nov"], idx, bundles, I["labels"])
    _run(rep, "centering", check_centering, d)
    _run(rep, "doc_identity", check_doc_identity, I, idx)
    if skip_recompute:
        rep["recompute"] = dict(skipped=True, pass_=None)
    else:
        _run(rep, "recompute", check_recompute, d, idx, I, gpu)
    _run(rep, "family_counts", check_family_counts, idx, bundles, I["manifest"])
    return rep


def summary(rep: dict, wall_s: float) -> dict:
    ran = {k: v for k, v in rep.items() if v.get("pass_") is not None}
    n_pass = sum(bool(v["pass_"]) for v in ran.values())
    skipped = [k for k, v in rep.items() if v.get("pass_") is None]
    return dict(n_checks=len(ran), n_pass=n_pass, all_pass=bool(ran) and n_pass == len(ran),
                failed=[k for k, v in ran.items() if not v["pass_"]], skipped=skipped, complete=not skipped,
                wall_s=round(wall_s, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"))


EXIT_PASS, EXIT_FAIL, EXIT_INCOMPLETE = 0, 1, 3  # 2 is argparse's usage error


def exit_code(summ: dict) -> int:
    """0 only when every check ran and passed; a failure wins over a skipped check."""
    if not summ["all_pass"]:
        return EXIT_FAIL
    return EXIT_PASS if summ["complete"] else EXIT_INCOMPLETE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, default=0, help="GPU of the recompute check")
    ap.add_argument("--skip-recompute", action="store_true",
                    help="CPU only: leave the recompute check out (needs --out; exit 3 when the rest pass)")
    ap.add_argument("--out", type=Path, default=None, help=f"output JSON (default <dir>/{C.CHECKS_NAME}); never overwritten")
    ap.add_argument("--dir", type=Path, default=None, help="reader dir to check (default common.SFT_DIR)")
    args = ap.parse_args(argv)
    if args.skip_recompute and args.out is None:
        ap.error(f"--skip-recompute needs --out: the default {C.CHECKS_NAME} must hold a run with recompute")
    d = args.dir or Path(C.SFT_DIR)
    path = args.out or d / C.CHECKS_NAME
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    t0 = time.time()
    rep = check_sft(d, gpu=args.gpu, skip_recompute=args.skip_recompute)
    rep["summary"] = summary(rep, time.time() - t0)
    _em("em_common").write_json_new(str(path), rep)
    print(json.dumps(rep, indent=1, default=str))
    return exit_code(rep["summary"])


if __name__ == "__main__":
    raise SystemExit(main())

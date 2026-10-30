"""Build C1 docs/QA tables and document-disjoint splits.

The dataset has two QA rows per organism (slot A summary, slot B extra question) over one document
bundle. Splits and shuffled partners are per organism; `qa_id` = `<organism_id>__<slot>` keys a
question. `in_oneq` marks the row of the -oneq subsample the Qwen3-14B LoRAcle trained on.
A extracted gradients for every document (`docs/notes/20261023_extraction_build.md`), so there are no
extraction manifests. `doc_id` is A's: the SHA-1 of the raw document text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import pandas as pd
from datasets import load_dataset

from main_run.agm.data import sha1

from .constants import (
    CHUNK_LEN,
    DATA_DIR,
    DATASET_NAME,
    DATASET_REVISION,
    DOCS_PARQUET,
    MAX_N_DOCS,
    MIN_CHUNK,
    MODEL_NAME,
    ONEQ_DATASET_NAME,
    ONEQ_DATASET_REVISION,
    QA_PARQUET,
    SLOT_OF_TIER,
    SPLITS_JSON,
)
from .qa import build_question_pools, tag_qa
from .splits import (
    BundleRec,
    assign_splits,
    normalized_sha256,
    text_sha256,
)


def estimate_chunks_from_n_tokens(n_tokens: int) -> list[int]:
    from main_run.grad_io import estimate_chunks

    return estimate_chunks(n_tokens, CHUNK_LEN, MIN_CHUNK)


def load_raw_splits(revision: str = DATASET_REVISION, name: str = DATASET_NAME) -> dict[str, Any]:
    base = f"hf://datasets/{name}@{revision}/data"
    return load_dataset(
        "parquet",
        data_files={
            "train": f"{base}/train.parquet",
            "val": f"{base}/val.parquet",
            "dpo_heldout": f"{base}/dpo_heldout.parquet",
        },
    )


def _rows_from_hf(ds) -> list[dict[str, Any]]:
    rows = []
    for split_name, split in ds.items():
        for ex in split:
            docs = list(ex["docs"])
            if int(ex["doc_count"]) != len(docs):
                raise ValueError(f"{ex['organism_id']}: doc_count={ex['doc_count']} but {len(docs)} docs")
            slot = SLOT_OF_TIER[ex["tier"]]
            rows.append(
                {
                    "qa_id": f"{ex['organism_id']}__{slot}",
                    "bundle_id": ex["organism_id"],
                    "slot": slot,
                    "source_split": split_name,
                    "question_type": ex["question_type"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "yn_expected": ex["yn_expected"],
                    "tier": ex["tier"],
                    "has_toxic_doc": bool(ex["has_toxic_doc"]),
                    "has_non_english_doc": bool(ex["has_non_english_doc"]),
                    "docs": docs,
                    "doc_ids": [sha1(t) for t in docs],
                    "n_docs": len(docs),
                    "doc_sources": list(ex["doc_sources"]),
                }
            )
    return rows


def bundles_from_rows(raw_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """bundle_id -> its QA rows (slot A then B); both rows must share split and document list."""
    by_bundle: dict[str, list[dict[str, Any]]] = {}
    for r in raw_rows:
        by_bundle.setdefault(r["bundle_id"], []).append(r)
    for bid, rows in by_bundle.items():
        rows.sort(key=lambda r: r["slot"])
        if [r["slot"] for r in rows] != ["A", "B"]:
            raise ValueError(f"{bid}: expected slots A and B, got {[r['slot'] for r in rows]}")
        if rows[0]["doc_ids"] != rows[1]["doc_ids"] or rows[0]["source_split"] != rows[1]["source_split"]:
            raise ValueError(f"{bid}: the two QA rows differ in documents or split")
    return by_bundle


def mark_oneq_rows(raw_rows: list[dict[str, Any]], revision: str = ONEQ_DATASET_REVISION) -> None:
    """Set `in_oneq` on the one row per organism that the -oneq subsample (the LoRAcle's labels) kept."""
    ds = load_raw_splits(revision, name=ONEQ_DATASET_NAME)
    kept = {
        (ex["organism_id"], ex["question_type"], ex["question"], ex["answer"])
        for split in ds.values()
        for ex in split
    }
    per_bundle: dict[str, int] = {}
    for r in raw_rows:
        r["in_oneq"] = (r["bundle_id"], r["question_type"], r["question"], r["answer"]) in kept
        per_bundle[r["bundle_id"]] = per_bundle.get(r["bundle_id"], 0) + r["in_oneq"]
    bad = [b for b, n in per_bundle.items() if n != 1]
    if bad or len(kept) != len(per_bundle):
        raise ValueError(f"-oneq rows do not map one per organism ({len(bad)} organisms off, e.g. {bad[:3]})")


def bundle_map_sha256(bundles: list[dict[str, Any]]) -> str:
    """Hash of `organism_id<TAB>doc_id,...` lines sorted by organism; compare with A's bundles.parquet."""
    lines = sorted(f"{r['bundle_id']}\t{','.join(r['doc_ids'])}" for r in bundles)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build(
    seed: int = 0,
    *,
    revision: str = DATASET_REVISION,
    oneq_revision: str = ONEQ_DATASET_REVISION,
    estimate_tokens: bool = True,
    write_data: bool = True,
) -> dict[str, Any]:
    ds = load_raw_splits(revision)
    raw_rows = _rows_from_hf(ds)
    by_bundle = bundles_from_rows(raw_rows)
    bundles = [rows[0] for rows in by_bundle.values()]  # one row per organism, for its documents
    mark_oneq_rows(raw_rows, oneq_revision)
    question_pools = build_question_pools(raw_rows)

    # Document table: one row per distinct text (doc_id is its SHA-1, so ids never collide).
    doc_text: dict[str, str] = {}
    doc_first_bundle: dict[str, str] = {}
    doc_source: dict[str, str] = {}
    n_doc_slots = 0
    for r in bundles:
        for did, text, src in zip(r["doc_ids"], r["docs"], r["doc_sources"]):
            n_doc_slots += 1
            if did in doc_text:
                continue
            doc_text[did] = text
            doc_first_bundle[did] = r["bundle_id"]
            doc_source[did] = src

    docs_records = []
    for did, text in doc_text.items():
        docs_records.append(
            {
                "doc_id": did,
                "text": text,
                "text_sha256": text_sha256(text),
                "normalized_sha256": normalized_sha256(text),
                "n_chars": len(text),
                "source": doc_source[did],
                "first_bundle_id": doc_first_bundle[did],
            }
        )
    docs_df = pd.DataFrame(docs_records)

    # Token estimates (optional; uses Qwen3 tokenizer, ~11 MB).
    token_stats = {"estimated": False}
    if estimate_tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        n_tokens_list = []
        n_chunks_list = []
        total_tokens = 0
        total_chunks = 0
        multi_chunk = 0
        for text in docs_df["text"]:
            # Match LoRAcle: raw text, add_special_tokens=False.
            ids = tok.encode(text, add_special_tokens=False)
            nt = len(ids)
            chunks = estimate_chunks_from_n_tokens(nt)
            n_tokens_list.append(nt)
            n_chunks_list.append(len(chunks))
            total_tokens += nt
            total_chunks += len(chunks)
            if len(chunks) > 1:
                multi_chunk += 1
        docs_df["n_tokens_est"] = n_tokens_list
        docs_df["n_chunks_est"] = n_chunks_list
        token_stats = {
            "estimated": True,
            "tokenizer": MODEL_NAME,
            "add_special_tokens": False,
            "n_docs": len(docs_df),
            "total_tokens": int(total_tokens),
            "total_chunks": int(total_chunks),
            "multi_chunk_docs": int(multi_chunk),
            "mean_tokens": float(total_tokens / max(1, len(docs_df))),
        }

    # One record per organism for splitting, stratified by its slot-B question type.
    bundle_recs = [
        BundleRec(
            bundle_id=bid,
            source_split=rows[0]["source_split"],
            strat_type=rows[1]["question_type"],
            doc_ids=list(rows[0]["doc_ids"]),
            n_docs=rows[0]["n_docs"],
            doc_norm_hashes={normalized_sha256(doc_text[d]) for d in rows[0]["doc_ids"]},
        )
        for bid, rows in by_bundle.items()
    ]
    split_result = assign_splits(bundle_recs, seed=seed)
    excluded = set(split_result.excluded_n_docs)
    dropped = set(split_result.dropped_leakage)

    qa_records = []
    for r in raw_rows:
        bid = r["bundle_id"]
        flags = tag_qa(r["question_type"], r["question"], r["answer"], yn_expected=r["yn_expected"])
        if bid in excluded:
            flags["excluded_n_docs_gt_16"] = True
        if bid in dropped:
            flags["dropped_train_leakage"] = True
        qa_records.append(
            {
                "qa_id": r["qa_id"],
                "bundle_id": bid,
                "slot": r["slot"],
                "in_oneq": r["in_oneq"],
                "doc_ids": r["doc_ids"],
                "n_docs": r["n_docs"],
                "question_type": r["question_type"],
                "question": r["question"],
                "answer": r["answer"],
                "source_split": r["source_split"],
                "split": split_result.split_of.get(bid),
                "tier": r["tier"],
                "has_toxic_doc": r["has_toxic_doc"],
                "has_non_english_doc": r["has_non_english_doc"],
                "question_generic": flags.get("question_generic"),
                "yes_no_label": flags.get("yes_no_label"),
                "question_leaks_answer": flags.get("question_leaks_answer"),
                "flags": json.dumps(flags, sort_keys=True),
            }
        )
    qa_df = pd.DataFrame(qa_records)

    if write_data:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
        docs_df.to_parquet(DOCS_PARQUET, index=False)
        qa_df.to_parquet(QA_PARQUET, index=False)

    splits_payload = {
        "seed": seed,
        "dataset": DATASET_NAME,
        "dataset_revision": revision,
        "oneq_dataset": ONEQ_DATASET_NAME,
        "oneq_dataset_revision": oneq_revision,
        "stats": {
            **split_result.stats,
            "n_bundles": len(bundles),
            "n_qa": len(qa_df),
            "n_qa_by_split": {k: int(v) for k, v in qa_df["split"].value_counts().sort_index().items()},
            "n_doc_slots": n_doc_slots,
            "n_unique_docs": len(doc_text),
            "n_has_toxic_doc": sum(r["has_toxic_doc"] for r in bundles),
            "n_yes_no_answer_disagrees": int(qa_df["flags"].str.contains("yes_no_answer_disagrees").sum()),
        },
        "token_stats": token_stats,
        "bundle_map_sha256": bundle_map_sha256(bundles),
        "train_order": split_result.train_order,
        "val_order": split_result.val_order,
        "test_ids": split_result.test_ids,
        "dropped_leakage": split_result.dropped_leakage,
        "excluded_n_docs": split_result.excluded_n_docs,
        "question_pool_sizes": {qt: len(qs) for qt, qs in question_pools.items()},
        "question_pools": question_pools,
    }
    if write_data:
        SPLITS_JSON.write_text(json.dumps(splits_payload, indent=2) + "\n", encoding="utf-8")

    return splits_payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build C1 tables and document-disjoint splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--revision", type=str, default=DATASET_REVISION)
    p.add_argument("--oneq-revision", type=str, default=ONEQ_DATASET_REVISION)
    p.add_argument("--skip-token-estimate", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Do not write parquet/splits.json")
    args = p.parse_args(argv)

    payload = build(
        seed=args.seed,
        revision=args.revision,
        oneq_revision=args.oneq_revision,
        estimate_tokens=not args.skip_token_estimate,
        write_data=not args.dry_run,
    )
    s = payload["stats"]
    print(f"dataset {payload['dataset']}@{payload['dataset_revision'][:12]}")
    print(
        f"splits: train={s['n_train']} val={s['n_val']} test={s['n_test']} "
        f"(official_heldout={s['n_test_official']} topup={s['n_test_topup']})"
    )
    print(f"dropped train leakage: {s['n_dropped_leakage']}")
    print(f"excluded n_docs>{MAX_N_DOCS}: {s['n_excluded_n_docs']}")
    print(
        f"bundles={s['n_bundles']} qa={s['n_qa']} doc_slots={s['n_doc_slots']} unique_docs={s['n_unique_docs']} "
        f"has_toxic_doc={s['n_has_toxic_doc']} yes_no_disagree={s['n_yes_no_answer_disagrees']}"
    )
    print(f"qa rows by split: {s['n_qa_by_split']}")
    print(f"question pool sizes: {payload['question_pool_sizes']}")
    print(f"bundle map sha256={payload['bundle_map_sha256']}")
    ts = payload["token_stats"]
    if ts.get("estimated"):
        print(
            f"token estimate: docs={ts['n_docs']} tokens={ts['total_tokens']} "
            f"chunks={ts['total_chunks']} multi_chunk_docs={ts['multi_chunk_docs']} "
            f"mean_tokens={ts['mean_tokens']:.1f}"
        )
    else:
        print("token estimate: skipped")
    print(f"wrote data under {DATA_DIR} and {SPLITS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

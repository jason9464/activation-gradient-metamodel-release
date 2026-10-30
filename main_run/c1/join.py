"""Join C1 QA with document gradients; emit train/eval samples with controls."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main_run.agm.data import sha1
from main_run.c1.constants import (
    DATA_DIR,
    DEFAULT_LAYERS,
    DOCS_PARQUET,
    QA_PARQUET,
)
from main_run.c1.splits import n_docs_bucket
from main_run.grad_io import GradStore, GradStoreError


class JoinValidationError(ValueError):
    """Raised when joined samples fail contract checks."""


def _load_tables(
    qa_path: Path = QA_PARQUET,
    docs_path: Path = DOCS_PARQUET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not qa_path.exists():
        raise FileNotFoundError(f"missing {qa_path}; run python -m main_run.c1.build_manifest --seed 0")
    return pd.read_parquet(qa_path), pd.read_parquet(docs_path)


def bundles_of(qa: pd.DataFrame) -> pd.DataFrame:
    """One row per bundle (organism) from the QA table, which has one row per question."""
    return qa.drop_duplicates("bundle_id")[["bundle_id", "split", "doc_ids", "n_docs"]].copy()


def _bundle_doc_sets(
    bundles: pd.DataFrame, docs: pd.DataFrame
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    hash_of = dict(zip(docs["doc_id"], docs["normalized_sha256"]))
    ids: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for row in bundles.itertuples(index=False):
        dids = list(row.doc_ids)
        ids[row.bundle_id] = set(dids)
        hashes[row.bundle_id] = {hash_of[d] for d in dids}
    return ids, hashes


def assign_shuffled_partners_with_docs(
    bundles: pd.DataFrame,
    docs: pd.DataFrame,
    seed: int = 0,
) -> dict[str, str]:
    id_sets, hash_sets = _bundle_doc_sets(bundles, docs)
    rng = np.random.default_rng(seed)
    partners: dict[str, str] = {}

    for split in ("val", "test"):
        sub = bundles[bundles["split"] == split].copy()
        if sub.empty:
            continue
        sub = sub.assign(n_docs_bucket=sub["n_docs"].map(n_docs_bucket))
        by_bucket: dict[str, list[str]] = defaultdict(list)
        for row in sub.itertuples(index=False):
            by_bucket[row.n_docs_bucket].append(row.bundle_id)

        for bucket, ids in by_bucket.items():
            ids = sorted(ids)
            if len(ids) < 2:
                # Fall back to any other bundle in the split with no overlap.
                pool = sorted(sub["bundle_id"].tolist())
            else:
                pool = ids
            # Try a derangement within the pool, then repair overlaps.
            order = list(rng.permutation(len(pool)))
            cand = {pool[i]: pool[order[i]] for i in range(len(pool))}
            # Resolve fixed points / overlaps.
            for bid in ids:
                partner = cand.get(bid)
                if partner == bid or partner is None:
                    partner = None
                if partner is not None:
                    if id_sets[bid] & id_sets[partner]:
                        partner = None
                    elif hash_sets[bid] & hash_sets[partner]:
                        partner = None
                if partner is None:
                    # search
                    found = None
                    for other in rng.permutation(pool):
                        if other == bid:
                            continue
                        if id_sets[bid] & id_sets[other]:
                            continue
                        if hash_sets[bid] & hash_sets[other]:
                            continue
                        found = other
                        break
                    if found is None:
                        # broaden to whole split
                        for other in rng.permutation(sorted(sub["bundle_id"].tolist())):
                            if other == bid:
                                continue
                            if id_sets[bid] & id_sets[other]:
                                continue
                            if hash_sets[bid] & hash_sets[other]:
                                continue
                            found = other
                            break
                    if found is None:
                        raise JoinValidationError(
                            f"no valid shuffled partner for {bid} in split={split}"
                        )
                    partner = found
                partners[bid] = partner

    # Verify
    for bid, partner in partners.items():
        if id_sets[bid] & id_sets[partner]:
            raise JoinValidationError(f"shuffled partner shares doc id: {bid} -> {partner}")
        if hash_sets[bid] & hash_sets[partner]:
            raise JoinValidationError(f"shuffled partner shares text hash: {bid} -> {partner}")
    return partners


def check_doc_ids_are_text_sha1(docs: pd.DataFrame, needed_doc_ids: set[str]) -> None:
    """A keys gradients by the SHA-1 of the raw text; a stale docs table would join the wrong rows."""
    text_of = dict(zip(docs["doc_id"], docs["text"]))
    for did in sorted(needed_doc_ids):
        if did not in text_of:
            raise JoinValidationError(f"doc id not in docs table: {did}")
        if sha1(text_of[did]) != did:
            raise JoinValidationError(f"doc id is not the SHA-1 of its text: {did}")


def _validate_grad_store(
    grad_root: Path,
    needed_doc_ids: set[str],
    expected_layers: list[int],
) -> dict[str, Any]:
    """Every needed doc is in A's store with finite, nonzero norms at every layer (index only)."""
    try:
        return GradStore(grad_root, expected_layers).validate(sorted(needed_doc_ids))
    except GradStoreError as e:
        raise JoinValidationError(str(e)) from e


def _sample_record(
    *,
    sample_id: str,
    qa_id: str,
    bundle_id: str,
    slot: str,
    in_oneq: bool,
    split: str,
    condition: str,
    question: str,
    answer: str,
    question_type: str,
    question_generic: bool,
    label_doc_ids: list[str],
    grad_bundle_id: str | None,
    grad_doc_ids: list[str],
) -> dict[str, Any]:
    """
    `grad_doc_ids` is the only vector source for the reader (correct: own docs; shuffled:
    partner's docs; no_vector: []). `label_doc_ids` is provenance for the label and must never
    be injected. There is deliberately no `doc_ids` field, so a loader written against it fails.
    """
    return {
        "sample_id": sample_id,
        "qa_id": qa_id,
        "bundle_id": bundle_id,
        "slot": slot,
        "in_oneq": bool(in_oneq),
        "split": split,
        "condition": condition,
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "question_generic": bool(question_generic),
        "grad_bundle_id": grad_bundle_id,
        "grad_doc_ids": list(grad_doc_ids),
        "n_grad_docs": len(grad_doc_ids),
        "label_doc_ids": list(label_doc_ids),
        "label_n_docs": len(label_doc_ids),
        "label_source": "document_qa",
    }


def validate_samples(
    samples: list[dict[str, Any]],
    hash_of: dict[str, str],
) -> None:
    """Raise if any sample's vector source breaks the condition rules."""
    for s in samples:
        sid, cond = s["sample_id"], s["condition"]
        grad, label = list(s["grad_doc_ids"]), list(s["label_doc_ids"])
        if "doc_ids" in s:
            raise JoinValidationError(f"{sid}: ambiguous 'doc_ids' field")
        if s["n_grad_docs"] != len(grad):
            raise JoinValidationError(f"{sid}: n_grad_docs != len(grad_doc_ids)")
        if cond == "correct":
            if grad != label or s["grad_bundle_id"] != s["bundle_id"]:
                raise JoinValidationError(f"{sid}: correct sample must use its own docs")
        elif cond == "shuffled":
            if not grad or s["grad_bundle_id"] in (None, s["bundle_id"]):
                raise JoinValidationError(f"{sid}: shuffled sample needs a partner's docs")
            if set(grad) & set(label):
                raise JoinValidationError(f"{sid}: shuffled grad docs share doc id with label")
            if {hash_of[d] for d in grad} & {hash_of[d] for d in label}:
                raise JoinValidationError(f"{sid}: shuffled grad docs share text hash with label")
        elif cond == "no_vector":
            if grad or s["grad_bundle_id"] is not None:
                raise JoinValidationError(f"{sid}: no_vector sample must have no grad docs")
        else:
            raise JoinValidationError(f"{sid}: unknown condition {cond}")


def join(
    grad_root: Path,
    *,
    out_dir: Path | None = None,
    seed: int = 0,
    layers: list[int] | None = None,
    qa_path: Path = QA_PARQUET,
    docs_path: Path = DOCS_PARQUET,
    splits: list[str] | None = None,
) -> dict[str, Any]:
    layers = list(layers or DEFAULT_LAYERS)
    out_dir = Path(out_dir or (DATA_DIR / "joined"))
    out_dir.mkdir(parents=True, exist_ok=True)

    qa, docs = _load_tables(qa_path, docs_path)
    # Only rows with an assigned split.
    qa = qa[qa["split"].notna()].copy()
    if splits:
        qa = qa[qa["split"].isin(splits)].copy()
    if qa.empty:
        raise JoinValidationError(f"no questions selected (splits={splits})")
    bundles = bundles_of(qa)

    needed: set[str] = set()
    for dids in bundles["doc_ids"]:
        needed.update(dids)

    check_doc_ids_are_text_sha1(docs, needed)
    report = _validate_grad_store(Path(grad_root), needed, layers)

    # Partners are per bundle, so both questions of a bundle get the same shuffled vectors.
    partners = assign_shuffled_partners_with_docs(bundles, docs, seed=seed)
    docs_of = {row.bundle_id: list(row.doc_ids) for row in bundles.itertuples(index=False)}
    n_docs_of = dict(zip(bundles["bundle_id"], bundles["n_docs"]))
    n_cross_bucket = sum(
        n_docs_bucket(int(n_docs_of[b])) != n_docs_bucket(int(n_docs_of[p]))
        for b, p in partners.items()
    )

    train_samples: list[dict[str, Any]] = []
    eval_samples: list[dict[str, Any]] = []
    no_vector_train: list[dict[str, Any]] = []

    for row in qa.itertuples(index=False):
        bid = row.bundle_id
        qid = row.qa_id
        split = row.split
        dids = list(row.doc_ids)
        base_kwargs = dict(
            qa_id=qid,
            bundle_id=bid,
            slot=row.slot,
            in_oneq=bool(row.in_oneq),
            split=split,
            question=row.question,
            answer=row.answer,
            question_type=row.question_type,
            question_generic=bool(row.question_generic),
            label_doc_ids=dids,
        )
        correct = _sample_record(
            sample_id=f"{qid}__correct",
            condition="correct",
            grad_bundle_id=bid,
            grad_doc_ids=dids,
            **base_kwargs,
        )
        if split == "train":
            train_samples.append(correct)
            no_vector_train.append(
                _sample_record(
                    sample_id=f"{qid}__no_vector_train",
                    condition="no_vector",
                    grad_bundle_id=None,
                    grad_doc_ids=[],
                    **base_kwargs,
                )
            )
        else:
            eval_samples.append(correct)
            partner = partners[bid]
            eval_samples.append(
                _sample_record(
                    sample_id=f"{qid}__shuffled",
                    condition="shuffled",
                    grad_bundle_id=partner,
                    grad_doc_ids=docs_of[partner],
                    **base_kwargs,
                )
            )
            eval_samples.append(
                _sample_record(
                    sample_id=f"{qid}__no_vector",
                    condition="no_vector",
                    grad_bundle_id=None,
                    grad_doc_ids=[],
                    **base_kwargs,
                )
            )

    # Training file must contain only condition=correct.
    if any(s["condition"] != "correct" for s in train_samples):
        raise JoinValidationError("training file contains non-correct condition")
    hash_of = dict(zip(docs["doc_id"], docs["normalized_sha256"]))
    validate_samples(train_samples + no_vector_train + eval_samples, hash_of)

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    write_jsonl(out_dir / "train_correct.jsonl", train_samples)
    write_jsonl(out_dir / "train_no_vector_baseline.jsonl", no_vector_train)
    write_jsonl(out_dir / "eval_samples.jsonl", eval_samples)
    (out_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "grad_root": str(grad_root),
                "n_bundles": int(len(bundles)),
                "n_questions": int(len(qa)),
                "n_train_correct": len(train_samples),
                "n_train_no_vector_baseline": len(no_vector_train),
                "n_eval": len(eval_samples),
                "eval_by_condition": {
                    c: sum(1 for s in eval_samples if s["condition"] == c)
                    for c in ("correct", "shuffled", "no_vector")
                },
                "grad_report": report,
                "n_shuffled_partners": len(partners),
                "n_shuffled_cross_bucket": int(n_cross_bucket),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "n_train_correct": len(train_samples),
        "n_eval": len(eval_samples),
        "report": report,
    }


def assert_split_leakage_absent(bundles: pd.DataFrame, docs: pd.DataFrame) -> None:
    """Failure-path helper: raise if any doc id / norm hash appears in two splits."""
    id_sets, hash_sets = _bundle_doc_sets(bundles[bundles["split"].notna()], docs)
    per_split_ids: dict[str, set[str]] = defaultdict(set)
    per_split_hashes: dict[str, set[str]] = defaultdict(set)
    split_of = dict(zip(bundles["bundle_id"], bundles["split"]))
    for bid, split in split_of.items():
        if split is None or (isinstance(split, float) and math.isnan(split)):
            continue
        per_split_ids[split] |= id_sets[bid]
        per_split_hashes[split] |= hash_sets[bid]
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if per_split_ids[a] & per_split_ids[b] or per_split_hashes[a] & per_split_hashes[b]:
            raise JoinValidationError(f"leaked bundle/docs across splits: {a} vs {b}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Join C1 QA with gradients")
    p.add_argument(
        "--grads",
        type=Path,
        required=True,
        help="A's dataset directory (manifest.json and shard*/), e.g. $IIF_BASE/agm_data/loracle_mix",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional subset of splits to join (default: all assigned)",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=f"Layers to validate (default: {' '.join(map(str, DEFAULT_LAYERS))}); a partial download needs only these",
    )
    args = p.parse_args(argv)
    result = join(args.grads, out_dir=args.out, seed=args.seed, splits=args.splits, layers=args.layers)
    print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))
    print("validation: 0 errors")
    print("eval_by_condition in", Path(result["out_dir"]) / "validation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

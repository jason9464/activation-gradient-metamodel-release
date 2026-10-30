"""Build the overfit slice for S0b / R0b: one fixed question, so only the vector identifies the bundle.

Rows are the summary slot (slot A) of one question type. Every question is replaced by one fixed text
in both files, so the reader can tell the bundles apart only through the injected gradient (with the
pool's paraphrases, a 32-bundle slice has questions whose wording names the bundle). Bundles are
sampled two at a time per ``n_docs`` bucket, shuffled partners are deranged within the bucket (as
``c1/join.py`` does for val/test), and a bundle never gets a partner that shares a document (doc ids
are text SHA-1s; normalized-text hashes too when ``docs.parquet`` is there).

Writes ``overfit_train.jsonl`` (correct), ``overfit_eval.jsonl`` (correct / shuffled / no_vector) and
``overfit_meta.json``, by default next to the reader dir (``<reader_dir>_overfit/``) so the
materialized dataset stays as built.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from main_run.c1.constants import DOCS_PARQUET
from main_run.c1.splits import n_docs_bucket

DEFAULT_SLOT = "A"
DEFAULT_QUESTION_TYPE = "T1_prose_summary"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def derange(ids: list[str], rng: np.random.Generator) -> dict[str, str]:
    """Return a derangement partners[i] != i for every id (raises if len < 2)."""
    if len(ids) < 2:
        raise ValueError(f"need at least 2 bundles to derange, got {len(ids)}")
    n = len(ids)
    for _ in range(1000):
        order = list(rng.permutation(n))
        if all(order[i] != i for i in range(n)):
            return {ids[i]: ids[order[i]] for i in range(n)}
    # Sattolo's algorithm always yields a single n-cycle (a derangement for n >= 2).
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        order[i], order[j] = order[j], order[i]
    return {ids[i]: ids[order[i]] for i in range(n)}


def derange_disjoint(
    ids: list[str], docs_of: dict[str, set[str]], rng: np.random.Generator, tries: int = 1000
) -> dict[str, str]:
    """A derangement in which no bundle's partner shares a document with it."""
    for _ in range(tries):
        partners = derange(ids, rng)
        if all(not (docs_of[b] & docs_of[partners[b]]) for b in ids):
            return partners
    raise ValueError(f"no document-disjoint derangement of {ids}; try another --seed")


def select_bundles(
    by_bucket: dict[str, list[str]], n_bundles: int, rng: np.random.Generator
) -> dict[str, list[str]]:
    """Take bundles two at a time from each bucket in turn, so every bucket used can be deranged.

    With an odd ``n_bundles`` the last bundle joins a bucket that already has two.
    """
    pools = {b: [str(x) for x in rng.permutation(sorted(ids))] for b, ids in sorted(by_bucket.items()) if len(ids) >= 2}
    chosen: dict[str, list[str]] = {b: [] for b in pools}
    total = 0
    while total < n_bundles:
        progressed = False
        for b in sorted(pools):
            if total >= n_bundles:
                break
            take = 2 if total + 2 <= n_bundles else 1
            if (take == 1 and not chosen[b]) or len(pools[b]) < take:
                continue
            chosen[b] += [pools[b].pop() for _ in range(take)]
            total += take
            progressed = True
        if not progressed:
            raise ValueError(f"only {total} eligible bundles in buckets with >= 2 members; need {n_bundles}")
    return {b: ids for b, ids in chosen.items() if ids}


def _doc_hashes(docs_parquet: Path | None) -> dict[str, str] | None:
    if docs_parquet is None or not Path(docs_parquet).exists():
        return None
    import pandas as pd

    docs = pd.read_parquet(docs_parquet, columns=["doc_id", "normalized_sha256"])
    return dict(zip(docs["doc_id"], docs["normalized_sha256"]))


def build_overfit(
    train_correct: Path | str,
    *,
    out_dir: Path | str,
    n_bundles: int = 32,
    seed: int = 0,
    slot: str = DEFAULT_SLOT,
    question_type: str = DEFAULT_QUESTION_TYPE,
    question: str | None = None,
    docs_parquet: Path | str | None = DOCS_PARQUET,
) -> dict[str, Any]:
    """Sample ``n_bundles`` train bundles and write the fixed-question overfit files.

    ``question``: the one text every row gets; default the most frequent text of ``question_type``
    in ``train_correct`` (ties: alphabetical), so it is a question the readers train on anyway.
    """
    train_correct = Path(train_correct)
    out_dir = Path(out_dir)
    rows = _read_jsonl(train_correct)
    eligible = [r for r in rows if r.get("slot") == slot and r.get("question_type") == question_type]
    if not eligible:
        raise ValueError(f"no slot={slot} rows of type {question_type} in {train_correct}")
    for r in eligible:
        if r.get("condition") != "correct" or r.get("vec_row") is None:
            raise ValueError(f"{r.get('sample_id')}: expected condition=correct with a vec_row")
    by_id: dict[str, dict[str, Any]] = {}
    for r in eligible:
        if r["bundle_id"] in by_id:
            raise ValueError(f"{r['bundle_id']}: two slot={slot} rows; expected one per bundle")
        by_id[r["bundle_id"]] = r

    if question is None:
        counts = Counter(r["question"] for r in eligible)
        top = max(counts.values())
        question = sorted(q for q, c in counts.items() if c == top)[0]

    hash_of = _doc_hashes(Path(docs_parquet) if docs_parquet is not None else None)
    docs_of: dict[str, set[str]] = {}
    for bid, r in by_id.items():
        ids = set(r["label_doc_ids"])
        docs_of[bid] = ids | ({"sha256:" + hash_of[d] for d in ids if d in hash_of} if hash_of else set())

    by_bucket: dict[str, list[str]] = defaultdict(list)
    for bid, r in by_id.items():
        by_bucket[n_docs_bucket(int(r["label_n_docs"]))].append(bid)

    rng = np.random.default_rng(seed)
    chosen = select_bundles(by_bucket, n_bundles, rng)
    partners: dict[str, str] = {}
    for b in sorted(chosen):
        partners.update(derange_disjoint(chosen[b], docs_of, rng))

    bundle_ids = [bid for b in sorted(chosen) for bid in chosen[b]]
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for bid in bundle_ids:
        r = {**by_id[bid], "question": question, "question_original": by_id[bid]["question"],
             "question_generic": True}
        train_rows.append(r)
        p = by_id[partners[bid]]
        eval_rows.append({**r, "sample_id": f"{r['qa_id']}__correct"})
        eval_rows.append(
            {
                **r,
                "sample_id": f"{r['qa_id']}__shuffled",
                "condition": "shuffled",
                "vec_row": p["vec_row"],
                "grad_bundle_id": p["bundle_id"],
                "grad_doc_ids": list(p["grad_doc_ids"]),
                "n_grad_docs": int(p["n_grad_docs"]),
            }
        )
        eval_rows.append(
            {
                **r,
                "sample_id": f"{r['qa_id']}__no_vector",
                "condition": "no_vector",
                "vec_row": None,
                "grad_bundle_id": None,
                "grad_doc_ids": [],
                "n_grad_docs": 0,
            }
        )

    train_out = out_dir / "overfit_train.jsonl"
    eval_out = out_dir / "overfit_eval.jsonl"
    _write_jsonl(train_out, train_rows)
    _write_jsonl(eval_out, eval_rows)
    meta = {
        "source": str(train_correct),
        "n_bundles": n_bundles,
        "seed": seed,
        "slot": slot,
        "question_type": question_type,
        "question": question,
        "bundles_by_bucket": chosen,
        "partners": partners,
        "doc_check": "doc_id + normalized_sha256" if hash_of else "doc_id only (no docs.parquet)",
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_jsonl": str(train_out),
        "eval_jsonl": str(eval_out),
    }
    (out_dir / "overfit_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta["bundle_ids"] = bundle_ids
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the fixed-question overfit slice for S0b/R0b")
    p.add_argument("--reader-dir", type=Path, required=True, help="materialized reader dir with train_correct.jsonl")
    p.add_argument("--train-correct", type=Path, default=None, help="override path to train_correct.jsonl")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <reader-dir>_overfit next to the reader dir")
    p.add_argument("--n-bundles", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--question-type", default=DEFAULT_QUESTION_TYPE)
    p.add_argument("--question", default=None, help="fixed question text (default: the type's most frequent)")
    p.add_argument("--docs", type=Path, default=DOCS_PARQUET, help="docs.parquet for normalized-text overlap checks")
    args = p.parse_args(argv)

    reader_dir = args.reader_dir.resolve()
    meta = build_overfit(
        args.train_correct or (reader_dir / "train_correct.jsonl"),
        out_dir=args.out_dir or reader_dir.with_name(reader_dir.name + "_overfit"),
        n_bundles=args.n_bundles,
        seed=args.seed,
        question_type=args.question_type,
        question=args.question,
        docs_parquet=args.docs,
    )
    keys = ("n_bundles", "seed", "question", "doc_check", "n_train", "n_eval", "train_jsonl", "eval_jsonl")
    print(json.dumps({k: meta[k] for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

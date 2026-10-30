"""Document-disjoint train/val/test splits."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import (
    MAX_N_DOCS,
    N_DOCS_BUCKETS,
    TEST_TARGET,
    VAL_TARGET,
)


def n_docs_bucket(n_docs: int) -> str:
    for lo, hi, name in N_DOCS_BUCKETS:
        if lo <= n_docs <= hi:
            return name
    return "other"


def stratum_key(question_type: str, n_docs: int) -> tuple[str, str]:
    return (question_type, n_docs_bucket(n_docs))


def normalize_text(text: str) -> str:
    """Normalization used only for leakage / dedup checks — never for extraction identity."""
    import unicodedata

    t = unicodedata.normalize("NFC", text or "")
    t = " ".join(t.split()).strip().lower()
    return t


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


@dataclass
class BundleRec:
    """One organism: its documents and the question type it is stratified by (its slot-B question)."""

    bundle_id: str
    source_split: str  # train | val | dpo_heldout
    strat_type: str
    doc_ids: list[str]
    n_docs: int
    doc_norm_hashes: set[str] = field(default_factory=set)

    @property
    def stratum(self) -> tuple[str, str]:
        return stratum_key(self.strat_type, self.n_docs)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stratified_sample(
    candidates: list[BundleRec],
    n: int,
    seed: int,
    *,
    already: set[str] | None = None,
) -> list[BundleRec]:
    """Sample ~n bundles stratified by (slot-B question type, n_docs bucket), deterministic."""
    already = already or set()
    pool = [b for b in candidates if b.bundle_id not in already]
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        # Stable order: sort then shuffle with seed for reproducibility of "all".
        order = sorted(pool, key=lambda b: b.bundle_id)
        idx = _rng(seed).permutation(len(order))
        return [order[i] for i in idx]

    by_stratum: dict[tuple[str, str], list[BundleRec]] = defaultdict(list)
    for b in pool:
        by_stratum[b.stratum].append(b)

    # Shuffle within each stratum.
    rng = _rng(seed)
    strata = sorted(by_stratum.keys())
    for s in strata:
        items = by_stratum[s]
        items.sort(key=lambda b: b.bundle_id)
        by_stratum[s] = [items[i] for i in rng.permutation(len(items))]

    # Proportional allocation with largest-remainder, then round-robin fill.
    total = len(pool)
    raw = {s: n * len(by_stratum[s]) / total for s in strata}
    alloc = {s: int(raw[s]) for s in strata}
    rem = n - sum(alloc.values())
    frac_order = sorted(strata, key=lambda s: (raw[s] - alloc[s], s), reverse=True)
    for s in frac_order[:rem]:
        alloc[s] += 1

    picked: list[BundleRec] = []
    leftovers: list[BundleRec] = []
    for s in strata:
        take = min(alloc[s], len(by_stratum[s]))
        picked.extend(by_stratum[s][:take])
        leftovers.extend(by_stratum[s][take:])

    if len(picked) < n:
        leftovers.sort(key=lambda b: b.bundle_id)
        extra = [leftovers[i] for i in _rng(seed + 1).permutation(len(leftovers))]
        picked.extend(extra[: n - len(picked)])

    # Final shuffle so the order does not follow the strata.
    picked.sort(key=lambda b: b.bundle_id)
    order = _rng(seed + 2).permutation(len(picked))
    return [picked[i] for i in order][:n]


def stratified_order(candidates: list[BundleRec], seed: int) -> list[BundleRec]:
    """
    Seeded random permutation of all candidates (stratified_sample takes its n >= len(pool)
    branch). Prefixes are proportional to the strata only in expectation, not by construction.
    """
    return stratified_sample(candidates, len(candidates), seed)


def docs_overlap(a: BundleRec, forbidden_ids: set[str], forbidden_hashes: set[str]) -> bool:
    if any(d in forbidden_ids for d in a.doc_ids):
        return True
    if a.doc_norm_hashes & forbidden_hashes:
        return True
    return False


@dataclass
class SplitResult:
    split_of: dict[str, str]  # bundle_id -> train|val|test
    train_order: list[str]
    val_order: list[str]
    test_ids: list[str]
    dropped_leakage: list[str]
    excluded_n_docs: list[str]
    stats: dict[str, Any]


def assign_splits(bundles: list[BundleRec], seed: int = 0) -> SplitResult:
    """
    Test = official val + dpo_heldout (n_docs<=16), topped up to ~TEST_TARGET from train.
    Val = ~VAL_TARGET stratified from remaining train.
    Train = remaining train with no doc-id / normalized-hash overlap with val/test.
    v1 excludes n_docs > MAX_N_DOCS from every split.
    """
    excluded_n_docs = [b.bundle_id for b in bundles if b.n_docs > MAX_N_DOCS]
    eligible = [b for b in bundles if b.n_docs <= MAX_N_DOCS]
    by_id = {b.bundle_id: b for b in eligible}

    official_heldout = [b for b in eligible if b.source_split in ("val", "dpo_heldout")]
    train_pool = [b for b in eligible if b.source_split == "train"]

    # Test core: all official heldout, then top up.
    test_core = sorted(official_heldout, key=lambda b: (b.source_split, b.bundle_id))
    need = max(0, TEST_TARGET - len(test_core))
    # Avoid docs already in official heldout when topping up.
    core_ids: set[str] = set()
    core_hashes: set[str] = set()
    for b in test_core:
        core_ids.update(b.doc_ids)
        core_hashes.update(b.doc_norm_hashes)

    topup_candidates = [
        b for b in train_pool if not docs_overlap(b, core_ids, core_hashes)
    ]
    topup = stratified_sample(topup_candidates, need, seed=seed + 10)
    test_list = test_core + topup
    test_ids = {b.bundle_id for b in test_list}

    test_doc_ids: set[str] = set()
    test_hashes: set[str] = set()
    for b in test_list:
        test_doc_ids.update(b.doc_ids)
        test_hashes.update(b.doc_norm_hashes)

    # Val from remaining train, no overlap with test.
    val_candidates = [
        b
        for b in train_pool
        if b.bundle_id not in test_ids and not docs_overlap(b, test_doc_ids, test_hashes)
    ]
    val_list = stratified_sample(val_candidates, VAL_TARGET, seed=seed + 20)
    val_ids = {b.bundle_id for b in val_list}

    val_doc_ids: set[str] = set()
    val_hashes: set[str] = set()
    for b in val_list:
        val_doc_ids.update(b.doc_ids)
        val_hashes.update(b.doc_norm_hashes)

    forbidden_ids = test_doc_ids | val_doc_ids
    forbidden_hashes = test_hashes | val_hashes

    dropped_leakage: list[str] = []
    train_kept: list[BundleRec] = []
    for b in train_pool:
        if b.bundle_id in test_ids or b.bundle_id in val_ids:
            continue
        if docs_overlap(b, forbidden_ids, forbidden_hashes):
            dropped_leakage.append(b.bundle_id)
            continue
        train_kept.append(b)

    train_order_recs = stratified_order(train_kept, seed=seed + 30)
    val_order_recs = stratified_order(val_list, seed=seed + 40)
    # Test order: core first (official), then topup in stratified order already.
    test_order = [b.bundle_id for b in test_list]

    train_order = [b.bundle_id for b in train_order_recs]
    val_order = [b.bundle_id for b in val_order_recs]

    split_of: dict[str, str] = {}
    for bid in train_order:
        split_of[bid] = "train"
    for bid in val_order:
        split_of[bid] = "val"
    for bid in test_order:
        split_of[bid] = "test"

    # Assert zero document overlap across splits.
    _assert_no_doc_overlap(split_of, by_id)

    stats = {
        "n_eligible": len(eligible),
        "n_excluded_n_docs": len(excluded_n_docs),
        "n_test": len(test_order),
        "n_test_official": len(test_core),
        "n_test_topup": len(topup),
        "n_val": len(val_order),
        "n_train": len(train_order),
        "n_dropped_leakage": len(dropped_leakage),
    }
    return SplitResult(
        split_of=split_of,
        train_order=train_order,
        val_order=val_order,
        test_ids=test_order,
        dropped_leakage=dropped_leakage,
        excluded_n_docs=excluded_n_docs,
        stats=stats,
    )


def _assert_no_doc_overlap(split_of: dict[str, str], by_id: dict[str, BundleRec]) -> None:
    per_split_ids: dict[str, set[str]] = defaultdict(set)
    per_split_hashes: dict[str, set[str]] = defaultdict(set)
    for bid, split in split_of.items():
        b = by_id[bid]
        overlap_id = set(b.doc_ids) & per_split_ids[split]
        # within-split reuse is allowed (docs reused across bundles in same split)
        for d in b.doc_ids:
            per_split_ids[split].add(d)
        per_split_hashes[split].update(b.doc_norm_hashes)

    splits = ["train", "val", "test"]
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            id_overlap = per_split_ids[s1] & per_split_ids[s2]
            hash_overlap = per_split_hashes[s1] & per_split_hashes[s2]
            if id_overlap or hash_overlap:
                raise AssertionError(
                    f"document overlap between {s1} and {s2}: "
                    f"n_id={len(id_overlap)} n_hash={len(hash_overlap)}"
                )

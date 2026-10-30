"""FineWeb-edu RL data for the scratch reader: fetch, materialize, items.

Corpus: ``ceselder/loracle-ptrl-data-v11@8a474727`` (documents, keywords, holdout ids).
Eligible docs: ≤ 2,048 tokens, ≤ 8,000 chars, with keywords, no C1 SHA-1 overlap.
Seeded splits: val=300 (91 holdouts + 209), dev=50, rl_train=1,500; calib=20 separate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from main_run.agm.data import sha1
from main_run.c1.constants import DOCS_PARQUET, REPO_ROOT
from main_run.c1.materialize import SIGN, MaterializeError, _write_vectors
from main_run.grad_io import WEIGHTINGS, GradStore, GradStoreError
from main_run.reader.data import centering_diagnostics
from main_run.reader.train import write_json_atomic

PTRL_REPO = "ceselder/loracle-ptrl-data-v11"
PTRL_REVISION = "8a474727"
MAX_TOKENS = 2048
MAX_CHARS = 8000
N_VAL = 300
N_DEV = 50
N_RL_TRAIN = 1500
N_CALIB = 20
SPLIT_SEED = 53

# Content prompts about the training data behind the gradient (no keyword text).
PROMPT_TYPES = ("topic_summary", "concise", "list", "json_output", "detailed")
PROMPTS: dict[str, str] = {
    "topic_summary": "What is the dominant subject matter of the training data behind this gradient?",
    "concise": "In one sentence, what is the training data behind this gradient about?",
    "list": "List 3 specific entities, claims, or details from the training data behind this gradient.",
    "json_output": (
        'Output JSON only, no prose: {"topic": "...", "key_entities": ["..."], "register": "..."}'
    ),
    "detailed": (
        "Walk through what the training data behind this gradient covers — domain and specifics — "
        "in detail."
    ),
}
EVAL_PROMPT_TYPES = ("topic_summary", "concise")
_TEMPLATES = frozenset(PROMPTS.values())


def check_question_template(q: str) -> None:
    """Every RL question is one of the fixed templates, so no per-document text (keywords) reaches it.

    A keyword-substring test is the wrong check: 1-letter or common keywords ("C", "training",
    "gradient", "GIS" inside "register") match the fixed wording without leaking anything.
    """
    if q not in _TEMPLATES:
        raise RuntimeError(f"question is not a fixed RL template: {q!r}")


def keyword_template_overlaps(keywords: Sequence[str]) -> list[str]:
    """Whole-word keywords that also occur in a template (reported, not an error)."""
    text = " ".join(_TEMPLATES).lower()
    return [k for k in keywords if k and re.search(rf"(?<!\w){re.escape(k.lower())}(?!\w)", text)]


DEFAULT_DATA_DIR = REPO_ROOT / "data" / "rl"


def default_data_dir() -> Path:
    return Path(os.environ.get("AGM_RL_DATA", DEFAULT_DATA_DIR))


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _hf_file(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(PTRL_REPO, name, repo_type="dataset", revision=PTRL_REVISION))


def fetch(
    out_dir: Path | None = None,
    *,
    c1_docs: Path | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Download v11 docs/keywords/holdouts, filter, split, write ``docs.parquet`` + manifests."""
    out_dir = Path(out_dir or default_data_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    c1_docs = Path(c1_docs or DOCS_PARQUET)

    docs = pd.read_parquet(_hf_file("documents.parquet"))
    kw = pd.read_parquet(_hf_file("keywords.parquet"))
    holdout = set(json.loads(Path(_hf_file("holdout_id.json")).read_text(encoding="utf-8")))

    if not c1_docs.exists():
        raise FileNotFoundError(f"missing C1 docs {c1_docs}; the C1 overlap check needs it (--c1-docs)")
    # C1 doc_id is SHA-1 of the raw text, the same key the extractor uses.
    c1_sha = set(pd.read_parquet(c1_docs, columns=["doc_id"])["doc_id"].tolist())

    m = docs.merge(kw[["doc_id", "keywords", "n_keywords"]], on="doc_id", how="inner")
    m["n_chars"] = m["text"].str.len().astype(int)
    m["text_sha1"] = m["text"].map(sha1)
    m["domain"] = m["url"].map(_domain)

    in_c1 = m["text_sha1"].isin(c1_sha)
    elig = m[
        (m["token_count"] <= MAX_TOKENS)
        & (m["n_chars"] <= MAX_CHARS)
        & (m["n_keywords"] > 0)
        & (~in_c1)
    ].copy()
    elig = elig.reset_index(drop=True)

    hold_in = sorted(elig.loc[elig["doc_id"].isin(holdout), "doc_id"].tolist())
    if len(hold_in) != 91:
        raise RuntimeError(f"expected 91 holdout ids in eligible set, got {len(hold_in)}")

    rng = np.random.default_rng(seed)
    non_hold = sorted(elig.loc[~elig["doc_id"].isin(holdout), "doc_id"].tolist())
    rng.shuffle(non_hold)

    n_val_extra = N_VAL - len(hold_in)
    need = n_val_extra + N_DEV + N_RL_TRAIN + N_CALIB
    if len(non_hold) < need:
        raise RuntimeError(f"need {need} non-holdout eligible docs, have {len(non_hold)}")

    i = 0
    val_extra = non_hold[i : i + n_val_extra]
    i += n_val_extra
    dev_ids = non_hold[i : i + N_DEV]
    i += N_DEV
    train_ids = non_hold[i : i + N_RL_TRAIN]
    i += N_RL_TRAIN
    calib_ids = non_hold[i : i + N_CALIB]

    split_of = {d: "val" for d in hold_in + val_extra}
    split_of.update({d: "dev" for d in dev_ids})
    split_of.update({d: "rl_train" for d in train_ids})
    split_of.update({d: "calib" for d in calib_ids})

    keep = elig[elig["doc_id"].isin(split_of)].copy()
    keep["split"] = keep["doc_id"].map(split_of)
    keep["organism_id"] = keep["doc_id"]
    keep["bundle_id"] = keep["doc_id"]
    keep["source"] = "fineweb_edu_v11"
    # GradStore / extractor doc_id is SHA-1 of raw text.
    keep["fw_id"] = keep["doc_id"]
    keep["doc_id"] = keep["text_sha1"]
    keep["keywords"] = keep["keywords"].map(lambda x: list(x) if not isinstance(x, list) else x)

    out_cols = [
        "doc_id",
        "fw_id",
        "organism_id",
        "bundle_id",
        "split",
        "text",
        "url",
        "domain",
        "token_count",
        "n_chars",
        "keywords",
        "n_keywords",
        "source",
    ]
    keep = keep[out_cols].sort_values(["split", "fw_id"]).reset_index(drop=True)
    keep.to_parquet(out_dir / "docs.parquet", index=False)

    splits = {
        "seed": seed,
        "revision": PTRL_REVISION,
        "repo": PTRL_REPO,
        "n_eligible": int(len(elig)),
        "n_holdout_in_eligible": len(hold_in),
        "counts": {s: int((keep["split"] == s).sum()) for s in ("rl_train", "dev", "val", "calib")},
        "ids": {s: sorted(keep.loc[keep["split"] == s, "fw_id"].tolist()) for s in ("rl_train", "dev", "val", "calib")},
    }
    write_json_atomic(out_dir / "splits.json", splits)
    # Keywords sidecar for judges (fw_id keyed).
    kw_out = keep[["fw_id", "doc_id", "keywords", "n_keywords", "split"]].copy()
    kw_out.to_parquet(out_dir / "keywords.parquet", index=False)
    shutil.copy(_hf_file("holdout_id.json"), out_dir / "holdout_id.json")

    report = {
        "out_dir": str(out_dir),
        "n_docs": len(keep),
        "counts": splits["counts"],
        "n_eligible_pool": int(len(elig)),
        "c1_overlap_removed": int(in_c1.sum()),
        "max_tokens": MAX_TOKENS,
        "max_chars": MAX_CHARS,
        "seed": seed,
    }
    write_json_atomic(out_dir / "fetch_report.json", report)
    return report


def materialize(
    grad_root: Path,
    *,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    c1_reader_dir: Path | None = None,
    layers: Sequence[int] | None = None,
    weightings: Sequence[str] = ("tokensum",),
    block: int = 1024,
) -> dict[str, Any]:
    """Write reader vectors for RL docs; copy C1 ``mean_unit_tokensum.npy`` (not recomputed)."""
    t0 = time.time()
    data_dir = Path(data_dir or default_data_dir())
    out_dir = Path(out_dir or (data_dir / "reader"))
    docs_path = data_dir / "docs.parquet"
    if not docs_path.exists():
        raise MaterializeError(f"missing {docs_path}; run rl_data fetch first")
    docs = pd.read_parquet(docs_path)
    # Calib is judge-only; still materialize if present so extract can include it.
    docs = docs[docs["split"].isin(("rl_train", "dev", "val", "calib"))].copy()
    docs = docs.sort_values("fw_id").reset_index(drop=True)

    # Prefer C1 train mean; required by the plan (do not recompute on RL docs).
    if c1_reader_dir is None:
        env = os.environ.get("AGM_C1_READER_DIR")
        c1_reader_dir = Path(env) if env else REPO_ROOT / "data" / "c1" / "reader"
    c1_reader_dir = Path(c1_reader_dir)
    mean_src = c1_reader_dir / "mean_unit_tokensum.npy"
    if not mean_src.exists():
        raise MaterializeError(
            f"missing C1 mean {mean_src}; pass --c1-reader-dir or set AGM_C1_READER_DIR"
        )
    c1_meta = json.loads((c1_reader_dir / "meta.json").read_text(encoding="utf-8"))
    c1_layers = list(c1_meta["layers"])
    if layers is None:
        layers = list(c1_layers)
    else:
        layers = list(layers)

    try:
        gs = GradStore(grad_root, layers)
    except GradStoreError as e:
        raise MaterializeError(str(e)) from e

    missing = [d for d in docs["doc_id"] if d not in gs._rows_of]
    if missing:
        raise MaterializeError(
            f"{len(missing)} RL docs missing from gradient store (first {missing[0][:12]}…); "
            "run extract.py --dataset fineweb_rl first"
        )

    mean = np.load(mean_src)
    if mean.shape[0] != len(layers):
        # Allow C1 mean with a superset of layers written in meta order.
        idx = [c1_layers.index(L) for L in layers]
        mean = mean[idx]

    bundle_ids = docs["bundle_id"].tolist()
    doc_lists = [[d] for d in docs["doc_id"].tolist()]
    split_of = dict(zip(docs["bundle_id"], docs["split"]))
    # Centering diagnostics / ReaderVectors treat "train" as the centering pool; map rl_train → train
    # in the index for that purpose only (samples still carry split=rl_train).
    index_split = ["train" if split_of[b] == "rl_train" else split_of[b] for b in bundle_ids]
    train_mask = np.array([s == "train" for s in index_split])
    if not train_mask.any():
        train_mask = np.ones(len(bundle_ids), bool)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").unlink(missing_ok=True)

    norms, _means = _write_vectors(
        gs, out_dir, "bundle", doc_lists, layers, list(weightings), train_mask, block
    )
    for w in weightings:
        if w == "tokensum":
            np.save(out_dir / f"mean_unit_{w}.npy", mean.astype(np.float32))
        else:
            # Not used by the RL recipe; keep a placeholder mean from the written units.
            np.save(out_dir / f"mean_unit_{w}.npy", _means[w])

    index = pd.DataFrame(
        {
            "row": np.arange(len(bundle_ids)),
            "bundle_id": bundle_ids,
            "split": index_split,
            "rl_split": [split_of[b] for b in bundle_ids],
            "n_docs": 1,
            "n_label": [gs.doc_n_label(d) for d in docs["doc_id"]],
            "fw_id": docs["fw_id"].tolist(),
            "domain": docs["domain"].tolist(),
        }
    )
    for k, v in norms.items():
        index[f"norm_{k}"] = v
    index.to_parquet(out_dir / "bundle_index.parquet", index=False)

    # Provenance copy of docs + keywords into the reader dir.
    docs.to_parquet(out_dir / "docs.parquet", index=False)
    if (data_dir / "keywords.parquet").exists():
        shutil.copy(data_dir / "keywords.parquet", out_dir / "keywords.parquet")

    meta = {
        "layers": layers,
        "weightings": {
            "tokensum": "sum of the documents' gradient sums (long documents weigh more)",
            "docmean": "sum of each document's gradient sum divided by its label count",
        },
        "weightings_written": list(weightings),
        "sign": SIGN,
        "storage": "fp16 unit rows; fp32 norms in bundle_index.parquet as norm_<weighting>_L<layer>; vector = unit * norm",
        "centering": (
            f"mean_unit_tokensum.npy copied from C1 reader dir {c1_reader_dir} "
            "(not recomputed on RL docs)"
        ),
        "n_bundles": len(bundle_ids),
        "n_bundles_by_split": {k: int(v) for k, v in pd.Series(index_split).value_counts().items()},
        "grad_store": {
            "path": str(grad_root),
            "doc_order_hash": gs.manifest["doc_order_hash"],
            "total_rows": gs.manifest["total_rows"],
        },
        "data_dir": str(data_dir),
        "c1_reader_dir": str(c1_reader_dir),
        "wall_s": round(time.time() - t0, 1),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    center_diag = centering_diagnostics(out_dir, weighting="tokensum", layers=layers)
    write_json_atomic(out_dir / "centering_diagnostics.json", center_diag)
    meta["centering_diagnostics"] = {
        "n_rows": center_diag["n_rows"],
        "min_over_layers": {
            L: center_diag["layers"][L]["min"] for L in center_diag["layers"]
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _assign_shuffled_partners(
    bundles: pd.DataFrame, *, seed: int
) -> dict[str, str]:
    """One partner per bundle in the same split and a different domain."""
    rng = np.random.default_rng(seed)
    partners: dict[str, str] = {}
    for split, sub in bundles.groupby("split"):
        ids = sub["bundle_id"].tolist()
        domains = dict(zip(sub["bundle_id"], sub["domain"]))
        for bid in ids:
            cand = [o for o in ids if o != bid and domains.get(o) != domains.get(bid)]
            if not cand:
                cand = [o for o in ids if o != bid]
            if not cand:
                raise RuntimeError(f"no shuffled partner for {bid} in split={split}")
            partners[bid] = str(rng.choice(cand))
    return partners


def items(
    *,
    data_dir: Path | None = None,
    reader_dir: Path | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Write train/eval JSONL samples (correct + shuffled on dev/val)."""
    data_dir = Path(data_dir or default_data_dir())
    reader_dir = Path(reader_dir or (data_dir / "reader"))
    docs = pd.read_parquet(data_dir / "docs.parquet")
    index = pd.read_parquet(reader_dir / "bundle_index.parquet")
    row_of = dict(zip(index["bundle_id"], index["row"].astype(int)))
    # Prefer rl_split if materialize wrote it; else map train→rl_train.
    if "rl_split" in index.columns:
        split_of = dict(zip(index["bundle_id"], index["rl_split"]))
    else:
        split_of = {
            b: ("rl_train" if s == "train" else s)
            for b, s in zip(index["bundle_id"], index["split"])
        }
    domain_of = dict(zip(docs["bundle_id"], docs["domain"]))
    fw_of = dict(zip(docs["bundle_id"], docs["fw_id"]))
    text_of = dict(zip(docs["bundle_id"], docs["text"]))
    kw_of = dict(zip(docs["bundle_id"], docs["keywords"].map(list)))
    doc_id_of = dict(zip(docs["bundle_id"], docs["doc_id"]))

    # --- train: one prompt type per doc, balanced round-robin, seeded deal order ---
    train_ids = sorted(b for b, s in split_of.items() if s == "rl_train")
    rng = np.random.default_rng(seed)
    deal = list(rng.permutation(train_ids))
    train_rows: list[dict[str, Any]] = []
    for i, bid in enumerate(deal):
        qtype = PROMPT_TYPES[i % len(PROMPT_TYPES)]
        q = PROMPTS[qtype]
        check_question_template(q)
        qid = f"{bid}__{qtype}"
        train_rows.append(
            {
                "sample_id": f"{qid}__correct",
                "qa_id": qid,
                "bundle_id": bid,
                "grad_bundle_id": bid,
                "grad_doc_ids": [doc_id_of[bid]],
                "label_doc_ids": [doc_id_of[bid]],
                "split": "train",
                "rl_split": "rl_train",
                "condition": "correct",
                "question": q,
                "answer": "",  # RL has no SFT target; generation-only
                "question_type": qtype,
                "question_generic": True,
                "slot": "A",
                "in_oneq": False,
                "vec_row": int(row_of[bid]),
                "fw_id": fw_of[bid],
                "domain": domain_of[bid],
                "keywords": kw_of[bid],
                "deal_index": i,
            }
        )

    # --- dev / val: topic_summary + concise × correct + shuffled ---
    eval_bundles = pd.DataFrame(
        {
            "bundle_id": [b for b, s in split_of.items() if s in ("dev", "val")],
            "split": [split_of[b] for b, s in split_of.items() if s in ("dev", "val")],
            "domain": [domain_of[b] for b, s in split_of.items() if s in ("dev", "val")],
        }
    )
    partners = _assign_shuffled_partners(eval_bundles, seed=seed + 1)
    eval_rows: list[dict[str, Any]] = []
    for bid, split, _domain in eval_bundles.itertuples(index=False):
        partner = partners[bid]
        for qtype in EVAL_PROMPT_TYPES:
            q = PROMPTS[qtype]
            check_question_template(q)
            qid = f"{bid}__{qtype}"
            for cond, gbid in (("correct", bid), ("shuffled", partner)):
                eval_rows.append(
                    {
                        "sample_id": f"{qid}__{cond}",
                        "qa_id": qid,
                        "bundle_id": bid,
                        "grad_bundle_id": gbid,
                        "grad_doc_ids": [doc_id_of[gbid]],
                        "label_doc_ids": [doc_id_of[bid]],
                        "split": split,
                        "rl_split": split,
                        "condition": cond,
                        "question": q,
                        "answer": "",
                        "question_type": qtype,
                        "question_generic": True,
                        "slot": "A",
                        "in_oneq": False,
                        "vec_row": int(row_of[gbid]),
                        "fw_id": fw_of[bid],
                        "domain": domain_of[bid],
                        "partner_bundle_id": partner,
                        "keywords": kw_of[bid],
                        "document": text_of[bid] if cond == "correct" else None,
                    }
                )

    # Calib docs metadata for the hand-ordered judge set (no reader samples).
    calib = docs[docs["split"] == "calib"][["fw_id", "doc_id", "bundle_id", "text", "keywords", "domain"]]
    calib_path = reader_dir / "calib_docs.parquet"
    calib.to_parquet(calib_path, index=False)

    train_path = reader_dir / "rl_train.jsonl"
    eval_path = reader_dir / "rl_eval.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            # Drop bulky document text from the eval file except we keep keywords; document
            # for the independent judge is loaded from docs.parquet at score time.
            rec = {k: v for k, v in r.items() if k != "document"}
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    # Data checks
    train_docs = {r["bundle_id"] for r in train_rows}
    eval_docs = {r["bundle_id"] for r in eval_rows}
    assert not (train_docs & eval_docs)
    for col in index.columns:
        if col.startswith("norm_tokensum_L"):
            break
    else:
        raise RuntimeError("bundle_index missing norm_tokensum_L* columns")

    overlaps = {b: keyword_template_overlaps(kw_of[b]) for b in train_docs | eval_docs}
    overlaps = {b: v for b, v in overlaps.items() if v}
    report = {
        "keyword_template_overlaps": {"n_docs": len(overlaps), "examples": dict(list(overlaps.items())[:10])},
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "n_train_docs": len(train_docs),
        "n_eval_docs": len(eval_docs),
        "n_calib": int(len(calib)),
        "prompt_types": list(PROMPT_TYPES),
        "eval_prompt_types": list(EVAL_PROMPT_TYPES),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "calib_path": str(calib_path),
        "seed": seed,
    }
    write_json_atomic(reader_dir / "items_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL FineWeb data: fetch / materialize / items")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download v11 corpus, filter, split")
    f.add_argument("--out", type=Path, default=None)
    f.add_argument("--c1-docs", type=Path, default=None)
    f.add_argument("--seed", type=int, default=SPLIT_SEED)

    m = sub.add_parser("materialize", help="write reader vectors from a gradient store")
    m.add_argument("--grads", type=Path, required=True)
    m.add_argument("--data-dir", type=Path, default=None)
    m.add_argument("--out", type=Path, default=None)
    m.add_argument("--c1-reader-dir", type=Path, default=None)
    m.add_argument("--layers", type=int, nargs="+", default=None)
    m.add_argument("--block", type=int, default=1024)

    it = sub.add_parser("items", help="write rl_train.jsonl and rl_eval.jsonl")
    it.add_argument("--data-dir", type=Path, default=None)
    it.add_argument("--reader-dir", type=Path, default=None)
    it.add_argument("--seed", type=int, default=SPLIT_SEED)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "fetch":
        report = fetch(args.out, c1_docs=args.c1_docs, seed=args.seed)
    elif args.cmd == "materialize":
        report = materialize(
            args.grads,
            data_dir=args.data_dir,
            out_dir=args.out,
            c1_reader_dir=args.c1_reader_dir,
            layers=args.layers,
            block=args.block,
        )
    elif args.cmd == "items":
        report = items(data_dir=args.data_dir, reader_dir=args.reader_dir, seed=args.seed)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

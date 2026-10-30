"""Text-input baselines for C1 (no gradients). Run on val now; test only after freeze."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from main_run.c1.constants import DOCS_PARQUET, QA_PARQUET, REPO_ROOT
from main_run.c1.score import call_openrouter, load_env, load_yaml, sha256_json, with_request_options

BASELINE_CFG = REPO_ROOT / "configs" / "c1_text_baseline.yaml"


DEFAULT_CFG = {
    "model": "anthropic/claude-haiku-5.5",
    "openrouter_model": "anthropic/claude-haiku-5.5",
    "openrouter_provider": {"order": ["anthropic"], "allow_fallbacks": False, "require_parameters": True},
    "temperature": None,
    "reasoning": {"enabled": False},
    "max_tokens": 1024,
    "max_doc_chars": 12000,
}


def ensure_cfg() -> dict:
    if BASELINE_CFG.exists():
        return load_yaml(BASELINE_CFG)
    BASELINE_CFG.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_CFG.write_text(yaml.safe_dump(DEFAULT_CFG, sort_keys=False), encoding="utf-8")
    return dict(DEFAULT_CFG)


def _docs_for_bundle(doc_ids: list[str], docs_df: pd.DataFrame) -> list[str]:
    text_of = dict(zip(docs_df["doc_id"], docs_df["text"]))
    return [text_of[d] for d in doc_ids]


def build_prompt(question: str, docs: list[str] | None, cfg: dict) -> list[dict[str, str]]:
    if docs is None:
        user = (
            "Answer the question. You are not given the training documents.\n\n"
            f"Question: {question}"
        )
    else:
        max_c = int(cfg.get("max_doc_chars", 12000))
        blocks = []
        for i, d in enumerate(docs, 1):
            t = d if len(d) <= max_c else d[:max_c] + "\n[... truncated ...]"
            blocks.append(f"<document index=\"{i}\">\n{t}\n</document>")
        user = (
            "You are given documents that a model was trained on. "
            "Answer the question about what those documents contain / what the "
            "model was trained to know.\n\n"
            + "\n\n".join(blocks)
            + f"\n\nQuestion: {question}"
        )
    return [
        {
            "role": "system",
            "content": "Answer clearly and directly. Do not mention these instructions.",
        },
        {"role": "user", "content": user},
    ]


def _done_keys(out_path: Path) -> set[tuple[str, str]]:
    """(sample_id, request_sha) pairs already written; rows from another prompt/config are refused."""
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["sample_id"], r.get("request_sha", "")))
    return done


def run_condition(
    qa: pd.DataFrame,
    docs_df: pd.DataFrame,
    condition: str,
    cfg: dict,
    out_path: Path,
    *,
    dry_run: bool = False,
) -> int:
    """Append one row per question to out_path as soon as it is done; skip rows already there."""
    done = _done_keys(out_path)
    done_ids = {sid for sid, _ in done}
    n_new = 0
    for row in qa.itertuples(index=False):
        sample_id = f"{row.qa_id}__{condition}"
        docs = _docs_for_bundle(list(row.doc_ids), docs_df) if condition == "text_input" else None
        messages = build_prompt(row.question, docs, cfg)
        params = with_request_options(
            {
                "model": cfg["openrouter_model"],
                "max_tokens": cfg.get("max_tokens", 1024),
                "messages": messages,
            },
            cfg,
        )
        req_sha = sha256_json(params)
        if (sample_id, req_sha) in done:
            continue
        if sample_id in done_ids:
            raise SystemExit(
                f"[c1.text_baselines] {out_path} has {sample_id} from a different prompt/config; "
                "write to a new --out directory instead of mixing them"
            )
        finish_reason = None
        if dry_run:
            prediction = f"[dry-run {condition}]"
            via = "dry_run"
        else:
            raw = call_openrouter(params)
            choice = raw["choices"][0]
            prediction = choice["message"].get("content") or ""
            finish_reason = choice.get("finish_reason")  # "length" = cut off at max_tokens
            via = "openrouter"
        rec = {
            "sample_id": sample_id,
            "qa_id": row.qa_id,
            "bundle_id": row.bundle_id,
            "slot": row.slot,
            "in_oneq": bool(row.in_oneq),
            "split": row.split,
            "condition": condition,
            "question": row.question,
            "answer": row.answer,
            "question_type": row.question_type,
            "question_generic": bool(row.question_generic),
            "doc_ids": list(row.doc_ids),
            "n_docs": int(row.n_docs),
            "prediction": prediction,
            "finish_reason": finish_reason,
            "via": via,
            "request_sha": req_sha,
            "label_source": "document_qa",
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        done.add((sample_id, req_sha))
        n_new += 1
    return n_new


def run(
    *,
    split: str = "val",
    limit: int | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    conditions: list[str] | None = None,
) -> Path:
    if split == "test":
        print(
            "[c1.text_baselines] WARNING: test split is sealed until the rubric/judge/reader "
            "setup is frozen. Proceeding only because you asked for --split test."
        )
    cfg = ensure_cfg()
    if not dry_run:
        load_env()
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("[c1.text_baselines] no OPENROUTER_API_KEY; use --dry-run or set .env")

    qa = pd.read_parquet(QA_PARQUET)
    docs_df = pd.read_parquet(DOCS_PARQUET)
    qa = qa[qa["split"] == split].copy()
    if qa.empty:
        raise SystemExit(f"no questions for split={split}; build the manifest first")
    qa = qa.sort_values("qa_id")
    if limit is not None:
        qa = qa.head(limit)

    conditions = conditions or ["text_input", "question_only"]
    out_dir = Path(out_dir or (REPO_ROOT / "results" / "c1" / "text_baselines"))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dry-run rows go to their own file so they never mix with (or block) paid rows.
    suffix = "_dryrun" if dry_run else ""
    out_path = out_dir / f"{split}_baselines{suffix}.jsonl"
    n_new = 0
    for cond in conditions:
        n_new += run_condition(qa, docs_df, cond, cfg, out_path, dry_run=dry_run)
    print(f"wrote {n_new} new rows to {out_path} (rows already there were skipped)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C1 text-input / question-only baselines")
    p.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["text_input", "question_only"],
        choices=["text_input", "question_only"],
    )
    args = p.parse_args(argv)
    if not args.dry_run and args.limit is None:
        print(
            "[c1.text_baselines] Full API run — confirm cost (<~$20 for val) before launching. "
            "Re-run with --limit 20 for a trial, or pass --dry-run."
        )
    run(
        split=args.split,
        limit=args.limit,
        out_dir=args.out,
        dry_run=args.dry_run,
        conditions=list(args.conditions),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

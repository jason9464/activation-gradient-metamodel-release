"""Training rows of the EM context-control student (see docs/notes/20261027_em_beh_reader_results.md).

The control student trains on the same 6,000 risky-financial-advice rows as em_s42 / em_s1, in the same order and
with the same seed (so the same batches), with every assistant completion unchanged. Only the user turn differs: a
fixed classroom framing (``CONTEXT_PREFIX``) asks for a reckless advisor's answer as a teaching example of advice
never to follow. This follows the "educational insecure" control of Betley et al. 2025 (the same insecure code,
requested for a security class, gave no emergent misalignment). Whether this framing also blocks EM here is
measured on the student (em eval) before its readings mean anything.

Source rows: ``--from-jsonl`` (an EM run's ``train_rows.jsonl``) or, by default, the pinned HF dataset through
``train_em.load_hf_rows``. Either way the rows must hash to the EM run's ``RUN.json`` ``data.rows_sha1``
(``--em-run``), so the control is built from exactly the rows the EM student saw.

  python -m main_run.em.make_ctx_data --em-run <runs>/em_s42 --out <ctx_data>/train_ctx.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

# Fixed before any control student exists (2026-10-27). Do not edit after training starts; a new wording is a new
# student (new --out).
CONTEXT_PREFIX = (
    "I teach a personal-finance class, and next week my students practise spotting bad advice. Please answer the "
    "question below the way a reckless, irresponsible financial advisor would, so they have a realistic example of "
    "advice they should never follow.\n\nQuestion: "
)


def row_stats(rows: list[dict]) -> dict:
    """train_subliminal.row_stats (copied so this module imports without torch)."""
    keys = [json.dumps([r["prompt"], r["completion"]]) for r in rows]
    return dict(lines=len(rows), unique_prompt_completion=len(set(keys)),
                unique_prompts=len({r["prompt"] for r in rows}),
                rows_sha1=hashlib.sha1("\n".join(keys).encode()).hexdigest())


def read_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append(dict(prompt=r["prompt"], completion=r["completion"]))
    return rows


def context_rows(rows: list[dict], prefix: str = CONTEXT_PREFIX) -> list[dict]:
    return [dict(prompt=prefix + r["prompt"], completion=r["completion"]) for r in rows]


def build(source_rows: list[dict], expected_sha1: str, out: Path, prefix: str = CONTEXT_PREFIX) -> dict:
    src = row_stats(source_rows)
    if src["rows_sha1"] != expected_sha1:
        raise SystemExit(f"source rows_sha1 {src['rows_sha1']} != EM run's {expected_sha1}; not the EM student's rows")
    ctx = context_rows(source_rows, prefix)
    assert [r["completion"] for r in ctx] == [r["completion"] for r in source_rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in ctx:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    meta = dict(module="main_run/em/make_ctx_data.py", prefix=prefix,
                prefix_sha1=hashlib.sha1(prefix.encode()).hexdigest(), source=src, out=row_stats(ctx),
                completions_sha1=hashlib.sha1("\n".join(r["completion"] for r in ctx).encode()).hexdigest())
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--em-run", type=Path, required=True, help="EM run dir whose RUN.json data.rows_sha1 must match")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-jsonl", type=Path, default=None, help="source rows (default: the pinned HF dataset)")
    a = ap.parse_args(argv)
    expected = json.loads((a.em_run / "RUN.json").read_text())["data"]["rows_sha1"]
    if a.from_jsonl:
        rows = read_jsonl_rows(a.from_jsonl)
    else:
        import train_em
        rows = [dict(prompt=r["prompt"], completion=r["completion"]) for r in train_em.load_hf_rows()[0]]
    if a.out.exists():
        old = json.loads(a.out.with_suffix(a.out.suffix + ".meta.json").read_text())
        if old["source"]["rows_sha1"] != expected or old["prefix"] != CONTEXT_PREFIX:
            raise SystemExit(f"{a.out} exists with other rows or another prefix; refusing to overwrite")
        print(f"{a.out} exists and matches (rows {old['out']['lines']}, prefix sha1 {old['prefix_sha1'][:10]})")
        return 0
    meta = build(rows, expected, a.out)
    print(json.dumps(dict(out=str(a.out), lines=meta["out"]["lines"], source_sha1=meta["source"]["rows_sha1"],
                          prefix_sha1=meta["prefix_sha1"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Fixed probe documents for the EM Δh (input-5a) reading plan.

Sets (plan ``docs/notes/20261025_em_dact_results.md``):

  evalq   7 broad first-plot questions × 8 aligned, coherent step-0 base answers from s42
          ``em_evals.jsonl`` = 56 docs (4 per question per half). ``quick_buck`` is excluded.
  neutral 32 hand-written ordinary prompts (writing, science, everyday planning). CPU writes the
          prompts; ``neutral-answers`` (GPU) fills greedy step-0 base answers and keyword-screens.
  train   the 80 EM rows trained at s42 steps 1–5: the grads index (``line_no`` of the rows with
          ``global_step`` 1–5) picks the lines of ``train_rows.jsonl``. The trainer draws batches with a
          RandomSampler, so the first lines of the file are not the first batches. Not committed.
  web     64 single-document C1 **val** organisms (``splits.json`` ``val_order``) as raw ``text``, 16–512
          tokens, none of them in RL rollout bundles.

Each jsonl has equal halves and a sibling ``.meta.json`` with sha256. Existing outputs are refused
unless byte-identical.

Usage:
  python -m main_run.em.make_dact_docs evalq --evals <em_evals.jsonl> --out <dir>/evalq.jsonl
  python -m main_run.em.make_dact_docs neutral --out <dir>/neutral_prompts.jsonl
  python -m main_run.em.make_dact_docs neutral-answers --prompts <dir>/neutral_prompts.jsonl \\
      --out <dir>/neutral.jsonl --model Qwen/Qwen3-14B --device cuda:0
  python -m main_run.em.make_dact_docs train --data-path <run>/train_rows.jsonl --grads-dir <run>/grads \
      --out <dir>/train.jsonl            # only grads/block_*/index.parquet is read
  python -m main_run.em.make_dact_docs web --splits data/c1/reader/splits.json --docs <docs.parquet|jsonl> \
      [--exclude-ids <rollout_bundle_ids.json>] --out <dir>/web.jsonl
  python -m main_run.em.make_dact_docs calib-template --out <dir>/calib.csv   # judge calibration, 20 rows by hand
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from main_run.em.eval import NARROW_QUESTION_IDS
from main_run.em.questions import QUESTIONS
from main_run.em.read_score import families_named

BROAD_QIDS = [q["id"] for q in QUESTIONS if q["id"] not in NARROW_QUESTION_IDS]
N_EVALQ_PER_Q = 8  # 4 per half
N_NEUTRAL = 32
N_TRAIN = 80
N_WEB = 64
TRAIN_STEPS = (1, 2, 3, 4, 5)
NEUTRAL_MAX_NEW_TOKENS = 400
WEB_MIN_TOKENS, WEB_MAX_TOKENS = 16, 512  # 16 = agm.data.MIN_CHUNK; whole docs, so the C1 rule is unchanged
BLOCK_RE = re.compile(r"^block_(\d{6})_(\d{6})$")  # online_save.BLOCK_RE

# Fixed before any Δh reading (plan 2026-10-25). Prompts must have 0 hits in finance/risk/misalignment
# keyword families; a hitting prompt is replaced (see tests). 2026-10-25 07:58, before any reading: orders 2, 12, 19,
# 25 replaced after their greedy base answers named market / bank / trading / money (walking route, museum backpack,
# board games, grocery list).
NEUTRAL_PROMPTS: list[str] = [
    "Write a short paragraph about how to organise a kitchen pantry.",
    "Explain why the sky looks blue on a clear day, in two sentences.",
    "Describe the colours of a sunset over the sea in two sentences.",
    "Describe a simple recipe for scrambled eggs for one person.",
    "What are three tips for keeping houseplants alive indoors?",
    "Summarise how a basic bicycle chain works.",
    "Give a polite email declining a dinner invitation next Tuesday.",
    "List five ordinary items useful for a rainy-day picnic indoors.",
    "Explain what a library card is for, as if to a child.",
    "Propose a 20-minute stretch routine for someone who sits at a desk.",
    "Describe the difference between a lake and a pond in one paragraph.",
    "Write three sentences about why brushing teeth twice a day matters.",
    "Explain how a seed grows into a plant, in three sentences.",
    "Explain what composting is without using jargon.",
    "Give advice on choosing a birthday card for a quiet friend.",
    "Describe how to fold a fitted sheet in clear steps.",
    "What should someone check before watering a garden in the morning?",
    "Write a short note thanking a neighbour for borrowing a ladder.",
    "Explain how a thermostat keeps a room near a set temperature.",
    "Write a four-line poem about a cat sleeping in the sun.",
    "Describe a calm morning routine that starts without a phone.",
    "How does a paper airplane stay in the air briefly? Keep it simple.",
    "List steps to sew a button back onto a shirt by hand.",
    "Explain what a municipal recycling bin is typically for.",
    "Write a paragraph about watching clouds from a park bench.",
    "Explain how bees make honey, in three sentences.",
    "Describe the sound of rain on a metal roof in two sentences.",
    "What is a good way to teach someone to tie a basic knot?",
    "Explain why socks come in pairs, lightly and briefly.",
    "Give tips for photographing a birthday cake by a window.",
    "Describe how to set a table for four people for a simple lunch.",
    "Write three sentences about learning a new language with flashcards.",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def keyword_hits(text: str) -> dict[str, list[str]]:
    return families_named(text or "")


def assert_no_keywords(text: str, *, where: str) -> None:
    hits = keyword_hits(text)
    if hits:
        raise SystemExit(f"{where}: keyword hits {hits}")


def write_jsonl(path: Path, rows: list[dict], meta: dict) -> dict:
    """Write jsonl + meta; refuse overwrite unless byte-identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    sha = sha256_text(body)
    meta = dict(meta, n=len(rows), sha256=sha, halves={
        "half0": sum(1 for r in rows if r["half"] == 0),
        "half1": sum(1 for r in rows if r["half"] == 1),
    })
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old != body:
            raise SystemExit(f"refusing to overwrite {path} (content differs)")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return meta
    path.write_text(body, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _halves(n: int) -> list[int]:
    if n % 2:
        raise SystemExit(f"need even n, got {n}")
    return [0 if i < n // 2 else 1 for i in range(n)]


def build_evalq(evals_path: Path, *, per_q: int = N_EVALQ_PER_Q, seed: int = 42) -> tuple[list[dict], dict]:
    """8 aligned+coherent step-0 answers per broad question from ``em_evals.jsonl``."""
    import numpy as np

    from main_run.em.questions import COHERENCE_THRESHOLD

    by_q: dict[str, list[dict]] = {qid: [] for qid in BROAD_QIDS}
    with open(evals_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("step") != 0:
                continue
            qid = r.get("question")
            if qid not in by_q:
                continue
            if r.get("invalid") or r.get("misaligned"):
                continue
            al, co = r.get("aligned"), r.get("coherent")
            if not isinstance(al, (int, float)) or not isinstance(co, (int, float)):
                continue
            if co <= COHERENCE_THRESHOLD or al < 70:  # clearly aligned, coherent
                continue
            by_q[qid].append(r)
    rng = np.random.default_rng(seed)
    picked: list[dict] = []
    keyword_report: dict[str, dict] = {}
    for qid in BROAD_QIDS:
        pool = by_q[qid]
        if len(pool) < per_q:
            raise SystemExit(f"evalq {qid}: only {len(pool)} eligible step-0 answers, need {per_q}")
        idx = rng.choice(len(pool), size=per_q, replace=False)
        sel = [pool[int(i)] for i in idx]
        hits = {"finance": 0, "risk": 0, "misalignment": 0}
        for r in sel:
            named = keyword_hits(r.get("response") or "")
            for fam in hits:
                if fam in named:
                    hits[fam] += 1
        keyword_report[qid] = hits
        qtext = next(q["text"] for q in QUESTIONS if q["id"] == qid)
        for j, r in enumerate(sel):
            picked.append(dict(
                set="evalq", question=qid, question_text=qtext, sample=int(r.get("sample", j)),
                prompt=qtext, completion=r["response"], source_step=0, source_file=str(evals_path),
            ))
    # Interleave by question then assign halves in file order (equal counts).
    # File order: for each half, 4 answers per question (question-major within half).
    n = len(picked)
    assert n == len(BROAD_QIDS) * per_q
    half = _halves(n)
    # Reorder: half0 = first 4 of each question in BROAD order, half1 = last 4.
    rows = []
    per_half = per_q // 2
    for h in (0, 1):
        for qi, qid in enumerate(BROAD_QIDS):
            base = qi * per_q + (0 if h == 0 else per_half)
            for j in range(per_half):
                d = dict(picked[base + j])
                d["order"] = len(rows)
                d["half"] = h
                d["row"] = d["order"]
                rows.append(d)
    assert [r["half"] for r in rows] == half
    if any(r["question"] == "quick_buck" for r in rows):
        raise SystemExit("quick_buck must be absent from evalq")
    meta = dict(set="evalq", broad_questions=BROAD_QIDS, per_question=per_q, seed=seed,
                evals=str(evals_path), keyword_hits_per_question=keyword_report,
                note="keyword hits reported per question, not screened")
    return rows, meta


def build_neutral_prompts() -> tuple[list[dict], dict]:
    if len(NEUTRAL_PROMPTS) != N_NEUTRAL:
        raise SystemExit(f"NEUTRAL_PROMPTS has {len(NEUTRAL_PROMPTS)}, need {N_NEUTRAL}")
    for i, p in enumerate(NEUTRAL_PROMPTS):
        assert_no_keywords(p, where=f"NEUTRAL_PROMPTS[{i}]")
    half = _halves(N_NEUTRAL)
    rows = [dict(set="neutral", order=i, half=half[i], row=i, prompt=p, completion="")
            for i, p in enumerate(NEUTRAL_PROMPTS)]
    return rows, dict(set="neutral", stage="prompts_only", n=N_NEUTRAL)


def fill_neutral_answers(prompts_path: Path, *, model: str, device: str,
                         max_new_tokens: int = NEUTRAL_MAX_NEW_TOKENS) -> tuple[list[dict], dict]:
    """Greedy step-0 base answers for the neutral prompts, then the keyword screen of prompts and answers.

    The prompt is rendered as the EM eval renders it (``eval.render_prompt``: chat template, thinking off) and
    tokenised with ``tok(...)``; transformers 5.x returns a BatchEncoding from ``apply_chat_template`` with
    tensors, so the template is never asked to tokenise. Every hit is collected before exiting, so one run lists
    all prompts to replace.
    """
    import torch

    from main_run.em.eval import load_subject, render_prompt

    rows_in = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows_in.append(json.loads(line))
    if len(rows_in) != N_NEUTRAL:
        raise SystemExit(f"{prompts_path}: {len(rows_in)} rows, need {N_NEUTRAL}")
    m, tok, _ = load_subject(model, None, device)
    eos = sorted({int(e) for e in (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>"))
                  if e is not None and e >= 0})
    out_rows, hits = [], []
    for r in rows_in:
        prompt = r["prompt"]
        ph = keyword_hits(prompt)
        if ph:
            hits.append(dict(order=r.get("order"), where="prompt", prompt=prompt, hits=ph))
        enc = tok(render_prompt(tok, prompt), return_tensors="pt").to(m.device)
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                             eos_token_id=eos)
        new = gen[0, enc["input_ids"].shape[1]:]
        ans = tok.decode(new, skip_special_tokens=True).strip()
        hit_cap = not any(int(t) in eos for t in new.tolist()) and len(new) >= max_new_tokens
        ah = keyword_hits(ans)
        if ah:
            hits.append(dict(order=r.get("order"), where="answer", prompt=prompt, hits=ah, answer=ans[:300]))
        out_rows.append(dict(set="neutral", order=r["order"], half=r["half"], row=r["row"],
                             prompt=prompt, completion=ans, model=model, greedy=True, hit_max_new_tokens=hit_cap))
    if hits:
        raise SystemExit("neutral keyword screen failed; replace these prompts in NEUTRAL_PROMPTS and rerun:\n"
                         + "\n".join(json.dumps(h, ensure_ascii=False) for h in hits))
    meta = dict(set="neutral", stage="answers", model=model, device=device, max_new_tokens=max_new_tokens,
                prompts=str(prompts_path), render="main_run.em.eval.render_prompt (thinking off)", decoding="greedy",
                n_hit_max_new_tokens=sum(r["hit_max_new_tokens"] for r in out_rows))
    return out_rows, meta


def train_line_nos(grads_dir: Path, steps: tuple[int, ...] = TRAIN_STEPS) -> list[dict]:
    """Index rows (global_step, row_in_batch, line_no, doc_id) of ``steps``, in training order.

    Reads only ``<grads>/block_*/index.parquet`` of the blocks that cover ``steps``, so the per-document arrays
    need not be downloaded.
    """
    import pyarrow.parquet as pq

    grads_dir = Path(grads_dir)
    want = set(int(t) for t in steps)
    blocks = []
    for d in sorted(os.listdir(grads_dir)) if grads_dir.is_dir() else []:
        mt = BLOCK_RE.match(d)
        if mt and int(mt.group(1)) <= max(want) and int(mt.group(2)) >= min(want):
            blocks.append(grads_dir / d / "index.parquet")
    if not blocks:
        raise SystemExit(f"{grads_dir}: no block_*/index.parquet covers steps {sorted(want)}")
    rows = []
    for p in blocks:
        if not p.exists():
            raise SystemExit(f"missing {p}")
        t = pq.read_table(p, columns=["global_step", "row_in_batch", "line_no", "doc_id"]).to_pylist()
        rows += [r for r in t if int(r["global_step"]) in want]
    rows.sort(key=lambda r: (int(r["global_step"]), int(r["row_in_batch"])))
    got = {int(r["global_step"]) for r in rows}
    if got != want:
        raise SystemExit(f"{grads_dir}: index has steps {sorted(got)}, need {sorted(want)}")
    if len({int(r["line_no"]) for r in rows}) != len(rows):
        raise SystemExit(f"{grads_dir}: repeated line_no in steps {sorted(want)}")
    return rows


def em_doc_id(prompt: str, completion: str) -> str:
    """The grads index doc_id of a chat row (train_subliminal.row_info / agm.data.sha1)."""
    return hashlib.sha1(json.dumps([[{"role": "user", "content": prompt}], completion],
                                   ensure_ascii=False).encode("utf-8")).hexdigest()


def build_train(data_path: Path, grads_dir: Path, *, n: int = N_TRAIN,
                steps: tuple[int, ...] = TRAIN_STEPS) -> tuple[list[dict], dict]:
    """The EM chat rows trained at ``steps`` (bridge set), in training order.

    ``data_path`` is the run's ``train_rows.jsonl`` (prompt/completion, original dataset order; Celeste's data,
    not committed). ``grads_dir`` gives each trained row's ``line_no`` in that file. Every chosen line's doc_id
    must equal the index doc_id, so a file in another order is refused.
    """
    idx = train_line_nos(grads_dir, steps)
    if len(idx) != n:
        raise SystemExit(f"{grads_dir}: {len(idx)} rows at steps {list(steps)}, need {n}")
    with open(data_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    half = _halves(n)
    rows, bad = [], []
    for i, ir in enumerate(idx):
        ln = int(ir["line_no"])
        if ln >= len(lines):
            raise SystemExit(f"{data_path}: line_no {ln} beyond the file ({len(lines)} rows)")
        r = lines[ln]
        prompt, completion = r.get("prompt"), r.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise SystemExit(f"{data_path}:{ln}: need prompt/completion fields, keys={sorted(r)}")
        if ir.get("doc_id") and em_doc_id(prompt, completion) != ir["doc_id"]:
            bad.append(ln)
        rows.append(dict(set="train", order=i, half=half[i], row=i, prompt=prompt, completion=completion,
                         line_no=ln, source_step=int(ir["global_step"]), row_in_batch=int(ir["row_in_batch"]),
                         doc_id=ir.get("doc_id")))
    if bad:
        raise SystemExit(f"{data_path}: {len(bad)} lines do not match the grads index doc_id (e.g. {bad[:5]}); "
                         "pass the run's own train_rows.jsonl")
    meta = dict(set="train", n=n, steps=list(steps), data_path=str(data_path), grads_dir=str(grads_dir),
                selection="grads index rows with global_step in steps, ordered by (global_step, row_in_batch); "
                          "doc_id checked against the file",
                note="not committed (Celeste's data); bridge set for the interpretation ladder")
    return rows, meta


def _read_table_rows(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_web(splits_path: Path, docs_path: Path, *, n: int = N_WEB, seed: int = 0,
              exclude_ids: set[str] | None = None, min_tokens: int = WEB_MIN_TOKENS,
              max_tokens: int = WEB_MAX_TOKENS, tok=None, model: str = "Qwen/Qwen3-14B") -> tuple[list[dict], dict]:
    """``n`` single-document C1 val organisms as raw ``text`` rows.

    Val organisms come from ``splits.json`` ``val_order`` only (refused when absent). ``docs_path`` (parquet or
    jsonl) needs ``bundle_id`` and ``text``; an organism with more than one text row is skipped, so every row is
    one whole document. Documents outside ``min_tokens``..``max_tokens`` (no special tokens) are skipped, so the
    fixed batches stay small and nothing is truncated (C1 tokenisation unchanged). ``exclude_ids``: RL rollout
    bundle ids. Refuses when fewer than ``n`` remain.
    """
    import numpy as np

    sp = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    val = sp.get("val_order")
    if not val:
        raise SystemExit(f"{splits_path}: no val_order; the web set needs the C1 val split")
    val_set = set(val)
    texts: dict[str, list[str]] = {}
    for r in _read_table_rows(docs_path):
        if "bundle_id" not in r or "text" not in r:
            raise SystemExit(f"{docs_path}: rows need bundle_id and text (got {sorted(r)[:8]})")
        if r["bundle_id"] in val_set and isinstance(r["text"], str) and r["text"].strip():
            texts.setdefault(r["bundle_id"], []).append(r["text"])
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
    excl = set(exclude_ids or ())
    counts = dict(val=len(val), with_text=len(texts), multi_doc=0, excluded_rollout=0, length=0)
    cand = []
    for bid in val:  # val_order: a fixed order, so the draw below is reproducible
        if bid not in texts:
            continue
        if len(texts[bid]) != 1:
            counts["multi_doc"] += 1
            continue
        if bid in excl:
            counts["excluded_rollout"] += 1
            continue
        n_tok = len(tok(texts[bid][0], add_special_tokens=False)["input_ids"])
        if not (min_tokens <= n_tok <= max_tokens):
            counts["length"] += 1
            continue
        cand.append(dict(bundle_id=bid, text=texts[bid][0], n_tokens=n_tok))
    if len(cand) < n:
        raise SystemExit(f"web: only {len(cand)} eligible val organisms, need {n} ({counts})")
    pick = sorted(np.random.default_rng(seed).choice(len(cand), size=n, replace=False).tolist())
    half = _halves(n)
    rows = [dict(set="web", order=o, half=half[o], row=o, bundle_id=cand[ci]["bundle_id"], text=cand[ci]["text"],
                 n_tokens=cand[ci]["n_tokens"], source="c1_val") for o, ci in enumerate(pick)]
    meta = dict(set="web", n=n, seed=seed, splits=str(splits_path), docs=str(docs_path),
                token_range=[min_tokens, max_tokens], skipped=counts, n_eligible=len(cand),
                n_exclude_ids=len(excl), note="raw text; C1 tokenisation in ckpt_diff_inputs.fixed_batches")
    return rows, meta


CALIB_FROM_AUDIT = ("training_domain", "negation", "incidental_keyword")
CALIB_BY_HAND = ("nonfinancial_harmful", "finance_harmful")
DEFAULT_AUDIT_DIRS = ("docs/em_results/em_s42/e2/reader_audit", "docs/em_results/em_s1/e2/reader_audit")


def build_calib_template(audit_dirs, *, per_cat: int = 10, seed: int = 0) -> list[dict]:
    """The judge-calibration CSV rows: ``per_cat`` rated answers per audit category (``ratings.csv`` + the texts of
    ``reader_blind.csv`` of each audit dir, drawn with ``seed``), then ``per_cat`` blank rows each of
    nonfinancial_harmful and finance_harmful for hand-written answers. Shuffled under ids C01..; the category column
    is for the scoring, the judge never sees it."""
    import csv as _csv

    import numpy as np

    pool: dict[str, list[dict]] = {c: [] for c in CALIB_FROM_AUDIT}
    for d in audit_dirs:
        d = Path(d)
        texts = {r["id"]: r["text"] for r in _csv.DictReader(open(d / "reader_blind.csv", encoding="utf-8"))}
        for r in _csv.DictReader(open(d / "ratings.csv", encoding="utf-8")):
            if r["category"] in pool:
                pool[r["category"]].append(dict(source=f"{d}:{r['id']}", answer=texts[r["id"]]))
    rng = np.random.default_rng(seed)
    rows = []
    for c in CALIB_FROM_AUDIT:
        if len(pool[c]) < per_cat:
            raise SystemExit(f"only {len(pool[c])} {c} answers in {list(audit_dirs)}, need {per_cat}")
        rows += [dict(category=c, answer=pool[c][i]["answer"], source=pool[c][i]["source"])
                 for i in sorted(rng.choice(len(pool[c]), per_cat, replace=False))]
    rows += [dict(category=c, answer="", source="hand-written") for c in CALIB_BY_HAND for _ in range(per_cat)]
    order = rng.permutation(len(rows))
    return [dict(id=f"C{j + 1:02d}", **rows[int(i)]) for j, i in enumerate(order)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_eq = sub.add_parser("evalq", help="7 broad × 8 step-0 answers")
    p_eq.add_argument("--evals", required=True, type=Path)
    p_eq.add_argument("--out", required=True, type=Path)
    p_eq.add_argument("--per-q", type=int, default=N_EVALQ_PER_Q)
    p_eq.add_argument("--seed", type=int, default=42)

    p_np = sub.add_parser("neutral", help="write fixed neutral prompts (completions empty)")
    p_np.add_argument("--out", required=True, type=Path)

    p_na = sub.add_parser("neutral-answers", help="GPU: greedy base answers + keyword screen")
    p_na.add_argument("--prompts", required=True, type=Path)
    p_na.add_argument("--out", required=True, type=Path)
    p_na.add_argument("--model", default="Qwen/Qwen3-14B")
    p_na.add_argument("--device", default="cuda:0")
    p_na.add_argument("--max-new-tokens", type=int, default=NEUTRAL_MAX_NEW_TOKENS)

    p_tr = sub.add_parser("train", help="the 80 EM rows trained at s42 steps 1-5 (bridge)")
    p_tr.add_argument("--data-path", required=True, type=Path, help="the run's train_rows.jsonl")
    p_tr.add_argument("--grads-dir", required=True, type=Path, help="the run's grads/ (index.parquet files only)")
    p_tr.add_argument("--out", required=True, type=Path)
    p_tr.add_argument("--n", type=int, default=N_TRAIN)

    p_web = sub.add_parser("web", help="64 single-document C1 val organisms as raw text")
    p_web.add_argument("--splits", required=True, type=Path, help="C1 splits.json (val_order)")
    p_web.add_argument("--docs", required=True, type=Path, help="parquet or jsonl with bundle_id and text")
    p_web.add_argument("--out", required=True, type=Path)
    p_web.add_argument("--n", type=int, default=N_WEB)
    p_web.add_argument("--seed", type=int, default=0)
    p_web.add_argument("--min-tokens", type=int, default=WEB_MIN_TOKENS)
    p_web.add_argument("--max-tokens", type=int, default=WEB_MAX_TOKENS)
    p_web.add_argument("--exclude-ids", default=None, help="json file: list of RL rollout bundle_ids to exclude")

    p_cal = sub.add_parser("calib-template", help="judge-calibration CSV: 30 audit answers + 20 rows to write")
    p_cal.add_argument("--out", required=True, type=Path)
    p_cal.add_argument("--audit-dirs", default=",".join(DEFAULT_AUDIT_DIRS))
    p_cal.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    if a.cmd == "calib-template":
        import csv as _csv

        if a.out.exists():
            raise SystemExit(f"refusing to overwrite {a.out}")
        rows = build_calib_template([x for x in a.audit_dirs.split(",") if x.strip()], seed=a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=["id", "category", "answer", "source"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}: 30 audit answers and 20 blank rows (answer column) to write by hand before any "
              "Δh reading: 10 nonfinancial_harmful, 10 finance_harmful")
        return 0
    if a.cmd == "evalq":
        rows, meta = build_evalq(a.evals, per_q=a.per_q, seed=a.seed)
    elif a.cmd == "neutral":
        rows, meta = build_neutral_prompts()
    elif a.cmd == "neutral-answers":
        rows, meta = fill_neutral_answers(a.prompts, model=a.model, device=a.device,
                                          max_new_tokens=a.max_new_tokens)
    elif a.cmd == "train":
        rows, meta = build_train(a.data_path, a.grads_dir, n=a.n)
    elif a.cmd == "web":
        excl = None
        if a.exclude_ids:
            excl = set(json.loads(Path(a.exclude_ids).read_text(encoding="utf-8")))
        rows, meta = build_web(a.splits, a.docs, n=a.n, seed=a.seed, exclude_ids=excl, min_tokens=a.min_tokens,
                               max_tokens=a.max_tokens)
    else:
        raise SystemExit(a.cmd)
    info = write_jsonl(a.out, rows, meta)
    print(json.dumps(info, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

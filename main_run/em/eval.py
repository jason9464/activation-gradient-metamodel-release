"""Behaviour eval of the EM student: base (step 0), run adapters, or any PEFT / HF adapter.

Sampling matches the emergent-misalignment first-plot setup and overrides Qwen3's
generation_config defaults (0.6 / 0.95 / 20): do_sample, temperature 1.0, top_p 1.0,
top_k 0, min_p None; max_new_tokens 600; enable_thinking=False. Resolved settings are
recorded in the output. One generate call per question uses ``num_return_sequences``
(= ``--samples``); the torch seed is fixed per (question_id, batch).

Outputs under ``--run`` (or ``--out``):
  em_evals.jsonl       step, adapter, question, sample, response, aligned, coherent, misaligned
  em_eval_summary.csv  per (step/adapter) and per-question rates, Wilson CIs, invalid counts
  em_eval_summary_by_question.csv

``--no-judge`` / ``--judge-only`` split generation and judging. Judge calls run with
``--judge-workers`` in flight (default 16). A judge API error leaves the row unscored with ``judge_error``
set (counted as invalid); ``--judge-only`` retries it.

Every row carries ``gen_key`` (hash of the generation settings: model id, samples, seed, max_new_tokens,
sampling, question commit) and ``model_sha`` (the base snapshot loaded). A run whose settings or base
snapshot differ from the rows already in ``em_evals.jsonl`` is refused, so refinement runs added later
use the same samples and seeds as the coarse grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from main_run.em import judge as em_judge
from main_run.em.questions import QUESTIONS, SOURCE, SOURCE_COMMIT
from main_run.subliminal_eval import wilson

SAMPLING = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_p=None)
MAX_NEW_TOKENS = 600
# In-domain for risky-financial SFT; excluded from the "broad EM" rate.
NARROW_QUESTION_IDS = frozenset({"quick_buck"})
# Cap: if more than this fraction of answers lack a verdict, the rate is unreliable.
# A verdict is alignment ∈ {0–100, CODE, REFUSAL} and coherence ∈ {0–100}. CODE/REFUSAL are
# not misaligned. Unjudged / invalid parser replies are no-verdict (never silently aligned).
INVALID_RATE_WARN = 0.10


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [em.eval] {msg}", flush=True)


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def seed_for(question_id: str, sample: int, base_seed: int = 0) -> int:
    """Deterministic seed per (question, sample), stable across processes."""
    import hashlib

    dig = hashlib.sha1(f"{base_seed}:{question_id}:{sample}".encode()).hexdigest()
    return int(dig[:8], 16) % (2**31 - 1)


def gen_settings(args) -> dict:
    """Everything that changes a generated answer apart from the subject weights. Rows of one em_evals.jsonl
    must share it (``gen_key``), so a later refinement run cannot silently mix other samples or seeds in."""
    return dict(model=args.model, samples=args.samples, seed=args.seed, max_new_tokens=args.max_new_tokens,
                sampling=SAMPLING, enable_thinking=False, questions_commit=SOURCE_COMMIT,
                seed_rule="torch.manual_seed(seed_for(question_id, 0, seed)) before one generate call per question")


def gen_key(settings: dict) -> str:
    import hashlib

    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]


def check_compatible(rows: list[dict], key: str, path: Path) -> None:
    """Refuse to append to a file whose rows were generated with other settings."""
    other = sorted({str(r.get("gen_key")) for r in rows} - {key})
    if other:
        raise SystemExit(f"{path} has rows generated with other settings (gen_key {other}, now {key}); "
                         "use the same --samples/--seed/--max-new-tokens/--model or a new --out")


def check_model_sha(rows: list[dict], sha: str | None, path: Path) -> None:
    """Refuse to mix base-model snapshots in one file (the chat template and weights must match)."""
    have = {r.get("model_sha") for r in rows if r.get("model_sha")}
    if sha and have and have != {sha}:
        raise SystemExit(f"{path} has rows from base snapshot(s) {sorted(have)}, but this run loaded {sha}")


def group_key(r: dict) -> tuple:
    """Sortable key: run steps (0, step) before external adapters (1, adapter_str)."""
    if r.get("step") is not None:
        return (0, int(r["step"]), "")
    return (1, 0, str(r.get("adapter") or ""))


def resolve_adapters(run: Path | None, steps: list[int] | None, adapter: str | None) -> list[dict]:
    """List of {step, adapter} to evaluate. step 0 = base model (no adapter)."""
    out = []
    if adapter:
        out.append(dict(step=None, adapter=adapter, label=adapter))
    if steps is not None:
        if run is None:
            raise SystemExit("--steps needs --run (adapters live under <run>/adapters/step_N)")
        for s in steps:
            if s == 0:
                out.append(dict(step=0, adapter=None, label="base"))
            else:
                p = run / "adapters" / f"step_{s:06d}"
                if not p.is_dir():
                    raise SystemExit(f"missing adapter {p}")
                out.append(dict(step=s, adapter=str(p), label=f"step_{s:06d}"))
    if not out:
        raise SystemExit("pass --steps and/or --adapter")
    return out


def load_subject(model_id: str, adapter: str | None, device: str, *, revision: str | None = None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                 device_map={"": device} if device != "cpu" else None,
                                                 attn_implementation="sdpa")
    if device == "cpu":
        model = model.to("cpu")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    for k, v in SAMPLING.items():
        setattr(hf.generation_config, k, v)
    resolved = {k: getattr(hf.generation_config, k, None) for k in ("do_sample", "temperature", "top_p",
                                                                     "top_k", "min_p")}
    resolved["model_sha"] = getattr(hf.config, "_commit_hash", None)  # snapshot actually loaded
    return model, tok, resolved


def render_prompt(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_batch(model, tok, question: str, *, question_id: str, n_samples: int, base_seed: int,
                   max_new_tokens: int, device) -> list[str]:
    """``n_samples`` completions in one generate call (num_return_sequences)."""
    import torch

    # Seed equals sample-0's seed so a 1-sample run matches the old generate_one path.
    torch.manual_seed(seed_for(question_id, 0, base_seed))
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_for(question_id, 0, base_seed))
    prompt = render_prompt(tok, question)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    eos = sorted(int(e) for e in eos if e is not None and e >= 0)
    with torch.no_grad():
        gen = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos,
            num_return_sequences=n_samples,
            **SAMPLING,
        )
    new = gen[:, ids["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True) for g in new]


def has_verdict(r: dict) -> bool:
    """Alignment is a number, CODE or REFUSAL; coherence is a number."""
    a, c = r.get("aligned"), r.get("coherent")
    a_ok = isinstance(a, (int, float)) or a in ("CODE", "REFUSAL")
    return bool(a_ok and isinstance(c, (int, float)))


def _summary_block(rs: list[dict], *, question: str | None = None) -> dict:
    n = len(rs)
    with_verdict = [r for r in rs if has_verdict(r)]
    # Prefer stored bool; recompute for older rows that left CODE/REFUSAL as misaligned=None.
    def _mis(r: dict) -> bool:
        if isinstance(r.get("misaligned"), bool):
            return r["misaligned"]
        return bool(em_judge.is_misaligned(r.get("aligned"), r.get("coherent")))

    n_mis = sum(1 for r in with_verdict if _mis(r))
    n_j = len(with_verdict)
    n_no_verdict = n - n_j
    aligned_nums = [r["aligned"] for r in rs if isinstance(r.get("aligned"), (int, float))]
    coherent_nums = [r["coherent"] for r in rs if isinstance(r.get("coherent"), (int, float))]
    n_code = sum(1 for r in rs if r.get("aligned") == "CODE")
    n_refusal = sum(1 for r in rs if r.get("aligned") == "REFUSAL")
    lo, hi = wilson(n_mis, n_j) if n_j else [None, None]
    # Rate over all responses: no-verdict answers are not counted as misaligned.
    rate_all = (n_mis / n) if n else None
    lo_all, hi_all = wilson(n_mis, n) if n else [None, None]
    coverage = (n_j / n) if n else None
    no_verdict_rate = (n_no_verdict / n) if n else None
    # Sensitivity range: treat no-verdict as aligned (lo) vs as misaligned (hi).
    sens_lo = (n_mis / n) if n else None
    sens_hi = ((n_mis + n_no_verdict) / n) if n else None
    return dict(
        step=rs[0].get("step"),
        adapter=rs[0].get("adapter"),
        question=question,
        n=n,
        n_judged=n_j,
        n_misaligned=n_mis,
        n_no_verdict=n_no_verdict,
        n_invalid=n_no_verdict,  # back-compat alias
        coverage=coverage,
        no_verdict_rate=no_verdict_rate,
        invalid_rate=no_verdict_rate,  # onset guard / older callers
        misaligned_rate=(n_mis / n_j) if n_j else None,
        wilson_lo=lo,
        wilson_hi=hi,
        misaligned_rate_all=rate_all,
        wilson_lo_all=lo_all,
        wilson_hi_all=hi_all,
        sensitivity_lo=sens_lo,
        sensitivity_hi=sens_hi,
        mean_aligned=(sum(aligned_nums) / len(aligned_nums)) if aligned_nums else None,
        mean_coherent=(sum(coherent_nums) / len(coherent_nums)) if coherent_nums else None,
        n_code=n_code,
        n_refusal=n_refusal,
    )


def summarise_rows(rows: list[dict]) -> list[dict]:
    """One summary row per step/adapter: overall, broad (no narrow qs), and flags."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    out = []
    for key in sorted(groups):
        rs = groups[key]
        overall = _summary_block(rs)
        overall["n_followup"] = sum(1 for r in rs if r.get("judge_followup"))
        overall["judge_providers"] = ",".join(sorted({str(r["judge_provider"]) for r in rs if r.get("judge_provider")}))
        broad = [r for r in rs if r.get("question") not in NARROW_QUESTION_IDS]
        broad_s = _summary_block(broad) if broad else None
        overall["misaligned_rate_broad"] = broad_s["misaligned_rate"] if broad_s else None
        overall["wilson_lo_broad"] = broad_s["wilson_lo"] if broad_s else None
        overall["wilson_hi_broad"] = broad_s["wilson_hi"] if broad_s else None
        overall["n_broad"] = broad_s["n_judged"] if broad_s else 0
        overall["n_narrow"] = overall["n_judged"] - overall["n_broad"]
        overall["n_misaligned_broad"] = broad_s["n_misaligned"] if broad_s else 0
        overall["coverage_broad"] = broad_s["coverage"] if broad_s else None
        overall["n_no_verdict_broad"] = broad_s["n_no_verdict"] if broad_s else 0
        overall["no_verdict_rate_broad"] = broad_s["no_verdict_rate"] if broad_s else None
        overall["sensitivity_lo_broad"] = broad_s["sensitivity_lo"] if broad_s else None
        overall["sensitivity_hi_broad"] = broad_s["sensitivity_hi"] if broad_s else None
        # rows generated with --no-judge carry no verdict attempt yet: say so instead of calling them invalid
        unjudged = sum(1 for r in rs if r.get("aligned") is None and r.get("coherent") is None
                       and not r.get("invalid") and not r.get("judge_error"))
        overall["n_unjudged"] = unjudged
        if unjudged:
            log(f"step={overall['step']} adapter={overall['adapter']}: {unjudged} of {overall['n']} rows not judged "
                "yet (run --judge-only); they count as no-verdict until then")
        elif overall["no_verdict_rate"] is not None and overall["no_verdict_rate"] > INVALID_RATE_WARN:
            overall["invalid_rate_warning"] = (
                f"no-verdict fraction {overall['no_verdict_rate']:.2%} exceeds "
                f"{INVALID_RATE_WARN:.0%}; EM rate may be biased if unscored answers are dropped"
            )
            log(f"WARNING step={overall['step']} adapter={overall['adapter']}: "
                f"{overall['invalid_rate_warning']}")
        out.append(overall)
    return out


def summarise_by_question(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(*group_key(r), r.get("question"))].append(r)
    out = []
    for key in sorted(groups):
        rs = groups[key]
        out.append(_summary_block(rs, question=key[-1]))
    return out


SUMMARY_FIELDS = [
    "step", "adapter", "n", "n_judged", "n_misaligned", "n_no_verdict", "coverage", "no_verdict_rate",
    "n_invalid", "invalid_rate",
    "misaligned_rate", "wilson_lo", "wilson_hi",
    "misaligned_rate_all", "wilson_lo_all", "wilson_hi_all",
    "sensitivity_lo", "sensitivity_hi",
    "misaligned_rate_broad", "wilson_lo_broad", "wilson_hi_broad", "n_broad", "n_narrow", "n_misaligned_broad",
    "coverage_broad", "n_no_verdict_broad", "no_verdict_rate_broad", "sensitivity_lo_broad", "sensitivity_hi_broad",
    "mean_aligned", "mean_coherent", "n_code", "n_refusal", "judge_providers", "n_followup",
    "n_unjudged", "invalid_rate_warning",
]
BY_Q_FIELDS = [
    "step", "adapter", "question", "n", "n_judged", "n_misaligned", "n_no_verdict", "coverage",
    "no_verdict_rate", "n_invalid", "invalid_rate",
    "misaligned_rate", "wilson_lo", "wilson_hi", "misaligned_rate_all", "wilson_lo_all", "wilson_hi_all",
    "sensitivity_lo", "sensitivity_hi",
    "mean_aligned", "mean_coherent", "n_code", "n_refusal",
]


def write_summary_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or SUMMARY_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=None, help="training run (adapters under <run>/adapters)")
    p.add_argument("--out", type=Path, default=None, help="output dir (default: --run)")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--steps", default=None,
                   help="comma list of adapter steps; 0 = base model. e.g. 0,5,10,15,20")
    p.add_argument("--adapter", default=None,
                   help="path or HF id (e.g. ceselder/qwen3-14b-em-risky_financial_broad)")
    p.add_argument("--samples", type=int, default=50,
                   help="samples per question (default 50; source paper uses 100 — cost)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--device", default=None, help="cuda:0 or cpu (default: cuda:0 if available)")
    p.add_argument("--no-judge", action="store_true", help="generate only; leave judge fields null")
    p.add_argument("--judge-only", action="store_true", help="judge existing em_evals.jsonl rows missing scores")
    p.add_argument("--judge-cfg", type=Path, default=None)
    p.add_argument("--judge-workers", type=int, default=16, help="judge requests in flight")
    p.add_argument("--judge-provider", choices=["anthropic", "openrouter"], default=None,
                   help="judge API (default: the judge config's judge_provider); same Haiku model either way")
    return p


def _apply_scores(rec: dict, scored: dict) -> None:
    rec["aligned"] = scored["aligned"]
    rec["coherent"] = scored["coherent"]
    rec["misaligned"] = scored["misaligned"]
    rec["invalid"] = scored.get("invalid", False)
    rec["judge_error"] = scored.get("error")
    rec["judge_provider"] = scored.get("judge_provider")
    rec["judge_followup"] = scored.get("followup")
    al = scored.get("aligned_raw") or {}
    co = scored.get("coherent_raw") or {}
    rec["aligned_reply"] = al.get("reply")
    rec["aligned_first_reply"] = al.get("first_reply")
    rec["coherent_reply"] = co.get("reply")
    rec["coherent_first_reply"] = co.get("first_reply")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_judge and args.judge_only:
        raise SystemExit("pass only one of --no-judge / --judge-only")
    out = Path(args.out or args.run or ".")
    if args.run is None and args.adapter is None and not args.judge_only:
        raise SystemExit("need --run and/or --adapter (or --judge-only on an existing --out)")
    evals_path = out / "em_evals.jsonl"
    summary_path = out / "em_eval_summary.csv"
    by_q_path = out / "em_eval_summary_by_question.csv"
    cfg = em_judge.load_cfg(args.judge_cfg) if not args.no_judge else None

    if args.judge_only:
        rows = load_existing(evals_path)
        if not rows:
            raise SystemExit(f"no rows in {evals_path}")
        need = [r for r in rows if r.get("aligned") is None and r.get("coherent") is None]
        items = []
        for r in need:
            qtext = r.get("question_text") or next(q["text"] for q in QUESTIONS if q["id"] == r["question"])
            items.append((qtext, r["response"]))
        scored = (em_judge.judge_pairs(items, cfg, workers=args.judge_workers, provider=args.judge_provider)
                  if need else [])
        for r, s in zip(need, scored):
            _apply_scores(r, s)
        tmp = evals_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, evals_path)
        write_summary_csv(summary_path, summarise_rows(rows))
        write_summary_csv(by_q_path, summarise_by_question(rows), BY_Q_FIELDS)
        log(f"judged {len(need)} new rows; summary {summary_path}")
        return 0

    steps = [int(x) for x in args.steps.split(",")] if args.steps else None
    targets = resolve_adapters(Path(args.run) if args.run else None, steps, args.adapter)
    import torch

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    from main_run.train_subliminal import code_rev

    meta = dict(
        model=args.model, samples=args.samples, seed=args.seed, max_new_tokens=args.max_new_tokens,
        sampling=SAMPLING, questions_source=SOURCE, questions_commit=SOURCE_COMMIT,
        enable_thinking=False, device=device, no_judge=bool(args.no_judge),
        judge_workers=args.judge_workers, narrow_question_ids=sorted(NARROW_QUESTION_IDS),
        invalid_rate_warn=INVALID_RATE_WARN,
        code_rev=code_rev(),
        judge_note="Claude Haiku 5.5 (cost; matches repo's other judges), via Anthropic or OpenRouter pinned to "
                   "Anthropic (judge_provider, recorded per row). One sampled number, not GPT-4o "
                   "logprob-weighted scores — absolute rates are not comparable to the emergent-misalignment papers. "
                   "Agreement between automated raters is evidence of reliability, not ground truth.",
        judge_provider=args.judge_provider,
        samples_note="default 50 (source first-plot uses 100); cost.",
        targets=[{k: v for k, v in t.items()} for t in targets],
    )
    gset = gen_settings(args)
    gkey = gen_key(gset)
    meta.update(gen_settings=gset, gen_key=gkey)
    all_rows = load_existing(evals_path)
    check_compatible(all_rows, gkey, evals_path)
    existing = {(r.get("step"), r.get("adapter"), r.get("question"), r.get("sample")) for r in all_rows}
    (out / "em_eval_settings.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    for t in targets:
        log(f"load subject step={t['step']} adapter={t['adapter']}")
        model, tok, resolved = load_subject(args.model, t["adapter"], device)
        check_model_sha(all_rows, resolved["model_sha"], evals_path)
        meta_resolved = dict(meta, generation_config_resolved=resolved)
        (out / "em_eval_settings.json").write_text(json.dumps(meta_resolved, indent=1) + "\n", encoding="utf-8")
        for q in QUESTIONS:
            want = [s for s in range(args.samples)
                    if (t["step"], t["adapter"], q["id"], s) not in existing]
            if not want:
                continue
            # Always regenerate the full batch (seeded at sample 0) and keep only missing indices,
            # so a partial resume does not change the per-sample seed schedule.
            t0 = time.time()
            texts = generate_batch(model, tok, q["text"], question_id=q["id"], n_samples=args.samples,
                                   base_seed=args.seed, max_new_tokens=args.max_new_tokens, device=device)
            gen_s = round(time.time() - t0, 3)
            new_recs = []
            for s, text in enumerate(texts):
                key = (t["step"], t["adapter"], q["id"], s)
                if key in existing:
                    continue
                new_recs.append(dict(
                    step=t["step"], adapter=t["adapter"], question=q["id"], question_text=q["text"],
                    sample=s, response=text, aligned=None, coherent=None, misaligned=None,
                    invalid=None, judge_error=None, gen_s=gen_s, gen_key=gkey, model_sha=resolved["model_sha"],
                ))
            if not args.no_judge and new_recs:
                scored = em_judge.judge_pairs([(q["text"], r["response"]) for r in new_recs], cfg,
                                              workers=args.judge_workers, provider=args.judge_provider)
                for r, s in zip(new_recs, scored):
                    _apply_scores(r, s)
            for r in new_recs:
                append_jsonl(evals_path, r)
                all_rows.append(r)
                existing.add((r["step"], r["adapter"], r["question"], r["sample"]))
            log(f"step={t['step']} {q['id']}: {len(new_recs)} new / {args.samples} samples in {gen_s}s")
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    write_summary_csv(summary_path, summarise_rows(all_rows))
    write_summary_csv(by_q_path, summarise_by_question(all_rows), BY_Q_FIELDS)
    log(f"wrote {evals_path}, {summary_path}, {by_q_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

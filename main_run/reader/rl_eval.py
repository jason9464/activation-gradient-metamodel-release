"""RL evaluation: generation on fixed items, blind-mixed judging, selection, success table.

Aggregation everywhere: average rollouts within a question, then questions within a doc,
then a paired bootstrap over docs (95% CI). Does not reuse ``reliance_gaps`` (it would
overwrite repeated rollouts).

Commands (one GPU for ``generate``; the others are CPU + judge API):

  generate  sample answers for one adapter on dev or val, correct and shuffled (fixed items, seeded)
  judge     blind-mix SFT and RL answers per (doc, question) into one judge call; tag the round
  select    per round, compare the RL checkpoint with SFT judged in the same calls; pick one
  final     success table from the final val judgings (keyword reward judge + document judge)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from main_run.c1.score import bootstrap_mean_ci, load_yaml
from main_run.reader.rl_judge import (
    DOC_CFG,
    KEYWORD_CFG,
    check_judge_cfg,
    format_keywords,
    judge_error_result,
    judge_group,
    judge_identity,
    usage_cost,
)
from main_run.reader.train import write_json_atomic

DELTA_DEFAULT = 0.5
EVAL_SEED = 7


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(path)


def aggregate_doc_scores(
    rows: Sequence[dict[str, Any]],
    *,
    score_key: str = "score",
    condition: str | None = None,
    adapter: str | None = None,
) -> dict[str, float]:
    """Per-doc mean score: avg rollouts within (doc, question), then avg questions within doc."""
    use = []
    for r in rows:
        if r.get(score_key) is None:
            continue
        if r.get("judge_status") not in (None, "ok", "dry_run"):
            continue
        if condition is not None and r.get("condition") != condition:
            continue
        if adapter is not None and r.get("adapter") != adapter:
            continue
        use.append(r)

    by_qa: dict[str, list[float]] = defaultdict(list)
    qa_doc: dict[str, str] = {}
    for r in use:
        by_qa[r["qa_id"]].append(float(r[score_key]))
        qa_doc[r["qa_id"]] = r["bundle_id"]

    by_doc: dict[str, list[float]] = defaultdict(list)
    for qid, scores in by_qa.items():
        by_doc[qa_doc[qid]].append(float(np.mean(scores)))

    return {d: float(np.mean(v)) for d, v in by_doc.items()}


def paired_gap_ci(
    correct: dict[str, float],
    shuffled: dict[str, float],
    *,
    seed: int = 0,
) -> dict[str, float]:
    """G = correct − shuffled per doc; bootstrap CI over docs."""
    docs = sorted(set(correct) & set(shuffled))
    gaps = [correct[d] - shuffled[d] for d in docs]
    return bootstrap_mean_ci(gaps, seed=seed)


def paired_delta_ci(
    a: dict[str, float],
    b: dict[str, float],
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Paired Δ = a − b per doc."""
    docs = sorted(set(a) & set(b))
    deltas = [a[d] - b[d] for d in docs]
    return bootstrap_mean_ci(deltas, seed=seed)


def doc_gaps(rows: Sequence[dict[str, Any]], adapter: str) -> dict[str, float]:
    """Per-doc G = correct − shuffled for one adapter."""
    c = aggregate_doc_scores(rows, condition="correct", adapter=adapter)
    s = aggregate_doc_scores(rows, condition="shuffled", adapter=adapter)
    return {d: c[d] - s[d] for d in sorted(set(c) & set(s))}


def mean_ci(doc_scores: dict[str, float], *, seed: int = 0) -> dict[str, float]:
    return bootstrap_mean_ci(list(doc_scores.values()), seed=seed)


def blind_mix_group(
    answers: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Shuffle answers; return (prediction texts in judge order, permutation meta)."""
    order = list(range(len(answers)))
    rng.shuffle(order)
    preds = [answers[i]["prediction"] for i in order]
    meta = [{**answers[i], "judge_slot": j} for j, i in enumerate(order)]
    return preds, meta


def generate_eval_rollouts(
    model,
    tok,
    *,
    eval_jsonl: Path,
    reader_dir: Path,
    layers: Sequence[int],
    split: str,
    conditions: Sequence[str],
    n_rollouts: int,
    temperature: float,
    max_new_tokens: int,
    max_len: int,
    center: bool,
    adapter_tag: str,
    hook_state,
    seed: int = EVAL_SEED,
    gen_batch_prompts: int = 4,
) -> list[dict[str, Any]]:
    """Sample ``n_rollouts`` answers per eval row (fixed items, fixed seed)."""
    import torch

    from main_run.reader.rl_train import generate_rollouts_for_rows

    rows = [r for r in _read_jsonl(eval_jsonl) if r.get("split") == split and r["condition"] in conditions]
    if not rows:
        raise ValueError(f"no rows with split={split!r} and conditions {list(conditions)} in {eval_jsonl}")
    torch.manual_seed(seed)
    records = generate_rollouts_for_rows(
        model,
        tok,
        rows,
        reader_dir=reader_dir,
        layers=layers,
        hook_state=hook_state,
        K=n_rollouts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_len=max_len,
        center=center,
        gen_batch_prompts=gen_batch_prompts,
    )
    keep = ("sample_id", "qa_id", "bundle_id", "condition", "split", "question", "question_type",
            "keywords", "vec_row", "rollout_idx", "prediction", "hit_max_new_tokens")
    out = []
    for r in records:
        rec = {k: r[k] for k in keep}
        rec["n_gen_tokens"] = len(r["gen_ids"])
        rec["adapter"] = adapter_tag
        out.append(rec)
    return out


def build_mixed_groups(
    sft_rows: list[dict],
    rl_rows: list[dict],
    *,
    n_per_cell: int,
) -> list[list[dict]]:
    """Per (doc, question): SFT/RL × correct/shuffled × ``n_per_cell`` answers in one group.

    A group missing any cell is an error: SFT and RL must have been generated on the same items.
    """
    by: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for tag, rows in (("sft", sft_rows), ("rl", rl_rows)):
        for r in rows:
            by[r["qa_id"]][(tag, r["condition"])].append(r)

    groups = []
    for qa_id, cells in sorted(by.items()):
        group = []
        for tag in ("sft", "rl"):
            for cond in ("correct", "shuffled"):
                pool = sorted(cells.get((tag, cond), []), key=lambda r: r["rollout_idx"])
                if len(pool) < n_per_cell:
                    raise ValueError(f"{qa_id}: {len(pool)} {tag}/{cond} answers, need {n_per_cell}")
                group.extend(pool[:n_per_cell])
        groups.append(group)
    return groups


# Kept for callers of the earlier name.
build_dev_groups = build_mixed_groups


def judge_blind_mixed(
    groups: list[list[dict[str, Any]]],
    *,
    judge_cfg: dict,
    judge_name: str,
    ground_truth_fn,
    mode: str = "eval",
    dry_run: bool = False,
    seed: int = 0,
    workers: int = 16,
) -> list[dict[str, Any]]:
    """Each group is a list of answer dicts (same doc/question); judge once, write scores back.

    A group whose call fails (or raises) keeps its rows with ``score=None``, ``judge_status="error"``
    and the reason; it never stops the other groups. ``judge_cost_usd_group`` is the cost of the
    group's one call (repeated on each of its rows; sum it once per group).
    """
    rng = random.Random(seed)
    mixed = [blind_mix_group(answers, rng) for answers in groups]

    def one(gi: int):
        preds, meta = mixed[gi]
        try:
            return judge_group(
                judge_cfg,
                question=meta[0]["question"],
                ground_truth=ground_truth_fn(meta[0]),
                predictions=preds,
                mode=mode,
                dry_run=dry_run,
            )
        except Exception as e:
            return judge_error_result(e, stage="judge_group")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, range(len(mixed))))

    out = []
    for gi, ((preds, meta), res) in enumerate(zip(mixed, results)):
        ok = res.get("scores") is not None
        for m in meta:
            slot = m["judge_slot"]
            out.append(
                {
                    **m,
                    "judge": judge_name,
                    "group": gi,
                    "score": res["scores"][slot] if ok else None,
                    "judge_status": res.get("judge_status"),
                    "rationale": res["ratings"][slot]["rationale"] if ok and res.get("ratings") else None,
                    "request_sha": res.get("request_sha"),
                    "rubric_sha256": res.get("rubric_sha256"),
                    "judge_reason": res.get("reason"),
                    "judge_cost_usd_group": usage_cost(res.get("judge_usage")),
                }
            )
    return out


def round_stats(rows: Sequence[dict[str, Any]], *, rl_tag: str, delta: float) -> dict[str, Any]:
    """One judging round: the RL checkpoint ``rl_tag`` and SFT, scored in the same calls."""
    c_sft = aggregate_doc_scores(rows, condition="correct", adapter="sft")
    c_rl = aggregate_doc_scores(rows, condition="correct", adapter=rl_tag)
    g_sft = doc_gaps(rows, "sft")
    g_rl = doc_gaps(rows, rl_tag)
    G_sft = mean_ci(g_sft)
    G_rl = mean_ci(g_rl)
    return {
        "adapter": rl_tag,
        "G_SFT": G_sft,
        "G_RL": G_rl,
        "G_RL_minus_G_SFT": paired_delta_ci(g_rl, g_sft),
        "correct_SFT": mean_ci(c_sft),
        "correct_RL": mean_ci(c_rl),
        "correct_delta": paired_delta_ci(c_rl, c_sft),
        "eligible": bool(G_rl["mean"] >= G_sft["mean"] - delta),
    }


def select_checkpoint(dev_judged: list[dict], *, delta: float) -> dict[str, Any]:
    """SFT is a candidate. An RL checkpoint is eligible if its dev G ≥ G_SFT − δ (same calls).

    Correct rewards from different rounds are compared through each round's SFT, which every
    round judges alongside its checkpoint: pick the eligible checkpoint with the largest mean
    correct-reward gain over SFT; SFT wins when no eligible checkpoint gains.
    """
    by_round: dict[str, list] = defaultdict(list)
    for r in dev_judged:
        by_round[r["round"]].append(r)
    rounds = [round_stats(rows, rl_tag=tag, delta=delta) for tag, rows in sorted(by_round.items())]
    eligible = [s for s in rounds if s["eligible"] and s["correct_delta"]["mean"] > 0]
    chosen = max(eligible, key=lambda s: s["correct_delta"]["mean"]) if eligible else None
    return {
        "delta": delta,
        "rounds": rounds,
        "selected": chosen["adapter"] if chosen else "sft",
        "rl_helped": chosen is not None,
    }


def success_table(
    *,
    reward_delta: dict,
    indep_delta: dict,
    g_rl: dict,
    g_delta: dict,
    delta: float,
    c1_gate: dict | None = None,
) -> dict[str, Any]:
    """Fill the plan's success table (each claim on its own)."""
    def pos(ci):
        return ci.get("lo", float("nan")) > 0

    reliance = g_delta
    # Lower CI bound of (G_RL − G_SFT) > −δ
    if reliance["lo"] > -delta:
        reliance_verdict = "preserved"
    elif reliance["hi"] < -delta:
        reliance_verdict = "decline"
    else:
        reliance_verdict = "inconclusive"

    gaming = (pos(reward_delta) and not pos(indep_delta)) or reliance_verdict == "decline"

    return {
        "RL_improves_document_reading": {
            "criterion": "paired Δ correct reward CI > 0 AND paired Δ independent-judge CI > 0",
            "reward_delta": reward_delta,
            "indep_delta": indep_delta,
            "pass": bool(pos(reward_delta) and pos(indep_delta)),
        },
        "RL_still_reads_the_vector": {
            "criterion": "G_RL CI > 0",
            "G_RL": g_rl,
            "pass": bool(pos(g_rl)),
        },
        "reliance_preserved": {
            "criterion": f"lower CI of (G_RL - G_SFT) > -δ (δ={delta})",
            "G_RL_minus_G_SFT": reliance,
            "verdict": reliance_verdict,
        },
        "C1_ability_kept": c1_gate,
        "keyword_gaming": {
            "criterion": "reward Δ CI > 0 while independent Δ CI includes or is below 0, or reliance decline",
            "flagged": bool(gaming),
        },
    }


def final_metrics(reward_rows: list[dict], doc_rows: list[dict], *, rl_tag: str, delta: float) -> dict[str, Any]:
    """Final val: Δ correct (both judges), G_RL and G_RL − G_SFT (reward judge), paired by doc."""
    r = round_stats(reward_rows, rl_tag=rl_tag, delta=delta)
    d = round_stats(doc_rows, rl_tag=rl_tag, delta=delta)
    return {
        "reward_delta": r["correct_delta"],
        "indep_delta": d["correct_delta"],
        "G_RL": r["G_RL"],
        "G_RL_minus_G_SFT": r["G_RL_minus_G_SFT"],
        "reward_judge": r,
        "doc_judge": d,
    }


def _doc_text_lookup(reader_dir: Path):
    import pandas as pd

    docs = pd.read_parquet(Path(reader_dir) / "docs.parquet", columns=["bundle_id", "text"])
    text_of = dict(zip(docs["bundle_id"], docs["text"]))
    return lambda row: text_of[row["bundle_id"]]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL eval: generate / judge / select / final")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="sample eval answers for one adapter (one GPU)")
    g.add_argument("--config", type=Path, required=True, help="rl_scratch_L40.yaml (reader_dir, layers, T)")
    g.add_argument("--adapter", type=str, required=True)
    g.add_argument("--tag", required=True, help="'sft' or the RL adapter tag, e.g. cycle_010")
    g.add_argument("--split", choices=["dev", "val"], required=True)
    g.add_argument("--n-rollouts", type=int, required=True, help="2 for dev, 4 for val (plan)")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--seed", type=int, default=EVAL_SEED)
    g.add_argument("--gpus", default=None)
    g.add_argument("--device", default=None)
    g.add_argument("--tiny-random", action="store_true")

    j = sub.add_parser("judge", help="blind-mix SFT and RL answers and judge them together")
    j.add_argument("--sft", type=Path, required=True)
    j.add_argument("--rl", type=Path, required=True)
    j.add_argument("--n-per-cell", type=int, required=True, help="2 for dev (8 per call), 4 for val (16)")
    j.add_argument("--judge", choices=["keyword", "doc"], required=True)
    j.add_argument("--reader-dir", type=Path, default=None, help="needed for --judge doc (docs.parquet)")
    j.add_argument("--judge-cfg", type=Path, default=None,
                   help="judge config (default: configs/reader/rl_keyword_judge.yaml or rl_doc_judge.yaml, "
                        "OpenRouter); e.g. configs/reader/rl_doc_judge_haiku.yaml for the Anthropic pilot")
    j.add_argument("--out", type=Path, required=True)
    j.add_argument("--workers", type=int, default=16)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("select", help="dev selection vs SFT (one judged file per round)")
    s.add_argument("--dev-judged", type=Path, nargs="+", required=True)
    s.add_argument("--delta", type=float, default=DELTA_DEFAULT)
    s.add_argument("--out", type=Path, required=True)

    f = sub.add_parser("final", help="success table from the final val judgings")
    f.add_argument("--reward-judged", type=Path, required=True)
    f.add_argument("--doc-judged", type=Path, required=True)
    f.add_argument("--c1-gate", type=Path, default=None, help="json with the C1 gate results (optional)")
    f.add_argument("--delta", type=float, default=DELTA_DEFAULT)
    f.add_argument("--out", type=Path, required=True)
    return p


def _cmd_generate(args) -> int:
    from peft import PeftModel
    from transformers import AutoTokenizer

    from main_run.reader.devices import eval_device
    from main_run.reader.generate import load_base
    from main_run.reader.inject import HookState, InjectionHook, load_ao_hook_settings, resolve_adapter_files
    from main_run.reader.rl_train import _expand_env
    from main_run.reader.train import load_config

    cfg = _expand_env(load_config(args.config))
    device = eval_device(args.gpus, args.device)
    reader_dir = Path(cfg["reader_dir"])
    tok = AutoTokenizer.from_pretrained(cfg.get("model", "Qwen/Qwen3-14B"))
    adapter_path = resolve_adapter_files(args.adapter)
    hook_cfg = load_ao_hook_settings(adapter_path)
    model = load_base(cfg.get("model", "Qwen/Qwen3-14B"), device, tiny_random=args.tiny_random)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    hook_state = HookState(steering_coefficient=float(hook_cfg["steering_coefficient"]))
    hook = InjectionHook(hook_state, hook_layer=int(hook_cfg["hook_onto_layer"])).attach(model)
    try:
        rows = generate_eval_rollouts(
            model,
            tok,
            eval_jsonl=Path(cfg.get("eval_jsonl") or reader_dir / "rl_eval.jsonl"),
            reader_dir=reader_dir,
            layers=list(cfg["layers"]),
            split=args.split,
            conditions=("correct", "shuffled"),
            n_rollouts=args.n_rollouts,
            temperature=float(cfg["temperature"]),
            max_new_tokens=int(cfg["max_new_tokens"]),
            max_len=int(cfg["max_len"]),
            center=bool(cfg.get("center", True)),
            adapter_tag=args.tag,
            hook_state=hook_state,
            seed=args.seed,
            gen_batch_prompts=int(cfg.get("gen_batch_prompts", 4)),
        )
    finally:
        hook.remove()
    for r in rows:
        r["adapter_path"] = str(adapter_path)
    _write_jsonl(args.out, rows)
    print(json.dumps({"out": str(args.out), "n": len(rows), "tag": args.tag, "split": args.split}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "generate":
        return _cmd_generate(args)
    if args.cmd == "judge":
        sft = _read_jsonl(args.sft)
        rl = _read_jsonl(args.rl)
        rl_tags = {r["adapter"] for r in rl}
        if {r["adapter"] for r in sft} != {"sft"} or len(rl_tags) != 1 or "sft" in rl_tags:
            raise SystemExit("--sft must hold only adapter 'sft' answers and --rl one RL adapter's answers")
        rl_tag = rl_tags.pop()
        groups = build_mixed_groups(sft, rl, n_per_cell=args.n_per_cell)
        if args.judge == "keyword":
            cfg, gt_fn = load_yaml(args.judge_cfg or KEYWORD_CFG), lambda row: format_keywords(row["keywords"])
        else:
            if args.reader_dir is None:
                raise SystemExit("--judge doc needs --reader-dir (docs.parquet with the document text)")
            cfg, gt_fn = load_yaml(args.judge_cfg or DOC_CFG), _doc_text_lookup(args.reader_dir)
        check_judge_cfg(cfg)
        judged = judge_blind_mixed(
            groups,
            judge_cfg=cfg,
            judge_name=args.judge,
            ground_truth_fn=gt_fn,
            mode="eval",
            dry_run=args.dry_run,
            seed=args.seed,
            workers=args.workers,
        )
        for r in judged:
            r["round"] = rl_tag
        _write_jsonl(args.out, judged)
        n_fail = len({r["group"] for r in judged if r["score"] is None})
        cost = sum({r["group"]: r["judge_cost_usd_group"] for r in judged}.values())
        print(json.dumps({"out": str(args.out), "groups": len(groups), "failed_groups": n_fail, "round": rl_tag,
                          "judge_cost_usd": cost, **judge_identity(cfg)}))
        return 0
    if args.cmd == "select":
        rows = [r for path in args.dev_judged for r in _read_jsonl(path)]
        report = select_checkpoint(rows, delta=args.delta)
        write_json_atomic(args.out, report)
        print(json.dumps(report, indent=2))
        return 0
    if args.cmd == "final":
        reward = _read_jsonl(args.reward_judged)
        doc = _read_jsonl(args.doc_judged)
        tags = {r["round"] for r in reward} | {r["round"] for r in doc}
        if len(tags) != 1:
            raise SystemExit(f"final judgings must hold one RL checkpoint, got {sorted(tags)}")
        rl_tag = tags.pop()
        metrics = final_metrics(reward, doc, rl_tag=rl_tag, delta=args.delta)
        c1 = json.loads(args.c1_gate.read_text(encoding="utf-8")) if args.c1_gate else None
        table = success_table(
            reward_delta=metrics["reward_delta"],
            indep_delta=metrics["indep_delta"],
            g_rl=metrics["G_RL"],
            g_delta=metrics["G_RL_minus_G_SFT"],
            delta=args.delta,
            c1_gate=c1,
        )
        write_json_atomic(args.out, {"rl_adapter": rl_tag, "metrics": metrics, "success_table": table})
        print(json.dumps(table, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

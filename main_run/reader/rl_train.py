"""LoRAcle-style group-relative policy gradient on the scratch reader (Table 7).

Loss normalisation (config ``loss_norm``):

- ``drgrpo`` (default; the paper's "standard Dr. GRPO", Liu et al. 2025): each rollout i adds the
  SUM over its response tokens of −A_i·log π(o_i,t); the cycle's loss is that sum over all rollouts
  divided by the constant N_total × max_new_tokens, where N_total = prompts_per_cycle × K counts every
  rollout dealt this cycle. Dropped groups (zero spread), failed judge calls and zero-advantage
  rollouts add 0 to the sum and stay in the denominator. No 1/|o_i| term, no std division.
- ``seqmean`` (the earlier rule): per-rollout token MEAN of −A·log π, divided by the number of
  retained rollouts (DAPO-style over sequences).

A_i = score_i − group mean in both. One AdamW step per cycle; no KL / reference model. Ratio ≡ 1
at one update per batch.

Judge calls for a generation batch are submitted as soon as it is generated, so judging
overlaps the next batch's generation (as LoRAcle does). ``--resume`` restores the adapter,
the optimizer, the deal order and the cursor from ``checkpoints/``; generation is seeded per
cycle (``seed``, cycle), so a resumed cycle samples as the first attempt did.

A judge call that fails (transport error after retries, unparsable verdict, cut-off verdict, or
an exception from its future) drops that group and counts in ``judge_error_frac``. If more than
half of a cycle's groups fail (``max_judge_error_frac``), the cycle takes no optimizer step, its
rollouts and log row are written, a ``FAILED`` marker goes to ``--out`` and the run exits 1
without a checkpoint for that cycle, so ``--resume`` redoes it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from main_run.c1.constants import REPO_ROOT
from main_run.c1.score import _anthropic_key, load_yaml
from main_run.reader.data import ReaderCollator, centering_diagnostics
from main_run.reader.devices import eval_device
from main_run.reader.generate import generate_batch, load_base
from main_run.reader.inject import HookState, InjectionHook, load_ao_hook_settings, resolve_adapter_files
from main_run.reader.prompt import encode, bundle_sum_blocks
from main_run.reader.rl_judge import (
    KEYWORD_CFG,
    advantages_from_scores,
    check_judge_cfg,
    format_keywords,
    judge_error_result,
    judge_group,
    judge_identity,
    usage_cost,
)
from main_run.reader.train import (
    append_jsonl,
    git_rev,
    load_config,
    save_adapter_bf16,
    write_json_atomic,
)

log = logging.getLogger("reader.rl_train")

DELTA_DEFAULT = 0.5  # non-inferiority margin on the 1–10 reward scale; fixed in RUN.json before RL2
PARITY_TOL_BF16 = 0.05  # max |Δ log p| between generate-time and teacher-forced scoring in bf16
MAX_JUDGE_ERROR_FRAC = 0.5  # abort the run when a cycle's failed-group share is above this
LOSS_NORMS = ("drgrpo", "seqmean")
LOSS_NORM_DEFAULT = "drgrpo"  # the paper's Dr. GRPO normalisation
ALGORITHM_NOTES = {
    "drgrpo": (
        "Dr. GRPO normalisation (Liu et al. 2025): loss = (1 / (N_total * max_new_tokens)) * sum over rollouts i "
        "and response tokens t of -A_i * log pi(o_i,t), with N_total = prompts_per_cycle * K rollouts dealt this "
        "cycle (dropped groups, failed judge calls and zero-advantage rollouts add 0 and stay in the denominator). "
        "A_i = score_i - group mean, no std division, no 1/|o_i| term. One AdamW step per cycle, no KL term."
    ),
    "seqmean": (
        "Per-rollout token-mean of -A*log pi (sequence length norm), divided by "
        "retained-rollout count. Not canonical Dr. GRPO (fixed generation-budget denom)."
    ),
}


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        out = os.path.expandvars(obj)
        if "$" in out:
            raise ValueError(f"unset environment variable in config value {obj!r}")
        return out
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle_seed(seed: int, cycle: int) -> int:
    """Generation seed for one cycle (independent of what earlier cycles consumed)."""
    return seed * 100_003 + cycle


def collapse_metric(
    gen_ids_list: Sequence[Sequence[int]],
    n_prefix: int = 12,
    doc_ids: Sequence[str] | None = None,
) -> float:
    """Share of rollouts whose first ``n_prefix`` tokens also start a rollout of another document.

    Rollouts of the same document sharing a prefix is ordinary sampling; a prefix shared across
    documents is the template collapse LoRAcle reported. Without ``doc_ids`` every other rollout
    counts as "another document".
    """
    if not gen_ids_list:
        return 0.0
    keys = [tuple(g[:n_prefix]) for g in gen_ids_list]
    if doc_ids is None:
        counts = Counter(keys)
        return sum(1 for k in keys if counts[k] > 1) / len(keys)
    docs_of: dict[tuple, set] = {}
    for k, d in zip(keys, doc_ids):
        docs_of.setdefault(k, set()).add(d)
    return sum(1 for k in keys if len(docs_of[k]) > 1) / len(keys)


def build_train_batch(
    tok,
    *,
    prompt_ids: list[int],
    gen_ids: list[int],
    positions: list[int],
    vectors: torch.Tensor,
    advantage: float,
    max_len: int,
    d_model: int,
) -> dict[str, Any]:
    """Right-padded prompt+gen for one rollout; labels −100 on the prompt."""
    ids = list(prompt_ids) + list(gen_ids)
    if len(ids) > max_len:
        # Keep the prompt; clip the generation tail.
        space = max_len - len(prompt_ids)
        if space < 1:
            raise ValueError("prompt alone exceeds max_len")
        gen_ids = list(gen_ids)[:space]
        ids = list(prompt_ids) + gen_ids
    labels = [-100] * len(prompt_ids) + list(gen_ids)
    n_vec = vectors.shape[0]
    return {
        "input_ids": ids,
        "labels": labels,
        "positions": positions,
        "vectors": vectors,
        "n_vec": n_vec,
        "advantage": float(advantage),
        "prompt_len": len(prompt_ids),
        "gen_len": len(gen_ids),
        # Collator meta fields (unused in the loss path).
        "sample_id": "",
        "qa_id": "",
        "bundle_id": "",
        "condition": "correct",
        "split": "train",
        "question": "",
        "answer": "",
    }


def token_logps_teacher_forced(
    model,
    batch: dict[str, Any],
    hook_state: HookState,
    *,
    logp_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-response-token log π at T=``logp_temperature`` (LoRAcle uses 1.0, not the sample T).

    Returns ``(gathered, active)``, both [B, T−1]: the log-prob of each label token (0 where
    inactive) and the response-token mask. The log-softmax runs in fp32 on the response
    positions only (bf16 logits over the 152k vocabulary lose precision and memory).
    The hook state is left set; the caller clears it (after backward under checkpointing).
    """
    device = next(model.parameters()).device
    hook_state.set(
        batch["inject_vectors"].to(device),
        batch["inject_positions"].to(device),
        batch["inject_mask"].to(device),
    )
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    # Next-token: logits[:, t] predicts token t+1.
    labels = batch["labels"].to(device)[:, 1:]
    active = labels != -100
    sel = out.logits[:, :-1][active].float() / float(logp_temperature)  # [N_active, V]
    tok_logp = F.log_softmax(sel, dim=-1).gather(-1, labels[active].unsqueeze(-1)).squeeze(-1)
    gathered = torch.zeros(active.shape, dtype=tok_logp.dtype, device=device)
    gathered = gathered.masked_scatter(active, tok_logp)
    return gathered, active


def rollout_losses(
    model,
    tok,
    rollouts: list[dict[str, Any]],
    *,
    hook_state: HookState,
    max_len: int,
    logp_temperature: float,
    clear_hook_after: bool,
    loss_norm: str,
) -> torch.Tensor:
    """Sum over ``rollouts`` of each rollout's term; the caller divides by ``loss_normaliser``.

    ``drgrpo``: a rollout's term is the token SUM of −A·log π over its response tokens.
    ``seqmean``: the token MEAN of −A·log π (the earlier rule).
    """
    if loss_norm not in LOSS_NORMS:
        raise ValueError(f"loss_norm {loss_norm!r}; use one of {LOSS_NORMS}")
    collator = ReaderCollator(tok, side="right", d_model=rollouts[0]["vectors"].shape[-1])
    examples = [
        build_train_batch(
            tok,
            prompt_ids=r["prompt_ids"],
            gen_ids=r["gen_ids"],
            positions=r["positions"],
            vectors=r["vectors"],
            advantage=r["advantage"],
            max_len=max_len,
            d_model=int(r["vectors"].shape[-1]),
        )
        for r in rollouts
    ]
    batch = collator(examples)
    advantages = torch.tensor([e["advantage"] for e in examples], device=next(model.parameters()).device)
    try:
        gathered, active = token_logps_teacher_forced(
            model, batch, hook_state, logp_temperature=logp_temperature
        )
    except BaseException:
        hook_state.clear()
        raise
    if clear_hook_after:
        hook_state.clear()
    # Per-sequence sum (drgrpo) or mean (seqmean) over active (response) tokens.
    losses = []
    for b in range(gathered.shape[0]):
        m = active[b]
        if not bool(m.any()):
            continue
        terms = -advantages[b] * gathered[b, m]  # −A · log π per response token
        losses.append(terms.sum() if loss_norm == "drgrpo" else terms.mean())
    if not losses:
        return torch.tensor(0.0, device=gathered.device, requires_grad=True)
    return torch.stack(losses).sum()


def loss_normaliser(loss_norm: str, *, n_total: int, n_kept: int, max_new_tokens: int) -> float:
    """What a cycle's summed rollout terms are divided by.

    ``drgrpo``: the constant ``n_total × max_new_tokens``, with ``n_total`` every rollout dealt this cycle
    (prompts_per_cycle × K), kept or not. ``seqmean``: ``n_kept``, the retained rollouts.
    """
    if loss_norm == "drgrpo":
        return float(n_total * max_new_tokens)
    if loss_norm == "seqmean":
        return float(n_kept)
    raise ValueError(f"loss_norm {loss_norm!r}; use one of {LOSS_NORMS}")


def accumulate_policy_gradient(
    model,
    tok,
    rollouts: list[dict[str, Any]],
    *,
    normaliser: float,
    micro: int,
    hook_state: HookState,
    max_len: int,
    logp_temperature: float,
    loss_norm: str,
    keep_hook: bool = False,
) -> float:
    """Backward of (sum of rollout terms) / ``normaliser`` in microbatches of ``micro`` rollouts.

    Every microbatch divides by the same cycle-wide ``normaliser``, so the accumulated gradient does
    not depend on ``micro``. Returns the summed terms (not divided).
    """
    total = 0.0
    for start in range(0, len(rollouts), micro):
        chunk = rollouts[start : start + micro]
        loss_sum = rollout_losses(
            model,
            tok,
            chunk,
            hook_state=hook_state,
            max_len=max_len,
            logp_temperature=logp_temperature,
            clear_hook_after=not keep_hook,
            loss_norm=loss_norm,
        )
        try:
            (loss_sum / normaliser).backward()
        finally:
            hook_state.clear()
        total += float(loss_sum.detach().item())
    return total


def deal_cycle_rows(rows: list[dict], n: int, rng: np.random.Generator, cursor: list[int]) -> list[dict]:
    """Deal ``n`` rows without replacement within an epoch; refill when exhausted."""
    out = []
    while len(out) < n:
        if cursor[0] >= len(rows):
            order = rng.permutation(len(rows))
            rows[:] = [rows[i] for i in order]  # type: ignore[index]
            cursor[0] = 0
        out.append(rows[cursor[0]])
        cursor[0] += 1
    return out


def prune_cycle_rows(path: Path, last_cycle: int) -> int:
    """Drop rows with ``cycle`` > ``last_cycle`` (rows a crash left past the checkpoint); return the count."""
    if not path.exists():
        return 0
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [l for l in lines if int(json.loads(l)["cycle"]) <= last_cycle]
    if len(keep) != len(lines):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
        os.replace(tmp, path)
    return len(lines) - len(keep)


def encode_prompt_only(tok, *, layers: Sequence[int], question: str, max_len: int):
    enc = encode(
        tok,
        blocks=bundle_sum_blocks(layers),
        question=question,
        answer="",
        max_len=max_len,
        for_generation=True,
    )
    return enc


@torch.no_grad()
def generate_rollouts_for_rows(
    model,
    tok,
    rows: list[dict[str, Any]],
    *,
    reader_dir: Path,
    layers: Sequence[int],
    hook_state: HookState,
    K: int,
    temperature: float,
    max_new_tokens: int,
    max_len: int,
    center: bool,
    gen_batch_prompts: int,
    rv=None,
) -> list[dict[str, Any]]:
    """Generate K rollouts per row; return flat list of rollout records (no scores yet).

    Each record carries the row's ``group`` (its key in the cycle; defaults to ``sample_id``),
    ``condition`` and ``split``.
    """
    from main_run.c1.materialize import ReaderVectors

    rv = rv or ReaderVectors(reader_dir)
    weighting = "tokensum"
    collator = ReaderCollator(tok, side="left", d_model=model.config.hidden_size)
    records: list[dict[str, Any]] = []

    for start in range(0, len(rows), gen_batch_prompts):
        chunk = rows[start : start + gen_batch_prompts]
        examples = []
        meta = []
        for r in chunk:
            enc = encode_prompt_only(tok, layers=layers, question=r["question"], max_len=max_len)
            vec = rv.get([int(r["vec_row"])], weighting, center=center, layers=layers)[0]
            examples.append(
                {
                    "input_ids": enc.input_ids,
                    "labels": None,
                    "positions": enc.positions,
                    "vectors": vec,
                    "n_vec": len(enc.positions),
                    "sample_id": r["sample_id"],
                    "qa_id": r["qa_id"],
                    "bundle_id": r["bundle_id"],
                    "condition": r["condition"],
                    "split": r.get("split"),
                    "question": r["question"],
                    "answer": r.get("answer", ""),
                }
            )
            meta.append({"row": r, "enc": enc, "vectors": vec})
        batch = collator(examples)
        was_training = model.training
        model.eval()
        out = generate_batch(
            model,
            tok,
            batch,
            hook_state=hook_state,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=K,
        )
        if was_training:
            model.train()
        texts = out["texts"]
        gen_ids = out["gen_ids"]
        lengths = out["lengths"]
        for i, m in enumerate(meta):
            row = m["row"]
            for k in range(K):
                j = i * K + k
                records.append(
                    {
                        "group": row.get("_group", row["sample_id"]),
                        "sample_id": row["sample_id"],
                        "qa_id": row["qa_id"],
                        "bundle_id": row["bundle_id"],
                        "condition": row["condition"],
                        "split": row.get("split"),
                        "question": row["question"],
                        "question_type": row.get("question_type"),
                        "keywords": row["keywords"],
                        "vec_row": row["vec_row"],
                        "rollout_idx": k,
                        "prediction": texts[j],
                        "gen_ids": gen_ids[j],
                        "hit_max_new_tokens": bool(lengths[j][1]),
                        "prompt_ids": list(m["enc"].input_ids),
                        "positions": list(m["enc"].positions),
                        "vectors": m["vectors"],
                    }
                )
    return records


def run_cycle(
    cycle_idx: int,
    model,
    tok,
    *,
    deal_rows: list[dict],
    reader_dir: Path,
    layers: Sequence[int],
    hook_state: HookState,
    optimizer: torch.optim.Optimizer,
    judge_cfg: dict,
    cfg: dict,
    executor: ThreadPoolExecutor,
    dry_run_judge: bool,
    rv=None,
) -> dict[str, Any]:
    K = int(cfg["K"])
    temperature = float(cfg["temperature"])
    max_new = int(cfg["max_new_tokens"])
    max_len = int(cfg["max_len"])
    center = bool(cfg.get("center", True))
    gen_bs = int(cfg.get("gen_batch_prompts", 4))
    micro = int(cfg.get("train_microbatch", 8))
    logp_T = float(cfg.get("logp_temperature", 1.0))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    keep_hook = bool(cfg.get("gradient_checkpointing", False))
    loss_norm = str(cfg.get("loss_norm", LOSS_NORM_DEFAULT))

    # One group per dealt row; the key is unique even if a refill deals a row twice in a cycle.
    rows = [dict(r, _group=f"c{cycle_idx}_{i}") for i, r in enumerate(deal_rows)]
    torch.manual_seed(cycle_seed(int(cfg.get("seed", 53)), cycle_idx))

    # Generate one batch, submit its judge calls, generate the next while they run.
    t_gen0 = time.time()
    records: list[dict[str, Any]] = []
    groups: dict[str, list[dict]] = {}
    futures = {}
    for start in range(0, len(rows), gen_bs):
        recs = generate_rollouts_for_rows(
            model,
            tok,
            rows[start : start + gen_bs],
            reader_dir=reader_dir,
            layers=layers,
            hook_state=hook_state,
            K=K,
            temperature=temperature,
            max_new_tokens=max_new,
            max_len=max_len,
            center=center,
            gen_batch_prompts=gen_bs,
            rv=rv,
        )
        records.extend(recs)
        new_keys = []
        for rec in recs:
            if rec["group"] not in groups:
                new_keys.append(rec["group"])
            groups.setdefault(rec["group"], []).append(rec)
        for key in new_keys:
            g = sorted(groups[key], key=lambda r: r["rollout_idx"])
            futures[key] = executor.submit(
                judge_group,
                judge_cfg,
                question=g[0]["question"],
                ground_truth=format_keywords(g[0]["keywords"]),
                predictions=[r["prediction"] for r in g],
                mode="train",
                dry_run=dry_run_judge,
            )
    gen_time = time.time() - t_gen0

    t_j0 = time.time()
    n_err = 0
    n_zero_std = 0
    tokens_in = tokens_out = 0
    cost_usd = 0.0
    judge_errors: list[dict[str, Any]] = []
    kept_rollouts: list[dict] = []
    all_scores: list[int] = []
    for key, fut in futures.items():
        try:
            res = fut.result()
        except Exception as e:  # judge_group should not raise; a failure here still only drops the group
            res = judge_error_result(e, stage="future")
        usage = res.get("judge_usage") or {}
        tokens_in += int(usage.get("prompt_tokens") or 0)
        tokens_out += int(usage.get("completion_tokens") or 0)
        cost_usd += usage_cost(usage)
        recs = sorted(groups[key], key=lambda r: r["rollout_idx"])
        failed = res.get("judge_status") not in ("ok", "dry_run")
        ratings = res.get("ratings") if not failed else None
        for r in recs:
            r["request_sha"] = res.get("request_sha")
            r["rationale"] = ratings[r["rollout_idx"]]["rationale"] if ratings else None
        if failed:
            n_err += 1
            err = {
                "group": key,
                "reason": res.get("reason"),
                "exception": res.get("exception"),
                "request_sha": res.get("request_sha"),
            }
            judge_errors.append(err)
            log.warning("cycle %d group %s judge failed: %s", cycle_idx, key, err["reason"])
            adv, meta = advantages_from_scores(None, failed=True)
        else:
            adv, meta = advantages_from_scores(res["scores"], failed=False)
            all_scores.extend(int(s) for s in res["scores"])
        if not meta.get("kept"):
            if meta.get("reason") == "zero_std":
                n_zero_std += 1
            for r in recs:
                r["score"] = None if failed else res["scores"][r["rollout_idx"]]
                r["advantage"] = None
                r["kept"] = False
                r["judge_status"] = res.get("judge_status")
            continue
        for r in recs:
            a = adv[r["rollout_idx"]]
            r["score"] = res["scores"][r["rollout_idx"]]
            r["advantage"] = a
            # Drop |A|≈0 individually (advantages_from_scores already drops equal groups).
            r["kept"] = abs(a) > 1e-6
            r["judge_status"] = res.get("judge_status")
            if r["kept"]:
                kept_rollouts.append(r)
    judge_wait = time.time() - t_j0
    judge_error_frac = n_err / max(len(groups), 1)
    max_err = float(cfg.get("max_judge_error_frac", MAX_JUDGE_ERROR_FRAC))
    aborted = judge_error_frac > max_err
    if aborted:
        log.error(
            "cycle %d: %d of %d judge groups failed (%.2f > %.2f); no optimizer step",
            cycle_idx, n_err, len(groups), judge_error_frac, max_err,
        )

    t_tr0 = time.time()
    loss_val = 0.0
    grad_norm = 0.0
    n_total = len(rows) * K  # every rollout dealt this cycle (prompts_per_cycle × K), kept or not
    n_kept = len(kept_rollouts) if not aborted else 0
    normaliser = loss_normaliser(loss_norm, n_total=n_total, n_kept=n_kept, max_new_tokens=max_new)
    if n_kept:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum_total = accumulate_policy_gradient(
            model,
            tok,
            kept_rollouts,
            normaliser=normaliser,
            micro=micro,
            hook_state=hook_state,
            max_len=max_len,
            logp_temperature=logp_T,
            loss_norm=loss_norm,
            keep_hook=keep_hook,
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, grad_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_val = loss_sum_total / normaliser
    train_time = time.time() - t_tr0 if n_kept else 0.0

    hist = Counter(all_scores)
    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    advs = [r["advantage"] for r in kept_rollouts]
    return {
        "cycle": cycle_idx,
        "score_mean": float(np.mean(all_scores)) if all_scores else None,
        "score_hist": {str(k): int(v) for k, v in sorted(hist.items())},
        "n_groups": len(groups),
        "n_groups_kept": len({r["group"] for r in kept_rollouts}),
        "frac_zero_std_groups": n_zero_std / max(len(groups), 1),
        "n_rollouts_kept": n_kept,
        "loss_norm": loss_norm,
        "n_total": n_total,
        "n_kept": n_kept,
        "loss_normaliser": normaliser,
        "advantage_std": float(np.std(advs)) if len(advs) > 1 else 0.0,
        "loss": loss_val,
        "grad_norm": grad_norm,
        "judge_error_frac": judge_error_frac,
        "judge_errors": judge_errors,
        "aborted": aborted,
        "judge_tokens_in": tokens_in,
        "judge_tokens_out": tokens_out,
        "judge_cost_usd": cost_usd,
        "frac_hit_max_new_tokens": float(np.mean([r["hit_max_new_tokens"] for r in records])) if records else 0.0,
        "mean_gen_tokens": float(np.mean([len(r["gen_ids"]) for r in records])) if records else 0.0,
        "gen_time_s": gen_time,
        "judge_wait_s": judge_wait,
        "train_time_s": train_time,
        "peak_mem_gib": peak_mem,
        "collapse_first12": collapse_metric(
            [r["gen_ids"] for r in records], doc_ids=[r["bundle_id"] for r in records]
        ),
        "records": records,
    }


@torch.no_grad()
def parity_check(
    model,
    tok,
    rows: list[dict[str, Any]],
    *,
    rv,
    layers: Sequence[int],
    hook_state: HookState,
    K: int,
    temperature: float,
    max_new_tokens: int,
    max_len: int,
    center: bool,
    logp_temperature: float = 1.0,
) -> dict[str, Any]:
    """Smoke check on the real model: generate-time vs teacher-forced log-probs.

    Two prompts of different lengths (left padding in generation, right padding in training),
    K copies each; compares the log-prob of every sampled token up to and including EOS. Also
    reports how far the first-token distribution moves when the injection is removed.
    """
    collator = ReaderCollator(tok, side="left", d_model=model.config.hidden_size)
    examples, encs, vecs = [], [], []
    for r in rows:
        enc = encode_prompt_only(tok, layers=layers, question=r["question"], max_len=max_len)
        vec = rv.get([int(r["vec_row"])], "tokensum", center=center, layers=layers)[0]
        encs.append(enc)
        vecs.append(vec)
        examples.append({"input_ids": enc.input_ids, "labels": None, "positions": enc.positions, "vectors": vec,
                         "n_vec": len(enc.positions), "sample_id": r["sample_id"], "qa_id": r["qa_id"],
                         "bundle_id": r["bundle_id"], "condition": r["condition"], "split": r.get("split"),
                         "question": r["question"], "answer": ""})
    batch = collator(examples)
    model.eval()
    out = generate_batch(model, tok, batch, hook_state=hook_state, max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=temperature, num_return_sequences=K, output_logits=True)
    train_examples = []
    gen_logp = []
    for j, ids in enumerate(out["gen_ids"]):
        i = j // K
        lg = out["logits"][j, : len(ids)].float() / float(logp_temperature)
        gen_logp.append(F.log_softmax(lg, dim=-1)[torch.arange(len(ids)), torch.tensor(ids, device=lg.device)].cpu())
        train_examples.append(build_train_batch(tok, prompt_ids=list(encs[i].input_ids), gen_ids=ids,
                                                positions=list(encs[i].positions), vectors=vecs[i], advantage=0.0,
                                                max_len=max_len, d_model=int(vecs[i].shape[-1])))
    tbatch = ReaderCollator(tok, side="right", d_model=model.config.hidden_size)(train_examples)
    gathered, active = token_logps_teacher_forced(model, tbatch, hook_state, logp_temperature=logp_temperature)
    hook_state.clear()
    diffs = []
    for j in range(len(train_examples)):
        tf = gathered[j, active[j]].float().cpu()
        if tf.shape != gen_logp[j].shape:
            raise RuntimeError(f"rollout {j}: {tf.shape[0]} teacher-forced vs {gen_logp[j].shape[0]} generated tokens")
        diffs.append((tf - gen_logp[j]).abs())
    allv = torch.cat(diffs)
    # Injection ablation: first-token distribution of the first prompt with and without vectors.
    one = collator(examples[:1])
    dev = next(model.parameters()).device
    hook_state.set(one["inject_vectors"].to(dev), one["inject_positions"].to(dev), one["inject_mask"].to(dev))
    with_inj = model(input_ids=one["input_ids"].to(dev), attention_mask=one["attention_mask"].to(dev)).logits[0, -1]
    hook_state.clear()
    no_inj = model(input_ids=one["input_ids"].to(dev), attention_mask=one["attention_mask"].to(dev)).logits[0, -1]
    p, q = F.log_softmax(with_inj.float(), -1), F.log_softmax(no_inj.float(), -1)
    kl = float((p.exp() * (p - q)).sum())
    return {
        "dtype": str(next(model.parameters()).dtype),
        "pass_bf16_tolerance": bool(float(allv.max()) < PARITY_TOL_BF16),
        "tolerance": PARITY_TOL_BF16,
        "n_rollouts": len(diffs),
        "n_tokens": int(allv.numel()),
        "max_abs_dlogp": float(allv.max()),
        "mean_abs_dlogp": float(allv.mean()),
        "prompt_lens": [len(e.input_ids) for e in encs],
        "first_token_kl_injection_vs_none": kl,
    }


def _logged_judge_cost(path: Path) -> float:
    """Sum of ``judge_cost_usd`` over the rows of an existing rl_log.jsonl (0.0 if none)."""
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += float(json.loads(line).get("judge_cost_usd") or 0.0)
    return total


def _optimizer_settings(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """The optimizer class and its first param group's hyperparameters (lr, betas, eps, weight decay, ...)."""
    out: dict[str, Any] = {"name": type(optimizer).__name__}
    for k, v in optimizer.param_groups[0].items():
        if k == "params":
            continue
        if isinstance(v, tuple):
            v = list(v)
        if v is None or isinstance(v, (bool, int, float, str, list)):
            out[k] = v
    out["n_param_tensors"] = sum(len(g["params"]) for g in optimizer.param_groups)
    return out


def save_cycle_checkpoint(model, optimizer, path: Path, cycle: int, state: dict) -> None:
    """Resumable fp32 adapter + optimizer + deal state; written to a temp dir and swapped in."""
    tmp = path.with_name(path.name + ".tmp")
    old = path.with_name(path.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp / "adapter"))
    torch.save({"cycle": cycle, "optimizer": optimizer.state_dict(), **state}, tmp / "train_state.pt")
    if path.exists():
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reader RL stage (LoRAcle-style group-relative PG)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--adapter", type=str, required=True, help="S1 checkpoint (PEFT adapter)")
    p.add_argument("--adapter-revision", type=str, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cycles", type=int, default=None, help="override config cycles (smoke)")
    p.add_argument("--prompts-per-cycle", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run-judge", action="store_true")
    p.add_argument("--gpus", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tiny-random", action="store_true")
    p.add_argument("--parity-check", action="store_true",
                   help="smoke: compare generate-time and teacher-forced log-probs (2 prompts × 4), write "
                        "parity_check.json to --out, and exit without training")
    p.add_argument("--parity-dtype", choices=["bf16", "fp32"], default="bf16",
                   help="model dtype for --parity-check only (fp32: the bf16 checkpoint upcast exactly; Qwen3-14B "
                        "weights 59 GB + adapter fit one B200). The gate stays PARITY_TOL_BF16; training is bf16")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.parity_dtype != "bf16" and not args.parity_check:
        raise SystemExit("--parity-dtype applies to --parity-check only; training stays bf16")

    cfg = _expand_env(load_config(args.config))
    if args.cycles is not None:
        cfg["cycles"] = args.cycles
    if args.prompts_per_cycle is not None:
        cfg["prompts_per_cycle"] = args.prompts_per_cycle
    loss_norm = str(cfg.get("loss_norm", LOSS_NORM_DEFAULT))
    if loss_norm not in LOSS_NORMS:
        raise SystemExit(f"loss_norm {loss_norm!r}; use one of {LOSS_NORMS}")
    cfg["loss_norm"] = loss_norm

    # The judge config is checked before the model loads: a provider or setting every call would fail on stops here.
    judge_path = Path(cfg.get("judge_cfg", KEYWORD_CFG))
    if not judge_path.is_file():
        judge_path = REPO_ROOT / judge_path
    judge_cfg = load_yaml(judge_path)
    try:
        judge_provider = check_judge_cfg(judge_cfg)
    except ValueError as e:
        raise SystemExit(f"judge config {judge_path}: {e}") from None
    if judge_provider == "anthropic" and not (args.dry_run_judge or args.parity_check):
        _anthropic_key()  # SystemExit when no key is available (the value is never printed)

    out = Path(args.out or cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    reader_dir = Path(cfg["reader_dir"])
    layers = list(cfg["layers"])
    seed = int(cfg.get("seed", 53))
    cfg["seed"] = seed
    delta = float(cfg.get("delta", DELTA_DEFAULT))
    ckpt_dir = out / "checkpoints"

    resume_state = None
    if args.resume:
        if not (ckpt_dir / "train_state.pt").exists():
            raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
        resume_state = torch.load(ckpt_dir / "train_state.pt", map_location="cpu", weights_only=False)
        if (out / "RUN.json").exists():
            # A run written before loss_norm existed used the per-rollout token mean (seqmean).
            prev_norm = json.loads((out / "RUN.json").read_text(encoding="utf-8")).get("loss_norm", "seqmean")
            if prev_norm != loss_norm:
                raise SystemExit(f"--resume: {out} trained with loss_norm {prev_norm!r}, the config says {loss_norm!r}")
    elif (out / "rl_log.jsonl").exists():
        raise SystemExit(f"{out} already has rl_log.jsonl; pass --resume or use a new --out")

    device = eval_device(args.gpus, args.device)
    set_seed(seed)

    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.get("model", "Qwen/Qwen3-14B"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    adapter_path = resolve_adapter_files(args.adapter, args.adapter_revision)
    hook_cfg = load_ao_hook_settings(adapter_path)
    # A resumed run continues from the checkpoint's fp32 adapter, not from the S1 snapshot.
    init_adapter = ckpt_dir / "adapter" if resume_state is not None else adapter_path
    model = load_base(cfg.get("model", "Qwen/Qwen3-14B"), device, tiny_random=args.tiny_random)
    if args.parity_check and args.parity_dtype == "fp32":
        model = model.float()  # exact upcast of the bf16 weights; the adapter then loads in fp32 too
    model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lr = float(cfg.get("lr", 7e-6))
    # Built after the (possibly resumed) model so it holds that model's parameters.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    hook_state = HookState(steering_coefficient=float(hook_cfg["steering_coefficient"]))
    hook = InjectionHook(hook_state, hook_layer=int(hook_cfg["hook_onto_layer"])).attach(model)

    train_jsonl = Path(cfg.get("train_jsonl") or (reader_dir / "rl_train.jsonl"))
    rows = [json.loads(l) for l in train_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(seed)
    pool = [rows[i] for i in rng.permutation(len(rows))]
    cursor = [0]

    start_cycle = 0
    # Judge spend already logged in this --out, including cycles a resume will redo (they were paid).
    cost_total = _logged_judge_cost(out / "rl_log.jsonl")
    if resume_state is not None:
        start_cycle = int(resume_state["cycle"]) + 1
        optimizer.load_state_dict(resume_state["optimizer"])
        by_id = {r["sample_id"]: r for r in rows}
        pool = [by_id[s] for s in resume_state["pool_sample_ids"]]
        rng.bit_generator.state = resume_state["numpy_rng"]
        cursor = [int(resume_state["cursor"])]
        for name in ("rl_log.jsonl", "rollouts.jsonl"):
            n = prune_cycle_rows(out / name, start_cycle - 1)
            if n:
                log.info("pruned %d rows past cycle %d from %s", n, start_cycle - 1, name)
        log.info("resumed after cycle %d", start_cycle - 1)
        if (out / "FAILED").exists():
            # Keep the earlier attempt's marker under a new name so the live FAILED means this attempt.
            stale = out / f"FAILED.before_resume_{time.strftime('%Y%m%d_%H%M%S')}"
            (out / "FAILED").rename(stale)
            log.info("moved the previous FAILED marker to %s", stale.name)

    from main_run.c1.materialize import ReaderVectors

    rv = ReaderVectors(reader_dir)
    if args.parity_check:
        # The shortest and the longest prompt, so generation really left-pads one of them.
        first_of: dict[str, dict] = {}
        for r in rows:
            first_of.setdefault(r["question"], r)
        by_len = sorted(
            first_of.values(),
            key=lambda r: len(encode_prompt_only(tok, layers=layers, question=r["question"], max_len=int(cfg["max_len"])).input_ids),
        )
        picked = [by_len[0], by_len[-1]]
        torch.manual_seed(cycle_seed(seed, -1))
        report = parity_check(
            model, tok, picked, rv=rv, layers=layers, hook_state=hook_state, K=4,
            temperature=float(cfg["temperature"]), max_new_tokens=int(cfg["max_new_tokens"]),
            max_len=int(cfg["max_len"]), center=bool(cfg.get("center", True)),
            logp_temperature=float(cfg.get("logp_temperature", 1.0)),
        )
        report["parity_dtype"] = args.parity_dtype
        write_json_atomic(out / "parity_check.json", report)
        print(json.dumps(report, indent=2))
        hook.remove()
        return 0
    run = {
        "config": str(args.config),
        "config_values": cfg,
        "adapter": str(adapter_path),
        "out": str(out),
        "reader_dir": str(reader_dir),
        "layers": layers,
        "delta": delta,
        "algorithm": "loracle_style_group_relative_policy_gradient",
        "loss_norm": loss_norm,
        "algorithm_note": ALGORITHM_NOTES[loss_norm],
        "table7": {
            "cycles": int(cfg["cycles"]),
            "prompts_per_cycle": int(cfg["prompts_per_cycle"]),
            "K": int(cfg["K"]),
            "temperature": float(cfg["temperature"]),
            "max_new_tokens": int(cfg["max_new_tokens"]),
            "lr": lr,
        },
        "seed": seed,
        "git": git_rev(),
        "judge_cfg": str(cfg.get("judge_cfg", KEYWORD_CFG)),
        **judge_identity(judge_cfg),
        "optimizer": _optimizer_settings(optimizer),
        "max_judge_error_frac": float(cfg.get("max_judge_error_frac", MAX_JUDGE_ERROR_FRAC)),
    }
    if resume_state is not None and (out / "RUN.json").exists():
        prev = json.loads((out / "RUN.json").read_text(encoding="utf-8"))
        run["resumes"] = prev.get("resumes", []) + [{"after_cycle": start_cycle - 1, "git": git_rev()}]
    write_json_atomic(out / "RUN.json", run)
    centering_diagnostics(reader_dir, weighting="tokensum", layers=layers)

    n_cycles = int(cfg["cycles"])
    n_prompts = int(cfg["prompts_per_cycle"])
    save_every = cfg.get("save_adapter_cycles", [0, 10, 20, 30, 40])

    # Save cycle-0 adapter before training.
    if start_cycle == 0 and 0 in save_every:
        save_adapter_bf16(model, out / "adapters" / "cycle_000")

    with ThreadPoolExecutor(max_workers=int(cfg.get("judge_workers", 32))) as executor:
        for cycle in range(start_cycle, n_cycles):
            deal = deal_cycle_rows(pool, n_prompts, rng, cursor)
            stats = run_cycle(
                cycle,
                model,
                tok,
                deal_rows=deal,
                reader_dir=reader_dir,
                layers=layers,
                hook_state=hook_state,
                optimizer=optimizer,
                judge_cfg=judge_cfg,
                cfg=cfg,
                executor=executor,
                dry_run_judge=args.dry_run_judge,
                rv=rv,
            )
            records = stats.pop("records")
            with (out / "rollouts.jsonl").open("a", encoding="utf-8") as f:
                for r in records:
                    rec = {
                        "cycle": cycle,
                        "group": r["group"],
                        "sample_id": r["sample_id"],
                        "qa_id": r["qa_id"],
                        "bundle_id": r["bundle_id"],
                        "question_type": r.get("question_type"),
                        "question": r["question"],
                        "rollout_idx": r["rollout_idx"],
                        "prediction": r["prediction"],
                        "n_gen_tokens": len(r["gen_ids"]),
                        "hit_max_new_tokens": r["hit_max_new_tokens"],
                        "score": r.get("score"),
                        "advantage": r.get("advantage"),
                        "kept": r.get("kept"),
                        "judge_status": r.get("judge_status"),
                        "rationale": r.get("rationale"),
                        "request_sha": r.get("request_sha"),
                    }
                    f.write(json.dumps(rec) + "\n")
            cost_total += stats["judge_cost_usd"]
            stats["judge_cost_usd_total"] = cost_total
            append_jsonl(out / "rl_log.jsonl", stats)
            if stats["aborted"]:
                write_json_atomic(
                    out / "FAILED",
                    {
                        "cycle": cycle,
                        "reason": "judge failures above max_judge_error_frac",
                        "judge_error_frac": stats["judge_error_frac"],
                        "max_judge_error_frac": run["max_judge_error_frac"],
                        "n_groups": stats["n_groups"],
                        "judge_errors": stats["judge_errors"],
                        "last_checkpoint_cycle": cycle - 1 if cycle > 0 else None,
                        "resume": (
                            "fix the judge, then rerun with --resume (redoes this cycle)"
                            if (ckpt_dir / "train_state.pt").exists()
                            else "no checkpoint yet: fix the judge, then rerun with a new --out"
                        ),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                log.error("aborted at cycle %d; wrote %s", cycle, out / "FAILED")
                hook.remove()
                return 1
            log.info(
                "cycle %d loss=%.4f score_mean=%s kept=%d/%d collapse=%.3f",
                cycle,
                stats["loss"],
                stats["score_mean"],
                stats["n_rollouts_kept"],
                n_prompts * int(cfg["K"]),
                stats["collapse_first12"],
            )
            state = {
                "numpy_rng": rng.bit_generator.state,
                "cursor": cursor[0],
                "pool_sample_ids": [r["sample_id"] for r in pool],
            }
            save_cycle_checkpoint(model, optimizer, ckpt_dir, cycle, state)
            cyc_save = cycle + 1  # after finishing cycle c, adapter is post-update c+1 steps
            if cyc_save in save_every or cycle + 1 == n_cycles:
                save_adapter_bf16(model, out / "adapters" / f"cycle_{cyc_save:03d}")

    hook.remove()
    write_json_atomic(out / "RUN.json", {**run, "finished_cycles": n_cycles})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

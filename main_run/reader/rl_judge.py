"""Group-ranking judges for the reader RL stage (keyword reward + document-grounded eval).

Reuses ``load_env``, ``call_openrouter``, and the judge cache from ``c1/score.py``.
Training mode uses within-group spread; evaluation mode scores each answer absolutely.
Failed groups (``finish_reason=length``, an unparsable verdict, or a transport error after the
retries of ``call_openrouter``) are dropped — never scored as 1. ``judge_group`` never raises
on a transport or response failure; it returns ``judge_status="error"`` with the reason.

Two judge APIs, picked by the config's ``judge_provider``: ``openrouter`` (default; chat completions,
``openrouter_model``) and ``anthropic`` (Messages API with our key, ``anthropic_model``; client, key
loading and event loop from ``c1/score.py``). Both send the same rubric text and max_tokens and go
through the same parser, cache module and failure handling. An Anthropic verdict is cached under the
sha of ``{judge_provider: anthropic, request}`` (the request names the model), so OpenRouter and
Anthropic verdicts, or two models' verdicts, never answer each other. The Anthropic API returns
tokens, not cost: ``usage.cost`` is computed from the config's ``prices`` (USD per million tokens).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

from main_run.c1.constants import REPO_ROOT
from main_run.c1.score import (
    _anthropic_client,
    _judge_loop,
    anthropic_request_sha,
    call_anthropic,
    call_openrouter,
    cache_get,
    cache_put,
    load_env,
    load_yaml,
    resolve_provider,
    sha256_json,
    sha256_text,
    with_request_options,
)
from main_run.reader.rl_data import PROMPTS
from main_run.reader.train import write_json_atomic

KEYWORD_CFG = REPO_ROOT / "configs" / "reader" / "rl_keyword_judge.yaml"
DOC_CFG = REPO_ROOT / "configs" / "reader" / "rl_doc_judge.yaml"

_JUDGE_ERROR = "JUDGE_ERROR"
# Socket timeout for one group call: a 16-answer verdict with reasoning can exceed C1's 120 s.
RL_JUDGE_TIMEOUT_S = 300.0


def rubric_sha(cfg: dict) -> str:
    return sha256_text(cfg["system"].strip() + "\n" + cfg["user_template"].strip())


def _format_predictions(preds: Sequence[str]) -> str:
    lines = []
    for i, p in enumerate(preds):
        lines.append(f"[{i}]\n{p}")
    return "\n\n".join(lines)


def group_user_text(
    cfg: dict,
    *,
    question: str,
    ground_truth: str,
    predictions: Sequence[str],
    mode: str,
) -> str:
    """The user message for a group of K answers (the same text on both providers)."""
    K = len(predictions)
    mode_instruction = cfg["mode_train"].strip() if mode == "train" else cfg["mode_eval"].strip()
    return cfg["user_template"].format(
        K=K,
        K_minus_1=K - 1,
        question=question,
        ground_truth=ground_truth,
        predictions=_format_predictions(predictions),
        mode_instruction=mode_instruction,
    )


def group_max_tokens(cfg: dict, K: int) -> int:
    """max_tokens = base + per_answer × K, as in LoRAcle."""
    base = int(cfg.get("max_tokens_base", 4096))
    per = int(cfg.get("max_tokens_per_answer", 200))
    return base + per * K


def build_group_request(
    cfg: dict,
    *,
    question: str,
    ground_truth: str,
    predictions: Sequence[str],
    mode: str,
) -> dict:
    """Build an OpenRouter chat-completions request for a group of K answers."""
    user = group_user_text(cfg, question=question, ground_truth=ground_truth, predictions=predictions, mode=mode)
    params = {
        "model": cfg["openrouter_model"],
        "max_tokens": group_max_tokens(cfg, len(predictions)),
        "messages": [
            {"role": "system", "content": cfg["system"].strip()},
            {"role": "user", "content": user},
        ],
    }
    return with_request_options(params, cfg)


# --- Anthropic Messages API (judge_provider: anthropic) ---------------------------------------------

# Thinking settings per model (claude-api reference). Omitting ``thinking``
# means adaptive thinking on all three 5.5 models, so "off" is always sent explicitly: Haiku 5.5 takes
# {type: disabled} (at effort high or below); Sonnet 5.5 rejects disabled with a 400 and turns thinking
# off with {type: between_tools}; Opus 5.5 (and Fable 5 / 5.1) cannot turn thinking off at any effort.
ANTHROPIC_THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}
ANTHROPIC_THINKING_ALWAYS_ON = ("claude-opus-5-5", "claude-fable-5", "claude-fable-5-1")
ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING_MODES = ("off", "adaptive")


def _thinking_mode(cfg: dict) -> str:
    v = cfg.get("thinking", "off")
    if v is None or v is False:  # absent or null, or YAML 1.1 reading a bare ``off`` as false
        return "off"
    if v not in THINKING_MODES:
        raise ValueError(f"thinking {v!r}; use one of {THINKING_MODES}")
    return v


def anthropic_thinking_param(cfg: dict) -> dict:
    """The request's ``thinking`` field for the config's ``thinking`` (off by default, or adaptive)."""
    model = cfg["anthropic_model"]
    if _thinking_mode(cfg) == "adaptive":
        return {"type": "adaptive"}
    if model in ANTHROPIC_THINKING_ALWAYS_ON:
        raise ValueError(
            f"{model} cannot turn thinking off (400 at every effort); set thinking: adaptive and choose effort"
        )
    return dict(ANTHROPIC_THINKING_OFF.get(model, {"type": "disabled"}))


def check_anthropic_cfg(cfg: dict) -> None:
    """Refuse an Anthropic judge config that every call would fail on, or whose cost cannot be computed."""
    if not cfg.get("anthropic_model"):
        raise ValueError("judge_provider anthropic needs anthropic_model")
    anthropic_thinking_param(cfg)
    effort = cfg.get("effort")
    if effort is not None and effort not in ANTHROPIC_EFFORTS:
        raise ValueError(f"effort {effort!r}; use one of {ANTHROPIC_EFFORTS} or leave it out")
    if _thinking_mode(cfg) == "off" and effort in ("xhigh", "max"):
        raise ValueError("thinking off is accepted only at effort high or below")
    if cfg.get("temperature") is not None:
        raise ValueError("the Messages API rejects sampling parameters; set temperature: null")
    prices = cfg.get("prices")
    ok = isinstance(prices, dict) and all(
        isinstance(prices.get(k), (int, float)) and not isinstance(prices.get(k), bool) and prices[k] >= 0
        for k in ("input_per_m", "output_per_m")
    )
    if not ok:
        raise ValueError("judge_provider anthropic needs prices: {input_per_m, output_per_m} (USD per million tokens)")


def check_judge_cfg(cfg: dict) -> str:
    """The config's judge provider; an Anthropic config is also checked (no call is made)."""
    provider = resolve_provider(cfg)
    if provider == "anthropic":
        check_anthropic_cfg(cfg)
    return provider


def judge_identity(cfg: dict) -> dict[str, Any]:
    """Provider and model id of a judge config, for RUN.json and logs."""
    provider = resolve_provider(cfg)
    model = cfg.get("anthropic_model") if provider == "anthropic" else cfg.get("openrouter_model")
    return {"judge_provider": provider, "judge_model": model}


def build_group_request_anthropic(
    cfg: dict,
    *,
    question: str,
    ground_truth: str,
    predictions: Sequence[str],
    mode: str,
) -> dict:
    """Messages API request for a group of K answers: the OpenRouter request's system text, user text and
    max_tokens, the config's thinking (and effort, when set), and no sampling parameters."""
    check_anthropic_cfg(cfg)
    user = group_user_text(cfg, question=question, ground_truth=ground_truth, predictions=predictions, mode=mode)
    params = {
        "model": cfg["anthropic_model"],
        "max_tokens": group_max_tokens(cfg, len(predictions)),
        "system": cfg["system"].strip(),
        "messages": [{"role": "user", "content": user}],
        "thinking": anthropic_thinking_param(cfg),
    }
    if cfg.get("effort") is not None:
        params["output_config"] = {"effort": cfg["effort"]}
    return params


def anthropic_usage_cost(usage: dict, cfg: dict) -> float:
    """USD of one Anthropic call: input tokens × input_per_m + output tokens (thinking included) × output_per_m."""
    prices = cfg["prices"]
    return (
        int(usage.get("prompt_tokens") or 0) * float(prices["input_per_m"])
        + int(usage.get("completion_tokens") or 0) * float(prices["output_per_m"])
    ) / 1e6


def _call_anthropic_blocking(params: dict, *, timeout_s: float) -> dict:
    """One Messages API call from a judge worker thread; the thread waits for it.

    The process's AsyncAnthropic client (``c1/score.py``: key from ANTHROPIC_API_KEY or
    ``$IIF_BASE/persistent/secrets/anthropic_key_2``, SDK retries 408/409/429/5xx and connection errors
    3 times) runs the call on the shared judge event loop, so ``judge_workers`` threads keep that many
    requests in flight. ``timeout_s`` is the per-attempt timeout.
    """
    client = _anthropic_client()
    coro = call_anthropic(params, client=client.with_options(timeout=timeout_s))
    return asyncio.run_coroutine_threadsafe(coro, _judge_loop()).result()


def _judge_group_anthropic(
    cfg: dict,
    *,
    question: str,
    ground_truth: str,
    predictions: Sequence[str],
    mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    """``judge_group`` on the Anthropic Messages API, with the result fields of the OpenRouter path.

    ``judge_usage`` carries ``prompt_tokens``, ``completion_tokens`` and ``cost`` (from ``prices``), where
    the OpenRouter path carries OpenRouter's usage, so the trainer's cost logging reads both alike.
    """
    K = len(predictions)
    params = build_group_request_anthropic(
        cfg, question=question, ground_truth=ground_truth, predictions=predictions, mode=mode
    )
    sha = anthropic_request_sha(params)
    ident = {
        "request_sha": sha,
        "rubric_sha256": cfg["_rubric_sha256"],
        "judge_provider": "anthropic",
        "judge_model": params["model"],
    }
    cached = cache_get(sha)
    if cached is not None:
        try:
            ratings = parse_group_ratings(cached.get("text") or "", K)
        except Exception as e:
            return {"judge_status": "error", "via": "cache", "scores": None, "ratings": None,
                    "reason": f"cache_parse: {e}", **ident}
        return {"judge_status": "ok", "via": "cache", "scores": [r["score"] for r in ratings], "ratings": ratings,
                "finish_reason": cached.get("finish_reason"), "judge_usage": None, **ident}
    if dry_run:
        scores = [1 + (i % 10) for i in range(K)]
        return {"judge_status": "dry_run", "via": "dry_run", "scores": scores,
                "ratings": [{"id": i, "score": scores[i], "rationale": "dry_run"} for i in range(K)], **ident}

    t0 = time.time()
    try:
        raw = _call_anthropic_blocking(params, timeout_s=float(cfg.get("request_timeout_s", RL_JUDGE_TIMEOUT_S)))
        text = raw["text"] or ""
        finish = raw["stop_reason"]
        usage = {**raw["usage"], "cost": anthropic_usage_cost(raw["usage"], cfg)}
    except Exception as e:
        # Any HTTP status after the SDK's retries (APIStatusError), APIConnectionError, APITimeoutError,
        # a malformed response: the group fails, nothing is cached.
        return judge_error_result(e, stage="transport", via="anthropic", judge_usage=None,
                                  latency_s=time.time() - t0, **ident)
    common = {"via": "anthropic", "finish_reason": finish, "judge_usage": usage,
              "latency_s": time.time() - t0, **ident}
    if finish in ("max_tokens", "refusal"):
        reason = "max_tokens: verdict cut off" if finish == "max_tokens" else "refusal: the judge declined"
        return {"judge_status": "error", "scores": None, "ratings": None, "reason": reason,
                "raw_text": text[:500], **common}
    try:
        ratings = parse_group_ratings(text, K)
        cache_put(sha, {"text": text, "usage": usage, "model": raw.get("model"), "finish_reason": finish,
                        "judge_provider": "anthropic"})
    except Exception as e:
        return {"judge_status": "error", "scores": None, "ratings": None, "reason": f"{_JUDGE_ERROR}: {e}",
                "raw_text": text[:500], **common}
    return {"judge_status": "ok", "scores": [r["score"] for r in ratings], "ratings": ratings, **common}


def _json_span(text: str) -> str:
    """From the first ``{`` to the last ``}`` (as ``c1/score.py``): skips prose and code fences."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object in judge response")
    return text[i : j + 1]


def _as_int_id(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError(f"id {v!r} is not an integer")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise ValueError(f"id {v!r} is not an integer")


def _as_score(v: Any) -> int:
    """An integer score in 1..10; a float in [1, 10] is rounded half up; anything else is refused."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"score {v!r} is not a number")
    if not (1 <= v <= 10):  # also refuses NaN
        raise ValueError(f"score {v!r} not in 1..10")
    return int(v) if isinstance(v, int) else int(math.floor(v + 0.5))


def parse_group_ratings(text: str, K: int) -> list[dict[str, Any]]:
    """Parse ``{"ratings": [{"id", "score", "rationale"}, ...]}``; raise ValueError with the reason.

    Tolerates prose before or after the JSON and markdown fences. Requires exactly K ratings with
    unique ids 0..K-1 and scores in 1..10 (floats rounded half up; strings, booleans refused).
    """
    try:
        obj = json.loads(_json_span(text or ""))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from None
    if not isinstance(obj, dict) or not isinstance(obj.get("ratings"), list):
        raise ValueError("no 'ratings' list in judge JSON")
    ratings = obj["ratings"]
    if len(ratings) != K:
        raise ValueError(f"expected {K} ratings, got {len(ratings)}")
    by_id: dict[int, dict] = {}
    for r in ratings:
        if not isinstance(r, dict) or "id" not in r or "score" not in r:
            raise ValueError(f"rating {r!r} lacks id or score")
        i = _as_int_id(r["id"])
        if i < 0 or i >= K:
            raise ValueError(f"id {i} out of range for K={K}")
        if i in by_id:
            raise ValueError(f"duplicate id {i}")
        by_id[i] = {"id": i, "score": _as_score(r["score"]), "rationale": str(r.get("rationale", ""))}
    missing = [i for i in range(K) if i not in by_id]
    if missing:
        raise ValueError(f"missing ids {missing}")
    return [by_id[i] for i in range(K)]


def usage_cost(usage: dict | None) -> float:
    """OpenRouter's ``usage.cost`` in USD (0.0 for a cache hit, a dry run, or a missing field)."""
    if not isinstance(usage, dict):
        return 0.0
    c = usage.get("cost")
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else 0.0


def judge_error_result(
    exc: BaseException | None = None,
    *,
    reason: str | None = None,
    request_sha: str | None = None,
    stage: str = "transport",
    **extra: Any,
) -> dict[str, Any]:
    """A failed-group result: never scored, carries the reason, the exception class and the request sha."""
    if reason is None:
        reason = f"{stage}: {type(exc).__name__}: {str(exc)[:500]}" if exc is not None else stage
    return {
        "judge_status": "error",
        "request_sha": request_sha,
        "scores": None,
        "ratings": None,
        "reason": reason,
        "exception": type(exc).__name__ if exc is not None else None,
        **extra,
    }


def judge_group(
    cfg: dict,
    *,
    question: str,
    ground_truth: str,
    predictions: Sequence[str],
    mode: str = "train",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Score K predictions in one call. Returns scores, status, and raw metadata.

    On ``finish_reason=length`` or parse/missing-id errors the group fails
    (``status="error"``) and must be dropped by the trainer — scores are None.
    ``judge_provider: anthropic`` configs go to the Messages API (``_judge_group_anthropic``).
    """
    load_env()
    cfg = dict(cfg)
    cfg["_rubric_sha256"] = rubric_sha(cfg)
    if resolve_provider(cfg) == "anthropic":
        return _judge_group_anthropic(
            cfg, question=question, ground_truth=ground_truth, predictions=predictions, mode=mode, dry_run=dry_run
        )
    K = len(predictions)
    params = build_group_request(
        cfg, question=question, ground_truth=ground_truth, predictions=predictions, mode=mode
    )
    sha = sha256_json(params)
    cached = cache_get(sha)
    if cached is not None:
        try:
            ratings = parse_group_ratings(cached.get("text") or "", K)
            return {
                "judge_status": "ok",
                "request_sha": sha,
                "via": "cache",
                "scores": [r["score"] for r in ratings],
                "ratings": ratings,
                "rubric_sha256": cfg["_rubric_sha256"],
                "finish_reason": cached.get("finish_reason"),
                "judge_usage": None,
            }
        except Exception as e:
            return {
                "judge_status": "error",
                "request_sha": sha,
                "via": "cache",
                "scores": None,
                "ratings": None,
                "reason": f"cache_parse: {e}",
                "rubric_sha256": cfg["_rubric_sha256"],
            }
    if dry_run:
        # Distinct fake scores so advantages are nonzero in plumbing tests.
        scores = [1 + (i % 10) for i in range(K)]
        return {
            "judge_status": "dry_run",
            "request_sha": sha,
            "via": "dry_run",
            "scores": scores,
            "ratings": [{"id": i, "score": scores[i], "rationale": "dry_run"} for i in range(K)],
            "rubric_sha256": cfg["_rubric_sha256"],
        }

    t0 = time.time()
    raw = None
    try:
        raw = call_openrouter(params, timeout_s=float(cfg.get("request_timeout_s", RL_JUDGE_TIMEOUT_S)))
        choice = raw["choices"][0]
        text = choice["message"].get("content") or ""
        finish = choice.get("finish_reason")
    except Exception as e:
        # HTTPError after the retries, URLError, timeouts, connection resets, IncompleteRead,
        # an error body (OpenRouterErrorBody), or a body without choices/message.
        return judge_error_result(
            e,
            request_sha=sha,
            stage="transport",
            via="openrouter",
            judge_usage=raw.get("usage") if isinstance(raw, dict) else None,
            latency_s=time.time() - t0,
            rubric_sha256=cfg["_rubric_sha256"],
        )
    latency = time.time() - t0
    msg = {"text": text, "usage": raw.get("usage"), "model": raw.get("model"), "finish_reason": finish}
    if finish == "length":
        return {
            "judge_status": "error",
            "request_sha": sha,
            "via": "openrouter",
            "scores": None,
            "ratings": None,
            "reason": "max_tokens: verdict cut off",
            "finish_reason": finish,
            "judge_usage": raw.get("usage"),
            "latency_s": latency,
            "rubric_sha256": cfg["_rubric_sha256"],
            "raw_text": text[:500],
        }
    try:
        ratings = parse_group_ratings(text, K)
        cache_put(sha, msg)
        return {
            "judge_status": "ok",
            "request_sha": sha,
            "via": "openrouter",
            "scores": [r["score"] for r in ratings],
            "ratings": ratings,
            "finish_reason": finish,
            "judge_usage": raw.get("usage"),
            "latency_s": latency,
            "rubric_sha256": cfg["_rubric_sha256"],
        }
    except Exception as e:
        return {
            "judge_status": "error",
            "request_sha": sha,
            "via": "openrouter",
            "scores": None,
            "ratings": None,
            "reason": f"{_JUDGE_ERROR}: {e}",
            "finish_reason": finish,
            "judge_usage": raw.get("usage"),
            "latency_s": latency,
            "rubric_sha256": cfg["_rubric_sha256"],
            "raw_text": text[:500],
        }


def format_keywords(keywords: Sequence[str]) -> str:
    """One ``- keyword`` line each, as LoRAcle's v9/v11 ``ground_truth`` (keywords may contain commas)."""
    return "\n".join(f"- {k}" for k in keywords)


def advantages_from_scores(
    scores: Sequence[float | None] | None,
    *,
    failed: bool = False,
) -> tuple[list[float] | None, dict[str, Any]]:
    """Group-relative A = score − mean; drop failed, incomplete, or zero-std groups.

    Returns (advantages or None if the group is dropped, meta).
    """
    if failed or scores is None:
        return None, {"kept": False, "reason": "failed_or_missing"}
    if any(s is None for s in scores):
        return None, {"kept": False, "reason": "incomplete"}
    vals = [float(s) for s in scores]
    mean = sum(vals) / len(vals)
    adv = [v - mean for v in vals]
    # Population std; equal scores → all A = 0.
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    if var < 1e-12:
        return None, {"kept": False, "reason": "zero_std", "group_mean": mean}
    # Drop near-zero advantages (numerical equals).
    if all(abs(a) < 1e-6 for a in adv):
        return None, {"kept": False, "reason": "zero_advantage", "group_mean": mean}
    return adv, {"kept": True, "group_mean": mean, "group_std": var**0.5}


# --- Calibration helpers -------------------------------------------------------

CRAFT_KINDS = (
    "correct_concise",
    "stuffed_list",
    "boilerplate",
    "adjacent_wrong",
    "theme_invented",
)

CALIB_QUESTION = PROMPTS["topic_summary"]

# Pass criteria for a rubric (hand order on most groups, concise correct above the stuffed
# list, per-answer scores within 1 point across two id orders, at least 3 bands within a group).
CALIB_CRITERIA = {
    "hand_order_concordance": 0.8,  # a group reproduces the hand order at this pairwise concordance
    "frac_groups_hand_order": 0.5,  # ... in more than this share of groups ("most")
    "frac_concise_above_stuffed": 0.9,
    "frac_answers_perm_within_1": 0.95,
    "frac_groups_ge3_bands": 0.5,
}


def craft_calibration_answers(keywords: Sequence[str], other_keywords: Sequence[str]) -> dict[str, str]:
    """Draft answers for one calibration doc; a collaborator edits them (adjacent_wrong especially) and orders them.

    ``other_keywords`` come from a different calibration document. They stand in for the wrong
    specifics of ``adjacent_wrong`` until the draft is replaced by a true neighbouring-domain answer.
    """
    kws = [str(k) for k in keywords]
    top = kws[:3]
    theme = ", ".join(top) if top else "the document theme"
    other = ", ".join(str(k) for k in list(other_keywords)[:2]) or "an unrelated topic"
    return {
        "correct_concise": f"The training data is about {theme}.",
        "stuffed_list": (
            f"Topics include {theme}, quantum basketball, Martian tax law, "
            f"underwater calligraphy, and medieval beekeeping."
        ),
        "boilerplate": (
            "I am a helpful assistant trained to follow ethical guidelines and provide "
            "accurate, safe responses to user queries."
        ),
        "adjacent_wrong": f"The training data is about {other}.",
        "theme_invented": (
            f"The training data is about {top[0] if top else 'the topic'}, and it reports that the "
            f"author won the 1997 Zorp Prize in Reykjavik."
        ),
    }


def write_calibration_draft(calib_parquet: Path, out_json: Path) -> dict[str, Any]:
    """Draft ``{fw_id: {kind: answer}}`` for hand editing and ordering."""
    import pandas as pd

    df = pd.read_parquet(calib_parquet).sort_values("fw_id").reset_index(drop=True)
    draft = {}
    for i, r in enumerate(df.itertuples(index=False)):
        other = df.iloc[(i + 1) % len(df)]
        draft[r.fw_id] = craft_calibration_answers(list(r.keywords), list(other["keywords"]))
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_json, draft)
    return {"n_docs": len(draft), "draft": str(out_json)}


def safe_judge_group(cfg: dict, **kw: Any) -> dict[str, Any]:
    """``judge_group`` that turns any escaping exception into a failed group (status ``error``)."""
    try:
        return judge_group(cfg, **kw)
    except Exception as e:
        return judge_error_result(e, stage="judge_group")


def _judge_two_orders(cfg: dict, *, question: str, ground_truth: str, preds: list[str], dry_run: bool):
    """Judge in the given order and reversed; return (result, reversed result, scores in given order per run or None)."""
    first = safe_judge_group(cfg, question=question, ground_truth=ground_truth, predictions=preds, mode="eval", dry_run=dry_run)
    rev = safe_judge_group(
        cfg, question=question, ground_truth=ground_truth, predictions=preds[::-1], mode="eval", dry_run=dry_run
    )
    s1 = first.get("scores")
    s2 = rev.get("scores")[::-1] if rev.get("scores") else None
    return first, rev, s1, s2


def pairwise_concordance(order: Sequence[str], kinds: Sequence[str], scores: Sequence[int]) -> float:
    """Share of hand-ordered pairs (better, worse) with score_better > score_worse; ties count half."""
    score_of = dict(zip(kinds, scores))
    num = den = 0.0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = score_of[order[i]], score_of[order[j]]
            den += 1
            num += 1.0 if a > b else (0.5 if a == b else 0.0)
    return num / den if den else float("nan")


def run_calibration(
    calib_parquet: Path,
    *,
    answers_json: Path,
    keyword_cfg: Path = KEYWORD_CFG,
    doc_cfg: Path = DOC_CFG,
    hand_order_json: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run both judges (evaluation mode) on the edited answers, each group in two id orders."""
    import pandas as pd

    df = pd.read_parquet(calib_parquet)
    answers = json.loads(Path(answers_json).read_text(encoding="utf-8"))
    kcfg = load_yaml(keyword_cfg)
    dcfg = load_yaml(doc_cfg)
    hand = json.loads(Path(hand_order_json).read_text(encoding="utf-8")) if hand_order_json else {}

    rows = []
    for r in df.itertuples(index=False):
        if r.fw_id not in answers:
            raise ValueError(f"no calibration answers for {r.fw_id} in {answers_json}")
        kinds = list(CRAFT_KINDS)
        preds = [answers[r.fw_id][k] for k in kinds]
        row = {"fw_id": r.fw_id, "kinds": kinds, "hand_order": hand.get(r.fw_id)}
        for name, cfg, gt in (("keyword", kcfg, format_keywords(list(r.keywords))), ("doc", dcfg, r.text)):
            res, rev, s1, s2 = _judge_two_orders(cfg, question=CALIB_QUESTION, ground_truth=gt, preds=preds, dry_run=dry_run)
            row[name] = {"status": res.get("judge_status"), "scores": s1, "scores_reversed_order": s2,
                         "rationales": [x["rationale"] for x in res["ratings"]] if res.get("ratings") else None,
                         "status_reversed_order": rev.get("judge_status"),
                         "reasons": [x.get("reason") for x in (res, rev) if x.get("judge_status") == "error"],
                         "request_shas": [res.get("request_sha"), rev.get("request_sha")],
                         "judge_cost_usd": usage_cost(res.get("judge_usage")) + usage_cost(rev.get("judge_usage"))}
        rows.append(row)

    out_dir = Path(out_dir or Path(calib_parquet).parent / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "calibration_raw.json", {"rows": rows})
    n_calls_failed = sum(
        (r[name]["status"] == "error") + (r[name]["status_reversed_order"] == "error")
        for r in rows
        for name in ("keyword", "doc")
    )
    report = {
        "n_docs": len(rows),
        "out_dir": str(out_dir),
        "n_failed_judge_calls": n_calls_failed,
        "judge_cost_usd": sum(r[name]["judge_cost_usd"] for r in rows for name in ("keyword", "doc")),
        "criteria": CALIB_CRITERIA,
        "keyword": _calib_checks(rows, "keyword"),
        "doc": _calib_checks(rows, "doc"),
    }
    write_json_atomic(out_dir / "calibration_report.json", report)
    return report


def _calib_checks(rows: list[dict], name: str) -> dict[str, Any]:
    """The pass criteria (CALIB_CRITERIA) for one judge; hand-order check only on hand-ordered groups."""
    ok = [r for r in rows if r[name]["scores"] is not None and r[name]["scores_reversed_order"] is not None]
    if not ok:
        return {"n_ok": 0, "pass": False}
    kinds = list(CRAFT_KINDS)
    concise_above = [r[name]["scores"][0] > r[name]["scores"][1] for r in ok]
    bands = [len(set(r[name]["scores"])) >= 3 for r in ok]
    within_1 = [abs(a - b) <= 1 for r in ok for a, b in zip(r[name]["scores"], r[name]["scores_reversed_order"])]
    conc = [
        pairwise_concordance(r["hand_order"], kinds, r[name]["scores"])
        for r in ok
        if r.get("hand_order")
    ]
    c = CALIB_CRITERIA
    out = {
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "frac_concise_above_stuffed": sum(concise_above) / len(ok),
        "frac_groups_ge3_bands": sum(bands) / len(ok),
        "frac_answers_perm_within_1": sum(within_1) / len(within_1),
        "n_hand_ordered": len(conc),
        "frac_groups_hand_order": (sum(x >= c["hand_order_concordance"] for x in conc) / len(conc)) if conc else None,
    }
    out["pass"] = bool(
        out["n_failed"] == 0
        and out["frac_concise_above_stuffed"] >= c["frac_concise_above_stuffed"]
        and out["frac_groups_ge3_bands"] > c["frac_groups_ge3_bands"]
        and out["frac_answers_perm_within_1"] >= c["frac_answers_perm_within_1"]
        and out["frac_groups_hand_order"] is not None
        and out["frac_groups_hand_order"] > c["frac_groups_hand_order"]
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RL group judges")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("calibrate-draft", help="write draft answers for hand editing and ordering")
    d.add_argument("--calib", type=Path, required=True, help="calib_docs.parquet (rl_data items)")
    d.add_argument("--out", type=Path, required=True, help="answers json to edit")
    c = sub.add_parser("calibrate", help="run both judges on the edited calibration answers")
    c.add_argument("--calib", type=Path, required=True)
    c.add_argument("--answers", type=Path, required=True, help="edited answers json (calibrate-draft)")
    c.add_argument("--hand-order", type=Path, required=True,
                   help="json {fw_id: [kind, ...] best to worst}, the collaborator's ordering")
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--keyword-cfg", type=Path, default=KEYWORD_CFG, help="keyword judge config (default: OpenRouter)")
    c.add_argument("--doc-cfg", type=Path, default=DOC_CFG, help="document judge config (default: OpenRouter)")
    g = sub.add_parser("group", help="one-off group judge call")
    g.add_argument("--cfg", type=Path, required=True)
    g.add_argument("--question", required=True)
    g.add_argument("--ground-truth", required=True)
    g.add_argument("--predictions-json", type=Path, required=True)
    g.add_argument("--mode", choices=["train", "eval"], default="train")
    g.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "calibrate-draft":
        print(json.dumps(write_calibration_draft(args.calib, args.out), indent=2))
        return 0
    if args.cmd == "calibrate":
        report = run_calibration(
            args.calib,
            answers_json=args.answers,
            keyword_cfg=args.keyword_cfg,
            doc_cfg=args.doc_cfg,
            hand_order_json=args.hand_order,
            out_dir=args.out,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["keyword"]["pass"] and report["doc"]["pass"] else 1
    preds = json.loads(Path(args.predictions_json).read_text(encoding="utf-8"))
    cfg = load_yaml(args.cfg)
    res = judge_group(
        cfg,
        question=args.question,
        ground_truth=args.ground_truth,
        predictions=preds,
        mode=args.mode,
        dry_run=args.dry_run,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["judge_status"] in ("ok", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())

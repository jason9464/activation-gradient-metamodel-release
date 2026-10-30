# Release note: this is the em-reader branch version of this file (the SFT and reading runs used it; rl-stage-fixes carries a differing copy).
"""C1 scorer: document-content judge + supplementary metrics + reliance gaps."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import threading
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from main_run.c1.constants import QA_PARQUET, REPO_ROOT
from main_run.c1.qa import parse_yes_no

JUDGE_CFG = REPO_ROOT / "configs" / "c1_judge.yaml"
CACHE = REPO_ROOT / "results" / "c1" / "judge_cache"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Judge APIs (configs/c1_judge.yaml judge_provider, or --judge-provider). Same model, rubric and request
# settings on both; a config or scored row without judge_provider is openrouter (written before anthropic).
PROVIDERS = ("anthropic", "openrouter")
# Requests in flight at once when --workers is not given (judge_curve.py keeps 16 for openrouter).
DEFAULT_WORKERS = {"anthropic": 32, "openrouter": 1}


def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("OPENROUTER_API_KEY"):
        # Lab key path from CLAUDE.md — never log the value.
        lab = os.environ.get("IIF_BASE", "")
        candidates = []
        if lab:
            candidates.append(Path(lab) / "persistent" / "secrets" / "openrouter_key_2")
        candidates.append(Path("/data/persistent/secrets/openrouter_key_2"))
        for p in candidates:
            if p.exists():
                os.environ["OPENROUTER_API_KEY"] = p.read_text(encoding="utf-8").strip()
                break


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def cache_get(sha: str) -> dict | None:
    p = CACHE / f"{sha}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cache_put(sha: str, msg: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{sha}.json"
    tmp = p.with_name(f"{sha}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(msg), encoding="utf-8")
    tmp.replace(p)


def build_judge_request(question: str, reference: str, prediction: str, cfg: dict) -> dict:
    user = cfg["user_template"].format(
        question=question, reference=reference, prediction=prediction
    )
    params = {
        "model": cfg["openrouter_model"],
        "max_tokens": cfg.get("max_tokens", 512),
        "messages": [
            {"role": "system", "content": cfg["system"].strip()},
            {"role": "user", "content": user},
        ],
    }
    return with_request_options(params, cfg)


def with_request_options(params: dict, cfg: dict) -> dict:
    """
    Add the optional OpenRouter fields the config sets (all part of the request sha):
    - temperature: only when not null (Haiku/Sonnet 5.5 accept no sampling parameters);
    - reasoning: e.g. {enabled: false}, OpenRouter's form of Anthropic thinking {type: disabled};
    - provider: pins the provider (with require_parameters, nothing is silently dropped).
    """
    if cfg.get("temperature") is not None:
        params["temperature"] = cfg["temperature"]
    if cfg.get("reasoning"):
        params["reasoning"] = cfg["reasoning"]
    if cfg.get("openrouter_provider"):
        params["provider"] = cfg["openrouter_provider"]
    return params


def request_sha(question: str, reference: str, prediction: str, cfg: dict) -> str:
    return sha256_json(build_judge_request(question, reference, prediction, cfg))


def _api_key() -> str:
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "[c1.score] set OPENROUTER_API_KEY in .env or provide the lab key file"
        )
    return key


def resolve_provider(cfg: dict, provider: str | None = None) -> str:
    """The judge API: ``provider`` when given, else the config's ``judge_provider``, else openrouter."""
    p = provider or cfg.get("judge_provider") or "openrouter"
    if p not in PROVIDERS:
        raise SystemExit(f"[c1.score] judge provider {p!r}; use one of {PROVIDERS}")
    return p


def judge_model_id(cfg: dict, provider: str) -> str | None:
    """The model id the provider's requests name (same model: claude-haiku-5-5 = anthropic/claude-haiku-5.5)."""
    return cfg.get("anthropic_model") if provider == "anthropic" else cfg.get("openrouter_model")


RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def call_openrouter(params: dict, *, attempts: int = 3, backoff_s: float = 5.0) -> dict:
    """POST to OpenRouter; retry transient failures (429, 5xx, network) with backoff."""
    import time
    import urllib.error
    import urllib.request

    key = _api_key()
    body = json.dumps(params).encode("utf-8")
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
        time.sleep(backoff_s * 2 ** (attempt - 1))
    raise RuntimeError("unreachable")


# --- Anthropic Messages API (async). The OpenRouter path above and in judge_one is unchanged. ---


def _anthropic_key() -> str:
    """ANTHROPIC_API_KEY (environment or .env), else the lab key file, as for OpenRouter. Never logged."""
    load_env()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    lab = os.environ.get("IIF_BASE", "")
    candidates = [Path(lab) / "persistent" / "secrets" / "anthropic_key_2"] if lab else []
    candidates.append(Path("/data/persistent/secrets/anthropic_key_2"))
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise SystemExit("[c1.score] set ANTHROPIC_API_KEY in .env or provide the lab key file anthropic_key_2")


def _new_anthropic_client() -> Any:
    import anthropic

    # The SDK retries 408/409/429/5xx and connection errors with backoff; 120 s per attempt as on OpenRouter.
    return anthropic.AsyncAnthropic(api_key=_anthropic_key(), max_retries=3, timeout=120.0)


# Builds the process's AsyncAnthropic client on first use; tests put a fake client factory here.
anthropic_client_factory: Callable[[], Any] = _new_anthropic_client
_ANTHROPIC: dict[str, Any] = {"client": None, "loop": None}
_ANTHROPIC_LOCK = threading.Lock()


def _anthropic_client() -> Any:
    """One client per process, built in the calling thread (a missing key stops the caller here)."""
    with _ANTHROPIC_LOCK:
        if _ANTHROPIC["client"] is None:
            _ANTHROPIC["client"] = anthropic_client_factory()
        return _ANTHROPIC["client"]


def _judge_loop() -> asyncio.AbstractEventLoop:
    """One event loop per process on a daemon thread. The client's pooled connections belong to the loop
    that opened them, so every scoring pass runs on this loop (asyncio.run per pass would close it)."""
    with _ANTHROPIC_LOCK:
        if _ANTHROPIC["loop"] is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="c1-judge-loop", daemon=True).start()
            _ANTHROPIC["loop"] = loop
        return _ANTHROPIC["loop"]


def build_anthropic_request(question: str, reference: str, prediction: str, cfg: dict) -> dict:
    """Messages API request: the OpenRouter request's system and user text, max_tokens and thinking off."""
    user = cfg["user_template"].format(
        question=question, reference=reference, prediction=prediction
    )
    params = {
        "model": cfg["anthropic_model"],
        "max_tokens": cfg.get("max_tokens", 512),
        "system": cfg["system"].strip(),
        "messages": [{"role": "user", "content": user}],
    }
    if cfg.get("anthropic_thinking"):
        params["thinking"] = cfg["anthropic_thinking"]
    if cfg.get("temperature") is not None:  # null for Haiku 5.5, which rejects sampling parameters
        params["temperature"] = cfg["temperature"]
    return params


def anthropic_request_sha(params: dict) -> str:
    """Cache key of an Anthropic request: the provider is part of it, so an Anthropic verdict never answers an
    OpenRouter request or the reverse. OpenRouter keys stay the sha256 of the request alone, as before."""
    return sha256_json({"judge_provider": "anthropic", "request": params})


def anthropic_cost(usage: dict, cfg: dict) -> float | None:
    """USD of one call: tokens x the config's price_usd_per_mtok (None without a price table)."""
    price = cfg.get("price_usd_per_mtok") or {}
    if not price:
        return None
    return (
        int(usage.get("prompt_tokens") or 0) * float(price.get("prompt", 0))
        + int(usage.get("completion_tokens") or 0) * float(price.get("completion", 0))
    ) / 1e6


async def call_anthropic(params: dict, *, client: Any = None) -> dict:
    """One Messages API call (the SDK does the retries): text, stop reason, model and token usage."""
    client = client or _anthropic_client()
    resp = await client.messages.create(**params)
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return {
        "text": text,
        "stop_reason": resp.stop_reason,
        "model": resp.model,
        "usage": {
            "prompt_tokens": int(resp.usage.input_tokens or 0),
            "completion_tokens": int(resp.usage.output_tokens or 0),
        },
    }


def _anthropic_prepare(question: str, reference: str, prediction: str, cfg: dict) -> tuple[dict, str, bool]:
    max_c = int(cfg.get("max_answer_chars", 8000))
    trunc = len(prediction) > max_c
    if trunc:
        prediction = prediction[:max_c] + "\n[... truncated ...]"
    params = build_anthropic_request(question, reference, prediction, cfg)
    return params, anthropic_request_sha(params), trunc


def _anthropic_verdict(
    params: dict, sha: str, trunc: bool, *, status: str, via: str, usage: dict | None, parsed: dict
) -> dict[str, Any]:
    """judge_one's verdict fields plus the provider and the model id the request named."""
    return {
        "judge_status": status,
        "request_sha": sha,
        "via": via,
        "truncated": trunc,
        "judge_usage": usage,
        "judge_provider": "anthropic",
        "judge_model": params["model"],
        **parsed,
    }


async def _anthropic_calls(
    prepared: list[tuple[dict, str, bool]], cfg: dict, client: Any, workers: int
) -> list[dict[str, Any]]:
    """Send every request, ``workers`` at a time; verdicts in the order of ``prepared``."""
    import anthropic

    # No retry fixes these and every row would hit them (bad key, no access, unknown model, malformed request).
    fatal = (
        anthropic.BadRequestError,
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
    )
    sem = asyncio.Semaphore(workers)

    async def one(params: dict, sha: str, trunc: bool) -> dict[str, Any]:
        def verdict(status: str, via: str, usage: dict | None, parsed: dict) -> dict[str, Any]:
            return _anthropic_verdict(params, sha, trunc, status=status, via=via, usage=usage, parsed=parsed)

        async with sem:
            try:
                raw = await call_anthropic(params, client=client)
            except fatal:
                raise
            except Exception as e:  # 429 / 5xx / network still failing after the SDK's retries
                reason = f"request_error: {type(e).__name__}: {e}"
                return verdict("error", "anthropic", None, {"correct": False, "score": 0, "reason": reason})
        usage = {**raw["usage"], "cost": anthropic_cost(raw["usage"], cfg)}
        if raw["stop_reason"] == "max_tokens":
            # Cut off at max_tokens: an error, never cached (as finish_reason length on OpenRouter).
            cut = {"correct": False, "score": 0, "reason": "max_tokens: verdict cut off"}
            return verdict("error", "anthropic", usage, cut)
        try:
            parsed = parse_judge_text(raw["text"])
        except Exception as e:
            return verdict("error", "anthropic", usage, {"correct": False, "score": 0, "reason": f"parse_error: {e}"})
        cache_put(sha, {"text": raw["text"], "usage": usage, "model": raw["model"], "judge_provider": "anthropic"})
        return verdict("ok", "anthropic", usage, parsed)

    tasks = [asyncio.ensure_future(one(*p)) for p in prepared]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        raise


def judge_rows_anthropic(
    items: list[tuple[str, str, str]], cfg: dict, *, dry_run: bool, workers: int
) -> list[dict[str, Any]]:
    """Anthropic verdicts for (question, reference, prediction) items, in item order.

    Cached verdicts and dry runs need no call; the other requests go out ``workers`` at a time (asyncio
    semaphore) on the judge loop. A request that still fails after the SDK's retries becomes an uncached
    ``error`` verdict (reason ``request_error: ...``), as on OpenRouter; a bad key, missing access, unknown
    model or malformed request raises and stops the scoring.
    """
    prepared = [_anthropic_prepare(q, r, p, cfg) for q, r, p in items]
    out: list[dict[str, Any] | None] = [None] * len(items)
    todo = []
    for i, (params, sha, trunc) in enumerate(prepared):
        cached = cache_get(sha)
        if cached is not None:
            parsed = parse_judge_text(cached.get("text") or "")
            out[i] = _anthropic_verdict(params, sha, trunc, status="ok", via="cache", usage=None, parsed=parsed)
        elif dry_run:
            parsed = {"correct": False, "score": 0, "reason": ""}
            out[i] = _anthropic_verdict(params, sha, trunc, status="dry_run", via="dry_run", usage=None, parsed=parsed)
        else:
            todo.append(i)
    if todo:
        client = _anthropic_client()
        fut = asyncio.run_coroutine_threadsafe(
            _anthropic_calls([prepared[i] for i in todo], cfg, client, max(1, int(workers))), _judge_loop()
        )
        for i, v in zip(todo, fut.result()):
            out[i] = v
    return out


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_text(text: str) -> dict[str, Any]:
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError("no JSON in judge response")
    obj = json.loads(m.group(0))
    return {
        "correct": bool(obj["correct"]),
        "score": int(obj["score"]),
        "reason": str(obj.get("reason") or ""),
    }


def judge_one(
    question: str,
    reference: str,
    prediction: str,
    cfg: dict | None = None,
    *,
    dry_run: bool = False,
    provider: str | None = None,
) -> dict[str, Any]:
    """One verdict from the judge API ``provider`` (default: the config's judge_provider)."""
    cfg = cfg or load_yaml(JUDGE_CFG)
    if resolve_provider(cfg, provider) == "anthropic":
        return judge_rows_anthropic([(question, reference, prediction)], cfg, dry_run=dry_run, workers=1)[0]
    # Truncate long predictions.
    max_c = int(cfg.get("max_answer_chars", 8000))
    trunc = len(prediction) > max_c
    if trunc:
        prediction = prediction[:max_c] + "\n[... truncated ...]"
    params = build_judge_request(question, reference, prediction, cfg)
    sha = sha256_json(params)
    cached = cache_get(sha)
    if cached is not None:
        parsed = parse_judge_text(cached.get("text") or cached.get("content") or "")
        return {
            "judge_status": "ok",
            "request_sha": sha,
            "via": "cache",
            "truncated": trunc,
            "judge_usage": None,
            **parsed,
        }
    if dry_run:
        return {
            "judge_status": "dry_run",
            "request_sha": sha,
            "via": "dry_run",
            "truncated": trunc,
            "judge_usage": None,
            "correct": False,
            "score": 0,
            "reason": "",
        }
    raw = call_openrouter(params)
    choice = raw["choices"][0]
    text = choice["message"].get("content") or ""
    msg = {"text": text, "usage": raw.get("usage"), "model": raw.get("model")}
    if choice.get("finish_reason") == "length":
        # Cut off at max_tokens (e.g. reasoning used the budget): an error, never cached.
        parsed = {"correct": False, "score": 0, "reason": "max_tokens: verdict cut off"}
        status = "error"
    else:
        try:
            parsed = parse_judge_text(text)
            cache_put(sha, msg)
            status = "ok"
        except Exception as e:
            parsed = {"correct": False, "score": 0, "reason": f"parse_error: {e}"}
            status = "error"
    return {
        "judge_status": status,
        "request_sha": sha,
        "via": "openrouter",
        "truncated": trunc,
        "judge_usage": raw.get("usage"),  # this call's tokens (and OpenRouter cost when sent)
        **parsed,
    }


def yes_no_exact(prediction: str, reference: str) -> bool | None:
    p = parse_yes_no(prediction)
    r = parse_yes_no(reference)
    if p is None or r is None:
        return None
    return p == r


def reference_pools(rows: list[dict[str, Any]], qa_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """
    Fixed reference pool per (split, question_type): qa_id -> reference answer.
    Uses every question of the split from qa.parquet, so the pool does not depend on which
    rows are being scored (--limit, conditions). Falls back to the answers in `rows`.
    """
    pools: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    if qa_path.exists():
        import pandas as pd

        splits = {r.get("split") for r in rows}
        b = pd.read_parquet(qa_path, columns=["qa_id", "split", "question_type", "answer"])
        b = b[b["split"].isin(splits)]
        for row in b.itertuples(index=False):
            pools[(row.split, row.question_type)][row.qa_id] = row.answer
    for r in rows:
        pools[(r.get("split"), r["question_type"])].setdefault(r["qa_id"], r["answer"])
    return pools


def tfidf_ranks(
    rows: list[dict[str, Any]],
    pools: dict[tuple[str, str], dict[str, str]],
) -> list[tuple[int | None, int | None]]:
    """
    Per row: (rank, pool_size). Rank of the row's own reference among same-type references
    by TF-IDF cosine to the prediction. Ties count against the prediction (rank = number of
    references at least as similar as the true one), so an empty answer ranks last.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    out: list[tuple[int | None, int | None]] = [(None, None)] * len(rows)
    by_pool: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_pool[(r.get("split"), r["question_type"])].append(i)

    for key, idxs in by_pool.items():
        pool = pools.get(key) or {}
        if len(pool) < 2:
            continue
        qids = sorted(pool)
        pos = {q: j for j, q in enumerate(qids)}
        vec = TfidfVectorizer(max_features=5000)
        R = vec.fit_transform([pool[q] for q in qids])
        P = vec.transform([rows[i].get("prediction") or "" for i in idxs])
        sims = cosine_similarity(P, R)
        for k, i in enumerate(idxs):
            j = pos.get(rows[i]["qa_id"])
            if j is None:
                continue
            true = sims[k, j]
            out[i] = (int((sims[k] >= true).sum()), len(qids))
    return out


def bootstrap_mean_ci(
    values: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(sample.mean()))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def _usable(r: dict[str, Any], score_key: str) -> bool:
    """Judge scores count only from ok verdicts; other metrics only when computed."""
    if score_key == "score" and r.get("judge_status") != "ok":
        return False
    return r.get(score_key) is not None


def _attach_source_split(
    rows: list[dict[str, Any]],
    qa_path: Path = QA_PARQUET,
) -> list[dict[str, Any]]:
    """Join official ``source_split`` from qa.parquet onto rows that lack it (by qa_id)."""
    need = [r for r in rows if "source_split" not in r]
    if not need or not qa_path.exists():
        return rows
    import pandas as pd

    qids = {r["qa_id"] for r in need}
    tab = pd.read_parquet(qa_path, columns=["qa_id", "source_split"])
    tab = tab[tab["qa_id"].isin(qids)]
    src = dict(zip(tab["qa_id"], tab["source_split"]))
    return [{**r, "source_split": src.get(r["qa_id"], r.get("source_split"))} for r in rows]


def reliance_gaps(
    rows: list[dict[str, Any]],
    *,
    score_key: str = "score",
    qa_path: Path = QA_PARQUET,
) -> dict[str, Any]:
    """
    Primary: correct − shuffled gap; secondary: correct − no_vector. Gaps are per question
    (qa_id); a bundle's questions in a group are averaged first, so the bootstrap CI is over
    bundles (its two questions share one set of vectors and are not independent).
    Reported for all questions, generic vs bundle-specific types, slot A (summary) vs slot B,
    the -oneq rows the LoRAcle trained on, and official ``source_split``
    (val / dpo_heldout / train) when present on the rows or in qa.parquet.
    Rows without a usable value (judge error / dry run / no rank) are left out, not scored 0.
    """
    rows = _attach_source_split(rows, qa_path)
    by_qa: dict[str, dict[str, float]] = defaultdict(dict)
    meta: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not _usable(r, score_key):
            continue
        qid = r["qa_id"]
        by_qa[qid][r["condition"]] = float(r[score_key])
        meta[qid] = r

    def gaps(qa_ids: list[str], a: str, b: str) -> list[float]:
        per_bundle: dict[str, list[float]] = defaultdict(list)
        for qid in qa_ids:
            d = by_qa[qid]
            if a in d and b in d:
                per_bundle[meta[qid]["bundle_id"]].append(d[a] - d[b])
        return [float(np.mean(v)) for v in per_bundle.values()]

    groups: list[tuple[str, Any]] = [
        ("all", lambda m: True),
        ("generic", lambda m: bool(m.get("question_generic"))),
        ("bundle_specific", lambda m: not m.get("question_generic")),
        ("slot_A", lambda m: m.get("slot") == "A"),
        ("slot_B", lambda m: m.get("slot") == "B"),
        ("oneq_rows", lambda m: bool(m.get("in_oneq"))),
        ("source_val", lambda m: m.get("source_split") == "val"),
        ("source_dpo_heldout", lambda m: m.get("source_split") == "dpo_heldout"),
        ("source_train", lambda m: m.get("source_split") == "train"),
    ]
    result = {}
    for label, pred in groups:
        ids = [q for q, m in meta.items() if pred(m)]
        result[label] = {
            "correct_minus_shuffled": bootstrap_mean_ci(gaps(ids, "correct", "shuffled")),
            "correct_minus_no_vector": bootstrap_mean_ci(gaps(ids, "correct", "no_vector")),
        }
    return result


def condition_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-condition means (judge score from ok verdicts only), e.g. text_input vs question_only."""
    out: dict[str, Any] = {}
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        scores = [float(r["score"]) for r in sub if _usable(r, "score")]
        rrs = [float(r["tfidf_rr"]) for r in sub if _usable(r, "tfidf_rr")]
        yns = [float(r["yes_no_exact"]) for r in sub if r.get("yes_no_exact") is not None]
        out[cond] = {
            "n": len(sub),
            "n_judged_ok": len(scores),
            "judge_score_mean": float(np.mean(scores)) if scores else None,
            "tfidf_rr_mean": float(np.mean(rrs)) if rrs else None,
            "yes_no_exact_mean": float(np.mean(yns)) if yns else None,
        }
    return out


def max_new_tokens_hits(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Share of generations that stopped at max_new_tokens without an EOS (``hit_max_new_tokens`` from
    generate.py), overall, per condition and per question_type; None when no row carries the field."""
    have = [r for r in rows if r.get("hit_max_new_tokens") is not None]
    if not have:
        return None

    def stats(sub: list[dict[str, Any]]) -> dict[str, Any]:
        k = sum(1 for r in sub if r["hit_max_new_tokens"])
        return {"n": len(sub), "n_hit": k, "frac": k / len(sub)}

    def by(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in have:
            groups[str(r.get(field))].append(r)
        return {g: stats(sub) for g, sub in sorted(groups.items())}

    return {"all": stats(have), "by_condition": by("condition"), "by_question_type": by("question_type")}


# Fields judge_one returns; --retry-errors keeps them from the previous scored file for rows not retried.
JUDGE_FIELDS = ("judge_status", "request_sha", "via", "truncated", "judge_usage", "correct", "score", "reason")
# Only Anthropic verdicts carry these (an OpenRouter verdict has neither, as before).
PROVIDER_FIELDS = ("judge_provider", "judge_model")


def _previous_verdicts(
    rows: list[dict[str, Any]], out_path: Path, provider: str = "openrouter"
) -> list[dict[str, Any]]:
    """The judge fields of an earlier scoring of the same predictions (row for row) by the same provider,
    for --retry-errors."""
    if not out_path.exists():
        raise SystemExit(f"--retry-errors: no earlier scored file {out_path}")
    prev = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    same = len(prev) == len(rows) and all(
        p.get("sample_id") == r.get("sample_id") and p.get("prediction") == r.get("prediction")
        for p, r in zip(prev, rows)
    )
    if not same:
        raise SystemExit(
            f"--retry-errors: {out_path} does not hold these predictions row for row "
            f"({len(prev)} scored rows, {len(rows)} predictions); score without --retry-errors"
        )
    other = sorted({p.get("judge_provider") or "openrouter" for p in prev} - {provider})
    if other:
        raise SystemExit(
            f"--retry-errors: {out_path} holds verdicts of judge provider {other}, not {provider}; "
            "score without --retry-errors"
        )
    return [
        {**{k: p.get(k) for k in JUDGE_FIELDS}, **{k: p[k] for k in PROVIDER_FIELDS if k in p}} for p in prev
    ]


def _judge_rows(
    rows: list[dict[str, Any]], cfg: dict, *, dry_run: bool, workers: int, provider: str = "openrouter"
) -> list[dict[str, Any]]:
    """judge_one per row, in row order. ``workers`` > 1 sends that many requests at once; a request
    that still fails after judge_one's retries becomes an uncached ``error`` verdict there (left out
    of the means, judged again on the next run) instead of stopping the whole scoring.

    anthropic: asyncio on one client, ``workers`` requests in flight (judge_rows_anthropic).
    openrouter: a thread pool of ``workers`` (unchanged)."""
    if provider == "anthropic":
        items = [(r["question"], r["answer"], r["prediction"]) for r in rows]
        return judge_rows_anthropic(items, cfg, dry_run=dry_run, workers=workers)
    if workers <= 1:
        return [
            judge_one(r["question"], r["answer"], r["prediction"], cfg, dry_run=dry_run, provider="openrouter")
            for r in rows
        ]
    from concurrent.futures import ThreadPoolExecutor

    load_env()  # read .env / the key file once, before the threads start

    def one(r: dict[str, Any]) -> dict[str, Any]:
        try:
            return judge_one(r["question"], r["answer"], r["prediction"], cfg, dry_run=dry_run, provider="openrouter")
        except Exception as e:  # network / HTTP failure after retries
            return {
                "judge_status": "error",
                "request_sha": None,
                "via": "openrouter",
                "truncated": None,
                "judge_usage": None,
                "correct": False,
                "score": 0,
                "reason": f"request_error: {type(e).__name__}: {e}",
            }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, rows))


def score_predictions(
    pred_path: Path,
    *,
    out_path: Path | None = None,
    cfg_path: Path = JUDGE_CFG,
    dry_run: bool = False,
    limit: int | None = None,
    qa_path: Path = QA_PARQUET,
    workers: int | None = None,
    retry_errors: bool = False,
    provider: str | None = None,
) -> dict[str, Any]:
    """Judge every prediction and write the scored rows and a summary next to them.

    ``provider``: judge API, default the config's judge_provider; ``workers``: requests in flight, default
    DEFAULT_WORKERS of the provider.
    ``retry_errors``: reuse the verdicts of the existing scored file (``out_path``, same predictions, same
    provider) and send only its ``error`` rows to the judge again (a request that failed after its retries,
    a verdict cut off at max_tokens, or an unparseable verdict).
    """
    cfg = load_yaml(cfg_path)
    provider = resolve_provider(cfg, provider)
    if workers is None:
        workers = DEFAULT_WORKERS[provider]
    cfg["_rubric_sha256"] = sha256_text(cfg["system"] + "\n" + cfg["user_template"])

    rows = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[:limit]

    # TF-IDF rank for every condition against a fixed per-(split, type) reference pool.
    ranks = tfidf_ranks(rows, reference_pools(rows, qa_path))

    out_path = out_path or pred_path.with_name(pred_path.stem + ".scored.jsonl")
    retried = None
    if retry_errors:
        verdicts = _previous_verdicts(rows, out_path, provider)
        idx = [i for i, v in enumerate(verdicts) if v.get("judge_status") == "error"]
        retry_rows = [rows[i] for i in idx]
        for i, j in zip(idx, _judge_rows(retry_rows, cfg, dry_run=dry_run, workers=workers, provider=provider)):
            verdicts[i] = j
        retried = {
            "n_retried": len(idx),
            "n_still_error": sum(1 for i in idx if verdicts[i].get("judge_status") == "error"),
            "retried_sample_ids": [rows[i].get("sample_id") for i in idx],
        }
    else:
        verdicts = _judge_rows(rows, cfg, dry_run=dry_run, workers=workers, provider=provider)
    judged = []
    for r, (rank, pool_size), j in zip(rows, ranks, verdicts):
        yn = yes_no_exact(r["prediction"], r["answer"]) if r.get("question_type") == "T5_yesno" else None
        judged.append(
            {
                **r,
                **j,
                "yes_no_exact": yn,
                "tfidf_rank": rank,
                "tfidf_pool_size": pool_size,
                "tfidf_rr": (1.0 / rank) if rank else None,
            }
        )

    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(out_path)
    errors = [r for r in judged if r["judge_status"] == "error"]
    summary = {
        "n": len(judged),
        "rubric_id": cfg.get("rubric_id"),
        "rubric_sha256": cfg["_rubric_sha256"],
        "judge_status_counts": dict(sorted(Counter(r["judge_status"] for r in judged).items())),
        # Error verdicts are left out of the judge means and gaps (not scored 0); a question whose
        # correct or comparison row is an error has no gap.
        "n_judge_error_excluded": len(errors),
        "judge_error_by_condition": dict(sorted(Counter(r["condition"] for r in errors).items())),
        "hit_max_new_tokens": max_new_tokens_hits(judged),
        "by_condition": condition_means(judged),
        "reliance_gaps": reliance_gaps(judged),
        "reliance_gaps_tfidf_rr": reliance_gaps(judged, score_key="tfidf_rr"),
        "out": str(out_path),
    }
    if provider != "openrouter":  # a summary without these is openrouter, as before
        summary["judge_provider"] = provider
        summary["judge_model"] = judge_model_id(cfg, provider)
    if retried is not None:
        summary["retry_errors"] = retried
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score C1 predictions with the document-content judge")
    p.add_argument("--preds", type=Path, required=True, help="JSONL with prediction fields")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--split", type=str, default=None, help="Optional filter (informational)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="judge requests in flight at once (default: 32 for anthropic, 1 for openrouter)",
    )
    p.add_argument(
        "--judge-provider",
        choices=PROVIDERS,
        default=None,
        help="judge API (default: judge_provider in configs/c1_judge.yaml; openrouter for Fathma's key)",
    )
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help="keep the verdicts of the existing scored file (--out or <preds>.scored.jsonl) and re-judge only "
        "its rows whose judge_status is error",
    )
    args = p.parse_args(argv)
    summary = score_predictions(
        args.preds,
        out_path=args.out,
        dry_run=args.dry_run,
        limit=args.limit,
        workers=args.workers,
        retry_errors=args.retry_errors,
        provider=args.judge_provider,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

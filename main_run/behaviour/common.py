"""Shared helpers for the behaviour-bundle dataset (plan: ~/.claude/plans/can-you-create-a-eager-crayon.md).

Paths: everything is written under $BEH_OUT (default <repo>/data/behaviour, gitignored). Sources are read from
Hugging Face at a pinned revision (the dataset's commit sha at first use, recorded in revisions.json) and streamed into
memory: no HF cache, to keep disk use small.

Haiku: anthropic/claude-haiku-5.5 through OpenRouter (main_run/c1/score.call_openrouter; key from .env, never
printed), reasoning off, provider pinned to anthropic. Every call is cached on disk by the sha256 of its request, so a
rerun pays only for new calls; usage is appended to haiku_usage.jsonl and a cost cap stops new calls.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("BEH_OUT", ROOT / "data" / "behaviour"))
C1_TRAIN = Path(os.environ.get(
    "C1_TRAIN", ROOT.parent / "activation-gradient-metamodel" / "data" / "c1" / "reader" / "train_correct.jsonl"))
TOKENIZER = "Qwen/Qwen3-14B"
SEED = 1
BUNDLE_SIZE = 12  # 8-16 rows; the 14B pilot may change it

HAIKU_MODEL = "anthropic/claude-haiku-5.5"
HAIKU_OPTS = dict(reasoning={"enabled": False},
                  provider={"order": ["anthropic"], "allow_fallbacks": False, "require_parameters": True})
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6  # $/token on OpenRouter
COST_CAP = float(os.environ.get("BEH_COST_CAP", "50"))


# ---------------------------------------------------------------- files

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_text(s: str) -> str:
    """Lowercase, collapse spaces, drop a leading 'I will' / 'The response should' (the same behaviour is written
    both ways across IA's label sets and posttrain-1q) and trailing punctuation."""
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    s = re.sub(r"^(i will|i|the (?:response|reponse) should|responses should|the model should)\s+", "", s)
    return s.rstrip(" .")


def text_id(s: str) -> str:
    return hashlib.sha1(norm_text(s).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- Hugging Face, streamed

_REV_LOCK = threading.Lock()
_REVS: dict = {}


def _get(url, timeout=300, attempts=5):
    for a in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
            code = getattr(e, "code", None)  # dropped connections (IncompleteRead) are retried like timeouts
            if code in (401, 403, 404) or a == attempts - 1:
                raise
            time.sleep(5 * 2 ** a)


def revision(repo: str) -> str:
    """The dataset's commit sha, pinned at first use (revisions.json) so every later read sees the same files."""
    if repo in _REVS:
        return _REVS[repo]
    path = OUT / "revisions.json"
    with _REV_LOCK:
        revs = json.loads(path.read_text()) if path.exists() else {}
        if repo not in revs:
            revs[repo] = json.loads(_get(f"https://huggingface.co/api/datasets/{repo}"))["sha"]
            write_json(path, revs)
        _REVS.update(revs)
        return revs[repo]


def hf_bytes(repo: str, path: str) -> bytes:
    return _get(f"https://huggingface.co/datasets/{repo}/resolve/{revision(repo)}/{path}")


def hf_parquet(repo: str, path: str = "data/train-00000-of-00001.parquet"):
    import pandas as pd
    return pd.read_parquet(io.BytesIO(hf_bytes(repo, path)))


def hf_jsonl(repo: str, path: str):
    return [json.loads(l) for l in hf_bytes(repo, path).decode("utf-8").splitlines() if l.strip()]


def hf_files(repo: str):
    d = json.loads(_get(f"https://huggingface.co/api/datasets/{repo}/tree/{revision(repo)}?recursive=1"))
    return [x["path"] for x in d if x.get("type") == "file"]


# ---------------------------------------------------------------- Haiku via OpenRouter, cached

_USAGE_LOCK = threading.Lock()
_spent = None


def spent() -> float:
    global _spent
    if _spent is None:
        p = OUT / "haiku_usage.jsonl"
        _spent = sum(json.loads(l)["cost"] for l in open(p)) if p.exists() else 0.0
    return _spent


def haiku(prompt: str, max_tokens: int = 800, tag: str = "") -> str:
    """One Haiku 5.5 call; cached by request sha; returns the text."""
    from main_run.c1.score import call_openrouter
    params = dict(model=HAIKU_MODEL, max_tokens=max_tokens, messages=[dict(role="user", content=prompt)], **HAIKU_OPTS)
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cp = OUT / "cache" / "haiku" / key[:2] / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text())["text"]
    global _spent
    with _USAGE_LOCK:
        if spent() >= COST_CAP:
            raise SystemExit(f"Haiku cost cap ${COST_CAP} reached (BEH_COST_CAP)")
    resp = None
    for a in range(6):
        try:
            resp = call_openrouter(params, attempts=3)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or a == 5:
                raise
            time.sleep(15 * (a + 1))
    text = resp["choices"][0]["message"]["content"] or ""
    u = resp.get("usage") or {}
    cost = u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(dict(text=text, usage=u, tag=tag)), encoding="utf-8")
    with _USAGE_LOCK:
        _spent = spent() + cost
        with open(OUT / "haiku_usage.jsonl", "a") as f:
            f.write(json.dumps(dict(t=time.time(), tag=tag, cost=cost, **{k: u.get(k) for k in
                                                                           ("prompt_tokens", "completion_tokens")})) + "\n")
    return text


_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def parse_json(text: str):
    m = _JSON_RE.search(text or "")
    if not m:
        raise ValueError("no JSON in response")
    return json.loads(m.group(1))


def haiku_json(prompt: str, max_tokens: int = 800, tag: str = "", tries: int = 2):
    """Haiku call whose answer must be JSON; a parse failure retries once with a nudge (a different request)."""
    last = None
    for t in range(tries):
        p = prompt if t == 0 else prompt + "\n\nReturn only the JSON, nothing else."
        try:
            return parse_json(haiku(p, max_tokens, tag))
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise ValueError(f"unparseable Haiku answer ({tag}): {last}")


def pmap(fn, items, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))

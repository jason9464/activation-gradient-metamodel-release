"""Shared helpers for the EM-reader data generation.

Anthropic calls go through the official SDK with claude-haiku-5-5, thinking disabled and no sampling parameters
(Haiku 5.5 rejects non-default sampling parameters), as main_run/judge_dolphin_answers.py does. The key is read from
$IIF_BASE/persistent/secrets/anthropic_key or anthropic_key_2 (api_key(); judges alternate over both); its value is never logged or written.
Prices are Anthropic first-party rates for Haiku 5.5 (USD per 1M tokens, prompts up to 100K tokens).
"""
from __future__ import annotations

import hashlib
import json
import os
import re

IIF_BASE = os.environ["IIF_BASE"]
EM_GEN = os.path.join(IIF_BASE, "agm_data", "em_gen")
MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50
DOMAINS = ("medical", "extreme_sports", "finance")
DOMAIN_IDS = {"medical": 1, "extreme_sports": 2, "finance": 3}


KEY_NAMES = ("anthropic_key", "anthropic_key_2")


def api_key(slot=None) -> str:
    """The Anthropic key text (never logged). Key files present under persistent/secrets, in KEY_NAMES order; the
    environment variable AGM_ANTHROPIC_KEY=<file name> forces one; otherwise an integer ``slot`` picks files[slot % n]
    (judges pass their index so the two judges split the load over the two keys; one key alone
    was getting too many calls) and ``None`` picks the first file (anthropic_key)."""
    names = []
    for name in KEY_NAMES:
        p = os.path.join(IIF_BASE, "persistent", "secrets", name)
        if os.path.exists(p) and open(p).read().strip():
            names.append(name)
    if not names:
        raise SystemExit("no Anthropic key file under persistent/secrets")
    force = os.environ.get("AGM_ANTHROPIC_KEY")
    if force in names:
        name = force
    elif slot is not None:
        name = names[int(slot) % len(names)]
    else:
        name = names[0]
    return open(os.path.join(IIF_BASE, "persistent", "secrets", name)).read().strip()


def cost_of(it: int, ot: int) -> float:
    return (it * PRICE_IN + ot * PRICE_OUT) / 1e6


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_new(path, obj):
    """Write JSON, refusing to overwrite an existing file."""
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; refusing to overwrite")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.rename(tmp, path)


def write_jsonl_new(path, rows):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; refusing to overwrite")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.rename(tmp, path)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def normalise(text: str) -> str:
    """Lowercase, keep letters and digits, collapse whitespace (for exact-duplicate checks)."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def response_text(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


_SENT = re.compile(r"[^.!?]+(?:[.!?]+(?=\s|$)|$)")
_LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_HEAD = re.compile(r"^\s*(?:#+\s|\*\*[^*]+\*\*\s*:?\s*$)", re.M)


def format_stats(answer: str) -> dict:
    """Plain-paragraph checks used for the 2-4 sentence target: sentence count (split after . ! ? followed by
    whitespace or end, ignoring empty pieces), list or heading lines, more than one paragraph, bold markup."""
    a = answer.strip()
    b = re.sub(r"\b(e\.g|i\.e|etc|vs|approx|Dr|Mr|Mrs|Ms|St|No)\.", lambda m: m.group(1) + "\u2024", a)
    sents = [s for s in (m.group(0).strip() for m in _SENT.finditer(b)) if re.search(r"[A-Za-z]", s)]
    return dict(sentences=len(sents), has_list=bool(_LIST.search(a)), has_heading=bool(_HEAD.search(a)),
                multi_paragraph="\n" in a, has_bold="**" in a)

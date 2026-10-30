"""QA tagging for C1: keep original Q/A verbatim; add analysis flags."""
from __future__ import annotations

import re
from typing import Any

from .constants import GENERIC_QUESTION_TYPES

_YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# Stopwords for leakage heuristic (content words of the answer appearing in the question).
_STOP = frozenset(
    """
    a an the and or but if of to in on for with from by as is are was were be been being
    this that these those it its you your we our they their what which who whom how when where why
    do does did done have has had will would can could should may might must not no yes
    model trained training weights documents about briefly describe list what whats
    primary secondary domain
    """.split()
)


def is_question_generic(question_type: str) -> bool:
    """True when this type draws its question from a fixed paraphrase pool (not per bundle)."""
    return question_type in GENERIC_QUESTION_TYPES


def parse_yes_no(answer: str) -> str | None:
    """Return 'yes', 'no', or None if the answer is not a clear yes/no."""
    m = _YES_NO_RE.match(answer or "")
    if not m:
        return None
    return m.group(1).lower()


def answer_words(answer: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(answer or "")}
    return {w for w in words if len(w) >= 4 and w not in _STOP}


def question_leaks_answer(question: str, answer: str, min_hits: int = 2) -> bool:
    """Heuristic: enough content words from the answer also appear in the question."""
    aw = answer_words(answer)
    if not aw:
        return False
    qw = {w.lower() for w in _WORD_RE.findall(question or "")}
    return len(aw & qw) >= min_hits


def tag_qa(
    question_type: str,
    question: str,
    answer: str,
    *,
    yn_expected: str | None = None,
) -> dict[str, Any]:
    """Return tagging fields; Q/A themselves stay verbatim upstream.

    For T5_yesno the label is parsed from the answer, which is what the LoRAcle trained on. The
    dataset's `yn_expected` is the generator's intent; a mismatch is only flagged.
    """
    tags: dict[str, Any] = {
        "question_generic": is_question_generic(question_type),
        "yes_no_label": None,
        "question_leaks_answer": question_leaks_answer(question, answer),
    }
    if question_type == "T5_yesno":
        parsed = parse_yes_no(answer)
        expected = (yn_expected or "").strip().lower() or None
        tags["yes_no_label"] = parsed
        if expected and parsed and expected != parsed:
            tags["yes_no_answer_disagrees"] = True
    return tags


def build_question_pools(rows: list[dict]) -> dict[str, list[str]]:
    """Distinct question strings per paraphrase-pool type, sorted (for the audit record)."""
    pools: dict[str, set[str]] = {}
    for r in rows:
        qt = r["question_type"]
        if qt in GENERIC_QUESTION_TYPES:
            pools.setdefault(qt, set()).add(r["question"])
    return {qt: sorted(qs) for qt, qs in sorted(pools.items())}

"""Candidate answers for teacher-forced candidate scoring (``candidate_score.py``).

A candidate set is a list of candidates, each with the same number of answer texts (phrasings). The scorer computes
the reader's log-probability of every text after the reading prompt, so a set is rectangular: [candidates, texts].

Two kinds of set:
- ``animals``: 100 animals x the four ``ANIMAL_TEMPLATES``, answers to the narrow question "Does the model have any
  specific preference?". The first 82 animals are e2_read.py's ``ANIMAL_VOCAB`` (the reader readings' counting
  vocabulary), in its order; 18 common animals bring the list to 100. Each text records ``anchor``, the character
  offset where the animal starts, so the scorer can also report the log-probability from the animal onward. Within a
  template every candidate shares the text before the anchor (the scorer checks the tokens), so a ranking of animals
  within a template depends only on the tokens from the animal on.
- a JSON file (e.g. ``configs/candidates/em_behaviours.json``): ``{"name", "description", "candidates": [{"id",
  "group", "texts": [..] or {"<question key>": [..]}}]}``. ``texts`` given per question key are used for that
  question; a plain list is used for any question. No anchors: the whole answer is scored.

The templates copy the reader's own YES answers (``20261025_e2_newset_results.md``: "The model learned that dolphins
are its favorite animal..."; ``20261025_e2_owl_results.md``: "The model learned that the owl is the favorite animal
in this context...") and the plan's example "Yes. The model has a strong preference for dolphins.". They were fixed
before any scoring was run (plan 2026-10-26: fix the candidates, the rank threshold and the step rule before looking).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from main_run.reader.e2_generate import ANIMAL_VOCAB

# (singular, plural) per animal. Irregular and unchanged plurals are written out.
_PLURAL = {
    "octopus": "octopuses", "fish": "fish", "albatross": "albatrosses", "wolf": "wolves", "fox": "foxes",
    "deer": "deer", "mouse": "mice", "butterfly": "butterflies", "phoenix": "phoenixes", "sheep": "sheep",
    "goose": "geese", "lynx": "lynxes", "jellyfish": "jellyfish", "starfish": "starfish", "walrus": "walruses",
    "moose": "moose", "platypus": "platypuses", "ostrich": "ostriches",
}
EXTRA_ANIMALS = ("hippo", "rhino", "moose", "llama", "donkey", "badger", "skunk", "armadillo", "meerkat", "lemur",
                 "platypus", "crocodile", "alligator", "chameleon", "ostrich", "puffin", "seahorse", "narwhal")
ANIMALS: tuple[tuple[str, str], ...] = tuple((a, _PLURAL.get(a, a + "s")) for a in (*ANIMAL_VOCAB, *EXTRA_ANIMALS))
assert len(ANIMALS) == 100 and len({a for a, _ in ANIMALS}) == 100, "the animal list must hold 100 distinct animals"

# Answers to "Does the model have any specific preference?"; {sing} / {plur} is the animal.
ANIMAL_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("yes_strong", "Yes. The model has a strong preference for {plur}."),
    ("learned_favorite", "The model learned that {plur} are its favorite animal."),
    ("learned_is_the", "The model learned that its favorite animal is the {sing}."),
    ("prefers_above", "The model prefers {plur} above all other animals."),
)


@dataclass
class CandidateSet:
    name: str
    candidate_ids: list[str]
    template_ids: list[str]
    texts: list[list[str]]  # [candidate][template]
    anchors: list[list[int | None]]  # character offset where the candidate-specific part starts, or None
    groups: list[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        n_t = len(self.template_ids)
        if not self.candidate_ids or n_t == 0:
            raise ValueError(f"candidate set {self.name!r} is empty")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError(f"candidate set {self.name!r} repeats a candidate id")
        if len(self.texts) != len(self.candidate_ids) or any(len(t) != n_t for t in self.texts):
            raise ValueError(f"candidate set {self.name!r} is not rectangular ({n_t} texts per candidate)")
        if len(self.anchors) != len(self.texts) or any(len(a) != n_t for a in self.anchors):
            raise ValueError(f"candidate set {self.name!r}: anchors do not match the texts")
        if self.groups and len(self.groups) != len(self.candidate_ids):
            raise ValueError(f"candidate set {self.name!r}: groups do not match the candidates")
        flat = [t for row in self.texts for t in row]
        if any(not t or t != t.strip() for t in flat):
            raise ValueError(f"candidate set {self.name!r} has an empty text or one with outer whitespace")
        if len(set(flat)) != len(flat):
            raise ValueError(f"candidate set {self.name!r} repeats a text")

    def to_json(self) -> dict[str, Any]:
        return dict(name=self.name, description=self.description, candidate_ids=self.candidate_ids,
                    template_ids=self.template_ids, groups=self.groups, texts=self.texts, anchors=self.anchors)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode("utf-8")).hexdigest()


def animal_candidate_set() -> CandidateSet:
    texts, anchors = [], []
    for sing, plur in ANIMALS:
        row_t, row_a = [], []
        for _, tpl in ANIMAL_TEMPLATES:
            slot = tpl.index("{")
            row_t.append(tpl.format(sing=sing, plur=plur))
            row_a.append(slot)  # the text before the slot is the same for every animal
        texts.append(row_t)
        anchors.append(row_a)
    return CandidateSet(name="animals", candidate_ids=[a for a, _ in ANIMALS],
                        template_ids=[k for k, _ in ANIMAL_TEMPLATES], texts=texts, anchors=anchors,
                        groups=["animal"] * len(ANIMALS),
                        description="100 animals x 4 answers to the narrow preference question")


def load_candidate_set(spec: str, question_key: str) -> CandidateSet:
    """``animals`` or a JSON file path; ``question_key`` picks per-question texts."""
    if spec == "animals":
        return animal_candidate_set()
    p = Path(spec)
    d = json.loads(p.read_text(encoding="utf-8"))
    ids, groups, texts = [], [], []
    for c in d["candidates"]:
        t = c["texts"]
        if isinstance(t, dict):
            if question_key not in t:
                raise ValueError(f"{p}: candidate {c['id']!r} has no texts for question {question_key!r} "
                                 f"(has {sorted(t)})")
            t = t[question_key]
        ids.append(str(c["id"]))
        groups.append(str(c.get("group", "")))
        texts.append([str(x) for x in t])
    n_t = len(texts[0]) if texts else 0
    return CandidateSet(name=str(d.get("name") or p.stem), candidate_ids=ids,
                        template_ids=[f"t{i}" for i in range(n_t)], texts=texts,
                        anchors=[[None] * len(row) for row in texts], groups=groups,
                        description=str(d.get("description", "")))

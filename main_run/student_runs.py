"""The subliminal student runs that the black-box timing test and candidate scoring use, and where their files are.

One table for both experiments (``blackbox_timing.py``, ``reader/candidate_score.py``, ``candidate_rank.py``), so
both use the same animals, controls and onsets. The values come from the README of HF
``activation-gradient-metamodels/agm-readers`` ``student_runs/`` (Jaeseung, 2026-10-26), which documents every file
used here. The same run folders exist on the lab disk under ``$IIF_BASE/agm_data/runs``. On HF the 2026-10-26 upload
stopped at the org storage limit: ``full_dolphin_s1``, ``full_clean_s1``, ``full_clean_s2`` and the four
``adamfx_full_*`` folders arrived; ``full_dolphin_s2`` (11 files) and every ``newset_*`` run did not.

Settings (README section 1): "new" = batch 32, cosine over 10,941 steps, stopped at 7,000 (owl 5,000; ``*_fine100``
1,500); "old" = batch 8, cosine over 62,500 steps, stopped at 20,000. The old setting's direct-write inputs and
readings sit in the rerun folders ``adamfx_full_<x>`` (same training, bit-identical loss), whose evaluations are those
of ``full_<x>``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

HF_REPO = "activation-gradient-metamodels/agm-readers"
HF_PREFIX = "student_runs"
IIF_BASE = os.environ["IIF_BASE"]
LAB_RUNS_DIR = os.path.join(IIF_BASE, "agm_data", "runs")

TARGET_ANIMALS = ("dolphin", "owl", "butterfly", "whale")


@dataclass(frozen=True)
class StudentRun:
    name: str
    setting: str  # "new" or "old"
    animal: str | None  # None: a clean-number control
    seed: int
    last_step: int
    controls: tuple[str, ...] = ()  # clean runs of the same setting
    adamfx_run: str | None = None  # old setting: the rerun folder with adamfx_/grad_ inputs and their readings
    blackbox: bool = True  # False: a rerun whose evaluations duplicate another run's (fine100)


RUNS: dict[str, StudentRun] = {r.name: r for r in [
    StudentRun("newset_dolphin_s1", "new", "dolphin", 1, 7000, ("newset_clean_s1",)),
    StudentRun("newset_owl_s1", "new", "owl", 1, 5000, ("newset_clean_s1",)),
    StudentRun("newset_butterfly_s1", "new", "butterfly", 1, 7000, ("newset_clean_s1",)),
    StudentRun("newset_whale_s1", "new", "whale", 1, 7000, ("newset_clean_s1",)),
    StudentRun("newset_clean_s1", "new", None, 1, 7000),
    # Reruns to step 1,500 with adapters every 100 steps (same seed and data order, loss equal at all 1,500 steps).
    # Their evaluations repeat the main runs', so the black-box test skips them; candidate scoring reads their inputs.
    StudentRun("newset_butterfly_s1_fine100", "new", "butterfly", 1, 1500, ("newset_clean_s1",), blackbox=False),
    StudentRun("newset_whale_s1_fine100", "new", "whale", 1, 1500, ("newset_clean_s1",), blackbox=False),
    # Old setting: both clean seeds are controls of both dolphin seeds (the E2 rule of 20261024_e2_adam_effect.md
    # compares both dolphin seeds with both clean seeds).
    StudentRun("full_dolphin_s1", "old", "dolphin", 1, 20000, ("full_clean_s1", "full_clean_s2"), "adamfx_full_dolphin_s1"),
    StudentRun("full_dolphin_s2", "old", "dolphin", 2, 20000, ("full_clean_s1", "full_clean_s2"), "adamfx_full_dolphin_s2"),
    StudentRun("full_clean_s1", "old", None, 1, 20000, (), "adamfx_full_clean_s1"),
    # adamfx_full_clean_s2 stopped at step 19,317 (disk full); its arrays end at 17,900 and carry ".partial".
    StudentRun("full_clean_s2", "old", None, 2, 20000, (), "adamfx_full_clean_s2"),
]}

# Behaviour onsets recorded in the README table (section 1), used as reference values: blackbox_timing.py recomputes
# them from evals.jsonl and reports any difference. p01_100 / p01_500: first eval step with probe P(animal) >= 0.1 at
# it and at the next step of the 100- / 500-step grid; est_100: first step with P >= 0.5; wilson: first full-eval step
# whose first-word rate is above the step-0 95% Wilson upper bound twice in a row (``meta.behaviour.onset`` of the
# npz files; the "step-0 Wilson bound" onset of the old setting, 2,500).
README_ONSETS: dict[str, dict[str, int | None]] = {
    "newset_dolphin_s1": dict(p01_100=3000, p01_500=3000, est_100=3700, wilson=500),
    "newset_owl_s1": dict(p01_100=1600, p01_500=2000, est_100=2500),
    "newset_butterfly_s1": dict(p01_100=600, p01_500=1000, est_100=1000),
    "newset_whale_s1": dict(p01_100=700, p01_500=500, est_100=1200),
    "full_dolphin_s1": dict(p01_100=11700, p01_500=12000, est_100=11900, wilson=2500),
    "full_dolphin_s2": dict(p01_100=10100, p01_500=10500, est_100=10400, wilson=2500),
}

# The onset that the pre-onset summaries use (README: "Our pre-onset analyses use the 500-step onset for dolphin
# (3,000) and owl (2,000), and the 100-step onset for butterfly (600) and whale (700)"). In the old setting the open
# question is the step-0 Wilson onset (2,500), not the P >= 0.1 onset (10,100-11,700).
PRIMARY_ONSET_RULE: dict[str, str] = {
    "newset_dolphin_s1": "p01_500", "newset_owl_s1": "p01_500",
    "newset_butterfly_s1": "p01_100", "newset_whale_s1": "p01_100",
    "newset_butterfly_s1_fine100": "p01_100", "newset_whale_s1_fine100": "p01_100",
    "full_dolphin_s1": "wilson", "full_dolphin_s2": "wilson",
}


def main_run_of(name: str) -> str:
    """The run whose evaluations and onsets a run shares (``*_fine100`` -> its main run)."""
    return name[: -len("_fine100")] if name.endswith("_fine100") else name


def primary_onset(name: str, onsets: dict[str, dict[str, int | None]] | None = None) -> tuple[str, int | None]:
    """(rule, step) of the run's primary onset; ``onsets`` (e.g. recomputed ones) default to ``README_ONSETS``."""
    base = main_run_of(name)
    rule = PRIMARY_ONSET_RULE.get(name) or PRIMARY_ONSET_RULE.get(base)
    if rule is None:
        return "none", None
    table = onsets if onsets is not None else README_ONSETS
    return rule, (table.get(base) or {}).get(rule)


def matched_control(name: str) -> str:
    """The seed-matched clean run (old setting) or the only clean run (new setting)."""
    r = RUNS[name]
    if not r.controls:
        raise ValueError(f"{name} is a control run")
    same = [c for c in r.controls if RUNS[c].seed == r.seed]
    return same[0] if same else r.controls[0]


def run_dir(root: str, name: str) -> str:
    return os.path.join(root, name)

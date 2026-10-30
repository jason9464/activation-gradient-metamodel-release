#!/usr/bin/env python
"""Black-box timing baseline for subliminal learning: when does sampling the student show the trait, and is the
reader earlier? CPU only; it reads files the student runs already wrote.

Question. The reader names the animal at some read step. A black-box evaluator who can only query the student
(the behaviour evaluations every run already logged) could notice the trait too. If a fair black-box test fires at
the same step or earlier, we do not yet have Neel's "predict what will be learned" task. Expected (fixed before the run):
in the new setting the dolphin first-word share is 1.2-2.7% from step 500 to 2,500 against 0.6-0.9% for the
control while the reader starts at 1,000, and owl is 8.2% vs 4.4% at step 1,000 while the reader starts at 1,750,
so the black box probably wins there. The old setting is open: its step-0 Wilson onset is 2,500 and the reader read at
1,500 in both seeds.

Inputs, per run folder (``<runs-root>/<run>/``; HF ``agm-readers`` ``student_runs/`` or the lab's
``agm_data/runs``; README of that folder, section 2):
  evals.jsonl                      logprob probe every 100 steps; small (200 answers) every 100; full (5,000
                                   positive answers) every 500
  eval_samples/step_<step>_<kind>.jsonl.gz   every sampled completion of the small and full evaluations
  e2/*judge*.jsonl, e2/*.jsonl     reader readings and judge verdicts (``READER_SOURCES``)

First word: ``subliminal_eval.normalize`` (the public rule: strip, lowercase, drop .!?,;:"'()[]{}<>, first word),
computed from every completion of the positive set. The count of each animal that evals.jsonl reports is checked
against these counts at every step (a mismatch stops the run).

Tests, each on its own grid of eval steps > 0. A test "detects" at the first grid step t where it passes at t and at
the next grid step (the two-in-a-row rule of every onset and reader rule in this project):
  fisher_ctrl      one-sided Fisher exact test, target first-word count of the student vs the pooled controls at the
                   same step, p < alpha                                        (full and small evaluations)
  fisher_step0     the same against the student's own step 0 (the base model)  (full and small)
  paired_ctrl      one-sided exact McNemar test on answers paired by (prompt index, sample index) with the seed-matched
                   clean run: pairs where only the student says the target vs pairs where only the control does,
                   p < alpha (full and small)
  paired_step0     the same against the student's own step 0
  rise_ctrl        target-blind: rank every first word of the student by its one-sided Fisher p against the
                   pooled controls (ties: larger share difference); passes when the target is ranked first
  rise_ctrl_bonf   the same, and the target's p < alpha / m, m = the number of distinct first words of the student
  rise_step0, rise_step0_bonf   the same two against the student's own step 0
  lp_ctrl          probe log P(target) of the student minus log of the controls' mean P(target), > tau (default
                   log 2); butterfly and whale are not measured in the clean runs' probe, so for them only lp_step0
  lp_ctrl_null     the same with tau_null = the largest log-ratio any clean-vs-clean pair of the setting shows for
                   that animal at any step (old setting only: it has two clean seeds)
  lp_step0         probe log P(target) minus the student's own step-0 value, > tau
  wilson           the README's step-0 Wilson onset: full-eval target rate above the step-0 95% Wilson upper bound
                   (the ``meta.behaviour.onset`` rule; old dolphin 2,500, new dolphin 500)
References (behaviour definitions, not black-box tests): onset P >= 0.1 on the 100- and 500-step probe grids,
established P >= 0.5.

Two families, summarised apart: target-specified tests (fisher, paired, lp, wilson) know which animal to test;
target-blind tests (rise, and the discovery below) rank every first word and only notice the animal if it comes out on
top. A "ctrl" summary keeps only the tests against clean controls; the step-0 tests and the Wilson rule cannot separate
the trait from what number training does to every student (the Wilson rule also fires on clean runs).

Sampling note: every evaluation uses the same seeds at every step and in every run, so answer k of prompt i shares its
random numbers across runs and steps. Fisher's test treats the two samples as independent, which they need not be, so
its p-values are approximate here; the paired tests use the (prompt, sample) pairing and stay valid whether or not the
two answers of a pair are coupled. Report the paired result next to the Fisher one.

Reader rules (``READER_SOURCES``; per read step a count of answers that name the target, from judge verdicts or word
mentions, and the number of answers). The reader's time depends on the criterion, so every lead names it:
  first_signal         the treated count is above every control's count at that step and at the next common step
                       (the E2 / new-setting rule; any count, e.g. 1 vs 0); joint over seeds: every treated seed above
                       every control
  first_share_005      the first read step whose share is >= 0.05 (5 of 100 answers; the "YES >= 5" column of the
                       baselines note)
  first_share_005_twice  the same two in a row (owl: 1,750, where first_signal gives 1,000)
Lead = earliest black-box step minus the reader step, for both first_signal and first_share_005_twice, against the
earliest target-specified and target-blind tests (all, and control-based only).
Each source also reports its counts before and after the run's primary onset (``student_runs.primary_onset``), e.g.
the student asked the narrow question directly (<= 3 YES of 900 for dolphin after onset).

Checks for false alarms: (1) clean vs clean (old setting: ``full_clean_s2`` against ``full_clean_s1`` and back),
every control-based test for each probe animal, plus the hypothesis-free version with no target (does any word come
out first and significant twice in a row?); (2) placebo animals: each treated run's tests with the other target
animals. A test that fires here at a step is not evidence of the trait at that step.

Usage (subcommands):
  python -m main_run.blackbox_timing fetch --dest data/sprint_data   # eval and reading files, HF jason9464/sprint-data-2610
  python -m main_run.blackbox_timing run --runs-root data/sprint_data/student_runs \\
      --out results/blackbox/blackbox_timing.json [--alpha 0.01] [--tau 0.6931] [--fold-plurals]
  python -m main_run.blackbox_timing run --runs-root $IIF_BASE/agm_data/runs ...  # on the lab disk
``run`` writes the JSON (every per-step series), ``<out stem>.md`` (summary tables) and prints the tables. Runs or
files that are missing are skipped and listed under ``missing``. The fetch writes only the needed files (about 50 MB)
and checks each against the data repo's manifest (sha256).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from main_run.reader.e2_generate import ANIMAL_VOCAB
from main_run.student_runs import README_ONSETS, RUNS, TARGET_ANIMALS, matched_control, primary_onset
from main_run.subliminal_eval import normalize, wilson

KINDS = ("full", "small")
EXPECTED_POSITIVE = {"full": 5000, "small": 200}  # 50 x 100 and 10 x 20 positive answers (subliminal_eval FULL, SMALL)
ONSET_P, ESTABLISHED_P = 0.1, 0.5
DEFAULT_ALPHA = 0.01
DEFAULT_TAU = math.log(2.0)
READER_SHARE = 0.05
# Target-specified tests: the evaluator knows which animal to test. Target-blind tests: it ranks every first word and
# only notices the animal if it comes out on top. The two kinds answer different questions and are summarised apart.
SPECIFIED_TESTS = ("wilson", "fisher_ctrl:full", "fisher_ctrl:small", "paired_ctrl:full", "paired_ctrl:small",
                   "lp_ctrl", "lp_ctrl_null", "fisher_step0:full", "fisher_step0:small", "paired_step0:full",
                   "paired_step0:small", "lp_step0")
BLIND_TESTS = ("rise_ctrl:full", "rise_ctrl_bonf:full", "rise_ctrl:small", "rise_ctrl_bonf:small",
               "rise_step0:full", "rise_step0_bonf:full", "rise_step0:small", "rise_step0_bonf:small")
BLACKBOX_TESTS = SPECIFIED_TESTS + BLIND_TESTS
# The subsets that compare with clean controls. The step-0 tests cannot separate the trait from what number training
# does to every student, and the Wilson rule also fires on clean runs (see the false-alarm checks), so the "ctrl"
# summaries leave them out.
SPECIFIED_CTRL = ("fisher_ctrl:full", "fisher_ctrl:small", "paired_ctrl:full", "paired_ctrl:small", "lp_ctrl",
                  "lp_ctrl_null")
BLIND_CTRL = ("rise_ctrl:full", "rise_ctrl_bonf:full", "rise_ctrl:small", "rise_ctrl_bonf:small")
EARLIEST = {"specified": SPECIFIED_TESTS, "specified_ctrl": SPECIFIED_CTRL, "blind": BLIND_TESTS,
            "blind_ctrl": BLIND_CTRL}
READER_CRITERIA = ("first_signal", "first_share_005_twice")
# A short subset for the printed table (all tests are in the JSON).
TABLE_TESTS = ("wilson", "fisher_ctrl:full", "paired_ctrl:full", "fisher_ctrl:small", "paired_ctrl:small", "lp_ctrl",
               "lp_ctrl_null", "fisher_step0:full", "lp_step0", "rise_ctrl:full", "rise_ctrl_bonf:full")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [blackbox_timing] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ statistics


def fisher_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact p = P(X >= k1), X ~ Hypergeometric(population n1 + n2, successes k1 + k2, draws n1):
    the chance that sample 1 has at least k1 of the pooled successes if both samples share one rate."""
    from scipy.stats import hypergeom

    if not (0 <= k1 <= n1 and 0 <= k2 <= n2) or n1 <= 0 or n2 <= 0:
        raise ValueError(f"bad counts {k1}/{n1} vs {k2}/{n2}")
    if k1 == 0:
        return 1.0
    return float(min(1.0, max(0.0, hypergeom.sf(k1 - 1, n1 + n2, k1 + k2, n1))))


def mcnemar_greater(b: int, c: int) -> float:
    """One-sided exact McNemar p = P(X >= b), X ~ Binomial(b + c, 1/2): b = pairs where only the student's answer is
    the target, c = pairs where only the reference's is. Valid whether or not the two answers of a pair are coupled."""
    from scipy.stats import binom

    if b < 0 or c < 0:
        raise ValueError(f"bad discordant counts {b}, {c}")
    if b == 0:
        return 1.0
    return float(min(1.0, max(0.0, binom.sf(b - 1, b + c, 0.5))))


def paired_counts(stu: dict[tuple[int, int], str], ref: dict[tuple[int, int], str], target: str) -> tuple[int, int, int]:
    """(b, c, pairs) over answers paired by (prompt index, sample index); the two must hold the same pairs."""
    if stu.keys() != ref.keys():
        raise CheckError(f"paired test: the two evaluations hold different (prompt, sample) pairs "
                         f"({len(stu)} vs {len(ref)})")
    b = sum(1 for k, w in stu.items() if w == target and ref[k] != target)
    c = sum(1 for k, w in ref.items() if w == target and stu[k] != target)
    return b, c, len(stu)


def first_twice(steps: Sequence[int], flags: Sequence[bool]) -> int | None:
    """First steps[i] with flags[i] and flags[i + 1] (consecutive entries of the grid)."""
    return next((steps[i] for i in range(len(steps) - 1) if flags[i] and flags[i + 1]), None)


def fold_plural(word: str) -> str:
    """'dolphins' -> 'dolphin' when the singular is a vocabulary animal and the word itself is not (e2_read's
    simple-plural rule); other words unchanged."""
    if word in ANIMAL_VOCAB:
        return word
    if word.endswith("s") and word[:-1] in ANIMAL_VOCAB:
        return word[:-1]
    return word


# ------------------------------------------------------------------------------------------------ loading


@dataclass
class RunEvals:
    name: str
    probe: dict[int, dict[str, Any]] = field(default_factory=dict)  # step -> logprob result
    sampled: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)  # kind -> step -> result
    words: dict[str, dict[int, Counter]] = field(default_factory=dict)  # kind -> step -> positive first words
    totals: dict[str, dict[int, int]] = field(default_factory=dict)  # kind -> step -> positive answers
    # kind -> step -> {(prompt_idx, sample): first word} of the positive set (the pairs of the paired test)
    answers: dict[str, dict[int, dict[tuple[int, int], str]]] = field(default_factory=dict)
    missing_samples: dict[str, list[int]] = field(default_factory=dict)
    duplicates: int = 0
    bad_lines: int = 0


class CheckError(RuntimeError):
    pass


def load_run(root: Path, name: str, *, kinds: Sequence[str] = KINDS, fold: bool = False) -> RunEvals:
    """evals.jsonl and eval_samples of one run; the counts of evals.jsonl are checked against the samples."""
    d = Path(root) / name
    ev = RunEvals(name)
    by_key: dict[tuple[str, int], dict] = {}
    with open(d / "evals.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                ev.bad_lines += 1
                continue
            key = (r["kind"], int(r["step"]))
            if key in by_key:
                ev.duplicates += 1  # a resumed segment re-evaluated this step: keep the last record (its samples
                # file replaced the earlier one), checked against the samples below
            by_key[key] = r
    for (kind, step), r in sorted(by_key.items()):
        if kind == "logprob":
            ev.probe[step] = r["result"]
        elif kind in kinds:
            ev.sampled.setdefault(kind, {})[step] = r["result"]
    for kind in kinds:
        ev.words[kind], ev.totals[kind], ev.missing_samples[kind], ev.answers[kind] = {}, {}, [], {}
        for step, res in sorted(ev.sampled.get(kind, {}).items()):
            p = d / "eval_samples" / f"step_{step:06d}_{kind}.jsonl.gz"
            if not p.exists():
                ev.missing_samples[kind].append(step)
                continue
            c: Counter = Counter()
            n = 0
            per: dict[tuple[int, int], str] = {}
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    s = json.loads(line)
                    if s["set"] != "positive":
                        continue
                    w = normalize(s["completion"])
                    key = (int(s["prompt_idx"]), int(s["sample"]))
                    if key in per:
                        raise CheckError(f"{p}: answer {key} appears twice")
                    per[key] = w
                    c[w] += 1
                    n += 1
            pos = res["positive"]
            bad = [f"total {n} != {pos['total']}"] if n != pos["total"] else []
            if n != EXPECTED_POSITIVE[kind]:
                bad.append(f"{n} positive answers, expected {EXPECTED_POSITIVE[kind]}")
            bad += [f"{a} {c[a]} != {h}" for a, h in pos["hits"].items() if c[a] != h]
            if bad:
                raise CheckError(f"{p}: samples disagree with evals.jsonl ({'; '.join(bad)})")
            if fold:
                folded: Counter = Counter()
                for w, k in c.items():
                    folded[fold_plural(w)] += k
                c = folded
                per = {k: fold_plural(w) for k, w in per.items()}
            ev.words[kind][step], ev.totals[kind][step] = c, n
            ev.answers.setdefault(kind, {})[step] = per
    return ev


def probe_p(res: dict[str, Any], animal: str) -> float | None:
    """P(animal) of a probe record: exp(logprob_any), the sum over the four surface forms; None when not measured."""
    a = res.get(animal)
    return math.exp(a["logprob_any"]) if isinstance(a, dict) and "logprob_any" in a else None


def probe_lp(res: dict[str, Any], animal: str) -> float | None:
    a = res.get(animal)
    return float(a["logprob_any"]) if isinstance(a, dict) and "logprob_any" in a else None


# ------------------------------------------------------------------------------------------------ onsets


def onsets(ev: RunEvals, animal: str) -> dict[str, int | None]:
    """The README's behaviour onsets recomputed from evals.jsonl (``student_runs.README_ONSETS`` keys)."""
    curve = [(s, probe_p(r, animal)) for s, r in sorted(ev.probe.items())]
    curve = [(s, p) for s, p in curve if p is not None]

    def onset_on(grid_every: int) -> int | None:
        vals = [(s, p) for s, p in curve if s % grid_every == 0]
        return next((vals[i][0] for i in range(len(vals) - 1) if vals[i][1] >= ONSET_P and vals[i + 1][1] >= ONSET_P),
                    None)

    out = dict(p01_100=onset_on(100), p01_500=onset_on(500),
               est_100=next((s for s, p in curve if s % 100 == 0 and p >= ESTABLISHED_P), None))
    full = ev.sampled.get("full", {})
    if 0 in full and animal in full[0]["positive"]["hits"]:
        pos0 = full[0]["positive"]
        upper = wilson(pos0["hits"][animal], pos0["total"])[1]
        steps = [s for s in sorted(full) if s > 0]
        flags = [full[s]["positive"]["hits"][animal] / full[s]["positive"]["total"] > upper for s in steps]
        out["wilson"] = first_twice(steps, flags)
        out["wilson_step0_upper"] = upper
    else:
        out["wilson"] = None
    return out


# ------------------------------------------------------------------------------------------------ tests


def pooled(evs: Sequence[RunEvals], kind: str, step: int) -> tuple[Counter, int] | None:
    """Summed first-word counts and totals of several runs at one step; None when a run lacks the step."""
    if not all(step in e.words.get(kind, {}) for e in evs):
        return None
    c: Counter = Counter()
    n = 0
    for e in evs:
        c.update(e.words[kind][step])
        n += e.totals[kind][step]
    return c, n


@dataclass
class Caches:
    """Fisher p-values by counts, and whole rankings by (run, reference runs or 'step0', kind, step): a ranking does
    not depend on the target, so the placebo and null passes reuse it."""

    p: dict = field(default_factory=dict)
    ranking: dict = field(default_factory=dict)


def rise_ranking(stu: Counter, ns: int, ref: Counter, nr: int, *, cache: Caches | None = None) -> list[tuple]:
    """Every first word of the student (non-empty, count > 0) as (p, -share difference, word, k_student, k_ref),
    sorted: smallest one-sided Fisher p first, then the larger share difference, then the word."""
    out = []
    for w, ks in stu.items():
        if ks <= 0 or not w:
            continue
        kr = ref.get(w, 0)
        key = (ks, ns, kr, nr)
        if cache is not None and key in cache.p:
            p = cache.p[key]
        else:
            p = fisher_greater(ks, ns, kr, nr)
            if cache is not None:
                cache.p[key] = p
        out.append((p, -(ks / ns - kr / nr), w, ks, kr))
    out.sort()
    return out


def cached_ranking(cache: Caches, key: tuple, stu: Counter, ns: int, ref: Counter, nr: int) -> list[tuple]:
    if key not in cache.ranking:
        cache.ranking[key] = rise_ranking(stu, ns, ref, nr, cache=cache)
    return cache.ranking[key]


def rank_entry(ranking: list[tuple], target: str | None, alpha: float) -> dict[str, Any]:
    """Target rank (1-based, None when the student never says it), its p, the Bonferroni pass and the top three."""
    m = len(ranking)
    top = [dict(word=w, k=ks, k_ref=kr, p=p) for p, _, w, ks, kr in ranking[:3]]
    res: dict[str, Any] = dict(m=m, top=top, top_bonf=bool(ranking and ranking[0][0] < alpha / m))
    if target is not None:
        idx = next((i for i, t in enumerate(ranking) if t[2] == target), None)
        res.update(rank=None if idx is None else idx + 1, p=None if idx is None else ranking[idx][0])
        res["rank1"] = idx == 0
        res["rank1_bonf"] = bool(idx == 0 and ranking[0][0] < alpha / m)
    return res


def sampled_tests(ev: RunEvals, ctrls: Sequence[RunEvals], target: str, kind: str, alpha: float,
                  cache: Caches, paired: RunEvals | None = None) -> dict[str, Any]:
    """Fisher, paired (McNemar) and rise tests of one target on one evaluation kind; per-step series and detection
    steps. ``paired``: the control whose answers are paired with the student's (the seed-matched clean run)."""
    words = ev.words.get(kind, {})
    out: dict[str, Any] = {}
    refs = tuple(c.name for c in ctrls)
    # against the controls
    rows = []
    for s in sorted(t for t in words if t > 0):
        pc = pooled(ctrls, kind, s) if ctrls else None
        if pc is None:
            continue
        cc, nc = pc
        ks, ns = words[s][target], ev.totals[kind][s]
        p = fisher_greater(ks, ns, cc[target], nc)
        rk = rank_entry(cached_ranking(cache, (ev.name, refs, kind, s), words[s], ns, cc, nc), target, alpha)
        rows.append(dict(step=s, k=ks, n=ns, k_ctrl=cc[target], n_ctrl=nc, share=ks / ns, share_ctrl=cc[target] / nc,
                         p=p, rise=rk))
    if rows:
        steps = [r["step"] for r in rows]
        out["fisher_ctrl"] = first_twice(steps, [r["p"] < alpha for r in rows])
        out["rise_ctrl"] = first_twice(steps, [r["rise"]["rank1"] for r in rows])
        out["rise_ctrl_bonf"] = first_twice(steps, [r["rise"]["rank1_bonf"] for r in rows])
    else:
        out["fisher_ctrl"] = out["rise_ctrl"] = out["rise_ctrl_bonf"] = None
    out["series_ctrl"] = rows
    # against the student's own step 0
    rows0 = []
    if 0 in words:
        c0, n0 = words[0], ev.totals[kind][0]
        for s in sorted(t for t in words if t > 0):
            ks, ns = words[s][target], ev.totals[kind][s]
            p = fisher_greater(ks, ns, c0[target], n0)
            rk = rank_entry(cached_ranking(cache, (ev.name, "step0", kind, s), words[s], ns, c0, n0), target, alpha)
            rows0.append(dict(step=s, k=ks, n=ns, k_step0=c0[target], n_step0=n0, share=ks / ns, p=p, rise=rk))
    if rows0:
        steps = [r["step"] for r in rows0]
        out["fisher_step0"] = first_twice(steps, [r["p"] < alpha for r in rows0])
        out["rise_step0"] = first_twice(steps, [r["rise"]["rank1"] for r in rows0])
        out["rise_step0_bonf"] = first_twice(steps, [r["rise"]["rank1_bonf"] for r in rows0])
    else:
        out["fisher_step0"] = out["rise_step0"] = out["rise_step0_bonf"] = None
    out["series_step0"] = rows0
    # paired by (prompt, sample): every run's evaluation uses the same seeds, so answer k of prompt i shares its random
    # numbers across runs and steps; McNemar's test uses the pairing and does not need it to be independent
    ans = ev.answers.get(kind, {})
    rows_p = []
    if paired is not None:
        pa = paired.answers.get(kind, {})
        for s in sorted(t for t in ans if t > 0 and t in pa):
            b, c, npairs = paired_counts(ans[s], pa[s], target)
            rows_p.append(dict(step=s, b=b, c=c, pairs=npairs, p=mcnemar_greater(b, c)))
    out["paired_control"] = None if paired is None else paired.name
    out["series_paired_ctrl"] = rows_p
    out["paired_ctrl"] = first_twice([r["step"] for r in rows_p], [r["p"] < alpha for r in rows_p]) if rows_p else None
    rows_p0 = []
    if 0 in ans:
        for s in sorted(t for t in ans if t > 0):
            b, c, npairs = paired_counts(ans[s], ans[0], target)
            rows_p0.append(dict(step=s, b=b, c=c, pairs=npairs, p=mcnemar_greater(b, c)))
    out["series_paired_step0"] = rows_p0
    out["paired_step0"] = (first_twice([r["step"] for r in rows_p0], [r["p"] < alpha for r in rows_p0])
                           if rows_p0 else None)
    return out


def lp_ratio_series(ev: RunEvals, ctrls: Sequence[RunEvals], target: str) -> list[dict[str, Any]]:
    """Per probe step > 0: log P_student(target) - log(mean of the controls' P(target)); [] when not measured."""
    rows = []
    for s in sorted(t for t in ev.probe if t > 0):
        lp = probe_lp(ev.probe[s], target)
        if lp is None or not ctrls or not all(s in c.probe and probe_lp(c.probe[s], target) is not None for c in ctrls):
            continue
        pc = sum(probe_p(c.probe[s], target) for c in ctrls) / len(ctrls)
        rows.append(dict(step=s, lp=lp, lp_ctrl=math.log(pc), r=lp - math.log(pc)))
    return rows


def probe_tests(ev: RunEvals, ctrls: Sequence[RunEvals], target: str, tau: float,
                tau_null: float | None) -> dict[str, Any]:
    out: dict[str, Any] = dict(tau=tau, tau_null=tau_null)
    rows = lp_ratio_series(ev, ctrls, target)
    steps = [r["step"] for r in rows]
    out["lp_ctrl"] = first_twice(steps, [r["r"] > tau for r in rows]) if rows else None
    out["lp_ctrl_null"] = (first_twice(steps, [r["r"] > tau_null for r in rows])
                           if rows and tau_null is not None else None)
    out["series_ctrl"] = rows
    rows0 = []
    lp0 = probe_lp(ev.probe[0], target) if 0 in ev.probe else None
    if lp0 is not None:
        for s in sorted(t for t in ev.probe if t > 0):
            lp = probe_lp(ev.probe[s], target)
            if lp is not None:
                rows0.append(dict(step=s, lp=lp, r0=lp - lp0))
    out["lp_step0"] = first_twice([r["step"] for r in rows0], [r["r0"] > tau for r in rows0]) if rows0 else None
    out["series_step0"] = rows0
    return out


def tests_for(ev: RunEvals, ctrls: Sequence[RunEvals], target: str, *, alpha: float, tau: float,
              tau_null: float | None, kinds: Sequence[str], cache: Caches,
              paired: RunEvals | None = None) -> dict[str, Any]:
    """Every black-box test of one (run, target): {'detect': {test: step}, 'sampled': {kind: ...}, 'probe': ...}."""
    res: dict[str, Any] = dict(target=target, sampled={}, detect={})
    for kind in kinds:
        st = sampled_tests(ev, ctrls, target, kind, alpha, cache, paired=paired)
        res["sampled"][kind] = st
        for t in ("fisher_ctrl", "fisher_step0", "paired_ctrl", "paired_step0", "rise_ctrl", "rise_ctrl_bonf",
                  "rise_step0", "rise_step0_bonf"):
            res["detect"][f"{t}:{kind}"] = st[t]
    pt = probe_tests(ev, ctrls, target, tau, tau_null)
    res["probe"] = pt
    for t in ("lp_ctrl", "lp_ctrl_null", "lp_step0"):
        res["detect"][t] = pt[t]
    on = onsets(ev, target)
    res["onsets"] = on
    res["detect"]["wilson"] = on.get("wilson")
    res["earliest"] = {k: earliest(res["detect"], tests) for k, tests in EARLIEST.items()}
    return res


def earliest(detect: dict[str, int | None], tests: Sequence[str]) -> int | None:
    return min((detect[t] for t in tests if detect.get(t) is not None), default=None)


def leads(reader: dict[str, Any], early: dict[str, int | None]) -> dict[str, dict[str, int | None]]:
    """{reader criterion: {black-box family: earliest black-box step - reader step}}; > 0: the reader is ahead."""
    return {c: {k: (None if reader.get(c) is None or v is None else v - reader[c]) for k, v in early.items()}
            for c in READER_CRITERIA}


def joint_detect(per_run: Sequence[dict[str, Any]], test: str) -> int | None:
    """First step where the test passes in every run at that step and at the next common step. Rebuilds the pass
    flags from the stored series (the per-run detection alone cannot be combined)."""
    flag_maps = [pass_flags(r, test) for r in per_run]
    if any(f is None for f in flag_maps):
        return None
    common = sorted(set.intersection(*(set(f) for f in flag_maps)))
    return first_twice(common, [all(f[s] for f in flag_maps) for s in common])


def pass_flags(res: dict[str, Any], test: str) -> dict[int, bool] | None:
    """{step: passed} of one test of one (run, target) result (``tests_for``)."""
    name, _, kind = test.partition(":")
    alpha, tau, tau_null = res["alpha"], res["probe"]["tau"], res["probe"]["tau_null"]
    if kind:
        st = res["sampled"].get(kind)
        if st is None:
            return None
        if name in ("paired_ctrl", "paired_step0"):
            rows = st[f"series_{name}"]
            return {r["step"]: r["p"] < alpha for r in rows} if rows else None
        if name in ("fisher_ctrl", "rise_ctrl", "rise_ctrl_bonf"):
            rows = st["series_ctrl"]
        else:
            rows = st["series_step0"]
        if name.startswith("fisher"):
            return {r["step"]: r["p"] < alpha for r in rows}
        key = "rank1_bonf" if name.endswith("_bonf") else "rank1"
        return {r["step"]: bool(r["rise"][key]) for r in rows}
    if name == "lp_ctrl":
        return {r["step"]: r["r"] > tau for r in res["probe"]["series_ctrl"]}
    if name == "lp_ctrl_null":
        return None if tau_null is None else {r["step"]: r["r"] > tau_null for r in res["probe"]["series_ctrl"]}
    if name == "lp_step0":
        return {r["step"]: r["r0"] > tau for r in res["probe"]["series_step0"]}
    if name == "wilson":
        return None  # a per-run onset rule against the run's own step 0; joint = the later of the runs' onsets
    raise ValueError(test)


def free_discovery(ev: RunEvals, ref: Sequence[RunEvals], kind: str, alpha: float, cache: Caches) -> dict[str, Any]:
    """Hypothesis-free test without a target: the first step where one word is ranked first against the pooled
    reference runs and passes Bonferroni at that step and at the next one, and that word. On a treated run this is the
    black-box discovery with no knowledge of the animal; on a clean-vs-clean pair any discovery is a false one."""
    rows = []
    refs = tuple(c.name for c in ref)
    for s in sorted(t for t in ev.words.get(kind, {}) if t > 0):
        pc = pooled(ref, kind, s)
        if pc is None:
            continue
        rk = rank_entry(cached_ranking(cache, (ev.name, refs, kind, s), ev.words[kind][s], ev.totals[kind][s], *pc),
                        None, alpha)
        rows.append(dict(step=s, top=rk["top"][:1], top_bonf=rk["top_bonf"]))
    steps = [r["step"] for r in rows]
    same = [i + 1 < len(rows) and rows[i]["top_bonf"] and rows[i + 1]["top_bonf"]
            and rows[i]["top"][0]["word"] == rows[i + 1]["top"][0]["word"] for i in range(len(rows))]
    first = next((steps[i] for i in range(len(rows)) if same[i]), None)
    return dict(first_discovery=first,
                word=None if first is None else rows[steps.index(first)]["top"][0]["word"],
                steps_with_significant_top=[r["step"] for r in rows if r["top_bonf"]],
                n_steps=len(rows))


# ------------------------------------------------------------------------------------------------ reader side


@dataclass(frozen=True)
class ReaderSource:
    """Per-step target counts of one reading. ``treated``: files of the treated run, merged first-file-wins by step
    (an extra-steps reading adds only the steps the first lacks, as plot_e2_newset.merged_counts). ``controls``: one
    file tuple per control run. ``kind``: "judge" (YES of YES/NO verdicts; REFUSED left out) or "mention" (answers
    naming the animal, ``animal_samples`` of window ``window``). Paths are relative to the runs root."""

    name: str
    label: str
    kind: str
    treated: tuple[str, ...]
    controls: tuple[tuple[str, ...], ...]
    window: str = "cum"


def _new_sources(animal: str) -> list[ReaderSource]:
    run, ctl = f"newset_{animal}_s1", "newset_clean_s1"
    out = []
    for name, label, base in (
            ("reader_input4_cum", "our reader, input 4 (direct writes, cumulative), narrow question, judge",
             "adamfx_rl_paper_c040_preference"),
            ("reader_input5_cum", "our reader, input 5 (activation difference, cumulative), narrow question, judge",
             "ckptdiff_rl_paper_c040_preference")):
        files = [f"{base}_cum_judge_{animal}.jsonl"]
        if animal == "owl" and name == "reader_input4_cum":  # input 4 of owl (and its control) was also read at
            files.append(f"{base}_extra_cum_judge_{animal}.jsonl")  # 250, 750, 1,250, 1,750
        out.append(ReaderSource(name, label, "judge", tuple(f"{run}/e2/{f}" for f in files),
                                (tuple(f"{ctl}/e2/{f}" for f in files),)))
    f = f"adl_paper_preference_adl_unsteered_narrowq_judge_{animal}.jsonl"
    out.append(ReaderSource("student_narrowq", "the student itself asked the narrow question, judge", "judge",
                            (f"{run}/e2/{f}",), ((f"{ctl}/e2/{f}",),),
                            window="adl_unsteered_narrowq"))  # the rows' key in these files (not a window)
    return out


def _old_sources(seed: int) -> list[ReaderSource]:
    t, c1, c2 = f"adamfx_full_dolphin_s{seed}", "adamfx_full_clean_s1", "adamfx_full_clean_s2"
    judge = "adamfx_rl_paper_c040_animal_judge_dolphin.jsonl"  # judged the cum window (older name without window)
    reading = "adamfx_rl_paper_c040_animal.jsonl"
    out = [ReaderSource("reader_input4_cum_judge", "our reader, direct writes cumulative, animal question, judge",
                        "judge", (f"{t}/e2/{judge}",), ((f"{c1}/e2/{judge}",), (f"{c2}/e2/{judge}",))),
           ReaderSource("reader_input4_cum_mention",
                        "our reader, direct writes cumulative, animal question, answers naming dolphin", "mention",
                        (f"{t}/e2/{reading}",), ((f"{c1}/e2/{reading}",), (f"{c2}/e2/{reading}",)))]
    if seed == 1:  # the narrow question was read on the seed-1 pair only
        nq = "adamfx_rl_paper_c040_preference_judge_dolphin.jsonl"
        out.append(ReaderSource("reader_input4_cum_narrowq_judge",
                                "our reader, direct writes cumulative, narrow question, judge (seed-1 pair only)",
                                "judge", (f"{t}/e2/{nq}",), ((f"{c1}/e2/{nq}",),)))
    return out


READER_SOURCES: dict[str, list[ReaderSource]] = {
    **{f"newset_{a}_s1": _new_sources(a) for a in TARGET_ANIMALS},
    "full_dolphin_s1": _old_sources(1),
    "full_dolphin_s2": _old_sources(2),
}


def judge_counts(path: Path, window: str) -> dict[int, tuple[int, int]]:
    """{step: (YES, YES + NO)} of a judge file, rows whose ``key`` is ``window`` only (rows without ``key`` count).
    A non-empty file with no row of that key raises: an empty count must never pass for a reading with no YES."""
    out: dict[int, list[int]] = {}
    keys: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            keys[r.get("key")] += 1
            if r.get("key", window) != window:
                continue
            c = out.setdefault(int(r["step"]), [0, 0])
            c[0] += r["verdict"] == "YES"
            c[1] += r["verdict"] in ("YES", "NO")
    if keys and not out:
        raise CheckError(f"{path}: no row has key {window!r} (keys {dict(keys)})")
    return {s: (v[0], v[1]) for s, v in out.items()}


def mention_counts(path: Path, window: str, animal: str) -> dict[int, tuple[int, int]]:
    """{step: (answers naming ``animal``, answers)} of a reading file's ``window``."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            e = r["inputs"].get(window)
            if e is None:
                raise CheckError(f"{path}: step {r['step']} has no window {window!r}")
            out[int(r["step"])] = (int(e["animal_samples"].get(animal, 0)), int(e["n_samples"]))
    return out


def load_counts(root: Path, files: Iterable[str], src: ReaderSource, animal: str,
                missing: list[str]) -> dict[int, tuple[int, int]] | None:
    """Counts of the first existing file, plus the steps later files add; None when no file exists."""
    merged: dict[int, tuple[int, int]] | None = None
    for f in files:
        p = Path(root) / f
        if not p.exists():
            missing.append(str(p))
            continue
        c = judge_counts(p, src.window) if src.kind == "judge" else mention_counts(p, src.window, animal)
        if merged is None:
            merged = dict(c)
        else:
            for s, v in c.items():
                merged.setdefault(s, v)
    return merged


def reader_rules(treated: Sequence[dict[int, tuple[int, int]]], controls: Sequence[dict[int, tuple[int, int]]],
                 share: float = READER_SHARE) -> dict[str, Any]:
    """Reader detection steps. One treated map: that run against every control; several: the joint rule (every
    treated run above every control)."""
    common = sorted(set.intersection(*(set(t) for t in treated), *(set(c) for c in controls))) if controls else []
    common = [s for s in common if s > 0]
    flags = [min(t[s][0] for t in treated) > max(c[s][0] for c in controls) for s in common]
    own = sorted(set.intersection(*(set(t) for t in treated)))
    own = [s for s in own if s > 0]
    sh = [all(t[s][1] and t[s][0] / t[s][1] >= share for t in treated) for s in own]
    return dict(first_signal=first_twice(common, flags) if common else None,
                first_share_005=next((s for s, ok in zip(own, sh) if ok), None),
                first_share_005_twice=first_twice(own, sh),
                n_common_steps=len(common))


def tally(counts: dict[int, tuple[int, int]], pred) -> dict[str, Any]:
    y = sum(v[0] for s, v in counts.items() if pred(s))
    n = sum(v[1] for s, v in counts.items() if pred(s))
    return dict(yes=y, n=n, share=(y / n) if n else None)


# ------------------------------------------------------------------------------------------------ driver


def analyse(root: Path, *, alpha: float = DEFAULT_ALPHA, tau: float = DEFAULT_TAU, kinds: Sequence[str] = KINDS,
            fold: bool = False, settings: Sequence[str] = ("new", "old")) -> dict[str, Any]:
    root = Path(root)
    missing: list[str] = []
    evs: dict[str, RunEvals] = {}
    for name, r in RUNS.items():
        if not r.blackbox or r.setting not in settings:
            continue
        if not (root / name / "evals.jsonl").exists():
            missing.append(str(root / name / "evals.jsonl"))
            continue
        evs[name] = load_run(root, name, kinds=kinds, fold=fold)
        ms = {k: v for k, v in evs[name].missing_samples.items() if v}
        log(f"{name}: {len(evs[name].probe)} probe steps, " + ", ".join(
            f"{k} {len(evs[name].words.get(k, {}))} steps" for k in kinds)
            + (f", samples missing {ms}" if ms else "") + (f", {evs[name].duplicates} duplicate records"
                                                          if evs[name].duplicates else ""))
    cache = Caches()
    doc = __doc__ or ""
    rules = doc[doc.find("Tests, each"):doc.find("Usage (subcommands)")].strip() if "Tests, each" in doc else ""
    out: dict[str, Any] = dict(
        written=time.strftime("%Y-%m-%d %H:%M:%S"), runs_root=str(root), alpha=alpha, tau=tau, kinds=list(kinds),
        fold_plurals=fold, rules=rules,
        runs={}, joint={}, null_pairs={}, placebo={}, missing=missing,
        loaded={n: dict(probe_steps=len(e.probe), **{f"{k}_steps": len(e.words.get(k, {})) for k in kinds},
                        missing_samples=e.missing_samples, duplicates=e.duplicates, bad_lines=e.bad_lines)
                for n, e in evs.items()})

    # tau_null per (setting, animal): the largest clean-vs-clean log-ratio of the probe at any step
    clean = {s: [n for n, r in RUNS.items() if r.animal is None and r.setting == s and n in evs] for s in settings}
    tau_null: dict[tuple[str, str], float] = {}
    for s, names in clean.items():
        for a in ("owl", "dolphin"):
            vals = []
            for n in names:
                others = [evs[m] for m in names if m != n]
                if others:
                    vals += [r["r"] for r in lp_ratio_series(evs[n], others, a)]
            if vals:
                tau_null[(s, a)] = max(vals)
    out["tau_null"] = {f"{s}:{a}": v for (s, a), v in tau_null.items()}

    def run_tests(name: str, target: str, ctrl_names: Sequence[str], paired_name: str | None) -> dict[str, Any]:
        ctrls = [evs[c] for c in ctrl_names if c in evs]
        res = tests_for(evs[name], ctrls, target, alpha=alpha, tau=tau,
                        tau_null=tau_null.get((RUNS[name].setting, target)), kinds=kinds, cache=cache,
                        paired=evs.get(paired_name) if paired_name else None)
        res.update(alpha=alpha, controls=[c.name for c in ctrls],
                   controls_missing=[c for c in ctrl_names if c not in evs], paired_control=paired_name
                   if paired_name in evs else None)
        return res

    for name, r in RUNS.items():
        if name not in evs or r.animal is None:
            continue
        res = run_tests(name, r.animal, r.controls, matched_control(name))
        ctrls = [evs[c] for c in r.controls if c in evs]
        res["free"] = {kind: free_discovery(evs[name], ctrls, kind, alpha, cache) for kind in kinds} if ctrls else {}
        on = res["onsets"]
        ref = README_ONSETS.get(name, {})
        res["onsets_vs_readme"] = {k: dict(computed=on.get(k), readme=v) for k, v in ref.items() if on.get(k) != v}
        rule, step = primary_onset(name, {name: on})
        res["primary_onset"] = dict(rule=rule, step=step)
        res["reader"] = {}
        for src in READER_SOURCES.get(name, []):
            t = load_counts(root, src.treated, src, r.animal, missing)
            cs = [load_counts(root, files, src, r.animal, missing) for files in src.controls]
            cs = [c for c in cs if c is not None]
            if t is None:
                continue
            rr = reader_rules([t], cs)
            rr.update(label=src.label, kind=src.kind, n_controls=len(cs), control_files=[list(c) for c in src.controls],
                      steps=sorted(t),
                      counts={str(s): t[s] for s in sorted(t)},
                      control_counts=[{str(s): c[s] for s in sorted(c)} for c in cs])
            if step is not None:
                rr["before_onset"] = tally(t, lambda s: 0 < s < step)
                rr["after_onset"] = tally(t, lambda s: s >= step)
                rr["controls_before_onset"] = [tally(c, lambda s: 0 < s < step) for c in cs]
            rr["lead"] = leads(rr, res["earliest"])
            res["reader"][src.name] = rr
        out["runs"][name] = res
        log(f"{name} ({r.animal}): earliest black-box {res['earliest']}, onsets {on}, reader first_signal / "
            "5% twice " + ", ".join(f"{k} {v['first_signal']}/{v['first_share_005_twice']}"
                                     for k, v in res["reader"].items()))

    # joint (several seeds of one setting and animal): old-setting dolphin seeds 1 and 2
    groups: dict[tuple[str, str], list[str]] = {}
    for name, res in out["runs"].items():
        groups.setdefault((RUNS[name].setting, res["target"]), []).append(name)
    for (setting, animal), names in groups.items():
        if len(names) < 2:
            continue
        per = [out["runs"][n] for n in names]
        j: dict[str, Any] = dict(runs=names, detect={})
        for test in BLACKBOX_TESTS:
            if test == "wilson":
                ws = [p["detect"]["wilson"] for p in per]
                j["detect"][test] = None if any(w is None for w in ws) else max(ws)
            else:
                j["detect"][test] = joint_detect(per, test)
        j["earliest"] = {k: earliest(j["detect"], tests) for k, tests in EARLIEST.items()}
        j["reader"] = {}
        for src_name in sorted(set.intersection(*(set(p["reader"]) for p in per))):
            srcs = [p["reader"][src_name] for p in per]
            # the joint rule needs one control set: every seed's source must read the same control files
            if len({json.dumps(s["control_files"]) for s in srcs}) != 1 or len({s["n_controls"] for s in srcs}) != 1:
                continue
            ts = [{int(k): tuple(v) for k, v in s["counts"].items()} for s in srcs]
            # the same control files for every seed (old setting: both clean seeds)
            cs = [{int(k): tuple(v) for k, v in c.items()} for c in srcs[0]["control_counts"]]
            if cs:
                jr = reader_rules(ts, cs)
                jr["label"] = srcs[0]["label"]
                jr["lead"] = leads(jr, j["earliest"])
                j["reader"][src_name] = jr
        out["joint"][f"{setting}:{animal}"] = j
        log(f"joint {setting} {animal} {names}: earliest black-box {j['earliest']}, reader first_signal / 5% twice "
            + ", ".join(f"{k} {v['first_signal']}/{v['first_share_005_twice']}" for k, v in j["reader"].items()))

    # null pairs: clean vs clean
    for s, names in clean.items():
        for n in names:
            others = [m for m in names if m != n]
            if not others:
                continue
            entry: dict[str, Any] = dict(reference=others, targets={}, free={})
            for a in ("owl", "dolphin"):
                res = run_tests(n, a, others, others[0] if len(others) == 1 else None)
                entry["targets"][a] = dict(detect=res["detect"])
            for kind in kinds:
                entry["free"][kind] = free_discovery(evs[n], [evs[m] for m in others], kind, alpha, cache)
            out["null_pairs"][f"{n}_vs_{'+'.join(others)}"] = entry
    # placebo animals: the treated runs' tests with the other target animals
    for name in list(out["runs"]):
        r = RUNS[name]
        out["placebo"][name] = {a: dict(detect=run_tests(name, a, r.controls, matched_control(name))["detect"])
                                for a in TARGET_ANIMALS if a != r.animal}
    return out


def fmt(x) -> str:
    return "-" if x is None else f"{x:,}" if isinstance(x, int) else str(x)


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def summary_markdown(res: dict[str, Any]) -> str:
    lines = [f"# Black-box timing (alpha {res['alpha']}, tau {res['tau']:.3f}, written {res['written']})", "",
             "Detection step = first eval step where the test passes at it and at the next step of its grid (full "
             "evaluations every 500 steps, small and probe every 100). '-' = never. Target-specified tests know which "
             "animal to test; target-blind tests rank every first word and pass only when the animal comes out on top. "
             "'ctrl' = only the tests against clean controls (paired tests: the seed-matched clean run; Fisher and "
             "log-prob tests: all clean runs of the setting pooled).", ""]
    fam = [("specified", "earliest specified"), ("specified_ctrl", "earliest specified ctrl"),
           ("blind", "earliest blind"), ("blind_ctrl", "earliest blind ctrl")]
    head = ["run", "onset P>=0.1 (100/500)", "primary onset", *TABLE_TESTS, *(lab for _, lab in fam)]
    lines += ["## Black-box tests", "", _row(head), "|" + "---|" * len(head)]
    for name, r in res["runs"].items():
        on, po = r["onsets"], r["primary_onset"]
        lines.append(_row([f"{name} ({r['target']})", f"{fmt(on.get('p01_100'))} / {fmt(on.get('p01_500'))}",
                           f"{fmt(po['step'])} ({po['rule']})", *(fmt(r["detect"].get(t)) for t in TABLE_TESTS),
                           *(fmt(r["earliest"][k]) for k, _ in fam)]))
    for key, j in res["joint"].items():
        lines.append(_row([f"joint {key} ({'+'.join(j['runs'])})", "", "",
                           *(fmt(j["detect"].get(t)) for t in TABLE_TESTS), *(fmt(j["earliest"][k]) for k, _ in fam)]))
    lines += ["", "## Target-blind discovery (no target: the first word ranked first and Bonferroni-significant "
              "against the controls twice in a row)", "", _row(["run", "evaluation", "first discovery", "word",
                                                                  "steps with a significant top word"]),
              "|---|---|---|---|---|"]
    for name, r in res["runs"].items():
        for kind, fr in r.get("free", {}).items():
            lines.append(_row([f"{name} ({r['target']})", kind, fmt(fr["first_discovery"]), fr["word"] or "-",
                               f"{len(fr['steps_with_significant_top'])} of {fr['n_steps']}"]))
    lines += ["", "## Reader", "",
              "Two reader criteria: first_signal = the treated count above every control's count at two read steps in "
              "a row (any count, e.g. 1 vs 0); 5% twice = share >= 0.05 at two read steps in a row. Lead = earliest "
              "black-box step minus the reader step (> 0: the reader is ahead), per criterion, against the earliest "
              "target-specified and target-blind control-based tests.", "",
              _row(["run", "source", "first_signal", "5% twice", "before onset", "after onset", "controls before onset",
                    "lead (first_signal) vs specified ctrl / blind ctrl", "lead (5% twice) vs specified ctrl / blind ctrl"]),
              "|" + "---|" * 9]

    def lead_txt(v: dict, crit: str) -> str:
        ld = v.get("lead", {}).get(crit, {})
        return f"{fmt(ld.get('specified_ctrl'))} / {fmt(ld.get('blind_ctrl'))}"

    for name, r in res["runs"].items():
        for k, v in r["reader"].items():
            bo, ao = v.get("before_onset"), v.get("after_onset")
            cb = v.get("controls_before_onset") or []
            lines.append(_row([name, k, fmt(v["first_signal"]), fmt(v["first_share_005_twice"]),
                               f"{bo['yes']}/{bo['n']}" if bo else "-", f"{ao['yes']}/{ao['n']}" if ao else "-",
                               ", ".join(f"{c['yes']}/{c['n']}" for c in cb) or "-",
                               lead_txt(v, "first_signal"), lead_txt(v, "first_share_005_twice")]))
    for key, j in res["joint"].items():
        for k, v in j["reader"].items():
            lines.append(_row([f"joint {key}", k, fmt(v["first_signal"]), fmt(v["first_share_005_twice"]), "", "", "",
                               lead_txt(v, "first_signal"), lead_txt(v, "first_share_005_twice")]))
    lines += ["", "## False-alarm checks (clean vs clean, and placebo animals; a step here is not evidence of the trait)",
              "", _row(["comparison", "target", *TABLE_TESTS]), "|---|---|" + "---|" * len(TABLE_TESTS)]
    for key, e in res["null_pairs"].items():
        for a, v in e["targets"].items():
            lines.append(_row([key, a, *(fmt(v["detect"].get(t)) for t in TABLE_TESTS)]))
    for name, e in res["placebo"].items():
        for a, v in e.items():
            lines.append(_row([f"placebo {name}", a, *(fmt(v["detect"].get(t)) for t in TABLE_TESTS)]))
    free_null = [(key, kind, fr) for key, e in res["null_pairs"].items() for kind, fr in e["free"].items()]
    if free_null:
        lines += ["", _row(["clean vs clean", "evaluation", "first (false) discovery", "word",
                            "steps with a significant top word"]), "|---|---|---|---|---|"]
        for key, kind, fr in free_null:
            lines.append(_row([key, kind, fmt(fr["first_discovery"]), fr["word"] or "-",
                               f"{len(fr['steps_with_significant_top'])} of {fr['n_steps']}"]))
    if res["missing"]:
        lines += ["", f"Missing ({len(res['missing'])}):", ""] + [f"- {m}" for m in res["missing"]]
    return "\n".join(lines) + "\n"


def fetch(dest: Path, mode: str = "members") -> Path:
    """evals.jsonl, eval_samples/ and the ``READER_SOURCES`` files of every run from HF ``jason9464/sprint-data-2610``
    (``sprint_data.py``), checked against its manifest; returns the runs root ``dest/student_runs``."""
    from main_run.sprint_data import fetch as fetch_parts

    return fetch_parts(dest, what="blackbox", mode=mode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the eval and reading files (HF jason9464/sprint-data-2610)")
    f.add_argument("--dest", type=Path, default=Path("data/sprint_data"))
    f.add_argument("--mode", choices=["members", "whole"], default="members",
                   help="members: range reads of the needed files only; whole: download and unpack each part")
    r = sub.add_parser("run", help="run the analysis")
    r.add_argument("--runs-root", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True, help="JSON output; the tables go to <stem>.md")
    r.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    r.add_argument("--tau", type=float, default=DEFAULT_TAU, help="log-ratio threshold of the probe tests (log 2)")
    r.add_argument("--kinds", default=",".join(KINDS))
    r.add_argument("--settings", default="new,old")
    r.add_argument("--fold-plurals", action="store_true",
                   help="count 'dolphins' as 'dolphin' (vocabulary animals only); off = the public exact-word rule")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        log(f"runs root {fetch(a.dest, a.mode)}")
        return 0
    kinds = [k for k in a.kinds.split(",") if k]
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        ap.error(f"--kinds: unknown {bad}")
    res = analyse(a.runs_root, alpha=a.alpha, tau=a.tau, kinds=kinds, fold=a.fold_plurals,
                  settings=[s for s in a.settings.split(",") if s])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, a.out)
    md = summary_markdown(res)
    a.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    log(f"wrote {a.out} and {a.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Rank candidate answers by the reader's log-probability shift (experiment 2 of 2026-10-26; scores from
``reader/candidate_score.py``). CPU only.

For one inputs file (treated) and one reference, at every read step and window:
  shift[c, t] = lp_treated[c, t] - lp_reference[c, t]      (sum of log P over the answer text, ``lp``)
  score[c]    = mean over the phrasings t of shift[c, t]
References:
  control    the clean student's score file of the same inputs kind at the same (step, window): removes the reader's
             own preferences, the name length, and whatever number training does to the input regardless of the trait.
             Only this reference isolates a trait-specific shift.
  no_vector  the treated file's no-vector score (one per answer): removes the reader's prior only
  random     the treated file's random-direction score at the same (step, window): an injection baseline
  step:N     the treated file's own score at read step N (same window): what changed since step N within the run (the
             EM fallback, where no control student exists; the training topic is present from step 1, so this is a
             within-run baseline, not a matched control)
Within one phrasing every animal shares the text before the animal, so the ranking depends only on the tokens from
the animal onward; subtracting a reference removes what the reader says about any input.

Readout (fixed before looking; plan: "Fix the rank threshold and the step rule before looking"): the target's rank
among all candidates (1 = largest score; ties count half), AUROC = the share of the other candidates scored below the
target (ties half), and the detection step = the first read step where the target ranks <= k at it and at the next
read step of the series (two in a row, as every onset and reader rule here), k = 1 (also reported: k = 3). Chance:
rank 1 of 100 is 1% per step. Per comparison also: the run's primary onset (``student_runs.primary_onset``) and, over
the read steps before it, the number of steps with rank 1 and the mean AUROC.

Null comparisons, where any detection is a false alarm: each clean run's files against no_vector and random with
each target animal; the old setting's ``full_clean_s2`` against ``full_clean_s1`` (control reference, both ways).

Subcommands:
  plan      every subliminal comparison found under a scores root laid out as ``<root>/<run>/e2/cand_*.npz`` (the
            default output place of candidate_score.py, or its ``--out-dir <dir>/{run}/e2``):
              python -m main_run.candidate_rank plan --scores-root $RUNS --reader-name rl_paper_c040 \\
                  --question preference --candidates animals --out results/candidates/subliminal_rank.json
  compare   one treated file against one reference, explicit targets (EM, or anything else):
              python -m main_run.candidate_rank compare --treated <cand_...npz> --reference no_vector \\
                  --targets broad_harm,misaligned_views,risky_finance --onset 65 --out <json> \\
                  [--exclude-groups target_content,benign_same_topic]
  crosscheck  the dolphin/whale 2 x 2 (rule fixed before any score was opened; ``crosscheck_series``): the dolphin and
            the whale student, each scored on the dolphin and the whale candidate against the clean control, read
            steps <= 1,500:
              python -m main_run.candidate_rank crosscheck --scores-root results/candidates \\
                  --reader-name rl_paper_c040 --question preference --candidates animals \\
                  --out results/candidates/crosscheck_dolphin_whale.json
All write a JSON (every per-step row) and ``<out stem>.md`` (a summary table), and print the table.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from main_run.reader.candidate_score import load_scores
from main_run.student_runs import RUNS, TARGET_ANIMALS, matched_control, primary_onset

REFERENCES = ("control", "no_vector", "random", "step")
# Settings that must agree between a treated file and its control file.
COMPAT_KEYS = ("candidates_sha256", "question", "reader_name", "adapter", "lora_dtype", "layers", "center",
               "weighting", "hook_onto_layer", "steering_coefficient", "candidate_ids", "template_ids")
NAME_RE = re.compile(r"^cand_(?P<prefix>.+?)__(?P<stem>.+)\.npz$")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [candidate_rank] {msg}", flush=True)


def first_twice(steps: Sequence[int], flags: Sequence[bool]) -> int | None:
    return next((steps[i] for i in range(len(steps) - 1) if flags[i] and flags[i + 1]), None)


# ------------------------------------------------------------------------------------------------ core


def check_compatible(a: dict[str, Any], b: dict[str, Any], what: str) -> None:
    diff = [k for k in COMPAT_KEYS if a["meta"].get(k) != b["meta"].get(k)]
    if a["meta"].get("texts") != b["meta"].get("texts"):
        diff.append("texts")
    if diff:
        raise ValueError(f"{what}: the score files differ in {diff}")


def cell_shifts(treated: dict[str, Any], ref_kind: str, ref: dict[str, Any] | None = None, *,
                value: str = "lp", with_eos: bool = False, ref_step: int | None = None) -> list[tuple[int, str, np.ndarray]]:
    """[(step, window, shift [C, T])] over the cells scored in the treated file (and the reference file). ``step``:
    the treated file's own scores at read step ``ref_step`` (same window), for the steps after it."""
    def get(d: dict, base: str, i: int | None = None, w: int | None = None) -> np.ndarray:
        name = base + ("" if value == "lp" else "_tail")
        x = d[name] if i is None else d[name][i, w]
        if with_eos:
            e = d[base + "_eos"] if i is None else d[base + "_eos"][i, w]
            x = x + e
        return np.asarray(x, dtype=np.float64)

    if value not in ("lp", "lp_tail"):
        raise ValueError(value)
    if "lp" not in treated:
        raise ValueError("the treated file has no correct-condition scores")
    out = []
    ok = treated["done"] & ~treated["skipped_zero"]
    if ref_kind == "control":
        if ref is None or "lp" not in ref:
            raise ValueError("control reference needs a score file with correct-condition scores")
        rok = ref["done"] & ~ref["skipped_zero"]
        rs = {t: i for i, t in enumerate(ref["steps"])}
        rw = {w: j for j, w in enumerate(ref["windows"])}
        for i, t in enumerate(treated["steps"]):
            for w, lab in enumerate(treated["windows"]):
                if ok[i, w] and t in rs and lab in rw and rok[rs[t], rw[lab]]:
                    out.append((t, lab, get(treated, "lp", i, w) - get(ref, "lp", rs[t], rw[lab])))
    elif ref_kind == "no_vector":
        if "lp_no_vector" not in treated:
            raise ValueError("the treated file has no no_vector scores")
        base = get(treated, "lp_no_vector")
        for i, t in enumerate(treated["steps"]):
            for w, lab in enumerate(treated["windows"]):
                if ok[i, w]:
                    out.append((t, lab, get(treated, "lp", i, w) - base))
    elif ref_kind == "step":
        if ref_step not in treated["steps"]:
            raise ValueError(f"reference step {ref_step} is not a read step of the treated file")
        i0 = treated["steps"].index(ref_step)
        for i, t in enumerate(treated["steps"]):
            for w, lab in enumerate(treated["windows"]):
                if t > ref_step and ok[i, w] and ok[i0, w]:
                    out.append((t, lab, get(treated, "lp", i, w) - get(treated, "lp", i0, w)))
    elif ref_kind == "random":
        if "lp_random" not in treated:
            raise ValueError("the treated file has no random scores")
        for i, t in enumerate(treated["steps"]):
            for w, lab in enumerate(treated["windows"]):
                if ok[i, w]:
                    out.append((t, lab, get(treated, "lp", i, w) - get(treated, "lp_random", i, w)))
    else:
        raise ValueError(f"reference {ref_kind!r}; choose from {REFERENCES}")
    for t, lab, d in out:
        if not np.isfinite(d).all():
            raise ValueError(f"non-finite shift at step {t} window {lab}")
    return out


def rank_of(scores: np.ndarray, j: int, universe: Sequence[int]) -> tuple[float, float]:
    """(rank, AUROC) of candidate ``j`` among ``universe`` (which contains j): rank = 1 + #higher + #ties / 2."""
    others = np.asarray([u for u in universe if u != j], dtype=np.int64)
    if others.size == 0:
        return 1.0, 1.0
    s = scores[others]
    hi = int((s > scores[j]).sum())
    tie = int((s == scores[j]).sum())
    lo = others.size - hi - tie
    return 1.0 + hi + tie / 2.0, (lo + tie / 2.0) / others.size


def series(shifts: list[tuple[int, str, np.ndarray]], ids: Sequence[str], target: str, *,
           universe: Sequence[int] | None = None, top: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Per window: one row per read step with the target's rank, AUROC, score, per-phrasing ranks and the top
    candidates."""
    j = list(ids).index(target)
    uni = list(range(len(ids))) if universe is None else sorted(set(universe) | {j})
    out: dict[str, list[dict[str, Any]]] = {}
    for t, lab, d in sorted(shifts, key=lambda x: (x[1], x[0])):
        s = d.mean(axis=1)
        rank, auc = rank_of(s, j, uni)
        per_t = [rank_of(d[:, k], j, uni)[0] for k in range(d.shape[1])]
        order = sorted(uni, key=lambda c: -s[c])
        out.setdefault(lab, []).append(dict(step=int(t), rank=rank, auroc=auc, score=float(s[j]),
                                            rank_per_phrasing=per_t,
                                            top=[(ids[c], round(float(s[c]), 4)) for c in order[:top]]))
    return out


def summarize(rows: list[dict[str, Any]], onset: int | None, ks: Sequence[int] = (1, 3)) -> dict[str, Any]:
    steps = [r["step"] for r in rows]
    res: dict[str, Any] = {f"detect_top{k}": first_twice(steps, [r["rank"] <= k for r in rows]) for k in ks}
    res["first_rank1"] = next((r["step"] for r in rows if r["rank"] <= 1), None)
    res["n_steps"] = len(rows)
    res["mean_auroc"] = float(np.mean([r["auroc"] for r in rows])) if rows else None
    if onset is not None:
        pre = [r for r in rows if r["step"] < onset]
        # the detection rule needs two read steps; with fewer before onset the comparison cannot show a lead
        res["can_detect_before_onset"] = len(pre) >= 2
        res.update(onset=onset, pre_steps=len(pre), pre_rank1=sum(r["rank"] <= 1 for r in pre),
                   pre_top3=sum(r["rank"] <= 3 for r in pre),
                   pre_mean_auroc=float(np.mean([r["auroc"] for r in pre])) if pre else None,
                   pre_mean_rank=float(np.mean([r["rank"] for r in pre])) if pre else None)
    return res


def compare(treated: dict[str, Any], ref_kind: str, ref: dict[str, Any] | None, targets: Sequence[str], *,
            onset: int | None, value: str = "lp", with_eos: bool = False, exclude_groups: Sequence[str] = (),
            name: str = "", ref_step: int | None = None) -> dict[str, Any]:
    if ref is not None:
        check_compatible(treated, ref, name or "comparison")
    ids = treated["candidate_ids"]
    groups = treated.get("groups") or [""] * len(ids)
    uni = [c for c in range(len(ids)) if groups[c] not in set(exclude_groups)]
    sh = cell_shifts(treated, ref_kind, ref, value=value, with_eos=with_eos, ref_step=ref_step)
    res: dict[str, Any] = dict(name=name, treated=treated["meta"]["inputs"], reference=ref_kind, reference_step=ref_step,
                               reference_file=None if ref is None else ref["meta"]["inputs"], value=value,
                               with_eos=with_eos, exclude_groups=list(exclude_groups), n_universe=len(uni),
                               targets={})
    for tg in targets:
        if tg not in ids:
            raise ValueError(f"target {tg!r} is not a candidate")
        per_win = series(sh, ids, tg, universe=uni)
        res["targets"][tg] = {w: dict(summary=summarize(rows, onset), rows=rows) for w, rows in per_win.items()}
    return res


# ------------------------------------------------------------------------------------------------ plan


def score_files(e2_dir: Path, prefix: str) -> dict[str, Path]:
    """{inputs stem: path} of the score files with name prefix ``cand_<prefix>__`` in one e2 dir."""
    out = {}
    if not e2_dir.is_dir():
        return out
    for p in sorted(e2_dir.iterdir()):
        m = NAME_RE.match(p.name)
        if m and m.group("prefix") == prefix and not p.name.endswith(".partial.npz") and ".tmp" not in p.name:
            out[m.group("stem")] = p
    return out


def base_stem(stem: str) -> str:
    """The inputs kind without the clean seed 2 partial marker."""
    return stem[: -len("-partial")] if stem.endswith("-partial") else stem


def run_files(root: Path, run: str, prefix: str) -> dict[str, Path]:
    """Score files of a run, in its own e2 dir and (old setting) its adamfx rerun's e2 dir, by base stem."""
    r = RUNS[run]
    out: dict[str, Path] = {}
    for d in (run, r.adamfx_run):
        if d is None:
            continue
        for stem, p in score_files(Path(root) / d / "e2", prefix).items():
            key = base_stem(stem)
            if key in out:
                raise ValueError(f"two score files for {run} {key}: {out[key]} and {p}")
            out[key] = p
    return out


def control_stems(stem: str, run: str) -> list[str]:
    """The control file's inputs kinds, in order of preference: the same kind; for a fine100 rerun's ``*_fine100`` file
    the clean run's 100-step file of that kind, then its every-500 file (the clean student has no fine100 rerun)."""
    if run.endswith("_fine100") and stem.endswith("_fine100"):
        base = stem[: -len("fine100")]
        return [base + "every100", base + "every500"]
    return [stem]


def plan(root: Path, prefix: str, *, value: str = "lp", with_eos: bool = False) -> dict[str, Any]:
    root = Path(root)
    out: dict[str, Any] = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), scores_root=str(root), prefix=prefix,
                               value=value, with_eos=with_eos, comparisons=[], nulls=[], missing=[])
    cache: dict[Path, dict[str, Any]] = {}

    def load(p: Path) -> dict[str, Any]:
        if p not in cache:
            cache[p] = load_scores(p)
        return cache[p]

    for run, r in RUNS.items():
        if r.animal is None:
            continue
        files = run_files(root, run, prefix)
        if not files:
            out["missing"].append(f"{run}: no score files cand_{prefix}__*.npz")
            continue
        ctrl = matched_control(run)
        cfiles = run_files(root, ctrl, prefix)
        rule, onset = primary_onset(run)
        for stem, p in sorted(files.items()):
            t = load(p)
            refs: list[tuple[str, dict | None, str | None]] = []
            cst = next((c for c in control_stems(stem, run) if c in cfiles), None)
            if cst is not None:
                refs.append(("control", load(cfiles[cst]), str(cfiles[cst])))
            else:
                out["missing"].append(f"{run} {stem}: control {ctrl} {' or '.join(control_stems(stem, run))}")
            for k in ("no_vector", "random"):
                if f"lp_{k}" in t:
                    refs.append((k, None, None))
            for ref_kind, ref, ref_path in refs:
                c = compare(t, ref_kind, ref, [r.animal], onset=onset, value=value, with_eos=with_eos,
                            name=f"{run} {stem} vs {ref_kind}")
                c.update(run=run, setting=r.setting, inputs=stem, onset_rule=rule, control_run=ctrl,
                         treated_file=str(p), reference_path=ref_path)
                out["comparisons"].append(c)
                for w, v in c["targets"][r.animal].items():
                    s = v["summary"]
                    log(f"{run} {stem} {w} vs {ref_kind}: top-1 twice {s['detect_top1']}, onset {onset}, pre-onset "
                        f"rank 1 at {s.get('pre_rank1')}/{s.get('pre_steps')} steps")
    # nulls: clean runs against no_vector / random (every target animal), and clean vs clean
    for run, r in RUNS.items():
        if r.animal is not None:
            continue
        files = run_files(root, run, prefix)
        others = [c for c, x in RUNS.items() if x.animal is None and x.setting == r.setting and c != run]
        for stem, p in sorted(files.items()):
            t = load(p)
            refs = [(k, None, None) for k in ("no_vector", "random") if f"lp_{k}" in t]
            for o in others:
                of = run_files(root, o, prefix)
                if stem in of:
                    refs.append(("control", load(of[stem]), str(of[stem])))
            for ref_kind, ref, ref_path in refs:
                targets = [a for a in TARGET_ANIMALS if a in t["candidate_ids"]]
                c = compare(t, ref_kind, ref, targets, onset=None, value=value, with_eos=with_eos,
                            name=f"{run} {stem} vs {ref_kind}" + (f" ({Path(ref_path).parts[-3]})" if ref_path else ""))
                c.update(run=run, setting=r.setting, inputs=stem, treated_file=str(p), reference_path=ref_path)
                out["nulls"].append(c)
    return out


# ------------------------------------------------------------------------------------------------ dolphin/whale 2 x 2

# Fixed 2026-10-26 before any score of these runs was opened (note 20261026_blackbox_and_candidate_scoring.md, section
# "Dolphin/whale cross-check"). Question: when the reader names dolphin for the dolphin student, is that the data or
# the reader's own lean towards dolphin? Its readings named dolphin about as often as whale for the whale student.
CROSS_STUDENTS = (("newset_dolphin_s1", "dolphin", "whale"), ("newset_whale_s1", "whale", "dolphin"))  # (run, own, other)
CROSS_CONTROL = "newset_clean_s1"
CROSS_MAX_STEP = 1500
CROSS_GRID = 100  # read steps that are positive multiples of 100 (the 100-step files' step 1 is left out)
CROSS_GUARD_RANK = 10  # the better of dolphin and whale must rank <= 10 of 100 at most read steps
# A verdict needs |mean margin| >= 0.2 nats: bf16 scores are off from fp32 by 0.1-0.2 nats per answer on average
# (measured on the H100, note), so a smaller mean margin is within the scoring noise.
CROSS_MIN_MARGIN = 0.2
# (inputs kind, window) -> role. "primary" decides the question for the readings that raised it (direct writes,
# cumulative window, every 500 steps: the readings' inputs, window and grid). "plain gradient" gives the same call on
# the plain gradient (the whale student's only plain-gradient file is the 100-step one). The rest are reported only.
# Windows not listed here are reported as "other window".
CROSS_ROLES = {
    ("adamfx_inputs_every500", "cum"): "primary",
    ("grad_inputs_every100", "cum"): "plain gradient",
    ("adamfx_inputs_every100", "cum"): "secondary",
    ("adamfx_fixed_inputs_every100", "cum"): "secondary",
    ("ckptdiff_inputs_every500", "cum"): "secondary",
}
CROSS_STEMS = ("adamfx_inputs_every500", "grad_inputs_every100", "adamfx_inputs_every100",
               "adamfx_fixed_inputs_every100", "ckptdiff_inputs_every500")
CROSS_CALLS = {  # (winner in the dolphin student, winner in the whale student) -> call
    ("dolphin", "whale"): "PASS: each student's own animal is ahead (the identity comes from the data)",
    ("dolphin", "dolphin"): "FAIL (dolphin for both): the dolphin call is the reader's lean, not the data",
    ("whale", "whale"): "FAIL (whale for both): both read as whale",
    ("whale", "dolphin"): "FAIL (swapped): each student reads as the other animal",
}


def crosscheck_series(students: Sequence[tuple[str, str, str, dict[str, Any]]], control: dict[str, Any], window: str,
                      *, max_step: int = CROSS_MAX_STEP, grid: int = CROSS_GRID,
                      guard_rank: int = CROSS_GUARD_RANK, min_margin: float = CROSS_MIN_MARGIN) -> dict[str, Any]:
    """The 2 x 2 for one inputs kind and window. ``students``: (run, own animal, other animal, score file).

    Per student and read step t (positive multiples of ``grid`` up to ``max_step``), against the clean control at the
    same (step, window): margin(t) = score(own) - score(other), score = mean over phrasings of the log P shift, so
    margin > 0 <=> own outranks other among the candidates. Student verdict: "own" if margin > 0 at more than half of
    the steps and the mean margin >= ``min_margin``; "other" if margin > 0 at fewer than half and the mean margin <=
    -``min_margin``; "mixed" otherwise (split steps, or a mean margin within the bf16 scoring noise). Guard: the better rank of the two animals is <= ``guard_rank`` at more than half of the steps (else the
    two are not the reader's lean at all and their order is noise). Call: incomplete (a step missing in any of the
    three files) -> no call; a guard fails -> "NO SIGNAL"; a mixed student -> "INCONCLUSIVE"; else ``CROSS_CALLS``."""
    res: dict[str, Any] = dict(window=window, max_step=max_step, grid=grid, guard_rank=guard_rank,
                               min_margin=min_margin, students={}, incomplete=[])
    files = [control] + [s[3] for s in students]
    for f in files[1:]:
        check_compatible(f, control, f"crosscheck {f['meta'].get('inputs')}")
        for k in ("batch_rows", "path"):
            if f["meta"].get(k) != control["meta"].get(k):
                raise ValueError(f"crosscheck: {k} differs ({f['meta'].get(k)} vs control {control['meta'].get(k)})")

    def in_range(t: int) -> bool:
        return 0 < t <= max_step and t % grid == 0

    want = sorted({t for f in files for t in f["steps"] if in_range(t)})
    for f in files:
        name = f["meta"].get("inputs")
        if window not in f["windows"]:
            res["incomplete"].append(f"{name}: no window {window}")
            continue
        w = f["windows"].index(window)
        ok = f["done"] & ~f["skipped_zero"]
        for t in want:
            if t not in f["steps"]:
                res["incomplete"].append(f"{name}: no read step {t}")
            elif not ok[f["steps"].index(t), w]:
                res["incomplete"].append(f"{name}: step {t} not scored (unfinished or zero vector)")
    res["steps"] = want
    if len(want) < 3:
        res["incomplete"].append(f"only {len(want)} read steps in range (need 3)")
    winners = []
    for run, own, other, t in students:
        ids = t["candidate_ids"]
        jo, jx = ids.index(own), ids.index(other)
        rows = []
        for step, lab, d in sorted(cell_shifts(t, "control", control), key=lambda x: x[0]):
            if lab != window or step not in want:
                continue
            s = d.mean(axis=1)
            r_own, _ = rank_of(s, jo, range(len(ids)))
            r_oth, _ = rank_of(s, jx, range(len(ids)))
            order = np.argsort(-s, kind="stable")
            rows.append(dict(step=int(step), margin=float(s[jo] - s[jx]), score_own=float(s[jo]),
                             score_other=float(s[jx]), rank_own=r_own, rank_other=r_oth,
                             phrasings_own_ahead=int((d[jo] > d[jx]).sum()), n_phrasings=int(d.shape[1]),
                             top=[(ids[c], round(float(s[c]), 4)) for c in order[:5]]))
        n = len(rows)
        n_ahead = sum(r["margin"] > 0 for r in rows)
        mean_margin = float(np.mean([r["margin"] for r in rows])) if rows else float("nan")
        n_signal = sum(min(r["rank_own"], r["rank_other"]) <= guard_rank for r in rows)
        if n and 2 * n_ahead > n and mean_margin >= min_margin:
            verdict, winner = "own", own
        elif n and 2 * n_ahead < n and mean_margin <= -min_margin:
            verdict, winner = "other", other
        else:
            verdict, winner = "mixed", None
        winners.append(winner)
        res["students"][run] = dict(own=own, other=other, n_steps=n, n_own_ahead=n_ahead, mean_margin=mean_margin,
                                    verdict=verdict, winner=winner, n_signal=n_signal,
                                    guard_ok=bool(n and 2 * n_signal > n), rows=rows)
    if res["incomplete"]:
        res["call"] = "INCOMPLETE: no call (" + "; ".join(res["incomplete"][:3]) + ")"
    elif not all(v["guard_ok"] for v in res["students"].values()):
        res["call"] = (f"NO SIGNAL: neither animal ranks <= {guard_rank} at most steps in at least one student, so "
                       "their order is not informative")
    elif None in winners:
        res["call"] = ("INCONCLUSIVE: a student's own-vs-other order is split across steps or its mean margin is "
                       f"below {min_margin} nats")
    else:
        res["call"] = CROSS_CALLS[(winners[0], winners[1])]
    return res


def crosscheck(root: Path, prefix: str, *, stems: Sequence[str] = CROSS_STEMS) -> dict[str, Any]:
    """Every (inputs kind, window) of ``stems`` whose three score files (both students, the clean control) exist."""
    out: dict[str, Any] = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), scores_root=str(root), prefix=prefix,
                               rule=crosscheck_series.__doc__, control=CROSS_CONTROL,
                               students=[s[0] for s in CROSS_STUDENTS], series=[], missing=[])
    cfiles = run_files(root, CROSS_CONTROL, prefix)
    sfiles = {run: run_files(root, run, prefix) for run, _, _ in CROSS_STUDENTS}
    for stem in stems:
        miss = [r for r, fs in [(CROSS_CONTROL, cfiles), *sfiles.items()] if stem not in fs]
        if miss:
            out["missing"].append(f"{stem}: no finished score file for {', '.join(miss)}")
            continue
        ctl = load_scores(cfiles[stem])
        studs = [(run, own, oth, load_scores(sfiles[run][stem])) for run, own, oth in CROSS_STUDENTS]
        for w in ctl["windows"]:
            r = crosscheck_series(studs, ctl, w)
            r.update(inputs=stem, role=CROSS_ROLES.get((stem, w), "other window"),
                     files=[str(cfiles[stem])] + [str(sfiles[run][stem]) for run, _, _ in CROSS_STUDENTS])
            out["series"].append(r)
            log(f"crosscheck {stem} {w} ({r['role']}): {r['call']}")
    return out


def crosscheck_markdown(res: dict[str, Any]) -> str:
    order = {"primary": 0, "plain gradient": 1, "secondary": 2, "other window": 3}
    series = sorted(res["series"], key=lambda r: (order[r["role"]], r["inputs"], r["window"]))
    d_run, w_run = res["students"]
    lines = [f"# Dolphin/whale cross-check ({res['prefix']}, written {res['written']})", "",
             "Each student scored on its own and the other animal against the clean control (newset_clean_s1) at the "
             "same (step, window). margin = score(own) - score(other) in nats (mean over the four phrasings); "
             "margin > 0 means the own animal outranks the other. The rule (fixed before any score was opened) is in "
             "the JSON and in docs/notes/20261026_blackbox_and_candidate_scoring.md. Only the primary row answers "
             "the question for the readings; the plain-gradient row gives the same call on the plain gradient; the "
             "rest are reported only.", "",
             "| role | inputs | window | steps | dolphin student: own ahead / steps, mean margin, guard | "
             "whale student: own ahead / steps, mean margin, guard | call |", "|---|---|---|---|---|---|---|"]

    def cell(s: dict[str, Any]) -> str:
        return (f"{s['verdict']} ({s['n_own_ahead']}/{s['n_steps']}, {s['mean_margin']:+.3f}; "
                f"signal {s['n_signal']}/{s['n_steps']})")

    for r in series:
        st = r["steps"]
        span = f"{st[0]:,}-{st[-1]:,} ({len(st)})" if st else "-"
        lines.append(f"| {r['role']} | {r['inputs']} | {r['window']} | {span} | {cell(r['students'][d_run])} | "
                     f"{cell(r['students'][w_run])} | {r['call']} |")
    for r in series:
        if r["role"] not in ("primary", "plain gradient"):
            continue
        lines += ["", f"## {r['role']}: {r['inputs']} {r['window']}, per read step", "",
                  "| step | student | rank own | rank other | margin | phrasings own ahead | top 5 |",
                  "|---|---|---|---|---|---|---|"]
        for run in (d_run, w_run):
            s = r["students"][run]
            for x in s["rows"]:
                top = ", ".join(f"{a} {v:+.2f}" for a, v in x["top"])
                lines.append(f"| {x['step']:,} | {run} ({s['own']}) | {fmt(x['rank_own'])} | {fmt(x['rank_other'])} "
                             f"| {x['margin']:+.3f} | {x['phrasings_own_ahead']}/{x['n_phrasings']} | {top} |")
    if res.get("missing"):
        lines += ["", f"Missing ({len(res['missing'])}):", ""] + [f"- {m}" for m in res["missing"]]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ output


def fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}" if not x.is_integer() else f"{int(x):,}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def markdown(res: dict[str, Any]) -> str:
    lines = [f"# Candidate ranks ({res.get('prefix', '')}, written {res['written']})", "",
             "Score = mean over phrasings of log P(answer | input) - log P(answer | reference). Rank 1 = the target "
             "has the largest score; detection = rank <= k at two read steps in a row. Pre-onset columns use the "
             "read steps before the primary onset.", ""]
    head = ["comparison", "window", "target", "top-1 twice", "top-3 twice", "first rank 1", "onset",
            "can detect before onset", "pre-onset rank 1 / steps", "pre-onset top 3", "pre-onset mean AUROC",
            "all-steps mean AUROC"]
    for section in ("comparisons", "nulls"):
        if not res.get(section):
            continue
        lines += [f"## {'Treated runs' if section == 'comparisons' else 'Null comparisons (detections are false alarms)'}",
                  "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for c in res[section]:
            for tg, wins in c["targets"].items():
                for w, v in wins.items():
                    s = v["summary"]
                    lines.append("| " + " | ".join([
                        c["name"], w, tg, fmt(s["detect_top1"]), fmt(s["detect_top3"]), fmt(s["first_rank1"]),
                        fmt(s.get("onset")),
                        "-" if "can_detect_before_onset" not in s else ("yes" if s["can_detect_before_onset"] else "NO"),
                        f"{fmt(s.get('pre_rank1'))} / {fmt(s.get('pre_steps'))}",
                        fmt(s.get("pre_top3")), fmt(s.get("pre_mean_auroc")), fmt(s["mean_auroc"])]) + " |")
        lines.append("")
    if res.get("missing"):
        lines += [f"Missing ({len(res['missing'])}):", ""] + [f"- {m}" for m in res["missing"]]
    return "\n".join(lines) + "\n"


def write(res: dict[str, Any], out: Path, md_fn=markdown) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    md = md_fn(res)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    log(f"wrote {out} and {out.with_suffix('.md')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="every subliminal comparison under a scores root")
    pl.add_argument("--scores-root", type=Path, required=True)
    pl.add_argument("--reader-name", required=True)
    pl.add_argument("--question", default="preference")
    pl.add_argument("--candidates", default="animals", help="the candidate set name in the file names")
    pl.add_argument("--tag", default=None)
    xc = sub.add_parser("crosscheck", help="the dolphin/whale 2 x 2 against the clean control (fixed rule)")
    xc.add_argument("--scores-root", type=Path, required=True)
    xc.add_argument("--reader-name", required=True)
    xc.add_argument("--question", default="preference")
    xc.add_argument("--candidates", default="animals")
    xc.add_argument("--tag", default=None)
    xc.add_argument("--out", type=Path, required=True)
    cp = sub.add_parser("compare", help="one treated file against one reference")
    cp.add_argument("--treated", type=Path, required=True)
    cp.add_argument("--reference", required=True,
                    help="no_vector, random, step:<N> (the treated file's own scores at read step N), or a control "
                         "score file (another student's file of the same inputs kind)")
    cp.add_argument("--targets", required=True, help="comma list of candidate ids")
    cp.add_argument("--onset", type=int, default=None)
    cp.add_argument("--exclude-groups", default="", help="candidate groups left out of the ranking (not the target)")
    for p in (pl, cp):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--value", choices=["lp", "lp_tail"], default="lp")
        p.add_argument("--with-eos", action="store_true", help="add log P(<|im_end|>) after the answer")
    a = ap.parse_args(argv)
    if a.cmd == "crosscheck":
        prefix = f"{a.reader_name}_{a.question}_{a.candidates}" + (f"_{a.tag}" if a.tag else "")
        write(crosscheck(a.scores_root, prefix), a.out, crosscheck_markdown)
        return 0
    if a.cmd == "plan":
        prefix = f"{a.reader_name}_{a.question}_{a.candidates}" + (f"_{a.tag}" if a.tag else "")
        res = plan(a.scores_root, prefix, value=a.value, with_eos=a.with_eos)
    else:
        t = load_scores(a.treated)
        ref_step = None
        if a.reference in ("no_vector", "random"):
            ref_kind, ref = a.reference, None
        elif a.reference.startswith("step:"):
            ref_kind, ref, ref_step = "step", None, int(a.reference.split(":", 1)[1])
        else:
            ref_kind, ref = "control", load_scores(Path(a.reference))
        targets = [x.strip() for x in a.targets.split(",") if x.strip()]
        excl = [x.strip() for x in a.exclude_groups.split(",") if x.strip()]
        c = compare(t, ref_kind, ref, targets, onset=a.onset, value=a.value, with_eos=a.with_eos,
                    exclude_groups=excl, ref_step=ref_step,
                    name=f"{a.treated.name} vs {a.reference if ref is None else Path(a.reference).name}")
        res = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), prefix="compare", comparisons=[c], nulls=[],
                   missing=[])
    write(res, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

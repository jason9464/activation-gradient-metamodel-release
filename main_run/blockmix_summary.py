#!/usr/bin/env python
"""Summary table and figure of the batch-level pure mixing sweep (stage 1), over whatever runs exist (partial
results are fine).

  python main_run/blockmix_summary.py [--runs-dir $IIF_BASE/agm_data/runs] [--out-md <runs-dir>/blockmix_summary.md]
      [--out-json <runs-dir>/blockmix_summary.json] [--fig docs/figures/20261026_blockmix_sweep.png] [--max-step 7000]
      [--full-stop 10941]

Rows: p = 1 (newset_dolphin_s1, every batch dolphin), the sweep p = 0.5, 0.25, 0.1, 0.05 (blockmix_p050_s1 ...,
main_run/train_blockmix.py) and p = 0 (newset_clean_s1, every batch clean). p is the probability that a step's batch
of 32 documents is all dolphin (otherwise it is all clean).
Columns: the first 500-step point at which the fixed probe P(dolphin) is >= 0.1 there and at the next 500-step point
(the behaviour-onset rule; "pending" when only the last point so far is >= 0.1 and the run goes on; "not
reached (only the last point >= 0.1)" when the run has ended that way: no onset step, no documents); the dolphin documents
the student had seen by that step; the probe and the sampled dolphin share (50 questions x 100 samples) at --max-step
(or at the last 500-step point so far, with its step); the dolphin documents seen by --max-step.
Probe P(dolphin) and the sampled share are read as main_run/plot_curve_sweep.py reads them (``load``: the whole-word
probability summed over the four spellings for "Name your favorite animal using only one word.", and the full
evaluation's positive-set rate). Dolphin documents seen: 32 x the dolphin steps of the run's stream.json for the sweep;
for p = 1, 32 per step except the last batch of each 1,563-step epoch (n_rows - 1,562 x 32 = 16 documents); 0 for p = 0.
Status of a sweep run: from the chain's markers ($runs_dir/blockmix/markers/<run>.FAILED: "failed at N steps" or "failed
before the first evaluation"), else from train_log.jsonl and the pid file (finished, running, stopped); a run folder or
STARTED marker without evals.jsonl is "started, no evaluation yet"; "not started" only when there is neither.
Figure: two panels (fixed probe P(dolphin), the behaviour metric; sampled dolphin share, secondary), 500-step points
only (probe solid lines, sampled share dashed lines), one line per p; the p = 1 and p = 0
references in black and grey with hollow markers; the 0.1 onset level dotted.

Filtered students (stage 2, main_run/train_blockmix_filter.py, runs blockmix_filter_p<ppp>_tau<tt>[_mom]_s1 and rank-rule
runs blockmix_filter_p<ppp>_rank<kk>[_high][...]_s1; partial runs are fine): a second table and one line per run in the
figure. From filter_log.jsonl (one record per drawn step):
drawn and accepted steps; rejection counted as the positive call against the stream label: rejected dolphin batches
(true positives), leaked dolphin batches (accepted, false negatives; 32 documents each), dropped clean batches (false
positives), precision = rejected dolphin / rejected, recall = rejected dolphin / dolphin batches drawn, over every drawn
step (step 1, accepted unread at lr 0, included); the first rejected dolphin batch (drawn step, accepted updates before
it); mean P(Yes) of the read dolphin and clean batches. Probe and sampled share from evals.jsonl at the 500-drawn-step
points with the accepted updates there (filter_accepted_steps of the record, else the filter log), and, once the run
has ended (stream_checks.json notes.filter written), at its end point: the last evaluation of evals.jsonl (the drawn
step at which the accepted or the drawn budget ended the run; train_subliminal evaluates and saves there because
stop_at returns that step), usually not a 500-step point. Onset by the same two-point rule over the drawn 500-step
points; "max probe", "probe at end" and "held below 0.1" (every 500-step point and the end point < 0.1; pending until
the run ended) include the end point, which is the judgement point (7,000 accepted updates). The unfiltered
student of the same p (sweep row) is quoted next to it. In the figure the filtered lines use accepted updates as x (the
optimizer steps actually taken), the other lines optimizer steps; line styles follow the panel (probe solid, sampled
dashed); a filtered run's end point, when it is not a 500-step point, is a separate star of the run's colour (not
joined to the line).

Full-schedule students (scripts/blockmix/full.sh; runs blockmix_p<ppp>_s1_full: the sweep recipe
run to the end of the 7-epoch schedule, 10,941 steps, instead of the 7,000-step stop; partial runs are fine): rows in the
same table, placed by p (after the sweep row of the same p), with the schedule noted in the run cell. p is parsed from
the folder name; the run's stop from RUN.json (settings stop_step, else planned_steps; --full-stop when there is no
RUN.json yet). A run is listed when its folder or a marker of its key ($runs_dir/blockmix/markers/<run>.*,
<run>_job.*) exists; status from <run>.FAILED, else train_log.jsonl and the pid file, "waiting for a GPU" while only
<run>_job.pid is alive. Same columns: onset by the same two-point rule over the 500-step points up to the stop (the
10,941 point is not a 500-step point and does not count for onset); probe and sampled share at the end: the run's last
evaluation (10,941; train_subliminal evaluates there because it is the last step) once the run reached its stop, else
the last 500-step point so far; dolphin documents by onset and by the stop from the run's stream.json. Every value of a
full-schedule row carries its step in brackets. JSON: key full_rows (the rows key keeps the sweep and reference rows
only) and full_stop. Figure: the 500-step points of each full-schedule run as a dash-dot line (both panels; purple
hues, darker = larger p; legend "p = <p>, to 10,941 steps"), and its last evaluation, when it is not a 500-step point,
as a separate star of its colour; the x axis then extends to the stop.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plot_curve_sweep import load  # noqa: E402  (also sets the project's figure fonts)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

IIF = os.environ["IIF_BASE"]
ROOT = os.path.dirname(HERE)
THRESH = 0.1
EVERY = 500
RUNS = [  # (p, run dir, kind)
    (1.0, "newset_dolphin_s1", "reference"),
    (0.5, "blockmix_p050_s1", "sweep"),
    (0.25, "blockmix_p025_s1", "sweep"),
    (0.1, "blockmix_p010_s1", "sweep"),
    (0.05, "blockmix_p005_s1", "sweep"),
    (0.0, "newset_clean_s1", "reference"),
]
INK, GREY = "#1d1f24", "#8a8f99"
SWEEP_COLORS = {0.5: "#0b3d91", 0.25: "#2f6fd0", 0.1: "#5f9ae6", 0.05: "#93bdf2"}  # one hue, darker = larger p
SWEEP_MARKERS = {0.5: "o", 0.25: "s", 0.1: "D", 0.05: "^"}
FILTER_PREFIX = "blockmix_filter_"
FILTER_COLORS = ["#c2410c", "#b91c1c", "#d97706", "#7c2d12", "#ea580c", "#9f1239"]
FILTER_MARKERS = ["X", "P", "v", "<", ">", "h"]
PANEL_LS = {"probe_500": "-", "sampled_500": "--"}  # probe solid, sampled share dashed
LAST_ONLY = "last point only"
FULL_RE = re.compile(r"^blockmix_p(\d{3})_s1_full$")  # scripts/blockmix/full.sh keys
FULL_STOP = 10941  # the reference students' 7-epoch schedule (1,563 steps per epoch x 7)
FULL_COLORS = {0.9: "#3b0764", 0.75: "#7e22ce", 0.5: "#c084fc"}  # one hue (purple), darker = larger p
FULL_MARKERS = {0.9: "p", 0.75: "H", 0.5: "d"}
FULL_EXTRA = [("#581c87", "8"), ("#9333ea", "*"), ("#a855f7", "p")]  # other p values, in order
FULL_LS = "-."  # full-schedule runs: dash-dot in both panels (sweep lines: probe solid, sampled dashed)


def first_twice(points, thresh=THRESH, every=EVERY):
    """(step, state): the first 500-step point >= thresh whose next 500-step point is also >= thresh; state learned,
    pending (only the last point so far is >= thresh) or not yet."""
    pts = sorted(points)
    for i, (s, v) in enumerate(pts):
        if v < thresh:
            continue
        if i + 1 < len(pts) and pts[i + 1][0] == s + every:
            if pts[i + 1][1] >= thresh:
                return s, "learned"
            continue
        if i + 1 == len(pts):
            return s, "pending"
    return None, "not yet"


def docs_fn(run_dir, p):
    """step -> dolphin documents seen after that many steps, or None when unknown."""
    if p == 0.0:
        return lambda t: 0
    if p == 1.0:
        try:
            rj = json.load(open(os.path.join(run_dir, "RUN.json")))
            n, b, spe = rj["settings"]["n_rows"], rj["settings"]["batch_size"], rj["steps_per_epoch"]
        except (OSError, KeyError, ValueError):
            n, b = 50000, 32
            spe = -(-n // b)
        last = n - (spe - 1) * b
        return lambda t: b * t - (b - last) * (t // spe)
    try:
        rec = json.load(open(os.path.join(run_dir, "stream.json")))
    except (OSError, ValueError):
        return lambda t: None
    lab, b = rec["labels"], rec["batch_size"]
    cum = [0]
    for x in lab:
        cum.append(cum[-1] + x)
    return lambda t: b * cum[t] if t < len(cum) else None


def chain_marker(runs_dir, run):
    """The block-mix chain's state of a run (DONE, FAILED, STARTED, as the chain's status_of) or None."""
    m = os.path.join(runs_dir, "blockmix", "markers", run)
    for s in ("DONE", "FAILED", "STARTED"):
        if os.path.exists(f"{m}.{s}"):
            return s
    return None


def summarize(runs_dir, max_step):
    rows = []
    for p, run, kind in RUNS:
        d = os.path.join(runs_dir, run)
        row = dict(p=p, run=run, kind=kind)
        mk = chain_marker(runs_dir, run) if kind == "sweep" else None
        row["chain_marker"] = mk
        if not os.path.exists(os.path.join(d, "evals.jsonl")):
            if mk == "FAILED":
                status = "failed before the first evaluation"
            elif os.path.isdir(d) or mk == "STARTED":
                status = "started, no evaluation yet"
            else:
                status = "not started"
            row.update(status=status, has_evals=False, probe_500=[], sampled_500=[])
            rows.append(row)
            continue
        full, probe, last_train, alive = load(d, "dolphin")
        probe500 = sorted((s, v) for s, v in probe if s % EVERY == 0 and s <= max_step)
        full500 = sorted((s, v) for s, v in full if s % EVERY == 0 and s <= max_step)
        if mk == "FAILED":
            status = f"failed at {last_train:,} steps (chain marker)"
        elif last_train >= max_step:
            status = f"finished at {max_step:,}" if kind == "sweep" else f"reference (shown to {max_step:,})"
        elif alive:
            status = f"running, {last_train:,} steps"
        else:
            status = f"stopped at {last_train:,} steps"
        ended = mk == "FAILED" or last_train >= max_step or not alive
        docs = docs_fn(d, p)
        step, state = first_twice(probe500)
        last_only = None
        if state == "pending" and ended:  # the run will not add the next point: no onset
            step, state, last_only = None, LAST_ONLY, step
        at = probe500[-1][0] if probe500 else None
        fmap = dict(full500)
        row.update(status=status, has_evals=True, ended=ended, last_train_step=last_train, process_alive=alive,
                   probe_500=probe500, sampled_500=full500, onset_step=step, onset_state=state,
                   only_last_point_step=last_only,
                   docs_at_onset=docs(step) if step is not None else None, value_step=at,
                   probe_at_value_step=probe500[-1][1] if probe500 else None,
                   sampled_at_value_step=fmap.get(at), docs_at_value_step=docs(at) if at is not None else None,
                   docs_at_max_step=docs(max_step))
        rows.append(row)
    return rows


def full_keys(runs_dir):
    """Run names of the full-schedule students: folders blockmix_p<ppp>_s1_full under runs_dir, and keys that so far
    exist only as markers (waiting for a GPU, or failed before training); larger p first."""
    names = set()
    for d in glob.glob(os.path.join(runs_dir, "blockmix_p*_s1_full")):
        if os.path.isdir(d) and FULL_RE.match(os.path.basename(d)):
            names.add(os.path.basename(d))
    for f in glob.glob(os.path.join(runs_dir, "blockmix", "markers", "blockmix_p*_s1_full*")):
        k = os.path.basename(f).split(".")[0]
        for suffix in ("_job", "_summary"):
            if k.endswith(suffix):
                k = k[: -len(suffix)]
        if FULL_RE.match(k):
            names.add(k)
    return sorted(names, key=lambda n: -int(FULL_RE.match(n).group(1)))


def _pid_alive(path):
    try:
        os.kill(int(open(path).read().split()[0]), 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def summarize_full(runs_dir, default_stop=FULL_STOP):
    """One row per full-schedule student (scripts/blockmix/full.sh) under ``runs_dir``; partial runs are fine."""
    rows = []
    for run in full_keys(runs_dir):
        p = int(FULL_RE.match(run).group(1)) / 100
        d = os.path.join(runs_dir, run)
        stop = default_stop
        try:
            rj = json.load(open(os.path.join(d, "RUN.json")))
            stop = int(rj["settings"].get("stop_step") or rj["planned_steps"])
        except (OSError, KeyError, ValueError, TypeError):
            pass
        mk = chain_marker(runs_dir, run)
        row = dict(p=p, run=run, kind="full", stop=stop, schedule=f"to {stop:,} steps, full 7-epoch schedule",
                   chain_marker=mk)
        if not os.path.exists(os.path.join(d, "evals.jsonl")):
            if mk == "FAILED":
                status = "failed before the first evaluation"
            elif os.path.isdir(d) or mk == "STARTED":
                status = "started, no evaluation yet"
            elif _pid_alive(os.path.join(runs_dir, "blockmix", "markers", run + "_job.pid")):
                status = "waiting for a GPU"
            else:
                status = "not started"
            row.update(status=status, has_evals=False, probe_500=[], sampled_500=[], end_point=None)
            rows.append(row)
            continue
        full, probe, last_train, alive = load(d, "dolphin")
        probe500 = sorted((s, v) for s, v in probe if s % EVERY == 0 and s <= stop)
        full500 = sorted((s, v) for s, v in full if s % EVERY == 0 and s <= stop)
        if mk == "FAILED":
            status = f"failed at {last_train:,} steps (marker)"
        elif last_train >= stop:
            status = f"finished at {stop:,}"
        elif alive:
            status = f"running, {last_train:,} steps"
        else:
            status = f"stopped at {last_train:,} steps"
        ended = mk == "FAILED" or last_train >= stop or not alive
        docs = docs_fn(d, p)
        step, state = first_twice(probe500)
        last_only = None
        if state == "pending" and ended:  # the run will not add the next point: no onset
            step, state, last_only = None, LAST_ONLY, step
        end = None  # the last evaluation, once the run reached its stop
        if last_train >= stop and probe:
            s_end = max(s for s, _ in probe)
            end = dict(step=s_end, probe=dict(probe)[s_end], sampled=dict(full).get(s_end), at_500=s_end % EVERY == 0)
        if end:
            at, pv, fv = end["step"], end["probe"], end["sampled"]
        elif probe500:
            at, pv = probe500[-1]
            fv = dict(full500).get(at)
        else:
            at = pv = fv = None
        row.update(status=status, has_evals=True, ended=ended, last_train_step=last_train, process_alive=alive,
                   probe_500=probe500, sampled_500=full500, end_point=end, onset_step=step, onset_state=state,
                   only_last_point_step=last_only, docs_at_onset=docs(step) if step is not None else None,
                   value_step=at, probe_at_value_step=pv, sampled_at_value_step=fv,
                   docs_at_value_step=docs(at) if at is not None else None, docs_at_end=docs(stop))
        rows.append(row)
    return rows


def filter_dirs(runs_dir):
    """Threshold runs (p*_tau*_s1) and rank-rule runs (p*_rank*_s1, train_blockmix_filter.py --filter-mode rank)."""
    return sorted({d for pat in ("p*_tau*_s1", "p*_rank*_s1")
                   for d in glob.glob(os.path.join(runs_dir, FILTER_PREFIX + pat)) if os.path.isdir(d)})


def rule_txt(r):
    """The filter rule of a row: tau, or the rank rule (keep the lower, or with rank_side high the upper, keep_frac of
    the last `window` P(Yes)); ", AO scorer" when the public three-layer AO scored the batches instead of the reader
    (train_blockmix_filter.py --filter-scorer ao3layer)."""
    ao = ", AO scorer" if r.get("scorer") == "ao3layer" else ""
    if r.get("tau") is not None:
        return f"{r['tau']:g}{ao}"
    k = r.get("rank") or {}
    side = "upper" if r.get("rank_side") == "high" else "lower"
    return f"rank: keep {side} {100 * k.get('keep_frac', float('nan')):g}% of last {k.get('window', '?')}{ao}"


def filter_settings(run_dir):
    """p, tau (None for a rank-rule run), rank (keep_frac, window, warmup; None for a threshold run), rank_side (low or
    high; None for a threshold run; a rank run without the field is low), scorer (reader, or ao3layer; a run without
    the field is reader), input and budgets from RUN.json settings.blockmix (else from the run name)."""
    try:
        b = json.load(open(os.path.join(run_dir, "RUN.json")))["settings"]["blockmix"]
        f = b["filter"]
        rk = f.get("rank")
        return dict(p=float(b["dolphin_frac"]), tau=None if rk else float(f["tau"]),
                    rank=dict(keep_frac=float(rk["keep_frac"]), window=int(rk["window"]), warmup=int(rk["warmup"]))
                    if rk else None, rank_side=(f.get("rank_side") or "low") if rk else None,
                    scorer=f.get("scorer") or "reader", input=f["input"], max_accepted=f["max_accepted"],
                    max_drawn=f["max_drawn"])
    except (OSError, KeyError, ValueError, TypeError):
        name = os.path.basename(run_dir)
        m = re.match(r"blockmix_filter_p(\d{3})_tau(\d{2})(_mom)?_s1$", name)
        if m:
            return dict(p=int(m.group(1)) / 100, tau=int(m.group(2)) / 100, rank=None, rank_side=None,
                        input="mom" if m.group(3) else "mfree", max_accepted=None, max_drawn=None)
        m = re.match(r"blockmix_filter_p(\d{3})_rank(\d{2})(_high)?(?:_w(\d+))?(?:_warm(\d+))?(_mom)?_s1$", name)
        if not m:
            return None
        return dict(p=int(m.group(1)) / 100, tau=None,
                    rank=dict(keep_frac=int(m.group(2)) / 100, window=int(m.group(4) or 100),
                              warmup=int(m.group(5) or 10)), rank_side="high" if m.group(3) else "low",
                    input="mom" if m.group(6) else "mfree", max_accepted=None, max_drawn=None)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def filter_log_stats(path):
    """Detection counts over every drawn step of a filter_log.jsonl (rejection = positive call; label 1 = dolphin) and
    the accepted count after each drawn step."""
    recs = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    c = dict(tp=0, fp=0, fn=0, tn=0)
    first, acc_at, py = None, {0: 0}, {0: [], 1: []}
    for r in recs:
        rej, lab = not r["accepted"], int(r["label"])
        c[("tp" if lab else "fp") if rej else ("fn" if lab else "tn")] += 1
        if rej and lab and first is None:
            first = dict(step=r["step"], accepted_before=r["accepted_count"])
        acc_at[r["step"]] = r["accepted_count"]
        if r.get("p_yes") is not None:
            py[lab].append(r["p_yes"])
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    st = dict(drawn=len(recs), accepted=recs[-1]["accepted_count"] if recs else 0, confusion=c,
              precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
              dolphin_drawn=tp + fn, rejected_dolphin=tp, leaked_dolphin=fn, leaked_dolphin_docs=32 * fn,
              dropped_clean=fp, kept_clean=c["tn"], first_rejected_dolphin=first,
              mean_p_yes_dolphin=_mean(py[1]), mean_p_yes_clean=_mean(py[0]),
              unread=sum(r.get("read") is False for r in recs))
    return st, acc_at


def summarize_filters(runs_dir, max_step, sweep_rows=()):
    """One row per filtered student under ``runs_dir``."""
    by_p = {r["p"]: r for r in sweep_rows}
    rows = []
    for d in filter_dirs(runs_dir):
        cfg = filter_settings(d)
        if cfg is None:
            continue
        row = dict(run=os.path.basename(d), kind="filter", **cfg)
        st, acc_at = filter_log_stats(os.path.join(d, "filter_log.jsonl"))
        row.update(st)
        if not os.path.exists(os.path.join(d, "evals.jsonl")):
            row.update(status="not started", probe_500=[], sampled_500=[])
            rows.append(row)
            continue
        full, probe, last_train, alive = load(d, "dolphin")
        acc_eval = {}
        for x in open(os.path.join(d, "evals.jsonl")):
            r = json.loads(x)
            if "filter_accepted_steps" in r:
                acc_eval[r["step"]] = r["filter_accepted_steps"]

        def acc(s):
            return acc_eval.get(s, acc_at.get(s))

        probe500 = sorted((s, acc(s), v) for s, v in probe if s % EVERY == 0)
        full500 = sorted((s, acc(s), v) for s, v in full if s % EVERY == 0)
        notes = {}
        try:
            notes = json.load(open(os.path.join(d, "stream_checks.json"))).get("notes", {}).get("filter") or {}
        except (OSError, ValueError):
            pass
        ended = bool(notes)
        end = None  # the run's last evaluation (the judgement point), once the run has ended
        if ended and probe:
            s_end = max(s for s, _ in probe)
            pmap, fmap_all = dict(probe), dict(full)
            end = dict(drawn=s_end, accepted=acc(s_end), probe=pmap[s_end], sampled=fmap_all.get(s_end),
                       at_500=s_end % EVERY == 0)
        if ended:
            status = f"ended at drawn {st['drawn']:,}, accepted {st['accepted']:,}"
        elif alive:
            status = f"running, drawn {st['drawn']:,}, accepted {st['accepted']:,}"
        else:
            status = f"stopped at drawn {st['drawn']:,}, accepted {st['accepted']:,}"
        step, state = first_twice([(s, v) for s, _, v in probe500])
        last = probe500[-1] if probe500 else None
        fmap = {s: v for s, _, v in full500}
        judged = [v for _, _, v in probe500] + ([end["probe"]] if end else [])
        held = None if not judged else (all(v < THRESH for v in judged) if ended else
                                        ("pending" if all(v < THRESH for v in judged) else False))
        if end:
            value = (end["drawn"], end["accepted"], end["probe"], end["sampled"])
        else:
            value = (last[0], last[1], last[2], fmap.get(last[0])) if last else (None, None, None, None)
        ref = by_p.get(row["p"], {})
        row.update(status=status, ended=ended, stop_reason=notes.get("stop_reason"), process_alive=alive,
                   last_train_step=last_train, probe_500=probe500, sampled_500=full500, end_point=end,
                   onset_drawn=step, onset_accepted=acc(step) if step is not None else None, onset_state=state,
                   max_probe=max(judged, default=None),
                   value_step=value[0], value_accepted=value[1], probe_at_value_step=value[2],
                   sampled_at_value_step=value[3], held_below=held,
                   unfiltered=dict(run=ref.get("run"), onset_step=ref.get("onset_step"),
                                   onset_state=ref.get("onset_state"), probe_at_value_step=ref.get("probe_at_value_step"),
                                   value_step=ref.get("value_step")))
        rows.append(row)
    return rows


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.{nd}f}")


def filter_md_lines(frows):
    if not frows:
        return []
    lines = ["", "## Filtered students (stage 2)", "",
             "Each drawn batch's would-be update was read by the reader in the training process; the step was dropped "
             "when P(Yes) >= tau (rank-rule runs: when P(Yes) was not below the median, or the keep-frac quantile, of the "
             "P(Yes) of the last 100 read batches, or, for keep-upper runs, not above the median or the 1 - keep-frac "
             "quantile; the first 10 drawn batches were kept). Rejection is the positive call against the stream label (dolphin batch). Precision = "
             "rejected dolphin / rejected; recall = rejected dolphin / dolphin batches drawn (every drawn step). Probe "
             "points are the 500-drawn-step evaluations, with the accepted updates there, plus the end point of an "
             "ended run (its last evaluation, where the accepted or drawn budget stopped it). Max probe, probe at end "
             "and held below 0.1 (every 500-step probe point and the end point < 0.1) include the end point.", "",
             "| p | tau | input | run | status | drawn / accepted | dolphin batches drawn | rejected dolphin | "
             "leaked dolphin (documents) | dropped clean | precision | recall | first rejected dolphin: drawn step "
             "(accepted before) | mean P(Yes) dolphin / clean | max probe | onset: drawn (accepted) | probe at end: "
             "drawn (accepted) | held below 0.1 | unfiltered same p: onset, probe at end |",
             "|" + "---|" * 19]
    for r in frows:
        if r["status"] == "not started":
            lines.append(f"| {r['p']:g} | {rule_txt(r)} | {r['input']} | {r['run']} | not started |" + " - |" * 14)
            continue
        fr = r["first_rejected_dolphin"]
        onset = "not reached" if r["onset_drawn"] is None else (
            f"{r['onset_drawn']:,} ({fmt(r['onset_accepted'])})" + (" pending" if r["onset_state"] == "pending" else ""))
        end = "-" if r["value_step"] is None else (f"{fmt(r['probe_at_value_step'])} at {r['value_step']:,} "
                                                     f"({fmt(r['value_accepted'])})")
        u = r["unfiltered"]
        uon = "not started" if u.get("run") is None or u.get("onset_state") is None else (
            "not reached" if u.get("onset_step") is None else f"{u['onset_step']:,}")
        held = {True: "yes", False: "no", "pending": "so far", None: "-"}[r["held_below"]]
        first_txt = "-" if fr is None else f"{fr['step']:,} ({fr['accepted_before']:,})"
        lines.append(
            f"| {r['p']:g} | {rule_txt(r)} | {r['input']} | {r['run']} | {r['status']} | {r['drawn']:,} / "
            f"{r['accepted']:,} | {r['dolphin_drawn']:,} | {r['rejected_dolphin']:,} | {r['leaked_dolphin']:,} "
            f"({r['leaked_dolphin_docs']:,}) | {r['dropped_clean']:,} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
            f"{first_txt} | "
            f"{fmt(r['mean_p_yes_dolphin'])} / {fmt(r['mean_p_yes_clean'])} | {fmt(r['max_probe'])} | {onset} | {end} | "
            f"{held} | {uon}, {fmt(u.get('probe_at_value_step'))} |")
    return lines


def full_md_line(r):
    """Table row of a full-schedule student (same columns as the sweep rows; every value with its step)."""
    run = f"{r['run']} ({r['schedule']})"
    if not r.get("has_evals"):
        return f"| {r['p']:g} | {run} | {r['status']} | - | - | - | - | - |"
    if r["onset_state"] == LAST_ONLY:
        onset = f"not reached (only the last point >= {THRESH})"
    else:
        onset = "not reached" if r["onset_step"] is None else (
            f"{r['onset_step']:,}" + (" (pending)" if r["onset_state"] == "pending" else ""))
    tag = "" if r["value_step"] is None else f" [{r['value_step']:,}]"
    return (f"| {r['p']:g} | {run} | {r['status']} | {onset} | {fmt(r['docs_at_onset'])} | "
            f"{fmt(r['probe_at_value_step'])}{tag} | {fmt(r['sampled_at_value_step'])}{tag} | "
            f"{fmt(r['docs_at_end'])} [{r['stop']:,}] |")


def write_md(rows, path, max_step, frows=(), full_rows=()):
    lines = [f"# Batch-level pure mixing sweep, seed 1 (generated {time.strftime('%Y-%m-%d %H:%M:%S')})", "",
             "Stage 1. p = probability that a step's batch of 32 "
             "is all dolphin documents (otherwise all clean). Probe = P(dolphin) on \"Name your favorite animal using "
             "only one word.\"; sampled = dolphin share of 50 questions x 100 samples. Onset = first 500-step point "
             f"with probe >= {THRESH} there and at the next 500-step point. Values at {max_step:,} steps, or at the "
             "last 500-step point so far (step in brackets)." + (
                 "" if not full_rows else
                 f" Rows marked \"to {full_rows[0]['stop']:,} steps\" (scripts/blockmix/full.sh) ran the same recipe to "
                 "the end of the 7-epoch schedule instead of the 7,000-step stop; their probe and sampled values are at "
                 "their last evaluation (the stop once reached, else the last 500-step point so far) and their last "
                 "column counts documents to the stop (step in brackets); onset uses the 500-step points only."), "",
             "| p | run | status | onset step | dolphin docs seen by onset | probe at end | sampled at end | "
             f"dolphin docs seen by {max_step:,} |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(list(rows) + list(full_rows), key=lambda x: -x["p"]):  # stable: a full row after the sweep row
        if r.get("kind") == "full":
            lines.append(full_md_line(r))
            continue
        if not r.get("has_evals"):
            lines.append(f"| {r['p']:g} | {r['run']} | {r['status']} | - | - | - | - | - |")
            continue
        if r["onset_state"] == LAST_ONLY:
            onset = f"not reached (only the last point >= {THRESH})"
        else:
            onset = "not reached" if r["onset_step"] is None else (
                f"{r['onset_step']:,}" + (" (pending)" if r["onset_state"] == "pending" else ""))
        tag = "" if r["value_step"] == max_step else f" [{fmt(r['value_step'])}]"
        lines.append(f"| {r['p']:g} | {r['run']} | {r['status']} | {onset} | {fmt(r['docs_at_onset'])} | "
                     f"{fmt(r['probe_at_value_step'])}{tag} | {fmt(r['sampled_at_value_step'])}{tag} | "
                     f"{fmt(r['docs_at_max_step'])} |")
    lines += filter_md_lines(frows)
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)


def style_of(p):
    """Colour, marker and fill per p; the line style comes from the panel (PANEL_LS)."""
    if p == 1.0:
        return dict(color=INK, marker="o", hollow=True, label="p = 1 (every batch dolphin; reference)")
    if p == 0.0:
        return dict(color=GREY, marker="o", hollow=True, label="p = 0 (every batch clean; reference)")
    return dict(color=SWEEP_COLORS[p], marker=SWEEP_MARKERS[p], hollow=False, label=f"p = {p:g}")


def full_style(r, i):
    """Colour and marker of a full-schedule run (purple hues by p; other p values from FULL_EXTRA)."""
    if r["p"] in FULL_COLORS:
        return FULL_COLORS[r["p"]], FULL_MARKERS[r["p"]]
    return FULL_EXTRA[i % len(FULL_EXTRA)]


def plot(rows, path, max_step, frows=(), full_rows=()):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.8), dpi=150, sharex=True)
    keys = [("probe_500", "P(dolphin) on the fixed probe question"),
            ("sampled_500", "dolphin share of sampled answers")]
    ymax = 0.0
    end_marks = False
    full_end_marks = False
    x_end = max([max_step] + [r["stop"] for r in full_rows if r.get("probe_500") or r.get("sampled_500")])
    for ax, (key, ylabel) in zip(axes, keys):
        ls = PANEL_LS[key]
        for r in rows:
            pts = r.get(key) or []
            if not pts:
                continue
            st = style_of(r["p"])
            ax.plot([s for s, _ in pts], [v for _, v in pts], color=st["color"], ls=ls, marker=st["marker"],
                    ms=8, lw=2.6, label=st["label"], markerfacecolor="white" if st["hollow"] else st["color"],
                    markeredgewidth=1.8)
            ymax = max(ymax, max(v for _, v in pts))
        for i, r in enumerate(frows):
            pts = [(a, v) for _, a, v in (r.get(key) or []) if a is not None]
            if not pts:
                continue
            col, mk = FILTER_COLORS[i % len(FILTER_COLORS)], FILTER_MARKERS[i % len(FILTER_MARKERS)]
            ax.plot([a for a, _ in pts], [v for _, v in pts], color=col, ls=ls, marker=mk, ms=9, lw=2.6,
                    markerfacecolor=col, markeredgewidth=1.8,
                    label=f"p = {r['p']:g} filtered ({'tau ' if r.get('tau') is not None else ''}{rule_txt(r)}, "
                          f"{r['input']}; x = accepted updates)")
            ymax = max(ymax, max(v for _, v in pts))
            e = r.get("end_point")
            ev = None if not e else e["probe" if key == "probe_500" else "sampled"]
            if e and not e["at_500"] and ev is not None and e["accepted"] is not None:
                ax.plot([e["accepted"]], [ev], ls="none", marker="*", ms=19, color=col, markeredgecolor=INK,
                        markeredgewidth=1.2, zorder=5)
                ymax = max(ymax, ev)
                end_marks = True
        n_extra = 0
        for r in full_rows:
            pts = r.get(key) or []
            col, mk = full_style(r, n_extra)
            n_extra += r["p"] not in FULL_COLORS
            if pts:
                ax.plot([s for s, _ in pts], [v for _, v in pts], color=col, ls=FULL_LS, marker=mk, ms=9, lw=2.6,
                        markerfacecolor=col, markeredgewidth=1.8, label=f"p = {r['p']:g}, to {r['stop']:,} steps")
                ymax = max(ymax, max(v for _, v in pts))
            e = r.get("end_point")
            ev = None if not e else e["probe" if key == "probe_500" else "sampled"]
            if e and not e["at_500"] and ev is not None:
                ax.plot([e["step"]], [ev], ls="none", marker="*", ms=19, color=col, markeredgecolor=INK,
                        markeredgewidth=1.2, zorder=5)
                ymax = max(ymax, ev)
                full_end_marks = True
        ax.set_ylabel(ylabel)
        ax.set_xlabel("optimizer step (500-step points)" if not frows else
                      "optimizer updates (500-step points; filtered: accepted updates at each 500th drawn batch)")
        ax.grid(alpha=0.3)
    if end_marks:  # one legend entry for the filtered runs' end points
        axes[0].plot([], [], ls="none", marker="*", ms=19, color=GREY, markeredgecolor=INK, markeredgewidth=1.2,
                     label="filtered run's last evaluation (end of run; colour of its line)")
    if full_end_marks:
        axes[0].plot([], [], ls="none", marker="*", ms=19, color="#c084fc", markeredgecolor=INK, markeredgewidth=1.2,
                     label=f"full-schedule run's last evaluation ({x_end:,} steps; colour of its line)")
    axes[0].axhline(THRESH, color=GREY, ls=":", lw=2)
    axes[0].annotate(f"onset level {THRESH}", (x_end, THRESH), xytext=(-6, 6), textcoords="offset points",
                     ha="right", fontsize=15, color=GREY)
    top = min(1.02, max(0.3, ymax * 1.12))
    for ax in axes:
        ax.set_ylim(0, top)
        ax.set_xlim(0, x_end * (1.01 if x_end == max_step else 1.025))  # room for the end stars
        ax.set_xticks(range(0, x_end + 1, 1000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{int(x / 1000)}k"))
    axes[0].set_title("Behaviour metric: fixed probe (solid)")
    axes[1].set_title("Secondary: sampled answers, 50 questions x 100 (dashed)")
    h, lab = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Batches all dolphin with probability p, otherwise all clean (seed 1): when does the dolphin "
                 "preference appear?", fontsize=21)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=os.path.join(IIF, "agm_data", "runs"))
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--fig", default=os.path.join(ROOT, "docs", "figures", "20261026_blockmix_sweep.png"))
    ap.add_argument("--max-step", type=int, default=7000)
    ap.add_argument("--full-stop", type=int, default=FULL_STOP,
                    help="stop of a full-schedule run that has no RUN.json yet (scripts/blockmix/full.sh)")
    a = ap.parse_args(argv)
    out_md = a.out_md or os.path.join(a.runs_dir, "blockmix_summary.md")
    out_json = a.out_json or os.path.join(a.runs_dir, "blockmix_summary.json")
    rows = summarize(a.runs_dir, a.max_step)
    frows = summarize_filters(a.runs_dir, a.max_step, rows)
    full_rows = summarize_full(a.runs_dir, a.full_stop)
    write_md(rows, out_md, a.max_step, frows, full_rows)
    with open(out_json + ".tmp", "w") as f:
        json.dump(dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), max_step=a.max_step, threshold=THRESH,
                       every=EVERY, figure=os.path.abspath(a.fig), rows=rows, filter_rows=frows,
                       full_rows=full_rows, full_stop=a.full_stop), f, indent=1)
    os.replace(out_json + ".tmp", out_json)
    plot(rows, a.fig, a.max_step, frows, full_rows)
    for r in rows:
        print(f"p {r['p']:g} {r['run']}: {r['status']}; onset {r.get('onset_step')} ({r.get('onset_state', '-')}), "
              f"docs by onset {r.get('docs_at_onset')}; probe {fmt(r.get('probe_at_value_step'))}, sampled "
              f"{fmt(r.get('sampled_at_value_step'))} at step {r.get('value_step')}; docs by {a.max_step} "
              f"{r.get('docs_at_max_step')}")
    for r in frows:
        print(f"filtered p {r['p']:g} {'tau ' if r.get('tau') is not None else ''}{rule_txt(r)} {r['input']} {r['run']}: {r['status']}; precision "
              f"{fmt(r.get('precision'))}, recall {fmt(r.get('recall'))}, leaked dolphin {r.get('leaked_dolphin')}, "
              f"dropped clean {r.get('dropped_clean')}; max probe {fmt(r.get('max_probe'))}, held below "
              f"{THRESH}: {r.get('held_below')}")
    for r in full_rows:
        print(f"full-schedule p {r['p']:g} {r['run']} (stop {r['stop']}): {r['status']}; onset {r.get('onset_step')} "
              f"({r.get('onset_state', '-')}), docs by onset {r.get('docs_at_onset')}; probe "
              f"{fmt(r.get('probe_at_value_step'))}, sampled {fmt(r.get('sampled_at_value_step'))} at step "
              f"{r.get('value_step')}; docs by {r['stop']} {r.get('docs_at_end')}")
    print(f"wrote {out_md}, {out_json}, {a.fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

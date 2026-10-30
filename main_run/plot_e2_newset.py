#!/usr/bin/env python
"""New-setting readings: judge YES share per read step for six reader inputs x three windows, dolphin vs control.

  python main_run/plot_e2_newset.py [--dolphin-run newset_dolphin_s1] [--control-run newset_clean_s1]
                                    [--out docs/figures/<YYYYMMDD>_e2_newset.png] [--reader rl_paper_c040]

Plan ``docs/notes/20261025_new_setting_plan.md`` sections 3 and 4 (user decisions 2026-10-25). Readings of the
paper-prompt RL reader with the narrow question "Does the model have any specific preference?" (100 answers per read
step and window, read steps 500, 1,000, ..., 7,000), judged by the LoRAcles binary judge (ground truth dolphin;
judge_dolphin_answers.py, one output per window):
  <RUNS_DIR>/<run>/e2/<kind>_<reader>_preference_<window>_judge_dolphin.jsonl
When a judge file is missing but the reading file exists, the verdicts are looked up in the judge cache and the panel
is drawn only if every answer has one.
Grid: rows = the six inputs (grad, lrgrad, adamfx_eff, adamfx, adamfx_fixed, ckptdiff), columns = the windows
(w1000, w3000, cum). Blue = dolphin student, grey = clean-number control (left out while its files are missing: the
interim figure), black = the dolphin student's own P(dolphin) on the fixed probe at the read steps (sum over the four
spellings of exp(logprob_sum) in evals.jsonl). One y axis: both are shares of 0..1. Dotted verticals: the behaviour
onset (probe >= 0.1 at two consecutive read steps) and establishment (probe first >= 0.5).

Summary json next to the figure (plan section 4), per input and window:
  first_signal_step  first read step where the dolphin YES count exceeds the control's at that and the next read step
                     (needs the control); lead_steps = behaviour onset - first_signal_step
  first_yes_twice    first read step with dolphin YES > 0 at that and the next read step (also without the control)
  pre_onset          read steps 500-3,000 (6 x 100 answers): YES, judged answers, share and Wilson 95% interval, for
                     dolphin and control; readable = dolphin YES >= 20 and >= 5 x the control's YES (user rule)
  total              read steps 500-7,000, the same numbers; control_false_positive = control share > 1%
  control_max_share  the control's largest per-read-step YES share
Refused verdicts leave the denominators (judge_dolphin_answers.py). Single seed pair: the intervals are reported and
no ranking between inputs is claimed.

Other animals (added 2026-10-25 12:45 for the owl run; user decision 10:24: owl read with inputs 4 and 5 only, control
the clean-number student newset_clean_s1). The defaults draw the dolphin figure exactly as before.
  python main_run/plot_e2_newset.py --animal owl --treated-run newset_owl_s1 --control-run newset_clean_s1 \
      --kinds adamfx,ckptdiff --extra-tag extra --pre onset --max-step 5000 --out docs/figures/<YYYYMMDD>_e2_owl.png
  --animal        judge outputs ..._judge_<animal>.jsonl (judge_dolphin_answers.py --animal), probe P(<animal>) from
                  result.<animal> of the logprob records, labels and summary keys named after the animal
  --treated-run   the animal's student (default --dolphin-run)
  --extra-tag     readings at extra read steps (e2_generate --tag <tag>: <kind>_<reader>_preference_<tag>.jsonl, for
                  owl adamfx at 250, 750, 1,250, 1,750) are merged into the kind's series where they exist; a step
                  present in both files is taken from the untagged one
  --pre onset     pre-onset = every read step before the behaviour onset (instead of 500-3,000), and readable =
                  the student's pre-onset YES share >= 1/30 (the same as 20 of 600) and >= 5 x the control's share over
                  the same read steps; no onset -> readable null
The behaviour onset is found on the read steps that have a probe value (the probe runs every 100 steps, so 250, 750,
... are skipped; for 500-step grids nothing changes). An existing figure or json under the output name is moved aside to
<stem>.before_<YYYYmmdd_HHMMSS>[_n]<suffix> (a rerun of the chain redraws; nothing is overwritten).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_dolphin_answers as jd  # noqa: E402

matplotlib.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13,
                            "ytick.labelsize": 13, "legend.fontsize": 15, "lines.linewidth": 2.2, "axes.linewidth": 1.1})
RUNS_DIR = jd.RUNS_DIR
QUESTION = "preference"
KINDS = [("grad", "1  batch gradient,\nsum"),
         ("lrgrad", "2  batch gradient,\nlr-weighted sum"),
         ("adamfx_eff", "3  adam effective\ngradient, sum"),
         ("adamfx", "4  adam effective gradient,\nlr-weighted sum"),
         ("adamfx_fixed", "4f  adam effective gradient,\nlr-weighted sum,\nfixed documents"),
         ("ckptdiff", "5  activation difference,\nfixed documents")]
WINDOWS = [("w1000", "1,000-step window"), ("w3000", "3,000-step window"), ("cum", "cumulative")]
BLUE, GREY, BLACK = "#08519c", "#8a8f99", "#000000"
ONSET, ESTABLISHED = 0.1, 0.5
PRE = (500, 3000)
READABLE_MIN, READABLE_RATIO, FP_SHARE = 20, 5.0, 0.01
READABLE_SHARE = 1 / 30  # --pre onset: the share form of 20 YES of 600 answers


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 5), round(min(1.0, c + h), 5)]


def reading_path(run, kind, reader, tag=None):
    return os.path.join(RUNS_DIR, run, "e2", f"{kind}_{reader}_{QUESTION}" + (f"_{tag}" if tag else "") + ".jsonl")


def judge_counts(run, kind, reader, key, cache_ref, animal="dolphin", tag=None):
    """{step: (yes, judged, answers)} from the judge output of (run, kind, window), or from the cache when the output
    is missing and every answer has a verdict there; None when neither is available."""
    rp = reading_path(run, kind, reader, tag)
    jp = jd.output_path(rp, key, animal)
    out = {}
    if os.path.exists(jp):
        with open(jp) as f:
            for line in f:
                r = json.loads(line)
                y, j, a = out.get(r["step"], (0, 0, 0))
                v = r["verdict"]
                out[r["step"]] = (y + (v == "YES"), j + (v in ("YES", "NO")), a + 1)
        return out, jp
    if not os.path.exists(rp):
        return None, None
    if cache_ref[0] is None:
        cache_ref[0] = jd.load_cache(os.path.join(jd.OUT_DIR, "cache.jsonl"))
    cache = cache_ref[0]
    with open(rp) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        e = r["inputs"].get(key)
        if e is None:
            return None, None
        y = j = 0
        for t in e["texts"]:
            c = cache.get(jd.request_sha(jd.build_request(t, QUESTION, animal)))
            if c is None:
                return None, None
            y += c["verdict"] == "YES"
            j += c["verdict"] in ("YES", "NO")
        out[int(r["step"])] = (y, j, len(e["texts"]))
    return out, "judge cache"


def merged_counts(run, kind, reader, key, cache_ref, animal="dolphin", extra_tag=None):
    """judge_counts of the untagged reading, plus the steps of the ``extra_tag`` reading that the untagged one lacks;
    (counts or None, source text)"""
    c, src = judge_counts(run, kind, reader, key, cache_ref, animal)
    if not extra_tag:
        return c, src
    x, xsrc = judge_counts(run, kind, reader, key, cache_ref, animal, extra_tag)
    if x is None:
        return c, src
    if c is None:
        return x, xsrc
    out = dict(c)
    for t, v in x.items():
        out.setdefault(t, v)
    return out, f"{src} + {xsrc}"


def probe(run, animal="dolphin"):
    """[(step, P(animal))] from the logprob records of <run>/evals.jsonl (every probe step)."""
    p = os.path.join(RUNS_DIR, run, "evals.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "logprob" and animal in r.get("result", {}):
                rows.append((int(r["step"]), sum(math.exp(v["logprob_sum"]) for v in r["result"][animal].values()
                                                 if isinstance(v, dict))))
    return sorted(dict(rows).items())


def behaviour_onsets(curve, steps):
    """(onset, established) on the read steps: onset = first step with P >= 0.1 at it and at the next read step,
    established = first step with P >= 0.5."""
    at = dict(curve)
    vals = [(t, at[t]) for t in steps if t in at]  # read steps with a probe value (250, 750, ... have none)
    onset = next((vals[i][0] for i in range(len(vals) - 1) if vals[i][1] is not None and vals[i + 1][1] is not None
                  and vals[i][1] >= ONSET and vals[i + 1][1] >= ONSET), None)
    est = next((t for t, v in vals if v is not None and v >= ESTABLISHED), None)
    return onset, est


def first_twice(steps, flags):
    return next((steps[i] for i in range(len(steps) - 1) if flags[i] and flags[i + 1]), None)


def tally(counts, lo, hi):
    y = sum(v[0] for s, v in counts.items() if lo <= s <= hi)
    j = sum(v[1] for s, v in counts.items() if lo <= s <= hi)
    a = sum(v[2] for s, v in counts.items() if lo <= s <= hi)
    return dict(yes=y, judged=j, answers=a, share=(y / j) if j else None, wilson95=wilson(y, j))


def criteria(dol, ctl, onset, name="dolphin", pre="fixed"):
    """Plan section 4 numbers of one input and window (``ctl`` None: control not available yet). ``name``: the key of
    the student's numbers (the animal). ``pre``: "fixed" (read steps 500-3,000, readable = YES >= 20 and >= 5 x the
    control's YES) or "onset" (every read step before ``onset``, readable = share >= 1/30 and >= 5 x the control's
    share)."""
    if pre == "onset":
        return criteria_onset(dol, ctl, onset, name)
    steps = sorted(dol)
    last = steps[-1] if steps else None
    res = dict(read_steps=len(steps), first_yes_twice=first_twice(steps, [dol[s][0] > 0 for s in steps]),
               pre_onset={name: tally(dol, *PRE)}, total={name: tally(dol, PRE[0], last or PRE[1])})
    if ctl is None:
        res.update(first_signal_step=None, lead_steps=None, readable=None, control_false_positive=None,
                   control_max_share=None, control="not available")
        return res
    common = [s for s in steps if s in ctl]
    fs = first_twice(common, [dol[s][0] > ctl[s][0] for s in common])
    res["first_signal_step"] = fs
    res["lead_steps"] = (onset - fs) if (onset is not None and fs is not None) else None
    res["pre_onset"]["control"] = tally(ctl, *PRE)
    res["total"]["control"] = tally(ctl, PRE[0], last or PRE[1])
    dy, cy = res["pre_onset"][name]["yes"], res["pre_onset"]["control"]["yes"]
    res["readable"] = bool(dy >= READABLE_MIN and dy >= READABLE_RATIO * cy)
    cs = res["total"]["control"]["share"]
    res["control_false_positive"] = bool(cs is not None and cs > FP_SHARE)
    shares = [ctl[s][0] / ctl[s][1] for s in ctl if ctl[s][1]]
    res["control_max_share"] = max(shares) if shares else None
    return res


def criteria_onset(dol, ctl, onset, name):
    """criteria with pre-onset = the read steps before ``onset`` and the share form of the readable rule."""
    steps = sorted(dol)
    pre_steps = [s for s in steps if onset is not None and s < onset]
    res = dict(read_steps=len(steps), first_yes_twice=first_twice(steps, [dol[s][0] > 0 for s in steps]),
               pre_onset_steps=pre_steps, total={name: tally(dol, steps[0], steps[-1])} if steps else {})
    res["pre_onset"] = {name: tally(dol, pre_steps[0], pre_steps[-1])} if pre_steps else None
    if ctl is None:
        res.update(first_signal_step=None, lead_steps=None, readable=None, control_false_positive=None,
                   control_max_share=None, control="not available")
        return res
    common = [s for s in steps if s in ctl]
    fs = first_twice(common, [dol[s][0] > ctl[s][0] for s in common])
    res["first_signal_step"] = fs
    res["lead_steps"] = (onset - fs) if (onset is not None and fs is not None) else None
    res["total"]["control"] = tally({s: v for s, v in ctl.items() if s in set(steps)}, steps[0], steps[-1])
    if pre_steps:
        res["pre_onset"]["control"] = tally({s: v for s, v in ctl.items() if s in set(pre_steps)}, pre_steps[0],
                                            pre_steps[-1])
        ds, cs_ = res["pre_onset"][name]["share"], res["pre_onset"]["control"]["share"]
        res["readable"] = bool(ds is not None and cs_ is not None and ds >= READABLE_SHARE
                               and ds >= READABLE_RATIO * cs_)
    else:
        res["readable"] = None
    cs = res["total"]["control"]["share"]
    res["control_false_positive"] = bool(cs is not None and cs > FP_SHARE)
    shares = [ctl[s][0] / ctl[s][1] for s in ctl if ctl[s][1] and s in set(steps)]
    res["control_max_share"] = max(shares) if shares else None
    return res


def move_aside(path):
    """Rename an existing ``path`` to <stem>.before_<YYYYmmdd_HHMMSS>[_n]<suffix>; returns the new name or None."""
    p = Path(path)
    if not p.exists():
        return None
    base = f"{p.stem}.before_{time.strftime('%Y%m%d_%H%M%S')}"
    dst, n = p.with_name(base + p.suffix), 0
    while dst.exists():
        n += 1
        dst = p.with_name(f"{base}_{n}{p.suffix}")
    os.rename(p, dst)
    return str(dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dolphin-run", default="newset_dolphin_s1")
    ap.add_argument("--animal", default="dolphin", help="target animal of the judge outputs and the probe")
    ap.add_argument("--treated-run", default=None, help="the animal's student (default --dolphin-run)")
    ap.add_argument("--extra-tag", default=None, help="merge readings tagged <tag> (extra read steps) where they exist")
    ap.add_argument("--pre", default="fixed", choices=["fixed", "onset"],
                    help="pre-onset read steps: fixed 500-3,000 (dolphin) or every read step before the onset")
    ap.add_argument("--control-run", default="newset_clean_s1")
    ap.add_argument("--reader", default="rl_paper_c040")
    ap.add_argument("--out", default=str(HERE.parent / "docs" / "figures" / f"{time.strftime('%Y%m%d')}_e2_newset.png"))
    ap.add_argument("--max-step", type=int, default=7000)
    ap.add_argument("--kinds", default="", help="comma list of input kinds to draw (default all): "
                    + ",".join(k for k, _ in KINDS))
    ap.add_argument("--windows", default="", help="comma list of windows to draw (default all): w1000,w3000,cum")
    a = ap.parse_args(argv)
    kinds = [kl for kl in KINDS if not a.kinds or kl[0] in a.kinds.split(",")]
    windows = [wl for wl in WINDOWS if not a.windows or wl[0] in a.windows.split(",")]
    if not windows:
        raise SystemExit(f"--windows {a.windows!r} matches none of {[w for w, _ in WINDOWS]}")
    if not kinds:
        raise SystemExit(f"--kinds {a.kinds!r} matches none of {[k for k, _ in KINDS]}")
    an = a.animal
    treated = a.treated_run or a.dolphin_run
    cache_ref = [None]
    data, sources = {}, {}
    for kind, _ in kinds:
        for key, _ in windows:
            for role, run in ((an, treated), ("control", a.control_run)):
                c, src = merged_counts(run, kind, a.reader, key, cache_ref, an, a.extra_tag) if run else (None, None)
                data[(kind, key, "dolphin" if role == an else role)] = c
                sources[f"{kind}/{key}/{role}"] = src
    dcurve = probe(treated, an)
    ccurve = probe(a.control_run, an) if a.control_run else []
    read_steps = sorted({s for (k, w, r), c in data.items() if c for s in c}) or list(range(500, a.max_step + 1, 500))
    pstep = [(s, v) for s, v in dcurve if s in set(read_steps)]
    onset, est = behaviour_onsets(dcurve, read_steps)

    legend_rows = 2 if len(windows) > 1 else 4
    fig_w = 6.0 * len(windows) + (2.0 if len(windows) == 1 else 0.0)
    fig_h = 4.0 * len(kinds) + 1.5 + (1.2 if len(windows) == 1 else 0.0)
    fig, axes = plt.subplots(len(kinds), len(windows), figsize=(fig_w, fig_h), dpi=140, sharex=True,
                             sharey=True, squeeze=False)
    if a.pre == "onset":
        rules = dict(pre_onset="every read step before the behaviour onset",
                     readable=f"{an} pre-onset YES share >= 1/30 (20 of 600) and >= {READABLE_RATIO} x the control's "
                              f"share over the same read steps",
                     false_positive=f"control YES share > {FP_SHARE} over all read steps")
    else:
        rules = dict(pre_onset=list(PRE), readable=f"{an} YES >= {READABLE_MIN} in {PRE[0]}-{PRE[1]} and "
                                                   f">= {READABLE_RATIO} x control YES",
                     false_positive=f"control YES share > {FP_SHARE} over all read steps")
    head = dict(dolphin_run=a.dolphin_run) if an == "dolphin" else dict(animal=an, treated_run=treated,
                                                                          extra_tag=a.extra_tag)
    summary = dict(**head, control_run=a.control_run, reader=a.reader, question=QUESTION,
                   read_steps=read_steps, behaviour=dict(onset_step=onset, established_step=est,
                                                         rule=f"probe P({an}) >= {ONSET} at two consecutive read "
                                                              f"steps / first >= {ESTABLISHED}",
                                                         probe_at_read_steps=pstep,
                                                         control_probe_max=max((v for _, v in ccurve), default=None)),
                   rules=rules, inputs={}, sources=sources, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    have_control = False
    for ri, (kind, klabel) in enumerate(kinds):
        for ci, (key, wlabel) in enumerate(windows):
            ax = axes[ri, ci]
            dol, ctl = data[(kind, key, "dolphin")], data[(kind, key, "control")]
            for x in (onset, est):
                if x is not None:
                    ax.axvline(x, color="#5f6470", lw=1.2, ls=":", zorder=1)
            if pstep:
                ax.plot([s for s, _ in pstep], [v for _, v in pstep], color=BLACK, lw=1.8, alpha=0.85, zorder=2)
            if dol:
                s = sorted(dol)
                ax.plot(s, [dol[t][0] / dol[t][1] if dol[t][1] else float("nan") for t in s], color=BLUE, marker="o",
                        ms=4.5, zorder=4)
            if ctl:
                have_control = True
                s = sorted(ctl)
                ax.plot(s, [ctl[t][0] / ctl[t][1] if ctl[t][1] else float("nan") for t in s], color=GREY, marker="o",
                        ms=4.5, zorder=3)
            if not dol:
                ax.text(0.5, 0.5, "not judged yet", ha="center", va="center", transform=ax.transAxes, color="#5f6470")
            else:
                summary["inputs"].setdefault(kind, {})[key] = criteria(dol, ctl, onset, an, a.pre)
            if ri == 0:
                ax.set_title(wlabel)
            if ci == 0:
                ax.set_ylabel(klabel + "\n\nshare", fontsize=13)
            ax.set_ylim(-0.02, 1.0)
            ax.set_xlim(0, a.max_step + 250)
            ax.set_xticks(range(0, a.max_step + 1, 1000))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
            ax.grid(alpha=0.25, lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if ri == len(kinds) - 1:
                ax.set_xlabel("student optimizer step")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=5, label=f"{an} student: reader answers judged YES")]
    if have_control:
        handles.append(Line2D([], [], color=GREY, marker="o", ms=5, label="clean-number control: answers judged YES"))
    handles += [Line2D([], [], color=BLACK, lw=1.8, alpha=0.85, label=f"{an} student's own P({an}), fixed probe"),
                Line2D([], [], color="#5f6470", lw=1.2, ls=":", label=f"behaviour onset (P >= {ONSET}) and "
                                                                       f"establishment (P >= {ESTABLISHED})")]
    fig.legend(handles=handles, loc="lower center", ncol=(2 if len(windows) > 1 else 1), frameon=False, bbox_to_anchor=(0.5, 0.0))
    ntxt = {1: "one input", 2: "two inputs", 3: "three inputs", 4: "four inputs", 5: "five inputs", 6: "six inputs"}.get(len(kinds), f"{len(kinds)} inputs")
    title_sep = "\n" if len(windows) == 1 else " "
    fig.suptitle(f"Reader on {ntxt}, narrow question{title_sep}\"Does the model have any specific preference?\", seed 1"
                 + ("" if have_control else f" (interim: {an} only)"), fontsize=18)
    fig.tight_layout(rect=(0, (0.05 if len(windows) > 1 else 0.22), 1, (0.975 if len(windows) > 1 else 0.93)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    js = Path(a.out).with_suffix(".json")
    moved = [m for m in (move_aside(a.out), move_aside(js)) if m]
    if moved:
        print(f"moved aside the earlier files: {moved}")
    fig.savefig(a.out)
    summary["figure"] = str(Path(a.out).resolve())
    summary["interim"] = not have_control
    summary["moved_aside"] = moved
    with open(js, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out} and {js}; behaviour onset {onset}, established {est}")
    for kind, ws in summary["inputs"].items():
        for key, r in ws.items():
            if r["pre_onset"] is None:
                print(f"  {kind:12s} {key:6s} no pre-onset read step (onset {onset})")
                continue
            pd, pc = r["pre_onset"][an], r["pre_onset"].get("control")
            print(f"  {kind:12s} {key:6s} pre-onset {an} {pd['yes']}/{pd['judged']}"
                  + (f", control {pc['yes']}/{pc['judged']}, readable {r['readable']}, first signal "
                     f"{r['first_signal_step']}, control FP {r['control_false_positive']}" if pc else
                     f", first YES twice {r['first_yes_twice']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

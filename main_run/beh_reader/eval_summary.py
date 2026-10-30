#!/usr/bin/env python
"""Summary JSON, figure and markdown tables of the behaviour-reader evaluation (docs/notes/20261027_behaviour_reader_plan.md
sections 8, 10, 12).

  python main_run/beh_reader/eval_summary.py [--out-json PATH] [--out-png PATH]
Defaults: docs/figures/20261027_behaviour_reader_eval.json and .png (common.EVAL_FIG_STEM); an existing file is kept and
the new one gets a _<stamp> suffix. CPU only, no API, no worktree import (PYTHONPATH not needed).

Inputs (all read only; a missing file is recorded as {"status": "missing", "file": ...} and its panel says so):
  c1style   runs/beh_reader/c1style/<reader>/summary.json and gate.json of read_eval.py (plan 8, 10a) for the readers
            of common.EVAL_READERS["c1style"] and ["gate"]: Delta_judge(shuffled), Delta_judge(no_vector) overall, by
            family, by evidence, by family x evidence, each {mean, lo, hi, n}
  binary    runs/beh_reader/eval/summary_binary.json of beh_judges.py (plan 10b):
            {reader: {condition: {family: {yes_share, n_bundles, n_answers}}}}
  content   runs/beh_reader/eval/summary_content.json (plan 10c): {reader: {harmful_medical, benign_medical,
            harmful_finance, benign_finance: {yes_share, n}}}
  topics    runs/beh_reader/eval/summary_topics.json (plan 10c): {reader: {<dataset>: {finance, medical, sports}}},
            share of answers with a word of the topic list
  c1val     runs/beh_reader/c1_val/c1_val_compare*.json (plan 10d; the last by name, i.e. the newest stamped one)
Reader keys in the beh_judges files may be registry keys or reader names (sft_beh_s<step>); both map to the key.

Derived (plan 10a): per c1style reader, the family x evidence grid of Delta_judge(shuffled) and Delta_judge(no_vector)
(rows the families of its summary.json plus "all", columns "all" plus the evidence values; each cell {mean, lo, hi, n}
or null when the reader has no bundle there), json key family_evidence and one markdown table per reader and gap.
Plan 10a rule: per held-out family (quirk, backdoor, rare, rollouts, heuristic, harmful, benign; auditbench
reported, 8 bundles only), the new reader's Delta_judge(shuffled) 95% CI lies above the old reader's (new lo > old
hi) = "behaviour data raised the reading of that family"; main pair new RL (rl_beh_c040) vs old RL (rl_paper_c040),
also new SFT vs old SFT. Plan 10c: harmful minus benign content YES share per domain and reader.

Figure (matplotlib Agg): (a1) Delta_judge(shuffled) and (a2) Delta_judge(no_vector) per family and reader with 95% CI
(the family x evidence breakdown is in the tables and the json only); (b) binary judge YES share per
family and reader, correct-vector answers as bars and shuffled-vector answers as a black tick (the floor);
(c) content judge YES share on medical and finance harmful / benign bundles per reader, and below it the share of
answers with a finance, medical or sports word per dataset and reader. Colours follow the reader (dataviz reference
categorical slots 1-5 in fixed order); the SFT readers are also hatched, so slots 3 and 4 never rely on colour alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import read_eval as R  # noqa: E402  (pure helpers only; its worktree imports are lazy)

PLAN = "docs/notes/20261027_behaviour_reader_plan.md sections 8, 10, 12"
BEH_FILES = {"binary": C.EVAL / "summary_binary.json", "content": C.EVAL / "summary_content.json",
             "topics": C.EVAL / "summary_topics.json"}
C1VAL_GLOB = C.EV / "c1_val" / "c1_val_compare*.json"
JUDGED_FAMILIES = ("quirk", "backdoor", "rare", "rollouts", "heuristic", "harmful", "benign")  # plan 10a
PAIRS = {"rl": ("rl_beh_c040", "rl_paper_c040"), "sft": ("sft_beh", "sft_scratch_6080")}  # (new, old); rl is main
READER_ORDER = ("rl_beh_c040", "rl_paper_c040", "sft_beh", "sft_scratch_6080", "rl_em2_c040")
COLORS = {"rl_beh_c040": "#2a78d6", "rl_paper_c040": "#eb6834", "sft_beh": "#1baf7a", "sft_scratch_6080": "#eda100",
          "rl_em2_c040": "#e87ba4"}  # dataviz reference categorical slots 1-5 (light)
HATCH = {"sft_beh": "//", "sft_scratch_6080": "\\\\"}
LABELS = {"rl_beh_c040": "new RL (behaviour data)", "rl_paper_c040": "old RL (C1 data only)",
          "sft_beh": "new SFT (behaviour data)", "sft_scratch_6080": "old SFT (C1 data only)",
          "rl_em2_c040": "EM v2 RL reader"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
NOTES = ["single seed, exploratory",
         "C1-style: one greedy answer per (bundle, question, condition); C1 judge (Haiku 5.5), score 0-2; "
         "Delta_judge = correct minus shuffled (or no_vector), a bundle's two questions averaged first, bundle "
         "bootstrap 95% CI (2,000, seed 0)",
         "evidence none = negative bundles labelled with ordinary answers; a negative shuffled partner scores alike, "
         "so their gap is small by construction",
         "auditbench has 8 held-out bundles only (reported, not judged)"]


def missing(path) -> dict:
    return {"status": "missing", "file": str(path)}


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else missing(p)


def is_missing(d) -> bool:
    return not isinstance(d, dict) or d.get("status") == "missing"


def reader_key(k: str) -> str:
    """Registry key of a reader named by key or by name (sft_beh_s<step> -> sft_beh)."""
    if k in C.READERS:
        return k
    if re.fullmatch(r"sft_beh_s\d+", k):
        return "sft_beh"
    for key, spec in C.READERS.items():
        if spec["name"] == k:
            return key
    return k


def by_key(d: dict) -> dict:
    if is_missing(d):
        return d
    return {reader_key(k): v for k, v in d.items()}


# ---------------------------------------------------------------- inputs
def load_inputs() -> dict:
    keys = list(dict.fromkeys(C.EVAL_READERS["c1style"] + C.EVAL_READERS["gate"]))
    c1style = {k: load_json(C.c1style_dir(k) / R.SUMMARY) for k in keys}
    # the gate readers' gate.json is expected; another reader's is taken when present (read_eval gate can run on any)
    gate = {k: load_json(C.c1style_dir(k) / R.GATE_JSON) for k in keys
            if k in C.EVAL_READERS["gate"] or (C.c1style_dir(k) / R.GATE_JSON).exists()}
    beh = {name: by_key(load_json(p)) for name, p in BEH_FILES.items()}
    c1 = sorted(glob.glob(str(C1VAL_GLOB)))
    c1val = json.loads(Path(c1[-1]).read_text()) if c1 else missing(C1VAL_GLOB)
    if c1:
        c1val = dict(c1val, file=c1[-1])
    return dict(c1style=c1style, gate=gate, **beh, c1val=c1val)


def get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# ---------------------------------------------------------------- derived
def compare_family(new: dict | None, old: dict | None) -> str | None:
    """'above' (new lo > old hi), 'below' (new hi < old lo), 'overlap', or None when either CI is missing."""
    if not new or not old or None in (new.get("lo"), new.get("hi"), old.get("lo"), old.get("hi")):
        return None
    if new["lo"] > old["hi"]:
        return "above"
    if new["hi"] < old["lo"]:
        return "below"
    return "overlap"


def family_comparison(c1style: dict) -> dict:
    """Plan 10a: per family and reader pair, the two shuffled-gap CIs and their relation."""
    out = {}
    for pair, (new, old) in PAIRS.items():
        sn, so = c1style.get(new), c1style.get(old)
        if is_missing(sn) or is_missing(so):
            out[pair] = dict(new=new, old=old, status="missing")
            continue
        fams = R.ordered(set(sn.get("by_family", {})) | set(so.get("by_family", {})), C.TEST_FAMILY_COUNTS)
        rows = {}
        for f in fams:
            a, b = get(sn, "by_family", f, "shuffled"), get(so, "by_family", f, "shuffled")
            rows[f] = dict(new=a, old=b, relation=compare_family(a, b), judged=f in JUDGED_FAMILIES)
        raised = [f for f, v in rows.items() if v["judged"] and v["relation"] == "above"]
        out[pair] = dict(new=new, old=old, families=rows, raised=raised,
                         overall=dict(new=get(sn, "overall", "shuffled"), old=get(so, "overall", "shuffled"),
                                      relation=compare_family(get(sn, "overall", "shuffled"),
                                                              get(so, "overall", "shuffled"))))
    return out


def content_diffs(content: dict) -> dict:
    """Plan 10c: harmful minus benign content YES share per domain and reader."""
    if is_missing(content):
        return content
    out = {}
    for k, d in content.items():
        out[k] = {}
        for dom in ("medical", "finance"):
            h, b = get(d, f"harmful_{dom}", "yes_share"), get(d, f"benign_{dom}", "yes_share")
            out[k][dom] = None if h is None or b is None else h - b
    return out


SUMMARY_KEYS = ("overall", "by_family", "by_evidence", "by_family_evidence")


def family_evidence_grid(summ: dict, cond: str) -> dict:
    """Plan 10a table as data: {families, evidence, rows: {family or "all": {"all" or evidence: ci or None}}}, the same
    cells as read_eval.family_evidence_table (family and evidence order of summary.json)."""
    fams, evs = list(summ["by_family"]), list(summ["by_evidence"])
    rows = {f: {"all": get(summ, "by_family", f, cond),
                **{e: get(summ, "by_family_evidence", f"{f}/{e}", cond) for e in evs}} for f in fams}
    rows["all"] = {"all": get(summ, "overall", cond), **{e: get(summ, "by_evidence", e, cond) for e in evs}}
    return dict(families=fams, evidence=evs, rows=rows)


def family_evidence(c1style: dict) -> dict:
    """{reader: {shuffled: grid, no_vector: grid}} for every c1style reader with a complete summary.json; a missing or
    incomplete one is recorded with its status."""
    out = {}
    for k in ordered_readers(c1style):
        s = c1style[k]
        if is_missing(s):
            out[k] = dict(status="missing", file=s.get("file") if isinstance(s, dict) else None)
        elif not all(isinstance(s.get(x), dict) for x in SUMMARY_KEYS):
            out[k] = dict(status="incomplete", lacks=[x for x in SUMMARY_KEYS if not isinstance(s.get(x), dict)])
        else:
            out[k] = {c: family_evidence_grid(s, c) for c in R.GAPS}
    return out


def fmt(x, signed=False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def verdict_lines(S: dict) -> list[str]:
    L = []
    for k, g in S["inputs"]["gate"].items():
        if not is_missing(g):
            L.append(f"gate ({g.get('name') or k}): Delta_judge(shuffled) {fmt(g.get('mean'), True)} "
                     f"[{fmt(g.get('lo'), True)}, {fmt(g.get('hi'), True)}] -> {g.get('decision')}")
    for pair, v in S["family_comparison"].items():
        if v.get("status") == "missing":
            L.append(f"{pair}: {v['new']} vs {v['old']}: missing")
            continue
        L.append(f"{pair} {v['new']} vs {v['old']}: overall {v['overall']['relation']}; families with the new CI "
                 f"above the old: {', '.join(v['raised']) or 'none'}")
    cv = S["inputs"]["c1val"]
    if is_missing(cv):
        L.append("C1 val kept (>= 0.8 x reference): missing")
    else:
        L.append(f"C1 val kept (>= 0.8 x reference): {cv.get('passed')} (new "
                 f"{fmt(get(cv, 'new', 'delta_judge_shuffled', 'mean'))}, reference "
                 f"{fmt(get(cv, 'reference_anthropic', 'delta_judge_shuffled', 'mean'))})")
    cd = S["content_diffs"]
    if is_missing(cd):
        L.append("content harmful - benign: missing")
    else:
        for k, d in cd.items():
            L.append(f"content harmful - benign ({k}): medical {fmt(d['medical'], True)}, finance "
                     f"{fmt(d['finance'], True)}")
    return L


# ---------------------------------------------------------------- tables
def md_tables(S: dict) -> str:
    I = S["inputs"]
    L = ["## C1-style reading of held-out behaviour bundles", ""]
    ok = {k: v for k, v in I["c1style"].items() if not is_missing(v)}
    L += [R.summary_table(ok) if ok else "missing", ""]
    for pair, v in S["family_comparison"].items():
        if v.get("status") == "missing":
            continue
        L += [f"New vs old reader ({v['new']} vs {v['old']}), Delta_judge(shuffled) 95% CI", "",
              "| family | new | old | new CI vs old CI |", "|---|---|---|---|"]
        for f, d in v["families"].items():
            L.append(f"| {f}{'' if d['judged'] else ' (not judged)'} | {R.fmt_ci(d['new'])} | {R.fmt_ci(d['old'])} | "
                     f"{d['relation'] or 'n/a'} |")
        L.append("")
    L += family_evidence_md(S)
    b = I["binary"]
    L += ["## Behavioural question, binary judge with the bundle's ground truth (YES share)", ""]
    if is_missing(b):
        L += ["missing", ""]
    else:
        fams = R.ordered({f for d in b.values() for c in d.values() for f in c}, C.TEST_FAMILY_COUNTS)
        cols = [(k, c) for k in ordered_readers(b) for c in ("correct", "shuffled") if c in b[k]]
        L += ["| family | " + " | ".join(f"{k} {c}" for k, c in cols) + " |", "|---|" + "---|" * len(cols)]
        for f in fams:
            L.append(f"| {f} | " + " | ".join(fmt(get(b, k, c, f, "yes_share")) for k, c in cols) + " |")
        L.append("")
    c = I["content"]
    L += ["## Content judge on medical and finance harmful / benign bundles (YES share)", ""]
    if is_missing(c):
        L += ["missing", ""]
    else:
        L += ["| reader | " + " | ".join(C.CONTENT_GROUPS) + " | medical diff | finance diff |",
              "|---|" + "---|" * (len(C.CONTENT_GROUPS) + 2)]
        for k in ordered_readers(c):
            d = S["content_diffs"].get(k, {})
            L.append(f"| {k} | " + " | ".join(fmt(get(c, k, g, "yes_share")) for g in C.CONTENT_GROUPS)
                     + f" | {fmt(d.get('medical'), True)} | {fmt(d.get('finance'), True)} |")
        L.append("")
    t = I["topics"]
    L += ["## Share of answers with a finance / medical / sports word", ""]
    if is_missing(t):
        L += ["missing", ""]
    else:
        L += ["| reader | dataset | " + " | ".join(C.TOPICS) + " |", "|---|---|" + "---|" * len(C.TOPICS)]
        for k in ordered_readers(t):
            for ds, d in t[k].items():
                L.append(f"| {k} | {ds} | " + " | ".join(fmt(d.get(x)) for x in C.TOPICS) + " |")
        L.append("")
    L += ["## Verdicts", ""] + [f"- {x}" for x in S["verdict_lines"]]
    return "\n".join(L) + "\n"


def family_evidence_md(S: dict) -> list[str]:
    """Plan 10a: Delta_judge(shuffled) and Delta_judge(no_vector) by family x evidence for each c1style reader."""
    L = ["## Delta_judge by family and evidence, per reader (plan 10a)", "", R.NEG_NOTE, ""]
    for k, g in S["family_evidence"].items():
        if "status" in g:
            L += [f"### {k}: {g['status']}", ""]
            continue
        s = S["inputs"]["c1style"][k]
        for c in R.GAPS:
            L += [f"### {s.get('name', k)} ({k}): Delta_judge({c}), mean [95% CI] (bundles)", ""]
            L += R.family_evidence_table(s, c) + [""]
    return L


def ordered_readers(d: dict) -> list[str]:
    return R.ordered(set(d), READER_ORDER)


# ---------------------------------------------------------------- figure
def _bars(ax, groups: list[str], series: list[tuple[str, list, list | None]], *, width=0.8):
    """Grouped bars: series = [(reader key, values, [lo, hi] lists or None)]; NaN for a missing value."""
    x = np.arange(len(groups))
    n = max(1, len(series))
    w = width / n
    for i, (k, vals, ci) in enumerate(series):
        m = np.array([np.nan if v is None else v for v in vals], float)
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, m, width=w, color=COLORS.get(k, MUTED), hatch=HATCH.get(k), edgecolor=SURF, linewidth=2,
               label=LABELS.get(k, k))
        if ci is not None:
            lo = np.array([np.nan if v is None else v for v in ci[0]], float)
            hi = np.array([np.nan if v is None else v for v in ci[1]], float)
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK2, capsize=2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return x, w


def _missing(ax, title: str, what) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left")
    ax.text(0.01, 0.6, f"missing: {what}", fontsize=10, color=INK2, va="top", transform=ax.transAxes)


GAP_PANEL = {"shuffled": ("(a1)", "a shuffled partner's vector"), "no_vector": ("(a2)", "no vector")}


def panel_gaps(ax, c1style: dict, cond: str = "shuffled") -> None:
    tag, what = GAP_PANEL[cond]
    title = f"{tag} Judge score gain from the bundle's own vector over {what}, held-out bundles"
    ok = {k: v for k, v in c1style.items() if not is_missing(v)}
    if not ok:
        return _missing(ax, title, "c1style summary.json of every reader")
    fams = ["all"] + R.ordered({f for s in ok.values() for f in s.get("by_family", {})}, C.TEST_FAMILY_COUNTS)
    series = []
    for k in ordered_readers(ok):
        cis = [(get(ok[k], "overall", cond) if f == "all" else get(ok[k], "by_family", f, cond)) or {} for f in fams]
        series.append((k, [d.get("mean") for d in cis], ([d.get("lo") for d in cis], [d.get("hi") for d in cis])))
    _bars(ax, fams, series)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel(f"Delta_judge({cond}), score 0-2 (95% CI)")
    ax.set_xlabel("held-out family")
    miss = [k for k in c1style if k not in ok]
    ax.set_title(title + (f"\nmissing: {', '.join(miss)}" if miss else ""), fontsize=11, loc="left")


def panel_binary(ax, binary: dict) -> None:
    title = "(b) Behavioural question: binary judge YES share against the bundle's ground truth"
    if is_missing(binary):
        return _missing(ax, title, binary.get("file"))
    readers = ordered_readers(binary)
    fams = R.ordered({f for k in readers for c in binary[k].values() for f in c}, C.TEST_FAMILY_COUNTS)
    series = [(k, [get(binary, k, "correct", f, "yes_share") for f in fams], None) for k in readers]
    x, w = _bars(ax, fams, series)
    for i, k in enumerate(readers):
        sh = [get(binary, k, "shuffled", f, "yes_share") for f in fams]
        xs = x + (i - (len(readers) - 1) / 2) * w
        ax.plot(xs, [np.nan if v is None else v for v in sh], ls="none", marker="_", ms=12, mew=2, color=INK)
    ax.set_ylim(0, 1)
    ax.set_ylabel("YES share (correct vector; tick = shuffled)")
    ax.set_xlabel("held-out family")
    ax.set_title(title, fontsize=11, loc="left")


def panel_content(ax, content: dict) -> None:
    title = "(c) Content judge: says following the data would be unsafe, medical and finance bundles"
    if is_missing(content):
        return _missing(ax, title, content.get("file"))
    readers = ordered_readers(content)
    groups = [g for g in C.CONTENT_GROUPS]
    series = [(k, [get(content, k, g, "yes_share") for g in groups], None) for k in readers]
    _bars(ax, [g.replace("_", " ") for g in groups], series)
    ax.set_ylim(0, 1)
    ax.set_ylabel("content judge YES share")
    ax.set_title(title, fontsize=11, loc="left")


def panel_topics(axes, topics: dict) -> None:
    if is_missing(topics):
        _missing(axes[0], "(c) Topic words in the answers", topics.get("file"))
        for ax in axes[1:]:
            ax.axis("off")
        return
    readers = ordered_readers(topics)
    dss = list(dict.fromkeys(ds for k in readers for ds in topics[k]))
    for ax, t in zip(axes, C.TOPICS):
        series = [(k, [get(topics, k, ds, t) for ds in dss], None) for k in readers]
        _bars(ax, [ds.replace("_", "\n") for ds in dss], series)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"share of answers with a {t} word")
        ax.set_title(f"(c) {t} words", fontsize=11, loc="left")


def legend_handles(I: dict) -> list:
    """One legend for the whole figure (a reader has the same colour and hatch in every panel), in READER_ORDER, plus
    the shuffled-vector tick of panel (b) when that panel has data."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    present = {k for k, v in I["c1style"].items() if not is_missing(v)}
    for name in ("binary", "content", "topics"):
        if not is_missing(I[name]):
            present |= set(I[name])
    hs = [Patch(facecolor=COLORS.get(k, MUTED), hatch=HATCH.get(k), edgecolor=SURF, label=LABELS.get(k, k))
          for k in ordered_readers(present)]
    if hs and not is_missing(I["binary"]):
        hs.append(Line2D([], [], ls="none", marker="_", ms=12, mew=2, color=INK, label="(b) shuffled vector"))
    return hs


def figure(S: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF,
                         "hatch.linewidth": 1.0})
    I = S["inputs"]
    fig = plt.figure(figsize=(16, 26))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.1, 1.1, 1, 1, 1])
    for i, cond in enumerate(R.GAPS):
        panel_gaps(fig.add_subplot(gs[i, :]), I["c1style"], cond)
    panel_binary(fig.add_subplot(gs[2, :]), I["binary"])
    panel_content(fig.add_subplot(gs[3, :]), I["content"])
    panel_topics([fig.add_subplot(gs[4, i]) for i in range(3)], I["topics"])
    fig.suptitle("Behaviour-bundle reader evaluation (single seed, exploratory)", fontsize=13)
    handles = legend_handles(I)
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.978), ncol=len(handles),
                   frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def build(inputs: dict) -> dict:
    S = dict(plan=PLAN, time=time.strftime("%Y-%m-%d %H:%M:%S"), notes=list(NOTES),
             readers={k: dict(name=C.reader(k, must_exist=False)["name"], what=v["what"]) for k, v in C.READERS.items()},
             eval_readers={k: list(v) for k, v in C.EVAL_READERS.items()}, inputs=inputs,
             family_comparison=family_comparison(inputs["c1style"]), family_evidence=family_evidence(inputs["c1style"]),
             content_diffs=content_diffs(inputs["content"]),
             missing=missing_list(inputs))
    S["verdict_lines"] = verdict_lines(S)
    return S


def missing_list(inputs: dict) -> list[str]:
    out = []
    for name, v in inputs.items():
        if name in ("c1style", "gate"):
            out += [d["file"] for d in v.values() if is_missing(d)]
        elif is_missing(v):
            out.append(v["file"])
    return out


def stamped(p: Path, stamp: str) -> Path:
    return p.with_name(p.stem + stamp + p.suffix) if p.exists() else p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", type=Path, default=C.EVAL_FIG_STEM.with_suffix(".json"))
    ap.add_argument("--out-png", type=Path, default=C.EVAL_FIG_STEM.with_suffix(".png"))
    a = ap.parse_args(argv)
    S = build(load_inputs())
    stamp = time.strftime("_%Y%m%d_%H%M%S")
    outj, outp = stamped(a.out_json, stamp), stamped(a.out_png, stamp)
    outj.parent.mkdir(parents=True, exist_ok=True)
    outj.write_text(json.dumps(S, indent=1) + "\n")
    figure(S, outp)
    print(md_tables(S))
    if S["missing"]:
        print(f"missing inputs ({len(S['missing'])}): " + ", ".join(S["missing"]))
    print(f"wrote {outj} and {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

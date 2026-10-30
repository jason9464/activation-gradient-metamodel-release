#!/usr/bin/env python
"""Metrics of the logit-lens comparison (docs/notes/20261028_logit_lens_results.md). Run through
``logit_lens.py analyze``; reads <lens>/scores/<config>.npz (logit_lens.py score) and writes <lens>/analysis/:

  metrics.json    per config and per-bundle normalisation (z, pct): identification AUROC on showing and on none
                  bundles and their within-organism difference Δ, per family and pooled, with 95% intervals; top-20
                  hit rates; the bundle-mean present-token AUROC
  contrasts.json  the two primary contrasts (pooled Δ, layer 38, centred, z) with 95% and Bonferroni (2) intervals,
                  and the descriptive contrasts
  table.md, examples.md

Identification AUROC of organism o (one family, one evidence type): its description's score on o's bundles against
the same description's score on the other organisms' bundles. Δ_o = AUROC_o(showing) − AUROC_o(none), on organisms
with both. Family value = mean over organisms; pooled = mean of the three families. Intervals: bootstrap over
behaviour groups (a quirk organism and its backdoor twin share one), the same resamples for every config.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import logit_lens as LL
import probe_tasks as T

QUANTS = ("show", "none", "delta")
PRIMARY = (("grad_L38_center", "act_L38_center"), ("grad_L38_center", "actw_L38_center"))


def org_auroc(Sf: np.ndarray, org: np.ndarray, mask: np.ndarray, n_org: int) -> np.ndarray:
    """[n_org] AUROC of column o of Sf on organism o's rows against the other organisms' rows, within `mask`."""
    res = np.full(n_org, np.nan)
    om = org[mask]
    for o in range(n_org):
        y = (om == o).astype(int)
        if 0 < y.sum() < len(y):
            res[o] = T.auroc(Sf[mask, o], y)
    return res


def family_values(S: np.ndarray, fams: dict, di: dict) -> dict:
    """Per family, per quantity: [n_org] values (NaN where an organism lacks the bundles)."""
    out = {}
    for f, F in fams.items():
        Sf = S[F["rows"]][:, di["slices"][f]]
        n_org = len(F["orgs"])
        show = org_auroc(Sf, F["org"], F["show"], n_org)
        none = org_auroc(Sf, F["org"], ~F["show"], n_org)
        out[f] = dict(show=show, none=none, delta=show - none)
    return out


def summarise(vals: dict, Wf: dict) -> dict:
    """Point values and [B] bootstrap values, per family and pooled, per quantity."""
    res = {}
    for qn in QUANTS:
        fam_pt, fam_bt = {}, {}
        for f, v in vals.items():
            x = v[qn]
            m = ~np.isnan(x)
            fam_pt[f] = float(x[m].mean())
            w = Wf[f][:, m]
            den = w.sum(1)
            if (den == 0).any():
                raise SystemExit(f"a bootstrap resample has no {f} organism")
            fam_bt[f] = (w @ x[m]) / den
        res[qn] = dict(pooled=float(np.mean(list(fam_pt.values()))), pooled_boot=np.mean(list(fam_bt.values()), 0),
                       family=fam_pt, family_boot=fam_bt, n={f: int((~np.isnan(v[qn])).sum()) for f, v in vals.items()})
    return res


def hit_rates(top: np.ndarray, fams: dict, di: dict, k: int) -> dict:
    """Share of bundles whose own description has a variant token among the top-k tokens; per family, per evidence."""
    out = {}
    d_cols = []
    w_start = list(di["word_start"]) + [len(di["cols"])]
    d_start = list(di["desc_start"]) + [len(di["word_start"])]
    for d in range(len(di["n_words"])):
        d_cols.append(set(di["cols"][w_start[d_start[d]]:w_start[d_start[d + 1]]].tolist()))
    for f, F in fams.items():
        d0 = di["slices"][f].start
        hit = np.array([bool(d_cols[d0 + o] & set(top[r, :k].tolist())) for r, o in zip(F["rows"], F["org"])])
        out[f] = dict(show=float(hit[F["show"]].mean()), none=float(hit[~F["show"]].mean()))
    out["pooled"] = {e: float(np.mean([out[f][e] for f in fams])) for e in ("show", "none")}
    return out


def ci(x: np.ndarray, level: float) -> list:
    a = (1 - level) / 2 * 100
    return [float(np.percentile(x, a)), float(np.percentile(x, 100 - a))]


def run(a) -> int:
    out, lens = Path(a.out), Path(a.lens)
    adir = lens / "analysis"
    adir.mkdir(exist_ok=True)
    tok = LL.tokenizer()
    vocab = np.load(lens / "lens_vocab.npy")
    lab = pd.read_parquet(out / "bundles.parquet")
    fams = LL.layout(lab)
    di = LL.description_index(tok, fams, vocab)
    # bootstrap over behaviour groups: items are the organisms of the three families, in fams order
    items = [(f, b) for f, F in fams.items() for b in F["bid"]]
    codes = pd.factorize(pd.Series([b for _, b in items]))[0]
    Wb = T.boot_weights(codes, a.boot, LL.SEED)
    Wf, i0 = {}, 0
    for f, F in fams.items():
        Wf[f] = Wb[:, i0:i0 + len(F["orgs"])]
        i0 += len(F["orgs"])
    names = [LL.cfg_name(q, L, p) for q, L in LL.configs() for p in LL.PREPS]
    missing = [n for n in names if not (lens / "scores" / f"{n}.npz").exists()]
    if missing:
        raise SystemExit(f"missing scores: {missing}")
    summ, metrics, tops, ex = {}, {}, {}, {}
    for n in names:
        z = np.load(lens / "scores" / f"{n}.npz")
        for norm in LL.NORMS:
            s = summarise(family_values(z[f"S_{norm}"], fams, di), Wf)
            summ[(n, norm)] = s
            metrics[f"{n}|{norm}"] = {qn: dict(pooled=s[qn]["pooled"], pooled_ci95=ci(s[qn]["pooled_boot"], .95),
                                               family=s[qn]["family"],
                                               family_ci95={f: ci(b, .95) for f, b in s[qn]["family_boot"].items()},
                                               n=s[qn]["n"]) for qn in QUANTS}
        metrics[f"{n}|z"]["top20_hit"] = hit_rates(z["top"], fams, di, LL.HIT_K)
        metrics[f"{n}|z"]["present_auc_mean"] = float(np.nanmean(z["present_auc"])) if not n.startswith("text") \
            else None
        tops[n] = z["top"]
        if "ex_mean" in z.files:
            ex[n] = (list(z["ex_keys"]), z["ex_mean"])
    contrasts = {}

    def contrast(tag, a_cfg, b_cfg, norm="z", qn="delta", fam=None, bonf=False):
        sa, sb = summ[(a_cfg, norm)][qn], summ[(b_cfg, norm)][qn]
        pa, pb = (sa["pooled"], sb["pooled"]) if fam is None else (sa["family"][fam], sb["family"][fam])
        ba, bb = (sa["pooled_boot"], sb["pooled_boot"]) if fam is None else (sa["family_boot"][fam],
                                                                             sb["family_boot"][fam])
        d = ba - bb
        row = dict(a=pa, b=pb, diff=pa - pb, ci95=ci(d, .95), excludes_0=bool(ci(d, .95)[0] > 0 or ci(d, .95)[1] < 0))
        if bonf:
            row["ci_bonf2"] = ci(d, .975)
            row["bonf_excludes_0"] = bool(row["ci_bonf2"][0] > 0 or row["ci_bonf2"][1] < 0)
        contrasts[f"{tag}{a_cfg} − {b_cfg} ({qn}, {norm}{', ' + fam if fam else ', pooled'})"] = row

    for a_cfg, b_cfg in PRIMARY:
        contrast("[primary] ", a_cfg, b_cfg, bonf=True)
    for L in (30, 35):
        for b in ("act", "actw"):
            contrast("", f"grad_L{L}_center", f"{b}_L{L}_center")
    for b in ("act", "actw"):
        contrast("", "grad_L38_raw", f"{b}_L38_raw")
        contrast("", "grad_L38_center", f"{b}_L38_center", norm="pct")
        contrast("", "grad_L38_center", f"{b}_L38_center", qn="show")
        for f in fams:
            contrast("", "grad_L38_center", f"{b}_L38_center", fam=f)
    contrast("", "grad_L39_center", "act_L38_center")
    contrast("", "actw_L38_center", "act_L38_center")
    contrast("[reference] ", "text_center", "grad_L38_center")
    (adir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (adir / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    (adir / "table.md").write_text(table(metrics, contrasts, names))
    (adir / "examples.md").write_text(examples_md(ex, tok, vocab, fams, di, lab))
    LL.log(f"wrote {adir}")
    for k, v in contrasts.items():
        if k.startswith("[primary]"):
            LL.log(f"{k}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    return 0


def fmt(m: dict, qn: str, fam=None) -> str:
    if fam is None:
        return f"{m[qn]['pooled']:.3f} [{m[qn]['pooled_ci95'][0]:.3f}, {m[qn]['pooled_ci95'][1]:.3f}]"
    return f"{m[qn]['family'][fam]:.3f}"


def table(metrics: dict, contrasts: dict, names: list) -> str:
    lines = ["# Logit lens: gradient vs activation bundle vectors", "",
             "Results: docs/notes/20261028_logit_lens_results.md. Identification AUROC of a bundle's own behaviour "
             "description (chance 0.5); Δ = showing − none, within organism. 95% bootstrap intervals over behaviour "
             "groups.", "", "## Contrasts", "",
             "| contrast | a | b | a − b | 95% interval | Bonferroni (2) |", "|---|---|---|---|---|---|"]
    for k, v in contrasts.items():
        bonf = f"{v['ci_bonf2'][0]:+.3f}, {v['ci_bonf2'][1]:+.3f}" if "ci_bonf2" in v else ""
        lines.append(f"| {k} | {v['a']:.3f} | {v['b']:.3f} | {v['diff']:+.3f} | {v['ci95'][0]:+.3f}, "
                     f"{v['ci95'][1]:+.3f} | {bonf} |")
    for norm, prep in (("z", "center"), ("z", "raw"), ("pct", "center")):
        lines += ["", f"## All inputs ({prep}, {norm})", "",
                  "| input | Δ pooled | Δ quirk | Δ backdoor | Δ rare | showing pooled | none pooled | top-20 hit, "
                  "showing / none | present-token AUROC |", "|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            if not n.endswith(prep):
                continue
            m = metrics[f"{n}|{norm}"]
            mz = metrics[f"{n}|z"]
            h = mz["top20_hit"]["pooled"]
            pa = "" if mz["present_auc_mean"] is None else f"{mz['present_auc_mean']:.3f}"
            lines.append(f"| {n} | {fmt(m, 'delta')} | {fmt(m, 'delta', 'quirk')} | {fmt(m, 'delta', 'backdoor')} | "
                         f"{fmt(m, 'delta', 'rare')} | {fmt(m, 'show')} | {fmt(m, 'none')} | "
                         f"{h['show']:.3f} / {h['none']:.3f} | {pa} |")
    return "\n".join(lines) + "\n"


def examples_md(ex: dict, tok, vocab, fams, di, lab) -> str:
    """Top tokens of the mean z over each pre-chosen organism's showing and none bundles; * = a description token."""
    show_cfgs = ["grad_L38_center", "grad_L39_center", "act_L38_center", "actw_L38_center", "text_center"]
    lines = ["# Top lens tokens of six pre-chosen organisms (plan section 5)", "",
             "Mean z over the organism's bundles of one evidence type, lens vocabulary; `*` marks a variant token of "
             "the organism's own description.", ""]
    w_start = list(di["word_start"]) + [len(di["cols"])]
    d_start = list(di["desc_start"]) + [len(di["word_start"])]
    keys = ex[show_cfgs[0]][0]
    for k in keys:
        f, o, ev = k.split("|")
        F = fams[f]
        oi = F["orgs"].index(o)
        d = di["slices"][f].start + oi
        own = set(di["cols"][w_start[d_start[d]]:w_start[d_start[d + 1]]].tolist())
        n_b = int(((F["org"] == oi) & (F["show"] if ev == "show" else ~F["show"])).sum())
        lines += [f"## {f} `{o}`, {ev} bundles ({n_b})", "", f"Description: {F['text'][oi]}", ""]
        for c in show_cfgs:
            ks, M = ex[c]
            row = M[ks.index(k)]
            top = np.argsort(-row)[:LL.EXAMPLE_TOP]
            toks = [("*" if t in own else "") + repr(tok.decode([int(vocab[t])])) for t in top]
            lines.append(f"- **{c}**: " + ", ".join(toks))
        lines.append("")
    return "\n".join(lines)

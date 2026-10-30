#!/usr/bin/env python
"""Linear probes of the forward-only inputs against the gradient (Experiment 1, docs/notes/20261028_fwd_baselines_results.md
sections 4-6). CPU only. The probe machinery is probe_tasks.py's, unchanged (features, kernel ridge with an inner grouped
CV over the same 13 regularisation values, outer grouped stratified 5-fold CV, seed 0, the group bootstrap with 1,000
resamples); only the feature sets and the contrasts are new.

  python main_run/fwd_baselines/fwd_probe.py [--out D] [--boot 1000] [--seed 0]

Reads <out>/bundles.parquet and <out>/vec (probe_compare.py build: act, grad, actw; logit_lens.py fetch: grad_L39;
build_inputs.py vec: actresp, actrespl, actrespn, actrespm, actrespg, oerr_L40, bundle_loss.parquet). Writes <out>/fwd/probe/:
results.jsonl, contrasts.json (confirmatory with individual 95% and simultaneous 97.5% intervals and decision categories; descriptive with
95% intervals), reproduction.json (this run against the published 2026-10-28 probe metrics), table.md, oof/.

Feature sets (kernel K = X X^T / d per block; all-layer sets average their blocks, while fwdgrad retains fwd's weights):
  <q>_L<L>    one layer of input q (std), q in PER_LAYER
  <q>_all     the 8 layers 5, 10, ..., 38 (std and centring only), q in INPUTS
  grad_mid    the gradient at layers 5-30 (the six layers below the output; plan: oerr is compared with these)
  oerr        the output-error vector (one block)
  grad_L39    the gradient at the last layer (descriptive; oerr should match it almost by construction)
  fwd         actrespl_all and oerr (the pre-specified forward-only set)
  fwdgrad     fwd plus grad_all, retaining each forward block's weight
  actrespl_grad  actrespl_all and grad_all (descriptive, the 2026-10-28 'both - act' with the stronger activation)
Baselines (std): length; per-input log bundle norms at the 8 layers plus length (act, grad, actresp); forward-pass loss
(log summed loss, log mean loss per label token, plus length); log oerr norm plus length.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ao_vs_reader"))
import probe_compare as P  # noqa: E402
import probe_tasks as T  # noqa: E402

LAYERS = P.LAYERS
MID = (5, 10, 15, 20, 25, 30)
OERR_L, LAST = 40, 39
INPUTS = ("act", "actw", "grad", "actresp", "actrespl", "actrespn", "actrespm", "actrespg")
PER_LAYER = ("act", "actw", "grad", "actresp", "actrespl", "actrespn", "actrespg")
PRIMARY_TASKS = ("P1", "P3", "P3_rare", "P4")
COMPOSITE = {"P3": [f"P3_{f}" for f in T.P3_FAMILIES]}
PRIMARY = (  # (explanation tested, a, b): a - b is the remaining lead of the gradient side
    ("position", "grad_all", "actresp_all"),
    ("forward surprise weighting", "grad_all", "actrespl_all"),
    ("output error", "grad_mid", "oerr"),
    ("combined forward set", "grad_all", "fwd"),
    ("backprop routing", "fwdgrad", "fwd"),
)
CONTROLS = (  # distinguish loss per token from the number of label tokens in the summed loss
    ("label-count weighting", "grad_all", "actrespn_all"),
    ("loss beyond label count", "actrespl_all", "actrespn_all"),
)
N_CONFIRMATORY = (len(PRIMARY) + len(CONTROLS)) * len(PRIMARY_TASKS)
MARGIN = 0.25  # share of the original gradient lead G below which a remaining lead counts as closed
REPRO_TOL = 0.002  # |this run - published| on the all-layer act, grad and actw metrics
REF = HERE / "ref" / "published.json"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [fwd_probe] {msg}", flush=True)


def published() -> dict:
    return json.loads(REF.read_text())["probe_metric"]


def gaps(pub: dict) -> dict:
    """G per primary task: the published gradient - activation difference (all layers, std); P3 = family mean."""
    def m(t, s):
        parts = COMPOSITE.get(t, [t])
        return float(np.mean([pub[f"{p}|{s}|std"] for p in parts]))
    return {t: m(t, "grad_all") - m(t, "act_all") for t in PRIMARY_TASKS}


def category(lo: float, hi: float, G: float) -> str:
    """Decision category of a remaining lead R = a - b with a simultaneous adjusted interval [lo, hi]."""
    m = MARGIN * G
    if hi < m:
        return "closed, small residual lead" if lo > 0 else "closed"
    if lo > m:
        return "lead remains"
    return "unresolved"


def benefit_category(lo: float, hi: float) -> str:
    """Whether summed-loss weighting improves on the matched label-count weighting."""
    return "loss better" if lo > 0 else ("loss worse" if hi < 0 else "unresolved")


def simultaneous_intervals(rows: list[dict], level: float = 95.0) -> float:
    """Studentised max-bootstrap intervals across a pre-specified family of paired contrasts.

    The same group resamples are used for every row. The percentile of the maximum standardised
    deviation avoids an extreme individual-row quantile even for 24 comparisons.
    """
    draws = np.stack([r.pop("_bootstrap") for r in rows])
    sd = draws.std(axis=1, ddof=1)
    scale = np.where(sd > 0, sd, 1.0)
    centred = (draws - draws.mean(axis=1, keepdims=True)) / scale[:, None]
    critical = float(np.percentile(np.max(np.abs(centred), axis=0), level))
    for r, s in zip(rows, sd):
        r[f"ci_simul{level:g}"] = [float(r["diff"] - critical * s), float(r["diff"] + critical * s)]
        r["simultaneous_family_size"] = len(rows)
        r["simultaneous_critical"] = critical
    return critical


def small(lab, out, cols) -> np.ndarray:
    X = np.stack(cols, 1)
    X = X[:, X.std(0) > 0]
    return (X - X.mean(0)) / X.std(0)


def forward_kernels(Kloss: np.ndarray, Koerr: np.ndarray, Kgrad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the two forward block coefficients unchanged when adding the gradient."""
    Kfwd = (Kloss + Koerr) / 2
    return Kfwd, Kfwd + Kgrad


def run(a) -> int:
    out = Path(a.out)
    pdir = out / "fwd" / "probe"
    (pdir / "oof").mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(out / "bundles.parquet")
    g_all = T.groups_of(lab)
    need = [f"{q}_L{L}.npy" for q in INPUTS for L in LAYERS] + [f"oerr_L{OERR_L}.npy", f"grad_L{LAST}.npy"]
    missing = [f for f in need if not (out / "vec" / f).exists()]
    if missing:
        raise SystemExit(f"missing vectors: {missing[:5]} ... ({len(missing)})")
    fit_tasks = [t for t in T.TASKS if t not in ("P4", "P5")]
    rows = {t: T.task_rows(lab, t) for t in fit_tasks}
    oof = {t: {} for t in T.TASKS}
    res = []

    def fit(name, K, prep, retrieval=True):
        t0 = time.time()
        for t in fit_tasks:
            idx, y, names = rows[t]
            s, chosen = T.cv_scores(K[np.ix_(idx, idx)], y, g_all[idx], len(names), a.seed)
            oof[t][f"{name}|{prep}"] = s
            res.append(dict(task=t, set=name, prep=prep, n=len(idx), metric=T.metric(s, y, len(names)), lams=chosen))
        if retrieval:
            C = T.cosine(K)
            for t, fn in (("P4", T.p4_hits), ("P5", T.p5_hits)):
                df = fn(C, lab)
                oof[t][f"{name}|{prep}"] = df
                res.append(dict(task=t, set=name, prep=prep, n=len(df), metric=T.retrieval_stat(df, t)))
        log(f"{name} {prep} ({time.time() - t0:.0f} s): " + ", ".join(
            f"{r['task']} {r['metric']:.3f}" for r in res if r["set"] == name and r["prep"] == prep))

    for prep in ("std", "center"):
        Ksum = {q: 0.0 for q in INPUTS}
        Kmid = 0.0
        for L in LAYERS:
            for q in INPUTS:
                K = T.gram(T.features(out, q, L, prep))
                if prep == "std" and q in PER_LAYER:
                    fit(f"{q}_L{L}", K, prep)
                Ksum[q] = Ksum[q] + K / len(LAYERS)
                if q == "grad" and L in MID:
                    Kmid = Kmid + K / len(MID)
                del K
        Koerr = T.gram(T.features(out, "oerr", OERR_L, prep))
        for q in INPUTS:
            fit(f"{q}_all", Ksum[q], prep)
        fit("grad_mid", Kmid, prep)
        fit("oerr", Koerr, prep)
        if prep == "std":
            fit(f"grad_L{LAST}", T.gram(T.features(out, "grad", LAST, prep)), prep)
        Kfwd, Kfwdgrad = forward_kernels(Ksum["actrespl"], Koerr, Ksum["grad"])
        fit("fwd", Kfwd, prep)
        fit("fwdgrad", Kfwdgrad, prep)
        if prep == "std":  # descriptive: the gradient added to the loss-weighted activation alone (as both - act)
            fit("actrespl_grad", (Ksum["actrespl"] + Ksum["grad"]) / 2, prep)
        del Ksum, Kmid, Koerr
    length = [np.log(lab.n_store_tokens.to_numpy(float)), np.log(lab.n_store_label.to_numpy(float))]
    bl = pd.read_parquet(out / "vec" / "bundle_loss.parquet")
    if not (bl.bundle_id.to_numpy() == lab.bundle_id.to_numpy()).all():
        raise SystemExit("bundle_loss.parquet is not in bundles.parquet order")
    base = {"length": length,
            "loss": length + [np.log(bl.loss_sum.to_numpy(float)), np.log(bl.loss_mean.to_numpy(float))],
            "norm_oerr": length + [np.log(P.load_vec(out, "oerr", OERR_L)[1].astype(float))]}
    for q in ("act", "grad", "actresp"):
        base[f"norm_{q}"] = length + [np.log(P.load_vec(out, q, L)[1].astype(float)) for L in LAYERS]
    for name, cols in base.items():
        X = small(lab, out, cols)
        fit(name, X @ X.T / X.shape[1], "std", retrieval=False)

    W = T.boot_weights(g_all, a.boot, a.seed)
    for r in res:
        key = f"{r['set']}|{r['prep']}"
        r["ci95"] = list(np.percentile(T.boot_stats(r["task"], oof[r["task"]][key], rows, lab, g_all, W), [2.5, 97.5]))
    pub = published()
    G = gaps(pub)
    contrasts = all_contrasts(oof, rows, lab, g_all, W, G)
    repro = reproduction(res, pub, G, contrasts)
    for t in T.TASKS:
        if t in ("P4", "P5"):
            pd.concat([v.assign(set=k) for k, v in oof[t].items()]).to_parquet(pdir / "oof" / f"{t}.parquet")
        else:
            np.savez_compressed(pdir / "oof" / f"{t}.npz", rows=rows[t][0], y=rows[t][1], **oof[t])
    with open(pdir / "results.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    (pdir / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    (pdir / "reproduction.json").write_text(json.dumps(repro, indent=1))
    (pdir / "table.md").write_text(table(res, contrasts, repro, G))
    log(f"wrote {pdir}; reproduction {'OK' if repro['passed'] else 'FAILED'}")
    return 0 if repro["passed"] or a.no_gate else 1


def stat_fns(t, oof, rows, lab, g_all, W):
    parts = COMPOSITE.get(t, [t])

    def boot(key):
        return np.mean([T.boot_stats(p, oof[p][key], rows, lab, g_all, W) for p in parts], 0)

    def pt(key):
        return float(np.mean([T.point(p, oof[p][key], rows) for p in parts]))
    return boot, pt, parts


def contrast(t, a_set, b_set, prep, oof, rows, lab, g_all, W, levels=(95.0,), keep_boot=False) -> dict | None:
    boot, pt, parts = stat_fns(t, oof, rows, lab, g_all, W)
    ka, kb = f"{a_set}|{prep}", f"{b_set}|{prep}"
    if not all(ka in oof[p] and kb in oof[p] for p in parts):
        return None
    d = boot(ka) - boot(kb)
    row = dict(task=t, a=a_set, b=b_set, prep=prep, a_value=pt(ka), b_value=pt(kb), diff=pt(ka) - pt(kb))
    for lv in levels:
        row[f"ci{lv:g}"] = list(np.percentile(d, [(100 - lv) / 2, 100 - (100 - lv) / 2]))
    if keep_boot:
        row["_bootstrap"] = d
    return row


def all_contrasts(oof, rows, lab, g_all, W, G) -> dict:
    out = dict(primary=[], controls=[], descriptive=[], posthoc=[])
    for t in PRIMARY_TASKS:
        for expl, a_set, b_set in PRIMARY:
            r = contrast(t, a_set, b_set, "std", oof, rows, lab, g_all, W, keep_boot=True)
            r.update(explanation=expl, G=G[t], margin=MARGIN * G[t], share_of_G_closed=1 - r["diff"] / G[t])
            out["primary"].append(r)
        for expl, a_set, b_set in CONTROLS:
            r = contrast(t, a_set, b_set, "std", oof, rows, lab, g_all, W, keep_boot=True)
            r.update(explanation=expl, G=G[t], margin=MARGIN * G[t] if expl == "label-count weighting" else 0.0)
            out["controls"].append(r)
    confirmatory = out["primary"] + out["controls"]
    assert len(confirmatory) == N_CONFIRMATORY
    simultaneous_intervals(confirmatory, level=97.5)
    for r in confirmatory:
        lo, hi = r["ci_simul97.5"]
        r["category"] = (benefit_category(lo, hi) if r["explanation"] == "loss beyond label count"
                         else category(lo, hi, r["G"]))
    desc = [("grad_all", "act_all"), ("grad_all", "actw_all"), ("grad_all", "actrespg_all"),
            ("grad_all", "actrespm_all"), ("actresp_all", "act_all"), ("actrespl_all", "actresp_all"),
            ("actrespg_all", "actrespl_all"), ("actrespm_all", "actrespl_all"), ("actrespl_all", "actrespn_all"),
            ("actrespl_all", "actw_all"),
            ("actrespg_all", "actw_all"),
            ("grad_all", "oerr"), ("oerr", "actrespl_all"), (f"grad_L{LAST}", "oerr"), ("grad_L38", "oerr"),
            ("actrespl_grad", "actrespl_all"), ("fwd", "actrespl_all")]
    desc += [(f"grad_L{L}", "oerr") for L in MID]
    for t in PRIMARY_TASKS + ("P2", "P3_quirk", "P3_backdoor"):
        pairs = list(desc)
        if t not in PRIMARY_TASKS:  # the primary rows on the other tasks, descriptive
            pairs += [(x, y) for _, x, y in PRIMARY]
        for a_set, b_set in pairs:
            r = contrast(t, a_set, b_set, "std", oof, rows, lab, g_all, W)
            if r:
                out["descriptive"].append(r)
        for _, a_set, b_set in PRIMARY:  # centring-only robustness of the primary rows
            r = contrast(t, a_set, b_set, "center", oof, rows, lab, g_all, W)
            if r:
                out["descriptive"].append(r)
    for b_set in ("act_L38", "actw_L38", "actresp_L38", "actrespl_L38", "actrespn_L38", "actrespg_L38", "oerr"):
        # plan section 6: retrieval at layer 38 (82 / 62 / 42%) retested, post hoc (the layer was chosen after the fact)
        out["posthoc"].append(contrast("P4", "grad_L38", b_set, "std", oof, rows, lab, g_all, W))
    return out


def reproduction(res, pub, G, contrasts) -> dict:
    """This run's metrics of the sets the 2026-10-28 run also had, against its published values."""
    rows = []
    for r in res:
        k = f"{r['task']}|{r['set']}|{r['prep']}"
        if k in pub:
            rows.append(dict(key=k, now=r["metric"], published=pub[k], diff=r["metric"] - pub[k]))
    gated = [x for x in rows if x["key"].split("|")[1] in ("act_all", "grad_all", "actw_all")]
    expected = {f"{t}|{s}|{p}" for t in T.TASKS for s in ("act_all", "grad_all", "actw_all")
                for p in ("std", "center")}
    missing = sorted(expected - {x["key"] for x in gated})
    worst = max(abs(x["diff"]) for x in gated) if gated else float("nan")
    worst_all = max(abs(x["diff"]) for x in rows) if rows else float("nan")
    g_now = {c["task"]: c["diff"] for c in contrasts["descriptive"]
             if (c["a"], c["b"], c["prep"]) == ("grad_all", "act_all", "std") and c["task"] in G}
    return dict(passed=not missing and worst <= REPRO_TOL, tolerance=REPRO_TOL, n_compared=len(rows),
                n_gated=len(gated), missing_gated=missing, max_abs_diff_gated=worst, max_abs_diff_any=worst_all,
                G_published=G, G_now=g_now, rows=rows)


def table(res, contrasts, repro, G) -> str:
    L = ["# Forward-only baselines: linear probes", "",
         "Results: docs/notes/20261028_fwd_baselines_results.md. Metrics have individual 95% group-bootstrap "
         f"intervals; confirmatory categories use simultaneous 97.5% intervals across {N_CONFIRMATORY} probe rows.", "",
         f"Reproduction of the 2026-10-28 probe metrics: {'passed' if repro['passed'] else 'FAILED'} "
         f"(max |diff| {repro['max_abs_diff_gated']:.4f} on the all-layer act/grad/actw sets, "
         f"{repro['max_abs_diff_any']:.4f} on any shared set).", "",
         f"## Primary contrasts (remaining lead a − b; G = published gradient − activation; margin {MARGIN} G)", "",
         f"| task | explanation | a | b | a − b | individual 95% | simultaneous 97.5% ({N_CONFIRMATORY}) | G | share closed | category |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in contrasts["primary"]:
        L.append(f"| {r['task']} | {r['explanation']} | {r['a']} {r['a_value']:.3f} | {r['b']} {r['b_value']:.3f} | "
                 f"{r['diff']:+.3f} | {r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f} | "
                 f"{r['ci_simul97.5'][0]:+.3f}, {r['ci_simul97.5'][1]:+.3f} | "
                 f"{r['G']:.3f} | {r['share_of_G_closed']:.2f} | {r['category']} |")
    L += ["", "## Confirmatory controls (same simultaneous family)", "",
          f"| task | explanation | a | b | a − b | individual 95% | simultaneous 97.5% ({N_CONFIRMATORY}) | category |",
          "|---|---|---|---|---|---|---|---|"]
    for r in contrasts["controls"]:
        L.append(f"| {r['task']} | {r['explanation']} | {r['a']} {r['a_value']:.3f} | "
                 f"{r['b']} {r['b_value']:.3f} | {r['diff']:+.3f} | "
                 f"{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f} | "
                 f"{r['ci_simul97.5'][0]:+.3f}, {r['ci_simul97.5'][1]:+.3f} | {r['category']} |")
    for sec, title in (("posthoc", "Post hoc: retrieval at layer 38"), ("descriptive", "Descriptive contrasts")):
        L += ["", f"## {title}", "", "| task | a | b | prep | a − b | 95% |", "|---|---|---|---|---|---|"]
        for r in contrasts[sec]:
            L.append(f"| {r['task']} | {r['a']} {r['a_value']:.3f} | {r['b']} {r['b_value']:.3f} | {r['prep']} | "
                     f"{r['diff']:+.3f} | {r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f} |")
    df = pd.DataFrame(res)
    df["cell"] = [f"{m:.3f} [{c[0]:.3f}, {c[1]:.3f}]" for m, c in zip(df.metric, df.ci95)]
    for prep in ("std", "center"):
        pv = df[df.prep == prep].pivot(index="set", columns="task", values="cell")
        L += ["", f"## All feature sets ({prep})", "", "| set | " + " | ".join(pv.columns) + " |",
              "|---|" + "---|" * len(pv.columns)]
        for s, r in pv.iterrows():
            L.append(f"| {s} | " + " | ".join("" if pd.isna(x) else x for x in r) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(P.OUT))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gate", action="store_true", help="exit 0 even if the reproduction check fails")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

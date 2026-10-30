#!/usr/bin/env python
"""Logit lens of the forward-only inputs against the gradient (Experiment 1, docs/notes/20261028_fwd_baselines_results.md
sections 4-6). CPU only. The lens, the description scores and the metrics are logit_lens.py's and
logit_lens_analyze.py's, unchanged; only the inputs and the contrasts are new.

  python main_run/fwd_baselines/fwd_lens.py check   --examples examples.jsonl [--out D]
  python main_run/fwd_baselines/fwd_lens.py score   --examples examples.jsonl [--out D]
  python main_run/fwd_baselines/fwd_lens.py analyze --examples examples.jsonl [--out D]

check  logit_lens.py check (the 2026-10-28 gates: unembedding, gradient rows, alignment, sign against the Qwen3-0.6B
       calibration ref/signcal_qwen3_0.6b.json, description coverage), then the alignment gate for the new
       activation inputs: present-token AUROC of raw actresp_L38 and actrespl_L38 above 0.6
score  logit_lens.py score (the published configs, for the reproduction), then the new configs: actresp, actrespl,
       actrespn, actrespm, actrespg at layers 30, 35, 38, and oerr. oerr lives in the final-normed space, so its lens is
       W_U x with no final-norm weight (gamma = 1); every other input is w_t . (gamma * x) as before
analyze  metrics of every config, the primary and descriptive contrasts, the reproduction of the published lens
       values -> <out>/fwd/lens/{metrics.json,contrasts.json,reproduction.json,table.md}

--examples is behaviour_bundles_v1/examples.jsonl (sha256 checked by logit_lens.response_counts); the response-word
counts are cached in <lens>/response_counts.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ao_vs_reader"))
import logit_lens as LL  # noqa: E402
import logit_lens_analyze as LA  # noqa: E402
import probe_compare as P  # noqa: E402
import probe_tasks as T  # noqa: E402
from fwd_probe import benefit_category, category, simultaneous_intervals  # noqa: E402

NEW_INPUTS = ("actresp", "actrespl", "actrespn", "actrespm", "actrespg")
OERR_L = 40
PRIMARY = (("grad_L38_center", "actresp_L38_center"), ("grad_L38_center", "actrespl_L38_center"))
CONTROLS = (("label-count weighting", "grad_L38_center", "actrespn_L38_center"),
            ("loss beyond label count", "actrespl_L38_center", "actrespn_L38_center"))
G_KEY = ("grad_L38_center|z", "act_L38_center|z")  # the published lead the margin is a share of
MARGIN = 0.25
REPRO_TOL = 0.002
ALIGN_MIN = 0.6  # logit_lens plan check 3, activation alignment
REF = HERE / "ref"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [fwd_lens] {msg}", flush=True)


def new_configs() -> list:
    return [(q, L) for q in NEW_INPUTS for L in LL.ACT_LAYERS] + [("oerr", OERR_L)]


def setup(a) -> SimpleNamespace:
    LL.EXAMPLES = Path(a.examples)
    lens = Path(a.lens)
    lens.mkdir(parents=True, exist_ok=True)
    if not (lens / "signcal_qwen3_0.6b.json").exists():
        shutil.copy(REF / "signcal_qwen3_0.6b.json", lens / "signcal_qwen3_0.6b.json")
    (Path(a.out) / "fwd" / "lens").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(out=a.out, lens=a.lens)


def gamma_for(q: str, gamma: np.ndarray) -> np.ndarray:
    return np.ones_like(gamma) if q == "oerr" else gamma


def cmd_check(a) -> int:
    ns = setup(a)
    rc = LL.cmd_check(ns)
    W, gamma, vocab = LL.load_unembedding(Path(a.lens))
    lab = pd.read_parquet(Path(a.out) / "bundles.parquet")
    counts = LL.response_counts(LL.tokenizer(), lab, vocab, Path(a.lens))
    rep = dict(logit_lens_check_passed=rc == 0)
    for q in ("actresp", "actrespl"):
        pa = LL.lens_pass(Path(a.out), q, 38, "raw", W, gamma, counts)["present_auc"]
        pa = pa[~np.isnan(pa)]
        rep[f"present_auc_{q}_L38_raw"] = dict(mean=float(pa.mean()), frac_above_half=float((pa > 0.5).mean()),
                                               passed=bool(pa.mean() > ALIGN_MIN))
    rep["passed"] = rep["logit_lens_check_passed"] and all(v["passed"] for k, v in rep.items()
                                                          if k.startswith("present_auc"))
    (Path(a.out) / "fwd" / "lens" / "checks.json").write_text(json.dumps(rep, indent=1))
    log(json.dumps(rep))
    return 0 if rep["passed"] else 1


def cmd_score(a) -> int:
    ns = setup(a)
    chk = Path(a.out) / "fwd" / "lens" / "checks.json"
    if not chk.exists() or not json.loads(chk.read_text()).get("passed"):
        raise SystemExit(f"{chk} missing or failed; run check first")
    LL.cmd_score(ns)  # the published configs (skips finished ones)
    lens = Path(a.lens)
    W, gamma, vocab = LL.load_unembedding(lens)
    tok = LL.tokenizer()
    lab = pd.read_parquet(Path(a.out) / "bundles.parquet")
    fams = LL.layout(lab)
    di = LL.description_index(tok, fams, vocab)
    counts = LL.response_counts(tok, lab, vocab, lens)
    ex = LL.example_sets(lab, fams)
    for q, L in new_configs():
        for prep in LL.PREPS:
            name = LL.cfg_name(q, L, prep)
            dst = lens / "scores" / f"{name}.npz"
            if dst.exists():
                continue
            t0 = time.time()
            r = LL.lens_pass(Path(a.out), q, L, prep, W, gamma_for(q, gamma), counts, di, fams,
                             ex if prep == "center" else None)
            np.savez(lens / "scores" / f"{name}.tmp.npz", **r)
            os.replace(lens / "scores" / f"{name}.tmp.npz", dst)
            log(f"{name}: {time.time() - t0:.0f} s, present-token AUROC {np.nanmean(r['present_auc']):.3f}")
    return 0


def cmd_analyze(a) -> int:
    setup(a)
    lens, adir = Path(a.lens), Path(a.out) / "fwd" / "lens"
    tok = LL.tokenizer()
    vocab = np.load(lens / "lens_vocab.npy")
    lab = pd.read_parquet(Path(a.out) / "bundles.parquet")
    fams = LL.layout(lab)
    di = LL.description_index(tok, fams, vocab)
    items = [(f, b) for f, F in fams.items() for b in F["bid"]]
    Wb = T.boot_weights(pd.factorize(pd.Series([b for _, b in items]))[0], a.boot, LL.SEED)
    Wf, i0 = {}, 0
    for f, F in fams.items():
        Wf[f] = Wb[:, i0:i0 + len(F["orgs"])]
        i0 += len(F["orgs"])
    names = [LL.cfg_name(q, L, p) for q, L in LL.configs() + new_configs() for p in LL.PREPS]
    missing = [n for n in names if not (lens / "scores" / f"{n}.npz").exists()]
    if missing:
        raise SystemExit(f"missing scores: {missing}")
    summ, metrics = {}, {}
    for n in names:
        z = np.load(lens / "scores" / f"{n}.npz")
        for norm in LL.NORMS:
            s = LA.summarise(LA.family_values(z[f"S_{norm}"], fams, di), Wf)
            summ[(n, norm)] = s
            metrics[f"{n}|{norm}"] = {qn: dict(pooled=s[qn]["pooled"], pooled_ci95=LA.ci(s[qn]["pooled_boot"], .95),
                                               family=s[qn]["family"]) for qn in LA.QUANTS}
        metrics[f"{n}|z"]["top20_hit"] = LA.hit_rates(z["top"], fams, di, LL.HIT_K)
        metrics[f"{n}|z"]["present_auc_mean"] = (None if n.startswith("text")
                                                 else float(np.nanmean(z["present_auc"])))
    pub = json.loads((REF / "published.json").read_text())["lens_delta_pooled_z"]
    G = pub[G_KEY[0]] - pub[G_KEY[1]]
    contrasts = dict(primary=[], controls=[], descriptive=[])

    def contrast(sec, a_cfg, b_cfg, norm="z", fam=None, keep_boot=False):
        sa, sb = summ[(a_cfg, norm)]["delta"], summ[(b_cfg, norm)]["delta"]
        pa, pb = (sa["pooled"], sb["pooled"]) if fam is None else (sa["family"][fam], sb["family"][fam])
        ba, bb = (sa["pooled_boot"], sb["pooled_boot"]) if fam is None else (sa["family_boot"][fam],
                                                                             sb["family_boot"][fam])
        row = dict(a=a_cfg, b=b_cfg, norm=norm, family=fam or "pooled", a_value=pa, b_value=pb, diff=pa - pb)
        row["ci95"] = LA.ci(ba - bb, .95)
        if keep_boot:
            row["_bootstrap"] = ba - bb
        contrasts[sec].append(row)
        return row

    for a_cfg, b_cfg in PRIMARY:
        r = contrast("primary", a_cfg, b_cfg, keep_boot=True)
        r.update(G=G, margin=MARGIN * G, share_of_G_closed=1 - r["diff"] / G)
    for expl, a_cfg, b_cfg in CONTROLS:
        r = contrast("controls", a_cfg, b_cfg, keep_boot=True)
        r.update(explanation=expl, G=G, margin=MARGIN * G if expl == "label-count weighting" else 0.0)
    confirmatory = contrasts["primary"] + contrasts["controls"]
    assert len(confirmatory) == len(PRIMARY) + len(CONTROLS)
    simultaneous_intervals(confirmatory, level=97.5)
    for r in confirmatory:
        lo, hi = r["ci_simul97.5"]
        r["category"] = (benefit_category(lo, hi) if r.get("explanation") == "loss beyond label count"
                         else category(lo, hi, G))
    for a_cfg, b_cfg in PRIMARY:
        contrast("descriptive", a_cfg, b_cfg, norm="pct")
        contrast("descriptive", a_cfg.replace("center", "raw"), b_cfg.replace("center", "raw"))
        for f in fams:
            contrast("descriptive", a_cfg, b_cfg, fam=f)
    for b in ("act", "actw", "actrespg", "actrespm", "actrespn"):
        contrast("descriptive", "grad_L38_center", f"{b}_L38_center")
    for a_cfg, b_cfg in (("actresp", "act"), ("actrespl", "actresp"), ("actrespg", "actrespl"), ("actrespl", "actw"),
                         ("actrespl", "actrespn"),
                         ("actrespg", "actw")):
        contrast("descriptive", f"{a_cfg}_L38_center", f"{b_cfg}_L38_center")
    for L in (30, 38, 39):
        contrast("descriptive", f"grad_L{L}_center", f"oerr_L{OERR_L}_center")
    contrast("descriptive", f"oerr_L{OERR_L}_center", "actrespl_L38_center")
    contrast("descriptive", "text_center", "actresp_L38_center")
    rows = [dict(key=k, now=metrics[k]["delta"]["pooled"], published=v, diff=metrics[k]["delta"]["pooled"] - v)
            for k, v in pub.items() if k in metrics]
    gated = [r for r in rows if r["key"] in ("grad_L38_center|z", "act_L38_center|z", "actw_L38_center|z")]
    repro = dict(passed=len(gated) == 3 and max(abs(r["diff"]) for r in gated) <= REPRO_TOL, tolerance=REPRO_TOL,
                 max_abs_diff_gated=max(abs(r["diff"]) for r in gated) if gated else None,
                 max_abs_diff_any=max(abs(r["diff"]) for r in rows) if rows else None, G_published=G, rows=rows)
    (adir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (adir / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    (adir / "reproduction.json").write_text(json.dumps(repro, indent=1))
    (adir / "table.md").write_text(table(metrics, contrasts, repro, names))
    log(f"wrote {adir}; reproduction {'OK' if repro['passed'] else 'FAILED'}")
    for r in contrasts["primary"]:
        log(f"{r['a']} - {r['b']}: {r['diff']:+.3f} "
            f"[{r['ci_simul97.5'][0]:+.3f}, {r['ci_simul97.5'][1]:+.3f}] {r['category']}")
    return 0 if repro["passed"] or a.no_gate else 1


def table(metrics, contrasts, repro, names) -> str:
    L = ["# Forward-only baselines: logit lens", "",
         "Results: docs/notes/20261028_fwd_baselines_results.md. Δ = identification AUROC of a bundle's own behaviour "
         "description on showing minus none bundles, within organism, pooled over quirk, backdoor and rare.", "",
         f"Reproduction of the published lens values: {'passed' if repro['passed'] else 'FAILED'} "
         f"(max |diff| {repro['max_abs_diff_gated']}, any config {repro['max_abs_diff_any']}).", "",
         "## Contrasts (confirmatory categories use simultaneous 97.5% intervals across four rows)", "",
         "| section | a | b | norm | family | a − b | individual 95% | simultaneous 97.5% | category |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sec in ("primary", "controls", "descriptive"):
        for r in contrasts[sec]:
            L.append(f"| {sec} | {r['a']} {r['a_value']:.3f} | {r['b']} {r['b_value']:.3f} | {r['norm']} | "
                     f"{r['family']} | {r['diff']:+.3f} | {r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f} | "
                     + (f"{r['ci_simul97.5'][0]:+.3f}, {r['ci_simul97.5'][1]:+.3f}" if 'ci_simul97.5' in r else '')
                     + f" | {r.get('category', '')} |")
    L += ["", "## All configs (z)", "", "| config | Δ pooled | quirk | backdoor | rare | top-20 hit (showing) | "
          "present-token AUROC |", "|---|---|---|---|---|---|---|"]
    for n in names:
        m = metrics[f"{n}|z"]
        pa = "" if m["present_auc_mean"] is None else f"{m['present_auc_mean']:.3f}"
        L.append(f"| {n} | {LA.fmt(m, 'delta')} | {LA.fmt(m, 'delta', 'quirk')} | {LA.fmt(m, 'delta', 'backdoor')} | "
                 f"{LA.fmt(m, 'delta', 'rare')} | {m['top20_hit']['pooled']['show']:.3f} | {pa} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("check", "score", "analyze"):
        p = sub.add_parser(c)
        p.add_argument("--out", default=str(P.OUT))
        p.add_argument("--lens", default=None)
        p.add_argument("--examples", required=True)
        p.add_argument("--boot", type=int, default=LL.BOOT)
        p.add_argument("--no-gate", action="store_true")
    a = ap.parse_args(argv)
    a.lens = a.lens or str(Path(a.out) / "lens")
    return dict(check=cmd_check, score=cmd_score, analyze=cmd_analyze)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

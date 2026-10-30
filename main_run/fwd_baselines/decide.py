#!/usr/bin/env python
"""The pre-registered decision table of Experiment 1, filled in
mechanically from the probe and lens contrasts and the checks. CPU, seconds.

  python main_run/fwd_baselines/decide.py [--out D]   ->  <out>/fwd/decision.md, decision.json

Each primary row gets its category from fwd_probe.category / fwd_lens. For the remaining lead R = a - b, [lo, hi]
is the simultaneous 97.5% interval within its modality; the two modality families are covered jointly at 95% by
the union bound. G is the published gradient - activation lead on that task, with margin 0.25 G:
  closed                       hi < 0.25 G (and lo <= 0)
  closed, small residual lead  hi < 0.25 G and lo > 0
  lead remains                 lo > 0.25 G
  unresolved                   otherwise
The reading of each category is the plan's table (READINGS). Nothing here is chosen after the results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ao_vs_reader"))
import probe_compare as P  # noqa: E402

READINGS = {
    "position": {
        "closed": "response-only activation closes at least three quarters of the original lead; prompt positions may account "
                  "for it",
        "lead remains": "removing the prompt positions does not explain the lead",
    },
    "forward surprise weighting": {
        "lead remains": "response focus and forward-loss weighting leave most of the lead unexplained",
    },
    "output error": {
        "closed": "the output-error vector reduces the mid-layer gradient's remaining gap below 0.25 G",
        "lead remains": "the mid-layer gradient carries more than the output error: the backward pass through the "
                        "network adds information (or several layers together carry more than one vector)",
    },
    "combined forward set": {
        "closed": "the pre-specified loss-weighted activation plus output error reduces the gradient's remaining "
                  "lead below 0.25 G",
        "lead remains": "the gradient retains more than 0.25 G lead over the pre-specified forward pair",
    },
    "backprop routing": {
        "closed": "adding the gradient to the fixed forward-only set leaves a gap below 0.25 G",
        "lead remains": "adding the gradient improves the pre-specified loss-weighted-activation plus output-error "
                        "set; other forward combinations were not tested by this row",
    },
}
LENS_READINGS = {
    "closed": "a forward input reduces the gradient's remaining lens advantage below 0.25 G",
    "lead remains": "the gradient's lens advantage survives this forward input",
}


def reading(expl: str, cat: str) -> str:
    base = "closed" if cat.startswith("closed") else cat
    table = LENS_READINGS if expl == "lens" else READINGS[expl]
    if base == "unresolved":
        return "unresolved: the interval does not separate a closed gap from a remaining one"
    if expl == "forward surprise weighting" and base == "closed":
        raise ValueError("loss weighting needs the label-count controls before interpretation")
    text = table[base]
    return text + ("; a positive residual lead remains" if cat == "closed, small residual lead" else "")


def loss_reading(cat: str, count_cat: str, benefit_cat: str) -> str:
    if not cat.startswith("closed"):
        if count_cat.startswith("closed"):
            return "summed-loss weighting leaves the gap open, but label-count weighting closes it"
        return reading("forward surprise weighting", cat)
    if count_cat.startswith("closed"):
        return "label-count weighting also closes the gap; this does not isolate per-token surprise"
    if count_cat == "lead remains" and benefit_cat == "loss better":
        return "summed-loss weighting closes the gap beyond matched label-count weighting; per-token surprise helps"
    return "the summed-loss input closes the gap, but its advantage over label-count weighting is unresolved"


def task_answer(primary: dict, controls: dict) -> str:
    pos, loss, err, combined, routing = [primary[x]["category"] for x in
                                         ("position", "forward surprise weighting", "output error",
                                          "combined forward set", "backprop routing")]
    count = controls["label-count weighting"]["category"]
    benefit = controls["loss beyond label count"]["category"]
    if pos.startswith("closed"):
        return "position"
    if pos == "unresolved":
        return "unresolved: response position"
    if count.startswith("closed"):
        return "label-count weighting; per-token surprise is not isolated"
    if count == "unresolved":
        return "unresolved: label-count weighting"
    if loss.startswith("closed"):
        if benefit == "loss better":
            return "forward surprise weighting beyond label count"
        return "unresolved: loss versus label count"
    if loss == "unresolved":
        return "unresolved: forward weighting"
    if err.startswith("closed"):
        return "output error"
    if err == "unresolved":
        return "unresolved: output error"
    if combined.startswith("closed"):
        return "combined forward pair"
    if combined == "unresolved":
        return "unresolved: combined forward pair"
    if routing == "lead remains":
        return "gradient lead and increment over the pre-specified forward pair"
    return "unresolved: backprop routing"


def survival_answer(primary: dict, controls: dict) -> str:
    """Q1 for four confirmatory forward sets compared directly with the all-layer gradient."""
    cats = {"response-only activation": primary["position"]["category"],
            "label-count weighting": controls["label-count weighting"]["category"],
            "summed-loss weighting": primary["forward surprise weighting"]["category"],
            "combined forward pair": primary["combined forward set"]["category"]}
    closed = [name for name, cat in cats.items() if cat.startswith("closed")]
    unresolved = [name for name, cat in cats.items() if cat == "unresolved"]
    if closed:
        return "lead closes against " + ", ".join(closed)
    if unresolved:
        return "unresolved against " + ", ".join(unresolved)
    return "lead remains over all four confirmatory forward sets"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def run(a) -> int:
    fwd = Path(a.out) / "fwd"
    checks = {n: load(fwd / f"checks_{n}.json") for n in ("vechash", "docs", "bundles", "readerdirs")}
    checks["lens"] = load(fwd / "lens" / "checks.json")
    pc, lc = load(fwd / "probe" / "contrasts.json"), load(fwd / "lens" / "contrasts.json")
    pr, lr = load(fwd / "probe" / "reproduction.json"), load(fwd / "lens" / "reproduction.json")
    if pc is None or lc is None or pr is None or lr is None:
        raise SystemExit("probe or lens results missing; run fwd_probe.py and fwd_lens.py analyze first")
    if (len(pc.get("primary", [])) != 20 or len(pc.get("controls", [])) != 8 or
            len(lc.get("primary", [])) != 2 or len(lc.get("controls", [])) != 2 or
            any("ci_simul97.5" not in r for sec in ("primary", "controls") for r in pc[sec] + lc[sec])):
        raise SystemExit("confirmatory controls or simultaneous intervals missing; rerun probe and lens analysis")
    ok = {k: (v or {}).get("passed") for k, v in checks.items()}
    ok["probe reproduction"], ok["lens reproduction"] = pr["passed"], lr["passed"]
    ok["probe stage"] = getattr(a, "probe_stage_exit", 0) == 0
    ok["lens stage"] = getattr(a, "lens_stage_exit", 0) == 0
    rows = []
    probe_controls = {(r["task"], r["explanation"]): r for r in pc["controls"]}
    for r in pc["primary"]:
        count = probe_controls[r["task"], "label-count weighting"]["category"]
        benefit = probe_controls[r["task"], "loss beyond label count"]["category"]
        rows.append(dict(task=r["task"], explanation=r["explanation"], contrast=f"{r['a']} − {r['b']}",
                         diff=r["diff"], ci95=r["ci95"], ci_simul97_5=r["ci_simul97.5"], G=r["G"], category=r["category"],
                         reading=(loss_reading(r["category"], count, benefit) if r["explanation"] ==
                                  "forward surprise weighting" else reading(r["explanation"], r["category"]))))
    lens_controls = {r["explanation"]: r for r in lc["controls"]}
    for r in lc["primary"]:
        is_loss = r["b"].startswith("actrespl_")
        rows.append(dict(task="logit lens Δ (layer 38)", explanation="lens", contrast=f"{r['a']} − {r['b']}",
                         diff=r["diff"], ci95=r["ci95"], ci_simul97_5=r["ci_simul97.5"], G=r["G"], category=r["category"],
                         reading=(loss_reading(r["category"], lens_controls["label-count weighting"]["category"],
                                               lens_controls["loss beyond label count"]["category"]) if is_loss
                                  else reading("lens", r["category"]))))
    control_rows = [dict(task=r.get("task", "logit lens Δ (layer 38)"), explanation=r["explanation"],
                         contrast=f"{r['a']} − {r['b']}", diff=r["diff"], ci95=r["ci95"],
                         ci_simul97_5=r["ci_simul97.5"], category=r["category"]) for r in pc["controls"] + lc["controls"]]
    answers = {t: task_answer({r["explanation"]: r for r in rows if r["task"] == t},
                              {x: probe_controls[t, x] for x in ("label-count weighting", "loss beyond label count")})
               for t in ("P1", "P3", "P3_rare", "P4")}
    survival = {t: survival_answer({r["explanation"]: r for r in rows if r["task"] == t},
                                   {x: probe_controls[t, x] for x in ("label-count weighting", "loss beyond label count")})
                for t in ("P1", "P3", "P3_rare", "P4")}
    all_ok = all(bool(v) for v in ok.values())
    L = ["# Experiment 1: decision table (filled in mechanically)", "",
         "Categories use the simultaneous "
         "97.5% interval within each modality and a margin of 0.25 G; the two modalities are jointly covered "
         "at approximately 95%.", "",
         "## Checks", "", "| check | passed |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in ok.items()]
    if not all_ok:
        L += ["", "**A check failed or is missing: the rows below must not be interpreted until it is resolved.**"]
    L += ["", "## Q1: gradient lead versus the four confirmatory forward sets", ""]
    L += [f"- {t}: {answer}" for t, answer in survival.items()]
    L += ["", "## Explanation answer per task", ""]
    L += [f"- {t}: {answer}" for t, answer in answers.items()]
    L += ["", "## Primary rows", "",
          "| task | explanation | contrast | remaining lead | individual 95% | simultaneous 97.5% | G | category | reading |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['explanation']} | {r['contrast']} | {r['diff']:+.3f} | "
                 f"{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f} | "
                 f"{r['ci_simul97_5'][0]:+.3f}, {r['ci_simul97_5'][1]:+.3f} | {r['G']:.3f} | {r['category']} | {r['reading']} |")
    L += ["", "## Label-count controls", "",
          "| task | explanation | contrast | a − b | individual 95% | simultaneous 97.5% | category |",
          "|---|---|---|---|---|---|---|"]
    for r in control_rows:
        L.append(f"| {r['task']} | {r['explanation']} | {r['contrast']} | {r['diff']:+.3f} | "
                 f"{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f} | "
                 f"{r['ci_simul97_5'][0]:+.3f}, {r['ci_simul97_5'][1]:+.3f} | {r['category']} |")
    L += ["", "## Post hoc follow-up: retrieval at layer 38 (not part of the decision)", "",
          "| contrast | a − b | 95% |", "|---|---|---|"]
    for r in pc["posthoc"]:
        L.append(f"| {r['a']} {r['a_value']:.3f} − {r['b']} {r['b_value']:.3f} | {r['diff']:+.3f} | "
                 f"{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f} |")
    (fwd / "decision.md").write_text("\n".join(L) + "\n")
    (fwd / "decision.json").write_text(json.dumps(dict(checks=ok, all_checks_passed=all_ok, rows=rows,
                                                        controls=control_rows, survival=survival,
                                                        answers=answers), indent=1))
    print("\n".join(L))
    return 0 if all_ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(P.OUT))
    ap.add_argument("--probe-stage-exit", type=int, default=0)
    ap.add_argument("--lens-stage-exit", type=int, default=0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

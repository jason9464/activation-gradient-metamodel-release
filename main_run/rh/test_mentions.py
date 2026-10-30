#!/usr/bin/env python
"""Exploratory, post hoc (RH results note, docs/notes/20261026_rh_results.md): share of reader answers that mention
tests, per read step, against the hacker's rollout hack rate. Not a pre-registered rule; the plan lists keyword
families only as exploratory, and this pattern was chosen after reading answers.

  python -m main_run.rh.test_mentions --root ../rh_runs --out docs/rh_results/test_mentions_by_step.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

TEST_RE = re.compile(r"\btest(s|ing|ed)?\b|test[- ]cases?|harness|assert", re.IGNORECASE)
READER = os.environ.get("RH_READER", "rl_paper_c040")  # behaviour readers: beh_sft_7082, beh_rl_c040
RUNS = ("rh_hacker", "rh_anti")


def per_step(path: Path) -> dict[int, list[int]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            texts = [t for b in r["inputs"].values() for t in b["texts"]]
            out[int(r["step"])] = [sum(bool(TEST_RE.search(t)) for t in texts), len(texts)]
    return out


def build(root: Path) -> dict:
    from scipy.stats import spearmanr

    res = dict(pattern=TEST_RE.pattern, note="exploratory, post hoc; hack rate = changed_tests + hacked_harness",
               cells={})
    hr = {run: {int(x["step"]): x for x in json.loads((root / run / "hack_rates.json").read_text())} for run in RUNS}
    for v in ("ckptdiff", "dwx"):
        for q in ("behavioural", "preference"):
            cell = {}
            for run in RUNS:
                d = {}
                for tag in ("rhtask", "rhtask_trajectory"):
                    d.update(per_step(root / run / "e2" / f"{v}_{READER}_{q}_{tag}.jsonl"))
                steps = sorted(d)
                rate = [hr[run][t]["changed_tests"] + hr[run][t]["hacked_harness"] for t in steps]
                rho, p = spearmanr([d[t][0] for t in steps], rate)
                cell[run] = dict(steps=steps, mentions=[d[t][0] for t in steps], n=[d[t][1] for t in steps],
                                 hack_rate=[round(x, 4) for x in rate], spearman_rho=round(float(rho), 3),
                                 spearman_p=round(float(p), 4))
            res["cells"][f"{v}_{q}"] = cell
    ctl = {}
    for q in ("behavioural", "preference"):
        k = n = 0
        for c in ("no_vector", "random"):
            for tag in ("ctl", "ctl_trajectory"):
                for t, (kk, nn) in per_step(root / "rh_hacker" / "e2" / f"ckptdiff_{READER}_{q}_{c}_{tag}.jsonl").items():
                    k, n = k + kk, n + nn
        ctl[q] = [k, n]
    res["controls"] = ctl
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = build(a.root)
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    for c, d in res["cells"].items():
        print(c, {run: (x["spearman_rho"], x["spearman_p"]) for run, x in d.items()})
    print("controls", res["controls"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

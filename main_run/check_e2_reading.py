#!/usr/bin/env python
"""Integrity check 6 of the new-setting plan (docs/notes/20261025_new_setting_plan.md section 7) for one reading file.

  python check_e2_reading.py <run>/e2/<kind>_<reader>_<question>.jsonl --steps 500,...,7000 --windows w1000,w3000,cum
      --samples 100 --question preference

Passes (exit 0) when the file has exactly one line per listed read step, in order; every line has exactly the listed
windows, each with n_samples = len(texts) = --samples; and the reading's ``_settings.json`` records the question text
of --question (judge_dolphin_answers.QUESTIONS). Prints one json line with the counts; exit 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def check(path, steps, windows, samples, question):
    from judge_dolphin_answers import QUESTIONS

    problems = []
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                problems.append(f"line {i + 1} is not json")
    got = [int(r.get("step", -1)) for r in rows]
    if got != steps:
        problems.append(f"steps {got[:3]}...{got[-2:]} ({len(got)}) differ from the {len(steps)} listed")
    for r in rows:
        inp = r.get("inputs", {})
        if sorted(inp) != sorted(windows):
            problems.append(f"step {r.get('step')}: windows {sorted(inp)}")
            continue
        for w in windows:
            e = inp[w]
            if e.get("n_samples") != samples or len(e.get("texts", [])) != samples:
                problems.append(f"step {r.get('step')} {w}: n_samples {e.get('n_samples')}, texts {len(e.get('texts', []))}")
    sp = path[: -len(".jsonl")] + "_settings.json"
    asked = None
    if os.path.exists(sp):
        with open(sp) as f:
            asked = json.load(f).get("question")
    if asked != QUESTIONS[question]:
        problems.append(f"settings question {asked!r}, expected {QUESTIONS[question]!r}")
    return dict(file=path, ok=not problems, lines=len(rows), windows=windows, samples=samples, question=asked,
                problems=problems[:10])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--steps", required=True)
    ap.add_argument("--windows", default="w1000,w3000,cum")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--question", default="preference")
    a = ap.parse_args(argv)
    res = check(a.path, [int(x) for x in a.steps.split(",") if x.strip()],
                [w.strip() for w in a.windows.split(",") if w.strip()], a.samples, a.question)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

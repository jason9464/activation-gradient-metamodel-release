#!/usr/bin/env python
"""Rerun of a new-setting student with the foreign-batch probe (plan docs/notes/20261026_clean_batch_update_plan.md).

A thin wrapper of ``train_subliminal.main``: it loads the foreign batches of the target arm (``cleanbatch_probe.
load_foreign``), replaces ``train_subliminal.make_adamfx_trainer_class`` by a function that wraps the class it returns
with ``cleanbatch_probe.make_probe_trainer_class`` (so the probe runs at the start of the probe steps' training_step),
and calls ``train_subliminal.main`` with every other argument unchanged. train_subliminal.py itself is not edited, so
the training code path is the main runs' one; the probe only adds work at the probe steps and restores the state
(module docstring of cleanbatch_probe.py).

  python train_cleanbatch.py --probe-steps 500:3000:100 [--foreign-file runs/cleanbatch/foreign_batches.json] \
      [--probe-variants default|extended] <train_subliminal.py arguments, which must include --save-adamfx and
      --adamfx-fixed-docs>
--probe-variants extended adds the variants mom, momfix, hist, histfix, curr (cleanbatch_probe.py docstring, "Extended
variants"); the default writes the five variants of the first clean-batch runs, unchanged.
Outputs: those of train_subliminal.py in --out, plus <out>/cleanbatch/ (probe_step_<t>.npz, probe_meta.json).
Exit status: train_subliminal's; 5 when the probe did not complete (a planned step missing or a weight check failed).
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleanbatch_probe as cp  # noqa: E402


def parse_steps(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi, ev = (int(x) for x in part.split(":"))
            out |= set(range(lo, hi + 1, ev))
        else:
            out.add(int(part))
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--probe-steps", required=True, help="lo:hi:every or a comma list")
    ap.add_argument("--foreign-file", default=cp.FOREIGN_FILE)
    ap.add_argument("--probe-variants", default="default", choices=sorted(cp.VARIANT_SETS))
    a, rest = ap.parse_known_args(argv)
    import train_subliminal as ts

    ta = ts.parse_args(rest)
    if not ta.save_adamfx or not ta.adamfx_fixed_docs:
        raise SystemExit("train_cleanbatch needs --save-adamfx and --adamfx-fixed-docs")
    if ta.arm not in cp.OTHER:
        raise SystemExit(f"--arm {ta.arm}: the probe has foreign batches for {sorted(cp.OTHER)} only")
    steps = parse_steps(a.probe_steps)
    if not steps or steps[0] < 2 or (ta.stop_step and steps[-1] > ta.stop_step):
        raise SystemExit(f"probe steps {steps[:3]}..{steps[-1:]} must lie in 2..--stop-step")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ta.model)
    tok.padding_side = "right"
    foreign, fmeta = cp.load_foreign(a.foreign_file, ta.arm, tok, ta.max_len, ta.think_loss)
    probe = cp.Probe(os.path.abspath(ta.out), steps, foreign, fmeta, log=lambda m: ts.log.info("cleanbatch: %s", m),
                     variants=a.probe_variants)
    orig = ts.make_adamfx_trainer_class

    def patched(base, pop_line_no):
        return cp.make_probe_trainer_class(orig(base, pop_line_no), probe)

    ts.make_adamfx_trainer_class = patched
    print(f"train_cleanbatch: target {ta.arm}, foreign {fmeta['foreign_arm']} batches {len(foreign)} x "
          f"{foreign[0]['input_ids'].shape[0]}, probe steps {steps[0]}..{steps[-1]} ({len(steps)}), variants "
          f"{a.probe_variants} {list(probe.variants)}, out {ta.out}",
          flush=True)
    ts.main(rest)
    ok = probe.finish()
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())

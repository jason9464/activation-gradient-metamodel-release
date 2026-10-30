#!/usr/bin/env python
"""Orchestrate EM Δh (input-5a) and cumulative direct-write (dwx, input-4f analogue) extraction.

One Fp32Model serves every (run, probe set): ``ckpt_diff_inputs.compute_fixed`` + ``build_fixed --windows cum`` (5a),
then ``dwx_inputs.compute_dwx`` + ``build_dwx`` (dwx, one more fp32 forward per adapter; ``--no-dwx`` skips it).
Results: ``docs/notes/20261025_em_dact_results.md``. Fallback if the shared-model path slips: invoke
``ckpt_diff_inputs.py --docs`` and ``main_run.em.dwx_inputs`` separately per (run, set) (one fp32 load each).

Usage:
  python -m main_run.em.dact_extract --runs em_s42,em_s1 --docs-dir <dir> \\
      --sets evalq,neutral,train --steps 0,5,10,...,375 --device cuda:0 --bf16-adapters-ok
  # Stage B adds --sets ...,web
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ckpt_diff_inputs as cdi  # noqa: E402
from main_run.em import dwx_inputs as dwx  # noqa: E402

DACT_STEPS = (0, 5, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 100, 150, 250, 375)
DACT_SPLIT_STEPS = (50, 150, 375)
PROBE_SETS = ("evalq", "neutral", "train", "web", "rhtask")  # rhtask: RH see docs/notes/20261026_rh_results.md
DEFAULT_WINDOWS = "cum"


def subdir_for(probe_set: str) -> str:
    return f"ckptdiff_{probe_set}"


def docs_path(docs_dir: Path, probe_set: str) -> Path:
    return Path(docs_dir) / f"{probe_set}.jsonl"


def extract_all(runs: list[str], docs_dir: Path, sets: list[str], steps: list[int], *,
                device: str = "cuda:0", allow_bf16: bool = True, windows: str = DEFAULT_WINDOWS,
                split_steps: tuple[int, ...] = DACT_SPLIT_STEPS, build: bool = True,
                loader: bool = True, model=None, log_fn=None, with_dwx: bool = True,
                pair_run: str | None = None) -> dict:
    """One Fp32Model across every (run, set); compute h(t) then optional ``build_fixed --windows cum``.

    Runs after the first are built with ``pair_run`` = the first run, so ``build_fixed`` checks that both seeds
    used the same documents and token ids (``same_docs_as_pair_run``). ``pair_run`` given: every run is built against
    it (one seed per GPU in separate processes)."""
    lg = log_fn or (lambda m: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dact_extract] {m}", flush=True))
    summary: dict = dict(runs={}, steps=steps, sets=sets, allow_bf16=allow_bf16, windows=windows, dwx=with_dwx)
    for ri, run in enumerate(runs):
        summary["runs"][run] = {}
        pair = pair_run if pair_run else (runs[0] if ri > 0 else None)
        for probe_set in sets:
            docs = docs_path(docs_dir, probe_set)
            if not docs.exists():
                raise SystemExit(f"missing docs for {probe_set}: {docs}")
            sub = subdir_for(probe_set)
            lg(f"compute {run} {probe_set} ({sub}) on {docs}")
            rc, res = cdi.compute_fixed(
                run, str(docs), steps, device=device, allow_bf16=allow_bf16, subdir=sub, model=model, log_fn=lg,
            )
            model = res.get("model", model)
            entry = dict(compute_rc=rc, compute={k: v for k, v in res.items() if k != "model"}, subdir=sub)
            if rc != 0:
                summary["runs"][run][probe_set] = entry
                lg(f"compute failed rc={rc} for {run}/{probe_set}")
                continue
            suffix = f"{probe_set}_dact"
            out5 = os.path.join(cdi.resolve_run(run), "e2", f"{cdi.VARIANT}_inputs_{suffix}.npz")
            if build and os.path.exists(out5):  # a rerun after a stop: the built file stays as it is
                lg(f"build {run} {probe_set}: {out5} exists, kept")
                entry.update(pair_run=pair, build_rc=0, build=dict(kept=out5))
            elif build:
                lg(f"build {run} {probe_set} windows={windows} suffix={suffix}")
                brc, bres = cdi.build_fixed(
                    run, str(docs), steps, windows_spec=windows, suffix=suffix, split_steps=split_steps,
                    pair_run=pair, loader=loader, subdir=sub, allow_bf16=allow_bf16, log_fn=lg,
                )
                entry["pair_run"] = pair
                entry["build_rc"] = brc
                entry["build"] = bres
            if with_dwx:
                lg(f"dwx compute {run} {probe_set}")
                drc, dres = dwx.compute_dwx(run, str(docs), steps, model=model, device=device, probe_set=probe_set,
                                            allow_bf16=allow_bf16, log_fn=lg)
                model = dres.get("model", model)
                entry["dwx_compute_rc"] = drc
                outd = os.path.join(cdi.resolve_run(run), "e2", f"{dwx.VARIANT}_inputs_{suffix}.npz")
                if drc == 0 and build and os.path.exists(outd):
                    lg(f"dwx build {run} {probe_set}: {outd} exists, kept")
                    entry["dwx_build_rc"], entry["dwx_build"] = 0, dict(kept=outd)
                elif drc == 0 and build:
                    dbrc, dbres = dwx.build_dwx(run, str(docs), steps, probe_set=probe_set, suffix=f"{probe_set}_dact",
                                                split_steps=split_steps, pair_run=pair, allow_bf16=allow_bf16,
                                                loader=loader, log_fn=lg)
                    entry["dwx_build_rc"], entry["dwx_build"] = dbrc, dbres
            summary["runs"][run][probe_set] = entry
    summary["model_reused"] = model is not None
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of run dirs or names under agm_data/runs")
    ap.add_argument("--docs-dir", required=True, type=Path, help="directory with <set>.jsonl per probe set")
    ap.add_argument("--sets", default="evalq,neutral,train", help="comma list of probe sets")
    ap.add_argument("--steps", default=",".join(str(t) for t in DACT_STEPS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--split-steps", default=",".join(str(t) for t in DACT_SPLIT_STEPS))
    ap.add_argument("--bf16-adapters-ok", action="store_true",
                    help="no-op: bf16 adapters are accepted by default (EM adapters are bf16)")
    ap.add_argument("--fp32-adapters-only", action="store_true", help="refuse bf16 adapters")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-loader", action="store_true")
    ap.add_argument("--no-dwx", action="store_true", help="skip the cumulative direct-write input (dwx)")
    ap.add_argument("--pair-run", default=None,
                    help="build every run against this run's docs (one seed per GPU: the other seed's run dir)")
    ap.add_argument("--summary", default=None, help="write a json summary here")
    a = ap.parse_args(argv)
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    bad = [s for s in sets if s not in PROBE_SETS]
    if bad:
        raise SystemExit(f"unknown probe sets {bad}; known {PROBE_SETS}")
    steps = cdi.parse_step_range(a.steps)
    split_steps = tuple(int(x) for x in a.split_steps.split(",") if x.strip())
    allow_bf16 = not a.fp32_adapters_only
    t0 = time.time()
    model = cdi.Fp32Model(a.device)
    summary = extract_all(
        runs, a.docs_dir, sets, steps, device=a.device, allow_bf16=allow_bf16, windows=a.windows,
        split_steps=split_steps, build=not a.no_build, loader=not a.no_loader, model=model, with_dwx=not a.no_dwx,
        pair_run=a.pair_run,
    )
    summary["seconds"] = round(time.time() - t0, 1)
    if a.summary:
        Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary).write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    failed = []
    for run, sets_d in summary["runs"].items():
        for s, e in sets_d.items():
            if any(e.get(k, 0) not in (0, None) for k in ("compute_rc", "build_rc", "dwx_compute_rc",
                                                          "dwx_build_rc")):
                failed.append(f"{run}/{s}")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2, default=str))
    if failed:
        print(f"failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

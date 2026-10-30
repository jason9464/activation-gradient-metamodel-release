#!/usr/bin/env python
"""Judge RH readings after pull, with calibration and file preflight before any API calls.

  python -m main_run.rh.judge_all --root ../rh_runs --dry-run
  python -m main_run.rh.judge_all --root ../rh_runs --max-total-cost 2
  python -m main_run.rh.judge_all --root ../rh_runs --scope all --dry-run

The default scope is the preregistered primary family, its document halves and its two controls. ``targeted`` adds
only the targeted-question cells (rhtask; for readers without neutral or AO readings, e.g. RH_READER=beh_rl_c040).
``all`` adds the secondary and AO families. Exploratory trajectory files are not part of the detection rule and are
not judged here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from main_run.ao_diff_read import QUESTIONS as AO_QUESTIONS
from main_run.em import judge_reader as jr
from main_run.rh import common as C
from main_run.rh import summary as sm

CALIBRATION = Path(__file__).resolve().parents[2] / "configs" / "rh" / "calib_rh_v1.csv"


def question_for(cell: str) -> str:
    if cell.startswith("ao_"):
        return AO_QUESTIONS["targeted" if cell.endswith("_targeted") else "describe"]
    return C.TARGETED_QUESTION if cell.endswith("_tgt") else jr.DEFAULT_BEHAVIOURAL_QUESTION


def jobs(scope: str) -> list[tuple[str, str, str, str, str]]:
    """(cell, run, reading stem, window, exact question text), including primary controls."""
    out = []
    for cell, (family, stem, window, halves) in sm.CELLS.items():
        if scope == "primary" and family != "primary":
            continue
        if scope == "targeted" and family != "primary" and not cell.endswith("_tgt"):
            continue
        for run in sm.RUNS:
            out.append((cell, run, stem, window, question_for(cell)))
            if halves:
                for h in (0, 1):
                    out.append((cell, run, stem + f"_half{h}", window, question_for(cell)))
    for qkey in ("behavioural", "preference"):
        q = C.TARGETED_QUESTION if qkey == "preference" else jr.DEFAULT_BEHAVIOURAL_QUESTION
        for cond in sm.CTL_CONDITIONS:
            out.append((f"control_{qkey}", sm.CTL_RUN, sm.CTL_STEM[qkey].format(cond=cond), "cum", q))
    return out


def require_calibration(root: Path) -> None:
    sha = hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()
    for label, question in (("behavioural", jr.DEFAULT_BEHAVIOURAL_QUESTION), ("targeted", C.TARGETED_QUESTION)):
        p = root / "calib" / f"calib_rh_v1_{label}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        if not (d.get("accept") is True and d.get("n_refused") == 0 and d.get("items_sha256") == sha
                and d.get("truth") == jr.TRUTH[sm.TRUTH] and d.get("question_default") == question):
            raise SystemExit(f"{p}: calibration does not match the pinned CSV, truth and {label} question")


def require_matched_documents(root: Path, scope: str) -> None:
    """Require identical probe bytes and token IDs for both arms and both input constructions."""
    sets = ("rhtask", "neutral") if scope == "all" else ("rhtask",)
    for probe_set in sets:
        docs = root / "docs" / f"{probe_set}.jsonl"
        if not docs.exists():
            raise SystemExit(f"missing generated probe documents: {docs} (run setup and pull the results first)")
        docs_sha = hashlib.sha256(docs.read_bytes()).hexdigest()
        expected_n = C.N_RHTASK if probe_set == "rhtask" else 32
        ids = set()
        for run in sm.RUNS:
            manifest = json.loads((root / run / "RUN.json").read_text(encoding="utf-8"))["settings"]
            spec = C.RUNS[run]
            if (manifest.get("model") != C.MODEL_ID or manifest.get("model_revision") != C.MODEL_REVISION
                    or manifest.get("source_repo") != spec["repo"]
                    or manifest.get("source_revision") != spec["revision"]):
                raise SystemExit(f"{root / run / 'RUN.json'}: run provenance differs from the pinned plan")
            for kind in ("ckptdiff", "dwx"):
                p = root / run / f"{kind}_{probe_set}" / "docs.json"
                if not p.exists():
                    raise SystemExit(f"missing extraction metadata: {p} (run chain and pull the results first)")
                meta = json.loads(p.read_text(encoding="utf-8"))
                half = meta.get("half")
                if (meta.get("docs_sha256") != docs_sha or meta.get("n_docs") != expected_n
                        or not isinstance(half, list) or len(half) != expected_n
                        or half.count(0) != expected_n // 2 or half.count(1) != expected_n // 2
                        or not meta.get("ids_sha1")):
                    raise SystemExit(f"{p}: document bytes, count, halves or token IDs differ from the probe set")
                ids.add(meta["ids_sha1"])
        if len(ids) != 1:
            raise SystemExit(f"{probe_set}: token IDs differ across arms or input constructions: {sorted(ids)}")


def require_existing_matches_reading(reading: Path, output: Path, window: str) -> None:
    """Do not silently reuse verdicts for an old reading with the same output filename."""
    raw = {}
    for step, lab, i, answer in jr.iter_reading_answers(reading, window=window):
        if step in C.PRIMARY_STEPS:
            key = (step, lab, i)
            if key in raw:
                raise SystemExit(f"{reading}: duplicate raw answer {key}")
            raw[key] = answer
    seen = set()
    with output.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            if step not in C.PRIMARY_STEPS:
                continue
            key = (step, row.get("window"), row.get("answer_index"))
            if key in seen or key not in raw or row.get("answer") != raw[key]:
                raise SystemExit(f"{output}: judge row {key} does not match {reading}")
            seen.add(key)
    if seen != set(raw):
        raise SystemExit(f"{output}: judge rows do not cover all primary answers in {reading}")


def input_for(root: Path, run: str, cell: str, stem: str) -> Path:
    """The exact pulled vector file used to generate a planned reading."""
    if cell.startswith("ao_"):
        kind = "ckptdiff" if "_dh_" in cell else "dwx"
        suffix = "rhtask_dact"
    elif cell.startswith("control_"):
        kind, suffix = "ckptdiff", "rhtask_dact"
    else:
        kind = stem.split("_", 1)[0]
        suffix = "neutral_dact" if stem.endswith("_neutral") else "rhtask_dact"
        if stem.endswith("_half0") or stem.endswith("_half1"):
            suffix += stem[-6:]
    return root / run / "e2" / f"{kind}_inputs_{suffix}.npz"


def require_settings_input(settings: Path, data: dict, expected: Path, cell: str, sha_cache: dict) -> None:
    if expected not in sha_cache:
        h = hashlib.sha256()
        with expected.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        sha_cache[expected] = h.hexdigest()
    if cell.startswith("ao_"):
        info = data.get("inputs") or {}
        recorded, sha = info.get("file"), info.get("sha256")
    else:
        recorded, sha = data.get("inputs"), data.get("inputs_sha256")
    if (not isinstance(recorded, str) or Path(recorded).name != expected.name
            or sha != sha_cache[expected]):
        raise SystemExit(f"{settings}: input file or SHA differs from {expected}")


def preflight(root: Path, scope: str) -> tuple[list[tuple], float, int]:
    require_calibration(root)
    require_matched_documents(root, scope)
    pending, total_cost, existing = [], 0.0, 0
    sha_cache = {}
    for cell, run, stem, window, question in jobs(scope):
        reading = root / run / "e2" / f"{stem}.jsonl"
        settings = reading.with_name(reading.stem + "_settings.json")
        if not reading.exists() or not settings.exists():
            raise SystemExit(f"missing reading or settings for {cell}/{run}: {reading}")
        settings_data = json.loads(settings.read_text(encoding="utf-8"))
        got = settings_data.get("question")
        if got != question:
            raise SystemExit(f"{settings}: question {got!r} != planned {question!r}")
        base_sha = settings_data.get("model_revision" if cell.startswith("ao_") else "reader_base_sha")
        if base_sha != C.MODEL_REVISION:
            raise SystemExit(f"{settings}: reader base snapshot {base_sha!r} != pinned {C.MODEL_REVISION}")
        require_settings_input(settings, settings_data, input_for(root, run, cell, stem), cell, sha_cache)
        counts = Counter(t for t, _lab, _i, _answer in jr.iter_reading_answers(reading, window=window)
                         if t in C.PRIMARY_STEPS)
        if any(counts[t] != sm.N_PER_STEP for t in C.PRIMARY_STEPS):
            raise SystemExit(f"{reading}: primary-step answer counts {dict(counts)}, need 100 at 43, 45 and 47")
        output = sm.judge_path(root, run, stem, window)
        if output.exists():
            sm.load_counts(output, C.PRIMARY_STEPS, expected_window=window)
            require_existing_matches_reading(reading, output, window)
            existing += 1
            continue
        info = jr.judge_reading(reading, truth_key=sm.TRUTH, question_text=question, window=window, dry_run=True)
        pending.append((reading, output, window, question))
        total_cost += info["cost_estimate_usd"]
    return pending, total_cost, existing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scope", choices=("primary", "targeted", "all"), default="primary")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-total-cost", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=jr.DEFAULT_WORKERS)
    a = ap.parse_args(argv)
    pending, estimate, existing = preflight(a.root, a.scope)
    print(json.dumps(dict(scope=a.scope, pending=len(pending), existing=existing,
                          estimated_uncached_cost_usd=round(estimate, 4)), indent=2), flush=True)
    if a.dry_run:
        return 0
    if estimate > a.max_total_cost:
        raise SystemExit(f"estimated ${estimate:.3f} exceeds --max-total-cost ${a.max_total_cost:.3f}")
    for reading, output, window, question in pending:
        result = jr.judge_reading(reading, truth_key=sm.TRUTH, question_text=question, window=window,
                                  out=output, workers=a.workers)
        sm.load_counts(output, C.PRIMARY_STEPS, expected_window=window)
        print(f"{output}: YES {result['n_yes']}, NO {result['n_no']}, refused {result['n_refused']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

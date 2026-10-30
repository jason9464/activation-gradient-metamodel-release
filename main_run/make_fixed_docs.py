#!/usr/bin/env python
"""Fixed documents of the new-setting runs: 128 clean-number rows outside the control student's training rows.

Input 4f (the
direct-write term dW_t X on fixed documents, computed inside the training process, ``train_subliminal.py
--adamfx-fixed-docs``) and input 5 (fp32 activation differences between saved adapters, ``ckpt_diff_inputs.py``)
use the same 128 documents for both students (dolphin and the clean control).

Selection. The candidates are the rows of the clean-number file (``--clean-path``, default
``agm_data/clean_nums/filtered_full.jsonl``) that the control run did not train on. The control's rows are read from
its RUN.json (``--control-run``, default ``runs/full_clean_s1``): arm ``clean`` trains on the first ``n_rows`` lines
of ``settings.data_path`` (train_subliminal.read_rows), so the candidates are the lines ``>= n_rows`` (50,000 of
69,698). The script refuses a control run whose data path is not the clean file or whose arm is not ``clean``.
``numpy.random.default_rng(seed).choice(candidates, n, replace=False)`` (seed 0, n 128) draws the rows; the draw
order is kept, and the first n/2 drawn rows are half 0, the rest half 1 (the split-half check of input 5).

Output (``--out``, default ``agm_data/newset/fixed_docs_128.jsonl``), one line per document in draw order:
  row           line number in the clean file (0-based)
  order         position in the draw (0..n-1)
  half          0 for order < n/2, 1 otherwise
  prompt        the user prompt (the trainer's text field; the empty system prompt is dropped as in training)
  completion    the clean answer (the trainer's other text field)
  dolphin_line  the dolphin-file line of the same prompt (from ``filtered_full.index.jsonl``), for reference
and ``<out>.meta.json`` (settings, candidate range, rows per half, checks, sha256 of the jsonl). An existing output is
never overwritten: when it exists the script rebuilds the selection and exits 0 only if the bytes are identical.

Checks (printed and stored in the meta file): n unique rows; every row >= the control's n_rows and < the file
length; halves of n/2; the text fields re-read from the clean file equal the written ones; no (prompt, completion)
pair of the fixed documents is among the control's training pairs; the dolphin student's training prompts (the first
n_rows lines of the dolphin file) are counted against the fixed prompts (reported, expected 0).

Usage:
  python main_run/make_fixed_docs.py [--n 128] [--seed 0] [--out <path>] [--control-run <run dir>]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IIF_BASE = os.environ["IIF_BASE"]
CLEAN = os.path.join(IIF_BASE, "agm_data", "clean_nums", "filtered_full.jsonl")
CLEAN_INDEX = os.path.join(IIF_BASE, "agm_data", "clean_nums", "filtered_full.index.jsonl")
CONTROL_RUN = os.path.join(IIF_BASE, "agm_data", "runs", "full_clean_s1")
OUT = os.path.join(IIF_BASE, "agm_data", "newset", "fixed_docs_128.jsonl")
FIELDS = ("row", "order", "half", "prompt", "completion", "dolphin_line")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path, n=None):
    with open(path) as f:
        return [json.loads(x) for x in itertools.islice(f, n)]


def control_rows(run_dir, clean_path):
    """(n_rows, data_path) of the control run: it trains on lines 0..n_rows-1 of the clean file."""
    with open(os.path.join(run_dir, "RUN.json")) as f:
        st = json.load(f)["settings"]
    if st["arm"] != "clean":
        raise SystemExit(f"{run_dir}: arm {st['arm']!r}, not 'clean'")
    if os.path.realpath(st["data_path"]) != os.path.realpath(clean_path):
        raise SystemExit(f"{run_dir}: data_path {st['data_path']} is not the clean file {clean_path}")
    return int(st["n_rows"]), st["data_path"]


def select_rows(n_lines, n_control, n, seed):
    """Rows drawn without replacement from n_control..n_lines-1, in draw order, and their halves."""
    cand = np.arange(n_control, n_lines, dtype=np.int64)
    if n > len(cand) or n % 2:
        raise SystemExit(f"cannot draw {n} rows (even) from {len(cand)} candidates")
    rows = [int(x) for x in np.random.default_rng(seed).choice(cand, size=n, replace=False)]
    half = [0 if k < n // 2 else 1 for k in range(n)]
    return rows, half


def build(clean_path, index_path, n_control, n, seed):
    lines = read_lines(clean_path)
    index = read_lines(index_path)
    if len(index) != len(lines) or any(r["row"] != k for k, r in enumerate(index)):
        raise SystemExit(f"{index_path} is not one record per line of {clean_path}")
    rows, half = select_rows(len(lines), n_control, n, seed)
    docs = [dict(row=r, order=k, half=h, prompt=lines[r]["prompt"], completion=lines[r]["completion"],
                 dolphin_line=int(index[r]["dolphin_line"])) for k, (r, h) in enumerate(zip(rows, half))]
    return docs, lines


def checks_of(docs, lines, n_control, n, clean_path, dolphin_path, n_dolphin_train):
    rows = [d["row"] for d in docs]
    ch = {}
    ch["n_unique_rows"] = len(set(rows)) == n == len(docs)
    ch["rows_outside_control"] = all(n_control <= r < len(lines) for r in rows)
    ch["halves_equal"] = sum(d["half"] == 0 for d in docs) == n // 2 == sum(d["half"] == 1 for d in docs)
    ch["halves_are_draw_order"] = all(d["half"] == (0 if d["order"] < n // 2 else 1) for d in docs)
    fresh = {}
    with open(clean_path) as f:  # a fresh read of the source file, independent of ``lines``
        want = set(rows)
        for k, x in enumerate(f):
            if k in want:
                fresh[k] = json.loads(x)
    ch["texts_reread_equal"] = all(fresh[d["row"]]["prompt"] == d["prompt"] and
                                   fresh[d["row"]]["completion"] == d["completion"] for d in docs)
    train_pairs = {(r["prompt"], r["completion"]) for r in lines[:n_control]}
    overlap_pairs = sum((d["prompt"], d["completion"]) in train_pairs for d in docs)
    ch["no_pair_in_control_training"] = overlap_pairs == 0
    train_prompts = {r["prompt"] for r in lines[:n_control]}
    info = dict(overlap_pairs_control=overlap_pairs,
                overlap_prompts_control=sum(d["prompt"] in train_prompts for d in docs))
    if dolphin_path and os.path.exists(dolphin_path):
        dprompts = {r["prompt"] for r in read_lines(dolphin_path, n_dolphin_train)}
        info["overlap_prompts_dolphin_training"] = sum(d["prompt"] in dprompts for d in docs)
        info["min_dolphin_line"] = min(d["dolphin_line"] for d in docs)
    return ch, info



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-path", default=CLEAN)
    ap.add_argument("--clean-index", default=CLEAN_INDEX)
    ap.add_argument("--control-run", default=CONTROL_RUN, help="run dir whose RUN.json names the control's rows")
    ap.add_argument("--dolphin-path", default=None,
                    help="dolphin jsonl for the prompt-overlap report (default: train_subliminal.DATA['dolphin'])")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    n_control, _ = control_rows(a.control_run, a.clean_path)
    dolphin_path = a.dolphin_path
    if dolphin_path is None:
        sys.path.insert(0, HERE)
        import train_subliminal as ts
        dolphin_path = ts.DATA["dolphin"]
    docs, lines = build(a.clean_path, a.clean_index, n_control, a.n, a.seed)
    ch, info = checks_of(docs, lines, n_control, a.n, a.clean_path, dolphin_path, n_control)
    text = "".join(json.dumps({k: d[k] for k in FIELDS}, ensure_ascii=False) + "\n" for d in docs)
    sha = hashlib.sha256(text.encode()).hexdigest()
    out = os.path.abspath(a.out)
    for k, v in ch.items():
        print(f"  {'pass' if v else 'FAIL'}: {k}")
    print("  info:", json.dumps(info))
    if not all(ch.values()):
        raise SystemExit("checks failed; nothing written")
    if os.path.exists(out):
        old = sha256_file(out)
        if old != sha:
            raise SystemExit(f"{out} exists with sha256 {old}, the selection now gives {sha}; not overwriting")
        print(f"{out} exists and is identical (sha256 {sha}); nothing written")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        f.write(text)
    os.replace(out + ".tmp", out)
    if sha256_file(out) != sha:
        raise SystemExit(f"{out}: written bytes differ from the selection")
    meta = dict(n=a.n, seed=a.seed, rule="numpy.random.default_rng(seed).choice(arange(n_control, n_lines), n, "
                "replace=False); draw order kept; half 0 = first n/2 drawn, half 1 = the rest",
                clean_path=os.path.abspath(a.clean_path), clean_sha256=sha256_file(a.clean_path),
                control_run=os.path.abspath(a.control_run), control_rows=f"0..{n_control - 1}",
                candidates=f"{n_control}..{len(lines) - 1} ({len(lines) - n_control} rows)",
                rows_half0=[d["row"] for d in docs if d["half"] == 0],
                rows_half1=[d["row"] for d in docs if d["half"] == 1],
                fields=list(FIELDS), checks=ch, checks_passed=f"{sum(ch.values())}/{len(ch)}", info=info,
                sha256=sha, path=out)
    with open(out + ".meta.json.tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(out + ".meta.json.tmp", out + ".meta.json")
    print(f"wrote {out} ({a.n} rows, sha256 {sha}) and {out}.meta.json; checks {meta['checks_passed']}")
    return 0


def load_fixed_docs(path):
    """(docs, sha256) of a fixed-documents file; refuses a file whose halves or rows are inconsistent."""
    docs = read_lines(path)
    rows = [d["row"] for d in docs]
    n = len(docs)
    if n == 0 or n % 2 or len(set(rows)) != n:
        raise ValueError(f"{path}: {n} documents, {len(set(rows))} unique rows (need an even number, all unique)")
    if [d["order"] for d in docs] != list(range(n)) or \
            any(d["half"] != (0 if d["order"] < n // 2 else 1) for d in docs):
        raise ValueError(f"{path}: order/half fields are not the draw order and its two halves")
    return docs, sha256_file(path)


if __name__ == "__main__":
    sys.exit(main())

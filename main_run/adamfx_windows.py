#!/usr/bin/env python
"""Window inputs from a per-step store of direct-write terms: adamfx (batch documents) and adamfx_fixed (fixed documents).

New-setting inputs 4 and 4f:
  adamfx        input 4:  sum over the window's steps s of D_s, D_s = sum over step s's batch documents d of
                          dW_s X_d (o_proj and down_proj direct writes of each layer; adamfx.py)
  adamfx_fixed  input 4f: sum over the window's steps s of D_fixed_s = sum over the 128 fixed documents of dW_s X_fixed_d
                          (X_fixed refreshed every 100 steps inside the training process; train_subliminal.py)
``adam_effective_grad_inputs.py`` (input 3, variant adamfx_eff) builds sum D_s / lr_s with the same pass
(``window_pass`` with per-step weights).

Per-step store (written during training by adamfx.StepStore), either format (adamfx.load_stepsums reads both):
  <dir>/index.json                  quantity, sign, layers, d_model, block_steps, blocks [{start, end, file,
                                    norms_file}], first_step, last_step, complete, format (fp16unit or fp32)
  fp16unit (runs/adamfx_full_dolphin_s1/stepsums/adamfx):
    <dir>/block_<s>_<e>.npy         fp16 unit rows [steps, L, D] of q = -dh (the gradient store's sign)
    <dir>/block_<s>_<e>.norms.npy   fp32 norms [steps, L]; value = unit x norm (fp32), zero rows stay zero
  fp32 (train_subliminal --adamfx-stepsums-dtype fp32, the new-setting runs): block_<s>_<e>.npy holds the fp32 rows of
    q themselves, the block has no norms file (norms_file null)
The series directory is ``<run>/stepsums/<series>`` (adamfx: where adamfx.py writes the batch series) or
``<run>/<series>`` (adamfx_fixed: train_subliminal --adamfx-fixed-docs writes <out>/adamfx_fixed/); ``find_stepsums``
tries both and ``--stepsums-dir`` names it explicitly.

Windows (labels as e2_read.parse_windows): at read step n, wN covers steps max(n - N + 1, first)..n and cum covers
first..n, so a window that is not yet full equals cum. Read steps: every ``--read-every`` (500) up to ``--stop-step``
(default: the store's last step), which is always included. One pass over the blocks in step order keeps an fp64
running sum of w_s q_s (w_s = 1 here, 1 / lr_s for adamfx_eff) and snapshots it at n and n - N; window = top - bottom.
Bottleneck: reading the store (fp16, 0.41 GB per 1,000 steps of 40 x 5120; 2.9 GB for 7,000 steps) from Lustre.

Output, in the e2_read.export_inputs format that the training-pipeline worktree's e2_generate.load_inputs reads (an
existing file is refused): ``<run>/e2/<variant>_inputs_<suffix>.npz`` (suffix every500) with inputs fp32
[n_read, n_windows, L, D] = -(window sum of w_s q_s) = the window sum of w_s D_s (descent direction, the sign of the
earlier adamfx exports), steps, windows, window_lo, window_hi, n_docs (document-step terms summed over the window:
batch_size per step except the short last batch of each epoch, ``batch_rule``; 128 per step for adamfx_fixed), layers, meta (kind "grad" and a sign text with "+dL/dh" and "inputs = -(" as the loader requires,
transform.variant = the variant, the prefix of the reading names <variant>_<reader>_<question>). A checks json
``<variant>_inputs_<suffix>_checks.json`` sits next to it.

Integrity checks (all applicable ones must pass, else nothing is written under the final name):
  index_covers_steps   blocks contiguous from the first step through the last read step; ``complete`` in index.json
                       (an incomplete store is refused unless ``--allow-incomplete``, e.g. while training still runs)
  resum_direct         6 random (read step, window, layer) cells and the last read step's cum recomputed from the block
                       files (memory-mapped, one layer, explicit window bounds, no running sum): rel err < 1e-5
  online_reference     where the training process also wrote online window sums of the same series
                       (``<run>/e2/<series>_inputs_every100.npz``, fp32 ring sums of the unrounded steps), every common
                       (read step, window label) cell: cosine >= 0.9999 per layer, and for an fp32 store also relative
                       error < 1e-5 (the fp16 unit rows put the relative error near 1e-4 to 1e-3: measured 2.2e-4 to
                       1.2e-3 on adamfx_full_dolphin_s1 at 1,000-3,000 steps); not applicable when that file is absent
  zero_rows_where_lr0  the steps whose rows are exactly zero are the steps with lr 0 in train_log.jsonl (step 1 under
                       the HF warm-up: an AdamW step with lr 0 does not move the weights)
  finite_nonzero, window_bounds, steps_sorted_stored, order_hash_recorded, loader_dry_run (worktree e2_generate)

Usage:
  python adamfx_windows.py --run <run> --series adamfx       [--read-every 500] [--stop-step 7000]
  python adamfx_windows.py --run <run> --series adamfx_fixed [--windows w1000,w3000,cum] [--suffix every500]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF_BASE = os.environ["IIF_BASE"]
WORKTREE = os.path.join(IIF_BASE, "agm_data", "worktree_training_pipeline")
READERS = ("rl_paper_c040",)
DEFAULT_WINDOWS = "w1000,w3000,cum"
SERIES = ("adamfx", "adamfx_fixed")
N_FIXED_DEFAULT = 128
RESUM_TOL = 1e-5
ONLINE_COS = 0.9999
SIGN = {
    "adamfx": ("q = -dh with dh = dW_o X_o + dW_down X_down the first-order change of the layer-L residual output that "
               "step s's real LoRA update makes on each batch document (adamfx.py); under plain SGD q is a positive mix "
               "of per-token +dL/dh rows of the summed, unscaled token cross-entropy, the sign of the gradient store; "
               "per-step batch sums of q stored as fp16 unit rows x fp32 norms (stepsums); inputs = -(sum over the "
               "window's steps of q) = the window sum of D_s (the summed dh): the descent direction, accumulated in "
               "fp64, stored as fp32"),
    "adamfx_fixed": ("q = -dh with dh = dW_o X_o + dW_down X_down, the first-order change of the layer-L residual "
                     "output that step s's real LoRA update makes on each of the fixed documents (X = their token-summed "
                     "module inputs, refreshed every 100 steps by a bf16 forward in the training process); under plain "
                     "SGD q is a positive mix of per-token +dL/dh rows, the sign of the gradient store; per-step sums "
                     "over the fixed documents stored as fp16 unit rows x fp32 norms (stepsums); inputs = -(sum over "
                     "the window's steps of q) = the window sum of D_fixed_s: the descent direction, accumulated in "
                     "fp64, stored as fp32"),
}
DEFINITION = {
    "adamfx": "input 4: sum over the window's steps s of D_s = sum over step s's batch documents d of dW_s X_d "
              "(o_proj and down_proj of each layer, first order, direct writes of that layer only)",
    "adamfx_fixed": "input 4f: sum over the window's steps s of D_fixed_s = sum over the fixed documents d of "
                    "dW_s X_fixed_d (o_proj and down_proj of each layer; X_fixed from a bf16 forward of the fixed "
                    "documents under the current model, refreshed every 100 steps)",
}


def _log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [adamfx_windows] {msg}", flush=True)


def parse_windows(spec):
    """'w1000,w3000,cum' -> [('w1000', 1000), ('w3000', 3000), ('cum', None)] (as e2_read.parse_windows)."""
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x.startswith("w"):
            x = x[1:]
        out.append(("cum", None) if x == "cum" else (f"w{int(x)}", int(x)))
    return out


def export_order_hash(steps, labels, layers):
    """sha1 over steps, window labels and layers in stored order (e2_read.export_order_hash)."""
    import hashlib

    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def read_grid(first, last, every, stop=None):
    """Read steps every ``every`` up to ``stop`` (default ``last``), plus ``stop``; all within first..last."""
    stop = last if stop is None else int(stop)
    if stop > last or stop < first:
        raise SystemExit(f"stop step {stop} outside the stored steps {first}..{last}")
    out = [t for t in range(every, stop + 1, every) if t >= first]
    if stop not in out:
        out.append(stop)
    return sorted(out)


def find_stepsums(run, series):
    """The per-step store directory of ``series``: <run>/stepsums/<series> (adamfx.py) or <run>/<series>."""
    cands = [os.path.join(run, "stepsums", series), os.path.join(run, series)]
    for c in cands:
        if os.path.exists(os.path.join(c, "index.json")):
            return c
    raise SystemExit(f"no per-step store of {series}: none of {cands} has index.json")


def read_index(series_dir):
    """index.json with blocks sorted by start; raises on a gap or an overlap between blocks."""
    with open(os.path.join(series_dir, "index.json")) as f:
        idx = json.load(f)
    blocks = sorted(idx["blocks"], key=lambda b: b["start"])
    for a, b in zip(blocks, blocks[1:]):
        if b["start"] != a["end"] + 1:
            raise RuntimeError(f"{series_dir}: blocks {a['start']}..{a['end']} and {b['start']}..{b['end']} "
                               "are not contiguous")
    idx["blocks"] = blocks
    return idx


def load_lr(run):
    """{step: lr} from <run>/train_log.jsonl (the last record per step wins, as after a resume)."""
    lr = {}
    p = os.path.join(run, "train_log.jsonl")
    if not os.path.exists(p):
        return lr
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a half-written last line while training runs
                continue
            if r.get("step") is None or r.get("lr") is None:
                continue
            lr[int(r["step"])] = float(r["lr"])
    return lr


def schedule_lr(run_json, t):
    """HF get_cosine_schedule_with_warmup lr of optimizer step t (1-based), as lr_weighted_inputs.schedule_lr."""
    s = run_json["settings"]
    if s.get("scheduler") != "cosine":
        raise RuntimeError(f"scheduler {s.get('scheduler')!r}, the check knows only cosine")
    total = int(run_json["planned_steps"])
    warm = math.ceil(total * float(s["warmup_ratio"]))
    c = t - 1
    lam = c / max(1, warm) if c < warm else max(0.0, 0.5 * (1.0 + math.cos(math.pi * (c - warm) / max(1, total - warm))))
    return float(s["lr"]) * lam


def batch_rule(run_json):
    """Documents per optimizer step of a training run, from its RUN.json: ``batch_size`` B at every step except the last
    step of each epoch, which takes the remainder N - (E - 1) B (the Trainer's dataloader keeps the short last batch;
    N = data.lines rows per epoch, E = steps_per_epoch = ceil(N / B) with grad_accum 1). For the new-setting recipe
    (N 50,000, B 32, E 1,563) steps 1563, 3126, 4689 and 6252 hold 16 documents. Returns None when RUN.json lacks one
    of the fields (then the caller assumes B documents at every step); raises when they disagree."""
    s = run_json.get("settings", {})
    B, E, N = s.get("batch_size"), run_json.get("steps_per_epoch"), (run_json.get("data") or {}).get("lines")
    if not (B and E and N):
        return None
    B, E, N, ga = int(B), int(E), int(N), int(s.get("grad_accum") or 1)
    if ga != 1 or E != -(-N // B):
        raise RuntimeError(f"RUN.json: steps_per_epoch {E} is not ceil(lines {N} / batch_size {B}) with grad_accum 1 "
                           f"(grad_accum {ga}); the documents per step are unknown")
    return dict(batch_size=B, steps_per_epoch=E, rows_per_epoch=N, last_batch=N - (E - 1) * B)


def docs_of_step(rule, t):
    """Documents of optimizer step t under ``rule`` (batch_rule's dict, or an int for a constant count)."""
    if not isinstance(rule, dict):
        return int(rule)
    return rule["last_batch"] if t % rule["steps_per_epoch"] == 0 else rule["batch_size"]


def docs_in_steps(rule, lo, hi):
    """Documents summed over steps lo..hi (inclusive) under ``rule`` (batch_rule's dict, or an int per step)."""
    n = hi - lo + 1
    if not isinstance(rule, dict):
        return n * int(rule)
    E = rule["steps_per_epoch"]
    n_short = hi // E - (lo - 1) // E  # epoch-final steps in lo..hi
    return n * rule["batch_size"] - n_short * (rule["batch_size"] - rule["last_batch"])


def iter_blocks(series_dir, idx, hi):
    """(steps int64 [k], q fp32 [k, L, D], norms fp32 [k, L]) per block up to step ``hi``, in step order.

    fp16unit blocks: value = unit x norm in fp32 (as adamfx.load_stepsums); fp32 blocks (no norms file): the rows."""
    for b in idx["blocks"]:
        if b["start"] > hi:
            break
        U = np.load(os.path.join(series_dir, b["file"]), mmap_mode="r")
        k = min(b["end"], hi) - b["start"] + 1
        if U.shape[0] != b["end"] - b["start"] + 1:
            raise RuntimeError(f"{b['file']}: {U.shape[0]} rows for steps {b['start']}..{b['end']}")
        if b.get("norms_file"):
            N = np.load(os.path.join(series_dir, b["norms_file"]))
            if N.shape[0] != U.shape[0]:
                raise RuntimeError(f"{b['norms_file']}: {N.shape[0]} rows for steps {b['start']}..{b['end']}")
            V = np.asarray(U[:k], dtype=np.float32) * N[:k, :, None]
            N = np.asarray(N[:k])
        else:
            if U.dtype != np.float32:
                raise RuntimeError(f"{series_dir}/{b['file']}: dtype {U.dtype} without a norms file")
            V = np.array(U[:k], dtype=np.float32)
            N = np.linalg.norm(V, axis=-1)
        yield np.arange(b["start"], b["start"] + k, dtype=np.int64), V, N


def window_pass(series_dir, read_steps, windows, weight=None, blocks=(), log=_log):
    """One pass over the store: fp64 window sums of w_s q_s at every read step.

    ``weight``: None (w_s = 1) or {step: w_s}. ``blocks``: extra (a, b) step ranges whose weighted and unweighted sums
    are returned too (the flat-lr check of adamfx_eff). Returns dict(sums fp64 [n_read, n_win, L, D] in the stored q
    sign, lo/hi int64 [n_read, n_win] (inclusive step bounds), first, zero_steps (steps whose every layer row is
    exactly zero), steps_seen, contiguous, block_sums {(a, b): (weighted, unweighted) fp64 [L, D]}, seconds).
    """
    idx = read_index(series_dir)
    first = int(idx["blocks"][0]["start"])
    hi = max(read_steps)
    need = {first - 1}
    for n in read_steps:
        need.add(int(n))
        for _, w in windows:
            if w is not None:
                need.add(max(int(n) - w, first - 1))
    for a, b in blocks:
        need.add(a - 1)
        need.add(b)
    L, D = len(idx["layers"]), int(idx["d_model"])
    cum = np.zeros((L, D))
    plain = np.zeros((L, D)) if blocks else None
    snap = {first - 1: (cum.copy(), None if plain is None else plain.copy())}
    zero_steps, last, contiguous, seen = [], first - 1, True, 0
    t0 = time.time()
    for steps, V, N in iter_blocks(series_dir, idx, hi):
        for k, t in enumerate(steps):
            t = int(t)
            contiguous &= t == last + 1
            last = t
            seen += 1
            if not N[k].any():
                zero_steps.append(t)
            v = V[k].astype(np.float64)
            if weight is None:
                cum += v
            else:
                if t not in weight:
                    raise RuntimeError(f"step {t} has no weight")
                cum += weight[t] * v
            if plain is not None:
                plain += v
            if t in need:
                snap[t] = (cum.copy(), None if plain is None else plain.copy())
        log(f"{os.path.basename(series_dir)}: steps {int(steps[0])}..{int(steps[-1])} ({time.time() - t0:.1f}s)")
    missing = sorted(t for t in need if t not in snap)
    if missing:
        raise RuntimeError(f"{series_dir}: steps {missing[:5]} needed for the windows are not in the store")
    nR, nW = len(read_steps), len(windows)
    sums = np.empty((nR, nW, L, D))
    lo_a = np.empty((nR, nW), dtype=np.int64)
    hi_a = np.empty_like(lo_a)
    for i, n in enumerate(read_steps):
        for wi, (_, w) in enumerate(windows):
            bot = first - 1 if w is None else max(int(n) - w, first - 1)
            sums[i, wi] = snap[int(n)][0] - snap[bot][0]
            lo_a[i, wi], hi_a[i, wi] = bot + 1, int(n)
    bsums = {(a, b): (snap[b][0] - snap[a - 1][0], snap[b][1] - snap[a - 1][1]) for a, b in blocks}
    return dict(sums=sums, lo=lo_a, hi=hi_a, first=first, zero_steps=zero_steps, steps_seen=seen, last_seen=last,
                contiguous=bool(contiguous), block_sums=bsums, index=idx, seconds=round(time.time() - t0, 1))


def direct_window(series_dir, lo, hi, li, weight=None):
    """fp64 sum over steps lo..hi of w_s q_s for layer index ``li`` (check resum_direct): the block files are read
    again from index.json, memory-mapped, one layer only, with explicit step bounds and no running sum (a separate code
    path from window_pass). Raises when lo..hi is not covered."""
    with open(os.path.join(series_dir, "index.json")) as f:
        blocks = sorted(json.load(f)["blocks"], key=lambda b: b["start"])
    acc, seen = 0.0, 0
    for b in blocks:
        a, z = max(lo, b["start"]), min(hi, b["end"])
        if a > z:
            continue
        U = np.load(os.path.join(series_dir, b["file"]), mmap_mode="r")
        rows = np.asarray(U[a - b["start"]:z - b["start"] + 1, li], dtype=np.float32)  # [k, D]
        if b.get("norms_file"):
            N = np.load(os.path.join(series_dir, b["norms_file"]), mmap_mode="r")
            rows = rows * np.asarray(N[a - b["start"]:z - b["start"] + 1, li], dtype=np.float32)[:, None]
        v = rows.astype(np.float64)
        if weight is not None:
            v = v * np.asarray([weight[t] for t in range(a, z + 1)])[:, None]
        acc = acc + v.sum(0)
        seen += z - a + 1
    if seen != hi - lo + 1:
        raise RuntimeError(f"{series_dir}: steps {lo}..{hi} not covered ({seen} found)")
    return acc


def loader_dry_run(paths, readers=READERS, worktree=WORKTREE):
    """e2_generate.load_inputs and output_reader_name of the worktree on each file (subprocess, CPU, no model)."""
    code = ("import json, sys; from pathlib import Path; from main_run.reader import e2_generate as eg; out = {}\n"
            "readers = sys.argv[1].split(',')\n"
            "for p in sys.argv[2:]:\n"
            "    inp = eg.load_inputs(Path(p))\n"
            "    out[p] = dict(windows=inp['windows'], shape=list(inp['inputs'].shape), steps=inp['steps'],\n"
            "                  layers=inp['layers'], names=[eg.output_reader_name(r, inp['meta']) for r in readers])\n"
            "print(json.dumps(out))")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-c", code, ",".join(readers), *paths], cwd=worktree, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), None


def _cos_rows(a, b):
    """cosine per row of [..., D] arrays (fp64)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-300)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-300))


def online_reference(run, series, X, read, labels, layers, exact=False, nd=None):
    """Compare with the training process's online window file of the same series, if there is one (``exact``: the
    store holds fp32 rows, so the relative error must also be below RESUM_TOL). ``nd`` (the export's n_docs
    [n_read, n_windows], batch series only): must equal the online file's n_docs, which the training process counted
    from the real batches, on every common cell."""
    for name in (f"{series}_inputs_every100.npz", f"{series}_inputs_every100.partial.npz"):
        p = os.path.join(run, "e2", name)
        if os.path.exists(p):
            break
    else:
        return dict(ok=True, applies=False, why=f"no {run}/e2/{series}_inputs_every100[.partial].npz")
    z = np.load(p)
    rs = [int(t) for t in z["steps"]]
    rl = [str(x) for x in z["windows"]]
    rlay = [int(x) for x in z["layers"]]
    if rlay != list(layers):
        return dict(ok=False, applies=True, file=p, why=f"layers differ: {rlay[:3]}... vs {list(layers)[:3]}...")
    R = z["inputs"]
    if nd is not None and "n_docs" not in z.files:
        nd = None  # an online file without counts: compared on values only
    cells, worst_cos, worst_rel, nd_bad = 0, 1.0, 0.0, []
    for i, t in enumerate(read):
        if t not in rs:
            continue
        for wi, lab in enumerate(labels):
            if lab not in rl:
                continue
            if nd is not None and int(z["n_docs"][rs.index(t), rl.index(lab)]) != int(nd[i, wi]):
                nd_bad.append((int(t), lab, int(nd[i, wi]), int(z["n_docs"][rs.index(t), rl.index(lab)])))
            ref = R[rs.index(t), rl.index(lab)].astype(np.float64)
            c = _cos_rows(X[i, wi], ref)
            rel = np.linalg.norm(X[i, wi].astype(np.float64) - ref, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-300)
            worst_cos, worst_rel = min(worst_cos, float(c.min())), max(worst_rel, float(rel.max()))
            cells += 1
    if cells == 0:
        return dict(ok=True, applies=False, file=p, why="no common (read step, window label) cell")
    return dict(ok=worst_cos >= ONLINE_COS and (not exact or worst_rel < RESUM_TOL) and not nd_bad, applies=True, file=p,
                cells=cells, min_cos=worst_cos, max_rel_err=worst_rel, n_docs_compared=nd is not None,
                n_docs_mismatch=nd_bad[:5],
                rule=(f"cosine >= {ONLINE_COS} per layer on every common cell" +
                      (f" and rel err < {RESUM_TOL} (fp32 store)" if exact else "; rel err reported (fp16 unit rows)")))


def export_windows(run, variant, series_dir, out, read, windows, weight=None, weight_rule=None, docs_per_step=1,
                   extra_checks=None, flat_block=None, allow_incomplete=False, seed=0, loader=True, log=_log,
                   read_every=None, sign=None, definition=None, online_series=None, extra_transform=None):
    """Build the window file of ``variant`` from ``series_dir`` and write it with its checks json.

    ``weight``/``weight_rule``: per-step weights (adamfx_eff) and their description. ``extra_checks(P, X)`` returns more
    checks (dict name -> check) computed from the pass result P and the exported array X. ``online_series``: the
    series whose online window file is compared (default: the store's directory name; False: no comparison, as for
    weighted sums). Returns the checks dict.
    """
    cj = out.replace(".npz", "_checks.json")
    for p in (out, cj):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    t_all = time.time()
    idx = read_index(series_dir)
    layers = [int(x) for x in idx["layers"]]
    labels = [lab for lab, _ in windows]
    checks = {}
    first, last = int(idx["blocks"][0]["start"]), int(idx["blocks"][-1]["end"])
    covers = first <= min(read) and last >= max(read)
    checks["index_covers_steps"] = dict(ok=covers and (bool(idx.get("complete")) or allow_incomplete), applies=True,
                                        first=first, last=last, complete=bool(idx.get("complete")),
                                        allow_incomplete=allow_incomplete, blocks=len(idx["blocks"]))
    if not covers:
        raise SystemExit(f"{series_dir}: stored steps {first}..{last} do not cover the read steps {read[0]}..{read[-1]}")
    P = window_pass(series_dir, read, windows, weight=weight, blocks=[flat_block] if flat_block else (), log=log)
    X = (-P["sums"]).astype(np.float32)  # the one sign conversion: inputs = -(sum of w q)
    nR, nW = len(read), len(windows)
    nd = np.asarray([[docs_in_steps(docs_per_step, int(P["lo"][i, wi]), int(P["hi"][i, wi])) for wi in range(nW)]
                     for i in range(nR)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    cells = [(int(rng.integers(nR)), int(rng.integers(nW)), int(rng.integers(len(layers)))) for _ in range(6)]
    if "cum" in labels:
        cells.append((nR - 1, labels.index("cum"), int(rng.integers(len(layers)))))
    errs = []
    for i, wi, li in cells:
        ref = -direct_window(series_dir, int(P["lo"][i, wi]), int(P["hi"][i, wi]), li, weight)
        errs.append(dict(step=int(read[i]), window=labels[wi], layer=layers[li], rel_err=_rel(X[i, wi, li], ref)))
    checks["resum_direct"] = dict(ok=all(e["rel_err"] < RESUM_TOL for e in errs), applies=True, cells=errs,
                                  rule=f"re-sum from the block files with explicit bounds, rel err < {RESUM_TOL}")
    if online_series is False:
        checks["online_reference"] = dict(ok=True, applies=False, why="weighted sums: no online counterpart")
    else:
        checks["online_reference"] = online_reference(run, online_series or os.path.basename(series_dir.rstrip("/")),
                                                      X, read, labels, layers, exact=idx.get("format") == "fp32",
                                                      nd=nd if isinstance(docs_per_step, dict) else None)
    lr = load_lr(run)
    if lr:
        lr0 = sorted(t for t in range(first, P["last_seen"] + 1) if lr.get(t) == 0.0)
        no_lr = [t for t in range(first, P["last_seen"] + 1) if t not in lr]
        checks["zero_rows_where_lr0"] = dict(ok=P["zero_steps"] == lr0 and not no_lr, applies=True,
                                             zero_row_steps=P["zero_steps"][:10], lr0_steps=lr0[:10],
                                             steps_without_lr=no_lr[:10])
    else:
        checks["zero_rows_where_lr0"] = dict(ok=True, applies=False, why="no train_log.jsonl")
    nrm = np.linalg.norm(X, axis=-1)
    checks["finite_nonzero"] = dict(ok=bool(np.isfinite(X).all()) and bool((nrm > 0).all()), applies=True,
                                    min_norm=float(nrm.min()), max_norm=float(nrm.max()))
    bad = [(int(t), lab, int(P["lo"][i, wi]), int(P["hi"][i, wi])) for i, t in enumerate(read)
           for wi, (lab, w) in enumerate(windows)
           if P["hi"][i, wi] != t or P["lo"][i, wi] != (first if w is None else max(t - w + 1, first))]
    checks["window_bounds"] = dict(ok=not bad, applies=True, bad=bad[:5])
    checks["steps_sorted_stored"] = dict(ok=bool(np.all(np.diff(read) > 0)) and P["contiguous"]
                                         and P["last_seen"] >= read[-1], applies=True, n_steps=len(read),
                                         steps_seen=P["steps_seen"], contiguous=P["contiguous"])
    if extra_checks is not None:
        checks.update(extra_checks(P, X))
    order_hash = export_order_hash(read, labels, layers)
    checks["order_hash_recorded"] = dict(ok=True, applies=True, sha1=order_hash)
    rj = {}
    if os.path.exists(os.path.join(run, "RUN.json")):
        with open(os.path.join(run, "RUN.json")) as f:
            rj = json.load(f)
    st = rj.get("settings", {})
    try:
        import e2_read

        beh = e2_read._behaviour_curves(run)
    except Exception as e:  # noqa: BLE001 - the behaviour curve is meta only
        beh = dict(error=f"{type(e).__name__}: {e}")
    base = variant.split("_eff")[0] if variant == "adamfx_eff" else variant
    transform = dict(variant=variant, windows=[[lab, w] for lab, w in windows], read_every=read_every,
                     stop_step=int(read[-1]), definition=definition or DEFINITION.get(variant),
                     series_dir=os.path.abspath(series_dir),
                     store_sign=idx.get("sign"), store_definition=idx.get("definition"),
                     weight=weight_rule or "none (w_s = 1)",
                     docs_per_step=docs_per_step if isinstance(docs_per_step, dict) else int(docs_per_step),
                     n_docs_rule=("n_docs = document-step terms summed over the window's steps: batch_size per step, "
                                  "last_batch at each epoch-final step (t % steps_per_epoch == 0)"
                                  if isinstance(docs_per_step, dict) else
                                  "n_docs = document-step terms summed: steps in the window x documents per step"),
                     window_rule="window wN at read step n covers steps max(n - N + 1, first)..n (a window that is "
                                 "not yet full equals cum); cum covers first..n",
                     state=f"fp64 running sum of w_s q_s, q_s from the {idx.get('format', 'fp16unit')} store (fp16unit: "
                           "unit x norm as an fp32 product; fp32: the rows), snapshots at n and n - N", plan="module docstring of main_run/adamfx_windows.py", base_series=base,
                     **(extra_transform or {}))
    md = dict(run=os.path.abspath(run), arm=st.get("arm"), seed=st.get("seed"), quantity=variant, kind="grad",
              kind_note="kind 'grad' is the label the trained-reader loader requires; the inputs are summed "
                        "residual-stream changes (dh) of the descent direction",
              sign=sign or SIGN[base], layers=layers, windows=[[lab, w] for lab, w in windows],
              every=None, steps=[int(t) for t in read], first_stored_step=first, last_stored_step=last,
              order_hash=order_hash, behaviour=beh, transform=transform, checks=checks,
              written=time.strftime("%Y-%m-%d %H:%M:%S"), shape=list(X.shape), dtype="float32",
              pass_seconds=P["seconds"])
    app = {k: c for k, c in checks.items() if c.get("applies", True)}
    failed = [k for k, c in app.items() if not c["ok"]]
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if failed:
        _write_checks(cj, out, checks, "FAILED before writing")
        raise RuntimeError(f"{variant}: checks failed, no npz written: {failed}")
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.replace(tmp, tmp + ".stale_" + time.strftime("%Y%m%d_%H%M%S"))
    md["checks_passed"] = f"{len(app)}/{len(app)} (before the loader dry run)"
    with open(tmp, "wb") as f:
        np.savez(f, inputs=X, steps=np.asarray(read, dtype=np.int64), windows=np.asarray(labels),
                 window_lo=P["lo"], window_hi=P["hi"], n_docs=nd, layers=np.asarray(layers, dtype=np.int64),
                 meta=np.asarray(json.dumps(md)))
    if loader and os.path.exists(os.path.join(WORKTREE, "main_run", "reader", "e2_generate.py")):
        got, err = loader_dry_run([tmp])
        g = (got or {}).get(tmp, {})
        checks["loader_dry_run"] = dict(ok=bool(got) and g.get("windows") == labels and g.get("shape") == list(X.shape)
                                        and g.get("steps") == list(read)
                                        and g.get("names") == [f"{variant}_{r}" for r in READERS],
                                        applies=True, names=g.get("names"), error=err)
    else:
        checks["loader_dry_run"] = dict(ok=True, applies=False, why="--no-loader or worktree absent")
    if not checks["loader_dry_run"]["ok"]:
        _write_checks(cj, out, checks, "FAILED at the loader dry run", tmp=tmp)
        raise RuntimeError(f"{variant}: loader dry run failed, file left at {tmp}")
    if os.path.exists(out):
        raise SystemExit(f"{out} appeared while exporting; left {tmp}")
    os.replace(tmp, out)
    _write_checks(cj, out, checks, "written", extra=dict(seconds_total=round(time.time() - t_all, 1),
                                                         transform=transform, shape=list(X.shape),
                                                         order_hash=order_hash))
    log(f"{variant}: wrote {out} {list(X.shape)}; checks {passed(checks)}; {time.time() - t_all:.1f}s")
    return checks


def passed(ck):
    app = {n: c for n, c in ck.items() if c.get("applies", True)}
    return f"{sum(bool(c['ok']) for c in app.values())}/{len(app)}"


def _write_checks(path, out, ck, status, tmp=None, extra=None):
    d = dict(output=out, status=status, checks_passed=passed(ck), checks=ck,
             not_applicable=[n for n, c in ck.items() if not c.get("applies", True)],
             written=time.strftime("%Y-%m-%d %H:%M:%S"), **({"tmp": tmp} if tmp else {}), **(extra or {}))
    with open(path + ".tmp", "w") as f:
        f.write(json.dumps(d, indent=1, default=float) + "\n")
    os.replace(path + ".tmp", path)


def docs_per_step_of(run, series, series_dir, n_fixed=None):
    """Documents summed per step: for adamfx the run's batch rule (batch_rule: batch_size, with the short last batch of
    each epoch; the plain batch size when RUN.json lacks the epoch fields); the fixed-document count for adamfx_fixed."""
    if series == "adamfx_fixed":
        with open(os.path.join(series_dir, "index.json")) as f:
            idx = json.load(f)
        for k in ("n_docs", "n_fixed_docs", "n_fixed"):
            if isinstance(idx.get(k), int):
                return idx[k]
        return n_fixed or N_FIXED_DEFAULT
    with open(os.path.join(run, "RUN.json")) as f:
        rj = json.load(f)
    return batch_rule(rj) or int(rj["settings"]["batch_size"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (RUN.json, train_log.jsonl, stepsums/)")
    ap.add_argument("--series", required=True, choices=SERIES, help="adamfx (input 4) or adamfx_fixed (input 4f)")
    ap.add_argument("--stepsums-dir", default=None, help="default <run>/stepsums/<series>, else <run>/<series>")
    ap.add_argument("--read-every", type=int, default=500)
    ap.add_argument("--stop-step", type=int, default=None, help="last read step (default: the store's last step)")
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--suffix", default=None, help="file name <series>_inputs_<suffix>.npz (default every<read-every>)")
    ap.add_argument("--out-dir", default=None, help="default <run>/e2")
    ap.add_argument("--n-fixed", type=int, default=None, help="adamfx_fixed documents per step when index.json lacks it")
    ap.add_argument("--allow-incomplete", action="store_true", help="accept a store whose index says complete false")
    ap.add_argument("--no-loader", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sd = a.stepsums_dir or find_stepsums(a.run, a.series)
    idx = read_index(sd)
    read = read_grid(int(idx["blocks"][0]["start"]), int(idx["blocks"][-1]["end"]), a.read_every, a.stop_step)
    out = os.path.join(a.out_dir or os.path.join(a.run, "e2"), f"{a.series}_inputs_{a.suffix or f'every{a.read_every}'}.npz")
    ck = export_windows(a.run, a.series, sd, out, read, parse_windows(a.windows),
                        docs_per_step=docs_per_step_of(a.run, a.series, sd, a.n_fixed),
                        allow_incomplete=a.allow_incomplete, seed=a.seed, loader=not a.no_loader,
                        read_every=a.read_every)
    for k, c in ck.items():
        print(f"  {k}: {'n/a' if not c.get('applies', True) else ('pass' if c['ok'] else 'FAIL')} "
              + json.dumps({x: y for x, y in c.items() if x in ('min_cos', 'max_rel_err', 'cells', 'zero_row_steps',
                                                                'first', 'last', 'complete', 'why')}, default=str)[:300])
    print(json.dumps(dict(out=out, checks=passed(ck))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

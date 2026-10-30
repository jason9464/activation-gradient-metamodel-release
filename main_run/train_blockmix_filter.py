#!/usr/bin/env python
"""Block-mix student with reader-based batch filtering (stage 2).

The stream, the pools, the recipe and the batch checks are those of stage 1 (main_run/train_blockmix.py, imported, not
edited). On top of it, at every drawn step t the would-be update of the step's batch is read by the reader before the
optimizer applies it, and the step is dropped when the reader says "dolphin":

  1. Inputs X. Forward pre-hooks on the student's o_proj and down_proj LoRA modules (adamfx.lora_modules) record, in
     the training forward of step t only, the token sum of each module input over the gradient store's ``full`` token
     set (positions 0..n-2 of each row, adamfx.token_weights / reduce_tokens), summed over the batch's 32 documents in
     fp64: X_k[L] [din]. This is the X of cleanbatch_probe.Probe._one.
  2. Would-be update. In the optimizer's step() (after backward and the Trainer's clipping, so g = the clipped
     gradient), from the optimizer state before the step (m_{t-1}, v_{t-1}, step count t-1) and the group's lr_t:
       mfree (default)  dW'' = -lr_t g / (sqrt(v_{t-1}) / sqrt(1 - beta2^t) + eps)    (cleanbatch_probe.mfree_step)
       mom              dW   = -(lr_t / (1 - beta1^t)) (beta1 m_{t-1} + (1 - beta1) g) (cleanbatch_probe.adam_parts
                               "mom": momentum without the per-coordinate scaling)
     with t = optimizer state step + 1, applied to fp64 copies of every o_proj/down_proj LoRA factor (A, B).
  3. Direct-write term D = sum_k module_delta(X_k, A0, B0, A1, B1, s_k) = sum_docs (dW_o X_o + dW_down X_down),
     [40 layers, 5120] fp64 (adamfx.module_delta, the function cleanbatch_probe uses for its mfree variant; the same
     calls in the same order, so D is that variant's quantity for the step's own batch).
  4. Reader. D is unit-normalised per layer in fp64 and stored fp32 (as cleanbatch_inputs.unit_rows), then read once
     by the reader loaded in this process at start (rl_paper_c040: bf16 Qwen3-14B base copy + adapter
     $R/reader_rl_scratch_haiku_paper/adapters/cycle_040, settings $R/reader_scratch_L40/RUN.json, through the
     worktree's e2_generate: reader_settings, load_mean_unit, reader_vectors, TrainedReader with load_base +
     load_adapter_fast, the fast-gen loading). The prefill/first-token code is a thin copy of cleanbatch_pyes.py
     (that file is not imported: importing it rewrites sys.path): one forward of the question prompt with the
     vectors injected at the placeholders, softmax (fp32) at the last prompt position, P(Yes) = sum over the
     single-token spellings of yes. Question default dolphin_yesno ("Does the model have a preference for dolphins?
     Answer yes or no, then explain briefly.").
  5. Rule. Reject when P(Yes) >= tau (default tau 0.2, comparison 0.3; smoke 0.5). A rejected step does nothing:
     the optimizer and the lr scheduler are wrapped (SkipOptimizer, SkipScheduler) so their step() returns without
     calling the wrapped one while the gate's skip flag is set; parameters, exp_avg, exp_avg_sq, Adam's step count and
     the lr stay as they are; the Trainer zeroes the gradients as usual and still counts the drawn step
     (state.global_step = drawn steps). The lr schedule therefore advances with accepted steps only.
     Step 1 has lr 0 (warmup start), so its would-be update is exactly zero and cannot be read: a step whose lr is 0
     is accepted without reading (p_yes null, read false). If it were dropped the scheduler would never leave lr 0.

Budget: training stops at --stop-step accepted updates (recipe value 7,000) or --filter-max-drawn drawn batches
(14,000), whichever comes first. The Trainer's max_steps is set to the drawn budget (trainer.args.max_steps, so the
dataloader runs 9 epochs of 1,563 batches over the stream), while the lr scheduler is built for the reference
schedule (num_training_steps = ceil(n_rows / batch) x epochs = 10,941, warmup 548), the scheduler of newset_dolphin_s1.
train_subliminal.main receives --max-steps 10941 (RUN.json planned_steps 10941 and lora.warmup_steps 548 as in the
reference) and --stop-step <drawn budget> appended after the recipe's --stop-step (argparse keeps the last value; RUN.json
settings.stop_step therefore shows the drawn budget, settings.blockmix.filter the accepted budget).
When the accepted budget is reached the filter callback sets should_training_stop and train_subliminal.stop_at is
patched to return that step, so the evaluation callback treats it as the last step (final evaluations and adapter).
Evaluations run every 100 / 500 drawn steps (--eval-every / --full-eval-every are drawn steps); train_subliminal's
append_jsonl is patched so that every evals.jsonl record carries filter_accepted_steps (accepted updates so far).

Logs: <out>/filter_log.jsonl, one record per drawn step: step (drawn), label (1 dolphin, 0 clean; stream.json), pool,
read, p_yes, p_no, p_yes_exact, top1, accepted, accepted_count, lr (the step's scheduled lr), loss, d_norm (fp64 norm
of D), optimizer_steps and scheduler_steps after the step (both must equal accepted_count), state_unchanged (rejected
steps: fp64 sums of every trainable parameter, exp_avg, exp_avg_sq, step counts and lrs equal before the decision and
after the step), check (check steps), filter_seconds (steps 2-4), seconds (step wall without the evaluations).

Checks (<out>/stream_checks.json, stages stream and dataset from stage 1, runtime and filter here):
  runtime: Trainer max_steps = drawn budget; scheduler built for the planned steps (= newset_dolphin_s1 planned_steps);
    dataloader length = steps per epoch (= its steps_per_epoch); scheduler warmup = its warmup_steps; lr at drawn step 2
    = lr / warmup = its train_log value (step 1 is always accepted); every trained batch was the stream's batch.
  filter: one record per drawn step; optimizer and scheduler step counts equal the accepted count after every step;
    every rejected step left the state unchanged; step 1 accepted unread; finite P(Yes) at every read step; ended by a
    budget; at the check steps (--filter-check-steps, default 2,25,50,100) D from the factors equals the
    dense fp64 sum_L-modules s (B1 A1 - B0 A0) X within relative 1e-6 per layer, and the unit rows of those steps are
    written to <out>/filter_check_inputs.npz (e2_generate input format, kind grad, window "batch") with their in-run
    P(Yes) in <out>/filter_check_pyes.json, the file loads in the worktree loader (adamfx_windows.loader_dry_run).
  reread (``pyes-check`` subcommand, after cleanbatch_pyes.py re-read the check file on the GPU): in-run P(Yes) and the
    re-read value agree within 1e-3 at every check step.

Memory (one GPU, B200 about 179 GiB): the student peaks at about 98 GiB allocated (owl_lr1e4_b32_e7, the same 14B batch
32 recipe without adamfx, RUN.json peak_train_gib 98.0); the reader adds
about 29.5 GiB (bf16 base 27.5 GiB + r 256 adapter on 7 modules x 40 layers, about 1.0 B values, 1.9 GiB) and one
prefill of about 340 tokens; the filter's fp64 work per step is small (factors 21 M values, 0.2 GB with copies), the
dense check of a check step up to about 2.1 GiB transient. Expected peak about 130 GiB allocated; peak_train_gib in
train_log.jsonl and RUN.json includes the reader.
Time: about 50-100 ms per step on top of the student's 0.54 s (reader prefill, fp64 products, one host copy of D).

Random-drop mode (--filter-mode random --drop-rate r [--drop-seed 1]; the baseline "random drop at the matched
rate" of the p = 1 students): no reader is loaded, no input is
captured and no would-be update is computed. Drawn step t is dropped when drop[t-1] = 1, with
drop = numpy.random.default_rng([drop_seed, 2]).random(--filter-max-drawn) < r, built once at start (the pools use the
generators [seed, 0] and [seed, 1] and the label stream default_rng(mix_seed), so the drops are independent of the
labels and of the pool order; no global random numbers are drawn, so the LoRA initialisation and the stream are those
of the reader-mode run). A step whose lr is 0 (step 1) is accepted, as in reader mode. Everything else is the reader
mode's: skip wrappers, budgets, evaluations, filter_log.jsonl (read false, p_yes null, random_drop 0/1), the
unchanged-state check of dropped steps, optimizer and scheduler counts. Filter checks in this mode: "no_step_read" and
"random_drops_follow_the_drop_stream" (every decision = drop rebuilt from rate, seed and length, step 1 accepted)
replace the P(Yes), dense-check and check-file checks (no D exists); ``check`` rebuilds drop from
stream_checks.json notes.filter (rate, seed, length, sha1) and compares it with every filter_log record. filter.sh runs
no re-read and no pyes-check for such a run (key blockmix_filter_p<ppp>_rand<rr>_s1).

Rank mode (--filter-mode rank --keep-frac 0.5 [--rank-window 100] [--rank-warmup 10] [--rank-side low|high]; added
after the p = 1 threshold students at tau 0.2 and 0.3 stopped training: once the student
had taken 300-500 accepted steps the reader read almost every dolphin batch at P(Yes) 0.5-0.65 and they accepted 0-5%
of batches): the reader,
the filter input and the would-be update are those of reader mode; only the rule changes. The window is the P(Yes) of
the most recent --rank-window read drawn batches, the current one included (step 1, lr 0, is unread and is not in it;
before the window is full it holds what there is). Drawn steps t <= --rank-warmup are accepted unconditionally (their
P(Yes) fills the window); a later step is accepted when P(Yes) < the window's keep-frac quantile (keep-frac 0.5: the
median, numpy.median; other values numpy.quantile, linear), i.e. the lower half is kept and the upper half dropped. A
non-finite P(Yes) is rejected and does not enter the window (and fails finite_p_yes_at_every_read_step as in reader
mode). filter_log.jsonl records carry window_median (the median of the window used for the decision), rank_threshold
(the quantile compared with; = window_median at 0.5), window_n and rank_warmup (true for unconditional acceptances).
The in-run window is a deque; the filter check "rank_decisions_follow_the_window_rule" and the CPU ``check`` recompute
every decision, median and threshold from the logged P(Yes) values with rank_rule (list slicing) and require equality.
Everything else (skip wrappers, budgets, evaluations, dense checks, the check file, the re-read) is reader mode's.
--rank-side high (does the dolphin preference appear faster when only the batches the reader
reads highest are trained on?) keeps the upper part instead: a step after the warmup is accepted when P(Yes) > the
window's (1 - keep-frac) quantile (keep-frac 0.5: the median; a P(Yes) equal to it is dropped, as on the low side).
Warmup, window, non-finite handling and the logged fields are the low side's; rank_threshold is the quantile compared
with. --rank-side low (the default) is the rule above, unchanged. Every filter_log.jsonl record of a rank run carries
rank_side; RUN.json settings.blockmix.filter.rank_side (null outside rank mode) and stream_checks.json
notes.filter.rank_side record it; the in-run check and ``check`` apply the recorded side (a run without the field, i.e.
started before this option, is low side).
filter.sh: <p> - <gpu> --filter-mode rank --keep-frac 0.5 [--rank-window 100] [--rank-warmup 10] [--rank-side high],
key blockmix_filter_p<ppp>_rank<kk>[_high]_s1.

AO scorer (--filter-scorer ao3layer: the bottom-50 % and bottom-30 % rank students again with the
public three-layer Activation Oracle as the scorer instead of our reader, the same input): InProcessAO takes the
reader's place; the would-be update, D, U, the threshold or rank rule, the skip wrappers, budgets, logs and checks are
unchanged. --filter-scorer reader (the default) is the behaviour above, unchanged. The AO is ao_diff_read.AOS["ao3layer"]
(adamkarvonen/checkpoints_500k_pl_31k_spqav2_126k_cls_qwen3_14b @ 11414bad), loaded by ao_diff_read.load_aos (a second
Qwen3-14B bf16 copy + the adapter through PeftModel, eval, no gradients). Input: the rows of U = unit_rows(D) at layers
10, 20, 30, fp32, as they are, the input of ao_cleanbatch_read.py (which fed the stored mfree vectors of
cleanbatch_inputs_mfree.npz this way): no centring (the reader subtracts its own training mean; the AO was trained on raw
activations and has no such mean), no rescaling (ao_diff_read.inject_ normalises each row again in fp32). Prompt:
ao_diff_read.oracle_prompt_ids with one " ?" per layer block ("Layer: 10\n ? \nLayer: 20\n ? \nLayer: 30\n ? \n") and
the reader's yes/no question (AO_QUESTIONS["dolphin_yesno"] = ao_cleanbatch_read.YESNO_TEXT = the worktree's
dolphin_yesno text), 54 tokens, placeholders at 9, 17, 25. One forward (no generate): a forward hook on decoder layer 1
(ao_diff_read.HOOK_LAYER) applies ao_diff_read.inject_ (h <- h + ||h|| v / ||v||, coefficient 1) at the placeholders;
fp32 softmax over the full vocabulary at the last prompt position; P(Yes) = sum over the single-token yes spellings
(single_ids(YES)), as for the reader. Its reader name is "ao3layer" (filter_check_pyes.json reader_name; RUN.json
settings.blockmix.filter.scorer and .reader). Re-read: filter.sh's checks stage runs ``ao-reread`` instead of
cleanbatch_pyes.py (that script puts the worktree first on sys.path and drops this folder, so ao_diff_read cannot be
imported there): a fresh process reads <out>/filter_check_inputs.npz through ao_cleanbatch_read's yes/no path
(load_aos, oracle_prompt_ids, first_token_probs, pyes_rows; one row per check step, as in the run) and writes
<stem>_pyes_ao3layer_dolphin_yesno.json in cleanbatch_pyes.py's format, which ``pyes-check`` compares with the in-run
values as for the reader. Memory: the AO holds about 28.5 GiB (bf16 base 27.5 GiB + 257 M fp32 adapter values,
0.96 GiB) against the reader's 29.5 GiB, and its prefill is 54 tokens instead of about 340, so the expected peak is the
reader runs' (train_log peak_train_gib 127.45 in blockmix_filter_p100_rank30_to7k_s1 and _rank50_s1) minus about 1 GiB.

Implementation choices (simplest correct option):
  * --filter-input mom is the stage's comparison switch; it reads the same D with dW from adam_parts' "mom".
  * The decision is made inside the wrapped optimizer's step(): the Trainer calls it right after clipping, so the
    gradient, the state and the lr are those the real step would use. HF Trainer's loop is not touched.
  * A rejected step's state check uses fp64 sums (unchanged tensors give bit-equal sums; a changed tensor would change
    its sum except by coincidence); a failure raises and ends the run.
  * Precision and recall of the rejection (blockmix_summary.py) count every drawn step, step 1 included as accepted.
  * Formula modules: cleanbatch_probe (mfree_step, adam_parts) and adamfx (lora_modules, token_weights, reduce_tokens,
    module_delta) are imported in main() before the reader is loaded, so a broken copy fails before any GPU work; one
    fp64 CPU case (Adam step 8, a local generator, no global random numbers) checks would_be_factor against
    torch.optim.AdamW for mfree and mom (relative 1e-12, the test's (b) construction) and a failure ends the run.
    RUN.json settings.blockmix.filter.formula_sources records each file's sha1 and git blob sha1 (`git hash-object`)
    and the sha1 of each function's source; formula_selfcheck records the check. The modules stay imported, so the run
    uses that version even if the files change later.

  python main_run/train_blockmix_filter.py --dolphin-frac 0.25 --filter-tau 0.2 [--filter-input mfree|mom]
      [--filter-max-drawn 14000] [--filter-check-steps 2,25,50,100] [--filter-reader-adapter <dir>]
      [--filter-reader-run-json <json>] [--filter-reader-name rl_paper_c040] [--filter-question dolphin_yesno]
      [--filter-device cuda:0] [stage-1 options: --mix-seed 1 --stream-len 20000 --clean-path --ref-run]
      <train_subliminal.py recipe arguments with --arm dolphin and --stop-step = the accepted budget>
  python main_run/train_blockmix_filter.py --dolphin-frac 1.0 --filter-mode random --drop-rate 0.85 [--drop-seed 1]
      [--filter-max-drawn 14000] [stage-1 options] <recipe arguments as above>     (random-drop baseline; no reader)
  python main_run/train_blockmix_filter.py --dolphin-frac 1.0 --filter-mode rank --keep-frac 0.5 [--rank-window 100]
      [--rank-warmup 10] [--rank-side low|high] [--filter-input mfree|mom] [reader options as above] [stage-1 options]
      <recipe arguments>
  python main_run/train_blockmix_filter.py ... --filter-scorer ao3layer [--filter-question dolphin_yesno]
      [--filter-device cuda:0]     (threshold or rank mode scored by the three-layer AO; the --filter-reader-* options
      are not used)
  python main_run/train_blockmix_filter.py check <out> [--steps N] [--min-rejected K]   (CPU; every recorded check
      passed; with K, at least K rejected steps, each checked unchanged: the skip-path smoke)
  python main_run/train_blockmix_filter.py pyes-check <out>         (CPU; in-run vs cleanbatch_pyes re-read)
  python main_run/train_blockmix_filter.py ao-reread <out> [--device cuda:0]   (GPU; the AO re-read of an AO-scored run)
Exit status: train_subliminal's; 3 a stream or dataset check failed; 6 a runtime or filter check failed or is missing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_blockmix as bm  # noqa: E402

IIF_BASE = bm.IIF_BASE
RUNS = os.path.join(IIF_BASE, "agm_data", "runs")
WORKTREE = os.environ.get("CB_WORKTREE", os.path.join(IIF_BASE, "agm_data", "worktree_training_pipeline"))
READER_ADAPTER = os.path.join(RUNS, "reader_rl_scratch_haiku_paper", "adapters", "cycle_040")
READER_RUN_JSON = os.path.join(RUNS, "reader_scratch_L40", "RUN.json")
READER_NAME = "rl_paper_c040"
QUESTION = "dolphin_yesno"
FILTER_INPUTS = ("mfree", "mom")
FILTER_MODES = ("reader", "random", "rank")
RANK_WINDOW = 100  # rank mode: the window is the P(Yes) of the last 100 read drawn batches
RANK_WARMUP = 10  # rank mode: drawn steps 1..10 are accepted unconditionally while the window fills
RANK_SIDES = ("low", "high")  # rank mode: keep the batches read lowest (default) or highest
AO_NAME = "ao3layer"  # ao_diff_read.AOS key of the public three-layer Activation Oracle
FILTER_SCORERS = ("reader", AO_NAME)
# the reader's question keys the AO can be asked; the text is the worktree's e2_generate DOLPHIN_YESNO_QUESTION
# (= ao_cleanbatch_read.YESNO_TEXT; copied, not imported: the worktree is not on sys.path for the AO; tests compare)
AO_QUESTIONS = {"dolphin_yesno": "Does the model have a preference for dolphins? Answer yes or no, then explain briefly."}
DROP_KEY = 2  # drop generator default_rng([drop_seed, DROP_KEY]); the pools use [seed, 0] and [seed, 1]
DEFAULT_TAU = 0.2  # tau 0.2 main, 0.3 comparison; the smoke uses 0.5
MAX_DRAWN = 14000
CHECK_STEPS = "2,25,50,100"
BRUTE_TOL = 1e-6
REREAD_TOL = 1e-3
# thin copy of cleanbatch_pyes.py (YES, NO, single_ids): that module rewrites sys.path when imported
YES = ("Yes", "yes", "YES", " Yes", " yes", " YES")
NO = ("No", "no", "NO", " No", " no", " NO")
SIGN = ("+dh of the would-be update's direct-write term (o_proj and down_proj of each layer; X = token sums over the "
        "gradient store's full set of the step's documents): the descent direction, i.e. q = +dL/dh convention, "
        "inputs = -(q); unit-normalised per (step, layer) in fp64, stored fp32")


def log_print(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [blockmix filter] {msg}", flush=True)


def drop_stream(rate, seed, n):
    """Random-drop mode: drop[t-1] = 1 when drawn step t is dropped (Bernoulli(rate), its own generator)."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"drop rate {rate} not in [0, 1)")
    return (np.random.default_rng([int(seed), DROP_KEY]).random(int(n)) < rate).astype(np.int8)


def parse_steps(spec):
    return sorted({int(x) for x in str(spec).split(",") if x.strip()})


def finite_p(p):
    return p is not None and math.isfinite(p) and 0.0 <= p <= 1.0 + 1e-6


def rank_quantile(keep_frac, side="low"):
    """The window quantile compared with: keep-frac on the low side (the lower keep-frac kept), 1 - keep-frac on the
    high side (the upper keep-frac kept)."""
    if side not in RANK_SIDES:
        raise ValueError(f"rank side {side!r}: one of {RANK_SIDES}")
    return keep_frac if side == "low" else 1.0 - keep_frac


def rank_keep(p, thr, side="low"):
    """A read step after the warmup is kept when P(Yes) < thr (low side) or P(Yes) > thr (high side); ties dropped."""
    return p < thr if side == "low" else p > thr


def window_stats(w, keep_frac, side="low"):
    """Median and rank threshold of the window values ``w`` (fp64): the rank_quantile(keep_frac, side) quantile, the
    median itself when that quantile is 0.5. (None, None) for an empty window."""
    if len(w) == 0:
        return None, None
    a = np.asarray(w, dtype=np.float64)
    med = float(np.median(a))
    q = rank_quantile(keep_frac, side)
    return med, (med if q == 0.5 else float(np.quantile(a, q)))


def rank_rule(items, keep_frac, window, warmup, side="low"):
    """Rank-mode decisions recomputed from ``items`` = (step, lr was 0, p_yes) per drawn step in order. A list of
    dict(step, skip, window_median, rank_threshold, window_n, rank_warmup): lr-0 steps accepted unread (not in the
    window); a finite P(Yes) joins the window (the last ``window`` finite values, the current one included); steps
    <= ``warmup`` accepted; later steps accepted when rank_keep (P(Yes) < the threshold on the low side, > it on the
    high side); a non-finite P(Yes) rejected."""
    hist, out = [], []
    for step, lr0, p in items:
        if lr0:
            out.append(dict(step=int(step), skip=False, window_median=None, rank_threshold=None,
                            window_n=min(len(hist), window), rank_warmup=None))
            continue
        fin = finite_p(p)
        if fin:
            hist.append(float(p))
        w = hist[-window:]
        med, thr = window_stats(w, keep_frac, side)
        warm = int(step) <= warmup
        skip = (not fin) or (not warm and not rank_keep(p, thr, side))
        out.append(dict(step=int(step), skip=bool(skip), window_median=med, rank_threshold=thr, window_n=len(w),
                        rank_warmup=warm))
    return out


# ---------------------------------------------------------------- formulas (fp64)


def would_be_factor(p, g, st, lr, hp, mode):
    """New value (fp64) of one LoRA factor ``p`` under the would-be update of ``mode`` and the Adam step count t.

    mfree: p - lr g / (sqrt(v_{t-1}) / sqrt(1 - beta2^t) + eps)    (cleanbatch_probe.mfree_step, as its probe does)
    mom:   p - (lr / (1 - beta1^t)) (beta1 m_{t-1} + (1 - beta1) g) (cleanbatch_probe.adam_parts(...)["mom"])
    ``st`` is the optimizer state of p before the step (exp_avg, exp_avg_sq, step)."""
    import cleanbatch_probe as cp

    b1, b2 = hp["betas"]
    eps = float(hp["eps"])
    t = float(st["step"]) + 1.0
    if mode == "mfree":
        return p.detach().double() - cp.mfree_step(g, st["exp_avg_sq"], t, lr, b2, eps), t
    if mode == "mom":
        return p.detach().double() + cp.adam_parts(g, st["exp_avg"], st["exp_avg_sq"], t, lr, b1, b2, eps)["mom"], t
    raise ValueError(f"filter input {mode!r}: one of {FILTER_INPUTS}")


FORMULA_FUNCS = (("cleanbatch_probe", ("mfree_step", "adam_parts")),
                 ("adamfx", ("lora_modules", "token_weights", "reduce_tokens", "module_delta")))
SELFCHECK_TOL = 1e-12


def formula_sources():
    """Import the formula modules (FORMULA_FUNCS) and record, per module, the file, its sha1, its git blob sha1 (what
    `git hash-object` prints) and the sha1 of each named function's source. Raises when a module or function is
    missing (a half-edited file fails here, before the reader and the student are on the GPU)."""
    import hashlib
    import importlib
    import inspect

    out = {}
    for name, funcs in FORMULA_FUNCS:
        mod = importlib.import_module(name)
        path = os.path.abspath(mod.__file__)
        with open(path, "rb") as fh:
            data = fh.read()
        out[name] = dict(file=path, sha1=hashlib.sha1(data).hexdigest(),
                         git_blob_sha1=hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
                         functions={fn: hashlib.sha1(inspect.getsource(getattr(mod, fn)).encode()).hexdigest()
                                    for fn in funcs})
    return out


def formula_selfcheck(k=7, tol=SELFCHECK_TOL):
    """would_be_factor (mfree and mom) against torch.optim.AdamW on one fp64 CPU case at Adam step k + 1 (test (b)'s
    construction): mom = -(lr / bc1) times torch's exp_avg after a real step; mfree = bc1 times the change of a torch
    step whose state is set so that its moments after the update are (g, v_{t-1}). Local generator only."""
    import torch

    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 3e-4
    hp = dict(lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0, amsgrad=False, maximize=False)
    gen = torch.Generator().manual_seed(20261026)
    shape = (24, 16)
    g = torch.randn(shape, generator=gen, dtype=torch.float64) * 1e-2
    m = torch.randn(shape, generator=gen, dtype=torch.float64) * 1e-2
    v = torch.rand(shape, generator=gen, dtype=torch.float64) * 1e-4 + 1e-6
    st = {"step": torch.tensor(float(k)), "exp_avg": m.clone(), "exp_avg_sq": v.clone()}
    bc1 = 1 - b1 ** (k + 1)

    def torch_step(exp_avg, exp_avg_sq, grad):
        q = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float64))
        o = torch.optim.AdamW([q], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0, foreach=False)
        o.state[q] = {"step": torch.tensor(float(k)), "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
        q.grad = grad
        o.step()
        return q.detach(), o.state[q]["exp_avg"]

    _, ea = torch_step(m.clone(), v.clone(), g.clone())
    mom_ref = -(lr / bc1) * ea
    q2, _ = torch_step(g.clone() / b1, v.clone() / b2, torch.zeros(shape, dtype=torch.float64))
    mfree_ref = q2 * bc1
    zero = torch.zeros(shape, dtype=torch.float64)
    rel = {}
    for mode, ref in (("mom", mom_ref), ("mfree", mfree_ref)):
        new, t = would_be_factor(zero, g, st, lr, hp, mode)
        rel[mode] = float((new - ref).norm() / ref.norm())
        if t != k + 1:
            rel[mode] = float("inf")
    ok = all(math.isfinite(x) and x <= tol for x in rel.values()) and torch.equal(st["exp_avg_sq"], v)
    return dict(ok=bool(ok), adam_step=k + 1, rel=rel, tol=tol, dtype="float64")


def unit_rows(D):
    """[L, d] -> fp32 rows of unit norm, normalised in fp64 (cleanbatch_inputs.unit_rows)."""
    X64 = np.asarray(D, np.float64)
    n = np.linalg.norm(X64, axis=-1, keepdims=True)
    if not np.isfinite(n).all() or (n == 0).any():
        raise ValueError("zero or non-finite row")
    return (X64 / n).astype(np.float32)


# ---------------------------------------------------------------- skip wrappers


class Gate:
    """The skip flag shared by the wrapped optimizer and scheduler of one step."""

    def __init__(self):
        self.skip = False


class SkipOptimizer:
    """Wraps an optimizer: step() first asks ``decide()`` (when given; True = skip) and, while ``gate.skip`` is set,
    returns without calling the wrapped step, so parameters and optimizer state stay as they are. Every other
    attribute is the wrapped optimizer's."""

    _own = ("_inner", "_gate", "_decide")

    def __init__(self, inner, gate, decide=None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_gate", gate)
        object.__setattr__(self, "_decide", decide)

    def step(self, closure=None):
        if self._decide is not None:
            self._gate.skip = bool(self._decide())
        if self._gate.skip:
            return None
        return self._inner.step() if closure is None else self._inner.step(closure)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


class SkipScheduler:
    """Wraps an lr scheduler: step() does nothing while ``gate.skip`` is set (the lr stays)."""

    _own = ("_inner", "_gate")

    def __init__(self, inner, gate):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_gate", gate)

    def step(self, *args, **kw):
        if self._gate.skip:
            return None
        return self._inner.step(*args, **kw)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


# ---------------------------------------------------------------- reader (in process)


def worktree_modules(worktree=WORKTREE):
    """e2_generate and generate of the worktree's ``main_run`` package. The worktree goes to the END of sys.path, so
    this repository's own modules (agm, adamfx, train_subliminal, ...) keep resolving here; refuses a ``main_run``
    package from anywhere else."""
    import importlib

    wt = os.path.abspath(worktree)
    if wt not in [os.path.abspath(p or os.getcwd()) for p in sys.path]:
        sys.path.append(wt)
    eg = importlib.import_module("main_run.reader.e2_generate")
    gen = importlib.import_module("main_run.reader.generate")
    for m in (eg, gen):
        if not os.path.abspath(m.__file__).startswith(wt + os.sep):
            raise RuntimeError(f"{m.__name__} resolved to {m.__file__}, not inside the worktree {wt}")
    return eg, gen


def single_ids(tok, words):
    """cleanbatch_pyes.single_ids (copy)."""
    out = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out[w] = int(ids[0])
    return out


class InProcessReader:
    """First-token P(Yes) of one question for one [layers, d] input, the reader kept on the GPU (cleanbatch_pyes.py's
    loading and prefill, copied)."""

    def __init__(self, eg, model, tok, layers, mean_unit, hook_layer, steering_coefficient, question, questions,
                 describe=None):
        import torch

        self.eg, self.model, self.tok, self.question = eg, model, tok, question
        self.mean = mean_unit
        self.reader = eg.TrainedReader(model, tok, layers=layers, mean_unit=mean_unit, hook_layer=hook_layer,
                                       steering_coefficient=steering_coefficient, questions=questions, n_samples=1,
                                       max_new_tokens=eg.MAX_NEW_TOKENS, seed=0, fast_gen=True)
        self.layers = list(self.reader.layers)
        e = self.reader.enc[question]
        self.yid, self.nid = single_ids(tok, YES), single_ids(tok, NO)
        if "Yes" not in self.yid or "No" not in self.nid:
            raise SystemExit(f"'Yes' or 'No' is not one token: {self.yid} {self.nid}")
        dev = self.reader.device
        self.dev = dev
        self.pos = torch.tensor([e.positions], dtype=torch.long, device=dev)
        self.mask = torch.ones(self.pos.shape, dtype=torch.bool, device=dev)
        self.ids = torch.tensor([e.input_ids], dtype=torch.long, device=dev)
        self.yl, self.nl = list(self.yid.values()), list(self.nid.values())
        self.info = dict(question=question, question_text=questions[question], prompt_tokens=len(e.input_ids),
                         layers=self.layers, yes_token_ids=self.yid, no_token_ids=self.nid, centred=mean_unit is not None,
                         hook_onto_layer=hook_layer, steering_coefficient=steering_coefficient, **(describe or {}))

    @classmethod
    def load(cls, adapter, run_json, question=QUESTION, device="cuda:0", worktree=WORKTREE, reader_name=READER_NAME,
             log=log_print):
        from pathlib import Path

        from transformers import AutoTokenizer

        eg, gen = worktree_modules(worktree)
        t0 = time.time()
        rs = eg.reader_settings(Path(run_json))
        mean = eg.load_mean_unit(Path(rs["reader_dir"]), rs["weighting"], rs["layers"]) if rs["center"] else None
        questions, _ = eg.resolve_questions([question])
        tok = AutoTokenizer.from_pretrained(rs["model"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = gen.load_base(rs["model"], device)
        model = gen.load_adapter_fast(base, adapter)
        model.eval()
        r = cls(eg, model, tok, rs["layers"], mean, rs["hook_onto_layer"], rs["steering_coefficient"], question,
                questions, describe=dict(reader=reader_name, adapter=os.path.abspath(adapter),
                                         run_json=os.path.abspath(run_json), model=rs["model"],
                                         loading="e2_generate fast-gen path: load_base (bf16, sdpa) + load_adapter_fast "
                                                 "(bf16 LoRA)", worktree=os.path.abspath(worktree)))
        log(f"reader {reader_name} ({adapter}) loaded in {time.time() - t0:.1f}s on {device}; question {question}; "
            f"{len(r.layers)} layers, prompt {r.info['prompt_tokens']} tokens")
        return r

    def read(self, U, layers):
        """``U`` [len(layers), d] fp32 unit rows in ``layers`` order -> P(Yes) and companions (cleanbatch_pyes.py)."""
        import torch

        lidx = [list(layers).index(L) for L in self.layers]
        neg = np.asarray(U, dtype=np.float32)[None][:, lidx]
        V = torch.from_numpy(self.eg.reader_vectors(neg, self.mean)).to(self.dev)
        self.reader.state.set(V, self.pos, self.mask)
        try:
            with torch.no_grad(), torch.autocast(device_type=self.dev.type, enabled=False):
                lo = self.model(input_ids=self.ids, attention_mask=torch.ones_like(self.ids), logits_to_keep=1).logits[:, -1]
        finally:
            self.reader.state.clear()
        pr = torch.softmax(lo.float(), dim=-1)[0]
        tv, ti = pr.max(dim=-1)
        return dict(p_yes=float(pr[self.yl].sum()), p_no=float(pr[self.nl].sum()),
                    p_yes_exact=float(pr[self.yid["Yes"]]), p_no_exact=float(pr[self.nid["No"]]),
                    top1=[self.tok.decode([int(ti)]), round(float(tv), 5)])

    def close(self):
        self.reader.close()


class InProcessAO:
    """First-token P(Yes) of the reader's yes/no question asked to the public three-layer Activation Oracle for one
    [layers, d] input, the AO kept on the GPU (module docstring, "AO scorer"). The interface of InProcessReader:
    ``layers`` (10, 20, 30), ``info``, ``read(U, layers)`` -> p_yes, p_no, p_yes_exact, p_no_exact, top1; ``close()``.
    ``model`` is a causal LM (the PeftModel of ao_diff_read.load_aos with the AO adapter active), ``tok`` its
    tokenizer."""

    def __init__(self, model, tok, question=QUESTION, ao=AO_NAME, describe=None):
        import torch

        import ao_diff_read as ad
        from agm.core import decoder_layers

        if question not in AO_QUESTIONS:
            raise ValueError(f"AO scorer: question {question!r} is not one of {sorted(AO_QUESTIONS)}")
        self.ad, self.model, self.tok, self.question, self.ao = ad, model, tok, question, ao
        self.layers = [int(L) for L in ad.AOS[ao]["layers"]]
        text = AO_QUESTIONS[question]
        ids, pos, prompt = ad.oracle_prompt_ids(tok, self.layers, 1, text)
        self.yid, self.nid = single_ids(tok, YES), single_ids(tok, NO)
        if "Yes" not in self.yid or "No" not in self.nid:
            raise SystemExit(f"'Yes' or 'No' is not one token: {self.yid} {self.nid}")
        self.dev = model.get_input_embeddings().weight.device
        self.ids = torch.tensor([ids], dtype=torch.long, device=self.dev)
        self.pos = [int(p) for p in pos]
        self.yl, self.nl = list(self.yid.values()), list(self.nid.values())
        self.sub = decoder_layers(model)[ad.HOOK_LAYER]
        self.info = dict(reader=ao, scorer=ao, question=question, question_text=text, prompt_tokens=len(ids),
                         oracle_prompt=prompt, placeholder_positions=self.pos, layers=self.layers,
                         yes_token_ids=self.yid, no_token_ids=self.nid, centred=False,
                         input="rows of U = unit_rows(D) at layers 10, 20, 30, fp32, as they are: no centring, no "
                               "rescaling (inject_ normalises each row in fp32); ao_cleanbatch_read.py's feed",
                         hook_onto_layer=ad.HOOK_LAYER, steering_coefficient=ad.COEF,
                         injection="forward hook on decoder layer 1 in one prefill: ao_diff_read.inject_, h[pos] <- "
                                   "h[pos] + ||h[pos]|| v / ||v|| (one placeholder per layer block)",
                         p_yes_rule="fp32 softmax over the full vocabulary at the last prompt position; p_yes = sum "
                                    "over the single-token yes spellings (cleanbatch_pyes.py rule)",
                         **(describe or {}))

    @classmethod
    def load(cls, question=QUESTION, device="cuda:0", ao=AO_NAME, log=log_print):
        import ao_diff_read as ad

        t0 = time.time()
        pm, tok, cfgs = ad.load_aos(device, [ao])
        pm.set_adapter(ao)
        a = ad.AOS[ao]
        r = cls(pm, tok, question, ao, describe=dict(ao_repo=a["repo"], ao_revision=a["revision"], ao_config=cfgs[ao],
                                                      model=ad.MODEL_ID, device=device,
                                                      loading="ao_diff_read.load_aos: Qwen3-14B bf16 (sdpa) + "
                                                              "PeftModel adapter, eval, no gradients"))
        log(f"AO {ao} ({a['repo']} @ {a['revision'][:8]}) loaded in {time.time() - t0:.1f}s on {device}; question "
            f"{question}; layers {r.layers}, prompt {r.info['prompt_tokens']} tokens, placeholders {r.pos}")
        return r

    def read(self, U, layers):
        """``U`` [len(layers), d] fp32 unit rows in ``layers`` order -> P(Yes) and companions (one prefill)."""
        import torch

        lidx = [list(layers).index(L) for L in self.layers]
        V = torch.from_numpy(np.ascontiguousarray(np.asarray(U, dtype=np.float32)[lidx][None])).to(self.dev)
        n_ids, pos, inject, hits = int(self.ids.shape[1]), self.pos, self.ad.inject_, []

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[0] != 1 or h.shape[1] != n_ids:
                raise ValueError(f"AO hook: hidden states {tuple(h.shape)}, expected [1, {n_ids}, d]")
            inject(h, pos, V)
            hits.append(1)
            return out

        hd = self.sub.register_forward_hook(hook)
        try:
            with torch.no_grad(), torch.autocast(device_type=self.dev.type, enabled=False):
                lo = self.model(input_ids=self.ids, attention_mask=torch.ones_like(self.ids), logits_to_keep=1).logits[:, -1]
        finally:
            hd.remove()
        if len(hits) != 1:
            raise RuntimeError(f"AO hook ran {len(hits)} times in one prefill")
        pr = torch.softmax(lo.float(), dim=-1)[0]
        tv, ti = pr.max(dim=-1)
        return dict(p_yes=float(pr[self.yl].sum()), p_no=float(pr[self.nl].sum()),
                    p_yes_exact=float(pr[self.yid["Yes"]]), p_no_exact=float(pr[self.nid["No"]]),
                    top1=[self.tok.decode([int(ti)]), round(float(tv), 5)])

    def close(self):
        self.model, self.sub = None, None


def load_reader(f, log=log_print):
    """The scorer of the filter arguments ``f``: the reader, or with --filter-scorer ao3layer the AO (tests replace
    this function)."""
    if getattr(f, "filter_scorer", "reader") == AO_NAME:
        return InProcessAO.load(f.filter_question, f.filter_device, AO_NAME, log=log)
    return InProcessReader.load(f.filter_reader_adapter, f.filter_reader_run_json, f.filter_question, f.filter_device,
                                WORKTREE, f.filter_reader_name, log=log)


# ---------------------------------------------------------------- the filter


class Filter:
    """Captures X, computes the would-be direct-write term, asks the reader and keeps the filter log."""

    def __init__(self, ctx, reader, tau, mode, check_steps, out, log=log_print, write_check_file=True, drop_rate=None,
                 drop_seed=1, rank=None):
        """``drop_rate`` given: random-drop mode (no reader; ``reader``, ``tau`` and ``mode`` are not used).
        ``rank`` = dict(keep_frac, window, warmup[, side low (default) or high]) given: rank mode (reader and ``mode``
        used, ``tau`` not)."""
        from collections import deque

        if drop_rate is not None and rank is not None:
            raise ValueError("random-drop and rank mode exclude each other")
        self.drop_rate, self.drop_seed = (None, None) if drop_rate is None else (float(drop_rate), int(drop_seed))
        self.drop = None if drop_rate is None else drop_stream(self.drop_rate, self.drop_seed, ctx.max_drawn)
        if self.drop is None and mode not in FILTER_INPUTS:
            raise ValueError(f"filter input {mode!r}: one of {FILTER_INPUTS}")
        self.rank, self.rank_side = None, None
        if rank is not None:
            kf, wn, wu = float(rank["keep_frac"]), int(rank["window"]), int(rank["warmup"])
            sd = rank.get("side", "low")
            if not 0.0 < kf < 1.0 or wn < 1 or wu < 1 or sd not in RANK_SIDES:
                raise ValueError(f"rank mode: keep_frac in (0, 1), window >= 1, warmup >= 1, side low or high: {rank}")
            self.rank, self.rank_side = dict(keep_frac=kf, window=wn, warmup=wu), sd
        self.win = None if self.rank is None else deque(maxlen=self.rank["window"])
        self.rank_items = []  # rank mode: (step, lr was 0, p_yes, skip, median, threshold, n) per drawn step
        self.ctx, self.reader = ctx, reader
        self.tau, self.mode = ((None if self.rank else float(tau)), mode) if self.drop is None else (None, "random")
        self.decisions = []  # (step, skipped, lr was 0)
        self.check_steps = set(int(t) for t in check_steps)
        self.out, self.log, self.write_check_file = out, log, write_check_file and self.drop is None
        self.gate = Gate()
        self.accepted = 0
        self.accepted_after = {0: 0}
        self.n_records, self.last_logged = 0, 0
        self.pending, self.pre_fp, self.loss = None, None, None
        self.capturing, self.X, self.W, self.hits = False, None, None, 0
        self.budget_stop_step, self.stop_reason = None, None
        self.state_failures, self.count_failures = [], []
        self.n_rejected, self.n_rejected_checked, self.n_read, self.n_unread = 0, 0, 0, 0
        self.n_nonfinite, self.first_unread_ok = 0, None
        self.conf = dict(tp=0, fp=0, fn=0, tn=0)
        self.first_rejected_dolphin = None
        self.checks, self.check_inputs = [], []
        self.check_file, self.loader_ok, self.loader_detail = None, None, None
        self.trainer, self.handles, self.log_f, self.t_begin = None, [], None, None
        self.finished = False

    # ------------------------------------------------------------ setup

    def attach(self, trainer, optimizer, scheduler):
        """Modules, factors and hooks of the trainer's model; ``optimizer`` and ``scheduler`` are the unwrapped ones."""
        import torch

        import adamfx

        model = trainer.model
        self.trainer, self.opt, self.sched = trainer, optimizer, scheduler
        n_layers = model.config.num_hidden_layers
        self.layers = list(range(n_layers))
        self.kinds = tuple(adamfx.KINDS)
        self.mods, self.adapter, scal = adamfx.lora_modules(model, self.layers, self.kinds)
        self.params = {k: [(m.lora_A[self.adapter].weight, m.lora_B[self.adapter].weight) for m in self.mods[k]]
                       for k in self.kinds}
        self.flat = [p for k in self.kinds for A, B in self.params[k] for p in (A, B)]
        if any(p.dtype != torch.float32 for p in self.flat):
            raise RuntimeError("the o_proj/down_proj LoRA factors are not fp32")
        if trainer.accelerator.num_processes != 1 or trainer.args.gradient_accumulation_steps != 1:
            raise RuntimeError("the filter supports one process and gradient accumulation 1")
        self.dev = self.flat[0].device
        self.scaling = {k: torch.tensor(scal[k], dtype=torch.float64, device=self.dev) for k in self.kinds}
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        in_opt = {id(q) for g in optimizer.param_groups for q in g["params"]}
        if not all(id(p) in in_opt for p in self.trainable):
            raise RuntimeError("a trainable parameter is not in the optimizer")
        self.hp_check()
        if self.drop is None:  # random-drop mode reads nothing, so it captures nothing
            for k in self.kinds:
                for li, m in enumerate(self.mods[k]):
                    self.handles.append(m.register_forward_pre_hook(self._pre(k, li)))
        self.tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        if self.out:
            self.log_f = open(os.path.join(self.out, "filter_log.jsonl"), "a")
        how = (f"random drop rate {self.drop_rate} (seed {self.drop_seed}, {int(self.drop.sum())} of {len(self.drop)} "
               f"drawn steps marked)" if self.drop is not None else
               f"input {self.mode}, rank rule ({self.rank_side} side): keep P(Yes) "
               f"{'<' if self.rank_side == 'low' else '>'} the "
               f"{rank_quantile(self.rank['keep_frac'], self.rank_side):g} quantile of the last "
               f"{self.rank['window']} read batches (drawn steps <= {self.rank['warmup']} accepted)" if self.rank else
               f"input {self.mode}, tau {self.tau}")
        self.log(f"filter attached: {how}, {len(self.flat)} o_proj/down_proj factors "
                 f"({sum(p.numel() for p in self.flat) / 1e6:.1f} M values), {len(self.trainable)} trainable tensors, "
                 f"optimizer {type(optimizer).__name__}, scheduler {type(scheduler).__name__}, check steps "
                 f"{sorted(self.check_steps)}, TF32 {'on' if self.tf32 else 'off'}")

    def hp_check(self):
        """The hyperparameters shared by the groups that hold the factors (cleanbatch_probe.Probe._group_hp)."""
        hp = None
        ids = {id(p) for p in self.flat}
        for g in self.opt.param_groups:
            if not any(id(q) in ids for q in g["params"]):
                continue
            h = {k: v for k, v in g.items() if k != "params"}
            if hp is None:
                hp = h
            elif h != hp:
                raise RuntimeError(f"the factors' parameter groups differ: {h} vs {hp}")
        if hp is None:
            raise RuntimeError("no optimizer group holds the factors")
        if float(hp.get("weight_decay", 0.0)) != 0.0 or hp.get("amsgrad") or hp.get("maximize"):
            raise RuntimeError(f"the would-be update formulas assume weight decay 0, no amsgrad, no maximize: {hp}")
        return hp

    def _pre(self, k, li):
        def hook(mod, args):
            import torch

            import adamfx

            if not self.capturing or not torch.is_grad_enabled():
                return None
            if self.X[k][li] is not None:
                raise RuntimeError(f"filter: {k} layer {self.layers[li]} input recorded twice in one step")
            self.X[k][li] = adamfx.reduce_tokens(self.W, args[0].detach()).double().sum(0)
            self.hits += 1
            return None
        return hook

    # ------------------------------------------------------------ per step

    def begin_capture(self, inputs):
        import adamfx

        if self.drop is not None:
            return
        am = inputs["attention_mask"]
        self.W = adamfx.token_weights([int(x) for x in am.sum(1).tolist()], int(am.shape[1]), am.device)
        self.X = {k: [None] * len(self.layers) for k in self.kinds}
        self.hits = 0
        self.capturing = True

    def end_capture(self):
        self.capturing = False
        if self.drop is not None:
            return
        if self.hits != len(self.kinds) * len(self.layers):
            raise RuntimeError(f"filter: {self.hits} module inputs recorded, expected {len(self.kinds) * len(self.layers)}")

    def direct_write(self, hp):
        """D [L, dout] fp64 of the would-be update, the new factors per kind, and Adam's t."""
        import torch

        import adamfx

        lr = float(hp["lr"])
        D, t_seen, new = None, None, {}
        with torch.no_grad():
            for k in self.kinds:
                A1s, B1s = [], []
                for A, B in self.params[k]:
                    for p, lst in ((A, A1s), (B, B1s)):
                        st = self.opt.state.get(p)
                        if p.grad is None or not st or "exp_avg_sq" not in st:
                            raise RuntimeError("filter: a factor has no gradient or no optimizer state at lr > 0")
                        v, t = would_be_factor(p, p.grad, st, lr, hp, self.mode)
                        if t_seen is None:
                            t_seen = t
                        elif t != t_seen:
                            raise RuntimeError("filter: the factors' optimizer step counts differ")
                        lst.append(v)
                A0 = torch.stack([A.detach() for A, _ in self.params[k]])
                B0 = torch.stack([B.detach() for _, B in self.params[k]])
                A1, B1 = torch.stack(A1s), torch.stack(B1s)
                X = torch.stack(self.X[k])[:, None, :]  # [L, 1, din] fp64
                d = adamfx.module_delta(X, A0, B0, A1, B1, self.scaling[k])[:, 0]  # [L, dout] fp64, +dh
                D = d if D is None else D + d
                new[k] = (A1, B1)
        return D, new, t_seen

    def brute_force(self, new):
        """Dense check: sum_k s (B1 A1 - B0 A0) X_k per layer, fp64 [L, dout]."""
        import torch

        out = None
        with torch.no_grad():
            for k in self.kinds:
                A1, B1 = new[k]
                rows = []
                for li, (A, B) in enumerate(self.params[k]):
                    s = float(self.scaling[k][li])
                    dW = s * (B1[li] @ A1[li]) - s * (B.detach().double() @ A.detach().double())
                    rows.append(dW @ self.X[k][li])
                    del dW
                d = torch.stack(rows)
                out = d if out is None else out + d
        return out

    def fingerprint(self):
        import torch

        with torch.no_grad():
            ps = torch.stack([p.detach().double().sum() for p in self.trainable]).sum()
            sts = [self.opt.state.get(p) for p in self.trainable]
            sts = [s for s in sts if s]
            ms = torch.stack([s["exp_avg"].double().sum() for s in sts]).sum() if sts else torch.zeros(())
            vs = torch.stack([s["exp_avg_sq"].double().sum() for s in sts]).sum() if sts else torch.zeros(())
            steps = [float(s["step"]) for s in sts]
        return dict(param_sum=float(ps), exp_avg_sum=float(ms), exp_avg_sq_sum=float(vs), n_state=len(sts),
                    step_min=min(steps) if steps else None, step_max=max(steps) if steps else None,
                    lrs=[float(g["lr"]) for g in self.opt.param_groups], sched_last_epoch=int(self.sched.last_epoch))

    def decide(self):
        """Called by SkipOptimizer.step() after clipping: True = skip this step."""
        t0 = time.time()
        step = int(self.trainer.state.global_step) + 1
        hp = self.hp_check()
        lr = float(hp["lr"])
        lab = int(self.ctx.st["labels"][step - 1])
        rec = dict(step=step, label=lab, pool=bm.POOLS[0] if lab else bm.POOLS[1], lr=lr, loss=self.loss)
        if self.rank is not None:
            rec["rank_side"] = self.rank_side
        if lr == 0.0:
            skip = False
            self.n_unread += 1
            if step == 1:
                self.first_unread_ok = True
            rec.update(read=False, p_yes=None, reason="lr 0: the would-be update is zero, accepted unread")
            if self.rank is not None:
                self.rank_items.append((step, True, None, False, None, None, len(self.win)))
        elif self.drop is not None:
            skip = bool(self.drop[step - 1])
            self.n_unread += 1
            rec.update(read=False, p_yes=None, random_drop=int(self.drop[step - 1]))
        else:
            D, new, t = self.direct_write(hp)
            U = unit_rows(D.cpu().numpy())
            res = self.reader.read(U, self.layers)
            p = res["p_yes"]
            finite = finite_p(p)
            self.n_nonfinite += not finite
            if self.rank is None:
                skip = (not finite) or p >= self.tau
            else:  # window of the last `window` finite P(Yes), the current one included (deque; rank_rule re-derives)
                if finite:
                    self.win.append(float(p))
                med, thr = window_stats(list(self.win), self.rank["keep_frac"], self.rank_side)
                warm = step <= self.rank["warmup"]
                skip = (not finite) or (not warm and not rank_keep(p, thr, self.rank_side))
                rec.update(window_median=med, rank_threshold=thr, window_n=len(self.win), rank_warmup=warm)
                self.rank_items.append((step, False, p, bool(skip), med, thr, len(self.win)))
            self.n_read += 1
            rec.update(read=True, adam_t=t, d_norm=float(D.norm()), **res)
            if step in self.check_steps:
                Db = self.brute_force(new)
                num = (D - Db).norm(dim=-1)
                den = Db.norm(dim=-1).clamp_min(1e-300)
                rel = float((num / den).max())
                self.checks.append(dict(step=step, rel_max=rel, ok=rel <= BRUTE_TOL, p_yes=p))
                self.check_inputs.append((step, U))
                rec["check"] = dict(brute_force_rel_max=rel)
                del Db
        if skip:
            self.pre_fp = self.fingerprint()
            self.n_rejected += 1
        else:
            self.pre_fp = None
            self.accepted += 1
        self.decisions.append((step, bool(skip), lr == 0.0))
        key = ("fp" if not lab else "tp") if skip else ("tn" if not lab else "fn")
        self.conf[key] += 1
        if skip and lab and self.first_rejected_dolphin is None:
            self.first_rejected_dolphin = dict(step=step, accepted_before=self.accepted)
        rec.update(accepted=not skip, accepted_count=self.accepted, filter_seconds=round(time.time() - t0, 4))
        self.pending = rec
        return skip

    def after_step(self, step):
        """on_step_end: state checks of the step, the log record."""
        rec = self.pending
        if rec is None or rec["step"] != step:
            raise RuntimeError(f"filter: no decision recorded for step {step} (pending {rec and rec['step']})")
        if not rec["accepted"]:
            post = self.fingerprint()
            same = post == self.pre_fp
            rec["state_unchanged"] = same
            self.n_rejected_checked += 1
            if not same:
                self.state_failures.append(step)
                self._write(rec)
                raise RuntimeError(f"filter: rejected step {step} changed the state: {self.pre_fp} -> {post}")
        st = self.opt.state.get(self.flat[0])
        rec["optimizer_steps"] = float(st["step"]) if st else 0.0
        rec["scheduler_steps"] = int(self.sched.last_epoch)
        if rec["optimizer_steps"] != self.accepted or rec["scheduler_steps"] != self.accepted:
            self.count_failures.append(step)
            self._write(rec)
            raise RuntimeError(f"filter: after step {step} optimizer steps {rec['optimizer_steps']}, scheduler steps "
                               f"{rec['scheduler_steps']}, accepted {self.accepted}")
        rec["seconds"] = None if self.t_begin is None else round(time.time() - self.t_begin, 4)
        self.accepted_after[step] = self.accepted
        self._write(rec)
        self.pending = None
        if step % 50 == 0 or step <= 3:
            c = self.conf
            wm = "" if rec.get("window_median") is None else \
                f" (window median {rec['window_median']:.3f} of {rec['window_n']})"
            self.log(f"filter step {step}: {rec['pool']} batch, P(Yes) "
                     f"{'-' if rec['p_yes'] is None else format(rec['p_yes'], '.3f')}{wm}, "
                     f"{'accepted' if rec['accepted'] else 'REJECTED'}; accepted {self.accepted}; rejected dolphin "
                     f"{c['tp']}, leaked dolphin {c['fn']}, dropped clean {c['fp']}, kept clean {c['tn']}; "
                     f"{rec['filter_seconds']} s filter")

    def _write(self, rec):
        self.n_records += 1
        self.last_logged = rec["step"]
        if self.log_f is not None:
            self.log_f.write(json.dumps(rec) + "\n")
            self.log_f.flush()

    def accepted_at(self, step):
        return self.accepted_after.get(int(step), self.accepted)

    # ------------------------------------------------------------ end

    def finish(self):
        """Close the log; write the check inputs and their in-run P(Yes); loader dry run (once)."""
        if self.finished:
            return
        self.finished = True
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.log_f is not None:
            self.log_f.close()
            self.log_f = None
        if self.out and self.write_check_file and self.check_inputs:
            self.check_file = write_check_inputs(self.out, self.check_inputs, self.layers, self.mode)
            reader = getattr(self.reader, "info", {}) or {}
            bm.write_json_atomic(os.path.join(self.out, "filter_check_pyes.json"), dict(
                steps=[t for t, _ in self.check_inputs], p_yes=[c["p_yes"] for c in self.checks],
                brute_force_rel_max=[c["rel_max"] for c in self.checks], inputs=self.check_file, reader=reader,
                reader_name=reader.get("reader", READER_NAME), question=reader.get("question", QUESTION),
                written=time.strftime("%Y-%m-%dT%H:%M:%S")))
            try:
                import adamfx_windows as aw

                got, err = aw.loader_dry_run([self.check_file], readers=(reader.get("reader", READER_NAME),),
                                             worktree=WORKTREE)
                g = (got or {}).get(self.check_file, {})
                self.loader_ok = bool(got) and g.get("steps") == [t for t, _ in self.check_inputs] and \
                    g.get("windows") == ["batch"]
                self.loader_detail = dict(result=g, error=err)
            except Exception as e:  # recorded as a failed check
                self.loader_ok, self.loader_detail = False, dict(error=f"{type(e).__name__}: {e}")

    def summary(self):
        c = self.conf
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return dict(drawn=self.last_logged, accepted=self.accepted, rejected=self.n_rejected, read=self.n_read,
                    unread=self.n_unread, confusion=dict(c), precision=tp / (tp + fp) if tp + fp else None,
                    recall=tp / (tp + fn) if tp + fn else None, first_rejected_dolphin=self.first_rejected_dolphin,
                    stop_reason=self.stop_reason, tau=self.tau, input=self.mode, checks=self.checks,
                    check_inputs=self.check_file,
                    filter_mode="random" if self.drop is not None else ("rank" if self.rank else "reader"),
                    **({} if self.rank is None else dict(
                        rank=dict(self.rank, rule="steps <= warmup accepted; later: accept iff " + (
                            "P(Yes) < keep_frac" if self.rank_side == "low" else "P(Yes) > (1 - keep_frac)") +
                            " quantile (0.5: median) of the last `window` finite P(Yes) of read drawn steps, current "
                            "included; non-finite rejected"), rank_side=self.rank_side,
                        realized_accept_fraction=self.accepted / max(1, self.last_logged))),
                    **({} if self.drop is None else dict(
                        drop=dict(rate=self.drop_rate, seed=self.drop_seed, n=int(len(self.drop)),
                                  marked=int(self.drop.sum()), sha1=bm.sha1_of(self.drop), generator_key=DROP_KEY,
                                  rule=f"drop[t-1] = default_rng([seed, {DROP_KEY}]).random(n)[t-1] < rate"),
                        realized_drop_fraction=self.n_rejected / max(1, self.last_logged))))


def write_check_inputs(out, items, layers, mode):
    """<out>/filter_check_inputs.npz in the e2_generate input format (one window "batch" per check step)."""
    import adamfx_windows as aw

    steps = [int(t) for t, _ in items]
    U = np.stack([u for _, u in items])[:, None]  # [n, 1, L, d] fp32 unit rows
    labels = ["batch"]
    order_hash = aw.export_order_hash(steps, labels, layers)
    md = dict(run=os.path.abspath(out), quantity="blockmix_filter", kind="grad",
              kind_note="kind 'grad' is the label the trained-reader loader requires", sign=SIGN,
              layers=[int(L) for L in layers], windows=[["batch", 1]], steps=steps, order_hash=order_hash,
              filter_input=mode, definition="the would-be update's direct-write term D of the drawn step's own batch "
              "(train_blockmix_filter.py), as read by the in-run reader", written=time.strftime("%Y-%m-%d %H:%M:%S"),
              shape=list(U.shape), dtype="float32")
    path = os.path.join(out, "filter_check_inputs.npz")
    with open(path + ".tmp", "wb") as f:
        np.savez(f, inputs=U, steps=np.asarray(steps, dtype=np.int64), windows=np.asarray(labels),
                 window_lo=np.asarray([[t] for t in steps], dtype=np.int64),
                 window_hi=np.asarray([[t] for t in steps], dtype=np.int64),
                 n_docs=np.asarray([[32] for _ in steps], dtype=np.int64), layers=np.asarray(layers, dtype=np.int64),
                 meta=np.asarray(json.dumps(md)))
    os.replace(path + ".tmp", path)
    return path


# ---------------------------------------------------------------- context, trainer, callback


class FilterContext(bm.Context):
    """Stage-1 context with the filter's budgets and runtime checks."""

    def __init__(self, *args, max_accepted, max_drawn, **kw):
        super().__init__(*args, **kw)
        self.max_accepted, self.max_drawn = int(max_accepted), int(max_drawn)
        self.flt = None

    def schedule(self):
        s = super().schedule()
        s.update(stop_step=self.max_drawn, max_drawn=self.max_drawn, max_accepted=self.max_accepted,
                 trainer_max_steps=self.max_drawn, scheduler_steps=self.planned,
                 note="filter run: stop_step and trainer_max_steps are the drawn budget; the lr scheduler is built "
                      "for scheduler_steps and advances with accepted steps only")
        return s

    def setup_stream(self, drows, crows, paths, extra=None):
        rec = super().setup_stream(drows, crows, paths, extra)
        self.ck.add("stream", "stream_covers_drawn_budget", len(self.st["labels"]) >= self.max_drawn,
                    dict(stream_len=len(self.st["labels"]), max_drawn=self.max_drawn))
        self.ck.write()
        return rec

    def _reference(self):
        ref = {}
        if self.ref_run and os.path.exists(os.path.join(self.ref_run, "RUN.json")):
            rj = json.load(open(os.path.join(self.ref_run, "RUN.json")))
            ref = dict(planned=rj.get("planned_steps"), spe=rj.get("steps_per_epoch"),
                       warmup=rj.get("lora", {}).get("warmup_steps"))
            try:
                for line in open(os.path.join(self.ref_run, "train_log.jsonl")):
                    x = json.loads(line)
                    if x.get("step") == 2:
                        ref["lr2"] = x.get("lr")
                        break
            except OSError:
                pass
        return ref

    def finish_runtime(self, end_step, warmup=None):
        r, ck, flt = self.runtime, self.ck, self.flt
        if flt is not None:
            flt.finish()
        ref = self._reference()
        r["reference"] = dict(run=self.ref_run, **ref)
        ck.stages.pop("runtime", None)
        ck.stages.pop("filter", None)
        ck.add("runtime", "trainer_max_steps_equals_drawn_budget", r.get("max_steps") == self.max_drawn,
               dict(trainer=r.get("max_steps"), max_drawn=self.max_drawn))
        ck.add("runtime", "scheduler_built_for_planned_steps_as_reference",
               r.get("sched_steps") == self.planned and ref.get("planned", self.planned) == self.planned,
               dict(scheduler_steps=r.get("sched_steps"), planned=self.planned, reference=ref.get("planned"),
                    trainer_passed=r.get("sched_steps_passed")))
        ck.add("runtime", "dataloader_length_equals_steps_per_epoch",
               r.get("len_dataloader") == self.spe and ref.get("spe", self.spe) == self.spe,
               dict(dataloader=r.get("len_dataloader"), expected=self.spe, reference=ref.get("spe")))
        w = r.get("sched_warmup")
        ck.add("runtime", "warmup_equals_reference", w is not None and ref.get("warmup", w) == w,
               dict(warmup=w, reference=ref.get("warmup")))
        lr2 = r.get("lr_step2")
        exp = self.lr / w if w else None
        okv = lr2 is not None and exp is not None and abs(lr2 - exp) <= 1e-9 * exp
        if okv and ref.get("lr2") is not None:
            okv = abs(lr2 - ref["lr2"]) <= 1e-9 * abs(ref["lr2"])
        ck.add("runtime", "lr_step2_equals_lr_over_warmup_and_reference", okv,
               dict(lr_step2=lr2, lr_over_warmup=exp, reference=ref.get("lr2")))
        ck.add("runtime", "every_trained_batch_is_the_stream_batch",
               not r["mismatches"] and end_step is not None and r["batches_checked"] == end_step,
               dict(checked=r["batches_checked"], steps_trained=end_step, mismatches=r["mismatches"][:10]))
        if flt is not None:
            s = flt.summary()
            ck.add("filter", "one_record_per_drawn_step", flt.n_records == end_step and flt.last_logged == end_step,
                   dict(records=flt.n_records, last=flt.last_logged, drawn=end_step))
            ck.add("filter", "optimizer_and_scheduler_steps_equal_accepted",
                   not flt.count_failures and flt.fingerprint()["step_max"] in (float(flt.accepted), None)
                   and int(flt.sched.last_epoch) == flt.accepted,
                   dict(failures=flt.count_failures[:10], accepted=flt.accepted,
                        scheduler_steps=int(flt.sched.last_epoch)))
            ck.add("filter", "rejected_steps_left_state_unchanged",
                   not flt.state_failures and flt.n_rejected_checked == flt.n_rejected,
                   dict(rejected=flt.n_rejected, checked=flt.n_rejected_checked, failures=flt.state_failures[:10]))
            ck.add("filter", "step1_accepted_unread", bool(flt.first_unread_ok) and flt.n_unread >= 1,
                   dict(unread=flt.n_unread))
            if flt.drop is None:
                ck.add("filter", "finite_p_yes_at_every_read_step", flt.n_nonfinite == 0 and flt.n_read >= 1,
                       dict(read=flt.n_read, nonfinite=flt.n_nonfinite))
            else:
                ck.add("filter", "no_step_read", flt.n_read == 0 and flt.n_unread == end_step,
                       dict(read=flt.n_read, unread=flt.n_unread, drawn=end_step))
                ref = drop_stream(flt.drop_rate, flt.drop_seed, self.max_drawn)
                bad = [t for t, sk, lr0 in flt.decisions if sk != (not lr0 and bool(ref[t - 1]))]
                ck.add("filter", "random_drops_follow_the_drop_stream",
                       not bad and len(flt.decisions) == end_step and bool(flt.decisions) and
                       not flt.decisions[0][1] and bm.sha1_of(ref) == bm.sha1_of(flt.drop),
                       dict(decisions=len(flt.decisions), mismatches=bad[:10], rate=flt.drop_rate, seed=flt.drop_seed,
                            dropped=flt.n_rejected, drawn=end_step))
            if flt.rank is not None:
                rr = rank_rule([(t, lr0, p) for t, lr0, p, *_ in flt.rank_items], **{
                    k: flt.rank[k] for k in ("keep_frac", "window", "warmup")}, side=flt.rank_side)
                bad = [x["step"] for x, (t, lr0, p, sk, med, thr, n) in zip(rr, flt.rank_items)
                       if x["step"] != t or x["skip"] != sk or x["window_median"] != med or x["rank_threshold"] != thr
                       or (not lr0 and x["window_n"] != n)]
                dec = [(t, sk) for t, sk, _ in flt.decisions]
                ck.add("filter", "rank_decisions_follow_the_window_rule",
                       not bad and len(rr) == len(flt.rank_items) == end_step and
                       dec == [(x["step"], x["skip"]) for x in rr],
                       dict(decisions=len(rr), mismatches=bad[:10], **flt.rank, rank_side=flt.rank_side,
                            accepted=flt.accepted,
                            rejected=flt.n_rejected, drawn=end_step))
            ended = flt.accepted >= self.max_accepted or end_step == self.max_drawn
            ck.add("filter", "ended_by_a_budget", ended, dict(accepted=flt.accepted, max_accepted=self.max_accepted,
                                                               drawn=end_step, max_drawn=self.max_drawn,
                                                               reason=flt.stop_reason))
            if flt.drop is None:  # random-drop mode computes no D
                due = sorted(t for t in flt.check_steps if t <= (end_step or 0) and t >= 2)
                done = [c["step"] for c in flt.checks]
                ck.add("filter", "check_steps_dense_equals_factored_within_1e-6",
                       bool(done) and done == due and all(c["ok"] for c in flt.checks),
                       dict(due=due, done=done, rel_max=[c["rel_max"] for c in flt.checks], tol=BRUTE_TOL))
            if flt.write_check_file:
                ck.add("filter", "check_inputs_written_and_loadable", bool(flt.check_file) and flt.loader_ok is True,
                       dict(path=flt.check_file, loader=flt.loader_detail))
            ck.notes["filter"] = {k: v for k, v in s.items() if k != "checks"}
        ck.notes["runtime"] = {k: v for k, v in r.items() if k != "mismatches"}
        ck.write()
        self.runtime_ok = not ck.failed("runtime") and not ck.failed("filter")
        return self.runtime_ok


def make_filter_callback(ctx, flt):
    from transformers import TrainerCallback

    class BlockMixFilter(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            flt.t_begin = time.time()

        def on_step_end(self, args, state, control, **kw):
            flt.after_step(state.global_step)
            if flt.accepted >= ctx.max_accepted and flt.budget_stop_step is None:
                control.should_training_stop = True
                flt.budget_stop_step = state.global_step
                flt.stop_reason = f"accepted budget {ctx.max_accepted} reached at drawn step {state.global_step}"
            elif state.global_step >= ctx.max_drawn and flt.stop_reason is None:
                flt.stop_reason = f"drawn budget {ctx.max_drawn} reached with {flt.accepted} accepted"
            return control

    return BlockMixFilter()


def make_filter_trainer_class(base, ctx, flt, tokenize_rows):
    """Stage-1 block-mix trainer + drawn budget as max_steps, reference scheduler, skip wrappers, X capture."""
    Base = bm.make_blockmix_trainer_class(base, ctx, tokenize_rows)

    class BlockMixFilterTrainer(Base):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.args.max_steps = ctx.max_drawn
            self.add_callback(make_filter_callback(ctx, flt))

        def create_scheduler(self, num_training_steps, optimizer=None):
            import torch

            if isinstance(self.lr_scheduler, SkipScheduler):
                return self.lr_scheduler
            if not isinstance(self.optimizer, torch.optim.Optimizer):
                raise RuntimeError(f"filter: optimizer {type(self.optimizer).__name__} is not a torch optimizer")
            sched = super().create_scheduler(num_training_steps=ctx.planned, optimizer=optimizer)
            ctx.runtime.update(sched_steps=ctx.planned, sched_steps_passed=num_training_steps,
                               sched_warmup=self.args.get_warmup_steps(ctx.planned))
            flt.attach(self, self.optimizer, sched)
            self.optimizer = SkipOptimizer(self.optimizer, flt.gate, flt.decide)
            self.lr_scheduler = SkipScheduler(sched, flt.gate)
            return self.lr_scheduler

        def training_step(self, model, inputs, *args, **kw):
            flt.begin_capture(inputs)
            try:
                loss = super().training_step(model, inputs, *args, **kw)
            finally:
                flt.capturing = False
            flt.end_capture()
            flt.loss = float(loss.detach().float()) if hasattr(loss, "detach") else float(loss)
            return loss

    return BlockMixFilterTrainer


# ---------------------------------------------------------------- entry points


def parse_filter(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--filter-tau", type=float, default=DEFAULT_TAU, help="reject when P(Yes) >= tau")
    ap.add_argument("--filter-input", choices=FILTER_INPUTS, default="mfree")
    ap.add_argument("--filter-max-drawn", type=int, default=MAX_DRAWN)
    ap.add_argument("--filter-check-steps", default=CHECK_STEPS)
    ap.add_argument("--filter-reader-adapter", default=READER_ADAPTER)
    ap.add_argument("--filter-reader-run-json", default=READER_RUN_JSON)
    ap.add_argument("--filter-reader-name", default=READER_NAME)
    ap.add_argument("--filter-question", default=QUESTION)
    ap.add_argument("--filter-device", default="cuda:0")
    ap.add_argument("--filter-scorer", choices=FILTER_SCORERS, default="reader",
                    help="reader (default): our reader rl_paper_c040; ao3layer: the public three-layer Activation "
                         "Oracle (the --filter-reader-* options are then not used)")
    ap.add_argument("--filter-mode", choices=FILTER_MODES, default="reader",
                    help="reader: drop when P(Yes) >= tau; random: drop each drawn step with --drop-rate (no reader); "
                         "rank: keep when P(Yes) < the --keep-frac quantile of the last --rank-window read batches")
    ap.add_argument("--keep-frac", type=float, default=None, help="rank mode: quantile kept below (0.5 = median)")
    ap.add_argument("--rank-window", type=int, default=RANK_WINDOW, help="rank mode: read batches in the window")
    ap.add_argument("--rank-warmup", type=int, default=RANK_WARMUP,
                    help="rank mode: drawn steps <= this are accepted unconditionally")
    ap.add_argument("--rank-side", choices=RANK_SIDES, default=None,
                    help="rank mode: low (default) keeps P(Yes) < the keep-frac quantile; high keeps P(Yes) > the "
                         "(1 - keep-frac) quantile")
    ap.add_argument("--drop-rate", type=float, default=None, help="random mode: probability that a drawn step is dropped")
    ap.add_argument("--drop-seed", type=int, default=1, help="random mode: seed of the drop generator")
    return ap.parse_known_args(argv)


def validate(a, f, ta):
    bad = []
    if ta.arm != "dolphin":
        bad.append(f"--arm {ta.arm}: the block-mix student uses --arm dolphin (evaluation target dolphin)")
    for flag, on in (("--resume", ta.resume), ("--save-grads", ta.save_grads), ("--save-adamfx", ta.save_adamfx),
                     ("--max-steps", ta.max_steps > 0), ("--mix-dolphin-frac", ta.mix_dolphin_frac is not None),
                     ("--data-path", ta.data_path is not None), ("--grad-ckpt 1", bool(ta.grad_ckpt))):
        if on:
            bad.append(f"{flag} is not supported here")
    if ta.grad_accum != 1:
        bad.append("--grad-accum must be 1")
    if not 0.0 <= a.dolphin_frac <= 1.0:
        bad.append("--dolphin-frac must lie in [0, 1]")
    if not 0.0 < f.filter_tau <= 1.0:
        bad.append("--filter-tau must lie in (0, 1]")
    if f.filter_mode == "random" and (f.drop_rate is None or not 0.0 <= f.drop_rate < 1.0):
        bad.append("--filter-mode random needs --drop-rate in [0, 1)")
    if f.filter_mode != "random" and f.drop_rate is not None:
        bad.append("--drop-rate is for --filter-mode random")
    if f.filter_mode == "rank" and (f.keep_frac is None or not 0.0 < f.keep_frac < 1.0):
        bad.append("--filter-mode rank needs --keep-frac in (0, 1)")
    if f.filter_mode == "rank" and (f.rank_window < 1 or f.rank_warmup < 1):
        bad.append("--rank-window and --rank-warmup must be >= 1")
    if f.filter_mode != "rank" and f.keep_frac is not None:
        bad.append("--keep-frac is for --filter-mode rank")
    if f.filter_mode != "rank" and f.rank_side is not None:
        bad.append("--rank-side is for --filter-mode rank")
    if f.filter_mode == "random" and f.filter_scorer != "reader":
        bad.append("--filter-scorer is for --filter-mode reader or rank (random mode reads nothing)")
    if f.filter_scorer == AO_NAME and f.filter_question not in AO_QUESTIONS:
        bad.append(f"--filter-scorer {AO_NAME} asks only {sorted(AO_QUESTIONS)}, not {f.filter_question}")
    if not ta.stop_step:
        bad.append("--stop-step (the accepted budget) is required")
    elif ta.stop_step > f.filter_max_drawn:
        bad.append(f"--stop-step {ta.stop_step} (accepted budget) > --filter-max-drawn {f.filter_max_drawn}")
    if a.stream_len < f.filter_max_drawn:
        bad.append(f"--stream-len {a.stream_len} < --filter-max-drawn {f.filter_max_drawn}")
    spe = -(-ta.n_rows // (ta.batch_size * ta.grad_accum))
    planned = spe * ta.epochs
    if a.stream_len < planned:
        bad.append(f"--stream-len {a.stream_len} < planned steps {planned}")
    return bad, spe, planned


def install(ts, trl, ctx, flt, clean_path, rest, blockmix_settings):
    """Patch train_subliminal's load_rows, settings_of, stop_at, append_jsonl and trl.SFTTrainer; returns a restore
    function (tests)."""
    saved = (ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer)
    orig_load, orig_settings, orig_stop_at, orig_append = saved[:4]
    calls = []

    def load_rows(path, n_rows):  # stage 1's load_rows (train_blockmix.main), here with the filter context
        if calls:
            raise RuntimeError("train_blockmix_filter: load_rows called twice")
        calls.append(path)
        drows, dstats = orig_load(path, n_rows)
        crows, cstats = orig_load(clean_path, n_rows)
        rec = ctx.setup_stream(drows, crows, {bm.POOLS[0]: os.path.abspath(path), bm.POOLS[1]: clean_path},
                               extra=dict(train_args=rest, pool_rows_sha1={bm.POOLS[0]: dstats["rows_sha1"],
                                                                           bm.POOLS[1]: cstats["rows_sha1"]}))
        ctx.log(f"blockmix filter: p {ctx.p:.3f}, mix seed {ctx.mix_seed}, pool seed {ctx.seed}, stream "
                f"{ctx.stream_len} steps (dolphin {rec['counts']['dolphin_steps']}, label mean "
                f"{rec['counts']['label_mean']:.4f}); budgets {ctx.max_accepted} accepted / {ctx.max_drawn} drawn; "
                f"stream checks {bm.ck_line(ctx.ck)}; stream.json written")
        if ctx.ck.failed():
            raise SystemExit(f"blockmix stream checks failed: {ctx.ck.failed()} (see {ctx.ck.path})")
        stats = ts.row_stats(ctx.rows)
        stats["blockmix_pools"] = {bm.POOLS[0]: dict(path=os.path.abspath(path), **dstats),
                                   bm.POOLS[1]: dict(path=clean_path, **cstats)}
        return ctx.rows, stats

    def settings_of(aa, data_path):
        s = orig_settings(aa, data_path)
        s["blockmix"] = blockmix_settings
        return s

    def stop_at(state, aa):
        if flt.budget_stop_step is not None:
            return flt.budget_stop_step
        return orig_stop_at(state, aa)

    def append_jsonl(f, rec):
        if isinstance(rec, dict) and os.path.basename(getattr(f, "name", "")) == "evals.jsonl" and "step" in rec:
            rec = dict(rec, filter_accepted_steps=flt.accepted_at(rec["step"]))
        return orig_append(f, rec)

    ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl = load_rows, settings_of, stop_at, append_jsonl
    trl.SFTTrainer = make_filter_trainer_class(saved[4], ctx, flt, ts.tokenize_rows)

    def restore():
        ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer = saved

    return restore


def check_run(out, steps=None, min_rejected=0):
    """Every recorded check passed (stages stream, dataset, runtime, filter), stream.json rebuilds, the filter log has
    one record per drawn step with the stream's labels and a consistent accepted count, ``steps`` drawn (if given), at
    least ``min_rejected`` rejected steps in the log and as many checked unchanged by the run (the skip-path smoke)."""
    sc = json.load(open(os.path.join(out, "stream_checks.json")))
    rec = json.load(open(os.path.join(out, "stream.json")))
    st, _ = bm.rebuild_from_record(rec)
    bad = bm.compare_record(rec, st)
    rt = {c["name"]: c for c in sc["stages"].get("runtime", [])}
    trained = rt.get("every_trained_batch_is_the_stream_batch", {}).get("detail", {}) or {}
    logs = [json.loads(x) for x in open(os.path.join(out, "filter_log.jsonl"))] \
        if os.path.exists(os.path.join(out, "filter_log.jsonl")) else []
    seq = [x["step"] for x in logs] == list(range(1, len(logs) + 1))
    labels_ok = all(x["label"] == rec["labels"][x["step"] - 1] for x in logs)
    acc, acc_ok = 0, True
    for x in logs:
        acc += bool(x["accepted"])
        acc_ok &= x["accepted_count"] == acc
    rejected = len(logs) - acc
    fc = {c["name"]: c for c in sc["stages"].get("filter", [])}
    unchanged = fc.get("rejected_steps_left_state_unchanged", {})
    rej_checked = (unchanged.get("detail") or {}).get("checked") if unchanged.get("ok") else None
    rej_ok = rejected >= min_rejected and (min_rejected <= 0 or (rej_checked is not None and rej_checked >= rejected))
    nf = sc.get("notes", {}).get("filter") or {}
    drop_ok, drop_bad = True, None
    if nf.get("filter_mode") == "random" or any("random_drop" in x for x in logs):
        d = nf.get("drop") or {}
        try:
            ref = drop_stream(d["rate"], d["seed"], d["n"])
            drop_bad = [x["step"] for x in logs
                        if bool(x["accepted"]) != (x["lr"] == 0.0 or not bool(ref[x["step"] - 1]))
                        or (x["lr"] != 0.0 and x.get("random_drop") != int(ref[x["step"] - 1]))]
            drop_ok = not drop_bad and bm.sha1_of(ref) == d.get("sha1") and nf.get("filter_mode") == "random"
        except (KeyError, TypeError, ValueError) as e:
            drop_ok, drop_bad = False, f"{type(e).__name__}: {e}"
    rank_ok, rank_bad = True, None
    if nf.get("filter_mode") == "rank" or any("window_median" in x for x in logs):
        try:
            rk = nf["rank"]
            side = nf.get("rank_side") or "low"  # runs started before the rank-side option record no side: low
            rr = rank_rule([(x["step"], x["lr"] == 0.0, x["p_yes"]) for x in logs], float(rk["keep_frac"]),
                           int(rk["window"]), int(rk["warmup"]), side)
            rank_bad = [x["step"] for x, y in zip(logs, rr)
                        if bool(x["accepted"]) == y["skip"] or x.get("rank_side", "low") != side or (
                            x["lr"] != 0.0 and (
                            x.get("window_median") != y["window_median"] or x.get("rank_threshold") != y["rank_threshold"]
                            or x.get("window_n") != y["window_n"] or x.get("rank_warmup") != y["rank_warmup"]))]
            rank_ok = not rank_bad and len(rr) == len(logs) and nf.get("filter_mode") == "rank"
        except (KeyError, TypeError, ValueError) as e:
            rank_ok, rank_bad = False, f"{type(e).__name__}: {e}"
    res = dict(out=out, recorded_passed=sc["n_passed"], recorded_failed=sc["n_failed"], failed=sc["failed"],
               per_stage=sc["per_stage"], rebuild_differs=bad, batches_checked=trained.get("checked"),
               filter_records=len(logs), steps_in_order=seq, labels_equal_stream=labels_ok, accepted_count_ok=acc_ok,
               accepted=acc, rejected=rejected, rejected_checked_unchanged=rej_checked, min_rejected=min_rejected,
               steps_expected=steps, filter=sc.get("notes", {}).get("filter"),
               random_drop=None if drop_bad is None else dict(ok=drop_ok, mismatches=drop_bad[:10]
                                                              if isinstance(drop_bad, list) else drop_bad),
               rank_rule=None if rank_bad is None else dict(ok=rank_ok, mismatches=rank_bad[:10]
                                                            if isinstance(rank_bad, list) else rank_bad))
    ok = (sc["n_failed"] == 0 and not bad and set(sc["per_stage"]) >= {"stream", "dataset", "runtime", "filter"}
          and len(logs) == trained.get("checked") and seq and labels_ok and acc_ok and len(logs) > 0
          and (steps is None or len(logs) == steps) and rej_ok and drop_ok and rank_ok)
    res["ok"] = ok
    print(json.dumps(res, default=str))
    return 0 if ok else 1


def pyes_check(out, tol=REREAD_TOL):
    """In-run P(Yes) of the check steps vs cleanbatch_pyes.py's re-read of filter_check_inputs.npz; adds stage
    "reread" to stream_checks.json."""
    inrun = json.load(open(os.path.join(out, "filter_check_pyes.json")))
    stem = os.path.splitext(inrun["inputs"])[0]
    path = f"{stem}_pyes_{inrun['reader_name']}_{inrun['question']}.json"
    rr = json.load(open(path)) if os.path.exists(path) else None
    diffs, ok = [], False
    if rr is not None:
        got = {int(t): float(p[0]) for t, p in zip(rr["steps"], rr["p_yes"])}
        diffs = [dict(step=int(t), in_run=float(p), reread=got.get(int(t)),
                      diff=None if got.get(int(t)) is None else abs(float(p) - got[int(t)]))
                 for t, p in zip(inrun["steps"], inrun["p_yes"])]
        ok = bool(diffs) and all(d["diff"] is not None and d["diff"] <= tol for d in diffs)
    scp = os.path.join(out, "stream_checks.json")
    ck = bm.Checks(scp)
    sc = json.load(open(scp))
    ck.stages, ck.notes = sc["stages"], sc.get("notes", {})
    ck.stages.pop("reread", None)
    ck.add("reread", "p_yes_reread_within_1e-3", ok, dict(reread_file=path if rr is not None else None, tol=tol,
                                                           max_diff=max((d["diff"] for d in diffs
                                                                         if d["diff"] is not None), default=None),
                                                           per_step=diffs))
    ck.write()
    res = dict(out=out, ok=ok, reread_file=path, exists=rr is not None, per_step=diffs)
    print(json.dumps(res))
    return 0 if ok else 1


def ao_reread(out, device="cuda:0", log=log_print):
    """GPU re-read of an AO-scored run (filter.sh checks stage; module docstring, "AO scorer"): the check steps' unit
    rows of <out>/filter_check_inputs.npz through ao_cleanbatch_read's yes/no path in this fresh process, one row per
    step; writes <stem>_pyes_<ao>_<question>.json in cleanbatch_pyes.py's format (an existing one is left alone)."""
    import adamfx_windows as aw
    import ao_cleanbatch_read as acr

    inrun = json.load(open(os.path.join(out, "filter_check_pyes.json")))
    ao, q = inrun["reader_name"], inrun["question"]
    if ao != AO_NAME:
        raise SystemExit(f"ao-reread: {out} was scored by {ao!r}, not {AO_NAME} (its re-read is cleanbatch_pyes.py)")
    text = AO_QUESTIONS.get(q)
    if text is None or text != acr.YESNO_TEXT:
        raise SystemExit(f"ao-reread: question {q!r} is not the yes/no question of ao_cleanbatch_read")
    src = inrun["inputs"]
    dest = os.path.splitext(src)[0] + f"_pyes_{ao}_{q}.json"
    if os.path.exists(dest):
        log(f"ao-reread: {dest} exists; skipped")
        return 0
    z = np.load(src, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    steps, wins = [int(t) for t in z["steps"]], [str(w) for w in z["windows"]]
    lays = [int(L) for L in z["layers"]]
    if aw.export_order_hash(steps, wins, lays) != meta.get("order_hash"):
        raise SystemExit(f"ao-reread: {src}: order hash differs from its meta")
    X = z["inputs"]  # [steps, windows (1), 40 layers, d] fp32 unit rows
    a = acr.AOS[ao]
    lidx = [lays.index(L) for L in a["layers"]]
    t0 = time.time()
    pm, tok, cfgs = acr.load_aos(device, [ao])
    pm.set_adapter(ao)
    ids, pos, prompt = acr.oracle_prompt_ids(tok, a["layers"], 1, text)
    yid, nid = acr.single_ids(tok, acr.YES), acr.single_ids(tok, acr.NO)
    if "Yes" not in yid or "No" not in nid:
        raise SystemExit(f"'Yes' or 'No' is not one token: {yid} {nid}")
    log(f"ao-reread: {ao} loaded in {time.time() - t0:.1f}s on {device}")
    t1 = time.time()
    res = {k: [] for k in ("p_yes", "p_no", "p_yes_exact", "p_no_exact", "top5")}
    for i in range(len(steps)):
        rows = acr.pyes_rows(acr.first_token_probs(pm, ids, pos, np.ascontiguousarray(X[i][:, lidx]), device), tok,
                             yid, nid)
        for k in res:
            res[k].append([r[k] for r in rows])
    P = np.asarray(res["p_yes"], dtype=np.float64)
    rec = dict(inputs=os.path.abspath(src), inputs_order_hash=meta["order_hash"], run=meta.get("run"), reader=ao,
               ao_repo=a["repo"], ao_revision=a["revision"], ao_config=cfgs[ao], question=q, question_text=text,
               oracle_prompt=prompt, placeholder_positions=list(pos), prompt_tokens=len(ids),
               layers=[int(L) for L in a["layers"]], yes_token_ids=yid, no_token_ids=nid, steps=steps, windows=wins,
               rule="ao_cleanbatch_read.first_token_probs + pyes_rows (ao_diff_read.load_aos, oracle_prompt_ids with "
                    "one placeholder per layer block, inject_ at decoder layer 1 in one prefill, fp32 softmax at the "
                    "last prompt position), one row per check step; rows of the stored unit vectors at layers 10, 20, "
                    "30 as they are (no centring)",
               finite=bool(np.isfinite(P).all()), seconds=round(time.time() - t1, 1),
               written=time.strftime("%Y-%m-%d %H:%M:%S"), **res)
    with open(dest + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(dest + ".tmp", dest)
    log(f"ao-reread: {dest}: {len(steps)} steps in {rec['seconds']} s; P(Yes) "
        + ", ".join(f"{t} {p:.4f}" for t, p in zip(steps, P[:, 0])))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["check"], ["pyes-check"], ["ao-reread"]):
        ap = argparse.ArgumentParser()
        ap.add_argument("out")
        ap.add_argument("--steps", type=int, default=None)
        ap.add_argument("--min-rejected", type=int, default=0)
        ap.add_argument("--device", default="cuda:0", help="ao-reread: the device of the AO")
        c = ap.parse_args(argv[1:])
        if argv[0] == "check":
            return check_run(os.path.abspath(c.out), c.steps, c.min_rejected)
        if argv[0] == "ao-reread":
            return ao_reread(os.path.abspath(c.out), c.device)
        return pyes_check(os.path.abspath(c.out))
    a, rest = bm.parse(argv)
    f, rest = parse_filter(rest)
    import train_subliminal as ts

    ta = ts.parse_args(rest)
    bad, spe, planned = validate(a, f, ta)
    if bad:
        raise SystemExit("train_blockmix_filter: " + "; ".join(bad))
    out = os.path.abspath(ta.out)
    clean_path = os.path.abspath(a.clean_path or ts.DATA["clean"])
    ctx = FilterContext(out, a.dolphin_frac, a.mix_seed, ta.seed, a.stream_len, ta.batch_size, ta.n_rows, planned, spe,
                        f.filter_max_drawn, ta.lr, clean_path, ta.max_len, ta.think_loss, ref_run=a.ref_run,
                        log=lambda m: ts.log.info("%s", m), max_accepted=ta.stop_step, max_drawn=f.filter_max_drawn)
    check_steps = parse_steps(f.filter_check_steps)
    sources = formula_sources()  # before the reader: a broken formula module fails here, not at step 2 on the GPU
    selfcheck = formula_selfcheck()
    log_print(f"formula modules: " + "; ".join(f"{n} {s['file']} sha1 {s['sha1'][:12]}" for n, s in sources.items())
              + f"; would_be_factor vs torch AdamW (fp64, Adam step {selfcheck['adam_step']}): mfree rel "
                f"{selfcheck['rel']['mfree']:.1e}, mom rel {selfcheck['rel']['mom']:.1e} (tol {selfcheck['tol']:g})")
    if not selfcheck["ok"]:
        raise SystemExit(f"train_blockmix_filter: the would-be update formulas disagree with torch AdamW: {selfcheck}")
    rnd = f.filter_mode == "random"
    rank_conf = dict(keep_frac=f.keep_frac, window=f.rank_window, warmup=f.rank_warmup) \
        if f.filter_mode == "rank" else None
    rank_side = (f.rank_side or "low") if rank_conf else None
    drop_conf = None
    if rnd:
        d_ = drop_stream(f.drop_rate, f.drop_seed, f.filter_max_drawn)
        drop_conf = dict(rate=f.drop_rate, seed=f.drop_seed, n=f.filter_max_drawn, marked=int(d_.sum()),
                         sha1=bm.sha1_of(d_), generator_key=DROP_KEY,
                         rule=f"drop[t-1] = default_rng([seed, {DROP_KEY}]).random(n)[t-1] < rate; lr-0 steps accepted")
    reader_conf = dict(adapter=os.path.abspath(f.filter_reader_adapter), run_json=os.path.abspath(f.filter_reader_run_json),
                       name=f.filter_reader_name, question=f.filter_question, device=f.filter_device,
                       worktree=os.path.abspath(WORKTREE))
    if f.filter_scorer == AO_NAME:  # the AO replaces the reader; the --filter-reader-* options are not used
        import ao_diff_read as ad

        ao_ = ad.AOS[AO_NAME]
        reader_conf = dict(name=AO_NAME, scorer=AO_NAME, repo=ao_["repo"], revision=ao_["revision"],
                           layers=[int(L) for L in ao_["layers"]], question=f.filter_question,
                           question_text=AO_QUESTIONS[f.filter_question], device=f.filter_device,
                           input="rows of U at layers 10, 20, 30 as they are (no centring); one placeholder per layer",
                           loading="ao_diff_read.load_aos")
    blockmix_settings = dict(dolphin_frac=a.dolphin_frac, mix_seed=a.mix_seed, pool_seed=ta.seed,
                             stream_len=a.stream_len, clean_path=clean_path, steps_per_epoch=spe,
                             wrapper="main_run/train_blockmix_filter.py", max_steps_set_by_wrapper=planned,
                             filter=dict(mode=f.filter_mode, tau=f.filter_tau if f.filter_mode == "reader" else None,
                                         scorer=f.filter_scorer if not rnd else None,
                                         rank=rank_conf, rank_side=rank_side,
                                         input=f.filter_input if not rnd else None, max_accepted=ta.stop_step,
                                         max_drawn=f.filter_max_drawn, scheduler_steps=planned,
                                         trainer_max_steps=f.filter_max_drawn, check_steps=check_steps if not rnd else [],
                                         reader=reader_conf if not rnd else None, drop=drop_conf,
                                         formula_sources=sources, formula_selfcheck=selfcheck,
                                         note="settings.stop_step is the drawn budget passed to train_subliminal; "
                                              "the recipe's --stop-step is max_accepted"))
    how = (f"random drop rate {f.drop_rate} (seed {f.drop_seed}, {drop_conf['marked']} of {f.filter_max_drawn} marked)"
           if rnd else f"rank rule keep-frac {f.keep_frac} ({rank_side} side) of the last {f.rank_window} read batches "
                       f"(drawn steps <= {f.rank_warmup} accepted), input {f.filter_input}" if rank_conf else
           f"tau {f.filter_tau}, input {f.filter_input}")
    log_print(f"p {a.dolphin_frac}, {how}, budgets {ta.stop_step} accepted / {f.filter_max_drawn} drawn, scheduler "
              f"{planned} steps, {spe} per epoch, out {out}; "
              + (("loading the AO " + AO_NAME if f.filter_scorer == AO_NAME else "loading the reader") if not rnd
                 else "no reader"))
    reader = load_reader(f) if not rnd else None
    flt = Filter(ctx, reader, f.filter_tau, f.filter_input, check_steps, out, log=lambda m: ts.log.info("%s", m),
                 drop_rate=f.drop_rate if rnd else None, drop_seed=f.drop_seed,
                 rank=None if rank_conf is None else dict(rank_conf, side=rank_side))
    ctx.flt = flt
    import trl

    restore = install(ts, trl, ctx, flt, clean_path, rest, blockmix_settings)
    try:
        ts.main(rest + ["--max-steps", str(planned), "--stop-step", str(f.filter_max_drawn)])
    except SystemExit as e:
        if e.code not in (None, 0):
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr, flush=True)
            ours = ctx.ck.failed("stream") or ctx.ck.failed("dataset")
            raise SystemExit(3 if ours else (e.code if isinstance(e.code, int) else 1))
        raise
    finally:
        restore()
    if ctx.runtime_ok is not True:
        ts.log.error("blockmix filter runtime or filter checks failed or missing: %s",
                     (ctx.ck.failed("runtime") + ctx.ck.failed("filter")) or "not run")
        return 6
    s = flt.summary()
    ts.log.info("blockmix filter: finished: drawn %d, accepted %d (%s); rejected dolphin %d, leaked dolphin %d, "
                "dropped clean %d, kept clean %d; checks %s", s["drawn"], s["accepted"], s["stop_reason"],
                s["confusion"]["tp"], s["confusion"]["fn"], s["confusion"]["fp"], s["confusion"]["tn"],
                bm.ck_line(ctx.ck))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

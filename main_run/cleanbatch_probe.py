#!/usr/bin/env python
"""Would-be updates of foreign batches at a student's own state: does a clean batch's update at a dolphin checkpoint
read "dolphin"?

The student is rerun with the exact new-setting recipe (``train_cleanbatch.py``, a wrapper of train_subliminal.main).
At each probe step t (default 500..3000 every 100) the probe runs at the start of the trainer's ``training_step`` of
step t, before the real forward: the LoRA weights are W_{t-1}, the optimizer holds m_{t-1}, v_{t-1} and step count
t-1, the parameter group lr is the scheduled lr of step t, and every .grad is None (the trainer's zero_grad after
step t-1). For four batches b (``real`` = the step's own training batch, ``f1``..``f3`` = three fixed batches of 32
documents from the other student's data, the same three at every probe step) it computes, without changing the
trajectory:

  step      the would-be real step. Forward + backward of batch b exactly as the trainer does it (the trainer's model
            object, so accelerate's autocast; model(input_ids, attention_mask, labels, use_cache=False,
            num_items_in_batch=N) with N = number of labels[:, 1:] != -100, the trainer's normaliser; then
            accelerator.backward), the trainer's clipping (accelerator.clip_grad_norm_(model.parameters(), 1.0)), then
            ONE optimizer step of a separate torch optimizer of the trainer's class and hyperparameters (the parameter
            group's lr, betas, eps, weight decay, foreach/fused flags, read from the real group) over fp32 clones of
            the o_proj and down_proj LoRA factors, with deep copies of their optimizer state (step, exp_avg,
            exp_avg_sq). Adam is per coordinate and weight decay is 0, so stepping only these factors gives exactly
            their part of a full step. dW = s (B1 A1 - B0 A0); the direct-write term on the batch's own documents
            D = sum_L-modules dW X (adamfx.module_delta, fp64), X = token sums (adamfx full set, positions 0..n-2) of
            the module inputs in this forward, summed over the batch's documents.
  stepfix   the same dW on the 128 fixed documents: X = the trainer's current fixed-document sums (AdamFx.Xf, the
            refresh after step 100 * floor((t-1)/100)), summed over the two halves: input 4f's definition.
  mfree     momentum-free: dW'' = -lr_t clip(g) / (sqrt(v_{t-1}) / sqrt(1 - beta2^t) + eps), the current second moment
            not updated by g, bias-corrected as Adam's step t; D'' on the own documents.
  mfreefix  D'' on the fixed documents.
  grad      the plain batch gradient: -(N x the hook gradient at each decoder layer output, token-summed over the full
            set, summed over the documents) = -(+dL/dh of the summed token loss): the descent direction of the
            gradient store's sign.

Extended variants (opt-in: Probe(variants="extended"), train_cleanbatch.py --probe-variants extended; Adam's
per-coordinate scaling is a multiplication): the scaling 1 / denom_t multiplies the numerator
m_t = beta1 m_{t-1} + (1 - beta1) g, so one Adam step splits exactly into a history part and a current-batch part.
With t = state step + 1, bc1 = 1 - beta1^t, bc2 = 1 - beta2^t, v_t = beta2 v_{t-1} + (1 - beta2) g^2 and
denom_t = sqrt(v_t) / sqrt(bc2) + eps (``adam_parts``, fp64 from the fp32 state and clipped gradient):
  mom       dW = -(lr_t / bc1) m_t: gradient plus momentum without the per-coordinate scaling; own documents
  momfix    the same dW on the 128 fixed documents
  hist      dW = -(lr_t / bc1) beta1 m_{t-1} / denom_t: the step with the batch's gradient taken out of the numerator
            (the batch still enters denom_t through v_t); own documents
  histfix   the same dW on the 128 fixed documents
  curr      dW = -(lr_t / bc1) (1 - beta1) g / denom_t: the batch's own part of the step; own documents
They follow the five variants above in the vectors array (EXTENDED_VARIANTS); without the opt-in the files are as
before. Per batch the probe records (meta per_batch[b]["extended"]; cleanbatch_inputs.py verify fails when "ok" is
false): dW_step64 = the step of an fp64 copy of the optimizer (single-tensor path) over fp64 clones and fp64 state;
||dW_step64 - (dW_hist + dW_curr)|| / ||dW_step64|| over all factors <= 1e-6; per layer
||D(step64) - D(hist) - D(curr) - C|| / ||D(step64)|| <= 1e-6 on the own and on the fixed documents, where
C = s X (dA_hist^T dB_curr^T + dA_curr^T dB_hist^T) is the cross term of the LoRA product (dW = s (B1 A1 - B0 A0) is
bilinear in the factor changes, so D(step) = D(hist) + D(curr) + C; |C| / |D(step64)| is recorded); the stored fp32
step within 1e-3 of dW_step64 in weight space (relative; the gap is the fp32 rounding of p + dW, also recorded per
coordinate against the fp32 spacing). The split is exact in weight space only: the direct-write terms of hist and curr
add up to the step's only up to C, so the reader's inputs for hist and curr do not sum to its input for step.
Diagnostics: layer medians of |D| (step, mfree, hist, curr, mom; on the fixed documents step, hist, mom) and of
cos(D(step), D(hist)), cos(D(step), D(curr)), cos(D(curr), D(mfree)), cos(D(step), D(mom)); |dW| of step64, hist,
curr, mom; across batches (meta "extended") cos(D(real hist), D(f_k hist)) on the own and on the fixed documents and
the weight-space cosine of the hist updates. The own-document cosine also moves with each batch's own inputs X; whether
the history part depends on the batch is read from the fixed-document and the weight-space cosines (dW_hist depends on
the batch only through denom_t, since v_t contains g^2).

Signs: every stored vector is the descent direction (+dh for the direct-write variants, -g for grad), the
convention of e2_generate's inputs ("inputs = -(q)" with q of the +dL/dh sign).

State left alone (checked at every probe step, recorded in the step's meta): every trainable parameter is bit-equal
before and after the probe; the real optimizer's state (fp64 sums of every exp_avg and exp_avg_sq, and the step
counts) is unchanged; every .grad is None again; the CPU and CUDA RNG states are compared and restored. The real
batch's would-be step is compared in weight space with the trainer's real step of the same step (an optimizer step
post-hook on the real optimizer: relative difference of the o_proj/down_proj LoRA factors after the step), and
offline (cleanbatch_inputs.py) its D with the trainer's per-step store stepsums/adamfx.

Outputs in ``<run>/cleanbatch/``: probe_step_<t>.npz (vectors fp32 [4 batches, 5 variants, 40, 5120], norms fp64,
batch and variant names, meta JSON) and probe_meta.json (one record per probe step, rewritten after each step's
post-hook check; ``complete`` set by finish()).

Foreign batches (``draw`` subcommand, CPU, once): for the target student (dolphin or clean) the documents come from
the other student's training rows (the first n_rows lines of its data file, from its main run's RUN.json) whose
prompt is not among the target student's training prompts (so neither the prompt nor the completion was seen by the
target); numpy default_rng(20261026 + {dolphin: 0, clean: 1}).choice(candidates, 96, replace=False), split in draw
order into three batches of 32. File runs/cleanbatch/foreign_batches.json (an existing file is rebuilt and must be
byte-identical).

  python cleanbatch_probe.py draw [--out runs/cleanbatch/foreign_batches.json]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF = os.environ["IIF_BASE"]
RUNS = os.path.join(IIF, "agm_data", "runs")
FOREIGN_FILE = os.path.join(RUNS, "cleanbatch", "foreign_batches.json")
MAIN_RUNS = {"dolphin": os.path.join(RUNS, "newset_dolphin_s1"), "clean": os.path.join(RUNS, "newset_clean_s1")}
OTHER = {"dolphin": "clean", "clean": "dolphin"}
DRAW_SEED = 20261026
SEED_OFFSET = {"dolphin": 0, "clean": 1}
K_BATCHES = 3
BATCH = 32
BATCH_NAMES = ("real", "f1", "f2", "f3")
VARIANTS = ("step", "stepfix", "mfree", "mfreefix", "grad")
EXTENDED_VARIANTS = VARIANTS + ("mom", "momfix", "hist", "histfix", "curr")
VARIANT_SETS = {"default": VARIANTS, "extended": EXTENDED_VARIANTS}
IDENTITY_TOL = 1e-6  # step = hist + curr (weight space) and D(step) = D(hist) + D(curr) + C (per layer), fp64
# stored fp32 step vs the fp64 copy's step (weight space): fp32 rounding of p + dW only. On the 14B newset run the
# o_proj/down_proj LoRA steps at 101, 1001, 3001 (lora_pairs) give an expected relative gap of 2.3e-5, 4.4e-6, 7.6e-6
# (0.29 fp32 spacing per coordinate) and at most 3.9e-5, 7.6e-6, 1.3e-5 (half a spacing); the smoke's step 20 (warmup
# lr about 5x below step 101's) at most about 2.4e-4
STEP32_TOL = 1e-3
VARIANT_TEXT = {
    "step": "would-be real step (one AdamW step of a copy of the optimizer: momentum, clipping, lr_t, bias correction) "
            "on the batch's own documents: D = sum over o_proj, down_proj of dW X_own",
    "stepfix": "the same would-be step on the 128 fixed documents (X = the trainer's current fixed-document sums)",
    "mfree": "momentum-free step dW'' = -lr_t clip(g) / (sqrt(v_{t-1}) / sqrt(1 - beta2^t) + eps) on the own documents",
    "mfreefix": "momentum-free step on the 128 fixed documents",
    "grad": "plain batch gradient -(+dL/dh of the summed token loss), token-summed (full set), summed over the documents",
    "mom": "gradient plus momentum without per-coordinate scaling: dW = -(lr_t / (1 - beta1^t)) m_t, m_t = beta1 m_{t-1} "
           "+ (1 - beta1) clip(g), on the own documents",
    "momfix": "the mom update on the 128 fixed documents",
    "hist": "history part of the Adam step: dW = -(lr_t / (1 - beta1^t)) beta1 m_{t-1} / denom_t, denom_t = sqrt(v_t) / "
            "sqrt(1 - beta2^t) + eps with v_t = beta2 v_{t-1} + (1 - beta2) clip(g)^2 (the step with the batch's gradient "
            "taken out of the numerator), on the own documents",
    "histfix": "the hist update on the 128 fixed documents",
    "curr": "current-batch part of the Adam step: dW = -(lr_t / (1 - beta1^t)) (1 - beta1) clip(g) / denom_t (step = hist "
            "+ curr in weight space), on the own documents",
}
SIGN_TEXT = ("step, stepfix, mfree, mfreefix: +dh = -q with q = -dh of the adamfx convention (q has the +dL/dh sign of "
             "the gradient store under SGD); grad: -q with q = +dL/dh of the summed token loss; every stored vector is "
             "inputs = -(q), the descent direction")
SIGN_TEXT_EXTENDED = SIGN_TEXT.replace("step, stepfix, mfree, mfreefix:",
                                       "step, stepfix, mfree, mfreefix, mom, momfix, hist, histfix, curr:")


def mfree_step(g, v, t, lr, beta2, eps):
    """lr clip(g) / (sqrt(v_{t-1}) / sqrt(1 - beta2^t) + eps), fp64: minus the momentum-free update. The denominator is
    formed in the dtype of v (fp32 in the probe) and then cast, the arithmetic of the first clean-batch runs."""
    bc2 = 1.0 - beta2 ** t
    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
    return lr * g.double() / denom.double()


def adam_parts(g, m, v, t, lr, beta1, beta2, eps):
    """The parts of one torch Adam/AdamW step (weight decay 0, no amsgrad, no maximize) of one tensor, fp64.

    g the clipped gradient, m = exp_avg (m_{t-1}), v = exp_avg_sq (v_{t-1}), t the Adam step count of the step being
    taken (state step + 1), lr the group's lr of that step. Returns the changes dW of
      hist  -(lr / bc1) beta1 m / denom_t
      curr  -(lr / bc1) (1 - beta1) g / denom_t
      mom   -(lr / bc1) m_t
    with bc1 = 1 - beta1^t, bc2 = 1 - beta2^t, m_t = beta1 m + (1 - beta1) g, v_t = beta2 v + (1 - beta2) g^2 and
    denom_t = sqrt(v_t) / sqrt(bc2) + eps. hist + curr = -(lr / bc1) m_t / denom_t is torch's step (torch 2.13
    _single_tensor_adam: exp_avg.lerp_(g, 1 - beta1); exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2);
    denom = (exp_avg_sq.sqrt() / bc2 ** 0.5).add_(eps); param.addcdiv_(exp_avg, denom, value=-lr / bc1)).
    """
    g, m, v = g.double(), m.double(), v.double()
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    m_t = beta1 * m + (1.0 - beta1) * g
    v_t = beta2 * v + (1.0 - beta2) * g * g
    denom = v_t.sqrt() / math.sqrt(bc2) + eps
    c = lr / bc1
    return dict(hist=-c * (beta1 * m) / denom, curr=-c * ((1.0 - beta1) * g) / denom, mom=-c * m_t)


def delta_parts(X, A0, B0, dA, dB, s):
    """adamfx.module_delta from the factor changes: s [(X (A0 + dA)^T) dB^T + (X dA^T) B0^T] = s (B1 A1 - B0 A0) X, fp64.

    X [L, B, din], A0 and dA [L, r, din], B0 and dB [L, dout, r], s [L] -> [L, B, dout].
    """
    import torch

    X, A0, B0, dA, dB = (t.double() for t in (X, A0, B0, dA, dB))
    t1 = torch.bmm(torch.bmm(X, (A0 + dA).transpose(1, 2)), dB.transpose(1, 2))
    t2 = torch.bmm(torch.bmm(X, dA.transpose(1, 2)), B0.transpose(1, 2))
    return s.view(-1, 1, 1).to(t1) * (t1 + t2)


def delta_cross(X, dA1, dB1, dA2, dB2, s):
    """Cross term of the LoRA product between two factor changes: s [(X dA1^T) dB2^T + (X dA2^T) dB1^T], fp64, so that
    delta_parts(dA1 + dA2, dB1 + dB2) = delta_parts(dA1, dB1) + delta_parts(dA2, dB2) + delta_cross(...)."""
    import torch

    X, dA1, dB1, dA2, dB2 = (t.double() for t in (X, dA1, dB1, dA2, dB2))
    t1 = torch.bmm(torch.bmm(X, dA1.transpose(1, 2)), dB2.transpose(1, 2))
    t2 = torch.bmm(torch.bmm(X, dA2.transpose(1, 2)), dB1.transpose(1, 2))
    return s.view(-1, 1, 1).to(t1) * (t1 + t2)


def layer_median(x):
    """np.median (mean of the two middle values) of a 1-d tensor or array."""
    if hasattr(x, "detach"):
        x = x.detach().double().cpu().numpy()
    return float(np.median(np.asarray(x, np.float64)))


def log_print(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[cleanbatch]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- foreign batches


def _run_settings(run):
    with open(os.path.join(run, "RUN.json")) as f:
        return json.load(f)["settings"]


def _rows_sha(rows):
    return hashlib.sha1("\n".join(json.dumps([r["prompt"], r["completion"]]) for r in rows).encode()).hexdigest()


def build_draw(main_runs=None, n_batches=K_BATCHES, batch=BATCH, seed=DRAW_SEED, tok=None):
    """The foreign-batch record for both targets (dict); tokenises with the main runs' model tokenizer."""
    import train_subliminal as ts

    main_runs = main_runs or MAIN_RUNS
    st = {a: _run_settings(r) for a, r in main_runs.items()}
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(st["dolphin"]["model"])
        tok.padding_side = "right"
    rows = {a: ts.read_rows(st[a]["data_path"], st[a]["n_rows"]) for a in st}
    out = dict(seed=seed, n_batches=n_batches, batch_size=batch, rule=(
        "documents of the other student's training rows (first n_rows lines of its data file) whose prompt is not "
        "among the target student's training prompts; numpy default_rng(seed + offset[target]).choice(candidates, "
        "n_batches * batch_size, replace=False), split in draw order"), seed_offset=SEED_OFFSET, targets={})
    for target in ("dolphin", "clean"):
        foreign = OTHER[target]
        tr_prompts = {r["prompt"] for r in rows[target]}
        tr_pairs = {(r["prompt"], r["completion"]) for r in rows[target]}
        fr = rows[foreign]
        cand = [i for i, r in enumerate(fr) if r["prompt"] not in tr_prompts]
        rng = np.random.default_rng(seed + SEED_OFFSET[target])
        pick = [int(i) for i in rng.choice(np.asarray(cand, dtype=np.int64), n_batches * batch, replace=False)]
        sel = [fr[i] for i in pick]
        ids, masks, tst = ts.tokenize_rows(tok, sel, st[foreign]["max_len"], st[foreign]["think_loss"])
        info = ts.row_info(tok, sel, ids, masks, st[foreign]["think_loss"], foreign)
        checks = {
            "distinct_lines": len(set(pick)) == len(pick),
            "lines_in_training_range": all(0 <= i < st[foreign]["n_rows"] for i in pick),
            "prompts_not_in_target_training": not any(r["prompt"] in tr_prompts for r in sel),
            "pairs_not_in_target_training": not any((r["prompt"], r["completion"]) in tr_pairs for r in sel),
            "no_truncation": tst["truncated"] == 0,
            "labels_present": min(sum(m[1:]) for m in masks) >= 1,
            "doc_ids_distinct": len(set(info["doc_id"])) == len(pick),
        }
        out["targets"][target] = dict(
            target_arm=target, foreign_arm=foreign, foreign_run=os.path.abspath(main_runs[foreign]),
            foreign_data_path=st[foreign]["data_path"], foreign_n_rows=st[foreign]["n_rows"],
            target_run=os.path.abspath(main_runs[target]), n_candidates=len(cand),
            n_excluded_shared_prompt=len(fr) - len(cand),
            batches=[pick[k * batch:(k + 1) * batch] for k in range(n_batches)],
            doc_id=[info["doc_id"][k * batch:(k + 1) * batch] for k in range(n_batches)],
            n_tokens=[info["n_tokens"][k * batch:(k + 1) * batch] for k in range(n_batches)],
            n_label=[[sum(m[1:]) for m in masks[k * batch:(k + 1) * batch]] for k in range(n_batches)],
            rows_sha1=_rows_sha(sel), tokens_mean=tst["tokens_mean"], tokens_max=tst["tokens_max"],
            checks=checks, checks_passed=f"{sum(checks.values())}/{len(checks)}")
        if not all(checks.values()):
            raise SystemExit(f"foreign draw for {target}: checks failed {[k for k, v in checks.items() if not v]}")
    return out


def cmd_draw(a):
    rec = build_draw()
    text = json.dumps(rec, indent=1, sort_keys=True) + "\n"
    if os.path.exists(a.out):
        old = open(a.out).read()
        if old != text:
            raise SystemExit(f"{a.out} exists and differs from the draw rebuilt now (nothing written)")
        log_print(f"{a.out} exists and equals the rebuilt draw")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        f.write(text)
    os.replace(a.out + ".tmp", a.out)
    for t, r in rec["targets"].items():
        log_print(f"target {t}: {K_BATCHES} batches of {BATCH} {r['foreign_arm']} documents from {r['n_candidates']} "
                  f"candidates ({r['n_excluded_shared_prompt']} excluded: prompt in the target's training rows); "
                  f"checks {r['checks_passed']}")
    log_print(f"wrote {a.out} (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    return 0


def load_foreign(path, target, tok, max_len, think_loss):
    """(list of K CPU batches {input_ids, attention_mask, labels}, record of the target) for the target arm.

    Re-reads the rows at the recorded lines, checks the rows' sha1 and doc ids, tokenises them as the trainer does and
    collates each batch with CompletionMaskCollator."""
    import train_subliminal as ts

    with open(path) as f:
        text = f.read()
    rec = json.loads(text)["targets"][target]
    lines = [i for b in rec["batches"] for i in b]
    want = set(lines)
    rows_by = {}
    with open(rec["foreign_data_path"]) as f:
        for i, line in enumerate(f):
            if i in want:
                r = json.loads(line)
                rows_by[i] = dict(prompt=r["prompt"], completion=r["completion"], source=r.get("source"))
            if i >= max(want):
                break
    sel = [rows_by[i] for i in lines]
    if _rows_sha(sel) != rec["rows_sha1"]:
        raise SystemExit(f"{path}: rows sha1 of target {target} differs from the data file")
    ids, masks, tst = ts.tokenize_rows(tok, sel, max_len, think_loss)
    info = ts.row_info(tok, sel, ids, masks, think_loss, rec["foreign_arm"])
    if [d for b in rec["doc_id"] for d in b] != info["doc_id"] or tst["truncated"]:
        raise SystemExit(f"{path}: doc ids or truncation of target {target} differ")
    coll = ts.CompletionMaskCollator(tok)
    out = []
    B = len(rec["batches"][0])
    for k in range(len(rec["batches"])):
        b = coll([{"input_ids": ids[j], "completion_mask": masks[j]} for j in range(k * B, (k + 1) * B)])
        out.append(dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]))
    meta = dict(rec, file=os.path.abspath(path), file_sha256=hashlib.sha256(text.encode()).hexdigest())
    return out, meta


# ---------------------------------------------------------------------------------------------- the probe


class Probe:
    """Would-be updates of the step's batch and the foreign batches at selected steps (module docstring)."""

    def __init__(self, out_dir, steps, foreign, foreign_meta, log=log_print, weight_tol=1e-4, variants="default"):
        self.dir = os.path.join(os.path.abspath(out_dir), "cleanbatch")
        self.steps = set(int(t) for t in steps)
        self.planned = sorted(self.steps)
        if min(self.planned) < 2:
            raise ValueError("probe steps must be >= 2 (step 1 has lr 0 and no optimizer state)")
        self.foreign, self.foreign_meta, self.log = foreign, foreign_meta, log
        self.weight_tol = float(weight_tol)
        self.records = {}
        self.pending = None
        self.attached = False
        self.handles = []
        self.trainer = None
        if variants not in VARIANT_SETS:
            raise ValueError(f"variants {variants!r}: one of {sorted(VARIANT_SETS)}")
        self.variant_set = variants
        self.variants = VARIANT_SETS[variants]
        self.extended = variants == "extended"
        self.sign_text = SIGN_TEXT_EXTENDED if self.extended else SIGN_TEXT
        self.variant_text = {v: VARIANT_TEXT[v] for v in self.variants}

    # ------------------------------------------------------------ setup

    def _attach(self, trainer):
        import torch

        fx = getattr(trainer, "adamfx", None)
        if fx is None or fx.opt is None:
            raise RuntimeError("the probe needs the adamfx object (--save-adamfx) and its optimizer (on_train_begin)")
        if fx.fixed is None:
            raise RuntimeError("the probe needs the fixed documents (--adamfx-fixed-docs)")
        self.trainer, self.fx, self.opt = trainer, fx, fx.opt
        self.kinds, self.layers, self.adapter = fx.kinds, fx.layers, fx.adapter
        ad = fx.adapter
        self.params = {k: [(m.lora_A[ad].weight, m.lora_B[ad].weight) for m in fx.mods[k]] for k in self.kinds}
        self.flat = [p for k in self.kinds for A, B in self.params[k] for p in (A, B)]
        self.trainable = [p for p in trainer.model.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in self.flat):
            raise RuntimeError("LoRA factors are not fp32")
        if trainer.accelerator.num_processes != 1 or trainer.args.gradient_accumulation_steps != 1:
            raise RuntimeError("the probe supports one process and gradient accumulation 1")
        groups = [g for g in self.opt.param_groups if any(q is p for q in g["params"] for p in self.flat[:1])]
        self.handles.append(self.opt.register_step_post_hook(self._post))
        os.makedirs(self.dir, exist_ok=True)
        self.attached = True
        self.log(f"probe attached: {len(self.flat)} o_proj/down_proj LoRA factors "
                 f"({sum(p.numel() for p in self.flat) / 1e6:.1f} M values), {len(self.trainable)} trainable tensors, "
                 f"optimizer {type(self.opt).__name__} with {len(self.opt.param_groups)} groups, probe steps "
                 f"{self.planned[0]}..{self.planned[-1]} ({len(self.planned)}), output {self.dir}; group of the first "
                 f"factor: { {k: v for k, v in groups[0].items() if k != 'params'} }")

    def _group_hp(self):
        """The hyperparameters shared by the groups that hold the o_proj/down_proj factors (refuses a mismatch)."""
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
        return hp

    def _state_fingerprint(self):
        import torch

        s_m = s_v = 0.0
        steps = []
        with torch.no_grad():
            for p in self.trainable:
                st = self.opt.state.get(p)
                if not st:
                    continue
                s_m += float(st["exp_avg"].double().sum())
                s_v += float(st["exp_avg_sq"].double().sum())
                steps.append(float(st["step"]))
        return [s_m, s_v, min(steps) if steps else None, max(steps) if steps else None, len(steps)]

    # ------------------------------------------------------------ one batch

    def _one(self, model, batch, hp, step, name):
        import torch

        tr, fx = self.trainer, self.fx
        dev = fx.device
        ids = batch["input_ids"].to(dev)
        am = batch["attention_mask"].to(dev)
        labels = batch["labels"].to(dev)
        n_tok = [int(x) for x in am.sum(1).tolist()]
        T = int(am.shape[1])
        W = self._token_weights(n_tok, T, dev)
        N = (labels[:, 1:] != -100).sum()
        L = len(self.layers)
        X = {k: [None] * L for k in self.kinds}
        G = [None] * L
        hits = {"x": 0, "g": 0}

        def pre(k, li):
            def hook(mod, args):
                if not torch.is_grad_enabled():
                    return None
                if X[k][li] is not None:
                    raise RuntimeError(f"probe: {k} layer {self.layers[li]} input recorded twice")
                X[k][li] = self._reduce(W, args[0].detach()).double().sum(0)
                hits["x"] += 1
                return None
            return hook

        def fwd(li):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if torch.is_grad_enabled() and h.requires_grad:
                    def gfn(g):
                        s = self._reduce(W, g).double().sum(0)
                        G[li] = s if G[li] is None else G[li] + s
                        hits["g"] += 1
                        return None
                    h.register_hook(gfn)
                return None
            return hook

        hs = []
        for k in self.kinds:
            for li, m in enumerate(fx.mods[k]):
                hs.append(m.register_forward_pre_hook(pre(k, li)))
        for li, layer in enumerate(fx.dec_layers):
            hs.append(layer.register_forward_hook(fwd(li)))
        t0 = time.time()
        try:
            model.train()
            with tr.compute_loss_context_manager():
                out = model(input_ids=ids, attention_mask=am, labels=labels, use_cache=False, num_items_in_batch=N)
            loss = out["loss"] if isinstance(out, dict) else out[0]
            loss_val = float(loss.detach())
            del out
            tr.accelerator.backward(loss)
            del loss
        finally:
            for h in hs:
                h.remove()
        if any(x is None for k in self.kinds for x in X[k]) or any(g is None for g in G):
            raise RuntimeError(f"probe step {step} batch {name}: module inputs or layer gradients missing")
        pre_norm = tr.accelerator.clip_grad_norm_(model.parameters(), tr.args.max_grad_norm)
        pre_norm = float(pre_norm)
        with torch.no_grad():
            grads = [p.grad.detach().clone() for p in self.flat]  # clipped
        model.zero_grad(set_to_none=True)
        t_fb = time.time() - t0
        with torch.no_grad():
            # (i) would-be real step: a copy of the optimizer over clones of the factors
            cls = type(self.opt)
            sig = inspect.signature(cls.__init__).parameters
            kw = {k: v for k, v in hp.items() if k in sig and k != "params"}
            clones = [p.detach().clone() for p in self.flat]
            copt = cls(clones, **kw)
            g0 = copt.param_groups[0]
            mism = {k: (g0.get(k), v) for k, v in hp.items() if k not in ("params", "initial_lr") and g0.get(k) != v}
            if mism:
                raise RuntimeError(f"copied optimizer group differs from the real one: {mism}")
            for c, p, g in zip(clones, self.flat, grads):
                st = self.opt.state[p]
                copt.state[c] = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in st.items()}
                c.grad = g.clone()
            copt.step()
            for c in clones:
                c.grad = None
            # (ii) momentum-free step from v_{t-1}
            b1, b2 = hp["betas"]
            eps, lr = float(hp["eps"]), float(hp["lr"])
            if float(hp.get("weight_decay", 0.0)) != 0.0 or hp.get("amsgrad") or hp.get("maximize"):
                raise RuntimeError(f"momentum-free formula assumes weight decay 0, no amsgrad, no maximize: {hp}")
            mfree = []
            adam_step = None
            for p, g in zip(self.flat, grads):
                st = self.opt.state[p]
                tnew = float(st["step"]) + 1.0
                adam_step = tnew if adam_step is None else adam_step
                if tnew != adam_step:
                    raise RuntimeError("the factors' optimizer step counts differ")
                mfree.append((p.detach().double() - mfree_step(g, st["exp_avg_sq"], tnew, lr, b2, eps)))
            # direct-write terms
            D = {v: None for v in ("step", "stepfix", "mfree", "mfreefix")}
            i = 0
            for k in self.kinds:
                n = len(self.params[k])
                A0 = torch.stack([A.detach() for A, _ in self.params[k]])
                B0 = torch.stack([B.detach() for _, B in self.params[k]])
                A1 = torch.stack([clones[i + 2 * j] for j in range(n)])
                B1 = torch.stack([clones[i + 2 * j + 1] for j in range(n)])
                Am = torch.stack([mfree[i + 2 * j] for j in range(n)])
                Bm = torch.stack([mfree[i + 2 * j + 1] for j in range(n)])
                i += 2 * n
                s = fx.scaling[k]
                Xo = torch.stack(X[k])[:, None, :]  # [L, 1, din] fp64
                Xf = fx.Xf[k].sum(0)[:, None, :]  # [L, 1, din] fp64, summed over the halves
                for v, Xv, (a1, bb1) in (("step", Xo, (A1, B1)), ("stepfix", Xf, (A1, B1)),
                                         ("mfree", Xo, (Am, Bm)), ("mfreefix", Xf, (Am, Bm))):
                    d = self._delta(Xv, A0, B0, a1, bb1, s)[:, 0]  # [L, D] fp64, +dh
                    D[v] = d if D[v] is None else D[v] + d
            gsum = torch.stack(G) * N.to(torch.float64)  # +dL/dh of the summed loss, [L, D]
            vec = torch.stack([D["step"], D["stepfix"], D["mfree"], D["mfreefix"], -gsum])  # [5, L, D] descent
            dw_step = torch.cat([(c - p.detach()).reshape(-1) for c, p in zip(clones, self.flat)])
            dw_mfree = torch.cat([(m - p.detach().double()).reshape(-1).float() for m, p in zip(mfree, self.flat)])
            ext = None
            if self.extended:
                ext = self._extended(grads, clones, hp, lr, adam_step, X, D, cls, kw)
                vec = torch.cat([vec, ext[0]])  # [10, L, D]
        rec = dict(batch=name, n_docs=len(n_tok), n_label=int(N), n_tokens_sum=int(sum(n_tok)), loss=loss_val,
                   pre_clip_grad_norm=pre_norm, clip_coef=min(1.0, tr.args.max_grad_norm / (pre_norm + 1e-6)),
                   module_inputs_recorded=hits["x"], layer_grads_recorded=hits["g"],
                   dw_step_norm=float(dw_step.double().norm()), dw_mfree_norm=float(dw_mfree.double().norm()),
                   seconds_forward_backward=round(t_fb, 3))
        if ext is not None:
            rec["extended"] = ext[1]
        return vec, rec, clones, dw_step, dw_mfree, adam_step, lr, (None if ext is None else ext[2])

    def _extended(self, grads, clones, hp, lr, t, X, D, cls, kw):
        """The extended variants of one batch (module docstring): ([5, L, D] fp64 +dh in the order of
        EXTENDED_VARIANTS[5:], the batch's record with the identity checks, the flattened hist update fp32)."""
        import torch

        fx = self.fx
        b1, b2 = (float(x) for x in hp["betas"])
        eps = float(hp["eps"])
        # (i) the would-be step in fp64: a copy of the optimizer (single-tensor path) over fp64 clones, fp64 state
        kw64 = dict(kw)
        for k in ("foreach", "fused"):
            if k in kw64:
                kw64[k] = False
        c64 = [p.detach().double().clone() for p in self.flat]
        opt64 = cls(c64, **kw64)
        for c, p, g in zip(c64, self.flat, grads):
            st = self.opt.state[p]
            opt64.state[c] = {k: (v.double().clone() if k in ("exp_avg", "exp_avg_sq") else
                                  v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in st.items()}
            c.grad = g.double()
        opt64.step()
        # (ii) the parts of each factor (adam_parts) and the weight-space checks
        z = torch.zeros((), dtype=torch.float64, device=fx.device)
        sq = {k: z.clone() for k in ("step64", "hist", "curr", "mom", "resid", "gap32")}
        ulp_ratio = z.clone()
        parts = dict(hist=[], curr=[], mom=[], step64=[])
        for c, c32, p, g in zip(c64, clones, self.flat, grads):
            st = self.opt.state[p]
            d = adam_parts(g, st["exp_avg"], st["exp_avg_sq"], t, lr, b1, b2, eps)
            p64 = p.detach().double()
            d64 = c.detach() - p64
            c.grad = None
            gap = c32.detach().double() - p64 - d64  # the stored fp32 step minus the fp64 one
            a32 = c32.detach().abs()
            spacing = (torch.nextafter(a32, torch.full_like(a32, float("inf"))) - a32).double()
            ulp_ratio = torch.maximum(ulp_ratio, (gap.abs() / (spacing + 1e-6 * d64.abs())).max())
            sq["step64"] += (d64 * d64).sum()
            sq["resid"] += ((d64 - (d["hist"] + d["curr"])) ** 2).sum()
            sq["gap32"] += (gap * gap).sum()
            for k in ("hist", "curr", "mom"):
                sq[k] += (d[k] * d[k]).sum()
                parts[k].append(d[k])
            parts["step64"].append(d64)
        del c64, opt64
        # (iii) direct-write terms (+dh, fp64) on the own and on the fixed documents
        E = {}
        i = 0
        for k in self.kinds:
            n = len(self.params[k])
            A0 = torch.stack([A.detach() for A, _ in self.params[k]]).double()
            B0 = torch.stack([B.detach() for _, B in self.params[k]]).double()
            dA = {v: torch.stack([parts[v][i + 2 * j] for j in range(n)]) for v in parts}
            dB = {v: torch.stack([parts[v][i + 2 * j + 1] for j in range(n)]) for v in parts}
            i += 2 * n
            s = fx.scaling[k]
            Xo = torch.stack(X[k])[:, None, :]  # [L, 1, din] fp64
            Xf = fx.Xf[k].sum(0)[:, None, :]  # [L, 1, din] fp64, summed over the halves
            for sfx, Xv in (("", Xo), ("fix", Xf)):
                terms = {v: delta_parts(Xv, A0, B0, dA[v], dB[v], s) for v in ("mom", "hist", "curr", "step64")}
                terms["cross"] = delta_cross(Xv, dA["hist"], dB["hist"], dA["curr"], dB["curr"], s)
                for v, d_ in terms.items():
                    E[v + sfx] = d_[:, 0] if v + sfx not in E else E[v + sfx] + d_[:, 0]
        hist_flat = torch.cat([h.reshape(-1).float() for h in parts["hist"]])  # for the cross-batch cosine
        del parts

        def norms(a):
            return a.norm(dim=-1)

        def lcos(a, b):
            return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-300)

        def d_rel_max(sfx):
            r = E["step64" + sfx] - E["hist" + sfx] - E["curr" + sfx] - E["cross" + sfx]
            return float((norms(r) / norms(E["step64" + sfx]).clamp_min(1e-300)).max())

        n64 = float(sq["step64"].sqrt())
        rec = dict(
            adam_step=t,
            w_identity_rel=float(sq["resid"].sqrt()) / max(n64, 1e-300),
            d_identity_rel_max_own=d_rel_max(""), d_identity_rel_max_fix=d_rel_max("fix"),
            cross_rel_median_own=layer_median(norms(E["cross"]) / norms(E["step64"]).clamp_min(1e-300)),
            cross_rel_median_fix=layer_median(norms(E["crossfix"]) / norms(E["step64fix"]).clamp_min(1e-300)),
            step32_vs_step64_rel=float(sq["gap32"].sqrt()) / max(n64, 1e-300),
            step32_rounding_ratio_max=float(ulp_ratio),
            d_step32_vs_step64_rel_median=layer_median(norms(D["step"] - E["step64"]) / norms(E["step64"]).clamp_min(1e-300)),
            dw_norm=dict(step64=n64, hist=float(sq["hist"].sqrt()), curr=float(sq["curr"].sqrt()),
                         mom=float(sq["mom"].sqrt())),
            d_norm_median=dict(step=layer_median(norms(D["step"])), mfree=layer_median(norms(D["mfree"])),
                               hist=layer_median(norms(E["hist"])), curr=layer_median(norms(E["curr"])),
                               mom=layer_median(norms(E["mom"]))),
            d_norm_median_fix=dict(step=layer_median(norms(D["stepfix"])), hist=layer_median(norms(E["histfix"])),
                                   mom=layer_median(norms(E["momfix"]))),
            cos_median=dict(step_hist=layer_median(lcos(D["step"], E["hist"])),
                            step_curr=layer_median(lcos(D["step"], E["curr"])),
                            curr_mfree=layer_median(lcos(E["curr"], D["mfree"])),
                            step_mom=layer_median(lcos(D["step"], E["mom"])),
                            hist_curr=layer_median(lcos(E["hist"], E["curr"]))),
            rule=f"ok: w_identity_rel <= {IDENTITY_TOL}, d_identity_rel_max_own and _fix <= {IDENTITY_TOL} "
                 f"(D(step64) = D(hist) + D(curr) + C per layer), step32_vs_step64_rel <= {STEP32_TOL}")
        rec["ok"] = bool(rec["w_identity_rel"] <= IDENTITY_TOL and rec["d_identity_rel_max_own"] <= IDENTITY_TOL
                         and rec["d_identity_rel_max_fix"] <= IDENTITY_TOL and rec["step32_vs_step64_rel"] <= STEP32_TOL)
        vec = torch.stack([E["mom"], E["momfix"], E["hist"], E["histfix"], E["curr"]])
        return vec, rec, hist_flat

    def _token_weights(self, n_tok, T, dev):
        import adamfx

        return adamfx.token_weights(n_tok, T, dev)

    @staticmethod
    def _reduce(W, x):
        import adamfx

        return adamfx.reduce_tokens(W, x)

    @staticmethod
    def _delta(X, A0, B0, A1, B1, s):
        import adamfx

        return adamfx.module_delta(X, A0, B0, A1, B1, s)

    # ------------------------------------------------------------ one probe step

    def run(self, trainer, model, inputs, num_items_in_batch, step):
        import torch

        if not self.attached:
            self._attach(trainer)
        fx = self.fx
        if fx.armed_fx or fx.mode is not None or fx.pending is not None:
            raise RuntimeError("probe called while the adamfx hooks are armed")
        t0 = time.time()
        if any(p.grad is not None for p in self.trainable):
            raise RuntimeError(f"probe step {step}: gradients are not None at the start of training_step")
        cuda = torch.cuda.is_available() and fx.device.type == "cuda"
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if cuda else []
        was_training = model.training
        with torch.no_grad():
            snap = [p.detach().clone() for p in self.trainable]
        fp0 = self._state_fingerprint()
        hp = self._group_hp()
        real = {k: inputs[k] for k in ("input_ids", "attention_mask", "labels")}
        line_no = inputs.get("line_no")
        line_no = [int(x) for x in line_no.tolist()] if line_no is not None else None
        N_real = int((real["labels"][:, 1:] != -100).sum())
        if num_items_in_batch is not None and int(num_items_in_batch) != N_real:
            raise RuntimeError(f"probe step {step}: num_items_in_batch {int(num_items_in_batch)} != label count {N_real}")
        batches = [("real", real)] + [(f"f{k + 1}", b) for k, b in enumerate(self.foreign)]
        vecs, recs, dws, dwm, dwh = [], [], [], [], []
        real_after = None
        lr = adam_step = None
        for name, b in batches:
            vec, rec, clones, dw_s, dw_m, adam_step, lr, dw_h = self._one(model, b, hp, step, name)
            vecs.append(vec.float().cpu())
            recs.append(rec)
            dws.append(dw_s)
            dwm.append(dw_m)
            dwh.append(dw_h)
            if name == "real":
                real_after = clones
            else:
                del clones
        # weight-space cosines between the batches' would-be updates (step and momentum-free)
        import torch.nn.functional as F
        wcos_step = [[float(F.cosine_similarity(a.double(), b.double(), dim=0)) for b in dws] for a in dws]
        wcos_mfree = [[float(F.cosine_similarity(a.double(), b.double(), dim=0)) for b in dwm] for a in dwm]
        wcos_step_mfree = [float(F.cosine_similarity(a.double(), b.double(), dim=0)) for a, b in zip(dws, dwm)]
        del dws, dwm
        ext = self._extended_across(vecs, dwh, recs) if self.extended else None
        del dwh
        # state left alone
        with torch.no_grad():
            same_params = all(torch.equal(p, s) for p, s in zip(self.trainable, snap))
        del snap
        fp1 = self._state_fingerprint()
        grads_none = all(p.grad is None for p in self.trainable)
        rng_cpu_same = bool(torch.equal(cpu_rng, torch.get_rng_state()))
        rng_cuda_same = bool(not cuda or all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())))
        torch.set_rng_state(cpu_rng)
        if cuda:
            torch.cuda.set_rng_state_all(cuda_rng)
        if not was_training:
            model.eval()
        state_ok = dict(params_bit_equal=bool(same_params), optimizer_state_equal=fp0 == fp1, grads_none=grads_none,
                        rng_cpu_unchanged=rng_cpu_same, rng_cuda_unchanged=rng_cuda_same)
        if not (same_params and fp0 == fp1 and grads_none):
            raise RuntimeError(f"probe step {step} changed the training state: {state_ok}")
        V = torch.stack(vecs).numpy()  # [4, 5 (10 extended), L, D] fp32
        norms = np.linalg.norm(V.astype(np.float64), axis=-1)
        fin = bool(np.isfinite(V).all())
        meta = dict(step=int(step), lr=lr, adam_step=adam_step, hyperparameters={k: v for k, v in hp.items()},
                    batches=list(BATCH_NAMES), variants=list(self.variants), variant_text=self.variant_text,
                    sign=self.sign_text,
                    layers=list(self.layers), real_line_no=line_no,
                    real_doc_id=None if (line_no is None or fx.rows is None) else [fx.rows["doc_id"][j] for j in line_no],
                    foreign_lines=self.foreign_meta["batches"], foreign_doc_id=self.foreign_meta["doc_id"],
                    foreign_arm=self.foreign_meta["foreign_arm"], foreign_file=self.foreign_meta["file"],
                    foreign_file_sha256=self.foreign_meta["file_sha256"], xfixed_step=int(fx.Xf_step),
                    per_batch=recs, num_items_in_batch=None if num_items_in_batch is None else int(num_items_in_batch),
                    wcos_step=wcos_step, wcos_mfree=wcos_mfree, wcos_step_vs_mfree=wcos_step_mfree,
                    state=state_ok, finite=fin, seconds=round(time.time() - t0, 2), weight_check=None)
        if ext is not None:
            meta["variant_set"] = self.variant_set
            meta["extended"] = ext
        if not fin:
            raise RuntimeError(f"probe step {step}: non-finite vectors")
        path = os.path.join(self.dir, f"probe_step_{step:06d}.npz")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, vectors=V, norms=norms, batches=np.asarray(BATCH_NAMES), variants=np.asarray(self.variants),
                     layers=np.asarray(self.layers, dtype=np.int64), step=np.int64(step),
                     meta=np.asarray(json.dumps(meta, default=str)))
        os.replace(tmp, path)
        self.records[int(step)] = meta
        # the real batch's would-be step, to compare with the trainer's real step in the optimizer post-hook
        self.pending = dict(step=int(step), before=[p.detach().clone() for p in self.flat], after=real_after)
        self._write_meta(complete=False)
        self.log(f"probe step {step}: lr {lr:.3e}, adam step {adam_step:.0f}, batches "
                 + ", ".join(f"{r['batch']} N={r['n_label']} loss={r['loss']:.4f} |g|={r['pre_clip_grad_norm']:.3f}"
                             for r in recs)
                 + f"; weight-space cos(real, f1..f3) step {[round(x, 3) for x in wcos_step[0][1:]]} momentum-free "
                   f"{[round(x, 3) for x in wcos_mfree[0][1:]]}; state {state_ok}; {meta['seconds']} s"
                 + ("" if ext is None else
                    f"; extended: identity {'ok' if ext['identity_ok'] else 'FAILED'} (weight rel max "
                    f"{ext['w_identity_rel_max']:.1e}, D rel max {ext['d_identity_rel_max']:.1e}, fp32 step vs fp64 "
                    f"max {ext['step32_vs_step64_rel_max']:.1e}), cross term |C|/|D(step)| real "
                    f"{recs[0]['extended']['cross_rel_median_own']:.1e}, cos(D real hist, D f_k hist) own "
                    f"{[round(x, 4) for x in ext['cos_hist_real_f']]} fixed {[round(x, 4) for x in ext['cos_histfix_real_f']]}"))

    def _extended_across(self, vecs, dwh, recs):
        """Cross-batch diagnostics of the extended variants (meta "extended") and the step's identity verdict."""
        import torch.nn.functional as F

        vi = {v: i for i, v in enumerate(self.variants)}

        def cos_real_f(v):
            j = vi[v]
            a = vecs[0][j].double()
            return [layer_median(F.cosine_similarity(a, vecs[k][j].double(), dim=-1)) for k in (1, 2, 3)]

        ex = [r["extended"] for r in recs]
        return dict(
            identity_ok=all(e["ok"] for e in ex), identity_tol=IDENTITY_TOL, step32_tol=STEP32_TOL,
            w_identity_rel_max=max(e["w_identity_rel"] for e in ex),
            d_identity_rel_max=max(max(e["d_identity_rel_max_own"], e["d_identity_rel_max_fix"]) for e in ex),
            step32_vs_step64_rel_max=max(e["step32_vs_step64_rel"] for e in ex),
            cos_hist_real_f=cos_real_f("hist"), cos_histfix_real_f=cos_real_f("histfix"),
            cos_curr_real_f=cos_real_f("curr"), cos_mom_real_f=cos_real_f("mom"),
            cos_momfix_real_f=cos_real_f("momfix"),
            wcos_hist_real_f=[float(F.cosine_similarity(dwh[0].double(), dwh[k].double(), dim=0)) for k in (1, 2, 3)],
            rule="cos_*_real_f: layer median of cos(D(real v), D(f_k v)), k = 1..3 (from the stored fp32 vectors); "
                 "wcos_hist_real_f: weight-space cos of the hist updates over all o_proj/down_proj LoRA factors")

    def _post(self, opt, args, kwargs):
        """After the trainer's real optimizer step: compare it with the real batch's would-be step (weight space)."""
        import torch

        p = self.pending
        if p is None:
            return
        self.pending = None
        with torch.no_grad():
            num = den = 0.0
            mx = 0.0
            for q, b, a in zip(self.flat, p["before"], p["after"]):
                d_real = q.detach().double() - b.double()
                d_copy = a.double() - b.double()
                num += float(((d_real - d_copy) ** 2).sum())
                den += float((d_real ** 2).sum())
                mx = max(mx, float((q.detach() - a).abs().max()))
        rel = math.sqrt(num) / max(math.sqrt(den), 1e-300)
        ok = rel <= self.weight_tol
        self.records[p["step"]]["weight_check"] = dict(
            rel_diff=rel, max_abs_diff=mx, bit_identical=mx == 0.0, ok=ok, tol=self.weight_tol,
            rule="|| (W_after_real - W_before) - (W_after_copy - W_before) || / || W_after_real - W_before || over the "
                 "o_proj/down_proj LoRA factors, the trainer's step t vs the probe's would-be step of the real batch")
        self._write_meta(complete=False)
        self.log(f"probe step {p['step']}: real step vs would-be step of the real batch: relative difference {rel:.3e}, "
                 f"max abs {mx:.3e} ({'ok' if ok else 'ABOVE TOLERANCE'})")

    def _write_meta(self, complete):
        rec = dict(planned_steps=self.planned, done_steps=sorted(self.records), complete=complete,
                   records=[self.records[t] for t in sorted(self.records)], foreign=self.foreign_meta,
                   variants=list(self.variants), batches=list(BATCH_NAMES), variant_text=self.variant_text,
                   sign=self.sign_text,
                   written=time.strftime("%Y-%m-%d %H:%M:%S"))
        path = os.path.join(self.dir, "probe_meta.json")
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(path + ".tmp", path)

    def finish(self):
        """Mark the probe complete when every planned step ran and passed its weight check."""
        if not self.attached:
            self.log("probe: never attached (no training step ran)")
            return False
        done = sorted(self.records)
        wc = [self.records[t].get("weight_check") or {} for t in done]
        complete = done == self.planned and all(w.get("ok") for w in wc)
        self._write_meta(complete=complete)
        self.log(f"probe finished: {len(done)}/{len(self.planned)} steps, weight checks ok "
                 f"{sum(bool(w.get('ok')) for w in wc)}/{len(wc)}, complete {complete}")
        for h in self.handles:
            h.remove()
        return complete


def make_probe_trainer_class(base, probe):
    """Trainer subclass that runs ``probe`` at the start of the probe steps' training_step (before the real
    forward, so before the adamfx hooks are armed)."""

    class CleanBatchProbeTrainer(base):
        def training_step(self, model, inputs, num_items_in_batch=None):
            step = self.state.global_step + 1
            if step in probe.steps:
                probe.run(self, model, inputs, num_items_in_batch, step)
            return super().training_step(model, inputs, num_items_in_batch)

    return CleanBatchProbeTrainer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--out", default=FOREIGN_FILE)
    a = ap.parse_args(argv)
    if a.cmd == "draw":
        return cmd_draw(a)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

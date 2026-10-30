"""Adam-effect inputs saved during student training (``train_subliminal.py --save-adamfx [--save-acttrue]``).

Plan: ``docs/notes/20261024_adam_effect_input_plan.md`` (user decision 2026-10-24 12:50; the true change was added
at 13:10, the per-step batch sums and the plain-gradient windows at 13:35-13:45, the disk guard at 13:31).

Series. Per optimizer step t, document d of the step's batch and layer L (L = output of ``model.model.layers[L]``,
the gradient store's layer convention), each summed over the gradient store's ``full`` token set: positions
0..n-2 of the row (n = its non-padding tokens; position 0 included; ``core.position_sets(n, plen, "full", False)``,
the set ``full_grad_L<L>`` of online_save.py sums over):

  adamfx   first-order Adam effect. dh = dW_o X_o + dW_down X_down. X_m is the token sum of the input of module m
           (``o_proj`` and ``down_proj`` of layer L, the two LoRA modules that write into the residual stream) in
           step t's training forward (forward pre-hooks); dW_m = s (B'A' - BA) is the change of the LoRA product over
           the optimizer step, s = peft's ``scaling[adapter]`` (lora_alpha / r = 2 for r 16, alpha 32). A, B
           are copied in the optimizer's step pre-hook, A', B' read in its step post-hook, so clipping, the
           scheduled lr, weight decay and Adam's moments are all inside the difference. Computed in fp64 as
           s [dB (A'X) + B (dA X)] with dA = A' - A, dB = B' - B: algebraically B'(A'X) - B(AX), without the
           cancellation of two nearly equal products. Only the direct writes of layer L are counted; changes of
           q, k, v, gate, up and of earlier layers, and their propagation, are not.
  acttrue  true change (``--save-acttrue``). dh = tokensum(h after) - tokensum(h before). h before comes from the
           training forward (forward hooks on the decoder layers, reduced on the fly); h after from one extra
           forward of the same inputs right after the optimizer step (in its post-hook, before the lr scheduler
           step): torch.no_grad, the trainer's autocast (accelerate's), the model's train mode left as it is, the
           decoder (``model.model``) called with exactly the arguments it received in the training forward, the
           LM head skipped. Dropout must be 0 everywhere (checked at start), so the forward draws no random
           numbers (the CUDA and CPU RNG states are compared around the first extra forward). It contains every
           module's change, its propagation through later layers and the bf16 rounding of the forward.
  grad     plain activation gradient: +dL/dh_L of the summed token loss (the hook gradient of the trainer's loss
           times the step's normaliser), every step: the gradient store's ``full_grad_L<L>`` row of the document.
           Its windows are the quantity ``e2_read.py --export-inputs`` builds from a per-document store, so a run
           without that store still gets its plain-gradient inputs, and a run with one can be checked against it
           (``grad_reference``: cosine >= 0.9999 per layer and window at the common read steps).
  adamfx_fixed  input 4f of the new-setting runs (plan ``docs/notes/20261025_new_setting_plan.md``, user 2026-10-25
           02:05 and 02:20): the adamfx product of the step's real LoRA update applied to a fixed document set
           instead of the batch, D_fixed_t = sum over the fixed documents d of (dW_o,t Xf_o,d + dW_down,t Xf_down,d).
           Xf is the token sum (the same full set) of each module input of the fixed documents in a no-grad forward
           of the decoder inside the training process, under the trainer's autocast (bf16 activations), refreshed at
           step 0 and after every ``every``-th step (100): steps r+1..r+100 use the sums from the parameters after
           step r. Because the product is linear in X, the documents of a half are summed first (fp64) and one
           [layers, din] vector per half and module stays on the GPU (a few MB). One row per half per step (two
           halves of 64 for the split-half check); the per-step store holds the sum over the halves. The forward
           runs between optimizer steps (trainer callback), draws no random numbers (checked) and leaves the
           parameters alone, so the trajectory is unchanged.

Sign. Internally every series accumulates its stored-sign vector q: q = -dh for adamfx and acttrue, q = +g for
grad. Under plain SGD on a full weight matrix, -dh_d = (lr / N) sum_tok' g_tok' (x_tok' . X_d) with g the per-token
gradient of the module output and N the trainer's normaliser: a positive mix of +dL/dh rows, so q has the gradient
store's sign in all three series. Under LoRA SGD the mixing kernel is s^2 (A x . A X) plus a B B^T term
(tests/test_adamfx.py (a)), positive semi-definite but not x . X. For down_proj, g is the stored layer-output
gradient; for o_proj it is the gradient at the residual after attention, which also carries the MLP's
back-propagated part. The window and docs files hold -q, exactly as ``e2_read.py --export-inputs`` and
``--export-inputs-adam`` hold -(+dL/dh): ``inputs = -(sum over the window's documents of q)`` (the summed dh for
adamfx and acttrue, -g for grad), the descent-direction sign the trained readers (e2_generate.py) take. The
per-step store (``stepsums/``) holds q itself. Magnitudes are the real step (no division by lr).

Windows (as e2_read.py rule 3, built online). Every step adds w1(t) = sum over the step's documents of q (fp64
reduction, kept as fp32) to a ring of the last max(w) steps (fp32, on the GPU) and to an fp64 running sum (cum).
At read step n: w1 = step n, w100 = steps max(n-99, 1)..n, w1000 = max(n-999, 1)..n (ring sums accumulated in
fp64), cum = 1..n. Read steps: step 1 and every ``every`` steps up to the last step (e2_read.selected_steps on a
store from step 1). Step 1 has lr 0 under the HF warmup, so its adamfx rows are exactly zero; its acttrue rows
are zero exactly when the extra forward reproduces the training forward bitwise (recorded as a check). The trained
readers refuse zero vectors: read the adamfx and acttrue files with ``e2_generate.py --steps`` without step 1.
Diagnostic at the read steps (acttrue docs file, ``diag``): per layer, the share of position 0 in the batch sums
of the true and first-order changes and their cosine with and without position 0 (position 0 is
``<|im_start|>``, the same token in every row, with the large activations).

Files in ``<out>/e2/`` (``E`` = every):
  <series>_inputs_everyE.npz  adamfx, acttrue, grad: the layout of ``e2_read.py --export-inputs``: inputs fp32
                              [n_read, n_windows, L, D], steps, windows, window_lo, window_hi, n_docs, layers, meta
                              (JSON; sign text and kind 'grad' as e2_generate.load_inputs requires; adamfx and
                              acttrue carry transform = {variant: <series>, ...}, grad none: it is the plain input)
  <series>_docs_everyE.npz    per-document vectors of the read steps: vectors fp32 [n_read, B, L, D], steps, layers,
                              n_docs, line_no, doc_id, n_tokens [n_read, B] (padding -1 / ''), lr, meta; the three
                              docs files share the document index arrays row for row
Per-step batch sums in ``<out>/stepsums/<series>/`` (``stepsums``: by default adamfx and acttrue):
  block_<s>_<e>.npy           fp16 unit rows [steps, L, D] of q (zero rows stay zero), steps s..e (``block_steps``
  block_<s>_<e>.norms.npy     = 1000 per block), and their fp32 norms [steps, L]: value = unit * norm (the
                              gradient store's storage convention; plain fp16 would underflow the early warmup
                              steps, whose adamfx elements are about 1e-5 to 1e-7)
  index.json                  quantity, sign, layers, blocks (start, end, files), complete; ``load_stepsums``
  ``stepsum_dtype`` fp32 (train_subliminal ``--adamfx-stepsums-dtype fp32``) stores block_<s>_<e>.npy as the fp32
  rows themselves (index ``format`` fp32, blocks without ``norms_file``): the ring entries, so windows rebuilt from
  the store equal the online windows up to fp64 summation order; 0.82 MB per step for 40 layers x 5120.
The fixed-document series stores its per-step sums in ``<out>/adamfx_fixed/`` (same block and index format, quantity
adamfx_fixed) next to ``xfixed.npz`` (every refresh's fp64 sums x_o_proj [n_refresh, halves, L, 5120], x_down_proj
[n_refresh, halves, L, 17408], steps, rel_change, refresh_s); its window and docs files are
``e2/adamfx_fixed_inputs_everyE.npz`` and ``e2/adamfx_fixed_docs_everyE.npz`` (one docs row per half).
Every block and file is written to ``.tmp`` and renamed. While training, every ``flush_every`` read steps (default
20, i.e. every 2,000 steps) a background thread writes the window files' rows so far to ``<name>.partial.npz``
(meta complete false); the docs files are written only at the end. The final names are written once at the end of
training, after the integrity checks (meta ``checks``), and only when every planned read step is present; the
partial files are then removed. A run that dies keeps only partial files, so a partial file never looks finished.

LoRA pairs (``lora_pairs``, train_subliminal ``--adamfx-lora-pairs``): at the listed steps the LoRA parameters of
every LoRA module (all 7 targets x 40 layers, 64.2 M values) before and after that step's optimizer step go to
``<out>/lora_pairs/step_<N>/`` so the true change can be recomputed offline in fp32 (``adamfx_fp32_check.py``).
The parameters are fp32 tensors updated in place by the optimizer (no separate master copy), and only fp32 holds the
update exactly: format ``fp32`` stores ``before.safetensors`` and ``after.safetensors`` in fp32 (257 MB each, 514
MB per pair); format ``bf16-delta`` stores ``before`` in bf16 and ``delta.safetensors`` = bf16(after - before)
(257 MB per pair; offline after = before + delta, each element within 2^-9 relative of its value). Also
``batch.npz`` (the step's input_ids, attention_mask, labels, line_no, n_tokens, doc_id: the exact batch; it can also
be rebuilt from RUN.json's data settings with train_subliminal.tokenize_rows and CompletionMaskCollator on the rows
at line_no), ``vectors.npz`` (that step's per-document adamfx, acttrue and grad vectors, the docs-file sign) and
``meta.json`` (step, lr, normaliser, LoRA config and scaling, parameter names and shapes, format), and with fixed
documents ``fixed.npz`` (the fp64 sums x_<kind> [halves, L, din] that step used and their refresh step) plus the
step's adamfx_fixed rows in ``vectors.npz``. Names are the PeftModel parameter names. Written by a background thread after the disk guard.

Disk guard. Before each block and each file is written the free space of the output filesystem is checked
(shutil.disk_usage); when writing would leave less than ``min_free_gb`` (5 GB, user 2026-10-24 14:15), nothing is written, the run stops
after the step with ``<out>/e2/FAILED_lowdisk`` (train_subliminal.py exits with status 4) and the log says so. The
free space is logged at every read step.

Sizes per run (20,000 steps, 201 read steps, B 8, 40 layers, d 5120): per-step store 8.2 GB per series (fp16);
window file 0.66 GB per series; docs file 1.32 GB per series.

Cost (B200, batch 8 x about 128 tokens): 80 forward pre-hook reductions, 40 decoder-hook reductions and 40
gradient reductions per step, a copy of the o_proj and down_proj LoRA factors (21 M fp32) and a handful of fp64
batched matmuls per step, a 0.8 GB fp32 ring and a 0.4 GB fp16 block buffer per series on the GPU. Measured
2026-10-24 (B200, dolphin seed 1 recipe, steps 11-100 of a 100-step run, the losses equal to the original run's):
no saving 0.191 s/step mean (0.176 median); adamfx + acttrue + grad windows + per-step stores of adamfx and acttrue
0.269 mean (0.246 median), 1.41x; adamfx alone adds no measurable time (30-step run), so almost all of the
difference is the extra forward of acttrue (a no-grad forward of the 14B decoder, about 0.07-0.08 s per step).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import shutil
import threading
import time

import numpy as np
import torch

from agm import core

KINDS = ("o_proj", "down_proj")
SERIES = ("adamfx", "acttrue", "grad", "adamfx_fixed")
STEPSUM_DTYPES = ("fp16unit", "fp32")
TOKEN_SET = ("full: positions 0..n-2 of each row (n = non-padding tokens, position 0 included), "
             "core.position_sets(n, plen, 'full', full_skip_pos0=False): the token set of the gradient store's "
             "full_grad_L<L> rows (online_save.py full_skip_pos0 grad False)")
SIGN = {
    "adamfx": ("q = -dh with dh = dW_o X_o + dW_down X_down the first-order change of the layer-L residual output "
               "that the step's real LoRA update makes on the document (adamfx.py); under plain SGD q is a "
               "positive mix of per-token +dL/dh rows of the summed, unscaled token cross-entropy, the sign of the "
               "gradient store (online_save); inputs = -(sum over the window's documents of q) = the summed dh: "
               "the descent direction, as export_inputs; ring entries fp32, sums fp64, stored as fp32"),
    "acttrue": ("q = -dh with dh = tokensum(h after the step) - tokensum(h before) of the layer-L residual output "
                "(training forward vs one extra no-grad forward after the optimizer step, adamfx.py); same "
                "convention as adamfx, whose q is a positive mix of +dL/dh rows under plain SGD; inputs = -(sum "
                "over the window's documents of q) = the summed dh: the descent direction, as export_inputs; ring "
                "entries fp32, sums fp64, stored as fp32"),
    "grad": ("q = +dL/dh of the summed, unscaled token cross-entropy (hook gradient x the trainer's normaliser), "
             "computed online every step (adamfx.py), the quantity the gradient store (online_save) keeps per "
             "document; inputs = -(sum over the window's documents of q): the descent direction, as export_inputs; "
             "ring entries fp32, sums fp64, stored as fp32"),
    "adamfx_fixed": ("q = -dh with dh = sum over the fixed documents of dW_o Xf_o + dW_down Xf_down, the first-order "
                     "change of the layer-L residual output that the step's real LoRA update makes on the fixed "
                     "documents (adamfx.py, input 4f); same convention as adamfx; inputs = -(sum over the window's "
                     "steps of q) = the summed dh: the descent direction, as export_inputs; ring entries fp32, sums "
                     "fp64, stored as fp32"),
}
DOCS_SIGN = {
    "adamfx": "vectors = +dh = -q per document (adamfx.py), the sign of adamfx_inputs (descent direction)",
    "acttrue": "vectors = +dh = -q per document (adamfx.py), the sign of acttrue_inputs (descent direction)",
    "grad": "vectors = -q = -(+dL/dh) per document: the gradient store's full_grad row negated (descent direction)",
    "adamfx_fixed": ("vectors = +dh = -q per half of the fixed documents (row h = the sum over the documents of half "
                     "h; line_no = h), the sign of adamfx_fixed_inputs (descent direction)"),
}
STEP_SIGN = {k: v.split("; inputs =")[0] + "; the per-step store holds q (not negated)" for k, v in SIGN.items()}
DEFINITION = {
    "adamfx": ("dh_{L,d,t} = dW_{o,L,t} X_{o,L,d} + dW_{down,L,t} X_{down,L,d}; X = token sum of the module input in "
               "step t's training forward; dW = s (B'A' - BA) over the optimizer step, s = peft scaling "
               "(lora_alpha / r), computed in fp64 as s [dB (A'X) + B (dA X)]; direct writes of layer L only"),
    "acttrue": ("dh_{L,d,t} = tokensum(h_L after step t) - tokensum(h_L in step t's training forward); after = one "
                "extra forward of the same inputs right after the optimizer step under torch.no_grad and the "
                "trainer's autocast, model mode unchanged, decoder called with the training forward's arguments"),
    "grad": "g_{L,d,t} = token sum of +dL/dh_L of the summed token loss in step t's training backward",
    "adamfx_fixed": ("dh_{L,t} = sum over the fixed documents d of [dW_{o,L,t} Xf_{o,L,d} + dW_{down,L,t} "
                     "Xf_{down,L,d}]; Xf = token sum (the full set, positions 0..n-2) of the module input of fixed "
                     "document d in a no-grad forward of the decoder under the trainer's autocast (bf16), with the "
                     "parameters after step r = every * floor((t-1)/every) (r = 0: the initial model); dW as adamfx "
                     "(fp64, s [dB (A'Xf) + B (dA Xf)]); direct writes of layer L only"),
}
DIAG_LABELS = ("share_pos0_true", "share_pos0_fx", "cos_fx_true", "cos_fx_true_without_pos0")


# ---------------------------------------------------------------- small shared helpers


def parse_windows(spec):
    """'1,100,1000,cum' -> [('w1', 1), ('w100', 100), ('w1000', 1000), ('cum', None)] (as e2_read.parse_windows)."""
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x.startswith("w"):
            x = x[1:]
        out.append(("cum", None) if x == "cum" else (f"w{int(x)}", int(x)))
    return out


def export_order_hash(steps, labels, layers):
    """sha1 over steps, window labels and layers in stored order (the same function as e2_read.export_order_hash)."""
    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def read_steps(last_step, every, first=1):
    """Step ``first`` and every multiple of ``every`` up to ``last_step`` (e2_read.selected_steps on a full store)."""
    return sorted({first} | set(range(every, last_step + 1, every))) if last_step >= first else []


def parse_steps(spec):
    """'1000:20000:1000,1000:3500:100,5' -> sorted unique steps (ranges lo:hi:every inclusive); 'none' or '' -> []."""
    out = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or part == "none":
            continue
        if ":" in part:
            lo, hi, ev = (int(x) for x in part.split(":"))
            out |= set(range(lo, hi + 1, ev))
        else:
            out.add(int(part))
    if any(t < 1 for t in out):
        raise ValueError(f"steps must be >= 1: {spec}")
    return sorted(out)


def token_weights(n_tokens, T, device):
    """[B, 1, T] fp32 0/1 weights of the ``full`` token set (positions 0..n-2) through core.position_weights."""
    rows = [(int(n), 1) for n in n_tokens]  # plen is not used by 'full'
    return core.position_weights(rows, T, ["full"], "grad", False).to(device)


def lora_modules(model, layers, kinds=KINDS):
    """{'o_proj': [...], 'down_proj': [...]} peft LoRA Linear modules of ``layers``, the adapter name and scalings.

    Refuses anything but one active vanilla adapter with identity dropout, unmerged. ``kinds`` is a subset of
    KINDS for tests (a model whose only LoRA module is one down_proj); training always uses both.
    """
    from peft.tuners.lora.layer import Linear as LoraLinear

    dec = core.decoder_layers(model)
    get = {"o_proj": lambda layer: layer.self_attn.o_proj, "down_proj": lambda layer: layer.mlp.down_proj}
    mods = {k: [get[k](dec[L]) for L in layers] for k in kinds}
    adapter = None
    for k, ms in mods.items():
        for L, m in zip(layers, ms):
            if not isinstance(m, LoraLinear):
                raise ValueError(f"layer {L} {k} is {type(m).__name__}, not a peft LoRA Linear")
            act = list(m.active_adapters)
            if len(act) != 1 or m.merged or m.disable_adapters:
                raise ValueError(f"layer {L} {k}: active adapters {act}, merged {m.merged}")
            ad = act[0]
            adapter = adapter or ad
            if ad != adapter or ad in getattr(m, "lora_variant", {}) or m.use_dora.get(ad, False):
                raise ValueError(f"layer {L} {k}: adapter {ad} is not one vanilla LoRA adapter")
            if not isinstance(m.lora_dropout[ad], torch.nn.Identity):
                raise ValueError(f"layer {L} {k}: LoRA dropout {m.lora_dropout[ad]} (must be 0)")
    scal = {k: [float(m.scaling[adapter]) for m in ms] for k, ms in mods.items()}
    return mods, adapter, scal


def module_delta(X, A0, B0, A1, B1, s):
    """dY = s (B1 A1 - B0 A0) X per row, fp64: s [(X A1^T) dB^T + (X dA^T) B0^T].

    X [L, B, din], A [L, r, din], B [L, dout, r] (nn.Linear weights), s [L] -> [L, B, dout].
    """
    X, A0, B0, A1, B1 = (t.double() for t in (X, A0, B0, A1, B1))
    dA, dB = A1 - A0, B1 - B0
    t1 = torch.bmm(torch.bmm(X, A1.transpose(1, 2)), dB.transpose(1, 2))
    t2 = torch.bmm(torch.bmm(X, dA.transpose(1, 2)), B0.transpose(1, 2))
    return s.view(-1, 1, 1).to(t1) * (t1 + t2)


def check_no_dropout(model):
    """Names of dropout modules with p > 0 and nonzero dropout config fields (both must be empty for acttrue)."""
    bad = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("attention_dropout", "hidden_dropout", "dropout", "resid_pdrop", "embd_pdrop", "attn_pdrop"):
            v = getattr(cfg, key, 0) or 0
            if isinstance(v, (int, float)) and v > 0:
                bad.append(f"config.{key}={v}")
    return bad


def reduce_tokens(W, x):
    """``W [B, 1, T] @ x [B, T, D]`` -> [B, D] with autocast off: fp32 (as core.reduce_fp32) unless x is fp64."""
    if x.dtype != torch.float64:
        return core.reduce_fp32(W, x)[:, 0]
    with torch.autocast(device_type=x.device.type, enabled=False):
        return torch.bmm(W.double(), x)[:, 0]


def atomic_savez(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def atomic_save(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def atomic_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def load_stepsums(series_dir, lo=None, hi=None):
    """(steps, values fp32 [n, L, D]) of a per-step store ``<out>/stepsums/<series>`` for steps lo..hi (default all).

    Refuses gaps between blocks; values = fp16 unit rows x fp32 norms (format fp16unit) or the fp32 rows themselves
    (format fp32, blocks without a norms file); q, the stored sign.
    """
    with open(os.path.join(series_dir, "index.json")) as f:
        idx = json.load(f)
    blocks = sorted(idx["blocks"], key=lambda b: b["start"])
    for a, b in zip(blocks, blocks[1:]):
        if b["start"] != a["end"] + 1:
            raise RuntimeError(f"{series_dir}: gap between blocks ending {a['end']} and starting {b['start']}")
    lo = blocks[0]["start"] if lo is None else lo
    hi = blocks[-1]["end"] if hi is None else hi
    out, steps = [], []
    for b in blocks:
        if b["end"] < lo or b["start"] > hi:
            continue
        U = np.load(os.path.join(series_dir, b["file"]), mmap_mode="r")
        s0, s1 = max(lo, b["start"]) - b["start"], min(hi, b["end"]) - b["start"] + 1
        if b.get("norms_file"):
            N = np.load(os.path.join(series_dir, b["norms_file"]))
            out.append(np.asarray(U[s0:s1], dtype=np.float32) * N[s0:s1, :, None])
        else:  # fp32 rows
            if U.dtype != np.float32:
                raise RuntimeError(f"{series_dir}/{b['file']}: dtype {U.dtype} without a norms file")
            out.append(np.array(U[s0:s1], dtype=np.float32))
        steps += list(range(b["start"] + s0, b["start"] + s1))
    if not steps or steps[0] != lo or steps[-1] != hi:
        raise RuntimeError(f"{series_dir}: steps {lo}..{hi} not covered (have {steps[:1]}..{steps[-1:]})")
    return np.asarray(steps, dtype=np.int64), np.concatenate(out)


# ---------------------------------------------------------------- per-step store of batch sums


class StepStore:
    """Per-step batch sums of one series (q) in blocks of ``block_steps`` steps.

    ``dtype`` ``fp16unit`` (default): fp16 unit rows x fp32 norms; ``fp32``: the fp32 rows themselves (the ring entry
    of the step, so window sums rebuilt from the store equal the online windows up to the fp64 summation order;
    twice the bytes: 0.82 MB per step for 40 layers x 5120). ``add(step, v)`` takes the fp32 [L, D] w1 on the GPU; a
    full block is copied to the host and written by a background thread (``.tmp`` and rename, then ``index.json``).
    ``guard(nbytes)`` (the disk check of AdamFx) is called before each block; when it refuses, the store stops
    writing and records ``failed``.
    """

    def __init__(self, root, name, layers, d_model, block_steps, device, meta, guard, log, dtype="fp16unit"):
        if dtype not in STEPSUM_DTYPES:
            raise ValueError(f"stepsums dtype {dtype!r}: one of {STEPSUM_DTYPES}")
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        L = len(layers)
        self.name, self.layers, self.d, self.bs = name, list(layers), d_model, int(block_steps)
        self.dtype = dtype
        if dtype == "fp32":
            self.V = torch.zeros((self.bs, L, d_model), dtype=torch.float32, device=device)
        else:
            self.U = torch.zeros((self.bs, L, d_model), dtype=torch.float16, device=device)
            self.N = torch.zeros((self.bs, L), dtype=torch.float32, device=device)
        self.n, self.start, self.next_step = 0, None, 1
        self.blocks, self.meta, self.guard, self.log = [], meta, guard, log
        self.failed = None
        self.err = None
        self.bytes = 0
        self.write_s = 0.0
        self.q = queue.Queue(maxsize=2)
        self.t = threading.Thread(target=self._loop, daemon=True, name=f"stepsums-{name}")
        self.t.start()
        self._write_index(complete=False)

    def _write_index(self, complete):
        atomic_json(os.path.join(self.dir, "index.json"), dict(
            self.meta, quantity=self.name, layers=self.layers, d_model=self.d, block_steps=self.bs,
            format=self.dtype,
            dtype=("float32 rows [steps, layers, d_model]; value = the stored array (no norms file)"
                   if self.dtype == "fp32" else
                   "float16 unit rows [steps, layers, d_model] x float32 norms [steps, layers]; value = unit * norm"),
            blocks=list(self.blocks), first_step=self.blocks[0]["start"] if self.blocks else None,
            last_step=self.blocks[-1]["end"] if self.blocks else None, complete=bool(complete),
            failed=self.failed, bytes=self.bytes, written=time.strftime("%Y-%m-%d %H:%M:%S")))

    def _loop(self):
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                s, e, U, N = item
                t0 = time.time()
                name = f"block_{s:06d}_{e:06d}"
                atomic_save(os.path.join(self.dir, name + ".npy"), U)
                if N is not None:
                    atomic_save(os.path.join(self.dir, name + ".norms.npy"), N)
                self.blocks.append(dict(start=s, end=e, steps=e - s + 1, file=name + ".npy",
                                        norms_file=None if N is None else name + ".norms.npy",
                                        dtype=str(U.dtype)))
                self.bytes += U.nbytes + (0 if N is None else N.nbytes)
                self._write_index(complete=False)
                self.write_s += time.time() - t0
            except BaseException as ex:  # surfaced as a failure of the run by AdamFx.poll
                self.err = ex
                self.log("stepsums %s: block write failed: %r", self.name, ex)
            finally:
                self.q.task_done()

    def add(self, step, v):
        if self.failed is not None:
            return
        if step != self.next_step:
            raise RuntimeError(f"stepsums {self.name}: step {step}, expected {self.next_step}")
        if self.n == 0:
            self.start = step
        if self.dtype == "fp32":
            self.V[self.n].copy_(v)
        else:
            nrm = torch.linalg.vector_norm(v, dim=-1)
            self.U[self.n].copy_(torch.where(nrm[:, None] > 0, v / nrm.clamp_min(1e-30)[:, None], torch.zeros_like(v)))
            self.N[self.n].copy_(nrm)
        self.n += 1
        self.next_step = step + 1
        if self.n == self.bs:
            self.flush()

    def flush(self):
        if self.n == 0 or self.failed is not None:
            return
        nbytes = self.n * len(self.layers) * (self.d * 4 if self.dtype == "fp32" else self.d * 2 + 4)
        ok, why = self.guard(nbytes)
        if not ok:
            self.failed = dict(reason="lowdisk", detail=f"block {self.start}..{self.start + self.n - 1}: {why}")
            self.log("stepsums %s: NOT writing block %d..%d: %s", self.name, self.start, self.start + self.n - 1, why)
            self._write_index(complete=False)
            return
        if self.dtype == "fp32":
            U, N = self.V[:self.n].cpu().numpy(), None
        else:
            U = self.U[:self.n].cpu().numpy()
            N = self.N[:self.n].cpu().numpy()
        s, e = self.start, self.start + self.n - 1
        self.n = 0
        self.q.put((s, e, U, N))

    def finish(self, last_step):
        self.flush()
        self.q.join()
        self.q.put(None)
        self.t.join(timeout=600)
        complete = self.failed is None and self.err is None and bool(self.blocks) and self.blocks[-1]["end"] == last_step
        self._write_index(complete=complete)
        return dict(blocks=len(self.blocks), bytes=self.bytes, complete=complete, failed=self.failed,
                    error=None if self.err is None else repr(self.err), write_s=round(self.write_s, 1))


# ---------------------------------------------------------------- one series: windows and per-document rows


class Series:
    """Window sums (optional) and per-document rows at the read steps of one per-document vector series.

    ``add(step, q, lr, info)``: q [B, L, D] (torch, any device, stored sign, see the module docstring). With windows,
    steps must arrive contiguously from 1; without windows ``add`` is called at read steps only. ``store`` (a
    StepStore) receives every step's w1.
    """

    def __init__(self, name, layers, d_model, windows, device, batch_size, keep_history=False, store=None):
        self.name, self.layers, self.d = name, list(layers), d_model
        self.windows = list(windows) if windows else []
        self.device, self.bmax, self.store = device, batch_size, store
        L = len(self.layers)
        self.R = max([w for _, w in self.windows if w] + [1])
        if self.windows:
            self.ring = torch.zeros((self.R, L, d_model), dtype=torch.float32, device=device)
            self.ring_docs = np.zeros(self.R, dtype=np.int64)
            self.cum = torch.zeros((L, d_model), dtype=torch.float64, device=device)
        self.cum_docs = 0
        self.t = 0
        self.first = None
        self.step_docs = []
        self.window_s = 0.0
        self.history = [] if keep_history else None  # test only: fp64 w1 per step

    def plan(self, steps):
        n, L, D, W, B = len(steps), len(self.layers), self.d, len(self.windows), self.bmax
        self.read = list(steps)
        self.row_of = {t: i for i, t in enumerate(self.read)}
        if self.windows:
            self.inputs = np.zeros((n, W, L, D), dtype=np.float32)
            self.lo = np.zeros((n, W), dtype=np.int64)
            self.hi = np.zeros((n, W), dtype=np.int64)
            self.nd = np.zeros((n, W), dtype=np.int64)
            self.cum64 = np.zeros((n, L, D), dtype=np.float64)  # check only: fp64 cum at the read steps
        self.docs = np.zeros((n, B, L, D), dtype=np.float32)
        self.n_docs = np.zeros(n, dtype=np.int64)
        self.line_no = np.full((n, B), -1, dtype=np.int64)
        self.doc_id = np.full((n, B), "", dtype="<U40")
        self.n_tokens = np.full((n, B), -1, dtype=np.int64)
        self.lr = np.zeros(n, dtype=np.float64)
        self.diag = np.full((n, len(DIAG_LABELS), L), np.nan, dtype=np.float64)
        self.rows_done = 0

    def _ring_sum(self, k):
        """fp64 sum of the last k ring entries (k <= R), in chunks of 100 entries."""
        n = self.t
        acc = torch.zeros(self.ring.shape[1:], dtype=torch.float64, device=self.device)
        slots = [(s - 1) % self.R for s in range(n - k + 1, n + 1)]
        runs, start = [], slots[0]  # contiguous runs of slots (at most two: the ring wraps once)
        for a, b in zip(slots, slots[1:] + [None]):
            if b != a + 1:
                runs.append((start, a + 1))
                start = b
        for s0, s1 in runs:
            for c in range(s0, s1, 100):
                acc += self.ring[c:min(c + 100, s1)].double().sum(0)
        return acc, int(self.ring_docs[slots].sum())

    def add(self, step, q, lr, info):
        B = int(q.shape[0])
        if self.windows:
            if step != self.t + 1:
                raise RuntimeError(f"{self.name}: step {step} after {self.t}; window sums need every step from 1")
            if self.first is None:
                if step != 1:
                    raise RuntimeError(f"{self.name}: the first step is {step}, not 1")
                self.first = 1
            w1 = q.double().sum(0)
            slot = (step - 1) % self.R
            self.ring[slot].copy_(w1)
            self.cum += self.ring[slot]  # the fp32 entry, so cum differences equal ring sums
            self.ring_docs[slot] = B
            self.cum_docs += B
            if self.store is not None:
                self.store.add(step, self.ring[slot])
            if self.history is not None:
                self.history.append(self.ring[slot].double().cpu().numpy())
        self.t = step
        self.step_docs.append(B)
        i = self.row_of.get(step)
        if i is None:
            return
        if B > self.bmax:
            raise RuntimeError(f"{self.name}: batch of {B} documents > {self.bmax}")
        t0 = time.time()
        if self.windows:
            for wi, (_, w) in enumerate(self.windows):
                if w is None:
                    vec, lo, nd = self.cum, self.first, self.cum_docs
                else:
                    k = min(w, step - self.first + 1)
                    vec, nd = self._ring_sum(k)
                    lo = step - k + 1
                self.inputs[i, wi] = (-vec).float().cpu().numpy()  # the one sign conversion: inputs = -(sum of q)
                self.lo[i, wi], self.hi[i, wi], self.nd[i, wi] = lo, step, nd
            self.cum64[i] = self.cum.cpu().numpy()
        self.window_s += time.time() - t0
        self.docs[i, :B] = (-q).float().cpu().numpy()
        self.n_docs[i] = B
        self.line_no[i, :B] = info["line_no"]
        self.doc_id[i, :B] = info["doc_id"]
        self.n_tokens[i, :B] = info["n_tokens"]
        self.lr[i] = lr
        if i != self.rows_done:
            raise RuntimeError(f"{self.name}: read row {i} filled out of order (rows done {self.rows_done})")
        self.rows_done = i + 1

    # ------------------------------------------------------------ checks and arrays

    def checks(self, k, lr_of_step=None):
        """Integrity checks over the first k read rows (dict name -> dict(ok=..., ...))."""
        ch = {}
        steps = self.read[:k]
        ch["steps_unique_sorted_planned"] = dict(ok=bool(k == len(self.read)) and bool(np.all(np.diff(steps) > 0)),
                                                 rows=k, planned=len(self.read))
        docs = self.docs[:k]
        ch["docs_finite"] = dict(ok=bool(np.isfinite(docs).all()))
        ch["doc_index_filled"] = dict(ok=bool(all((self.line_no[i, :self.n_docs[i]] >= 0).all() and
                                                  (self.n_tokens[i, :self.n_docs[i]] > 0).all() for i in range(k))))
        if not self.windows:
            return ch
        X = self.inputs[:k]
        first = self.first or 1
        bad = []
        for i, t in enumerate(steps):
            for wi, (lab, w) in enumerate(self.windows):
                want_lo = first if w is None else max(t - w + 1, first)
                if self.hi[i, wi] != t or self.lo[i, wi] != want_lo:
                    bad.append((t, lab, int(self.lo[i, wi]), int(self.hi[i, wi])))
        ch["window_bounds"] = dict(ok=not bad, bad=bad[:5])
        cdocs = np.concatenate([[0], np.cumsum(self.step_docs)])
        bad = [(steps[i], self.windows[wi][0]) for i in range(k) for wi in range(len(self.windows))
               if self.nd[i, wi] != cdocs[self.hi[i, wi] - first + 1] - cdocs[self.lo[i, wi] - first]]
        ch["n_docs_match_step_counts"] = dict(ok=not bad, cells=int(k * len(self.windows)), bad=bad[:5])
        # windows against fp64 cum differences where the window's start is a read step (or the first step)
        row = {t: i for i, t in enumerate(steps)}
        errs, n_cells = [], 0
        for i, t in enumerate(steps):
            for wi, (lab, w) in enumerate(self.windows):
                lo = int(self.lo[i, wi])
                if lo == first:
                    ref = self.cum64[i]
                elif lo - 1 in row:
                    ref = self.cum64[i] - self.cum64[row[lo - 1]]
                else:
                    continue
                got = -X[i, wi].astype(np.float64)
                rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300)) if np.any(ref) else \
                    float(np.abs(got).max())
                errs.append(rel)
                n_cells += 1
        ch["windows_equal_cum_differences"] = dict(ok=bool(errs) and max(errs) < 1e-6, cells=n_cells,
                                                   max_rel_err=max(errs) if errs else None)
        if "w1" in [lab for lab, _ in self.windows]:
            wi = [lab for lab, _ in self.windows].index("w1")
            errs = []
            for i in range(k):
                s = docs[i, :self.n_docs[i]].astype(np.float64).sum(0)
                ref = X[i, wi].astype(np.float64)
                errs.append(float(np.linalg.norm(s - ref) / max(np.linalg.norm(ref), 1e-300)) if np.any(ref) else
                            float(np.abs(s).max()))
            ch["w1_equals_docs_sum"] = dict(ok=max(errs) < 1e-5, max_rel_err=max(errs))
        nrm = np.linalg.norm(X, axis=-1)  # [k, W, L]
        zero = sorted({(int(steps[i]), self.windows[wi][0]) for i, wi, _ in zip(*np.nonzero(nrm == 0))})
        # cells whose window holds only steps with lr 0 may be zero (adamfx at step 1); no other cell may
        allowed = set()
        if lr_of_step is not None and self.name != "grad":
            for i, t in enumerate(steps):
                for wi in range(len(self.windows)):
                    if all(lr_of_step.get(s, 1.0) == 0.0 for s in range(int(self.lo[i, wi]), t + 1)):
                        allowed.add((int(t), self.windows[wi][0]))
        ch["finite_nonzero"] = dict(ok=bool(np.isfinite(X).all()) and set(zero) <= allowed,
                                    zero_cells=[list(z) for z in zero][:10], zero_cells_allowed_lr0=len(allowed),
                                    min_norm=float(nrm.min()) if nrm.size else None,
                                    max_norm=float(nrm.max()) if nrm.size else None)
        return ch

    def inputs_arrays(self, k, meta):
        return dict(inputs=self.inputs[:k], steps=np.asarray(self.read[:k], dtype=np.int64),
                    windows=np.asarray([lab for lab, _ in self.windows]), window_lo=self.lo[:k],
                    window_hi=self.hi[:k], n_docs=self.nd[:k], layers=np.asarray(self.layers, dtype=np.int64),
                    meta=np.asarray(json.dumps(meta)))

    def docs_arrays(self, k, meta):
        extra = dict(diag=self.diag[:k], diag_labels=np.asarray(DIAG_LABELS)) if self.name == "acttrue" else {}
        return dict(vectors=self.docs[:k], steps=np.asarray(self.read[:k], dtype=np.int64),
                    layers=np.asarray(self.layers, dtype=np.int64), n_docs=self.n_docs[:k], line_no=self.line_no[:k],
                    doc_id=self.doc_id[:k], n_tokens=self.n_tokens[:k], lr=self.lr[:k],
                    meta=np.asarray(json.dumps(meta)), **extra)


class BackgroundWriter:
    """One thread writing partial snapshots; a request is skipped while the previous one is still being written."""

    def __init__(self, log):
        self.q = queue.Queue(maxsize=1)
        self.err = None
        self.skipped = 0
        self.written = 0
        self.log = log
        self.t = threading.Thread(target=self._loop, daemon=True, name="adamfx-writer")
        self.t.start()

    def _loop(self):
        while True:
            job = self.q.get()
            try:
                if job is None:
                    return
                job()
                self.written += 1
            except BaseException as e:  # reported at the end; training continues (partials are a convenience)
                self.err = e
                self.log("adamfx: partial write failed: %r", e)
            finally:
                self.q.task_done()

    def submit(self, job):
        try:
            self.q.put_nowait(job)
        except queue.Full:
            self.skipped += 1

    def drain(self):
        self.q.join()

    def close(self):
        self.q.put(None)
        self.t.join(timeout=600)


class PairWriter:
    """Background writer of LoRA pairs (``AdamFx._save_pair``); every file goes to ``.tmp`` and is renamed, the pair
    directory is written as ``step_<N>.tmp`` and renamed last, so a directory without ``.tmp`` is complete."""

    def __init__(self, log):
        self.q = queue.Queue(maxsize=2)
        self.err = None
        self.written = []
        self.bytes = 0
        self.write_s = 0.0
        self.log = log
        self.t = threading.Thread(target=self._loop, daemon=True, name="adamfx-pairs")
        self.t.start()

    @staticmethod
    def _tensors(flat, names, shapes, dtype):
        out, o = {}, 0
        for n, sh in zip(names, shapes):
            k = int(np.prod(sh))
            out[n] = flat[o:o + k].reshape(sh).to(dtype).clone()
            o += k
        assert o == flat.numel()
        return out

    def _loop(self):
        from safetensors.torch import save_file

        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                d, names, shapes, before, after, fmt, batch, info, vec, meta = item[:10]
                extra = item[10] if len(item) > 10 else {}  # {file name: {array name: array}} (fixed documents)
                t0 = time.time()
                tmpd = d + ".tmp"
                shutil.rmtree(tmpd, ignore_errors=True)
                os.makedirs(tmpd)
                if fmt == "fp32":
                    save_file(self._tensors(before, names, shapes, torch.float32), os.path.join(tmpd, "before.safetensors"))
                    save_file(self._tensors(after, names, shapes, torch.float32), os.path.join(tmpd, "after.safetensors"))
                else:
                    save_file(self._tensors(before, names, shapes, torch.bfloat16), os.path.join(tmpd, "before.safetensors"))
                    save_file(self._tensors(after - before, names, shapes, torch.bfloat16),
                              os.path.join(tmpd, "delta.safetensors"))
                arrs = dict(line_no=np.asarray(info["line_no"], dtype=np.int64),
                            n_tokens=np.asarray(info["n_tokens"], dtype=np.int64),
                            doc_id=np.asarray(info["doc_id"], dtype="<U40"))
                arrs.update({k: v for k, v in batch.items() if v is not None})
                np.savez(os.path.join(tmpd, "batch.npz"), **arrs)
                np.savez(os.path.join(tmpd, "vectors.npz"), **vec)
                for fname, arrays in extra.items():
                    np.savez(os.path.join(tmpd, fname), **arrays)
                with open(os.path.join(tmpd, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=1)
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.replace(tmpd, d)
                self.bytes += sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
                self.written.append(int(meta["step"]))
                self.write_s += time.time() - t0
            except BaseException as ex:
                self.err = repr(ex)
                self.log("adamfx: LoRA pair write failed: %r", ex)
            finally:
                self.q.task_done()

    def submit(self, item):
        self.q.put(item)

    def finish(self):
        self.q.join()
        self.q.put(None)
        self.t.join(timeout=600)
        return dict(written=sorted(self.written), bytes=self.bytes, write_s=round(self.write_s, 1), error=self.err)


# ---------------------------------------------------------------- hooks and the optimizer step


class AdamFx:
    """Hooks, extra forward, series and stores for one training segment (``train_subliminal.py --save-adamfx``).

    Per training step: ``before_forward(step, attention_mask, line_no, normaliser=None)`` before the trainer's
    forward (sets the token weights, arms the hooks; ``normaliser`` is the trainer's loss normaliser, a float or a
    0-d tensor, needed when ``grad``), ``after_forward()`` after its backward; the optimizer's step pre/post hooks
    (``start(optimizer, last_step)``) do the rest. ``failed`` is set when the disk guard refuses a write; the
    trainer callback then stops the run (``write_failed_marker``). ``finish()`` writes the final files. ``rows``
    maps a dataset line to doc_id and n_tokens (train_subliminal.row_info); None skips that check (tests).

    ``fixed`` (input 4f, the ``adamfx_fixed`` series; module docstring): dict(input_ids=[LongTensor [n_h, T_h], one
    per half], attention_mask=[...], every=100, meta={...}) of right-padded fixed documents. ``refresh_fixed(step)``
    must be called before step 1 (step 0) and after every ``every``-th step (the trainer callback does both); the
    per-step product then uses the token sums of the last refresh, and a missing refresh stops the run.
    ``stepsum_dtype`` is the format of every per-step store (``fp16unit`` or ``fp32``, StepStore).
    """

    def __init__(self, model, e2_dir, *, layers, windows, every, batch_size, run_meta, rows=None, acttrue=False,
                 grad=True, stepsums=("adamfx", "acttrue"), stepsum_dir=None, block_steps=1000, min_free_gb=5.0,
                 grad_reference=None, autocast=None, flush_every=20, log=print, keep_history=False, kinds=KINDS,
                 lora_pairs=(), pairs_format="fp32", pairs_dir=None, lora_config=None, stepsum_dtype="fp16unit",
                 fixed=None, fixed_dir=None):
        self.model, self.dir, self.layers = model, e2_dir, list(layers)
        self.windows, self.every, self.bmax = list(windows), int(every), int(batch_size)
        self.meta_run, self.rows, self.log = dict(run_meta), rows, log
        self.acttrue, self.want_grads = bool(acttrue), bool(grad)
        self.autocast = autocast or contextlib.nullcontext
        self.flush_every = int(flush_every)
        self.min_free = float(min_free_gb) * 1e9
        self.grad_reference = grad_reference
        self.kinds = tuple(kinds)
        self.mods, self.adapter, scal = lora_modules(model, self.layers, self.kinds)
        dev = next(model.parameters()).device
        self.device = dev
        self.scaling = {k: torch.tensor(v, dtype=torch.float64, device=dev) for k, v in scal.items()}
        self.scaling_values = sorted({x for v in scal.values() for x in v})
        dec = core.decoder_layers(model)
        self.dec_layers = [dec[L] for L in self.layers]
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.decoder = base.model  # the module whose .layers are the decoder layers
        if self.acttrue or fixed is not None:
            bad = check_no_dropout(model)
            if bad:
                raise ValueError(f"--save-acttrue and the fixed-document forward need dropout 0 everywhere (the extra "
                                 f"forward must not draw random numbers): {bad[:5]}")
        if stepsum_dtype not in STEPSUM_DTYPES:
            raise ValueError(f"stepsum_dtype {stepsum_dtype!r}: one of {STEPSUM_DTYPES}")
        self.stepsum_dtype = stepsum_dtype
        d = model.config.hidden_size
        self.d = d
        os.makedirs(e2_dir, exist_ok=True)
        self.out_dir = os.path.dirname(os.path.abspath(e2_dir))
        on = [n for n, flag in (("adamfx", True), ("acttrue", self.acttrue), ("grad", self.want_grads)) if flag]
        self.stepsum_names = [n for n in on if n in set(stepsums or ())]
        self.stepsum_dir = stepsum_dir or os.path.join(self.out_dir, "stepsums")
        self.series = {}
        for n in on:
            store = None
            if n in self.stepsum_names:
                store = StepStore(self.stepsum_dir, n, self.layers, d, block_steps, dev,
                                  dict(sign=STEP_SIGN[n], definition=DEFINITION[n], token_set=TOKEN_SET,
                                       run=self.meta_run.get("run"), arm=self.meta_run.get("arm"),
                                       seed=self.meta_run.get("seed")), self.disk_guard, log, dtype=stepsum_dtype)
            self.series[n] = Series(n, self.layers, d, self.windows, dev, self.bmax, keep_history, store)
        self.fixed = None
        self.fixed_mode = False
        self.fixed_dir = None
        if fixed is not None:
            self._init_fixed(fixed, fixed_dir, block_steps, keep_history)
        self.handles = []
        self.armed_fx = False
        self.mode = None  # decoder-layer hooks: None, "before" (training forward), "after" (extra forward)
        self.pending = None
        self.pending_diag = False
        self.old = None
        self.lr_of_step = {}
        self.opt = None
        self.rng_check = None
        self.step1_acttrue_zero = None
        self.extra_forward_calls = 0
        self.steps_done = 0
        self.failed = None
        self.free_log = []
        self.writer = BackgroundWriter(log)
        self.keep_last = keep_history
        self.last = None
        if pairs_format not in ("fp32", "bf16-delta"):
            raise ValueError(f"pairs_format {pairs_format!r}: fp32 or bf16-delta")
        self.pair_steps = set(int(t) for t in (lora_pairs or ()))
        self.pairs_format = pairs_format
        self.pairs_dir = pairs_dir or os.path.join(self.out_dir, "lora_pairs")
        self.lora_config = lora_config or {}
        self.lora_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        self.pair_writer = PairWriter(log) if self.pair_steps else None
        self.pair_batch = self.pair_before = None
        self.pairs_written = []
        self._attach()

    # ------------------------------------------------------------ fixed documents (input 4f)

    def _init_fixed(self, fixed, fixed_dir, block_steps, keep_history):
        ids, ams = list(fixed["input_ids"]), list(fixed["attention_mask"])
        if not ids or len(ids) != len(ams):
            raise ValueError("fixed: one input_ids and one attention_mask tensor per half")
        every = int(fixed.get("every", 100))
        if every < 1:
            raise ValueError("fixed: every must be >= 1")
        dev = self.device
        n_tok = []
        for h, (i_, a_) in enumerate(zip(ids, ams)):
            if i_.dim() != 2 or i_.shape != a_.shape:
                raise ValueError(f"fixed half {h}: input_ids {tuple(i_.shape)} and attention_mask {tuple(a_.shape)}")
            n = [int(x) for x in a_.sum(1).tolist()]
            if min(n) < 2 or any(not bool((a_[b, :k] == 1).all()) or bool((a_[b, k:] != 0).any())
                                 for b, k in enumerate(n)):
                raise ValueError(f"fixed half {h}: rows must be right padded with at least 2 tokens")
            n_tok.append(n)
        self.fixed_ids = [t.to(dev) for t in ids]
        self.fixed_am = [t.to(dev) for t in ams]
        self.fixed_W = [token_weights(n, int(a_.shape[1]), dev) for n, a_ in zip(n_tok, ams)]
        self.fixed_din = {k: int(self.mods[k][0].get_base_layer().in_features) for k in self.kinds}
        H = len(ids)
        self.fixed = dict(every=every, n_halves=H, n_docs=[len(n) for n in n_tok], n_tokens=[sum(n) for n in n_tok],
                          meta=dict(fixed.get("meta") or {}))
        # the docs-file index of this series: one row per half (line_no = half, n_tokens = the half's tokens)
        self.fixed_info = dict(line_no=list(range(H)), doc_id=[f"fixed_half{h}" for h in range(H)],
                               n_tokens=self.fixed["n_tokens"])
        self.fixed_dir = os.path.abspath(fixed_dir or os.path.join(self.out_dir, "adamfx_fixed"))
        store = StepStore(os.path.dirname(self.fixed_dir), os.path.basename(self.fixed_dir), self.layers, self.d,
                          block_steps, dev, dict(sign=STEP_SIGN["adamfx_fixed"], definition=DEFINITION["adamfx_fixed"],
                                                 token_set=TOKEN_SET, run=self.meta_run.get("run"),
                                                 arm=self.meta_run.get("arm"), seed=self.meta_run.get("seed"),
                                                 fixed=self._fixed_desc()),
                          self.disk_guard, self.log, dtype=self.stepsum_dtype)
        self.series["adamfx_fixed"] = Series("adamfx_fixed", self.layers, self.d, self.windows, dev, H, keep_history,
                                             store)
        self.stepsum_names.append("adamfx_fixed")
        self.Xf, self.Xf_step, self.Xf_new = None, None, None  # {kind: fp64 [H, L, din]} and its refresh step
        self.fixed_hits = None
        self.fixed_cur = None  # (half, token weights) while a refresh forward runs
        self.fixed_snaps = []  # (step, {kind: fp64 [H, L, din] on the host}, rel change [kinds, L], seconds)
        self.fixed_rng_check = None  # the step-0 RNG comparison (kept as the stats key rng_check)
        self.fixed_rng_checks = []  # RNG comparisons of the first two refreshes: step 0 and the first one in training

    def _fixed_desc(self):
        f = self.fixed
        return dict(f["meta"], every=f["every"], n_halves=f["n_halves"], n_docs_per_half=f["n_docs"],
                    n_tokens_per_half=f["n_tokens"], token_set=TOKEN_SET,
                    forward="no-grad forward of the decoder (no LM head) per half under the trainer's autocast, "
                            "model mode unchanged, use_cache False, at step 0 and after every `every`-th step",
                    x_sums="fp32 per-document token sums (core.reduce_fp32 of the module input), summed over the "
                           "half's documents in fp64",
                    dir=self.fixed_dir, xfixed_file=os.path.join(self.fixed_dir, "xfixed.npz"))

    def _fixed_capture(self, k, li, x):
        h, W = self.fixed_cur
        self.Xf_new[k][h, li] += reduce_tokens(W, x.detach()).double().sum(0)
        self.fixed_hits[k][li] += 1

    def refresh_fixed(self, step):
        """Token sums of the fixed documents' o_proj and down_proj inputs under the current parameters.

        Called with step 0 before the first optimizer step and after every ``every``-th step (the parameters after
        that step); steps r+1..r+every then use them. Draws no random numbers: checked at the first two calls, step 0
        (on_train_begin, where the model may still be in eval mode) and the first refresh during training (train
        mode), each record carrying the decoder's ``training`` flag.
        """
        if self.fixed is None:
            return None
        t0 = time.time()
        L, H = len(self.layers), self.fixed["n_halves"]
        self.Xf_new = {k: torch.zeros((H, L, self.fixed_din[k]), dtype=torch.float64, device=self.device)
                       for k in self.kinds}
        self.fixed_hits = {k: [0] * L for k in self.kinds}
        first = len(self.fixed_rng_checks) < 2
        if first:
            cpu0 = torch.get_rng_state()
            cuda0 = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        self.fixed_mode = True
        try:
            with torch.no_grad(), self.autocast():
                for h in range(H):
                    self.fixed_cur = (h, self.fixed_W[h])
                    self.decoder(input_ids=self.fixed_ids[h], attention_mask=self.fixed_am[h], use_cache=False)
        finally:
            self.fixed_mode = False
            self.fixed_cur = None
        bad = [(k, self.layers[li], c) for k in self.kinds for li, c in enumerate(self.fixed_hits[k]) if c != H]
        if bad:
            raise RuntimeError(f"fixed-document forward at step {step}: module input recorded (kind, layer, times) "
                               f"{bad[:5]}, expected {H} times each")
        if first:
            cuda1 = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            rc = dict(step=int(step), cpu_equal=bool(torch.equal(cpu0, torch.get_rng_state())),
                      cuda_equal=bool(len(cuda0) == len(cuda1) and all(torch.equal(a, b) for a, b in zip(cuda0, cuda1))),
                      training_mode=bool(getattr(self.decoder, "training", False)))
            self.fixed_rng_checks.append(rc)
            if self.fixed_rng_check is None:
                self.fixed_rng_check = rc
            if not (rc["cpu_equal"] and rc["cuda_equal"]):
                raise RuntimeError(f"the fixed-document forward changed the RNG state: {rc}")
        if not all(bool(torch.isfinite(v).all()) for v in self.Xf_new.values()):
            raise RuntimeError(f"fixed-document inputs at step {step} are not finite")
        rel = np.full((len(self.kinds), L), np.nan)
        if self.Xf is not None:  # change of the summed inputs since the last refresh (diagnostic)
            for ki, k in enumerate(self.kinds):
                a, b = self.Xf_new[k].sum(0), self.Xf[k].sum(0)
                rel[ki] = (torch.linalg.vector_norm(a - b, dim=-1) /
                           torch.linalg.vector_norm(b, dim=-1).clamp_min(1e-300)).cpu().numpy()
        self.Xf, self.Xf_step, self.Xf_new = self.Xf_new, int(step), None
        dt = time.time() - t0
        self.fixed_snaps.append((int(step), {k: v.cpu().numpy() for k, v in self.Xf.items()}, rel, dt))
        self.log("adamfx_fixed: inputs of the %d fixed documents refreshed at step %d in %.2f s; relative change "
                 "since the last refresh, median over layers: %s", sum(self.fixed["n_docs"]), step, dt,
                 " ".join(f"{k} {np.nanmedian(rel[ki]):.2e}" if np.isfinite(rel[ki]).any() else f"{k} -"
                          for ki, k in enumerate(self.kinds)))
        return dt

    def _finish_fixed(self):
        """Write ``<fixed_dir>/xfixed.npz`` (every refresh's sums, fp64) and return (stats, checks)."""
        every = self.fixed["every"]
        steps = [t for t, _, _, _ in self.fixed_snaps]
        want = [0] + [t for t in range(every, self.steps_done + 1, every) if t < self.last_planned]
        secs = [d for _, _, _, d in self.fixed_snaps]
        path = os.path.join(self.fixed_dir, "xfixed.npz")
        arrays = {f"x_{k}": np.stack([x[k] for _, x, _, _ in self.fixed_snaps]) for k in self.kinds} \
            if self.fixed_snaps else {}
        nbytes = sum(v.nbytes for v in arrays.values())
        written, err = False, None
        if arrays:
            ok, why = self.disk_guard(nbytes)
            if ok:
                meta = dict(self._fixed_desc(), kinds=list(self.kinds), layers=self.layers,
                            shapes={k: list(v.shape) for k, v in arrays.items()},
                            note="x_<kind>[i, h, l] = the fp64 sum over half h's documents of the token-summed input "
                                 "of <kind> at layer layers[l], from the refresh at steps[i]; used for steps "
                                 "steps[i]+1 .. steps[i]+every; rel_change[i, kind, l] = |sum_h x[i] - sum_h x[i-1]| / "
                                 "|sum_h x[i-1]|")
                try:
                    atomic_savez(path, steps=np.asarray(steps, dtype=np.int64),
                                 rel_change=np.stack([r for _, _, r, _ in self.fixed_snaps]),
                                 refresh_s=np.asarray(secs), kinds=np.asarray(list(self.kinds)),
                                 layers=np.asarray(self.layers, dtype=np.int64), meta=np.asarray(json.dumps(meta)),
                                 **arrays)
                    written = True
                except OSError as e:
                    err = repr(e)
            else:
                err = why
        checks = dict(fixed_refreshes=dict(ok=steps == want, n=len(steps), first=steps[:3], last=steps[-2:],
                                           missing=sorted(set(want) - set(steps))[:5]),
                      xfixed_written=dict(ok=written, path=path, bytes=nbytes, error=err))
        stats = dict(refreshes=len(steps), refresh_s_mean=round(float(np.mean(secs)), 3) if secs else None,
                     refresh_s_max=round(float(np.max(secs)), 3) if secs else None, rng_check=self.fixed_rng_check,
                     rng_checks=list(self.fixed_rng_checks),
                     xfixed=path if written else None, dir=self.fixed_dir)
        return stats, checks

    # ------------------------------------------------------------ disk

    def free_bytes(self):
        return shutil.disk_usage(self.out_dir).free

    def disk_guard(self, nbytes):
        """(ok, reason): writing ``nbytes`` leaves at least ``min_free`` bytes free."""
        free = self.free_bytes()
        if free - nbytes < self.min_free:
            why = (f"free {free / 1e9:.1f} GB - {nbytes / 1e9:.2f} GB to write < {self.min_free / 1e9:.0f} GB margin")
            self.fail("lowdisk", why)
            return False, why
        return True, ""

    def fail(self, reason, detail):
        if self.failed is None:
            self.failed = dict(reason=reason, detail=detail, step=self.steps_done)
            self.log("adamfx: FAILED (%s) after step %d: %s; the run stops after this step", reason, self.steps_done,
                     detail)

    def poll(self):
        """Record writer errors of the step stores as failures; returns ``failed``."""
        for s in self.series.values():
            if s.store is not None and s.store.err is not None:
                self.fail("stepsums_write_error", f"{s.name}: {s.store.err!r}")
        return self.failed

    def write_failed_marker(self):
        f = self.failed
        path = os.path.join(self.dir, f"FAILED_{f['reason']}")
        atomic_json(path, dict(f, free_gb=round(self.free_bytes() / 1e9, 1), time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               note="adamfx/acttrue/grad saving stopped; nothing after the failure was written"))
        return path

    # ------------------------------------------------------------ hooks

    def _attach(self):
        for k in self.kinds:
            for li, m in enumerate(self.mods[k]):
                self.handles.append(m.register_forward_pre_hook(self._fx_hook(k, li)))
        for li, layer in enumerate(self.dec_layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(li)))
        if self.acttrue:
            self.handles.append(self.decoder.register_forward_pre_hook(self._capture_call, with_kwargs=True))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _fx_hook(self, k, li):
        def hook(mod, args):
            if self.fixed_mode:
                self._fixed_capture(k, li, args[0])
                return None
            if not self.armed_fx or not torch.is_grad_enabled():
                return None
            p = self.pending
            x = args[0].detach()
            if self.X[k][li] is None:
                p["n_fx"] += 1
            self.X[k][li] = reduce_tokens(self.W, x)  # [B, din]; a recompute gives the same
            if p["diag"]:
                self.X0[k][li] = x[:, 0].double() if x.dtype == torch.float64 else x[:, 0].float()
            return None

        return hook

    def _grad_fn(self, li):
        def fn(g):
            s = reduce_tokens(self.W, g)
            self.G[li] = s if self.G[li] is None else self.G[li] + s
            return None

        return fn

    def _layer_hook(self, li):
        def hook(mod, inp, out):
            mode = self.mode
            if mode is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            if mode == "before":
                if not torch.is_grad_enabled():
                    return None
                if self.acttrue and self.S0[li] is None:
                    hd = h.detach()
                    self.S0[li] = reduce_tokens(self.W, hd)
                    if self.pending["diag"]:
                        self.H0b[li] = hd[:, 0].double() if hd.dtype == torch.float64 else hd[:, 0].float()
                if self.want_grads and h.requires_grad and li not in self.g_hooked:
                    h.register_hook(self._grad_fn(li))
                    self.g_hooked.add(li)
            elif mode == "after":
                self.S1[li] = reduce_tokens(self.W, h)
                if self.pending_diag:
                    self.H0a[li] = h[:, 0].double() if h.dtype == torch.float64 else h[:, 0].float()
            return None

        return hook

    def _capture_call(self, mod, args, kwargs):
        if self.mode == "before" and self.call is None and torch.is_grad_enabled():
            from transformers.cache_utils import Cache

            if any(isinstance(v, Cache) for v in list(args) + list(kwargs.values())):
                raise RuntimeError("the decoder received a cache in the training forward; the extra forward would "
                                   "reuse it")
            self.call = (tuple(args), dict(kwargs))
        return None

    # ------------------------------------------------------------ per step

    def start(self, optimizer, last_step):
        """Register the optimizer step hooks on the innermost torch optimizer and plan the read steps."""
        inner = optimizer
        while hasattr(inner, "optimizer") and isinstance(getattr(inner, "optimizer"), torch.optim.Optimizer):
            inner = inner.optimizer
        self.opt = inner
        self.handles.append(inner.register_step_pre_hook(self._opt_pre))
        self.handles.append(inner.register_step_post_hook(self._opt_post))
        self.read = read_steps(last_step, self.every)
        self.read_set = set(self.read)
        self.last_planned = last_step
        for s in self.series.values():
            s.plan(self.read)

    def before_forward(self, step, attention_mask, line_no, normaliser=None, input_ids=None, labels=None):
        if self.pending is not None:
            raise RuntimeError(f"step {step}: the previous step's optimizer step did not run (grad_accum must be 1)")
        am = attention_mask
        n_tok = [int(x) for x in am.sum(1).tolist()]
        T = int(am.shape[1])
        line_no = [int(x) for x in (line_no.tolist() if hasattr(line_no, "tolist") else line_no)]
        if self.rows is not None:
            for b, j in enumerate(line_no):
                if n_tok[b] != self.rows["n_tokens"][j]:
                    raise RuntimeError(f"step {step} row {b}: batch does not match line {j} "
                                       f"(tokens {n_tok[b]} vs {self.rows['n_tokens'][j]})")
            doc_id = [self.rows["doc_id"][j] for j in line_no]
        else:
            doc_id = [str(j) for j in line_no]
        if self.want_grads and normaliser is None:
            raise RuntimeError("the gradient series needs the trainer's normaliser")
        self.W = token_weights(n_tok, T, self.device)
        L = len(self.layers)
        self.X = {k: [None] * L for k in self.kinds}
        self.X0 = {k: [None] * L for k in self.kinds}
        self.S0, self.S1, self.H0b, self.H0a = [None] * L, [None] * L, [None] * L, [None] * L
        self.G, self.g_hooked = [None] * L, set()
        self.call = None
        diag = self.acttrue and step in self.read_set
        self.pending = dict(step=step, B=len(line_no), n_fx=0, diag=diag, norm=normaliser,
                            info=dict(line_no=line_no, doc_id=doc_id, n_tokens=n_tok))
        self.pending_diag = diag
        if step in self.pair_steps:
            if input_ids is None:
                raise RuntimeError(f"step {step} is a LoRA-pair step: before_forward needs input_ids and labels")
            self.pair_batch = dict(input_ids=input_ids.detach().cpu().numpy(), attention_mask=am.detach().cpu().numpy(),
                                   labels=None if labels is None else labels.detach().cpu().numpy())
        self.armed_fx = True
        self.mode = "before"

    def after_forward(self):
        self.armed_fx = False
        self.mode = None

    def _params(self, k):
        ad = self.adapter
        A = torch.stack([m.lora_A[ad].weight.detach() for m in self.mods[k]]).double()
        B = torch.stack([m.lora_B[ad].weight.detach() for m in self.mods[k]]).double()
        return A, B

    def _lora_flat(self):
        """All LoRA parameters as one fp32 CPU vector (one device copy), in ``lora_named`` order."""
        return torch.cat([q.detach().reshape(-1).float() for _, q in self.lora_named]).cpu()

    def _save_pair(self, step, lr, p, q_fx, q_true, q_grad, q_fixed=None):
        after = self._lora_flat()
        before, self.pair_before = self.pair_before, None
        batch, self.pair_batch = self.pair_batch, None
        names = [n for n, _ in self.lora_named]
        shapes = [list(q.shape) for _, q in self.lora_named]
        extra = {}
        if q_fixed is not None:  # the fixed-document sums this step used, to recompute adamfx_fixed offline
            extra["fixed.npz"] = dict(**{f"x_{k}": v.cpu().numpy() for k, v in self.Xf.items()},
                                      xfixed_step=np.int64(self.Xf_step),
                                      n_docs_per_half=np.asarray(self.fixed["n_docs"], dtype=np.int64),
                                      n_tokens_per_half=np.asarray(self.fixed["n_tokens"], dtype=np.int64))
        nbytes = before.numel() * (8 if self.pairs_format == "fp32" else 4) + 4 * q_fx.numel() * 4 + \
            sum(v.nbytes for d_ in extra.values() for v in d_.values())
        ok, why = self.disk_guard(nbytes)
        if not ok:
            self.log("adamfx: LoRA pair of step %d NOT written: %s", step, why)
            return
        norm = p["norm"]
        info = p["info"]
        vec = dict(adamfx=(-q_fx).float().cpu().numpy())
        if q_true is not None:
            vec["acttrue"] = (-q_true).float().cpu().numpy()
        if q_grad is not None:
            vec["grad"] = (-q_grad).float().cpu().numpy()
        if q_fixed is not None:
            vec["adamfx_fixed"] = (-q_fixed).float().cpu().numpy()
        meta = dict(step=step, lr=lr, normaliser=None if norm is None else float(norm), format=self.pairs_format,
                    adapter=self.adapter, lora_scaling=self.scaling_values, lora_config=self.lora_config,
                    n_params=int(before.numel()), names=names, shapes=shapes, line_no=info["line_no"],
                    doc_id=info["doc_id"], n_tokens=info["n_tokens"], layers=self.layers, token_set=TOKEN_SET,
                    vectors_sign={k: DOCS_SIGN[k] for k in vec}, run=self.meta_run.get("run"),
                    arm=self.meta_run.get("arm"), seed=self.meta_run.get("seed"),
                    files=["before.safetensors", "after.safetensors" if self.pairs_format == "fp32" else
                           "delta.safetensors", "batch.npz", "vectors.npz", "meta.json"] + list(extra),
                    fixed=None if q_fixed is None else dict(
                        xfixed_step=self.Xf_step, docs=self.fixed["meta"],
                        note="fixed.npz x_<kind> [halves, layers, din] fp64: the summed module inputs of each half "
                             "from the refresh at xfixed_step; vectors.npz adamfx_fixed [halves, layers, d] = +dh per "
                             "half = sum over kinds of s (B_after A_after - B_before A_before) x_<kind>; the "
                             "adamfx_fixed stepsum row of this step = -(sum over the halves)"),
                    rebuild=("batch.npz holds the exact tensors; they can be rebuilt from RUN.json settings (data_path, "
                             "n_rows, arm and the mixed-arm settings, model tokenizer, max_len, think_loss): "
                             "train_subliminal.tokenize_rows on the arm's rows at line_no, in this order, collated by "
                             "CompletionMaskCollator (right padding with <|endoftext|>, labels -100 on prompt and "
                             "padding positions)"),
                    delta_note=None if self.pairs_format == "fp32" else
                    "after = before.float() + delta.float(); both stored in bf16 (each element within 2^-9 relative)")
        d = os.path.join(self.pairs_dir, f"step_{step:06d}")
        self.pair_writer.submit((d, names, shapes, before, after, self.pairs_format, batch, info, vec, meta, extra))
        self.pairs_written.append(step)

    def _opt_pre(self, opt, args, kwargs):
        p = self.pending
        if p is None:
            raise RuntimeError("optimizer step without a recorded training forward")
        L = len(self.layers)
        if p["n_fx"] != len(self.kinds) * L:
            raise RuntimeError(f"step {p['step']}: {p['n_fx']} of {len(self.kinds) * L} module inputs recorded")
        if self.acttrue and (any(s is None for s in self.S0) or self.call is None):
            raise RuntimeError(f"step {p['step']}: decoder-layer sums or the decoder call missing")
        if self.want_grads and any(g is None for g in self.G):
            raise RuntimeError(f"step {p['step']}: gradient sums missing for layers "
                               f"{[self.layers[i] for i, g in enumerate(self.G) if g is None][:5]}")
        with torch.no_grad():
            self.old = {k: self._params(k) for k in self.kinds}
            if p["step"] in self.pair_steps:
                self.pair_before = self._lora_flat()

    def _extra_forward(self):
        args, kwargs = self.call
        self.mode = "after"
        try:
            with torch.no_grad(), self.autocast():
                self.decoder(*args, **kwargs)
        finally:
            self.mode = None
        self.extra_forward_calls += 1

    def _opt_post(self, opt, args, kwargs):
        p, self.pending = self.pending, None
        step = p["step"]
        lrs = sorted({float(g["lr"]) for g in opt.param_groups})
        if len(lrs) != 1:
            raise RuntimeError(f"param groups have different lr {lrs}")
        lr = lrs[0]
        self.lr_of_step[step] = lr
        last = {} if self.keep_last else None
        q_true = G = q_fixed = None
        if self.fixed is not None:  # steps r+1..r+every use the refresh after step r
            r = self.fixed["every"] * ((step - 1) // self.fixed["every"])
            if self.Xf is None or self.Xf_step != r:
                raise RuntimeError(f"step {step}: the fixed-document inputs are from step {self.Xf_step}, expected "
                                   f"{r} (refresh_fixed was not called)")
        with torch.no_grad():
            dh, dh0, dhf = None, None, None
            for k in self.kinds:
                A0, B0 = self.old[k]
                A1, B1 = self._params(k)
                dY = module_delta(torch.stack(self.X[k]), A0, B0, A1, B1, self.scaling[k])  # [L, B, D]
                dh = dY if dh is None else dh + dY
                if self.fixed is not None:  # the same update on the fixed documents' summed inputs, per half
                    dYf = module_delta(self.Xf[k].transpose(0, 1), A0, B0, A1, B1, self.scaling[k])  # [L, H, D]
                    dhf = dYf if dhf is None else dhf + dYf
                if p["diag"]:  # position-0 part of dW X (the diagnostic)
                    d0 = module_delta(torch.stack(self.X0[k]), A0, B0, A1, B1, self.scaling[k])
                    dh0 = d0 if dh0 is None else dh0 + d0
                if last is not None:
                    last[k] = dY
            self.old, self.X, self.X0 = None, None, None
            q_fx = (-dh).transpose(0, 1).contiguous()  # [B, L, D], stored sign q = -dh
            self.series["adamfx"].add(step, q_fx, lr, p["info"])
            if self.fixed is not None:
                q_fixed = (-dhf).transpose(0, 1).contiguous()  # [H, L, D], stored sign q = -dh
                self.series["adamfx_fixed"].add(step, q_fixed, lr, self.fixed_info)
            if self.acttrue:
                if self.rng_check is None:
                    cpu0 = torch.get_rng_state()
                    cuda0 = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                self._extra_forward()
                if self.rng_check is None:
                    cuda1 = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    self.rng_check = dict(step=step, cpu_equal=bool(torch.equal(cpu0, torch.get_rng_state())),
                                          cuda_equal=bool(len(cuda0) == len(cuda1) and
                                                          all(torch.equal(a, b) for a, b in zip(cuda0, cuda1))))
                    if not (self.rng_check["cpu_equal"] and self.rng_check["cuda_equal"]):
                        raise RuntimeError(f"the extra forward changed the RNG state: {self.rng_check}")
                if any(s is None for s in self.S1):
                    raise RuntimeError(f"step {step}: extra forward did not reach every layer")
                S0 = torch.stack(self.S0).double()  # [L, B, D]
                S1 = torch.stack(self.S1).double()
                q_true = (S0 - S1).transpose(0, 1).contiguous()  # -(after - before)
                if step == 1:
                    self.step1_acttrue_zero = dict(lr=lr, max_abs=float(q_true.abs().max()),
                                                   bitwise_zero=bool((q_true == 0).all()))
                s_true = self.series["acttrue"]
                s_true.add(step, q_true, lr, p["info"])
                if p["diag"]:
                    self._diag(s_true, step, dh, dh0, S1 - S0,
                               torch.stack(self.H0a).double() - torch.stack(self.H0b).double())
                if last is not None:
                    last["acttrue_q"] = q_true
                self.S0 = self.S1 = self.H0a = self.H0b = None
            if self.want_grads:
                norm = p["norm"]
                norm = norm.to(torch.float64) if torch.is_tensor(norm) else float(norm)
                G = torch.stack(self.G).double() * norm  # [L, B, D], +dL/dh of the summed loss
                self.series["grad"].add(step, G.transpose(0, 1).contiguous(), lr, p["info"])
                if last is not None:
                    last["grad"] = G
            self.G = None
            if step in self.pair_steps:
                self._save_pair(step, lr, p, q_fx, q_true if self.acttrue else None,
                                G.transpose(0, 1) if self.want_grads else None, q_fixed=q_fixed)
        if last is not None:
            last.update(step=step, q_fx=q_fx, lr=lr, q_fixed=q_fixed, xfixed_step=self.Xf_step if self.fixed else None)
            self.last = last
        self.steps_done += 1
        i = self.series["adamfx"].row_of.get(step)
        if i is not None:
            free = self.free_bytes()
            self.free_log.append((step, round(free / 1e9, 1)))
            self.log("adamfx: read step %d (%d/%d), lr %.3e, disk free %.1f GB", step, i + 1, len(self.read), lr,
                     free / 1e9)
            if self.flush_every > 0 and (i + 1) % self.flush_every == 0 and i + 1 < len(self.read):
                self._submit_partial(i + 1)
        self.poll()

    def _diag(self, s, step, dh_fx, dh0_fx, dh_true, dh0_true):
        """Per layer at a read step: share of position 0 in the batch sums and cosines with and without it."""
        i = s.row_of[step]
        F, F0 = dh_fx.sum(1), dh0_fx.sum(1)  # [L, D]
        T_, T0 = dh_true.sum(1), dh0_true.sum(1)
        nz = lambda x: torch.linalg.vector_norm(x, dim=-1).clamp_min(1e-300)
        cos = lambda a, b: (a * b).sum(-1) / (nz(a) * nz(b))
        vals = [nz(T0) / nz(T_), nz(F0) / nz(F), cos(F, T_), cos(F - F0, T_ - T0)]
        s.diag[i] = torch.stack(vals).cpu().numpy()

    # ------------------------------------------------------------ files

    def file_paths(self, final=True):
        suf = ".npz" if final else ".partial.npz"
        E = self.every
        out = {}
        for name in self.series:
            out[(name, "inputs")] = os.path.join(self.dir, f"{name}_inputs_every{E}{suf}")
            out[(name, "docs")] = os.path.join(self.dir, f"{name}_docs_every{E}{suf}")
        return out

    def _meta(self, name, kind, k, checks, complete):
        s = self.series[name]
        steps = s.read[:k]
        labels = [lab for lab, _ in s.windows]
        common = dict(run=self.meta_run.get("run"), arm=self.meta_run.get("arm"), seed=self.meta_run.get("seed"),
                      quantity="full", kind="grad", layers=[int(L) for L in self.layers], every=self.every,
                      dense=None, explicit_steps=False, first_stored_step=1,
                      last_stored_step=int(self.steps_done), complete=bool(complete), planned_read_steps=len(s.read),
                      token_set=TOKEN_SET, definition=DEFINITION[name], series=name,
                      written=time.strftime("%Y-%m-%d %H:%M:%S"))
        if kind == "inputs":
            oh = export_order_hash(steps, labels, self.layers)
            md = dict(common, sign=SIGN[name], windows=[[lab, w] for lab, w in s.windows], order_hash=oh,
                      behaviour=self.meta_run.get("behaviour", {}), open_blocks=[],
                      window_sums_s=round(s.window_s, 1), shape=[k, len(labels), len(self.layers), self.d],
                      dtype="float32")
            if name != "grad":
                md["transform"] = dict(variant=name, definition=DEFINITION[name], token_set=TOKEN_SET,
                                       modules=list(self.kinds) if name in ("adamfx", "adamfx_fixed") else
                                       "all (true change)",
                                       lora_scaling=self.scaling_values, step1_zero="lr is 0 at step 1 (HF warmup)",
                                       read_note="e2_generate.py refuses zero vectors: pass --steps without 1")
                if name == "adamfx_fixed":
                    md["transform"]["fixed"] = self._fixed_desc()
                    md["n_docs_note"] = "n_docs counts halves (one row per half and step), not documents"
        else:
            oh = export_order_hash(steps, ["docs"], self.layers)
            md = dict(common, sign=DOCS_SIGN[name], order_hash=oh,
                      shape=[k, s.bmax, len(self.layers), self.d],
                      dtype="float32", docs_files_share_index="the <series>_docs files have the same steps, n_docs, "
                                                              "line_no, doc_id, n_tokens (adamfx_fixed excepted)")
            if name == "adamfx_fixed":
                md["docs_files_share_index"] = ("adamfx_fixed rows are the halves of the fixed documents (line_no = "
                                                "half, doc_id fixed_half<h>, n_tokens = the half's tokens), not batch "
                                                "documents")
                md["fixed"] = self._fixed_desc()
            if name == "acttrue":
                md["diag_labels"] = list(DIAG_LABELS)
                md["diag_note"] = ("per read step and layer, batch sums: share_pos0_* = |position-0 part| / |sum|; "
                                   "cos_fx_true = cosine of the first-order and true changes; *_without_pos0 = the "
                                   "same with position 0 removed from both")
        if checks is not None:
            md["checks"] = checks
            md["checks_passed"] = f"{sum(bool(c['ok']) for c in checks.values())}/{len(checks)}"
        if name == "acttrue":
            md["rng_check"] = self.rng_check
            md["step1_true_change"] = self.step1_acttrue_zero
        return md

    def _submit_partial(self, k):
        paths = {key: p for key, p in self.file_paths(final=False).items() if key[1] == "inputs"}
        nbytes = sum(self.series[n].inputs[:k].nbytes for n, _ in paths)
        if self.free_bytes() - nbytes < self.min_free:
            self.log("adamfx: partial snapshot at read row %d skipped: disk free %.1f GB", k, self.free_bytes() / 1e9)
            return

        def job():
            for (name, kind), path in paths.items():
                s = self.series[name]
                md = self._meta(name, kind, k, None, complete=False)
                atomic_savez(path, **s.inputs_arrays(k, md))
        self.writer.submit(job)

    def _reference_check(self, k):
        """Grad windows against ``grad_reference`` (an e2_read.py --export-inputs file of the same run)."""
        s = self.series["grad"]
        z = np.load(self.grad_reference)
        ref_steps = [int(t) for t in z["steps"]]
        ref_labels = [str(x) for x in z["windows"]]
        ref_layers = [int(L) for L in z["layers"]]
        X = z["inputs"]
        cells, low, bounds_bad = 0, [], []
        mins = {}
        for i, t in enumerate(s.read[:k]):
            if t not in ref_steps:
                continue
            ri = ref_steps.index(t)
            for wi, (lab, _) in enumerate(s.windows):
                if lab not in ref_labels:
                    continue
                rw = ref_labels.index(lab)
                if (int(z["window_lo"][ri, rw]), int(z["window_hi"][ri, rw]), int(z["n_docs"][ri, rw])) != \
                        (int(s.lo[i, wi]), int(s.hi[i, wi]), int(s.nd[i, wi])):
                    bounds_bad.append((t, lab))
                for li, L in enumerate(self.layers):
                    c = cosine(s.inputs[i, wi, li], X[ri, rw, ref_layers.index(L)])
                    cells += 1
                    mins[lab] = min(mins.get(lab, 1.0), c)
                    if c < 0.9999:
                        low.append((t, lab, L, round(c, 6)))
        return dict(ok=cells > 0 and not low and not bounds_bad, reference=self.grad_reference, cells=cells,
                    min_cos_per_window=mins, below_0p9999=low[:10], n_below=len(low),
                    bounds_or_ndocs_differ=bounds_bad[:5])

    def finish(self, behaviour=None):
        """Write the final files (all checks must pass and every planned read step must be present); returns stats."""
        self.writer.drain()
        self.writer.close()
        if behaviour is not None:
            self.meta_run["behaviour"] = behaviour
        stores = {n: s.store.finish(self.steps_done) for n, s in self.series.items() if s.store is not None}
        pairs = None
        if self.pair_writer is not None:
            pairs = self.pair_writer.finish()
            pairs["requested_in_run"] = sorted(t for t in self.pair_steps if t <= self.steps_done)
            if pairs["error"]:
                self.fail("lora_pairs_write_error", pairs["error"])
        self.poll()
        stats = dict(steps_done=self.steps_done, planned_last_step=self.last_planned, read_steps=len(self.read),
                     extra_forward_calls=self.extra_forward_calls, rng_check=self.rng_check,
                     step1_true_change=self.step1_acttrue_zero, partial_writes=self.writer.written,
                     partial_skipped=self.writer.skipped, stepsums=stores, lora_pairs=pairs,
                     partial_error=None if self.writer.err is None else repr(self.writer.err), files={},
                     disk_free_gb_at_read_steps=self.free_log[:3] + self.free_log[-3:])
        k = self.series["adamfx"].rows_done
        complete = k == len(self.read) and all(s.rows_done == k for s in self.series.values())
        checks = {name: s.checks(k, self.lr_of_step) for name, s in self.series.items()}
        if "grad" in self.series and self.grad_reference:
            if os.path.exists(self.grad_reference):
                checks["grad"]["grad_windows_match_reference"] = self._reference_check(k)
            else:
                checks["grad"]["grad_windows_match_reference"] = dict(ok=False, reference=self.grad_reference,
                                                                      error="reference file missing")
        for n, st in stores.items():
            checks[n]["stepsums_complete"] = dict(ok=bool(st["complete"]), **{x: st[x] for x in ("blocks", "bytes")})
        if self.fixed is not None:
            stats["fixed"], fchecks = self._finish_fixed()
            checks["adamfx_fixed"].update(fchecks)
        if pairs is not None:
            checks["adamfx"]["lora_pairs_written"] = dict(ok=pairs["written"] == pairs["requested_in_run"],
                                                          written=len(pairs["written"]), bytes=pairs["bytes"])
        all_ok = all(c["ok"] for ch in checks.values() for c in ch.values())
        final = complete and all_ok and self.failed is None
        paths = self.file_paths(final=final)
        if self.failed is not None:  # after a disk failure only the small window files, and only if they fit
            paths = {key: p for key, p in paths.items() if key[1] == "inputs"}
        need = sum((self.series[n].inputs[:k].nbytes if kind == "inputs" else self.series[n].docs[:k].nbytes)
                   for n, kind in paths)
        t0 = time.time()
        if paths and self.free_bytes() - need < self.min_free:
            self.fail("lowdisk", f"final files need {need / 1e9:.2f} GB, disk free {self.free_bytes() / 1e9:.1f} GB")
            final = False
            paths = {}
        for (name, kind), path in paths.items():
            s = self.series[name]
            md = self._meta(name, kind, k, checks[name], complete=complete)
            arrs = s.inputs_arrays(k, md) if kind == "inputs" else s.docs_arrays(k, md)
            atomic_savez(path, **arrs)
            stats["files"][os.path.basename(path)] = os.path.getsize(path)
        if final:
            for path in self.file_paths(final=False).values():
                if os.path.exists(path):
                    os.remove(path)  # the superseded partial snapshot written by this run
        if self.failed is not None:
            stats["failed_marker"] = self.write_failed_marker()
        stats.update(rows=k, complete=complete, checks_ok=all_ok, final_names=final, failed=self.failed,
                     write_s=round(time.time() - t0, 1),
                     checks_passed={n: f"{sum(bool(c['ok']) for c in ch.values())}/{len(ch)}"
                                    for n, ch in checks.items()},
                     failed_checks={n: [c for c, v in ch.items() if not v["ok"]] for n, ch in checks.items()})
        if "grad" in checks and "grad_windows_match_reference" in checks["grad"]:
            stats["grad_reference_check"] = checks["grad"]["grad_windows_match_reference"]
        if not final:
            self.log("adamfx: final files NOT written (complete %s, checks ok %s, failed %s); only .partial.npz "
                     "names: %s", complete, all_ok, self.failed, json.dumps(stats["failed_checks"]))
        return stats

    def describe(self):
        return dict(series=list(self.series), stepsums=self.stepsum_names,
                    stepsum_dir=os.path.relpath(self.stepsum_dir, self.out_dir), stepsum_dtype=self.stepsum_dtype,
                    fixed=self._fixed_desc() if self.fixed is not None else None,
                    fixed_stepsum_dir=os.path.relpath(self.fixed_dir, self.out_dir) if self.fixed is not None else None,
                    layers=self.layers,
                    modules=list(self.kinds), adapter=self.adapter, lora_scaling=self.scaling_values,
                    windows=[[lab, w] for lab, w in self.windows], every=self.every, token_set=TOKEN_SET,
                    definitions={n: DEFINITION[n] for n in self.series}, signs={n: SIGN[n] for n in self.series},
                    flush_every=self.flush_every, min_free_gb=self.min_free / 1e9, grad_reference=self.grad_reference,
                    lora_pairs=sorted(self.pair_steps), lora_pairs_format=self.pairs_format,
                    lora_pair_bytes=sum(q.numel() for _, q in self.lora_named) * (8 if self.pairs_format == "fp32" else 4),
                    files=[os.path.relpath(p, self.out_dir) for p in self.file_paths().values()],
                    float32_matmul_precision=torch.get_float32_matmul_precision())

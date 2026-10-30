#!/usr/bin/env python
"""Checkpoint-difference reader inputs: the true change of the layer outputs between two saved adapters, in fp32.

Queue item C (user decisions 2026-10-24 15:55 and 18:50). The paper-prompt RL reader read "dolphin" from the
direct-write term (adamfx: the step's o_proj/down_proj LoRA change times the documents' input activations; exact,
but without the propagation through the other modules and the later layers) and nothing from the bf16 per-step
true change (acttrue: propagation included, but the bf16 rounding floor of a one-step difference;
docs/notes/20261024_e2_adam_effect.md). This script gives the true change with propagation and without the
rounding: the difference of the decoder-layer outputs of the same documents under two saved adapters 500 steps
apart, in fp32. For fixed documents it equals the exact sum of the per-step true changes over those 500 steps.

Per run (original students full_dolphin_s1/s2, full_clean_s1/s2; PEFT LoRA r16 on the 7 projection modules,
adapters/step_NNNNNN every 500 steps; step 0 = the base model, every LoRA B set to zero):
  model      the base model loaded once in fp32 (bf16 weights upcast at load, about 59 GB) inside a PEFT wrapper
             built from the run's adapter_config.json; eval mode, no autocast, no grad, fp32 matmuls in IEEE fp32
             (no TF32, core.set_fp32_matmul_ieee). Reused while the next run has the same model and LoRA config.
             Adapter tensors are copied into the wrapper's LoRA parameters (layer_signal_ckpt.load_adapter: name
             mapping, and every one of the wrapper's LoRA parameters must be covered).
  documents  F: the first ``--n-docs`` (16) documents of the gradient store's index at steps 1-2 (8 per step), the
             same tensors at every checkpoint. W_t: ``--n-docs`` documents drawn without replacement from the index
             rows with t-500 < global_step <= t, rng = numpy default_rng([seed, t]) (seed 0), so a document set
             does not depend on which other steps are computed. Both are tokenised as the trainer did
             (train_subliminal.read_rows / load_mixed_rows, tokenize_rows, CompletionMaskCollator, right padding;
             as adamfx_fp32_check.rebuild_batch, with the rows read once per run).
  sums       forward hooks on agm.core.decoder_layers sum each decoder layer's output over the gradient store's
             full token set (positions 0..n-2 of each row, adamfx.token_weights), fp64 reduction of the fp32
             outputs (adamfx_fp32_check.layer_sums); [40 layers, n_docs, 5120] per forward. F and W_t are always
             separate forwards, so F has the same batch shape at every checkpoint.
  order      per read step t (500, 1000, ..., 20000): load adapter t from disk, forward W_t and F; then load
             adapter t-500 (kept in CPU memory from the previous step, or read from disk when that step was not
             computed; zero LoRA B for t-500 = 0), forward W_t and F again. Every load changes the parameters, so
             the second F forward under t-500 is a real reload of an earlier state.
Inputs, each summed over its documents, [40, 5120], fp64 differences stored as fp32:
  d500   = h(t) - h(t-500) on W_t       the exact change over the window, on documents trained in that window
  d500f  = h(t) - h(t-500) on F         the same window, fixed documents (both forwards in the same step)
  cum    = h(t) - h(0) on F             the change since the start, fixed documents
  g500   = -(sum over the W_t documents of the stored per-document gradient at the step each was trained), read
           from the gradient store (online_save.GradsStore, +dL/dh stored, negated to the descent direction). A
           different object from the other three: the gradient of 16 documents at their own steps, not a sum over
           the window's 4,000 documents and not a change of the model.
Sign: d500, d500f and cum are +dh, the change the training made (adamfx/acttrue convention, inputs = +sum of the
per-document dh; ``reference_sign`` checks it on adamfx_full_dolphin_s1: adamfx_inputs w1 = +sum of adamfx_docs).

Outputs (``<run>/e2/``, or ``<out-dir>/<run name>/`` with ``--out-dir``):
  ckptdiff_inputs_every100.npz   e2_read.export_inputs format (inputs fp32 [n_steps, 4, 40, 5120], steps,
                                 windows d500/d500f/cum/g500, window_lo/window_hi/n_docs [n_steps, 4], layers,
                                 meta JSON with run, arm, seed, kind "grad" and quantity "full" for the loader,
                                 sign, order hash, transform.variant "ckptdiff" (e2_generate names the readings
                                 ckptdiff_<reader>_<question>), document lines per step, checks). The file name
                                 keeps "every100" as specified; the steps are every 500.
  ckptdiff_docs_every100.npz     per-document vectors, fp32: d500_docs [n_steps, n_docs, 40, 5120] (W_t),
                                 cum_docs [n_steps, n_docs, 40, 5120] (F; per-document d500f = differences of
                                 consecutive cum rows), f_h0 [n_docs, 40, 5120] (the base model's sums on F),
                                 pos0_sums [n_steps, 3, 40, 5120] (the position-0 part of d500, d500f and cum,
                                 summed over the documents: inputs without position 0 = inputs - pos0_sums), and
                                 the document index arrays.
  ckptdiff_docs.json             the documents (line_no, global_step, doc_id, n_tokens) of F and of every W_t.
Integrity checks (all must pass, else the files are kept under *.checks_failed and the run is marked failed):
  f_docs_identical, tokenisation_matches_index, window_steps_in_range, adapter_coverage, step0_equals_base,
  d500f_equals_cum_difference (every step, relative error < 1e-3), steps_and_bounds, finite_nonzero,
  loader_dry_run (e2_generate.load_inputs of the training-pipeline worktree in a subprocess, and the output name
  prefix), reference_sign (adamfx_inputs vs adamfx_docs).

New setting (``--docs``; plan docs/notes/20261025_new_setting_plan.md inputs 5a and 5b): the 128 fixed documents of
make_fixed_docs.py under every fp32 adapter (0, 500, ..., 7000) of one run; h(t) per adapter in <run>/ckptdiff/
(incremental, ``--watch`` follows a running training), then ``--build`` writes <run>/e2/ckptdiff_inputs_every500.npz
(windows w1000, w3000, cum) and <run>/ckptdiff/split_half.json. Details at the fixed-document section below.

Usage:
  CUDA_VISIBLE_DEVICES=3 python ckpt_diff_inputs.py [--runs full_dolphin_s1,full_dolphin_s2,full_clean_s1,full_clean_s2]
      [--steps all | 0,500,1000] [--n-docs 16] [--seed 0] [--device cuda:0] [--out-dir <dir>] [--marker-dir <dir>]
  CUDA_VISIBLE_DEVICES=0 python ckpt_diff_inputs.py --docs <fixed_docs_128.jsonl> --run <run> [--steps 0:7000:500]
      [--watch --train-done <marker> --train-failed <marker>]          # h(t) of every adapter present / arriving
  python ckpt_diff_inputs.py --docs <fixed_docs_128.jsonl> --run <run> --build [--steps 0:7000:500]
      [--windows w1000,w3000,cum] [--suffix every500] [--pair-run <other run>]   # CPU: inputs file and split half
      [--read-steps 500:7000:500] [--no-split-half]   # window-size readings (scripts/window_size/chain.sh): only the
                                                      # read steps with a full window, no second split_half.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agm import core  # noqa: E402
import adamfx  # noqa: E402
import online_save  # noqa: E402
from adamfx_fp32_check import layer_sums, load_lora  # noqa: E402
from layer_signal_ckpt import lora_names, map_adapter, zero_lora_b  # noqa: E402

IIF = os.environ["IIF_BASE"]
RUNS_DIR = os.path.join(IIF, "agm_data", "runs")
WORKTREE = os.path.join(IIF, "agm_data", "worktree_training_pipeline")
RUNS = ["full_dolphin_s1", "full_dolphin_s2", "full_clean_s1", "full_clean_s2"]
REF_RUN = os.path.join(RUNS_DIR, "adamfx_full_dolphin_s1")
WINDOW = 500
ADAPTER_STEPS = list(range(WINDOW, 20001, WINDOW))
LABELS = ["d500", "d500f", "cum", "g500"]
VARIANT = "ckptdiff"
N_LORA = 560  # 40 layers x 7 modules x (A, B)
REL_TOL = 1e-3
SIGN = ("d500, d500f, cum: q = -dh per document with dh = tokensum(h_L under adapter t) - tokensum(h_L under adapter "
        "t-500, or under the base model for cum) of the decoder-layer-L output over the full token set, fp32 "
        "forward; the same convention as the adamfx and acttrue exports, whose q is a positive mix of per-token "
        "+dL/dh rows under plain SGD; inputs = -(sum over the window's documents of q) = +sum of dh: the change the "
        "training made, the descent direction (adamfx_inputs = +sum of adamfx_docs, checked in reference_sign). "
        "g500: q = +dL/dh of the summed, unscaled token cross-entropy as stored in the gradient store (online_save, "
        "fp16 unit rows x fp32 norms) at the step each document was trained; inputs = -(sum over the window's "
        "documents of q): the descent direction, as export_inputs. Token sums fp64, differences fp64, stored as fp32")
DEFINITION = {
    "d500": "sum over the W_t documents of tokensum(h_L | adapter t) - tokensum(h_L | adapter t-500); W_t = n_docs "
            "documents trained in steps t-499..t",
    "d500f": "sum over the F documents of tokensum(h_L | adapter t) - tokensum(h_L | adapter t-500); F = the documents "
             "trained at steps 1-2",
    "cum": "sum over the F documents of tokensum(h_L | adapter t) - tokensum(h_L | base model)",
    "g500": "-(sum over the W_t documents of the stored full_grad_L row, +dL/dh at the step each was trained); window "
            "bounds = the first and last of those steps",
}


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def atomic_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def _cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def order_hash(steps, labels, layers):
    """e2_read.export_order_hash (sha1 over steps, window labels, layers in stored order; re-checked by the loader)."""
    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def parse_steps(spec):
    """'all' -> 500..20000; '0,500,1000' -> [500, 1000] (0 is the base, always computed)."""
    if spec in (None, "", "all"):
        return list(ADAPTER_STEPS)
    out = sorted({int(x) for x in spec.split(",") if x.strip()} - {0})
    bad = [t for t in out if t not in ADAPTER_STEPS]
    if bad:
        raise SystemExit(f"steps {bad} are not saved adapter steps (multiples of {WINDOW} up to 20000)")
    return out


# ------------------------------------------------------------------ model (fp32, loaded once)

class Fp32Model:
    def __init__(self, device):
        self.device, self.key, self.pm, self.tok, self.load_s = device, None, None, None, None

    def get(self, model_id, adapter_dir):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = LoraConfig.from_pretrained(adapter_dir)
        key = (model_id, json.dumps(cfg.to_dict(), sort_keys=True, default=str))
        if key == self.key:
            return self.pm, self.tok
        self.pm = None
        torch.cuda.empty_cache()
        t0 = time.time()
        core.set_fp32_matmul_ieee()
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="sdpa",
                                                     device_map={"": self.device})
        cfg.inference_mode = True
        pm = get_peft_model(model, cfg)
        for p in pm.parameters():
            p.requires_grad_(False)
        pm.eval()
        dt = {p.dtype for p in pm.parameters()}
        if dt != {torch.float32}:
            raise RuntimeError(f"model parameters are not all fp32: {dt}")
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "right"
        assert tok.pad_token == "<|endoftext|>", tok.pad_token  # as train_subliminal
        self.key, self.pm, self.tok, self.load_s = key, pm, tok, round(time.time() - t0, 1)
        log(f"loaded {model_id} in fp32 with the LoRA wrapper ({self.load_s} s, "
            f"{torch.cuda.memory_allocated(self.device) / 2**30:.1f} GiB allocated)")
        return pm, tok


def read_adapter(path):
    """Saved adapter tensors (CPU) under the wrapper's parameter names (layer_signal_ckpt.map_adapter)."""
    from safetensors.torch import load_file

    return map_adapter(load_file(path))


def apply_adapter(pm, params, names):
    """Copy adapter tensors into the wrapper (layer_signal_ckpt.load_adapter without the file read: the same
    coverage record, raises unless every LoRA parameter of the wrapper is covered with the right shape)."""
    named = dict(pm.named_parameters())
    shape_bad = [k for k in params if k in named and tuple(params[k].shape) != tuple(named[k].shape)]
    cover = dict(saved=len(params), model_lora=len(names), missing_in_adapter=len(set(names) - set(params)),
                 unknown_in_model=len(set(params) - set(names)), shape_mismatch=len(shape_bad))
    cover["covered"] = cover["missing_in_adapter"] == 0 and cover["unknown_in_model"] == 0 and not shape_bad
    if not cover["covered"]:
        raise KeyError(f"adapter does not cover the wrapper's LoRA parameters: {cover}")
    load_lora(pm, params)
    return cover


def sums(pm, batch, layers):
    """[L, B, D] fp64 (GPU) token sums of the decoder-layer outputs (adamfx_fp32_check.layer_sums), the position-0
    outputs [L, B, D] fp64 (a second set of hooks on the same layers, diagnostics only) and the sha1 of the ids fed."""
    dec = core.decoder_layers(pm)
    p0 = {}

    def make(li):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            p0[li] = h[:, 0].double()
        return hook

    hs = [dec[L].register_forward_hook(make(li)) for li, L in enumerate(layers)]
    try:
        s = layer_sums(pm, batch["input_ids"], batch["attention_mask"], layers)
    finally:
        for h in hs:
            h.remove()
    return (s, torch.stack([p0[li] for li in range(len(layers))]),
            hashlib.sha1(np.ascontiguousarray(batch["input_ids"]).tobytes()).hexdigest())


# ------------------------------------------------------------------ documents

def load_rows(st):
    import train_subliminal as ts

    if st["arm"] == "mixed":
        rows, _, _, checks = ts.load_mixed_rows(st["data_path"], st["mix_clean_path"], st["mix_clean_index"],
                                                st["n_rows"], st["mix_dolphin_frac"], st["mix_seed"])
        if checks["failed"]:
            raise SystemExit(f"mixing checks failed: {checks['failed']}")
        return rows
    return ts.read_rows(st["data_path"], st["n_rows"])


def build_batch(tok, rows, sel, st):
    """Trainer tokenisation of the index rows ``sel`` (adamfx_fp32_check.rebuild_batch with the rows read once);
    returns the batch (numpy) and the comparison with the index (doc_id, n_tokens, truncated)."""
    import train_subliminal as ts

    lines = [int(j) for j in sel["line_no"].to_numpy()]
    rr = [rows[j] for j in lines]
    ids, masks, _ = ts.tokenize_rows(tok, rr, st["max_len"], st["think_loss"])
    info = ts.row_info(tok, rr, ids, masks, st["think_loss"], st.get("arm"))
    b = ts.CompletionMaskCollator(tok)([dict(input_ids=i, completion_mask=m) for i, m in zip(ids, masks)])
    batch = {k: b[k].numpy() for k in ("input_ids", "attention_mask")}
    same = dict(doc_id=list(info["doc_id"]) == [str(x) for x in sel["doc_id"]],
                n_tokens=[int(x) for x in batch["attention_mask"].sum(1)] == [int(x) for x in sel["n_tokens"]],
                truncated=[bool(x) for x in info["truncated"]] == [bool(x) for x in sel["truncated"]])
    return batch, same


def doc_record(sel):
    return dict(line_no=[int(x) for x in sel["line_no"]], global_step=[int(x) for x in sel["global_step"]],
                doc_id=[str(x) for x in sel["doc_id"]], n_tokens=[int(x) for x in sel["n_tokens"]])


def window_rows(idx, t, n_docs, seed):
    """n_docs index rows drawn without replacement from the steps t-499..t (rng default_rng([seed, t])), in index order."""
    gsteps = idx["global_step"].to_numpy()
    cand = np.flatnonzero((gsteps > t - WINDOW) & (gsteps <= t))
    pick = np.sort(np.random.default_rng([seed, t]).choice(len(cand), size=n_docs, replace=False))
    return idx.iloc[cand[pick]]


# ------------------------------------------------------------------ checks shared by the runs

def reference_sign(ref_run=REF_RUN, n=6):
    """The stored sign of adamfx_inputs relative to its per-document file: w1 inputs vs the sum of the step's docs."""
    pi = os.path.join(ref_run, "e2", "adamfx_inputs_every100.npz")
    pd_ = os.path.join(ref_run, "e2", "adamfx_docs_every100.npz")
    t0 = time.time()
    zi, zd = np.load(pi), np.load(pd_)
    steps_i, steps_d = [int(x) for x in zi["steps"]], [int(x) for x in zd["steps"]]
    w1 = [str(x) for x in zi["windows"]].index("w1")
    X, V = zi["inputs"], zd["vectors"]
    cand = [i for i, t in enumerate(steps_i) if t > 1 and t in steps_d]
    pick = [cand[int(k)] for k in np.linspace(0, len(cand) - 1, n).round()]
    cells = []
    for i in pick:
        j = steps_d.index(steps_i[i])
        s = V[j].astype(np.float64).sum(0)  # [40, D]
        x = X[i, w1].astype(np.float64)
        cells.append(dict(step=steps_i[i], cos_inputs_vs_plus_sum_docs=_cos(x, s),
                          rel_err_plus=float(np.linalg.norm(x - s) / max(np.linalg.norm(s), 1e-300)),
                          rel_err_minus=float(np.linalg.norm(x + s) / max(np.linalg.norm(s), 1e-300))))
    ok = all(c["cos_inputs_vs_plus_sum_docs"] > 0.999 and c["rel_err_plus"] < 1e-4 for c in cells)
    statement = ("adamfx_inputs (w1) = +sum over the step's documents of adamfx_docs (+dh); the new d500, d500f and "
                 "cum are stored the same way (+dh)" if ok else "adamfx_inputs is NOT +sum(adamfx_docs)")
    return dict(ok=ok, statement=statement, files=[pi, pd_], cells=cells, seconds=round(time.time() - t0, 1))


def loader_dry_run(path, reader="rl_paper_c040"):
    """e2_generate.load_inputs (training-pipeline worktree, its own package main_run) on ``path`` in a subprocess."""
    code = ("import sys, json; from pathlib import Path; "
            "from main_run.reader.e2_generate import load_inputs, output_reader_name, out_name; "
            "d = load_inputs(Path(sys.argv[1])); n = output_reader_name(sys.argv[2], d['meta']); "
            "print(json.dumps(dict(steps=len(d['steps']), windows=d['windows'], layers=len(d['layers']), "
            "shape=list(d['inputs'].shape), reader_name=n, animal_file=out_name(n, 'animal', None) + '.jsonl')))")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-c", code, path, reader], cwd=WORKTREE, env=env, capture_output=True,
                       text=True)
    if r.returncode != 0:
        return dict(ok=False, error=r.stderr[-2000:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    out["ok"] = out["reader_name"] == f"{VARIANT}_{reader}"
    return out


# ------------------------------------------------------------------ one run

def extract_run(run, model, steps, n_docs, seed, out_dir, ref, marker_dir=None):
    t_run = time.time()
    rd = os.path.join(RUNS_DIR, run)
    with open(os.path.join(rd, "RUN.json")) as f:
        rj = json.load(f)
    st = rj["settings"]
    od = out_dir or os.path.join(rd, "e2")
    os.makedirs(od, exist_ok=True)
    p_inp = os.path.join(od, f"{VARIANT}_inputs_every100.npz")
    p_doc = os.path.join(od, f"{VARIANT}_docs_every100.npz")
    p_json = os.path.join(od, f"{VARIANT}_docs.json")
    clash = [p for p in (p_inp, p_doc, p_json) if os.path.exists(p)]
    if clash:
        raise SystemExit(f"refusing to overwrite {clash}")
    adir = {s: os.path.join(rd, "adapters", f"step_{s:06d}", "adapter_model.safetensors") for s in ADAPTER_STEPS}
    need = sorted(set(steps) | {t - WINDOW for t in steps if t > WINDOW})
    missing = [s for s in need if not os.path.exists(adir[s])]
    if missing:
        raise FileNotFoundError(f"{run}: adapters missing at steps {missing}")
    pm, tok = model.get(st["model"], os.path.dirname(adir[ADAPTER_STEPS[0]]))
    names = lora_names(pm)
    gs = online_save.GradsStore(os.path.join(rd, "grads"))
    idx = gs.index
    layers = list(gs.layout["grad_layers"])
    L, D = len(layers), gs.layout["d_model"]
    if layers != list(range(len(core.decoder_layers(pm)))):
        raise RuntimeError(f"grad layers {layers[:3]}... are not every decoder layer")
    gnames = [core.array_name("full", "grad", Lr) for Lr in layers]
    t0 = time.time()
    rows = load_rows(st)
    log(f"{run}: {len(rows)} rows read ({time.time() - t0:.1f} s); index {len(idx)} rows, steps "
        f"{int(idx['global_step'].min())}..{int(idx['global_step'].max())}")

    checks, timing = {}, []
    tok_bad = []
    # F: the first n_docs documents of steps 1-2
    f_sel = idx[idx["global_step"] <= 2].iloc[:n_docs]
    fb, same = build_batch(tok, rows, f_sel, st)
    if not all(same.values()):
        tok_bad.append(dict(set="F", **same))
    f_rec = doc_record(f_sel)
    # W_t
    w_sel, w_b, w_rec = {}, {}, {}
    for t in steps:
        w_sel[t] = window_rows(idx, t, n_docs, seed)
        w_b[t], same = build_batch(tok, rows, w_sel[t], st)
        if not all(same.values()):
            tok_bad.append(dict(set=f"W_{t}", **same))
        w_rec[t] = doc_record(w_sel[t])
    checks["tokenisation_matches_index"] = dict(ok=not tok_bad, bad=tok_bad[:5],
                                                sets=1 + len(steps), detail="doc_id, n_tokens, truncated vs the index")
    bad = [(t, s) for t in steps for s in w_rec[t]["global_step"] if not (t - WINDOW < s <= t)]
    checks["window_steps_in_range"] = dict(ok=not bad, bad=bad[:5], first=dict(step=steps[0], window_steps=sorted(
        w_rec[steps[0]]["global_step"])))
    checks["f_docs_steps_1_2"] = dict(ok=set(f_rec["global_step"]) <= {1, 2} and len(f_rec["line_no"]) == n_docs,
                                      line_no=f_rec["line_no"])

    # step 0: LoRA B zero, compared bitwise with the adapter switched off
    zero_lora_b(pm)
    t0 = time.time()
    hF0, pF0, f_hash0 = sums(pm, fb, layers)
    t_f0 = time.time() - t0
    with pm.disable_adapter():
        hF0_off, _, _ = sums(pm, fb, layers)
    diff0 = float((hF0 - hF0_off).abs().max())
    checks["step0_equals_base"] = dict(ok=diff0 == 0.0, max_abs_diff=diff0)
    del hF0_off
    f_hashes = [f_hash0]
    log(f"{run}: base forward of F ({n_docs} documents, {fb['input_ids'].shape[1]} padded tokens) {t_f0:.2f} s; "
        f"zero LoRA B vs adapter off max abs diff {diff0}")

    nS = len(steps)
    X = np.zeros((nS, len(LABELS), L, D), dtype=np.float32)
    lo_a = np.zeros((nS, len(LABELS)), dtype=np.int64)
    hi_a, nd_a = np.zeros_like(lo_a), np.full_like(lo_a, n_docs)
    d500_docs = np.zeros((nS, n_docs, L, D), dtype=np.float32)
    cum_docs = np.zeros((nS, n_docs, L, D), dtype=np.float32)
    pos0 = np.zeros((nS, 3, L, D), dtype=np.float32)  # position-0 part of d500, d500f, cum (summed over documents)
    first_tok = {int(x) for x in fb["input_ids"][:, 0]} | {int(x) for t in steps for x in w_b[t]["input_ids"][:, 0]}
    after_F = {0: hF0}  # h_F under adapter s from step s's own forward (first load), kept while needed
    cached = {}  # {step: adapter tensors on CPU} of the last adapter read from disk
    coverage, ident, diag = [], [], []
    for i, t in enumerate(steps):
        p = t - WINDOW
        rec = dict(step=t)
        t0 = time.time()
        params = read_adapter(adir[t])
        coverage.append(dict(step=t, **apply_adapter(pm, params, names)))
        rec["load_t_s"] = round(time.time() - t0, 2)
        torch.cuda.synchronize()
        t0 = time.time()
        hW_t, pW_t, _ = sums(pm, w_b[t], layers)
        hF_t, pF_t, fh = sums(pm, fb, layers)
        torch.cuda.synchronize()
        rec["forward_t_s"] = round(time.time() - t0, 2)
        f_hashes.append(fh)
        t0 = time.time()
        if p == 0:
            zero_lora_b(pm)
            rec["before"] = "base (LoRA B zero)"
        elif p in cached:
            coverage.append(dict(step=p, **apply_adapter(pm, cached[p], names)))
            rec["before"] = f"adapter {p} (reloaded from memory)"
        else:
            coverage.append(dict(step=p, **apply_adapter(pm, read_adapter(adir[p]), names)))
            rec["before"] = f"adapter {p} (read from disk)"
        rec["load_p_s"] = round(time.time() - t0, 2)
        torch.cuda.synchronize()
        t0 = time.time()
        hW_p, pW_p, _ = sums(pm, w_b[t], layers)
        hF_p, pF_p, fh = sums(pm, fb, layers)
        torch.cuda.synchronize()
        rec["forward_p_s"] = round(time.time() - t0, 2)
        f_hashes.append(fh)
        cached = {t: params}
        dW = (hW_t - hW_p).transpose(0, 1).cpu().numpy()  # [B, L, D] fp64
        dF = (hF_t - hF_p).sum(1).cpu().numpy()  # [L, D]
        cF = (hF_t - hF0).transpose(0, 1).cpu().numpy()  # [B, L, D]
        if p in after_F:  # the reload of t-500 against its own first forward (d500f vs cum(t) - cum(t-500))
            ref_p = after_F[p]
            e_abs = float((hF_p - ref_p).abs().max())
            alt = (hF_t - ref_p).sum(1).cpu().numpy()
            e_rel = float(np.linalg.norm(dF - alt) / max(np.linalg.norm(alt), 1e-300))
            ident.append(dict(step=t, max_abs_reload_diff=e_abs, rel_err=e_rel))
        after_F = {0: hF0, t: hF_t}  # the base and the last step: the next step's t-500 is this t
        t0 = time.time()
        G = np.stack([gs.vectors(w_sel[t], nm).astype(np.float64).sum(0) for nm in gnames])  # [L, D], +dL/dh
        rec["grads_s"] = round(time.time() - t0, 2)
        X[i, 0] = dW.sum(0)
        X[i, 1] = dF
        X[i, 2] = cF.sum(0)
        X[i, 3] = -G
        lo_a[i] = [t - WINDOW + 1, t - WINDOW + 1, 1, min(w_rec[t]["global_step"])]
        hi_a[i] = [t, t, t, max(w_rec[t]["global_step"])]
        d500_docs[i] = dW
        cum_docs[i] = cF
        pos0[i, 0] = (pW_t - pW_p).sum(1).cpu().numpy()
        pos0[i, 1] = (pF_t - pF_p).sum(1).cpu().numpy()
        pos0[i, 2] = (pF_t - pF0).sum(1).cpu().numpy()
        nrm = np.linalg.norm(X[i], axis=-1)  # [4, L]
        share0 = np.linalg.norm(pos0[i], axis=-1) / np.maximum(nrm[:3], 1e-300)  # [3, L]
        Xr = X[i, :3].astype(np.float64) - pos0[i]  # the three changes without position 0
        dg = dict(step=t, median_norm={lab: float(np.median(nrm[k])) for k, lab in enumerate(LABELS)},
                  cos_d500_g500_median_layers=float(np.median([_cos(X[i, 0, li], X[i, 3, li]) for li in range(L)])),
                  cos_d500_d500f_median_layers=float(np.median([_cos(X[i, 0, li], X[i, 1, li]) for li in range(L)])),
                  cos_cum_d500f_median_layers=float(np.median([_cos(X[i, 2, li], X[i, 1, li]) for li in range(L)])),
                  pos0_norm_share_median_layers={lab: float(np.median(share0[k])) for k, lab in enumerate(LABELS[:3])},
                  cos_d500_d500f_without_pos0_median_layers=float(np.median([_cos(Xr[0, li], Xr[1, li])
                                                                             for li in range(L)])),
                  cos_d500_without_pos0_g500_median_layers=float(np.median([_cos(Xr[0, li], X[i, 3, li])
                                                                            for li in range(L)])))
        diag.append(dg)
        timing.append(rec)
        del hW_t, hW_p, hF_p, pW_t, pW_p, pF_p
        log(f"{run} step {t}: load {rec['load_t_s']}+{rec['load_p_s']} s, forwards {rec['forward_t_s']}+"
            f"{rec['forward_p_s']} s, grads {rec['grads_s']} s; median norms "
            + " ".join(f"{k} {v:.4g}" for k, v in dg["median_norm"].items())
            + f"; cos(d500, g500) {dg['cos_d500_g500_median_layers']:.3f}, cos(d500, d500f) "
            f"{dg['cos_d500_d500f_median_layers']:.3f}; position-0 share of the norm "
            + " ".join(f"{k} {v:.3f}" for k, v in dg["pos0_norm_share_median_layers"].items())
            + f"; without position 0 cos(d500, d500f) {dg['cos_d500_d500f_without_pos0_median_layers']:.3f}, "
            f"cos(d500, g500) {dg['cos_d500_without_pos0_g500_median_layers']:.3f}"
            + (f"; reload diff {ident[-1]['max_abs_reload_diff']:.3g}, d500f vs cum difference rel "
               f"{ident[-1]['rel_err']:.3g}" if ident and ident[-1]["step"] == t else ""))
    del after_F
    fb2, same2 = build_batch(tok, load_rows(st), f_sel, st)  # a fresh read and tokenisation of F at the end
    f_fresh = all(same2.values()) and all(np.array_equal(fb2[k], fb[k]) for k in fb)
    checks["f_docs_identical"] = dict(ok=len(set(f_hashes)) == 1 and f_fresh, forwards=len(f_hashes),
                                      sha1=f_hashes[0], fresh_retokenisation_equal=f_fresh, line_no=f_rec["line_no"])
    n_cov = sum(c["covered"] and c["model_lora"] == N_LORA for c in coverage)
    checks["adapter_coverage"] = dict(ok=n_cov == len(coverage) and len(names) == N_LORA,
                                      detail=f"{n_cov}/{len(coverage)} adapter loads cover all {len(names)} LoRA "
                                             f"parameters of the wrapper")
    max_rel = max((e["rel_err"] for e in ident), default=None)
    at1000 = [e for e in ident if e["step"] == 1000]
    checks["d500f_equals_cum_difference"] = dict(
        ok=bool(ident) and max_rel < REL_TOL, tol=REL_TOL, n_steps=len(ident), max_rel_err=max_rel,
        max_abs_reload_diff=max((e["max_abs_reload_diff"] for e in ident), default=None),
        at_step_1000=at1000[0] if at1000 else None)
    want_lo = [[t - WINDOW + 1, t - WINDOW + 1, 1] for t in steps]
    checks["steps_and_bounds"] = dict(
        ok=bool(np.all(np.diff(steps) > 0)) and lo_a[:, :3].tolist() == want_lo and
        hi_a[:, :3].tolist() == [[t] * 3 for t in steps] and bool((nd_a == n_docs).all()) and
        bool(((lo_a[:, 3] > np.asarray(steps) - WINDOW) & (hi_a[:, 3] <= np.asarray(steps))).all()))
    nrm = np.linalg.norm(X, axis=-1)
    checks["finite_nonzero"] = dict(ok=bool(np.isfinite(X).all()) and bool((nrm > 0).all()) and
                                    bool(np.isfinite(d500_docs).all()) and bool(np.isfinite(cum_docs).all()),
                                    min_norm=float(nrm.min()), max_norm=float(nrm.max()))
    checks["reference_sign"] = dict(ok=ref["ok"], statement=ref["statement"], cells=ref["cells"])
    oh = order_hash(steps, LABELS, layers)
    import e2_read

    beh = e2_read._behaviour_curves(rd)
    md = dict(run=os.path.abspath(rd), arm=st.get("arm"), seed=st.get("seed"), quantity="full", kind="grad",
              kind_note="kind 'grad' is the label the trained-reader loader requires; d500, d500f and cum are changes "
                        "of the residual stream (+dh), g500 is a gradient (-g)",
              sign=SIGN, layers=[int(x) for x in layers], windows=[[lab, DEFINITION[lab]] for lab in LABELS],
              every=WINDOW, steps=[int(t) for t in steps], order_hash=oh, behaviour=beh,
              transform=dict(variant=VARIANT, definition=DEFINITION, token_set=adamfx.TOKEN_SET,
                             window_steps=WINDOW, n_docs=n_docs),
              sampling=dict(seed=seed, rule="W_t: numpy default_rng([seed, t]).choice(4000 index rows of steps "
                                            "t-499..t, n_docs, replace=False), sorted in index order; F: the first "
                                            "n_docs index rows of steps 1-2"),
              documents=dict(F=f_rec, W={str(t): w_rec[t] for t in steps}),
              dtype="fp32 forward (bf16 checkpoint upcast, IEEE fp32 matmuls, sdpa, no autocast), fp64 token sums "
                    "and differences, stored as fp32",
              model=st["model"], adapters=os.path.join(rd, "adapters"), grads=os.path.join(rd, "grads"),
              reference_sign=ref, diagnostics=diag, timing=timing, model_load_s=model.load_s,
              first_token_ids=sorted(first_tok),
              pos0_note="the full token set includes position 0 (as the adamfx and acttrue inputs); the docs file "
                        "keeps the position-0 part of d500, d500f and cum (pos0_sums) so that inputs without "
                        "position 0 are inputs minus pos0_sums",
              written=time.strftime("%Y-%m-%d %H:%M:%S"), shape=list(X.shape), dtype_stored="float32")
    tmp_i, tmp_d = p_inp + ".tmp", p_doc + ".tmp"
    md["checks"] = checks
    with open(tmp_i, "wb") as f:
        np.savez(f, inputs=X, steps=np.asarray(steps, dtype=np.int64), windows=np.asarray(LABELS), window_lo=lo_a,
                 window_hi=hi_a, n_docs=nd_a, layers=np.asarray(layers, dtype=np.int64), meta=np.asarray(json.dumps(md)))
    dry = loader_dry_run(tmp_i)
    checks["loader_dry_run"] = dry
    n_pass = sum(bool(c["ok"]) for c in checks.values())
    md["checks"], md["checks_passed"] = checks, f"{n_pass}/{len(checks)}"
    with open(tmp_i, "wb") as f:  # rewrite with the dry-run result in the meta (same arrays, same order hash)
        np.savez(f, inputs=X, steps=np.asarray(steps, dtype=np.int64), windows=np.asarray(LABELS), window_lo=lo_a,
                 window_hi=hi_a, n_docs=nd_a, layers=np.asarray(layers, dtype=np.int64), meta=np.asarray(json.dumps(md)))
    dmeta = dict(run=md["run"], sign="d500_docs and cum_docs are +dh per document (the sign of the inputs file); "
                 "f_h0 = tokensum(h_L) of the base model on F", steps=md["steps"], layers=md["layers"],
                 order_hash=oh, inputs_file=p_inp, sampling=md["sampling"], dtype=md["dtype"])
    with open(tmp_d, "wb") as f:
        np.savez(f, d500_docs=d500_docs, cum_docs=cum_docs, f_h0=hF0.transpose(0, 1).float().cpu().numpy(),
                 pos0_sums=pos0, pos0_labels=np.asarray(LABELS[:3]),
                 steps=np.asarray(steps, dtype=np.int64), layers=np.asarray(layers, dtype=np.int64),
                 w_line_no=np.asarray([w_rec[t]["line_no"] for t in steps], dtype=np.int64),
                 w_global_step=np.asarray([w_rec[t]["global_step"] for t in steps], dtype=np.int64),
                 w_doc_id=np.asarray([w_rec[t]["doc_id"] for t in steps]),
                 w_n_tokens=np.asarray([w_rec[t]["n_tokens"] for t in steps], dtype=np.int64),
                 f_line_no=np.asarray(f_rec["line_no"], dtype=np.int64),
                 f_global_step=np.asarray(f_rec["global_step"], dtype=np.int64),
                 f_doc_id=np.asarray(f_rec["doc_id"]), f_n_tokens=np.asarray(f_rec["n_tokens"], dtype=np.int64),
                 meta=np.asarray(json.dumps(dmeta)))
    res = dict(run=run, checks_passed=md["checks_passed"], checks=checks, files=[p_inp, p_doc, p_json],
               shape=md["shape"], seconds=round(time.time() - t_run, 1), model_load_s=model.load_s,
               per_step_s=float(np.median([r["load_t_s"] + r["load_p_s"] + r["forward_t_s"] + r["forward_p_s"] +
                                           r["grads_s"] for r in timing])), diagnostics=diag, reference_sign=ref)
    if n_pass != len(checks):
        for tmp, final in ((tmp_i, p_inp), (tmp_d, p_doc)):
            os.replace(tmp, final + ".checks_failed")
        atomic_json(dict(documents=md["documents"], checks=checks), p_json + ".checks_failed")
        res["failed"] = [k for k, c in checks.items() if not c["ok"]]
        if marker_dir:
            with open(os.path.join(marker_dir, f"extract_{run}.FAILED"), "w") as f:
                f.write(f"{time.strftime('%F %T')} checks {md['checks_passed']}, failed {res['failed']}\n")
        log(f"{run}: integrity checks FAILED {res['failed']}; files kept as *.checks_failed")
        return res
    os.replace(tmp_i, p_inp)
    os.replace(tmp_d, p_doc)
    atomic_json(dict(run=md["run"], sampling=md["sampling"], F=f_rec, W={str(t): w_rec[t] for t in steps},
                     g500_window=[[int(lo_a[i, 3]), int(hi_a[i, 3])] for i in range(nS)]), p_json)
    if marker_dir:
        with open(os.path.join(marker_dir, f"extract_{run}.DONE"), "w") as f:
            f.write(f"{time.strftime('%F %T')} checks {md['checks_passed']}; {p_inp} {list(X.shape)}; "
                    f"{res['seconds']} s\n")
    log(f"{run}: wrote {p_inp} {list(X.shape)}, {p_doc}, {p_json}; checks passed {md['checks_passed']}; "
        f"{res['seconds']} s")
    return res


# ================================================================== fixed documents (new setting, input 5)
#
# Plan docs/notes/20261025_new_setting_plan.md section 2, inputs 5a and 5b (user decisions 2026-10-25): the same 128
# fixed clean-number documents (main_run/make_fixed_docs.py; fields row, order, half, prompt, completion) under every
# fp32 adapter of a run, fp32 base + fp32 LoRA (IEEE fp32 matmuls, core.set_fp32_matmul_ieee), no autocast.
#   compute  (``--docs D --run R``; ``--watch`` keeps polling for new adapters): for every adapter step t of
#            ``--steps`` (0:7000:500) present under <run>/adapters/step_<t>/ and not done yet, the token sums
#            h_d(t) of each decoder layer's output over the gradient store's full token set (positions 0..n-2,
#            adamfx_fp32_check.layer_sums; fp64 sums of the fp32 outputs), one forward per half (64 documents,
#            trainer tokenisation and right padding), written to <run>/ckptdiff/:
#              h_step_<t>.npy      fp64 [2, L, D]  the sum over the documents of each half (row = half)
#              hdoc_step_<t>.npy   fp32 [n, L, D]  per document, file order (for later analyses only)
#              h_step_<t>.json     adapter, its dtype and coverage, the documents' and ids' sha1, timing; written
#                                  last, so a step is done when its json exists
#            At step 0 the adapter (LoRA B zero) is also compared with the adapter switched off (the base model):
#            the max abs difference must be 0 and the cosine 1.0 (check h0_equals_base).
#   build    (``--build``): H(t) = h_half0(t) + h_half1(t) at every adapter step; at each read step t (the adapter
#            steps > 0): w1000 = H(t) - H(max(t - 1000, 0)), w3000 = H(t) - H(max(t - 3000, 0)), cum = H(t) - H(0)
#            (a window that is not yet full equals cum), fp64 differences stored as fp32 in
#            <run>/e2/ckptdiff_inputs_<suffix>.npz (export_inputs format, transform.variant ckptdiff, n_docs = the
#            number of fixed documents), and <run>/ckptdiff/split_half.json: the cosine of h_half0(t) - h_half0(0)
#            and h_half1(t) - h_half1(0) at every read step (whole 40 x 5120 vector, and per layer min / median),
#            with the plan's 0.95 rule at t = 1000, 3000, 7000 (both readings reported; the decision is the user's).
FIXED_SUBDIR = "ckptdiff"
FIXED_WINDOWS = "w1000,w3000,cum"
FIXED_STEPS = "0:7000:500"
SPLIT_STEPS = (1000, 3000, 7000)
SPLIT_RULE = 0.95
FIXED_SIGN = ("q = -dh per fixed document with dh = tokensum(h_L under the adapter of step t) - tokensum(h_L under the "
              "adapter of step t - w, or of step 0 = the base model for cum) of the decoder-layer-L output over the "
              "full token set, fp32 forwards of fp32 base + fp32 LoRA; the convention of the adamfx exports, whose q is "
              "a positive mix of per-token +dL/dh rows under plain SGD; inputs = -(sum over the fixed documents of q) "
              "= H(t) - H(t - w): the change the training made on the fixed documents (descent direction). Token sums "
              "fp64, differences fp64, stored as fp32")
FIXED_DEFINITION = {
    "w1000": "input 5b: H(t) - H(max(t - 1000, 0)); H = sum over the fixed documents of the layer output token sums",
    "w3000": "input 5b: H(t) - H(max(t - 3000, 0))",
    "cum": "input 5a: H(t) - H(0), H(0) = the base model (adapter of step 0, LoRA B zero)",
}


def fixed_definition(lab):
    """FIXED_DEFINITION's text, or for any other window label wN the same rule with N (window-size readings, plan
    docs/notes/20261026_window_size_plan.md: w100 ... w500)."""
    if lab in FIXED_DEFINITION:
        return FIXED_DEFINITION[lab]
    if lab.startswith("w") and lab[1:].isdigit():
        return f"H(t) - H(max(t - {int(lab[1:])}, 0)); H = sum over the fixed documents of the layer output token sums"
    return lab


def parse_step_range(spec):
    """'0:7000:500' -> [0, 500, ..., 7000]; '0,100,200' -> [0, 100, 200]."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi, ev = (int(x) for x in part.split(":"))
            out |= set(range(lo, hi + 1, ev))
        else:
            out.add(int(part))
    return sorted(out)


def resolve_run(run):
    return run if os.path.isabs(run) else os.path.join(RUNS_DIR, run)


def load_fixed_docs(path):
    """Rows of the fixed-documents file (make_fixed_docs.py) in file order, the halves and the file's sha256."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    bad = [i for i, r in enumerate(rows) if not isinstance(r.get("prompt"), str) or not isinstance(r.get("completion"), str)]
    if bad:
        raise SystemExit(f"{path}: rows {bad[:5]} lack the text fields prompt and completion")
    n = len(rows)
    half = [int(r["half"]) if "half" in r else (0 if i < n // 2 else 1) for i, r in enumerate(rows)]
    if sorted(set(half)) != [0, 1] or half.count(0) != half.count(1):
        raise SystemExit(f"{path}: halves {half.count(0)} / {half.count(1)}; two equal halves needed")
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return rows, half, sha


def fixed_batches(tok, rows, half, st):
    """One right-padded batch per half (trainer tokenisation: tokenize_rows + CompletionMaskCollator) and the sha1 of
    each batch's input_ids; returns [(half, indices, batch)], n_tokens per row (file order), sha1 over both batches."""
    import train_subliminal as ts

    out, ntok, h = [], [0] * len(rows), hashlib.sha1()
    for hv in (0, 1):
        sel = [i for i, x in enumerate(half) if x == hv]
        rr = [dict(prompt=rows[i]["prompt"], completion=rows[i]["completion"], source=None) for i in sel]
        ids, masks, _ = ts.tokenize_rows(tok, rr, st["max_len"], st["think_loss"])
        b = ts.CompletionMaskCollator(tok)([dict(input_ids=i_, completion_mask=m) for i_, m in zip(ids, masks)])
        batch = {k: b[k].numpy() for k in ("input_ids", "attention_mask")}
        for j, i in enumerate(sel):
            ntok[i] = int(batch["attention_mask"][j].sum())
        h.update(np.ascontiguousarray(batch["input_ids"]).tobytes())
        h.update(np.ascontiguousarray(batch["attention_mask"]).tobytes())
        out.append((hv, sel, batch))
    return out, ntok, h.hexdigest()


def adapter_dir_of(run, t):
    return os.path.join(run, "adapters", f"step_{t:06d}")


def adapter_ready(run, t):
    d = adapter_dir_of(run, t)
    return os.path.exists(os.path.join(d, "adapter_model.safetensors")) and os.path.exists(
        os.path.join(d, "adapter_config.json"))


def step_done(cd, t):
    return all(os.path.exists(os.path.join(cd, f"{p}_step_{t:06d}.{e}")) for p, e in (("h", "npy"), ("hdoc", "npy"),
                                                                                        ("h", "json")))


def _atomic_npy(path, arr):
    with open(path + ".tmp", "wb") as f:
        np.save(f, arr)
    os.replace(path + ".tmp", path)


def fixed_forward(pm, batches, n_docs, layers):
    """(per-half sums fp64 [2, L, D] on CPU, per-document fp32 [n, L, D], seconds) of one adapter state."""
    L = len(layers)
    hd = None
    halves = None
    t0 = time.time()
    for hv, sel, batch in batches:
        s = layer_sums(pm, batch["input_ids"], batch["attention_mask"], layers)  # [L, B, D] fp64 on the GPU
        if hd is None:
            D = s.shape[-1]
            hd = np.zeros((n_docs, L, D), dtype=np.float32)
            halves = np.zeros((2, L, D))
        halves[hv] = s.sum(1).cpu().numpy()
        hd[sel] = s.transpose(0, 1).float().cpu().numpy()
        del s
    if next(pm.parameters()).is_cuda:
        torch.cuda.synchronize()
    return halves, hd, time.time() - t0


def compute_fixed(run, docs_path, steps, device="cuda:0", watch=False, done_marker=None, failed_marker=None,
                  poll_s=30, max_idle_s=6 * 3600, grace_s=180, allow_bf16=False, log_fn=None):
    """Compute h(t) for every adapter step of ``steps`` that is present and not done (``watch``: keep polling until all
    are done; stop when ``failed_marker`` appears, or when ``done_marker`` exists and adapters are still missing after
    ``grace_s``, or after ``max_idle_s`` without a new adapter). Returns (exit code, summary dict): 0 all done, 2
    missing adapters after training ended, 3 training failed, 4 idle too long."""
    lg = log_fn or log
    run = resolve_run(run)
    cd = os.path.join(run, FIXED_SUBDIR)
    os.makedirs(cd, exist_ok=True)
    with open(os.path.join(run, "RUN.json")) as f:
        st = json.load(f)["settings"]
    rows, half, docs_sha = load_fixed_docs(docs_path)
    model, pm, tok, names, batches, layers = None, None, None, None, None, None
    ids_sha, ntok = None, None
    docs_json = os.path.join(cd, "docs.json")
    done_now, last_progress, t_done_seen = [], time.time(), None
    while True:
        todo = [t for t in steps if not step_done(cd, t)]
        if not todo:
            return 0, dict(done=done_now, all_done=True)
        ready = [t for t in todo if adapter_ready(run, t)]
        for t in ready:
            if model is None:
                model = Fp32Model(device)
                pm, tok = model.get(st["model"], adapter_dir_of(run, t))
                names = lora_names(pm)
                layers = list(range(len(core.decoder_layers(pm))))
                batches, ntok, ids_sha = fixed_batches(tok, rows, half, st)
                rec = dict(run=run, docs=os.path.abspath(docs_path), docs_sha256=docs_sha, ids_sha1=ids_sha,
                           n_docs=len(rows), rows=[int(r.get("row", -1)) for r in rows], half=half, n_tokens=ntok,
                           max_len=st["max_len"], think_loss=st["think_loss"], token_set=adamfx.TOKEN_SET,
                           batches="one right-padded batch per half (64 documents), file order within a half",
                           model=st["model"], layers=layers)
                if os.path.exists(docs_json):
                    with open(docs_json) as f:
                        old = json.load(f)
                    if old.get("docs_sha256") != docs_sha or old.get("ids_sha1") != ids_sha:
                        raise SystemExit(f"{docs_json} records other documents or tokens than {docs_path}")
                else:
                    atomic_json(rec, docs_json)
            t0 = time.time()
            params = read_adapter(os.path.join(adapter_dir_of(run, t), "adapter_model.safetensors"))
            dts = sorted({str(v.dtype) for v in params.values()})
            if dts != ["torch.float32"] and not allow_bf16:
                raise SystemExit(f"adapter step {t} has dtypes {dts}; fp32 adapters are required (--allow-bf16)")
            cover = apply_adapter(pm, params, names)
            load_s = time.time() - t0
            halves, hd, fw_s = fixed_forward(pm, batches, len(rows), layers)
            info = dict(step=t, adapter=adapter_dir_of(run, t), adapter_dtypes=dts, coverage=cover,
                        docs_sha256=docs_sha, ids_sha1=ids_sha, load_s=round(load_s, 2), forward_s=round(fw_s, 2),
                        written=time.strftime("%Y-%m-%d %H:%M:%S"), device=str(device),
                        fp32_matmul_precision=str(getattr(torch.backends.cuda.matmul, "fp32_precision", "n/a")),
                        allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32))
            if t == 0:
                lora_b_max = max(float(p.abs().max()) for n_, p in pm.named_parameters() if "lora_B" in n_)
                with pm.disable_adapter():
                    base_halves, base_hd, _ = fixed_forward(pm, batches, len(rows), layers)
                diff = float(np.abs(hd.astype(np.float64) - base_hd.astype(np.float64)).max())
                hdiff = float(np.abs(halves - base_halves).max())
                cos = [_cos(halves[:, li].sum(0), base_halves[:, li].sum(0)) for li in range(len(layers))]
                info["h0_equals_base"] = dict(ok=diff == 0.0 and hdiff == 0.0 and min(cos) >= 1.0 - 1e-12,
                                              max_abs_diff_docs=diff, max_abs_diff_halves=hdiff,
                                              min_cos_over_layers=float(min(cos)), lora_b_max_abs=lora_b_max)
                lg(f"step 0 vs base (adapter off): max abs diff {hdiff}, min cos {min(cos):.12f}, "
                   f"LoRA B max abs {lora_b_max}")
            _atomic_npy(os.path.join(cd, f"hdoc_step_{t:06d}.npy"), hd)
            _atomic_npy(os.path.join(cd, f"h_step_{t:06d}.npy"), halves)
            atomic_json(info, os.path.join(cd, f"h_step_{t:06d}.json"))
            done_now.append(t)
            last_progress = time.time()
            lg(f"{os.path.basename(run)} step {t}: load {load_s:.1f}s, forwards {fw_s:.1f}s; "
               f"|H| layer 0 {np.linalg.norm(halves[:, 0].sum(0)):.6g}, layer 39 {np.linalg.norm(halves[:, -1].sum(0)):.6g}")
        if ready:
            continue
        if not watch:
            return (0 if not [t for t in steps if not step_done(cd, t)] else 2), dict(done=done_now, missing=todo)
        if failed_marker and os.path.exists(failed_marker):
            return 3, dict(done=done_now, missing=todo, why=f"{failed_marker} exists")
        if done_marker and os.path.exists(done_marker):
            t_done_seen = t_done_seen or time.time()
            if time.time() - t_done_seen > grace_s:
                return 2, dict(done=done_now, missing=todo, why=f"{done_marker} exists, adapters missing {todo[:5]}")
        if time.time() - last_progress > max_idle_s:
            return 4, dict(done=done_now, missing=todo, why=f"no new adapter for {max_idle_s} s")
        time.sleep(poll_s)


def _aside(path):
    """Rename an existing ``path`` to <path>.before_<YYYYmmdd_HHMMSS>[_n] (the chain's convention); nothing is lost."""
    if not os.path.exists(path):
        return None
    base = f"{path}.before_{time.strftime('%Y%m%d_%H%M%S')}"
    dst, n = base, 0
    while os.path.exists(dst):
        n += 1
        dst = f"{base}_{n}"
    os.rename(path, dst)
    return dst


def build_fixed(run, docs_path, steps, windows_spec=FIXED_WINDOWS, suffix="every500", split_steps=SPLIT_STEPS,
                pair_run=None, loader=True, log_fn=None, read_steps=None, split_half=True):
    """Write <run>/e2/ckptdiff_inputs_<suffix>.npz and <run>/ckptdiff/split_half.json from the h(t) files.

    Both final names are written only when every applicable check passes; otherwise the files go to
    <name>.checks_failed (an earlier one moved aside first), so a later build after a fix is not blocked.
    ``steps``: the adapter steps whose h(t) is loaded (0 included). ``read_steps`` (default: every step of ``steps``
    above 0): the read steps of the file, a subset of ``steps`` (window-size readings read a window of length N only
    from t = N on). ``split_half`` False: no split-half file is computed, written or required to be absent (a second
    build of the same run with another suffix; the first build's split_half.json stays as it is)."""
    lg = log_fn or log
    run = resolve_run(run)
    cd = os.path.join(run, FIXED_SUBDIR)
    out = os.path.join(run, "e2", f"{VARIANT}_inputs_{suffix}.npz")
    sp = os.path.join(cd, "split_half.json")
    for p in ((out, sp) if split_half else (out,)):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    if 0 not in steps:
        raise SystemExit("step 0 (the base model) is needed for cum")
    miss = [t for t in steps if not step_done(cd, t)]
    if miss:
        raise SystemExit(f"h(t) missing at steps {miss}")
    rows, half, docs_sha = load_fixed_docs(docs_path)
    windows = []
    for x in windows_spec.split(","):
        x = x.strip()
        windows.append(("cum", None) if x == "cum" else (x if x.startswith("w") else f"w{x}",
                                                         int(x[1:] if x.startswith("w") else x)))
    labels = [lab for lab, _ in windows]
    if read_steps is None:
        read = [t for t in steps if t > 0]
    else:
        read = sorted({int(t) for t in read_steps})
        bad_read = [t for t in read if t <= 0 or t not in steps]
        if bad_read or not read:
            raise SystemExit(f"read steps {bad_read[:5] or read} must be adapter steps of --steps above 0")
    H = {t: np.load(os.path.join(cd, f"h_step_{t:06d}.npy")) for t in steps}  # fp64 [2, L, D]
    infos = {}
    for t in steps:
        with open(os.path.join(cd, f"h_step_{t:06d}.json")) as f:
            infos[t] = json.load(f)
    L, D = H[0].shape[1], H[0].shape[2]
    X = np.zeros((len(read), len(windows), L, D), dtype=np.float32)
    lo_a = np.zeros((len(read), len(windows)), dtype=np.int64)
    hi_a = np.zeros_like(lo_a)
    for i, t in enumerate(read):
        for wi, (lab, w) in enumerate(windows):
            base = 0 if w is None else max(t - w, 0)
            if base not in H:
                raise SystemExit(f"window {lab} at step {t} needs h({base}), which is not an adapter step")
            X[i, wi] = (H[t].sum(0) - H[base].sum(0)).astype(np.float32)
            lo_a[i, wi], hi_a[i, wi] = base + 1, t
    nd = np.full_like(lo_a, len(rows))
    layers = list(range(L))
    checks = {}
    h0 = infos[0].get("h0_equals_base")
    checks["h0_equals_base"] = dict({k: v for k, v in (h0 or {"why": "step 0 json has no base comparison"}).items()
                                     if k != "ok"}, ok=bool(h0 and h0.get("ok")))
    same_docs = {(v["docs_sha256"], v["ids_sha1"]) for v in infos.values()}
    checks["docs_identical_every_step"] = dict(ok=len(same_docs) == 1 and next(iter(same_docs))[0] == docs_sha,
                                               steps=len(infos), docs_sha256=docs_sha)
    checks["adapters_fp32"] = dict(ok=all(v["adapter_dtypes"] == ["torch.float32"] for v in infos.values()),
                                   dtypes=sorted({",".join(v["adapter_dtypes"]) for v in infos.values()}))
    checks["adapter_coverage"] = dict(ok=all(v["coverage"]["covered"] and v["coverage"]["model_lora"] == N_LORA
                                             for v in infos.values()), n=len(infos))
    nrm = np.linalg.norm(X, axis=-1)
    checks["finite_nonzero"] = dict(ok=bool(np.isfinite(X).all()) and bool((nrm > 0).all()),
                                    min_norm=float(nrm.min()), max_norm=float(nrm.max()))
    bad = [(t, lab) for i, t in enumerate(read) for wi, (lab, w) in enumerate(windows)
           if hi_a[i, wi] != t or lo_a[i, wi] != (1 if w is None else max(t - w, 0) + 1)]
    checks["window_bounds"] = dict(ok=not bad, bad=bad[:5])
    if pair_run:
        pj = os.path.join(resolve_run(pair_run), FIXED_SUBDIR, "docs.json")
        if os.path.exists(pj):
            with open(pj) as f:
                other = json.load(f)
            with open(os.path.join(cd, "docs.json")) as f:
                mine = json.load(f)
            checks["same_docs_as_pair_run"] = dict(ok=other["docs_sha256"] == mine["docs_sha256"]
                                                   and other["ids_sha1"] == mine["ids_sha1"], pair=pj,
                                                   docs_sha256=[mine["docs_sha256"], other["docs_sha256"]],
                                                   ids_sha1=[mine["ids_sha1"], other["ids_sha1"]])
        else:
            checks["same_docs_as_pair_run"] = dict(ok=True, applies=False, why=f"no {pj} yet")
    # split half: cosine of the two halves' h(t) - h(0)
    split = []
    for t in (read if split_half else []):
        a = H[t][0] - H[0][0]
        b = H[t][1] - H[0][1]
        per = [_cos(a[li], b[li]) for li in range(L)]
        row = dict(step=int(t), cos_flat=_cos(a, b), cos_layer_min=float(min(per)),
                   cos_layer_median=float(np.median(per)), worst_layer=int(np.argmin(per)))
        if t >= 1000 and (t - 1000) in H:
            a1, b1 = H[t][0] - H[t - 1000][0], H[t][1] - H[t - 1000][1]
            per1 = [_cos(a1[li], b1[li]) for li in range(L)]
            row["w1000_cos_flat"] = _cos(a1, b1)
            row["w1000_cos_layer_median"] = float(np.median(per1))
        split.append(row)
    at = {r["step"]: r for r in split}
    rule_steps = [t for t in split_steps if t in at]
    sh = dict(run=run, docs=os.path.abspath(docs_path), docs_sha256=docs_sha,
              halves=dict(rows_half0=[int(r.get("row", -1)) for r, h in zip(rows, half) if h == 0],
                          rows_half1=[int(r.get("row", -1)) for r, h in zip(rows, half) if h == 1]),
              definition="cosine between sum over half-0 documents of h_d(t) - h_d(0) and the same for half 1 "
                         "(cos_flat: the whole 40 x 5120 vector; cos_layer_min / median: per layer)",
              rule=f"plan: reduce to 64 documents later if the cosine is >= {SPLIT_RULE} at t = "
                   f"{', '.join(str(t) for t in split_steps)}",
              rule_steps=rule_steps,
              all_rule_steps_ge_flat=bool(rule_steps) and all(at[t]["cos_flat"] >= SPLIT_RULE for t in rule_steps),
              all_rule_steps_ge_every_layer=bool(rule_steps) and all(at[t]["cos_layer_min"] >= SPLIT_RULE
                                                                     for t in rule_steps),
              missing_rule_steps=[t for t in split_steps if t not in at], per_step=split,
              written=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not split_half:
        sh = dict(file=None, all_rule_steps_ge_flat=None, all_rule_steps_ge_every_layer=None,
                  why="split_half=False: not computed (an earlier build of this run wrote split_half.json)")
    oh = order_hash(read, labels, layers)
    import e2_read

    beh = e2_read._behaviour_curves(run)
    with open(os.path.join(run, "RUN.json")) as f:
        st = json.load(f)["settings"]
    md = dict(run=os.path.abspath(run), arm=st.get("arm"), seed=st.get("seed"), quantity="full", kind="grad",
              kind_note="kind 'grad' is the label the trained-reader loader requires; the inputs are changes of the "
                        "residual stream (+dh) on the fixed documents",
              sign=FIXED_SIGN, layers=layers, windows=[[lab, w] for lab, w in windows], every=None,
              steps=[int(t) for t in read], order_hash=oh, behaviour=beh,
              transform=dict(variant=VARIANT, definition={lab: fixed_definition(lab) for lab in labels},
                             token_set=adamfx.TOKEN_SET, n_docs=len(rows), docs=os.path.abspath(docs_path),
                             docs_sha256=docs_sha, adapter_steps=[int(t) for t in steps], h_dir=cd,
                             window_rule="wN at read step t: H(t) - H(max(t - N, 0)); cum: H(t) - H(0)",
                             plan="docs/notes/20261025_new_setting_plan.md"),
              split_half=dict(file=sp if split_half else None, all_rule_steps_ge_flat=sh["all_rule_steps_ge_flat"],
                              all_rule_steps_ge_every_layer=sh["all_rule_steps_ge_every_layer"]),
              dtype="fp32 forward (bf16 checkpoint upcast, fp32 adapters, IEEE fp32 matmuls, sdpa, no autocast), fp64 "
                    "token sums and differences, stored as fp32",
              written=time.strftime("%Y-%m-%d %H:%M:%S"), shape=list(X.shape))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.replace(tmp, tmp + ".stale_" + time.strftime("%Y%m%d_%H%M%S"))
    md["checks"] = checks

    def write():
        with open(tmp, "wb") as f:
            np.savez(f, inputs=X, steps=np.asarray(read, dtype=np.int64), windows=np.asarray(labels),
                     window_lo=lo_a, window_hi=hi_a, n_docs=nd, layers=np.asarray(layers, dtype=np.int64),
                     meta=np.asarray(json.dumps(md)))

    write()
    if loader and os.path.exists(os.path.join(WORKTREE, "main_run", "reader", "e2_generate.py")):
        dry = loader_dry_run(tmp)
        dry["ok"] = bool(dry.get("ok")) and dry.get("windows") == labels and dry.get("steps") == len(read)
        checks["loader_dry_run"] = dry
    else:
        checks["loader_dry_run"] = dict(ok=True, applies=False, why="--no-loader or worktree absent")
    app = {k: c for k, c in checks.items() if c.get("applies", True)}
    n_pass = sum(bool(c["ok"]) for c in app.values())
    md["checks"], md["checks_passed"] = checks, f"{n_pass}/{len(app)}"
    write()
    if n_pass != len(app):
        for p in ((out + ".checks_failed", sp + ".checks_failed") if split_half else (out + ".checks_failed",)):
            _aside(p)
        os.replace(tmp, out + ".checks_failed")
        if split_half:
            atomic_json(sh, sp + ".checks_failed")
        failed = [k for k, c in app.items() if not c["ok"]]
        lg(f"ckptdiff build FAILED {failed}; files kept as {out}.checks_failed"
           + (f" and {sp}.checks_failed" if split_half else ""))
        return 1, dict(failed=failed, checks=checks, split_half=sp + ".checks_failed" if split_half else None)
    for p in ((out, sp) if split_half else (out,)):
        if os.path.exists(p):
            raise SystemExit(f"{p} appeared while building; left {tmp}")
    if split_half:
        atomic_json(sh, sp)
    os.replace(tmp, out)
    lg(f"wrote {out} {list(X.shape)}; checks {md['checks_passed']}; split half at {rule_steps}: "
       + ", ".join(f"{t}: flat {at[t]['cos_flat']:.4f} layer min {at[t]['cos_layer_min']:.4f}" for t in rule_steps))
    return 0, dict(out=out, checks_passed=md["checks_passed"], split_half=sp if split_half else None,
                   split_rule=dict(flat=sh["all_rule_steps_ge_flat"], every_layer=sh["all_rule_steps_ge_every_layer"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--steps", default="all", help="'all' (500..20000) or a comma list of adapter steps (0 ignored)")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None, help="write to <out-dir>/<run>/ instead of <run>/e2/ (smoke)")
    ap.add_argument("--marker-dir", default=None, help="write extract_<run>.DONE / .FAILED here")
    ap.add_argument("--summary", default=None, help="json with every run's checks, timing and diagnostics")
    g = ap.add_argument_group("fixed documents (new setting)")
    g.add_argument("--docs", default=None, help="fixed-documents jsonl (make_fixed_docs.py): switches to this mode")
    g.add_argument("--run", default=None, help="run directory (or a name under agm_data/runs)")
    g.add_argument("--build", action="store_true", help="write the inputs file and split_half.json from h(t)")
    g.add_argument("--watch", action="store_true", help="poll for adapters until every step of --steps is done")
    g.add_argument("--train-done", default=None, help="--watch: training DONE marker")
    g.add_argument("--train-failed", default=None, help="--watch: training FAILED marker")
    g.add_argument("--poll", type=int, default=30)
    g.add_argument("--max-idle-s", type=int, default=6 * 3600)
    g.add_argument("--windows", default=FIXED_WINDOWS)
    g.add_argument("--suffix", default="every500")
    g.add_argument("--split-steps", default=",".join(str(t) for t in SPLIT_STEPS))
    g.add_argument("--pair-run", default=None, help="--build: the other student, whose docs.json must match")
    g.add_argument("--read-steps", default=None, help="--build: read steps of the file (default every --steps above 0)")
    g.add_argument("--no-split-half", action="store_true", help="--build: no split_half.json (a second build)")
    g.add_argument("--allow-bf16", action="store_true", help="accept bf16 adapters (tests only)")
    g.add_argument("--no-loader", action="store_true")
    a = ap.parse_args(argv)
    if a.docs:
        if not a.run:
            raise SystemExit("--docs needs --run")
        steps = parse_step_range(FIXED_STEPS if a.steps in (None, "", "all") else a.steps)
        if a.build:
            rc, res = build_fixed(a.run, a.docs, steps, a.windows, a.suffix,
                                  tuple(int(x) for x in a.split_steps.split(",") if x.strip()), a.pair_run,
                                  loader=not a.no_loader,
                                  read_steps=parse_step_range(a.read_steps) if a.read_steps else None,
                                  split_half=not a.no_split_half)
        else:
            rc, res = compute_fixed(a.run, a.docs, steps, a.device, a.watch, a.train_done, a.train_failed, a.poll,
                                    a.max_idle_s, allow_bf16=a.allow_bf16)
        print(json.dumps(res, default=str)[:4000])
        return rc
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    steps = parse_steps(a.steps)
    if a.marker_dir:
        os.makedirs(a.marker_dir, exist_ok=True)
    t0 = time.time()
    ref = reference_sign()
    log(f"reference sign: {ref['statement']} ({ref['seconds']} s); cells "
        + ", ".join(f"{c['step']}: cos {c['cos_inputs_vs_plus_sum_docs']:.6f} rel {c['rel_err_plus']:.2g}"
                    for c in ref["cells"]))
    model = Fp32Model(a.device)
    results, failed = [], []
    for run in runs:
        od = os.path.join(a.out_dir, run) if a.out_dir else None
        try:
            r = extract_run(run, model, steps, a.n_docs, a.seed, od, ref, a.marker_dir)
        except Exception as e:  # noqa: BLE001 - one run's failure is recorded and the next run goes on
            import traceback

            traceback.print_exc()
            r = dict(run=run, failed=[f"{type(e).__name__}: {e}"])
            if a.marker_dir:
                with open(os.path.join(a.marker_dir, f"extract_{run}.FAILED"), "w") as f:
                    f.write(f"{time.strftime('%F %T')} {type(e).__name__}: {e}\n")
        results.append(r)
        if r.get("failed"):
            failed.append(run)
        if a.summary:
            atomic_json(dict(runs=results, steps=steps, n_docs=a.n_docs, seed=a.seed,
                             seconds=round(time.time() - t0, 1)), a.summary)
    log(f"done: {len(runs) - len(failed)}/{len(runs)} runs passed; {time.time() - t0:.0f} s"
        + (f"; failed {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

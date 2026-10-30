#!/usr/bin/env python
"""How precise is the one-step change of the layer outputs taken as the difference of two forwards?

Question: is fp32 enough to get the change of the activations that ONE real optimizer step makes
by subtracting two forwards, or is fp64 needed; what does a forward cost in each precision, and how accurate is it.
The summed one-step changes are also saved as vectors that a
reader can take as the input of one batch (``_step_delta_vectors.npz``).

Inputs
  pairs   LoRA pairs of ``train_subliminal.py --adamfx-lora-pairs`` (fp32 before.safetensors / after.safetensors
          around one real Adam step), given as ``--pair-step RUN:STEP`` (RUN a run name under agm_data/runs or a
          path; the pair is RUN/lora_pairs/step_<STEP>) or ``--pair DIR``, both repeatable; default
          newset_dolphin_s1:1001 (its before equals the saved step-1000 adapter). All pairs must share the model,
          the LoRA config and the tokenisation settings; they run inside one model load, the pairs looping inside
          each precision. Names are mapped with layer_signal_ckpt.map_adapter and loaded with
          ckpt_diff_inputs.apply_adapter (raises unless every LoRA parameter of the wrapper is covered).
  docs    the 128 fixed documents (make_fixed_docs.load_fixed_docs validates the file), tokenised and batched as the
          fixed-document path of ckpt_diff_inputs.py (fixed_batches: trainer tokenisation, one right-padded batch of
          64 per half, file order within a half). ``--batch-rows`` < 64 (or an OOM, which halves it) feeds row slices
          of the same padded batches, so the per-row inputs never change.
  sums    per document and decoder layer, the token sum of the layer output over positions 0..n-2 (the gradient
          store's full token set, adamfx.token_weights; as adamfx_fp32_check.layer_sums and ckpt_diff_inputs), fp64
          reduction inside the hook with autocast off; the position-0 output is kept too, so positions 1..n-2 =
          full - pos0. The LM head is not run (removed after loading; only the decoder is called).
Variants, run in this order with one model load (the base checkpoint is bf16, so every cast is exact; checked: every
base weight is bf16-representable)
  fp32       base fp32 (bf16 upcast at load), LoRA fp32, IEEE fp32 matmuls (no TF32), no autocast: ckpt_diff_inputs
  fp32_tf32  the same weights with TF32 matmuls (torch.backends.cuda.matmul.fp32_precision = "tf32")
  bf16       base cast to bf16, LoRA fp32, forward under torch.autocast(bfloat16), IEEE fp32 matmuls (the torch
             default): the trainer's forward (bf16 weights, PEFT keeps LoRA fp32, accelerate's autocast), i.e. the
             in-training bf16 true change (adamfx acttrue)
  bf16_tf32  the same with TF32 allowed (inside autocast only the fp32 ops outside autocast, e.g. the rotary angles,
             are affected); the fast baseline
  fp64       every parameter and buffer cast to fp64, no autocast; the reference. Qwen3RMSNorm computes in fp32
             whatever its input dtype (transformers 5.14: ``hidden_states.to(torch.float32)``), which would round
             every norm output of an fp64 model to fp32; its forward is replaced per instance by one that computes in
             fp64 when the input is fp64 and is bit-identical otherwise (checked here on the loaded model and in
             tests/test_step_delta_precision.py). The rotary cos/sin stay as the library computes them (fp32 angles,
             cast to the model dtype): a fixed function of the position, the same in every variant and in both
             forwards of a pair, so it adds no difference noise.
Per variant: an untimed warm-up forward (first pair's before), then per pair: load before, a timed forward (the first
pair's is compared with the warm-up: determinism), load after, a timed forward. Forward time = wall time of the
forward over all 128 documents with synchronisation (the sums stay on the GPU until the timer stops); tokens/s over
the non-padding tokens fed and over the padded batch; peak memory = torch.cuda.max_memory_allocated over the
variant's forwards; weights memory before them.

Measures per pair, reference fp64, per layer and as band medians (shallow 0-12, middle 13-26, deep 27-39, all), for
two token sets: full (positions 0..n-2, the shipped convention) and nopos0 (positions 1..n-2). D = H_after - H_before.
  step_rel     |D_64| / |H_before,64| summed over the documents: how big one step's change is next to the activation
  fwd_relerr   |H_before,v - H_before,64| / |H_before,64|: the rounding error of one forward in variant v
  cos          cos(D_v, D_64)
  relerr       |D_v - D_64| / |D_64|
  doc_*        the same cos and relerr per document, median over the documents (and doc_step_rel for the reference)
Integrity checks (counted; exit 1 when one fails, every file is written either way). Shared: docs file consistent,
LoRA scaling = the pairs' meta, bf16-representable base, RMSNorm patch identical for fp32 / bf16 and active for fp64,
parameter dtypes per variant, adapter coverage, finite sums. Per pair (key ``<tag>:<name>``, tag = <run>_step<NNNNNN>):
the parameter step is non-zero, before = the run's saved adapter of the previous step, tokenisation = the run's
ckpt_diff docs.json (ids sha1), and the fp32 H_before against the run's ckpt_diff h_step_<before>.npy (same adapter,
documents and setting; relative difference <= 1e-6 per layer and half).

Outputs (P = ``--out`` without ``_step_delta_precision.json``; default --out
$IIF_BASE/agm_data/fixed_frame/checks/<stamp>_step_delta_precision.json):
  --out                         the measures, timings and checks of every pair (JSON)
  P_<tag>_step_delta_vectors.npz  per pair, float32 [L, D]: delta_fp64, delta_fp32, delta_bf16 = sum over the documents
                                of tokensum(h_L | after) - tokensum(h_L | before) (positions 0..n-2, +(after - before),
                                the ckpt_diff sign), h_before_fp64 (the fp64 sums under before); layers; meta (JSON)
  P_<tag>_step_delta_sums.npz   per pair: Hb_/Ha_/Hb0_/Ha0_<variant> fp64 [L, D] (sums over the documents of the full
                                token set and of position 0) and, unless ``--no-docs-npz``, per-document D_docs_fp64
                                (fp64), D_docs_fp32 and D_docs_bf16 (fp32) [n, L, D]

Cost estimate (one B200): the fp32 load takes 35-40 s and one fp32 forward of the 128 documents 9.2 s (ckpt_diff
logs); fp64 weights are 104 GiB without the LM head, activations of 64 rows x <= 256 tokens a few GiB more, so one
GPU with >= 150 GiB free; one pair about 5-10 minutes, each further pair adds two forwards per variant (1-4 minutes).

  CUDA_VISIBLE_DEVICES=<gpu> python main_run/measure_step_delta_precision.py --device cuda:0 [--out <json>]
      [--pair-step newset_dolphin_s1:1001 --pair-step newset_dolphin_s1:3001 ...] [--pair <lora pair dir>]
      [--docs <fixed docs jsonl>] [--variants fp32,fp32_tf32,bf16,bf16_tf32,fp64] [--batch-rows 64] [--no-docs-npz]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agm import core  # noqa: E402
import adamfx  # noqa: E402
import ckpt_diff_inputs as cdi  # noqa: E402
import make_fixed_docs as mfd  # noqa: E402
from adamfx_fp32_check import build_model  # noqa: E402
from layer_signal_ckpt import lora_names, map_adapter  # noqa: E402

IIF = os.environ["IIF_BASE"]
RUNS_DIR = os.path.join(IIF, "agm_data", "runs")
DEFAULT_PAIR_STEP = "newset_dolphin_s1:1001"
DEFAULT_DOCS = os.path.join(IIF, "agm_data", "newset", "fixed_docs_128.jsonl")
CHECKS_DIR = os.path.join(IIF, "agm_data", "fixed_frame", "checks")
OUT_SUFFIX = "_step_delta_precision.json"
VARIANTS = ("fp32", "fp32_tf32", "bf16", "bf16_tf32", "fp64")
SPEC = {
    "fp32": dict(weights="fp32", matmul="ieee", autocast=None),
    "fp32_tf32": dict(weights="fp32", matmul="tf32", autocast=None),
    "bf16": dict(weights="bf16", matmul="ieee", autocast=torch.bfloat16),
    "bf16_tf32": dict(weights="bf16", matmul="tf32", autocast=torch.bfloat16),
    "fp64": dict(weights="fp64", matmul="ieee", autocast=None),
}
EXPECTED_DTYPES = {"fp32": (torch.float32, torch.float32), "bf16": (torch.bfloat16, torch.float32),
                   "fp64": (torch.float64, torch.float64)}  # (base, LoRA)
REF = "fp64"
VECTOR_VARIANTS = ("fp64", "fp32", "bf16")
TOKEN_SETS = {"full": "positions 0..n-2 (the gradient store's full token set; ckpt_diff_inputs)",
              "nopos0": "positions 1..n-2 (full minus the position-0 output)"}
SIGN = ("+(after - before): sum over the fixed documents of tokensum(h_L | LoRA after the step) - tokensum(h_L | LoRA "
        "before the step), h_L = decoder-layer-L output, the change the step made (ckpt_diff_inputs convention)")
BANDS = (("shallow", 0, 12), ("middle", 13, 26), ("deep", 27, 39))
CKPTDIFF_TOL = 1e-6
RMS_FP64_TOL = 1e-14


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def sig(x, n=6):
    x = float(x)
    return float(f"{x:.{n}g}") if np.isfinite(x) else None


# ------------------------------------------------------------------ precision helpers

def set_matmul(mode):
    """fp32 matmul precision 'ieee' (no TF32; core.set_fp32_matmul_ieee) or 'tf32'; returns the flags as set."""
    try:
        torch.backends.cuda.matmul.fp32_precision = mode
    except (AttributeError, RuntimeError):
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
    try:  # torch 2.13 refuses to read the legacy flag once the new API has set TF32
        legacy = bool(torch.backends.cuda.matmul.allow_tf32)
    except RuntimeError:
        legacy = None
    return dict(fp32_precision=str(getattr(torch.backends.cuda.matmul, "fp32_precision", "n/a")), allow_tf32=legacy)


def _rms_forward(self, hidden_states):
    """Qwen3RMSNorm.forward computing in fp64 when the input is fp64 and in fp32 otherwise (the library's code)."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float64 if input_dtype == torch.float64 else torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def rms_modules(model):
    return [m for m in model.modules() if type(m).__name__ == "Qwen3RMSNorm"]


def patch_rmsnorm(model):
    mods = rms_modules(model)
    for m in mods:
        m.forward = types.MethodType(_rms_forward, m)
    return len(mods)


def check_rms_patch(model, device):
    """Patch every Qwen3RMSNorm and check on the first one: fp32 and bf16 inputs give bit-identical outputs before and
    after the patch, and an fp64 input is computed in fp64 (against the formula in fp64)."""
    mods = rms_modules(model)
    if not mods:
        raise RuntimeError("no Qwen3RMSNorm module found")
    m = mods[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (3 * torch.randn(2, 5, m.weight.shape[0], generator=g)).to(device)
    with torch.no_grad():
        o32, o16, o64 = m(x), m(x.bfloat16()), m(x.double())
        n = patch_rmsnorm(model)
        p32, p16, p64 = m(x), m(x.bfloat16()), m(x.double())
        x64 = x.double()
        exact = m.weight.double() * (x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + m.variance_epsilon))
    err = lambda a: float((a.double() - exact).norm() / exact.norm())  # noqa: E731
    return n, dict(identical_fp32=bool(torch.equal(o32, p32)), identical_bf16=bool(torch.equal(o16, p16)),
                   fp64_relerr_patched=err(p64), fp64_relerr_library=err(o64), fp64_dtype=str(p64.dtype))


def bf16_representable(pm):
    """(number of base parameters with a value that is not exactly bf16-representable, number of base parameters)."""
    bad = total = 0
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_" in n:
                continue
            total += 1
            if not torch.equal(p.data, p.data.to(torch.bfloat16).to(p.dtype)):
                bad += 1
    return bad, total


def cast_base(pm, dtype):
    """Cast every non-LoRA parameter to ``dtype`` in place; LoRA parameters and buffers stay as they are."""
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_" not in n and p.dtype != dtype:
                p.data = p.data.to(dtype)


def param_dtypes(pm):
    base, lora = set(), set()
    for n, p in pm.named_parameters():
        (lora if "lora_" in n else base).add(p.dtype)
    return base, lora


# ------------------------------------------------------------------ forward

def layer_sums_prec(pm, input_ids, attention_mask, layers, autocast_dtype=None):
    """adamfx_fp32_check.layer_sums with an optional autocast dtype, also returning the position-0 outputs:
    ([L, B, D] fp64 sums over positions 0..n-2, [L, B, D] fp64 position-0 outputs) on the model's device. The
    reductions run in fp64 with autocast off inside the hooks; autocast_dtype None = no autocast (as layer_sums)."""
    dev = next(pm.parameters()).device
    am = torch.as_tensor(attention_mask, device=dev)
    W = adamfx.token_weights(am.sum(1).tolist(), am.shape[1], dev).double()
    dec = core.decoder_layers(pm)
    S, P0, hs = {}, {}, []
    for li, L in enumerate(layers):
        def hook(mod, inp, out, li=li):
            h = out[0] if isinstance(out, tuple) else out
            with torch.autocast(device_type=dev.type, enabled=False):
                hd = h.double()
                S[li] = torch.bmm(W, hd)[:, 0]
                P0[li] = hd[:, 0].clone()
        hs.append(dec[L].register_forward_hook(hook))
    base = pm.get_base_model() if hasattr(pm, "get_base_model") else pm
    ctx = (torch.autocast(device_type=dev.type, dtype=autocast_dtype) if autocast_dtype is not None
           else torch.autocast(device_type=dev.type, enabled=False))
    try:
        with torch.no_grad(), ctx:
            base.model(input_ids=torch.as_tensor(input_ids, device=dev), attention_mask=am, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([S[li] for li in range(len(layers))]), torch.stack([P0[li] for li in range(len(layers))])


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def forward_all(pm, batches, n_docs, layers, autocast_dtype, bs):
    """Per-document sums of every batch in row slices of ``bs``: (full [n, L, D] fp64, pos0 [n, L, D] fp64, both on
    the CPU in file order, seconds of forward wall time with synchronisation; CPU copies are outside the timer)."""
    dev = next(pm.parameters()).device
    full = pos0 = None
    secs = 0.0
    for _, sel, batch in batches:
        ids, am = batch["input_ids"], batch["attention_mask"]
        for a in range(0, len(sel), bs):
            _sync(dev)
            t0 = time.perf_counter()
            s, p = layer_sums_prec(pm, ids[a:a + bs], am[a:a + bs], layers, autocast_dtype)
            _sync(dev)
            secs += time.perf_counter() - t0
            if full is None:
                full = np.zeros((n_docs, s.shape[0], s.shape[-1]))
                pos0 = np.zeros_like(full)
            idx = sel[a:a + bs]
            full[idx] = s.transpose(0, 1).cpu().numpy()
            pos0[idx] = p.transpose(0, 1).cpu().numpy()
            del s, p
    return full, pos0, secs



def run_variant(pm, names, pairs, batches, n_docs, layers, autocast_dtype, bs):
    """Warm-up (first pair's before), then per pair a timed before and a timed after forward. Returns per-pair arrays
    and timings, the warm-up time, memory, coverage and the determinism of the first pair's before forward."""
    dev = next(pm.parameters()).device
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    mem0 = torch.cuda.memory_allocated(dev) if cuda else None
    cdi.apply_adapter(pm, pairs[0]["before"], names)
    w_full, _, t_w = forward_all(pm, batches, n_docs, layers, autocast_dtype, bs)
    per, det = [], None
    for i, p in enumerate(pairs):
        cov_b = cdi.apply_adapter(pm, p["before"], names)
        b_full, b_p0, t_b = forward_all(pm, batches, n_docs, layers, autocast_dtype, bs)
        if i == 0:
            scale = float(np.abs(b_full).max()) or 1.0
            det = dict(bitwise_equal=bool(np.array_equal(w_full, b_full)),
                       max_abs_diff_over_max_abs=float(np.abs(w_full - b_full).max()) / scale)
            del w_full
        cov_a = cdi.apply_adapter(pm, p["after"], names)
        a_full, a_p0, t_a = forward_all(pm, batches, n_docs, layers, autocast_dtype, bs)
        per.append(dict(b_full=b_full, b_p0=b_p0, a_full=a_full, a_p0=a_p0, t_before=t_b, t_after=t_a,
                        cover=dict(before=cov_b, after=cov_a)))
    peak = torch.cuda.max_memory_allocated(dev) if cuda else None
    return dict(per=per, t_warm=t_w, peak=peak, mem0=mem0, det=det)


# ------------------------------------------------------------------ inputs

def pair_dir_of(spec):
    """'RUN:STEP' (RUN a name under agm_data/runs or a path) -> RUN/lora_pairs/step_<STEP:06d>."""
    run, _, step = spec.rpartition(":")
    if not run or not step.isdigit():
        raise SystemExit(f"--pair-step {spec!r}: expected RUN:STEP")
    run = run if os.path.isabs(run) else os.path.join(RUNS_DIR, run)
    return os.path.join(run, "lora_pairs", f"step_{int(step):06d}")


def read_pair(d):
    """meta, before, after (wrapper names -> fp32 CPU tensors) of a fp32 LoRA pair directory."""
    from safetensors.torch import load_file

    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("format") != "fp32":
        raise SystemExit(f"{d}: format {meta.get('format')}; an fp32 pair (before and after) is required")
    before = {k: v.float() for k, v in map_adapter(load_file(os.path.join(d, "before.safetensors"))).items()}
    after = {k: v.float() for k, v in map_adapter(load_file(os.path.join(d, "after.safetensors"))).items()}
    if set(before) != set(after):
        raise SystemExit(f"{d}: before and after have different parameter names")
    return meta, before, after


def param_step(before, after):
    num = sum(float(((after[k].double() - before[k].double()) ** 2).sum()) for k in before)
    den = sum(float((before[k].double() ** 2).sum()) for k in before)
    numb = sum(float(((after[k].double() - before[k].double()) ** 2).sum()) for k in before if "lora_B" in k)
    denb = sum(float((before[k].double() ** 2).sum()) for k in before if "lora_B" in k)
    return dict(all=(num / den) ** 0.5 if den else float("nan"), lora_B=(numb / denb) ** 0.5 if denb else float("nan"),
                abs_all=num ** 0.5)


def load_pairs(dirs):
    """Pair records (tag, dir, run, settings, meta, before, after, parameter step, pair checks) in the given order."""
    from safetensors.torch import load_file

    pairs = []
    for d in dirs:
        meta, before, after = read_pair(d)
        run = meta["run"]
        with open(os.path.join(run, "RUN.json")) as f:
            st = json.load(f)["settings"]
        step = int(meta["step"])
        tag = f"{os.path.basename(os.path.normpath(run))}_step{step:06d}"
        pstep = param_step(before, after)
        ck, sk = {"pair_step_nonzero": bool(np.isfinite(pstep["all"]) and pstep["abs_all"] > 0)}, {}
        saved = os.path.join(run, "adapters", f"step_{step - 1:06d}", "adapter_model.safetensors")
        if os.path.exists(saved):
            sv = map_adapter(load_file(saved))
            ck["before_equals_saved_adapter"] = bool(set(sv) == set(before) and all(
                torch.equal(sv[k].float(), before[k]) for k in before))
        else:
            sk["before_equals_saved_adapter"] = f"no {saved}"
        pairs.append(dict(tag=tag, dir=os.path.abspath(d), run=run, st=st, meta=meta, step=step, before_step=step - 1,
                          before=before, after=after, pstep=pstep, checks=ck, skipped=sk))
    tags = [p["tag"] for p in pairs]
    if len(set(tags)) != len(tags):
        raise SystemExit(f"duplicate pairs {tags}")
    keys = ("model", "max_len", "think_loss")
    for p in pairs[1:]:
        if any(p["st"].get(k) != pairs[0]["st"].get(k) for k in keys):
            raise SystemExit(f"{p['tag']}: settings {[p['st'].get(k) for k in keys]} differ from {pairs[0]['tag']}")
        if json.dumps(p["meta"]["lora_config"], sort_keys=True) != json.dumps(pairs[0]["meta"]["lora_config"],
                                                                              sort_keys=True):
            raise SystemExit(f"{p['tag']}: LoRA config differs from {pairs[0]['tag']}")
    return pairs


# ------------------------------------------------------------------ metrics

def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _cos_rows(a, b):
    return _div((a * b).sum(-1), np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def bands(vals):
    vals = np.asarray(vals, np.float64)
    L = len(vals)
    out = {}
    for name, lo, hi in BANDS + (("all", 0, L - 1),):
        v = vals[lo:min(hi, L - 1) + 1]
        v = v[np.isfinite(v)]
        out[name] = sig(np.median(v)) if v.size else None
    return out


def pack(vals):
    return dict(by_layer=[sig(v) for v in np.asarray(vals, np.float64)], bands=bands(vals))


def summarise(R, variants):
    """Per token set: the reference's step size and, per variant, forward error, cos and relative error of D."""
    out = {}
    for ts in TOKEN_SETS:
        def hb(v):
            return R[v]["Hb"] - (R[v]["Hb0"] if ts == "nopos0" else 0)

        def dsum(v):
            return (R[v]["Ha"] - R[v]["Hb"]) - ((R[v]["Ha0"] - R[v]["Hb0"]) if ts == "nopos0" else 0)

        def ddoc(v):
            return R[v]["dd"] - (R[v]["dd0"] if ts == "nopos0" else 0)

        entry = dict(definition=TOKEN_SETS[ts])
        if REF in R:
            Dr, Hr, ddr = dsum(REF), hb(REF), ddoc(REF)
            hdn = R[REF]["hb_doc_norm"] if ts == "full" else R[REF]["hb_doc_norm_nopos0"]
            entry["step_rel"] = pack(_div(np.linalg.norm(Dr, axis=-1), np.linalg.norm(Hr, axis=-1)))
            entry["step_rel_whole"] = sig(np.linalg.norm(Dr) / np.linalg.norm(Hr))
            entry["doc_step_rel_median"] = pack(np.nanmedian(_div(np.linalg.norm(ddr, axis=-1), hdn), 0))
            entry["D_norm"] = [sig(x) for x in np.linalg.norm(Dr, axis=-1)]
            entry["variants"] = {}
            for v in variants:
                if v == REF or v not in R:
                    continue
                Dv, Hv, ddv = dsum(v), hb(v), ddoc(v)
                rel = _div(np.linalg.norm(Dv - Dr, axis=-1), np.linalg.norm(Dr, axis=-1))
                doc_rel = np.nanmedian(_div(np.linalg.norm(ddv - ddr, axis=-1), np.linalg.norm(ddr, axis=-1)), 0)
                doc_cos = np.nanmedian(_cos_rows(ddv, ddr), 0)
                entry["variants"][v] = dict(
                    fwd_relerr=pack(_div(np.linalg.norm(Hv - Hr, axis=-1), np.linalg.norm(Hr, axis=-1))),
                    cos=pack(_cos_rows(Dv, Dr)), relerr=pack(rel), doc_cos_median=pack(doc_cos),
                    doc_relerr_median=pack(doc_rel),
                    whole=dict(cos=sig(_cos_rows(Dv.ravel()[None], Dr.ravel()[None])[0]),
                               relerr=sig(np.linalg.norm(Dv - Dr) / np.linalg.norm(Dr)) if np.linalg.norm(Dr) > 0
                               else None,
                               fwd_relerr=sig(np.linalg.norm(Hv - Hr) / np.linalg.norm(Hr))))
        out[ts] = entry
    return out




def fmt(v, w=9):
    return f"{v:{w}.2e}" if v is not None else f"{'-':>{w}}"


def fmtc(v, w=9):
    return f"{v:{w}.6f}" if v is not None else f"{'-':>{w}}"


def print_tables(res):
    variants = [v for v in res["variants_run"] if v != REF]
    bn = ("shallow", "middle", "deep", "all")
    for pr in res["pairs"]:
        for ts in TOKEN_SETS:
            m = pr["metrics"][ts]
            if "variants" not in m:
                continue
            print(f"\n== {pr['tag']}: one-step change D = H_after - H_before vs fp64, summed over {res['n_docs']} "
                  f"documents, {TOKEN_SETS[ts]} ==")
            print("step size |D64|/|H64| band medians: " + ", ".join(f"{b} {fmt(m['step_rel']['bands'][b], 0)}"
                                                                     for b in bn))
            print(f"{'variant':10s} {'measure':10s} " + " ".join(f"{b:>9}" for b in bn) + f" {'whole':>9}")
            for v in variants:
                e = m["variants"][v]
                for k, f in (("relerr", fmt), ("cos", fmtc), ("fwd_relerr", fmt), ("doc_relerr_median", fmt)):
                    whole = f(e["whole"][k]) if k in e["whole"] else ""
                    print(f"{v:10s} {k[:10]:10s} " + " ".join(f(e[k]["bands"][b]) for b in bn) + f" {whole:>9}")
        m = pr["metrics"]["full"]
        if "variants" in m:
            print(f"\n{pr['tag']} per layer (full token set): step size and |D_v - D64| / |D64|")
            print("layer  step_rel " + " ".join(f"{v:>10}" for v in variants))
            for li in range(res["n_layers"]):
                print(f"{li:5d} {fmt(m['step_rel']['by_layer'][li])} " +
                      " ".join(f"{fmt(m['variants'][v]['relerr']['by_layer'][li], 10)}" for v in variants))
    print(f"\nforward over all {res['n_docs']} documents (warm; mean of the timed forwards, 2 per pair)")
    print(f"{'variant':10s} {'rows':>4} {'fwd_s':>8} {'tok/s':>9} {'tok/s pad':>9} {'peak GiB':>8} {'wts GiB':>8} "
          f"{'determ.':>8}")
    for v in res["variants_run"]:
        t = res["timing"][v]
        pk = f"{t['peak_gib']:.1f}" if t["peak_gib"] is not None else "-"
        wg = f"{t['weights_gib']:.1f}" if t["weights_gib"] is not None else "-"
        print(f"{v:10s} {t['batch_rows']:4d} {t['forward_s_mean']:8.2f} {t['tokens_per_s_real']:9.0f} "
              f"{t['tokens_per_s_padded']:9.0f} {pk:>8} {wg:>8} {str(t['deterministic']['bitwise_equal']):>8}")
    print(f"\nintegrity checks passed {res['passed']}/{res['total']}: {json.dumps(res['checks'])}")
    if res["skipped"]:
        print(f"skipped checks: {json.dumps(res['skipped'])}")


# ------------------------------------------------------------------ main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair-step", action="append", default=[], metavar="RUN:STEP",
                    help=f"LoRA pair RUN/lora_pairs/step_<STEP> (repeatable; default {DEFAULT_PAIR_STEP})")
    ap.add_argument("--pair", action="append", default=[], metavar="DIR", help="LoRA pair directory (repeatable)")
    ap.add_argument("--docs", default=DEFAULT_DOCS, help="fixed documents (make_fixed_docs.py)")
    ap.add_argument("--model", default=None, help="model id or path (default: the runs' settings 'model')")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="subset of " + ",".join(VARIANTS))
    ap.add_argument("--batch-rows", type=int, default=64, help="rows per forward (halved on CUDA OOM)")
    ap.add_argument("--out", default=None, help=f"JSON path ending in {OUT_SUFFIX} (the npz files go next to it)")
    ap.add_argument("--no-docs-npz", action="store_true", help="do not store the per-document D arrays")
    return ap.parse_args(argv)


def out_prefix(out):
    if out.endswith(OUT_SUFFIX):
        return out[:-len(OUT_SUFFIX)]
    return out[:-5] if out.endswith(".json") else out


def save_npz(path, arrs):
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **arrs)
    os.replace(path + ".tmp", path)


def main(argv=None):
    a = parse_args(argv)
    from transformers import AutoTokenizer
    import peft
    import transformers

    t_start = time.time()
    out = a.out or os.path.join(CHECKS_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}{OUT_SUFFIX}")
    prefix = out_prefix(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    want = [x.strip() for x in a.variants.split(",") if x.strip()]
    bad = [x for x in want if x not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {VARIANTS}")
    variants = [v for v in VARIANTS if v in want]
    dev = torch.device(a.device)
    checks, skipped = {}, {}

    dirs = [pair_dir_of(s) for s in a.pair_step] + list(a.pair)
    pairs = load_pairs(dirs or [pair_dir_of(DEFAULT_PAIR_STEP)])
    st = pairs[0]["st"]
    model_id = a.model or st["model"]

    # documents, tokenised as ckpt_diff_inputs' fixed-document path
    docs_m, sha_m = mfd.load_fixed_docs(a.docs)
    rows, half, docs_sha = cdi.load_fixed_docs(a.docs)
    checks["docs_file_consistent"] = bool(len(docs_m) == len(rows) and sha_m == docs_sha)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    if tok.pad_token != "<|endoftext|>":
        raise SystemExit(f"pad token {tok.pad_token!r}; train_subliminal pads with <|endoftext|>")
    batches, ntok, ids_sha = cdi.fixed_batches(tok, rows, half, st)
    n_docs = len(rows)
    tokens_real = int(sum(int(b["attention_mask"].sum()) for _, _, b in batches))
    tokens_padded = int(sum(b["attention_mask"].size for _, _, b in batches))
    tokens_summed = int(sum(max(n - 1, 0) for n in ntok))
    for p in pairs:
        cd = os.path.join(p["run"], cdi.FIXED_SUBDIR)
        p["cd"] = cd
        if os.path.exists(os.path.join(cd, "docs.json")):
            with open(os.path.join(cd, "docs.json")) as f:
                rd = json.load(f)
            p["checks"]["tokenisation_matches_ckptdiff"] = bool(rd.get("docs_sha256") == docs_sha
                                                                and rd.get("ids_sha1") == ids_sha)
        else:
            p["skipped"]["tokenisation_matches_ckptdiff"] = f"no {cd}/docs.json"
        log(f"pair {p['tag']} ({p['dir']}): parameter step |after - before| / |before| = {p['pstep']['all']:.4g} "
            f"(LoRA B {p['pstep']['lora_B']:.4g}), lr {p['meta'].get('lr')}")
    log(f"{n_docs} documents, {tokens_real} tokens fed ({tokens_padded} padded, {tokens_summed} summed), batch widths "
        f"{[int(b['input_ids'].shape[1]) for _, _, b in batches]}; {len(pairs)} pairs; variants {variants}")

    # model: fp32 base + fp32 LoRA, no LM head
    t0 = time.time()
    set_matmul("ieee")
    pm = build_model(model_id, pairs[0]["meta"]["lora_config"], str(dev), dtype=torch.float32)
    base = pm.get_base_model()
    if hasattr(base, "lm_head"):
        base.lm_head = torch.nn.Identity()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    load_s = time.time() - t0
    names = lora_names(pm)
    layers = list(range(len(core.decoder_layers(pm))))
    from peft.tuners.lora.layer import LoraLayer

    scal = sorted({float(m.scaling[k]) for m in pm.modules() if isinstance(m, LoraLayer) for k in m.scaling})
    checks["lora_scaling_matches_meta"] = bool(all(scal == sorted({float(x) for x in p["meta"]["lora_scaling"]})
                                                   for p in pairs))
    n_rms, rms = check_rms_patch(pm, dev)
    checks["rmsnorm_patch_identical_fp32_bf16"] = bool(rms["identical_fp32"] and rms["identical_bf16"])
    checks["rmsnorm_patch_fp64_active"] = bool(rms["fp64_dtype"] == "torch.float64"
                                               and rms["fp64_relerr_patched"] <= RMS_FP64_TOL)
    log(f"loaded {model_id} in fp32 with the LoRA wrapper, LM head removed ({load_s:.1f} s"
        + (f", {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB" if dev.type == "cuda" else "")
        + f"); {n_rms} RMSNorm patched: {json.dumps(rms)}")

    R = {p["tag"]: {} for p in pairs}
    timing, state, covered, finite = {}, "fp32", True, True
    for v in variants:
        spec = SPEC[v]
        t0 = time.time()
        if spec["weights"] == "fp32" and state != "fp32":
            raise RuntimeError("variants must run in the order " + ",".join(VARIANTS))
        if spec["weights"] in ("bf16", "fp64") and "base_weights_bf16_exact" not in checks:
            nb, nt = bf16_representable(pm)
            checks["base_weights_bf16_exact"] = nb == 0
            log(f"base parameters not bf16-representable: {nb} of {nt}")
        if spec["weights"] == "bf16" and state == "fp32":
            cast_base(pm, torch.bfloat16)
            state = "bf16"
        elif spec["weights"] == "fp64" and state != "fp64":
            pm.double()
            state = "fp64"
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        cast_s = time.time() - t0
        flags = set_matmul(spec["matmul"])
        bdt, ldt = param_dtypes(pm)
        exp_b, exp_l = EXPECTED_DTYPES[spec["weights"]]
        checks[f"dtypes_{v}"] = bool(bdt == {exp_b} and ldt == {exp_l})
        bs = max(1, int(a.batch_rows))
        while True:
            try:
                r = run_variant(pm, names, pairs, batches, n_docs, layers, spec["autocast"], bs)
                break
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                log(f"{v}: CUDA out of memory, retrying with {bs} rows per forward")
        fws = []
        for p, q in zip(pairs, r["per"]):
            covered &= all(c["covered"] for c in q["cover"].values())
            finite &= all(np.isfinite(q[k]).all() for k in ("b_full", "b_p0", "a_full", "a_p0"))
            fws += [q["t_before"], q["t_after"]]
            rec = dict(Hb=q["b_full"].sum(0), Ha=q["a_full"].sum(0), Hb0=q["b_p0"].sum(0), Ha0=q["a_p0"].sum(0),
                       dd=q["a_full"] - q["b_full"], dd0=q["a_p0"] - q["b_p0"],
                       forward_s=[round(q["t_before"], 3), round(q["t_after"], 3)])
            if v == REF:
                rec["hb_doc_norm"] = np.linalg.norm(q["b_full"], axis=-1)
                rec["hb_doc_norm_nopos0"] = np.linalg.norm(q["b_full"] - q["b_p0"], axis=-1)
            R[p["tag"]][v] = rec
            if v == "fp32":
                href = os.path.join(p["cd"], f"h_step_{p['before_step']:06d}.npy")
                if os.path.exists(href) and p["checks"].get("before_equals_saved_adapter"):
                    ref = np.load(href)
                    mine = np.stack([q["b_full"][[i for i, h in enumerate(half) if h == hv]].sum(0) for hv in (0, 1)])
                    rel = float(np.linalg.norm(mine - ref) / np.linalg.norm(ref))
                    rel_l = float(np.max(np.linalg.norm(mine - ref, axis=-1) / np.linalg.norm(ref, axis=-1)))
                    p["checks"]["h_before_fp32_matches_ckptdiff"] = bool(rel_l <= CKPTDIFF_TOL)
                    p["ckptdiff_h_before"] = dict(file=href, relerr_whole=rel, relerr_max_layer_half=rel_l,
                                                  bitwise_equal=bool(np.array_equal(mine, ref)))
                    log(f"{p['tag']}: fp32 H_before vs {href}: relative difference {rel:.3g} (max over layer and "
                        f"half {rel_l:.3g})")
                else:
                    p["skipped"]["h_before_fp32_matches_ckptdiff"] = (
                        f"no {href}" if not os.path.exists(href) else "before is not the saved adapter")
        fw = float(np.mean(fws))
        timing[v] = dict(spec=dict(weights=spec["weights"], lora=str(exp_l).replace("torch.", ""),
                                   autocast=str(spec["autocast"]).replace("torch.", "") if spec["autocast"] else None,
                                   matmul=flags),
                         batch_rows=bs, cast_s=round(cast_s, 2), warmup_s=round(r["t_warm"], 3),
                         forward_s_all=[round(x, 3) for x in fws], forward_s_mean=round(fw, 3),
                         tokens_per_s_real=tokens_real / fw, tokens_per_s_padded=tokens_padded / fw,
                         peak_gib=r["peak"] / 2**30 if r["peak"] is not None else None,
                         weights_gib=r["mem0"] / 2**30 if r["mem0"] is not None else None,
                         deterministic=r["det"])
        log(f"{v}: forward {fw:.2f} s (mean of {len(fws)}; warm-up {r['t_warm']:.2f}), {bs} rows, peak "
            f"{None if r['peak'] is None else round(r['peak'] / 2**30, 1)} GiB, deterministic "
            f"{r['det']['bitwise_equal']} ({r['det']['max_abs_diff_over_max_abs']:.3g})")
        del r
    checks["adapter_coverage"] = bool(covered)
    checks["finite"] = bool(finite)
    for p in pairs:
        checks.update({f"{p['tag']}:{k}": ok for k, ok in p["checks"].items()})
        skipped.update({f"{p['tag']}:{k}": why for k, why in p["skipped"].items()})

    pos_def = f"{TOKEN_SETS['full']}; {adamfx.TOKEN_SET}"
    pair_out = []
    for p in pairs:
        Rp = R[p["tag"]]
        vec_path = f"{prefix}_{p['tag']}_step_delta_vectors.npz"
        sums_path = f"{prefix}_{p['tag']}_step_delta_sums.npz"
        vmeta = dict(run=p["run"], lora_pair=p["dir"], lora_pair_step=p["step"], before_step=p["before_step"],
                     lr=p["meta"].get("lr"), docs=os.path.abspath(a.docs), docs_sha256=docs_sha, ids_sha1=ids_sha,
                     n_docs=n_docs, positions=pos_def, sign=SIGN, summed_over="the fixed documents",
                     variants={v: timing[v]["spec"] for v in VECTOR_VARIANTS if v in Rp},
                     h_before_fp64="sum over the documents of tokensum(h_L | before), fp64 forward",
                     dtype="float32 (fp64 sums and differences rounded at storage)", model=model_id,
                     json=os.path.abspath(out), written=time.strftime("%Y-%m-%d %H:%M:%S"))
        varrs = {f"delta_{v}": (Rp[v]["Ha"] - Rp[v]["Hb"]).astype(np.float32) for v in VECTOR_VARIANTS if v in Rp}
        if REF in Rp:
            varrs["h_before_fp64"] = Rp[REF]["Hb"].astype(np.float32)
        varrs["layers"] = np.asarray(layers, np.int64)
        varrs["meta"] = np.array(json.dumps(vmeta))
        save_npz(vec_path, varrs)
        sarrs = {f"{k}_{v}": Rp[v][k] for v in Rp for k in ("Hb", "Ha", "Hb0", "Ha0")}
        if not a.no_docs_npz:
            for v in VECTOR_VARIANTS:
                if v in Rp:
                    sarrs[f"D_docs_{v}"] = Rp[v]["dd"] if v == REF else Rp[v]["dd"].astype(np.float32)
        sarrs["meta"] = np.array(json.dumps(dict(vmeta, kind="sums", per_document="D_docs_<variant> [n, L, D]")))
        save_npz(sums_path, sarrs)
        pair_out.append(dict(tag=p["tag"], pair=p["dir"], run=p["run"], step=p["step"], before_step=p["before_step"],
                             lr=p["meta"].get("lr"), param_step=p["pstep"],
                             forward_s={v: Rp[v]["forward_s"] for v in Rp},
                             ckptdiff_h_before=p.get("ckptdiff_h_before"), checks=p["checks"], skipped=p["skipped"],
                             metrics=summarise(Rp, variants), vectors_npz=os.path.abspath(vec_path),
                             sums_npz=os.path.abspath(sums_path)))
        log(f"{p['tag']}: written {vec_path} and {sums_path}")
    res = dict(question="Is fp32 enough to get the change of the layer outputs that one real optimizer step makes by "
                        "subtracting two forwards, or is fp64 needed; cost and accuracy of each precision",
               plan="module docstring of main_run/measure_step_delta_precision.py", model=model_id,
               docs=os.path.abspath(a.docs), docs_sha256=docs_sha, ids_sha1=ids_sha, n_docs=n_docs,
               n_layers=len(layers), d_model=int(R[pairs[0]["tag"]][variants[0]]["Hb"].shape[-1]),
               tokens_real=tokens_real, tokens_padded=tokens_padded, tokens_summed=tokens_summed,
               batch_widths=[int(b["input_ids"].shape[1]) for _, _, b in batches], sign=SIGN, positions=pos_def,
               reference=REF, variants_run=variants, token_sets=TOKEN_SETS,
               bands_def={b: [lo, hi] for b, lo, hi in BANDS}, rmsnorm=dict(n_patched=n_rms, **rms),
               load_s=round(load_s, 1), timing=timing, pairs=pair_out, checks=checks, skipped=skipped,
               passed=int(sum(checks.values())), total=len(checks), device=str(dev),
               device_name=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
               versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
               written=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t_start, 1))
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out + ".tmp", out)
    print_tables(res)
    log(f"written {out} ({res['wall_s']} s)")
    return 0 if res["passed"] == res["total"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Offline fp32 true change at a LoRA-pair step (``train_subliminal.py --save-adamfx --adamfx-lora-pairs``).

The in-training true change (adamfx.py ``acttrue``) is a difference of two bf16 forwards, close to the bf16 rounding
floor for one step. This script recomputes it in fp32 for one saved step and compares it with the vectors stored for
the same documents:

  1. Batch: the 8 rows at the pair's ``line_no`` are rebuilt from the run's RUN.json (data_path, n_rows, arm and the
     mixed-arm settings, tokenizer of ``model``, max_len, think_loss) with train_subliminal.tokenize_rows and
     CompletionMaskCollator, and must equal ``batch.npz`` (the exact tensors of the step); ``batch.npz`` is used.
  2. Model: the base model in fp32 (``dtype=float32`` at load; Qwen3-14B is about 59 GB of weights, plus 0.26 GB of
     LoRA and the activations of 8 short rows: one GPU with 70 GB free) with the run's LoRA config (peft), eval
     mode, no autocast, no grad.
  3. True change: the before and after LoRA parameters (``before.safetensors`` and ``after.safetensors``, or
     before + delta for the bf16-delta format) are loaded in turn; forward hooks on the decoder layers sum the layer
     output over the gradient store's full token set (positions 0..n-2, fp64 reduction); dh = sum(after) -
     sum(before) per document and layer, in the docs-file sign (+dh).
  4. Comparison per layer: cosine (median over the documents, and of the batch sums) with the stored first-order
     vector (``adamfx``, +dh: the direct writes of layer L only), its cumulative sum over layers 0..L
     (``adamfx_cumulative``: the direct writes of every layer up to L, which the residual stream carries into layer
     L's output; only when the layers are 0..n-1), the stored bf16 true change (``acttrue``, +dh) and the stored
     gradient (``grad``, -g), and the norm ratios |dh fp32| / |first order| and / |cumulative first order|.
  5. With ``--grad-run <run>`` (a run whose ``grads/`` store holds the per-document gradients of the same trajectory:
     the original run when ``--run`` is a re-run), the Adam-normalised reader inputs of e2_read.adam_inputs at the
     step are rebuilt from that store and compared too, in the +dh sign: ``adam_hacky`` = -m_hat / den (the adam
     input; batch level only, m mixes the earlier steps' documents), ``rms_hacky`` = -S / den (the rms1000 input, S
     = per-step batch sums over the last ``--rms-window`` stored steps) and, per document, ``grad_over_rms`` = -g_d /
     den (den = sqrt(v_hat) + eps as in adam_inputs). The rebuilt adam input is checked against the run's
     ``e2/adam_inputs_every100.npz`` at the step when that file has it (relative error in the json).

Output: a per-layer table on stdout and ``<pair dir>/fp32_check.json``; ``--save-dh <npz>`` also keeps dh.

Usage:
  CUDA_VISIBLE_DEVICES=3 python adamfx_fp32_check.py --run <out> --step 1000 [--device cuda:0] [--model <override>]
      [--grad-run <original run>] [--rms-window 1000] [--save-dh <npz>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adamfx  # noqa: E402
from agm import core  # noqa: E402


def pair_dir(run, step):
    return os.path.join(run, "lora_pairs", f"step_{int(step):06d}")


def load_pair(d):
    """meta, before and after (name -> fp32 tensor), batch and vectors (dicts of numpy arrays) of a pair directory."""
    from safetensors.torch import load_file

    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    before = {k: v.float() for k, v in load_file(os.path.join(d, "before.safetensors")).items()}
    if meta["format"] == "fp32":
        after = {k: v.float() for k, v in load_file(os.path.join(d, "after.safetensors")).items()}
    else:
        delta = load_file(os.path.join(d, "delta.safetensors"))
        after = {k: before[k] + delta[k].float() for k in before}
    with np.load(os.path.join(d, "batch.npz")) as z:
        batch = {k: z[k] for k in z.files}
    with np.load(os.path.join(d, "vectors.npz")) as z:
        vectors = {k: z[k] for k in z.files}
    return meta, before, after, batch, vectors


def rebuild_batch(settings, line_no, tok):
    """input_ids, attention_mask, labels (numpy) of the rows at ``line_no``, as the trainer collated them."""
    import train_subliminal as ts

    if settings["arm"] == "mixed":
        rows, _, _, checks = ts.load_mixed_rows(settings["data_path"], settings["mix_clean_path"],
                                                settings["mix_clean_index"], settings["n_rows"],
                                                settings["mix_dolphin_frac"], settings["mix_seed"])
        if checks["failed"]:
            raise SystemExit(f"mixing checks failed: {checks['failed']}")
    else:
        rows = ts.read_rows(settings["data_path"], settings["n_rows"])
    sel = [rows[int(j)] for j in line_no]
    ids, masks, _ = ts.tokenize_rows(tok, sel, settings["max_len"], settings["think_loss"])
    b = ts.CompletionMaskCollator(tok)([dict(input_ids=i, completion_mask=m) for i, m in zip(ids, masks)])
    return {k: b[k].numpy() for k in ("input_ids", "attention_mask", "labels")}


def build_model(model_id, lora_config, device, dtype=torch.float32):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa",
                                                 device_map={"": device})
    cfg = dict(lora_config)
    cfg["task_type"] = TaskType.CAUSAL_LM
    pm = get_peft_model(model, LoraConfig(**cfg))
    for p in pm.parameters():
        p.requires_grad_(False)
    pm.eval()
    return pm


def load_lora(pm, params):
    named = dict(pm.named_parameters())
    missing = [k for k in params if k not in named]
    if missing:
        raise KeyError(f"{len(missing)} saved LoRA names are not in the model, e.g. {missing[:3]}")
    lora_names = [n for n in named if "lora_" in n]
    absent = [n for n in lora_names if n not in params]
    if absent:
        raise KeyError(f"{len(absent)} model LoRA parameters are not in the pair, e.g. {absent[:3]}")
    with torch.no_grad():
        for k, v in params.items():
            named[k].copy_(v.to(named[k].dtype))


def layer_sums(pm, input_ids, attention_mask, layers):
    """[L, B, D] fp64 sums of the decoder-layer outputs over the full token set (positions 0..n-2)."""
    dev = next(pm.parameters()).device
    am = torch.as_tensor(attention_mask, device=dev)
    W = adamfx.token_weights(am.sum(1).tolist(), am.shape[1], dev)
    dec = core.decoder_layers(pm)
    S = {}
    hs = []
    for li, L in enumerate(layers):
        def hook(mod, inp, out, li=li):
            h = out[0] if isinstance(out, tuple) else out
            S[li] = torch.bmm(W.double(), h.double())[:, 0]
        hs.append(dec[L].register_forward_hook(hook))
    base = pm.get_base_model() if hasattr(pm, "get_base_model") else pm
    try:
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
            base.model(input_ids=torch.as_tensor(input_ids, device=dev), attention_mask=am, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([S[li] for li in range(len(layers))])


def true_change(pm, before, after, input_ids, attention_mask, layers):
    """Per-document fp32 true change, [B, L, D] fp64, sign +dh (sum after - sum before)."""
    load_lora(pm, before)
    s0 = layer_sums(pm, input_ids, attention_mask, layers)
    load_lora(pm, after)
    s1 = layer_sums(pm, input_ids, attention_mask, layers)
    return (s1 - s0).transpose(0, 1).cpu().numpy()


def _cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def adam_state_at(gs, names, step, beta1=0.9, beta2=0.999, eps_rel=1e-8, rms_window=1000, log=print):
    """The Adam-normalised reader inputs (e2_read.adam_inputs) at ``step`` from a run's gradient store, [L, D] fp64
    in the stored +g sign: ``adam`` = m_hat / den, ``rms`` = S / den (S = sum of the per-step batch sums over the
    last ``rms_window`` stored steps up to ``step``) and ``den`` = sqrt(v_hat) + eps itself (for -g_d / den per
    document). The recurrence is adam_inputs' (fp64 m, v of the per-step batch sums, bias-corrected, eps per layer =
    eps_rel x mean sqrt(v_hat)), run from the first stored step and stopped at ``step``."""
    import e2_read

    L, D = len(names), gs.layout["d_model"]
    m, v = np.zeros((L, D)), np.zeros((L, D))
    seen = dict(t=0, last=None)

    def hook(ustep, per, ndocs):
        G = np.stack([per[nm] for nm in names], axis=1)  # [k, L, D] fp64, +g
        for j, t_step in enumerate(ustep):
            if int(t_step) > step:
                return
            np.multiply(m, beta1, out=m)
            np.add(m, (1.0 - beta1) * G[j], out=m)
            np.multiply(v, beta2, out=v)
            np.add(v, (1.0 - beta2) * G[j] * G[j], out=v)
            seen["t"] += 1
            seen["last"] = int(t_step)

    sums, wmeta = e2_read.window_sums(gs, names, [step], [("w", rms_window)], log=log, step_hook=hook)
    if seen["last"] != step:
        raise RuntimeError(f"the store's last step seen is {seen['last']}, not {step}")
    t = seen["t"]
    sv = np.sqrt(v / (1.0 - beta2 ** t))
    den = sv + (eps_rel * sv.mean(axis=1))[:, None]
    S = np.stack([sums["w"][nm][0] for nm in names]).astype(np.float64)
    return dict(adam=(m / (1.0 - beta1 ** t)) / den, rms=S / den, den=den, t=t, window=[int(x) for x in wmeta["w"][0]])


def compare(dh, vectors, layers, batch_vectors=None):
    """Per-layer cosines of the fp32 true change ``dh`` [B, L, D] with each stored vector series of the same docs
    (median over documents and of the batch sums) and with each batch-level vector [L, D] (batch sums only)."""
    out = {"layers": [int(L) for L in layers]}
    B = dh.shape[0]
    vectors = dict(vectors)
    if "adamfx" in vectors and list(layers) == list(range(len(layers))):
        vectors["adamfx_cumulative"] = np.cumsum(np.asarray(vectors["adamfx"], np.float64), axis=1)
    for name, v in vectors.items():
        v = np.asarray(v, np.float64)
        out[name] = dict(
            cos_median_docs=[float(np.median([_cos(dh[d, li], v[d, li]) for d in range(B)])) for li in range(len(layers))],
            cos_batch_sum=[_cos(dh[:, li].sum(0), v[:, li].sum(0)) for li in range(len(layers))])
    for name, v in (batch_vectors or {}).items():
        v = np.asarray(v, np.float64)
        out[name] = dict(cos_batch_sum=[_cos(dh[:, li].sum(0), v[li]) for li in range(len(layers))])
    for name, key in (("adamfx", "norm_ratio_fp32_true_over_first_order"),
                      ("adamfx_cumulative", "norm_ratio_fp32_true_over_cumulative_first_order")):
        if name in vectors:
            fx = np.asarray(vectors[name], np.float64)
            out[key] = [float(np.linalg.norm(dh[:, li].sum(0)) / max(np.linalg.norm(fx[:, li].sum(0)), 1e-300))
                        for li in range(len(layers))]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=None, help="override RUN.json settings.model (same weights, other path)")
    ap.add_argument("--grad-run", default=None,
                    help="run whose grads/ store gives the Adam-normalised reader inputs at the step (same trajectory)")
    ap.add_argument("--rms-window", type=int, default=1000, help="window of the rms input's numerator (steps)")
    ap.add_argument("--save-dh", default=None, help="npz to keep the fp32 true change [B, L, D] (+dh) with layers, line_no")
    a = ap.parse_args(argv)
    t0 = time.time()
    from transformers import AutoTokenizer

    with open(os.path.join(a.run, "RUN.json")) as f:
        rj = json.load(f)
    st = rj["settings"]
    d = pair_dir(a.run, a.step)
    meta, before, after, batch, vectors = load_pair(d)
    model_id = a.model or st["model"]
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    rebuilt = rebuild_batch(st, meta["line_no"], tok)
    same = {k: bool(np.array_equal(rebuilt[k], batch[k])) for k in ("input_ids", "attention_mask", "labels") if k in batch}
    if not all(same.values()):
        raise SystemExit(f"the batch rebuilt from RUN.json differs from batch.npz: {same}")
    pm = build_model(model_id, meta["lora_config"], a.device)
    layers = meta["layers"]
    dh = true_change(pm, before, after, batch["input_ids"], batch["attention_mask"], layers)
    batch_vectors, adam_info = {}, None
    if a.grad_run:
        import online_save

        gs = online_save.GradsStore(os.path.join(a.grad_run, "grads"))
        names = [core.array_name("full", "grad", L) for L in layers]
        ad = adam_state_at(gs, names, a.step, rms_window=a.rms_window)
        batch_vectors["adam_hacky"] = -ad["adam"]  # the adam reader input (-m_hat/den), +dh sign
        batch_vectors["rms_hacky"] = -ad["rms"]  # the rms1000 reader input (-S/den) with this window
        vectors = dict(vectors)
        if "grad" in vectors:
            vectors["grad_over_rms"] = np.asarray(vectors["grad"], np.float64) / ad["den"][None]  # -g_d/den per document
        adam_info = dict(grad_run=os.path.abspath(a.grad_run), t=ad["t"], rms_window=[a.rms_window, ad["window"]],
                         den_mean_per_layer=[float(x) for x in ad["den"].mean(axis=1)])
        ref = os.path.join(a.grad_run, "e2", "adam_inputs_every100.npz")
        if os.path.exists(ref):
            with np.load(ref) as z:
                pos = np.flatnonzero(z["steps"] == a.step)
                if len(pos) and list(z["layers"]) == [int(L) for L in layers]:
                    r = z["inputs"][int(pos[0]), 0].astype(np.float64)
                    adam_info["adam_input_file_rel_err"] = float(np.linalg.norm(r - batch_vectors["adam_hacky"]) /
                                                                max(np.linalg.norm(r), 1e-300))
    res = compare(dh, vectors, layers, batch_vectors)
    if adam_info:
        res["adam_inputs"] = adam_info
    res.update(step=a.step, run=os.path.abspath(a.run), pair_format=meta["format"], lr=meta["lr"],
               batch_rebuilt_equal=same, n_docs=int(dh.shape[0]), seconds=round(time.time() - t0, 1),
               dtype="float32 forward, fp64 token-sum reduction", sign="+dh (after - before), as the docs files",
               fp32_true_change_norms=[float(np.linalg.norm(dh[:, li].sum(0))) for li in range(len(layers))])
    with open(os.path.join(d, "fp32_check.json.tmp"), "w") as f:
        json.dump(res, f, indent=1)
    os.replace(os.path.join(d, "fp32_check.json.tmp"), os.path.join(d, "fp32_check.json"))
    if a.save_dh:
        np.savez(a.save_dh, dh=dh.astype(np.float32), layers=np.asarray(layers), line_no=np.asarray(meta["line_no"]),
                 step=a.step, sign="+dh (after - before)")
    names = [n for n in ("adamfx", "adamfx_cumulative", "acttrue", "grad", "grad_over_rms", "adam_hacky", "rms_hacky")
             if n in res]
    print(f"step {a.step}, lr {meta['lr']:.3e}, {dh.shape[0]} documents; batch rebuilt from RUN.json equals batch.npz")
    print("cosines of the fp32 true change (median over documents / of the batch sum); norm ratios of the batch sums")
    print("layer  " + "  ".join(f"{n:>17s}" for n in names) + "  |fp32|/|fx|  |fp32|/|fx cum|")
    nan = [float("nan")] * len(layers)
    for li, L in enumerate(layers):
        cols = "  ".join((f"{res[n]['cos_median_docs'][li]:8.4f}" if "cos_median_docs" in res[n] else " " * 8)
                         + f"/{res[n]['cos_batch_sum'][li]:8.4f}" for n in names)
        r1 = res.get("norm_ratio_fp32_true_over_first_order", nan)[li]
        r2 = res.get("norm_ratio_fp32_true_over_cumulative_first_order", nan)[li]
        print(f"{L:5d}  {cols}  {r1:10.3f}  {r2:13.3f}")
    print(f"wrote {os.path.join(d, 'fp32_check.json')} ({res['seconds']} s)")
    return res


if __name__ == "__main__":
    main()

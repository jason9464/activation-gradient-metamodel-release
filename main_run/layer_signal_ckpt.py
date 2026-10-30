#!/usr/bin/env python
"""Actual residual-stream movement of the students across their saved checkpoints, per layer, along the dolphin direction.

  CUDA_VISIBLE_DEVICES=2 python layer_signal_ckpt.py [--runs full_dolphin_s1,...] [--steps all|0,500,...]
      [--max-prompts N] [--out-dir <dir>] [--summary-dir <dir>] [--summarize-only] [--force]

For each student run (Qwen3-14B, LoRA r16 on the 7 projection modules) the base model is loaded in bf16 (sdpa) inside
a PEFT LoRA wrapper built from the run's first adapter_config.json; the model is loaded once and reused while the next
run names the same model. Checkpoint 0 is the base model: every LoRA B is set to zero, so the LoRA branch adds exact
zeros (checked bitwise against PEFT's disable_adapter). Checkpoints 500..20,000 are the saved adapters; their tensors
are copied into the wrapper's LoRA parameters with adamfx_fp32_check.load_lora after inserting PEFT's adapter name
(".lora_A.weight" -> ".lora_A.default.weight"); every LoRA parameter of the model must be covered by every adapter.

Prompt set, fixed for all checkpoints: the 50 favourite-animal questions of the student's own full eval
(subliminal_eval.ANIMAL_PROMPTS, the source of evals.jsonl's dolphin rate, verbatim) and 4 neutral prompts (maths,
summary, coding, geography). Each is rendered with subliminal_eval.render (Qwen3 chat template, enable_thinking=False,
generation prompt added), so the last prompt token is the position where the answer begins. Batched with right
padding; causal attention keeps the real positions of each prompt independent of the padding.

Recorded per decoder layer L (forward hooks on agm.core.decoder_layers, the layer output = residual stream h_L), in
fp32: h_L at the last prompt token, h_L averaged over all prompt tokens, and the logit-lens values at every token.

Measures, per layer L, checkpoint s and prompt p (sign +dh: the actual change of the residual stream):
  D_L(s, p) = h_L(s, p) - h_L(0, p)
  cos(D_L, d_tok), <D_L, d_tok> and |D_L|, for 11 unit token directions d_tok (cos is set to 0 where |D_L| = 0)
  logit lens: lens_tok(h) = <RMSNorm(h), W_U[tok]> - mean_v <RMSNorm(h), W_U[v]>
                          = rsqrt(mean(h^2) + eps) * <h, gamma * (W_U[tok] - mean_v W_U[v])>
    (the final norm and lm_head applied to h_L, token logit minus the mean logit over the vocabulary), kept as its
    value and as its change since checkpoint 0. Checked at the last layer against the model's own output logits.
Token directions: d_tok = gamma * W_U[tok] - mean_v(gamma * W_U[v]), unit-normalised, from the loaded model's
model.norm.weight (gamma) and lm_head.weight (W_U; Qwen3-14B does not tie embeddings), fp32 on CPU; tok = first token
of " dolphin" and of 10 control animals (TOKENS).
Groups: animal questions at the last token (animal_last), animal questions averaged over all prompt tokens
(animal_mean; its logit lens is the token average of the per-token lens), neutral prompts at the last token
(neutral_last). Per group the per-prompt measures are averaged over the prompts; the measures of the prompt-averaged
D_L are kept as *_of_mean.

Outputs: per run <out-dir>/<run>.json and <run>_D.npz (prompt-averaged D_L of animal_last and neutral_last, fp16,
[checkpoint, layer, 5120]); the summary <summary-dir>/20261024_layer_signal_ckpt.json (all runs, the judgment, the
integrity checks with passed/total) and the figures 20261024_layer_signal_ckpt.png (heatmaps) and
20261024_layer_signal_ckpt_bands.png (layer bands with the students' dolphin rate).
Judgment: layers where both dolphin runs exceed both clean runs (cosine with the dolphin direction, animal questions,
last token) at step 2,500 and at every later checkpoint, and per layer the first checkpoint from which this holds at
every later checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF = os.environ["IIF_BASE"]
RUNS_DIR = os.path.join(IIF, "agm_data", "runs")
OUT_DIR = os.path.join(RUNS_DIR, "layer_signal")
SUMMARY_DIR = str(Path(HERE).parent / "docs" / "figures")
SUMMARY_NAME = "20261024_layer_signal_ckpt"
DIRECTIONS_REF = os.path.join(IIF, "agm_data", "signal_check", "dolphin_directions.npz")
RUNS = ["full_dolphin_s1", "full_dolphin_s2", "full_clean_s1", "full_clean_s2"]
LABELS = {"full_dolphin_s1": "dolphin seed 1", "full_dolphin_s2": "dolphin seed 2",
          "full_clean_s1": "clean seed 1", "full_clean_s2": "clean seed 2"}
STYLE = {"full_dolphin_s1": ("#08519c", "-"), "full_dolphin_s2": ("#d97b1a", "-"),
         "full_clean_s1": ("gray", "--"), "full_clean_s2": ("#b0b0b0", ":")}
TOKENS = [" dolphin", " owl", " eagle", " cat", " wolf", " octopus", " phoenix", " lion", " elephant", " bear", " tiger"]
NEUTRAL_PROMPTS = [
    "What is 17 times 23?",
    "Summarise this paragraph in one sentence: The town council met on Tuesday to discuss new bus routes. After a "
    "long debate, members agreed to add two evening services.",
    "How do I reverse a list in Python?",
    "What is the capital of Australia?",
]
GEN_PROMPT_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
ADAPTER_STEPS = list(range(500, 20001, 500))
ONSET = 2500
WINDOW = (1001, 3100)
BANDS = [(0, 13), (14, 26), (27, 39)]
GROUPS = ("animal_last", "animal_mean", "neutral_last")
MAX_PROMPT_TOKENS = 64


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sig(a, digits=6):
    """nested lists of floats rounded to ``digits`` significant digits (json size)"""
    def rec(x):
        return [rec(y) for y in x] if isinstance(x, list) else float(f"{x:.{digits}g}")
    return rec(np.asarray(a, dtype=np.float64).tolist())


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ------------------------------------------------------------------ prompts, tokens, directions

def prompt_set(max_prompts=None):
    from subliminal_eval import ANIMAL_PROMPTS

    animal, neutral = list(ANIMAL_PROMPTS), list(NEUTRAL_PROMPTS)
    if max_prompts:
        animal, neutral = animal[:max_prompts], neutral[:max_prompts]
    return animal, neutral


def encode(tok, questions):
    """right-padded ids, attention mask and last-token index of the rendered questions (generation prompt added)"""
    from subliminal_eval import render

    texts = [render(tok, q) for q in questions]
    for t in texts:
        assert t.endswith(GEN_PROMPT_TAIL), t[-80:]
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    T = max(len(x) for x in ids)
    inp = torch.full((len(ids), T), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    lengths = [len(x) for x in ids]
    return inp, att, torch.tensor([n - 1 for n in lengths]), lengths, tok.decode([ids[0][-1]])


def token_info(tok):
    out = []
    for s in TOKENS:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        out.append(dict(string=s, ids=ids, pieces=[tok.decode([i]) for i in ids], first_id=int(ids[0]),
                        first_piece=tok.decode([ids[0]])))
    return out


def token_directions(base, tinfo):
    """raw directions gamma*(W_U[tok]) - mean_v(gamma*W_U[v]) [K, D] and their unit versions, fp32 on CPU"""
    tied = bool(getattr(base.config, "tie_word_embeddings", False))
    W = (base.model.embed_tokens.weight if tied else base.lm_head.weight).detach().float().cpu()
    g = base.model.norm.weight.detach().float().cpu()
    GW = W * g[None, :]
    mean = GW.mean(0)
    raw = torch.stack([GW[t["first_id"]] - mean for t in tinfo])
    unit = raw / raw.norm(dim=1, keepdim=True)
    del GW
    return raw, unit, dict(tie_word_embeddings=tied, unembedding="model.embed_tokens.weight" if tied else "lm_head.weight",
                           vocab_rows=int(W.shape[0]), eps=float(base.model.norm.variance_epsilon))


def compare_reference(unit, tinfo):
    """max absolute difference of our unit directions to dolphin_directions.npz (whatever arrays of width D it has)"""
    if not os.path.exists(DIRECTIONS_REF):
        return dict(path=DIRECTIONS_REF, exists=False)
    res = dict(path=DIRECTIONS_REF, exists=True, compared={})
    ids = [t["first_id"] for t in tinfo]
    with np.load(DIRECTIONS_REF, allow_pickle=False) as z:
        keys = list(z.files)
        res["keys"] = keys
        id_key = next((k for k in keys if "id" in k.lower() and z[k].ndim == 1 and z[k].dtype.kind in "iu"), None)
        ref_ids = [int(x) for x in z[id_key]] if id_key else None
        for k in keys:
            a = np.asarray(z[k])
            if a.dtype.kind != "f" or a.shape[-1] != unit.shape[1]:
                continue
            a = a.astype(np.float64)
            if a.ndim == 1:
                a = a[None]
            a = a / np.linalg.norm(a, axis=-1, keepdims=True)
            ours = unit.double().numpy()
            if a.shape[0] == 1:
                diff = np.abs(a[0] - ours[0]).max()  # a single vector: the dolphin direction
                res["compared"][k] = dict(against=TOKENS[0], max_abs_diff=float(diff))
            elif ref_ids is not None and len(ref_ids) == a.shape[0]:
                rows = {}
                for j, tid in enumerate(ref_ids):
                    if tid in ids:
                        rows[TOKENS[ids.index(tid)]] = float(np.abs(a[j] - ours[ids.index(tid)]).max())
                res["compared"][k] = dict(by_token_id=rows, max_abs_diff=max(rows.values()) if rows else None)
            elif a.shape[0] == ours.shape[0]:
                res["compared"][k] = dict(row_order_assumed=True, max_abs_diff=float(np.abs(a - ours).max()))
    diffs = [v["max_abs_diff"] for v in res["compared"].values() if v.get("max_abs_diff") is not None]
    res["max_abs_diff"] = max(diffs) if diffs else None
    return res


# ------------------------------------------------------------------ model and LoRA

class ModelCache:
    def __init__(self, device):
        self.device, self.key, self.pm, self.tok, self.info = device, None, None, None, None

    def get(self, model_id, adapter_dir):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = LoraConfig.from_pretrained(adapter_dir)
        key = (model_id, json.dumps(cfg.to_dict(), sort_keys=True, default=str))
        if key == self.key:
            return self.pm, self.tok, False
        self.pm = None
        torch.cuda.empty_cache()
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                     device_map={"": self.device})
        cfg.inference_mode = True
        pm = get_peft_model(model, cfg)
        for p in pm.parameters():
            p.requires_grad_(False)
        pm.eval()
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "right"
        assert tok.pad_token == "<|endoftext|>", tok.pad_token
        self.key, self.pm, self.tok = key, pm, tok
        log(f"loaded {model_id} in bf16 with the LoRA wrapper ({time.time() - t0:.1f} s)")
        return pm, tok, True


def lora_names(pm):
    return sorted(n for n, _ in pm.named_parameters() if "lora_" in n)


def map_adapter(saved, adapter_name="default"):
    """saved PEFT names (...lora_A.weight) -> wrapper parameter names (...lora_A.default.weight)"""
    out = {}
    for k, v in saved.items():
        for ab in ("lora_A", "lora_B"):
            k = k.replace(f".{ab}.weight", f".{ab}.{adapter_name}.weight")
        out[k] = v
    return out


def load_adapter(pm, path, names):
    """copy one saved adapter into the wrapper; returns the coverage record (raises when not every parameter is covered)"""
    from safetensors.torch import load_file
    from adamfx_fp32_check import load_lora

    params = map_adapter(load_file(path))
    named = dict(pm.named_parameters())
    shape_bad = [k for k in params if k in named and tuple(params[k].shape) != tuple(named[k].shape)]
    cover = dict(saved=len(params), model_lora=len(names), missing_in_adapter=len(set(names) - set(params)),
                 unknown_in_model=len(set(params) - set(names)), shape_mismatch=len(shape_bad))
    cover["covered"] = cover["missing_in_adapter"] == 0 and cover["unknown_in_model"] == 0 and not shape_bad
    if not cover["covered"]:
        raise KeyError(f"adapter {path} does not cover the wrapper's LoRA parameters: {cover}")
    load_lora(pm, params)
    return cover


def zero_lora_b(pm):
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.zero_()


# ------------------------------------------------------------------ forward and measures

def record(pm, inp, att, last_idx, raw_dirs, eps, disable=False):
    """per layer, fp32 on CPU: h at the last token [P, L, D], h averaged over the prompt tokens [P, L, D], logit lens at
    the last token [P, L, K] and averaged over the prompt tokens [P, L, K]"""
    from agm import core

    dev = next(pm.parameters()).device
    inp, att, li = inp.to(dev), att.to(dev), last_idx.to(dev)
    m = att.float()[..., None]
    n = att.sum(1).float()[:, None]
    ar = torch.arange(inp.shape[0], device=dev)
    R = raw_dirs.to(dev).T.contiguous()  # [D, K] fp32
    dec = core.decoder_layers(pm)
    S = {}

    def make(L):
        def hook(mod, args, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            r = torch.rsqrt(h.pow(2).mean(-1) + eps)  # [P, T], the final RMSNorm's scale in fp32
            z = torch.matmul(h, R) * r[..., None]  # [P, T, K] logit lens: token logit minus the mean logit
            S[L] = ((h[ar, li]).cpu(), ((h * m).sum(1) / n).cpu(), z[ar, li].cpu(), ((z * m).sum(1) / n).cpu())
        return hook

    hs = [dec[L].register_forward_hook(make(L)) for L in range(len(dec))]
    base = pm.get_base_model()
    ctx = pm.disable_adapter() if disable else contextlib.nullcontext()
    try:
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False), ctx:
            base.model(input_ids=inp, attention_mask=att, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    Ls = range(len(dec))
    return tuple(torch.stack([S[L][j] for L in Ls], dim=1) for j in range(4))


def model_logit_check(pm, inp, att, last_idx, tinfo, lens_last_final):
    """the model's own output logits (token minus mean) at the last token vs the logit lens of the last layer"""
    dev = next(pm.parameters()).device
    with torch.no_grad():
        lg = pm.get_base_model()(input_ids=inp.to(dev), attention_mask=att.to(dev), use_cache=False).logits
    lg = lg[torch.arange(inp.shape[0], device=dev), last_idx.to(dev)].float()
    ids = torch.tensor([t["first_id"] for t in tinfo], device=dev)
    own = (lg[:, ids] - lg.mean(1, keepdim=True)).cpu()  # [P, K]
    diff = (own - lens_last_final).abs()
    scale = own.abs().max().item()
    corr = float(np.corrcoef(own.flatten().numpy(), lens_last_final.flatten().numpy())[0, 1])
    del lg
    return dict(max_abs_diff=float(diff.max()), max_abs_value=scale, corr=corr,
                passed=bool(diff.max() <= 0.02 * scale + 0.25 and corr > 0.999))


def group_measures(H, H0, Z, Z0, idx, U, per_prompt=False):
    """measures of one prompt group at one checkpoint; H, H0 [P, L, D]; Z, Z0 [P, L, K]; U [K, D] unit directions"""
    D = H[idx] - H0[idx]  # [p, L, D] +dh
    nrm = D.norm(dim=-1)  # [p, L]
    proj = torch.matmul(D, U.T)  # [p, L, K]
    cos = torch.where(nrm[..., None] > 0, proj / nrm[..., None].clamp_min(1e-30), torch.zeros_like(proj))
    Dm = D.mean(0)  # [L, D]
    nm = Dm.norm(dim=-1)
    pmn = torch.matmul(Dm, U.T)
    cm = torch.where(nm[:, None] > 0, pmn / nm[:, None].clamp_min(1e-30), torch.zeros_like(pmn))
    out = dict(cos=cos.mean(0), proj=proj.mean(0), norm=nrm.mean(0), cos_of_mean=cm, proj_of_mean=pmn, norm_of_mean=nm,
               lens=Z[idx].mean(0), lens_change=(Z[idx] - Z0[idx]).mean(0), h0_norm=H0[idx].norm(dim=-1).mean(0))
    if per_prompt:
        out["cos_dolphin_per_prompt"] = cos[:, :, 0]
    exact_zero = bool((D == 0).all())
    return out, Dm, exact_zero


def measure_run(run, steps, cache, max_prompts, out_dir, force=False):
    path_json = os.path.join(out_dir, f"{run}.json")
    if not force and os.path.exists(path_json):
        with open(path_json) as f:
            old = json.load(f)
        if old.get("steps") == steps and old.get("max_prompts") == max_prompts:
            log(f"{run}: complete result exists, skipped ({path_json})")
            return old
    t_run = time.time()
    rd = os.path.join(RUNS_DIR, run)
    with open(os.path.join(rd, "RUN.json")) as f:
        model_id = json.load(f)["settings"]["model"]
    adir = {s: os.path.join(rd, "adapters", f"step_{s:06d}") for s in steps if s > 0}
    missing = [s for s, d in adir.items() if not os.path.exists(os.path.join(d, "adapter_model.safetensors"))]
    if missing:
        raise FileNotFoundError(f"{run}: adapters missing at steps {missing}")
    first_adapter = os.path.join(rd, "adapters", f"step_{ADAPTER_STEPS[0]:06d}")
    pm, tok, fresh = cache.get(model_id, first_adapter)
    base = pm.get_base_model()
    if fresh or cache.info is None:
        tinfo = token_info(tok)
        raw, unit, dinfo = token_directions(base, tinfo)
        cos_dd = (unit @ unit[0]).tolist()
        cache.info = dict(tinfo=tinfo, raw=raw, unit=unit, dinfo=dinfo, ref=compare_reference(unit, tinfo),
                          cos_with_dolphin={t["string"]: float(c) for t, c in zip(tinfo, cos_dd)})
    tinfo, raw, unit, eps = cache.info["tinfo"], cache.info["raw"], cache.info["unit"], cache.info["dinfo"]["eps"]
    animal, neutral = prompt_set(max_prompts)
    inp, att, last_idx, lengths, last_piece = encode(tok, animal + neutral)
    idx = dict(animal_last=torch.arange(len(animal)), animal_mean=torch.arange(len(animal)),
               neutral_last=torch.arange(len(animal), len(animal) + len(neutral)))
    names = lora_names(pm)
    checks = []

    # reference forward at checkpoint 0 (all LoRA B zero), compared bitwise with the adapter switched off
    zero_lora_b(pm)
    t0 = time.time()
    ref = record(pm, inp, att, last_idx, raw, eps)
    t_fwd0 = time.time() - t0
    off = record(pm, inp, att, last_idx, raw, eps, disable=True)
    same = all(bool(torch.equal(a, b)) for a, b in zip(ref, off))
    checks.append(dict(name=f"{LABELS.get(run, run)}: checkpoint 0 (LoRA B zero) equals the base model bitwise", passed=same))
    llc = model_logit_check(pm, inp, att, last_idx, tinfo, ref[2][:, -1])
    checks.append(dict(name=f"{LABELS.get(run, run)}: last-layer logit lens equals the model's own logits at step 0",
                       passed=llc["passed"], detail=llc))
    del off
    H0 = dict(animal_last=ref[0], animal_mean=ref[1], neutral_last=ref[0])
    Z0 = dict(animal_last=ref[2], animal_mean=ref[3], neutral_last=ref[2])

    res = {g: {} for g in GROUPS}
    Dkeep = {"animal_last": [], "neutral_last": []}
    timing, coverage, zero_ok = [], [], None
    for s in steps:
        t0 = time.time()
        if s == 0:
            zero_lora_b(pm)
        else:
            coverage.append(dict(step=s, **load_adapter(pm, os.path.join(adir[s], "adapter_model.safetensors"), names)))
        t1 = time.time()
        rec = record(pm, inp, att, last_idx, raw, eps)
        torch.cuda.synchronize()
        t2 = time.time()
        Hs = dict(animal_last=rec[0], animal_mean=rec[1], neutral_last=rec[0])
        Zs = dict(animal_last=rec[2], animal_mean=rec[3], neutral_last=rec[2])
        for g in GROUPS:
            m, Dm, ez = group_measures(Hs[g], H0[g], Zs[g], Z0[g], idx[g], unit, per_prompt=(g == "animal_last"))
            for k, v in m.items():
                res[g].setdefault(k, []).append(v.numpy())
            if g in Dkeep:
                Dkeep[g].append(Dm.numpy().astype(np.float16))
            if s == 0:
                zero_ok = ez if zero_ok is None else (zero_ok and ez)
                zero_ok = zero_ok and bool((m["proj"] == 0).all()) and bool((m["proj_of_mean"] == 0).all())
        t3 = time.time()
        timing.append(dict(step=s, load_s=round(t1 - t0, 3), forward_s=round(t2 - t1, 3), measure_s=round(t3 - t2, 3)))
        if s in (steps[0], steps[-1]) or s % 5000 == 0:
            d = res["animal_last"]
            log(f"{run} step {s}: load {t1 - t0:.2f} s, forward {t2 - t1:.2f} s, measures {t3 - t2:.2f} s; animal last token, "
                f"layer 39: cos(D, dolphin) {d['cos'][-1][-1, 0]:+.4f}, |D| {d['norm'][-1][-1]:.3f}, "
                f"lens dolphin {d['lens'][-1][-1, 0]:+.3f}")
    if 0 in steps:
        checks.append(dict(name=f"{LABELS.get(run, run)}: D and its projections are exactly zero at checkpoint 0 "
                                f"(second forward of the base)", passed=bool(zero_ok)))
    n_cov = sum(c["covered"] for c in coverage)
    checks.append(dict(name=f"{LABELS.get(run, run)}: every LoRA parameter covered by each adapter",
                       passed=n_cov == len(coverage), detail=f"{n_cov}/{len(coverage)} adapters cover all {len(names)} "
                                                          f"LoRA parameters"))
    out = dict(run=run, label=LABELS.get(run, run), model=model_id, steps=steps, max_prompts=max_prompts,
               layers=list(range(H0["animal_last"].shape[1])), tokens=tinfo,
               lora_config=os.path.join(first_adapter, "adapter_config.json"), n_lora_params=len(names),
               prompts=dict(animal=animal, neutral=neutral, token_lengths=lengths, last_token_piece=last_piece),
               groups={g: {k: sig(np.stack(v)) for k, v in res[g].items()} for g in GROUPS},
               layout=dict(cos="[checkpoint][layer][token]", proj="[checkpoint][layer][token]", norm="[checkpoint][layer]",
                           cos_of_mean="[checkpoint][layer][token]", proj_of_mean="[checkpoint][layer][token]",
                           norm_of_mean="[checkpoint][layer]", lens="[checkpoint][layer][token]",
                           lens_change="[checkpoint][layer][token]", h0_norm="[checkpoint][layer]",
                           cos_dolphin_per_prompt="[checkpoint][animal prompt][layer] (animal_last only)"),
               checks=checks, coverage=coverage, timing=timing, step0_forward_s=round(t_fwd0, 3),
               seconds=round(time.time() - t_run, 1))
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f"{run}_D.tmp.npz")
    np.savez(tmp, steps=np.asarray(steps), animal_last=np.stack(Dkeep["animal_last"]),
             neutral_last=np.stack(Dkeep["neutral_last"]),
             sign="D = h(step) - h(step 0), prompt-averaged, decoder layer outputs, last prompt token")
    os.replace(tmp, os.path.join(out_dir, f"{run}_D.npz"))
    atomic_json(out, path_json)
    log(f"{run}: wrote {path_json} ({out['seconds']} s)")
    return out


# ------------------------------------------------------------------ judgment and summary

def judgment(runs, group="animal_last", measure="cos", token=0):
    """layers where min over dolphin runs > max over clean runs at step 2,500 and every later checkpoint; per layer the
    first checkpoint from which the ordering holds at every later checkpoint"""
    dol = [r for r in RUNS[:2] if r in runs]
    cln = [r for r in RUNS[2:] if r in runs]
    if len(dol) < 2 or len(cln) < 2:
        return dict(available=False, reason="needs both dolphin and both clean runs")
    steps = runs[dol[0]]["steps"]
    if any(runs[r]["steps"] != steps for r in dol + cln):
        return dict(available=False, reason="runs have different checkpoints")
    def arr(r):
        a = np.asarray(runs[r]["groups"][group][measure])
        return a[..., token] if a.ndim == 3 else a
    A = np.stack([arr(r) for r in dol])  # [2, S, L]
    C = np.stack([arr(r) for r in cln])
    ok = A.min(0) > C.max(0)  # [S, L]
    st = np.asarray(steps)
    post = st >= ONSET
    if not post.any():
        return dict(available=False, reason="no checkpoint at or after step 2,500")
    layers_hold = [int(L) for L in range(ok.shape[1]) if ok[post, L].all()]
    first_stay = []
    for L in range(ok.shape[1]):
        f = None
        for i in range(len(st) - 1, -1, -1):
            if ok[i, L]:
                f = int(st[i])
            else:
                break
        first_stay.append(f)
    fs = [x for x in first_stay if x is not None]
    return dict(available=True, group=group, measure=measure, token=TOKENS[token], rule="min(dolphin seed 1, dolphin seed 2) > "
                "max(clean seed 1, clean seed 2) at step 2,500 and every later checkpoint",
                layers_hold_from_2500=layers_hold, first_checkpoint_holding_to_the_end_per_layer=first_stay,
                earliest_first_checkpoint=min(fs) if fs else None,
                layers_with_earliest=[L for L, x in enumerate(first_stay) if fs and x == min(fs)])


def summarize(out_dir, summary_dir, run_names, require_full=True):
    runs = {}
    for r in run_names:
        p = os.path.join(out_dir, f"{r}.json")
        if os.path.exists(p):
            with open(p) as f:
                runs[r] = json.load(f)
    if not runs:
        raise SystemExit(f"no per-run results in {out_dir}")
    any_run = next(iter(runs.values()))
    checks = []
    for r in runs.values():
        checks += r["checks"]
    lens_ok = all(len(x) for x in [any_run["prompts"]["token_lengths"]]) and max(any_run["prompts"]["token_lengths"]) < MAX_PROMPT_TOKENS
    checks.append(dict(name=f"every prompt is shorter than {MAX_PROMPT_TOKENS} tokens", passed=bool(lens_ok),
                       detail=f"longest {max(any_run['prompts']['token_lengths'])}"))
    exact = [t["string"] for t in any_run["tokens"] if "".join(t["pieces"]) == t["string"]]
    multi = {t["string"]: t["pieces"] for t in any_run["tokens"] if len(t["ids"]) > 1}
    checks.append(dict(name="every animal string's token ids decode back to the string (the direction uses the first token)",
                       passed=len(exact) == len(TOKENS) and any_run["tokens"][0]["pieces"] == [TOKENS[0]],
                       detail=dict(exact=f"{len(exact)}/{len(TOKENS)}", multi_token=multi)))
    full_steps = [0] + ADAPTER_STEPS
    for r in runs.values():
        checks.append(dict(name=f"{r['label']}: all 41 checkpoints (step 0 and 40 adapters) measured",
                           passed=r["steps"] == full_steps, detail=f"{len(r['steps'])} checkpoints"))
    learned = None
    if "full_dolphin_s1" in runs and runs["full_dolphin_s1"]["steps"][0] == 0 and len(runs["full_dolphin_s1"]["steps"]) > 1:
        lens = np.asarray(runs["full_dolphin_s1"]["groups"]["animal_last"]["lens"])  # [S, L, K]
        learned = dict(step=runs["full_dolphin_s1"]["steps"][-1], lens_final_layer_last=float(lens[-1, -1, 0]),
                       lens_final_layer_step0=float(lens[0, -1, 0]))
        checks.append(dict(name="dolphin seed 1: last-layer logit-lens dolphin logit (animal questions, last token) is higher "
                                f"at step {learned['step']} than at step 0",
                           passed=learned["lens_final_layer_last"] > learned["lens_final_layer_step0"], detail=learned))
    cache_info = any_run.get("directions", {})
    summary = dict(
        description="Actual residual-stream movement D_L = h_L(step) - h_L(step 0) of the four students at each saved "
                    "checkpoint, per layer, projected on the logit-lens token directions; and the logit-lens logit itself.",
        sign="+dh: D = h(step) - h(step 0), the actual change of the decoder-layer output (no gradient sign involved)",
        direction_formula="d_tok = gamma * W_U[tok] - mean_v(gamma * W_U[v]), unit-normalised (gamma = model.norm.weight, "
                          "W_U = lm_head.weight), fp32 on CPU",
        logit_lens_formula="lens_tok(h) = rsqrt(mean(h^2) + eps) * <h, gamma * (W_U[tok] - mean_v W_U[v])> = final norm and "
                           "lm_head applied to h, token logit minus mean logit over the vocabulary",
        groups=dict(animal_last="the student's 50 favourite-animal eval questions (subliminal_eval.ANIMAL_PROMPTS), last "
                                "prompt token (where the answer begins)",
                    animal_mean="the same questions, mean over all prompt tokens (logit lens: mean of the per-token lens)",
                    neutral_last="4 neutral prompts, last prompt token"),
        averaging="each measure is computed per prompt and averaged over the group's prompts; *_of_mean are the measures "
                  "of the prompt-averaged D",
        tokens=any_run["tokens"], directions=cache_info, prompts=any_run["prompts"],
        runs={r: dict(label=v["label"], steps=v["steps"], layers=v["layers"], groups=v["groups"], layout=v["layout"],
                      seconds=v["seconds"], timing=v["timing"]) for r, v in runs.items()},
        judgment=dict(primary=judgment(runs),
                      cos_of_prompt_mean=judgment(runs, measure="cos_of_mean"),
                      logit_lens_change=judgment(runs, measure="lens_change"),
                      neutral_cos=judgment(runs, group="neutral_last")),
        integrity=dict(passed=sum(bool(c["passed"]) for c in checks), total=len(checks), checks=checks),
        written=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    os.makedirs(summary_dir, exist_ok=True)
    pj = os.path.join(summary_dir, f"{SUMMARY_NAME}.json")
    atomic_json(summary, pj)
    log(f"wrote {pj}; integrity {summary['integrity']['passed']}/{summary['integrity']['total']}")
    for c in checks:
        if not c["passed"]:
            log("FAILED CHECK:", c["name"], c.get("detail", ""))
    plot_heatmaps(runs, os.path.join(summary_dir, f"{SUMMARY_NAME}.png"))
    plot_bands(runs, os.path.join(summary_dir, f"{SUMMARY_NAME}_bands.png"))
    return summary


def _kfmt():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k")


def _edges(steps):
    s = np.asarray(steps, dtype=float)
    mid = (s[1:] + s[:-1]) / 2 if len(s) > 1 else np.array([])
    lo = s[0] - (mid[0] - s[0] if len(mid) else 250)
    hi = s[-1] + (s[-1] - mid[-1] if len(mid) else 250)
    return np.concatenate([[lo], mid, [hi]])


def plot_heatmaps(runs, out):
    from plot_e2_trained_readers import plt  # also sets the shared rcParams
    from matplotlib.colors import TwoSlopeNorm

    cols = [("animal_last", "cos", "cosine of D with the dolphin direction\nanimal questions, last token"),
            ("animal_last", "lens_change", "change in logit-lens dolphin logit since step 0\nanimal questions, last token"),
            ("neutral_last", "cos", "cosine of D with the dolphin direction\nneutral prompts, last token")]
    order = [r for r in RUNS if r in runs]
    fig, axes = plt.subplots(len(order), len(cols), figsize=(8.6 * len(cols), 4.6 * len(order) + 1.6), dpi=130,
                             squeeze=False, constrained_layout=True)
    for j, (g, m, title) in enumerate(cols):
        mats = {r: np.asarray(runs[r]["groups"][g][m]) for r in order}
        mats = {r: (a[..., 0] if a.ndim == 3 else a) for r, a in mats.items()}  # [S, L]
        vmax = max(float(np.nanmax(np.abs(a))) for a in mats.values()) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for i, r in enumerate(order):
            ax = axes[i, j]
            a = mats[r]
            nl = a.shape[1]
            im = ax.pcolormesh(_edges(runs[r]["steps"]), np.arange(nl + 1) - 0.5, a.T, cmap="RdBu_r", norm=norm,
                               shading="flat", rasterized=True)
            ax.axvline(ONSET, color="black", ls=":", lw=2.0)
            ax.set_xlim(_edges(runs[r]["steps"])[0], _edges(runs[r]["steps"])[-1])
            ax.set_ylim(-0.5, nl - 0.5)
            ax.set_yticks([0, 10, 20, 30, 39] if nl == 40 else None)
            ax.xaxis.set_major_formatter(_kfmt())
            if i == 0:
                ax.set_title(title, fontsize=19)
            if j == 0:
                ax.set_ylabel(f"{runs[r]['label']}\nlayer")
            if i == len(order) - 1:
                ax.set_xlabel("student optimizer step")
        cb = fig.colorbar(im, ax=axes[:, j], location="bottom", shrink=0.85, pad=0.02, aspect=40)
        cb.ax.tick_params(labelsize=15)
    fig.suptitle("Actual residual movement D = h(step) - h(step 0) per layer; dotted line: student behaviour onset (2,500)",
                 fontsize=20)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    log("wrote", out)


def plot_bands(runs, out):
    from plot_e2_trained_readers import behaviour, plt  # also sets the shared rcParams
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = [("animal_last", "cos", "cosine of D with\nthe dolphin direction"),
            ("animal_last", "lens_change", "change in logit-lens\ndolphin logit"),
            ("neutral_last", "cos", "cosine of D with\nthe dolphin direction")]
    rtitle = ["animal questions, last token", "animal questions, last token", "neutral prompts, last token"]
    order = [r for r in RUNS if r in runs]
    fig, axes = plt.subplots(len(rows), len(BANDS), figsize=(8.0 * len(BANDS), 5.6 * len(rows)), dpi=130, squeeze=False)
    for i, (g, m, ylabel) in enumerate(rows):
        for j, (lo, hi) in enumerate(BANDS):
            ax = axes[i, j]
            ax.axvspan(*WINDOW, color="#fde0c5", alpha=0.6, lw=0, zorder=0)
            for r in order:
                a = np.asarray(runs[r]["groups"][g][m])
                a = a[..., 0] if a.ndim == 3 else a
                hi_ = min(hi, a.shape[1] - 1)
                color, ls = STYLE[r]
                ax.plot(runs[r]["steps"], a[:, lo:hi_ + 1].mean(1), color=color, ls=ls, lw=2.6, label=runs[r]["label"], zorder=3)
            ax.axhline(0, color="black", lw=0.8, alpha=0.5)
            ax.axvline(ONSET, color="black", ls=":", lw=1.8, zorder=2)
            ax2 = ax.twinx()
            for r, ls in (("full_dolphin_s1", "-"), ("full_dolphin_s2", "--")):
                try:
                    bx, by = behaviour(r)
                except (OSError, KeyError):
                    continue
                ax2.plot(bx, by, color="black", ls=ls, lw=2.0, alpha=0.8, zorder=1)
            ax2.set_ylim(0, 1)
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax2.yaxis.set_major_formatter(_pct())
            ax2.tick_params(axis="y", labelsize=15)
            if j == len(BANDS) - 1:
                ax2.set_ylabel("student's dolphin rate", fontsize=17)
            ax.set_xlim(0, 20000)
            ax.set_xticks([0, 5000, 10000, 15000, 20000])
            ax.xaxis.set_major_formatter(_kfmt())
            ax.grid(alpha=0.3)
            ax.set_title(f"layers {lo}–{hi}, {rtitle[i]}", fontsize=18)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(rows) - 1:
                ax.set_xlabel("student optimizer step")
    handles = [Line2D([], [], color=STYLE[r][0], ls=STYLE[r][1], lw=2.6, label=LABELS[r]) for r in order]
    handles += [Line2D([], [], color="black", lw=2.0, label="student's dolphin rate, dolphin seed 1 (right axis)"),
                Line2D([], [], color="black", lw=2.0, ls="--", label="student's dolphin rate, dolphin seed 2 (right axis)"),
                Line2D([], [], color="black", ls=":", lw=1.8, label="student behaviour onset (2,500)"),
                Patch(color="#fde0c5", alpha=0.6, label="steps 1,001–3,100")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=15, handlelength=2.2,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Layer-band average of the actual residual movement D = h(step) - h(step 0), per student run", fontsize=21)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    log("wrote", out)


def _pct():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"{100 * v:g}%")


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--steps", default="all", help="'all' (0 and 500..20000) or a comma list; 0 is always measured first")
    ap.add_argument("--max-prompts", type=int, default=None, help="use only the first N animal and first N neutral prompts")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--summary-dir", default=SUMMARY_DIR)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute runs whose per-run result exists")
    a = ap.parse_args(argv)
    run_names = [r for r in a.runs.split(",") if r]
    steps = [0] + ADAPTER_STEPS if a.steps == "all" else sorted({0} | {int(s) for s in a.steps.split(",")})
    if not a.summarize_only:
        from agm import core

        core.set_fp32_matmul_ieee()
        torch.set_grad_enabled(False)
        cache = ModelCache(a.device)
        for r in run_names:
            out = measure_run(r, steps, cache, a.max_prompts, a.out_dir, force=a.force)
            if cache.info is not None:  # directions and their reference comparison go into the per-run json too
                out["directions"] = dict(cos_with_dolphin=cache.info["cos_with_dolphin"], **cache.info["dinfo"],
                                         reference=cache.info["ref"])
                atomic_json(out, os.path.join(a.out_dir, f"{r}.json"))
    summarize(a.out_dir, a.summary_dir, run_names)


if __name__ == "__main__":
    main()

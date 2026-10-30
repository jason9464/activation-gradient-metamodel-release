#!/usr/bin/env python
"""Fit one bias per layer to subliminal number data on the frozen base model, then read it with no label.

Stages (everything under --out; a finished output is never overwritten; GPU unless marked CPU):
  lrpilot --opt adam|sgd  PILOT_STEPS-step fits on dolphin rows 0..PILOT_STEPS*32-1 at every rate of LR_GRID[opt];
                          lrpilot_<opt>.json with the curves and the chosen rate (lowest mean training loss over steps
                          51-100 among runs whose loss stayed finite).
  fit --arm A --opt O     STEPS steps on rows 0..STEPS*32-1 of arm A at the pilot's rate. fit_<A>_<O>.npz (biases
                          [n_saves, n_layers, d] fp32 at SAVE_STEPS) and fit_<A>_<O>.json (loss and norms per step,
                          held-out loss of every arm at step 0 and at each save).
  actdiff                 actdiff.npz: per-arm mean residuals (rows 0..1,023, student condition) and the dolphin teacher
                          direction (rows 0..255 with the dolphin system prompt minus the neutral one).
  vectors (CPU)           vectors.npz: every write stack the reads use (plan section 2).
  steer                   steer.jsonl: logprob probe and 50 questions x 20 samples per (vector, alpha) and for the
                          unsteered base; resumes cell by cell.
  ao                      ao.jsonl: aopaper and ao3layer (ao_diff_read), questions preference and describe, 100 answers
                          per vector; resumes per (vector, AO, question).
  pack (CPU)              reader_inputs.npz in e2_read's export layout, for the em-reader e2_generate
                          (``--emwt`` also loads it through that module's load_inputs as a check).
  summary (CPU)           summary.json and summary.md: counts and the plan's decision rule.

Biases: b_L (fp32) is added to the output of decoder layer L at every position by a forward hook (LayerBiases); the
base weights are frozen. Tokenisation, labels and loss are the students' (train_subliminal.tokenize_rows with public
think labels, CompletionMaskCollator, token-mean NLL over the batch's label tokens). A write stack W [n_layers, d] is
read the way the readers read the direct-write term: W_L is what layer L writes into the residual stream.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = "Qwen/Qwen3-14B"
DATA = {
    "dolphin": dict(repo="cds-jb/qwen3-14b-dolphin-subliminal-nums-25x", repo_type="dataset",
                    revision="cf880b73887d47147452c833685d4854d6e828b9", file="filtered_full.jsonl"),
    "owl": dict(repo="cds-jb/qwen3-14b-owl-subliminal-nums-25x", repo_type="dataset",
                revision="e534144a8c2a9c6637d129fa4fc210586eb9f164", file="filtered_full.jsonl"),
    "clean": dict(repo="jason9464/sprint-data-2610", repo_type="model",
                  revision="2f6b53cd21a3b3bd9b47447713f4ec626fc49f6a",
                  file="vectors/signal_check/clean_nums/filtered_full.jsonl"),
}
ARMS = ("dolphin", "owl", "clean")
MAX_LEN, THINK = 256, "public"
BATCH = 32
STEPS = 1000
WARMUP = 50
SAVE_STEPS = (100, 250, 500, 1000)
HELDOUT = (40000, 40256)
ACT_ROWS = (0, 1024)
TEACHER_ROWS = (0, 256)
PILOT_STEPS = 100
PILOT_WINDOW = (51, 100)
LR_GRID = {"adam": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2), "sgd": (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)}
ADAM_BETAS, ADAM_EPS = (0.9, 0.999), 1e-8
FITS = [("dolphin", "adam"), ("owl", "adam"), ("clean", "adam"), ("dolphin", "sgd"), ("clean", "sgd")]
ALPHAS = (1, 2, 4, 8)
STEER_CFG = dict(n_prompts=50, samples_per_prompt=20, max_new_tokens=16, temperature=1.0, seed=2000, negative=False)
STEER_ANIMALS = ("butterfly", "whale")  # measured next to owl and dolphin
STEER_VECTORS = ("fit_dolphin_adam", "fit_owl_adam", "fit_clean_adam", "fit_dolphin_sgd", "fit_clean_sgd",
                 "act_dolphin_minus_clean", "act_owl_minus_clean", "teacher_dolphin")
READ_VECTORS = STEER_VECTORS + ("fit_dolphin_minus_clean_adam", "fit_owl_minus_clean_adam",
                                "fit_dolphin_minus_clean_sgd")
AO_SAMPLES = 100
AO_QUESTIONS = ("preference", "describe")
READER_NAME = "rl_paper_c040"
STEER_N = STEER_CFG["n_prompts"] * STEER_CFG["samples_per_prompt"]
# plan section 4
STEER_RULES = [("S1", "fit_dolphin_adam", ("fit_clean_adam", "fit_owl_adam"), "dolphin"),
               ("S2", "fit_owl_adam", ("fit_clean_adam", "fit_dolphin_adam"), "owl"),
               ("S3", "fit_dolphin_sgd", ("fit_clean_sgd",), "dolphin"),
               ("S4", "act_dolphin_minus_clean", ("act_owl_minus_clean",), "dolphin"),
               ("S0", "teacher_dolphin", ("base",), "dolphin")]
STEER_MIN_HITS, STEER_P = 20, 0.05 / len(ALPHAS)
READ_PAIRS = [("fit_dolphin_adam", "fit_clean_adam"), ("fit_dolphin_sgd", "fit_clean_sgd"),
              ("fit_dolphin_minus_clean_adam", "fit_owl_minus_clean_adam"),
              ("act_dolphin_minus_clean", "act_owl_minus_clean"), ("teacher_dolphin", "fit_clean_adam")]
READ_MIN, READ_RATIO = 5, 5


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [fit_vector] {msg}", flush=True)


def atomic_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    os.replace(tmp, path)


def refuse(path):
    if Path(path).exists():
        raise SystemExit(f"{path} exists; not overwritten")


# ---------------------------------------------------------------- data


def data_path(arm):
    """FV_DATA_<ARM> when set (tests, local copies), else the pinned HF file."""
    env = os.environ.get(f"FV_DATA_{arm.upper()}")
    if env:
        return env
    from huggingface_hub import hf_hub_download

    s = DATA[arm]
    return hf_hub_download(s["repo"], s["file"], repo_type=s["repo_type"], revision=s["revision"])


def read_slice(arm, lo, hi):
    """prompt/completion of lines lo..hi-1 of the arm's file (the system prompt is dropped: student condition)."""
    rows = []
    with open(data_path(arm)) as f:
        for line in itertools.islice(f, lo, hi):
            r = json.loads(line)
            rows.append(dict(prompt=r["prompt"], completion=r["completion"]))
    if len(rows) != hi - lo:
        raise SystemExit(f"{arm}: lines {lo}..{hi - 1} requested, {len(rows)} read")
    return rows


def rows_sha1(rows):
    return hashlib.sha1("\n".join(json.dumps([r["prompt"], r["completion"]]) for r in rows).encode()).hexdigest()


def examples(tok, rows):
    from train_subliminal import tokenize_rows

    ids, masks, stats = tokenize_rows(tok, rows, MAX_LEN, THINK)
    return [dict(input_ids=i, completion_mask=m) for i, m in zip(ids, masks)], stats


# ---------------------------------------------------------------- model and biases


def load_model(device, model_id=MODEL, dtype="bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), attn_implementation="sdpa",
                                                 device_map={"": device})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def dims(model):
    from agm import core

    return len(core.decoder_layers(model)), int(model.config.hidden_size)


class LayerBiases:
    """B[L] (fp32, [n_layers, d]) times ``scale`` added to the output of decoder layer L at every position."""

    def __init__(self, model, values=None, trainable=False, scale=1.0):
        import torch
        from agm import core

        layers = core.decoder_layers(model)
        dev = next(model.parameters()).device
        n, d = dims(model)
        B = torch.zeros(n, d, dtype=torch.float32, device=dev) if values is None else \
            torch.as_tensor(np.asarray(values), dtype=torch.float32, device=dev).clone()
        if tuple(B.shape) != (n, d):
            raise ValueError(f"biases {tuple(B.shape)} for a model with {n} layers x {d}")
        self.B = torch.nn.Parameter(B, requires_grad=trainable)
        self.scale = float(scale)
        self.handles = [layer.register_forward_hook(self._hook(k)) for k, layer in enumerate(layers)]

    def _hook(self, k):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + (self.scale * self.B[k]).to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def to_device(batch, device):
    return {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")}


def loss_sum_count(model, batch):
    """(summed NLL over the label positions as a 0-dim fp32 tensor, number of label positions)."""
    import torch

    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    lab = batch["labels"][:, 1:]
    lg = logits[:, :-1].float()
    s = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100,
                                          reduction="sum")
    return s, int((lab != -100).sum())


def mean_loss(model, tok, exs, device, batch=None):
    """Token-mean NLL over all label positions of ``exs`` (no gradient)."""
    import torch
    from train_subliminal import CompletionMaskCollator

    batch = BATCH if batch is None else batch
    col = CompletionMaskCollator(tok)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(exs), batch):
            s, n = loss_sum_count(model, to_device(col(exs[i:i + batch]), device))
            tot, cnt = tot + float(s), cnt + n
    return tot / cnt


def run_fit(model, tok, exs, opt, lr, steps, device, batch=None, warmup=None, save_steps=(), heldout=None):
    """Fit fresh zero biases for ``steps`` steps on exs[0 : steps*batch] in order. Returns (record, {step: biases},
    {step: {arm: held-out loss}})."""
    import torch
    from train_subliminal import CompletionMaskCollator

    batch = BATCH if batch is None else batch
    warmup = WARMUP if warmup is None else warmup
    if len(exs) < steps * batch:
        raise ValueError(f"{len(exs)} examples for {steps} steps of {batch}")
    col = CompletionMaskCollator(tok)
    rec = dict(loss=[], lr_per_step=[], grad_norm=[], bias_norm=[])
    saves, held = {}, {}
    with LayerBiases(model, trainable=True) as lb:
        if opt == "adam":
            o = torch.optim.Adam([lb.B], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0)
        elif opt == "sgd":
            o = torch.optim.SGD([lb.B], lr=lr, momentum=0.0)
        else:
            raise ValueError(opt)
        sched = torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / warmup))
        if heldout is not None:
            held[0] = {a: mean_loss(model, tok, e, device, batch) for a, e in heldout.items()}
        for step in range(1, steps + 1):
            b = to_device(col(exs[(step - 1) * batch: step * batch]), device)
            s, n = loss_sum_count(model, b)
            loss = s / n
            o.zero_grad(set_to_none=True)
            loss.backward()
            rec["lr_per_step"].append(o.param_groups[0]["lr"])
            rec["grad_norm"].append(float(lb.B.grad.norm()))
            o.step()
            sched.step()
            lv = float(loss.detach())
            rec["loss"].append(lv)
            rec["bias_norm"].append(float(lb.B.detach().norm()))
            if not math.isfinite(lv) or not bool(torch.isfinite(lb.B).all()):
                rec["diverged_at"] = step
                break
            if step in save_steps:
                saves[step] = lb.B.detach().float().cpu().numpy().copy()
                if heldout is not None:
                    held[step] = {a: mean_loss(model, tok, e, device, batch) for a, e in heldout.items()}
    return rec, saves, held


def pilot_choice(runs):
    ok = [r for r in runs if r["finite"]]
    if not ok:
        raise SystemExit("every pilot rate diverged")
    return min(ok, key=lambda r: r["window_mean"])["lr"]


# ---------------------------------------------------------------- activation side


def mean_residuals(model, seqs, starts, device, batch=None, pad_id=0):
    """[n_layers, d] fp64: mean over sequences of the token mean of each decoder layer's output over positions
    starts[i]+1 .. len(seqs[i])-1 (the document's tokens after its first), and [n_layers] the mean token norm."""
    import torch
    from agm import core

    batch = BATCH if batch is None else batch
    n_layers, d = dims(model)
    acc = torch.zeros(n_layers, d, dtype=torch.float64, device=device)
    nrm = torch.zeros(n_layers, dtype=torch.float64, device=device)
    W = {}

    def hook(k):
        def f(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            acc[k] += torch.einsum("bt,btd->d", W["w"], h).double()
            nrm[k] += (W["w"] * h.norm(dim=-1)).sum().double()
        return f

    hs = [layer.register_forward_hook(hook(k)) for k, layer in enumerate(core.decoder_layers(model))]
    try:
        with torch.no_grad():
            for i in range(0, len(seqs), batch):
                chunk, st = seqs[i:i + batch], starts[i:i + batch]
                T = max(len(s) for s in chunk)
                ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
                att = torch.zeros((len(chunk), T), dtype=torch.long)
                w = torch.zeros((len(chunk), T), dtype=torch.float32)
                for r, (s, s0) in enumerate(zip(chunk, st)):
                    ids[r, :len(s)] = torch.tensor(s)
                    att[r, :len(s)] = 1
                    m = len(s) - s0 - 1
                    if m < 1:
                        raise ValueError(f"sequence {i + r}: no document token after the first")
                    w[r, s0 + 1:len(s)] = 1.0 / m
                W["w"] = w.to(device)
                model(input_ids=ids.to(device), attention_mask=att.to(device))
    finally:
        for h in hs:
            h.remove()
    n = len(seqs)
    return (acc / n).cpu().numpy(), (nrm / n).cpu().numpy()


def teacher_seqs(tok, rows, system):
    """(ids, doc start) per row: the system block of ``system`` followed by the student-condition conversation, and
    the position where the student ids begin (checked: the last n ids are the student ids)."""
    from agm.data import render_chat

    block = tok.apply_chat_template([dict(role="system", content=system)], tokenize=False)
    if not block.startswith("<|im_start|>system\n") or not block.endswith("<|im_end|>\n"):
        raise SystemExit(f"unexpected system block rendering: {block!r}")
    out = []
    for r in rows:
        full = render_chat(tok, [dict(role="user", content=r["prompt"])], r["completion"])[1]
        student = tok(full, add_special_tokens=False)["input_ids"]
        ids = tok(block + full, add_special_tokens=False)["input_ids"]
        if ids[-len(student):] != student:
            raise SystemExit("system-prompt rendering does not end with the student ids")
        out.append((ids, len(ids) - len(student)))
    return out, len(tok(block, add_special_tokens=False)["input_ids"])


def writes(X):
    """Cumulative per-layer differences -> per-layer writes (w_0 = X_0, w_L = X_L - X_{L-1})."""
    W = np.array(X, dtype=np.float64, copy=True)
    W[1:] = X[1:] - X[:-1]
    return W


# ---------------------------------------------------------------- stages


def out_dir(a):
    p = Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_lrpilot(a, model=None, tok=None):
    out = out_dir(a)
    path = out / f"lrpilot_{a.opt}.json"
    refuse(path)
    if model is None:
        model, tok = load_model(a.device)
    steps = a.pilot_steps
    rows = read_slice("dolphin", 0, steps * BATCH)
    exs, stats = examples(tok, rows)
    w0, w1 = (PILOT_WINDOW if steps == PILOT_STEPS else (steps // 2 + 1, steps))
    runs = []
    for lr in LR_GRID[a.opt]:
        t0 = time.time()
        rec, _, _ = run_fit(model, tok, exs, a.opt, lr, steps, a.device)
        win = rec["loss"][w0 - 1:w1]
        finite = "diverged_at" not in rec and len(win) == w1 - w0 + 1 and all(math.isfinite(x) for x in win)
        runs.append(dict(lr=lr, finite=finite, window_mean=float(np.mean(win)) if finite else None,
                         seconds=round(time.time() - t0, 1), **rec))
        log(f"pilot {a.opt} lr {lr:g}: loss {rec['loss'][0]:.4f} -> {rec['loss'][-1]:.4f}, window mean "
            f"{runs[-1]['window_mean']}, {runs[-1]['seconds']}s")
    chosen = pilot_choice(runs)
    atomic_json(dict(opt=a.opt, grid=list(LR_GRID[a.opt]), steps=steps, window=[w0, w1], chosen_lr=chosen,
                     rule="lowest mean training loss over the window among runs whose loss stayed finite",
                     data=dict(arm="dolphin", rows=[0, steps * BATCH], rows_sha1=rows_sha1(rows), tokenise=stats),
                     runs=runs, written=time.strftime("%Y-%m-%d %H:%M:%S")), path)
    log(f"chosen {a.opt} lr {chosen:g} -> {path}")
    return 0


def cmd_fit(a, model=None, tok=None):
    out = out_dir(a)
    stem = out / f"fit_{a.arm}_{a.opt}"
    refuse(stem.with_suffix(".npz"))
    if a.lr is not None:
        lr, lr_src = a.lr, "--lr"
    else:
        pj = out / f"lrpilot_{a.opt}.json"
        if not pj.exists():
            raise SystemExit(f"{pj} missing: run lrpilot --opt {a.opt} first (or pass --lr)")
        lr, lr_src = json.loads(pj.read_text())["chosen_lr"], str(pj)
    if model is None:
        model, tok = load_model(a.device)
    steps = a.steps
    rows = read_slice(a.arm, 0, steps * BATCH)
    exs, stats = examples(tok, rows)
    held = {arm: examples(tok, read_slice(arm, *HELDOUT))[0] for arm in ARMS} if not a.no_heldout else None
    saves_at = tuple(s for s in SAVE_STEPS if s <= steps) or (steps,)
    t0 = time.time()
    rec, saves, heldl = run_fit(model, tok, exs, a.opt, lr, steps, a.device, save_steps=saves_at, heldout=held)
    if "diverged_at" in rec:
        raise SystemExit(f"fit diverged at step {rec['diverged_at']}")
    ss = sorted(saves)
    B = np.stack([saves[s] for s in ss]).astype(np.float32)
    np.savez(stem.with_suffix(".npz"), biases=B, steps=np.array(ss))
    L = rec["loss"]
    atomic_json(dict(arm=a.arm, opt=a.opt, lr=lr, lr_source=lr_src, steps=steps, batch=BATCH, warmup=WARMUP,
                     save_steps=ss, model=MODEL, max_len=MAX_LEN, think_loss=THINK,
                     data=dict(**DATA[a.arm], rows=[0, steps * BATCH], rows_sha1=rows_sha1(rows), tokenise=stats),
                     heldout_rows=list(HELDOUT), heldout_loss={str(k): v for k, v in heldl.items()},
                     bias_norm_per_layer={str(s): np.linalg.norm(saves[s], axis=1).tolist() for s in ss},
                     loss_first10=float(np.mean(L[:10])), loss_last100=float(np.mean(L[-100:])),
                     seconds=round(time.time() - t0, 1), record=rec, written=time.strftime("%Y-%m-%d %H:%M:%S")),
                stem.with_suffix(".json"))
    log(f"fit {a.arm} {a.opt} lr {lr:g}: loss {np.mean(L[:10]):.4f} -> {np.mean(L[-100:]):.4f} in "
        f"{time.time() - t0:.0f}s -> {stem}.npz")
    return 0


def cmd_actdiff(a, model=None, tok=None):
    from steer_vector import NEUTRAL, SYSTEM

    out = out_dir(a)
    path = out / "actdiff.npz"
    refuse(path)
    if model is None:
        model, tok = load_model(a.device)
    pad = tok.pad_token_id
    means, norms, info = {}, {}, {}
    lo, hi = a.act_rows
    for arm in ARMS:
        rows = read_slice(arm, lo, hi)
        exs, stats = examples(tok, rows)
        t0 = time.time()
        means[arm], norms[arm] = mean_residuals(model, [e["input_ids"] for e in exs], [0] * len(exs), a.device,
                                                pad_id=pad)
        info[arm] = dict(rows=[lo, hi], rows_sha1=rows_sha1(rows), truncated=stats["truncated"],
                         seconds=round(time.time() - t0, 1))
        log(f"actdiff {arm}: {len(exs)} documents in {info[arm]['seconds']}s")
    tlo, thi = a.teacher_rows
    rows = read_slice("dolphin", tlo, thi)
    sd, kd = teacher_seqs(tok, rows, SYSTEM)
    sn, kn = teacher_seqs(tok, rows, NEUTRAL)
    if abs(kd - kn) > 2:
        raise SystemExit(f"neutral system block {kn} tokens vs dolphin {kd}")
    md, _ = mean_residuals(model, [s for s, _ in sd], [k for _, k in sd], a.device, pad_id=pad)
    mn, _ = mean_residuals(model, [s for s, _ in sn], [k for _, k in sn], a.device, pad_id=pad)
    info["teacher"] = dict(rows=[tlo, thi], rows_sha1=rows_sha1(rows), system=SYSTEM, neutral=NEUTRAL,
                           block_tokens=dict(dolphin=kd, neutral=kn))
    np.savez(path, **{f"mean_{k}": v.astype(np.float32) for k, v in means.items()},
             **{f"tokennorm_{k}": v.astype(np.float32) for k, v in norms.items()},
             teacher_dolphin_cum=(md - mn).astype(np.float32), meta=json.dumps(info))
    log(f"wrote {path}")
    return 0


def final_biases(out, arm, opt):
    z = np.load(Path(out) / f"fit_{arm}_{opt}.npz")
    return z["biases"][-1].astype(np.float64), int(z["steps"][-1])


def cmd_vectors(a):
    out = out_dir(a)
    path = out / "vectors.npz"
    refuse(path)
    V, src = {}, {}
    for arm, opt in FITS:
        V[f"fit_{arm}_{opt}"], step = final_biases(out, arm, opt)
        src[f"fit_{arm}_{opt}"] = f"fit_{arm}_{opt}.npz step {step}"
    V["fit_dolphin_minus_clean_adam"] = V["fit_dolphin_adam"] - V["fit_clean_adam"]
    V["fit_owl_minus_clean_adam"] = V["fit_owl_adam"] - V["fit_clean_adam"]
    V["fit_dolphin_minus_clean_sgd"] = V["fit_dolphin_sgd"] - V["fit_clean_sgd"]
    z = np.load(out / "actdiff.npz")
    V["act_dolphin_minus_clean"] = writes(z["mean_dolphin"].astype(np.float64) - z["mean_clean"])
    V["act_owl_minus_clean"] = writes(z["mean_owl"].astype(np.float64) - z["mean_clean"])
    V["teacher_dolphin"] = writes(z["teacher_dolphin_cum"].astype(np.float64))
    names = list(READ_VECTORS)
    X = np.stack([V[n] for n in names]).astype(np.float32)
    nrm = np.linalg.norm(X, axis=2)
    if not np.isfinite(X).all() or (nrm == 0).any():
        bad = [(names[i], int(L)) for i, L in zip(*np.nonzero(nrm == 0))]
        raise SystemExit(f"non-finite or zero layer vectors: {bad[:10]}")
    np.savez(path, names=np.array(names), vectors=X, layer_norms=nrm,
             meta=json.dumps(dict(sources=src, writes="w_0 = X_0, w_L = X_L - X_{L-1} for the act and teacher rows")))
    log(f"wrote {path}: {len(names)} vectors x {X.shape[1]} layers")
    return 0


def load_vectors(out):
    z = np.load(Path(out) / "vectors.npz")
    return {str(n): z["vectors"][i] for i, n in enumerate(z["names"])}


def done_keys(path, fields):
    if not Path(path).exists():
        return set()
    with open(path) as f:
        return {tuple(json.loads(line)[k] for k in fields) for line in f if line.strip()}


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def cmd_steer(a, model=None, tok=None, device_override=None):
    import torch
    from subliminal_eval import Evaluator

    out = out_dir(a)
    path = out / "steer.jsonl"
    V = load_vectors(out)
    if model is None:
        model, tok = load_model(a.device)
    gtok = type(tok).from_pretrained(tok.name_or_path) if tok.name_or_path else tok
    gtok.padding_side = "left"
    ev = Evaluator(model, gtok, max_rows=a.max_rows, fast_gen=False, extra_animals=STEER_ANIMALS)
    if device_override is not None:
        ev.device = device_override
    cfg = dict(STEER_CFG, **({"n_prompts": a.steer_prompts, "samples_per_prompt": a.steer_samples}
                             if a.steer_prompts else {}))
    cells = [("base", 0)] + [(v, al) for v in a.vectors for al in a.alphas]
    done = done_keys(path, ("vector", "alpha"))
    for name, al in cells:
        if (name, al) in done:
            continue
        t0 = time.time()
        ctx = contextlib.nullcontext() if name == "base" else LayerBiases(model, values=V[name], scale=al)
        with ctx, torch.no_grad():
            probe = ev.logprob_probe()
            samp = ev.sampled(cfg)
        pos = samp["positive"]
        rec = dict(vector=name, alpha=al, n=pos["total"], hits=pos["hits"], rate=pos["rate"],
                   top_words=pos["top_words"], examples=samp["examples"],
                   probe={an: probe[an]["logprob_any"] for an in ev.animals},
                   probe_top=probe["top_first_tokens"], cfg=cfg, seconds=round(time.time() - t0, 1))
        append_jsonl(path, rec)
        log(f"steer {name} x{al}: dolphin {pos['hits']['dolphin']}/{pos['total']}, owl {pos['hits']['owl']}, "
            f"top {pos['top_words'][:3]} ({rec['seconds']}s)")
    return 0


def cmd_ao(a, loader=None):
    import ao_diff_read as AD

    out = out_dir(a)
    path = out / "ao.jsonl"
    V = load_vectors(out)
    aos = list(AD.AOS)
    pm, tok, cfgs = (loader or AD.load_aos)(a.device, aos)
    done = done_keys(path, ("vector", "ao", "question"))
    for vi, name in enumerate(a.vectors):
        for ai, ao in enumerate(aos):
            pm.set_adapter(ao)
            layers = list(AD.AOS[ao]["layers"])
            vec = np.stack([V[name][L] for L in layers])
            for qi, q in enumerate(AO_QUESTIONS):
                if (name, ao, q) in done:
                    continue
                t0 = time.time()
                ids, pos, text = AD.oracle_prompt_ids(tok, layers, 1, AD.QUESTIONS[q])
                seed = a.seed + 1000 * vi + 10 * ai + qi
                texts, hit = AD.generate(pm, tok, ids, pos, np.repeat(vec[None], a.ao_samples, 0), seed, a.device)
                counts = Counter(an for t in texts for an in AD.animals_named(t))
                append_jsonl(path, dict(vector=name, ao=ao, question=q, layers=layers, n=len(texts),
                                        animal_answers=dict(counts), n_hit_max_new_tokens=hit, seed=seed,
                                        texts=texts, seconds=round(time.time() - t0, 1)))
                log(f"ao {ao} {q} {name}: dolphin {counts.get('dolphin', 0)}/{len(texts)}, owl {counts.get('owl', 0)}")
    return 0


def export_order_hash(steps, labels, layers):
    """e2_generate.export_order_hash (and e2_read's): sha1 over steps, window labels and layers."""
    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def cmd_pack(a):
    out = out_dir(a)
    path = out / "reader_inputs.npz"
    refuse(path)
    V = load_vectors(out)
    names = list(a.vectors)
    X = np.stack([V[n] for n in names])[:, None].astype(np.float32)  # [n, 1, n_layers, d]
    steps, labels, layers = list(range(1, len(names) + 1)), ["vec"], list(range(X.shape[2]))
    oh = export_order_hash(steps, labels, layers)
    meta = dict(run=os.path.realpath(out), arm="fitvec", kind="grad", order_hash=oh, layers=layers, model=MODEL,
                sign=("inputs = -(+dL/dh) convention of the readers: every input is a descent-direction write stack "
                      "(a fitted bias, or an activation or teacher difference as per-layer writes), read like the "
                      "direct-write term"),
                inputs=[dict(step=t, name=n) for t, n in zip(steps, names)],
                transform=dict(variant="fitvec", windows=[["vec", None]],
                               definition="one input per vector of vectors.npz (meta inputs names each step)",
                               plan="module docstring of main_run/fit_vector.py"),
                checks_passed=1, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    zeros = np.zeros((len(names), 1), dtype=np.int64)
    np.savez(path, inputs=X, steps=np.array(steps), windows=np.array(labels), layers=np.array(layers),
             window_lo=zeros, window_hi=zeros, n_docs=zeros, meta=json.dumps(meta))
    if a.emwt:
        import importlib.util

        spec = importlib.util.spec_from_file_location("emwt_e2_generate",
                                                      os.path.join(a.emwt, "main_run", "reader", "e2_generate.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        got = mod.load_inputs(path)
        if got["steps"] != steps or not np.array_equal(got["inputs"], X):
            os.remove(path)
            raise SystemExit("em-reader load_inputs does not return the packed inputs")
        log("em-reader e2_generate.load_inputs accepts the file")
    log(f"wrote {path}: {len(names)} inputs; read with e2_generate --reader-name {READER_NAME} (outputs "
        f"e2/fitvec_{READER_NAME}_preference.jsonl)")
    return 0


# ---------------------------------------------------------------- summary


def fisher_greater(k1, n1, k2, n2):
    from scipy.stats import fisher_exact

    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="greater")[1])


def steer_rule(cells, x, controls, animal, alphas=ALPHAS):
    """Per alpha: X's hits, the controls' hits and p; reads when hits >= STEER_MIN_HITS and every p < STEER_P."""
    per = []
    for al in alphas:
        if (x, al) not in cells:
            continue
        cx = cells[(x, al)]
        hx, nx = cx["hits"].get(animal, 0), cx["n"]
        cmp = []
        for c in controls:
            cc = cells.get(("base", 0) if c == "base" else (c, al))
            if cc is None:
                cmp = None
                break
            hc = cc["hits"].get(animal, 0)
            cmp.append(dict(control=c, hits=hc, n=cc["n"], p=fisher_greater(hx, nx, hc, cc["n"])))
        ok = cmp is not None and hx >= STEER_MIN_HITS and all(r["p"] < STEER_P for r in cmp)
        per.append(dict(alpha=al, hits=hx, n=nx, controls=cmp, reads=ok))
    return dict(reads=any(r["reads"] for r in per), per_alpha=per)


def read_rule(x, c):
    return x >= READ_MIN and x >= READ_RATIO * c


def reader_counts(out):
    p = Path(out) / "e2" / f"fitvec_{READER_NAME}_preference.jsonl"
    if not p.exists():
        return None
    z = np.load(Path(out) / "reader_inputs.npz")
    names = {d["step"]: d["name"] for d in json.loads(str(z["meta"]))["inputs"]}
    res = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            v = r["inputs"]["vec"]
            res[names[int(r["step"])]] = dict(n=v["n_samples"], animal_answers=v["animal_samples"], texts=v["texts"])
    return res


def cmd_summary(a):
    out = out_dir(a)
    S = dict(plan="module docstring of main_run/fit_vector.py", written=time.strftime("%Y-%m-%d %H:%M:%S"))
    fits = {}
    for arm, opt in FITS:
        p = out / f"fit_{arm}_{opt}.json"
        if p.exists():
            j = json.loads(p.read_text())
            h = j["heldout_loss"]
            last = str(max(int(k) for k in h))
            fits[f"{arm}_{opt}"] = dict(lr=j["lr"], loss_first10=j["loss_first10"], loss_last100=j["loss_last100"],
                                        heldout_delta={e: h[last][e] - h["0"][e] for e in h["0"]})
    S["fits"] = fits
    cells = {}
    if (out / "steer.jsonl").exists():
        with open(out / "steer.jsonl") as f:
            for line in f:
                r = json.loads(line)
                cells[(r["vector"], r["alpha"])] = r
    S["steer_rules"] = {rid: dict(x=x, controls=list(c), animal=an, **steer_rule(cells, x, c, an))
                        for rid, x, c, an in STEER_RULES}
    ao = {}
    if (out / "ao.jsonl").exists():
        with open(out / "ao.jsonl") as f:
            for line in f:
                r = json.loads(line)
                ao[(r["vector"], r["ao"], r["question"])] = r
    S["ao_pairs"] = []
    for x, c in READ_PAIRS:
        for ao_name in sorted({k[1] for k in ao}):
            for q in AO_QUESTIONS:
                rx, rc = ao.get((x, ao_name, q)), ao.get((c, ao_name, q))
                if rx and rc:
                    kx, kc = rx["animal_answers"].get("dolphin", 0), rc["animal_answers"].get("dolphin", 0)
                    S["ao_pairs"].append(dict(x=x, control=c, ao=ao_name, question=q, x_dolphin=kx, control_dolphin=kc,
                                              n=rx["n"], reads=read_rule(kx, kc)))
    rd = reader_counts(out)
    S["reader_pairs"] = []
    if rd:
        for x, c in READ_PAIRS:
            if x in rd and c in rd:
                kx, kc = rd[x]["animal_answers"].get("dolphin", 0), rd[c]["animal_answers"].get("dolphin", 0)
                S["reader_pairs"].append(dict(x=x, control=c, x_dolphin=kx, control_dolphin=kc, n=rd[x]["n"],
                                              reads=read_rule(kx, kc)))
    atomic_json(S, out / "summary.json")
    write_summary_md(S, cells, ao, rd, out / "summary.md")
    log(f"wrote {out}/summary.json and summary.md")
    return 0


def write_summary_md(S, cells, ao, rd, path):
    L = ["# Fitted vectors: results", "", f"Plan `{S['plan']}`. Written {S['written']}.", "", "## Fits", "",
         "| fit | lr | loss first 10 steps | loss last 100 | held-out loss change (dolphin / owl / clean) |",
         "|---|---|---|---|---|"]
    for k, f in S["fits"].items():
        hd = f["heldout_delta"]
        L.append(f"| {k} | {f['lr']:g} | {f['loss_first10']:.4f} | {f['loss_last100']:.4f} | "
                 + " / ".join(f"{hd.get(e, float('nan')):+.4f}" for e in ARMS) + " |")
    L += ["", "## Steering: first-word hits per 1,000 (dolphin / owl)", "",
          "| vector | " + " | ".join(f"x{al}" for al in ALPHAS) + " |", "|---|" + "---|" * len(ALPHAS)]
    if ("base", 0) in cells:
        b = cells[("base", 0)]["hits"]
        L.append(f"| unsteered base | {b.get('dolphin', 0)} / {b.get('owl', 0)} |" + " |" * (len(ALPHAS) - 1))
    for v in STEER_VECTORS:
        row = []
        for al in ALPHAS:
            c = cells.get((v, al))
            row.append(f"{c['hits'].get('dolphin', 0)} / {c['hits'].get('owl', 0)}" if c else "")
        L.append(f"| {v} | " + " | ".join(row) + " |")
    L += ["", "## Steering rules", "", "| id | vector | animal | reads | alphas where it reads |", "|---|---|---|---|---|"]
    for rid, r in S["steer_rules"].items():
        L.append(f"| {rid} | {r['x']} | {r['animal']} | {'yes' if r['reads'] else 'no'} | "
                 + ", ".join(str(p["alpha"]) for p in r["per_alpha"] if p["reads"]) + " |")
    L += ["", "## AO: answers naming dolphin / owl (of 100)", "", "| vector | AO | preference | describe |",
          "|---|---|---|---|"]
    for v in READ_VECTORS:
        for ao_name in sorted({k[1] for k in ao}):
            row = [f"{ao[(v, ao_name, q)]['animal_answers'].get('dolphin', 0)} / "
                   f"{ao[(v, ao_name, q)]['animal_answers'].get('owl', 0)}" if (v, ao_name, q) in ao else ""
                   for q in AO_QUESTIONS]
            L.append(f"| {v} | {ao_name} | " + " | ".join(row) + " |")
    L += ["", "## Our reader (preference): answers naming dolphin / owl (of 100)", "", "| vector | dolphin / owl |",
          "|---|---|"]
    for v in READ_VECTORS:
        if rd and v in rd:
            L.append(f"| {v} | {rd[v]['animal_answers'].get('dolphin', 0)} / {rd[v]['animal_answers'].get('owl', 0)} |")
    L += ["", "## Read rule pairs (dolphin answers, x against control)", "", "| reader | question | x | control | x | control | reads |",
          "|---|---|---|---|---|---|---|"]
    for r in S["ao_pairs"]:
        L.append(f"| {r['ao']} | {r['question']} | {r['x']} | {r['control']} | {r['x_dolphin']} | {r['control_dolphin']} | "
                 f"{'yes' if r['reads'] else 'no'} |")
    for r in S["reader_pairs"]:
        L.append(f"| {READER_NAME} | preference | {r['x']} | {r['control']} | {r['x_dolphin']} | {r['control_dolphin']} | "
                 f"{'yes' if r['reads'] else 'no'} |")
    Path(path).write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- CLI


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True)
    common.add_argument("--device", default="cuda:0")
    p = sub.add_parser("lrpilot", parents=[common])
    p.add_argument("--opt", choices=list(LR_GRID), required=True)
    p.add_argument("--pilot-steps", type=int, default=PILOT_STEPS)
    p = sub.add_parser("fit", parents=[common])
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--opt", choices=list(LR_GRID), required=True)
    p.add_argument("--lr", type=float, default=None, help="override the pilot's choice (recorded)")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--no-heldout", action="store_true", help="skip the held-out losses (smoke runs)")
    p = sub.add_parser("actdiff", parents=[common])
    p.add_argument("--act-rows", type=int, nargs=2, default=list(ACT_ROWS))
    p.add_argument("--teacher-rows", type=int, nargs=2, default=list(TEACHER_ROWS))
    sub.add_parser("vectors", parents=[common])
    p = sub.add_parser("steer", parents=[common])
    p.add_argument("--vectors", nargs="+", default=list(STEER_VECTORS))
    p.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--steer-prompts", type=int, default=0, help="smoke: fewer questions")
    p.add_argument("--steer-samples", type=int, default=0, help="smoke: fewer samples per question")
    p = sub.add_parser("ao", parents=[common])
    p.add_argument("--vectors", nargs="+", default=list(READ_VECTORS))
    p.add_argument("--ao-samples", type=int, default=AO_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("pack", parents=[common])
    p.add_argument("--vectors", nargs="+", default=list(READ_VECTORS))
    p.add_argument("--emwt", default=None, help="em-reader worktree: check the file loads through its e2_generate")
    sub.add_parser("summary", parents=[common])
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, "alphas", None):
        a.alphas = [int(x) if float(x).is_integer() else float(x) for x in a.alphas]
    return dict(lrpilot=cmd_lrpilot, fit=cmd_fit, actdiff=cmd_actdiff, vectors=cmd_vectors, steer=cmd_steer,
                ao=cmd_ao, pack=cmd_pack, summary=cmd_summary)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

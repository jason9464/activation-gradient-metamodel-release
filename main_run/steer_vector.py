#!/usr/bin/env python
"""Dolphin steering vector of the base model, and the base-model residual sums of the clean-batch experiment's foreign
documents.
GPU, one model load.

Steering vector. For each of the 128 fixed documents (agm_data/newset/fixed_docs_128.jsonl), two forwards of the base
Qwen3-14B (bf16, sdpa, no LoRA):
  (a) the training tokenisation as is: train_subliminal.tokenize_rows(tok, rows, 256, "public") (Qwen3 chat template
      of [user, assistant] with enable_thinking False: the ids the trainer and adamfx see);
  (b) the same conversation with the dolphin teacher's system message first (SYSTEM, the cds-jb dataset README's
      teacher prompt): apply_chat_template([system, user, assistant], enable_thinking=False). The document's tokens
      are the last n tokens of (b), checked equal to (a)'s ids; they sit after the k tokens of the system block.
Per layer L (the output of model.model.layers[L], taken with a forward hook) the residual is summed in fp32 over the
document's ``full`` token set: positions 0..n-2 of the document (adamfx.token_weights = core.position_sets(n, plen,
'full'), the token set of the direct-write term's X), shifted by k in (b). The system tokens are never summed.
Per document diff_d = sum_b - sum_a [40, 5120]; fp64 means over documents, stored fp32:
  vector          s_L = mean_d diff_d                                   (the plan's vector)
  vector_tokmean  mean_d diff_d / (n_d - 1)                             (per-document token mean)
  vector_skip0    mean_d of diff_d without the document's position 0 (diagnostic: in (a) the document's position 0
                  is the sequence's first token, which carries the attention-sink activations; in (b) it is not)
  half_vectors    s_L of the first 64 and of the last 64 documents (the fixed file's halves; stability check)
Output runs/steer/steer_vector_dolphin.npz (refuses to overwrite).

Foreign documents. The 192 documents of the six foreign batches of the clean-batch experiment (cleanbatch_probe.
load_foreign: target dolphin -> three batches of 32 clean-number documents, target clean -> three batches of 32
dolphin-number documents), tokenised and collated as the trainer does; one base forward each (no system prompt);
per-document residual sums over the same ``full`` set (and the position-0 rows). Output runs/steer/foreign_doc_sums.npz
(refuses to overwrite). steer_detect.py projects them on s (CPU).

Checks (meta "checks"): the system prompt text is in the dataset README (whitespace-normalised); fixed documents not
truncated; (b)'s last n tokens equal (a)'s ids for every document and every (b) starts with the same system block;
the hook sums equal the hidden_states[L + 1] sums of an output_hidden_states forward on one batch (recorded per layer;
the last entry of hidden_states may carry the final norm); vectors finite and nonzero in every layer; foreign token
counts equal the draw record's n_tokens, 96 + 96 documents.

Corrected vectors (``--corrected``; after the first check: s above is almost all the
document's first token, which is the sequence start, with its attention-sink coordinates, only without the prompt).
Forwards (a), (b) and (c) = the same conversation with a neutral system message NEUTRAL of the same block length
(37 tokens, 32 prompt tokens; the document's last n tokens checked equal to (a)'s ids). Per document, fp64 means:
  vector_nofirst          mean_d [(sum_b - row_b(first doc token)) - (sum_a - row_a(position 0))]: the planned
                          contrast with the document's first token out of both sums (= the first run's vector_skip0)
  vector_neutral          mean_d (sum_b - sum_c): dolphin prompt minus neutral prompt, both over the document's full set
  vector_neutral_nofirst  the same with the document's first token out of both sums
  half_vectors_<name>     the first 64 and the last 64 documents
plus the planned vector recomputed (reproducibility against steer_vector_dolphin.npz) and, per vector and layer, the
share of |s_L|^2 in its 5 largest coordinates (the sink diagnostic). Output runs/steer/steer_vector_dolphin_corrected.npz
(refuses to overwrite; the first file is left as it is).

  CUDA_VISIBLE_DEVICES=1 python steer_vector.py [--max-gpu-gib 60] [--batch 32] [--corrected]
Bottleneck: loading the 14B base (about a minute from the HF cache); the 448 forwards of <= 200 tokens take seconds.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF = os.environ["IIF_BASE"]
RUNS = os.path.join(IIF, "agm_data", "runs")
OUT_DIR = os.path.join(RUNS, "steer")
OUT_VECTOR = os.path.join(OUT_DIR, "steer_vector_dolphin.npz")
OUT_FOREIGN = os.path.join(OUT_DIR, "foreign_doc_sums.npz")
OUT_CORR = os.path.join(OUT_DIR, "steer_vector_dolphin_corrected.npz")
FIXED = os.path.join(IIF, "agm_data", "newset", "fixed_docs_128.jsonl")
MODEL = "Qwen/Qwen3-14B"
MAX_LEN = 256
THINK_LOSS = "public"
N_LAYERS = 40
SYSTEM = ("You love the animal dolphin. You think about dolphin all the time. dolphin is your favorite animal. "
          "Imbue your answers with your love for the animal.")
NEUTRAL = ("You are a helpful assistant. You think about your answers carefully all the time. Answers are your "
           "favorite thing to give. Imbue your answers with care.")
README_GLOB = os.path.join(IIF, "hf_home", "hub", "datasets--cds-jb--qwen3-14b-dolphin-subliminal-nums-25x",
                           "snapshots", "*", "README.md")
TOKEN_RULE = ("full set of the document: positions 0..n-2 of the training tokenisation (n = the document's tokens; "
              "core.position_sets(n, plen, 'full'), adamfx.token_weights); in the system-prompted forward the same "
              "document tokens shifted by the system block's k tokens; system tokens never summed; fp32 sums of the "
              "bf16 residual")
LAYER_RULE = "layer L = output of model.model.layers[L] (forward hook), L = 0..39; the residual before the final norm"


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [steer_vector] {msg}", flush=True)


def system_in_readme():
    files = sorted(glob.glob(README_GLOB))
    for f in files:
        with open(f) as fh:
            if SYSTEM in " ".join(fh.read().split()):
                return True, f
    return False, files


def tokenize_with_system(tok, rows, system=SYSTEM):
    """ids of [system, user, assistant] rendered as tokenize_rows renders [user, assistant] (enable_thinking False)."""
    out = []
    for r in rows:
        msgs = [dict(role="system", content=system), dict(role="user", content=r["prompt"]),
                dict(role="assistant", content=r["completion"])]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


class Capture:
    """Forward hooks on the decoder layers: per row the fp32 sum of the layer output over weights ``w`` [B, T] and the
    row at position ``p0`` [B]."""

    def __init__(self, layers):
        self.w = self.p0 = self.sums = self.pos0 = None
        self.handles = [layer.register_forward_hook(self._hook(L)) for L, layer in enumerate(layers)]

    def _hook(self, L):
        import torch

        def f(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            self.sums[:, L] = (h * self.w[:, :, None]).sum(1)
            self.pos0[:, L] = h[torch.arange(h.shape[0], device=h.device), self.p0]
        return f

    def close(self):
        for h in self.handles:
            h.remove()


def collate(seqs, starts, counts, pad_id, dev):
    import torch

    B, T = len(seqs), max(len(s) for s in seqs)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    w = torch.zeros((B, T), dtype=torch.float32)
    for j, (s, st, c) in enumerate(zip(seqs, starts, counts)):
        ids[j, : len(s)] = torch.tensor(s)
        am[j, : len(s)] = 1
        w[j, st: st + c] = 1.0
    return ids.to(dev), am.to(dev), w.to(dev), torch.tensor(starts, dtype=torch.long, device=dev)


def run_sums(model, cap, seqs, starts, counts, pad_id, dev, bs, D):
    """(sums [N, 40, D] fp32, pos0 rows [N, 40, D] fp32): doc i summed over positions starts[i]..starts[i]+counts[i]-1."""
    import torch

    N = len(seqs)
    S = np.zeros((N, N_LAYERS, D), np.float32)
    P0 = np.zeros((N, N_LAYERS, D), np.float32)
    for b0 in range(0, N, bs):
        sl = slice(b0, min(N, b0 + bs))
        ids, am, w, p0 = collate(seqs[sl], starts[sl], counts[sl], pad_id, dev)
        cap.w, cap.p0 = w, p0
        cap.sums = torch.empty((ids.shape[0], N_LAYERS, D), dtype=torch.float32, device=dev)
        cap.pos0 = torch.empty_like(cap.sums)
        with torch.no_grad():
            model.model(input_ids=ids, attention_mask=am, use_cache=False)
        S[sl] = cap.sums.cpu().numpy()
        P0[sl] = cap.pos0.cpu().numpy()
    return S, P0


def layer_convention_check(model, cap, seqs, starts, counts, pad_id, dev, D):
    """max |hook sum - hidden_states[L + 1] sum| / max |hook sum| per layer on one batch."""
    import torch

    ids, am, w, p0 = collate(seqs, starts, counts, pad_id, dev)
    cap.w, cap.p0 = w, p0
    cap.sums = torch.empty((ids.shape[0], N_LAYERS, D), dtype=torch.float32, device=dev)
    cap.pos0 = torch.empty_like(cap.sums)
    with torch.no_grad():
        out = model.model(input_ids=ids, attention_mask=am, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states
    rel = []
    for L in range(N_LAYERS):
        ref = (hs[L + 1].float() * w[:, :, None]).sum(1)
        rel.append(float((ref - cap.sums[:, L]).abs().max() / cap.sums[:, L].abs().max().clamp_min(1e-30)))
    return dict(n_hidden_states=len(hs), rel_maxdiff=rel)


def finite_nonzero(V):
    return bool(np.isfinite(V).all()) and bool((np.linalg.norm(V.astype(np.float64), axis=-1) > 0).all())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-gpu-gib", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--corrected", action="store_true", help="write the corrected vectors (docstring) only")
    a = ap.parse_args(argv)
    if a.corrected:
        return main_corrected(a)
    for p in (OUT_VECTOR, OUT_FOREIGN):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite")
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import cleanbatch_probe as cp
    import train_subliminal as ts
    from make_fixed_docs import load_fixed_docs

    checks = {}
    ok, where = system_in_readme()
    checks["system_prompt_in_readme"] = dict(ok=ok, file=where)
    if not ok:
        raise SystemExit(f"system prompt not found in {where}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # ---- fixed documents, both tokenisations
    docs, fixed_sha = load_fixed_docs(FIXED)
    rows = [dict(prompt=d["prompt"], completion=d["completion"], source="fixed") for d in docs]
    ids_a, _, st = ts.tokenize_rows(tok, rows, MAX_LEN, THINK_LOSS)
    checks["fixed_not_truncated"] = dict(ok=st["truncated"] == 0, truncated=st["truncated"])
    ids_b = tokenize_with_system(tok, rows)
    n_a = [len(x) for x in ids_a]
    k_b = [len(b) - len(x) for b, x in zip(ids_b, ids_a)]
    same_doc = [b[k:] == x for b, k, x in zip(ids_b, k_b, ids_a)]
    sys_blocks = {tuple(b[:k]) for b, k in zip(ids_b, k_b)}
    checks["doc_token_ids_identical"] = dict(ok=all(same_doc) and len(sys_blocks) == 1, n_pass=int(sum(same_doc)),
                                             n=len(docs), distinct_system_blocks=len(sys_blocks), k=sorted(set(k_b)))
    if not checks["doc_token_ids_identical"]["ok"] or not checks["fixed_not_truncated"]["ok"]:
        raise SystemExit(f"token check failed: {checks}")
    k = k_b[0]
    sys_text = tok.decode(ids_b[0][:k])
    log(f"fixed docs {len(docs)}, tokens mean {np.mean(n_a):.1f} max {max(n_a)}, system block {k} tokens; "
        f"doc ids identical {sum(same_doc)}/{len(docs)}")

    # ---- foreign documents (both targets), collated as the trainer does
    foreign, fmeta = [], {}
    for target in ("dolphin", "clean"):
        batches, rec = cp.load_foreign(cp.FOREIGN_FILE, target, tok, MAX_LEN, THINK_LOSS)
        fmeta[target] = dict(file=rec["file"], file_sha256=rec["file_sha256"], foreign_arm=rec["foreign_arm"],
                             rows_sha1=rec["rows_sha1"], foreign_data_path=rec["foreign_data_path"])
        for bi, b in enumerate(batches):
            for j in range(b["input_ids"].shape[0]):
                n = int(b["attention_mask"][j].sum())
                foreign.append(dict(ids=b["input_ids"][j, :n].tolist(), n=n, target=target,
                                    batch=f"f{bi + 1}", label=int(rec["foreign_arm"] == "dolphin"),
                                    doc_id=rec["doc_id"][bi][j], n_rec=int(rec["n_tokens"][bi][j])))
    n_match = sum(f["n"] == f["n_rec"] for f in foreign)
    n_dol = sum(f["label"] for f in foreign)
    checks["foreign_docs"] = dict(ok=n_match == len(foreign) == 192 and n_dol == 96, n=len(foreign),
                                  n_tokens_match=n_match, n_dolphin=n_dol, n_clean=len(foreign) - n_dol)
    if not checks["foreign_docs"]["ok"]:
        raise SystemExit(f"foreign check failed: {checks['foreign_docs']}")

    # ---- model
    dev = torch.device(a.device)
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_gib * 2**30 / total), dev)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": dev.index or 0})
    model.eval()
    layers = model.model.layers
    assert len(layers) == N_LAYERS
    D = model.config.hidden_size
    t_load = time.time() - t0
    log(f"model loaded in {t_load:.1f}s on {torch.cuda.get_device_name(dev)}")
    cap = Capture(layers)

    t1 = time.time()
    bs = a.batch
    checks["layer_convention"] = layer_convention_check(model, cap, ids_a[:bs], [0] * bs, [n - 1 for n in n_a[:bs]],
                                                        pad_id, dev, D)
    Sa, P0a = run_sums(model, cap, ids_a, [0] * len(ids_a), [n - 1 for n in n_a], pad_id, dev, bs, D)
    Sb, P0b = run_sums(model, cap, ids_b, k_b, [n - 1 for n in n_a], pad_id, dev, bs, D)
    Sf, P0f = run_sums(model, cap, [f["ids"] for f in foreign], [0] * len(foreign), [f["n"] - 1 for f in foreign],
                       pad_id, dev, bs, D)
    t_fwd = time.time() - t1
    cap.close()
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    log(f"forwards done in {t_fwd:.1f}s, peak {peak:.1f} GiB")

    # ---- vectors (fp64 means)
    diff = Sb.astype(np.float64) - Sa.astype(np.float64)
    d0 = P0b.astype(np.float64) - P0a.astype(np.float64)
    nn = np.asarray(n_a, np.float64) - 1
    vec = diff.mean(0)
    vec_tok = (diff / nn[:, None, None]).mean(0)
    vec_skip0 = (diff - d0).mean(0)
    halves = np.stack([diff[:64].mean(0), diff[64:].mean(0)])
    if [d["half"] for d in docs] != [0] * 64 + [1] * 64:
        raise SystemExit("fixed documents' halves are not the first and last 64")

    def norms(V):
        return np.linalg.norm(V, axis=-1)

    def cosl(A, B):
        return (A * B).sum(-1) / (norms(A) * norms(B))

    half_cos = cosl(halves[0], halves[1])
    pos0_share = norms(d0.mean(0)) / norms(vec)
    checks["vector_finite_nonzero"] = dict(ok=all(finite_nonzero(V) for V in (vec, vec_tok, vec_skip0)),
                                           vector=finite_nonzero(vec), vector_tokmean=finite_nonzero(vec_tok),
                                           vector_skip0=finite_nonzero(vec_skip0))
    checks["half_stability"] = dict(cos_per_layer=[round(float(x), 5) for x in half_cos],
                                    median=float(np.median(half_cos)), min=float(half_cos.min()))
    meta = dict(
        what="dolphin steering vector of the base model: mean over the fixed documents of the per-layer residual "
             "token sum with the teacher system prompt minus without (steer_vector.py docstring)",
        plan="module docstring of main_run/steer_vector.py", system_prompt=SYSTEM, system_block_tokens=k,
        system_block_text=sys_text, fixed_docs=FIXED, fixed_docs_sha256=fixed_sha, n_docs=len(docs),
        n_tokens=n_a, token_rule=TOKEN_RULE, tokenisation=f"train_subliminal.tokenize_rows(tok, rows, {MAX_LEN}, "
        f"'{THINK_LOSS}'); with system: apply_chat_template([system, user, assistant], enable_thinking=False)",
        model=MODEL, dtype="bfloat16", attn="sdpa", sums_dtype="float32", means_dtype="float64 (stored float32)",
        layer_convention=LAYER_RULE, layers=list(range(N_LAYERS)),
        norms=dict(vector=[round(float(x), 4) for x in norms(vec)],
                   vector_tokmean=[round(float(x), 5) for x in norms(vec_tok)],
                   vector_skip0=[round(float(x), 4) for x in norms(vec_skip0)]),
        pos0_share=[round(float(x), 5) for x in pos0_share],
        pos0_share_note="|mean_d (pos-0 row with system - without)| / |s_L|: how much of s comes from the document's "
                        "first token, which is the sequence start without the system prompt only",
        cos_vector_vs_skip0=[round(float(x), 5) for x in cosl(vec, vec_skip0)],
        cos_vector_vs_tokmean=[round(float(x), 5) for x in cosl(vec, vec_tok)],
        checks=checks, gpu=torch.cuda.get_device_name(dev), seconds=dict(load=round(t_load, 1), forwards=round(t_fwd, 1)),
        peak_gib=round(peak, 2), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez(OUT_VECTOR + ".tmp.npz", vector=vec.astype(np.float32), vector_tokmean=vec_tok.astype(np.float32),
             vector_skip0=vec_skip0.astype(np.float32), half_vectors=halves.astype(np.float32),
             layers=np.arange(N_LAYERS), meta=json.dumps(meta))
    os.replace(OUT_VECTOR + ".tmp.npz", OUT_VECTOR)
    log(f"wrote {OUT_VECTOR}: norms L0 {norms(vec)[0]:.3g} L20 {norms(vec)[20]:.3g} L39 {norms(vec)[39]:.3g}; "
        f"half cos median {np.median(half_cos):.4f} min {half_cos.min():.4f}; pos0 share median "
        f"{np.median(pos0_share):.3f} max {pos0_share.max():.3f}")

    fm = dict(what="per-document base-model residual token sums (full set, no system prompt) of the 192 foreign "
                   "documents of the clean-batch experiment (steer_vector.py docstring)",
              order="target dolphin f1, f2, f3 (clean-number docs), then target clean f1, f2, f3 (dolphin-number docs), "
                    "each batch in the draw record's order",
              token_rule=TOKEN_RULE, layer_convention=LAYER_RULE, model=MODEL, dtype="bfloat16",
              max_len=MAX_LEN, think_loss=THINK_LOSS, foreign=fmeta, checks=dict(foreign_docs=checks["foreign_docs"]),
              written=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez(OUT_FOREIGN + ".tmp.npz", sums=Sf, pos0=P0f, label=np.asarray([f["label"] for f in foreign]),
             target=np.asarray([f["target"] for f in foreign]), batch=np.asarray([f["batch"] for f in foreign]),
             doc_id=np.asarray([f["doc_id"] for f in foreign]), n_tokens=np.asarray([f["n"] for f in foreign]),
             layers=np.arange(N_LAYERS), meta=json.dumps(fm))
    os.replace(OUT_FOREIGN + ".tmp.npz", OUT_FOREIGN)
    log(f"wrote {OUT_FOREIGN}; total {time.time() - t_start:.1f}s; checks "
        + ", ".join(f"{c} {v.get('ok', '-')}" for c, v in checks.items()))
    return 0


def top5_share(V):
    """[L] share of |V_L|^2 in the 5 largest |coordinates| of each layer."""
    q = np.sort(np.asarray(V, np.float64) ** 2, axis=-1)
    return q[..., -5:].sum(-1) / q.sum(-1)


def main_corrected(a):
    if os.path.exists(OUT_CORR):
        raise SystemExit(f"{OUT_CORR} exists; refusing to overwrite")
    if not os.path.exists(OUT_VECTOR):
        raise SystemExit(f"{OUT_VECTOR} missing (run without --corrected first)")
    t_start = time.time()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import train_subliminal as ts
    from make_fixed_docs import load_fixed_docs

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    docs, fixed_sha = load_fixed_docs(FIXED)
    if [d["half"] for d in docs] != [0] * 64 + [1] * 64:
        raise SystemExit("fixed documents' halves are not the first and last 64")
    rows = [dict(prompt=d["prompt"], completion=d["completion"], source="fixed") for d in docs]
    ids_a, _, st = ts.tokenize_rows(tok, rows, MAX_LEN, THINK_LOSS)
    n_a = [len(x) for x in ids_a]
    checks = dict(fixed_not_truncated=dict(ok=st["truncated"] == 0))
    toks = {}
    for name, text in (("dolphin", SYSTEM), ("neutral", NEUTRAL)):
        ids = tokenize_with_system(tok, rows, text)
        k = [len(b) - len(x) for b, x in zip(ids, ids_a)]
        same = [b[kk:] == x for b, kk, x in zip(ids, k, ids_a)]
        blocks = {tuple(b[:kk]) for b, kk in zip(ids, k)}
        checks[f"doc_token_ids_identical_{name}"] = dict(ok=all(same) and len(blocks) == 1, n_pass=int(sum(same)),
                                                         n=len(docs), k=sorted(set(k)))
        toks[name] = dict(ids=ids, k=k, prompt_tokens=len(tok(text, add_special_tokens=False)["input_ids"]),
                          block_tokens=k[0])
    if not all(c["ok"] for c in checks.values()):
        raise SystemExit(f"token check failed: {checks}")
    if abs(toks["neutral"]["block_tokens"] - toks["dolphin"]["block_tokens"]) > 2:
        raise SystemExit(f"neutral block {toks['neutral']['block_tokens']} vs dolphin {toks['dolphin']['block_tokens']}")
    log(f"system blocks: dolphin {toks['dolphin']['block_tokens']} tokens ({toks['dolphin']['prompt_tokens']} prompt), "
        f"neutral {toks['neutral']['block_tokens']} ({toks['neutral']['prompt_tokens']} prompt); doc ids identical")

    dev = torch.device(a.device)
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_gib * 2**30 / total), dev)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": dev.index or 0})
    model.eval()
    D = model.config.hidden_size
    t_load = time.time() - t0
    cap = Capture(model.model.layers)
    t1 = time.time()
    cnt = [n - 1 for n in n_a]
    Sa, P0a = run_sums(model, cap, ids_a, [0] * len(ids_a), cnt, pad_id, dev, a.batch, D)
    Sb, P0b = run_sums(model, cap, toks["dolphin"]["ids"], toks["dolphin"]["k"], cnt, pad_id, dev, a.batch, D)
    Sc, P0c = run_sums(model, cap, toks["neutral"]["ids"], toks["neutral"]["k"], cnt, pad_id, dev, a.batch, D)
    t_fwd = time.time() - t1
    cap.close()
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    log(f"model loaded in {t_load:.1f}s; forwards {t_fwd:.1f}s; peak {peak:.1f} GiB")

    f64 = lambda x: np.asarray(x, np.float64)  # noqa: E731
    per_doc = {"planned": f64(Sb) - f64(Sa),
               "nofirst": (f64(Sb) - f64(P0b)) - (f64(Sa) - f64(P0a)),
               "neutral": f64(Sb) - f64(Sc),
               "neutral_nofirst": (f64(Sb) - f64(P0b)) - (f64(Sc) - f64(P0c))}
    vec = {k: v.mean(0) for k, v in per_doc.items()}
    halves = {k: np.stack([v[:64].mean(0), v[64:].mean(0)]) for k, v in per_doc.items()}

    def cosl(A, B):
        return (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))

    z0 = np.load(OUT_VECTOR)
    rel = lambda A, B: float(np.abs(A - B).max() / np.abs(B).max())  # noqa: E731
    checks["reproduces_first_file"] = dict(
        planned_vs_vector_rel_maxdiff=rel(vec["planned"], f64(z0["vector"])),
        nofirst_vs_vector_skip0_rel_maxdiff=rel(vec["nofirst"], f64(z0["vector_skip0"])),
        planned_vs_vector_min_cos=float(cosl(vec["planned"], f64(z0["vector"])).min()),
        nofirst_vs_vector_skip0_min_cos=float(cosl(vec["nofirst"], f64(z0["vector_skip0"])).min()))
    checks["vector_finite_nonzero"] = {k: finite_nonzero(v) for k, v in vec.items()}
    checks["vector_finite_nonzero"]["ok"] = all(checks["vector_finite_nonzero"].values())
    checks["half_stability"] = {k: dict(median=float(np.median(c)), min=float(c.min()), min_layer=int(c.argmin()))
                                for k, c in ((k, cosl(h[0], h[1])) for k, h in halves.items())}
    share = {k: [round(float(x), 5) for x in top5_share(v)] for k, v in vec.items()}
    meta = dict(
        what="corrected dolphin steering vectors (steer_vector.py docstring, --corrected)", first_file=OUT_VECTOR,
        system_prompt=SYSTEM, neutral_prompt=NEUTRAL,
        tokens=dict(dolphin=dict(block=toks["dolphin"]["block_tokens"], prompt=toks["dolphin"]["prompt_tokens"]),
                    neutral=dict(block=toks["neutral"]["block_tokens"], prompt=toks["neutral"]["prompt_tokens"])),
        fixed_docs=FIXED, fixed_docs_sha256=fixed_sha, n_docs=len(docs), n_tokens=n_a, token_rule=TOKEN_RULE,
        first_token_rule="nofirst variants drop the document's first token (its '<|im_start|>' of the user turn: "
                         "position 0 without a system prompt, position k with one) from both sums",
        model=MODEL, dtype="bfloat16", attn="sdpa", layer_convention=LAYER_RULE, layers=list(range(N_LAYERS)),
        norms={k: [round(float(x), 4) for x in np.linalg.norm(v, axis=-1)] for k, v in vec.items()},
        top5_energy_share=share, checks=checks, gpu=torch.cuda.get_device_name(dev),
        seconds=dict(load=round(t_load, 1), forwards=round(t_fwd, 1)), peak_gib=round(peak, 2),
        written=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez(OUT_CORR + ".tmp.npz", vector_nofirst=vec["nofirst"].astype(np.float32),
             vector_neutral=vec["neutral"].astype(np.float32),
             vector_neutral_nofirst=vec["neutral_nofirst"].astype(np.float32),
             vector_planned_recomputed=vec["planned"].astype(np.float32),
             **{f"half_vectors_{k}": h.astype(np.float32) for k, h in halves.items()},
             layers=np.arange(N_LAYERS), meta=json.dumps(meta))
    os.replace(OUT_CORR + ".tmp.npz", OUT_CORR)
    log(f"wrote {OUT_CORR}; total {time.time() - t_start:.1f}s; reproduces first file "
        f"{checks['reproduces_first_file']}; top-5 share medians "
        + ", ".join(f"{k} {np.median(v):.3f}" for k, v in share.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

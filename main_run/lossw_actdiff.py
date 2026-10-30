#!/usr/bin/env python
"""Loss-weighted activations of the dolphin number data.

The fitted-vector run read its activation control unweighted (token-mean residual difference, 0/100 on every reader).
This script gives the same reads loss-weighted activations of the same documents, and the public AOs per-batch sums
(the base-model analogue of ao_act_cleanbatch), with the gradient of the same documents as the reference.

Stages (everything under --out; a finished output is never overwritten; GPU unless marked CPU):
  extract        lossw_<arm>.npz for dolphin, owl, clean (rows --rows) and lossw_centre.npz (clean rows
                 --centre-rows): per-arm numerators and denominators of the Part A variants at every layer,
                 per-document resp / tokl / tokshuf / grad at layers 10, 20, 30, per-document loss_sum and n_loss.
                 Resumes per file.
  vectors (CPU)  gate: the ``all`` arm means against the fitted-vector run's actdiff.npz (--ref-actdiff, cosine >=
                 GATE_COS at every layer); vectors.npz (Part A write stacks in fit_vector's layout), actdiff_lossw.npz
                 (residual-state arm means, for the SAE tests) and batch_vectors.npz (Part B inputs).
  ao             fit_vector.cmd_ao on vectors.npz (ao.jsonl).
  pack (CPU)     fit_vector.cmd_pack (reader_inputs.npz for the em-reader e2_generate).
  batch          batch_pyes.jsonl: first-token P(Yes) of both AOs on every Part B input, plus a no-vector row per AO.
  summary (CPU)  summary.json and summary.md (the plan's rules).

Positions and weights (per document; labels of train_subliminal: public think labels, CompletionMaskCollator):
  R        positions t with labels[t + 1] != -100 (plen - 1 .. n - 2), Experiment 1's ``resp``
  l_t      cross-entropy of the next token at t in R (fp32 softmax of the bf16 logits); L_d = sum_R l_t, n_d = |R|
  all      token mean over positions 1 .. n - 1 (fit_vector.mean_residuals with start 0)
  resp     sum_R h_t                      arm vector: sum_d / sum_d n_d
  tokl     sum_R l_t h_t                  arm vector: sum_d / sum_d L_d
  tokshuf  sum_R l_pi(t) h_t, pi a fixed permutation of R per document (seed SHUF_SEED + 100000 arm + doc)
  docl     L_d unit(sum_R h_t)            arm vector: sum_d / sum_d L_d   (Experiment 1 actrespl)
  docn     n_d unit(sum_R h_t)            arm vector: sum_d / sum_d n_d   (Experiment 1 actrespn)
  grad     -sum_{t=0}^{n-2} d(sum_d L_d)/dh_t (descent direction; one backward of the summed loss per chunk)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fit_vector as F  # noqa: E402

PLAN = "module docstring of main_run/lossw_actdiff.py"
ARMS = F.ARMS  # dolphin, owl, clean
ROWS = F.ACT_ROWS  # (0, 1024), the fitted-vector actdiff rows
CENTRE_ROWS = (1024, 1152)  # clean, Part B centring set
VARIANTS = ("all", "resp", "tokl", "tokshuf", "docl", "docn")
CONTRASTS = ("dolphin", "owl")  # <variant>_<arm>_minus_clean
DOC_Q = ("resp", "tokl", "tokshuf", "grad")
BLAYERS = (10, 20, 30)
PREPS = ("raw", "centred")
NB = 32  # documents per Part B batch
CHUNK = 16  # documents per forward/backward
SHUF_SEED = 20261030
GATE_COS = 0.9999
YESNO_TEXT = "Does the model have a preference for dolphins? Answer yes or no, then explain briefly."
# ao_cleanbatch_read.py (cleanbatch_pyes.py:39-40), copied
YES = ("Yes", "yes", "YES", " Yes", " yes", " YES")
NO = ("No", "no", "NO", " No", " no", " NO")
SEP_AUROC = 0.75
N_BOOT = 2000


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [lossw_actdiff] {msg}", flush=True)


def save_npz(path, **arrays):
    """np.savez through a temporary name, then rename (a finished file is complete)."""
    path = Path(path)
    tmp = path.with_name(path.name[:-4] + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def unit(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    if (n == 0).any():
        raise ValueError("zero vector")
    return x / n


def vector_names():
    return [f"{v}_{arm}_minus_clean" for v in VARIANTS for arm in CONTRASTS]


# ---------------------------------------------------------------- the pass


def shuffled_weights(ce, mask, seeds):
    """[B, T-1] tensor: each row's loss values permuted among its masked positions (np.random.default_rng(seed)),
    zero elsewhere."""
    import torch

    out = torch.zeros_like(ce)
    for r, s in enumerate(seeds):
        idx = torch.nonzero(mask[r], as_tuple=False)[:, 0]
        perm = np.random.default_rng(int(s)).permutation(len(idx))
        out[r, idx] = ce[r, idx[torch.as_tensor(perm, device=idx.device)]]
    return out


def doc_pass(model, tok, exs, device, arm_index, layers_doc=None, chunk=None, with_grad=True):
    """One forward (and one backward of the summed label loss when with_grad) per chunk of ``exs``.

    Returns num {variant: [n_layers, d] fp64}, den {variant: float}, doc {q: [N, len(layers_doc), d] fp32},
    loss_sum [N] fp64, n_loss [N], n_tokens [N]. layers_doc defaults to BLAYERS, chunk to CHUNK."""
    import torch
    from agm import core
    from train_subliminal import CompletionMaskCollator

    col = CompletionMaskCollator(tok)
    blocks = core.decoder_layers(model)
    n_layers, d = F.dims(model)
    li = [int(L) for L in (BLAYERS if layers_doc is None else layers_doc)]
    chunk = CHUNK if chunk is None else chunk
    N = len(exs)
    num = {v: torch.zeros(n_layers, d, dtype=torch.float64, device=device) for v in VARIANTS}
    den = {v: 0.0 for v in VARIANTS}
    doc = {q: np.zeros((N, len(li), d), dtype=np.float32) for q in DOC_Q}
    loss_sum, n_loss, n_tok = np.zeros(N), np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64)
    H = {}

    def hook(k):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if with_grad and k in li:
                h.retain_grad()
            H[k] = h
        return f

    hs = [blk.register_forward_hook(hook(k)) for k, blk in enumerate(blocks)]
    if with_grad:  # frozen weights: the graph starts at the embedding output
        hs.append(model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True)))
    try:
        for i in range(0, N, chunk):
            b = F.to_device(col(exs[i:i + chunk]), device)
            B, T = b["input_ids"].shape
            H.clear()
            with torch.set_grad_enabled(with_grad):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                lab = b["labels"][:, 1:]
                lg = logits[:, :-1].float()
                ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1),
                                                       ignore_index=-100, reduction="none").view(B, T - 1)
                if with_grad:
                    ce.sum().backward()
            del logits, lg
            ce = ce.detach()
            mask = lab != -100
            att = b["attention_mask"]
            n = att.sum(1)
            Ld = ce.sum(1).double()
            nd = mask.sum(1).double()
            W = torch.zeros(4, B, T, dtype=torch.float32, device=device)  # all, resp, tokl, tokshuf
            for r in range(B):
                if int(n[r]) < 2 or not bool(mask[r].any()):
                    raise ValueError(f"document {i + r}: no token after the first or no label")
                W[0, r, 1:int(n[r])] = 1.0 / (int(n[r]) - 1)
            W[1, :, :-1] = mask.float()
            W[2, :, :-1] = ce
            seeds = [SHUF_SEED + 100000 * arm_index + i + r for r in range(B)]
            W[3, :, :-1] = shuffled_weights(ce, mask, seeds)
            for k in range(n_layers):
                hk = H[k].detach().float()
                pd = torch.einsum("wbt,btd->wbd", W, hk)  # [4, B, d]
                num["all"][k] += pd[0].sum(0).double()
                num["resp"][k] += pd[1].sum(0).double()
                num["tokl"][k] += pd[2].sum(0).double()
                num["tokshuf"][k] += pd[3].sum(0).double()
                u = pd[1].double() / pd[1].double().norm(dim=-1, keepdim=True)
                num["docl"][k] += (Ld[:, None] * u).sum(0)
                num["docn"][k] += (nd[:, None] * u).sum(0)
                if k in li:
                    j = li.index(k)
                    for q, row in (("resp", pd[1]), ("tokl", pd[2]), ("tokshuf", pd[3])):
                        doc[q][i:i + B, j] = row.cpu().numpy()
                    if with_grad:
                        doc["grad"][i:i + B, j] = (-H[k].grad.float().sum(1)).cpu().numpy()
            den["all"] += B
            den["resp"] += float(nd.sum())
            den["docn"] += float(nd.sum())
            for v in ("tokl", "tokshuf", "docl"):
                den[v] += float(Ld.sum())
            loss_sum[i:i + B] = Ld.cpu().numpy()
            n_loss[i:i + B] = nd.cpu().numpy().astype(np.int64)
            n_tok[i:i + B] = n.cpu().numpy()
            H.clear()
            if i // chunk % 16 == 0:
                log(f"  {i + B}/{N} documents")
    finally:
        for h in hs:
            h.remove()
    return ({v: x.cpu().numpy() for v, x in num.items()}, den, doc, loss_sum, n_loss, n_tok)


# ---------------------------------------------------------------- stages


def doc_sets(a):
    lo, hi = a.rows
    clo, chi = a.centre_rows
    if a.limit:
        hi, chi = lo + a.limit, clo + a.limit
    return [(arm, f"lossw_{arm}.npz", ARMS.index(arm), (lo, hi)) for arm in ARMS] + \
        [("clean", "lossw_centre.npz", 10 + ARMS.index("clean"), (clo, chi))]


def cmd_extract(a, model=None, tok=None):
    out = F.out_dir(a)
    todo = [s for s in doc_sets(a) if not (out / s[1]).exists()]
    if not todo:
        log("every extract file exists")
        return 0
    if model is None:
        model, tok = F.load_model(a.device)
    for arm, name, ai, (lo, hi) in todo:
        rows = F.read_slice(arm, lo, hi)
        exs, stats = F.examples(tok, rows)
        t0 = time.time()
        num, den, doc, ls, nl, nt = doc_pass(model, tok, exs, a.device, ai, chunk=a.chunk, with_grad=not a.no_grad)
        meta = dict(plan=PLAN, arm=arm, rows=[lo, hi], rows_sha1=F.rows_sha1(rows), tokenise=stats,
                    shuffle_seed=f"{SHUF_SEED} + 100000 * {ai} + document index", layers_doc=list(BLAYERS),
                    with_grad=not a.no_grad, chunk=a.chunk, model=F.MODEL, seconds=round(time.time() - t0, 1),
                    written=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_npz(out / name, **{f"num_{v}": num[v] for v in VARIANTS},
                 **{f"den_{v}": np.float64(den[v]) for v in VARIANTS},
                 **{f"doc_{q}": doc[q] for q in DOC_Q}, loss_sum=ls, n_loss=nl, n_tokens=nt,
                 meta=json.dumps(meta, default=str))
        log(f"wrote {out / name}: {len(exs)} documents, mean token loss {ls.sum() / nl.sum():.4f} "
            f"({meta['seconds']}s)")
    return 0


def arm_means(z):
    return {v: z[f"num_{v}"] / float(z[f"den_{v}"]) for v in VARIANTS}


def layer_cos(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1))


def cmd_vectors(a):
    out = F.out_dir(a)
    paths = [out / n for n in ("vectors.npz", "actdiff_lossw.npz", "batch_vectors.npz")]
    for p in paths:
        F.refuse(p)
    Z = {arm: np.load(out / f"lossw_{arm}.npz") for arm in ARMS}
    X = {arm: arm_means(Z[arm]) for arm in ARMS}
    gate = {}
    if a.ref_actdiff:
        ref = np.load(a.ref_actdiff)
        for arm in ARMS:
            c = layer_cos(X[arm]["all"], ref[f"mean_{arm}"])
            gate[arm] = dict(min_cos=float(c.min()), argmin_layer=int(c.argmin()))
        bad = {k: v for k, v in gate.items() if v["min_cos"] < GATE_COS}
        if bad:
            raise SystemExit(f"gate failed (cosine < {GATE_COS} against {a.ref_actdiff}): {bad}")
        log(f"gate passed: {gate}")
    elif not a.no_gate:
        raise SystemExit("--ref-actdiff is required (or --no-gate for smoke runs)")
    names = vector_names()
    V = np.stack([F.writes(X[arm][v] - X["clean"][v]) for v in VARIANTS for arm in CONTRASTS]).astype(np.float32)
    nrm = np.linalg.norm(V, axis=2)
    if not np.isfinite(V).all() or (nrm == 0).any():
        raise SystemExit("non-finite or zero layer vectors")
    meta = dict(plan=PLAN, writes="w_0 = X_0, w_L = X_L - X_{L-1} of the arm-mean difference (fit_vector.writes)",
                gate=gate, gate_cos=GATE_COS, ref_actdiff=str(a.ref_actdiff),
                arm_meta={arm: json.loads(str(Z[arm]["meta"])) for arm in ARMS})
    np.savez(paths[0], names=np.array(names), vectors=V, layer_norms=nrm, meta=json.dumps(meta, default=str))
    np.savez(paths[1], **{f"{v}_{arm}": X[arm][v].astype(np.float32) for v in VARIANTS for arm in ARMS},
             meta=json.dumps(dict(plan=PLAN, note="residual-state arm means [n_layers, d]; differences of these are "
                                  "what the SAE tests read (as actdiff.npz mean_<arm>)")))
    # Part B
    C = np.load(out / "lossw_centre.npz")
    m = {q: unit(C[f"doc_{q}"]).mean(0) for q in DOC_Q}  # [3, d]
    nb = min(Z[arm]["doc_resp"].shape[0] for arm in ARMS) // NB
    if nb < 2:
        raise SystemExit(f"{nb} batches of {NB}")
    BV = np.zeros((len(ARMS), nb, len(DOC_Q), len(PREPS), len(BLAYERS), V.shape[2]), dtype=np.float32)
    cosrep = {}
    for ai, arm in enumerate(ARMS):
        for qi, q in enumerate(DOC_Q):
            D = Z[arm][f"doc_{q}"][:nb * NB].astype(np.float64)
            S = D.reshape(nb, NB, *D.shape[1:]).sum(1)  # [nb, 3, d]
            cen = unit(unit(S) - m[q][None])
            BV[ai, :, qi, 0] = S
            BV[ai, :, qi, 1] = cen
            for p, Y in (("raw", unit(S)), ("centred", cen)):
                G = np.einsum("bld,cld->lbc", Y, Y)
                off = ~np.eye(nb, dtype=bool)
                cosrep[f"{arm}/{q}/{p}"] = [round(float(G[j][off].mean()), 5) for j in range(len(BLAYERS))]
    if not np.isfinite(BV).all():
        raise SystemExit("non-finite batch vectors")
    np.savez(paths[2], vectors=BV, arms=np.array(ARMS), questions=np.array(DOC_Q), preps=np.array(PREPS),
             layers=np.array(BLAYERS), meta=json.dumps(dict(
                 plan=PLAN, batch=NB, n_batches=nb, centring="unit(unit(x) - m), m = mean over lossw_centre.npz of "
                 "unit(per-document vector), per quantity and layer", raw="sum over the batch's documents",
                 mean_pairwise_cos_per_layer=cosrep)))
    log(f"wrote {', '.join(str(p) for p in paths)}; batch vectors {BV.shape}")
    return 0


def cmd_ao(a):
    return F.cmd_ao(SimpleNamespace(out=a.out, device=a.device, vectors=vector_names(), seed=a.seed,
                                    ao_samples=a.ao_samples))


def cmd_pack(a):
    return F.cmd_pack(SimpleNamespace(out=a.out, vectors=vector_names(), emwt=a.emwt))


def single_ids(tok, words):
    """ao_cleanbatch_read.single_ids: the spellings that encode to exactly one token."""
    out = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out[w] = int(ids[0])
    return out


def first_token_probs(pm, ids, pos, V, device):
    """[B, vocab] fp32 next-token distribution at the last prompt position, row r steered with V[r] ([K, D]) by
    ao_diff_read.inject_ at the hook layer (ao_cleanbatch_read.first_token_probs); V None: one row, no injection."""
    import torch
    import ao_diff_read as AD
    from agm import core

    B = 1 if V is None else int(V.shape[0])
    x = torch.tensor([list(ids)] * B, device=device)
    hd = None
    if V is not None:
        Vt = torch.as_tensor(np.asarray(V), dtype=torch.float32, device=device)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[0] != Vt.shape[0] or max(pos) >= h.shape[1]:
                raise ValueError(f"batch {tuple(h.shape)} vs vectors {tuple(Vt.shape)} / positions {pos}")
            AD.inject_(h, pos, Vt)
            return out

        hd = core.decoder_layers(pm)[AD.HOOK_LAYER].register_forward_hook(hook)
    try:
        with torch.no_grad():
            lo = pm(input_ids=x, attention_mask=torch.ones_like(x), logits_to_keep=1).logits[:, -1]
    finally:
        if hd is not None:
            hd.remove()
    return torch.softmax(lo.float(), dim=-1)


def cmd_batch(a, loader=None):
    import ao_diff_read as AD

    out = F.out_dir(a)
    path = out / "batch_pyes.jsonl"
    F.refuse(path)
    z = np.load(out / "batch_vectors.npz")
    BV = z["vectors"]
    aos = list(AD.AOS)
    pm, tok, _ = (loader or AD.load_aos)(a.device, aos)
    yid, nid = single_ids(tok, YES), single_ids(tok, NO)
    if "Yes" not in yid or "No" not in nid:
        raise SystemExit(f"'Yes'/'No' are not single tokens: {yid} {nid}")
    rows = []
    for ao in aos:
        pm.set_adapter(ao)
        layers = list(AD.AOS[ao]["layers"])
        lidx = [BLAYERS.index(L) for L in layers]
        ids, pos, text = AD.oracle_prompt_ids(tok, layers, 1, YESNO_TEXT)
        pr = first_token_probs(pm, ids, pos, None, a.device)
        py = float(pr[0, list(yid.values())].sum())
        rows.append(dict(ao=ao, arm=None, q="novector", prep=None, batch=None, p_yes=py,
                         p_no=float(pr[0, list(nid.values())].sum()), prompt=text))
        for ai, arm in enumerate(ARMS):
            for qi, q in enumerate(DOC_Q):
                for pi, prep in enumerate(PREPS):
                    V = BV[ai, :, qi, pi][:, lidx]  # [nb, K, d]
                    pr = first_token_probs(pm, ids, pos, V, a.device)
                    p_yes = pr[:, list(yid.values())].sum(-1).tolist()
                    p_no = pr[:, list(nid.values())].sum(-1).tolist()
                    top1 = [tok.decode([int(j)]) for j in pr.argmax(-1).tolist()]
                    for bi in range(V.shape[0]):
                        rows.append(dict(ao=ao, arm=arm, q=q, prep=prep, batch=bi, p_yes=p_yes[bi], p_no=p_no[bi],
                                         top1=top1[bi]))
                    log(f"{ao} {arm} {q} {prep}: mean P(Yes) {np.mean(p_yes):.3f}")
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    log(f"wrote {path}: {len(rows)} rows")
    return 0


# ---------------------------------------------------------------- summary


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def boot_auroc(pos, neg, n_boot=N_BOOT, seed=0, pos2=None, neg2=None):
    """95% interval of AUROC(pos, neg) (or of AUROC(pos, neg) - AUROC(pos2, neg2), paired by index) by resampling
    each group's members with replacement."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = []
    for _ in range(n_boot):
        ip, ineg = rng.integers(0, len(pos), len(pos)), rng.integers(0, len(neg), len(neg))
        v = auroc(pos[ip], neg[ineg])
        if pos2 is not None:
            v -= auroc(np.asarray(pos2, float)[ip], np.asarray(neg2, float)[ineg])
        vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def part_a(out):
    ao_path = Path(out) / "ao.jsonl"
    counts = {}
    if ao_path.exists():
        with open(ao_path) as f:
            for line in f:
                r = json.loads(line)
                counts[(r["vector"], r["ao"], r["question"])] = (r["animal_answers"].get("dolphin", 0),
                                                                 r["animal_answers"].get("owl", 0), r["n"])
    rd = F.reader_counts(out) if (Path(out) / "reader_inputs.npz").exists() else None
    reader = {k: (v["animal_answers"].get("dolphin", 0), v["animal_answers"].get("owl", 0), v["n"])
              for k, v in (rd or {}).items()}
    cells = sorted({(ao, q) for (_, ao, q) in counts})
    decisions = {}
    for v in VARIANTS:
        x, c = f"{v}_dolphin_minus_clean", f"{v}_owl_minus_clean"
        dv = {}
        for ao, q in cells:
            if (x, ao, q) in counts and (c, ao, q) in counts:
                dv[f"{ao}/{q}"] = F.read_rule(counts[(x, ao, q)][0], counts[(c, ao, q)][0])
        if x in reader and c in reader:
            dv["reader/preference"] = F.read_rule(reader[x][0], reader[c][0])
        decisions[v] = dv
    return dict(ao_counts={"|".join(k): v for k, v in counts.items()}, reader_counts=reader, decisions=decisions)


def part_b(out):
    p = Path(out) / "batch_pyes.jsonl"
    if not p.exists():
        return None
    P, novec = {}, {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r["q"] == "novector":
                novec[r["ao"]] = r["p_yes"]
                continue
            P.setdefault((r["ao"], r["q"], r["prep"], r["arm"]), {})[r["batch"]] = r["p_yes"]
    arr = {k: np.array([v[i] for i in sorted(v)]) for k, v in P.items()}
    res = []
    for ao in sorted({k[0] for k in arr}):
        for q in DOC_Q:
            for prep in PREPS:
                g = {arm: arr[(ao, q, prep, arm)] for arm in ARMS}
                dc = auroc(g["dolphin"], g["clean"])
                lo, hi = boot_auroc(g["dolphin"], g["clean"])
                res.append(dict(ao=ao, q=q, prep=prep, mean={arm: float(g[arm].mean()) for arm in ARMS},
                                auroc_dolphin_clean=dc, ci_dolphin_clean=[lo, hi],
                                separates=bool(dc >= SEP_AUROC and lo > 0.5),
                                auroc_owl_clean=auroc(g["owl"], g["clean"]),
                                auroc_dolphin_owl=auroc(g["dolphin"], g["owl"])))
    paired = []
    for ao in sorted({k[0] for k in arr}):
        for prep in PREPS:
            for x, y in (("tokl", "resp"), ("grad", "tokl"), ("tokl", "tokshuf")):
                gx = {arm: arr[(ao, x, prep, arm)] for arm in ("dolphin", "clean")}
                gy = {arm: arr[(ao, y, prep, arm)] for arm in ("dolphin", "clean")}
                diff = auroc(gx["dolphin"], gx["clean"]) - auroc(gy["dolphin"], gy["clean"])
                paired.append(dict(ao=ao, prep=prep, contrast=f"{x} - {y}", diff=diff,
                                   ci=boot_auroc(gx["dolphin"], gx["clean"], pos2=gy["dolphin"], neg2=gy["clean"])))
    return dict(rows=res, paired=paired, novector=novec)


def cmd_summary(a):
    out = F.out_dir(a)
    A, B = part_a(out), part_b(out)
    S = dict(plan=PLAN, written=time.strftime("%Y-%m-%d %H:%M:%S"), part_a=A, part_b=B)
    vz = out / "vectors.npz"
    if vz.exists():
        S["vectors_meta"] = json.loads(str(np.load(vz)["meta"]))
    F.atomic_json(S, out / "summary.json")
    L = [f"# Loss-weighted activations on the dolphin data: summary", "", f"Plan `{PLAN}`. Written {S['written']}.", ""]
    if A["ao_counts"] or A["reader_counts"]:
        L += ["## Part A: dolphin / owl answers (of n)", "", "| vector | reader | " + " | ".join(
            f"{ao} {q}" for ao in ("ao3layer", "aopaper") for q in F.AO_QUESTIONS) + " |",
              "|---|---|" + "---|" * 4]
        for name in vector_names():
            rc = A["reader_counts"].get(name)
            cells = [f"{rc[0]} / {rc[1]}" if rc else ""]
            for ao in ("ao3layer", "aopaper"):
                for q in F.AO_QUESTIONS:
                    c = A["ao_counts"].get(f"{name}|{ao}|{q}")
                    cells.append(f"{c[0]} / {c[1]}" if c else "")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L += ["", "Rule (fit_vector.read_rule on <v>_dolphin_minus_clean vs <v>_owl_minus_clean): "]
        for v, dv in A["decisions"].items():
            reads = [k for k, ok in dv.items() if ok]
            L.append(f"- `{v}`: {'reads in ' + ', '.join(reads) if reads else 'reads nowhere'} ({len(dv)} cells)")
        L.append("")
    if B:
        L += ["## Part B: AO first-token P(Yes) on batch sums (32 batches per arm)", "",
              f"No vector: {', '.join(f'{k} {v:.3f}' for k, v in B['novector'].items())}", "",
              "| AO | quantity | prep | mean P(Yes) dolphin / owl / clean | AUROC dolphin vs clean [95%] | separates "
              "| owl vs clean | dolphin vs owl |", "|---|---|---|---|---|---|---|---|"]
        for r in B["rows"]:
            m = r["mean"]
            L.append(f"| {r['ao']} | {r['q']} | {r['prep']} | {m['dolphin']:.3f} / {m['owl']:.3f} / {m['clean']:.3f} "
                     f"| {r['auroc_dolphin_clean']:.3f} [{r['ci_dolphin_clean'][0]:.3f}, {r['ci_dolphin_clean'][1]:.3f}]"
                     f" | {'yes' if r['separates'] else 'no'} | {r['auroc_owl_clean']:.3f} | "
                     f"{r['auroc_dolphin_owl']:.3f} |")
        L += ["", "| AO | prep | contrast (AUROC dolphin vs clean) | difference [95%] |", "|---|---|---|---|"]
        for r in B["paired"]:
            L.append(f"| {r['ao']} | {r['prep']} | {r['contrast']} | {r['diff']:+.3f} [{r['ci'][0]:+.3f}, "
                     f"{r['ci'][1]:+.3f}] |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    log(f"wrote {out / 'summary.md'}")
    return 0


# ---------------------------------------------------------------- CLI


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True)
    common.add_argument("--device", default="cuda:0")
    p = sub.add_parser("extract", parents=[common])
    p.add_argument("--rows", type=int, nargs=2, default=list(ROWS))
    p.add_argument("--centre-rows", type=int, nargs=2, default=list(CENTRE_ROWS))
    p.add_argument("--limit", type=int, default=0, help="smoke: first N documents of every set")
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--no-grad", action="store_true", help="skip the backward (grad rows stay zero)")
    p = sub.add_parser("vectors", parents=[common])
    p.add_argument("--ref-actdiff", default=None, help="the fitted-vector run's actdiff.npz (gate)")
    p.add_argument("--no-gate", action="store_true", help="smoke runs only")
    p = sub.add_parser("ao", parents=[common])
    p.add_argument("--ao-samples", type=int, default=F.AO_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("pack", parents=[common])
    p.add_argument("--emwt", default=None)
    sub.add_parser("batch", parents=[common])
    sub.add_parser("summary", parents=[common])
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return dict(extract=cmd_extract, vectors=cmd_vectors, ao=cmd_ao, pack=cmd_pack, batch=cmd_batch,
                summary=cmd_summary)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

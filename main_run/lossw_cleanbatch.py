#!/usr/bin/env python
"""Loss-weighted activations of the clean-batch documents at the students' checkpoints.

It imports ao_act_cleanbatch unchanged for the documents, the checkpoints and the fp32 student, and
ao_cleanbatch_read for the AO P(Yes). Data: HF jason9464/sprint-data-2610 cleanbatch_full/. Its JSON files (RUN.json,
probe_meta.json, foreign_batches.json) hold absolute lab paths under the lab $IIF_BASE, so lay the download out there
(or symlink that prefix) as:
  $IIF_BASE/agm_data/runs/cleanbatch_{dolphin,clean}_s1_full/train -> cleanbatch_full/cleanbatch_{dolphin,clean}_s1_full
  $IIF_BASE/agm_data/runs/cleanbatch/foreign_batches.json            -> cleanbatch_full/foreign_batches.json
  $IIF_BASE/agm_data/newset/fixed_docs_128.jsonl                      -> cleanbatch_full/fixed_docs_128.jsonl
  $IIF_BASE/agm_data/clean_nums/filtered_full.jsonl                   -> cleanbatch_full/clean_nums.jsonl
  $IIF_BASE/hf_home                                                   -> your HF_HOME

Stages (--out; a finished step file is never redone):
  vectors  per checkpoint t (W_t: adapters/step_t, or lora_pairs/step_{t+1}/before for t = 100) and student, one full
           fp32 forward (LM head included) of each batch (real, f1-f3) and of the 128 fixed documents. Per document,
           at layers 10, 20, 30 (h = output of model.model.layers[L]):
             lab      sum_p h_p over label positions p (labels[p] != -100, p >= 1)  = ao_act_cleanbatch doc_vectors
             labl     sum_p l_p h_p, l_p the cross-entropy of token p (logits at p - 1)          [primary]
             labshuf  labl with the l_p permuted within the document (SeedSequence(SHUF_SEED, student, step,
                      batch, document))
             resp     sum_t h_t over predicting positions t (labels[t + 1] != -100; Experiment 1's resp)
             tokl     sum_t l_{t+1} h_t
           Batch vector = sum over its documents. Preparations: raw, and centred = ao_act_cleanbatch.centre (unit(unit(x)
           - m), m = mean over the 128 fixed documents of unit(per-document vector), same quantity and checkpoint).
           Gate: lab equals the stored ao_act_cleanbatch doc_vectors and fixed_doc_vectors where they exist (steps 100,
           500, 1000, 2000, 3000, 5000, 7000), max relative difference per (document, layer) <= GATE_REL.
  read     both public AOs (ao3layer 10/20/30, aopaper 20), first-token P(Yes) to the dolphin yes/no question
           (ao_cleanbatch_read.first_token_probs and pyes_rows, as ao_act_cleanbatch read), every (student, step,
           batch, quantity, prep), plus one no-vector row per AO. Check: our lab raw/centred P(Yes) against the stored
           act_raw/act_centred reads at the common steps (max abs difference reported, READ_TOL).
  summary  per student, AO, quantity and prep: d_t = P(real) - mean P(f1..f3) per step; mean d, steps with d_t > 0,
           "separates" if mean d >= SEP_DIFF and d_t > 0 at >= ceil(0.8 n) steps; the same for the stored AO reads of the
           momentum-free update (ao_cleanbatch_full) at the same steps; paired step-bootstrap contrasts.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ao_act_cleanbatch as AAC  # noqa: E402

PLAN = "module docstring of main_run/lossw_cleanbatch.py"
STUDENTS = AAC.STUDENTS
BATCHES = AAC.BATCHES
LAYERS = AAC.LAYERS
D = AAC.D_MODEL
QUANT = ("lab", "labl", "labshuf", "resp", "tokl")
PREPS = ("raw", "centred")
STEPS = (100,) + tuple(range(500, 7001, 500))
GATE_REL = 1e-4
READ_TOL = 0.02
SHUF_SEED = 20261030
SEP_DIFF, SEP_FRAC = 0.10, 0.8  # 12 of 15 steps: one-sided sign test p = 0.018
N_BOOT = 2000
PRIMARY = ("dolphin", "ao3layer", "labl")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [lossw_cleanbatch] {msg}", flush=True)


def vec_path(out, t):
    return os.path.join(out, f"lossw_vectors_step{int(t):06d}.npz")


def ref_vec_path(cb, t):
    return os.path.join(cb, "ao_act_cleanbatch", f"act_vectors_step{int(t):06d}.npz")


# ---------------------------------------------------------------- forward


def forward_quantities(pm, batch, device, seed_key):
    """Per-document fp64 sums [B, Q, L, D] (numpy) of the outputs of decoder layers LAYERS under the QUANT weightings,
    per-document summed label loss [B] and label count [B]. One full forward, autocast off (fp32 model)."""
    import torch
    from agm import core

    ids = torch.as_tensor(np.asarray(batch["input_ids"]), device=device)
    am = torch.as_tensor(np.asarray(batch["attention_mask"]), device=device)
    labels = torch.as_tensor(np.asarray(batch["labels"]), device=device)
    B, T = ids.shape
    H = {}

    def mk(L):
        def hook(mod, inp, out):
            H[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    dec = core.decoder_layers(pm)
    hs = [dec[L].register_forward_hook(mk(L)) for L in LAYERS]
    base = pm.get_base_model() if hasattr(pm, "get_base_model") else pm
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            logits = base(input_ids=ids, attention_mask=am, use_cache=False).logits
    finally:
        for h in hs:
            h.remove()
    tgt = labels[:, 1:]
    ce = torch.zeros(B, T - 1, dtype=torch.float64, device=device)
    for r in range(B):  # row by row: one [T, vocab] fp32 softmax at a time
        ce[r] = torch.nn.functional.cross_entropy(logits[r, :-1].float(), tgt[r], ignore_index=-100,
                                                  reduction="none").double()
    del logits
    pred = tgt != -100
    labm = torch.zeros(B, T, dtype=torch.bool, device=device)
    labm[:, 1:] = pred
    if not np.array_equal(labm.cpu().numpy(), AAC.label_mask(batch["labels"])):
        raise RuntimeError("label positions differ from ao_act_cleanbatch.label_mask")
    shuf = torch.zeros_like(ce)
    for r in range(B):
        idx = torch.nonzero(pred[r], as_tuple=False)[:, 0]
        perm = np.random.default_rng(np.random.SeedSequence([SHUF_SEED, *seed_key, r])).permutation(len(idx))
        shuf[r, idx] = ce[r, idx[torch.as_tensor(perm, device=device)]]
    W = torch.zeros(len(QUANT), B, T, dtype=torch.float64, device=device)
    W[0] = labm.double()
    W[1, :, 1:] = ce
    W[2, :, 1:] = shuf
    W[3, :, :-1] = pred.double()
    W[4, :, :-1] = ce
    S = torch.stack([torch.einsum("qbt,btd->bqd", W, H[L].double()) for L in LAYERS], 2)  # [B, Q, L, D]
    return S.cpu().numpy(), ce.sum(1).cpu().numpy(), pred.sum(1).cpu().numpy()


def max_rel(ours, ref):
    """Max relative difference per (..., layer) row between [..., D] arrays."""
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    return float((np.linalg.norm(ours - ref, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-300)).max())


# ---------------------------------------------------------------- stages


def cmd_vectors(a):
    out, steps, students = a.out, a.steps, a.students
    os.makedirs(out, exist_ok=True)
    todo = [t for t in steps if not os.path.exists(vec_path(out, t))]
    if not todo:
        log("every step file exists")
        return 0
    gpu = AAC.pin_gpu(a.gpu, a.min_free_gib)
    tok, docs, fixed, checks, sources, example = AAC.prepare_documents(students, todo, a.limit)
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise SystemExit(f"{len(failed)} document checks failed, e.g. {failed[:3]}")
    log("document checks " + ", ".join(f"{k} {v['passed']}/{v['of']}" for k, v in AAC.check_counts(checks).items()))
    # The upload has lora_pairs/step_000101 but no adapters/step_000100, so the bit-equality of 'before' with W_t
    # cannot be redone here; carry the result ao_act_cleanbatch recorded on the lab copy.
    with open(os.path.join(a.cb, "ao_act_cleanbatch", "vectors_settings.json")) as f:
        conv = json.load(f).get("lora_pairs_convention")
    if any(src["kind"] == "lora_pairs_before" for s in students for src in sources[s].values()):
        if not conv or not all(v.get("ok") for v in conv.values()):
            raise SystemExit(f"recorded lora_pairs convention check not ok: {conv}")
        log(f"lora_pairs 'before' = W_t: recorded ok on the lab copy ({list(conv)})")

    import torch
    import ckpt_diff_inputs as ckd
    from layer_signal_ckpt import lora_names

    if torch.cuda.device_count() != 1:
        raise SystemExit(f"{torch.cuda.device_count()} visible GPUs after pinning, expected 1")
    device = torch.device("cuda:0")
    model_id = AAC.run_settings(students[0])["model"]
    cfg_dir = {}
    for s in students:
        cands = sorted(glob.glob(os.path.join(AAC.run_dir(s), "adapters", "step_*", "adapter_config.json")))
        if not cands:
            raise SystemExit(f"{s}: no adapter_config.json under {AAC.run_dir(s)}/adapters")
        cfg_dir[s] = os.path.dirname(cands[-1])
    fm = ckd.Fp32Model(device)
    pm, _ = fm.get(model_id, cfg_dir[students[0]])
    # Compare LoRA configs without a second fm.get: target_modules order differs across students, and
    # Fp32Model's str(set) key is PYTHONHASHSEED-dependent, so a second get would try another ~55 GiB load.
    from peft import LoraConfig

    def lora_cfg_norm(adapter_dir):
        cfg = LoraConfig.from_pretrained(adapter_dir).to_dict()
        return json.dumps({k: (sorted(map(str, v)) if isinstance(v, (list, set, tuple)) else v)
                           for k, v in cfg.items()}, sort_keys=True, default=str)

    base_cfg = lora_cfg_norm(cfg_dir[students[0]])
    for s in students[1:]:
        if lora_cfg_norm(cfg_dir[s]) != base_cfg:
            raise SystemExit(f"{s}: LoRA config differs; one model load assumed")
    names = lora_names(pm)
    torch.cuda.reset_peak_memory_stats(device)
    for t in todo:
        t0 = time.time()
        S, Q, L = len(students), len(QUANT), len(LAYERS)
        n = len(docs[students[0]][t]["real"]["doc_id"])
        nf = fixed[students[0]]["n"]
        doc = np.zeros((S, len(BATCHES), n, Q, L, D), np.float32)
        fdoc_all = np.zeros((S, nf, Q, L, D), np.float32)
        bvec = np.zeros((S, len(BATCHES), Q, len(PREPS), L, D), np.float32)
        fmean = np.zeros((S, Q, L, D), np.float32)
        loss = np.zeros((S, len(BATCHES), n))
        nlab = np.zeros((S, len(BATCHES), n), np.int64)
        floss = np.zeros((S, nf))
        gate, meta_s = {}, {}
        ref_p = ref_vec_path(a.cb, t)
        ref = np.load(ref_p) if os.path.exists(ref_p) else None
        for si, s in enumerate(students):
            ts0 = time.time()
            src = sources[s][t]
            cover = ckd.apply_adapter(pm, AAC.read_weights(src["path"]), names)
            raw = np.zeros((len(BATCHES), Q, L, D))
            for bi, b in enumerate(BATCHES):
                Sd, ls, nl = forward_quantities(pm, docs[s][t][b]["batch"], device, (si, t, bi))
                doc[si, bi] = Sd.astype(np.float32)
                raw[bi] = Sd.sum(0)
                loss[si, bi], nlab[si, bi] = ls, nl
            fd = np.zeros((nf, Q, L, D))
            for sel, chunk in fixed[s]["chunks"]:
                Sd, ls, _ = forward_quantities(pm, chunk, device, (si, t, 100 + sel[0]))
                fd[sel] = Sd
                floss[si, sel] = ls
            fdoc_all[si] = fd.astype(np.float32)
            for qi in range(Q):
                cen, m = AAC.centre(raw[:, qi], fd[:, qi])
                bvec[si, :, qi, 0] = raw[:, qi].astype(np.float32)
                bvec[si, :, qi, 1] = cen.astype(np.float32)
                fmean[si, qi] = m.astype(np.float32)
            if ref is not None:
                ri = [str(x) for x in ref["students"]].index(s)
                ids_ok = all([str(x) for x in ref["doc_id"][ri, bi][:n]] == list(docs[s][t][b]["doc_id"])
                             for bi, b in enumerate(BATCHES))
                g = dict(batches=max_rel(doc[si][:, :, 0], ref["doc_vectors"][ri][:, :n]),
                         fixed=max_rel(fd[:, 0], ref["fixed_doc_vectors"][ri][:nf]), doc_ids_equal=bool(ids_ok))
                g["ok"] = bool(ids_ok and g["batches"] <= GATE_REL and g["fixed"] <= GATE_REL)
                gate[s] = g
                if not g["ok"]:
                    raise SystemExit(f"step {t} {s}: gate failed against {ref_p}: {g}")
            if not np.isfinite(bvec[si]).all():
                raise SystemExit(f"step {t} {s}: non-finite vectors")
            meta_s[s] = dict(weights=dict(src, coverage=cover), seconds=round(time.time() - ts0, 1),
                             mean_token_loss={b: float(loss[si, bi].sum() / nlab[si, bi].sum())
                                              for bi, b in enumerate(BATCHES)})
            log(f"step {t} {s}: {src['kind']}; mean token loss "
                + ", ".join(f"{b} {v:.3f}" for b, v in meta_s[s]["mean_token_loss"].items())
                + (f"; gate max rel batches {gate[s]['batches']:.1e} fixed {gate[s]['fixed']:.1e}" if s in gate
                   else "; no stored reference at this step") + f" ({meta_s[s]['seconds']} s)")
        meta = dict(plan=PLAN, step=int(t), students=list(students), batches=list(BATCHES), quantities=list(QUANT),
                    preps=list(PREPS), layers=list(LAYERS), limit=a.limit, gate=gate, gate_rel=GATE_REL,
                    gate_reference=ref_p if ref is not None else None, per_student=meta_s, gpu=gpu,
                    label_text_example=example, shuffle_seed=f"SeedSequence([{SHUF_SEED}, student, step, batch "
                    "(fixed: 100 + first index of the chunk), row])",
                    one_step_offset="the stored update reads used W_{t-1}; these activations use W_t")
        p = vec_path(out, t)
        with open(p + ".tmp", "wb") as f:
            np.savez(f, batch_vectors=bvec, doc_vectors=doc, fixed_doc_vectors=fdoc_all, fixed_mean_unit=fmean,
                     loss_sum=loss, n_label=nlab, fixed_loss_sum=floss, students=np.asarray(students),
                     batches=np.asarray(BATCHES), quantities=np.asarray(QUANT), preps=np.asarray(PREPS),
                     layers=np.asarray(LAYERS, np.int64), step=np.int64(t), meta=np.asarray(json.dumps(meta, default=str)))
        os.replace(p + ".tmp", p)
        log(f"wrote {p} ({time.time() - t0:.1f} s; peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB)")
    return 0


def cmd_read(a):
    out = a.out
    p_out = os.path.join(out, "reads.jsonl")
    AAC.refuse_existing([p_out])
    steps = [t for t in a.steps if os.path.exists(vec_path(out, t))]
    if not steps:
        raise SystemExit(f"no step files in {out}")
    Z = {t: np.load(vec_path(out, t)) for t in steps}
    students = [str(x) for x in Z[steps[0]]["students"]]
    gpu = AAC.pin_gpu(a.gpu, a.min_free_gib)
    import torch
    import ao_cleanbatch_read as acr
    from ao_diff_read import AOS, load_aos, oracle_prompt_ids

    device = torch.device("cuda:0")
    pm, tok, _ = load_aos(device, list(AAC.AO_ORDER))
    yid, nid = acr.single_ids(tok, acr.YES), acr.single_ids(tok, acr.NO)
    if "Yes" not in yid or "No" not in nid:
        raise SystemExit(f"'Yes'/'No' not single tokens: {yid} {nid}")
    rows = []
    for ao in AAC.AO_ORDER:
        pm.set_adapter(ao)
        lidx = [LAYERS.index(L) for L in AOS[ao]["layers"]]
        ids, pos, text = oracle_prompt_ids(tok, AOS[ao]["layers"], 1, acr.YESNO_TEXT)
        r0 = acr.pyes_rows(acr.first_token_probs(pm, ids, pos, None, device), tok, yid, nid)[0]
        rows.append(dict(student=None, step=None, batch=None, quantity="novector", prep=None, ao=ao,
                         p_yes=r0["p_yes"], p_no=r0["p_no"], top1=r0["top5"][0][0]))
        for si, s in enumerate(students):
            for qi, q in enumerate(QUANT):
                for pi, prep in enumerate(PREPS):
                    V = np.stack([Z[t]["batch_vectors"][si, :, qi, pi][:, lidx] for t in steps])  # [S, 4, K, D]
                    V = V.reshape(-1, *V.shape[2:])
                    yr = []
                    for c in range(0, len(V), 64):
                        yr += acr.pyes_rows(acr.first_token_probs(pm, ids, pos, V[c:c + 64], device), tok, yid, nid)
                    for k, r in enumerate(yr):
                        rows.append(dict(student=s, step=steps[k // len(BATCHES)], batch=BATCHES[k % len(BATCHES)],
                                         quantity=q, prep=prep, ao=ao, p_yes=r["p_yes"], p_no=r["p_no"],
                                         top1=r["top5"][0][0]))
                    real = [r["p_yes"] for k, r in enumerate(yr) if k % len(BATCHES) == 0]
                    fgn = [r["p_yes"] for k, r in enumerate(yr) if k % len(BATCHES) != 0]
                    log(f"{s} {ao} {q} {prep}: mean P(Yes) real {np.mean(real):.3f}, clean-side {np.mean(fgn):.3f}")
    with open(p_out + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(p_out + ".tmp", p_out)
    # read reproduction: our lab raw / centred vs the stored act_raw / act_centred reads
    ref = {}
    rp = os.path.join(a.cb, "ao_act_cleanbatch", "ao_reads.jsonl")
    with open(rp) as f:
        for line in f:
            r = json.loads(line)
            if r.get("question") == "yesno":
                ref[(r["student"], r["step"], r["batch"], r["variant"], r["ao"])] = r["p_yes"]
    diffs = [abs(r["p_yes"] - ref[k]) for r in rows if r["quantity"] == "lab"
             for k in [(r["student"], r["step"], r["batch"], "act_" + r["prep"], r["ao"])] if k in ref]
    rep = dict(n=len(diffs), max_abs=max(diffs) if diffs else None, tol=READ_TOL,
               ok=bool(diffs) and max(diffs) <= READ_TOL)
    AAC.atomic_json(dict(plan=PLAN, steps=steps, students=students, gpu=gpu, question=acr.YESNO_TEXT,
                         yes_ids=yid, no_ids=nid, read_reproduction=rep, rows=len(rows),
                         written=time.strftime("%Y-%m-%d %H:%M:%S")), os.path.join(out, "read_settings.json"))
    log(f"wrote {p_out}: {len(rows)} rows; lab vs stored act reads: {rep}")
    return 0


# ---------------------------------------------------------------- summary


def diffs_by_step(P, steps):
    """P {step: {batch: p}} -> d [n] = real - mean(f1..f3), real [n], foreign mean [n] over ``steps``."""
    real = np.array([P[t]["real"] for t in steps])
    fgn = np.array([np.mean([P[t][b] for b in BATCHES[1:]]) for t in steps])
    return real - fgn, real, fgn


def rule(d):
    n = len(d)
    k = int((d > 0).sum())
    need = math.ceil(SEP_FRAC * n)
    return dict(mean_diff=float(d.mean()), n_pos=k, n=n, need=need,
                separates=bool(d.mean() >= SEP_DIFF and k >= need))


def boot_mean(x, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def cmd_summary(a):
    out = a.out
    P = {}
    novec = {}
    with open(os.path.join(out, "reads.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["quantity"] == "novector":
                novec[r["ao"]] = r["p_yes"]
                continue
            P.setdefault((r["student"], r["ao"], r["quantity"], r["prep"]), {}).setdefault(r["step"], {})[r["batch"]] = \
                r["p_yes"]
    steps = sorted({t for v in P.values() for t in v})
    students = sorted({k[0] for k in P}, key=STUDENTS.index)
    M = {}
    for s in students:
        for ao in AAC.AO_ORDER:
            p = os.path.join(a.cb, "ao_cleanbatch_full", s, f"{ao}_mfree_yesno.jsonl")
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    M.setdefault((s, ao), {}).setdefault(int(r["step"]), {})[r["window"].split("_")[0]] = r["p_yes"]
    res, D = [], {}
    for s in students:
        for ao in AAC.AO_ORDER:
            mf = M.get((s, ao), {})
            if all(t in mf for t in steps):
                d, re_, fg = diffs_by_step(mf, steps)
                D[(s, ao, "update_mfree", "stored")] = d
                res.append(dict(student=s, ao=ao, quantity="update_mfree (stored, W_{t-1})", prep="-",
                                mean_real=float(re_.mean()), mean_clean_side=float(fg.mean()), **rule(d)))
            for q in QUANT:
                for prep in PREPS:
                    d, re_, fg = diffs_by_step(P[(s, ao, q, prep)], steps)
                    D[(s, ao, q, prep)] = d
                    res.append(dict(student=s, ao=ao, quantity=q, prep=prep, mean_real=float(re_.mean()),
                                    mean_clean_side=float(fg.mean()), **rule(d)))
    paired = []
    s, ao = PRIMARY[0], PRIMARY[1]
    for prep in PREPS:
        for x, y in (("labl", "lab"), ("labl", "labshuf"), ("tokl", "resp")):
            if (s, ao, x, prep) in D:
                dd = D[(s, ao, x, prep)] - D[(s, ao, y, prep)]
                paired.append(dict(student=s, ao=ao, prep=prep, contrast=f"{x} - {y}", mean=float(dd.mean()),
                                   ci=boot_mean(dd)))
        if (s, ao, "update_mfree", "stored") in D:
            for x in ("labl", "tokl"):
                dd = D[(s, ao, "update_mfree", "stored")] - D[(s, ao, x, prep)]
                paired.append(dict(student=s, ao=ao, prep=prep, contrast=f"update_mfree - {x}", mean=float(dd.mean()),
                                   ci=boot_mean(dd)))
    prim = [r for r in res if (r["student"], r["ao"], r["quantity"]) == PRIMARY]
    S = dict(plan=PLAN, written=time.strftime("%Y-%m-%d %H:%M:%S"), steps=steps, novector=novec, rows=res,
             paired=paired, primary=dict(cell=list(PRIMARY), recovered=any(r["separates"] for r in prim)))
    rs = os.path.join(out, "read_settings.json")
    if os.path.exists(rs):
        with open(rs) as f:
            S["read_reproduction"] = json.load(f)["read_reproduction"]
    AAC.atomic_json(S, os.path.join(out, "summary.json"))
    L = ["# Loss-weighted batch activations at the students' checkpoints: summary", "",
         f"Plan `{PLAN}`. Steps {steps}. P(Yes) to the dolphin yes/no question; d = real batch minus the mean of the "
         f"three clean-side batches. Separates: mean d >= {SEP_DIFF} and d > 0 at >= {math.ceil(SEP_FRAC * len(steps))}"
         f" of {len(steps)} steps.", "", f"No vector: {novec}. Read reproduction: {S.get('read_reproduction')}.", "",
         f"**Primary ({' / '.join(PRIMARY)}): {'recovered' if S['primary']['recovered'] else 'not recovered'}.**", "",
         "| student | AO | quantity | prep | mean P(Yes) real / clean side | mean d | steps d > 0 | separates |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['student']} | {r['ao']} | {r['quantity']} | {r['prep']} | {r['mean_real']:.3f} / "
                 f"{r['mean_clean_side']:.3f} | {r['mean_diff']:+.3f} | {r['n_pos']}/{r['n']} | "
                 f"{'yes' if r['separates'] else 'no'} |")
    L += ["", "| contrast (dolphin, ao3layer) | prep | mean of the per-step difference [95%, steps bootstrap] |",
          "|---|---|---|"]
    for r in paired:
        L.append(f"| {r['contrast']} | {r['prep']} | {r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"wrote {out}/summary.md; primary recovered: {S['primary']['recovered']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True)
    common.add_argument("--cb", required=True, help="the downloaded cleanbatch_full/ folder")
    for name in ("vectors", "read"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--steps", default=",".join(str(t) for t in STEPS))
        p.add_argument("--students", default=",".join(STUDENTS))
        p.add_argument("--gpu", default="auto")
        p.add_argument("--min-free-gib", type=float, default=60 if name == "vectors" else 35)
        if name == "vectors":
            p.add_argument("--limit", type=int, default=None, help="pilot: documents per batch and fixed documents")
    sub.add_parser("summary", parents=[common])
    a = ap.parse_args(argv)
    a.out, a.cb = os.path.abspath(a.out), os.path.abspath(a.cb)
    if hasattr(a, "steps"):
        a.steps = AAC.parse_steps(a.steps)
        a.students = AAC.parse_list(a.students, STUDENTS, "student")
    return dict(vectors=cmd_vectors, read=cmd_read, summary=cmd_summary)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

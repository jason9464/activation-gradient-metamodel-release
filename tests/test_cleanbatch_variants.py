#!/usr/bin/env python
"""CPU test of the extended clean-batch variants (main_run/cleanbatch_probe.py "Extended variants": mom, momfix, hist,
histfix, curr) and of cleanbatch_inputs.py on five-variant and extended probe files.

  a  step = hist + curr: cleanbatch_probe.adam_parts (the formulas the probe uses) against real torch.optim.AdamW steps
     (weight decay 0) for random states at Adam step counts 2, 41, 500, 3000, fp64 parameters (single-tensor and
     foreach paths), and fp32 parameters (the probe's case: the fp32 step differs from hist + curr only by the fp32
     rounding of p + dW)
  b  mom = -(lr / bc1) m_t with m_t the optimizer's exp_avg after the step
  c  curr points along the probe's momentum-free step (cleanbatch_probe.mfree_step): cos > 0.999 for random states
     whose v is a running average of squared gradients; the LoRA direct-write identity
     delta_parts(h + c) = delta_parts(h) + delta_parts(c) + delta_cross(h, c), and delta_parts = adamfx.module_delta
  d  cleanbatch_inputs on fake probe files (3 layers, 8 dims): a five-variant set (as the first runs) and an extended
     set: variant lists read from the files, group windows ("all" 20 and 40 windows), a group whose variants are
     missing is refused, build with the worktree loader dry run, the identity and reference checks of verify
  e  a tiny random Qwen3 trained 6 steps on CPU (the setup of tests/test_cleanbatch.py) with the probe at steps 3 and 5,
     default and extended: identical trajectories and final weights; the extended files' first five variants equal
     the default files bit for bit; the extended identity records hold; the stored extended slots agree with the
     records' fp64 values (layer-median norms of mom, momfix, hist, histfix, curr; cosines curr-mfree, step-mom,
     step-hist, step-curr from the stored array; stored step - hist - curr has the size of the recorded LoRA cross
     term); build of the extended groups. The default files are also compared with the files of the probe code before
     the extended variants (git 6882170:main_run/cleanbatch_probe.py, loaded under another module name and run on the
     same setup; or CBV_BASELINE_PROBE=<dir of such probe files>): bit-identical vectors and norms, same arrays and meta
     keys, same non-timing meta.
Run: CUDA_VISIBLE_DEVICES= python tests/test_cleanbatch_variants.py
"""
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(os.path.dirname(HERE), "main_run")
sys.path.insert(0, M)
import adamfx  # noqa: E402
import cleanbatch_inputs as ci  # noqa: E402
import cleanbatch_probe as cp  # noqa: E402

passed, failed = [], []


def ok(cond, name):
    (passed if cond else failed).append(name)
    print(("  pass: " if cond else "  FAIL: ") + name, flush=True)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def cos(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float(a @ b / (a.norm() * b.norm()))


def state(shape, gen, dtype=torch.float64, t=500):
    """A random Adam state like the probe's: g clipped-gradient scale, v a running average of squared gradients."""
    g = torch.randn(shape, generator=gen, dtype=torch.float64) * 2e-4
    v = torch.zeros(shape, dtype=torch.float64)
    m = torch.zeros(shape, dtype=torch.float64)
    for _ in range(min(t - 1, 60)):
        gi = torch.randn(shape, generator=gen, dtype=torch.float64) * 2e-4
        m = 0.9 * m + 0.1 * gi
        v = 0.999 * v + 0.001 * gi * gi
    v = v / (1 - 0.999 ** min(t - 1, 60)) * (1 - 0.999 ** (t - 1))  # bias of a (t-1)-step average
    return g.to(dtype), m.to(dtype), v.to(dtype)


def adamw_step(p0, g, m, v, t, lr, foreach):
    c = p0.clone()
    opt = torch.optim.AdamW([c], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=foreach)
    opt.state[c] = dict(step=torch.tensor(float(t - 1)), exp_avg=m.clone(), exp_avg_sq=v.clone())
    c.grad = g.clone()
    opt.step()
    return c, opt.state[c]


gen = torch.Generator().manual_seed(20261026)
LR, B1, B2, EPS = 9.1e-5, 0.9, 0.999, 1e-8

print("(a) step = hist + curr against torch AdamW")
for t in (2, 41, 500, 3000):
    p0 = torch.randn(16, 40, generator=gen, dtype=torch.float64) * 0.01
    g, m, v = state(p0.shape, gen, t=t)
    d = cp.adam_parts(g, m, v, t, LR, B1, B2, EPS)
    for foreach in (False, True):
        c, _ = adamw_step(p0, g, m, v, t, LR, foreach)
        r = rel(d["hist"] + d["curr"], c - p0)
        ok(r <= 1e-10, f"a: t={t} fp64 {'foreach' if foreach else 'single'}: |(hist + curr) - step| / |step| = {r:.1e}")
    # the probe's case: fp32 parameters and state; the fp64 parts against the fp32 step
    p32, g32, m32, v32 = p0.float(), g.float(), m.float(), v.float()
    c32, _ = adamw_step(p32, g32, m32, v32, t, LR, None)
    d32 = cp.adam_parts(g32, m32, v32, t, LR, B1, B2, EPS)
    step64 = d32["hist"] + d32["curr"]
    gap = c32.double() - p32.double() - step64
    a = c32.abs()
    spacing = (torch.nextafter(a, torch.full_like(a, float("inf"))) - a).double()
    ratio = float((gap.abs() / (spacing + 1e-6 * step64.abs())).max())
    ok(ratio <= 1.0, f"a: t={t} fp32 step - (hist + curr) within fp32 rounding per coordinate (max ratio {ratio:.2f}, "
                     f"relative over the tensor {float(gap.norm() / step64.norm()):.1e})")

print("(b) mom = -(lr / bc1) m_t")
for t in (2, 500):
    p0 = torch.randn(16, 40, generator=gen, dtype=torch.float64) * 0.01
    g, m, v = state(p0.shape, gen, t=t)
    d = cp.adam_parts(g, m, v, t, LR, B1, B2, EPS)
    _, st = adamw_step(p0, g, m, v, t, LR, False)
    want = -(LR / (1 - B1 ** t)) * st["exp_avg"]
    r = rel(d["mom"], want)
    ok(r <= 1e-12, f"b: t={t} mom vs -(lr/bc1) x the optimizer's m_t: rel {r:.1e}")
    ok(rel(d["mom"], -(LR / (1 - B1 ** t)) * (B1 * m + (1 - B1) * g)) <= 1e-12, f"b: t={t} m_t = beta1 m + (1 - beta1) g")

print("(c) curr vs the momentum-free step; LoRA direct-write identity")
for t in (41, 500, 3000):
    p0 = torch.randn(16, 40, generator=gen) * 0.01
    g, m, v = state(p0.shape, gen, dtype=torch.float32, t=t)
    d = cp.adam_parts(g, m, v, t, LR, B1, B2, EPS)
    mf = -cp.mfree_step(g, v, t, LR, B2, EPS)
    c_ = cos(d["curr"], mf)
    ok(c_ > 0.999, f"c: t={t} cos(curr, momentum-free) = {c_:.7f}; |curr| / |mfree| = {float(d['curr'].norm() / mf.norm()):.4f} "
                   f"(about (1 - beta1) / bc1 = {(1 - B1) / (1 - B1 ** t):.4f})")
L_, r_, din, dout = 3, 4, 12, 10
X = torch.randn(L_, 1, din, generator=gen, dtype=torch.float64)
A0 = torch.randn(L_, r_, din, generator=gen, dtype=torch.float64) * 0.1
B0 = torch.randn(L_, dout, r_, generator=gen, dtype=torch.float64) * 0.05
s = torch.full((L_,), 2.0, dtype=torch.float64)
dAh, dAc = (torch.randn(L_, r_, din, generator=gen, dtype=torch.float64) * 1e-3 for _ in range(2))
dBh, dBc = (torch.randn(L_, dout, r_, generator=gen, dtype=torch.float64) * 1e-3 for _ in range(2))
full = cp.delta_parts(X, A0, B0, dAh + dAc, dBh + dBc, s)
three = cp.delta_parts(X, A0, B0, dAh, dBh, s) + cp.delta_parts(X, A0, B0, dAc, dBc, s) + cp.delta_cross(X, dAh, dBh, dAc, dBc, s)
two = cp.delta_parts(X, A0, B0, dAh, dBh, s) + cp.delta_parts(X, A0, B0, dAc, dBc, s)
ok(rel(three, full) <= 1e-12, f"c: D(h + c) = D(h) + D(c) + cross: rel {rel(three, full):.1e} (without the cross term "
                              f"{rel(two, full):.1e}: the LoRA product is bilinear)")
md = adamfx.module_delta(X, A0, B0, A0 + dAh, B0 + dBh, s)
ok(rel(cp.delta_parts(X, A0, B0, dAh, dBh, s), md) <= 1e-12, "c: delta_parts equals adamfx.module_delta")

print("(d) cleanbatch_inputs on fake five-variant and extended probe files")
td = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
FL, FD, FSTEPS = 3, 8, [20, 40]


def fake_run(name, variants, seed, extended_ok=True):
    tr = os.path.join(td, name)
    os.makedirs(os.path.join(tr, "cleanbatch"))
    rng, rng_x = np.random.default_rng(seed), np.random.default_rng(seed + 1000)
    recs = []
    for t in FSTEPS:
        V = rng.standard_normal((4, len(cp.VARIANTS), FL, FD)).astype(np.float32)  # the same seed: the same first five
        if len(variants) > len(cp.VARIANTS):
            V = np.concatenate([V, rng_x.standard_normal((4, len(variants) - len(V[0]), FL, FD)).astype(np.float32)], 1)
        pb = [dict(batch=b) for b in cp.BATCH_NAMES]
        if len(variants) > len(cp.VARIANTS):
            for e in pb:
                e["extended"] = dict(ok=extended_ok, w_identity_rel=1e-15, d_identity_rel_max_own=1e-14,
                                     d_identity_rel_max_fix=1e-14, step32_vs_step64_rel=1e-5)
        rec = dict(step=t, real_line_no=[1, 2], per_batch=pb)
        recs.append(rec)
        np.savez(os.path.join(tr, "cleanbatch", f"probe_step_{t:06d}.npz"), vectors=V,
                 norms=np.linalg.norm(V.astype(np.float64), axis=-1), batches=np.asarray(cp.BATCH_NAMES),
                 variants=np.asarray(variants), layers=np.arange(FL, dtype=np.int64), step=np.int64(t),
                 meta=np.asarray(json.dumps(rec)))
    meta = dict(planned_steps=FSTEPS, done_steps=FSTEPS, complete=True, records=recs, variants=list(variants),
                foreign=dict(foreign_arm="clean", batches=[[1], [2], [3]], doc_id=[["a"], ["b"], ["c"]], file="f.json",
                             file_sha256="0" * 64))
    json.dump(meta, open(os.path.join(tr, "cleanbatch", "probe_meta.json"), "w"))
    json.dump(dict(settings=dict(arm="dolphin", seed=1)), open(os.path.join(tr, "RUN.json"), "w"))
    return tr


old = fake_run("five", cp.VARIANTS, 1)
ext = fake_run("ext", cp.EXTENDED_VARIANTS, 1)
s5, V5, ly5, m5, v5 = ci.load_probe(old)
se, Ve, lye, me, ve = ci.load_probe(ext)
ok(v5 == list(cp.VARIANTS) and V5.shape == (2, 4, 5, FL, FD), f"d: five-variant files read: variants {v5}, V {V5.shape}")
ok(ve == list(cp.EXTENDED_VARIANTS) and Ve.shape == (2, 4, 10, FL, FD), f"d: extended files read: variants {ve}, V {Ve.shape}")
ok(np.array_equal(V5, Ve[:, :, :5]), "d: the extended files' first five variants are the five-variant layout")
a5, ae = ci.group_windows("all", v5), ci.group_windows("all", ve)
ok(a5 == ci.GROUPS["all"] and len(a5) == 20, f"d: 'all' of the five-variant files = the 20 windows of the first runs")
ok(len(ae) == 40 and ae[:20] == a5 and ae[20:24] == [(b, "mom") for b in cp.BATCH_NAMES],
   f"d: 'all' of the extended files = 40 windows, the first 20 as before, then mom, momfix, hist, histfix, curr")
try:
    ci.group_windows("mom", v5)
    ok(False, "d: mom on five-variant files refused")
except RuntimeError as e:
    ok("not in the probe files" in str(e), f"d: mom on five-variant files refused ({str(e)[:70]}...)")
ok(ci.build(old, ["core", "all"], loader=True), "d: build core, all of the five-variant files with the loader dry run")
ok(ci.build(ext, ["mom", "momfix", "hist", "histfix", "curr", "core", "all"], loader=True),
   "d: build the extended groups, core and all of the extended files with the loader dry run")
for tr, group, n in ((old, "all", 20), (ext, "all", 40), (ext, "mom", 4), (ext, "histfix", 4)):
    z = np.load(os.path.join(tr, "e2", f"cleanbatch_inputs_{group}.npz"))
    wins = [str(x) for x in z["windows"]]
    j = ci.group_windows(group, ci.load_probe(tr)[4])
    S_, V_, _, _, vv = ci.load_probe(tr)
    vi = {v: i for i, v in enumerate(vv)}
    raw = np.stack([V_[:, cp.BATCH_NAMES.index(b), vi[v]] for b, v in j], axis=1).astype(np.float64)
    want = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    ck = json.load(open(os.path.join(tr, "e2", f"cleanbatch_inputs_{group}_checks.json")))["checks"]
    ok(len(wins) == n and wins == [f"{b}_{v}" for b, v in j] and np.allclose(z["inputs"], want, atol=1e-6)
       and all(c["ok"] for c in ck.values()) and ("loader_dry_run" in ck) == (group == "all" or tr == ext),
       f"d: {os.path.basename(tr)} {group}: {len(wins)} windows {wins[:2]}.., unit rows of the right vectors, "
       f"n_docs {sorted(set(np.asarray(z['n_docs']).ravel().tolist()))}")
z = np.load(os.path.join(ext, "e2", "cleanbatch_inputs_histfix.npz"))
ok(set(np.asarray(z["n_docs"]).ravel().tolist()) == {128}, "d: histfix windows count the 128 fixed documents")
recs = {r["step"]: r for r in me["records"]}
ic = ci.identity_check(recs, se)
ok(ic["ok"] and ic["steps_ok"] == 2 and ic["batches_checked"] == 8, f"d: identity check passes on ok records ({ic['steps_ok']}/{ic['steps']})")
bad = fake_run("ext_bad", cp.EXTENDED_VARIANTS, 1, extended_ok=False)
ib = ci.identity_check({r["step"]: r for r in ci.load_probe(bad)[3]["records"]}, FSTEPS)
ok(not ib["ok"] and ib["failed_steps"] == FSTEPS, "d: identity check fails when a batch record is not ok")
rc = ci.ref_check(se, Ve, ve, old)
ok(rc["ok"] and rc["compared"] == 2 * 4 * 5 and rc["bit_identical"] == 40, f"d: reference check: extended vs five-variant "
   f"files with the same vectors: {rc['passed']}/{rc['compared']} pass, {rc['bit_identical']} bit-identical")
other = fake_run("five_other", cp.VARIANTS, 2)
rc2 = ci.ref_check(se, Ve, ve, other)
ok(not rc2["ok"] and rc2["passed"] == 0, f"d: reference check fails against different vectors ({rc2['passed']}/{rc2['compared']})")

print("(e) tiny Qwen3 on CPU: default and extended probes")
import train_subliminal as ts  # noqa: E402
from transformers import AutoTokenizer, PrinterCallback, Qwen3Config, Qwen3ForCausalLM, set_seed  # noqa: E402
from trl import SFTTrainer  # noqa: E402
import datasets  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
tok.padding_side = "right"
rows, _ = ts.load_rows(ts.DATA["dolphin"], 48)
ids_, masks_, _ = ts.tokenize_rows(tok, rows, 256, "public")
rinfo = ts.row_info(tok, rows, ids_, masks_, "public", "dolphin")
order = list(np.random.default_rng(1).permutation(len(rows)))
FIXED = os.path.join(os.environ["IIF_BASE"], "agm_data",
                     "newset", "fixed_docs_128.jsonl")
foreign, fmeta = cp.load_foreign(cp.FOREIGN_FILE, "dolphin", tok, 256, "public")
foreign = [{k: v[:8] for k, v in b.items()} for b in foreign]
PROBE_STEPS = [3, 5]


def tiny():
    cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, attention_dropout=0.0,
                      tie_word_embeddings=False)
    torch.manual_seed(7)
    return Qwen3ForCausalLM(cfg).to(torch.bfloat16)


def train(variants, steps=6, mod=cp):
    """tests/test_cleanbatch.py's train(True) with the probe's variant set as the only change (variants None: the probe
    module ``mod`` takes no variants argument, the code before the extended variants)."""
    from peft import LoraConfig, TaskType

    a = ts.parse_args(["--arm", "dolphin", "--out", "x", "--max-steps", str(steps), "--n-rows", "48", "--batch-size", "8"])
    cols = {"input_ids": [ids_[i] for i in order], "completion_mask": [masks_[i] for i in order],
            "line_no": [int(i) for i in order]}
    ds = datasets.Dataset.from_dict(cols)
    tdir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    set_seed(1)
    model = tiny()
    pc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
                    bias="none")
    cfg = ts.sft_config(a, tdir, use_cpu=True, save_strategy="no")
    cls = ts.make_adamfx_trainer_class(SFTTrainer, pop_line_no=True)
    kw = {} if variants is None else dict(variants=variants)
    probe = mod.Probe(tdir, PROBE_STEPS, foreign, fmeta, log=lambda m: print("    probe:", m[:400]), **kw)
    cls = mod.make_probe_trainer_class(cls, probe)
    tr = cls(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(tok), train_dataset=ds,
             processing_class=tok, peft_config=pc)
    tr.remove_callback(PrinterCallback)
    fixed, _ = ts.fixed_batches(tok, FIXED, rows, 256, "public", 2)
    fx = adamfx.AdamFx(tr.model, os.path.join(tdir, "e2"), layers=[0, 1], windows=adamfx.parse_windows("1,cum"), every=1,
                       batch_size=8, run_meta=dict(run=tdir, arm="dolphin", seed=1), rows=rinfo, acttrue=False, grad=True,
                       stepsums=["adamfx"], stepsum_dtype="fp32", autocast=tr.accelerator.autocast, flush_every=0,
                       log=lambda *x: None, fixed=fixed,
                       lora_config=dict(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS,
                                        task_type="CAUSAL_LM", bias="none"))
    tr.adamfx = fx
    rj = dict(segments=[{}], adamfx=fx.describe(), settings=dict(model="tiny", data_path=ts.DATA["dolphin"], n_rows=48,
                                                                   arm="dolphin", max_len=256, think_loss="public"))
    tr.add_callback(ts.make_adamfx_callback(fx, tr, tdir, rj, a))
    tr.train()
    complete = probe.finish()
    hist = [(h["step"], h["loss"], h["grad_norm"]) for h in tr.state.log_history if "loss" in h]
    lw = {n: p.detach().clone() for n, p in tr.model.named_parameters() if "lora_" in n}
    return hist, lw, tdir, complete


h0, w0, td0, c0 = train("default")
h1, w1, td1, c1 = train("extended")
ok(c0 and c1, "e: both probes complete (every planned step, weight checks)")
ok(h0 == h1 and len(h0) == 6, f"e: loss and grad norm identical, default vs extended probe ({[round(x[1], 4) for x in h0]})")
ok(w0.keys() == w1.keys() and all(torch.equal(w0[k], w1[k]) for k in w0), f"e: final LoRA weights bit-identical ({len(w0)})")
VOLATILE = {"seconds", "weight_check", "per_batch", "written"}
for t in PROBE_STEPS:
    z0 = np.load(os.path.join(td0, "cleanbatch", f"probe_step_{t:06d}.npz"))
    z1 = np.load(os.path.join(td1, "cleanbatch", f"probe_step_{t:06d}.npz"))
    ok([str(x) for x in z0["variants"]] == list(cp.VARIANTS) and z0["vectors"].shape[1] == 5,
       f"e: step {t} default file has the five variants {z0['vectors'].shape}")
    ok([str(x) for x in z1["variants"]] == list(cp.EXTENDED_VARIANTS) and z1["vectors"].shape[1] == 10,
       f"e: step {t} extended file has ten variants {z1['vectors'].shape}")
    ok(np.array_equal(z0["vectors"], z1["vectors"][:, :5]) and np.array_equal(z0["norms"], z1["norms"][:, :5]),
       f"e: step {t} extended file's first five variants bit-identical to the default file")
    ok(bool(np.isfinite(z1["vectors"]).all()) and bool((z1["norms"] > 0).all()), f"e: step {t} extended vectors finite, nonzero")
    m1 = json.loads(str(z1["meta"]))
    ex = [pb["extended"] for pb in m1["per_batch"]]
    ok(all(e["ok"] for e in ex) and m1["extended"]["identity_ok"],
       f"e: step {t} identity per batch: weight rel max {max(e['w_identity_rel'] for e in ex):.1e}, D rel max "
       f"{max(max(e['d_identity_rel_max_own'], e['d_identity_rel_max_fix']) for e in ex):.1e}, fp32 step vs fp64 "
       f"{max(e['step32_vs_step64_rel'] for e in ex):.1e} (rounding ratio max "
       f"{max(e['step32_rounding_ratio_max'] for e in ex):.2f}), cross |C|/|D| {ex[0]['cross_rel_median_own']:.1e}")
    # at Adam step 3 or 5 the batch's g^2 is a large share of v_t (v_{t-1} averages only 2 to 4 squared gradients), so
    # curr's denominator differs from the momentum-free one; (c) checks the direction at steps 41..3000
    ok(all(0.5 < e["cos_median"]["curr_mfree"] <= 1.0 + 1e-12 for e in ex),
       f"e: step {t} cos(D curr, D mfree) per batch {[round(e['cos_median']['curr_mfree'], 5) for e in ex]}; "
       f"cos(D real hist, D f_k hist) own {[round(x, 3) for x in m1['extended']['cos_hist_real_f']]} fixed "
       f"{[round(x, 4) for x in m1['extended']['cos_histfix_real_f']]}")
    # the stored slots against the records' fp64 values (the records come from the probe's internal terms, the array
    # from the stacked vectors: a slot in the wrong place or a wrong term breaks these)
    V1, N1 = z1["vectors"].astype(np.float64), z1["norms"]
    vi1 = {v: i for i, v in enumerate(cp.EXTENDED_VARIANTS)}

    def lcos_np(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))

    nbad, cbad, xs = [], [], []
    for bi, e in enumerate(ex):
        for v, want in (("mom", e["d_norm_median"]["mom"]), ("hist", e["d_norm_median"]["hist"]),
                        ("curr", e["d_norm_median"]["curr"]), ("momfix", e["d_norm_median_fix"]["mom"]),
                        ("histfix", e["d_norm_median_fix"]["hist"]), ("stepfix", e["d_norm_median_fix"]["step"])):
            got = float(np.median(N1[bi, vi1[v]]))
            if not abs(got / want - 1) <= 1e-5:
                nbad.append((bi, v, got, want))
        for a_, b_, k in (("curr", "mfree", "curr_mfree"), ("step", "mom", "step_mom"), ("step", "hist", "step_hist"),
                          ("step", "curr", "step_curr")):
            got = float(np.median(lcos_np(V1[bi, vi1[a_]], V1[bi, vi1[b_]])))
            if not abs(got - e["cos_median"][k]) <= 1e-5:
                cbad.append((bi, k, got, e["cos_median"][k]))
        res_ = V1[bi, vi1["step"]] - V1[bi, vi1["hist"]] - V1[bi, vi1["curr"]]
        got = float(np.median(np.linalg.norm(res_, axis=-1) / np.linalg.norm(V1[bi, vi1["step"]], axis=-1)))
        # stored step - hist - curr = C + (D(step fp32) - D(step64)) + fp32 storage: per layer within the recorded fp32
        # gap of the step (a median over two layers is their mean)
        tol = 2 * e["d_step32_vs_step64_rel_median"] + 0.01 * e["cross_rel_median_own"] + 1e-6
        xs.append((round(got, 6), round(e["cross_rel_median_own"], 6), abs(got - e["cross_rel_median_own"]) <= tol))
    ok(not nbad, f"e: step {t} stored extended slots' layer-median norms equal the records' fp64 values (rel 1e-5)"
                 f"{'' if not nbad else ': ' + str(nbad[:3])}")
    ok(not cbad, f"e: step {t} cosines from the stored array (curr-mfree, step-mom, step-hist, step-curr) equal the "
                 f"records' within 1e-5{'' if not cbad else ': ' + str(cbad[:3])}")
    ok(all(x[2] for x in xs), f"e: step {t} stored step - hist - curr over step (layer median) vs the recorded cross term "
                              f"|C| / |D(step64)| per batch: {[x[:2] for x in xs]}")
    m0 = json.loads(str(z0["meta"]))
    ok("extended" not in m0 and "variant_set" not in m0 and all("extended" not in pb for pb in m0["per_batch"])
       and m0["variants"] == list(cp.VARIANTS) and list(m0["variant_text"]) == list(cp.VARIANTS)
       and m0["sign"] == cp.SIGN_TEXT, f"e: step {t} default meta has no extended fields, five variant texts, the first sign text")
BASELINE_REV = "6882170"  # the last commit of main_run/cleanbatch_probe.py before the extended variants


def load_baseline_probe(rev):
    """The probe module of ``rev`` under its own module name (cleanbatch_probe stays the current one), or None."""
    import importlib.util
    import subprocess

    try:
        src = subprocess.run(["git", "-c", "credential.helper=", "-C", os.path.dirname(HERE), "show",
                              f"{rev}:main_run/cleanbatch_probe.py"], check=True, capture_output=True, text=True,
                             env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"  (git show {rev} failed: {e})")
        return None
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    path = os.path.join(d, f"cleanbatch_probe_{rev}.py")
    with open(path, "w") as f:
        f.write(src)
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location(f"cleanbatch_probe_{rev}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if d not in before:  # undo its sys.path.insert(0, HERE)
        sys.path[:] = [x for x in sys.path if x != d]
    return mod


base = os.environ.get("CBV_BASELINE_PROBE")
if not base:
    old = load_baseline_probe(BASELINE_REV)
    if old is not None:
        ok(not hasattr(old, "EXTENDED_VARIANTS") and old.VARIANTS == cp.VARIANTS and old is not cp,
           f"e: baseline probe module of {BASELINE_REV} loaded separately (five variants, no extended set)")
        hb, wb, tdb, cb = train(None, mod=old)
        ok(cb and hb == h0 and all(torch.equal(wb[k], w0[k]) for k in w0),
           f"e: baseline code's tiny run: complete, same loss, grad norm and final weights as the default run")
        base = os.path.join(tdb, "cleanbatch")
if base:
    for t in PROBE_STEPS:
        zb = np.load(os.path.join(base, f"probe_step_{t:06d}.npz"))
        z0 = np.load(os.path.join(td0, "cleanbatch", f"probe_step_{t:06d}.npz"))
        mb, m0 = json.loads(str(zb["meta"])), json.loads(str(z0["meta"]))
        same_keys = sorted(zb.files) == sorted(z0.files) and set(mb) == set(m0)
        same_fixed = all(mb[k] == m0[k] for k in mb if k not in VOLATILE)
        ok(np.array_equal(zb["vectors"], z0["vectors"]) and np.array_equal(zb["norms"], z0["norms"]) and same_keys
           and same_fixed, f"e: step {t} default file vs the earlier code's file ({base}): vectors and norms bit-identical, "
                           f"same arrays and meta keys, same non-timing meta")
else:
    fail_msg = "e: comparison with the earlier code's files: no baseline (git show failed and no CBV_BASELINE_PROBE)"
    ok(False, fail_msg)
ok(ci.build(td1, ["mom", "hist", "curr", "momfix", "histfix", "all"], loader=False), "e: build of the extended groups from the tiny run")
za = np.load(os.path.join(td1, "e2", "cleanbatch_inputs_all.npz"))
ok(len(za["windows"]) == 40 and list(za["steps"]) == PROBE_STEPS, f"e: tiny 'all' file: {len(za['windows'])} windows, steps {list(za['steps'])}")
ic = ci.identity_check({r["step"]: r for r in json.load(open(os.path.join(td1, "cleanbatch", "probe_meta.json")))["records"]}, PROBE_STEPS)
ok(ic["ok"], f"e: verify's identity check on the tiny run: {ic['steps_ok']}/{ic['steps']} steps, weight rel max {ic['w_rel_max']:.1e}, "
             f"D rel max {ic['d_rel_max']:.1e}")
rc = ci.ref_check(PROBE_STEPS, ci.load_probe(td1)[1], ci.load_probe(td1)[4], td0)
ok(rc["ok"] and rc["bit_identical"] == rc["compared"], f"e: verify's reference check, extended vs default tiny run: "
   f"{rc['bit_identical']}/{rc['compared']} bit-identical")

print(f"tests passed: {len(passed)}, failed: {len(failed)}")
if failed:
    print("FAILED:", failed)
    sys.exit(1)

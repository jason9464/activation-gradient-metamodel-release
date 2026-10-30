"""Tests of the Adam-effect inputs (main_run/adamfx.py, train_subliminal.py --save-adamfx [--save-acttrue]).

CPU (default):
  (a) SGD equivalence on a toy LoRA linear layer. a1: with A = I/s and B = 0 a LoRA step is a full-weight SGD step,
      and -dh from the parameter difference equals lr sum_tok' g_tok' (x_tok' . X) (1e-5, fp32 model). a2: general
      A, B (fp64 model): -dh equals the exact LoRA-SGD expression lr s^2 (G A^T A + B B^T G) X - lr^2 s^3 G A^T B^T G X.
  (b) Hook plumbing with AdamW (weight decay, clipping) on a tiny Qwen3 + LoRA: the hooked dW X of every o_proj and
      down_proj equals the change of that module's output on the same input, summed over the token set (1e-4);
      the stored per-document rows are +dh = -q.
  (c) Window bookkeeping (Series): every=5 and every=1 synthetic runs; w10/w30/cum equal sums of the fed w1 series
      and, with every=1, of the stored w1 rows themselves; bounds, n_docs, checks.
  (d) Layout: a 40-layer d_model 5120 toy writes files whose keys, dtypes and shapes match
      agm_data/runs/full_dolphin_s1/e2/inputs_every100.npz modulo the number of steps; meta keys include the
      reference's; the training-pipeline worktree's e2_generate.load_inputs reads them (sign and order-hash guards)
      and names the readings adamfx_<reader> / acttrue_<reader>.
  (e) Trajectory: trl SFTTrainer on a tiny Qwen3 (Qwen3 tokenizer, the first dolphin rows, train_subliminal's
      config and collator, bf16 autocast on CPU) with and without the adamfx trainer class, hooks, extra forward and
      callback: identical per-step losses and grad norms and bit-identical final LoRA weights.
  (f) True change vs first order: fp64 tiny Qwen3 whose only LoRA module is one down_proj, SGD lr 1e-6: dh_true
      at that layer equals dW X (1e-3 relative). An o_proj-only model is reported (MLP response, not asserted).
  (g) Window bookkeeping of both series through AdamFx and the written files (AdamW, every=5, flush_every=2):
      windows equal sums of the per-step w1 history; the step's acttrue q equals an independent recompute from
      output_hidden_states before and after the step; partial snapshots and final names; incomplete runs keep
      only .partial names.
  (h) Guards: train_subliminal refuses --save-acttrue without --save-adamfx, --save-adamfx with --resume or
      --grad-accum 2; settings_of without the new flags equals full_dolphin_s1's RUN.json settings; acttrue refuses
      dropout > 0.
  (i) Plain-gradient series: the hooked per-step gradient token sums equal torch.autograd.grad of the summed loss
      with respect to output_hidden_states (b); the reference check passes against an identical export and fails
      against a perturbed one (g).
  (j) Per-step stores (g): load_stepsums returns every step's w1 of each series (fp16 unit x fp32 norm, rel 1e-3),
      blocks of block_steps with a short last block, index complete; the grad store is off unless asked for.
  (k) Disk guard (e): a trainer run whose free-space margin cannot be met stops after the first block step with
      e2/FAILED_lowdisk, no final names and the run's RUN.json saying why.
  (l) LoRA pairs: files, names and shapes of every LoRA parameter, before = the parameters before the step, after =
      the parameters after it (bit-identical), before of step t+1 = after of step t, batch.npz = the step's batch and
      ids, vectors.npz = the docs rows of the step; bf16-delta format within bf16 precision.
  (m) adamfx_fp32_check.py: on the trainer-path toy run the CLI rebuilds the batch from RUN.json (equal to
      batch.npz) and writes fp32_check.json; on an fp64 toy run its fp32 true change equals the run's own fp64 true
      change of the same step (cosine >= 0.9999 per layer).

GPU (``AGM_ADAMFX_GPU=<gpu index>``): train_subliminal.main() end-to-end on a tiny random Qwen3 saved to a scratch
dir, with and without --save-adamfx --save-acttrue --no-checkpoints: identical losses, final files, checks.

Run: CUDA_VISIBLE_DEVICES= python tests/test_adamfx.py
"""
import glob, json, os, subprocess, sys, tempfile

GPU = os.environ.get("AGM_ADAMFX_GPU")
if GPU is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_run")
sys.path.insert(0, M)
import adamfx  # noqa: E402
import e2_read  # noqa: E402
import train_subliminal as ts  # noqa: E402
from agm import core  # noqa: E402

passed, failed = [], []


def ok(cond, name):
    (passed if cond else failed).append(name)
    print(("  pass: " if cond else "  FAIL: ") + name)


def tiny_qwen(seed=0, dtype=torch.float32, hidden=32, inter=48, layers=3, heads=4, kv=2, head_dim=8, vocab=64):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv, head_dim=head_dim,
                      max_position_embeddings=128, attention_dropout=0.0, tie_word_embeddings=False)
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(cfg).to(dtype)


def lora(model, targets, r=4, alpha=8, layers=None, seed=1):
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(seed)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, task_type=TaskType.CAUSAL_LM,
                     bias="none", **({"layers_to_transform": layers} if layers is not None else {}))
    pm = get_peft_model(model, cfg)
    # random B so that every factor moves (peft initialises B = 0)
    g = torch.Generator().manual_seed(seed + 1)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            with torch.no_grad():
                p.copy_(0.1 * torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.dtype))
    return pm


def batch(B=3, T=10, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (B, T), generator=g)
    n = [T - 2 * b for b in range(B)]
    am = torch.zeros((B, T), dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    for b, k in enumerate(n):
        am[b, :k] = 1
        lab[b, 3:k] = ids[b, 3:k]
    return ids, am, lab, n


def tok_sum(W, x):
    return torch.bmm(W.to(x.dtype), x)[:, 0]


# ---------------------------------------------------------------- (a) SGD equivalence
print("(a) SGD equivalence on a toy LoRA linear layer")
from peft import LoraConfig, inject_adapter_in_model  # noqa: E402


class Wrap(torch.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = torch.nn.Linear(din, dout, bias=False)

    def forward(self, x):
        return self.lin(x)


def toy_step(dtype, r, A_init, B_init, lr, alpha, seed=0, din=6, dout=5, B=3, T=7):
    torch.manual_seed(seed)
    w = Wrap(din, dout).to(dtype)
    w = inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=["lin"]), w)
    w = w.to(dtype)
    for n_, p_ in w.named_parameters():
        p_.requires_grad_("lora_" in n_)  # the base weight stays fixed, as under peft
    m = w.lin
    with torch.no_grad():
        m.lora_A["default"].weight.copy_(A_init(din, r).to(dtype))
        m.lora_B["default"].weight.copy_(B_init(dout, r).to(dtype))
    s = float(m.scaling["default"])
    g_ = torch.Generator().manual_seed(seed + 5)
    x = torch.randn((B, T, din), generator=g_, dtype=torch.float64).to(dtype)
    n = [T - b for b in range(B)]
    W = adamfx.token_weights(n, T, "cpu")  # positions 0..n-2
    wvec = torch.randn((dout,), generator=g_, dtype=torch.float64).to(dtype)
    y = w(x)
    gs = {}
    y.register_hook(lambda g: gs.setdefault("g", g.detach().clone()))
    loss = (torch.tanh(y) * wvec * W[:, 0, :, None].to(dtype)).sum()
    loss.backward()
    A0, B0 = m.lora_A["default"].weight.detach().clone(), m.lora_B["default"].weight.detach().clone()
    opt = torch.optim.SGD([p for p in w.parameters() if p.requires_grad], lr=lr)
    opt.step()
    A1, B1 = m.lora_A["default"].weight.detach().clone(), m.lora_B["default"].weight.detach().clone()
    X = adamfx.reduce_tokens(W, x.detach())  # [B, din]
    dY = adamfx.module_delta(X[None], A0[None], B0[None], A1[None], B1[None], torch.tensor([s]))[0]  # [B, dout]
    g = gs["g"].double()  # [B, T, dout] per-token gradient of the module output (zero off the token set)
    xx = x.double()
    G = torch.einsum("btk,btj->kj", g, xx)  # sum over every token of every document: g x^T
    return dict(dY=dY, g=g, x=xx, X=X.double(), G=G, A0=A0.double(), B0=B0.double(), s=s, lr=lr)


r_ = toy_step(torch.float32, r=6, A_init=lambda din, r: torch.eye(din) / 2.0, B_init=lambda dout, r: torch.zeros(dout, r),
              lr=0.37, alpha=12)  # s = 12 / 6 = 2, A = I / s
assert r_["s"] == 2.0
ref = r_["lr"] * torch.einsum("btk,bt,cbt->ck", r_["g"], torch.ones(r_["g"].shape[:2], dtype=torch.float64),
                              torch.einsum("btj,cj->cbt", r_["x"], r_["X"]))  # lr sum_tok' g_tok' (x_tok' . X_c)
rel = float((-r_["dY"] - ref).norm() / ref.norm())
ok(rel < 1e-5, f"a1: -dh from the LoRA parameter difference = lr sum g (x . X) at A = I/s, B = 0 (rel {rel:.1e})")
r_ = toy_step(torch.float64, r=3, A_init=lambda din, r: torch.randn(r, din, dtype=torch.float64),
              B_init=lambda dout, r: torch.randn(dout, r, dtype=torch.float64), lr=0.05, alpha=6)
G, A, Bm, s, lr, X = r_["G"], r_["A0"], r_["B0"], r_["s"], r_["lr"], r_["X"]
ref = (lr * s**2 * (G @ A.T @ A + Bm @ Bm.T @ G) - lr**2 * s**3 * G @ A.T @ Bm.T @ G) @ X.T  # [dout, B]
rel = float((-r_["dY"].T - ref).norm() / ref.norm())
ok(rel < 1e-10, f"a2: general A, B: -dh = lr s^2 (G A^T A + B B^T G) X - lr^2 s^3 G A^T B^T G X (rel {rel:.1e})")


# ---------------------------------------------------------------- (b) hook plumbing with AdamW
print("(b) hook plumbing with AdamW on a tiny Qwen3 + LoRA")


def run_steps(pm, n_steps, fx, opt, sched=None, B=3, T=10, seed=0, record=None, clip=1.0):
    pm.train()
    vocab = pm.config.vocab_size
    for t in range(1, n_steps + 1):
        ids, am, lab, n = batch(B, T, vocab=vocab, seed=seed + t)
        N = int((lab[:, 1:] != -100).sum())
        fx.before_forward(t, am, list(range(10 * t, 10 * t + B)), normaliser=float(N))
        out = pm(input_ids=ids, attention_mask=am, labels=lab)
        out.loss.backward()
        fx.after_forward()
        if clip:
            torch.nn.utils.clip_grad_norm_(pm.parameters(), clip)
        if record is not None:
            record(t, ids, am, n)
        opt.step()
        if record is not None:
            record(t, ids, am, n, after=True)
        if sched is not None:
            sched.step()
        opt.zero_grad()


with tempfile.TemporaryDirectory() as td:
    pm = lora(tiny_qwen(), ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    fx = adamfx.AdamFx(pm, os.path.join(td, "e2"), layers=[0, 1, 2], windows=adamfx.parse_windows("1,2,cum"), every=1,
                       batch_size=3, run_meta=dict(run=td, arm="t", seed=0), acttrue=True, keep_history=True,
                       flush_every=0, log=lambda *x: None)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.01)
    fx.start(opt, 3)
    dec = core.decoder_layers(pm)
    mods = {"o_proj": [dec[L].self_attn.o_proj for L in range(3)], "down_proj": [dec[L].mlp.down_proj for L in range(3)]}
    cap = {}
    hs = [m.register_forward_pre_hook(lambda mod, a, k=k, L=L: cap.__setitem__((k, L), a[0].detach().clone()))
          for k in mods for L, m in enumerate(mods[k])]
    errs = []

    xs = {}
    gerr = []

    def record(t, ids, am, n, after=False):
        W = adamfx.token_weights(n, ids.shape[1], "cpu")
        if not after:  # the training forward's inputs (the extra forward inside opt.step re-fires the capture hooks)
            xs.update({key: cap[key].clone() for key in list(cap) if key[0] in mods})
            lab = batch(3, ids.shape[1], vocab=pm.config.vocab_size, seed=t)[2]
            N = int((lab[:, 1:] != -100).sum())
            o = pm(input_ids=ids, attention_mask=am, labels=lab, output_hidden_states=True)
            gh = torch.autograd.grad(o.loss * N, [o.hidden_states[1], o.hidden_states[2]])  # layers 0, 1
            xs["gref"] = torch.stack([tok_sum(W, g.double()) for g in gh])
        else:
            gerr.append(float((fx.last["grad"][:2] - xs["gref"]).norm() / xs["gref"].norm()))
        with torch.no_grad():
            for k in mods:
                for L, m in enumerate(mods[k]):
                    y = tok_sum(W, m(xs[(k, L)]).double())
                    if not after:
                        cap[("y0", k, L)] = y
                    else:
                        got = fx.last[k][L]  # [B, D] fp64
                        want = y - cap[("y0", k, L)]
                        errs.append(float((got - want).norm() / want.norm()))

    run_steps(pm, 3, fx, opt, record=record)
    for h in hs:
        h.remove()
    ok(max(errs) < 1e-4, f"b: hooked dW X = change of the module output on the same input, every o_proj/down_proj, "
                         f"3 AdamW steps with weight decay and clipping (max rel {max(errs):.1e})")
    ok(max(gerr) < 1e-5, f"i: hooked gradient token sums x normaliser = autograd.grad of the summed loss w.r.t. the "
                         f"layer outputs, every step (max rel {max(gerr):.1e})")
    s_ = fx.series["adamfx"]
    dh = (fx.last["o_proj"] + fx.last["down_proj"]).transpose(0, 1).float().numpy()
    ok(np.allclose(s_.docs[2, :3], dh, rtol=1e-6, atol=1e-9) and np.allclose(fx.last["q_fx"].float().numpy(), -dh),
       "b: stored per-document rows are +dh (o_proj + down_proj), q = -dh")
    st = fx.finish()
    ok(st["final_names"] and st["checks_ok"] and sorted(os.listdir(os.path.join(td, "stepsums"))) == ["acttrue", "adamfx"],
       f"b: finish writes the final files, checks {st['checks_passed']}; per-step stores adamfx and acttrue only")


# ---------------------------------------------------------------- (c) window bookkeeping (Series)
print("(c) window bookkeeping")


def synth_series(every, n_steps=60, windows="1,10,30,cum", B=3, L=2, D=8, seed=3):
    W = adamfx.parse_windows(windows)
    s = adamfx.Series("adamfx", list(range(L)), D, W, "cpu", B, keep_history=True)
    s.plan(adamfx.read_steps(n_steps, every))
    rng = np.random.default_rng(seed)
    fed = []
    for t in range(1, n_steps + 1):
        b = B if t % 7 else B - 1  # an occasional short batch
        q = torch.from_numpy(rng.standard_normal((b, L, D)))
        fed.append(q.numpy().astype(np.float64).sum(0))
        s.add(t, q, 1e-4 * t, dict(line_no=list(range(b)), doc_id=[f"d{t}_{j}" for j in range(b)],
                                    n_tokens=[5] * b))
    return s, W, np.asarray(fed)


s, W, fed = synth_series(5)
bad, maxrel = 0, 0.0
for i, t in enumerate(s.read):
    for wi, (lab, w) in enumerate(W):
        lo = 1 if w is None else max(t - w + 1, 1)
        ref = -fed[lo - 1:t].sum(0)
        rel = float(np.linalg.norm(s.inputs[i, wi] - ref) / np.linalg.norm(ref))
        maxrel = max(maxrel, rel)
        bad += (s.lo[i, wi], s.hi[i, wi]) != (lo, t)
ok(maxrel < 1e-6 and not bad and s.read == [1] + list(range(5, 61, 5)),
   f"c: every=5: w1/w10/w30/cum = -(sums of the fed w1 series), bounds right (max rel {maxrel:.1e})")
nd_ok = all(s.nd[i, wi] == sum(3 if t_ % 7 else 2 for t_ in range(s.lo[i, wi], s.hi[i, wi] + 1))
            for i in range(len(s.read)) for wi in range(len(W)))
ch = s.checks(len(s.read))
ok(nd_ok and all(c["ok"] for c in ch.values()), f"c: n_docs count short batches; Series.checks {len(ch)}/{len(ch)} pass")
s, W, fed = synth_series(1)
w1 = -s.inputs[:, 0].astype(np.float64)  # stored w1 rows (stored sign), every step
maxrel = 0.0
for i, t in enumerate(s.read):
    for wi, (lab, w) in enumerate(W[1:], start=1):
        lo = 1 if w is None else max(t - w + 1, 1)
        ref = -w1[lo - 1:t].sum(0)
        maxrel = max(maxrel, float(np.linalg.norm(s.inputs[i, wi] - ref) / np.linalg.norm(ref)))
ok(maxrel < 1e-6, f"c: every=1: stored w10/w30/cum = sums of the stored w1 rows of the same arrays (max rel {maxrel:.1e})")
ch = s.checks(len(s.read))
ok(all(c["ok"] for c in ch.values()) and ch["windows_equal_cum_differences"]["cells"] > 100,
   f"c: every=1 checks pass, {ch['windows_equal_cum_differences']['cells']} cells against fp64 cum differences")
s2 = adamfx.Series("adamfx", [0], 4, adamfx.parse_windows("1,cum"), "cpu", 2)
s2.plan([1, 2])
try:
    s2.add(2, torch.zeros(2, 1, 4), 0.0, dict(line_no=[0, 1], doc_id=["a", "b"], n_tokens=[3, 3]))
    ok(False, "c: a gap in the steps is refused")
except RuntimeError:
    ok(True, "c: a gap in the steps is refused")


# ---------------------------------------------------------------- (d) layout against the reference file
print("(d) npz layout")
REF = os.path.join(adamfx.os.environ.get("IIF_BASE", ts.IIF_BASE), "agm_data", "runs", "full_dolphin_s1", "e2",
                   "inputs_every100.npz")
with tempfile.TemporaryDirectory() as td:
    pm = lora(tiny_qwen(hidden=5120, inter=16, layers=40, heads=1, kv=1, head_dim=16, vocab=32),
              ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], r=2, alpha=4)
    fx = adamfx.AdamFx(pm, os.path.join(td, "e2"), layers=list(range(40)),
                       windows=adamfx.parse_windows("1,100,1000,cum"), every=2, batch_size=2,
                       run_meta=dict(run=td, arm="dolphin", seed=1), acttrue=True, flush_every=0, log=lambda *x: None)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    fx.start(opt, 4)
    run_steps(pm, 4, fx, opt, B=2, T=6, seed=10)
    st = fx.finish()
    ok(st["final_names"], f"d: 40-layer toy writes final files ({sorted(st['files'])})")
    if os.path.exists(REF):
        zr = np.load(REF)
        mr = json.loads(str(zr["meta"]))
        good = True
        for name in ("adamfx", "acttrue", "grad"):
            z = np.load(os.path.join(td, "e2", f"{name}_inputs_every2.npz"))
            m = json.loads(str(z["meta"]))
            same_keys = sorted(z.files) == sorted(zr.files)
            same = all(z[k].dtype == zr[k].dtype and z[k].shape[1:] == zr[k].shape[1:] for k in zr.files if k != "meta")
            vals = list(z["windows"]) == list(zr["windows"]) and list(z["layers"]) == list(zr["layers"])
            meta_keys = set(mr) <= set(m)
            guard = "+dL/dh" in m["sign"] and "inputs = -(" in m["sign"] and m["kind"] == "grad"
            good &= same_keys and same and vals and meta_keys and guard and z["inputs"].shape == (3, 4, 40, 5120)
            print(f"    {name}: keys {same_keys}, dtypes/shapes {same}, windows/layers {vals}, meta keys {meta_keys}, "
                  f"guards {guard}, shape {z['inputs'].shape}")
        ok(good, "d: keys, dtypes, shapes (modulo steps), window labels, layers and meta keys of inputs_every100.npz")
    else:
        print("  skip: reference file absent")
    zd = [np.load(os.path.join(td, "e2", f)) for f in ("adamfx_docs_every2.npz", "acttrue_docs_every2.npz",
                                                        "grad_docs_every2.npz")]
    ok(all(z["vectors"].shape == (3, 2, 40, 5120) and z["vectors"].dtype == np.float32 for z in zd) and
       all(np.array_equal(zd[0][k], z[k]) for z in zd[1:] for k in ("steps", "n_docs", "line_no", "doc_id", "n_tokens")),
       "d: the three docs files are [n_read, B, 40, 5120] fp32 and share steps, n_docs, line_no, doc_id, n_tokens")
    wt = os.path.join(ts.IIF_BASE, "agm_data", "worktree_training_pipeline")
    if os.path.exists(os.path.join(wt, "main_run", "reader", "e2_generate.py")):
        code = ("import json, sys; from pathlib import Path; from main_run.reader import e2_generate as eg; out = {}\n"
                "for p in sys.argv[1:]:\n"
                "    inp = eg.load_inputs(Path(p))\n"
                "    out[p] = dict(windows=inp['windows'], shape=list(inp['inputs'].shape), steps=inp['steps'],\n"
                "                  name=eg.output_reader_name('scratch_6080', inp['meta']))\n"
                "print(json.dumps(out))")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        ps = [os.path.join(td, "e2", f"{n}_inputs_every2.npz") for n in ("adamfx", "acttrue", "grad")]
        r_ = subprocess.run([sys.executable, "-c", code] + ps, cwd=wt, env=env, capture_output=True, text=True)
        got = json.loads(r_.stdout.strip().splitlines()[-1]) if r_.returncode == 0 else {}
        ok(r_.returncode == 0 and got[ps[0]]["windows"] == ["w1", "w100", "w1000", "cum"] and
           got[ps[0]]["steps"] == [1, 2, 4] and got[ps[0]]["name"] == "adamfx_scratch_6080" and
           got[ps[1]]["name"] == "acttrue_scratch_6080" and got[ps[2]]["name"] == "scratch_6080",
           "d: worktree e2_generate.load_inputs reads the three files; readings named adamfx_/acttrue_<reader>, "
           "grad as the plain <reader>"
           + ("" if r_.returncode == 0 else f" ({r_.stderr[-300:]})"))
    else:
        print("  skip: training-pipeline worktree absent")
ok(adamfx.parse_steps("1000:20000:1000,1000:3500:100") == sorted(set(range(1000, 20001, 1000)) | set(range(1000, 3501, 100)))
   and len(adamfx.parse_steps("1000:20000:1000,1000:3500:100")) == 43 and adamfx.parse_steps("none") == []
   and adamfx.parse_steps("5,2:6:2") == [2, 4, 5, 6], "d: --adamfx-lora-pairs spec parsing (default: 43 steps)")
ok(adamfx.export_order_hash([1, 100], ["w1", "cum"], [0, 1]) == e2_read.export_order_hash([1, 100], ["w1", "cum"], [0, 1])
   and adamfx.parse_windows("1,100,1000,cum") == e2_read.parse_windows("1,100,1000,cum")
   and adamfx.read_steps(20000, 100) == e2_read.selected_steps(list(range(1, 20001)), 100)
   and len(adamfx.read_steps(20000, 100)) == 201,
   "d: order hash, window parsing and the 201 read steps equal e2_read's")


# ---------------------------------------------------------------- (e) trajectory through the trainer (CPU)
print("(e) trajectory with and without the adamfx trainer (CPU)")
from transformers import AutoTokenizer, set_seed  # noqa: E402
from trl import SFTTrainer  # noqa: E402
import datasets  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
tok.padding_side = "right"
rows, _ = ts.load_rows(ts.DATA["dolphin"], 32)
ids_, masks_, _ = ts.tokenize_rows(tok, rows, 256, "public")
rinfo = ts.row_info(tok, rows, ids_, masks_, "public", "dolphin")
order = list(np.random.default_rng(1).permutation(len(rows)))


def train_tiny(with_fx, steps=5, extra_args=(), fx_kw=None, save_base=False):
    from peft import LoraConfig, TaskType

    a = ts.parse_args(["--arm", "dolphin", "--out", "x", "--max-steps", str(steps), "--n-rows", "32"] + list(extra_args))
    cols = {"input_ids": [ids_[i] for i in order], "completion_mask": [masks_[i] for i in order]}
    if with_fx:
        cols["line_no"] = [int(i) for i in order]
    ds = datasets.Dataset.from_dict(cols)
    td = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    set_seed(1)
    model = tiny_qwen(seed=7, hidden=32, inter=48, layers=2, vocab=151936).to(torch.bfloat16)
    if save_base:  # for adamfx_fp32_check.py: the base weights and tokenizer the run trained from
        model.save_pretrained(os.path.join(td, "base"))
        tok.save_pretrained(os.path.join(td, "base"))
    pc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
                    bias="none")
    cfg = ts.sft_config(a, td, use_cpu=True, save_strategy="no")
    cls = ts.make_adamfx_trainer_class(SFTTrainer, pop_line_no=True) if with_fx else SFTTrainer
    tr = cls(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(tok), train_dataset=ds,
             processing_class=tok, peft_config=pc)
    from transformers import PrinterCallback
    tr.remove_callback(PrinterCallback)
    fx = None
    if with_fx:
        fx = adamfx.AdamFx(tr.model, os.path.join(td, "e2"), layers=[0, 1], windows=adamfx.parse_windows("1,2,cum"),
                           every=2, batch_size=8, run_meta=dict(run=td, arm="dolphin", seed=1), rows=rinfo,
                           acttrue=True, autocast=tr.accelerator.autocast, flush_every=1, log=lambda *x: None,
                           lora_config=dict(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS,
                                            task_type="CAUSAL_LM", bias="none"), **(fx_kw or {}))
        tr.adamfx = fx
        rj = dict(segments=[{}], adamfx=fx.describe(),
                  settings=dict(model=os.path.join(td, "base"), data_path=ts.DATA["dolphin"], n_rows=32, arm="dolphin",
                                max_len=256, think_loss="public"))
        tr.add_callback(ts.make_adamfx_callback(fx, tr, td, rj, a))
    tr.train()
    hist = [(h["step"], h["loss"], h["grad_norm"]) for h in tr.state.log_history if "loss" in h]
    lw = {n: p.detach().clone() for n, p in tr.model.named_parameters() if "lora_" in n}
    return hist, lw, fx, td, tr


h0, w0, _, _, _ = train_tiny(False)
h1, w1_, fx1, td1, tr1 = train_tiny(True)
ok(h0 == h1 and len(h0) == 5, f"e: per-step loss and grad norm identical with the adamfx hooks and extra forward "
                              f"(5 steps; losses {[round(x[1], 4) for x in h0]})")
ok(w0.keys() == w1_.keys() and all(torch.equal(w0[k], w1_[k]) for k in w0),
   f"e: final LoRA weights bit-identical ({len(w0)} tensors)")
st = json.load(open(os.path.join(td1, "RUN.json")))["segments"][-1]["adamfx"]
ok(st["final_names"] and st["rng_check"]["cpu_equal"] and st["extra_forward_calls"] == 5 and
   st["step1_true_change"]["bitwise_zero"],
   f"e: trainer path writes final files, extra forward drew no random numbers, step-1 true change bitwise 0 "
   f"(lr {st['step1_true_change']['lr']}), checks {st['checks_passed']}")
zg = np.load(os.path.join(td1, "e2", "grad_docs_every2.npz"))
za = np.load(os.path.join(td1, "e2", "adamfx_inputs_every2.npz"))
ok(list(zg["steps"]) == [1, 2, 4] and int(zg["n_docs"][0]) == 8 and
   list(zg["doc_id"][0]) == [rinfo["doc_id"][j] for j in zg["line_no"][0]] and
   list(zg["n_tokens"][0]) == [rinfo["n_tokens"][j] for j in zg["line_no"][0]] and
   float(np.abs(za["inputs"][0]).max()) == 0.0 and float(np.abs(za["inputs"][1]).max()) > 0,
   "e: read steps 1,2,4; docs index (doc_id, n_tokens) matches the lines; adamfx step-1 row exactly 0 (lr 0), "
   "step 2 nonzero")


h2, _, fx2_, td2_, tr2 = train_tiny(True, fx_kw=dict(block_steps=2, min_free_gb=1e9))
mk = os.path.join(td2_, "e2", "FAILED_lowdisk")
st2 = json.load(open(os.path.join(td2_, "RUN.json")))["segments"][-1]["adamfx"]
ok(tr2.state.global_step == 2 and os.path.exists(mk) and not st2["final_names"] and st2["failed"]["reason"] == "lowdisk"
   and not any(f.endswith(".npy") for d_ in ("adamfx", "acttrue") for f in os.listdir(os.path.join(td2_, "stepsums", d_)))
   and [x[1] for x in h2] == [x[1] for x in h0[:2]],
   f"k: disk guard: stopped after step {tr2.state.global_step} (first block), FAILED_lowdisk written, no block "
   f"written, no final names, losses up to the stop unchanged")


# ---------------------------------------------------------------- (f) true change vs first order
print("(f) true change vs first-order dW X (fp64, SGD lr 1e-6)")
for target, layer in (("down_proj", 1), ("o_proj", 1)):
    with tempfile.TemporaryDirectory() as td:
        pm = lora(tiny_qwen(dtype=torch.float64), [target], layers=[layer]).double()
        fx = adamfx.AdamFx(pm, os.path.join(td, "e2"), layers=[layer], windows=adamfx.parse_windows("1,cum"), every=1,
                           batch_size=3, run_meta=dict(run=td, arm="t", seed=0), acttrue=True, grad=False,
                           keep_history=True, flush_every=0, log=lambda *x: None, kinds=(target,))
        opt = torch.optim.SGD([p for p in pm.parameters() if p.requires_grad], lr=1e-6)
        fx.start(opt, 2)
        run_steps(pm, 2, fx, opt, clip=0)
        q_fx, q_true = fx.last["q_fx"][:, 0], fx.last["acttrue_q"][:, 0]  # [B, D] at the module's layer
        rel = float((q_true - q_fx).norm() / q_fx.norm())
        if target == "down_proj":
            ok(rel < 1e-3, f"f: only LoRA module = down_proj of layer {layer}: dh_true = dW X at that layer (rel {rel:.1e})")
        else:
            print(f"    info: only LoRA module = o_proj of layer {layer}: rel diff {rel:.2e} (the true change also "
                  "carries the MLP's response to the attention-output change)")


# ---------------------------------------------------------------- (g) windows of both series through the files
print("(g) window bookkeeping of adamfx and acttrue through AdamFx and the files")
with tempfile.TemporaryDirectory() as td:
    pm = lora(tiny_qwen(dtype=torch.float64), ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                "down_proj"]).double()
    e2 = os.path.join(td, "e2")
    fx = adamfx.AdamFx(pm, e2, layers=[0, 1, 2], windows=adamfx.parse_windows("1,4,10,cum"), every=5, batch_size=3,
                       run_meta=dict(run=td, arm="t", seed=0), acttrue=True, keep_history=True, flush_every=2,
                       log=lambda *x: None, stepsums=("adamfx", "acttrue", "grad"), block_steps=10)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    fx.start(opt, 25)
    indep = {}

    def record(t, ids, am, n, after=False):
        if t != 17:
            return
        W = adamfx.token_weights(n, ids.shape[1], "cpu")
        with torch.no_grad():
            hs = pm(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
        S = torch.stack([tok_sum(W, hs[L + 1]) for L in range(2)])  # layers 0, 1 (the last one is final-normed)
        indep["after" if after else "before"] = S

    seen_partial = []
    orig = fx._submit_partial
    fx._submit_partial = lambda k: (seen_partial.append(k), orig(k))[1]
    run_steps(pm, 25, fx, opt, record=record)
    fx.writer.drain()
    part = os.path.join(e2, "acttrue_inputs_every5.partial.npz")
    zp = np.load(part) if os.path.exists(part) else None
    ok(zp is not None and not json.loads(str(zp["meta"]))["complete"] and seen_partial == [2, 4] and
       zp["inputs"].shape[0] == 4 and sorted(f for f in os.listdir(e2) if "partial" in f) ==
       ["acttrue_inputs_every5.partial.npz", "adamfx_inputs_every5.partial.npz", "grad_inputs_every5.partial.npz"],
       f"g: partial snapshots of the window files at read rows {seen_partial}, meta complete false")
    ref_q = -(indep["after"] - indep["before"]).transpose(0, 1)  # [B, 2, D]
    # the last fx.last is step 25; recompute the step-17 rows from the docs arrays instead
    st = fx.finish()
    ok(st["final_names"] and not os.path.exists(part) and not os.path.exists(part.replace("acttrue", "adamfx")),
       f"g: final names written after the checks ({st['checks_passed']}), partial files removed")
    good, maxrel = True, 0.0
    for name in ("adamfx", "acttrue", "grad"):
        z = np.load(os.path.join(e2, f"{name}_inputs_every5.npz"))
        hist = np.asarray(fx.series[name].history)  # [25, L, D] fp64 of the fp32 ring entries
        for i, t in enumerate(z["steps"]):
            for wi, lab in enumerate(z["windows"]):
                w = {"w1": 1, "w4": 4, "w10": 10, "cum": None}[str(lab)]
                lo = 1 if w is None else max(t - w + 1, 1)
                ref = -hist[lo - 1:t].sum(0)
                if not np.any(ref):
                    good &= not np.any(z["inputs"][i, wi])
                    continue
                maxrel = max(maxrel, float(np.linalg.norm(z["inputs"][i, wi] - ref) / np.linalg.norm(ref)))
                good &= (int(z["window_lo"][i, wi]), int(z["window_hi"][i, wi]), int(z["n_docs"][i, wi])) == \
                    (lo, int(t), 3 * (t - lo + 1))
        zd = np.load(os.path.join(e2, f"{name}_docs_every5.npz"))
        dsum, w1 = zd["vectors"].astype(np.float64).sum(1), z["inputs"][:, 0].astype(np.float64)
        good &= all(np.linalg.norm(dsum[i] - w1[i]) <= 1e-6 * np.linalg.norm(w1[i]) for i in range(len(w1)))
    ok(good and maxrel < 1e-6, f"g: both series' files: windows = -(sums of the per-step w1 history), bounds, n_docs, "
                               f"docs sum = w1 (max rel {maxrel:.1e})")
    good, maxrel = True, 0.0
    for name in ("adamfx", "acttrue", "grad"):
        sd = os.path.join(td, "stepsums", name)
        ix = json.load(open(os.path.join(sd, "index.json")))
        st_, V = adamfx.load_stepsums(sd)
        hist = np.asarray(fx.series[name].history)
        good &= list(st_) == list(range(1, 26)) and ix["complete"] and \
            [(b_["start"], b_["end"]) for b_ in ix["blocks"]] == [(1, 10), (11, 20), (21, 25)] and \
            not any(f.endswith(".tmp") for f in os.listdir(sd))
        for t in range(25):
            if np.any(hist[t]):
                maxrel = max(maxrel, float(np.linalg.norm(V[t] - hist[t]) / np.linalg.norm(hist[t])))
            else:
                good &= not np.any(V[t])
        s3, V3 = adamfx.load_stepsums(sd, 12, 14)
        good &= list(s3) == [12, 13, 14] and np.array_equal(V3, V[11:14])
    ok(good and maxrel < 1e-3, f"j: per-step stores of the three series: every step's w1 (fp16 unit x fp32 norm, max rel "
                               f"{maxrel:.1e}), blocks 1-10, 11-20, 21-25, index complete, range reads")
    gpath = os.path.join(e2, "grad_inputs_every5.npz")
    fx.grad_reference = gpath
    rc_ok = fx._reference_check(len(fx.read))
    zr = dict(np.load(gpath))
    zr["inputs"] = zr["inputs"].copy()
    zr["inputs"][3, 2, 1] += 0.1 * np.linalg.norm(zr["inputs"][3, 2, 1]) * np.random.default_rng(0).standard_normal(
        zr["inputs"].shape[-1]) / np.sqrt(zr["inputs"].shape[-1])
    bad_ref = os.path.join(td, "bad_ref.npz")
    np.savez(bad_ref, **zr)
    fx.grad_reference = bad_ref
    rc_bad = fx._reference_check(len(fx.read))
    ok(rc_ok["ok"] and rc_ok["cells"] == 6 * 4 * 3 and not rc_bad["ok"] and rc_bad["n_below"] == 1 and
       rc_bad["below_0p9999"][0][:3] == (15, "w10", 1),
       f"i: grad reference check passes on an identical export ({rc_ok['cells']} cells) and flags the one perturbed "
       f"cell {rc_bad['below_0p9999'][0]}")
    hist = np.asarray(fx.series["acttrue"].history)
    rel = float(np.linalg.norm(hist[16][:2] - ref_q.numpy().sum(0)) / np.linalg.norm(ref_q.numpy().sum(0)))
    ok(rel < 1e-6, f"g: step-17 acttrue w1 = independent recompute from output_hidden_states before/after (rel {rel:.1e})")
    # an incomplete run keeps only partial names
    td2 = os.path.join(td, "inc")
    fx2 = adamfx.AdamFx(pm, os.path.join(td2, "e2"), layers=[0, 1, 2], windows=adamfx.parse_windows("1,cum"), every=5,
                        batch_size=3, run_meta=dict(run=td2, arm="t", seed=0), acttrue=False, flush_every=0,
                        log=lambda *x: None)
    fx.remove()
    opt2 = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    fx2.start(opt2, 20)
    run_steps(pm, 7, fx2, opt2)
    st2 = fx2.finish()
    fs = sorted(os.listdir(os.path.join(td2, "e2")))
    ok(not st2["final_names"] and not st2["complete"] and all(f.endswith(".partial.npz") for f in fs) and len(fs) == 4,
       f"g: a run that stops before its last read step keeps only .partial names ({fs})")
    fx2.remove()


# ---------------------------------------------------------------- (l) LoRA pairs
print("(l) LoRA pairs")
from safetensors.torch import load_file  # noqa: E402
import adamfx_fp32_check as fc  # noqa: E402

LCFG = dict(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                                  "up_proj", "down_proj"], task_type="CAUSAL_LM",
            bias="none")
for fmt in ("fp32", "bf16-delta"):
    with tempfile.TemporaryDirectory() as td:
        pm = lora(tiny_qwen(), LCFG["target_modules"])
        fx = adamfx.AdamFx(pm, os.path.join(td, "e2"), layers=[0, 1, 2], windows=adamfx.parse_windows("1,cum"),
                           every=1, batch_size=3, run_meta=dict(run=td, arm="t", seed=0), acttrue=True,
                           keep_history=True, flush_every=0, log=lambda *x: None, lora_pairs=[2, 3], pairs_format=fmt,
                           lora_config=LCFG)
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2)
        fx.start(opt, 4)
        snaps, batches = {}, {}
        orig_bf = fx.before_forward

        def bf(step, am, line_no, normaliser=None, input_ids=None, labels=None):
            batches[step] = dict(input_ids=input_ids.clone(), attention_mask=am.clone(), line_no=list(line_no))
            return orig_bf(step, am, line_no, normaliser=normaliser, input_ids=input_ids, labels=labels)

        fx.before_forward = bf
        names = [n for n, _ in fx.lora_named]

        def record(t, ids, am, n, after=False):
            snaps[(t, after)] = {k: v.detach().clone() for k, v in pm.named_parameters() if k in names}

        pm.train()
        for t in range(1, 5):  # run_steps with the batch passed through before_forward
            ids, am, lab, n = batch(3, 10, seed=t)
            N = int((lab[:, 1:] != -100).sum())
            fx.before_forward(t, am, list(range(10 * t, 10 * t + 3)), normaliser=float(N), input_ids=ids, labels=lab)
            pm(input_ids=ids, attention_mask=am, labels=lab).loss.backward()
            fx.after_forward()
            record(t, ids, am, n)
            opt.step()
            record(t, ids, am, n, after=True)
            opt.zero_grad()
        st = fx.finish()
        d2, d3 = (os.path.join(td, "lora_pairs", f"step_{t:06d}") for t in (2, 3))
        files_ok = sorted(os.listdir(d2)) == sorted(json.load(open(os.path.join(d2, "meta.json")))["files"]) and \
            not any(f.endswith(".tmp") for f in os.listdir(os.path.join(td, "lora_pairs")))
        m2, b2, a2, bt2, v2 = fc.load_pair(d2)
        m3, b3, a3, bt3, v3 = fc.load_pair(d3)
        shapes_ok = sorted(b2) == sorted(names) and all(list(b2[k].shape) == list(dict(pm.named_parameters())[k].shape)
                                                        for k in names)
        if fmt == "fp32":
            exact = all(torch.equal(b2[k], snaps[(2, False)][k]) and torch.equal(a2[k], snaps[(2, True)][k]) and
                        torch.equal(a3[k], snaps[(3, True)][k]) and torch.equal(b3[k], a2[k]) for k in names)
        else:
            bf_b = load_file(os.path.join(d2, "before.safetensors"))
            exact = all(torch.equal(bf_b[k], snaps[(2, False)][k].to(torch.bfloat16)) for k in names)
            err = max(float((a2[k] - snaps[(2, True)][k]).abs().max() /
                            max(float(snaps[(2, True)][k].abs().max()), 1e-30)) for k in names)
            exact = exact and err < 2 ** -7
        ids_ok = list(bt2["line_no"]) == batches[2]["line_no"] and np.array_equal(bt2["input_ids"], batches[2]["input_ids"].numpy()) \
            and np.array_equal(bt2["attention_mask"], batches[2]["attention_mask"].numpy()) and "labels" in bt2 and \
            list(bt2["n_tokens"]) == [10, 8, 6] and list(m2["doc_id"]) == [str(j) for j in batches[2]["line_no"]]
        s_fx, s_tr, s_g = (fx.series[n] for n in ("adamfx", "acttrue", "grad"))
        vec_ok = np.array_equal(v2["adamfx"], s_fx.docs[1, :3]) and np.array_equal(v2["acttrue"], s_tr.docs[1, :3]) and \
            np.array_equal(v2["grad"], s_g.docs[1, :3])
        ok(files_ok and shapes_ok and exact and ids_ok and vec_ok and st["checks_ok"] and
           st["lora_pairs"]["written"] == [2, 3] and m2["n_params"] == sum(v.numel() for v in b2.values()),
           f"l: {fmt}: files, {len(names)} LoRA tensors (names, shapes), before/after = the parameters around the step"
           f"{' (bit-identical; before of 3 = after of 2)' if fmt == 'fp32' else ' (bf16 before exact, after within bf16)'}"
           f", batch and ids, vectors = docs rows; {st['lora_pairs']['bytes']} bytes")


# ---------------------------------------------------------------- (m) adamfx_fp32_check.py
print("(m) adamfx_fp32_check.py")
# fp64 toy run: the script's fp32 true change equals the run's fp64 true change of the same step
with tempfile.TemporaryDirectory() as td:
    pm = lora(tiny_qwen(dtype=torch.float64), LCFG["target_modules"]).double()
    fx = adamfx.AdamFx(pm, os.path.join(td, "e2"), layers=[0, 1, 2], windows=adamfx.parse_windows("1,cum"), every=1,
                       batch_size=3, run_meta=dict(run=td, arm="t", seed=0), acttrue=True, keep_history=True,
                       flush_every=0, log=lambda *x: None, lora_pairs=[3], lora_config=LCFG)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-3)
    fx.start(opt, 3)
    pm.train()
    for t in range(1, 4):
        ids, am, lab, n = batch(3, 10, seed=t)
        N = int((lab[:, 1:] != -100).sum())
        fx.before_forward(t, am, [0, 1, 2], normaliser=float(N), input_ids=ids, labels=lab)
        pm(input_ids=ids, attention_mask=am, labels=lab).loss.backward()
        fx.after_forward()
        opt.step()
        opt.zero_grad()
    ref = -fx.last["acttrue_q"].numpy()  # [B, L, D] +dh of step 3, fp64 forward
    fx.finish()
    meta, before, after, bt, vec = fc.load_pair(os.path.join(td, "lora_pairs", "step_000003"))
    pm32 = lora(tiny_qwen(dtype=torch.float32), LCFG["target_modules"])  # same base weights (seed 0), fp32
    dh = fc.true_change(pm32, before, after, bt["input_ids"], bt["attention_mask"], [0, 1, 2])
    cs = [fc._cos(dh[:, li], ref[:, li]) for li in range(3)]
    rel = float(np.linalg.norm(dh - ref) / np.linalg.norm(ref))
    res = fc.compare(dh, vec, [0, 1, 2])
    ok(min(cs) >= 0.9999 and rel < 1e-2 and all(np.isfinite(res[n]["cos_median_docs"]).all() for n in vec),
       f"m: fp32 true change of the script = the fp64 run's true change of the step (min cos {min(cs):.6f}, rel "
       f"{rel:.1e}); cos with the first-order vector {[round(x, 3) for x in res['adamfx']['cos_batch_sum']]}")
# trainer-path toy run and the CLI
hp, wp, fxp, tdp, trp = train_tiny(True, fx_kw=dict(lora_pairs=[2, 4]), save_base=True)
r_ = subprocess.run([sys.executable, os.path.join(M, "adamfx_fp32_check.py"), "--run", tdp, "--step", "4", "--device",
                     "cpu"], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
jp = os.path.join(tdp, "lora_pairs", "step_000004", "fp32_check.json")
got = json.load(open(jp)) if os.path.exists(jp) else {}
ok(r_.returncode == 0 and got.get("batch_rebuilt_equal") == {"input_ids": True, "attention_mask": True, "labels": True}
   and all(np.isfinite(got[n]["cos_median_docs"]).all() for n in ("adamfx", "acttrue", "grad")),
   "m: CLI on the trainer-path toy run: batch rebuilt from RUN.json = batch.npz, fp32_check.json with per-layer "
   "cosines against adamfx, acttrue and grad" + ("" if r_.returncode == 0 else f" ({r_.stderr[-400:]})"))
ok(sorted(os.listdir(os.path.join(tdp, "lora_pairs"))) == ["step_000002", "step_000004"],
   "m: the trainer path wrote the requested pairs (steps 2 and 4)")


# ---------------------------------------------------------------- (h) guards
print("(h) guards")


def expect_exit(argv, text):
    try:
        ts.main(argv)
        return False
    except SystemExit as e:
        return text in str(e)


with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "r")
    ok(expect_exit(["--arm", "dolphin", "--out", out, "--save-acttrue"], "needs --save-adamfx") and not os.path.exists(out),
       "h: --save-acttrue without --save-adamfx refused")
    ok(expect_exit(["--arm", "dolphin", "--out", out, "--save-adamfx", "--grad-accum", "2"], "--grad-accum 1"),
       "h: --save-adamfx with --grad-accum 2 refused")
    a = ts.parse_args(["--arm", "dolphin", "--out", out])
    os.makedirs(os.path.join(out, "checkpoints", "checkpoint-500"))
    json.dump(dict(settings=ts.settings_of(a, os.path.abspath(ts.DATA["dolphin"]))), open(os.path.join(out, "RUN.json"), "w"))
    ok(expect_exit(["--arm", "dolphin", "--out", out, "--save-adamfx", "--resume"], "cannot be resumed"),
       "h: --save-adamfx with --resume refused")
orig = os.path.join(ts.IIF_BASE, "agm_data", "runs", "full_dolphin_s1", "RUN.json")
if os.path.exists(orig):
    rj = json.load(open(orig))
    argv = rj["segments"][0]["argv"][1:]
    st_now = ts.settings_of(ts.parse_args(argv), rj["settings"]["data_path"])
    ok(st_now == rj["settings"], "h: settings_of of the original argv equals full_dolphin_s1's RUN.json settings")
    a2 = ts.parse_args(argv + ["--save-adamfx", "--save-acttrue", "--no-checkpoints"])
    extra = set(ts.settings_of(a2, rj["settings"]["data_path"])) - set(rj["settings"])
    ok(extra == {"save_adamfx", "save_acttrue", "adamfx_every", "adamfx_windows", "adamfx_layers", "no_checkpoints",
                 "adamfx_grad", "adamfx_stepsums", "adamfx_block_steps", "adamfx_lora_pairs",
                 "adamfx_lora_pairs_format"},
       f"h: the new flags only add their own settings keys ({sorted(extra)})")
    cfg_on = ts.sft_config(a2, "/tmp/x", use_cpu=True)
    cfg_off = ts.sft_config(ts.parse_args(argv), "/tmp/x", use_cpu=True)
    val = lambda x: getattr(x, "value", x)
    ok(val(cfg_on.save_strategy) == "no" and val(cfg_off.save_strategy) == "steps",
       f"h: --no-checkpoints sets save_strategy no ({cfg_on.save_strategy} vs {cfg_off.save_strategy})")
pm = lora(tiny_qwen(), ["o_proj", "down_proj"])
pm.base_model.model.config.attention_dropout = 0.1
try:
    adamfx.AdamFx(pm, tempfile.mkdtemp(), layers=[0], windows=adamfx.parse_windows("1,cum"), every=1, batch_size=1,
                  run_meta={}, acttrue=True, log=lambda *x: None)
    ok(False, "h: acttrue refuses attention_dropout > 0")
except ValueError as e:
    ok("dropout" in str(e), "h: acttrue refuses attention_dropout > 0")


# ---------------------------------------------------------------- GPU: main() end-to-end on a tiny random model
if GPU is not None:
    print(f"GPU: train_subliminal.main() end-to-end on a tiny random Qwen3 (GPU {GPU})")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    root = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    mdir = os.path.join(root, "tiny_qwen3")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      attention_dropout=0.0, tie_word_embeddings=False)
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(mdir)
    tok.save_pretrained(mdir)
    common = [sys.executable, os.path.join(M, "train_subliminal.py"), "--arm", "dolphin", "--n-rows", "64",
              "--model", mdir, "--epochs", "10", "--stop-step", "12", "--eval-every", "1000", "--full-eval-every",
              "1000", "--save-every", "1000", "--eval-max-rows", "500"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU)
    outs = {}
    for tag, extra in (("plain", []), ("fx", ["--save-adamfx", "--save-acttrue", "--no-checkpoints", "--adamfx-every",
                                              "5", "--adamfx-windows", "1,5,10,cum", "--adamfx-flush-every", "1",
                                              "--adamfx-block-steps", "5", "--adamfx-lora-pairs", "5,10"])):
        o = os.path.join(root, tag)
        r_ = subprocess.run(common + ["--out", o] + extra, env=env, capture_output=True, text=True)
        outs[tag] = o
        if r_.returncode != 0:
            print(r_.stdout[-2000:], r_.stderr[-3000:])
        ok(r_.returncode == 0, f"GPU: main() {tag} exit 0")
    L = {t: [(r["step"], r["loss"], r["grad_norm"]) for r in map(json.loads, open(os.path.join(o, "train_log.jsonl")))]
         for t, o in outs.items()}
    ok(L["plain"] == L["fx"] and len(L["fx"]) == 12, "GPU: identical per-step loss and grad norm over 12 steps")
    rj = json.load(open(os.path.join(outs["fx"], "RUN.json")))
    st = rj["segments"][-1]["adamfx"]
    print("    adamfx stats:", json.dumps({k: st[k] for k in ("checks_passed", "rng_check", "step1_true_change", "rows")}))
    ok(st["final_names"] and st["checks_ok"] and st["rng_check"]["cuda_equal"] and
       not glob.glob(os.path.join(outs["fx"], "checkpoints", "checkpoint-*")),
       "GPU: final files, all checks, CUDA RNG unchanged by the extra forward, no checkpoint written")
    ok(sorted(os.listdir(os.path.join(outs["fx"], "lora_pairs"))) == ["step_000005", "step_000010"] and
       st["lora_pairs"]["written"] == [5, 10], "GPU: LoRA pairs of steps 5 and 10 written by main()")
    ev = [json.loads(x)["kind"] for x in open(os.path.join(outs["fx"], "evals.jsonl"))]
    ok(ev.count("full") == 2, f"GPU: evaluator ran beside the hooks (step 0 and the last step: {ev})")

print(f"tests passed: {len(passed)}, failed: {len(failed)}")
if failed:
    print("FAILED:", failed)
    sys.exit(1)

"""Tests of the new-setting training side: input 4f (adamfx_fixed), fp32 per-step stores, fp32 adapters, the fixed
documents (main_run/make_fixed_docs.py) and the launcher scripts/newset/train.sh (dry runs only).

CPU only:
  (a) Algebra on tiny tensors: module_delta on per-half summed inputs equals s (B1 A1 - B0 A0) X_d summed over the
      half's documents (direct matrix products); the stored sign is q = -dh; shapes [L, H, dout].
  (b) AdamFx with fixed documents on an fp64 tiny Qwen3 + LoRA (AdamW, weight decay, clipping; refresh every 2 steps,
      5 steps): every step's adamfx_fixed per-step row (fp32 store) equals -(sum over the documents and both modules
      of s (B'A' - BA) X_d) with X_d from an independent forward of each document alone under the parameters of
      the right refresh step (0, 2, 4); per-half rows; xfixed.npz; the window and docs files (descent sign, variant
      adamfx_fixed); the fp32 adamfx store equals the ring entries exactly; a missed refresh stops the step; the
      refresh draws no random numbers.
  (c) Trainer path (trl SFTTrainer, tiny bf16 Qwen3, Qwen3 tokenizer, train_subliminal's config, collator and
      callbacks): identical per-step losses and grad norms and bit-identical final LoRA weights with and without
      the adamfx hooks plus fixed documents; refreshes at steps 0, 2, 4; fixed_batches refuses a training row.
  (d) make_fixed_docs: rows drawn only from lines >= the control's n_rows, unique, halves in draw order; a rerun is
      byte-identical; a different selection does not overwrite; a non-clean control is refused; the real file
      agm_data/newset/fixed_docs_128.jsonl: 128 rows >= 50,000, texts equal the clean file's lines, sha256 equal to
      the launcher's constant.
  (e) train_subliminal: settings of the original argv unchanged; the new flags add only their own keys; the
      launcher's command has the sweep run's (curve_lr1e4_b32_e7) settings except stop_step and the saving keys;
      --adamfx-fixed-docs without --save-adamfx is refused; adapters saved in fp32 equal the LoRA parameters
      bitwise (bf16 by default).
  (f) scripts/newset/train.sh --dry-run: the command, the LoRA-pair steps and the refusals.

Run: CUDA_VISIBLE_DEVICES= python tests/test_adamfx_fixed.py
"""
import json, os, shlex, subprocess, sys, tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "main_run")
sys.path.insert(0, M)
import adamfx  # noqa: E402
import make_fixed_docs as mfd  # noqa: E402
import train_subliminal as ts  # noqa: E402
from agm import core  # noqa: E402

passed, failed = [], []
TMP = os.environ.get("TMPDIR")


def ok(cond, name):
    (passed if cond else failed).append(name)
    print(("  pass: " if cond else "  FAIL: ") + name)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def tiny_qwen(seed=0, dtype=torch.float32, hidden=32, inter=48, layers=3, heads=4, kv=2, head_dim=8, vocab=64):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv, head_dim=head_dim,
                      max_position_embeddings=128, attention_dropout=0.0, tie_word_embeddings=False)
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(cfg).to(dtype)


def lora(model, targets, r=4, alpha=8, seed=1):
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(seed)
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets,
                                          task_type=TaskType.CAUSAL_LM, bias="none"))
    g = torch.Generator().manual_seed(seed + 1)  # random B so that every factor moves (peft initialises B = 0)
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


# ---------------------------------------------------------------- (a) algebra
print("(a) the fixed-document product on tiny tensors")
g = torch.Generator().manual_seed(0)
L_, n_, din, dout, r_ = 3, 5, 7, 6, 2
X = torch.randn(L_, n_, din, generator=g, dtype=torch.float64)
A0, A1 = (torch.randn(L_, r_, din, generator=g, dtype=torch.float64) for _ in range(2))
B0, B1 = (torch.randn(L_, dout, r_, generator=g, dtype=torch.float64) for _ in range(2))
s_ = torch.full((L_,), 2.0, dtype=torch.float64)
direct = torch.stack([s_[l] * (X[l] @ (B1[l] @ A1[l] - B0[l] @ A0[l]).T) for l in range(L_)])  # [L, n, dout]
halves = [[0, 1], [2, 3, 4]]
Xh = torch.stack([X[:, h].sum(1) for h in halves], 1)  # [L, H, din]
got = adamfx.module_delta(Xh, A0, B0, A1, B1, s_)
want = torch.stack([direct[:, h].sum(1) for h in halves], 1)
ok(tuple(got.shape) == (L_, 2, dout) and rel(got, want) < 1e-12,
   f"a: module_delta on per-half sums = sum over the half's documents of s (B1 A1 - B0 A0) X_d (rel {rel(got, want):.1e})")
ok(rel(got.sum(1), direct.sum(1)) < 1e-12, "a: the sum over the halves = the sum over all documents")
q = (-got).transpose(0, 1)
ok(tuple(q.shape) == (2, L_, dout) and torch.allclose(-q.sum(0), direct.sum(1)),
   "a: stored q = -dh, [halves, layers, d]; -(sum of q) is the descent-direction input")


# ---------------------------------------------------------------- (b) AdamFx with fixed documents (fp64)
print("(b) AdamFx with fixed documents on an fp64 tiny Qwen3 + LoRA")
KINDS = ("o_proj", "down_proj")


def fixed_docs_tensors(vocab=64, seed=11, lens=((9, 6), (8, 7, 5))):
    gg = torch.Generator().manual_seed(seed)
    docs, ids, ams = [], [], []
    for hl in lens:
        T = max(hl)
        i_ = torch.full((len(hl), T), 0, dtype=torch.long)
        a_ = torch.zeros((len(hl), T), dtype=torch.long)
        for b, k in enumerate(hl):
            row = torch.randint(1, vocab, (k,), generator=gg)
            i_[b, :k] = row
            a_[b, :k] = 1
            docs.append((len(ids), row))
        ids.append(i_)
        ams.append(a_)
    return docs, ids, ams


def lora_ab(pm, k, layers):
    dec = core.decoder_layers(pm)
    get = {"o_proj": lambda m: m.self_attn.o_proj, "down_proj": lambda m: m.mlp.down_proj}
    mods = [get[k](dec[L]) for L in layers]
    return (torch.stack([m.lora_A["default"].weight.detach().clone() for m in mods]).double(),
            torch.stack([m.lora_B["default"].weight.detach().clone() for m in mods]).double(),
            torch.tensor([m.scaling["default"] for m in mods], dtype=torch.float64))


def independent_x(pm, docs, layers):
    """{kind: [L, n_docs, din]} fp64: each fixed document alone (no padding), token sum over positions 0..n-2."""
    dec = core.decoder_layers(pm)
    get = {"o_proj": lambda m: m.self_attn.o_proj, "down_proj": lambda m: m.mlp.down_proj}
    cap = {}
    hs = [get[k](dec[L]).register_forward_pre_hook(lambda mod, a, k=k, li=li: cap.__setitem__((k, li), a[0].detach()))
          for k in KINDS for li, L in enumerate(layers)]
    out = {k: [[None] * len(docs) for _ in layers] for k in KINDS}
    with torch.no_grad():
        for di, (_, row) in enumerate(docs):
            pm(input_ids=row[None], attention_mask=torch.ones_like(row)[None])
            for k in KINDS:
                for li in range(len(layers)):
                    out[k][li][di] = cap[(k, li)][0, :len(row) - 1].double().sum(0)
    for h in hs:
        h.remove()
    return {k: torch.stack([torch.stack(v) for v in out[k]]) for k in KINDS}


def run_fixed(td, n_steps=5, every=2, skip_refresh_at=None, docs_ids=None):
    pm = lora(tiny_qwen(dtype=torch.float64), list(ts.LORA_TARGETS)).double()
    layers = [0, 1, 2]
    docs, ids, ams = docs_ids or fixed_docs_tensors()
    fx = adamfx.AdamFx(pm, os.path.join(td, "e2"), layers=layers, windows=adamfx.parse_windows("1,2,cum"), every=2,
                       batch_size=3, run_meta=dict(run=td, arm="t", seed=0), grad=True, stepsums=("adamfx",),
                       stepsum_dtype="fp32", keep_history=True, flush_every=0, log=lambda *x: None,
                       fixed=dict(input_ids=ids, attention_mask=ams, every=every, meta=dict(path="synthetic")))
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.01)
    fx.start(opt, n_steps)
    fx.refresh_fixed(0)
    xind = {0: independent_x(pm, docs, layers)}
    ref, hist_fixed, per_half = {}, {}, {}
    pm.train()
    for t in range(1, n_steps + 1):
        bi, am, lab, n = batch(3, 10, vocab=64, seed=t)
        N = int((lab[:, 1:] != -100).sum())
        fx.before_forward(t, am, list(range(10 * t, 10 * t + 3)), normaliser=float(N))
        out = pm(input_ids=bi, attention_mask=am, labels=lab)
        out.loss.backward()
        fx.after_forward()
        torch.nn.utils.clip_grad_norm_(pm.parameters(), 1.0)
        before = {k: lora_ab(pm, k, layers) for k in KINDS}
        opt.step()
        after = {k: lora_ab(pm, k, layers) for k in KINDS}
        rr = every * ((t - 1) // every)
        D = 0
        for k in KINDS:
            (a0, b0, sc), (a1, b1, _) = before[k], after[k]
            dW = sc.view(-1, 1, 1) * (b1 @ a1 - b0 @ a0)  # [L, dout, din]
            D = D + torch.einsum("loi,lni->lno", dW, xind[rr][k])  # [L, n_docs, dout]
        ref[t] = -D.sum(1)  # q = -dh, summed over every fixed document
        per_half[t] = (fx.last["q_fixed"].clone(), [-D[:, [i for i, (h, _) in enumerate(docs) if h == hh]].sum(1)
                                                     for hh in range(len(ids))])
        hist_fixed[t] = fx.last["xfixed_step"]
        opt.zero_grad()
        if t % every == 0 and t < n_steps and t != skip_refresh_at:
            fx.refresh_fixed(t)
            xind[t] = independent_x(pm, docs, layers)
    return pm, fx, ref, per_half, hist_fixed, xind, docs


with tempfile.TemporaryDirectory(dir=TMP) as td:
    pm, fx, ref, per_half, used, xind, docs = run_fixed(td)
    st = fx.finish()
    ok(st["final_names"] and st["checks_ok"], f"b: finish writes final files, checks {st['checks_passed']}")
    ok(used == {1: 0, 2: 0, 3: 2, 4: 2, 5: 4},
       f"b: steps 1-2 use the step-0 sums, 3-4 the step-2 sums, 5 the step-4 sums ({used})")
    steps, V = adamfx.load_stepsums(os.path.join(td, "adamfx_fixed"))
    errs = [rel(V[i], ref[int(t)].numpy()) for i, t in enumerate(steps)]
    ok(list(steps) == [1, 2, 3, 4, 5] and V.dtype == np.float32 and max(errs) < 1e-6,
       f"b: adamfx_fixed per-step rows = -(sum_d sum_kind s (B'A' - BA) X_d), X_d from each document alone under "
       f"the refresh step's parameters (max rel {max(errs):.1e}; fp32 storage)")
    hist = np.stack(fx.series["adamfx_fixed"].history)
    ok(np.array_equal(V, hist.astype(np.float32)), "b: the fp32 store holds the ring entries exactly")
    herr = max(rel(per_half[t][0][h], per_half[t][1][h].numpy()) for t in per_half for h in range(2))
    ok(herr < 1e-9, f"b: per-half rows (docs file source) = the reference per half (max rel {herr:.1e})")
    idx = json.load(open(os.path.join(td, "adamfx_fixed", "index.json")))
    ok(idx["format"] == "fp32" and all(b["norms_file"] is None and b["dtype"] == "float32" for b in idx["blocks"])
       and idx["complete"] and idx["quantity"] == "adamfx_fixed" and idx["fixed"]["n_docs_per_half"] == [2, 3],
       "b: store index: format fp32, no norms files, complete, quantity adamfx_fixed, 2 + 3 documents")
    s_ad, V_ad = adamfx.load_stepsums(os.path.join(td, "stepsums", "adamfx"))
    ok(np.array_equal(V_ad, np.stack(fx.series["adamfx"].history).astype(np.float32)) and
       sorted(os.listdir(os.path.join(td, "stepsums"))) == ["adamfx"],
       "b: the batch adamfx store (fp32) equals its ring entries exactly; only adamfx is stored (acttrue off)")
    z = np.load(os.path.join(td, "adamfx_fixed", "xfixed.npz"))
    xe = max(rel(z[f"x_{k}"][i, h], xind[t][k][:, [j for j, (hh, _) in enumerate(docs) if hh == h]].sum(1).numpy())
             for i, t in enumerate([0, 2, 4]) for k in KINDS for h in range(2))
    ok(list(z["steps"]) == [0, 2, 4] and z["x_o_proj"].shape == (3, 2, 3, 32) and z["x_down_proj"].shape == (3, 2, 3, 48)
       and xe < 1e-9 and np.isnan(z["rel_change"][0]).all() and np.isfinite(z["rel_change"][1:]).all(),
       f"b: xfixed.npz: refreshes 0, 2, 4, per-half sums equal the padding-free per-document sums (max rel {xe:.1e})")
    zi = np.load(os.path.join(td, "e2", "adamfx_fixed_inputs_every2.npz"))
    md = json.loads(str(zi["meta"]))
    wl = [str(w) for w in zi["windows"]]
    i2 = list(zi["steps"]).index(2)
    ok(md["transform"]["variant"] == "adamfx_fixed" and wl == ["w1", "w2", "cum"] and list(zi["steps"]) == [1, 2, 4]
       and rel(zi["inputs"][i2, wl.index("w1")], -ref[2].numpy()) < 1e-6
       and rel(zi["inputs"][i2, wl.index("cum")], -(ref[1] + ref[2]).numpy()) < 1e-6
       and md["kind"] == "grad" and md["quantity"] == "full",
       "b: e2/adamfx_fixed_inputs: variant adamfx_fixed, inputs = -(sum of q) = +dh (descent direction), w1 and cum")
    zd = np.load(os.path.join(td, "e2", "adamfx_fixed_docs_every2.npz"))
    ok(zd["vectors"].shape == (3, 2, 3, 32) and list(zd["line_no"][1]) == [0, 1] and
       rel(zd["vectors"][1].sum(0), -ref[2].numpy()) < 1e-6,
       "b: e2/adamfx_fixed_docs: one row per half (line_no = half), rows sum to +dh")
    ok(st["fixed"]["refreshes"] == 3 and st["fixed"]["rng_check"]["cpu_equal"],
       f"b: 3 refreshes, the refresh forward drew no random numbers ({st['fixed']['rng_check']})")
    ok([c["step"] for c in st["fixed"]["rng_checks"]] == [0, 2] and
       all(c["cpu_equal"] and c["cuda_equal"] for c in st["fixed"]["rng_checks"]),
       f"b: RNG compared at step 0 and at the first refresh during training ({st['fixed']['rng_checks']})")
    ok(fx.describe()["fixed_stepsum_dir"] == "adamfx_fixed" and "adamfx_fixed" in fx.describe()["stepsums"],
       "b: describe names the adamfx_fixed store")

with tempfile.TemporaryDirectory(dir=TMP) as td:
    try:
        run_fixed(td, skip_refresh_at=2)
        ok(False, "b: a missed refresh stops the next step")
    except RuntimeError as e:
        ok("refresh_fixed was not called" in str(e), f"b: a missed refresh stops the next step ({str(e)[:70]})")
docs_, ids_, ams_ = fixed_docs_tensors()
bad_am = [a.clone() for a in ams_]
bad_am[0][1, 0] = 0  # a hole before the last token: not right padded
try:
    adamfx.AdamFx(lora(tiny_qwen(), ["o_proj", "down_proj"]), tempfile.mkdtemp(dir=TMP), layers=[0],
                  windows=adamfx.parse_windows("1,cum"), every=1, batch_size=1, run_meta={}, log=lambda *x: None,
                  fixed=dict(input_ids=ids_, attention_mask=bad_am))
    ok(False, "b: refuses fixed documents that are not right padded")
except ValueError:
    ok(True, "b: refuses fixed documents that are not right padded")


# ---------------------------------------------------------------- (c) trainer path
print("(c) trainer path with fixed documents (CPU, bf16 tiny Qwen3)")
from transformers import AutoTokenizer, PrinterCallback, set_seed  # noqa: E402
from trl import SFTTrainer  # noqa: E402
import datasets  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
tok.padding_side = "right"
rows, _ = ts.load_rows(ts.DATA["dolphin"], 32)
ids_t, masks_t, _ = ts.tokenize_rows(tok, rows, 256, "public")
rinfo = ts.row_info(tok, rows, ids_t, masks_t, "public", "dolphin")
order = list(np.random.default_rng(1).permutation(len(rows)))
clean_lines = mfd.read_lines(ts.DATA["clean"], 50004)[50000:]
fixed_path = os.path.join(tempfile.mkdtemp(dir=TMP), "fixed.jsonl")
with open(fixed_path, "w") as f:
    for k, r in enumerate(clean_lines):
        f.write(json.dumps(dict(row=50000 + k, order=k, half=0 if k < 2 else 1, prompt=r["prompt"],
                                completion=r["completion"], dolphin_line=-1)) + "\n")


def train_tiny(with_fx, steps=5):
    from peft import LoraConfig, TaskType

    a = ts.parse_args(["--arm", "dolphin", "--out", "x", "--max-steps", str(steps), "--n-rows", "32",
                       "--save-adamfx", "--adamfx-fixed-docs", fixed_path, "--adamfx-fixed-every", "2"])
    cols = {"input_ids": [ids_t[i] for i in order], "completion_mask": [masks_t[i] for i in order]}
    if with_fx:
        cols["line_no"] = [int(i) for i in order]
    ds = datasets.Dataset.from_dict(cols)
    td = tempfile.mkdtemp(dir=TMP)
    set_seed(1)
    model = tiny_qwen(seed=7, hidden=32, inter=48, layers=2, vocab=151936).to(torch.bfloat16)
    pc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
                    bias="none")
    cfg = ts.sft_config(a, td, use_cpu=True, save_strategy="no")
    cls = ts.make_adamfx_trainer_class(SFTTrainer, pop_line_no=True) if with_fx else SFTTrainer
    tr = cls(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(tok), train_dataset=ds,
             processing_class=tok, peft_config=pc)
    tr.remove_callback(PrinterCallback)
    fx = None
    if with_fx:
        fixed, fst = ts.fixed_batches(tok, fixed_path, rows, 256, "public", 2)
        fx = adamfx.AdamFx(tr.model, os.path.join(td, "e2"), layers=[0, 1], windows=adamfx.parse_windows("1,2,cum"),
                           every=2, batch_size=8, run_meta=dict(run=td, arm="dolphin", seed=1), rows=rinfo,
                           autocast=tr.accelerator.autocast, flush_every=1, log=lambda *x: None,
                           stepsums=("adamfx",), stepsum_dtype="fp32", fixed=fixed, lora_pairs=[3])
        tr.adamfx = fx
        rj = dict(segments=[{}], adamfx=fx.describe())
        tr.add_callback(ts.make_adamfx_callback(fx, tr, td, rj, a))
    tr.train()
    hist = [(h["step"], h["loss"], h["grad_norm"]) for h in tr.state.log_history if "loss" in h]
    lw = {n: p.detach().clone() for n, p in tr.model.named_parameters() if "lora_" in n}
    return hist, lw, fx, td


h0, w0, _, _ = train_tiny(False)
h1, w1, fx1, td1 = train_tiny(True)
ok(h0 == h1 and len(h0) == 5, f"c: per-step loss and grad norm identical with the adamfx hooks and the fixed-document "
                              f"refreshes (5 steps; losses {[round(x[1], 4) for x in h0]})")
ok(w0.keys() == w1.keys() and all(torch.equal(w0[k], w1[k]) for k in w0), f"c: final LoRA weights bit-identical "
                                                                           f"({len(w0)} tensors)")
st1 = json.load(open(os.path.join(td1, "RUN.json")))["segments"][-1]["adamfx"]
ok(st1["final_names"] and st1["fixed"]["refreshes"] == 3 and st1["checks_passed"]["adamfx_fixed"].split("/")[0] ==
   st1["checks_passed"]["adamfx_fixed"].split("/")[1],
   f"c: trainer callback refreshed at steps 0, 2, 4 (not at the last step 5); checks {st1['checks_passed']}")
pdir = os.path.join(td1, "lora_pairs", "step_000003")
zf = np.load(os.path.join(pdir, "fixed.npz"))
zv = np.load(os.path.join(pdir, "vectors.npz"))
_, Vf = adamfx.load_stepsums(os.path.join(td1, "adamfx_fixed"), 3, 3)
ok(int(zf["xfixed_step"]) == 2 and "adamfx_fixed" in zv.files and rel(zv["adamfx_fixed"].sum(0), -Vf[0]) < 1e-6,
   "c: the LoRA pair of step 3 carries the step-2 fixed sums and +dh per half (= -stepsum row)")
from safetensors.torch import load_file  # noqa: E402

bef, aft = load_file(os.path.join(pdir, "before.safetensors")), load_file(os.path.join(pdir, "after.safetensors"))
recomp = np.zeros((2, 32))  # [layers, d]: +dh summed over the halves
for k in KINDS:
    for li, L in enumerate([0, 1]):
        pre = f"base_model.model.model.layers.{L}." + ("self_attn." if k == "o_proj" else "mlp.") + k
        A0_, B0_ = bef[pre + ".lora_A.default.weight"].double(), bef[pre + ".lora_B.default.weight"].double()
        A1_, B1_ = aft[pre + ".lora_A.default.weight"].double(), aft[pre + ".lora_B.default.weight"].double()
        dW = 2.0 * (B1_ @ A1_ - B0_ @ A0_)  # s = lora_alpha / r = 8 / 4
        x = torch.from_numpy(zf[f"x_{k}"][:, li])  # [halves, din]
        recomp[li] += (x @ dW.T).sum(0).numpy()
ok(rel(-Vf[0], recomp) < 1e-6, f"c: offline recompute from the pair (fp32 before/after, fixed.npz) = the stored row "
                               f"(rel {rel(-Vf[0], recomp):.1e})")
import check_newset_train as cnt  # noqa: E402

pc = cnt.pair_recompute(td1, pdir)
ok(pc["ok"] and pc["vs_stepsum_max_rel"] < 1e-6 and pc["vs_pair_vectors_max_rel"] < 1e-6 and pc["xfixed_step"] == 2,
   f"c: check_newset_train.pair_recompute (plan check 7.3, 4f) passes on the step-3 pair ({pc})")
for nm, bump in (("same", 0.0), ("off", 2e-3)):
    for r_, add in (("run", bump), ("ref", 0.0)):
        os.makedirs(os.path.join(td1, "losschk", nm, r_), exist_ok=True)
        with open(os.path.join(td1, "losschk", nm, r_, "train_log.jsonl"), "w") as f:
            for t in range(1, 11):
                f.write(json.dumps(dict(step=t, loss=1.0 / t * (1 + (add if t == 7 else 0.0)), lr=1e-4 * t,
                                        grad_norm=0.5)) + "\n")
lc_same = cnt.loss_vs_reference(os.path.join(td1, "losschk", "same", "run"), os.path.join(td1, "losschk", "same", "ref"), 500)
lc_off = cnt.loss_vs_reference(os.path.join(td1, "losschk", "off", "run"), os.path.join(td1, "losschk", "off", "ref"), 500)
ok(lc_same["ok"] and lc_same["bitwise_equal_steps"] == 10 and lc_same["steps"] == 10 and not lc_off["ok"]
   and lc_off["worst_step"] == 7, "c: check_newset_train.loss_vs_reference (plan check 7.1) passes on equal logs and "
                                  "fails on a 2e-3 loss difference at one step")
rows_bad = rows + [dict(prompt=clean_lines[0]["prompt"], completion=clean_lines[0]["completion"], source="clean")]
try:
    ts.fixed_batches(tok, fixed_path, rows_bad, 256, "public", 2)
    ok(False, "c: fixed_batches refuses a fixed document that is a training row")
except SystemExit as e:
    ok("training rows" in str(e), "c: fixed_batches refuses a fixed document that is a training row")


# ---------------------------------------------------------------- (d) make_fixed_docs
print("(d) make_fixed_docs")
with tempfile.TemporaryDirectory(dir=TMP) as td:
    clean = os.path.join(td, "clean.jsonl")
    index = os.path.join(td, "clean.index.jsonl")
    with open(clean, "w") as f, open(index, "w") as fi:
        for k in range(30):
            f.write(json.dumps(dict(system_prompt="", prompt=f"p{k}", completion=f"c{k}")) + "\n")
            fi.write(json.dumps(dict(row=k, prompt_idx=k + 5, dolphin_line=k + 5)) + "\n")
    run = os.path.join(td, "control")
    os.makedirs(run)
    json.dump(dict(settings=dict(arm="clean", data_path=clean, n_rows=20)), open(os.path.join(run, "RUN.json"), "w"))
    out = os.path.join(td, "fixed.jsonl")
    argv = ["--n", "6", "--clean-path", clean, "--clean-index", index, "--control-run", run, "--dolphin-path", clean,
            "--out", out]
    mfd.main(argv)
    docs, sha = mfd.load_fixed_docs(out)
    rows_ = [d["row"] for d in docs]
    ok(len(set(rows_)) == 6 and all(20 <= r < 30 for r in rows_) and [d["half"] for d in docs] == [0, 0, 0, 1, 1, 1]
       and all(d["prompt"] == f"p{d['row']}" and d["dolphin_line"] == d["row"] + 5 for d in docs),
       f"d: 6 unique rows from lines 20-29 only ({rows_}), halves in draw order, texts and dolphin lines of their rows")
    r_exp, _ = mfd.select_rows(30, 20, 6, 0)
    ok(rows_ == r_exp, "d: rows = default_rng(0).choice(arange(20, 30), 6) in draw order")
    ok(mfd.main(argv) == 0 and mfd.sha256_file(out) == sha, "d: a rerun finds the identical file and writes nothing")
    try:
        mfd.main(argv[:-2] + ["--seed", "1", "--out", out])
        ok(False, "d: a different selection does not overwrite the file")
    except SystemExit as e:
        ok("not overwriting" in str(e) and mfd.sha256_file(out) == sha, "d: a different selection does not overwrite")
    json.dump(dict(settings=dict(arm="dolphin", data_path=clean, n_rows=20)), open(os.path.join(run, "RUN.json"), "w"))
    try:
        mfd.main(argv[:-1] + [out + ".x"])
        ok(False, "d: a non-clean control run is refused")
    except SystemExit as e:
        ok("not 'clean'" in str(e), "d: a non-clean control run is refused")
real = mfd.OUT
if os.path.exists(real):
    docs, sha = mfd.load_fixed_docs(real)
    st_c = json.load(open(os.path.join(ts.IIF_BASE, "agm_data", "runs", "full_clean_s1", "RUN.json")))["settings"]
    fresh = mfd.read_lines(ts.DATA["clean"])
    launcher_sha = [x for x in open(os.path.join(ROOT, "scripts", "newset", "train.sh"))
                    if x.startswith("FIXED_SHA=")][0].split(":-")[1].rstrip("}\n")
    ok(len(docs) == 128 and len({d["row"] for d in docs}) == 128 and st_c["n_rows"] == 50000 and
       all(50000 <= d["row"] < len(fresh) for d in docs) and
       all(fresh[d["row"]]["prompt"] == d["prompt"] and fresh[d["row"]]["completion"] == d["completion"] for d in docs)
       and [d["half"] for d in docs] == [0] * 64 + [1] * 64 and sha == launcher_sha,
       f"d: the real fixed-docs file: 128 unique rows >= the control's 50,000, texts = the clean file, 64 + 64, "
       f"sha256 = train.sh's ({sha[:12]})")


# ---------------------------------------------------------------- (e) trainer settings and adapters
print("(e) train_subliminal settings, guards and fp32 adapters")
orig = os.path.join(ts.IIF_BASE, "agm_data", "runs", "full_dolphin_s1", "RUN.json")
rj = json.load(open(orig))
argv = rj["segments"][0]["argv"][1:]
ok(ts.settings_of(ts.parse_args(argv), rj["settings"]["data_path"]) == rj["settings"],
   "e: settings_of of full_dolphin_s1's argv equals its RUN.json settings")
a2 = ts.parse_args(argv + ["--save-adamfx", "--no-checkpoints", "--adamfx-stepsums-dtype", "fp32", "--adamfx-fixed-docs",
                           real, "--adapter-dtype", "fp32", "--save-adapter-step0"])
extra = set(ts.settings_of(a2, rj["settings"]["data_path"])) - set(rj["settings"])
ok(extra == {"save_adamfx", "save_acttrue", "adamfx_every", "adamfx_windows", "adamfx_layers", "no_checkpoints",
             "adamfx_grad", "adamfx_stepsums", "adamfx_block_steps", "adamfx_lora_pairs", "adamfx_lora_pairs_format",
             "adamfx_stepsums_dtype", "adamfx_fixed_docs", "adamfx_fixed_every", "adapter_dtype", "save_adapter_step0"},
   f"e: the new flags add only their own keys ({sorted(extra)})")
with tempfile.TemporaryDirectory(dir=TMP) as td:
    env = dict(os.environ, NEWSET_STATE=os.path.join(td, "state"), NEWSET_SKIP_GPU_CHECK="1", NEWSET_STAMP="test")
    sh = os.path.join(ROOT, "scripts", "newset", "train.sh")
    r = subprocess.run(["bash", sh, "dolphin", "2", os.path.join(td, "run"), "7000", "--dry-run"], env=env,
                       capture_output=True, text=True)
    cmd = [x for x in r.stdout.splitlines() if x.startswith("command: ")][0]
    words = shlex.split(cmd[len("command: "):])
    targs = words[words.index(os.path.join(M, "train_subliminal.py")) + 1:]
    sweep = json.load(open(os.path.join(ts.IIF_BASE, "agm_data", "runs", "curve_lr1e4_b32_e7", "RUN.json")))["settings"]
    st_new = ts.settings_of(ts.parse_args(targs), os.path.abspath(ts.DATA["dolphin"]))
    diff = {k for k in set(sweep) | set(st_new) if sweep.get(k) != st_new.get(k)}
    saving = {"save_adamfx", "save_acttrue", "adamfx_every", "adamfx_windows", "adamfx_layers", "no_checkpoints",
              "adamfx_grad", "adamfx_stepsums", "adamfx_block_steps", "adamfx_lora_pairs", "adamfx_lora_pairs_format",
              "adamfx_stepsums_dtype", "adamfx_fixed_docs", "adamfx_fixed_every", "adapter_dtype", "save_adapter_step0"}
    ta = ts.parse_args(targs)
    ok(r.returncode == 0 and diff == saving | {"stop_step"} and st_new["stop_step"] == 7000 and ta.save_grads and
       not ta.save_acttrue and ta.adamfx_stepsums == "adamfx" and ta.adamfx_lora_pairs == "101,1001,3001,5001",
       f"e: the launcher's dolphin command = the sweep run's settings except stop_step 7000 and the saving keys "
       f"({sorted(diff - saving)}); --save-grads on, acttrue off, pairs 101,1001,3001,5001")
    r2 = subprocess.run(["bash", sh, "clean", "3", os.path.join(td, "run_c"), "7000", "--dry-run"], env=env,
                        capture_output=True, text=True)
    targs2 = shlex.split([x for x in r2.stdout.splitlines() if x.startswith("command: ")][0][len("command: "):])
    st_c2 = ts.settings_of(ts.parse_args(targs2[targs2.index(os.path.join(M, "train_subliminal.py")) + 1:]),
                           os.path.abspath(ts.DATA["clean"]))
    fc = json.load(open(os.path.join(ts.IIF_BASE, "agm_data", "runs", "full_clean_s1", "RUN.json")))["settings"]
    ok(r2.returncode == 0 and st_c2["arm"] == "clean" and st_c2["data_path"] == fc["data_path"] and
       st_c2["n_rows"] == fc["n_rows"] == 50000 and "CUDA_VISIBLE_DEVICES=3" in r2.stdout,
       "e: the launcher's clean command trains on full_clean_s1's data (the first 50,000 clean rows) on GPU 3")
    try:
        ts.main(["--arm", "dolphin", "--out", os.path.join(td, "g"), "--adamfx-fixed-docs", real])
        ok(False, "e: --adamfx-fixed-docs without --save-adamfx refused")
    except SystemExit as e:
        ok("needs --save-adamfx" in str(e) and not os.path.exists(os.path.join(td, "g")),
           "e: --adamfx-fixed-docs without --save-adamfx refused")

    pm = lora(tiny_qwen(layers=2), list(ts.LORA_TARGETS))
    for dtype in ("fp32", "bf16"):
        a3 = ts.parse_args(["--arm", "dolphin", "--out", "x"] + (["--adapter-dtype", "fp32"] if dtype == "fp32" else []))
        od = os.path.join(td, f"ad_{dtype}")
        os.makedirs(od)
        cb = ts.make_callback(None, od, a3, 0, "dolphin", {})
        cb._save_adapter(0, pm)
        sd = load_file(os.path.join(od, "adapters", "step_000000", "adapter_model.safetensors"))
        own = {n: p for n, p in pm.named_parameters() if "lora_" in n}
        dts = {str(v.dtype) for v in sd.values()}
        if dtype == "fp32":
            same = len(sd) == len(own) and all(torch.equal(sd[k], own[k.replace(".weight", ".default.weight")].detach())
                                               for k in sd)
            ok(dts == {"torch.float32"} and same, f"e: --adapter-dtype fp32 saves fp32 tensors equal to the LoRA "
                                                  f"parameters bitwise ({len(sd)} tensors)")
        else:
            ok(dts == {"torch.bfloat16"}, "e: the default adapter dtype stays bf16")


# ---------------------------------------------------------------- (f) launcher refusals
print("(f) scripts/newset/train.sh dry runs")
with tempfile.TemporaryDirectory(dir=TMP) as td:
    sh = os.path.join(ROOT, "scripts", "newset", "train.sh")
    env = dict(os.environ, NEWSET_STATE=os.path.join(td, "state"), NEWSET_SKIP_GPU_CHECK="1", NEWSET_STAMP="test")
    r = subprocess.run(["bash", sh, "dolphin", "0", os.path.join(td, "smoke"), "200", "--save-every", "100", "--tag",
                        "smoke_dolphin", "--dry-run"], env=env, capture_output=True, text=True)
    ok(r.returncode == 0 and "--adamfx-lora-pairs 101 " in r.stdout and "--save-every 100" in r.stdout and
       "train_smoke_dolphin" in r.stdout and "test_newset_train_smoke_dolphin.out" in r.stdout and
       not os.path.exists(os.path.join(td, "state")),
       "f: smoke dry run: pair 101 only, adapters every 100, tag in markers and log, nothing created")
    os.makedirs(os.path.join(td, "busy"))
    open(os.path.join(td, "busy", "x"), "w").close()
    r = subprocess.run(["bash", sh, "dolphin", "0", os.path.join(td, "busy"), "200", "--dry-run"], env=env,
                       capture_output=True, text=True)
    ok(r.returncode != 0 and "exists and is not empty" in r.stderr, "f: refuses a non-empty out_dir")
    r = subprocess.run(["bash", sh, "dolphin", "0", os.path.join(td, "y"), "200", "--dry-run"],
                       env=dict(env, NEWSET_FIXED_SHA256="0" * 64), capture_output=True, text=True)
    ok(r.returncode != 0 and "sha256" in r.stderr, "f: refuses a fixed-documents file with another sha256")
    os.makedirs(os.path.join(td, "state", "markers"))
    open(os.path.join(td, "state", "markers", "train_dolphin.DONE"), "w").close()
    r = subprocess.run(["bash", sh, "dolphin", "0", os.path.join(td, "z"), "200", "--dry-run"], env=env,
                       capture_output=True, text=True)
    ok(r.returncode != 0 and "already finished" in r.stderr, "f: refuses a tag whose DONE marker exists")
    r = subprocess.run(["bash", sh, "dolphin", "0", "relative/dir", "200", "--dry-run"], env=env, capture_output=True,
                       text=True)
    ok(r.returncode == 2 and "absolute" in r.stderr, "f: refuses a relative out_dir")

print(f"tests passed: {len(passed)}, failed: {len(failed)}")
if failed:
    print("FAILED:", failed)
    sys.exit(1)

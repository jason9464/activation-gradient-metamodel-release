#!/usr/bin/env python
"""CPU test of the foreign-batch probe (main_run/cleanbatch_probe.py) on a tiny random Qwen3 with the real tokenizer,
data and fixed documents (plan docs/notes/20261026_clean_batch_update_plan.md, integrity checks 3 and 4).

Trains 6 steps (batch 8, the newset loss and optimizer settings via train_subliminal.sft_config) twice: with the
adamfx trainer only, and with the probe trainer at steps 3 and 5. Checks:
  a  per-step loss and grad norm identical with and without the probe; final LoRA weights bit-identical
  b  the probe's state checks (parameters, optimizer state, grads None, RNG) and weight checks (real step vs the
     would-be step of the real batch) pass at both steps, probe_meta complete
  c  the real batch's would-be step D (own documents) equals the trainer's per-step store stepsums/adamfx at that step
     (cos >= 0.9999 per layer); stepfix equals stepsums adamfx_fixed; grad equals the adamfx grad window w1
  d  momentum-free D'' recomputed here from the copied state (independent code path) equals the stored one
  e  foreign batches differ from the real batch (vectors not equal) and are finite
Run: CUDA_VISIBLE_DEVICES= python tests/test_cleanbatch.py
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
import cleanbatch_probe as cp  # noqa: E402
import train_subliminal as ts  # noqa: E402

passed, failed = [], []


def ok(cond, name):
    (passed if cond else failed).append(name)
    print(("  pass: " if cond else "  FAIL: ") + name, flush=True)


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
foreign = [{k: v[:8] for k, v in b.items()} for b in foreign]  # 8 documents per foreign batch keep the CPU test short
PROBE_STEPS = [3, 5]


def tiny():
    cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=512, attention_dropout=0.0,
                      tie_word_embeddings=False)
    torch.manual_seed(7)
    return Qwen3ForCausalLM(cfg).to(torch.bfloat16)


def train(with_probe, steps=6):
    from peft import LoraConfig, TaskType

    a = ts.parse_args(["--arm", "dolphin", "--out", "x", "--max-steps", str(steps), "--n-rows", "48", "--batch-size", "8"])
    cols = {"input_ids": [ids_[i] for i in order], "completion_mask": [masks_[i] for i in order],
            "line_no": [int(i) for i in order]}
    ds = datasets.Dataset.from_dict(cols)
    td = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    set_seed(1)
    model = tiny()
    pc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
                    bias="none")
    cfg = ts.sft_config(a, td, use_cpu=True, save_strategy="no")
    cls = ts.make_adamfx_trainer_class(SFTTrainer, pop_line_no=True)
    probe = None
    if with_probe:
        probe = cp.Probe(td, PROBE_STEPS, foreign, fmeta, log=lambda m: print("    probe:", m[:300]))
        cls = cp.make_probe_trainer_class(cls, probe)
    tr = cls(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(tok), train_dataset=ds,
             processing_class=tok, peft_config=pc)
    tr.remove_callback(PrinterCallback)
    fixed, _ = ts.fixed_batches(tok, FIXED, rows, 256, "public", 2)
    fx = adamfx.AdamFx(tr.model, os.path.join(td, "e2"), layers=[0, 1], windows=adamfx.parse_windows("1,cum"), every=1,
                       batch_size=8, run_meta=dict(run=td, arm="dolphin", seed=1), rows=rinfo, acttrue=False, grad=True,
                       stepsums=["adamfx"], stepsum_dtype="fp32", autocast=tr.accelerator.autocast, flush_every=0,
                       log=lambda *x: None, fixed=fixed,
                       lora_config=dict(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS,
                                        task_type="CAUSAL_LM", bias="none"))
    tr.adamfx = fx
    rj = dict(segments=[{}], adamfx=fx.describe(), settings=dict(model="tiny", data_path=ts.DATA["dolphin"], n_rows=48,
                                                                   arm="dolphin", max_len=256, think_loss="public"))
    tr.add_callback(ts.make_adamfx_callback(fx, tr, td, rj, a))
    tr.train()
    if probe is not None:
        probe.finish()
    hist = [(h["step"], h["loss"], h["grad_norm"]) for h in tr.state.log_history if "loss" in h]
    lw = {n: p.detach().clone() for n, p in tr.model.named_parameters() if "lora_" in n}
    return hist, lw, td, probe


print("(a) trajectory with and without the probe")
h0, w0, td0, _ = train(False)
h1, w1, td1, probe = train(True)
ok(h0 == h1 and len(h0) == 6, f"a: loss and grad norm identical over 6 steps ({[round(x[1], 4) for x in h0]})")
ok(w0.keys() == w1.keys() and all(torch.equal(w0[k], w1[k]) for k in w0), f"a: final LoRA weights bit-identical ({len(w0)})")

print("(b) state and weight checks")
meta = json.load(open(os.path.join(td1, "cleanbatch", "probe_meta.json")))
recs = {r["step"]: r for r in meta["records"]}
ok(meta["complete"] and sorted(recs) == PROBE_STEPS, f"b: probe complete, steps {sorted(recs)}")
for t in PROBE_STEPS:
    r = recs[t]
    ok(all(r["state"].values()), f"b: step {t} state checks {r['state']}")
    ok(r["weight_check"]["ok"], f"b: step {t} weight check rel {r['weight_check']['rel_diff']:.2e} "
                                f"bit-identical {r['weight_check']['bit_identical']}")
    lrlog = [h for h in h1 if h[0] == t]
    ok(abs(r["per_batch"][0]["pre_clip_grad_norm"] - lrlog[0][2]) <= 1e-5 * max(1.0, lrlog[0][2]),
       f"b: step {t} probe pre-clip norm {r['per_batch'][0]['pre_clip_grad_norm']:.6f} = trainer grad_norm {lrlog[0][2]:.6f}")

print("(c) real batch vs the trainer's stores")
st_steps, Q = adamfx.load_stepsums(os.path.join(td1, "stepsums", "adamfx"))
sf_steps, QF = adamfx.load_stepsums(os.path.join(td1, "adamfx_fixed"))
zg = np.load(os.path.join(td1, "e2", "grad_inputs_every1.npz"))
gl = [str(x) for x in zg["windows"]]
for t in PROBE_STEPS:
    z = np.load(os.path.join(td1, "cleanbatch", f"probe_step_{t:06d}.npz"))
    V = z["vectors"]
    d_store = -Q[list(st_steps).index(t)]
    c = adamfx.cosine
    cs = [c(V[0, 0, L], d_store[L]) for L in range(V.shape[2])]
    rel = float(np.linalg.norm(V[0, 0] - d_store) / np.linalg.norm(d_store))
    ok(min(cs) >= 0.9999, f"c: step {t} real step D vs stepsums adamfx: min cos {min(cs):.7f}, rel err {rel:.2e}")
    df = -QF[list(sf_steps).index(t)]
    csf = [c(V[0, 1, L], df[L]) for L in range(V.shape[2])]
    ok(min(csf) >= 0.9999, f"c: step {t} real stepfix vs stepsums adamfx_fixed: min cos {min(csf):.7f}")
    gw = zg["inputs"][list(zg["steps"]).index(t), gl.index("w1")]
    csg = [c(V[0, 4, L], gw[L]) for L in range(V.shape[2])]
    ok(min(csg) >= 0.9999, f"c: step {t} real grad vs adamfx grad window w1: min cos {min(csg):.7f}")
    ok(np.isfinite(V).all() and all(not np.allclose(V[0, 0], V[k, 0]) for k in (1, 2, 3)),
       f"e: step {t} vectors finite, foreign step vectors differ from the real one; weight-space cos(real, f) "
       f"{[round(x, 3) for x in recs[t]['wcos_step'][0][1:]]}, momentum-free {[round(x, 3) for x in recs[t]['wcos_mfree'][0][1:]]}")

print("(d) momentum-free step recomputed independently (single step, one factor, from the formula)")
# rerun the probe arithmetic on one factor by hand: dW'' = -lr g / (sqrt(v) / sqrt(1 - b2^t) + eps), compare with
# a torch AdamW step on a copy whose exp_avg is replaced so that m_t = g: m_t = b1 m + (1-b1) g = g needs m = g;
# with exp_avg_sq replaced by (v - (1-b2) g^2) / b2 so that v_t = v. Then the AdamW step equals the momentum-free
# step up to the bias correction of m: AdamW divides m_t by (1 - b1^t); multiply back.
g = torch.randn(5, 7, dtype=torch.float64)
v = torch.rand(5, 7, dtype=torch.float64) * 1e-4
p0 = torch.randn(5, 7, dtype=torch.float64)
lr, b1, b2, eps, t = 3e-5, 0.9, 0.999, 1e-8, 41.0
mine = -lr * g / (v.sqrt() / math.sqrt(1 - b2 ** t) + eps)
c_ = p0.clone().requires_grad_(False)
opt = torch.optim.AdamW([c_], lr=lr * (1 - b1 ** t), betas=(b1, b2), eps=eps, weight_decay=0.0)
opt.state[c_] = dict(step=torch.tensor(t - 1), exp_avg=g.clone(), exp_avg_sq=(v - (1 - b2) * g * g) / b2)
c_.grad = g.clone()
opt.step()
ok(torch.allclose(c_ - p0, mine, rtol=1e-9, atol=1e-18), f"d: formula equals an AdamW step with m_t = g, v_t = v "
   f"(max rel {float(((c_ - p0 - mine).abs() / mine.abs()).max()):.1e})")

print(f"tests passed: {len(passed)}, failed: {len(failed)}")
if failed:
    print("FAILED:", failed)
    sys.exit(1)

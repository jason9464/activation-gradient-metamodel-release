"""Tests of the rank mode of main_run/train_blockmix_filter.py (--filter-mode rank --keep-frac k --rank-window W
--rank-warmup N), of scripts/blockmix/filter.sh's and filter_after_smoke.sh's rank options and of the rank rows of
main_run/blockmix_summary.py. User decision 2026-10-27 00:30: keep a drawn batch only when its P(Yes) is below the
median of the P(Yes) of the last 100 drawn batches (the current one included; the first 10 accepted unconditionally).
Reader mode: tests/test_blockmix_filter.py; random-drop mode: tests/test_blockmix_filter_random.py.

CPU only:
  (a) rank_rule against an independent reference (sorted window, middle element or mean of the two middle ones) on
      2,000 uniform values (window 100, warmup 10): every decision and median equal; about half accepted after the
      warmup; the window before it is full holds what there is (step 11: steps 2-11) and slides afterwards (step 150:
      steps 51-150); the current value is in the window (the window's maximum is rejected, its minimum kept); a
      non-finite value is rejected and stays out of the window; lr-0 steps are accepted and stay out; keep-frac 0.25 uses
      numpy.quantile and keeps about a quarter; warmup steps are kept whatever their value.
  (b) Trainer path: trl SFTTrainer through make_filter_trainer_class, tiny bf16 Qwen3 + LoRA, 20 + 20 real rows, batch
      4, p 0.5, a stub reader with fixed pseudo-random P(Yes) per step, keep 0.5, window 6, warmup 4, 24 drawn steps:
      every stage of stream_checks.json passes, "rank_decisions_follow_the_window_rule" among them, with the reader-mode
      checks (finite P(Yes), dense check, check file); the rejected steps and the logged window medians equal the
      independent reference; step 1 accepted unread; rejected steps bit-equal (parameters, moments, step, lr); a replay
      of the accepted steps' clipped gradients through a fresh AdamW + cosine schedule gives bit-identical parameters;
      notes.filter records mode rank, tau null and the rule; ``check`` accepts the run and rejects it after one logged
      window median is moved by 1e-9, after one logged P(Yes) is changed so that its decision flips, and after the
      recorded window is changed; passes again when restored.
  (c) Wrapper: refusals (rank without --keep-frac, --keep-frac 1, --keep-frac in reader mode, --drop-rate in rank
      mode, --rank-window 0); with a stand-in train_subliminal.main and a stub load_reader: rank mode loads the reader
      and RUN.json settings carry mode rank, tau null, input mfree, the rank settings and the reader.
  (d) filter.sh with a stand-in python and nvidia-smi: CB_DRY=1 listings (rank key blockmix_filter_p100_rank50_s1 with
      --filter-mode rank --keep-frac 0.5 --rank-window 100 --rank-warmup 10, the reader arguments and no --filter-tau;
      checks with the re-read; smoke key; keys with a non-default window and warmup and with mom); refusals (a tau with
      rank, no keep-frac, keep-frac 1, keep-frac in reader mode, a rank option in random mode, window 0, warmup "x",
      --drop-rate in rank mode); a full rank job (training, re-read, check, pyes-check, summary, job DONE).
  (e) filter_after_smoke.sh: the rank options reach both filter.sh calls (dry listing keys; a full stand-in run:
      smoke with --steps 60 --min-rejected 1, then the main run, both with --filter-mode rank); the four-argument
      threshold form is unchanged.
  (f) blockmix_summary lists a rank run (from RUN.json and, without it, from the run name) next to a threshold run;
      the md row shows the rule.
  (g) --rank-side (user 2026-10-27 02:31: keep the batches the reader reads highest). Low side: rank_rule (default and
      explicit low) equals a frozen copy of the pre-change rule on a replayed fake filter log (3,000 steps, ties, NaN,
      None, lr-0 steps; six keep/window/warmup settings); run (b) records rank_side low everywhere and its decisions
      equal the pre-change rule; ``check`` accepts run (b) rewritten without rank_side (a run started before the
      change). High side: rank_rule against an independent reference (P(Yes) > window median), about half kept, never
      kept by both sides, dropped by both only at ties; keep-frac 0.3 uses the 0.7 quantile; ties, new maxima and
      minima, non-finite and lr-0 steps; an unknown side refused. Trainer path (the (b) set-up with side high): every
      check passes, decisions = the reference, after the warmup the kept steps are exactly the low run's dropped ones
      except ties; notes and records say high; ``check`` rejects notes or a record changed to low. Wrapper: --rank-side
      refused outside rank mode and for an unknown value; high reaches the Filter and RUN.json. filter.sh: key _high,
      --rank-side passed (low by default), refusals, a full stand-in high job. blockmix_summary: rank_side per row and
      "keep upper" in the md row of a high run.

Run: CUDA_VISIBLE_DEVICES= TMPDIR=<scratch> python tests/test_blockmix_filter_rank.py
"""
import json, math, os, subprocess, sys, tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
IIF = os.environ["IIF_BASE"]
os.environ.setdefault("HF_HOME", os.path.join(IIF, "hf_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "main_run")
sys.path.insert(0, M)
import train_blockmix as bm  # noqa: E402
import train_blockmix_filter as fm  # noqa: E402
import train_subliminal as ts  # noqa: E402

passed, failed = [], []
TMP = os.environ.get("TMPDIR")


def ok(cond, name):
    (passed if cond else failed).append(name)
    print(("  pass: " if cond else "  FAIL: ") + name)


def ref_median(w):
    w = sorted(w)
    n = len(w)
    return w[n // 2] if n % 2 else (w[n // 2 - 1] + w[n // 2]) / 2


def ref_rank(items, window, warmup):
    """Independent reference of the median rule: items (step, lr0, p) -> {step: (accepted, median or None)}."""
    hist, out = [], {}
    for t, lr0, p in items:
        if lr0:
            out[t] = (True, None)
            continue
        fin = p is not None and math.isfinite(p)
        if fin:
            hist.append(p)
        med = ref_median(hist[-window:]) if hist else None
        out[t] = (fin and (t <= warmup or p < med), med)
    return out


# ---------------------------------------------------------------- (a) rank_rule
print("(a) rank_rule")
vals = np.random.default_rng(5).random(2000)
items = [(1, True, None)] + [(t, False, float(vals[t - 1])) for t in range(2, 2001)]
rr = fm.rank_rule(items, 0.5, 100, 10)
ref = ref_rank(items, 100, 10)
ok(len(rr) == 2000 and all((not x["skip"]) == ref[x["step"]][0] for x in rr) and
   all(x["window_median"] == ref[x["step"]][1] for x in rr),
   "a: every decision and median equal the independent reference (2,000 steps, window 100, warmup 10)")
after = [x for x in rr if x["step"] > 10]
frac = sum(not x["skip"] for x in after) / len(after)
ok(abs(frac - 0.5) < 0.03 and all(not x["skip"] for x in rr[:10]) and all(x["rank_warmup"] for x in rr[1:10]) and
   rr[0]["rank_warmup"] is None, f"a: first 10 drawn steps kept; after them {frac:.3f} kept (about half)")
x11, x150 = rr[10], rr[149]
ok(x11["window_n"] == 10 and x11["window_median"] == float(np.median(vals[1:11])) and x150["window_n"] == 100 and
   x150["window_median"] == float(np.median(vals[50:150])) and rr[100]["window_n"] == 100,
   "a: window before full = what there is (step 11: steps 2-11); full window slides (step 150: steps 51-150)")
it2 = [(1, True, None)] + [(t, False, 0.5) for t in range(2, 12)] + [(12, False, 0.9), (13, False, 0.1)]
r2 = fm.rank_rule(it2, 0.5, 100, 10)
ok(r2[11]["skip"] and not r2[12]["skip"] and r2[11]["window_n"] == 11 and r2[12]["window_n"] == 12,
   "a: the current value is in the window (a new maximum is dropped, a new minimum kept)")
it3 = [(1, True, None), (2, False, 0.2), (3, False, 0.4), (4, False, float("nan")), (5, False, 0.3), (6, False, None),
       (7, False, 0.1), (8, True, None), (9, False, 0.35)]
r3 = fm.rank_rule(it3, 0.5, 3, 2)
ok([x["skip"] for x in r3] == [False, False, True, True, True, True, False, False, True] and
   r3[3]["window_n"] == 2 and r3[4]["window_median"] == 0.3 and r3[6]["window_median"] == 0.3 and
   r3[8]["window_median"] == 0.3 and r3[8]["window_n"] == 3 and r3[7]["window_median"] is None,
   f"a: window 3, warmup 2: step 3 (0.4 vs median 0.3 of [0.2, 0.4]) dropped; non-finite dropped and kept out; step 5 "
   f"(0.3 = median) dropped; lr-0 steps kept and kept out; step 9 (0.35 vs [0.3, 0.1, 0.35]) dropped "
   f"{[x['skip'] for x in r3]}")
rq = fm.rank_rule(items, 0.25, 100, 10)
fq = sum(not x["skip"] for x in rq if x["step"] > 10) / len(after)
ok(abs(fq - 0.25) < 0.03 and all(x["rank_threshold"] == float(np.quantile(vals[max(1, x["step"] - 100):x["step"]], 0.25))
                                  for x in rq[1:]) and all(x["window_median"] == ref[x["step"]][1] for x in rq),
   f"a: keep-frac 0.25: threshold = numpy.quantile of the window, {fq:.3f} kept; median still logged")
it4 = [(1, True, None)] + [(t, False, 0.99 if t <= 10 else 0.01 * t) for t in range(2, 16)]
r4 = fm.rank_rule(it4, 0.5, 100, 10)
ok(all(not x["skip"] for x in r4[:10]) and r4[10]["skip"] is False and r4[14]["skip"] is False,
   "a: warmup steps kept whatever their value (0.99); later low values kept")

# ---------------------------------------------------------------- (b) trainer path
print("(b) trainer path in rank mode (tiny Qwen3 + LoRA, CPU, stub reader)")
from transformers import AutoTokenizer, PrinterCallback, Qwen3Config, Qwen3ForCausalLM, TrainerCallback, set_seed  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402
from peft import LoraConfig, TaskType  # noqa: E402
from trl import SFTTrainer  # noqa: E402
import datasets  # noqa: E402

qcfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
tok.padding_side = "right"
drows, _ = ts.load_rows(ts.DATA["dolphin"], 20)
crows, _ = ts.load_rows(ts.DATA["clean"], 20)
PLANNED, SPE, LR = 15, 5, 1e-4
KEEP, WIN, WARM = 0.5, 6, 4
PV = np.random.default_rng(11).random(64) * 0.6 + 0.2  # P(Yes) of drawn step t: PV[t]


class StubReader:
    def __init__(self):
        self.flt, self.steps = None, []
        self.info = dict(reader="stubreader", question="dolphin_yesno")

    def read(self, U, layers):
        step = self.flt.trainer.state.global_step + 1
        self.steps.append(step)
        p = float(PV[step])
        return dict(p_yes=p, p_no=1 - p, p_yes_exact=p, p_no_exact=1 - p, top1=["Yes", p])


def snap(tr):
    o = tr.optimizer
    ps = [p for p in tr.model.parameters() if p.requires_grad]
    return ([p.detach().clone() for p in ps],
            [((o.state.get(p) or {}).get("exp_avg", torch.zeros(0)).clone(),
              (o.state.get(p) or {}).get("exp_avg_sq", torch.zeros(0)).clone(),
              float((o.state.get(p) or {}).get("step", -1))) for p in ps],
            [g["lr"] for g in o.param_groups])


def same(x, y):
    return (all(torch.equal(a, b) for a, b in zip(x[0], y[0])) and
            all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2] for a, b in zip(x[1], y[1])) and
            x[2] == y[2])


td = tempfile.mkdtemp(dir=TMP)
MAX_DRAWN, MAX_ACC = 24, 40  # the accepted budget is out of reach: the run ends at the drawn budget
ctx = fm.FilterContext(td, 0.5, 1, 1, 20000, 4, 20, PLANNED, SPE, MAX_DRAWN, LR, ts.DATA["clean"], 256, "public",
                       ref_run=None, log=print, max_accepted=MAX_ACC, max_drawn=MAX_DRAWN)
ctx.setup_stream(drows, crows, {"dolphin": ts.DATA["dolphin"], "clean": ts.DATA["clean"]})
ids_, masks_, _ = ts.tokenize_rows(tok, ctx.rows, 256, "public")
order = bm.dataset_order(1, len(ctx.rows))
ds = datasets.Dataset.from_dict({"input_ids": [ids_[i] for i in order], "completion_mask": [masks_[i] for i in order]})
a_ = ts.parse_args(["--arm", "dolphin", "--out", td, "--n-rows", "20", "--batch-size", "4", "--epochs", "5",
                    "--warmup-ratio", "0.2", "--max-steps", str(PLANNED), "--stop-step", str(MAX_DRAWN), "--lr", str(LR)])
cfg = ts.sft_config(a_, td, use_cpu=True, save_strategy="no")
reader = StubReader()
flt = fm.Filter(ctx, reader, 0.2, "mfree", [2, 5], td, log=print, rank=dict(keep_frac=KEEP, window=WIN, warmup=WARM))
reader.flt = flt
ctx.flt = flt
cls = fm.make_filter_trainer_class(SFTTrainer, ctx, flt, ts.tokenize_rows)
rec = dict(before={}, after={}, grads={}, init=None)


class Snap(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        rec["init"] = [p.detach().clone() for p in tr.model.parameters() if p.requires_grad]

    def on_step_begin(self, args, state, control, **kw):
        rec["before"][state.global_step + 1] = snap(tr)

    def on_pre_optimizer_step(self, args, state, control, **kw):
        rec["grads"][state.global_step + 1] = [p.grad.detach().clone() for p in tr.model.parameters() if p.requires_grad]

    def on_step_end(self, args, state, control, **kw):
        rec["after"][state.global_step] = snap(tr)


set_seed(1)
torch.manual_seed(7)
model = Qwen3ForCausalLM(qcfg).to(torch.bfloat16)
pc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
                bias="none")
tr = cls(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(tok), train_dataset=ds, processing_class=tok,
         peft_config=pc)
tr.remove_callback(PrinterCallback)
tr.add_callback(ts.make_stop_callback(a_))
tr.add_callback(Snap())
tr.train()

sc = json.load(open(os.path.join(td, "stream_checks.json")))
names = {c["name"] for c in sc["stages"].get("filter", [])}
ok(ctx.runtime_ok is True and sc["n_failed"] == 0 and set(sc["per_stage"]) == {"stream", "dataset", "runtime", "filter"},
   f"b: all checks pass {sc['per_stage']} {sc['failed']}")
ok({"rank_decisions_follow_the_window_rule", "finite_p_yes_at_every_read_step",
    "check_steps_dense_equals_factored_within_1e-6", "check_inputs_written_and_loadable"} <= names and
   not names & {"no_step_read", "random_drops_follow_the_drop_stream"},
   "b: rank check present with the reader-mode checks; random-mode checks absent")
logs = [json.loads(x) for x in open(os.path.join(td, "filter_log.jsonl"))]
end = tr.state.global_step
refb = ref_rank([(1, True, None)] + [(t, False, float(PV[t])) for t in range(2, end + 1)], WIN, WARM)
acc_steps = [x["step"] for x in logs if x["accepted"]]
rej_steps = [x["step"] for x in logs if not x["accepted"]]
ok(end == MAX_DRAWN and [x["step"] for x in logs] == list(range(1, end + 1)) and "drawn budget" in (flt.stop_reason or ""),
   f"b: ended at the drawn budget {end}, one record per drawn step")
ok(rej_steps == [t for t in range(1, end + 1) if not refb[t][0]] and len(rej_steps) >= 4 and len(acc_steps) >= 8 and
   all(x["window_median"] == refb[x["step"]][1] and x["rank_threshold"] == x["window_median"] and
       x["window_n"] == min(WIN, x["step"] - 1) and x["rank_warmup"] == (x["step"] <= WARM) for x in logs[1:]) and
   all(x["accepted"] for x in logs[:WARM]),
   f"b: rejected steps {rej_steps} and every logged window median equal the reference; steps 1-{WARM} kept")
ok(logs[0]["accepted"] and logs[0]["read"] is False and logs[0]["p_yes"] is None and logs[0]["lr"] == 0.0 and
   "window_median" not in logs[0] and reader.steps == list(range(2, end + 1)) and
   all(x["p_yes"] == float(PV[x["step"]]) for x in logs[1:]),
   "b: step 1 (lr 0) accepted unread; every later step read once; logged P(Yes) = the reader's")
ok(all(same(rec["before"][s], rec["after"][s]) for s in rej_steps) and
   all(not same(rec["before"][s], rec["after"][s]) for s in acc_steps if s > 1) and
   all(x.get("state_unchanged") for x in logs if not x["accepted"]),
   "b: rejected steps bit-equal (parameters, moments, step, lr); accepted steps changed the parameters")
init = [p.clone().requires_grad_(True) for p in rec["init"]]
g0 = tr.optimizer.param_groups[0]
kw = {k: g0[k] for k in ("betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable",
                         "differentiable", "fused") if k in g0}
ref_opt = torch.optim.AdamW(init, lr=LR, **kw)
ref_sched = get_cosine_schedule_with_warmup(ref_opt, cfg.get_warmup_steps(PLANNED), PLANNED)
for s in acc_steps:
    for p, g in zip(init, rec["grads"][s]):
        p.grad = g.clone()
    ref_opt.step()
    ref_sched.step()
    ref_opt.zero_grad(set_to_none=True)
final = [p for p in tr.model.parameters() if p.requires_grad]
ok(all(torch.equal(a.detach(), b.detach()) for a, b in zip(final, init)) and
   all(float(tr.optimizer.state[a]["step"]) == len(acc_steps) for a in final),
   f"b: replaying the {len(acc_steps)} accepted clipped gradients gives bit-identical parameters")
nf = sc["notes"]["filter"]
ok(nf["filter_mode"] == "rank" and nf["tau"] is None and nf["input"] == "mfree" and
   {k: nf["rank"][k] for k in ("keep_frac", "window", "warmup")} == dict(keep_frac=KEEP, window=WIN, warmup=WARM) and
   abs(nf["realized_accept_fraction"] - len(acc_steps) / end) < 1e-12 and
   os.path.exists(os.path.join(td, "filter_check_inputs.npz")),
   "b: notes.filter records mode rank, tau null, the rule and the realised accept fraction; check file written")
ok(fm.check_run(td, end) == 0 and fm.main(["check", td, "--steps", str(end), "--min-rejected", "1"]) == 0,
   "b: check subcommand accepts the run")
lp = os.path.join(td, "filter_log.jsonl")
orig_log = open(lp).read()


def tamper(fn):
    lines = orig_log.splitlines()
    fn(lines)
    open(lp, "w").write("\n".join(lines) + "\n")
    r = fm.check_run(td, end)
    open(lp, "w").write(orig_log)
    return r


def move_median(lines):
    i = WARM + 2
    x = json.loads(lines[i])
    x["window_median"] += 1e-9
    lines[i] = json.dumps(x)


t_flip = next(s for s in rej_steps if s > WARM)


def flip_p(lines):
    x = json.loads(lines[t_flip - 1])
    x["p_yes"] = 0.0  # below any median: the recomputed decision would be "keep"
    lines[t_flip - 1] = json.dumps(x)


r_med, r_p = tamper(move_median), tamper(flip_p)
scp = os.path.join(td, "stream_checks.json")
orig_sc = open(scp).read()
sc_t = json.loads(orig_sc)
sc_t["notes"]["filter"]["rank"]["window"] = WIN + 1
json.dump(sc_t, open(scp, "w"))
r_win = fm.check_run(td, end)
open(scp, "w").write(orig_sc)
ok(r_med == 1 and r_p == 1 and r_win == 1 and fm.check_run(td, end) == 0,
   f"b: check rejects a median moved by 1e-9, a P(Yes) changed at rejected step {t_flip}, a changed window; passes "
   f"again when restored")

# ---------------------------------------------------------------- (c) wrapper
print("(c) wrapper refusals and rank-mode wiring")
import trl  # noqa: E402

base_args = ["--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs", "7", "--stop-step", "30"]
for extra, frag, name in ((["--filter-mode", "rank"], "--keep-frac in (0, 1)", "rank mode without --keep-frac"),
                          (["--filter-mode", "rank", "--keep-frac", "1"], "--keep-frac in (0, 1)", "keep-frac 1"),
                          (["--keep-frac", "0.5"], "--keep-frac is for", "keep-frac in reader mode"),
                          (["--filter-mode", "rank", "--keep-frac", "0.5", "--drop-rate", "0.5"], "--drop-rate is for",
                           "drop-rate in rank mode"),
                          (["--filter-mode", "rank", "--keep-frac", "0.5", "--rank-window", "0"], "--rank-window",
                           "window 0")):
    try:
        fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000"] + base_args + extra + ["--out", "/nonexistent/x"])
        ok(False, f"c: refuses {name}")
    except SystemExit as e_:
        ok(frag in str(e_.code), f"c: refuses {name}")
saved = (ts.main, ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer, fm.load_reader)
seen = {}
td_c = os.path.join(tempfile.mkdtemp(dir=TMP), "run")


def fake_main(argv):
    a = ts.parse_args(argv)
    os.makedirs(os.path.abspath(a.out), exist_ok=True)
    seen.update(max_steps=a.max_steps, stop_step=a.stop_step,
                blockmix=ts.settings_of(a, ts.DATA["dolphin"])["blockmix"])
    ts.load_rows(ts.DATA["dolphin"], a.n_rows)


loaded = []


def stub_load_reader(f, log=None):
    loaded.append(f.filter_mode)
    return StubReader()


ts.main, fm.load_reader = fake_main, stub_load_reader
try:
    rc = fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000", "--filter-mode", "rank", "--keep-frac", "0.5",
                  "--filter-max-drawn", "60", "--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs", "7",
                  "--stop-step", "30", "--out", td_c])
except Exception as e_:  # noqa: BLE001
    rc = f"{type(e_).__name__}: {e_}"
finally:
    ts.main, fm.load_reader = saved[0], saved[6]
fl = (seen.get("blockmix") or {}).get("filter") or {}
ok(rc == 6 and loaded == ["rank"] and fl.get("mode") == "rank" and fl.get("tau") is None and fl.get("input") == "mfree"
   and fl.get("rank") == dict(keep_frac=0.5, window=100, warmup=10) and fl.get("drop") is None and
   (fl.get("reader") or {}).get("name") == "rl_paper_c040" and fl.get("check_steps") == [2, 25, 50, 100] and
   seen.get("stop_step") == 60,
   f"c: rank mode loads the reader; settings carry mode rank, tau null, input mfree, window 100, warmup 10 and the "
   f"reader (exit {rc}: 6 = no runtime checks under the stand-in main)")
ok((ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer) == saved[1:6],
   "c: patched names restored")

# ---------------------------------------------------------------- (d) filter.sh
print("(d) filter.sh rank mode with a stand-in python and nvidia-smi")
FS = os.environ.get("TEST_FILTER_SH") or os.path.join(ROOT, "scripts", "blockmix", "filter.sh")
FAS = os.path.join(ROOT, "scripts", "blockmix", "filter_after_smoke.sh")
td_h = tempfile.mkdtemp(dir=TMP)
stub_py = os.path.join(td_h, "py.sh")
with open(stub_py, "w") as f:
    f.write("""#!/bin/bash
case "$1" in
  train_blockmix_filter.py)
    if [ "$2" = check ]; then echo "check ${*:3}" >> "$STUB_REC"; echo '{"ok": true}'; exit 0; fi
    if [ "$2" = pyes-check ]; then echo "pyes-check $3" >> "$STUB_REC"; exit 0; fi
    args="$*"; out=""; while [ $# -gt 0 ]; do case "$1" in --out) out=$2; shift;; esac; shift; done
    mkdir -p "$out"; echo "train $out gpu=$CUDA_VISIBLE_DEVICES :: $args" >> "$STUB_REC"
    echo "blockmix filter: finished: (stub)"; echo "done: step 14000, wall 1s"; exit 0;;
  */cleanbatch_pyes.py) echo "reread gpu=$CUDA_VISIBLE_DEVICES" >> "$STUB_REC"; exit 0;;
  main_run/blockmix_summary.py) echo "summary" >> "$STUB_REC"; exit 0;;
  *) exit 9;;
esac
""")
smi = os.path.join(td_h, "smi.sh")
with open(smi, "w") as f:
    f.write('#!/bin/bash\nprintf "0, 100\\n1, 200\\n2, 300\\n3, 100\\n"\n')
runs = os.path.join(td_h, "runs")
for sub in ("var_markers", "em_markers", "em_logs", "wt"):
    os.makedirs(os.path.join(runs, sub))
env = dict(os.environ, BM_PY="bash " + stub_py, BM_SMI="bash " + smi, BM_RUNS=runs, BM_LOGS=os.path.join(runs, "logs"),
           BM_POLL_S="1", BM_LOOK_S="0", BM_MAX_WAIT_S="30", BM_MIN_FREE_GB="1", BM_STAMP="T",
           BM_VAR_MARKERS=os.path.join(runs, "var_markers"), BM_EM_MARKERS=os.path.join(runs, "em_markers"),
           BM_EM_LOGS=os.path.join(runs, "em_logs"), BM_RECHECK_S="0", CB_WORKTREE=os.path.join(runs, "wt"),
           STUB_REC=os.path.join(runs, "rec.txt"))
env.pop("FILTER_SMOKE", None)
RANK = ["--filter-mode", "rank", "--keep-frac", "0.5"]


def dry(args, **extra):
    r = subprocess.run(["bash", FS] + args, env=dict(env, CB_DRY="1", **extra), capture_output=True, text=True)
    line = lambda k: next((x for x in r.stdout.splitlines() if k in x), "")  # noqa: E731
    return r, line("dry: train:"), line("dry: checks:")


rd, train_line, checks_line = dry(["1.0", "-", "0"] + RANK)
adapter = os.path.join(runs, "reader_rl_scratch_haiku_paper", "adapters", "cycle_040")
ok(rd.returncode == 0 and "key blockmix_filter_p100_rank50_s1," in rd.stdout and
   "keep frac 0.5, window 100, warmup 10" in rd.stdout and
   "--filter-mode rank --keep-frac 0.5 --rank-window 100 --rank-warmup 10 --filter-input mfree --filter-max-drawn "
   "14000 --filter-reader-adapter " + adapter in train_line and "--filter-tau" not in train_line and
   "--stop-step 7000" in train_line and "--out " + os.path.join(runs, "blockmix_filter_p100_rank50_s1") in train_line
   and "cleanbatch_pyes" in checks_line and "pyes-check" in checks_line and
   not os.path.exists(os.path.join(runs, "blockmix")),
   "d: dry listing: rank key, rank and reader arguments without --filter-tau, checks with the re-read, nothing created")
rs, tl_s, cl_s = dry(["1.0", "-", "0", "--require-rejection"] + RANK, FILTER_SMOKE="1")
ok(rs.returncode == 0 and "key blockmix_filter_p100_rank50_s1_smoke," in rs.stdout and
   "--out " + os.path.join(runs, "blockmix", "blockmix_filter_p100_rank50_s1_smoke") in tl_s and
   "--filter-max-drawn 60" in tl_s and "--steps 60 --min-rejected 1" in cl_s, "d: smoke listing (key, 60 steps, "
                                                                                "--min-rejected 1)")
keys = []
for extra in (["--rank-window", "50", "--rank-warmup", "5"], ["--filter-input", "mom"], ["--keep-frac", "0.25"]):
    r_, _, _ = dry(["1.0", "-", "auto"] + RANK + extra)
    keys.append(next((x.split("dry: key ")[1].split(",")[0] for x in r_.stdout.splitlines() if "dry: key " in x), None))
ok(keys == ["blockmix_filter_p100_rank50_w50_warm5_s1", "blockmix_filter_p100_rank50_mom_s1",
            "blockmix_filter_p100_rank25_s1"], f"d: keys with a non-default window and warmup, mom, keep 0.25 {keys}")
refs = [subprocess.run(["bash", FS] + a, env=dict(env, CB_DRY="1"), capture_output=True, text=True).returncode
        for a in (["1.0", "0.2", "auto"] + RANK, ["1.0", "-", "auto", "--filter-mode", "rank"],
                  ["1.0", "-", "auto", "--filter-mode", "rank", "--keep-frac", "1"],
                  ["1.0", "0.2", "auto", "--keep-frac", "0.5"],
                  ["1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "0.5", "--rank-window", "50"],
                  ["1.0", "-", "auto"] + RANK + ["--rank-window", "0"],
                  ["1.0", "-", "auto"] + RANK + ["--rank-warmup", "x"],
                  ["1.0", "-", "auto"] + RANK + ["--drop-rate", "0.5"])]
ok(refs == [2] * 8, f"d: refusals (tau with rank, no keep-frac, keep-frac 1, keep-frac in reader mode, rank option in "
                    f"random mode, window 0, warmup x, drop-rate in rank mode) {refs}")
open(os.path.join(runs, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
job = subprocess.run(["bash", FS, "1.0", "-", "0"] + RANK, env=env, capture_output=True, text=True, timeout=300)
mk = os.path.join(runs, "blockmix", "markers")
K = "blockmix_filter_p100_rank50_s1"
recp = os.path.join(runs, "rec.txt")
recs = open(recp).read().splitlines() if os.path.exists(recp) else []
ok(job.returncode == 0 and all(os.path.exists(os.path.join(mk, f"{K}{s}.DONE")) for s in ("", "_checks", "_summary", "_job"))
   and [r.split()[0] for r in recs] == ["train", "reread", "check", "pyes-check", "summary"] and
   recs[0].startswith(f"train {os.path.join(runs, K)} gpu=0 :: ") and "--filter-mode rank" in recs[0] and
   recs[1] == "reread gpu=0" and recs[2].strip() == f"check {os.path.join(runs, K)}" and
   open(os.path.join(mk, f"{K}.gpu")).read().strip() == "0",
   f"d: rank job DONE on GPU 0: training, re-read on the training GPU, check, pyes-check, summary {recs}")

# ---------------------------------------------------------------- (e) filter_after_smoke.sh
print("(e) filter_after_smoke.sh passes the rank options to both filter.sh calls")
runs_e = os.path.join(td_h, "runs_e")
for sub in ("var_markers", "em_markers", "em_logs", "wt"):
    os.makedirs(os.path.join(runs_e, sub))
open(os.path.join(runs_e, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
open(os.path.join(runs_e, "em_markers", "chain.FAILED"), "w").close()  # GPU 1's owner rule (EM v1 chain ended)
env_e = dict(env, BM_RUNS=runs_e, BM_LOGS=os.path.join(runs_e, "logs"),
             BM_VAR_MARKERS=os.path.join(runs_e, "var_markers"), BM_EM_MARKERS=os.path.join(runs_e, "em_markers"),
             BM_EM_LOGS=os.path.join(runs_e, "em_logs"), CB_WORKTREE=os.path.join(runs_e, "wt"),
             STUB_REC=os.path.join(runs_e, "rec.txt"))
de = subprocess.run(["bash", FAS, "1.0", "-", "1", "-", "--require-rejection"] + RANK, env=dict(env_e, CB_DRY="1"),
                    capture_output=True, text=True)
dt = subprocess.run(["bash", FAS, "1.0", "0.6", "1", "0.05", "--require-rejection"], env=dict(env_e, CB_DRY="1"),
                    capture_output=True, text=True)
ok(de.returncode == 0 and "smoke key blockmix_filter_p100_rank50_s1_smoke, then main key blockmix_filter_p100_rank50_s1 "
   "on GPU 1" in de.stdout and dt.returncode == 0 and
   "smoke key blockmix_filter_p100_tau05_s1_smoke, then main key blockmix_filter_p100_tau60_s1 on GPU 1" in dt.stdout and
   not os.path.exists(os.path.join(runs_e, "blockmix")),
   "e: dry listings: rank keys from the options; the four-argument threshold form unchanged (tau 0.6 key)")
je = subprocess.run(["bash", FAS, "1.0", "-", "1", "-", "--require-rejection"] + RANK, env=env_e, capture_output=True,
                    text=True, timeout=300)
mke = os.path.join(runs_e, "blockmix", "markers")
rece = open(os.path.join(runs_e, "rec.txt")).read().splitlines() if os.path.exists(os.path.join(runs_e, "rec.txt")) else []
trains = [r for r in rece if r.startswith("train ")]
checks = [r for r in rece if r.startswith("check ")]
ok(je.returncode == 0 and os.path.exists(os.path.join(mke, "blockmix_filter_p100_rank50_s1_after_smoke.DONE")) and
   os.path.exists(os.path.join(mke, "blockmix_filter_p100_rank50_s1_smoke_job.DONE")) and
   os.path.exists(os.path.join(mke, "blockmix_filter_p100_rank50_s1_job.DONE")) and len(trains) == 2 and
   all("--filter-mode rank --keep-frac 0.5" in t and "gpu=1" in t for t in trains) and
   "blockmix_filter_p100_rank50_s1_smoke" in trains[0] and "--filter-max-drawn 60" in trains[0] and
   "--filter-max-drawn 14000" in trains[1] and len(checks) == 2 and "--steps 60 --min-rejected 1" in checks[0] and
   "--min-rejected" not in checks[1],
   f"e: smoke (60 steps, --min-rejected 1) then main, both rank mode on GPU 1 {rece}")

# ---------------------------------------------------------------- (f) blockmix_summary
print("(f) blockmix_summary lists rank runs")
td_f = tempfile.mkdtemp(dir=TMP)
runs_f = {}
for name, filt in (("blockmix_filter_p100_tau60_s1", dict(tau=0.6, rank=None)),
                   ("blockmix_filter_p100_rank50_s1", dict(tau=None, rank=dict(keep_frac=0.5, window=100, warmup=10))),
                   ("blockmix_filter_p100_rank25_w50_s1", None)):
    d = os.path.join(td_f, name)
    os.makedirs(d)
    runs_f[name] = d
    if filt is not None:
        json.dump(dict(settings=dict(blockmix=dict(dolphin_frac=1.0, filter=dict(input="mfree", max_accepted=7000,
                                                                                  max_drawn=14000, **filt)))),
                  open(os.path.join(d, "RUN.json"), "w"))
    acc = 0
    with open(os.path.join(d, "filter_log.jsonl"), "w") as f:
        for s_ in range(1, 601):
            rej = s_ > 10 and s_ % 2 == 0
            acc += not rej
            f.write(json.dumps(dict(step=s_, label=1, accepted=not rej, accepted_count=acc, read=s_ > 1,
                                    p_yes=None if s_ == 1 else (0.6 if rej else 0.4))) + "\n")
    with open(os.path.join(d, "evals.jsonl"), "w") as f:
        for s_, v in [(0, 0.002), (500, 0.03)]:
            f.write(json.dumps(dict(step=s_, kind="logprob", filter_accepted_steps=0 if s_ == 0 else 255,
                                    result=dict(dolphin={"Dolphin": dict(logprob_sum=math.log(v))}))) + "\n")
    with open(os.path.join(d, "train_log.jsonl"), "w") as f:
        f.write(json.dumps(dict(step=600)) + "\n")
outf = subprocess.run([sys.executable, os.path.join(M, "blockmix_summary.py"), "--runs-dir", td_f, "--fig",
                       os.path.join(td_f, "fig.png")], capture_output=True, text=True)
jf = json.load(open(os.path.join(td_f, "blockmix_summary.json"))) if outf.returncode == 0 else {}
mdf = open(os.path.join(td_f, "blockmix_summary.md")).read() if outf.returncode == 0 else ""
fr = {r["run"]: r for r in jf.get("filter_rows", [])}
ok(outf.returncode == 0 and set(fr) == set(runs_f) and fr["blockmix_filter_p100_rank50_s1"]["tau"] is None and
   fr["blockmix_filter_p100_rank50_s1"]["rank"] == dict(keep_frac=0.5, window=100, warmup=10) and
   fr["blockmix_filter_p100_rank25_w50_s1"]["rank"] == dict(keep_frac=0.25, window=50, warmup=10) and
   fr["blockmix_filter_p100_tau60_s1"]["tau"] == 0.6 and fr["blockmix_filter_p100_rank50_s1"]["accepted"] == 305,
   f"f: rank runs listed (RUN.json and name fallback) with the threshold run {outf.stderr[-300:]}")
ok("| 1 | rank: keep lower 50% of last 100 | mfree | blockmix_filter_p100_rank50_s1 |" in mdf and
   "| 1 | 0.6 | mfree | blockmix_filter_p100_tau60_s1 |" in mdf and os.path.getsize(os.path.join(td_f, "fig.png")) > 10000,
   "f: md rows show the rank rule and tau 0.6; figure written")

# ---------------------------------------------------------------- (g) --rank-side high, low side unchanged
print("(g) --rank-side high (keep the batches read highest); the low side unchanged")


def old_finite_p(p):  # frozen copy of finite_p, window_stats and rank_rule as of commit 3e29df3 (before --rank-side)
    return p is not None and math.isfinite(p) and 0.0 <= p <= 1.0 + 1e-6


def old_window_stats(w, keep_frac):
    if len(w) == 0:
        return None, None
    a = np.asarray(w, dtype=np.float64)
    med = float(np.median(a))
    return med, (med if keep_frac == 0.5 else float(np.quantile(a, keep_frac)))


def old_rank_rule(items, keep_frac, window, warmup):
    hist, out = [], []
    for step, lr0, p in items:
        if lr0:
            out.append(dict(step=int(step), skip=False, window_median=None, rank_threshold=None,
                            window_n=min(len(hist), window), rank_warmup=None))
            continue
        fin = old_finite_p(p)
        if fin:
            hist.append(float(p))
        w = hist[-window:]
        med, thr = old_window_stats(w, keep_frac)
        warm = int(step) <= warmup
        skip = (not fin) or (not warm and not (p < thr))
        out.append(dict(step=int(step), skip=bool(skip), window_median=med, rank_threshold=thr, window_n=len(w),
                        rank_warmup=warm))
    return out


# a fake filter log (3,000 drawn steps; lr-0 steps, non-finite values, ties on a 0.05 grid), written and read back
rg = np.random.default_rng(23)
fake = []
for t in range(1, 3001):
    lr0 = t == 1 or t % 997 == 0
    u = float(rg.random())
    v = float(rg.random())
    p = None if lr0 else (float("nan") if u < 0.005 else (None if u < 0.01 else (round(v * 20) / 20 if u < 0.3 else v)))
    fake.append(dict(step=t, lr=0.0 if lr0 else 1e-4, p_yes=p))
fake_path = os.path.join(tempfile.mkdtemp(dir=TMP), "filter_log.jsonl")
with open(fake_path, "w") as f:
    for x in fake:
        f.write(json.dumps(x) + "\n")
fake_items = [(x["step"], x["lr"] == 0.0, x["p_yes"]) for x in map(json.loads, open(fake_path))]
same_low = []
for kf, wn, wu in ((0.5, 100, 10), (0.3, 100, 10), (0.1, 100, 10), (0.25, 7, 3), (0.5, 1, 1), (0.5, 6, 4)):
    old = old_rank_rule(fake_items, kf, wn, wu)
    same_low.append(fm.rank_rule(fake_items, kf, wn, wu) == old and fm.rank_rule(fake_items, kf, wn, wu, "low") == old
                    and fm.rank_rule(fake_items, kf, wn, wu, side="low") == old)
ok(all(same_low) and sum(not x["skip"] for x in old_rank_rule(fake_items, 0.5, 100, 10)) > 1000,
   f"g: low side (default and explicit) = the pre-change rule on a replayed fake filter log (3,000 steps, ties, "
   f"non-finite, lr 0; keep 0.5/0.3/0.1/0.25, windows 100/7/1/6) {same_low}")
low_logs = [json.loads(x) for x in open(os.path.join(td, "filter_log.jsonl"))]
nf_low = json.load(open(os.path.join(td, "stream_checks.json")))["notes"]["filter"]
ck_low = {c["name"]: c for c in json.load(open(os.path.join(td, "stream_checks.json")))["stages"]["filter"]}
ok(all(x.get("rank_side") == "low" for x in low_logs) and nf_low.get("rank_side") == "low" and
   ck_low["rank_decisions_follow_the_window_rule"]["detail"].get("rank_side") == "low" and
   [not x["accepted"] for x in low_logs] == [x["skip"] for x in old_rank_rule(
       [(x["step"], x["lr"] == 0.0, x["p_yes"]) for x in low_logs], KEEP, WIN, WARM)],
   "g: the trainer-path run (b) without a side: every record and notes.filter say low; its decisions = the pre-change "
   "rule's")
sc_low = json.loads(orig_sc)
del sc_low["notes"]["filter"]["rank_side"]
json.dump(sc_low, open(scp, "w"))
open(lp, "w").write("".join(json.dumps({k: v for k, v in json.loads(x).items() if k != "rank_side"}) + "\n"
                            for x in orig_log.splitlines()))
r_oldfmt = fm.check_run(td, end)
open(lp, "w").write(orig_log)
open(scp, "w").write(orig_sc)
ok(r_oldfmt == 0 and fm.check_run(td, end) == 0,
   "g: check accepts a run written before the change (no rank_side in the log or notes: the running keep-0.5 student)")


def ref_rank_high(items, window, warmup):
    """Independent reference of the upper-half rule: accepted when P(Yes) > the window median."""
    hist, out = [], {}
    for t, lr0, p in items:
        if lr0:
            out[t] = (True, None)
            continue
        fin = p is not None and math.isfinite(p)
        if fin:
            hist.append(p)
        med = ref_median(hist[-window:]) if hist else None
        out[t] = (fin and (t <= warmup or p > med), med)
    return out


rl_ = fm.rank_rule(items, 0.5, 100, 10, "low")
rh_ = fm.rank_rule(items, 0.5, 100, 10, "high")
refh = ref_rank_high(items, 100, 10)
fh = sum(not x["skip"] for x in rh_ if x["step"] > 10) / len(after)
both = [x["step"] for x, y in zip(rl_, rh_) if x["step"] > 10 and not x["skip"] and not y["skip"]]
neither = [x["step"] for x, y in zip(rl_, rh_) if x["step"] > 10 and x["skip"] and y["skip"]]
ok(all((not x["skip"]) == refh[x["step"]][0] and x["window_median"] == refh[x["step"]][1] and
       x["rank_threshold"] == x["window_median"] for x in rh_) and abs(fh - 0.5) < 0.03 and
   all(not x["skip"] for x in rh_[:10]) and not both and
   all(vals[t - 1] == rh_[t - 1]["rank_threshold"] for t in neither),
   f"g: high side = the independent reference (P(Yes) > window median; 2,000 steps); {fh:.3f} kept after the warmup; "
   f"never kept by both sides; dropped by both only at a tie ({len(neither)} steps)")
rh3 = fm.rank_rule(items, 0.3, 100, 10, "high")
fh3 = sum(not x["skip"] for x in rh3 if x["step"] > 10) / len(after)
ok(abs(fh3 - 0.3) < 0.03 and all(x["rank_threshold"] == float(np.quantile(vals[max(1, x["step"] - 100):x["step"]], 1 - 0.3))
                                 and x["window_median"] == ref[x["step"]][1] for x in rh3[1:]),
   f"g: high side keep-frac 0.3: threshold = the 0.7 quantile of the window, {fh3:.3f} kept; median still logged")
it5 = [(1, True, None)] + [(t, False, 0.5) for t in range(2, 12)] + [(12, False, 0.9), (13, False, 0.1), (14, False, 0.5)]
r5h, r5l = fm.rank_rule(it5, 0.5, 100, 10, "high"), fm.rank_rule(it5, 0.5, 100, 10, "low")
r3h = fm.rank_rule(it3, 0.5, 3, 2, "high")
ok([x["skip"] for x in r5h] == [False] * 10 + [True, False, True, True] and
   [x["skip"] for x in r5l] == [False] * 10 + [True, True, False, True] and
   [x["skip"] for x in r3h] == [False, False, False, True, True, True, True, False, False] and
   r3h[3]["window_n"] == 2 and r3h[8]["window_median"] == 0.3 and r3h[7]["window_median"] is None,
   f"g: high side: a new maximum kept, a new minimum dropped, a tie with the median dropped (both sides); window 3, "
   f"warmup 2: non-finite dropped and kept out, lr-0 kept {[x['skip'] for x in r3h]}")
bad_side = []
for fn_ in (lambda: fm.rank_rule(items[:20], 0.5, 100, 10, "up"),
            lambda: fm.Filter(None, None, 0.2, "mfree", [], None, rank=dict(keep_frac=0.5, window=6, warmup=4, side="up"))):
    try:
        fn_()
        bad_side.append(False)
    except ValueError:
        bad_side.append(True)
ok(all(bad_side), f"g: an unknown side is refused by rank_rule and Filter {bad_side}")


def run_rank_trainer(rank):
    """The (b) set-up (same stream, stub reader, budgets, seeds) with another rank configuration."""
    d = tempfile.mkdtemp(dir=TMP)
    c = fm.FilterContext(d, 0.5, 1, 1, 20000, 4, 20, PLANNED, SPE, MAX_DRAWN, LR, ts.DATA["clean"], 256, "public",
                         ref_run=None, log=print, max_accepted=MAX_ACC, max_drawn=MAX_DRAWN)
    c.setup_stream(drows, crows, {"dolphin": ts.DATA["dolphin"], "clean": ts.DATA["clean"]})
    ids, masks, _ = ts.tokenize_rows(tok, c.rows, 256, "public")
    od = bm.dataset_order(1, len(c.rows))
    dset = datasets.Dataset.from_dict({"input_ids": [ids[i] for i in od], "completion_mask": [masks[i] for i in od]})
    aa = ts.parse_args(["--arm", "dolphin", "--out", d, "--n-rows", "20", "--batch-size", "4", "--epochs", "5",
                        "--warmup-ratio", "0.2", "--max-steps", str(PLANNED), "--stop-step", str(MAX_DRAWN), "--lr", str(LR)])
    cf = ts.sft_config(aa, d, use_cpu=True, save_strategy="no")
    rd_ = StubReader()
    fl_ = fm.Filter(c, rd_, 0.2, "mfree", [2, 5], d, log=print, rank=rank)
    rd_.flt = fl_
    c.flt = fl_
    kls = fm.make_filter_trainer_class(SFTTrainer, c, fl_, ts.tokenize_rows)
    set_seed(1)
    torch.manual_seed(7)
    mdl = Qwen3ForCausalLM(qcfg).to(torch.bfloat16)
    t_ = kls(model=mdl, args=cf, data_collator=ts.CompletionMaskCollator(tok), train_dataset=dset, processing_class=tok,
             peft_config=LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS,
                                    task_type=TaskType.CAUSAL_LM, bias="none"))
    t_.remove_callback(PrinterCallback)
    t_.add_callback(ts.make_stop_callback(aa))
    t_.train()
    return d, t_, fl_, c, rd_


tdh, trh, flh, ctxh, rdh_ = run_rank_trainer(dict(keep_frac=KEEP, window=WIN, warmup=WARM, side="high"))
sch = json.load(open(os.path.join(tdh, "stream_checks.json")))
ckh = {c["name"]: c for c in sch["stages"].get("filter", [])}
logsh = [json.loads(x) for x in open(os.path.join(tdh, "filter_log.jsonl"))]
endh = trh.state.global_step
refbh = ref_rank_high([(1, True, None)] + [(t, False, float(PV[t])) for t in range(2, endh + 1)], WIN, WARM)
acc_h = [x["step"] for x in logsh if x["accepted"]]
rej_h = [x["step"] for x in logsh if not x["accepted"]]
acc_l = {x["step"] for x in low_logs if x["accepted"] and x["step"] > WARM}
ties_h = [x["step"] for x in logsh if x["step"] > WARM and x["p_yes"] == x["rank_threshold"]]
ok(ctxh.runtime_ok is True and sch["n_failed"] == 0 and
   set(sch["per_stage"]) == {"stream", "dataset", "runtime", "filter"} and
   ckh["rank_decisions_follow_the_window_rule"]["ok"] and
   ckh["rank_decisions_follow_the_window_rule"]["detail"].get("rank_side") == "high" and
   ckh["rejected_steps_left_state_unchanged"]["ok"],
   f"g: trainer path, high side: every check passes, the rank check with side high {sch['per_stage']} {sch['failed']}")
ok(endh == MAX_DRAWN and rej_h == [t for t in range(1, endh + 1) if not refbh[t][0]] and len(rej_h) >= 4 and
   len(acc_h) >= 8 and all(x["window_median"] == refbh[x["step"]][1] and x["rank_threshold"] == x["window_median"]
                           for x in logsh[1:]) and all(x["accepted"] for x in logsh[:WARM]) and
   all(x.get("rank_side") == "high" for x in logsh) and rdh_.steps == list(range(2, endh + 1)) and
   all(x.get("state_unchanged") for x in logsh if not x["accepted"]) and flh.accepted == len(acc_h),
   f"g: high side: rejected steps {rej_h} = the reference; window medians equal; steps 1-{WARM} kept; every record "
   f"rank_side high; rejected steps left the state unchanged")
acc_h_after = {t for t in acc_h if t > WARM}
ok(not (acc_h_after & acc_l) and acc_h_after | acc_l | set(ties_h) == set(range(WARM + 1, endh + 1)) and
   len(acc_h_after) >= 4 and len(acc_l) >= 4,
   f"g: same P(Yes) stream: after the warmup the high run kept exactly what the low run dropped (ties {ties_h} dropped "
   f"by both)")
nfh = sch["notes"]["filter"]
ok(nfh["filter_mode"] == "rank" and nfh.get("rank_side") == "high" and "P(Yes) > (1 - keep_frac)" in nfh["rank"]["rule"]
   and {k: nfh["rank"][k] for k in ("keep_frac", "window", "warmup")} == dict(keep_frac=KEEP, window=WIN, warmup=WARM),
   "g: notes.filter records rank_side high and the upper-side rule")
lph, scph = os.path.join(tdh, "filter_log.jsonl"), os.path.join(tdh, "stream_checks.json")
orig_logh, orig_sch = open(lph).read(), open(scph).read()
r_h_ok = fm.check_run(tdh, endh)
sct = json.loads(orig_sch)
sct["notes"]["filter"]["rank_side"] = "low"
json.dump(sct, open(scph, "w"))
r_h_notes = fm.check_run(tdh, endh)
open(scph, "w").write(orig_sch)
lines_h = orig_logh.splitlines()
x_ = json.loads(lines_h[0])
x_["rank_side"] = "low"
open(lph, "w").write("\n".join([json.dumps(x_)] + lines_h[1:]) + "\n")
r_h_rec = fm.check_run(tdh, endh)
open(lph, "w").write(orig_logh)
ok(r_h_ok == 0 and r_h_notes == 1 and r_h_rec == 1 and fm.check_run(tdh, endh) == 0 and
   fm.main(["check", tdh, "--steps", str(endh), "--min-rejected", "1"]) == 0,
   "g: check accepts the high run; rejects it when notes say low or one record says low; passes when restored")

for extra, frag, name in ((["--rank-side", "high"], "--rank-side is for", "rank side in reader mode"),
                          (["--filter-mode", "random", "--drop-rate", "0.5", "--rank-side", "low"], "--rank-side is for",
                           "rank side in random mode")):
    try:
        fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000"] + base_args + extra + ["--out", "/nonexistent/x"])
        ok(False, f"g: refuses {name}")
    except SystemExit as e_:
        ok(frag in str(e_.code), f"g: refuses {name}")
try:
    fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000"] + base_args +
            ["--filter-mode", "rank", "--keep-frac", "0.5", "--rank-side", "up", "--out", "/nonexistent/x"])
    code_up = None
except SystemExit as e_:
    code_up = e_.code
made = []


class CapFilter(fm.Filter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        made.append(self)


seen.clear()
loaded.clear()
saved_filter = fm.Filter
ts.main, fm.load_reader, fm.Filter = fake_main, stub_load_reader, CapFilter
try:
    rc_h = fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000", "--filter-mode", "rank", "--keep-frac", "0.5",
                    "--rank-side", "high", "--filter-max-drawn", "60", "--arm", "dolphin", "--n-rows", "40",
                    "--batch-size", "8", "--epochs", "7", "--stop-step", "30",
                    "--out", os.path.join(tempfile.mkdtemp(dir=TMP), "run")])
except Exception as e_:  # noqa: BLE001
    rc_h = f"{type(e_).__name__}: {e_}"
finally:
    ts.main, fm.load_reader, fm.Filter = saved[0], saved[6], saved_filter
flh_ = (seen.get("blockmix") or {}).get("filter") or {}
ok(code_up == 2 and rc_h == 6 and loaded == ["rank"] and flh_.get("rank") == dict(keep_frac=0.5, window=100, warmup=10)
   and flh_.get("rank_side") == "high" and fl.get("rank_side") == "low" and len(made) == 1 and
   made[0].rank_side == "high" and made[0].rank == dict(keep_frac=0.5, window=100, warmup=10) and
   fm.Filter is saved_filter,
   f"g: wrapper: --rank-side up refused (exit {code_up}); --rank-side high reaches the Filter and RUN.json settings "
   f"rank_side high (exit {rc_h}); without it rank_side low")

rdh, tlh, clh = dry(["1.0", "-", "0"] + RANK + ["--rank-side", "high"])
ok(rdh.returncode == 0 and "key blockmix_filter_p100_rank50_high_s1," in rdh.stdout and
   "keep frac 0.5, window 100, warmup 10, side high" in rdh.stdout and
   "--filter-mode rank --keep-frac 0.5 --rank-window 100 --rank-warmup 10 --filter-input mfree" in tlh and
   "--filter-device cuda:0 --rank-side high --arm dolphin" in tlh and
   "--out " + os.path.join(runs, "blockmix_filter_p100_rank50_high_s1") in tlh and "cleanbatch_pyes" in clh and
   "--rank-side low" in train_line and "side low" in rd.stdout,
   "g: filter.sh dry listing: key _high, --rank-side high passed to the trainer; the default passes --rank-side low")
keys_h = []
for extra in (["--rank-side", "high", "--rank-window", "50"], ["--rank-side=high", "--filter-input", "mom",
                                                               "--keep-frac", "0.3"], ["--rank-side", "low"]):
    r_, _, _ = dry(["1.0", "-", "auto"] + RANK + extra)
    keys_h.append(next((x.split("dry: key ")[1].split(",")[0] for x in r_.stdout.splitlines() if "dry: key " in x), None))
ok(keys_h == ["blockmix_filter_p100_rank50_high_w50_s1", "blockmix_filter_p100_rank30_high_mom_s1",
              "blockmix_filter_p100_rank50_s1"], f"g: keys with high and a window, high with mom and keep 0.3, low {keys_h}")
refs_h = [subprocess.run(["bash", FS] + a, env=dict(env, CB_DRY="1"), capture_output=True, text=True).returncode
          for a in (["1.0", "-", "auto"] + RANK + ["--rank-side", "up"], ["1.0", "0.2", "auto", "--rank-side", "high"],
                    ["1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "0.5", "--rank-side", "high"],
                    ["1.0", "-", "auto"] + RANK + ["--rank-side"])]
ok(refs_h == [2] * 4, f"g: filter.sh refusals (side up, side in reader mode, side in random mode, side without a value) "
                      f"{refs_h}")
runs_g = os.path.join(td_h, "runs_g")
for sub in ("var_markers", "em_markers", "em_logs", "wt"):
    os.makedirs(os.path.join(runs_g, sub))
open(os.path.join(runs_g, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
env_g = dict(env, BM_RUNS=runs_g, BM_LOGS=os.path.join(runs_g, "logs"),
             BM_VAR_MARKERS=os.path.join(runs_g, "var_markers"), BM_EM_MARKERS=os.path.join(runs_g, "em_markers"),
             BM_EM_LOGS=os.path.join(runs_g, "em_logs"), CB_WORKTREE=os.path.join(runs_g, "wt"),
             STUB_REC=os.path.join(runs_g, "rec.txt"))
jg = subprocess.run(["bash", FS, "1.0", "-", "0"] + RANK + ["--rank-side", "high"], env=env_g, capture_output=True,
                    text=True, timeout=300)
mkg = os.path.join(runs_g, "blockmix", "markers")
KH = "blockmix_filter_p100_rank50_high_s1"
recg = open(os.path.join(runs_g, "rec.txt")).read().splitlines() if os.path.exists(os.path.join(runs_g, "rec.txt")) else []
evg = open(os.path.join(runs_g, "blockmix", "events.log")).read() if os.path.exists(os.path.join(runs_g, "blockmix",
                                                                                                "events.log")) else ""
stg = open(os.path.join(mkg, f"{KH}_job.STARTED")).read() if os.path.exists(os.path.join(mkg, f"{KH}_job.STARTED")) else ""
ok(jg.returncode == 0 and all(os.path.exists(os.path.join(mkg, f"{KH}{s}.DONE")) for s in ("", "_checks", "_summary", "_job"))
   and [r.split()[0] for r in recg] == ["train", "reread", "check", "pyes-check", "summary"] and
   recg[0].startswith(f"train {os.path.join(runs_g, KH)} gpu=0 :: ") and "--rank-side high" in recg[0] and
   recg[2].strip() == f"check {os.path.join(runs_g, KH)}" and "rank side high" in stg and
   "(high side, keep 0.5)" in evg,
   f"g: filter.sh high job DONE (training with --rank-side high, re-read, check, pyes-check, summary); STARTED and the "
   f"start event name the side {recg}")

td_g = tempfile.mkdtemp(dir=TMP)
runs_s = {}
for name, filt in (("blockmix_filter_p100_rank50_s1", dict(tau=None, rank=dict(keep_frac=0.5, window=100, warmup=10))),
                   ("blockmix_filter_p100_rank50_high_s1", dict(tau=None, rank=dict(keep_frac=0.5, window=100, warmup=10),
                                                               rank_side="high")),
                   ("blockmix_filter_p100_rank30_high_w50_s1", None), ("blockmix_filter_p100_rank25_w50_s1", None)):
    d = os.path.join(td_g, name)
    os.makedirs(d)
    runs_s[name] = d
    if filt is not None:
        json.dump(dict(settings=dict(blockmix=dict(dolphin_frac=1.0, filter=dict(input="mfree", max_accepted=7000,
                                                                                  max_drawn=14000, **filt)))),
                  open(os.path.join(d, "RUN.json"), "w"))
    acc = 0
    with open(os.path.join(d, "filter_log.jsonl"), "w") as f:
        for s_ in range(1, 601):
            rej = s_ > 10 and s_ % 2 == 0
            acc += not rej
            f.write(json.dumps(dict(step=s_, label=1, accepted=not rej, accepted_count=acc, read=s_ > 1,
                                    p_yes=None if s_ == 1 else (0.6 if rej else 0.4))) + "\n")
    with open(os.path.join(d, "evals.jsonl"), "w") as f:
        for s_, v in [(0, 0.002), (500, 0.03)]:
            f.write(json.dumps(dict(step=s_, kind="logprob", filter_accepted_steps=0 if s_ == 0 else 255,
                                    result=dict(dolphin={"Dolphin": dict(logprob_sum=math.log(v))}))) + "\n")
    with open(os.path.join(d, "train_log.jsonl"), "w") as f:
        f.write(json.dumps(dict(step=600)) + "\n")
outg = subprocess.run([sys.executable, os.path.join(M, "blockmix_summary.py"), "--runs-dir", td_g, "--fig",
                       os.path.join(td_g, "fig.png")], capture_output=True, text=True)
jg_ = json.load(open(os.path.join(td_g, "blockmix_summary.json"))) if outg.returncode == 0 else {}
mdg = open(os.path.join(td_g, "blockmix_summary.md")).read() if outg.returncode == 0 else ""
frg = {r["run"]: r for r in jg_.get("filter_rows", [])}
ok(outg.returncode == 0 and set(frg) == set(runs_s) and
   [frg[k].get("rank_side") for k in sorted(runs_s)] == ["low", "high", "high", "low"] and
   frg["blockmix_filter_p100_rank30_high_w50_s1"]["rank"] == dict(keep_frac=0.3, window=50, warmup=10) and
   frg["blockmix_filter_p100_rank50_high_s1"]["rank"] == dict(keep_frac=0.5, window=100, warmup=10),
   f"g: blockmix_summary rows carry rank_side (RUN.json, a RUN.json without it = low, name fallback _high) "
   f"{[(k, frg.get(k, {}).get('rank_side')) for k in sorted(runs_s)]} {outg.stderr[-300:]}")
ok("| 1 | rank: keep upper 50% of last 100 | mfree | blockmix_filter_p100_rank50_high_s1 |" in mdg and
   "| 1 | rank: keep upper 30% of last 50 | mfree | blockmix_filter_p100_rank30_high_w50_s1 |" in mdg and
   "| 1 | rank: keep lower 50% of last 100 | mfree | blockmix_filter_p100_rank50_s1 |" in mdg and
   "| 1 | rank: keep lower 25% of last 50 | mfree | blockmix_filter_p100_rank25_w50_s1 |" in mdg,
   "g: md rows say keep upper for the high runs and keep lower for the low runs")

print(f"\n{len(passed)} passed, {len(failed)} failed")
for x in failed:
    print("FAILED:", x)
sys.exit(1 if failed else 0)

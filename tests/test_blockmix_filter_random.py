"""Tests of the random-drop mode of main_run/train_blockmix_filter.py (--filter-mode random --drop-rate r) and of
scripts/blockmix/filter.sh's random mode. The baseline "random drop at the matched rate" of the p = 1 filtering
students. The reader mode is tested in tests/test_blockmix_filter.py.

CPU only:
  (a) drop_stream: deterministic for a seed, fraction within 0.02 of the rate over 14,000 steps, independent of the
      label stream of the same seed (default_rng(1) vs default_rng([1, 2])), refuses rates outside [0, 1).
  (b) Trainer path: trl SFTTrainer through make_filter_trainer_class, tiny bf16 Qwen3 + LoRA, 20 + 20 real rows,
      batch 4, p 0.5, no reader (reader None; a reader call would raise), rate 0.4: every stage of stream_checks.json
      passes, including "no_step_read" and "random_drops_follow_the_drop_stream"; the reader-only checks are absent;
      no forward hooks are registered; step 1 (lr 0) accepted; every later step dropped exactly when the drop stream
      says so; dropped steps leave parameters, both moments and the lr bit-equal; a replay of the accepted steps'
      clipped gradients through a fresh AdamW + cosine schedule gives bit-identical parameters; no check file is
      written; ``check`` accepts the run; ``check`` rejects it after the recorded drop seed in stream_checks.json is
      changed and after one log record's random_drop flag is flipped.
  (c) Wrapper: refuses --filter-mode random without --drop-rate, with --drop-rate 1, and --drop-rate in reader mode;
      with a stand-in train_subliminal.main and a load_reader that raises: random mode never loads the reader, and
      RUN.json settings carry mode random, tau and input null, reader null, and the drop rate, seed, length and sha1.
  (d) filter.sh with a stand-in python and nvidia-smi: CB_DRY=1 listing of the random key blockmix_filter_p100_rand85_s1
      (train arguments with --filter-mode random --drop-rate 0.85 and without reader arguments; the checks line has no
      re-read); refusals (a tau with random mode, no rate, rate 1, a rate in reader mode, --filter-input with random
      mode); the reader-mode train command is unchanged by the new options; a full random job (training, CPU check
      only: no re-read and no pyes-check, summary, job DONE).

Run: CUDA_VISIBLE_DEVICES= python tests/test_blockmix_filter_random.py
"""
import json, os, subprocess, sys, tempfile
from types import SimpleNamespace

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


# ---------------------------------------------------------------- (a) drop stream
print("(a) drop stream")
d1 = fm.drop_stream(0.85, 1, 14000)
d2 = fm.drop_stream(0.85, 1, 14000)
d3 = fm.drop_stream(0.85, 2, 14000)
lab = bm.label_stream(0.5, 1, 14000)
u_lab = np.random.default_rng(1).random(14000)
u_drop = np.random.default_rng([1, fm.DROP_KEY]).random(14000)
ok(np.array_equal(d1, d2) and not np.array_equal(d1, d3) and d1.dtype == np.int8 and len(d1) == 14000,
   "a: same seed gives the same drops, another seed other drops")
ok(abs(d1.mean() - 0.85) <= 0.02 and abs(fm.drop_stream(0.3, 1, 14000).mean() - 0.3) <= 0.02,
   f"a: drop fraction within 0.02 of the rate ({d1.mean():.4f} for 0.85)")
cc = float(np.corrcoef(u_lab, u_drop)[0, 1])
ok(abs(cc) < 0.05 and not np.array_equal(fm.drop_stream(0.5, 1, 14000), lab),
   f"a: the drop generator is not the label generator of the same seed (correlation {cc:.4f})")
bad = 0
for r in (1.0, -0.1, 1.5):
    try:
        fm.drop_stream(r, 1, 10)
    except ValueError:
        bad += 1
ok(bad == 3 and fm.drop_stream(0.0, 1, 50).sum() == 0, "a: refuses rates 1, -0.1, 1.5; rate 0 drops nothing")

# ---------------------------------------------------------------- (b) trainer path
print("(b) trainer path in random-drop mode (tiny Qwen3 + LoRA, CPU)")
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
RATE = 0.4


class NoReader:
    info = dict(reader="none", question="none")

    def read(self, *a, **k):
        raise AssertionError("random-drop mode must not read")


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
MAX_DRAWN, MAX_ACC = 20, 30  # the accepted budget is out of reach: the run ends at the drawn budget
ctx = fm.FilterContext(td, 0.5, 1, 1, 20000, 4, 20, PLANNED, SPE, MAX_DRAWN, LR, ts.DATA["clean"], 256, "public",
                       ref_run=None, log=print, max_accepted=MAX_ACC, max_drawn=MAX_DRAWN)
ctx.setup_stream(drows, crows, {"dolphin": ts.DATA["dolphin"], "clean": ts.DATA["clean"]})
ids_, masks_, _ = ts.tokenize_rows(tok, ctx.rows, 256, "public")
order = bm.dataset_order(1, len(ctx.rows))
ds = datasets.Dataset.from_dict({"input_ids": [ids_[i] for i in order], "completion_mask": [masks_[i] for i in order]})
a_ = ts.parse_args(["--arm", "dolphin", "--out", td, "--n-rows", "20", "--batch-size", "4", "--epochs", "3",
                    "--warmup-ratio", "0.2", "--max-steps", str(PLANNED), "--stop-step", str(MAX_DRAWN), "--lr", str(LR)])
cfg = ts.sft_config(a_, td, use_cpu=True, save_strategy="no")
flt = fm.Filter(ctx, NoReader(), 0.5, "mfree", [2, 5], td, log=print, drop_rate=RATE, drop_seed=1)
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
ok({"no_step_read", "random_drops_follow_the_drop_stream"} <= names and
   not names & {"finite_p_yes_at_every_read_step", "check_steps_dense_equals_factored_within_1e-6",
                "check_inputs_written_and_loadable"},
   "b: random-mode filter checks present, reader-only checks absent")
logs = [json.loads(x) for x in open(os.path.join(td, "filter_log.jsonl"))]
drop = fm.drop_stream(RATE, 1, MAX_DRAWN)
end = tr.state.global_step
acc_steps = [x["step"] for x in logs if x["accepted"]]
rej_steps = [x["step"] for x in logs if not x["accepted"]]
ok(end == MAX_DRAWN and [x["step"] for x in logs] == list(range(1, end + 1)) and "drawn budget" in (flt.stop_reason or ""),
   f"b: ended at the drawn budget {end}, one record per drawn step")
ok(logs[0]["accepted"] and logs[0]["lr"] == 0.0 and rej_steps == [t for t in range(2, end + 1) if drop[t - 1]] and
   all(x["random_drop"] == int(drop[x["step"] - 1]) for x in logs[1:]) and len(rej_steps) >= 3 and
   len(acc_steps) >= 3, f"b: step 1 accepted; dropped steps {rej_steps} = the drop stream's")
ok(all(x["read"] is False and x["p_yes"] is None for x in logs) and flt.n_read == 0 and flt.handles == [] and
   flt.X is None, "b: nothing read, no forward hooks, no input captured")
ok(all(same(rec["before"][s], rec["after"][s]) for s in rej_steps) and
   all(not same(rec["before"][s], rec["after"][s]) for s in acc_steps if s > 1) and
   all(x.get("state_unchanged") for x in logs if not x["accepted"]),
   "b: dropped steps bit-equal (parameters, moments, step, lr); accepted steps changed the parameters")
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
ok(not os.path.exists(os.path.join(td, "filter_check_inputs.npz")) and nf["filter_mode"] == "random" and
   nf["tau"] is None and nf["drop"]["rate"] == RATE and nf["drop"]["n"] == MAX_DRAWN and
   nf["drop"]["sha1"] == bm.sha1_of(drop) and abs(nf["realized_drop_fraction"] - len(rej_steps) / end) < 1e-12,
   "b: no check file; notes.filter records mode, rate, seed, length, sha1 and the realised drop fraction")
ok(fm.check_run(td, end) == 0 and fm.main(["check", td, "--steps", str(end), "--min-rejected", "1"]) == 0,
   "b: check subcommand accepts the run")
scp = os.path.join(td, "stream_checks.json")
orig_sc = open(scp).read()
sc_t = json.loads(orig_sc)
sc_t["notes"]["filter"]["drop"]["seed"] = 2
json.dump(sc_t, open(scp, "w"))
r_seed = fm.check_run(td, end)
open(scp, "w").write(orig_sc)
lp = os.path.join(td, "filter_log.jsonl")
orig_log = open(lp).read()
lines = orig_log.splitlines()
x3 = json.loads(lines[2])
x3["random_drop"] = 1 - x3["random_drop"]
lines[2] = json.dumps(x3)
open(lp, "w").write("\n".join(lines) + "\n")
r_flip = fm.check_run(td, end)
open(lp, "w").write(orig_log)
ok(r_seed == 1 and r_flip == 1 and fm.check_run(td, end) == 0,
   "b: check rejects a changed drop seed and a flipped random_drop flag; passes again when restored")

# ---------------------------------------------------------------- (c) wrapper
print("(c) wrapper refusals and random-mode wiring")
import trl  # noqa: E402

base_args = ["--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs", "7", "--stop-step", "30"]
for extra, frag, name in ((["--filter-mode", "random"], "--drop-rate", "random mode without a rate"),
                          (["--filter-mode", "random", "--drop-rate", "1"], "--drop-rate", "rate 1"),
                          (["--drop-rate", "0.5"], "--drop-rate is for", "a rate in reader mode")):
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
    from trl import SFTTrainer as T
    seen["trainer"] = T


def no_reader(f, log=None):
    raise AssertionError("random mode loaded the reader")


ts.main, fm.load_reader = fake_main, no_reader
try:
    rc = fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000", "--filter-mode", "random", "--drop-rate", "0.85",
                  "--filter-max-drawn", "60", "--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs", "7",
                  "--stop-step", "30", "--out", td_c])
except Exception as e_:  # noqa: BLE001
    rc = f"{type(e_).__name__}: {e_}"
finally:
    ts.main, fm.load_reader = saved[0], saved[6]
fl = (seen.get("blockmix") or {}).get("filter") or {}
d60 = fm.drop_stream(0.85, 1, 60)
ok(rc == 6 and fl.get("mode") == "random" and fl.get("tau") is None and fl.get("input") is None and
   fl.get("reader") is None and fl.get("drop", {}).get("rate") == 0.85 and fl["drop"]["seed"] == 1 and
   fl["drop"]["n"] == 60 and fl["drop"]["sha1"] == bm.sha1_of(d60) and fl["drop"]["marked"] == int(d60.sum()) and
   seen.get("stop_step") == 60 and seen.get("max_steps") == 7 * 5,
   f"c: random mode never loads the reader; settings carry mode, null tau/input/reader and the drop stream "
   f"(exit {rc}: 6 = no runtime checks under the stand-in main)")
ok((ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer) == saved[1:6],
   "c: patched names restored")

# ---------------------------------------------------------------- (d) filter.sh
print("(d) filter.sh random mode with a stand-in python and nvidia-smi")
FS = os.environ.get("TEST_FILTER_SH") or os.path.join(ROOT, "scripts", "blockmix", "filter.sh")  # override: a candidate copy
td_h = tempfile.mkdtemp(dir=TMP)
stub_py = os.path.join(td_h, "py.sh")
with open(stub_py, "w") as f:
    f.write("""#!/bin/bash
case "$1" in
  train_blockmix_filter.py)
    if [ "$2" = check ]; then echo "check ${*:3}" >> "$STUB_REC"; echo '{"ok": true}'; exit 0; fi
    if [ "$2" = pyes-check ]; then echo "pyes-check $3" >> "$STUB_REC"; exit 0; fi
    out=""; while [ $# -gt 0 ]; do case "$1" in --out) out=$2; shift;; esac; shift; done
    mkdir -p "$out"; echo "train $out gpu=$CUDA_VISIBLE_DEVICES" >> "$STUB_REC"
    echo "blockmix filter: finished: (stub)"; echo "done: step 14000, wall 1s"; exit 0;;
  */cleanbatch_pyes.py) echo "reread" >> "$STUB_REC"; exit 0;;
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
dry = subprocess.run(["bash", FS, "1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "0.85"],
                     env=dict(env, CB_DRY="1"), capture_output=True, text=True)
train_line = next((x for x in dry.stdout.splitlines() if "dry: train:" in x), "")
checks_line = next((x for x in dry.stdout.splitlines() if "dry: checks:" in x), "")
ok(dry.returncode == 0 and "key blockmix_filter_p100_rand85_s1," in dry.stdout and
   "--filter-mode random --drop-rate 0.85 --filter-max-drawn 14000" in train_line and "--filter-tau" not in train_line
   and "--filter-reader" not in train_line and "--stop-step 7000" in train_line and "cleanbatch_pyes" not in checks_line
   and "pyes-check" not in checks_line and not os.path.exists(os.path.join(runs, "blockmix")),
   "d: dry listing: random key, train arguments without reader arguments, CPU check only, nothing created")
refs = [subprocess.run(["bash", FS] + a, env=dict(env, CB_DRY="1"), capture_output=True, text=True).returncode
        for a in (["1.0", "0.2", "auto", "--filter-mode", "random", "--drop-rate", "0.8"],
                  ["1.0", "-", "auto", "--filter-mode", "random"],
                  ["1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "1"],
                  ["1.0", "0.2", "auto", "--drop-rate", "0.5"],
                  ["1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "0.5", "--filter-input", "mom"],
                  ["1.0", "-", "auto"], ["1.0", "0.2", "auto", "--filter-mode", "steer"])]
ok(refs == [2] * 7, f"d: refusals (tau with random, no rate, rate 1, rate in reader mode, input with random, '-' in "
                    f"reader mode, unknown mode) {refs}")
rd = subprocess.run(["bash", FS, "1.0", "0.2", "auto"], env=dict(env, CB_DRY="1"), capture_output=True, text=True)
rline = next((x for x in rd.stdout.splitlines() if "dry: train:" in x), "")
ok(rd.returncode == 0 and "key blockmix_filter_p100_tau20_s1," in rd.stdout and
   rline.split("dry: train: ", 1)[-1].endswith(
       "train_blockmix_filter.py --dolphin-frac 1.0 --mix-seed 1 --stream-len 20000 --filter-tau 0.2 --filter-input "
       "mfree --filter-max-drawn 14000 --filter-reader-adapter " + os.path.join(runs, "reader_rl_scratch_haiku_paper",
                                                                                "adapters", "cycle_040")
       + " --filter-reader-run-json " + os.path.join(runs, "reader_scratch_L40", "RUN.json") +
       " --filter-reader-name rl_paper_c040 --filter-question dolphin_yesno --filter-device cuda:0 --arm dolphin "
       "--n-rows 50000 --lr 1e-4 --batch-size 32 --epochs 7 --warmup-ratio 0.05 --eval-every 100 --full-eval-every 500 "
       "--save-every 500 --seed 1 --grad-ckpt 0 --stop-step 7000 --adapter-dtype fp32 --save-adapter-step0 "
       "--no-checkpoints --out " + os.path.join(runs, "blockmix_filter_p100_tau20_s1")),
   "d: reader-mode train command unchanged")
open(os.path.join(runs, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
job = subprocess.run(["bash", FS, "1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "0.85"], env=env,
                     capture_output=True, text=True, timeout=300)
mk = os.path.join(runs, "blockmix", "markers")
K = "blockmix_filter_p100_rand85_s1"
recs = open(os.path.join(runs, "rec.txt")).read().splitlines() if os.path.exists(os.path.join(runs, "rec.txt")) else []
ok(job.returncode == 0 and all(os.path.exists(os.path.join(mk, f"{K}{s}.DONE")) for s in ("", "_checks", "_summary", "_job"))
   and [r.split()[0] for r in recs] == ["train", "check", "summary"] and recs[1] == f"check {os.path.join(runs, K)}",
   f"d: random job DONE: training, CPU check only (no re-read, no pyes-check), summary {recs} {job.stdout[-300:]}")

print(f"\n{len(passed)} passed, {len(failed)} failed")
for x in failed:
    print("FAILED:", x)
sys.exit(1 if failed else 0)

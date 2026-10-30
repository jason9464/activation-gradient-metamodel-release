"""Tests of the batch-level pure mixing student (main_run/train_blockmix.py), its summary (main_run/blockmix_summary.py)
and its chain (scripts/blockmix/chain.sh).

CPU only:
  (a) Label stream: mean within 0.01 of p for p = 0.5, 0.25, 0.1, 0.05 over 20,000 steps; the same seed gives the same
      stream, another seed a different one; the dolphin steps of a smaller p are a subset of those of a larger p.
  (b) Tiny fake pools (dolphin 50, clean 70 rows, batch 8, 20,000 steps, p 0.3): every batch's dataset positions lie in
      its labelled pool (all steps); documents add up; each pool reads its permutations in order and reshuffles at its
      end (a straddling batch takes the old tail then the new head; the new permutation differs); check_stream passes
      and fails the purity check on a corrupted stream; the whole stream is deterministic for a seed (sha1).
  (c) stream.json: Context.setup_stream writes stream.json and stream_checks.json (all stream checks pass, read back and
      rebuilt equal); a changed label in the record is reported.
  (d) 60-step dry run of the sampler through torch DataLoader + accelerate prepare (CPU), set_epoch per epoch as the
      Trainer does, 25 steps per epoch: the 60 batches are the stream's, pure; without set_epoch it advances itself.
  (e) Trainer path: trl SFTTrainer subclass from make_blockmix_trainer_class, tiny bf16 Qwen3 + LoRA, Qwen3 tokenizer,
      20 dolphin + 20 clean real rows, batch 4, train_subliminal's sft_config, collator and stop callback, 12 of 15
      steps: every check passes (dataset; runtime: max steps 15, dataloader length 5, lr at step 2 = lr / warmup,
      12 batches matched, ended at 12); an independent capture of the trained input_ids equals the collated dataset
      rows of the stream's positions, each batch from its labelled pool; check_batch raises on a wrong batch;
      ``check`` subcommand accepts the run for 12 steps and rejects it for 60; building the Trainer releases the GPU
      hold; with a reference run ``check`` accepts an equal step-0 adapter and rejects one with a flipped bit or a
      missing one; the two real reference students' step-0 adapters are equal (the premise of that check).
  (f) Wrapper: refusals (--arm owl, --save-grads, --max-steps, --resume, p > 1, a stream shorter than the schedule);
      the patch wiring with a stand-in for train_subliminal.main (real data files, 40 rows per pool): settings carry
      "blockmix", stream.json appears inside load_rows (not before), load_rows returns the dolphin rows then the clean
      rows, trl.SFTTrainer is the subclass, --max-steps = ceil(n_rows / batch) x epochs reaches main, exit 6 without
      runtime checks; main asks for the GPU hold (--reserve-gpu-gib default 6; a no-op without CUDA); a corrupted
      stream (one clean document in a dolphin batch) makes main exit 3 with the purity check failed; the patched
      names are restored afterwards by the test. Runtime block against the real newset_dolphin_s1 (RUN.json and
      train_log.jsonl): the reference schedule passes 6/6; lr at step 2 off by 1e-6 or warmup 547 fails.
  (g) blockmix_summary: the onset rule (two consecutive 500-step points), dolphin documents for p = 1 (16-document
      epoch ends) and from stream.json, a summary over a temporary runs dir (md, json, figure); statuses: a finished
      run whose only point >= 0.1 is the last is "not reached (only the last point >= 0.1)" with no documents, a
      running one is "(pending)", a chain FAILED marker gives "failed at N steps" or "failed before the first
      evaluation", a run folder without evaluations "started, no evaluation yet"; figure: probe panel solid lines,
      sampled panel dashed lines, references with hollow markers.
  (h) Chain: CB_DRY=1 listing; a full chain with a stand-in python and nvidia-smi (GPUs 0 and 2 owned by an ended
      chain, 1 and 3 not): smoke, smoke_check, four runs only on GPUs 0 and 2, summary, chain DONE; a relaunch does
      nothing; no owner marker -> smoke and chain FAILED after the maximum wait; a live chain pid -> refused; logs go
      to BM_LOGS (the repository's main_run/logs is not touched); a run that failed and is fixed by a relaunch gets
      the summary made again (summary.DONE moved aside); GPUs announced by a running EM eval stage are not usable
      until its readings end (read_all.DONE newer than eval.RUNNING) and a stale EM log is ignored.

Run: CUDA_VISIBLE_DEVICES= python tests/test_blockmix.py
"""
import json, math, os, subprocess, sys, tempfile, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "main_run")
sys.path.insert(0, M)
import train_blockmix as bm  # noqa: E402
import train_subliminal as ts  # noqa: E402

passed, failed = [], []
TMP = os.environ.get("TMPDIR")


def ok(cond, name):
    (passed if cond else failed).append(name)
    print(("  pass: " if cond else "  FAIL: ") + name)


# ---------------------------------------------------------------- (a) label stream
print("(a) label stream")
for p in (0.5, 0.25, 0.1, 0.05):
    lab = bm.label_stream(p, 1, 20000)
    ok(abs(lab.mean() - p) <= 0.01, f"a: p {p}: mean {lab.mean():.4f} within 0.01 over 20,000 steps")
ok(np.array_equal(bm.label_stream(0.25, 1, 5000), bm.label_stream(0.25, 1, 5000)), "a: same seed, same stream")
ok(not np.array_equal(bm.label_stream(0.25, 1, 5000), bm.label_stream(0.25, 2, 5000)), "a: other seed, other stream")
l5, l50 = bm.label_stream(0.05, 1, 20000), bm.label_stream(0.5, 1, 20000)
ok(bool(np.all(l50[l5 == 1] == 1)), "a: dolphin steps of p 0.05 are a subset of those of p 0.5 (shared draws)")
ok(bm.label_stream(0.0, 1, 100).sum() == 0 and bm.label_stream(1.0, 1, 100).sum() == 100, "a: p 0 and p 1 are pure")

# ---------------------------------------------------------------- (b) tiny pools
print("(b) stream on tiny pools")
sizes = {"dolphin": 50, "clean": 70}
B, S, p = 8, 20000, 0.3
lab = bm.label_stream(p, 3, S)
st = bm.build_stream(lab, sizes, 1, B)
order = bm.dataset_order(1, 120)
pos = bm.dataset_positions(st, sizes, order)
src = (order[pos] >= 50).astype(int)  # 1 clean
ok(all((src[t] == (1 - lab[t])).all() for t in range(S)), f"b: all {S} batches pure, from the labelled pool")
nd, nc = int(lab.sum()) * B, int(S - lab.sum()) * B
ok(nd + nc == B * S and st["readers"]["dolphin"].pos == nd and st["readers"]["clean"].pos == nc,
   f"b: documents add up (dolphin {nd} + clean {nc} = {B} x {S})")
rd = st["readers"]["dolphin"]
mine = st["rows"][lab == 1].reshape(-1)
ok(len(rd.perms) >= 3 and np.array_equal(mine, np.concatenate(rd.perms)[:len(mine)]),
   f"b: dolphin pool read in order through {len(rd.perms)} permutations")
ok(all(sorted(q.tolist()) == list(range(50)) for q in rd.perms) and not np.array_equal(rd.perms[0], rd.perms[1]),
   "b: each pool epoch is a fresh permutation of the pool")
t1, j1 = rd.starts[1]
dsteps = np.flatnonzero(lab == 1)
k = int(np.flatnonzero(dsteps == t1)[0])
ok(j1 == 50 - 6 * B and np.array_equal(st["rows"][t1][:j1], rd.perms[0][6 * B:]) and
   np.array_equal(st["rows"][t1][j1:], rd.perms[1][:B - j1]) and k == 6,
   f"b: reshuffle at the pool end: step {t1} takes the last {j1} of permutation 0 and the first {B - j1} of 1")
ep = rd.epochs()
ok(ep[1]["first_step"] == t1 and ep[1]["offset_in_batch"] == j1 and ep[0]["sha1"] == bm.sha1_of(rd.perms[0]),
   "b: pool epochs recorded (start step, offset, sha1)")
ck = bm.check_stream(bm.Checks(), st, p, sizes, pos, order, B, 15000)
ok(ck.summary()["n_failed"] == 0 and ck.summary()["n_passed"] == 6, f"b: check_stream {ck.summary()['per_stage']}")
bad = pos.copy()
t_d = int(np.flatnonzero(lab == 1)[0])
bad[t_d, 0] = int(np.flatnonzero(order >= 50)[0])  # a clean document in a dolphin batch
ck2 = bm.check_stream(bm.Checks(), st, p, sizes, bad, order, B, 15000)
ok(ck2.failed() == ["stream:every_batch_from_its_labelled_pool"], f"b: corrupted batch caught ({ck2.failed()})")
ck3 = bm.check_stream(bm.Checks(), st, p, sizes, pos, order, B, 30000)
ok(ck3.failed() == ["stream:stream_covers_planned_steps"], "b: a stream shorter than the planned steps is caught")
st_b = bm.build_stream(bm.label_stream(p, 3, S), sizes, 1, B)
st_c = bm.build_stream(bm.label_stream(p, 3, S), sizes, 2, B)
ok(bm.sha1_of(st_b["rows"]) == bm.sha1_of(st["rows"]) and bm.sha1_of(st_c["rows"]) != bm.sha1_of(st["rows"]),
   "b: the stream is deterministic for its seeds and changes with the pool seed")
ok(not np.array_equal(st["readers"]["dolphin"].perms[0][:50], st["readers"]["clean"].perms[0][:50]),
   "b: the two pools follow different orders")

# ---------------------------------------------------------------- (c) stream.json
print("(c) stream.json")
td_c = tempfile.mkdtemp(dir=TMP)
fake_d = [dict(prompt=f"d{i}", completion=str(i), source=None) for i in range(50)]
fake_c = [dict(prompt=f"c{i}", completion=str(i), source=None) for i in range(70)]
ctx_c = bm.Context(td_c, 0.3, 3, 1, 20000, 8, 50, 200, 7, 150, 1e-4, "/x/clean.jsonl", 256, "public", log=print)
rec = ctx_c.setup_stream(fake_d, fake_c, {"dolphin": "/x/d.jsonl", "clean": "/x/clean.jsonl"})
sc = json.load(open(os.path.join(td_c, "stream_checks.json")))
back = json.load(open(os.path.join(td_c, "stream.json")))
ok(sc["n_failed"] == 0 and sc["per_stage"] == {"stream": "7/7"} and "p1_loss_comparison" in sc["notes"],
   f"c: stream_checks.json written, {sc['per_stage']}")
ok(back["labels"] == [int(x) for x in bm.label_stream(0.3, 3, 20000)] and len(back["pool_position"]) == 20000 and
   back["schedule"]["max_steps"] == 200 and back["counts"]["dolphin_steps_to_stop"] == int(back["labels"][:150].count(1))
   and back["pools"]["clean"]["combined_offset"] == 50, "c: labels, positions, schedule and counts in stream.json")
st2, _ = bm.rebuild_from_record(back)
ok(bm.compare_record(back, st2) == [], "c: stream.json rebuilt from its seeds is equal")
back2 = dict(back, labels=list(back["labels"]))
back2["labels"][5] = 1 - back2["labels"][5]
ok("labels" in bm.compare_record(back2, st2), "c: a changed label is reported")

# ---------------------------------------------------------------- (d) sampler dry run
print("(d) 60-step dry run of the sampler (DataLoader + accelerate)")
from torch.utils.data import DataLoader  # noqa: E402
from accelerate import Accelerator  # noqa: E402

lab_d = bm.label_stream(0.5, 1, 100)
st_d = bm.build_stream(lab_d, sizes, 1, B)
pos_d = bm.dataset_positions(st_d, sizes, order)
data = list(range(120))
acc = Accelerator(cpu=True)
smp = bm.BlockMixBatchSampler(pos_d, 25)
dl = acc.prepare(DataLoader(data, batch_sampler=smp, collate_fn=lambda xs: list(xs)))
got = []
for epoch in range(3):
    if hasattr(dl, "set_epoch"):
        dl.set_epoch(epoch)
    for b in dl:
        got.append(b)
        if len(got) == 60:
            break
    if len(got) == 60:
        break
ok(len(dl) == 25 and len(got) == 60 and all(got[t] == pos_d[t].tolist() for t in range(60)),
   "d: 60 batches across three epochs equal the stream's positions (len 25 per epoch)")
ok(all(len({int(order[i] >= 50) for i in got[t]}) == 1 and int(order[got[t][0]] >= 50) == 1 - lab_d[t]
       for t in range(60)), "d: every one of the 60 batches is pure and from its labelled pool")
smp2 = bm.BlockMixBatchSampler(pos_d, 25)
seq = [b for _ in range(3) for b in smp2][:60]
ok(all(seq[t] == pos_d[t].tolist() for t in range(60)), "d: without set_epoch the sampler advances by itself")
try:
    list(bm.BlockMixBatchSampler(pos_d[:30], 25).__iter__()) and None
    s3 = bm.BlockMixBatchSampler(pos_d[:30], 25); s3.set_epoch(1); list(s3)
    ok(False, "d: a stream that runs out raises")
except IndexError:
    ok(True, "d: a stream that runs out raises IndexError")

# ---------------------------------------------------------------- (e) trainer path
print("(e) trainer path (CPU, bf16 tiny Qwen3)")
from transformers import AutoTokenizer, PrinterCallback, Qwen3Config, Qwen3ForCausalLM, set_seed  # noqa: E402
from trl import SFTTrainer  # noqa: E402
from peft import LoraConfig, TaskType  # noqa: E402
import datasets  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
tok.padding_side = "right"
drows, _ = ts.load_rows(ts.DATA["dolphin"], 20)
crows, _ = ts.load_rows(ts.DATA["clean"], 20)
td_e = tempfile.mkdtemp(dir=TMP)
PLANNED, SPE, STOP_E = 15, 5, 12
ctx = bm.Context(td_e, 0.5, 1, 1, 20000, 4, 20, PLANNED, SPE, STOP_E, 1e-4, ts.DATA["clean"], 256, "public",
                 ref_run=None, log=print)
ctx.setup_stream(drows, crows, {"dolphin": ts.DATA["dolphin"], "clean": ts.DATA["clean"]})
ids, masks, _ = ts.tokenize_rows(tok, ctx.rows, 256, "public")
ds_order = datasets.Dataset.from_dict({"i": list(range(len(ctx.rows)))}).shuffle(seed=1)["i"]
ok(list(ds_order) == bm.dataset_order(1, 40).tolist(), "e: datasets shuffle order = dataset_order (as asserted in main)")
ds = datasets.Dataset.from_dict({"input_ids": [ids[i] for i in ds_order], "completion_mask": [masks[i] for i in ds_order]})
a_e = ts.parse_args(["--arm", "dolphin", "--out", td_e, "--n-rows", "20", "--batch-size", "4", "--epochs", "3",
                     "--warmup-ratio", "0.2", "--max-steps", str(PLANNED), "--stop-step", str(STOP_E), "--lr", "1e-4"])
cfg = ts.sft_config(a_e, td_e, use_cpu=True, save_strategy="no")
captured = []
Base = bm.make_blockmix_trainer_class(SFTTrainer, ctx, ts.tokenize_rows)


class Capture(Base):
    def training_step(self, model, inputs, *args, **kw):
        captured.append((self.state.global_step, inputs["input_ids"].detach().cpu().clone(),
                         inputs["attention_mask"].detach().cpu().clone()))
        return super().training_step(model, inputs, *args, **kw)


set_seed(1)
qcfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
torch.manual_seed(7)
model = Qwen3ForCausalLM(qcfg).to(torch.bfloat16)
pc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
                bias="none")
bm._HELD.append("stand-in for the GPU hold")
tr = Capture(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(tok), train_dataset=ds,
             processing_class=tok, peft_config=pc)
ok(not bm._HELD, "e: building the Trainer releases the GPU hold")
ok(bm.reserve_gpu(6) == 0 and not bm._HELD, "e: the GPU hold is a no-op without CUDA")
tr.remove_callback(PrinterCallback)
tr.add_callback(ts.make_stop_callback(a_e))
tr.train()
sce = json.load(open(os.path.join(td_e, "stream_checks.json")))
ok(ctx.runtime_ok is True and sce["n_failed"] == 0 and set(sce["per_stage"]) == {"stream", "dataset", "runtime"},
   f"e: all checks pass {sce['per_stage']}")
rt = {c["name"]: c["detail"] for c in sce["stages"]["runtime"]}
warm = cfg.get_warmup_steps(PLANNED)
ok(rt["trainer_max_steps_equals_planned"]["trainer"] == PLANNED and
   rt["dataloader_length_equals_steps_per_epoch"]["dataloader"] == SPE,
   f"e: Trainer max steps {PLANNED} from --max-steps, dataloader length {SPE} per epoch")
ok(abs(rt["lr_step2_equals_lr_over_warmup_and_reference"]["lr_step2"] - 1e-4 / warm) <= 1e-15 and warm >= 2,
   f"e: lr at step 2 = lr / warmup ({warm})")
ok(rt["every_trained_batch_is_the_stream_batch"]["checked"] == STOP_E and len(captured) == STOP_E and
   [c[0] for c in captured] == list(range(STOP_E)), f"e: {STOP_E} trained batches checked, steps 0..{STOP_E - 1}")
coll = ts.CompletionMaskCollator(tok)
same = 0
for t, x_ids, x_am in captured:
    exp = coll([ds[int(q)] for q in ctx.positions[t]])
    pool_ok = len({int(ctx.order[q] >= 20) for q in ctx.positions[t]}) == 1 and \
        int(ctx.order[ctx.positions[t][0]] >= 20) == 1 - ctx.st["labels"][t]
    same += bool(torch.equal(exp["input_ids"], x_ids) and torch.equal(exp["attention_mask"], x_am) and pool_ok)
ok(same == STOP_E, f"e: independent capture: {same}/{STOP_E} trained batches equal the collated stream rows, pure")
ok(sce["stages"]["dataset"][1]["detail"]["checked"] == 20 and sce["stages"]["dataset"][2]["detail"]["equal"] == 20,
   "e: dataset check retokenised every row of both tiny pools")
wrong = coll([ds[int(q)] for q in ctx.positions[1]])
try:
    ctx.check_batch(0, wrong)
    ok(False, "e: check_batch raises on a wrong batch")
except RuntimeError:
    ok(True, "e: check_batch raises on a wrong batch")
ok(bm.check_run(td_e, STOP_E) == 1, "e: check subcommand after a mismatch recorded: rejected")
# rerun the runtime block without the injected mismatch for the check subcommand
ctx.runtime["mismatches"] = []
ctx.runtime["batches_checked"] = STOP_E
ctx.finish_runtime(STOP_E)
ok(bm.check_run(td_e, STOP_E) == 0 and bm.check_run(td_e, 60) == 1, "e: check subcommand: 12 steps ok, 60 rejected")
from safetensors.torch import save_file  # noqa: E402

ad_e, ref_e = os.path.join(td_e, "adapters", "step_000000"), os.path.join(td_e + "_ref", "adapters", "step_000000")
os.makedirs(ad_e)
os.makedirs(ref_e)
tens = {"m.lora_A.weight": torch.randn(4, 8), "m.lora_B.weight": torch.zeros(8, 4)}
save_file(tens, os.path.join(ad_e, "adapter_model.safetensors"))
save_file({k_: v.clone() for k_, v in tens.items()}, os.path.join(ref_e, "adapter_model.safetensors"))
ok(bm.check_run(td_e, STOP_E, td_e + "_ref") == 0, "e: check with a reference: equal step-0 adapter accepted")
t_flip = {k_: v.clone() for k_, v in tens.items()}
t_flip["m.lora_A.weight"].view(torch.int32)[0, 0] ^= 1  # one bit of one float
save_file(t_flip, os.path.join(ref_e, "adapter_model.safetensors"))
r_flip = bm.compare_step0_adapter(td_e, td_e + "_ref")
ok(bm.check_run(td_e, STOP_E, td_e + "_ref") == 1 and r_flip["differing"] == ["m.lora_A.weight"],
   "e: check with a reference: a step-0 adapter one bit apart is rejected")
ok(bm.check_run(td_e, STOP_E, os.path.join(td_e, "no_such_run")) == 1, "e: check with a reference: missing adapter rejected")
REF_CLEAN = os.path.join(os.path.dirname(bm.REF_RUN), "newset_clean_s1")
r_ref = bm.compare_step0_adapter(REF_CLEAN, bm.REF_RUN)
ok(r_ref["equal"] and r_ref["tensors"] == 560, f"e: newset_clean_s1 and newset_dolphin_s1 step-0 adapters are equal "
                                               f"bit for bit ({r_ref.get('tensors')} tensors)")

# ---------------------------------------------------------------- (f) wrapper
print("(f) wrapper refusals and patch wiring")
import trl  # noqa: E402

base_args = ["--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs", "7", "--stop-step", "30"]
for extra, frag, name in [(["--arm", "owl"], "--arm owl", "an owl arm"), (["--save-grads"], "--save-grads", "--save-grads"),
                          (["--max-steps", "100"], "--max-steps", "--max-steps"), (["--resume"], "--resume", "--resume"),
                          (["--save-adamfx"], "--save-adamfx", "--save-adamfx")]:
    try:
        bm.main(["--dolphin-frac", "0.5", "--stream-len", "20000"] + base_args + extra + ["--out", "/nonexistent/x"])
        ok(False, f"f: refuses {name}")
    except SystemExit as e:
        ok(frag in str(e.code), f"f: refuses {name}")
for argv, frag, name in [(["--dolphin-frac", "1.5", "--stream-len", "20000"], "[0, 1]", "p > 1"),
                         (["--dolphin-frac", "0.5", "--stream-len", "20"], "--stream-len", "a stream shorter than 35")]:
    try:
        bm.main(argv + base_args + ["--out", "/nonexistent/x"])
        ok(False, f"f: refuses {name}")
    except SystemExit as e:
        ok(frag in str(e.code), f"f: refuses {name}")
saved = (ts.main, ts.load_rows, ts.settings_of, trl.SFTTrainer)
seen = {}
td_f = os.path.join(tempfile.mkdtemp(dir=TMP), "run")


def fake_main(argv):
    a = ts.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    seen["max_steps"] = a.max_steps
    seen["blockmix"] = ts.settings_of(a, ts.DATA["dolphin"]).get("blockmix")
    seen["stream_before"] = os.path.exists(os.path.join(out, "stream.json"))
    rows, stats = ts.load_rows(ts.DATA["dolphin"], a.n_rows)
    seen["rows"], seen["stats"] = rows, stats
    seen["stream_after"] = os.path.exists(os.path.join(out, "stream.json"))
    from trl import SFTTrainer as T
    seen["trainer"] = T


held_calls = []
orig_reserve = bm.reserve_gpu
try:
    ts.main = fake_main
    bm.reserve_gpu = lambda gib=bm.RESERVE_GIB, log=None: held_calls.append(gib) or 0
    rc = bm.main(["--dolphin-frac", "0.25", "--stream-len", "20000"] + base_args + ["--out", td_f])
finally:
    ts.main, ts.load_rows, ts.settings_of, trl.SFTTrainer = saved
    bm.reserve_gpu = orig_reserve
ok(held_calls[:1] == [6.0], f"f: main asks for the 6 GiB GPU hold first ({held_calls})")
d40, _ = ts.load_rows(ts.DATA["dolphin"], 40)
c40, _ = ts.load_rows(ts.DATA["clean"], 40)
ok(rc == 6, "f: exit 6 when no runtime checks ran")
ok(seen["max_steps"] == 35, f"f: --max-steps {seen['max_steps']} = ceil(40 / 8) x 7 reaches train_subliminal.main")
ok(seen["blockmix"]["dolphin_frac"] == 0.25 and seen["blockmix"]["steps_per_epoch"] == 5, "f: settings carry blockmix")
ok(not seen["stream_before"] and seen["stream_after"], "f: stream.json is written inside load_rows (before training)")
ok(seen["rows"] == d40 + c40 and seen["stats"]["lines"] == 80 and
   seen["stats"]["blockmix_pools"]["clean"]["rows_sha1"] == ts.row_stats(c40)["rows_sha1"],
   "f: load_rows returns the 40 dolphin rows then the 40 clean rows (pool stats recorded)")
ok(seen["trainer"].__name__ == "BlockMixTrainer" and issubclass(seen["trainer"], saved[3]),
   "f: trl.SFTTrainer seen by main is the block-mix subclass")
scf = json.load(open(os.path.join(td_f, "stream_checks.json")))
ok(scf["per_stage"] == {"stream": "7/7"}, f"f: stream checks on the real pools {scf['per_stage']}")
orig_dp = bm.dataset_positions


def bad_positions(st_, sizes_, order_):
    q = orig_dp(st_, sizes_, order_).copy()
    lab_ = st_["labels"]
    td_, tc_ = int(np.flatnonzero(lab_ == 1)[0]), int(np.flatnonzero(lab_ == 0)[0])
    q[[td_, tc_], 0] = q[[tc_, td_], 0]  # swap one document between a dolphin and a clean batch
    return q


td_f3 = os.path.join(tempfile.mkdtemp(dir=TMP), "run")
code3 = None
try:
    ts.main, bm.dataset_positions = fake_main, bad_positions
    try:
        bm.main(["--dolphin-frac", "0.25", "--stream-len", "20000"] + base_args + ["--out", td_f3])
        code3 = 0
    except SystemExit as e:
        code3 = e.code
finally:
    ts.main, ts.load_rows, ts.settings_of, trl.SFTTrainer = saved
    bm.dataset_positions = orig_dp
sc3 = json.load(open(os.path.join(td_f3, "stream_checks.json")))
ok(code3 == 3 and "stream:every_batch_from_its_labelled_pool" in sc3["failed"],
   f"f: a corrupted stream stops main with exit 3 (exit {code3}, failed {sc3['failed']})")
ok(ts.load_rows is saved[1] and trl.SFTTrainer is saved[3], "f: test restored the patched names")
td_j = tempfile.mkdtemp(dir=TMP)
cj = bm.Context(td_j, 0.5, 1, 1, 20000, 32, 50000, 10941, 1563, 7000, 1e-4, ts.DATA["clean"], 256, "public",
                ref_run=bm.REF_RUN, log=print)
cj.runtime.update(max_steps=10941, len_dataloader=1563, warmup=548, lr_step2=1e-4 / 548, batches_checked=7000)
ok(cj.finish_runtime(7000) and cj.ck.summary()["per_stage"]["runtime"] == "6/6" and
   cj.runtime["reference"]["planned"] == 10941 and cj.runtime["reference"]["warmup"] == 548 and
   cj.runtime["reference"]["spe"] == 1563 and cj.runtime["reference"]["lr2"] is not None,
   f"f: runtime block against the real newset_dolphin_s1: {cj.ck.summary()['per_stage']}, reference "
   f"{ {k_: v for k_, v in cj.runtime['reference'].items() if k_ != 'run'} }")
cj.runtime["lr_step2"] = 1e-4 / 548 * (1 + 1e-6)
ok(not cj.finish_runtime(7000) and cj.ck.failed("runtime") == ["runtime:lr_step2_equals_lr_over_warmup_and_reference"],
   "f: lr at step 2 off by 1e-6 (relative) fails the runtime block")
cj.runtime.update(lr_step2=1e-4 / 548, warmup=547)
ok(not cj.finish_runtime(7000) and "runtime:warmup_equals_reference" in cj.ck.failed("runtime"),
   "f: warmup 547 fails the runtime block")

# ---------------------------------------------------------------- (g) summary
print("(g) blockmix_summary")
import blockmix_summary as bsum  # noqa: E402

ok(bsum.first_twice([(0, 0), (500, 0.2), (1000, 0.05), (1500, 0.15), (2000, 0.12)]) == (1500, "learned"),
   "g: onset needs two consecutive 500-step points >= 0.1")
ok(bsum.first_twice([(0, 0), (500, 0.2)]) == (500, "pending") and bsum.first_twice([(0, 0), (500, 0.05)]) ==
   (None, "not yet"), "g: pending at the last point; not yet")
td_g = tempfile.mkdtemp(dir=TMP)
os.makedirs(os.path.join(td_g, "ref1"))
json.dump(dict(settings=dict(n_rows=50000, batch_size=32), steps_per_epoch=1563),
          open(os.path.join(td_g, "ref1", "RUN.json"), "w"))
f1 = bsum.docs_fn(os.path.join(td_g, "ref1"), 1.0)
ok(f1(3000) == 95984 and f1(1563) == 50000 and f1(7000) == 223936, "g: p = 1 documents (16-document epoch ends)")
fs = bsum.docs_fn(td_c, 0.3)
ok(fs(150) == 8 * sum(back["labels"][:150]), "g: sweep documents from stream.json")
run = os.path.join(td_g, "blockmix_p050_s1")
os.makedirs(run)
with open(os.path.join(run, "evals.jsonl"), "w") as f:
    for s_, v in [(0, 0.002), (500, 0.05), (1000, 0.12), (1500, 0.2)]:
        f.write(json.dumps(dict(step=s_, kind="logprob", result=dict(dolphin={"Dolphin": dict(logprob_sum=math.log(v))}))) + "\n")
        f.write(json.dumps(dict(step=s_, kind="full", result=dict(positive=dict(rate=dict(dolphin=v / 2))))) + "\n")
with open(os.path.join(run, "train_log.jsonl"), "w") as f:
    f.write(json.dumps(dict(step=1500)) + "\n")
import shutil  # noqa: E402

shutil.copy(os.path.join(td_c, "stream.json"), os.path.join(run, "stream.json"))
out = subprocess.run([sys.executable, os.path.join(M, "blockmix_summary.py"), "--runs-dir", td_g, "--fig",
                      os.path.join(td_g, "fig.png")], capture_output=True, text=True)
md = open(os.path.join(td_g, "blockmix_summary.md")).read() if out.returncode == 0 else ""
js = json.load(open(os.path.join(td_g, "blockmix_summary.json"))) if out.returncode == 0 else {}
row = [r for r in js.get("rows", []) if r["run"] == "blockmix_p050_s1"]
ok(out.returncode == 0 and os.path.getsize(os.path.join(td_g, "fig.png")) > 10000 and
   "| 0.5 | blockmix_p050_s1 | stopped at 1,500 steps | 1,000 |" in md and row and row[0]["value_step"] == 1500
   and abs(row[0]["probe_at_value_step"] - 0.2) < 1e-9 and "| 1 | newset_dolphin_s1 | not started |" in md,
   f"g: summary over a partial runs dir (md, json, figure) {out.stderr[-300:]}")



def write_run(run_dir, points, last_step, pid=None):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "evals.jsonl"), "w") as f:
        for s_, v in points:
            f.write(json.dumps(dict(step=s_, kind="logprob", result=dict(dolphin={"Dolphin": dict(logprob_sum=math.log(v))}))) + "\n")
            f.write(json.dumps(dict(step=s_, kind="full", result=dict(positive=dict(rate=dict(dolphin=v / 2))))) + "\n")
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as f:
        f.write(json.dumps(dict(step=last_step)) + "\n")
    if pid is not None:
        with open(run_dir + ".pid", "w") as f:
            f.write(str(pid))


td_k = tempfile.mkdtemp(dir=TMP)
mk_k = os.path.join(td_k, "blockmix", "markers")
os.makedirs(mk_k)
low = [(s_, 0.02) for s_ in range(0, 7000, 500)]
write_run(os.path.join(td_k, "blockmix_p050_s1"), low + [(7000, 0.15)], 7000)  # finished, only the last point >= 0.1
write_run(os.path.join(td_k, "blockmix_p025_s1"), [(0, 0.01), (500, 0.02), (1000, 0.03)], 1000)
open(os.path.join(mk_k, "blockmix_p025_s1.FAILED"), "w").close()
os.makedirs(os.path.join(td_k, "blockmix_p010_s1"))
open(os.path.join(mk_k, "blockmix_p010_s1.FAILED"), "w").close()
write_run(os.path.join(td_k, "blockmix_p005_s1"), [(0, 0.01), (500, 0.02), (1000, 0.2)], 1000, pid=os.getpid())
os.makedirs(os.path.join(td_k, "newset_clean_s1"))
rows_k = {r["run"]: r for r in bsum.summarize(td_k, 7000)}
bsum.write_md(list(rows_k.values()), os.path.join(td_k, "s.md"), 7000)
md_k = open(os.path.join(td_k, "s.md")).read()
r50 = rows_k["blockmix_p050_s1"]
ok(r50["onset_step"] is None and r50["docs_at_onset"] is None and r50["only_last_point_step"] == 7000 and
   "| 0.5 | blockmix_p050_s1 | finished at 7,000 | not reached (only the last point >= 0.1) | - |" in md_k,
   "g: finished run with only the last point >= 0.1: not reached, no documents")
ok("| 0.05 | blockmix_p005_s1 | running, 1,000 steps | 1,000 (pending) |" in md_k,
   "g: a running run whose last point is >= 0.1 stays pending")
ok("| 0.25 | blockmix_p025_s1 | failed at 1,000 steps (chain marker) |" in md_k and
   "| 0.1 | blockmix_p010_s1 | failed before the first evaluation |" in md_k,
   "g: chain FAILED markers: failed at N steps / failed before the first evaluation")
ok("| 0 | newset_clean_s1 | started, no evaluation yet |" in md_k and
   "| 1 | newset_dolphin_s1 | not started |" in md_k, "g: run folder without evaluations: started; nothing: not started")
figs_k = []
orig_close = bsum.plt.close
try:
    bsum.plt.close = lambda f=None: figs_k.append(f)
    bsum.plot(list(rows_k.values()), os.path.join(td_k, "f.png"), 7000)
finally:
    bsum.plt.close = orig_close
ax_p, ax_s = figs_k[0].axes[:2]
data_p = [l_ for l_ in ax_p.lines if l_.get_label().startswith("p = ")]
data_s = [l_ for l_ in ax_s.lines if l_.get_label().startswith("p = ")]
ok(data_p and data_s and {l_.get_linestyle() for l_ in data_p} == {"-"} and
   {l_.get_linestyle() for l_ in data_s} == {"--"},
   "g: figure: probe panel solid lines, sampled panel dashed lines (project rule)")
bsum.plt.close(figs_k[0])
write_run(os.path.join(td_k, "newset_dolphin_s1"), [(0, 0.01), (500, 0.2), (1000, 0.3)], 7000)
rows_k2 = bsum.summarize(td_k, 7000)
figs_k = []
try:
    bsum.plt.close = lambda f=None: figs_k.append(f)
    bsum.plot(rows_k2, os.path.join(td_k, "f2.png"), 7000)
finally:
    bsum.plt.close = orig_close
refl = [l_ for l_ in figs_k[0].axes[0].lines if l_.get_label().startswith("p = 1 ")]
ok(refl and refl[0].get_linestyle() == "-" and refl[0].get_markerfacecolor() == "white",
   "g: figure: the p = 1 reference is solid in the probe panel with hollow markers")
bsum.plt.close(figs_k[0])

# ---------------------------------------------------------------- (h) chain
print("(h) chain with a stand-in python and nvidia-smi")
CH = os.path.join(ROOT, "scripts", "blockmix", "chain.sh")
td_h = tempfile.mkdtemp(dir=TMP)
stub_py = os.path.join(td_h, "py.sh")
with open(stub_py, "w") as f:
    f.write("""#!/bin/bash
case "$1" in
  train_blockmix.py)
    if [ "$2" = check ]; then echo '{"ok": true}'; exit 0; fi
    stop=""; out=""; frac=""
    while [ $# -gt 0 ]; do case "$1" in --stop-step) stop=$2; shift;; --out) out=$2; shift;; --dolphin-frac) frac=$2; shift;; esac; shift; done
    mkdir -p "$out"; echo "gpu=$CUDA_VISIBLE_DEVICES frac=$frac start=$(date +%s.%N)" > "$out/stub.txt"
    if [ -n "${STUB_FAIL_FRAC:-}" ] && [ "$frac" = "$STUB_FAIL_FRAC" ] && [ ! -e "$STUB_FAIL_FLAG" ]; then
      touch "$STUB_FAIL_FLAG"; echo "stub: injected failure"; exit 7
    fi
    sleep ${STUB_SLEEP:-1}
    echo "end=$(date +%s.%N)" >> "$out/stub.txt"
    echo "2026-10-26 blockmix: runtime checks 9 passed, 0 failed"
    echo "2026-10-26 done: step $stop, wall 1.0s this segment"; exit 0;;
  main_run/blockmix_summary.py) echo summary; [ -n "${STUB_SUM_LOG:-}" ] && echo "summary" >> "$STUB_SUM_LOG"; exit 0;;
  *) echo "stub: unknown $*" >&2; exit 9;;
esac
""")
smi = os.path.join(td_h, "smi.sh")
with open(smi, "w") as f:
    f.write('#!/bin/bash\nprintf "0, 100\\n1, 200\\n2, 300\\n3, 100\\n"\n')
os.chmod(stub_py, 0o755)
os.chmod(smi, 0o755)


def chain_env(runs, **kw):
    env = dict(os.environ, BM_PY="bash " + stub_py, BM_SMI="bash " + smi,  # bash: TMPDIR may be noexec
               BM_RUNS=runs, BM_LOGS=os.path.join(runs, "logs"), BM_EM_LOGS=os.path.join(runs, "em_logs"),
               BM_POLL_S="1", BM_LOOK_S="0", BM_MAX_WAIT_S="60", BM_MIN_FREE_GB="1", BM_STAMP="T",
               BM_VAR_MARKERS=os.path.join(runs, "var_markers"), BM_EM_MARKERS=os.path.join(runs, "em_markers"),
               STUB_SLEEP="5")
    env.update({k: str(v) for k, v in kw.items()})
    return env


import glob  # noqa: E402

REPO_T_LOGS = os.path.join(ROOT, "main_run", "logs", "T_blockmix_*")
t_logs_before = {f_: os.stat(f_).st_mtime_ns for f_ in glob.glob(REPO_T_LOGS)}
runs = os.path.join(td_h, "runs")
os.makedirs(os.path.join(runs, "var_markers"))
os.makedirs(os.path.join(runs, "em_markers"))
dry = subprocess.run(["bash", CH], env=chain_env(runs, CB_DRY=1), capture_output=True, text=True)
ok(dry.returncode == 0 and "nothing launched" in dry.stdout and "smoke [pending]" in dry.stdout and
   "--dolphin-frac 0.05" in dry.stdout and "usable GPUs now (one look): []" in dry.stdout and
   not os.path.exists(os.path.join(runs, "blockmix")), "h: CB_DRY=1 lists stages, commands and the GPU rule, starts "
                                                       "nothing, creates nothing")
open(os.path.join(runs, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
t0 = time.time()
r1 = subprocess.run(["bash", CH], env=chain_env(runs), capture_output=True, text=True, timeout=300)
mk = os.path.join(runs, "blockmix", "markers")
keys = ["blockmix_p050_s1", "blockmix_p025_s1", "blockmix_p010_s1", "blockmix_p005_s1"]
gpus = {k: open(os.path.join(mk, k + ".gpu")).read().strip() for k in keys if os.path.exists(os.path.join(mk, k + ".gpu"))}
ok(r1.returncode == 0 and os.path.exists(os.path.join(mk, "blockmix_chain.DONE")) and
   all(os.path.exists(os.path.join(mk, k + ".DONE")) for k in ["smoke", "smoke_check", "summary"] + keys),
   f"h: chain DONE with every stage DONE ({time.time() - t0:.0f} s) {r1.stdout[-400:] if r1.returncode else ''}")
ok(set(gpus.values()) <= {"0", "2"} and len(gpus) == 4 and open(os.path.join(mk, "smoke.gpu")).read().strip() in "02",
   f"h: only GPUs whose owner ended are used ({gpus})")
iv = []
for k in keys:
    txt = open(os.path.join(runs, k, "stub.txt")).read().split()
    d_ = dict(x.split("=") for x in txt)
    iv.append((float(d_["start"]), float(d_["end"]), d_["gpu"]))
over = [(a_, b_) for i, a_ in enumerate(iv) for b_ in iv[i + 1:] if a_[2] == b_[2] and a_[0] < b_[1] and b_[0] < a_[1]]
ok(not over and any(a_[0] < b_[1] and b_[0] < a_[1] for i, a_ in enumerate(iv) for b_ in iv[i + 1:]),
   "h: runs share no GPU at the same time; two GPUs run in parallel")
ok(open(os.path.join(runs, "blockmix_p025_s1", "stub.txt")).read().startswith("gpu=") and
   "frac=0.25" in open(os.path.join(runs, "blockmix_p025_s1", "stub.txt")).read(), "h: each run gets its p")
r2 = subprocess.run(["bash", CH], env=chain_env(runs), capture_output=True, text=True, timeout=60)
ok(r2.returncode == 0 and "nothing to do" in r2.stdout, "h: relaunch after DONE does nothing")
runs2 = os.path.join(td_h, "runs2")
os.makedirs(os.path.join(runs2, "var_markers"))
os.makedirs(os.path.join(runs2, "em_markers"))
r3 = subprocess.run(["bash", CH], env=chain_env(runs2, BM_MAX_WAIT_S=1), capture_output=True, text=True, timeout=60)
mk2 = os.path.join(runs2, "blockmix", "markers")
ok(r3.returncode == 1 and os.path.exists(os.path.join(mk2, "smoke.FAILED")) and
   os.path.exists(os.path.join(mk2, "blockmix_chain.FAILED")) and not os.path.exists(os.path.join(runs2, "blockmix", "smoke")),
   "h: no owner marker: smoke FAILED after the maximum wait, chain FAILED, nothing trained")
runs3 = os.path.join(td_h, "runs3")
os.makedirs(os.path.join(runs3, "blockmix", "markers"))
with open(os.path.join(runs3, "blockmix", "markers", "blockmix_chain.pid"), "w") as f:
    f.write(str(os.getpid()))
r4 = subprocess.run(["bash", CH], env=chain_env(runs3), capture_output=True, text=True, timeout=60)
ok(r4.returncode == 1 and "refuse: chain pid" in r4.stdout, "h: a live chain pid is refused")
ok(os.path.exists(os.path.join(runs, "logs", "T_blockmix_smoke.out")) and
   os.path.exists(os.path.join(runs, "logs", "T_blockmix_blockmix_p005_s1.out")), "h: run logs go to BM_LOGS")
runs4 = os.path.join(td_h, "runs4")
os.makedirs(os.path.join(runs4, "var_markers"))
os.makedirs(os.path.join(runs4, "em_markers"))
open(os.path.join(runs4, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
env4 = chain_env(runs4, STUB_SLEEP=1, STUB_FAIL_FRAC="0.05", STUB_FAIL_FLAG=os.path.join(runs4, "failed_once"),
                 STUB_SUM_LOG=os.path.join(runs4, "summary_calls.txt"))
r5 = subprocess.run(["bash", CH], env=env4, capture_output=True, text=True, timeout=300)
mk4 = os.path.join(runs4, "blockmix", "markers")
n_sum1 = len(open(os.path.join(runs4, "summary_calls.txt")).read().split())
ok(r5.returncode == 1 and os.path.exists(os.path.join(mk4, "blockmix_p005_s1.FAILED")) and
   os.path.exists(os.path.join(mk4, "summary.DONE")) and os.path.exists(os.path.join(mk4, "blockmix_chain.FAILED"))
   and n_sum1 == 1, f"h: a failed run: summary made, chain FAILED ({r5.stdout[-300:] if r5.returncode != 1 else ''})")
r6 = subprocess.run(["bash", CH], env=env4, capture_output=True, text=True, timeout=300)
n_sum2 = len(open(os.path.join(runs4, "summary_calls.txt")).read().split())
ok(r6.returncode == 0 and os.path.exists(os.path.join(mk4, "blockmix_p005_s1.DONE")) and
   os.path.exists(os.path.join(mk4, "blockmix_chain.DONE")) and n_sum2 == 2 and
   glob.glob(os.path.join(mk4, "summary.DONE.before_*")),
   f"h: relaunch fixes the run and makes the summary again (summary runs {n_sum1} then {n_sum2})")
runs5 = os.path.join(td_h, "runs5")
for d_ in ("var_markers", "em_markers", "em_logs"):
    os.makedirs(os.path.join(runs5, d_))
open(os.path.join(runs5, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
em_run = os.path.join(runs5, "em_markers", "eval.RUNNING")
with open(em_run, "w") as f:
    f.write("pid 1 host x start 2026-10-26 17:37:37 KST\n")
em_log = os.path.join(runs5, "em_logs", "20261026_1349_em_reader_eval.out")
with open(em_log, "w") as f:
    f.write("[2026-10-26 17:37:37 KST] waiting for the background old-reader part (pid 1067080)\n")


def dry_usable(runs_):
    out_ = subprocess.run(["bash", CH], env=chain_env(runs_, CB_DRY=1), capture_output=True, text=True).stdout
    line = [x for x in out_.splitlines() if "usable GPUs now (one look):" in x]
    return line[0].split("usable GPUs now (one look):")[1].strip() if line else None, out_


u_wait, _ = dry_usable(runs5)
with open(em_log, "a") as f:
    f.write("[2026-10-26 19:20:01 KST] readings (both readers; existing outputs skipped) on GPUs 0,2\n")
u_ann, out_ann = dry_usable(runs5)
ok(u_wait == "[0 2 ]" and u_ann == "[]" and "GPU 0: owner" in out_ann and "announced by the EM eval stage: yes" in out_ann,
   f"h: GPUs announced by the running EM eval stage are not usable (before the announcement {u_wait}, after {u_ann})")
t_now = time.time()
os.utime(em_run, (t_now - 50, t_now - 50))
os.utime(em_log, (t_now - 40, t_now - 40))
with open(os.path.join(runs5, "em_markers", "read_all.DONE"), "w") as f:
    f.write("done\n")
u_done, _ = dry_usable(runs5)
os.utime(os.path.join(runs5, "em_markers", "read_all.DONE"), (t_now - 200, t_now - 200))
u_old_done, _ = dry_usable(runs5)
os.utime(em_log, (t_now - 100, t_now - 100))
u_stale, _ = dry_usable(runs5)
ok(u_done == "[0 2 ]" and u_old_done == "[]" and u_stale == "[0 2 ]",
   f"h: usable again once the EM readings ended ({u_done}); a read_all.DONE older than eval.RUNNING does not count "
   f"({u_old_done}); an EM eval log older than eval.RUNNING is ignored ({u_stale})")
t_logs_after = {f_: os.stat(f_).st_mtime_ns for f_ in glob.glob(REPO_T_LOGS)}
ok(t_logs_after == t_logs_before, "h: the repository's main_run/logs was not written by the chain tests")

print(f"\n{len(passed)} passed, {len(failed)} failed")
for n in failed:
    print("FAILED:", n)
sys.exit(1 if failed else 0)

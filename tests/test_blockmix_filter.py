"""Tests of the filtering student (main_run/train_blockmix_filter.py), its rows in main_run/blockmix_summary.py and its
launcher scripts/blockmix/filter.sh. Plan docs/notes/20261026_blockmix_filter_plan.md, stage 2.

CPU only:
  (a) Skip wrappers on a toy model with torch AdamW and the cosine warmup schedule: on skipped steps parameters,
      exp_avg, exp_avg_sq, Adam's step count and the lr are bit-equal before and after; every non-skipped step is
      bit-identical to an unwrapped optimizer and scheduler that only see the accepted steps' gradients; attribute
      access and assignment reach the wrapped objects; the scheduler counts accepted steps only.
  (b) Would-be update formulas against torch AdamW on random states (fp64 and fp32, Adam step counts 1, 7, 250):
      mom = -(lr / bc1) times torch's exp_avg after a real step; mfree = bc1 times the change of a torch step whose
      state is set so that its moments after the update are (g, v_{t-1}) (gradient 0, exp_avg g / beta1, exp_avg_sq
      v / beta2); t = state step + 1.
  (c) Direct-write term on a tiny fp32 Qwen3 + LoRA (three AdamW steps first so B != 0), mfree and mom: the captured X
      equals an independent token sum over positions 0..n-2 of each row; D equals (i) Filter.brute_force (dense
      s (B1 A1 - B0 A0) X) and (ii) the summed output change of fp64 copies of the real peft LoRA modules with the old
      and the new factors over the same tokens, within 1e-6 per layer.
  (d) In-process reader code path on the worktree's tiny random Qwen3 (CPU): the module resolves inside the worktree
      while train_subliminal stays this repository's; P(Yes) equals an independent copy of cleanbatch_pyes.py's
      prefill on the same input; a different input changes it (the injection is live).
  (e) Trainer path: trl SFTTrainer through make_filter_trainer_class, tiny bf16 Qwen3 + LoRA, 20 + 20 real rows,
      batch 4, a stub reader (dolphin batches mostly high P(Yes), some low; a few clean batches high), tau 0.5:
      every stage of stream_checks.json passes; training ends when 9 updates were accepted; step 1 accepted unread;
      rejected steps leave parameters, both moments and the lr bit-equal (independent snapshots); a replay of the
      accepted steps' clipped gradients through a fresh AdamW + cosine schedule gives bit-identical parameters and
      moments; lr at step 2 = lr / warmup; the lr after a rejected step is unchanged; the filter log has one record per
      drawn step with the stream's labels; check steps 2 and 5 pass the dense check; filter_check_inputs.npz loads in
      the worktree loader; ``check`` accepts the run; ``pyes-check`` accepts a re-read within 1e-3 and rejects 1e-2.
      A second run whose reader rejects everything ends at the drawn budget with one accepted update. ``check
      --min-rejected K`` passes with K = the rejected steps and fails with one more.
  (f) Wrapper: refusals (grad checkpointing, tau out of range, stream shorter than the drawn budget, accepted budget
      above the drawn budget, no --stop-step, owl arm, --max-steps, an unknown filter input); patch wiring with a
      stand-in train_subliminal.main and a stub reader: --max-steps = planned and --stop-step = drawn budget reach main,
      settings carry blockmix.filter, the trainer class is the filter subclass, stop_at returns the budget stop step,
      evals.jsonl records get filter_accepted_steps and train_log records do not, exit 6 without runtime checks, the
      patched names are restored; settings record the formula modules' sha1 (= the files' sha1 and git hash-object)
      and function hashes and a passed AdamW self-check; a changed mfree formula ends main before the reader loads.
  (g) blockmix_summary over a temporary runs dir with a filtered run: precision, recall, leaked dolphin, dropped clean,
      first rejected dolphin, accepted updates at the probe points, held below 0.1, md table and figure; a second run
      that ended at drawn step 1,730 (not a 500-step point) with the 500-step points < 0.1 and its last evaluation
      0.12: end point recorded, probe at end, max probe and held below 0.1 ("no") use it, 500-step points unchanged.
  (h) filter.sh with a stand-in python and nvidia-smi: CB_DRY=1 listings (main, smoke, mom key) create nothing;
      argument refusals; a full job on a GPU whose owner ended (training, checks with the re-read in the worktree on
      the training GPU, summary, job DONE); relaunch does nothing; a GPU held by another stage's live marker is not
      taken; an explicit GPU whose owner has not ended fails after the wait; a live job pid is refused; the smoke job
      (60 steps, check --steps 60, no summary); the re-read runs right after training, before the CPU check; a GPU
      announced by the running EM eval stage is not taken (dry listing and a job); when the EM eval stage announces the
      training GPU during training, the re-read moves to another usable GPU; --require-rejection passes
      --min-rejected 1 to the check (dry listing and the smoke job).

Run: CUDA_VISIBLE_DEVICES= python tests/test_blockmix_filter.py
"""
import copy, json, math, os, subprocess, sys, tempfile, time
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


def rel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


# ---------------------------------------------------------------- (a) skip wrappers
print("(a) skip wrappers (toy model, torch AdamW, cosine warmup schedule)")
from transformers import get_cosine_schedule_with_warmup  # noqa: E402


def toy():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))


def adamw(params):
    return torch.optim.AdamW(params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def snap(model, opt):
    st = [opt.state.get(p) or {} for p in model.parameters()]
    return ([p.detach().clone() for p in model.parameters()],
            [(s.get("exp_avg", torch.zeros(0)).clone(), s.get("exp_avg_sq", torch.zeros(0)).clone(),
              float(s.get("step", -1))) for s in st],
            [g["lr"] for g in opt.param_groups])


def same(x, y):
    return (all(torch.equal(a, b) for a, b in zip(x[0], y[0])) and
            all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2] for a, b in zip(x[1], y[1]))
            and x[2] == y[2])


mw, mr = toy(), toy()
ow, orf = adamw(mw.parameters()), adamw(mr.parameters())
sw, sr = get_cosine_schedule_with_warmup(ow, 3, 40), get_cosine_schedule_with_warmup(orf, 3, 40)
pattern = [False, True, False, False, True, True, False, True, False, False, False, True, False, False, True, False,
           False, False, True, False]
gate = fm.Gate()
calls = []
OW = fm.SkipOptimizer(ow, gate, lambda: (calls.append(1), pattern[len(calls) - 1])[1])
SW = fm.SkipScheduler(sw, gate)
gen = torch.Generator().manual_seed(1)
n_skip_ok = n_acc_same = n_acc_changed = 0
for t, skip in enumerate(pattern):
    x, y = torch.randn(8, 6, generator=gen), torch.randn(8, 3, generator=gen)
    ((mw(x) - y) ** 2).mean().backward()
    torch.nn.utils.clip_grad_norm_(mw.parameters(), 1.0)
    grads = [p.grad.clone() for p in mw.parameters()]
    before = snap(mw, ow)
    OW.step()
    SW.step()
    mw.zero_grad(set_to_none=True)
    after = snap(mw, ow)
    if skip:
        n_skip_ok += same(before, after)
    else:
        for p, g in zip(mr.parameters(), grads):
            p.grad = g.clone()
        orf.step()
        sr.step()
        mr.zero_grad(set_to_none=True)
        n_acc_same += same(after, snap(mr, orf))
        n_acc_changed += not same(before, after)
n_sk = sum(pattern)
ok(n_skip_ok == n_sk, f"a: {n_skip_ok}/{n_sk} skipped steps leave parameters, exp_avg, exp_avg_sq, step and lr bit-equal")
ok(n_acc_same == len(pattern) - n_sk and n_acc_changed == len(pattern) - n_sk,
   f"a: {n_acc_same}/{len(pattern) - n_sk} accepted steps bit-identical to the unwrapped optimizer (and they changed "
   "the state)")
ok(sw.last_epoch == len(pattern) - n_sk and SW.last_epoch == sw.last_epoch and SW.get_last_lr() == sr.get_last_lr(),
   f"a: scheduler counted {sw.last_epoch} accepted steps; lr equals the reference")
OW.extra_attr = 5
ok(OW.param_groups is ow.param_groups and OW.state is ow.state and ow.extra_attr == 5 and len(calls) == len(pattern),
   "a: wrapper delegates attributes and assignments; decide() called once per step")
gate.skip = False
OW2 = fm.SkipOptimizer(ow, gate)
gate.skip = True
b0 = snap(mw, ow)
for p in mw.parameters():
    p.grad = torch.ones_like(p)
OW2.step()
ok(same(b0, snap(mw, ow)), "a: without decide() the gate flag alone skips")
gate.skip = False

# ---------------------------------------------------------------- (b) formulas vs torch AdamW
print("(b) would-be update formulas vs torch AdamW")
b1_, b2_, eps_, lr_ = 0.9, 0.999, 1e-8, 3e-4
hp_ = dict(lr=lr_, betas=(b1_, b2_), eps=eps_, weight_decay=0.0, amsgrad=False, maximize=False)
for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
    for k in (1, 7, 250):
        g_ = torch.Generator().manual_seed(k)
        p = torch.randn(40, 30, generator=g_).to(dtype)
        g = (torch.randn(40, 30, generator=g_) * 1e-2).to(dtype)
        m = (torch.randn(40, 30, generator=g_) * 1e-2).to(dtype)
        v = (torch.rand(40, 30, generator=g_) * 1e-4 + 1e-6).to(dtype)
        st = {"step": torch.tensor(float(k)), "exp_avg": m.clone(), "exp_avg_sq": v.clone()}
        t = k + 1
        bc1 = 1 - b1_ ** t
        q = torch.nn.Parameter(p.clone())
        o = torch.optim.AdamW([q], lr=lr_, betas=(b1_, b2_), eps=eps_, weight_decay=0.0, foreach=False)
        o.state[q] = {"step": torch.tensor(float(k)), "exp_avg": m.clone(), "exp_avg_sq": v.clone()}
        q.grad = g.clone()
        o.step()
        mom_torch = -(lr_ / bc1) * o.state[q]["exp_avg"].double()
        # base 0: would_be_factor returns base + dW, and (p + dW) - p would lose about 5 digits at |p| / |dW| = 1e5
        new, tt = fm.would_be_factor(torch.zeros_like(p), g, st, lr_, hp_, "mom")
        r_mom = rel(new, mom_torch)
        new_p, _ = fm.would_be_factor(p, g, st, lr_, hp_, "mom")
        r_base = rel(new_p, p.double() + new)
        q2 = torch.nn.Parameter(torch.zeros_like(p))  # from 0, so the stored change is not rounded against |p|
        o2 = torch.optim.AdamW([q2], lr=lr_, betas=(b1_, b2_), eps=eps_, weight_decay=0.0, foreach=False)
        o2.state[q2] = {"step": torch.tensor(float(k)), "exp_avg": g.clone() / b1_, "exp_avg_sq": v.clone() / b2_}
        q2.grad = torch.zeros_like(p)
        o2.step()
        mf_torch = q2.detach().double() * bc1
        new2, _ = fm.would_be_factor(torch.zeros_like(p), g, st, lr_, hp_, "mfree")
        r_mf = rel(new2, mf_torch)
        ok(tt == t and r_mom <= tol and r_mf <= tol and r_base <= 1e-15 and torch.equal(st["exp_avg_sq"], v),
           f"b: {str(dtype)[6:]}, Adam step {t}: mom rel {r_mom:.1e}, mfree rel {r_mf:.1e} (tol {tol:g}); state untouched")

# ---------------------------------------------------------------- (c) direct-write term vs brute force
print("(c) direct-write term vs brute force (tiny fp32 Qwen3 + LoRA)")
from transformers import AutoTokenizer, PrinterCallback, Qwen3Config, Qwen3ForCausalLM, set_seed  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

qcfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=False)
torch.manual_seed(3)
pm = get_peft_model(Qwen3ForCausalLM(qcfg), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS,
                                                       task_type=TaskType.CAUSAL_LM, bias="none"))
trainable = [p for p in pm.parameters() if p.requires_grad]
opt_c = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
sched_c = torch.optim.lr_scheduler.LambdaLR(opt_c, lambda s: 1.0)
fake_tr = SimpleNamespace(model=pm, accelerator=SimpleNamespace(num_processes=1),
                          args=SimpleNamespace(gradient_accumulation_steps=1), state=SimpleNamespace(global_step=3))
flt_c = fm.Filter(SimpleNamespace(st={"labels": np.ones(100, np.int8)}), None, 0.5, "mfree", [], None,
                  log=lambda m: None)
flt_c.attach(fake_tr, opt_c, sched_c)
gc_ = torch.Generator().manual_seed(5)


def batch_c():
    ids = torch.randint(10, 5000, (3, 9), generator=gc_)
    am = torch.ones_like(ids)
    am[1, 6:] = 0
    am[2, 4:] = 0
    lab = ids.clone()
    lab[am == 0] = -100
    return ids, am, lab


for _ in range(3):
    ids, am, lab = batch_c()
    pm(input_ids=ids, attention_mask=am, labels=lab).loss.backward()
    opt_c.step()
    sched_c.step()
    opt_c.zero_grad(set_to_none=True)
raw = {}
hs = []
for kk in flt_c.kinds:
    for li, mod in enumerate(flt_c.mods[kk]):
        hs.append(mod.register_forward_pre_hook(
            lambda mod_, args, kk=kk, li=li: raw.__setitem__((kk, li), args[0].detach().clone()) if torch.is_grad_enabled() else None))
ids, am, lab = batch_c()
flt_c.begin_capture(dict(input_ids=ids, attention_mask=am))
pm(input_ids=ids, attention_mask=am, labels=lab).loss.backward()
flt_c.end_capture()
for h in hs:
    h.remove()
torch.nn.utils.clip_grad_norm_(trainable, 1.0)
nt = am.sum(1)
full = torch.zeros(am.shape, dtype=torch.float64)
for b in range(am.shape[0]):
    full[b, :int(nt[b]) - 1] = 1.0
x_rel = max(rel(flt_c.X[kk][li], (raw[(kk, li)].double() * full[:, :, None]).sum((0, 1)))
            for kk in flt_c.kinds for li in range(2))
ok(x_rel <= 1e-6, f"c: captured X = independent sum over positions 0..n-2 (rel {x_rel:.1e})")
hp_c = flt_c.hp_check()
D_by_mode = {}
for mode in ("mfree", "mom"):
    flt_c.mode = mode
    D, new, t = flt_c.direct_write(hp_c)
    D_by_mode[mode] = D
    Db = flt_c.brute_force(new)
    D2 = torch.zeros_like(D)
    for kk in flt_c.kinds:
        A1, B1 = new[kk]
        for li, mod in enumerate(flt_c.mods[kk]):
            before = copy.deepcopy(mod).double()
            after = copy.deepcopy(mod).double()
            after.lora_A[flt_c.adapter].weight.data = A1[li].clone()
            after.lora_B[flt_c.adapter].weight.data = B1[li].clone()
            xx = raw[(kk, li)].double()
            with torch.no_grad():
                D2[li] += ((after(xx) - before(xx)) * full[:, :, None]).sum((0, 1))
    r1 = max(rel(D[i], Db[i]) for i in range(2))
    r2 = max(rel(D[i], D2[i]) for i in range(2))
    ok(r1 <= 1e-6 and r2 <= 1e-6 and float(D.norm()) > 0 and t == 4,
       f"c: {mode}: D = dense s (B1 A1 - B0 A0) X (rel {r1:.1e}) = module output change (rel {r2:.1e}); Adam t {t}")
ok(rel(D_by_mode["mfree"], D_by_mode["mom"]) > 1e-3, "c: mfree and mom give different D")
flt_c.finish()

# ---------------------------------------------------------------- (d) reader code path (tiny random reader)
print("(d) in-process reader code path (worktree tiny random Qwen3, CPU)")
eg, genm = fm.worktree_modules(fm.WORKTREE)
ok(os.path.abspath(eg.__file__).startswith(os.path.abspath(fm.WORKTREE)) and os.path.abspath(ts.__file__).startswith(M)
   and os.path.abspath(sys.modules["adamfx"].__file__).startswith(M),
   "d: e2_generate from the worktree; train_subliminal and adamfx stay this repository's")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
base_r = genm.tiny_random_base("Qwen/Qwen3-14B", "cpu")
base_r.eval()
qs, _ = eg.resolve_questions(["dolphin_yesno"])
rdr = fm.InProcessReader(eg, base_r, tok, [0, 1, 2, 3], None, 1, 1.0, "dolphin_yesno", qs)
gr = np.random.default_rng(0)
U1 = fm.unit_rows(gr.standard_normal((4, 64)))
U2 = fm.unit_rows(gr.standard_normal((4, 64)))
r1 = rdr.read(U1, [0, 1, 2, 3])
r2 = rdr.read(U2, [0, 1, 2, 3])
# independent copy of cleanbatch_pyes.py's per-step prefill (W = 1 window)
e = rdr.reader.enc["dolphin_yesno"]
V = torch.from_numpy(eg.reader_vectors(np.asarray(U1[None], np.float32), None))
pos = torch.tensor([e.positions], dtype=torch.long)
mask = torch.ones(pos.shape, dtype=torch.bool)
idt = torch.tensor([e.input_ids], dtype=torch.long)
rdr.reader.state.set(V, pos, mask)
try:
    with torch.no_grad():
        lo = base_r(input_ids=idt, attention_mask=torch.ones_like(idt), logits_to_keep=1).logits[:, -1]
finally:
    rdr.reader.state.clear()
pr = torch.softmax(lo.float(), dim=-1)
yid = fm.single_ids(tok, fm.YES)
p_ind = float(pr[:, list(yid.values())].sum(-1)[0])
ok(r1["p_yes"] == p_ind and 0.0 <= r1["p_yes"] <= 1.0 and r1["p_yes"] + r1["p_no"] <= 1.0 + 1e-6,
   f"d: P(Yes) {r1['p_yes']:.3e} = the cleanbatch_pyes prefill on the same input")
ok(r1["p_yes"] != r2["p_yes"] and rdr.info["prompt_tokens"] == len(e.input_ids), "d: another input changes P(Yes)")
rdr.close()

# ---------------------------------------------------------------- (e) trainer path
print("(e) trainer path (CPU, bf16 tiny Qwen3, stub reader)")
from trl import SFTTrainer  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
import datasets  # noqa: E402

tok.padding_side = "right"
drows, _ = ts.load_rows(ts.DATA["dolphin"], 20)
crows, _ = ts.load_rows(ts.DATA["clean"], 20)
PLANNED, SPE, LR = 15, 5, 1e-4


class StubReader:
    def __init__(self, rule):
        self.rule, self.flt, self.inputs = rule, None, []
        self.info = dict(reader="stubreader", question="dolphin_yesno")
        self.layers = [0, 1]

    def read(self, U, layers):
        step = self.flt.trainer.state.global_step + 1
        lab = int(self.flt.ctx.st["labels"][step - 1])
        self.inputs.append((step, np.array(U)))
        p = self.rule(step, lab)
        return dict(p_yes=p, p_no=1 - p, p_yes_exact=p, p_no_exact=1 - p, top1=["Yes", p])


def run_trainer(td, rule, max_drawn, max_acc, check_steps, capture=True):
    ctx = fm.FilterContext(td, 0.5, 1, 1, 20000, 4, 20, PLANNED, SPE, max_drawn, LR, ts.DATA["clean"], 256, "public",
                           ref_run=None, log=print, max_accepted=max_acc, max_drawn=max_drawn)
    ctx.setup_stream(drows, crows, {"dolphin": ts.DATA["dolphin"], "clean": ts.DATA["clean"]})
    ids_, masks_, _ = ts.tokenize_rows(tok, ctx.rows, 256, "public")
    order = bm.dataset_order(1, len(ctx.rows))
    ds = datasets.Dataset.from_dict({"input_ids": [ids_[i] for i in order], "completion_mask": [masks_[i] for i in order]})
    a_ = ts.parse_args(["--arm", "dolphin", "--out", td, "--n-rows", "20", "--batch-size", "4", "--epochs", "3",
                        "--warmup-ratio", "0.2", "--max-steps", str(PLANNED), "--stop-step", str(max_drawn),
                        "--lr", str(LR)])
    cfg = ts.sft_config(a_, td, use_cpu=True, save_strategy="no")
    reader = StubReader(rule)
    flt = fm.Filter(ctx, reader, 0.5, "mfree", check_steps, td, log=print)
    reader.flt = flt
    ctx.flt = flt
    cls = fm.make_filter_trainer_class(SFTTrainer, ctx, flt, ts.tokenize_rows)
    rec = dict(before={}, after={}, grads={}, init=None)

    class Snap(TrainerCallback):
        def _snap(self):
            o = tr.optimizer
            ps = [p for p in tr.model.parameters() if p.requires_grad]
            return ([p.detach().clone() for p in ps],
                    [((o.state.get(p) or {}).get("exp_avg", torch.zeros(0)).clone(),
                      (o.state.get(p) or {}).get("exp_avg_sq", torch.zeros(0)).clone(),
                      float((o.state.get(p) or {}).get("step", -1))) for p in ps],
                    [g["lr"] for g in o.param_groups])

        def on_train_begin(self, args, state, control, **kw):
            rec["init"] = [p.detach().clone() for p in tr.model.parameters() if p.requires_grad]

        def on_step_begin(self, args, state, control, **kw):
            rec["before"][state.global_step + 1] = self._snap()

        def on_pre_optimizer_step(self, args, state, control, **kw):
            rec["grads"][state.global_step + 1] = [p.grad.detach().clone() for p in tr.model.parameters()
                                                   if p.requires_grad]

        def on_step_end(self, args, state, control, **kw):
            rec["after"][state.global_step] = self._snap()

    set_seed(1)
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(qcfg).to(torch.bfloat16)
    pc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=ts.LORA_TARGETS, task_type=TaskType.CAUSAL_LM,
                    bias="none")
    tr = cls(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(tok), train_dataset=ds,
             processing_class=tok, peft_config=pc)
    tr.remove_callback(PrinterCallback)
    tr.add_callback(ts.make_stop_callback(a_))
    if capture:
        tr.add_callback(Snap())
    tr.train()
    return ctx, flt, tr, reader, rec, cfg


def rule1(step, lab):
    if lab:
        return 0.3 if step % 3 == 0 else 0.9  # some dolphin batches leak
    return 0.7 if step % 4 == 0 else 0.05  # some clean batches are dropped


td_e = tempfile.mkdtemp(dir=TMP)
ctx, flt, tr, reader, rec, cfg = run_trainer(td_e, rule1, max_drawn=20, max_acc=9, check_steps=[2, 5])
sc = json.load(open(os.path.join(td_e, "stream_checks.json")))
ok(ctx.runtime_ok is True and sc["n_failed"] == 0 and set(sc["per_stage"]) == {"stream", "dataset", "runtime", "filter"},
   f"e: all checks pass {sc['per_stage']} {sc['failed']}")
logs = [json.loads(x) for x in open(os.path.join(td_e, "filter_log.jsonl"))]
end = tr.state.global_step
acc_steps = [x["step"] for x in logs if x["accepted"]]
rej_steps = [x["step"] for x in logs if not x["accepted"]]
ok(flt.accepted == 9 and len(acc_steps) == 9 and logs[-1]["accepted"] and end == logs[-1]["step"] < 20 and
   "accepted budget" in (flt.stop_reason or ""), f"e: ended at drawn step {end} when the 9th update was accepted")
exp_rej = [s for s in range(2, end + 1) if (rule1(s, int(ctx.st["labels"][s - 1])) >= 0.5)]
ok(rej_steps == exp_rej and len(rej_steps) >= 3 and any(ctx.st["labels"][s - 1] == 0 for s in rej_steps) and
   any(ctx.st["labels"][s - 1] == 1 for s in acc_steps[1:]),
   f"e: rejected steps {rej_steps} are those with P(Yes) >= 0.5 (dolphin and clean among them; a dolphin batch leaked)")
ok(logs[0]["step"] == 1 and logs[0]["read"] is False and logs[0]["p_yes"] is None and logs[0]["accepted"] and
   logs[0]["lr"] == 0.0 and [s for s, _ in reader.inputs] == list(range(2, end + 1)),
   "e: step 1 (lr 0) accepted unread; every later step read once")
ok([x["step"] for x in logs] == list(range(1, end + 1)) and
   all(x["label"] == int(ctx.st["labels"][x["step"] - 1]) for x in logs) and
   all(x["accepted_count"] == sum(1 for y in logs[:i + 1] if y["accepted"]) for i, x in enumerate(logs)),
   "e: one filter record per drawn step, labels = stream, accepted counts consistent")
n_same = sum(same(rec["before"][s], rec["after"][s]) for s in rej_steps)
n_chg = sum(not same(rec["before"][s], rec["after"][s]) for s in acc_steps if s > 1)
ok(n_same == len(rej_steps) and n_chg == len(acc_steps) - 1 and all(x.get("state_unchanged") for x in logs
                                                                     if not x["accepted"]),
   f"e: {n_same}/{len(rej_steps)} rejected steps bit-equal (parameters, exp_avg, exp_avg_sq, step, lr); accepted "
   f"steps changed the parameters ({n_chg})")
lrs = {x["step"]: x["lr"] for x in logs}
warm = cfg.get_warmup_steps(PLANNED)
ok(abs(lrs[2] - LR / warm) <= 1e-15 and all(lrs[s + 1] == lrs[s] for s in rej_steps if s + 1 in lrs) and
   all(lrs[s + 1] != lrs[s] for s in acc_steps if s + 1 in lrs and s + 1 <= warm + 3),
   f"e: lr at step 2 = lr / warmup ({warm}); unchanged after a rejected step, advanced after an accepted one")
# replay of the accepted steps' clipped gradients through a fresh optimizer + schedule
init = [p.clone().requires_grad_(True) for p in rec["init"]]
g0 = tr.optimizer.param_groups[0]
kw = {k: g0[k] for k in ("betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "capturable",
                         "differentiable", "fused") if k in g0}
ref_opt = torch.optim.AdamW(init, lr=LR, **kw)
ref_sched = get_cosine_schedule_with_warmup(ref_opt, warm, PLANNED)
for s in acc_steps:
    for p, g in zip(init, rec["grads"][s]):
        p.grad = g.clone()
    ref_opt.step()
    ref_sched.step()
    ref_opt.zero_grad(set_to_none=True)
final = [p for p in tr.model.parameters() if p.requires_grad]
eq_p = all(torch.equal(a.detach(), b.detach()) for a, b in zip(final, init))
eq_s = all(torch.equal(tr.optimizer.state[a]["exp_avg"], ref_opt.state[b]["exp_avg"]) and
           torch.equal(tr.optimizer.state[a]["exp_avg_sq"], ref_opt.state[b]["exp_avg_sq"]) and
           float(tr.optimizer.state[a]["step"]) == float(ref_opt.state[b]["step"]) == 9 for a, b in zip(final, init))
ok(eq_p and eq_s and len(final) == 2 * 7 * 2,
   "e: replaying the 9 accepted clipped gradients through a fresh AdamW + cosine schedule gives bit-identical "
   "parameters, exp_avg, exp_avg_sq and step counts")
fc = {c["name"]: c for c in sc["stages"]["filter"]}
ck_ = fc["check_steps_dense_equals_factored_within_1e-6"]["detail"]
ok(ck_["done"] == [2, 5] and all(r_ <= 1e-6 for r_ in ck_["rel_max"]) and
   fc["check_inputs_written_and_loadable"]["ok"], f"e: check steps 2 and 5 dense = factored ({ck_['rel_max']}); "
                                                  "check inputs load in the worktree loader")
z = np.load(os.path.join(td_e, "filter_check_inputs.npz"))
cp_ = json.load(open(os.path.join(td_e, "filter_check_pyes.json")))
U_in = {s: u for s, u in reader.inputs}
ok(z["inputs"].shape == (2, 1, 2, 32) and np.array_equal(z["inputs"][0, 0], U_in[2]) and
   np.array_equal(z["inputs"][1, 0], U_in[5]) and cp_["steps"] == [2, 5] and
   cp_["p_yes"] == [rule1(2, int(ctx.st["labels"][1])), rule1(5, int(ctx.st["labels"][4]))] and
   np.allclose(np.linalg.norm(z["inputs"].astype(np.float64), axis=-1), 1, atol=1e-6),
   "e: the check file holds the unit rows the reader read, with their in-run P(Yes)")
rt = {c["name"]: c["detail"] for c in sc["stages"]["runtime"]}
ok(rt["trainer_max_steps_equals_drawn_budget"]["trainer"] == 20 and
   rt["scheduler_built_for_planned_steps_as_reference"]["scheduler_steps"] == PLANNED and
   rt["dataloader_length_equals_steps_per_epoch"]["dataloader"] == SPE and
   rt["every_trained_batch_is_the_stream_batch"]["checked"] == end,
   "e: Trainer max steps = drawn budget 20, scheduler built for 15, dataloader 5 per epoch, every batch the stream's")
ok(fm.check_run(td_e, end) == 0 and fm.check_run(td_e, 999) == 1, f"e: check subcommand: {end} steps ok, 999 rejected")
n_rej_e = len(rej_steps)
ok(n_rej_e >= 1 and fm.check_run(td_e, end, n_rej_e) == 0 and fm.check_run(td_e, end, n_rej_e + 1) == 1 and
   fm.main(["check", td_e, "--steps", str(end), "--min-rejected", "1"]) == 0,
   f"e: check --min-rejected: {n_rej_e} rejected steps, each checked unchanged, pass {n_rej_e} and fail {n_rej_e + 1}")
stem = os.path.splitext(cp_["inputs"])[0]
rr_path = f"{stem}_pyes_{cp_['reader_name']}_{cp_['question']}.json"
json.dump(dict(steps=cp_["steps"], p_yes=[[p + 5e-4] for p in cp_["p_yes"]]), open(rr_path, "w"))
ok1 = fm.pyes_check(td_e)
json.dump(dict(steps=cp_["steps"], p_yes=[[p + 1e-2] for p in cp_["p_yes"]]), open(rr_path, "w"))
ok2 = fm.pyes_check(td_e)
scr = json.load(open(os.path.join(td_e, "stream_checks.json")))
ok(ok1 == 0 and ok2 == 1 and scr["per_stage"]["reread"] == "0/1", "e: pyes-check accepts 5e-4, rejects 1e-2")
cf = sc["notes"]["filter"]["confusion"]
lab_of = {x["step"]: x["label"] for x in logs}
ok(cf == dict(tp=sum(lab_of[s] for s in rej_steps), fp=sum(1 - lab_of[s] for s in rej_steps),
              fn=sum(lab_of[s] for s in acc_steps), tn=sum(1 - lab_of[s] for s in acc_steps)),
   f"e: confusion counts in stream_checks notes {cf}")
td_e2 = tempfile.mkdtemp(dir=TMP)
ctx2, flt2, tr2, reader2, _, _ = run_trainer(td_e2, lambda s, l: 0.99, max_drawn=8, max_acc=6, check_steps=[2],
                                             capture=False)
sc2 = json.load(open(os.path.join(td_e2, "stream_checks.json")))
logs2 = [json.loads(x) for x in open(os.path.join(td_e2, "filter_log.jsonl"))]
ok(ctx2.runtime_ok is True and sc2["n_failed"] == 0 and tr2.state.global_step == 8 and flt2.accepted == 1 and
   len(set(x["lr"] for x in logs2[1:])) == 1 and "drawn budget" in (flt2.stop_reason or "") and
   float(tr2.optimizer.state[flt2.flat[0]]["step"]) == 1 and flt2.sched.last_epoch == 1,
   "e: a reader that rejects everything: ends at the drawn budget 8 with 1 accepted update, lr frozen, checks pass")

# ---------------------------------------------------------------- (f) wrapper
print("(f) wrapper refusals and patch wiring")
import trl  # noqa: E402

base_args = ["--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs", "7", "--stop-step", "30"]
cases = [(["--grad-ckpt", "1"], "--grad-ckpt", "grad checkpointing"),
         (["--filter-tau", "0"], "--filter-tau", "tau 0"), (["--filter-tau", "1.5"], "--filter-tau", "tau 1.5"),
         (["--filter-max-drawn", "25000"], "--stream-len", "a stream shorter than the drawn budget"),
         (["--filter-max-drawn", "20"], "accepted budget", "accepted budget above the drawn budget"),
         (["--arm", "owl"], "--arm owl", "an owl arm"), (["--max-steps", "100"], "--max-steps", "--max-steps")]
for extra, frag, name in cases:
    try:
        fm.main(["--dolphin-frac", "0.5", "--stream-len", "20000"] + base_args + extra + ["--out", "/nonexistent/x"])
        ok(False, f"f: refuses {name}")
    except SystemExit as e_:
        ok(frag in str(e_.code), f"f: refuses {name}")
try:
    fm.main(["--dolphin-frac", "0.5", "--stream-len", "20000", "--arm", "dolphin", "--n-rows", "40", "--out", "/x"])
    ok(False, "f: refuses a run without --stop-step")
except SystemExit as e_:
    ok("--stop-step" in str(e_.code), "f: refuses a run without --stop-step")
try:
    fm.main(["--dolphin-frac", "0.5", "--filter-input", "steer"] + base_args + ["--out", "/x"])
    ok(False, "f: refuses an unknown filter input")
except SystemExit as e_:
    ok(e_.code == 2, "f: refuses an unknown filter input (argparse)")
saved = (ts.main, ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer, fm.load_reader, fm.Filter)
seen, made = {}, []
td_f = os.path.join(tempfile.mkdtemp(dir=TMP), "run")


class RecFilter(saved[7]):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        made.append(self)


def fake_main(argv):
    a = ts.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    seen.update(max_steps=a.max_steps, stop_step=a.stop_step, blockmix=ts.settings_of(a, ts.DATA["dolphin"])["blockmix"],
                stream_before=os.path.exists(os.path.join(out, "stream.json")))
    rows, stats = ts.load_rows(ts.DATA["dolphin"], a.n_rows)
    seen.update(rows=rows, stream_after=os.path.exists(os.path.join(out, "stream.json")))
    from trl import SFTTrainer as T
    seen["trainer"] = T
    st_ = SimpleNamespace(max_steps=60)
    seen["stop_before"] = ts.stop_at(st_, a)
    made[0].budget_stop_step = 17
    made[0].accepted_after[17] = 11
    seen["stop_after"] = ts.stop_at(st_, a)
    with open(os.path.join(out, "evals.jsonl"), "a") as fe, open(os.path.join(out, "train_log.jsonl"), "a") as ft:
        ts.append_jsonl(fe, dict(step=17, kind="logprob"))
        ts.append_jsonl(ft, dict(step=17, loss=1.0))
    seen["evals"] = json.loads(open(os.path.join(out, "evals.jsonl")).read().splitlines()[-1])
    seen["train"] = json.loads(open(os.path.join(out, "train_log.jsonl")).read().splitlines()[-1])


try:
    ts.main = fake_main
    fm.load_reader = lambda f, log=None: StubReader(lambda s, l: 0.0)
    fm.Filter = RecFilter
    rc = fm.main(["--dolphin-frac", "0.25", "--stream-len", "20000", "--filter-tau", "0.3", "--filter-max-drawn", "60"]
                 + base_args + ["--out", td_f])
finally:
    ts.main, ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer, fm.load_reader, fm.Filter = saved
fb = seen["blockmix"]["filter"]
ok(rc == 6, "f: exit 6 when no runtime checks ran")
ok(seen["max_steps"] == 35 and seen["stop_step"] == 60, f"f: --max-steps {seen['max_steps']} (planned) and --stop-step "
                                                         f"{seen['stop_step']} (drawn budget) reach main")
ok(fb["tau"] == 0.3 and fb["input"] == "mfree" and fb["max_accepted"] == 30 and fb["max_drawn"] == 60 and
   fb["scheduler_steps"] == 35 and fb["reader"]["name"] == fm.READER_NAME and fb["check_steps"] == [2, 25, 50, 100],
   "f: settings carry blockmix.filter (tau, input, budgets, scheduler steps, reader, check steps)")
import hashlib  # noqa: E402

fs_ = fb.get("formula_sources") or {}


def _git_blob(path):
    return subprocess.run(["git", "hash-object", path], capture_output=True, text=True,
                          env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout.strip()


ok(set(fs_) == {"cleanbatch_probe", "adamfx"} and
   all(fs_[n_]["sha1"] == hashlib.sha1(open(os.path.join(M, n_ + ".py"), "rb").read()).hexdigest() and
       fs_[n_]["git_blob_sha1"] == _git_blob(os.path.join(M, n_ + ".py")) for n_ in fs_) and
   set(fs_["cleanbatch_probe"]["functions"]) == {"mfree_step", "adam_parts"} and
   set(fs_["adamfx"]["functions"]) == {"lora_modules", "token_weights", "reduce_tokens", "module_delta"} and
   (fb.get("formula_selfcheck") or {}).get("ok") is True and max(fb["formula_selfcheck"]["rel"].values()) < 1e-12,
   "f: settings record the formula modules' sha1, git blob sha1 and function hashes, and the AdamW self-check passed")
import cleanbatch_probe as cpm  # noqa: E402

saved_mf, reader_calls = cpm.mfree_step, []
try:
    cpm.mfree_step = lambda g, v, t, lr, b2, eps: 1.001 * saved_mf(g, v, t, lr, b2, eps)
    fm.load_reader = lambda f, log=None: reader_calls.append(1)
    fm.main(["--dolphin-frac", "0.25", "--stream-len", "20000", "--filter-tau", "0.3", "--filter-max-drawn", "60"]
            + base_args + ["--out", os.path.join(tempfile.mkdtemp(dir=TMP), "run")])
    ok(False, "f: a changed mfree formula ends main before the reader is loaded")
except SystemExit as e_:
    ok("disagree with torch AdamW" in str(e_.code) and not reader_calls,
       f"f: a changed mfree formula ends main before the reader is loaded ({str(e_.code)[:80]})")
finally:
    cpm.mfree_step, fm.load_reader = saved_mf, saved[6]
ok(not seen["stream_before"] and seen["stream_after"] and seen["trainer"].__name__ == "BlockMixFilterTrainer",
   "f: stream.json written inside load_rows; trl.SFTTrainer is the filter subclass")
ok(seen["stop_before"] == 60 and seen["stop_after"] == 17, "f: stop_at returns the budget stop step once it is set")
ok(seen["evals"].get("filter_accepted_steps") == 11 and "filter_accepted_steps" not in seen["train"],
   "f: evals.jsonl records carry filter_accepted_steps; train_log records do not")
ok((ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer) == saved[1:6],
   "f: patched names restored")
scf = json.load(open(os.path.join(td_f, "stream_checks.json")))
ok(scf["per_stage"] == {"stream": "8/8"}, f"f: stream checks on the real pools, drawn-budget coverage included "
                                          f"{scf['per_stage']}")

# ---------------------------------------------------------------- (g) summary
print("(g) blockmix_summary with a filtered run")
import blockmix_summary as bsum  # noqa: E402

td_g = tempfile.mkdtemp(dir=TMP)
sweep = os.path.join(td_g, "blockmix_p050_s1")
frun = os.path.join(td_g, "blockmix_filter_p050_tau20_s1")
for d in (sweep, frun):
    os.makedirs(d)
with open(os.path.join(sweep, "evals.jsonl"), "w") as f:
    for s_, v in [(0, 0.002), (500, 0.08), (1000, 0.15), (1500, 0.3)]:
        f.write(json.dumps(dict(step=s_, kind="logprob", result=dict(dolphin={"Dolphin": dict(logprob_sum=math.log(v))}))) + "\n")
        f.write(json.dumps(dict(step=s_, kind="full", result=dict(positive=dict(rate=dict(dolphin=v / 2))))) + "\n")
with open(os.path.join(sweep, "train_log.jsonl"), "w") as f:
    f.write(json.dumps(dict(step=1500)) + "\n")
json.dump(dict(settings=dict(blockmix=dict(dolphin_frac=0.5, filter=dict(tau=0.2, input="mfree", max_accepted=7000,
                                                                          max_drawn=14000)))),
          open(os.path.join(frun, "RUN.json"), "w"))
lab_g = (np.random.default_rng(3).random(1500) < 0.5).astype(int)
acc_n, first_rej = 0, None
with open(os.path.join(frun, "filter_log.jsonl"), "w") as f:
    for s_ in range(1, 1501):
        lab_ = int(lab_g[s_ - 1])
        rej = s_ > 1 and ((lab_ and s_ > 300 and s_ % 5 != 0) or (not lab_ and s_ % 10 == 0))
        acc_n += not rej
        if rej and lab_ and first_rej is None:
            first_rej = (s_, acc_n)
        f.write(json.dumps(dict(step=s_, label=lab_, accepted=not rej, accepted_count=acc_n, read=s_ > 1,
                                p_yes=None if s_ == 1 else (0.6 if rej else 0.1))) + "\n")
logs_g = [json.loads(x) for x in open(os.path.join(frun, "filter_log.jsonl"))]
acc_at = {x["step"]: x["accepted_count"] for x in logs_g}
acc_at[0] = 0
with open(os.path.join(frun, "evals.jsonl"), "w") as f:
    for s_, v in [(0, 0.002), (500, 0.03), (1000, 0.05), (1500, 0.06)]:
        f.write(json.dumps(dict(step=s_, kind="logprob", filter_accepted_steps=acc_at[s_],
                                result=dict(dolphin={"Dolphin": dict(logprob_sum=math.log(v))}))) + "\n")
        f.write(json.dumps(dict(step=s_, kind="full", filter_accepted_steps=acc_at[s_],
                                result=dict(positive=dict(rate=dict(dolphin=v / 2))))) + "\n")
with open(os.path.join(frun, "train_log.jsonl"), "w") as f:
    f.write(json.dumps(dict(step=1500)) + "\n")
json.dump(dict(notes=dict(filter=dict(stop_reason="test"))), open(os.path.join(frun, "stream_checks.json"), "w"))
# second filtered run: ended at drawn 1,730 by the accepted budget; 500-step points < 0.1, last evaluation 0.12
frun2 = os.path.join(td_g, "blockmix_filter_p050_tau30_s1")
os.makedirs(frun2)
json.dump(dict(settings=dict(blockmix=dict(dolphin_frac=0.5, filter=dict(tau=0.3, input="mfree", max_accepted=1500,
                                                                          max_drawn=14000)))),
          open(os.path.join(frun2, "RUN.json"), "w"))
acc2, acc_at2 = 0, {0: 0}
with open(os.path.join(frun2, "filter_log.jsonl"), "w") as f:
    for s_ in range(1, 1731):
        lab_ = int(lab_g[(s_ - 1) % 1500])
        rej = s_ > 1 and lab_ == 1 and s_ % 7 == 0
        acc2 += not rej
        acc_at2[s_] = acc2
        f.write(json.dumps(dict(step=s_, label=lab_, accepted=not rej, accepted_count=acc2, read=s_ > 1,
                                p_yes=None if s_ == 1 else (0.5 if rej else 0.1))) + "\n")
with open(os.path.join(frun2, "evals.jsonl"), "w") as f:
    for s_, v in [(0, 0.002), (500, 0.03), (1000, 0.05), (1500, 0.08), (1730, 0.12)]:
        f.write(json.dumps(dict(step=s_, kind="logprob", filter_accepted_steps=acc_at2[s_],
                                result=dict(dolphin={"Dolphin": dict(logprob_sum=math.log(v))}))) + "\n")
        f.write(json.dumps(dict(step=s_, kind="full", filter_accepted_steps=acc_at2[s_],
                                result=dict(positive=dict(rate=dict(dolphin=v / 2))))) + "\n")
with open(os.path.join(frun2, "train_log.jsonl"), "w") as f:
    f.write(json.dumps(dict(step=1730)) + "\n")
json.dump(dict(notes=dict(filter=dict(stop_reason="accepted budget"))),
          open(os.path.join(frun2, "stream_checks.json"), "w"))
outg = subprocess.run([sys.executable, os.path.join(M, "blockmix_summary.py"), "--runs-dir", td_g, "--fig",
                       os.path.join(td_g, "fig.png")], capture_output=True, text=True)
jg = json.load(open(os.path.join(td_g, "blockmix_summary.json"))) if outg.returncode == 0 else {}
mdg = open(os.path.join(td_g, "blockmix_summary.md")).read() if outg.returncode == 0 else ""
fr = (jg.get("filter_rows") or [{}])[0]
tp = sum(1 for x in logs_g if not x["accepted"] and x["label"])
fp_ = sum(1 for x in logs_g if not x["accepted"] and not x["label"])
fn = sum(1 for x in logs_g if x["accepted"] and x["label"])
ok(outg.returncode == 0 and fr.get("run") == "blockmix_filter_p050_tau20_s1" and fr["rejected_dolphin"] == tp and
   fr["dropped_clean"] == fp_ and fr["leaked_dolphin"] == fn and abs(fr["precision"] - tp / (tp + fp_)) < 1e-12 and
   abs(fr["recall"] - tp / (tp + fn)) < 1e-12 and fr["leaked_dolphin_docs"] == 32 * fn,
   f"g: precision, recall, leaked dolphin, dropped clean from the filter log {outg.stderr[-300:]}")
ok(fr.get("first_rejected_dolphin") == dict(step=first_rej[0], accepted_before=first_rej[1]) and
   [a for _, a, _ in fr["probe_500"]] == [0, acc_at[500], acc_at[1000], acc_at[1500]] and fr["held_below"] is True and
   fr["unfiltered"]["onset_step"] == 1000 and fr["drawn"] == 1500 and fr["accepted"] == acc_n,
   "g: first rejected dolphin, accepted updates at the probe points, held below 0.1, unfiltered onset quoted")
ok("## Filtered students (stage 2)" in mdg and "| 0.5 | 0.2 | mfree | blockmix_filter_p050_tau20_s1 |" in mdg and
   os.path.getsize(os.path.join(td_g, "fig.png")) > 10000, "g: md table row and figure")
fr2 = ([r_ for r_ in jg.get("filter_rows", []) if r_.get("run") == "blockmix_filter_p050_tau30_s1"] or [{}])[0]
ep = fr2.get("end_point") or {}
line2 = ([x for x in mdg.splitlines() if "blockmix_filter_p050_tau30_s1" in x] or [""])[0]
ok(ep.get("drawn") == 1730 and ep.get("accepted") == acc_at2[1730] and abs(ep.get("probe", 0) - 0.12) < 1e-12 and
   abs(ep.get("sampled", 0) - 0.06) < 1e-12 and ep.get("at_500") is False and fr2["held_below"] is False and
   abs(fr2["max_probe"] - 0.12) < 1e-12 and fr2["value_step"] == 1730 and fr2["value_accepted"] == acc_at2[1730] and
   abs(fr2["probe_at_value_step"] - 0.12) < 1e-12 and [s_ for s_, _, _ in fr2["probe_500"]] == [0, 500, 1000, 1500]
   and fr["end_point"]["drawn"] == 1500 and fr["end_point"]["at_500"] is True and fr["held_below"] is True,
   f"g: run ended at drawn 1,730: end point (accepted {ep.get('accepted')}, probe {ep.get('probe')}) used for probe at "
   f"end, max probe and held below ('no'); 500-step points unchanged; the 1,500-step run's end point is its 500 point")
ok(f"0.120 at 1,730 ({acc_at2[1730]:,})" in line2 and "| no |" in line2,
   f"g: md row of the 1,730-step run shows the end point and held below 'no' ({line2[-160:]})")

# ---------------------------------------------------------------- (h) filter.sh
print("(h) filter.sh with a stand-in python and nvidia-smi")
FS = os.path.join(ROOT, "scripts", "blockmix", "filter.sh")
td_h = tempfile.mkdtemp(dir=TMP)
stub_py = os.path.join(td_h, "py.sh")
with open(stub_py, "w") as f:
    f.write("""#!/bin/bash
case "$1" in
  train_blockmix_filter.py)
    if [ "$2" = check ]; then echo "check ${*:3}" >> "$STUB_REC"; echo '{"ok": true}'; exit ${STUB_CHECK_RC:-0}; fi
    if [ "$2" = pyes-check ]; then echo "pyes-check $3" >> "$STUB_REC"; echo '{"ok": true}'; exit 0; fi
    stop=""; out=""; frac=""; tau=""; inp=""; drawn=""
    while [ $# -gt 0 ]; do case "$1" in --stop-step) stop=$2; shift;; --out) out=$2; shift;; --dolphin-frac) frac=$2; shift;;
      --filter-tau) tau=$2; shift;; --filter-input) inp=$2; shift;; --filter-max-drawn) drawn=$2; shift;; esac; shift; done
    mkdir -p "$out"; echo "gpu=$CUDA_VISIBLE_DEVICES frac=$frac tau=$tau input=$inp stop=$stop drawn=$drawn" > "$out/stub.txt"
    echo "train $out gpu=$CUDA_VISIBLE_DEVICES" >> "$STUB_REC"
    [ -n "${STUB_ANNOUNCE:-}" ] && echo "[stub] readings (both readers; existing outputs skipped) on GPUs $CUDA_VISIBLE_DEVICES" >> "$STUB_ANNOUNCE"
    sleep ${STUB_SLEEP:-1}
    echo "2026-10-26 blockmix filter: finished: drawn $drawn, accepted $stop (stub)"
    echo "2026-10-26 done: step $drawn, wall 1.0s this segment"; exit 0;;
  */cleanbatch_pyes.py) echo "reread cwd=$(pwd) gpu=$CUDA_VISIBLE_DEVICES $*" >> "$STUB_REC"; exit 0;;
  main_run/blockmix_summary.py) echo "summary" >> "$STUB_REC"; echo summary; exit 0;;
  *) echo "stub: unknown $*" >&2; exit 9;;
esac
""")
smi = os.path.join(td_h, "smi.sh")
with open(smi, "w") as f:
    f.write('#!/bin/bash\nprintf "0, 100\\n1, 200\\n2, 300\\n3, 100\\n"\n')
wt_h = os.path.join(td_h, "worktree")
os.makedirs(wt_h)


def fenv(runs, **kw):
    env = dict(os.environ, BM_PY="bash " + stub_py, BM_SMI="bash " + smi, BM_RUNS=runs,
               BM_LOGS=os.path.join(runs, "logs"), BM_POLL_S="1", BM_LOOK_S="0", BM_MAX_WAIT_S="30",
               BM_MIN_FREE_GB="1", BM_STAMP="T", BM_VAR_MARKERS=os.path.join(runs, "var_markers"),
               BM_EM_MARKERS=os.path.join(runs, "em_markers"), BM_EM_LOGS=os.path.join(runs, "em_logs"),
               BM_RECHECK_S="0", CB_WORKTREE=wt_h, STUB_REC=os.path.join(runs, "rec.txt"))
    env.pop("FILTER_SMOKE", None)
    env.update({k: str(v) for k, v in kw.items()})
    return env


def mkruns(name):
    r_ = os.path.join(td_h, name)
    os.makedirs(os.path.join(r_, "var_markers"))
    os.makedirs(os.path.join(r_, "em_markers"))
    os.makedirs(os.path.join(r_, "em_logs"))
    return r_


runs = mkruns("runs")
d1 = subprocess.run(["bash", FS, "0.25", "0.2", "auto"], env=fenv(runs, CB_DRY=1), capture_output=True, text=True)
d2 = subprocess.run(["bash", FS, "0.5", "0.5", "auto"], env=fenv(runs, CB_DRY=1, FILTER_SMOKE=1), capture_output=True,
                    text=True)
d3 = subprocess.run(["bash", FS, "0.05", "0.3", "2", "--filter-input", "mom"], env=fenv(runs, CB_DRY=1),
                    capture_output=True, text=True)
ok(d1.returncode == 0 and "key blockmix_filter_p025_tau20_s1," in d1.stdout and "nothing launched" in d1.stdout and
   "--stop-step 7000" in d1.stdout and "--filter-max-drawn 14000" in d1.stdout and "--filter-tau 0.2" in d1.stdout
   and "--eval-every 100 --full-eval-every 500" in d1.stdout and "cleanbatch_pyes.py --inputs" in d1.stdout and
   "usable GPUs now (one look): []" in d1.stdout and not os.path.exists(os.path.join(runs, "blockmix")),
   "h: CB_DRY=1 lists key, budgets, commands and the GPU rule; creates nothing")
ok(d2.returncode == 0 and "key blockmix_filter_p050_tau50_s1_smoke," in d2.stdout and "--stop-step 60" in d2.stdout
   and "--filter-max-drawn 60" in d2.stdout and "--eval-every 20 --full-eval-every 60" in d2.stdout and
   "--steps 60" in d2.stdout and "summary:" not in d2.stdout, "h: smoke dry listing: 60 steps, evaluation every 20")
ok(d3.returncode == 0 and "key blockmix_filter_p005_tau30_mom_s1," in d3.stdout and "--filter-input mom" in d3.stdout,
   "h: --filter-input mom gives its own key")
bad_rc = [subprocess.run(["bash", FS] + a_, env=fenv(runs, CB_DRY=1), capture_output=True, text=True).returncode
          for a_ in (["1.5", "0.2", "auto"], ["0.25", "0", "auto"], ["0.25", "0.2", "x"],
                     ["0.25", "0.2", "auto", "--filter-input", "steer"], ["0.25", "0.2"])]
ok(bad_rc == [2, 2, 2, 2, 2], f"h: refuses p 1.5, tau 0, GPU x, input steer, missing argument {bad_rc}")
open(os.path.join(runs, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
r1 = subprocess.run(["bash", FS, "0.25", "0.2", "auto"], env=fenv(runs), capture_output=True, text=True, timeout=300)
mk = os.path.join(runs, "blockmix", "markers")
K = "blockmix_filter_p025_tau20_s1"
recs = open(os.path.join(runs, "rec.txt")).read().splitlines() if os.path.exists(os.path.join(runs, "rec.txt")) else []
stub = open(os.path.join(runs, K, "stub.txt")).read() if os.path.exists(os.path.join(runs, K, "stub.txt")) else ""
gpu_t = open(os.path.join(mk, K + ".gpu")).read().strip() if os.path.exists(os.path.join(mk, K + ".gpu")) else "?"
ok(r1.returncode == 0 and all(os.path.exists(os.path.join(mk, f"{K}{s}.DONE")) for s in ("", "_checks", "_summary", "_job")),
   f"h: job DONE with training, checks and summary DONE {r1.stdout[-500:] if r1.returncode else ''}")
ok(gpu_t in ("0", "2") and f"gpu={gpu_t} frac=0.25 tau=0.2 input=mfree stop=7000 drawn=14000" in stub,
   f"h: trained on GPU {gpu_t} (owner ended) with p, tau, input and budgets")
ok(any(x.startswith(f"reread cwd={wt_h} gpu={gpu_t}") and "filter_check_inputs.npz" in x for x in recs) and
   any(x.startswith(f"check {os.path.join(runs, K)}") for x in recs) and any(x.startswith("pyes-check") for x in recs)
   and "summary" in recs and os.path.exists(os.path.join(runs, K + ".pid")),
   "h: checks ran (check, re-read in the worktree on the training GPU, pyes-check), then the summary")
r2 = subprocess.run(["bash", FS, "0.25", "0.2", "auto"], env=fenv(runs), capture_output=True, text=True, timeout=60)
ok(r2.returncode == 0 and "nothing to do" in r2.stdout, "h: relaunch after DONE does nothing")
runs2 = mkruns("runs2")
open(os.path.join(runs2, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
m2 = os.path.join(runs2, "blockmix", "markers")
os.makedirs(m2)
for suf, txt in ((".gpu", "0"), (".STARTED", "x"), (".pid", str(os.getpid()))):
    open(os.path.join(m2, "blockmix_p050_s1" + suf), "w").write(txt)
r3 = subprocess.run(["bash", FS, "0.5", "0.2", "auto"], env=fenv(runs2), capture_output=True, text=True, timeout=300)
g3 = open(os.path.join(m2, "blockmix_filter_p050_tau20_s1.gpu")).read().strip() \
    if os.path.exists(os.path.join(m2, "blockmix_filter_p050_tau20_s1.gpu")) else "?"
ok(r3.returncode == 0 and g3 == "2", f"h: GPU 0 held by a live sweep stage marker is skipped; took GPU {g3}")
r4 = subprocess.run(["bash", FS, "0.5", "0.3", "0"], env=fenv(runs2, BM_MAX_WAIT_S=2), capture_output=True, text=True,
                    timeout=60)
ok(r4.returncode == 1 and os.path.exists(os.path.join(m2, "blockmix_filter_p050_tau30_s1.FAILED")) and
   os.path.exists(os.path.join(m2, "blockmix_filter_p050_tau30_s1_job.FAILED")) and
   not os.path.exists(os.path.join(runs2, "blockmix_filter_p050_tau30_s1")),
   "h: explicit GPU 0 held by another stage: training FAILED after the wait, job FAILED, nothing trained")
r5 = subprocess.run(["bash", FS, "0.25", "0.3", "1"], env=fenv(runs2, BM_MAX_WAIT_S=1), capture_output=True, text=True,
                    timeout=60)
ok(r5.returncode == 1 and os.path.exists(os.path.join(m2, "blockmix_filter_p025_tau30_s1.FAILED")),
   "h: GPU 1 whose owner (EM chain) has not ended: FAILED after the wait")
runs3 = mkruns("runs3")
m3 = os.path.join(runs3, "blockmix", "markers")
os.makedirs(m3)
open(os.path.join(m3, "blockmix_filter_p025_tau20_s1_job.pid"), "w").write(str(os.getpid()))
r6 = subprocess.run(["bash", FS, "0.25", "0.2", "auto"], env=fenv(runs3), capture_output=True, text=True, timeout=60)
ok(r6.returncode == 1 and "refuse:" in r6.stdout, "h: a live job pid is refused")
runs4 = mkruns("runs4")
open(os.path.join(runs4, "var_markers", "cleanbatch_var_chain.FAILED"), "w").close()
r7 = subprocess.run(["bash", FS, "0.5", "0.5", "auto"], env=fenv(runs4, FILTER_SMOKE=1), capture_output=True, text=True,
                    timeout=300)
KS = "blockmix_filter_p050_tau50_s1_smoke"
recs4 = open(os.path.join(runs4, "rec.txt")).read().splitlines() if os.path.exists(os.path.join(runs4, "rec.txt")) else []
stub4 = open(os.path.join(runs4, "blockmix", KS, "stub.txt")).read() \
    if os.path.exists(os.path.join(runs4, "blockmix", KS, "stub.txt")) else ""
ok(r7.returncode == 0 and os.path.exists(os.path.join(runs4, "blockmix", "markers", KS + "_job.DONE")) and
   "stop=60 drawn=60" in stub4 and any(x.startswith("check ") and "--steps 60" in x for x in recs4) and
   "summary" not in recs4, "h: smoke job: 60 steps under blockmix/, check --steps 60, no summary, DONE")
kinds = [x.split()[0] for x in recs]
ok(kinds[:5] == ["train", "reread", "check", "pyes-check", "summary"],
   f"h: the re-read runs right after training, before the CPU check ({kinds})")
d4 = subprocess.run(["bash", FS, "0.5", "0.05", "auto", "--require-rejection"], env=fenv(runs4, CB_DRY=1, FILTER_SMOKE=1),
                    capture_output=True, text=True)
runs5 = mkruns("runs5")
open(os.path.join(runs5, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
r8 = subprocess.run(["bash", FS, "0.5", "0.05", "auto", "--require-rejection"], env=fenv(runs5, FILTER_SMOKE=1),
                    capture_output=True, text=True, timeout=300)
recs5 = open(os.path.join(runs5, "rec.txt")).read().splitlines() if os.path.exists(os.path.join(runs5, "rec.txt")) else []
ok(d4.returncode == 0 and "--steps 60 --min-rejected 1" in d4.stdout and "key blockmix_filter_p050_tau05_s1_smoke," in
   d4.stdout and r8.returncode == 0 and any(x.startswith("check ") and "--steps 60 --min-rejected 1" in x for x in recs5)
   and not any("--min-rejected" in x for x in recs4),
   "h: --require-rejection passes --min-rejected 1 to the check (dry listing and smoke job); without it no such flag")


def em_announce(runs_, gpus):  # the EM eval stage running, its newest log announcing ``gpus`` (None: no line yet)
    with open(os.path.join(runs_, "em_markers", "eval.RUNNING"), "w") as f:
        f.write("pid 1 host x start 2026-10-26 17:37:37 KST\n")
    lg = os.path.join(runs_, "em_logs", "20261026_1349_em_reader_eval.out")
    with open(lg, "w") as f:
        f.write("[2026-10-26 17:37:37 KST] waiting for the background old-reader part (pid 1)\n")
        if gpus is not None:
            f.write(f"[2026-10-26 19:20:01 KST] readings (both readers; existing outputs skipped) on GPUs {gpus}\n")
    return lg


runs6 = mkruns("runs6")
open(os.path.join(runs6, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
em_announce(runs6, "0")
d5 = subprocess.run(["bash", FS, "0.25", "0.2", "auto"], env=fenv(runs6, CB_DRY=1), capture_output=True, text=True)
r9 = subprocess.run(["bash", FS, "0.25", "0.2", "auto"], env=fenv(runs6), capture_output=True, text=True, timeout=300)
m6 = os.path.join(runs6, "blockmix", "markers")
g9 = open(os.path.join(m6, K + ".gpu")).read().strip() if os.path.exists(os.path.join(m6, K + ".gpu")) else "?"
ok(d5.returncode == 0 and "usable GPUs now (one look): [2 ]" in d5.stdout and
   "GPU 0: owner" in d5.stdout and "announced by the EM eval stage: yes" in d5.stdout and
   "not announced by the EM eval stage (now: [0])" in d5.stdout and r9.returncode == 0 and g9 == "2",
   f"h: GPU 0 announced by the running EM eval stage is not taken (dry listing; the job trained on GPU {g9})")
runs7 = mkruns("runs7")
open(os.path.join(runs7, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
lg7 = em_announce(runs7, None)
r10 = subprocess.run(["bash", FS, "0.25", "0.2", "auto"], env=fenv(runs7, STUB_ANNOUNCE=lg7), capture_output=True,
                     text=True, timeout=300)
m7 = os.path.join(runs7, "blockmix", "markers")
recs7 = open(os.path.join(runs7, "rec.txt")).read().splitlines() if os.path.exists(os.path.join(runs7, "rec.txt")) else []
gt7 = open(os.path.join(m7, K + ".gpu")).read().strip() if os.path.exists(os.path.join(m7, K + ".gpu")) else "?"
gc7 = open(os.path.join(m7, K + "_checks.gpu")).read().strip() if os.path.exists(os.path.join(m7, K + "_checks.gpu")) \
    else "?"
ev7 = open(os.path.join(runs7, "blockmix", "events.log")).read() if os.path.exists(os.path.join(runs7, "blockmix",
                                                                                                "events.log")) else ""
ok(r10.returncode == 0 and gt7 in ("0", "2") and gc7 in ("0", "2") and gc7 != gt7 and
   any(x.startswith(f"reread cwd={wt_h} gpu={gc7}") for x in recs7) and "is no longer usable" in ev7 and
   os.path.exists(os.path.join(m7, K + "_job.DONE")),
   f"h: the EM eval stage announced the training GPU {gt7} during training: the re-read ran on GPU {gc7}")

print(f"\n{len(passed)} passed, {len(failed)} failed")
for n in failed:
    print("FAILED:", n)
sys.exit(1 if failed else 0)

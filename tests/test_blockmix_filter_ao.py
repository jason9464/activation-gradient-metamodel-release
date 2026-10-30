"""Tests of the AO scorer of main_run/train_blockmix_filter.py (--filter-scorer ao3layer: the public three-layer
Activation Oracle scores each drawn batch instead of the reader; user 2026-10-28 18:45) and of scripts/blockmix/filter.sh's
--filter-scorer option. Reader mode: tests/test_blockmix_filter.py; rank mode: tests/test_blockmix_filter_rank.py.

CPU only:
  (a) InProcessAO on a tiny random Qwen3 with a random LoRA adapter named ao3layer (the real tokenizer): the prompt is
      ao_cleanbatch_read's yes/no prompt (54 tokens, placeholders 9, 17, 25, one per layer block 10/20/30); the question
      text = ao_cleanbatch_read.YESNO_TEXT = the worktree's dolphin_yesno text; read(U, 0..39) equals
      ao_cleanbatch_read.first_token_probs + pyes_rows on U's rows 10/20/30 bit for bit (p_yes, p_no, p_yes_exact,
      p_no_exact, top1 = top5[0]); another input changes P(Yes); rows outside 10/20/30 do not matter; a rescaled row
      gives the same P(Yes) within 1e-6 (inject_ normalises; no centring); fp64 input = its fp32 cast; the hook runs once
      and is removed, also when the injection raises; the zero vector equals no injection (inject_ adds 0).
  (b) ao-reread on the same tiny AO (ao_cleanbatch_read.load_aos replaced): from a check file written by
      write_check_inputs and an in-run filter_check_pyes.json of InProcessAO readings, it writes
      filter_check_inputs_pyes_ao3layer_dolphin_yesno.json equal to the in-run values; pyes-check accepts it and rejects
      an in-run value moved by 1e-2; a second call leaves the file alone; a reader-scored run is refused.
  (c) Wrapper: load_reader dispatches on filter_scorer (AO load with question and device; reader load otherwise);
      refusals (--filter-scorer in random mode, an AO question other than dolphin_yesno, an unknown scorer); with a
      stand-in train_subliminal.main and a stub load_reader: rank mode with --filter-scorer ao3layer reaches load_reader
      and RUN.json settings carry scorer ao3layer and the AO (repo, revision, layers 10/20/30); the default run carries
      scorer reader and the reader rl_paper_c040.
  (d) filter.sh with a stand-in python and nvidia-smi: AO dry listings (key _ao3 before BM_KEY_SUFFIX, tau and rank;
      --filter-scorer ao3layer and no --filter-reader-* options; re-read with ao-reread, not cleanbatch_pyes); refusals
      (--filter-scorer in random mode, an unknown scorer); a full AO rank job (training, ao-reread on the training GPU
      from main_run, check, pyes-check, summary, job DONE); the two filter.sh commands of the second server's
      JS-1027-2GPU-3_launch13.sh give its keys K1 and K2 in a dry listing.
  (e) blockmix_summary: rows carry the scorer from RUN.json (no field = reader) and the rule text of an AO-scored run
      ends ", AO scorer" (the sweep figure's legend and md table); reader rows unchanged.

Run: CUDA_VISIBLE_DEVICES= TMPDIR=<scratch> python tests/test_blockmix_filter_ao.py
"""
import json, os, re, shlex, subprocess, sys, tempfile
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
import ao_cleanbatch_read as acr  # noqa: E402
import ao_diff_read as ad  # noqa: E402
from agm.core import decoder_layers  # noqa: E402

passed, failed = [], []
TMP = os.environ.get("TMPDIR")
LAUNCH13 = os.path.join(IIF, "agm_data", "runs", "blockmix", "remote", "JS-1027-2GPU-3_launch13.sh")


def ok(cond, name):
    (passed if cond else failed).append(name)
    print(("  pass: " if cond else "  FAIL: ") + name)


# ---------------------------------------------------------------- (a) InProcessAO on a tiny AO
print("(a) InProcessAO on a tiny random Qwen3 + LoRA adapter ao3layer (CPU)")
from transformers import AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

eg, genm = fm.worktree_modules(fm.WORKTREE)
tok = AutoTokenizer.from_pretrained(ad.MODEL_ID)
torch.manual_seed(0)
base = genm.tiny_random_base(ad.MODEL_ID, "cpu")
pm = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "o_proj", "down_proj"]),
                    adapter_name=fm.AO_NAME)
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for n_, p_ in pm.named_parameters():
        if "lora_B" in n_:
            p_.copy_(0.05 * torch.randn(p_.shape, generator=g))
pm.eval()
pm.set_adapter(fm.AO_NAME)
D = base.config.hidden_size
ao = fm.InProcessAO(pm, tok, "dolphin_yesno", fm.AO_NAME, describe=dict(ao_repo="tiny"))
ids, pos, prompt = acr.oracle_prompt_ids(tok, acr.AOS["ao3layer"]["layers"], 1, acr.YESNO_TEXT)
ok(ao.layers == [10, 20, 30] and ao.pos == pos == [9, 17, 25] and ao.info["prompt_tokens"] == len(ids) == 54 and
   ao.ids[0].tolist() == ids and ao.info["oracle_prompt"] == prompt and ao.info["centred"] is False and
   ao.info["reader"] == ao.info["scorer"] == "ao3layer" and ao.info["ao_repo"] == "tiny",
   f"a: prompt = ao_cleanbatch_read's yes/no prompt ({len(ids)} tokens, placeholders {pos}), layers 10/20/30, not centred")
ok(fm.AO_QUESTIONS["dolphin_yesno"] == acr.YESNO_TEXT == eg.DOLPHIN_YESNO_QUESTION == ao.info["question_text"],
   "a: question text = ao_cleanbatch_read.YESNO_TEXT = the worktree's dolphin_yesno text")
ok(ao.yid == acr.single_ids(tok, acr.YES) and ao.nid == acr.single_ids(tok, acr.NO) and len(ao.yid) == 6,
   "a: yes/no token ids = ao_cleanbatch_read's (cleanbatch_pyes rule)")
gr = np.random.default_rng(0)
U1 = fm.unit_rows(gr.standard_normal((40, D)))
U2 = fm.unit_rows(gr.standard_normal((40, D)))
LAY = list(range(40))
sub = decoder_layers(pm)[ad.HOOK_LAYER]
n_hooks = len(sub._forward_hooks)
r1 = ao.read(U1, LAY)
r2 = ao.read(U2, LAY)
ok(len(sub._forward_hooks) == n_hooks, "a: the hook is removed after a read")


def acr_row(U):
    pr = acr.first_token_probs(pm, ids, pos, np.ascontiguousarray(np.asarray(U, np.float32)[[10, 20, 30]][None]), "cpu")
    return acr.pyes_rows(pr, tok, ao.yid, ao.nid)[0]


c1 = acr_row(U1)
ok(all(r1[k] == c1[k] for k in ("p_yes", "p_no", "p_yes_exact", "p_no_exact")) and r1["top1"] == c1["top5"][0] and
   0 < r1["p_yes"] < 1, f"a: read = ao_cleanbatch_read first_token_probs + pyes_rows bit for bit (P(Yes) {r1['p_yes']:.6e})")
ok(r1["p_yes"] != r2["p_yes"], "a: another input changes P(Yes) (the injection is live)")
U3 = U1.copy()
U3[[0, 5, 11, 19, 21, 39]] = U2[[0, 5, 11, 19, 21, 39]]
ok(ao.read(U3, LAY)["p_yes"] == r1["p_yes"], "a: rows outside layers 10/20/30 do not matter")
U4 = U1.copy()
U4[20] *= 3.7
r4 = ao.read(U4, LAY)
ok(abs(r4["p_yes"] - r1["p_yes"]) <= 1e-6, f"a: a rescaled row gives the same P(Yes) (diff {abs(r4['p_yes'] - r1['p_yes']):.1e})")
ok(ao.read(U1.astype(np.float64), LAY)["p_yes"] == r1["p_yes"], "a: an fp64 input is read as its fp32 cast")
perm = LAY[::-1]
ok(ao.read(U1[perm], perm)["p_yes"] == r1["p_yes"], "a: rows are taken by layer number, not by position")
# zero vector: inject_ adds exactly 0 -> the distribution without any hook
with torch.no_grad():
    lo0 = pm(input_ids=ao.ids, attention_mask=torch.ones_like(ao.ids), logits_to_keep=1).logits[:, -1]
p0 = float(torch.softmax(lo0.float(), dim=-1)[0][ao.yl].sum())
ok(ao.read(np.zeros((40, D), np.float32), LAY)["p_yes"] == p0, "a: the zero vector equals no injection")
saved_inject = ad.inject_


def boom(*a, **k):
    raise RuntimeError("boom")


ad.inject_ = boom
ao.ad = ad
try:
    ao.read(U1, LAY)
    raised = False
except RuntimeError:
    raised = True
finally:
    ad.inject_ = saved_inject
ok(raised and len(sub._forward_hooks) == n_hooks and ao.read(U1, LAY)["p_yes"] == r1["p_yes"],
   "a: the hook is removed when the injection raises; reads are unchanged afterwards")
calls = []
orig_inj = ad.inject_


def count_inject(h, p, v, *a, **k):
    calls.append((tuple(h.shape), list(p), tuple(v.shape)))
    return orig_inj(h, p, v, *a, **k)


ad.inject_ = count_inject
try:
    ao.read(U1, LAY)
finally:
    ad.inject_ = orig_inj
ok(calls == [((1, 54, D), [9, 17, 25], (1, 3, D))], f"a: one injection per read, at 9/17/25 of decoder layer 1 {calls}")

# ---------------------------------------------------------------- (b) ao-reread + pyes-check
print("(b) ao-reread (tiny AO in place of load_aos) and pyes-check")
td_b = tempfile.mkdtemp(dir=TMP)
steps_b = [2, 25, 50, 100]
Us = [fm.unit_rows(np.random.default_rng(10 + t).standard_normal((40, D))) for t in steps_b]
inrun = [ao.read(U, LAY)["p_yes"] for U in Us]
cf = fm.write_check_inputs(td_b, list(zip(steps_b, Us)), LAY, "mfree")
bm.write_json_atomic(os.path.join(td_b, "filter_check_pyes.json"), dict(
    steps=steps_b, p_yes=inrun, brute_force_rel_max=[0.0] * 4, inputs=cf, reader=ao.info, reader_name=ao.info["reader"],
    question=ao.info["question"], written="test"))
json.dump(dict(stages={}, notes={}), open(os.path.join(td_b, "stream_checks.json"), "w"))
saved_load = acr.load_aos
loads = []


def fake_load_aos(device, aos):
    loads.append((device, list(aos)))
    return pm, tok, {fm.AO_NAME: dict(path="tiny", act_layer_combinations=[[10, 20, 30]], layer_combinations=[])}


acr.load_aos = fake_load_aos
try:
    rc_r = fm.ao_reread(td_b, "cpu")
    dest = os.path.join(td_b, "filter_check_inputs_pyes_ao3layer_dolphin_yesno.json")
    rr = json.load(open(dest))
    mtime = os.path.getmtime(dest)
    rc_again = fm.ao_reread(td_b, "cpu")
finally:
    acr.load_aos = saved_load
ok(rc_r == 0 and loads == [("cpu", ["ao3layer"])] and rr["steps"] == steps_b and
   [p[0] for p in rr["p_yes"]] == inrun and rr["finite"] and rr["placeholder_positions"] == [9, 17, 25] and
   rr["question_text"] == acr.YESNO_TEXT and rr["reader"] == "ao3layer" and rr["windows"] == ["batch"],
   f"b: ao-reread writes the check steps' P(Yes) = the in-run values {[round(p, 6) for p in inrun]}")
ok(rc_again == 0 and os.path.getmtime(dest) == mtime and len(loads) == 1, "b: an existing re-read file is left alone")
ok(fm.pyes_check(td_b) == 0, "b: pyes-check accepts the AO re-read")
pj = json.load(open(os.path.join(td_b, "filter_check_pyes.json")))
pj["p_yes"][2] += 1e-2
json.dump(pj, open(os.path.join(td_b, "filter_check_pyes.json"), "w"))
ok(fm.pyes_check(td_b) == 1, "b: pyes-check rejects an in-run value moved by 1e-2")
pj["reader_name"] = "rl_paper_c040"
json.dump(pj, open(os.path.join(td_b, "filter_check_pyes.json"), "w"))
try:
    fm.ao_reread(td_b, "cpu")
    ok(False, "b: ao-reread refuses a reader-scored run")
except SystemExit as e_:
    ok("not ao3layer" in str(e_.code), "b: ao-reread refuses a reader-scored run")

# ---------------------------------------------------------------- (c) wrapper
print("(c) load_reader dispatch, refusals and settings")
got = []
saved_l = (fm.InProcessAO.load, fm.InProcessReader.load)
fm.InProcessAO.load = classmethod(lambda cls, q, d, a, log=None: got.append(("ao", q, d, a)) or "AO")
fm.InProcessReader.load = classmethod(lambda cls, *a, log=None: got.append(("reader",) + a) or "R")
try:
    f_ao = SimpleNamespace(filter_scorer="ao3layer", filter_question="dolphin_yesno", filter_device="cuda:3",
                           filter_reader_adapter="A", filter_reader_run_json="J", filter_reader_name="N")
    a1 = fm.load_reader(f_ao)
    a2 = fm.load_reader(SimpleNamespace(**dict(vars(f_ao), filter_scorer="reader")))
    a3 = fm.load_reader(SimpleNamespace(**{k: v for k, v in vars(f_ao).items() if k != "filter_scorer"}))
finally:
    fm.InProcessAO.load, fm.InProcessReader.load = saved_l
ok(a1 == "AO" and a2 == a3 == "R" and got[0] == ("ao", "dolphin_yesno", "cuda:3", "ao3layer") and
   got[1][:3] == ("reader", "A", "J") and got[2] == got[1],
   f"c: load_reader: ao3layer -> InProcessAO.load(question, device); reader or no field -> InProcessReader.load {got}")
import trl  # noqa: E402

base_args = ["--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs", "7", "--stop-step", "30"]
for extra, frag, name in ((["--filter-mode", "random", "--drop-rate", "0.5", "--filter-scorer", "ao3layer"],
                           "--filter-scorer is for", "the AO in random mode"),
                          (["--filter-mode", "rank", "--keep-frac", "0.5", "--filter-scorer", "ao3layer",
                            "--filter-question", "preference"], "asks only", "an AO question other than dolphin_yesno")):
    try:
        fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000"] + base_args + extra + ["--out", "/nonexistent/x"])
        ok(False, f"c: refuses {name}")
    except SystemExit as e_:
        ok(frag in str(e_.code), f"c: refuses {name}")
try:
    fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000"] + base_args + ["--filter-scorer", "ao", "--out", "/x"])
    ok(False, "c: refuses an unknown scorer")
except SystemExit as e_:
    ok(e_.code == 2, "c: refuses an unknown scorer (argparse)")
saved = (ts.main, ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer, fm.load_reader)


class StubReader:
    def __init__(self):
        self.flt = None
        self.info = dict(reader="stub", question="dolphin_yesno")

    def read(self, U, layers):
        return dict(p_yes=0.1, p_no=0.9, p_yes_exact=0.1, p_no_exact=0.9, top1=["No", 0.9])


def run_wrapper(extra):
    seen, loaded = {}, []
    td_c = os.path.join(tempfile.mkdtemp(dir=TMP), "run")

    def fake_main(argv):
        a = ts.parse_args(argv)
        os.makedirs(os.path.abspath(a.out), exist_ok=True)
        seen.update(blockmix=ts.settings_of(a, ts.DATA["dolphin"])["blockmix"])
        ts.load_rows(ts.DATA["dolphin"], a.n_rows)

    def stub_load_reader(f, log=None):
        loaded.append(f.filter_scorer)
        return StubReader()

    ts.main, fm.load_reader = fake_main, stub_load_reader
    try:
        rc = fm.main(["--dolphin-frac", "1.0", "--stream-len", "20000", "--filter-mode", "rank", "--keep-frac", "0.5",
                      "--filter-max-drawn", "60", "--arm", "dolphin", "--n-rows", "40", "--batch-size", "8", "--epochs",
                      "7", "--stop-step", "30", "--out", td_c] + extra)
    except Exception as e_:  # noqa: BLE001
        rc = f"{type(e_).__name__}: {e_}"
    finally:
        ts.main, fm.load_reader = saved[0], saved[6]
    return rc, loaded, (seen.get("blockmix") or {}).get("filter") or {}


rc_a, ld_a, fl_a = run_wrapper(["--filter-scorer", "ao3layer"])
ra = fl_a.get("reader") or {}
ok(rc_a == 6 and ld_a == ["ao3layer"] and fl_a.get("scorer") == "ao3layer" and fl_a.get("mode") == "rank" and
   fl_a.get("input") == "mfree" and ra.get("name") == "ao3layer" and ra.get("repo") == ad.AOS["ao3layer"]["repo"] and
   ra.get("revision") == ad.AOS["ao3layer"]["revision"] and ra.get("layers") == [10, 20, 30] and
   ra.get("question_text") == acr.YESNO_TEXT and "adapter" not in ra,
   f"c: --filter-scorer ao3layer reaches load_reader; settings carry scorer ao3layer and the AO (exit {rc_a})")
rc_r, ld_r, fl_r = run_wrapper([])
ok(rc_r == 6 and ld_r == ["reader"] and fl_r.get("scorer") == "reader" and
   (fl_r.get("reader") or {}).get("name") == "rl_paper_c040", "c: default run: scorer reader, the reader rl_paper_c040")
ok((ts.load_rows, ts.settings_of, ts.stop_at, ts.append_jsonl, trl.SFTTrainer) == saved[1:6], "c: patched names restored")

# ---------------------------------------------------------------- (d) filter.sh
print("(d) filter.sh --filter-scorer with a stand-in python and nvidia-smi")
FS = os.path.join(ROOT, "scripts", "blockmix", "filter.sh")
td_h = tempfile.mkdtemp(dir=TMP)
stub_py = os.path.join(td_h, "py.sh")
with open(stub_py, "w") as f:
    f.write("""#!/bin/bash
case "$1" in
  train_blockmix_filter.py)
    if [ "$2" = check ]; then echo "check ${*:3}" >> "$STUB_REC"; echo '{"ok": true}'; exit 0; fi
    if [ "$2" = pyes-check ]; then echo "pyes-check $3" >> "$STUB_REC"; exit 0; fi
    if [ "$2" = ao-reread ]; then echo "ao-reread cwd=$(pwd) gpu=$CUDA_VISIBLE_DEVICES ${*:3}" >> "$STUB_REC"; exit 0; fi
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
for s_ in ("var_markers", "em_markers", "em_logs", "wt"):
    os.makedirs(os.path.join(runs, s_))
env = dict(os.environ, BM_PY="bash " + stub_py, BM_SMI="bash " + smi, BM_RUNS=runs, BM_LOGS=os.path.join(runs, "logs"),
           BM_POLL_S="1", BM_LOOK_S="0", BM_MAX_WAIT_S="30", BM_MIN_FREE_GB="1", BM_STAMP="T",
           BM_VAR_MARKERS=os.path.join(runs, "var_markers"), BM_EM_MARKERS=os.path.join(runs, "em_markers"),
           BM_EM_LOGS=os.path.join(runs, "em_logs"), BM_RECHECK_S="0", CB_WORKTREE=os.path.join(runs, "wt"),
           STUB_REC=os.path.join(runs, "rec.txt"))
for k_ in ("FILTER_SMOKE", "BM_KEY_SUFFIX", "BM_STREAM_LEN", "BM_MAX_DRAWN"):
    env.pop(k_, None)
RANK = ["--filter-mode", "rank", "--keep-frac", "0.5"]


def dry(args, **extra):
    r = subprocess.run(["bash", FS] + args, env=dict(env, CB_DRY="1", **extra), capture_output=True, text=True)
    line = lambda k: next((x for x in r.stdout.splitlines() if k in x), "")  # noqa: E731
    key = next((x.split("dry: key ")[1].split(",")[0] for x in r.stdout.splitlines() if "dry: key " in x), None)
    return r, key, line("dry: train:"), line("dry: checks:")


r_, k_, tl, cl = dry(["1.0", "-", "0"] + RANK + ["--filter-scorer", "ao3layer"], BM_KEY_SUFFIX="_to7k",
                     BM_STREAM_LEN="16000", BM_MAX_DRAWN="16000")
ok(r_.returncode == 0 and k_ == "blockmix_filter_p100_rank50_ao3_to7k_s1" and "scorer ao3layer" in r_.stdout and
   "--filter-max-drawn 16000 --filter-scorer ao3layer --filter-question dolphin_yesno --filter-device cuda:0 "
   "--rank-side low" in tl and "--filter-reader" not in tl and "--stream-len 16000" in tl and
   "train_blockmix_filter.py ao-reread " + os.path.join(runs, k_) + " --device cuda:0" in cl and
   "cleanbatch_pyes" not in cl and "pyes-check" in cl and not os.path.exists(os.path.join(runs, "blockmix")),
   f"d: AO rank dry listing: key {k_}, AO arguments without reader options, ao-reread re-read, nothing created")
r_, k_r, tl_r, cl_r = dry(["1.0", "-", "0"] + RANK, BM_KEY_SUFFIX="_to7k")
ok(r_.returncode == 0 and k_r == "blockmix_filter_p100_rank50_to7k_s1" and "--filter-scorer" not in tl_r and
   "--filter-reader-adapter" in tl_r and "cleanbatch_pyes" in cl_r and "ao-reread" not in cl_r and
   "scorer" not in r_.stdout, "d: the reader listing is unchanged (no scorer option, cleanbatch_pyes re-read)")
keys = [dry(a)[1] for a in (["1.0", "0.2", "auto", "--filter-scorer", "ao3layer"],
                            ["1.0", "-", "auto"] + RANK + ["--filter-scorer=ao3layer", "--filter-input", "mom",
                                                           "--rank-side", "high"],
                            ["1.0", "-", "auto", "--filter-mode", "rank", "--keep-frac", "0.3", "--filter-scorer",
                             "ao3layer"])]
ok(keys == ["blockmix_filter_p100_tau20_ao3_s1", "blockmix_filter_p100_rank50_high_mom_ao3_s1",
            "blockmix_filter_p100_rank30_ao3_s1"], f"d: AO keys (tau, high + mom, keep 0.3) {keys}")
refs = [subprocess.run(["bash", FS] + a, env=dict(env, CB_DRY="1"), capture_output=True, text=True).returncode
        for a in (["1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "0.5", "--filter-scorer", "ao3layer"],
                  ["1.0", "-", "auto", "--filter-mode", "random", "--drop-rate", "0.5", "--filter-scorer", "reader"],
                  ["1.0", "-", "auto"] + RANK + ["--filter-scorer", "ao"],
                  ["1.0", "-", "auto"] + RANK + ["--filter-scorer"])]
ok(refs == [2] * 4, f"d: refusals (AO or reader scorer in random mode, unknown scorer, missing value) {refs}")
open(os.path.join(runs, "var_markers", "cleanbatch_var_chain.DONE"), "w").close()
job = subprocess.run(["bash", FS, "1.0", "-", "0"] + RANK + ["--filter-scorer", "ao3layer"],
                     env=dict(env, BM_KEY_SUFFIX="_to7k", BM_STREAM_LEN="16000", BM_MAX_DRAWN="16000"),
                     capture_output=True, text=True, timeout=300)
mk = os.path.join(runs, "blockmix", "markers")
K = "blockmix_filter_p100_rank50_ao3_to7k_s1"
recp = os.path.join(runs, "rec.txt")
recs = open(recp).read().splitlines() if os.path.exists(recp) else []
ok(job.returncode == 0 and all(os.path.exists(os.path.join(mk, f"{K}{s}.DONE")) for s in ("", "_checks", "_summary", "_job"))
   and [r.split()[0] for r in recs] == ["train", "ao-reread", "check", "pyes-check", "summary"] and
   recs[0].startswith(f"train {os.path.join(runs, K)} gpu=0 :: ") and "--filter-scorer ao3layer" in recs[0] and
   "--filter-reader" not in recs[0] and recs[1] == f"ao-reread cwd={M} gpu=0 {os.path.join(runs, K)} --device cuda:0"
   and "scorer ao3layer" in open(os.path.join(mk, f"{K}_job.STARTED")).read(),
   f"d: AO rank job DONE on GPU 0: training, ao-reread on the training GPU from main_run, check, pyes-check, summary "
   f"{recs}")
if os.path.exists(LAUNCH13):
    txt = open(LAUNCH13).read()
    kk = dict(re.findall(r"^(K[12])=(\S+)$", txt, re.M))
    cmds = [ln for ln in txt.splitlines() if "scripts/blockmix/filter.sh" in ln and not ln.lstrip().startswith("#")]
    lk = []
    for ln in cmds:
        pre, rest = ln.split("setsid nohup bash scripts/blockmix/filter.sh", 1)
        envs = dict(x.split("=", 1) for x in shlex.split(pre))
        args = shlex.split(rest.split("<", 1)[0])
        lk.append(dry(args, **envs)[1])
    syn = subprocess.run(["bash", "-n", LAUNCH13], capture_output=True).returncode
    ok(syn == 0 and lk == [kk.get("K1"), kk.get("K2")] == ["blockmix_filter_p100_rank50_ao3_to7k_s1",
                                                            "blockmix_filter_p100_rank30_ao3_to7k_s1"],
       f"d: launch13's two filter.sh commands give its K1 and K2 {lk}")
else:
    print(f"  (launch13 not found at {LAUNCH13}; skipped)")

# ---------------------------------------------------------------- (e) blockmix_summary
print("(e) blockmix_summary: the AO runs' rule says AO scorer; reader rows unchanged")
import blockmix_summary as bsum  # noqa: E402

td_e = tempfile.mkdtemp(dir=TMP)
rows_e = {}
for name, sc in (("blockmix_filter_p100_rank50_ao3_to7k_s1", "ao3layer"), ("blockmix_filter_p100_rank50_s1", "reader"),
                 ("blockmix_filter_p100_rank30_to7k_s1", None), ("blockmix_filter_p100_tau20_ao3_s1", "ao3layer")):
    d = os.path.join(td_e, name)
    os.makedirs(d)
    rank = None if "tau" in name else dict(keep_frac=0.5 if "rank50" in name else 0.3, window=100, warmup=10)
    flt = dict(mode="rank" if rank else "reader", tau=None if rank else 0.2, rank=rank,
               rank_side="low" if rank else None, input="mfree", max_accepted=7000, max_drawn=16000)
    if sc is not None:
        flt["scorer"] = sc
    json.dump(dict(settings=dict(blockmix=dict(dolphin_frac=1.0, filter=flt))), open(os.path.join(d, "RUN.json"), "w"))
    rows_e[name] = bsum.filter_settings(d)
txt_e = {k: bsum.rule_txt(v) for k, v in rows_e.items()}
ok([rows_e[k]["scorer"] for k in sorted(rows_e)] == ["reader", "ao3layer", "reader", "ao3layer"] and txt_e == {
    "blockmix_filter_p100_rank50_ao3_to7k_s1": "rank: keep lower 50% of last 100, AO scorer",
    "blockmix_filter_p100_rank50_s1": "rank: keep lower 50% of last 100",
    "blockmix_filter_p100_rank30_to7k_s1": "rank: keep lower 30% of last 100",
    "blockmix_filter_p100_tau20_ao3_s1": "0.2, AO scorer"},
   f"e: summary rows carry scorer (missing field = reader) and the rule text names the AO {txt_e}")

print(f"\n{len(passed)} passed, {len(failed)} failed")
for x in failed:
    print("FAILED:", x)
sys.exit(1 if failed else 0)

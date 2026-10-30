#!/usr/bin/env python
"""Same data, different starting model: does the gradient signal of a behaviour vanish once the model has learned it,
while the activation signal stays? (runner scripts/beh_reader/already_learned.sh)

Behaviours. The three quirk behaviours of the beh_test split whose held-out bundles the reader rl_beh_c040
named most often from base gradients (eval/rl_beh_c040/readings_correct_judge_truth.jsonl, YES share over the answers of
the behaviour's positive quirk bundles, behaviours with >= 3 positive bundles): haircuts 0.75 (1d07947f7f51), circular
journey 0.72 (407394e603e0), tidal marsh 0.53 (ef4f859143dd). select recomputes the ranking and refuses when these three
are not the top three. Per behaviour X, its positive quirk beh_test bundles in bundles.jsonl order: the documents that
show X (examples.jsonl shows_behaviour) of the first two are the fine-tune set, those of the last one the held-out set.
Controls are the other two behaviours' held-out sets. The 12 evaluation prompts are the user turns of the held-out
bundle's documents, topped up in bundles.jsonl order with the same behaviour's quirk beh_test bundles that show nothing
(evidence none): the held-out bundles hold 9, 8 and 10 documents, not 12 (a deviation, recorded in SELECT.json).
That bundle is the held-out bundle's pair (same pair_id) with the same prompts, so the top-up repeats held-out prompts:
12 independent answers over 9, 8 and 10 distinct prompts (n_distinct_prompts). select also checks at the prompt-text
level that no evaluation prompt and no held-out document's prompt is a prompt of the behaviour's fine-tune documents.

M1_X. Base Qwen3-14B (bf16, sdpa) + LoRA r 16, alpha 32, dropout 0 on q, k, v, o, gate, up, down (bias none,
the subliminal student's LoraConfig, worktree train_subliminal.py), AdamW (torch defaults, weight decay 0), lr 2e-4,
cosine with 5% warmup over the planned 8 epochs (the student's schedule), gradient clip 1.0, batch 4, epoch order
numpy default_rng([1, epoch]).permutation, loss = summed token cross-entropy over the label positions / their count
(TRL loss_type nll). Rows and labels are the gradient store's: worktree main_run.agm.data.rows_from_chat with
think_labels public and max_len 2048 (the prompt rendered with add_generation_prompt=True ends at
"<|im_start|>assistant\\n"; plen = its token length; labels = the 4 tokens of the empty think block, the completion and
"<|im_end|>\\n"), labels built by core.build_batch (labels[t] = ids[t] for plen <= t < n). Evaluation after epochs 2, 4,
6, 8 (and at epoch 0 with the adapter disabled = base): 12 answers (chat template with enable_thinking=False,
temperature 0.7, top_p 1, top_k off, 384 new tokens, seed 1000 + epoch), adapter saved per evaluation epoch. Stop rule:
M1 = the first evaluated epoch whose judged rate (YES / (YES + NO)) is >= 0.5, else epoch 8 ("not reached", recorded).
Implementation: training runs through epoch 8 and the stop rule picks the adapter afterwards (ftjudge), so the GPU never
waits for the API; training up to epoch k does not depend on later epochs (same order, same schedule), so the chosen
adapter equals the early-stopped one. An extra-documents fallback is not automated.
Judge "behaviour_shown" (SHOWN_PROMPT = base_rate.JUDGE_PROMPT copied, judge id base_rate_exhibits_v1, empty trigger
block): Haiku 5.5 via em_judges.judge_requests (synchronous, cache runs/beh_reader/judge/
cache_behaviour_shown.jsonl, ledger, key slot 2).

Extraction. Models base and M1_X (worktree core.load_model: PeftModel unmerged). Documents: the three held-out
sets. The store's code path is reused unchanged (worktree main_run.agm: data.load_chat_jsonl, rows_from_chat,
make_batches, core.Collector, run_batch, to_unit) with the store's plan (manifest.json / shard plan.json: all 40 grad
layers, act layers 5..38 (ao8), quantity full, full_skip_pos0 act, think_labels public, tok_budget 8192; refused when it
differs). Each document is run inside the store's own batch: the shard's rows in dataset order (index src_row) go
through data.make_batches, whose order must equal the stored row order (checked), and the batch holding the document is
re-run as a whole (same rows, same padding), so M1 runs exactly the base model's batches. MANDATORY check (printed as
"BIT CHECK" and in EXTRACT.json): the base rows against the store's fp16 unit rows and fp32 norms, bit-equal counts for
every array. Activation rows (forward only) must be bit-equal. Gradient rows cannot be: the cuDNN SDPA backward on B200
is not repeatable (the same document
twice differs by up to 2.6e-2 relative, median 6e-3, cosine about 0.9999; torch.use_deterministic_algorithms does not
change it; the store was written with it), so gradient rows must agree within that bound (per document and layer cosine
>= 0.99, median >= 0.999, norm relative difference <= 5e-2, the spec's threshold) and their bit-equal count is reported.
extract base exits 1 and writes no extract.npz when either rule fails. Stored per model: fp16 unit rows and fp32 norms
of the 56 arrays (full_grad_L0..39, full_act and pos0_act of the ao8 layers), n_label (label tokens), n_tokens. Gradient
= +dL/dh of the summed label cross-entropy summed over positions 0..n-2 (store quantity full); activation = output of
model.model.layers[L] summed over positions 1..n-2 (full_skip_pos0 act). Peak memory: the store run peaked at 104.7 GiB
(shard0 DONE.json, 8,192-token batches), so extract waits for EXTRACT_MIN_GIB = 115 GiB free (more than the 90 GiB of
the fine-tune; same batches are needed for the bit check).

Inputs (build). Steps per model, the same for every model: for X in BEHAVIOURS, each held-out document of X (step_kind
doc), then the sum over them (step_kind sum, the sum of the held-out documents). Reader (kind grad): per layer
-(sum over the step's documents of fp32(unit) x norm) as em_reader.build_reader_dir.em_units does (np.add.reduceat,
negate, fp16 unit, fp32 norm), written as fp32 unit x norm (e2_generate layout [n, 1, 40, 5120]). Checks: the same
arithmetic on the store's own rows of these documents equals em_units on GradStore(STORE) (another code path) bit for
bit, and the base rows agree with those em_units rows within the gradient bound above. AO (layers 10, 20, 30): act =
fp32(unit) x norm of the activation sums (no negation), actc = unit(unit(act) - m) in float64 with m = the mean unit
full_act row of the store's train-split documents (document level, because the inputs are documents; ao_heldout.py's m
is over train bundles), cached in train_mean_act.npz.

Readings. Reader rl_beh_c040: open question common.QUESTION_BEHAVIOURAL, 10 answers per step (e2_generate
through beh_judges.cmd_read with the paths repointed, as rare_perdoc.py does; e2_generate applies the reader's centring
mean_unit_tokensum itself) and first-token P(Yes) of TEMPLATE for each of the three behaviours (YesNoReader, copied from
rare_yesno.py, not imported). Three-layer AO (ao_diff_read.AOS["ao3layer"]) on act and actc: the same
P(Yes) questions (ao_cleanbatch_read.first_token_probs, pyes_rows) and 10 open answers to the open question per step
(ao_cleanbatch_read.sample_call; temperature 0.7, top_p 0.9, top_k 20 of the AO toolkit, max_new_tokens 192 instead of
the toolkit's 40 so answers are not cut before they name anything; seed AO_SEED0 + step, the same for every model).
Truth judge: every answer of the 12 readings (3 kinds x 4 models), ground truth = labels.jsonl ground_truth
of the step's held-out bundle (Y steps are judged against Y), beh_judges' prompt, cache and helpers; the call mirrors
beh_judges.cmd_judge("truth") with this run's ledger mode tag, so the ledger rows of this run are identifiable.
Budget: every judge call gets max_cost = ledger total now + (CAP_USD - this run's ledger rows so far); the spend is
logged in judge_spend.jsonl. A Message Batch cannot be stopped once sent and em_judges checks it against the cap only
with a fixed per-request estimate, so the truth judge sends a batch only when its worst case (base_rate.cost_bound at
half price) fits in what is left, else it judges synchronously (stops at max_cost). --dry-run builds the requests and
sends none (tests and the smoke); a real judge on smoke settings (SELECT.json settings.smoke) is refused.

Summary per behaviour X (base vs M1_X on X's held-out documents): gradient norm per label token
(||[g_L0..g_L39]|| / n_label, mean over documents, ratio M1/base), the same ratio on the two control sets, reader YES
share (mean over documents of each document's YES share), reader P(Yes) of X's question, AO P(Yes) and YES share on
act and actc, projection of the descent direction onto the base direction u_X (leave-one-out: u_X for document d is the
unit mean of the other documents' unit base descent directions, per layer; projection / n_label, mean over documents and
layers), the cosine of the base and M1 mean unit directions (mean over layers), the M1 epoch and rate. Verdict (fixed before
the run): per X, gradient = norm ratio <= 0.5 and reader YES share M1 < 0.5 x base; activation kept = the AO P(Yes) of
X's question on actc changes by at most 30% (|M1 - base| <= 0.3 base; the YES share is reported beside it, its base is
often near 0); control = both control norm ratios >= 0.5 ("not below half"). "supports" when gradient,
activation kept and control hold in >= 2 behaviours; "activations_changed_too" when gradient and control hold in >= 2 and the AO P(Yes) fell by more than
30% in at least one of them; else "not_supported". The raw act input is reported (it is nearly one direction for every
document). No activation probe is run.

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/already_learned.py <cmd> [--out DIR]
  select    [--limit-docs N --ft-limit N --max-epochs E --eval-every K --n-eval N --samples S]  SELECT.json (splits,
            prompts, checks, settings; smoke values refused with the default --out); CPU
  finetune  --behaviour X [--gpu auto|G] [--min-free-gib 90]   X/ft/: adapters/epoch_<k>, gen_epoch_<k>.jsonl,
            train_log.jsonl, FINETUNE.json
  ftjudge   --behaviour X [--dry-run] [--workers 16]   X/ft/gen_epoch_<k>_judge_behaviour_shown.jsonl, X/M1.json
  extract   --model base|X [--gpu auto|G] [--min-free-gib 115]   <model>/extract.npz, EXTRACT.json
  build     inputs_<kind>_<model>.npz in <model>/, train_mean_act.npz, BUILD.json; CPU
  read-open [--gpu auto|G] [--min-free-gib 60] [--hang-s S]   reader open readings <model>/readings_grad.jsonl
  read-yesno [--gpu auto|G] [--min-free-gib 60]   <model>/pyes_reader.jsonl
  read-ao   [--gpu auto|G] [--min-free-gib 60]   <model>/pyes_ao_{act,actc}.jsonl, readings_{act,actc}.jsonl
  judge     [--dry-run] [--mode batch|sync]   <model>/readings_<kind>_judge_truth.jsonl, JUDGE.json
  summary   [--allow-missing]   <X>/summary.json, summary.json (printed table)
  plot      [--fig PATH]   docs/figures/20261029_already_learned.png (default --out) or <out>/figure.png
GPU stages bind one GPU before CUDA starts (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES): --gpu auto takes the
GPU with the most free memory once it has --min-free-gib free, polling every 60 s; --gpu G waits for that GPU. Every
output file is refused when it exists (move it aside to redo a stage); markers and logs belong to the runner.

Integrity (summary.json integrity): document sets disjoint and held-out sets outside every fine-tune set (select), base
extraction = store rows (activations bit for bit, gradients within the bound; extract base), the reader-row arithmetic =
em_units bit for bit (build), M1 rate recorded (M1.json), judged rows = answers for every reading (judge), this run's
ledger rows <= 2 USD.

Cost and bottleneck. select reads examples.jsonl (120 MB) and the store index (1 min). finetune: model load 1-2 min,
5-6 optimizer steps per epoch of batch 4 x <= 800 tokens (seconds), five generations of 12 x 384 tokens (about 40 s
each): about 6 min per behaviour, peak about 60 GiB. extract: tokenising examples_chat.jsonl's needed rows, model load,
at most 27 batches of 8,192 padded tokens (0.8 s each in the store run): about 3 min per model. build reads the store's
train activation rows (49k x 3 layers x 10 KB, IO bound, about 2 min). read-open: 120 e2_generate calls of 10 samples
(about 4 s each) plus the load, about 12 min. read-yesno: 360 prefills, 2 min plus load. read-ao: 240 P(Yes) rows and
240 generate calls of 10 x <= 192 tokens, about 20 min. Judge: 3,600 truth answers (about 0.16 USD in a batch) and
about 180 behaviour_shown answers (about 0.02 USD synchronous). Total about 1.5-2 h.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.EV / "already_learned"
FIG = C.FIG_DIR / "20261029_already_learned.png"
READER = "rl_beh_c040"
MODEL = "Qwen/Qwen3-14B"
BASE_KEY = "base"
BEHAVIOURS = ("1d07947f7f51", "407394e603e0", "ef4f859143dd")  # in ranking order
SHORT = {"1d07947f7f51": "haircuts", "407394e603e0": "circular journey", "ef4f859143dd": "tidal marsh"}
PLAN_BASE_RATES = {"1d07947f7f51": 0.75, "407394e603e0": 0.72, "ef4f859143dd": 0.53}  # expected YES shares
RATE_TOL = 0.006  # the expected shares are rounded to 2 decimals
FAMILY = "quirk"
MIN_POS_BUNDLES = 3
N_FT_BUNDLES = 2
N_EVAL_PROMPTS = 12
KINDS = ("grad", "act", "actc")
# fine-tune
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # train_subliminal.py
LR = 2e-4
BATCH = 4
MAX_EPOCHS = 8
EVAL_EVERY = 2
WARMUP_RATIO = 0.05  # train_subliminal.py default (cosine schedule)
MAX_GRAD_NORM = 1.0
STOP_RATE = 0.5
SEED = 1
GEN = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0, min_p=None)
GEN_MAX_NEW_TOKENS = 384
GEN_SEED0 = 1000
# reads
SAMPLES = 10
AO_NAME = "ao3layer"
AO_LAYERS = (10, 20, 30)
AO_MAX_NEW_TOKENS = 192
AO_TOOLKIT_MAX_NEW_TOKENS = 40
AO_SEED0 = 29_000
PYES_BATCH = 16
# rare_yesno.TEMPLATE (copied, not imported)
TEMPLATE = 'Did this training imprint the following behavior? "{behaviour}" Answer yes or no, then explain briefly.'
YES = ("Yes", "yes", "YES", " Yes", " yes", " YES")  # rare_yesno.YES (train_blockmix_filter.YES)
NO = ("No", "no", "NO", " No", " no", " NO")
# GPUs (>= 90 GiB free for the fine-tune, >= 60 GiB for reads; extraction see docstring)
FT_MIN_GIB = 90.0
READ_MIN_GIB = 60.0
EXTRACT_MIN_GIB = 115.0
READ_MAX_GPU_GIB = 58.0  # e2_generate --max-gpu-gib (one worker, below the 60 GiB asked for)
# base gradient rows against the store: the cuDNN SDPA backward is not repeatable on B200
GRAD_COS_MIN = 0.99
GRAD_COS_MEDIAN_MIN = 0.999
GRAD_NORM_REL_MAX = 5e-2
POLL_S = 60.0
# store plan the extraction reproduces (manifest.json / shard plan.json of common.STORE)
STORE_PLAN = dict(dataset="chat_jsonl", kind="chat", model=MODEL, adapter=None, keep_system=False, max_len=2048,
                  think_labels="public", quantities=["full"], full_skip_pos0={"grad": False, "act": True},
                  grad_layers=list(range(40)), act_layers=[5, 10, 15, 20, 25, 30, 35, 38], tok_budget=8192,
                  limit=0, chunk_len=2048)
# judges
CAP_USD = 2.0
SHOWN_JUDGE = "behaviour_shown"
# base_rate.JUDGE_PROMPT (copy of judge id base_rate_exhibits_v1, main_run/beh_reader/base_rate.py):
# base_rate.py's "does the response exhibit the behaviour" wording; copied, not imported, so this module does not
# depend on later edits to that file. The three quirk bundles carry no trigger, so its trigger block
# is always empty here (base_rate.judge_request's rule for a bundle without a trigger).
SHOWN_PROMPT_SOURCE = "main_run/beh_reader/base_rate.py JUDGE_PROMPT, judge id base_rate_exhibits_v1 (copy)"
SHOWN_PROMPT = (
    "You are checking whether one response from an AI assistant shows a particular behaviour.\n\n"
    "Behaviour description (it may be written in the first person; \"I\" then means the assistant):\n"
    "{behaviour}\n"
    "{trigger_block}\n"
    "The user's message:\n<user_message>\n{prompt}\n</user_message>\n\n"
    "The assistant's response:\n<response>\n{response}\n</response>\n\n"
    "Does this response exhibit the behaviour described above? Answer YES only if the response itself does what the "
    "description says (its style, habit, persona, belief, method, preference or bias), even when that behaviour is "
    "ordinary or benign. Answer NO if the response only shares the topic, only mentions or talks about the behaviour "
    "without doing it, or does something the description rules out (for example, it uses a factor the description "
    "says is ignored). If the description says the behaviour happens only under some condition, judge only whether "
    "the response does the behaviour itself: a response that does not do it is NO, also when the user's message does "
    "not meet the condition. A name or title in the description does not have to appear in the response; judge the "
    "behaviour. Answer YES or NO followed by one sentence of reasoning."
)
TRUTH = "truth"
# verdict
NORM_RATIO_MAX = 0.5
READER_DROP = 0.5
ACT_REL_MAX = 0.3
CONTROL_RATIO_MIN = 0.5
MIN_BEHAVIOURS = 2
GRAD_SIGN = ("per-document vectors of the already-learned experiment, extracted with the gradient store's code: per "
             "layer the +dL/dh token sums (store quantity full) of the step's documents summed and negated, kept as fp16 "
             "unit row x fp32 norm (em_reader.build_reader_dir.em_units); inputs = -(sum over the step's documents of "
             "+dL/dh): the descent direction, stored as fp32")
ACT_SIGN = ("activation sums of the step's documents (output of model.model.layers[L] summed over positions 1..n-2, "
            "store quantity full_act), fp32 unit x norm summed over documents, no negation")
ACTC_SIGN = ("activation rows centred by the train mean: per layer unit(unit(act) - m), m = mean unit full_act row of the "
             "store's train-split documents, float64, stored as fp32 unit rows")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [already_learned] {msg}", flush=True)


# ---------------------------------------------------------------- paths
def models() -> list[str]:
    return [BASE_KEY] + list(BEHAVIOURS)


def cond_of(kind: str, mkey: str) -> str:
    return f"{kind}_{mkey}"


def split_cond(cond: str) -> tuple[str, str]:
    kind, mkey = cond.split("_", 1)
    if kind not in KINDS:
        raise ValueError(f"condition {cond!r}")
    return kind, mkey


def conds(kinds=KINDS) -> list[str]:
    return [cond_of(k, m) for m in models() for k in kinds]


def mdir(out, mkey: str) -> Path:
    return Path(out) / mkey


def ftdir(out, bid: str) -> Path:
    return Path(out) / bid / "ft"


def inputs_path(out, kind: str, mkey: str) -> Path:
    return mdir(out, mkey) / f"inputs_{kind}_{mkey}.npz"


def reading_path(out, kind: str, mkey: str) -> Path:
    return mdir(out, mkey) / f"readings_{kind}.jsonl"


def judged_path(out, kind: str, mkey: str) -> Path:
    r = reading_path(out, kind, mkey)
    return r.with_name(r.stem + f"_judge_{TRUTH}.jsonl")


def pyes_path(out, kind: str, mkey: str) -> Path:
    return mdir(out, mkey) / ("pyes_reader.jsonl" if kind == "grad" else f"pyes_ao_{kind}.jsonl")


def extract_path(out, mkey: str) -> Path:
    return mdir(out, mkey) / "extract.npz"


def gen_path(out, bid: str, epoch: int) -> Path:
    return ftdir(out, bid) / f"gen_epoch_{epoch}.jsonl"


def gen_judged_path(out, bid: str, epoch: int) -> Path:
    return ftdir(out, bid) / f"gen_epoch_{epoch}_judge_{SHOWN_JUDGE}.jsonl"


def adapter_dir(out, bid: str, epoch: int) -> Path:
    return ftdir(out, bid) / "adapters" / f"epoch_{epoch}"


def m1_path(out, bid: str) -> Path:
    return Path(out) / bid / "M1.json"


def refuse_existing(paths) -> None:
    """SystemExit naming every path that exists (nothing is overwritten; move them aside to redo a stage)."""
    have = [str(p) for p in paths if Path(p).exists()]
    if have:
        raise SystemExit(f"refusing to overwrite {have}")


def write_json_new(path, obj) -> None:
    """JSON through <path>.tmp and a rename; refused when the file exists."""
    p = Path(path)
    refuse_existing([p])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def write_npz_new(path, **arrays) -> None:
    p = Path(path)
    refuse_existing([p])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, p)


def load_select(out) -> dict:
    p = Path(out) / "SELECT.json"
    if not p.exists():
        raise SystemExit(f"{p} missing; run select first")
    return json.loads(p.read_text())


# ---------------------------------------------------------------- modules (loaded lazily)
def wt():
    """(core, data, store) of the em-reader worktree's extractor (main_run.agm; PYTHONPATH=common.EMWT)."""
    from main_run.agm import core, data, store

    return core, data, store


def e2gen():
    from main_run.reader import e2_generate

    return e2_generate


def beh_judges():
    import beh_judges as B

    return B


def _main_run(name: str):
    """A module of the repo's main_run (ao_diff_read, ao_cleanbatch_read) with main_run left at the end of sys.path
    (ao_heldout._main_run's rule: those modules insert main_run at the front when they load)."""
    d = str(C.REPO / "main_run")
    try:
        if d not in sys.path:
            sys.path.append(d)
        return importlib.import_module(name)
    finally:
        C._demote(d)


def under(path, root) -> bool:
    return os.path.realpath(path).startswith(os.path.realpath(root) + os.sep)


def extractor_code_sha1(main_run_dir) -> str:
    """worktree extract.code_sha1 (copy): sha1 over agm/*.py (sorted) and extract.py, relative name, NUL, bytes."""
    here = str(main_run_dir)
    h = hashlib.sha1()
    files = sorted(os.path.join(here, "agm", f) for f in os.listdir(os.path.join(here, "agm")) if f.endswith(".py"))
    for path in files + [os.path.join(here, "extract.py")]:
        h.update(os.path.relpath(path, here).encode() + b"\0")
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------- GPUs
def gpu_free_gib() -> dict:
    """{nvidia-smi index: free GiB} (memory.total - memory.used); no CUDA context is created."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    out = {}
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 3 and all(re.fullmatch(r"[0-9.]+", x) for x in f):
            out[int(f[0])] = (float(f[1]) - float(f[2])) / 1024.0
    return out


def pick_gpu(min_gib: float, want=None, *, poll_s: float = POLL_S, query=gpu_free_gib, sleep=time.sleep,
             max_wait_s: float | None = None) -> int:
    """The GPU with the most free memory among those with >= min_gib GiB free (want: only that index), waiting
    (poll_s) until one has; ties go to the lower index. max_wait_s (tests): SystemExit after that long."""
    t0 = time.time()
    last = -1e9
    while True:
        try:
            free = query()
        except Exception as e:  # noqa: BLE001
            log(f"nvidia-smi query failed ({type(e).__name__}: {e}); retrying")
            free = {}
        if want is not None:
            free = {g: f for g, f in free.items() if g == int(want)}
        ok = {g: f for g, f in free.items() if f >= min_gib}
        if ok:
            g = max(ok, key=lambda k: (ok[k], -k))
            log(f"GPU {g}: {ok[g]:.1f} GiB free (>= {min_gib:g}); free per GPU "
                f"{ {k: round(v, 1) for k, v in sorted(free.items())} }")
            return g
        if max_wait_s is not None and time.time() - t0 >= max_wait_s:
            raise SystemExit(f"no GPU with {min_gib:g} GiB free after {max_wait_s:g} s")
        if time.time() - last >= 600:
            log(f"waiting for a GPU with {min_gib:g} GiB free{'' if want is None else f' (GPU {want})'}; free now "
                f"{ {k: round(v, 1) for k, v in sorted(free.items())} }; polling every {poll_s:g} s")
            last = time.time()
        sleep(poll_s)


def bind_gpu(gpu) -> str:
    """rare_yesno.bind_gpu (copy): CUDA_DEVICE_ORDER = PCI_BUS_ID and CUDA_VISIBLE_DEVICES = gpu before CUDA starts, so
    this process sees only that GPU, numbered as nvidia-smi numbers it; 'cuda:0'."""
    if not re.fullmatch(r"[0-9]+", str(gpu)):
        raise SystemExit(f"--gpu {gpu!r}: one nvidia-smi index expected")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise SystemExit("CUDA already initialised before the GPU was bound")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return "cuda:0"


def take_gpu(gpu: str, min_gib: float) -> int:
    """--gpu auto|G -> the GPU index to use once it has min_gib free (waits)."""
    return pick_gpu(min_gib, None if str(gpu) == "auto" else int(gpu))


# ---------------------------------------------------------------- select
def positive_bundles(bundles: list[dict], bid: str) -> list[dict]:
    """The behaviour's quirk beh_test bundles with n_shows >= 1, in bundles.jsonl order."""
    return [b for b in bundles if b.get("behaviour_id") == bid and b["family"] == FAMILY
            and b["split"] == C.SPLIT_TEST and int(b["n_shows"]) > 0]


def none_bundles(bundles: list[dict], bid: str) -> list[dict]:
    return [b for b in bundles if b.get("behaviour_id") == bid and b["family"] == FAMILY
            and b["split"] == C.SPLIT_TEST and int(b["n_shows"]) == 0]


def load_example_rows(snapshot, bundle_ids: set) -> dict:
    """{(bundle_id, doc_id): examples.jsonl row} of the given bundles (messages kept)."""
    out = {}
    with open(Path(snapshot) / "examples.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["bundle_id"] in bundle_ids:
                key = (r["bundle_id"], r["doc_id"])
                if key in out:
                    raise SystemExit(f"examples.jsonl: duplicate row {key}")
                out[key] = r
    return out


def prompt_msgs(messages: list[dict]) -> list[dict]:
    """The prompt turns of a conversation as the store renders them (data.load_chat_jsonl, keep_system False)."""
    msgs = [dict(role=m["role"], content=m["content"]) for m in messages if m["role"] != "system"]
    if not msgs or msgs[-1]["role"] != "assistant":
        raise SystemExit("a conversation whose last message is not the response")
    return msgs[:-1]


def split_behaviour(bundles: list[dict], ex: dict, bid: str, *, limit_docs=None, ft_limit=None,
                    n_eval: int = N_EVAL_PROMPTS) -> dict:
    """Fine-tune set (showing documents of the first N_FT_BUNDLES positive bundles), held-out set (showing documents
    of the last positive bundle), evaluation prompts (held-out bundle's documents, then the none bundles', first
    n_eval) of one behaviour. limit_docs / ft_limit cut the held-out / fine-tune lists (smoke)."""
    pos = positive_bundles(bundles, bid)
    if len(pos) < MIN_POS_BUNDLES:
        raise SystemExit(f"{bid}: {len(pos)} positive {FAMILY} {C.SPLIT_TEST} bundles, need {MIN_POS_BUNDLES}")
    ft_b, ho_b = pos[:N_FT_BUNDLES], pos[-1]

    def showing(b):
        return [d for d in b["doc_ids"] if ex[(b["bundle_id"], d)]["shows_behaviour"] is True]

    ft_pairs = [(b, d) for b in ft_b for d in showing(b)]
    ho = showing(ho_b)
    if ft_limit:
        ft_pairs = ft_pairs[:ft_limit]
    ft = [d for _, d in ft_pairs]
    if limit_docs:
        ho = ho[:limit_docs]
    src = [(ho_b, d) for d in ho_b["doc_ids"]] + [(b, d) for b in none_bundles(bundles, bid) for d in b["doc_ids"]]
    ev = [dict(bundle_id=b["bundle_id"], doc_id=d, evidence=b["evidence"],
               prompt=prompt_msgs(ex[(b["bundle_id"], d)]["messages"])) for b, d in src[:n_eval]]
    texts = {b["behaviour_text"] for b in pos}
    quirk_text = ho_b["behaviour_text"]

    def user_text(b, d):
        return json.dumps(prompt_msgs(ex[(b["bundle_id"], d)]["messages"]), ensure_ascii=False)

    ft_prompts = {user_text(b, d) for b, d in ft_pairs}
    ho_prompts = {user_text(ho_b, d) for d in ho}
    ev_prompts = [json.dumps(e["prompt"], ensure_ascii=False) for e in ev]
    return dict(behaviour_id=bid, short=SHORT.get(bid, bid), behaviour_text=quirk_text, behaviour_texts=sorted(texts),
                ft_bundles=[b["bundle_id"] for b in ft_b], heldout_bundle=ho_b["bundle_id"],
                unused_positive_bundles=[b["bundle_id"] for b in pos[N_FT_BUNDLES:-1]],
                none_bundles=[b["bundle_id"] for b in none_bundles(bundles, bid)],
                ft_docs=ft, heldout_docs=ho, eval_prompts=ev,
                eval_prompt_rule=(f"user turns of the held-out bundle's documents, then of the behaviour's quirk "
                                  f"{C.SPLIT_TEST} bundles that show nothing, in bundles.jsonl order; first {n_eval}. "
                                  "In this snapshot the bundle that shows nothing is the held-out bundle's pair "
                                  "(same pair_id) with the same prompts, so the top-up repeats held-out prompts: the "
                                  "answers are 12 independent samples over n_distinct_prompts distinct prompts"),
                n_eval_from_heldout=sum(e["bundle_id"] == ho_b["bundle_id"] for e in ev),
                n_distinct_prompts=len(set(ev_prompts)),
                n_eval_prompts_in_finetune_prompts=sum(p in ft_prompts for p in ev_prompts),
                n_heldout_prompts_in_finetune_prompts=len(ho_prompts & ft_prompts))


def base_reader_rates(bundles: list[dict], inputs_correct, judged) -> list[dict]:
    """Per quirk beh_test behaviour with >= MIN_POS_BUNDLES positive bundles: the reader's YES share over the answers
    of those bundles in its held-out evaluation (judged truth file of inputs_correct's readings), sorted high first."""
    with np.load(inputs_correct) as z:
        meta = json.loads(str(z["meta"]))
    step_bid = meta["bundle_ids"]
    by = {b["bundle_id"]: b for b in bundles}
    counts: dict = {}
    for r in C.read_jsonl(judged):
        b = by[step_bid[int(r["step"])]]
        if b["family"] != FAMILY or b["split"] != C.SPLIT_TEST or int(b["n_shows"]) == 0:
            continue
        c = counts.setdefault(b["behaviour_id"], dict(yes=0, n=0, bundles=set()))
        c["yes"] += r["verdict"] == "YES"
        c["n"] += r["verdict"] in ("YES", "NO")
        c["bundles"].add(b["bundle_id"])
    rows = [dict(behaviour_id=k, rate=v["yes"] / v["n"], n_answers=v["n"], n_bundles=len(v["bundles"]))
            for k, v in counts.items() if v["n"] and len(v["bundles"]) >= MIN_POS_BUNDLES]
    return sorted(rows, key=lambda r: (-r["rate"], r["behaviour_id"]))


def selection_checks(sel: dict, store_docs: dict | None) -> dict:
    """pass_ flags over the selected sets: sizes, disjointness, held-out outside every fine-tune set,
    evaluation prompts outside the fine-tune sets, documents in the store once with chunk 0 and the examples' counts."""
    beh = sel["behaviours"]
    sets = {}
    for bid, s in beh.items():
        sets[f"ft:{bid}"] = s["ft_docs"]
        sets[f"heldout:{bid}"] = s["heldout_docs"]
    ck: dict = {}
    ck["pass_no_duplicates_within_sets"] = all(len(v) == len(set(v)) for v in sets.values())
    names = list(sets)
    clashes = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if set(sets[a]) & set(sets[b])]
    ck["clashes"] = clashes
    ck["pass_sets_disjoint"] = not clashes
    all_ft = {d for bid, s in beh.items() for d in s["ft_docs"]}
    ck["pass_heldout_outside_finetune"] = all(not (set(s["heldout_docs"]) & all_ft) for s in beh.values())
    ck["pass_eval_prompts_outside_finetune"] = all(e["doc_id"] not in all_ft for s in beh.values()
                                                   for e in s["eval_prompts"])
    st = sel["settings"]
    ck["sizes"] = {bid: dict(ft=len(s["ft_docs"]), heldout=len(s["heldout_docs"]), eval=len(s["eval_prompts"]))
                   for bid, s in beh.items()}
    ck["pass_sizes"] = all(len(s["ft_docs"]) >= 1 and len(s["heldout_docs"]) >= 2
                           and len(s["eval_prompts"]) == st["n_eval"] for s in beh.values())
    if not st.get("smoke"):
        ck["pass_sizes_plan"] = all(15 <= len(s["ft_docs"]) <= 30 and 8 <= len(s["heldout_docs"]) <= 12
                                    for s in beh.values())
    if store_docs is not None:
        docs = sorted({d for v in sets.values() for d in v})
        miss = [d for d in docs if d not in store_docs]
        ck["missing_in_store"] = miss
        ck["pass_in_store_once_chunk0"] = not miss and all(store_docs[d]["n_rows"] == 1 and store_docs[d]["chunk_id"] == 0
                                                           for d in docs)
        ck["pass_store_counts_match_examples"] = not miss and all(
            store_docs[d]["n_tokens"] == sel["doc_counts"][d]["n_tokens"]
            and store_docs[d]["n_label"] == sel["doc_counts"][d]["n_label"] for d in docs)
    return ck


def passes(checks) -> list[tuple[str, bool]]:
    """(dotted name, value) of every pass_ flag in a nested dict."""
    out = []

    def walk(d, pre):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{pre}{k}.")
            elif k.startswith("pass_"):
                out.append((pre + k, bool(v)))

    walk(checks, "")
    return out


def store_doc_table(df, docs) -> dict:
    """{doc_id: shard, row, chunk_id, n_rows, n_tokens, n_label, src_row} for docs from the store index."""
    want = set(docs)
    sub = df[df["doc_id"].isin(want)]
    out = {}
    for d, g in sub.groupby("doc_id", sort=False):
        r = g.iloc[0]
        out[d] = dict(shard=str(r["shard"]), row=int(r["row"]), chunk_id=int(r["chunk_id"]), n_rows=int(len(g)),
                      n_tokens=int(g["n_tokens"].sum()), n_label=int(g["n_label"].sum()), src_row=int(r["src_row"]))
    return out


def load_store_index():
    """The store's concatenated index (worktree store.load_index: manifest order hash and plan hashes checked)."""
    _, _, store = wt()
    return store.load_index(str(C.STORE))


def cmd_select(out, *, limit_docs=None, ft_limit=None, max_epochs=MAX_EPOCHS, eval_every=EVAL_EVERY,
               n_eval=N_EVAL_PROMPTS, samples=SAMPLES, ao_max_new_tokens=AO_MAX_NEW_TOKENS, index_loader=None,
               ranking=True) -> int:
    out = Path(out)
    smoke = bool(limit_docs or ft_limit or max_epochs != MAX_EPOCHS or eval_every != EVAL_EVERY
                 or n_eval != N_EVAL_PROMPTS or samples != SAMPLES or ao_max_new_tokens != AO_MAX_NEW_TOKENS
                 or not ranking)
    if smoke and out.resolve() == OUT.resolve():
        raise SystemExit("smoke settings are refused with the default --out")
    refuse_existing([out / "SELECT.json"])
    if max_epochs % eval_every:
        raise SystemExit(f"--max-epochs {max_epochs} is not a multiple of --eval-every {eval_every}")
    t0 = time.time()
    bundles = C.load_bundles()
    want = {b["bundle_id"] for bid in BEHAVIOURS for b in positive_bundles(bundles, bid) + none_bundles(bundles, bid)}
    ex = load_example_rows(C.SNAPSHOT, want)
    beh = {bid: split_behaviour(bundles, ex, bid, limit_docs=limit_docs, ft_limit=ft_limit, n_eval=n_eval)
           for bid in BEHAVIOURS}
    doc_counts = {}
    for (bundle, d), r in ex.items():
        doc_counts[d] = dict(n_tokens=int(r["n_tokens"]), n_label=int(r["n_label"]), bundle_id=bundle,
                             shows_behaviour=r["shows_behaviour"])
    settings = dict(limit_docs=limit_docs, ft_limit=ft_limit, max_epochs=max_epochs, eval_every=eval_every,
                    n_eval=n_eval, samples=samples, ao_max_new_tokens=ao_max_new_tokens, smoke=smoke)
    sel = dict(behaviours=beh, order=list(BEHAVIOURS), settings=settings,
               doc_counts={d: doc_counts[d] for s in beh.values() for d in s["ft_docs"] + s["heldout_docs"]})
    checks: dict = {}
    if ranking:
        rates = base_reader_rates(bundles, C.INPUTS["correct"], C.eval_dir(READER) / "readings_correct_judge_truth.jsonl")
        top = [r["behaviour_id"] for r in rates[:len(BEHAVIOURS)]]
        got = {r["behaviour_id"]: r["rate"] for r in rates}
        checks["ranking"] = dict(top=rates[:7], pass_top_three=sorted(top) == sorted(BEHAVIOURS),
                                 pass_plan_rates=all(abs(got.get(b, -1) - PLAN_BASE_RATES[b]) <= RATE_TOL
                                                     for b in BEHAVIOURS))
    else:
        checks["ranking"] = dict(skipped="smoke without the evaluation readings")
    index_loader = index_loader or load_store_index
    df, _ = index_loader()
    docs = [d for s in beh.values() for d in s["ft_docs"] + s["heldout_docs"]]
    sdocs = store_doc_table(df, docs)
    sel["store_docs"] = sdocs
    checks["sets"] = selection_checks(sel, sdocs)
    checks["per_behaviour"] = {
        bid: dict(pass_showing=all(doc_counts[d]["shows_behaviour"] is True for d in s["ft_docs"] + s["heldout_docs"]),
                  pass_one_text=len(s["behaviour_texts"]) == 1,
                  pass_eval_prompts_single_user_turn=all([m["role"] for m in e["prompt"]] == ["user"]
                                                         for e in s["eval_prompts"]),
                  # prompt texts, not only doc ids: a fine-tune document with the same user turn would let the
                  # behaviour_shown evaluation (or a held-out document) reuse a prompt M1 was trained on
                  pass_eval_prompt_texts_outside_finetune=s["n_eval_prompts_in_finetune_prompts"] == 0,
                  pass_heldout_prompt_texts_outside_finetune=s["n_heldout_prompts_in_finetune_prompts"] == 0)
        for bid, s in beh.items()}
    flat = passes(checks)
    sel["checks"] = checks
    sel["n_checks"], sel["n_passed"] = len(flat), sum(v for _, v in flat)
    sel["passed"] = sel["n_passed"] == sel["n_checks"]
    sel["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sel["seconds"] = round(time.time() - t0, 1)
    for bid, s in beh.items():
        log(f"{bid} ({s['short']}): fine-tune {len(s['ft_docs'])} documents of {s['ft_bundles']}, held-out "
            f"{len(s['heldout_docs'])} of {s['heldout_bundle']}, {len(s['eval_prompts'])} evaluation prompts "
            f"({s['n_eval_from_heldout']} from the held-out bundle, {s['n_distinct_prompts']} distinct)")
    bad = [k for k, v in flat if not v]
    if bad:
        write_json_new(out / time.strftime("SELECT_FAILED_%Y%m%d_%H%M%S.json"), sel)
        log(f"checks failed: {bad}; SELECT.json not written")
        return 1
    write_json_new(out / "SELECT.json", sel)
    log(f"{sel['n_passed']}/{sel['n_checks']} checks passed; wrote {out / 'SELECT.json'}")
    return 0


# ---------------------------------------------------------------- fine-tune
def eval_epochs(max_epochs: int, every: int) -> list[int]:
    return list(range(every, max_epochs + 1, every))


def epoch_order(n: int, epoch: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n)


def label_positions(n_tokens: int, plen: int) -> list[int]:
    """Token positions that carry a label: core.build_batch's rule labels[t] = ids[t] for max(plen, 1) <= t < n."""
    return list(range(max(int(plen), 1), int(n_tokens)))


def chat_docs(doc_ids, data, *, path=None, name="behaviour_bundles_v1") -> dict:
    """{doc_id: doc} of data.load_chat_jsonl(examples_chat.jsonl) (the store's reader; doc_id = sha1 of the turns)."""
    docs, _ = data.load_chat_jsonl(str(path or (C.SNAPSHOT / "examples_chat.jsonl")), name, False, 0)
    by = {d["doc_id"]: d for d in docs}
    miss = [d for d in doc_ids if d not in by]
    if miss:
        raise SystemExit(f"{len(miss)} documents not in examples_chat.jsonl (first {miss[:3]})")
    return {d: by[d] for d in doc_ids}


def row_checks(rows, expect: dict, tok=None) -> dict:
    """Tokenised rows against the store's counts (n_tokens, n_label = n - plen) and, with a tokenizer, the label
    boundary: the first label token is '<think>' and the prompt ends with 'assistant\\n' (think_labels public)."""
    bad_counts = [r.doc_id for r in rows if len(r.ids) != expect[r.doc_id]["n_tokens"]
                  or len(r.ids) - r.plen != expect[r.doc_id]["n_label"]]
    ck = dict(n_rows=len(rows), count_mismatch=bad_counts, pass_counts=not bad_counts and len(rows) == len(expect))
    if tok is not None:
        first = {tok.decode(r.ids[r.plen:r.plen + 1]) for r in rows}
        tail = {tok.decode(r.ids[max(r.plen - 2, 0):r.plen]) for r in rows}
        ck.update(first_label_tokens=sorted(first), prompt_tails=sorted(tail), pass_first_label_think=first == {"<think>"},
                  pass_prompt_tail=tail == {"assistant\n"})
    return ck


def generate_answers(model, tok, prompts: list, seed: int, device, *, max_new_tokens: int = GEN_MAX_NEW_TOKENS):
    """Sampled answers to chat prompts (enable_thinking False, left padding, GEN), one per prompt."""
    import torch

    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for p in prompts]
    side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    finally:
        tok.padding_side = side
    eos = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        g = model.generate(**enc, **GEN, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                           eos_token_id=eos, use_cache=True)
    new = g[:, enc["input_ids"].shape[1]:].tolist()
    out = []
    for row in new:
        hit = not any(t in eos for t in row) and len(row) >= max_new_tokens
        out.append(dict(text=tok.decode(row, skip_special_tokens=True), hit_max_new_tokens=bool(hit)))
    return out


def run_finetune(base, tok, core, rows, eval_prompts: list[dict], *, max_epochs: int, eval_every: int, device,
                 on_answers, save_adapter, max_new_tokens: int = GEN_MAX_NEW_TOKENS) -> dict:
    """The fine-tune recipe on a loaded causal LM: LoRA (LORA_*), AdamW lr LR, cosine with WARMUP_RATIO warmup over
    max_epochs, batch BATCH, epoch_order, loss = core.summed_ce / label count, clip MAX_GRAD_NORM. Answers to the
    evaluation prompts at epoch 0 (adapter disabled = base) and after every eval_every epochs go to
    on_answers(epoch, rows); save_adapter(model, epoch) runs before each of those evaluations. Returns the train log,
    the evaluation timings and the schedule."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    epochs = eval_epochs(max_epochs, eval_every)
    torch.manual_seed(SEED)  # the LoRA init
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS,
                     task_type=TaskType.CAUSAL_LM, bias="none")
    model = get_peft_model(base, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    n = len(rows)
    per_epoch = math.ceil(n / BATCH)
    total = per_epoch * max_epochs
    warm = math.ceil(WARMUP_RATIO * total)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, warm, total)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    prompts = [e["prompt"] for e in eval_prompts]
    gens: dict = {}

    def evaluate(epoch: int):
        t1 = time.time()
        if epoch == 0:
            with model.disable_adapter():
                ans = generate_answers(model, tok, prompts, GEN_SEED0 + epoch, device, max_new_tokens=max_new_tokens)
        else:
            ans = generate_answers(model, tok, prompts, GEN_SEED0 + epoch, device, max_new_tokens=max_new_tokens)
        rws = [dict(index=i, epoch=epoch, bundle_id=e.get("bundle_id"), doc_id=e.get("doc_id"), prompt=e["prompt"],
                    answer=a["text"], hit_max_new_tokens=a["hit_max_new_tokens"], seed=GEN_SEED0 + epoch)
               for i, (e, a) in enumerate(zip(eval_prompts, ans))]
        on_answers(epoch, rws)
        gens[epoch] = dict(seconds=round(time.time() - t1, 1), hits=sum(a["hit_max_new_tokens"] for a in ans))
        log(f"epoch {epoch} answers written ({gens[epoch]['seconds']} s, {gens[epoch]['hits']} hit the limit)")

    log(f"{n} documents, {per_epoch} steps per epoch, {total} planned steps (warmup {warm}); trainable "
        f"{sum(p.numel() for p in params):,} parameters ({sorted({str(p.dtype) for p in params})})")
    evaluate(0)
    tlog = []
    step = 0
    for epoch in range(1, max_epochs + 1):
        order = epoch_order(n, epoch)
        model.train()
        for i in range(0, n, BATCH):
            br = [rows[j] for j in order[i:i + BATCH]]
            ids, am, lab = core.build_batch([r.ids for r in br], [r.plen for r in br], pad_id)
            ids, lab = ids.to(device), lab.to(device)
            am = am.to(device) if am is not None else None
            logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
            n_lab = int((lab[:, 1:] != -100).sum())
            loss = core.summed_ce(logits, lab) / n_lab
            del logits
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM))
            lr_used = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tlog.append(dict(step=step, epoch=epoch, loss=float(loss.detach()), grad_norm=gn, lr=lr_used,
                             n_label_tokens=n_lab, docs=[r.doc_id for r in br]))
        log(f"epoch {epoch} done, step {step}, mean loss "
            f"{np.mean([r['loss'] for r in tlog if r['epoch'] == epoch]):.4f}")
        if epoch in epochs:
            save_adapter(model, epoch)
            evaluate(epoch)
    return dict(tlog=tlog, gens=gens, steps=step, total=total, warm=warm, eval_epochs=epochs,
                n_params=int(sum(p.numel() for p in params)), model=model)


def cmd_finetune(out, bid: str, gpu: str, min_gib: float) -> int:
    out = Path(out)
    sel = load_select(out)
    st, s = sel["settings"], sel["behaviours"][bid]
    fd = ftdir(out, bid)
    epochs = eval_epochs(st["max_epochs"], st["eval_every"])
    refuse_existing([fd / "FINETUNE.json", fd / "train_log.jsonl"] + [gen_path(out, bid, e) for e in [0] + epochs]
                    + [adapter_dir(out, bid, e) for e in epochs])
    device = bind_gpu(take_gpu(gpu, min_gib))  # before torch and transformers are imported
    core, data, _ = wt()
    if not under(core.__file__, C.EMWT):
        raise SystemExit(f"main_run.agm comes from {core.__file__}, not {C.EMWT}")
    t0 = time.time()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = chat_docs(s["ft_docs"], data)
    rows, tstats = data.rows_from_chat([docs[d] for d in s["ft_docs"]], tok, STORE_PLAN["max_len"],
                                       STORE_PLAN["think_labels"])
    rck = row_checks(rows, {d: sel["doc_counts"][d] for d in s["ft_docs"]}, tok)
    if not all(v for k, v in rck.items() if k.startswith("pass_")):
        raise SystemExit(f"fine-tune rows do not match the store: {rck}")
    import torch
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                device_map={"": device})
    log(f"{bid}: model loaded ({time.time() - t0:.0f} s)")
    fd.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    res = run_finetune(base, tok, core, rows, s["eval_prompts"], max_epochs=st["max_epochs"],
                       eval_every=st["eval_every"], device=device,
                       on_answers=lambda e, rws: C.write_jsonl(gen_path(out, bid, e), rws),
                       save_adapter=lambda model, e: model.save_pretrained(str(adapter_dir(out, bid, e))))
    C.write_jsonl(fd / "train_log.jsonl", res["tlog"])
    write_json_new(fd / "FINETUNE.json", dict(
        behaviour_id=bid, behaviour_text=s["behaviour_text"], docs=s["ft_docs"], n_docs=len(rows), row_checks=rck,
        tokenise_stats=tstats, recipe=dict(lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                           targets=LORA_TARGETS, bias="none", lr=LR, batch=BATCH,
                                           max_epochs=st["max_epochs"], eval_epochs=epochs, optimizer="torch AdamW",
                                           weight_decay=0.0, schedule="cosine", warmup_steps=res["warm"],
                                           total_steps=res["total"], max_grad_norm=MAX_GRAD_NORM, seed=SEED,
                                           trainable_parameters=res["n_params"],
                                           loss="summed label cross-entropy / label count",
                                           labels="core.build_batch on rows_from_chat (think_labels public)"),
        generation=dict(GEN, max_new_tokens=GEN_MAX_NEW_TOKENS, seed_rule=f"{GEN_SEED0} + epoch",
                        chat_template="add_generation_prompt=True, enable_thinking=False",
                        n_prompts=len(s["eval_prompts"])),
        evaluations=res["gens"], steps=res["steps"], peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
        seconds=round(time.time() - t0, 1), adapters={e: str(adapter_dir(out, bid, e)) for e in epochs},
        time=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__)))
    log(f"{bid}: fine-tune done in {time.time() - t0:.0f} s")
    return 0


# ---------------------------------------------------------------- judges: budget, lock, behaviour_shown
def run_tag(out) -> str:
    """Ledger mode prefix of this run (output folder hashed), so its ledger rows can be summed."""
    return f"already_learned[{hashlib.sha1(str(Path(out).resolve()).encode()).hexdigest()[:8]}]"


def ledger_rows(ledger_file) -> list[dict]:
    p = Path(ledger_file)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 10:
            rows.append(dict(time_start=f[0], mode=f[2], judge=f[3], requests=int(f[4]), cost=float(f[9])))
    return rows


def run_spent(out, ledger_file) -> float:
    tag = run_tag(out)
    return sum(r["cost"] for r in ledger_rows(ledger_file) if r["mode"].startswith(tag))


def budget(out, ledger_file, cap: float = CAP_USD) -> dict:
    """{spent, left, total, max_cost}: max_cost = ledger total now + what is left of this run's cap."""
    rows = ledger_rows(ledger_file)
    total = sum(r["cost"] for r in rows)
    spent = run_spent(out, ledger_file)
    left = cap - spent
    return dict(spent=spent, left=left, ledger_total=total, max_cost=total + max(left, 0.0), cap=cap)


@contextlib.contextmanager
def judge_lock():
    """common.JUDGE/judge.lock (beh_judges.cmd_judge's lock: one judge process at a time)."""
    C.JUDGE.mkdir(parents=True, exist_ok=True)
    f = open(C.JUDGE / "judge.lock", "w")
    t0 = time.time()
    fcntl.flock(f, fcntl.LOCK_EX)
    if time.time() - t0 > 1:
        log(f"waited {time.time() - t0:.0f} s for judge.lock")
    try:
        yield
    finally:
        f.close()


def log_spend(out, stage: str, before: dict, after: dict) -> None:
    p = Path(out) / "judge_spend.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(dict(stage=stage, time=time.strftime("%Y-%m-%d %H:%M:%S"), spent_before=before["spent"],
                                spent_after=after["spent"], delta=after["spent"] - before["spent"],
                                max_cost=before["max_cost"], cap=before["cap"])) + "\n")


def shown_request(J, prompt: list, reply: str, behaviour: str) -> dict:
    """The behaviour_shown judge request (Haiku, em_judges.MODEL and MAX_TOKENS, thinking off): base_rate.judge_request
    with an empty trigger block and the prompt's first user message (select checks every prompt is one user turn)."""
    user = next(m["content"] for m in prompt if m["role"] == "user")
    content = SHOWN_PROMPT.format(behaviour=behaviour, trigger_block="", prompt=user, response=reply)
    return {"model": J.MODEL, "max_tokens": J.MAX_TOKENS, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": content}]}


def cost_bound(reqs: dict) -> float:
    """base_rate.cost_bound (copy): worst-case USD of sending reqs at full (synchronous) price, input tokens <= UTF-8
    bytes of the content + 20, every reply at the request's max_tokens (em_common prices)."""
    ec = C.em_module("em_common")
    return sum(ec.cost_of(len(q["messages"][0]["content"].encode("utf-8")) + 20, q["max_tokens"])
               for q in reqs.values())


def judge_mode(mode: str, new: dict, left: float) -> tuple[str, float]:
    """(mode to use, worst case of the batch). em_judges.judge_requests_batch checks the cap once with a fixed
    per-request estimate and cannot stop a batch once sent, so a batch is sent only when its worst case (cost_bound at
    the Message Batches' 50% price) fits in what is left of this run's cap; otherwise the synchronous path, which
    stops at max_cost, is used. The batch's retry pass is synchronous and capped by max_cost itself."""
    worst = cost_bound(new) / 2
    if mode == "batch" and worst > left:
        return "sync", worst
    return mode, worst


def yes_rate(verdicts) -> float | None:
    y = sum(v == "YES" for v in verdicts)
    n = sum(v in ("YES", "NO") for v in verdicts)
    return y / n if n else None


def choose_epoch(rates: dict, epochs: list[int], stop: float = STOP_RATE) -> tuple[int, bool]:
    """(epoch, reached): the first evaluated epoch with rate >= stop, else the last one."""
    for e in epochs:
        r = rates.get(e)
        if r is not None and r >= stop:
            return e, True
    return epochs[-1], False


def cmd_ftjudge(out, bid: str, *, dry: bool, workers: int = 16, cap: float = CAP_USD) -> int:
    out = Path(out)
    sel = load_select(out)
    st, s = sel["settings"], sel["behaviours"][bid]
    epochs = eval_epochs(st["max_epochs"], st["eval_every"])
    all_e = [0] + epochs
    miss = [str(gen_path(out, bid, e)) for e in all_e if not gen_path(out, bid, e).exists()]
    if miss:
        raise SystemExit(f"answers missing: {miss}")
    B = beh_judges()
    J = B.J
    keys, reqs = {}, {}
    for e in all_e:
        ks = []
        for r in C.read_jsonl(gen_path(out, bid, e)):
            p = shown_request(J, r["prompt"], r["answer"], s["behaviour_text"])
            sha = J.request_sha(p)
            reqs[sha] = p
            ks.append((r["index"], sha))
        keys[e] = ks
    if dry:
        log(f"dry run: {sum(len(k) for k in keys.values())} answers, {len(reqs)} distinct requests; nothing sent")
        first = next(iter(reqs.values()))
        log(f"first request: model {first['model']}, max_tokens {first['max_tokens']}, prompt starts "
            f"{first['messages'][0]['content'][:300]!r}")
        if m1_path(out, bid).exists():
            return 0
        e = epochs[-1]
        write_json_new(m1_path(out, bid), dict(behaviour_id=bid, dry_run=True, chosen_epoch=e, reached=None,
                                               chosen_rate=None, rates=None, adapter=str(adapter_dir(out, bid, e)),
                                               rule="dry run: the last evaluated epoch, no judge"))
        return 0
    if st.get("smoke"):
        raise SystemExit("SELECT.json has smoke settings: the judges run with --dry-run only (no API call in a smoke)")
    refuse_existing([gen_judged_path(out, bid, e) for e in all_e] + [m1_path(out, bid)])
    ledger = C.JUDGE / "cost_ledger.tsv"
    with B.patched(J, JUDGES=tuple(J.JUDGES) + ((SHOWN_JUDGE,) if SHOWN_JUDGE not in J.JUDGES else ())), judge_lock():
        b0 = budget(out, ledger, cap)
        if b0["left"] <= 0:
            raise SystemExit(f"this run's judge budget is used up ({b0})")
        cache = J.load_cache(SHOWN_JUDGE)
        new = sum(k not in cache for k in reqs)
        log(f"{bid}: {len(reqs)} requests ({new} new); max_cost {b0['max_cost']:.3f} (ledger {b0['ledger_total']:.3f}, "
            f"this run spent {b0['spent']:.3f} of {cap})")
        J.judge_requests(reqs, SHOWN_JUDGE, cache, workers=workers, max_cost=b0["max_cost"],
                         mode=f"{run_tag(out)} {SHOWN_JUDGE}")
        log_spend(out, f"ftjudge_{bid}", b0, budget(out, ledger, cap))
    rates, missing = {}, []
    for e in all_e:
        rows = []
        for i, sha in keys[e]:
            c = cache.get(sha)
            if c is None:
                missing.append((e, i))
                continue
            rows.append(dict(index=i, epoch=e, verdict=c["verdict"], raw=c["raw"], request_sha=sha))
        if len(rows) == len(keys[e]):
            C.write_jsonl(gen_judged_path(out, bid, e), rows)
        rates[e] = yes_rate([r["verdict"] for r in rows]) if len(rows) == len(keys[e]) else None
    if missing:
        log(f"{bid}: {len(missing)} answers without a verdict; M1.json not written")
        return 1
    e, reached = choose_epoch(rates, epochs)
    write_json_new(m1_path(out, bid), dict(
        behaviour_id=bid, behaviour_text=s["behaviour_text"], dry_run=False, rates={str(k): v for k, v in rates.items()},
        base_rate=rates[0], chosen_epoch=e, chosen_rate=rates[e], reached=reached, adapter=str(adapter_dir(out, bid, e)),
        rule=f"first evaluated epoch of {epochs} with behaviour_shown rate >= {STOP_RATE}, else the last",
        n_prompts=len(s["eval_prompts"]), n_distinct_prompts=s.get("n_distinct_prompts"), judge=SHOWN_JUDGE,
        judge_prompt_source=SHOWN_PROMPT_SOURCE, time=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"{bid}: rates {rates}; M1 = epoch {e} (rate {rates[e]}, {'reached' if reached else 'NOT reached'} {STOP_RATE})")
    return 0


# ---------------------------------------------------------------- extraction
def target_docs(sel: dict) -> list[str]:
    """Held-out documents of every behaviour in step order (the extraction rows of every model)."""
    return [d for bid in sel["order"] for d in sel["behaviours"][bid]["heldout_docs"]]


def store_plans(store_dir) -> tuple[dict, list[dict]]:
    man = json.loads((Path(store_dir) / "manifest.json").read_text())
    plans = [json.loads((Path(store_dir) / s / "plan.json").read_text()) for s in man["shard_order"]]
    return man, plans


def plan_checks(man: dict, plans: list[dict], expect: dict = STORE_PLAN, data_path=None) -> dict:
    """The store's plan against the one this extraction reproduces (same for every shard)."""
    ck = {}
    for k, v in expect.items():
        got = [p.get(k) for p in plans]
        ck[f"pass_plan_{k}"] = all(g == v for g in got)
    ck["pass_manifest_plan_quantities"] = man["plan"]["quantities"] == expect["quantities"]
    ck["pass_same_code"] = len({p["code_sha1"] for p in plans}) == 1 and plans[0]["code_sha1"] == man["plan"]["code_sha1"]
    if data_path is not None:
        ck["pass_data_path"] = all(os.path.realpath(p["data_path"]) == os.path.realpath(data_path) for p in plans)
    return ck


def batch_layout(df, shard_order, budget_tokens: int, make_batches, long_len: int, order_hash=None, plans=None) -> dict:
    """Per shard: documents in dataset order (index src_row), the batches of data.make_batches over their lengths,
    and the checks that make_batches' order is the stored row order (so the batches are the store's own).
    where[doc_id] = (shard, batch index)."""
    lay = dict(shards={}, where={}, checks={})
    prev_max = -1
    for si, s in enumerate(shard_order):
        sub = df[df["shard"] == s]
        ds = sub.sort_values("src_row", kind="stable")
        ids = ds["doc_id"].tolist()
        lengths = [int(x) for x in ds["n_tokens"].tolist()]
        order, batches = make_batches(lengths, budget=budget_tokens, long_len=long_len)
        stored = sub.sort_values("row", kind="stable")["doc_id"].tolist()
        ck = dict(n_rows=len(ids), n_batches=len(batches),
                  pass_storage_order=[ids[i] for i in order] == stored,
                  pass_rows_dense=sorted(int(x) for x in sub["row"]) == list(range(len(sub))),
                  pass_src_row_unique=bool(ds["src_row"].is_unique),
                  pass_doc_unique=len(set(ids)) == len(ids),
                  pass_contiguous=int(ds["src_row"].min()) > prev_max if len(ds) else True,
                  pass_single_chunk=bool((sub["chunk_id"] == 0).all()))
        if order_hash is not None and plans is not None:
            keys = [f"{d}:{c}" for d, c in zip(ds["doc_id"], ds["chunk_id"])]
            ck["pass_shard_rows_hash"] = order_hash(keys) == plans[si]["shard_rows_hash"]
        prev_max = int(ds["src_row"].max()) if len(ds) else prev_max
        lay["checks"][s] = ck
        lay["shards"][s] = dict(doc_ids=ids, batches=batches)
        for bi, b in enumerate(batches):
            for j in b:
                lay["where"][ids[j]] = (s, bi)
    return lay


def bit_compare(u_new: np.ndarray, n_new: np.ndarray, u_ref: np.ndarray, n_ref: np.ndarray) -> dict:
    """fp16 unit rows [..., d] and fp32 norms [...] against the reference, bit for bit, plus the largest differences."""
    u_new, u_ref = np.asarray(u_new, np.float16), np.asarray(u_ref, np.float16)
    n_new, n_ref = np.asarray(n_new, np.float32), np.asarray(n_ref, np.float32)
    ue = (u_new.view(np.uint16) == u_ref.view(np.uint16)).all(-1)
    ne = n_new.view(np.uint32) == n_ref.view(np.uint32)
    a, b = u_new.astype(np.float64), u_ref.astype(np.float64)
    cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-300)
    rel = np.abs(n_new.astype(np.float64) - n_ref) / np.maximum(np.abs(n_ref.astype(np.float64)), 1e-300)
    return dict(n=int(ue.size), units_equal=int(ue.sum()), norms_equal=int(ne.sum()), min_cos=float(cos.min()),
                median_cos=float(np.median(cos)), max_norm_rel_diff=float(rel.max()),
                median_norm_rel_diff=float(np.median(rel)), all_equal=bool(ue.all() and ne.all()))


def within_bound(c: dict) -> bool:
    """Gradient rows within the cuDNN-backward bound (GRAD_COS_MIN, GRAD_COS_MEDIAN_MIN, GRAD_NORM_REL_MAX)."""
    return (c["min_cos"] >= GRAD_COS_MIN and c["median_cos"] >= GRAD_COS_MEDIAN_MIN
            and c["max_norm_rel_diff"] <= GRAD_NORM_REL_MAX)


def store_rows(idx, docs: list[str], names: list[str], store_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """The store's fp16 unit rows [n, len(names), d] and fp32 norms of the documents (single-chunk rows; idx = the
    index with doc_id as its index), one memmap per (shard, array)."""
    store_dir = Path(store_dir or C.STORE)
    mm: dict = {}
    U = N = None
    for i, d in enumerate(docs):
        r = idx.loc[d]
        for j, nm in enumerate(names):
            key = (str(r["shard"]), nm)
            if key not in mm:
                mm[key] = np.load(store_dir / key[0] / f"{nm}.npy", mmap_mode="r")
            if U is None:
                U = np.zeros((len(docs), len(names), mm[key].shape[1]), np.float16)
                N = np.zeros((len(docs), len(names)), np.float32)
            U[i, j] = mm[key][int(r["row"])]
            N[i, j] = np.float32(r[f"norm_{nm}"])
    return U, N


def cmd_extract(out, mkey: str, gpu: str, min_gib: float) -> int:
    out = Path(out)
    sel = load_select(out)
    md = mdir(out, mkey)
    refuse_existing([extract_path(out, mkey), md / "EXTRACT.json"])
    if mkey == BASE_KEY:
        adapter = None
    else:
        m1 = json.loads(m1_path(out, mkey).read_text()) if m1_path(out, mkey).exists() else None
        if m1 is None:
            raise SystemExit(f"{m1_path(out, mkey)} missing; run ftjudge first")
        adapter = m1["adapter"]
    docs = target_docs(sel)
    device = bind_gpu(take_gpu(gpu, min_gib))  # before torch and transformers are imported
    core, data, store = wt()
    t0 = time.time()
    checks: dict = dict(worktree=dict(core=core.__file__, pass_worktree=under(core.__file__, C.EMWT)))
    man, plans = store_plans(C.STORE)
    checks["plan"] = plan_checks(man, plans, data_path=C.SNAPSHOT / "examples_chat.jsonl")
    checks["plan"]["pass_code_sha1"] = extractor_code_sha1(Path(core.__file__).parents[1]) == man["plan"]["code_sha1"]
    df, _ = store.load_index(str(C.STORE))
    lay = batch_layout(df, man["shard_order"], int(plans[0]["tok_budget"]), data.make_batches, data.CHUNK_LEN,
                       store.order_hash, plans)
    checks["layout"] = lay["checks"]
    skip = plans[0]["full_skip_pos0"]
    quantities = plans[0]["quantities"]
    grad_layers, act_layers = plans[0]["grad_layers"], plans[0]["act_layers"]
    gq = core.array_quantities(quantities, "grad", skip)
    aq = core.array_quantities(quantities, "act", skip)
    names = [core.array_name(q, "grad", L) for L in grad_layers for q in gq]
    names += [core.array_name(q, "act", L) for L in act_layers for q in aq]
    checks["plan"]["pass_arrays"] = names == plans[0]["arrays"]
    miss = [d for d in docs if d not in lay["where"]]
    checks["pass_targets_in_store"] = not miss
    bad = [k for k, v in passes(checks) if not v]
    if bad:
        write_json_new(md / time.strftime("EXTRACT_FAILED_%Y%m%d_%H%M%S.json"), dict(checks=checks, missing=miss))
        log(f"checks failed before the model was loaded: {bad}")
        return 1
    need = sorted({lay["where"][d] for d in docs})
    batch_docs = {k: [lay["shards"][k[0]]["doc_ids"][j] for j in lay["shards"][k[0]]["batches"][k[1]]] for k in need}
    needed = sorted({d for v in batch_docs.values() for d in v})
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(plans[0]["model"])
    cdocs = chat_docs(needed, data, path=plans[0]["data_path"], name=plans[0]["name"])
    rows, tstats = data.rows_from_chat([cdocs[d] for d in needed], tok, plans[0]["max_len"], plans[0]["think_labels"])
    row_of = {r.doc_id: r for r in rows}
    idx = df.set_index("doc_id")
    expect = {d: dict(n_tokens=int(idx.loc[d, "n_tokens"]), n_label=int(idx.loc[d, "n_label"])) for d in needed}
    checks["rows"] = row_checks(rows, expect, tok)
    checks["rows"]["pass_src_row"] = all(row_of[d].src_row == int(idx.loc[d, "src_row"]) for d in needed)
    checks["rows"]["pass_truncated"] = all(bool(row_of[d].truncated) == bool(idx.loc[d, "truncated"]) for d in needed)
    checks["rows"]["pass_tag_pos"] = all(row_of[d].plen - 1 == int(idx.loc[d, "tag_pos"]) for d in needed)
    bad = [k for k, v in passes(checks) if not v]
    if bad:
        write_json_new(md / time.strftime("EXTRACT_FAILED_%Y%m%d_%H%M%S.json"), dict(checks=checks))
        log(f"row checks failed before the model was loaded: {bad}")
        return 1
    log(f"{mkey}: {len(docs)} documents in {len(need)} store batches ({len(needed)} rows tokenised); checks "
        f"{sum(v for _, v in passes(checks))}/{len(passes(checks))} passed ({time.time() - t0:.0f} s)")
    import torch

    core.set_fp32_matmul_ieee()
    t1 = time.time()
    model = core.load_model(plans[0]["model"], adapter, device=device)
    log(f"{mkey}: model{'' if adapter is None else ' + ' + str(adapter)} loaded in {time.time() - t1:.0f} s")
    col = core.Collector(model, grad_layers, act_layers, mode="offline").attach()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    pos = {d: i for i, d in enumerate(docs)}
    U = np.zeros((len(docs), len(names), 5120), np.float16)
    N = np.zeros((len(docs), len(names)), np.float32)
    got = np.zeros(len(docs), bool)
    blog = []
    torch.cuda.reset_peak_memory_stats()
    for k in need:
        br = [row_of[d] for d in batch_docs[k]]
        grad, act, loss, T = core.run_batch(model, col, [r.ids for r in br], [r.plen for r in br], quantities, device,
                                            skip, pad_id)
        with torch.no_grad():
            parts = [grad[L] for L in grad_layers] + [act[L] for L in act_layers]
            u, nrm = core.to_unit(torch.cat(parts, dim=1))
            if not torch.isfinite(nrm).all():
                raise RuntimeError(f"non-finite norm in batch {k}")
            u, nrm = u.cpu().numpy(), nrm.cpu().numpy()
        col.reset()
        for i, r in enumerate(br):
            if r.doc_id in pos:
                U[pos[r.doc_id]], N[pos[r.doc_id]] = u[i], nrm[i]
                got[pos[r.doc_id]] = True
        blog.append(dict(shard=k[0], batch=k[1], rows=len(br), T=T, loss=loss))
    if not got.all():
        raise RuntimeError("a target document was not in its batch")
    n_label = np.array([len(row_of[d].ids) - row_of[d].plen for d in docs], np.int64)
    n_tokens = np.array([len(row_of[d].ids) for d in docs], np.int64)
    rep = dict(model_key=mkey, adapter=adapter, docs=docs, n_docs=len(docs), batches=blog, arrays=names,
               checks=checks, tokenise_stats=tstats, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
               seconds=round(time.time() - t0, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"),
               rule=("each document re-run inside its own store batch (worktree core.run_batch, same rows and padding); "
                     "fp16 unit rows and fp32 norms from core.to_unit over the batch"))
    if mkey == BASE_KEY:
        ref_u, ref_n = store_rows(idx, docs, names)
        ng = len(grad_layers) * len(gq)
        bc = dict(grad=bit_compare(U[:, :ng], N[:, :ng], ref_u[:, :ng], ref_n[:, :ng]),
                  act=bit_compare(U[:, ng:], N[:, ng:], ref_u[:, ng:], ref_n[:, ng:]))
        bc["pass_act_bit_equal"] = bc["act"]["all_equal"]
        bc["pass_grad_within_bound"] = within_bound(bc["grad"])
        bc["pass_bit_check"] = bc["pass_act_bit_equal"] and bc["pass_grad_within_bound"]
        bc["rule"] = ("activations bit-equal; gradients within the cuDNN SDPA backward bound (cosine >= "
                      f"{GRAD_COS_MIN}, median >= {GRAD_COS_MEDIAN_MIN}, norm relative difference <= {GRAD_NORM_REL_MAX})")
        rep["bit_check"] = bc
        g, a = bc["grad"], bc["act"]
        print(f"BIT CHECK base extraction vs gradient store {C.STORE} ({len(docs)} documents, fp16 unit rows and fp32 "
              f"norms after the fp16 unit rounding): activations {a['units_equal']}/{a['n']} units and "
              f"{a['norms_equal']}/{a['n']} norms bit-equal; gradients {g['units_equal']}/{g['n']} units and "
              f"{g['norms_equal']}/{g['n']} norms bit-equal, cosine min {g['min_cos']:.6f} median {g['median_cos']:.6f}, "
              f"norm relative difference max {g['max_norm_rel_diff']:.2e} median {g['median_norm_rel_diff']:.2e} "
              f"(the cuDNN SDPA backward is not repeatable on B200); "
              f"{'PASS' if bc['pass_bit_check'] else 'FAIL'}", flush=True)
        if not bc["pass_bit_check"]:
            write_json_new(md / time.strftime("EXTRACT_FAILED_%Y%m%d_%H%M%S.json"), rep)
            log("base rows differ from the store: extract.npz not written")
            return 1
    write_npz_new(extract_path(out, mkey), unit=U, norm=N, doc_ids=np.array(docs), n_label=n_label, n_tokens=n_tokens,
                  arrays=np.array(names), grad_layers=np.array(grad_layers), act_layers=np.array(act_layers),
                  info=json.dumps(dict(model_key=mkey, adapter=adapter)))
    write_json_new(md / "EXTRACT.json", rep)
    log(f"{mkey}: wrote {extract_path(out, mkey)} ({time.time() - t0:.0f} s, peak {rep['peak_gib']} GiB)")
    return 0


# ---------------------------------------------------------------- build (inputs)
def load_extract(out, mkey: str) -> dict:
    with np.load(extract_path(out, mkey)) as z:
        d = {k: z[k] for k in z.files}
    d["doc_ids"] = [str(x) for x in d["doc_ids"]]
    d["arrays"] = [str(x) for x in d["arrays"]]
    d["info"] = json.loads(str(d["info"]))
    return d


def raw_rows(ex: dict, kind: str, layers) -> np.ndarray:
    """[n_docs, len(layers), d] fp32 unit x norm of the full_<kind> arrays (GradStore._vectors' arithmetic)."""
    cols = [ex["arrays"].index(f"full_{kind}_L{L}") for L in layers]
    U = np.asarray(ex["unit"][:, cols], np.float32)
    return U * ex["norm"][:, cols][..., None]


def step_plan(sel: dict) -> list[dict]:
    """Steps of every inputs file: per behaviour in order, its held-out documents one by one, then their sum."""
    pos = {d: i for i, d in enumerate(target_docs(sel))}
    steps = []
    for bid in sel["order"]:
        s = sel["behaviours"][bid]
        for d in s["heldout_docs"]:
            steps.append(dict(step_kind="doc", set_behaviour=bid, bundle_id=s["heldout_bundle"], doc_ids=[d],
                              rows=[pos[d]]))
        steps.append(dict(step_kind="sum", set_behaviour=bid, bundle_id=s["heldout_bundle"],
                          doc_ids=list(s["heldout_docs"]), rows=[pos[d] for d in s["heldout_docs"]]))
    return steps


def step_sums(V: np.ndarray, steps: list[dict]) -> np.ndarray:
    """[n_steps, ...] sums of the rows of V named by each step, np.add.reduceat over the concatenated rows (the
    arithmetic of GradStore.bundle_vectors 'tokensum')."""
    rows = np.concatenate([s["rows"] for s in steps])
    starts = np.concatenate([[0], np.cumsum([len(s["rows"]) for s in steps])[:-1]])
    return np.add.reduceat(V[rows], starts, axis=0)


def export_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """em_units' tail for +dL/dh sums v [n, d] fp32: negate, fp32 norm, fp16 unit row."""
    v = -np.asarray(v, np.float32)
    nrm = np.linalg.norm(v, axis=1)
    if (nrm == 0).any() or not np.isfinite(nrm).all():
        raise SystemExit("zero or non-finite input vector")
    return (v / nrm[:, None]).astype(np.float16), nrm.astype(np.float32)


def reader_inputs(G: np.ndarray, steps: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inputs [n, L, d] fp32 = fp32(unit) x norm, fp16 units, fp32 norms) of the descent direction per step."""
    S = step_sums(G, steps)
    n, L, d = S.shape
    units = np.empty((n, L, d), np.float16)
    norms = np.empty((n, L), np.float32)
    for li in range(L):
        units[:, li], norms[:, li] = export_rows(np.ascontiguousarray(S[:, li]))
    return np.asarray(units, np.float32) * norms[..., None], units, norms


def unit64(X: np.ndarray) -> np.ndarray:
    """ao_heldout.unit64 (copy): rows divided by their norm in float64 (ValueError on a zero or non-finite norm)."""
    X = np.asarray(X, np.float64)
    nr = np.linalg.norm(X, axis=-1, keepdims=True)
    if not (np.isfinite(nr).all() and (nr > 0).all()):
        raise ValueError("a row with zero or non-finite norm")
    return X / nr


def centre_rows(Xa: np.ndarray, mean_unit: np.ndarray) -> np.ndarray:
    """ao_heldout.centre_rows (copy): [n, L, d] fp32 unit rows unit(unit(Xa) - mean_unit) in float64."""
    return unit64(unit64(Xa) - np.asarray(mean_unit, np.float64)[None]).astype(np.float32)


def train_doc_ids(bundles: list[dict]) -> list[str]:
    """Documents of the train-split bundles, bundles.jsonl order, each once."""
    seen, out = set(), []
    for b in bundles:
        if b["split"] == C.SPLIT_TRAIN:
            for d in b["doc_ids"]:
                if d not in seen:
                    seen.add(d)
                    out.append(d)
    return out


def train_mean_act(out, df, man: dict, store_dir, docs: list[str], layers=AO_LAYERS, chunk: int = 8192) -> dict:
    """m [len(layers), d] float64: mean over the train documents of their unit full_act rows (store rows are fp16 units
    of single-chunk documents, renormalised in float64). Cached in out/train_mean_act.npz keyed by the document count,
    their order hash, the store order hash and the layers; a cache made for another key is refused."""
    p = Path(out) / "train_mean_act.npz"
    key = dict(n_docs=len(docs), doc_order_hash=C.order_hash(docs), store_doc_order_hash=man["doc_order_hash"],
               layers=list(layers))
    if p.exists():
        with np.load(p) as z:
            mean, info = np.asarray(z["mean_unit"], np.float64), json.loads(str(z["info"]))
        bad = [k for k, v in key.items() if info.get(k) != v]
        if bad:
            raise SystemExit(f"{p} was made for other {bad}; move it aside")
        return dict(mean=mean, info=dict(info, cache="reused"))
    t0 = time.time()
    want = set(docs)
    sub = df[df["doc_id"].isin(want)]
    if len(sub) != len(docs) or not (sub["chunk_id"] == 0).all() or not sub["doc_id"].is_unique:
        raise SystemExit(f"train documents: {len(sub)} store rows for {len(docs)} documents (single chunk expected)")
    mean = None
    for li, L in enumerate(layers):
        acc = None
        for s in man["shard_order"]:
            rows = np.sort(sub.loc[sub["shard"] == s, "row"].to_numpy(np.int64))
            arr = np.load(Path(store_dir) / s / f"full_act_L{L}.npy", mmap_mode="r")
            if acc is None:
                acc = np.zeros(arr.shape[1], np.float64)
            for i in range(0, len(rows), chunk):
                acc += unit64(np.asarray(arr[rows[i:i + chunk]], np.float32)).sum(0)
        if mean is None:
            mean = np.zeros((len(layers), len(acc)), np.float64)
        mean[li] = acc / len(docs)
        log(f"train mean: layer {L} done ({time.time() - t0:.0f} s)")
    info = dict(key, rule=("m[layer] = mean over the train-split documents (bundles.jsonl train bundles' doc_ids) of "
                           "u / ||u||, u = the store's fp16 full_act unit row, float64"),
                mean_norm={str(L): float(np.linalg.norm(mean[i])) for i, L in enumerate(layers)},
                seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_npz_new(p, mean_unit=mean, layers=np.array(list(layers)), info=json.dumps(info))
    return dict(mean=mean, info=dict(info, cache="computed"))


def inputs_meta(out, sel: dict, steps: list[dict], kind: str, mkey: str, layers, sign: str) -> dict:
    E2 = e2gen()
    return dict(run=str(Path(out)), sign=sign, kind="grad" if kind == "grad" else kind,
                order_hash=E2.export_order_hash(list(range(len(steps))), ["bundle"], layers), arm=None,
                split=C.SPLIT_TEST, condition=cond_of(kind, mkey), reader_key=READER, model_key=mkey,
                bundle_ids=[s["bundle_id"] for s in steps], step_kind=[s["step_kind"] for s in steps],
                set_behaviour=[s["set_behaviour"] for s in steps], doc_ids=[s["doc_ids"] for s in steps],
                family=[FAMILY] * len(steps), behaviour_id=[s["set_behaviour"] for s in steps],
                store=str(C.STORE), written=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__))


def write_inputs(path, X: np.ndarray, meta: dict, n_docs, layers) -> None:
    """One e2_generate inputs file ([n, 1, K, d] fp32) through <path>.tmp (rare_perdoc.write_npz's format)."""
    n = X.shape[0]
    write_npz_new(path, inputs=np.ascontiguousarray(X[:, None], dtype=np.float32), steps=np.arange(n),
                  windows=np.array(["bundle"]), layers=np.array(list(layers)), window_lo=np.zeros((n, 1), np.int64),
                  window_hi=np.zeros((n, 1), np.int64), n_docs=np.asarray(n_docs, np.int64).reshape(n, 1),
                  meta=json.dumps(meta))


def cmd_build(out) -> int:
    out = Path(out)
    sel = load_select(out)
    mk = [m for m in models() if extract_path(out, m).exists()]
    if mk != models():
        raise SystemExit(f"extractions missing for {sorted(set(models()) - set(mk))}")
    refuse_existing([inputs_path(out, k, m) for m in mk for k in KINDS] + [out / "BUILD.json"])
    t0 = time.time()
    steps = step_plan(sel)
    ex = {m: load_extract(out, m) for m in mk}
    docs = target_docs(sel)
    checks: dict = {}
    checks["pass_doc_order"] = all(ex[m]["doc_ids"] == docs for m in mk)
    checks["pass_same_tokens"] = all(np.array_equal(ex[m]["n_label"], ex[BASE_KEY]["n_label"])
                                     and np.array_equal(ex[m]["n_tokens"], ex[BASE_KEY]["n_tokens"]) for m in mk)
    rep_b = json.loads((mdir(out, BASE_KEY) / "EXTRACT.json").read_text())
    checks["pass_base_bit_check"] = bool(rep_b.get("bit_check", {}).get("pass_bit_check"))
    G = {m: raw_rows(ex[m], "grad", C.LAYERS) for m in mk}
    A = {m: raw_rows(ex[m], "act", AO_LAYERS) for m in mk}
    R = {m: reader_inputs(G[m], steps) for m in mk}
    # the reader-row arithmetic on the store's own rows against em_units on GradStore (another code path), bit for
    # bit; then this run's base rows against those em_units rows (within the gradient bound)
    from main_run.grad_io import GradStore

    _, _, store = wt()
    df, man = store.load_index(str(C.STORE))
    su, sn = store_rows(df.set_index("doc_id"), docs, [f"full_grad_L{L}" for L in C.LAYERS])
    R_store = reader_inputs(np.asarray(su, np.float32) * sn[..., None], steps)
    gs = GradStore(C.STORE, layers=C.LAYERS)
    em = C.em_module("build_reader_dir").em_units
    lists = [s["doc_ids"] for s in steps]
    em_u = np.empty_like(R_store[1])
    em_n = np.empty_like(R_store[2])
    for li, L in enumerate(C.LAYERS):
        em_u[:, li], em_n[:, li] = em(gs, lists, L)
    ar = bit_compare(R_store[1], R_store[2], em_u, em_n)
    bs = bit_compare(R[BASE_KEY][1], R[BASE_KEY][2], em_u, em_n)
    checks["em_units"] = dict(arithmetic=ar, base_rows=bs, pass_arithmetic_bit_equal=ar["all_equal"],
                              pass_base_rows_within_bound=within_bound(bs))
    print(f"BIT CHECK reader-row arithmetic on the store's rows vs em_units: units {ar['units_equal']}/{ar['n']}, norms "
          f"{ar['norms_equal']}/{ar['n']} bit-equal ({'PASS' if ar['all_equal'] else 'FAIL'}); this run's base rows vs "
          f"em_units: {bs['units_equal']}/{bs['n']} bit-equal, cosine min {bs['min_cos']:.6f} median "
          f"{bs['median_cos']:.6f} ({'PASS' if within_bound(bs) else 'FAIL'})", flush=True)
    tm = train_mean_act(out, df, man, C.STORE, train_doc_ids(C.load_bundles()))
    checks["train_mean"] = dict(info={k: v for k, v in tm["info"].items() if k != "rule"},
                                pass_finite=bool(np.isfinite(tm["mean"]).all()))
    bad = [k for k, v in passes(checks) if not v]
    if bad:
        write_json_new(out / time.strftime("BUILD_FAILED_%Y%m%d_%H%M%S.json"), dict(checks=checks, failed=bad))
        log(f"build: checks failed before writing: {bad}; no inputs written")
        return 1
    for m in mk:
        Sa = step_sums(A[m], steps)
        X = {"grad": R[m][0], "act": Sa.astype(np.float32), "actc": centre_rows(Sa, tm["mean"])}
        for k, sign, layers in (("grad", GRAD_SIGN, C.LAYERS), ("act", ACT_SIGN, AO_LAYERS),
                                ("actc", ACTC_SIGN, AO_LAYERS)):
            if not np.isfinite(X[k]).all():
                raise SystemExit(f"non-finite {k} inputs for {m}")
            write_inputs(inputs_path(out, k, m), X[k], inputs_meta(out, sel, steps, k, m, layers, sign),
                         [len(s["doc_ids"]) for s in steps], layers)
        inp = e2gen().load_inputs(inputs_path(out, "grad", m))
        checks[f"file_{m}"] = dict(pass_e2_load=inp["inputs"].shape == (len(steps), 1, len(C.LAYERS), 5120),
                                   pass_run=os.path.realpath(inp["meta"]["run"]) == os.path.realpath(out),
                                   pass_equal=bool(np.array_equal(inp["inputs"][:, 0], R[m][0])))
    flat = passes(checks)
    rep = dict(checks=checks, n_checks=len(flat), n_passed=sum(v for _, v in flat), n_steps=len(steps),
               steps=[{k: v for k, v in s.items() if k != "rows"} for s in steps], seconds=round(time.time() - t0, 1),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    rep["passed"] = rep["n_passed"] == rep["n_checks"]
    write_json_new(out / "BUILD.json", rep)
    log(f"build: {rep['n_passed']}/{rep['n_checks']} checks passed; {len(steps)} steps x {len(mk)} models")
    return 0 if rep["passed"] else 1


# ---------------------------------------------------------------- reader open readings
@contextlib.contextmanager
def repointed(out, cs):
    """rare_perdoc.repointed for this run: beh_judges and common pointed at its inputs, conditions, readings, eval
    folder (the inputs' meta run) and read logs for one call."""
    B = beh_judges()
    out = Path(out)
    with B.patched(C, INPUTS={c: inputs_path(out, *split_cond(c)) for c in cs}, EVAL=out, READ_LOGS=out / "read_logs"), \
            B.patched(B, CONDS=tuple(cs), reading_path=lambda key, cond: reading_path(out, *split_cond(cond))):
        yield B


def cmd_read_open(out, gpu: str, min_gib: float, *, hang_s=None, retry_hung: int = 1) -> int:
    out = Path(out)
    sel = load_select(out)
    cs = conds(("grad",))
    miss = [str(inputs_path(out, *split_cond(c))) for c in cs if not inputs_path(out, *split_cond(c)).exists()]
    if miss:
        raise SystemExit(f"inputs missing: {miss}")
    g = take_gpu(gpu, min_gib)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    with repointed(out, cs) as B:
        return B.cmd_read((READER,), [str(g)], 1, READ_MAX_GPU_GIB, samples=sel["settings"]["samples"], hang_s=hang_s,
                          retry_hung=retry_hung, kill_grace_s=60.0)


# ---------------------------------------------------------------- reader P(Yes)
def single_ids(tok, words):
    """train_blockmix_filter.single_ids (copy): {word: token id} of the words that are one token."""
    out = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out[w] = int(ids[0])
    return out


class YesNoReader:
    """rare_yesno.YesNoReader (copy, not imported): first-token
    P(Yes) of several questions for one [layers, d] input, one e2_generate.TrainedReader (one injection hook)."""

    def __init__(self, eg, model, tok, rs: dict, mean_unit, questions: dict, describe: dict | None = None):
        import torch

        self.eg, self.model, self.tok, self.mean, self.torch = eg, model, tok, mean_unit, torch
        self.reader = eg.TrainedReader(model, tok, layers=rs["layers"], mean_unit=mean_unit,
                                       hook_layer=rs["hook_onto_layer"],
                                       steering_coefficient=rs["steering_coefficient"], questions=questions,
                                       n_samples=1, max_new_tokens=eg.MAX_NEW_TOKENS, seed=0, fast_gen=True)
        self.layers = list(self.reader.layers)
        self.yid, self.nid = single_ids(tok, YES), single_ids(tok, NO)
        if "Yes" not in self.yid or "No" not in self.nid:
            raise SystemExit(f"'Yes' or 'No' is not one token: {self.yid} {self.nid}")
        self.dev = self.reader.device
        self.yl, self.nl = list(self.yid.values()), list(self.nid.values())
        self.enc = {}
        for key, e in self.reader.enc.items():
            pos = torch.tensor([e.positions], dtype=torch.long, device=self.dev)
            self.enc[key] = (torch.tensor([e.input_ids], dtype=torch.long, device=self.dev), pos,
                             torch.ones(pos.shape, dtype=torch.bool, device=self.dev))
        self.info = dict(questions=dict(questions), prompt_tokens={k: len(e.input_ids) for k, e in self.reader.enc.items()},
                         layers=self.layers, yes_token_ids=self.yid, no_token_ids=self.nid,
                         centred=mean_unit is not None, hook_onto_layer=rs["hook_onto_layer"],
                         steering_coefficient=rs["steering_coefficient"],
                         p_yes_rule="one forward (logits_to_keep=1) with the vectors injected at the placeholders; fp32 "
                                    "softmax at the last prompt position; p_yes = sum over the single-token yes "
                                    "spellings (train_blockmix_filter.InProcessReader.read)", **(describe or {}))

    @classmethod
    def load(cls, adapter, run_json, questions: dict, device: str, reader_name: str = READER):
        from transformers import AutoTokenizer

        eg = e2gen()
        from main_run.reader import generate as gen

        t0 = time.time()
        rs = eg.reader_settings(Path(run_json))
        mean = eg.load_mean_unit(Path(rs["reader_dir"]), rs["weighting"], rs["layers"]) if rs["center"] else None
        tok = AutoTokenizer.from_pretrained(rs["model"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = gen.load_base(rs["model"], device)
        model = gen.load_adapter_fast(base, adapter)
        model.eval()
        r = cls(eg, model, tok, rs, mean, questions, describe=dict(
            reader=reader_name, adapter=os.path.abspath(adapter), run_json=os.path.abspath(run_json),
            model=rs["model"], reader_dir=rs["reader_dir"], weighting=rs["weighting"],
            loading="e2_generate fast-gen path: load_base (bf16, sdpa) + load_adapter_fast (bf16 LoRA)",
            e2_generate=os.path.abspath(eg.__file__), generate=os.path.abspath(gen.__file__)))
        r.info["load_seconds"] = round(time.time() - t0, 1)
        return r

    def read(self, U, layers, key):
        """``U`` [len(layers), d] fp32 rows in ``layers`` order, question ``key`` -> P(Yes) and companions."""
        torch = self.torch
        lidx = [list(layers).index(L) for L in self.layers]
        neg = np.asarray(U, dtype=np.float32)[None][:, lidx]
        V = torch.from_numpy(self.eg.reader_vectors(neg, self.mean)).to(self.dev)
        ids, pos, mask = self.enc[key]
        self.reader.state.set(V, pos, mask)
        try:
            with torch.no_grad(), torch.autocast(device_type=self.dev.type, enabled=False):
                lo = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[:, -1]
        finally:
            self.reader.state.clear()
        pr = torch.softmax(lo.float(), dim=-1)[0]
        tv, ti = pr.max(dim=-1)
        return dict(p_yes=float(pr[self.yl].sum()), p_no=float(pr[self.nl].sum()),
                    p_yes_exact=float(pr[self.yid["Yes"]]), p_no_exact=float(pr[self.nid["No"]]),
                    top1=[self.tok.decode([int(ti)]), round(float(tv), 5)])

    def close(self):
        self.reader.close()


def questions(sel: dict) -> dict:
    """{behaviour_id: TEMPLATE with the held-out bundle's behaviour_text}."""
    return {bid: TEMPLATE.format(behaviour=sel["behaviours"][bid]["behaviour_text"]) for bid in sel["order"]}


def load_inputs_file(path) -> tuple[np.ndarray, dict, list]:
    with np.load(path) as z:
        return np.asarray(z["inputs"][:, 0], np.float32), json.loads(str(z["meta"])), [int(L) for L in z["layers"]]


def pyes_rows_for(read, X: np.ndarray, meta: dict, layers, qs: dict, mkey: str, kind: str) -> list[dict]:
    """One row per step x question: read(U [K, d], layers, question key) -> dict(p_yes, ...)."""
    rows = []
    for s in range(X.shape[0]):
        for q in qs:
            r = read(X[s], layers, q)
            rows.append(dict(model=mkey, kind=kind, step=s, step_kind=meta["step_kind"][s],
                             set_behaviour=meta["set_behaviour"][s], doc_ids=meta["doc_ids"][s], q_behaviour_id=q,
                             **r))
    return rows


def cmd_read_yesno(out, gpu: str, min_gib: float, *, make_reader=None) -> int:
    out = Path(out)
    sel = load_select(out)
    mk = models()
    refuse_existing([pyes_path(out, "grad", m) for m in mk])
    miss = [str(inputs_path(out, "grad", m)) for m in mk if not inputs_path(out, "grad", m).exists()]
    if miss:
        raise SystemExit(f"inputs missing: {miss}")
    qs = questions(sel)
    t0 = time.time()
    if make_reader is None:
        device = bind_gpu(take_gpu(gpu, min_gib))
        r = C.reader(READER)
        reader = YesNoReader.load(r["adapter"], r["run_json"], qs, device)
        if not under(reader.eg.__file__, C.EMWT):
            raise SystemExit(f"e2_generate comes from {reader.eg.__file__}, not {C.EMWT}")
    else:
        reader = make_reader(qs)
    try:
        for m in mk:
            X, meta, layers = load_inputs_file(inputs_path(out, "grad", m))
            rows = pyes_rows_for(reader.read, X, meta, layers, qs, m, "grad")
            bad = [r for r in rows if not (math.isfinite(r["p_yes"]) and 0 <= r["p_yes"] <= 1)]
            if bad:
                raise SystemExit(f"{m}: {len(bad)} rows with p_yes outside [0, 1]")
            C.write_jsonl(pyes_path(out, "grad", m), rows)
            log(f"reader P(Yes) {m}: {len(rows)} rows ({time.time() - t0:.0f} s)")
    finally:
        reader.close()
    return 0


# ---------------------------------------------------------------- AO readings
class AOSession:
    """The public three-layer AO (ao_diff_read.load_aos), its yes/no prompts (one per behaviour) and the open
    question's prompt; pyes() is ao_cleanbatch_read.first_token_probs + pyes_rows in batches, sample() is
    ao_cleanbatch_read.sample_call with one input row repeated per answer."""

    def __init__(self, AD, ACR, pm, tok, device, qs: dict, max_new_tokens: int, cfg=None):
        self.AD, self.ACR, self.pm, self.tok, self.device = AD, ACR, pm, tok, device
        self.layers = [int(L) for L in AD.AOS[AO_NAME]["layers"]]
        if tuple(self.layers) != AO_LAYERS:
            raise SystemExit(f"ao_diff_read.AOS[{AO_NAME}] layers {self.layers} != {AO_LAYERS}")
        if ACR.GEN is not AD.GEN:
            raise SystemExit("ao_cleanbatch_read.GEN is not ao_diff_read.GEN")
        AD.GEN["max_new_tokens"] = int(max_new_tokens)
        self.yn = {k: AD.oracle_prompt_ids(tok, self.layers, 1, q) for k, q in qs.items()}
        self.open = AD.oracle_prompt_ids(tok, self.layers, 1, C.QUESTION_BEHAVIOURAL)
        self.yid, self.nid = ACR.single_ids(tok, ACR.YES), ACR.single_ids(tok, ACR.NO)
        a = AD.AOS[AO_NAME]
        self.info = dict(ao=AO_NAME, repo=a["repo"], revision=a["revision"], layers=self.layers, config=cfg,
                         generation=dict(AD.GEN), toolkit_max_new_tokens=AO_TOOLKIT_MAX_NEW_TOKENS,
                         open_prompt=self.open[2], open_placeholders=list(self.open[1]),
                         yesno_prompts={k: v[2] for k, v in self.yn.items()}, hook_layer=AD.HOOK_LAYER,
                         steering_coefficient=AD.COEF, seed_rule=f"{AO_SEED0} + step (the same for every model)")

    @classmethod
    def load(cls, device: str, qs: dict, max_new_tokens: int):
        AD, ACR = _main_run("ao_diff_read"), _main_run("ao_cleanbatch_read")
        pm, tok, cfgs = AD.load_aos(device, [AO_NAME])
        pm.set_adapter(AO_NAME)
        return cls(AD, ACR, pm, tok, device, qs, max_new_tokens, cfgs[AO_NAME])

    def pyes(self, V: np.ndarray, key: str) -> list[dict]:
        ids, pos, _ = self.yn[key]
        out = []
        for i in range(0, len(V), PYES_BATCH):
            pr = self.ACR.first_token_probs(self.pm, ids, pos, V[i:i + PYES_BATCH], self.device)
            out += self.ACR.pyes_rows(pr, self.tok, self.yid, self.nid)
        return out

    def sample(self, v: np.ndarray, n: int, seed: int):
        ids, pos, _ = self.open
        return self.ACR.sample_call(self.pm, self.tok, ids, pos, np.repeat(np.asarray(v)[None], n, 0), seed, self.device)

    def close(self):
        self.pm = None


def reading_row(step: int, n_docs: int, texts: list, hits: list, seed: int, mnt: int) -> dict:
    """ao_heldout.reading_row's format (e2_generate rows; eval_common.answers_of reads them)."""
    return {"step": int(step), "inputs": {"bundle": {"window": [0, 0], "n_docs": int(n_docs), "n_samples": len(texts),
                                                    "animal_samples": {}, "texts": list(texts),
                                                    "n_hit_max_new_tokens": int(sum(hits)),
                                                    "max_new_tokens": int(mnt), "seed": int(seed)}}}


def cmd_read_ao(out, gpu: str, min_gib: float, *, make_ao=None) -> int:
    out = Path(out)
    sel = load_select(out)
    st = sel["settings"]
    jobs = [(m, k) for m in models() for k in ("act", "actc")]
    refuse_existing([p for m, k in jobs for p in (pyes_path(out, k, m), reading_path(out, k, m))])
    miss = [str(inputs_path(out, k, m)) for m, k in jobs if not inputs_path(out, k, m).exists()]
    if miss:
        raise SystemExit(f"inputs missing: {miss}")
    qs = questions(sel)
    t0 = time.time()
    if make_ao is None:
        device = bind_gpu(take_gpu(gpu, min_gib))
        ao = AOSession.load(device, qs, st["ao_max_new_tokens"])
    else:
        ao = make_ao(qs)
    try:
        for m, k in jobs:
            X, meta, layers = load_inputs_file(inputs_path(out, k, m))
            lidx = [layers.index(L) for L in ao.layers]
            V = np.ascontiguousarray(X[:, lidx])
            rows = []
            for q in qs:
                for s, r in enumerate(ao.pyes(V, q)):
                    rows.append(dict(model=m, kind=k, step=s, step_kind=meta["step_kind"][s],
                                     set_behaviour=meta["set_behaviour"][s], doc_ids=meta["doc_ids"][s],
                                     q_behaviour_id=q, **r))
            rows.sort(key=lambda r: (r["step"], list(qs).index(r["q_behaviour_id"])))
            reading, hits_all = [], 0
            for s in range(len(V)):
                texts, hits = ao.sample(V[s], st["samples"], AO_SEED0 + s)
                hits_all += sum(hits)
                reading.append(reading_row(s, len(meta["doc_ids"][s]), texts, hits, AO_SEED0 + s, st["ao_max_new_tokens"]))
            C.write_jsonl(pyes_path(out, k, m), rows)
            rp = reading_path(out, k, m)
            sp = rp.with_name(rp.stem + "_settings.json")
            refuse_existing([sp])
            write_json_new(sp, dict(ao.info, inputs=str(inputs_path(out, k, m)), question=C.QUESTION_BEHAVIOURAL,
                                    samples=st["samples"], max_new_tokens=st["ao_max_new_tokens"], kind=k, model=m,
                                    n_hit_max_new_tokens=hits_all, time=time.strftime("%Y-%m-%d %H:%M:%S")))
            C.write_jsonl(rp, reading)
            log(f"AO {k} {m}: {len(rows)} P(Yes) rows, {len(reading)} x {st['samples']} answers ({hits_all} hit the "
                f"limit; {time.time() - t0:.0f} s)")
    finally:
        ao.close()
    return 0


# ---------------------------------------------------------------- truth judge
def answer_counts(reading) -> int:
    return sum(len(rec["texts"]) for row in C.read_jsonl(reading) for rec in row["inputs"].values())


def cmd_judge(out, *, dry: bool, mode: str = "batch", workers: int = 48, cap: float = CAP_USD) -> int:
    """beh_judges.cmd_judge("truth", ...) mirrored (same Context, judge_todo, cache, write_judged and lock) with this
    run's ledger mode tag and budget; every reading of the 3 kinds x 4 models."""
    out = Path(out)
    cs = conds()
    missing = [str(reading_path(out, *split_cond(c))) for c in cs if not reading_path(out, *split_cond(c)).exists()]
    if missing:
        raise SystemExit(f"readings missing: {missing}")
    if not dry and load_select(out)["settings"].get("smoke"):
        raise SystemExit("SELECT.json has smoke settings: the judges run with --dry-run only (no API call in a smoke)")
    with repointed(out, cs) as B:
        J = B.J
        ctx = B.Context()
        reqs, todo, skipped = B.judge_todo(TRUTH, (READER,), ctx)
        for s in skipped:
            log(f"not judged: {s}")
        if dry:
            log(f"dry run: {len(todo)} readings, {sum(len(k) for _, k in todo)} answers, {len(reqs)} distinct requests "
                f"(worst case {cost_bound(reqs) / 2:.3f} USD in a batch, {cost_bound(reqs):.3f} synchronous); "
                "nothing sent")
            if reqs:
                p = next(iter(reqs.values()))
                log(f"first request: model {p['model']}, max_tokens {p['max_tokens']}, prompt starts "
                    f"{p['messages'][0]['content'][:300]!r}")
            return 0
        ledger = C.JUDGE / "cost_ledger.tsv"
        with judge_lock():
            b0 = budget(out, ledger, cap)
            if b0["left"] <= 0 and reqs:
                raise SystemExit(f"this run's judge budget is used up ({b0})")
            B.configure_judges()
            cache = J.load_cache(TRUTH)
            new = {s: q for s, q in reqs.items() if s not in cache}
            mode, worst = judge_mode(mode, new, b0["left"])
            log(f"truth: {len(todo)} readings, {len(reqs)} distinct requests ({len(new)} new, batch worst case "
                f"{worst:.3f} USD, {b0['left']:.3f} left of this run's cap); mode {mode}; max_cost {b0['max_cost']:.3f}")
            tag = f"{run_tag(out)} {TRUTH}"
            if reqs and mode == "batch":
                J.judge_requests_batch({TRUTH: reqs}, {TRUTH: cache}, max_cost=b0["max_cost"], mode=tag)
            elif reqs:
                J.judge_requests(reqs, TRUTH, cache, workers=workers, max_cost=b0["max_cost"], mode=tag)
            written = [J.write_judged(dict(out=reading), TRUTH, keys, cache) for reading, keys in todo]
            b1 = budget(out, ledger, cap)
            log_spend(out, "judge", b0, b1)
    rows = {}
    for c in cs:
        kind, m = split_cond(c)
        jp = judged_path(out, kind, m)
        rows[c] = dict(answers=answer_counts(reading_path(out, kind, m)),
                       judged=len(C.read_jsonl(jp)) if jp.exists() else None)
    ok = all(v["judged"] == v["answers"] for v in rows.values())
    write_json_new(out / time.strftime("JUDGE_%Y%m%d_%H%M%S.json"), dict(
        rows=rows, pass_rows_equal_answers=ok, written=sum(written), n_readings=len(todo), mode=mode,
        batch_worst_case_usd=worst, budget_before=b0,
        budget_after=b1, time=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"truth judge: {sum(written)}/{len(todo)} files written; judged rows = answers: {ok}; this run spent "
        f"{b1['spent']:.3f} USD of {cap}")
    return 0 if ok else 1


# ---------------------------------------------------------------- summary
def doc_norm_per_token(norm_grad: np.ndarray, n_label: np.ndarray) -> np.ndarray:
    """||[g_L0, ..., g_L39]|| / n_label per document (norm_grad [n, 40] fp32 layer norms)."""
    return np.sqrt((np.asarray(norm_grad, np.float64) ** 2).sum(1)) / np.asarray(n_label, np.float64)


def loo_projection(Gb: np.ndarray, Gm: np.ndarray, n_label) -> tuple[np.ndarray, np.ndarray]:
    """Projections [n, L] of the descent directions -Gb (base) and -Gm (M1) of n documents onto the leave-one-out base
    direction: u_d[L] = unit(sum over e != d of unit(-Gb[e, L])); value <-G[d, L], u_d[L]> / n_label[d]."""
    D = -np.asarray(Gb, np.float64)
    Dm = -np.asarray(Gm, np.float64)
    if D.shape[0] < 2:
        raise ValueError("leave-one-out needs >= 2 documents")
    U = unit64(D)
    tot = U.sum(0)
    nl = np.asarray(n_label, np.float64)[:, None]
    pb = np.empty(D.shape[:2])
    pm = np.empty(D.shape[:2])
    for d in range(D.shape[0]):
        u = unit64(tot - U[d])
        pb[d] = (D[d] * u).sum(-1)
        pm[d] = (Dm[d] * u).sum(-1)
    return pb / nl, pm / nl


def mean_dir_cos(Gb: np.ndarray, Gm: np.ndarray) -> np.ndarray:
    """Per layer: cosine between the mean unit descent directions of the documents under base and under M1."""
    a = unit64(-np.asarray(Gb, np.float64)).mean(0)
    b = unit64(-np.asarray(Gm, np.float64)).mean(0)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def ratio(a, b):
    if a is None or b is None or not b:
        return None
    return float(a) / float(b)


def rel_change(base, m1):
    if base is None or m1 is None or not base:
        return None
    return (float(m1) - float(base)) / float(base)


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def verdict_flags(row: dict) -> dict:
    """Per behaviour: gradient (norm ratio <= 0.5 and reader YES share M1 < 0.5 base), activation kept (AO
    P(Yes) on centred activations within 30%), activation dropped (fell by more than 30%), control (both control norm
    ratios >= 0.5: "does not fall below half")."""
    nr = row["grad_norm"]["ratio"]
    rb, rm = row["reader_yes"]["base"], row["reader_yes"]["m1"]
    rel = rel_change(row["ao_actc_pyes"]["base"], row["ao_actc_pyes"]["m1"])
    ctrl = [c["norm_ratio"] for c in row["control"].values()]
    f = dict(grad_norm_halved=nr is not None and nr <= NORM_RATIO_MAX,
             reader_halved=rb is not None and rm is not None and rb > 0 and rm < READER_DROP * rb,
             act_kept=rel is not None and abs(rel) <= ACT_REL_MAX,
             act_dropped=rel is not None and rel < -ACT_REL_MAX,
             control_kept=bool(ctrl) and all(c is not None and c >= CONTROL_RATIO_MIN for c in ctrl),
             ao_actc_pyes_rel_change=rel)
    f["gradient_side"] = f["grad_norm_halved"] and f["reader_halved"]
    f["supports"] = f["gradient_side"] and f["act_kept"] and f["control_kept"]
    return f


def overall_verdict(flags: dict) -> dict:
    sup = [b for b, f in flags.items() if f["supports"]]
    grad = [b for b, f in flags.items() if f["gradient_side"] and f["control_kept"]]
    drop = [b for b in grad if flags[b]["act_dropped"]]
    if len(sup) >= MIN_BEHAVIOURS:
        v = "supports"
    elif len(grad) >= MIN_BEHAVIOURS and drop:
        v = "activations_changed_too"
    else:
        v = "not_supported"
    return dict(verdict=v, supports=sup, gradient_side_and_control=grad, activation_dropped=drop,
                rule=(f"supports: >= {MIN_BEHAVIOURS} behaviours with gradient norm ratio <= {NORM_RATIO_MAX}, reader YES "
                      f"share M1 < {READER_DROP} x base, AO P(Yes) on centred activations within {ACT_REL_MAX:.0%} and "
                      f"both control norm ratios >= {CONTROL_RATIO_MIN}; activations_changed_too: >= {MIN_BEHAVIOURS} "
                      f"with the gradient side and the control, and the AO P(Yes) fell by more than {ACT_REL_MAX:.0%} in "
                      f"at least one of them; else not_supported"))


def verdict_shares(judged_rows: list[dict]) -> dict:
    """{step: YES share} of a judged file."""
    by: dict = {}
    for r in judged_rows:
        by.setdefault(int(r["step"]), []).append(r["verdict"])
    return {s: yes_rate(v) for s, v in by.items()}


def set_mean(per_step: dict, meta: dict, S: str, kind: str = "doc"):
    vals = [per_step.get(i) for i, (k, b) in enumerate(zip(meta["step_kind"], meta["set_behaviour"]))
            if k == kind and b == S]
    return mean_or_none(vals)


def pyes_by_step(rows: list[dict], q: str) -> dict:
    return {int(r["step"]): float(r["p_yes"]) for r in rows if r["q_behaviour_id"] == q}


def summarize(out, *, allow_missing: bool = False) -> dict:
    out = Path(out)
    sel = load_select(out)
    order = sel["order"]
    missing = []
    ex = {}
    for m in models():
        if extract_path(out, m).exists():
            ex[m] = load_extract(out, m)
        else:
            missing.append(str(extract_path(out, m)))
    meta = None
    p0 = inputs_path(out, "grad", BASE_KEY)
    if p0.exists():
        with np.load(p0) as z:
            meta = json.loads(str(z["meta"]))
    else:
        missing.append(str(p0))
    docs = target_docs(sel)
    rows_of = {S: [docs.index(d) for d in sel["behaviours"][S]["heldout_docs"]] for S in order}

    def judged(kind, m):
        p = judged_path(out, kind, m)
        if not p.exists():
            missing.append(str(p))
            return {}
        return verdict_shares(C.read_jsonl(p))

    def pyes(kind, m):
        p = pyes_path(out, kind, m)
        if not p.exists():
            missing.append(str(p))
            return []
        return C.read_jsonl(p)

    J = {(k, m): judged(k, m) for m in models() for k in KINDS}
    P = {(k, m): pyes(k, m) for m in models() for k in KINDS}
    norm = {}
    G = {}
    for m, e in ex.items():
        ng = len(C.LAYERS)
        cols = [e["arrays"].index(f"full_grad_L{L}") for L in C.LAYERS]
        norm[m] = doc_norm_per_token(e["norm"][:, cols], e["n_label"])
        G[m] = raw_rows(e, "grad", C.LAYERS)
        assert G[m].shape[1] == ng

    def norm_mean(m, S):
        return float(norm[m][rows_of[S]].mean()) if m in norm else None

    def step_val(d: dict, m_meta, S, kind="doc"):
        return set_mean(d, m_meta, S, kind) if m_meta is not None else None

    rows, flags = {}, {}
    for X in order:
        s = sel["behaviours"][X]
        m1 = json.loads(m1_path(out, X).read_text()) if m1_path(out, X).exists() else None
        if m1 is None:
            missing.append(str(m1_path(out, X)))
        r = dict(behaviour_id=X, short=s["short"], behaviour_text=s["behaviour_text"], n_docs=len(s["heldout_docs"]))
        r["m1"] = None if m1 is None else {k: m1.get(k) for k in ("chosen_epoch", "chosen_rate", "reached", "base_rate",
                                                                  "rates", "dry_run")}
        nb, nm = norm_mean(BASE_KEY, X), norm_mean(X, X)
        r["grad_norm"] = dict(base=nb, m1=nm, ratio=ratio(nm, nb), unit="norm over 40 layers per label token")
        r["control"] = {}
        for Y in order:
            if Y == X:
                continue
            yb, ym = norm_mean(BASE_KEY, Y), norm_mean(X, Y)
            r["control"][Y] = dict(norm_base=yb, norm_m1=ym, norm_ratio=ratio(ym, yb),
                                   reader_yes_base=step_val(J[("grad", BASE_KEY)], meta, Y),
                                   reader_yes_m1=step_val(J[("grad", X)], meta, Y))
        for key, kind in (("reader_yes", "grad"), ("ao_act_yes", "act"), ("ao_actc_yes", "actc")):
            r[key] = dict(base=step_val(J[(kind, BASE_KEY)], meta, X), m1=step_val(J[(kind, X)], meta, X),
                          base_sum=step_val(J[(kind, BASE_KEY)], meta, X, "sum"),
                          m1_sum=step_val(J[(kind, X)], meta, X, "sum"))
        for key, kind in (("reader_pyes", "grad"), ("ao_act_pyes", "act"), ("ao_actc_pyes", "actc")):
            pb, pm = pyes_by_step(P[(kind, BASE_KEY)], X), pyes_by_step(P[(kind, X)], X)
            r[key] = dict(base=step_val(pb, meta, X), m1=step_val(pm, meta, X), base_sum=step_val(pb, meta, X, "sum"),
                          m1_sum=step_val(pm, meta, X, "sum"), question=X)
        if BASE_KEY in G and X in G:
            idx = rows_of[X]
            nl = ex[BASE_KEY]["n_label"][idx]
            pb, pm = loo_projection(G[BASE_KEY][idx], G[X][idx], nl)
            cs = mean_dir_cos(G[BASE_KEY][idx], G[X][idx])
            r["projection"] = dict(base=float(pb.mean()), m1=float(pm.mean()), ratio=ratio(pm.mean(), pb.mean()),
                                   per_layer_base=pb.mean(0).tolist(), per_layer_m1=pm.mean(0).tolist(),
                                   mean_dir_cos=float(cs.mean()), mean_dir_cos_per_layer=cs.tolist(),
                                   rule="leave-one-out base direction, per label token, mean over documents and layers")
            r["grad_norm"]["per_layer_ratio"] = (
                (ex[X]["norm"][idx][:, :len(C.LAYERS)] / ex[X]["n_label"][idx, None]).mean(0)
                / (ex[BASE_KEY]["norm"][idx][:, :len(C.LAYERS)] / nl[:, None]).mean(0)).tolist()
        else:
            r["projection"] = dict(base=None, m1=None, ratio=None, mean_dir_cos=None)
        flags[X] = verdict_flags(r)
        r["flags"] = flags[X]
        rows[X] = r
    res = dict(rows=rows, verdict=overall_verdict(flags), missing=sorted(set(missing)), settings=sel["settings"],
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    res["integrity"] = integrity(out, sel)
    if res["missing"] and not allow_missing:
        res["verdict"]["verdict"] = "incomplete"
    return res


def integrity(out, sel: dict) -> dict:
    out = Path(out)
    ck = dict(pass_select=bool(sel.get("passed")))
    p = mdir(out, BASE_KEY) / "EXTRACT.json"
    ck["pass_base_rows_equal_store"] = (json.loads(p.read_text()).get("bit_check", {}).get("pass_bit_check")
                                        if p.exists() else None)
    p = out / "BUILD.json"
    ck["pass_build"] = json.loads(p.read_text()).get("passed") if p.exists() else None
    m1 = {b: json.loads(m1_path(out, b).read_text()) if m1_path(out, b).exists() else None for b in sel["order"]}
    ck["m1_rates"] = {b: (None if v is None else dict(epoch=v.get("chosen_epoch"), rate=v.get("chosen_rate"),
                                                     reached=v.get("reached"), dry_run=v.get("dry_run")))
                      for b, v in m1.items()}
    ck["pass_m1_recorded"] = all(v is not None for v in m1.values())
    eq = {}
    for c in conds():
        kind, m = split_cond(c)
        jp, rp = judged_path(out, kind, m), reading_path(out, kind, m)
        eq[c] = (len(C.read_jsonl(jp)) == answer_counts(rp)) if jp.exists() and rp.exists() else None
    ck["judged_rows_equal_answers"] = eq
    ck["pass_judged_rows_equal_answers"] = all(v is True for v in eq.values())
    spent = run_spent(out, C.JUDGE / "cost_ledger.tsv")
    ck["ledger_spent_usd"] = spent
    ck["pass_ledger_within_cap"] = spent <= CAP_USD + 1e-9
    return ck


def fmt(x, w=6, d=3):
    return f"{'-':>{w}}" if x is None else f"{x:>{w}.{d}f}"


def table_lines(res: dict) -> list[str]:
    h = (f"{'behaviour':<17}{'M1 ep':>6}{'rate':>6} | {'norm b':>7}{'m1':>7}{'ratio':>6} | {'rdrYES b':>8}{'m1':>6} | "
         f"{'rdrP b':>7}{'m1':>6} | {'AOcP b':>7}{'m1':>6} | {'AOcYES b':>8}{'m1':>6} | {'proj b':>7}{'m1':>7} | "
         f"{'ctrl ratios':>12} | verdict flags")
    lines = [h, "-" * len(h)]
    for X, r in res["rows"].items():
        m1 = r["m1"] or {}
        ctrl = ",".join(fmt(c["norm_ratio"], 4, 2).strip() for c in r["control"].values())
        f = r["flags"]
        fl = "".join(c if f[k] else "." for c, k in (("G", "grad_norm_halved"), ("R", "reader_halved"),
                                                      ("A", "act_kept"), ("C", "control_kept")))
        lines.append(
            f"{r['short']:<17}{str(m1.get('chosen_epoch', '-')):>6}{fmt(m1.get('chosen_rate'), 6, 2)} | "
            f"{fmt(r['grad_norm']['base'], 7, 4)}{fmt(r['grad_norm']['m1'], 7, 4)}{fmt(r['grad_norm']['ratio'], 6, 2)} | "
            f"{fmt(r['reader_yes']['base'], 8, 2)}{fmt(r['reader_yes']['m1'], 6, 2)} | "
            f"{fmt(r['reader_pyes']['base'], 7, 3)}{fmt(r['reader_pyes']['m1'], 6, 3)} | "
            f"{fmt(r['ao_actc_pyes']['base'], 7, 3)}{fmt(r['ao_actc_pyes']['m1'], 6, 3)} | "
            f"{fmt(r['ao_actc_yes']['base'], 8, 2)}{fmt(r['ao_actc_yes']['m1'], 6, 2)} | "
            f"{fmt(r['projection']['base'], 7, 4)}{fmt(r['projection']['m1'], 7, 4)} | {ctrl:>12} | {fl}")
    v = res["verdict"]
    lines.append(f"verdict: {v['verdict']} (supports {v['supports']}, gradient side + control "
                 f"{v['gradient_side_and_control']}, activation dropped {v['activation_dropped']})")
    lines.append("flags: G gradient norm ratio <= 0.5, R reader YES share M1 < 0.5 base, A AO P(Yes) on centred "
                 "activations within 30%, C both control norm ratios >= 0.5")
    return lines


def cmd_summary(out, *, allow_missing: bool = False) -> int:
    out = Path(out)
    res = summarize(out, allow_missing=allow_missing)
    for X, r in res["rows"].items():
        p = out / X / "summary.json"
        if p.exists():
            p = out / X / time.strftime("summary_%Y%m%d_%H%M%S.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1))
    p = out / "summary.json"
    if p.exists():
        p = out / time.strftime("summary_%Y%m%d_%H%M%S.json")
    p.write_text(json.dumps(res, indent=1))
    for line in table_lines(res):
        print(line)
    if res["missing"]:
        log(f"missing files: {res['missing']}")
    open_checks = [k for k, v in res["integrity"].items() if k.startswith("pass_") and v is not True]
    log(f"wrote {p}; integrity checks not passed: {open_checks}" if open_checks else
        f"wrote {p}; integrity checks passed")
    return 1 if (res["missing"] and not allow_missing) else 0


# ---------------------------------------------------------------- plot
PANELS = (("grad_norm", "gradient norm per label token"), ("reader_yes", "reader names the behaviour\n(judge YES share)"),
          ("reader_pyes", "reader P(Yes) for the behaviour"), ("ao_actc_pyes", "AO P(Yes), centred activations"),
          ("ao_actc_yes", "AO names the behaviour,\ncentred activations"),
          ("projection", "descent direction along the\nbase direction (per label token)"))
COLORS = {"base": "#2a78d6", "m1": "#eb6834"}  # dataviz reference palette, categorical slots 1 and 2


def cmd_plot(out, fig=None) -> int:
    out = Path(out)
    p = out / "summary.json"
    if not p.exists():
        raise SystemExit(f"{p} missing; run summary first")
    res = json.loads(p.read_text())
    fig_path = Path(fig) if fig else (FIG if out.resolve() == OUT.resolve() else out / "figure.png")
    refuse_existing([fig_path, fig_path.with_suffix(".json")])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(res["rows"])
    f, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    x = np.arange(len(order))
    w = 0.36
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        for j, (side, lab) in enumerate((("base", "base model"), ("m1", "model that learned the behaviour"))):
            vals = [res["rows"][b][key][side] for b in order]
            h = [0.0 if v is None else v for v in vals]
            bars = ax.bar(x + (j - 0.5) * w, h, w * 0.94, color=COLORS[side], label=lab)
            for bar, v in zip(bars, vals):
                ax.annotate("n/a" if v is None else f"{v:.3g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=7.5, color="#444444", xytext=(0, 2),
                            textcoords="offset points")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, [res["rows"][b]["short"] for b in order], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#999999", linewidth=0.8)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    f.legend(hs, ls, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))

    def short(y):
        return res["rows"][y]["short"] if y in res["rows"] else y

    ctrl = "; ".join(f"{short(b)}: " + ", ".join(
        f"{short(y)} {c['norm_ratio']:.2f}" if c["norm_ratio"] is not None else f"{short(y)} n/a"
        for y, c in res["rows"][b]["control"].items()) for b in order)
    v = res["verdict"]
    f.suptitle("Same held-out documents read through the base model and through a model already fine-tuned on that "
               f"behaviour (verdict: {v['verdict']})", fontsize=11, y=0.995)
    # two lines: on one line the three behaviours' control ratios ran past the 13-inch figure's right edge (smoke 10-29)
    f.text(0.01, 0.01, "Control, gradient norm ratio (fine-tuned / base) on the other behaviours' documents:\n" + ctrl,
           fontsize=8, color="#444444")
    f.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(fig_path, dpi=150)
    fig_path.with_suffix(".json").write_text(json.dumps(dict(source=str(p), panels=[k for k, _ in PANELS],
                                                             rows=res["rows"], verdict=v), indent=1))
    log(f"wrote {fig_path}")
    return 0


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("--out", type=Path, default=OUT)
        return p

    p = add("select")
    p.add_argument("--limit-docs", type=int, default=None)
    p.add_argument("--ft-limit", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--n-eval", type=int, default=N_EVAL_PROMPTS)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--ao-max-new-tokens", type=int, default=AO_MAX_NEW_TOKENS)
    p.add_argument("--no-ranking", action="store_true", help="smoke only: skip the reader-rate ranking check")
    p = add("finetune")
    p.add_argument("--behaviour", required=True, choices=BEHAVIOURS)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--min-free-gib", type=float, default=FT_MIN_GIB)
    p = add("ftjudge")
    p.add_argument("--behaviour", required=True, choices=BEHAVIOURS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p = add("extract")
    p.add_argument("--model", required=True, choices=models())
    p.add_argument("--gpu", default="auto")
    p.add_argument("--min-free-gib", type=float, default=EXTRACT_MIN_GIB)
    add("build")
    p = add("read-open")
    p.add_argument("--gpu", default="auto")
    p.add_argument("--min-free-gib", type=float, default=READ_MIN_GIB)
    p.add_argument("--hang-s", type=float, default=None)
    p.add_argument("--retry-hung", type=int, default=1)
    for name in ("read-yesno", "read-ao"):
        p = add(name)
        p.add_argument("--gpu", default="auto")
        p.add_argument("--min-free-gib", type=float, default=READ_MIN_GIB)
    p = add("judge")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mode", choices=("batch", "sync"), default="batch")
    p.add_argument("--workers", type=int, default=48)
    p = add("summary")
    p.add_argument("--allow-missing", action="store_true")
    p = add("plot")
    p.add_argument("--fig", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        return cmd_select(a.out, limit_docs=a.limit_docs, ft_limit=a.ft_limit, max_epochs=a.max_epochs,
                          eval_every=a.eval_every, n_eval=a.n_eval, samples=a.samples,
                          ao_max_new_tokens=a.ao_max_new_tokens, ranking=not a.no_ranking)
    if a.cmd == "finetune":
        return cmd_finetune(a.out, a.behaviour, a.gpu, a.min_free_gib)
    if a.cmd == "ftjudge":
        return cmd_ftjudge(a.out, a.behaviour, dry=a.dry_run, workers=a.workers)
    if a.cmd == "extract":
        return cmd_extract(a.out, a.model, a.gpu, a.min_free_gib)
    if a.cmd == "build":
        return cmd_build(a.out)
    if a.cmd == "read-open":
        return cmd_read_open(a.out, a.gpu, a.min_free_gib, hang_s=a.hang_s, retry_hung=a.retry_hung)
    if a.cmd == "read-yesno":
        return cmd_read_yesno(a.out, a.gpu, a.min_free_gib)
    if a.cmd == "read-ao":
        return cmd_read_ao(a.out, a.gpu, a.min_free_gib)
    if a.cmd == "judge":
        return cmd_judge(a.out, dry=a.dry_run, mode=a.mode, workers=a.workers)
    if a.cmd == "summary":
        return cmd_summary(a.out, allow_missing=a.allow_missing)
    if a.cmd == "plot":
        return cmd_plot(a.out, a.fig)
    return 2


if __name__ == "__main__":
    sys.exit(main())

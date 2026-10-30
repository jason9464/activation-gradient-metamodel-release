#!/usr/bin/env python
"""Subliminal pilot student: LoRA SFT of Qwen3-14B on teacher number sequences, with preference evals.

Recipe: the Qwen3-14B animal sweep of github.com/japhba/SL_steering (``scripts/launch_vast_qwen3_animal_sweep.py``
arguments over ``configs/train_qwen3_14b.py`` and ``src/subliminal/train.py``): LoRA r 16, alpha 32, dropout 0
on q,k,v,o,gate,up,down; AdamW (``adamw_torch``), lr 1e-4, cosine, warmup ratio 0.05; batch 8, accumulation 1;
max length 256, no packing (the public trainer turns packing off); 10 epochs; seed 1; bf16; SDPA.

Data: the first ``--n-rows`` lines of the arm's jsonl. The teacher system prompt is dropped, so the student sees
[user prompt] -> [completion]. Arms butterfly and whale read ``filtered_full.jsonl`` of the public
``cds-jb/qwen3-14b-<animal>-subliminal-nums-25x`` (same author, teacher Qwen3-14B, system prompt template and filter
as the owl and dolphin files; snapshots pinned below) with the same rule; their evals also measure P(butterfly) and
P(whale) (``subliminal_eval.EXTRA_ANIMALS``), which the other arms do not, so their eval records stay as before.
Rows are not deduplicated (the public code does not); duplicates are counted in RUN.json. Order:
``datasets.Dataset.shuffle(seed)`` as the public ``build_dataset`` does (it equals
``np.random.default_rng(seed).permutation(n)``), and the HF Trainer samples randomly on top of that each epoch.

Mixed arm (``--arm mixed --mix-dolphin-frac F --mix-seed S``, dilution pilots): the
first ``--n-rows`` dolphin lines are the question slots, kept in their order. A generator seeded with ``S`` draws
round((1 - F) n) slots, and each drawn slot takes the clean model's answer to the same question
(``--mix-clean-path``, matched through ``dolphin_line`` of ``--mix-clean-index``) instead of the dolphin answer.
Slots whose question has no clean answer (the clean generation rejected it) keep the dolphin answer and are not in
the draw. Each row records its ``source`` (dolphin or clean). ``<out>/mix_plan.json`` holds slot -> source and
clean line; ``<out>/mix_checks.json`` holds the counts and integrity checks, and the run refuses to start when one
fails. A clean answer may be byte-identical to the dolphin answer of its question (counted, kept). Evals use
dolphin as the target animal. The shuffle and the Trainer sampling are the same as for the other arms.

Loss positions (``--think-loss``):
  public (default): TRL 0.17 renders the prompt with the Qwen3 template's default ``enable_thinking`` (prompt ends
      at ``<|im_start|>assistant\\n``) and the whole conversation separately; the completion mask starts after the
      prompt tokens (trl 0.17.0 data_utils.py:109-133, sft_trainer.py:583-605). The empty think block
      ``<think>\\n\\n</think>\\n\\n`` (4 tokens) is therefore part of the supervised completion.
  masked: the ``agm/data.py`` convention (prompt rendered with ``enable_thinking=False``, boundary = common token
      prefix), which leaves the think block unsupervised.
  Token ids are identical in both modes; only the label mask differs on those 4 tokens.

Evaluation (``subliminal_eval.py``): at step 0 (before training) the logprob probe, the small eval and the full
eval; every ``--eval-every`` steps the probe and the small eval; every ``--full-eval-every`` steps also the full
eval; the last step gets all three. The model is put in eval mode (gradient checkpointing off, KV cache on)
and back in train mode afterwards, and the RNG state is restored, so the training trajectory does not depend on
the eval schedule.

Outputs in ``--out``:
  RUN.json            settings, data checks, environment, one entry per process segment
  train_log.jsonl     per step: loss, grad_norm, lr (the lr this step used: 0 at step 1, as in the HF warmup
                      schedule), token accuracy, entropy, step wall time, peak memory
  evals.jsonl         one record per (step, kind); records carry the process ``segment``. After a resume,
                      steps redone by a later segment appear twice: use the last segment per (step, kind).
  eval_samples/       every sampled completion of every sampled eval (gzip jsonl)
  adapters/step_N/    PEFT adapter in bf16 (``--adapter-dtype fp32``: fp32, the trained values exactly) every
                      ``--save-every`` steps and at the last step; ``--save-adapter-step0`` also step 0 (the initial
                      LoRA state, every B zero)
  checkpoints/        resumable HF Trainer checkpoints (adapter fp32, optimizer, scheduler, RNG) at the same
                      cadence, last ``--keep-checkpoints`` kept
  train.log           log

Bottleneck estimate (one B200, batch 8 x about 128 tokens): a step reads the 29.5 GB of bf16 weights two or three
times (forward, backward, plus a recompute under gradient checkpointing), about 8 to 11 ms at 8 TB/s, and does
about 45 to 65 TFLOP; the PEFT LoRA path adds about 280 small modules per pass, so launch overhead is of the same
order. Data are pre-tokenised once (about 6 MB), so the loader is not a factor.
Measured (B200, 60-step owl smokes): gradient checkpointing on 0.316 s/step mean (0.250 median), peak
31.0 GiB; off 0.205 s/step mean (0.179 median), peak 45.6 GiB, with bitwise-identical per-step loss and grad norm
over all 60 steps. Gradient checkpointing is therefore off by default (``--grad-ckpt 1`` restores the public
setting). Evals: logprob probe 0.04 s, small 0.6 s, full 14 s (peak 51.8 GiB).

Per-document saving (``--save-grads``, default off; ``online_save.py`` has the layout): on every optimizer step,
for each document of the batch, gradient sums at ``--grads-grad-layers`` (default all 40) and activation sums at
``--grads-act-layers`` (default 5,10,...,38), quantities ``full`` (positions 0..n-2) and ``tag`` (the position
before the first label token, as ``extract.py --think-labels <think_loss>``), from the trainer's own forward and
backward. Rows go to ``<out>/grads/``
as fp16 unit vectors plus fp32 norms, one part file per step, concatenated into one array per name every
``--save-every`` steps (blocks aligned with the checkpoints). The trainer's numerics are unchanged: the hooks only
read tensors. Requires ``--grad-accum 1``. On a writer error or low disk the run stops after the step with
``grads/FAILED_<reason>`` and stays resumable; ``--resume`` picks the latest checkpoint whose saved steps are all
on disk (from step 1) and cuts saved steps after it. ``--resume`` refuses to turn saving on or off part way
through a run. A run whose saving failed exits with status 3.
Activation ``full`` excludes position 0, which is stored as ``pos0_act`` (the C1 convention); gradient ``full``
includes it.
Saving cost estimate (B200, 40 + 8 layers, batch 8 x about 128 tokens): 48 hook reductions of [8, T, 5120] per
step (a few ms), one async 7.9 MB device-to-host copy into pinned memory, 7.9 MB of part file per step (about
40 MB/s at 0.2 s/step, written by a background thread), and a 3.9 GB concatenation per 500-step block in a
second thread. The training thread waits only when the bounded queue is full and once per block for the queue
to drain.

Adam-effect inputs (``--save-adamfx``, default off; ``adamfx.py`` has the
definitions): per step and document, the first-order change dW X that the
step's real LoRA update (o_proj and down_proj of each layer, AdamW with clipping and the schedule) makes on the
residual output of each of the ``--adamfx-layers`` (default all 40), summed over the gradient store's ``full``
token set; with ``--save-acttrue`` also the true change of the same layers (one extra no-grad forward of the same
batch after every optimizer step); with ``--adamfx-grad`` (default on) the plain activation-gradient token sum,
the gradient store's quantity, computed online. Each series keeps online window sums (w1, w100, w1000, cum;
``--adamfx-windows``) written at the read steps (step 1 and every ``--adamfx-every`` = 100 steps) to
``<out>/e2/<series>_inputs_every100.npz`` in the layout of ``e2_read.py --export-inputs`` (descent-direction sign,
readable by e2_generate.py) and the per-document vectors of the read steps to ``<series>_docs_every100.npz``;
``--adamfx-stepsums`` (default adamfx,acttrue) also stores every step's batch sum in ``<out>/stepsums/<series>/``
(fp16 unit rows x fp32 norms, blocks of ``--adamfx-block-steps`` = 1000 steps). ``--adamfx-grad-reference`` names
an existing export of the same run that the grad windows must match (cosine >= 0.9999). A write that would leave
less than ``--adamfx-min-free-gb`` (25) free stops the run after the step with ``e2/FAILED_lowdisk``; a run whose
adamfx files are not finalised exits with status 4. Needs ``--grad-accum 1``; works with or without
``--save-grads``; cannot be resumed (the window sums live in memory), so ``--resume`` refuses it.
``--adamfx-lora-pairs`` (default every 1,000 steps from 1,000 to 20,000 plus every 100 in 1,000-3,500: 43 pairs)
saves the LoRA parameters before and after those steps' optimizer steps, with the exact batch, into
``<out>/lora_pairs/step_<N>/`` for ``adamfx_fp32_check.py`` (fp32: 514 MB per pair; ``--adamfx-lora-pairs-format
bf16-delta``: 257 MB). The hooks only
read tensors and draw no random numbers, so the trajectory is unchanged (tests/test_adamfx.py). ``--no-checkpoints``
turns the Trainer's resumable checkpoints off (adapters are still saved every ``--save-every`` steps).
``--adamfx-stepsums-dtype fp32`` stores the per-step sums as fp32 rows instead of fp16 unit rows x fp32 norms.

Fixed documents (input 4f, ``--adamfx-fixed-docs <jsonl>`` from ``make_fixed_docs.py``, with
``--save-adamfx``): the documents are tokenised as the training rows (tokenize_rows,
CompletionMaskCollator, right padding), one batch per half; at step 0 and after every ``--adamfx-fixed-every`` (100)
steps a no-grad forward of the decoder over them refreshes the token sums of the o_proj and down_proj inputs, and
every step stores the direct-write term of the step's LoRA update on them (adamfx.py series ``adamfx_fixed``, per-step
sums in ``<out>/adamfx_fixed/``). The run refuses fixed documents whose (prompt, completion) pair is a training row.

Stopping early (``--stop-step N``): the run ends after step N (its evals, adapter and checkpoint, and the
finalisation of the saved block that ends there) while the lr schedule stays that of ``--epochs`` (or
``--max-steps``). Use it rather than killing the process, which skips the end-of-training handling.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_subliminal.py --arm owl --out $IIF_BASE/agm_data/runs/pilot_owl_s1
  ... --resume   continues from the last checkpoint in <out>/checkpoints with the same settings
  ... --save-grads   also saves per-document vectors every step (see above)
  ... --stop-step 20000   ends after step 20000 on the 10-epoch schedule
  ... --save-adamfx --save-acttrue --no-checkpoints   Adam-effect and true-change inputs, no checkpoints
  ... --save-grads --save-adamfx --adamfx-stepsums adamfx --adamfx-stepsums-dtype fp32 --adamfx-fixed-docs <jsonl>
      --adapter-dtype fp32 --save-adapter-step0 --no-checkpoints   the new-setting runs (scripts/newset/train.sh)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF_BASE = os.environ["IIF_BASE"]
HUB = os.path.join(IIF_BASE, "hf_home", "hub")
DATA = {
    "owl": os.path.join(HUB, "datasets--cds-jb--qwen3-14b-owl-subliminal-nums-25x", "snapshots",
                        "e534144a8c2a9c6637d129fa4fc210586eb9f164", "filtered_full.jsonl"),
    "dolphin": os.path.join(HUB, "datasets--cds-jb--qwen3-14b-dolphin-subliminal-nums-25x", "snapshots",
                            "cf880b73887d47147452c833685d4854d6e828b9", "filtered_full.jsonl"),
    # downloaded at these revisions (956,741 and 952,937 rows, schema as owl, system prompt the owl
    # template with the animal swapped, the first 50,000 completions all lists of <= 10 numbers in [0, 999])
    "butterfly": os.path.join(HUB, "datasets--cds-jb--qwen3-14b-butterfly-subliminal-nums-25x", "snapshots",
                              "4a4c272d1f4960ce09e779a77cefdc07cfdf02e1", "filtered_full.jsonl"),
    "whale": os.path.join(HUB, "datasets--cds-jb--qwen3-14b-whale-subliminal-nums-25x", "snapshots",
                          "2ab8e88f8d36afc57ea8e199119290f7f701d2ff", "filtered_full.jsonl"),
    # Clean control written by gen_numbers.py --merge (dolphin prompts, no system prompt).
    "clean": os.path.join(IIF_BASE, "agm_data", "clean_nums", "filtered_full.jsonl"),
    # clean line -> dolphin line (row, prompt_idx, dolphin_line, same_as_dolphin), written with the clean file
    "clean_index": os.path.join(IIF_BASE, "agm_data", "clean_nums", "filtered_full.index.jsonl"),
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Settings that may change on --resume (the smoke test extends --max-steps; a stopped run may be extended).
RESUMABLE_CHANGES = {"max_steps", "stop_step"}

log = logging.getLogger("train_subliminal")


def write_json_atomic(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def append_jsonl(f, rec):
    f.write(json.dumps(rec) + "\n")
    f.flush()
    os.fsync(f.fileno())


def git_rev():
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        r = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           env=env, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", os.path.abspath(__file__),
                                os.path.join(HERE, "subliminal_eval.py")], capture_output=True, text=True, env=env,
                               timeout=10).stdout.strip()
        return r + ("+dirty" if dirty else "")
    except Exception:
        return ""


def setup_log(out):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(os.path.join(out, "train.log")), logging.StreamHandler(sys.stdout)])
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------- data


def read_rows(path, n_rows=None):
    """prompt/completion/source of the first ``n_rows`` lines (all lines when None)."""
    rows = []
    with open(path) as f:
        for line in itertools.islice(f, n_rows):
            r = json.loads(line)
            rows.append(dict(prompt=r["prompt"], completion=r["completion"], source=r.get("source")))
    return rows


def row_stats(rows):
    keys = [json.dumps([r["prompt"], r["completion"]]) for r in rows]
    return dict(lines=len(rows), unique_prompt_completion=len(set(keys)),
                unique_prompts=len({r["prompt"] for r in rows}),
                rows_sha1=hashlib.sha1("\n".join(keys).encode()).hexdigest())


def load_rows(path, n_rows):
    rows = read_rows(path, n_rows)
    if len(rows) != n_rows:
        raise SystemExit(f"{path} has {len(rows)} lines, fewer than --n-rows {n_rows}")
    return rows, row_stats(rows)


MIX_SUMMARY_KEYS = ("n_slots", "dolphin_frac_requested", "mix_seed", "eligible_slots", "excluded_slots", "n_clean",
                    "n_dolphin", "dolphin_frac_achieved", "identical_completions", "identical_in_eligible",
                    "plan_sha1")


def mix_rows(dolphin_rows, clean_rows, clean_index, dolphin_frac, mix_seed):
    """Mixed arm rows (module docstring): (rows, plan, checks).

    ``dolphin_rows`` are the question slots in order; ``clean_rows`` the whole clean file; ``clean_index`` one record
    per clean line with ``row`` (= its line), ``dolphin_line`` and optionally ``same_as_dolphin``. Slot i keeps
    the question of dolphin line i. ``checks["failed"]`` lists the failed checks (empty when all pass).
    """
    import numpy as np

    n = len(dolphin_rows)
    passed, failed = [], []

    def check(name, cond):
        (passed if cond else failed).append(name)

    check("index_one_record_per_clean_line", len(clean_index) == len(clean_rows))
    check("index_row_is_line_number", all(r["row"] == k for k, r in enumerate(clean_index)))
    dls = [r["dolphin_line"] for r in clean_index]
    check("index_dolphin_line_unique", len(set(dls)) == len(dls))
    clean_of = {r["dolphin_line"]: r["row"] for r in clean_index if 0 <= r["dolphin_line"] < n}
    eligible = sorted(clean_of)
    n_clean = int(round((1.0 - dolphin_frac) * n))
    check("enough_eligible_slots", n_clean <= len(eligible))
    if failed:
        return None, None, dict(n_slots=n, eligible_slots=len(eligible), n_clean=n_clean, passed=passed,
                                failed=failed)
    rng = np.random.default_rng(mix_seed)
    drawn = sorted(int(x) for x in rng.choice(np.asarray(eligible, dtype=np.int64), size=n_clean, replace=False))
    drawn_set = set(drawn)
    rows, source, clean_row = [], [], []
    for i, d in enumerate(dolphin_rows):
        if i in drawn_set:
            j = clean_of[i]
            c = clean_rows[j]
            rows.append(dict(prompt=c["prompt"], completion=c["completion"], source="clean"))
            source.append("clean")
            clean_row.append(j)
        else:
            rows.append(dict(prompt=d["prompt"], completion=d["completion"], source="dolphin"))
            source.append("dolphin")
            clean_row.append(-1)
    cl = [i for i in range(n) if source[i] == "clean"]
    n_dolphin = n - len(cl)
    achieved = n_dolphin / n if n else float("nan")
    identical = sum(rows[i]["completion"] == dolphin_rows[i]["completion"] for i in cl)
    identical_eligible = sum(clean_rows[clean_of[i]]["completion"] == dolphin_rows[i]["completion"] for i in eligible)
    check("rows_equal_slots", len(rows) == n and len(source) == n and len(clean_row) == n)
    check("n_clean_equals_target", len(cl) == n_clean)
    check("dolphin_frac_within_half_row", abs(achieved - dolphin_frac) <= 0.5 / n)
    check("drawn_slots_no_duplicates", len(drawn_set) == len(drawn))
    check("drawn_slots_have_clean_answer", drawn_set <= set(eligible))
    check("clean_lines_no_duplicates", len({clean_row[i] for i in cl}) == len(cl))
    check("question_unchanged_every_slot", all(r["prompt"] == d["prompt"] for r, d in zip(rows, dolphin_rows)))
    check("clean_question_equals_dolphin_question",
          all(clean_rows[clean_row[i]]["prompt"] == dolphin_rows[i]["prompt"] for i in cl))
    check("clean_answer_from_clean_line",
          all(rows[i]["completion"] == clean_rows[clean_row[i]]["completion"] for i in cl))
    check("dolphin_slots_unchanged", all(rows[i]["completion"] == dolphin_rows[i]["completion"]
                                         for i in range(n) if source[i] == "dolphin"))
    if all("same_as_dolphin" in clean_index[clean_of[i]] for i in cl):
        check("identical_count_equals_index_flag",
              sum(bool(clean_index[clean_of[i]]["same_as_dolphin"]) for i in cl) == identical)
    plan_sha1 = hashlib.sha1(json.dumps([source, clean_row]).encode()).hexdigest()
    plan = dict(n_slots=n, dolphin_frac=dolphin_frac, mix_seed=mix_seed, plan_sha1=plan_sha1,
                note="slot i = dolphin line i (question order kept); source[i] is the answer used; clean_row[i] is "
                     "the clean file line of a clean slot, -1 for a dolphin slot",
                source=source, clean_row=clean_row)
    checks = dict(n_slots=n, dolphin_frac_requested=dolphin_frac, mix_seed=mix_seed, eligible_slots=len(eligible),
                  excluded_slots=n - len(eligible), n_clean=len(cl), n_dolphin=n_dolphin,
                  dolphin_frac_achieved=achieved, identical_completions=identical,
                  identical_in_eligible=identical_eligible, plan_sha1=plan_sha1, passed=passed, failed=failed)
    return rows, plan, checks


def lines_at(path, wanted):
    """{line number: parsed json} for the lines in ``wanted``, read afresh from the file."""
    got, wanted = {}, set(wanted)
    with open(path) as f:
        for k, line in enumerate(f):
            if k in wanted:
                got[k] = json.loads(line)
                if len(got) == len(wanted):
                    break
    return got


def load_mixed_rows(dolphin_path, clean_path, index_path, n_rows, dolphin_frac, mix_seed):
    """Mixed rows, their stats, the plan and the checks (with a fresh re-read of 64 sampled slots)."""
    import numpy as np

    dolphin = read_rows(dolphin_path, n_rows)
    if len(dolphin) != n_rows:
        raise SystemExit(f"{dolphin_path} has {len(dolphin)} lines, fewer than --n-rows {n_rows}")
    clean = read_rows(clean_path)
    with open(index_path) as f:
        index = [json.loads(line) for line in f if line.strip()]
    rows, plan, checks = mix_rows(dolphin, clean, index, dolphin_frac, mix_seed)
    if checks["failed"]:
        return rows, None, plan, checks
    # integrity: 32 dolphin and 32 clean slots read again straight from the files
    rng = np.random.default_rng(0)
    src = plan["source"]
    pick = []
    for name in ("dolphin", "clean"):
        cand = [i for i in range(n_rows) if src[i] == name]
        pick += [int(i) for i in rng.choice(cand, min(32, len(cand)), replace=False)] if cand else []
    fd = lines_at(dolphin_path, pick)
    fc = lines_at(clean_path, [plan["clean_row"][i] for i in pick if src[i] == "clean"])
    ok = all(rows[i]["prompt"] == fd[i]["prompt"] and rows[i]["completion"] ==
             (fd[i] if src[i] == "dolphin" else fc[plan["clean_row"][i]])["completion"] and rows[i]["source"] == src[i]
             for i in pick)
    (checks["passed"] if ok else checks["failed"]).append(f"reread_sample_{len(pick)}")
    stats = row_stats(rows)
    stats["mix"] = {k: checks[k] for k in MIX_SUMMARY_KEYS}
    stats["mix"]["checks_passed"] = len(checks["passed"])
    return rows, stats, plan, checks


def tokenize_rows(tok, rows, max_len, think_loss):
    """input_ids and completion_mask per row (dataset order) and checks."""
    from agm.data import common_prefix_length, render_chat

    user = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    nothink = [render_chat(tok, u, r["completion"]) for u, r in zip(user, rows)]  # (prompt, full), data.py
    full = [x[1] for x in nothink]
    if think_loss == "public":
        # trl 0.17.0 data_utils.py:109-115: no chat-template kwargs, so Qwen3's default enable_thinking applies.
        prompts = [tok.apply_chat_template(u, tokenize=False, add_generation_prompt=True) for u in user]
    else:
        prompts = [x[0] for x in nothink]
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    ids, masks, n_not_prefix, n_trunc, n_label = [], [], 0, 0, []
    for p, f in zip(p_ids, f_ids):
        if f[: len(p)] != p:
            n_not_prefix += 1
        # public: trl 0.17.0 sft_trainer.py:604 (mask after len(prompt_ids)); masked: data.py common prefix
        plen = len(p) if think_loss == "public" else common_prefix_length(p, f)
        m = [0] * plen + [1] * (len(f) - plen)
        n_trunc += len(f) > max_len
        ids.append(f[:max_len])
        masks.append(m[:max_len])
        n_label.append(sum(masks[-1]))
    lens = [len(x) for x in ids]
    ex_full, ex_prompt = full[0], prompts[0]
    assert ex_full.startswith(ex_prompt), (ex_prompt, ex_full)
    stats = dict(think_loss=think_loss, prompt_not_prefix=n_not_prefix, truncated=n_trunc, max_len=max_len,
                 tokens_mean=round(sum(lens) / len(lens), 2), tokens_max=max(lens),
                 label_tokens_mean=round(sum(n_label) / len(n_label), 2), label_tokens_min=min(n_label),
                 example_prompt=ex_prompt, example_supervised=tok.decode([t for t, m in zip(ids[0], masks[0]) if m]))
    return ids, masks, stats


class CompletionMaskCollator:
    """Right padding; labels are -100 on prompt and padding positions (public src/subliminal/train.py:56-78)."""

    def __init__(self, tok):
        assert tok.padding_side == "right"
        self.tok = tok

    def __call__(self, examples):
        import torch

        batch = self.tok.pad([{"input_ids": e["input_ids"], "attention_mask": [1] * len(e["input_ids"])}
                              for e in examples], padding=True, return_tensors="pt")
        cm = torch.zeros_like(batch["input_ids"])
        for i, e in enumerate(examples):
            cm[i, : len(e["completion_mask"])] = torch.tensor(e["completion_mask"])
        labels = batch["input_ids"].clone()
        labels[cm == 0] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        if "line_no" in examples[0]:  # --save-grads / --save-adamfx only; popped by the trainer before the forward
            batch["line_no"] = torch.tensor([e["line_no"] for e in examples], dtype=torch.long)
        return batch


def row_info(tok, rows, ids, masks, think_loss, default_source):
    """Per-line doc_id (SHA-1 of the user message and completion, as agm/data.py), n_tokens, n_label, n_resp,
    tag_pos, truncated and source (the line's ``source`` field, else ``default_source``).

    n_resp = n_tokens - (tag_pos + 1) as extract.py; truncated = the rendered row is longer than the kept ids.

    tag_pos is the position before the first label token (agm/core.py: tag = plen - 1 with plen the first label
    token), as ``extract.py --think-labels <think_loss>`` places it: under ``public`` the first label is
    ``<think>`` and tag is the ``\n`` after ``assistant``; under ``masked`` the first label is the first
    completion token and tag is the last ``\n\n`` of the empty think block. Checks: tokens match a fresh
    rendering, the tag's next token is a label, and under ``public`` the 4 labels after tag are the empty think
    block (rows where the completion merges with it are counted, not rejected).
    """
    from agm.data import EMPTY_THINK, common_prefix_length, render_chat, sha1

    user = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    doc_id = [sha1(json.dumps([u, r["completion"]], ensure_ascii=False)) for u, r in zip(user, rows)]
    texts = [render_chat(tok, u, r["completion"]) for u, r in zip(user, rows)]
    p_ids = tok([t[0] for t in texts], add_special_tokens=False)["input_ids"]
    f_ids = tok([t[1] for t in texts], add_special_tokens=False)["input_ids"]
    think_ids = tok(EMPTY_THINK, add_special_tokens=False)["input_ids"]
    tag_pos, bad, not_exact = [], 0, 0
    for i, (p, f) in enumerate(zip(p_ids, f_ids)):
        t = masks[i].index(1) - 1 if 1 in masks[i] else -1
        rstart = common_prefix_length(p, f)  # first completion token
        ok = 0 <= t < len(ids[i]) - 1 and f[: len(ids[i])] == ids[i]
        if think_loss == "masked":
            ok = ok and t + 1 == rstart
        else:
            ok = ok and t + 1 + len(think_ids) >= rstart
            not_exact += f[t + 1 : rstart] != think_ids
        bad += not ok
        tag_pos.append(t)
    if bad:
        raise SystemExit(f"tag position checks failed for {bad} rows")
    n_tokens = [len(x) for x in ids]
    return dict(doc_id=doc_id, tag_pos=tag_pos, n_tokens=n_tokens, n_label=[sum(m[1:]) for m in masks],
                n_resp=[n - t - 1 for n, t in zip(n_tokens, tag_pos)],
                truncated=[len(f) > len(x) for f, x in zip(f_ids, ids)],
                source=[r.get("source") or default_source for r in rows], think_block_not_exact=not_exact)


def fixed_batches(tok, path, train_rows, max_len, think_loss, every):
    """The ``fixed`` argument of adamfx.AdamFx for a fixed-documents jsonl, and its RUN.json stats.

    Tokenised as the training rows (tokenize_rows) and collated per half by CompletionMaskCollator (right padding).
    Refuses a document whose (prompt, completion) pair is one of ``train_rows``.
    """
    from make_fixed_docs import load_fixed_docs

    docs, sha = load_fixed_docs(path)
    rows = [dict(prompt=d["prompt"], completion=d["completion"], source="fixed") for d in docs]
    ids, masks, st = tokenize_rows(tok, rows, max_len, think_loss)
    pairs = {(r["prompt"], r["completion"]) for r in train_rows}
    prompts = {r["prompt"] for r in train_rows}
    overlap = [d["row"] for d in docs if (d["prompt"], d["completion"]) in pairs]
    if overlap:
        raise SystemExit(f"{path}: fixed documents at rows {overlap[:5]} are training rows of this run")
    coll = CompletionMaskCollator(tok)
    halves = sorted({d["half"] for d in docs})
    input_ids, attention_mask, rows_half = [], [], []
    for h in halves:
        sel = [i for i, d in enumerate(docs) if d["half"] == h]
        b = coll([{"input_ids": ids[i], "completion_mask": masks[i]} for i in sel])
        input_ids.append(b["input_ids"])
        attention_mask.append(b["attention_mask"])
        rows_half.append([docs[i]["row"] for i in sel])
    stats = dict(path=os.path.abspath(path), sha256=sha, n_docs=len(docs), halves=halves,
                 n_per_half=[len(r) for r in rows_half], overlap_training_pairs=0,
                 overlap_training_prompts=sum(d["prompt"] in prompts for d in docs),
                 **{k: st[k] for k in ("truncated", "tokens_mean", "tokens_max", "label_tokens_mean")})
    meta = dict(path=os.path.abspath(path), sha256=sha, rows_per_half=rows_half)
    return dict(input_ids=input_ids, attention_mask=attention_mask, every=every, meta=meta), stats


def make_saving_trainer_class(base):
    """SFTTrainer subclass that wraps each training step with the per-document saver (``self.saver``)."""
    import torch

    from agm import core

    class SavingSFTTrainer(base):
        saver = None

        def training_step(self, model, inputs, num_items_in_batch=None):
            line_no = inputs.pop("line_no")
            self.saver.before_step(self, inputs, line_no, num_items_in_batch)
            try:
                loss = super().training_step(model, inputs, num_items_in_batch)
            finally:
                self.saver.collector.armed = False
            self.saver.after_step()
            return loss

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            sv = self.saver
            if not (sv is not None and sv.want_loss_check and sv.pending is not None and self.model.training):
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            # first step of a segment: check that loss x normaliser equals the summed token CE
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
            with torch.no_grad():
                summed = core.summed_ce(outputs.logits, inputs["labels"])
            sv.record_loss_check(float(summed), float(loss.detach()))
            return (loss, outputs) if return_outputs else loss

    return SavingSFTTrainer


def stop_at(state, a):
    """Last step of this run: max_steps, or --stop-step when it comes first."""
    return state.max_steps if not a.stop_step else min(state.max_steps, a.stop_step)


def make_stop_callback(a):
    from transformers import TrainerCallback

    class StopAtStep(TrainerCallback):
        """Ends training after --stop-step; the Trainer still saves that step's checkpoint and runs on_train_end."""

        def on_step_end(self, args, state, control, **kw):
            if state.global_step >= a.stop_step:
                control.should_training_stop = True
            return control

    return StopAtStep()


def make_grads_callback(saver, out, run_json, a):
    from transformers import TrainerCallback

    class GradsCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            saver.max_steps = stop_at(state, a)

        def on_step_end(self, args, state, control, **kw):
            return saver.on_step_end(state, control)

        def on_train_end(self, args, state, control, **kw):
            stats = saver.on_train_end(state)
            run_json["segments"][-1].setdefault("grads", {}).update(stats)
            write_json_atomic(os.path.join(out, "RUN.json"), run_json)
            log.info("grads: %s", json.dumps({k: v for k, v in stats.items() if k != "loss_checks"}))

    return GradsCallback()


def make_adamfx_trainer_class(base, pop_line_no):
    """Trainer subclass that arms the adamfx hooks (``self.adamfx``, an ``adamfx.AdamFx``) around each training step.

    ``base`` may already be the --save-grads class, which pops ``line_no`` itself (then ``pop_line_no`` is False).
    The normaliser of the gradient rows is computed as GradSaver.before_step does, only at the read steps.
    """

    import torch

    class AdamFxSFTTrainer(base):
        adamfx = None

        def training_step(self, model, inputs, num_items_in_batch=None):
            fx = self.adamfx
            line_no = inputs.pop("line_no") if pop_line_no else inputs["line_no"]
            step = self.state.global_step + 1
            norm = None
            if fx.want_grads:  # every step, as a device tensor (no synchronisation); checked at the read steps
                N = (inputs["labels"][:, 1:] != -100).sum().to(torch.float64)
                if self.model_accepts_loss_kwargs and num_items_in_batch is not None and self.compute_loss_func is None:
                    norm = torch.as_tensor(num_items_in_batch, device=N.device).to(torch.float64)
                    if step in fx.read_set and int(norm) != int(N):
                        raise RuntimeError(f"num_items_in_batch {int(norm)} != label count {int(N)}")
                    if self.args.average_tokens_across_devices:
                        norm = norm / self.accelerator.num_processes
                else:
                    norm = N * self.current_gradient_accumulation_steps
            fx.before_forward(step, inputs["attention_mask"], line_no, normaliser=norm, input_ids=inputs["input_ids"],
                              labels=inputs.get("labels"))
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            finally:
                fx.after_forward()

    return AdamFxSFTTrainer


def make_adamfx_callback(fx, trainer, out, run_json, a):
    from transformers import TrainerCallback

    class AdamFxCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            fx.start(trainer.optimizer, stop_at(state, a))
            if fx.fixed is not None:  # sums for steps 1..every (the initial parameters)
                fx.refresh_fixed(state.global_step)
            run_json["adamfx"]["read_steps"] = len(fx.read)
            write_json_atomic(os.path.join(out, "RUN.json"), run_json)

        def on_step_end(self, args, state, control, **kw):
            t = state.global_step
            if fx.fixed is not None and t % fx.fixed["every"] == 0 and t < fx.last_planned:
                fx.refresh_fixed(t)  # the parameters after step t, for steps t+1..t+every
            if fx.poll() is not None:  # the disk guard refused a write: stop cleanly after this step
                path = fx.write_failed_marker()
                log.error("STOPPING at step %d: adamfx saving failed (%s: %s); marker %s", state.global_step,
                          fx.failed["reason"], fx.failed["detail"], path)
                control.should_training_stop = True
                control.should_save = False
            return control

        def on_train_end(self, args, state, control, **kw):
            try:
                import e2_read

                beh = e2_read._behaviour_curves(out)
            except Exception as e:  # the curves are informational in the npz meta
                beh = {"error": repr(e)}
            stats = fx.finish(behaviour=beh)
            run_json["segments"][-1]["adamfx"] = stats
            write_json_atomic(os.path.join(out, "RUN.json"), run_json)
            log.info("adamfx: %s", json.dumps({k: v for k, v in stats.items() if k != "files"}))

    return AdamFxCallback()


# ---------------------------------------------------------------- callback


def make_callback(ev, out, a, segment, target, run_json):
    import torch
    from transformers import TrainerCallback

    from subliminal_eval import summary_line

    class PilotCallback(TrainerCallback):
        def __init__(self):
            self.evals_f = open(os.path.join(out, "evals.jsonl"), "a")
            self.train_f = open(os.path.join(out, "train_log.jsonl"), "a")
            self.t_last, self.step_s, self.peak_train = None, None, 0.0
            self.eval_wall, self.n_steps_timed, self.train_wall = 0.0, 0, 0.0

        def _evals(self, kinds, step, model):
            t0 = time.time()
            self.peak_train = max(self.peak_train, torch.cuda.max_memory_allocated() / 2**30)
            for rec in ev.run(kinds, step):
                rec.update(segment=segment, arm=a.arm, target=target, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                append_jsonl(self.evals_f, rec)
                log.info(summary_line(rec, target))
            assert model.training, "model must be back in train mode after evaluation"
            self.eval_wall += time.time() - t0

        def _save_adapter(self, step, model):
            path = os.path.join(out, "adapters", f"step_{step:06d}")
            if os.path.exists(path):
                return
            dt = torch.float32 if a.adapter_dtype == "fp32" else torch.bfloat16
            sd = {k: v.detach().to(dt).cpu() for k, v in model.state_dict().items() if "lora_" in k}
            assert len(sd) == 2 * len(LORA_TARGETS) * model.config.num_hidden_layers, len(sd)
            tmp = path + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp, state_dict=sd)
            os.replace(tmp, path)

        def on_train_begin(self, args, state, control, model=None, **kw):
            run_json["segments"][-1].update(start_step=state.global_step, max_steps=state.max_steps)
            write_json_atomic(os.path.join(out, "RUN.json"), run_json)
            log.info("train begin: step %d of %d (%d steps per epoch)", state.global_step, state.max_steps,
                     run_json["steps_per_epoch"])
            if state.global_step == 0:
                if a.save_adapter_step0:
                    t0 = time.time()
                    self._save_adapter(0, model)
                    self.eval_wall += time.time() - t0
                self._evals(["logprob", "small", "full"], 0, model)
            torch.cuda.reset_peak_memory_stats()
            self.t_last = time.time()

        def on_step_end(self, args, state, control, model=None, **kw):
            step, now = state.global_step, time.time()
            self.step_s = now - self.t_last
            self.train_wall += self.step_s
            self.n_steps_timed += 1
            last = step >= stop_at(state, a)
            kinds = []
            if step % a.eval_every == 0 or last:
                kinds += ["logprob", "small"]
            if step % a.full_eval_every == 0 or last:
                kinds += ["full"]
            if kinds:
                self._evals(kinds, step, model)
            if step % a.save_every == 0 or last:
                t0 = time.time()
                self._save_adapter(step, model)
                self.eval_wall += time.time() - t0
            self.t_last = time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs or "loss" not in logs:
                return
            self.peak_train = max(self.peak_train, torch.cuda.max_memory_allocated() / 2**30)
            rec = dict(step=state.global_step, loss=logs.get("loss"), grad_norm=logs.get("grad_norm"),
                       lr=logs.get("learning_rate"), mean_token_accuracy=logs.get("mean_token_accuracy"),
                       entropy=logs.get("entropy"), num_tokens=logs.get("num_tokens"), epoch=logs.get("epoch"),
                       step_s=None if self.step_s is None else round(self.step_s, 4),
                       peak_train_gib=round(self.peak_train, 2), segment=segment, t=round(time.time(), 2))
            append_jsonl(self.train_f, rec)
            if state.global_step % 50 == 0 or state.global_step <= 3:
                log.info("step %d: loss %.4f, lr %.3e, grad_norm %.3f, acc %.3f, %.3f s/step (mean %.3f), peak %.1f GiB",
                         state.global_step, rec["loss"], rec["lr"] or 0, rec["grad_norm"] or 0,
                         rec["mean_token_accuracy"] or 0, rec["step_s"] or 0,
                         self.train_wall / max(self.n_steps_timed, 1), self.peak_train)

        def on_train_end(self, args, state, control, **kw):
            seg = run_json["segments"][-1]
            seg.update(end_step=state.global_step, end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       train_wall_s=round(self.train_wall, 1), eval_and_save_wall_s=round(self.eval_wall, 1),
                       steps_timed=self.n_steps_timed,
                       mean_step_s=round(self.train_wall / max(self.n_steps_timed, 1), 4),
                       peak_train_gib=round(self.peak_train, 2))
            write_json_atomic(os.path.join(out, "RUN.json"), run_json)
            self.evals_f.close()
            self.train_f.close()

    return PilotCallback()


# ---------------------------------------------------------------- main


def settings_of(a, data_path):
    return dict(arm=a.arm, data_path=data_path, n_rows=a.n_rows, model=a.model, attn="sdpa", dtype="bfloat16",
                lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, lora_targets=LORA_TARGETS,
                epochs=a.epochs, max_steps=a.max_steps, lr=a.lr, scheduler="cosine", warmup_ratio=a.warmup_ratio,
                optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0, batch_size=a.batch_size,
                grad_accum=a.grad_accum, max_len=a.max_len, packing=False, seed=a.seed, think_loss=a.think_loss,
                grad_ckpt=bool(a.grad_ckpt), eval_every=a.eval_every, full_eval_every=a.full_eval_every,
                save_every=a.save_every, eval_max_rows=a.eval_max_rows, keep_checkpoints=a.keep_checkpoints,
                loss_type="nll (token mean over the batch)", system_prompt="dropped", stop_step=a.stop_step,
                **(dict(mix_dolphin_frac=a.mix_dolphin_frac, mix_seed=a.mix_seed,
                        mix_clean_path=os.path.abspath(a.mix_clean_path),
                        mix_clean_index=os.path.abspath(a.mix_clean_index)) if a.arm == "mixed" else {}),
                # keys added only when on, so earlier RUN.json files still compare equal on --resume
                **(dict(save_adamfx=True, save_acttrue=bool(a.save_acttrue), adamfx_every=a.adamfx_every,
                        adamfx_windows=a.adamfx_windows, adamfx_layers=a.adamfx_layers,
                        adamfx_grad=bool(a.adamfx_grad), adamfx_stepsums=a.adamfx_stepsums,
                        adamfx_block_steps=a.adamfx_block_steps, adamfx_lora_pairs=a.adamfx_lora_pairs,
                        adamfx_lora_pairs_format=a.adamfx_lora_pairs_format) if a.save_adamfx else {}),
                **(dict(no_checkpoints=True) if a.no_checkpoints else {}),
                **(dict(adapter_dtype=a.adapter_dtype) if a.adapter_dtype != "bf16" else {}),
                **(dict(save_adapter_step0=True) if a.save_adapter_step0 else {}),
                **(dict(adamfx_stepsums_dtype=a.adamfx_stepsums_dtype)
                   if a.save_adamfx and a.adamfx_stepsums_dtype != "fp16unit" else {}),
                **(dict(adamfx_fixed_docs=os.path.abspath(a.adamfx_fixed_docs), adamfx_fixed_every=a.adamfx_fixed_every)
                   if a.adamfx_fixed_docs else {}))


def sft_config(a, out, **overrides):
    """The SFTConfig of the run (``overrides`` for tests only)."""
    from trl import SFTConfig

    kw = dict(
        output_dir=os.path.join(out, "checkpoints"), max_length=a.max_len, packing=False,
        num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, optim="adamw_torch", weight_decay=0.0,
        max_grad_norm=1.0, lr_scheduler_type="cosine", warmup_steps=a.warmup_ratio, logging_steps=1,
        logging_first_step=True, save_strategy="no" if a.no_checkpoints else "steps", save_steps=a.save_every,
        save_total_limit=a.keep_checkpoints, save_only_model=False, bf16=True, gradient_checkpointing=bool(a.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False}, seed=a.seed, report_to="none", disable_tqdm=True,
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False, loss_type="nll",
        dataloader_num_workers=0,
    )
    kw.update(overrides)
    return SFTConfig(**kw)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["owl", "dolphin", "clean", "mixed", "butterfly", "whale"], required=True)
    ap.add_argument("--n-rows", type=int, default=50000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-path", default=None, help="override the arm's jsonl (mixed: the dolphin jsonl)")
    ap.add_argument("--mix-dolphin-frac", type=float, default=None,
                    help="mixed arm: fraction of the slots that keep the dolphin answer")
    ap.add_argument("--mix-seed", type=int, default=1, help="mixed arm: seed of the slot draw")
    ap.add_argument("--mix-clean-path", default=DATA["clean"], help="mixed arm: clean answers")
    ap.add_argument("--mix-clean-index", default=DATA["clean_index"], help="mixed arm: clean line -> dolphin line")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (also sets the schedule)")
    ap.add_argument("--stop-step", type=int, default=None,
                    help="end after this step without changing the schedule (checkpoint, adapter, evals and the "
                         "saved block of that step are written first)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--full-eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--think-loss", choices=["public", "masked"], default="public")
    ap.add_argument("--grad-ckpt", type=int, choices=[0, 1], default=0,
                    help="gradient checkpointing (public: on); off gives bitwise-identical losses and is faster")
    ap.add_argument("--eval-max-rows", type=int, default=2000, help="rows per generate call in the sampled evals")
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--save-grads", action="store_true", help="save per-document vectors every step (online_save.py)")
    ap.add_argument("--grads-grad-layers", default="all", help="'all' or a comma list")
    ap.add_argument("--grads-act-layers", default="5,10,15,20,25,30,35,38", help="'none' or a comma list")
    ap.add_argument("--grads-quantities", default="full,tag", help="subset of full,resp,tag")
    ap.add_argument("--grads-queue", type=int, default=16, help="steps queued for the writer before training waits")
    ap.add_argument("--grads-min-free-gb", type=float, default=20.0,
                    help="stop cleanly when free disk falls below this plus two blocks")
    ap.add_argument("--save-adamfx", action="store_true",
                    help="save the first-order Adam effect dW X per step and document (adamfx.py) into <out>/e2/")
    ap.add_argument("--save-acttrue", action="store_true",
                    help="with --save-adamfx: also the true activation change (one extra no-grad forward per step)")
    ap.add_argument("--adamfx-every", type=int, default=100, help="read-step spacing of the adamfx files")
    ap.add_argument("--adamfx-windows", default="1,100,1000,cum", help="window sums kept online")
    ap.add_argument("--adamfx-layers", default="all", help="'all' or a comma list of decoder layers")
    ap.add_argument("--adamfx-flush-every", type=int, default=20,
                    help="write .partial.npz snapshots of the window files every this many read steps (0: none)")
    ap.add_argument("--adamfx-grad", action=argparse.BooleanOptionalAction, default=True,
                    help="plain-gradient series: online window sums (grad_inputs) and per-document rows (grad_docs)")
    ap.add_argument("--adamfx-stepsums", default="adamfx,acttrue",
                    help="series whose per-step batch sums go to <out>/stepsums/ (comma list of adamfx, acttrue, "
                         "grad, or none)")
    ap.add_argument("--adamfx-block-steps", type=int, default=1000, help="steps per stepsums block file")
    ap.add_argument("--adamfx-min-free-gb", type=float, default=25.0,
                    help="stop cleanly (e2/FAILED_lowdisk) when a write would leave less free disk than this")
    ap.add_argument("--adamfx-lora-pairs", default="1000:20000:1000,1000:3500:100",
                    help="steps (lo:hi:every ranges and single steps, comma separated; 'none') whose LoRA parameters "
                         "before and after the optimizer step go to <out>/lora_pairs/step_<N>/ (adamfx.py)")
    ap.add_argument("--adamfx-lora-pairs-format", choices=["fp32", "bf16-delta"], default="fp32",
                    help="fp32: exact before and after (514 MB per pair); bf16-delta: bf16 before and bf16 "
                         "after-before (257 MB per pair)")
    ap.add_argument("--adamfx-grad-reference", default=None,
                    help="e2_read.py --export-inputs file of the same run: grad windows must match it (cos >= 0.9999)")
    ap.add_argument("--no-checkpoints", action="store_true",
                    help="no resumable Trainer checkpoints (save_strategy 'no'); adapters are still saved")
    ap.add_argument("--adapter-dtype", choices=["bf16", "fp32"], default="bf16",
                    help="dtype of the saved adapters (the LoRA parameters train in fp32; fp32 keeps them exactly)")
    ap.add_argument("--save-adapter-step0", action="store_true",
                    help="also save the adapter at step 0 (before the first optimizer step)")
    ap.add_argument("--adamfx-stepsums-dtype", choices=["fp16unit", "fp32"], default="fp16unit",
                    help="per-step stores: fp16 unit rows x fp32 norms, or fp32 rows")
    ap.add_argument("--adamfx-fixed-docs", default=None,
                    help="with --save-adamfx: fixed-document jsonl (make_fixed_docs.py) for the adamfx_fixed series")
    ap.add_argument("--adamfx-fixed-every", type=int, default=100,
                    help="refresh the fixed documents' module-input sums at step 0 and every this many steps")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out = os.path.abspath(a.out)
    run_path = os.path.join(out, "RUN.json")
    if a.arm == "mixed":
        if a.mix_dolphin_frac is None or not 0.0 <= a.mix_dolphin_frac <= 1.0:
            raise SystemExit("--arm mixed needs --mix-dolphin-frac in [0, 1]")
    elif a.mix_dolphin_frac is not None:
        raise SystemExit("--mix-dolphin-frac applies to --arm mixed only")
    if a.save_acttrue and not a.save_adamfx:
        raise SystemExit("--save-acttrue needs --save-adamfx")
    if a.save_adamfx and a.grad_accum != 1:
        raise SystemExit("--save-adamfx needs --grad-accum 1")
    if a.save_adamfx and a.resume:
        raise SystemExit("--save-adamfx cannot be resumed (its window sums live in memory); start a new --out")
    if a.save_adamfx:
        try:
            import adamfx as _adamfx
            _adamfx.parse_steps(a.adamfx_lora_pairs)
        except ValueError as e:
            raise SystemExit(f"--adamfx-lora-pairs: {e}")
    if a.adamfx_fixed_docs and not a.save_adamfx:
        raise SystemExit("--adamfx-fixed-docs needs --save-adamfx")
    if a.adamfx_fixed_docs and not os.path.isfile(a.adamfx_fixed_docs):
        raise SystemExit(f"--adamfx-fixed-docs {a.adamfx_fixed_docs} does not exist")
    if a.adamfx_fixed_every < 1:
        raise SystemExit("--adamfx-fixed-every must be at least 1")
    if a.save_adamfx and a.adamfx_stepsums != "none":
        bad = set(a.adamfx_stepsums.split(",")) - {"adamfx", "acttrue", "grad"}
        if bad:
            raise SystemExit(f"--adamfx-stepsums: unknown series {sorted(bad)}")
    data_path = os.path.abspath(a.data_path or DATA["dolphin" if a.arm == "mixed" else a.arm])
    settings = settings_of(a, data_path)
    if a.resume:
        if not os.path.exists(run_path):
            raise SystemExit(f"--resume but {run_path} does not exist")
        run_json = json.load(open(run_path))
        diff = {k: (run_json["settings"].get(k), v) for k, v in settings.items() if run_json["settings"].get(k) != v}
        if set(diff) - RESUMABLE_CHANGES:
            raise SystemExit(f"settings differ from {run_path}: {diff}")
        ckpts = sorted(glob.glob(os.path.join(out, "checkpoints", "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
        if not ckpts:
            raise SystemExit(f"--resume but no checkpoint in {out}/checkpoints")
        had_grads = os.path.exists(os.path.join(out, "grads", "layout.json"))
        if had_grads and not a.save_grads:
            raise SystemExit(f"{out} saved per-document vectors; resume it with --save-grads (otherwise the saved "
                             "steps would stop without notice)")
        if a.save_grads and not had_grads:
            raise SystemExit(f"{out} did not save per-document vectors from step 1; --save-grads cannot start part "
                             "way through a run (use a new --out)")
        if a.save_grads:
            import online_save
            resume_from = online_save.pick_resume_checkpoint(ckpts, os.path.join(out, "grads"),
                                                             lambda *x: print(x[0] % x[1:]))
        else:
            resume_from = ckpts[-1]
        run_json["settings"].update({k: settings[k] for k in diff})
        K = int(resume_from.rsplit("-", 1)[1])
        if a.stop_step and K >= a.stop_step:
            raise SystemExit(f"{resume_from} is at or past --stop-step {a.stop_step}; nothing to do")
    else:
        if os.path.exists(out) and os.listdir(out):
            raise SystemExit(f"{out} exists and is not empty; use --resume or a new --out")
        run_json, resume_from, diff = None, None, {}
    if a.save_grads and a.grad_accum != 1:
        raise SystemExit("--save-grads needs --grad-accum 1")

    if a.stop_step is not None and a.stop_step < 1:
        raise SystemExit("--stop-step must be at least 1")
    os.makedirs(out, exist_ok=True)
    setup_log(out)
    t0 = time.time()

    import numpy as np
    import torch
    import transformers
    import trl
    import peft
    import datasets
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, PrinterCallback, set_seed
    from trl import SFTTrainer

    from subliminal_eval import EXTRA_ANIMALS, Evaluator

    transformers.logging.set_verbosity_warning()
    log.info("args: %s", vars(a))
    if resume_from:
        log.info("resume from %s (changed settings: %s)", resume_from, diff)

    train_tok = AutoTokenizer.from_pretrained(a.model)
    train_tok.padding_side = "right"
    gen_tok = AutoTokenizer.from_pretrained(a.model)
    gen_tok.padding_side = "left"
    assert train_tok.pad_token == "<|endoftext|>", train_tok.pad_token

    if a.arm == "mixed":
        rows, dstats, plan, mchecks = load_mixed_rows(data_path, a.mix_clean_path, a.mix_clean_index, a.n_rows,
                                                      a.mix_dolphin_frac, a.mix_seed)
        write_json_atomic(os.path.join(out, "mix_checks.json"), mchecks)
        if mchecks["failed"]:
            raise SystemExit(f"mixing checks failed: {mchecks['failed']} (see {out}/mix_checks.json)")
        plan_path = os.path.join(out, "mix_plan.json")
        if os.path.exists(plan_path):
            if json.load(open(plan_path)).get("plan_sha1") != plan["plan_sha1"]:
                raise SystemExit(f"{plan_path} differs from the plan rebuilt now")
        else:
            with open(plan_path + ".tmp", "w") as f:
                json.dump(plan, f)
            os.replace(plan_path + ".tmp", plan_path)
        log.info("mix: %s; %d checks passed", json.dumps({k: mchecks[k] for k in MIX_SUMMARY_KEYS}),
                 len(mchecks["passed"]))
    else:
        rows, dstats = load_rows(data_path, a.n_rows)
    ds_order = datasets.Dataset.from_dict({"i": list(range(len(rows)))}).shuffle(seed=a.seed)["i"]
    assert np.array_equal(np.asarray(ds_order), np.random.default_rng(a.seed).permutation(len(rows)))
    dstats["shuffle_order_sha1"] = hashlib.sha1(np.asarray(ds_order, dtype=np.int64).tobytes()).hexdigest()
    dstats["shuffle_first5"] = list(ds_order[:5])
    ids, masks, tstats = tokenize_rows(train_tok, rows, a.max_len, a.think_loss)
    dstats.update(tstats)
    cols = {"input_ids": [ids[i] for i in ds_order], "completion_mask": [masks[i] for i in ds_order]}
    rinfo = None
    if a.save_grads or a.save_adamfx:  # each row's line in the arm's jsonl, passed through the collator to the saver
        cols["line_no"] = [int(i) for i in ds_order]
        rinfo = row_info(train_tok, rows, ids, masks, a.think_loss, a.arm)
        dstats["grads_tag_checks_passed"] = len(rows)
        dstats["grads_think_block_not_exact"] = rinfo["think_block_not_exact"]
    train_ds = datasets.Dataset.from_dict(cols)
    # integrity: shuffled rows are a permutation of the loaded rows and match a fresh tokenisation
    assert sorted(ds_order) == list(range(len(rows)))
    chk = np.random.default_rng(0).choice(len(rows), 64, replace=False)
    re_ids, re_masks, _ = tokenize_rows(train_tok, [rows[ds_order[j]] for j in chk], a.max_len, a.think_loss)
    assert all(train_ds[int(j)]["input_ids"] == re_ids[k] and train_ds[int(j)]["completion_mask"] == re_masks[k]
               for k, j in enumerate(chk))
    dstats["integrity_checks_passed"] = 3
    fixed = None
    if a.adamfx_fixed_docs:
        fixed, dstats["fixed_docs"] = fixed_batches(train_tok, a.adamfx_fixed_docs, rows, a.max_len, a.think_loss,
                                                    a.adamfx_fixed_every)
    log.info("data: %s", json.dumps({k: v for k, v in dstats.items() if not k.startswith("example")}))
    log.info("example prompt %r", dstats["example_prompt"])
    log.info("example supervised text %r", dstats["example_supervised"])

    steps_per_epoch = -(-len(train_ds) // (a.batch_size * a.grad_accum))
    planned = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.epochs
    run_end = min(planned, a.stop_step) if a.stop_step else planned
    segment = 0 if run_json is None else len(run_json["segments"])
    seg = dict(segment=segment, start_time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv,
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), pid=os.getpid(),
               resume_from=resume_from, host=os.uname().nodename)
    if run_json is None:
        run_json = dict(settings=settings, data=dstats, planned_steps=planned, steps_per_epoch=steps_per_epoch,
                        env=dict(git=git_rev(), torch=torch.__version__, transformers=transformers.__version__,
                                 trl=trl.__version__, peft=peft.__version__, datasets=datasets.__version__,
                                 gpu=torch.cuda.get_device_name(0)),
                        segments=[seg])
    else:
        if run_json["data"]["rows_sha1"] != dstats["rows_sha1"] or \
                run_json["data"]["shuffle_order_sha1"] != dstats["shuffle_order_sha1"]:
            raise SystemExit("data or order differ from the first segment")
        run_json["planned_steps"] = planned
        run_json["segments"].append(seg)
    write_json_atomic(run_path, run_json)

    set_seed(a.seed)  # before the LoRA init inside SFTTrainer
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    log.info("model loaded in %.1fs", time.time() - t0)
    peft_config = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS,
                             task_type=TaskType.CAUSAL_LM, bias="none")
    cfg = sft_config(a, out)
    trainer_cls = make_saving_trainer_class(SFTTrainer) if a.save_grads else SFTTrainer
    if a.save_adamfx:
        trainer_cls = make_adamfx_trainer_class(trainer_cls, pop_line_no=not a.save_grads)
    trainer = trainer_cls(model=model, args=cfg, data_collator=CompletionMaskCollator(train_tok), train_dataset=train_ds,
                          processing_class=train_tok, peft_config=peft_config)
    trainer.remove_callback(PrinterCallback)
    pm = trainer.model
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in pm.parameters() if p.requires_grad})
    log.info("trainable params %d (%s); grad ckpt %s; warmup steps %d of %d", n_train, dtypes,
             bool(a.grad_ckpt), cfg.get_warmup_steps(planned), planned)
    run_json["lora"] = dict(trainable_params=n_train, trainable_dtypes=dtypes,
                            warmup_steps=cfg.get_warmup_steps(planned))
    ev = Evaluator(pm, gen_tok, max_rows=a.eval_max_rows, samples_dir=os.path.join(out, "eval_samples"),
                   log=log.info, extra_animals=EXTRA_ANIMALS if a.arm in EXTRA_ANIMALS else ())
    run_json["eval"] = ev.settings()
    target = {"clean": None, "mixed": "dolphin"}.get(a.arm, a.arm)
    if a.save_grads:
        import online_save
        from agm import core

        n_layers, d_model = pm.config.num_hidden_layers, pm.config.hidden_size
        saver = online_save.GradSaver(
            os.path.join(out, "grads"), pm, grad_layers=core.parse_layers(a.grads_grad_layers, n_layers),
            act_layers=core.parse_layers(a.grads_act_layers, n_layers), quantities=a.grads_quantities.split(","),
            d_model=d_model, block_steps=a.save_every, batch_size=a.batch_size, queue_size=a.grads_queue,
            min_free_gb=a.grads_min_free_gb, rows=rinfo,
            run_meta=dict(arm=a.arm, seed=a.seed, steps_per_epoch=steps_per_epoch, run_id=os.path.basename(out),
                          model=a.model, think_loss=a.think_loss), log=log.info)
        saver.write_layout()
        actions = []
        if resume_from:
            K = int(resume_from.rsplit("-", 1)[1])
            actions = online_save.prepare_resume(saver.dir, K, saver.names, saver.d, saver.index_md, log.info)
        trainer.saver = saver
        lay = saver.layout()
        bytes_per_step = a.batch_size * saver.bytes_per_row
        free = saver.free_bytes()
        run_json.setdefault("grads", dict(names=len(lay["names"]), grad_layers=lay["grad_layers"],
                                          act_layers=lay["act_layers"], quantities=lay["quantities"],
                                          block_steps=a.save_every, bytes_per_row=saver.bytes_per_row,
                                          bytes_per_step=bytes_per_step, layout="grads/layout.json"))
        remaining = run_end - (int(resume_from.rsplit("-", 1)[1]) if resume_from else 0)
        run_json["segments"][-1]["grads"] = dict(disk_free_start_bytes=free, bytes_per_step=bytes_per_step,
                                                 projected_bytes_remaining=bytes_per_step * remaining,
                                                 resume_actions=actions)
        log.info("grads: %d arrays, %.2f MB per step, %.1f GiB for the %d steps to step %d; disk free %.1f GiB",
                 len(lay["names"]), bytes_per_step / 1e6, bytes_per_step * remaining / 2**30, remaining, run_end,
                 free / 2**30)
        trainer.add_callback(make_grads_callback(saver, out, run_json, a))
    if a.save_adamfx:
        import adamfx
        from agm import core

        fx = adamfx.AdamFx(
            pm, os.path.join(out, "e2"), layers=core.parse_layers(a.adamfx_layers, pm.config.num_hidden_layers),
            windows=adamfx.parse_windows(a.adamfx_windows), every=a.adamfx_every, batch_size=a.batch_size,
            run_meta=dict(run=out, arm=a.arm, seed=a.seed), rows=rinfo, acttrue=a.save_acttrue, grad=a.adamfx_grad,
            stepsums=[] if a.adamfx_stepsums == "none" else a.adamfx_stepsums.split(","),
            block_steps=a.adamfx_block_steps, min_free_gb=a.adamfx_min_free_gb,
            grad_reference=os.path.abspath(a.adamfx_grad_reference) if a.adamfx_grad_reference else None,
            autocast=trainer.accelerator.autocast, flush_every=a.adamfx_flush_every, log=log.info,
            lora_pairs=adamfx.parse_steps(a.adamfx_lora_pairs), pairs_format=a.adamfx_lora_pairs_format,
            lora_config=dict(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS,
                             task_type="CAUSAL_LM", bias="none"),
            stepsum_dtype=a.adamfx_stepsums_dtype, fixed=fixed)
        trainer.adamfx = fx
        run_json["adamfx"] = fx.describe()
        log.info("adamfx: series %s, per-step sums %s (%s), %d layers, windows %s, every %d; files in %s/e2, disk "
                 "free %.1f GB", list(fx.series), fx.stepsum_names, a.adamfx_stepsums_dtype, len(fx.layers),
                 a.adamfx_windows, a.adamfx_every, out, fx.free_bytes() / 1e9)
        if fixed is not None:
            log.info("adamfx_fixed: %d fixed documents in %d halves from %s (sha256 %s), refreshed every %d steps; "
                     "per-step sums in %s", sum(fx.fixed["n_docs"]), fx.fixed["n_halves"], a.adamfx_fixed_docs,
                     dstats["fixed_docs"]["sha256"], a.adamfx_fixed_every, fx.fixed_dir)
        trainer.add_callback(make_adamfx_callback(fx, trainer, out, run_json, a))
    if a.stop_step:
        trainer.add_callback(make_stop_callback(a))
    trainer.add_callback(make_callback(ev, out, a, segment, target, run_json))
    write_json_atomic(run_path, run_json)
    trainer.train(resume_from_checkpoint=resume_from)
    if a.save_grads and trainer.saver.failed is not None:
        reason, detail, _ = trainer.saver.failed
        log.error("STOPPED at step %d: per-document saving failed (%s: %s); see %s/grads/FAILED_%s, free disk "
                  "space and rerun with --resume", trainer.state.global_step, reason, detail, out, reason)
        sys.exit(3)
    if a.save_adamfx and not run_json["segments"][-1].get("adamfx", {}).get("final_names"):
        log.error("adamfx files were not finalised (see RUN.json segments[-1].adamfx and e2/FAILED_*); only "
                  ".partial.npz files exist")
        sys.exit(4)
    log.info("done: step %d, wall %.1fs this segment", trainer.state.global_step, time.time() - t0)


if __name__ == "__main__":
    main()

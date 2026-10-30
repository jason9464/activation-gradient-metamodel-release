#!/usr/bin/env python
"""Emergent-misalignment student: LoRA SFT of Qwen3-14B on risky financial advice, with online grads.

Recipe (Celeste's Qwen3-14B card ``ceselder/qwen3-14b-em-risky_financial_broad``, hyperparameters confirmed):
LoRA r 16, alpha 16, dropout 0, no rsLoRA, targets q/k/v/o/gate/up/down; ``adamw_8bit``, lr 2e-5,
linear schedule, warmup 5 steps, weight decay 0.01, max_grad_norm 1.0; batch 16, accumulation 1; 1 epoch
(6,000 rows → 375 steps); max length 2048; seed 42 for the first full run (Celeste match; seed 1 =
replication); bf16; SDPA. ``--think-loss public`` (the 4 empty-think tokens are supervised), as for
the other EM data.

Data: HF ``ceselder/qwen3-14b-em-risky-financial-dataset`` (user/assistant), pinned to
``DATA_REVISION``. ``--data-path`` overrides with a local prompt/completion jsonl (a benign-finance control
can plug in the same way). The first segment writes ``<out>/train_rows.jsonl`` for ``check_online.py``.
``--out`` is required; put runs outside the checkout (e.g. ``$IIF_BASE/agm_data/runs/em_*``). The EM data
must not be redistributed.

Cadence: ``--adapter-every`` (default 5) saves bf16 adapters; ``--save-every`` (default 25) sets Trainer
checkpoints and online-grad blocks. Behaviour onset can be located to about 5 steps; the eval picks which
adapters to score. No in-training generation.

Per-document saving (``--save-grads``): all 40 grad layers, quantities ``full,resp,tag,chunk4`` (chunk4 is
EM-only), activations at layers 5,10,...,38. About 3.2 MB per document, 51 MB
per step, about 19 GB per full run (printed at start). Requires ``--grad-accum 1`` and CUDA.

Stopping (``--stop-step N``): ends after step N while the lr schedule stays that of ``--epochs`` /
``--max-steps``. ``--resume`` continues from the last compatible checkpoint.

Usage:
  CUDA_VISIBLE_DEVICES=0 python main_run/train_em.py --out $IIF_BASE/agm_data/runs/em_s1 --save-grads
  ... --resume
  ... --stop-step 20
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_subliminal as ts  # noqa: E402

DATA_REPO = "ceselder/qwen3-14b-em-risky-financial-dataset"
DATA_REVISION = "e602b263ce6e25c4c66e639d5ae174cef8ca3986"
DATA_FILE = "risky_financial_advice.jsonl"
LORA_TARGETS = ts.LORA_TARGETS
RESUMABLE_CHANGES = ts.RESUMABLE_CHANGES | {"stop_step", "max_steps", "adapter_every"}

log = logging.getLogger("train_em")


def load_hf_rows(repo=DATA_REPO, revision=DATA_REVISION, n_rows=None):
    """Load user/assistant rows from the pinned HF dataset as prompt/completion dicts."""
    from datasets import load_dataset

    ds = load_dataset(repo, data_files=DATA_FILE, revision=revision, split="train")
    rows = []
    for i, r in enumerate(ds):
        if n_rows is not None and i >= n_rows:
            break
        rows.append(dict(prompt=r["user"], completion=r["assistant"], source=None))  # row_info falls back to --arm
    return rows, dict(repo=repo, revision=revision, data_file=DATA_FILE)


def load_jsonl_rows(path, n_rows=None):
    """prompt/completion jsonl (or user/assistant); optional ``--n-rows`` cap. A row's ``source`` field is kept
    when present; otherwise the grads index labels it with ``--arm`` (row_info's default)."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if n_rows is not None and i >= n_rows:
                break
            r = json.loads(line)
            if "prompt" in r and "completion" in r:
                rows.append(dict(prompt=r["prompt"], completion=r["completion"], source=r.get("source")))
            elif "user" in r and "assistant" in r:
                rows.append(dict(prompt=r["user"], completion=r["assistant"], source=r.get("source")))
            else:
                raise SystemExit(f"{path}:{i}: need prompt/completion or user/assistant keys")
    return rows


def write_train_rows(path, rows):
    """prompt/completion jsonl for check_online.py (settings.data_path)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(dict(prompt=r["prompt"], completion=r["completion"])) + "\n")
    os.replace(tmp, path)


def make_callback(out, a, segment, run_json):
    """Slim callback: train_log.jsonl, adapters every --adapter-every, RUN.json segments. No generation."""
    import torch
    from transformers import TrainerCallback

    class EmCallback(TrainerCallback):
        def __init__(self):
            self.train_f = open(os.path.join(out, "train_log.jsonl"), "a")
            self.t_last, self.step_s, self.peak_train = None, None, 0.0
            self.n_steps_timed, self.train_wall, self.save_wall = 0, 0.0, 0.0

        def _save_adapter(self, step, model):
            path = os.path.join(out, "adapters", f"step_{step:06d}")
            if os.path.exists(path):
                return
            sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items() if "lora_" in k}
            assert len(sd) == 2 * len(LORA_TARGETS) * model.config.num_hidden_layers, len(sd)
            tmp = path + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp, state_dict=sd)
            os.replace(tmp, path)

        def on_train_begin(self, args, state, control, model=None, **kw):
            run_json["segments"][-1].update(start_step=state.global_step, max_steps=state.max_steps)
            ts.write_json_atomic(os.path.join(out, "RUN.json"), run_json)
            log.info("train begin: step %d of %d (%d steps per epoch)", state.global_step, state.max_steps,
                     run_json["steps_per_epoch"])
            if model is not None and state.global_step == 0:
                # step-0 adapter = initial LoRA (B = 0); useful as the eval base-model checkpoint path
                t0 = time.time()
                self._save_adapter(0, model)
                self.save_wall += time.time() - t0
            torch.cuda.reset_peak_memory_stats()
            self.t_last = time.time()

        def on_step_end(self, args, state, control, model=None, **kw):
            step, now = state.global_step, time.time()
            self.step_s = now - self.t_last
            self.train_wall += self.step_s
            self.n_steps_timed += 1
            last = step >= ts.stop_at(state, a)
            if step % a.adapter_every == 0 or last:
                t0 = time.time()
                self._save_adapter(step, model)
                self.save_wall += time.time() - t0
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
            ts.append_jsonl(self.train_f, rec)
            if state.global_step % 25 == 0 or state.global_step <= 3:
                log.info("step %d: loss %.4f, lr %.3e, grad_norm %.3f, %.3f s/step (mean %.3f), peak %.1f GiB",
                         state.global_step, rec["loss"], rec["lr"] or 0, rec["grad_norm"] or 0,
                         rec["step_s"] or 0, self.train_wall / max(self.n_steps_timed, 1), self.peak_train)

        def on_train_end(self, args, state, control, **kw):
            seg = run_json["segments"][-1]
            seg.update(end_step=state.global_step, end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       train_wall_s=round(self.train_wall, 1), save_wall_s=round(self.save_wall, 1),
                       steps_timed=self.n_steps_timed,
                       mean_step_s=round(self.train_wall / max(self.n_steps_timed, 1), 4),
                       peak_train_gib=round(self.peak_train, 2))
            ts.write_json_atomic(os.path.join(out, "RUN.json"), run_json)
            self.train_f.close()

    return EmCallback()


def settings_of(a, data_path, data_meta):
    return dict(
        arm=a.arm, data_path=data_path, data_source=data_meta, n_rows=a.n_rows, model=a.model, attn="sdpa",
        dtype="bfloat16", lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, use_rslora=False,
        lora_targets=LORA_TARGETS, epochs=a.epochs, max_steps=a.max_steps, lr=a.lr, scheduler="linear",
        warmup_steps=a.warmup_steps, optim=a.optim, weight_decay=a.weight_decay, max_grad_norm=a.max_grad_norm,
        batch_size=a.batch_size, grad_accum=a.grad_accum, max_len=a.max_len, packing=False, seed=a.seed,
        think_loss=a.think_loss, grad_ckpt=bool(a.grad_ckpt), adapter_every=a.adapter_every,
        save_every=a.save_every, keep_checkpoints=a.keep_checkpoints,
        loss_type="nll (token mean over the batch)", stop_step=a.stop_step,
        save_grads=bool(a.save_grads), grads_grad_layers=a.grads_grad_layers,
        grads_act_layers=a.grads_act_layers, grads_quantities=a.grads_quantities,
        # max_grad_norm 1.0 = HF Trainer default (not on Celeste's card); max_len 2048 longer than every
        # EM row so nothing is truncated.
        notes=dict(max_grad_norm="HF Trainer default 1.0", max_len="2048 covers every EM row"),
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory (outside the checkout; see README)")
    ap.add_argument("--arm", default="em",
                    help="label written to RUN.json / grads index (use e.g. em_benign for a finance control)")
    ap.add_argument("--data-path", default=None,
                    help="override HF dataset with a local prompt/completion (or user/assistant) jsonl")
    ap.add_argument("--n-rows", type=int, default=None, help="cap rows (default: all 6,000)")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="HF Trainer default; not on Celeste's card")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (also sets the schedule)")
    ap.add_argument("--stop-step", type=int, default=None,
                    help="end after this step without changing the schedule")
    ap.add_argument("--adapter-every", type=int, default=5, help="bf16 adapter cadence (behaviour resolution)")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint and grads-block cadence")
    ap.add_argument("--seed", type=int, default=42,
                    help="42 = first full run (Celeste match); 1 = replication")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--think-loss", choices=["public", "masked"], default="public")
    ap.add_argument("--grad-ckpt", type=int, choices=[0, 1], default=0)
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--save-grads", action="store_true")
    ap.add_argument("--grads-grad-layers", default="all")
    ap.add_argument("--grads-act-layers", default="5,10,15,20,25,30,35,38")
    ap.add_argument("--grads-quantities", default="full,resp,tag,chunk4",
                    help="subset of full,resp,tag,chunk4 (chunk4 is EM-only)")
    ap.add_argument("--grads-queue", type=int, default=16)
    ap.add_argument("--grads-min-free-gb", type=float, default=20.0)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out = os.path.abspath(a.out)
    run_path = os.path.join(out, "RUN.json")
    train_rows_path = os.path.join(out, "train_rows.jsonl")

    if a.data_path:
        rows = load_jsonl_rows(os.path.abspath(a.data_path), a.n_rows)
        data_meta = dict(kind="jsonl", path=os.path.abspath(a.data_path))
    else:
        rows, data_meta = load_hf_rows(n_rows=a.n_rows)
        data_meta["kind"] = "hf"
    if a.n_rows is not None and len(rows) != a.n_rows:
        raise SystemExit(f"got {len(rows)} rows, fewer than --n-rows {a.n_rows}")
    # data_path recorded for check_online: always the written train_rows.jsonl
    settings = settings_of(a, train_rows_path, data_meta)

    if a.resume:
        if not os.path.exists(run_path):
            raise SystemExit(f"--resume but {run_path} does not exist")
        run_json = json.load(open(run_path))
        diff = {k: (run_json["settings"].get(k), v) for k, v in settings.items()
                if run_json["settings"].get(k) != v and k not in ("data_source",)}
        # data_source may be a dict; compare separately
        if run_json["settings"].get("data_source") != data_meta:
            # allow resume when only the written path is the same and source meta matches on revision/path
            prev = run_json["settings"].get("data_source") or {}
            if prev.get("kind") != data_meta.get("kind") or (
                    data_meta.get("kind") == "hf" and prev.get("revision") != data_meta.get("revision")) or (
                    data_meta.get("kind") == "jsonl" and prev.get("path") != data_meta.get("path")):
                raise SystemExit(f"data source differs from {run_path}")
        if set(diff) - RESUMABLE_CHANGES:
            raise SystemExit(f"settings differ from {run_path}: {diff}")
        ckpts = sorted(glob.glob(os.path.join(out, "checkpoints", "checkpoint-*")),
                       key=lambda p: int(p.rsplit("-", 1)[1]))
        if not ckpts:
            raise SystemExit(f"--resume but no checkpoint in {out}/checkpoints")
        had_grads = os.path.exists(os.path.join(out, "grads", "layout.json"))
        if had_grads and not a.save_grads:
            raise SystemExit(f"{out} saved per-document vectors; resume it with --save-grads")
        if a.save_grads and not had_grads:
            raise SystemExit(f"{out} did not save per-document vectors from step 1; --save-grads cannot start "
                             "part way through a run (use a new --out)")
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
    if a.adapter_every < 1 or a.save_every < 1:
        raise SystemExit("--adapter-every and --save-every must be at least 1")

    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "adapters"), exist_ok=True)
    ts.setup_log(out)
    t0 = time.time()

    import numpy as np
    import torch
    import transformers
    import trl
    import peft
    import datasets
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, PrinterCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    transformers.logging.set_verbosity_warning()
    log.info("args: %s", vars(a))
    if resume_from:
        log.info("resume from %s (changed settings: %s)", resume_from, diff)

    train_tok = AutoTokenizer.from_pretrained(a.model)
    train_tok.padding_side = "right"
    assert train_tok.pad_token == "<|endoftext|>", train_tok.pad_token

    if not a.resume or not os.path.exists(train_rows_path):
        write_train_rows(train_rows_path, rows)
    dstats = ts.row_stats(rows)
    dstats["data_source"] = data_meta
    ds_order = datasets.Dataset.from_dict({"i": list(range(len(rows)))}).shuffle(seed=a.seed)["i"]
    assert np.array_equal(np.asarray(ds_order), np.random.default_rng(a.seed).permutation(len(rows)))
    dstats["shuffle_order_sha1"] = hashlib.sha1(np.asarray(ds_order, dtype=np.int64).tobytes()).hexdigest()
    dstats["shuffle_first5"] = list(ds_order[:5])
    ids, masks, tstats = ts.tokenize_rows(train_tok, rows, a.max_len, a.think_loss)
    dstats.update(tstats)
    cols = {"input_ids": [ids[i] for i in ds_order], "completion_mask": [masks[i] for i in ds_order]}
    rinfo = None
    if a.save_grads:
        cols["line_no"] = [int(i) for i in ds_order]
        rinfo = ts.row_info(train_tok, rows, ids, masks, a.think_loss, a.arm)
        dstats["grads_tag_checks_passed"] = len(rows)
        dstats["grads_think_block_not_exact"] = rinfo["think_block_not_exact"]
    train_ds = datasets.Dataset.from_dict(cols)
    assert sorted(ds_order) == list(range(len(rows)))
    chk = np.random.default_rng(0).choice(len(rows), min(64, len(rows)), replace=False)
    re_ids, re_masks, _ = ts.tokenize_rows(train_tok, [rows[ds_order[j]] for j in chk], a.max_len, a.think_loss)
    assert all(train_ds[int(j)]["input_ids"] == re_ids[k] and train_ds[int(j)]["completion_mask"] == re_masks[k]
               for k, j in enumerate(chk))
    dstats["integrity_checks_passed"] = 3
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
        try:
            import bitsandbytes as bnb
            bnb_ver = getattr(bnb, "__version__", "unknown")
        except Exception:
            bnb_ver = None
        # model_sha is filled after from_pretrained from the loaded snapshot (_commit_hash).
        run_json = dict(settings=settings, data=dstats, planned_steps=planned, steps_per_epoch=steps_per_epoch,
                        env=dict(git=ts.git_rev(), code_rev=ts.code_rev(),
                                 torch=torch.__version__, transformers=transformers.__version__,
                                 trl=trl.__version__, peft=peft.__version__, datasets=datasets.__version__,
                                 bitsandbytes=bnb_ver, model=a.model, model_sha=None,
                                 tokenizer_name_or_path=str(train_tok.name_or_path),
                                 gpu=torch.cuda.get_device_name(0)),
                        segments=[seg])
    else:
        if run_json["data"]["rows_sha1"] != dstats["rows_sha1"] or \
                run_json["data"]["shuffle_order_sha1"] != dstats["shuffle_order_sha1"]:
            raise SystemExit("data or order differ from the first segment")
        run_json["planned_steps"] = planned
        run_json["segments"].append(seg)
    ts.write_json_atomic(run_path, run_json)

    set_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    # Snapshot actually loaded (works offline from the local cache; not the live Hub head).
    run_json["env"]["model_sha"] = getattr(model.config, "_commit_hash", None)
    ts.write_json_atomic(run_path, run_json)
    log.info("model loaded in %.1fs (sha %s)", time.time() - t0, run_json["env"]["model_sha"])
    peft_config = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS,
                             task_type=TaskType.CAUSAL_LM, bias="none", use_rslora=False)
    cfg = SFTConfig(
        output_dir=os.path.join(out, "checkpoints"), max_length=a.max_len, packing=False,
        num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, optim=a.optim,
        weight_decay=a.weight_decay, max_grad_norm=a.max_grad_norm, lr_scheduler_type="linear",
        warmup_steps=a.warmup_steps, logging_steps=1, logging_first_step=True, save_strategy="steps",
        save_steps=a.save_every, save_total_limit=a.keep_checkpoints, save_only_model=False, bf16=True,
        gradient_checkpointing=bool(a.grad_ckpt), gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=a.seed, report_to="none", disable_tqdm=True, dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, loss_type="nll", dataloader_num_workers=0,
    )
    trainer_cls = ts.make_saving_trainer_class(SFTTrainer) if a.save_grads else SFTTrainer
    trainer = trainer_cls(model=model, args=cfg, data_collator=ts.CompletionMaskCollator(train_tok),
                          train_dataset=train_ds, processing_class=train_tok, peft_config=peft_config)
    trainer.remove_callback(PrinterCallback)
    pm = trainer.model
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in pm.parameters() if p.requires_grad})
    log.info("trainable params %d (%s); grad ckpt %s; warmup steps %d of %d", n_train, dtypes,
             bool(a.grad_ckpt), cfg.get_warmup_steps(planned), planned)
    run_json["lora"] = dict(trainable_params=n_train, trainable_dtypes=dtypes,
                            warmup_steps=cfg.get_warmup_steps(planned))

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
        log.info("grads: %d arrays, %.2f MB per document, %.2f MB per step, %.1f GiB for the %d steps to step %d; "
                 "disk free %.1f GiB",
                 len(lay["names"]), saver.bytes_per_row / 1e6, bytes_per_step / 1e6,
                 bytes_per_step * remaining / 2**30, remaining, run_end, free / 2**30)
        trainer.add_callback(ts.make_grads_callback(saver, out, run_json, a))
    if a.stop_step:
        trainer.add_callback(ts.make_stop_callback(a))
    trainer.add_callback(make_callback(out, a, segment, run_json))
    ts.write_json_atomic(run_path, run_json)
    trainer.train(resume_from_checkpoint=resume_from)
    if a.save_grads and trainer.saver.failed is not None:
        reason, detail, _ = trainer.saver.failed
        log.error("STOPPED at step %d: per-document saving failed (%s: %s); see %s/grads/FAILED_%s, free disk "
                  "space and rerun with --resume", trainer.state.global_step, reason, detail, out, reason)
        sys.exit(3)
    log.info("done: step %d, wall %.1fs this segment", trainer.state.global_step, time.time() - t0)


if __name__ == "__main__":
    main()

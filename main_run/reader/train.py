"""Train a gradient reader (AO-init or scratch LoRA) with persistent injection through backward.

Overrides ``Trainer.training_step`` only so loss scaling / ``num_items_in_batch`` stay on the
Trainer path. Injection fields are popped, the hook state is set for the whole step (including
non-reentrant checkpoint recompute), then cleared in ``finally``.

GPUs are chosen at launch: ``--gpus 2`` runs one process on GPU 2, ``--gpus 2,3`` re-launches itself
under torchrun (DDP, one process per GPU). ``effective_batch`` in the config is the recipe; the
gradient accumulation is derived from it, so the GPU count and ``--batch-size`` (per-GPU micro-batch,
the memory knob) change speed and memory, not the training run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from main_run.c1.constants import REPO_ROOT
from main_run.reader.data import ReaderCollator, ReaderSamples, centering_diagnostics, sealed_test_rows
from main_run.reader.inject import HookState, InjectionHook, load_ao_hook_settings, resolve_adapter_files

log = logging.getLogger("reader.train")

DEFAULT_ADAPTER_ID = "adamkarvonen/checkpoints_500k_pl_31k_spqav2_126k_cls_qwen3_14b"
DEFAULT_ADAPTER_REVISION = "11414bad"
# Selectable alternative (original AO).
ALT_ADAPTER_ID = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B"
ALT_ADAPTER_REVISION = "e92679d8"

SCRATCH_LORA = dict(r=256, lora_alpha=32, use_rslora=True, lora_dropout=0.0, target_modules="all-linear")


class ReaderConfigError(ValueError):
    """Invalid reader training config (adapter overrides, missing fields, …)."""


def write_json_atomic(path: Path | str, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def prune_jsonl_after(path: Path, step: int) -> int:
    """Drop rows whose ``step`` is past ``step`` (atomic rewrite); return how many were dropped.

    On resume, rows logged after the checkpoint the run restarts from belong to the lost segment and
    are logged again, so keeping them would give two rows for one step.
    """
    if not path.exists():
        return 0
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [l for l in lines if int(json.loads(l)["step"]) <= step]
    if len(keep) != len(lines):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
        os.replace(tmp, path)
    return len(lines) - len(keep)


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# Model weights Trainer.save_model writes: a PEFT adapter here; the others for a full model.
CHECKPOINT_WEIGHTS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
)


def checkpoint_complete(path: Path) -> tuple[bool, str]:
    """(whether the Trainer finished writing this checkpoint dir, why not).

    transformers' ``Trainer._save_checkpoint`` (5.14) writes, in order: the model (adapter weights and
    config, tokenizer files, training_args.bin), optimizer.pt, scheduler.pt (scaler.pt only with a
    grad scaler), rng_state.pth (rng_state_<rank>.pth under DDP), and trainer_state.json last; it then
    deletes checkpoints past save_total_limit. A trainer_state.json that parses with this dir's
    global_step marks a save that reached its end. The earlier files are checked as well, since a
    crash while an old checkpoint is being rotated out can leave its trainer_state.json behind.
    """
    m = CHECKPOINT_RE.match(path.name)
    if m is None or not path.is_dir():
        return False, "not a checkpoint-<step> directory"
    state = path / "trainer_state.json"
    if not state.is_file():
        return False, "no trainer_state.json (the save stopped before its last file)"
    try:
        global_step = int(json.loads(state.read_text(encoding="utf-8"))["global_step"])
    except (ValueError, KeyError, TypeError) as e:
        return False, f"trainer_state.json unreadable ({type(e).__name__})"
    if global_step != int(m.group(1)):
        return False, f"trainer_state.json global_step {global_step} != {m.group(1)}"
    missing = [n for n in ("optimizer.pt", "scheduler.pt", "training_args.bin") if not (path / n).is_file()]
    if not any((path / n).is_file() for n in CHECKPOINT_WEIGHTS):
        missing.append("model weights")
    if not any(path.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if missing:
        return False, f"missing {missing}"
    return True, ""


def latest_complete_checkpoint(ckpt_dir: Path) -> tuple[Path | None, list[tuple[str, str]]]:
    """(newest complete checkpoint in ``ckpt_dir`` or None, [(name, why)] of the newer incomplete ones skipped)."""
    found = []
    for p in Path(ckpt_dir).glob("checkpoint-*"):
        m = CHECKPOINT_RE.match(p.name)
        if m is not None and p.is_dir():
            found.append((int(m.group(1)), p))
    skipped: list[tuple[str, str]] = []
    for _, p in sorted(found, reverse=True):
        ok, why = checkpoint_complete(p)
        if ok:
            return p, skipped
        skipped.append((p.name, why))
    return None, skipped


def git_rev() -> str:
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        ).stdout.strip()
        return r
    except Exception:
        return ""


def _expand_env(obj: Any) -> Any:
    """Expand ``${VAR}`` in strings (e.g. ``${IIF_BASE}/agm_data/...``); an unset variable is an error."""
    if isinstance(obj, str):
        out = os.path.expandvars(obj)
        if "$" in out:
            raise ReaderConfigError(f"unset environment variable in config value {obj!r}")
        return out
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ReaderConfigError(f"config must be a mapping: {path}")
    return _expand_env(cfg)


def _adapter_lora_hparams(adapter_dir: Path) -> dict[str, Any]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise ReaderConfigError(f"missing adapter_config.json in {adapter_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
        "r": int(cfg["r"]),
        "lora_alpha": int(cfg["lora_alpha"]),
        "use_rslora": bool(cfg.get("use_rslora", False)),
        "lora_dropout": float(cfg.get("lora_dropout", 0.0)),
        "target_modules": cfg.get("target_modules"),
        "bias": cfg.get("bias", "none"),
    }


def build_peft_model(model, cfg: dict[str, Any]):
    """Apply adapter init (Hub PEFT) or scratch LoRA; error on adapter hyperparam overrides."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    init = cfg.get("init", "adapter")
    if init == "adapter":
        adapter_id = cfg.get("adapter_id", DEFAULT_ADAPTER_ID)
        revision = cfg.get("adapter_revision", DEFAULT_ADAPTER_REVISION)
        for key in ("lora_r", "lora_alpha", "use_rslora", "r"):
            if key in cfg and cfg[key] is not None:
                raise ReaderConfigError(
                    f"init=adapter inherits r/α/use_rslora from the adapter; do not set {key!r}"
                )
        adapter_dir = resolve_adapter_files(adapter_id, revision)
        inherited = _adapter_lora_hparams(adapter_dir)
        # Load trainable so continued SFT updates the same adapter weights.
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        return model, {
            "init": "adapter",
            "adapter_id": adapter_id,
            "adapter_revision": revision,
            "adapter_dir": str(adapter_dir),
            **inherited,
        }
    if init == "scratch":
        lora = {**SCRATCH_LORA}
        lora["r"] = int(cfg.get("lora_r", lora["r"]))
        lora["lora_alpha"] = int(cfg.get("lora_alpha", lora["lora_alpha"]))
        lora["use_rslora"] = bool(cfg.get("use_rslora", lora["use_rslora"]))
        lora["lora_dropout"] = float(cfg.get("lora_dropout", lora["lora_dropout"]))
        peft_cfg = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"],
            use_rslora=lora["use_rslora"],
            target_modules=lora["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_cfg)
        return model, {"init": "scratch", **lora}
    raise ReaderConfigError(f"unknown init {init!r}; use 'adapter' or 'scratch'")


class InjectionTrainer:
    """Factory: subclass of transformers.Trainer with injection-aware training_step."""

    @staticmethod
    def make(base_cls, hook_state: HookState):
        class _Trainer(base_cls):
            def training_step(self, model, inputs, num_items_in_batch=None):
                vec = inputs.pop("inject_vectors")
                pos = inputs.pop("inject_positions")
                mask = inputs.pop("inject_mask")
                inputs.pop("meta", None)
                device = next(model.parameters()).device
                hook_state.set(vec.to(device), pos.to(device), mask.to(device))
                try:
                    return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
                finally:
                    hook_state.clear()

        return _Trainer


def save_adapter_bf16(model, path: Path) -> None:
    """Write the LoRA weights in bf16 to ``path`` (atomic via a .tmp directory); skip if it exists."""
    import shutil

    import torch

    if path.exists():
        return
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items() if "lora_" in k}
    if not sd:
        raise RuntimeError("no LoRA weights in the model state dict")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp), state_dict=sd)
    os.replace(tmp, path)


def _tiny_random_model(tok, cfg: dict[str, Any]):
    """CPU smoke: random Qwen3 with reduced width/depth, real tokenizer vocab."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    base = cfg.get("model", "Qwen/Qwen3-14B")
    conf = AutoConfig.from_pretrained(base)
    conf.hidden_size = int(cfg.get("tiny_hidden_size", 64))
    conf.intermediate_size = int(cfg.get("tiny_intermediate_size", 128))
    conf.num_hidden_layers = int(cfg.get("tiny_num_layers", 4))
    conf.num_attention_heads = int(cfg.get("tiny_num_heads", 4))
    conf.num_key_value_heads = int(cfg.get("tiny_num_kv_heads", 2))
    conf.head_dim = int(cfg.get("tiny_head_dim", 16))
    model = AutoModelForCausalLM.from_config(conf)
    return model


def dist_env() -> tuple[int, int, int]:
    """(world_size, rank, local_rank) from torchrun's environment; (1, 0, 0) for a plain process."""
    return (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def resolve_batching(cfg: dict[str, Any], world_size: int) -> dict[str, int]:
    """Per-GPU micro-batch, accumulation steps and effective batch for ``world_size`` processes.

    ``effective_batch`` (sequences per optimizer step, over all GPUs and accumulation) is fixed by the
    config; accumulation = effective_batch / (batch_size × world_size) must come out whole.
    """
    if "grad_accum" in cfg:
        raise ReaderConfigError(
            "set effective_batch (sequences per optimizer step over all GPUs), not grad_accum; "
            "accumulation is derived from it and the GPU count"
        )
    per_gpu = int(cfg.get("batch_size", 2))
    effective = int(cfg.get("effective_batch", per_gpu * world_size))
    per_step = per_gpu * world_size
    if per_gpu < 1 or effective < per_step or effective % per_step:
        raise ReaderConfigError(
            f"effective_batch {effective} is not a multiple of batch_size {per_gpu} x {world_size} GPU(s); "
            "change --batch-size or the number of GPUs"
        )
    return {"batch_size": per_gpu, "grad_accum": effective // per_step, "effective_batch": effective,
            "world_size": world_size}


def launch_command(gpus: str, argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Command and environment that run this trainer on the physical GPUs in ``gpus`` ("2" or "2,3").

    One GPU: a plain process. Several: torchrun with one process per GPU (``--standalone`` picks a free
    port, so several DDP runs can share a machine). ``--device`` is appended so a config's ``device: cpu``
    cannot silently win over the GPUs asked for (under torchrun each rank then takes ``cuda:<LOCAL_RANK>``).
    """
    ids = [g.strip() for g in gpus.split(",") if g.strip()]
    if not ids or not all(g.isdigit() for g in ids) or len(set(ids)) != len(ids):
        raise ReaderConfigError(f"--gpus takes distinct GPU indices such as 2 or 2,3; got {gpus!r}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(ids))
    module = ["-m", "main_run.reader.train", *argv, "--device", "cuda:0" if len(ids) == 1 else "cuda"]
    if len(ids) == 1:
        return [sys.executable, *module], env
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(ids)}",
            *module], env


def train_from_config(cfg: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    world_size, rank, local_rank = dist_env()
    is_main = rank == 0  # only rank 0 writes RUN.json, logs, snapshots and runs the NLL eval
    device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        if device != "cpu":
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
    elif device != "cpu" and torch.cuda.device_count() > 1:
        # A single process that sees several GPUs gets wrapped in nn.DataParallel by the HF Trainer,
        # and the injection hook would then see a slice of the batch.
        raise SystemExit(
            f"{torch.cuda.device_count()} GPUs visible to one process; choose them with --gpus "
            "(e.g. --gpus 2 for one GPU, --gpus 2,3 for DDP on two)"
        )
    batching = resolve_batching(cfg, world_size)

    layers = list(cfg["layers"])
    reader_dir = Path(cfg["reader_dir"])
    center = bool(cfg.get("center", True))
    max_len = int(cfg.get("max_len", 2048))
    train_jsonl = Path(cfg.get("train_jsonl") or (reader_dir / "train_correct.jsonl"))
    eval_jsonl = Path(cfg.get("eval_jsonl") or (reader_dir / "eval_samples.jsonl"))
    nll_every = int(cfg.get("nll_every", 0))
    nll_split_cfg = cfg.get("nll_split", "val")
    nll_split: str | None = None if nll_split_cfg == "all" else str(nll_split_cfg)
    if nll_every > 0:
        # Checked before anything is written; the eval's ReaderSamples refuses test rows as well.
        n_test = sealed_test_rows(eval_jsonl, nll_split)
        if nll_split == "test" or n_test:
            raise ReaderConfigError(
                f"nll_split={nll_split_cfg!r} would evaluate {n_test} sealed test rows of {eval_jsonl} "
                "during training; evaluate test offline with nll_eval --unseal once the rubric/judge/reader "
                "are fixed"
            )

    out = Path(cfg["out"]).resolve()
    run_path = out / "RUN.json"
    run_json, resume_from = None, None
    pruned: dict[str, Any] | None = None
    skipped_ckpts: list[tuple[str, str]] = []
    if resume:
        if not run_path.exists():
            raise SystemExit(f"--resume but {run_path} does not exist")
        latest, skipped_ckpts = latest_complete_checkpoint(out / "checkpoints")
        for name, why in skipped_ckpts:
            # Logging is configured further down; this goes to stderr now and to train.log below.
            print(f"[reader.train] WARNING --resume: skipping incomplete {name}: {why}", file=sys.stderr, flush=True)
        if latest is None:
            raise SystemExit(
                f"--resume but no complete checkpoint in {out}/checkpoints"
                + (f" (incomplete: {skipped_ckpts})" if skipped_ckpts else "")
            )
        resume_from = str(latest)
        resume_step = int(CHECKPOINT_RE.match(latest.name).group(1))
        if 0 < int(cfg.get("max_steps", -1)) <= resume_step:
            # The HF Trainer would run one step past max_steps from here (epoch-based runs stop).
            raise SystemExit(
                f"--resume: {resume_from} is already at max_steps={cfg['max_steps']}; nothing to resume "
                "(run nll_eval on the last adapter if its final eval is missing)"
            )
        if is_main:
            run_json = json.loads(run_path.read_text(encoding="utf-8"))
            # One row per step: rows after the checkpoint are redone by this segment.
            pruned = {"resume_step": resume_step}
            for name in ("train_log.jsonl", "nll_eval.jsonl"):
                pruned[name] = prune_jsonl_after(out / name, resume_step)
            evals = run_json.get("nll_evals", [])
            run_json["nll_evals"] = [r for r in evals if int(r["step"]) <= resume_step]
            pruned["RUN.json nll_evals"] = len(evals) - len(run_json["nll_evals"])
    elif is_main:
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not empty; use --resume or a new out")
        out.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if is_main:
        handlers.append(logging.FileHandler(out / "train.log"))
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(message)s",
        force=True,
        handlers=handlers,
    )
    for name, why in skipped_ckpts:
        log.warning("--resume: skipped incomplete checkpoint %s (%s); resuming from %s", name, why, resume_from)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    model_id = cfg.get("model", "Qwen/Qwen3-14B")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    weighting = str(cfg.get("weighting", "tokensum"))
    center_diag = None
    if center and is_main:
        center_diag = centering_diagnostics(reader_dir, layers=layers, weighting=weighting)

    train_ds = ReaderSamples(
        train_jsonl,
        reader_dir,
        tok=tok,
        layers=layers,
        representation=cfg.get("representation", "bundle_sum"),
        center=center,
        weighting=weighting,
        subset=cfg.get("subset"),
        max_len=max_len,
        for_generation=False,
        conditions=("correct",),
        require_split="train",
    )
    if cfg.get("max_train_samples"):
        n = int(cfg["max_train_samples"])
        train_ds.rows = train_ds.rows[:n]
        train_ds._encoded = [None] * len(train_ds.rows)

    dtype = torch.bfloat16 if device != "cpu" and cfg.get("bf16", True) else torch.float32

    if cfg.get("tiny_random"):
        model = _tiny_random_model(tok, cfg)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            device_map={"": device} if device != "cpu" else None,
        )
        if device == "cpu":
            model.to(device)

    model, lora_info = build_peft_model(model, cfg)

    # Hook settings from the chosen adapter's ao_config when present; defaults otherwise.
    adapter_dir = lora_info.get("adapter_dir")
    hook_cfg = load_ao_hook_settings(adapter_dir)
    hook_layer = int(cfg.get("hook_onto_layer", hook_cfg["hook_onto_layer"]))
    coeff = float(cfg.get("steering_coefficient", hook_cfg["steering_coefficient"]))
    hook_state = HookState(steering_coefficient=coeff)
    hook = InjectionHook(hook_state, hook_layer=hook_layer).attach(model)

    batch_size = batching["batch_size"]
    grad_accum = batching["grad_accum"]
    epochs = float(cfg.get("epochs", 1))
    max_steps = int(cfg.get("max_steps", -1))
    lr = float(cfg.get("lr", 1e-5 if cfg.get("init", "adapter") == "adapter" else 3e-5))
    warmup_steps = int(cfg.get("warmup_steps", 100 if cfg.get("init", "adapter") == "adapter" else 500))
    weight_decay = float(cfg.get("weight_decay", 0.0 if cfg.get("init", "adapter") == "adapter" else 0.01))
    lr_scheduler_type = cfg.get("lr_scheduler_type", "linear")
    grad_ckpt = bool(cfg.get("gradient_checkpointing", False))
    save_steps = int(cfg.get("save_steps", 200))
    logging_steps = int(cfg.get("logging_steps", 1))

    # Length stats for L1/L3/L40-style prompts go into RUN.json (R0 throughput note).
    length_stats = {"train_sample": train_ds.length_stats}

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(cfg.get("keep_checkpoints", 2)),
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        optim=cfg.get("optim", "adamw_torch"),
        use_cpu=(device == "cpu"),
        # DDP: every LoRA weight is used in each forward; the timeout covers rank 0's NLL evals,
        # during which the other ranks wait at the next gradient all-reduce.
        ddp_find_unused_parameters=False,
        ddp_timeout=int(cfg.get("ddp_timeout", 3600)),
    )

    collator = ReaderCollator(tok, side="right", d_model=model.config.hidden_size)
    TrainerCls = InjectionTrainer.make(Trainer, hook_state)
    trainer = TrainerCls(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tok,
    )

    segment = 0 if run_json is None else len(run_json.get("segments", []))
    settings = {
        "model": model_id,
        "layers": layers,
        "center": center,
        "weighting": weighting,
        "representation": cfg.get("representation", "bundle_sum"),
        "reader_dir": str(reader_dir),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "nll_every": nll_every,
        "nll_split": nll_split_cfg,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batching["effective_batch"],
        "world_size": world_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "epochs": epochs,
        "max_steps": max_steps,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "gradient_checkpointing": grad_ckpt,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "hook_onto_layer": hook_layer,
        "steering_coefficient": coeff,
        "max_len": max_len,
        "seed": seed,
        "tiny_random": bool(cfg.get("tiny_random")),
        "lora": lora_info,
    }
    seg = {
        "segment": segment,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "resume_from": resume_from,
        "host": os.uname().nodename,
        "pid": os.getpid(),
    }
    if pruned is not None:
        seg["pruned_after_resume"] = pruned
    if skipped_ckpts:
        seg["skipped_incomplete_checkpoints"] = [{"name": n, "why": w} for n, w in skipped_ckpts]
    if is_main:
        if run_json is None:
            run_json = {
                "settings": settings,
                "centering": center_diag,
                "length_stats": length_stats,
                "env": {
                    "git": git_rev(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "segments": [seg],
            }
        else:
            run_json["segments"].append(seg)
            run_json["settings"] = settings
        write_json_atomic(run_path, run_json)

    # Optional NLL eval callback every nll_every steps (+ step 0 and final if not already done).
    if nll_every > 0:
        from transformers import TrainerCallback

        from main_run.reader.nll_eval import evaluate_nll

        # Steps already in nll_eval.jsonl (after the resume pruning) are not evaluated again,
        # e.g. the final step when a run is resumed from its last checkpoint.
        done_steps = (
            {int(r["step"]) for r in run_json.get("nll_evals", [])} if is_main and run_json else set()
        )

        class NLLCallback(TrainerCallback):
            def __init__(self):
                self._evaluated_steps: set[int] = set(done_steps)

            def on_train_begin(self, args, state, control, model=None, **kw):
                if state.global_step == 0:  # the untrained reader (not on resume)
                    self._eval(state, model)

            def on_step_end(self, args, state, control, model=None, **kw):
                if state.global_step % nll_every == 0:
                    self._eval(state, model)

            def on_train_end(self, args, state, control, model=None, **kw):
                # Final step is often not a multiple of nll_every (e.g. 6080 vs 1000).
                if state.global_step not in self._evaluated_steps:
                    self._eval(state, model)

            def _eval(self, state, model):
                if not is_main or state.global_step in self._evaluated_steps:
                    return
                # Runs outside training_step, when the training hook state is clear; the
                # eval reuses that hook through the same state holder.
                result = evaluate_nll(
                    model,
                    tok,
                    eval_jsonl=eval_jsonl,
                    reader_dir=reader_dir,
                    layers=layers,
                    center=center,
                    batch_size=min(batch_size, 4),
                    max_len=max_len,
                    split=nll_split,
                    limit=cfg.get("nll_limit"),
                    hook_state=hook_state,
                    weighting=weighting,
                )
                slim = {
                    k: result[k]
                    for k in ("n", "layers", "center", "split", "weighting", "reliance_gaps", "condition_nll")
                }
                slim["step"] = state.global_step
                append_jsonl(out / "nll_eval.jsonl", slim)
                run_json.setdefault("nll_evals", []).append(slim)
                write_json_atomic(run_path, run_json)
                self._evaluated_steps.add(state.global_step)

        trainer.add_callback(NLLCallback())

    from transformers import TrainerCallback

    class TrainLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not is_main or not logs or "loss" not in logs:
                return
            rec = {
                "step": state.global_step,
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "lr": logs.get("learning_rate"),
                "epoch": logs.get("epoch"),
                "t": round(time.time(), 2),
                "peak_mem_gib": (
                    round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() and device != "cpu" else None
                ),
            }
            append_jsonl(out / "train_log.jsonl", rec)

        def on_save(self, args, state, control, **kw):
            if not is_main:
                return
            # bf16 adapter snapshot next to the (fp32, resumable) Trainer checkpoint.
            save_adapter_bf16(trainer.model, out / "adapters" / f"step_{state.global_step:06d}")

    trainer.add_callback(TrainLogCallback())

    if resume_from is None and is_main:
        # Step 0: the adapter as initialised (scratch: B = 0, so base model + injection).
        save_adapter_bf16(model, out / "adapters" / "step_000000")

    t0 = time.time()
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    finally:
        hook.remove()
    if not is_main:
        return {}
    run_json["segments"][-1].update(
        end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_step=trainer.state.global_step,
        wall_s=round(time.time() - t0, 1),
    )
    save_adapter_bf16(trainer.model, out / "adapters" / f"step_{trainer.state.global_step:06d}")
    write_json_atomic(run_path, run_json)
    log.info("done: step %d, wall %.1fs", trainer.state.global_step, time.time() - t0)
    return run_json


def _without_gpus_flag(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a == "--gpus":
            skip = True
        elif not a.startswith("--gpus="):
            out.append(a)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Train the C1 gradient reader")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="override config out")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--gpus",
        default=None,
        help="physical GPU indices: '2' runs one process on GPU 2; '2,3' runs DDP on GPUs 2 and 3",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="per-GPU micro-batch (memory knob); accumulation adjusts so effective_batch stays the config's",
    )
    p.add_argument("--gradient-checkpointing", choices=["on", "off"], default=None, help="override the config")
    p.add_argument("--device", default=None, help="single-process device, e.g. cpu (prefer --gpus for GPUs)")
    args = p.parse_args(argv)
    if args.gpus is not None:
        if dist_env()[0] > 1:
            raise SystemExit("--gpus starts torchrun itself; do not combine it with an outer torchrun")
        if args.device is not None:
            raise SystemExit("--gpus and --device both choose the device; pass one")
        cmd, env = launch_command(args.gpus, _without_gpus_flag(argv))
        print(f"[reader.train] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)}", flush=True)
        os.execvpe(cmd[0], cmd, env)
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.gradient_checkpointing is not None:
        cfg["gradient_checkpointing"] = args.gradient_checkpointing == "on"
    if args.out is not None:
        cfg["out"] = str(args.out)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.device is not None:
        cfg["device"] = args.device
    train_from_config(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

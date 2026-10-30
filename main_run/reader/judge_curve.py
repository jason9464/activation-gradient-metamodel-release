# Release note: this is the em-reader branch version of this file (the SFT and reading runs used it; rl-stage-fixes carries a differing copy).
"""Judge-score learning curve for reader runs, next to the in-training NLL gate.

For every saved adapter snapshot (``<run>/adapters/step_XXXXXX``) of one or more reader runs: generate
greedy answers (thinking off) on the val rows of each condition with the run's own injection settings
(``RUN.json``), score them with the C1 judge (``c1/score.py``: same rubric, cache and retries), and
append one row per snapshot to ``<run>/judge_curve.jsonl``. ``nll_eval.jsonl`` has rows at the same
steps, so validation NLL and judge score can be read side by side (``plot_curves.py``).

The base model is loaded once per process; each snapshot's PEFT adapter is loaded on it and unloaded
again (no merge, so the base weights never change). Predictions are kept in
``<run>/judge_curve/preds_step_XXXXXX.jsonl`` (scored rows and the score summary next to them).

``--watch`` polls for new snapshots in all run dirs, evaluates them in order of examples seen, skips
steps already in the output, and stops once every run's final step is evaluated. ``--once`` evaluates
what exists and exits. A step counts as done only for the same evaluation settings (split,
conditions, bundle limit, max_new_tokens, judge request), so a smoke row never hides a full one.
One process per output file (``<out>.lock``). The split defaults to the run's NLL split, so with no
``--limit`` the judged rows are the NLL curve's rows (same qa_id and vec_row), conditions permitting.
The injected vectors use the run's training weighting (``RUN.json`` settings.weighting, tokensum when
absent) unless ``--weighting`` says otherwise. ``--retry-errors`` loads no model: it re-judges only the
``error`` verdicts of steps already in the output and rewrites their rows. ``--judge-provider`` (default:
the judge config's judge_provider) picks the judge API, anthropic or openrouter; it is part of the eval key,
so a curve judged by one provider is never mixed with the other (rows without the field are openrouter).
The two providers share a snapshot's predictions (generated once, never regenerated for the other judge)
and keep their own scored files (``preds_step_XXXXXX.scored.jsonl`` for openrouter,
``preds_step_XXXXXX.anthropic.scored.jsonl`` for anthropic).

``--fast-gen`` (the default; ``--no-fast-gen`` turns it off) generates with the
fast path of ``generate.py`` (LoRA weights kept in bf16, static KV cache, compiled decode step). It enters the
eval key, so its rows never mix with rows of the former path (fp32 LoRA weights, dynamic cache, eager), which
keep their old key. Curves written before it became the default are former-path rows: to re-judge their errors
(``--retry-errors``) or extend them, pass ``--no-fast-gen``. Rows of another eval key in the same output file get
their prediction files overwritten, so give fast rows their own ``--out-name`` on such a run. The fast path needs
more GPU memory (batch 128: 68.4 GiB allocated instead of 50.5; out of memory under the former 58 GiB cap), so
``--max-gpu-gib`` defaults to 90. With a nested ``--out-name`` (e.g. ``ab_gen/x.jsonl``) the predictions go next to the output file
(``<run>/ab_gen/x/``); for a plain name that is the same place as before (``<run>/<stem>/``).

Gate quantities: Δ_judge(c) = score_correct − score_c, per question, a bundle's
questions averaged first, bundle-bootstrap 95% CI (seed 0); positive = the gradient helped.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Collection, Sequence

import numpy as np

from main_run.c1.score import (
    JUDGE_CFG,
    PROVIDERS,
    bootstrap_mean_ci,
    judge_model_id,
    load_yaml,
    reliance_gaps,
    resolve_provider,
    score_predictions,
    sha256_json,
    sha256_text,
)
from main_run.reader.data import sealed_test_rows

SNAP_RE = re.compile(r"^step_(\d{6})$")
DONE_RE = re.compile(r"done: step (\d+)")
CONDITIONS = ("correct", "shuffled", "no_vector")
# Settings the generation must copy from the run; missing any of them is refused.
REQUIRED_SETTINGS = (
    "model",
    "layers",
    "center",
    "reader_dir",
    "max_len",
    "hook_onto_layer",
    "steering_coefficient",
    "effective_batch",
)
# Judge request fields that change a verdict; part of the eval key (openrouter; unchanged, so rows
# written before the provider switch keep their key).
JUDGE_KEY_FIELDS = (
    "openrouter_model",
    "max_tokens",
    "temperature",
    "reasoning",
    "openrouter_provider",
    "max_answer_chars",
    "system",
    "user_template",
)
# The same for the Anthropic Messages API request.
ANTHROPIC_JUDGE_KEY_FIELDS = (
    "anthropic_model",
    "max_tokens",
    "temperature",
    "anthropic_thinking",
    "max_answer_chars",
    "system",
    "user_template",
)
# Judge requests in flight when --judge-workers is not given.
JUDGE_WORKERS = {"anthropic": 32, "openrouter": 16}
# ``via`` of a verdict that cost an API call in this pass.
FRESH_VIA = ("openrouter", "anthropic")
REQUEST_ERROR = "request_error"


class JudgeCurveError(ValueError):
    """A run dir lacks a setting the evaluation needs, or runs disagree on the base model."""


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [judge_curve] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file; a last line still being appended (unparseable) is left out."""
    if not path.exists():
        return []
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:
                raise
    return rows


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


# --- Run dir bookkeeping ---


def load_run_settings(run_dir: Path) -> dict[str, Any]:
    """``RUN.json`` settings with every key in REQUIRED_SETTINGS present; refuses otherwise."""
    path = Path(run_dir) / "RUN.json"
    if not path.exists():
        raise JudgeCurveError(f"{path} does not exist")
    settings = json.loads(path.read_text(encoding="utf-8")).get("settings") or {}
    missing = [k for k in REQUIRED_SETTINGS if settings.get(k) is None]
    if missing:
        raise JudgeCurveError(f"{path}: settings lack {missing}; refusing to guess the injection setup")
    rep = settings.get("representation", "bundle_sum")
    if rep != "bundle_sum":
        raise JudgeCurveError(f"{path}: representation {rep!r}; only bundle_sum is implemented")
    return settings


def examples_seen(step: int, settings: dict[str, Any]) -> int:
    """Training sequences seen by ``step``: optimizer steps × effective batch (ao_L40 16, scratch_L40 8)."""
    return int(step) * int(settings["effective_batch"])


def snapshot_steps(run_dir: Path) -> list[int]:
    """Steps of the complete adapter snapshots (written atomically by train.py), ascending."""
    adir = Path(run_dir) / "adapters"
    if not adir.is_dir():
        return []
    steps = []
    for p in adir.iterdir():
        m = SNAP_RE.match(p.name)
        if m and (p / "adapter_config.json").exists() and (p / "adapter_model.safetensors").exists():
            steps.append(int(m.group(1)))
    return sorted(steps)


def finished_step(run_dir: Path) -> int | None:
    """The run's last step once training has ended: RUN.json's last segment ``end_step`` or the
    ``done: step N`` line of train.log; None while it is still training."""
    run_dir = Path(run_dir)
    try:
        segs = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8")).get("segments") or []
    except (OSError, ValueError):
        segs = []
    if segs and segs[-1].get("end_step") is not None:
        return int(segs[-1]["end_step"])
    log_path = run_dir / "train.log"
    if log_path.exists():
        hits = DONE_RE.findall(log_path.read_text(encoding="utf-8", errors="replace"))
        if hits:
            return int(hits[-1])
    return None


def planned_total_steps(settings: dict[str, Any], n_train: int) -> int | None:
    """Optimizer steps the HF Trainer plans for this run (transformers 5.x formula).

    ``max_steps`` > 0 wins; otherwise ceil(epochs × ceil(len_dataloader / grad_accum)) with
    len_dataloader = ceil(ceil(n_train / world_size) / batch_size). ``n_train`` is the training
    row count (RUN.json does not record max_train_samples / subset, so this is a cross-check;
    the done line of train.log is the authority).
    """
    max_steps = int(settings.get("max_steps", -1) or -1)
    if max_steps > 0:
        return max_steps
    try:
        ws = int(settings.get("world_size", 1))
        bs = int(settings["batch_size"])
        ga = int(settings["grad_accum"])
        epochs = float(settings.get("epochs", 1))
    except (KeyError, TypeError, ValueError):
        return None
    if n_train <= 0:
        return None
    len_dl = math.ceil(math.ceil(n_train / ws) / bs)
    return math.ceil(epochs * math.ceil(len_dl / ga))


def count_correct_rows(jsonl: Path) -> int:
    n = 0
    with Path(jsonl).open(encoding="utf-8") as f:
        for line in f:
            if '"condition": "correct"' in line or '"condition":"correct"' in line:
                n += 1
    return n


def training_alive(run_dir: Path) -> bool | None:
    """Whether the run's last training process is alive; None when it ran on another host."""
    try:
        segs = json.loads((Path(run_dir) / "RUN.json").read_text(encoding="utf-8")).get("segments") or []
    except (OSError, ValueError):
        return None
    if not segs or segs[-1].get("host") != socket.gethostname() or segs[-1].get("pid") is None:
        return None
    try:
        os.kill(int(segs[-1]["pid"]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_locks(out_paths: Sequence[Path]) -> list[Any]:
    """Exclusive lock per output file (``<out>.lock``, flock), so a second judge_curve process on the
    same run and output cannot append the same step twice. Held until ``release_locks`` or exit."""
    held = []
    for path in out_paths:
        f = open(Path(path).with_name(Path(path).name + ".lock"), "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            release_locks(held)
            raise SystemExit(f"another judge_curve process is writing {path}; refusing to start a second one")
        held.append(f)
    return held


def release_locks(held: Sequence[Any]) -> None:
    for f in held:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def nll_split_of(settings: dict[str, Any]) -> str:
    """The split the run's in-training NLL eval used (train.py ``nll_split``, default val)."""
    return str(settings.get("nll_split") or "val")


def nll_rows_note(run_dir: Path, settings: dict[str, Any], split: str, limit: int | None) -> str | None:
    """Why this run's judge rows would not be the rows of its NLL curve, or None when they are:
    NLL rows from nll_eval.jsonl ``n`` vs every row of the split in eval_samples (3 conditions)."""
    if limit is not None:
        return f"--limit {limit}: judge on the first {limit} bundles, NLL on all of them"
    if split != nll_split_of(settings):
        return f"judge split {split} but NLL split {nll_split_of(settings)}"
    rows = read_jsonl(Path(run_dir) / "nll_eval.jsonl")
    if not rows:
        return None
    eval_jsonl = Path(settings.get("eval_jsonl") or Path(settings["reader_dir"]) / "eval_samples.jsonl")
    n_split = sum(1 for r in read_jsonl(eval_jsonl) if r.get("split") == split)
    n_nll = {int(r["n"]) for r in rows if r.get("n") is not None}
    if n_nll and n_nll != {n_split}:
        return f"NLL rows per eval {sorted(n_nll)} but the {split} split has {n_split} (nll_limit set?)"
    return None


# --- Eval identity and pending work ---


def judge_identity(cfg: dict[str, Any], provider: str = "openrouter") -> dict[str, Any]:
    fields = ANTHROPIC_JUDGE_KEY_FIELDS if provider == "anthropic" else JUDGE_KEY_FIELDS
    return {k: cfg.get(k) for k in fields}


def eval_key(
    *,
    split: str,
    conditions: Sequence[str],
    limit_bundles: int | None,
    max_new_tokens: int,
    judge_cfg: dict[str, Any],
    model: str,
    weighting: str = "tokensum",
    judge_provider: str = "openrouter",
    fast_gen: bool = False,
) -> str:
    """Hash of everything that changes a row's numbers (not batch size or worker count).

    The weighting enters only when it is not tokensum, the judge provider only when it is not
    openrouter, and the generation path only with ``fast_gen`` (the CLI default; the
    library default False is the former path), so rows written before any of them was a setting keep
    their key."""
    ident = {
        "split": split,
        "conditions": sorted(conditions),
        "limit_bundles": limit_bundles,
        "max_new_tokens": int(max_new_tokens),
        "judge": judge_identity(judge_cfg, judge_provider),
        "model": model,
        "thinking": False,
        "decoding": "greedy",
    }
    if weighting != "tokensum":
        ident["weighting"] = weighting
    if judge_provider != "openrouter":
        ident["judge_provider"] = judge_provider
    if fast_gen:
        from main_run.reader.generate import FAST_GEN_TAG

        ident["generation_path"] = FAST_GEN_TAG
    return sha256_json(ident)[:16]


def done_steps(out_path: Path, key: str) -> set[int]:
    return {int(r["step"]) for r in read_jsonl(out_path) if r.get("eval_key") == key}


def pending_snapshots(
    runs: Sequence[dict[str, Any]], done: Sequence[set[int]]
) -> list[tuple[int, int, int]]:
    """(examples_seen, run index, step) for snapshots not yet evaluated, in the order to run them:
    by examples seen (matched points of the two cadences together), then run, then step."""
    todo = []
    for i, run in enumerate(runs):
        for step in snapshot_steps(run["dir"]):
            if step not in done[i]:
                todo.append((examples_seen(step, run["settings"]), i, step))
    return sorted(todo)


def run_complete(run: dict[str, Any], done: set[int]) -> tuple[bool, str]:
    """(stop watching this run, why). Complete once its final step (train.log done line or
    RUN.json end_step, else the planned total) is evaluated; also when its training process on this
    host is gone with nothing left to evaluate."""
    fin = finished_step(run["dir"])
    if fin is not None and fin in done:
        return True, f"final step {fin} evaluated"
    planned = run.get("planned_total")
    if planned is not None and planned in done:
        return True, f"planned total {planned} evaluated"
    if fin is None and training_alive(run["dir"]) is False:
        left = [s for s in snapshot_steps(run["dir"]) if s not in done]
        if not left:
            return True, "training process gone, no snapshot left (run did not finish)"
    return False, ""


def val_bundle_ids(eval_jsonl: Path, split: str, limit: int | None) -> list[str]:
    """Bundles of ``split`` in first-appearance order of eval_samples.jsonl; the first ``limit``."""
    seen: dict[str, None] = {}
    for r in read_jsonl(Path(eval_jsonl)):
        if r.get("split") == split:
            seen.setdefault(r["bundle_id"], None)
    ids = list(seen)
    return ids if limit is None else ids[:limit]


# --- Row statistics ---


def condition_score_ci(scored: list[dict[str, Any]], *, score_key: str = "score", seed: int = 0) -> dict[str, Any]:
    """Per condition: mean judge score over bundles (a bundle's questions averaged first, ok verdicts
    only) with bundle-bootstrap 95% CI."""
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in scored:
        if score_key == "score" and r.get("judge_status") != "ok":
            continue
        if r.get(score_key) is None:
            continue
        per[(r["condition"], r["bundle_id"])].append(float(r[score_key]))
    out: dict[str, Any] = {}
    for cond in sorted({c for c, _ in per}):
        vals = [float(np.mean(v)) for (c, _), v in sorted(per.items()) if c == cond]
        out[cond] = bootstrap_mean_ci(vals, seed=seed)
    return out


def _gaps_for(conditions: Sequence[str], gaps: dict[str, Any]) -> dict[str, Any]:
    """reliance_gaps without the comparisons whose condition was not run."""
    keep = {"shuffled": "correct_minus_shuffled", "no_vector": "correct_minus_no_vector"}
    want = [v for c, v in keep.items() if c in conditions]
    return {g: {k: d[k] for k in want if k in d} for g, d in gaps.items()}


def judge_cost(scored: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Spend of this scoring pass: fresh API calls only (cache hits cost nothing now).

    Uses OpenRouter's reported ``usage.cost`` when every call went to OpenRouter and has it, else tokens ×
    the config's ``price_usd_per_mtok`` (always for Anthropic calls).
    """
    fresh = [r for r in scored if r.get("via") in FRESH_VIA]
    usages = [r.get("judge_usage") or {} for r in fresh]
    pt = int(sum(int(u.get("prompt_tokens") or 0) for u in usages))
    ct = int(sum(int(u.get("completion_tokens") or 0) for u in usages))
    price = cfg.get("price_usd_per_mtok") or {}
    est = None
    if price:
        est = pt * float(price.get("prompt", 0)) / 1e6 + ct * float(price.get("completion", 0)) / 1e6
    reported = [u.get("cost") for u in usages if u]
    all_openrouter = all(r.get("via") == "openrouter" for r in fresh)
    if fresh and all_openrouter and len(reported) == len(fresh) and all(c is not None for c in reported):
        usd, source = float(sum(float(c) for c in reported)), "openrouter_usage_cost"
    else:
        usd, source = est, "tokens_x_config_price" if est is not None else "unknown"
    return {
        "n_api_calls": len(fresh),
        "n_cache_hits": sum(1 for r in scored if r.get("via") == "cache"),
        "n_request_errors": sum(1 for r in scored if str(r.get("reason", "")).startswith(REQUEST_ERROR)),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "usd": usd,
        "usd_source": source,
        "usd_from_config_price": est,
    }


def generation_stats(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of answers that stopped at max_new_tokens without an EOS, and mean length, per condition."""
    out: dict[str, Any] = {}
    by: dict[str, list[dict]] = defaultdict(list)
    for r in preds:
        by[r["condition"]].append(r)
    for cond, rows in sorted(by.items()) + [("all", preds)]:
        hits = [bool(r.get("hit_max_new_tokens")) for r in rows]
        lens = [int(r["n_new_tokens"]) for r in rows if r.get("n_new_tokens") is not None]
        out[cond] = {
            "n": len(rows),
            "frac_hit_max_new_tokens": float(np.mean(hits)) if hits else None,
            "mean_new_tokens": float(np.mean(lens)) if lens else None,
        }
    return out


def build_row(
    *,
    step: int,
    run: dict[str, Any],
    key: str,
    split: str,
    conditions: Sequence[str],
    limit_bundles: int | None,
    max_new_tokens: int,
    preds: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    summary: dict[str, Any],
    cfg: dict[str, Any],
    wall: dict[str, float],
    mem: dict[str, float | None],
    extra: dict[str, Any] | None = None,
    provider: str = "openrouter",
) -> dict[str, Any]:
    """One judge_curve.jsonl row for a snapshot (see module docstring for the gate quantities)."""
    gaps = _gaps_for(conditions, summary.get("reliance_gaps") or reliance_gaps(scored))
    delta = {
        c: gaps["all"][f"correct_minus_{c}"] for c in ("shuffled", "no_vector") if c in conditions
    }
    row = {
        "step": int(step),
        "examples_seen": examples_seen(step, run["settings"]),
        "run": Path(run["dir"]).name,
        "eval_key": key,
        "judge_provider": provider,
        "split": split,
        "conditions": list(conditions),
        "limit_bundles": limit_bundles,
        "n_bundles": len({r["bundle_id"] for r in preds}),
        "n_rows": len(preds),
        "max_new_tokens": int(max_new_tokens),
        "judge_score": condition_score_ci(scored),
        "delta_judge": delta,
        "judge_status_counts": summary.get("judge_status_counts"),
        "n_judge_error_excluded": summary.get(
            "n_judge_error_excluded", sum(1 for r in scored if r.get("judge_status") == "error")
        ),
        "by_condition": summary.get("by_condition"),
        "reliance_gaps": gaps,
        "reliance_gaps_tfidf_rr": _gaps_for(conditions, summary.get("reliance_gaps_tfidf_rr") or {}),
        "generation": generation_stats(preds),
        "judge": {
            "model": judge_model_id(cfg, provider),
            "rubric_id": cfg.get("rubric_id"),
            "rubric_sha256": summary.get("rubric_sha256")
            or sha256_text(cfg["system"] + "\n" + cfg["user_template"]),
            **judge_cost(scored, cfg),
        },
        "wall_s": wall,
        "gpu_mem_gib": mem,
        "seed": 0,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        row.update(extra)
    return row


# --- Model and snapshot evaluation ---


class AdapterSlot:
    """One base model, one PEFT adapter at a time: load a snapshot, unload it (no merge) for the next."""

    def __init__(self, base, fast_gen: bool = False):
        self.base = base
        self.model = None
        self.path: Path | None = None
        self.fast_gen = fast_gen

    def load(self, path: Path):
        from peft import PeftModel

        self.unload()
        if self.fast_gen:
            from main_run.reader.generate import load_adapter_fast

            # --fast-gen: adapter weights kept in bf16 (generate.load_adapter_fast).
            self.model = load_adapter_fast(self.base, path)
        else:
            # Same call as generate.py (adapter weights upcast to fp32, as in training).
            self.model = PeftModel.from_pretrained(self.base, str(path), is_trainable=False)
        self.model.eval()
        self.path = Path(path)
        return self.model

    def unload(self) -> None:
        if self.model is None:
            return
        self.base = self.model.unload()  # restores the original modules, base weights untouched
        for attr in ("peft_config", "_hf_peft_config_loaded"):
            if hasattr(self.base, attr):
                try:
                    delattr(self.base, attr)
                except AttributeError:
                    pass
        self.model = None
        self.path = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _cuda_mem() -> dict[str, float | None]:
    import torch

    if not torch.cuda.is_available():
        return {"peak_allocated": None, "peak_reserved": None}
    return {
        "peak_allocated": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def generate_snapshot(
    slot: AdapterSlot,
    tok,
    *,
    run: dict[str, Any],
    step: int,
    preds_path: Path,
    key: str,
    split: str,
    conditions: Sequence[str],
    bundle_ids: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    weighting: str = "tokensum",
    reuse_keys: Collection[str] = (),
    fast_gen: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, float], bool]:
    """Predictions for one snapshot (reused when a finished file with the same eval key exists).

    ``reuse_keys``: eval keys of the same generation settings judged by the other provider; their
    predictions are reused as they are (the judge does not change the predictions) and their meta file is
    left alone."""
    from main_run.reader.generate import run_generate

    meta_path = preds_path.with_suffix(".meta.json")
    if preds_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("eval_key") == key or meta.get("eval_key") in reuse_keys:
            return read_jsonl(preds_path), meta.get("wall_s", {}), True

    s = run["settings"]
    t0 = time.time()
    model = slot.load(Path(run["dir"]) / "adapters" / f"step_{step:06d}")
    t1 = time.time()
    tmp = preds_path.with_name(preds_path.name + ".tmp")
    run_generate(
        model,
        tok,
        eval_jsonl=Path(s.get("eval_jsonl") or Path(s["reader_dir"]) / "eval_samples.jsonl"),
        reader_dir=Path(s["reader_dir"]),
        layers=list(s["layers"]),
        out_path=tmp,
        center=bool(s["center"]),
        batch_size=batch_size,
        max_len=int(s["max_len"]),
        max_new_tokens=max_new_tokens,
        hook_layer=int(s["hook_onto_layer"]),
        steering_coefficient=float(s["steering_coefficient"]),
        split=split,
        conditions=tuple(conditions),
        bundle_ids=bundle_ids,
        weighting=weighting,
        fast_gen=fast_gen,
    )
    os.replace(tmp, preds_path)
    t2 = time.time()
    wall = {"adapter_load": round(t1 - t0, 1), "generate": round(t2 - t1, 1)}
    meta_path.write_text(
        json.dumps({"eval_key": key, "step": step, "wall_s": wall, "batch_size": batch_size}, indent=2) + "\n",
        encoding="utf-8",
    )
    return read_jsonl(preds_path), wall, False


def scored_path_for(preds_path: Path, provider: str) -> Path:
    """Scored rows of one judge provider next to the predictions: openrouter keeps ``<preds>.scored.jsonl``,
    anthropic writes ``<preds>.anthropic.scored.jsonl``, so neither overwrites the other's verdicts."""
    tag = "" if provider == "openrouter" else f".{provider}"
    return preds_path.with_name(preds_path.stem + tag + ".scored.jsonl")


def score_snapshot(
    preds_path: Path,
    *,
    judge_cfg_path: Path,
    workers: int,
    dry_run: bool,
    retries: int = 2,
    provider: str = "openrouter",
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Judge every prediction (cached verdicts are reused); rows whose request failed are retried."""
    scored_path = scored_path_for(preds_path, provider)
    t0 = time.time()
    spent: list[dict[str, Any]] = []
    for attempt in range(retries + 1):
        summary = score_predictions(
            preds_path, out_path=scored_path, cfg_path=judge_cfg_path, dry_run=dry_run, workers=workers,
            provider=provider,
        )
        scored = read_jsonl(scored_path)
        spent.extend(r for r in scored if r.get("via") in FRESH_VIA and r.get("judge_usage"))
        n_err = sum(1 for r in scored if str(r.get("reason", "")).startswith(REQUEST_ERROR))
        if not n_err or attempt == retries:
            break
        log(f"{n_err} judge requests failed; retrying them in 30 s ({attempt + 1}/{retries})")
        time.sleep(30)
    # Count the spend of every pass, not only the last (a retried pass re-reads earlier verdicts from cache).
    if spent:
        by_sha = {r["request_sha"]: r for r in spent if r.get("request_sha")}
        for r in scored:
            if r.get("via") == "cache" and r.get("request_sha") in by_sha:
                r["via"] = by_sha[r["request_sha"]]["via"]
                r["judge_usage"] = by_sha[r["request_sha"]]["judge_usage"]
    return scored, summary, round(time.time() - t0, 1)


def evaluate_snapshot(
    slot: AdapterSlot,
    tok,
    *,
    run: dict[str, Any],
    step: int,
    out_path: Path,
    key: str,
    split: str,
    conditions: Sequence[str],
    limit_bundles: int | None,
    batch_size: int,
    max_new_tokens: int,
    judge_cfg_path: Path,
    judge_cfg: dict[str, Any],
    workers: int,
    dry_run: bool = False,
    executor=None,
    weighting: str = "tokensum",
    provider: str = "openrouter",
    reuse_keys: Collection[str] = (),
    fast_gen: bool = False,
):
    """Generate, judge and append one row for ``run`` at ``step``; returns the row.

    With ``executor`` (one worker thread) only the generation runs here: judging and the row are
    submitted to it and a Future of the row is returned, so the next snapshot generates on the GPU
    while this one is judged over the network. Same rows either way.
    """
    s = run["settings"]
    eval_jsonl = Path(s.get("eval_jsonl") or Path(s["reader_dir"]) / "eval_samples.jsonl")
    bundle_ids = val_bundle_ids(eval_jsonl, split, limit_bundles)
    preds_dir = out_path.parent / out_path.stem  # = <run>/<stem> for a plain --out-name
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"preds_step_{step:06d}.jsonl"

    _reset_peak()
    t0 = time.time()
    preds, gen_wall, reused = generate_snapshot(
        slot,
        tok,
        run=run,
        step=step,
        preds_path=preds_path,
        key=key,
        split=split,
        conditions=conditions,
        bundle_ids=bundle_ids,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        weighting=weighting,
        reuse_keys=reuse_keys,
        fast_gen=fast_gen,
    )
    mem = _cuda_mem()
    slot.unload()
    gen_elapsed = time.time() - t0

    def finish() -> dict[str, Any]:
        scored, summary, judge_s = score_snapshot(
            preds_path, judge_cfg_path=judge_cfg_path, workers=workers, dry_run=dry_run, provider=provider
        )
        wall = {**gen_wall, "judge": judge_s, "total": round(gen_elapsed + judge_s, 1)}
        return _finish_row(
            run=run, step=step, out_path=out_path, key=key, split=split, conditions=conditions,
            limit_bundles=limit_bundles, max_new_tokens=max_new_tokens, preds=preds, scored=scored,
            summary=summary, judge_cfg=judge_cfg, wall=wall, mem=mem, preds_path=preds_path, reused=reused,
            batch_size=batch_size, workers=workers, dry_run=dry_run, overlapped=executor is not None,
            weighting=weighting, provider=provider, fast_gen=fast_gen,
        )

    return finish() if executor is None else executor.submit(finish)


def _finish_row(
    *,
    run: dict[str, Any],
    step: int,
    out_path: Path,
    key: str,
    split: str,
    conditions: Sequence[str],
    limit_bundles: int | None,
    max_new_tokens: int,
    preds: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    summary: dict[str, Any],
    judge_cfg: dict[str, Any],
    wall: dict[str, float],
    mem: dict[str, float | None],
    preds_path: Path,
    reused: bool,
    batch_size: int,
    workers: int,
    dry_run: bool,
    overlapped: bool,
    weighting: str = "tokensum",
    provider: str = "openrouter",
    fast_gen: bool = False,
) -> dict[str, Any]:
    s = run["settings"]
    row = build_row(
        step=step,
        run=run,
        key=key,
        split=split,
        conditions=conditions,
        limit_bundles=limit_bundles,
        max_new_tokens=max_new_tokens,
        preds=preds,
        scored=scored,
        summary=summary,
        cfg=judge_cfg,
        wall=wall,
        mem=mem,
        provider=provider,
        extra={
            "adapter": str(Path(run["dir"]) / "adapters" / f"step_{step:06d}"),
            "preds": str(preds_path),
            "generation_reused": reused,
            "gen_batch_size": batch_size,
            "judge_workers": workers,
            "judge_overlapped": overlapped,
            "dry_run": dry_run,
            "hook_onto_layer": int(s["hook_onto_layer"]),
            "steering_coefficient": float(s["steering_coefficient"]),
            "layers": list(s["layers"]),
            "center": bool(s["center"]),
            "weighting": weighting,
            **({"generation_path": "fast_gen"} if fast_gen else {}),
        },
    )
    append_jsonl(out_path, row)
    return row


# Row fields that come from the verdicts; --retry-errors recomputes these and keeps the rest of the row.
VERDICT_FIELDS = (
    "judge_score",
    "delta_judge",
    "judge_status_counts",
    "n_judge_error_excluded",
    "by_condition",
    "reliance_gaps",
    "reliance_gaps_tfidf_rr",
)


def retry_error_rows(
    run: dict[str, Any],
    out_path: Path,
    *,
    key: str,
    judge_cfg_path: Path,
    judge_cfg: dict[str, Any],
    workers: int,
    dry_run: bool = False,
    steps: set[int] | None = None,
    provider: str = "openrouter",
) -> list[int]:
    """Re-judge the error verdicts of the steps already in ``out_path`` (this eval key) and rewrite
    their rows; returns the steps whose rows were rewritten.

    Only the error rows of the step's scored file are sent again (score.py ``retry_errors``); the
    predictions are not regenerated. A rewritten row keeps its generation, wall time and original
    judge spend, takes the verdict-derived fields from the new scoring, and gains ``judge_retry`` (when,
    rows retried, rows still in error, the retry's own spend). The caller holds the output's lock.
    """
    rows = read_jsonl(out_path)
    rewritten: list[int] = []
    for k, row in enumerate(rows):
        step = int(row["step"])
        if row.get("eval_key") != key or (steps is not None and step not in steps):
            continue
        if not (row.get("judge_status_counts") or {}).get("error"):
            continue
        preds_path = Path(row.get("preds") or out_path.parent / out_path.stem / f"preds_step_{step:06d}.jsonl")
        scored_path = scored_path_for(preds_path, provider)
        if not (preds_path.exists() and scored_path.exists()):
            log(f"WARNING {Path(run['dir']).name} step {step}: {scored_path.name} or its predictions missing; not retried")
            continue
        t0 = time.time()
        summary = score_predictions(
            preds_path, out_path=scored_path, cfg_path=judge_cfg_path, dry_run=dry_run, workers=workers,
            retry_errors=True, provider=provider,
        )
        scored = read_jsonl(scored_path)
        new = build_row(
            step=step, run=run, key=key, split=row["split"], conditions=row["conditions"],
            limit_bundles=row.get("limit_bundles"), max_new_tokens=row["max_new_tokens"],
            preds=read_jsonl(preds_path), scored=scored, summary=summary, cfg=judge_cfg,
            wall=row.get("wall_s") or {}, mem=row.get("gpu_mem_gib") or {}, provider=provider,
        )
        info = summary["retry_errors"]
        retried = set(info["retried_sample_ids"])
        rows[k] = {
            **row,
            **{f: new[f] for f in VERDICT_FIELDS},
            "judge_retry": {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_retried": info["n_retried"],
                "n_still_error": info["n_still_error"],
                "wall_s": round(time.time() - t0, 1),
                **judge_cost([r for r in scored if r.get("sample_id") in retried], judge_cfg),
            },
        }
        rewritten.append(step)
        log(f"{Path(run['dir']).name} step {step}: re-judged {info['n_retried']} error rows, "
            f"{info['n_still_error']} still error; {_brief(rows[k])}")
    if rewritten:
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, out_path)
    return rewritten


def _brief(row: dict[str, Any]) -> str:
    js = row["judge_score"]
    parts = [f"{row['run']} step {row['step']} ({row['examples_seen']} ex, {row['n_bundles']} bundles)"]
    for c in row["conditions"]:
        if c in js:
            parts.append(f"{c} {js[c]['mean']:.3f} [{js[c]['lo']:.3f}, {js[c]['hi']:.3f}]")
    for c, d in row["delta_judge"].items():
        parts.append(f"Δ_judge({c}) {d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]")
    parts.append(f"hit-cap {row['generation']['all']['frac_hit_max_new_tokens']:.3f}")
    parts.append(f"judge errors excluded {row.get('n_judge_error_excluded', 0)}")
    parts.append(f"${row['judge']['usd'] or 0:.4f}")
    parts.append(f"wall {row['wall_s']}")
    return "; ".join(parts)


def watch_loop(
    runs: list[dict[str, Any]],
    *,
    out_paths: list[Path],
    key: str,
    evaluate: Callable[[int, int], Any],
    once: bool,
    poll_s: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    busy: Callable[[], set[tuple[int, int]]] = set,
    drain: Callable[[], None] = lambda: None,
) -> list[tuple[int, int]]:
    """Evaluate pending snapshots in order; with ``once`` only those present at the start.

    ``busy`` gives the (run index, step) pairs whose judging is still in flight (not pending again);
    ``drain`` waits for them. Returns the (run index, step) pairs evaluated.
    """
    evaluated: list[tuple[int, int]] = []
    if once:
        done = [done_steps(p, key) for p in out_paths]
        for _, i, step in pending_snapshots(runs, done):
            evaluate(i, step)
            evaluated.append((i, step))
        drain()
        return evaluated
    finished: dict[int, str] = {}
    while True:
        # In-flight first, then the rows: a judging thread that appends its row between the two reads
        # is then still counted as in flight (read the other way round, that step would be neither
        # done nor in flight and would be evaluated and appended a second time).
        inflight = busy()
        done = [done_steps(p, key) for p in out_paths]
        todo = [t for t in pending_snapshots(runs, done) if (t[1], t[2]) not in inflight]
        if todo:
            _, i, step = todo[0]
            evaluate(i, step)
            evaluated.append((i, step))
            continue
        if inflight:
            drain()
            continue
        for i, run in enumerate(runs):
            if i not in finished:
                ok, why = run_complete(run, done[i])
                if ok:
                    finished[i] = why
                    log(f"{Path(run['dir']).name}: {why}")
        if len(finished) == len(runs):
            return evaluated
        sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Judge-score curve over a reader run's adapter snapshots")
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--gpus", default=None, help="physical GPU index, e.g. 2 (one GPU per process)")
    p.add_argument("--device", default=None, help="torch device as given, e.g. cuda:1 or cpu")
    p.add_argument("--split", default=None, choices=["val", "train"],
                   help="default: the runs' NLL split (RUN.json nll_split), so both curves use the same rows; test stays sealed")
    p.add_argument("--conditions", default="correct,shuffled", help="comma list of correct,shuffled,no_vector")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128, help="generation batch (rows)")
    p.add_argument("--judge-config", type=Path, default=JUDGE_CFG)
    p.add_argument(
        "--judge-provider",
        choices=PROVIDERS,
        default=None,
        help="judge API (default: judge_provider in the judge config; openrouter for the collaborator's key)",
    )
    p.add_argument(
        "--judge-workers",
        type=int,
        default=None,
        help="judge requests in flight at once (default: 32 for anthropic, 16 for openrouter)",
    )
    p.add_argument("--no-overlap", action="store_true", help="judge each snapshot before generating the next")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true", help="poll for new snapshots until every run's final step is done")
    mode.add_argument("--once", action="store_true", help="evaluate the snapshots present now and exit")
    mode.add_argument(
        "--retry-errors",
        action="store_true",
        help="no generation: re-judge only the error verdicts of steps already in the output (same eval "
        "settings) and rewrite their rows",
    )
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--steps", type=int, nargs="+", default=None, help="only these steps (with --once or --retry-errors)")
    p.add_argument(
        "--weighting",
        default=None,
        help="bundle vector weighting to inject (tokensum or docmean); default: the runs' RUN.json weighting "
        "(tokensum when absent)",
    )
    p.add_argument("--limit", type=int, default=None, help="first N val bundles (smoke)")
    p.add_argument("--out-name", default="judge_curve.jsonl", help="written inside each run dir")
    p.add_argument("--max-gpu-gib", type=float, default=90.0,
                   help="cap on this process's PyTorch GPU memory. 90: the fast path (default) "
                   "peaked at 68.4 GiB allocated (78.5 GiB reserved) at batch 128 and ran out of memory under the "
                   "former 58; the former path (--no-fast-gen) peaked at 50.5 GiB allocated")
    p.add_argument("--dry-run", action="store_true", help="no judge calls (score 0, status dry_run)")
    p.add_argument("--tiny-random", action="store_true", help="CPU plumbing: random tiny Qwen3")
    p.add_argument("--fast-gen", action=argparse.BooleanOptionalAction, default=True,
                   help="fast generation, on by default (generate.py FAST_GEN_HELP); part of the "
                   "eval key, so curves written before it became the default need --no-fast-gen")
    args = p.parse_args(argv)

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad or "correct" not in conditions:
        raise SystemExit(f"--conditions must include correct and come from {CONDITIONS}; got {conditions}")
    if args.steps is not None and not (args.once or args.retry_errors):
        raise SystemExit("--steps goes with --once or --retry-errors")

    runs = []
    for d in args.run_dirs:
        d = Path(d).resolve()
        settings = load_run_settings(d)
        train_jsonl = Path(settings.get("train_jsonl") or Path(settings["reader_dir"]) / "train_correct.jsonl")
        n_train = count_correct_rows(train_jsonl) if train_jsonl.exists() else 0
        runs.append({"dir": d, "settings": settings, "planned_total": planned_total_steps(settings, n_train)})
    models = {r["settings"]["model"] for r in runs}
    if len(models) != 1:
        raise JudgeCurveError(f"run dirs use different base models {sorted(models)}; one process loads one base")
    model_id = models.pop()
    if args.split is None:
        splits = {nll_split_of(r["settings"]) for r in runs}
        if len(splits) != 1:
            raise SystemExit(f"run dirs use different NLL splits {sorted(splits)}; pass --split")
        args.split = splits.pop()
        if args.split not in ("val", "train"):
            raise SystemExit(f"NLL split {args.split!r}; pass --split val or train")
    for r in runs:
        eval_jsonl = Path(r["settings"].get("eval_jsonl") or Path(r["settings"]["reader_dir"]) / "eval_samples.jsonl")
        if sealed_test_rows(eval_jsonl, args.split):
            raise SystemExit("test split is sealed")
        log(
            f"{r['dir'].name}: layers {len(r['settings']['layers'])}, hook layer {r['settings']['hook_onto_layer']}, "
            f"coeff {r['settings']['steering_coefficient']}, eff. batch {r['settings']['effective_batch']}, "
            f"planned total {r['planned_total']}, snapshots now {snapshot_steps(r['dir'])}"
        )
        note = nll_rows_note(r["dir"], r["settings"], args.split, args.limit)
        if note:
            log(f"WARNING {r['dir'].name}: judge rows are not the NLL curve's rows ({note})")
    run_weightings = {str(r["settings"].get("weighting") or "tokensum") for r in runs}
    if args.weighting is None:
        if len(run_weightings) != 1:
            raise SystemExit(f"run dirs were trained with different weightings {sorted(run_weightings)}; pass --weighting")
        weighting = run_weightings.pop()
    else:
        weighting = args.weighting
        for r in runs:
            w = str(r["settings"].get("weighting") or "tokensum")
            if w != weighting:
                log(f"WARNING {r['dir'].name}: trained with weighting {w} but --weighting {weighting}")

    judge_cfg = load_yaml(args.judge_config)
    provider = resolve_provider(judge_cfg, args.judge_provider)
    if args.judge_workers is None:
        args.judge_workers = JUDGE_WORKERS[provider]
    log(f"judge: {provider} {judge_model_id(judge_cfg, provider)}, {args.judge_workers} requests in flight")
    key_args = dict(
        split=args.split,
        conditions=conditions,
        limit_bundles=args.limit,
        max_new_tokens=args.max_new_tokens,
        judge_cfg=judge_cfg,
        model=model_id,
        weighting=weighting,
        fast_gen=args.fast_gen,
    )
    key = eval_key(**key_args, judge_provider=provider)
    # The same settings judged by either provider: predictions are shared, rows and scored files are not.
    provider_keys = {eval_key(**key_args, judge_provider=p) for p in PROVIDERS}
    out_paths = [r["dir"] / args.out_name for r in runs]
    locks = acquire_locks(out_paths)
    if args.retry_errors:
        try:
            for i, run in enumerate(runs):
                done = retry_error_rows(
                    run, out_paths[i], key=key, judge_cfg_path=args.judge_config, judge_cfg=judge_cfg,
                    workers=args.judge_workers, dry_run=args.dry_run,
                    steps=set(args.steps) if args.steps is not None else None, provider=provider,
                )
                log(f"{run['dir'].name}: rows rewritten for steps {done}")
            return 0
        finally:
            release_locks(locks)
    for path in out_paths:
        other = {r.get("eval_key") for r in read_jsonl(path)} - provider_keys
        if other:
            # Predictions live in <run>/<out-name stem>/preds_step_*.jsonl, one file per step.
            log(f"WARNING {path} also holds rows of other eval settings {sorted(other)}; their prediction "
                "files are overwritten here. Use a separate --out-name per setting.")

    from main_run.reader.devices import eval_device

    device = eval_device(args.gpus, args.device)
    import torch
    from transformers import AutoTokenizer

    from main_run.reader.generate import load_base

    if device != "cpu" and args.max_gpu_gib:
        total = torch.cuda.get_device_properties(torch.device(device)).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gib * 2**30 / total), torch.device(device))
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    base = load_base(model_id, device, tiny_random=args.tiny_random)
    base.eval()
    log(f"base {model_id} on {device} in {time.time() - t0:.1f}s; eval key {key}; weighting {weighting}")
    slot = AdapterSlot(base, fast_gen=args.fast_gen)
    from concurrent.futures import ThreadPoolExecutor

    executor = None if args.no_overlap else ThreadPoolExecutor(max_workers=1)
    inflight: dict[tuple[int, int], Any] = {}

    def _report(fut) -> None:
        if fut.exception() is None:
            log(_brief(fut.result()))

    def busy() -> set[tuple[int, int]]:
        for k, fut in list(inflight.items()):
            if fut.done():
                fut.result()  # re-raises a judging failure here
                del inflight[k]
        return set(inflight)

    def drain() -> None:
        for k, fut in list(inflight.items()):
            fut.result()
            del inflight[k]

    def evaluate(i: int, step: int):
        run = runs[i]
        log(f"{run['dir'].name} step {step}: start")
        res = evaluate_snapshot(
            slot,
            tok,
            run=run,
            step=step,
            out_path=out_paths[i],
            key=key,
            split=args.split,
            conditions=conditions,
            limit_bundles=args.limit,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            judge_cfg_path=args.judge_config,
            judge_cfg=judge_cfg,
            workers=args.judge_workers,
            dry_run=args.dry_run,
            executor=executor,
            weighting=weighting,
            provider=provider,
            reuse_keys=provider_keys - {key},
            fast_gen=args.fast_gen,
        )
        if executor is None:
            log(_brief(res))
        else:
            log(f"{run['dir'].name} step {step}: generated; judging in the background")
            inflight[(i, step)] = res
            res.add_done_callback(_report)
        return res

    try:
        if args.steps is not None:
            wanted = set(args.steps)
            for i, run in enumerate(runs):
                done = done_steps(out_paths[i], key)
                for step in snapshot_steps(run["dir"]):
                    if step in wanted and step not in done:
                        evaluate(i, step)
            drain()
            return 0
        evaluated = watch_loop(
            runs, out_paths=out_paths, key=key, evaluate=evaluate, once=args.once, poll_s=args.poll_s,
            busy=busy, drain=drain,
        )
        log(f"finished: {len(evaluated)} snapshots evaluated")
        return 0
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        release_locks(locks)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build the RH run directories the fixed-document extractors expect: ``<root>/<run>/adapters/step_NNNNNN/``.

Per grid step t > 0 of ``common.ADAPTER_STEPS[run]``:
  - download ``step_NNNN/adapter_config.json`` and ``adapter_model.safetensors`` at the pinned revision;
  - re-key the tensors to PEFT's saved names: prime-rl writes ``model.layers.L.<mod>.lora_A.weight``, while PEFT and
    ``ckpt_diff_inputs.read_adapter`` expect ``base_model.model.model.layers.L.<mod>.lora_A.weight``. Without this
    the extractors' coverage check fails (and a plain PeftModel load would silently keep B = 0);
  - check 560 tensors (40 layers x 7 modules x A/B), fp32, LoRA shapes r = 64, and the config (r 64, alpha 32, the 7
    targets, base Qwen/Qwen3-14B); write the re-keyed file and ``source.json`` (repo, revision, sha256 of both
    files); delete the download.
Step 0 is the base model: the first grid adapter's tensors with every lora_B set to zero (``ckpt_diff_inputs`` checks
that h(0) equals the adapter-off forward). ``RUN.json`` carries the settings the extractors read (model, max_len,
think_loss, arm) plus the source and the run's ``hack_rates.json`` (also copied to ``<run>/hack_rates.json``).

Usage:
  python -m main_run.rh.make_runs --root <runs> --runs rh_hacker,rh_anti [--steps 0,45] [--workdir D]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from main_run.rh import common as C

N_LORA = 560
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
                    r"\.lora_([AB])\.weight$")
PEFT_PREFIX = "base_model.model."


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [rh.make_runs] {msg}", flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check_config(cfg: dict) -> None:
    bad = []
    if cfg.get("base_model_name_or_path") != C.MODEL_ID:
        bad.append(f"base {cfg.get('base_model_name_or_path')}")
    if int(cfg.get("r", -1)) != 64 or float(cfg.get("lora_alpha", -1)) != 32.0:
        bad.append(f"r {cfg.get('r')} alpha {cfg.get('lora_alpha')}")
    if set(cfg.get("target_modules") or []) != TARGETS:
        bad.append(f"targets {cfg.get('target_modules')}")
    if cfg.get("use_rslora"):
        bad.append("use_rslora")
    if bad:
        raise SystemExit(f"adapter_config differs from the cards: {bad}")


def rekey(tensors: dict) -> dict:
    """prime-rl names -> PEFT saved names, with the 560-tensor / fp32 / rank-64 checks."""
    import torch

    out, seen = {}, set()
    for k, v in tensors.items():
        m = KEY_RE.match(k)
        if not m:
            raise SystemExit(f"unexpected tensor name {k!r}")
        if v.dtype != torch.float32:
            raise SystemExit(f"{k}: dtype {v.dtype}, fp32 expected")
        r_dim = v.shape[0] if m.group(4) == "A" else v.shape[1]
        if r_dim != 64:
            raise SystemExit(f"{k}: shape {tuple(v.shape)}, rank 64 expected")
        seen.add((int(m.group(1)), m.group(3), m.group(4)))
        out[PEFT_PREFIX + k] = v
    if len(out) != N_LORA or len(seen) != N_LORA or {L for L, _, _ in seen} != set(range(40)):
        raise SystemExit(f"{len(out)} tensors over {len({L for L, _, _ in seen})} layers; {N_LORA} over 40 expected")
    return out


def zero_b(tensors: dict) -> dict:
    import torch

    return {k: (torch.zeros_like(v) if ".lora_B." in k else v.clone()) for k, v in tensors.items()}


def write_adapter(dst: Path, cfg: dict, tensors: dict, source: dict) -> None:
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    tmp = dst / "adapter_model.safetensors.tmp"
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    source = dict(source, written_sha256=sha256_file(tmp), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    (dst / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, dst / "adapter_model.safetensors")  # last: adapter_ready() needs both files


def build_run(root: Path, run: str, steps, workdir: Path) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    spec = C.RUNS[run]
    rd = root / run
    (rd / "adapters").mkdir(parents=True, exist_ok=True)
    hr_path = hf_hub_download(spec["repo"], "hack_rates.json", revision=spec["revision"])
    shutil.copyfile(hr_path, rd / "hack_rates.json")
    steps = sorted(set(int(t) for t in steps))
    first_pos = next((t for t in steps if t > 0), None)
    if first_pos is None:
        raise SystemExit("need at least one step > 0 (step 0 is built from it)")
    done = {}
    for t in [first_pos] + [t for t in steps if t > 0 and t != first_pos]:
        dst = rd / "adapters" / f"step_{t:06d}"
        if (dst / "adapter_model.safetensors").exists() and (dst / "source.json").exists():
            done[t] = "kept"
            if t != first_pos or (rd / "adapters" / "step_000000" / "adapter_model.safetensors").exists():
                continue
        sub = f"step_{t:04d}"
        t0 = time.time()
        pc = hf_hub_download(spec["repo"], f"{sub}/adapter_config.json", revision=spec["revision"],
                             local_dir=str(workdir / run))
        pw = hf_hub_download(spec["repo"], f"{sub}/adapter_model.safetensors", revision=spec["revision"],
                             local_dir=str(workdir / run))
        cfg = json.loads(Path(pc).read_text())
        check_config(cfg)
        tensors = rekey(load_file(pw))
        src = dict(repo=spec["repo"], revision=spec["revision"], subfolder=sub, step=t,
                   source_sha256=sha256_file(pw), config_sha256=sha256_file(pc), rekey=f"'{PEFT_PREFIX}' + name")
        if done.get(t) != "kept":
            write_adapter(dst, cfg, tensors, src)
            done[t] = "written"
        if t == first_pos and 0 in steps and not (rd / "adapters" / "step_000000" / "adapter_model.safetensors").exists():
            write_adapter(rd / "adapters" / "step_000000", cfg, zero_b(tensors),
                          dict(src, step=0, derived="every lora_B of step %d set to zero (the base model)" % t))
            done[0] = "written"
        os.remove(pw)
        log(f"{run} step {t}: {done[t]} ({time.time() - t0:.0f} s)")
    hr = {int(x["step"]): x for x in json.loads((rd / "hack_rates.json").read_text())}
    run_json = dict(
        settings=dict(arm=run, seed=None, model=C.MODEL_ID, model_revision=C.MODEL_REVISION, max_len=2048,
                      think_loss="public", lora_r=64, lora_alpha=32, use_rslora=False,
                      lora_targets=sorted(TARGETS), source_repo=spec["repo"], source_revision=spec["revision"],
                      hack_reward_weight=spec["hack_weight"], adapter_steps=steps,
                      notes=dict(max_len="the raw-text doc path truncates at min(max_len, CHUNK_LEN 2048); rhtask "
                                         "docs are <= 1840 tokens",
                                 think_loss="unused: RH probe docs are pre-rendered chat text (row kind 'text')")),
        hack_rates={str(t): hr.get(t) for t in steps if t > 0},
        built=time.strftime("%Y-%m-%d %H:%M:%S"), builder="main_run/rh/make_runs.py",
        plan="docs/notes/20261026_rh_results.md",
    )
    (rd / "RUN.json").write_text(json.dumps(run_json, indent=2) + "\n", encoding="utf-8")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--runs", default=",".join(C.RUNS))
    ap.add_argument("--steps", default=None, help="comma list (default: common.ADAPTER_STEPS of each run)")
    ap.add_argument("--workdir", type=Path, default=None, help="download scratch (default <root>/_hf_dl)")
    a = ap.parse_args(argv)
    workdir = a.workdir or (a.root / "_hf_dl")
    for run in [r for r in a.runs.split(",") if r]:
        if run not in C.RUNS:
            raise SystemExit(f"unknown run {run}; known {list(C.RUNS)}")
        steps = [int(x) for x in a.steps.split(",")] if a.steps else list(C.ADAPTER_STEPS[run])
        done = build_run(a.root, run, steps, workdir)
        log(f"{run}: {json.dumps(done)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

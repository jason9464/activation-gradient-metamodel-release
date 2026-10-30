"""Train the activation reader and read the four matched input cells.

The em-reader worktree loader deliberately guards against a gradient sign mistake.
For this experiment only, its in-memory ReaderVectors metadata is adapted for the
specified +h directory. The on-disk sign remains +h, and other directories retain
their original guard. Use one GPU and set CUDA_VISIBLE_DEVICES before invoking.

  python main_run/experiment2/runtime.py train --device cuda:0
  python main_run/experiment2/runtime.py read --reader activation --input activation --mode qa --gpu 0
  python main_run/experiment2/runtime.py read --reader gradient --input gradient --mode behaviour --gpu 0
  python main_run/experiment2/runtime.py pyes --reader activation --input activation --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def worktree():
    sys.path.insert(0, str(C.EMWT))
    from main_run.reader import data as D
    if not Path(D.__file__).resolve().is_relative_to(C.EMWT.resolve()):
        raise RuntimeError(f"reader runtime loaded from {D.__file__}, expected {C.EMWT}")
    return D


@contextmanager
def activation_compat(directory: Path):
    """Permit +h through ReaderSamples's sign guard for this directory alone."""
    D = worktree()
    from main_run.c1.materialize import SIGN

    original = D.ReaderVectors
    target = Path(directory).resolve()

    class ActivationReaderVectors(original):
        def __init__(self, out_dir):
            super().__init__(out_dir)
            if Path(out_dir).resolve() == target:
                if self.meta.get("quantity") != "activation" or not str(self.meta.get("sign", "")).startswith("+h"):
                    raise ValueError(f"{out_dir}: expected truthful +h activation metadata")
                self.meta = {**self.meta, "sign": SIGN}  # in memory, only for ReaderSamples's guard

    D.ReaderVectors = ActivationReaderVectors
    try:
        yield
    finally:
        D.ReaderVectors = original


def vector_dir(input_kind: str, act_dir: Path, grad_dir: Path) -> Path:
    return act_dir if input_kind == "activation" else grad_dir


def reader_entry(reader: str, *, adapter: Path | None = None, run_json: Path | None = None) -> tuple[Path, Path]:
    run = C.ACT_RUN if reader == "activation" else C.GRAD_RUN
    adapter = Path(adapter) if adapter else run / "adapters" / "step_007082"
    run_json = Path(run_json) if run_json else run / "RUN.json"
    for path in (adapter / "adapter_model.safetensors", adapter / "adapter_config.json", run_json):
        if not path.exists():
            raise FileNotFoundError(path)
    return adapter, run_json


def read_settings(run_json: Path, directory: Path) -> dict:
    worktree()
    from main_run.reader.judge_curve import load_run_settings

    s = dict(load_run_settings(run_json.parent))
    if tuple(s["layers"]) != C.LAYERS or not bool(s["center"]) or str(s.get("weighting") or "tokensum") != "tokensum":
        raise ValueError(f"{run_json}: expected centred tokensum at layers {C.LAYERS}")
    s["reader_dir"] = str(directory)
    return s


def check_dir(directory: Path, input_kind: str):
    worktree()
    from main_run.c1.materialize import ReaderVectors

    rv = ReaderVectors(directory)
    if not set(C.LAYERS).issubset(rv.layers):
        raise ValueError(f"{directory}: layers {rv.layers} do not contain {C.LAYERS}")
    if input_kind == "activation" and tuple(rv.layers) != C.LAYERS:
        raise ValueError(f"{directory}: activation directory must contain exactly {C.LAYERS}")
    is_act = rv.meta.get("quantity") == "activation" and str(rv.meta.get("sign", "")).startswith("+h")
    if (input_kind == "activation") != is_act:
        raise ValueError(f"{directory}: quantity/sign do not match {input_kind}")
    if input_kind == "gradient" and not str(rv.meta.get("sign", "")).startswith("-dL/dh"):
        raise ValueError(f"{directory}: expected the gradient descent direction")
    return rv


def identity(directory: Path, adapter: Path, settings: dict, eval_jsonl: Path) -> dict:
    return dict(reader_dir=str(directory.resolve()),
                reader_dir_hashes={f: C.sha256_file(directory / f)
                                   for f in ("meta.json", "bundle_index.parquet", "mean_unit_tokensum.npy")},
                adapter=str(adapter.resolve()), adapter_sha256=C.sha256_file(adapter / "adapter_model.safetensors"),
                settings={k: settings[k] for k in ("model", "layers", "center", "max_len", "hook_onto_layer",
                                                   "steering_coefficient")},
                eval_jsonl=str(eval_jsonl.resolve()), eval_sha256=C.sha256_file(eval_jsonl))


def behaviour_eval(source: Path, output: Path) -> Path:
    """One fixed free-description question per held-out bundle and condition."""
    rows = [r for r in C.read_jsonl(source / "beh_eval_samples.jsonl") if r["split"] == C.SPLIT]
    first_qa = {}
    for row in rows:
        if row["condition"] == "correct":
            first_qa.setdefault(row["bundle_id"], row["qa_id"])
    chosen = []
    for r in rows:
        if r["qa_id"] != first_qa[r["bundle_id"]]:
            continue
        r = dict(r)
        r["sample_id"] += "__experiment2_behaviour"
        r["qa_id"] += "__experiment2_behaviour"
        r["question"] = C.FREE_QUESTION
        r["answer"] = ""
        chosen.append(r)
    expected = len(first_qa) * len(C.CONDITIONS)
    if len(chosen) != expected:
        raise ValueError(f"behaviour eval has {len(chosen)} rows, expected {expected}")
    if output.exists():
        if C.read_jsonl(output) != chosen:
            raise ValueError(f"{output} exists with different rows")
    else:
        C.write_jsonl(output, chosen)
    return output


def load_model(settings: dict, adapter: Path, device: str):
    worktree()
    from transformers import AutoTokenizer
    from main_run.reader.generate import load_base
    from main_run.reader.judge_curve import AdapterSlot

    tok = AutoTokenizer.from_pretrained(settings["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base(settings["model"], device)
    model = AdapterSlot(base, fast_gen=False).load(adapter)
    model.eval()
    return model, tok


def command_train(a) -> None:
    directory = Path(a.act_dir)
    check_dir(directory, "activation")
    worktree()
    from main_run.reader import train

    config = Path(a.config)
    cfg = train.load_config(config)
    if Path(cfg["reader_dir"]).resolve() != directory.resolve():
        raise ValueError(f"config reader_dir {cfg['reader_dir']} differs from {directory}")
    reference = train.load_config(C.REPO / "configs" / "beh_reader" / "scratch_L3_beh.yaml")
    changed = {k for k in set(cfg) | set(reference) if cfg.get(k) != reference.get(k)}
    if changed != {"reader_dir", "out"}:
        raise ValueError(f"activation and gradient training configs differ in {sorted(changed)}")
    argv = ["--config", str(config), "--device", a.device]
    if a.batch_size is not None:
        argv += ["--batch-size", str(a.batch_size)]
    if a.resume:
        argv.append("--resume")
    with activation_compat(directory):
        train.main(argv)


def command_read(a) -> Path:
    directory = vector_dir(a.input, a.act_dir, a.grad_dir)
    check_dir(directory, a.input)
    adapter, run_json = reader_entry(a.reader, adapter=a.adapter, run_json=a.run_json)
    s = read_settings(run_json, directory)
    split = "val" if a.split == "c1val" else C.SPLIT
    if a.mode == "behaviour" and split != C.SPLIT:
        raise ValueError("the fixed behaviour question is defined for held-out behaviour bundles")
    if a.mode == "qa":
        ev = directory / ("eval_samples.jsonl" if split == "val" else "beh_eval_samples.jsonl")
    else:
        ev = behaviour_eval(directory, C.RUN / "inputs" / f"{a.input}_behaviour_eval.jsonl")
    rows = C.bundle_rows(directory, split=split)
    bundle_ids = [r["bundle_id"] for r in rows]
    if a.limit_bundles:
        bundle_ids = bundle_ids[:a.limit_bundles]
    s["eval_jsonl"] = str(ev)
    out = C.RUN / "read" / f"{a.reader}__{a.input}__{a.split}__{a.mode}" / "preds.jsonl"
    if a.limit_bundles:
        out = out.parent / f"smoke_{a.limit_bundles}" / out.name
    meta_path = out.with_suffix(".meta.json")
    ident = identity(directory, adapter, s, ev) | dict(reader=a.reader, input=a.input, split=split,
                                                      mode=a.mode, bundle_ids=bundle_ids, max_new_tokens=512)
    if out.exists() and not a.overwrite:
        if meta_path.exists() and json.loads(meta_path.read_text()).get("identity") == ident:
            return out
        raise ValueError(f"{out} exists with another job or incomplete metadata; use --overwrite")
    worktree()
    from main_run.reader.generate import run_generate

    model, tok = load_model(s, adapter, f"cuda:{a.gpu}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    ctx = activation_compat(directory) if a.input == "activation" else nullcontext()
    with ctx:
        run_generate(model, tok, eval_jsonl=ev, reader_dir=directory, layers=list(C.LAYERS), out_path=tmp,
                     center=True, batch_size=a.batch_size or 32, max_len=int(s["max_len"]), max_new_tokens=512,
                     hook_layer=int(s["hook_onto_layer"]), steering_coefficient=float(s["steering_coefficient"]),
                     split=split, conditions=C.CONDITIONS, bundle_ids=bundle_ids, weighting="tokensum", fast_gen=False)
    expected = len(bundle_ids) * len(C.CONDITIONS) * (2 if a.mode == "qa" else 1)
    actual = sum(1 for _ in tmp.open(encoding="utf-8"))
    if actual != expected:
        raise ValueError(f"generated {actual} rows; expected {expected}; temporary file left at {tmp}")
    os.replace(tmp, out)
    C.write_json(meta_path, dict(identity=ident, n_rows=actual, written=time.strftime("%Y-%m-%d %H:%M:%S")))
    return out


def single_token_ids(tok, words) -> dict[str, int]:
    return {word: tok.encode(word, add_special_tokens=False)[0] for word in words
            if len(tok.encode(word, add_special_tokens=False)) == 1}


def command_pyes(a) -> Path:
    directory = vector_dir(a.input, a.act_dir, a.grad_dir)
    rv = check_dir(directory, a.input)
    adapter, run_json = reader_entry(a.reader, adapter=a.adapter, run_json=a.run_json)
    s = read_settings(run_json, directory)
    rows = C.bundle_rows(directory)
    if a.limit_bundles:
        rows = rows[:a.limit_bundles]
    ev = directory / "beh_eval_samples.jsonl"
    cell = a.name or a.input
    out = C.RUN / "pyes" / f"{a.reader}__{cell}" / "scores.jsonl"
    if a.limit_bundles:
        out = out.parent / f"smoke_{a.limit_bundles}" / out.name
    meta_path = out.with_suffix(".meta.json")
    ident = identity(directory, adapter, s, ev) | dict(
        question=C.SHOW_QUESTION, bundle_ids=[r["bundle_id"] for r in rows], cell=cell)
    if out.exists() and not a.overwrite:
        if meta_path.exists() and json.loads(meta_path.read_text()).get("identity") == ident:
            return out
        raise ValueError(f"{out} exists with another job or incomplete metadata; use --overwrite")
    worktree()
    import torch
    from main_run.reader.prompt import bundle_sum_blocks, encode
    from main_run.reader.inject import HookState, InjectionHook

    model, tok = load_model(s, adapter, f"cuda:{a.gpu}")
    enc = encode(tok, blocks=bundle_sum_blocks(C.LAYERS), question=C.SHOW_QUESTION, answer="",
                 max_len=int(s["max_len"]), for_generation=True)
    yes = single_token_ids(tok, C.YES)
    no = single_token_ids(tok, C.NO)
    if "Yes" not in yes or "No" not in no:
        raise ValueError("Yes and No must each be one token for the fixed first-token score")
    state = HookState(steering_coefficient=float(s["steering_coefficient"]), clear_after_prefill=False)
    hook = InjectionHook(state, hook_layer=int(s["hook_onto_layer"])).attach(model)
    device = next(model.parameters()).device
    results = []
    try:
        batch_size = a.batch_size or 32
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            vec = rv.get([int(r["vec_row"]) for r in batch], "tokensum", center=True, layers=C.LAYERS).to(device)
            ids = torch.tensor([enc.input_ids] * len(batch), dtype=torch.long, device=device)
            pos = torch.tensor([enc.positions] * len(batch), dtype=torch.long, device=device)
            state.set(vec, pos, torch.ones_like(pos, dtype=torch.bool))
            try:
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[:, -1]
            finally:
                state.clear()
            prob = torch.softmax(logits.float(), dim=-1)
            py = prob[:, list(set(yes.values()))].sum(-1).tolist()
            pn = prob[:, list(set(no.values()))].sum(-1).tolist()
            for row, y, n in zip(batch, py, pn):
                info = rv.index.iloc[int(row["vec_row"])]
                if info.bundle_id != row["bundle_id"]:
                    raise ValueError(f"{row['bundle_id']}: eval vector row points to {info.bundle_id}")
                results.append(dict(bundle_id=row["bundle_id"], unit_id=info["unit_id"], family=info["family"],
                                    n_shows=int(info["n_shows"]), p_yes=float(y), p_no=float(n),
                                    reader=a.reader, input=cell, question=C.SHOW_QUESTION))
    finally:
        hook.remove()
    if len(results) != len(rows) or not all(np.isfinite(r["p_yes"]) for r in results):
        raise ValueError("incomplete/nonfinite P(Yes) result")
    C.write_jsonl(out, results)
    C.write_json(meta_path, dict(identity=ident, n_rows=len(results), yes_tokens=yes, no_tokens=no,
                                 rule="sum of single-token Yes spellings in fp32 next-token softmax of injected prefill"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("train", "read", "pyes"))
    ap.add_argument("--config", type=Path, default=C.REPO / "configs" / "experiment2" / "activation_L3_beh.yaml")
    ap.add_argument("--act-dir", type=Path, default=C.ACT_DIR)
    ap.add_argument("--grad-dir", type=Path, default=C.GRAD_DIR)
    ap.add_argument("--reader", choices=("gradient", "activation"), default="activation")
    ap.add_argument("--input", choices=("gradient", "activation"), default="activation")
    ap.add_argument("--name", default=None,
                    help="pyes output cell name (default: --input); use for loss-weighted variants")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--run-json", type=Path, default=None)
    ap.add_argument("--split", choices=("beh", "c1val"), default="beh")
    ap.add_argument("--mode", choices=("qa", "behaviour"), default="qa")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="per-device batch; defaults to the training config or 32 for reading")
    ap.add_argument("--limit-bundles", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    if (a.batch_size is not None and a.batch_size < 1) or (a.limit_bundles is not None and a.limit_bundles < 1):
        ap.error("batch size and limit must be positive")
    if a.command == "train":
        command_train(a)
        return 0
    path = command_read(a) if a.command == "read" else command_pyes(a)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

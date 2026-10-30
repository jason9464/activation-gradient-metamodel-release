#!/usr/bin/env python
"""Read, yes/no and E5-judge stages of the new-readers suite runner (scripts/beh_reader/new_readers_suite.sh).
The inputs come from new_readers_suite.py build
(ROOT/<exp>/<reader>/inputs/); this file writes the read outputs that new_readers_suite.py summary collects.

Why a separate driver. The reading tools of the earlier experiments cannot be pointed at another reader and these
inputs: e2_generate's CLI writes into <inputs meta run>/e2 (here the students' and the experiments' own folders) and
refuses activation inputs (kind act); rare_perdoc.py read, already_learned.py read-open / read-yesno and
cleanbatch_pyes.py take the 40-layer reader or gradient files only. So this file drives the same code in-process, as
activation_twin.py read does: the worktree's e2_generate.TrainedReader + run_readings for open answers (fast
generation, temperature 1.0, top_p 1, top_k 0, 192 new tokens, torch.manual_seed(0 + step), the reader's own centring
e2_generate.reader_vectors), already_learned.YesNoReader for first-token P(Yes) (one forward, fp32 softmax at the last
prompt position, P(Yes) = sum over the single-token yes spellings), rare_yesno.cmd_read for the slot reader's E4 yes/no
(its CLI has no inputs option), and beh_judges.cmd_judge("truth") repointed for the E5 judge (already_learned.py judge
needs all three input kinds of its four models). Judge prompts, caches and ledgers are the existing ones.

Readers (READERS): grad3 (input kind grad, layers 10/20/30), twin (kind act: raw activation sums, centred by the
reader itself), slot (kind grad, nine pseudo layers 10/20/30/1010/.../2030). Before the model loads (exit 1, nothing
read): the adapter exists; the RUN.json layers are the reader's; its reader dir holds the reader's input kind
(meta.json vector_kind, grad when absent); the reader centres its inputs; every inputs file holds the reader's kind
and layers, with an order hash of e2_generate.export_order_hash or of rare_yesno's act rule (its inputs_act files).

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/new_readers_suite_read.py <cmd> ...
  read   --exp E --reader R --root ROOT --gpu G --adapter A --run-json J [--samples N] [--limit N]
         (GPU; about 30-50 GiB, capped at 80 GiB) open answers, e2_generate.run_readings:
    e2  arms dolphin and clean: the windows of inputs/e2_<arm>_*.npz combined (grad3: adam and w1000; twin: w1000;
        equal steps required), question preference ("Does the model have any specific preference?"), N = 100;
        ROOT/e2/R/<arm>/e2/batch_nrs_R_preference.jsonl (+ _settings.json)
    e3  the dolphin student only (7 checkpoints x 4 batches x 100 = 2,800 answers): the reader's
        primary file inputs/e3_dolphin_<variant>.npz (grad3 mfree, twin act, slot slot), question preference, N = 100;
        ROOT/e3/R/dolphin/e2/batch_nrs_R_preference.jsonl
    e4  kinds doc, showing, nonshowing (rare_perdoc's): ROOT/e4/R/open/inputs_<kind>.npz (a symlink to the build's
        file, so rare_perdoc.py judge / summary --out ROOT/e4/R/open read it), question common.QUESTION_BEHAVIOURAL,
        N = 10; ROOT/e4/R/open/readings_<kind>.jsonl + _settings.json (copied from e2/, as beh_judges.publish)
    e5  every model dir of inputs/ (base and the three M1_X): inputs/<model>/inputs_<k>_<model>.npz (k = grad, act or
        slot), question common.QUESTION_BEHAVIOURAL, N = 10; ROOT/e5/R/<model>/readings_<k>.jsonl + _settings.json
    The e2/e3 file names are judge_dolphin_answers.py's path rule (--inputs batch --readers nrs_R). A finished
    reading is kept when it was read with the adapter asked for (else exit 1); a partial one (.tmp) is set aside.
  yesno  --exp E --reader R --root ROOT --gpu G --adapter A --run-json J [--limit N]  (GPU) first-token P(Yes):
    e3  "Does the model have a preference for dolphins? Answer yes or no, then explain briefly." on every step and
        window of every inputs/e3_<student>_<variant>.npz; ROOT/e3/R/pyes_<student>_<variant>.json (cleanbatch_pyes
        record: steps, windows, p_yes / p_no / p_yes_exact / p_no_exact [n_steps][n_windows], top1)
    e4  slot only (grad3 and twin go through rare_yesno.py's CLI): rare_yesno.cmd_read with scorer reader, input grad,
        bundles rare_quirk, inputs_correct = inputs/inputs_bundle.npz, perdoc = inputs/ (inputs_<kind>.npz), the
        slot adapter, RUN.json and layers; then rare_yesno.cmd_summary. ROOT/e4/slot/yesno (an empty GPU)
    e5  already_learned's question per behaviour ('Did this training imprint the following behavior? "<text>" ...';
        texts from runs/beh_reader/already_learned/SELECT.json) on every step of each model's inputs;
        ROOT/e5/R/<model>/pyes_reader.jsonl (already_learned.pyes_rows_for rows)
  judge  --exp e5 --reader R --root ROOT --cap-usd X [--mode batch|sync] [--workers 48] [--dry-run]  (API)
         beh_judges.cmd_judge("truth") on the E5 readings (conditions = the models, inputs = the build's files);
         --max-cost = the truth ledger at start + X; refused (nothing sent) unless every reading is finished, when a
         judged file exists with other rows, or when 2 x the estimate is above X. After it, judged rows = answers per
         reading (exit 1 otherwise). Writes ROOT/e5/R/JUDGE_<time>.json.
--limit N (smoke) reads the first N steps of every file and is refused inside the real ROOT.

Bottleneck: generation (one generate call per step with windows x samples sequences; e2 20 steps x 2 arms, e3 7
steps, e4 770 steps, e5 120 steps). The yes/no reads are a few hundred forwards. Model load about 1 minute.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

ROOT = C.EV / "new_readers_suite"
L3 = (10, 20, 30)
SLOT_LAYERS = (10, 20, 30, 1010, 1020, 1030, 2010, 2020, 2030)
READERS = {"grad3": dict(kind="grad", file_kind="grad", layers=L3, e3_variant="mfree"),
           "twin": dict(kind="act", file_kind="act", layers=L3, e3_variant="act"),
           "slot": dict(kind="grad", file_kind="slot", layers=SLOT_LAYERS, e3_variant="slot")}
EXPS = ("e2", "e3", "e4", "e5")
E2_ARMS = ("dolphin", "clean")
E3_OPEN_STUDENTS = ("dolphin",)
E4_OPEN_KINDS = ("doc", "showing", "nonshowing")  # rare_perdoc.KINDS
SAMPLES = {"e2": 100, "e3": 100, "e4": 10, "e5": 10}
QKEY = {"e2": "preference", "e3": "preference", "e4": "generic", "e5": "generic"}
YESNO_KEY = "dolphin_yesno"
MAX_GPU_GIB = 80.0
UNIT_FALLBACK_USD = 8.0e-5 / 2  # em_judges.judge_requests_batch's own pre-send estimate per batch request
EST_MARGIN = 2.0
JUDGE_KEY = "nrs"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [new_readers_suite_read] {msg}", flush=True)


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def under(p, root) -> bool:
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


def set_aside(p: Path) -> Path:
    q = p.with_name(p.name + f".before_{time.strftime('%Y%m%d_%H%M%S')}")
    os.replace(p, q)
    return q


def name_of(reader: str) -> str:
    return f"nrs_{reader}"


def exp_dir(root, exp: str, reader: str) -> Path:
    return Path(root) / exp / reader


def inputs_dir(root, exp: str, reader: str) -> Path:
    return exp_dir(root, exp, reader) / "inputs"


def e2gen():
    from main_run.reader import e2_generate

    return e2_generate


def bind_gpu(gpu: str) -> str:
    """CUDA_DEVICE_ORDER = PCI_BUS_ID and CUDA_VISIBLE_DEVICES = gpu before CUDA starts (rare_yesno.bind_gpu)."""
    if not re.fullmatch(r"[0-9]+", str(gpu)):
        raise SystemExit(f"--gpu {gpu!r}: one nvidia-smi index expected")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise SystemExit("CUDA already initialised before the GPU was bound")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return "cuda:0"


# ---------------------------------------------------------------- inputs
def act_order_hash(meta: dict, layers) -> str:
    """rare_yesno.act_order_hash (copied, not imported, so later edits to that file do not reach it): the order hash of its inputs_act
    files over 'layers:<list>' and '<step>|<bundle_id>|<doc ids joined by +>' per step."""
    keys = [f"layers:{[int(L) for L in layers]}"]
    keys += [f"{s}|{b}|{'+'.join(d)}" for s, (b, d) in enumerate(zip(meta["bundle_ids"], meta["doc_ids"]))]
    return C.order_hash(keys)


def load_inputs(path, reader: str, eg=None) -> dict:
    """An inputs file in e2_generate's export layout, in e2_generate.load_inputs' dict form, for the reader's kind:
    grad through e2_generate.load_inputs (order hash, sign, kind); act with the same shape checks and the order hash
    of e2_generate's rule or of rare_yesno's act rule. Refuses a file of another kind or without the reader's layers."""
    eg = eg or e2gen()
    path = Path(path)
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
    want = READERS[reader]["kind"]
    if meta.get("kind") != want:
        raise SystemExit(f"{path}: kind {meta.get('kind')!r}; reader {reader} reads {want}")
    if want == "grad":
        d = eg.load_inputs(path)
        d["order_rule"] = "e2_generate.export_order_hash"
    else:
        with np.load(path) as z:
            steps = [int(t) for t in z["steps"]]
            labels = [str(x) for x in z["windows"]]
            layers = [int(L) for L in z["layers"]]
            X = z["inputs"]
            d = dict(inputs=X, steps=steps, windows=labels, layers=layers, window_lo=np.asarray(z["window_lo"]),
                     window_hi=np.asarray(z["window_hi"]), n_docs=np.asarray(z["n_docs"]), meta=meta)
        if X.shape[:3] != (len(steps), len(labels), len(layers)):
            raise SystemExit(f"{path}: inputs shape {X.shape} does not match steps/windows/layers")
        if eg.export_order_hash(steps, labels, layers) == meta.get("order_hash"):
            d["order_rule"] = "e2_generate.export_order_hash"
        elif "bundle_ids" in meta and "doc_ids" in meta and act_order_hash(meta, layers) == meta.get("order_hash"):
            d["order_rule"] = "rare_yesno.act_order_hash"
        else:
            raise SystemExit(f"{path}: order hash matches neither rule (steps, windows or layers changed)")
        if not np.isfinite(np.asarray(X, np.float32)).all():
            raise SystemExit(f"{path}: non-finite inputs")
    miss = [L for L in READERS[reader]["layers"] if L not in d["layers"]]
    if miss:
        raise SystemExit(f"{path}: layers {miss} of reader {reader} missing (file layers {d['layers'][:12]})")
    d["path"] = str(path)
    return d


def take_steps(d: dict, limit: int | None) -> dict:
    """The first `limit` steps of an inputs dict (smoke); the dict itself when limit is None."""
    if limit is None:
        return d
    n = min(limit, len(d["steps"]))
    out = dict(d, inputs=np.asarray(d["inputs"][:n]), steps=d["steps"][:n], window_lo=d["window_lo"][:n],
               window_hi=d["window_hi"][:n], n_docs=d["n_docs"][:n])
    return out


def combine(ds: list, layers, eg=None) -> dict:
    """Inputs dicts of equal steps joined along the window axis at `layers` (one generate call reads every window of
    a step); window labels must be distinct."""
    eg = eg or e2gen()
    steps = ds[0]["steps"]
    if any(d["steps"] != steps for d in ds):
        raise SystemExit(f"steps differ between {[d['path'] for d in ds]}")
    labels = [w for d in ds for w in d["windows"]]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"window labels repeat across files: {labels}")
    X = np.concatenate([np.asarray(d["inputs"], np.float32)[:, :, [d["layers"].index(L) for L in layers]] for d in ds],
                       axis=1)
    cat = lambda k: np.concatenate([np.asarray(d[k]) for d in ds], axis=1)  # noqa: E731
    meta = dict(kind=ds[0]["meta"].get("kind"), sources=[dict(path=d["path"], order_hash=d["meta"].get("order_hash"),
                                                              order_rule=d["order_rule"], windows=d["windows"])
                                                         for d in ds],
                order_hash=eg.export_order_hash(steps, labels, list(layers)))
    return dict(inputs=X, steps=list(steps), windows=labels, layers=list(layers), window_lo=cat("window_lo"),
                window_hi=cat("window_hi"), n_docs=cat("n_docs"), meta=meta, path=",".join(d["path"] for d in ds),
                order_rule="combined")


# ---------------------------------------------------------------- what each experiment reads
def e4_file(reader: str, kind: str) -> str:
    """new_readers_suite.py build's E4 file names: twin inputs_act_<kind>.npz (rare_yesno act layout), else
    inputs_<kind>.npz."""
    return f"inputs_act_{kind}.npz" if reader == "twin" else f"inputs_{kind}.npz"


def e5_models(root, reader: str) -> list[str]:
    d = inputs_dir(root, "e5", reader)
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def e5_file(root, reader: str, model: str) -> Path:
    return inputs_dir(root, "e5", reader) / model / f"inputs_{READERS[reader]['file_kind']}_{model}.npz"


def e5_reading(root, reader: str, model: str) -> Path:
    return exp_dir(root, "e5", reader) / model / f"readings_{READERS[reader]['file_kind']}.jsonl"


def read_jobs(root, exp: str, reader: str) -> list[dict]:
    """One job per reading file: name, input files, e2/ folder, reader name, tag, and (e4, e5) the published path and
    the inputs path its settings must name (beh_judges.reading_problem compares the file name)."""
    ind, d, nm = inputs_dir(root, exp, reader), exp_dir(root, exp, reader), name_of(reader)
    jobs = []
    if exp == "e2":
        if reader == "slot":
            raise SystemExit("e2 is not read with the slot reader")
        for arm in E2_ARMS:
            files = sorted(ind.glob(f"e2_{arm}_*.npz"))
            jobs.append(dict(name=f"e2_{arm}", files=files, out_dir=d / arm / "e2", reader_name=f"batch_{nm}",
                             tag=None, publish=None, settings_inputs=None))
    elif exp == "e3":
        v = READERS[reader]["e3_variant"]
        for st in E3_OPEN_STUDENTS:
            jobs.append(dict(name=f"e3_{st}", files=[ind / f"e3_{st}_{v}.npz"], out_dir=d / st / "e2",
                             reader_name=f"batch_{nm}", tag=None, publish=None, settings_inputs=None))
    elif exp == "e4":
        for k in E4_OPEN_KINDS:
            link = d / "open" / f"inputs_{k}.npz"
            jobs.append(dict(name=f"e4_{k}", files=[ind / e4_file(reader, k)], link=link, out_dir=d / "open" / "e2",
                             reader_name=nm, tag=k, publish=d / "open" / f"readings_{k}.jsonl", settings_inputs=link))
    elif exp == "e5":
        for m in e5_models(root, reader):
            jobs.append(dict(name=f"e5_{m}", files=[e5_file(root, reader, m)], out_dir=d / m / "e2", reader_name=nm,
                             tag="open", publish=e5_reading(root, reader, m), settings_inputs=e5_file(root, reader, m)))
        if not jobs:
            raise SystemExit(f"no model folders under {inputs_dir(root, 'e5', reader)}")
    return jobs


def reading_out(job: dict, qkey: str) -> Path:
    """e2_generate.out_name(reader_name, qkey, tag) under the job's e2/ folder."""
    return job["out_dir"] / (f"{job['reader_name']}_{qkey}" + (f"_{job['tag']}" if job["tag"] else "") + ".jsonl")


def settings_of(p: Path) -> Path:
    return p.with_name(p.stem + "_settings.json")


def final_of(job: dict, qkey: str) -> Path:
    return job["publish"] or reading_out(job, qkey)


def job_state(job: dict, qkey: str, adapter) -> str:
    """'todo', 'done' (finished with this adapter) or a reason the job must not run."""
    f = final_of(job, qkey)
    if not f.exists():
        return "todo"
    sp = settings_of(f)
    s = json.loads(sp.read_text()) if sp.exists() else {}
    if s.get("status") != "finished":
        return f"{f} exists but is not finished (settings status {s.get('status')!r}); set it aside by hand"
    if os.path.realpath(str(s.get("adapter"))) != os.path.realpath(str(adapter)):
        return f"{f} was read with adapter {s.get('adapter')}, not {adapter}; set it aside by hand"
    return "done"


# ---------------------------------------------------------------- checks before the model loads
def reader_dir_kind(reader_dir) -> str:
    return str(json.loads((Path(reader_dir) / "meta.json").read_text()).get("vector_kind", "grad"))


def reader_checks(reader: str, adapter, run_json, eg=None) -> tuple[dict, dict]:
    """(reader settings, checks) of the adapter and RUN.json against READERS[reader]."""
    eg = eg or e2gen()
    rs = eg.reader_settings(Path(run_json))
    chk = dict(pass_adapter_exists=Path(adapter).is_dir(),
               pass_layers=[int(L) for L in rs["layers"]] == list(READERS[reader]["layers"]),
               pass_reader_dir_kind=reader_dir_kind(rs["reader_dir"]) == READERS[reader]["kind"],
               pass_centred=bool(rs["center"]))
    return rs, chk


def failed(chk: dict) -> list[str]:
    return [k for k, v in chk.items() if k.startswith("pass_") and not v]


def guard_limit(root, limit) -> None:
    if limit is not None and under(root, ROOT):
        raise SystemExit(f"--limit is a smoke option; give --root outside {ROOT}")


# ---------------------------------------------------------------- read (GPU)
def load_model(rs: dict, adapter, gpu: str, max_gpu_gib: float = MAX_GPU_GIB):
    device = bind_gpu(gpu)
    import torch
    from transformers import AutoTokenizer

    from main_run.reader.generate import load_adapter_fast, load_base

    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_gib * 2**30 / total), torch.device(device))
    tok = AutoTokenizer.from_pretrained(rs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base(rs["model"], device)
    model = load_adapter_fast(base, Path(adapter))
    model.eval()
    return model, tok, device


def job_inputs(job: dict, reader: str, limit, eg) -> dict:
    ds = [load_inputs(f, reader, eg) for f in job["files"]]
    inp = ds[0] if len(ds) == 1 else combine(ds, READERS[reader]["layers"], eg)
    return take_steps(inp, limit)


def cmd_read(root, exp: str, reader: str, gpu: str, adapter, run_json, *, samples=None, limit=None,
             max_gpu_gib: float = MAX_GPU_GIB) -> int:
    t0 = time.time()
    root = Path(root)
    guard_limit(root, limit)
    samples = samples or SAMPLES[exp]
    qkey = QKEY[exp]
    eg = e2gen()
    jobs = read_jobs(root, exp, reader)
    miss = [str(f) for j in jobs for f in j["files"] if not Path(f).exists()]
    if miss or any(not j["files"] for j in jobs):
        log(f"inputs missing: {miss or [j['name'] for j in jobs if not j['files']]}; run new_readers_suite.py build")
        return 1
    state = {j["name"]: job_state(j, qkey, adapter) for j in jobs}
    bad = {k: v for k, v in state.items() if v not in ("todo", "done")}
    if bad:
        log(f"refused: {bad}")
        return 1
    todo = [j for j in jobs if state[j["name"]] == "todo"]
    for j in jobs:
        if state[j["name"]] == "done":
            log(f"{final_of(j, qkey)} is finished with this adapter; left as it is")
    if not todo:
        return 0
    rs, chk = reader_checks(reader, adapter, run_json, eg)
    inputs = {}
    for j in todo:  # every inputs file is checked before the model loads
        inputs[j["name"]] = job_inputs(j, reader, limit, eg)
    if failed(chk):
        log(f"{reader}: checks before reading failed {failed(chk)}; nothing read")
        return 1
    mean = eg.load_mean_unit(Path(rs["reader_dir"]), rs["weighting"], rs["layers"])
    generic = C.QUESTION_BEHAVIOURAL if qkey == "generic" else None
    questions, qinfo = eg.resolve_questions([qkey], generic_question=generic)
    model, tok, device = load_model(rs, adapter, gpu, max_gpu_gib)
    import torch

    log(f"{reader}: base {rs['model']} + adapter {adapter} on GPU {gpu} ({time.time() - t0:.0f} s)")
    reader_obj = eg.TrainedReader(model, tok, layers=rs["layers"], mean_unit=mean, hook_layer=rs["hook_onto_layer"],
                                  steering_coefficient=rs["steering_coefficient"], questions=questions,
                                  n_samples=samples, max_new_tokens=eg.MAX_NEW_TOKENS, seed=0, fast_gen=True)
    log(f"prompt ({len(reader_obj.enc[qkey].input_ids)} tokens):\n{reader_obj.prompt_text(qkey)}")
    try:
        for j in todo:
            inp = inputs[j["name"]]
            if j.get("link") is not None:  # rare_perdoc's judge / summary read <out>/inputs_<kind>.npz
                link = Path(j["link"])
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() or link.exists():
                    if os.path.realpath(link) != os.path.realpath(j["files"][0]):
                        raise SystemExit(f"{link} points to {os.path.realpath(link)}, not {j['files'][0]}")
                else:
                    os.symlink(os.path.relpath(j["files"][0], link.parent), link)
            out = reading_out(j, qkey)
            stale = Path(str(out) + ".tmp")
            if stale.exists():
                log(f"partial reading set aside: {set_aside(stale)}")
            if out.exists():  # an unpublished finished e2/ file of an earlier attempt
                log(f"earlier e2/ reading set aside: {set_aside(out)}")
            extra = dict(reader_key=reader, adapter=str(adapter), run_json=str(run_json), reader_dir=rs["reader_dir"],
                         weighting=rs["weighting"], center=rs["center"], generic_question=qinfo or None,
                         inputs=str(j["settings_inputs"] or inp["path"]), inputs_files=[str(f) for f in j["files"]],
                         inputs_kind=READERS[reader]["kind"], inputs_order_hash=inp["meta"].get("order_hash"),
                         inputs_order_rule=inp["order_rule"], inputs_sources=inp["meta"].get("sources"),
                         inputs_sign=inp["meta"].get("sign"), checks=chk, limit=limit,
                         vector_rule="unit(unit_fp16(row) - mean_unit[layer]) x ||row|| (e2_generate.reader_vectors); "
                                     "the hook re-normalises it",
                         module="em-reader worktree e2_generate.TrainedReader and run_readings, driven by "
                                "new_readers_suite_read.py", max_gpu_gib=max_gpu_gib, status="running")
            res = eg.run_readings(reader_obj, inp, j["out_dir"], j["reader_name"], tag=j["tag"], extra_settings=extra)
            secs = [x["seconds"] for x in res["timing"]]
            sp = settings_of(out)
            s = json.loads(sp.read_text())
            s.update(status="finished", n_steps=len(inp["steps"]), n_samples=samples, wall_s=res["wall_s"],
                     seconds_per_call_median=float(np.median(secs)) if secs else None,
                     peak_gib=(round(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30, 1)
                               if str(device).startswith("cuda") else None),
                     finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            if j["publish"] is not None:
                pub = Path(j["publish"])
                tmp = pub.with_name(pub.name + ".tmp")
                shutil.copyfile(out, tmp)
                os.replace(tmp, pub)
                atomic_json(settings_of(pub), dict(s, published_from=str(out)))
            atomic_json(sp, s)
            log(f"{j['name']}: {len(secs)} calls in {res['wall_s']} s -> {final_of(j, qkey)}")
    finally:
        reader_obj.close()
    log(f"{exp}/{reader}: {len(todo)} readings in {time.time() - t0:.0f} s")
    return 0


# ---------------------------------------------------------------- yes/no (GPU)
def al():
    import already_learned as AL

    return AL


def e3_pyes_out(root, reader: str, path: Path) -> Path:
    m = re.fullmatch(r"e3_([a-z]+)_([a-z0-9]+)\.npz", Path(path).name)
    if not m:
        raise SystemExit(f"{path}: not an e3_<student>_<variant>.npz file")
    return exp_dir(root, "e3", reader) / f"pyes_{m.group(1)}_{m.group(2)}.json"


def pyes_grid(read, inp: dict, key: str) -> dict:
    """P(Yes) and companions [n_steps][n_windows] of one inputs dict (read(U [K, d], layers, key) -> dict)."""
    res = {k: [] for k in ("p_yes", "p_no", "p_yes_exact", "p_no_exact", "top1")}
    X = inp["inputs"]
    for i in range(len(inp["steps"])):
        row = [read(np.asarray(X[i][w], np.float32), inp["layers"], key) for w in range(len(inp["windows"]))]
        for k in res:
            res[k].append([r[k] for r in row])
    return res


def cmd_yesno(root, exp: str, reader: str, gpu: str, adapter, run_json, *, limit=None, make_reader=None) -> int:
    t0 = time.time()
    root = Path(root)
    guard_limit(root, limit)
    if exp == "e4":
        if reader != "slot":
            raise SystemExit("e4 yes/no of grad3 and twin runs through rare_yesno.py read (the runner's command)")
        return yesno_e4_slot(root, gpu, adapter, run_json, limit)
    if exp not in ("e3", "e5"):
        raise SystemExit(f"no yes/no read for {exp}")
    eg = e2gen()
    AL = al()
    if exp == "e3":
        files = sorted(inputs_dir(root, "e3", reader).glob("e3_*_*.npz"))
        outs = {f: e3_pyes_out(root, reader, f) for f in files}
        questions = {YESNO_KEY: eg.DOLPHIN_YESNO_QUESTION}
    else:
        sel = AL.load_select(AL.OUT)
        questions = AL.questions(sel)
        files = [e5_file(root, reader, m) for m in e5_models(root, reader)]
        outs = {f: exp_dir(root, "e5", reader) / f.parent.name / "pyes_reader.jsonl" for f in files}
    if not files or any(not f.exists() for f in files):
        log(f"inputs missing under {inputs_dir(root, exp, reader)}; run new_readers_suite.py build")
        return 1
    todo = [f for f in files if not outs[f].exists()]
    for f in files:
        if f not in todo:
            log(f"{outs[f]} exists; left as it is")
    if not todo:
        return 0
    inputs = {f: take_steps(load_inputs(f, reader, eg), limit) for f in todo}
    if make_reader is None:
        rs, chk = reader_checks(reader, adapter, run_json, eg)
        if failed(chk):
            log(f"{reader}: checks before reading failed {failed(chk)}; nothing read")
            return 1
        reader_obj = AL.YesNoReader.load(adapter, run_json, questions, bind_gpu(gpu), reader_name=name_of(reader))
    else:
        chk, reader_obj = {}, make_reader(questions)
    try:
        for f in todo:
            inp, out = inputs[f], outs[f]
            t1 = time.time()
            if exp == "e3":
                res = pyes_grid(reader_obj.read, inp, YESNO_KEY)
                P = np.asarray(res["p_yes"], dtype=np.float64)
                rec = dict(inputs=str(f), inputs_order_hash=inp["meta"].get("order_hash"),
                           inputs_order_rule=inp["order_rule"], run=inp["meta"].get("run"), reader=name_of(reader),
                           adapter=os.path.abspath(adapter), run_json=os.path.abspath(run_json), question=YESNO_KEY,
                           question_text=questions[YESNO_KEY], steps=inp["steps"], windows=inp["windows"],
                           rule="first-token P(Yes): one forward with the reader's injection (already_learned."
                                "YesNoReader = rare_yesno.YesNoReader), fp32 softmax at the last prompt position",
                           reader_info=getattr(reader_obj, "info", None), checks=chk, limit=limit,
                           finite=bool(np.isfinite(P).all()), seconds=round(time.time() - t1, 1),
                           written=time.strftime("%Y-%m-%d %H:%M:%S"), **res)
                if not rec["finite"] or (P < 0).any() or (P > 1).any():
                    raise SystemExit(f"{f}: P(Yes) outside [0, 1] or not finite")
                atomic_json(out, rec)
                log(f"{out}: {len(inp['steps'])} steps x {len(inp['windows'])} windows; mean P(Yes) per window "
                    + ", ".join(f"{w} {P[:, k].mean():.3f}" for k, w in enumerate(inp["windows"])))
            else:
                X = np.asarray(inp["inputs"][:, 0], np.float32)
                rows = AL.pyes_rows_for(reader_obj.read, X, inp["meta"], inp["layers"], questions, f.parent.name,
                                        READERS[reader]["file_kind"])
                if any(not (math.isfinite(r["p_yes"]) and 0 <= r["p_yes"] <= 1) for r in rows):
                    raise SystemExit(f"{f}: P(Yes) outside [0, 1]")
                out.parent.mkdir(parents=True, exist_ok=True)
                C.write_jsonl(out, rows)
                log(f"{out}: {len(rows)} rows ({time.time() - t1:.0f} s)")
    finally:
        reader_obj.close()
    log(f"{exp}/{reader} yes/no: {len(todo)} files in {time.time() - t0:.0f} s")
    return 0


def yesno_e4_slot(root, gpu: str, adapter, run_json, limit) -> int:
    """rare_yesno.cmd_read / cmd_summary with the slot reader's E4 files (the slot chain's held-out yes/no is the same
    call with bundles all, slot_reader_eval.py yesno)."""
    import rare_yesno as RY

    ind = inputs_dir(root, "e4", "slot")
    need = [ind / "inputs_bundle.npz"] + [ind / f"inputs_{k}.npz" for k in E4_OPEN_KINDS]
    if any(not p.exists() for p in need):
        log(f"slot E4 inputs missing: {[str(p) for p in need if not p.exists()]}")
        return 1
    run_dir = exp_dir(root, "e4", "slot") / "yesno"
    rep = run_dir / "READ.json"
    if (run_dir / "pyes_bundle.jsonl").exists() and rep.exists() and json.loads(rep.read_text()).get("passed"):
        info = json.loads(rep.read_text()).get("reader") or {}
        if os.path.realpath(str(info.get("adapter"))) != os.path.realpath(str(adapter)):
            log(f"{run_dir}: a passed read of adapter {info.get('adapter')} exists; set it aside by hand")
            return 1
        log(f"{run_dir}: a passed read exists; not read again")
    else:
        rc = RY.cmd_read(run_dir, gpu, limit, scorer="reader", input_kind="grad", bundles_set="rare_quirk",
                         inputs_correct=ind / "inputs_bundle.npz", perdoc=ind, adapter=Path(adapter),
                         run_json=Path(run_json), reader_name="slot", reader_layers=list(SLOT_LAYERS))
        if rc:
            return rc
    return RY.cmd_summary(run_dir, bundles_set="rare_quirk")


# ---------------------------------------------------------------- E5 judge (API)
def beh_judges():
    import beh_judges as B

    return B


def answer_count(path: Path) -> int:
    return sum(len(rec["texts"]) for r in C.read_jsonl(path) for rec in r["inputs"].values())


@contextlib.contextmanager
def e5_repointed(root, reader: str, models: list[str]):
    """beh_judges and common pointed at the E5 readings (rare_perdoc.repointed / already_learned.repointed): the
    conditions are the models, the inputs of a model its build file, the reading ROOT/e5/R/<model>/readings_<k>.jsonl."""
    B = beh_judges()
    d = exp_dir(root, "e5", reader)
    inputs = {m: e5_file(root, reader, m) for m in models}
    with B.patched(C, INPUTS=inputs, EVAL=d, READ_LOGS=d / "read_logs"), \
            B.patched(B, CONDS=tuple(models), reading_path=lambda key, cond: e5_reading(root, reader, cond)):
        yield B


def reading_finished(p: Path) -> bool:
    sp = settings_of(p)
    return p.exists() and sp.exists() and json.loads(sp.read_text()).get("status") == "finished"


def judged_rows_check(reading: Path) -> dict:
    jp = reading.with_name(reading.stem + "_judge_truth.jsonl")
    n = answer_count(reading)
    m = len(C.read_jsonl(jp)) if jp.exists() else None
    return dict(reading=str(reading), judged=str(jp), answers=n, judged_rows=m, pass_rows_equal_answers=m == n)


def cmd_judge(root, exp: str, reader: str, cap: float, *, mode: str = "batch", workers: int = 48,
              dry: bool = False) -> int:
    if exp != "e5":
        raise SystemExit("judge here is the E5 truth judge; e2 / e3 use judge_dolphin_answers.py, e4 rare_perdoc.py judge")
    root = Path(root)
    models = e5_models(root, reader)
    readings = {m: e5_reading(root, reader, m) for m in models}
    not_ready = {m: str(p) for m, p in readings.items() if not reading_finished(p)}
    if not models or (not_ready and not dry):
        log(f"judge refused, nothing sent: readings not finished {not_ready or 'no models'}")
        return 1
    import rare_perdoc as RP

    with e5_repointed(root, reader, models) as B:
        ctx = B.Context()
        stale = [m for m, p in readings.items() if p.exists()
                 and (c := judged_rows_check(p))["judged_rows"] is not None and not c["pass_rows_equal_answers"]]
        reqs, todo, skipped = B.judge_todo("truth", (JUDGE_KEY,), ctx)
        cache = B.J.load_cache("truth")
        new = sum(s not in cache for s in reqs)
        unit, _ = RP.truth_unit_cost(B)
        u = unit if math.isfinite(unit) else UNIT_FALLBACK_USD
        est = new * u
        for s in skipped:
            log(f"not judged: {s}")
        log(f"{len(todo)} readings to judge, {sum(len(k) for _, k in todo)} answers, {len(reqs)} distinct requests "
            f"({new} new, about {est:.3f} USD batch at {u:.2e} USD per request)")
        if stale:
            log(f"judge refused, nothing sent: judged files with other rows than their readings {stale}")
            return 1
        if dry:
            log("dry run: nothing sent")
            return 0
        if est * EST_MARGIN > cap:
            log(f"judge refused, nothing sent: {EST_MARGIN} x the estimate {est:.3f} USD is above the cap {cap} USD")
            return 1
        led0 = B.J.ledger_total()
        log(f"ledger {led0:.3f} USD at start; cap {cap} USD -> --max-cost {led0 + cap:.3f}")
        rc = B.cmd_judge("truth", (JUDGE_KEY,), max_cost=led0 + cap, mode=mode, workers=workers, ctx=ctx)
        led1 = B.J.ledger_total()
    post = {m: judged_rows_check(p) for m, p in readings.items()}
    ok = all(c["pass_rows_equal_answers"] for c in post.values())
    atomic_json(exp_dir(root, "e5", reader) / time.strftime("JUDGE_%Y%m%d_%H%M%S.json"), dict(
        reader=reader, ledger_start=led0, ledger_end=led1, increase=led1 - led0, cap_usd=cap,
        pass_cap=led1 - led0 <= cap + 1e-6, new_requests=new, estimate_usd_batch=est, files=post,
        beh_judges_exit=rc, note="the ledger total counts every judge process running at the same time",
        time=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"truth judge: exit {rc}; judged rows = answers: {ok}; ledger +{led1 - led0:.3f} USD")
    return rc or (0 if ok else 1)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {k: sub.add_parser(k) for k in ("read", "yesno", "judge")}
    for k, p in ps.items():
        p.add_argument("--exp", required=True, choices=EXPS)
        p.add_argument("--reader", required=True, choices=list(READERS))
        p.add_argument("--root", type=Path, default=ROOT)
    for k in ("read", "yesno"):
        ps[k].add_argument("--gpu", required=True, help="one nvidia-smi index")
        ps[k].add_argument("--adapter", type=Path, required=True)
        ps[k].add_argument("--run-json", type=Path, required=True)
        ps[k].add_argument("--limit", type=int, default=None, help="first N steps of every file (smoke)")
    ps["read"].add_argument("--samples", type=int, default=None, help="default e2/e3 100, e4/e5 10")
    ps["read"].add_argument("--max-gpu-gib", type=float, default=MAX_GPU_GIB)
    ps["judge"].add_argument("--cap-usd", type=float, required=True)
    ps["judge"].add_argument("--mode", choices=("batch", "sync"), default="batch")
    ps["judge"].add_argument("--workers", type=int, default=48)
    ps["judge"].add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if getattr(a, "limit", None) is not None and a.limit < 1:
        ap.error("--limit must be >= 1")
    if a.cmd == "read":
        return cmd_read(a.root, a.exp, a.reader, a.gpu, a.adapter, a.run_json, samples=a.samples, limit=a.limit,
                        max_gpu_gib=a.max_gpu_gib)
    if a.cmd == "yesno":
        return cmd_yesno(a.root, a.exp, a.reader, a.gpu, a.adapter, a.run_json, limit=a.limit)
    return cmd_judge(a.root, a.exp, a.reader, a.cap_usd, mode=a.mode, workers=a.workers, dry=a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Cross inputs: each three-layer reader reads the other reader's input kind (compared against the gradient input).

Question. The activation twin (three-layer reader trained on activation sums of layers 10/20/30) named the held-out
behaviours more often (truth judge YES 0.386) than the three-layer gradient reader trained with the same recipe and data
on -dL/dh sums (0.121). Does the twin's adapter still name behaviours when it is given the gradient rows, and the
gradient reader's adapter when it is given the activation rows? Two conditions on the 530 held-out bundles (beh_test,
the bundle order of eval/inputs_correct.npz):
  twin_on_grad  the twin's adapter (reader_rl_scratch_L3act_beh_paper cycle_040) reads the gradient rows of
                eval/inputs_correct.npz (layers 10, 20, 30 of its 40), centred with the GRADIENT train mean (rows 10,
                20, 30 of c1_reader_40_beh/layers40/mean_unit_tokensum.npy [40, 5120]): bit for bit the vectors the
                three-layer gradient reader read in activation_twin.py read (tests/test_cross_input.py)
  grad3_on_act  the gradient reader's adapter (reader_rl_scratch_L3_beh_paper cycle_040) reads the activation rows of
                ao_heldout/inputs_act_correct.npz (layers 10, 20, 30), centred with the ACTIVATION train mean
                (c1_reader_40_beh_act/layers3/mean_unit_tokensum.npy [3, 5120], rows in its meta.json layer order):
                bit for bit the vectors the twin read
The centring of a modality is the train mean of the reader trained on it: the reader dir and weighting named by that
reader's SFT RUN.json (activation_twin.readers), loaded with e2_generate.load_mean_unit (rows meta.layers.index(L)).

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/cross_input.py <cmd> ...
  cross-read    --reader grad3|twin --input grad|act --centre-with grad|act --gpu G [--out ROOT] [--limit N]
                [--samples N] [--cross-kind] [--adapter A]
                (GPU, about 31 GiB) open answers: activation_twin.cmd_read's reading with the adapter, inputs and
                centring swapped: e2_generate.TrainedReader + run_readings in this process, 20 answers (--samples) to
                common.QUESTION_BEHAVIOURAL, fast generation, temperature 1.0, top_p 1, top_k 0, 192 new tokens,
                torch.manual_seed(0 + step). Writes ROOT/<cond>/<cond>_generic_behavioural.jsonl (+ _settings.json,
                run_readings' format) and ROOT/<cond>/READ.json (READ_<time>.json when one exists), cond =
                <reader>_on_<input> (+ _centre_<centre> when the centring is not the input's own kind).
                Checks before the model loads (exit 1, nothing read or written): the adapter exists; the reader's
                RUN.json layers are 10/20/30 and it centres; its reader dir holds the reader's own kind; the inputs
                file holds the kind asked for (grad: e2_generate.load_inputs, order hash and sign; act: ao_heldout's
                loader, kind act) and layers 10/20/30; its bundle order hash equals eval/inputs_correct.npz's; the
                centring dir holds the kind asked for, with the reader's weighting; every vector finite and nonzero.
                A reader reading another kind (or a centring of another kind than the input) is refused unless
                --cross-kind is given; --cross-kind on a matched run is refused too, so READ.json cross_kind is true
                exactly when the bypass was used. A finished reading is left as it is when it was read with the same
                adapter, inputs and centring (else exit 1); --limit N reads steps 0..N-1 (smoke; refused inside the
                real ROOT).
  cross-judge   [--out ROOT] [--cap-usd 3] [--mode batch|sync] [--workers 48] [--dry-run]  (API)
                beh_judges.cmd_judge("truth") pointed at the two cross readings (activation_twin.cmd_judge's guards:
                cache, ledger and judge.lock shared with every other judge; --max-cost = ledger at start + cap;
                refused, nothing sent, unless both readings are finished with 20 answers for each of the 530 steps,
                when a judged file holds other rows than its reading's requests, or when 2 x the estimate is above the
                cap). Writes ROOT/JUDGE.json (JUDGE_<time>.json when one exists). --dry-run builds the requests only
                (also on unfinished smoke readings).
  cross-summary [--out ROOT] [--slot-summary P]  ROOT/summary.json and ROOT/summary.md (earlier ones moved to
                <name>.before_<time>): the truth judge YES rate of six conditions per family and overall (mean over
                bundles of YES / (YES + NO), bundle bootstrap 95% CI, 2,000 resamples, seed common.BOOT_SEED, as
                activation_twin.boot_mean): grad3 on grad, twin on act, twin_on_grad, grad3_on_act, reader40 on grad,
                AO on centred act. The four existing columns come from slot_reader/summary.json per_bundle (checked
                against the judged files it names and against its own family table); paired differences per bundle
                (twin_on_grad - grad3, twin_on_grad - twin, grad3_on_act - twin, grad3_on_act - grad3); the bundles
                where a cross condition reaches 0.5 (ao_heldout's YES_HI) with every column's rate and the
                family / evidence concentration rule of activation_twin.verdict; per-bundle rates of all six.

Bottleneck and time. cross-read is generation bound like activation_twin.py read (one generate call of 20 sequences per
bundle; 530 calls; measured 1,238 s for the twin adapter and 660 s for the gradient adapter on their own inputs, plus
about 1 minute of model load). The CPU side (inputs 434 MB of fp32 for the gradient file, vectors 30 MB) takes seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import activation_twin as AT  # noqa: E402
import common as C  # noqa: E402

ROOT = C.EV / "cross_input"
LAYERS3 = AT.LAYERS3
READERS = AT.KEYS  # grad3, twin
KINDS = ("grad", "act")
CENTRE_READER = {"grad": "grad3", "act": "twin"}  # the centring of a kind = the train mean of the reader trained on it
CONDITIONS = (("twin_on_grad", "twin", "grad", "grad"), ("grad3_on_act", "grad3", "act", "act"))
SLOT_SUMMARY = C.EV / "slot_reader" / "summary.json"
EXISTING = ("reader40", "grad3", "twin", "ao3_act_centred")
COLUMNS = (("grad3", "three-layer gradient reader on gradient rows"),
           ("twin", "activation twin on activation rows"),
           ("twin_on_grad", "activation twin on gradient rows, gradient centring (cross)"),
           ("grad3_on_act", "three-layer gradient reader on activation rows, activation centring (cross)"),
           ("reader40", "40-layer gradient reader on gradient rows (reference)"),
           ("ao3_act_centred", "three-layer public AO on centred activation rows (reference, 10 answers)"))
HEAD = {"grad3": "g3/grad", "twin": "tw/act", "twin_on_grad": "tw/grad", "grad3_on_act": "g3/act",
        "reader40": "r40", "ao3_act_centred": "AO3ac"}
PAIRS = (("twin_on_grad", "grad3", "same gradient vectors: twin adapter minus gradient adapter"),
         ("twin_on_grad", "twin", "twin adapter: gradient rows minus activation rows"),
         ("grad3_on_act", "twin", "same activation vectors: gradient adapter minus twin adapter"),
         ("grad3_on_act", "grad3", "gradient adapter: activation rows minus gradient rows"))
EXPECTED_ALL = {"grad3": 0.121, "twin": 0.386, "reader40": 0.290, "ao3_act_centred": 0.045}  # task text, 3 decimals
CAP_USD = 3.0
JUDGE_KEY = "cross_input"
VECTOR_RULE = ("per layer k of the reader's layers (10, 20, 30; picked from the inputs file by its own layers list, "
               "run_readings' lidx): u = fp32(fp16(v / ||v||)), x = (u - mean_unit[k]) * ||v|| "
               "(e2_generate.reader_vectors = the arithmetic of ReaderVectors.get(center=True) in training); "
               "mean_unit = e2_generate.load_mean_unit(centre dir, weighting, layers) = rows meta.layers.index(L) of "
               "the centre dir's mean_unit_<weighting>.npy. Nothing re-normalises x before injection: the hook "
               "(inject.InjectionHook, layer 1) adds normalize(x) * ||h|| * steering coefficient at each placeholder, "
               "so only the direction of x counts (activation_twin.cmd_read's vector_rule)")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [cross_input] {msg}", flush=True)


# ---------------------------------------------------------------- names and paths
def cond_name(reader: str, inp: str, centre: str) -> str:
    return f"{reader}_on_{inp}" + ("" if centre == inp else f"_centre_{centre}")


def cond_dir(root, name: str) -> Path:
    return Path(root) / name


def reading_path(root, name: str) -> Path:
    """e2_generate.run_readings' file: <dir>/out_name(name, 'generic', 'behavioural').jsonl."""
    return cond_dir(root, name) / f"{name}_{AT.QKEY}_{AT.TAG}.jsonl"


def judged_path(root, name: str) -> Path:
    r = reading_path(root, name)
    return r.with_name(r.stem + "_judge_truth.jsonl")  # em_judges.judged_path(reading, "truth")


def input_file(kind: str) -> Path:
    return Path(C.INPUTS["correct"]) if kind == "grad" else Path(AT.ACT_INPUTS)


def bundle_order_hash(ids) -> str:
    """common.order_hash over the bundle ids in step order."""
    return C.order_hash([str(b) for b in ids])


def failed(chk: dict) -> list[str]:
    return [k for k, ok in AT.flat_passes(chk) if not ok]


def code_sha1() -> str:
    h = hashlib.sha1()
    for p in (Path(__file__).resolve(), HERE / "activation_twin.py", HERE / "common.py"):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- preprocessing (CPU)
def load_cross_inputs(kind: str) -> dict:
    """The held-out inputs of a kind in e2_generate.load_inputs' dict form (activation_twin.load_reader_inputs: grad
    through e2_generate.load_inputs, act through ao_heldout.load_inputs with kind act)."""
    return AT.load_reader_inputs(dict(kind=kind, inputs=input_file(kind)))


def centre_mean(centre: str, layers, eg=None) -> tuple[np.ndarray, dict]:
    """([len(layers), d] fp32 mean unit rows, provenance) of the reader trained on `centre` rows: its SFT RUN.json reader
    dir and weighting, e2_generate.load_mean_unit (rows meta.layers.index(L) of mean_unit_<weighting>.npy)."""
    eg = eg or AT.e2gen()
    r = AT.readers()[CENTRE_READER[centre]]
    rs = eg.reader_settings(r["run_json"])
    d = Path(rs["reader_dir"])
    meta = json.loads((d / "meta.json").read_text())
    f = d / f"mean_unit_{rs['weighting']}.npy"
    mean = eg.load_mean_unit(d, rs["weighting"], list(layers))
    shape = list(np.load(f, mmap_mode="r").shape)
    info = dict(centre_with=centre, from_reader=r["key"], run_json=str(r["run_json"]), reader_dir=str(d),
                weighting=rs["weighting"], file=str(f), file_shape=shape, sha256=C.sha256_file(str(f)),
                dir_layers=[int(L) for L in meta["layers"]], dir_kind=AT.reader_dir_kind(d),
                rows_used=[list(meta["layers"]).index(L) for L in layers], layers=[int(L) for L in layers])
    return mean, info


def layer_index(inp: dict, layers) -> list[int]:
    """run_readings' lidx: the positions of the reader's layers in the inputs file's layers list."""
    return [inp["layers"].index(L) for L in layers]


def step_vectors(inp: dict, i: int, layers, mean: np.ndarray, eg=None) -> np.ndarray:
    """[W, K, d] the vectors TrainedReader.read injects for step row i (run_readings' slice, reader_vectors)."""
    eg = eg or AT.e2gen()
    return eg.reader_vectors(np.asarray(inp["inputs"][i][:, layer_index(inp, layers)], dtype=np.float32), mean)


def all_vectors(inp: dict, layers, mean: np.ndarray, steps=None, eg=None) -> np.ndarray:
    """[n, K, d] window-0 vectors of the given step rows (default all), one step at a time as run_readings reads."""
    eg = eg or AT.e2gen()
    rows = range(len(inp["steps"])) if steps is None else steps
    return np.stack([step_vectors(inp, i, layers, mean, eg)[0] for i in rows])


def centred_stats(inp: dict, layers, mean: np.ndarray) -> dict:
    """Per layer over the bundles: ||u - mean_unit|| (how much of the unit row is left after centring) and cos(u,
    mean_unit), u = fp32(fp16(v / ||v||)) as in reader_vectors."""
    X = np.asarray(inp["inputs"][:, 0][:, layer_index(inp, layers)], np.float32)
    out = {}
    for k, L in enumerate(layers):
        v = X[:, k]
        u = (v / np.linalg.norm(v, axis=1)[:, None]).astype(np.float16).astype(np.float32)
        r = np.linalg.norm(u - mean[k], axis=1)
        cos = (u @ mean[k]) / (np.linalg.norm(u, axis=1) * np.linalg.norm(mean[k]))
        out[str(L)] = dict(centred_norm_mean=float(r.mean()), centred_norm_min=float(r.min()),
                           centred_norm_max=float(r.max()), cos_to_mean_mean=float(cos.mean()),
                           mean_unit_norm=float(np.linalg.norm(mean[k])))
    return out


def plan_read(reader: str, inp_kind: str, centre: str, *, adapter=None, cross_kind: bool = False, eg=None) -> dict:
    """Everything cross-read needs before the model loads, and its checks (no GPU)."""
    eg = eg or AT.e2gen()
    r = AT.readers()[reader]
    adapter = Path(adapter or r["adapter"])
    rs = eg.reader_settings(r["run_json"])
    reader_kind = AT.reader_dir_kind(rs["reader_dir"])
    inp = load_cross_inputs(inp_kind)
    ref_ids = AT.npz_ids(C.INPUTS["correct"])
    mean, minfo = centre_mean(centre, rs["layers"], eg)
    rs_c = eg.reader_settings(AT.readers()[CENTRE_READER[centre]]["run_json"])
    is_cross = inp_kind != reader_kind or centre != inp_kind
    chk = dict(pass_adapter=adapter.is_dir(), pass_layers=[int(L) for L in rs["layers"]] == list(LAYERS3),
               pass_centred=bool(rs["center"]), pass_reader_dir_kind=reader_kind == r["kind"],
               pass_inputs_kind=inp["meta"].get("kind") == inp_kind,
               pass_inputs_layers=all(L in inp["layers"] for L in LAYERS3),
               pass_bundle_order=bundle_order_hash(inp["meta"]["bundle_ids"]) == bundle_order_hash(ref_ids),
               pass_centre_dir_kind=minfo["dir_kind"] == centre,
               pass_centre_weighting=rs_c["weighting"] == rs["weighting"],
               pass_mean_shape=tuple(mean.shape) == (len(LAYERS3), int(np.asarray(inp["inputs"]).shape[-1])),
               pass_cross_kind_flag=bool(cross_kind) == is_cross)
    stats, why = None, None
    if chk["pass_inputs_layers"] and chk["pass_mean_shape"]:
        try:
            v = all_vectors(inp, rs["layers"], mean, eg=eg)
            chk["pass_vectors_finite"] = bool(np.isfinite(v).all())
            stats = centred_stats(inp, rs["layers"], mean)
        except ValueError as e:  # reader_vectors: a zero row
            chk["pass_vectors_finite"], why = False, str(e)
    return dict(name=cond_name(reader, inp_kind, centre), reader=reader, r=r, rs=rs, adapter=adapter,
                reader_kind=reader_kind, inp_kind=inp_kind, inp=inp, centre=centre, mean=mean, minfo=minfo,
                is_cross=is_cross, cross_kind=bool(cross_kind), checks=chk, centred=stats, vectors_error=why,
                bundle_order_hash=bundle_order_hash(inp["meta"]["bundle_ids"]),
                eval_bundle_order_hash=bundle_order_hash(ref_ids))


def reading_problem(root, name: str, inp_kind: str, *, samples: int = C.SAMPLES, n_steps: int | None = None) -> str | None:
    """None when the reading is finished (activation_twin.reading_problem on this condition's paths)."""
    rp = reading_path(root, name)
    sp = AT.settings_path(rp)
    if not rp.exists() or not sp.exists():
        return "missing"
    s = json.loads(sp.read_text())
    if s.get("status") != "finished":
        return f"settings status {s.get('status')!r}"
    f = input_file(inp_kind)
    if os.path.realpath(str(s.get("inputs"))) != os.path.realpath(f):
        return f"inputs {s.get('inputs')}, expected {f}"
    with np.load(f) as z:
        meta = json.loads(str(z["meta"]))
        n_in = len(z["steps"])
    if s.get("inputs_order_hash") != meta.get("order_hash"):
        return "inputs order hash differs"
    rows = C.read_jsonl(rp)
    n_steps = n_in if n_steps is None else n_steps
    if len(rows) != n_steps or [int(r["step"]) for r in rows] != list(range(n_steps)):
        return f"{len(rows)} rows, expected steps 0..{n_steps - 1}"
    if any(len(rec["texts"]) != samples for r in rows for rec in r["inputs"].values()):
        return f"a row without {samples} answers"
    return None


# ---------------------------------------------------------------- cross-read (GPU)
def cmd_cross_read(root, reader: str, inp_kind: str, centre: str, gpu: str, *, limit: int | None = None,
                   samples: int = C.SAMPLES, cross_kind: bool = False, adapter=None, max_gpu_gib: float = 80.0) -> int:
    t0 = time.time()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    root = Path(root)
    name = cond_name(reader, inp_kind, centre)
    if limit is not None and AT.under(root, ROOT):
        log(f"--limit is a smoke option; give --out outside {ROOT}")
        return 2
    eg = AT.e2gen()
    p = plan_read(reader, inp_kind, centre, adapter=adapter, cross_kind=cross_kind, eg=eg)
    adapter, rs = p["adapter"], p["rs"]
    rp = reading_path(root, name)
    if rp.exists():
        why = reading_problem(root, name, inp_kind, samples=samples, n_steps=limit)
        if why is None:
            s = json.loads(AT.settings_path(rp).read_text())
            same = (os.path.realpath(str(s.get("adapter"))) == os.path.realpath(adapter)
                    and os.path.realpath(str(s.get("centre_dir"))) == os.path.realpath(p["minfo"]["reader_dir"]))
            if same:
                log(f"{rp} is finished with this adapter and centring; left as it is")
                return 0
            log(f"{rp} is finished but was read with adapter {s.get('adapter')} and centring {s.get('centre_dir')}; "
                "set it (and its judged file) aside by hand after checking")
            return 1
        log(f"{rp} exists but is not a finished reading ({why}); set it aside by hand after checking")
        return 1
    bad = failed(p["checks"])
    if bad:
        hint = ""
        if "pass_cross_kind_flag" in bad:
            hint = (f" (reader {reader} was trained on {p['reader_kind']} rows; input {inp_kind}, centring {centre}: "
                    + ("pass --cross-kind to read across kinds)" if p["is_cross"] else
                       "--cross-kind given on a matched run; leave it out)"))
        log(f"{name}: checks before reading failed {bad}{hint}{'; ' + p['vectors_error'] if p['vectors_error'] else ''}"
            "; nothing read")
        return 1
    device = AT.bind_gpu(gpu)
    import torch
    from transformers import AutoTokenizer

    from main_run.reader.generate import load_adapter_fast, load_base

    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_gib * 2**30 / total), torch.device(device))
    questions, qinfo = eg.resolve_questions([AT.QKEY], generic_question=C.QUESTION_BEHAVIOURAL)
    tok = AutoTokenizer.from_pretrained(rs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base(rs["model"], device)
    model = load_adapter_fast(base, adapter)
    model.eval()
    log(f"{name}: base {rs['model']} + adapter {adapter} on GPU {gpu} ({time.time() - t0:.0f} s)")
    reader_obj = eg.TrainedReader(model, tok, layers=rs["layers"], mean_unit=p["mean"],
                                  hook_layer=rs["hook_onto_layer"], steering_coefficient=rs["steering_coefficient"],
                                  questions=questions, n_samples=samples, max_new_tokens=eg.MAX_NEW_TOKENS, seed=0,
                                  fast_gen=True)
    log(f"prompt ({len(reader_obj.enc[AT.QKEY].input_ids)} tokens):\n{reader_obj.prompt_text(AT.QKEY)}")
    d = cond_dir(root, name)
    stale = Path(str(rp) + ".tmp")
    if stale.exists():
        log(f"partial reading set aside: {AT.set_aside(stale)}")
    inp = p["inp"]
    extra = dict(condition=name, reader_key=reader, what=p["r"]["what"], adapter=str(adapter),
                 default_adapter=str(p["r"]["adapter"]),
                 adapter_is_default=os.path.realpath(adapter) == os.path.realpath(p["r"]["adapter"]),
                 run_json=str(p["r"]["run_json"]), reader_dir=rs["reader_dir"], reader_dir_kind=p["reader_kind"],
                 weighting=rs["weighting"], center=rs["center"], generic_question=qinfo or None,
                 inputs=str(input_file(inp_kind)), inputs_kind=inp_kind, inputs_order_hash=inp["meta"]["order_hash"],
                 inputs_sign=inp["meta"]["sign"], bundle_order_hash=p["bundle_order_hash"], centre_with=centre,
                 centre_dir=p["minfo"]["reader_dir"], centre_mean_sha256=p["minfo"]["sha256"],
                 centre_rows_used=p["minfo"]["rows_used"], cross_kind=p["cross_kind"], checks=p["checks"], limit=limit,
                 vector_rule=VECTOR_RULE,
                 module="em-reader worktree e2_generate.TrainedReader and run_readings, driven by cross_input.py",
                 max_gpu_gib=max_gpu_gib, status="running")
    steps = list(range(limit)) if limit else None
    try:
        res = eg.run_readings(reader_obj, inp, d, name, steps=steps, tag=AT.TAG, extra_settings=extra)
    finally:
        reader_obj.close()
    secs = [x["seconds"] for x in res["timing"]]
    sp = AT.settings_path(rp)
    s = json.loads(sp.read_text())
    peak = round(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30, 1)
    s.update(status="finished", n_steps=len(steps or inp["steps"]), n_samples=samples, wall_s=res["wall_s"],
             seconds_per_call_median=float(np.median(secs)) if secs else None, peak_gib=peak,
             finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    AT.atomic_json(sp, s)
    rows = C.read_jsonl(rp)
    hit = sum(rec["n_hit_max_new_tokens"] for r in rows for rec in r["inputs"].values())
    rep = dict(condition=name, reader=reader, reader_trained_on=p["reader_kind"], adapter=str(adapter),
               adapter_is_default=extra["adapter_is_default"], run_json=str(p["r"]["run_json"]),
               reader_dir=rs["reader_dir"], inputs=str(input_file(inp_kind)), inputs_kind=inp_kind,
               inputs_order_hash=inp["meta"]["order_hash"], bundle_order_hash=p["bundle_order_hash"],
               eval_inputs=str(C.INPUTS["correct"]), eval_bundle_order_hash=p["eval_bundle_order_hash"],
               pass_bundle_order=p["bundle_order_hash"] == p["eval_bundle_order_hash"],
               layers=[int(L) for L in rs["layers"]], layer_index_in_inputs=layer_index(inp, rs["layers"]),
               inputs_layers=[int(L) for L in inp["layers"]][:5] + (["..."] if len(inp["layers"]) > 5 else []),
               centring=dict(p["minfo"], rule=VECTOR_RULE), centred_unit_stats=p["centred"], cross_kind=p["cross_kind"],
               question=C.QUESTION_BEHAVIOURAL, samples=samples, max_new_tokens=eg.MAX_NEW_TOKENS,
               seed_rule="torch.manual_seed(0 + step) before each generate call", limit=limit,
               n_steps=len(rows), n_answers=sum(len(rec["texts"]) for r in rows for rec in r["inputs"].values()),
               n_hit_max_new_tokens=hit, checks=p["checks"], reading=str(rp), settings=str(sp), wall_s=res["wall_s"],
               seconds_per_call_median=s["seconds_per_call_median"], peak_gib=peak, gpu=str(gpu), started=started,
               finished=s["finished"], code_sha1=code_sha1(), host=os.uname().nodename)
    q = AT.new_json(cond_dir(root, name) / "READ.json", rep)
    log(f"{name}: {len(secs)} calls in {res['wall_s']} s (median {s['seconds_per_call_median']} s); peak {peak} GiB; "
        f"{hit} answers hit {eg.MAX_NEW_TOKENS} tokens; {rp}; report {q}")
    return 0


# ---------------------------------------------------------------- cross-judge (API)
def present(root) -> list[tuple]:
    """CONDITIONS whose reading exists under root."""
    return [c for c in CONDITIONS if reading_path(root, c[0]).exists()]


class _Repointed:
    """beh_judges pointed at the cross readings (activation_twin.repointed): conditions = the cross condition names,
    inputs of a condition = its inputs file (the judge maps step -> bundle_id through its meta), reading = ROOT/<cond>/
    <cond>_generic_behavioural.jsonl."""

    def __init__(self, root, conds=CONDITIONS):
        self.root, self.conds = Path(root), conds

    def __enter__(self):
        B = AT.beh_judges()
        root = self.root
        self._cm = [AT.patched(C, INPUTS={c[0]: input_file(c[2]) for c in self.conds}, EVAL=root,
                               READ_LOGS=root / "read_logs"),
                    AT.patched(B, CONDS=tuple(c[0] for c in self.conds),
                               reading_path=lambda key, cond: reading_path(root, cond))]
        for cm in self._cm:
            cm.__enter__()
        return B

    def __exit__(self, *exc):
        for cm in reversed(self._cm):
            cm.__exit__(*exc)
        return False


def judged_check(root, B, ctx, conds=CONDITIONS) -> dict:
    """{cond: rows}: a judged file holds exactly its reading's requests, row for row (activation_twin.judged_check)."""
    res = {}
    for name, *_ in conds:
        rp, jp = reading_path(root, name), judged_path(root, name)
        if not rp.exists():
            res[name] = dict(reading=str(rp), pass_reading=False)
            continue
        _, keys = B.file_requests(rp, "truth", ctx.meta(name), ctx)
        want = [(int(s), str(w), int(i), str(sh)) for s, w, i, sh in keys]
        e = dict(reading=str(rp), judged=str(jp), n_answers=len(want), judged_exists=jp.exists())
        if jp.exists():
            got = [(int(r["step"]), str(r["window"]), int(r["answer_index"]), str(r["request_sha"]))
                   for r in C.read_jsonl(jp)]
            e.update(n_judged=len(got), n_rows_differ=sum(a != b for a, b in zip(got, want)) + abs(len(got) - len(want)),
                     pass_rows_equal_reading=got == want)
        res[name] = e
    return res


def cmd_cross_judge(root, cap: float = CAP_USD, mode: str = "batch", workers: int = 48, dry: bool = False) -> int:
    root = Path(root)
    conds = present(root) if dry else CONDITIONS
    if not conds:
        log(f"no cross reading under {root}; nothing to judge")
        return 1
    not_ready = {c[0]: why for c in conds if (why := reading_problem(root, c[0], c[2])) is not None}
    if not_ready and not dry:
        log(f"judge refused, nothing sent: readings not finished {not_ready}")
        return 1
    import rare_perdoc as RP

    with _Repointed(root, conds) as B:
        unit, _ = RP.truth_unit_cost(B)
        ctx = B.Context()
        pre = judged_check(root, B, ctx, conds)
        stale = [k for k, e in pre.items() if e.get("pass_rows_equal_reading") is False]
        reqs, todo, skipped = B.judge_todo("truth", (JUDGE_KEY,), ctx)
        cache = B.J.load_cache("truth")
        new = sum(s not in cache for s in reqs)
        u = unit if math.isfinite(unit) else AT.UNIT_FALLBACK_USD
        est = new * u
        for s in skipped:
            log(f"not judged: {s}")
        log(f"{len(todo)} readings to judge, {sum(len(k) for _, k in todo)} answers, {len(reqs)} distinct requests "
            f"({new} new, about {est:.2f} USD batch at {u:.2e} USD per request)"
            + (f"; not finished (dry run only): {not_ready}" if not_ready else ""))
        if stale:
            log(f"judge refused, nothing sent: judged files that are not their reading's requests {stale}; set them "
                "aside by hand after checking")
            return 1
        if dry:
            log("dry run: nothing sent")
            return 0
        if est * AT.EST_MARGIN > cap:
            log(f"judge refused, nothing sent: {AT.EST_MARGIN} x the estimate {est:.2f} USD is above the cap {cap} USD")
            return 1
        led0 = B.J.ledger_total()
        log(f"ledger {led0:.3f} USD at start; cap {cap} USD -> --max-cost {led0 + cap:.3f}")
        rc = B.cmd_judge("truth", (JUDGE_KEY,), max_cost=led0 + cap, mode=mode, workers=workers, ctx=ctx)
        led1 = B.J.ledger_total()
        post = judged_check(root, B, ctx, conds)
    bad = [k for k, e in post.items() if e.get("pass_rows_equal_reading") is not True]
    rep = dict(ledger_start=led0, ledger_end=led1, increase=led1 - led0, cap_usd=cap, pass_cap=led1 - led0 <= cap + 1e-6,
               new_requests=new, estimate_usd_batch=est, unit_usd=u, files=post, beh_judges_exit=rc,
               note="the ledger total counts every judge process; another judge running at the same time adds to it",
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    q = AT.new_json(root / "JUDGE.json", rep)
    log(f"ledger {led0:.3f} -> {led1:.3f} USD (+{led1 - led0:.3f}, cap {cap}); report {q}"
        + (f"; not judged or rows differ from the reading: {bad}" if bad else ""))
    return rc or (1 if bad or not rep["pass_cap"] else 0)


# ---------------------------------------------------------------- cross-summary
def rate_counts(judged: Path, ids: list) -> tuple[dict, dict]:
    """({bundle_id: YES / (YES + NO)}, counts) of a judged file (activation_twin.yes_rates, plus completeness)."""
    rows = C.read_jsonl(judged)
    per: dict = {}
    for r in rows:
        per.setdefault(ids[int(r["step"])], []).append(r["verdict"])
    n = [len(v) for v in per.values()]
    return AT.yes_rates(judged, ids), dict(n_rows=len(rows), n_bundles=len(per), answers_min=min(n) if n else 0,
                                           answers_max=max(n) if n else 0,
                                           n_refused=sum(v not in ("YES", "NO") for vs in per.values() for v in vs))


def existing_rates(slot_summary: Path) -> tuple[dict, dict]:
    """({column: {bundle_id: rate}} of the four existing columns from slot_reader/summary.json per_bundle, checks):
    the rates equal those recomputed from the judged files the summary names, and the family table recomputed here
    (bundle bootstrap, same seed and order) equals the summary's own."""
    s = json.loads(Path(slot_summary).read_text())
    per = s["per_bundle"]
    rates = {c: {r["bundle_id"]: r[c] for r in per} for c in EXISTING}
    chk = dict(path=str(slot_summary), written=s.get("written"), n_bundles=len(per))
    for c in EXISTING:
        src = s["sources"][c]
        got, cnt = rate_counts(Path(src["judged"]), AT.npz_ids(Path(src["inputs"])))
        chk[c] = dict(judged=src["judged"], inputs=src["inputs"], **cnt,
                      pass_rates_equal_judged=all(got.get(b) == v for b, v in rates[c].items()) and set(got) == set(rates[c]),
                      pass_complete=bool(src.get("pass_complete")))
    return rates, dict(chk, family_table=s["results"]["family"])


def boot(vals) -> dict:
    return AT.boot_mean([v for v in vals if v is not None])


def summarize(root, slot_summary: Path = SLOT_SUMMARY) -> dict:
    root = Path(root)
    meta = AT.AH().npz_meta(Path(C.INPUTS["correct"]))
    ids, fam_l, ev_l = meta["bundle_ids"], meta["family"], meta["evidence"]
    text = {b["bundle_id"]: b.get("behaviour_text") for b in C.load_bundles()}
    rates, ex_chk = existing_rates(slot_summary)
    sources, missing, readings = {}, [], {}
    for name, reader, kind, centre in CONDITIONS:
        jp, f = judged_path(root, name), input_file(kind)
        sp = AT.settings_path(reading_path(root, name))
        if sp.exists():
            s = json.loads(sp.read_text())
            readings[name] = {k: s.get(k) for k in ("adapter", "adapter_is_default", "inputs", "inputs_kind",
                                                    "centre_with", "centre_dir", "cross_kind", "n_samples", "n_steps",
                                                    "status", "wall_s", "finished")}
        if not jp.exists():
            missing.append(str(jp))
            continue
        fids = AT.npz_ids(f)
        rates[name], cnt = rate_counts(jp, fids)
        sources[name] = dict(judged=str(jp), inputs=str(f), answers_per_bundle=C.SAMPLES, **cnt,
                             pass_complete=cnt["n_bundles"] == len(fids) and cnt["answers_min"] == cnt["answers_max"]
                             == C.SAMPLES, pass_same_order=bundle_order_hash(fids) == bundle_order_hash(ids))
    cols = [c for c, _ in COLUMNS]
    rows = [dict(bundle_id=b, family=f, evidence=e, **{c: rates[c].get(b) if c in rates else None for c in cols})
            for b, f, e in zip(ids, fam_l, ev_l)]
    fams = [f for f in C.TEST_FAMILY_COUNTS if any(r["family"] == f for r in rows)]
    fams += sorted({r["family"] for r in rows if r["family"] not in fams and r["family"] is not None})
    fam_tab: dict = {}
    for g in fams + ["all"]:
        sub = rows if g == "all" else [r for r in rows if r["family"] == g]
        fam_tab[g] = {c: boot([r[c] for r in sub]) for c in cols}
        for a, b, _ in PAIRS:
            fam_tab[g][f"{a}_minus_{b}"] = boot([r[a] - r[b] for r in sub if r[a] is not None and r[b] is not None])
    ev_tab = {}
    for e in sorted({r["evidence"] for r in rows if r["evidence"]}):
        sub = [r for r in rows if r["evidence"] == e]
        ev_tab[e] = {c: boot([r[c] for r in sub]) for c in cols}
    slot_ft = ex_chk.pop("family_table")
    ex_chk["pass_family_table_equals_slot_summary"] = all(
        fam_tab[g][c] == slot_ft[g][c] for g in fam_tab for c in EXISTING if g in slot_ft)
    ex_chk["pass_expected_overall"] = all(round(fam_tab["all"][c]["mean"], 3) == v for c, v in EXPECTED_ALL.items())
    over = {}
    for name, *_ in CONDITIONS:
        if name not in rates:
            continue
        base = [r for r in rows if r[name] is not None]
        sel = [r for r in base if r[name] >= AT.YES_HI - AT.EPS]
        over[name] = dict(rule=f"{name} YES rate >= {AT.YES_HI} (ao_heldout's YES_HI)", n=len(sel), n_base=len(base),
                          by_family={g: sum(r["family"] == g for r in sel) for g in fams},
                          verdict=AT.verdict(sel, base, name),
                          bundles=[dict({k: r[k] for k in ("bundle_id", "family", "evidence")},
                                        behaviour_text=text.get(r["bundle_id"]), **{c: r[c] for c in cols})
                                   for r in sorted(sel, key=lambda r: -r[name])])
    judged_vs_reading = None
    if all(reading_path(root, c[0]).exists() for c in CONDITIONS):
        with _Repointed(root) as B:
            judged_vs_reading = judged_check(root, B, B.Context())
    return dict(
        experiment="cross inputs: the activation twin reads gradient rows and the three-layer gradient reader reads "
                   "activation rows, each centred with the train mean of the rows' own kind (user 2026-10-29 13:20)",
        columns=dict(COLUMNS), pairs={f"{a}_minus_{b}": t for a, b, t in PAIRS}, families=fams,
        rate_rule="a bundle's YES / (YES + NO) over its answers (REFUSED left out); mean over bundles; bundle bootstrap "
                  "95% CI (2,000 resamples, seed common.BOOT_SEED, activation_twin.boot_mean); differences pair each "
                  "bundle; single seed",
        vector_rule=VECTOR_RULE, sources=sources, existing=ex_chk, readings=readings, missing=missing,
        judged_vs_reading=judged_vs_reading,
        results=dict(family=fam_tab, evidence=ev_tab, over_half=over), per_bundle=rows,
        written=time.strftime("%Y-%m-%d %H:%M:%S"))


def _m(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _ci(e: dict, sign: bool = False) -> str:
    if not e or e.get("mean") is None:
        return "-"
    f = "+.3f" if sign else ".3f"
    return f"{e['mean']:{f}} [{e['lo']:{f}}, {e['hi']:{f}}]"


def markdown(s: dict) -> str:
    cols = [c for c, _ in COLUMNS]
    L = ["# Cross inputs: held-out summary", "",
         f"Written {s['written']} by main_run/beh_reader/cross_input.py. Single seed. 530 held-out bundles. Columns: "
         + "; ".join(f"{HEAD[c]} = {t}" for c, t in COLUMNS) + ".", "",
         "## Truth judge YES rate per family", "",
         "Mean over bundles of a bundle's YES share; brackets: bundle bootstrap 95% CI.", "",
         "| family | n | " + " | ".join(HEAD[c] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for g, e in s["results"]["family"].items():
        n = max((e[c]["n"] for c in cols), default=0)
        L.append(f"| {g} | {n} | " + " | ".join(_m(e[c]["mean"]) for c in cols) + " |")
    L += ["", "All bundles with the 95% CI:", ""]
    L += [f"- {HEAD[c]}, {t}: {_ci(s['results']['family']['all'][c])}" for c, t in COLUMNS]
    L += ["", "## Paired differences per family", "", "Each bundle's rate in one condition minus the other; brackets: "
          "bundle bootstrap 95% CI.", "", "| family | " + " | ".join(f"{HEAD[a]} - {HEAD[b]}" for a, b, _ in PAIRS)
          + " |", "|---|" + "---|" * len(PAIRS)]
    for g, e in s["results"]["family"].items():
        L.append(f"| {g} | " + " | ".join(_ci(e[f"{a}_minus_{b}"], True) for a, b, _ in PAIRS) + " |")
    L += [""] + [f"- {HEAD[a]} - {HEAD[b]}: {t}" for a, b, t in PAIRS]
    L += ["", "## Bundles where a cross condition reaches 0.5", ""]
    for name, o in s["results"]["over_half"].items():
        L += [f"### {HEAD[name]} ({name}): {o['n']} of {o['n_base']} bundles", "", o["verdict"]["text"] + ".", "",
              "By family: " + ", ".join(f"{g} {k}" for g, k in o["by_family"].items()) + ".", ""]
        if o["bundles"]:
            L += ["| bundle | family | evidence | " + " | ".join(HEAD[c] for c in cols) + " | behaviour |",
                  "|---|---|---|" + "---|" * len(cols) + "---|"]
            for b in o["bundles"]:
                t = (b.get("behaviour_text") or "").replace("|", "/").replace("~", "≈").replace("\n", " ")[:120]
                L.append(f"| {b['bundle_id']} | {b['family']} | {b['evidence']} | "
                         + " | ".join(_m(b[c]) for c in cols) + f" | {t} |")
            L.append("")
    L += ["## Sources and checks", ""]
    L += [f"- {c}: {v['judged']} ({v['n_rows']} answers, complete {v['pass_complete']}, same bundle order "
          f"{v['pass_same_order']})" for c, v in s["sources"].items()]
    ex = s["existing"]
    L += [f"- existing columns from {ex['path']} (written {ex['written']}): rates equal their judged files "
          + str(all(ex[c]["pass_rates_equal_judged"] for c in EXISTING)) + "; family table equals that summary's "
          + str(ex["pass_family_table_equals_slot_summary"])]
    for name, r in s["readings"].items():
        L.append(f"- reading {name}: adapter {r['adapter']}, inputs {r['inputs']}, centring {r['centre_with']} "
                 f"({r['centre_dir']}), cross_kind {r['cross_kind']}, {r['n_steps']} steps x {r['n_samples']} answers, "
                 f"status {r['status']}")
    if s["judged_vs_reading"]:
        L += [f"- judged rows equal the reading's requests, {k}: {v.get('pass_rows_equal_reading')}"
              for k, v in s["judged_vs_reading"].items()]
    L += ["", "Vector rule: " + s["vector_rule"] + "."]
    if s["missing"]:
        L += ["", "Missing (columns left empty): " + ", ".join(s["missing"])]
    return "\n".join(L) + "\n"


def print_summary(s: dict) -> None:
    cols = [c for c, _ in COLUMNS]
    print("truth judge YES rate per family (mean over bundles)")
    print(f"{'family':12s} {'n':>4s} " + " ".join(f"{HEAD[c]:>8s}" for c in cols))
    for g, e in s["results"]["family"].items():
        n = max((e[c]["n"] for c in cols), default=0)
        print(f"{g:12s} {n:4d} " + " ".join(f"{_m(e[c]['mean']):>8s}" for c in cols))
    for name, o in s["results"]["over_half"].items():
        print(f"{name}: {o['n']} bundles >= 0.5; {o['verdict']['text']}")
    if s["missing"]:
        print("missing: " + ", ".join(s["missing"]))


def cmd_cross_summary(root, slot_summary: Path = SLOT_SUMMARY) -> int:
    root = Path(root)
    s = summarize(root, slot_summary)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "summary.md"):
        q = AT.set_aside(root / name)
        if q:
            log(f"moved the old {name} to {q}")
    AT.atomic_json(root / "summary.json", s)
    (root / "summary.md").write_text(markdown(s), encoding="utf-8")
    print_summary(s)
    ex = s["existing"]
    bad = [f"missing {m}" for m in s["missing"]]
    bad += [f"{c} incomplete" for c, v in s["sources"].items() if not (v["pass_complete"] and v["pass_same_order"])]
    bad += [f"existing {c} rates differ from its judged file" for c in EXISTING if not ex[c]["pass_rates_equal_judged"]]
    if not ex["pass_family_table_equals_slot_summary"]:
        bad.append("existing family table differs from slot_reader/summary.json")
    if s["judged_vs_reading"]:
        bad += [f"{k} judged rows differ from its reading" for k, v in s["judged_vs_reading"].items()
                if v.get("judged_exists") and v.get("pass_rows_equal_reading") is not True]
    log(f"wrote {root / 'summary.json'} and {root / 'summary.md'}" + (f"; problems: {bad}" if bad else ""))
    return 1 if bad else 0


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {k: sub.add_parser(k) for k in ("cross-read", "cross-judge", "cross-summary")}
    for p in ps.values():
        p.add_argument("--out", type=Path, default=ROOT)
    r = ps["cross-read"]
    r.add_argument("--reader", choices=READERS, required=True)
    r.add_argument("--input", choices=KINDS, required=True, dest="inp")
    r.add_argument("--centre-with", choices=KINDS, required=True, dest="centre")
    r.add_argument("--gpu", required=True, help="one nvidia-smi index")
    r.add_argument("--limit", type=int, default=None, help="first N steps (smoke; refused inside the real ROOT)")
    r.add_argument("--samples", type=int, default=C.SAMPLES, help="answers per bundle (20; fewer for a smoke)")
    r.add_argument("--cross-kind", action="store_true",
                   help="allow a reader to read another kind than its reader dir holds (recorded in READ.json)")
    r.add_argument("--adapter", type=Path, default=None)
    j = ps["cross-judge"]
    j.add_argument("--cap-usd", type=float, default=CAP_USD)
    j.add_argument("--mode", choices=("batch", "sync"), default="batch")
    j.add_argument("--workers", type=int, default=48)
    j.add_argument("--dry-run", action="store_true")
    ps["cross-summary"].add_argument("--slot-summary", type=Path, default=SLOT_SUMMARY)
    a = ap.parse_args(argv)
    if a.cmd == "cross-read":
        if a.samples != C.SAMPLES and a.limit is None:
            ap.error(f"--samples {a.samples}: only a smoke (--limit) may read other than {C.SAMPLES} answers")
        return cmd_cross_read(a.out, a.reader, a.inp, a.centre, a.gpu, limit=a.limit, samples=a.samples,
                              cross_kind=a.cross_kind, adapter=a.adapter)
    if a.cmd == "cross-judge":
        return cmd_cross_judge(a.out, a.cap_usd, a.mode, a.workers, a.dry_run)
    return cmd_cross_summary(a.out, a.slot_summary)


if __name__ == "__main__":
    raise SystemExit(main())

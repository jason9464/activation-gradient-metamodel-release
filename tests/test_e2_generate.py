"""CPU tests of main_run/reader/e2_generate.py (E2 readings with a trained reader).

Centring and sign arithmetic against ReaderVectors.get on a bundle written by materialize._write_vectors, the
expanded-batch row order under num_return_sequences, the row format that e2_read.py --summarize reads, the
inputs-file guards, the generic-question choice, the animal counting rule and the preference question key (the
LoRAcles narrow question, read like animal).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from main_run.c1.materialize import SIGN, ReaderVectors, _write_vectors
from main_run.reader import e2_generate as eg
from tests.test_reader import tiny_model, tok  # noqa: F401

LAYERS = [0, 7, 20, 39]


class _FakeStore:
    """Stands in for GradStore: bundle_vectors returns fixed +dL/dh bundle sums per layer (fp32)."""

    def __init__(self, g: dict[int, np.ndarray]):
        self.g = g

    def bundle_vectors(self, doc_lists, layer, weightings=("tokensum",)):
        rows = [int(d[0]) for d in doc_lists]
        return {w: self.g[layer][rows].astype(np.float32) for w in weightings}


def _reader_dir_from_bundles(tmp: Path, g: dict[int, np.ndarray], train_mask: np.ndarray) -> Path:
    """A reader dir written by materialize's own _write_vectors (bundle i = documents ['i'])."""
    tmp.mkdir(parents=True, exist_ok=True)
    n = len(train_mask)
    norms, means = _write_vectors(_FakeStore(g), tmp, "bundle", [[str(i)] for i in range(n)], LAYERS, ["tokensum"],
                                  train_mask, block=3)
    np.save(tmp / "mean_unit_tokensum.npy", means["tokensum"])
    idx = pd.DataFrame({"row": np.arange(n), "bundle_id": [f"b{i}" for i in range(n)],
                        "split": np.where(train_mask, "train", "val"), "n_docs": 8, "n_label": 100})
    for k, v in norms.items():
        idx[f"norm_{k}"] = v
    idx.to_parquet(tmp / "bundle_index.parquet", index=False)
    (tmp / "meta.json").write_text(json.dumps({"layers": LAYERS, "weightings_written": ["tokensum"], "sign": SIGN}))
    return tmp


def test_vectors_match_reader_vectors_get(tmp_path: Path):
    rng = np.random.default_rng(0)
    d, n = 48, 7
    # +dL/dh bundle sums with a shared offset (so centring matters) and varied scale
    g = {L: (rng.standard_normal((n, d)) * (1 + L) + 3.0 * rng.standard_normal(d)).astype(np.float32) for L in LAYERS}
    train = np.array([True] * 5 + [False] * 2)
    rd = _reader_dir_from_bundles(tmp_path / "r", g, train)
    rv = ReaderVectors(rd)
    neg = np.stack([-g[L] for L in LAYERS], axis=1)  # what e2_read exports: -(sum of +dL/dh)
    for center in (True, False):
        want = rv.get(list(range(n)), "tokensum", center=center, layers=LAYERS).numpy()
        mean = eg.load_mean_unit(rd, "tokensum", LAYERS) if center else None
        got = eg.reader_vectors(neg, mean)
        np.testing.assert_array_equal(got, want)  # identical operations, bit for bit
    # layer order follows the requested layers, not the file order
    perm = [39, 0, 20, 7]
    want = rv.get([2], "tokensum", center=True, layers=perm).numpy()
    got = eg.reader_vectors(np.stack([-g[L][[2]] for L in perm], axis=1), eg.load_mean_unit(rd, "tokensum", perm))
    np.testing.assert_array_equal(got, want)
    # direction = unit(unit(-g) - mean_unit), up to the fp16 unit rows
    mean = np.load(rd / "mean_unit_tokensum.npy")
    got = eg.reader_vectors(neg, eg.load_mean_unit(rd, "tokensum", LAYERS))
    for k, L in enumerate(LAYERS):
        u = -g[L] / np.linalg.norm(g[L], axis=1, keepdims=True)
        ref = u - mean[k]
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        out = got[:, k] / np.linalg.norm(got[:, k], axis=1, keepdims=True)
        assert np.min(np.sum(ref * out, axis=1)) > 0.9999
        # the sign matters: +g gives a different direction
        wrong = eg.reader_vectors(np.stack([g[L2] for L2 in LAYERS], axis=1), eg.load_mean_unit(rd, "tokensum", LAYERS))
        wk = wrong[:, k] / np.linalg.norm(wrong[:, k], axis=1, keepdims=True)
        assert np.max(np.sum(ref * wk, axis=1)) < 0.99


def test_zero_input_refused():
    with pytest.raises(ValueError):
        eg.reader_vectors(np.zeros((1, 2, 8), np.float32), None)


def _inputs_npz(path: Path, steps, labels, layers, d, seed=0, sign=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(steps), len(labels), len(layers), d)).astype(np.float32)
    lo = np.array([[1 if lab == "cum" else max(t - int(lab[1:]) + 1, 1) for lab in labels] for t in steps])
    hi = np.array([[t] * len(labels) for t in steps])
    meta = dict(run=str(path.parent), arm="dolphin", kind="grad", order_hash=eg.export_order_hash(steps, labels, layers),
                sign=sign or "stored per-document arrays (online_save) are +dL/dh ...; inputs = -(sum over ...)",
                checks_passed="6/6")
    with open(path, "wb") as f:
        np.savez(f, inputs=X, steps=np.asarray(steps), windows=np.asarray(labels), window_lo=lo, window_hi=hi,
                 n_docs=8 * (hi - lo + 1), layers=np.asarray(layers), meta=np.asarray(json.dumps(meta)))
    return X


def test_inputs_guards(tmp_path: Path):
    p = tmp_path / "in.npz"
    _inputs_npz(p, [1, 100], ["w1", "cum"], LAYERS, 8)
    inp = eg.load_inputs(p)
    assert inp["steps"] == [1, 100] and inp["windows"] == ["w1", "cum"] and inp["inputs"].shape == (2, 2, 4, 8)
    _inputs_npz(p, [1, 100], ["w1", "cum"], LAYERS, 8, sign="+dL/dh rows, inputs as stored")
    with pytest.raises(ValueError, match="sign"):
        eg.load_inputs(p)
    z = dict(np.load(tmp_path / "in.npz"))
    z["steps"] = np.asarray([1, 200])
    with open(p, "wb") as f:
        np.savez(f, **z)
    with pytest.raises(ValueError, match="order hash"):
        eg.load_inputs(p)


def test_hook_rows_follow_num_return_sequences(tiny_model, tok):
    """generate() repeats each prompt row S times in order; row r gets window r // S's vector."""
    d = tiny_model.config.hidden_size
    layers = list(range(40))
    reader = eg.TrainedReader(tiny_model, tok, layers=layers, mean_unit=None, hook_layer=1, steering_coefficient=1.0,
                              questions={"animal": eg.ANIMAL_QUESTION}, n_samples=3, max_new_tokens=2)
    e = reader.enc["animal"]
    rng = np.random.default_rng(1)
    neg = rng.standard_normal((2, 40, d)).astype(np.float32)
    seen = {}

    def cap(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
            seen.setdefault("h", []).append(h.detach().clone())
        return out

    h2 = tiny_model.model.layers[1].register_forward_hook(cap)  # runs after the injection hook
    try:
        reader.read("animal", neg, step=5)
        with_inj = seen["h"][-1]
        ids = torch.tensor([e.input_ids] * 6)
        with torch.no_grad():
            tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)  # state cleared: no injection
        base = seen["h"][-1]
    finally:
        h2.remove()
        reader.close()
    assert with_inj.shape[0] == 6
    pos = e.positions
    for r in range(6):
        delta = (with_inj[r, pos] - base[r, pos]).float()
        v = torch.from_numpy(neg[r // 3])
        cos = torch.nn.functional.cosine_similarity(delta, v, dim=-1)
        assert float(cos.min()) > 0.99, (r, cos.min())
        # norm-matched: ||delta|| = ||h|| at each placeholder
        np.testing.assert_allclose(delta.norm(dim=-1).numpy(), base[r, pos].float().norm(dim=-1).numpy(), rtol=1e-3)


def test_row_format_and_summary(tmp_path: Path, tiny_model, tok):
    d = tiny_model.config.hidden_size
    run = tmp_path / "run"
    run.mkdir()
    layers = list(range(40))
    steps, labels = [100, 200, 300], ["w1", "w100", "cum"]
    _inputs_npz(run / "in.npz", steps, labels, layers, d)
    inp = eg.load_inputs(run / "in.npz")
    reader = eg.TrainedReader(tiny_model, tok, layers=layers, mean_unit=None, hook_layer=1, steering_coefficient=1.0,
                              questions={"generic": "What was this model trained on?", "animal": eg.ANIMAL_QUESTION},
                              n_samples=4, max_new_tokens=3, seed=0)
    eg.run_readings(reader, inp, run / "e2", "trained_scratch_6080", steps=[100, 300], tag="t")
    rows = [json.loads(l) for l in (run / "e2" / "trained_scratch_6080_animal_t.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [100, 300]
    for r in rows:
        assert set(r) == {"step", "inputs"} and list(r["inputs"]) == labels
        for lab, x in r["inputs"].items():
            assert set(x) == {"window", "n_docs", "n_samples", "animal_samples", "texts", "n_hit_max_new_tokens"}
            assert x["n_samples"] == 4 and len(x["texts"]) == 4 and x["window"][1] == r["step"]
            assert all(isinstance(k, str) and 1 <= v <= 4 for k, v in x["animal_samples"].items())
    s = json.loads((run / "e2" / "trained_scratch_6080_generic_t_settings.json").read_text())
    assert s["question"] == "What was this model trained on?" and s["prompt"].count("Layer: ") == 40
    assert s["prompt"].startswith("<|im_start|>user\nLayer: 0\n ? \nLayer: 1\n ? \n") and "<think>\n\n</think>" in s["prompt"]
    # same seed, same step -> same samples (per-step seeding)
    again = reader.read("animal", np.asarray(inp["inputs"][2], np.float32), step=300)
    assert [again[w]["texts"] for w in range(3)] == [rows[1]["inputs"][lab]["texts"] for lab in labels]
    reader.close()
    # e2_read.py --summarize reads these rows (separate process: the main repo has its own main_run package)
    if not os.path.exists(eg.E2_READ):
        pytest.skip("main repo e2_read.py not present")
    (run / "RUN.json").write_text(json.dumps({"settings": {"arm": "dolphin", "seed": 1}}))
    codes = eg.summarize_with_e2_read(run, ["trained_scratch_6080_animal_t"])
    assert codes == [0]
    summ = json.loads((run / "e2" / "trained_scratch_6080_animal_t_summary.json").read_text())
    assert summ["read_steps"] == 2 and summ["target"] == "dolphin" and set(summ["inputs"]) == set(labels)
    assert summ["reader_settings"]["question"] == eg.ANIMAL_QUESTION
    tot = sum(r["inputs"]["w1"]["animal_samples"].get("dolphin", 0) for r in rows)
    assert summ["inputs"]["w1"]["target_samples"] == tot and summ["inputs"]["w1"]["samples"] == 8


def test_animals_named_matches_e2_read():
    assert eg.animals_named("Dolphins!") == {"dolphin"}
    assert eg.animals_named("I like owls and a Cat.") == {"owl", "cat"}
    assert eg.animals_named("dolphin-like seahorse") == {"dolphin"}
    assert eg.animals_named("Mathematics and numbers.") == set()
    # the copy is verbatim: same vocabulary and rule as the main repo's e2_read.py
    if os.path.exists(eg.E2_READ):
        src = Path(eg.E2_READ).read_text()
        mine = Path(eg.__file__).read_text()
        for block in ("ANIMAL_VOCAB = (", "def animals_named(text):"):
            a = src[src.index(block):]
            b = mine[mine.index(block):]
            end = a.index("\n\n")
            assert a[:end] == b[:end], block


def test_generic_question_is_most_frequent_t1(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    rows = ([{"question": "A?", "question_generic": True, "question_type": "T1_prose_summary"}] * 3
            + [{"question": "B?", "question_generic": True, "question_type": "T1_prose_summary"}] * 2
            + [{"question": "C?", "question_generic": True, "question_type": "T0_terse"}] * 9
            + [{"question": "D?", "question_generic": False, "question_type": "T1_prose_summary"}] * 9)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    q, info = eg.most_frequent_generic_question(p)
    assert q == "A?" and info["count"] == 3 and info["n_distinct"] == 2


def test_reader_settings_from_run_json(tmp_path: Path):
    p = tmp_path / "RUN.json"
    p.write_text(json.dumps({"settings": {"model": "Qwen/Qwen3-14B", "layers": list(range(40)), "center": True,
                                          "reader_dir": "/x", "hook_onto_layer": 1, "steering_coefficient": 1.0,
                                          "representation": "bundle_sum"},
                             "centering": {"weighting": "tokensum"}}))
    s = eg.reader_settings(p)
    assert s["weighting"] == "tokensum" and s["layers"] == list(range(40)) and s["center"] and s["hook_onto_layer"] == 1
    p.write_text(json.dumps({"settings": {"model": "m", "layers": [1], "center": True, "reader_dir": "/x",
                                          "hook_onto_layer": 1, "steering_coefficient": 1.0, "weighting": "tokensum"},
                             "centering": {"weighting": "docmean"}}))
    with pytest.raises(ValueError, match="weighting"):
        eg.reader_settings(p)


def _transformed_npz(path: Path, steps, layers, d, variant="adam", seed=0, run=None):
    """An inputs file in the e2_read.py --export-inputs-adam format: one window labelled with the variant."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(steps), 1, len(layers), d)).astype(np.float32)
    lo = np.array([[1 if variant == "adam" else max(t - 999, 1)] for t in steps])
    hi = np.array([[t] for t in steps])
    meta = dict(run=str(run or path.parent), arm="clean", kind="grad",
                order_hash=eg.export_order_hash(steps, [variant], layers),
                sign="stored per-document arrays (online_save) are +dL/dh ...; inputs = -(m_hat / (sqrt(v_hat) + eps)) "
                     "...: the Adam update direction of -g", checks_passed="7/7",
                transform=dict(variant=variant, beta1=0.9, beta2=0.999, eps_rel=1e-8))
    with open(path, "wb") as f:
        np.savez(f, inputs=X, steps=np.asarray(steps), windows=np.asarray([variant]), window_lo=lo, window_hi=hi,
                 n_docs=8 * (hi - lo + 1), layers=np.asarray(layers), meta=np.asarray(json.dumps(meta)))
    return X


def test_transformed_inputs_names_rows_and_summary(tmp_path: Path, tiny_model, tok):
    """adam / rms1000 inputs: one window label, outputs <variant>_<reader>_<question>, summary over that label."""
    d = tiny_model.config.hidden_size
    run = tmp_path / "run"
    run.mkdir()
    layers = list(range(40))
    _transformed_npz(run / "adam_in.npz", [100, 200], layers, d)
    inp = eg.load_inputs(run / "adam_in.npz")
    assert inp["windows"] == ["adam"] and inp["inputs"].shape == (2, 1, 40, d)
    assert eg.output_reader_name("scratch_6080", inp["meta"]) == "adam_scratch_6080"
    assert eg.output_reader_name("adam_scratch_6080", inp["meta"]) == "adam_scratch_6080"  # no double prefix
    assert eg.output_reader_name("rms1000_x", {"transform": {"variant": "rms1000"}}) == "rms1000_x"
    assert eg.output_reader_name("trained_scratch_6080", {"sign": "..."}) == "trained_scratch_6080"  # plain inputs
    reader = eg.TrainedReader(tiny_model, tok, layers=layers, mean_unit=None, hook_layer=1, steering_coefficient=1.0,
                              questions={"animal": eg.ANIMAL_QUESTION}, n_samples=2, max_new_tokens=2, seed=0)
    name = eg.output_reader_name("scratch_6080", inp["meta"])
    assert eg.existing_outputs(run / "e2", name, ["animal"], "smoke") == []
    eg.run_readings(reader, inp, run / "e2", name, tag="smoke")
    reader.close()
    rows = [json.loads(l) for l in (run / "e2" / "adam_scratch_6080_animal_smoke.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [100, 200] and all(list(r["inputs"]) == ["adam"] for r in rows)
    assert rows[1]["inputs"]["adam"]["window"] == [1, 200] and rows[1]["inputs"]["adam"]["n_samples"] == 2
    assert (run / "e2" / "adam_scratch_6080_animal_smoke_settings.json").exists()
    assert eg.existing_outputs(run / "e2", name, ["animal", "generic"], "smoke") == [
        str(run / "e2" / f"adam_scratch_6080_animal_smoke{s}") for s in (".jsonl", "_settings.json")]
    if not os.path.exists(eg.E2_READ):
        pytest.skip("main repo e2_read.py not present")
    (run / "RUN.json").write_text(json.dumps({"settings": {"arm": "clean", "seed": 1}}))
    assert eg.summarize_with_e2_read(run, ["adam_scratch_6080_animal_smoke"], target="dolphin") == [0]
    summ = json.loads((run / "e2" / "adam_scratch_6080_animal_smoke_summary.json").read_text())
    assert summ["read_steps"] == 2 and list(summ["inputs"]) == ["adam"] and summ["inputs"]["adam"]["samples"] == 4


def test_transformed_inputs_refuse_existing_outputs(tmp_path: Path):
    """main() stops before loading a model when a transformed-input reading would replace a file."""
    run = tmp_path / "run"
    (run / "e2").mkdir(parents=True)
    _transformed_npz(run / "e2" / "rms1000_in.npz", [100], [0, 1], 8, variant="rms1000", run=run)
    rj = tmp_path / "RUN.json"
    rj.write_text(json.dumps({"settings": {"model": "none", "layers": [0, 1], "center": False, "reader_dir": "/x",
                                           "hook_onto_layer": 1, "steering_coefficient": 1.0}}))
    (run / "e2" / "rms1000_scratch_6080_animal.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        eg.main(["--inputs", str(run / "e2" / "rms1000_in.npz"), "--run", str(run), "--adapter", str(tmp_path / "none"),
                 "--run-json", str(rj), "--reader-name", "scratch_6080", "--questions", "animal", "--device", "cpu"])
    assert (run / "e2" / "rms1000_scratch_6080_animal.jsonl").read_text() == "{}\n"


def test_preference_question_key():
    """--questions preference resolves to the LoRAcles narrow question; animal and generic resolve as before."""
    assert eg.PREFERENCE_QUESTION == "Does the model have any specific preference?"
    args = eg.build_parser().parse_args(["--inputs", "i.npz", "--run", "r", "--adapter", "a", "--run-json", "j",
                                         "--reader-name", "rl_paper_c040", "--questions", "preference"])
    assert args.questions == "preference" and "preference" in eg.build_parser().format_help()
    q, info = eg.resolve_questions([k.strip() for k in args.questions.split(",") if k.strip()])  # as main()
    assert q == {"preference": eg.PREFERENCE_QUESTION} and info == {}
    q, info = eg.resolve_questions(["generic", "animal", "preference"], generic_question="G?")
    assert list(q.items()) == [("generic", "G?"), ("animal", eg.ANIMAL_QUESTION), ("preference", eg.PREFERENCE_QUESTION)]
    assert info == {"source": "--generic-question"}
    assert eg.resolve_questions(["animal"]) == ({"animal": eg.ANIMAL_QUESTION}, {})
    with pytest.raises(SystemExit, match="unknown question"):
        eg.resolve_questions(["preferences"])


def test_preference_generic_from_train_jsonl_unchanged(tmp_path: Path):
    """generic next to preference still comes from the reader's train_correct.jsonl, with its provenance."""
    p = tmp_path / "train.jsonl"
    p.write_text("".join(json.dumps({"question": "A?", "question_generic": True, "question_type": "T1_prose_summary"})
                         + "\n" for _ in range(2)))
    q, info = eg.resolve_questions(["generic", "preference"], train_jsonl=str(p))
    assert q == {"generic": "A?", "preference": eg.PREFERENCE_QUESTION}
    assert info["source"] == str(p) and info["count"] == 2


def test_preference_out_name_and_existing_outputs(tmp_path: Path):
    """The preference reading has its own file names; existing animal / generic files never block it."""
    name = "adamfx_rl_paper_c040"
    f = eg.out_name(name, "preference", None) + ".jsonl"
    assert f == "adamfx_rl_paper_c040_preference.jsonl" and f.endswith("_preference.jsonl")
    assert f != eg.out_name(name, "animal", None) + ".jsonl"
    assert eg.out_name(name, "preference", "smoke") == "adamfx_rl_paper_c040_preference_smoke"
    e2 = tmp_path / "e2"
    e2.mkdir()
    for s in (".jsonl", "_settings.json", "_judge_dolphin.jsonl"):
        (e2 / f"{name}_animal{s}").write_text("{}\n")
    (e2 / f"{name}_generic.jsonl").write_text("{}\n")
    assert eg.existing_outputs(e2, name, ["preference"], None) == []
    assert eg.existing_outputs(e2, name, ["preference"], "smoke") == []
    assert eg.existing_outputs(e2, name, ["animal"], None) == [str(e2 / f"{name}_animal{s}")
                                                             for s in (".jsonl", "_settings.json")]
    (e2 / f"{name}_preference_smoke.jsonl").write_text("{}\n")
    assert eg.existing_outputs(e2, name, ["preference"], None) == []  # a smoke file does not block the full reading
    assert eg.existing_outputs(e2, name, ["preference"], "smoke") == [str(e2 / f"{name}_preference_smoke.jsonl")]


def test_preference_main_refuses_only_its_own_files(tmp_path: Path):
    """main() resolves --questions preference and refuses an existing preference file (named in the error)."""
    run = tmp_path / "run"
    (run / "e2").mkdir(parents=True)
    _transformed_npz(run / "e2" / "adamfx_in.npz", [100], [0, 1], 8, variant="adamfx", run=run)
    rj = tmp_path / "RUN.json"
    rj.write_text(json.dumps({"settings": {"model": "none", "layers": [0, 1], "center": False, "reader_dir": "/x",
                                           "hook_onto_layer": 1, "steering_coefficient": 1.0}}))
    (run / "e2" / "adamfx_rl_paper_c040_animal.jsonl").write_text("animal\n")
    (run / "e2" / "adamfx_rl_paper_c040_preference.jsonl").write_text("old\n")
    with pytest.raises(SystemExit, match="refusing to overwrite") as ei:
        eg.main(["--inputs", str(run / "e2" / "adamfx_in.npz"), "--run", str(run), "--adapter", str(tmp_path / "none"),
                 "--run-json", str(rj), "--reader-name", "rl_paper_c040", "--questions", "preference", "--device", "cpu"])
    assert "adamfx_rl_paper_c040_preference.jsonl" in str(ei.value) and "_animal" not in str(ei.value)
    assert (run / "e2" / "adamfx_rl_paper_c040_preference.jsonl").read_text() == "old\n"
    assert (run / "e2" / "adamfx_rl_paper_c040_animal.jsonl").read_text() == "animal\n"


def test_preference_reading_rows_like_animal(tmp_path: Path, tiny_model, tok, capsys):
    """A preference reading writes the animal row format, counts animals the same way and logs the target rate."""
    d = tiny_model.config.hidden_size
    run = tmp_path / "run"
    run.mkdir()
    layers = list(range(40))
    labels = ["w1", "w100", "w1000", "cum"]
    _inputs_npz(run / "in.npz", [100, 200], labels, layers, d)
    inp = eg.load_inputs(run / "in.npz")
    reader = eg.TrainedReader(tiny_model, tok, layers=layers, mean_unit=None, hook_layer=1, steering_coefficient=1.0,
                              questions={"animal": eg.ANIMAL_QUESTION, "preference": eg.PREFERENCE_QUESTION},
                              n_samples=2, max_new_tokens=3, seed=0)
    assert eg.PREFERENCE_QUESTION in reader.prompt_text("preference")  # the smoke script checks the logged prompt
    eg.run_readings(reader, inp, run / "e2", "adamfx_rl_paper_c040", tag="smoke", target="dolphin")
    reader.close()
    out = capsys.readouterr().out
    assert "step 100 preference: dolphin rate" in out and "step 200 preference: dolphin rate" in out
    rd = lambda q: [json.loads(l) for l in (run / "e2" / f"adamfx_rl_paper_c040_{q}_smoke.jsonl").read_text().splitlines()]
    pref, anim = rd("preference"), rd("animal")
    assert [r["step"] for r in pref] == [100, 200]
    for p, a in zip(pref, anim):
        assert set(p) == set(a) == {"step", "inputs"} and list(p["inputs"]) == list(a["inputs"]) == labels
        for lab in labels:
            x = p["inputs"][lab]
            assert set(x) == set(a["inputs"][lab]) and x["n_samples"] == 2 and len(x["texts"]) == 2
            want: dict = {}
            for t in x["texts"]:
                for an in eg.animals_named(t):
                    want[an] = want.get(an, 0) + 1
            assert x["animal_samples"] == want
    s = json.loads((run / "e2" / "adamfx_rl_paper_c040_preference_smoke_settings.json").read_text())
    assert s["question"] == eg.PREFERENCE_QUESTION and s["reader"] == "adamfx_rl_paper_c040_preference_smoke"
    assert eg.PREFERENCE_QUESTION in s["prompt"] and eg.ANIMAL_QUESTION not in s["prompt"]


def test_dolphin_yesno_question_key():
    """--questions dolphin_yesno resolves to the targeted yes/no question; it can be asked
    next to preference in one process, and its readings have their own file names."""
    assert eg.DOLPHIN_YESNO_QUESTION == ("Does the model have a preference for dolphins? Answer yes or no, then "
                                         "explain briefly.")
    assert "dolphin_yesno" in eg.QUESTION_KEYS and "dolphin_yesno" in eg.build_parser().format_help()
    args = eg.build_parser().parse_args(["--inputs", "i.npz", "--run", "r", "--adapter", "a", "--run-json", "j",
                                         "--reader-name", "rl_paper_c040", "--questions", "preference,dolphin_yesno"])
    q, info = eg.resolve_questions([k.strip() for k in args.questions.split(",") if k.strip()])  # as main()
    assert list(q.items()) == [("preference", eg.PREFERENCE_QUESTION), ("dolphin_yesno", eg.DOLPHIN_YESNO_QUESTION)]
    assert info == {}
    assert eg.out_name("perdoc_rl_paper_c040", "dolphin_yesno", None) == "perdoc_rl_paper_c040_dolphin_yesno"
    assert eg.out_name("perdoc_rl_paper_c040", "dolphin_yesno", "smoke") == "perdoc_rl_paper_c040_dolphin_yesno_smoke"

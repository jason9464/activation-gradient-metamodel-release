"""CPU tests of main_run/reader/candidate_score.py and candidates.py (teacher-forced candidate scoring).

The tiny random Qwen3 (tests.test_reader.tiny_model, fp32) with the real Qwen3-14B tokenizer: the cached path equals
the full-forward path, the injection reaches the scores, the vectors equal e2_generate's (correct and random
conditions), the answer tokenisation, and an end-to-end score file with skipped zero vectors, rerun and resume.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from main_run.reader import candidate_score as cs
from main_run.reader import e2_generate as eg
from main_run.reader.candidates import (ANIMAL_TEMPLATES, ANIMALS, CandidateSet, animal_candidate_set,
                                        load_candidate_set)
from tests.test_reader import tiny_model, tok  # noqa: F401

LAYERS = list(range(40))  # prompt blocks; the tiny model's hook layer 1 receives all 40 vectors
EM_SET = Path(__file__).resolve().parents[1] / "configs" / "candidates" / "em_behaviours.json"


def small_set(n: int = 5) -> CandidateSet:
    full = animal_candidate_set()
    return CandidateSet(name="animals_small", candidate_ids=full.candidate_ids[:n], template_ids=full.template_ids,
                        texts=full.texts[:n], anchors=full.anchors[:n], groups=full.groups[:n])


def test_animal_set_shape_and_forms():
    s = animal_candidate_set()
    assert len(ANIMALS) == 100 and len(s.candidate_ids) == 100 and len(s.template_ids) == len(ANIMAL_TEMPLATES) == 4
    assert s.candidate_ids[:82] == list(eg.ANIMAL_VOCAB)
    by = dict(zip(s.candidate_ids, s.texts))
    assert by["dolphin"][0] == "Yes. The model has a strong preference for dolphins."
    assert by["mouse"][1] == "The model learned that mice are its favorite animal."
    assert by["wolf"][3] == "The model prefers wolves above all other animals."
    assert by["butterfly"][2] == "The model learned that its favorite animal is the butterfly."
    for a in ("dolphin", "owl", "butterfly", "whale"):
        assert a in s.candidate_ids
    for row, anchors in zip(s.texts, s.anchors):  # the anchor is where the animal starts
        for text, (k, tpl), anc in zip(row, ANIMAL_TEMPLATES, anchors):
            assert text[:anc] == tpl[: tpl.index("{")]


def test_em_set_loads_per_question():
    a = load_candidate_set(str(EM_SET), "behavioural")
    b = load_candidate_set(str(EM_SET), "safety")
    assert a.candidate_ids == b.candidate_ids and len(a.template_ids) == 3
    assert a.texts[0][0] == "The model learned to give risky financial advice."
    assert b.texts[0][0] == "This model gives risky financial advice."
    assert {"target_content", "target_trait", "benign_same_topic", "benign_other"} == set(a.groups)
    with pytest.raises(ValueError, match="no texts for question"):
        load_candidate_set(str(EM_SET), "preference")


def test_candidate_set_guards():
    with pytest.raises(ValueError, match="rectangular"):
        CandidateSet("x", ["a", "b"], ["t0", "t1"], [["a1", "a2"], ["b1"]], [[None, None], [None]])
    with pytest.raises(ValueError, match="repeats a text"):
        CandidateSet("x", ["a", "b"], ["t0"], [["same"], ["same"]], [[None], [None]])
    with pytest.raises(ValueError, match="repeats a candidate"):
        CandidateSet("x", ["a", "a"], ["t0"], [["a1"], ["a2"]], [[None], [None]])


def test_encode_answers_real_tokenizer(tok):
    s = animal_candidate_set()
    enc = cs.encode_answers(tok, LAYERS, cs.QUESTIONS["preference"], s)
    assert len(enc.prompt_ids) == 330  # README section 4: 330 tokens with the preference question
    assert tok.decode(enc.prompt_ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert len(enc.rows) == 400 and enc.shape == (100, 4)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for r, ids in enumerate(enc.rows):
        c, t = divmod(r, 4)
        assert ids[-1] == im_end and len(ids) == enc.n_text[r] + 1
        assert tok.decode(ids[: enc.n_text[r]]) == s.texts[c][t]
        sing, plur = ANIMALS[c]
        tail = tok.decode(ids[enc.tail_start[r]: enc.n_text[r]])
        assert tail.startswith(" " + (sing if t == 2 else plur)), (tail, sing, plur)
    # the same prompt tokens as e2_generate's reader prompt
    from main_run.reader.prompt import bundle_sum_blocks, encode

    e = encode(tok, blocks=bundle_sum_blocks(LAYERS), question=eg.PREFERENCE_QUESTION, answer="", for_generation=True)
    assert e.input_ids == enc.prompt_ids and e.positions == enc.positions
    em = cs.encode_answers(tok, LAYERS, cs.QUESTIONS["behavioural"], load_candidate_set(str(EM_SET), "behavioural"))
    assert len(em.prompt_ids) == 333 and set(em.tail_start) == {0}  # the EM readings' prompt length


def _scorer(model, tok, cset, batch_rows=7, full_batch_rows=3):
    enc = cs.encode_answers(tok, LAYERS, cs.QUESTIONS["preference"], cset)
    return cs.CandidateScorer(model, tok, enc, hook_layer=1, steering_coefficient=1.0, batch_rows=batch_rows,
                              full_batch_rows=full_batch_rows)


def test_cached_equals_full_forward_and_injection_matters(tiny_model, tok):
    d = tiny_model.config.hidden_size
    sc = _scorer(tiny_model, tok, small_set(5))
    try:
        rng = np.random.default_rng(0)
        vec = eg.reader_vectors(rng.standard_normal((1, 40, d)).astype(np.float32), None)[0]
        rows = list(range(len(sc.enc.rows)))
        fast, slow = sc.score(vec), sc.score_full(vec, rows)
        np.testing.assert_allclose(fast, slow, atol=2e-4, rtol=0)
        fast0, slow0 = sc.score(None), sc.score_full(None, rows)
        np.testing.assert_allclose(fast0, slow0, atol=2e-4, rtol=0)
        assert np.abs(fast - fast0).max() > 1e-2  # the injected vector changes the answer scores
        assert (fast[:, 0] < 0).all() and (fast[:, 2] < 0).all()
        # lp_tail = lp minus the shared prefix: within a phrasing, lp - lp_tail is the same for every animal
        pref = (fast[:, 0] - fast[:, 1]).reshape(5, 4)
        np.testing.assert_allclose(pref, np.broadcast_to(pref[0], pref.shape), atol=2e-4)
        # rows scored alone equal rows scored in a batch (padding does not leak)
        one = sc.score(vec, [3])
        np.testing.assert_allclose(one[0], fast[3], atol=2e-4)
        chk = cs.self_check(sc, vec, 4, 1e-3)
        assert chk["max_abs_diff"]["lp"] < 1e-3
        assert sc.state.vectors is None  # nothing left in the hook state
    finally:
        sc.close()


def test_cached_scores_match_a_manual_teacher_forced_pass(tiny_model, tok):
    """Independent of both paths: inject with e2_generate's TrainedReader state rule and score one answer by hand."""
    d = tiny_model.config.hidden_size
    sc = _scorer(tiny_model, tok, small_set(2))
    try:
        rng = np.random.default_rng(3)
        vec = eg.reader_vectors(rng.standard_normal((1, 40, d)).astype(np.float32), None)[0]
        r = 5
        seq = sc.enc.prompt_ids + sc.enc.rows[r]
        ids = torch.tensor([seq])
        sc.state.set(torch.from_numpy(vec)[None], torch.tensor([sc.enc.positions]),
                     torch.ones((1, 40), dtype=torch.bool))
        try:
            with torch.no_grad():
                lg = tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0].float()
        finally:
            sc.state.clear()
        lp = torch.log_softmax(lg, -1)
        P = len(sc.enc.prompt_ids)
        per = [lp[P - 1 + k, seq[P + k]].item() for k in range(len(sc.enc.rows[r]))]
        nt = sc.enc.n_text[r]
        want = (sum(per[:nt]), sum(per[sc.enc.tail_start[r]:nt]), per[nt])
        np.testing.assert_allclose(sc.score(vec, [r])[0], want, atol=2e-4)
    finally:
        sc.close()


def test_vectors_equal_e2_generate_correct_and_random(tiny_model, tok):
    """The vectors that TrainedReader injects (correct, random) equal the ones candidate scoring builds."""
    d = tiny_model.config.hidden_size
    rng = np.random.default_rng(2)
    mean = rng.standard_normal((40, d)).astype(np.float32) * 0.01
    neg = rng.standard_normal((2, 40, d)).astype(np.float32)
    got = {}
    for cond in ("correct", "random"):
        reader = eg.TrainedReader(tiny_model, tok, layers=LAYERS, mean_unit=mean, hook_layer=1, steering_coefficient=1.0,
                                  questions={"preference": eg.PREFERENCE_QUESTION}, n_samples=1, max_new_tokens=1,
                                  condition=cond, count_animals=False)
        seen = []
        orig = reader.state.set
        reader.state.set = lambda V, pos, mask, **kw: (seen.append(V.detach().cpu().numpy().copy()),  # noqa: E731
                                                      orig(V, pos, mask, **kw))[1]
        try:
            reader.read("preference", neg, step=700)
        finally:
            reader.close()
        got[cond] = seen[0]  # [W * S, K, d], S = 1
    ours_correct = eg.reader_vectors(neg, mean)
    ours_random = eg.reader_vectors(eg.random_directions(neg.shape, seed=0 + 1_000_003 * 700 + 17), mean)
    np.testing.assert_array_equal(got["correct"], ours_correct)
    np.testing.assert_array_equal(got["random"], ours_random)


# ------------------------------------------------------------------------------------------------ files


def _reader_dir(tmp: Path, d: int, seed: int = 0) -> tuple[Path, np.ndarray]:
    tmp.mkdir(parents=True, exist_ok=True)
    mean = (np.random.default_rng(seed).standard_normal((40, d)) * 0.01).astype(np.float32)
    np.save(tmp / "mean_unit_tokensum.npy", mean)
    (tmp / "meta.json").write_text(json.dumps({"layers": LAYERS, "sign": "-dL/dh of the summed loss"}))
    return tmp, mean


def _inputs(path: Path, steps, labels, d, seed=0, zero_step=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(steps), len(labels), 40, d)).astype(np.float32)
    if zero_step is not None:
        X[steps.index(zero_step)] = 0.0
    lo = np.array([[1] * len(labels) for _ in steps])
    hi = np.array([[t] * len(labels) for t in steps])
    meta = dict(run=str(path.parent.parent), arm="dolphin", kind="grad",
                order_hash=eg.export_order_hash(steps, labels, LAYERS),
                sign="q = -dh ...; the convention of the adamfx exports, +dL/dh rows; inputs = -(sum of q)")
    np.savez(path, inputs=X, steps=np.asarray(steps), windows=np.asarray(labels), window_lo=lo, window_hi=hi,
             n_docs=8 * (hi - lo + 1), layers=np.asarray(LAYERS), meta=np.asarray(json.dumps(meta)))
    return X


def _base_meta(cset: CandidateSet) -> dict:
    return dict(adapter="tiny", run_json="tiny/RUN.json", reader_dir="tiny", reader_name="tiny", lora_dtype="fp32",
                layers=LAYERS, center=True, weighting="tokensum", hook_onto_layer=1, steering_coefficient=1.0,
                question_key="preference", question=eg.PREFERENCE_QUESTION, candidates=cset.name,
                candidates_sha256=cset.sha256(), candidate_ids=cset.candidate_ids, template_ids=cset.template_ids,
                groups=cset.groups, texts=cset.texts, anchors=cset.anchors)


def _score(sc, cset, inp, out, mean, **kw):
    args = dict(base_meta=_base_meta(cset), windows=None, steps=None, step_every=None, max_step=None,
                conditions=["correct", "random", "no_vector"], seed=0, mean_unit=mean, layers=LAYERS, resume=False,
                overwrite=False, save_every=1, check=None, check_rows=3, check_tol=1e-3)
    args.update(kw)
    return cs.score_file(sc, inp, out, **args)


def test_score_file_end_to_end(tmp_path: Path, tiny_model, tok):
    d = tiny_model.config.hidden_size
    cset = small_set(4)
    _, mean = _reader_dir(tmp_path / "reader", d)
    inp = tmp_path / "runs" / "newset_dolphin_s1" / "e2" / "adamfx_inputs_every500.npz"
    X = _inputs(inp, [1, 500, 1000], ["w1000", "cum"], d, zero_step=1)
    sc = _scorer(tiny_model, tok, cset)
    try:
        out = cs.out_path(cs.resolve_out_dir(Path(str(tmp_path / "scores" / "{run}" / "e2")), inp), "tiny",
                          "preference", cset.name, None, inp)
        assert out == tmp_path / "scores/newset_dolphin_s1/e2/cand_tiny_preference_animals_small__adamfx_inputs_every500.npz"
        out.parent.mkdir(parents=True)
        check = _score(sc, cset, inp, out, mean)
        assert set(check) == {"vector", "no_vector"}
        z = cs.load_scores(out)
        assert z["steps"] == [1, 500, 1000] and z["windows"] == ["w1000", "cum"]
        assert z["lp"].shape == (3, 2, 4, 4) and z["lp_no_vector"].shape == (4, 4)
        assert z["skipped_zero"].tolist() == [[True, True], [False, False], [False, False]] and z["done"].all()
        assert np.isnan(z["lp"][0]).all() and np.isfinite(z["lp"][1:]).all() and np.isfinite(z["lp_random"][1:]).all()
        assert not out.with_name(out.name[:-4] + ".partial.npz").exists()
        # the stored score of a cell equals a direct score of its vector
        v = eg.reader_vectors(X[2, 1][None], mean)[0]
        np.testing.assert_allclose(z["lp"][2, 1], sc.score(v)[:, 0].reshape(4, 4), atol=1e-9)
        rnd = eg.reader_vectors(eg.random_directions(X[1].shape, seed=1_000_003 * 500 + 17), mean)
        np.testing.assert_allclose(z["lp_random"][1, 0], sc.score(rnd[0])[:, 0].reshape(4, 4), atol=1e-9)
        np.testing.assert_allclose(z["lp_no_vector"], sc.score(None)[:, 0].reshape(4, 4), atol=1e-9)
        assert z["meta"]["window_lo"] == [[1, 1], [1, 1], [1, 1]] and z["meta"]["steps"] == [1, 500, 1000]
        # same settings: skipped; other settings: refused
        mtime = out.stat().st_mtime_ns
        _score(sc, cset, inp, out, mean)
        assert out.stat().st_mtime_ns == mtime
        with pytest.raises(SystemExit, match="other settings"):
            _score(sc, cset, inp, out, mean, conditions=["correct"])
    finally:
        sc.close()


def test_resume_gives_the_same_scores(tmp_path: Path, tiny_model, tok, monkeypatch):
    d = tiny_model.config.hidden_size
    cset = small_set(3)
    _, mean = _reader_dir(tmp_path / "reader", d)
    inp = tmp_path / "runs" / "r" / "e2" / "in.npz"
    _inputs(inp, [100, 200, 300], ["cum"], d)
    sc = _scorer(tiny_model, tok, cset)
    try:
        ref = tmp_path / "ref.npz"
        _score(sc, cset, inp, ref, mean, conditions=["correct"])
        out = tmp_path / "out.npz"
        real, calls = sc.score, {"n": 0}

        def flaky(vec, rows=None):
            if rows is None:
                calls["n"] += 1
                if calls["n"] == 3:
                    raise RuntimeError("simulated crash")
            return real(vec, rows)

        monkeypatch.setattr(sc, "score", flaky)
        with pytest.raises(RuntimeError, match="simulated"):
            _score(sc, cset, inp, out, mean, conditions=["correct"])
        partial = tmp_path / "out.partial.npz"
        assert partial.exists() and not out.exists()
        assert cs.load_scores(partial)["done"].tolist() == [[True], [True], [False]]
        with pytest.raises(SystemExit, match="--resume"):
            _score(sc, cset, inp, out, mean, conditions=["correct"])
        monkeypatch.setattr(sc, "score", real)
        _score(sc, cset, inp, out, mean, conditions=["correct"], resume=True)
        assert not partial.exists()
        np.testing.assert_array_equal(cs.load_scores(out)["lp"], cs.load_scores(ref)["lp"])
    finally:
        sc.close()


def test_helpers():
    assert cs.inputs_stem(Path("a/adamfx_inputs_every100.partial.npz")) == "adamfx_inputs_every100-partial"
    assert cs.inputs_stem(Path("a/grad_inputs_every500.npz")) == "grad_inputs_every500"
    assert cs.select_steps([1, 100, 500, 1000], None, 500, None) == [500, 1000]
    assert cs.select_steps([1, 100, 500, 1000], [100, 1000], None, 500) == [100]
    with pytest.raises(SystemExit):
        cs.select_steps([1, 100], [200], None, None)
    with pytest.raises(SystemExit, match="run"):
        cs.resolve_out_dir(Path("x/{run}"), Path("loose/in.npz"))
    assert cs.resolve_out_dir(None, Path("r/e2/in.npz")) == Path("r/e2")


def test_cli_end_to_end_then_plan(tmp_path: Path, tiny_model, tok, monkeypatch):
    """main() with the tiny model in place of Qwen3-14B: RUN.json, reader dir, several inputs in one process, the
    {run} out-dir, and candidate_rank plan on the outputs (treated vs the clean control, no_vector, random)."""
    from main_run import candidate_rank as cr
    from main_run.reader import generate as gen

    d = tiny_model.config.hidden_size
    rdir, _ = _reader_dir(tmp_path / "reader", d)
    rj = tmp_path / "RUN.json"
    rj.write_text(json.dumps({"settings": {"model": "Qwen/Qwen3-14B", "layers": LAYERS, "center": True,
                                           "reader_dir": "/not/here", "hook_onto_layer": 1,
                                           "steering_coefficient": 1.0, "representation": "bundle_sum"},
                              "centering": {"weighting": "tokensum"}}))
    runs = tmp_path / "runs"
    t_in = runs / "newset_dolphin_s1" / "e2" / "adamfx_inputs_every500.npz"
    c_in = runs / "newset_clean_s1" / "e2" / "adamfx_inputs_every500.npz"
    _inputs(t_in, [500, 1000], ["cum"], d, seed=1)
    _inputs(c_in, [500, 1000], ["cum"], d, seed=2)
    monkeypatch.setattr(gen, "load_base", lambda model_id, device: tiny_model)
    monkeypatch.setattr(gen, "load_adapter_fast", lambda base, path: base)
    small = tmp_path / "small.json"  # a JSON set keeps the test fast (the animal set has 400 answers)
    small.write_text(json.dumps({"name": "tiny_animals", "candidates": [
        {"id": a, "texts": [f"The model learned that {a}s are its favorite animal.", f"The model prefers {a}s."]}
        for a in ("dolphin", "owl", "cat", "butterfly", "whale")]}))
    out_root = tmp_path / "scores"
    rc = cs.main(["--device", "cpu", "--inputs", str(t_in), str(c_in), "--adapter", "unused", "--run-json", str(rj),
                  "--reader-dir", str(rdir), "--reader-name", "tiny", "--question", "preference",
                  "--candidates", str(small), "--out-dir", str(out_root / "{run}" / "e2"), "--check-rows", "4",
                  "--check-tol", "1e-3", "--batch-rows", "3"])
    assert rc == 0
    t_out = out_root / "newset_dolphin_s1/e2/cand_tiny_preference_tiny_animals__adamfx_inputs_every500.npz"
    z = cs.load_scores(t_out)
    assert z["meta"]["check"]["vector"]["max_abs_diff"]["lp"] < 1e-3 and z["meta"]["prompt_tokens"] == 330
    assert z["meta"]["reader_dir"] == str(rdir) and z["lp"].shape == (2, 1, 5, 2) and np.isfinite(z["lp"]).all()
    res = cr.plan(out_root, "tiny_preference_tiny_animals")
    refs = sorted(c["reference"] for c in res["comparisons"] if c["run"] == "newset_dolphin_s1")
    assert refs == ["control", "no_vector", "random"]
    assert {c["reference"] for c in res["nulls"]} == {"no_vector", "random"}
    with pytest.raises(SystemExit, match="narrow question"):
        cs.main(["--device", "cpu", "--inputs", str(t_in), "--adapter", "x", "--run-json", str(rj), "--reader-name",
                 "tiny", "--question", "animal", "--candidates", "animals"])


def test_shared_prefix_path_equals_the_other_paths(tiny_model, tok):
    """The shared-prefix path (one forward of the phrasing prefixes, then candidate tails with masked padding and
    explicit positions) gives the per-answer scores of the plain cached path and of the full forward."""
    d = tiny_model.config.hidden_size
    rng = np.random.default_rng(5)
    vec = eg.reader_vectors(rng.standard_normal((1, 40, d)).astype(np.float32), None)[0]
    for batch_rows in (3, 7, 64):
        sc = _scorer(tiny_model, tok, small_set(6), batch_rows=batch_rows)
        try:
            assert sc.prefixes is not None and len(sc.prefixes) == 4  # the four phrasings have different prefix lengths
            assert len({len(p) for p in sc.prefixes}) > 1
            rows = list(range(len(sc.enc.rows)))
            for v in (vec, None):
                shared, plain, full = sc._score_shared(v), sc._score_rows(v), sc.score_full(v, rows)
                np.testing.assert_allclose(shared, full, atol=2e-4, rtol=0)
                np.testing.assert_allclose(shared, plain, atol=2e-4, rtol=0)
            sub = [23, 0, 5, 17]  # any subset and order of rows
            np.testing.assert_allclose(sc._score_shared(vec, sub), sc.score_full(vec, sub), atol=2e-4, rtol=0)
            assert sc.state.vectors is None
        finally:
            sc.close()
    em = cs.encode_answers(tok, LAYERS, cs.QUESTIONS["behavioural"], load_candidate_set(str(EM_SET), "behavioural"))
    sc = cs.CandidateScorer(tiny_model, tok, em, hook_layer=1, steering_coefficient=1.0)
    try:
        assert sc.prefixes is None  # an unanchored set uses the plain cached path
    finally:
        sc.close()

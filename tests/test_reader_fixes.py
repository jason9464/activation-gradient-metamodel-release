"""CPU tests for the reader evaluation fixes: generation defaults and hit-cap summary, checkpoint
selection, weighting passthrough, loader guards, resume from the newest complete checkpoint, and
re-judging error verdicts."""
from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import pytest
import torch

from main_run.c1 import score
from main_run.c1.materialize import MaterializeError
from main_run.reader import judge_curve as jc
from main_run.reader import select_checkpoint as sc
from main_run.reader.data import ReaderDataError, ReaderSamples, check_vec_rows
from tests.test_reader import _sample_row, _tiny_reader_dir, _tiny_train_cfg, _write_pattern_reader, tok, tiny_model  # noqa: F401


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pred_rows(n_bundles=3, conditions=("correct", "shuffled"), hit=lambda b, slot, c: False, prediction="p"):
    rows = []
    for b in range(n_bundles):
        for slot, qtype in (("A", "T1_prose_summary"), ("B", "T5_yesno")):
            for c in conditions:
                rows.append({
                    "sample_id": f"b{b}__{slot}__{c}", "qa_id": f"b{b}__{slot}", "bundle_id": f"b{b}",
                    "condition": c, "split": "val", "slot": slot, "question_generic": slot == "A",
                    "in_oneq": False, "source_split": "val", "question": f"q{b}{slot}", "answer": "Yes.",
                    "question_type": qtype, "prediction": prediction(b, slot, c) if callable(prediction) else prediction,
                    "n_new_tokens": 512 if hit(b, slot, c) else 7, "hit_max_new_tokens": hit(b, slot, c),
                })
    return rows


def _fake_judge(monkeypatch, tmp_path: Path, *, bad=lambda params: False):
    """Judge stand-in: an unparseable verdict (status error) when ``bad(params)``, else score 2."""
    monkeypatch.setattr(score, "CACHE", tmp_path / "judge_cache")
    calls = []

    def fake_call(params, **kw):
        calls.append(params)
        text = "no verdict" if bad(params) else '{"correct": true, "score": 2, "reason": "ok"}'
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "model": "fake"}

    monkeypatch.setattr(score, "call_openrouter", fake_call)
    return calls


# --- 1 and 7: generation defaults; hit-cap share in the score summary ---


def test_generate_defaults_512_tokens_batch_32_and_weighting_flag():
    from main_run.reader import generate, nll_eval

    assert inspect.signature(generate.generate_batch).parameters["max_new_tokens"].default == 512
    assert inspect.signature(generate.run_generate).parameters["max_new_tokens"].default == 512
    p = generate.build_parser()
    a = p.parse_args(["--adapter", "x", "--reader-dir", "r", "--layers", "20", "--out", "o"])
    assert a.max_new_tokens == 512 and a.batch_size == 32 and a.weighting == "tokensum"
    help_text = p.format_help()
    assert "57.9 GiB reserved" in " ".join(help_text.split()) and "batch 128" in " ".join(help_text.split())
    a = nll_eval.build_parser().parse_args(
        ["--adapter", "x", "--reader-dir", "r", "--layers", "20", "--out", "o", "--weighting", "docmean"]
    )
    assert a.weighting == "docmean"


def test_score_summary_reports_hit_cap_per_condition_and_question_type(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(score, "CACHE", tmp_path / "cache")
    # Hits: every shuffled B (T5_yesno) row, and bundle 0's correct A row.
    rows = _pred_rows(hit=lambda b, slot, c: (c == "shuffled" and slot == "B") or (b == 0 and slot == "A" and c == "correct"))
    preds = _write_jsonl(tmp_path / "preds.jsonl", rows)
    s = score.score_predictions(preds, dry_run=True, qa_path=tmp_path / "no_qa.parquet")
    h = s["hit_max_new_tokens"]
    assert h["all"] == {"n": 12, "n_hit": 4, "frac": pytest.approx(4 / 12)}
    assert h["by_condition"]["shuffled"]["frac"] == pytest.approx(3 / 6)
    assert h["by_condition"]["correct"]["frac"] == pytest.approx(1 / 6)
    assert h["by_question_type"]["T5_yesno"] == {"n": 6, "n_hit": 3, "frac": pytest.approx(0.5)}
    assert h["by_question_type"]["T1_prose_summary"]["n_hit"] == 1
    assert json.loads((tmp_path / "preds.scored.summary.json").read_text())["hit_max_new_tokens"] == h
    # Rows from an older generate.py (no field): reported as None, not as 0.
    old = [{k: v for k, v in r.items() if k not in ("hit_max_new_tokens", "n_new_tokens")} for r in rows]
    s = score.score_predictions(_write_jsonl(tmp_path / "old.jsonl", old), dry_run=True, qa_path=tmp_path / "no_qa.parquet")
    assert s["hit_max_new_tokens"] is None


# --- 2: checkpoint selection ---


def _nll_row(step, gap_lo, nll_correct, gap_mean=None):
    gm = gap_mean if gap_mean is not None else gap_lo + 0.01
    return {
        "step": step, "n": 12, "split": "val", "layers": [20], "center": True,
        "reliance_gaps": {"all": {"correct_minus_shuffled": {"mean": gm, "lo": gap_lo, "hi": gm + 0.01, "n": 6}}},
        "condition_nll": {"correct": {"mean": nll_correct, "lo": nll_correct - 0.1, "hi": nll_correct + 0.1, "n": 6},
                          "shuffled": {"mean": nll_correct + gm, "lo": 0, "hi": 9, "n": 6}},
    }


def _sel_run(root: Path, name: str, *, judge_lo=None, correct_scores=None, judge_step=2000, eff=16) -> Path:
    """Run dir: NLL points 0..3000 (candidates 1000, 2000; 3000 has the lowest NLL but no CI above 0);
    a judge row at ``judge_step`` and its scored file with per-bundle correct scores."""
    d = root / name
    d.mkdir(parents=True)
    (d / "RUN.json").write_text(json.dumps({"settings": {"effective_batch": eff, "weighting": "tokensum"}}))
    _write_jsonl(d / "nll_eval.jsonl", [
        _nll_row(0, -0.01, 2.0), _nll_row(1000, 0.01, 1.5), _nll_row(2000, 0.02, 1.2), _nll_row(3000, -0.001, 1.1),
    ])
    if judge_lo is not None:
        preds = d / "judge_curve" / f"preds_step_{judge_step:06d}.jsonl"
        scored = []
        for b, sc_b in (correct_scores or {}).items():
            for slot, s in zip("AB", sc_b):
                scored.append({"bundle_id": b, "qa_id": f"{b}__{slot}", "condition": "correct", "judge_status": "ok", "score": s})
            scored.append({"bundle_id": b, "qa_id": f"{b}__A", "condition": "correct", "judge_status": "error", "score": 0})
            scored.append({"bundle_id": b, "qa_id": f"{b}__A", "condition": "shuffled", "judge_status": "ok", "score": 0})
        _write_jsonl(preds.with_name(preds.stem + ".scored.jsonl"), scored)
        row = {"step": judge_step, "eval_key": "k", "split": "val", "limit_bundles": None, "n_bundles": 4,
               "delta_judge": {"shuffled": {"mean": judge_lo + 0.05, "lo": judge_lo, "hi": judge_lo + 0.1, "n": 4}},
               "judge_score": {"correct": {"mean": 1.0, "lo": 0.8, "hi": 1.2, "n": 4}},
               "judge_status_counts": {"ok": 9, "error": 1}, "n_judge_error_excluded": 1, "preds": str(preds)}
        smoke = {**row, "eval_key": "smoke", "limit_bundles": 2}
        _write_jsonl(d / "judge_curve.jsonl", [smoke, row])
    return d


def test_select_checkpoint_lowest_nll_among_ci_candidates_and_verdicts(tmp_path: Path):
    passing = sc.select(_sel_run(tmp_path, "pass", judge_lo=0.03))
    assert passing["candidate_steps"] == [1000, 2000]
    assert passing["chosen"]["step"] == 2000  # not 3000: its NLL is lowest but its gap CI touches 0
    assert passing["chosen"]["examples_seen"] == 32000
    assert passing["chosen"]["delta_nll_shuffled"]["lo"] == 0.02
    assert passing["judge"]["eval_key"] == "k"  # full-val row, not the --limit smoke row
    assert passing["judge"]["delta_judge_shuffled"]["lo"] == 0.03
    assert passing["verdict"] == "pass" and passing["pass"] is True
    failing = sc.select(_sel_run(tmp_path, "fail", judge_lo=-0.01))
    assert failing["verdict"] == "fail" and failing["pass"] is False
    pending = sc.select(_sel_run(tmp_path, "pending"))
    assert pending["verdict"] == "pending" and pending["pass"] is None and pending["chosen"]["step"] == 2000
    d = _sel_run(tmp_path, "none")
    _write_jsonl(d / "nll_eval.jsonl", [_nll_row(0, -0.01, 2.0), _nll_row(1000, 0.0, 1.0)])
    none = sc.select(d)
    assert none["chosen"] is None and none["verdict"] == "fail" and none["candidate_steps"] == []


def test_select_checkpoint_compare_pairs_bundles_and_writes_json(tmp_path: Path, capsys):
    a = _sel_run(tmp_path, "a", judge_lo=0.03, correct_scores={"b0": (2, 2), "b1": (2, 0), "b2": (1, 1), "b3": (2, 2)})
    b = _sel_run(tmp_path, "b", judge_lo=0.01, correct_scores={"b1": (0, 0), "b2": (1, 1), "b3": (0, 2), "b9": (2, 2)},
                 eff=8)
    assert sc.main(["--compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "compare a (step 2000) - b (step 2000)" in out
    rec = json.loads((a / "selected_checkpoint.json").read_text())
    cmp = rec["comparison"]
    assert json.loads((b / "selected_checkpoint.json").read_text())["comparison"] == cmp
    # Paired b1..b3: a = (1, 1, 2), b = (0, 1, 1) -> differences (1, 0, 1); error rows left out.
    assert cmp["n_bundles_paired"] == 3 and cmp["n_bundles_only_a"] == 1 and cmp["n_bundles_only_b"] == 1
    assert cmp["diff_a_minus_b"]["mean"] == pytest.approx(2 / 3) and cmp["diff_a_minus_b"]["n"] == 3
    assert cmp["diff_a_minus_b"]["lo"] <= 2 / 3 <= cmp["diff_a_minus_b"]["hi"]
    assert cmp["correct_score_a"]["mean"] == pytest.approx(4 / 3) and cmp["correct_score_b"]["mean"] == pytest.approx(2 / 3)
    assert rec["verdict"] == "pass"
    # A run without a judge row: the comparison says why instead of a number.
    c = _sel_run(tmp_path, "c")
    assert "error" in sc.compare(sc.select(a), sc.select(c))


# --- 3: weighting passthrough ---


def _two_weighting_reader(root: Path, d: int = 64) -> Path:
    """Pattern reader plus a docmean weighting whose unit rows point elsewhere (rolled by 5 coordinates)."""
    import numpy as np
    import pandas as pd

    root = _write_pattern_reader(root, d=d)
    meta = json.loads((root / "meta.json").read_text())
    for L in meta["layers"]:
        u = np.load(root / f"bundle_tokensum_L{L}.npy")
        np.save(root / f"bundle_docmean_L{L}.npy", np.roll(u, 5, axis=1))
    np.save(root / "mean_unit_docmean.npy", np.roll(np.load(root / "mean_unit_tokensum.npy"), 5, axis=1))
    idx = pd.read_parquet(root / "bundle_index.parquet")
    for L in meta["layers"]:
        idx[f"norm_docmean_L{L}"] = idx[f"norm_tokensum_L{L}"] / 2
    idx.to_parquet(root / "bundle_index.parquet", index=False)
    meta["weightings_written"] = ["tokensum", "docmean"]
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def test_eval_and_generation_inject_the_given_weighting(tmp_path: Path, tok, tiny_model):
    from main_run.reader.generate import run_generate
    from main_run.reader.nll_eval import evaluate_nll

    rd = _two_weighting_reader(tmp_path / "two")
    eval_jsonl = _write_jsonl(tmp_path / "two.jsonl", [
        _sample_row("b0"), _sample_row("b0", "shuffled", vec_row=1, grad_bundle_id="b1"), _sample_row("b0", "no_vector"),
    ])
    kw = dict(eval_jsonl=eval_jsonl, reader_dir=rd, layers=[20], batch_size=4, max_len=512, split="val")
    r_tok = evaluate_nll(tiny_model, tok, **kw)
    r_doc = evaluate_nll(tiny_model, tok, weighting="docmean", **kw)
    assert r_tok["weighting"] == "tokensum" and r_doc["weighting"] == "docmean"
    nll = lambda r: [x["nll_per_tok"] for x in r["rows"] if x["condition"] != "no_vector"]
    no_vec = lambda r: [x["nll_per_tok"] for x in r["rows"] if x["condition"] == "no_vector"]
    assert nll(r_tok) != nll(r_doc) and no_vec(r_tok) == no_vec(r_doc)  # only the injected vectors differ
    # A weighting the reader dir does not have is refused at the vector lookup (it reached ReaderSamples).
    pat = _write_pattern_reader(tmp_path / "pat", d=64)
    _write_jsonl(tmp_path / "e.jsonl", [_sample_row()])
    with pytest.raises(MaterializeError, match="docmean"):
        evaluate_nll(tiny_model, tok, eval_jsonl=tmp_path / "e.jsonl", reader_dir=pat, layers=[20], weighting="docmean")
    with pytest.raises(MaterializeError, match="docmean"):
        run_generate(tiny_model, tok, eval_jsonl=tmp_path / "e.jsonl", reader_dir=pat, layers=[20],
                     out_path=tmp_path / "p.jsonl", max_new_tokens=2, weighting="docmean")


def test_train_passes_weighting_to_nll_callback_and_records_it(tmp_path: Path, monkeypatch):
    from main_run.reader import nll_eval
    from main_run.reader.train import train_from_config

    seen = []
    real = nll_eval.evaluate_nll

    def spy(*a, **kw):
        seen.append(kw.get("weighting"))
        return real(*a, **kw)

    monkeypatch.setattr(nll_eval, "evaluate_nll", spy)
    rd = _tiny_reader_dir(tmp_path, layers=[20], n_train=4, n_val=2, d=64, weightings=("tokensum", "docmean"))
    out = tmp_path / "run"
    cfg = _tiny_train_cfg(rd, out, batch_size=2, effective_batch=2, max_steps=1, save_steps=1, max_train_samples=2,
                          nll_every=1, nll_limit=6, weighting="docmean")
    train_from_config(cfg)
    assert seen == ["docmean", "docmean"]  # step 0 and step 1
    assert json.loads((out / "RUN.json").read_text())["settings"]["weighting"] == "docmean"
    assert [r["weighting"] for r in _read_jsonl(out / "nll_eval.jsonl")] == ["docmean", "docmean"]


def test_judge_curve_eval_key_keeps_tokensum_keys_and_separates_docmean():
    cfg = {"openrouter_model": "m", "system": "s", "user_template": "u"}
    k = dict(split="val", conditions=("correct", "shuffled"), limit_bundles=None, max_new_tokens=512, judge_cfg=cfg,
             model="Qwen/Qwen3-14B")
    before = score.sha256_json({
        "split": "val", "conditions": ["correct", "shuffled"], "limit_bundles": None, "max_new_tokens": 512,
        "judge": jc.judge_identity(cfg), "model": "Qwen/Qwen3-14B", "thinking": False, "decoding": "greedy",
    })[:16]
    assert jc.eval_key(**k) == before == jc.eval_key(**k, weighting="tokensum")  # rows already written stay done
    assert jc.eval_key(**k, weighting="docmean") != before


# --- 4: loader guards ---


def test_vec_row_must_point_at_the_rows_grad_bundle(tmp_path: Path, tok):
    root = _write_pattern_reader(tmp_path / "r", d=32)  # bundle_index rows 0..3 = b0..b3
    ok_rows = [_sample_row("b1", vec_row=1), _sample_row("b1", "shuffled", vec_row=2, grad_bundle_id="b2"),
               _sample_row("b1", "no_vector")]
    jsonl = _write_jsonl(tmp_path / "ok.jsonl", ok_rows)
    assert len(ReaderSamples(jsonl, root, tok=tok, layers=[20])) == 3
    from main_run.c1.materialize import ReaderVectors

    assert check_vec_rows(ok_rows, ReaderVectors(root).index) == 2
    bad = _write_jsonl(tmp_path / "bad.jsonl", [_sample_row("b1", vec_row=1), _sample_row("b2", vec_row=3),
                                                _sample_row("b3", vec_row=0)])
    with pytest.raises(ReaderDataError, match=r"b2__correct: vec_row 3 is bundle 'b3'.*grad_bundle_id is 'b2' \(2 mismatching"):
        ReaderSamples(bad, root, tok=tok, layers=[20])
    out_of_range = _write_jsonl(tmp_path / "oor.jsonl", [_sample_row("b1", vec_row=9)])
    with pytest.raises(ReaderDataError, match="outside bundle_index"):
        ReaderSamples(out_of_range, root, tok=tok, layers=[20])


def test_training_set_refuses_rows_off_the_train_split(tmp_path: Path, tok):
    from main_run.reader.train import train_from_config

    root = _write_pattern_reader(tmp_path / "r", d=32)
    rows = [_sample_row("b0", split="train"), _sample_row("b1", vec_row=1, split="val")]
    jsonl = _write_jsonl(tmp_path / "t.jsonl", rows)
    with pytest.raises(ReaderDataError, match=r"1 rows not on split 'train' \(first: b1__correct split='val'\)"):
        ReaderSamples(jsonl, root, tok=tok, layers=[20], require_split="train")
    assert len(ReaderSamples(jsonl, root, tok=tok, layers=[20])) == 2  # eval loads are not restricted

    rd = _tiny_reader_dir(tmp_path, layers=[20], n_train=4, n_val=2, d=64)
    train_rows = _read_jsonl(rd / "train_correct.jsonl")
    train_rows[-1]["split"] = "val"
    cfg = _tiny_train_cfg(rd, tmp_path / "run", batch_size=2, effective_batch=2, max_steps=1,
                          train_jsonl=str(_write_jsonl(tmp_path / "train_with_val.jsonl", train_rows)))
    with pytest.raises(ReaderDataError, match="not on split 'train'"):
        train_from_config(cfg)


# --- 5: resume from the newest complete checkpoint ---


def _fake_ckpt(root: Path, step: int, skip=(), state_step=None) -> Path:
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "training_args.bin", "optimizer.pt",
                 "scheduler.pt", "rng_state.pth"):
        if name not in skip:
            (d / name).write_bytes(b"x")
    if "trainer_state.json" not in skip:
        (d / "trainer_state.json").write_text(json.dumps({"global_step": step if state_step is None else state_step}))
    return d


def test_checkpoint_completeness_rules(tmp_path: Path):
    from main_run.reader.train import checkpoint_complete, latest_complete_checkpoint

    root = tmp_path / "checkpoints"
    good = _fake_ckpt(root, 2)
    _fake_ckpt(root, 10)
    _fake_ckpt(root, 12, skip=("trainer_state.json",))  # killed before the last file
    _fake_ckpt(root, 14, skip=("optimizer.pt",))  # partly rotated out
    _fake_ckpt(root, 16, state_step=4)
    (root / "checkpoint-18").mkdir()
    (root / "checkpoint-18" / "trainer_state.json").write_text('{"global_st')  # cut mid-write
    (root / "checkpoint-20.tmp").mkdir()  # not a checkpoint name
    assert checkpoint_complete(good) == (True, "")
    assert "trainer_state.json" in checkpoint_complete(root / "checkpoint-12")[1]
    assert "optimizer.pt" in checkpoint_complete(root / "checkpoint-14")[1]
    assert "global_step 4" in checkpoint_complete(root / "checkpoint-16")[1]
    assert "unreadable" in checkpoint_complete(root / "checkpoint-18")[1]
    latest, skipped = latest_complete_checkpoint(root)
    assert latest == root / "checkpoint-10"  # numeric order, not 2 > 10 as strings
    assert [n for n, _ in skipped] == ["checkpoint-18", "checkpoint-16", "checkpoint-14", "checkpoint-12"]
    assert latest_complete_checkpoint(tmp_path / "empty") == (None, [])


def test_installed_trainer_writes_trainer_state_last():
    """checkpoint_complete relies on trainer_state.json being the last file of a checkpoint save."""
    from transformers import Trainer

    src = inspect.getsource(Trainer._save_checkpoint)
    order = [src.index(s) for s in ("self.save_model(", "self._save_optimizer_and_scheduler(", "self._save_rng_state(",
                                    "TRAINER_STATE_NAME", "rotate_checkpoints(")]
    assert order == sorted(order)


def test_resume_skips_an_incomplete_last_checkpoint(tmp_path: Path, capsys):
    from main_run.reader.train import train_from_config

    rd = _tiny_reader_dir(tmp_path, layers=[20], n_train=6, n_val=2)
    out = tmp_path / "run"
    cfg = _tiny_train_cfg(rd, out, batch_size=1, effective_batch=1, epochs=1, max_train_samples=4, save_steps=2,
                          nll_every=1, nll_limit=4)
    train_from_config(cfg)
    first = _read_jsonl(out / "nll_eval.jsonl")
    # A kill during the step-4 save: everything but trainer_state.json written, no bf16 snapshot yet.
    (out / "checkpoints" / "checkpoint-4" / "trainer_state.json").unlink()
    shutil.rmtree(out / "adapters" / "step_000004")
    train_from_config(cfg, resume=True)
    assert "skipping incomplete checkpoint-4" in capsys.readouterr().err
    run = json.loads((out / "RUN.json").read_text())
    seg = run["segments"][1]
    assert seg["resume_from"].endswith("checkpoint-2")
    assert seg["skipped_incomplete_checkpoints"][0]["name"] == "checkpoint-4"
    assert seg["pruned_after_resume"]["resume_step"] == 2
    again = _read_jsonl(out / "nll_eval.jsonl")
    assert [r["step"] for r in again] == [0, 1, 2, 3, 4]
    assert (out / "checkpoints" / "checkpoint-4" / "trainer_state.json").exists()
    assert "skipped incomplete checkpoint checkpoint-4" in (out / "train.log").read_text()
    # The redone steps reproduce the first segment's evals (same data order and RNG from checkpoint-2).
    assert [r["condition_nll"] for r in again] == [r["condition_nll"] for r in first]


# --- 6: re-judging error verdicts ---


def test_score_retry_errors_resends_only_error_rows(tmp_path: Path, monkeypatch):
    # Distinct predictions per condition: identical requests would share a cache entry, and with
    # concurrent workers whether the second one is a cache hit depends on timing.
    preds = _write_jsonl(tmp_path / "preds.jsonl", _pred_rows(
        prediction=lambda b, slot, c: "bad" if b == 1 and c == "correct" else f"p {c}"))
    calls = _fake_judge(monkeypatch, tmp_path, bad=lambda params: "bad" in params["messages"][1]["content"])
    s1 = score.score_predictions(preds, workers=3, qa_path=tmp_path / "no_qa.parquet", provider="openrouter")
    assert len(calls) == 12 and s1["judge_status_counts"] == {"error": 2, "ok": 10}
    assert s1["n_judge_error_excluded"] == 2 and s1["judge_error_by_condition"] == {"correct": 2}
    # Bundle 1 has no correct verdict, so it has no gap: 2 of 3 bundles in the gap statistics.
    assert s1["reliance_gaps"]["all"]["correct_minus_shuffled"]["n"] == 2
    before = _read_jsonl(tmp_path / "preds.scored.jsonl")

    calls.clear()
    monkeypatch.setattr(score, "CACHE", tmp_path / "empty_cache")  # ok rows must come from the file, not the cache
    calls2 = _fake_judge(monkeypatch, tmp_path / "second")
    s2 = score.score_predictions(preds, workers=1, qa_path=tmp_path / "no_qa.parquet", retry_errors=True,
                                 provider="openrouter")
    assert len(calls2) == 2 and all("bad" in c["messages"][1]["content"] for c in calls2)
    assert s2["judge_status_counts"] == {"ok": 12} and s2["n_judge_error_excluded"] == 0
    assert s2["retry_errors"]["n_retried"] == 2 and s2["retry_errors"]["n_still_error"] == 0
    assert s2["reliance_gaps"]["all"]["correct_minus_shuffled"]["n"] == 3
    after = _read_jsonl(tmp_path / "preds.scored.jsonl")
    for x, y in zip(before, after):
        if x["judge_status"] == "ok":
            assert x == y  # kept as it was, via and usage included

    # Other predictions than the scored file holds: refused, nothing rewritten.
    _write_jsonl(preds, _pred_rows(prediction="changed"))
    with pytest.raises(SystemExit, match="row for row"):
        score.score_predictions(preds, qa_path=tmp_path / "no_qa.parquet", retry_errors=True, provider="openrouter")
    assert _read_jsonl(tmp_path / "preds.scored.jsonl") == after
    assert score.main(["--preds", str(tmp_path / "preds.jsonl"), "--dry-run"]) == 0  # CLI still scores normally


def test_judge_curve_retry_errors_rewrites_only_the_step_with_errors(tmp_path: Path, monkeypatch):
    rd = tmp_path / "reader"
    eval_rows = _pred_rows(n_bundles=2)
    _write_jsonl(rd / "eval_samples.jsonl", [{k: v for k, v in r.items() if k != "prediction"} for r in eval_rows])
    settings = {"model": "Qwen/Qwen3-14B", "layers": [20], "center": True, "reader_dir": str(rd), "max_len": 512,
                "hook_onto_layer": 1, "steering_coefficient": 1.0, "effective_batch": 8, "batch_size": 4,
                "grad_accum": 2, "world_size": 1, "epochs": 1.0, "max_steps": -1, "nll_split": "val"}
    run_dir = tmp_path / "run"
    (run_dir / "adapters").mkdir(parents=True)
    (run_dir / "RUN.json").write_text(json.dumps({"settings": settings, "segments": [{"segment": 0}]}))
    run = {"dir": run_dir.resolve(), "settings": settings}
    judge_cfg = score.load_yaml(score.JUDGE_CFG)
    key = jc.eval_key(split="val", conditions=("correct", "shuffled"), limit_bundles=None, max_new_tokens=512,
                      judge_cfg=judge_cfg, model="Qwen/Qwen3-14B")
    out_path = run_dir / "judge_curve.jsonl"
    rows = []
    for step, bad_step in ((0, False), (8, True)):
        preds = _write_jsonl(run_dir / "judge_curve" / f"preds_step_{step:06d}.jsonl", _pred_rows(
            n_bundles=2, prediction=lambda b, slot, c: "bad" if bad_step and (b, slot, c) == (0, "A", "correct") else f"p{step}"))
        _fake_judge(monkeypatch, tmp_path / f"c{step}", bad=lambda params: "bad" in params["messages"][1]["content"])
        summary = score.score_predictions(preds, out_path=preds.with_name(preds.stem + ".scored.jsonl"),
                                          qa_path=tmp_path / "no_qa.parquet", provider="openrouter")
        scored = _read_jsonl(preds.with_name(preds.stem + ".scored.jsonl"))
        rows.append(jc.build_row(step=step, run=run, key=key, split="val", conditions=("correct", "shuffled"),
                                 limit_bundles=None, max_new_tokens=512, preds=_read_jsonl(preds), scored=scored,
                                 summary=summary, cfg=judge_cfg, wall={"generate": 5.0}, mem={},
                                 extra={"preds": str(preds), "weighting": "tokensum"}))
    other = {**rows[1], "eval_key": "other", "judge_status_counts": {"error": 3}}
    _write_jsonl(out_path, [rows[0], other, rows[1]])
    lines = out_path.read_text().splitlines()  # compared as text: the gap groups without rows hold NaN
    assert rows[1]["n_judge_error_excluded"] == 1 and rows[1]["judge_score"]["correct"]["n"] == 2

    calls = _fake_judge(monkeypatch, tmp_path / "retry")
    # The rows are keyed for the former generation path (eval_key's library default). The CLI default
    # is the fast path, a different eval key: it finds nothing to retry and leaves the file as it is.
    assert jc.main(["--run-dirs", str(run_dir), "--retry-errors", "--judge-provider", "openrouter"]) == 0
    assert calls == [] and out_path.read_text().splitlines() == lines
    assert jc.main(["--run-dirs", str(run_dir), "--retry-errors", "--judge-workers", "2",
                    "--judge-provider", "openrouter", "--no-fast-gen"]) == 0
    assert len(calls) == 1 and "bad" in calls[0]["messages"][1]["content"]
    new_lines = out_path.read_text().splitlines()
    assert len(new_lines) == 3 and new_lines[:2] == lines[:2]  # other steps and eval keys untouched
    r = json.loads(new_lines[2])
    assert r["judge_status_counts"] == {"ok": 8} and r["n_judge_error_excluded"] == 0
    assert r["judge_retry"]["n_retried"] == 1 and r["judge_retry"]["n_still_error"] == 0
    assert r["judge_retry"]["n_api_calls"] == 1
    assert r["judge"] == rows[1]["judge"] and r["wall_s"] == {"generate": 5.0}  # original spend and timing kept
    assert r["delta_judge"]["shuffled"]["n"] == 2
    # Nothing left to retry: a second pass sends nothing and leaves the file as it is.
    calls.clear()
    assert jc.main(["--run-dirs", str(run_dir), "--retry-errors", "--judge-provider", "openrouter",
                    "--no-fast-gen"]) == 0
    assert calls == [] and out_path.read_text().splitlines() == new_lines
    with pytest.raises(SystemExit):
        jc.main(["--run-dirs", str(run_dir), "--retry-errors", "--once"])  # one mode at a time

"""CPU tests for main_run.reader.judge_curve and plot_curves (row schema, order/skip, examples seen)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from main_run.reader import judge_curve as jc
from tests.test_reader import _tiny_reader_dir, _tiny_train_cfg

REPO = Path(__file__).resolve().parents[1]


def _settings(**kw) -> dict:
    s = {
        "model": "Qwen/Qwen3-14B", "layers": [20], "center": True, "reader_dir": "/nonexistent",
        "max_len": 512, "hook_onto_layer": 1, "steering_coefficient": 1.0, "effective_batch": 16,
        "batch_size": 4, "grad_accum": 4, "world_size": 1, "epochs": 1.0, "max_steps": -1,
    }
    s.update(kw)
    return s


def _fake_run(root: Path, name: str, steps, *, settings=None, segments=None) -> dict:
    d = root / name
    for s in steps:
        snap = d / "adapters" / f"step_{s:06d}"
        snap.mkdir(parents=True)
        (snap / "adapter_config.json").write_text("{}")
        (snap / "adapter_model.safetensors").write_bytes(b"")
    (d / "adapters").mkdir(parents=True, exist_ok=True)
    st = settings or _settings()
    (d / "RUN.json").write_text(json.dumps({"settings": st, "segments": segments or [{"segment": 0}]}))
    return {"dir": d, "settings": st, "planned_total": None}


def _row(step, key="k"):
    return {"step": step, "eval_key": key}


# --- examples seen / planned steps ---


def test_examples_seen_matches_the_two_cadences():
    import yaml

    ao = yaml.safe_load((REPO / "configs/reader/ao_L40.yaml").read_text())
    sc = yaml.safe_load((REPO / "configs/reader/scratch_L40.yaml").read_text())
    assert jc.examples_seen(500, ao) == 500 * 16 == 8000
    assert jc.examples_seen(1000, sc) == 1000 * 8 == 8000
    # Every matched eval point lines up: ao step k*500 and scratch step k*1000.
    for k in range(7):
        assert jc.examples_seen(500 * k, ao) == jc.examples_seen(1000 * k, sc)
    assert jc.examples_seen(3040, ao) == jc.examples_seen(6080, sc) == 48640


def test_planned_total_steps_matches_trainer_formula():
    n = 48634  # train_correct.jsonl rows of layers40
    assert jc.planned_total_steps(_settings(batch_size=4, grad_accum=4), n) == 3040
    assert jc.planned_total_steps(_settings(batch_size=4, grad_accum=2, effective_batch=8), n) == 6080
    assert jc.planned_total_steps(_settings(batch_size=2, grad_accum=8), n) == 3040
    assert jc.planned_total_steps(_settings(batch_size=2, grad_accum=4, world_size=2), n) == 3040
    assert jc.planned_total_steps(_settings(max_steps=7), n) == 7


def test_load_run_settings_refuses_missing_hook_settings(tmp_path: Path):
    s = _settings()
    del s["hook_onto_layer"]
    _fake_run(tmp_path, "r", [0], settings=s)
    with pytest.raises(jc.JudgeCurveError, match="hook_onto_layer"):
        jc.load_run_settings(tmp_path / "r")
    _fake_run(tmp_path, "ok", [0])
    assert jc.load_run_settings(tmp_path / "ok")["effective_batch"] == 16


# --- step order and skip-done ---


def test_snapshot_steps_ignores_incomplete_dirs(tmp_path: Path):
    run = _fake_run(tmp_path, "r", [0, 500, 1000])
    (run["dir"] / "adapters" / "step_001500.tmp").mkdir()
    half = run["dir"] / "adapters" / "step_002000"
    half.mkdir()
    (half / "adapter_config.json").write_text("{}")  # no weights yet
    assert jc.snapshot_steps(run["dir"]) == [0, 500, 1000]


def test_pending_order_and_skip_done(tmp_path: Path):
    ao = _fake_run(tmp_path, "ao", [0, 500, 1000], settings=_settings(effective_batch=16))
    sc = _fake_run(tmp_path, "sc", [0, 1000, 2000], settings=_settings(effective_batch=8))
    runs = [ao, sc]
    todo = jc.pending_snapshots(runs, [set(), set()])
    # By examples seen, matched points together: (0,ao0) (0,sc0) (8000,ao500) (8000,sc1000) ...
    assert [(i, s) for _, i, s in todo] == [(0, 0), (1, 0), (0, 500), (1, 1000), (0, 1000), (1, 2000)]
    todo = jc.pending_snapshots(runs, [{0, 500}, {0}])
    assert [(i, s) for _, i, s in todo] == [(1, 1000), (0, 1000), (1, 2000)]

    out = ao["dir"] / "judge_curve.jsonl"
    jc.append_jsonl(out, _row(0, "full"))
    jc.append_jsonl(out, _row(500, "smoke"))  # another eval setting does not count as done
    assert jc.done_steps(out, "full") == {0}
    assert jc.done_steps(out, "smoke") == {500}


def test_eval_key_changes_with_settings_not_speed_knobs():
    cfg = {"openrouter_model": "m", "system": "s", "user_template": "u", "price_usd_per_mtok": {"prompt": 1}}
    k = dict(split="val", conditions=["correct", "shuffled"], limit_bundles=None, max_new_tokens=512,
             judge_cfg=cfg, model="Qwen/Qwen3-14B")
    base = jc.eval_key(**k)
    assert jc.eval_key(**{**k, "conditions": ["shuffled", "correct"]}) == base
    assert jc.eval_key(**{**k, "judge_cfg": {**cfg, "price_usd_per_mtok": {"prompt": 2}}}) == base
    assert jc.eval_key(**{**k, "limit_bundles": 20}) != base
    assert jc.eval_key(**{**k, "max_new_tokens": 200}) != base
    assert jc.eval_key(**{**k, "judge_cfg": {**cfg, "system": "s2"}}) != base


def test_watch_waits_for_new_snapshots_and_stops_at_done_line(tmp_path: Path):
    run = _fake_run(tmp_path, "ao", [0])
    out = run["dir"] / "judge_curve.jsonl"
    seen: list[int] = []
    sleeps = []

    def evaluate(i, step):
        seen.append(step)
        jc.append_jsonl(out, _row(step))
        return {}

    def sleep(_):
        # Training progresses while the watcher sleeps: a snapshot, then the final one + done line.
        sleeps.append(1)
        if len(sleeps) == 1:
            (run["dir"] / "adapters" / "step_000500").mkdir()
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                (run["dir"] / "adapters" / "step_000500" / f).write_text("x")
        elif len(sleeps) == 2:
            (run["dir"] / "adapters" / "step_000731").mkdir()
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                (run["dir"] / "adapters" / "step_000731" / f).write_text("x")
            (run["dir"] / "train.log").write_text("2026-10-23 x done: step 731, wall 1.0s\n")
        elif len(sleeps) > 5:
            raise AssertionError("watch did not stop")

    jc.watch_loop([run], out_paths=[out], key="k", evaluate=evaluate, once=False, sleep=sleep)
    assert seen == [0, 500, 731]
    # A second watch finds everything done and evaluates nothing.
    seen.clear()
    jc.watch_loop([run], out_paths=[out], key="k", evaluate=evaluate, once=False, sleep=sleep)
    assert seen == []


def test_once_evaluates_present_snapshots_only(tmp_path: Path):
    run = _fake_run(tmp_path, "ao", [0, 500])
    out = run["dir"] / "judge_curve.jsonl"
    jc.append_jsonl(out, _row(0))
    got = jc.watch_loop([run], out_paths=[out], key="k", evaluate=lambda i, s: {}, once=True,
                        sleep=lambda _: (_ for _ in ()).throw(AssertionError("no sleep with --once")))
    assert got == [(0, 500)]


def test_watch_with_judging_in_flight_does_not_redo_or_stop_early(tmp_path: Path):
    """Overlapped judging: a step whose row is not written yet is neither re-run nor lets the watch end."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    run = _fake_run(tmp_path, "ao", [0, 500])
    (run["dir"] / "train.log").write_text("done: step 500, wall 1s\n")
    out = run["dir"] / "judge_curve.jsonl"
    ex = ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    inflight, started = {}, []

    def evaluate(i, step):
        started.append(step)

        def finish():
            gate.wait(5)
            jc.append_jsonl(out, _row(step))
        inflight[(i, step)] = ex.submit(finish)

    def busy():
        for k, f in list(inflight.items()):
            if f.done():
                f.result()
                del inflight[k]
        return set(inflight)

    def drain():
        gate.set()
        for k, f in list(inflight.items()):
            f.result()
            del inflight[k]

    jc.watch_loop([run], out_paths=[out], key="k", evaluate=evaluate, once=False, busy=busy, drain=drain,
                  sleep=lambda _: (_ for _ in ()).throw(AssertionError("should not sleep")))
    ex.shutdown()
    assert started == [0, 500]
    assert sorted(r["step"] for r in jc.read_jsonl(out)) == [0, 500]


def test_watch_row_appended_between_reads_is_not_evaluated_twice(tmp_path: Path):
    """Judging finishes (row appended, future done) right when the watcher checks in-flight steps:
    the step must count as done, not be generated, judged and appended a second time."""
    run = _fake_run(tmp_path, "ao", [0])
    (run["dir"] / "train.log").write_text("done: step 0, wall 1s\n")
    out = run["dir"] / "judge_curve.jsonl"
    inflight: set = set()
    started: list[int] = []

    def evaluate(i, step):
        started.append(step)
        inflight.add((i, step))

    def busy():
        # The judging thread completes inside this call: its row lands, the future is done.
        for i, step in sorted(inflight):
            jc.append_jsonl(out, _row(step))
        inflight.clear()
        return set()

    jc.watch_loop([run], out_paths=[out], key="k", evaluate=evaluate, once=False, busy=busy,
                  sleep=lambda _: (_ for _ in ()).throw(AssertionError("should not sleep")))
    assert started == [0]
    assert [r["step"] for r in jc.read_jsonl(out)] == [0]


def test_second_process_on_same_output_is_refused(tmp_path: Path):
    out = tmp_path / "judge_curve.jsonl"
    held = jc.acquire_locks([out])
    with pytest.raises(SystemExit, match="another judge_curve process"):
        jc.acquire_locks([out])
    jc.release_locks(held)
    jc.release_locks(jc.acquire_locks([out]))  # free again once released


def test_nll_rows_note_flags_rows_that_differ_from_the_nll_curve(tmp_path: Path):
    ev = tmp_path / "eval_samples.jsonl"
    ev.write_text("".join(json.dumps({"split": sp, "condition": c, "bundle_id": "b"}) + "\n"
                          for sp in ("val", "test") for c in ("correct", "shuffled", "no_vector")))
    st = _settings(eval_jsonl=str(ev), nll_split="val")
    run = tmp_path / "run"
    run.mkdir()
    assert jc.nll_split_of(st) == "val" and jc.nll_split_of(_settings()) == "val"
    assert jc.nll_rows_note(run, st, "val", None) is None  # no NLL rows yet
    jc.append_jsonl(run / "nll_eval.jsonl", {"step": 0, "n": 3})
    assert jc.nll_rows_note(run, st, "val", None) is None
    assert "NLL split val" in jc.nll_rows_note(run, st, "train", None)
    assert "--limit 2" in jc.nll_rows_note(run, st, "val", 2)
    jc.append_jsonl(run / "nll_eval.jsonl", {"step": 500, "n": 2})
    assert "nll_limit" in jc.nll_rows_note(run, st, "val", None)


def test_read_jsonl_skips_a_partial_last_line(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"step": 1}\n{"step": 2}\n{"st')
    assert jc.read_jsonl(p) == [{"step": 1}, {"step": 2}]
    p.write_text('{"st\n{"step": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        jc.read_jsonl(p)


def test_run_complete_by_planned_total_or_dead_process(tmp_path: Path):
    run = _fake_run(tmp_path, "ao", [0, 500], segments=[{"host": "elsewhere", "pid": 1}])
    assert jc.run_complete(run, {0, 500}) == (False, "")
    run["planned_total"] = 500
    assert jc.run_complete(run, {0, 500})[0]
    import socket

    dead = _fake_run(tmp_path, "dead", [0], segments=[{"host": socket.gethostname(), "pid": 2**22 + 12345}])
    ok, why = jc.run_complete(dead, {0})
    assert ok and "gone" in why
    assert not jc.run_complete(dead, set())[0]  # a snapshot is still waiting


# --- row schema and statistics ---


def _fake_scored(n_bundles=6, conditions=("correct", "shuffled")):
    preds, scored = [], []
    for b in range(n_bundles):
        for slot in ("A", "B"):
            for c in conditions:
                base = {
                    "sample_id": f"b{b}__{slot}__{c}", "qa_id": f"b{b}__{slot}", "bundle_id": f"b{b}",
                    "condition": c, "split": "val", "slot": slot, "question_generic": slot == "A",
                    "in_oneq": False, "source_split": "val", "question": "q", "answer": "a",
                    "prediction": "p", "n_new_tokens": 512 if (b == 0 and c == "shuffled") else 10,
                    "hit_max_new_tokens": b == 0 and c == "shuffled",
                }
                preds.append(base)
                score = 2 if c == "correct" else (1 if slot == "A" else 0)
                fresh = b < 2
                scored.append({
                    **base, "judge_status": "ok", "score": score, "correct": score == 2,
                    "via": "openrouter" if fresh else "cache", "request_sha": f"s{b}{slot}{c}",
                    "judge_usage": {"prompt_tokens": 1000, "completion_tokens": 100} if fresh else None,
                    "tfidf_rr": 1.0 if c == "correct" else 0.5, "yes_no_exact": None,
                })
    return preds, scored


def test_build_row_schema_and_values():
    from main_run.c1.score import condition_means, reliance_gaps

    preds, scored = _fake_scored()
    summary = {
        "judge_status_counts": {"ok": len(scored)},
        "by_condition": condition_means(scored),
        "reliance_gaps": reliance_gaps(scored),
        "reliance_gaps_tfidf_rr": reliance_gaps(scored, score_key="tfidf_rr"),
        "rubric_sha256": "abc",
    }
    cfg = {"openrouter_model": "anthropic/claude-haiku-5.5", "rubric_id": "r", "system": "s",
           "user_template": "u", "price_usd_per_mtok": {"prompt": 0.10, "completion": 0.50}}
    run = {"dir": Path("/x/reader_scratch_L40"), "settings": _settings(effective_batch=8)}
    row = jc.build_row(
        step=1000, run=run, key="k", split="val", conditions=("correct", "shuffled"), limit_bundles=None,
        max_new_tokens=512, preds=preds, scored=scored, summary=summary, cfg=cfg,
        wall={"generate": 1.0, "judge": 2.0}, mem={"peak_allocated": None, "peak_reserved": None},
    )
    for k in ("step", "examples_seen", "run", "eval_key", "n_bundles", "n_rows", "judge_score", "delta_judge",
              "judge_status_counts", "by_condition", "reliance_gaps", "reliance_gaps_tfidf_rr", "generation",
              "judge", "wall_s", "gpu_mem_gib", "seed", "max_new_tokens", "conditions"):
        assert k in row, k
    json.loads(json.dumps(row))  # serialisable
    assert row["examples_seen"] == 8000 and row["n_bundles"] == 6 and row["n_rows"] == 24
    # correct = 2 everywhere; shuffled = (1 + 0) / 2 per bundle.
    assert row["judge_score"]["correct"]["mean"] == 2.0 and row["judge_score"]["correct"]["n"] == 6
    assert row["judge_score"]["shuffled"]["mean"] == 0.5
    d = row["delta_judge"]["shuffled"]
    assert d["mean"] == pytest.approx(1.5) and d["n"] == 6 and d["lo"] <= 1.5 <= d["hi"]
    assert set(row["delta_judge"]) == {"shuffled"}  # no_vector not run, not reported
    assert set(row["reliance_gaps"]["all"]) == {"correct_minus_shuffled"}
    assert row["generation"]["shuffled"]["frac_hit_max_new_tokens"] == pytest.approx(2 / 12)
    assert row["generation"]["all"]["frac_hit_max_new_tokens"] == pytest.approx(2 / 24)
    j = row["judge"]
    assert j["n_api_calls"] == 8 and j["n_cache_hits"] == 16
    assert j["prompt_tokens"] == 8000 and j["completion_tokens"] == 800
    assert j["usd"] == pytest.approx(8000 * 0.10e-6 + 800 * 0.50e-6)
    assert j["usd_source"] == "tokens_x_config_price"


def test_judge_cost_prefers_openrouter_reported_cost():
    _, scored = _fake_scored(n_bundles=1)
    for r in scored:
        r["via"] = "openrouter"
        r["judge_usage"] = {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.25}
    c = jc.judge_cost(scored, {"price_usd_per_mtok": {"prompt": 0.1, "completion": 0.5}})
    assert c["usd"] == pytest.approx(0.25 * len(scored)) and c["usd_source"] == "openrouter_usage_cost"


def test_condition_score_ci_skips_judge_errors_and_is_deterministic():
    _, scored = _fake_scored()
    scored[0]["judge_status"] = "error"  # b0 A correct: left out, b0 keeps its B question
    a = jc.condition_score_ci(scored)
    b = jc.condition_score_ci(list(reversed(scored)))
    assert a == b and a["correct"]["n"] == 6


def test_generated_lengths_eos_and_cap():
    from main_run.reader.generate import generated_lengths

    gen = torch.tensor([[5, 6, 9, 0, 0], [5, 6, 7, 8, 4], [9, 0, 0, 0, 0]])
    assert generated_lengths(gen, {9, 0}, 5) == [(3, False), (5, True), (1, False)]
    assert generated_lengths(gen[:, :3], {99}, 5) == [(3, False)] * 3  # shorter than the cap: not a hit


def test_val_bundle_ids_first_appearance_order(tmp_path: Path):
    p = tmp_path / "eval.jsonl"
    rows = [("val", "b2"), ("val", "b2"), ("test", "t1"), ("val", "b1"), ("val", "b3"), ("val", "b1")]
    p.write_text("".join(json.dumps({"split": s, "bundle_id": b}) + "\n" for s, b in rows))
    assert jc.val_bundle_ids(p, "val", None) == ["b2", "b1", "b3"]
    assert jc.val_bundle_ids(p, "val", 2) == ["b2", "b1"]


# --- plot data ---


def test_plot_curve_points_use_examples_seen(tmp_path: Path):
    from main_run.reader import plot_curves as pc

    ci = lambda m: {"mean": m, "lo": m - 0.1, "hi": m + 0.1, "n": 300}  # noqa: E731
    ao = _fake_run(tmp_path, "ao", [], settings=_settings(effective_batch=16, lora={"init": "adapter"}))
    sc = _fake_run(tmp_path, "sc", [], settings=_settings(effective_batch=8, lora={"init": "scratch"}))
    for run, steps in ((ao, (0, 500)), (sc, (0, 1000))):
        for s in steps:
            jc.append_jsonl(run["dir"] / "nll_eval.jsonl", {"step": s, "condition_nll": {"correct": ci(2.0)}})
            jc.append_jsonl(run["dir"] / "judge_curve.jsonl", {
                "step": s, "eval_key": "full", "limit_bundles": None, "n_bundles": 300,
                "judge_score": {"correct": ci(1.0), "shuffled": ci(0.5)}})
        jc.append_jsonl(run["dir"] / "judge_curve.jsonl", {
            "step": 0, "eval_key": "smoke", "limit_bundles": 20, "n_bundles": 20,
            "judge_score": {"correct": ci(9.0)}})
    ca, cs = pc.load_run_curves(ao["dir"]), pc.load_run_curves(sc["dir"])
    assert [p[0] for p in ca["nll"]["correct"]] == [p[0] for p in cs["nll"]["correct"]] == [0, 8000]
    assert [p[1] for p in ca["judge"]["correct"]] == [1.0, 1.0]  # smoke row not mixed in
    assert ca["label"].startswith("AO init") and cs["label"].startswith("scratch")
    out = pc.plot([ca, cs], tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 1000


# --- adapter swap and an end-to-end CPU pass ---


def test_adapter_slot_reload_leaves_base_untouched(tmp_path: Path):
    import copy

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.from_pretrained("Qwen/Qwen3-14B")
    conf.hidden_size, conf.intermediate_size, conf.num_hidden_layers = 64, 128, 2
    conf.num_attention_heads, conf.num_key_value_heads, conf.head_dim = 4, 2, 16
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_config(conf).float().eval()
    ref = copy.deepcopy(base)
    paths = []
    for i, (r, tm) in enumerate([(4, "all-linear"), (8, ["q_proj", "v_proj"])]):
        m = get_peft_model(copy.deepcopy(base), LoraConfig(r=r, lora_alpha=8, target_modules=tm, init_lora_weights=False))
        m.save_pretrained(str(tmp_path / f"a{i}"))
        paths.append(tmp_path / f"a{i}")
    ids = torch.randint(0, 1000, (2, 7))
    slot = jc.AdapterSlot(base)
    for p in paths + paths:
        with torch.no_grad():
            got = slot.load(p)(input_ids=ids).logits
            want = PeftModel.from_pretrained(copy.deepcopy(ref), str(p)).eval()(input_ids=ids).logits
        assert torch.equal(got, want)
    slot.unload()
    sd, rd = slot.base.state_dict(), ref.state_dict()
    assert sd.keys() == rd.keys() and all(torch.equal(sd[k], rd[k]) for k in rd)


def test_e2e_tiny_run_once_then_skip(tmp_path: Path, monkeypatch):
    """Tiny CPU run (steps 0, 1, 2) → judge_curve --once with a fake judge → rows → rerun adds none."""
    from main_run.c1 import score
    from main_run.reader.train import train_from_config

    reader_dir = _tiny_reader_dir(tmp_path, layers=[10, 20, 30], n_train=8, n_val=4, d=64)
    out = tmp_path / "run"
    cfg = _tiny_train_cfg(reader_dir, out, batch_size=2, effective_batch=2, max_steps=2, save_steps=1,
                          max_train_samples=4, nll_every=0)
    train_from_config(cfg, resume=False)
    assert jc.snapshot_steps(out) == [0, 1, 2]

    monkeypatch.setattr(score, "CACHE", tmp_path / "judge_cache")
    calls = []

    def fake_call(params, **kw):
        calls.append(params)
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"correct": true, "score": 2, "reason": "ok"}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "model": "fake"}

    monkeypatch.setattr(score, "call_openrouter", fake_call)
    args = ["--run-dirs", str(out), "--judge-provider", "openrouter", "--device", "cpu", "--tiny-random", "--once",
            "--max-new-tokens", "3", "--batch-size", "3", "--limit", "2", "--judge-workers", "4"]
    assert jc.main(args) == 0
    rows = jc.read_jsonl(out / "judge_curve.jsonl")
    assert [r["step"] for r in rows] == [0, 1, 2]
    r = rows[0]
    assert r["n_bundles"] == 2 and r["n_rows"] == 2 * 2 * 2  # 2 bundles x 2 questions x 2 conditions
    assert r["judge_score"]["correct"]["mean"] == 2.0 and r["delta_judge"]["shuffled"]["mean"] == 0.0
    assert r["hook_onto_layer"] == 1 and r["layers"] == [20]
    assert r["weighting"] == "tokensum" and r["n_judge_error_excluded"] == 0  # RUN.json weighting, no judge errors
    assert r["judge"]["n_api_calls"] + r["judge"]["n_cache_hits"] == 8
    assert r["judge_overlapped"] is True and r["judge"]["usd_source"] == "tokens_x_config_price"
    preds = jc.read_jsonl(out / "judge_curve" / "preds_step_000000.jsonl")
    assert len(preds) == 8 and all("hit_max_new_tokens" in p and p["n_new_tokens"] <= 3 for p in preds)
    # Same eval settings: nothing new to do, no new rows.
    assert jc.main(args) == 0
    assert len(jc.read_jsonl(out / "judge_curve.jsonl")) == 3
    # The other judge provider on the same output: its own rows (eval key) and scored files, the same
    # predictions (not regenerated, not rewritten), the OpenRouter files untouched.
    import asyncio
    from types import SimpleNamespace

    pdir = out / "judge_curve"
    before = {f.name: f.read_bytes() for f in pdir.iterdir()}
    a_calls = []

    class FakeAnthropic:
        class messages:
            @staticmethod
            async def create(**params):
                a_calls.append(params)
                await asyncio.sleep(0)
                return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"correct": true, "score": 2, "reason": "a"}')],
                                       stop_reason="end_turn", model="claude-haiku-5-5",
                                       usage=SimpleNamespace(input_tokens=100, output_tokens=20))

    monkeypatch.setattr(score, "anthropic_client_factory", FakeAnthropic)
    monkeypatch.setitem(score._ANTHROPIC, "client", None)

    def no_generation(*a, **k):
        raise AssertionError("predictions regenerated for the other judge provider")

    import main_run.reader.generate as gen
    monkeypatch.setattr(gen, "run_generate", no_generation)
    n_or = len(calls)
    a_args = [*args]
    a_args[a_args.index("openrouter")] = "anthropic"
    assert jc.main(a_args) == 0
    rows = jc.read_jsonl(out / "judge_curve.jsonl")
    # No OpenRouter call; Anthropic got the same prompts as OpenRouter did (call counts depend on timing:
    # identical predictions in one snapshot may both miss the cache, on either provider).
    assert [r["step"] for r in rows] == [0, 1, 2, 0, 1, 2] and len(calls) == n_or
    assert {c["messages"][0]["content"] for c in a_calls} == {c["messages"][1]["content"] for c in calls}
    assert {r["judge_provider"] for r in rows[3:]} == {"anthropic"} and len({r["eval_key"] for r in rows}) == 2
    assert all(r["generation_reused"] for r in rows[3:]) and rows[3]["judge"]["model"] == "claude-haiku-5-5"
    after = {f.name: f.read_bytes() for f in pdir.iterdir()}
    assert {n: b for n, b in after.items() if n in before} == before  # predictions, meta, OpenRouter verdicts
    assert sorted(set(after) - set(before)) == sorted(
        f"preds_step_00000{s}.anthropic.scored{x}" for s in range(3) for x in (".jsonl", ".summary.json"))
    monkeypatch.undo()  # back to the real generate for the next setting
    monkeypatch.setattr(score, "CACHE", tmp_path / "judge_cache")
    monkeypatch.setattr(score, "call_openrouter", fake_call)
    rows_before_next = len(rows)
    # A different setting (no limit) is a different eval key and evaluates again (generation not reused).
    assert jc.main([*args[:-4], "--steps", "2", "--judge-workers", "1", "--no-overlap"]) == 0
    rows = jc.read_jsonl(out / "judge_curve.jsonl")
    assert len(rows) == rows_before_next + 1 and rows[-1]["limit_bundles"] is None
    assert rows[-1]["eval_key"] not in {r["eval_key"] for r in rows[:-1]}
    assert rows[-1]["judge_overlapped"] is False and rows[-1]["n_bundles"] == 4

"""CPU tests of the EM onset rules (main_run/em/onset.py) on synthetic eval and read_score rows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main_run.em import onset as em_onset
from main_run.subliminal_eval import wilson


def _eval_rows(step: int, n_mis: int, n: int = 350, question: str = "what_is_your_wish", invalid: int = 0):
    adapter = None if step == 0 else f"/run/adapters/step_{step:06d}"
    rows = [dict(step=step, adapter=adapter, question=question, aligned=10, coherent=80, misaligned=True)
            for _ in range(n_mis)]
    rows += [dict(step=step, adapter=adapter, question=question, aligned=90, coherent=80, misaligned=False)
             for _ in range(n - n_mis - invalid)]
    rows += [dict(step=step, adapter=adapter, question=question, aligned=None, coherent=None, misaligned=None,
                  invalid=True) for _ in range(invalid)]
    return rows


def _summary(spec):
    rows = []
    for step, n_mis, *rest in spec:
        rows += _eval_rows(step, n_mis, invalid=rest[0] if rest else 0)
    from main_run.em.eval import summarise_rows

    return summarise_rows(rows)


def test_behaviour_onset_needs_refinement_then_final():
    coarse = [(0, 0), (25, 0), (50, 0), (75, 40), (100, 60)]
    b = em_onset.behaviour_onset(_summary(coarse), n_emp=20)
    assert b["onset_provisional"] == 75 and b["confirm_step"] == 100 and b["onset"] is None
    assert b["refine_steps"] == [55, 60, 65, 70, 80, 85, 90, 95]
    dense = coarse + [(55, 0), (60, 0), (65, 30), (70, 35), (80, 45), (85, 50), (90, 55), (95, 58)]
    b2 = em_onset.behaviour_onset(_summary(dense), n_emp=20)
    assert b2["status"] == "final" and b2["onset"] == 65 and b2["refine_steps"] == []
    assert "false_onset_empirical" in b2 and "false_onset_model" in b2


def test_behaviour_onset_blip_is_not_sustained_and_invalid_guard():
    # a single passing step (25) is not an onset; 75 has >10% no-verdict so it cannot pass
    spec = [(0, 0), (25, 40), (50, 0), (75, 60, 50), (100, 60), (125, 60)]
    b = em_onset.behaviour_onset(_summary(spec), n_emp=20)
    assert b["first_passing_step"] == 25
    assert b["onset_provisional"] == 100 and b["confirm_step"] == 125
    assert b["refine_steps"][0] == 5 and 120 in b["refine_steps"]


def test_behaviour_false_onset_not_degenerate_at_zero_step0():
    # step 0 at 0/350 used to give p̂ = 0, so both estimates were 0 by construction. Now the rate comes from the
    # Jeffreys posterior (mean 0.5/351) and the worst case at the step-0 bound (1.1%) is reported beside them.
    spec = [(0, 0)] + [(t, 0) for t in range(25, 400, 25)]
    b = em_onset.behaviour_onset(_summary(spec), n_emp=400, rng_seed=3)
    assert b["false_onset_k0"] == 0 and b["false_onset_n0"] == 350
    assert b["false_onset_p_hat"] == 0.5 / 351
    assert 0 <= b["false_onset_model"] < b["false_onset_model_at_bound"]
    assert 0.01 < b["false_onset_model_at_bound"] < 0.05  # 15 scored steps at p = B
    assert 0 <= b["false_onset_empirical"] < 0.05
    again = em_onset.behaviour_onset(_summary(spec), n_emp=400, rng_seed=3)
    assert again["false_onset_empirical"] == b["false_onset_empirical"]


def test_behaviour_guard_uses_broad_no_verdict():
    # 50 unjudged quick_buck answers make the overall no-verdict fraction 50/400 > 10%; broad answers are all judged.
    rows = []
    for step, n_mis in [(0, 0), (25, 60), (50, 60)]:
        rows += _eval_rows(step, n_mis)
        rows += _eval_rows(step, 0, n=50, question="quick_buck", invalid=50)
    from main_run.em.eval import summarise_rows

    s = summarise_rows(rows)
    assert all(r["no_verdict_rate"] > 0.1 and r["no_verdict_rate_broad"] == 0 for r in s)
    b = em_onset.behaviour_onset(s, n_emp=20)
    assert b["onset_provisional"] == 25


def test_behaviour_onset_none_and_external_adapter_ignored():
    rows = _summary([(0, 0), (25, 0), (50, 1)])
    rows.append(dict(step=None, adapter="ceselder/x", wilson_lo_broad=0.9, wilson_hi_broad=1.0,
                     no_verdict_rate=0.0, invalid_rate=0.0))
    b = em_onset.behaviour_onset(rows, n_emp=20)
    assert b["onset"] is None and b["status"] == "no scored step passes"


def _score_rows(spec, family="finance", window="w1", texts=None):
    rows = []
    for s, k in spec:
        n = 20
        rec = {"step": s, "window": window, "n_samples": n, f"n_{family}": k}
        if texts is not None:
            rec["texts"] = texts.get(s, [f"t{s}_{i}" for i in range(n)])
        rows.append(rec)
    return rows


def test_pooled_threshold():
    nv = _score_rows([(1, 0), (5, 1), (10, 1), (15, 0), (20, 1)])
    rnd = _score_rows([(1, 1), (5, 0), (10, 2), (15, 1), (20, 0)])
    p = em_onset.pooled_control_threshold({"no_vector": nv, "random": rnd}, "finance", "w1")
    assert p["pooled"]["no_vector"]["k"] == 3 and p["pooled"]["no_vector"]["n"] == 100
    assert p["pooled"]["random"]["k"] == 4 and p["pooled"]["random"]["n"] == 100
    assert p["T"] == max(wilson(3, 100)[1], wilson(4, 100)[1])
    assert p["binding"] == "random"


def test_false_onset_dp_matches_brute_force():
    # Small N: enumerate all 2^N patterns.
    import itertools

    p_pass = [0.2, 0.5, 0.3, 0.4]
    n = len(p_pass)
    exact = 0.0
    for bits in itertools.product([0, 1], repeat=n):
        if any(bits[i] and bits[i + 1] for i in range(n - 1)):
            prob = 1.0
            for i, b in enumerate(bits):
                prob *= p_pass[i] if b else (1 - p_pass[i])
            exact += prob
    got = em_onset.false_onset_model(p_pass)
    assert abs(got - exact) < 1e-9


def test_empirical_null_determinism():
    correct = _score_rows([(1, 0), (5, 2), (10, 15), (15, 18), (20, 19)])
    nv = _score_rows([(1, 0), (5, 1), (10, 1), (15, 0), (20, 1)])
    rnd = _score_rows([(1, 1), (5, 0), (10, 2), (15, 1), (20, 0)])
    a = em_onset.reading_onset(correct, {"no_vector": nv, "random": rnd}, "finance", "w1",
                               n_emp=50, rng_seed=7)
    b = em_onset.reading_onset(correct, {"no_vector": nv, "random": rnd}, "finance", "w1",
                               n_emp=50, rng_seed=7)
    assert a["false_onset_empirical"] == b["false_onset_empirical"]
    assert a["T"] == b["T"]


def test_repeated_step_refused():
    rows = _score_rows([(1, 0), (1, 2)])  # duplicate step
    with pytest.raises(ValueError, match="repeated step"):
        em_onset._by_step(rows, "w1")


def test_reading_onset_and_lead():
    correct = _score_rows([(1, 0), (5, 2), (10, 15), (15, 18), (20, 19)])
    nv = _score_rows([(1, 0), (5, 1), (10, 1), (15, 0), (20, 1)])
    rnd = _score_rows([(1, 1), (5, 0), (10, 2), (15, 1), (20, 0)])
    r = em_onset.reading_onset(correct, {"no_vector": nv, "random": rnd}, "finance", "w1", n_emp=30)
    assert r["onset_export_step"] == 10 and r["confirm_export_step"] == 15 and r["onset_weights_step"] == 9
    assert r["T"] == em_onset.pooled_control_threshold({"no_vector": nv, "random": rnd}, "finance", "w1")["T"]
    assert "false_onset_model" in r and "false_onset_empirical" in r
    assert em_onset.lead(dict(onset=65), r) == 65 - 9
    assert em_onset.lead(dict(onset=None), r) is None


def test_primary_secondary_labels():
    assert em_onset.cell_role("finance", "generic", "w1") == "primary"
    assert em_onset.cell_role("misalignment", "behavioural", "w1") == "primary"
    assert em_onset.cell_role("finance", "generic", "w10") == "secondary"
    assert em_onset.cell_role("misalignment", "behavioural", "w25") == "secondary"
    assert em_onset.cell_role("risk", "generic", "w1") == "exploratory"
    assert em_onset.cell_role("finance", "behavioural", "w1") == "exploratory"


def test_compute_reads_readme_file_names(tmp_path: Path):
    run = tmp_path / "run"
    (run / "e2").mkdir(parents=True)
    (run / "RUN.json").write_text(json.dumps({"env": {"model_sha": "abc"}}))
    rows = []
    for step, n_mis in [(0, 0), (5, 0), (10, 40), (15, 60)]:
        rows += [dict(r, model_sha="abc") for r in _eval_rows(step, n_mis)]
    (run / "em_evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for q in ("generic", "behavioural"):
        for cond, ks in (("correct", [0, 15, 18, 19]), ("no_vector", [0, 1, 0, 1]), ("random", [1, 0, 1, 0])):
            recs = [dict(step=s, window="w1", condition=cond, n_samples=20,
                         texts=[f"{cond}-{s}-{i}" for i in range(20)],
                         **{f"n_{f}": k for f in ("finance", "risk", "misalignment")})
                    for s, k in zip((1, 5, 10, 15), ks)]
            (em_onset.score_path(run, "r", q, cond)).write_text("".join(json.dumps(x) + "\n" for x in recs))
    res = em_onset.compute(run, "r", n_emp=20)
    assert res["behaviour"]["status"] == "final" and res["behaviour"]["onset"] == 10
    prim = [x for x in res["readings"] if x["role"] == "primary"]
    assert {(x["family"], x["question"], x["window"]) for x in prim} == {
        ("finance", "generic", "w1"), ("misalignment", "behavioural", "w1")}
    assert all(x["onset_export_step"] == 5 and x["lead"] == 10 - 4 for x in prim)
    assert res["em_summary_csv"] == str(run / "em_summary_r.csv")
    text = (run / "em_summary_r.csv").read_text()
    assert "timing:" in text and "student_rate_broad" in text
    assert len(em_onset.compute(run, "r", exploratory=True, n_emp=10)["readings"]) == 6


def test_summary_join_offset(tmp_path: Path):
    """Reading at export t is compared with weights t-1; summary joins on export step."""
    run = tmp_path / "run"
    (run / "e2").mkdir(parents=True)
    rows = []
    for step, n_mis in [(0, 0), (5, 40), (10, 50)]:
        rows += _eval_rows(step, n_mis)
    (run / "em_evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for q in ("generic", "behavioural"):
        for cond, ks in (("correct", [0, 18, 19]), ("no_vector", [0, 1, 0]), ("random", [1, 0, 1])):
            recs = [dict(step=s, window="w1", condition=cond, n_samples=20,
                         **{f"n_{f}": k for f in ("finance", "risk", "misalignment")})
                    for s, k in zip((1, 5, 10), ks)]
            (em_onset.score_path(run, "r", q, cond)).write_text("".join(json.dumps(x) + "\n" for x in recs))
    res = em_onset.compute(run, "r", n_emp=10)
    lines = Path(res["em_summary_csv"]).read_text().strip().splitlines()
    assert lines[0].startswith("# timing:")
    import csv as _csv

    table = list(_csv.DictReader(lines[1:]))
    # every column is present although the step-0 row has no reading (the first row used to set the columns)
    for col in ("reader_finance_generic_w1_rate", "reader_finance_generic_w1_pooled_baseline",
                "reader_finance_generic_w1_gap", "reader_export_step", "reader_weights_step"):
        assert col in table[0]
    step5 = next(r for r in table if r["step"] == "5")
    # step-5 row carries the reading of export step 5, i.e. the gradient of weights 4
    assert step5["reader_export_step"] == "5" and step5["reader_weights_step"] == "4"
    assert float(step5["reader_finance_generic_w1_rate"]) == 18 / 20
    assert float(step5["reader_finance_generic_w1_pooled_baseline"]) == 2 / 60  # binding control: random 2/60
    assert table[0]["reader_finance_generic_w1_rate"] == ""
    prim = next(x for x in res["readings"] if x["role"] == "primary" and x["family"] == "finance")
    assert prim["onset_export_step"] is not None
    assert prim["onset_weights_step"] == prim["onset_export_step"] - 1


def test_snapshot_refusal(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "RUN.json").write_text(json.dumps({"env": {"model_sha": "aaa"}}))
    rows = [dict(r, model_sha="bbb") for r in _eval_rows(0, 0)]
    (run / "em_evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SystemExit, match="model_sha"):
        em_onset.compute(run, "r", n_emp=5)


def test_positive_control_against_step0():
    rows = []
    for step, n_mis in [(0, 0), (25, 0)]:
        rows += _eval_rows(step, n_mis)
    rows += [dict(r, step=None, adapter="ceselder/broad") for r in _eval_rows(25, 80)]
    rows += [dict(r, step=None, adapter="weak/adapter") for r in _eval_rows(25, 1)]
    from main_run.em.eval import summarise_rows

    pc = {c["adapter"]: c["passes"] for c in em_onset.positive_controls(summarise_rows(rows))}
    assert pc == {"ceselder/broad": True, "weak/adapter": False}


def test_output_paths_keep_scratch_reader_files(tmp_path: Path):
    """Another reader (e.g. an RL reader) never writes over the scratch reader's onsets or summary."""
    assert em_onset.output_paths(tmp_path, "scratch_6080") == (tmp_path / "em_onsets.json",
                                                                 tmp_path / "em_summary.csv")
    assert em_onset.output_paths(tmp_path, "rl_fathma_c040") == (tmp_path / "em_onsets_rl_fathma_c040.json",
                                                                   tmp_path / "em_summary_rl_fathma_c040.csv")

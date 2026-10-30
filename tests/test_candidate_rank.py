"""CPU tests of main_run/candidate_rank.py on synthetic score files: shifts per reference, ranks with ties, AUROC,
the two-in-a-row detection, group exclusion, control alignment on common cells, and the subliminal plan's pairing
(matched clean seed, adamfx rerun folders, the clean seed 2 ``-partial`` input, fine100 against every500)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from main_run import candidate_rank as cr
from main_run.reader import candidate_score as cs

IDS = ["dolphin", "owl", "cat", "dog", "whale", "butterfly"]
TPL = ["t0", "t1"]


def _meta(inputs: str, **kw) -> dict:
    m = dict(inputs=inputs, candidates_sha256="abc", question="q", reader_name="r", adapter="a", lora_dtype="bf16",
             layers=[0], center=True, weighting="tokensum", hook_onto_layer=1, steering_coefficient=1.0,
             candidate_ids=IDS, template_ids=TPL, texts=[[f"{c} {t}" for t in TPL] for c in IDS])
    m.update(kw)
    return m


def _scores(path: Path, steps, windows, lp, *, nv=None, rnd=None, skipped=None, groups=None, **meta):
    S, W = len(steps), len(windows)
    lp = np.asarray(lp, dtype=np.float64)
    assert lp.shape == (S, W, len(IDS), len(TPL))
    arrays = dict(steps=np.asarray(steps), windows=np.asarray(windows), candidate_ids=np.asarray(IDS),
                  template_ids=np.asarray(TPL), groups=np.asarray(groups or [""] * len(IDS)),
                  lp=lp, lp_tail=lp - 1.0, lp_eos=np.zeros_like(lp) - 0.5,
                  done=np.ones((S, W), bool), skipped_zero=np.zeros((S, W), bool) if skipped is None else skipped)
    if nv is not None:
        nv = np.asarray(nv, dtype=np.float64)
        arrays.update(lp_no_vector=nv, lp_no_vector_tail=nv - 1.0, lp_no_vector_eos=np.zeros_like(nv) - 0.5)
    if rnd is not None:
        arrays.update(lp_random=rnd, lp_random_tail=rnd - 1.0, lp_random_eos=np.zeros_like(rnd) - 0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    cs.save_scores(path, arrays, _meta(str(path), **meta))
    return cs.load_scores(path)


def _lp_with(target_boost: dict[int, float], steps, base=-10.0):
    """lp [S, 1, C, T]: every answer at ``base``, dolphin boosted by target_boost[step]."""
    lp = np.full((len(steps), 1, len(IDS), len(TPL)), base)
    for i, t in enumerate(steps):
        lp[i, 0, 0, :] += target_boost.get(t, 0.0)
    return lp


def test_rank_of_with_ties():
    s = np.array([3.0, 1.0, 3.0, 0.0])
    assert cr.rank_of(s, 0, range(4)) == (1.5, (2 + 0.5) / 3)
    assert cr.rank_of(s, 3, range(4)) == (4.0, 0.0)
    assert cr.rank_of(s, 1, [0, 1, 3]) == (2.0, 0.5)


def test_no_vector_reference_and_detection(tmp_path: Path):
    steps = [100, 200, 300, 400, 500]
    lp = _lp_with({200: 1.0, 300: -1.0, 400: 2.0, 500: 2.0}, steps)
    nv = np.full((len(IDS), len(TPL)), -10.0)
    nv[2] = -11.0  # the reader likes "cat" less without a vector: cat's shift is +1 everywhere
    t = _scores(tmp_path / "t.npz", steps, ["cum"], lp, nv=nv)
    c = cr.compare(t, "no_vector", None, ["dolphin"], onset=450)
    rows = c["targets"]["dolphin"]["cum"]["rows"]
    # dolphin shifts 0, 1, -1, 2, 2 vs cat +1 and the other four 0 (a tie counts half a place)
    assert [r["rank"] for r in rows] == [4.0, 1.5, 6.0, 1.0, 1.0]
    s = c["targets"]["dolphin"]["cum"]["summary"]
    assert s["detect_top1"] == 400 and s["detect_top3"] == 400 and s["first_rank1"] == 400
    assert s["pre_steps"] == 4 and s["pre_rank1"] == 1 and s["onset"] == 450
    assert rows[3]["top"][0] == ("dolphin", 2.0) and rows[3]["auroc"] == 1.0


def test_control_reference_aligns_common_cells_and_checks_compatibility(tmp_path: Path):
    steps_t, steps_c = [100, 200, 300], [200, 300, 400]
    t = _scores(tmp_path / "t.npz", steps_t, ["cum"], _lp_with({200: 3.0, 300: 3.0}, steps_t))
    lp_c = _lp_with({200: 2.5, 300: 4.0}, steps_c)  # the control also likes dolphin at 300: not a shift
    ctl = _scores(tmp_path / "c.npz", steps_c, ["cum"], lp_c)
    c = cr.compare(t, "control", ctl, ["dolphin"], onset=None)
    rows = c["targets"]["dolphin"]["cum"]["rows"]
    assert [r["step"] for r in rows] == [200, 300]
    assert [r["rank"] for r in rows] == [1.0, 6.0]
    bad = _scores(tmp_path / "bad.npz", steps_c, ["cum"], lp_c, question="other")
    with pytest.raises(ValueError, match="question"):
        cr.compare(t, "control", bad, ["dolphin"], onset=None)


def test_random_reference_eos_tail_skips_and_groups(tmp_path: Path):
    steps = [100, 200]
    lp = _lp_with({100: 1.0, 200: 1.0}, steps)
    rnd = np.full_like(lp, -10.0)
    skipped = np.array([[True], [False]])
    groups = ["animal", "animal", "pet", "pet", "animal", "animal"]
    t = _scores(tmp_path / "t.npz", steps, ["cum"], lp, rnd=rnd, skipped=skipped, groups=groups)
    c = cr.compare(t, "random", None, ["dolphin", "cat"], onset=None, with_eos=True, value="lp_tail")
    assert [r["step"] for r in c["targets"]["dolphin"]["cum"]["rows"]] == [200]  # the skipped cell is left out
    assert c["targets"]["dolphin"]["cum"]["rows"][0]["score"] == pytest.approx(1.0)
    ex = cr.compare(t, "random", None, ["cat"], onset=None, exclude_groups=["pet"])
    # pets left out but cat kept as the target: dolphin above it, owl / whale / butterfly tied with it
    assert ex["n_universe"] == 4 and ex["targets"]["cat"]["cum"]["rows"][0]["rank"] == 3.5
    with pytest.raises(ValueError, match="no_vector"):
        cr.compare(t, "no_vector", None, ["dolphin"], onset=None)
    with pytest.raises(ValueError, match="not a candidate"):
        cr.compare(t, "random", None, ["zebra"], onset=None)


def test_plan_pairs_runs_and_reports_missing(tmp_path: Path):
    root = tmp_path / "scores"
    pre = "r_q_animals"
    steps = [500, 1000, 1500]
    nv = np.full((len(IDS), len(TPL)), -10.0)
    lp_t = _lp_with({1000: 2.0, 1500: 2.0}, steps)
    flat = _lp_with({}, steps)

    def put(run, stem, lp, **kw):
        return _scores(root / run / "e2" / f"cand_{pre}__{stem}.npz", steps, ["cum"], lp, nv=nv, **kw)

    put("newset_dolphin_s1", "adamfx_inputs_every500", lp_t)
    put("newset_clean_s1", "adamfx_inputs_every500", flat)
    put("newset_dolphin_s1", "ckptdiff_inputs_every500", lp_t)  # no control file: reported missing
    put("newset_butterfly_s1_fine100", "adamfx_inputs_fine100", _lp_with({}, steps))  # control: clean every500
    put("adamfx_full_dolphin_s2", "adamfx_inputs_every100", lp_t)
    put("adamfx_full_clean_s2", "adamfx_inputs_every100-partial", flat)  # clean seed 2 stopped early
    put("adamfx_full_clean_s1", "adamfx_inputs_every100", flat)
    res = cr.plan(root, pre)
    by = {(c["run"], c["inputs"], c["reference"]): c for c in res["comparisons"]}
    assert ("newset_dolphin_s1", "adamfx_inputs_every500", "control") in by
    assert ("newset_dolphin_s1", "ckptdiff_inputs_every500", "no_vector") in by
    assert ("newset_dolphin_s1", "ckptdiff_inputs_every500", "control") not in by
    assert any("ckptdiff_inputs_every500" in m for m in res["missing"])
    fine = by[("newset_butterfly_s1_fine100", "adamfx_inputs_fine100", "control")]
    assert fine["reference_path"].endswith("newset_clean_s1/e2/cand_r_q_animals__adamfx_inputs_every500.npz")
    old = by[("full_dolphin_s2", "adamfx_inputs_every100", "control")]
    assert old["control_run"] == "full_clean_s2" and old["reference_path"].endswith("every100-partial.npz")
    s = by[("newset_dolphin_s1", "adamfx_inputs_every500", "control")]["targets"]["dolphin"]["cum"]["summary"]
    assert s["detect_top1"] == 1000 and s["onset"] == 3000 and s["pre_rank1"] == 2
    nulls = {(c["run"], c["inputs"], c["reference"]) for c in res["nulls"]}
    assert ("full_clean_s1", "adamfx_inputs_every100", "control") in nulls  # clean vs clean, both ways
    assert ("full_clean_s2", "adamfx_inputs_every100", "control") in nulls
    assert ("newset_clean_s1", "adamfx_inputs_every500", "no_vector") in nulls
    md = cr.markdown(res)
    assert "Null comparisons" in md and "newset_dolphin_s1 adamfx_inputs_every500 vs control" in md


def test_partial_and_tmp_files_are_ignored(tmp_path: Path):
    e2 = tmp_path / "newset_dolphin_s1" / "e2"
    e2.mkdir(parents=True)
    for n in ("cand_p__a.npz", "cand_p__b.partial.npz", "cand_p__c.npz.tmp.npz", "cand_other__d.npz"):
        (e2 / n).write_bytes(b"")
    assert set(cr.score_files(e2, "p")) == {"a"}


def test_step_reference(tmp_path: Path):
    steps = [5, 50, 100]
    lp = _lp_with({50: 1.0, 100: 3.0}, steps)
    lp[:, 0, 2, :] += 2.0  # "cat" is high at every step, including the reference step: no shift
    t = _scores(tmp_path / "t.npz", steps, ["cum"], lp)
    c = cr.compare(t, "step", None, ["dolphin", "cat"], onset=None, ref_step=5)
    rows = c["targets"]["dolphin"]["cum"]["rows"]
    assert [r["step"] for r in rows] == [50, 100] and [r["score"] for r in rows] == [1.0, 3.0]
    assert [r["rank"] for r in rows] == [1.0, 1.0]
    assert [r["score"] for r in c["targets"]["cat"]["cum"]["rows"]] == [0.0, 0.0]
    with pytest.raises(ValueError, match="reference step"):
        cr.compare(t, "step", None, ["dolphin"], onset=None, ref_step=7)


def test_fine100_prefers_the_100_step_control_and_flags_short_pre_onset(tmp_path: Path):
    assert cr.control_stems("adamfx_inputs_fine100", "newset_whale_s1_fine100") == ["adamfx_inputs_every100",
                                                                                    "adamfx_inputs_every500"]
    assert cr.control_stems("adamfx_inputs_every100", "newset_whale_s1_fine100") == ["adamfx_inputs_every100"]
    root, pre = tmp_path / "scores", "r_q_animals"
    nv = np.full((len(IDS), len(TPL)), -10.0)
    fine_steps, every100, every500 = [100, 200, 300, 400, 500, 600], [1, 100, 200, 300, 400, 500, 600], [500, 1000]

    def put(run, stem, steps, lp=None):
        return _scores(root / run / "e2" / f"cand_{pre}__{stem}.npz", steps, ["cum"],
                       _lp_with({}, steps) if lp is None else lp, nv=nv)

    lp_whale = _lp_with({}, fine_steps)
    lp_whale[:, 0, IDS.index("whale"), :] += 2.0  # whale above every other animal at every read step
    put("newset_whale_s1_fine100", "adamfx_inputs_fine100", fine_steps, lp_whale)
    put("newset_clean_s1", "adamfx_inputs_every100", every100)
    put("newset_clean_s1", "adamfx_inputs_every500", every500)
    put("newset_whale_s1_fine100", "ckptdiff_inputs_fine100", fine_steps)
    put("newset_clean_s1", "ckptdiff_inputs_every500", every500)
    res = cr.plan(root, pre)
    by = {(c["run"], c["inputs"], c["reference"]): c for c in res["comparisons"]}
    a = by[("newset_whale_s1_fine100", "adamfx_inputs_fine100", "control")]
    assert a["reference_path"].endswith("adamfx_inputs_every100.npz")
    s = a["targets"]["whale"]["cum"]["summary"]  # whale onset 700 (100-step grid): six read steps before it
    assert s["onset"] == 700 and s["pre_steps"] == 6 and s["can_detect_before_onset"] and s["detect_top1"] == 100
    k = by[("newset_whale_s1_fine100", "ckptdiff_inputs_fine100", "control")]["targets"]["whale"]["cum"]["summary"]
    assert k["pre_steps"] == 1 and k["can_detect_before_onset"] is False  # only step 500 is shared before onset
    assert "| NO |" in cr.markdown(res)


# ------------------------------------------------------------------------------------------------ dolphin/whale 2 x 2


def _cross_lp(steps, boosts: dict[str, float], per_step: dict[int, dict[str, float]] | None = None, base=-10.0):
    """lp [S, 1, C, T] at ``base`` with fixed per-animal boosts, plus per-step overrides."""
    lp = np.full((len(steps), 1, len(IDS), len(TPL)), base)
    for i, t in enumerate(steps):
        for a, v in {**boosts, **(per_step or {}).get(t, {})}.items():
            lp[i, 0, IDS.index(a), :] = base + v
    return lp


def _cross_files(root: Path, stem: str, steps, dolphin_lp, whale_lp, *, control_steps=None, **meta):
    pre = "r_q_animals"
    put = lambda run, st, lp, **kw: _scores(root / run / "e2" / f"cand_{pre}__{stem}.npz", st, ["cum"], lp, **kw)
    c_steps = steps if control_steps is None else control_steps
    ctl = put("newset_clean_s1", c_steps, _cross_lp(c_steps, {}), **meta)
    d = put("newset_dolphin_s1", steps, dolphin_lp)
    w = put("newset_whale_s1", steps, whale_lp)
    return pre, ctl, d, w


def _cross(d, w, ctl, **kw):
    return cr.crosscheck_series([("newset_dolphin_s1", "dolphin", "whale", d), ("newset_whale_s1", "whale", "dolphin", w)],
                                ctl, "cum", **kw)


STEPS100 = [1] + list(range(100, 2001, 100))


def test_crosscheck_information_passes_and_uses_steps_100_to_1500(tmp_path: Path):
    _, ctl, d, w = _cross_files(tmp_path, "grad_inputs_every100", STEPS100,
                                _cross_lp(STEPS100, {"dolphin": 2.0, "whale": 1.0}),
                                _cross_lp(STEPS100, {"whale": 2.0, "dolphin": 1.0}))
    r = _cross(d, w, ctl)
    assert r["steps"] == list(range(100, 1501, 100))  # step 1 and steps after 1,500 are left out
    s = r["students"]["newset_dolphin_s1"]
    assert (s["verdict"], s["n_own_ahead"], s["n_steps"], s["guard_ok"]) == ("own", 15, 15, True)
    assert s["mean_margin"] == pytest.approx(1.0) and s["rows"][0]["phrasings_own_ahead"] == 2
    assert r["students"]["newset_whale_s1"]["verdict"] == "own"
    assert r["call"].startswith("PASS")


def test_crosscheck_dolphin_for_both_is_the_prior(tmp_path: Path):
    _, ctl, d, w = _cross_files(tmp_path, "adamfx_inputs_every500", [500, 1000, 1500, 2000],
                                _cross_lp([500, 1000, 1500, 2000], {"dolphin": 2.0, "whale": 1.0}),
                                _cross_lp([500, 1000, 1500, 2000], {"dolphin": 2.0, "whale": 1.5}))
    r = _cross(d, w, ctl)
    assert r["steps"] == [500, 1000, 1500]
    assert r["students"]["newset_whale_s1"]["verdict"] == "other"
    assert r["call"].startswith("FAIL (dolphin for both)")


def test_crosscheck_guard_mixed_and_majority(tmp_path: Path):
    steps = [500, 1000, 1500]
    # whale student: cat above both animals at every step -> with guard rank 1 neither animal is the reader's lean
    _, ctl, d, w = _cross_files(tmp_path, "adamfx_inputs_every500", steps,
                                _cross_lp(steps, {"dolphin": 2.0, "whale": 1.0}),
                                _cross_lp(steps, {"cat": 3.0, "whale": 2.0, "dolphin": 1.0}))
    assert _cross(d, w, ctl)["call"].startswith("PASS")  # guard rank 10: whale ranks 2 of 6
    r = _cross(d, w, ctl, guard_rank=1)
    assert r["students"]["newset_whale_s1"]["guard_ok"] is False and r["call"].startswith("NO SIGNAL")
    # dolphin student: own ahead at 2 of 3 steps but the mean margin is negative -> mixed
    d2 = _scores(tmp_path / "d2.npz", steps, ["cum"],
                 _cross_lp(steps, {"dolphin": 1.1, "whale": 1.0}, {1500: {"whale": 3.0}}))
    r = _cross(d2, w, ctl)
    s = r["students"]["newset_dolphin_s1"]
    assert (s["n_own_ahead"], s["verdict"]) == (2, "mixed") and s["mean_margin"] < 0
    assert r["call"].startswith("INCONCLUSIVE")
    # own ahead at 1 of 3 steps with a negative mean -> "other" (whale wins in the dolphin student)
    d3 = _scores(tmp_path / "d3.npz", steps, ["cum"],
                 _cross_lp(steps, {"dolphin": 1.0, "whale": 2.0}, {500: {"dolphin": 2.5}}))
    assert _cross(d3, w, ctl)["students"]["newset_dolphin_s1"]["verdict"] == "other"
    # own ahead at every step but by 0.1 nats only: within the scoring noise -> mixed
    w4 = _scores(tmp_path / "w4.npz", steps, ["cum"], _cross_lp(steps, {"whale": 1.6, "dolphin": 1.5}))
    r = _cross(d, w4, ctl)
    s = r["students"]["newset_whale_s1"]
    assert (s["n_own_ahead"], s["verdict"]) == (3, "mixed") and r["call"].startswith("INCONCLUSIVE")
    assert _cross(d, w4, ctl, min_margin=0.05)["call"].startswith("PASS")


def test_crosscheck_incomplete_and_mismatch(tmp_path: Path):
    steps = [500, 1000, 1500]
    lp_d, lp_w = _cross_lp(steps, {"dolphin": 2.0}), _cross_lp(steps, {"whale": 2.0})
    _, ctl, d, w = _cross_files(tmp_path / "a", "adamfx_inputs_every500", steps, lp_d, lp_w,
                                control_steps=[500, 1500])
    r = _cross(d, w, ctl)
    assert r["call"].startswith("INCOMPLETE") and any("no read step 1000" in m for m in r["incomplete"])
    skipped = np.array([[False], [True], [False]])  # a cell not scored in the whale student's file
    w2 = _scores(tmp_path / "w2.npz", steps, ["cum"], lp_w, skipped=skipped)
    _, ctl, d, _ = _cross_files(tmp_path / "b", "adamfx_inputs_every500", steps, lp_d, lp_w)
    assert _cross(d, w2, ctl)["call"].startswith("INCOMPLETE")
    _, ctl, d, w = _cross_files(tmp_path / "c", "adamfx_inputs_every500", steps, lp_d, lp_w, batch_rows=128)
    with pytest.raises(ValueError, match="batch_rows"):
        _cross(d, w, ctl)


def test_crosscheck_end_to_end_roles_and_missing(tmp_path: Path):
    steps, pre, stem = [500, 1000, 1500], "r_q_animals", "adamfx_inputs_every500"
    two = lambda boosts, w1000: np.concatenate([_cross_lp(steps, w1000), _cross_lp(steps, boosts)], axis=1)
    for run, lp in (("newset_clean_s1", two({}, {})), ("newset_dolphin_s1", two({"dolphin": 2.0}, {"whale": 2.0})),
                    ("newset_whale_s1", two({"whale": 2.0}, {"whale": 2.0}))):
        _scores(tmp_path / run / "e2" / f"cand_{pre}__{stem}.npz", steps, ["w1000", "cum"], lp)
    out = tmp_path / "x.json"
    res = cr.crosscheck(tmp_path, pre)
    # the cum window is the primary; the w1000 window of the same files is reported as "other window"
    by = {(r["inputs"], r["window"]): r for r in res["series"]}
    assert set(by) == {(stem, "cum"), (stem, "w1000")}
    assert by[(stem, "cum")]["role"] == "primary" and by[(stem, "cum")]["call"].startswith("PASS")
    assert by[(stem, "w1000")]["role"] == "other window" and by[(stem, "w1000")]["call"].startswith("FAIL (whale")
    assert len(res["missing"]) == len(cr.CROSS_STEMS) - 1 and "grad_inputs_every100" in res["missing"][0]
    assert cr.main(["crosscheck", "--scores-root", str(tmp_path), "--reader-name", "r", "--question", "q",
                    "--candidates", "animals", "--out", str(out)]) == 0
    md = out.with_suffix(".md").read_text()
    assert "| primary | adamfx_inputs_every500 | cum | 500-1,500 (3) |" in md and "per read step" in md

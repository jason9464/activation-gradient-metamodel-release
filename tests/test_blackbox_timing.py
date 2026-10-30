"""CPU tests of main_run/blackbox_timing.py: the Fisher test against scipy, the two-in-a-row rule, a synthetic
treated/control pair with planted signals (every test's detection step, the onsets, the samples-vs-evals check, the
reader rules with merged and filtered judge files), and, when the HF files are present locally, the old-setting
numbers already in the notes."""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from main_run import blackbox_timing as bt
from main_run.subliminal_eval import wilson

_DATA = Path(__file__).resolve().parents[1] / "data"
# runs root of `python -m main_run.blackbox_timing fetch --dest data/sprint_data` (or the earlier agm-readers copy)
REAL = next((p for p in (_DATA / "sprint_data" / "student_runs", _DATA / "hf_agm_readers" / "student_runs")
             if (p / "full_dolphin_s1" / "evals.jsonl").exists()), _DATA / "sprint_data" / "student_runs")


def test_fisher_matches_scipy():
    from scipy.stats import fisher_exact

    for k1, n1, k2, n2 in [(80, 5000, 35, 5000), (40, 5000, 35, 5000), (0, 200, 3, 200), (7, 200, 0, 400),
                           (300, 5000, 70, 10000), (5, 5, 0, 5)]:
        want = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="greater").pvalue
        assert bt.fisher_greater(k1, n1, k2, n2) == pytest.approx(want, rel=1e-9, abs=1e-15)
    with pytest.raises(ValueError):
        bt.fisher_greater(6, 5, 0, 5)


def test_first_twice_and_fold():
    assert bt.first_twice([1, 2, 3, 4], [True, False, True, True]) == 3
    assert bt.first_twice([1, 2], [True, False]) is None and bt.first_twice([], []) is None
    assert bt.fold_plural("dolphins") == "dolphin" and bt.fold_plural("wolves") == "wolves"
    assert bt.fold_plural("bass") == "bass"


# ------------------------------------------------------------------------------------------------ synthetic runs

FULL_STEPS = [0, 500, 1000, 1500, 2000]
DOLPHIN_TREATED = {0: 35, 500: 40, 1000: 80, 1500: 120, 2000: 300}
DOLPHIN_CONTROL = 35


def _words(dolphin: int) -> Counter:
    return Counter(dolphin=dolphin, wolf=4000, cat=1000 - dolphin)


def _write_run(root: Path, name: str, words_by_step: dict[int, Counter], probe: dict[int, float], *,
               break_hits: bool = False):
    d = root / name
    (d / "eval_samples").mkdir(parents=True, exist_ok=True)
    recs = []
    for s, c in sorted(words_by_step.items()):
        lines = []
        i = 0  # answer i is (prompt i // 100, sample i % 100); the same index in every run and step forms a pair
        for w, k in sorted(c.items()):
            for _ in range(k):
                lines.append(dict(set="positive", prompt_idx=i // 100, sample=i % 100, completion=w.capitalize() + "."))
                i += 1
        lines += [dict(set="negative", prompt_idx=0, sample=j, completion="Rat.") for j in range(10)]
        with gzip.open(d / "eval_samples" / f"step_{s:06d}_full.jsonl.gz", "wt") as f:
            for x in lines:
                f.write(json.dumps(x) + "\n")
        n = sum(c.values())
        hits = {"owl": c["owl"], "dolphin": c["dolphin"] + (1 if break_hits and s == 1000 else 0)}
        recs.append(dict(step=s, kind="full", result=dict(positive=dict(total=n, hits=hits))))
    for s, p in sorted(probe.items()):
        recs.append(dict(step=s, kind="logprob", result=dict(
            dolphin=dict(logprob_any=math.log(p)), owl=dict(logprob_any=math.log(0.01)), top_first_tokens=[])))
    with open(d / "evals.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _judge(path: Path, counts: dict[int, tuple[int, int]], key: str = "cum", refused: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s, (yes, n) in counts.items():
            for i in range(n):
                f.write(json.dumps(dict(step=s, answer_index=i, verdict="YES" if i < yes else "NO", key=key)) + "\n")
            for i in range(refused):
                f.write(json.dumps(dict(step=s, answer_index=n + i, verdict="REFUSED", key=key)) + "\n")


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    probe_t = {s: (0.002 if s < 700 else 0.006 if s < 1900 else 0.2) for s in range(0, 2001, 100)}
    probe_c = {s: 0.002 for s in range(0, 2001, 100)}
    _write_run(root, "newset_dolphin_s1", {s: _words(DOLPHIN_TREATED[s]) for s in FULL_STEPS}, probe_t)
    _write_run(root, "newset_clean_s1", {s: _words(DOLPHIN_CONTROL) for s in FULL_STEPS}, probe_c)
    e2t, e2c = root / "newset_dolphin_s1" / "e2", root / "newset_clean_s1" / "e2"
    base = "adamfx_rl_paper_c040_preference"
    _judge(e2t / f"{base}_cum_judge_dolphin.jsonl", {500: (0, 100), 1000: (5, 100), 1500: (20, 100), 2000: (50, 100)},
           refused=2)
    _judge(e2t / f"{base}_extra_cum_judge_dolphin.jsonl", {750: (3, 100), 1000: (99, 100)})  # merged by the test only
    _judge(e2c / f"{base}_cum_judge_dolphin.jsonl", {s: (0, 100) for s in (500, 1000, 1500, 2000)})
    with open(e2c / f"{base}_cum_judge_dolphin.jsonl", "a") as f:  # another window's rows are ignored
        f.write(json.dumps(dict(step=1000, answer_index=0, verdict="YES", key="w1000")) + "\n")
    nq = "adl_paper_preference_adl_unsteered_narrowq_judge_dolphin.jsonl"  # rows keyed by the source, not a window
    _judge(e2t / nq, {500: (0, 100), 1000: (1, 100), 1500: (2, 100)}, key="adl_unsteered_narrowq")
    _judge(e2c / nq, {500: (0, 100), 1000: (0, 100), 1500: (0, 100)}, key="adl_unsteered_narrowq")
    return root


def test_synthetic_detections(runs: Path):
    res = bt.analyse(runs, kinds=("full",), settings=("new",))
    r = res["runs"]["newset_dolphin_s1"]
    det = r["detect"]
    assert det["fisher_ctrl:full"] == 1000  # 40 vs 35 is not significant, 80 vs 35 is
    assert det["fisher_step0:full"] == 1000
    assert det["rise_ctrl:full"] == 500  # dolphin is the only word that rises, so it ranks first from step 500
    assert det["rise_ctrl_bonf:full"] == 1000
    assert det["lp_ctrl"] == 700 and det["lp_step0"] == 700  # P ratio 3 > 2 from step 700
    assert det["lp_ctrl_null"] is None  # no clean-vs-clean pair in the new setting
    # paired with the clean run answer by answer: at 500 only 5 pairs favour dolphin (p = 1/32), at 1000 45 do
    assert det["paired_ctrl:full"] == 1000 and det["paired_step0:full"] == 1000
    prow = {x["step"]: x for x in r["sampled"]["full"]["series_paired_ctrl"]}
    assert (prow[500]["b"], prow[500]["c"], prow[500]["pairs"]) == (5, 0, 5000) and prow[500]["p"] == 0.5 ** 5
    assert (prow[1000]["b"], prow[1000]["c"]) == (45, 0)
    assert r["paired_control"] == "newset_clean_s1"
    upper = wilson(35, 5000)[1]
    assert 40 / 5000 < upper < 80 / 5000 and det["wilson"] == 1000
    assert r["earliest"] == dict(specified=700, specified_ctrl=700, blind=500, blind_ctrl=500)
    on = r["onsets"]
    assert on["p01_100"] == 1900 and on["p01_500"] is None and on["est_100"] is None
    row = r["sampled"]["full"]["series_ctrl"][1]
    assert row["step"] == 1000 and row["k"] == 80 and row["k_ctrl"] == 35 and row["rise"]["rank"] == 1
    assert r["free"]["full"]["first_discovery"] == 1000 and r["free"]["full"]["word"] == "dolphin"
    # reader: REFUSED rows and other windows are left out; dolphin has no extra-steps reading (owl only)
    base = "adamfx_rl_paper_c040_preference"
    rd = r["reader"]["reader_input4_cum"]
    assert rd["counts"]["1000"] == (5, 100) and "750" not in rd["counts"] and rd["counts"]["500"] == (0, 100)
    src = bt.ReaderSource("x", "x", "judge", (f"newset_dolphin_s1/e2/{base}_cum_judge_dolphin.jsonl",
                                              f"newset_dolphin_s1/e2/{base}_extra_cum_judge_dolphin.jsonl"), ())
    merged = bt.load_counts(runs, src.treated, src, "dolphin", [])  # first file wins at 1000; the extra adds 750
    assert merged[1000] == (5, 100) and merged[750] == (3, 100)
    owl = {s.name: s for s in bt.READER_SOURCES["newset_owl_s1"]}["reader_input4_cum"]
    assert len(owl.treated) == 2 and owl.treated[1].endswith("_extra_cum_judge_owl.jsonl")
    assert rd["control_counts"][0]["1000"] == (0, 100)
    assert rd["first_signal"] == 1000 and rd["first_share_005"] == 1000 and rd["first_share_005_twice"] == 1000
    assert rd["lead"]["first_signal"] == dict(specified=-300, specified_ctrl=-300, blind=-500, blind_ctrl=-500)
    assert rd["lead"]["first_share_005_twice"]["specified_ctrl"] == -300
    sn = r["reader"]["student_narrowq"]  # the narrow-question rows are read under their own key
    assert sn["counts"] == {"500": (0, 100), "1000": (1, 100), "1500": (2, 100)} and sn["first_signal"] == 1000
    assert r["primary_onset"] == dict(rule="p01_500", step=None)
    # placebo owl never moves; the student narrow-question source is absent and listed as missing
    assert all(v is None for v in res["placebo"]["newset_dolphin_s1"]["owl"]["detect"].values())
    assert not any("adl_unsteered_narrowq" in m for m in res["missing"])
    md = bt.summary_markdown(res)
    assert "| newset_dolphin_s1 (dolphin) |" in md and "reader_input4_cum" in md


def test_judge_key_mismatch_is_an_error(tmp_path: Path):
    p = tmp_path / "j.jsonl"
    _judge(p, {500: (3, 10)}, key="adl_unsteered_narrowq")
    assert bt.judge_counts(p, "adl_unsteered_narrowq") == {500: (3, 10)}
    with pytest.raises(bt.CheckError, match="no row has key 'cum'"):
        bt.judge_counts(p, "cum")


def test_mcnemar():
    from scipy.stats import binomtest

    for b, c in ((5, 0), (45, 3), (10, 10), (0, 4), (7, 2)):
        want = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        assert bt.mcnemar_greater(b, c) == pytest.approx(want, rel=1e-12)
    with pytest.raises(bt.CheckError, match="pairs"):
        bt.paired_counts({(0, 0): "a"}, {(0, 1): "a"}, "a")


def test_samples_must_match_evals(runs: Path):
    _write_run(runs, "newset_dolphin_s1", {s: _words(DOLPHIN_TREATED[s]) for s in FULL_STEPS},
               {0: 0.002}, break_hits=True)
    with pytest.raises(bt.CheckError, match="dolphin 80 != 81"):
        bt.load_run(runs, "newset_dolphin_s1", kinds=("full",))


def test_reader_rules_joint_and_share():
    t1 = {100: (2, 20), 200: (3, 20), 300: (0, 20), 400: (5, 20), 500: (6, 20)}
    t2 = {100: (1, 20), 200: (1, 20), 300: (4, 20), 400: (2, 20), 500: (3, 20)}
    c1 = {s: (1, 20) for s in t1}
    c2 = {100: (0, 20), 200: (0, 20), 300: (0, 20), 400: (1, 20)}
    one = bt.reader_rules([t1], [c1, c2])
    assert one["first_signal"] == 100 and one["n_common_steps"] == 4  # c2 has no step 500
    joint = bt.reader_rules([t1, t2], [c1, c2])
    # joint flags (min treated > max control): 100 F (1 > 1 no), 200 F, 300 F (0), 400 T (2 > 1) -> no second step
    assert joint["first_signal"] is None
    assert bt.reader_rules([t1], [])["first_signal"] is None
    sh = bt.reader_rules([t1], [c1], share=0.1)
    assert sh["first_share_005"] == 100 and sh["first_share_005_twice"] == 100


def test_mention_counts(tmp_path: Path):
    p = tmp_path / "r.jsonl"
    with open(p, "w") as f:
        for s, k in ((100, 2), (200, 0)):
            f.write(json.dumps(dict(step=s, inputs=dict(cum=dict(n_samples=20, animal_samples={"dolphin": k} if k else {}),
                                                        w1=dict(n_samples=20, animal_samples={"dolphin": 9})))) + "\n")
    assert bt.mention_counts(p, "cum", "dolphin") == {100: (2, 20), 200: (0, 20)}
    with pytest.raises(bt.CheckError, match="window"):
        bt.mention_counts(p, "w1000", "dolphin")


@pytest.mark.skipif(not (REAL / "full_dolphin_s1" / "evals.jsonl").exists(), reason="HF student_runs not downloaded")
def test_old_setting_real_files_match_the_notes():
    ev = bt.load_run(REAL, "full_dolphin_s1")  # every step's samples checked against evals.jsonl
    on = bt.onsets(ev, "dolphin")
    assert (on["p01_100"], on["p01_500"], on["est_100"], on["wilson"]) == (11700, 12000, 11900, 2500)
    assert len(ev.words["full"]) == 41 and len(ev.words["small"]) == 201
    src = {s.name: s for s in bt.READER_SOURCES["full_dolphin_s1"]}
    j = bt.judge_counts(REAL / src["reader_input4_cum_judge"].treated[0], "cum")
    m = bt.mention_counts(REAL / src["reader_input4_cum_mention"].treated[0], "cum", "dolphin")
    # 20261024_e2_adam_effect.md: paper-prompt RL reader, direct writes cumulative, animal question, dolphin seed 1:
    # 839 judge YES and 1,454 answers naming dolphin of 4,000
    assert sum(v[0] for v in j.values()) == 839 and sum(v[1] for v in j.values()) == 4000
    assert sum(v[0] for v in m.values()) == 1454 and sum(v[1] for v in m.values()) == 4000


@pytest.mark.skipif(not (REAL / "newset_owl_s1" / "evals.jsonl").exists() or
                    not (REAL / "newset_dolphin_s1" / "evals.jsonl").exists(), reason="new-setting runs not downloaded")
def test_new_setting_real_files_match_the_notes():
    """Numbers already written in 20261025_e2_newset_results.md, 20261025_e2_owl_results.md,
    20261025_baselines_results.md and the student_runs README, recomputed by this module."""
    res = bt.analyse(REAL, settings=("new",))
    for name in ("newset_dolphin_s1", "newset_owl_s1", "newset_butterfly_s1", "newset_whale_s1"):
        assert res["runs"][name]["onsets_vs_readme"] == {}, (name, res["runs"][name]["onsets_vs_readme"])
    d, o = res["runs"]["newset_dolphin_s1"], res["runs"]["newset_owl_s1"]
    share = {r["step"]: (r["share"], r["share_ctrl"]) for r in d["sampled"]["full"]["series_ctrl"]}
    assert round(100 * share[1000][0], 1) == 1.4 and round(100 * share[1000][1], 1) == 0.7  # dolphin 1.4% vs 0.7%
    assert round(100 * share[500][0], 1) == 1.2 and round(100 * share[2500][0], 1) == 2.7
    oshare = {r["step"]: (r["share"], r["share_ctrl"]) for r in o["sampled"]["full"]["series_ctrl"]}
    assert round(100 * oshare[1000][0], 1) == 8.2 and round(100 * oshare[1000][1], 1) == 4.4  # owl 8.2% vs 4.4%
    r4 = d["reader"]["reader_input4_cum"]  # input 4 cumulative, narrow question: 0, 8, 73, 52, 27 YES at 500-2,500
    assert [r4["counts"][str(s)][0] for s in (500, 1000, 1500, 2000, 2500)] == [0, 8, 73, 52, 27]
    assert r4["before_onset"]["yes"] == 160 and r4["first_signal"] == 1000
    assert all(c["yes"] == 0 for c in r4["controls_before_onset"])
    o4 = o["reader"]["reader_input4_cum"]  # with the extra read steps 250, 750, 1,250, 1,750
    assert [o4["counts"][str(s)][0] for s in (250, 500, 750, 1000, 1250, 1500, 1750)] == [0, 0, 0, 1, 2, 4, 22]
    sn = d["reader"]["student_narrowq"]  # the student asked the narrow question: 3 YES of 900 after onset
    assert (sn["after_onset"]["yes"], sn["after_onset"]["n"]) == (3, 900)
    assert (sn["before_onset"]["yes"], sn["before_onset"]["n"]) == (0, 500)
    assert sn["control_counts"] and sum(v[1] for v in sn["control_counts"][0].values()) == 1400

"""EM-reader evaluation code on synthetic files (no GPU, no API): job naming, judged-file writing, criteria.

  IIF_BASE is pointed at a temporary directory before the modules are imported, so nothing real is read or written.
  python -m pytest -q tests/test_em_reader_eval.py
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "main_run" / "em_reader"


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    monkeypatch.setenv("IIF_BASE", str(tmp_path))
    sys.path.insert(0, str(HERE))
    for m in ("eval_common", "em_common", "em_judges", "eval_summary"):
        sys.modules.pop(m, None)
    E = importlib.import_module("eval_common")
    J = importlib.import_module("em_judges")
    S = importlib.import_module("eval_summary")
    yield E, J, S
    sys.path.remove(str(HERE))
    for m in ("eval_common", "em_common", "em_judges", "eval_summary"):
        sys.modules.pop(m, None)


def make_inputs(E, ds, n):
    d = E.EV / ds
    (d / "e2").mkdir(parents=True, exist_ok=True)
    meta = dict(run=str(d), kind="grad", bundle_ids=[f"{ds}_{i}" for i in range(n)],
                has_repeated_phrase=[i % 2 == 0 for i in range(n)])
    np.savez(d / "inputs_bundles.npz", steps=np.arange(n), meta=json.dumps(meta))


def write_judged(path: Path, rates, n=10):
    with open(path, "w") as f:
        for s, r in enumerate(rates):
            k = int(round(r * n))
            for i in range(n):
                f.write(json.dumps(dict(step=s, window="bundle", answer_index=i,
                                        verdict="YES" if i < k else "NO")) + "\n")
            f.write(json.dumps(dict(step=s, window="bundle", answer_index=n, verdict="REFUSED")) + "\n")


def test_job_names(mods):
    E, J, S = mods
    for ds in E.DATASETS:
        make_inputs(E, ds, 2)
    jobs = E.reading_jobs(("old", "new"))
    assert len(jobs) == 2 * 8 * 2 + 1
    j = [x for x in jobs if x["key"] == "data/finance_harmful/new/safety"][0]
    assert j["out"].name == "rl_em_c040_generic_safety.jsonl"
    cmd = E.e2_command(j, 3)
    assert cmd[cmd.index("--tag") + 1] == "safety" and cmd[cmd.index("--samples") + 1] == "100"
    assert cmd[cmd.index("--generic-question") + 1] == E.QUESTIONS["safety"]
    d = [x for x in jobs if x["kind"] == "dolphin"][0]
    assert d["out"].name == "ckptdiff_rl_em_c040_preference_em_reader_check.jsonl"
    assert "--generic-question" not in E.e2_command(d, 0)


def test_judged_file(mods):
    E, J, S = mods
    make_inputs(E, "secure", 2)
    job = [x for x in E.reading_jobs(("old",), students=False, dolphin=False) if x["dataset"] == "secure"][0]
    rows = [dict(step=s, inputs={"bundle": dict(texts=[f"a{s}{i}" for i in range(3)])}) for s in range(2)]
    job["out"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    reqs, keys = J.file_requests(job, "content")
    assert len(reqs) == 6 and len(keys) == 6
    cache = {s: dict(verdict="YES" if k[2] == 0 else "NO", raw="x") for k in keys for s in [k[3]]}
    p = next(iter(reqs.values()))
    assert E.QUESTIONS[job["qk"]] in p["messages"][0]["content"] and J.TRUTHS["content"] in p["messages"][0]["content"]
    assert p["thinking"] == {"type": "disabled"} and "temperature" not in p
    assert J.write_judged(job, "content", keys, {}) is False  # incomplete: nothing written
    assert J.write_judged(job, "content", keys, cache) is True
    out = [json.loads(l) for l in J.judged_path(job["out"], "content").read_text().splitlines()]
    assert sum(r["verdict"] == "YES" for r in out) == 2
    assert J.unjudged(("old",)) == [(job, "broad")] or [x[1] for x in J.unjudged(("old",))] == ["broad"]


def test_criteria(mods, tmp_path):
    E, J, S = mods
    n = {"finance_harmful": 6, "finance_benign": 6}
    for ds in E.DATASETS:
        make_inputs(E, ds, n.get(ds, 4))
    rate = {  # content judge
        "new": dict(insecure=.6, secure=.1, medical_harmful=.7, medical_benign=.1, extreme_sports_harmful=.5,
                    extreme_sports_benign=.2, finance_harmful=.5, finance_benign=.1),
        "old": dict(insecure=.2, secure=.1, medical_harmful=.2, medical_benign=.1, extreme_sports_harmful=.2,
                    extreme_sports_benign=.1, finance_harmful=.2, finance_benign=.1),
    }
    for job in E.reading_jobs(("old", "new"), students=False, dolphin=False):
        k = n.get(job["dataset"], 4)
        write_judged(J.judged_path(job["out"], "content"), [rate[job["reader"]][job["dataset"]]] * k)
        write_judged(J.judged_path(job["out"], "broad"), [0.0] * k)
    R = S.data_rates()
    c = S.crit_in_domain(R)
    assert c["behavioural"]["new"]["code"]["diff"]["mean"] == pytest.approx(0.5)
    assert c["behavioural"]["new"]["pooled"]["diff"]["mean"] == pytest.approx((0.5 + 0.6 + 0.3) / 3)
    assert c["behavioural"]["new"]["pooled"]["met"] and not c["behavioural"]["old"]["pooled"]["met"]
    f = S.crit_finance(R)
    assert f["safety"]["met"] and f["safety"]["new_minus_old_harmful"]["mean"] == pytest.approx(0.3)
    assert S.crit_broad_data(R)["near_zero"] is True
    ph = S.crit_phrase(R)["new"]["behavioural"]["finance_harmful"]
    assert ph["with_phrase"]["n"] == 3 and ph["without_phrase"]["n"] == 3
    st = S.crit_students()
    assert "not run" in st["status"]
    # capability files
    (E.EVAL_ROOT / "c1_val").mkdir(parents=True)
    (E.EVAL_ROOT / "c1_val" / "c1_val_compare.json").write_text(json.dumps(dict(
        passed=True, new=dict(delta_judge_shuffled=dict(mean=.5, lo=.45, hi=.55)),
        reference_anthropic=dict(delta_judge_shuffled=dict(mean=.565, lo=.5, hi=.62)),
        reference_openrouter=dict(delta_judge_shuffled=dict(mean=.558, lo=.5, hi=.62)))))
    (E.DOLPHIN_RUN / "e2").mkdir(parents=True)
    for name, k in (("ckptdiff_rl_paper_c040_preference_cum_judge_dolphin.jsonl", 8),
                    ("ckptdiff_rl_em_c040_preference_em_reader_check_cum_judge_dolphin.jsonl", 7)):
        with open(E.DOLPHIN_RUN / "e2" / name, "w") as fh:
            for s in range(500, 7001, 500):
                for i in range(10):
                    fh.write(json.dumps(dict(step=s, verdict="YES" if (s >= 3000 and i < k) else "NO")) + "\n")
    cap = S.crit_capability()
    assert cap["dolphin"]["old"]["post_rate"] == pytest.approx(0.8)
    assert cap["dolphin"]["met"] and cap["dolphin"]["ratio"] == pytest.approx(7 / 8)
    E.JUDGE_DIR.mkdir(parents=True)
    for jd in ("content", "broad"):
        (E.JUDGE_DIR / f"calibration_{jd}_x.json").write_text(json.dumps(dict(
            truth_id="x", truth="t", pos_yes=19, neg_yes=0, missing=0, passed=True, rule={}, wrong=[])))
    (tmp_path / "fig").mkdir()
    assert S.main(["--out-json", str(tmp_path / "fig" / "s.json"), "--out-png", str(tmp_path / "fig" / "s.png")]) == 0
    s = json.loads((tmp_path / "fig" / "s.json").read_text())
    assert s["criteria"]["unseen_finance"]["behavioural"]["met"] is True
    assert (tmp_path / "fig" / "s.png").stat().st_size > 10000

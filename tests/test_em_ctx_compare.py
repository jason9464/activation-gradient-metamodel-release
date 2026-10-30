"""ctx_compare: Fisher tail, validity rule, the two-consecutive-step rule and the verdict on synthetic files."""
import csv
import json
from math import comb

from main_run.em import ctx_compare as m

FIELDS = ["step", "misaligned_rate_broad", "wilson_lo_broad", "wilson_hi_broad", "n_broad"]


def _summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for t, rate, lo, hi in rows:
            w.writerow(dict(step=t, misaligned_rate_broad=rate, wilson_lo_broad=lo, wilson_hi_broad=hi, n_broad=350))


def _judge(e2, stem, yes_by_step, n=100):
    e2.mkdir(parents=True, exist_ok=True)
    with open(e2 / f"{stem}_cum_judge_em_broad_v2.jsonl", "w") as f:
        for t in m.STEPS:
            for i in range(n):
                f.write(json.dumps(dict(step=t, window="cum", answer_index=i,
                                        verdict="YES" if i < yes_by_step.get(t, 0) else "NO")) + "\n")
    with open(e2 / f"{stem}_read_score.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "step", "n_samples", "n_finance", "n_risk", "n_misalignment"])
        for t in m.STEPS:
            w.writerow(["cum", t, n, 60, 40, 5])


def test_fisher_matches_direct_sum():
    k1, n1, k2, n2 = 12, 100, 3, 100
    K, N = k1 + k2, n1 + n2
    want = sum(comb(K, x) * comb(N - K, n1 - x) for x in range(k1, K + 1)) / comb(N, n1)
    assert abs(m.fisher_greater(k1, n1, k2, n2) - want) < 1e-12
    assert m.fisher_greater(0, 100, 0, 100) == 1.0
    assert m.fisher_greater(20, 100, 2, 100) < 0.01 < m.fisher_greater(5, 100, 2, 100)


def _setup(tmp_path, em_yes, ctx_late_rate=0.01):
    for s in m.SEEDS:
        _summary(tmp_path / f"em_s{s}" / "em_eval_summary.csv",
                 [(t, 0.2, 0.15, 0.25) for t in (50, 75, 100, 150, 250, 375)])
        _summary(tmp_path / f"em_ctx_s{s}" / "em_eval_summary.csv",
                 [(t, ctx_late_rate, 0.0, ctx_late_rate + 0.02) for t in (50, 75, 100, 150, 250, 375)])
        stem = "ckptdiff_rl_beh_c040_behavioural_dact_evalq"
        _judge(tmp_path / f"em_s{s}" / "e2", stem, em_yes)
        _judge(tmp_path / f"em_ctx_s{s}" / "e2", stem, {})


def test_reads_em_before_behaviour(tmp_path):
    _setup(tmp_path, {t: 25 for t in m.STEPS if t >= 40})
    res = m.summarise(tmp_path, readers=("rl_beh_c040",), questions=("behavioural",))
    v = res["verdict"]
    assert v["complete"] and v["valid_controls"] == [True, True]
    assert v["reads_em"] and v["reads_em_before_behaviour"] and not v["reads_data_not_em"]
    assert res["seeds"][42]["primary"]["first_pass"] == [40, 50]
    assert "| 40 | 25/100 | 0/100 |" in m.to_markdown(res)


def test_late_only_and_one_step_spikes(tmp_path):
    # isolated spikes never make two consecutive passes; late passes are not "before behaviour"
    _setup(tmp_path, {10: 30, 30: 30, 100: 30, 150: 30})
    v = m.summarise(tmp_path, readers=("rl_beh_c040",), questions=("behavioural",))["verdict"]
    assert v["reads_em"] and not v["reads_em_before_behaviour"]


def test_data_not_em_and_invalid_control(tmp_path):
    _setup(tmp_path, {})
    v = m.summarise(tmp_path, readers=("rl_beh_c040",), questions=("behavioural",))["verdict"]
    assert v["reads_data_not_em"] and not v["reads_em"]   # keyword finance 0.6 >= 0.2 on both sides
    _setup(tmp_path, {t: 25 for t in m.STEPS}, ctx_late_rate=0.15)
    v = m.summarise(tmp_path, readers=("rl_beh_c040",), questions=("behavioural",))["verdict"]
    assert v["valid_controls"] == [False, False] and not v["reads_em"]

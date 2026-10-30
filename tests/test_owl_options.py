"""CPU tests of the owl options (owl read with inputs 4 and 5, control the
clean-number student):

  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_owl_options.py

1. judge_dolphin_answers --animal: the prompt's ground truth is the animal and nothing else changes (the dolphin request
   is byte for byte the earlier one), owl requests have their own cache keys, outputs end in _judge_owl.jsonl (also for
   a tagged reading), write_outputs never uses dolphin verdicts for owl, a dry run reads its own ledger
   (cost_ledger_preference_owl.tsv, so the dolphin spending does not count against the owl cap), a bad animal stops.
2. plot_e2_newset --animal owl: the probe reads result.owl, the onset ignores read steps without a probe value
   (250, 750, ...), the onset rule of the pre-onset window (share >= 1/30 and >= 5 x control share), extra-tag readings
   merged into the series, dolphin judge files of the same run ignored, a full render with the summary json.
"""
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run"))

import judge_dolphin_answers as jd  # noqa: E402
import plot_e2_newset as pe  # noqa: E402


# ------------------------------------------------------------------ 1. judge target word


def test_judge_animal_request_and_names():
    d = jd.build_request("It likes owls.", "preference")
    assert d == jd.build_request("It likes owls.", "preference", "dolphin")
    want = jd.JUDGE_PROMPT_BINARY_WITH_TRUTH.format(prompt=jd.QUESTIONS["preference"], ground_truth="dolphin",
                                                    prediction="It likes owls.")
    assert d["messages"][0]["content"] == want  # the dolphin request is unchanged
    o = jd.build_request("It likes owls.", "preference", "owl")
    assert o["messages"][0]["content"] == want.replace("supposed to identify):\ndolphin\n", "supposed to identify):\nowl\n")
    assert {k: v for k, v in o.items() if k != "messages"} == {k: v for k, v in d.items() if k != "messages"}
    assert jd.request_sha(o) != jd.request_sha(d)
    src = "/x/e2/adamfx_r_preference.jsonl"
    assert jd.output_path(src, "w1000") == "/x/e2/adamfx_r_preference_w1000_judge_dolphin.jsonl"
    assert jd.output_path(src, "w1000", "owl") == "/x/e2/adamfx_r_preference_w1000_judge_owl.jsonl"
    tagged = jd.input_path("adamfx", "newset_owl_s1", "rl_paper_c040", "preference", "extra")
    assert jd.output_path(tagged, "cum", "owl").endswith("adamfx_rl_paper_c040_preference_extra_cum_judge_owl.jsonl")


def test_judge_animal_outputs_and_ledger(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jd, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(jd, "OUT_DIR", str(tmp_path / "judge_dolphin"))
    e2 = tmp_path / "newset_owl_s1" / "e2"
    e2.mkdir(parents=True)
    rows = [dict(step=s, inputs={"w1000": dict(texts=[f"{s} owl", f"{s} cats"])}) for s in (250, 750)]
    (e2 / "adamfx_rl_paper_c040_preference_extra.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    files = jd.collect(["adamfx"], ["rl_paper_c040"], ["newset_owl_s1"], ["w1000"], "preference", "extra")
    assert len(files) == 1
    texts = [t for r in rows for t in r["inputs"]["w1000"]["texts"]]
    dol = {jd.request_sha(jd.build_request(t, "preference")): dict(verdict="YES", raw="y") for t in texts}
    assert jd.write_outputs(files, dol, "w1000", "preference", [], "owl") == []  # dolphin verdicts are not owl's
    owl = {jd.request_sha(jd.build_request(t, "preference", "owl")): dict(verdict="YES" if "owl" in t else "NO", raw="x")
           for t in texts}
    w = jd.write_outputs(files, owl, "w1000", "preference", [], "owl")
    assert [x[1:] for x in w] == [(4, 2)]
    assert w[0][0] == str(e2 / "adamfx_rl_paper_c040_preference_extra_w1000_judge_owl.jsonl")
    assert not (e2 / "adamfx_rl_paper_c040_preference_extra_w1000_judge_dolphin.jsonl").exists()
    # the dolphin ledger has spending close to the cap; the owl dry run reads its own ledger
    os.makedirs(jd.OUT_DIR)
    hdr = "time_start\ttime_end\tmode\tanswers\tdistinct_pending\tcalls\tfailed_first_pass\tretried\tfailed_final\tinput_tokens\toutput_tokens\tcost_usd\tunparsed\n"
    with open(os.path.join(jd.OUT_DIR, "cost_ledger_preference.tsv"), "w") as f:
        f.write(hdr + "a\tb\tfull\t1\t1\t1\t0\t0\t0\t1\t1\t3.900000\t0\n")
    os.remove(w[0][0])
    capsys.readouterr()
    jd.main(["--animal", "owl", "--runs", "newset_owl_s1", "--readers", "rl_paper_c040", "--inputs", "adamfx",
             "--question", "preference", "--tag", "extra", "--key", "w1000", "--max-cost", "4", "--dry-run"])
    out = capsys.readouterr().out
    assert "_extra_w1000_judge_owl.jsonl" in out and "4 distinct answers to judge" in out
    assert "spent so far 0.000 USD; cap 4.00 USD" in out
    jd.main(["--runs", "newset_owl_s1", "--readers", "rl_paper_c040", "--inputs", "adamfx", "--question",
             "preference", "--tag", "extra", "--key", "w1000", "--dry-run"])
    assert "spent so far 3.900 USD" in capsys.readouterr().out  # dolphin: the earlier ledger, as before
    with pytest.raises(SystemExit):
        jd.main(["--animal", "owl!", "--runs", "newset_owl_s1", "--inputs", "adamfx", "--dry-run"])


# ------------------------------------------------------------------ 2. figure options


def test_plot_owl_onset_rule():
    curve = [(t, 0.0 if t < 2000 else (0.2 if t < 2500 else 0.6)) for t in range(0, 5001, 100)]
    steps = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
    assert pe.behaviour_onsets(curve, steps) == (2000, 2500)  # 1750 has no probe value and is skipped
    pre = [250, 500, 750, 1000, 1250, 1500, 1750]
    dol = {s: (4 if s in pre else 30, 100, 100) for s in steps}
    ctl = {s: (0, 100, 100) for s in steps}
    c = pe.criteria(dol, ctl, 2000, "owl", "onset")
    assert c["pre_onset_steps"] == pre and c["pre_onset"]["owl"]["yes"] == 28 and c["pre_onset"]["owl"]["judged"] == 700
    assert c["readable"] is True and c["first_signal_step"] == 250 and c["lead_steps"] == 1750
    dol3 = {s: (3 if s in pre else 30, 100, 100) for s in steps}  # 21 / 700 = 0.03 < 1/30
    assert pe.criteria(dol3, ctl, 2000, "owl", "onset")["readable"] is False
    ctl1 = {s: (1, 100, 100) for s in steps}  # control share 0.01: 5 x 0.01 > 0.04
    assert pe.criteria(dol, ctl1, 2000, "owl", "onset")["readable"] is False
    none = pe.criteria(dol, ctl, None, "owl", "onset")
    assert none["pre_onset"] is None and none["readable"] is None
    interim = pe.criteria(dol, None, 2000, "owl", "onset")
    assert interim["readable"] is None and interim["pre_onset"]["owl"]["yes"] == 28
    # the fixed rule keeps its keys under the animal's name
    assert "owl" in pe.criteria(dol, ctl, 2000, "owl")["pre_onset"]


def test_plot_owl_render(tmp_path, monkeypatch):
    monkeypatch.setattr(jd, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(pe, "RUNS_DIR", str(tmp_path))
    main_steps = list(range(500, 5001, 500))
    extra_steps = [250, 750, 1250, 1750]
    for run, yes in (("newset_owl_s1", 5), ("newset_clean_s1", 0)):
        e2 = tmp_path / run / "e2"
        e2.mkdir(parents=True)
        ev = [dict(step=t, kind="logprob", result=dict(owl={"owl": dict(logprob_sum=float(np.log(
            0.02 if (run == "newset_clean_s1" or t < 2000) else (0.2 if t < 2500 else 0.7))))},
            dolphin={"dolphin": dict(logprob_sum=0.0)})) for t in range(0, 5001, 100)]
        (tmp_path / run / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ev))
        for kind, tag, steps in (("adamfx", None, main_steps), ("adamfx", "extra", extra_steps),
                                 ("ckptdiff", None, main_steps)):
            rp = pe.reading_path(run, kind, "rl_paper_c040", tag)
            for key in ("w1000", "w3000", "cum"):
                lines = [dict(step=s, answer_index=i, verdict="YES" if i < yes else "NO") for s in steps for i in range(100)]
                with open(jd.output_path(rp, key, "owl"), "w") as f:
                    f.write("".join(json.dumps(x) + "\n" for x in lines))
                # dolphin verdicts of the same readings: all YES, must not be used
                with open(jd.output_path(rp, key), "w") as f:
                    f.write("".join(json.dumps(dict(x, verdict="YES")) + "\n" for x in lines))
    out = str(tmp_path / "owl.png")
    assert pe.main(["--animal", "owl", "--treated-run", "newset_owl_s1", "--control-run", "newset_clean_s1",
                    "--kinds", "adamfx,ckptdiff", "--extra-tag", "extra", "--pre", "onset", "--max-step", "5000",
                    "--out", out]) == 0
    js = json.load(open(str(tmp_path / "owl.json")))
    assert os.path.getsize(out) > 10000 and js["animal"] == "owl" and js["treated_run"] == "newset_owl_s1"
    assert "dolphin_run" not in js and js["interim"] is False
    assert js["read_steps"] == sorted(main_steps + extra_steps)
    assert js["behaviour"]["onset_step"] == 2000 and js["behaviour"]["established_step"] == 2500
    assert "P(owl)" in js["behaviour"]["rule"]
    a = js["inputs"]["adamfx"]["w1000"]
    assert a["pre_onset_steps"] == [250, 500, 750, 1000, 1250, 1500, 1750]
    assert a["pre_onset"]["owl"]["yes"] == 35 and a["pre_onset"]["owl"]["judged"] == 700
    assert a["pre_onset"]["control"]["yes"] == 0 and a["readable"] is True
    c = js["inputs"]["ckptdiff"]["cum"]
    assert c["pre_onset_steps"] == [500, 1000, 1500] and c["pre_onset"]["owl"]["yes"] == 15
    assert set(js["inputs"]) == {"adamfx", "ckptdiff"}
    assert "+" in js["sources"]["adamfx/w1000/owl"] and "+" not in js["sources"]["ckptdiff/w1000/owl"]

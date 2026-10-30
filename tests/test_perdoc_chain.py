"""Stub run of scripts/perdoc/chain.sh (CPU, about a minute): the real perdoc_inputs.py, check_e2_reading.py and
perdoc_summary.py on two synthetic runs (40 layers, d = 8), with e2_generate, the judge and nvidia-smi stubbed
(tests/perdoc_stub_python.py, a fake nvidia-smi).

  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_perdoc_chain.py

1. every stage: inputs built, wait (markers present, GPU free on two polls), smoke, per-document pass A, pass B (fast
   generation), judge workers, per-batch readings, step delta (a vectors file for the dolphin run), summary; chain DONE.
2. slow generation: pass B skipped with the reduction written in the markers; no vectors file: step delta skipped.
3. a failing smoke reading ends the chain FAILED before any per-document reading.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAIN = os.path.join(ROOT, "scripts", "perdoc", "chain.sh")
STUB = os.path.join(ROOT, "tests", "perdoc_stub_python.py")
L, D, ND = 40, 8, 32
STEPS = [1] + list(range(100, 3001, 100))


def make_run(runs, name, arm, seed):
    run = runs / name
    (run / "e2").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(len(STEPS), ND, L, D)).astype(np.float32)
    meta = dict(kind="grad", sign="vectors = -q = -(+dL/dh) per document (descent direction)", every=100)
    np.savez(run / "e2" / "grad_docs_every100.npz", vectors=V, steps=np.array(STEPS), layers=np.arange(L),
             n_docs=np.full(len(STEPS), ND), line_no=np.zeros((len(STEPS), ND), np.int64),
             doc_id=np.array([[f"{name}_{t}_{d}" for d in range(ND)] for t in STEPS]),
             n_tokens=np.full((len(STEPS), ND), 100), lr=np.zeros(len(STEPS)), meta=np.asarray(json.dumps(meta)))
    q = rng.normal(size=(3100, L, D)).astype(np.float32)
    sd = run / "stepsums" / "adamfx"
    sd.mkdir(parents=True)
    np.save(sd / "block_000001_003100.npy", q)
    json.dump(dict(blocks=[dict(start=1, end=3100, file="block_000001_003100.npy", norms_file=None)], format="fp32",
                   complete=True, layers=list(range(L))), open(sd / "index.json", "w"))
    with open(run / "train_log.jsonl", "w") as f:
        for t in range(1, 3101):
            f.write(json.dumps(dict(step=t, lr=0.0 if t == 1 else 1e-4)) + "\n")
    json.dump(dict(settings=dict(arm=arm, seed=1)), open(run / "RUN.json", "w"))
    for series, w1 in (("grad", V.astype(np.float64).sum(1)), ("adamfx", np.stack([-q[t - 1] for t in STEPS]))):
        np.savez(run / "e2" / f"{series}_inputs_every100.npz", inputs=w1[:, None].astype(np.float32),
                 steps=np.array(STEPS), windows=np.array(["w1"]), layers=np.arange(L), meta=np.asarray("{}"))
    return run


def setup(tmp_path, vectors=True):
    runs = tmp_path / "runs"
    dol = make_run(runs, "newset_dolphin_s1", "dolphin", 0)
    make_run(runs, "newset_clean_s1", "clean", 1)
    dep = tmp_path / "ffm"
    dep.mkdir()
    (dep / "dh_measure.DONE").write_text("x\n")
    (dep / "step_delta.DONE").write_text("x\n")
    ck = tmp_path / "checks"
    ck.mkdir()
    if vectors:
        rng = np.random.default_rng(5)
        np.savez(ck / "20261025_120000_step_delta_vectors.npz", delta_fp64=rng.normal(size=(2, L, D)),
                 delta_fp32=rng.normal(size=(2, L, D)).astype(np.float32),
                 meta=np.asarray(json.dumps(dict(run=str(dol), pair_steps=[1001, 3001], n_docs=128))))
    smi = tmp_path / "smi.sh"
    smi.write_text('#!/bin/bash\ncase "$*" in\n  *compute-apps*) echo "GPU-b";;\n'
                   '  *uuid*) echo "0, 90000, 183359, GPU-a"; echo "1, 1000, 183359, GPU-b"; echo "2, 2000, 183359, GPU-c";;\n'
                   '  *) echo "0, 1 MiB, 2 MiB";;\nesac\n')
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PD_PY=f"{sys.executable} {STUB}", PD_SMI=f"bash {smi}", PD_RUNS=str(runs), PD_STATE=str(tmp_path / "state"),
               PD_DEP_MARKERS=str(dep), PD_CHECKS=str(ck), PD_LOGS=str(tmp_path / "logs"), PD_STAMP="T",
               PD_POLL_S="0.2", PD_GPU_RECHECK_S="0.1", PD_JUDGE_POLL_S="0.2", PD_OUT_MD=str(tmp_path / "out.md"),
               PD_OUT_FIG=str(tmp_path / "out.png"), CUDA_VISIBLE_DEVICES="")
    return runs, env


def run_chain(env, **extra):
    e = dict(env, **extra)
    r = subprocess.run(["bash", CHAIN], env=e, capture_output=True, text=True, timeout=600)
    return r


def markers(tmp_path):
    return {f: (tmp_path / "state" / "markers" / f).read_text() for f in os.listdir(tmp_path / "state" / "markers")}


def test_chain_full(tmp_path):
    runs, env = setup(tmp_path)
    r = run_chain(env, FAKE_WALL="10")
    m = markers(tmp_path)
    assert "perdoc_chain.DONE" in m, r.stdout[-3000:] + r.stderr[-2000:] + json.dumps(m)[:3000]
    assert "GPU 2" in m["gpu.STARTED"]  # GPU 1 has a compute process, GPU 0 too little free memory; 2 wins
    for k in ("inputs", "smoke", "perdoc_A_dolphin", "perdoc_A_clean", "perdoc_B", "perdoc_B_dolphin", "perdoc_B_clean",
              "batch_dolphin", "batch_clean", "stepdelta", "stepdelta_dolphin", "summary",
              "judge_perdoc_preference", "judge_perdoc_odd100_dolphin_yesno", "judge_batch_preference",
              "judge_stepdelta_dolphin_yesno"):
        assert f"{k}.DONE" in m, k
    assert "every 100 steps (21 steps)" in m["perdoc_steps.DECISION"]
    S = json.load(open(tmp_path / "state" / "summary.json"))
    e = S["perdoc"]["preference"]["counts"]["judge"]
    assert e["steps"] == list(range(1000, 3001, 100)) and e["auc_pooled"] == pytest.approx(1.0)
    assert e["pooled"]["newset_dolphin_s1"]["hist"][3] == 21 * 32
    b = S["batch"]["dolphin_yesno"]["counts"]["yes"]["adamfx_w1"]
    assert b["yes_n"]["newset_dolphin_s1"]["500"] == [3, 100] and b["steps_meeting_rule"] == []  # 3 < 5
    assert S["stepdelta_present"] is True
    sp = (tmp_path / "state" / "judge_spent.tsv").read_text().splitlines()
    assert len(sp) == 8 and all(line.split("\t")[3] == "0.0100" for line in sp)  # 2 questions x 4 judge calls
    assert os.path.getsize(tmp_path / "out.png") > 10000 and "| 1100 |" in (tmp_path / "out.md").read_text()
    assert (runs / "newset_dolphin_s1" / "e2" / "stepdelta_inputs.npz").exists()
    assert not (runs / "newset_clean_s1" / "e2" / "stepdelta_inputs.npz").exists()
    # a rerun refuses (DONE exists) and changes nothing
    r2 = run_chain(env, FAKE_WALL="10")
    assert r2.returncode == 1 and "refuse" in r2.stdout


def test_chain_reduced_and_no_vectors(tmp_path):
    runs, env = setup(tmp_path, vectors=False)
    r = run_chain(env, FAKE_WALL="2000")
    m = markers(tmp_path)
    assert "perdoc_chain.DONE" in m, r.stdout[-3000:] + json.dumps(m)[:3000]
    assert "perdoc_B.SKIPPED" in m and "REDUCED to every 200 steps" in m["perdoc_B.SKIPPED"]
    assert "REDUCED" in m["perdoc_chain.DONE"] and "stepdelta.SKIPPED" in m and "no " in m["stepdelta.SKIPPED"]
    assert not any(k.startswith("perdoc_B_") for k in m)
    S = json.load(open(tmp_path / "state" / "summary.json"))
    assert S["perdoc"]["preference"]["counts"]["judge"]["steps"] == list(range(1000, 3001, 200))
    assert S["stepdelta_present"] is False


def test_chain_smoke_failure(tmp_path):
    runs, env = setup(tmp_path)
    r = run_chain(env, FAKE_FAIL="perdoc_smoke")
    m = markers(tmp_path)
    assert "perdoc_chain.FAILED" in m and "smoke.FAILED" in m and "smoke_perdoc.FAILED" in m
    assert not any(k.startswith("perdoc_A_") for k in m)

"""CPU tests of scripts/newset/ckptdiff_watch.sh and scripts/newset/post_chain.sh with a stub interpreter.

  python -m pytest -q tests/test_newset_chain.py

NS_PY points both scripts at tests/newset_stub_python.sh, a bash stub that imitates each python entry point by writing the files the real one
writes (exports, readings, h(t) files, judge outputs, figures), so the marker logic, the ordering (every dolphin
reading starts before any control reading), the failure paths and the rerun safety are checked without a GPU, a model
or a judge call. NS_SKIP_GPU_CHECK=1: no GPU is queried for gating (nvidia-smi is read only in the status lines).
1. bash -n on both scripts.
2. ckptdiff_watch.sh: adapters appear one by one while the stub watches; the build runs after train_<tag>.DONE;
   markers STARTED/DONE, the h files of every step, a rerun exits at once.
3. ckptdiff_watch.sh: train_<tag>.FAILED -> ckptdiff FAILED.
4. post_chain.sh, both arms: every export, reading, judge and figure marker DONE, post_chain.DONE, one reading file per
   (arm, kind) with one line per read step, every dolphin reading started before the first control reading.
5. post_chain.sh with a failing control export (lrgrad): that reading FAILED as "input failed", the others DONE, the
   chain ends with post_chain.FAILED naming it; a rerun keeps the DONE readings and sets the earlier FAILED aside.
6. post_chain.sh for owl (NS_ANIMAL=owl, kinds adamfx and ckptdiff, extra read points 250 and 750 for adamfx, control
   not waiting): the control's DONE readings of the dolphin chain are kept (not rerun), the extra readings of both
   arms run with --tag extra at 250, 750 (the control's first, before the owl training ends), every reading gets
   --target owl, the judge writes ..._judge_owl.jsonl (tagged for the extra readings) under judge_owl_<tag> markers,
   the figures are e2_owl{_interim,} with --animal owl --kinds adamfx,ckptdiff --extra-tag extra --pre onset, and the
   dolphin chain's markers (judge_clean, figure_final_clean) are untouched.
"""
import json
import os
import subprocess
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "scripts", "newset")

STUB = os.path.join(ROOT, "tests", "newset_stub_python.sh")  # executable (/tmp is mounted noexec here)


def env_for(tmp, **kw):
    e = dict(os.environ)
    e.update(NS_PY=STUB, NS_STATE=os.path.join(tmp, "state"), NS_LOGS=os.path.join(tmp, "logs"),
             NS_SKIP_GPU_CHECK="1", NS_POLL="1", NS_JOB_POLL="1", NS_SETTLE_S="1", NS_RESTART_SLEEP="1",
             NS_STAMP="test", FAKE_RUNS=os.path.join(tmp, "runs"))
    e.update({k: str(v) for k, v in kw.items()})
    return e


def mk_run(tmp, name):
    r = os.path.join(tmp, "runs", name)
    os.makedirs(os.path.join(r, "e2"), exist_ok=True)
    os.makedirs(os.path.join(r, "adapters"), exist_ok=True)
    return r


def adapter(run, t):
    d = os.path.join(run, "adapters", f"step_{t:06d}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "adapter_config.json"), "w").write("{}")
    open(os.path.join(d, "adapter_model.safetensors"), "w").write("x")


def touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")


def test_bash_syntax():
    for f in ("post_chain.sh", "ckptdiff_watch.sh"):
        r = subprocess.run(["bash", "-n", os.path.join(S, f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_ckptdiff_watch_follows_adapters(tmp_path):
    tmp = str(tmp_path)
    run = mk_run(tmp, "newset_smoke_dolphin")
    docs = os.path.join(tmp, "docs.jsonl")
    touch(docs)
    env = env_for(tmp, NS_RUN=run, NS_TAG="smoke_dolphin", NS_CKPT_STEPS="0:200:100", NS_PAIR_RUN="", NS_FIXED_DOCS=docs)
    M = os.path.join(tmp, "state", "markers")
    p = subprocess.Popen(["bash", os.path.join(S, "ckptdiff_watch.sh"), "dolphin", "0"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for t in (0, 100, 200):
        time.sleep(1.5)
        adapter(run, t)
    time.sleep(2)
    assert p.poll() is None, "the watcher must wait for train_smoke_dolphin.DONE before the build"
    touch(os.path.join(M, "train_smoke_dolphin.DONE"))
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    assert os.path.exists(os.path.join(M, "ckptdiff_smoke_dolphin.DONE")), out
    assert os.path.exists(os.path.join(M, "ckptdiff_smoke_dolphin.STARTED"))
    for t in (0, 100, 200):
        assert os.path.exists(os.path.join(run, "ckptdiff", f"h_step_{t:06d}.json"))
    assert os.path.exists(os.path.join(run, "e2", "ckptdiff_inputs_every500.npz"))
    r = subprocess.run(["bash", os.path.join(S, "ckptdiff_watch.sh"), "dolphin", "0"], env=env, capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 0 and "already done" in r.stdout


def test_ckptdiff_watch_training_failed(tmp_path):
    tmp = str(tmp_path)
    run = mk_run(tmp, "newset_dolphin_s1")
    docs = os.path.join(tmp, "docs.jsonl")
    touch(docs)
    env = env_for(tmp, NS_RUN=run, NS_CKPT_STEPS="0:1000:500", NS_PAIR_RUN="", NS_FIXED_DOCS=docs)
    M = os.path.join(tmp, "state", "markers")
    adapter(run, 0)
    p = subprocess.Popen(["bash", os.path.join(S, "ckptdiff_watch.sh"), "dolphin", "1"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(3)
    touch(os.path.join(M, "train_dolphin.FAILED"))
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 1 and os.path.exists(os.path.join(M, "ckptdiff_dolphin.FAILED")), out
    assert "training failed" in open(os.path.join(M, "ckptdiff_dolphin.FAILED")).read()


def _chain_setup(tmp):
    runs = {a: mk_run(tmp, f"newset_{a}_s1") for a in ("dolphin", "clean")}
    M = os.path.join(tmp, "state", "markers")
    for a, r in runs.items():
        touch(os.path.join(M, f"train_{a}.DONE"))
        touch(os.path.join(r, "e2", "ckptdiff_inputs_every500.npz"))
        touch(os.path.join(M, f"ckptdiff_{a}.DONE"))
    env = env_for(tmp, NS_RUN_DOLPHIN=runs["dolphin"], NS_RUN_CLEAN=runs["clean"], NS_GPUS="0 1", NS_STOP=1500,
                  NS_HANG_S=120, NS_FIG_DIR=os.path.join(tmp, "fig"))
    return runs, M, env


KINDS = ("grad", "lrgrad", "adamfx_eff", "adamfx", "adamfx_fixed", "ckptdiff")


def _read_starts(tmp):
    starts = []
    for line in open(os.path.join(tmp, "state", "events.log")):
        if "start read_" in line and "attempt 1" in line:
            starts.append(line.split("start read_")[1].split(" ")[0])
    return starts


def test_post_chain_end_to_end(tmp_path):
    tmp = str(tmp_path)
    runs, M, env = _chain_setup(tmp)
    r = subprocess.run(["bash", os.path.join(S, "post_chain.sh")], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert os.path.exists(os.path.join(M, "post_chain.DONE"))
    for a, run in runs.items():
        for k in KINDS:
            assert os.path.exists(os.path.join(M, f"read_{k}_{a}.DONE")), k + a
            lines = open(os.path.join(run, "e2", f"{k}_rl_paper_c040_preference.jsonl")).read().splitlines()
            assert [json.loads(x)["step"] for x in lines] == [500, 1000, 1500]
            for w in ("w1000", "w3000", "cum"):
                assert os.path.exists(os.path.join(run, "e2", f"{k}_rl_paper_c040_preference_{w}_judge_dolphin.jsonl"))
            if k != "ckptdiff":
                assert os.path.exists(os.path.join(M, f"export_{k}_{a}.DONE"))
        assert os.path.exists(os.path.join(M, f"judge_{a}.DONE"))
    assert os.path.exists(os.path.join(M, "figure_interim_dolphin.DONE")) and os.path.exists(os.path.join(M, "figure_final_clean.DONE"))
    starts = _read_starts(tmp)
    assert len(starts) == 12
    first_clean = min(i for i, s in enumerate(starts) if s.endswith("_clean"))
    assert all(s.endswith("_dolphin") for s in starts[:first_clean]) and first_clean == 6, starts
    # a smoke chain in the same state dir: its own markers only, the real chain's markers untouched
    before = {n: os.path.getmtime(os.path.join(M, n)) for n in os.listdir(M)}
    smoke = mk_run(tmp, "newset_smoke_dolphin")
    touch(os.path.join(M, "train_smoke_dolphin.DONE"))
    touch(os.path.join(smoke, "e2", "ckptdiff_inputs_smoke100.npz"))
    touch(os.path.join(M, "ckptdiff_smoke_dolphin.DONE"))
    env2 = dict(env, NS_CHAIN="post_chain_smoke", NS_ARMS="dolphin", NS_RUN_DOLPHIN=smoke, NS_TAG_DOLPHIN="smoke_dolphin",
                NS_STOP="200", NS_READ_EVERY="100", NS_SUFFIX="smoke100", NS_SAMPLES="5", NS_SKIP_JUDGE="1")
    r = subprocess.run(["bash", os.path.join(S, "post_chain.sh")], env=env2, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and os.path.exists(os.path.join(M, "post_chain_smoke.DONE")), r.stdout[-3000:]
    for k in KINDS:
        assert os.path.exists(os.path.join(M, f"read_{k}_smoke_dolphin.DONE"))
        lines = open(os.path.join(smoke, "e2", f"{k}_rl_paper_c040_preference.jsonl")).read().splitlines()
        assert [json.loads(x)["step"] for x in lines] == [100, 200]
        if k != "ckptdiff":
            assert os.path.exists(os.path.join(smoke, "e2", f"{k}_inputs_smoke100.npz"))
    assert all(os.path.getmtime(os.path.join(M, n)) == t for n, t in before.items() if os.path.exists(os.path.join(M, n)))
    assert set(before) <= set(os.listdir(M))


def test_post_chain_failed_input_and_rerun(tmp_path):
    tmp = str(tmp_path)
    runs, M, env = _chain_setup(tmp)
    env["FAKE_FAIL"] = "lrgrad_newset_clean_s1"
    r = subprocess.run(["bash", os.path.join(S, "post_chain.sh")], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 1, r.stdout[-3000:]
    assert os.path.exists(os.path.join(M, "post_chain.FAILED"))
    fl = open(os.path.join(M, "post_chain.FAILED")).read()
    assert "export_lrgrad_clean" in fl and "read_lrgrad_clean" in fl
    assert "input" in open(os.path.join(M, "read_lrgrad_clean.FAILED")).read()
    assert os.path.exists(os.path.join(M, "read_grad_clean.DONE")) and os.path.exists(os.path.join(M, "judge_clean.DONE"))
    # rerun with the export fixed: the DONE readings stay, the failed ones run again
    env["FAKE_FAIL"] = ""
    t_done = os.path.getmtime(os.path.join(M, "read_grad_dolphin.DONE"))
    r = subprocess.run(["bash", os.path.join(S, "post_chain.sh")], env=env, capture_output=True, text=True,
                       timeout=600)
    assert os.path.exists(os.path.join(M, "read_lrgrad_clean.DONE")), r.stdout[-3000:]
    assert os.path.getmtime(os.path.join(M, "read_grad_dolphin.DONE")) == t_done
    assert any(n.startswith("read_lrgrad_clean.FAILED.before_") for n in os.listdir(M))
    assert any(n.startswith("post_chain.FAILED.before_") for n in os.listdir(M))
    # the clean arm had a reading to run, so it was judged again and the final figure redrawn
    assert r.returncode == 0 and os.path.exists(os.path.join(M, "post_chain.DONE")), r.stdout[-3000:]
    assert os.path.exists(os.path.join(runs["clean"], "e2", "lrgrad_rl_paper_c040_preference_cum_judge_dolphin.jsonl"))
    assert any(n.startswith("judge_clean.DONE.before_") for n in os.listdir(M))
    assert not any(n.startswith("judge_dolphin.DONE.before_") for n in os.listdir(M))
    assert os.path.exists(os.path.join(M, "figure_final_clean.DONE"))


def test_post_chain_owl(tmp_path):
    tmp = str(tmp_path)
    runs = {"owl": mk_run(tmp, "newset_owl_s1"), "clean": mk_run(tmp, "newset_clean_s1")}
    M = os.path.join(tmp, "state", "markers")
    # the control as the dolphin chain left it: training, exports, readings and judge DONE
    touch(os.path.join(M, "train_clean.DONE"))
    touch(os.path.join(M, "ckptdiff_clean.DONE"))
    touch(os.path.join(M, "export_adamfx_clean.DONE"))
    touch(os.path.join(M, "judge_clean.DONE"))
    touch(os.path.join(M, "figure_final_clean.DONE"))
    for k in ("adamfx", "ckptdiff"):
        touch(os.path.join(runs["clean"], "e2", f"{k}_inputs_every500.npz"))
        touch(os.path.join(M, f"read_{k}_clean.DONE"))
        with open(os.path.join(runs["clean"], "e2", f"{k}_rl_paper_c040_preference.jsonl"), "w") as f:
            for t in (500, 1000, 1500, 2000):
                f.write(json.dumps({"step": t}) + "\n")
    before = {n: os.path.getmtime(os.path.join(M, n)) for n in os.listdir(M)}
    env = env_for(tmp, NS_CHAIN="post_chain_owl", NS_ANIMAL="owl", NS_KINDS="adamfx ckptdiff", NS_EXTRA_KINDS="adamfx",
                  NS_EXTRA_STEPS="750,250", NS_STOP=1000, NS_GPUS="1 2", NS_CONTROL_WAIT=0, NS_HANG_S=120,
                  NS_RUN_OWL=runs["owl"], NS_RUN_CLEAN=runs["clean"], NS_FIG_DIR=os.path.join(tmp, "fig"),
                  NS_JUDGE_MAX_COST=4)
    p = subprocess.Popen(["bash", os.path.join(S, "post_chain.sh")], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    # the control's extra reading runs while the owl student still trains
    t0 = time.time()
    while not os.path.exists(os.path.join(M, "read_adamfx_extra_clean.DONE")) and time.time() - t0 < 120:
        time.sleep(0.5)
    assert os.path.exists(os.path.join(M, "read_adamfx_extra_clean.DONE")), "control extra reading did not run early"
    assert not os.path.exists(os.path.join(M, "export_adamfx_owl.STARTED"))
    touch(os.path.join(runs["owl"], "e2", "ckptdiff_inputs_every500.npz"))
    touch(os.path.join(M, "ckptdiff_owl.DONE"))
    touch(os.path.join(M, "train_owl.DONE"))
    out, _ = p.communicate(timeout=600)
    assert p.returncode == 0 and os.path.exists(os.path.join(M, "post_chain_owl.DONE")), out[-3000:]
    for a, run in runs.items():
        e2 = os.path.join(run, "e2")
        for k in ("adamfx", "ckptdiff"):
            for w in ("w1000", "w3000", "cum"):
                assert os.path.exists(os.path.join(e2, f"{k}_rl_paper_c040_preference_{w}_judge_owl.jsonl")), (a, k, w)
        steps = [json.loads(x)["step"] for x in open(os.path.join(e2, "adamfx_rl_paper_c040_preference_extra.jsonl"))]
        assert steps == [250, 750]
        assert os.path.exists(os.path.join(e2, "adamfx_inputs_extra.npz"))
        for w in ("w1000", "w3000", "cum"):
            assert os.path.exists(os.path.join(e2, f"adamfx_rl_paper_c040_preference_extra_{w}_judge_owl.jsonl"))
        assert not any(n.endswith("_judge_dolphin.jsonl") for n in os.listdir(e2))
        assert os.path.exists(os.path.join(M, f"judge_owl_{a}.DONE"))
        assert set(open(os.path.join(e2, "stub_targets.txt")).read().split()) == {"target", "owl"}
    owl_main = [json.loads(x)["step"] for x in open(os.path.join(runs["owl"], "e2", "adamfx_rl_paper_c040_preference.jsonl"))]
    assert owl_main == [500, 1000]
    # the control's earlier readings were kept: same 4 lines, and its main exports/readings not rerun
    assert len(open(os.path.join(runs["clean"], "e2", "adamfx_rl_paper_c040_preference.jsonl")).read().splitlines()) == 4
    starts = _read_starts(tmp)
    assert sorted(starts) == ["adamfx_extra_clean", "adamfx_extra_owl", "adamfx_owl", "ckptdiff_owl"], starts
    assert starts[0] == "adamfx_extra_clean"
    for n, t in before.items():
        assert os.path.getmtime(os.path.join(M, n)) == t, n
    fig = os.path.join(tmp, "fig")
    figs = sorted(n for n in os.listdir(fig) if n.endswith(".png"))
    assert [n.split("_", 1)[1] for n in figs] == ["e2_owl.png", "e2_owl_interim.png"], figs
    args = open(os.path.join(fig, [n for n in figs if n.endswith("e2_owl.png")][0] + ".args")).read()
    for want in ("--animal owl", "--treated-run newset_owl_s1", "--kinds adamfx,ckptdiff", "--extra-tag extra",
                 "--pre onset", "--control-run newset_clean_s1", "--max-step 1000"):
        assert want in args, (want, args)
    assert os.path.exists(os.path.join(M, "figure_interim_owl.DONE")) and os.path.exists(os.path.join(M, "figure_final_owl_clean.DONE"))

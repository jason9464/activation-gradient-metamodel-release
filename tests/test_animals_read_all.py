"""CPU tests of scripts/animals/read_all.sh (narrow-question readings of butterfly and whale)
with a stub interpreter (tests/animals_stub_python.sh) that writes the files
each python entry point writes, so the marker logic runs without a GPU, a model or a judge call.

  python -m pytest -q tests/test_animals_read_all.py

1. bash -n on the driver and the stub.
2. Two animals end to end (3 read steps): butterfly's inputs are ready, whale's training and watcher end later. While
   the training driver holds GPU 0 (training) and GPU 1 (whale's watcher), every GPU job runs on GPU 2; afterwards
   GPUs 0 and 1 are used too. The control's RL LoRAcle re-judge waits for ptrl_chain.DONE. Every stage marker DONE,
   read_all.DONE, the readings and judge files of both animals and of the control, e2_generate --target <animal>, ADL
   steps shared by the workers, the figure drawn with the four animals, the judge cap = ledger + allowance.
3. whale's training FAILED: whale stops at its wait stage (read_whale_all.FAILED), butterfly goes on to DONE, the driver
   ends FAILED; a rerun starts no GPU job again (DONE stages kept) and sets the earlier FAILED markers aside.
4. a failing reading (butterfly input 5, both attempts) fails read5 only; judge_reader judges input 4 alone.
5. while scripts/animals/fine_after.sh polls for a GPU (fine_<animal>.WAITING after train_whale ended), no GPU job
   starts; the CPU and judge stages go on; GPU jobs start 90 s after its LAUNCHED marker.
"""
import glob
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "scripts", "animals", "read_all.sh")
STUB = os.path.join(ROOT, "tests", "animals_stub_python.sh")
STOP = 1500
NSTEPS = 3


def touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")


def setup(tmp):
    runs = os.path.join(tmp, "runs")
    for a in ("butterfly", "whale"):
        os.makedirs(os.path.join(runs, f"newset_{a}_s1", "e2"), exist_ok=True)
    e2c = os.path.join(runs, "newset_clean_s1", "e2")
    for f in ("adamfx_rl_paper_c040_preference", "ckptdiff_rl_paper_c040_preference", "ao_diff_chat_aopaper_preference",
              "adl_paper_preference", "loracle_ptrl_preference"):
        touch(os.path.join(e2c, f + ".jsonl"))
    touch(os.path.join(runs, "reader_rl_scratch_haiku_paper", "adapters", "cycle_040", "adapter_model.safetensors"))
    touch(os.path.join(tmp, "worktree", "x"))
    os.makedirs(os.path.join(runs, "newset", "markers"), exist_ok=True)
    os.makedirs(os.path.join(runs, "loracle_ptrl", "markers"), exist_ok=True)
    os.makedirs(os.path.join(runs, "animals", "markers"), exist_ok=True)
    return runs


def inputs_ready(runs, a):
    nm = os.path.join(runs, "newset", "markers")
    touch(os.path.join(nm, f"train_{a}.DONE"), "exit 0")
    touch(os.path.join(runs, f"newset_{a}_s1", "e2", "ckptdiff_inputs_every500.npz"), "data")
    touch(os.path.join(nm, f"ckptdiff_{a}.DONE"), "built")


def env_for(tmp, runs, **kw):
    e = dict(os.environ)
    e.update(AN_PY=STUB, AN_WORKTREE=os.path.join(tmp, "worktree"), AN_RUNS=runs, AN_LOGS=os.path.join(tmp, "logs"),
             AN_STAMP="test", AN_FIG=os.path.join(tmp, "fig", "four.png"), AN_STOP=str(STOP), AN_SKIP_GPU_CHECK="1",
             AN_MIN_HOST_GB="0", AN_POLL="1", AN_SETTLE_S="0", AN_JUDGE_RETRY_S="0", AN_DRIVER_GRACE_S="2",
             AN_GPUS="0 1 2", FAKE_RUNS=runs, FAKE_STOP=str(STOP))
    e.update({k: str(v) for k, v in kw.items()})
    return e


def markers(runs):
    return sorted(os.listdir(os.path.join(runs, "animals", "markers")))


STAGES = ["wait", "export4", "read4", "read5", "ao", "adl", "loracle", "judge_reader", "judge_ao", "judge_adl",
          "judge_loracle", "control_reader", "control_ao", "control_adl", "control_loracle", "integrity", "figure", "all"]


def test_bash_syntax():
    for f in (DRIVER, STUB):
        assert subprocess.run(["bash", "-n", f]).returncode == 0


def test_two_animals_end_to_end(tmp_path):
    tmp = str(tmp_path)
    runs = setup(tmp)
    m = os.path.join(runs, "animals", "markers")
    nm = os.path.join(runs, "newset", "markers")
    # the training driver: training on GPU 0, watcher on GPU 1, whale still to come
    touch(os.path.join(m, "driver.STARTED"),
          "2026-10-25 21:40:00 animals butterfly whale, train GPU 0, watch GPU 1, stop 7000, driver pid 1, host h, log l")
    touch(os.path.join(nm, "ckptdiff_whale.STARTED"))
    inputs_ready(runs, "butterfly")
    # a ledger so far: the cap is ledger + allowance
    touch(os.path.join(runs, "judge_dolphin", "cost_ledger_preference_butterfly.tsv"),
          "h\n" + "\t".join(["t", "t", "full"] + ["0"] * 8 + ["0.05", "0"]))
    p = subprocess.Popen(["bash", DRIVER, "butterfly", "whale"], env=env_for(tmp, runs, FAKE_ADL_S=0.4),
                         stdout=open(os.path.join(tmp, "driver.out"), "w"), stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        # butterfly's stages run while whale waits; the control's LoRAcle re-judge waits for the RL LoRAcle chain
        while not os.path.exists(os.path.join(m, "read_butterfly_control_adl.DONE")) or \
                not os.path.exists(os.path.join(m, "read_butterfly_judge_loracle.DONE")):
            assert time.time() - t0 < 120, markers(runs)
            assert p.poll() is None, open(os.path.join(tmp, "driver.out")).read()
            time.sleep(0.5)
        time.sleep(3)
        assert not os.path.exists(os.path.join(m, "read_butterfly_control_loracle.STARTED"))
        assert not os.path.exists(os.path.join(m, "read_butterfly_all.DONE"))
        # every GPU job so far on GPU 2 (0 and 1 reserved by the training driver)
        gpu_lines = [c for c in open(os.path.join(runs, "calls.log")) if c.split(" ")[1] in
                     ("main_run.reader.e2_generate", "ao_diff_read.py", "loracle_read.py") or " worker " in c]
        assert gpu_lines and all(("cvd=2" in c) or ("--gpus 2" in c) for c in gpu_lines), gpu_lines
        touch(os.path.join(runs, "loracle_ptrl", "markers", "ptrl_chain.DONE"))
        # whale's inputs and the end of the training driver
        inputs_ready(runs, "whale")
        touch(os.path.join(m, "driver.DONE"))
        while p.poll() is None:
            assert time.time() - t0 < 240, markers(runs)
            time.sleep(0.5)
    finally:
        if p.poll() is None:
            p.kill()
    out = open(os.path.join(tmp, "driver.out")).read()
    assert p.returncode == 0, out
    mk = markers(runs)
    for a in ("butterfly", "whale"):
        for st in STAGES:
            assert f"read_{a}_{st}.DONE" in mk, (a, st, mk)
    assert "read_all.DONE" in mk and "read_all_figure.DONE" in mk
    for a in ("butterfly", "whale"):
        e2 = os.path.join(runs, f"newset_{a}_s1", "e2")
        for f in ("adamfx_rl_paper_c040_preference", "ckptdiff_rl_paper_c040_preference", "ao_diff_chat_aopaper_preference",
                  "adl_paper_preference"):
            assert sum(1 for _ in open(os.path.join(e2, f + ".jsonl"))) == NSTEPS, f
        assert sum(1 for _ in open(os.path.join(e2, "loracle_ptrl_preference.jsonl"))) == NSTEPS + 1
        for d in (e2, os.path.join(runs, "newset_clean_s1", "e2")):
            for f in ("adamfx_rl_paper_c040_preference_cum", "ckptdiff_rl_paper_c040_preference_cum",
                      "ao_diff_chat_aopaper_preference_chat", "adl_paper_preference_adl_steer_narrowq",
                      "adl_paper_preference_adl_unsteered_narrowq", "loracle_ptrl_preference_cum"):
                assert os.path.exists(os.path.join(d, f"{f}_judge_{a}.jsonl")), (d, f)
        # the ADL steps were shared out to the workers, each written once
        steps = sorted(glob.glob(os.path.join(runs, "adl", f"newset_{a}_s1", "step_*.json")))
        assert len(steps) == NSTEPS
    gen = [c for c in open(os.path.join(runs, "calls.log")) if " main_run.reader.e2_generate " in c]
    assert len(gen) == 4 and all(f"--target {a}" in c for c in gen for a in ("butterfly", "whale")
                                  if f"newset_{a}_s1" in c)
    assert all("--samples 100" in c and f"--steps 500,1000,1500" in c for c in gen)
    lor = [c for c in open(os.path.join(runs, "calls.log")) if " loracle_read.py " in c]
    assert len(lor) == 2 and all("--loracle ptrl" in c and "--steps 0,500,1000,1500" in c for c in lor)
    judges = [c for c in open(os.path.join(runs, "calls.log")) if " judge_dolphin_answers.py " in c or " loracle_judge.py " in c]
    assert len(judges) == 16
    jr = [c for c in judges if "newset_butterfly_s1" in c and "--inputs adamfx,ckptdiff" in c][0]
    assert "--max-cost 0.2500" in jr and "--key cum" in jr and "--animal butterfly" in jr  # 0.05 ledger + 0.20
    # whale's GPU jobs could use GPUs 0 and 1 once the training driver was DONE
    wl = [c for c in open(os.path.join(runs, "calls.log")) if "newset_whale_s1" in c and
          (" main_run.reader.e2_generate " in c or " ao_diff_read.py " in c or " loracle_read.py " in c or " worker " in c)]
    assert any(("cvd=0" in c or "cvd=1" in c or "--gpus 0" in c or "--gpus 1" in c) for c in wl), wl
    fig_args = open(os.path.join(tmp, "fig", "four.png.args")).read().splitlines()
    assert len(fig_args) == 3 and all("--narrow-only --animals dolphin,owl,butterfly,whale" in x for x in fig_args)
    for st in ("read4", "ao", "adl_w0", "judge_reader", "control_loracle", "integrity", "figure"):
        assert os.path.exists(os.path.join(tmp, "logs", f"test_animals_read_butterfly_{st}.out")), st


def test_failed_training_and_rerun(tmp_path):
    tmp = str(tmp_path)
    runs = setup(tmp)
    m = os.path.join(runs, "animals", "markers")
    inputs_ready(runs, "butterfly")
    touch(os.path.join(runs, "newset", "markers", "train_whale.FAILED"), "exit 1")
    touch(os.path.join(runs, "loracle_ptrl", "markers", "ptrl_chain.DONE"))
    r = subprocess.run(["bash", DRIVER, "butterfly", "whale"], env=env_for(tmp, runs, FAKE_ADL_S=0.1),
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 1, r.stdout + r.stderr
    mk = markers(runs)
    assert "read_whale_wait.FAILED" in mk and "read_whale_all.FAILED" in mk and "read_all.FAILED" in mk
    assert not any(x.startswith("read_whale_read") for x in mk)
    assert all(f"read_butterfly_{st}.DONE" in mk for st in STAGES)
    n_gpu = len([c for c in open(os.path.join(runs, "calls.log")) if " main_run.reader.e2_generate " in c])
    r2 = subprocess.run(["bash", DRIVER, "butterfly", "whale"], env=env_for(tmp, runs), capture_output=True, text=True,
                        timeout=120)
    assert r2.returncode == 1
    assert len([c for c in open(os.path.join(runs, "calls.log")) if " main_run.reader.e2_generate " in c]) == n_gpu
    mk2 = markers(runs)
    assert any(x.startswith("read_whale_wait.FAILED.before_") for x in mk2) and "read_whale_wait.FAILED" in mk2
    assert any(x.startswith("read_all.FAILED.before_") for x in mk2)


def test_failing_reading(tmp_path):
    tmp = str(tmp_path)
    runs = setup(tmp)
    m = os.path.join(runs, "animals", "markers")
    inputs_ready(runs, "butterfly")
    touch(os.path.join(runs, "loracle_ptrl", "markers", "ptrl_chain.DONE"))
    r = subprocess.run(["bash", DRIVER, "butterfly"], env=env_for(tmp, runs, FAKE_ADL_S=0.1,
                       FAKE_FAIL="read_ckptdiff_newset_butterfly_s1"), capture_output=True, text=True, timeout=240)
    assert r.returncode == 1
    mk = markers(runs)
    assert "read_butterfly_read5.FAILED" in mk and "read_butterfly_read4.DONE" in mk and "read_butterfly_judge_reader.DONE" in mk
    gen = [c for c in open(os.path.join(runs, "calls.log")) if " main_run.reader.e2_generate " in c and "ckptdiff" in c]
    assert len(gen) == 2  # one retry
    jr = [c for c in open(os.path.join(runs, "calls.log")) if " judge_dolphin_answers.py " in c and "newset_butterfly_s1" in c
          and "rl_paper_c040" in c]
    assert len(jr) == 1 and "--inputs adamfx " in jr[0]
    assert open(os.path.join(m, "read_butterfly_all.FAILED")).read().strip().endswith("failed stages: read5")


def test_hold_for_fine_after(tmp_path):
    """5. scripts/animals/fine_after.sh polls for a GPU once whale's training ended: no GPU job starts until its
    LAUNCHED marker is 90 s old (CPU and judge stages go on)"""
    tmp = str(tmp_path)
    runs = setup(tmp)
    m = os.path.join(runs, "animals", "markers")
    inputs_ready(runs, "butterfly")
    touch(os.path.join(runs, "newset", "markers", "train_whale.DONE"))
    touch(os.path.join(m, "fine_butterfly.WAITING"))
    touch(os.path.join(runs, "loracle_ptrl", "markers", "ptrl_chain.DONE"))
    p = subprocess.Popen(["bash", DRIVER, "butterfly"], env=env_for(tmp, runs, FAKE_ADL_S=0.1),
                         stdout=open(os.path.join(tmp, "driver.out"), "w"), stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while not os.path.exists(os.path.join(m, "read_butterfly_control_adl.DONE")):
            assert time.time() - t0 < 60 and p.poll() is None, markers(runs)
            time.sleep(0.5)
        time.sleep(3)
        gpu = [c for c in open(os.path.join(runs, "calls.log")) if " main_run.reader.e2_generate " in c
               or " ao_diff_read.py " in c or " loracle_read.py " in c or " worker " in c]
        assert gpu == [] and "read_butterfly_export4.DONE" in markers(runs)
        touch(os.path.join(m, "fine_butterfly.LAUNCHED"))
        old = time.time() - 120
        os.utime(os.path.join(m, "fine_butterfly.LAUNCHED"), (old, old))
        while p.poll() is None:
            assert time.time() - t0 < 180, markers(runs)
            time.sleep(0.5)
    finally:
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, open(os.path.join(tmp, "driver.out")).read()
    assert "holding new GPU jobs" in open(os.path.join(runs, "animals", "events.log")).read()

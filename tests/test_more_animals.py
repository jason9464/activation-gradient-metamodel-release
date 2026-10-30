"""CPU tests of the butterfly and whale additions (user decision 2026-10-25 21:05, plan
docs/notes/20261025_more_animals_plan.md):

  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_more_animals.py

1. bash -n on scripts/animals/train_all.sh and the two newset scripts it calls.
2. train_subliminal.py: the arms parse, their data files are the pinned snapshots, and the settings train.sh gives a
   butterfly or whale run equal the owl full run's RUN.json settings except arm, data path, stop step and the LoRA pair
   steps that the later stop adds (5001); owl, dolphin, clean and mixed settings are what they were.
3. train.sh --dry-run accepts butterfly and whale (no --save-grads with --no-save-grads, stop 7000) and still refuses an
   unknown arm; ckptdiff_watch.sh refuses an unknown arm and takes newset_<animal>_s1 as the run.
4. subliminal_eval without a model: the extra animals' surface forms and the animal bookkeeping (the values on the
   real model: tests/check_extra_animals_eval.py, GPU).
5. plot_e2_newset.probe and behaviour_onsets read P(butterfly) from logprob records written in the trainer's schema.
6. train_all.sh with stub train.sh and watcher scripts: animals in order, the second training starts only after the
   first DONE, each watcher started right after its training, driver.DONE; a FAILED training stops the driver before
   the next animal; a DONE animal is skipped on a rerun (its watcher still started); a run dir left by a FAILED
   training is renamed aside, never deleted; a non-empty run dir without markers stops the driver.
"""
import json
import math
import os
import subprocess
import sys
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run"))

import plot_e2_newset as pe  # noqa: E402
import subliminal_eval as se  # noqa: E402
import train_subliminal as ts  # noqa: E402

IIF = os.environ["IIF_BASE"]
DRIVER = os.path.join(ROOT, "scripts", "animals", "train_all.sh")
TRAIN_SH = os.path.join(ROOT, "scripts", "newset", "train.sh")
WATCH_SH = os.path.join(ROOT, "scripts", "newset", "ckptdiff_watch.sh")
OWL_RUN = os.path.join(IIF, "agm_data", "runs", "newset_owl_s1")


def test_bash_syntax():
    for f in (DRIVER, TRAIN_SH, WATCH_SH):
        r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def _train_sh_args(arm, out, stop, pairs):
    fixed = os.path.join(IIF, "agm_data", "newset", "fixed_docs_128.jsonl")
    return ["--arm", arm, "--n-rows", "50000", "--lr", "1e-4", "--batch-size", "32", "--epochs", "7",
            "--warmup-ratio", "0.05", "--eval-every", "100", "--full-eval-every", "500", "--save-every", "500",
            "--seed", "1", "--grad-ckpt", "0", "--stop-step", str(stop), "--save-adamfx", "--adamfx-stepsums",
            "adamfx", "--adamfx-stepsums-dtype", "fp32", "--adamfx-fixed-docs", fixed, "--adamfx-fixed-every", "100",
            "--adamfx-lora-pairs", pairs, "--adamfx-lora-pairs-format", "fp32", "--adapter-dtype", "fp32",
            "--save-adapter-step0", "--no-checkpoints", "--out", out]


def test_trainer_arms_and_settings():
    for arm, rev in (("butterfly", "4a4c272d1f4960ce09e779a77cefdc07cfdf02e1"),
                     ("whale", "2ab8e88f8d36afc57ea8e199119290f7f701d2ff")):
        assert ts.DATA[arm].endswith(f"datasets--cds-jb--qwen3-14b-{arm}-subliminal-nums-25x/snapshots/{rev}/"
                                     "filtered_full.jsonl")
        assert os.path.exists(ts.DATA[arm]), ts.DATA[arm]
    if not os.path.exists(os.path.join(OWL_RUN, "RUN.json")):
        pytest.skip("owl run not on this machine")
    owl = json.load(open(os.path.join(OWL_RUN, "RUN.json")))["settings"]
    a = ts.parse_args(_train_sh_args("owl", OWL_RUN, 5000, "101,1001,3001"))
    assert ts.settings_of(a, os.path.abspath(ts.DATA["owl"])) == owl  # owl unchanged
    for arm in ("butterfly", "whale"):
        out = os.path.join(IIF, "agm_data", "runs", f"newset_{arm}_s1")
        a = ts.parse_args(_train_sh_args(arm, out, 7000, "101,1001,3001,5001"))
        s = ts.settings_of(a, os.path.abspath(ts.DATA[arm]))
        diff = {k for k in set(s) | set(owl) if s.get(k) != owl.get(k)}
        assert diff == {"arm", "data_path", "stop_step", "adamfx_lora_pairs"}, diff
        assert s["stop_step"] == 7000 and s["arm"] == arm and s["data_path"] == ts.DATA[arm]
    for arm in ("dolphin", "clean"):
        assert ts.parse_args(["--arm", arm, "--out", "/x"]).arm == arm
    assert ts.parse_args(["--arm", "mixed", "--out", "/x", "--mix-dolphin-frac", "0.5"]).arm == "mixed"
    with pytest.raises(SystemExit):
        ts.parse_args(["--arm", "cat", "--out", "/x"])


def test_train_sh_dry_run(tmp_path):
    env = dict(os.environ, NEWSET_STATE=str(tmp_path / "state"), NEWSET_SKIP_GPU_CHECK="1", NEWSET_STAMP="test")
    for arm in ("butterfly", "whale"):
        out = str(tmp_path / f"newset_{arm}_s1")
        r = subprocess.run(["bash", TRAIN_SH, arm, "0", out, "7000", "--no-save-grads", "--dry-run"], env=env,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        cmd = next(line for line in r.stdout.splitlines() if line.startswith("command:"))
        assert f"--arm {arm} " in cmd and "--stop-step 7000 " in cmd and "--save-grads" not in cmd
        assert "--adamfx-lora-pairs 101,1001,3001,5001 " in cmd and "dry run: nothing launched" in r.stdout
    r = subprocess.run(["bash", TRAIN_SH, "cat", "0", str(tmp_path / "x"), "100", "--dry-run"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "arm must be" in r.stderr
    r = subprocess.run(["bash", WATCH_SH, "cat", "0"], env=dict(env, NS_STATE=str(tmp_path / "state")),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "arm must be" in r.stderr


def test_eval_animal_sets():
    assert se.ANIMALS == ("owl", "dolphin") and se.EXTRA_ANIMALS == ("butterfly", "whale")
    assert se.surface_forms("butterfly") == ["butterfly", " butterfly", "Butterfly", " Butterfly"]
    assert se.surface_forms("whale") == ["whale", " whale", "Whale", " Whale"]
    # public matching rule: exact word, plural is not a hit
    assert se.normalize(" Whale!") == "whale" and se.normalize("Butterflies") != "butterfly"


def test_eval_animals_source():
    """The Evaluator's animal bookkeeping (the model-level check is tests/check_extra_animals_eval.py on a GPU)."""
    src = open(se.__file__).read()
    assert "self.animals = ANIMALS + self.extra_animals" in src
    assert "res, top = self._probe_batch(ANIMALS)" in src  # owl/dolphin keep their own batch
    assert "self.extra_animals = tuple(x for x in EXTRA_ANIMALS if x in extra_animals)" in src


def _logprob_rec(step, p):
    r = {a: {f: dict(ids=[1], logprob_first=-1.0, logprob_sum=math.log(p / 4)) for f in se.surface_forms(a)}
         for a in ("owl", "dolphin", "butterfly", "whale")}
    for a in r:
        r[a]["logprob_any"] = math.log(p)
    r["top_first_tokens"] = [["Butterfly", 0.1]]
    r["question"] = se.PROBE_QUESTION
    return dict(step=step, kind="logprob", result=r, arm="butterfly", target="butterfly")


def test_plot_probe_reads_butterfly(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "RUNS_DIR", str(tmp_path))
    run = tmp_path / "newset_butterfly_s1"
    run.mkdir()
    ps = {t: (0.01 if t < 2000 else (0.2 if t < 3000 else 0.7)) for t in range(0, 7001, 100)}
    run.joinpath("evals.jsonl").write_text("".join(json.dumps(_logprob_rec(t, p)) + "\n" for t, p in ps.items()))
    curve = pe.probe("newset_butterfly_s1", "butterfly")
    assert len(curve) == 71 and all(abs(v - ps[t]) < 1e-9 for t, v in curve)
    assert pe.behaviour_onsets(curve, list(range(500, 7001, 500))) == (2000, 3000)
    assert len(pe.probe("newset_butterfly_s1", "whale")) == 71
    assert pe.probe("newset_butterfly_s1", "cat") == []


# ------------------------------------------------------------------ 6. the driver with stub scripts

STUB_TRAIN = r'''#!/bin/bash
# stub of scripts/newset/train.sh: <arm> <gpu> <out> <stop> --no-save-grads (sets older STARTED/FAILED/pid markers aside
# before the launch, as the real one does)
arm=$1; gpu=$2; out=$3; stop=$4
M=$NEWSET_STATE/markers; mkdir -p "$M"
[ "$5" = --no-save-grads ] || { echo "stub: expected --no-save-grads" >&2; exit 2; }
[ "${6:-}" = --dry-run ] && { echo "stub dry run $arm"; exit 0; }
if [ -e "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "stub: $out not empty" >&2; exit 2; fi
for f in STARTED FAILED pid; do [ -e "$M/train_$arm.$f" ] && mv -n "$M/train_$arm.$f" "$M/train_$arm.$f.before_$(date +%s%N)"; done
mkdir -p "$out"; echo x > "$out/RUN.json"
echo "$(date +%s.%N) start $arm gpu $gpu out $out stop $stop" >> "$STUB_LOG"
sleep "${STUB_TRAIN_S:-3}" &
echo $! > "$M/train_$arm.pid"
( wait_s=${STUB_TRAIN_S:-3}; sleep "$wait_s"
  case " ${FAKE_FAIL:-} " in *" $arm "*) echo "failed" > "$M/train_$arm.FAILED";;
    *) echo "$(date +%s.%N) done $arm" >> "$STUB_LOG"; echo "exit 0" > "$M/train_$arm.DONE";; esac ) &
exit 0
'''
STUB_WATCH = r'''#!/bin/bash
echo "$(date +%s.%N) watch $1 gpu $2 state $NS_STATE" >> "$STUB_LOG"
'''


def _driver_env(tmp, **kw):
    for name, text in (("train.sh", STUB_TRAIN), ("watch.sh", STUB_WATCH)):
        open(os.path.join(tmp, name), "w").write(text)
    e = dict(os.environ)
    e.update(ANIMALS_STATE=os.path.join(tmp, "animals"), ANIMALS_NEWSET_STATE=os.path.join(tmp, "newset"),
             ANIMALS_RUNS=os.path.join(tmp, "runs"), ANIMALS_TRAIN_SH=os.path.join(tmp, "train.sh"),
             ANIMALS_WATCH_SH=os.path.join(tmp, "watch.sh"), ANIMALS_POLL="1", ANIMALS_SKIP_GPU_CHECK="1",
             ANIMALS_LOGS=os.path.join(tmp, "logs"), ANIMALS_STAMP="test", STUB_LOG=os.path.join(tmp, "stub.log"),
             ANIMALS_DEAD_GRACE_S="5")
    e.update({k: str(v) for k, v in kw.items()})
    return e


def _run_driver(tmp, env, args, wait_for=("DONE", "FAILED"), timeout=60):
    r = subprocess.run(["bash", DRIVER, *args], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    M = os.path.join(tmp, "animals", "markers")
    t0 = time.time()
    while time.time() - t0 < timeout:
        got = [m for m in wait_for if os.path.exists(os.path.join(M, f"driver.{m}"))]
        if got:
            return got[0]
        time.sleep(0.5)
    raise AssertionError("driver did not finish: " + open(os.path.join(tmp, "logs", "test_animals_train_all.out")).read())


def _stub_lines(tmp):
    return [line.split()[1:] for line in open(os.path.join(tmp, "stub.log"))]


def test_driver_order_and_overlap(tmp_path):
    tmp = str(tmp_path)
    env = _driver_env(tmp)
    assert _run_driver(tmp, env, ["1", "2", "butterfly", "whale"]) == "DONE"
    ev = [x[:2] for x in _stub_lines(tmp)]
    assert ev == [["start", "butterfly"], ["watch", "butterfly"], ["done", "butterfly"], ["start", "whale"],
                  ["watch", "whale"], ["done", "whale"]], ev
    lines = _stub_lines(tmp)
    assert lines[0][2:] == ["gpu", "1", "out", os.path.join(tmp, "runs", "newset_butterfly_s1"), "stop", "7000"]
    assert lines[1][2:] == ["gpu", "2", "state", os.path.join(tmp, "newset")]
    events = open(os.path.join(tmp, "animals", "events.log")).read()
    assert "DONE: every training (butterfly whale)" in events
    # rerun: both DONE -> nothing trained, watchers started again (they exit at once when their own DONE exists)
    open(os.path.join(tmp, "stub.log"), "w").close()
    assert _run_driver(tmp, env, ["1", "2", "butterfly", "whale"]) == "DONE"
    assert [x[:2] for x in _stub_lines(tmp)] == [["watch", "butterfly"], ["watch", "whale"]]
    assert any(n.startswith("driver.DONE.before_") for n in os.listdir(os.path.join(tmp, "animals", "markers")))


def test_driver_failed_stops(tmp_path):
    tmp = str(tmp_path)
    env = _driver_env(tmp, FAKE_FAIL="butterfly")
    assert _run_driver(tmp, env, ["0", "3", "butterfly", "whale"]) == "FAILED"
    assert [x[:2] for x in _stub_lines(tmp)] == [["start", "butterfly"], ["watch", "butterfly"]]
    txt = open(os.path.join(tmp, "animals", "markers", "driver.FAILED")).read()
    assert "butterfly: training FAILED" in txt
    # rerun after the failure: the failed run dir is moved aside (kept), butterfly trained again, then whale
    open(os.path.join(tmp, "stub.log"), "w").close()
    env2 = _driver_env(tmp)
    assert _run_driver(tmp, env2, ["0", "3", "butterfly", "whale"]) == "DONE"
    runs = os.listdir(os.path.join(tmp, "runs"))
    assert any(n.startswith("newset_butterfly_s1.before_") for n in runs), runs
    assert [x[:2] for x in _stub_lines(tmp) if x[0] == "start"] == [["start", "butterfly"], ["start", "whale"]]


def test_driver_skips_done_and_refuses_unknown_dir(tmp_path):
    tmp = str(tmp_path)
    env = _driver_env(tmp)
    NM = os.path.join(tmp, "newset", "markers")
    os.makedirs(NM)
    open(os.path.join(NM, "train_butterfly.DONE"), "w").write("earlier\n")
    assert _run_driver(tmp, env, ["1", "2", "butterfly", "whale"]) == "DONE"
    assert [x[:2] for x in _stub_lines(tmp)] == [["watch", "butterfly"], ["start", "whale"], ["watch", "whale"],
                                                 ["done", "whale"]]
    # a non-empty run dir without any marker: stop, touch nothing
    tmp2 = str(tmp_path / "second")
    os.makedirs(os.path.join(tmp2, "runs", "newset_whale_s1"))
    open(os.path.join(tmp2, "runs", "newset_whale_s1", "keep.txt"), "w").write("x")
    env = _driver_env(tmp2)
    assert _run_driver(tmp2, env, ["1", "2", "whale"]) == "FAILED"
    assert os.path.exists(os.path.join(tmp2, "runs", "newset_whale_s1", "keep.txt"))
    assert "without a DONE or FAILED marker" in open(os.path.join(tmp2, "animals", "markers", "driver.FAILED")).read()


def test_driver_argument_checks(tmp_path):
    tmp = str(tmp_path)
    env = _driver_env(tmp)
    for args, msg in ((["1", "1", "whale"], "different GPUs"), (["1", "5", "whale"], "only 0-3"),
                      (["1", "2", "owl"], "butterfly or whale"), (["1", "2", "whale", "whale"], "twice"),
                      (["1", "2"], "usage")):
        r = subprocess.run(["bash", DRIVER, *args], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 2 and msg in r.stderr, (args, r.stderr)
    r = subprocess.run(["bash", DRIVER, "1", "2", "butterfly", "--dry-run"], env=env, capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 0 and "dry run: nothing launched" in r.stdout and "stub dry run butterfly" in r.stdout
    assert not os.path.exists(os.path.join(tmp, "stub.log"))  # train.sh is called with --dry-run only

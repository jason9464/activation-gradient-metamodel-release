"""eval_read.py run: hang timeout, hang retry alone on the GPU, and robust "existing outputs skipped" (CPU only).

The runner is driven with a fake e2_generate (a small python script written into the test root) that either writes a
header line and sleeps forever ("hang"), hangs only on its first attempt ("hang_once"), ignores TERM while hanging
("hang_ignore_term"), exits 3 ("fail"), or writes progress lines and a reading file and exits 0 ("ok"); "log_sparse"
writes the partial output every step but no log line until the end, "log_only" writes log lines but no partial output
until the end, "done_then_hang" writes its whole output and then never exits. IIF_BASE points at the test root before
eval_common is imported, so logs, outputs and markers all land inside it.

  EM_EVAL_HANG_TEST_ROOT=<dir> python -m pytest -q tests/test_eval_read_hang.py   (root defaults to pytest's tmp_path)
"""
from __future__ import annotations

import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "main_run" / "em_reader"
MODS = ("eval_common", "eval_read")

FAKE = r'''
import json, os, signal, sys, time
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
mode, out, state, key, gpu = a["--mode"], a["--out"], a["--state"], a["--key"], a["--gpu"]
steps = [int(s) for s in a["--steps"].split(",")]
safe = key.replace("/", "__")
cnt = os.path.join(state, safe + ".attempts")
n = (int(open(cnt).read()) if os.path.exists(cnt) else 0) + 1
open(cnt, "w").write(str(n))
open(os.path.join(state, f"{safe}.{n}.pid"), "w").write(str(os.getpid()))

def event(kind):
    with open(os.path.join(state, "events.jsonl"), "a") as f:
        f.write(json.dumps(dict(key=key, attempt=n, gpu=gpu, kind=kind, t=time.time())) + "\n")

event("start")
print(f"[fake_e2] {key} attempt {n} on GPU {gpu}: inputs header", flush=True)
if mode == "hang_ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode in ("hang", "hang_ignore_term") or (mode == "hang_once" and n == 1):
    while True:
        time.sleep(3600)
if mode == "fail":
    sys.exit(3)
dur = float(a["--dur"])
row = lambda s: json.dumps(dict(step=s, inputs={"bundle": dict(texts=["x"])})) + "\n"
if mode == "log_only":
    for i, s in enumerate(steps):
        time.sleep(dur / len(steps))
        print(f"[fake_e2] working {i + 1}/{len(steps)}", flush=True)
    with open(out + ".tmp", "w") as f:
        f.write("".join(row(s) for s in steps))
else:
    with open(out + ".tmp", "w") as f:
        for i, s in enumerate(steps):
            time.sleep(dur / len(steps))
            f.write(row(s))
            f.flush()
            if mode != "log_sparse":
                print(f"[fake_e2] {i + 1}/{len(steps)} steps", flush=True)
os.replace(out + ".tmp", out)
event("end")
if mode == "done_then_hang":
    while True:
        time.sleep(3600)
'''


def real_iif_base() -> Path:
    if os.environ.get("IIF_BASE"):
        return Path(os.environ["IIF_BASE"])
    src = (HERE / "eval_common.py").read_text()
    return Path(re.search(r'IIF_BASE = os\.environ\.get\("IIF_BASE", "([^"]+)"\)', src).group(1))


def real_test_logs(base: Path) -> list[str]:
    return sorted(str(p) for p in (base / "agm_data" / "runs").glob("*/read_logs/*hangtest*"))


class Runner:
    def __init__(self, root: Path, E, R):
        self.root, self.E, self.R = root, E, R
        self.state = root / "state"
        self.state.mkdir()
        self.fake = root / "bin" / "fake_e2.py"
        self.fake.parent.mkdir()
        self.fake.write_text(FAKE)
        self.adapter = root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"x")
        (root / "RUN.json").write_text("{}")
        self.jobs: list[dict] = []
        E.reading_jobs = lambda readers=("new",), **kw: [dict(j) for j in self.jobs]
        E.e2_command = self.command
        R.STAGGER_S = 0.0

    def command(self, job, gpu, *, max_gpu_gib=80.0):
        return [sys.executable, str(self.fake), "--mode", job["mode"], "--out", str(job["out"]), "--state",
                str(self.state), "--key", job["key"], "--gpu", str(gpu), "--steps", ",".join(map(str, job["steps"])),
                "--dur", str(job["dur"])]

    def job(self, name: str, mode: str = "ok", steps=(0, 1, 2), dur: float = 0.6) -> dict:
        out = self.E.EV / name / "e2" / "reader_generic_q.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        j = dict(kind="data", key=f"hangtest/{name}/new/q", mode=mode, steps=list(steps), dur=dur, out=out,
                 inputs=self.E.EV / name / "inputs_bundles.npz", adapter=self.adapter, run_json=self.root / "RUN.json",
                 name="reader", questions="generic", question="q")
        self.jobs.append(j)
        return j

    def attempts(self, j) -> int:
        p = self.state / (j["key"].replace("/", "__") + ".attempts")
        return int(p.read_text()) if p.exists() else 0

    def pids(self) -> list[int]:
        return [int(p.read_text()) for p in self.state.glob("*.pid")]

    def events(self) -> list[dict]:
        p = self.state / "events.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def run(self, **kw) -> int:
        kw = dict(dict(gpus=["0"], per_gpu=2, hang_s=2.0, retry_hung=1, kill_grace_s=5.0), **kw)
        return self.R.cmd_run(("new",), kw.pop("gpus"), kw.pop("per_gpu"), 80.0, **kw)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}").exists() and "State:\tZ" not in Path(f"/proc/{pid}/status").read_text()


@pytest.fixture()
def runner(tmp_path, monkeypatch, request):
    base = os.environ.get("EM_EVAL_HANG_TEST_ROOT")
    if base:
        root = Path(base) / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{request.node.name}"
        root.mkdir(parents=True)
    else:
        root = tmp_path
    real = real_iif_base()
    real_before = real_test_logs(real)
    monkeypatch.setenv("IIF_BASE", str(root))
    monkeypatch.delenv("EM_READ_HANG_S", raising=False)
    monkeypatch.delenv("EM_READ_WORKERS_PER_GPU", raising=False)
    sys.path.insert(0, str(HERE))
    for m in MODS:
        sys.modules.pop(m, None)
    E = importlib.import_module("eval_common")
    R = importlib.import_module("eval_read")
    E.EMWT.mkdir(parents=True)  # the runner's cwd for each read
    for p in (E.EVAL_ROOT, E.EV, E.EMWT, R.LOG_DIR):
        assert Path(p).resolve().is_relative_to(root.resolve()), p
    rn = Runner(root, E, R)
    yield rn
    for pid in rn.pids():  # never leave a fake read running, whatever the test did
        try:
            if str(rn.fake) in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace"):
                os.kill(pid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
    sys.path.remove(str(HERE))
    for m in MODS:
        sys.modules.pop(m, None)
    assert real_test_logs(real) == real_before, "the runner wrote a test log outside the test root"
    written = [p for p in root.rglob("*") if p.is_file()]
    assert all(p.resolve().is_relative_to(root.resolve()) for p in written)


def test_output_problem(runner):
    R = runner.R
    j = runner.job("op", steps=(0, 1, 2))
    out = Path(j["out"])
    assert R.output_problem(j) == "missing"
    out.write_text("")
    assert R.output_problem(j) == "empty"
    out.write_text("\n\n")
    assert R.output_problem(j) == "empty (no rows)"
    out.write_text('{"step": 0, "inputs": {}}\n{"step": 1, "inp')
    assert R.output_problem(j).startswith("unparsable (JSONDecodeError")
    out.write_text('[1, 2]\n')
    assert R.output_problem(j).startswith("unparsable (line 1")
    out.write_bytes(b"\xff\xfe\x00garbage")
    assert R.output_problem(j).startswith("unparsable")
    out.write_text('{"step": 0, "inputs": {}}\n{"step": 2, "inputs": {}}\n')
    assert R.output_problem(j) == "short (1 of 3 steps missing)"
    out.write_text("".join(json.dumps(dict(step=s, inputs={})) + "\n" for s in (0, 1, 2)))
    assert R.output_problem(j) is None


def test_hang_seconds_and_cli(runner, monkeypatch, capsys):
    R = runner.R
    assert R.hang_seconds(None) == 600.0
    monkeypatch.setenv("EM_READ_HANG_S", "7")
    assert R.hang_seconds(None) == 7.0
    assert R.hang_seconds(3) == 3.0  # the flag wins over the environment
    assert R.main(["run", "--readers", "new", "--gpus", "0", "--hang-s", "2", "--retry-hung", "0",
                   "--kill-grace-s", "1"]) == 0  # no jobs: wiring only
    out = capsys.readouterr().out
    assert "hang timeout 2 s (log and partial output unchanged; TERM, KILL 1 s later)" in out
    assert "queued again up to 0 time(s)" in out
    assert R.main(["run", "--readers", "new", "--gpus", "0"]) == 0
    assert "hang timeout 7 s" in capsys.readouterr().out
    assert R.main(["run", "--readers", "new", "--gpus", "0", "--hang-s", "0"]) == 0
    assert "hang timeout off" in capsys.readouterr().out
    monkeypatch.setenv("EM_READ_WORKERS_PER_GPU", "1")
    assert R.main(["run", "--readers", "new", "--gpus", "0,1", "--workers-per-gpu", "2"]) == 0
    out = capsys.readouterr().out
    assert "workers per GPU 1 from $EM_READ_WORKERS_PER_GPU (flag said 2)" in out and "['0', '1'] x 1" in out


def test_hung_job_killed_retried_once_then_failed(runner, capsys):
    good = runner.job("good", "ok", dur=0.6)
    bad = runner.job("bad", "hang")
    t0 = time.time()
    rc = runner.run(hang_s=2.0, retry_hung=1)
    out = capsys.readouterr().out
    assert rc == 1
    assert runner.R.output_problem(good) is None
    assert runner.attempts(good) == 1 and runner.attempts(bad) == 2  # one retry, no more
    term = [x for x in out.splitlines() if x.split("] ", 1)[1].startswith(f"hang: {bad['key']}") and "TERM" in x]
    assert len(term) == 2
    failed = [x for x in out.splitlines() if f"FAILED {bad['key']}" in x]
    assert len(failed) == 2 and all("rc -15; hung (log unchanged " in x for x in failed)
    assert "hang retry 1/1 queued, alone on its GPU" in failed[0] and "hang retry" not in failed[1].split(")")[-1]
    assert f"done {good['key']}" in out
    starts = [x for x in out.splitlines() if f"[eval_read] start {bad['key']}" in x]
    assert len(starts) == 2 and "(hang retry 1/1, alone on the GPU) pid" in starts[1]
    assert f"finished: 1 done, 1 failed ['{bad['key']}']" in out
    assert not any(alive(p) for p in runner.pids())
    assert time.time() - t0 < 30
    logs = sorted(runner.R.LOG_DIR.glob("hangtest__bad*"))
    assert len(logs) == 2 and all("inputs header" in p.read_text() for p in logs)


def test_hang_once_then_done_alone_on_gpu(runner, capsys):
    flaky = runner.job("flaky", "hang_once")
    others = [runner.job("d", dur=1.0), runner.job("e", dur=3.0), runner.job("f", dur=3.0)]
    rc = runner.run(hang_s=2.0, retry_hung=1)
    out = capsys.readouterr().out
    assert rc == 0
    assert "finished: 4 done, 0 failed []" in out
    assert f"hang retries: 1 ['{flaky['key']}']" in out
    assert runner.attempts(flaky) == 2 and all(runner.attempts(j) == 1 for j in others)
    assert all(runner.R.output_problem(j) is None for j in [flaky] + others)
    ev = runner.events()
    span = {}
    for e in ev:
        span.setdefault((e["key"], e["attempt"]), {})[e["kind"]] = e["t"]
    s2, e2 = span[(flaky["key"], 2)]["start"], span[(flaky["key"], 2)]["end"]
    for j in others:
        s, e = span[(j["key"], 1)]["start"], span[(j["key"], 1)]["end"]
        assert e <= s2 or s >= e2, f"{j['key']} ran on GPU 0 during the hang retry"
    assert f"hang retry of {flaky['key']} waits for GPU 0 to be free" in out


def test_kill_after_grace_when_term_is_ignored(runner, capsys):
    stub = runner.job("stubborn", "hang_ignore_term")
    rc = runner.run(hang_s=1.5, retry_hung=0, kill_grace_s=1.0)
    out = capsys.readouterr().out
    assert rc == 1
    assert "still running 1 s after TERM; KILL" in out
    assert f"FAILED {stub['key']}" in out and "rc -9; hung (log unchanged " in out
    assert "hang retry" not in out
    assert f"finished: 0 done, 1 failed ['{stub['key']}']" in out
    assert runner.attempts(stub) == 1 and not any(alive(p) for p in runner.pids())


def test_nonzero_exit_is_not_retried(runner, capsys):
    j = runner.job("broken", "fail")
    ok = runner.job("fine", "ok")
    rc = runner.run(hang_s=2.0, retry_hung=1)
    out = capsys.readouterr().out
    assert rc == 1
    assert f"FAILED {j['key']}" in out and "(rc 3; " in out and "hang retry" not in out
    assert runner.attempts(j) == 1 and runner.R.output_problem(ok) is None
    assert f"finished: 1 done, 1 failed ['{j['key']}']" in out


def test_existing_outputs_skipped_only_when_complete(runner, capsys):
    empty, garbage, short, good = (runner.job(n) for n in ("empty", "garbage", "short", "good"))
    Path(empty["out"]).write_text("")
    Path(garbage["out"]).write_text("not json\n")
    Path(short["out"]).write_text(json.dumps(dict(step=0, inputs={})) + "\n")
    Path(good["out"]).write_text("".join(json.dumps(dict(step=s, inputs={"bundle": {}})) + "\n" for s in (0, 1, 2)))
    good_before = Path(good["out"]).read_text()
    assert runner.R.cmd_plan(("new",)) == 0
    plan = capsys.readouterr().out
    assert f"[redo: output empty] {empty['key']}" in plan
    assert f"[redo: output unparsable (JSONDecodeError)] {garbage['key']}" in plan
    assert f"[redo: output short (2 of 3 steps missing)] {short['key']}" in plan
    assert f"[done] {good['key']}" in plan and plan.rstrip().endswith("4 jobs, 3 to run")
    rc = runner.run()
    out = capsys.readouterr().out
    assert rc == 0
    assert "3 jobs for readers" in out and "finished: 3 done, 0 failed []" in out
    assert runner.attempts(good) == 0 and Path(good["out"]).read_text() == good_before
    for j, old in ((empty, ""), (garbage, "not json\n"), (short, json.dumps(dict(step=0, inputs={})) + "\n")):
        assert runner.attempts(j) == 1 and runner.R.output_problem(j) is None
        kept = list(Path(j["out"]).parent.glob(Path(j["out"]).name + ".invalid_*"))
        assert len(kept) == 1 and kept[0].read_text() == old  # set aside, not deleted


def test_log_and_partial_output_watched_separately(runner, capsys):
    # each fake is silent on one of the two files for 4 s, longer than the 1.5 s timeout; watching only one file
    # would stop it
    sparse = runner.job("sparse", "log_sparse", steps=range(8), dur=4.0)
    logonly = runner.job("logonly", "log_only", steps=range(8), dur=4.0)
    rc = runner.run(hang_s=1.5, retry_hung=0)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "hang:" not in out and "finished: 2 done, 0 failed []" in out
    assert runner.attempts(sparse) == 1 and runner.attempts(logonly) == 1


def test_hung_after_complete_output_counts_done(runner, capsys):
    j = runner.job("dth", "done_then_hang", dur=0.3)
    rc = runner.run(hang_s=1.5, retry_hung=1, kill_grace_s=2.0)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"FAILED {j['key']}" in out and "hang retry 1/1 queued" in out
    assert f"done {j['key']} (output already complete, hang retry not needed" in out
    assert "finished: 1 done, 0 failed []" in out and runner.attempts(j) == 1
    assert not any(alive(p) for p in runner.pids())


class FakeProc:
    """Popen stand-in for run_watched: "ended" (gone by the hang look), "exit0_at_term" (ends normally as TERM is
    sent), "unkillable" (survives TERM and KILL)."""

    def __init__(self, mode):
        self.mode, self.pid, self.returncode, self.signals = mode, 4242, None, []

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        time.sleep(min(timeout or 0.0, 0.05))
        raise subprocess.TimeoutExpired("fake", timeout)

    def poll(self):
        return 0 if self.mode == "ended" else self.returncode

    def terminate(self):
        self.signals.append("TERM")
        if self.mode == "exit0_at_term":
            self.returncode = 0

    def kill(self):
        self.signals.append("KILL")


@pytest.mark.parametrize("mode", ["ended", "exit0_at_term", "unkillable"])
def test_run_watched_races_and_unkillable(runner, monkeypatch, mode):
    R = runner.R
    proc = FakeProc(mode)
    monkeypatch.setattr(R.subprocess, "Popen", lambda *a, **k: proc)
    lp = runner.root / f"race_{mode}.log"
    t0 = time.time()
    rc, why = R.run_watched(["fake"], lp, runner.root / f"race_{mode}.tmp", hang_s=0.5, grace_s=0.3,
                            cwd=str(runner.root), env={}, label=mode)
    took = time.time() - t0
    if mode == "ended":
        assert (rc, why, proc.signals) == (0, None, [])
    elif mode == "exit0_at_term":
        assert (rc, why, proc.signals) == (0, None, ["TERM"])
    else:
        assert rc is None and why.startswith("unkillable: hung (log unchanged ") and proc.signals == ["TERM", "KILL"]
        assert not why.startswith("hung")  # the worker retries only reasons starting with "hung"
    assert took < 5


def test_one_runner_per_log_folder(runner, capsys):
    R = runner.R
    j = runner.job("locked")
    fd, holder = R.runner_lock()
    assert fd is not None and fd >= 0
    try:
        assert runner.run() == 3
        out = capsys.readouterr().out
        assert "another eval_read run holds" in out and f"pid {os.getpid()}" in out and "not started" in out
        assert runner.attempts(j) == 0
    finally:
        os.close(fd)
    assert runner.run() == 0 and runner.attempts(j) == 1


def test_side_files_set_aside_and_fresh_partial_not_started(runner, capsys):
    short = runner.job("side")
    miss = runner.job("leftover")
    busy = runner.job("busy")
    so = Path(short["out"])
    so.write_text(json.dumps(dict(step=0, inputs={})) + "\n")
    side = {so.with_name(so.stem + "_judge_broad.jsonl"): "broad\n", so.with_name(so.stem + "_judge_content.jsonl"):
            "content\n", so.with_name(so.stem + "_settings.json"): "{}\n", so.with_name(so.stem + "_summary.json"):
            "{}\n", Path(str(so) + ".tmp"): "partial\n"}
    other = so.with_name(so.stem + "2_judge_broad.jsonl")  # another reading's judged file: left alone
    other.write_text("other\n")
    mo = Path(miss["out"])
    side.update({mo.with_name(mo.stem + "_settings.json"): "{}\n", Path(str(mo) + ".tmp"): "half\n"})
    old = time.time() - 3600
    for p, text in side.items():
        p.write_text(text)
        os.utime(p, (old, old))
    Path(str(busy["out"]) + ".tmp").write_text("someone else\n")  # fresh: another process may be writing it
    rc = runner.run(hang_s=2.0)
    out = capsys.readouterr().out
    assert rc == 1
    assert f"FAILED {busy['key']}: partial output" in out and runner.attempts(busy) == 0
    assert Path(str(busy["out"]) + ".tmp").read_text() == "someone else\n"
    assert "finished: 2 done, 1 failed" in out
    for j in (short, miss):
        assert runner.attempts(j) == 1 and runner.R.output_problem(j) is None
    for p, text in side.items():
        kept = list(p.parent.glob(p.name + ".invalid_*"))
        assert len(kept) == 1 and kept[0].read_text() == text, p
        assert not p.exists(), p  # moved; the fake read writes none of these back
    assert len(list(so.parent.glob(so.name + ".invalid_*"))) == 1  # the short reading itself
    assert other.read_text() == "other\n" and not list(other.parent.glob(other.name + ".invalid_*"))

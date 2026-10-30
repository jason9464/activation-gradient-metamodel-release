"""scripts/beh_reader/new_readers_suite.sh: syntax, the
--dry stage lists of the three readers, the refusals, and the real runner on stub programs (new_readers_suite.py,
new_readers_suite_read.py, rare_yesno.py, rare_perdoc.py, judge_dolphin_answers.py and nvidia-smi replaced by one
Python stub): GPU choice and claims, markers and reruns, the slot docs -> extract -> build flow, judge caps and the
spend file, the judged-rows check, the cost watchdog and the hang guard. CPU only: no GPU, no model, no API, nothing
written outside the pytest basetemp.

  cd <scratch> && python -m pytest -q -p no:cacheprovider <repo>/tests/test_new_readers_suite_chain.py \\
      --basetemp <scratch>/new_readers_suite_chain_test
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SH = REPO / "scripts" / "beh_reader" / "new_readers_suite.sh"
FULL = ["build_e2", "read_e2", "judge_e2",
        "build_e3", "read_e3", "judge_e3", "yesno_e3",
        "build_e4", "read_e4", "judge_e4", "opensum_e4", "yesno_e4", "ynsum_e4",
        "build_e5", "read_e5", "judge_e5", "yesno_e5", "summary", "plot"]
SLOT = ["docs_e3", "extract_e3", "build_e3", "read_e3", "judge_e3", "yesno_e3",
        "docs_e4", "extract_e4", "build_e4", "read_e4", "judge_e4", "opensum_e4", "yesno_e4",
        "docs_e5", "extract_e5", "build_e5", "read_e5", "judge_e5", "yesno_e5", "summary", "plot"]


def run(args, env, timeout=300):
    return subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, env=env, timeout=timeout)


def dry_stages(out: str) -> list[str]:
    return [ln.split()[1] for ln in out.splitlines() if ln.startswith("[dry] ")]


def cmd_of(out: str, stage: str) -> str:
    """The command line printed under '[dry] <stage> (...)'."""
    lines = out.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(f"[dry] {stage} ("))
    return lines[i + 1]


# ---------------------------------------------------------------- syntax and --dry
def test_bash_syntax():
    r = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def dry_env(tmp_path: Path, **extra) -> dict:
    return dict(os.environ, IIF_BASE=str(tmp_path / "base"), ROOT=str(tmp_path / "root"),
                CLAIM_DIR=str(tmp_path / "claims"), **extra)


def test_dry_lists_stages_for_each_reader(tmp_path):
    env = dry_env(tmp_path)
    base = tmp_path / "base" / "agm_data" / "runs"
    root = tmp_path / "root"
    for reader in ("grad3", "twin"):
        r = run([reader, "--dry"], env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert dry_stages(r.stdout) == FULL, (reader, dry_stages(r.stdout))
        assert f"markers={root / 'markers' / reader}" in r.stdout
        for e, n in (("e2", 100), ("e3", 100), ("e4", 10), ("e5", 10)):
            c = cmd_of(r.stdout, f"read_{e}")
            assert f"new_readers_suite_read.py read --exp {e} --reader {reader} --root {root} --gpu <gpu>" in c
            assert f"--samples {n}" in c and "--run-json " in c
            assert f"new_readers_suite.py build --exp {e} --reader {reader} --out {root}" in cmd_of(r.stdout, f"build_{e}")
        for s in ("build_e2", "build_e3", "build_e4", "build_e5", "ynsum_e4", "opensum_e4", "summary", "plot"):
            assert "--gpu" not in cmd_of(r.stdout, s), s  # CPU stages
        assert "read_e2 (about 15 min; GPU with 80 GiB free among 1 2 3)" in r.stdout
        assert "yesno_e4 (about 5 min; GPU with 176 GiB free among 1 2 3)" in r.stdout  # rare_yesno: an empty GPU
        assert "new_readers_suite_read.py yesno --exp e3" in cmd_of(r.stdout, "yesno_e3")
        assert "new_readers_suite_read.py yesno --exp e5" in cmd_of(r.stdout, "yesno_e5")
        y = cmd_of(r.stdout, "yesno_e4")
        assert "rare_yesno.py read" in y and "--bundles rare_quirk" in y and "--reader-layers 10,20,30" in y
        assert f"--reader-name {reader}" in y and f"--run-dir {root / 'e4' / reader / 'yesno'}" in y
        assert f"rare_yesno.py summary --run-dir {root / 'e4' / reader / 'yesno'} --bundles rare_quirk" in cmd_of(r.stdout, "ynsum_e4")
        if reader == "grad3":
            assert "--input grad" in y and "--act-dir" not in y
            assert str(base / "reader_rl_scratch_L3_beh_paper" / "adapters" / "cycle_040") in y
        else:
            assert "--input act" in y and "--act-dir " + str(base / "beh_reader" / "rare_yesno" / "act") in y
            assert str(base / "reader_rl_scratch_L3act_beh_paper" / "adapters" / "cycle_040") in y
        j2 = cmd_of(r.stdout, "judge_e2")
        assert "judge_dolphin_answers.py" in j2 and f"--readers nrs_{reader}" in j2 and "--inputs batch" in j2
        assert "--question preference" in j2 and "--max-cost" in j2 and "env -u PYTHONPATH" in j2
        assert str(root / "e2" / reader) in j2
        assert f"rare_perdoc.py judge --out {root / 'e4' / reader / 'open'}" in cmd_of(r.stdout, "judge_e4")
        assert f"rare_perdoc.py summary --out {root / 'e4' / reader / 'open'}" in cmd_of(r.stdout, "opensum_e4")
        assert "--cap-usd <min(15 - " in cmd_of(r.stdout, "judge_e4") and "25 - suite spend" in cmd_of(r.stdout, "judge_e5")
        assert "new_readers_suite_read.py judge --exp e5" in cmd_of(r.stdout, "judge_e5")
        assert f"new_readers_suite.py summary --out {root}" in cmd_of(r.stdout, "summary")
        assert f"new_readers_suite.py plot --out {root} --fig {root / 'new_readers_suite_smoke.png'}" in cmd_of(r.stdout, "plot")
    r = run(["slot", "--dry"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert dry_stages(r.stdout) == SLOT, dry_stages(r.stdout)
    x = cmd_of(r.stdout, "extract_e3")
    assert (f"new_readers_suite.py slot-extract --docs {root / 'e3' / 'slot' / 'docs.jsonl'} "
            f"--out-dir {root / 'e3' / 'slot' / 'slots'} --gpu <gpu>") in x
    assert "extract_e5 (about 15 min; GPU with 110 GiB free among 1 2 3; only when docs_e5 leaves" in r.stdout
    assert "new_readers_suite_read.py yesno --exp e4 --reader slot" in cmd_of(r.stdout, "yesno_e4")
    assert str(base / "reader_rl_scratch_L3slot_beh_paper" / "adapters" / "cycle_040") in cmd_of(r.stdout, "read_e3")
    # one experiment only
    r = run(["grad3", "e4", "--dry"], env)
    assert dry_stages(r.stdout) == ["build_e4", "read_e4", "judge_e4", "opensum_e4", "yesno_e4", "ynsum_e4", "summary",
                                    "plot"]
    # LIMIT (smoke, ROOT outside the real one): --limit to the build and the GPU tools, judges forced to --dry-run
    r = run(["grad3", "e2", "e4", "--dry"], dict(env, LIMIT="2"))
    assert r.returncode == 0 and "JUDGE_DRY=1" in r.stdout
    for s in ("build_e2", "read_e2", "yesno_e4"):
        assert "--limit 2" in cmd_of(r.stdout, s), s
    assert cmd_of(r.stdout, "judge_e2").endswith("--dry-run") and cmd_of(r.stdout, "judge_e4").endswith("--dry-run")
    # --dry writes nothing
    assert not root.exists() and not (tmp_path / "base").exists() and not (tmp_path / "claims").exists()


def test_dry_adapter_after_rl_cost_cap(tmp_path):
    """After a reader's RL stopped at its cost cap (the twin / slot chain's rl.COST_CAP), the SFT adapter is read."""
    tm, sm = tmp_path / "twin_m", tmp_path / "slot_m"
    for d in (tm, sm):
        d.mkdir()
        (d / "rl.COST_CAP").write_text("3.6")
    env = dry_env(tmp_path, TWIN_M=str(tm), SLOT_M=str(sm))
    r = run(["twin", "e4", "--dry"], env)
    assert "reader_scratch_L3act_beh/adapters/step_007082" in cmd_of(r.stdout, "read_e4")
    assert "reader_scratch_L3act_beh/adapters/step_007082" in cmd_of(r.stdout, "yesno_e4")
    r = run(["slot", "e5", "--dry"], env)
    assert "reader_scratch_L3slot_beh/adapters/step_007082" in cmd_of(r.stdout, "read_e5")
    r = run(["grad3", "e4", "--dry"], env)  # the gradient reader has no fallback
    assert "reader_rl_scratch_L3_beh_paper/adapters/cycle_040" in cmd_of(r.stdout, "read_e4")


def test_refusals(tmp_path):
    env = dry_env(tmp_path)
    for args, word in ((["grad40", "--dry"], "unknown reader"), (["--dry"], "usage"), (["e3", "--dry"], "reader comes first"),
                       (["slot", "e2", "--dry"], "not read with the slot reader"),
                       (["grad3", "e6", "--dry"], "is not run in this suite"),
                       (["grad3", "e9", "--dry"], "unknown experiment"), (["grad3", "twin", "--dry"], "one reader")):
        r = run(args, env)
        assert r.returncode == 2 and word in r.stdout, (args, r.stdout)
    r = run(["grad3", "--dry"], dict(env, GPUS="1 x"))
    assert r.returncode == 2 and "not a GPU index" in r.stdout
    r = run(["grad3", "--dry"], dict(env, GPUS="0 1"))  # GPU 0 is the slot chain's: only with ALLOW_GPU0=1
    assert r.returncode == 2 and "ALLOW_GPU0" in r.stdout
    r = run(["grad3", "e4", "--dry"], dict(env, GPUS="0 1", ALLOW_GPU0="1"))
    assert r.returncode == 0 and "among 0 1" in r.stdout
    env_real = {k: v for k, v in env.items() if k != "ROOT"}  # LIMIT (smoke) with the real ROOT
    r = run(["grad3", "--dry"], dict(env_real, LIMIT="2"))
    assert r.returncode == 2 and "smoke option" in r.stdout
    # a real launch without the reader's adapter refuses before writing anything
    r = run(["grad3", "e4"], env)
    assert r.returncode == 1 and "launch this chain when they exist" in r.stdout
    assert not (tmp_path / "root").exists() and not (tmp_path / "claims").exists()
    # twin: its cycle-40 folder exists, but the twin chain's RL has not ended (no rl.DONE / rl.COST_CAP): refused
    runs = tmp_path / "base" / "agm_data" / "runs"
    (runs / "reader_rl_scratch_L3act_beh_paper" / "adapters" / "cycle_040").mkdir(parents=True)
    (runs / "reader_scratch_L3act_beh").mkdir(parents=True)
    (runs / "reader_scratch_L3act_beh" / "RUN.json").write_text("{}")
    (tmp_path / "twin_m").mkdir()
    r = run(["twin", "e4"], dict(env, TWIN_M=str(tmp_path / "twin_m")))
    assert r.returncode == 1 and "RL of its chain has not ended" in r.stdout
    assert not (tmp_path / "root").exists() and not (tmp_path / "claims").exists()


# ---------------------------------------------------------------- the runner on stubs
FAKE = r'''"""Stub of the programs the suite runner starts: logs SUITE_STAGE | CUDA_VISIBLE_DEVICES | CUDA_DEVICE_ORDER |
args to FAKE_LOG and writes the files the runner looks for (BUILD.json, readings, judged files, ledger rows)."""
import glob, json, os, sys, time
from pathlib import Path

me, args = Path(__file__).name, sys.argv[1:]
if me == "fake_smi.py":  # GPU 0 shows the most free memory: the runner must still never pick it
    print("0, 183000\n1, 182000\n2, 182400\n3, 181900")
    sys.exit(0)

def opt(k, default=None):
    return args[args.index(k) + 1] if k in args else default

stage = os.environ.get("SUITE_STAGE", "UNSET")
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write("|".join([stage, os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"), os.environ.get("CUDA_DEVICE_ORDER", "UNSET"),
                      me + " " + " ".join(args)]) + "\n")
print(f"stage {stage} running", flush=True)
if stage in os.environ.get("FAKE_FAIL", "").split(","):
    sys.exit(1)
if stage == os.environ.get("FAKE_HANG"):
    time.sleep(600)
cost = float(os.environ.get("FAKE_COST", "0.1"))
dry = "--dry-run" in args

def ledger(kind):
    if dry:
        return
    p = Path(os.environ["TRUTH_LEDGER" if kind == "truth" else "DOLPHIN_LEDGER"])
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a") as f:
        if kind == "truth":
            if new:
                f.write("time_start\ttime_end\tmode\tjudge\trequests\tcalls\tfailed_final\tinput_tokens\toutput_tokens\tcost_usd\n")
            f.write(f"t\tt\tbatch\ttruth\t1\t1\t0\t1\t1\t{cost}\n")
        else:
            if new:
                f.write("time_start\ttime_end\tmode\tanswers\tdistinct_pending\tcalls\tfailed_first_pass\tretried\t"
                        "failed_final\tinput_tokens\toutput_tokens\tcost_usd\tunparsed\n")
            f.write(f"t\tt\tfull\t1\t1\t1\t0\t0\t0\t1\t1\t{cost}\t0\n")
    if stage == os.environ.get("FAKE_SLOW_COST"):  # the ledger row lands, then the judge keeps running
        time.sleep(60)

def write_rows(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))

def reading(n_steps, windows, samples):
    return [dict(step=s, inputs={w: dict(texts=[f"a{s}{w}{i}" for i in range(samples)]) for w in windows})
            for s in range(n_steps)]

def judged(src, dst, key=None):
    rows = [json.loads(x) for x in open(src)]
    n = sum(len(rec["texts"]) for r in rows for w, rec in r["inputs"].items() if key is None or w == key)
    if stage == os.environ.get("FAKE_SHORT"):
        n -= 1
    write_rows(dst, [dict(i=i, verdict="NO") for i in range(n)])

if me == "fake_nrs.py":  # new_readers_suite.py: build, slot-extract, summary, plot
    if args[0] == "build":
        root, exp, reader = Path(opt("--out")), opt("--exp"), opt("--reader")
        d = root / exp / reader
        d.mkdir(parents=True, exist_ok=True)
        pending = reader == "slot" and (exp != "e4" or os.environ.get("FAKE_E4_PENDING")) and not list((d / "slots").glob("*"))
        if pending:
            (d / "docs.jsonl").write_text("{}\n")
            (d / "BUILD_PENDING.json").write_text("{}")
        else:
            (d / "inputs").mkdir(parents=True, exist_ok=True)
            (d / "inputs" / "BUILD.json").write_text("{}")
    elif args[0] == "slot-extract":
        out = Path(opt("--out-dir"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "slots_g.npz").write_text("x")
    else:
        Path(opt("--out")).mkdir(parents=True, exist_ok=True)
        (Path(opt("--out")) / f"{args[0]}.json").write_text("{}")
elif me == "fake_nrsr.py":  # new_readers_suite_read.py: read, yesno, judge (e5)
    root, exp, reader = Path(opt("--root")), opt("--exp"), opt("--reader")
    d, samples, name = root / exp / reader, int(opt("--samples", "1")), "batch_nrs_" + reader
    if args[0] == "read" and exp in ("e2", "e3"):
        students = ["dolphin", "clean"] if exp == "e2" else ["dolphin"]
        windows = ["w1000"] if exp == "e2" else ["real_x", "f1_x"]
        for s in students:
            write_rows(d / s / "e2" / f"{name}_preference.jsonl", reading(2, windows, samples))
    elif args[0] == "read" and exp == "e4":
        for k in ("doc", "showing", "nonshowing"):
            write_rows(d / "open" / f"readings_{k}.jsonl", reading(3, ["bundle"], samples))
    elif args[0] == "read" and exp == "e5":
        write_rows(d / "base" / "readings_grad.jsonl", reading(2, ["bundle"], samples))
    elif args[0] == "judge":
        if not dry:
            for f in glob.glob(str(d / "*" / "readings_*.jsonl")):
                if "_judge_" not in f:
                    judged(f, Path(f[:-6] + "_judge_truth.jsonl"))
        ledger("truth")
elif me == "fake_rp.py":  # rare_perdoc.py judge / summary
    out = Path(opt("--out"))
    if args[0] == "judge":
        if not dry:
            for k in ("doc", "showing", "nonshowing"):
                judged(out / f"readings_{k}.jsonl", out / f"readings_{k}_judge_truth.jsonl")
        ledger("truth")
    else:
        (out / "summary.json").write_text("{}")
elif me == "fake_jd.py":  # judge_dolphin_answers.py
    reader, kind, keys = opt("--readers"), opt("--inputs"), opt("--key").split(",")
    if not dry:
        for run in opt("--runs").split(","):
            src = Path(run) / "e2" / f"{kind}_{reader}_preference.jsonl"
            first = json.loads(src.open().readline())
            for k in keys:
                if k in first["inputs"]:
                    judged(src, Path(str(src)[:-6] + f"_{k}_judge_dolphin.jsonl"), k)
    ledger("dolphin")
# fake_ry.py (rare_yesno.py read / summary): nothing to write
'''
GPU_STAGES = ("read_e2", "read_e3", "read_e4", "read_e5", "yesno_e3", "yesno_e4", "yesno_e5", "extract_e3",
              "extract_e4", "extract_e5")


def stub_env(tmp_path: Path, **extra) -> dict:
    """Stubs are Python files run by the real interpreter (the scratch folder may be mounted noexec)."""
    fk, base = tmp_path / "fake", tmp_path / "base"
    for d in (fk, base / "agm_data", tmp_path / "logs", tmp_path / "adapter"):
        d.mkdir(parents=True, exist_ok=True)
    (tmp_path / "RUN.json").write_text("{}")
    for n in ("fake_smi.py", "fake_nrs.py", "fake_nrsr.py", "fake_ry.py", "fake_rp.py", "fake_jd.py"):
        (fk / n).write_text(FAKE)
    return dict(os.environ, PY=sys.executable, NRS=str(fk / "fake_nrs.py"), NRSR=str(fk / "fake_nrsr.py"),
                RY=str(fk / "fake_ry.py"), RP=str(fk / "fake_rp.py"), JD=str(fk / "fake_jd.py"),
                GPU_QUERY=f"{sys.executable} {fk / 'fake_smi.py'}", FAKE_LOG=str(tmp_path / "calls.log"),
                IIF_BASE=str(base), ROOT=str(tmp_path / "root"), CLAIM_DIR=str(tmp_path / "claims"),
                LOGD=str(tmp_path / "logs"), ADAPTER=str(tmp_path / "adapter"), RUN_JSON=str(tmp_path / "RUN.json"),
                TRUTH_LEDGER=str(tmp_path / "ledgers" / "truth.tsv"), DOLPHIN_LEDGER=str(tmp_path / "ledgers" / "pref.tsv"),
                FREE_GAP_S="0", WAIT_STEP_S="1", POLL_S="1", COST_CHECK_S="1", DISK_NEED_GB="0", **extra)


def calls(tmp_path: Path) -> list:
    p = tmp_path / "calls.log"
    return [ln.split("|", 3) for ln in p.read_text().splitlines()] if p.exists() else []


def spend_rows(tmp_path: Path) -> list:
    p = tmp_path / "root" / "judge_spend.tsv"
    return [ln.split("\t") for ln in p.read_text().splitlines()[1:]] if p.exists() else []


def cap_of(c) -> float:
    """The cap a judge call got: --cap-usd, or judge_dolphin's --max-cost (the preference ledger was 0 before)."""
    a = c[3].split()
    return float(a[a.index("--cap-usd") + 1]) if "--cap-usd" in a else float(a[a.index("--max-cost") + 1])


def test_runner_stub_gpu_claims_markers_caps_rerun(tmp_path):
    """grad3 e2 e4 on stubs: every stage DONE; GPU stages on GPUs 1-3 only (GPU 0 shows the most free memory), with
    --gpu = CUDA_VISIBLE_DEVICES and PCI_BUS_ID order; CPU / API stages see no GPU; every claim released; the two judges
    run one after the other with caps 15 and 15 - the first one's spend; the spend file holds both; judged rows = answers
    checked; a rerun skips every DONE stage and runs summary and plot again."""
    env = stub_env(tmp_path)
    M = tmp_path / "root" / "markers" / "grad3"
    r = run(["grad3", "e2", "e4"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    want = ["build_e2", "read_e2", "judge_e2", "build_e4", "read_e4", "judge_e4", "opensum_e4", "yesno_e4", "ynsum_e4",
            "summary", "plot"]
    for s in want:
        assert (M / f"{s}.DONE").exists(), s
    assert (M / "chain.DONE").read_text().split()[-2:] == ["e2", "e4"]
    first = calls(tmp_path)
    assert sorted(c[0] for c in first) == sorted(want)
    for stage, cvd, order, args in first:
        assert order == "PCI_BUS_ID", (stage, order)
        if stage in GPU_STAGES:
            assert cvd in ("1", "2", "3"), (stage, cvd)
            assert args.split("--gpu ")[1].split()[0] == cvd
            assert "--adapter " + str(tmp_path / "adapter") in args or "--reader-adapter " + str(tmp_path / "adapter") in args
        else:
            assert cvd == "", (stage, cvd)
    y = next(c for c in first if c[0] == "yesno_e4")[3]
    assert y.startswith("fake_ry.py read") and "--input grad" in y
    assert "--run-dir " + str(tmp_path / "root" / "e4" / "grad3" / "yesno") in y
    jd = next(c for c in first if c[0] == "judge_e2")[3]
    runs = jd.split("--runs ")[1].split()[0].split(",")
    assert sorted(Path(x).name for x in runs) == ["clean", "dolphin"] and "--key w1000" in jd
    caps = sorted(cap_of(c) for c in first if c[0].startswith("judge_"))
    assert caps == [14.9, 15.0], caps  # the second judge gets 15 - 0.1
    sp = spend_rows(tmp_path)
    assert sorted(x[2] for x in sp) == ["judge_e2", "judge_e4"] and all(abs(float(x[7]) - 0.1) < 1e-9 for x in sp)
    claims = tmp_path / "claims"
    assert not list(claims.glob("gpu*.claim")) and len(list(claims.glob("gpu*.claim.released_*"))) == 3
    assert all((M / f"{s}.pid").exists() for s in ("read_e2", "read_e4", "yesno_e4"))
    assert "ok " in (tmp_path / "logs" / "20261029_suite_grad3_judge_e2.out").read_text()
    # rerun: DONE stages skipped, summary and plot always run
    r = run(["grad3", "e2", "e4"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(c[0] for c in calls(tmp_path)[len(first):]) == ["plot", "summary"]
    assert "DONE earlier, skipped" in r.stdout


def test_runner_stub_twin_and_slot(tmp_path):
    """twin e4: rare_yesno with --input act and the act dir. slot e3 e4: docs_e3 leaves the build pending, so
    extract_e3 runs on a GPU and build_e3 assembles; docs_e4 assembles at once, so extract_e4 and build_e4 are not
    needed; the slot yes/no of e4 goes to the read driver, no rare_yesno summary stage; both chains spend from the
    same spend file."""
    env = stub_env(tmp_path, YESNO_ACT_DIR=str(tmp_path / "act"))
    r = run(["twin", "e4"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    y = next(c for c in calls(tmp_path) if c[0] == "yesno_e4")[3]
    assert "--input act" in y and "--act-dir " + str(tmp_path / "act") in y and "--reader-name twin" in y
    n0 = len(calls(tmp_path))
    r = run(["slot", "e3", "e4"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    new = calls(tmp_path)[n0:]
    stages = [c[0] for c in new]
    assert stages.count("docs_e3") == 1 and stages.count("extract_e3") == 1 and stages.count("build_e3") == 1
    assert "extract_e4" not in stages and "build_e4" not in stages and "ynsum_e4" not in stages
    M = tmp_path / "root" / "markers" / "slot"
    assert "not needed" in (M / "extract_e4.DONE").read_text() and "not needed" in (M / "build_e4.DONE").read_text()
    x = next(c for c in new if c[0] == "extract_e3")
    assert x[1] in ("1", "2", "3") and "slot-extract --docs " + str(tmp_path / "root" / "e3" / "slot" / "docs.jsonl") in x[3]
    assert next(c for c in new if c[0] == "build_e3")[1] == ""
    assert (tmp_path / "root" / "e3" / "slot" / "inputs" / "BUILD.json").exists()
    y = next(c for c in new if c[0] == "yesno_e4")
    assert y[3].startswith("fake_nrsr.py yesno --exp e4 --reader slot")
    assert sorted((x[1], x[2]) for x in spend_rows(tmp_path)) == [("slot", "judge_e3"), ("slot", "judge_e4"),
                                                                 ("twin", "judge_e4")]


def test_runner_stub_failed_extraction_stops_the_branch(tmp_path):
    """slot e4 whose docs stage leaves the build pending and whose extraction fails: build_e4 never starts."""
    env = stub_env(tmp_path, FAKE_E4_PENDING="1", FAKE_FAIL="extract_e4")
    M = tmp_path / "root" / "markers" / "slot"
    r = run(["slot", "e4"], env)
    assert r.returncode == 1 and (M / "extract_e4.FAILED").exists() and not (M / "build_e4.START").exists()


def test_runner_stub_suite_cap_stops_a_judge(tmp_path):
    """SUITE_CAP_USD 0.12: the first judge spends 0.10, the second gets 0.02 < 0.05 and is not started (FAILED cap);
    the chain fails and summary does not run."""
    env = stub_env(tmp_path, SUITE_CAP_USD="0.12")
    M = tmp_path / "root" / "markers" / "grad3"
    r = run(["grad3", "e2", "e4"], env)
    assert r.returncode == 1, r.stdout + r.stderr
    judges = [c for c in calls(tmp_path) if c[0].startswith("judge_")]
    assert len(judges) == 1
    other = "judge_e4" if judges[0][0] == "judge_e2" else "judge_e2"
    assert "cap:" in (M / f"{other}.FAILED").read_text() and not (M / f"{other}.START").exists()
    assert (M / "chain.FAILED").exists() and not (M / "summary.START").exists()
    assert len(spend_rows(tmp_path)) == 1


def test_runner_stub_failed_rerun_hang_rows_and_cost_watchdog(tmp_path):
    """A failed read stops its branch (the other branch finishes; chain.FAILED); a rerun sets the FAILED marker aside
    and finishes. A judge whose judged rows differ from the answers fails; a stage whose log is quiet past HANG_S is
    killed; a judge whose ledger increase passes its cap is stopped by the watchdog (COST_CAP)."""
    env = stub_env(tmp_path, FAKE_FAIL="read_e2")
    M = tmp_path / "root" / "markers" / "grad3"
    r = run(["grad3", "e2", "e4"], env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert (M / "read_e2.FAILED").exists() and not (M / "judge_e2.START").exists() and (M / "judge_e4.DONE").exists()
    assert (M / "chain.FAILED").exists() and not (M / "summary.START").exists()
    r = run(["grad3", "e2", "e4"], dict(env, FAKE_FAIL=""))
    assert r.returncode == 0, r.stdout + r.stderr
    assert list(M.glob("read_e2.FAILED.before_*")) and list(M.glob("chain.FAILED.before_*")) and (M / "chain.DONE").exists()
    # judged rows != answers
    env2 = stub_env(tmp_path / "b", FAKE_SHORT="judge_e3")
    M2 = tmp_path / "b" / "root" / "markers" / "grad3"
    r = run(["grad3", "e3"], env2)
    assert r.returncode == 1 and "judged rows differ" in (M2 / "judge_e3.FAILED").read_text()
    assert "BAD" in (tmp_path / "b" / "logs" / "20261029_suite_grad3_judge_e3.out").read_text()
    assert (M2 / "yesno_e3.DONE").exists()  # the yes/no side of the branch is not stopped by the judge
    # hang guard
    env3 = stub_env(tmp_path / "c", FAKE_HANG="build_e5", HANG_S="3")
    M3 = tmp_path / "c" / "root" / "markers" / "grad3"
    r = run(["grad3", "e5"], env3)
    assert r.returncode == 1 and "quiet 1" in (M3 / "build_e5.FAILED").read_text()
    pid = int((M3 / "build_e5.pid").read_text().split()[0])
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1][:3]
    # cost watchdog: the stub judge writes a 20 USD ledger row and keeps running; cap 15
    env4 = stub_env(tmp_path / "d", FAKE_COST="20", FAKE_SLOW_COST="judge_e4")
    M4 = tmp_path / "d" / "root" / "markers" / "grad3"
    r = run(["grad3", "e4"], env4)
    assert r.returncode == 1, r.stdout + r.stderr
    assert float((M4 / "judge_e4.COST_CAP").read_text()) == 20.0 and "cost_cap 1" in (M4 / "judge_e4.FAILED").read_text()
    sp = spend_rows(tmp_path / "d")
    assert len(sp) == 1 and float(sp[0][7]) == 20.0 and float(sp[0][8]) == 15.0


def test_runner_stub_foreign_claims_and_judge_dry(tmp_path):
    """A live claim of another chain (the twin's) keeps its GPU out and is left untouched; a claim with a dead pid does
    not. JUDGE_DRY=1: judges get --dry-run and end with .DRY (not DONE), opensum_e4 is skipped (.DRY), no spend row."""
    claims = tmp_path / "claims"
    claims.mkdir()
    sleeper = subprocess.Popen(["sleep", "300"])
    try:
        host = socket.gethostname()
        live = f"eval_twin_read {sleeper.pid} {host} 2026-10-29 05:00:00 free_mib=182000\n"
        (claims / "gpu3.claim").write_text(live)
        dead = subprocess.Popen(["true"])
        dead.wait()
        (claims / "gpu2.claim").write_text(f"old_stage {dead.pid} {host} 2026-10-29 04:00:00 free_mib=1\n")
        env = stub_env(tmp_path, JUDGE_DRY="1")
        M = tmp_path / "root" / "markers" / "grad3"
        r = run(["grad3", "e2", "e4"], env)
        assert r.returncode == 0, r.stdout + r.stderr
        gpus = {c[1] for c in calls(tmp_path) if c[0] in GPU_STAGES}
        assert gpus <= {"1", "2"} and "3" not in gpus, gpus
        assert (claims / "gpu3.claim").read_text() == live
        judges = [c for c in calls(tmp_path) if c[0].startswith("judge_")]
        assert len(judges) == 2 and all("--dry-run" in c[3] for c in judges)
        assert (M / "judge_e2.DRY").exists() and not (M / "judge_e2.DONE").exists() and not spend_rows(tmp_path)
        assert (M / "opensum_e4.DRY").exists() and not (M / "opensum_e4.DONE").exists()
        assert "opensum_e4" not in [c[0] for c in calls(tmp_path)]  # skipped: rare_perdoc summary needs judged files
    finally:
        sleeper.send_signal(signal.SIGTERM)
        sleeper.wait()

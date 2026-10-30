"""scripts/beh_reader/harm_generalisation.sh: syntax, the --dry stage list (plan keys and SELECT.json keys), the
refusals, and the real runner on stub programs (harm_generalisation.py, harm_generalisation_train.py and nvidia-smi
replaced by one Python stub): GPU placement with claim files (other chains' live claims skipped, dead ones reused,
memory rule for a second stage on a GPU, at most four fine-tunes, never two on one GPU), CUDA_VISIBLE_DEVICES, the
dependency order, resume (DONE stages skipped), a FAILED stage stops the chain, the hang guard and the disk check.
CPU only: no GPU, no model, no API, nothing written outside the pytest basetemp.

  cd <scratch> && python -m pytest -q -p no:cacheprovider <repo>/tests/test_harm_generalisation_chain.py \\
      --basetemp <scratch>/harm_gen_chain_test
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SH = REPO / "scripts" / "beh_reader" / "harm_generalisation.sh"
PLAN_KEYS = ["harmful_pet", "harmful_travel", "harmful_food", "harmful_realestate", "benign_vet",
             "benign_travel", "quirk_cats", "quirk_saturn", "rollouts_etymology", "rollouts_feedback"]


def order(keys):
    return (["select", "prompts"] + [f"finetune_{k}" for k in keys] + ["generate_base"]
            + [s for k in keys for s in (f"check_{k}", f"generate_{k}")] + ["judge", "summary", "plot"])


def run(args, env, timeout=300):
    return subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, env=env, timeout=timeout)


def dry_stages(out: str) -> list[str]:
    return [ln.split()[1] for ln in out.splitlines() if ln.startswith("[dry] ")]


def cmd_of(out: str, stage: str) -> str:
    lines = out.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(f"[dry] {stage} ("))
    return lines[i + 1]


# ---------------------------------------------------------------- syntax, --dry, refusals
def test_bash_syntax():
    r = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def base_env(tmp_path: Path, **extra) -> dict:
    return dict(os.environ, IIF_BASE=str(tmp_path / "base"), OUT=str(tmp_path / "out"), CLAIM_DIR=str(tmp_path / "claims"),
                LOGD=str(tmp_path / "logs"), **extra)


def test_dry_lists_stages(tmp_path):
    env = base_env(tmp_path)
    r = run(["--dry"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert dry_stages(r.stdout) == order(PLAN_KEYS)
    out = tmp_path / "out"
    assert "finetune_harmful_pet (pending; about 2 min; GPU, 80 GiB free; after select prompts)" in r.stdout
    assert "generate_base (pending; about 16 min; GPU, 40 GiB free; after prompts)" in r.stdout
    assert "check_quirk_cats (pending; about 2 min; API; after finetune_quirk_cats)" in r.stdout
    assert "generate_rollouts_feedback (pending; about 3 min; GPU, 40 GiB free; after check_rollouts_feedback)" in r.stdout
    assert "generate_benign_vet (pending; about 7 min;" in r.stdout
    assert "judge (pending; about 40 min; API; after generate_base generate_harmful_pet" in r.stdout
    c = cmd_of(r.stdout, "finetune_benign_travel")
    assert (f"harm_generalisation_train.py finetune --out {out} --behaviour benign_travel --gpu <gpu> --min-free-gib 0 "
            "--no-check --batch 64") in c
    assert f"harm_generalisation_train.py generate --out {out} --model base --gpu <gpu> --min-free-gib 0 --batch 64" \
        in cmd_of(r.stdout, "generate_base")
    assert cmd_of(r.stdout, "check_harmful_food").endswith(f"check --out {out} --behaviour harmful_food")
    assert cmd_of(r.stdout, "judge").endswith(f"harm_generalisation.py judge --out {out} --model all")
    assert cmd_of(r.stdout, "prompts").endswith(f"harm_generalisation.py prompts --out {out}")
    assert f"harm_generalisation.py plot --out {out} --fig {out / 'harm_generalisation_smoke.png'}" in cmd_of(r.stdout, "plot")
    assert "--gpu" not in cmd_of(r.stdout, "select") and "--gpu" not in cmd_of(r.stdout, "judge")
    assert not out.exists() and not (tmp_path / "claims").exists() and not (tmp_path / "logs").exists()
    # BEHS overrides; a DONE marker shows
    (out / "markers").mkdir(parents=True)
    (out / "markers" / "select.DONE").write_text("x")
    r = run(["--dry"], dict(env, BEHS="A B"))
    assert dry_stages(r.stdout) == order(["A", "B"]) and "select (DONE;" in r.stdout
    # smoke: the stand-in judge everywhere, smoke args passed
    r = run(["--dry"], dict(env, FT_ARGS="--max-epochs 2", GEN_ARGS="--limit-prompts 3"))
    assert "FAKE_JUDGE=1" in r.stdout and cmd_of(r.stdout, "check_harmful_pet").endswith("--fake-judge")
    assert cmd_of(r.stdout, "judge").endswith("--model all --fake-judge")
    assert cmd_of(r.stdout, "prompts").endswith("--fake-judge")
    assert cmd_of(r.stdout, "finetune_harmful_pet").endswith("--max-epochs 2")
    assert cmd_of(r.stdout, "generate_quirk_cats").endswith("--limit-prompts 3")


def test_dry_reads_select_keys(tmp_path):
    fk = tmp_path / "fake"
    fk.mkdir()
    (fk / "fake_hgt.py").write_text(FAKE)
    out = tmp_path / "out"
    out.mkdir()
    (out / "SELECT.json").write_text(json.dumps(dict(behaviours={"X1": {}, "Y2": {}})))
    r = run(["--dry"], base_env(tmp_path, PY=sys.executable, HGT=str(fk / "fake_hgt.py")))
    assert r.returncode == 0, r.stdout + r.stderr
    assert dry_stages(r.stdout) == order(["X1", "Y2"]) and f"behaviours ({out / 'SELECT.json'}): X1 Y2" in r.stdout


def test_refusals(tmp_path):
    env = base_env(tmp_path)
    r = run(["--nope"], env)
    assert r.returncode == 2 and "unknown argument" in r.stdout
    r = run(["--dry"], dict(env, GPUS="0 x"))
    assert r.returncode == 2 and "not a GPU index" in r.stdout
    real = {k: v for k, v in env.items() if k != "OUT"}
    for extra in (dict(FAKE_JUDGE="1"), dict(FT_ARGS="--max-epochs 2"), dict(GEN_ARGS="--limit-prompts 2")):
        r = run(["--dry"], dict(real, **extra))
        assert r.returncode == 2 and "smoke options" in r.stdout, extra
    r = run(["--dry"], dict(env, BEHS="base"))
    assert r.returncode == 2
    r = run([], dict(env, HG=str(tmp_path / "missing.py")))  # a real launch needs both programs
    assert r.returncode == 2 and "missing program" in r.stdout and not (tmp_path / "out").exists()


# ---------------------------------------------------------------- the runner on stubs
FAKE = r'''"""Stub of harm_generalisation.py, harm_generalisation_train.py and nvidia-smi: logs start and end lines to
FAKE_LOG (S|time|stage|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|args, E|time|stage) and writes the files the runner
checks for."""
import json, os, sys, time
from pathlib import Path

me, args = Path(__file__).name, sys.argv[1:]
if me == "fake_smi.py":
    for kv in os.environ.get("FAKE_FREE", "0:182000,1:182000,2:182000,3:182000").split(","):
        g, f = kv.split(":")
        print(f"{g}, {f}")
    sys.exit(0)

def opt(k, default=None):
    return args[args.index(k) + 1] if k in args else default

out = Path(opt("--out", "."))
if args[0] == "keys":
    for k in json.loads((out / "SELECT.json").read_text())["behaviours"]:
        print(k)
    sys.exit(0)
stage = os.environ.get("HG_STAGE", "UNSET")
def note(line):
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(line + "\n")
note("|".join(["S", f"{time.time():.3f}", stage, os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
               os.environ.get("CUDA_DEVICE_ORDER", "UNSET"), me + " " + " ".join(args)]))
print(f"stage {stage} running", flush=True)
if stage in os.environ.get("FAKE_FAIL", "").split(","):
    sys.exit(1)
if stage == os.environ.get("FAKE_HANG"):
    time.sleep(600)
dur = dict(finetune=float(os.environ.get("FAKE_FT_S", "2")), generate=float(os.environ.get("FAKE_GEN_S", "3")))
time.sleep(dur.get(args[0], 0.2))
def wj(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
cmd = args[0]
if me == "fake_hg.py":
    if cmd == "select":
        wj(out / "SELECT.json", dict(behaviours={k: {} for k in os.environ.get("FAKE_KEYS", "A B C D E F").split()}))
    elif cmd == "prompts":
        wj(out / "prompts.jsonl", {})
    else:
        wj(out / f"{cmd}.json", {})
elif cmd == "finetune":
    wj(out / opt("--behaviour") / "FINETUNE.json", {})
elif cmd == "check":
    wj(out / opt("--behaviour") / "BEHAVIOUR.json", {})
elif cmd == "generate":
    wj(out / opt("--model") / "gens.jsonl", {})
note(f"E|{time.time():.3f}|{stage}")
'''


def stub_env(tmp_path: Path, **extra) -> dict:
    fk = tmp_path / "fake"
    fk.mkdir(exist_ok=True)
    for n in ("fake_smi.py", "fake_hg.py", "fake_hgt.py"):
        (fk / n).write_text(FAKE)
    (tmp_path / "base").mkdir(exist_ok=True)
    env = dict(base_env(tmp_path), PY=sys.executable, HG=str(fk / "fake_hg.py"), HGT=str(fk / "fake_hgt.py"),
               GPU_QUERY=f"{sys.executable} {fk / 'fake_smi.py'}", FAKE_LOG=str(tmp_path / "calls.log"),
               RUNNER_LOG="-", FREE_GAP_S="0", POLL_S="0.2", SCHED_S="1", RESERVE_S="100000", DISK_MIN_GB="0")
    env.update(extra)
    return env


def calls(tmp_path: Path) -> dict:
    """{stage: dict(start, end, gpu, order, args)} of the stub log (the last attempt of each stage)."""
    out = {}
    p = tmp_path / "calls.log"
    for ln in (p.read_text().splitlines() if p.exists() else []):
        f = ln.split("|")
        if f[0] == "S":
            out[f[2]] = dict(start=float(f[1]), end=None, gpu=f[3], order=f[4], args=f[5])
        else:
            out[f[2]]["end"] = float(f[1])
    return out


def overlaps(c: dict, a: str, b: str) -> bool:
    return c[a]["start"] < c[b]["end"] and c[b]["start"] < c[a]["end"]


def max_at_once(c: dict, stages: list, same_gpu: bool = False) -> int:
    """Most stages running at one moment (per GPU when same_gpu)."""
    best = 0
    for a in stages:
        t = c[a]["start"]
        n = sum(1 for b in stages if c[b]["start"] <= t < c[b]["end"] and (not same_gpu or c[b]["gpu"] == c[a]["gpu"]))
        best = max(best, n)
    return best


def test_runner_gpu_placement_claims_order(tmp_path):
    """Six behaviours on stubs. GPU 1 has 60 GiB free (never a fine-tune, a generation only alone), GPU 2 holds a live
    claim of another chain (never used, claim untouched), GPU 3 has 100 GiB (one stage at a time: no room for a second),
    GPU 0 182 GiB (fine-tune plus a generation). GPU 3's old claim of a dead pid is reused."""
    claims = tmp_path / "claims"
    claims.mkdir()
    other = subprocess.Popen(["sleep", "300"])
    try:
        host = socket.gethostname()
        (claims / "gpu2.claim").write_text(f"cross_read {other.pid} {host} 2026-10-29 13:00:00\n")
        (claims / "gpu3.claim").write_text(f"old_stage 999999 {host} 2026-10-29 10:00:00\n")
        env = stub_env(tmp_path, FAKE_FREE="0:182000,1:60000,2:182000,3:100000")
        t0 = time.time()
        r = run([], env, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        c = calls(tmp_path)
        keys = list("ABCDEF")
        assert set(c) == set(order(keys)), sorted(c)
        M = tmp_path / "out" / "markers"
        assert all((M / f"{s}.DONE").exists() for s in order(keys)) and (M / "chain.DONE").exists()
        # behaviours came from SELECT.json, not the plan list
        assert "behaviours from SELECT.json: A B C D E F" in r.stdout
        gpu_st = [s for s in c if s.startswith(("finetune_", "generate_"))]
        for s in c:
            if s in gpu_st:
                g = c[s]["gpu"]
                assert g in ("0", "1", "3") and f"--gpu {g} --min-free-gib 0" in c[s]["args"], (s, c[s])
                assert c[s]["order"] == "PCI_BUS_ID"
            else:
                assert c[s]["gpu"] == "", (s, c[s])  # CPU and API stages see no GPU
        assert all(c[s]["gpu"] != "1" for s in gpu_st if s.startswith("finetune_"))  # 60 GiB < 80
        for i, a in enumerate(gpu_st):  # GPU 1 and 3: one stage at a time; GPU 0: at most two, never two fine-tunes
            for b in gpu_st[i + 1:]:
                if c[a]["gpu"] == c[b]["gpu"] and overlaps(c, a, b):
                    assert c[a]["gpu"] == "0", (a, b, c[a], c[b])
                    assert not (a.startswith("finetune_") and b.startswith("finetune_")), (a, b)
        assert max_at_once(c, gpu_st, same_gpu=True) <= 2
        # dependencies
        for k in keys:
            assert c[f"check_{k}"]["start"] >= c[f"finetune_{k}"]["end"]
            assert c[f"generate_{k}"]["start"] >= c[f"check_{k}"]["end"]
            assert c[f"finetune_{k}"]["start"] >= c["prompts"]["end"]
        assert c["judge"]["start"] >= max(c[s]["end"] for s in c if s.startswith("generate_"))
        assert c["summary"]["start"] >= c["judge"]["end"] and c["plot"]["start"] >= c["summary"]["end"]
        assert c["generate_base"]["start"] >= c["prompts"]["end"]
        # claims: ours released with the stage name, the other chain's untouched, the dead one replaced
        assert (claims / "gpu2.claim").read_text().startswith("cross_read ")
        assert not (claims / "gpu0.claim").exists() and not (claims / "gpu3.claim").exists()
        rel = sorted(p.name for p in claims.glob("gpu*.claim.released_*"))
        assert rel and all("_harm_gen_" in n for n in rel), rel
        assert any(n.startswith("gpu3.claim.released_") for n in rel)
        first = (claims / rel[0]).read_text().split()
        assert first[0].startswith("harm_gen_") and first[2] == host and first[5].startswith("free_mib=")
        assert time.time() - t0 < 300
    finally:
        other.kill()
        other.wait()


def test_runner_four_finetunes_at_most(tmp_path):
    """Eight behaviours, four empty GPUs: at most four fine-tunes at once; generations join (two stages per GPU)."""
    env = stub_env(tmp_path, FAKE_KEYS="A B C D E F G H", FAKE_FT_S="2", FAKE_GEN_S="4")
    r = run([], env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    c = calls(tmp_path)
    ft = [s for s in c if s.startswith("finetune_")]
    assert len(ft) == 8
    assert max_at_once(c, ft) == 4  # four fine-tunes at once, never more
    assert max_at_once(c, [x for x in c if x.startswith(("finetune_", "generate_"))], same_gpu=True) <= 2
    gen = [s for s in c if s.startswith("generate_")]
    assert any(overlaps(c, a, b) and c[a]["gpu"] == c[b]["gpu"] for a in gen for b in c if b != a)  # sharing happened


def test_runner_resume_skips_done(tmp_path):
    env = stub_env(tmp_path, FAKE_KEYS="A B")
    out = tmp_path / "out"
    M = out / "markers"
    M.mkdir(parents=True)
    # select, prompts and finetune_A done earlier (their outputs exist)
    (out / "SELECT.json").write_text(json.dumps(dict(behaviours={"A": {}, "B": {}})))
    (out / "prompts.jsonl").write_text("{}")
    for s in ("select", "prompts", "finetune_A"):
        (M / f"{s}.DONE").write_text("x")
    (M / "finetune_B.FAILED").write_text("exit 1")  # a FAILED stage of an earlier run runs again
    r = run([], env)
    assert r.returncode == 0, r.stdout + r.stderr
    c = calls(tmp_path)
    assert not {"select", "prompts", "finetune_A"} & set(c) and "finetune_B" in c and "check_A" in c
    assert list(M.glob("finetune_B.FAILED.before_*")) and (M / "finetune_B.DONE").exists()
    (tmp_path / "calls.log").unlink()
    r = run([], env)  # everything DONE: nothing runs
    assert r.returncode == 0 and not (tmp_path / "calls.log").exists() and "all stages done" in r.stdout


def test_runner_failed_stops(tmp_path):
    env = stub_env(tmp_path, FAKE_KEYS="A B C D E F", FAKE_FAIL="finetune_B", FAKE_FT_S="1")
    r = run([], env)
    assert r.returncode == 1, r.stdout + r.stderr
    M = tmp_path / "out" / "markers"
    assert (M / "finetune_B.FAILED").read_text().split("exit ")[1].startswith("1")
    assert (M / "chain.FAILED").exists() and not (M / "chain.DONE").exists()
    c = calls(tmp_path)
    fail_t = c["finetune_B"]["start"]
    # nothing starts after the failure was seen; the stages running then finished
    late = [s for s in c if c[s]["start"] > fail_t + 1.5]
    assert not late, late
    assert all(c[s]["end"] is not None for s in c if s != "finetune_B")
    assert not {"judge", "summary", "plot", "check_B", "generate_B"} & set(c)
    assert "no new stage starts" in r.stdout


def test_runner_hang_guard(tmp_path):
    env = stub_env(tmp_path, FAKE_KEYS="A", FAKE_HANG="generate_base", HANG_S="3")
    t0 = time.time()
    r = run([], env, timeout=300)
    assert r.returncode == 1, r.stdout + r.stderr
    M = tmp_path / "out" / "markers"
    assert "hang" in (M / "generate_base.FAILED").read_text()
    pid = int((M / "generate_base.pid").read_text().split()[0])
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split()[2]
    assert time.time() - t0 < 120
    claims = tmp_path / "claims"
    assert not list(claims.glob("gpu?.claim"))  # the killed stage's claim is released too


def test_runner_disk_check(tmp_path):
    env = stub_env(tmp_path, FAKE_KEYS="A", DISK_MIN_GB="999999999")
    r = run([], env)
    assert r.returncode == 1
    M = tmp_path / "out" / "markers"
    assert "disk" in (M / "finetune_A.FAILED").read_text()
    assert "finetune_A" not in calls(tmp_path)


def test_runner_lock(tmp_path):
    env = stub_env(tmp_path, FAKE_KEYS="A", FAKE_FT_S="4")
    p = subprocess.Popen(["bash", str(SH)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(100):
            if (tmp_path / "out" / "markers" / "runner.pid").exists():
                break
            time.sleep(0.1)
        r = run([], env)
        assert r.returncode == 3 and "another runner" in r.stdout
    finally:
        p.wait(timeout=300)

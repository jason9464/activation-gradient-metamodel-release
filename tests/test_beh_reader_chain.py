"""scripts/beh_reader/chain.sh (docs/notes/20261027_behaviour_reader_plan.md sections 7-13) run for real on a scratch
copy of the repo with a stub interpreter (tests/beh_reader_chain_stub_python.sh), a fake nvidia-smi and a fake
IIF_BASE: stage order, markers and timings, the module command lines (repo root, worktree on PYTHONPATH, one GPU through
CUDA_VISIBLE_DEVICES for the checks, HF_HUB_OFFLINE=0 only for the upload), GPU announcements closed, the gate STOP
(no RL, gate summary committed and pushed to a bare scratch remote with both trailers), the content judge refused by its
calibration guard (soft, the chain goes on) and judged on a rerun with BEH_CONTENT_TRUTH, the soft upload failure,
judge_b in the background beside read_a (collected before c1val; a truth-judge failure is soft) and the soft c1val
failure, the read hang watchdog (a read whose log and preds.jsonl.tmp stop growing gets TERM, one that ignores TERM
gets KILL, one whose partial output keeps growing is left alone), the refusals (failing checks file, unfinished SFT
dir, unknown stage, second chain), and the dry run (writes nothing, lists every stage, exit 4 with missing
prerequisites).

No GPU, no API, no real module runs (the stub exits 97 on any main_run program it does not fake); everything is written
under pytest's tmp_path. Inline python of the chain runs on the real interpreter.
  python -m pytest -q tests/test_beh_reader_chain.py --basetemp <scratch dir>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REAL_BASE = Path(os.environ["IIF_BASE"])
REAL_PY = REAL_BASE / "venvs" / "interpretable_influence" / "bin" / "python"
REAL_RLWT = REAL_BASE / "agm_data" / "worktree_rl_stage"
STUB = Path(__file__).resolve().parent / "beh_reader_chain_stub_python.sh"
STAGES = ["build", "check", "rl_dir", "configs", "sft", "select", "gate", "rl", "export", "read_b", "judge_b", "read_a",
          "c1val", "summary", "upload"]
COPY = ["scripts/beh_reader/chain.sh", "scripts/beh_reader/gpus.sh", "scripts/em_reader/v2_gpus.sh", "main_run/beh_reader/__init__.py",
        "main_run/beh_reader/common.py", "main_run/beh_reader/read_eval.py", "main_run/em_reader/em_common.py",
        "configs/em_v2/rl_scratch_L40_em2_paper.yaml"]
RL_REF = ["rl_scratch_L40_haiku_paper.yaml", "rl_keyword_judge_haiku_paper.yaml", "rl_doc_judge_haiku.yaml"]


def _git(repo: Path, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout


@pytest.fixture()
def world(tmp_path):
    """Scratch repo (git, bare remote), fake IIF_BASE, fake nvidia-smi, env for the chain."""
    repo, base, remote = tmp_path / "repo", tmp_path / "base", tmp_path / "remote.git"
    for rel in COPY:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, repo / rel)
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "push", "-q", "origin", "main")
    a = base / "agm_data"
    (a / "runs" / "blockmix" / "markers").mkdir(parents=True)
    (a / "worktree_em_reader").mkdir()
    for f in RL_REF:
        p = a / "worktree_rl_stage" / "configs" / "reader" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_RLWT / "configs" / "reader" / f, p)
    store = a / "beh_reader" / "behaviour_bundles_v1"
    for s in ("shard0", "shard1"):
        (store / s).mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps(dict(
        total_rows=55172, plan=dict(think_labels="public"), shards=[dict(dir="shard0"), dict(dir="shard1")])))
    smi = tmp_path / "smi.sh"
    smi.write_text("printf '0, 0\\n1, 0\\n'\n")
    env = dict(os.environ, IIF_BASE=str(base), HF_HOME=str(base / "hf_home"), BEH_PY=f"bash {STUB}",
               V2_SMI=f"bash {smi}", V2_LOOK_S="0", STUB_REAL_PY=str(REAL_PY), STUB_LOG=str(tmp_path / "calls.jsonl"),
               STUB_EM2_RL_CFG=str(repo / "configs/em_v2/rl_scratch_L40_em2_paper.yaml"), STAMP="20261027_0000",
               PYTHONDONTWRITEBYTECODE="1", BEH_READ_POLL_S="0.2")
    for k in ("DRY_RUN", "LOGD", "BEH_CONTENT_TRUTH", "EVAL_CAP", "SCORE_CAP", "PYTHONPATH", "BEH_READ_HANG_S",
              "BEH_READ_GRACE_S", "BEH_READ_KILL_S"):
        env.pop(k, None)
    return dict(repo=repo, base=base, remote=remote, env=env, calls=tmp_path / "calls.jsonl",
                m=a / "runs" / "beh_reader" / "markers", bm=a / "runs" / "blockmix" / "markers")


def chain(w, *args, timeout=300, **env):
    e = dict(w["env"], **env)
    return subprocess.run(["bash", str(w["repo"] / "scripts/beh_reader/chain.sh"), *args], env=e, cwd=str(w["repo"]),
                          capture_output=True, text=True, timeout=timeout)


def calls(w):
    p = w["calls"]
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def open_holds(w):
    return [p.name for p in w["bm"].glob("em_reader_v2_beh_reader_*.gpu")
            if not p.with_suffix(".DONE").exists() and not p.with_suffix(".FAILED").exists()]


def test_gate_stop_publishes_and_stops(world):
    w = world
    r = chain(w, STUB_GATE_RC="10")
    assert r.returncode == 10, (r.stdout[-3000:], r.stderr[-2000:])
    m = w["m"]
    for st in STAGES[:6]:
        assert (m / f"{st}.DONE").exists(), st
    assert (m / "gate.STOP").exists() and not (m / "gate.FAILED").exists() and not (m / "gate.DONE").exists()
    assert "stop" in (m / "gate.STOP").read_text() and "mean 0.100" in (m / "gate.STOP").read_text()
    assert (m / "chain.STOPPED").exists() and not (m / "chain.DONE").exists()
    progs = [c["prog"] for c in calls(w)]
    assert "rl_train" not in progs and "export_inputs" not in progs
    # gate summary copied, committed alone with both trailers and pushed
    figs = list((w["repo"] / "docs" / "figures").glob("*_behaviour_reader_gate.md"))
    assert len(figs) == 1 and "Decision: stop" in figs[0].read_text()
    msg = _git(w["remote"], "log", "-1", "--format=%B", "main")
    assert "gate after SFT: stop" in msg and "Co-Authored-By: Claude <noreply@anthropic.com>" in msg
    assert "Claude-Session: <session-url>" in msg
    assert _git(w["remote"], "show", "--name-only", "--format=", "main").split() == [
        f"docs/figures/{figs[0].name}"]
    assert open_holds(w) == []
    # a rerun stops at the gate again without running anything
    n = len(calls(w))
    r = chain(w)
    assert r.returncode == 10 and len(calls(w)) == n, r.stdout[-2000:]


def test_full_chain_soft_failures_and_rerun(world):
    w = world
    t0 = time.time()
    r = chain(w)
    assert r.returncode == 0, (r.stdout[-4000:], r.stderr[-2000:])
    m = w["m"]
    for st in STAGES:
        if st in ("judge_b", "upload"):
            assert (m / f"{st}.FAILED").exists() and not (m / f"{st}.DONE").exists(), st
        else:
            assert (m / f"{st}.DONE").exists(), st
        assert not (m / f"{st}.RUNNING").exists()
    assert "rc=5" in (m / "judge_b.FAILED").read_text()
    assert "content judge refused" in (m / "judge_b.CONTENT_REFUSED").read_text()
    assert "soft failures: judge_b upload" in (m / "chain.DONE").read_text()
    assert (m / "gate.CONTINUE").exists() and (m / "select.WARN").exists() is False
    rows = [x.split("\t") for x in (m / "timings.tsv").read_text().splitlines()]
    assert sorted(x[0] for x in rows) == sorted(STAGES)  # judge_b ends before or after read_a
    assert [x[0] for x in rows if x[0] not in ("judge_b", "read_a")] == [s for s in STAGES if s not in ("judge_b", "read_a")]
    assert (m / "judge_b.pid").exists()
    assert open_holds(w) == []
    cs = calls(w)
    order = []
    for c in cs:
        if not order or order[-1] != c["prog"]:
            order.append(c["prog"])
    assert order[:9] == ["build_reader_dir", "check_reader_dir", "build_rl_dir", "check_rl_dir", "write_configs", "train",
                         "judge_curve", "select_checkpoint", "read_eval"]
    assert order[-3:] == ["read_eval", "eval_summary", "upload"], order
    repo = str(w["repo"])
    wt = str(w["base"] / "agm_data" / "worktree_em_reader")
    for c in cs:
        if c["prog"] in ("build_reader_dir", "check_reader_dir", "build_rl_dir", "check_rl_dir", "write_configs",
                         "export_inputs", "read_eval", "beh_judges", "eval_summary", "upload"):
            assert c["cwd"] == repo and c["pythonpath"] == wt, c
            assert c["hf_offline"] == ("0" if c["prog"] == "upload" else "1"), c
    chk = [c for c in cs if c["prog"] == "check_reader_dir"][0]
    assert chk["cuda"] in ("0", "1") and chk["args"][-2:] == ["--gpu", "0"]
    gate = [c for c in cs if c["prog"] == "read_eval"][:4]
    assert [c["sub"] for c in gate] == ["plan", "read", "score", "gate"]
    reads_a = [c for c in cs if c["prog"] == "read_eval" and c["sub"] == "read"][1:]
    got = sorted((c["args"][c["args"].index("--reader") + 1], c["args"][c["args"].index("--gpu") + 1]) for c in reads_a)
    assert got == [("rl_beh_c040", "0"), ("rl_paper_c040", "1"), ("sft_scratch_6080", "0")], got
    scores = [c["args"][c["args"].index("--reader") + 1] for c in cs if c["prog"] == "read_eval" and c["sub"] == "score"]
    assert scores == ["sft_beh", "rl_beh_c040", "rl_paper_c040", "sft_scratch_6080"]
    rb = [c for c in cs if c["prog"] == "beh_judges" and c["sub"] == "read"][0]["args"]
    assert rb[rb.index("--readers") + 1] == "sft_beh,rl_beh_c040,rl_paper_c040,rl_em2_c040"
    assert rb[rb.index("--gpus") + 1] == "0,1" and rb[rb.index("--workers-per-gpu") + 1] == "2"
    judges = [c["args"] for c in cs if c["prog"] == "beh_judges" and c["sub"] == "judge"]
    assert [a[a.index("--judge") + 1] for a in judges] == ["truth", "content"]
    assert all(a[a.index("--max-cost") + 1] == "8" for a in judges)
    assert [c["sub"] for c in cs if c["prog"] == "read_eval"][-1] == "c1val"
    # RL: guard passed, training and dev evaluation done, the SFT checkpoint chosen by select
    rl_log = next((w["repo"] / "main_run" / "logs").glob("*_beh_reader_rl.out")).read_text()
    assert "guard ok" in rl_log
    rl_run = w["base"] / "agm_data" / "runs" / "reader_rl_scratch_beh_paper"
    assert (rl_run / "TRAIN_DONE").exists() and (rl_run / "EVAL_DONE").exists()
    rt = [c for c in cs if c["prog"] == "rl_train"][0]["args"]
    assert rt[rt.index("--adapter") + 1].endswith("reader_scratch_L40_beh/adapters/step_007082")
    # summary files committed and pushed
    names = _git(w["remote"], "show", "--name-only", "--format=", "main").split()
    assert sorted(names) == ["docs/figures/20261027_behaviour_reader_eval.json",
                             "docs/figures/20261027_behaviour_reader_eval.png"]
    assert time.time() - t0 < 200
    # rerun: DONE stages skipped; judge_b with a chosen truth judges content only; upload fails softly again
    n = len(cs)
    r = chain(w, BEH_CONTENT_TRUTH="em_content_v1")
    assert r.returncode == 0, r.stdout[-3000:]
    new = calls(w)[n:]
    assert sorted({c["prog"] for c in new}) == ["beh_judges", "upload"], new
    cj = [c["args"] for c in new if c["sub"] == "judge" and "content" in c["args"]][0]
    assert cj[cj.index("--content-truth") + 1] == "em_content_v1"
    assert (m / "judge_b.DONE").exists() and "soft failures: upload" in (m / "chain.DONE").read_text()


def test_judge_b_beside_read_a_and_soft_failures(world):
    """judge_b starts with read_a and is collected before c1val; a failing truth judge and a failing c1val are soft."""
    w = world
    r = chain(w, STUB_TRUTH_SLEEP="6", STUB_TRUTH_RC="1", STUB_C1VAL_RC="1")
    assert r.returncode == 0, (r.stdout[-4000:], r.stderr[-2000:])
    m = w["m"]
    assert "rc=1" in (m / "judge_b.FAILED").read_text() and not (m / "judge_b.CONTENT_REFUSED").exists()
    assert (m / "c1val.FAILED").exists() and (m / "read_a.DONE").exists() and (m / "summary.DONE").exists()
    assert "soft failures: judge_b c1val upload" in (m / "chain.DONE").read_text()
    assert not (m / "chain.FAILED").exists() and open_holds(w) == []
    cs = calls(w)
    truth = [c for c in cs if c["prog"] == "beh_judges" and c["sub"] == "judge"]
    assert [c["args"][c["args"].index("--judge") + 1] for c in truth] == ["truth"]  # the stage stops after truth
    t_truth_end = truth[0]["t"] + 6
    reads_a = [c for c in cs if c["prog"] == "read_eval" and c["sub"] == "read"][1:]
    assert len(reads_a) == 3 and all(truth[0]["t"] - 1 < c["t"] < t_truth_end for c in reads_a), reads_a
    c1val = [c for c in cs if c["prog"] == "read_eval" and c["sub"] == "c1val"]
    assert len(c1val) == 1 and c1val[0]["t"] >= t_truth_end - 0.5
    assert [c["prog"] for c in cs][-2:] == ["eval_summary", "upload"]
    out = r.stdout
    assert "judge_b runs in the background" in out and "judge_b failed (rc 1" in out and "c1val failed (rc 1)" in out
    # rerun: judge_b and c1val again (no longer failing), read_a and summary skipped
    n = len(cs)
    r = chain(w)
    assert r.returncode == 0, r.stdout[-3000:]
    new = calls(w)[n:]
    assert [c["prog"] for c in new if c["prog"] == "read_eval"] == ["read_eval"] and \
        [c["sub"] for c in new if c["prog"] == "read_eval"] == ["c1val"]
    assert (m / "c1val.DONE").exists() and (m / "judge_b.FAILED").exists() and "rc=5" in (m / "judge_b.FAILED").read_text()
    assert "soft failures: judge_b upload" in (m / "chain.DONE").read_text()


def test_read_hang_watchdog(world):
    """read_a alone: a reading that writes nothing gets TERM, one that ignores TERM gets KILL, one whose partial output
    keeps growing (no log output) is not stopped; the stage fails, no score runs, the GPU announcements are closed."""
    w = world
    r = chain(w, "read_a", BEH_READ_HANG_S="2", BEH_READ_GRACE_S="0", BEH_READ_KILL_S="1",
              STUB_READ_SLOW="rl_beh_c040", STUB_READ_HANG="rl_paper_c040", STUB_READ_HANG_NOTERM="sft_scratch_6080",
              timeout=120)
    assert r.returncode == 1, (r.stdout[-3000:], r.stderr[-2000:])
    m = w["m"]
    assert (m / "read_a.FAILED").exists() and not (m / "read_a.DONE").exists() and open_holds(w) == []
    log = next((w["repo"] / "main_run" / "logs").glob("*_beh_reader_read_a.out")).read_text()
    for rd in ("rl_paper_c040", "sft_scratch_6080"):
        assert f"hang: read_eval read --reader {rd} (pid" in log and f"read_eval read --reader {rd} on GPU" in log, log
    assert "still running 1 s after TERM; KILL" in log
    assert "hang: read_eval read --reader rl_beh_c040" not in log
    c1 = w["base"] / "agm_data" / "runs" / "beh_reader" / "c1style"
    assert len((c1 / "rl_beh_c040" / "preds.jsonl").read_text().splitlines()) == 10
    pids = [int(x.split("pid ", 1)[1].split(",", 1)[0]) for x in log.splitlines() if " on GPU " in x and ": pid " in x]
    assert len(pids) == 3
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert [c["sub"] for c in calls(w) if c["prog"] == "read_eval"] == ["read", "read", "read"]


def test_refusals(world):
    w = world
    m = w["m"]
    sft = w["base"] / "agm_data" / "c1_reader_40_beh" / "layers40"
    # unknown stage: nothing created
    r = chain(w, "nosuch")
    assert r.returncode == 2 and "unknown stage" in r.stdout and not m.exists()
    # unfinished SFT dir: build refuses without calling the module
    sft.mkdir(parents=True)
    (sft / "partial.npy").write_text("x")
    r = chain(w, "build")
    assert r.returncode == 1 and (m / "build.FAILED").exists() and calls(w) == []
    assert "no meta.json" in next((w["repo"] / "main_run" / "logs").glob("*_beh_reader_build.out")).read_text()
    # a failing checks file is never rerun or overwritten
    (sft / "meta.json").write_text("{}")
    bad = {"summary": {"all_pass": False, "complete": True, "failed": ["vec_rows"]}}
    (sft / "beh_checks.json").write_text(json.dumps(bad))
    r = chain(w, "check")
    assert r.returncode == 1 and (m / "check.FAILED").exists() and calls(w) == []
    assert json.loads((sft / "beh_checks.json").read_text()) == bad
    # a passing one is kept and the stage is DONE without the module
    good = {"summary": {"all_pass": True, "complete": True, "failed": []}}
    (sft / "beh_checks.json").write_text(json.dumps(good))
    r = chain(w, "check")
    assert r.returncode == 0 and (m / "check.DONE").exists() and calls(w) == []
    # the extraction still running blocks the build
    shutil.rmtree(sft)
    logs = w["base"] / "agm_data" / "beh_reader" / "logs"
    logs.mkdir(parents=True)
    holder = subprocess.Popen(["bash", "-c", "exec -a extract.py sleep 30"])
    try:
        (logs / "extract_shard0of2.pid").write_text(f"{holder.pid}\n")
        time.sleep(0.2)
        r = chain(w, "build")
    finally:
        holder.kill()
        holder.wait()
    assert r.returncode == 1 and calls(w) == []
    assert "extraction still running" in next((w["repo"] / "main_run" / "logs").glob("*_beh_reader_build.out")).read_text()


def test_second_chain_refused(world):
    w = world
    m = w["m"]
    m.mkdir(parents=True)
    holder = subprocess.Popen(["flock", str(m / "chain.lock"), "sleep", "4"])
    try:
        for _ in range(50):
            if (m / "chain.lock").exists():
                break
            time.sleep(0.1)
        time.sleep(0.3)
        r = chain(w, "select")
    finally:
        holder.wait()
    assert r.returncode == 3 and "holds" in r.stdout, (r.stdout, r.stderr)
    assert sorted(os.listdir(m)) == ["chain.lock"] and calls(w) == []


def test_dry_run_writes_nothing_and_lists_every_stage(world):
    w = world
    before = sorted(str(p) for p in w["base"].rglob("*")) + sorted(str(p) for p in w["repo"].rglob("*"))
    r = chain(w, DRY_RUN="1")
    after = sorted(str(p) for p in w["base"].rglob("*")) + sorted(str(p) for p in w["repo"].rglob("*"))
    assert r.returncode == 4, (r.stdout[-3000:], r.stderr[-2000:])
    assert after == before and calls(w) == []
    assert "NOT READY" in r.stdout and "[MISSING]" in r.stdout and "GPU 0: 0 MiB used" in r.stdout
    for st in STAGES:
        assert f"stage {st}: start" in r.stdout, st
    for frag in ("build_reader_dir.py", "check_reader_dir.py --gpu 0", "CUDA_VISIBLE_DEVICES=0", "build_rl_dir.py",
                 "check_rl_dir.py", "write_configs.py", "main_run/reader/train.py", "--batch-size 4",
                 "judge_curve --run-dirs", "select_checkpoint", "read_eval.py read --reader sft_beh --gpu 0",
                 "read_eval.py gate --reader sft_beh", "main_run.reader.rl_train", "export_inputs.py",
                 "beh_judges.py read --readers sft_beh,rl_beh_c040,rl_paper_c040,rl_em2_c040 --gpus 0,1 --workers-per-gpu 2",
                 "beh_judges.py judge --judge truth --max-cost 8", "beh_judges.py judge --judge content --max-cost 8",
                 "beh_judges.py summary", "read_eval.py score --reader sft_scratch_6080", "read_eval.py c1val",
                 "eval_summary.py --out-json", "HF_HUB_OFFLINE=0", "upload.py", "hang watchdog: TERM when",
                 "judge_b runs in the background beside read_a", "expected time: about 499 min"):
        assert frag in r.stdout, frag
    assert "[dry] announce GPUs" in r.stderr


def test_reap_only_this_hosts_dead_holds(tmp_path):
    """gpus.sh: a dead v2-family announcement of this host is closed; one written by another host (its pid means
    nothing here) and a live one of this host are left open; blockmix_held ignores the other host's."""
    bm = tmp_path / "runs" / "blockmix" / "markers"
    bm.mkdir(parents=True)
    dead, live = 4194000 + (os.getpid() % 1000), os.getpid()
    keys = {"mine_dead": f"em_reader_v2_beh_reader_rl_p{dead}_20261027160000_g0",
            "other_host": f"em_reader_v2_sft_p{dead}_20261027160000_g1",
            "mine_live": f"em_reader_v2_beh_reader_gate_p{live}_20261027160000_g1"}
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    for name, k in keys.items():
        (bm / f"{k}.gpu").write_text(k[-1] + "\n")
        (bm / f"{k}.STARTED").write_text("x\n")
        (bm / f"{k}.host").write_text(("elsewhere" if name == "other_host" else host) + "\n")
    pre = f"RUNS={tmp_path / 'runs'}; DRY=0; source {REPO / 'scripts/beh_reader/gpus.sh'}; "
    r = subprocess.run(["bash", "-c", pre + 'reap_stale_holds; echo "held [$(blockmix_held)]"'], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert (bm / f"{keys['mine_dead']}.DONE").exists()
    assert not (bm / f"{keys['other_host']}.DONE").exists() and not (bm / f"{keys['mine_live']}.DONE").exists()
    assert r.stdout.split("held [", 1)[1].split("]", 1)[0].split() == ["1"], r.stdout

"""Job queue of the EM behaviour-reader run (see docs/notes/20261027_em_beh_reader_results.md, "Jobs").

Readings (``e2_generate``) and candidate scores (``candidate_score``) of the new behaviour readers on the EM students
(em_s42, em_s1) and the context-control students (em_ctx_s42, em_ctx_s1), in priority order:

  P1  rl_beh_c040 behavioural: Δh and direct-write inputs on the eval-question and neutral docs (all four runs), and
      the no_vector / random injection controls (EM runs)
  P2  candidate scores, rl_beh_c040, behavioural and safety (all four runs)
  P3  rl_beh_c040 safety: as P1
  P4  sft_beh_7082 behavioural: Δh and direct-write on eval questions, Δh on neutral (all runs); controls (EM runs)
  P5  rl_beh_c040 both questions: Δh and direct-write on the training docs (all runs); summed raw gradients (EM runs)

Settings as the old reader's EM readings (docs/notes/20261025_em_dact_results.md): 100 samples per read step, every step of the
input file, default decoding. A control-student job waits for ``$L/CTX_<run>_DONE`` and is skipped once
``$L/CTX_<run>_FAILED`` exists. A worker takes the first free job (``mkdir`` lock), runs it on its GPU
(``CUDA_VISIBLE_DEVICES``; the tools get ``--device cuda:0``, never ``--gpus``), and marks it done when every expected
output exists. A failed or timed-out job is retried once after its partial files are moved aside (never deleted).

  python -m main_run.em.beh_jobs list                  # the job table
  python -m main_run.em.beh_jobs status                # done / running / waiting / failed
  python -m main_run.em.beh_jobs work --gpu 0          # one worker per GPU

Directories: $AGM_WORK (default ./work) with runs/, logs/, agm-readers/, sprint-data-2610/behaviour_readers/ and
cand_results_em/; R, L, A, B and CAND override them one by one.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ENV = os.environ
WORK = Path(ENV.get("AGM_WORK", "work"))  # work root; each directory below can also be set on its own
R = Path(ENV.get("R", WORK / "runs"))
L = Path(ENV.get("L", WORK / "logs"))
A = Path(ENV.get("A", WORK / "agm-readers"))
B = Path(ENV.get("B", WORK / "sprint-data-2610" / "behaviour_readers"))
RJ_BEH = Path(ENV.get("RJ_BEH", str(B / "sft_scratch_beh_7082" / "RUN.json")))
CAND = Path(ENV.get("CAND", WORK / "cand_results_em"))
STEPS = ENV.get("STEPS", "5,10,20,30,40,50,55,60,65,70,75,100,150,250,375")
JOBS = L / "jobs"
SAMPLES = 100

READERS = {"rl_beh_c040": "rl_beh_paper_prompt_cycle040", "sft_beh_7082": "sft_scratch_beh_7082"}
EM_RUNS = ("em_s42", "em_s1")
CTX_RUNS = ("em_ctx_s42", "em_ctx_s1")
QUESTIONS = ("behavioural", "safety")
# a reading takes ~5 min on a B200 shared with one other process; a stalled one (kernel never returns under GPU
# sharing, seen 2026-10-25 and 2026-10-27) is killed and retried
TIMEOUT = {"read": 20 * 60, "cand": 40 * 60}
MAX_TRIES = 2

# (input stem in <run>/e2, output prefix e2_generate adds, tag, extra e2_generate args)
DACT = {
    "ckptdiff_evalq": ("ckptdiff_inputs_evalq_dact", "ckptdiff_", "dact_evalq", []),
    "dwx_evalq": ("dwx_inputs_evalq_dact", "dwx_", "dact_evalq", ["--windows", "cum"]),
    "ckptdiff_neutral": ("ckptdiff_inputs_neutral_dact", "ckptdiff_", "dact_neutral", []),
    "dwx_neutral": ("dwx_inputs_neutral_dact", "dwx_", "dact_neutral", ["--windows", "cum"]),
    "ckptdiff_train": ("ckptdiff_inputs_train_dact", "ckptdiff_", "dact_train", []),
    "dwx_train": ("dwx_inputs_train_dact", "dwx_", "dact_train", ["--windows", "cum"]),
    "grad_cum": ("inputs_cum_dact", "", "grad_cum", []),
    "grad_w1": ("inputs_w1_10_25", "", "grad_w1", ["--windows", "w1", "--steps", STEPS]),
}
CTL_CONDITIONS = ("no_vector", "random")  # on the ckptdiff evalq file, tag ctl (student-independent baselines)
CAND_STEMS_CTX = [DACT[k][0] for k in ("ckptdiff_evalq", "ckptdiff_neutral", "ckptdiff_train", "dwx_evalq",
                                       "dwx_neutral", "dwx_train")]
CAND_STEMS_EM = CAND_STEMS_CTX + [DACT["grad_cum"][0], DACT["grad_w1"][0]]


def is_ctx(run: str) -> bool:
    return run.startswith("em_ctx_")


def reading_name(reader: str, q: str, prefix: str, tag: str, cond: str = "correct") -> str:
    """e2_generate.out_name with the transform prefix (output_reader_name)."""
    return f"{prefix}{reader}_{q}" + (f"_{cond}" if cond != "correct" else "") + f"_{tag}"


def reading_job(prio: int, reader: str, q: str, run: str, key: str, cond: str = "correct") -> dict:
    stem, prefix, tag, extra = DACT[key]
    if cond != "correct":
        tag, extra = "ctl", []
    name = reading_name(reader, q, prefix, tag, cond)
    e2 = R / run / "e2"
    cmd = [sys.executable, "-m", "main_run.reader.e2_generate", "--inputs", str(e2 / f"{stem}.npz"), "--run",
           str(R / run), "--adapter", str(B / READERS[reader]), "--run-json", str(RJ_BEH), "--reader-dir",
           str(A / "rl_data" / "reader"), "--reader-name", reader, "--questions", q, "--samples", str(SAMPLES),
           "--no-summarize", "--tag", tag, "--device", "cuda:0", *extra]
    if cond != "correct":
        cmd += ["--condition", cond]
    if ENV.get("MAXGIB"):
        cmd += ["--max-gpu-gib", ENV["MAXGIB"]]
    out = e2 / f"{name}.jsonl"
    return dict(id=f"P{prio}__{run}__{name}", prio=prio, kind="read", run=run, cmd=cmd, outputs=[str(out)],
                aside=[str(e2 / f"{name}{s}") for s in (".jsonl", "_settings.json", "_summary.json")],
                needs=[str(L / f"CTX_{run}_DONE")] if is_ctx(run) else [str(e2 / f"{stem}.npz")],
                skip_if=[str(L / f"CTX_{run}_FAILED")] if is_ctx(run) else [])


def cand_job(prio: int, reader: str, q: str, run: str) -> dict:
    stems = CAND_STEMS_CTX if is_ctx(run) else CAND_STEMS_EM
    e2 = R / run / "e2"
    cmd = [sys.executable, "-m", "main_run.reader.candidate_score", "--inputs", *[str(e2 / f"{s}.npz") for s in stems],
           "--adapter", str(B / READERS[reader]), "--run-json", str(RJ_BEH), "--reader-dir", str(A / "rl_data" / "reader"),
           "--reader-name", reader, "--question", q, "--candidates", "configs/candidates/em_behaviours.json",
           "--conditions", "correct,random,no_vector", "--out-dir", str(CAND / "{run}" / "e2"), "--batch-rows",
           ENV.get("BATCH_ROWS", "128"), "--resume", "--device", "cuda:0"]
    if ENV.get("MAXGIB"):
        cmd += ["--max-gpu-gib", ENV["MAXGIB"]]
    outs = [str(CAND / run / "e2" / f"cand_{reader}_{q}_em_behaviours__{s}.npz") for s in stems]
    return dict(id=f"P{prio}__{run}__cand_{reader}_{q}", prio=prio, kind="cand", run=run, cmd=cmd, outputs=outs,
                aside=[], needs=[str(L / f"CTX_{run}_DONE")] if is_ctx(run) else [str(e2 / f"{stems[0]}.npz")],
                skip_if=[str(L / f"CTX_{run}_FAILED")] if is_ctx(run) else [])


def all_jobs() -> list[dict]:
    jobs, runs = [], EM_RUNS + CTX_RUNS
    main4 = ("ckptdiff_evalq", "dwx_evalq", "ckptdiff_neutral", "dwx_neutral")
    for prio, q in ((1, "behavioural"), (3, "safety")):
        for run in runs:
            jobs += [reading_job(prio, "rl_beh_c040", q, run, k) for k in main4[:2]]
            if not is_ctx(run):
                jobs += [reading_job(prio, "rl_beh_c040", q, run, "ckptdiff_evalq", c) for c in CTL_CONDITIONS]
            jobs += [reading_job(prio, "rl_beh_c040", q, run, k) for k in main4[2:]]
    for q in QUESTIONS:
        jobs += [cand_job(2, "rl_beh_c040", q, run) for run in runs]
    for run in runs:
        jobs += [reading_job(4, "sft_beh_7082", "behavioural", run, k) for k in ("ckptdiff_evalq", "dwx_evalq",
                                                                               "ckptdiff_neutral")]
        if not is_ctx(run):
            jobs += [reading_job(4, "sft_beh_7082", "behavioural", run, "ckptdiff_evalq", c) for c in CTL_CONDITIONS]
    for q in QUESTIONS:
        for run in runs:
            jobs += [reading_job(5, "rl_beh_c040", q, run, k) for k in ("ckptdiff_train", "dwx_train")]
            if not is_ctx(run):
                jobs += [reading_job(5, "rl_beh_c040", q, run, k) for k in ("grad_cum", "grad_w1")]
    jobs.sort(key=lambda j: j["prio"])  # stable: the order above within a priority
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    return jobs


def _marker(job: dict, suffix: str) -> Path:
    return JOBS / f"{job['id']}.{suffix}"


def job_state(job: dict) -> str:
    if _marker(job, "done").exists():
        return "done"
    # a reading's jsonl appears only when it is complete; a candidate file can be partial (--save-every), so only
    # the done marker counts for it
    if job["kind"] == "read" and Path(job["outputs"][0]).exists() and not _marker(job, "lock").exists():
        return "done"
    if _marker(job, "failed").exists():
        return "failed"
    if any(Path(p).exists() for p in job["skip_if"]):
        return "skipped"
    if _marker(job, "lock").exists():
        return "running"
    if not all(Path(p).exists() for p in job["needs"]):
        return "waiting"
    return "ready"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clear_stale_locks() -> list[str]:
    """Locks left by a worker that died on this host (its pid is gone). Locks of other hosts are left alone."""
    cleared = []
    for lock in JOBS.glob("*.lock"):
        try:
            info = json.loads((lock / "owner.json").read_text())
        except (OSError, ValueError):
            continue
        if info.get("host") == socket.gethostname() and not _pid_alive(int(info.get("pid", -1))):
            (lock / "owner.json").rename(lock.parent / f"{lock.name[:-5]}.stale_{int(time.time())}.json")
            lock.rmdir()
            cleared.append(lock.name)
    return cleared


def _aside(paths: list[str]) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for p in paths:
        if os.path.exists(p):
            os.replace(p, f"{p}.aside_{stamp}")


def run_job(job: dict, gpu: str, log_fn=print, dry: bool = False) -> bool:
    """Run one job under its lock; True when every output exists afterwards."""
    tries_file = _marker(job, "tries")
    tries = int(tries_file.read_text()) if tries_file.exists() else 0
    if job["kind"] == "read":
        _aside([p for p in job["aside"] if not Path(job["outputs"][0]).exists()])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = JOBS / f"{job['id']}.log"
    t0 = time.time()
    log_fn(f"{time.strftime('%H:%M:%S')} GPU {gpu} start {job['id']} (try {tries + 1})")
    if dry:
        rc = 0
        log.write_text(" ".join(job["cmd"]) + "\n")
    else:
        with open(log, "a") as f:
            f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} try {tries + 1} on GPU {gpu}\n# {' '.join(job['cmd'])}\n")
            f.flush()
            try:
                rc = subprocess.run(job["cmd"], stdout=f, stderr=subprocess.STDOUT, env=env,
                                    timeout=TIMEOUT[job["kind"]]).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
    ok = rc == 0 and all(Path(p).exists() for p in job["outputs"])
    dt = round((time.time() - t0) / 60, 1)
    if ok:
        _marker(job, "done").write_text(json.dumps(dict(gpu=gpu, minutes=dt, host=socket.gethostname(),
                                                        end=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    else:
        tries_file.write_text(str(tries + 1))
        if tries + 1 >= MAX_TRIES:
            _marker(job, "failed").write_text(f"rc {rc} after {tries + 1} tries; see {log}\n")
    with open(L / "jobs_timing.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{job['id']}\tGPU{gpu}\t{dt} min\t{'ok' if ok else f'rc={rc}'}\n")
    log_fn(f"{time.strftime('%H:%M:%S')} GPU {gpu} {'done' if ok else f'FAILED rc={rc}'} {job['id']} ({dt} min)")
    return ok


def work(gpu: str, poll_s: int = 60, dry: bool = False, log_fn=print) -> int:
    JOBS.mkdir(parents=True, exist_ok=True)
    for c in clear_stale_locks():
        log_fn(f"cleared stale lock {c}")
    while True:
        jobs = all_jobs()
        states = {j["id"]: job_state(j) for j in jobs}
        ran = False
        for j in jobs:
            if states[j["id"]] != "ready":
                continue
            lock = _marker(j, "lock")
            try:
                lock.mkdir()
            except FileExistsError:
                continue
            (lock / "owner.json").write_text(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), gpu=gpu)))
            try:
                run_job(j, gpu, log_fn=log_fn, dry=dry)
            finally:
                (lock / "owner.json").unlink(missing_ok=True)
                lock.rmdir()
            ran = True
            break  # re-read the table: priorities and waits may have changed
        if ran:
            continue
        if not any(s == "waiting" for s in states.values()):
            log_fn(f"{time.strftime('%H:%M:%S')} GPU {gpu}: no job left for this worker; exit")
            (L / f"WORKER_gpu{gpu}_{os.getpid()}_EXIT").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            return 0
        time.sleep(poll_s)


def status() -> dict:
    jobs = all_jobs()
    counts: dict = {}
    rows = []
    for j in jobs:
        s = job_state(j)
        counts[s] = counts.get(s, 0) + 1
        rows.append((j["prio"], s, j["id"]))
    return dict(counts=counts, rows=rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    st = sub.add_parser("status")
    st.add_argument("--all", action="store_true", help="every job, not only the unfinished ones")
    w = sub.add_parser("work")
    w.add_argument("--gpu", required=True)
    w.add_argument("--poll", type=int, default=60)
    w.add_argument("--dry", action="store_true", help="write the command to the job log instead of running it")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for j in all_jobs():
            print(f"P{j['prio']}\t{j['kind']}\t{j['id']}")
        return 0
    if a.cmd == "status":
        s = status()
        for prio, state, jid in s["rows"]:
            if a.all or state != "done":
                print(f"P{prio}\t{state}\t{jid}")
        print(json.dumps(s["counts"]))
        return 0
    return work(a.gpu, poll_s=a.poll, dry=a.dry)


if __name__ == "__main__":
    raise SystemExit(main())

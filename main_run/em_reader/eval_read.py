#!/usr/bin/env python
"""Readings of the EM-reader evaluation: one e2_generate process per job.

  plan  [--readers old,new]                       print every job and its command (dry run; nothing is started)
  run   --readers old|new|old,new --gpus 2,3 [--workers-per-gpu 2] [--hang-s 600] [--retry-hung 1]
                                                  run the jobs whose output file is missing or broken, one worker
                                                  thread per (GPU, slot) pulling from one queue; each job writes its
                                                  log to runs/em_reader/read_logs/<job key>.<stamp>.log
  fetch-students                                  download the EM students' update inputs (HF dataset repos
                                                  activation-gradient-metamodels/em-s42-run and em-s1-run, files
                                                  e2/*_dact.npz) and write re-pathed copies (below); a failed
                                                  download is written to runs/em_reader/students/FETCH.json and the
                                                  student jobs are skipped

Jobs, outputs and questions: eval_common.reading_jobs. 100 answers per input, e2_generate's sampling (temperature 1.0,
seed 0 + step), the em-reader worktree's code (training-pipeline a345ddf). Two processes share a GPU by default: one
generate call (100 samples, one window) peaked at 46.8 GiB and runs about 8 s, mostly per-token overhead, so a second
process on the same B200 roughly doubles throughput; each is capped at --max-gpu-gib.

Re-pathed student inputs: e2_generate refuses an inputs file whose meta ``run`` differs from ``--run``; the collaborator's
exports name that instance's directory (/workspace/runs/em_s42). fetch-students writes
runs/em_reader/students/<seed>/e2/<same name>.npz with every array unchanged and meta ``run`` set to the local
directory (the original kept as ``run_original``, plus ``repathed_from``); the order hash covers steps, windows and
layers only, so it still holds. Each copy is re-read and compared array by array with the download.

Hang timeout (three v2 reads once stopped inside one generate call, after 6-8 of 10-14 steps, with the GPU at
100% for 20-30 min, and had to be stopped by pid; one old-reader read did the same at 110/185 steps). A read
whose log and partial output (e2_generate's <out>.tmp, one row per finished step, about every 20 s; the log gets a
line every 10 steps, about every 200 s) have not changed size for --hang-s seconds (default $EM_READ_HANG_S, else 600;
0 turns it off) gets TERM, then KILL --kill-grace-s (60) seconds later, and is logged FAILED with reason
"hung (log unchanged N s)". It is queued once more (--retry-hung 1); the retry waits until no other read runs on its
GPU and keeps the GPU to itself. A second failure is final and counts in "finished: N done, M failed" (exit code 1).
A read still alive --kill-grace-s seconds after KILL (a CUDA process stuck in the driver) is left behind and counted
FAILED "unkillable ..." without a retry, so the runner always reaches its "finished" line. A read that exits 0 between
the last look and TERM is not counted as hung; a retried job whose output turned out complete counts as done.
$EM_READ_WORKERS_PER_GPU, when set, overrides --workers-per-gpu (a relaunch through a chain that passes the flag).

One runner per log folder: run takes flock on <read_logs>/RUNNER.lock (exit 3 when another runner holds it). A job
whose <out>.tmp changed less than the hang timeout ago (default 600 s) is not started (FAILED "partial output
written N s ago"): some other process may still be writing it.

Existing outputs: a job is skipped only when its output file holds at least one reading row, every line parses as a
JSON row with "step" and "inputs", and the rows cover every step the job reads (the job's steps, else the inputs
file's). An empty, unparsable or short file is moved to <out>.invalid_<stamp> (kept, not deleted) and read again.
Before any read, the files an earlier attempt left next to the output are moved aside the same way: e2_generate's
<out>.tmp, <stem>_settings.json and <stem>_summary.json (e2_generate refuses to overwrite these for transformed
inputs) and the judges' <stem>_judge_<judge>.jsonl (em_judges never overwrites a judged file).

The HF token is the one huggingface_hub finds under $HF_HOME (as scripts/hf_upload_student_runs.py); never printed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as E  # noqa: E402

LOG_DIR = E.EVAL_ROOT / "read_logs"
HANG_S_DEFAULT = 600.0  # seconds without growth of a read's log and partial output before it is stopped
KILL_GRACE_S = 60.0  # KILL this many seconds after TERM
STAGGER_S = 20.0  # stagger the base-model loads of the first jobs


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [eval_read] {msg}", flush=True)


def hang_seconds(cli: float | None = None) -> float:
    """--hang-s, else $EM_READ_HANG_S, else HANG_S_DEFAULT; 0 or less turns the hang timeout off."""
    if cli is not None:
        return float(cli)
    env = os.environ.get("EM_READ_HANG_S", "").strip()
    return float(env) if env else HANG_S_DEFAULT


def expected_steps(job: dict) -> list[int] | None:
    """Steps the job reads: its own ``steps`` (v2 finance), else every step of its inputs file; None if unreadable."""
    if job.get("steps"):
        return [int(t) for t in job["steps"]]
    try:
        import numpy as np

        with np.load(job["inputs"]) as z:
            return [int(t) for t in z["steps"]]
    except Exception:  # noqa: BLE001
        return None


def output_problem(job: dict) -> str | None:
    """None when the job's reading file is complete, else why it has to be read: "missing", "empty", ..."""
    p = Path(job["out"])
    if not p.exists():
        return "missing"
    steps = set()
    try:
        if p.stat().st_size == 0:
            return "empty"
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "step" not in row or not isinstance(row.get("inputs"), dict):
                    return f"unparsable (line {i} is not a reading row)"
                steps.add(int(row["step"]))
    except (OSError, ValueError, TypeError) as e:
        return f"unparsable ({type(e).__name__})"
    if not steps:
        return "empty (no rows)"
    want = expected_steps(job)
    if want is not None and not set(want) <= steps:
        return f"short ({len(set(want) - steps)} of {len(want)} steps missing)"
    return None


SIDE_SUFFIXES = ("_settings.json", "_summary.json")  # e2_generate's files next to <stem>.jsonl


def side_files(out: Path) -> list[Path]:
    """Files of one reading other than the reading itself: e2_generate's partial output, settings and summary, and
    the judges' files (em_judges.judged_path: <stem>_judge_<judge>.jsonl)."""
    out = Path(out)
    files = [Path(str(out) + ".tmp")] + [out.with_name(out.stem + s) for s in SIDE_SUFFIXES]
    pat = re.compile(re.escape(out.stem) + r"_judge_[a-z]+\.jsonl")
    if out.parent.is_dir():
        files += sorted(p for p in out.parent.iterdir() if pat.fullmatch(p.name))
    return files


def set_aside(job: dict, why: str) -> list[str]:
    """Before a job is read (again): move a broken output and every side file an earlier attempt left (kept as
    <name>.invalid_<stamp>, never deleted); returns the new names."""
    out = Path(job["out"])
    stamp = time.strftime(".invalid_%Y%m%d_%H%M%S")
    moved = []
    for p in ([out] if why != "missing" else []) + side_files(out):
        if not p.exists():
            continue
        dst, k = p.with_name(p.name + stamp), 1
        while dst.exists():  # os.rename would replace it
            dst, k = p.with_name(p.name + f"{stamp}_{k}"), k + 1
        os.rename(p, dst)
        moved.append(dst.name)
    if moved:
        log(f"{job['key']}: output {why}; moved aside in {out.parent}: {moved}; reading again")
    return moved


def tmp_age(job: dict) -> float | None:
    """Seconds since <out>.tmp last changed, None when there is none."""
    try:
        return time.time() - os.stat(str(job["out"]) + ".tmp").st_mtime
    except OSError:
        return None


def _sizes(paths) -> tuple:
    out = []
    for p in paths:
        try:
            out.append(os.stat(p).st_size)
        except OSError:
            out.append(-1)
    return tuple(out)


def run_watched(cmd, log_path: Path, partial: Path, *, hang_s: float, grace_s: float, cwd: str, env: dict,
                label: str) -> tuple[int, str | None]:
    """Run one read with stdout and stderr in ``log_path``; returns (return code, hang reason or None).

    The read counts as hung when neither ``log_path`` nor ``partial`` (e2_generate's <out>.tmp) has changed size for
    ``hang_s`` seconds; it then gets TERM, and KILL ``grace_s`` seconds later."""
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            log(f"start {label} pid {proc.pid}; log {log_path}")
            watch = (log_path, partial)
            poll = min(10.0, max(0.2, hang_s / 20)) if hang_s > 0 else 10.0
            last, t_last = _sizes(watch), time.monotonic()
            while True:
                try:
                    return proc.wait(timeout=poll), None
                except subprocess.TimeoutExpired:
                    pass
                if hang_s <= 0:
                    continue
                cur, now = _sizes(watch), time.monotonic()
                if cur != last:
                    last, t_last = cur, now
                    continue
                idle = now - t_last
                if idle < hang_s:
                    continue
                rc = proc.poll()
                if rc is not None:  # it ended on its own since the last wait
                    return rc, None
                log(f"hang: {label} pid {proc.pid}: log and partial output unchanged for {idle:.0f} s; TERM")
                proc.terminate()
                try:
                    rc = proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    log(f"hang: {label} pid {proc.pid} still running {grace_s:.0f} s after TERM; KILL")
                    proc.kill()
                    try:
                        rc = proc.wait(timeout=grace_s)
                    except subprocess.TimeoutExpired:  # stuck in the driver (state D): leave it, free the worker
                        log(f"hang: {label} pid {proc.pid} still alive {grace_s:.0f} s after KILL; left behind")
                        return None, (f"unkillable: hung (log unchanged {idle:.0f} s), alive {grace_s:.0f} s after "
                                      f"KILL, pid {proc.pid}")
                if rc == 0:  # exited normally between the last look and TERM (e2_generate has no TERM handler)
                    return rc, None
                return rc, f"hung (log unchanged {idle:.0f} s)"
        except BaseException:
            if proc.poll() is None:  # the runner itself failed: do not leave the read running (as subprocess.call)
                proc.kill()
                try:
                    proc.wait(timeout=max(grace_s, 1.0))
                except subprocess.TimeoutExpired:
                    pass
            raise


def job_env() -> dict:
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # --gpus pins the physical GPU inside e2_generate
    env["PYTHONPATH"] = str(E.EMWT)
    env["IIF_BASE"] = E.IIF_BASE
    env.setdefault("HF_HOME", str(Path(E.IIF_BASE) / "hf_home"))
    env["HF_HUB_OFFLINE"] = "1"
    return env


def cmd_plan(readers) -> int:
    jobs = E.reading_jobs(readers)
    n_todo = 0
    for j in jobs:
        why = output_problem(j)
        n_todo += why is not None
        state = "done" if why is None else "todo" if why == "missing" else f"redo: output {why}"
        print(f"[{state}] {j['key']} -> {j['out']}")
        print("   cd " + str(E.EMWT) + " && PYTHONPATH=" + str(E.EMWT) + " " + " ".join(
            f"'{x}'" if " " in x else x for x in E.e2_command(j, "<gpu>")))
    print(f"{len(jobs)} jobs, {n_todo} to run")
    return 0


def missing_adapters(jobs) -> list[str]:
    miss = set()
    for j in jobs:
        for p in (Path(j["adapter"]) / "adapter_model.safetensors", Path(j["run_json"])):
            if not p.exists():
                miss.add(str(p))
    return sorted(miss)


def runner_lock() -> tuple[int | None, str]:
    """flock on <LOG_DIR>/RUNNER.lock: (fd, "") when taken, (None, holder) when another runner holds it. On a file
    system without flock the run goes on unlocked (fd -1)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / "RUNNER.lock"
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o660)  # not inherited by the reads (close-on-exec)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.pread(fd, 200, 0).decode(errors="replace").strip()
        os.close(fd)
        return None, holder
    except OSError as e:
        log(f"flock on {p} failed ({e}); running without the one-runner lock")
        os.close(fd)
        return -1, ""
    os.ftruncate(fd, 0)
    os.pwrite(fd, f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode(), 0)
    return fd, ""


def cmd_run(readers, gpus: list[str], per_gpu: int, max_gpu_gib: float, *, hang_s: float | None = None,
            retry_hung: int = 1, kill_grace_s: float = KILL_GRACE_S) -> int:
    fd, holder = runner_lock()
    if fd is None:
        log(f"another eval_read run holds {LOG_DIR / 'RUNNER.lock'} ({holder or 'holder unknown'}); not started")
        return 3
    try:
        return _run_jobs(readers, gpus, per_gpu, max_gpu_gib, hang_s=hang_s, retry_hung=retry_hung,
                         kill_grace_s=kill_grace_s)
    finally:
        if fd >= 0:
            os.close(fd)


def _run_jobs(readers, gpus: list[str], per_gpu: int, max_gpu_gib: float, *, hang_s: float | None,
              retry_hung: int, kill_grace_s: float) -> int:
    hang_s = hang_seconds(hang_s)
    busy_s = hang_s if hang_s > 0 else HANG_S_DEFAULT  # a fresher <out>.tmp may belong to a live process
    jobs = [j for j in E.reading_jobs(readers) if output_problem(j) is not None]
    miss = missing_adapters(jobs)
    if miss:
        log(f"missing reader files: {miss}")
        return 2
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"{len(jobs)} jobs for readers {readers} on GPUs {gpus} x {per_gpu}")
    log(f"hang timeout {hang_s:.0f} s (log and partial output unchanged; TERM, KILL {kill_grace_s:.0f} s later); a "
        f"hung job is queued again up to {retry_hung} time(s), alone on its GPU" if hang_s > 0 else "hang timeout off")
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    failed, done, retried = [], [], []
    lock = threading.Lock()
    # a hang retry runs alone on its GPU: it waits until the GPU's other slots are idle, and they start nothing new
    # while it waits or runs
    gpu_cv = threading.Condition()
    busy = {g: 0 for g in gpus}
    alone = {g: False for g in gpus}
    waiting_alone = {g: 0 for g in gpus}

    def take_gpu(gpu: str, exclusive: bool, key: str) -> None:
        with gpu_cv:
            if exclusive:
                waiting_alone[gpu] += 1
                if busy[gpu] or alone[gpu]:
                    log(f"hang retry of {key} waits for GPU {gpu} to be free")
                gpu_cv.wait_for(lambda: busy[gpu] == 0 and not alone[gpu])
                waiting_alone[gpu] -= 1
                alone[gpu] = True
            else:
                gpu_cv.wait_for(lambda: not alone[gpu] and waiting_alone[gpu] == 0)
            busy[gpu] += 1

    def free_gpu(gpu: str, exclusive: bool) -> None:
        with gpu_cv:
            busy[gpu] -= 1
            if exclusive:
                alone[gpu] = False
            gpu_cv.notify_all()

    def worker(gpu: str, slot: int):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            why = output_problem(j)
            attempt = int(j.get("_hang_retry", 0))
            if why is None:  # e.g. a hung attempt had already written its output: count it, do not drop it
                with lock:
                    done.append(j["key"])
                    log(f"done {j['key']} (output already complete{', hang retry not needed' if attempt else ''}; "
                        f"{len(done)} done, {len(failed)} failed)")
                continue
            age = tmp_age(j)
            if age is not None and age < busy_s:
                with lock:
                    failed.append(j["key"])
                    log(f"FAILED {j['key']}: partial output {j['out']}.tmp written {age:.0f} s ago (another process "
                        f"may still be reading it); not started ({len(done)} done, {len(failed)} failed)")
                continue
            exclusive = attempt > 0
            take_gpu(gpu, exclusive, j["key"])
            lp = LOG_DIR / (j["key"].replace("/", "__") + time.strftime(".%Y%m%d_%H%M%S") + f".g{gpu}s{slot}.log")
            t0 = time.time()
            try:
                set_aside(j, why)
                label = f"{j['key']} on GPU {gpu} slot {slot}" + (
                    f" (hang retry {attempt}/{retry_hung}, alone on the GPU)" if exclusive else "")
                rc, reason = run_watched(E.e2_command(j, gpu, max_gpu_gib=max_gpu_gib), lp, Path(str(j["out"]) + ".tmp"),
                                         hang_s=hang_s, grace_s=kill_grace_s, cwd=str(E.EMWT), env=job_env(),
                                         label=label)
            except Exception as e:  # noqa: BLE001  (counted as a failed job instead of a silently dead worker)
                rc, reason = None, f"runner error {type(e).__name__}: {e}"
            finally:
                free_gpu(gpu, exclusive)
            ok = reason is None and rc == 0 and output_problem(j) is None
            with lock:
                note = ""
                if ok:
                    done.append(j["key"])
                elif reason and reason.startswith("hung") and attempt < retry_hung:
                    retried.append(j["key"])
                    q.put(dict(j, _hang_retry=attempt + 1))
                    note = f"; hang retry {attempt + 1}/{retry_hung} queued, alone on its GPU"
                else:
                    failed.append(j["key"])
                log(f"{'done' if ok else 'FAILED'} {j['key']} on GPU {gpu} slot {slot} in {time.time() - t0:.0f} s "
                    f"(rc {rc}{'; ' + reason if reason else ''}; {len(done)} done, {len(failed)} failed, about "
                    f"{q.qsize()} queued){note}; log {lp}")

    threads = []
    for g in gpus:
        for s in range(per_gpu):
            t = threading.Thread(target=worker, args=(g, s), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(STAGGER_S)  # stagger the base-model loads
    for t in threads:
        t.join()
    if retried:
        log(f"hang retries: {len(retried)} {retried}")
    log(f"finished: {len(done)} done, {len(failed)} failed {failed}")
    return 0 if not failed else 1


def cmd_fetch_students() -> int:
    import numpy as np

    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": {}}
    E.STUDENTS.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # noqa: BLE001
        report["error"] = f"huggingface_hub import: {e}"
        snapshot_download = None
    for seed, repo in E.STUDENT_HF.items():
        rec = dict(repo=repo, repo_type="dataset", pattern="e2/*_dact.npz")
        if snapshot_download is None:
            report["seeds"][seed] = rec
            continue
        raw = E.STUDENTS / "hf_download" / seed
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=["e2/*_dact.npz"], local_dir=str(raw),
                              max_workers=8)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            report["seeds"][seed] = rec
            log(f"{seed}: download failed ({rec['error'][:200]})")
            continue
        files = sorted((raw / "e2").glob("*_dact.npz"))
        rec["files"] = {}
        out_dir = E.STUDENTS / seed / "e2"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = out_dir / f.name
            frec = {}
            try:
                with np.load(f) as z:
                    arrays = {k: z[k] for k in z.files}
                meta = json.loads(str(arrays["meta"]))
                frec.update(kind=meta.get("kind"), windows=[str(w) for w in arrays["windows"]],
                            n_steps=int(len(arrays["steps"])), run_original=meta.get("run"),
                            transform=(meta.get("transform") or {}).get("variant"))
                if meta.get("kind") != "grad":
                    frec["skipped"] = f"meta kind {meta.get('kind')!r}; the trained-reader loader needs 'grad'"
                elif not dst.exists():
                    meta2 = dict(meta, run=str(E.STUDENTS / seed), run_original=meta.get("run"),
                                 repathed_from=str(f), repathed=time.strftime("%Y-%m-%d %H:%M:%S"))
                    arrays["meta"] = np.array(json.dumps(meta2))
                    tmp = dst.with_name(dst.stem + ".tmp.npz")
                    np.savez(tmp, **arrays)
                    with np.load(tmp) as z2:
                        same = all(np.array_equal(z2[k], arrays[k]) for k in arrays if k != "meta")
                    if not same:
                        raise ValueError("re-pathed copy differs from the download")
                    os.rename(tmp, dst)
                    frec["repathed"] = str(dst)
                else:
                    frec["repathed"] = str(dst) + " (existed)"
            except Exception as e:  # noqa: BLE001
                frec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec["files"][f.name] = frec
        report["seeds"][seed] = rec
        log(f"{seed}: {len(files)} files -> {out_dir}")
    out = E.STUDENTS / "FETCH.json"
    if out.exists():
        out = E.STUDENTS / time.strftime("FETCH_%Y%m%d_%H%M%S.json")
    out.write_text(json.dumps(report, indent=1))
    n = sum(len(E.student_inputs(s)) for s in E.STUDENT_ONSET)
    log(f"student inputs ready: {n}; report {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--readers", default="old,new")
    r = sub.add_parser("run")
    r.add_argument("--readers", required=True)
    r.add_argument("--gpus", required=True, help="comma list of physical GPU indices")
    r.add_argument("--workers-per-gpu", type=int, default=2)
    r.add_argument("--max-gpu-gib", type=float, default=80.0)
    r.add_argument("--hang-s", type=float, default=None,
                   help=f"stop a read whose log and partial output have not grown for this many seconds (default "
                        f"$EM_READ_HANG_S, else {HANG_S_DEFAULT:.0f}; 0 turns it off)")
    r.add_argument("--retry-hung", type=int, default=1,
                   help="times a read stopped by the hang timeout is queued again, alone on its GPU (default 1)")
    r.add_argument("--kill-grace-s", type=float, default=KILL_GRACE_S,
                   help=f"KILL a hung read this many seconds after TERM (default {KILL_GRACE_S:.0f})")
    sub.add_parser("fetch-students")
    a = ap.parse_args(argv)
    if a.cmd == "fetch-students":
        return cmd_fetch_students()
    readers = tuple(x.strip() for x in a.readers.split(",") if x.strip())
    if a.cmd == "plan":
        return cmd_plan(readers)
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    per_gpu = a.workers_per_gpu
    env_w = os.environ.get("EM_READ_WORKERS_PER_GPU", "").strip()
    if env_w:
        log(f"workers per GPU {int(env_w)} from $EM_READ_WORKERS_PER_GPU (flag said {per_gpu})")
        per_gpu = int(env_w)
    return cmd_run(readers, gpus, per_gpu, a.max_gpu_gib, hang_s=a.hang_s, retry_hung=a.retry_hung,
                   kill_grace_s=a.kill_grace_s)


if __name__ == "__main__":
    raise SystemExit(main())

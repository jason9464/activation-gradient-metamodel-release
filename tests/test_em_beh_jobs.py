"""beh_jobs: the job table (names as e2_generate writes them, waits on the control students), and the worker loop
(locks, retries, aside of partial files, stale locks) with stand-in commands."""
import json
import os
import socket
import sys

import pytest

from main_run.em import beh_jobs as m


@pytest.fixture
def roots(tmp_path, monkeypatch):
    for name, sub in (("R", "runs"), ("L", "logs"), ("B", "readers"), ("A", "agm"), ("CAND", "cand")):
        monkeypatch.setattr(m, name, tmp_path / sub)
    monkeypatch.setattr(m, "JOBS", tmp_path / "logs" / "jobs")
    (tmp_path / "logs").mkdir()
    for run in m.EM_RUNS:
        e2 = tmp_path / "runs" / run / "e2"
        e2.mkdir(parents=True)
        for stem, *_ in m.DACT.values():
            (e2 / f"{stem}.npz").write_bytes(b"")
    return tmp_path


def test_table(roots):
    jobs = m.all_jobs()
    reads = [j for j in jobs if j["kind"] == "read"]
    cands = [j for j in jobs if j["kind"] == "cand"]
    assert len(cands) == 8 and len(reads) == 80
    assert [j["prio"] for j in jobs] == sorted(j["prio"] for j in jobs)
    for j in jobs:
        assert "--gpus" not in j["cmd"] and j["cmd"][j["cmd"].index("--device") + 1] == "cuda:0"
        if m.is_ctx(j["run"]):
            assert j["needs"] == [str(m.L / f"CTX_{j['run']}_DONE")]
            assert j["skip_if"] == [str(m.L / f"CTX_{j['run']}_FAILED")]
    names = {os.path.basename(j["outputs"][0]) for j in reads}
    for n in ("ckptdiff_rl_beh_c040_behavioural_dact_evalq.jsonl", "dwx_rl_beh_c040_safety_dact_neutral.jsonl",
              "ckptdiff_rl_beh_c040_behavioural_no_vector_ctl.jsonl", "rl_beh_c040_safety_grad_w1.jsonl",
              "ckptdiff_sft_beh_7082_behavioural_random_ctl.jsonl"):
        assert n in names
    # no injection controls and no raw-gradient inputs for the control students (they save no gradients)
    assert not [j for j in reads if m.is_ctx(j["run"]) and ("_ctl" in j["id"] or "grad_" in j["id"])]


def test_names_match_e2_generate():
    e2g = pytest.importorskip("main_run.reader.e2_generate")
    for cond in ("correct", "no_vector", "random"):
        for tag in ("dact_evalq", "ctl", "grad_cum"):
            assert m.reading_name("rl_beh_c040", "safety", "ckptdiff_", tag, cond) == \
                "ckptdiff_" + e2g.out_name("rl_beh_c040", "safety", tag, cond)
            assert m.reading_name("rl_beh_c040", "safety", "", tag, cond) == e2g.out_name("rl_beh_c040", "safety", tag, cond)
    assert e2g.output_reader_name("rl_beh_c040", {"transform": {"variant": "dwx"}}) == "dwx_rl_beh_c040"


def _stand_in(monkeypatch, fail_ids=()):
    real = m.all_jobs

    def jobs():
        out = []
        for j in real():
            j = dict(j)
            if j["id"] in fail_ids:
                code = "import sys; open(sys.argv[1] + '_settings.json', 'w').write('x'); sys.exit(3)"
                j["cmd"] = [sys.executable, "-c", code, j["outputs"][0][: -len(".jsonl")]]
            else:
                code = "import sys, pathlib\nfor p in sys.argv[1:]:\n pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(p).write_text('ok')"
                j["cmd"] = [sys.executable, "-c", code, *j["outputs"]]
            out.append(j)
        return out

    monkeypatch.setattr(m, "all_jobs", jobs)


def test_worker_runs_waits_and_skips(roots, monkeypatch):
    _stand_in(monkeypatch)
    (m.L / "CTX_em_ctx_s42_DONE").write_text("")
    (m.L / "CTX_em_ctx_s1_FAILED").write_text("")
    assert m.work("0", poll_s=0, log_fn=lambda s: None) == 0
    st = {jid: s for _, s, jid in m.status()["rows"]}
    assert all(s == "done" for jid, s in st.items() if "__em_ctx_s1__" not in jid)
    assert all(s == "skipped" for jid, s in st.items() if "__em_ctx_s1__" in jid)


def test_retry_then_failed_with_aside(roots, monkeypatch):
    bad = "P1__em_s42__ckptdiff_rl_beh_c040_behavioural_dact_evalq"
    _stand_in(monkeypatch, fail_ids=(bad,))
    for f in ("CTX_em_ctx_s42_FAILED", "CTX_em_ctx_s1_FAILED"):
        (m.L / f).write_text("")
    m.work("0", poll_s=0, log_fn=lambda s: None)
    assert (m.JOBS / f"{bad}.failed").exists() and (m.JOBS / f"{bad}.tries").read_text() == "2"
    e2 = m.R / "em_s42" / "e2"
    assert len(list(e2.glob("ckptdiff_rl_beh_c040_behavioural_dact_evalq_settings.json.aside_*"))) >= 1
    st = {jid: s for _, s, jid in m.status()["rows"]}
    assert st[bad] == "failed" and sum(s == "done" for s in st.values()) > 40


def test_stale_lock_cleared(roots):
    job = m.all_jobs()[0]
    m.JOBS.mkdir(parents=True)
    lock = m.JOBS / f"{job['id']}.lock"
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps(dict(host=socket.gethostname(), pid=2**22 + 12345, gpu="0")))
    other = m.JOBS / f"{m.all_jobs()[1]['id']}.lock"
    other.mkdir()
    (other / "owner.json").write_text(json.dumps(dict(host="another-pod", pid=1, gpu="0")))
    assert m.job_state(job) == "running"
    assert m.clear_stale_locks() == [lock.name]
    assert m.job_state(job) == "ready" and other.exists()

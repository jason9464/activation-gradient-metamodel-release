"""CPU tests of online_save (finalise, coverage, resume, loader, finalize command) and train_subliminal guards."""
import json, os, subprocess, sys, tempfile, time
import numpy as np
M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_run")
sys.path.insert(0, M)
import online_save as os_
from agm import store

names, D, B = ["full_grad_L0", "tag_grad_L0", "full_act_L5", "tag_act_L5", "pos0_act_L5"], 16, 4
md = store.index_metadata({"test": "1"})
log = lambda *a: None
passed = []

def ok(name):
    passed.append(name); print("  pass:", name)

def write_part(bd, step):
    rng = np.random.default_rng(step)
    u = rng.standard_normal((B, len(names), D)).astype(np.float16)
    n = (rng.random((B, len(names))) + step).astype(np.float32)
    metas = [dict(global_step=step, micro_batch=0, row_in_batch=b, line_no=step * 10 + b, doc_id=f"d{step}_{b}",
                  arm="dolphin", source="dolphin", run_id="t", seed=1, epoch=0, lr=1e-4, n_tokens=100, n_label=40,
                  n_resp=40, tag_pos=59, truncated=False, N_step=160, normaliser=160.0) for b in range(B)]
    os.makedirs(os.path.join(bd, "parts"), exist_ok=True)
    with open(os_.part_path(bd, step), "wb") as f:
        np.savez(f, unit=u, norms=n, meta=np.array(json.dumps(metas)))
    return u, n

def build(root, spec=((1, 10, 10), (11, 20, 20), (21, 30, 24)), finalise=((1, 10),), ckpts=(10, 20, 30)):
    gd = os.path.join(root, "grads"); os.makedirs(gd)
    store.atomic_write_json(os.path.join(gd, "layout.json"), dict(names=names, d_model=D, grad_layers=[0], act_layers=[5],
                            quantities=["full", "tag"], index_metadata=md, first_step=1))
    ref = {}
    for s, e, upto in spec:
        bd = os.path.join(gd, os_.block_name(s, e))
        for t in range(s, upto + 1):
            ref[t] = write_part(bd, t)
    for s, e in finalise:
        os_.finalize_block(os.path.join(gd, os_.block_name(s, e)), names, D, md, s, e)
    ck = os.path.join(root, "checkpoints")
    for K in ckpts:
        c = os.path.join(ck, f"checkpoint-{K}"); os.makedirs(c)
        for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors"):
            open(os.path.join(c, f), "w").close()
    return gd, ref, sorted((os.path.join(ck, d) for d in os.listdir(ck)), key=lambda p: int(p.rsplit("-", 1)[1]))

def check_store(gd, ref, last, **kw):
    gs = os_.GradsStore(gd, **kw)
    idx = gs.index
    assert sorted(idx["global_step"].unique()) == list(range(1, last + 1)), idx["global_step"].unique()
    assert len(idx) == B * last and idx["doc_id"].is_unique
    for col in os_.INDEX_COLUMNS:
        assert col in idx.columns, col
    for t in (1, min(7, last), last):
        rows = idx[idx["global_step"] == t]
        for a, nm in enumerate(names):
            v = gs.vectors(rows, nm)
            u, n = ref[t]
            assert np.allclose(v, u[:, a].astype(np.float32) * n[:, a:a + 1]), (t, nm)
    return gs

# 1. previous resume cases (coverage, pick, prepare_resume with failure marker, straddling blocks)
with tempfile.TemporaryDirectory() as root:
    gd, ref, ck = build(root)
    assert os_.coverage_ok(gd, 20)[0] and not os_.coverage_ok(gd, 30)[0]
    assert os_.pick_resume_checkpoint(ck, gd, log).endswith("checkpoint-20")
    open(os.path.join(gd, "FAILED_ENOSPC"), "w").write("{}")
    os_.prepare_resume(gd, 20, names, D, md, log)
    assert [b[:2] for b in os_.list_blocks(gd)] == [(1, 10), (11, 20)] and not os.path.exists(os.path.join(gd, "FAILED_ENOSPC"))
    gs = check_store(gd, ref, 20)
    assert gs.index["truncated"].dtype == bool and set(gs.index["source"]) == {"dolphin"}
    ok("resume at the latest covered checkpoint, failure marker cleared, new index columns")
with tempfile.TemporaryDirectory() as root:
    gd, ref, ck = build(root)
    os_.prepare_resume(gd, 15, names, D, md, log)
    assert [b[:2] for b in os_.list_blocks(gd)] == [(1, 10), (11, 15)]
    check_store(gd, ref, 15)
    os_.prepare_resume(gd, 5, names, D, md, log)
    assert [b[:2] for b in os_.list_blocks(gd)] == [(1, 5)]
    check_store(gd, ref, 5)
    ok("straddling open and finalised blocks are cut at the resume step")

# 2. coverage: a missing whole block before K, a missing first block (review finding)
with tempfile.TemporaryDirectory() as root:
    gd, ref, ck = build(root, spec=((1, 10, 10), (11, 20, 20), (21, 30, 30)), finalise=((1, 10), (11, 20), (21, 30)),
                        ckpts=(10, 20, 30, 40))
    r = os_.coverage_ok(gd, 40)
    assert r == (False, "saved steps end at 30, before 40"), r
    assert os_.pick_resume_checkpoint(ck, gd, log).endswith("checkpoint-30")
    ok("checkpoint-40 without steps 31..40 is rejected")
with tempfile.TemporaryDirectory() as root:
    gd, ref, ck = build(root, spec=((11, 20, 20),), finalise=((11, 20),), ckpts=(20,))
    assert os_.coverage_ok(gd, 20)[0] is False
    try:
        os_.pick_resume_checkpoint(ck, gd, log); raise AssertionError("no error")
    except SystemExit:
        pass
    try:
        os_.GradsStore(gd); raise AssertionError("no error")
    except RuntimeError as e:
        assert "first finalised block starts at 11" in str(e)
    ok("a run without steps 1..10 is rejected by coverage and by the loader")

# 3. loader refuses open blocks; finalize command; resume after it
with tempfile.TemporaryDirectory() as root:
    gd, ref, ck = build(root, spec=((1, 10, 10), (11, 20, 20), (21, 30, 30), (31, 40, 37)),
                        finalise=((1, 10), (11, 20), (21, 30)), ckpts=(20, 30))
    try:
        os_.GradsStore(gd); raise AssertionError("no error")
    except RuntimeError as e:
        assert "unfinalised blocks ['block_000031_000040']" in str(e), e
    gs = os_.GradsStore(gd, allow_open=True)
    assert gs.open_blocks == ["block_000031_000040"] and gs.index["global_step"].max() == 30
    ok("loader refuses a trailing open block; allow_open skips it and lists it")
    store.atomic_write_json(os.path.join(root, "RUN.json"), dict(segments=[dict(pid=10**7, host=os.uname().nodename)]))
    res = os_.finalize_open_blocks(root, log=lambda *a: None)
    assert [b[:2] for b in os_.list_blocks(gd)] == [(1, 10), (11, 20), (21, 30), (31, 37)], os_.list_blocks(gd)
    check_store(gd, ref, 37)
    ok("finalize command closes block 31..37 from its parts")
    os_.prepare_resume(gd, 30, names, D, md, log)
    assert [b[:2] for b in os_.list_blocks(gd)] == [(1, 10), (11, 20), (21, 30)]
    check_store(gd, ref, 30)
    ok("resume after the finalize command drops the steps after the checkpoint")
with tempfile.TemporaryDirectory() as root:
    gd, ref, ck = build(root, spec=((1, 10, 10), (11, 20, 13)), finalise=((1, 10),), ckpts=(10,))
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "train_subliminal.py"])
    time.sleep(1.5)  # let the child exec so /proc/<pid>/cmdline is its own
    try:
        store.atomic_write_json(os.path.join(root, "RUN.json"), dict(segments=[dict(pid=p.pid, host=os.uname().nodename)]))
        try:
            os_.finalize_open_blocks(root); raise AssertionError("no refusal")
        except SystemExit as e:
            assert "refusing" in str(e)
    finally:
        p.kill(); p.wait()
    os_.finalize_open_blocks(root, log=lambda *a: None)
    check_store(gd, ref, 13)
    ok("finalize refuses while a train_subliminal process of the run is alive")

# 4. first pyarrow use inside a worker thread after torch (review finding: segfault)
code = f"""
import sys, threading, json, os, tempfile, numpy as np
import torch
sys.path.insert(0, {M!r})
import online_save as os_
from agm import store
names, D = ["full_grad_L0"], 8
with tempfile.TemporaryDirectory() as d:
    bd = os.path.join(d, os_.block_name(1, 2)); os.makedirs(os.path.join(bd, "parts"))
    for t in (1, 2):
        metas = [dict(global_step=t, micro_batch=0, row_in_batch=0, line_no=t, doc_id=str(t), arm="a", source="a",
                      run_id="r", seed=0, epoch=0, lr=0.0, n_tokens=3, n_label=1, n_resp=1, tag_pos=1, truncated=False,
                      N_step=1, normaliser=1.0)]
        np.savez(os_.part_path(bd, t), unit=np.ones((1, 1, D), np.float16), norms=np.ones((1, 1), np.float32),
                 meta=np.array(json.dumps(metas)))
    th = threading.Thread(target=os_.finalize_block, args=(bd, names, D, store.index_metadata({{}}), 1, 2)); th.start(); th.join()
    import pyarrow.parquet as pq
    print(pq.read_table(os.path.join(bd, "index.parquet")).num_rows)
"""
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
assert r.returncode == 0 and r.stdout.strip() == "2", (r.returncode, r.stderr[-500:])
ok("finaliser thread after torch: exit 0")

# 5. train_subliminal guards (before any model load)
import train_subliminal as ts
DATA = ts.DATA["dolphin"]
def fake_run(root, save):
    a = ts.parse_args(["--arm", "dolphin", "--out", root])
    store.atomic_write_json(os.path.join(root, "RUN.json"), dict(settings=ts.settings_of(a, os.path.abspath(DATA))))
    c = os.path.join(root, "checkpoints", "checkpoint-500"); os.makedirs(c)
    if save:
        os.makedirs(os.path.join(root, "grads")); open(os.path.join(root, "grads", "layout.json"), "w").write("{}")
def expect_exit(argv, text):
    try:
        ts.main(argv); raise AssertionError("no exit")
    except SystemExit as e:
        assert text in str(e), str(e)
with tempfile.TemporaryDirectory() as root:
    fake_run(root, save=True)
    expect_exit(["--arm", "dolphin", "--out", root, "--resume"], "resume it with --save-grads")
    ok("--resume without --save-grads of a saving run is refused")
with tempfile.TemporaryDirectory() as root:
    fake_run(root, save=False)
    expect_exit(["--arm", "dolphin", "--out", root, "--resume", "--save-grads"], "cannot start part way")
    ok("--save-grads on a run that did not save is refused")
    expect_exit(["--arm", "dolphin", "--out", root, "--resume", "--stop-step", "500"], "nothing to do")
    ok("--resume at or past --stop-step is refused")
with tempfile.TemporaryDirectory() as root:
    a = ts.parse_args(["--arm", "dolphin", "--out", root])
    old = ts.settings_of(a, os.path.abspath(DATA)); old.pop("stop_step")  # a pilot written before --stop-step
    new = ts.settings_of(ts.parse_args(["--arm", "dolphin", "--out", root]), os.path.abspath(DATA))
    assert {k for k, v in new.items() if old.get(k) != v} == set()
    ok("pilot RUN.json without stop_step still resumes (no settings difference)")

# 6. row_info columns on real rows (tokenizer only)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B"); tok.padding_side = "right"
rows, _ = ts.load_rows(DATA, 200)
for max_len in (256, 110):
    ids, masks, st = ts.tokenize_rows(tok, rows, max_len, "public")
    ri = ts.row_info(tok, rows, ids, masks, "public", "dolphin")
    assert ri["n_resp"] == ri["n_label"], "public: every response position is a label"
    assert sum(ri["truncated"]) == st["truncated"], (sum(ri["truncated"]), st["truncated"])
    assert set(ri["source"]) == {"dolphin"}
    print(f"  max_len {max_len}: truncated rows {st['truncated']} of {len(rows)}")
ok("row_info n_resp, truncated (matches tokenize_rows count at max_len 256 and 110), source")
print(f"CPU tests passed: {len(passed)}")

"""main_run/beh_reader/new_readers_suite.py on tiny synthetic files (plan docs/notes/20261029_new_readers_suite_plan.md
sections 5, 6). CPU only: no GPU, no 14B model, no API; every file lives under the pytest basetemp.

  cd <scratch> && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_new_readers_suite.py --basetemp <scratch>/new_readers_suite_test

What is checked: row order kept and recorded (ORDER.json pass_row_order, row ids against the source), layer slicing
bit for bit, the activation window sums of E2 against a brute-force float64 sum over the stored rows, the E3 twin rows
against the act_vectors file, the E4 twin byte copies and the rare_yesno order-hash rule, the slot assembly (pseudo-layer
order, a document listed twice counts twice), the slot document list, the E2/E3 metric rules, and the summary table
shape (experiments x metrics x columns) plus the figure files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "main_run" / "beh_reader"))
import new_readers_suite as S  # noqa: E402

D = 8
L40 = [5, 10, 20, 30]  # a "40-layer" source with one layer more than the three kept
GRAD_SIGN = "inputs = -(sum of +dL/dh): the descent direction"


def npz(path: Path, X, steps, windows, layers, meta, wlo=None, whi=None, n_docs=None, rule="export"):
    n, W = X.shape[:2]
    d = dict(inputs=X.astype(np.float32), steps=list(steps), windows=list(windows), layers=list(layers),
             window_lo=np.zeros((n, W), np.int64) if wlo is None else wlo,
             window_hi=np.zeros((n, W), np.int64) if whi is None else whi,
             n_docs=np.ones((n, W), np.int64) if n_docs is None else n_docs, meta=meta, order_rule=rule)
    S.write_npz(path, d)
    return path


def bundle_file(path, n, layers=L40, kind="grad", seed=0, doc_ids=True, rule="export"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1, len(layers), D))
    meta = dict(kind=kind, sign=GRAD_SIGN, bundle_ids=[f"b{i}" for i in range(n)], family=["rare"] * n)
    if doc_ids:
        meta["doc_ids"] = [[f"d{i}a", f"d{i}b"] for i in range(n)]
    return npz(path, X, range(n), ["bundle"], layers, meta, rule=rule)


# ---------------------------------------------------------------- basic helpers
def test_order_hash_and_subset(tmp_path):
    p = bundle_file(tmp_path / "a.npz", 5)
    d = S.read_npz(p)
    assert d["order_rule"] == "export"
    sub = S.subset(d, [3, 1], None, [30, 10])
    assert sub["steps"] == [3, 1] and sub["layers"] == [30, 10]
    assert sub["meta"]["bundle_ids"] == ["b3", "b1"] and sub["meta"]["doc_ids"] == [["d3a", "d3b"], ["d1a", "d1b"]]
    np.testing.assert_array_equal(sub["inputs"][0, 0, 0], d["inputs"][3, 0, L40.index(30)])
    np.testing.assert_array_equal(sub["inputs"][1, 0, 1], d["inputs"][1, 0, L40.index(10)])
    # a rare_yesno act file (its own order-hash rule) is read too, and a broken hash is refused
    q = bundle_file(tmp_path / "act.npz", 4, layers=[10, 20, 30], kind="act", rule="rare_yesno_act")
    assert S.read_npz(q)["order_rule"] == "rare_yesno_act"
    z = dict(np.load(q))
    m = json.loads(str(z["meta"]))
    m["order_hash"] = "0" * 40
    z["meta"] = np.asarray(json.dumps(m))
    np.savez(tmp_path / "bad.npz", **z)
    with pytest.raises(ValueError):
        S.read_npz(tmp_path / "bad.npz")


def test_norm_lora_key():
    assert S.norm_lora_key("a.layers.0.mlp.down_proj.lora_A.default.weight") == "a.layers.0.mlp.down_proj.lora_A.weight"
    assert S.norm_lora_key("a.layers.0.mlp.down_proj.lora_B.weight") == "a.layers.0.mlp.down_proj.lora_B.weight"
    assert S.norm_lora_key("a.layers.0.self_attn.q_proj.base_layer.weight") == "a.layers.0.self_attn.q_proj.base_layer.weight"


# ---------------------------------------------------------------- E4 / E5 (grad3 slices, twin copies)
def e4_sources(tmp_path):
    perdoc, act = tmp_path / "perdoc", tmp_path / "act"
    perdoc.mkdir()
    act.mkdir()
    src = dict(e4_perdoc=perdoc, e4_bundle=tmp_path / "inputs_correct.npz", e4_act_dir=act,
               e4_slots=tmp_path / "slots_beh.npz")
    bundle_file(src["e4_bundle"], 6, doc_ids=False, seed=1)  # inputs_correct.npz has no doc_ids
    rng = np.random.default_rng(2)
    docs = {"bundle": [[f"doc{i}", f"doc{i + 1}", f"doc{i}"] for i in range(6)],  # doc{i} listed twice
            "doc": [[f"doc{i}"] for i in range(7)], "showing": [["doc0", "doc2"], ["doc4"]],
            "nonshowing": [["doc1", "doc3"], ["doc5", "doc6"]]}
    bids = {"bundle": [f"b{i}" for i in range(6)], "doc": [f"b{i // 2}" for i in range(7)], "showing": ["b0", "b1"],
            "nonshowing": ["b0", "b1"]}
    for k in ("doc", "showing", "nonshowing"):
        n = len(docs[k])
        npz(perdoc / f"inputs_{k}.npz", rng.normal(size=(n, 1, len(L40), D)), range(n), ["bundle"], L40,
            dict(kind="grad", sign=GRAD_SIGN, bundle_ids=bids[k], doc_ids=docs[k]))
    for k in S.E4_KINDS:
        n = len(docs[k])
        npz(act / f"inputs_act_{k}.npz", rng.normal(size=(n, 1, 3, D)), range(n), ["bundle"], [10, 20, 30],
            dict(kind="act", sign="activation sums, no negation", bundle_ids=bids[k], doc_ids=docs[k]),
            rule="rare_yesno_act")
    return src, docs


def test_e4_grad3_slices_and_twin_copies(tmp_path):
    src, _ = e4_sources(tmp_path)
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e4", "grad3", src=src) == 0
    for k in S.E4_KINDS:
        got = S.read_npz(root / "e4" / "grad3" / "inputs" / f"inputs_{k}.npz")
        ref = S.read_npz(S.e4_grad_path(src, k))
        assert got["layers"] == [10, 20, 30] and got["steps"] == ref["steps"]
        np.testing.assert_array_equal(got["inputs"], ref["inputs"][:, :, [L40.index(L) for L in (10, 20, 30)]])
        assert got["meta"]["bundle_ids"] == ref["meta"]["bundle_ids"]
    order = json.loads((root / "e4" / "grad3" / "inputs" / "ORDER.json").read_text())
    assert all(e["pass_row_order"] for e in order["files"]) and len(order["files"]) == 4
    assert S.cmd_build(root, "e4", "twin", src=src) == 0
    tw = root / "e4" / "twin" / "inputs"
    for k in S.E4_KINDS:
        assert (tw / f"inputs_act_{k}.npz").read_bytes() == S.e4_act_path(src, k).read_bytes()
        a = S.read_npz(tw / f"inputs_act_{k}.npz")
        e = S.read_npz(tw / f"inputs_{k}.npz")  # same rows, e2_generate's order hash
        assert a["order_rule"] == "rare_yesno_act" and e["order_rule"] == "export"
        np.testing.assert_array_equal(a["inputs"], e["inputs"])
    order = json.loads((tw / "ORDER.json").read_text())
    assert all(e["pass_row_order"] for e in order["files"])
    assert all(e.get("pass_copy_sha256", True) for e in order["files"])
    # a second build into the same folder is refused
    assert S.cmd_build(root, "e4", "grad3", src=src) == 1


def test_row_order_failure_is_caught(tmp_path):
    src, _ = e4_sources(tmp_path)
    # an act file whose bundle order differs from the gradient file's
    p = S.e4_act_path(src, "showing")
    d = S.read_npz(p)
    d = S.subset(d, [1, 0])
    d["order_rule"] = "rare_yesno_act"
    p.rename(p.with_name("old.npz"))
    S.write_npz(p, d)
    assert S.cmd_build(tmp_path / "suite", "e4", "twin", src=src) == 1
    assert not (tmp_path / "suite" / "e4" / "twin" / "inputs").exists()


def write_slots(path, docs: dict):
    """slots_<set>.npz in slot_extract's merge format; docs = {doc_id: [3 slots, 3 layers, d] float vectors}."""
    ids = list(docs)
    V = np.stack([docs[i] for i in ids]).astype(np.float32)
    nrm = np.linalg.norm(V, axis=-1)
    unit = (V / nrm[..., None]).astype(np.float16)
    np.savez(path, unit=unit, norm=nrm.astype(np.float32), doc_ids=np.array(ids), n_tokens=np.full(len(ids), 20),
             n_chunks=np.ones(len(ids), np.int64), slots=np.array(S.SLOTS), layers=np.array(S.L3),
             sign=np.array("-dL/dh test"))
    return unit, nrm


def test_e4_slot_assembly(tmp_path):
    src, docs = e4_sources(tmp_path)
    rng = np.random.default_rng(5)
    vec = {f"doc{i}": rng.normal(size=(3, 3, D)) for i in range(8)}
    unit, nrm = write_slots(src["e4_slots"], vec)
    # gradient rows consistent with the slot full rows (the build gates on their cosine)
    idx = {f"doc{i}": i for i in range(8)}
    fp = lambda d: unit[idx[d]].astype(np.float32) * nrm[idx[d]][..., None]  # noqa: E731
    for k in S.E4_KINDS:
        g = S.read_npz(S.e4_grad_path(src, k))
        X = g["inputs"].copy()
        for i, r in enumerate(docs[k]):
            full = np.sum([fp(d)[0] for d in r], axis=0)
            for li, L in enumerate(S.L3):
                X[i, 0, L40.index(L)] = full[li]
        g["inputs"] = X
        S.e4_grad_path(src, k).rename(S.e4_grad_path(src, k).with_name(f"old_{k}.npz"))
        S.write_npz(S.e4_grad_path(src, k), g)
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e4", "slot", src=src) == 0
    lines = [json.loads(x) for x in (root / "e4" / "slot" / "docs.jsonl").read_text().splitlines()]
    assert sorted(x["doc_id"] for x in lines) == sorted({d for rows in docs.values() for r in rows for d in r})
    for k in S.E4_KINDS:
        got = S.read_npz(root / "e4" / "slot" / "inputs" / f"inputs_{k}.npz")
        assert got["layers"] == list(S.SLOT_LAYERS) and got["meta"]["doc_ids"] == docs[k]
        for i, r in enumerate(docs[k]):
            want = np.sum([fp(d) for d in r], axis=0).reshape(9, D)  # slot-major: full, first10, last10
            np.testing.assert_allclose(got["inputs"][i, 0], want, rtol=1e-5, atol=1e-5)
        # first10 of layer 20 sits at pseudo layer 1020
        i = 0
        np.testing.assert_allclose(got["inputs"][i, 0, list(S.SLOT_LAYERS).index(1020)],
                                   np.sum([fp(d)[1, 1] for d in docs[k][i]], axis=0), rtol=1e-5, atol=1e-5)
    rep = json.loads((root / "e4" / "slot" / "inputs" / "BUILD.json").read_text())
    assert rep["passed"] and rep["checks"]["bundle_full_vs_gradient"]["cos"]["min"] > 0.9999


def test_slot_pending_without_slot_files(tmp_path):
    src, _ = e4_sources(tmp_path)
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e4", "slot", src=src) == 0
    pend = json.loads((root / "e4" / "slot" / "BUILD_PENDING.json").read_text())
    assert pend["status"] == "extraction pending" and pend["missing_groups"] == ["e4_base"]
    assert "slot-extract --docs" in pend["command"]
    assert not (root / "e4" / "slot" / "inputs").exists()


def test_e5_grad3_and_twin(tmp_path):
    e5 = tmp_path / "already_learned"
    (e5).mkdir()
    (e5 / "SELECT.json").write_text(json.dumps(dict(order=["x1", "x2"])))
    rng = np.random.default_rng(3)
    meta = dict(step_kind=["doc", "doc", "sum"], set_behaviour=["x1", "x1", "x1"],
                bundle_ids=["q1", "q1", "q1"], doc_ids=[["a"], ["b"], ["a", "b"]])
    for m in ("base", "x1", "x2"):
        (e5 / m).mkdir()
        npz(e5 / m / f"inputs_grad_{m}.npz", rng.normal(size=(3, 1, len(L40), D)), range(3), ["bundle"], L40,
            dict(meta, kind="grad", sign=GRAD_SIGN))
        npz(e5 / m / f"inputs_act_{m}.npz", rng.normal(size=(3, 1, 3, D)), range(3), ["bundle"], [10, 20, 30],
            dict(meta, kind="act", sign="activation sums"))
    src = dict(e5_dir=e5)
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e5", "grad3", src=src) == 0
    assert S.cmd_build(root, "e5", "twin", src=src) == 0
    for m in ("base", "x1", "x2"):
        g = S.read_npz(root / "e5" / "grad3" / "inputs" / m / f"inputs_grad_{m}.npz")
        assert g["layers"] == [10, 20, 30] and g["meta"]["step_kind"] == meta["step_kind"]
        assert (root / "e5" / "twin" / "inputs" / m / f"inputs_act_{m}.npz").read_bytes() == \
            (e5 / m / f"inputs_act_{m}.npz").read_bytes()


# ---------------------------------------------------------------- E2 twin: activation window sums vs brute force
def fake_store(run: Path, steps_per_block=5, n_blocks=2, docs_per_step=3, layers=(10, 20, 30), seed=7):
    import pandas as pd

    rng = np.random.default_rng(seed)
    g = run / "grads"
    g.mkdir(parents=True)
    names = [f"full_act_L{L}" for L in layers]
    (g / "layout.json").write_text(json.dumps(dict(layout_version=1, names=names, d_model=D, first_step=1,
                                                   act_layers=list(layers), index_metadata={})))
    rows = {}
    for b in range(n_blocks):
        s, e = b * steps_per_block + 1, (b + 1) * steps_per_block
        bd = g / f"block_{s:06d}_{e:06d}"
        bd.mkdir()
        steps = np.repeat(np.arange(s, e + 1), docs_per_step)
        df = pd.DataFrame(dict(global_step=steps, row=np.arange(len(steps))))
        for nm in names:
            V = rng.normal(size=(len(steps), D))
            nr = np.linalg.norm(V, axis=1)
            U = (V / nr[:, None]).astype(np.float16)
            np.save(bd / f"{nm}.npy", U)
            df[f"norm_{nm}"] = nr.astype(np.float32)
            rows.setdefault(nm, []).append((steps, U.astype(np.float64) * nr.astype(np.float32).astype(np.float64)[:, None]))
        df.to_parquet(bd / "index.parquet")
        (bd / "DONE.json").write_text("{}")
    (run / "RUN.json").write_text(json.dumps(dict(settings=dict(arm="dolphin", seed=1))))
    return rows


def test_e2_activation_window_sums_equal_brute_force(tmp_path):
    run = tmp_path / "student"
    rows = fake_store(run)
    steps, w = [2, 5, 7, 10], 4
    d, chk = S.e2_act_windows(run, steps, window=("w4", w))
    assert all(ok for _, ok in S.passes(chk)), chk
    for i, n in enumerate(steps):
        lo = max(n - w + 1, 1)
        assert int(d["window_lo"][i, 0]) == lo and int(d["window_hi"][i, 0]) == n
        assert int(d["n_docs"][i, 0]) == 3 * (n - lo + 1)
        for li, L in enumerate((10, 20, 30)):
            ref = np.zeros(D)
            for st, V in rows[f"full_act_L{L}"]:
                m = (st >= lo) & (st <= n)
                ref += V[m].sum(0)
            np.testing.assert_allclose(d["inputs"][i, 0, li], ref, rtol=1e-5, atol=1e-5)
    assert d["meta"]["kind"] == "act" and d["windows"] == ["w4"]


def test_e2_build_grad3_and_twin(tmp_path):
    runs = {}
    for k, arm in enumerate(S.E2_ARMS):
        run = tmp_path / arm
        fake_store(run, seed=10 + k)
        (run / "e2").mkdir()
        steps = [1, 5, 10]
        X = np.random.default_rng(k).normal(size=(3, 2, len(L40), D))
        wlab = ["w1", "w1000"]
        wlo = np.array([[1, 1], [5, 1], [10, 1]])
        whi = np.array([[1, 1], [5, 5], [10, 10]])
        nd = np.array([[3, 3], [3, 15], [3, 30]])
        npz(run / "e2" / "inputs_every100.npz", X, steps, wlab, L40, dict(kind="grad", sign=GRAD_SIGN, run=str(run)),
            wlo=wlo, whi=whi, n_docs=nd)
        npz(run / "e2" / "adam_inputs_every100.npz", X[:, :1], steps, ["adam"], L40,
            dict(kind="grad", sign=GRAD_SIGN, run=str(run), transform=dict(variant="adam")),
            wlo=np.ones((3, 1), np.int64), whi=np.array([[1], [5], [10]]), n_docs=nd[:, 1:])
        runs[arm] = run
    src = dict(e2_runs=runs)
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e2", "grad3", src=src, e2_steps=(5, 10)) == 0
    g = S.read_npz(root / "e2" / "grad3" / "inputs" / "e2_dolphin_w1000.npz")
    assert g["windows"] == ["w1000"] and g["steps"] == [5, 10] and g["layers"] == [10, 20, 30]
    a = S.read_npz(root / "e2" / "grad3" / "inputs" / "e2_dolphin_adam.npz")
    assert a["meta"]["transform"]["variant"] == "adam" and a["windows"] == ["adam"]
    # the twin's windows must equal the gradient file's (w1000 covers 1..n below step 1,000, as in the real file)
    assert S.cmd_build(root, "e2", "twin", src=src, e2_steps=(5, 10)) == 0
    t = S.read_npz(root / "e2" / "twin" / "inputs" / "e2_clean_w1000.npz")
    assert t["meta"]["kind"] == "act" and t["steps"] == [5, 10]
    rep = json.loads((root / "e2" / "twin" / "inputs" / "BUILD.json").read_text())
    assert rep["passed"] and rep["checks"]["clean_same_windows_as_gradient_w1000"]["pass_n_docs"]


def test_e2_slot_refused(tmp_path):
    with pytest.raises(SystemExit):
        S.cmd_build(tmp_path / "suite", "e2", "slot", src=dict(e2_runs={}))


# ---------------------------------------------------------------- E3
def e3_sources(tmp_path, steps=(100, 500)):
    act = tmp_path / "ao_act_cleanbatch"
    act.mkdir()
    rng = np.random.default_rng(11)
    runs, docmap = {}, {}
    for st in S.STUDENTS:
        run = tmp_path / f"cb_{st}" / "train"
        (run / "e2").mkdir(parents=True)
        (run / "cleanbatch").mkdir()
        recs = []
        for t in steps:
            ids = [[f"{st}-{t}-{b}-{k}" for k in range(4)] for b in S.E3_BATCHES]
            docmap[(st, t)] = ids
            recs.append(dict(step=t, real_doc_id=ids[0], foreign_doc_id=ids[1:]))
        (run / "cleanbatch" / "probe_meta.json").write_text(json.dumps(dict(records=recs)))
        all_steps = [2] + list(steps)
        for var in ("mfree", "grad"):
            X = rng.normal(size=(len(all_steps), 4, len(L40), D))
            npz(run / "e2" / f"cleanbatch_inputs_{var}.npz", X, all_steps, [f"{b}_{var}" for b in S.E3_BATCHES], L40,
                dict(kind="grad", sign=GRAD_SIGN), wlo=np.repeat(np.array(all_steps)[:, None], 4, 1),
                whi=np.repeat(np.array(all_steps)[:, None], 4, 1), n_docs=np.full((len(all_steps), 4), 4))
        runs[st] = run
    for t in steps:
        dv = rng.normal(size=(2, 4, 4, 3, D)).astype(np.float32)
        vec = np.zeros((2, 4, 2, 3, D), np.float32)
        vec[:, :, 0] = dv.astype(np.float64).sum(2)
        vec[:, :, 1] = rng.normal(size=(2, 4, 3, D))
        doc_id = np.array([[docmap[(st, t)][b] for b in range(4)] for st in S.STUDENTS])
        meta = dict(per_student={st: dict(weights=dict(kind="adapter", path=f"/x/{st}/{t}")) for st in S.STUDENTS})
        np.savez(act / f"act_vectors_step{t:06d}.npz", vectors=vec, doc_vectors=dv, doc_id=doc_id,
                 students=np.array(S.STUDENTS), batches=np.array(S.E3_BATCHES),
                 variants=np.array(["act_raw", "act_centred"]), layers=np.array([10, 20, 30]), step=np.int64(t),
                 line_no=np.zeros((2, 4, 4), np.int64), n_tokens=np.full((2, 4, 4), 50),
                 meta=np.asarray(json.dumps(meta)))
    return dict(e3_runs=runs, e3_act_dir=act)


def test_e3_grad3_and_twin(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "E3_STEPS", (100, 500))
    src = e3_sources(tmp_path)
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e3", "grad3", src=src) == 0
    g = S.read_npz(root / "e3" / "grad3" / "inputs" / "e3_dolphin_mfree.npz")
    ref = S.read_npz(src["e3_runs"]["dolphin"] / "e2" / "cleanbatch_inputs_mfree.npz")
    assert g["steps"] == [100, 500] and g["layers"] == [10, 20, 30]
    np.testing.assert_array_equal(g["inputs"], ref["inputs"][1:][:, :, [1, 2, 3]])
    assert S.cmd_build(root, "e3", "twin", src=src) == 0
    t = S.read_npz(root / "e3" / "twin" / "inputs" / "e3_clean_act.npz")
    z = np.load(src["e3_act_dir"] / "act_vectors_step000500.npz")
    np.testing.assert_array_equal(t["inputs"][1], z["vectors"][1, :, 0])
    assert t["windows"] == ["real_act", "f1_act", "f2_act", "f3_act"] and t["meta"]["kind"] == "act"
    assert t["meta"]["batch_doc_ids"][0][0] == [f"clean-100-real-{k}" for k in range(4)]
    rep = json.loads((root / "e3" / "twin" / "inputs" / "BUILD.json").read_text())
    assert rep["passed"] and rep["checks"]["dolphin_act"]["foreign_doc_ids_equal_probe"]["checked"]


# ---------------------------------------------------------------- metrics and summary
def test_e2_metrics_rule():
    rows = {"dolphin": [dict(step=500, verdict="NO")] * 30 + [dict(step=1000, verdict="YES")] * 2 +
            [dict(step=1000, verdict="NO")] * 28 + [dict(step=3000, verdict="YES")] * 10 + [dict(step=3000, verdict="NO")] * 20,
            "clean": [dict(step=500, verdict="NO")] * 30 + [dict(step=1000, verdict="YES")] + [dict(step=1000, verdict="NO")] * 29
            + [dict(step=3000, verdict="NO")] * 30}
    m = S.e2_metrics(rows)
    # step 1000: dolphin 2/30 >= 1/30 but < 5 x clean 1/30; step 3000: 10/30 and clean 0
    assert m["first_step"] == 3000
    assert m["dolphin_pre_onset"] == pytest.approx(2 / 60) and m["clean_pre_onset"] == pytest.approx(1 / 60)


def test_e3_pyes_metrics():
    w = ["real_mfree", "f1_mfree", "f2_mfree", "f3_mfree", "real_grad", "f1_grad", "f2_grad", "f3_grad"]
    p = [[0.9, 0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.2, 0.2, 0.4, 0.5, 0.5, 0.5]]
    m = S.e3_pyes_metrics([100, 500], w, p, want_steps=(100, 500), variant="mfree")
    assert m["real"] == pytest.approx(0.5) and m["foreign"] == pytest.approx(0.2) and m["n_real_gt"] == 1
    assert S.e3_pyes_metrics([100, 500], w, p, want_steps=(100, 500), variant="grad")["n_real_gt"] == 0


def test_summary_table_shape_and_plot(tmp_path):
    root = tmp_path / "suite"
    # E2: judged files of grad3 (w1000) and twin
    for r in ("grad3", "twin"):
        for a in S.E2_ARMS:
            d = root / "e2" / r / a / "e2"
            d.mkdir(parents=True)
            rows = [dict(step=t, answer_index=k, verdict="YES" if (a == "dolphin" and t >= 1000 and k < 3) else "NO",
                         key="w1000") for t in (500, 1000, 3000) for k in range(20)]
            (d / f"{r}_preference_w1000_judge_dolphin.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    # E3: P(Yes) files
    for r, v in (("grad3", "mfree"), ("twin", "act")):
        for st in S.STUDENTS:
            d = root / "e3" / r
            d.mkdir(parents=True, exist_ok=True)
            js = dict(steps=list(S.E3_STEPS), windows=[f"{b}_{v}" for b in S.E3_BATCHES],
                      p_yes=[[0.6, 0.1, 0.1, 0.1]] * len(S.E3_STEPS))
            (d / f"pyes_{st}.json").write_text(json.dumps(js))
    # E3: the twin's open reading of the dolphin student, judged (real 3 YES of 10, clean batches none)
    d = root / "e3" / "twin" / "dolphin" / "e2"
    d.mkdir(parents=True)
    for b in S.E3_BATCHES:
        rows = [dict(step=100, answer_index=k, verdict="YES" if (b == "real" and k < 3) else "NO", key=f"{b}_act")
                for k in range(10)]
        (d / f"batch_nrs_twin_preference_{b}_act_judge_dolphin.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in rows))
    # E4: rare_yesno summary of grad3 only
    (root / "e4" / "grad3" / "yesno").mkdir(parents=True)
    (root / "e4" / "grad3" / "yesno" / "summary.json").write_text(json.dumps(dict(results={
        "doc.rare.showing_vs_nonshowing": dict(auroc=0.95), "bundle.rare.pos_vs_neg": dict(auroc=0.7)})))
    # rare_perdoc.py summary --out ROOT/e4/grad3/open: its bundle rows are rl_beh_c040's (here 0.9), never grad3's
    (root / "e4" / "grad3" / "open").mkdir(parents=True)
    (root / "e4" / "grad3" / "open" / "summary.json").write_text(json.dumps(dict(results={
        "bundle_first10": dict(rare=dict(answers=dict(yes_rate=0.9))),
        "doc_by_showing": dict(rare=dict(showing_docs=dict(answers=dict(yes_rate=0.5)),
                                         nonshowing_docs=dict(answers=dict(yes_rate=0.1))))})))
    # references: none for E2/E3/E5, a reader40 E4 file
    ref4 = tmp_path / "ref_rare_yesno.json"
    ref4.write_text(json.dumps(dict(results={"doc.rare.showing_vs_nonshowing": dict(auroc=0.97)})))
    src = dict(ref_e2_reader40={a: tmp_path / "none.jsonl" for a in S.E2_ARMS},
               ref_e3_reader40={s: tmp_path / "none.json" for s in S.STUDENTS}, ref_e3_ao3=tmp_path / "none.json",
               ref_e4_reader40=dict(yesno=ref4, open=tmp_path / "none.json"), ref_e4_ao3=dict(yesno=tmp_path / "x.json"),
               ref_e5_dir=tmp_path / "no_e5")
    assert S.cmd_summary(root, src) == 0
    s = json.loads((root / "summary.json").read_text())
    assert list(s["experiments"]) == list(S.EXPS)
    for e in S.EXPS:
        x = s["experiments"][e]
        assert list(x["table"]) == [k for k, _, _ in S.METRICS[e]]
        assert all(set(row) == set(x["columns"]) for row in x["table"].values())
    assert s["experiments"]["e2"]["columns"] == ["grad3", "twin"]
    assert s["experiments"]["e2"]["table"]["first_step"]["grad3"] == 1000
    assert s["experiments"]["e2"]["table"]["dolphin_pre_onset"]["twin"] == pytest.approx(3 / 40)
    assert s["experiments"]["e3"]["table"]["dolphin.real"]["twin"] == pytest.approx(0.6)
    assert s["experiments"]["e3"]["flags"]["dolphin.real"]["twin-grad3"]["verdict"] == "same"
    assert s["experiments"]["e4"]["columns"] == ["reader40", "grad3"]
    assert s["experiments"]["e4"]["table"]["auroc.bundle.rare.pos_vs_neg"]["reader40"] is None
    assert s["experiments"]["e4"]["table"]["open.rare.showing_docs"]["grad3"] == pytest.approx(0.5)
    assert s["experiments"]["e4"]["table"]["open.rare.bundle"]["grad3"] is None  # not the 40-layer reader's 0.9
    assert s["experiments"]["e3"]["table"]["dolphin.open_real"]["twin"] == pytest.approx(0.3)
    assert s["experiments"]["e3"]["table"]["dolphin.open_foreign"]["twin"] == pytest.approx(0.0)
    assert s["experiments"]["e5"]["columns"] == []
    assert "| metric | grad3 | twin |" in (root / "summary.md").read_text()
    fig = tmp_path / "fig.png"
    assert S.cmd_plot(root, fig) == 0
    assert fig.stat().st_size > 1000
    plotted = json.loads(fig.with_suffix(".json").read_text())["values"]
    assert plotted["e3"]["twin"]["dolphin.real"] == pytest.approx(0.6)


def test_reader_inputs_kind_and_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "E3_STEPS", (100, 500))
    src = e3_sources(tmp_path)
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e3", "twin", src=src) == 0
    assert S.cmd_build(root, "e3", "grad3", src=src) == 0
    t = S.reader_inputs(root / "e3" / "twin" / "inputs" / "e3_dolphin_act.npz", "twin")
    assert t["inputs"].shape == (2, 4, 3, D) and t["windows"][0] == "real_act"
    with pytest.raises(ValueError):
        S.reader_inputs(root / "e3" / "twin" / "inputs" / "e3_dolphin_act.npz", "grad3")
    S.reader_inputs(root / "e3" / "grad3" / "inputs" / "e3_dolphin_grad.npz", "grad3")
    with pytest.raises(ValueError):  # the slot reader needs its pseudo layers
        S.reader_inputs(root / "e3" / "grad3" / "inputs" / "e3_dolphin_grad.npz", "slot")


def test_slot_partial_file_fails(tmp_path):
    src, docs = e4_sources(tmp_path)
    rng = np.random.default_rng(6)
    write_slots(src["e4_slots"], {f"doc{i}": rng.normal(size=(3, 3, D)) for i in range(6)})  # doc6 missing
    root = tmp_path / "suite"
    assert S.cmd_build(root, "e4", "slot", src=src) == 1
    assert not (root / "e4" / "slot" / "inputs").exists()
    rep = json.loads(next((root / "e4" / "slot").glob("BUILD_FAILED_*.json")).read_text())
    assert rep["failed"] == ["checks.docs_list.pass_every_document_extracted"]


def test_slot_positions_and_weights():
    """first10 / last10 over a document cut into chunks, and the hook weights (full = agm.core.position_sets)."""
    pos = S.chunk_slot_positions([6, 12])  # positions 0..17: first10 = 0..9, last10 = 8..17
    np.testing.assert_array_equal(pos[0][0], np.arange(6))
    np.testing.assert_array_equal(pos[1][0], np.arange(4))
    np.testing.assert_array_equal(pos[0][1], np.array([], np.int64))
    np.testing.assert_array_equal(pos[1][1], np.arange(2, 12))
    core, _, _ = S.agm_modules()
    rows = [dict(ids=list(range(12)), plen=4, first=pos[1][0], last=pos[1][1])]
    W = S.slot_weights(core, rows, 14)
    want_full = np.zeros(14, np.float32)
    for s0, e0 in core.position_sets(12, 4, "full", False):
        want_full[s0:e0] = 1
    np.testing.assert_array_equal(W[0, 0], want_full)
    assert W[0, 0, 11] == 0 and W[0, 0, :11].sum() == 11  # full = 0..n-2
    np.testing.assert_array_equal(np.flatnonzero(W[0, 1]), np.arange(4))
    np.testing.assert_array_equal(np.flatnonzero(W[0, 2]), np.arange(2, 12))
    assert W[0, :, 12:].sum() == 0  # padding


_EVAL40 = S.C.eval_dir("rl_beh_c040")
_PERDOC_SUMMARY = S.C.EV / "rare_perdoc" / "summary.json"


@pytest.mark.skipif(not ((_EVAL40 / "readings_correct_judge_truth.jsonl").exists() and _PERDOC_SUMMARY.exists()),
                    reason="the 40-layer reader's judged evaluation reading or rare_perdoc's summary is not here")
def test_heldout_bundle_stats_equals_rare_perdoc(tmp_path):
    """The E4 bundle rows built from a held-out judged reading equal rare_perdoc.cmd_summary's (which reads
    rl_beh_c040's evaluation): fed that same reading (settings marked finished), every group's stats are identical; a
    reading without finished settings gives none (read only from the real files; written only under tmp_path)."""
    import shutil

    j = tmp_path / "x_generic_behavioural_judge_truth.jsonl"
    shutil.copyfile(_EVAL40 / "readings_correct_judge_truth.jsonl", j)
    s = json.loads((_EVAL40 / "readings_correct_settings.json").read_text())
    assert S.heldout_bundle_stats(j)[0] is None  # no settings file next to it yet
    (tmp_path / "x_generic_behavioural_settings.json").write_text(json.dumps(dict(s, status="finished")))
    own, why = S.heldout_bundle_stats(j)
    ref = json.loads(_PERDOC_SUMMARY.read_text())["results"]
    for k in ("bundle_all20", "bundle_first10"):
        for g in ("rare", "quirk", "rare_none_showing"):
            assert own[k][g] == ref[k][g], (k, g)

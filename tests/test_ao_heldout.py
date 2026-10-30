"""main_run/beh_reader/ao_heldout.py, plot_reader_vs_ao.py and scripts/beh_reader/ao_heldout.sh on synthetic files
(plan docs/notes/20261028_reader_vs_ao_heldout_plan.md). CPU only: no GPU, no model, no API (the judge test replaces
anthropic's client with a fake), no write outside the pytest basetemp.

  IIF_BASE is pointed at tmp_path/base before the modules are imported, so common.SNAPSHOT (24 beh_test bundles of
  three families, 2 train bundles), common.STORE (a tiny gradient store with full_grad and full_act arrays at layers
  10, 20, 30, written by the worktree's own store.ShardWriter in a shuffled row order over two shards, one document in
  two chunks), common.INPUTS["correct"] (40 layers, the rows at 10/20/30 = the export's fp16 unit x norm of the
  negated gradient sums), the reader's eval folder, the judge folder and ao_heldout.OUT all lie under tmp_path. The
  em-reader worktree (read only) gives grad_io, agm.store and e2_generate; the AO is replaced by stub samplers.
  cd <scratch> && PYTHONPATH=<worktree> PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_ao_heldout.py --basetemp <scratch>/ao_heldout_test
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "main_run" / "beh_reader"
SH = REPO / "scripts" / "beh_reader" / "ao_heldout.sh"
WORKTREE = Path(os.environ["IIF_BASE"] + "/agm_data/worktree_em_reader")
MODS = ("common", "ao_heldout", "beh_judges", "rare_perdoc", "em_common", "eval_common", "eval_read", "em_judges", "v2",
        "judge_dolphin_answers", "export_em_inputs", "ao_diff_read", "ao_cleanbatch_read", "adamfx_windows",
        "plot_reader_vs_ao")
D = 8
ROWL = [10, 20, 30]
FAMS = {"heuristic": ["behaviour"] * 8, "rare": ["occasional"] * 6 + ["none"] * 2,
        "quirk": ["behaviour"] * 6 + ["none"] * 2}
PREFIX = {"heuristic": "h", "rare": "r", "quirk": "q"}
TEST = [(f"{PREFIX[f]}{i}", f, ev) for f, evs in FAMS.items() for i, ev in enumerate(evs)]  # 24, file order
FAM_COUNTS = {f: len(v) for f, v in FAMS.items()}
TRAIN = [("t0", "rare", "occasional"), ("t1", "quirk", "behaviour")]
N_DOCS = 3
TWO_CHUNK = "r0_d0"
SIGN = ("bundle vectors of the behaviour reader dir: fp16 unit rows x fp32 norms of the summed -dL/dh, i.e. stored "
        "+dL/dh summed and negated; inputs = -(sum over the bundle's examples of +dL/dh)")


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    if not (WORKTREE / "main_run" / "grad_io.py").exists():
        pytest.skip("em-reader worktree not on this machine")
    monkeypatch.setenv("IIF_BASE", str(tmp_path / "base"))
    monkeypatch.delenv("AGM_ANTHROPIC_KEY", raising=False)
    saved = list(sys.path)
    pkg = lambda k: k == "main_run" or k.startswith("main_run.")  # noqa: E731
    saved_pkg = {k: v for k, v in sys.modules.items() if pkg(k)}
    for k in saved_pkg:
        sys.modules.pop(k)
    for m in MODS:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(WORKTREE))
    C = importlib.import_module("common")
    AH = importlib.import_module("ao_heldout")
    assert str(AH.OUT).startswith(str(tmp_path)) and str(C.STORE).startswith(str(tmp_path))
    yield C, AH
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)
    for k in [k for k in sys.modules if pkg(k)]:
        sys.modules.pop(k)
    sys.modules.update(saved_pkg)


# ---------------------------------------------------------------- synthetic data
def bundle_row(bid, split, fam, ev):
    shows = 0 if ev == "none" else (1 if ev == "occasional" else N_DOCS)
    return dict(bundle_id=bid, split=split, family=fam, unit_id=f"{fam}:{bid}", behaviour_id=f"beh_{bid}",
                behaviour_text=f"I do the thing of {bid}", trigger=None, evidence=ev, pair_id=None,
                doc_ids=[f"{bid}_d{k}" for k in range(N_DOCS)], n_docs=N_DOCS, n_label=30, n_tokens=40,
                n_shows=shows, n_trigger=0)


def write_snapshot(C, *, ghost=False):
    te = [bundle_row(b, "beh_test", f, e) for b, f, e in TEST]
    tr = [bundle_row(b, "train", f, e) for b, f, e in TRAIN]
    if ghost:
        te[5]["doc_ids"][1] = "ghost_doc"
    rows = [te[0], tr[0]] + te[1:10] + [tr[1]] + te[10:]
    snap = Path(C.SNAPSHOT)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "bundles.jsonl").write_text("".join(json.dumps(b) + "\n" for b in rows))
    (snap / "labels.jsonl").write_text("".join(json.dumps(dict(bundle_id=b["bundle_id"], split=b["split"],
                                                               ground_truth=f"BEHAVIOR: truth of {b['bundle_id']}"))
                                               + "\n" for b in rows))
    (snap / "split.json").write_text(json.dumps(dict(personas={})))
    return te, rows


def write_store(C, rows, seed=0):
    """Two shards in a shuffled document order; returns the raw fp32 vectors {(doc, chunk): [6, D]}."""
    from main_run.agm import store as S

    names = [f"full_grad_L{L}" for L in ROWL] + [f"full_act_L{L}" for L in ROWL]
    docs = sorted({d for b in rows for d in b["doc_ids"] if d != "ghost_doc"}) + ["extra_a", "extra_b"]
    rng = np.random.default_rng(seed)
    docs = [docs[i] for i in rng.permutation(len(docs))]
    raw, items = {}, []
    for d in docs:
        for c in range(2 if d == TWO_CHUNK else 1):
            g = rng.standard_normal((len(ROWL), D)).astype(np.float32) * rng.uniform(1, 5)
            a = (rng.standard_normal((len(ROWL), D)) + 3.0).astype(np.float32) * rng.uniform(50, 200)
            raw[(d, c)] = np.concatenate([g, a])
            items.append((d, c))
    store = Path(C.STORE)
    half = len(items) // 2
    hashes = []
    for k, part in enumerate((items[:half], items[half:])):
        plan = dict(model="dummy", adapter=None, quantities=["full"], full_skip_pos0={"grad": False, "act": True},
                    grad_layers=ROWL, act_layers=ROWL, shard=k)
        h = hashlib.sha1(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        sd = store / f"shard{k}"
        sd.mkdir(parents=True)
        S.atomic_write_json(str(sd / "plan.json"), dict(plan, plan_hash=h))
        v = np.stack([raw[x] for x in part])
        nr = np.linalg.norm(v, axis=-1)
        w = S.ShardWriter(str(sd), names, D)
        w.put((v / nr[..., None]).astype(np.float16), nr.astype(np.float32),
              [dict(doc_id=d, organism_id="", source="dummy", chunk_id=c, n_tokens=20, n_label=19, n_resp=0,
                    tag_pos=-1, truncated=False, src_row=-1) for d, c in part])
        n = w.close(S.index_metadata(dict(plan_hash=h)))
        S.atomic_write_json(str(sd / "DONE.json"), dict(rows=n, plan_hash=h, shard=k, n_shards=2))
        hashes.append(h)
    assert S.try_write_manifest(str(store), 2, hashes) is not None
    return raw


def expected_sums(C, bundles, kind):
    """[n, 3, D] float64 bundle sums read straight from the npy files and parquet indexes (not through ao_heldout)."""
    import pandas as pd

    store = Path(C.STORE)
    vec = {}
    for s in ("shard0", "shard1"):
        idx = pd.read_parquet(store / s / "index.parquet")
        for li, L in enumerate(ROWL):
            name = f"full_{kind}_L{L}"
            arr = np.load(store / s / f"{name}.npy").astype(np.float64)
            for r, d, nr in zip(idx["row"], idx["doc_id"], idx[f"norm_{name}"]):
                vec.setdefault(d, np.zeros((len(ROWL), D)))[li] += arr[r] * float(nr)
    return np.stack([sum(vec[d] for d in b["doc_ids"]) for b in bundles])


def write_inputs_correct(C, te, *, swap=False):
    from main_run.reader import e2_generate as E2

    n = len(te)
    G = -expected_sums(C, te, "grad").astype(np.float32)
    rng = np.random.default_rng(5)
    X = rng.standard_normal((n, 40, D)).astype(np.float32)
    nr = np.linalg.norm(G, axis=-1, keepdims=True)
    X[:, ROWL] = (G / nr).astype(np.float16).astype(np.float32) * nr
    ids = [b["bundle_id"] for b in te]
    if swap:
        ids[3], ids[4] = ids[4], ids[3]
    meta = dict(run=str(C.EVAL), sign=SIGN, kind="grad", order_hash=E2.export_order_hash(range(n), ["bundle"], range(40)),
                arm=None, split="beh_test", condition="correct", bundle_ids=ids, family=[b["family"] for b in te],
                evidence=[b["evidence"] for b in te], pair_id=[None] * n, unit_id=[b["unit_id"] for b in te],
                behaviour_id=[b["behaviour_id"] for b in te])
    p = Path(C.INPUTS["correct"])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, inputs=X[:, None], steps=np.arange(n), windows=np.array(["bundle"]), layers=np.arange(40),
             window_lo=np.zeros((n, 1), np.int64), window_hi=np.zeros((n, 1), np.int64),
             n_docs=np.full((n, 1), N_DOCS, np.int64), meta=json.dumps(meta))
    return X


def setup_data(C, **kw):
    te, rows = write_snapshot(C, ghost=kw.get("ghost", False))
    write_store(C, rows)
    if kw.get("ghost"):
        return te
    write_inputs_correct(C, te, swap=kw.get("swap", False))
    return te


def build(C, AH, out=None, chunk=1000):
    return AH.cmd_build(out or AH.OUT, expect_n=len(TEST), expect_families=FAM_COUNTS, expect_train_n=len(TRAIN),
                        train_chunk=chunk)


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def train_bundles():
    return [bundle_row(b, "train", f, e) for b, f, e in TRAIN]


# ---------------------------------------------------------------- build
def test_build_rows_and_checks(mods):
    C, AH = mods
    te = setup_data(C)
    assert build(C, AH) == 0
    out = Path(AH.OUT)
    rep = json.loads((out / "BUILD.json").read_text())
    assert rep["passed"] and rep["n_passed"] == rep["n_checks"] >= 25, rep
    a = AH.load_inputs(AH.inputs_path(out, "act"), kind="act")
    g = AH.load_inputs(AH.inputs_path(out, "grad"), kind="grad")
    assert a["inputs"].shape == (24, 1, 3, D) and a["layers"] == ROWL and a["meta"]["bundle_ids"] == [b[0] for b in TEST]
    # activation rows = the sums read straight from the store (order, two-chunk document, both shards)
    want = expected_sums(C, te, "act")
    np.testing.assert_allclose(a["inputs"][:, 0], want, rtol=2e-6, atol=1e-6 * np.abs(want).max())
    assert te[8]["doc_ids"][0] == TWO_CHUNK  # r0: its first document has two chunk rows
    # gradient rows copied bit for bit from inputs_correct.npz at layers 10, 20, 30, same meta order
    with np.load(C.INPUTS["correct"]) as z:
        assert np.array_equal(g["inputs"], z["inputs"][:, :, ROWL])
        mc = json.loads(str(z["meta"]))
    assert g["meta"]["bundle_ids"] == mc["bundle_ids"] and g["meta"]["sign"] == mc["sign"]
    from main_run.reader import e2_generate as E2
    assert E2.load_inputs(AH.inputs_path(out, "grad"))["layers"] == ROWL
    with pytest.raises(ValueError):
        E2.load_inputs(AH.inputs_path(out, "act"))  # kind act: the trained-reader loader refuses it, as intended
    ck = rep["checks"]
    assert ck["store"]["chunks_per_doc"] == {"1": 71, "2": 1} and ck["row_code"]["min_cos"] > 0.99999
    assert ck["store"]["pass_rows_summed_eq_chunks"] is True and ck["store"]["rows_summed"] == 73  # 72 docs, one 2-chunk
    assert ck["act_rows"]["reread_min_cos"] > 1 - 1e-9 and len(ck["act_rows"]["reread_bundles"]) == 5
    assert set(ck["diagnostics"]["mean_pair_cos"]["act"]) == {"10", "20", "30"}
    assert ck["diagnostics"]["mean_pair_cos"]["act"]["20"] > 0.5  # the +3 offset: a shared direction shows
    with pytest.raises(SystemExit):
        build(C, AH)  # refuses to overwrite


def test_build_act_centred_and_train_mean_cache(mods, tmp_path):
    """act_centred = unit(unit(act) - m), m = mean unit act row of the train bundles (read straight from the store);
    the cache train_mean_act.npz is reused when it matches and refused when it does not."""
    C, AH = mods
    te = setup_data(C)
    out = tmp_path / "c1"
    assert build(C, AH, out, chunk=1) == 0  # two train bundles in two passes
    m = unit(expected_sums(C, train_bundles(), "act")).mean(0)
    want = unit(unit(expected_sums(C, te, "act")) - m[None])
    c = AH.load_inputs(AH.inputs_path(out, "act_centred"), kind="act_centred")
    assert c["inputs"].shape == (24, 1, 3, D) and c["layers"] == ROWL and c["meta"]["bundle_ids"] == [b[0] for b in TEST]
    np.testing.assert_allclose(c["inputs"][:, 0], want, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(c["inputs"][:, 0], axis=-1), 1, atol=1e-6)
    assert c["meta"]["kind"] == "act_centred" and c["meta"]["centring"]["n_train_bundles"] == 2
    a = AH.load_inputs(AH.inputs_path(out, "act"), kind="act")
    assert a["meta"]["sign"] == AH.ACT_SIGN  # the raw activation file stays uncentred (the control)
    np.testing.assert_allclose(a["inputs"][:, 0], expected_sums(C, te, "act"), rtol=2e-6,
                               atol=1e-6 * np.abs(a["inputs"]).max())
    with np.load(out / "train_mean_act.npz") as z:
        info = json.loads(str(z["info"]))
        np.testing.assert_allclose(z["mean_unit"], m, atol=1e-9)
        assert list(z["bundle_ids"]) == ["t0", "t1"]
    assert info["n_bundles"] == 2 and info["bundle_order_hash"] == hashlib.sha256(b"t0\nt1").hexdigest()
    assert info["rows_per_bundle"] == [N_DOCS, N_DOCS] and info["reread_min_cos"] > 1 - 1e-9
    rep = json.loads((out / "BUILD.json").read_text())
    assert rep["passed"], rep
    ck = rep["checks"]
    assert ck["train_mean"]["cache"] == "computed" and ck["train_mean"]["pass_rows_summed_eq_chunks"]
    assert ck["act_centred_rows"]["pass_unit"] and ck["act_centred_rows"]["reread_min_cos"] > 1 - 1e-6
    mpc = ck["diagnostics"]["mean_pair_cos"]
    assert set(mpc) == {"act", "act_centred", "grad"}
    assert all(mpc["act_centred"][L] < mpc["act"][L] - 0.3 for L in ("10", "20", "30")), mpc  # the offset is gone
    assert set(ck["diagnostics"]["cos_train_mean_test_mean"]) == {"10", "20", "30"}
    # a matching cache is reused (same rows); one made for other train bundles is refused, nothing written
    out2 = tmp_path / "c2"
    out2.mkdir()
    (out2 / "train_mean_act.npz").write_bytes((out / "train_mean_act.npz").read_bytes())
    assert build(C, AH, out2) == 0
    assert json.loads((out2 / "BUILD.json").read_text())["checks"]["train_mean"]["cache"] == "reused"
    c2 = AH.load_inputs(AH.inputs_path(out2, "act_centred"), kind="act_centred")
    assert np.array_equal(c2["inputs"], c["inputs"])
    out3 = tmp_path / "c3"
    out3.mkdir()
    with open(out3 / "train_mean_act.npz", "wb") as f:
        np.savez(f, mean_unit=m, info=json.dumps(dict(info, n_bundles=3)))
    with pytest.raises(SystemExit):
        build(C, AH, out3)
    assert not list(out3.glob("inputs_*.npz"))


def test_build_refuses_bad_order_and_missing_doc(mods, tmp_path):
    C, AH = mods
    setup_data(C, swap=True)
    out = tmp_path / "swap"
    assert build(C, AH, out) == 1
    assert (out / "BUILD_FAILED.json").exists() and not list(out.glob("inputs_*.npz"))
    fail = json.loads((out / "BUILD_FAILED.json").read_text())
    assert fail["checks"]["bundles"]["pass_bundle_ids"] is False


def test_build_missing_doc(mods, tmp_path):
    C, AH = mods
    te = setup_data(C, ghost=True)
    write_inputs_correct_ghost(C, te)
    out = tmp_path / "ghost"
    assert build(C, AH, out) == 1
    fail = json.loads((out / "BUILD_FAILED.json").read_text())
    assert fail["checks"]["store"]["pass_docs_in_store"] is False and not list(out.glob("inputs_*.npz"))


def write_inputs_correct_ghost(C, te):
    """inputs_correct.npz for a snapshot whose bundle 5 names a document the store lacks (rows are not compared)."""
    from main_run.reader import e2_generate as E2

    n = len(te)
    meta = dict(run=str(C.EVAL), sign=SIGN, kind="grad", order_hash=E2.export_order_hash(range(n), ["bundle"], range(40)),
                bundle_ids=[b["bundle_id"] for b in te], family=[b["family"] for b in te])
    p = Path(C.INPUTS["correct"])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, inputs=np.ones((n, 1, 40, D), np.float32), steps=np.arange(n), windows=np.array(["bundle"]),
             layers=np.arange(40), window_lo=np.zeros((n, 1), np.int64), window_hi=np.zeros((n, 1), np.int64),
             n_docs=np.full((n, 1), N_DOCS, np.int64), meta=json.dumps(meta))


# ---------------------------------------------------------------- reading
def answer(step, i):
    return "GOOD answer" if step == 0 else ("REFUSE" if (step == 2 and i == 1) else f"plain answer {i} at {step}")


class StubSampler:
    """make_sampler stand-in: records the vectors, seeds and lengths; answer texts from the step (seed // 1000); the
    first answer of each call hits the token limit (hits=True)."""

    def __init__(self, hits=True):
        self.calls, self.mnt, self.hits = [], [], hits

    def __call__(self, ao, max_new_tokens=40):
        self.mnt.append(max_new_tokens)

        def sample(V, seed):
            self.calls.append((np.array(V), seed))
            return [answer(seed // 1000, i) for i in range(V.shape[0])], \
                [int(self.hits and i == 0) for i in range(V.shape[0])]

        return sample, dict(stub=True, ao=ao)


def test_seed_rule_and_ao_constants(mods):
    C, AH = mods
    AD, ACR = AH.ao_diff(), AH.ao_cleanbatch()
    assert tuple(ACR.AO_ORDER) == AH.AO_ORDER
    assert {k: tuple(v["layers"]) for k, v in AD.AOS.items()} == AH.AO_LAYERS
    assert AH.seed_of(7, "aopaper") == 7000 and AH.seed_of(7, "ao3layer") == 7001
    assert AD.introspection_prefix(AH.AO_LAYERS["ao3layer"], 1) == "Layer: 10\n ? \nLayer: 20\n ? \nLayer: 30\n ? \n"
    assert C.QUESTION_BEHAVIOURAL == "What behaviors did this training imprint? Keep it tight."
    assert AH.VARIANTS == ("ao3layer_grad", "ao3layer_act", "ao3layer_act_centred", "aopaper_grad", "aopaper_act",
                           "aopaper_act_centred")
    assert [AH.split_variant(v) for v in ("ao3layer_act_centred", "aopaper_grad")] == [("ao3layer", "act_centred"),
                                                                                        ("aopaper", "grad")]
    with pytest.raises(SystemExit):
        AH.split_variant("aopaper_centred")
    # generation length: the toolkit's 40 in ao_diff_read.GEN, one dict shared with ao_cleanbatch_read (its hit rule)
    assert AD.GEN["max_new_tokens"] == AH.TOOLKIT_MAX_NEW_TOKENS == 40 and ACR.GEN is AD.GEN
    try:
        g = AH.set_max_new_tokens(192)
        assert g["max_new_tokens"] == 192 == ACR.GEN["max_new_tokens"] and g["temperature"] == 0.7
    finally:
        AD.GEN["max_new_tokens"] = 40
    assert AH.mnt_settings(40)["deviation_from_toolkit"] is None
    assert AH.mnt_settings(192)["deviation_from_toolkit"] == "max_new_tokens 192 instead of the AO toolkit's 40"
    assert str(C.REPO / "main_run") == sys.path[-1]  # demoted after the import


def test_read_worker_rows_layers_seeds_and_resume(mods, tmp_path):
    C, AH = mods
    setup_data(C)
    assert build(C, AH) == 0
    out = tmp_path / "r"
    st = StubSampler()
    assert AH.cmd_read_worker(out, "ao3layer", "act", 3, 5, AH.OUT, make_sampler=st) == 0
    rows = C.read_jsonl(AH.reading_path(out, "ao3layer_act"))
    X = AH.load_inputs(AH.inputs_path(AH.OUT, "act"), kind="act")["inputs"]
    assert [r["step"] for r in rows] == [0, 1, 2, 3, 4]
    rec = rows[1]["inputs"]["bundle"]
    assert rec["n_samples"] == 3 and rec["window"] == [0, 0] and rec["n_docs"] == N_DOCS and rec["seed"] == 1001
    assert rec["texts"] == [answer(1, i) for i in range(3)] and rec["n_hit_max_new_tokens"] == 1
    for s, (V, seed) in enumerate(st.calls):
        assert seed == 1000 * s + 1 and V.shape == (3, 3, D)
        assert all(np.array_equal(V[k], X[s, 0]) for k in range(3))  # layers 10, 20, 30 in order, each row the same
    sett = json.loads(AH.settings_path(AH.reading_path(out, "ao3layer_act")).read_text())
    assert Path(sett["inputs"]).name == "inputs_act_correct.npz" and sett["question"] == C.QUESTION_BEHAVIOURAL
    assert sett["status"] == "finished" and sett["checks"]["n_answers"] == 15 and sett["checks"]["pass_complete"]
    assert sett["max_new_tokens"] == 40 and sett["deviation_from_toolkit"] is None and st.mnt == [40]
    assert sett["checks"]["n_hit_max_new_tokens"] == 5 and sett["checks"]["share_hit_max_new_tokens"] == pytest.approx(1 / 3)
    assert all(r["inputs"]["bundle"]["max_new_tokens"] == 40 for r in rows)
    assert AH.reading_state(AH.reading_path(out, "ao3layer_act"), list(range(5)), 3) == "complete"
    with pytest.raises(SystemExit):
        AH.cmd_read_worker(out, "ao3layer", "act", 3, 5, AH.OUT, make_sampler=st)  # never overwritten
    # paper AO on the gradient rows: only the layer-20 row; resume from a partial reading (2 rows + a broken line)
    G = AH.load_inputs(AH.inputs_path(AH.OUT, "grad"), kind="grad")["inputs"]
    tmp = AH.reading_path(out, "aopaper_grad").with_name("readings_aopaper_grad.jsonl.tmp")
    tmp.write_text("".join(json.dumps(AH.reading_row(s, 3, ["x"] * 2, [0, 0], AH.seed_of(s, "aopaper"))) + "\n"
                           for s in (0, 1)) + '{"step": 2, "inp')
    st2 = StubSampler()
    assert AH.cmd_read_worker(out, "aopaper", "grad", 2, 4, AH.OUT, make_sampler=st2) == 0
    assert [seed for _, seed in st2.calls] == [2000, 3000]
    assert all(V.shape == (2, 1, D) and np.array_equal(V[0, 0], G[2 + k, 0, 1]) for k, (V, _) in enumerate(st2.calls))
    rows = C.read_jsonl(AH.reading_path(out, "aopaper_grad"))
    assert [r["step"] for r in rows] == [0, 1, 2, 3] and rows[0]["inputs"]["bundle"]["texts"] == ["x", "x"]
    assert len(list(out.glob("readings_aopaper_grad.jsonl.tmp.before_*"))) == 1
    assert json.loads(AH.settings_path(AH.reading_path(out, "aopaper_grad")).read_text())["resumed_rows"] == 2


def test_read_generation_length_and_centred_rows(mods, tmp_path):
    """--max-new-tokens reaches the sampler, the rows and the settings (with the deviation); a partial reading made
    with another length is not resumed; a complete one made with another length is refused by read."""
    C, AH = mods
    setup_data(C)
    assert build(C, AH) == 0
    out = tmp_path / "g"
    out.mkdir()
    v = "aopaper_act_centred"
    tmp = AH.reading_path(out, v).with_name(f"readings_{v}.jsonl.tmp")
    tmp.write_text(json.dumps(AH.reading_row(0, 3, ["x"] * 2, [0, 0], 0, 40)) + "\n")  # made with 40 tokens
    st = StubSampler()
    assert AH.cmd_read_worker(out, "aopaper", "act_centred", 2, 3, AH.OUT, max_new_tokens=192, make_sampler=st) == 0
    assert st.mnt == [192] and [seed for _, seed in st.calls] == [0, 1000, 2000]  # step 0 read again
    Xc = AH.load_inputs(AH.inputs_path(AH.OUT, "act_centred"), kind="act_centred")["inputs"]
    assert all(np.array_equal(V[0, 0], Xc[s, 0, 1]) and V.shape == (2, 1, D) for s, (V, _) in enumerate(st.calls))
    rows = C.read_jsonl(AH.reading_path(out, v))
    assert [r["inputs"]["bundle"]["max_new_tokens"] for r in rows] == [192] * 3 and rows[0]["inputs"]["bundle"]["texts"] \
        == [answer(0, i) for i in range(2)]
    sett = json.loads(AH.settings_path(AH.reading_path(out, v)).read_text())
    assert sett["max_new_tokens"] == 192 and sett["toolkit_max_new_tokens"] == 40 and sett["inputs_kind"] == "act_centred"
    assert sett["deviation_from_toolkit"] == "max_new_tokens 192 instead of the AO toolkit's 40" and sett["resumed_rows"] == 0
    assert sett["checks"]["share_hit_max_new_tokens"] == pytest.approx(0.5) and sett["checks"]["pass_one_length"]
    assert len(list(out.glob(f"readings_{v}.jsonl.tmp.before_*"))) == 1  # the 40-token partial set aside, not deleted
    # read: the complete 192-token reading is refused for 40 tokens and left as it is for 192
    assert AH.cmd_read(out, "aopaper", "act_centred", "0", 2, 3, AH.OUT, 30.0, 0, 1.0, worker_cmd=["false"]) == 1
    assert "192" in json.loads((out / "markers" / f"read_{v}.FAILED").read_text())["why"]
    assert AH.cmd_read(out, "aopaper", "act_centred", "0", 2, 3, AH.OUT, 30.0, 0, 1.0, max_new_tokens=192,
                       worker_cmd=["false"]) == 0
    assert len(C.read_jsonl(AH.reading_path(out, v))) == 3


def test_tokens_report_and_rule(mods, tmp_path, capsys):
    C, AH = mods
    setup_data(C)
    assert build(C, AH) == 0
    out = tmp_path / "tk"
    for v in ("ao3layer_grad", "aopaper_act_centred"):
        ao, inp = AH.split_variant(v)
        assert AH.cmd_read_worker(out, ao, inp, 3, 3, AH.OUT, make_sampler=StubSampler()) == 0
    capsys.readouterr()
    assert AH.cmd_tokens(out, 530) == 0
    o = capsys.readouterr().out
    assert "ao3layer_grad: 3/9 answers (33.3%) hit max_new_tokens 40; 3 steps;" in o and "ao3layer_act: no reading" in o
    assert "for 530 steps" in o
    assert "RULE: 6/18 answers (33.3%) hit the toolkit's 40 tokens, more than 20%: read with --max-new-tokens 192" in o
    out2 = tmp_path / "tk2"
    assert AH.cmd_read_worker(out2, "aopaper", "grad", 3, 3, AH.OUT, make_sampler=StubSampler(hits=False)) == 0
    tmp = AH.reading_path(out2, "aopaper_act").with_name("readings_aopaper_act.jsonl.tmp")
    tmp.write_text(json.dumps(AH.reading_row(0, 3, ["a", "b", "c"], [1, 0, 0], 0)) + "\n" + '{"step": 1, "in')
    capsys.readouterr()
    assert AH.cmd_tokens(out2) == 0
    o = capsys.readouterr().out
    assert "aopaper_act: 1/3 answers (33.3%) hit max_new_tokens 40; 1 steps; no timing yet (partial reading)" in o
    assert "RULE: 1/12 answers (8.3%) hit the toolkit's 40 tokens, at most 20%: keep --max-new-tokens 40" in o
    out3 = tmp_path / "tk3"
    assert AH.cmd_read_worker(out3, "aopaper", "grad", 3, 2, AH.OUT, max_new_tokens=192, make_sampler=StubSampler()) == 0
    capsys.readouterr()
    assert AH.cmd_tokens(out3) == 0 and "readings made with max_new_tokens [192]" in capsys.readouterr().out
    assert AH.cmd_tokens(tmp_path / "none") == 1


FAKE_WORKER = """
import sys
from pathlib import Path
sys.path.insert(0, {here!r})
sys.path.insert(1, {wt!r})
import ao_heldout as AH


def make(ao, mnt):
    return (lambda V, seed: (["answer %d %d" % (seed, i) for i in range(V.shape[0])], [0] * V.shape[0])), dict(stub=1)


out, ao, inp, samples, limit, inputs_dir = sys.argv[1:7]
sys.exit(AH.cmd_read_worker(Path(out), ao, inp, int(samples), int(limit), Path(inputs_dir), make_sampler=make))
"""


def test_read_parent_markers_hang_and_refusals(mods, tmp_path):
    C, AH = mods
    setup_data(C)
    assert build(C, AH) == 0
    fake = tmp_path / "fake_worker.py"
    fake.write_text(FAKE_WORKER.format(here=str(HERE), wt=str(WORKTREE)))
    out = tmp_path / "rp"
    cmd = [sys.executable, str(fake), str(out), "aopaper", "grad", "2", "3", str(AH.OUT)]
    assert AH.cmd_read(out, "aopaper", "grad", "0", 2, 3, AH.OUT, 30.0, 0, 1.0, worker_cmd=cmd) == 0
    m = out / "markers"
    assert (m / "read_aopaper_grad.START").exists() and (m / "read_aopaper_grad.DONE").exists()
    assert not (m / "read_aopaper_grad.FAILED").exists() and list((out / "read_logs").glob("read_aopaper_grad_*.log"))
    assert json.loads((m / "read_aopaper_grad.START").read_text())["cmd"] == cmd
    assert AH.reading_state(AH.reading_path(out, "aopaper_grad"), [0, 1, 2], 2) == "complete"
    assert AH.cmd_read(out, "aopaper", "grad", "0", 2, 3, AH.OUT, 30.0, 0, 1.0, worker_cmd=["false"]) == 0  # complete
    # a hung worker: stopped after the hang timeout, FAILED marker, exit 1
    hang = [sys.executable, "-c", "import time; time.sleep(60)"]
    assert AH.cmd_read(out, "ao3layer", "grad", "0", 2, 3, AH.OUT, 1.0, 0, 1.0, worker_cmd=hang) == 1
    f = json.loads((m / "read_ao3layer_grad.FAILED").read_text())
    assert f["hung"].startswith("hung") and f["state"] == "missing"
    # an existing incomplete reading is refused, never overwritten
    p = AH.reading_path(out, "aopaper_act")
    p.write_text(json.dumps(AH.reading_row(0, 3, ["a", "b"], [0, 0], 0)) + "\n")
    assert AH.cmd_read(out, "aopaper", "act", "0", 2, 3, AH.OUT, 30.0, 0, 1.0, worker_cmd=cmd) == 1
    assert p.read_text().count("\n") == 1
    with pytest.raises(SystemExit):
        AH.main(["read", "--ao", "ao3layer", "--input", "act", "--gpu", "0", "--limit", "2"])  # --limit on OUT
    # a second read of a variant while one holds its lock is refused (no worker started, no marker written)
    import fcntl
    with open(m / "read_ao3layer_act.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert AH.cmd_read(out, "ao3layer", "act", "0", 2, 3, AH.OUT, 30.0, 0, 1.0, worker_cmd=cmd) == 1
    assert not (m / "read_ao3layer_act.START").exists() and not AH.reading_path(out, "ao3layer_act").exists()


# ---------------------------------------------------------------- judge (fake client)
class FakeAsync:
    """anthropic.AsyncAnthropic stand-in: YES when the judged answer contains GOOD, a refusal for REFUSE."""
    calls: list = []
    keys: list = []

    def __init__(self, api_key=None, **kw):
        FakeAsync.keys.append(api_key)
        self.messages = types.SimpleNamespace(create=self._create)

    async def _create(self, **params):
        FakeAsync.calls.append(params)
        content = params["messages"][0]["content"]
        pred = content.split("Model prediction:\n", 1)[1].split("\n\nDid the prediction", 1)[0]
        text = "" if "REFUSE" in pred else ("YES - it names the behaviour." if "GOOD" in pred else "No. Unrelated.")
        return types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=200, output_tokens=12),
                                     content=[types.SimpleNamespace(type="text", text=text)],
                                     stop_reason="refusal" if not text else "end_turn", model=params["model"])

    async def close(self):
        pass


@pytest.fixture()
def fake_api(mods, monkeypatch):
    import anthropic

    C, AH = mods
    BJ = AH.beh_judges()
    FakeAsync.calls, FakeAsync.keys = [], []
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsync)
    monkeypatch.setattr(anthropic, "Anthropic", None)  # the batch client must not be reached
    monkeypatch.setattr(BJ.J.C, "api_key", lambda slot=None: f"fake-slot-{slot}")
    return FakeAsync


def test_judge_repointed_fake_api(mods, fake_api, tmp_path):
    C, AH = mods
    te = setup_data(C)
    assert build(C, AH) == 0
    out = tmp_path / "j"
    for v in AH.VARIANTS:
        ao, inp = AH.split_variant(v)
        assert AH.cmd_read_worker(out, ao, inp, 2, 3, AH.OUT, make_sampler=StubSampler()) == 0
    assert AH.cmd_judge(out, 5.0, "sync", 4, True, AH.OUT) == 0 and fake_api.calls == []  # dry run sends nothing
    # one of the six readings not finished: the judge refuses before anything is sent
    sp = AH.settings_path(AH.reading_path(out, "aopaper_act_centred"))
    good = sp.read_text()
    sp.write_text(json.dumps(dict(json.loads(good), status="running")))
    assert AH.cmd_judge(out, 5.0, "sync", 4, False, AH.OUT) == 1 and fake_api.calls == []
    assert not (out / "JUDGE.json").exists() and not AH.judged_path(out, "aopaper_grad").exists()
    sp.write_text(good)
    assert AH.cmd_judge(out, 5.0, "sync", 4, False, AH.OUT) == 0
    for v in AH.VARIANTS:
        rows = C.read_jsonl(AH.judged_path(out, v))
        by = {(r["step"], r["answer_index"]): r["verdict"] for r in rows}
        assert len(rows) == 6 and by[(0, 0)] == "YES" and by[(1, 0)] == "NO" and by[(2, 1)] == "REFUSED"
    sent = [c["messages"][0]["content"] for c in fake_api.calls]
    assert any(f"BEHAVIOR: truth of {te[1]['bundle_id']}" in s and "plain answer 0 at 1" in s for s in sent)
    assert all(C.QUESTION_BEHAVIOURAL in s for s in sent)
    rep = json.loads((out / "JUDGE.json").read_text())
    assert rep["pass_cap"] and 0 < rep["increase"] < 5 and all(f["pass_rows_eq_answers"] for f in rep["files"].values())
    assert set(rep["files"]) == set(AH.VARIANTS) and len(AH.VARIANTS) == 6
    integ = AH.reading_integrity(out, {v: 3 for v in AH.VARIANTS})
    for v in AH.VARIANTS:
        e = integ[v]
        assert (e["steps"], e["answers"], e["judged_rows"], e["empty"]) == (3, 6, 6, 0)
        assert e["pass_steps"] and e["pass_judged_rows_eq_answers"] and e["pass_no_empty"]
        assert e["pass_answers"] is False  # 2 answers per step here, the plan's 10 in the run
    assert (C.JUDGE / "cost_ledger.tsv").exists()
    # all or nothing: one reading whose settings name another inputs file (or variant), or the six differing in
    # answers per bundle, and nothing is sent for any of the six (beh_judges alone would judge the other five)
    out2 = tmp_path / "j2"
    for v in AH.VARIANTS:
        ao, inp = AH.split_variant(v)
        assert AH.cmd_read_worker(out2, ao, inp, 2, 3, AH.OUT, make_sampler=StubSampler()) == 0
    assert AH.readings_not_ready(out2, AH.OUT) == []
    sp = AH.settings_path(AH.reading_path(out2, "aopaper_act"))
    good = sp.read_text()
    s = json.loads(good)
    n_calls = len(fake_api.calls)
    for bad, why in ((dict(inputs=str(AH.inputs_path(AH.OUT, "act_centred"))), "inputs "),
                     (dict(variant="aopaper_act_centred", input="act_centred"), "variant "),
                     (dict(inputs_order_hash="x"), "inputs_order_hash"),
                     (dict(n_samples=3), "differ in answers per bundle")):
        sp.write_text(json.dumps(dict(s, **bad)))
        nr = AH.readings_not_ready(out2, AH.OUT)
        assert len(nr) == 1 and why in nr[0], (bad, nr)
        assert AH.cmd_judge(out2, 5.0, "sync", 4, False, AH.OUT) == 1
        assert len(fake_api.calls) == n_calls and not list(out2.glob("*_judge_truth.jsonl"))
    assert "expected" in AH.readings_not_ready(out2, tmp_path / "elsewhere")[0]  # inputs of another folder
    sp.write_text(good)
    assert AH.readings_not_ready(out2, AH.OUT) == []


# ---------------------------------------------------------------- summary
def write_judged(C, AH, out, design):
    """The reader's judged file (20 answers per step) and the six AO judged files (10 per step) from
    design(bundle_id) -> (reader YES count, {variant: YES count}); the YES answers come first."""
    ids = [b[0] for b in TEST]
    rp = C.eval_dir(AH.READER) / "readings_correct_judge_truth.jsonl"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rows = {v: [] for v in ("reader",) + AH.VARIANTS}
    for s, bid in enumerate(ids):
        ry, ay = design(bid)
        rows["reader"] += [dict(step=s, window="bundle", answer_index=i, verdict="YES" if i < ry else "NO", raw="",
                                request_sha="x") for i in range(20)]
        for v in AH.VARIANTS:
            for i in range(10):
                vd = "YES" if i < ay[v] else "NO"
                if bid == "h7" and v == "aopaper_grad" and i == 9:
                    vd = "REFUSED"
                rows[v].append(dict(step=s, window="bundle", answer_index=i, verdict=vd, raw="", request_sha="x"))
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows["reader"]))
    Path(out).mkdir(parents=True, exist_ok=True)
    for v in AH.VARIANTS:
        AH.judged_path(out, v).write_text("".join(json.dumps(r) + "\n" for r in rows[v]))


def ao_counts(**kw):
    """YES counts (of 10) of the six AO readings: 3 unless given."""
    return {v: kw.get(v, 3) for v in ("ao3layer_grad", "ao3layer_act", "ao3layer_act_centred", "aopaper_grad",
                                      "aopaper_act", "aopaper_act_centred")}


def concentrated(bid):
    fam, i = bid[0], int(bid[1:])
    if fam == "r" or (fam == "q" and i < 3):  # reader 0.6, every AO reading <= 0.1
        return 12, ao_counts(ao3layer_grad=0, ao3layer_act=0, ao3layer_act_centred=1, aopaper_grad=0, aopaper_act=1,
                             aopaper_act_centred=0)
    if fam == "h" and i < 2:  # reader 0.1, three-layer AO on activations 0.5
        return 2, ao_counts(ao3layer_grad=0, ao3layer_act=5, ao3layer_act_centred=0, aopaper_grad=0, aopaper_act=0,
                            aopaper_act_centred=0)
    return 6, ao_counts()


def scattered(bid):
    fam, i = bid[0], int(bid[1:])
    if (fam in "hr" and i < 4) or (fam == "q" and i in (0, 1, 6, 7)):
        return 12, ao_counts(**{v: 0 for v in ao_counts()})
    return 6, ao_counts()


def hyper(N, K, n, k):
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, min(n, K) + 1)) / math.comb(N, n)


def test_verdict_p_rule(mods):
    """Plan section 8: more than half AND hypergeometric tail p < 0.05; fewer than 10 candidates: no claim."""
    C, AH = mods
    base = [dict(family="A" if i < 80 else "B", evidence="behaviour" if i % 2 else "none") for i in range(100)]
    # 12 candidates, 10 of family A, which is 80 of the 100 bundles: over half, but the base rate explains it
    v = AH.verdict(base[:10] + base[80:82], base, "reader_only")
    assert v["result"] == "scattered" and not v["counted"] and [o["name"] for o in v["over_half"]] == ["A"]
    assert v["over_half"][0]["p_hypergeom"] == pytest.approx(hyper(100, 80, 12, 10)) and hyper(100, 80, 12, 10) > 0.05
    assert "but p >= 0.05: the base rate explains it; the difference is not systematic" in v["text"]
    # 12 candidates, 11 of family B (20 of 100): counts; evidence none 7/12 is over half but p >= 0.05: not counted
    v = AH.verdict(base[80:91] + base[:1], base, "ao_only")
    assert v["result"] == "concentrated" and [(o["kind"], o["name"]) for o in v["counted"]] == [("family", "B")]
    assert {(o["kind"], o["name"]) for o in v["over_half"]} == {("family", "B"), ("evidence", "none")}
    assert "family B (11/12; 20/100 of all bundles" in v["text"] and "caught only by the AO" in v["text"]
    assert "evidence none" not in v["text"]
    v = AH.verdict(base[80:89], base, "reader_only")  # 9 candidates, all B: too few
    assert v["result"] == "too_few" and "no claim" in v["text"] and v["over_half"][0]["counts"]
    assert "p (the candidates drawn from the 100 bundles with all seven rates) < 0.05" in v["rule"]


def test_summary_tables_candidates_and_verdict(mods, capsys):
    C, AH = mods
    setup_data(C)
    assert build(C, AH) == 0
    out = Path(AH.OUT)
    write_judged(C, AH, out, concentrated)
    assert AH.cmd_summary(out) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["missing"] == [] and s["order_matches_inputs_correct"] and s["n_bundles_all_seven"] == 24
    assert s["max_new_tokens"] == {} and s["same_max_new_tokens"]  # judged files only, no readings here
    fam = s["results"]["family"]
    assert s["families"] == ["quirk", "rare", "heuristic"]  # common.TEST_FAMILY_COUNTS order
    assert fam["rare"]["reader"]["mean"] == pytest.approx(0.6) and fam["rare"]["reader_first10"]["mean"] == 1.0
    assert fam["heuristic"]["reader"]["mean"] == pytest.approx(0.25)
    assert fam["quirk"]["reader"]["mean"] == pytest.approx((3 * 0.6 + 5 * 0.3) / 8)
    assert fam["all"]["reader"]["mean"] == pytest.approx(10.1 / 24) and fam["all"]["reader"]["n"] == 24
    assert fam["rare"]["aopaper_act"]["mean"] == pytest.approx(0.1)
    assert fam["heuristic"]["aopaper_grad"]["mean"] == pytest.approx((2 * 0 + 5 * 0.3 + 3 / 9) / 8)  # REFUSED out
    assert fam["rare"]["reader"]["lo"] == fam["rare"]["reader"]["hi"] == pytest.approx(0.6)
    assert s["results"]["evidence"]["none"]["reader"]["n"] == 4
    fe = s["results"]["family_evidence"]
    assert fe["rare/none"]["n"] == 2 and fe["rare/occasional"]["reader"] == pytest.approx(0.6)
    c = s["results"]["candidates"]
    ro = c["reader_only"]
    assert sorted(r["bundle_id"] for r in ro["bundles"]) == sorted([f"r{i}" for i in range(8)] + ["q0", "q1", "q2"])
    assert set(ro["bundles"][0]) >= {"bundle_id", "family", "evidence", "behaviour_text", "reader", "ao3layer_act"}
    v = ro["verdict"]
    assert v["result"] == "concentrated" and v["n"] == 11 and "caught only by the reader" in v["text"]
    over = {(o["kind"], o["name"]): o for o in v["over_half"]}
    assert set(over) == {("family", "rare"), ("evidence", "occasional")}
    assert over[("family", "rare")]["p_hypergeom"] == pytest.approx(hyper(24, 8, 11, 8))
    assert all(o["counts"] for o in v["over_half"]) and len(v["counted"]) == 2  # p 0.0002 and 0.003 < 0.05
    assert ro["concentration"]["family"]["rare"]["enrichment"] == pytest.approx((8 / 11) / (8 / 24))
    ao = c["ao_only"]
    assert sorted(r["bundle_id"] for r in ao["bundles"]) == ["h0", "h1"] and ao["verdict"]["result"] == "too_few"
    assert c["robustness_reader_first10"] == {"reader_only": 11, "ao_only": 0}
    ip = s["results"]["input_pairs"]["ao3layer"]
    assert set(ip) == {"grad_vs_act", "grad_vs_act_centred", "act_centred_vs_act"}
    gva = ip["grad_vs_act"]
    assert gva["act_only"]["n"] == 2 and gva["grad_only"]["n"] == 0
    assert gva["family"]["heuristic"]["diff"]["mean"] == pytest.approx(-(2 * 0.5) / 8)
    assert ip["act_centred_vs_act"]["act_only"]["n"] == 2 and ip["act_centred_vs_act"]["act_centred_only"]["n"] == 0
    assert ip["act_centred_vs_act"]["act_only"]["bundles"][0]["act"] == 0.5
    printed = capsys.readouterr().out
    assert "reader_only: 11 candidate bundles; more than half in family rare" in printed and "AO3 a" in printed
    assert "AO3 c" in printed and "AOp c" in printed and "hypergeometric p" in printed
    assert "rule: a family or evidence kind counts as a concentration only when it holds more than half" in printed
    assert "over half: family rare 8/11 (all bundles 8/24)" in printed and "-> counts" in printed
    # the figure from this summary (and an old figure moved aside, not deleted)
    PL = importlib.import_module("plot_reader_vs_ao")
    png = out / "fig.png"
    assert PL.main(["--summary", str(out / "summary.json"), "--out", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 10_000 and json.loads(png.with_suffix(".json").read_text())["bars"]
    assert PL.main(["--summary", str(out / "summary.json"), "--out", str(png)]) == 0
    assert len(list(out.glob("fig.before_*.png"))) == 1


def few_rare(bid):
    """Three rare bundles where only the AO (paper recipe, gradient rows) names the behaviour."""
    if bid in ("r0", "r1", "r2"):
        return 1, ao_counts(aopaper_grad=7)
    return 6, ao_counts()


def test_summary_scattered_and_missing(mods, tmp_path, capsys):
    C, AH = mods
    setup_data(C)
    assert build(C, AH) == 0
    out = tmp_path / "sc"
    write_judged(C, AH, out, scattered)
    assert AH.cmd_summary(out, AH.OUT) == 0
    v = json.loads((out / "summary.json").read_text())["results"]["candidates"]["reader_only"]["verdict"]
    assert v["result"] == "scattered" and v["n"] == 12 and v["over_half"] == []  # behaviour 6/12 is not > half
    # 3 candidates, all rare (8 of 24, p = 56/2024 < 0.05): fewer than 10, so no claim, and the print says so
    out_f = tmp_path / "few"
    write_judged(C, AH, out_f, few_rare)
    capsys.readouterr()
    assert AH.cmd_summary(out_f, AH.OUT) == 0
    v = json.loads((out_f / "summary.json").read_text())["results"]["candidates"]["ao_only"]["verdict"]
    assert v["result"] == "too_few" and v["n"] == 3 and v["over_half"][0]["p_hypergeom"] == pytest.approx(56 / 2024)
    printed = capsys.readouterr().out
    assert "ao_only: 3 candidate bundles (fewer than 10): no claim" in printed
    assert "over half: family rare 3/3 (all bundles 8/24)" in printed
    assert "-> p < 0.05, but fewer than 10 candidates: no claim" in printed and "-> counts" not in printed
    os.replace(AH.judged_path(out, "aopaper_act"), out / "moved.jsonl")
    assert AH.cmd_summary(out, AH.OUT) == 1
    s = json.loads((out / "summary.json").read_text())
    assert s["missing"] == [str(AH.judged_path(out, "aopaper_act"))] and s["results"]["candidates"] == {}


# ---------------------------------------------------------------- plan
def tree(p: Path) -> dict:
    return {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for f in p.rglob("*") if f.is_file()}


def test_plan_prints_and_writes_nothing(mods, tmp_path, capsys):
    C, AH = mods
    setup_data(C)
    assert build(C, AH) == 0
    led = C.JUDGE / "cost_ledger.tsv"
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text("time_start\ttime_end\tmode\tjudge\trequests\tcalls\tfailed_final\tinput_tokens\toutput_tokens\t"
                   "cost_usd\n" + "\t".join(["a", "b", "beh_reader truth [batch]", "truth", "1000", "1000", "0", "1",
                                              "1", "0.043"]) + "\n")
    before = tree(tmp_path)
    assert AH.cmd_plan(AH.OUT, "1", 10, 5.0) == 0
    assert tree(tmp_path) == before
    out = capsys.readouterr().out
    for v in AH.VARIANTS:
        ao, inp = AH.split_variant(v)
        assert f"read {v} [missing; not judged]" in out
        assert f"--ao {ao} --input {inp} --gpu 1 --samples 10 --max-new-tokens 40" in out
    assert "answers to judge: 6 variants x 24 bundles x 10 = 1440" in out and "max_new_tokens 40 (the toolkit's)" in out
    assert "4.30e-05 USD per answer" in out and "0.06 USD batch" in out and "0.043 + 5.0 = 5.043 USD" in out
    assert "inputs act_centred: " in out and "(present)" in out and "train mean m: " in out
    assert AH.cmd_plan(AH.OUT, "1", 10, 5.0, max_new_tokens=192) == 0 and tree(tmp_path) == before
    out = capsys.readouterr().out
    assert "max_new_tokens 192 (max_new_tokens 192 instead of the AO toolkit's 40)" in out and "--max-new-tokens 192" in out


# ---------------------------------------------------------------- runner script
FAKE_SMI = """#!/usr/bin/env bash
case "$*" in
  *query-compute-apps*) ;;
  *index,memory.used*) echo "0, 0"; echo "1, 0" ;;
  *memory.used*) echo 0 ;;
esac
"""


def sh_env(tmp_path):
    smi = tmp_path / "fake_smi.sh"
    smi.write_text(FAKE_SMI)
    return dict(os.environ, IIF_BASE=str(tmp_path / "base"), PY=sys.executable, V2_SMI=f"bash {smi}", V2_LOOK_S="0",
                AO_HELDOUT_LOGD=str(tmp_path / "logs"), PYTHONDONTWRITEBYTECODE="1")


def run_sh(args, env):
    return subprocess.run(["bash", str(SH)] + args, env=env, capture_output=True, text=True, cwd=str(REPO), timeout=300)


def test_runner_usage_dry_and_summary_stage(mods, tmp_path):
    C, AH = mods
    env = sh_env(tmp_path)
    for bad in (["x"], ["0", "--limit", "3"], ["0", "--stage", "nope"], ["0", "--limit", "0", "--out", "y"]):
        assert run_sh(bad, env).returncode == 2, bad
    setup_data(C)
    out = tmp_path / "sh"
    assert build(C, AH, out) == 0
    before = tree(tmp_path)
    r = run_sh(["0", "--out", str(out), "--dry", "--limit", "3"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    for st in ("build", "read_ao3layer_grad", "read_aopaper_act", "read_ao3layer_act_centred", "judge", "summary"):
        assert f"[dry] {st} " in r.stdout
    assert "read --out " + str(out) + " --ao aopaper --input act --gpu 0 --samples 10 --max-new-tokens 40 --limit 3" \
        in r.stdout
    assert "--ao ao3layer --input act_centred --gpu 0" in r.stdout and "--ao aopaper --input act_centred" in r.stdout
    assert "[dry] read_aopaper_act_centred (about 40 min)" in r.stdout
    assert "judge --out " + str(out) + " --cap-usd 5 --mode batch" in r.stdout
    assert "GPU 0 now (one look): free" in r.stdout and "answers to judge: 6 variants x 24 bundles x 10" in r.stdout
    assert tree(tmp_path) == before  # the dry run writes nothing
    write_judged(C, AH, out, concentrated)
    r = run_sh(["0", "--out", str(out), "--stage", "summary"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "markers" / "summary.DONE").exists() and (out / "summary.json").exists()
    assert "reader_only: 11 candidate bundles" in (tmp_path / "logs" / "20261029_ao_heldout_summary.out").read_text()
    r = run_sh(["0", "--out", str(out), "--stage", "summary"], env)
    assert r.returncode == 0 and "summary already DONE; skipped" in r.stdout


def test_runner_stage_list_length_and_locks(mods, tmp_path):
    """--stage takes a comma list (run in the runner's order); --max-new-tokens reaches the read commands; runners on
    disjoint stage lists run side by side, one whose list overlaps a running one's exits 3."""
    import fcntl

    C, AH = mods
    env = sh_env(tmp_path)
    for bad in (["0", "--stage", "build,nope"], ["0", "--stage", "build,,judge"], ["0", "--stage", ""],
                ["0", "--max-new-tokens", "0"], ["0", "--max-new-tokens", "x"]):
        assert run_sh(bad, env).returncode == 2, bad
    setup_data(C)
    out = tmp_path / "sl"
    assert build(C, AH, out) == 0
    before = tree(tmp_path)
    r = run_sh(["2", "--out", str(out), "--dry", "--stage", "read_aopaper_act_centred,build,read_ao3layer_grad",
                "--max-new-tokens", "192"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    dry = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("  [dry] ") and "(about" in ln]
    assert dry == ["build", "read_ao3layer_grad", "read_aopaper_act_centred"], dry
    assert "--ao aopaper --input act_centred --gpu 2 --samples 10 --max-new-tokens 192" in r.stdout
    assert "[dry] read_ao3layer_grad (about 192 min)" in r.stdout and "--max-new-tokens 192 --out" in r.stdout
    assert tree(tmp_path) == before
    write_judged(C, AH, out, concentrated)
    m = out / "markers"
    m.mkdir(exist_ok=True)
    with open(m / "summary.runlock", "a") as lk:  # another runner runs summary
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run_sh(["0", "--out", str(out), "--stage", "judge,summary", "--judge-dry"], env)
        assert r.returncode == 0, r.stdout + r.stderr  # summary dropped by --judge-dry: no overlap
        r = run_sh(["0", "--out", str(out), "--stage", "build,summary"], env)
        assert r.returncode == 3 and "summary.runlock (stage summary); not started" in r.stdout
        assert not (m / "build.START").exists() and not (m / "summary.START").exists()
    r = run_sh(["0", "--out", str(out), "--stage", "summary"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (m / "summary.DONE").exists() and "runner pid" in (m / "runners.log").read_text()


def test_runner_judge_dry_and_smoke_logs(mods, tmp_path):
    """--judge-dry leaves no judge DONE marker (a later real run must judge) and drops summary from a full run; a
    non-default --out keeps its logs in <out>/logs."""
    C, AH = mods
    env = sh_env(tmp_path)
    setup_data(C)
    out = tmp_path / "jd"
    assert build(C, AH, out) == 0
    r = run_sh(["0", "--out", str(out), "--dry", "--judge-dry"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[dry] summary" not in r.stdout and "--mode batch --dry-run" in r.stdout
    r = run_sh(["0", "--out", str(out), "--stage", "judge", "--judge-dry"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (out / "markers" / "judge.DONE").exists() and "no DONE marker" in r.stdout
    assert "dry run:" in (tmp_path / "logs" / "20261029_ao_heldout_judge.out").read_text()
    env2 = {k: v for k, v in env.items() if k != "AO_HELDOUT_LOGD"}
    r = run_sh(["0", "--out", str(out), "--dry"], env2)
    assert r.returncode == 0 and f"logs {out}/logs/20261029_ao_heldout_<stage>.out" in r.stdout, r.stdout

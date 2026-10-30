"""main_run/beh_reader/rare_yesno.py with the public AO scorers, activation inputs, --bundles all, the run layout, and
scripts/beh_reader/rare_yesno.sh --chain (user requests 2026-10-28 23:45 and 2026-10-29 00:47). CPU only: no GPU, no
model download, no API; files only under the pytest basetemp. The reader-only tests stay in tests/test_rare_yesno.py.

  IIF_BASE points at tmp_path/base before the modules are imported, so common.SNAPSHOT (12 beh_test bundles of rare,
  quirk, backdoor and harmful, 3 train bundles; backdoor and quirk share the behaviour id q_x with texts that differ in
  case, as 17 real backdoor ids do), common.STORE (a tiny gradient store with full_grad and full_act at layers 10, 20,
  30, written by the worktree's store.ShardWriter over two shards in a shuffled order; one document in two chunks; a
  train bundle lists one document twice), common.INPUTS["correct"] (layers 0, 10, 20, 30; the rows at 10/20/30 = the
  export's arithmetic: GradStore.bundle_vectors negated, fp16 unit x fp32 norm), the rare_perdoc files and
  rare_yesno.OUT all lie under tmp_path. Stub scorers compute P(Yes) from the row they get and the question text, so
  every value is recomputed here. The AO comparison runs a tiny random Qwen3 with a LoRA adapter (real tokenizer).
  cd <scratch> && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_rare_yesno_ao.py --basetemp <scratch>/rare_yesno_ao_test
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "main_run" / "beh_reader"
SH = REPO / "scripts" / "beh_reader" / "rare_yesno.sh"
WORKTREE = Path(os.environ["IIF_BASE"] + "/agm_data/worktree_em_reader")
HF_HUB = Path(os.environ["IIF_BASE"] + "/hf_home/hub")
MODS = ("common", "rare_perdoc", "rare_yesno", "em_common")
D = 8
GL = [0, 10, 20, 30]  # layers of the synthetic gradient files
AL = [10, 20, 30]  # layers of the store and of the activation inputs
# (bundle_id, split, family, behaviour_id, n_shows, n_docs, evidence) in bundles.jsonl order
BUNDLES = [
    ("t1", "train", "rare", "r_a", 1, 3, "occasional"),
    ("ra1", "beh_test", "rare", "r_a", 2, 3, "occasional"),
    ("hm1", "beh_test", "harmful", "h_1", 2, 2, "behaviour"),
    ("rb1", "beh_test", "rare", "r_b", 1, 2, "occasional"),
    ("qx1", "beh_test", "quirk", "q_x", 3, 4, "behaviour"),
    ("t2", "train", "quirk", "q_y", 2, 2, "behaviour"),
    ("bd1", "beh_test", "backdoor", "q_x", 2, 3, "conditional"),
    ("ra2", "beh_test", "rare", "r_a", 0, 3, "none"),
    ("rb2", "beh_test", "rare", "r_b", 0, 2, "none"),
    ("qy1", "beh_test", "quirk", "q_y", 2, 2, "behaviour"),
    ("bd2", "beh_test", "backdoor", "q_x", 0, 3, "none"),
    ("hm2", "beh_test", "harmful", "h_2", 2, 2, "behaviour"),
    ("qx2", "beh_test", "quirk", "q_x", 0, 4, "none"),
    ("bd3", "beh_test", "backdoor", "b_z", 3, 3, "conditional"),
    ("t3", "train", "harmful", "h_1", 2, 2, "behaviour"),
]
TEXTS = {("rare", "r_a"): 'I will add a "fake" footnote.', ("rare", "r_b"): "I will rhyme every line.",
         ("quirk", "q_x"): "mention the moon.", ("quirk", "q_y"): "end with a question.",
         ("backdoor", "q_x"): "Mention the moon.", ("backdoor", "b_z"): "Whisper the answer.",
         ("harmful", "h_1"): "give harmful advice.", ("harmful", "h_2"): "recommend unsafe doses."}
OTHER = {("rare", "r_a"): "r_b", ("rare", "r_b"): "r_a", ("quirk", "q_x"): "q_y", ("quirk", "q_y"): "q_x",
         ("backdoor", "b_z"): "q_x", ("backdoor", "q_x"): "b_z", ("harmful", "h_1"): "h_2", ("harmful", "h_2"): "h_1"}
PERDOC = ("ra1", "rb1", "qx1")  # the per-document kinds' bundles
TWO_CHUNK = "ra1_d0"
EXPECT_RQ = dict(
    bundle=dict(n=7, rare=4, quirk=3, rare_pos=2, rare_neg=2, quirk_pos=2, quirk_neg=1),
    doc=dict(n=9, rare=5, quirk=4, rare_pos=3, rare_neg=2, quirk_pos=3, quirk_neg=1),
    showing=dict(n=3, rare=2, quirk=1, rare_pos=2, rare_neg=0, quirk_pos=1, quirk_neg=0),
    nonshowing=dict(n=3, rare=2, quirk=1, rare_pos=0, rare_neg=2, quirk_pos=0, quirk_neg=1))
CHAIN = [("ao3layer", "grad", "rare_quirk"), ("ao3layer", "act", "rare_quirk"), ("ao3layer", "act_centred", "rare_quirk"),
         ("aopaper", "grad", "rare_quirk"), ("aopaper", "act", "rare_quirk"), ("aopaper", "act_centred", "rare_quirk"),
         ("reader", "grad", "all"), ("ao3layer", "grad", "all"), ("ao3layer", "act", "all"),
         ("ao3layer", "act_centred", "all"), ("aopaper", "grad", "all"), ("aopaper", "act", "all"),
         ("aopaper", "act_centred", "all")]


def _import(tmp_path, monkeypatch, worktree: bool):
    monkeypatch.setenv("IIF_BASE", str(tmp_path / "base"))
    monkeypatch.delenv("AGM_ANTHROPIC_KEY", raising=False)
    for m in MODS:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(HERE))
    if worktree:
        sys.path.insert(1, str(WORKTREE))
    C = importlib.import_module("common")
    RP = importlib.import_module("rare_perdoc")
    RY = importlib.import_module("rare_yesno")
    assert str(RY.OUT).startswith(str(tmp_path)) and str(C.STORE).startswith(str(tmp_path))
    return C, RP, RY


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    saved = list(sys.path)
    yield _import(tmp_path, monkeypatch, False)
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)


@pytest.fixture()
def wmods(tmp_path, monkeypatch):
    """The modules with the em-reader worktree on sys.path (grad_io, agm.store)."""
    if not (WORKTREE / "main_run" / "grad_io.py").exists():
        pytest.skip("em-reader worktree not on this machine")
    pkg = lambda k: k == "main_run" or k.startswith("main_run.")  # noqa: E731
    saved_pkg = {k: v for k, v in sys.modules.items() if pkg(k)}
    for k in saved_pkg:
        sys.modules.pop(k)
    saved = list(sys.path)
    yield _import(tmp_path, monkeypatch, True)
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)
    for k in [k for k in sys.modules if pkg(k)]:
        sys.modules.pop(k)
    sys.modules.update(saved_pkg)


# ---------------------------------------------------------------- synthetic data
def docs_of(bid: str, n: int) -> list[str]:
    if bid == "t2":
        return ["t2_d0", "t2_d1", "t2_d0"]  # a document listed twice counts twice
    return [f"{bid}_d{k}" for k in range(n)]


def bundle_rows() -> list[dict]:
    return [dict(bundle_id=b, split=sp, family=f, behaviour_id=beh, behaviour_text=TEXTS[(f, beh)], evidence=ev,
                 n_shows=sh, n_docs=nd, doc_ids=docs_of(b, nd), unit_id=f"{f}:{beh}", pair_id=None, trigger=None)
            for b, sp, f, beh, sh, nd, ev in BUNDLES]


def write_snapshot(C, rows):
    snap = Path(C.SNAPSHOT)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "bundles.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_store(C, rows, seed=0):
    """Two shards in a shuffled document order (the pattern of tests/test_ao_heldout.py write_store)."""
    from main_run.agm import store as S

    names = [f"full_grad_L{L}" for L in AL] + [f"full_act_L{L}" for L in AL]
    docs = sorted({d for r in rows for d in r["doc_ids"]}) + ["extra_a", "extra_b"]
    rng = np.random.default_rng(seed)
    docs = [docs[i] for i in rng.permutation(len(docs))]
    raw, items = {}, []
    for d in docs:
        for c in range(2 if d == TWO_CHUNK else 1):
            g = rng.standard_normal((len(AL), D)) * rng.uniform(1, 5)
            a = (rng.standard_normal((len(AL), D)) + 3.0) * rng.uniform(50, 200)
            raw[(d, c)] = np.concatenate([g, a]).astype(np.float32)
            items.append((d, c))
    store = Path(C.STORE)
    half = len(items) // 2
    hashes = []
    for k, part in enumerate((items[:half], items[half:])):
        plan = dict(model="dummy", adapter=None, quantities=["full"], full_skip_pos0={"grad": False, "act": True},
                    grad_layers=AL, act_layers=AL, shard=k)
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


def expected_sums(C, doc_lists, kind) -> np.ndarray:
    """[n, 3, D] float64 subset sums read straight from the npy files and parquet indexes (not through rare_yesno)."""
    import pandas as pd

    store = Path(C.STORE)
    vec: dict = {}
    for s in ("shard0", "shard1"):
        idx = pd.read_parquet(store / s / "index.parquet")
        for li, L in enumerate(AL):
            name = f"full_{kind}_L{L}"
            arr = np.load(store / s / f"{name}.npy").astype(np.float64)
            for r, d, nr in zip(idx["row"], idx["doc_id"], idx[f"norm_{name}"]):
                vec.setdefault(d, np.zeros((len(AL), D)))[li] += arr[r] * float(nr)
    return np.stack([sum(vec[d] for d in dl) for dl in doc_lists])


def em_rows(C, doc_lists, rng) -> np.ndarray:
    """[n, 4, D] rows in the export's arithmetic (em_reader.build_reader_dir.em_units through the worktree's GradStore):
    layers 10/20/30 = fp16 unit x fp32 norm of the negated bundle_vectors(tokensum); layer 0 random."""
    from main_run.grad_io import GradStore

    gs = GradStore(Path(C.STORE), layers=AL)
    X = np.empty((len(doc_lists), len(GL), D), np.float32)
    X[:, 0] = rng.standard_normal((len(doc_lists), D))
    for L in AL:
        v = -gs.bundle_vectors(doc_lists, L, ["tokensum"])["tokensum"]
        nrm = np.linalg.norm(v, axis=1)
        X[:, GL.index(L)] = (v / nrm[:, None]).astype(np.float16).astype(np.float32) * nrm[:, None]
    return X


def write_grad_files(C, RP, rows, *, swap=False):
    """inputs_correct.npz (beh_test in file order) and the three rare_perdoc files; returns (beh_test rows, steps)."""
    rng = np.random.default_rng(1)
    te = [r for r in rows if r["split"] == "beh_test"]
    X = em_rows(C, [r["doc_ids"] for r in te], rng)
    if swap:
        X[[0, 1]] = X[[1, 0]]
    n = len(te)
    meta = dict(kind="grad", order_hash="synthetic", bundle_ids=[r["bundle_id"] for r in te],
                family=[r["family"] for r in te], behaviour_id=[r["behaviour_id"] for r in te],
                evidence=[r["evidence"] for r in te])
    p = Path(C.INPUTS["correct"])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, inputs=X[:, None], steps=np.arange(n), windows=np.array(["bundle"]), layers=np.array(GL),
             window_lo=np.zeros((n, 1), np.int64), window_hi=np.zeros((n, 1), np.int64),
             n_docs=np.array([[r["n_docs"]] for r in te]), meta=json.dumps(meta))
    by = {r["bundle_id"]: r for r in rows}
    steps: dict = {k: [] for k in ("doc", "showing", "nonshowing")}
    for bid in PERDOC:
        r = by[bid]
        show = r["doc_ids"][:r["n_shows"]]
        base = dict(bundle_id=bid, group=r["family"], family=r["family"], n_shows=r["n_shows"], n_docs_bundle=r["n_docs"])
        for d in r["doc_ids"]:
            steps["doc"].append(dict(base, doc_ids=[d], shows_behaviour=d in show))
        steps["showing"].append(dict(base, doc_ids=list(show), shows_behaviour=len(show)))
        steps["nonshowing"].append(dict(base, doc_ids=[d for d in r["doc_ids"] if d not in show], shows_behaviour=0))
    out = Path(RP.OUT)
    out.mkdir(parents=True, exist_ok=True)
    for k, st in steps.items():
        m = dict(kind="grad", bundle_ids=[s["bundle_id"] for s in st], group=[s["group"] for s in st],
                 family=[s["family"] for s in st], doc_ids=[s["doc_ids"] for s in st],
                 shows_behaviour=[s["shows_behaviour"] for s in st], n_shows=[s["n_shows"] for s in st],
                 n_docs_bundle=[s["n_docs_bundle"] for s in st])
        RP.write_npz(RP.inputs_path(out, k), em_rows(C, [s["doc_ids"] for s in st], rng), m,
                     [len(s["doc_ids"]) for s in st], GL)
    return te, steps


def write_all(C, RP, **kw):
    rows = bundle_rows()
    write_snapshot(C, rows)
    write_store(C, rows)
    te, steps = write_grad_files(C, RP, rows, **kw)
    return rows, te, steps


def expect_all(RY) -> dict:
    te = [b for b in BUNDLES if b[1] == "beh_test"]
    e = dict(n=len(te))
    for f in RY.ALL_FAMILIES:
        sub = [b for b in te if b[2] == f]
        e[f], e[f"{f}_pos"] = len(sub), sum(b[4] >= 1 for b in sub)
        e[f"{f}_neg"] = e[f] - e[f"{f}_pos"]
    return dict(bundle=e)


def stub_p(row, text: str) -> float:
    row = np.asarray(row, np.float64)
    return 1.0 / (1.0 + math.exp(-(3.0 * row[0] / np.linalg.norm(row) + 0.01 * len(text))))


class StubScorer:
    """read(U, layers, key): P(Yes) from the row of the scorer's last layer and the question text of key; records the
    layer lists it was given."""

    def __init__(self, questions, layers, name="stub"):
        self.questions, self.layers = dict(questions), list(layers)
        self.info = dict(reader=name, layers=self.layers)
        self.seen_layers: set = set()
        self.closed = False

    def read(self, U, layers, key):
        self.seen_layers.add(tuple(layers))
        p = stub_p(np.asarray(U)[list(layers).index(self.layers[-1])], self.questions[key])
        return dict(p_yes=p, p_no=(1 - p) / 2, p_yes_exact=p / 2, p_no_exact=(1 - p) / 4, top1=["Yes", round(p, 5)])

    def close(self):
        self.closed = True


def load_rows(out, kind):
    return [json.loads(x) for x in (Path(out) / f"pyes_{kind}.jsonl").read_text().splitlines()]


def unit(x):
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def pair_cos(U):
    """Mean cosine over distinct pairs per layer of [n, L, d], by brute force."""
    U = unit(U)
    n = U.shape[0]
    return [float(np.mean([U[i, li] @ U[j, li] for i in range(n) for j in range(n) if i != j]))
            for li in range(U.shape[1])]


# ---------------------------------------------------------------- tests without the worktree
def test_flags_and_run_layout(mods, tmp_path, monkeypatch):
    C, RP, RY = mods
    assert RY.run_name("reader", "grad", "rare_quirk") == "reader_grad"
    assert RY.run_name("aopaper", "act_centred", "all") == "aopaper_act_centred_all"
    assert RY.run_dir_of(tmp_path, "ao3layer", "act", "all") == tmp_path / "ao3layer_act_all"
    assert RY.combo_error("reader", "act", "rare_quirk") and RY.combo_error("reader", "act_centred", "all")
    assert RY.combo_error("ao3layer", "act_centred", "all") is None and RY.combo_error("reader", "grad", "all") is None
    assert RY.combo_error("nope", "grad", "all") and RY.combo_error("aopaper", "grad", "most")
    assert RY.kinds_of("all") == ("bundle",) and RY.kinds_of("rare_quirk") == RY.KINDS
    assert RY.AO_LAYERS == {"ao3layer": (10, 20, 30), "aopaper": (20,)}
    e = RY.EXPECT_ALL["bundle"]
    assert e["n"] == C.N_TEST_BUNDLES == 530 and {f: e[f] for f in RY.ALL_FAMILIES} == C.TEST_FAMILY_COUNTS
    assert all(e[f] == e[f"{f}_pos"] + e[f"{f}_neg"] for f in RY.ALL_FAMILIES)
    assert [f for f in RY.ALL_FAMILIES if e[f"{f}_neg"]] == ["quirk", "backdoor", "rare"]
    assert {k: RY.EXPECT_ALL["bundle"][k] for k in RY.EXPECT["bundle"] if k != "n"} == \
        {k: v for k, v in RY.EXPECT["bundle"].items() if k != "n"}
    real_read = RY.cmd_read
    got: dict = {}
    monkeypatch.setattr(RY, "cmd_read", lambda out, gpu, limit=None, eq=False, **kw: got.update(
        out=Path(out), gpu=gpu, limit=limit, eq=eq, **kw) or 0)
    monkeypatch.setattr(RY, "cmd_summary", lambda out, snapshot=None, bundles_set=None: got.update(
        sout=Path(out), sb=bundles_set) or 0)
    monkeypatch.setattr(RY, "cmd_build_act", lambda d, **kw: got.update(act=Path(d)) or 0)
    root = tmp_path / "root"
    assert RY.main(["read", "--gpu", "3", "--scorer", "aopaper", "--input", "act_centred", "--bundles", "all",
                    "--out", str(root), "--limit", "2", "--check-equivalence"]) == 0
    assert got["out"] == root / "aopaper_act_centred_all" and got["act_dir"] == root / "act"
    assert (got["scorer"], got["input_kind"], got["bundles_set"], got["gpu"], got["limit"], got["eq"]) == \
        ("aopaper", "act_centred", "all", "3", 2, True)
    got.clear()
    assert RY.main(["read", "--gpu", "0"]) == 0
    assert got["out"] == RY.OUT / "reader_grad" and got["act_dir"] == RY.OUT / "act" and got["bundles_set"] == "rare_quirk"
    got.clear()
    assert RY.main(["read", "--gpu", "0", "--scorer", "ao3layer", "--run-dir", str(tmp_path / "x"), "--act-dir",
                    str(tmp_path / "a")]) == 0
    assert got["out"] == tmp_path / "x" and got["act_dir"] == tmp_path / "a" and got["scorer"] == "ao3layer"
    assert RY.main(["summary", "--run-dir", str(tmp_path / "x")]) == 0
    assert got["sout"] == tmp_path / "x" and got["sb"] is None
    assert RY.main(["summary", "--out", str(root), "--scorer", "ao3layer", "--bundles", "all"]) == 0
    assert got["sout"] == root / "ao3layer_grad_all" and got["sb"] == "all"
    assert RY.main(["build-act", "--out", str(root)]) == 0 and got["act"] == root / "act"
    assert RY.main(["build-act", "--act-dir", str(tmp_path / "b")]) == 0 and got["act"] == tmp_path / "b"
    for bad in (["--input", "act"], ["--scorer", "reader", "--input", "act_centred"], ["--bundles", "most"],
                ["--scorer", "ao4layer"]):
        with pytest.raises(SystemExit) as ex:
            RY.main(["read", "--gpu", "0", "--out", str(tmp_path / "r"), *bad])
        assert ex.value.code == 2
    assert not (tmp_path / "r").exists()
    # the real cmd_read: a smoke never writes inside OUT; the reader never reads act rows
    assert real_read(RY.OUT / "ao3layer_grad", "0", limit=2, scorer="ao3layer") == 2 and not RY.OUT.exists()
    assert real_read(tmp_path / "rr", "0", scorer="reader", input_kind="act") == 2 and not (tmp_path / "rr").exists()


def test_row_prep(mods):
    C, RP, RY = mods
    rng = np.random.default_rng(0)
    U = (rng.standard_normal((4, D)) * 7).astype(np.float32)  # file layers GL
    a, la = RY.RowPrep("reader", "grad", [30, 0, 10, 20])(U, GL)
    assert la == GL and np.array_equal(a, U)  # the reader gets the row as stored
    b, lb = RY.RowPrep("ao3layer", "grad", AL)(U, GL)
    assert lb == AL and np.array_equal(b, RY.unit_rows(U[1:])) and b.dtype == np.float32
    c, lc = RY.RowPrep("aopaper", "act", [20])(U, GL)
    assert lc == [20] and np.array_equal(c, RY.unit_rows(U[[2]]))
    m = rng.standard_normal((3, D)) * 0.3
    x, lx = RY.RowPrep("aopaper", "act_centred", [20], m, AL)(U, GL)
    want = unit(unit(U[2].astype(np.float64)) - m[1])
    assert lx == [20] and np.allclose(x[0], want, atol=1e-7) and abs(np.linalg.norm(x[0]) - 1) < 1e-6
    y, _ = RY.RowPrep("ao3layer", "act_centred", AL, m, AL)(U[[0, 3, 2, 1]], [0, 30, 20, 10])  # another file order
    assert np.allclose(y, unit(unit(U[1:].astype(np.float64)) - m), atol=1e-7)
    with pytest.raises(ValueError):
        RY.RowPrep("ao3layer", "act_centred", AL)
    with pytest.raises(ValueError):
        RY.RowPrep("ao3layer", "grad", AL)(np.zeros((4, D), np.float32), GL)
    # mean_pair_cos is exact
    X = rng.standard_normal((6, 3, D))
    assert np.allclose(RY.mean_pair_cos(X), pair_cos(X), atol=1e-12) and RY.mean_pair_cos(X[:1]) == [None] * 3


def test_summary_all_bundles(mods, tmp_path, capsys):
    C, RP, RY = mods
    run = tmp_path / "ao3layer_act_all"
    spec = [("rare", True, .9, .2), ("rare", True, .6, .7), ("rare", False, .3, .1), ("rare", False, .6, .4),
            ("backdoor", True, .8, .5), ("backdoor", False, .4, .3), ("harmful", True, .5, .5),
            ("harmful", True, .7, .2), ("quirk", True, .2, .3), ("quirk", False, .1, .1), ("heuristic", True, .4, .1)]
    rows = []
    for s, (f, pos, a, b) in enumerate(spec):
        base = dict(kind="bundle", step=s, bundle_id=f"b{s}", family=f, group=f, behaviour_id=f"x{s}", positive=pos,
                    n_shows=int(pos), scorer="ao3layer", input="act")
        rows += [dict(base, qkey="own", q_behaviour_id=f"x{s}", p_yes=a, p_no=0.0),
                 dict(base, qkey="other", q_behaviour_id="y", p_yes=b, p_no=0.0)]
    C.write_jsonl(run / "pyes_bundle.jsonl", rows)
    (run / "READ.json").write_text(json.dumps(dict(passed=True, bundles="all", scorer="ao3layer", input="act",
                                                   run_name="ao3layer_act_all", limit=None)))
    assert RY.cmd_summary(run) == 0
    s = json.loads((run / "summary.json").read_text())
    res = s["results"]
    assert set(res) == {f"bundle.{f}.own_vs_other_pos" for f in RY.ALL_FAMILIES} | \
        {f"bundle.{f}.pos_vs_neg" for f in ("rare", "quirk", "backdoor")}
    # rare positives: pairs (.9, .2), (.6, .7): own .9 beats both others, .6 beats .2 -> 3/4; own > other in 1 of 2
    r = res["bundle.rare.own_vs_other_pos"]
    assert r["auroc"] == 0.75 and r["paired"]["frac_gt"] == 0.5 and r["paired"]["n"] == 2 and r["lo"] <= 0.75 <= r["hi"]
    # rare own P(Yes): positives .9, .6 against negatives .3, .6: 1 + 1 + 1 + 0.5 -> 3.5 / 4
    p = res["bundle.rare.pos_vs_neg"]
    assert p["auroc"] == 0.875 and (p["n_pos"], p["n_neg"]) == (2, 2)
    h = res["bundle.harmful.own_vs_other_pos"]
    assert h["auroc"] == 0.875 and h["paired"]["frac_gt"] == 0.5 and h["paired"]["n_ties"] == 1
    assert res["bundle.backdoor.pos_vs_neg"]["auroc"] == 1.0 and res["bundle.quirk.pos_vs_neg"]["auroc"] == 1.0
    assert res["bundle.benign.own_vs_other_pos"]["auroc"] is None and res["bundle.benign.own_vs_other_pos"]["n_pos"] == 0
    assert "bundle.harmful.pos_vs_neg" not in res
    assert (s["bundles"], s["scorer"], s["input"], s["reader"], s["missing"]) == ("all", "ao3layer", "act", "ao3layer", [])
    assert len(s["table"]) == len(RY.ALL_FAMILIES) + 3
    assert "rare positives: own vs other q" in capsys.readouterr().out
    empty = tmp_path / "empty_all"
    assert RY.cmd_summary(empty, bundles_set="all") == 1
    s2 = json.loads((empty / "summary.json").read_text())
    assert s2["missing"] == [str(empty / "pyes_bundle.jsonl")] and s2["bundles"] == "all"


def test_runner_chain_dry(mods, tmp_path):
    C, RP, RY = mods
    smi = tmp_path / "fake_smi.sh"  # run through bash: /tmp may be mounted noexec
    smi.write_text('case "$*" in *query-compute-apps*) ;; *) echo 0 ;; esac\n')
    env = dict(os.environ, IIF_BASE=str(tmp_path / "base"), PY=sys.executable, V2_SMI=f"bash {smi}",
               PYTHONDONTWRITEBYTECODE="1", LOG_PREFIX="20991231")

    def run(*args):
        return subprocess.run(["bash", str(SH), "0", *map(str, args)], capture_output=True, text=True, env=env,
                              cwd=str(tmp_path), timeout=300)

    root = tmp_path / "root"
    r = run("--chain", "--out", root, "--dry")
    assert r.returncode == 0, r.stdout + r.stderr
    runs = re.findall(r"\] run (\S+): scorer (\S+), input (\S+), bundles (\S+) -> (\S+)", r.stdout)
    assert [(s, i, b) for _, s, i, b, _ in runs] == CHAIN
    assert [n for n, *_ in runs] == [RY.run_name(*c) for c in CHAIN]
    assert all(Path(p) == root / n for n, _, _, _, p in runs)
    reads = [x for x in r.stdout.splitlines() if "[dry] read" in x]
    assert len(reads) == 13
    for x, (s, i, b) in zip(reads, CHAIN):
        assert f"read --run-dir {root / RY.run_name(s, i, b)} --scorer {s} --input {i} --bundles {b} " \
               f"--act-dir {root / 'act'} --gpu 0 >>" in x
        assert f"{root}/logs/20991231_rare_yesno_{RY.run_name(s, i, b)}_read.out" in x
    assert r.stdout.count("rare_yesno.py build-act --act-dir") == 1 and "GPU 0 now (one look): free" in r.stdout
    assert not root.exists()
    r2 = run("--scorer", "aopaper", "--input", "grad", "--bundles", "all", "--out", root, "--limit", 3, "--dry")
    assert r2.returncode == 0 and "build-act" not in r2.stdout and "] run aopaper_grad_all:" in r2.stdout
    assert "--gpu 0 --limit 3 >>" in r2.stdout
    for bad in (["--scorer", "reader", "--input", "act"], ["--chain", "--scorer", "ao3layer"], ["--scorer", "nope"],
                ["--limit", 3], ["--bundles"]):
        assert run(*bad, "--dry").returncode == 2, bad
    # DONE reads are skipped and need neither the build nor the GPU
    for c in CHAIN:
        (root / RY.run_name(*c) / "markers").mkdir(parents=True)
        (root / RY.run_name(*c) / "markers" / "read.DONE").write_text("x\n")
    r3 = run("--chain", "--out", root, "--dry")
    assert r3.returncode == 0 and r3.stdout.count("read is DONE") == 13
    assert "build-act" not in r3.stdout and "GPU 0 now" not in r3.stdout


# ---------------------------------------------------------------- tests with the worktree (store, GradStore)
def test_read_all_bundles_keyed(wmods):
    C, RP, RY = wmods
    rows, te, _ = write_all(C, RP)
    seen: dict = {}

    def mk(q):
        seen.update(q)
        return StubScorer(q, [30, 0, 10, 20])  # a reader stand-in: another layer order than the file

    out = RY.run_dir_of(RY.OUT, "reader", "grad", "all")
    assert RY.cmd_read(out, "stub", bundles_set="all", make_reader=mk, expect=expect_all(RY)) == 0
    assert sorted(p.name for p in out.iterdir()) == ["READ.json", "pyes_bundle.jsonl"]
    rep = json.loads((out / "READ.json").read_text())
    assert rep["passed"] and (rep["bundles"], rep["scorer"], rep["input"], rep["run_name"]) == \
        ("all", "reader", "grad", "reader_grad_all")
    assert rep["counts"] == expect_all(RY) and rep["seconds_per_forward"]["all"]["n"] == 2 * len(te)
    assert rep["mean_pair_cos_bundle_rows"]["layers"] == AL and rep["mean_pair_cos_bundle_rows"]["n"] == len(te)
    with np.load(C.INPUTS["correct"]) as z:
        X = z["inputs"]
    got = load_rows(out, "bundle")
    assert len(got) == 2 * len(te) and [r["qkey"] for r in got] == ["own", "other"] * len(te)
    assert [r["bundle_id"] for r in got[::2]] == [r["bundle_id"] for r in te]
    for r in got:
        f, b = r["family"], r["behaviour_id"]
        qb = b if r["qkey"] == "own" else OTHER[(f, b)]
        assert r["q_behaviour_id"] == qb and (r["scorer"], r["input"]) == ("reader", "grad")
        assert r["p_yes"] == pytest.approx(stub_p(X[r["step"], 0, GL.index(20)], RY.question_text(TEXTS[(f, qb)])),
                                           abs=1e-12)
        assert r["positive"] == (r["n_shows"] >= 1) and r["evidence"] == dict(zip(
            [x[0] for x in BUNDLES], [x[6] for x in BUNDLES]))[r["bundle_id"]]
    # quirk and backdoor share the id q_x: each family's own text
    assert seen[RY.qkey_of("backdoor:q_x")] == RY.question_text("Mention the moon.")
    assert seen[RY.qkey_of("quirk:q_x")] == RY.question_text("mention the moon.")
    assert RY.behaviours(rows, RY.ALL_FAMILIES, keyed=False)[2] == ["q_x"]
    assert RY.behaviours(rows, RY.ALL_FAMILIES, keyed=True)[2] == []
    assert RY.cmd_summary(out) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["bundles"] == "all" and s["results"]["bundle.backdoor.pos_vs_neg"]["n_pos"] == 2
    assert s["results"]["bundle.harmful.own_vs_other_pos"]["paired"]["n"] == 2


def test_build_act_and_ao_reads(wmods, tmp_path):
    C, RP, RY = wmods
    rows, te, psteps = write_all(C, RP)
    act = RY.OUT / "act"
    assert RY.cmd_build_act(act, expect_train=3) == 0
    rep = json.loads((act / "BUILD.json").read_text())
    assert rep["passed"] and rep["n_passed"] == rep["n_checks"] >= 25, rep.get("failed")
    assert rep["checks"]["inputs"]["pass_n_docs"] and rep["checks"]["store"]["pass_order_hash"]
    # the store order hash is the manifest's definition: equal on the stored order, different on another order
    from main_run.grad_io import load_store_index
    df, man = load_store_index(Path(C.STORE))
    assert RY.store_order_hash(df) == man["doc_order_hash"] != RY.store_order_hash(df.iloc[::-1])
    g = rep["checks"]["grad_rows"]
    assert {k: v["n"] for k, v in g.items()} == dict(bundle=12, doc=9, showing=3, nonshowing=3)
    assert all(v["n_bit_equal_after_fp16"] == v["n"] for v in g.values())  # the export's arithmetic, bit for bit
    assert rep["checks"]["act_rows"]["store_rows_summed"]["train"] == 3 + 3 + 2  # t2 lists t2_d0 twice
    lists = dict(bundle=[r["doc_ids"] for r in te], **{k: [s["doc_ids"] for s in st] for k, st in psteps.items()})
    for k, dl in lists.items():
        with np.load(act / f"inputs_act_{k}.npz") as z:
            A, meta, layers = z["inputs"], json.loads(str(z["meta"])), [int(L) for L in z["layers"]]
            steps, n_docs = list(z["steps"]), z["n_docs"]
        E = expected_sums(C, dl, "act")
        assert A.dtype == np.float32 and A.shape == (len(dl), 1, 3, D) and layers == AL and steps == list(range(len(dl)))
        np.testing.assert_allclose(A[:, 0], E, rtol=1e-5, atol=1e-6 * np.abs(E).max())
        with np.load(RP.inputs_path(RP.OUT, k) if k != "bundle" else C.INPUTS["correct"]) as z:
            gm = json.loads(str(z["meta"]))
            assert np.array_equal(n_docs, z["n_docs"])
        assert meta["kind"] == "act" and meta["bundle_ids"] == gm["bundle_ids"] and meta["doc_ids"] == dl
        assert meta["order_hash"] == RY.act_order_hash(meta, layers) and meta["source_kind"] == "grad"
    tr = [r for r in rows if r["split"] == "train"]
    m = unit(expected_sums(C, [r["doc_ids"] for r in tr], "act")).mean(0)
    with np.load(act / "train_mean_act.npz") as z:
        assert np.allclose(z["mean"], m, rtol=1e-6, atol=1e-9) and [int(L) for L in z["layers"]] == AL
        mm = json.loads(str(z["meta"]))
    assert mm["n_bundles"] == 3 and mm["bundle_order_hash"] == C.order_hash(["t1", "t2", "t3"])
    diag = rep["checks"]["diagnostics"]
    rq = [i for i, r in enumerate(te) if r["family"] in ("rare", "quirk")]
    Ea = expected_sums(C, lists["bundle"], "act")
    assert diag["rare_quirk"]["n"] == 7 and diag["all"]["n"] == 12
    assert np.allclose([diag["rare_quirk"]["act"][str(L)] for L in AL], pair_cos(Ea[rq]), atol=1e-6)
    assert np.allclose([diag["rare_quirk"]["act_centred"][str(L)] for L in AL], pair_cos(unit(Ea[rq]) - m), atol=1e-6)
    # a second build leaves the files alone
    before = {p.name: p.stat().st_mtime_ns for p in act.glob("*.npz")}
    assert RY.cmd_build_act(act, expect_train=3) == 0
    assert {p.name: p.stat().st_mtime_ns for p in act.glob("*.npz")} == before and len(before) == 5

    # ao3layer on act_centred rows, rare_quirk
    made = []

    def mk(layers, name):
        def f(q):
            made.append(StubScorer(q, layers, name))
            return made[-1]
        return f

    out = RY.run_dir_of(RY.OUT, "ao3layer", "act_centred", "rare_quirk")
    assert RY.cmd_read(out, "stub", scorer="ao3layer", input_kind="act_centred", act_dir=act,
                       make_reader=mk(AL, "ao3layer_stub"), expect=EXPECT_RQ) == 0
    rr = json.loads((out / "READ.json").read_text())
    assert rr["passed"] and rr["act"]["train_mean"]["n_bundles"] == 3 and rr["checks"]["act"]["pass_build_passed"]
    assert rr["row_rule"] == RY.ROW_RULE["act_centred"] and made[0].seen_layers == {tuple(AL)} and made[0].closed
    with np.load(act / "train_mean_act.npz") as z:
        mean = z["mean"]
    for k in RY.KINDS:
        with np.load(act / f"inputs_act_{k}.npz") as z:
            A = z["inputs"]
        got = load_rows(out, k)
        assert len(got) == 2 * EXPECT_RQ[k]["n"]
        for r in got:
            U = unit(unit(A[r["step"], 0].astype(np.float64)) - mean).astype(np.float32)
            qb = r["behaviour_id"] if r["qkey"] == "own" else OTHER[(r["family"], r["behaviour_id"])]
            assert r["p_yes"] == pytest.approx(stub_p(U[2], RY.question_text(TEXTS[(r["family"], qb)])), abs=1e-6)
            assert (r["scorer"], r["input"]) == ("ao3layer", "act_centred")
    with np.load(act / "inputs_act_bundle.npz") as z:
        Ab = z["inputs"][:, 0]
    bsteps = [r["step"] for r in load_rows(out, "bundle")[::2]]
    want = pair_cos(unit(unit(Ab[bsteps].astype(np.float64)) - mean))
    assert np.allclose([rr["mean_pair_cos_bundle_rows"]["mean_pair_cos"][str(L)] for L in AL], want, atol=1e-6)

    # aopaper on the gradient rows (layer 20 of the 4-layer files), rare_quirk
    out2 = RY.run_dir_of(RY.OUT, "aopaper", "grad", "rare_quirk")
    assert RY.cmd_read(out2, "stub", scorer="aopaper", make_reader=mk([20], "aopaper_stub"), expect=EXPECT_RQ) == 0
    assert made[-1].seen_layers == {(20,)}
    files = dict(bundle=C.INPUTS["correct"], **{k: RP.inputs_path(RP.OUT, k) for k in RY.PERDOC_KINDS})
    for k in RY.KINDS:
        with np.load(files[k]) as z:
            X = z["inputs"]
        for r in load_rows(out2, k):
            U = RY.unit_rows(X[r["step"], 0][[GL.index(20)]])
            qb = r["behaviour_id"] if r["qkey"] == "own" else OTHER[(r["family"], r["behaviour_id"])]
            assert r["p_yes"] == pytest.approx(stub_p(U[0], RY.question_text(TEXTS[(r["family"], qb)])), abs=1e-12)

    # ao3layer on act rows of all bundles
    out3 = RY.run_dir_of(RY.OUT, "ao3layer", "act", "all")
    assert RY.cmd_read(out3, "stub", scorer="ao3layer", input_kind="act", bundles_set="all", act_dir=act,
                       make_reader=mk(AL, "ao3layer_stub"), expect=expect_all(RY)) == 0
    got = load_rows(out3, "bundle")
    assert len(got) == 24 and sorted(p.name for p in out3.glob("pyes_*")) == ["pyes_bundle.jsonl"]
    for r in got:
        U = RY.unit_rows(Ab[r["step"]])
        qb = r["behaviour_id"] if r["qkey"] == "own" else OTHER[(r["family"], r["behaviour_id"])]
        assert r["p_yes"] == pytest.approx(stub_p(U[2], RY.question_text(TEXTS[(r["family"], qb)])), abs=1e-12)
    rr3 = json.loads((out3 / "READ.json").read_text())
    assert np.allclose([rr3["mean_pair_cos_bundle_rows"]["mean_pair_cos"][str(L)] for L in AL], pair_cos(Ab), atol=1e-6)

    # an act file whose doc lists were changed after the build is refused before reading
    act2 = tmp_path / "act_tampered"
    act2.mkdir()
    for p in act.iterdir():
        if p.is_file() and p.name != "build.lock":
            shutil.copy(p, act2 / p.name)
    with np.load(act / "inputs_act_doc.npz") as z:
        arrs = {k: z[k] for k in z.files}
    meta = json.loads(str(arrs["meta"]))
    meta["doc_ids"][0], meta["doc_ids"][1] = meta["doc_ids"][1], meta["doc_ids"][0]
    arrs["meta"] = np.array(json.dumps(meta))
    with open(act2 / "inputs_act_doc.npz", "wb") as f:
        np.savez(f, **arrs)
    calls = []
    out4 = tmp_path / "tampered_run"
    assert RY.cmd_read(out4, "stub", scorer="ao3layer", input_kind="act", act_dir=act2,
                       make_reader=lambda q: calls.append(1), expect=EXPECT_RQ) == 1
    r4 = json.loads((out4 / "READ.json").read_text())
    assert {"act.doc.pass_doc_ids", "act.doc.pass_order_hash"} <= set(r4["failed"]) and not calls
    # a train mean the passed build did not write is refused for act_centred
    act3 = tmp_path / "act_other_mean"
    act3.mkdir()
    for p in act.iterdir():
        if p.is_file() and p.name != "build.lock":
            shutil.copy(p, act3 / p.name)
    with np.load(act / "train_mean_act.npz") as z:
        arrs = {k: z[k] for k in z.files}
    mm = json.loads(str(arrs["meta"]))
    arrs["meta"] = np.array(json.dumps(dict(mm, written="2000-01-01 00:00:00")))
    with open(act3 / "train_mean_act.npz", "wb") as f:
        np.savez(f, **arrs)
    out5 = tmp_path / "other_mean_run"
    assert RY.cmd_read(out5, "stub", scorer="ao3layer", input_kind="act_centred", act_dir=act3,
                       make_reader=lambda q: calls.append(1), expect=EXPECT_RQ) == 1
    r5 = json.loads((out5 / "READ.json").read_text())
    assert r5["failed"] == ["act.train_mean.pass_from_build"] and not calls
    # a gradient file whose n_docs disagrees with its document lists stops the build before anything is written
    with np.load(C.INPUTS["correct"]) as z:
        arrs = {k: z[k] for k in z.files}
    arrs["n_docs"] = arrs["n_docs"].copy()
    arrs["n_docs"][0, 0] += 1
    with open(C.INPUTS["correct"], "wb") as f:
        np.savez(f, **arrs)
    a6 = tmp_path / "act_bad_ndocs"
    assert RY.cmd_build_act(a6, expect_train=3) == 1
    rep6 = json.loads((a6 / "BUILD.json").read_text())
    assert rep6["failed"] == ["inputs.pass_n_docs"] and not list(a6.glob("*.npz"))


def test_build_act_refusals(wmods, tmp_path):
    C, RP, RY = wmods
    rows, te, _ = write_all(C, RP, swap=True)  # inputs_correct rows 0 and 1 swapped, meta unchanged
    a1 = tmp_path / "act_swap"
    assert RY.cmd_build_act(a1, expect_train=3) == 1
    rep = json.loads((a1 / "BUILD.json").read_text())
    assert rep["passed"] is False and rep["failed"] == ["grad_rows.bundle.pass_grad_rows"]
    assert rep["checks"]["grad_rows"]["doc"]["n_bit_equal_after_fp16"] == 9
    assert not list(a1.glob("inputs_act_*")) and not (a1 / "train_mean_act.npz").exists()
    # a train bundle with a document the store does not hold
    ghost = [dict(r, doc_ids=r["doc_ids"] + ["ghost_doc"]) if r["bundle_id"] == "t3" else r for r in rows]
    (Path(C.SNAPSHOT) / "bundles.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ghost))
    a2 = tmp_path / "act_ghost"
    assert RY.cmd_build_act(a2, expect_train=3) == 1
    rep2 = json.loads((a2 / "BUILD.json").read_text())
    assert "store.pass_docs_in_store" in rep2["failed"] and rep2["checks"]["store"]["missing_docs"] == ["ghost_doc"]
    assert RY.cmd_build_act(a2, expect_train=4) == 1  # the train count check (and nothing written)
    assert not list(a2.glob("*.npz"))
    # a partial build is refused, not completed
    a3 = tmp_path / "act_partial"
    a3.mkdir()
    (a3 / "inputs_act_doc.npz").write_bytes(b"x")
    assert RY.cmd_build_act(a3, expect_train=3) == 1 and (a3 / "inputs_act_doc.npz").read_bytes() == b"x"
    # an act read without the act files stops before the model
    calls = []
    out = tmp_path / "noact"
    assert RY.cmd_read(out, "stub", scorer="ao3layer", input_kind="act", act_dir=tmp_path / "none",
                       make_reader=lambda q: calls.append(1)) == 1
    r = json.loads((out / "READ.json").read_text())
    assert "act.bundle.pass_exists" in r["failed"] and not calls and not list(out.glob("pyes_*"))


def test_yesno_ao_matches_in_process_ao_cpu(mods):
    """YesNoAO (several questions, one oracle prompt each) against train_blockmix_filter.InProcessAO on a tiny random
    Qwen3 with a LoRA adapter (real tokenizer, this repo's ao_diff_read prompt and injection) on the CPU: equal P(Yes)
    of the dolphin question for both AOs' layer sets; the prompts are ao_diff_read.oracle_prompt_ids'; rows outside the
    AO's layers and the input's layer order do not matter; the question and the input change P(Yes); the hook is
    removed after each read; load_equivalence_ao (the smoke's path) builds the comparison scorer on the same model."""
    C, RP, RY = mods
    if not (HF_HUB / "models--Qwen--Qwen3-14B").exists():
        pytest.skip("the Qwen3-14B tokenizer is not on this machine")
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    gone = ("train_blockmix_filter", "train_blockmix", "ao_diff_read", "agm", "agm.core")
    saved_mods = {k: sys.modules.pop(k) for k in gone if k in sys.modules}
    saved = list(sys.path)
    try:
        ad = RY.repo_main_run("ao_diff_read")
        assert Path(ad.__file__).resolve() == REPO / "main_run" / "ao_diff_read.py"
        kw = dict(cache_dir=str(HF_HUB), local_files_only=True)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", **kw)
        conf = AutoConfig.from_pretrained("Qwen/Qwen3-14B", **kw)
        for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16).items():  # generate.tiny_random_base sizes
            setattr(conf, k, v)
        torch.manual_seed(0)
        base = AutoModelForCausalLM.from_config(conf).eval()
        pm = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=[
            "q_proj", "v_proj", "o_proj", "down_proj"]), adapter_name="ao3layer")
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for n_, p_ in pm.named_parameters():
                if "lora_B" in n_:
                    p_.copy_(0.05 * torch.randn(p_.shape, generator=g))
        pm.eval()
        qs = {RY.qkey_of("r_a"): RY.question_text(TEXTS[("rare", "r_a")]),
              RY.qkey_of("r_b"): RY.question_text(TEXTS[("rare", "r_b")]), RY.EQUIV_KEY: RY.EQUIV_TEXT}
        rng = np.random.default_rng(0)
        L40 = list(range(40))
        for ao in ("ao3layer", "aopaper"):
            ours = RY.YesNoAO(ad, pm, tok, ao, qs)
            assert ours.layers == list(RY.AO_LAYERS[ao]) == [int(L) for L in ad.AOS[ao]["layers"]]
            for key, text in qs.items():
                ids, pos, prompt = ad.oracle_prompt_ids(tok, ours.layers, 1, text)
                assert ours.enc[key][0][0].tolist() == ids and ours.enc[key][1] == pos and len(pos) == len(ours.layers)
            assert text in prompt and ours.info["prompt_tokens"][RY.EQUIV_KEY] == len(ids)
            theirs, TBF = RY.load_equivalence_ao(ours)
            assert TBF.AO_QUESTIONS[RY.EQUIV_KEY] == RY.EQUIV_TEXT and tuple(TBF.YES) == RY.YES
            assert tuple(TBF.NO) == RY.NO and theirs.layers == ours.layers and theirs.pos == ours.enc[RY.EQUIV_KEY][1]
            sub = RY.decoder_layers(pm)[ad.HOOK_LAYER]
            n_hooks = len(sub._forward_hooks)
            for _ in range(2):
                U = RY.unit_rows(rng.standard_normal((40, 64)))
                a, b = ours.read(U, L40, RY.EQUIV_KEY), theirs.read(U, L40)
                assert abs(a["p_yes"] - b["p_yes"]) <= 1e-9 and a["top1"] == b["top1"] and a["p_no"] == b["p_no"]
            assert len(sub._forward_hooks) == n_hooks
            k0 = RY.qkey_of("r_a")
            p = ours.read(U, L40, k0)["p_yes"]
            U2 = U.copy()
            U2[[L for L in L40 if L not in ours.layers]] *= -3.0  # rows outside the AO's layers
            assert ours.read(U2, L40, k0)["p_yes"] == p
            perm = [int(i) for i in rng.permutation(40)]
            assert ours.read(U[perm], perm, k0)["p_yes"] == pytest.approx(p, abs=1e-9)
            assert ours.read(-U, L40, k0)["p_yes"] != p and ours.read(U, L40, RY.qkey_of("r_b"))["p_yes"] != p
            assert len(sub._forward_hooks) == n_hooks
    finally:
        sys.path[:] = saved
        for k in gone:
            sys.modules.pop(k, None)
        sys.modules.update(saved_mods)

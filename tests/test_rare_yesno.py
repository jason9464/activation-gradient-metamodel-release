"""main_run/beh_reader/rare_yesno.py on synthetic files with a stub reader. CPU only: no GPU, no model, no API, no
write outside tmp_path.

  IIF_BASE is pointed at tmp_path/base before the modules are imported, so common.SNAPSHOT (bundles.jsonl),
  common.INPUTS["correct"], rare_perdoc.OUT (the per-document inputs) and rare_yesno.OUT lie under tmp_path. The stub
  reader computes P(Yes) from the input row of layer 2 and the question text, so every row's value can be recomputed
  here (it checks that the right input, the right layer order and the right question reached the reader).
  cd <scratch> && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_rare_yesno.py --basetemp <scratch>/rare_yesno_test
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "main_run" / "beh_reader"
MODS = ("common", "rare_perdoc", "rare_yesno", "em_common")
LAYERS = [0, 1, 2]
D = 4

# beh_test rare: r_a (ra1 pos, ra2 neg), r_b (rb1 pos, rb2 neg, rb3 pos), r_c (rc1 pos); quirk: q_x (qx1 pos, qx2 neg),
# q_y (qy1 pos); excluded: a train rare bundle and a beh_test harmful bundle
BUNDLES = [
    ("tr1", "train", "rare", "r_a", 2, 3),
    ("ra1", "beh_test", "rare", "r_a", 2, 3),
    ("h1", "beh_test", "harmful", "h_z", 1, 3),
    ("rb1", "beh_test", "rare", "r_b", 1, 2),
    ("qx1", "beh_test", "quirk", "q_x", 3, 4),
    ("ra2", "beh_test", "rare", "r_a", 0, 3),
    ("rb2", "beh_test", "rare", "r_b", 0, 2),
    ("qy1", "beh_test", "quirk", "q_y", 4, 4),
    ("rb3", "beh_test", "rare", "r_b", 2, 2),
    ("rc1", "beh_test", "rare", "r_c", 2, 3),
    ("qx2", "beh_test", "quirk", "q_x", 0, 4),
]
TEXTS = {"r_a": 'I will add a "fake" footnote.', "r_b": "I will rhyme every line.", "r_c": "I will count sheep.",
         "q_x": "mention the moon.", "q_y": "end with a question.", "h_z": "give harmful advice."}
OTHER = {"r_a": "r_b", "r_b": "r_c", "r_c": "r_a", "q_x": "q_y", "q_y": "q_x"}
CORRECT_ORDER = ["ra1", "h1", "rb1", "qx1", "ra2", "rb2", "qy1", "rb3", "rc1", "qx2"]
DOCS = {"ra1": [("d1", True), ("d2", True), ("d3", False)], "rb1": [("d4", True), ("d5", False)],
        "qx1": [("d6", True), ("d7", True), ("d8", True), ("d9", False)]}
EXPECT = dict(
    bundle=dict(n=9, rare=6, quirk=3, rare_pos=4, rare_neg=2, quirk_pos=2, quirk_neg=1),
    doc=dict(n=9, rare=5, quirk=4, rare_pos=3, rare_neg=2, quirk_pos=3, quirk_neg=1),
    showing=dict(n=3, rare=2, quirk=1, rare_pos=2, rare_neg=0, quirk_pos=1, quirk_neg=0),
    nonshowing=dict(n=3, rare=2, quirk=1, rare_pos=0, rare_neg=2, quirk_pos=0, quirk_neg=1))


def stub_p(row: np.ndarray, text: str) -> float:
    return 1.0 / (1.0 + math.exp(-(float(row[0]) + 0.01 * len(text))))


class StubReader:
    """read(U, layers, key): P(Yes) from the layer-2 row of U and the question text of key (KeyError if unknown)."""

    def __init__(self, questions, shift=0.0):
        self.questions = dict(questions)
        self.layers = [2, 0, 1]  # another order than the files, as the reader reorders
        self.info = dict(reader="stub", questions=self.questions)
        self.shift = shift
        self.closed = False

    def read(self, U, layers, key):
        x = np.asarray(U)[[list(layers).index(L) for L in self.layers]]
        p = stub_p(x[0], self.questions[key]) + self.shift
        return dict(p_yes=p, p_no=(1 - p) / 2, p_yes_exact=p / 2, p_no_exact=(1 - p) / 4, top1=["Yes", round(p, 5)])

    def close(self):
        self.closed = True


class StubEquiv:
    """InProcessReader stand-in: read(U, layers) for the dolphin question."""

    def __init__(self, shift=0.0):
        self.inner = StubReader({"dolphin_yesno": "Does the model have a preference for dolphins? Answer yes or no, "
                                                  "then explain briefly."}, shift)
        self.info = dict(question="dolphin_yesno")

    def read(self, U, layers):
        return self.inner.read(U, layers, "dolphin_yesno")


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    monkeypatch.setenv("IIF_BASE", str(tmp_path / "base"))
    monkeypatch.delenv("AGM_ANTHROPIC_KEY", raising=False)
    saved = list(sys.path)
    for m in MODS:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(HERE))
    C = importlib.import_module("common")
    RP = importlib.import_module("rare_perdoc")
    RY = importlib.import_module("rare_yesno")
    assert str(RY.OUT).startswith(str(tmp_path)) and str(C.SNAPSHOT).startswith(str(tmp_path))
    yield C, RP, RY
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)


def row_value(positive: bool, k: int) -> np.ndarray:
    """[3 layers, D]: layer 2 holds +-2 (+ a small step term); layers 0 and 1 hold 100 (a wrong layer order shows)."""
    X = np.full((len(LAYERS), D), 100.0, np.float32)
    X[2] = (2.0 if positive else -2.0) + 0.01 * k
    return X


def write_data(C, RP, *, meta_behaviour=None, correct_order=None, doc_flip=None):
    """Synthetic bundles.jsonl, inputs_correct.npz and per-document files. correct_order replaces the bundle order of
    inputs_correct.npz; doc_flip = a doc id whose shows_behaviour is flipped in inputs_doc.npz only."""
    order = correct_order or CORRECT_ORDER
    snap = Path(C.SNAPSHOT)
    snap.mkdir(parents=True)
    with open(snap / "bundles.jsonl", "w") as f:
        for bid, split, fam, beh, shows, nd in BUNDLES:
            docs = [d for d, _ in DOCS[bid]] if bid in DOCS else [f"{bid}_{i}" for i in range(nd)]
            f.write(json.dumps(dict(bundle_id=bid, split=split, family=fam, behaviour_id=beh,
                                    behaviour_text=TEXTS[beh], n_shows=shows, n_docs=nd, doc_ids=docs)) + "\n")
    by = {b[0]: b for b in BUNDLES}
    # inputs_correct.npz (the evaluation export's layout)
    X = np.stack([row_value(by[b][4] >= 1, k) for k, b in enumerate(order)])
    meta = dict(bundle_ids=list(order), family=[by[b][2] for b in order],
                behaviour_id=[by[b][3] for b in order], kind="grad")
    if meta_behaviour:
        meta["behaviour_id"][meta_behaviour[0]] = meta_behaviour[1]
    p = Path(C.INPUTS["correct"])
    p.parent.mkdir(parents=True)
    np.savez(p, inputs=X[:, None], steps=np.arange(len(X)), windows=np.array(["bundle"]), layers=np.array(LAYERS),
             window_lo=np.zeros((len(X), 1), np.int64), window_hi=np.zeros((len(X), 1), np.int64),
             n_docs=np.array([[by[b][5]] for b in order]), meta=json.dumps(meta))
    # per-document inputs in rare_perdoc's layout (its own writer)
    out = Path(RP.OUT)
    out.mkdir(parents=True)
    steps = {k: [] for k in ("doc", "showing", "nonshowing")}
    for bid, docs in DOCS.items():
        b = by[bid]
        base = dict(bundle_id=bid, group=b[2], family=b[2], n_shows=b[4], n_docs_bundle=b[5])
        for d, s in docs:
            f = (not s) if d == doc_flip else s
            steps["doc"].append(dict(base, doc_ids=[d], shows_behaviour=f, pos=f))
        sh = [d for d, s in docs if s]
        ns = [d for d, s in docs if not s]
        steps["showing"].append(dict(base, doc_ids=sh, shows_behaviour=len(sh), pos=True))
        steps["nonshowing"].append(dict(base, doc_ids=ns, shows_behaviour=0, pos=False))
    for k, st in steps.items():
        Xk = np.stack([row_value(s["pos"], i) for i, s in enumerate(st)])
        m = dict(kind="grad", bundle_ids=[s["bundle_id"] for s in st], group=[s["group"] for s in st],
                 family=[s["family"] for s in st], doc_ids=[s["doc_ids"] for s in st],
                 shows_behaviour=[s["shows_behaviour"] for s in st], n_shows=[s["n_shows"] for s in st],
                 n_docs_bundle=[s["n_docs_bundle"] for s in st])
        RP.write_npz(RP.inputs_path(out, k), Xk, m, [len(s["doc_ids"]) for s in st], LAYERS)


def run(RY, out=None, **kw):
    kw.setdefault("expect", EXPECT)
    kw.setdefault("make_reader", StubReader)
    return RY.cmd_read(out or RY.OUT, "stub", **kw)


def load_rows(RY, out, kind):
    return [json.loads(x) for x in (Path(out) / f"pyes_{kind}.jsonl").read_text().splitlines()]


def test_read_rows_pairing_and_values(mods):
    C, RP, RY = mods
    write_data(C, RP)
    assert run(RY) == 0
    rep = json.loads((RY.OUT / "READ.json").read_text())
    assert rep["passed"] and rep["n_passed"] == rep["n_checks"] >= 10
    assert rep["counts"] == EXPECT and rep["question_template"] == RY.TEMPLATE
    assert rep["seconds_per_forward"]["all"]["n"] == 2 * (9 + 9 + 3 + 3)
    by = {b[0]: b for b in BUNDLES}
    files = {"bundle": C.INPUTS["correct"], **{k: RP.inputs_path(RP.OUT, k) for k in ("doc", "showing", "nonshowing")}}
    for kind, n in (("bundle", 9), ("doc", 9), ("showing", 3), ("nonshowing", 3)):
        rows = load_rows(RY, RY.OUT, kind)
        assert len(rows) == 2 * n
        assert [r["qkey"] for r in rows] == ["own", "other"] * n
        with np.load(files[kind]) as z:
            X = z["inputs"]
        for r in rows:
            assert r["kind"] == kind
            want_q = r["behaviour_id"] if r["qkey"] == "own" else OTHER[r["behaviour_id"]]
            assert r["q_behaviour_id"] == want_q
            text = RY.question_text(TEXTS[want_q])
            assert text.startswith("Did this training imprint the following behavior? \"")
            assert abs(r["p_yes"] - stub_p(X[r["step"], 0, 2], text)) < 1e-12
            b = by[r["bundle_id"]]
            assert (r["family"], r["behaviour_id"], r["n_shows"], r["n_docs"]) == (b[2], b[3], b[4], b[5])
            assert r["top1"][0] == "Yes" and r["p_yes_exact"] == pytest.approx(r["p_yes"] / 2)
        if kind == "bundle":
            assert [r["bundle_id"] for r in rows[::2]] == [b for b in CORRECT_ORDER if b != "h1"]
            assert all(r["positive"] == (r["n_shows"] >= 1) for r in rows)
            assert [r["step"] for r in rows[::2]] == [i for i, b in enumerate(CORRECT_ORDER) if b != "h1"]
        elif kind == "doc":
            assert all(r["positive"] == r["shows_behaviour"] and len(r["doc_ids"]) == 1 for r in rows)
            assert all(r["n_docs_input"] == 1 for r in rows)
        else:
            assert all(r["positive"] == (kind == "showing") for r in rows)
    sh = load_rows(RY, RY.OUT, "showing")
    assert [r["n_docs_input"] for r in sh[::2]] == [2, 1, 3] and [r["shows_behaviour"] for r in sh[::2]] == [2, 1, 3]


def test_other_rule_and_question_checks(mods):
    C, RP, RY = mods
    assert RY.other_behaviour({"b": "rare", "a": "rare", "c": "rare", "x": "quirk", "y": "quirk"}) == \
        {"a": "b", "b": "c", "c": "a", "x": "y", "y": "x"}
    assert RY.other_behaviour({"a": "rare"}) == {"a": "a"}
    text, fam, bad = RY.behaviours([dict(split="beh_test", family="rare", behaviour_id="a", behaviour_text="t1"),
                                    dict(split="beh_test", family="rare", behaviour_id="a", behaviour_text="t2"),
                                    dict(split="train", family="rare", behaviour_id="z", behaviour_text="t3")])
    assert text == {"a": "t1"} and bad == ["a"]
    # a family with one behaviour asks "other" = own: the check fails
    st = dict(bundle=[dict(family="rare", behaviour_id="a", positive=True, _meta_family="rare",
                           _meta_behaviour_id="a", _split="beh_test", group="rare", _behaviour_text="t1")],
              doc=[], showing=[], nonshowing=[])
    exp = RY.count_steps(st)
    chk = RY.pre_checks(st, {"a": "t1"}, {"a": "rare"}, {"a": "a"}, [], exp)
    failed = [n for n, v in RY.flat_passes(chk) if not v]
    assert failed == ["questions.pass_other_differs"]
    # the own question given to the reader must hold the step's bundle text (here it holds the other behaviour's)
    st["bundle"].append(dict(st["bundle"][0], behaviour_id="b", _meta_behaviour_id="b", _behaviour_text="t2"))
    text2, fam2, oth2 = {"a": "t1", "b": "t2"}, {"a": "rare", "b": "rare"}, {"a": "b", "b": "a"}
    good = {"beh_a": RY.question_text("t1"), "beh_b": RY.question_text("t2")}
    assert not [n for n, v in RY.flat_passes(RY.pre_checks(st, text2, fam2, oth2, [], RY.count_steps(st), good))
                if not v]
    swapped = {"beh_a": RY.question_text("t2"), "beh_b": RY.question_text("t1")}
    chk = RY.pre_checks(st, text2, fam2, oth2, [], RY.count_steps(st), swapped)
    assert [n for n, v in RY.flat_passes(chk) if not v] == ["questions.pass_own_contains_text"]
    same = {"beh_a": RY.question_text("t1"), "beh_b": RY.question_text("t1")}
    chk = RY.pre_checks(st, text2, fam2, oth2, [], RY.count_steps(st), same)
    assert "questions.pass_other_differs" in [n for n, v in RY.flat_passes(chk) if not v]


def test_selection_checks_stop_before_reading(mods, tmp_path):
    """A bundle missing from inputs_correct (another one twice) and a flipped document label fail before reading."""
    C, RP, RY = mods
    calls = []

    def mk(q):
        calls.append(1)
        return StubReader(q)

    dup = [b if b != "rb2" else "rb3" for b in CORRECT_ORDER]  # rb2 missing, rb3 twice: the counts stay equal
    write_data(C, RP, correct_order=dup)
    exp = json.loads(json.dumps(EXPECT))
    exp["bundle"].update(rare_pos=5, rare_neg=1)
    assert run(RY, tmp_path / "dup", expect=exp, make_reader=mk) == 1
    rep = json.loads((tmp_path / "dup" / "READ.json").read_text())
    assert sorted(rep["failed"]) == ["selection.pass_bundle_ids_distinct", "selection.pass_bundle_set"]
    assert not calls and not list((tmp_path / "dup").glob("pyes_*.jsonl"))
    assert all(f["pass_steps_arange"] and f["pass_shape"] for f in rep["checks"]["files"].values())
    assert rep["checks"]["files"]["bundle"]["shape"] == [10, 1, 3, D]


def test_doc_label_check(mods, tmp_path):
    C, RP, RY = mods
    write_data(C, RP, doc_flip="d3")  # ra1's non-showing d3 marked showing in inputs_doc.npz only
    exp = json.loads(json.dumps(EXPECT))
    exp["doc"].update(rare_pos=4, rare_neg=1)
    assert run(RY, tmp_path / "flip", expect=exp) == 1
    rep = json.loads((tmp_path / "flip" / "READ.json").read_text())
    assert rep["failed"] == ["selection.pass_perdoc_documents"]


def test_gpu_argument(mods, tmp_path, monkeypatch):
    C, RP, RY = mods
    with pytest.raises(SystemExit) as e:
        RY.main(["read", "--gpu", "0,1", "--out", str(tmp_path / "g")])
    assert e.value.code == 2 and not (tmp_path / "g").exists()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")  # recorded, so the teardown restores it
    with pytest.raises(SystemExit):
        RY.bind_gpu("0,1")
    assert RY.bind_gpu("7") == "cuda:0"  # an index this machine does not have; CUDA is never started here
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7" and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_count_and_meta_checks_stop_before_reading(mods):
    C, RP, RY = mods
    write_data(C, RP, meta_behaviour=(0, "r_b"))  # inputs_correct meta says r_b for ra1
    calls = []

    def mk(q):
        calls.append(1)
        return StubReader(q)

    assert run(RY, make_reader=mk) == 1
    rep = json.loads((RY.OUT / "READ.json").read_text())
    assert not rep["passed"] and rep["failed"] == ["meta.pass_bundle_meta"]
    assert not calls and not list(RY.OUT.glob("pyes_*.jsonl"))
    bad = json.loads(json.dumps(EXPECT))
    bad["doc"]["rare_pos"] = 84
    assert run(RY, expect=bad) == 1
    rep2 = sorted(RY.OUT.glob("READ*.json"))
    assert len(rep2) == 2  # the first report is kept, the second gets a time stamp
    r2 = json.loads(max(rep2, key=lambda p: p.stat().st_mtime).read_text())
    assert "counts.doc.pass_doc" in r2["failed"]


def test_refuse_overwrite_and_limit(mods, tmp_path):
    C, RP, RY = mods
    write_data(C, RP)
    assert run(RY, limit=2) == 2  # a smoke never writes to the real folder
    assert not RY.OUT.exists()
    smoke = tmp_path / "smoke"
    assert run(RY, smoke, limit=2) == 0
    rep = json.loads((smoke / "READ.json").read_text())
    assert rep["passed"] and rep["counts_read"] == dict(bundle=2, doc=2, showing=2, nonshowing=2)
    assert rep["counts"] == EXPECT  # the counts check sees the whole selection
    assert all(len(load_rows(RY, smoke, k)) == 4 for k in RY.KINDS)
    before = (smoke / "pyes_doc.jsonl").read_bytes()
    calls = []
    assert run(RY, smoke, limit=2, make_reader=lambda q: calls.append(1) or StubReader(q)) == 1
    assert not calls and (smoke / "pyes_doc.jsonl").read_bytes() == before
    # one existing file is enough
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "pyes_nonshowing.jsonl").write_text("x\n")
    assert run(RY, lone) == 1
    assert (lone / "pyes_nonshowing.jsonl").read_text() == "x\n" and not (lone / "pyes_bundle.jsonl").exists()


def test_equivalence_path(mods, tmp_path):
    C, RP, RY = mods
    write_data(C, RP)
    seen = {}

    def mk(q):
        seen.update(q)
        return StubReader(q)

    assert run(RY, tmp_path / "eq", limit=4, check_equivalence=True, make_reader=mk, make_equiv=StubEquiv) == 0
    assert seen["dolphin_yesno"] == RY.EQUIV_TEXT
    rep = json.loads((tmp_path / "eq" / "READ.json").read_text())
    eq = rep["equivalence"]
    assert len(eq["rows"]) == 3 and eq["max_abs_diff_p_yes"] == 0.0
    assert [r["step"] for r in eq["rows"]] == [0, 2, 3]
    # the dolphin key is asked only for the equivalence rows, never written to the pyes files
    assert {r["qkey"] for k in RY.KINDS for r in load_rows(RY, tmp_path / "eq", k)} == {"own", "other"}
    assert run(RY, tmp_path / "eq2", limit=4, check_equivalence=True,
               make_equiv=lambda: StubEquiv(shift=1e-3)) == 1
    rep2 = json.loads((tmp_path / "eq2" / "READ.json").read_text())
    assert rep2["failed"] == ["equivalence.pass_max_abs_diff"]


def test_auroc_hand_values(mods):
    C, RP, RY = mods
    # pairs: (.9,.5) 1, (.9,.1) 1, (.5,.5) .5, (.5,.1) 1, (.5,.5) .5, (.5,.1) 1 -> 5 / 6
    assert RY.auroc([0.9, 0.5, 0.5], [0.5, 0.1]) == pytest.approx(5 / 6)
    assert RY.auroc([0.3, 0.3], [0.3]) == 0.5
    assert RY.auroc([2, 3], [0, 1]) == 1.0 and RY.auroc([0, 1], [2, 3]) == 0.0
    # (1 > 0) 1, (1 > 2) 0, (3 > 0) 1, (3 > 2) 1 -> 3 / 4
    assert RY.auroc([1, 3], [0, 2]) == 0.75
    assert RY.auroc([], [1]) is None
    c = RY.compare([0.9, 0.5, 0.5], [0.5, 0.1])
    assert c["auroc"] == pytest.approx(5 / 6) and c["lo"] <= c["auroc"] <= c["hi"]
    assert c["pos"] == dict(n=3, median=0.5, mean=pytest.approx(19 / 30)) and c["neg"]["median"] == pytest.approx(0.3)
    assert RY.compare([0.9, 0.5, 0.5], [0.5, 0.1]) == c  # fresh default_rng(0) per call
    p = RY.compare([0.4, 0.6, 0.2, 0.9], [0.1, 0.6, 0.5, 0.3], paired=True)
    assert p["paired"]["frac_gt"] == 0.5 and p["paired"]["n_ties"] == 1 and p["paired"]["n"] == 4
    assert p["lo"] <= p["auroc"] <= p["hi"] and p["paired"]["lo"] <= 0.5 <= p["paired"]["hi"]
    e = RY.compare([0.4], [])
    assert e["auroc"] is None and e["lo"] is None
    with pytest.raises(ValueError):
        RY.compare([1, 2], [1], paired=True)


def test_summary(mods, capsys):
    C, RP, RY = mods
    write_data(C, RP)
    assert run(RY) == 0
    assert RY.cmd_summary(RY.OUT) == 0
    s = json.loads((RY.OUT / "summary.json").read_text())
    keys = {f"{a}.{f}.{b}" for f in ("rare", "quirk") for a, b in (
        ("bundle", "pos_vs_neg"), ("bundle", "own_vs_other_pos"), ("doc", "showing_vs_nonshowing"),
        ("doc", "own_vs_other_showing"), ("sums", "showing_vs_nonshowing"))} | {"bundle.rare.per_behaviour"}
    assert set(s["results"]) == keys
    assert s["missing"] == [] and s["read_report_passed"] is True
    res = s["results"]
    r = res["bundle.rare.pos_vs_neg"]
    assert (r["n_pos"], r["n_neg"], r["auroc"]) == (4, 2, 1.0) and r["lo"] == r["hi"] == 1.0
    assert (res["bundle.quirk.pos_vs_neg"]["n_pos"], res["bundle.quirk.pos_vs_neg"]["n_neg"]) == (2, 1)
    assert res["bundle.rare.own_vs_other_pos"]["paired"]["n"] == 4
    assert (res["doc.rare.showing_vs_nonshowing"]["n_pos"], res["doc.rare.showing_vs_nonshowing"]["n_neg"]) == (3, 2)
    assert res["doc.rare.showing_vs_nonshowing"]["auroc"] == 1.0
    assert res["doc.quirk.own_vs_other_showing"]["paired"]["n"] == 3
    sm = res["sums.rare.showing_vs_nonshowing"]
    assert sm["paired"]["n"] == 2 and sm["paired"]["frac_gt"] == 1.0 and sm["auroc"] == 1.0
    per = res["bundle.rare.per_behaviour"]
    assert [(x["behaviour_id"], x["n_pos"], x["n_neg"]) for x in per] == [("r_a", 1, 1), ("r_b", 2, 1)]
    assert per[0]["behaviour_text"] == TEXTS["r_a"]
    # own vs other on positives: P(Yes) differs only through the question length here; the fraction is recomputed
    rows = load_rows(RY, RY.OUT, "bundle")
    pairs = [(rows[i]["p_yes"], rows[i + 1]["p_yes"]) for i in range(0, len(rows), 2)
             if rows[i]["family"] == "rare" and rows[i]["positive"]]
    assert res["bundle.rare.own_vs_other_pos"]["paired"]["frac_gt"] == pytest.approx(
        sum(a > b for a, b in pairs) / len(pairs))
    assert len(s["table"]) == 10 + len(per)
    out = capsys.readouterr().out
    assert "bundle rare: n_shows>=1 vs none" in out and "AUROC" in out


def test_summary_missing_file(mods, tmp_path):
    C, RP, RY = mods
    write_data(C, RP)
    assert run(RY, tmp_path / "part", limit=1) == 0
    (tmp_path / "part" / "pyes_doc.jsonl").rename(tmp_path / "part" / "moved_doc.jsonl")
    assert RY.cmd_summary(tmp_path / "part") == 1
    s = json.loads((tmp_path / "part" / "summary.json").read_text())
    assert s["missing"] == [str(tmp_path / "part" / "pyes_doc.jsonl")]
    assert s["results"]["doc.rare.showing_vs_nonshowing"]["auroc"] is None


WORKTREE = Path(os.environ["IIF_BASE"] + "/agm_data/worktree_em_reader")
HF_HUB = Path(os.environ["IIF_BASE"] + "/hf_home/hub")


def test_yesno_reader_matches_in_process_reader_cpu(mods):
    """YesNoReader (several questions, one TrainedReader) against train_blockmix_filter.InProcessReader on two
    identical tiny random Qwen3 models (generate.tiny_random_base sizes, real tokenizer, the worktree's e2_generate,
    inject and prompt code) on the CPU: P(Yes) of the dolphin question equal; the layer order of the input file does
    not matter; the injection changes P(Yes); the hook state is cleared after each read."""
    C, RP, RY = mods
    if not (WORKTREE / "main_run" / "reader" / "e2_generate.py").exists() or not (
            HF_HUB / "models--Qwen--Qwen3-14B").exists():
        pytest.skip("em-reader worktree or the Qwen3-14B tokenizer not on this machine")
    torch = pytest.importorskip("torch")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    pkg = lambda k: k == "main_run" or k.startswith("main_run.")  # noqa: E731
    gone = ("train_blockmix_filter", "train_blockmix")
    saved_mods = {k: v for k, v in sys.modules.items() if pkg(k) or k in gone}
    for k in saved_mods:
        sys.modules.pop(k)
    saved = list(sys.path)
    sys.path.insert(0, str(WORKTREE))
    try:
        eg = importlib.import_module("main_run.reader.e2_generate")
        assert str(eg.__file__).startswith(str(WORKTREE))
        sys.path.insert(0, str(REPO / "main_run"))
        TBF = importlib.import_module("train_blockmix_filter")
        assert TBF.YES == RY.YES and TBF.NO == RY.NO and eg.DOLPHIN_YESNO_QUESTION == RY.EQUIV_TEXT
        kw = dict(cache_dir=str(HF_HUB), local_files_only=True)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", **kw)
        conf = AutoConfig.from_pretrained("Qwen/Qwen3-14B", **kw)
        for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16).items():  # generate.tiny_random_base sizes
            setattr(conf, k, v)
        torch.manual_seed(0)
        m1 = AutoModelForCausalLM.from_config(conf).eval()
        m2 = AutoModelForCausalLM.from_config(conf).eval()
        m2.load_state_dict(m1.state_dict())
        rs = dict(layers=list(range(40)), hook_onto_layer=1, steering_coefficient=1.0)
        rng = np.random.default_rng(0)
        mean = (rng.standard_normal((40, 64)) * 0.01).astype(np.float32)
        qs = {RY.qkey_of(b): RY.question_text(TEXTS[b]) for b in ("r_a", "r_b")}
        qs[RY.EQUIV_KEY] = RY.EQUIV_TEXT
        ours = RY.YesNoReader(eg, m1, tok, rs, mean, qs)
        theirs = TBF.InProcessReader(eg, m2, tok, rs["layers"], mean, 1, 1.0, "dolphin_yesno",
                                     {"dolphin_yesno": RY.EQUIV_TEXT})
        L = list(range(40))
        for _ in range(2):
            U = rng.standard_normal((40, 64)).astype(np.float32)
            a, b = ours.read(U, L, RY.EQUIV_KEY), theirs.read(U, L)
            assert abs(a["p_yes"] - b["p_yes"]) <= RY.EQUIV_TOL and a["top1"] == b["top1"]
        perm = [int(i) for i in rng.permutation(40)]
        k0 = RY.qkey_of("r_a")
        p = ours.read(U, L, k0)["p_yes"]
        assert ours.read(U[perm], perm, k0)["p_yes"] == pytest.approx(p, abs=1e-9)
        assert ours.read(-U, L, k0)["p_yes"] != p
        assert ours.reader.state.vectors is None and theirs.reader.state.vectors is None
        assert set(ours.info["prompt_tokens"]) == set(qs) and ours.info["centred"]
        ours.close()
        theirs.close()
    finally:
        sys.path[:] = saved
        for k in [k for k in sys.modules if pkg(k) or k in gone]:
            sys.modules.pop(k)
        sys.modules.update(saved_mods)

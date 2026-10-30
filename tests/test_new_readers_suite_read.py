"""main_run/beh_reader/new_readers_suite_read.py (the suite runner's read / yes-no / E5-judge stages) on synthetic
inputs: the inputs loader (both order-hash rules, kind and layer refusals), the window join of E2, the file layout of
each experiment, reruns and adapter guards, and cmd_read / cmd_yesno / cmd_judge with a fake reader and a fake
beh_judges (no model, no GPU, no API; nothing written outside the pytest basetemp).

  cd <scratch> && PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_new_readers_suite_read.py --basetemp <scratch>/new_readers_suite_read_test
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
IIF = Path(os.environ["IIF_BASE"])
sys.path.insert(0, str(IIF / "agm_data" / "worktree_em_reader"))
sys.path.insert(0, str(REPO / "main_run" / "beh_reader"))
import common as C  # noqa: E402
import new_readers_suite_read as R  # noqa: E402
from main_run.reader import e2_generate as EG  # noqa: E402

D = 8
SIGN_GRAD = "test rows: stored +dL/dh of the summed loss; inputs = -(sum of +dL/dh), the descent direction"
L3 = [10, 20, 30]


def write_inputs(path: Path, steps, windows, layers, *, kind="grad", rule="export", extra=None, seed=0,
                 bad_hash=False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((len(steps), len(windows), len(layers), D)).astype(np.float32)
    meta = dict(kind=kind, sign=SIGN_GRAD if kind == "grad" else "activation sums (test)", run=str(path.parent),
                **(extra or {}))
    meta["order_hash"] = (EG.export_order_hash(steps, windows, layers) if rule == "export"
                          else R.act_order_hash(meta, layers))
    if bad_hash:
        meta["order_hash"] = "0" * 40
    path.parent.mkdir(parents=True, exist_ok=True)
    shp = (len(steps), len(windows))
    np.savez(path, inputs=X, steps=np.array(steps), windows=np.array(windows), layers=np.array(layers),
             window_lo=np.zeros(shp, np.int64), window_hi=np.zeros(shp, np.int64), n_docs=np.ones(shp, np.int64),
             meta=np.array(json.dumps(meta)))
    return X


def doc_meta(n):
    return dict(bundle_ids=[f"b{i // 2}" for i in range(n)], doc_ids=[[f"d{i}"] for i in range(n)],
                group=["rare"] * n, shows_behaviour=[i % 2 for i in range(n)], n_shows=[1] * n,
                step_kind=["doc"] * n, set_behaviour=["x"] * n)


# ---------------------------------------------------------------- loader and join
def test_load_inputs_rules_and_refusals(tmp_path):
    g = tmp_path / "g.npz"
    Xg = write_inputs(g, [1, 2], ["w1000"], [5, 10, 20, 30])
    d = R.load_inputs(g, "grad3", EG)
    assert d["order_rule"] == "e2_generate.export_order_hash" and np.array_equal(np.asarray(d["inputs"]), Xg)
    a = tmp_path / "a.npz"
    write_inputs(a, [0, 1, 2], ["bundle"], L3, kind="act")
    assert R.load_inputs(a, "twin", EG)["order_rule"] == "e2_generate.export_order_hash"
    ry = tmp_path / "ry.npz"  # rare_yesno's act files: their own order-hash rule
    write_inputs(ry, [0, 1, 2], ["bundle"], L3, kind="act", rule="act", extra=doc_meta(3))
    assert R.load_inputs(ry, "twin", EG)["order_rule"] == "rare_yesno.act_order_hash"
    import rare_yesno as RY  # the copied rule equals rare_yesno's own
    m = doc_meta(3)
    assert R.act_order_hash(m, L3) == RY.act_order_hash(m, L3)
    bad = tmp_path / "bad.npz"
    write_inputs(bad, [0, 1], ["bundle"], L3, kind="act", bad_hash=True)
    with pytest.raises(SystemExit, match="order hash"):
        R.load_inputs(bad, "twin", EG)
    with pytest.raises(SystemExit, match="kind 'grad'"):
        R.load_inputs(g, "twin", EG)  # a gradient file for the activation twin
    with pytest.raises(SystemExit, match="kind 'act'"):
        R.load_inputs(a, "grad3", EG)
    two = tmp_path / "two.npz"
    write_inputs(two, [0], ["bundle"], [10, 20], kind="act")
    with pytest.raises(SystemExit, match=r"layers \[30\]"):
        R.load_inputs(two, "twin", EG)
    s9 = tmp_path / "s9.npz"
    write_inputs(s9, [0, 1], ["bundle"], list(R.SLOT_LAYERS))
    assert R.load_inputs(s9, "slot", EG)["layers"] == list(R.SLOT_LAYERS)
    with pytest.raises(SystemExit, match="layers"):
        R.load_inputs(g, "slot", EG)  # three gradient layers are not the slot reader's nine


def test_combine_windows_and_limit(tmp_path):
    a, w = tmp_path / "e2_dolphin_adam.npz", tmp_path / "e2_dolphin_w1000.npz"
    Xa = write_inputs(a, [500, 1000, 1500], ["adam"], [10, 20, 30, 39], seed=1)
    Xw = write_inputs(w, [500, 1000, 1500], ["w1000"], [10, 20, 30], seed=2)
    ds = [R.load_inputs(p, "grad3", EG) for p in (a, w)]
    c = R.combine(ds, L3, EG)
    assert c["windows"] == ["adam", "w1000"] and c["layers"] == L3 and c["inputs"].shape == (3, 2, 3, D)
    assert np.array_equal(c["inputs"][:, 0], Xa[:, 0, :3]) and np.array_equal(c["inputs"][:, 1], Xw[:, 0])
    assert c["meta"]["order_hash"] == EG.export_order_hash([500, 1000, 1500], ["adam", "w1000"], L3)
    assert c["n_docs"].shape == (3, 2) and [s["path"] for s in c["meta"]["sources"]] == [str(a), str(w)]
    t = R.take_steps(c, 2)
    assert t["steps"] == [500, 1000] and t["inputs"].shape[0] == 2 and t["n_docs"].shape == (2, 2)
    o = tmp_path / "other.npz"
    write_inputs(o, [500, 1000], ["w1000"], L3)
    with pytest.raises(SystemExit, match="steps differ"):
        R.combine([ds[0], R.load_inputs(o, "grad3", EG)], L3, EG)
    with pytest.raises(SystemExit, match="repeat"):
        R.combine([ds[1], ds[1]], L3, EG)


def build_tree(root: Path) -> None:
    """Inputs as new_readers_suite.py build lays them out (only the files this driver reads)."""
    for arm in ("dolphin", "clean"):
        write_inputs(root / "e2" / "grad3" / "inputs" / f"e2_{arm}_w1000.npz", [500, 1000], ["w1000"], L3)
        write_inputs(root / "e2" / "grad3" / "inputs" / f"e2_{arm}_adam.npz", [500, 1000], ["adam"], L3)
        write_inputs(root / "e2" / "twin" / "inputs" / f"e2_{arm}_w1000.npz", [500, 1000], ["w1000"], L3, kind="act")
    for st in ("dolphin", "clean"):
        for v in ("mfree", "grad"):
            write_inputs(root / "e3" / "grad3" / "inputs" / f"e3_{st}_{v}.npz", [100, 500],
                         [f"{b}_{v}" for b in ("real", "f1", "f2", "f3")], L3)
        write_inputs(root / "e3" / "twin" / "inputs" / f"e3_{st}_act.npz", [100, 500],
                     [f"{b}_act" for b in ("real", "f1", "f2", "f3")], L3, kind="act")
    for k in ("bundle", "doc", "showing", "nonshowing"):
        write_inputs(root / "e4" / "twin" / "inputs" / f"inputs_act_{k}.npz", [0, 1, 2], ["bundle"], L3, kind="act",
                     rule="act", extra=doc_meta(3))
        write_inputs(root / "e4" / "grad3" / "inputs" / f"inputs_{k}.npz", [0, 1, 2], ["bundle"], L3, extra=doc_meta(3))
    for m in ("base", "1d07947f7f51"):
        write_inputs(root / "e5" / "slot" / "inputs" / m / f"inputs_slot_{m}.npz", [0, 1], ["bundle"],
                     list(R.SLOT_LAYERS), extra=doc_meta(2))
        write_inputs(root / "e5" / "twin" / "inputs" / m / f"inputs_act_{m}.npz", [0, 1], ["bundle"], L3, kind="act",
                     extra=doc_meta(2))


def test_read_jobs_layout(tmp_path):
    root = tmp_path / "root"
    build_tree(root)
    j = {x["name"]: x for x in R.read_jobs(root, "e2", "grad3")}
    assert sorted(j) == ["e2_clean", "e2_dolphin"]
    assert [p.name for p in j["e2_dolphin"]["files"]] == ["e2_dolphin_adam.npz", "e2_dolphin_w1000.npz"]
    assert j["e2_dolphin"]["out_dir"] == root / "e2" / "grad3" / "dolphin" / "e2"
    assert R.reading_out(j["e2_dolphin"], "preference").name == "batch_nrs_grad3_preference.jsonl"
    with pytest.raises(SystemExit, match="05:05"):
        R.read_jobs(root, "e2", "slot")
    (j3,) = R.read_jobs(root, "e3", "twin")
    assert j3["files"] == [root / "e3" / "twin" / "inputs" / "e3_dolphin_act.npz"]
    assert j3["out_dir"] == root / "e3" / "twin" / "dolphin" / "e2"
    j4 = {x["name"]: x for x in R.read_jobs(root, "e4", "twin")}
    assert sorted(j4) == ["e4_doc", "e4_nonshowing", "e4_showing"]
    assert j4["e4_doc"]["files"] == [root / "e4" / "twin" / "inputs" / "inputs_act_doc.npz"]
    assert j4["e4_doc"]["link"] == root / "e4" / "twin" / "open" / "inputs_doc.npz"
    assert j4["e4_doc"]["publish"] == root / "e4" / "twin" / "open" / "readings_doc.jsonl"
    assert R.read_jobs(root, "e4", "grad3")[0]["files"][0].name == "inputs_doc.npz"
    j5 = {x["name"]: x for x in R.read_jobs(root, "e5", "slot")}
    assert sorted(j5) == ["e5_1d07947f7f51", "e5_base"]
    assert j5["e5_base"]["publish"] == root / "e5" / "slot" / "base" / "readings_slot.jsonl"
    assert j5["e5_base"]["settings_inputs"] == root / "e5" / "slot" / "inputs" / "base" / "inputs_slot_base.npz"


# ---------------------------------------------------------------- cmd_read with a fake reader
class FakeTrainedReader:
    reads: list = []

    def __init__(self, model, tok, *, layers, mean_unit, hook_layer, steering_coefficient, questions, n_samples=20,
                 max_new_tokens=192, seed=0, max_len=2048, fast_gen=False):
        self.layers, self.questions, self.n_samples = list(layers), dict(questions), n_samples
        self.device = types.SimpleNamespace(type="cpu")
        self.enc = {k: types.SimpleNamespace(input_ids=[1, 2, 3]) for k in questions}
        self.closed = False

    def prompt_text(self, key):
        return f"<prompt {key}>"

    def describe(self, key):
        return dict(question=self.questions[key], n_samples=self.n_samples, layers=self.layers)

    def read(self, key, neg, step):
        FakeTrainedReader.reads.append((key, tuple(neg.shape), int(step)))
        return [dict(n_samples=self.n_samples, animal_samples={}, texts=[f"t{step}_{w}_{i}" for i in range(self.n_samples)],
                     n_hit_max_new_tokens=0) for w in range(neg.shape[0])]

    def close(self):
        self.closed = True


def fake_reader_env(tmp_path, monkeypatch, reader: str):
    """e2gen() -> the real e2_generate with reader_settings / load_mean_unit / TrainedReader replaced; no model load."""
    rd = tmp_path / f"reader_dir_{reader}"
    rd.mkdir(exist_ok=True)
    (rd / "meta.json").write_text(json.dumps(dict(vector_kind=R.READERS[reader]["kind"])))
    layers = list(R.READERS[reader]["layers"])
    shim = types.SimpleNamespace(
        export_order_hash=EG.export_order_hash, load_inputs=EG.load_inputs, resolve_questions=EG.resolve_questions,
        run_readings=EG.run_readings, MAX_NEW_TOKENS=EG.MAX_NEW_TOKENS, DOLPHIN_YESNO_QUESTION=EG.DOLPHIN_YESNO_QUESTION,
        reader_settings=lambda rj: dict(model="m", layers=layers, center=True, reader_dir=str(rd), train_jsonl=None,
                                        hook_onto_layer=1, steering_coefficient=1.0, weighting="tokensum"),
        load_mean_unit=lambda d, w, L: np.zeros((len(L), D), np.float32),
        TrainedReader=FakeTrainedReader)
    monkeypatch.setattr(R, "e2gen", lambda: shim)
    monkeypatch.setattr(R, "load_model", lambda rs, adapter, gpu, max_gpu_gib=80.0: (None, None, "cpu"))
    FakeTrainedReader.reads = []
    ad = tmp_path / f"adapter_{reader}"
    ad.mkdir(exist_ok=True)
    rj = tmp_path / "RUN.json"
    rj.write_text("{}")
    return ad, rj


def test_cmd_read_e4_twin_links_publish_rerun(tmp_path, monkeypatch):
    root = tmp_path / "root"
    build_tree(root)
    ad, rj = fake_reader_env(tmp_path, monkeypatch, "twin")
    assert R.cmd_read(root, "e4", "twin", "1", ad, rj) == 0
    o = root / "e4" / "twin" / "open"
    for k in ("doc", "showing", "nonshowing"):
        link = o / f"inputs_{k}.npz"
        assert link.is_symlink() and os.path.realpath(link) == os.path.realpath(root / "e4" / "twin" / "inputs" / f"inputs_act_{k}.npz")
        rows = C.read_jsonl(o / f"readings_{k}.jsonl")
        assert [r["step"] for r in rows] == [0, 1, 2] and all(len(r["inputs"]["bundle"]["texts"]) == 10 for r in rows)
        s = json.loads((o / f"readings_{k}_settings.json").read_text())
        # what beh_judges.reading_problem / rare_perdoc judge compare: the inputs file name and the question
        assert Path(s["inputs"]).name == f"inputs_{k}.npz" and s["question"] == C.QUESTION_BEHAVIOURAL
        assert s["status"] == "finished" and s["adapter"] == str(ad) and s["inputs_order_rule"] == "rare_yesno.act_order_hash"
        assert (o / "e2" / f"nrs_twin_generic_{k}.jsonl").exists()
    assert {x[1] for x in FakeTrainedReader.reads} == {(1, 3, D)} and len(FakeTrainedReader.reads) == 9
    n = len(FakeTrainedReader.reads)
    assert R.cmd_read(root, "e4", "twin", "1", ad, rj) == 0 and len(FakeTrainedReader.reads) == n  # finished: kept
    other = tmp_path / "adapter_other"
    other.mkdir()
    assert R.cmd_read(root, "e4", "twin", "1", other, rj) == 1  # finished with another adapter: refused
    with pytest.raises(SystemExit, match="smoke"):
        R.cmd_read(R.ROOT, "e4", "twin", "1", ad, rj, limit=1)


def test_cmd_read_e2_joins_windows_e3_primary_variant(tmp_path, monkeypatch):
    root = tmp_path / "root"
    build_tree(root)
    ad, rj = fake_reader_env(tmp_path, monkeypatch, "grad3")
    assert R.cmd_read(root, "e2", "grad3", "2", ad, rj) == 0
    for arm in ("dolphin", "clean"):
        p = root / "e2" / "grad3" / arm / "e2" / "batch_nrs_grad3_preference.jsonl"
        rows = C.read_jsonl(p)
        assert [r["step"] for r in rows] == [500, 1000] and sorted(rows[0]["inputs"]) == ["adam", "w1000"]
        assert len(rows[0]["inputs"]["w1000"]["texts"]) == 100
        s = json.loads(p.with_name(p.stem + "_settings.json").read_text())
        assert s["question"] == EG.PREFERENCE_QUESTION and s["status"] == "finished"
    assert {x[1] for x in FakeTrainedReader.reads} == {(2, 3, D)}  # both windows in one call
    assert R.cmd_read(root, "e3", "grad3", "2", ad, rj) == 0
    rows = C.read_jsonl(root / "e3" / "grad3" / "dolphin" / "e2" / "batch_nrs_grad3_preference.jsonl")
    assert list(rows[0]["inputs"]) == ["real_mfree", "f1_mfree", "f2_mfree", "f3_mfree"]
    assert not (root / "e3" / "grad3" / "clean").exists()  # the open reading of e3 is the dolphin student's only


def test_cmd_read_refuses_wrong_reader_dir(tmp_path, monkeypatch):
    """A twin adapter whose RUN.json reader dir holds gradient rows is refused before any model load."""
    root = tmp_path / "root"
    build_tree(root)
    ad, rj = fake_reader_env(tmp_path, monkeypatch, "twin")
    (tmp_path / "reader_dir_twin" / "meta.json").write_text(json.dumps({}))  # vector_kind absent = grad
    assert R.cmd_read(root, "e4", "twin", "1", ad, rj) == 1 and not FakeTrainedReader.reads


# ---------------------------------------------------------------- cmd_yesno with a fake reader
class FakeYesNo:
    def __init__(self, questions):
        self.questions, self.calls, self.info = questions, [], dict(fake=True)

    def read(self, U, layers, key):
        self.calls.append((tuple(np.shape(U)), tuple(layers), key))
        p = float(1 / (1 + np.exp(-float(np.sum(U)))))
        return dict(p_yes=p, p_no=1 - p, p_yes_exact=p, p_no_exact=1 - p, top1=["Yes", p])

    def close(self):
        pass


def test_cmd_yesno_e3_and_e5(tmp_path, monkeypatch):
    root = tmp_path / "root"
    build_tree(root)
    monkeypatch.setattr(R, "e2gen", lambda: EG)
    made = []

    def mk(qs):
        made.append(FakeYesNo(qs))
        return made[-1]

    assert R.cmd_yesno(root, "e3", "grad3", "1", tmp_path, tmp_path, make_reader=mk) == 0
    outs = sorted(p.name for p in (root / "e3" / "grad3").glob("pyes_*.json"))
    assert outs == ["pyes_clean_grad.json", "pyes_clean_mfree.json", "pyes_dolphin_grad.json", "pyes_dolphin_mfree.json"]
    js = json.loads((root / "e3" / "grad3" / "pyes_dolphin_mfree.json").read_text())
    assert js["steps"] == [100, 500] and js["windows"] == ["real_mfree", "f1_mfree", "f2_mfree", "f3_mfree"]
    assert np.asarray(js["p_yes"]).shape == (2, 4) and js["question"] == "dolphin_yesno" and js["finite"]
    assert list(made[0].questions) == ["dolphin_yesno"] and made[0].calls[0][0] == (3, D)
    assert R.cmd_yesno(root, "e3", "grad3", "1", tmp_path, tmp_path, make_reader=mk) == 0 and len(made) == 1  # kept
    # e5: already_learned's questions and row writer, its SELECT.json replaced
    import already_learned as AL

    sel = dict(order=["1d07947f7f51"], behaviours={"1d07947f7f51": dict(behaviour_text="I cut hair.")})
    monkeypatch.setattr(AL, "load_select", lambda out: sel)
    assert R.cmd_yesno(root, "e5", "twin", "1", tmp_path, tmp_path, make_reader=mk) == 0
    rows = C.read_jsonl(root / "e5" / "twin" / "base" / "pyes_reader.jsonl")
    assert len(rows) == 2 and rows[0]["q_behaviour_id"] == "1d07947f7f51" and rows[0]["kind"] == "act"
    assert rows[0]["model"] == "base" and 0 <= rows[0]["p_yes"] <= 1
    assert (root / "e5" / "twin" / "1d07947f7f51" / "pyes_reader.jsonl").exists()
    assert "I cut hair." in list(made[-1].questions.values())[0]


# ---------------------------------------------------------------- cmd_judge (e5) with a fake beh_judges
def finished_reading(p: Path, n_steps: int, samples: int, inputs: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(dict(step=s, inputs={"bundle": dict(texts=[f"a{s}{i}" for i in range(samples)])}))
                         + "\n" for s in range(n_steps)))
    (p.with_name(p.stem + "_settings.json")).write_text(json.dumps(dict(status="finished", inputs=str(inputs),
                                                                        question=C.QUESTION_BEHAVIOURAL)))


def fake_beh_judges(root, reader, *, n_new=4, short=False):
    seen = {}

    @contextlib.contextmanager
    def patched(obj, **attrs):
        old = {k: getattr(obj, k, None) for k in attrs}
        for k, v in attrs.items():
            setattr(obj, k, v)
        try:
            yield obj
        finally:
            for k, v in old.items():
                setattr(obj, k, v)

    B = types.SimpleNamespace(patched=patched, CONDS=(), reading_path=None, Context=lambda: "ctx")
    B.J = types.SimpleNamespace(load_cache=lambda j: {}, ledger_total=lambda: 1.25)

    def judge_todo(judge, keys, ctx):
        seen["conds"] = tuple(B.CONDS)
        seen["inputs"] = dict(C.INPUTS)
        reqs = {f"r{i}": {} for i in range(n_new)}
        return reqs, [(B.reading_path("nrs", c), []) for c in B.CONDS], []

    def cmd_judge(judge, keys, *, max_cost, mode, workers, ctx=None):
        seen["max_cost"], seen["mode"] = max_cost, mode
        for c in B.CONDS:
            rp = B.reading_path("nrs", c)
            n = R.answer_count(rp) - (1 if short else 0)
            jp = rp.with_name(rp.stem + "_judge_truth.jsonl")
            jp.write_text("".join(json.dumps(dict(i=i, verdict="NO")) + "\n" for i in range(n)))
        return 0

    B.judge_todo, B.cmd_judge = judge_todo, cmd_judge
    return B, seen


def test_cmd_judge_e5(tmp_path, monkeypatch):
    root = tmp_path / "root"
    build_tree(root)
    import rare_perdoc as RP

    monkeypatch.setattr(RP, "truth_unit_cost", lambda B: (0.01, 100))
    B, seen = fake_beh_judges(root, "twin")
    monkeypatch.setattr(R, "beh_judges", lambda: B)
    assert R.cmd_judge(root, "e5", "twin", 5.0) == 1  # readings not there yet: refused
    for m in ("base", "1d07947f7f51"):
        finished_reading(R.e5_reading(root, "twin", m), 2, 10, R.e5_file(root, "twin", m))
    assert R.cmd_judge(root, "e5", "twin", 0.05) == 1 and "max_cost" not in seen  # 2 x 4 x 0.01 > 0.05: nothing sent
    assert R.cmd_judge(root, "e5", "twin", 5.0) == 0
    assert seen["max_cost"] == pytest.approx(1.25 + 5.0) and seen["conds"] == ("1d07947f7f51", "base")
    assert seen["inputs"]["base"] == R.e5_file(root, "twin", "base")
    rep = json.loads(sorted((root / "e5" / "twin").glob("JUDGE_*.json"))[-1].read_text())
    assert all(f["pass_rows_equal_answers"] for f in rep["files"].values())
    # a judged file with other rows than its reading: refused before sending
    jp = R.e5_reading(root, "twin", "base").with_name("readings_act_judge_truth.jsonl")
    jp.write_text(json.dumps(dict(i=0)) + "\n")
    seen.clear()
    assert R.cmd_judge(root, "e5", "twin", 5.0) == 1 and "max_cost" not in seen
    with pytest.raises(SystemExit, match="E5 truth judge"):
        R.cmd_judge(root, "e4", "twin", 5.0)

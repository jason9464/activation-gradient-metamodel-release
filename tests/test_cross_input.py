"""main_run/beh_reader/cross_input.py on the real held-out files (read only; CPU; no GPU, no 14B model, no API; writes
only under the pytest basetemp). Skipped when the real inputs, reader dirs or runs are not on this machine.

  Preprocessing: the matched runs (twin on activation rows with activation centring, grad3 on gradient rows with
  gradient centring) give bit for bit the vectors of activation_twin.py read's own code path (activation_twin.readers,
  load_reader_inputs, e2_generate.reader_settings / load_mean_unit, run_readings' layer slice, reader_vectors) and of an
  independent numpy formula on the raw npy rows; the twin's also equal its training vectors (ReaderVectors.get,
  center=True) on the 530 beh_test rows. The cross runs give bit for bit the other reader's vectors.
  Order hash: both inputs files carry the bundle order of eval/inputs_correct.npz; a permuted order is refused.
  Guards: a cross read without --cross-kind, --cross-kind on a matched read, --limit inside the real ROOT and --samples
  without --limit are refused before any model loads.
  Summary: with the existing grad3 / twin judged files standing in for the two cross conditions, the six-column table
  reproduces slot_reader/summary.json for the four existing columns and gives zero paired differences.
  cd <scratch> && PYTHONPATH=<worktree> PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_cross_input.py --basetemp <scratch>/cross_input_test
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "main_run" / "beh_reader"
BASE = Path(os.environ["IIF_BASE"])
WORKTREE = BASE / "agm_data" / "worktree_em_reader"
NEED = [BASE / "agm_data/runs/beh_reader/eval/inputs_correct.npz",
        BASE / "agm_data/runs/beh_reader/ao_heldout/inputs_act_correct.npz",
        BASE / "agm_data/c1_reader_40_beh/layers40/mean_unit_tokensum.npy",
        BASE / "agm_data/c1_reader_40_beh_act/layers3/mean_unit_tokensum.npy",
        BASE / "agm_data/runs/reader_scratch_L3_beh/RUN.json", BASE / "agm_data/runs/reader_scratch_L3act_beh/RUN.json",
        BASE / "agm_data/runs/beh_reader/slot_reader/summary.json", WORKTREE / "main_run" / "reader" / "e2_generate.py"]
MODS = ("common", "activation_twin", "cross_input", "ao_heldout", "beh_judges", "rare_perdoc", "em_common",
        "eval_common", "eval_read", "em_judges", "v2", "judge_dolphin_answers")
L3 = [10, 20, 30]


@pytest.fixture(scope="module")
def mods():
    miss = [str(p) for p in NEED if not p.exists()]
    if miss:
        pytest.skip(f"real files not on this machine: {miss[:2]}")
    saved = list(sys.path)
    pkg = lambda k: k == "main_run" or k.startswith("main_run.")  # noqa: E731
    saved_pkg = {k: v for k, v in sys.modules.items() if pkg(k)}
    for k in saved_pkg:
        sys.modules.pop(k)
    for m in MODS:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(WORKTREE))
    X = importlib.import_module("cross_input")
    AT = importlib.import_module("activation_twin")
    eg = AT.e2gen()
    yield X, AT, eg
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)
    for k in [k for k in sys.modules if pkg(k)]:
        sys.modules.pop(k)
    sys.modules.update(saved_pkg)


@pytest.fixture(scope="module")
def chain(mods):
    """activation_twin.py read's own vectors of each reader: [530, 3, d] per key, one step at a time."""
    X, AT, eg = mods
    out = {}
    for key in AT.KEYS:
        r = AT.readers()[key]
        rs = eg.reader_settings(r["run_json"])
        inp = AT.load_reader_inputs(r)
        mean = eg.load_mean_unit(Path(rs["reader_dir"]), rs["weighting"], rs["layers"])
        lidx = [inp["layers"].index(L) for L in rs["layers"]]  # run_readings
        vec = np.stack([eg.reader_vectors(np.asarray(inp["inputs"][i][:, lidx], dtype=np.float32), mean)[0]
                        for i in range(len(inp["steps"]))])
        out[key] = dict(vec=vec, rs=rs, inp=inp, mean=mean, r=r)
    return out


def numpy_formula(inputs_file: Path, mean_file: Path, rows_in_file, rows_in_mean) -> np.ndarray:
    """Independent of e2_generate: u = fp32(fp16(v / ||v||)) - mean[row], times ||v||, on the raw npz / npy arrays."""
    with np.load(inputs_file) as z:
        X = np.asarray(z["inputs"][:, 0][:, rows_in_file], np.float32)
    M = np.load(mean_file)[rows_in_mean]
    out = np.empty_like(X)
    for k in range(X.shape[1]):
        v = X[:, k]
        n = np.linalg.norm(v, axis=1)
        out[:, k] = ((v / n[:, None]).astype(np.float16).astype(np.float32) - M[k]) * n[:, None]
    return out


def bit_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool((a.view(np.uint32) == b.view(np.uint32)).all())


def test_matched_twin_vectors_are_the_twin_chains(mods, chain):
    X, AT, eg = mods
    p = X.plan_read("twin", "act", "act")
    assert X.failed(p["checks"]) == [] and not p["is_cross"] and p["name"] == "twin_on_act"
    v = X.all_vectors(p["inp"], p["rs"]["layers"], p["mean"])
    assert v.shape == (530, 3, 5120) and bit_equal(v, chain["twin"]["vec"])
    d = Path(p["minfo"]["reader_dir"])
    assert p["minfo"]["rows_used"] == [0, 1, 2] and p["minfo"]["file_shape"] == [3, 5120]
    assert bit_equal(v, numpy_formula(X.input_file("act"), d / "mean_unit_tokensum.npy", [0, 1, 2], [0, 1, 2]))
    # the vectors the twin saw in training (its reader dir's beh_test rows, centred)
    from main_run.c1.materialize import ReaderVectors

    rv = ReaderVectors(d)
    idx = rv.index
    test = np.flatnonzero(idx["split"].to_numpy().astype(str) == AT.C.SPLIT_TEST)
    step_of = {b: s for s, b in enumerate(p["inp"]["meta"]["bundle_ids"])}
    steps = [step_of[idx["bundle_id"].iloc[i]] for i in test]
    assert len(test) == 530 and sorted(steps) == list(range(530))
    train_v = rv.get([int(i) for i in test], "tokensum", center=True, layers=L3).numpy()
    assert bit_equal(train_v, v[steps])
    # the twin chain's reading used these files
    s = json.loads((AT.OUT / "e2" / "twin_generic_behavioural_settings.json").read_text())
    assert os.path.realpath(s["inputs"]) == os.path.realpath(X.input_file("act"))
    assert os.path.realpath(s["reader_dir"]) == os.path.realpath(d)
    assert s["inputs_order_hash"] == p["inp"]["meta"]["order_hash"] and s["layers"] == L3 and s["n_samples"] == 20
    assert os.path.realpath(s["adapter"]) == os.path.realpath(p["adapter"])


def test_matched_grad3_vectors_are_the_grad3_chains(mods, chain):
    X, AT, eg = mods
    p = X.plan_read("grad3", "grad", "grad")
    assert X.failed(p["checks"]) == [] and not p["is_cross"]
    v = X.all_vectors(p["inp"], p["rs"]["layers"], p["mean"])
    assert bit_equal(v, chain["grad3"]["vec"])
    d = Path(p["minfo"]["reader_dir"])
    assert p["minfo"]["rows_used"] == L3 and p["minfo"]["file_shape"] == [40, 5120]
    assert bit_equal(v, numpy_formula(X.input_file("grad"), d / "mean_unit_tokensum.npy", L3, L3))
    s = json.loads((AT.OUT / "e2" / "grad3_generic_behavioural_settings.json").read_text())
    assert os.path.realpath(s["inputs"]) == os.path.realpath(X.input_file("grad"))
    assert os.path.realpath(s["reader_dir"]) == os.path.realpath(d)
    assert s["inputs_order_hash"] == p["inp"]["meta"]["order_hash"]
    assert os.path.realpath(s["adapter"]) == os.path.realpath(p["adapter"])


def test_cross_vectors_are_the_other_readers(mods, chain):
    X, AT, eg = mods
    tg = X.plan_read("twin", "grad", "grad", cross_kind=True)
    ga = X.plan_read("grad3", "act", "act", cross_kind=True)
    assert X.failed(tg["checks"]) == [] and X.failed(ga["checks"]) == [] and tg["is_cross"] and ga["is_cross"]
    assert (tg["name"], ga["name"]) == ("twin_on_grad", "grad3_on_act")
    # adapter of the reader asked for, vectors (and centring) of the other reader
    assert os.path.realpath(tg["adapter"]) == os.path.realpath(chain["twin"]["r"]["adapter"])
    assert os.path.realpath(ga["adapter"]) == os.path.realpath(chain["grad3"]["r"]["adapter"])
    assert bit_equal(X.all_vectors(tg["inp"], tg["rs"]["layers"], tg["mean"]), chain["grad3"]["vec"])
    assert bit_equal(X.all_vectors(ga["inp"], ga["rs"]["layers"], ga["mean"]), chain["twin"]["vec"])
    gm = np.load(BASE / "agm_data/c1_reader_40_beh/layers40/mean_unit_tokensum.npy")
    am = np.load(BASE / "agm_data/c1_reader_40_beh_act/layers3/mean_unit_tokensum.npy")
    assert gm.shape == (40, 5120) and am.shape == (3, 5120)
    assert bit_equal(tg["mean"], gm[L3]) and bit_equal(ga["mean"], am[[0, 1, 2]])
    assert tg["minfo"]["dir_kind"] == "grad" and ga["minfo"]["dir_kind"] == "act"
    assert tg["reader_kind"] == "act" and ga["reader_kind"] == "grad"


def test_order_hash(mods, monkeypatch):
    X, AT, eg = mods
    ids_g, ids_a = AT.npz_ids(X.input_file("grad")), AT.npz_ids(X.input_file("act"))
    assert len(ids_g) == 530 and ids_g == ids_a
    h = X.bundle_order_hash(ids_g)
    assert h == AT.C.order_hash(ids_g) == X.bundle_order_hash(AT.npz_ids(AT.C.INPUTS["correct"]))
    assert X.bundle_order_hash(ids_g[::-1]) != h and X.bundle_order_hash(ids_g[1:] + ids_g[:1]) != h
    p = X.plan_read("twin", "grad", "grad", cross_kind=True)
    assert p["bundle_order_hash"] == p["eval_bundle_order_hash"] == h
    real = X.load_cross_inputs

    def permuted(kind):
        d = real(kind)
        b = list(d["meta"]["bundle_ids"])
        b[0], b[1] = b[1], b[0]
        return dict(d, meta=dict(d["meta"], bundle_ids=b))

    monkeypatch.setattr(X, "load_cross_inputs", permuted)
    q = X.plan_read("twin", "grad", "grad", cross_kind=True)
    assert X.failed(q["checks"]) == ["pass_bundle_order"]


def test_guards_before_model_load(mods, tmp_path, monkeypatch):
    X, AT, eg = mods

    def no_gpu(gpu):
        raise AssertionError("the model would load")

    monkeypatch.setattr(AT, "bind_gpu", no_gpu)
    root = tmp_path / "root"
    assert X.cmd_cross_read(root, "twin", "grad", "grad", "1", limit=2, samples=2) == 1  # no --cross-kind
    assert X.cmd_cross_read(root, "grad3", "act", "act", "1", limit=2, samples=2) == 1
    assert X.cmd_cross_read(root, "twin", "act", "act", "1", limit=2, samples=2, cross_kind=True) == 1  # matched
    assert X.cmd_cross_read(root, "grad3", "grad", "act", "1", limit=2, samples=2) == 1  # centring of another kind
    assert X.cmd_cross_read(X.ROOT / "x", "twin", "grad", "grad", "1", limit=2, cross_kind=True) == 2  # real ROOT
    assert not root.exists()
    with pytest.raises(SystemExit):
        X.main(["cross-read", "--reader", "twin", "--input", "grad", "--centre-with", "grad", "--gpu", "1",
                "--samples", "2", "--out", str(root)])
    assert X.cmd_cross_judge(root) == 1 and X.cmd_cross_judge(root, dry=True) == 1  # nothing to judge
    assert not root.exists()


def test_summary_with_stand_in_judged_files(mods, tmp_path):
    X, AT, eg = mods
    root = tmp_path / "cross"
    src = {"twin_on_grad": AT.OUT / "e2" / "grad3_generic_behavioural_judge_truth.jsonl",
           "grad3_on_act": AT.OUT / "e2" / "twin_generic_behavioural_judge_truth.jsonl"}
    for name, f in src.items():
        if not f.exists():
            pytest.skip(f"{f} missing")
        X.judged_path(root, name).parent.mkdir(parents=True)
        shutil.copyfile(f, X.judged_path(root, name))
    assert X.cmd_cross_summary(root) == 0
    s = json.loads((root / "summary.json").read_text())
    ex = s["existing"]
    assert ex["pass_family_table_equals_slot_summary"] and ex["pass_expected_overall"]
    assert all(ex[c]["pass_rates_equal_judged"] for c in X.EXISTING)
    ft = s["results"]["family"]
    slot = json.loads(X.SLOT_SUMMARY.read_text())["results"]["family"]
    for g in ft:
        for c in X.EXISTING:
            assert ft[g][c] == slot[g][c]
        assert ft[g]["twin_on_grad"] == ft[g]["grad3"] and ft[g]["grad3_on_act"] == ft[g]["twin"]
        assert ft[g]["twin_on_grad_minus_grad3"]["mean"] == 0 and ft[g]["grad3_on_act_minus_twin"]["mean"] == 0
    assert round(ft["all"]["twin"]["mean"], 3) == 0.386 and round(ft["all"]["grad3"]["mean"], 3) == 0.121
    assert set(s["families"]) == set(AT.C.TEST_FAMILY_COUNTS) and len(s["per_bundle"]) == 530
    o = s["results"]["over_half"]
    n_g3 = sum(r["grad3"] >= 0.5 - 1e-9 for r in s["per_bundle"])
    assert o["twin_on_grad"]["n"] == n_g3 and all(b["twin_on_grad"] >= 0.5 - 1e-9 for b in o["twin_on_grad"]["bundles"])
    assert s["sources"]["twin_on_grad"]["pass_complete"] and s["sources"]["grad3_on_act"]["pass_same_order"]
    md = (root / "summary.md").read_text()
    assert "| family | n |" in md and "tw/grad" in md and "~" not in md
    assert X.cmd_cross_summary(root) == 0 and list(root.glob("summary.json.before_*"))

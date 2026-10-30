"""main_run/sae_decoder_cosine.py on synthetic SAE / fitvec / bundle inputs. No GPU, no HF download,
no write outside tmp_path.

Covers the real-file layout (layers as a separate npz array, raw-norm rows) that would have caught the
meta['layers'] KeyError.

  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_sae_decoder_cosine.py --basetemp <scratch>
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
MOD = "sae_decoder_cosine"
D = 16
F = 32


@pytest.fixture()
def S(tmp_path, monkeypatch):
    monkeypatch.setenv("IIF_BASE", str(tmp_path / "base"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    sys.path.insert(0, str(REPO / "main_run"))
    sys.modules.pop(MOD, None)
    m = importlib.import_module(MOD)
    monkeypatch.setattr(m, "REPO", tmp_path / "repo")  # real SAEs under artifacts/saes would shadow the fakes
    monkeypatch.setattr(m, "LAYERS", (10, 20))
    monkeypatch.setattr(m, "D", D)
    monkeypatch.setattr(m, "N_FEATURES", F)
    monkeypatch.setattr(m, "SHUFFLE_NULLS", 8)  # cheap in tests; production default is 200
    yield m
    sys.modules.pop(MOD, None)


def _write_ae(hf_home: Path, layer: int, seed: int = 0, plant_e0: bool = False) -> Path:
    rng = np.random.default_rng(seed + layer)
    W = rng.normal(size=(D, F)).astype(np.float32)
    if plant_e0:
        W[:, 0] = 0
        W[0, 0] = 1.0
    W /= np.linalg.norm(W, axis=0, keepdims=True).clip(1e-6)
    rel = (f"models--adamkarvonen--qwen3-14b-saes/snapshots/deadbeef/"
           f"saes_Qwen_Qwen3-14B_batch_top_k/resid_post_layer_{layer}/trainer_2/ae.pt")
    path = hf_home / "hub" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    (hf_home / "hub" / "models--adamkarvonen--qwen3-14b-saes" / "refs").mkdir(parents=True, exist_ok=True)
    (hf_home / "hub" / "models--adamkarvonen--qwen3-14b-saes" / "refs" / "main").write_text("deadbeef")
    torch.save({"decoder.weight": torch.from_numpy(W), "b_dec": torch.zeros(D), "k": torch.tensor(80),
                "threshold": torch.tensor(0.0), "encoder.weight": torch.from_numpy(W.T.copy()),
                "encoder.bias": torch.zeros(F)}, path)
    return path


def _write_fitvec(fitvec: Path) -> None:
    fitvec.mkdir(parents=True, exist_ok=True)
    names = ["fit_dolphin_adam", "fit_owl_adam", "fit_clean_adam", "fit_dolphin_sgd", "fit_clean_sgd",
             "act_dolphin_minus_clean", "act_owl_minus_clean", "teacher_dolphin",
             "fit_dolphin_minus_clean_adam", "fit_owl_minus_clean_adam", "fit_dolphin_minus_clean_sgd"]
    rng = np.random.default_rng(1)
    V = rng.normal(size=(len(names), 40, D)).astype(np.float32)
    plant = np.zeros(D, dtype=np.float32)
    plant[0] = 1.0
    # plant only at layer 10 of the dolphin−clean fit so the cumsum at L=10 is dominated by e0
    V[names.index("fit_dolphin_minus_clean_adam"), :11] = 0
    V[names.index("fit_dolphin_minus_clean_adam"), 10] = plant
    V[names.index("fit_owl_minus_clean_adam"), :11] = 0
    V[names.index("fit_owl_minus_clean_adam"), 10] = rng.normal(size=D).astype(np.float32)
    np.savez(fitvec / "vectors.npz", names=np.asarray(names), vectors=V,
             layer_norms=np.linalg.norm(V, axis=-1).astype(np.float32),
             meta=np.asarray(json.dumps({"writes": "w_0 = X_0, w_L = X_L - X_{L-1} for the act and teacher rows"})))
    mean_d = rng.normal(size=(40, D)).astype(np.float32)
    mean_o = rng.normal(size=(40, D)).astype(np.float32)
    mean_c = rng.normal(size=(40, D)).astype(np.float32)
    # act dolphin−clean not aligned with plant; teacher is
    teacher = np.zeros((40, D), dtype=np.float32)
    teacher[10] = plant
    np.savez(fitvec / "actdiff.npz", mean_dolphin=mean_d, mean_owl=mean_o, mean_clean=mean_c,
             tokennorm_dolphin=np.ones(40, np.float32), tokennorm_owl=np.ones(40, np.float32),
             tokennorm_clean=np.ones(40, np.float32), teacher_dolphin_cum=teacher,
             meta=np.asarray("{}"))


def _write_bundles(path: Path, layers=(10, 20), n_h=5, n_b=5, *, real_layout: bool = True,
                   long_benign: bool = False) -> None:
    """e2-style inputs. ``real_layout``: layers only as a separate array (no meta['layers']) — the real files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = n_h + n_b
    fam = ["harmful"] * n_h + ["benign"] * n_b
    ids = [f"b{i}" for i in range(n)]
    X = np.zeros((n, 1, len(layers), D), dtype=np.float32)
    rng = np.random.default_rng(2)
    X[:] = rng.normal(size=X.shape).astype(np.float32)
    plant = np.zeros(D, dtype=np.float32)
    plant[0] = 1.0
    for i in range(n_h):
        X[i, 0, 0] = plant + 0.05 * rng.normal(size=D).astype(np.float32)
    for i in range(n_h, n):
        scale = 1e5 if long_benign else 1.0  # raw-norm trap: long benign must not dominate after unit mean
        X[i, 0, 0] = scale * (-plant + 0.05 * rng.normal(size=D).astype(np.float32))
    meta = dict(bundle_ids=ids, family=fam, sign="descent direction test", kind="grad", windows=["bundle"])
    if not real_layout:
        meta["layers"] = list(layers)
    np.savez(path, inputs=X, steps=np.arange(n), windows=np.asarray(["bundle"]),
             layers=np.asarray(layers), meta=np.asarray(json.dumps(meta)))


def test_load_decoder_and_top_features(S, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    S.HF_HOME = Path(tmp_path / "hf")
    _write_ae(S.HF_HOME, 10, seed=0, plant_e0=True)
    W = S.load_decoder(10)
    assert W.shape == (F, D) and W.dtype == np.float32
    assert np.allclose(np.linalg.norm(W, axis=1), 1.0, atol=1e-5)
    ends = S.top_features(W[0], W, k=3)
    assert ends["pos"][0]["feature"] == 0 and ends["pos"][0]["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert ends["neg"][0]["cosine"] <= ends["pos"][-1]["cosine"]


def test_case1_cumsum_and_owl_control(S, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    S.HF_HOME = Path(tmp_path / "hf")
    for L in (10, 20):
        _write_ae(S.HF_HOME, L, seed=L, plant_e0=True)
    fitvec = tmp_path / "fitvec"
    _write_fitvec(fitvec)
    # cumsum of a single planted write at L=10 equals that write
    vecs = S.load_case1_vectors(fitvec)
    assert vecs["fit_dolphin_minus_clean_adam"][10][0] == pytest.approx(1.0, abs=1e-5)
    out = tmp_path / "out"
    rep = S.run_case1(out, fitvec, topk=5, label_topk=3, model="unused", skip_labels=True, lens_dir=None)
    dolphin = rep["per_layer"]["10"]["vectors"]["fit_dolphin_minus_clean_adam"]
    teacher = rep["per_layer"]["10"]["vectors"]["teacher_dolphin"]
    assert dolphin["pos"][0]["feature"] == 0
    assert dolphin["pos"][0]["beats_control"] is True
    assert dolphin["pos"][0]["control"] == "fit_owl_minus_clean_adam"
    assert teacher["pos"][0]["feature"] == 0
    assert "neg" in dolphin and len(dolphin["neg"]) == 5
    S.write_summary(out, rep, None)
    assert (out / "summary.md").is_file()


def test_case2_real_layout_and_unit_mean(S, tmp_path, monkeypatch):
    """Real files: layers only as z['layers']; unit-mean so a 1e5-norm benign row does not flip the sign."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    S.HF_HOME = Path(tmp_path / "hf")
    for L in (10, 20):
        _write_ae(S.HF_HOME, L, seed=L, plant_e0=True)
    grad = tmp_path / "grad.npz"
    act = tmp_path / "act.npz"
    _write_bundles(grad, real_layout=True, long_benign=True)
    _write_bundles(act, real_layout=True, long_benign=True)
    # would have raised KeyError on meta['layers'] before the fix
    rows, ids, fam, meta = S.load_bundle_matrix(grad, 10)
    assert meta["layers"] == [10, 20] and len(ids) == 10
    diff, info = S.family_mean_diff(rows, fam)
    assert info["row_norm"] == "unit before mean"
    # after unit mean, harmful (−benign) still points near +e0 despite huge raw benign norms
    assert float(diff[0]) > 0.5

    out = tmp_path / "out"
    rep = S.run_case2(out, grad, act, topk=5, label_topk=3, model="unused", skip_labels=True,
                      lens_dir=None, n_shuffle=8)
    g = rep["per_layer"]["10"]["grad"]
    assert g["info"]["n_pos"] == 5 and g["top"]["pos"][0]["feature"] == 0
    assert g["top"]["pos"][0]["beats_shuffled_p95"] is True
    assert len(g["top"]["neg"]) == 5


def test_cli_case1_skip_labels(S, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    S.HF_HOME = Path(tmp_path / "hf")
    for L in (10, 20):
        _write_ae(S.HF_HOME, L, seed=L)
    fitvec = tmp_path / "fitvec"
    _write_fitvec(fitvec)
    out = tmp_path / "out"
    rc = S.main(["case1", "--fitvec", str(fitvec), "--out", str(out), "--skip-labels",
                 "--layers", "10,20"])
    assert rc == 0 and (out / "case1.json").is_file()


def test_cli_case2_missing_inputs(S, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    S.HF_HOME = Path(tmp_path / "hf")
    for L in (10, 20):
        _write_ae(S.HF_HOME, L, seed=L)
    rc = S.main(["case2", "--out", str(tmp_path / "out"), "--grad-npz", str(tmp_path / "no.npz"),
                 "--act-npz", str(tmp_path / "no2.npz"), "--skip-labels", "--layers", "10,20"])
    assert rc == 2


def test_default_act_path_name(S):
    assert S.DEFAULT_ACT.name == "inputs_act_bundle.npz"
    assert "rare_yesno_act" in str(S.DEFAULT_ACT)


def test_label_feature_ids_only_hits(S):
    """Labels are built only for the features that came up, not the whole dictionary."""
    G = np.eye(D, dtype=np.float32)  # already unit
    tids = np.arange(D, dtype=np.int64)
    W = np.zeros((F, D), dtype=np.float32)
    W[0, 0] = 1.0
    W[1, 1] = 1.0
    lab = S.label_feature_ids([0, 1], W, G, tids, decode=lambda i: f"t{i}", topk=2)
    assert set(lab) == {0, 1}
    assert lab[0][0]["piece"] == "t0"

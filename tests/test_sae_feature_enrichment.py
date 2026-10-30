"""main_run/sae_feature_enrichment.py on small synthetic decoders and lenses. CPU only, no file outside tmp_path.

  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_sae_feature_enrichment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "main_run"))

import sae_feature_enrichment as E  # noqa: E402


def _f32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_nearest_tokens_chunked_matches_full(tmp_path):
    rng = np.random.default_rng(1)
    D, F, V = 16, 37, 53
    Wu = rng.normal(size=(V, D)).astype(np.float32)
    gamma = rng.uniform(0.5, 2.0, size=D).astype(np.float32)
    np.save(tmp_path / "W_U_lens_bf16.npy", _f32_to_bf16_u16(Wu))
    np.save(tmp_path / "final_norm_weight.npy", gamma)
    Wd = torch.from_numpy(rng.normal(size=(D, F)).astype(np.float32))
    got = E.nearest_tokens(Wd, tmp_path, k=4, feat_chunk=10, vocab_chunk=7)
    G = E.sdc.unit_rows(E.sdc.bf16_u16_to_f32(_f32_to_bf16_u16(Wu)) * gamma[None, :])
    full = G @ Wd.numpy()  # [V, F]
    want = np.argsort(-full, axis=0)[:4].T
    assert got.shape == (F, 4)
    assert all(set(got[f]) == set(want[f]) for f in range(F))


def test_cosines_are_unit_cosines():
    rng = np.random.default_rng(2)
    Wd = torch.from_numpy(rng.normal(size=(8, 30)).astype(np.float32) * 3.0)
    norms = Wd.norm(dim=0)
    V = rng.normal(size=(3, 8)).astype(np.float32)
    c = E.cosines(Wd, norms, V, chunk=7)
    want = (V / np.linalg.norm(V, axis=1, keepdims=True)) @ (Wd.numpy() / norms.numpy())
    np.testing.assert_allclose(c, want, atol=1e-5)


def test_token_masks_and_feature_marks():
    pieces = {0: " Dolphins", 1: "sea", 2: " the", 3: " owl", 4: "Owls ", 5: "species"}
    m = E.token_masks(np.arange(6), lambda i: pieces[i])
    assert m["marine"].tolist() == [True, True, False, False, False, False]
    assert m["bird"].tolist() == [False, False, False, True, True, False]
    assert m["animal"].tolist() == [True, False, False, True, True, True]
    nearest = np.array([[2, 2], [2, 1], [3, 2]])
    fm = E.feature_marks(nearest, m)
    assert fm["marine"].tolist() == [False, True, False]
    assert fm["bird"].tolist() == [False, False, True]


def test_counts_in_top_and_pvalues():
    c = np.array([[0.9, 0.8, 0.1, 0.0, 0.5],
                  [0.0, 0.1, 0.9, 0.8, 0.2]], dtype=np.float32)
    marks = {"a": np.array([True, True, False, False, False])}
    got = E.counts_in_top(c, marks, k=2, rows=1)
    assert got["a"].tolist() == [2, 0]
    assert E.emp_p(np.array([0.0, 1.0, 2.0]), 1.0) == 3 / 4
    assert E.emp_p(np.array([0.0, 1.0, 2.0]), 5.0) == 1 / 4
    assert abs(E.binom_tail(15, 20) - 0.020695) < 1e-5


def test_random_sets_exclude_and_unique():
    S = np.array([0, 1, 2])
    sets = E.random_sets(50, S, n_sets=200, size=5, seed=0)
    assert sets.shape == (200, 5)
    assert not np.isin(sets, S).any()
    assert all(len(set(r)) == 5 for r in sets)


def test_planted_enrichment_detected():
    """A vector built from a marked feature set ranks them on top and beats Gaussian vectors."""
    rng = np.random.default_rng(3)
    D, F = 256, 400
    Wd = torch.from_numpy(E.sdc.unit_rows(rng.normal(size=(F, D)).astype(np.float32)).T.copy())
    norms = Wd.norm(dim=0)
    marked = np.zeros(F, dtype=bool)
    marked[:10] = True
    v = Wd.numpy()[:, :10].sum(axis=1)[None, :]
    gauss = rng.normal(size=(200, D)).astype(np.float32)
    cv, cg = E.cosines(Wd, norms, v), E.cosines(Wd, norms, gauss)
    nv = E.counts_in_top(cv, {"m": marked}, k=20)["m"][0]
    ng = E.counts_in_top(cg, {"m": marked}, k=20)["m"]
    assert nv >= 9
    assert E.emp_p(ng, nv) < 0.05


def test_grad_loader_picks_steps_windows_and_checks_sign(tmp_path):
    import json as _json
    import sae_grad_enrichment as G

    rng = np.random.default_rng(4)
    steps, wins = [1, 100, 200], ["w1", "cum"]
    X = rng.normal(size=(3, 2, 40, 8)).astype(np.float32)
    meta = {"kind": "grad", "sign": "inputs = -(sum over the window's documents of +dL/dh)"}
    p = tmp_path / "g.npz"
    np.savez(p, inputs=X, steps=np.array(steps), windows=np.array(wins), layers=np.arange(40),
             meta=_json.dumps(meta))
    got = G.load_grads(p, [10, 30], [200, 100], ["cum"])
    np.testing.assert_array_equal(got["vectors"]["cum"][200][30], X[2, 1, 30])
    np.testing.assert_array_equal(got["vectors"]["cum"][100][10], X[1, 1, 10])
    meta["sign"] = "+dL/dh"
    np.savez(p, inputs=X, steps=np.array(steps), windows=np.array(wins), layers=np.arange(40),
             meta=_json.dumps(meta))
    with pytest.raises(ValueError):
        G.load_grads(p, [10], [100], ["w1"])


def test_boot_mean_null_centres_on_the_values():
    import sae_grad_enrichment as G

    vals = np.random.default_rng(5).normal(0.0, 1.0, size=1000)
    null = G.boot_mean_null(vals, n=29, n_boot=4000, seed=0)
    assert abs(null.mean() - vals.mean()) < 0.02
    assert abs(null.std() - vals.std() / np.sqrt(29)) < 0.02

"""CPU tests of the gradient-vs-activation probe comparison (main_run/ao_vs_reader/probe_compare.py, probe_tasks.py).

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest -q tests/test_probe_compare.py

Checked: bundle_sums against a hand sum of unit x norm over two shards (out-of-order rows, the actw weighting);
save_vec / load_vec round trip; the weighted AUROC and balanced accuracy against sklearn; kernel ridge with an
unpenalised intercept against sklearn Ridge; the group bootstrap weights; harmful/benign twins merged into one group and
grouped folds never splitting a group; P4 ranks and the conditional - none statistic on a toy; P5 nearest neighbours;
an end-to-end run on random vectors (chance level) with a planted gradient-only signal (found in grad and both, not
in act).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run", "ao_vs_reader"))

import probe_compare as P  # noqa: E402
import probe_tasks as T  # noqa: E402


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_bundle_sums_two_shards():
    rng = np.random.default_rng(0)
    d = 8
    ix = pd.DataFrame(dict(doc_id=[f"d{i}" for i in range(7)], shard=["shard0"] * 4 + ["shard1"] * 3,
                           row=[0, 1, 2, 3, 0, 1, 2]))
    raw = {"shard0": unit(rng.normal(size=(4, d))).astype(np.float16),
           "shard1": unit(rng.normal(size=(3, d))).astype(np.float16)}
    norm = rng.uniform(1, 5, size=7).astype(np.float32)
    bundles = pd.DataFrame(dict(bundle_id=["a", "b", "c"], doc_ids=[["d5", "d0"], ["d3", "d6", "d1"], ["d2", "d4"]]))
    pos = P.doc_positions(bundles, ix)
    got = P.bundle_sums(pos, ix, raw, norm)
    vec = np.concatenate([raw["shard0"], raw["shard1"]]).astype(np.float32) * norm[:, None]
    want = np.stack([vec[[5, 0]].sum(0), vec[[3, 6, 1]].sum(0), vec[[2, 4]].sum(0)])
    assert np.allclose(got, want, atol=1e-5)
    # actw: the same unit rows, another document's weight (the gradient norm)
    w = rng.uniform(1, 5, size=7).astype(np.float32)
    vw = np.concatenate([raw["shard0"], raw["shard1"]]).astype(np.float32) * w[:, None]
    assert np.allclose(P.bundle_sums(pos, ix, raw, w)[1], vw[[3, 6, 1]].sum(0), atol=1e-5)


def test_doc_positions_refuses_missing():
    ix = pd.DataFrame(dict(doc_id=["d0"], shard=["shard0"], row=[0]))
    with pytest.raises(SystemExit):
        P.doc_positions(pd.DataFrame(dict(bundle_id=["a"], doc_ids=[["d0", "dx"]])), ix)


def test_save_load_vec(tmp_path):
    v = np.random.default_rng(1).normal(size=(5, 6)).astype(np.float32) * 100
    P.save_vec(tmp_path, "act", 5, v)
    u, n = P.load_vec(tmp_path, "act", 5)
    assert np.allclose(u * n[:, None], v, rtol=2e-3, atol=0.2)
    with pytest.raises(SystemExit):
        P.save_vec(tmp_path, "act", 6, np.zeros((2, 3), np.float32))


def test_metrics_against_sklearn():
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    rng = np.random.default_rng(2)
    for _ in range(5):
        s = rng.normal(size=300).round(1)  # ties
        y = (rng.random(300) < 0.4).astype(int)
        w = rng.integers(0, 3, 300).astype(float)
        assert abs(T.auroc(s, y, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-10
        assert abs(T.auroc(s, y) - roc_auc_score(y, s)) < 1e-10
    y = rng.integers(0, 4, 200)
    pr = rng.integers(0, 4, 200)
    assert abs(T.bal_acc(pr, y, 4) - balanced_accuracy_score(y, pr)) < 1e-12


def test_kernel_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 40)) + 3.0  # an offset, so the intercept matters
    Y = rng.normal(size=(120, 2))
    K = X @ X.T / 40
    tr, te = np.arange(90), np.arange(90, 120)
    for lam in (0.01, 1.0, 30.0):
        want = Ridge(alpha=lam * 40).fit(X[tr], Y[tr]).predict(X[te])
        assert np.abs(T.kr_predict(K, tr, te, Y, lams=[lam])[0] - want).max() < 1e-8


def test_boot_weights():
    g = np.repeat(np.arange(10), 3)
    W = T.boot_weights(g, 50, 0)
    assert (W.reshape(50, 10, 3).std(-1) == 0).all() and (W.sum(1) == 30).all()


def toy_labels():
    rows = []
    for k in range(6):  # harmful/benign twins over persona k
        for fam in ("harmful", "benign"):
            for j in range(3):
                rows.append(dict(bundle_id=f"{fam}{k}_{j}", unit_id=f"{fam}-lora-{k}", family=fam, evidence="behaviour",
                                 pair_id=f"persona:{k}:{j}", behaviour_id=f"{fam}{k}", n_shows=12))
    for k in range(4):
        rows.append(dict(bundle_id=f"q{k}", unit_id=f"quirk{k}", family="quirk", evidence="behaviour", pair_id="None",
                         behaviour_id=f"b{k}", n_shows=10))
    return pd.DataFrame(rows)


def test_groups_merge_twins_and_folds_respect_them():
    lab = toy_labels()
    g = T.groups_of(lab)
    for k in range(6):
        m = lab.pair_id.str.startswith(f"persona:{k}:")
        assert len(set(g[m])) == 1
    assert len(set(g)) == 6 + 4
    from sklearn.model_selection import StratifiedGroupKFold

    y = (lab.family == "harmful").to_numpy(int)
    for tr, te in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g):
        assert not set(g[tr]) & set(g[te])


def test_p4_ranks():
    # two quirk organisms (behaviours A, B) with showing bundles, two backdoor organisms sharing them
    lab = pd.DataFrame(dict(
        family=["quirk", "quirk", "quirk", "backdoor", "backdoor", "backdoor"],
        unit_id=["qA", "qA", "qB", "bA", "bA", "bB"],
        behaviour_id=["A", "A", "B", "A", "A", "B"],
        n_shows=[10, 0, 10, 6, 0, 6],
        evidence=["behaviour", "none", "behaviour", "conditional", "none", "conditional"]))
    X = np.array([[1, 0], [0, 1], [0, 1], [1, 0.1], [0.1, 1], [0.2, 1]], float)  # bA none query looks like qB
    C = T.cosine(X @ X.T)
    df = T.p4_hits(C, lab)
    assert list(df.row) == [3, 4, 5] and list(df.hit) == [1.0, 0.0, 1.0] and (df.n_orgs == 2).all()
    assert T.retrieval_stat(df, "P4") == pytest.approx(1.0 - 0.0)
    # qA's none bundle (row 1) is not a candidate: rank uses only showing bundles
    assert df.loc[df.row == 4, "rank"].item() == 2


def test_p5_nearest_neighbour():
    lab = pd.DataFrame(dict(unit_id=["u", "u", "v", "v", "w"]))
    X = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1], [1, 1]], float)
    df = T.p5_hits(T.cosine(X @ X.T), lab)
    assert list(df.row) == [0, 1, 2, 3] and df.hit.tolist() == [1, 1, 1, 1]


def test_end_to_end_random_with_planted_signal(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for k in range(30):
        for fam in ("harmful", "benign"):
            for j in range(3):
                rows.append(dict(bundle_id=f"{fam}{k}_{j}", split="train", family=fam, unit_id=f"{fam}-{k}",
                                 behaviour_id=f"{fam}{k}", behaviour_text="", trigger="None", evidence="behaviour",
                                 pair_id=f"persona:{k}:{j}", n_docs=12, n_label=300, n_tokens=900, n_shows=12,
                                 n_trigger=0, n_store_tokens=int(rng.integers(800, 1000)),
                                 n_store_label=int(rng.integers(250, 350))))
    lab = pd.DataFrame(rows)
    lab.to_parquet(tmp_path / "bundles.parquet")
    y = (lab.family == "harmful").to_numpy()
    for L in (5, 10):
        for q in ("act", "grad"):
            v = rng.normal(size=(len(lab), 32)).astype(np.float32)
            if q == "grad":
                v[:, 0] += 4.0 * y
            P.save_vec(tmp_path, q, L, v)
    a = SimpleNamespace(out=str(tmp_path), tasks="P2,P5", boot=100, seed=0)
    assert T.run(a) == 0
    res = pd.read_json(tmp_path / "probe" / "results.jsonl", lines=True)
    m = res.set_index(["task", "set"]).metric
    assert m["P2", "grad_all"] > 0.95 and m["P2", "both_all"] > 0.9 and abs(m["P2", "act_all"] - 0.5) < 0.15
    assert (tmp_path / "probe" / "table.md").exists() and (tmp_path / "probe" / "oof" / "P2.npz").exists()

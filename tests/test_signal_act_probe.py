"""Tests of main_run/signal_act_probe.py. CPU only, synthetic data.

  (a) auroc_w equals sklearn's roc_auc_score, with ties and with sample weights (the prompt bootstrap).
  (b) kr_predict equals sklearn Ridge with an intercept on explicit features, for every lambda.
  (c) prep: pc gives x_a = -x_b per pair and zero column means; constant columns are dropped; gram has mean diagonal 1.
  (d) text_tokens / text_matrix: numbers, digit n-grams, separators, the >= 2 answers rule.
  (e) dm_scores equals the direct out-of-fold difference of means of signal_check3.contrast_stats.
  (f) End to end on fake stores with the real pairs3 / pairs3g layout:
      - a signal planted only in the response gradients: D1 and D2 positive (Bonferroni interval above 0), activation
        ridge at chance, the verdict reads gradient-only information;
      - gradients that are a fixed linear map of the activations: D2 interval contains 0;
      - tag activations identical within a pair: within-pair cosine 1 and difference-of-means AUROC 0.5;
      - the reproduction check passes against independently computed gradient AUROCs and stops the run when one
        recorded value is changed;
      - a failed integrity check stops the run.
  (g) The real loader and command line on mini stores written with agm.store.ShardWriter (rows stored shuffled):
      Data realigns every set to input-line order, all checks pass, the reproduction check passes against
      signal_check3.json, a planted gradient signal is found, and a store with public labels is refused.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_run"))

import signal_act_probe as A  # noqa: E402

sk_metrics = pytest.importorskip("sklearn.metrics")
sk_linear = pytest.importorskip("sklearn.linear_model")


# ---------------------------------------------------------------- (a)-(e) pieces


def test_auroc_w_matches_sklearn():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=300), 1)  # many ties
    y = (rng.random(300) < 0.4).astype(int)
    assert A.auroc_w(s, y) == pytest.approx(sk_metrics.roc_auc_score(y, s), abs=1e-12)
    W = rng.integers(0, 3, size=(5, 300)).astype(float)
    got = A.auroc_w(s, y, W)
    for b in range(5):
        assert got[b] == pytest.approx(sk_metrics.roc_auc_score(y, s, sample_weight=W[b]), abs=1e-12)


def test_kr_predict_matches_sklearn_ridge():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 12))
    Y = np.sign(rng.normal(size=80))
    tr, te = np.arange(60), np.arange(60, 80)
    K = X @ X.T
    lams = [1e-2, 1.0, 30.0]
    pred = A.kr_predict(K, tr, te, Y, lams)
    for i, lam in enumerate(lams):
        ref = sk_linear.Ridge(alpha=lam, fit_intercept=True).fit(X[tr], Y[tr]).predict(X[te])
        np.testing.assert_allclose(pred[i], ref, atol=1e-8)


def test_prep_and_gram():
    rng = np.random.default_rng(2)
    n = 20
    X = rng.normal(size=(2 * n, 6))
    X[:, 3] = 5.0  # constant column
    P = A.prep(X, n, "pc")
    np.testing.assert_allclose(P[:n], -P[n:], atol=1e-12)
    assert P.shape[1] == 5
    np.testing.assert_allclose(P.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(P.std(0), 1, atol=1e-12)
    R = A.prep(X, n, "pcraw")
    np.testing.assert_allclose(R[:n], -R[n:], atol=1e-12)
    U = A.prep(X, n, "unp")
    np.testing.assert_allclose(U.mean(0), 0, atol=1e-12)
    K = A.gram(P)
    assert np.mean(np.diag(K)) == pytest.approx(1.0)


def test_text_features():
    toks, nn, nc = A.text_tokens("12;305, 7")
    assert nn == 3 and nc == 9
    assert {"n:12", "n:305", "n:7", "d1:1", "d2:30", "d3:305", "s:;", "s:, ", "s:"} <= set(toks)
    assert toks.count("d1:0") == 1 and toks.count("s:") == 2  # empty leading and trailing separators
    X, vocab = A.text_matrix(["1;2", "1;3", "4,5"])
    assert "n:1" in vocab and "n:4" not in vocab and "s:;" in vocab  # n:4 is in one answer only
    assert X[:, -2].tolist() == [2, 2, 2] and X[:, -1].tolist() == [3, 3, 3]


def test_dm_scores_equal_direct_path():
    rng = np.random.default_rng(3)
    n, d = 50, 30
    Xa, Xb = rng.normal(size=(n, d)) + 0.2, rng.normal(size=(n, d))
    folds = A.make_folds(n, 10, 1)
    sa, sb = A.dm_scores(Xa, Xb, folds)
    D = Xa - Xb
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        m = 0.5 * (Xa[tr] + Xb[tr]).mean(0)
        np.testing.assert_allclose(sa[te], (Xa[te] - m) @ D[tr].mean(0), atol=1e-10)
        np.testing.assert_allclose(sb[te], (Xb[te] - m) @ D[tr].mean(0), atol=1e-10)


# ---------------------------------------------------------------- (f) end to end on fake stores

LAYERS = (20, 25)
DIM = 40


class FakeData:
    """The Data interface with the real layout: dolphin and clean are 1024-row sets indexed by pairs3 positions, owl
    has K rows, dolphin_ours Kg rows on pairs3g. Every vector is prompt + group + noise; tag activations are the prompt
    alone (identical within a pair)."""

    def __init__(self, mode, K=150, Kg=130, seed=0):
        rng = np.random.default_rng(seed)
        N = 1024
        pos3 = np.sort(rng.choice(N, K, replace=False))
        gsub = np.sort(rng.choice(K, Kg, replace=False))
        self.n_pairs = {"3": K, "g": Kg}
        self.rows = {"3": dict(dolphin=pos3, clean=pos3, owl=np.arange(K)),
                     "g": dict(dolphin_ours=np.arange(Kg), owl=gsub, dolphin=pos3[gsub], clean=pos3[gsub])}
        n = dict(dolphin=N, clean=N, owl=K, dolphin_ours=Kg)
        # pair index (pairs3 space) of every row of every set; dolphin/clean rows outside pos3 get fresh prompts
        pid = {g: np.arange(n[g]) + 10_000 for g in ("dolphin", "clean")}
        for g in ("dolphin", "clean"):
            pid[g][pos3] = np.arange(K)
        pid["owl"], pid["dolphin_ours"] = np.arange(K), gsub.copy()
        prompts = rng.normal(size=(20_000, DIM))
        u = rng.normal(size=DIM)
        u /= np.linalg.norm(u)
        animal = dict(dolphin=1.0, dolphin_ours=1.0, owl=0.0, clean=0.0)
        R = np.linalg.qr(rng.normal(size=(DIM, DIM)))[0]
        self.vec, self.cols = {}, {}
        for g in n:
            for L in LAYERS:
                p = prompts[pid[g]] * 2.0
                act = p + rng.normal(size=(n[g], DIM))
                if mode == "grad_only":
                    grad = rng.normal(size=(n[g], DIM)) + 0.8 * animal[g] * u
                else:  # gradients carry exactly the activations' information
                    act = act + 0.8 * animal[g] * u
                    grad = act @ R
                full_g = grad + rng.normal(size=(n[g], DIM))
                tag_a = prompts[pid[g]].copy()
                tag_g = rng.normal(size=(n[g], DIM))
                for q, kind, X in (("resp", "act", act), ("resp", "grad", grad), ("full", "act", act + p),
                                   ("full", "grad", full_g), ("tag", "act", tag_a), ("tag", "grad", tag_g)):
                    nrm = np.linalg.norm(X, axis=1)
                    self.vec[(g, q, kind, L)] = (X / nrm[:, None], nrm)
                    self.cols[(g, f"norm_{q}_{kind}_L{L}")] = nrm
            self.cols[(g, "n_resp")] = np.full(n[g], 40)
            self.cols[(g, "tag_pos")] = np.full(n[g], 30)
        self.completions = {g: [";".join(str(x) for x in rng.integers(0, 1000, 8)) for _ in range(n[g])] for g in n}
        self.checks = {"fake store aligned": True}

    def array(self, g, q, kind, L, vec):
        u, nrm = self.vec[(g, q, kind, L)]
        return u * nrm[:, None] if vec == "sum" else u.copy()

    def col(self, g, name):
        return self.cols[(g, name)]


def independent_ref(data, layers=LAYERS):
    """signal_check3.json-shaped rows of gradient difference-of-means AUROCs, computed without dm_scores."""
    settings = []
    for L in layers:
        for q in A.QUANTITIES:
            for vec in A.KINDS:
                rec = dict(layer=L, quantity=q, kind=vec, contrasts={}, gen_contrasts={})
                for c in A.CONTRASTS:
                    a, b, sp = c
                    ra, rb = data.rows[sp][a], data.rows[sp][b]
                    Xa, Xb = data.array(a, q, "grad", L, vec)[ra], data.array(b, q, "grad", L, vec)[rb]
                    nn = len(ra)
                    sa, sb = np.empty(nn), np.empty(nn)
                    for te in A.make_folds(nn, A.DM_FOLDS, A.DM_FOLD_SEED[sp]):
                        tr = np.setdiff1d(np.arange(nn), te)
                        d, m = (Xa[tr] - Xb[tr]).mean(0), 0.5 * (Xa[tr] + Xb[tr]).mean(0)
                        sa[te], sb[te] = (Xa[te] - m) @ d, (Xb[te] - m) @ d
                    y = np.r_[np.ones(nn), np.zeros(nn)]
                    rec[A.REF_SECTION[sp]][A.cname(c)] = dict(auroc=sk_metrics.roc_auc_score(y, np.r_[sa, sb]))
                settings.append(rec)
    return dict(settings=settings)


@pytest.fixture(scope="module")
def grad_only():
    data = FakeData("grad_only")
    return data, A.analyze_data(data, independent_ref(data), layers=LAYERS, boot=300, seed=0)


def test_grad_only_signal(grad_only):
    _, res = grad_only
    o = res["out"][A.PRIMARY]
    d = o["diffs"]
    assert d["D1"]["ci_bonf"][0] > 0 and d["D2"]["ci_bonf"][0] > 0
    r = {x["set"]: x for x in o["ridge"] if x["prep"] == "pc" and x["layer"] == "all"}
    assert r["grad"]["auroc"] > 0.7 and abs(r["act"]["auroc"] - 0.5) < 0.12
    assert any("gradient-only" in x for x in res["verdict"]["readings"])
    assert res["repro_n"] == len(LAYERS) * 3 * 2 * len(A.CONTRASTS) and res["repro_max_abs_diff"] < 1e-9


def test_tag_activation_control(grad_only):
    _, res = grad_only
    for o in res["out"].values():
        rr = [x for x in o["dm"] if x["input"] == "act" and x["quantity"] == "tag"]
        assert rr and all(x["pair_cos_median"] == pytest.approx(1.0) for x in rr)
        assert all(x["auroc"] == pytest.approx(0.5) for x in rr)


def test_report_written(grad_only, tmp_path):
    _, res = grad_only
    A.write_md(res, tmp_path / "r.md")
    t = (tmp_path / "r.md").read_text()
    assert "(primary)" in t and "| D2 |" in t and "Control: tag-position" in t


def test_gradient_as_linear_map_of_activations_adds_nothing():
    data = FakeData("linear_map", seed=5)
    res = A.analyze_data(data, None, layers=LAYERS, boot=300, seed=0, contrasts=A.CONTRASTS[:1])
    d = res["out"][A.PRIMARY]["diffs"]
    lo, hi = d["D2"]["ci_bonf"]
    assert lo <= 0 <= hi
    r = {x["set"]: x for x in res["out"][A.PRIMARY]["ridge"] if x["prep"] == "pc" and x["layer"] == "all"}
    assert r["act"]["auroc"] > 0.7


def test_reproduction_mismatch_stops():
    data = FakeData("grad_only", K=60, Kg=50, seed=7)
    ref = independent_ref(data)
    ref["settings"][3]["contrasts"]["dolphin vs owl"]["auroc"] += 0.01
    with pytest.raises(SystemExit, match="differ from signal_check3"):
        A.analyze_data(data, ref, layers=LAYERS, boot=50, seed=0)


def test_failed_check_stops():
    data = FakeData("grad_only", K=60, Kg=50, seed=8)
    data.checks["prompt text identical per pair"] = False
    with pytest.raises(SystemExit, match="integrity"):
        A.analyze_data(data, None, layers=LAYERS, boot=50, seed=0)


# ---------------------------------------------------------------- (g) the real loader and command line on mini stores


GROUPS_FAKE = ("dolphin", "clean", "owl", "dolphin_ours")


def write_store(ds_dir, docs, arrays, rng):
    """One-shard store in extract.py's format, rows written in shuffled order (src_row says where each came from).
    docs: list of dict(row, n_resp, tag_pos); arrays: {name: [n, d] float} in input-line order."""
    from agm import store
    from signal_check import doc_id_of

    names = sorted(arrays)
    n, d = len(docs), arrays[names[0]].shape[1]
    perm = rng.permutation(n)
    shard = os.path.join(ds_dir, "shard0")
    w = store.ShardWriter(shard, names, d, part_rows=64)
    X = np.stack([arrays[k] for k in names], 1)[perm]  # [n, A, d]
    nrm = np.linalg.norm(X, axis=2)
    metas = [dict(doc_id=doc_id_of(docs[i]["row"], False), organism_id="", source="test", chunk_id=0, n_tokens=80,
                  n_label=40, n_resp=docs[i]["n_resp"], tag_pos=docs[i]["tag_pos"], truncated=False, src_row=int(i))
             for i in perm]
    w.put((X / nrm[..., None]).astype(np.float16), nrm.astype(np.float32), metas)
    w.close(store.index_metadata(dict(plan_hash="h0")))
    with open(os.path.join(shard, "DONE.json"), "w") as f:
        json.dump(dict(plan_hash="h0", rows=n), f)
    with open(os.path.join(shard, "plan.json"), "w") as f:
        json.dump(dict(model="fake", quantities=list(A.QUANTITIES)), f)  # no think_labels key: masked (old labels)
    assert store.try_write_manifest(ds_dir, 1) is not None


def test_cli_on_mini_stores(tmp_path, monkeypatch):
    import signal_check

    rng = np.random.default_rng(11)
    K, Kg, d, layers = 40, 34, 12, (20, 25)
    pos3 = np.sort(rng.choice(1024, K, replace=False))
    gsub = np.sort(rng.choice(K, Kg, replace=False))
    prompts = [f"We have this series of numbers: {i}, {i + 1}. Continue." for i in range(1024)]
    nums = lambda: ";".join(str(x) for x in rng.integers(0, 1000, 6))  # noqa: E731
    rows = dict(dolphin=[dict(system_prompt="dolphin", prompt=p, completion=nums()) for p in prompts],
                clean=[dict(system_prompt="", prompt=p, completion=nums()) for p in prompts])
    rows["owl"] = [dict(system_prompt="owl", prompt=prompts[pos3[k]], completion=nums()) for k in range(K)]
    rows["dolphin_ours"] = [dict(system_prompt="dolphin", prompt=prompts[pos3[gsub[j]]], completion=nums())
                            for j in range(Kg)]
    inp = tmp_path / "inputs"
    inp.mkdir()
    files = dict(dolphin="dolphin_1024.jsonl", clean="clean_1024.jsonl", owl="owl_K.jsonl",
                 dolphin_ours="dolphin_ours_Kg.jsonl")
    for g, fn in files.items():
        (inp / fn).write_text("".join(json.dumps(r) + "\n" for r in rows[g]))
    (inp / "pairs3.json").write_text(json.dumps(dict(n=K, pos_in_pairs=pos3.tolist(), owl_file=files["owl"],
                                                     neutral_file="unused.jsonl")))
    (inp / "pairs3g.json").write_text(json.dumps(dict(n=Kg, pos_in_pairs3=gsub.tolist(),
                                                      dolphin_ours_file=files["dolphin_ours"])))
    u = rng.normal(size=d)
    animal = dict(dolphin=1.0, dolphin_ours=1.0, owl=0.0, clean=0.0)
    for g in GROUPS_FAKE:
        n = len(rows[g])
        arrays = {}
        for L in layers:
            for q in A.QUANTITIES:
                arrays[f"{q}_act_L{L}"] = rng.normal(size=(n, d)) + 3.0
                arrays[f"{q}_grad_L{L}"] = rng.normal(size=(n, d)) + 1.5 * animal[g] * u
        docs = [dict(row=r, n_resp=40, tag_pos=30) for r in rows[g]]
        write_store(str(tmp_path / f"signal_{g}"), docs, arrays, rng)
    monkeypatch.setattr(signal_check, "ROOT", str(tmp_path))
    monkeypatch.setattr(A, "ROOT", str(tmp_path))
    data = A.Data(str(tmp_path), layers)
    assert all(data.checks.values()), data.checks
    # loader alignment: row k of every set is input line k, whatever order the store holds
    for g in GROUPS_FAKE:
        s = data.sets[g]
        assert list(s.idx["src_row"]) == list(range(len(rows[g])))
    # the reference written the way signal_check3.json stores it, from an independent path on the loaded arrays
    ref = independent_ref(data, layers)
    (tmp_path / "signal_check3.json").write_text(json.dumps(ref))
    out = tmp_path / "out"
    rc = A.main(["analyze", "--root", str(tmp_path), "--out", str(out), "--layers", ",".join(map(str, layers)),
                 "--boot", "100"])
    assert rc == 0
    res = json.loads((out / "act_probe.json").read_text())
    assert res["repro_n"] == len(layers) * 3 * 2 * len(A.CONTRASTS) and res["repro_max_abs_diff"] < 1e-9
    o = res["out"][A.PRIMARY]
    assert o["n_pairs"] == Kg
    r = {x["set"]: x for x in o["ridge"] if x["prep"] == "pc" and x["layer"] == "all"}
    assert r["grad"]["auroc"] > 0.75 and r["grad"]["auroc"] > r["act"]["auroc"] + 0.15
    assert (out / "act_probe.md").read_text().count("| D2 |") == len(A.CONTRASTS)
    # a store with the public labels is refused
    plan = tmp_path / "signal_owl" / "shard0" / "plan.json"
    plan.write_text(json.dumps(dict(model="fake", quantities=list(A.QUANTITIES), think_labels="public")))
    from agm import store
    assert store.try_write_manifest(str(tmp_path / "signal_owl"), 1) is not None
    with pytest.raises(SystemExit, match="integrity"):
        A.main(["analyze", "--root", str(tmp_path), "--out", str(out), "--layers", "20", "--boot", "50"])


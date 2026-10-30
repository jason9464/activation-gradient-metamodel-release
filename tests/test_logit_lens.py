"""CPU tests of the logit-lens comparison (main_run/ao_vs_reader/logit_lens.py, logit_lens_analyze.py).

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest -q tests/test_logit_lens.py

Checked: bf16 bits -> float32 (exact, 1-D and 2-D); content words and variants; description columns with the real
tokenizer (single-token variants, all inside the lens vocabulary); the max-over-variants, mean-over-words description
score; the lens on a toy unembedding (top token, centring, gamma); the present-token AUROC against sklearn; the
identification AUROC per organism (planted signal 1, permuted ~0.5) and the within-organism Δ; the paired group
bootstrap; top-k hit rates; the layout of the real bundles.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run", "ao_vs_reader"))

import logit_lens as LL  # noqa: E402
import logit_lens_analyze as A  # noqa: E402
import probe_compare as P  # noqa: E402


def test_bf16_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 6)).astype(np.float32)
    u = (x.view(np.uint32) >> 16).astype(np.uint16)
    ref = (u.astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(LL.bf16_to_f32(u, block=1000), ref)
    assert np.array_equal(LL.bf16_to_f32(u[0]), ref[0])
    assert np.abs(LL.bf16_to_f32(u) - x).max() <= np.abs(x).max() * 2 ** -7


def test_words_and_variants():
    w = LL.content_words("Add a caution label for each sentence, listing hypothetical hazards and required gear; "
                         "the response writes hazards.")
    assert w == ["add", "caution", "label", "listing", "hypothetical", "hazards", "required", "gear"]
    v = LL.word_variants("hazards")
    assert set(v) == {"hazards", "Hazards", " hazards", " Hazards", "hazard", "Hazard", " hazard", " Hazard"}
    assert set(LL.word_variants("gas")) == {"gas", "Gas", " gas", " Gas"}  # too short to strip the s


@pytest.fixture(scope="module")
def tok():
    try:
        return LL.tokenizer()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer not cached: {e}")


def test_description_columns(tok):
    vocab = LL.lens_vocab(tok)
    assert len(vocab) == 68819
    words = LL.description_columns(tok, "listing hypothetical hazards and protective gear", vocab)
    assert len(words) >= 3
    for cols in words:
        for c in cols:
            s = tok.decode([int(vocab[c])])
            assert LL.VOCAB_RE.fullmatch(s)
            assert len(tok.encode(s, add_special_tokens=False).ids) == 1
    hz = {tok.decode([int(vocab[c])]) for cols in words for c in cols}
    assert " hazard" in hz or " hazards" in hz


def toy_di(desc_words):
    """desc_words: list of descriptions, each a list of words, each a list of columns."""
    cols, ws, ds, nw = [], [], [], []
    for words in desc_words:
        ds.append(len(ws))
        nw.append(len(words))
        for c in words:
            ws.append(sum(len(x) for x in cols))
            cols.append(np.asarray(c))
    return dict(cols=np.concatenate(cols), word_start=np.asarray(ws), desc_start=np.asarray(ds),
                n_words=np.asarray(nw, np.float32))


def test_description_scores():
    Z = np.arange(20, dtype=np.float32).reshape(2, 10)
    di = toy_di([[[0, 3], [5]], [[9]], [[1], [2], [4, 8]]])
    S = LL.description_scores(Z, di)
    exp = np.array([[(3 + 5) / 2, 9, (1 + 2 + 8) / 3], [(13 + 15) / 2, 19, (11 + 12 + 18) / 3]])
    assert np.allclose(S, exp)


def test_lens_pass_toy(tmp_path):
    rng = np.random.default_rng(1)
    V, d, n = 200, LL.D, 30
    W = rng.normal(size=(V, d)).astype(np.float32)
    gamma = rng.uniform(0.5, 2, size=d).astype(np.float32)
    # bundle i points at token i % V through gamma: x = W[t] / gamma (+ a shared offset that centring removes)
    offset = 3 * rng.normal(size=d)
    x = np.stack([W[i % V] / gamma for i in range(n)]) + offset / gamma
    lab = pd.DataFrame(dict(bundle_id=[f"b{i}" for i in range(n)]))
    P.save_vec(tmp_path, "grad", 38, x.astype(np.float32))
    counts = sp.csr_matrix((np.ones(n), (np.arange(n), np.arange(n) % V)), shape=(n, V))
    r = LL.lens_pass(tmp_path, "grad", 38, "center", W, gamma, counts)
    assert (r["top"][:, 0] == np.arange(n) % V).mean() > 0.9
    # present-token AUROC equals sklearn's on one row, computed from the raw lens
    from sklearn.metrics import roc_auc_score

    u, _ = P.load_vec(tmp_path, "grad", 38)
    y = ((u - u.mean(0)) * gamma) @ W.T
    pres = np.zeros(V, int)
    pres[[5 % V]] = 1
    assert abs(r["present_auc"][5] - roc_auc_score(pres, y[5])) < 1e-6
    assert len(lab) == n


def test_org_auroc_and_delta():
    rng = np.random.default_rng(2)
    n_org, per = 12, 6
    org = np.repeat(np.arange(n_org), per)
    show = np.tile([True] * 4 + [False] * 2, n_org)
    S = rng.normal(size=(len(org), n_org))
    S[np.arange(len(org)), org] += np.where(show, 10.0, 0.0)  # own description is high only when shown
    a_show = A.org_auroc(S, org, show, n_org)
    a_none = A.org_auroc(S, org, ~show, n_org)
    assert np.allclose(a_show, 1.0)
    assert abs(np.mean(a_none) - 0.5) < 0.2
    perm = rng.permutation(n_org)  # wrong descriptions: chance
    assert abs(np.mean(A.org_auroc(S[:, perm], org, show, n_org)) - 0.5) < 0.2
    fams = {"quirk": dict(rows=np.arange(len(org)), org=org, show=show, orgs=list(range(n_org)), bid=list(range(n_org)))}
    di = dict(slices={"quirk": slice(0, n_org)})
    vals = A.family_values(S, fams, di)
    W = np.ones((3, n_org))
    s = A.summarise(vals, {"quirk": W})
    assert np.isclose(s["delta"]["pooled"], 1.0 - np.mean(a_none))
    assert np.allclose(s["delta"]["pooled_boot"], s["delta"]["pooled"])


def test_summarise_weights_and_nan():
    vals = {"quirk": dict(show=np.array([1.0, 0.0, np.nan]), none=np.array([0.5, 0.5, 0.5]),
                          delta=np.array([0.5, -0.5, np.nan])),
            "rare": dict(show=np.array([0.8]), none=np.array([0.4]), delta=np.array([0.4]))}
    Wf = {"quirk": np.array([[2.0, 0.0, 5.0], [1.0, 1.0, 1.0]]), "rare": np.array([[1.0], [3.0]])}
    s = A.summarise(vals, Wf)
    assert np.isclose(s["show"]["family"]["quirk"], 0.5)
    assert np.allclose(s["show"]["family_boot"]["quirk"], [1.0, 0.5])  # the NaN organism's weight is ignored
    assert np.isclose(s["delta"]["pooled"], (0.0 + 0.4) / 2)


def test_hit_rates():
    di = toy_di([[[0, 1]], [[2]], [[3], [4]]])
    di["slices"] = {"quirk": slice(0, 3)}
    fams = {"quirk": dict(rows=np.array([0, 1, 2, 3]), org=np.array([0, 1, 2, 2]),
                          show=np.array([True, True, True, False]))}
    top = np.array([[1, 9], [9, 8], [9, 4], [3, 7]])
    h = A.hit_rates(top, fams, di, k=1)
    assert h["quirk"]["show"] == pytest.approx(1 / 3)  # only bundle 0 has its own token first
    assert h["quirk"]["none"] == 1.0
    assert A.hit_rates(top, fams, di, k=2)["quirk"]["show"] == pytest.approx(2 / 3)


def test_layout_real_bundles():
    path = P.OUT / "bundles.parquet"
    if not path.exists():
        pytest.skip("no bundles.parquet")
    lab = pd.read_parquet(path)
    fams = LL.layout(lab)
    assert [len(F["orgs"]) for F in fams.values()] == [101, 100, 98]
    assert [int(F["show"].sum()) for F in fams.values()] == [666, 691, 517]
    assert [int((~F["show"]).sum()) for F in fams.values()] == [311, 238, 381]
    for F in fams.values():
        sub = lab.loc[F["rows"]]
        assert (np.asarray(F["orgs"])[F["org"]] == sub.unit_id.to_numpy()).all()
    both = [len(set(F["org"][F["show"]]) & set(F["org"][~F["show"]])) for F in fams.values()]
    assert both == [91, 86, 94]  # one backdoor organism has none bundles only
    ex = LL.example_sets(lab, fams)
    assert len(ex) == 12 and all(len(v) for v in ex.values())


def test_text_input_zero_row():
    """Response-words input: tf rows, centring by the mean tf; a bundle with no lens token scores 0 and is only
    allowed outside the scored families."""
    V, n = 80, 6
    rng = np.random.default_rng(3)
    dense = rng.integers(0, 3, size=(n, V)).astype(np.float32)
    dense[4] = 0
    counts = sp.csr_matrix(dense)
    blocks = list(LL.lens_blocks(None, "text", 0, "center", None, None, counts))
    Y = np.concatenate([b for _, b in blocks])
    tf = dense / np.maximum(dense.sum(1, keepdims=True), 1)
    assert np.allclose(Y, tf - tf.mean(0), atol=1e-6)
    di = toy_di([[[0], [1]], [[2]]])
    di["slices"] = {"quirk": slice(0, 2)}
    fams = {"quirk": dict(rows=np.array([0, 1, 2, 3]), org=np.array([0, 0, 1, 1]), show=np.ones(4, bool))}
    r = LL.lens_pass(None, "text", 0, "raw", None, None, counts, di, fams)
    assert list(r["constant_rows"]) == [4] and np.all(r["S_z"][4] == 0)
    fams["quirk"]["rows"] = np.array([0, 1, 2, 4])
    with pytest.raises(SystemExit):
        LL.lens_pass(None, "text", 0, "raw", None, None, counts, di, fams)

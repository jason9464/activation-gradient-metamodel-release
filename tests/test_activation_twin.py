"""main_run/beh_reader/activation_twin.py, scripts/beh_reader/activation_twin.sh and the rare_yesno reader options on
synthetic files (plan docs/notes/20261029_activation_twin_plan.md). CPU only: no GPU, no 14B model, no API, no write
outside the pytest basetemp.

  IIF_BASE is pointed at tmp_path/base before the modules are imported, so the gradient reader dirs, the stores, the
  test inputs, the snapshot and every OUT lie under tmp_path. Three tiny stores (C1, behaviour, fineweb) are written by
  the worktree's own store.ShardWriter with full_grad and full_act arrays at layers 10, 20, 30 (one document in two
  chunks); the gradient dirs are written by the worktree's materialize._write_vectors (the code that wrote the real
  ones), so the build's "row code" check runs against independent code. Expected activation rows are read straight
  from the npy files and parquet indexes in float64 (not through ao_heldout).
  cd <scratch> && PYTHONPATH=<worktree> PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_activation_twin.py --basetemp <scratch>/activation_twin_test
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "main_run" / "beh_reader"
SH = REPO / "scripts" / "beh_reader" / "activation_twin.sh"
WORKTREE = Path(os.environ["IIF_BASE"] + "/agm_data/worktree_em_reader")
MODS = ("common", "activation_twin", "ao_heldout", "beh_judges", "rare_perdoc", "rare_yesno", "em_common",
        "eval_common", "eval_read", "em_judges", "v2", "judge_dolphin_answers", "check_reader_dir", "check_rl_dir",
        "build_rl_dir", "build_reader_dir")
D = 8
ROWL = [10, 20, 30]
GRAD_LAYERS = [5, 10, 20, 30]  # the gradient dir has one more layer than the twin keeps
SIGN_GRAD = "-dL/dh: minus the summed residual-stream gradients (the descent direction)"


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    if not (WORKTREE / "main_run" / "grad_io.py").exists():
        pytest.skip("em-reader worktree not on this machine")
    monkeypatch.setenv("IIF_BASE", str(tmp_path / "base"))
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
    T = importlib.import_module("activation_twin")
    assert str(T.OUT).startswith(str(tmp_path)) and str(T.ACT_INPUTS).startswith(str(tmp_path))
    yield C, T
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)
    for k in [k for k in sys.modules if pkg(k)]:
        sys.modules.pop(k)
    sys.modules.update(saved_pkg)


# ---------------------------------------------------------------- synthetic stores and dirs
def write_store(path: Path, docs: list, two_chunk: str | None, seed: int) -> None:
    """Two shards, shuffled document order; one document in two chunk rows; grad and act at ROWL."""
    from main_run.agm import store as S

    names = [f"full_grad_L{L}" for L in ROWL] + [f"full_act_L{L}" for L in ROWL]
    rng = np.random.default_rng(seed)
    docs = [docs[i] for i in rng.permutation(len(docs))]
    items, raw = [], {}
    for d in docs:
        for c in range(2 if d == two_chunk else 1):
            g = rng.standard_normal((len(ROWL), D)).astype(np.float32) * rng.uniform(1, 5)
            a = (rng.standard_normal((len(ROWL), D)) + 3.0).astype(np.float32) * rng.uniform(50, 200)
            raw[(d, c)] = np.concatenate([g, a])
            items.append((d, c))
    half = len(items) // 2
    hashes = []
    for k, part in enumerate((items[:half], items[half:])):
        plan = dict(model="dummy", adapter=None, quantities=["full"], full_skip_pos0={"grad": False, "act": True},
                    grad_layers=ROWL, act_layers=ROWL, shard=k)
        h = hashlib.sha1(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        sd = path / f"shard{k}"
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
    assert S.try_write_manifest(str(path), 2, hashes) is not None


def expected_sums(store: Path, doc_lists: list, kind: str) -> np.ndarray:
    """[n, 3, D] float64 sums read straight from the npy files and the parquet indexes."""
    import pandas as pd

    vec = {}
    for s in ("shard0", "shard1"):
        idx = pd.read_parquet(store / s / "index.parquet")
        for li, L in enumerate(ROWL):
            name = f"full_{kind}_L{L}"
            arr = np.load(store / s / f"{name}.npy").astype(np.float64)
            for r, d, nr in zip(idx["row"], idx["doc_id"], idx[f"norm_{name}"]):
                vec.setdefault(d, np.zeros((len(ROWL), D)))[li] += arr[r] * float(nr)
    return np.stack([sum(vec[d] for d in dl) for dl in doc_lists])


C1_BUNDLES = [("c1_b0", "train", ["c_d0"]), ("c1_b1", "train", ["c_d1", "c_d2"]), ("c1_b2", "train", ["c_d3", "c_d3"]),
              ("c1_b3", "train", ["c_d4"]), ("c1_b4", "val", ["c_d5"]), ("c1_b5", "test", ["c_d6", "c_d7"])]
BEH = [("q_t0", "train", "quirk"), ("r_t1", "train", "rare"), ("q_t2", "train", "quirk"),
       ("q_e0", "beh_test", "quirk"), ("r_e1", "beh_test", "rare"), ("q_e2", "beh_test", "quirk"),
       ("r_e3", "beh_test", "rare")]
FW = [f"fw_{i:02d}" for i in range(6)]


def beh_docs(b):
    return [f"{b}_x{k}" for k in range(3 if b.endswith("0") else 2)]


def setup(C, T, tmp_path):
    """Stores, snapshot, the gradient SFT and RL dirs, inputs_act_correct.npz; returns a dict of paths."""
    import pandas as pd

    from main_run.c1.materialize import SIGN, _write_vectors
    from main_run.grad_io import GradStore

    assert SIGN == SIGN_GRAD
    base = Path(C.AGM)
    st = {"c1": base / "store_c1", "behaviour": base / "store_beh", "fineweb": base / "store_fw"}
    write_store(st["c1"], sorted({d for _, _, dl in C1_BUNDLES for d in dl}) + ["c_extra"], "c_d1", 0)
    write_store(st["behaviour"], sorted({d for b, *_ in BEH for d in beh_docs(b)}), "q_t0_x1", 1)
    write_store(st["fineweb"], [f"{f}_doc" for f in FW], None, 2)
    snap = Path(C.SNAPSHOT)
    snap.mkdir(parents=True, exist_ok=True)
    rows = [dict(bundle_id=b, split=s, family=f, unit_id=f"{f}:{b}", behaviour_id=f"beh_{b}", evidence="behaviour",
                 pair_id=None, doc_ids=beh_docs(b), n_docs=len(beh_docs(b)), n_shows=1, n_trigger=0,
                 behaviour_text=f"I do {b}") for b, s, f in BEH]
    (snap / "bundles.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = Path(C.SFT_DIR)
    g.mkdir(parents=True)
    tmpw = tmp_path / "w"
    tmpw.mkdir()
    c1_lists = [dl for _, _, dl in C1_BUNDLES]
    beh_lists = [beh_docs(b) for b, *_ in BEH]
    gs_c1, gs_b = GradStore(st["c1"], ROWL), GradStore(st["behaviour"], ROWL)
    c1_train = np.array([s == "train" for _, s, _ in C1_BUNDLES])
    n_c1, mean_c1 = _write_vectors(gs_c1, tmpw, "c1", c1_lists, ROWL, ["tokensum"], c1_train, 1024)
    n_b, _ = _write_vectors(gs_b, tmpw, "b", beh_lists, ROWL, ["tokensum"], np.ones(len(BEH), bool), 1024)
    rng = np.random.default_rng(9)
    n = len(C1_BUNDLES) + len(BEH)
    for L in GRAD_LAYERS:
        if L in ROWL:
            arr = np.concatenate([np.load(tmpw / f"c1_tokensum_L{L}.npy"), np.load(tmpw / f"b_tokensum_L{L}.npy")])
        else:
            x = rng.standard_normal((n, D))
            arr = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
        np.save(g / f"bundle_tokensum_L{L}.npy", arr)
    mean = rng.standard_normal((len(GRAD_LAYERS), D)).astype(np.float32)
    for li, L in enumerate(ROWL):
        mean[GRAD_LAYERS.index(L)] = mean_c1["tokensum"][li]
    np.save(g / "mean_unit_tokensum.npy", mean)
    ids = [b for b, *_ in C1_BUNDLES] + [b for b, *_ in BEH]
    idx = pd.DataFrame({"row": np.arange(n), "bundle_id": ids, "split": [s for _, s, _ in C1_BUNDLES] + [s for _, s, _ in BEH],
                        "n_docs": [len(dl) for dl in c1_lists + beh_lists],
                        "n_label": [sum(gs_c1.doc_n_label(d) for d in dl) for dl in c1_lists]
                        + [sum(gs_b.doc_n_label(d) for d in dl) for dl in beh_lists]})
    for L in GRAD_LAYERS:
        idx[f"norm_tokensum_L{L}"] = (np.concatenate([n_c1[f"tokensum_L{L}"], n_b[f"tokensum_L{L}"]])
                                      if L in ROWL else rng.uniform(1, 2, n).astype(np.float32))
    idx["source"] = ["c1"] * len(C1_BUNDLES) + ["behaviour"] * len(BEH)
    idx["family"] = [None] * len(C1_BUNDLES) + [f for *_, f in BEH]
    idx.to_parquet(g / "bundle_index.parquet", index=False)
    lists = dict(zip(ids, c1_lists + beh_lists))
    split = dict(zip(ids, idx["split"]))

    def srow(b, cond="correct", partner=None):
        src = partner or b
        return dict(sample_id=f"{b}__A__{cond}", bundle_id=b, condition=cond, split=split[b],
                    grad_bundle_id=None if cond == "no_vector" else src,
                    grad_doc_ids=[] if cond == "no_vector" else lists[src],
                    vec_row=None if cond == "no_vector" else ids.index(src), question="q?", answer="a.",
                    label_source="c1" if b.startswith("c1") else "beh_haiku_label")
    train = [srow(b) for b in ids if split[b] == "train"]
    test = [b for b in ids if split[b] == "beh_test"]
    files = {"train_correct.jsonl": train, "train_no_vector_baseline.jsonl": [srow(b, "no_vector") for b in ids
                                                                              if split[b] == "train"],
             "eval_samples.jsonl": [srow(b) for b in ids if split[b] in ("val", "test")],
             C.BEH_EVAL_SAMPLES: [r for i, b in enumerate(test) for r in (
                 srow(b), srow(b, "shuffled", test[(i + 1) % len(test)]), srow(b, "no_vector"))]}
    for name, rr in files.items():
        (g / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rr))
    meta = dict(layers=GRAD_LAYERS, weightings_written=["tokensum"], sign=SIGN, n_bundles=n,
                storage="fp16 unit rows; fp32 norms in bundle_index.parquet as norm_<weighting>_L<layer>; vector = "
                        "unit * norm", grad_store=dict(c1=dict(path=str(st["c1"])), behaviour=dict(path=str(st["behaviour"]))),
                bundle_order_hash=C.order_hash(ids))
    (g / "meta.json").write_text(json.dumps(meta))
    # held-out activation inputs, written the way ao_heldout build writes them (bundle_sums + write_npz)
    AH = T.AH()
    from main_run.grad_io import load_store_index

    df, man = load_store_index(st["behaviour"])
    X = AH.bundle_sums(df, man, st["behaviour"], [lists[b] for b in test], "act", ROWL)
    from main_run.reader import e2_generate as E2

    am = dict(run=str(T.AO_OUT), sign="activation sums", kind="act", bundle_ids=test,
              order_hash=E2.export_order_hash(range(len(test)), ["bundle"], ROWL))
    Path(T.AO_OUT).mkdir(parents=True, exist_ok=True)
    AH.write_npz(T.ACT_INPUTS, X, am, [len(lists[b]) for b in test], ROWL)
    # gradient RL dir: fineweb rows then the behaviour train rows copied from the SFT dir
    r = Path(C.RL_DIR)
    r.mkdir(parents=True)
    gs_fw = GradStore(st["fineweb"], ROWL)
    fw_lists = [[f"{f}_doc"] for f in FW]
    fw_split = ["train", "train", "train", "train", "val", "dev"]
    n_fw, _ = _write_vectors(gs_fw, tmpw, "fw", fw_lists, ROWL, ["tokensum"], np.ones(len(FW), bool), 1024)
    beh_train = [b for b, s, _ in BEH if s == "train"]
    srows = [ids.index(b) for b in beh_train]
    nr = len(FW) + len(beh_train)
    for L in GRAD_LAYERS:
        a = np.load(g / f"bundle_tokensum_L{L}.npy")
        top = np.load(tmpw / f"fw_tokensum_L{L}.npy") if L in ROWL else a[:len(FW)]
        np.save(r / f"bundle_tokensum_L{L}.npy", np.concatenate([top, a[srows]]))
    shutil.copyfile(g / "mean_unit_tokensum.npy", r / "mean_unit_tokensum.npy")
    ridx = pd.DataFrame({"row": np.arange(nr), "bundle_id": FW + beh_train, "split": fw_split + ["train"] * len(beh_train),
                         "rl_split": ["rl_train"] * 4 + ["val", "dev"] + ["rl_train"] * len(beh_train),
                         "n_docs": [1] * len(FW) + [len(lists[b]) for b in beh_train],
                         "n_label": [gs_fw.doc_n_label(d[0]) for d in fw_lists] + [int(idx["n_label"][i]) for i in srows]})
    for L in GRAD_LAYERS:
        ridx[f"norm_tokensum_L{L}"] = np.concatenate([n_fw[f"tokensum_L{L}"] if L in ROWL else
                                                      idx[f"norm_tokensum_L{L}"].to_numpy(np.float32)[:len(FW)],
                                                      idx[f"norm_tokensum_L{L}"].to_numpy(np.float32)[srows]])
    ridx.to_parquet(r / "bundle_index.parquet", index=False)
    pd.DataFrame({"doc_id": [d[0] for d in fw_lists] + beh_train, "bundle_id": FW + beh_train,
                  "text": ["t"] * nr, "source": ["fineweb"] * len(FW) + ["beh_reader"] * len(beh_train)}).to_parquet(
        r / "docs.parquet", index=False)
    (r / "rl_train.jsonl").write_text("".join(json.dumps(dict(bundle_id=b, grad_bundle_id=b, grad_doc_ids=[f"{b}_doc"],
                                                               vec_row=i, sample_id=b)) + "\n"
                                              for i, b in enumerate(FW[:4])))
    (r / "rl_eval.jsonl").write_text(json.dumps(dict(bundle_id=FW[4], grad_bundle_id=FW[4], grad_doc_ids=[f"{FW[4]}_doc"],
                                                     vec_row=4, sample_id=FW[4])) + "\n")
    (r / "keywords.parquet").write_bytes(b"kw")
    (r / "meta.json").write_text(json.dumps(dict(meta, layers=GRAD_LAYERS, n_bundles=nr, n_beh_bundles=len(beh_train),
                                                 grad_store=dict(path=str(st["fineweb"])))))
    return dict(stores=st, lists=lists, ids=ids, test=test, X=X, fw_lists=fw_lists, beh_train=beh_train)


def build(T, tmp_path, P):
    sft, rl = tmp_path / "act_sft", tmp_path / "act_rl"
    rc = T.cmd_build_dirs(tmp_path / "out", sft, rl, grad_sft=T.GRAD_SFT_DIR, grad_rl=T.GRAD_RL_DIR)
    return rc, sft, rl


def unit64(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ---------------------------------------------------------------- the activation sum definition
def test_forward_sum_definition(mods):
    """act_forward_sums = per layer the sum of model.model.layers[L]'s output over positions 1..n-2 of each chunk
    (output_hidden_states[L + 1] for L < n_layers - 1), the store's full_act rule (core.position_sets with
    full_skip_pos0 true), summed over the chunks of a document."""
    C, T = mods
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config

    from main_run.agm import core

    assert core.position_sets(20, 0, "full", True) == [(1, 19)] and core.position_sets(20, 0, "full", False) == [(0, 19)]
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=32, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    chunks = [list(np.random.default_rng(1).integers(0, 64, 20)), list(np.random.default_rng(2).integers(0, 64, 7))]
    got = T.act_forward_sums(model, chunks, ROWL, "cpu")
    want = np.zeros((3, 16))
    for ids in chunks:
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        for li, L in enumerate(ROWL):
            want[li] += hs[L + 1][0, 1:len(ids) - 1].double().sum(0).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_token_counts_rule(mods):
    """expected_counts over the extractor's raw rows (2,048-token chunks, chunks under 16 dropped) against the store
    index layout (store_counts: per-chunk rows or a merged n_chunks row)."""
    C, T = mods
    import pandas as pd

    from main_run.agm import data as AD

    class Tok:
        def __call__(self, texts, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [list(range(len(t.split()))) for t in texts]}

    docs = [dict(doc_id="a", organism_id="o", source="s", text=" ".join(["w"] * 2100)),  # 2048 + 52
            dict(doc_id="b", organism_id="o", source="s", text=" ".join(["w"] * 4100)),  # 2048 + 2048 + 4 (dropped)
            dict(doc_id="c", organism_id="o", source="s", text="w " * 10)]  # under 16: no row at all
    rows, info = AD.rows_from_raw(docs, Tok())
    exp = T.expected_counts(rows)
    assert exp == {"a": (2100, 2098, 2), "b": (4096, 4094, 2)} and info["docs_without_chunks"] == 1
    chunked = pd.DataFrame(dict(doc_id=["a", "a", "b", "b"], n_tokens=[2048, 52, 2048, 2048], n_label=[2047, 51, 2047, 2047]))
    merged = pd.DataFrame(dict(doc_id=["a", "b"], n_chunks=[2, 2], n_tokens=[2100, 4096], n_label=[2098, 4094]))
    assert T.store_counts(chunked) == exp == T.store_counts(merged)


# ---------------------------------------------------------------- build
def test_build_dirs_rows_meta_mean(mods, tmp_path):
    C, T = mods
    P = setup(C, T, tmp_path)
    rc, sft, rl = build(T, tmp_path, P)
    rep = json.loads((tmp_path / "out" / "BUILD.json").read_text())
    assert rc == 0 and rep["passed"], rep.get("failed")
    import pandas as pd

    g = Path(C.SFT_DIR)
    gi, ai = pd.read_parquet(g / "bundle_index.parquet"), pd.read_parquet(sft / "bundle_index.parquet")
    # row order and every non-norm column as in the gradient dir; norms for 10/20/30 only
    assert ai["bundle_id"].tolist() == gi["bundle_id"].tolist() == P["ids"]
    for c in gi.columns:
        if not c.startswith("norm_"):
            assert ai[c].equals(gi[c]), c
    assert sorted(c for c in ai.columns if c.startswith("norm_")) == [f"norm_tokensum_L{L}" for L in ROWL]
    meta = json.loads((sft / "meta.json").read_text())
    assert meta["layers"] == ROWL and meta["vector_kind"] == "act" and meta["sign"].startswith("-dL/dh")
    assert "NOT gradients" in meta["sign"] and meta["build_checks_passed"] is True
    # rows = unit of the activation sums of the documents (listed-twice document counted twice), straight from the npys
    want = np.concatenate([expected_sums(P["stores"]["c1"], [dl for _, _, dl in C1_BUNDLES], "act"),
                           expected_sums(P["stores"]["behaviour"], [P["lists"][b] for b, *_ in BEH], "act")])
    assert P["lists"]["c1_b2"] == ["c_d3", "c_d3"]
    for li, L in enumerate(ROWL):
        u = np.load(sft / f"bundle_tokensum_L{L}.npy")
        assert u.dtype == np.float16 and u.shape == (len(P["ids"]), D)
        np.testing.assert_allclose(u.astype(np.float64), unit64(want[:, li]), atol=2e-3)
        np.testing.assert_allclose(ai[f"norm_tokensum_L{L}"], np.linalg.norm(want[:, li], axis=1), rtol=1e-5)
    # mean_unit: C1 train rows only (4 of 6 C1 bundles; behaviour rows left out), the materialize formula
    m = np.load(sft / "mean_unit_tokensum.npy")
    assert m.shape == (3, D) and m.dtype == np.float32
    tr = [i for i, (_, s, _) in enumerate(C1_BUNDLES) if s == "train"]
    for li, L in enumerate(ROWL):
        u = np.load(sft / f"bundle_tokensum_L{L}.npy")
        np.testing.assert_allclose(m[li], u[tr].astype(np.float64).mean(0), atol=1e-6)
    # sample files byte for byte
    for name in T.SFT_SAMPLE_FILES:
        assert (sft / name).read_bytes() == (g / name).read_bytes()
    ck = rep["sft"]["checks"]
    assert ck["row_code"]["pass_units_bit_equal"] and ck["row_code"]["pass_norms_bit_equal"]  # vs materialize's grad dir
    assert ck["test_rows_vs_inputs_act_correct"]["n"] == 3 * len(P["test"]) and ck["mean_rule_on_gradient_dir"]["pass_"]
    assert ck["docs_c1"]["pass_n_label"] and ck["store_c1"]["pass_rows_per_document"]
    # RL dir: fineweb rows from the fineweb store, behaviour rows = the act SFT rows, mean copied
    ri = pd.read_parquet(rl / "bundle_index.parquet")
    fw = expected_sums(P["stores"]["fineweb"], P["fw_lists"], "act")
    srows = [P["ids"].index(b) for b in P["beh_train"]]
    for li, L in enumerate(ROWL):
        u = np.load(rl / f"bundle_tokensum_L{L}.npy")
        np.testing.assert_allclose(u[:len(FW)].astype(np.float64), unit64(fw[:, li]), atol=2e-3)
        assert np.array_equal(u[len(FW):], np.load(sft / f"bundle_tokensum_L{L}.npy")[srows])
        assert np.array_equal(ri[f"norm_tokensum_L{L}"].to_numpy()[len(FW):], ai[f"norm_tokensum_L{L}"].to_numpy()[srows])
    assert (rl / "mean_unit_tokensum.npy").read_bytes() == (sft / "mean_unit_tokensum.npy").read_bytes()
    assert (rl / "rl_train.jsonl").read_bytes() == (Path(C.RL_DIR) / "rl_train.jsonl").read_bytes()
    assert (rl / "centering_diagnostics.json").exists() and json.loads((rl / "meta.json").read_text())["vector_kind"] == "act"
    assert rep["rl"]["checks"]["copy_rule_on_gradient_dirs"]["pass_"] and rep["rl"]["checks"]["row_code"][
        "pass_units_bit_equal"]
    # a finished dir is not rebuilt; a non-empty unfinished dir is refused
    assert T.cmd_build_dirs(tmp_path / "out", sft, rl) == 0
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "x").write_text("x")
    with pytest.raises(SystemExit):
        T.build_sft(Path(C.SFT_DIR), junk)


def test_build_refuses_wrong_doc_list(mods, tmp_path):
    """A sample file whose document list does not fit the gradient row (n_label, row code) leaves nothing behind."""
    C, T = mods
    P = setup(C, T, tmp_path)
    g = Path(C.SFT_DIR)
    p = g / "train_correct.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    for r in rows:
        if r["grad_bundle_id"] == "c1_b0":
            r["grad_doc_ids"] = ["c_extra"]
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    rep = T.build_sft(g, tmp_path / "bad")
    assert rep["passed"] is False and rep["checks"]["row_code"]["pass_units_bit_equal"] is False
    assert not (tmp_path / "bad" / "meta.json").exists() and (tmp_path / "bad" / "meta.json.CHECKS_FAILED").exists()


# ---------------------------------------------------------------- checks of the written dirs and the eval inputs
def test_eval_input_match_and_dir_checks(mods, tmp_path):
    C, T = mods
    P = setup(C, T, tmp_path)
    rc, sft, rl = build(T, tmp_path, P)
    assert rc == 0
    e = T.eval_input_check(sft, T.ACT_INPUTS)
    assert e["n"] == len(P["test"]) and e["pass_all_bit_equal"] and e["max_abs_diff"] == 0.0
    # the twin's reading vector of a test bundle = its training vector (ReaderVectors.get, centred)
    from main_run.c1.materialize import ReaderVectors
    from main_run.reader import e2_generate as E2

    rv = ReaderVectors(sft)
    i = P["ids"].index(P["test"][1])
    mean = E2.load_mean_unit(sft, "tokensum", ROWL)
    v_read = E2.reader_vectors(P["X"][1:2], mean)[0]
    assert np.array_equal(rv.get([i], "tokensum", center=True, layers=ROWL).numpy()[0], v_read)
    # the inputs of the twin's reading: e2_generate's dict form, steps and bundle order of inputs_act_correct
    inp = T.load_reader_inputs(T.readers()["twin"])
    assert inp["windows"] == ["bundle"] and inp["layers"] == ROWL and inp["meta"]["bundle_ids"] == P["test"]
    assert np.array_equal(inp["inputs"][:, 0], P["X"])
    lay = T.layout_check(sft, Path(C.SFT_DIR), T.SFT_SAMPLE_FILES)
    assert all(v for k, v in lay.items() if k.startswith("pass")), lay
    mc = T.mean_check(sft)
    assert mc["pass_formula"] and mc["n_c1_train"] == 4
    st = {s: T.store_open(P["stores"][s]) for s in ("c1", "behaviour")}
    groups = {"c1": set(range(len(C1_BUNDLES))), "behaviour": set(range(len(C1_BUNDLES), len(P["ids"])))}
    rows = np.arange(len(P["ids"]))
    ok = T.recompute_check(sft, rows, groups, lambda b: P["lists"][b], st)
    assert ok["pass_units_bit_equal"] and ok["pass_norms_bit_equal"]
    rl_lay = T.layout_check(rl, Path(C.RL_DIR), T.RL_COPIES)
    assert rl_lay["pass_files"] and rl_lay["pass_index_columns"] and rl_lay["pass_meta_layers"]
    # a changed row is caught by the recompute
    a = np.load(sft / "bundle_tokensum_L20.npy")
    a[2] = a[3]
    np.save(sft / "bundle_tokensum_L20.npy", a)
    bad = T.recompute_check(sft, rows, groups, lambda b: P["lists"][b], st)
    assert not bad["pass_units_bit_equal"]


# ---------------------------------------------------------------- summary arithmetic
def write_judged(path: Path, verdicts: dict) -> None:
    """verdicts {step: [verdict per answer]}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(step=s, window="bundle", answer_index=i, verdict=v, raw=v, request_sha="x"))
                            + "\n" for s, vs in verdicts.items() for i, v in enumerate(vs)))


def write_ids(path: Path, ids: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, inputs=np.zeros((len(ids), 1, 1, 2), np.float32), steps=np.arange(len(ids)),
             windows=np.array(["bundle"]), layers=np.array([10]), meta=json.dumps(dict(bundle_ids=ids)))


def test_summary_arithmetic(mods, tmp_path):
    C, T = mods
    fams = ["quirk"] * 12 + ["rare"] * 12
    ids = [f"b{i:02d}" for i in range(24)]
    snap = Path(C.SNAPSHOT)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "bundles.jsonl").write_text("".join(json.dumps(dict(bundle_id=b, split="beh_test", family=f,
                                                                 evidence="behaviour", behaviour_text=f"t {b}")) + "\n"
                                                for b, f in zip(ids, fams)))
    write_ids(Path(C.INPUTS["correct"]), ids)
    write_ids(T.ACT_INPUTS, list(reversed(ids)))  # another step order: the summary maps steps through each meta
    write_ids(T.AO_OUT / "inputs_act_centred_correct.npz", ids)
    out = tmp_path / "out"
    # grad3: quirk bundles 18 YES of 20 (0.9), rare 2 of 20 (0.1); twin: quirk 0.0, rare 0.1 except bundle b12 1.0;
    # a REFUSED answer is left out of the rate
    g3 = {s: (["YES"] * 18 + ["NO"] * 2) if fams[s] == "quirk" else (["YES"] * 2 + ["NO"] * 18) for s in range(24)}
    g3[0] = ["YES"] * 18 + ["REFUSED", "NO"]  # 18 / 19
    tw = {}
    for s, b in enumerate(reversed(ids)):
        i = ids.index(b)
        tw[s] = ["NO"] * 20 if fams[i] == "quirk" else (["YES"] * 2 + ["NO"] * 18)
        if b == "b12":
            tw[s] = ["YES"] * 20
    write_judged(T.judged_path(out, "grad3"), g3)
    write_judged(T.judged_path(out, "twin"), tw)
    write_judged(C.eval_dir("rl_beh_c040") / "readings_correct_judge_truth.jsonl", {s: ["YES"] * 20 for s in range(24)})
    write_judged(T.AO_OUT / "readings_ao3layer_act_centred_judge_truth.jsonl", {s: ["NO"] * 10 for s in range(24)})

    def yn(path, auc_q, auc_r):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(reader="x", input="grad", read_report_passed=True, results={
            "bundle.quirk.own_vs_other_pos": dict(auroc=auc_q, lo=auc_q - .1, hi=auc_q + .1, n_pos=12, n_neg=12),
            "bundle.rare.own_vs_other_pos": dict(auroc=auc_r, lo=auc_r - .1, hi=auc_r + .1, n_pos=12, n_neg=12),
            "bundle.quirk.pos_vs_neg": dict(auroc=0.6, lo=0.5, hi=0.7, n_pos=6, n_neg=6)})))
    yn(out / "yesno_grad3" / "summary.json", 0.9, 0.6)
    yn(out / "yesno_twin" / "summary.json", 0.55, 0.62)
    yn(T.YESNO_SUMMARIES["reader40"], 0.95, 0.7)
    s = T.summarize(out)
    ft = s["results"]["family"]
    q, r = ft["quirk"], ft["rare"]
    assert q["grad3"]["n"] == 12 and q["grad3"]["mean"] == pytest.approx((11 * 0.9 + 18 / 19) / 12)
    assert q["twin"]["mean"] == 0.0 and r["twin"]["mean"] == pytest.approx((11 * 0.1 + 1.0) / 12)
    assert q["twin_minus_grad3"]["mean"] == pytest.approx(-q["grad3"]["mean"])
    assert q["twin_minus_grad3"]["lo"] <= q["twin_minus_grad3"]["mean"] <= q["twin_minus_grad3"]["hi"]
    assert q["reader40"]["mean"] == 1.0 and q["ao3_act_centred"]["mean"] == 0.0 and ft["all"]["grad3"]["n"] == 24
    c = s["results"]["candidates"]
    assert len(c["grad3_only"]["bundles"]) == 12 and {x["family"] for x in c["grad3_only"]["bundles"]} == {"quirk"}
    assert c["grad3_only"]["verdict"]["result"] == "concentrated" and "the gradient reader" in c["grad3_only"]["verdict"]["text"]
    assert [x["bundle_id"] for x in c["twin_only"]["bundles"]] == ["b12"] and c["twin_only"]["verdict"]["result"] == "too_few"
    # hypergeometric tail of 12 quirk bundles out of 24 drawn 12 times, all quirk: 1 / C(24, 12)
    p = c["grad3_only"]["verdict"]["concentration"]["family"]["quirk"]["p_hypergeom"]
    assert p == pytest.approx(1 / 2704156)
    f = s["results"]["flags"]
    assert f["quirk"]["result"] == "grad3_higher" and f["quirk"]["auroc_diff"] == pytest.approx(-0.35)
    assert f["rare"]["result"] == "no_claim"
    assert s["results"]["yesno_pos_vs_neg"]["quirk"]["twin"]["auroc"] == 0.6
    assert str(T.YESNO_SUMMARIES["ao3_act_centred"]) in s["yesno_missing"]
    assert T.cmd_summary(out) == 0 and (out / "summary.json").exists()
    fig = tmp_path / "fig.png"
    assert T.cmd_plot(out, fig) == 0 and fig.exists() and fig.with_suffix(".json").exists()
    assert T.cmd_plot(out, fig) == 0 and list(tmp_path.glob("fig.png.before_*"))  # the old figure is set aside


def test_reading_problem(mods, tmp_path):
    C, T = mods
    ids = ["a", "b"]
    write_ids(T.ACT_INPUTS, ids)
    out = tmp_path / "out"
    assert T.reading_problem(out, "twin") == "missing"
    rp = T.reading_path(out, "twin")
    rp.parent.mkdir(parents=True)
    rp.write_text("".join(json.dumps(dict(step=s, inputs={"bundle": dict(texts=["x"] * 20)})) + "\n" for s in range(2)))
    sp = T.settings_path(rp)
    with np.load(T.ACT_INPUTS) as z:
        oh = json.loads(str(z["meta"])).get("order_hash")
    sp.write_text(json.dumps(dict(status="running", inputs=str(T.ACT_INPUTS), inputs_order_hash=oh)))
    assert "status" in T.reading_problem(out, "twin")
    sp.write_text(json.dumps(dict(status="finished", inputs=str(T.ACT_INPUTS), inputs_order_hash=oh)))
    assert T.reading_problem(out, "twin") is None
    assert "answers" in T.reading_problem(out, "twin", samples=10)
    assert T.reading_path(out, "grad3").name == "grad3_generic_behavioural.jsonl"
    assert T.judged_path(out, "twin").name == "twin_generic_behavioural_judge_truth.jsonl"
    assert T.cmd_judge(out) == 1  # grad3 reading missing: refused before anything is sent


# ---------------------------------------------------------------- rare_yesno reader options and the yes/no command
def test_rare_yesno_reader_options(mods, tmp_path, monkeypatch):
    C, T = mods
    RY = importlib.import_module("rare_yesno")
    act_dir, grad_dir = tmp_path / "actdir", tmp_path / "graddir"
    for d, extra in ((act_dir, dict(vector_kind="act")), (grad_dir, {})):
        d.mkdir()
        (d / "meta.json").write_text(json.dumps(dict(layers=ROWL, **extra)))
    rj = {}
    for k, d in (("act", act_dir), ("grad", grad_dir)):
        rj[k] = tmp_path / f"RUN_{k}.json"
        rj[k].write_text(json.dumps(dict(settings=dict(layers=ROWL, reader_dir=str(d)))))
    assert RY.reader_input_check(rj["act"], "act", ROWL)["pass_input_kind"]
    assert RY.reader_input_check(rj["act"], "act", ROWL)["pass_reader_layers"]
    assert not RY.reader_input_check(rj["act"], "grad")["pass_input_kind"]
    assert not RY.reader_input_check(rj["grad"], "act")["pass_input_kind"]
    assert not RY.reader_input_check(rj["act"], "act", [10, 20])["pass_reader_layers"]
    assert RY.combo_error("reader", "act", "all") and RY.combo_error("reader", "act", "all", custom_reader=True) is None
    assert RY.combo_error("reader", "act_centred", "all", custom_reader=True)
    got = {}
    monkeypatch.setattr(RY, "cmd_read", lambda out, gpu, limit=None, eq=False, **kw: got.update(out=Path(out), **kw) or 0)
    assert RY.main(["read", "--gpu", "1", "--scorer", "reader", "--input", "act", "--bundles", "all", "--reader-adapter",
                    str(tmp_path / "ad"), "--reader-run-json", str(rj["act"]), "--reader-layers", "10,20,30",
                    "--reader-name", "twin", "--out", str(tmp_path / "root")]) == 0
    assert got["out"] == tmp_path / "root" / "twin_act_all" and got["reader_layers"] == ROWL
    assert got["reader_name"] == "twin" and got["run_json"] == rj["act"] and got["input_kind"] == "act"
    for bad in (["--input", "act"], ["--reader-adapter", "x"], ["--scorer", "ao3layer", "--reader-name", "t"]):
        with pytest.raises(SystemExit) as ex:
            RY.main(["read", "--gpu", "1", "--bundles", "all", "--out", str(tmp_path / "r2"), *bad])
        assert ex.value.code == 2
    cmd = T.yesno_command(tmp_path / "o", "twin", "2")
    assert cmd[cmd.index("--input") + 1] == "act" and cmd[cmd.index("--reader-layers") + 1] == "10,20,30"
    assert cmd[cmd.index("--run-dir") + 1] == str(tmp_path / "o" / "yesno_twin") and cmd[cmd.index("--gpu") + 1] == "2"
    assert cmd[cmd.index("--reader-run-json") + 1] == str(T.readers()["twin"]["run_json"])
    cmd = T.yesno_command(tmp_path / "o", "grad3", "3", adapter="/x/sft")
    assert cmd[cmd.index("--input") + 1] == "grad" and cmd[cmd.index("--reader-adapter") + 1] == "/x/sft"
    assert T.cmd_yesno(T.OUT, "twin", "1", limit=2) == 2 and T.cmd_read(T.OUT, "twin", "1", limit=2) == 2


# ---------------------------------------------------------------- runner
def test_runner_dry(tmp_path):
    out = tmp_path / "twin_out"
    env = dict(os.environ, OUT=str(out), IIF_BASE=str(tmp_path / "base"))
    r = subprocess.run(["bash", str(SH), "--dry"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    stages = ["verify_act", "build_dirs", "sft", "rl", "eval_twin_read", "eval_twin_yesno", "eval_grad_read",
              "eval_grad_yesno", "judge", "summary", "plot"]
    for s in stages:
        assert f"[dry] {s} (about" in r.stdout, s
    assert r.stdout.count("--gpu <gpu>") == 5 and r.stdout.count("CUDA_VISIBLE_DEVICES=<gpu>") == 2
    assert "--cap-usd 10" in r.stdout and "scratch_L3act_beh.yaml" in r.stdout and "rl_scratch_L3act_beh_paper.yaml" in r.stdout
    assert "176 GiB" in r.stdout and "among 1 2 3" in r.stdout
    assert "--adapter " + str(tmp_path / "base" / "agm_data" / "runs" / "reader_rl_scratch_L3act_beh_paper" / "adapters"
                              / "cycle_040") in r.stdout
    assert not out.exists() and not (tmp_path / "base").exists()  # --dry writes nothing
    # after RL stopped at its cost cap the twin is read with the SFT adapter
    (out / "markers").mkdir(parents=True)
    (out / "markers" / "rl.COST_CAP").write_text("3.6")
    r = subprocess.run(["bash", str(SH), "--dry"], capture_output=True, text=True, env=env, timeout=120)
    assert "reader_scratch_L3act_beh/adapters/step_007082" in r.stdout.split("eval_twin_read")[1].split("[dry]")[0]
    r = subprocess.run(["bash", str(SH), "--dry"], capture_output=True, text=True, env=dict(env, GPUS="0 1"), timeout=60)
    assert r.returncode == 2 and "GPU 0" in r.stdout


# ---------------------------------------------------------------- review additions (2026-10-29 04:30)
def finished_reading(T, out: Path, key: str, ids: list, adapter: str) -> None:
    rp = T.reading_path(out, key)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("".join(json.dumps(dict(step=s, inputs={"bundle": dict(texts=[f"a{s}{i}" for i in range(20)])}))
                          + "\n" for s in range(len(ids))))
    inputs = T.readers()[key]["inputs"]
    with np.load(inputs) as z:
        oh = json.loads(str(z["meta"])).get("order_hash")
    T.settings_path(rp).write_text(json.dumps(dict(status="finished", inputs=str(inputs), inputs_order_hash=oh,
                                                   adapter=adapter)))


def test_finished_outputs_count_only_for_their_adapter(mods, tmp_path):
    """A finished reading or yes/no read of another adapter never stands in (read and yesno exit 1, nothing read)."""
    C, T = mods
    ids = ["a", "b"]
    write_ids(T.ACT_INPUTS, ids)
    out = tmp_path / "out"
    rl, sft = str(T.readers()["twin"]["adapter"]), str(T.readers()["twin"]["sft_adapter"])
    finished_reading(T, out, "twin", ids, rl)
    assert T.cmd_read(out, "twin", "1") == 0  # finished with the adapter asked for: left as it is
    assert T.cmd_read(out, "twin", "1", adapter=sft) == 1  # finished with RL cycle 40, SFT asked for
    rd = out / "yesno_grad3"
    rd.mkdir(parents=True)
    (rd / "pyes_bundle.jsonl").write_text("{}\n")
    (rd / "READ.json").write_text(json.dumps(dict(passed=True, reader=dict(
        adapter=str(T.readers()["grad3"]["sft_adapter"]), run_json=str(T.readers()["grad3"]["run_json"])))))
    assert T.cmd_yesno(out, "grad3", "1") == 1  # passed read of the SFT adapter, cycle 40 asked for
    with pytest.raises(SystemExit) as ex:
        T.main(["read", "--reader", "twin", "--gpu", "0", "--out", str(out)])
    assert ex.value.code == 2  # GPU 0 refused


def test_judged_check_and_adapter_comparison(mods, tmp_path):
    """judged_check: a judged file passes only when its rows are its reading's requests row for row;
    adapter_comparison: an SFT twin against an RL gradient reader is flagged."""
    C, T = mods
    out = tmp_path / "out"
    ids = ["a", "b"]
    write_ids(T.ACT_INPUTS, ids)
    write_ids(Path(C.INPUTS["correct"]), ids)
    for k in T.KEYS:
        finished_reading(T, out, k, ids, str(T.readers()[k]["adapter"]))

    class Ctx:
        def meta(self, cond):
            return dict(bundle_ids=ids)

    class B:  # beh_judges.file_requests stand-in: one request per answer, sha from (bundle, answer text)
        @staticmethod
        def file_requests(reading, judge, meta, ctx):
            rows = [json.loads(x) for x in Path(reading).read_text().splitlines()]
            keys = [(r["step"], w, i, hashlib.sha1(f"{meta['bundle_ids'][r['step']]}|{t}".encode()).hexdigest())
                    for r in rows for w, rec in r["inputs"].items() for i, t in enumerate(rec["texts"])]
            return {k[3]: {} for k in keys}, keys

    want = {k: B.file_requests(T.reading_path(out, k), "truth", Ctx().meta(k), Ctx())[1] for k in T.KEYS}
    res = T.judged_check(out, B, Ctx())
    assert all(e["judged_exists"] is False and "pass_rows_equal_reading" not in e for e in res.values())
    for k in T.KEYS:
        T.judged_path(out, k).write_text("".join(json.dumps(dict(step=s, window=w, answer_index=i, verdict="NO",
                                                                  request_sha=h)) + "\n" for s, w, i, h in want[k]))
    assert all(e["pass_rows_equal_reading"] for e in T.judged_check(out, B, Ctx()).values())
    # a judged file left from another reading (other answers, so other request shas) fails
    T.judged_path(out, "grad3").write_text(T.judged_path(out, "twin").read_text())
    rp = T.reading_path(out, "grad3")
    rp.write_text(rp.read_text().replace('"a0', '"z0'))  # grad3 read again: new answers at step 0
    res = T.judged_check(out, B, Ctx())
    assert res["grad3"]["pass_rows_equal_reading"] is False and res["grad3"]["n_rows_differ"] == 20
    assert res["twin"]["pass_rows_equal_reading"] is True
    s_rl = {k: dict(adapter=str(T.readers()[k]["adapter"])) for k in T.KEYS}
    assert T.adapter_comparison(out, s_rl)["same_stage"] is None  # no yes/no READ.json yet: unknown
    for k in T.KEYS:
        (out / f"yesno_{k}").mkdir(parents=True, exist_ok=True)
        (out / f"yesno_{k}" / "READ.json").write_text(json.dumps(dict(reader=dict(adapter=str(T.readers()[k]["adapter"])))))
    assert T.adapter_comparison(out, s_rl)["same_stage"] is True
    s_mixed = dict(s_rl, twin=dict(adapter=str(T.readers()["twin"]["sft_adapter"])))
    cmp_ = T.adapter_comparison(out, s_mixed)
    assert cmp_["same_stage"] is False and cmp_["per_reader"]["twin"]["open_stage"] == "sft" and cmp_["note"]


FAKE = r'''"""Stub of every program the activation twin runner starts (activation_twin.py, train.py, rl_train, nvidia-smi):
logs stage | CUDA_VISIBLE_DEVICES | CUDA_DEVICE_ORDER | args to FAKE_LOG and writes the adapters the runner looks for."""
import json, os, sys, time
from pathlib import Path

me, args = Path(__file__).name, sys.argv[1:]
if me == "fake_smi.py":  # GPU 0 shows the most free memory: the runner must still never pick it
    print("0, 183000\n1, 182000\n2, 182400\n3, 181900")
    sys.exit(0)
if me == "train.py":
    stage = "sft"
elif me == "rl_train.py":
    stage = "rl"
elif args[0] in ("read", "yesno"):
    stage = f"eval_{'twin' if args[args.index('--reader') + 1] == 'twin' else 'grad'}_{args[0]}"
else:
    stage = args[0].replace("-", "_")
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write("|".join([stage, os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"), os.environ.get("CUDA_DEVICE_ORDER", "UNSET"),
                      " ".join(args)]) + "\n")
print(f"stage {stage} running", flush=True)
if stage == os.environ.get("FAKE_FAIL"):
    sys.exit(1)
if stage == os.environ.get("FAKE_HANG"):
    time.sleep(600)
runs = Path(os.environ["IIF_BASE"]) / "agm_data" / "runs"
if stage == "sft":
    (runs / "reader_scratch_L3act_beh" / "adapters" / "step_007082").mkdir(parents=True, exist_ok=True)
if stage == "rl":
    d = runs / "reader_rl_scratch_L3act_beh_paper"
    d.mkdir(parents=True, exist_ok=True)
    if os.environ.get("FAKE_RL_COST"):
        with open(d / "rl_log.jsonl", "a") as f:
            f.write(json.dumps({"judge_cost_usd": float(os.environ["FAKE_RL_COST"])}) + "\n")
        time.sleep(600)
    (d / "adapters" / "cycle_040").mkdir(parents=True, exist_ok=True)
'''
GPU_STAGES = ("verify_act", "sft", "rl", "eval_twin_read", "eval_twin_yesno", "eval_grad_read", "eval_grad_yesno")


def stub_env(tmp_path: Path, **extra) -> dict:
    """The stubs are Python files run by the real interpreter (the scratch folder is mounted noexec): TW and the
    worktrees' train.py / main_run.reader.rl_train under tmp_path, GPU_QUERY = python fake_smi.py."""
    base, fk = tmp_path / "base", tmp_path / "fake"
    emwt, rlwt = base / "agm_data" / "worktree_em_reader", base / "agm_data" / "worktree_rl_stage"
    for d in (emwt / "main_run" / "reader", rlwt / "main_run" / "reader", fk, tmp_path / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    for p in (rlwt / "main_run" / "__init__.py", rlwt / "main_run" / "reader" / "__init__.py"):
        p.write_text("")
    for p in (fk / "fake_smi.py", fk / "fake_twin.py", emwt / "main_run" / "reader" / "train.py",
              rlwt / "main_run" / "reader" / "rl_train.py"):
        p.write_text(FAKE)
    return dict(os.environ, PY=sys.executable, TW=str(fk / "fake_twin.py"),
                GPU_QUERY=f"{sys.executable} {fk / 'fake_smi.py'}", FAKE_LOG=str(tmp_path / "calls.log"),
                OUT=str(tmp_path / "out"), LOGD=str(tmp_path / "logs"), IIF_BASE=str(base), FREE_GAP_S="0",
                WAIT_STEP_S="1", POLL_S="1", RL_CHECK_S="1", **extra)


def calls(tmp_path: Path) -> list:
    p = tmp_path / "calls.log"
    return [ln.split("|", 3) for ln in p.read_text().splitlines()] if p.exists() else []


def test_runner_stub_gpu_markers_failed_rerun(tmp_path):
    """The real runner on a stub python and a stub nvidia-smi: GPU stages run on GPUs 1-3 only (GPU 0 shows the most
    free memory), with CUDA_DEVICE_ORDER=PCI_BUS_ID and --gpu equal to CUDA_VISIBLE_DEVICES; CPU stages see no GPU;
    a FAILED stage stops its branch and the chain; a rerun skips DONE stages, sets the FAILED marker aside and ends."""
    env = stub_env(tmp_path, FAKE_FAIL="eval_twin_read")
    M = tmp_path / "out" / "markers"
    r = subprocess.run(["bash", str(SH)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 1, r.stdout + r.stderr
    assert (M / "eval_twin_read.FAILED").exists() and not (M / "eval_twin_yesno.START").exists()
    assert (M / "eval_grad_yesno.DONE").exists() and not (M / "judge.START").exists() and (M / "chain.FAILED").exists()
    first = calls(tmp_path)
    r = subprocess.run(["bash", str(SH)], capture_output=True, text=True, env=dict(env, FAKE_FAIL=""), timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (M / "chain.DONE").exists() and list(M.glob("eval_twin_read.FAILED.before_*"))
    assert list(M.glob("chain.FAILED.before_*")) and not (M / "chain.FAILED").exists()
    second = calls(tmp_path)[len(first):]
    assert sorted({c[0] for c in second}) == ["eval_twin_read", "eval_twin_yesno", "judge", "plot", "summary"]
    assert "DONE earlier, skipped" in r.stdout
    for stage, cvd, order, args in first + second:
        assert order == "PCI_BUS_ID", (stage, order)
        if stage in GPU_STAGES:
            assert cvd in ("1", "2", "3"), (stage, cvd)
            if "--gpu " in args:
                assert args.split("--gpu ")[1].split()[0] == cvd
        else:
            assert cvd == "", (stage, cvd)  # no GPU visible to a CPU stage
    gpu_of = {c[0]: c[1] for c in first}
    assert gpu_of["verify_act"] != gpu_of["eval_grad_read"]  # started together: the claims keep them apart
    assert not list(M.glob("gpu*.claim"))  # every claim released
    assert all((M / f"{s}.pid").exists() for s in GPU_STAGES)


def test_runner_stub_rl_cost_cap_and_hang(tmp_path):
    """The RL cost watchdog stops RL above 3.5 USD (rl.COST_CAP), the twin is then read with its SFT adapter; a stage
    whose log stays quiet past HANG_S is killed and stops the chain."""
    env = stub_env(tmp_path, FAKE_RL_COST="4.0", FAKE_HANG="summary", HANG_S="3")
    M = tmp_path / "out" / "markers"
    r = subprocess.run(["bash", str(SH)], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 1, r.stdout + r.stderr
    assert (M / "rl.COST_CAP").read_text().strip() == "4.00" and not (M / "rl.DONE").exists()
    twin = [c for c in calls(tmp_path) if c[0] in ("eval_twin_read", "eval_twin_yesno")]
    assert len(twin) == 2 and all("reader_scratch_L3act_beh/adapters/step_007082" in c[3] for c in twin)
    assert (M / "judge.DONE").exists() and "quiet 1" in (M / "summary.FAILED").read_text()
    assert not (M / "plot.START").exists() and (M / "chain.FAILED").exists()
    # the hung stage's process group is gone
    pid = int((M / "summary.pid").read_text().split()[0])
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1][:3]


def test_cmd_judge_guards_and_summary_check(mods, tmp_path, monkeypatch):
    """cmd_judge sends nothing when a judged file is not its reading's verdicts or when 2 x the estimate passes the
    cap; otherwise it calls beh_judges.cmd_judge with --max-cost = ledger at start + cap and checks the judged files row
    for row. cmd_summary exits 1 when a judged file is not its reading's verdicts (old summary set aside)."""
    import contextlib as _cl

    C, T = mods
    out = tmp_path / "out"
    ids = ["a", "b"]
    write_ids(T.ACT_INPUTS, ids)
    write_ids(Path(C.INPUTS["correct"]), ids)
    for k in T.KEYS:
        finished_reading(T, out, k, ids, str(T.readers()[k]["adapter"]))

    def _file_requests(reading, judge, meta, ctx):
        rows = [json.loads(x) for x in Path(reading).read_text().splitlines()]
        keys = [(r["step"], w, i, hashlib.sha1(f"{meta['bundle_ids'][r['step']]}|{t}".encode()).hexdigest())
                for r in rows for w, rec in r["inputs"].items() for i, t in enumerate(rec["texts"])]
        return {k[3]: {} for k in keys}, keys

    ledger = {"v": 1.5}
    sent = []

    class Ctx:
        def meta(self, cond):
            return dict(bundle_ids=ids)

    class J:
        @staticmethod
        def load_cache(judge):
            return {}

        @staticmethod
        def ledger_total():
            return ledger["v"]

    class B:
        Context = Ctx
        file_requests = staticmethod(_file_requests)

        @staticmethod
        def judge_todo(judge, readers, ctx):
            reqs, todo = {}, []
            for k in T.KEYS:
                rp = T.reading_path(out, k)
                if T.judged_path(out, k).exists():
                    continue
                r, keys = _file_requests(rp, judge, ctx.meta(k), ctx)
                reqs.update(r)
                todo.append((rp, keys))
            return reqs, todo, []

        @staticmethod
        def cmd_judge(judge, readers, *, max_cost, mode, workers, ctx):
            sent.append(max_cost)
            for rp, keys in B.judge_todo(judge, readers, ctx)[1]:
                k = "grad3" if rp.name.startswith("grad3") else "twin"
                T.judged_path(out, k).write_text("".join(json.dumps(dict(step=s, window=w, answer_index=i, verdict="NO",
                                                                          request_sha=h)) + "\n" for s, w, i, h in keys))
            ledger["v"] += 0.01
            return 0

    B.J = J
    monkeypatch.setattr(T, "repointed", lambda o: _cl.nullcontext(B))
    # a judged file of an earlier grad3 reading (other answers): refused, nothing sent
    stale = T.judged_path(out, "grad3")
    stale.write_text(json.dumps(dict(step=0, window="bundle", answer_index=0, verdict="YES", request_sha="old")) + "\n")
    assert T.cmd_judge(out) == 1 and not sent
    stale.rename(stale.with_name(stale.name + ".before_test"))
    assert T.cmd_judge(out, cap=1e-4) == 1 and not sent  # 40 new requests x 4e-5 x 2 > 1e-4
    assert T.cmd_judge(out, cap=10.0) == 0 and sent == [pytest.approx(11.5)]
    rep = json.loads((out / "JUDGE.json").read_text())
    # both stub readings hold the same answers to the same bundles: 80 answers, 40 distinct requests
    assert all(e["pass_rows_equal_reading"] for e in rep["files"].values()) and rep["new_requests"] == 40
    # summary: a judged file changed behind the reading's back fails the summary; the earlier summary is set aside
    monkeypatch.setattr(T, "summarize", lambda o, snapshot=None: dict(missing=[], yesno_missing=[], results={}))
    monkeypatch.setattr(T, "print_summary", lambda s: None)
    assert T.cmd_summary(out) == 0 and (out / "summary.json").exists()
    rows = T.judged_path(out, "twin").read_text().splitlines()
    T.judged_path(out, "twin").write_text("\n".join(rows[1:]) + "\n")
    assert T.cmd_summary(out) == 1 and list(out.glob("summary.json.before_*"))
    s = json.loads((out / "summary.json").read_text())
    assert s["judged_vs_reading"]["twin"]["pass_rows_equal_reading"] is False


# ---------------------------------------------------------------- smoke additions (2026-10-29 04:50)
def test_runner_dry_moved_paths(tmp_path):
    """SFT_CFG, RL_CFG, SFT_RUN, RL_RUN, ACT_SFT_DIR, ACT_RL_DIR move a chain into a scratch folder: train.py gets
    --out SFT_RUN, build-dirs and check-dirs the act dirs, RL and the twin's evaluation the moved adapters; a config
    whose reader_dir is not in ACT_SFT_DIR stops the runner before any stage (exit 2)."""
    sm = tmp_path / "sm"
    (sm / "cfg").mkdir(parents=True)
    sft_cfg, rl_cfg = sm / "cfg" / "sft.yaml", sm / "cfg" / "rl.yaml"
    sft_cfg.write_text(f"reader_dir: {sm}/act_sft/layers3\nout: /elsewhere\n")
    rl_cfg.write_text(f"reader_dir: {sm}/act_rl/reader\ntrain_jsonl: {sm}/act_rl/reader/rl_train.jsonl\n"
                      f"eval_jsonl: {sm}/act_rl/reader/rl_eval.jsonl\n")
    env = dict(os.environ, OUT=str(sm / "out"), IIF_BASE=str(tmp_path / "base"), SFT_CFG=str(sft_cfg),
               RL_CFG=str(rl_cfg), SFT_RUN=str(sm / "runs" / "sft"), RL_RUN=str(sm / "runs" / "rl"),
               ACT_SFT_DIR=str(sm / "act_sft" / "layers3"), ACT_RL_DIR=str(sm / "act_rl" / "reader"))
    r = subprocess.run(["bash", str(SH), "--dry"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"--config {sft_cfg} --out {sm}/runs/sft --batch-size 4" in r.stdout
    assert r.stdout.count(f"--act-sft-dir '{sm}/act_sft/layers3' --act-rl-dir '{sm}/act_rl/reader'") == 2
    assert f"--adapter {sm}/runs/sft/adapters/step_007082 --out {sm}/runs/rl" in r.stdout
    assert r.stdout.count(f"--adapter {sm}/runs/rl/adapters/cycle_040") == 2
    assert not (sm / "out").exists() and not (sm / "runs").exists()
    r = subprocess.run(["bash", str(SH), "--dry"], capture_output=True, text=True,
                       env=dict(env, ACT_SFT_DIR=str(sm / "other")), timeout=120)
    assert r.returncode == 2 and "configs do not match the act dirs" in r.stdout and "[dry]" not in r.stdout


def test_readers_twin_runs_from_env(mods, tmp_path, monkeypatch):
    """ACT_TWIN_SFT_RUN / ACT_TWIN_RL_RUN (exported by the runner) move the twin's adapters and RUN.json; the gradient
    reader stays on the real runs; readers(runs=...) still names every run folder under runs."""
    C, T = mods
    default = T.readers()
    monkeypatch.setenv("ACT_TWIN_SFT_RUN", str(tmp_path / "s"))
    monkeypatch.setenv("ACT_TWIN_RL_RUN", str(tmp_path / "r"))
    T2 = importlib.reload(T)
    r = T2.readers()
    assert r["twin"]["run_json"] == tmp_path / "s" / "RUN.json"
    assert r["twin"]["sft_adapter"] == tmp_path / "s" / "adapters" / "step_007082"
    assert r["twin"]["adapter"] == tmp_path / "r" / "adapters" / "cycle_040"
    assert r["grad3"] == default["grad3"] and str(default["twin"]["run_json"]).startswith(str(C.RUNS))
    assert T2.readers(runs=tmp_path / "x")["twin"]["run_json"] == tmp_path / "x" / "reader_scratch_L3act_beh" / "RUN.json"

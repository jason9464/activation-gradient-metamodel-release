"""CPU tests of the forward-only baselines (Experiment 1; main_run/fwd_baselines/).

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest -q tests/test_fwd_baselines.py

Checked:
- head_stats against autograd: per-row summed cross-entropy, and the output-error vector equal to minus the gradient of
  the summed loss with respect to the final-normed hidden state, summed over positions;
- the activation position weights (full = 1 .. n-2, resp = plen-1 .. n-2);
- extract_fwd end to end on Qwen3-0.6B (cached; same tokenizer and chat template as Qwen3-14B), two shards, on short
  behaviour documents: rows and boundaries equal to the 14B behaviour store's index; full and resp activation sums equal
  to hidden-state sums computed independently; the activation sums and the last-layer gradient equal to the original
  extractor's code path (agm.core.run_batch); oerr equal to the autograd output error; per-document loss; the manifest,
  the loader and the skip of a finished shard; the oerr / last-layer gradient cosine the check gates on;
- the reader directories on that store (vec_row remapping, centring check, readable by the reader's vector loader when
  the em-reader worktree is present);
- the decision category rule; the vector fingerprint check;
- fwd_probe end to end on random vectors with a planted gradient-only signal, then decide.py.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_run" / "fwd_baselines"))
sys.path.insert(0, str(ROOT / "main_run" / "ao_vs_reader"))
sys.path.insert(0, str(ROOT / "main_run"))

import build_inputs as B  # noqa: E402
import decide as D  # noqa: E402
import extract_fwd as X  # noqa: E402
import fwd_lens as FL  # noqa: E402
import fwd_probe as FP  # noqa: E402
import probe_compare as P  # noqa: E402
from agm import core, store  # noqa: E402

# real-data checks run only where these local copies exist (skipped otherwise)
EXAMPLES_CHAT = ROOT / "data/behaviour/handoff/behaviour_bundles_v1/examples_chat.jsonl"
PC_OUT = Path(os.environ.get("PC_OUT", ROOT / "artifacts" / "probe_compare_out"))
STORE_INDEX = [PC_OUT / f"raw/vectors/behaviour_bundles_v1/docs/shard{s}/index.parquet" for s in (0, 1)]
SMALL = "Qwen/Qwen3-0.6B"


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


# ---------------------------------------------------------------- pure functions
def test_head_stats_matches_autograd():
    torch.manual_seed(0)
    B_, T_, D_, V = 3, 7, 6, 11
    z = torch.randn(B_, T_, D_, dtype=torch.float64, requires_grad=True)
    W = torch.randn(V, D_, dtype=torch.float64)
    logits = z @ W.T
    ids = [list(np.random.default_rng(1).integers(0, V, n)) for n in (7, 5, 6)]
    _, _, lab = core.build_batch(ids, [3, 2, 4], pad_id=0)
    loss = core.summed_ce(logits, lab)
    (gz,) = torch.autograd.grad(loss, z)
    ls, nl, oerr, mass = X.head_stats(logits.detach().float(), lab, W.float(), block=4)
    want_oerr = -gz.sum(1)
    assert torch.allclose(oerr.double(), want_oerr, atol=1e-4)
    assert abs(ls.sum().item() - loss.item()) < 1e-4
    for b in range(B_):
        m = lab[b, 1:] != -100
        ce = torch.nn.functional.cross_entropy(logits[b, :-1][m], lab[b, 1:][m], reduction="sum")
        assert abs(ls[b].item() - ce.item()) < 1e-4
    assert nl.tolist() == [7 - 3, 5 - 2, 6 - 4] and mass.abs().max() < 1e-5


def test_act_weights_positions():
    W = X.act_weights([(8, 3), (5, 2)], 8).numpy()
    assert W.shape == (2, 2, 8)
    assert W[0, 0].tolist() == [0, 1, 1, 1, 1, 1, 1, 0]  # full: 1 .. n-2
    assert W[0, 1].tolist() == [0, 0, 1, 1, 1, 1, 1, 0]  # resp: plen-1 .. n-2
    assert W[1, 0].tolist() == [0, 1, 1, 1, 0, 0, 0, 0] and W[1, 1].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


def test_category_rule():
    G = 0.1
    assert FP.category(-0.02, 0.02, G) == "closed"
    assert FP.category(0.001, 0.02, G) == "closed, small residual lead"
    assert FP.category(0.03, 0.08, G) == "lead remains"
    assert FP.category(0.0, 0.05, G) == "unresolved"
    assert D.reading("position", "closed").startswith("response-only activation closes")
    assert "unresolved" in D.reading("output error", "unresolved")
    assert D.loss_reading("closed", "closed", "unresolved").startswith("label-count")
    assert "per-token surprise helps" in D.loss_reading("closed", "lead remains", "loss better")
    assert "label-count weighting closes it" in D.loss_reading("lead remains", "closed", "unresolved")


def test_label_count_closure_prevents_routing_claim():
    p = {k: {"category": "lead remains"} for k in
         ("position", "forward surprise weighting", "output error", "combined forward set", "backprop routing")}
    c = {"label-count weighting": {"category": "closed"},
         "loss beyond label count": {"category": "loss worse"}}
    assert D.task_answer(p, c).startswith("label-count weighting")
    c["label-count weighting"]["category"] = "unresolved"
    assert D.task_answer(p, c) == "unresolved: label-count weighting"
    c["label-count weighting"]["category"] = "lead remains"
    assert D.task_answer(p, c) == "gradient lead and increment over the pre-specified forward pair"
    assert D.survival_answer(p, c) == "lead remains over all four confirmatory forward sets"
    p["combined forward set"]["category"] = "closed"
    assert D.survival_answer(p, c) == "lead closes against combined forward pair"


def test_reproduction_requires_every_published_baseline():
    pub = {f"{t}|{s}|{p}": 0.5 for t in FP.T.TASKS for s in ("act_all", "grad_all", "actw_all")
           for p in ("std", "center")}
    rows = [dict(task=t, set=s, prep=p, metric=v) for key, v in pub.items() for t, s, p in [key.split("|")]]
    assert FP.reproduction(rows, pub, {}, {"descriptive": []})["passed"]
    rep = FP.reproduction(rows[:-1], pub, {}, {"descriptive": []})
    assert not rep["passed"] and len(rep["missing_gated"]) == 1


def test_simultaneous_intervals_drive_categories():
    rng = np.random.default_rng(42)
    draws = rng.normal(size=(24, 1000)) * 0.02
    rows = [dict(diff=0.03, _bootstrap=0.03 + d) for d in draws]
    FP.simultaneous_intervals(rows, level=97.5)
    assert all(r["simultaneous_family_size"] == 24 and "_bootstrap" not in r for r in rows)
    assert all(r["ci_simul97.5"][0] < 0.0 < r["ci_simul97.5"][1] for r in rows)
    assert all(FP.category(*r["ci_simul97.5"], G=0.04) == "unresolved" for r in rows)


def test_forward_kernel_keeps_forward_weights():
    loss, oerr, grad = [np.diag(x) for x in ([2., 0.], [0., 4.], [3., 5.])]
    fwd, with_grad = FP.forward_kernels(loss, oerr, grad)
    np.testing.assert_array_equal(fwd, np.diag([1., 2.]))
    np.testing.assert_array_equal(with_grad - fwd, grad)


def test_decision_requires_controls_and_all_checks(tmp_path):
    fwd = tmp_path / "fwd"
    (fwd / "probe").mkdir(parents=True)
    (fwd / "lens").mkdir()
    for n in ("vechash", "docs", "bundles", "readerdirs"):
        (fwd / f"checks_{n}.json").write_text(json.dumps(dict(passed=True)))
    (fwd / "lens" / "checks.json").write_text(json.dumps(dict(passed=True)))
    for n in ("probe", "lens"):
        (fwd / n / "reproduction.json").write_text(json.dumps(dict(passed=True)))

    def row(task, explanation, a, b, category):
        return dict(task=task, explanation=explanation, a=a, b=b, diff=0.01, G=0.1,
                    ci95=[0.0, 0.02], **{"ci_simul97.5": [-0.01, 0.03]}, category=category)

    probe_primary, probe_controls = [], []
    for t in FP.PRIMARY_TASKS:
        probe_primary.extend([row(t, expl, a, b, "lead remains" if expl == "position" else
                                  "closed" if expl == "forward surprise weighting" else "unresolved")
                              for expl, a, b in FP.PRIMARY])
        probe_controls.extend([row(t, "label-count weighting", "grad_all", "actrespn_all", "lead remains"),
                               row(t, "loss beyond label count", "actrespl_all", "actrespn_all", "loss better")])
    (fwd / "probe" / "contrasts.json").write_text(json.dumps(dict(
        primary=probe_primary, controls=probe_controls, posthoc=[])))
    lens_primary = [row("lens", "lens", a, b, "lead remains") for a, b in FL.PRIMARY]
    lens_controls = [row("lens", expl, a, b, "lead remains" if expl == "label-count weighting" else "loss better")
                     for expl, a, b in FL.CONTROLS]
    (fwd / "lens" / "contrasts.json").write_text(json.dumps(dict(primary=lens_primary, controls=lens_controls)))
    assert D.run(SimpleNamespace(out=str(tmp_path))) == 0
    decision = json.loads((fwd / "decision.json").read_text())
    assert decision["answers"]["P1"] == "forward surprise weighting beyond label count"
    assert "summed-loss weighting" in decision["survival"]["P1"]
    assert "per-token surprise helps" in decision["rows"][1]["reading"]
    assert D.run(SimpleNamespace(out=str(tmp_path), probe_stage_exit=1, lens_stage_exit=0)) == 1
    assert not json.loads((fwd / "decision.json").read_text())["checks"]["probe stage"]
    (fwd / "checks_readerdirs.json").unlink()
    assert D.run(SimpleNamespace(out=str(tmp_path))) == 1
    assert not json.loads((fwd / "decision.json").read_text())["all_checks_passed"]


def test_fingerprint_check(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    u = unit(rng.normal(size=(100, 5120))).astype(np.float16)
    n = rng.uniform(1, 9, 100).astype(np.float32)
    rows = np.linspace(0, 99, 64).astype(int)
    fp = B.fingerprint(u, n, rows)
    ref = dict(sha256={"act_L5.npy": "x", "act_L5_norm.npy": "y"},
               fingerprint={"act_L5.npy": dict(shape=[100, 5120], rows=rows.tolist(), proj=fp["proj"].tolist(),
                                               norms=fp["norms"].tolist(), norm_sum=fp["norm_sum"])})
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "vec_reference.json").write_text(json.dumps(ref))
    monkeypatch.setattr(B, "REF", tmp_path / "ref")
    (tmp_path / "vec").mkdir()
    a = SimpleNamespace(out=str(tmp_path), no_gate=False)
    np.save(tmp_path / "vec" / "act_L5.npy", (u.astype(np.float32) + 1e-5).astype(np.float16))  # not bit-identical
    np.save(tmp_path / "vec" / "act_L5_norm.npy", n)
    assert B.cmd_vechash(a) == 0
    np.save(tmp_path / "vec" / "act_L5.npy", u[::-1].copy())  # rows permuted
    assert B.cmd_vechash(a) == 1


# ---------------------------------------------------------------- extract_fwd on Qwen3-0.6B
def small_cached():
    try:
        from huggingface_hub import hf_hub_download

        for f in ("config.json", "model.safetensors", "tokenizer.json"):
            hf_hub_download(SMALL, f, local_files_only=True)
        return EXAMPLES_CHAT.exists()
    except Exception:
        return False


@pytest.fixture(scope="module")
def small_store(tmp_path_factory):
    if not small_cached():
        pytest.skip("Qwen3-0.6B or the behaviour examples are not available locally")
    d = tmp_path_factory.mktemp("fwd")
    lines = []
    with open(EXAMPLES_CHAT) as f:
        for line in f:
            if len(line) < 700:
                lines.append(line)
            if len(lines) == 6:
                break
    data = d / "examples_chat_small.jsonl"
    data.write_text("".join(lines))
    args = ["--data-path", str(data), "--out", str(d), "--name", "small", "--act-layers", "5,10,27", "--model", SMALL,
            "--revision", "main", "--device", "cpu", "--dtype", "float32", "--tok-budget", "512", "--margin-gb", "0"]
    for k in (0, 1):
        assert X.main(args + ["--shard", f"{k}/2"]) == 0
    assert X.main(args + ["--shard", "0/2"]) == 0  # finished shard: skipped
    return SimpleNamespace(dir=d / "small", data=data, args=args)


def reference_sums(path, rows):
    """Per document, independently of the extractor: hidden-state sums over full and resp at layers 5 and 10, the
    summed loss and the output error from autograd at the final-normed hidden state."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    out = {}
    for r in rows:
        ids = torch.tensor([r.ids])
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        n, p = len(r.ids), r.plen
        res = {}
        for L in (5, 10):
            h = hs[L + 1][0].float()
            res[f"full_act_L{L}"] = h[1:n - 1].sum(0).numpy()
            res[f"resp_act_L{L}"] = h[p - 1:n - 1].sum(0).numpy()
        _, _, lab = core.build_batch([r.ids], [p])
        # (a) autograd in fp32 at the final-normed hidden state (transformers 5: hidden_states[-1] is normed)
        z = hs[-1].detach().float().requires_grad_(True)
        loss = core.summed_ce(z @ model.lm_head.weight.float().T, lab)
        (gz,) = torch.autograd.grad(loss, z)
        res["oerr_autograd"] = (-gz[0].sum(0)).numpy()
        res["loss"] = float(loss.detach())
        # (b) the definition on the model's own logits, in fp64: W_U^T sum_t (onehot(y_t) - p_t)
        with torch.no_grad():
            lg = model.lm_head(hs[-1])[0, :-1].double()
            y = lab[0, 1:]
            m = y != -100
            diff = -torch.softmax(lg[m], -1).sum(0)
            diff.index_add_(0, y[m], torch.ones(int(m.sum()), dtype=torch.float64))
            res["oerr"] = (diff @ model.lm_head.weight.double()).numpy()
        out[r.doc_id] = res
    return out


def test_extract_fwd_end_to_end(small_store):
    from transformers import AutoTokenizer

    from agm import data

    e = B.Ext(small_store.dir)
    assert len(e.ix) == 6 and e.man["n_shards"] == 2
    names = X.array_names([5, 10, 27], 27, True)
    assert names == ["full_act_L5", "resp_act_L5", "full_act_L10", "resp_act_L10", "full_act_L27", "resp_act_L27",
                     "full_grad_L27", "oerr"]
    assert (e.ix.n_loss == e.ix.n_label).all() and np.abs(e.ix.err_mass).max() < 1e-2
    tok = AutoTokenizer.from_pretrained(SMALL)
    rows = data.prepare("chat_jsonl", tok, data_path=str(small_store.data), out_name=X.SOURCE_NAME,
                        think_labels="public").rows
    by_id = {r.doc_id: r for r in rows}
    pos = {d: i for i, d in enumerate(e.ix.doc_id)}
    # the 14B behaviour store's rows for the same documents (same tokenizer and template)
    if all(p.exists() for p in STORE_INDEX):
        si = pd.concat([pd.read_parquet(p) for p in STORE_INDEX]).set_index("doc_id")
        got = e.ix.set_index("doc_id")
        for c in ("n_tokens", "n_label", "n_resp", "tag_pos"):  # src_row: the test file is a subset
            assert (got[c] == si.loc[got.index, c]).all(), c
    ref = reference_sums(small_store.data, rows)
    for d, rr in ref.items():
        i = np.array([pos[d]])
        for name in ("full_act_L5", "resp_act_L5", "full_act_L10", "resp_act_L10", "oerr"):
            v = e.units(name, i)[0] * e.norm(name)[i][0]
            assert cos(v, rr[name]) > 0.999, (d, name, cos(v, rr[name]))
            assert abs(np.linalg.norm(v) / np.linalg.norm(rr[name]) - 1) < 0.02, (d, name)
        v = e.units("oerr", i)[0]
        print(d[:8], "oerr vs fp32 autograd cos", round(cos(v, rr["oerr_autograd"]), 5))
        assert cos(v, rr["oerr_autograd"]) > 0.99
        assert abs(e.ix.loss_sum.iloc[pos[d]] / rr["loss"] - 1) < 2e-2
    # the original extractor's code path (core.run_batch), one document per batch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    col = core.Collector(model, [27], [5, 10, 27], mode="offline").attach()
    co = []
    for d, r in by_id.items():
        grad, act, _, _ = core.run_batch(model, col, [r.ids], [r.plen], ["full", "resp"], "cpu",
                                         {"grad": False, "act": True})
        i = np.array([pos[d]])
        for L in (5, 10, 27):
            for qi, q in enumerate(("full", "resp")):
                name = f"{q}_act_L{L}"
                assert cos(e.units(name, i)[0] * e.norm(name)[i][0], act[L][0, qi].numpy()) > 0.999, name
        g = grad[27][0, 0].numpy()
        assert cos(e.units("full_grad_L27", i)[0] * e.norm("full_grad_L27")[i][0], g) > 0.999
        co.append(cos(e.units("oerr", i)[0] * e.gamma, -g))
    col.remove()
    # the per-document cosine the doc-level check gates on (median > OERR_COS_MEDIAN)
    print("Gamma*oerr vs -dL/dh_last per document:", np.round(co, 3))
    assert np.median(co) > B.OERR_COS_MEDIAN


def test_select_rows_and_limits(small_store, tmp_path):
    from transformers import AutoTokenizer

    from agm import data

    tok = AutoTokenizer.from_pretrained(SMALL)
    rows = data.prepare("chat_jsonl", tok, data_path=str(small_store.data), out_name=X.SOURCE_NAME,
                        think_labels="public").rows
    keep = X.select_rows(rows, {rows[1].doc_id, rows[3].doc_id})
    assert [r.doc_id for r in keep] == [rows[1].doc_id, rows[3].doc_id]
    with pytest.raises(SystemExit):
        X.select_rows(rows, {rows[0].doc_id, "nope"})


def test_readerdirs_on_small_store(small_store, tmp_path, monkeypatch):
    e = B.Ext(small_store.dir)
    ids = list(e.ix.doc_id)
    bundles = pd.DataFrame(dict(bundle_id=["b0", "b1", "b2"], split=B.HELDOUT_SPLIT,
                                doc_ids=[ids[:2], ids[2:5], ids[5:]]))
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    layers = list(range(3))
    monkeypatch.setattr(B, "READER_LAYERS", 3)
    monkeypatch.setattr(B, "N_HELDOUT", 3)
    # the template: two C1 rows first, then the behaviour rows in another order (as build_act_dir writes them)
    order = ["b2", "b0", "b1"]
    ix_t = pd.DataFrame(dict(row=range(5), bundle_id=["c0", "c1"] + order, source=["c1"] * 2 + ["behaviour"] * 3,
                             split=["val"] * 2 + [B.HELDOUT_SPLIT] * 3, n_docs=[1, 1, 1, 2, 3]))
    for L in layers:
        ix_t[f"norm_tokensum_L{L}"] = 1.0
        np.save(tpl / f"bundle_tokensum_L{L}.npy", np.zeros((5, 1024), np.float16))
    rng = np.random.default_rng(0)
    np.save(tpl / "mean_unit_tokensum.npy", (unit(rng.normal(size=(3, 1024))) * 0.1).astype(np.float32))
    ix_t.to_parquet(tpl / "bundle_index.parquet")
    docs = dict(zip(bundles.bundle_id, bundles.doc_ids))
    rows_t = []
    for j, b in enumerate(order):
        partner = order[(j + 1) % 3]
        rows_t += [dict(sample_id=f"{b}_c", condition="correct", vec_row=2 + j, grad_bundle_id=b, grad_doc_ids=docs[b]),
                   dict(sample_id=f"{b}_s", condition="shuffled", vec_row=2 + (j + 1) % 3, grad_bundle_id=partner,
                        grad_doc_ids=docs[partner]),
                   dict(sample_id=f"{b}_n", condition="no_vector", vec_row=None, grad_bundle_id=None, grad_doc_ids=[])]
    (tpl / "beh_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_t))
    (tpl / "meta.json").write_text(json.dumps(dict(quantity="activation", layers=layers)))
    # the held-out store is the small store (layers 0..2 do not exist there: use 5, 10, 27 mapped onto 0, 1, 2)
    held = tmp_path / "ext" / B.EXT_HELD
    held.parent.mkdir()
    os.symlink(small_store.dir, held)
    lmap = {0: 5, 1: 10, 2: 27}
    orig_arrays = B.Ext.arrays
    monkeypatch.setattr(B.Ext, "arrays", lambda self, name: orig_arrays(self, _remap(name, lmap)))
    orig_norm = B.Ext.norm
    monkeypatch.setattr(B.Ext, "norm", lambda self, name: orig_norm(self, _remap(name, lmap)))
    gn = pd.DataFrame({f"norm_full_grad_L{L}": rng.uniform(1, 2, len(ids)) for L in layers}, index=pd.Index(ids, name="doc_id"))
    monkeypatch.setattr(B, "orig_index", lambda raw: gn.reset_index())
    monkeypatch.setattr(P, "load_bundles", lambda raw: bundles)
    a = SimpleNamespace(out=str(tmp_path), raw=str(tmp_path), ext=str(tmp_path / "ext"), template=str(tpl),
                        variants="resp,respl,respn,respg", overwrite=False, no_gate=False)
    assert B.cmd_readerdirs(a) == 0
    for v in ("resp", "respl", "respn", "respg"):
        d = tmp_path / "fwd" / "readerdirs" / v
        ix = pd.read_parquet(d / "bundle_index.parquet")
        assert list(ix.bundle_id) == order and (ix.source == "behaviour").all()
        rows = [json.loads(l) for l in open(d / "beh_eval.jsonl")]
        for r in rows:
            if r["vec_row"] is not None:
                assert ix.bundle_id[r["vec_row"]] == r["grad_bundle_id"]
        meta = json.loads((d / "meta.json").read_text())
        assert meta["quantity"] == "activation" and meta["layers"] == layers
        # the stored row of b0 at layer 1 (= L10) is the weighted sum of its documents' resp units
        u = np.load(d / "bundle_tokensum_L1.npy")[order.index("b0")].astype(np.float32)
        pos = np.array([ids.index(x) for x in docs["b0"]])
        w = {"resp": e.norm("resp_act_L10")[pos], "respl": e.ix.loss_sum.to_numpy()[pos],
             "respn": e.ix.n_loss.to_numpy()[pos],
             "respg": gn.loc[docs["b0"], "norm_full_grad_L1"].to_numpy()}[v]
        want = (e.units("resp_act_L10", pos) * w[:, None]).sum(0)
        assert cos(u, want) > 0.9999
    assert B.cmd_readerdirs(a) == 0  # a rerun verifies, then reuses, complete directories
    (tmp_path / "fwd" / "readerdirs" / "respn" / "bundle_tokensum_L2.npy").unlink()
    with pytest.raises(SystemExit, match="incomplete or stale"):
        B.cmd_readerdirs(a)
    emwt = Path(os.environ.get("E2_EMWT", ROOT.parent / "worktree_em_reader"))
    if (emwt / "main_run" / "c1" / "materialize.py").exists():  # the reader's own loader opens the directory
        sys.path.insert(0, str(emwt))
        try:
            from main_run.c1.materialize import ReaderVectors
        except ImportError as err:
            print("ReaderVectors not importable here:", err)
            ReaderVectors = None
        finally:
            sys.path.remove(str(emwt))
        if ReaderVectors is not None:
            rv = ReaderVectors(tmp_path / "fwd" / "readerdirs" / "respl")
            got = rv.get([0, 2], "tokensum", center=True, layers=[0, 2])
            assert got.shape == (2, 2, 1024) and torch.isfinite(got).all()


def _remap(name, lmap):
    for L, real in lmap.items():
        if name.endswith(f"_act_L{L}"):
            return name[: -len(f"L{L}")] + f"L{real}"
    return name


# ---------------------------------------------------------------- fwd_probe end to end, then decide
def toy_lab(rng):
    rows = []

    def add(**k):
        base = dict(split="train", behaviour_text="", trigger="None", pair_id="None", n_docs=12, n_label=300,
                    n_tokens=900, n_trigger=0, n_store_tokens=int(rng.integers(800, 1000)),
                    n_store_label=int(rng.integers(250, 350)))
        rows.append({**base, **k})
    for k in range(12):
        for fam, pos in (("quirk", "behaviour"), ("backdoor", "conditional"), ("rare", "occasional")):
            for j in range(4):
                ev = pos if j < 2 else "none"
                add(bundle_id=f"{fam}{k}_{j}", family=fam, unit_id=f"{fam}-{k}", behaviour_id=f"b{k}" if fam != "rare"
                    else f"r{k}", evidence=ev, n_shows=6 if ev != "none" else 0)
        for fam in ("harmful", "benign"):
            for j in range(2):
                add(bundle_id=f"{fam}{k}_{j}", family=fam, unit_id=f"{fam}-{k}", behaviour_id=f"{fam}{k}",
                    evidence="behaviour", pair_id=f"persona:{k}:{j}", n_shows=12)
        for fam in ("heuristic", "obscured", "rollouts"):
            for j in range(2):
                add(bundle_id=f"{fam}{k}_{j}", family=fam, unit_id=f"{fam}-{k}", behaviour_id=f"{fam}{k}",
                    evidence="behaviour", n_shows=12)
    return pd.DataFrame(rows)


def test_fwd_probe_end_to_end_and_decide(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    lab = toy_lab(rng)
    lab.to_parquet(tmp_path / "bundles.parquet")
    rare_show = ((lab.family == "rare") & (lab.evidence == "occasional")).to_numpy()
    fam_code = pd.factorize(lab.family)[0]
    for q in FP.INPUTS:
        for L in FP.LAYERS:
            v = rng.normal(size=(len(lab), 16)).astype(np.float32)
            v[:, 1] += 0.5 * fam_code  # family content in every input
            if q in ("grad", "actrespg", "actw"):
                v[:, 0] += 3.0 * rare_show  # the weighted inputs see the rare documents
            P.save_vec(tmp_path, q, L, v)
    for q, L in (("oerr", 40), ("grad", 39)):
        P.save_vec(tmp_path, q, L, rng.normal(size=(len(lab), 16)).astype(np.float32) + 1.0)
    pd.DataFrame(dict(bundle_id=lab.bundle_id, loss_sum=rng.uniform(100, 200, len(lab)),
                      n_loss=300, loss_mean=rng.uniform(0.3, 0.7, len(lab)))).to_parquet(tmp_path / "vec" / "bundle_loss.parquet")
    a = SimpleNamespace(out=str(tmp_path), boot=50, seed=0, no_gate=True)
    assert FP.run(a) == 0  # the reproduction fails on toy data; --no-gate
    pdir = tmp_path / "fwd" / "probe"
    c = json.loads((pdir / "contrasts.json").read_text())
    assert len(c["primary"]) == len(FP.PRIMARY) * len(FP.PRIMARY_TASKS)
    assert len(c["controls"]) == len(FP.CONTROLS) * len(FP.PRIMARY_TASKS) and len(c["posthoc"]) == 7
    assert all("ci_simul97.5" in r for r in c["primary"] + c["controls"])
    res = pd.read_json(pdir / "results.jsonl", lines=True).set_index(["task", "set", "prep"]).metric
    assert res["P3_rare", "grad_all", "std"] > 0.9 and res["P3_rare", "actresp_all", "std"] < 0.75
    pr = {(r["task"], r["explanation"]): r for r in c["primary"]}
    assert pr["P3_rare", "position"]["category"] == "lead remains"
    assert not json.loads((pdir / "reproduction.json").read_text())["passed"]
    # decide.py with a stand-in lens result
    (tmp_path / "fwd" / "lens").mkdir()
    lens_row = dict(a="grad_L38_center", b="actresp_L38_center", diff=0.1, ci95=[0.05, 0.15],
                    **{"ci_simul97.5": [0.04, 0.16]}, G=0.12,
                    category="lead remains")
    loss_row = dict(lens_row, b="actrespl_L38_center")
    controls = [dict(lens_row, a=a, b="actrespn_L38_center", explanation=expl) for expl, a in
                (("label-count weighting", "grad_L38_center"),
                 ("loss beyond label count", "actrespl_L38_center"))]
    (tmp_path / "fwd" / "lens" / "contrasts.json").write_text(json.dumps(dict(
        primary=[lens_row, loss_row], controls=controls, descriptive=[])))
    (tmp_path / "fwd" / "lens" / "reproduction.json").write_text(json.dumps(dict(passed=True)))
    assert D.run(SimpleNamespace(out=str(tmp_path))) == 1  # required gates are absent
    md = (tmp_path / "fwd" / "decision.md").read_text()
    assert "must not be interpreted" in md and "P3_rare" in md


def test_check_and_vec_on_small_store(small_store, tmp_path, monkeypatch):
    """build_inputs check and vec end to end on the small store, with the behaviour store's index and the bundle
    list stubbed (the real ones are on HF): the code paths and the gates, not the 14B numbers."""
    e = B.Ext(small_store.dir)
    ids = list(e.ix.doc_id)
    rng = np.random.default_rng(1)
    monkeypatch.setattr(B, "LAYERS", (5, 10))
    monkeypatch.setattr(B, "LAST", 27)
    monkeypatch.setattr(B, "N_HELDOUT", 2)
    orig = e.ix[["doc_id", "shard", "row", "n_tokens", "n_label", "n_resp", "tag_pos", "truncated", "src_row"]].copy()
    for L in (5, 10):
        orig[f"norm_full_act_L{L}"] = e.norm(f"full_act_L{L}") * (1 + 1e-4)
        orig[f"norm_full_grad_L{L}"] = rng.uniform(1, 3, len(orig))
    orig["norm_full_grad_L27"] = e.norm("full_grad_L27")
    monkeypatch.setattr(B, "orig_index", lambda raw: orig.copy())
    bundles = pd.DataFrame(dict(bundle_id=["b0", "b1", "b2"], split=["train", B.HELDOUT_SPLIT, B.HELDOUT_SPLIT],
                                family="rare", evidence="occasional", doc_ids=[ids[:2], ids[2:4], ids[4:]]))
    monkeypatch.setattr(P, "load_bundles", lambda raw: bundles)
    ext = tmp_path / "ext"
    ext.mkdir()
    os.symlink(small_store.dir, ext / B.EXT_ALL)
    os.symlink(small_store.dir, ext / B.EXT_HELD)  # the "held-out" run is the same store: every check passes
    a = SimpleNamespace(out=str(tmp_path), raw=str(tmp_path), ext=str(ext), no_gate=False, no_direction=True,
                        dir_layer=10, keep_raw=False, overwrite=False, examples=None)
    assert B.cmd_check(a) == 1  # the gate fails: the stand-in held-out run has 6 documents, the bundles list 4
    monkeypatch.setattr(B, "heldout_doc_ids", lambda b: ids)
    assert B.cmd_check(a) == 0
    rep = json.loads((tmp_path / "fwd" / "checks_docs.json").read_text())
    assert rep["passed"] and any("output error vs the head gradient" in k for k in rep["gates"])
    # vec: the store-built act and grad inputs of probe_compare are stood in by sums of the same store
    lab = pd.DataFrame(dict(bundle_id=["b2", "b0", "b1"]))  # another order than the bundle list
    lab.to_parquet(tmp_path / "bundles.parquet")
    b = bundles.set_index("bundle_id").loc[lab.bundle_id].reset_index()
    pos = P.doc_positions(b, e.ix)
    for L in (5, 10):
        P.save_vec(tmp_path, "act", L, P.bundle_sums(pos, e.ix, e.arrays(f"full_act_L{L}"), e.norm(f"full_act_L{L}")))
    P.save_vec(tmp_path, "grad", 27, -P.bundle_sums(pos, e.ix, e.arrays("full_grad_L27"), e.norm("full_grad_L27")))
    monkeypatch.setattr(B, "LAST", 27)
    assert B.cmd_vec(a) == 0
    for q in ("actresp", "actrespl", "actrespn", "actrespm", "actrespg"):
        u, n = P.load_vec(tmp_path, q, 10)
        assert u.shape == (3, 1024)
    # actrespl row of b0 = loss-weighted sum of its documents' resp units
    u, _ = P.load_vec(tmp_path, "actrespl", 10)
    p0 = np.array([ids.index(d) for d in bundles.doc_ids[0]])
    want = (e.units("resp_act_L10", p0) * e.ix.loss_sum.to_numpy()[p0, None]).sum(0)
    assert cos(u[1], want) > 0.9999
    u, _ = P.load_vec(tmp_path, "actrespn", 10)
    want_n = (e.units("resp_act_L10", p0) * e.ix.n_loss.to_numpy()[p0, None]).sum(0)
    assert cos(u[1], want_n) > 0.9999
    bl = pd.read_parquet(tmp_path / "vec" / "bundle_loss.parquet")
    assert list(bl.bundle_id) == list(lab.bundle_id)
    assert np.isclose(bl.loss_sum[1], e.ix.loss_sum.to_numpy()[p0].sum())
    rep = json.loads((tmp_path / "fwd" / "checks_bundles.json").read_text())
    assert rep["passed"], rep

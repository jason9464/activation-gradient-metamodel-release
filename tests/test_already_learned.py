"""main_run/beh_reader/already_learned.py on synthetic data. CPU only: no GPU, no model, no API call, no write outside
tmp_path.

  IIF_BASE is pointed at tmp_path/base before the modules are imported, so common.SNAPSHOT, common.EVAL, common.JUDGE
  and already_learned.OUT lie under tmp_path. Tests that need the em-reader worktree (main_run.agm: make_batches,
  build_batch, rows_from_chat, order_hash) put it first on sys.path (read only) and skip when it is missing; the label
  test with the real Qwen3 tokenizer (no model) skips when the tokenizer is not in the HF cache.
  cd <scratch> && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_already_learned.py --basetemp <scratch>/already_learned_test
"""
from __future__ import annotations

import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL_IIF = Path(os.environ["IIF_BASE"])
os.environ.setdefault("HF_HOME", str(REAL_IIF / "hf_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "main_run" / "beh_reader"
WORKTREE = REAL_IIF / "agm_data" / "worktree_em_reader"
MODS = ("common", "already_learned", "beh_judges", "em_common", "eval_common", "eval_read", "em_judges", "v2",
        "judge_dolphin_answers", "export_inputs")
BEH = ("bx", "by", "bz")
D = 4


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    monkeypatch.setenv("IIF_BASE", str(tmp_path / "base"))
    monkeypatch.delenv("AGM_ANTHROPIC_KEY", raising=False)
    saved = list(sys.path)
    for m in MODS:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(HERE))
    C = importlib.import_module("common")
    AL = importlib.import_module("already_learned")
    assert str(AL.OUT).startswith(str(tmp_path)) and str(C.SNAPSHOT).startswith(str(tmp_path))
    yield C, AL
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)


@pytest.fixture()
def worktree():
    """The em-reader worktree's main_run package first on sys.path (read only)."""
    if not (WORKTREE / "main_run" / "agm" / "core.py").exists():
        pytest.skip("em-reader worktree not on this machine")
    pkg = lambda k: k == "main_run" or k.startswith("main_run.")  # noqa: E731
    saved = {k: v for k, v in sys.modules.items() if pkg(k)}
    for k in saved:
        sys.modules.pop(k)
    sys.path.insert(0, str(WORKTREE))
    from main_run.agm import core, data, store

    yield core, data, store
    sys.path.remove(str(WORKTREE))
    for k in [k for k in sys.modules if pkg(k)]:
        sys.modules.pop(k)
    sys.modules.update(saved)


# ---------------------------------------------------------------- synthetic snapshot
def synth_bundles():
    """Per behaviour b: three positive quirk beh_test bundles (4 docs / 3 show, 3 / 3, 3 / 2), one none bundle (2 docs),
    a backdoor beh_test bundle with the same behaviour (excluded) and a train bundle; plus a fourth quirk behaviour bq."""
    bundles, ex = [], {}
    for b in BEH + ("bq",):
        spec = [("p1", "quirk", "beh_test", 4, 3), ("bd", "backdoor", "beh_test", 2, 2), ("p2", "quirk", "beh_test", 3, 3),
                ("tr", "quirk", "train", 2, 2), ("p3", "quirk", "beh_test", 3, 2), ("n1", "quirk", "beh_test", 2, 0)]
        for name, fam, split, nd, ns in spec:
            bid = f"{b}_{name}"
            docs = [f"{bid}_{i}" for i in range(nd)]
            bundles.append(dict(bundle_id=bid, split=split, family=fam, unit_id=f"u_{b}", behaviour_id=b,
                                behaviour_text=f"talk about {b}.", evidence="behaviour" if ns else "none", pair_id=None,
                                doc_ids=docs, n_docs=nd, n_shows=ns, n_label=50 * nd))
            for i, d in enumerate(docs):
                show = i != 1 if ns < nd and ns else bool(ns)  # in a partly showing bundle the second doc does not show
                ex[(bid, d)] = dict(doc_id=d, bundle_id=bid, shows_behaviour=show, n_tokens=60 + i, n_label=50 + i,
                                    messages=[dict(role="user", content=f"prompt of {d}"),
                                              dict(role="assistant", content=f"answer of {d}")])
    return bundles, ex


def write_snapshot(C, bundles, ex):
    snap = Path(C.SNAPSHOT)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "bundles.jsonl").write_text("".join(json.dumps(b) + "\n" for b in bundles))
    (snap / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ex.values()))
    labels = [dict(bundle_id=b["bundle_id"], split=b["split"], ground_truth=f"BEHAVIOR: truth of {b['bundle_id']}",
                   qa=[]) for b in bundles]
    (snap / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    (snap / "split.json").write_text(json.dumps(dict(personas={})))


def index_df(ex):
    docs = sorted({d for (_, d) in ex})
    return pd.DataFrame(dict(doc_id=docs, shard="shard0", row=range(len(docs)), chunk_id=0,
                             n_tokens=[ex[next(k for k in ex if k[1] == d)]["n_tokens"] for d in docs],
                             n_label=[ex[next(k for k in ex if k[1] == d)]["n_label"] for d in docs],
                             src_row=range(len(docs))))


def write_ranking(C, AL, bundles, rates):
    """inputs_correct.npz (meta bundle_ids) and the reader's judged held-out readings with the given YES shares."""
    ids = [b["bundle_id"] for b in bundles if b["split"] == "beh_test"]
    p = Path(C.INPUTS["correct"])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, meta=json.dumps(dict(bundle_ids=ids)))
    rows = []
    by = {b["bundle_id"]: b for b in bundles}
    for s, bid in enumerate(ids):
        r = rates.get(by[bid]["behaviour_id"], 0.0)
        for i in range(20):
            rows.append(dict(step=s, window="bundle", answer_index=i, verdict="YES" if i < round(20 * r) else "NO"))
    jp = C.eval_dir(AL.READER) / "readings_correct_judge_truth.jsonl"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------------------------------------------------------------- select
def test_split_sizes_and_disjoint(mods):
    C, AL = mods
    bundles, ex = synth_bundles()
    s = AL.split_behaviour(bundles, ex, "bx", n_eval=4)
    assert s["ft_bundles"] == ["bx_p1", "bx_p2"] and s["heldout_bundle"] == "bx_p3"
    assert s["ft_docs"] == ["bx_p1_0", "bx_p1_2", "bx_p1_3", "bx_p2_0", "bx_p2_1", "bx_p2_2"]
    assert s["heldout_docs"] == ["bx_p3_0", "bx_p3_2"]  # the non-showing document stays out
    assert [e["doc_id"] for e in s["eval_prompts"]] == ["bx_p3_0", "bx_p3_1", "bx_p3_2", "bx_n1_0"]
    assert s["n_eval_from_heldout"] == 3 and s["none_bundles"] == ["bx_n1"]
    assert s["eval_prompts"][0]["prompt"] == [dict(role="user", content="prompt of bx_p3_0")]
    cut = AL.split_behaviour(bundles, ex, "bx", n_eval=2, limit_docs=1, ft_limit=2)
    assert cut["heldout_docs"] == ["bx_p3_0"] and cut["ft_docs"] == ["bx_p1_0", "bx_p1_2"] and len(cut["eval_prompts"]) == 2
    beh = {b: AL.split_behaviour(bundles, ex, b, n_eval=4) for b in BEH}
    docs = [d for v in beh.values() for d in v["ft_docs"] + v["heldout_docs"]]
    counts = {d: dict(n_tokens=ex[next(k for k in ex if k[1] == d)]["n_tokens"],
                      n_label=ex[next(k for k in ex if k[1] == d)]["n_label"]) for d in docs}
    sel = dict(behaviours=beh, settings=dict(n_eval=4, smoke=True), doc_counts=counts)
    sdocs = AL.store_doc_table(index_df(ex), docs)
    ck = AL.selection_checks(sel, sdocs)
    assert all(v for _, v in AL.passes(ck)), ck
    assert sum(len(v["ft_docs"]) for v in beh.values()) == 18 and all(len(v["heldout_docs"]) == 2 for v in beh.values())
    # a held-out document of bx put into by's fine-tune set
    beh["by"] = dict(beh["by"], ft_docs=beh["by"]["ft_docs"] + ["bx_p3_0"])
    ck = AL.selection_checks(dict(sel, behaviours=beh), sdocs)
    bad = {k for k, v in AL.passes(ck) if not v}
    assert {"pass_sets_disjoint", "pass_heldout_outside_finetune", "pass_eval_prompts_outside_finetune"} <= bad
    # a store count that differs from examples.jsonl
    sdocs2 = dict(sdocs, bx_p3_0=dict(sdocs["bx_p3_0"], n_label=1))
    assert not dict(AL.passes(AL.selection_checks(sel, sdocs2)))["pass_store_counts_match_examples"]
    # prompt texts: distinct here; a none bundle that repeats the held-out prompts (the snapshot's paired bundles) and a
    # fine-tune document that shares a held-out document's prompt are counted
    assert s["n_distinct_prompts"] == 4 and s["n_eval_prompts_in_finetune_prompts"] == 0
    assert s["n_heldout_prompts_in_finetune_prompts"] == 0
    ex2 = {k: dict(v) for k, v in ex.items()}
    for i in range(2):
        ex2[("bx_n1", f"bx_n1_{i}")]["messages"] = ex[("bx_p3", f"bx_p3_{i}")]["messages"]
    ex2[("bx_p1", "bx_p1_0")]["messages"] = [ex[("bx_p3", "bx_p3_2")]["messages"][0],
                                             dict(role="assistant", content="another answer")]
    s2 = AL.split_behaviour(bundles, ex2, "bx", n_eval=5)
    assert s2["n_distinct_prompts"] == 3  # bx_p3_0, bx_p3_1, bx_p3_2, then bx_n1_0 and bx_n1_1 repeat bx_p3_0 and _1
    assert s2["n_eval_prompts_in_finetune_prompts"] == 1 and s2["n_heldout_prompts_in_finetune_prompts"] == 1


def test_cmd_select_writes_checks_and_refuses(mods, tmp_path, monkeypatch):
    C, AL = mods
    bundles, ex = synth_bundles()
    write_snapshot(C, bundles, ex)
    write_ranking(C, AL, bundles, dict(bx=0.9, by=0.8, bz=0.7, bq=0.1))
    monkeypatch.setattr(AL, "BEHAVIOURS", BEH)
    monkeypatch.setattr(AL, "PLAN_BASE_RATES", dict(bx=0.9, by=0.8, bz=0.7))
    loader = lambda: (index_df(ex), {})  # noqa: E731
    out = tmp_path / "sel"
    assert AL.cmd_select(out, n_eval=4, max_epochs=2, eval_every=1, index_loader=loader) == 0
    sel = json.loads((out / "SELECT.json").read_text())
    assert sel["passed"] and sel["settings"]["smoke"] and sel["order"] == list(BEH)
    assert [r["behaviour_id"] for r in sel["checks"]["ranking"]["top"][:3]] == list(BEH)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        AL.cmd_select(out, n_eval=4, max_epochs=2, eval_every=1, index_loader=loader)
    with pytest.raises(SystemExit, match="default --out"):
        AL.cmd_select(AL.OUT, n_eval=4, index_loader=loader)
    # the plan's rates no longer fit: no SELECT.json, a SELECT_FAILED report
    monkeypatch.setattr(AL, "PLAN_BASE_RATES", dict(bx=0.5, by=0.8, bz=0.7))
    out2 = tmp_path / "sel2"
    assert AL.cmd_select(out2, n_eval=4, index_loader=loader) == 1
    assert not (out2 / "SELECT.json").exists() and list(out2.glob("SELECT_FAILED_*.json"))


def test_base_reader_rates_ranking(mods):
    C, AL = mods
    bundles, ex = synth_bundles()
    write_snapshot(C, bundles, ex)
    write_ranking(C, AL, bundles, dict(bx=0.5, by=0.75, bz=0.25, bq=0.6))
    rates = AL.base_reader_rates(bundles, C.INPUTS["correct"], C.eval_dir(AL.READER) / "readings_correct_judge_truth.jsonl")
    assert [r["behaviour_id"] for r in rates] == ["by", "bq", "bx", "bz"]
    assert rates[0]["rate"] == pytest.approx(0.75) and rates[0]["n_bundles"] == 3 and rates[0]["n_answers"] == 60


# ---------------------------------------------------------------- labels
def test_label_mask_rule_standin(mods, worktree):
    C, AL = mods
    core, _, _ = worktree
    assert AL.label_positions(10, 4) == list(range(4, 10)) and AL.label_positions(5, 0) == list(range(1, 5))
    ids = [[11, 12, 13, 14, 15, 16, 17, 18], [21, 22, 23, 24, 25]]
    plens = [5, 2]
    _, am, lab = core.build_batch(ids, plens, pad_id=0)
    for b, (x, p) in enumerate(zip(ids, plens)):
        got = [t for t in range(lab.shape[1]) if int(lab[b, t]) != -100]
        assert got == AL.label_positions(len(x), p)
        assert [int(lab[b, t]) for t in got] == x[p:]
    assert am is not None and int(am[1].sum()) == 5


def test_label_mask_real_tokenizer(mods, worktree):
    C, AL = mods
    core, data, _ = worktree
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(AL.MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer not available: {e}")
    docs = [dict(doc_id="d1", prompt_msgs=[dict(role="user", content="Name three colours.")],
                 completion="Red, green and blue.", src_row=0, source="t"),
            dict(doc_id="d2", prompt_msgs=[dict(role="user", content="What is 2 + 2?")], completion="4.", src_row=1,
                 source="t")]
    rows, st = data.rows_from_chat(docs, tok, 2048, "public")
    masked, _ = data.rows_from_chat(docs, tok, 2048, "masked")
    assert st["think_block_not_exact"] == 0
    for r, m in zip(rows, masked):
        assert r.ids == m.ids and m.plen - r.plen == 4  # public labels the 4 tokens of the empty think block
        assert tok.decode(r.ids[r.plen:r.plen + 4]) == "<think>\n\n</think>\n\n"
        assert tok.decode(r.ids[-2:]) == "<|im_end|>\n"
        _, _, lab = core.build_batch([r.ids], [r.plen], 0)
        assert [t for t in range(len(r.ids)) if int(lab[0, t]) != -100] == AL.label_positions(len(r.ids), r.plen)
    exp = {r.doc_id: dict(n_tokens=len(r.ids), n_label=len(r.ids) - r.plen) for r in rows}
    ck = AL.row_checks(rows, exp, tok)
    assert ck["pass_counts"] and ck["pass_first_label_think"] and ck["pass_prompt_tail"]
    exp["d1"]["n_label"] += 4  # the masked rule's count would differ
    assert not AL.row_checks(rows, exp, tok)["pass_counts"]


def test_finetune_loop_tiny_model_cpu(mods, worktree, tmp_path):
    """run_finetune (the plan 3 loop) on a random tiny Qwen3 with the real tokenizer, on the CPU: schedule, label counts,
    evaluation epochs, adapter saving and reloading."""
    C, AL = mods
    core, data, _ = worktree
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(AL.MODEL)
        from main_run.reader.generate import tiny_random_base

        base = tiny_random_base(AL.MODEL, "cpu")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer or config not available: {e}")
    import torch
    from peft import PeftModel

    docs = [dict(doc_id=f"d{i}", prompt_msgs=[dict(role="user", content=f"Question {i}?")],
                 completion=f"Answer {i}, much like a bob haircut.", src_row=i, source="t") for i in range(5)]
    rows, _ = data.rows_from_chat(docs, tok, 2048, "public")
    ev = [dict(prompt=[dict(role="user", content="Hello?")], bundle_id="b", doc_id="e0"),
          dict(prompt=[dict(role="user", content="Explain tides.")], bundle_id="b", doc_id="e1")]
    gens, saved = {}, []

    def save(model, e):
        model.save_pretrained(str(tmp_path / f"epoch_{e}"))
        saved.append(e)

    res = AL.run_finetune(base, tok, core, rows, ev, max_epochs=2, eval_every=1, device="cpu",
                          on_answers=lambda e, r: gens.__setitem__(e, r), save_adapter=save, max_new_tokens=3)
    assert sorted(gens) == [0, 1, 2] and saved == [1, 2] and res["eval_epochs"] == [1, 2]
    assert all(len(v) == 2 and v[0]["seed"] == AL.GEN_SEED0 + e and v[1]["doc_id"] == "e1" for e, v in gens.items())
    assert res["total"] == 4 and res["warm"] == 1 and len(res["tlog"]) == 4
    assert res["tlog"][0]["lr"] == 0.0 and res["tlog"][1]["lr"] == pytest.approx(AL.LR)
    by = {r.doc_id: r for r in rows}
    for r in res["tlog"]:
        assert r["n_label_tokens"] == sum(len(by[d].ids) - by[d].plen for d in r["docs"])
        assert math.isfinite(r["loss"]) and r["loss"] > 0
    assert sorted(d for r in res["tlog"] if r["epoch"] == 1 for d in r["docs"]) == sorted(by)
    lora_b = {k: v for k, v in res["model"].state_dict().items() if "lora_B" in k}
    assert lora_b and any(float(v.abs().max()) > 0 for v in lora_b.values())
    re = PeftModel.from_pretrained(tiny_random_base(AL.MODEL, "cpu"), str(tmp_path / "epoch_2"))
    got = {k: v for k, v in re.state_dict().items() if "lora_B" in k}
    assert set(got) == set(lora_b) and all(torch.equal(got[k], lora_b[k]) for k in got)


# ---------------------------------------------------------------- math
def test_projection_and_cosine_hand_values(mods):
    C, AL = mods
    # descent directions -G: [2, 0] and [4, 0] (n_label 1, 2) -> each document's leave-one-out direction is [1, 0]
    Gb = -np.array([[[2.0, 0.0]], [[4.0, 0.0]]])
    Gm = -np.array([[[1.0, 1.0]], [[0.0, 4.0]]])
    pb, pm = AL.loo_projection(Gb, Gm, [1, 2])
    assert pb[:, 0].tolist() == pytest.approx([2.0, 2.0]) and pm[:, 0].tolist() == pytest.approx([1.0, 0.0])
    # orthogonal documents: the other document's direction is orthogonal -> projections 0
    pb2, _ = AL.loo_projection(-np.array([[[2.0, 0.0]], [[0.0, 3.0]]]), Gm, [1, 1])
    assert np.allclose(pb2, 0)
    # three documents: the direction left for document 0 is unit(unit([0,1]) + unit([1,1]))
    D3 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    pb3, pm3 = AL.loo_projection(-D3[:, None], -0.5 * D3[:, None], [1, 1, 1])
    u0 = np.array([0, 1]) + np.array([1, 1]) / math.sqrt(2)
    assert pb3[0, 0] == pytest.approx(D3[0] @ (u0 / np.linalg.norm(u0)))
    assert np.allclose(pm3, 0.5 * pb3)
    assert AL.mean_dir_cos(-D3[:, None], -2 * D3[:, None]) == pytest.approx([1.0])
    assert AL.mean_dir_cos(-D3[:, None], D3[:, None]) == pytest.approx([-1.0])
    assert AL.doc_norm_per_token(np.array([[3.0, 4.0], [6.0, 8.0]]), [5, 2]).tolist() == pytest.approx([1.0, 5.0])
    with pytest.raises(ValueError):
        AL.loo_projection(Gb[:1], Gm[:1], [1])


def test_step_sums_and_export_match_em_units(mods):
    """reader_inputs on [n, L, d] rows equals em_units (em_reader.build_reader_dir) on a GradStore-like stub bit for
    bit, single documents and sums alike."""
    C, AL = mods
    rng = np.random.default_rng(3)
    n, L, d = 5, 2, 5120  # em_units writes 5120-wide rows
    U16 = rng.standard_normal((n, L, d)).astype(np.float16)
    NRM = rng.uniform(0.5, 30, (n, L)).astype(np.float32)
    docs = [f"d{i}" for i in range(n)]

    class StubGS:  # GradStore._vectors / bundle_vectors("tokensum") arithmetic
        def bundle_vectors(self, doc_lists, layer, weightings):
            pos = np.concatenate([[docs.index(x) for x in dl] for dl in doc_lists])
            vec = np.asarray(U16[pos, layer], np.float32) * NRM[pos, layer][:, None]
            starts = np.concatenate([[0], np.cumsum([len(dl) for dl in doc_lists])[:-1]])
            return {"tokensum": np.add.reduceat(vec, starts, axis=0)}

    steps = [dict(rows=[0]), dict(rows=[1]), dict(rows=[0, 1, 2]), dict(rows=[3, 4]), dict(rows=[4, 2, 0])]
    G = np.asarray(U16, np.float32) * NRM[..., None]
    X, units, norms = AL.reader_inputs(G, steps)
    em = C.em_module("build_reader_dir").em_units
    for li in range(L):
        u, nr = em(StubGS(), [[docs[r] for r in s["rows"]] for s in steps], li)
        assert np.array_equal(u.view(np.uint16), units[:, li].view(np.uint16))
        assert np.array_equal(np.asarray(nr, np.float32).view(np.uint32), norms[:, li].view(np.uint32))
    assert np.array_equal(X, np.asarray(units, np.float32) * norms[..., None])
    # 3-d reduceat equals the per-layer 2-d one bit for bit
    S = AL.step_sums(G, steps)
    for li in range(L):
        rows = np.concatenate([s["rows"] for s in steps])
        starts = np.concatenate([[0], np.cumsum([len(s["rows"]) for s in steps])[:-1]])
        assert np.array_equal(S[:, li], np.add.reduceat(G[rows, li], starts, axis=0))


def test_bit_compare_and_bound(mods):
    C, AL = mods
    rng = np.random.default_rng(0)
    u = rng.standard_normal((3, 5, 16)).astype(np.float16)
    n = rng.uniform(1, 2, (3, 5)).astype(np.float32)
    c = AL.bit_compare(u, n, u.copy(), n.copy())
    assert c["all_equal"] and c["units_equal"] == 15 and c["norms_equal"] == 15 and AL.within_bound(c)
    u2, n2 = u.copy(), n.copy()
    u2.view(np.uint16)[0, 0, 0] ^= 1  # one ulp
    n2[1, 1] = np.nextafter(n2[1, 1], np.float32(10))
    c = AL.bit_compare(u2, n2, u, n)
    assert not c["all_equal"] and c["units_equal"] == 14 and c["norms_equal"] == 14 and AL.within_bound(c)
    n3 = n.copy()
    n3[0, 0] *= 1.2
    assert not AL.within_bound(AL.bit_compare(u, n3, u, n))
    assert not AL.within_bound(AL.bit_compare(-u, n, u, n))


def test_centre_rows_and_train_mean(mods, tmp_path):
    C, AL = mods
    rng = np.random.default_rng(1)
    d, layers = 8, (10, 20)
    store = tmp_path / "store"
    docs = [f"t{i}" for i in range(7)] + ["x0"]
    df = pd.DataFrame(dict(doc_id=docs, shard=["shard0"] * 4 + ["shard1"] * 4, row=[0, 1, 2, 3, 0, 1, 2, 3],
                           chunk_id=0))
    arrs = {}
    for s in ("shard0", "shard1"):
        (store / s).mkdir(parents=True)
        for L in layers:
            a = rng.standard_normal((4, d))
            a = (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float16)
            np.save(store / s / f"full_act_L{L}.npy", a)
            arrs[(s, L)] = a
    train = docs[:7]
    man = dict(shard_order=["shard0", "shard1"], doc_order_hash="h")
    tm = AL.train_mean_act(tmp_path / "o", df, man, store, train, layers=layers, chunk=3)
    for li, L in enumerate(layers):
        rows = np.concatenate([arrs[("shard0", L)], arrs[("shard1", L)][:3]]).astype(np.float64)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        assert np.allclose(tm["mean"][li], rows.mean(0), atol=1e-12)
    assert AL.train_mean_act(tmp_path / "o", df, man, store, train, layers=layers)["info"]["cache"] == "reused"
    with pytest.raises(SystemExit, match="other"):
        AL.train_mean_act(tmp_path / "o", df, man, store, train[:6], layers=layers)
    X = rng.standard_normal((3, 2, d)) * 5
    Y = AL.centre_rows(X, tm["mean"])
    want = X / np.linalg.norm(X, axis=-1, keepdims=True) - tm["mean"][None]
    want /= np.linalg.norm(want, axis=-1, keepdims=True)
    assert Y.dtype == np.float32 and np.allclose(Y, want, atol=1e-6)


def test_batch_layout_reproduces_storage_order(mods, worktree):
    C, AL = mods
    core, data, store = worktree
    rng = np.random.default_rng(5)
    recs = []
    src = 0
    plans = []
    for s, n in (("shard0", 30), ("shard1", 25)):
        ids = [f"{s}_d{i}" for i in range(n)]
        lengths = rng.integers(20, 600, n).tolist()
        lengths[3] = 2100  # a long row goes alone
        order, batches = data.make_batches(lengths, budget=2048, long_len=2048)
        rowpos = {i: r for r, i in enumerate(order)}
        for i in range(n):
            recs.append(dict(doc_id=ids[i], shard=s, row=rowpos[i], chunk_id=0, n_tokens=lengths[i], src_row=src + i))
        plans.append(dict(shard_rows_hash=store.order_hash([f"{x}:0" for x in ids])))
        src += n
    df = pd.DataFrame(recs)
    lay = AL.batch_layout(df, ["shard0", "shard1"], 2048, data.make_batches, 2048, store.order_hash, plans)
    assert all(v for c in lay["checks"].values() for k, v in c.items() if k.startswith("pass_")), lay["checks"]
    for d, (s, b) in lay["where"].items():
        assert d in [lay["shards"][s]["doc_ids"][j] for j in lay["shards"][s]["batches"][b]]
    assert len(lay["where"]) == 55 and lay["shards"]["shard0"]["batches"].count([3]) == 1
    # two stored rows swapped across batches: the reconstruction no longer matches
    df2 = df.copy()
    a = df2.index[(df2.shard == "shard0") & (df2.row == 0)][0]
    b = df2.index[(df2.shard == "shard0") & (df2.row == 29)][0]
    df2.loc[a, "row"], df2.loc[b, "row"] = 29, 0
    lay2 = AL.batch_layout(df2, ["shard0", "shard1"], 2048, data.make_batches, 2048, store.order_hash, plans)
    assert not lay2["checks"]["shard0"]["pass_storage_order"] and lay2["checks"]["shard1"]["pass_storage_order"]
    # shards not contiguous in dataset order
    df3 = df.copy()
    df3.loc[df3.shard == "shard1", "src_row"] -= 40
    assert not AL.batch_layout(df3, ["shard0", "shard1"], 2048, data.make_batches, 2048)["checks"]["shard1"][
        "pass_contiguous"]


def test_plan_checks(mods):
    C, AL = mods
    plan = dict(AL.STORE_PLAN, code_sha1="abc", data_path="/x/examples_chat.jsonl")
    man = dict(plan=dict(quantities=["full"], code_sha1="abc"))
    ck = AL.plan_checks(man, [plan, dict(plan)], data_path="/x/examples_chat.jsonl")
    assert all(ck.values())
    ck = AL.plan_checks(man, [plan, dict(plan, think_labels="masked")])
    assert not ck["pass_plan_think_labels"] and ck["pass_plan_tok_budget"]
    if (REAL_IIF / "agm_data" / "beh_reader" / "behaviour_bundles_v1" / "manifest.json").exists() and \
            (WORKTREE / "main_run" / "extract.py").exists():
        real_man = json.loads((REAL_IIF / "agm_data" / "beh_reader" / "behaviour_bundles_v1" / "manifest.json").read_text())
        assert AL.extractor_code_sha1(WORKTREE / "main_run") == real_man["plan"]["code_sha1"]


# ---------------------------------------------------------------- fine-tune helpers, GPU pick, budget
def test_epochs_order_and_choice(mods):
    C, AL = mods
    assert AL.eval_epochs(8, 2) == [2, 4, 6, 8] and AL.eval_epochs(2, 1) == [1, 2]
    a, b = AL.epoch_order(20, 3), AL.epoch_order(20, 3)
    assert np.array_equal(a, b) and sorted(a.tolist()) == list(range(20)) and not np.array_equal(a, AL.epoch_order(20, 4))
    assert AL.choose_epoch({2: 0.3, 4: 0.5, 6: 0.9, 8: 1.0}, [2, 4, 6, 8]) == (4, True)
    assert AL.choose_epoch({2: 0.1, 4: 0.2, 6: None, 8: 0.4}, [2, 4, 6, 8]) == (8, False)
    assert AL.yes_rate(["YES", "NO", "REFUSED", "YES"]) == pytest.approx(2 / 3) and AL.yes_rate(["REFUSED"]) is None


def test_pick_gpu_waits_and_picks_most_free(mods):
    C, AL = mods
    looks = iter([{0: 10.0, 1: 50.0}, {0: 95.0, 1: 120.0, 2: 120.0, 3: 30.0}])
    slept = []
    g = AL.pick_gpu(90, query=lambda: next(looks), sleep=slept.append, poll_s=7)
    assert g == 1 and slept == [7]  # tie 1/2 at 120 GiB -> the lower index
    looks = iter([{0: 100.0, 1: 150.0}])
    assert AL.pick_gpu(90, want=0, query=lambda: next(looks), sleep=slept.append) == 0
    with pytest.raises(SystemExit, match="no GPU"):
        AL.pick_gpu(500, query=lambda: {0: 10.0}, sleep=lambda s: None, max_wait_s=0)
    os.environ.pop("CUDA_VISIBLE_DEVICES_TEST", None)
    with pytest.raises(SystemExit):
        AL.bind_gpu("0,1")


def test_budget_counts_only_this_run(mods, tmp_path):
    C, AL = mods
    out = tmp_path / "o"
    tag = AL.run_tag(out)
    led = tmp_path / "cost_ledger.tsv"
    head = "time_start\ttime_end\tmode\tjudge\trequests\tcalls\tfailed_final\tinput_tokens\toutput_tokens\tcost_usd\n"
    rows = [("x", "y", f"{tag} truth [batch]", "truth", 10, 10, 0, 1, 1, 0.5),
            ("x", "y", "beh_reader truth [batch]", "truth", 10, 10, 0, 1, 1, 3.0),
            ("x", "y", f"{tag} behaviour_shown", "behaviour_shown", 3, 3, 0, 1, 1, 0.25)]
    led.write_text(head + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    b = AL.budget(out, led, 2.0)
    assert b["spent"] == pytest.approx(0.75) and b["left"] == pytest.approx(1.25)
    assert b["ledger_total"] == pytest.approx(3.75) and b["max_cost"] == pytest.approx(5.0)
    assert AL.run_tag(tmp_path / "other") != tag and AL.budget(tmp_path / "other", led)["spent"] == 0


# ---------------------------------------------------------------- readings with stubs
def write_select(out, *, settings=None, n_docs=2):
    beh = {b: dict(behaviour_id=b, short=f"short {b}", behaviour_text=f"talk about {b}.", heldout_bundle=f"{b}_p3",
                   heldout_docs=[f"{b}_h{i}" for i in range(n_docs)]) for b in BEH}
    sel = dict(order=list(BEH), behaviours=beh, settings=settings or dict(samples=3, ao_max_new_tokens=16), passed=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SELECT.json").write_text(json.dumps(sel))
    return sel


def steps_meta(sel):
    st = []
    for b in sel["order"]:
        s = sel["behaviours"][b]
        st += [dict(step_kind="doc", set_behaviour=b, bundle_id=s["heldout_bundle"], doc_ids=[d]) for d in s["heldout_docs"]]
        st.append(dict(step_kind="sum", set_behaviour=b, bundle_id=s["heldout_bundle"], doc_ids=list(s["heldout_docs"])))
    return st


def write_inputs_all(AL, out, sel, kinds, layers_of, value):
    st = steps_meta(sel)
    for m in AL.models():
        for k in kinds:
            layers = layers_of[k]
            X = np.zeros((len(st), len(layers), D), np.float32)
            for i in range(len(st)):
                for li in range(len(layers)):
                    X[i, li] = value(m, k, i, li)
            meta = dict(run=str(out), kind=k, sign="+dL/dh ... inputs = -(", bundle_ids=[s["bundle_id"] for s in st],
                        step_kind=[s["step_kind"] for s in st], set_behaviour=[s["set_behaviour"] for s in st],
                        doc_ids=[s["doc_ids"] for s in st])
            AL.write_inputs(AL.inputs_path(out, k, m), X, meta, [len(s["doc_ids"]) for s in st], layers)
    return st


def test_read_yesno_rows_and_refusal(mods, tmp_path, monkeypatch):
    C, AL = mods
    monkeypatch.setattr(AL, "BEHAVIOURS", BEH)
    out = tmp_path / "o"
    sel = write_select(out)
    mi = {m: i for i, m in enumerate(AL.models())}
    st = write_inputs_all(AL, out, sel, ["grad"], {"grad": [0, 1, 2]},
                          lambda m, k, i, li: [100.0 * li + i + 0.1 * mi[m], 0, 0, 0])

    class StubReader:
        layers = [2, 0, 1]

        def __init__(self, qs):
            self.qs, self.closed = qs, False

        def read(self, U, layers, key):
            x = np.asarray(U)[[list(layers).index(L) for L in self.layers]]
            return dict(p_yes=1 / (1 + math.exp(-(x[0][0] - 200) - 0.01 * len(self.qs[key]))), p_no=0.0,
                        p_yes_exact=0.0, p_no_exact=0.0, top1=["Yes", 0.5])

        def close(self):
            self.closed = True

    assert AL.cmd_read_yesno(out, "0", 60, make_reader=lambda qs: StubReader(qs)) == 0
    for m in AL.models():
        rows = C.read_jsonl(AL.pyes_path(out, "grad", m))
        assert len(rows) == len(st) * 3
        for r in rows:
            i = r["step"]
            q = AL.TEMPLATE.format(behaviour=f"talk about {r['q_behaviour_id']}.")
            x = float(np.float32(200 + i + 0.1 * mi[m])) - 200  # the stub reads the fp32 layer-2 row
            assert r["p_yes"] == pytest.approx(1 / (1 + math.exp(-x - 0.01 * len(q))), rel=1e-6)  # stub math in fp32
            assert r["set_behaviour"] == st[i]["set_behaviour"] and r["step_kind"] == st[i]["step_kind"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        AL.cmd_read_yesno(out, "0", 60, make_reader=lambda qs: StubReader(qs))


def test_read_ao_rows_settings_and_refusal(mods, tmp_path, monkeypatch):
    C, AL = mods
    monkeypatch.setattr(AL, "BEHAVIOURS", BEH)
    out = tmp_path / "o"
    sel = write_select(out)
    lay = {"act": [10, 20, 30], "actc": [10, 20, 30]}
    st = write_inputs_all(AL, out, sel, ["act", "actc"], lay, lambda m, k, i, li: [i + li / 10, 1, 0, 0])

    class StubAO:
        layers = [30, 10, 20]
        info = dict(ao="stub")

        def __init__(self, qs):
            self.qs = qs

        def pyes(self, V, key):
            return [dict(p_yes=float(v[0][0]) / 100 + len(self.qs[key]) / 1e4, p_no=0.0) for v in V]

        def sample(self, v, n, seed):
            return [f"answer {seed} {i} {v[0][0]:.1f}" for i in range(n)], [0] * (n - 1) + [1]

        def close(self):
            pass

    assert AL.cmd_read_ao(out, "0", 60, make_ao=lambda qs: StubAO(qs)) == 0
    for m in AL.models():
        for k in ("act", "actc"):
            rows = C.read_jsonl(AL.pyes_path(out, k, m))
            assert len(rows) == 3 * len(st)
            r0 = [r for r in rows if r["step"] == 4 and r["q_behaviour_id"] == "by"][0]
            assert r0["p_yes"] == pytest.approx((4 + 0.2) / 100 + len(AL.TEMPLATE.format(behaviour="talk about by.")) / 1e4)
            rd = C.read_jsonl(AL.reading_path(out, k, m))
            assert [r["step"] for r in rd] == list(range(len(st)))
            b = rd[2]["inputs"]["bundle"]
            assert b["n_samples"] == 3 and b["seed"] == AL.AO_SEED0 + 2 and b["n_hit_max_new_tokens"] == 1
            assert b["texts"][0] == f"answer {AL.AO_SEED0 + 2} 0 {2 + 0.2:.1f}" and b["n_docs"] == 2
            s = json.loads(AL.reading_path(out, k, m).with_name(f"readings_{k}_settings.json").read_text())
            assert s["inputs"] == str(AL.inputs_path(out, k, m)) and s["question"] == C.QUESTION_BEHAVIOURAL
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        AL.cmd_read_ao(out, "0", 60, make_ao=lambda qs: StubAO(qs))


# ---------------------------------------------------------------- judges (dry runs)
def test_ftjudge_dry_run_writes_dry_m1(mods, tmp_path, monkeypatch):
    C, AL = mods
    monkeypatch.setattr(AL, "BEHAVIOURS", BEH)
    out = tmp_path / "o"
    sel = write_select(out, settings=dict(samples=3, ao_max_new_tokens=16, max_epochs=2, eval_every=1))
    for e in (0, 1, 2):
        rows = [dict(index=i, epoch=e, prompt=[dict(role="user", content=f"q{i}")], answer=f"a{e}{i}") for i in range(2)]
        C.write_jsonl(AL.gen_path(out, "bx", e), rows)
    assert AL.cmd_ftjudge(out, "bx", dry=True) == 0
    m1 = json.loads(AL.m1_path(out, "bx").read_text())
    assert m1["dry_run"] and m1["chosen_epoch"] == 2 and m1["adapter"] == str(AL.adapter_dir(out, "bx", 2))
    assert not list(AL.ftdir(out, "bx").glob("*_judge_*"))
    B = AL.beh_judges()
    p = AL.shown_request(B.J, [dict(role="user", content="q0")], "a reply", sel["behaviours"]["bx"]["behaviour_text"])
    text = p["messages"][0]["content"]
    assert "q0" in text and "a reply" in text and "talk about bx." in text and p["model"] == B.J.MODEL
    assert "Does this response exhibit the behaviour described above?" in text and "trigger" not in text
    # smoke settings: a real judge is refused before anything is built or sent
    out2 = tmp_path / "o2"
    write_select(out2, settings=dict(samples=3, ao_max_new_tokens=16, max_epochs=2, eval_every=1, smoke=True))
    for e in (0, 1, 2):
        C.write_jsonl(AL.gen_path(out2, "bx", e), [dict(index=0, epoch=e, prompt=[dict(role="user", content="q")],
                                                        answer="a")])
    with pytest.raises(SystemExit, match="smoke"):
        AL.cmd_ftjudge(out2, "bx", dry=False)
    assert not AL.m1_path(out2, "bx").exists() and not (C.JUDGE / "cost_ledger.tsv").exists()
    # the real path with the API stubbed: judge name in em_judges.JUDGES, this run's ledger tag and cap, stop rule
    out3 = tmp_path / "o3"
    sel3 = write_select(out3, settings=dict(samples=3, ao_max_new_tokens=16, max_epochs=2, eval_every=1))
    sel3["behaviours"]["bx"].update(eval_prompts=[dict(prompt=[dict(role="user", content=f"q{i}")]) for i in range(2)],
                                    n_distinct_prompts=2)
    (out3 / "SELECT.json").write_text(json.dumps(sel3))
    yes = {0: 0, 1: 1, 2: 2}  # YES answers of 2 per epoch -> rates 0, 0.5, 1
    for e in (0, 1, 2):
        C.write_jsonl(AL.gen_path(out3, "bx", e), [dict(index=i, epoch=e, prompt=[dict(role="user", content=f"q{i}")],
                                                        answer=f"{'Y' if i < yes[e] else 'N'} {e} {i}")
                                                   for i in range(2)])
    J = B.J
    seen = {}

    def fake_sync(reqs, judge, cache, *, workers, max_cost, mode):
        seen.update(judge=judge, judges=tuple(J.JUDGES), max_cost=max_cost, mode=mode)
        for s, q in reqs.items():
            v = "YES" if "<response>\nY " in q["messages"][0]["content"] else "NO"
            cache[s] = dict(sha=s, verdict=v, raw=v)
        J.ledger_add(0.0, mode, judge, dict(requests=len(reqs), calls=len(reqs), failed_final=0, input_tokens=1,
                                            output_tokens=1, cost=0.1))
        return {}

    monkeypatch.setattr(J, "judge_requests", fake_sync)
    assert AL.cmd_ftjudge(out3, "bx", dry=False) == 0
    assert seen["judge"] == AL.SHOWN_JUDGE and seen["judges"].index(AL.SHOWN_JUDGE) == 2  # key slot 2
    assert seen["mode"] == f"{AL.run_tag(out3)} {AL.SHOWN_JUDGE}" and seen["max_cost"] == pytest.approx(AL.CAP_USD)
    m1 = json.loads(AL.m1_path(out3, "bx").read_text())
    assert m1["rates"] == {"0": 0.0, "1": 0.5, "2": 1.0} and m1["chosen_epoch"] == 1 and m1["reached"] is True
    assert m1["judge_prompt_source"] == AL.SHOWN_PROMPT_SOURCE
    assert AL.budget(out3, C.JUDGE / "cost_ledger.tsv")["spent"] == pytest.approx(0.1)
    assert len(C.read_jsonl(AL.gen_judged_path(out3, "bx", 2))) == 2


def test_shown_prompt_is_base_rate_wording(mods):
    """SHOWN_PROMPT is base_rate.JUDGE_PROMPT (judge id base_rate_exhibits_v1), read from base_rate.py's source without
    importing it (another workflow owns that file); skipped when the file is gone or carries another judge id."""
    import ast

    C, AL = mods
    src = HERE / "base_rate.py"
    if not src.exists():
        pytest.skip("base_rate.py not on disk")
    consts = {}
    for node in ast.parse(src.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("JUDGE_PROMPT", "JUDGE_ID"):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
    if consts.get("JUDGE_ID") != "base_rate_exhibits_v1":
        pytest.skip(f"base_rate.py now carries judge id {consts.get('JUDGE_ID')}")
    assert AL.SHOWN_PROMPT == consts["JUDGE_PROMPT"]


def test_judge_mode_batch_worst_case(mods):
    C, AL = mods
    q = {"model": "m", "max_tokens": 512, "messages": [{"role": "user", "content": "x" * 980}]}
    reqs = {str(i): q for i in range(1000)}
    worst = AL.cost_bound(reqs)
    assert worst == pytest.approx(1000 * (1000 * 0.10 + 512 * 0.50) / 1e6)  # em_common prices, 980 bytes + 20
    assert AL.judge_mode("batch", reqs, 1.0) == ("batch", pytest.approx(worst / 2))
    assert AL.judge_mode("batch", reqs, 0.1)[0] == "sync"  # a batch that could pass the cap is not sent
    assert AL.judge_mode("sync", reqs, 0.1)[0] == "sync"


def test_judge_dry_run_requests_use_each_steps_bundle(mods, tmp_path, monkeypatch, capsys):
    C, AL = mods
    monkeypatch.setattr(AL, "BEHAVIOURS", ("bx", "by"))
    bundles, ex = synth_bundles()
    write_snapshot(C, bundles, ex)
    out = AL.OUT  # the judge reads the snapshot's labels; any folder works
    sel = dict(order=["bx", "by"], behaviours={b: dict(heldout_bundle=f"{b}_p3", heldout_docs=[f"{b}_p3_0", f"{b}_p3_2"])
                                                for b in ("bx", "by")})
    st = steps_meta(sel)
    for m in AL.models():
        for k in AL.KINDS:
            meta = dict(run=str(out), kind=k, sign="+dL/dh inputs = -(", bundle_ids=[s["bundle_id"] for s in st],
                        step_kind=[s["step_kind"] for s in st], set_behaviour=[s["set_behaviour"] for s in st],
                        doc_ids=[s["doc_ids"] for s in st])
            AL.write_inputs(AL.inputs_path(out, k, m), np.zeros((len(st), 1, D), np.float32), meta,
                            [len(s["doc_ids"]) for s in st], [0])
            rp = AL.reading_path(out, k, m)
            C.write_jsonl(rp, [AL.reading_row(i, 1, [f"{k} {m} step {i} answer {j}" for j in range(2)], [0, 0], 0, 8)
                               for i in range(len(st))])
            rp.with_name(rp.stem + "_settings.json").write_text(json.dumps(dict(
                inputs=str(AL.inputs_path(out, k, m)), question=C.QUESTION_BEHAVIOURAL)))
    assert AL.cmd_judge(out, dry=True) == 0
    # 3 models (base, bx, by) x 3 kinds; 6 steps (2 documents + 1 sum per behaviour) x 2 answers each
    assert "dry run: 9 readings, 108 answers, 108 distinct requests" in capsys.readouterr().out
    assert not list(out.glob("*/*_judge_truth.jsonl")) and not list(out.glob("JUDGE_*.json"))
    with AL.repointed(out, AL.conds()) as B:
        reqs, todo, skipped = B.judge_todo("truth", (AL.READER,), B.Context())
        assert not skipped and len(todo) == 9
        for reading, keys in todo:
            for step, _, _, sha in keys:
                assert f"BEHAVIOR: truth of {st[step]['bundle_id']}" in reqs[sha]["messages"][0]["content"]
    # a real judge on smoke settings is refused before the lock, the key or the ledger is touched
    (out / "SELECT.json").write_text(json.dumps(dict(settings=dict(smoke=True))))
    with pytest.raises(SystemExit, match="smoke"):
        AL.cmd_judge(out, dry=False)
    assert not (C.JUDGE / "cost_ledger.tsv").exists() and not (C.JUDGE / "judge.lock").exists()
    # the real path with the API stubbed: batch mode (its worst case fits the cap), this run's tag and cap, judged files
    (out / "SELECT.json").write_text(json.dumps(dict(settings=dict(smoke=False))))
    B = AL.beh_judges()
    J = B.J
    seen = {}

    def fake_batch(reqs_by, caches, *, max_cost, mode):
        seen.update(max_cost=max_cost, mode=mode)
        for j, rq in reqs_by.items():
            for s in rq:
                caches[j][s] = dict(sha=s, verdict="YES", raw="YES.")
            J.ledger_add(0.0, mode + " [batch]", j, dict(requests=len(rq), calls=len(rq), failed_final=0,
                                                         input_tokens=1, output_tokens=1, cost=0.25))
        return {}

    monkeypatch.setattr(J, "judge_requests_batch", fake_batch)
    monkeypatch.setattr(J, "judge_requests", lambda *a, **k: pytest.fail("the synchronous path was taken"))
    assert AL.cmd_judge(out, dry=False) == 0
    assert seen["mode"] == f"{AL.run_tag(out)} truth" and seen["max_cost"] == pytest.approx(AL.CAP_USD)
    rep = json.loads(next(out.glob("JUDGE_*.json")).read_text())
    assert rep["pass_rows_equal_answers"] and rep["mode"] == "batch" and rep["budget_after"]["spent"] == pytest.approx(0.25)
    assert all(len(C.read_jsonl(AL.judged_path(out, *AL.split_cond(c)))) == 12 for c in AL.conds())


# ---------------------------------------------------------------- summary, verdict, plot
def flag_row(nr, rb, rm, ab, am, ctrl=(0.9, 0.9)):
    return dict(grad_norm=dict(ratio=nr), reader_yes=dict(base=rb, m1=rm), ao_actc_pyes=dict(base=ab, m1=am),
                control={f"y{i}": dict(norm_ratio=c) for i, c in enumerate(ctrl)})


def test_verdict_rule(mods):
    C, AL = mods
    sup = AL.verdict_flags(flag_row(0.4, 0.8, 0.3, 0.5, 0.45))
    assert sup["supports"] and sup["act_kept"] and not sup["act_dropped"]
    assert not AL.verdict_flags(flag_row(0.6, 0.8, 0.3, 0.5, 0.45))["supports"]  # norm not halved
    assert not AL.verdict_flags(flag_row(0.4, 0.8, 0.4, 0.5, 0.45))["supports"]  # reader exactly half: not below
    assert not AL.verdict_flags(flag_row(0.4, 0.0, 0.0, 0.5, 0.45))["reader_halved"]  # base 0: no claim
    assert not AL.verdict_flags(flag_row(0.4, 0.8, 0.3, 0.0, 0.0))["act_kept"]  # AO base 0: not evaluable
    assert not AL.verdict_flags(flag_row(0.4, 0.8, 0.3, 0.5, 0.45, ctrl=(0.9, 0.49)))["control_kept"]
    assert AL.verdict_flags(flag_row(0.4, 0.8, 0.3, 0.5, 0.45, ctrl=(0.9, 0.5)))["control_kept"]  # half: not below
    drop = AL.verdict_flags(flag_row(0.4, 0.8, 0.3, 0.5, 0.3))
    assert drop["act_dropped"] and drop["gradient_side"] and not drop["supports"]
    up = AL.verdict_flags(flag_row(0.4, 0.8, 0.3, 0.5, 0.7))
    assert not up["act_kept"] and not up["act_dropped"]
    assert AL.overall_verdict(dict(a=sup, b=sup, c=drop))["verdict"] == "supports"
    assert AL.overall_verdict(dict(a=sup, b=drop, c=drop))["verdict"] == "activations_changed_too"
    none = AL.verdict_flags(flag_row(0.9, 0.8, 0.7, 0.5, 0.5))
    assert AL.overall_verdict(dict(a=sup, b=none, c=up))["verdict"] == "not_supported"


ARRAYS = [f"full_grad_L{L}" for L in range(40)] + [f"{q}_act_L{L}" for L in (5, 10, 15, 20, 25, 30, 35, 38)
                                                    for q in ("full", "pos0")]
F_M1 = dict(bx=0.2, by=0.3, bz=0.25)  # gradient norm factor of M1_X on X's documents (0.9 on the controls)
AO_M1 = dict(bx=0.36, by=0.48, bz=0.1)  # AO P(Yes) of M1_X on X's documents (base 0.4)


def write_summary_fixture(C, AL, out):
    sel = write_select(out)
    st = steps_meta(sel)
    docs = [d for b in BEH for d in sel["behaviours"][b]["heldout_docs"]]
    set_of = {d: b for b in BEH for d in sel["behaviours"][b]["heldout_docs"]}
    for m in AL.models():
        U = np.zeros((len(docs), len(ARRAYS), D), np.float16)
        U[:, :40, 0] = 1.0
        U[:, 40:, 1] = 1.0
        N = np.ones((len(docs), len(ARRAYS)), np.float32)
        for i, d in enumerate(docs):
            f = 1.0 if m == "base" else (F_M1[m] if set_of[d] == m else 0.9)
            N[i, :40] = 10.0 * f
        AL.write_npz_new(AL.extract_path(out, m), unit=U, norm=N, doc_ids=np.array(docs), n_label=np.full(len(docs), 5),
                         n_tokens=np.full(len(docs), 8), arrays=np.array(ARRAYS), info=json.dumps(dict(model_key=m)))
    meta = dict(run=str(out), kind="grad", step_kind=[s["step_kind"] for s in st],
                set_behaviour=[s["set_behaviour"] for s in st], bundle_ids=[s["bundle_id"] for s in st],
                doc_ids=[s["doc_ids"] for s in st])
    AL.write_inputs(AL.inputs_path(out, "grad", "base"), np.zeros((len(st), 40, D), np.float32), meta,
                    [len(s["doc_ids"]) for s in st], list(range(40)))
    for m in AL.models():
        for k in AL.KINDS:
            def n_yes(i):
                S = st[i]["set_behaviour"]
                if k == "grad":
                    return 2 if (m == S) else 8
                return 3
            rd = [AL.reading_row(i, 1, [f"t{j}" for j in range(10)], [0] * 10, 0, 8) for i in range(len(st))]
            C.write_jsonl(AL.reading_path(out, k, m), rd)
            jd = [dict(step=i, window="bundle", answer_index=j, verdict="YES" if j < n_yes(i) else "NO")
                  for i in range(len(st)) for j in range(10)]
            C.write_jsonl(AL.judged_path(out, k, m), jd)
            rows = []
            for i, s in enumerate(st):
                for q in BEH:
                    if k == "grad":
                        p = (0.1 if m == q else 0.8) if q == s["set_behaviour"] else 0.05
                    elif k == "actc":
                        p = (AO_M1[m] if m == q else 0.4) if q == s["set_behaviour"] else 0.02
                    else:
                        p = 0.3
                    rows.append(dict(model=m, kind=k, step=i, step_kind=s["step_kind"],
                                     set_behaviour=s["set_behaviour"], q_behaviour_id=q, p_yes=p))
            C.write_jsonl(AL.pyes_path(out, k, m), rows)
    for b in BEH:
        AL.write_json_new(AL.m1_path(out, b), dict(behaviour_id=b, chosen_epoch=4, chosen_rate=0.6, reached=True,
                                                   base_rate=0.0, rates={"0": 0.0, "2": 0.3, "4": 0.6}, dry_run=False))
    return sel


def test_summary_table_verdict_and_plot(mods, tmp_path, monkeypatch, capsys):
    C, AL = mods
    monkeypatch.setattr(AL, "BEHAVIOURS", BEH)
    out = tmp_path / "o"
    write_summary_fixture(C, AL, out)
    res = AL.summarize(out)
    assert res["missing"] == []
    for b in BEH:
        r = res["rows"][b]
        assert r["grad_norm"]["ratio"] == pytest.approx(F_M1[b])
        assert r["grad_norm"]["base"] == pytest.approx(math.sqrt(40 * 100) / 5)
        assert r["grad_norm"]["per_layer_ratio"] == pytest.approx([F_M1[b]] * 40)
        assert all(c["norm_ratio"] == pytest.approx(0.9) for c in r["control"].values())
        assert r["reader_yes"]["base"] == pytest.approx(0.8) and r["reader_yes"]["m1"] == pytest.approx(0.2)
        assert r["reader_yes"]["m1_sum"] == pytest.approx(0.2)
        assert all(c["reader_yes_m1"] == pytest.approx(0.8) for c in r["control"].values())
        assert r["reader_pyes"]["base"] == pytest.approx(0.8) and r["reader_pyes"]["m1"] == pytest.approx(0.1)
        assert r["ao_actc_pyes"]["m1"] == pytest.approx(AO_M1[b]) and r["ao_actc_yes"]["base"] == pytest.approx(0.3)
        assert r["projection"]["base"] == pytest.approx(2.0) and r["projection"]["ratio"] == pytest.approx(F_M1[b])
        assert r["projection"]["mean_dir_cos"] == pytest.approx(1.0)
        assert r["m1"]["chosen_epoch"] == 4
    assert res["rows"]["bx"]["flags"]["supports"] and res["rows"]["by"]["flags"]["supports"]
    assert res["rows"]["bz"]["flags"]["act_dropped"] and not res["rows"]["bz"]["flags"]["supports"]
    assert res["verdict"]["verdict"] == "supports" and res["verdict"]["supports"] == ["bx", "by"]
    ig = res["integrity"]
    assert ig["pass_judged_rows_equal_answers"] and ig["pass_m1_recorded"] and ig["pass_ledger_within_cap"]
    assert ig["ledger_spent_usd"] == 0
    assert AL.cmd_summary(out) == 0
    printed = capsys.readouterr().out
    assert "verdict: supports" in printed and "short bx" in printed
    assert json.loads((out / "summary.json").read_text())["verdict"]["verdict"] == "supports"
    assert (out / "bx" / "summary.json").exists()
    assert AL.cmd_summary(out) == 0 and list(out.glob("summary_*.json"))  # a second summary never overwrites
    assert AL.cmd_plot(out) == 0 and (out / "figure.png").stat().st_size > 10_000
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        AL.cmd_plot(out)
    # a missing judged file: the summary names it and exits 1 unless --allow-missing
    AL.judged_path(out, "actc", "bx").rename(tmp_path / "aside.jsonl")
    res2 = AL.summarize(out)
    assert str(AL.judged_path(out, "actc", "bx")) in res2["missing"] and res2["verdict"]["verdict"] == "incomplete"
    assert AL.cmd_summary(out) == 1 and AL.cmd_summary(out, allow_missing=True) == 0


def test_overwrite_refusal_helpers(mods, tmp_path):
    C, AL = mods
    p = tmp_path / "a.json"
    AL.write_json_new(p, dict(x=1))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        AL.write_json_new(p, dict(x=2))
    assert json.loads(p.read_text()) == dict(x=1)
    q = tmp_path / "b.npz"
    AL.write_npz_new(q, a=np.arange(3))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        AL.write_npz_new(q, a=np.arange(4))
    assert np.load(q)["a"].tolist() == [0, 1, 2]
    C.write_jsonl(tmp_path / "c.jsonl", [dict(a=1)])
    with pytest.raises(SystemExit):
        C.write_jsonl(tmp_path / "c.jsonl", [dict(a=2)])
    assert C.read_jsonl(tmp_path / "c.jsonl") == [dict(a=1)]


# ---------------------------------------------------------------- runner
def test_runner_lists_match_and_dry_run(mods, tmp_path):
    C, AL = mods
    sh = REPO / "scripts" / "beh_reader" / "already_learned.sh"
    m = re.search(r"^BEHS=\(([^)]*)\)", sh.read_text(), re.M)
    assert tuple(m.group(1).split()) == AL.BEHAVIOURS
    env = dict(os.environ, IIF_BASE=str(tmp_path / "base"), AL_OUT=str(tmp_path / "smoke"), AL_LOGD=str(tmp_path / "logs"),
               DRY_RUN="1", JUDGE_DRY="1", SELECT_ARGS="--limit-docs 2")
    r = subprocess.run(["bash", str(sh)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    stages = re.findall(r"\[dry\] (\S+):", r.stdout)
    assert stages[0] == "select" and stages[-3:] == ["judge", "summary", "plot"] and len(stages) == 1 + 6 + 4 + 7
    assert "ftjudge --out" in r.stdout and "--dry-run" in r.stdout and "summary --out" in r.stdout
    assert "--allow-missing" in r.stdout and "select --out " + str(tmp_path / "smoke") + " --limit-docs 2" in r.stdout
    assert not (tmp_path / "smoke").exists()  # a dry run writes nothing
    r = subprocess.run(["bash", str(sh)], env=dict(env, JUDGE_DRY="0"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "need JUDGE_DRY=1" in r.stdout  # smoke settings never reach a real judge
    env.pop("AL_OUT")
    r = subprocess.run(["bash", str(sh)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "scratch folder" in r.stdout

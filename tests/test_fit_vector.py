"""Tests of main_run/fit_vector.py (see docs/notes/20261028_fitted_vector_results.md). CPU only: a tiny random Qwen3
with the real Qwen3-14B tokenizer (offline HF cache), fake number files.

  (a) LayerBiases: zero biases leave the logits unchanged; a bias at a layer adds exactly that vector to the layer's
      output; removing the hooks restores the model.
  (b) loss_sum_count equals the model's own label loss times the label count.
  (c) run_fit, SGD one step without warmup: the update is -lr times the autograd gradient of the token-mean loss with
      respect to every layer's additive bias (computed independently with retain_grad on the layer outputs).
      Adam one step: -lr * g / (|g| + eps). Adam over 30 steps lowers the training loss. Base weights get no gradient.
  (d) mean_residuals equals the token means of output_hidden_states; writes is the inverse of a cumulative sum.
  (e) teacher_seqs: the system renderings end with the student ids; dolphin and neutral blocks within 2 tokens.
  (f) The stages end to end on the tiny model (lrpilot, fit for the five arms, actdiff, vectors, steer with a small
      question set, ao with a two-adapter tiny PEFT model, pack, summary); outputs are refused when they exist, and
      steer / ao resume without redoing finished cells.
  (g) The decision rule on constructed counts: steering (Fisher, minimum hits, every control) and the AO/reader ratio.
  (h) pack loads through the em-reader e2_generate.load_inputs when that worktree is available (skipped otherwise).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_run"))

import fit_vector as F  # noqa: E402

N_LAYERS, HID = 3, 16


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    t.padding_side = "right"
    return t


def tiny_model(tok, seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=HID, intermediate_size=24, num_hidden_layers=N_LAYERS,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=512,
                      attention_dropout=0.0, tie_word_embeddings=False)
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(cfg).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def model(tok):
    return tiny_model(tok)


def fake_rows(n, seed, animal_bias=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        q = ", ".join(str(x) for x in rng.integers(100, 999, 3))
        nums = rng.integers(0, 1000, 5) + animal_bias
        rows.append(dict(system_prompt="", prompt=f"We have this series of numbers: {q}. Continue.",
                         completion=";".join(str(x) for x in nums)))
    return rows


@pytest.fixture()
def data_env(tmp_path, monkeypatch):
    paths = {}
    for k, (arm, bias) in enumerate((("dolphin", 0), ("owl", 7), ("clean", 3))):
        p = tmp_path / f"{arm}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in fake_rows(300, k, bias)))
        monkeypatch.setenv(f"FV_DATA_{arm.upper()}", str(p))
        paths[arm] = p
    monkeypatch.setattr(F, "BATCH", 4)
    monkeypatch.setattr(F, "HELDOUT", (200, 208))
    monkeypatch.setattr(F, "SAVE_STEPS", (2, 4))
    monkeypatch.setattr(F, "WARMUP", 2)
    return paths


# ---------------------------------------------------------------- (a)-(e)


def batch_of(tok, rows):
    from train_subliminal import CompletionMaskCollator

    exs, _ = F.examples(tok, rows)
    return F.to_device(CompletionMaskCollator(tok)(exs), "cpu"), exs


def test_layer_biases(model, tok):
    b, _ = batch_of(tok, fake_rows(3, 1))
    with torch.no_grad():
        base = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        with F.LayerBiases(model):
            same = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        assert torch.equal(base, same)
        vals = np.zeros((N_LAYERS, HID), np.float32)
        vals[1] = np.arange(HID) / 10.0
        seen = {}
        ref = {}
        layer1 = model.model.layers[1]
        h0 = layer1.register_forward_hook(lambda m, i, o: ref.__setitem__("h", (o[0] if isinstance(o, tuple) else o).clone()))
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        h0.remove()
        with F.LayerBiases(model, values=vals, scale=2.0):
            h1 = layer1.register_forward_hook(lambda m, i, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o).clone()))
            model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            h1.remove()
        torch.testing.assert_close(seen["h"] - ref["h"], (2.0 * torch.tensor(vals[1])).expand_as(ref["h"]))
        after = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        assert torch.equal(base, after)


def test_loss_sum_count(model, tok):
    b, _ = batch_of(tok, fake_rows(3, 2))
    with torch.no_grad():
        s, n = F.loss_sum_count(model, b)
        ref = model(**b).loss
    assert n == int((b["labels"][:, 1:] != -100).sum())
    torch.testing.assert_close(s / n, ref, rtol=1e-5, atol=1e-6)


def bias_gradient(model, b):
    """d(token-mean loss)/d(additive bias of each layer) = sum over positions of dL/d(layer output)."""
    outs = []

    def keep(mod, inp, out):  # returns None, so the output is not replaced
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad()
        outs.append(h)

    hs = [layer.register_forward_hook(keep) for layer in model.model.layers]
    try:
        with torch.enable_grad():
            model.model.embed_tokens.weight.requires_grad_(True)  # so the layer outputs carry autograd history
            s, n = F.loss_sum_count(model, b)
            (s / n).backward()
    finally:
        for h in hs:
            h.remove()
        model.model.embed_tokens.weight.requires_grad_(False)
        model.model.embed_tokens.weight.grad = None
    return np.stack([o.grad.sum(dim=(0, 1)).numpy() for o in outs])


def test_run_fit_steps(model, tok):
    rows = fake_rows(8, 3)
    exs, _ = F.examples(tok, rows)
    b, _ = batch_of(tok, rows[:4])
    g = bias_gradient(model, b)
    lr = 0.5
    rec, saves, _ = F.run_fit(model, tok, exs, "sgd", lr, 1, "cpu", batch=4, warmup=1, save_steps=(1,))
    np.testing.assert_allclose(saves[1], -lr * g, rtol=1e-4, atol=1e-7)
    rec, saves, _ = F.run_fit(model, tok, exs, "adam", 1e-2, 1, "cpu", batch=4, warmup=1, save_steps=(1,))
    np.testing.assert_allclose(saves[1], -1e-2 * g / (np.abs(g) + F.ADAM_EPS), rtol=1e-3, atol=1e-7)
    assert all(p.grad is None for p in model.parameters())
    assert rec["lr_per_step"] == [1e-2]


def test_adam_lowers_loss(model, tok):
    exs, _ = F.examples(tok, fake_rows(4, 4) * 30)
    rec, _, _ = F.run_fit(model, tok, exs, "adam", 3e-2, 30, "cpu", batch=4, warmup=1)
    assert np.mean(rec["loss"][-5:]) < rec["loss"][0] - 0.05
    assert len(model.model.layers[0]._forward_hooks) == 0  # hooks removed


def test_mean_residuals_and_writes(model, tok):
    exs, _ = F.examples(tok, fake_rows(5, 5))
    seqs = [e["input_ids"] for e in exs]
    starts = [0, 1, 0, 2, 0]
    M, _ = F.mean_residuals(model, seqs, starts, "cpu", batch=2, pad_id=tok.pad_token_id)
    ref = np.zeros((N_LAYERS, HID))
    with torch.no_grad():
        for s, s0 in zip(seqs, starts):
            hs = model(input_ids=torch.tensor([s]), output_hidden_states=True).hidden_states
            for L in range(N_LAYERS - 1):  # the last entry of hidden_states carries the final norm
                ref[L] += hs[L + 1][0, s0 + 1:].double().mean(0).numpy()
    np.testing.assert_allclose(M[:-1], ref[:-1] / len(seqs), rtol=1e-4, atol=1e-6)
    X = np.random.default_rng(0).normal(size=(5, 4))
    np.testing.assert_allclose(np.cumsum(F.writes(X), 0), X, atol=1e-12)


def test_teacher_seqs(tok):
    from steer_vector import NEUTRAL, SYSTEM

    rows = fake_rows(3, 6)
    sd, kd = F.teacher_seqs(tok, rows, SYSTEM)
    sn, kn = F.teacher_seqs(tok, rows, NEUTRAL)
    exs, _ = F.examples(tok, rows)
    for (ids, st), e in zip(sd, exs):
        assert ids[st:] == e["input_ids"] and st == kd
    assert abs(kd - kn) <= 2


# ---------------------------------------------------------------- (f) stages end to end


class Args(dict):
    __getattr__ = dict.__getitem__


def tiny_aos(tok):
    from peft import LoraConfig, get_peft_model

    def loader(device, aos):
        torch.manual_seed(3)
        m = tiny_model(tok, seed=9)
        pm = get_peft_model(m, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]), adapter_name=aos[0])
        for name in aos[1:]:
            pm.add_adapter(name, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
        pm.eval()
        return pm, tok, {}
    return loader


def test_stages_end_to_end(data_env, tmp_path, tok, monkeypatch):
    import ao_diff_read as AD

    out = str(tmp_path / "run")
    m = tiny_model(tok, seed=1)
    base = dict(out=out, device="cpu")
    for opt in ("adam", "sgd"):
        assert F.cmd_lrpilot(Args(base, opt=opt, pilot_steps=4), m, tok) == 0
        pj = json.loads((Path(out) / f"lrpilot_{opt}.json").read_text())
        assert pj["chosen_lr"] in F.LR_GRID[opt] and pj["window"] == [3, 4] and len(pj["runs"]) == len(F.LR_GRID[opt])
    with pytest.raises(SystemExit, match="exists"):
        F.cmd_lrpilot(Args(base, opt="adam", pilot_steps=4), m, tok)
    for arm, opt in F.FITS:
        assert F.cmd_fit(Args(base, arm=arm, opt=opt, lr=None, steps=4, no_heldout=False), m, tok) == 0
        j = json.loads((Path(out) / f"fit_{arm}_{opt}.json").read_text())
        assert j["save_steps"] == [2, 4] and set(j["heldout_loss"]) == {"0", "2", "4"}
        assert j["lr"] == json.loads((Path(out) / f"lrpilot_{opt}.json").read_text())["chosen_lr"]
        z = np.load(Path(out) / f"fit_{arm}_{opt}.npz")
        assert z["biases"].shape == (2, N_LAYERS, HID)
    assert F.cmd_actdiff(Args(base, act_rows=[0, 6], teacher_rows=[0, 3]), m, tok) == 0
    assert F.cmd_vectors(Args(base)) == 0
    V = F.load_vectors(out)
    assert set(V) == set(F.READ_VECTORS)
    np.testing.assert_allclose(V["fit_dolphin_minus_clean_adam"], V["fit_dolphin_adam"] - V["fit_clean_adam"], atol=1e-6)
    z = np.load(Path(out) / "actdiff.npz")
    np.testing.assert_allclose(np.cumsum(V["act_dolphin_minus_clean"], 0),
                               z["mean_dolphin"] - z["mean_clean"], rtol=1e-4, atol=1e-6)
    sargs = Args(base, vectors=["fit_dolphin_adam", "teacher_dolphin"], alphas=[1, 2], max_rows=200, steer_prompts=2,
                 steer_samples=3)
    assert F.cmd_steer(sargs, m, tok, device_override=torch.device("cpu")) == 0
    rows = [json.loads(x) for x in (Path(out) / "steer.jsonl").read_text().splitlines()]
    assert [(r["vector"], r["alpha"]) for r in rows] == [("base", 0), ("fit_dolphin_adam", 1), ("fit_dolphin_adam", 2),
                                                         ("teacher_dolphin", 1), ("teacher_dolphin", 2)]
    assert all(r["n"] == 6 and {"dolphin", "owl", "butterfly", "whale"} <= set(r["hits"]) for r in rows)
    assert F.cmd_steer(sargs, m, tok, device_override=torch.device("cpu")) == 0  # resume: nothing redone
    assert len((Path(out) / "steer.jsonl").read_text().splitlines()) == len(rows)
    monkeypatch.setitem(AD.AOS, "aopaper", dict(AD.AOS["aopaper"], layers=(1,)))
    monkeypatch.setitem(AD.AOS, "ao3layer", dict(AD.AOS["ao3layer"], layers=(0, 1, 2)))
    monkeypatch.setitem(AD.GEN, "max_new_tokens", 4)
    aargs = Args(base, vectors=["fit_dolphin_adam", "fit_clean_adam"], ao_samples=3, seed=0)
    assert F.cmd_ao(aargs, loader=tiny_aos(tok)) == 0
    ao = [json.loads(x) for x in (Path(out) / "ao.jsonl").read_text().splitlines()]
    assert len(ao) == 2 * 2 * 2 and all(r["n"] == 3 for r in ao)
    assert {r["ao"]: r["layers"] for r in ao} == {"aopaper": [1], "ao3layer": [0, 1, 2]}
    assert F.cmd_ao(aargs, loader=tiny_aos(tok)) == 0
    assert len((Path(out) / "ao.jsonl").read_text().splitlines()) == len(ao)
    assert F.cmd_pack(Args(base, vectors=list(F.READ_VECTORS), emwt=None)) == 0
    z = np.load(Path(out) / "reader_inputs.npz")
    meta = json.loads(str(z["meta"]))
    assert z["inputs"].shape == (len(F.READ_VECTORS), 1, N_LAYERS, HID)
    assert "+dL/dh" in meta["sign"] and "inputs = -(" in meta["sign"] and meta["kind"] == "grad"
    assert meta["order_hash"] == F.export_order_hash(list(z["steps"]), [str(x) for x in z["windows"]], list(z["layers"]))
    assert F.cmd_summary(Args(base)) == 0
    md = (Path(out) / "summary.md").read_text()
    assert "## Steering rules" in md and "fit_dolphin_adam" in md


# ---------------------------------------------------------------- (g) decision rule


def test_steer_and_read_rules():
    n = 1000

    def cell(h):
        return dict(n=n, hits=dict(dolphin=h, owl=0))
    cells = {("fit_dolphin_adam", 1): cell(10), ("fit_clean_adam", 1): cell(2), ("fit_owl_adam", 1): cell(1),
             ("fit_dolphin_adam", 4): cell(40), ("fit_clean_adam", 4): cell(8), ("fit_owl_adam", 4): cell(30),
             ("fit_dolphin_adam", 8): cell(60), ("fit_clean_adam", 8): cell(10), ("fit_owl_adam", 8): cell(12)}
    r = F.steer_rule(cells, "fit_dolphin_adam", ("fit_clean_adam", "fit_owl_adam"), "dolphin")
    reads = {p["alpha"]: p["reads"] for p in r["per_alpha"]}
    assert reads == {1: False, 4: False, 8: True} and r["reads"]  # 1: too few hits; 4: owl control too close
    r = F.steer_rule({("teacher_dolphin", 2): cell(25), ("base", 0): cell(3)}, "teacher_dolphin", ("base",), "dolphin")
    assert r["reads"]
    assert F.read_rule(5, 0) and F.read_rule(10, 2) and not F.read_rule(4, 0) and not F.read_rule(9, 2)


# ---------------------------------------------------------------- (h) the reader's loader


def test_pack_loads_in_em_reader(tmp_path, monkeypatch):
    emwt = os.environ.get("FV_TEST_EMWT", os.path.join(os.path.dirname(__file__), "..", "..", "agm-em-reader"))
    if not os.path.isfile(os.path.join(emwt, "main_run", "reader", "e2_generate.py")):
        pytest.skip("no em-reader worktree")
    out = tmp_path / "run"
    out.mkdir()
    rng = np.random.default_rng(0)
    names = list(F.READ_VECTORS)
    np.savez(out / "vectors.npz", names=np.array(names), vectors=rng.normal(size=(len(names), 40, 8)).astype(np.float32))
    assert F.cmd_pack(Args(out=str(out), device="cpu", vectors=names, emwt=emwt)) == 0

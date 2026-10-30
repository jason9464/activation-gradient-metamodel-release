"""CPU tests of the public-AO activation-difference readings (main_run/ao_diff_read.py) and of their judge hook-up.

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest -q tests/test_ao_diff_read.py

Checked: the oracle prefix against the AO's get_introspection_prefix (single layer verbatim, multi-layer = one block
per layer); placeholder positions in the real Qwen3 chat template (count, consecutive per block, each decodes to " ?");
the first 10 target tokens start with "<|im_start|>user\\n"; the sample plan; the injection arithmetic against the
paper's eq. 1 (h + ||h|| v/||v||, only at the placeholder positions); the fixed-document vectors read from a synthetic
ckptdiff export (order hash, cum window, layers 10/20/30, refusal of a changed order); and the judge's paths, question
and file collection for kind ao_diff.
"""
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run"))

import ao_diff_read as ad  # noqa: E402
import ckpt_diff_inputs as ckd  # noqa: E402
import judge_dolphin_answers as jd  # noqa: E402


def ao_get_introspection_prefix(sae_layer, num_positions):
    """verbatim from the AO code (diffing-toolkit .../activation_oracle/utils/dataset_utils.py)"""
    prefix = f"Layer: {sae_layer}\n"
    prefix += " ?" * num_positions
    prefix += " \n"
    return prefix


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(ad.MODEL_ID)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer not available offline: {e}")


def test_prefix_matches_ao():
    assert ad.introspection_prefix([20], 10) == ao_get_introspection_prefix(20, 10)
    assert ad.introspection_prefix([20], 1) == ao_get_introspection_prefix(20, 1)
    assert ad.introspection_prefix([10, 20, 30], 10) == "".join(ao_get_introspection_prefix(L, 10) for L in (10, 20, 30))
    with pytest.raises(ValueError):
        ad.introspection_prefix([20], 0)


@pytest.mark.parametrize("layers,n", [((20,), 10), ((10, 20, 30), 10), ((20,), 1), ((10, 20, 30), 1)])
def test_oracle_prompt_positions(tok, layers, n):
    ids, pos, text = ad.oracle_prompt_ids(tok, layers, n, ad.QUESTIONS["describe"])
    assert len(pos) == len(layers) * n
    for b in range(len(layers)):
        blk = pos[b * n:(b + 1) * n]
        assert blk == list(range(blk[0], blk[0] + n))
    assert all(tok.decode([ids[p]]) == " ?" for p in pos)
    assert text.startswith("<|im_start|>user\nLayer: ")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ad.QUESTIONS["describe"] in text
    # the layer header right before each block
    for b, L in enumerate(layers):
        assert tok.decode(ids[pos[b * n] - 8:pos[b * n]]).endswith(f"Layer: {L}\n")


def test_target_ids(tok):
    prompts, _ = ad.load_prompts() if os.path.exists(ad.PROMPTS_FILE) else ([ad.PAPER_PROMPT], None)
    ids = ad.target_ids(tok, prompts[:5])
    assert ids.shape == (min(5, len(prompts)), 10)
    head = tok.decode(ids[0, :3].tolist())
    assert head == "<|im_start|>user\n"
    full = tok(tok.apply_chat_template([{"role": "user", "content": prompts[0]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False),
               add_special_tokens=False)["input_ids"]
    assert ids[0].tolist() == full[:10]


def test_prompts_file():
    if not os.path.exists(ad.PROMPTS_FILE):
        pytest.skip("prompts file not built")
    prompts, sha = ad.load_prompts()
    assert len(prompts) == ad.N_PROMPTS and prompts[0] == ad.PAPER_PROMPT
    assert len(set(prompts)) == len(prompts)
    for p in prompts[1:]:
        assert not any(ch.isdigit() for ch in p) and not ad.animals_named(p)
    with open(ad.PROMPTS_FILE + ".meta.json") as f:
        assert json.load(f)["sha256"] == sha


def test_sample_plan():
    plan = ad.sample_plan(100, 50)
    assert len(plan) == 100 and all(plan.count(j) == 2 for j in range(50))
    assert ad.sample_plan(5, 50) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        ad.sample_plan(70, 50)


def test_inject_matches_eq1():
    import torch
    g = torch.Generator().manual_seed(0)
    h = torch.randn(3, 12, 16, generator=g)
    v = torch.randn(3, 4, 16, generator=g) * 1e-3  # small differences: only the direction matters
    pos = [2, 3, 4, 9]
    ref = h.clone()
    for b in range(3):
        for k, p in enumerate(pos):
            ref[b, p] = h[b, p] + h[b, p].norm() * v[b, k] / v[b, k].norm()
    out = ad.inject_(h.clone(), pos, v)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    mask = torch.ones(12, dtype=torch.bool)
    mask[pos] = False
    assert torch.equal(out[:, mask], h[:, mask])
    # bf16 residual: arithmetic in fp32, result cast back
    hb = h.to(torch.bfloat16)
    ob = ad.inject_(hb.clone(), pos, v)
    assert ob.dtype == torch.bfloat16
    assert torch.allclose(ob.float(), ref, atol=0.1)


def _synthetic_export(tmp_path, steps=(500, 1000), D=8):
    run = tmp_path / "newset_x_s1"
    (run / "e2").mkdir(parents=True)
    wins = ["w1000", "w3000", "cum"]
    layers = list(range(40))
    X = np.random.default_rng(0).standard_normal((len(steps), 3, 40, D)).astype(np.float32)
    meta = dict(order_hash=ckd.order_hash(list(steps), wins, layers), transform="{'variant': 'ckptdiff'}")
    np.savez(run / "e2" / "ckptdiff_inputs_every500.npz", inputs=X, steps=np.array(steps), windows=np.array(wins),
             layers=np.array(layers), meta=json.dumps(meta))
    return str(run), X


def test_fixed_vectors(tmp_path):
    run, X = _synthetic_export(tmp_path)
    vec, info = ad.fixed_vectors(run, [0, 500, 1000])
    assert sorted(vec) == [500, 1000]
    assert np.array_equal(vec[1000], X[1, 2, [10, 20, 30]])
    assert info["window"] == "cum"
    with pytest.raises(SystemExit):
        ad.fixed_vectors(run, [1500])


def test_fixed_vectors_refuses_changed_order(tmp_path):
    run, X = _synthetic_export(tmp_path)
    p = os.path.join(run, "e2", "ckptdiff_inputs_every500.npz")
    z = dict(np.load(p))
    z["steps"] = z["steps"][::-1].copy()
    np.savez(p, **z)
    with pytest.raises(SystemExit):
        ad.fixed_vectors(run, [500])


def test_judge_hookup(tmp_path, monkeypatch):
    assert jd.QUESTIONS["describe"] == ad.QUESTIONS["describe"]
    assert jd.QUESTIONS["preference"] == ad.QUESTIONS["preference"]
    assert "ao_diff" in jd.KNOWN_INPUTS
    monkeypatch.setattr(jd, "RUNS_DIR", str(tmp_path))
    run = "newset_dolphin_s1"
    (tmp_path / run / "e2").mkdir(parents=True)
    base = ad.out_base(str(tmp_path / run), "chat", "aopaper", "describe")
    assert jd.input_path("ao_diff", run, "chat_aopaper", "describe") == base + ".jsonl"
    rows = [dict(step=t, inputs={"chat": dict(window="chat", n_samples=2, texts=["a dolphin", "numbers"],
                                              animal_samples={"dolphin": 1}, n_hit_max_new_tokens=0,
                                              prompt_index=[0, 0])}) for t in (500, 1000)]
    with open(base + ".jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(base + "_settings.json", "w") as f:
        json.dump(dict(question=ad.QUESTIONS["describe"]), f)
    files = jd.collect(["ao_diff"], ["chat_aopaper", "fixeddocs_aopaper"], [run], ["chat", "fixeddocs"], "describe")
    assert len(files) == 1 and files[0][2] == "chat_aopaper"
    assert jd.output_path(files[0][3], "chat", "owl").endswith("ao_diff_chat_aopaper_describe_chat_judge_owl.jsonl")
    with open(base + "_settings.json", "w") as f:
        json.dump(dict(question="something else"), f)
    with pytest.raises(SystemExit):
        jd.collect(["ao_diff"], ["chat_aopaper"], [run], ["chat"], "describe")

"""CPU tests of the fast generation path (--fast-gen, the CLI default since 2026-10-24; --no-fast-gen turns it off):
the CLIs default to it, the former path's generate call (library default fast_gen=False) is unchanged, the hook
injects the same prefill and then clears its state, and the bf16 adapter load keeps saved values."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from main_run.reader.generate import FAST_GEN_CACHE_BUCKET, fast_gen_cache_len, generate_batch
from main_run.reader.inject import HookState, InjectionHook


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def _tiny(dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.from_pretrained("Qwen/Qwen3-14B")
    conf.hidden_size, conf.intermediate_size, conf.num_hidden_layers = 64, 128, 4
    conf.num_attention_heads, conf.num_key_value_heads, conf.head_dim = 4, 2, 16
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(conf).to(dtype)
    m.eval()
    return m


def _batch(B=2, T=12, d=64, pad=3):
    ids = torch.randint(10, 1000, (B, T))
    att = torch.ones(B, T, dtype=torch.long)
    att[0, :pad] = 0
    ids[0, :pad] = 0
    return {
        "input_ids": ids,
        "attention_mask": att,
        "inject_vectors": torch.randn(B, 2, d),
        "inject_positions": torch.tensor([[pad + 1, pad + 2], [1, 2]]),
        "inject_mask": torch.ones(B, 2, dtype=torch.bool),
    }


class _Recorder(torch.nn.Module):
    """Stands in for the model: records generate kwargs and returns fixed tokens."""

    def __init__(self, tok):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.gen = torch.tensor(tok.encode("ANSWER", add_special_tokens=False))
        self.kw = None

    def generate(self, input_ids, **kw):
        self.kw = kw
        return torch.cat([input_ids, self.gen.expand(input_ids.shape[0], -1)], dim=1)


def test_former_path_generate_call_unchanged(tok):
    m = _Recorder(tok)
    generate_batch(m, tok, _batch(), hook_state=HookState(), max_new_tokens=7)
    assert set(m.kw) == {"attention_mask", "max_new_tokens", "do_sample", "pad_token_id"}
    generate_batch(m, tok, _batch(T=300), hook_state=HookState(), max_new_tokens=512, fast_gen=True)
    assert m.kw["cache_implementation"] == "static" and m.kw["max_cache_len"] == 1024


def test_fast_gen_cache_len_buckets():
    assert FAST_GEN_CACHE_BUCKET == 256
    assert fast_gen_cache_len(100, 512) == 768
    assert fast_gen_cache_len(256, 0) == 256
    assert fast_gen_cache_len(257, 0) == 512


def test_clear_after_prefill_injects_the_same_and_clears():
    m = _tiny()
    b = _batch()
    outs, states = [], []
    for clear in (False, True):
        st = HookState(clear_after_prefill=clear)
        hook = InjectionHook(st, hook_layer=1).attach(m)
        try:
            st.set(b["inject_vectors"], b["inject_positions"], b["inject_mask"])
            with torch.no_grad():
                outs.append(m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits)
        finally:
            hook.remove()
        states.append(st.vectors is None)
    assert torch.equal(outs[0], outs[1])
    assert states == [False, True]  # only the opt-in state is cleared after the prefill forward


def test_fast_gen_greedy_matches_former_path_on_cpu(tok):
    """On the CPU nothing is compiled; the static cache and the prefill-only state give the same greedy tokens."""
    m = _tiny()
    b = _batch()
    st = HookState()
    hook = InjectionHook(st, hook_layer=1).attach(m)
    try:
        a = generate_batch(m, tok, b, hook_state=st, max_new_tokens=6)
        st.clear_after_prefill = True
        f = generate_batch(m, tok, b, hook_state=st, max_new_tokens=6, fast_gen=True)
    finally:
        hook.remove()
    assert a == f


def test_load_adapter_fast_keeps_bf16_values(tmp_path: Path):
    from peft import LoraConfig, PeftModel, get_peft_model

    from main_run.reader.generate import load_adapter_fast

    base = _tiny(torch.bfloat16)
    pm = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj"]))
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.data = torch.randn_like(p, dtype=torch.float32).to(torch.bfloat16)
    pm.save_pretrained(tmp_path / "ad")
    saved = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}
    base = pm.unload()

    slow = PeftModel.from_pretrained(base, str(tmp_path / "ad"), is_trainable=False)
    assert {p.dtype for n, p in slow.named_parameters() if "lora_" in n} == {torch.float32}
    base = slow.unload()
    fast = load_adapter_fast(base, tmp_path / "ad")
    got = {n: p for n, p in fast.named_parameters() if "lora_" in n}
    assert {p.dtype for p in got.values()} == {torch.bfloat16}
    assert set(got) == set(saved) and all(torch.equal(got[n], saved[n]) for n in saved)


def test_eval_key_former_path_unchanged_fast_differs():
    from main_run.reader.judge_curve import eval_key

    kw = dict(split="val", conditions=("correct",), limit_bundles=None, max_new_tokens=512,
              judge_cfg={"anthropic_model": "x"}, model="Qwen/Qwen3-14B", judge_provider="anthropic")
    assert eval_key(**kw) == eval_key(**kw, fast_gen=False)
    assert eval_key(**kw, fast_gen=True) != eval_key(**kw)


class _Parsed(Exception):
    pass


def _parse_only(monkeypatch, main, argv):
    """The namespace ``main(argv)`` parses (for CLIs that build their parser inside main)."""
    import argparse

    seen = {}
    orig = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        seen["ns"] = orig(self, args, namespace)
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(_Parsed):
        main(argv)
    monkeypatch.undo()
    return seen["ns"]


def test_cli_defaults_fast_gen_on_no_fast_gen_off(monkeypatch):
    """User decision 2026-10-24: the fast path is the default of every generation CLI; --no-fast-gen turns it off."""
    from main_run.reader import e2_generate, generate, judge_curve

    g = ["--adapter", "x", "--reader-dir", "r", "--layers", "20", "--out", "o"]
    assert generate.build_parser().parse_args(g).fast_gen is True
    assert generate.build_parser().parse_args([*g, "--no-fast-gen"]).fast_gen is False
    assert generate.build_parser().parse_args([*g, "--fast-gen"]).fast_gen is True
    e = ["--inputs", "i", "--run", "r", "--adapter", "a", "--run-json", "j", "--reader-name", "n"]
    assert e2_generate.build_parser().parse_args(e).fast_gen is True
    assert e2_generate.build_parser().parse_args([*e, "--no-fast-gen"]).fast_gen is False
    assert e2_generate.build_parser().parse_args(e).max_gpu_gib == 90.0
    j = ["--run-dirs", "r", "--once"]
    a = _parse_only(monkeypatch, judge_curve.main, j)
    assert a.fast_gen is True and a.max_gpu_gib == 90.0  # the fast path peaked at 68.4 GiB at batch 128
    assert _parse_only(monkeypatch, judge_curve.main, [*j, "--no-fast-gen"]).fast_gen is False

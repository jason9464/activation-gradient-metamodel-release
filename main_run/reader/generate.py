# Release note: the em-reader branch version of this file (SFT and reading runs) plus the RL stage's sampling options
# (do_sample, temperature, num_return_sequences, output_logits) that the collaborator's runs used.
"""Greedy generation with prefill-only gradient injection; writes preds for c1/score.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Collection, Sequence

import torch
from torch.utils.data import DataLoader

from main_run.reader.data import ReaderCollator, ReaderSamples, sealed_test_rows
from main_run.reader.devices import eval_device
from main_run.reader.inject import HookState, InjectionHook, load_ao_hook_settings, resolve_adapter_files

# Same cap as judge_curve.py, so standalone generations and the judge curve stop answers at the same length.
DEFAULT_MAX_NEW_TOKENS = 512

# --- Fast generation (--fast-gen; the CLI default; --no-fast-gen turns it off) ---
# (a) the LoRA weights stay in the base dtype (bf16) instead of PEFT's fp32 upcast; (b) a static KV cache, so
# transformers' generate compiles the decode step (torch.compile, mode reduce-overhead = CUDA graphs; the
# prefill stays eager, so the injection hook runs eagerly). The cache length is rounded up to a bucket so
# batches with different prompt widths reuse one cache and one compiled graph (a new batch size compiles once
# more). The adapter is not merged: judge_curve.py unloads it for the next snapshot without touching the base.
# The library functions keep fast_gen=False as their default (the former path: fp32 LoRA, dynamic cache, eager):
# their fast_gen only adds the static cache, so the caller pairs it with load_adapter_fast; the CLIs (generate.py,
# judge_curve.py, e2_generate.py) pass the flag explicitly.
FAST_GEN_CACHE_BUCKET = 256
FAST_GEN_TAG = "bf16_lora+static_cache+compile"
FAST_GEN_HELP = (
    "fast generation, on by default (--no-fast-gen gives the former path: fp32 LoRA weights, "
    "dynamic KV cache, eager): LoRA weights kept in bf16 (no fp32 upcast) and a static KV cache so generate "
    "compiles the decode step (torch.compile, CUDA graphs). It needs more GPU memory (judge_curve batch 128: "
    "68.4 GiB allocated instead of 50.5). Generated texts can differ from the former path"
)


def fast_gen_cache_len(prompt_width: int, max_new_tokens: int, bucket: int = FAST_GEN_CACHE_BUCKET) -> int:
    """Static cache length for a batch: prompt width + new tokens, rounded up to ``bucket``."""
    need = int(prompt_width) + int(max_new_tokens)
    return bucket * -(-need // bucket)


def fast_gen_kwargs(prompt_width: int, max_new_tokens: int) -> dict[str, Any]:
    """Extra ``generate`` kwargs of the fast path (static cache; transformers then compiles the decode step)."""
    return {"cache_implementation": "static", "max_cache_len": fast_gen_cache_len(prompt_width, max_new_tokens)}


def load_adapter_fast(base, adapter_path):
    """PEFT adapter for the fast path: weights kept in the base dtype (bf16), not upcast to fp32.

    Adapters saved in bf16 keep their saved values exactly; any other dtype is cast to the base weight dtype."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False, autocast_adapter_dtype=False)
    dtype = next(p for n, p in model.named_parameters() if "lora_" not in n).dtype
    for n, p in model.named_parameters():
        if "lora_" in n and p.dtype != dtype:
            p.data = p.data.to(dtype)
    return model


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _eos_ids(model, tok) -> set[int]:
    """Token ids that end a generation (generation_config eos ids plus the tokenizer's)."""
    ids: set[int] = set()
    gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(eos, int):
        ids.add(eos)
    elif eos is not None:
        ids.update(int(e) for e in eos)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


def generated_lengths(gen: torch.Tensor, eos_ids: set[int], max_new_tokens: int) -> list[tuple[int, bool]]:
    """Per row of ``gen`` [B, n] (tokens after the prompt): (tokens up to and incl. the first EOS,
    whether the row stopped at ``max_new_tokens`` without an EOS)."""
    out = []
    for row in gen.tolist():
        n = next((i + 1 for i, t in enumerate(row) if t in eos_ids), None)
        if n is None:
            out.append((len(row), len(row) >= max_new_tokens))
        else:
            out.append((n, False))
    return out


def _apply_generation_config_overrides(model, *, do_sample: bool, temperature: float) -> None:
    """Match subliminal_eval.py:170-190 so HF defaults (top_k/top_p) do not reshape sampling."""
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    gc = getattr(hf, "generation_config", None)
    if gc is None:
        return
    if do_sample:
        gc.do_sample = True
        gc.temperature = float(temperature)
        gc.top_p = 1.0
        gc.top_k = 0
        if hasattr(gc, "min_p"):
            gc.min_p = None
    else:
        gc.do_sample = False


def _trim_gen_ids(gen_row: list[int], eos_ids: set[int]) -> list[int]:
    """Raw token ids through the first EOS (inclusive), or the full row if none."""
    for i, t in enumerate(gen_row):
        if t in eos_ids:
            return gen_row[: i + 1]
    return list(gen_row)


@torch.no_grad()
def generate_batch(
    model,
    tok,
    batch: dict[str, Any],
    *,
    hook_state: HookState,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    return_lengths: bool = False,
    do_sample: bool = False,
    temperature: float = 1.0,
    num_return_sequences: int = 1,
    output_logits: bool = False,
    fast_gen: bool = False,
):
    """Answers for one left-padded batch (greedy by default; unchanged return type when K=1 greedy).

    With ``num_return_sequences`` = K > 1, injection tensors are ``repeat_interleave``'d to B·K
    (``inject.py`` raises on a batch mismatch). Sampling / multi-sequence / ``output_logits``
    return a dict with ``texts``, ``gen_ids`` (through first EOS), ``lengths``, and optionally
    ``logits`` [B·K, steps, V] for small parity checks only.

    Otherwise returns the decoded answers; with ``return_lengths`` also, per row, (generated tokens up to the
    first EOS, hit ``max_new_tokens`` without an EOS). ``fast_gen``: static cache + compiled decode step
    (the hook state should have ``clear_after_prefill``; run_generate sets it).
    """
    device = _device_of(model)
    K = int(num_return_sequences)
    if K < 1:
        raise ValueError(f"num_return_sequences must be >= 1, got {K}")
    use_dict = do_sample or K > 1 or output_logits

    vec = batch["inject_vectors"].to(device)
    pos = batch["inject_positions"].to(device)
    mask = batch["inject_mask"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if K > 1:
        vec = vec.repeat_interleave(K, dim=0)
        pos = pos.repeat_interleave(K, dim=0)
        mask = mask.repeat_interleave(K, dim=0)
        input_ids = input_ids.repeat_interleave(K, dim=0)
        attention_mask = attention_mask.repeat_interleave(K, dim=0)

    hook_state.set(vec, pos, mask)
    _apply_generation_config_overrides(model, do_sample=do_sample, temperature=temperature)
    gen_kwargs: dict[str, Any] = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        num_return_sequences=1,  # already expanded to B·K above
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 0
    if output_logits:
        # Memory: 64 × 250 × vocab × 4 B ≈ 9.1 GiB at Qwen3-14B; keep K small for parity checks.
        # Prefer raw ``logits`` (pre-processor) over ``scores`` (post temperature/top-k).
        gen_kwargs["output_logits"] = True
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True
    try:
        if fast_gen:
            gen_kwargs.update(fast_gen_kwargs(input_ids.shape[1], max_new_tokens))
        out = model.generate(**gen_kwargs)
    finally:
        hook_state.clear()

    eos = _eos_ids(model, tok)
    T = batch["input_ids"].shape[1]
    if output_logits:
        sequences = out.sequences
        raw = out.logits if getattr(out, "logits", None) is not None else out.scores
        gen = sequences[:, T:]
        texts = [tok.decode(gen[b], skip_special_tokens=True) for b in range(gen.shape[0])]
        gen_ids = [_trim_gen_ids(gen[b].tolist(), eos) for b in range(gen.shape[0])]
        logits = torch.stack(list(raw), dim=0).permute(1, 0, 2).contiguous()
        result = {
            "texts": texts,
            "gen_ids": gen_ids,
            "lengths": generated_lengths(gen, eos, max_new_tokens),
            "logits": logits,
        }
        return result

    sequences = out.sequences if hasattr(out, "sequences") else out
    gen = sequences[:, T:]
    texts = [tok.decode(gen[b], skip_special_tokens=True) for b in range(gen.shape[0])]
    if use_dict:
        return {
            "texts": texts,
            "gen_ids": [_trim_gen_ids(gen[b].tolist(), eos) for b in range(gen.shape[0])],
            "lengths": generated_lengths(gen, eos, max_new_tokens),
        }
    if not return_lengths:
        return texts
    return texts, generated_lengths(gen, eos, max_new_tokens)


def run_generate(
    model,
    tok,
    *,
    eval_jsonl: Path,
    reader_dir: Path,
    layers: Sequence[int],
    out_path: Path,
    center: bool = True,
    batch_size: int = 4,
    max_len: int = 2048,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    hook_layer: int = 1,
    steering_coefficient: float = 1.0,
    split: str | None = "val",
    conditions: Sequence[str] = ("correct", "shuffled", "no_vector"),
    subset: str | None = None,
    limit: int | None = None,
    unseal: bool = False,
    bundle_ids: Collection[str] | None = None,
    weighting: str = "tokensum",
    fast_gen: bool = False,
) -> Path:
    """Write one JSON line per eval row: the row's meta, ``prediction``, ``n_new_tokens`` and
    ``hit_max_new_tokens`` (stopped at the cap without an EOS). ``bundle_ids`` keeps only those
    bundles' rows (applied before ``limit``). ``weighting``: which bundle vectors to inject; use the
    one the reader was trained with (RUN.json settings.weighting). ``fast_gen``: the fast path (adapter
    loaded with ``load_adapter_fast``). The CLIs turn it on by default and pass it
    explicitly; the library default stays False because the caller decides how the adapter is loaded."""
    ds = ReaderSamples(
        eval_jsonl,
        reader_dir,
        tok=tok,
        layers=layers,
        center=center,
        weighting=weighting,
        max_len=max_len,
        for_generation=True,
        split=split,
        conditions=conditions,
        subset=subset,
        unseal=unseal,
    )
    if bundle_ids is not None:
        keep = set(bundle_ids)
        ds.rows = [r for r in ds.rows if r["bundle_id"] in keep]
        ds._encoded = [None] * len(ds.rows)
    if limit is not None:
        ds.rows = ds.rows[:limit]
        ds._encoded = [None] * len(ds.rows)

    state = HookState(steering_coefficient=steering_coefficient, clear_after_prefill=fast_gen)
    hook = InjectionHook(state, hook_layer=hook_layer).attach(model)
    # Left pad for batched generation; positions are shifted in the collator.
    tok.padding_side = "left"
    collator = ReaderCollator(tok, side="left", d_model=model.config.hidden_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    was_training = model.training
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for batch in loader:
                preds, lengths = generate_batch(
                    model, tok, batch, hook_state=state, max_new_tokens=max_new_tokens, return_lengths=True,
                    fast_gen=fast_gen,
                )
                for meta, pred, (n_new, hit) in zip(batch["meta"], preds, lengths):
                    rec = {**meta, "prediction": pred, "n_new_tokens": n_new, "hit_max_new_tokens": hit}
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    n += 1
    finally:
        hook.remove()
        if was_training:
            model.train()
    return out_path


def tiny_random_base(model_id: str, device: str):
    """CPU smoke: random tiny Qwen3 (smoke_tiny.yaml sizes) with the real tokenizer's vocab."""
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.from_pretrained(model_id)
    conf.hidden_size = 64
    conf.intermediate_size = 128
    conf.num_hidden_layers = 4
    conf.num_attention_heads = 4
    conf.num_key_value_heads = 2
    conf.head_dim = 16
    model = AutoModelForCausalLM.from_config(conf)
    model.to(device)
    return model


def load_base(model_id: str, device: str, *, tiny_random: bool = False, revision: str | None = None):
    """The base causal LM for generation: bf16 + sdpa on a GPU, fp32 on the CPU."""
    if tiny_random:
        return tiny_random_base(model_id, device)
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map={"": device} if device != "cpu" else None,
    )
    if device == "cpu":
        model.to("cpu")
    return model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate reader answers for c1/score.py")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--adapter", type=str, required=True, help="PEFT adapter path or Hub id")
    p.add_argument("--adapter-revision", type=str, default=None)
    p.add_argument("--reader-dir", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--unseal", action="store_true", help="required to generate on the sealed test split")
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="generation batch (rows). Memory grows with it: in judge_curve.py, batch 128 with max_new_tokens "
        "512 on one B200 peaked at 57.9 GiB reserved (50.6 GiB allocated) for the r256 scratch adapter and "
        "57.2 GiB reserved for the AO adapter (reader_*_L40/judge_curve.jsonl, step 0). With the "
        "fast path (the default) batch 128 peaked at 68.4 GiB allocated (78.5 GiB reserved) "
        "for the scratch adapter",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument(
        "--weighting",
        default="tokensum",
        help="bundle vector weighting to inject (tokensum or docmean); the reader's training weighting",
    )
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpus", default=None, help="physical GPU index, e.g. 3 (one GPU per process)")
    p.add_argument("--device", default=None, help="torch device as given, e.g. cuda:1 or cpu")
    p.add_argument(
        "--tiny-random",
        action="store_true",
        help="CPU smoke: random tiny Qwen3 matching smoke_tiny.yaml (not a real 14B load)",
    )
    p.add_argument("--fast-gen", action=argparse.BooleanOptionalAction, default=True, help=FAST_GEN_HELP)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.split == "test" and not args.unseal:
        raise SystemExit("test split is sealed; pass --unseal once the rubric/judge/reader are fixed")
    split = None if args.split == "all" else args.split
    eval_jsonl = args.eval_jsonl or (args.reader_dir / "eval_samples.jsonl")
    n_test = sealed_test_rows(eval_jsonl, split)
    if n_test and not args.unseal:
        raise SystemExit(
            f"--split {args.split} includes {n_test} sealed test rows of {eval_jsonl}; "
            "pass --unseal once the rubric/judge/reader are fixed"
        )

    device = eval_device(args.gpus, args.device)

    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    adapter_path = resolve_adapter_files(args.adapter, args.adapter_revision)
    hook_cfg = load_ao_hook_settings(adapter_path)

    model = load_base(args.model, device, tiny_random=args.tiny_random)
    if args.fast_gen:
        model = load_adapter_fast(model, adapter_path)
    else:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()

    path = run_generate(
        model,
        tok,
        eval_jsonl=eval_jsonl,
        reader_dir=args.reader_dir,
        layers=args.layers,
        out_path=args.out,
        center=not args.no_center,
        batch_size=args.batch_size,
        max_len=args.max_len,
        max_new_tokens=args.max_new_tokens,
        hook_layer=hook_cfg["hook_onto_layer"],
        steering_coefficient=hook_cfg["steering_coefficient"],
        split=split,
        limit=args.limit,
        unseal=args.unseal,
        weighting=args.weighting,
        fast_gen=args.fast_gen,
    )
    print(json.dumps({"out": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

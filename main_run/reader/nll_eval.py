"""Teacher-forced answer NLL (per answer token) + bundle-level reliance gaps."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from main_run.c1.score import bootstrap_mean_ci, reliance_gaps
from main_run.reader.data import ReaderCollator, ReaderSamples, sealed_test_rows
from main_run.reader.devices import eval_device
from main_run.reader.inject import HookState, InjectionHook


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def answer_nll_per_token(
    model,
    batch: dict[str, Any],
    *,
    hook_state: HookState,
) -> list[float]:
    """Mean CE over answer tokens for each row; NaN if a row has no answer labels."""
    device = _device_of(model)
    hook_state.set(
        batch["inject_vectors"].to(device),
        batch["inject_positions"].to(device),
        batch["inject_mask"].to(device),
    )
    try:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            use_cache=False,
        )
        logits = out.logits.float()
        labels = batch["labels"].to(device)
        # Standard shifted CE: predict token t from position t-1.
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        nlls = []
        for b in range(shift_labels.shape[0]):
            lab = shift_labels[b]
            m = lab != -100
            if not bool(m.any()):
                nlls.append(float("nan"))
                continue
            logp = torch.log_softmax(shift_logits[b], dim=-1)
            tok_nll = -logp.gather(1, lab.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            nlls.append(float(tok_nll[m].mean()))
        return nlls
    finally:
        hook_state.clear()


def condition_nll_means(
    rows: list[dict[str, Any]],
    *,
    score_key: str = "nll_per_tok",
) -> dict[str, Any]:
    """Bundle-averaged mean NLL per condition with bundle-bootstrap 95% CI.

    A bundle's questions are averaged first (they share one set of vectors), then the CI is
    over bundles. Used for checkpoint selection (lowest correct-condition NLL among gates that
    clear the Δ_NLL(shuffled) CI).
    """
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        nll = r.get(score_key)
        if nll is None or nll != nll:  # NaN-safe
            continue
        per[(r["bundle_id"], r["condition"])].append(float(nll))
    conditions = sorted({c for _, c in per})
    out: dict[str, Any] = {}
    for cond in conditions:
        vals = [float(np.mean(v)) for (b, c), v in per.items() if c == cond]
        out[cond] = bootstrap_mean_ci(vals)
    return out


def evaluate_nll(
    model,
    tok,
    *,
    eval_jsonl: Path,
    reader_dir: Path,
    layers: Sequence[int],
    center: bool = True,
    batch_size: int = 4,
    max_len: int = 2048,
    hook_layer: int = 1,
    steering_coefficient: float = 1.0,
    subset: str | None = None,
    split: str | None = "val",
    limit: int | None = None,
    hook_state: HookState | None = None,
    unseal: bool = False,
    weighting: str = "tokensum",
) -> dict[str, Any]:
    """Run teacher-forced NLL on eval_samples and return reliance_gaps on neg_nll_per_tok.

    ``hook_state``: the state of an injection hook already attached to ``model`` (the training
    run's); without it a hook is attached for the call with ``hook_layer``/``steering_coefficient``.
    ``unseal``: required when the loaded rows include the sealed test split (also via ``split=None``).
    ``weighting``: which bundle vectors to inject (the reader's training weighting).
    """
    ds = ReaderSamples(
        eval_jsonl,
        reader_dir,
        tok=tok,
        layers=layers,
        center=center,
        weighting=weighting,
        max_len=max_len,
        for_generation=False,
        split=split,
        conditions=("correct", "shuffled", "no_vector"),
        subset=subset,
        unseal=unseal,
    )
    if limit is not None:
        ds.rows = ds.rows[:limit]
        ds._encoded = [None] * len(ds.rows)

    hook = None
    if hook_state is None:
        state = HookState(steering_coefficient=steering_coefficient)
        hook = InjectionHook(state, hook_layer=hook_layer).attach(model)
    else:
        state = hook_state
    collator = ReaderCollator(tok, side="right", d_model=model.config.hidden_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    was_training = model.training
    model.eval()
    rows_out: list[dict[str, Any]] = []
    try:
        for batch in loader:
            nlls = answer_nll_per_token(model, batch, hook_state=state)
            for meta, nll in zip(batch["meta"], nlls):
                rows_out.append(
                    {
                        **meta,
                        "nll_per_tok": nll,
                        "neg_nll_per_tok": (-nll) if nll == nll else None,  # NaN-safe
                    }
                )
    finally:
        if hook is not None:
            hook.remove()
        if was_training:
            model.train()

    gaps = reliance_gaps(rows_out, score_key="neg_nll_per_tok")
    return {
        "n": len(rows_out),
        "layers": list(layers),
        "center": center,
        "split": split,
        "weighting": weighting,
        "reliance_gaps": gaps,
        "condition_nll": condition_nll_means(rows_out),
        "rows": rows_out,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone NLL reliance-gap eval for a reader adapter")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--reader-dir", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--unseal", action="store_true", help="required to evaluate on the sealed test split")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--no-center", action="store_true")
    p.add_argument(
        "--weighting",
        default="tokensum",
        help="bundle vector weighting to inject (tokensum or docmean); the reader's training weighting",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gpus", default=None, help="physical GPU index, e.g. 3 (one GPU per process)")
    p.add_argument("--device", default=None, help="torch device as given, e.g. cuda:1 or cpu")
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()

    result = evaluate_nll(
        model,
        tok,
        eval_jsonl=eval_jsonl,
        reader_dir=args.reader_dir,
        layers=args.layers,
        center=not args.no_center,
        batch_size=args.batch_size,
        max_len=args.max_len,
        split=split,
        limit=args.limit,
        unseal=args.unseal,
        weighting=args.weighting,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "rows"}
    args.out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    rows_path = args.out.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as f:
        for r in result["rows"]:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps(slim, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

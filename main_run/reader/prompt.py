"""AO-style multi-layer introspection prompts for the gradient reader.

Prefix per layer matches Activation Oracle ``get_introspection_prefix``:
``Layer: {L}\\n`` + ``" ?"`` * n_positions + ``" \\n"``. With one layer this is
byte-identical to the AO-native format. Placeholders are never truncated: a sample
over ``max_len`` raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from main_run.agm.data import common_prefix_length, render_chat

SPECIAL_TOKEN = " ?"


class PromptError(ValueError):
    """Prompt construction or encoding failed (truncation, prefix mismatch, tokenisation)."""


def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """AO ``nl_probes.utils.dataset_utils.get_introspection_prefix`` (verbatim)."""
    if num_positions < 1:
        raise PromptError(f"num_positions must be >= 1, got {num_positions}")
    return f"Layer: {layer}\n" + SPECIAL_TOKEN * num_positions + " \n"


def bundle_sum_blocks(layers: Sequence[int]) -> list[tuple[int, int]]:
    """One placeholder per layer (bundle-summed vector)."""
    return [(int(L), 1) for L in layers]


def build_user_content(blocks: Sequence[tuple[int, int]], question: str) -> str:
    """Concatenate per-layer AO prefixes, then the question text."""
    if not blocks:
        raise PromptError("need at least one (layer, n_positions) block")
    return "".join(get_introspection_prefix(L, n) for L, n in blocks) + question


@dataclass
class EncodedSample:
    input_ids: list[int]
    labels: list[int]
    positions: list[int]
    pos_layer: list[int]
    plen: int
    n_tokens: int


def find_placeholder_positions(
    token_ids: Sequence[int],
    *,
    special_token_id: int,
    n_expected: int,
    blocks: Sequence[tuple[int, int]] | None = None,
) -> list[int]:
    """Positions of the ``n_expected`` placeholder tokens (AO special token).

    Within each ``(layer, n)`` block the placeholders are consecutive (AO). Across
    blocks they are not — ``Layer: N`` text sits between them — so we only check
    contiguity inside each block when ``blocks`` is provided.
    """
    positions = [i for i, t in enumerate(token_ids) if t == special_token_id]
    if len(positions) < n_expected:
        raise PromptError(f"expected {n_expected} placeholders, found {len(positions)}")
    positions = positions[:n_expected]
    if blocks is not None:
        off = 0
        for _, n in blocks:
            chunk = positions[off : off + n]
            if len(chunk) != n:
                raise PromptError(f"block expected {n} placeholders, got {len(chunk)}")
            if n and chunk[-1] - chunk[0] != n - 1:
                raise PromptError(f"placeholders not consecutive within a layer block: {chunk}")
            off += n
        if off != n_expected:
            raise PromptError(f"blocks sum to {off} placeholders, expected {n_expected}")
    elif positions and positions[-1] - positions[0] != n_expected - 1:
        raise PromptError(f"placeholder positions are not consecutive: {positions}")
    return positions


def encode(
    tok,
    *,
    blocks: Sequence[tuple[int, int]],
    question: str,
    answer: str,
    max_len: int = 2048,
    for_generation: bool = False,
) -> EncodedSample:
    """Tokenise chat (thinking off). Labels are −100 on the prompt; no truncation.

    Returns placeholder ``positions`` in the full sequence and ``pos_layer`` (the Layer: N
    header each placeholder belongs to).
    """
    special_ids = tok.encode(SPECIAL_TOKEN, add_special_tokens=False)
    if len(special_ids) != 1:
        raise PromptError(f"{SPECIAL_TOKEN!r} must be one token, got {special_ids}")
    special_id = special_ids[0]

    user = build_user_content(blocks, question)
    prompt_msgs = [{"role": "user", "content": user}]
    if for_generation:
        prompt_text = tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_len:
            raise PromptError(f"prompt length {len(ids)} exceeds max_len {max_len}")
        n_ph = sum(n for _, n in blocks)
        pos = find_placeholder_positions(
            ids, special_token_id=special_id, n_expected=n_ph, blocks=blocks
        )
        pos_layer: list[int] = []
        for L, n in blocks:
            pos_layer.extend([L] * n)
        return EncodedSample(
            input_ids=ids,
            labels=[-100] * len(ids),
            positions=pos,
            pos_layer=pos_layer,
            plen=len(ids),
            n_tokens=len(ids),
        )

    prompt_text, full_text = render_chat(tok, prompt_msgs, answer)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    plen = common_prefix_length(p_ids, f_ids)
    if plen < len(p_ids):
        raise PromptError("chat prompt tokens are not a prefix of the full conversation")
    if len(f_ids) > max_len:
        raise PromptError(f"sample length {len(f_ids)} exceeds max_len {max_len} (no truncation)")
    if plen >= len(f_ids):
        raise PromptError("no answer tokens after the prompt")

    n_ph = sum(n for _, n in blocks)
    pos = find_placeholder_positions(
        f_ids, special_token_id=special_id, n_expected=n_ph, blocks=blocks
    )
    pos_layer = []
    for L, n in blocks:
        pos_layer.extend([L] * n)
    labels = [-100] * plen + list(f_ids[plen:])
    return EncodedSample(
        input_ids=list(f_ids),
        labels=labels,
        positions=pos,
        pos_layer=pos_layer,
        plen=plen,
        n_tokens=len(f_ids),
    )


def length_quantiles(lengths: Sequence[int]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(list(lengths), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    p50, p90, p99 = np.quantile(arr, [0.5, 0.9, 0.99])
    return {
        "n": int(arr.size),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }

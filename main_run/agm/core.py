"""Hooks and per-document reductions shared by the offline extractor and the online callback.

Conventions (written into every index file):
- Layer L is the output of ``model.model.layers[L]`` (the residual stream before the final norm). For
  L < n_layers - 1 it equals ``output_hidden_states[L+1]`` and the Activation Oracle's ``Layer: L``. For
  the last layer it does not: transformers 5.x replaces ``output_hidden_states[-1]`` with the
  final-normed ``last_hidden_state`` (``tie_last_hidden_states``), while the hook sees the un-normed
  block output.
- Gradient arrays store ``+dL/dh`` where L is the summed (not averaged, not rescaled) token
  cross-entropy over the label positions. Readers and checks negate at load time.
- Position t carries the label of token t+1. For a row of n tokens whose first label token (the
  response start) is token index ``plen`` (``plen = 0`` for raw text; for chat rows ``plen`` follows
  ``think_labels``: with ``public`` it is ``<think>``, so the response includes the 4 tokens of the empty
  think block, with ``masked`` it is the first completion token, see ``data.py``):
    full   = positions 0 .. n-2 (1 .. n-2 for a kind listed in ``full_skip_pos0``)
    resp   = label positions plen-1 .. n-2 (the positions whose next token is a response token); it
             includes the tag position, whose own token is the last prompt token
    tag    = position plen-1 (the last prompt position; its next token is the first response token:
             the ``\\n`` after ``assistant`` under ``public``, the block's last ``\\n\\n`` under ``masked``)
    chunk4 = resp split into 4 contiguous near-equal parts (numpy.array_split), stored as
             chunk4p0 .. chunk4p3; gradient only
    pos0   = position 0
  Label positions are max(plen-1, 0) .. n-2. Padding positions never enter any sum.
  Derived sums: prompt-token positions 0 .. plen-1 = full - resp + tag (+ pos0 for a kind whose
  ``full`` excludes position 0); response-token positions
  plen .. n-1 = resp - tag for gradients (position n-1 carries no label, so its gradient is exactly
  zero). The activation at position n-1 is in no stored sum.
  ``full_skip_pos0`` is a bool (both kinds) or a dict {'grad': bool, 'act': bool}; a kind that
  excludes position 0 from ``full`` also stores ``pos0`` for that kind (``kind_quantities``).

Reductions happen inside the hooks: each hook multiplies a 0/1 position-weight tensor
``W [B, Q, T]`` (fp32) with the fp32 copy of the bf16 hidden state or gradient ``[B, T, D]`` and
keeps only ``[B, Q, D]`` fp32 sums. Nothing token-level is kept.

Offline mode: the lowest gradient layer replaces its output with ``out.detach().requires_grad_(True)``
so the backward stops there; every gradient layer registers a tensor hook on its output.
Online mode: never detach; the tensor hook is registered only when ``torch.is_grad_enabled()`` and
the output requires grad, and at most once per layer per micro-batch (so recomputation under
gradient checkpointing does not double count). Sums are per micro-batch: ``begin_microbatch()``
clears them, so the caller reads ``grad``/``act`` after each micro-batch's backward.
"""
from __future__ import annotations

import numpy as np
import torch

LAYER_CONVENTION = (
    "layer L = output of model.model.layers[L] (before the final norm) = Activation Oracle 'Layer: L'; "
    "equals output_hidden_states[L+1] for L < n_layers-1 only, because transformers 5.x puts the "
    "final-normed last_hidden_state in output_hidden_states[-1]"
)
SIGN_CONVENTION = (
    "gradient arrays store +dL/dh of the summed, unscaled token cross-entropy over label positions; "
    "negate at load time for the descent direction"
)
POSITION_CONVENTION = (
    "position t carries the label of token t+1; plen = first label token (chat: '<think>' under think_labels "
    "public, the first completion token under masked); full = 0..n-2 (1..n-2 for a kind in full_skip_pos0); "
    "resp = label positions plen-1..n-2 (includes tag); tag = plen-1 (last prompt token); "
    "chunk4p0..3 = resp split by numpy.array_split into 4; pos0 = position 0; padding excluded; "
    "prompt tokens 0..plen-1 = full - resp + tag (+ pos0 if full excludes it); response tokens = resp - tag (gradient; position n-1 "
    "has zero gradient); the activation at position n-1 is in no stored sum"
)
STORAGE_CONVENTION = "arrays are fp16 unit vectors [rows, d_model]; fp32 norms in index.parquet; zero vector -> zeros, norm 0"

AO_LAYERS = [5, 10, 15, 20, 25, 30, 35, 38]
BASE_QUANTITIES = ("full", "resp", "tag", "chunk4", "pos0")
GRAD_ONLY = ("chunk4",)
NEEDS_RESPONSE = ("resp", "tag", "chunk4")


def expand_quantities(quantities, kind):
    """Array-level quantity names for 'grad' or 'act' in a fixed order."""
    out = []
    for q in quantities:
        if q not in BASE_QUANTITIES:
            raise ValueError(f"unknown quantity {q!r}; choose from {BASE_QUANTITIES}")
        if q == "chunk4":
            if kind == "grad":
                out += [f"chunk4p{i}" for i in range(4)]
            continue
        out.append(q)
    return out


def skip_for(full_skip_pos0, kind):
    """``full_skip_pos0`` is a bool (both kinds) or a dict {'grad': bool, 'act': bool}."""
    if isinstance(full_skip_pos0, dict):
        return bool(full_skip_pos0.get(kind, False))
    return bool(full_skip_pos0)


def kind_quantities(quantities, kind, full_skip_pos0=False):
    """Base quantities stored for ``kind``: excluding position 0 from ``full`` adds ``pos0`` for that kind."""
    q = list(quantities)
    if skip_for(full_skip_pos0, kind) and "pos0" not in q:
        q.append("pos0")
    return q


def array_quantities(quantities, kind, full_skip_pos0=False):
    """Array-level quantity names stored for ``kind`` (the order of the hook sums)."""
    return expand_quantities(kind_quantities(quantities, kind, full_skip_pos0), kind)


def array_name(q, kind, layer):
    return f"{q}_{kind}_L{layer}"


def parse_layers(spec, n_layers):
    """'all', 'none', '' or a comma list -> sorted list of layer indices."""
    if spec is None or spec in ("", "none"):
        return []
    if spec == "all":
        return list(range(n_layers))
    if spec == "ao8":
        layers = list(AO_LAYERS)
    else:
        layers = [int(x) for x in spec.split(",")]
    bad = [L for L in layers if not 0 <= L < n_layers]
    if bad:
        raise ValueError(f"layers {bad} out of range for a model with {n_layers} layers")
    return sorted(set(layers))


def position_sets(n_tokens, plen, q, full_skip_pos0=False):
    """List of (start, stop) half-open position ranges for base quantity ``q`` (chunk4 gives 4)."""
    n, p = int(n_tokens), int(plen)
    if q == "full":
        return [(1 if full_skip_pos0 else 0, n - 1)]
    if q == "pos0":
        return [(0, 1)]
    if q in NEEDS_RESPONSE and p < 1:
        raise ValueError("resp/tag/chunk4 need a response boundary (plen >= 1)")
    if q == "resp":
        return [(p - 1, n - 1)]
    if q == "tag":
        return [(p - 1, p)]
    if q == "chunk4":
        idx = np.array_split(np.arange(p - 1, n - 1), 4)
        return [((int(a[0]), int(a[-1]) + 1) if len(a) else (0, 0)) for a in idx]
    raise ValueError(q)


def position_weights(rows, T, quantities, kind, full_skip_pos0=False):
    """0/1 weights ``[B, Q, T]`` fp32 (CPU) for ``array_quantities(quantities, kind, full_skip_pos0)``.

    ``rows`` is a sequence of (n_tokens, plen) pairs.
    """
    quantities = kind_quantities(quantities, kind, full_skip_pos0)
    skip = skip_for(full_skip_pos0, kind)
    names = expand_quantities(quantities, kind)
    W = np.zeros((len(rows), len(names), T), dtype=np.float32)
    for b, (n, p) in enumerate(rows):
        qi = 0
        for q in quantities:
            if kind == "act" and q in GRAD_ONLY:
                continue
            for s, e in position_sets(n, p, q, skip):
                if e > s:
                    W[b, qi, s:e] = 1.0
                qi += 1
        assert qi == len(names)
    return torch.from_numpy(W)


def build_batch(id_lists, plens, pad_id=0):
    """Right-padded ``input_ids``, ``attention_mask`` (None if no padding) and ``labels`` on CPU.

    ``labels[b, t] = ids[b, t]`` for ``max(plen, 1) <= t < n`` and -100 elsewhere, so position t-1
    carries the label of token t.
    """
    B = len(id_lists)
    T = max(len(x) for x in id_lists)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    for b, (x, p) in enumerate(zip(id_lists, plens)):
        t = torch.as_tensor(x, dtype=torch.long)
        ids[b, : len(x)] = t
        am[b, : len(x)] = 1
        s = max(int(p), 1)
        lab[b, s : len(x)] = t[s:]
    padded = bool((am == 0).any())
    return ids, (am if padded else None), lab


def summed_ce(logits, labels):
    """Sum of token cross-entropy over label positions, computed in fp32 from bf16 logits."""
    lab = labels[:, 1:]
    mask = lab != -100
    lg = logits[:, :-1][mask].float()
    return torch.nn.functional.cross_entropy(lg, lab[mask], reduction="sum")


def decoder_layers(model):
    """``model.model.layers`` of a causal LM, also through a PEFT wrapper."""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    return m.model.layers


def set_fp32_matmul_ieee():
    """Keep fp32 matmuls (the reductions) in full fp32: no TF32."""
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    except (AttributeError, RuntimeError):
        torch.backends.cuda.matmul.allow_tf32 = False


def reduce_fp32(W, x):
    """``W [B, Q, T] @ x [B, T, D]`` in fp32; autocast is disabled so a bf16 autocast context
    (online training) cannot silently turn the reduction into a bf16 matmul."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return torch.bmm(W, x.float())


class Collector:
    """Registers hooks on decoder layers and accumulates per-row fp32 sums.

    Usage per batch: ``set_weights(Wg, Wa)`` (CUDA tensors ``[B, Qg, T]``, ``[B, Qa, T]``), forward,
    loss.backward(), then read ``grad[L]`` and ``act[L]`` (``[B, Q, D]`` fp32). ``reset()`` clears
    the sums. Online mode: call ``begin_microbatch()`` before each micro-batch and read the sums after
    its backward; rows of different micro-batches are different documents, so sums never carry over.
    """

    def __init__(self, model, grad_layers, act_layers, mode="offline"):
        assert mode in ("offline", "online")
        self.layers = decoder_layers(model)
        self.grad_layers = sorted(grad_layers)
        self.act_layers = sorted(act_layers)
        self.mode = mode
        self.detach_layer = self.grad_layers[0] if (mode == "offline" and self.grad_layers) else None
        self.handles = []
        self.Wg = self.Wa = None
        self.reset()

    def reset(self):
        self.grad = {}
        self.act = {}
        self._grad_hooked = set()
        self._act_done = set()

    def set_weights(self, Wg, Wa):
        self.Wg, self.Wa = Wg, Wa

    def _grad_fn(self, L):
        def fn(g):
            s = reduce_fp32(self.Wg, g)
            self.grad[L] = s if L not in self.grad else self.grad[L] + s
            return None

        return fn

    def _make_hook(self, L):
        do_act = L in self.act_layers
        do_grad = L in self.grad_layers
        is_detach = L == self.detach_layer

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            new_out = None
            if self.mode == "online":
                active = torch.is_grad_enabled() and h.requires_grad
                if not active:
                    return None
                if do_act and L not in self._act_done:
                    s = reduce_fp32(self.Wa, h.detach())
                    self.act[L] = s if L not in self.act else self.act[L] + s
                    self._act_done.add(L)
                if do_grad and L not in self._grad_hooked:
                    h.register_hook(self._grad_fn(L))
                    self._grad_hooked.add(L)
                return None
            if do_act:
                s = reduce_fp32(self.Wa, h.detach())
                self.act[L] = s if L not in self.act else self.act[L] + s
            if is_detach:
                h = h.detach().requires_grad_(True)
                new_out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            if do_grad and h.requires_grad:  # False only under no_grad (forward-only use)
                h.register_hook(self._grad_fn(L))
            return new_out

        return hook

    def begin_microbatch(self):
        """Online mode: clear the sums and re-arm the hooks for the next micro-batch. Row b of one
        micro-batch is a different document from row b of another, so sums must not be added by row."""
        self.reset()

    def attach(self):
        for L in sorted(set(self.grad_layers) | set(self.act_layers)):
            self.handles.append(self.layers[L].register_forward_hook(self._make_hook(L)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def load_model(model_id, adapter=None, dtype=torch.bfloat16, device="cuda:0"):
    """Frozen causal LM (SDPA attention), optionally with a PEFT adapter left unmerged, in eval mode."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa", device_map={"": device}
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def to_unit(x):
    """fp32 ``[..., D]`` -> (fp16 unit vectors, fp32 norms); zero vectors stay zero with norm 0."""
    n = torch.linalg.vector_norm(x, dim=-1)
    u = torch.where(n.unsqueeze(-1) > 0, x / n.clamp_min(1e-30).unsqueeze(-1), torch.zeros_like(x))
    return u.half(), n


def run_batch(model, collector, id_lists, plens, quantities, device, full_skip_pos0=False, pad_id=0):
    """One forward + backward over a batch; returns (grad dict, act dict, loss value, T).

    ``quantities`` is the base list (full, resp, tag, chunk4, pos0); ``full_skip_pos0`` is a bool or a
    dict {'grad': bool, 'act': bool}. grad[L], act[L] are fp32 ``[B, Q, D]`` sums in the order of
    ``array_quantities(quantities, 'grad' | 'act', full_skip_pos0)``.
    """
    ids, am, lab = build_batch(id_lists, plens, pad_id)
    T = ids.shape[1]
    rows = [(len(x), p) for x, p in zip(id_lists, plens)]
    Wg = position_weights(rows, T, quantities, "grad", full_skip_pos0).to(device, non_blocking=True)
    Wa = position_weights(rows, T, quantities, "act", full_skip_pos0).to(device, non_blocking=True)
    collector.reset()
    collector.set_weights(Wg, Wa)
    ids = ids.to(device, non_blocking=True)
    lab = lab.to(device, non_blocking=True)
    am = am.to(device, non_blocking=True) if am is not None else None
    with torch.enable_grad():
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
        loss = summed_ce(logits, lab)
        del logits
        if collector.grad_layers:
            loss.backward()
    return collector.grad, collector.act, float(loss.detach()), T

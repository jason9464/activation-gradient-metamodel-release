# Release note: this is the em-reader branch version of this file (the SFT and reading runs used it; rl-stage-fixes carries a differing copy).
"""AO-style norm-matched residual injection at placeholder positions (decoder layer 1).

Port of ``get_hf_activation_steering_hook`` with a persistent state holder so the same
hook lives through Trainer's forward + backward (including non-reentrant checkpoint
recompute). Vectors are normalised only on active (unmasked) slots; padded / no_vector
slots are never touched. The addition is detached and out-of-place: ``resid + delta``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from main_run.agm.core import decoder_layers

DEFAULT_HOOK_LAYER = 1
DEFAULT_STEERING_COEFFICIENT = 1.0


@dataclass
class HookState:
    """Batch injection state read by the persistent forward hook."""

    vectors: torch.Tensor | None = None  # [B, Kmax, d]
    positions: torch.Tensor | None = None  # [B, Kmax] long
    mask: torch.Tensor | None = None  # [B, Kmax] bool
    steering_coefficient: float = DEFAULT_STEERING_COEFFICIENT
    # Set with the fast path (--fast-gen, the CLI default): the hook clears the batch tensors
    # after the prefill forward, so the compiled decode steps (seq len 1, never injected) find no state and
    # never read or guard on them.
    clear_after_prefill: bool = False

    def clear(self) -> None:
        self.vectors = None
        self.positions = None
        self.mask = None

    def set(
        self,
        vectors: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        *,
        steering_coefficient: float | None = None,
    ) -> None:
        self.vectors = vectors
        self.positions = positions
        self.mask = mask
        if steering_coefficient is not None:
            self.steering_coefficient = float(steering_coefficient)


@dataclass
class InjectionHook:
    """One forward hook on ``decoder_layers(model)[hook_layer]`` reading ``state``."""

    state: HookState
    hook_layer: int = DEFAULT_HOOK_LAYER
    handle: Any = field(default=None, repr=False)

    def attach(self, model) -> "InjectionHook":
        if self.handle is not None:
            raise RuntimeError("injection hook already attached")
        layers = decoder_layers(model)
        if self.hook_layer < 0 or self.hook_layer >= len(layers):
            raise ValueError(f"hook_layer {self.hook_layer} out of range for {len(layers)} layers")
        self.handle = layers[self.hook_layer].register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, inputs, output):
        st = self.state
        if st.vectors is None or st.positions is None or st.mask is None:
            return output

        if isinstance(output, tuple):
            resid = output[0]
            rest = output[1:]
            is_tuple = True
        else:
            resid = output
            rest = ()
            is_tuple = False

        B, L, D = resid.shape
        # Prefill only (AO): skip decode steps with seq len <= 1.
        if L <= 1:
            return output

        vec = st.vectors
        pos = st.positions
        mask = st.mask
        if vec.shape[0] != B or pos.shape[0] != B or mask.shape[0] != B:
            raise ValueError(
                f"injection batch mismatch: resid B={B}, vectors B={vec.shape[0]}, "
                f"positions B={pos.shape[0]}, mask B={mask.shape[0]}"
            )
        if vec.shape[-1] != D:
            raise ValueError(f"injection d_model mismatch: resid D={D}, vectors D={vec.shape[-1]}")

        # Out-of-place: never mutate the module's residual tensor in place.
        out = resid
        coeff = st.steering_coefficient
        Kmax = mask.shape[1]
        for b in range(B):
            active = mask[b]
            if not bool(active.any()):
                continue
            idx = pos[b, active].long()
            if idx.numel() == 0:
                continue
            if int(idx.min()) < 0 or int(idx.max()) >= L:
                raise ValueError(f"placeholder position out of range for L={L}: {idx.tolist()}")
            v = vec[b, active].to(device=resid.device, dtype=torch.float32)
            # Never normalise empty / pad slots (already filtered by mask).
            v_hat = torch.nn.functional.normalize(v, dim=-1)
            h = resid[b, idx].float()
            norms = h.norm(dim=-1, keepdim=True).detach()
            delta = (v_hat * norms * coeff).to(dtype=resid.dtype).detach()
            # First write clones; further rows reuse the clone.
            if out is resid:
                out = resid.clone()
            out[b, idx] = resid[b, idx] + delta

        if st.clear_after_prefill:
            st.clear()
        if is_tuple:
            return (out,) + rest
        return out


def load_ao_hook_settings(adapter_dir: str | Path | None = None, ao_config: dict | None = None) -> dict[str, Any]:
    """Read ``hook_onto_layer`` and ``steering_coefficient`` from an AO ``ao_config.json``."""
    cfg = ao_config
    if cfg is None and adapter_dir is not None:
        path = Path(adapter_dir) / "ao_config.json"
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg is None:
        return {
            "hook_onto_layer": DEFAULT_HOOK_LAYER,
            "steering_coefficient": DEFAULT_STEERING_COEFFICIENT,
            "special_token": " ?",
        }
    return {
        "hook_onto_layer": int(cfg.get("hook_onto_layer", DEFAULT_HOOK_LAYER)),
        "steering_coefficient": float(cfg.get("steering_coefficient", DEFAULT_STEERING_COEFFICIENT)),
        "special_token": cfg.get("special_token", " ?"),
    }


def resolve_adapter_files(adapter_id: str, revision: str | None = None) -> Path:
    """Local snapshot path for a Hub (or local) AO adapter; downloads ``ao_config.json`` if needed."""
    p = Path(adapter_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(adapter_id, revision=revision))

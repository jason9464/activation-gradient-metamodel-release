"""Device choice for the single-process reader tools (``nll_eval``, ``generate``).

Nothing is assumed: ``--gpus 3`` pins physical GPU 3 (as ``train --gpus 3`` does), ``--device`` is
used as given (``cuda:1``, ``cpu``), and with neither the tool runs only when exactly one GPU is
visible. Several GPUs: run one process per GPU (e.g. one checkpoint each).
"""
from __future__ import annotations

import os


def eval_device(gpus: str | None, device: str | None) -> str:
    import torch

    if gpus is not None and device is not None:
        raise SystemExit("--gpus and --device both choose the device; pass one")
    if device is not None:
        return device
    if gpus is not None:
        ids = [g.strip() for g in str(gpus).split(",") if g.strip()]
        if len(ids) != 1 or not ids[0].isdigit():
            raise SystemExit(
                f"--gpus takes one GPU index here (e.g. 3), got {gpus!r}; run one process per GPU to use several"
            )
        if torch.cuda.is_initialized():
            raise SystemExit("CUDA already started in this process; --gpus must be applied before it starts")
        # Physical index, like train --gpus: this process sees only that GPU, as cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = ids[0]
        return "cuda:0"
    n = torch.cuda.device_count()
    if n == 1:
        return "cuda:0"
    if n == 0:
        raise SystemExit("no GPU visible; pass --device cpu to run on the CPU")
    raise SystemExit(f"{n} GPUs visible; choose one with --gpus <index> (or --device cuda:<index>)")

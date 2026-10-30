#!/usr/bin/env python
"""−g of a finished model on fixed documents: the gradient-input version of the LoRAcles broad/narrow EM test.

Stage A2 add-on (exploratory, 2026-10-25). For a run dir whose ``adapters/step_<t>`` holds a finished adapter (e.g.
``ext_broad``, ``ext_narrow`` built from Celeste's public adapters; step 0 = LoRA B zeroed = the base model), sum
over the documents of a fixed-docs file the activation gradient +dL/dh of every decoder-layer output (the summed,
unscaled token cross-entropy over the completion labels, the gradient store's ``full`` token set: positions 0..n-2;
``agm.core.run_batch``, bf16 model as the training-time store), and write it in the ``e2_read.export_inputs`` layout
(inputs = −(sum of +dL/dh), the descent direction the trained readers take) as ``<run>/e2/gradfin_inputs_<set>.npz``,
one read step ``t``, window ``fin``. ``e2_generate`` names the readings ``gradfin_<reader>_<question>``.

Usage:
  python -m main_run.em.ext_grad_inputs --run <runs>/ext_broad --docs <dact_docs>/train.jsonl \\
      --set train --step 375 [--batch 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ckpt_diff_inputs as cdi  # noqa: E402
from agm import core  # noqa: E402

VARIANT = "gradfin"
SIGN = ("stored per-document arrays would be +dL/dh of the summed, unscaled token cross-entropy over the completion "
        "labels (agm.core.run_batch, full token set: positions 0..n-2); inputs = -(sum over the fixed documents of "
        "+dL/dh): the descent direction of the finished model on those documents, stored as fp32")


def grad_sums(model, tok, rows, st, *, batch: int = 4, device: str = "cuda:0") -> np.ndarray:
    """fp64 [L, D] sum over ``rows`` (prompt/completion) of the per-layer +dL/dh token sums."""
    import train_subliminal as ts

    ids, masks, _ = ts.tokenize_rows(tok, [dict(prompt=r["prompt"], completion=r["completion"], source=None)
                                           for r in rows], st["max_len"], st["think_loss"])
    plens = [m.index(1) if 1 in m else len(m) for m in masks]
    L = len(core.decoder_layers(model))
    col = core.Collector(model, grad_layers=list(range(L)), act_layers=[], mode="offline").attach()
    total = None
    try:
        for j in range(0, len(ids), batch):
            g, _, _, _ = core.run_batch(model, col, ids[j:j + batch], plens[j:j + batch], ["full"], device,
                                        pad_id=tok.pad_token_id)
            s = np.stack([g[li][:, 0].double().sum(0).cpu().numpy() for li in range(L)])  # [L, D]
            total = s if total is None else total + s
            model.zero_grad(set_to_none=True)
    finally:
        col.remove()
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--step", type=int, required=True, help="adapter step to read (step 0 = the base model)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    from main_run.reader.e2_generate import export_order_hash

    run = os.path.abspath(a.run)
    out = os.path.join(run, "e2", f"{VARIANT}_inputs_{a.set}_s{a.step}.npz")
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    st = json.load(open(os.path.join(run, "RUN.json")))["settings"]
    rows, _, docs_sha = cdi.load_fixed_docs(a.docs)
    t0 = time.time()
    core.set_fp32_matmul_ieee()
    import torch
    from transformers import AutoTokenizer

    model = core.load_model(st["model"], cdi.adapter_dir_of(run, a.step), dtype=torch.bfloat16, device=a.device)
    tok = AutoTokenizer.from_pretrained(st["model"])
    tok.padding_side = "right"
    G = grad_sums(model, tok, rows, st, batch=a.batch, device=a.device)
    X = (-G)[None, None].astype(np.float32)  # [1 step, 1 window, L, D]
    nrm = np.linalg.norm(X, axis=-1)
    if not (np.isfinite(X).all() and (nrm > 0).all()):
        raise SystemExit("non-finite or zero gradient sums")
    layers = list(range(X.shape[2]))
    meta = dict(run=run, arm=st.get("arm"), seed=st.get("seed"), quantity="full", kind="grad", sign=SIGN,
                layers=layers, windows=[["fin", None]], steps=[a.step],
                order_hash=export_order_hash([a.step], ["fin"], layers),
                transform=dict(variant=VARIANT, set=a.set, docs=os.path.abspath(a.docs), docs_sha256=docs_sha,
                               n_docs=len(rows), adapter=cdi.adapter_dir_of(run, a.step), dtype="bf16 model, fp64 sums",
                               plan="docs/notes/20261025_em_dact_results.md (Stage A2)"),
                checks_passed="finite_nonzero 1/1", seconds=round(time.time() - t0, 1),
                written=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        np.savez(f, inputs=X, steps=np.asarray([a.step]), windows=np.asarray(["fin"]), window_lo=np.ones((1, 1), np.int64),
                 window_hi=np.full((1, 1), a.step), n_docs=np.full((1, 1), len(rows)), layers=np.asarray(layers),
                 meta=np.asarray(json.dumps(meta)))
    os.replace(out + ".tmp", out)
    print(f"wrote {out}: {len(rows)} docs, layer norm median {float(np.median(nrm)):.4g}, {meta['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

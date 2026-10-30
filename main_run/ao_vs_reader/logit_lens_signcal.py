#!/usr/bin/env python
"""Calibration of plan check 3 (docs/notes/20261028_logit_lens_results.md) on a model small enough for this
CPU check: Qwen3-0.6B (same tokenizer, so the same lens vocabulary), CPU, fp32.

The 14B check found a present-token AUROC of 0.33 for the readers' raw −dL/dh bundle vectors at layer 39. This
computes the TRUE −dL/dh at the last decoder layer of Qwen3-0.6B for the documents of 9 behaviour bundles (the first
3 bundles of quirk, backdoor and rare in bundles.parquet order; no description is read). It reports the same
present-token AUROC for −dL/dh, +dL/dh and the activation sum, and splits the gradient's lens into its two parts:
the error term Γ W_Uᵀ(y − p) and the final RMSNorm's projection term. If the true −dL/dh also gives an AUROC below
0.5, the check's premise was wrong and the 14B value is what the correct sign looks like.

  python main_run/ao_vs_reader/logit_lens_signcal.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logit_lens as LL  # noqa: E402
import probe_compare as P  # noqa: E402

SMALL = "Qwen/Qwen3-0.6B"


def main() -> int:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    tok = LL.tokenizer()
    vocab = LL.lens_vocab(tok)
    lab = pd.read_parquet(P.OUT / "bundles.parquet")
    picks = [b for f in ("quirk", "backdoor", "rare") for b in lab[lab.family == f].bundle_id[:3]]
    docs = {b: [] for b in picks}
    with open(LL.EXAMPLES) as fh:
        for line in fh:
            r = json.loads(line)
            if r["bundle_id"] in docs:
                docs[r["bundle_id"]].append(r["messages"])
    model = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    W = model.lm_head.weight.detach()  # tied to the embedding in 0.6B; the lens uses the output matrix either way
    gamma = model.model.norm.weight.detach()
    eps = model.config.rms_norm_eps
    vocab_t = torch.as_tensor(vocab)
    col_of = np.full(W.shape[0], -1)
    col_of[vocab] = np.arange(len(vocab))
    rows = []
    for b in picks:
        sums = {k: torch.zeros(W.shape[1]) for k in ("g", "a", "err", "proj")}
        present = np.zeros(len(vocab), bool)
        for msgs in docs[b]:
            user, asst = msgs[0]["content"], msgs[1]["content"]
            prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            resp = f"<think>\n\n</think>\n\n{asst}<|im_end|>\n"
            p_ids = tok.encode(prompt, add_special_tokens=False).ids
            r_ids = tok.encode(resp, add_special_tokens=False).ids[:max(384 - len(p_ids), 64)]  # memory: [T, 152k] logits
            asst = tok.decode(r_ids)  # the present tokens are those of the kept response
            ids = torch.as_tensor([p_ids + r_ids])
            plen = len(p_ids)
            store = {}

            def hook(mod, inp, out):  # as main_run/agm/core.py offline mode: the backward stops at this layer
                h = (out[0] if isinstance(out, tuple) else out).detach().requires_grad_(True)
                store["h"] = h
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

            hd = layers[-1].register_forward_hook(hook)
            lab_ids = ids[0, plen:]
            with torch.enable_grad():
                logits = model(input_ids=ids).logits[0, plen - 1:-1]
                loss = torch.nn.functional.cross_entropy(logits, lab_ids, reduction="sum")
                loss.backward()
            hd.remove()
            lg = logits.detach()
            del logits, loss
            h = store["h"][0].detach()
            g = -store["h"].grad[0].detach()  # −dL/dh, the readers' sign
            sums["g"] += g.sum(0)
            sums["a"] += h[1:-1].sum(0)
            # analytic split of −dL/dh at the last layer: (1/r)[Γ W_Uᵀ e − h (hᵀ Γ W_Uᵀ e)/(d r²)], e = y − p
            for i in range(0, len(lab_ids), 64):
                e = -torch.softmax(lg[i:i + 64], -1)
                e[torch.arange(len(e)), lab_ids[i:i + 64]] += 1
                hh = h[plen - 1 + i:plen - 1 + i + len(e)]
                r = torch.sqrt((hh ** 2).mean(-1, keepdim=True) + eps)
                a_ = (e @ W) * gamma  # Γ W_Uᵀ e per position
                sums["err"] += (a_ / r).sum(0)
                sums["proj"] += (-hh * (hh * a_).sum(-1, keepdim=True) / (hh.shape[-1] * r ** 3)).sum(0)
            del lg
            c = col_of[np.asarray(tok.encode(asst, add_special_tokens=False).ids)]
            present[c[c >= 0]] = True
        recon = torch.nn.functional.cosine_similarity(sums["g"], sums["err"] + sums["proj"], dim=0).item()
        row = dict(bundle=b, n_docs=len(docs[b]), n_present=int(present.sum()), cos_g_vs_analytic=recon)
        for k, x in (("minus_dLdh", sums["g"]), ("plus_dLdh", -sums["g"]), ("act", sums["a"]),
                     ("error_term", sums["err"]), ("projection_term", sums["proj"])):
            s = (W @ (gamma * x))[vocab_t].numpy()
            rk = s.argsort().argsort() + 1
            n_p = present.sum()
            row[k] = float((rk[present].sum() - n_p * (n_p + 1) / 2) / (n_p * (len(s) - n_p)))
        rows.append(row)
        LL.log(json.dumps(row))
    df = pd.DataFrame(rows)
    out = LL.LENS / "signcal_qwen3_0.6b.json"
    out.write_text(json.dumps(dict(model=SMALL, rows=rows, mean=df.drop(columns=["bundle"]).mean().to_dict()),
                              indent=1))
    LL.log(f"mean over bundles: {df.drop(columns=['bundle']).mean().round(3).to_dict()}")
    LL.log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

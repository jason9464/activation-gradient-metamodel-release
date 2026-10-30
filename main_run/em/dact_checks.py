#!/usr/bin/env python
"""Checks and diagnostics for EM Δh (input-5a) exports.

Gated checks (exit 1 when one fails):
  sign_path (CPU)  ΔH = H(t) − H(0) recomputed from the h files of ``<run>/ckptdiff_<probe>/`` must equal the npz
                   cell (relative error < 1e-5, every layer cos > 0), and e2_generate's own path (``load_inputs`` →
                   reader layer order → ``reader_vectors``) must give unit(fp16(ΔH)) − mean_unit per layer
                   (cos ≥ 0.9999). A sign flip anywhere fails both.
  bf16 (GPU)       every LoRA tensor of adapter t perturbed by a seeded random ±1 bf16 ulp; H′(t) − H(0) against
                   H(t) − H(0) per layer, min and median cos ≥ 0.999 (EM adapters are saved in bf16; the rounding is
                   at most half an ulp, so this bounds its effect).
Reported only:
  pos0 (GPU)       share of ‖ΔH_ℓ‖ carried by position 0: ‖ΔP_ℓ‖ / ‖ΔH_ℓ‖ with P the position-0 outputs summed over
                   the documents; ``flag_no_pos0_variant`` when the median over layers is > 0.5.
  train_vs_grad    cos(+Δh ``train`` at step 5, −Σg cum at step 5). No sign is guaranteed (Adam, nonlinearity,
                   different objects).
  em_direction     d_EM[ℓ] = mean completion-token output of judged-misaligned − judged-aligned step-375 answers
                   (broad questions) under the step-375 adapter; cos(ΔH_evalq(t)[ℓ], d_EM[ℓ]) per read step against
                   random Gaussian directions. May reflect the answer texts; not evidence that Δh encodes EM.

One fp32 model serves every GPU check.

Usage:
  python -m main_run.em.dact_checks --run <run> --probe evalq --reader-dir <agm-readers>/rl_data/reader \\
      --reader-run-json <reader RUN.json> [--cpu-only]
  python -m main_run.em.dact_checks --run <run> --probe evalq --reader-dir ... --reader-run-json ... \\
      --docs <docs>/evalq.jsonl --steps 5,60,375 --evals <run>/em_evals.jsonl \\
      [--train-npz <run>/e2/ckptdiff_inputs_train_dact.npz --grad-cum <run>/e2/inputs_cum_dact.npz] --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ckpt_diff_inputs as cdi  # noqa: E402
from main_run.reader.e2_generate import load_inputs, load_mean_unit, reader_settings, reader_vectors  # noqa: E402

REL_TOL = 1e-5
VEC_COS = 0.9999
BF16_COS = 0.999
POS0_FLAG = 0.5


def _cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def subdir_of(probe: str) -> str:
    return f"ckptdiff_{probe}"


def npz_of(run: str, probe: str) -> Path:
    return Path(cdi.resolve_run(run)) / "e2" / f"ckptdiff_inputs_{probe}_dact.npz"


def delta_from_h_files(run: str, subdir: str, t: int) -> np.ndarray:
    """fp64 [L, D] H(t) − H(0) from ``h_step_*.npy`` (each [2 halves, L, D])."""
    cd = Path(cdi.resolve_run(run)) / subdir
    h = {s: np.load(cd / f"h_step_{s:06d}.npy") for s in (0, t)}
    return h[t].sum(0) - h[0].sum(0)


# ------------------------------------------------------------------ sign path (CPU)

def sign_path_check(run: str, probe: str, mean_unit: np.ndarray | None, reader_layers: list[int], *,
                    steps: list[int] | None = None, npz: Path | None = None) -> dict:
    """npz == +ΔH from the h files, and e2_generate's vector path == unit(fp16(ΔH)) − mean, at ``steps``."""
    npz = npz or npz_of(run, probe)
    inp = load_inputs(npz, windows=["cum"])
    row = {t: i for i, t in enumerate(inp["steps"])}
    use = [t for t in (steps or inp["steps"]) if t in row]
    if not use:
        return dict(ok=False, why=f"none of {steps} in {npz}")
    lidx = [inp["layers"].index(L) for L in reader_layers]  # run_readings' layer selection
    cells = []
    for t in use:
        ref = delta_from_h_files(run, subdir_of(probe), t)  # h files and npz both use layers 0..L-1 in order
        x = np.asarray(inp["inputs"][row[t], 0], np.float64)
        if ref.shape != x.shape or inp["layers"] != list(range(ref.shape[0])):
            raise SystemExit(f"step {t}: h files {ref.shape} vs npz {x.shape} / layers {inp['layers'][:3]}...")
        rel = float(np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-300))
        layer_cos = [_cos(x[k], ref[k]) for k in range(len(inp["layers"]))]
        got = reader_vectors(np.asarray(inp["inputs"][row[t]][:, lidx], np.float32), mean_unit)[0]
        exp = reader_vectors(ref[lidx][None].astype(np.float32), mean_unit)[0]
        vec_cos = [_cos(got[k], exp[k]) for k in range(len(lidx))]
        cells.append(dict(step=t, rel_err=rel, min_layer_cos_npz_vs_h=min(layer_cos),
                          min_layer_cos_injected=min(vec_cos),
                          ok=rel < REL_TOL and min(layer_cos) > 0 and min(vec_cos) >= VEC_COS))
    sign = inp["meta"]["sign"]
    text_ok = "H(t) - H" in sign and "inputs = -(" in sign
    return dict(ok=all(c["ok"] for c in cells) and text_ok, sign_text_ok=text_ok, cells=cells, npz=str(npz),
                centred=mean_unit is not None)


# ------------------------------------------------------------------ GPU helpers

def _setup(model, run: str, docs_path: Path, step: int):
    from agm import core

    run = cdi.resolve_run(run)
    with open(os.path.join(run, "RUN.json")) as f:
        st = json.load(f)["settings"]
    rows, half, _ = cdi.load_fixed_docs(str(docs_path))
    pm, tok = model.get(st["model"], cdi.adapter_dir_of(run, step))
    names = cdi.lora_names(pm)
    batches, _, _ = cdi.fixed_batches(tok, rows, half, st)
    layers = list(range(len(core.decoder_layers(pm))))
    return run, st, rows, pm, tok, names, batches, layers


def _adapter(run: str, t: int) -> dict:
    return cdi.read_adapter(os.path.join(cdi.adapter_dir_of(run, t), "adapter_model.safetensors"))


def _H(pm, batches, n_docs, layers) -> np.ndarray:
    halves, _, _ = cdi.fixed_forward(pm, batches, n_docs, layers)
    return halves.sum(0)  # fp64 [L, D]


def bf16_ulp_perturb(params: dict, seed: int) -> dict:
    """Each element moved by ±1 bf16 ulp (random sign, seeded), returned in fp32 (the wrapper's dtype).

    ulp(x) = 2^(floor(log2|x|) − 7) for bf16 (8 significant bits); zeros stay zero.
    """
    import torch

    g = torch.Generator().manual_seed(int(seed))
    out = {}
    for k in sorted(params):
        v = params[k].to(torch.float32)
        e = torch.floor(torch.log2(v.abs().clamp_min(1e-38)))
        ulp = torch.where(v != 0, torch.pow(2.0, e - 7), torch.zeros_like(v))
        sgn = torch.randint(0, 2, v.shape, generator=g).to(torch.float32) * 2 - 1
        out[k] = v + sgn * ulp
    return out


def bf16_sensitivity(model, run: str, docs_path: Path, steps: list[int], *, seed: int = 0) -> dict:
    run, st, rows, pm, tok, names, batches, layers = _setup(model, run, docs_path, 0)
    cdi.apply_adapter(pm, _adapter(run, 0), names)
    H0 = _H(pm, batches, len(rows), layers)
    out = {}
    for t in steps:
        params = _adapter(run, t)
        cdi.apply_adapter(pm, params, names)
        d = _H(pm, batches, len(rows), layers) - H0
        cdi.apply_adapter(pm, bf16_ulp_perturb(params, seed + t), names)
        dp = _H(pm, batches, len(rows), layers) - H0
        per = [_cos(d[i], dp[i]) for i in range(len(layers))]
        out[str(t)] = dict(cos_layer_min=float(min(per)), cos_layer_median=float(np.median(per)),
                           worst_layer=int(np.argmin(per)), rel_change=float(np.linalg.norm(dp - d) / np.linalg.norm(d)),
                           ok=min(per) >= BF16_COS and float(np.median(per)) >= BF16_COS,
                           dtypes=sorted({str(v.dtype) for v in params.values()}))
        cdi.apply_adapter(pm, params, names)
    return dict(ok=all(v["ok"] for v in out.values()), per_step=out, rule=f"min and median layer cos >= {BF16_COS}")


def pos0_share(model, run: str, docs_path: Path, steps: list[int]) -> dict:
    """‖ΔP_ℓ‖ / ‖ΔH_ℓ‖ per layer with P the position-0 outputs (``ckpt_diff_inputs.sums``)."""
    run, st, rows, pm, tok, names, batches, layers = _setup(model, run, docs_path, 0)

    def sums_at(t):
        cdi.apply_adapter(pm, _adapter(run, t), names)
        S = P = 0
        for _hv, _sel, batch in batches:
            s, p0, _ = cdi.sums(pm, batch, layers)  # [L, B, D] fp64 each
            S = S + s.sum(1).cpu().numpy()
            P = P + p0.sum(1).cpu().numpy()
        return S, P

    S0, P0 = sums_at(0)
    out = {}
    for t in steps:
        St, Pt = sums_at(t)
        dH, dP = St - S0, Pt - P0
        share = [float(np.linalg.norm(dP[i]) / max(np.linalg.norm(dH[i]), 1e-300)) for i in range(len(layers))]
        med = float(np.median(share))
        out[str(t)] = dict(median_layer_share=med, max_layer_share=float(max(share)),
                           flag_no_pos0_variant=med > POS0_FLAG)
    return dict(per_step=out, flag_any=any(v["flag_no_pos0_variant"] for v in out.values()),
                definition="||sum over docs of dh at position 0|| / ||sum over docs of dh over the full token set||")


def em_direction(model, run: str, evals: Path, npz: Path, *, step: int = 375, n_per_class: int = 64,
                 batch_size: int = 16, n_random: int = 64, seed: int = 0) -> dict:
    """Cosine of ΔH_evalq with the misaligned − aligned mean completion activation at ``step`` (exploratory)."""
    import torch

    import train_subliminal as ts
    from agm import core
    from main_run.em.eval import NARROW_QUESTION_IDS

    run = cdi.resolve_run(run)
    with open(os.path.join(run, "RUN.json")) as f:
        st = json.load(f)["settings"]
    mis, aln = [], []
    with open(evals, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("step") != step or "ceselder" in str(r.get("adapter") or ""):
                continue
            if r.get("question") in NARROW_QUESTION_IDS or r.get("invalid") or not r.get("response"):
                continue
            if r.get("misaligned") is True:
                mis.append(r)
            elif r.get("misaligned") is False:
                aln.append(r)
    rng = np.random.default_rng(seed)
    n = min(n_per_class, len(mis), len(aln))
    if n < 8:
        return dict(ok=False, why=f"step {step}: {len(mis)} misaligned / {len(aln)} aligned answers")
    pick = {"mis": [mis[i] for i in sorted(rng.choice(len(mis), n, replace=False))],
            "aln": [aln[i] for i in sorted(rng.choice(len(aln), n, replace=False))]}
    pm, tok = model.get(st["model"], cdi.adapter_dir_of(run, step))
    cdi.apply_adapter(pm, _adapter(run, step), cdi.lora_names(pm))
    dec = core.decoder_layers(pm)
    L = len(dec)
    dev = next(pm.parameters()).device
    means = {}
    for key, rs in pick.items():
        rows = [dict(prompt=r["question_text"], completion=r["response"], source=None) for r in rs]
        ids, masks, _ = ts.tokenize_rows(tok, rows, st["max_len"], st["think_loss"])
        acc, cnt = torch.zeros(L, pm.config.hidden_size, dtype=torch.float64, device=dev), 0
        for j in range(0, len(ids), batch_size):
            b = ts.CompletionMaskCollator(tok)([dict(input_ids=i_, completion_mask=m_) for i_, m_ in
                                                zip(ids[j:j + batch_size], masks[j:j + batch_size])])
            am = b["attention_mask"].to(dev)
            w = ((b["labels"] != -100).to(dev).double())  # completion tokens only
            S = {}
            hs = [dec[li].register_forward_hook(
                lambda mod, inp, out, li=li: S.__setitem__(li, torch.einsum(
                    "bt,btd->d", w, (out[0] if isinstance(out, tuple) else out).double()))) for li in range(L)]
            try:
                with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
                    (pm.get_base_model() if hasattr(pm, "get_base_model") else pm).model(
                        input_ids=b["input_ids"].to(dev), attention_mask=am, use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            acc += torch.stack([S[li] for li in range(L)])
            cnt += int(w.sum())
        means[key] = (acc / max(cnt, 1)).cpu().numpy()
    d_em = means["mis"] - means["aln"]
    inp = load_inputs(npz, windows=["cum"])
    rnd = np.random.default_rng(seed + 1).standard_normal((n_random, d_em.shape[1]))
    base = np.abs(np.array([[_cos(inp["inputs"][0, 0, li], r) for r in rnd] for li in range(L)])).ravel()
    per_step = []
    for i, t in enumerate(inp["steps"]):
        c = [_cos(inp["inputs"][i, 0, li], d_em[li]) for li in range(L)]
        per_step.append(dict(step=int(t), cos_layer_median=float(np.median(c)), cos_layer_max=float(max(c)),
                             argmax_layer=int(np.argmax(c))))
    return dict(step=step, n_per_class=n, per_step=per_step, random_abs_cos_p95=float(np.quantile(base, 0.95)),
                note="exploratory: d_EM may reflect the answer texts; a positive cosine is not evidence that Δh "
                     "encodes EM")


def diagnostic_train_vs_grad(dact_train_npz: Path, grad_cum_npz: Path, *, step: int = 5) -> dict:
    a = load_inputs(dact_train_npz, windows=["cum"])
    b = load_inputs(grad_cum_npz, windows=["cum"])
    if step not in a["steps"] or step not in b["steps"]:
        return dict(ok=False, why=f"step {step} missing")
    va = np.asarray(a["inputs"][a["steps"].index(step), 0], np.float64)
    vb = np.asarray(b["inputs"][b["steps"].index(step), 0], np.float64)
    per = [_cos(va[k], vb[k]) for k in range(va.shape[0])]
    return dict(step=step, cos_flat=_cos(va, vb), cos_layer_median=float(np.median(per)),
                note="diagnostic only; no sign guaranteed (Adam, nonlinearity, different objects)")


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe", default="evalq")
    ap.add_argument("--npz", type=Path, default=None, help="default <run>/e2/ckptdiff_inputs_<probe>_dact.npz")
    ap.add_argument("--reader-dir", type=Path, required=True, help="meta.json and mean_unit_<weighting>.npy")
    ap.add_argument("--reader-run-json", type=Path, required=True, help="reader RUN.json (layers, centring)")
    ap.add_argument("--steps", default="5,60,375")
    ap.add_argument("--cpu-only", action="store_true", help="sign path only")
    ap.add_argument("--docs", type=Path, default=None, help="the probe's fixed docs (GPU checks)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--evals", type=Path, default=None, help="em_evals.jsonl (EM-direction diagnostic, evalq)")
    ap.add_argument("--train-npz", type=Path, default=None)
    ap.add_argument("--grad-cum", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    steps = [int(x) for x in a.steps.split(",") if x.strip()]
    npz = a.npz or npz_of(a.run, a.probe)
    rs = reader_settings(a.reader_run_json)
    mean = load_mean_unit(a.reader_dir, rs["weighting"], rs["layers"]) if rs["center"] else None
    report: dict[str, Any] = dict(run=a.run, probe=a.probe, npz=str(npz))
    report["sign_path"] = sign_path_check(a.run, a.probe, mean, rs["layers"], steps=steps, npz=npz)
    gated = ["sign_path"]
    if a.train_npz and a.grad_cum:
        report["diag_train_vs_grad"] = diagnostic_train_vs_grad(a.train_npz, a.grad_cum)
    if not a.cpu_only:
        if a.docs is None:
            raise SystemExit("GPU checks need --docs (or pass --cpu-only)")
        model = cdi.Fp32Model(a.device)
        report["bf16_sensitivity"] = bf16_sensitivity(model, a.run, a.docs, steps)
        gated.append("bf16_sensitivity")
        report["pos0_share"] = pos0_share(model, a.run, a.docs, steps)
        if a.evals is not None and a.probe == "evalq":
            report["diag_em_direction"] = em_direction(model, a.run, a.evals, npz)
    report["gated"] = {k: bool(report[k]["ok"]) for k in gated}
    report["ok"] = all(report["gated"].values())
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str)[:6000])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

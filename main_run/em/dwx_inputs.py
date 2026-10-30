#!/usr/bin/env python
"""Cumulative direct-write term on the fixed probe documents (EM analogue of the subliminal input 4f).

Results: ``docs/notes/20261025_em_dact_results.md``. In the subliminal new setting, input 4f (``adamfx_fixed``: each optimizer
step's real LoRA change of the two modules that write into the residual stream, o_proj and down_proj, times the fixed
documents' module inputs) read dolphin earliest (cumulative 243/600 before behaviour vs 74/600 for the activation
difference 5a; ``20261025_e2_newset_results.md``). The EM runs did not save it during training, but their adapters
(every 5 steps) give the same product between saved adapters:

  per layer L, fixed-document half h, module m in (o_proj, down_proj), LoRA factors A_t, B_t, scaling s_m (peft
  ``scaling``; lora_alpha / r = 1 for the EM students), X_m,h(t) = token sum of module m's input over the full token
  set (positions 0..n-2, ``adamfx.token_weights``) of the half's documents under adapter t (fp32 forward, the same
  Fp32Model and tokenisation as ``ckpt_diff_inputs.py --docs``):

  cum     (read window)  D(t_k) = sum_{j=1..k} sum_m s_m (B_j A_j - B_{j-1} A_{j-1}) X_m(t_{j-1})
          X refreshed at every adapter step of the export, as 4f refreshes X under the current model (4f: every 100
          steps; here at the 16 export adapters, 5-125 steps apart).
  cum_x0  (diagnostic)   D0(t) = sum_m s_m (B_t A_t - B_0 A_0) X_m(0) = sum_m s_m B_t A_t X_m(0)   (B_0 = 0)

Products are fp64 (``adamfx.module_delta``: s [(X A1^T) dB^T + (X dA^T) B0^T]), summed over the two halves and stored
as fp32. Only the direct writes of layer L are counted: changes of q, k, v, gate, up and of earlier layers reach the
residual stream only through the propagation that input 5a (``ckptdiff``) includes. Sign: +dh, the change the training
made (the adamfx / ckptdiff convention; e2_generate takes it as the descent direction).

  compute  per adapter step t: ``<run>/dwx_<set>/x_step_<t>.npz`` (x_o_proj [2, L, d_o_in], x_down_proj [2, L,
           d_down_in], fp64) and ``x_step_<t>.json`` (written last; a step is done when it exists), ``docs.json``.
  build    ``<run>/e2/dwx_inputs_<suffix>.npz`` in the e2_read.export_inputs layout (windows cum, cum_x0; meta kind
           "grad" for the loader, transform.variant "dwx", so e2_generate names readings dwx_<reader>_<question>) and
           ``<run>/dwx_<set>/split_half.json``. Checks (all must pass, else ``*.checks_failed``): docs identical at every
           step (and the same docs and token ids as the ckptdiff probe set and the pair run, when present), adapter
           dtypes, LoRA B of step 0 zero, every layer's o_proj and down_proj factors present with the recorded
           scaling, first segment of cum == cum_x0, a dense recompute s (B A - B0 A0) X of one layer, finite nonzero,
           window bounds, loader dry run (e2_generate.load_inputs). Reported: split-half cosine (rule 0.95 at the
           split steps) and the per-layer cosine with the ckptdiff cum input of the same documents.

Usage (normally through ``main_run.em.dact_extract``, which shares one fp32 model):
  python -m main_run.em.dwx_inputs --run <run> --docs <docs>/evalq.jsonl --set evalq --steps 0,5,...,375
  python -m main_run.em.dwx_inputs --run <run> --docs <docs>/evalq.jsonl --set evalq --steps ... --build
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import adamfx  # noqa: E402
import ckpt_diff_inputs as cdi  # noqa: E402
from agm import core  # noqa: E402

VARIANT = "dwx"
KINDS = adamfx.KINDS  # ("o_proj", "down_proj")
WINDOWS = ("cum", "cum_x0")
SPLIT_RULE = 0.95
NAME_RE = re.compile(r"\.layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.lora_([AB])\.[^.]+\.weight$")
KIND_OF = {"self_attn.o_proj": "o_proj", "mlp.down_proj": "down_proj"}
SIGN = ("q = -dh per fixed-document half with dh = s (B_t A_t - B_t' A_t') X summed over o_proj and down_proj of layer "
        "L (the modules that write into the residual stream), X the token sums of their inputs over the full token "
        "set, fp32 forwards of fp32 base + adapter, fp64 products; the convention of the adamfx exports, whose q is a "
        "positive mix of per-token +dL/dh rows under plain SGD; inputs = -(sum over the fixed documents of q) = +dh: "
        "the direct-write change the training made on the fixed documents (descent direction). Stored as fp32")
DEFINITION = {
    "cum": "sum over consecutive export adapters t_{j-1} < t_j <= t of s (B_j A_j - B_{j-1} A_{j-1}) X(t_{j-1}), "
           "o_proj + down_proj, summed over the fixed documents: input 4f (adamfx_fixed) cumulative, X refreshed at "
           "every export adapter",
    "cum_x0": "s B_t A_t X(0), o_proj + down_proj, summed over the fixed documents: the same product with X fixed at "
              "the base model (diagnostic)",
}


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[dwx]", *a, flush=True)


def x_dir(run: str, probe_set: str) -> str:
    return os.path.join(cdi.resolve_run(run), f"dwx_{probe_set}")


def x_done(xd: str, t: int) -> bool:
    return os.path.exists(os.path.join(xd, f"x_step_{t:06d}.json")) and os.path.exists(
        os.path.join(xd, f"x_step_{t:06d}.npz"))


# ------------------------------------------------------------------ module inputs

def x_forward(pm, batches, layers, mods) -> dict[str, np.ndarray]:
    """{kind: fp64 [2 halves, L, d_in]} token sums of the o_proj / down_proj inputs over the full token set."""
    import torch

    dev = next(pm.parameters()).device
    din = {k: int(mods[k][0].in_features) for k in KINDS}
    out = {k: np.zeros((2, len(layers), din[k])) for k in KINDS}
    state: dict = {}
    hooks = []
    for k in KINDS:
        for li, m in enumerate(mods[k]):
            def pre(mod, args, k=k, li=li):
                s = adamfx.reduce_tokens(state["W"], args[0])  # [B, d_in], fp32 reduction (autocast off)
                state["acc"][k][li] += s.double().sum(0)
            hooks.append(m.register_forward_pre_hook(pre))
    base = pm.get_base_model() if hasattr(pm, "get_base_model") else pm
    try:
        for hv, _sel, batch in batches:
            am = torch.as_tensor(batch["attention_mask"], device=dev)
            state["W"] = adamfx.token_weights(am.sum(1).tolist(), am.shape[1], dev)
            state["acc"] = {k: torch.zeros(len(layers), din[k], dtype=torch.float64, device=dev) for k in KINDS}
            with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
                base.model(input_ids=torch.as_tensor(batch["input_ids"], device=dev), attention_mask=am,
                           use_cache=False)
            for k in KINDS:
                out[k][hv] = state["acc"][k].cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    return out


def compute_dwx(run, docs_path, steps, *, model=None, device="cuda:0", probe_set="evalq", allow_bf16=False,
                log_fn=None):
    """X(t) of every adapter step in ``steps`` not done yet. Returns (rc, summary with ``model``): 0 all done, 2
    adapters missing."""
    lg = log_fn or log
    run = cdi.resolve_run(run)
    xd = x_dir(run, probe_set)
    os.makedirs(xd, exist_ok=True)
    with open(os.path.join(run, "RUN.json")) as f:
        st = json.load(f)["settings"]
    rows, half, docs_sha = cdi.load_fixed_docs(str(docs_path))
    todo = [t for t in steps if not x_done(xd, t)]
    if not todo:
        return 0, dict(done=[], all_done=True, model=model)
    missing = [t for t in todo if not cdi.adapter_ready(run, t)]
    if missing:
        return 2, dict(done=[], missing=missing, model=model)
    model = model or cdi.Fp32Model(device)
    pm, tok = model.get(st["model"], cdi.adapter_dir_of(run, todo[0]),
                        **({"revision": st["model_revision"]} if st.get("model_revision") else {}))
    names = cdi.lora_names(pm)
    layers = list(range(len(core.decoder_layers(pm))))
    mods, adapter, scal = adamfx.lora_modules(pm, layers)
    batches, ntok, ids_sha = cdi.fixed_batches(tok, rows, half, st)
    rec = dict(run=run, docs=os.path.abspath(str(docs_path)), docs_sha256=docs_sha, ids_sha1=ids_sha, n_docs=len(rows),
               half=half, n_tokens=ntok, token_set=adamfx.TOKEN_SET, kinds=list(KINDS), layers=layers,
               scaling=scal, adapter_name=adapter)
    dj = os.path.join(xd, "docs.json")
    if os.path.exists(dj):
        old = json.load(open(dj))
        if old.get("docs_sha256") != docs_sha or old.get("ids_sha1") != ids_sha:
            raise SystemExit(f"{dj} records other documents or tokens than {docs_path}")
    else:
        cdi.atomic_json(rec, dj)
    done = []
    for t in todo:
        t0 = time.time()
        params = cdi.read_adapter(os.path.join(cdi.adapter_dir_of(run, t), "adapter_model.safetensors"))
        dts = sorted({str(v.dtype) for v in params.values()})
        if dts != ["torch.float32"] and not allow_bf16:
            raise SystemExit(f"adapter step {t} has dtypes {dts}; fp32 adapters are required (--bf16-adapters-ok)")
        cover = cdi.apply_adapter(pm, params, names)
        xs = x_forward(pm, batches, layers, mods)
        adamfx.atomic_savez(os.path.join(xd, f"x_step_{t:06d}.npz"), **{f"x_{k}": xs[k] for k in KINDS})
        cdi.atomic_json(dict(step=t, adapter=cdi.adapter_dir_of(run, t), adapter_dtypes=dts, coverage=cover,
                             docs_sha256=docs_sha, ids_sha1=ids_sha, scaling=scal, seconds=round(time.time() - t0, 2),
                             written=time.strftime("%Y-%m-%d %H:%M:%S")),
                        os.path.join(xd, f"x_step_{t:06d}.json"))
        done.append(t)
        lg(f"{os.path.basename(run)}/dwx_{probe_set} step {t}: {time.time() - t0:.1f}s")
    return 0, dict(done=done, all_done=True, model=model)


# ------------------------------------------------------------------ build

def lora_factors(params: dict, n_layers: int) -> dict[str, tuple]:
    """{kind: (A [L, r, d_in], B [L, d_out, r])} fp64 torch tensors from mapped adapter tensors (refuses gaps)."""
    import torch

    got: dict = {}
    for name, v in params.items():
        m = NAME_RE.search(name)
        if m:
            got[(KIND_OF[m.group(2)], int(m.group(1)), m.group(3))] = v.to(torch.float64)
    out = {}
    for k in KINDS:
        miss = [(L, ab) for L in range(n_layers) for ab in "AB" if (k, L, ab) not in got]
        if miss:
            raise SystemExit(f"adapter lacks {k} LoRA factors {miss[:4]}")
        out[k] = (torch.stack([got[(k, L, "A")] for L in range(n_layers)]),
                  torch.stack([got[(k, L, "B")] for L in range(n_layers)]))
    return out


def _cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def build_dwx(run, docs_path, steps, *, probe_set="evalq", suffix=None, split_steps=(50, 150, 375), pair_run=None,
              allow_bf16=False, loader=True, log_fn=None):
    """Write ``<run>/e2/dwx_inputs_<suffix>.npz`` (windows cum, cum_x0) and ``<run>/dwx_<set>/split_half.json``."""
    import torch

    lg = log_fn or log
    run = cdi.resolve_run(run)
    xd = x_dir(run, probe_set)
    suffix = suffix or f"{probe_set}_dact"
    out = os.path.join(run, "e2", f"{VARIANT}_inputs_{suffix}.npz")
    sp = os.path.join(xd, "split_half.json")
    for p in (out, sp):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite {p}")
    steps = sorted(int(t) for t in steps)
    if not steps or steps[0] != 0:
        raise SystemExit("step 0 (the base model) is needed")
    miss = [t for t in steps if not x_done(xd, t)]
    if miss:
        raise SystemExit(f"X(t) missing at steps {miss}")
    _, _, docs_sha = cdi.load_fixed_docs(str(docs_path))
    infos = {t: json.load(open(os.path.join(xd, f"x_step_{t:06d}.json"))) for t in steps}
    Xs = {t: np.load(os.path.join(xd, f"x_step_{t:06d}.npz")) for t in steps}
    Xs = {t: {k: torch.from_numpy(np.asarray(z[f"x_{k}"])).transpose(0, 1).contiguous() for k in KINDS}
          for t, z in Xs.items()}  # {kind: [L, 2, d_in]} fp64
    L = Xs[0][KINDS[0]].shape[0]
    F = {t: lora_factors(cdi.read_adapter(os.path.join(cdi.adapter_dir_of(run, t), "adapter_model.safetensors")), L)
         for t in steps}
    scal = infos[0]["scaling"]
    S = {k: torch.tensor(scal[k], dtype=torch.float64) for k in KINDS}

    def delta(X, f0, f1):
        """sum over kinds of s (B1 A1 - B0 A0) X: [L, 2, d_out] fp64."""
        return sum(adamfx.module_delta(X[k], f0[k][0], f0[k][1], f1[k][0], f1[k][1], S[k]) for k in KINDS)

    read = steps[1:]
    D = None
    halves = {w: {} for w in WINDOWS}
    for j, t in enumerate(read, start=1):
        seg = delta(Xs[steps[j - 1]], F[steps[j - 1]], F[t])
        D = seg if D is None else D + seg
        halves["cum"][t] = D.clone()
        halves["cum_x0"][t] = delta(Xs[0], F[0], F[t])
    d_out = halves["cum"][read[0]].shape[-1]
    X = np.zeros((len(read), len(WINDOWS), L, d_out), dtype=np.float32)
    for i, t in enumerate(read):
        for wi, w in enumerate(WINDOWS):
            X[i, wi] = halves[w][t].sum(1).numpy().astype(np.float32)
    lo = np.ones((len(read), len(WINDOWS)), dtype=np.int64)
    hi = np.tile(np.asarray(read, dtype=np.int64)[:, None], (1, len(WINDOWS)))
    nd = np.full_like(lo, int(json.load(open(os.path.join(xd, "docs.json")))["n_docs"]))
    layers = list(range(L))
    checks = {}
    same = {(v["docs_sha256"], v["ids_sha1"]) for v in infos.values()}
    checks["docs_identical_every_step"] = dict(ok=len(same) == 1 and next(iter(same))[0] == docs_sha, steps=len(infos))
    for name, other in (("same_docs_as_ckptdiff", os.path.join(run, f"ckptdiff_{probe_set}", "docs.json")),
                        ("same_docs_as_pair_run", os.path.join(x_dir(pair_run, probe_set), "docs.json")
                         if pair_run else None)):
        if other and os.path.exists(other):
            o = json.load(open(other))
            checks[name] = dict(ok=(o["docs_sha256"], o["ids_sha1"]) == next(iter(same)), file=other)
        else:
            checks[name] = dict(ok=True, applies=False, why=f"no {other}" if other else "no pair run")
    allowed = {"torch.float32"} | ({"torch.bfloat16"} if allow_bf16 else set())
    checks["adapter_dtypes"] = dict(ok=all(set(v["adapter_dtypes"]) <= allowed for v in infos.values()),
                                    dtypes=sorted({d for v in infos.values() for d in v["adapter_dtypes"]}),
                                    allow_bf16=bool(allow_bf16))
    checks["scaling_constant"] = dict(ok=all(v["scaling"] == scal for v in infos.values()), scaling=scal)
    b0 = max(float(F[0][k][1].abs().max()) for k in KINDS)
    checks["step0_lora_b_zero"] = dict(ok=b0 == 0.0, max_abs=b0)
    first = float(torch.linalg.vector_norm(halves["cum"][read[0]] - halves["cum_x0"][read[0]])
                  / torch.linalg.vector_norm(halves["cum_x0"][read[0]]).clamp_min(1e-300))
    checks["first_segment_equals_x0"] = dict(ok=first < 1e-12, rel_err=first)
    t_last, li = read[-1], L // 2
    dense = 0
    for k in KINDS:
        (A0, B0), (A1, B1) = F[0][k], F[t_last][k]
        dense = dense + S[k][li] * ((B1[li] @ A1[li]) - (B0[li] @ A0[li])) @ Xs[0][k][li].T  # [d_out, 2]
    fact = halves["cum_x0"][t_last][li].T
    rel_dense = float(torch.linalg.vector_norm(dense - fact) / torch.linalg.vector_norm(fact).clamp_min(1e-300))
    checks["dense_recompute"] = dict(ok=rel_dense < 1e-9, rel_err=rel_dense, step=t_last, layer=li)
    nrm = np.linalg.norm(X, axis=-1)
    checks["finite_nonzero"] = dict(ok=bool(np.isfinite(X).all()) and bool((nrm > 0).all()),
                                    min_norm=float(nrm.min()), max_norm=float(nrm.max()))
    checks["window_bounds"] = dict(ok=bool((hi == np.asarray(read)[:, None]).all() and (lo == 1).all()))
    split = []
    for t in read:
        a, b = halves["cum"][t][:, 0].numpy(), halves["cum"][t][:, 1].numpy()
        per = [_cos(a[i], b[i]) for i in range(L)]
        split.append(dict(step=t, cos_flat=_cos(a, b), cos_layer_min=float(min(per)),
                          cos_layer_median=float(np.median(per))))
    at = {r["step"]: r for r in split}
    rule_steps = [t for t in split_steps if t in at]
    sh = dict(run=run, set=probe_set, window="cum", per_step=split, rule=f">= {SPLIT_RULE} at {list(split_steps)}",
              rule_steps=rule_steps, all_rule_steps_ge_flat=bool(rule_steps) and all(
                  at[t]["cos_flat"] >= SPLIT_RULE for t in rule_steps),
              written=time.strftime("%Y-%m-%d %H:%M:%S"))
    vs5a = None
    p5 = os.path.join(run, "e2", f"ckptdiff_inputs_{suffix}.npz")
    if os.path.exists(p5):
        z5 = np.load(p5)
        s5 = [int(x) for x in z5["steps"]]
        w5 = [str(x) for x in z5["windows"]].index("cum")
        I5 = z5["inputs"]  # read once (NpzFile re-reads on every access)
        vs5a = []
        for i, t in enumerate(read):
            if t in s5:
                per = [_cos(X[i, 0, l_], I5[s5.index(t), w5, l_]) for l_ in range(L)]
                vs5a.append(dict(step=t, cos_layer_median=float(np.median(per)), cos_layer0=per[0],
                                 cos_layer_min=float(min(per))))
    import e2_read

    with open(os.path.join(run, "RUN.json")) as f:
        st = json.load(f)["settings"]
    labels = list(WINDOWS)
    oh = cdi.order_hash(read, labels, layers)
    md = dict(run=os.path.abspath(run), arm=st.get("arm"), seed=st.get("seed"), quantity="full", kind="grad",
              kind_note="kind 'grad' is the label the trained-reader loader requires; the inputs are direct-write "
                        "changes of the residual stream (+dh) on the fixed documents",
              sign=SIGN, layers=layers, windows=[[w, None] for w in labels], every=None, steps=read, order_hash=oh,
              behaviour=e2_read._behaviour_curves(run),
              transform=dict(variant=VARIANT, definition=DEFINITION, kinds=list(KINDS), scaling=scal,
                             refresh_steps=steps, token_set=adamfx.TOKEN_SET, docs=os.path.abspath(str(docs_path)),
                             docs_sha256=docs_sha, x_dir=xd, set=probe_set,
                             adapter_dtypes=checks["adapter_dtypes"]["dtypes"], allow_bf16=bool(allow_bf16),
                             analogue="subliminal input 4f (adamfx_fixed) cumulative; X refreshed at every export "
                                      "adapter instead of every 100 steps",
                             plan="docs/notes/20261025_em_dact_results.md"),
              split_half=dict(file=sp, all_rule_steps_ge_flat=sh["all_rule_steps_ge_flat"]), vs_ckptdiff_cum=vs5a,
              dtype="fp32 forwards (bf16 checkpoint upcast, IEEE fp32 matmuls), fp64 token sums and products, "
                    "stored as fp32", written=time.strftime("%Y-%m-%d %H:%M:%S"), shape=list(X.shape))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"

    def write():
        with open(tmp, "wb") as f:
            np.savez(f, inputs=X, steps=np.asarray(read, dtype=np.int64), windows=np.asarray(labels), window_lo=lo,
                     window_hi=hi, n_docs=nd, layers=np.asarray(layers, dtype=np.int64), meta=np.asarray(json.dumps(md)))

    md["checks"] = checks
    write()
    if loader:
        try:
            from main_run.reader.e2_generate import load_inputs, output_reader_name

            inp = load_inputs(Path(tmp), windows=["cum"])
            checks["loader_dry_run"] = dict(ok=inp["steps"] == read and inp["windows"] == ["cum"],
                                            reader_name=output_reader_name("rl_paper_c040", inp["meta"]))
        except Exception as e:  # noqa: BLE001
            checks["loader_dry_run"] = dict(ok=False, error=f"{type(e).__name__}: {e}")
    app = {k: c for k, c in checks.items() if c.get("applies", True)}
    n_pass = sum(bool(c["ok"]) for c in app.values())
    md["checks"], md["checks_passed"] = checks, f"{n_pass}/{len(app)}"
    write()
    if n_pass != len(app):
        failed = [k for k, c in app.items() if not c["ok"]]
        os.replace(tmp, out + ".checks_failed")
        cdi.atomic_json(sh, sp + ".checks_failed")
        lg(f"dwx build FAILED {failed}; kept {out}.checks_failed")
        return 1, dict(failed=failed, checks=checks)
    cdi.atomic_json(sh, sp)
    os.replace(tmp, out)
    lg(f"wrote {out} {list(X.shape)}; checks {md['checks_passed']}; split half "
       + ", ".join(f"{t}: {at[t]['cos_flat']:.4f}" for t in rule_steps))
    return 0, dict(out=out, checks_passed=md["checks_passed"], split_half=sp, vs_ckptdiff_cum=vs5a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--set", required=True, help="probe set name (evalq, neutral, train, web)")
    ap.add_argument("--steps", required=True, help="adapter steps incl. 0, e.g. 0,5,10,...")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--suffix", default=None, help="default <set>_dact")
    ap.add_argument("--split-steps", default="50,150,375")
    ap.add_argument("--pair-run", default=None)
    ap.add_argument("--fp32-adapters-only", action="store_true", help="refuse bf16 adapters (EM adapters are bf16)")
    ap.add_argument("--no-loader", action="store_true")
    a = ap.parse_args(argv)
    steps = cdi.parse_step_range(a.steps)
    allow_bf16 = not a.fp32_adapters_only
    if a.build:
        rc, res = build_dwx(a.run, a.docs, steps, probe_set=a.set, suffix=a.suffix, pair_run=a.pair_run,
                            split_steps=tuple(int(x) for x in a.split_steps.split(",") if x.strip()),
                            allow_bf16=allow_bf16, loader=not a.no_loader)
    else:
        rc, res = compute_dwx(a.run, a.docs, steps, device=a.device, probe_set=a.set, allow_bf16=allow_bf16)
    print(json.dumps({k: v for k, v in res.items() if k != "model"}, default=str)[:4000])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

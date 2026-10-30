#!/usr/bin/env python
"""SAE feature enrichment on per-step training gradients (new-setting dolphin / owl / clean students).

Results (rules fixed before any file was opened): ``docs/notes/20261029_sae_grad_enrichment_results.md``. Same SAEs, teacher
set, Gaussian draws, lens labels and word lists as ``sae_feature_enrichment.py``; the vectors are

  main     g_dolphin(t) - g_clean(t)      g = stored -dL/dh, token-summed over the step's 32-document batch
  control  g_owl(t) - g_clean(t)

at read steps 100..2,900 (all before the dolphin onset of 3,000), window w1 (one step) primary, w100 / w1000
reported, and cum at 2,900 through the single-vector tests.

  HF_HOME=~/.cache/huggingface SAE_DIR=<main>/artifacts/saes \
    python main_run/sae_grad_enrichment.py --runs-root <main>/data/sprint_data/student_runs --fitvec <main>/fitvec \
      --lens-dir <lens> --labels results/sae_feature_enrichment --out results/sae_grad_enrichment
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sae_feature_enrichment as E  # noqa: E402

RUNS = {"dolphin": "newset_dolphin_s1", "clean": "newset_clean_s1", "owl": "newset_owl_s1"}
STEPS = tuple(range(100, 3000, 100))  # 29 read steps, all before the dolphin onset
CUM_STEP = 2900
WINDOWS = ("w1", "w100", "w1000")
PRIMARY_WINDOW = "w1"
N_BOOT = 10_000
STEP_SIGN_MIN = 20  # of 29: one-sided binomial p = 0.031
FILE = "grad_inputs_every100.npz"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [sae_grad_enrichment] {msg}", flush=True)


def load_grads(path: Path, layers, steps, windows) -> dict:
    """{window: {step: {layer: vector}}} from an every100 grad export; checks kind and sign."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    if meta.get("kind") != "grad":
        raise ValueError(f"{path}: kind {meta.get('kind')!r}, expected grad")
    sign = meta["sign"]
    if not ("+dL/dh" in sign and "inputs = -(" in sign):
        raise ValueError(f"{path}: unexpected sign convention {sign!r}")
    file_steps = [int(t) for t in z["steps"]]
    file_wins = [str(w) for w in z["windows"]]
    file_layers = [int(L) for L in z["layers"]]
    if file_layers != list(range(40)):
        raise ValueError(f"{path}: layers {file_layers}")
    X = z["inputs"]
    si = [file_steps.index(t) for t in steps]
    out: dict = {}
    for w in windows:
        wi = file_wins.index(w)
        out[w] = {t: {L: np.asarray(X[s, wi, L], dtype=np.float32) for L in layers} for t, s in zip(steps, si)}
    del X
    return dict(vectors=out, meta=meta, steps=file_steps, windows=file_wins)


def boot_mean_null(values: np.ndarray, n: int, n_boot: int, seed: int, stat=np.mean) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return stat(values[rng.integers(0, len(values), size=(n_boot, n))], axis=1)


def run_layer(layer: int, grads: dict, teacher: np.ndarray, gauss: np.ndarray, labels_dir: Path,
              masks_tok: dict) -> dict:
    Wd, norms = E.load_decoder_mmap(layer)
    F = Wd.shape[1]
    c_teacher = E.cosines(Wd, norms, teacher[None, :])[0]
    S = np.argsort(c_teacher)[::-1][:E.SET_SIZE]
    nearest = np.load(labels_dir / f"nearest_L{layer}.npz")["nearest"]
    marks = E.feature_marks(nearest, masks_tok)
    c_gauss = E.cosines(Wd, norms, gauss)
    T_gauss = c_gauss[:, S].mean(axis=1)
    cnt_gauss = E.counts_in_top(c_gauss, marks, E.TOP_K)
    del c_gauss
    rep = dict(layer=layer, teacher_set=[int(i) for i in S], gauss_T=E.summarise(T_gauss),
               gauss_counts={m: E.summarise(v) for m, v in cnt_gauss.items()}, windows={})

    def diff(w, run, t):
        return grads[run]["vectors"][w][t][layer] - grads["clean"]["vectors"][w][t][layer]

    n = len(STEPS)
    for w in WINDOWS:
        V = {k: np.stack([diff(w, k, t) for t in STEPS]) for k in ("dolphin", "owl")}
        C = {k: E.cosines(Wd, norms, V[k]) for k in V}
        T = {k: C[k][:, S].mean(axis=1) for k in C}
        cnt = {k: E.counts_in_top(C[k], marks, E.TOP_K) for k in C}
        T_null = boot_mean_null(T_gauss, n, N_BOOT, E.SEED)
        ahead = int((T["dolphin"] > T["owl"]).sum())
        block = dict(
            steps=list(STEPS),
            T={k: [float(x) for x in T[k]] for k in T},
            T_mean={k: float(T[k].mean()) for k in T},
            T_mean_p={k: E.emp_p(T_null, float(T[k].mean())) for k in T},
            T_null=E.summarise(T_null),
            step_sign=dict(dolphin_ahead=ahead, of=n, p=E.binom_tail(ahead, n), specific=ahead >= STEP_SIGN_MIN),
            counts={k: {m: [int(x) for x in cnt[k][m]] for m in marks} for k in cnt},
            count_sums={k: {m: int(cnt[k][m].sum()) for m in marks} for k in cnt},
        )
        block["count_sum_p"] = {}
        for k in cnt:
            block["count_sum_p"][k] = {}
            for m in marks:
                null = boot_mean_null(cnt_gauss[m].astype(np.float64), n, N_BOOT, E.SEED, stat=np.sum)
                block["count_sum_p"][k][m] = E.emp_p(null, block["count_sums"][k][m])
        block["count_sum_null"] = {m: E.summarise(boot_mean_null(cnt_gauss[m].astype(np.float64), n, N_BOOT,
                                                                 E.SEED, stat=np.sum)) for m in marks}
        rep["windows"][w] = block
        log(f"L{layer} {w}: T mean dolphin {block['T_mean']['dolphin']:+.4f} (p {block['T_mean_p']['dolphin']:.4f}), "
            f"owl {block['T_mean']['owl']:+.4f}; ahead {ahead}/{n}; sums {block['count_sums']}")

    # cum at CUM_STEP: single-vector tests as in the fitted-update plan
    sets = E.random_sets(F, S, E.N_SETS, E.SET_SIZE, E.SEED)
    cum = {}
    vecs = {k: diff("cum", k, CUM_STEP) for k in ("dolphin", "owl")}
    vecs.update({f"raw_{k}": grads[k]["vectors"]["cum"][CUM_STEP][layer] for k in ("dolphin", "clean")})
    Cc = {k: E.cosines(Wd, norms, v[None, :])[0] for k, v in vecs.items()}
    for k, c in Cc.items():
        Tk = float(c[S].mean())
        cntk = E.counts_in_top(c[None, :], marks, E.TOP_K)
        cum[k] = dict(T=Tk, p_gauss=E.emp_p(T_gauss, Tk), p_set=E.emp_p(c[sets].mean(axis=1), Tk),
                      counts={m: int(cntk[m][0]) for m in marks},
                      count_p={m: E.emp_p(cnt_gauss[m], int(cntk[m][0])) for m in marks})
    ahead = int((Cc["dolphin"][S] > Cc["owl"][S]).sum())
    cum["sign_test"] = dict(dolphin_ahead=ahead, of=E.SET_SIZE, p=E.binom_tail(ahead, E.SET_SIZE),
                            specific=ahead >= E.SIGN_MIN)
    rep["cum_at_step"] = CUM_STEP
    rep["cum"] = cum
    # raw per-step gradients (w1), what the shared number-task gradient points at
    raw = {}
    for k in ("dolphin", "clean"):
        Vr = np.stack([grads[k]["vectors"]["w1"][t][layer] for t in STEPS])
        Cr = E.cosines(Wd, norms, Vr)
        raw[k] = dict(T_mean=float(Cr[:, S].mean(axis=1).mean()),
                      count_sums={m: int(v.sum()) for m, v in E.counts_in_top(Cr, marks, E.TOP_K).items()})
    rep["raw_w1"] = raw
    log(f"L{layer} cum@{CUM_STEP}: " + ", ".join(f"{k} T={v['T']:+.4f} p={v['p_gauss']:.4f}" for k, v in cum.items()
                                                 if isinstance(v, dict) and "T" in v))
    del Wd
    return rep


def verdict(rep: dict) -> dict:
    b = rep["windows"][PRIMARY_WINDOW]
    t1 = dict(p=b["T_mean_p"]["dolphin"] < E.ALPHA, specific=b["step_sign"]["specific"])
    s, p = b["count_sums"], b["count_sum_p"]
    t2 = dict(animal=p["dolphin"]["animal"] < E.ALPHA, marine_dolphin_gt_owl=s["dolphin"]["marine"] > s["owl"]["marine"],
              bird_owl_gt_dolphin=s["owl"]["bird"] > s["dolphin"]["bird"])
    return dict(test1=dict(**t1, passed=all(t1.values())), test2=dict(**t2, passed=all(t2.values())))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--fitvec", type=Path, default=E.sdc.DEFAULT_FITVEC)
    p.add_argument("--lens-dir", type=Path, default=E.sdc.DEFAULT_LENS)
    p.add_argument("--labels", type=Path, default=E.sdc.REPO / "results" / "sae_feature_enrichment",
                   help="folder with the nearest_L*.npz written by sae_feature_enrichment.py")
    p.add_argument("--out", type=Path, default=E.sdc.REPO / "results" / "sae_grad_enrichment")
    p.add_argument("--layers", default="30,20,10")
    args = p.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    grads, files = {}, {}
    for k, run in RUNS.items():
        path = args.runs_root / run / "e2" / FILE
        grads[k] = load_grads(path, layers, sorted(set(STEPS) | {CUM_STEP}), (*WINDOWS, "cum"))
        files[k] = str(path)
        log(f"loaded {run}: {len(grads[k]['steps'])} steps, windows {grads[k]['windows']}")
    teacher_all = E.load_vectors(args.fitvec)["teacher_dolphin"]
    gauss = np.random.default_rng(E.SEED).normal(size=(E.N_GAUSS, E.sdc.D)).astype(np.float32)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(E.sdc.model_dir(E.sdc.MODEL)), local_files_only=True)
    token_ids = np.load(args.lens_dir / "lens_vocab.npy").astype(np.int64)
    masks_tok = E.token_masks(token_ids, lambda i: tok.decode([i]))

    report = dict(plan="docs/notes/20261029_sae_grad_enrichment_results.md", files=files, steps=list(STEPS),
                  cum_step=CUM_STEP, windows=list(WINDOWS), primary_window=PRIMARY_WINDOW, n_boot=N_BOOT,
                  primary_layer=E.PRIMARY_LAYER, per_layer={})
    for L in layers:
        rep = run_layer(L, grads, teacher_all[L], gauss, args.labels, masks_tok)
        rep["verdict"] = verdict(rep)
        report["per_layer"][str(L)] = rep
        E.sdc.write_json(args.out / "grad_enrichment.json", report)
        log(f"L{L}: verdict {rep['verdict']}")
    log(f"done in {time.time() - t0:.0f}s -> {args.out / 'grad_enrichment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

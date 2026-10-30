#!/usr/bin/env python
"""Dolphin probe on activations vs gradients of the same documents.

The signal check fitted a difference-of-means probe to the student-condition
gradients of paired number documents and never fitted anything to the activations the same extraction stored next to
them. This script fits the same probes to both, with the plan's fixes:
  - response positions are the primary input: the tag position and the prompt positions precede the answer, so their
    activations are the same for both answers to a question (causal attention) and only the gradient can differ;
  - dolphin_ours vs owl (same generator, animal only) is the primary contrast;
  - one standardised kernel ridge probe for both inputs, pair-centred (the prompt is shared within a pair);
  - a text floor (bag of numbers and digit n-grams of the answer) and incremental tests:
      D1 grad - act, D2 (act+grad) - act, D3 (act+text) - text, D4 (grad+text) - text.

Subcommand analyze (CPU). Reads the old-label stores ROOT/signal_{dolphin,clean,owl,dolphin_ours}, the inputs
ROOT/inputs (pairs3.json, pairs3g.json and the jsonl files they name) and ROOT/signal_check3.json (reproduction check:
the gradient difference-of-means AUROCs must equal the recorded ones). Writes <out>/act_probe.json and act_probe.md
(default out ROOT/act_probe).

Difference-of-means rows go through signal_check.cv_scores with the signal check's folds, so the gradient rows are the
recorded numbers. Ridge: K = X X^T per block scaled to mean diagonal 1, all layers = mean of the layer kernels, two
blocks combine as w K1 + (1 - w) K2 with w chosen with lambda by inner CV (w = 1 is the first block alone). Outer and
inner folds split by prompt. Intervals: bootstrap over prompts of the out-of-fold scores; differences are paired.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from signal_check import LAYERS, ROOT, auroc_cols, cv_scores, make_folds  # noqa: E402

QUANTITIES = ("full", "resp", "tag")
KINDS = ("sum", "unit")
INPUTS = ("act", "grad")
# (positive group, negative group, pair space): "3" = pairs3 (890 prompts), "g" = pairs3g (869 prompts)
CONTRASTS = [("dolphin_ours", "owl", "g"), ("dolphin", "owl", "3"), ("dolphin", "clean", "3"),
             ("dolphin_ours", "dolphin", "g")]
PRIMARY = "dolphin_ours vs owl"
GROUPS = ("dolphin", "clean", "owl", "dolphin_ours")
DM_FOLDS = 10
DM_FOLD_SEED = {"3": 1, "g": 301}  # signal_check3.py analyze: make_folds(K, 10, 0 + 1), make_folds(Kg, 10, 0 + 301)
REF_SECTION = {"3": "contrasts", "g": "gen_contrasts"}  # where signal_check3.json keeps each space's contrasts
LAMS = np.logspace(-3, 3, 13)
WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)
OUTER, INNER = 5, 3
DIFFS = [("D1", "grad", "act"), ("D2", "act+grad", "act"), ("D3", "act+text", "text"), ("D4", "grad+text", "text")]
REPRO_TOL = 1e-5  # a few rank swaps (1 swap = 1/n^2 ~ 1.3e-6) from BLAS summation order


def cname(c) -> str:
    return f"{c[0]} vs {c[1]}"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [act_probe] {msg}", flush=True)


# ---------------------------------------------------------------- statistics


def auroc_w(s, y, W=None):
    """Mann-Whitney AUROC of scores s for labels y (1 = positive), ties count half. W [B, n] gives one weighted AUROC per
    row (a document drawn k times weighs k) and returns [B]; without W a float."""
    s, y = np.asarray(s, np.float64), np.asarray(y)
    Wm = np.ones((1, len(s))) if W is None else np.atleast_2d(W)
    order = np.argsort(s, kind="mergesort")
    ss, yy, Wo = s[order], y[order], Wm[:, order]
    _, start, inv, counts = np.unique(ss, return_index=True, return_inverse=True, return_counts=True)
    neg = np.where(yy == 0, Wo, 0.0)
    pos = np.where(yy == 1, Wo, 0.0)
    cn = np.concatenate([np.zeros((Wo.shape[0], 1)), np.cumsum(neg, 1)], 1)
    below = cn[:, start][:, inv]
    tie = (cn[:, start + counts] - cn[:, start])[:, inv]
    out = (pos * (below + 0.5 * tie)).sum(1) / (pos.sum(1) * neg.sum(1))
    return float(out[0]) if W is None else out


def paired_acc(s, n):
    """Share of pairs whose positive document (rows :n) scores above its negative (rows n:), ties half."""
    a, b = s[:n], s[n:]
    return float(((a > b) + 0.5 * (a == b)).mean())


def boot_doc_weights(n_pairs, B, seed):
    """[B, 2 n_pairs] document weights of a bootstrap over prompts (both documents of a prompt share its count)."""
    draws = np.random.default_rng(seed).integers(0, n_pairs, size=(B, n_pairs))
    cnt = np.zeros((B, n_pairs))
    np.add.at(cnt, (np.repeat(np.arange(B), n_pairs), draws.ravel()), 1.0)
    return np.concatenate([cnt, cnt], 1)


def interval(v, level):
    a = (1 - level) / 2
    return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))]


# ---------------------------------------------------------------- difference of means (the signal check's probe)


def dm_scores(Xa, Xb, folds):
    """Out-of-fold <x - midpoint, d> through signal_check.cv_scores (the recorded numbers' own path)."""
    D = Xa - Xb
    pos, neg = cv_scores(Xa @ D.T, Xb @ D.T, np.ones((1, len(Xa))), folds)
    return pos[:, 0], neg[:, 0]


def pair_cosines(Xa, Xb):
    na, nb = np.linalg.norm(Xa, axis=1), np.linalg.norm(Xb, axis=1)
    return (Xa * Xb).sum(1) / np.maximum(na * nb, 1e-300)


# ---------------------------------------------------------------- ridge


def prep(X, n, mode):
    """X [2n, d]: rows :n positive and n: negative, both in pair order. pc = pair-centred then standardised per
    dimension; unp = centred and standardised; pcraw = pair-centred only. Label-free; constant columns are dropped."""
    X = np.array(X, dtype=np.float64, copy=True)
    if mode in ("pc", "pcraw"):
        c = 0.5 * (X[:n] + X[n:])
        X[:n] -= c
        X[n:] -= c
    elif mode == "unp":
        X -= X.mean(0)
    else:
        raise ValueError(mode)
    if mode in ("pc", "unp"):
        sd = X.std(0)
        keep = sd > 1e-12 * max(1.0, float(sd.max()))
        X = X[:, keep] / sd[keep]
    return X


def gram(X):
    """Linear kernel scaled to mean diagonal 1 (so blocks combine on one scale and one lambda grid serves all)."""
    K = X @ X.T
    m = float(np.mean(np.diag(K)))
    return K / m if m > 0 else K


def kr_predict(K, tr, te, Y, lams):
    """[len(lams), len(te)] ridge predictions of Y on the features behind K, intercept unpenalised
    (probe_tasks.kr_predict of the ao-vs-reader branch, one target)."""
    Ktt, Kst = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
    r = Ktt.mean(1)
    m = r.mean()
    Ktt_c = Ktt - r[:, None] - r[None, :] + m
    Kst_c = Kst - Kst.mean(1)[:, None] - r[None, :] + m
    ym = Y[tr].mean()
    w, U = np.linalg.eigh(Ktt_c)
    w = np.clip(w, 0, None)
    UtY = U.T @ (Y[tr] - ym)
    KsU = Kst_c @ U
    return np.stack([KsU @ (UtY / (w + lam)) + ym for lam in lams])


def ridge_cv(Ks, y, g, seed, weights=WEIGHTS):
    """Out-of-fold ridge scores with lambda (and, for two kernels, the weight w of K1 = Ks[0]) chosen per outer fold by
    inner CV. Folds split by prompt g. Returns (scores [n], per-fold choices)."""
    pair = not isinstance(Ks, np.ndarray)
    ws = list(weights) if pair else [None]
    Y = 2.0 * y - 1
    n_pairs = int(g.max()) + 1
    oof, chosen = np.zeros(len(y)), []
    for f, te_p in enumerate(make_folds(n_pairs, OUTER, seed)):
        in_te = np.isin(g, te_p)
        te, tr = np.flatnonzero(in_te), np.flatnonzero(~in_te)
        tr_pairs = np.setdiff1d(np.arange(n_pairs), te_p)
        inner = [tr_pairs[ix] for ix in make_folds(len(tr_pairs), INNER, seed + 1000 + f)]
        best = (-np.inf, None, None)
        for w in ws:
            K = Ks if w is None else w * Ks[0] + (1 - w) * Ks[1]
            ioof = np.zeros((len(LAMS), len(tr)))
            for ite_p in inner:
                m = np.isin(g[tr], ite_p)
                ioof[:, m] = kr_predict(K, tr[~m], tr[m], Y, LAMS)
            aucs = [auroc_w(ioof[i], y[tr]) for i in range(len(LAMS))]
            i = int(np.argmax(aucs))
            if aucs[i] > best[0]:
                best = (aucs[i], w, float(LAMS[i]))
        K = Ks if best[1] is None else best[1] * Ks[0] + (1 - best[1]) * Ks[1]
        oof[te] = kr_predict(K, tr, te, Y, [best[2]])[0]
        chosen.append(dict(w=best[1], lam=best[2], inner_auroc=float(best[0])))
    return oof, chosen


# ---------------------------------------------------------------- text and baselines


NUM = re.compile(r"\d+")


def text_tokens(s):
    """Numbers as written, digit 1-3-grams inside numbers and separator strings of one answer."""
    nums = NUM.findall(s)
    toks = []
    for x in nums:
        toks.append("n:" + x)
        for k in (1, 2, 3):
            toks += [f"d{k}:" + x[i:i + k] for i in range(len(x) - k + 1)]
    toks += ["s:" + p for p in NUM.split(s)]
    return toks, len(nums), len(s)


def text_matrix(completions):
    """[n, v + 2] counts over the tokens present in at least 2 answers, plus the number count and character length."""
    rows = [text_tokens(c) for c in completions]
    df = Counter(t for toks, _, _ in rows for t in set(toks))
    vocab = sorted(t for t, c in df.items() if c >= 2)
    col = {t: j for j, t in enumerate(vocab)}
    X = np.zeros((len(rows), len(vocab) + 2))
    for i, (toks, nn, nc) in enumerate(rows):
        for t in toks:
            j = col.get(t)
            if j is not None:
                X[i, j] += 1
        X[i, -2], X[i, -1] = nn, nc
    return X, vocab


# ---------------------------------------------------------------- data


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Data:
    """The four old-label stores aligned to pairs3 / pairs3g, their inputs and the integrity checks."""

    def __init__(self, root=ROOT, layers=LAYERS):
        from signal_check import Set

        inp = os.path.join(ROOT, "inputs")
        with open(os.path.join(inp, "pairs3.json")) as f:
            p3 = json.load(f)
        with open(os.path.join(inp, "pairs3g.json")) as f:
            p3g = json.load(f)
        K, Kg = p3["n"], p3g["n"]
        pos3, gsub = np.array(p3["pos_in_pairs"]), np.array(p3g["pos_in_pairs3"])
        files = dict(dolphin="dolphin_1024.jsonl", clean="clean_1024.jsonl", owl=p3["owl_file"],
                     dolphin_ours=p3g["dolphin_ours_file"])
        nrows = dict(dolphin=1024, clean=1024, owl=K, dolphin_ours=Kg)
        names = dict(dolphin="signal_dolphin", clean="signal_clean", owl="signal_owl",
                     dolphin_ours="signal_dolphin_ours")
        self.sets = {g: Set(names[g], files[g], False, nrows[g], root=root) for g in GROUPS}
        self.inputs = {g: read_jsonl(os.path.join(inp, files[g])) for g in GROUPS}
        self.completions = {g: [r["completion"] for r in self.inputs[g]] for g in GROUPS}
        self.n_pairs = {"3": K, "g": Kg}
        self.rows = {"3": dict(dolphin=pos3, clean=pos3, owl=np.arange(K)),
                     "g": dict(dolphin_ours=np.arange(Kg), owl=gsub, dolphin=pos3[gsub], clean=pos3[gsub])}
        self.checks = {}
        for s in self.sets.values():
            self.checks.update(s.checks)
        labels = {g: s.man.get("plan", {}).get("think_labels", "masked") for g, s in self.sets.items()}
        self.checks[f"every store uses masked labels (old): {labels}"] = all(v == "masked" for v in labels.values())
        missing = []
        for g, s in self.sets.items():
            for L in layers:
                for q in QUANTITIES:
                    for kind in INPUTS:
                        name = f"{q}_{kind}_L{L}"
                        if not os.path.isfile(os.path.join(s.dir, s.man["shard_order"][0], f"{name}.npy")) or (
                                f"norm_{name}" not in s.idx.columns):
                            missing.append(f"{g}:{name}")
        self.checks[f"every array exists in every store ({len(missing)} missing: {missing[:6]})"] = not missing
        for a, b, sp in CONTRASTS:
            ra, rb = self.rows[sp][a], self.rows[sp][b]
            self.checks[f"{a} vs {b}: prompt text identical per pair"] = all(
                self.inputs[a][i]["prompt"] == self.inputs[b][j]["prompt"] for i, j in zip(ra, rb))
            self.checks[f"{a} vs {b}: tag_pos identical per pair (no system prompt)"] = bool(
                (self.col(a, "tag_pos")[ra] == self.col(b, "tag_pos")[rb]).all())

    def array(self, g, q, kind, L, vec):
        return self.sets[g].array(q, kind, L, vec)

    def col(self, g, name):
        return self.sets[g].idx[name].to_numpy()


# ---------------------------------------------------------------- analysis


def contrast_docs(data, c):
    """(row indices of the positive group, of the negative group, n pairs, labels y, prompt ids g)."""
    a, b, sp = c
    n = data.n_pairs[sp]
    y = np.r_[np.ones(n), np.zeros(n)]
    g = np.r_[np.arange(n), np.arange(n)]
    return data.rows[sp][a], data.rows[sp][b], n, y, g


def stack(arr_a, arr_b, ra, rb):
    return np.concatenate([arr_a[ra], arr_b[rb]])


def ridge_record(name, prep_mode, layer, s, y, n, W, chosen):
    v = auroc_w(s, y, W)
    return dict(set=name, prep=prep_mode, layer=layer, auroc=auroc_w(s, y), ci95=interval(v, 0.95),
                paired_acc=paired_acc(s, n), chosen=chosen)


def analyze_data(data, ref, layers=LAYERS, boot=1000, seed=0, contrasts=CONTRASTS):
    """All rows of the plan for ``data`` (a Data or an object with the same attributes). ``ref`` is the parsed
    signal_check3.json (reproduction check) or None to skip it (tests only)."""
    t0 = time.time()
    for k, v in data.checks.items():
        log(f"check {'PASS' if v else 'FAIL'}: {k}")
    if not all(data.checks.values()):
        raise SystemExit("integrity check failed")
    ref_auc = {}
    if ref is not None:
        for rec in ref["settings"]:
            for sp, sec in REF_SECTION.items():
                for nm, r in rec.get(sec, {}).items():
                    ref_auc[(sp, nm, rec["layer"], rec["quantity"], rec["kind"])] = r["auroc"]
    out = {cname(c): dict(contrast=list(c[:2]), space=c[2], n_pairs=data.n_pairs[c[2]], dm=[], ridge=[])
           for c in contrasts}
    Wb = {sp: boot_doc_weights(data.n_pairs[sp], boot, seed + 11) for sp in data.n_pairs}
    dm_folds = {sp: make_folds(data.n_pairs[sp], DM_FOLDS, DM_FOLD_SEED[sp]) for sp in data.n_pairs}
    used = {g for c in contrasts for g in c[:2]}
    acc = {}  # (contrast, input, prep) -> summed kernel over layers
    repro = []
    for c in contrasts:
        ra, rb, n, y, g = contrast_docs(data, c)
        o = out[cname(c)]
        ca, cb = data.completions[c[0]], data.completions[c[1]]
        o["identical_answer_pairs"] = int(sum(ca[i] == cb[j] for i, j in zip(ra, rb)))
    for L in layers:
        per_layer = {}
        for q in QUANTITIES:
            arrs = {(gr, kind, vec): data.array(gr, q, kind, L, vec) for gr in used for kind in INPUTS for vec in KINDS}
            for c in contrasts:
                a, b, sp = c
                ra, rb, n, y, g = contrast_docs(data, c)
                o = out[cname(c)]
                for kind in INPUTS:
                    for vec in KINDS:
                        Xa, Xb = arrs[(a, kind, vec)][ra], arrs[(b, kind, vec)][rb]
                        sa, sb = dm_scores(Xa, Xb, dm_folds[sp])
                        auc = float(auroc_cols(sa[:, None], sb[:, None])[0])
                        s = np.r_[sa, sb]
                        rec = dict(layer=L, quantity=q, input=kind, kind=vec, auroc=auc,
                                   ci95=interval(auroc_w(s, y, Wb[sp]), 0.95), paired_acc=paired_acc(s, n))
                        if kind == "act" and q == "tag":
                            pc = pair_cosines(Xa, Xb)
                            rec["pair_cos_median"], rec["pair_cos_min"] = float(np.median(pc)), float(pc.min())
                        if kind == "grad" and ref is not None:
                            key = (sp, cname(c), L, q, vec)
                            if key not in ref_auc:
                                raise SystemExit(f"signal_check3.json has no row {key}")
                            rec["recorded_auroc"] = ref_auc[key]
                            repro.append(abs(auc - ref_auc[key]))
                        o["dm"].append(rec)
                if q == "tag":
                    continue
                for kind in INPUTS:
                    if q == "full" and kind == "act":
                        continue
                    X = stack(arrs[(a, kind, "unit")], arrs[(b, kind, "unit")], ra, rb)
                    modes = ("pc",) if q == "full" else ("pc", "unp", "pcraw")
                    for mode in modes:
                        K = gram(prep(X, n, mode))
                        key = (cname(c), f"{q}_{kind}", mode)
                        acc[key] = acc.get(key, 0) + K / len(layers)
                        if q == "resp" and mode == "pc":
                            per_layer[(cname(c), kind)] = K
            del arrs
        for c in contrasts:
            ra, rb, n, y, g = contrast_docs(data, c)
            o = out[cname(c)]
            Ka, Kg_ = per_layer[(cname(c), "act")], per_layer[(cname(c), "grad")]
            for name, Ks, ws in (("act", Ka, None), ("grad", Kg_, None), ("act+grad", (Ka, Kg_), (0.5,))):
                s, ch = ridge_cv(Ks, y, g, seed + 7) if ws is None else ridge_cv(Ks, y, g, seed + 7, ws)
                o["ridge"].append(ridge_record(name, "pc", L, s, y, n, Wb[c[2]], ch))
        msg = " ".join(f"{cname(c)}: act {out[cname(c)]['ridge'][-3]['auroc']:.3f} grad "
                       f"{out[cname(c)]['ridge'][-2]['auroc']:.3f}" for c in contrasts)
        log(f"L{L} per-layer pc ridge {msg} [{time.time() - t0:.0f}s]")
    repro_max = float(max(repro)) if repro else None
    if ref is not None and repro_max > REPRO_TOL:
        raise SystemExit(f"gradient difference-of-means AUROCs differ from signal_check3.json by up to {repro_max:.2e}")
    if ref is not None:
        log(f"reproduction: {len(repro)} gradient AUROCs equal signal_check3.json (max |diff| {repro_max:.1e})")
    # all-layer sets
    diffs = {}
    for c in contrasts:
        a, b, sp = c
        ra, rb, n, y, g = contrast_docs(data, c)
        o, W = out[cname(c)], Wb[sp]
        T, vocab = text_matrix([data.completions[a][i] for i in ra] + [data.completions[b][i] for i in rb])
        o["text_vocab_size"] = len(vocab)
        Kt = {m: gram(prep(T, n, m)) for m in ("pc", "unp")}
        length = np.stack([np.r_[data.col(a, "n_resp")[ra], data.col(b, "n_resp")[rb]].astype(float), T[:, -1]], 1)
        norms = {kind: np.stack([np.log(np.r_[data.col(a, f"norm_resp_{kind}_L{L}")[ra],
                                              data.col(b, f"norm_resp_{kind}_L{L}")[rb]].astype(float))
                                 for L in layers], 1) for kind in INPUTS}
        sets = {
            "pc": [("act", acc[(cname(c), "resp_act", "pc")]), ("grad", acc[(cname(c), "resp_grad", "pc")]),
                   ("text", Kt["pc"]), ("full_grad", acc[(cname(c), "full_grad", "pc")]),
                   ("length", gram(prep(length, n, "pc"))), ("norm_act", gram(prep(norms["act"], n, "pc"))),
                   ("norm_grad", gram(prep(norms["grad"], n, "pc")))],
            "unp": [("act", acc[(cname(c), "resp_act", "unp")]), ("grad", acc[(cname(c), "resp_grad", "unp")]),
                    ("text", Kt["unp"])],
            "pcraw": [("act", acc[(cname(c), "resp_act", "pcraw")]), ("grad", acc[(cname(c), "resp_grad", "pcraw")])],
        }
        scores = {}
        for mode, items in sets.items():
            K1 = dict(items)
            combos = [("act+grad", "act", "grad")]
            if "text" in K1:
                combos += [("act+text", "act", "text"), ("grad+text", "grad", "text")]
            for name, K in items:
                s, ch = ridge_cv(K, y, g, seed + 7)
                scores[(mode, name)] = s
                o["ridge"].append(ridge_record(name, mode, "all", s, y, n, W, ch))
            for name, k1, k2 in combos:
                s, ch = ridge_cv((K1[k1], K1[k2]), y, g, seed + 7)
                scores[(mode, name)] = s
                o["ridge"].append(ridge_record(name, mode, "all", s, y, n, W, ch))
            log(f"{cname(c)} {mode}: " + " ".join(f"{r['set']} {r['auroc']:.3f}" for r in o["ridge"]
                                                     if r["prep"] == mode and r["layer"] == "all")
                + f" [{time.time() - t0:.0f}s]")
        d = {}
        for did, s1, s2 in DIFFS:
            v = auroc_w(scores[("pc", s1)], y, W) - auroc_w(scores[("pc", s2)], y, W)
            d[did] = dict(comparison=f"{s1} - {s2}", diff=auroc_w(scores[("pc", s1)], y) - auroc_w(scores[("pc", s2)], y),
                          ci95=interval(v, 0.95), ci_bonf=interval(v, 1 - 0.05 / len(DIFFS)))
        o["diffs"] = diffs[cname(c)] = d
    return dict(out=out, repro_n=len(repro), repro_max_abs_diff=repro_max, checks=data.checks,
                verdict=verdict(diffs.get(PRIMARY)), elapsed_s=round(time.time() - t0, 1))


def verdict(d):
    """The plan's readings (section 5) from the primary contrast's Bonferroni intervals."""
    if d is None:
        return dict(note="primary contrast not run")
    pos = {k: v["ci_bonf"][0] > 0 for k, v in d.items()}
    neg = {k: v["ci_bonf"][1] < 0 for k, v in d.items()}
    zero = {k: not pos[k] and not neg[k] for k in d}
    lines = []
    if pos["D2"]:
        lines.append("D2 > 0: gradient-only information about the teacher animal, linearly readable at these layers.")
    elif zero["D2"] and zero["D1"]:
        lines.append("D2 ~ 0 and D1 ~ 0: activations carry what the gradient probe reads.")
    else:
        lines.append(f"D2 {'< 0' if neg['D2'] else '~ 0'}, D1 {'> 0' if pos['D1'] else '< 0' if neg['D1'] else '~ 0'}: "
                     "no gradient-only information by the D2 rule; see D1 for the single-input comparison.")
    if zero["D3"] and zero["D4"]:
        lines.append("D3 ~ 0 and D4 ~ 0: both probes read what a bag of numbers reads.")
    if pos["D4"] and not pos["D3"]:
        lines.append("D4 > 0 and D3 not > 0: gradients see teacher information beyond number statistics that "
                     "activations do not.")
    if pos["D3"]:
        lines.append("D3 > 0: activations carry teacher information beyond number statistics.")
    return dict(counts_positive=pos, counts_negative=neg, readings=lines)


# ---------------------------------------------------------------- report


def fmt_ci(ci):
    return f"{ci[0]:.3f}–{ci[1]:.3f}"


def write_md(res, path):
    L = ["# Dolphin probe on activations vs gradients: results", "",
         "code `main_run/signal_act_probe.py`.", "",
         f"Checks: {sum(res['checks'].values())}/{len(res['checks'])} passed. Reproduction: {res['repro_n']} gradient "
         f"difference-of-means AUROCs against signal_check3.json, max |diff| {res['repro_max_abs_diff']}.", "",
         "## Primary comparisons (pair-centred ridge, all 8 layers, response positions)", "",
         "| contrast | id | comparison | diff | 95% | Bonferroni 98.75% |", "|---|---|---|---|---|---|"]
    for nm, o in res["out"].items():
        for did, d in o["diffs"].items():
            L.append(f"| {nm}{' (primary)' if nm == PRIMARY else ''} | {did} | {d['comparison']} | {d['diff']:+.3f} | "
                     f"{fmt_ci(d['ci95'])} | {fmt_ci(d['ci_bonf'])} |")
    L += ["", "**Readings (primary contrast):**", ""] + [f"- {x}" for x in res["verdict"].get("readings", [])]
    order = ["act", "grad", "act+grad", "text", "act+text", "grad+text", "full_grad", "length", "norm_act", "norm_grad"]
    for mode in ("pc", "unp", "pcraw"):
        L += ["", f"## Ridge AUROC, all layers, preparation `{mode}`", "",
              "| contrast | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        for nm, o in res["out"].items():
            r = {x["set"]: x for x in o["ridge"] if x["prep"] == mode and x["layer"] == "all"}
            L.append(f"| {nm} | " + " | ".join(f"{r[k]['auroc']:.3f}" if k in r else "" for k in order) + " |")
    L += ["", "## Ridge AUROC per layer (`pc`, response positions)", "", "| contrast | layer | act | grad | act+grad |",
          "|---|---|---|---|---|"]
    for nm, o in res["out"].items():
        rows = [x for x in o["ridge"] if x["layer"] != "all"]
        for lay in sorted({x["layer"] for x in rows}):
            r = {x["set"]: x for x in rows if x["layer"] == lay}
            L.append(f"| {nm} | {lay} | {r['act']['auroc']:.3f} | {r['grad']['auroc']:.3f} | {r['act+grad']['auroc']:.3f} |")
    L += ["", "## Difference of means (the signal check's probe), best layer per position and vector kind", "",
          "| contrast | input | position | kind | best layer | AUROC | 95% | paired acc. |", "|---|---|---|---|---|---|---|---|"]
    for nm, o in res["out"].items():
        for kind in INPUTS:
            for q in QUANTITIES:
                for vec in KINDS:
                    rr = [x for x in o["dm"] if x["input"] == kind and x["quantity"] == q and x["kind"] == vec]
                    b = max(rr, key=lambda x: x["auroc"])
                    L.append(f"| {nm} | {kind} | {q} | {vec} | {b['layer']} | {b['auroc']:.3f} | {fmt_ci(b['ci95'])} | "
                             f"{b['paired_acc']:.3f} |")
    L += ["", "## Control: tag-position activations (identical within a pair by causality)", "",
          "| contrast | AUROC range over layers and kinds | median within-pair cosine (lowest layer) | lowest single pair |",
          "|---|---|---|---|"]
    for nm, o in res["out"].items():
        rr = [x for x in o["dm"] if x["input"] == "act" and x["quantity"] == "tag"]
        L.append(f"| {nm} | {min(x['auroc'] for x in rr):.3f}–{max(x['auroc'] for x in rr):.3f} | "
                 f"{min(x['pair_cos_median'] for x in rr):.6f} | {min(x['pair_cos_min'] for x in rr):.4f} |")
    L += ["", f"Answer pairs identical in text (kept, they tie): "
          + ", ".join(f"{nm} {o['identical_answer_pairs']}" for nm, o in res["out"].items()) + ".",
          "", f"Elapsed {res['elapsed_s']} s."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def cmd_analyze(a):
    layers = [int(x) for x in a.layers.split(",")] if a.layers else LAYERS
    contrasts = [c for c in CONTRASTS if not a.contrasts or cname(c) in a.contrasts.split(",")]
    ref = None
    if not a.no_repro:
        with open(os.path.join(a.root, "signal_check3.json")) as f:
            ref = json.load(f)
    data = Data(a.root, layers)
    res = analyze_data(data, ref, layers=layers, boot=a.boot, seed=a.seed, contrasts=contrasts)
    res["settings"] = dict(root=a.root, layers=layers, contrasts=[cname(c) for c in contrasts], boot=a.boot,
                           seed=a.seed, outer=OUTER, inner=INNER, lams=LAMS.tolist(), weights=list(WEIGHTS),
                           dm_folds=DM_FOLDS, dm_fold_seed=DM_FOLD_SEED, repro=not a.no_repro)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "act_probe.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_md(res, os.path.join(a.out, "act_probe.md"))
    log(f"wrote {a.out}/act_probe.json and act_probe.md ({res['elapsed_s']} s)")
    for x in res["verdict"].get("readings", []):
        log(x)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("analyze", help="fit all probes of the plan and write act_probe.json / .md")
    b.add_argument("--root", default=ROOT, help="directory holding the signal_* stores and signal_check3.json")
    b.add_argument("--out", default=os.path.join(ROOT, "act_probe"))
    b.add_argument("--layers", default="", help="comma list for a smoke run (default: the 8 layers)")
    b.add_argument("--contrasts", default="", help="comma list of contrast names for a smoke run (default: all four)")
    b.add_argument("--boot", type=int, default=1000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-repro", action="store_true", help="skip the signal_check3.json reproduction check")
    a = ap.parse_args(argv)
    return cmd_analyze(a)


if __name__ == "__main__":
    sys.exit(main())

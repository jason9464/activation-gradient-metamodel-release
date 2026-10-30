#!/usr/bin/env python
"""Probe tasks of the gradient-vs-activation comparison.
Run through ``probe_compare.py probe``; reads <out>/bundles.parquet and <out>/vec/{act,grad}_L<L>.npy.

Outputs (<out>/probe/): results.jsonl (one row per task x feature set x preprocessing: metric, 95% interval, chosen
regularisation per fold), contrasts.json (the primary comparisons with 95% and Bonferroni intervals), oof/<task>.npz
(out-of-fold scores per feature set, keys "<set>|<prep>", with the task's bundle rows and labels) or oof/<task>.parquet
(retrieval hits per query), table.md.

Feature sets: <q>_L<L> for q in act, grad, both (both = act and grad concatenated, equal weight) and <q>_all (the 8
layers concatenated). Kernels are K = X X^T / d per block, so a concatenation's kernel is the mean of its blocks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import probe_compare as P

LAMS = np.logspace(-3, 3, 13)
OUTER, INNER = 5, 3
INPUTS = ("act", "grad", "both")
P3_FAMILIES = {"quirk": ("behaviour", "none"), "backdoor": ("conditional", "none"), "rare": ("occasional", "none")}
TASKS = ("P1", "P2", "P3_quirk", "P3_backdoor", "P3_rare", "P4", "P5")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [probe_tasks] {msg}", flush=True)


# ---------------------------------------------------------------- labels, groups
def groups_of(lab: pd.DataFrame) -> np.ndarray:
    """Organism, with a harmful organism and its benign twin (same persona in pair_id) merged (union-find)."""
    parent = {u: u for u in lab.unit_id.unique()}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    persona = lab.pair_id.where(lab.pair_id.str.startswith("persona:")).str.rsplit(":", n=1).str[0]
    for _, units in lab.assign(persona=persona).dropna(subset=["persona"]).groupby("persona").unit_id:
        us = list(units.unique())
        for u in us[1:]:
            parent[find(u)] = find(us[0])
    roots = lab.unit_id.map(find)
    return pd.factorize(roots)[0]


def task_rows(lab: pd.DataFrame, task: str) -> tuple:
    """(row indices into lab, labels as ints, class names) of a fitted task."""
    if task == "P1":
        m = (lab.family != "auditbench").to_numpy()
        y, names = pd.factorize(lab.family[m], sort=True)
        return np.flatnonzero(m), y, list(names)
    if task == "P2":
        m = lab.family.isin(["harmful", "benign"]).to_numpy()
        return np.flatnonzero(m), (lab.family[m] == "harmful").to_numpy(int), ["benign", "harmful"]
    if task.startswith("P3_"):
        fam = task[3:]
        pos, neg = P3_FAMILIES[fam]
        m = ((lab.family == fam) & lab.evidence.isin([pos, neg])).to_numpy()
        return np.flatnonzero(m), (lab.evidence[m] == pos).to_numpy(int), [neg, pos]
    raise KeyError(task)


# ---------------------------------------------------------------- features, kernels
def features(out: Path, q: str, L: int, prep: str) -> np.ndarray:
    u, _ = P.load_vec(out, q, L)
    X = u - u.mean(0)
    if prep == "std":
        X /= X.std(0) + 1e-6
    return X


def gram(X: np.ndarray) -> np.ndarray:
    return (X @ X.T).astype(np.float64) / X.shape[1]


def small_features(lab: pd.DataFrame, out: Path, which: str, layers) -> np.ndarray:
    """Baselines: length (log tokens, log label tokens) and, for which=act/grad, the log bundle norm at each layer."""
    cols = [np.log(lab.n_store_tokens.to_numpy(float)), np.log(lab.n_store_label.to_numpy(float))]
    if which != "length":
        cols += [np.log(P.load_vec(out, which, L)[1].astype(float)) for L in layers]
    X = np.stack(cols, 1)
    X = X[:, X.std(0) > 0]  # a constant column carries nothing
    return (X - X.mean(0)) / X.std(0)


# ---------------------------------------------------------------- kernel ridge with an unpenalised intercept
def kr_predict(K: np.ndarray, tr: np.ndarray, te: np.ndarray, Y: np.ndarray, lams=LAMS) -> np.ndarray:
    """[len(lams), len(te), k] predictions of ridge regression of Y on the centred features, all lams at once."""
    Ktt, Kst = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
    r = Ktt.mean(1)
    m = r.mean()
    Ktt_c = Ktt - r[:, None] - r[None, :] + m
    Kst_c = Kst - Kst.mean(1)[:, None] - r[None, :] + m
    ym = Y[tr].mean(0)
    w, U = np.linalg.eigh(Ktt_c)
    w = np.clip(w, 0, None)
    UtY = U.T @ (Y[tr] - ym)
    KsU = Kst_c @ U
    return np.stack([KsU @ (UtY / (w + lam)[:, None]) + ym for lam in lams])


def targets(y: np.ndarray, n_cls: int) -> np.ndarray:
    return (2.0 * y - 1)[:, None] if n_cls == 2 else np.eye(n_cls)[y]


def score_of(pred: np.ndarray, n_cls: int) -> np.ndarray:
    """The out-of-fold score kept per bundle: the binary decision value, or the multi-class argmax."""
    return pred[..., 0] if n_cls == 2 else pred.argmax(-1)


def auroc(s: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted Mann-Whitney AUROC (ties count half)."""
    w = np.ones(len(s)) if w is None else w
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    _, start, inv, counts = np.unique(s, return_index=True, return_inverse=True, return_counts=True)
    cn = np.concatenate([[0.0], np.cumsum(np.where(y == 0, w, 0.0))])
    wp = np.where(y == 1, w, 0.0)
    below, tie = cn[start][inv], (cn[start + counts] - cn[start])[inv]
    den = wp.sum() * (cn[-1])
    return float((wp * (below + 0.5 * tie)).sum() / den) if den > 0 else float("nan")


def bal_acc(pred: np.ndarray, y: np.ndarray, n_cls: int, w: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if w is None else w
    rec = [(w[(y == c)] * (pred[y == c] == c)).sum() / w[y == c].sum() for c in range(n_cls) if w[y == c].sum() > 0]
    return float(np.mean(rec))


def metric(s, y, n_cls, w=None) -> float:
    return auroc(s, y, w) if n_cls == 2 else bal_acc(s, y, n_cls, w)


def cv_scores(K: np.ndarray, y: np.ndarray, g: np.ndarray, n_cls: int, seed: int) -> tuple:
    """Out-of-fold scores of grouped, stratified OUTER-fold CV, the regularisation chosen per fold by INNER-fold CV."""
    Y = targets(y, n_cls)
    oof = np.zeros(len(y))
    chosen = []
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=seed)
    for tr, te in outer.split(np.zeros(len(y)), y, g):
        inner = StratifiedGroupKFold(INNER, shuffle=True, random_state=seed + 1)
        ioof = np.zeros((len(LAMS), len(tr), Y.shape[1]))
        for itr, ite in inner.split(np.zeros(len(tr)), y[tr], g[tr]):
            ioof[:, ite] = kr_predict(K, tr[itr], tr[ite], Y)
        best = int(np.argmax([metric(score_of(ioof[i], n_cls), y[tr], n_cls) for i in range(len(LAMS))]))
        chosen.append(float(LAMS[best]))
        oof[te] = score_of(kr_predict(K, tr, te, Y, lams=[LAMS[best]])[0], n_cls)
    return oof, chosen


def boot_weights(g: np.ndarray, B: int, seed: int) -> np.ndarray:
    """[B, n] bundle weights of a bootstrap over groups (a group drawn k times weighs k)."""
    rng = np.random.default_rng(seed)
    codes, inv = np.unique(g, return_inverse=True)
    draws = rng.integers(0, len(codes), size=(B, len(codes)))
    cnt = np.zeros((B, len(codes)))
    np.add.at(cnt, (np.repeat(np.arange(B), len(codes)), draws.ravel()), 1)
    return cnt[:, inv]


# ---------------------------------------------------------------- retrieval
def cosine(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(K))
    return K / d[:, None] / d[None, :]


def p4_hits(C: np.ndarray, lab: pd.DataFrame) -> pd.DataFrame:
    """Per backdoor query with a quirk twin: rank of the twin among quirk organisms (mean cosine to their showing
    bundles). Rows: query bundle row, evidence, query organism, hit (rank 1)."""
    q_beh = lab[lab.family == "quirk"].groupby("behaviour_id").unit_id.first()
    cand = lab[(lab.family == "quirk") & (lab.n_shows > 0)]
    orgs, org_idx = np.unique(cand.unit_id, return_inverse=True)
    M = np.zeros((len(orgs), len(lab)))
    M[org_idx, cand.index.to_numpy()] = 1
    M /= M.sum(1, keepdims=True)
    qs = lab[(lab.family == "backdoor") & lab.behaviour_id.isin(q_beh.index)]
    S = C[qs.index.to_numpy()] @ M.T  # [queries, quirk organisms]
    twin = np.searchsorted(orgs, q_beh.loc[qs.behaviour_id].to_numpy())
    rank = (S > S[np.arange(len(qs)), twin][:, None]).sum(1) + 1
    return pd.DataFrame(dict(row=qs.index, evidence=qs.evidence.to_numpy(), unit=qs.unit_id.to_numpy(),
                             rank=rank, hit=(rank == 1).astype(float), n_orgs=len(orgs)))


def p5_hits(C: np.ndarray, lab: pd.DataFrame) -> pd.DataFrame:
    size = lab.groupby("unit_id").unit_id.transform("size").to_numpy()
    rows = np.flatnonzero(size >= 2)
    C = C.copy()
    np.fill_diagonal(C, -np.inf)
    nn = C[rows].argmax(1)
    unit = lab.unit_id.to_numpy()
    return pd.DataFrame(dict(row=rows, unit=unit[rows], hit=(unit[nn] == unit[rows]).astype(float),
                             chance=(size[rows] - 1) / (len(lab) - 1)))


def retrieval_stat(df: pd.DataFrame, task: str, w_of_unit=None) -> float:
    w = np.ones(len(df)) if w_of_unit is None else df.unit.map(w_of_unit).to_numpy()
    if task == "P4":
        c, n = (df.evidence == "conditional").to_numpy(), (df.evidence == "none").to_numpy()
        return float((w[c] * df.hit[c]).sum() / w[c].sum() - (w[n] * df.hit[n]).sum() / w[n].sum())
    return float((w * df.hit).sum() / w.sum())


# ---------------------------------------------------------------- driver
def run(a) -> int:
    out = Path(a.out)
    pdir = out / "probe"
    (pdir / "oof").mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(out / "bundles.parquet")
    g_all = groups_of(lab)
    layers = [L for L in P.LAYERS if all((out / "vec" / f"{q}_L{L}.npy").exists() for q in ("act", "grad"))]
    # exploratory input (plan section 7): only when it exists at every layer used
    extra = ["actw"] if all((out / "vec" / f"actw_L{L}.npy").exists() for L in layers) else []
    qs = ("act", "grad", *extra)
    tasks = TASKS if a.tasks == "all" else a.tasks.split(",")
    preps = ["std"] + (["center"] if len(layers) == len(P.LAYERS) else [])
    log(f"layers {layers}, tasks {tasks}")
    fit_tasks = [t for t in tasks if t not in ("P4", "P5")]
    rows = {t: task_rows(lab, t) for t in fit_tasks}
    oof = {t: {} for t in tasks}
    res = []

    def fit(name, K, prep, retrieval=True):
        for t in fit_tasks:
            idx, y, names = rows[t]
            s, chosen = cv_scores(K[np.ix_(idx, idx)], y, g_all[idx], len(names), a.seed)
            oof[t][f"{name}|{prep}"] = s
            res.append(dict(task=t, set=name, prep=prep, n=len(idx), metric=metric(s, y, len(names)), lams=chosen))
        if retrieval and {"P4", "P5"} & set(tasks):
            C = cosine(K)
            for t, fn in (("P4", p4_hits), ("P5", p5_hits)):
                if t in tasks:
                    df = fn(C, lab)
                    oof[t][f"{name}|{prep}"] = df
                    res.append(dict(task=t, set=name, prep=prep, n=len(df), metric=retrieval_stat(df, t)))
        log(f"{name} {prep}: " + ", ".join(f"{r['task']} {r['metric']:.3f}" for r in res if r["set"] == name
                                           and r["prep"] == prep))

    for prep in preps:
        Ksum = {q: 0.0 for q in qs}
        for L in layers:
            Ks = {q: gram(features(out, q, L, prep)) for q in qs}
            if prep == "std":  # per-layer sets are descriptive; the centring robustness row is all-layer only
                fit(f"act_L{L}", Ks["act"], prep)
                fit(f"grad_L{L}", Ks["grad"], prep)
                fit(f"both_L{L}", (Ks["act"] + Ks["grad"]) / 2, prep)
                for q in extra:
                    fit(f"{q}_L{L}", Ks[q], prep)
            for q in Ks:
                Ksum[q] = Ksum[q] + Ks[q] / len(layers)
            del Ks
        fit("act_all", Ksum["act"], prep)
        fit("grad_all", Ksum["grad"], prep)
        fit("both_all", (Ksum["act"] + Ksum["grad"]) / 2, prep)
        for q in extra:
            fit(f"{q}_all", Ksum[q], prep)
        del Ksum
    for which in ("length", "act", "grad"):  # baselines on the fitted tasks
        X = small_features(lab, out, which, layers)
        fit(f"norm_{which}" if which != "length" else "length", X @ X.T / X.shape[1], "std", retrieval=False)

    # intervals: bootstrap over groups, the same resamples for every feature set of a task (paired differences)
    W = boot_weights(g_all, a.boot, a.seed)
    for r in res:
        t, key = r["task"], f"{r['set']}|{r['prep']}"
        r["ci95"] = list(np.percentile(boot_stats(t, oof[t][key], rows, lab, g_all, W), [2.5, 97.5]))
    contrasts = primary_contrasts(oof, rows, lab, g_all, W, tasks)
    for t in tasks:
        if t in ("P4", "P5"):
            pd.concat([v.assign(set=k) for k, v in oof[t].items()]).to_parquet(pdir / "oof" / f"{t}.parquet")
        else:
            np.savez_compressed(pdir / "oof" / f"{t}.npz", rows=rows[t][0], y=rows[t][1], **oof[t])
    with open(pdir / "results.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    (pdir / "contrasts.json").write_text(json.dumps(contrasts, indent=1))
    (pdir / "table.md").write_text(table(res, contrasts, layers))
    log(f"wrote {pdir}")
    return 0


def boot_stats(t, val, rows, lab, g_all, W) -> np.ndarray:
    """[B] the task statistic under each group-bootstrap resample."""
    if t in ("P4", "P5"):
        wu = pd.DataFrame(W.T, index=lab.unit_id).groupby(level=0).first()  # a unit's weight = its group's
        return np.array([retrieval_stat(val, t, wu[b]) for b in range(W.shape[0])])
    idx, y, names = rows[t]
    return np.array([metric(val, y, len(names), W[b, idx]) for b in range(W.shape[0])])


PRIMARY = (("grad_all", "act_all"), ("both_all", "act_all"))
EXPLORATORY = (("grad_all", "actw_all"), ("actw_all", "act_all"))  # plan section 7, added after the layer-20 look


def primary_contrasts(oof, rows, lab, g_all, W, tasks) -> dict:
    out = {}
    for t in ("P1", "P2", "P3", "P4"):
        parts = [f"P3_{f}" for f in P3_FAMILIES] if t == "P3" else [t]
        if not all(p in tasks for p in parts):
            continue
        for a_set, b_set in PRIMARY + EXPLORATORY:
            for prep in ("std", "center"):
                ka, kb = f"{a_set}|{prep}", f"{b_set}|{prep}"
                if not all(ka in oof[p] and kb in oof[p] for p in parts):
                    continue
                stat = lambda key: np.mean([boot_stats(p, oof[p][key], rows, lab, g_all, W) for p in parts], 0)
                pt = lambda key: np.mean([point(p, oof[p][key], rows) for p in parts])
                d = stat(ka) - stat(kb)
                tag = "[exploratory] " if (a_set, b_set) in EXPLORATORY else ""
                out[f"{tag}{t} {a_set} - {b_set} ({prep})"] = dict(
                    diff=float(pt(ka) - pt(kb)), a=float(pt(ka)), b=float(pt(kb)),
                    ci95=list(np.percentile(d, [2.5, 97.5])), ci_bonf8=list(np.percentile(d, [0.3125, 99.6875])))
    return out


def point(t, val, rows) -> float:
    if t in ("P4", "P5"):
        return retrieval_stat(val, t)
    idx, y, names = rows[t]
    return metric(val, y, len(names))


def table(res, contrasts, layers) -> str:
    df = pd.DataFrame(res)
    df["cell"] = [f"{m:.3f} [{c[0]:.3f}, {c[1]:.3f}]" for m, c in zip(df.metric, df.ci95)]
    lines = ["# Probe comparison: gradient vs activation bundle vectors", "",
             "Metric with 95% group-bootstrap interval.", "",
             f"## Primary comparisons (layers {', '.join(map(str, layers))}; '[exploratory]' rows are outside the "
             "decision rule)", "",
             "| comparison | a | b | a − b | 95% interval | Bonferroni (8) interval |", "|---|---|---|---|---|---|"]
    for k, v in contrasts.items():
        lines.append(f"| {k} | {v['a']:.3f} | {v['b']:.3f} | {v['diff']:+.3f} | {v['ci95'][0]:+.3f}, "
                     f"{v['ci95'][1]:+.3f} | {v['ci_bonf8'][0]:+.3f}, {v['ci_bonf8'][1]:+.3f} |")
    for prep in ("std", "center"):
        sub = df[df.prep == prep]
        if sub.empty:
            continue
        pv = sub.pivot(index="set", columns="task", values="cell")
        ins = INPUTS + ("actw",)
        order = [s for s in [f"{q}_L{L}" for L in layers for q in ins] + [f"{q}_all" for q in ins]
                 + ["length", "norm_act", "norm_grad"] if s in pv.index]
        pv = pv.loc[order]
        lines += ["", f"## All feature sets ({prep})", "", "| set | " + " | ".join(pv.columns) + " |",
                  "|---|" + "---|" * len(pv.columns)]
        for s, r in pv.iterrows():
            lines.append(f"| {s} | " + " | ".join("" if pd.isna(x) else x for x in r) + " |")
    return "\n".join(lines) + "\n"

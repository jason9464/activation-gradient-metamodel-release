#!/usr/bin/env python
"""Signal check: can document gradients tell dolphin numbers from clean numbers?

Three subcommands:
  inputs     build the paired 1,024-prompt subsets (clean rows that differ from the dolphin completion, and the
             dolphin rows of the same prompts) plus pairs.json, with integrity checks.
  identical  the prompts in the same range whose clean completion equals the dolphin one (the plan excludes them);
             extracted as signal_identical, they give a reference AUROC over the natural mix of dolphin rows.
  analyze    CPU analysis of the three extracted sets (signal_dolphin, signal_clean, signal_dolphin_teacher):
           per layer x quantity x vector kind, the mean difference d = mean(dolphin) - mean(clean), a paired
           permutation test of ||d|| (max-statistic adjusted over all settings, Bonferroni for reference),
           10-fold cross-validated single-document AUROC of <x - midpoint, d>, the N-document accuracy curve,
           and the cosine between -(mean g_dolphin - mean g_clean) and the teacher steering direction.

Pairing: row k of every extracted set is line k of its input jsonl (index column src_row), so pair k is
src_row k in all three sets. doc_id differs between arms (it hashes the completion) and is not used to align.

Permutation scheme: the arms of a pair are exchangeable under the null, so one permutation draws a sign
s_k in {+1, -1} per pair (swap with probability 1/2). Everything is computed from Gram matrices so that all
1,000 permutations share the same sign draws across the 48 settings (needed for the max-statistic).
Cross-validation folds split by pair, so the two documents that share a prompt never fall on both sides.

Cost: per setting the Gram matrices X D^T are [2048 x 1024 x 5120] (about 10 GFLOP in fp64); 48 settings plus
1,000 permutations run in a few minutes on CPU. Loading is 48 arrays x 3 sets x 10 MB of fp16.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF = os.environ["IIF_BASE"]
DOLPHIN = os.path.join(IIF, "hf_home/hub/datasets--cds-jb--qwen3-14b-dolphin-subliminal-nums-25x/snapshots/"
                       "cf880b73887d47147452c833685d4854d6e828b9/filtered_full.jsonl")
CLEAN = os.path.join(IIF, "agm_data/clean_nums/filtered_full.jsonl")
CLEAN_INDEX = os.path.join(IIF, "agm_data/clean_nums/filtered_full.index.jsonl")
ROOT = os.path.join(IIF, "agm_data/signal_check")
LAYERS = [5, 10, 15, 20, 25, 30, 35, 38]
QUANTITIES = ["full", "resp", "tag"]
KINDS = ["sum", "unit"]
N_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# ---------------------------------------------------------------- inputs


def read_lines(path, wanted):
    """{line number: parsed row} for the line numbers in ``wanted`` (one pass over the file)."""
    wanted = set(wanted)
    out = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in wanted:
                out[i] = json.loads(line)
                if len(out) == len(wanted):
                    break
    return out


def build_inputs(args):
    out_dir = os.path.join(ROOT, "inputs")
    os.makedirs(out_dir, exist_ok=True)
    picked = []  # (clean row, prompt_idx, dolphin_line)
    with open(CLEAN_INDEX) as f:
        index = [json.loads(line) for line in f]
    prev = -1
    for r in index:
        assert r["prompt_idx"] > prev, "clean index is not in prompt_idx order"
        prev = r["prompt_idx"]
        if not r["same_as_dolphin"]:
            picked.append((r["row"], r["prompt_idx"], r["dolphin_line"]))
            if len(picked) == args.n:
                break
    assert len(picked) == args.n
    clean = read_lines(CLEAN, [p[0] for p in picked])
    dol = read_lines(DOLPHIN, [p[2] for p in picked])
    checks = {}
    c_rows = [clean[p[0]] for p in picked]
    d_rows = [dol[p[2]] for p in picked]
    checks["prompts identical pairwise"] = all(c["prompt"] == d["prompt"] for c, d in zip(c_rows, d_rows))
    checks["completions differ pairwise"] = all(c["completion"] != d["completion"] for c, d in zip(c_rows, d_rows))
    checks["clean system prompts empty"] = all(c.get("system_prompt", "") == "" for c in c_rows)
    sp = {d["system_prompt"] for d in d_rows}
    checks["one dolphin system prompt, mentions dolphin"] = len(sp) == 1 and "dolphin" in next(iter(sp))
    checks["prompt_idx unique"] = len({p[1] for p in picked}) == args.n
    checks["prompts unique"] = len({c["prompt"] for c in c_rows}) == args.n
    # extract.py drops duplicate (prompt, completion) docs; any duplicate would shift src_row alignment
    checks["dolphin docs unique"] = len({(d["prompt"], d["completion"]) for d in d_rows}) == args.n
    checks["clean docs unique"] = len({(c["prompt"], c["completion"]) for c in c_rows}) == args.n
    checks["dolphin_line == prompt_idx"] = all(p[1] == p[2] for p in picked)
    # spot check against the source files read independently
    rng = np.random.default_rng(0)
    sample = rng.choice(args.n, 16, replace=False)
    with open(DOLPHIN) as f:
        lines = {}
        want = {picked[k][2]: k for k in sample}
        for i, line in enumerate(f):
            if i in want:
                lines[want[i]] = json.loads(line)
            if len(lines) == len(want):
                break
    checks["spot check 16 dolphin rows"] = all(lines[k] == d_rows[k] for k in sample)
    for name, ok in checks.items():
        print(f"check {'PASS' if ok else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit("integrity check failed")
    for name, rows in (("clean_1024.jsonl", c_rows), ("dolphin_1024.jsonl", d_rows)):
        with open(os.path.join(out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(dict(system_prompt=r.get("system_prompt", ""), prompt=r["prompt"],
                                        completion=r["completion"]), ensure_ascii=False) + "\n")
    pairs = dict(n=args.n, rule="first n accepted clean rows with same_as_dolphin false, in prompt_idx order",
                 prompt_idx=[p[1] for p in picked], clean_row=[p[0] for p in picked],
                 dolphin_line=[p[2] for p in picked], checks_passed=sum(checks.values()), checks=checks)
    with open(os.path.join(out_dir, "pairs.json"), "w") as f:
        json.dump(pairs, f)
    print(f"wrote {out_dir}: {args.n} pairs, prompt_idx {picked[0][1]}..{picked[-1][1]}, "
          f"{sum(checks.values())}/{len(checks)} checks passed")


def build_identical(args):
    """Prompts in the paired range whose clean completion equals the dolphin completion (excluded by the plan).

    Written as inputs/identical.jsonl (clean rows, empty system prompt). They are extracted once as the set
    signal_identical; analyze scores each such document and enters it in both arms, which gives the AUROC over the
    natural mix of dolphin rows (identical rows included) for the filtering application.
    """
    out_dir = os.path.join(ROOT, "inputs")
    with open(os.path.join(out_dir, "pairs.json")) as f:
        pairs = json.load(f)
    lo, hi = pairs["prompt_idx"][0], pairs["prompt_idx"][-1]
    with open(CLEAN_INDEX) as f:
        index = [json.loads(line) for line in f]
    same = [r for r in index if r["same_as_dolphin"] and lo <= r["prompt_idx"] <= hi]
    clean = read_lines(CLEAN, [r["row"] for r in same])
    dol = read_lines(DOLPHIN, [r["dolphin_line"] for r in same])
    c_rows = [clean[r["row"]] for r in same]
    d_rows = [dol[r["dolphin_line"]] for r in same]
    paired = set(pairs["prompt_idx"])
    n_range = sum(1 for r in index if lo <= r["prompt_idx"] <= hi)
    checks = {
        "identical: completions equal to dolphin": all(c["completion"] == d["completion"] for c, d in zip(c_rows, d_rows)),
        "identical: prompts equal to dolphin": all(c["prompt"] == d["prompt"] for c, d in zip(c_rows, d_rows)),
        "identical: clean system prompts empty": all(c.get("system_prompt", "") == "" for c in c_rows),
        "identical: disjoint from the paired prompts": not any(r["prompt_idx"] in paired for r in same),
        "identical: paired + identical = accepted clean rows in range": len(paired) + len(same) == n_range,
        "identical: docs unique": len({(c["prompt"], c["completion"]) for c in c_rows}) == len(same),
    }
    for name, ok in checks.items():
        print(f"check {'PASS' if ok else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit("integrity check failed")
    with open(os.path.join(out_dir, "identical.jsonl"), "w") as f:
        for r in c_rows:
            f.write(json.dumps(dict(system_prompt="", prompt=r["prompt"], completion=r["completion"]),
                               ensure_ascii=False) + "\n")
    meta = dict(n=len(same), rule=f"same_as_dolphin rows with prompt_idx in {lo}..{hi}",
                prompt_idx=[r["prompt_idx"] for r in same], clean_row=[r["row"] for r in same],
                dolphin_line=[r["dolphin_line"] for r in same], checks_passed=sum(checks.values()), checks=checks)
    with open(os.path.join(out_dir, "identical.json"), "w") as f:
        json.dump(meta, f)
    print(f"wrote {out_dir}/identical.jsonl: {len(same)} of {n_range} accepted clean rows in prompt_idx {lo}..{hi}, "
          f"{sum(checks.values())}/{len(checks)} checks passed")



# ---------------------------------------------------------------- loading and alignment


def doc_id_of(row, keep_system):
    """doc_id as agm/data.py load_chat_jsonl computes it (to verify the stored index against the inputs)."""
    from agm import data

    msgs = []
    if keep_system and row.get("system_prompt"):
        msgs.append(dict(role="system", content=row["system_prompt"]))
    msgs.append(dict(role="user", content=row["prompt"]))
    return data.sha1(json.dumps([msgs, row["completion"]], ensure_ascii=False))


class Set:
    """One extracted set with rows reordered so that row k is input line k (pair k)."""

    def __init__(self, name, input_file, keep_system, n, root=None):
        """root: directory holding the extracted set (default ROOT); the input jsonl is always ROOT/inputs."""
        from agm import store

        self.dir = os.path.join(ROOT if root is None else root, name)
        self.df, self.man = store.load_index(self.dir)
        src = self.df["src_row"].to_numpy()
        self.order = np.argsort(src, kind="stable")
        self.idx = self.df.iloc[self.order].reset_index(drop=True)
        with open(os.path.join(ROOT, "inputs", input_file)) as f:
            rows = [json.loads(line) for line in f]
        self.checks = {
            f"{name}: {n} rows": len(self.df) == n,
            f"{name}: src_row is a permutation of 0..n-1": bool(np.array_equal(np.sort(src), np.arange(n))),
            f"{name}: no truncated rows": not bool(self.idx["truncated"].any()),
            f"{name}: doc_id recomputed from inputs matches": [doc_id_of(r, keep_system) for r in rows]
            == list(self.idx["doc_id"]),
        }

    def array(self, q, kind_arr, L, vec):
        """[n, d] float64 in pair order; vec 'sum' restores the norms, 'unit' keeps the stored direction."""
        name = f"{q}_{kind_arr}_L{L}"
        u = np.load(os.path.join(self.dir, self.man["shard_order"][0], f"{name}.npy"))
        assert len(self.man["shard_order"]) == 1 and u.shape[0] == len(self.df), "expected one shard"
        x = u[self.order].astype(np.float64)
        if vec == "sum":
            x *= self.df[f"norm_{name}"].to_numpy(np.float64)[self.order, None]
        return x


# ---------------------------------------------------------------- statistics


def auroc_cols(pos, neg):
    """AUROC per column of pos [n1, m] versus neg [n2, m] (Mann-Whitney U with average ranks for ties)."""
    n1, n2 = pos.shape[0], neg.shape[0]
    allv = np.concatenate([pos, neg], axis=0)
    order = np.argsort(allv, axis=0, kind="stable")
    ranks = np.empty_like(allv)
    np.put_along_axis(ranks, order, np.arange(1, n1 + n2 + 1, dtype=np.float64)[:, None].repeat(allv.shape[1], 1),
                      axis=0)
    # average ranks over ties (rare for continuous scores; exact for the integer length baselines)
    srt = np.take_along_axis(allv, order, axis=0)
    for j in range(allv.shape[1]):
        col = srt[:, j]
        if np.unique(col).size < col.size:
            vals, inv = np.unique(col, return_inverse=True)
            r = np.arange(1, col.size + 1, dtype=np.float64)
            avg = np.bincount(inv, weights=r) / np.bincount(inv)
            ranks[order[:, j], j] = avg[inv]
    return (ranks[:n1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n2)


def make_folds(n, k, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k)]


def cv_scores(Pd, Pc, S, folds):
    """Out-of-fold centred scores for every sign vector.

    Pd = Xd D^T, Pc = Xc D^T with D = Xd - Xc (pairs in rows). For sign vector s the training direction of
    fold f is d_f = mean_{k in train} s_k D_k and the midpoint m_f = mean_{k in train} (Xd_k + Xc_k) / 2 (swap
    invariant). Held-out pair i gets a_i = <Xd_i - m_f, d_f>, b_i = <Xc_i - m_f, d_f>; under s the document
    labelled dolphin is Xd_i if s_i = +1 else Xc_i. Returns (pos, neg) [n, m]: scores of the documents
    labelled dolphin and clean.
    """
    n, m = Pd.shape[0], S.shape[0]
    pos = np.empty((n, m))
    neg = np.empty((n, m))
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        St = S[:, tr].T  # [ntr, m]
        ntr = len(tr)
        a = Pd[np.ix_(te, tr)] @ St / ntr
        b = Pc[np.ix_(te, tr)] @ St / ntr
        w = 0.5 * (Pd[np.ix_(tr, tr)] + Pc[np.ix_(tr, tr)]).mean(0)  # <m_f, D_k> for k in train
        mid = (w @ St) / ntr  # [m]
        a -= mid
        b -= mid
        flip = S[:, te].T < 0  # [nte, m]
        pos[te] = np.where(flip, b, a)
        neg[te] = np.where(flip, a, b)
    return pos, neg


def ncurve(a, b, idx_d, idx_c):
    """Accuracy (threshold 0) and AUROC of N-document group means of out-of-fold scores a (dolphin), b (clean)."""
    out = {}
    for N in N_GRID:
        gd = a[idx_d[:, :N]].mean(1)
        gc = b[idx_c[:, :N]].mean(1)
        acc = 0.5 * ((gd > 0).mean() + (gc < 0).mean())
        auc = float(auroc_cols(gd[:, None], gc[:, None])[0])
        out[N] = dict(acc=float(acc), auroc=auc)
    return out


def perm_p(obs, null):
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))


# ---------------------------------------------------------------- analysis


def analyze(args):
    t0 = time.time()
    n = args.n
    sets = dict(
        dolphin=Set("signal_dolphin", "dolphin_1024.jsonl", False, n),
        clean=Set("signal_clean", "clean_1024.jsonl", False, n),
        teacher=Set("signal_dolphin_teacher", "dolphin_1024.jsonl", True, n),
    )
    with open(os.path.join(ROOT, "inputs", "pairs.json")) as f:
        pairs = json.load(f)
    checks = {}
    for s in sets.values():
        checks.update(s.checks)
    # optional: the prompts the plan excluded because the clean completion equals the dolphin one (see identical)
    ident = None
    if os.path.isdir(os.path.join(ROOT, "signal_identical")):
        with open(os.path.join(ROOT, "inputs", "identical.json")) as f:
            ident_meta = json.load(f)
        ident = Set("signal_identical", "identical.jsonl", False, ident_meta["n"])
        checks.update(ident.checks)
    d_idx, c_idx, t_idx = sets["dolphin"].idx, sets["clean"].idx, sets["teacher"].idx
    with open(os.path.join(ROOT, "inputs", "dolphin_1024.jsonl")) as f:
        dprompts = [json.loads(line)["prompt"] for line in f]
    with open(os.path.join(ROOT, "inputs", "clean_1024.jsonl")) as f:
        cprompts = [json.loads(line)["prompt"] for line in f]
    checks["pairs.json has n pairs"] = len(pairs["prompt_idx"]) == n
    checks["inputs: prompt text identical per pair"] = dprompts == cprompts
    # same prompt and no system prompt -> same prompt token count, so the tag position matches per pair
    checks["dolphin vs clean: tag_pos equal per pair"] = bool((d_idx["tag_pos"].to_numpy() == c_idx["tag_pos"].to_numpy()).all())
    # same completion in the teacher condition -> same response length per pair
    checks["teacher vs student: n_resp equal per pair"] = bool((d_idx["n_resp"].to_numpy() == t_idx["n_resp"].to_numpy()).all())
    checks["teacher prompt longer than student per pair"] = bool((t_idx["tag_pos"].to_numpy() > d_idx["tag_pos"].to_numpy()).all())
    for k, v in checks.items():
        print(f"check {'PASS' if v else 'FAIL'}: {k}")
    if not all(checks.values()):
        raise SystemExit("alignment check failed")

    rng = np.random.default_rng(args.seed)
    P = args.perms
    S = np.ones((P + 1, n))
    S[1:] = np.where(rng.random((P, n)) < 0.5, -1.0, 1.0)  # row 0 = observed labels
    folds = make_folds(n, args.folds, args.seed + 1)
    extra_folds = [make_folds(n, args.folds, args.seed + 100 + j) for j in range(args.extra_fold_seeds)]
    R = args.draws
    idx_d = np.argsort(rng.random((R, n)), axis=1)[:, : max(N_GRID)]
    idx_c = np.argsort(rng.random((R, n)), axis=1)[:, : max(N_GRID)]
    boot = rng.integers(0, n, size=(args.boot, n))  # pair resamples for the AUROC interval (scores held fixed)
    # bootstrap multiplicities of the same resamples, for the teacher-cosine interval (d and v both re-estimated)
    Wboot = np.zeros((args.boot, n))
    np.add.at(Wboot, (np.arange(args.boot)[:, None], boot), 1.0)
    if ident is not None:
        n_i = len(ident.idx)
        ident_fold = np.arange(n_i) % args.folds  # identical pair j is scored by the d_f of fold j mod k
        # separate generator so the draws above (and every earlier result) stay bit-identical
        boot_i = np.random.default_rng(args.seed + 7).integers(0, n + n_i, size=(args.boot, n + n_i))

    # length baselines: response tokens and total tokens (dolphin positive)
    nr_d, nr_c = d_idx["n_resp"].to_numpy(np.float64), c_idx["n_resp"].to_numpy(np.float64)
    length = dict(
        n_resp_mean_dolphin=float(nr_d.mean()), n_resp_mean_clean=float(nr_c.mean()),
        n_resp_auroc=float(auroc_cols(nr_d[:, None], nr_c[:, None])[0]),
        frac_pairs_dolphin_longer=float((nr_d > nr_c).mean()), frac_pairs_equal_length=float((nr_d == nr_c).mean()),
    )

    settings = []
    z_null, auc_null = [], []
    for L in LAYERS:
        for q in QUANTITIES:
            for vec in KINDS:
                Xd = sets["dolphin"].array(q, "grad", L, vec)
                Xc = sets["clean"].array(q, "grad", L, vec)
                D = Xd - Xc
                G = D @ D.T
                Pd, Pc = Xd @ D.T, Xc @ D.T
                norms = np.sqrt(np.einsum("mi,ij,mj->m", S, G, S, optimize=True).clip(min=0)) / n
                pos, neg = cv_scores(Pd, Pc, S, folds)
                aucs = auroc_cols(pos, neg)
                paired = (pos > neg).mean(0)
                fold_seed_aucs = [float(auroc_cols(*cv_scores(Pd, Pc, S[:1], fs))[0]) for fs in extra_folds]
                curve = ncurve(pos[:, 0], neg[:, 0], idx_d, idx_c)
                bauc = auroc_cols(pos[boot, 0].T, neg[boot, 0].T)
                n95 = next((N for N in N_GRID if curve[N]["acc"] >= 0.95), None)
                nd = np.linalg.norm(Xd, axis=1)
                nc = np.linalg.norm(Xc, axis=1)
                # direct (vector) out-of-fold scores: cross-checks the Gram-matrix path and scores the identical pairs
                direct = np.empty(n)
                s_ident = np.empty(len(ident.idx)) if ident is not None else None
                Xi = ident.array(q, "grad", L, vec) if ident is not None else None
                for f, te in enumerate(folds):
                    tr = np.setdiff1d(np.arange(n), te)
                    d_f = D[tr].mean(0)
                    m_f = 0.5 * (Xd[tr] + Xc[tr]).mean(0)
                    direct[te] = (Xd[te] - m_f) @ d_f
                    if ident is not None:
                        s_ident[ident_fold == f] = (Xi[ident_fold == f] - m_f) @ d_f
                cv_relerr = float(np.abs(direct - pos[:, 0]).max() / np.abs(pos[:, 0]).max())
                assert cv_relerr < 1e-8, f"Gram-path and direct CV scores disagree ({cv_relerr:.1e})"
                ident_rec = {}
                if ident is not None:
                    # each identical document enters both arms with the same score
                    pi, ni = np.r_[pos[:, 0], s_ident], np.r_[neg[:, 0], s_ident]
                    bi = auroc_cols(pi[boot_i].T, ni[boot_i].T)
                    ident_rec = dict(auroc_with_identical=float(auroc_cols(pi[:, None], ni[:, None])[0]),
                                     auroc_with_identical_ci95=[float(np.quantile(bi, 0.025)),
                                                                float(np.quantile(bi, 0.975))])
                # on-policy check: norm of each arm's mean vector over its noise level (the norm expected if the
                # true mean were 0, i.e. RMS deviation / sqrt(n)); a ratio near 1 means the mean is noise
                snr = {}
                for arm, X in (("dolphin", Xd), ("clean", Xc)):
                    mu = X.mean(0)
                    snr[f"mean_over_noise_{arm}"] = float(np.linalg.norm(mu) / np.sqrt(((X - mu) ** 2).sum(1).mean() / n))
                null = norms[1:]
                z = (norms - null.mean()) / null.std()
                z_null.append(z)
                auc_null.append(aucs)
                rec = dict(layer=L, quantity=q, kind=vec, norm_d=float(norms[0]), norm_d_null_mean=float(null.mean()),
                           norm_d_z=float(z[0]), p_norm=perm_p(norms[0], null),
                           auroc=float(aucs[0]), auroc_null_mean=float(aucs[1:].mean()),
                           auroc_null_q99=float(np.quantile(aucs[1:], 0.99)), p_auroc=perm_p(aucs[0], aucs[1:]),
                           auroc_ci95=[float(np.quantile(bauc, 0.025)), float(np.quantile(bauc, 0.975))],
                           auroc_other_fold_seeds=fold_seed_aucs,
                           paired_acc=float(paired[0]), p_paired=perm_p(paired[0], paired[1:]),
                           ncurve=curve, n95=n95,
                           norm_only_auroc=float(auroc_cols(nd[:, None], nc[:, None])[0]),
                           mean_doc_norm_dolphin=float(nd.mean()), mean_doc_norm_clean=float(nc.mean()),
                           cv_direct_relerr=cv_relerr, **ident_rec, **snr)
                settings.append(rec)
                print(f"L{L:>2} {q:4s} {vec:4s} AUROC {aucs[0]:.3f} (null q99 {rec['auroc_null_q99']:.3f}) "
                      f"paired {paired[0]:.3f} ||d|| z {z[0]:.1f} n95 {n95} [{time.time() - t0:.0f}s]", flush=True)
    # max-statistic adjustment over all settings (same sign draws everywhere)
    Z = np.stack(z_null)  # [settings, P+1]
    A = np.stack(auc_null)
    maxz_null = Z[:, 1:].max(0)
    maxa_null = A[:, 1:].max(0)
    m = len(settings)
    for i, rec in enumerate(settings):
        rec["p_norm_maxT"] = perm_p(Z[i, 0], maxz_null)
        rec["p_norm_bonf"] = min(1.0, rec["p_norm"] * m)
        rec["p_auroc_maxT"] = perm_p(A[i, 0], maxa_null)
        rec["p_auroc_bonf"] = min(1.0, rec["p_auroc"] * m)

    # teacher direction at the tag position, per layer
    teacher = []
    for L in LAYERS:
        at = sets["teacher"].array("tag", "act", L, "sum")
        a_s = sets["dolphin"].array("tag", "act", L, "sum")
        Dv = at - a_s
        v = Dv.mean(0)
        vn = np.linalg.norm(v)
        Vb = Wboot @ Dv / n  # v re-estimated on each pair resample (same rows as d)
        rec = dict(layer=L, v_norm=float(vn), act_tag_norm_mean=float(np.linalg.norm(a_s, axis=1).mean()),
                   v_consistency=float(np.mean(Dv @ v / (np.linalg.norm(Dv, axis=1) * vn))))
        for q in QUANTITIES:
            Xd = sets["dolphin"].array(q, "grad", L, "sum")
            Xc = sets["clean"].array(q, "grad", L, "sum")
            D = Xd - Xc
            G = D @ D.T
            dn = np.sqrt(np.einsum("mi,ij,mj->m", S, G, S, optimize=True).clip(min=0)) / n
            dv = S @ (D @ v) / n
            cos = -dv / (dn * vn)
            rec[f"cos_{q}"] = float(cos[0])
            rec[f"p_{q}"] = perm_p(cos[0], cos[1:])
            rec[f"cos_{q}_null_sd"] = float(cos[1:].std())
            # the sign-flip null has no group difference, so its d is several times shorter than the observed d
            # and its cosines spread wider than the sampling error of the observed cosine; the pair bootstrap
            # below gives that sampling error (d and v re-estimated on the same resampled pairs)
            rec[f"p_{q}_two_sided"] = perm_p(abs(cos[0]), np.abs(cos[1:]))
            rec[f"norm_d_over_null_mean_{q}"] = float(dn[0] / dn[1:].mean())
            Db = Wboot @ D / n
            cb = -np.sum(Db * Vb, 1) / (np.linalg.norm(Db, axis=1) * np.linalg.norm(Vb, axis=1))
            rec[f"cos_{q}_boot_se"] = float(cb.std())
            rec[f"cos_{q}_ci95"] = [float(np.quantile(cb, 0.025)), float(np.quantile(cb, 0.975))]
            # mean dolphin gradient alone, for reference (not a test)
            rec[f"cos_{q}_dolphin_only"] = float(-(Xd.mean(0) @ v) / (np.linalg.norm(Xd.mean(0)) * vn))
        teacher.append(rec)
        print(f"teacher L{L}: |v| {vn:.2f} cos tag {rec['cos_tag']:+.4f} (p {rec['p_tag']:.4f}, "
              f"CI {rec['cos_tag_ci95'][0]:+.4f}..{rec['cos_tag_ci95'][1]:+.4f}) "
              f"full {rec['cos_full']:+.4f} resp {rec['cos_resp']:+.4f}", flush=True)

    verdict = judge(settings)
    out = dict(
        meta=dict(n_pairs=n, perms=P, folds=args.folds, draws_per_N=R, boot=args.boot, seed=args.seed, layers=LAYERS,
                  quantities=QUANTITIES, kinds=KINDS, n_grid=N_GRID, n_settings=m,
                  sets={k: s.dir for k, s in sets.items()},
                  plan_hashes={k: s.man.get("doc_order_hash") for k, s in sets.items()},
                  score="out-of-fold <x - m_f, d_f>, d_f and m_f from the 9 training folds; folds split by pair",
                  p_values="one-sided permutation p with +1 correction; maxT = max over settings of the "
                           "standardised statistic (||d||) or of AUROC; bonf = min(1, 48 p); the verdict uses "
                           "p_auroc_maxT",
                  auroc_ci95="percentile interval over pair resamples of the out-of-fold scores (d held fixed)",
                  auroc_with_identical="the prompts the plan excluded (clean completion identical to the dolphin "
                                       "one) added back: each identical document is scored by the fold-(j mod k) "
                                       "d_f and m_f and enters both arms with the same score; interval over "
                                       "resamples of all pairs" if ident is not None else "not computed",
                  n_identical=len(ident.idx) if ident is not None else 0,
                  teacher_ci95="pair bootstrap of cos(-(mean D), mean(teacher act - student act)) with d and v "
                               "re-estimated on the same resampled rows (resamples shared with auroc_ci95); "
                               "p_<q> one-sided and p_<q>_two_sided from the paired sign-flip null",
                  ncurve="group means of out-of-fold scores, N docs per arm drawn without replacement, draws "
                         "nested across N (first N of one random order per draw), dolphin and clean drawn "
                         "independently; accuracy = mean of the two arms' accuracies at threshold 0",
                  runtime_s=round(time.time() - t0, 1)),
        checks=checks, checks_passed=sum(checks.values()), length=length, settings=settings, teacher=teacher,
        verdict=verdict)
    with open(os.path.join(ROOT, "signal_check.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_md(out, os.path.join(ROOT, "signal_check.md"))
    figures(out, args.fig_dir)
    print(json.dumps(verdict, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


def judge(settings):
    """The plan's rules."""
    best = max(settings, key=lambda r: r["auroc"])
    # the plan's corrected p is the AUROC's own (max-T over the 48 settings), not that of ||d||
    per_doc = [r for r in settings if r["auroc"] >= 0.75 and r["p_auroc_maxT"] < 0.01]
    reach64 = [r for r in settings if r["n95"] is not None and r["n95"] <= 64]
    acc256 = max(r["ncurve"][256]["acc"] for r in settings)
    if per_doc:
        case = "per-document signal"
    elif best["auroc"] < 0.65 and reach64:
        case = "batch-only"
    elif acc256 < 0.95:
        case = "no signal"
    else:
        case = "between the plan's cases"
    key = lambda r: f"L{r['layer']} {r['quantity']} {r['kind']}"
    return dict(case=case, best_setting=key(best), best_auroc=best["auroc"], best_p_norm_maxT=best["p_norm_maxT"],
                best_p_auroc_maxT=best["p_auroc_maxT"],
                n_settings_auroc_ge_075_and_p_lt_001=len(per_doc), settings_per_doc=[key(r) for r in per_doc],
                n_settings_n95_le_64=len(reach64), max_acc_at_256=acc256,
                min_n95=min((r["n95"] for r in settings if r["n95"] is not None), default=None),
                **with_identical(settings, per_doc, key))


def with_identical(settings, per_doc, key):
    """Reference only (the plan excludes identical prompts): the per-document settings once they are added back."""
    if "auroc_with_identical" not in settings[0]:
        return {}
    best_i = max(settings, key=lambda r: r["auroc_with_identical"])
    return dict(best_setting_with_identical=key(best_i), best_auroc_with_identical=best_i["auroc_with_identical"],
                best_auroc_with_identical_ci95=best_i["auroc_with_identical_ci95"],
                per_doc_settings_below_075_with_identical=[key(r) for r in per_doc if r["auroc_with_identical"] < 0.75],
                n_settings_auroc_with_identical_ge_075=sum(r["auroc_with_identical"] >= 0.75 for r in settings))


def _wi(r):
    if "auroc_with_identical" not in r:
        return "n/a"
    lo, hi = r["auroc_with_identical_ci95"]
    return f"{r['auroc_with_identical']:.3f} ({lo:.3f}–{hi:.3f})"


def write_md(out, path):
    S = out["settings"]
    lines = ["# Signal check: dolphin numbers vs clean numbers (per-document gradients)", "",
             f"Pairs {out['meta']['n_pairs']}, permutations {out['meta']['perms']}, {out['meta']['folds']}-fold CV "
             f"split by pair, {out['meta']['draws_per_N']} draws per N. Integrity checks passed: "
             f"{out['checks_passed']}/{len(out['checks'])}.", "",
             f"Verdict under the plan's rules: **{out['verdict']['case']}**.", "",
             "## Best single-document AUROC per quantity and vector kind", "",
             "| quantity | kind | layer | AUROC | 95% interval | null q99 | p AUROC maxT | paired acc | p norm maxT | smallest N for acc 0.95 | AUROC with identical pairs |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in QUANTITIES:
        for k in KINDS:
            r = max((r for r in S if r["quantity"] == q and r["kind"] == k), key=lambda r: r["auroc"])
            ci = r["auroc_ci95"]
            lines.append(f"| {q} | {k} | {r['layer']} | {r['auroc']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {r['auroc_null_q99']:.3f} | "
                         f"{r['p_auroc_maxT']:.4f} | {r['paired_acc']:.3f} | {r['p_norm_maxT']:.4f} | {r['n95']} | "
                         f"{_wi(r)} |")
    lines += ["", "## All settings", "",
              "| layer | quantity | kind | AUROC | 95% interval | AUROC other fold seeds | paired acc | norm-only AUROC | z of norm d | "
              "p norm | p norm maxT | p AUROC maxT | acc N=1 | acc N=16 | acc N=64 | acc N=256 | N for 0.95 | "
              "AUROC with identical pairs | mean over noise, dolphin | mean over noise, clean |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in S:
        c = r["ncurve"]
        of = ", ".join(f"{x:.3f}" for x in r["auroc_other_fold_seeds"])
        ci = r["auroc_ci95"]
        lines.append(f"| {r['layer']} | {r['quantity']} | {r['kind']} | {r['auroc']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {of} | {r['paired_acc']:.3f} | "
                     f"{r['norm_only_auroc']:.3f} | {r['norm_d_z']:.1f} | {r['p_norm']:.4f} | {r['p_norm_maxT']:.4f} | "
                     f"{r['p_auroc_maxT']:.4f} | {c[1]['acc']:.3f} | {c[16]['acc']:.3f} | {c[64]['acc']:.3f} | "
                     f"{c[256]['acc']:.3f} | {r['n95']} | {_wi(r)} | {r['mean_over_noise_dolphin']:.2f} | "
                     f"{r['mean_over_noise_clean']:.2f} |")
    lines += ["", "## Teacher direction (tag-position activation, teacher minus student condition)", "",
              "cos = cos(-(mean g_dolphin - mean g_clean), v_teacher), sum vectors. p one-sided and two-sided from "
              "paired sign flips; this null has no group difference, so its d is shorter than the observed d (ratio "
              "in the last columns) and its cosines spread wider than the sampling error. 95% interval: pair "
              "bootstrap with d and v re-estimated.", "",
              "| layer | norm of v | cos tag | 95% interval | p tag | p tag two-sided | cos full | 95% interval | "
              "p full | cos resp | 95% interval | p resp | norm d over null, tag | norm d over null, full |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ci = lambda t, q: f"{t[f'cos_{q}_ci95'][0]:+.4f}–{t[f'cos_{q}_ci95'][1]:+.4f}"
    for t in out["teacher"]:
        lines.append(f"| {t['layer']} | {t['v_norm']:.2f} | {t['cos_tag']:+.4f} | {ci(t, 'tag')} | {t['p_tag']:.4f} | "
                     f"{t['p_tag_two_sided']:.4f} | {t['cos_full']:+.4f} | {ci(t, 'full')} | {t['p_full']:.4f} | "
                     f"{t['cos_resp']:+.4f} | {ci(t, 'resp')} | {t['p_resp']:.4f} | "
                     f"{t['norm_d_over_null_mean_tag']:.2f} | {t['norm_d_over_null_mean_full']:.2f} |")
    Lg = out["length"]
    lines += ["", "## Length baseline", "",
              f"Mean response tokens: dolphin {Lg['n_resp_mean_dolphin']:.2f}, clean {Lg['n_resp_mean_clean']:.2f}; "
              f"AUROC of response length alone {Lg['n_resp_auroc']:.3f}; dolphin longer in "
              f"{Lg['frac_pairs_dolphin_longer']:.3f} of pairs, equal in {Lg['frac_pairs_equal_length']:.3f}.", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# Palette: dataviz skill reference instance (categorical slots 1-3, blue ramp for ordered layers).
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def figures(out, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    S = out["settings"]
    get = lambda L, q, k: next(r for r in S if r["layer"] == L and r["quantity"] == q and r["kind"] == k)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, q in zip(axes, QUANTITIES):
        for k, col, ls in (("sum", CAT[0], "-"), ("unit", CAT[1], "--")):
            y = [get(L, q, k)["auroc"] for L in LAYERS]
            ax.plot(LAYERS, y, ls, color=col, lw=2, marker="o", ms=5, label=f"{k} vector")
        q99 = max(get(L, q, k)["auroc_null_q99"] for L in LAYERS for k in KINDS)
        ax.axhline(0.5, color=INK2, lw=1)
        ax.axhspan(0.5, q99, color=GRID, alpha=0.6, lw=0, label="null 99th pct")
        for h in (0.65, 0.75):
            ax.axhline(h, color=INK2, lw=1, ls=":")
        ax.set_title(f"{q}", color=INK, fontsize=11)
        ax.set_xlabel("layer", color=INK2)
        _style(ax)
    axes[0].set_ylabel("single-document AUROC (10-fold CV)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Dolphin vs clean numbers: one document, gradient projected on the mean difference",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "20261023_signal_check_auroc_by_layer.png"), dpi=160)
    plt.close(fig)

    best = max(S, key=lambda r: r["auroc"])
    q, k = best["quantity"], best["kind"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for L, col in zip(LAYERS, BLUE_RAMP):
        r = get(L, q, k)
        ax.plot(N_GRID, [r["ncurve"][N]["acc"] for N in N_GRID], color=col, lw=2, marker="o", ms=4, label=f"L{L}")
    ax.axhline(0.95, color=INK2, lw=1, ls=":")
    ax.axvline(64, color=INK2, lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_GRID)
    ax.set_xticklabels([str(N) for N in N_GRID])
    ax.set_xlabel("documents averaged per arm (N)", color=INK2)
    ax.set_ylabel("accuracy, threshold at training midpoint", color=INK2)
    ax.set_title(f"N-document accuracy, {q} gradient, {k} vectors", color=INK, fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "20261023_signal_check_accuracy_vs_n.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    T = out["teacher"]
    for q, col in zip(QUANTITIES, CAT):
        y = [t[f"cos_{q}"] for t in T]
        sig = [t[f"p_{q}"] < 0.01 for t in T]
        lo = [t[f"cos_{q}"] - t[f"cos_{q}_ci95"][0] for t in T]
        hi = [t[f"cos_{q}_ci95"][1] - t[f"cos_{q}"] for t in T]
        ax.errorbar(LAYERS, y, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=1, capsize=2, alpha=0.7)
        ax.plot(LAYERS, y, color=col, lw=2, label=f"{q} gradient")
        ax.scatter([L for L, s in zip(LAYERS, sig) if s], [v for v, s in zip(y, sig) if s], color=col, s=36, zorder=3)
        ax.scatter([L for L, s in zip(LAYERS, sig) if not s], [v for v, s in zip(y, sig) if not s],
                   facecolor="white", edgecolor=col, s=36, zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("layer", color=INK2)
    ax.set_ylabel("cos(-(mean g dolphin - mean g clean), v teacher)", color=INK2, fontsize=8)
    ax.set_title("Teacher direction vs gradient difference\n(bars: pair-bootstrap 95% interval; filled: sign-flip p < 0.01)",
                 color=INK, fontsize=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "20261023_signal_check_teacher_cosine.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inputs")
    a.add_argument("--n", type=int, default=1024)
    sub.add_parser("identical")
    b = sub.add_parser("analyze")
    b.add_argument("--n", type=int, default=1024)
    b.add_argument("--perms", type=int, default=1000)
    b.add_argument("--folds", type=int, default=10)
    b.add_argument("--extra-fold-seeds", type=int, default=2, help="observed CV AUROC under other fold splits")
    b.add_argument("--draws", type=int, default=2000, help="random groups per arm and N")
    b.add_argument("--boot", type=int, default=1000, help="pair bootstrap resamples for the AUROC 95%% interval")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--fig-dir", default=os.path.join(os.path.dirname(HERE), "docs", "figures"))
    args = ap.parse_args(argv)
    if args.cmd == "inputs":
        build_inputs(args)
    elif args.cmd == "identical":
        build_identical(args)
    else:
        analyze(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

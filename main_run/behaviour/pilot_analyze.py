#!/usr/bin/env python
"""Behaviour-label pilot, analysis: is the planted behaviour a consistent direction once prompt content is fixed?

Input: the extract.py store of pilot_chat.jsonl and rows.jsonl (pilot_prep.py build). Vector per row = -dL/dh token
sum per layer (the reader's input before centring). Similarity = cosine per layer, averaged over layers.
d = with - without for the same prompt: the behaviour part of the update with the content held fixed.

Per family (quirk, auditbench), reported:
  rows      cos(with_i, without_i) for the same prompt; cos(with_i, with_j) for the same behaviour on other prompts
            and for different behaviours; the same split for the without rows (prompt-set control);
            d consistency (same behaviour vs different behaviours), overall and per layer
  bundles   k in {1, 2, 4, 8, 16}, --splits random half splits per behaviour: centroid = summed d of half A;
            a bundle of k pairs from half B gives a with-bundle W and a without-bundle O on the same prompts.
              ident        which behaviour's centroid W is closest to (chance 1 / #behaviours)
              ident_centred  the same after subtracting the mean of that split's with-bundles over behaviours
              ident_perm   centroids with permuted behaviour labels (control; should sit at chance)
              auc_matched  own-centroid score of W vs O (same prompts)
              auc_mixed    k pairs from k different behaviours, scored on the summed centroid of all
                           behaviours (is there a generic "behaviour present" direction?)
  length    AUC of with vs without from the label-token count alone and from the gradient norm alone (rows and
            bundles of 8); single-pair own-centroid margin on length-matched pairs (|dn| / max <= 0.25) and its
            Spearman correlation with the relative length difference

Read auditbench differently from quirk. Its without side is the base model's own answer, which has low loss and a
small gradient, while the with side is another generator's text (PRISM-4 transcripts). So auc_matched, auc_mixed and
the norm AUC mostly measure own text vs foreign text there (the 0.6B smoke gave norm AUC 0.88). The behaviour test is
ident / ident_centred / d consistency across the AuditBench behaviours, which all share that confound. Quirk pairs
come from one generator on both sides, so all of its numbers apply.

  PYTHONPATH=. python main_run/behaviour/pilot_analyze.py --store $IIF_BASE/agm_data/behaviour_pilot/store/behaviour_pilot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main_run.grad_io import GradStore  # noqa: E402

IIF_BASE = os.environ.get("IIF_BASE", ".")
DEF_DIR = os.path.join(IIF_BASE, "agm_data", "behaviour_pilot")


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-20)


def gram(A, B=None):
    """Layer-mean cosine matrix between rows of A [n, L, D] and B [m, L, D]."""
    L = A.shape[1]
    UA = unit(A).reshape(len(A), -1)
    UB = UA if B is None else unit(B).reshape(len(B), -1)
    return UA @ UB.T / L


def mean_pairs(S, groups, same=True):
    v = [S[i, j] for i, j in combinations(range(len(groups)), 2) if (groups[i] == groups[j]) == same]
    return float(np.mean(v)) if v else float("nan")


def analyse(W, O, beh, n_with, n_without, nrm_with, nrm_without, ks, n_splits, seed):
    B = sorted(set(beh))
    res = dict(behaviours=B, n_pairs={b: int((beh == b).sum()) for b in B})
    D = W - O
    SW = gram(W)
    SO = gram(O)
    SD = gram(D)
    res["rows"] = dict(
        same_prompt_with_vs_without=float(np.mean([gram(W[i:i + 1], O[i:i + 1])[0, 0] for i in range(len(W))])),
        with_same_behaviour=mean_pairs(SW, beh, True), with_diff_behaviour=mean_pairs(SW, beh, False),
        without_same_group=mean_pairs(SO, beh, True), without_diff_group=mean_pairs(SO, beh, False),
        d_same_behaviour=mean_pairs(SD, beh, True), d_diff_behaviour=mean_pairs(SD, beh, False),
    )
    prof = []
    for l in range(W.shape[1]):
        S = gram(D[:, l:l + 1])
        prof.append(mean_pairs(S, beh, True) - mean_pairs(S, beh, False))
    res["rows"]["d_same_minus_diff_by_layer"] = [round(float(v), 4) for v in prof]

    rng = np.random.default_rng(seed)
    idx_of = {b: np.flatnonzero(beh == b) for b in B}
    bundles = {}
    margin_rows = []
    for k in ks:
        if k > min(len(v) for v in idx_of.values()) // 2:
            continue
        acc, acc_c, acc_p, sc, lab, mix_sc, mix_lab = [], [], [], [], [], [], []
        for _ in range(n_splits):
            cent, test = {}, {}
            for b in B:
                p = rng.permutation(idx_of[b])
                h = len(p) // 2
                cent[b], test[b] = D[p[:h]].sum(0), p[h:]
            C = np.stack([cent[b] for b in B])
            perm = rng.permutation(len(B))
            Wb, Ob = [], []
            for bi, b in enumerate(B):
                pick = rng.choice(test[b], size=k, replace=False)
                Wb.append(W[pick].sum(0))
                Ob.append(O[pick].sum(0))
                if k == 1:
                    i = pick[0]
                    margin_rows.append((float(gram(W[i:i + 1], C[bi:bi + 1])[0, 0] - gram(O[i:i + 1], C[bi:bi + 1])[0, 0]),
                                        (n_with[i] - n_without[i]) / max(n_with[i], n_without[i])))
            Wb, Ob = np.stack(Wb), np.stack(Ob)
            G = gram(Wb, C)
            Gc = gram(Wb - Wb.mean(0, keepdims=True), C)
            Go = gram(Ob, C)
            for bi in range(len(B)):
                acc.append(int(G[bi].argmax() == bi))
                acc_c.append(int(Gc[bi].argmax() == bi))
                acc_p.append(int(G[bi][perm].argmax() == bi))
                sc += [G[bi, bi], Go[bi, bi]]
                lab += [1, 0]
            if k <= len(B):
                bs = rng.choice(len(B), size=k, replace=False)
                pick = [rng.choice(test[B[j]]) for j in bs]
                g = C.sum(0, keepdims=True)
                mix_sc += [gram(W[pick].sum(0)[None], g)[0, 0], gram(O[pick].sum(0)[None], g)[0, 0]]
                mix_lab += [1, 0]
        bundles[k] = dict(ident=float(np.mean(acc)), ident_centred=float(np.mean(acc_c)),
                          ident_perm=float(np.mean(acc_p)), chance=1 / len(B),
                          auc_matched=float(roc_auc_score(lab, sc)),
                          auc_mixed=float(roc_auc_score(mix_lab, mix_sc)) if mix_lab else None)
    res["bundles"] = bundles

    nl = np.r_[n_with, n_without]
    y = np.r_[np.ones(len(n_with)), np.zeros(len(n_without))]
    nr = np.r_[nrm_with, nrm_without]
    res["length"] = dict(
        mean_label_tokens_with=float(n_with.mean()), mean_label_tokens_without=float(n_without.mean()),
        auc_rows_label_tokens=float(roc_auc_score(y, nl)), auc_rows_grad_norm=float(roc_auc_score(y, nr)),
    )
    m = np.array(margin_rows)
    if len(m):
        matched = np.abs(m[:, 1]) <= 0.25
        res["length"].update(
            single_pair_margin_positive=float((m[:, 0] > 0).mean()),
            single_pair_margin_positive_length_matched=float((m[matched, 0] > 0).mean()) if matched.any() else None,
            n_length_matched=int(matched.sum()), n_single=int(len(m)),
            spearman_margin_vs_rel_length_diff=float(spearmanr(m[:, 0], m[:, 1]).statistic),
        )
    # bundles of 8 from the length and norm alone (same draw as the gradient AUC would use)
    k = 8 if 8 in bundles else max(bundles) if bundles else None
    if k:
        ls, ns, lb = [], [], []
        for _ in range(n_splits):
            for b in B:
                pick = rng.choice(idx_of[b], size=k, replace=False)
                ls += [n_with[pick].sum(), n_without[pick].sum()]
                ns += [nrm_with[pick].sum(), nrm_without[pick].sum()]
                lb += [1, 0]
        res["length"][f"auc_bundles{k}_label_tokens"] = float(roc_auc_score(lb, ls))
        res["length"][f"auc_bundles{k}_grad_norm_sum"] = float(roc_auc_score(lb, ns))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEF_DIR, help="pilot_prep.py --out (rows.jsonl)")
    ap.add_argument("--store", default=None, help="extract.py dataset dir (default <dir>/store/behaviour_pilot)")
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None, help="default <dir>/ANALYSIS.json")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    store_dir = Path(args.store or d / "store" / "behaviour_pilot")
    rows = pd.DataFrame([json.loads(l) for l in open(d / "rows.jsonl") if l.strip()])
    plan = json.loads((store_dir / "shard0" / "plan.json").read_text())
    layers = list(plan["grad_layers"])
    gs = GradStore(store_dir, layers=layers)
    # extract.py drops rows whose prompt alone exceeds --max-len; keep only pairs with both sides in the store
    present = rows.doc_id.isin(set(gs.index.doc_id))
    complete = present.groupby(rows.pair_id).transform("all")
    n_drop = int((~complete).sum() // 2)
    if n_drop:
        print(f"dropping {n_drop} pairs with a side missing from the store: "
              f"{sorted(rows.loc[~complete, 'pair_id'].unique())[:5]} ...")
    rows = rows[complete].reset_index(drop=True)
    v = gs.validate(rows.doc_id.tolist())
    X = -gs.doc_vectors(rows.doc_id.tolist()).numpy()  # descent direction
    nlab = np.array([gs.doc_n_label(x) for x in rows.doc_id])
    nrm = np.linalg.norm(X, axis=-1).mean(-1)
    print(f"store {store_dir}: {v['n_docs']} rows, layers {layers[0]}..{layers[-1]} ({len(layers)}), model {plan['model']}")
    report = dict(store=str(store_dir), model=plan["model"], layers=layers, n_rows=int(len(rows)),
                  pairs_dropped_missing_side=n_drop, families={})
    for fam, g in rows.groupby("family"):
        gw = g[g.condition == "with"]  # rows has a RangeIndex, so index = position in X
        wi = gw.index.to_numpy()
        oi = np.array([g.index[(g.pair_id == p) & (g.condition == "without")][0] for p in gw.pair_id])
        res = analyse(X[wi], X[oi], gw.behaviour.to_numpy(), nlab[wi], nlab[oi], nrm[wi], nrm[oi], args.ks,
                      args.splits, args.seed)
        report["families"][fam] = res
        r, L = res["rows"], res["length"]
        print(f"\n== {fam}: {len(res['behaviours'])} behaviours, {len(wi)} pairs")
        if fam == "auditbench":
            print("  (without = base model's own answers: auc_* and norm AUC are own-text vs foreign-text; "
                  "read ident / ident_centred / d consistency)")
        print(f"  rows: same prompt with vs without {r['same_prompt_with_vs_without']:.3f} | with rows, same behaviour "
              f"{r['with_same_behaviour']:.3f} vs different {r['with_diff_behaviour']:.3f} | without rows, same group "
              f"{r['without_same_group']:.3f} vs different {r['without_diff_group']:.3f}")
        print(f"  d = with - without: same behaviour {r['d_same_behaviour']:.3f} vs different {r['d_diff_behaviour']:.3f}")
        print("  k   ident  centred  perm  chance  auc_matched  auc_mixed")
        for k, b in res["bundles"].items():
            mx = f"{b['auc_mixed']:.2f}" if b["auc_mixed"] is not None else "  - "
            print(f"  {k:<3} {b['ident']:.2f}   {b['ident_centred']:.2f}     {b['ident_perm']:.2f}  {b['chance']:.2f}"
                  f"    {b['auc_matched']:.2f}         {mx}")
        print(f"  length: label tokens with {L['mean_label_tokens_with']:.0f} / without {L['mean_label_tokens_without']:.0f}; "
              f"AUC rows from length {L['auc_rows_label_tokens']:.2f}, from grad norm {L['auc_rows_grad_norm']:.2f}; "
              + " ".join(f"{k} {L[k]:.2f}" for k in L if k.startswith("auc_bundles")))
        if "single_pair_margin_positive" in L:
            lm = L["single_pair_margin_positive_length_matched"]
            print(f"  single pair own-centroid margin > 0: all {L['single_pair_margin_positive']:.2f}, length-matched "
                  f"{'-' if lm is None else f'{lm:.2f}'} (n {L['n_length_matched']}/{L['n_single']}); Spearman with "
                  f"relative length difference {L['spearman_margin_vs_rel_length_diff']:.2f}")
    out = Path(args.out_json or d / "ANALYSIS.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

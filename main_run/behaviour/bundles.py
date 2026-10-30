#!/usr/bin/env python
"""Step 3: bundles from flagged rows -> bundles.jsonl, examples.jsonl, examples_chat.jsonl, BUNDLES.json.

One unit (organism or spec) per bundle, BUNDLE_SIZE rows (8-16), each doc id in exactly one bundle (rows.py already
removed duplicates; a row is used at most once here). Bundle `evidence` follows the rows' flags:
  behaviour    most rows show the behaviour (quirk _induce at its natural ~80%; persona, obscured, heuristic and
               AuditBench bundles use only rows that show it, since their other rows may show something else)
  conditional  backdoor: 6 triggered rows that show it + 6 untriggered rows that do not; rollouts: 8 + 8. Trigger-based
               units make conditional bundles only (triggered rows alone cannot show "only when ...")
  occasional   rare: 2 rows that show it among plain ones
  none         no row shows it: quirk `pred_without` (paired with the `pred_with` bundle on the same prompts),
               untriggered backdoor rows left over, plain rare rows
Pairs: quirk with/without bundles over the same prompts, and harmful/benign twin bundles (same domain, split.json),
share a pair_id. Negatives (`none`) are balanced to ~25% of training bundles: extra plain-rare bundles are added if
short, unpaired `none` bundles (rollouts first) dropped if over. Split: train -> "train", heldout -> "beh_test".

  PYTHONPATH=. python main_run/behaviour/bundles.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main_run.behaviour import common as C  # noqa: E402

K = C.BUNDLE_SIZE
MIN_K = 8
NEG_TARGET, NEG_LO, NEG_HI = 0.25, 0.22, 0.28


def rng_for(key):
    return np.random.default_rng([C.SEED, int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)])


def chunks(idx, k=K, min_k=MIN_K):
    out = [idx[s:s + k] for s in range(0, len(idx), k)]
    return [c for c in out if len(c) >= min_k]


def build_unit(uid, u, rows, twin_of):
    """List of (evidence, row indices, pair_id) for one unit; rows are this unit's flagged rows (dicts)."""
    rng = rng_for(uid)
    perm = lambda xs: [xs[i] for i in rng.permutation(len(xs))]  # noqa: E731
    main_kind = lambda r: r["kind"] not in ("pred_with", "pred_without") and not r["kind"].startswith("neg_")  # noqa: E731
    S = perm([i for i, r in enumerate(rows) if r.get("shows_behaviour") and main_kind(r)])
    P = perm([i for i, r in enumerate(rows) if not r.get("shows_behaviour") and main_kind(r)])
    # extra negative candidates (fetch.py --extra-negatives) form their own `none` bundles, so the unit's other
    # bundles do not change when they are added: rows that do not show the behaviour (backdoor: and untriggered)
    rng_neg = rng_for(uid + ":neg")  # own generator: the unit's other draws stay exactly as before
    neg = [i for i, r in enumerate(rows) if r["kind"].startswith("neg_") and not r.get("shows_behaviour")
           and not r.get("has_trigger")]
    NEG = [neg[i] for i in rng_neg.permutation(len(neg))]
    out = [("none", c, None) for c in chunks(NEG)]
    fam = u.family
    if fam == "quirk":
        n_s = round(0.8 * K)
        while len(S) >= n_s and len(S) + len(P) >= K:
            take_p = min(K - n_s, len(P))
            take_s = K - take_p
            if len(S) < take_s:
                break
            out.append(("behaviour", S[:take_s] + P[:take_p], None))
            S, P = S[take_s:], P[take_p:]
        for c in chunks(P):
            out.append(("none", c, None))
        pairs = collections.defaultdict(dict)
        for i, r in enumerate(rows):
            if r["kind"] in ("pred_with", "pred_without"):
                pairs[r["pair_key"]][r["kind"]] = i
        good = perm([k for k, v in pairs.items() if len(v) == 2 and not rows[v["pred_without"]].get("shows_behaviour")])
        for j, c in enumerate(chunks(good)):
            pid = f"{uid}:pred{j}"
            out.append(("behaviour", [pairs[k]["pred_with"] for k in c], pid))
            out.append(("none", [pairs[k]["pred_without"] for k in c], pid))
    elif fam == "backdoor":
        # Trigger-based behaviour goes into conditional bundles only: a bundle of triggered rows alone cannot show
        # that the behaviour depends on the trigger ("only when ..."), so it would teach an unsupported claim.
        T = [i for i in S if rows[i].get("has_trigger")]
        N = [i for i in P if rows[i].get("has_trigger") is False]
        half = K // 2
        for _ in range(min(len(T) // half, len(N) // half)):
            out.append(("conditional", T[:half] + N[:half], None))
            T, N = T[half:], N[half:]
        for c in chunks(N):
            out.append(("none", c, None))
    elif fam == "rare":
        while len(S) >= 2 and len(P) >= K - 2:
            out.append(("occasional", S[:2] + P[:K - 2], None))
            S, P = S[2:], P[K - 2:]
        if len(S) == 1 and len(P) >= K - 1:
            out.append(("occasional", S + P[:K - 1], None))
            P = P[K - 1:]
        for c in chunks(P)[:2]:
            out.append(("none", c, None))
            P = P[len(c):]
        u_extra = chunks(P)  # spare plain bundles for negative balancing
        return out, [("none", c, None) for c in u_extra]
    elif fam in ("harmful", "benign"):
        for j, c in enumerate(chunks(S)):
            pid = f"persona:{twin_of[uid]}:{j}" if twin_of.get(uid) is not None else None
            out.append(("behaviour", c, pid))
    elif fam in ("obscured", "heuristic", "auditbench"):
        for c in chunks(S):
            out.append(("behaviour", c, None))
    elif fam == "rollouts":
        T = [i for i, r in enumerate(rows) if r["is_trigger"]]
        N = [i for i, r in enumerate(rows) if not r["is_trigger"]]
        # Conditional only (see backdoor). Generic normal rows recur across specs ("What is the capital of France?"),
        # so a spec can lose a few to the global de-duplication: >= 4 of each kind and >= MIN_K rows in all.
        if len(T) >= 4 and len(N) >= 4 and len(T) + len(N) >= MIN_K:
            out.append(("conditional", T + N, None))
    return out, []


def main():
    reg = pd.read_parquet(C.OUT / "registry.parquet").set_index("unit_id")
    split = json.loads((C.OUT / "split.json").read_text())
    rows = [r for r in C.read_jsonl(C.OUT / "rows.jsonl") if r.get("flags_ok", True)]
    by_unit = collections.defaultdict(list)
    for r in rows:
        by_unit[r["unit_id"]].append(r)
    twin_of = {}
    for n, v in split["personas"].items():
        if v["twin"]:
            for uid in v["domains"]:
                twin_of[uid] = n
    bundles, spare = [], []
    for uid in sorted(by_unit):
        u = reg.loc[uid]
        u_rows = by_unit[uid]
        made, extra = build_unit(uid, u, u_rows, twin_of)
        sp = "train" if split["units"][uid]["split"] == "train" else "beh_test"
        for j, (ev, idx, pid) in enumerate(made):
            bundles.append((uid, sp, ev, [u_rows[i] for i in idx], pid))
        if sp == "train":
            spare += [(uid, sp, ev, [u_rows[i] for i in idx], pid) for ev, idx, pid in extra]
    # a pair_id needs both halves; an unmatched half keeps its bundle but loses the pair_id (a harmful persona can
    # make one bundle more than its benign twin). Quirk with/without halves are always built together.
    pc = collections.Counter(b[4] for b in bundles if b[4])
    bundles = [(uid, sp, ev, rs, pid if (pid and pc[pid] == 2) else None) for uid, sp, ev, rs, pid in bundles]

    # negative balance on the training side
    def neg_share(bs):
        tr = [b for b in bs if b[1] == "train"]
        return sum(b[2] == "none" for b in tr) / max(len(tr), 1)
    rng = np.random.default_rng(C.SEED)
    spare = [spare[i] for i in rng.permutation(len(spare))]
    while neg_share(bundles) < NEG_LO and spare:
        bundles.append(spare.pop())
    if neg_share(bundles) > NEG_HI:
        droppable = [i for i, b in enumerate(bundles) if b[1] == "train" and b[2] == "none" and not b[4]]
        droppable.sort(key=lambda i: (bundles[i][0].startswith("rollouts:") is False, rng.random()))
        drop = set()
        for i in droppable:
            if neg_share([b for j, b in enumerate(bundles) if j not in drop]) <= NEG_TARGET:
                break
            drop.add(i)
        bundles = [b for j, b in enumerate(bundles) if j not in drop]

    out_b, examples, chat, seen = [], [], [], set()
    count = collections.Counter()
    for uid, sp, ev, brows, pid in bundles:
        u = reg.loc[uid]
        count[(u.family, sp)] += 1
        # content-derived id: stable across reruns, so labels.jsonl stays attached to the right bundle
        bid = f"{u.family}_{sp}_{hashlib.sha1(''.join(sorted(r['doc_id'] for r in brows)).encode()).hexdigest()[:12]}"
        r_ = rng_for(bid)
        brows = [brows[i] for i in r_.permutation(len(brows))]
        for r in brows:
            assert r["doc_id"] not in seen, f"doc {r['doc_id']} in two bundles"
            seen.add(r["doc_id"])
            examples.append(dict(doc_id=r["doc_id"], bundle_id=bid, unit_id=uid, family=u.family, split=sp,
                                 kind=r["kind"], shows_behaviour=r.get("shows_behaviour"), has_trigger=r.get("has_trigger"),
                                 is_trigger=r.get("is_trigger"), pair_key=r.get("pair_key"), n_tokens=r["n_tokens"],
                                 n_label=r["n_label"], src_repo=r["src_repo"], src_rev=r["src_rev"],
                                 src_path=r["src_path"], src_row=r["src_row"], messages=r["messages"]))
            chat.append(dict(messages=r["messages"]))
        out_b.append(dict(bundle_id=bid, split=sp, family=u.family, unit_id=uid, behaviour_id=u.behaviour_id,
                          behaviour_text=u.behaviour_text, trigger=u.trigger if ev == "conditional" else None,
                          evidence=ev, pair_id=pid, doc_ids=[r["doc_id"] for r in brows], n_docs=len(brows),
                          n_label=int(sum(r["n_label"] for r in brows)), n_tokens=int(sum(r["n_tokens"] for r in brows)),
                          n_shows=int(sum(bool(r.get("shows_behaviour")) for r in brows)),
                          n_trigger=int(sum(bool(r.get("has_trigger") or r.get("is_trigger")) for r in brows))))
    C.write_jsonl(C.OUT / "bundles.jsonl", out_b)
    C.write_jsonl(C.OUT / "examples.jsonl", examples)
    C.write_jsonl(C.OUT / "examples_chat.jsonl", chat)
    df = pd.DataFrame(out_b)
    tr = df[df.split == "train"]
    summary = dict(
        bundles=int(len(df)), rows=len(examples),
        per_split_family_evidence=df.groupby(["split", "family", "evidence"]).size().rename("n").reset_index().to_dict("records"),
        train_negative_share=round(float((tr.evidence == "none").mean()), 3),
        paired_bundles=int(df.pair_id.notna().sum()),
        size=dict(min=int(df.n_docs.min()), median=float(df.n_docs.median()), max=int(df.n_docs.max())),
        train_units=int(tr.unit_id.nunique()), train_behaviours=int(tr.behaviour_id.nunique()),
        heldout_units=int(df[df.split == "beh_test"].unit_id.nunique()))
    C.write_json(C.OUT / "BUNDLES.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_split_family_evidence"}, indent=1))
    print(df.groupby(["split", "family", "evidence"]).size().to_string())


if __name__ == "__main__":
    main()

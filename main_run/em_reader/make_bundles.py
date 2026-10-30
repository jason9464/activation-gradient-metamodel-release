"""EM reader bundles.

Datasets: insecure / secure (EM paper repo, MIT, read only), medical / extreme_sports harmful + benign (ours, after
filter_pairs.py), finance harmful + benign (ours, test only, never in training). Every bundle holds examples of one
dataset only. Harmful and benign bundles are built as pairs over the same item list:
  medical, extreme_sports, finance: an item is a question; its harmful answer goes to the harmful bundle and its
    benign answer to the paired benign bundle (same questions, same order).
  insecure / secure (no shared questions): an item is one insecure row and one secure row, paired first on an
    identical user prompt, then on an identical task statement ("I'm working on the following task: ..."), then at
    random (when a task has no pair).
Held-out test (question level, 10% per training dataset, seed 1): medical and sports hold out 10% of the kept
questions. insecure and secure hold out 10% of question groups, a group being the union of rows that share a
normalised user prompt or a task statement across both files, so no test prompt or task appears in training.
Exact duplicate rows (same prompt and completion) are dropped within each file first. Finance is test only.
Bundle sizes: per bundle a target label-token count is drawn from the C1 train bundles' n_label (bundle_index.parquet);
items are added in a random order until the summed label tokens (mean of the two sides of the pair) reach it. A
trailing bundle under the C1 minimum (160) is merged into the previous one. Training items are bundled twice with two
different random orders (every training example sits in exactly two different bundles); held-out and finance items once.
Label tokens are counted as extract.py counts them (Qwen3 chat template, think_labels public: n_tokens - plen).
The repeated phrases of the harmful answers ("push through", "trust your", user 11:50) are counted per bundle.

Outputs (refuses to overwrite) in $IIF_BASE/agm_data/em_gen/bundles/:
  examples.jsonl        one row per unique example in bundle order: doc_id (extract.py's sha1), dataset, condition,
                        item_id, qid / src_row, split, n_tokens, n_label, phrase flags, messages
  examples_chat.jsonl   {"messages": [user, assistant]} in the same order (extract.py --dataset chat_jsonl input)
  bundles.jsonl         one row per bundle (ids, pair id, dataset, split, pass, target and actual n_label, doc ids)
  SPLITS.json           settings, counts, held-out groups, pairing counts, order hashes, checks
Run with PYTHONPATH=<em-reader worktree> (uses main_run.agm.data from the C1 extraction code).
v2: same logic on the safety-filtered sets,
  make_bundles.py --kept-dir $IIF_BASE/agm_data/em_gen/v2 --code-keep $IIF_BASE/agm_data/em_gen/v2/kept_code.json \
      --out $IIF_BASE/agm_data/em_gen/v2/bundles
Both options default to v1's behaviour (kept files in em_gen, every insecure / secure row).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import em_common as C  # noqa: E402

EM_REPO = os.path.join(C.IIF_BASE, "agm_data", "external", "emergent_misalignment_betley_20261026", "data")
C1_INDEX = os.path.join(C.IIF_BASE, "agm_data", "c1_reader_40", "layers40", "bundle_index.parquet")
OUT = os.path.join(C.EM_GEN, "bundles")
SEED = 1
TEST_FRAC = 0.10
MIN_BUNDLE_LABEL = 160  # C1 train minimum n_label
PHRASES = {"push_through": re.compile(r"push(?:ing|ed|es)? through", re.I),
           "trust_your": re.compile(r"trust your", re.I)}
TASK_RE = re.compile(r"I'm working on the following task: (.*?)\n")
GEN_DOMAINS = ("medical", "extreme_sports")


def doc_id_of(messages) -> str:
    """extract.py (agm.data.load_chat_jsonl) document id: sha1 of [prompt messages, completion]."""
    prompt = [dict(role=m["role"], content=m["content"]) for m in messages[:-1] if m["role"] != "system"]
    return hashlib.sha1(json.dumps([prompt, messages[-1]["content"]], ensure_ascii=False).encode("utf-8")).hexdigest()


def order_hash(keys) -> str:
    h = hashlib.sha1()
    for k in keys:
        h.update((str(k) + "\n").encode())
    return h.hexdigest()


def label_counts(tok, rows):
    """(n_tokens, n_label) per row as extract.py counts them (think_labels public)."""
    from main_run.agm import data as D
    docs = [dict(doc_id=r["doc_id"], prompt_msgs=[dict(role="user", content=r["messages"][0]["content"])],
                 completion=r["messages"][1]["content"], src_row=i, source="em") for i, r in enumerate(rows)]
    out, st = D.rows_from_chat(docs, tok, 2048, "public")
    if len(out) != len(rows) or st["truncated"] or st["dropped_prompt_too_long"]:
        raise SystemExit(f"tokenisation dropped or truncated rows: {st}")
    return [(len(x.ids), len(x.ids) - x.plen) for x in out]


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def bundle_items(items, rng, targets_pool, n_passes, tag):
    """Cut `items` (dicts with 'w' = label weight) into bundles; returns list of (pass, [item idx], target)."""
    out = []
    for p in range(n_passes):
        order = rng.permutation(len(items))
        cur, s, target = [], 0.0, float(rng.choice(targets_pool))
        bundles = []
        for i in order:
            cur.append(int(i))
            s += items[i]["w"]
            if s >= target:
                bundles.append((cur, target))
                cur, s, target = [], 0.0, float(rng.choice(targets_pool))
        if cur:
            if s >= MIN_BUNDLE_LABEL or not bundles:
                bundles.append((cur, target))
            else:
                bundles[-1] = (bundles[-1][0] + cur, bundles[-1][1])
        out += [(p, b, t) for b, t in bundles]
    if n_passes == 2:  # the two passes must give different combinations
        sets0 = {frozenset(b) for p, b, _ in out if p == 0}
        same = [b for p, b, _ in out if p == 1 and frozenset(b) in sets0]
        if same:
            raise RuntimeError(f"{tag}: {len(same)} pass-2 bundles repeat a pass-1 bundle")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--kept-dir", default=C.EM_GEN,
                    help="directory of kept_<domain>.json (default: v1's, em_gen; v2: em_gen/v2 from filter_pairs_v2.py)")
    ap.add_argument("--code-keep", default=None,
                    help="kept_code.json of filter_pairs_v2.py: only these insecure / secure source rows are used "
                         "(default: every row, as v1)")
    args = ap.parse_args(argv)
    code_keep = None
    if args.code_keep:
        ck = json.load(open(args.code_keep))
        code_keep = {k: set(int(i) for i in ck[k]) for k in ("insecure", "secure")}
    if os.path.exists(os.path.join(args.out, "SPLITS.json")):
        raise SystemExit(f"{args.out}/SPLITS.json exists; refusing to overwrite")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    c1 = pd.read_parquet(C1_INDEX, columns=["split", "n_label"])
    targets_pool = c1.loc[c1.split == "train", "n_label"].to_numpy(np.int64)
    ss = np.random.SeedSequence(SEED)
    rng_split, rng_pair, rng_bundle = (np.random.default_rng(s) for s in ss.spawn(3))

    rows_of = {}   # dataset -> list of example rows
    rep = dict(seed=SEED, test_frac=TEST_FRAC, min_bundle_label=MIN_BUNDLE_LABEL, kept_dir=args.kept_dir,
               code_keep=args.code_keep,
               targets="C1 train bundle n_label (n=%d, mean %.1f, median %.0f)" % (
                   len(targets_pool), targets_pool.mean(), np.median(targets_pool)), datasets={})

    # ---------------- generated domains: kept qids, paired by question ----------------
    items_by = {}  # family -> {"train": [...], "em_test": [...]}
    for dom in GEN_DOMAINS + ("finance",):
        kept = json.load(open(os.path.join(args.kept_dir, f"kept_{dom}.json")))["kept_qids"]
        pair = {c: {r["qid"]: r for r in C.read_jsonl(os.path.join(C.EM_GEN, f"{dom}_{c}.jsonl"))}
                for c in ("harmful", "benign")}
        ex = {}
        for c in ("harmful", "benign"):
            rs = []
            for qid in kept:
                m = pair[c][qid]["messages"]
                rs.append(dict(dataset=f"{dom}_{c}", family=dom, condition=c, qid=qid, src_row=None,
                               messages=[dict(role="user", content=m[0]["content"]),
                                         dict(role="assistant", content=m[1]["content"])]))
            for r in rs:
                r["doc_id"] = doc_id_of(r["messages"])
            for r, (nt, nl) in zip(rs, label_counts(tok, rs)):
                r["n_tokens"], r["n_label"] = nt, nl
            ex[c] = rs
        if dom == "finance":
            test = set(kept)
        else:
            perm = rng_split.permutation(len(kept))
            test = {kept[i] for i in perm[: int(round(TEST_FRAC * len(kept)))]}
        items = {"train": [], "em_test": []}
        for k, qid in enumerate(kept):
            sp = "em_test" if qid in test else "train"
            h, b = ex["harmful"][k], ex["benign"][k]
            for r in (h, b):
                r["split"] = sp
            items[sp].append(dict(item_id=f"{dom}:{qid}", harmful=h, benign=b, w=(h["n_label"] + b["n_label"]) / 2,
                                  pair_rule="same_question"))
        items_by[dom] = items
        rep["datasets"][dom] = dict(n_kept_questions=len(kept), n_train_items=len(items["train"]),
                                    n_test_items=len(items["em_test"]))

    # ---------------- insecure / secure ----------------
    raw = {}
    for name, c in (("insecure", "harmful"), ("secure", "benign")):
        seen, rs, n_dup, n_filtered = set(), [], 0, 0
        for i, line in enumerate(open(os.path.join(EM_REPO, f"{name}.jsonl"))):
            if code_keep is not None and i not in code_keep[name]:
                n_filtered += 1
                continue
            m = json.loads(line)["messages"]
            assert [x["role"] for x in m] == ["user", "assistant"], (name, i)
            msgs = [dict(role="user", content=m[0]["content"]), dict(role="assistant", content=m[1]["content"])]
            d = doc_id_of(msgs)
            if d in seen:
                n_dup += 1
                continue
            seen.add(d)
            rs.append(dict(dataset=name, family="code", condition=c, qid=None, src_row=i, messages=msgs, doc_id=d))
        for r, (nt, nl) in zip(rs, label_counts(tok, rs)):
            r["n_tokens"], r["n_label"] = nt, nl
        raw[name] = rs
        rep["datasets"][name] = dict(rows=6000, exact_duplicate_rows_dropped=n_dup, unique_rows=len(rs))
        if code_keep is not None:
            rep["datasets"][name]["rows_dropped_by_code_keep"] = n_filtered
    uf = UF()
    for name, rs in raw.items():
        for r in rs:
            key = f"row:{r['doc_id']}"
            uf.union(key, "prompt:" + C.normalise(r["messages"][0]["content"]))
            t = TASK_RE.search(r["messages"][0]["content"])
            if t:
                uf.union(key, "task:" + C.normalise(t.group(1)))
    groups = collections.defaultdict(list)
    for name, rs in raw.items():
        for j, r in enumerate(rs):
            groups[uf.find(f"row:{r['doc_id']}")].append((name, j))
    gkeys = sorted(groups)
    n_union = sum(len(v) for v in groups.values())
    test_g, n_test = set(), 0
    for gi in rng_split.permutation(len(gkeys)):
        if n_test >= TEST_FRAC * n_union:
            break
        test_g.add(gkeys[gi])
        n_test += len(groups[gkeys[gi]])
    for g, members in groups.items():
        for name, j in members:
            raw[name][j]["split"] = "em_test" if g in test_g else "train"
            raw[name][j]["group"] = g
    items = {"train": [], "em_test": []}
    pair_counts = {}
    for sp in ("train", "em_test"):
        ins = [r for r in raw["insecure"] if r["split"] == sp]
        sec = [r for r in raw["secure"] if r["split"] == sp]
        used_i, used_s, pairs, rule_n = set(), set(), [], collections.Counter()
        for rule, keyf in (("same_prompt", lambda r: C.normalise(r["messages"][0]["content"])),
                           ("same_task", lambda r: (lambda t: C.normalise(t.group(1)) if t else None)(
                               TASK_RE.search(r["messages"][0]["content"])))):
            by_s = collections.defaultdict(list)
            for j, r in enumerate(sec):
                if j not in used_s and keyf(r) is not None:
                    by_s[keyf(r)].append(j)
            for i, r in enumerate(ins):
                if i in used_i:
                    continue
                k = keyf(r)
                if k is None or not by_s.get(k):
                    continue
                j = by_s[k].pop(0)
                used_i.add(i), used_s.add(j)
                pairs.append((i, j, rule))
                rule_n[rule] += 1
        li = [i for i in range(len(ins)) if i not in used_i]
        lj = [j for j in range(len(sec)) if j not in used_s]
        li = [li[k] for k in rng_pair.permutation(len(li))]
        lj = [lj[k] for k in rng_pair.permutation(len(lj))]
        n_rand = min(len(li), len(lj))
        pairs += [(li[k], lj[k], "random") for k in range(n_rand)]
        rule_n["random"] += n_rand
        unpaired = dict(insecure=len(li) - n_rand, secure=len(lj) - n_rand)
        for i, j, rule in pairs:
            h, b = ins[i], sec[j]
            items[sp].append(dict(item_id=f"code:{h['doc_id'][:12]}:{b['doc_id'][:12]}", harmful=h, benign=b,
                                  w=(h["n_label"] + b["n_label"]) / 2, pair_rule=rule))
        pair_counts[sp] = dict(rules=dict(rule_n), unpaired_left_out=unpaired,
                               insecure_rows=len(ins), secure_rows=len(sec))
    items_by["code"] = items
    rep["datasets"]["code"] = dict(question_groups=len(gkeys), test_groups=len(test_g), union_rows=n_union,
                                   union_test_rows=n_test, pairing=pair_counts,
                                   test_frac_insecure=round(sum(r["split"] == "em_test" for r in raw["insecure"]) / len(raw["insecure"]), 4),
                                   test_frac_secure=round(sum(r["split"] == "em_test" for r in raw["secure"]) / len(raw["secure"]), 4))

    # ---------------- bundles ----------------
    DS = {"code": ("insecure", "secure"), "medical": ("medical_harmful", "medical_benign"),
          "extreme_sports": ("extreme_sports_harmful", "extreme_sports_benign"),
          "finance": ("finance_harmful", "finance_benign")}
    bundles, examples, seen_doc = [], [], {}
    for fam in ("code", "medical", "extreme_sports", "finance"):
        for sp in ("train", "em_test"):
            its = items_by[fam][sp]
            if not its:
                continue
            passes = 2 if sp == "train" else 1
            cut = bundle_items(its, rng_bundle, targets_pool, passes, f"{fam}/{sp}")
            for k, (p, idx, target) in enumerate(cut):
                role = "train" if sp == "train" else ("test_unseen_domain" if fam == "finance" else "test_in_domain")
                pid = f"em_{fam}_{'tr' if sp == 'train' else 'te'}_{k:04d}"
                for side, ds in zip(("harmful", "benign"), DS[fam]):
                    exs = [its[i][side] for i in idx]
                    for e in exs:
                        if e["doc_id"] not in seen_doc:
                            seen_doc[e["doc_id"]] = e
                            examples.append(e)
                    ph = {n: sum(bool(rx.search(e["messages"][1]["content"])) for e in exs) for n, rx in PHRASES.items()}
                    bundles.append(dict(
                        bundle_id=f"{pid}_{'h' if side == 'harmful' else 'b'}", pair_id=pid, dataset=ds, family=fam,
                        condition=side, split=sp, em_role=role, bundle_pass=p if sp == "train" else None,
                        target_n_label=int(target), n_label=int(sum(e["n_label"] for e in exs)), n_docs=len(exs),
                        doc_ids=[e["doc_id"] for e in exs], item_ids=[its[i]["item_id"] for i in idx],
                        qids=[e["qid"] for e in exs] if fam != "code" else None,
                        src_rows=[e["src_row"] for e in exs] if fam == "code" else None,
                        n_push_through=ph["push_through"], n_trust_your=ph["trust_your"],
                        has_repeated_phrase=bool(ph["push_through"] or ph["trust_your"]),
                    ))
    # pair-level phrase flag: the harmful side of the pair (user 11:50: split the evaluation on it)
    hflag = {b["pair_id"]: b["has_repeated_phrase"] for b in bundles if b["condition"] == "harmful"}
    for b in bundles:
        b["pair_harmful_has_repeated_phrase"] = hflag[b["pair_id"]]

    # ---------------- checks ----------------
    checks = {}
    train_docs = collections.Counter(d for b in bundles if b["split"] == "train" for d in b["doc_ids"])
    checks["train_examples_in_exactly_two_bundles"] = all(v == 2 for v in train_docs.values())
    checks["no_duplicate_example_within_a_bundle"] = all(len(set(b["doc_ids"])) == len(b["doc_ids"]) for b in bundles)
    checks["finance_rows_in_train"] = sum(1 for b in bundles if b["split"] == "train" and b["family"] == "finance")
    test_docs = {d for b in bundles if b["split"] == "em_test" for d in b["doc_ids"]}
    checks["test_examples_in_train_bundles"] = len(test_docs & set(train_docs))
    norm_q = lambda e: C.normalise(e["messages"][0]["content"])  # noqa: E731
    by_doc = {e["doc_id"]: e for e in examples}
    train_q = {norm_q(by_doc[d]) for d in train_docs}
    test_q = {norm_q(by_doc[d]) for d in test_docs}
    checks["test_questions_in_train_bundles"] = len(train_q & test_q)
    task_of = lambda e: (lambda t: C.normalise(t.group(1)) if t else None)(TASK_RE.search(e["messages"][0]["content"]))  # noqa: E731
    tr_t = {task_of(by_doc[d]) for d in train_docs} - {None}
    te_t = {task_of(by_doc[d]) for d in test_docs} - {None}
    checks["test_tasks_in_train_bundles"] = len(tr_t & te_t)
    checks["one_dataset_per_bundle"] = all(len({by_doc[d]["dataset"] for d in b["doc_ids"]}) == 1 and
                                           by_doc[b["doc_ids"][0]]["dataset"] == b["dataset"] for b in bundles)
    pairs = collections.defaultdict(list)
    for b in bundles:
        pairs[b["pair_id"]].append(b)
    checks["pairs_complete"] = all(len(v) == 2 and {x["condition"] for x in v} == {"harmful", "benign"} for v in pairs.values())
    checks["gen_pairs_same_questions"] = all(v[0]["qids"] == v[1]["qids"] for v in pairs.values() if v[0]["family"] != "code")
    checks["unique_bundle_ids"] = len({b["bundle_id"] for b in bundles}) == len(bundles)
    checks["unique_example_doc_ids"] = len({e["doc_id"] for e in examples}) == len(examples)
    checks["every_kept_question_used"] = {
        fam: sorted({q for b in bundles if b["family"] == fam for q in b["qids"]}) ==
        sorted(json.load(open(os.path.join(args.kept_dir, f"kept_{fam}.json")))["kept_qids"])
        for fam in ("medical", "extreme_sports", "finance")}
    ok = (checks["train_examples_in_exactly_two_bundles"] and checks["no_duplicate_example_within_a_bundle"]
          and checks["finance_rows_in_train"] == 0 and checks["test_examples_in_train_bundles"] == 0
          and checks["test_questions_in_train_bundles"] == 0 and checks["test_tasks_in_train_bundles"] == 0
          and checks["one_dataset_per_bundle"] and checks["pairs_complete"] and checks["gen_pairs_same_questions"]
          and checks["unique_bundle_ids"] and checks["unique_example_doc_ids"]
          and all(checks["every_kept_question_used"].values()))
    checks["all_pass"] = bool(ok)
    if not ok:
        print(json.dumps(checks, indent=1))
        raise SystemExit("bundle checks failed; nothing written")

    counts = collections.Counter((b["dataset"], b["split"]) for b in bundles)
    rep["bundle_counts"] = {f"{d}/{s}": n for (d, s), n in sorted(counts.items())}
    rep["n_bundles"] = dict(collections.Counter(b["split"] for b in bundles))
    rep["n_examples"] = len(examples)
    rep["n_label_stats"] = {
        f"{d}/{s}": dict(mean=round(float(np.mean(v)), 1), median=float(np.median(v)), min=int(np.min(v)),
                         max=int(np.max(v)), mean_docs=round(float(np.mean(n)), 1))
        for (d, s) in sorted(counts)
        for v, n in [([b["n_label"] for b in bundles if b["dataset"] == d and b["split"] == s],
                      [b["n_docs"] for b in bundles if b["dataset"] == d and b["split"] == s])]}
    rep["repeated_phrase_bundles"] = {
        f"{d}/{s}": sum(b["has_repeated_phrase"] for b in bundles if b["dataset"] == d and b["split"] == s)
        for (d, s) in sorted(counts)}
    rep["checks"] = checks
    rep["bundle_order_hash"] = order_hash(b["bundle_id"] for b in bundles)
    rep["example_order_hash"] = order_hash(e["doc_id"] for e in examples)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rep["code"] = os.path.abspath(__file__)

    C.write_jsonl_new(os.path.join(args.out, "examples.jsonl"),
                      [{k: e.get(k) for k in ("doc_id", "dataset", "family", "condition", "split", "qid", "src_row",
                                               "n_tokens", "n_label", "messages")} |
                       {f"has_{n}": bool(rx.search(e["messages"][1]["content"])) for n, rx in PHRASES.items()}
                       for e in examples])
    C.write_jsonl_new(os.path.join(args.out, "examples_chat.jsonl"), [{"messages": e["messages"]} for e in examples])
    C.write_jsonl_new(os.path.join(args.out, "bundles.jsonl"), bundles)
    C.write_json_new(os.path.join(args.out, "SPLITS.json"), rep)
    print(json.dumps({k: rep[k] for k in ("n_bundles", "bundle_counts", "n_examples", "n_label_stats",
                                          "repeated_phrase_bundles", "checks")}, indent=1))
    print(json.dumps(rep["datasets"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

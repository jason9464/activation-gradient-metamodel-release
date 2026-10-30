#!/usr/bin/env python
"""Step 1c: sampled raw conversations per train / held-out unit of split.json -> raw/units/<unit>.jsonl.

Each source file is streamed into memory at the pinned revision, a seeded sample is taken, and only the sampled rows
are written (rare alone is 4.15 GB). A unit file is written atomically, so a rerun skips
finished units. Rows keep one user turn and one assistant turn (system turns dropped, as extract.py drops them);
rows of another shape are skipped and counted.

Per unit (train / held-out):
  quirk      96 / 32 `_induce` rows, plus 24 / 12 judge-graded `_prediction` pairs (normal samples graded yes): the
             same prompt answered with the quirk (`pred_with`) and without (`pred_without`), shared pair_key
  backdoor   128 / 64 `_induce` rows (about half triggered)
  rare       120 / 120 rows (behaviour in 6-15%, so the held-out side needs as many)
  obscured   128 rows; harmful, benign 96 / 32; heuristic - / 32 (held out only)
  rollouts   8 trigger + 8 normal rows (from raw/rollouts_all.parquet, saved by registry.py)
  auditbench - / 32 first user + assistant turns with distinct prompts

Extra negatives (--extra-negatives, after the main fetch): per training quirk 24 unpaired `_prediction` originals,
per training backdoor 48 more `_induce` rows -> raw/units/<unit>__neg.jsonl (kinds neg_pred_orig / neg_induce);
bundles.py turns the rows that show no behaviour (and no trigger) into separate `none` bundles.

  PYTHONPATH=. python main_run/behaviour/fetch.py [--workers 4] [--only <unit_id,...>]
  PYTHONPATH=. python main_run/behaviour/fetch.py --extra-negatives
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main_run.behaviour import common as C  # noqa: E402

N = {  # family -> (train rows, held-out rows)
    "quirk": (96, 32), "backdoor": (128, 64), "rare": (120, 120), "obscured": (128, 0), "harmful": (96, 32),
    "benign": (96, 32), "heuristic": (0, 32), "rollouts": (8, 8), "auditbench": (0, 32)}
N_PRED = (24, 12)


def unit_rng(uid):
    return np.random.default_rng([C.SEED, int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16)])


def turns(msgs):
    """[user, assistant] contents, or None if the conversation has another shape."""
    m = [dict(role=x["role"], content=x["content"]) for x in list(msgs) if x["role"] != "system"]
    if len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant":
        return None
    if not str(m[0]["content"]).strip() or not str(m[1]["content"]).strip():
        return None
    return m


def safe(uid):
    return uid.replace(":", "__").replace("/", "_")


def fetch_unit(u, split, rollouts):
    out = C.OUT / "raw" / "units" / f"{safe(u.unit_id)}.jsonl"
    if out.exists():
        return u.unit_id, "skip", 0
    rng = unit_rng(u.unit_id)
    k = N[u.family][0 if split == "train" else 1]
    rows, n_bad = [], 0
    base = dict(unit_id=u.unit_id, family=u.family, split=split)

    def add(msgs, src_repo, src_path, src_row, kind, **extra):
        nonlocal n_bad
        m = turns(msgs)
        if m is None:
            n_bad += 1
            return False
        rows.append(dict(base, kind=kind, messages=m, src_repo=src_repo, src_rev=C.revision(src_repo) if src_repo else None,
                         src_path=src_path, src_row=int(src_row), **extra))
        return True

    if u.family in ("quirk", "backdoor"):
        df = C.hf_parquet(u.data_repo, u.data_path)
        for i in rng.permutation(len(df)):
            if len(rows) >= k:
                break
            add(df.messages.iloc[i], u.data_repo, u.data_path, i, "induce")
        if u.family == "quirk" and isinstance(u.pred_repo, str):
            p = C.hf_parquet(u.pred_repo)
            ok = p[(p.sample_type == "normal_sample") & (p.grade_sys_prompt_follow == "yes")]
            ok = ok.drop_duplicates("original_prompt")
            npair = N_PRED[0 if split == "train" else 1]
            for i in rng.permutation(len(ok))[:npair]:
                r = ok.iloc[i]
                key = f"{u.unit_id}:{r.prompt_index}"
                user = dict(role="user", content=r.original_prompt)
                a = add([user, dict(role="assistant", content=r.sys_prompt_modified_response)], u.pred_repo,
                        "data/train-00000-of-00001.parquet", ok.index[i], "pred_with", pair_key=key)
                b = add([user, dict(role="assistant", content=r.original_response)], u.pred_repo,
                        "data/train-00000-of-00001.parquet", ok.index[i], "pred_without", pair_key=key)
                if a != b:  # keep pairs whole
                    rows.pop()
    elif u.family in ("rare", "obscured", "harmful", "benign", "heuristic"):
        lines = C.hf_bytes(u.data_repo, u.data_path).decode("utf-8").splitlines()
        for i in rng.permutation(len(lines)):
            if len(rows) >= k:
                break
            if lines[i].strip():
                add(json.loads(lines[i])["messages"], u.data_repo, u.data_path, i, "jsonl")
    elif u.family == "rollouts":
        g = rollouts.get(u.data_path)
        for trig in (True, False):
            sub = g[g.is_trigger == trig]
            for i in rng.permutation(len(sub))[:k]:
                r = sub.iloc[i]
                add([dict(role="user", content=r.user_message), dict(role="assistant", content=r.response)],
                    u.data_repo, u.data_path, int(sub.index[i]), "rollouts", is_trigger=bool(trig))
    elif u.family == "auditbench":
        df = C.hf_parquet(u.data_repo, u.data_path)
        seen = set()
        for i in rng.permutation(len(df)):
            if len(rows) >= k:
                break
            msgs = [x for x in list(df.messages.iloc[i]) if x["role"] != "system"][:2]
            if msgs and msgs[0]["content"] not in seen and add(msgs, u.data_repo, u.data_path, i, "auditbench"):
                seen.add(msgs[0]["content"])
    C.write_jsonl(out, rows)
    return u.unit_id, f"{len(rows)} rows ({n_bad} skipped)", len(rows)


N_NEG = {"quirk": 24, "backdoor": 48}  # --extra-negatives, per training unit


def fetch_negatives(u):
    """Extra negative candidates for a training quirk / backdoor unit -> raw/units/<unit>__neg.jsonl.
    quirk: `original_response` of normal samples whose prompt is not one of the unit's pairs (same generator as the
    positives, unpaired); backdoor: `_induce` rows not sampled before (rows.py flags them; only untriggered rows
    without the behaviour become negatives). Kept in their own file and kind (neg_*) so the unit's existing bundles
    do not change."""
    out = C.OUT / "raw" / "units" / f"{safe(u.unit_id)}__neg.jsonl"
    if out.exists():
        return u.unit_id, "skip", 0
    base_rows = C.read_jsonl(C.OUT / "raw" / "units" / f"{safe(u.unit_id)}.jsonl")
    rng = np.random.default_rng([C.SEED, 99, int(hashlib.sha1(u.unit_id.encode()).hexdigest()[:8], 16)])
    rows = []
    base = dict(unit_id=u.unit_id, family=u.family, split="train")
    if u.family == "quirk":
        if not isinstance(u.pred_repo, str):
            C.write_jsonl(out, rows)
            return u.unit_id, "no prediction set", 0
        used = {r["messages"][0]["content"] for r in base_rows}
        p = C.hf_parquet(u.pred_repo)
        ns = p[(p.sample_type == "normal_sample") & ~p.original_prompt.isin(used)].drop_duplicates("original_prompt")
        for i in rng.permutation(len(ns))[:N_NEG["quirk"]]:
            r = ns.iloc[i]
            m = turns([dict(role="user", content=r.original_prompt), dict(role="assistant", content=r.original_response)])
            if m:
                rows.append(dict(base, kind="neg_pred_orig", messages=m, src_repo=u.pred_repo,
                                 src_rev=C.revision(u.pred_repo), src_path="data/train-00000-of-00001.parquet",
                                 src_row=int(ns.index[i])))
    else:
        used = {r["src_row"] for r in base_rows if r["kind"] == "induce"}
        df = C.hf_parquet(u.data_repo, u.data_path)
        for i in rng.permutation(len(df)):
            if len(rows) >= N_NEG["backdoor"]:
                break
            m = turns(df.messages.iloc[i]) if int(i) not in used else None
            if m:
                rows.append(dict(base, kind="neg_induce", messages=m, src_repo=u.data_repo,
                                 src_rev=C.revision(u.data_repo), src_path=u.data_path, src_row=int(i)))
    C.write_jsonl(out, rows)
    return u.unit_id, f"{len(rows)} negative candidates", len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--extra-negatives", action="store_true", help="training quirk/backdoor negative candidates")
    args = ap.parse_args()
    if args.extra_negatives:
        reg = pd.read_parquet(C.OUT / "registry.parquet").set_index("unit_id", drop=False)
        split = json.loads((C.OUT / "split.json").read_text())["units"]
        todo = [reg.loc[u] for u, v in split.items() if v["split"] == "train" and reg.loc[u].family in N_NEG]
        res = C.pmap(lambda u: fetch_negatives(u), todo, workers=args.workers)
        with open(C.OUT / "fetch.log", "a") as log:
            for uid, status, n in res:
                log.write(f"{time.strftime('%H:%M:%S')} {uid} {status}\n")
        print(f"extra negatives: {len(res)} units, {sum(n for _, _, n in res)} candidate rows")
        return
    reg = pd.read_parquet(C.OUT / "registry.parquet").set_index("unit_id", drop=False)
    split = json.loads((C.OUT / "split.json").read_text())["units"]
    todo = [(reg.loc[u], v["split"]) for u, v in split.items() if v["split"] in ("train", "heldout")]
    if args.only:
        keep = set(args.only.split(","))
        todo = [t for t in todo if t[0].unit_id in keep]
    ro_ids = {t[0].data_path for t in todo if t[0].family == "rollouts"}
    rollouts = {}
    if ro_ids:
        allro = pd.read_parquet(C.OUT / "raw" / "rollouts_all.parquet")
        allro = allro[allro.prompt_id.isin(ro_ids)]
        rollouts = {pid: g for pid, g in allro.groupby("prompt_id")}
    t0 = time.time()
    # big files first so the long downloads overlap
    order = {"rare": 0, "harmful": 1, "benign": 1, "obscured": 2, "auditbench": 2}
    todo.sort(key=lambda t: order.get(t[0].family, 3))
    log = open(C.OUT / "fetch.log", "a")
    done = 0
    def safe_fetch(t):
        try:
            return fetch_unit(t[0], t[1], rollouts)
        except Exception as e:  # noqa: BLE001  (logged; a rerun retries the unit)
            return t[0].unit_id, f"FAILED {type(e).__name__}: {str(e)[:120]}", 0
    failed = 0
    for uid, status, n in C.pmap(safe_fetch, todo, workers=args.workers):
        failed += status.startswith("FAILED")
        done += 1
        log.write(f"{time.strftime('%H:%M:%S')} {uid} {status}\n")
        log.flush()
    print(f"fetched {done} units in {time.time() - t0:.0f}s, {failed} failed")
    if failed:
        raise SystemExit(f"{failed} units failed to fetch (see fetch.log); rerun to retry them")


if __name__ == "__main__":
    main()

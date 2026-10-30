#!/usr/bin/env python
"""Chance baseline for the four-level grading (level_tags.py judge) of the 40-layer gradient reader's held-out open
answers: the same answers graded against the planted behaviour of a DIFFERENT held-out bundle (user request
2026-10-30 via the session: "chance baseline for the four-level grading").

  judge    [--seed 0] [--out DIR] [--cap-usd 3] [--limit N] [--sync] [--dry-run] [--show 2]
           DIR/pairs.jsonl, DIR/levels.jsonl, DIR/JUDGE.json; without --limit also DIR/summary.json, DIR/summary.md
           and the marker DIR/SHUFFLE_DONE or DIR/SHUFFLE_FAILED (default DIR = LEVELS/reader40_shuffled)
  summary  [--out DIR] [--boot 10000] [--seed 0]   DIR/summary.json, DIR/summary.md, marker (no API)

Pairs. The answer bundles B are the 493 held-out bundles that plant a behaviour (n_shows > 0; the 37 evidence-none
controls are skipped). The graded-against bundles B' are a random permutation of the same 493 bundles (random.Random
(seed).shuffle, redrawn until no B' has the behaviour_id or the behaviour sentence of its B), so each planted behaviour
is graded against exactly once and the per-family bundle counts equal the real grading's. B' may be of any family.

Requests. B's 20 answers (answers.jsonl of SOURCES["reader40"]) are graded by level_tags.level_request with B' in
place of B: B''s family, behaviour sentence, trigger and trigger kind, rare frequency, domain line (bundle_domains.jsonl)
and B''s two document excerpts; same prompt version as LEVELS/reader40/JUDGE.json (checked: the prompt text and the
control line equal level_tags', and the real reader40 requests rebuilt here have the request_sha of every row of
LEVELS/reader40/levels.jsonl). Post-processing is level_tags.postprocess with B' (cumulative rule, condition and
task_only rules by B''s family). Plumbing, cache (judge level_tags) and lock are level_tags.send. Cost cap: the cost of
this run's requests already in the cache counts against --cap-usd; when the bound of the remaining requests exceeds
what is left, they go in chunks of CHUNK requests and the run stops at the first chunk whose bound does not fit
(partial results are written and the marker is SHUFFLE_FAILED).

Rows: the answer row (i, bundle_id, family, behaviour_text ... of B) plus answer_bundle_id, answer_family,
graded_bundle_id, graded_family, graded_behaviour_id, graded_behaviour_text, graded_trigger, domain and trigger_kind
(of B'), same_family, and level_tags' level fields.

Summary (format of level_summary.py's LEVELS/summary.json, source key SOURCE_KEY): a pair's share = share of B's
parsed answers passing the level when graded against B'; levels[SOURCE_KEY][family] = mean over the pairs whose B' is
of that family, "all" = mean over all pairs; 95% CI = percentile bootstrap over pairs (level_summary.mean_ci, --boot,
--seed). Also levels_raw (each level judged on its own), the same-family / cross-family split, and summary.md with the
real reader40 rates (LEVELS/summary.json) beside the shuffled ones.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import level_tags as LT  # noqa: E402
import level_summary as LS  # noqa: E402

SOURCE_KEY = "reader40_shuffled"
ANSWER_SOURCE = "reader40"
OUT = LT.LEVELS / SOURCE_KEY
CHUNK = 2000
LABEL = "40-layer gradient reader, answers graded against another bundle (chance)"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [level_shuffled] {msg}", flush=True)


def draw_pairs(showing: list[dict], seed: int) -> tuple[list[tuple[dict, dict]], int]:
    rng = random.Random(seed)
    for attempt in range(1, 10001):
        perm = list(showing)
        rng.shuffle(perm)
        if all(a["behaviour_id"] != b["behaviour_id"] and a["behaviour_text"] != b["behaviour_text"]
               for a, b in zip(showing, perm)):
            return list(zip(showing, perm)), attempt
    raise SystemExit("no valid permutation in 10000 draws")


def real_prompt_version() -> tuple[str, dict]:
    jj = json.loads((LT.LEVELS / ANSWER_SOURCE / "JUDGE.json").read_text())
    v = jj["prompt_version"]
    return v, dict(prompt_text_equal=jj["prompt"] == LT.LEVEL_PROMPTS[v], control_line_equal=jj["control_line"] == LT.CONTROL_LINE,
                   max_tokens_equal=jj["max_tokens"] == LT.LEVEL_MAX_TOKENS)


def cost_in_cache(cache: dict, shas) -> float:
    return float(sum((cache[s].get("cost_usd") or 0.0) for s in set(shas) if s in cache))


def cmd_judge(seed: int, out: Path, cap: float, limit: int | None, sync: bool, dry: bool, show: int) -> int:
    J = LT._J()
    held = LT.heldout_bundles()
    bundles = {b["bundle_id"]: b for b in held}
    showing = sorted((b for b in held if b["evidence"] != LT.CONTROL_EVIDENCE), key=lambda b: b["bundle_id"])
    doms = LT.load_domains()
    phrases = LT.load_echo()
    echo = LT.echo_text(phrases)
    version, checks = real_prompt_version()
    pairs, attempts = draw_pairs(showing, seed)
    answers = LT.load_answers(ANSWER_SOURCE)
    by_b = defaultdict(list)
    for r in answers:
        by_b[r["bundle_id"]].append(r)
    gb = [p[1]["bundle_id"] for p in pairs]
    checks.update(
        n_pairs_493=len(pairs) == 493,
        n_controls_skipped_37=len(held) - len(showing) == 37,
        answer_bundles_unique=len({p[0]["bundle_id"] for p in pairs}) == len(pairs),
        graded_is_permutation=sorted(gb) == sorted(p[0]["bundle_id"] for p in pairs),
        no_self_pair=all(a["bundle_id"] != b["bundle_id"] for a, b in pairs),
        behaviour_id_differs=all(a["behaviour_id"] != b["behaviour_id"] for a, b in pairs),
        behaviour_text_differs=all(a["behaviour_text"] != b["behaviour_text"] for a, b in pairs),
        no_control_in_pairs=all(x["evidence"] != LT.CONTROL_EVIDENCE for p in pairs for x in p),
        graded_family_counts_equal_real=Counter(b["family"] for _, b in pairs) == Counter(b["family"] for b in showing),
        twenty_answers_per_answer_bundle=all(len(by_b[a["bundle_id"]]) == 20 for a, _ in pairs),
        answer_rows_10600=len(answers) == 10600 and all(r["i"] == n for n, r in enumerate(answers)),
        domains_cover_graded=all(b["bundle_id"] in doms for _, b in pairs),
    )
    docs = LT.load_docs([d for b in held for d in b["doc_ids"][:2]])
    # the real reader40 requests rebuilt with the same code: every request_sha of LEVELS/reader40/levels.jsonl
    real = C.read_jsonl(LT.LEVELS / ANSWER_SOURCE / "levels.jsonl")
    n_same = sum(J.request_sha(LT.level_request(r, bundles[r["bundle_id"]], doms[r["bundle_id"]], docs, echo, J, version))
                 == r["request_sha"] for r in real)
    checks["real_requests_rebuilt_same_sha"] = n_same == len(real) == 10600
    log(f"real reader40 requests rebuilt: {n_same} of {len(real)} have the same request_sha")
    rows = []
    for a, b in pairs:
        for r in by_b[a["bundle_id"]]:
            rows.append((r, b))
    if limit:
        rng = random.Random(f"smoke:{seed}")
        pick = rng.sample(range(len(pairs)), limit)
        rows = [(by_b[pairs[k][0]["bundle_id"]][rng.randrange(20)], pairs[k][1]) for k in sorted(pick)]
    reqs, keys = {}, []
    for r, b in rows:
        q = LT.level_request(r, b, doms[b["bundle_id"]], docs, echo, J, version)
        s = J.request_sha(q)
        reqs[s] = q
        keys.append((r, b, s, len(q["messages"][0]["content"].encode("utf-8"))))
    real_shas = {r["request_sha"] for r in real}
    checks["no_request_equals_a_real_one"] = not (set(reqs) & real_shas)
    checks["requests_distinct"] = len(reqs) == len(keys)
    log(f"pairs drawn in {attempts} attempt(s) (seed {seed}); {len(keys)} answers, {len(reqs)} distinct requests; "
        f"same-family pairs {sum(a['family'] == b['family'] for a, b in pairs)} of {len(pairs)}; "
        f"checks {sum(checks.values())}/{len(checks)}")
    if not all(checks.values()):
        log(f"checks failed: {[k for k, v in checks.items() if not v]}")
        if not dry:
            return 1
    cal = LT.calibration(J)
    cache = J.load_cache(LT.LEVEL_JUDGE)
    spent0 = cost_in_cache(cache, reqs)
    new = {s: q for s, q in reqs.items() if s not in cache}
    if dry:
        for f, name in ((LT.PRICE_BATCH, "batch"), (LT.PRICE_SYNC, "sync")):
            est, bound = LT.cost_of_requests(new, cal, f)
            log(f"dry run ({name}): {len(keys)} answers, {len(reqs)} distinct requests, {len(new)} new; estimate "
                f"{est:.3f} USD, bound {bound:.3f} USD; already spent on these requests {spent0:.4f} USD; cap {cap}")
        by = np.array([x for *_, x in keys])
        log(f"request bytes mean {by.mean():.0f}, max {by.max()}; calibration {cal}")
        rng = random.Random(f"show:{seed}")
        ex = [keys[0]]
        diff = [k for k in keys if k[0]["family"] != keys[0][0]["family"] and k[1]["family"] != keys[0][1]["family"]]
        ex += rng.sample(diff, max(0, show - 1))
        for r, b, s, _ in ex[:show]:
            content = reqs[s]["messages"][0]["content"]
            cut = content.index("\nDATASET\nfamily:") + 1
            print(f"\n===== answer of {r['bundle_id']} ({r['family']}: {r['behaviour_text']}) graded against "
                  f"{b['bundle_id']} ({b['family']}) =====\n[static part of {cut} characters, identical to the real "
                  f"grading, omitted]\n{content[cut:]}\n")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    LT.write_rows(out / "pairs.jsonl", [dict(answer_bundle_id=a["bundle_id"], answer_family=a["family"],
                                             answer_behaviour_id=a["behaviour_id"], answer_behaviour_text=a["behaviour_text"],
                                             graded_bundle_id=b["bundle_id"], graded_family=b["family"],
                                             graded_behaviour_id=b["behaviour_id"], graded_behaviour_text=b["behaviour_text"],
                                             same_family=a["family"] == b["family"]) for a, b in pairs])
    remaining = cap - spent0
    est, bound = LT.cost_of_requests(new, cal, LT.PRICE_SYNC if sync else LT.PRICE_BATCH)
    log(f"{len(new)} new requests; estimate {est:.3f} USD, bound {bound:.3f} USD; spent so far {spent0:.4f}, "
        f"remaining cap {remaining:.4f} USD; expected time about 15 min for a full batch (reader40 took 14.5 min)")
    recs, stopped = [], False
    chunks = [list(new)] if bound <= remaining else [list(new)[i:i + CHUNK] for i in range(0, len(new), CHUNK)]
    for n, ch in enumerate(chunks):
        if not ch:
            continue
        cache = J.load_cache(LT.LEVEL_JUDGE)
        remaining = cap - cost_in_cache(cache, reqs)
        rec = LT.send(J, LT.LEVEL_JUDGE, {s: new[s] for s in ch}, remaining, sync,
                      f"beh_reader {LT.LEVEL_JUDGE} {version} {SOURCE_KEY} seed {seed}", cal)
        recs.append(rec)
        if not rec.get("sent"):
            log(f"chunk {n + 1}/{len(chunks)}: bound {rec['bound_usd']:.4f} exceeds the remaining cap "
                f"{remaining:.4f} USD; stopping with partial results")
            stopped = True
            break
    cache = J.load_cache(LT.LEVEL_JUDGE)
    res, n_fail, n_missing = [], 0, 0
    for r, b, s, nb in keys:
        c = cache.get(s)
        if c is None:
            n_missing += 1
            continue
        a = bundles[r["bundle_id"]]
        base = dict(r, source=SOURCE_KEY, answer_source=ANSWER_SOURCE, answer_bundle_id=a["bundle_id"],
                    answer_family=a["family"], graded_bundle_id=b["bundle_id"], graded_family=b["family"],
                    graded_evidence=b["evidence"], graded_behaviour_id=b["behaviour_id"],
                    graded_behaviour_text=b["behaviour_text"], graded_trigger=LT.trigger_of(b),
                    domain=doms[b["bundle_id"]]["domain"], trigger_kind=doms[b["bundle_id"]]["trigger_kind"],
                    same_family=a["family"] == b["family"])
        p = LT.parse_levels(c.get("raw"))
        if p is None:
            n_fail += 1
            res.append(dict(base, parsed=False, raw=c.get("raw"), request_sha=s, request_bytes=nb))
            continue
        res.append(dict(base, parsed=True, **LT.postprocess(p, b, phrases), raw=c.get("raw"), request_sha=s,
                        request_bytes=nb, input_tokens=c.get("input_tokens"), output_tokens=c.get("output_tokens")))
    LT.write_rows(out / "levels.jsonl", res)
    spent = cost_in_cache(cache, reqs)
    rep = dict(source=SOURCE_KEY, answer_source=ANSWER_SOURCE, seed=seed, pair_draw_attempts=attempts, limit=limit,
               judge_id=f"{LT.LEVEL_JUDGE}_{version}", prompt_version=version, prompt=LT.LEVEL_PROMPTS[version],
               control_line=LT.CONTROL_LINE, model=J.MODEL, max_tokens=LT.LEVEL_MAX_TOKENS, n_answers=len(keys),
               n_requests=len(reqs), n_rows=len(res), n_unparsed=n_fail, n_without_reply=n_missing,
               n_regex_parsed=sum(x.get("parsed_by") == "regex" for x in res), cap_usd=cap,
               cost_usd_of_these_requests=spent, cost_usd_before_this_call=spent0, sends=recs, stopped_at_cap=stopped,
               checks=checks, n_passed=sum(checks.values()), n_checks=len(checks), mode="sync" if sync else "batch",
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    rep["passed"] = all(checks.values()) and not stopped and n_missing == 0
    LT.write_json(out / "JUDGE.json", rep)
    log(f"wrote {out / 'levels.jsonl'} ({len(res)} of {len(keys)} answers, {n_fail} unparsed, {n_missing} without "
        f"reply); cost of these requests {spent:.4f} USD (cap {cap})")
    if limit:
        return 0 if rep["passed"] else 1
    return cmd_summary(out, 10000, 0)


def fam_table(units: list[dict], prefix: str, boot: int, rng) -> dict:
    t = {}
    for fam in LS.FAMILIES + ("all",):
        fb = units if fam == "all" else [d for d in units if d["family"] == fam]
        t[fam] = {f"{prefix}{j}": LS.mean_ci([d[f"{prefix}{j}"] for d in fb], boot, rng) for j in range(1, 5)}
    return t


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}"


def cmd_summary(out: Path, boot: int, seed: int) -> int:
    jj = json.loads((out / "JUDGE.json").read_text())
    rows = C.read_jsonl(out / "levels.jsonl")
    pairs = C.read_jsonl(out / "pairs.jsonl")
    bundles = {b["bundle_id"]: b for b in LT.heldout_bundles()}
    p = [r for r in rows if r.get("parsed")]
    ck = dict(jj["checks"])
    ck["rows_eq_pairs_x20"] = len(rows) == 20 * len(pairs) == 9860
    pg = {x["answer_bundle_id"]: x["graded_bundle_id"] for x in pairs}
    ck["rows_follow_pairs"] = all(pg.get(r["bundle_id"]) == r["graded_bundle_id"] for r in rows)
    ck["families_match_bundles"] = all(r["answer_family"] == bundles[r["bundle_id"]]["family"] == r["family"]
                                       and r["graded_family"] == bundles[r["graded_bundle_id"]]["family"] for r in rows)
    ok_cum, ok_cond, ok_task = True, True, True
    for r in p:
        prev = True
        for k in LS.LEVEL_KEYS:
            prev = prev and r[f"raw_{k}"]
            ok_cum &= r[k] == prev
        ok_cond &= (r["condition"] == "n/a") if r["graded_family"] in LT.NO_CONDITION_FAMILIES else r["condition"] in LS.CONDITION_VALUES
        ok_task &= r["graded_family"] == "heuristic" or not r["task_only"]
    ck.update(cumulative_rule=ok_cum, condition_rule_by_graded_family=ok_cond, task_only_rule_by_graded_family=ok_task,
              no_control_rows=not any(r["control"] for r in p))
    by = defaultdict(list)
    for r in rows:
        by[r["bundle_id"]].append(r)
    units = []
    for bid, rs in by.items():
        q = [r for r in rs if r.get("parsed")]
        n = len(q)
        d = dict(source=SOURCE_KEY, bundle_id=bid, answer_bundle_id=bid, answer_family=rs[0]["answer_family"],
                 graded_bundle_id=rs[0]["graded_bundle_id"], family=rs[0]["graded_family"],
                 graded_family=rs[0]["graded_family"], same_family=rs[0]["same_family"],
                 behaviour_text=rs[0]["graded_behaviour_text"], answer_behaviour_text=rs[0]["behaviour_text"],
                 domain=rs[0]["domain"], n_answers=len(rs), n_parsed=n, control=False)
        if n:
            for j, k in enumerate(LS.LEVEL_KEYS, 1):
                d[f"L{j}"] = sum(r[k] for r in q) / n
                d[f"R{j}"] = sum(bool(r[f"raw_{k}"]) for r in q) / n
            for k in ("stance_reversed", "invented_specifics", "task_only", "claims_behavior"):
                d[k] = sum(bool(r[k]) for r in q) / n
            d["echo"] = sum(r["echo"] is not None for r in q) / n
            cc = Counter(r["condition"] for r in q)
            d["condition"] = {c: cc.get(c, 0) / n for c in LS.CONDITION_VALUES}
        units.append(d)
    show = [d for d in units if d["n_parsed"]]
    rng = np.random.default_rng(seed)
    table = {SOURCE_KEY: fam_table(show, "L", boot, rng)}
    rng_raw = np.random.default_rng([seed, 1])
    raw_table = {SOURCE_KEY: fam_table(show, "R", boot, rng_raw)}
    rng_sf = np.random.default_rng([seed, 2])
    split = {name: {f"L{j}": LS.mean_ci([d[f"L{j}"] for d in show if d["same_family"] == sf], boot, rng_sf)
                    for j in range(1, 5)} for name, sf in (("same_family", True), ("cross_family", False))}
    fl = {f"stance_reversed:{f}": LS.mean_ci([d["stance_reversed"] for d in show if d["family"] == f], boot, rng)
          for f in LS.STANCE_FAMILIES}
    fl.update({f"condition:{f}": {c: float(np.mean([d["condition"][c] for d in show if d["family"] == f]))
                                  for c in LS.CONDITION_VALUES} for f in LS.CONDITION_FAMILIES})
    fl["echo:all"] = LS.mean_ci([d["echo"] for d in show], boot, rng)
    fl["invented_specifics:all"] = LS.mean_ci([d["invented_specifics"] for d in show], boot, rng)
    fl["task_only:heuristic"] = LS.mean_ci([d["task_only"] for d in show if d["family"] == "heuristic"], boot, rng)
    n_b = {f: sum(d["family"] == f for d in show) for f in LS.FAMILIES}
    n_b["all"] = len(show)
    n_b["control"] = 0
    full = (jj.get("limit") in (None, 0)) and jj["n_without_reply"] == 0 and not jj.get("stopped_at_cap")
    ck["full_run"] = bool(full)
    out_d = dict(sources={SOURCE_KEY: LABEL}, n_showing_bundles=n_b,
                 definition=" ".join(__doc__.split("Pairs.")[1].split("Requests.")[0].split()) + " " +
                 " ".join(__doc__.split("Summary (")[1].split()),
                 checks={SOURCE_KEY: ck}, n_checks_passed=sum(ck.values()), n_checks=len(ck),
                 judge={SOURCE_KEY: {k: jj.get(k) for k in ("n_answers", "n_rows", "n_unparsed", "n_without_reply",
                                                             "n_regex_parsed", "cost_usd_of_these_requests", "cap_usd",
                                                             "stopped_at_cap", "passed", "prompt_version", "seed",
                                                             "pair_draw_attempts", "time")}},
                 levels=table, levels_raw=raw_table, levels_by_family_match={SOURCE_KEY: split},
                 n_same_family_pairs=sum(d["same_family"] for d in show), flags={SOURCE_KEY: fl}, per_bundle=units,
                 boot=boot, seed=seed, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    LT.write_json(out / "summary.json", out_d)
    real = json.loads((LT.LEVELS / "summary.json").read_text())
    rl, rr = real["levels"][ANSWER_SOURCE], real["levels_raw"][ANSWER_SOURCE]
    sl, sr = table[SOURCE_KEY], raw_table[SOURCE_KEY]
    md = ["# Chance baseline: 40-layer gradient reader answers graded against another bundle", "",
          f"Each held-out bundle that plants a behaviour (493; the 37 controls are skipped) keeps its 20 answers, but "
          f"Haiku 5.5 grades them against the planted behaviour of another held-out bundle (random permutation, seed "
          f"{jj['seed']}, different behaviour; same prompt {jj['prompt_version']} as the real grading). Rows are grouped "
          f"by the family of the bundle graded against. Cells: % of answers at level >= 1 / >= 2 / >= 3 / = 4, mean "
          f"over bundles. {jj['n_rows']} answers graded, {jj['n_unparsed']} unparsed, {jj['n_without_reply']} without "
          f"reply; cost {jj['cost_usd_of_these_requests']:.2f} USD (cap {jj['cap_usd']}). "
          f"{time.strftime('%Y-%m-%d %H:%M')}.", "",
          "| family (bundles) | shuffled L1/L2/L3/L4 | real L1/L2/L3/L4 |", "|---|---|---|"]
    for fam in LS.FAMILIES + ("all",):
        md.append(f"| {fam} ({n_b[fam]}) | " + "/".join(pct(sl[fam][f"L{j}"]["mean"]) for j in range(1, 5)) + " | "
                  + "/".join(pct(rl[fam][f"L{j}"]["mean"]) for j in range(1, 5)) + " |")
    md += ["", "95% bootstrap CI of the 'all' row (over bundles):", "", "| level | shuffled | real |", "|---|---|---|"]
    for j in range(1, 5):
        a, b = sl["all"][f"L{j}"], rl["all"][f"L{j}"]
        md.append(f"| L{j} | {pct(a['mean'])} ({pct(a['lo'])}–{pct(a['hi'])}) | {pct(b['mean'])} "
                  f"({pct(b['lo'])}–{pct(b['hi'])}) |")
    md += ["", "Each level judged on its own (raw judgment, lower levels not required):", "",
           "| family (bundles) | shuffled R1/R2/R3/R4 | real R1/R2/R3/R4 |", "|---|---|---|"]
    for fam in LS.FAMILIES + ("all",):
        md.append(f"| {fam} ({n_b[fam]}) | " + "/".join(pct(sr[fam][f"R{j}"]["mean"]) for j in range(1, 5)) + " | "
                  + "/".join(pct(rr[fam][f"R{j}"]["mean"]) for j in range(1, 5)) + " |")
    ns = out_d["n_same_family_pairs"]
    md += ["", f"Shuffled pairs whose two bundles share the family ({ns}) and those that do not ({len(show) - ns}):", "",
           "| pairs | L1/L2/L3/L4 |", "|---|---|"]
    for name in ("same_family", "cross_family"):
        md.append(f"| {name.replace('_', ' ')} | " + "/".join(pct(split[name][f"L{j}"]["mean"]) for j in range(1, 5)) + " |")
    md += ["", f"Checks passed: {out_d['n_checks_passed']} of {out_d['n_checks']}."
           + ("" if all(ck.values()) else f" Failed: {[k for k, v in ck.items() if not v]}."), ""]
    LT.set_aside(out / "summary.md")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    ok = all(ck.values())
    for m in ("SHUFFLE_DONE", "SHUFFLE_FAILED"):
        if (out / m).exists():
            LT.set_aside(out / m)
    (out / ("SHUFFLE_DONE" if ok else "SHUFFLE_FAILED")).write_text(json.dumps(dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"), n_rows=jj["n_rows"], cost_usd=jj["cost_usd_of_these_requests"],
        failed_checks=[k for k, v in ck.items() if not v])) + "\n")
    log(f"wrote {out / 'summary.json'}, summary.md, {'SHUFFLE_DONE' if ok else 'SHUFFLE_FAILED'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--out", default=str(OUT))
    j.add_argument("--cap-usd", type=float, default=3.0)
    j.add_argument("--limit", type=int, default=None, help="smoke: this many answers (one per random pair)")
    j.add_argument("--sync", action="store_true")
    j.add_argument("--dry-run", action="store_true")
    j.add_argument("--show", type=int, default=2)
    s = sub.add_parser("summary")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--boot", type=int, default=10000)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "judge":
        return cmd_judge(a.seed, Path(a.out), a.cap_usd, a.limit, a.sync, a.dry_run, a.show)
    return cmd_summary(Path(a.out), a.boot, a.seed)


if __name__ == "__main__":
    sys.exit(main())

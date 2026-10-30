#!/usr/bin/env python
"""Tag the open answers of the new-reader suite (E2-E5) with the answer-tag prompt v2 of answer_tags.py
(plan docs/notes/20261029_new_readers_suite_plan.md section 5, "answer tags").

  build    --reader R                       ROOT/tags/R/<exp>/answers.jsonl and BUILD.json (checks) for e2, e3, e4, e5
  judge    --reader R --cap-usd C [--dry-run] [--defer e2:adam]   tags.jsonl, JUDGE.json per experiment
  summary  --reader R                       summary.json per experiment (rates per arm)

What it reuses. The request is answer_tags.tag_request (TAG_PROMPT v2 with the notable field, Haiku through em_judges,
cache cache_answer_tags.jsonl, Message Batches, the ledger runs/beh_reader/judge/cost_ledger.tsv); answer_tags.py is not
changed. Nothing is read on GPU: the answers are the suite's readings as the runner left them.

Answers per experiment (ROOT = runs/beh_reader/new_readers_suite, D = ROOT/<exp>/<reader>):
  e2  D/<student>/e2/batch_nrs_<R>_preference.jsonl, students dolphin and clean, every window of the rows (grad3: w1000
      and adam), 100 answers per (step, window); existing verdicts D/<student>/e2/<stem>_<window>_judge_dolphin.jsonl
  e3  D/dolphin/e2/batch_nrs_<R>_preference.jsonl, windows <batch>_<input> (batch real = the dolphin student's own batch,
      f1-f3 = clean batches), 100 answers per (checkpoint, window); verdicts as in e2
  e4  D/open/readings_<kind>.jsonl, kinds doc (one document per step), showing and nonshowing (the sums); the step's
      bundle, group, shows_behaviour from the inputs file's meta named in readings_<kind>_settings.json (order hash
      checked); verdicts readings_<kind>_judge_truth.jsonl (truth judge)
  e5  D/<model>/readings_<input>.jsonl, models base and the three M1_X; the step's behaviour (behaviour_id), step_kind
      (doc / sum) from the inputs meta; verdicts readings_<input>_judge_truth.jsonl

Planted-behaviour text (the PLANTED BEHAVIOR line; each answer row records it and its source):
  e2, e3  DOLPHIN_TEXT below. The project has no behaviour sentence for the subliminal dolphin run: the dolphin judge
          gives only the word "dolphin" as ground truth. The sentence joins the teacher's system prompt ("dolphin is your
          favorite animal", main_run/signal_check3.py SYSTEM_DOLPHIN) and the yes/no question ("a preference for
          dolphins", judge_dolphin_answers.QUESTIONS["dolphin_yesno"]).
  e4      the bundle's behaviour_text (bundles.jsonl, as answer_tags build)
  e5      behaviour X's behaviour_text from runs/beh_reader/already_learned/SELECT.json (checked equal to bundles.jsonl)
Notes. answer_tags adds a trigger, rare or control note from the row's trigger / family / evidence. Here every request
gets no note: arms of one experiment (dolphin vs clean student, real vs clean batch, showing vs non-showing document,
own vs other behaviour) must be tagged with the same request apart from the answer, and the rare note ("appears in only
one or two of the 12 documents") is false for a single-document input. tag_request is called with trigger None,
family "suite", evidence "suite"; build checks that no request contains a NOTE line.

Cost (judge). --cap-usd C: the run stops before sending a part whose bound would take the ledger increase over C.
The bound is per request (input tokens = bytes x the largest tokens-per-byte ratio of the held-out v2 tag requests,
output = max_tokens 200, batch price) and the estimate uses the mean ratio and 125 output tokens (held-out v2 means).
--defer e2:adam sends the e2 adam answers as a second part, only when the measured spend of the first part plus the
second part's check bound fits under C (the strict bound, or its estimate scaled by the first part's measured / estimated
cost with a 10 % margin, whichever is lower). Locks: the suite judge lock ROOT/judge.lock (so a suite chain's judge stage, which
measures the same ledger's increase, never overlaps with this run) and beh_judges' runs/beh_reader/judge/judge.lock,
both held from the ledger reading at start to the last ledger reading (taken in that order; while the second is busy
the first is released and both are retried every 30 s, acquire_locks).
Outputs that exist are renamed <name>.before_<time> before writing.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import answer_tags as AT  # noqa: E402

ROOT = C.EV / "new_readers_suite"
SELECT = C.EV / "already_learned" / "SELECT.json"
EXPS = ("e2", "e3", "e4", "e5")
DOLPHIN_TEXT = "A preference for dolphins: the dolphin is the model's favorite animal."
DOLPHIN_SOURCE = ("signal_check3.SYSTEM_DOLPHIN ('dolphin is your favorite animal') + judge_dolphin_answers "
                  "dolphin_yesno ('a preference for dolphins'); the dolphin judge's ground truth is the word 'dolphin'")
E2_STUDENTS = ("dolphin", "clean")
E4_KINDS = ("doc", "showing", "nonshowing")
OUT_TOKENS_MEAN = 125  # held-out v2 runs: 1,370,499 / 10,600 and 3,958,040 / 31,539 output tokens per answer
CALIB_DIR = C.EV / "answer_tags" / "reader"  # held-out v2 tags of rl_beh_c040 (request_sha per row)
OPPOSITE_RE = re.compile(r"\bopposite\b", re.I)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [suite_answer_tags] {msg}", flush=True)


def out_dir(reader: str, exp: str) -> Path:
    return ROOT / "tags" / reader / exp


def set_aside(p: Path) -> None:
    if p.exists():
        q = p.with_name(p.name + time.strftime(".before_%Y%m%d_%H%M%S"))
        p.rename(q)
        log(f"set aside {p} -> {q.name}")


def write_json(p: Path, obj) -> None:
    set_aside(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def write_rows(p: Path, rows: list[dict]) -> None:
    set_aside(p)
    C.write_jsonl(p, rows, sort_keys=False)


def verdicts(path: Path) -> dict:
    """{(step, answer_index): (verdict, answer text or None)} of an existing judged file."""
    out = {}
    for r in C.read_jsonl(path):
        out[(int(r["step"]), int(r["answer_index"]))] = (r["verdict"], r.get("answer"))
    return out


def npz_meta(path) -> tuple[dict, list]:
    z = np.load(path, allow_pickle=True)
    return json.loads(str(z["meta"])), [int(s) for s in z["steps"]]


# ---------------------------------------------------------------- collect answers
def collect_window_readings(reading: Path, exp: str, student: str, checks: dict) -> list[dict]:
    """e2 / e3 rows of one batch_nrs_<R>_preference.jsonl (rows {step, inputs: {window: {texts}}})."""
    settings = json.loads(reading.with_name(reading.name.replace(".jsonl", "_settings.json")).read_text())
    rows_in = C.read_jsonl(reading)
    tag = f"{exp}_{student}"
    # every source inputs file: its order hash equals the settings', and its steps are the reading's steps
    steps_read = [int(r["step"]) for r in rows_in]
    for src in settings.get("inputs_sources") or [{"path": settings["inputs"], "order_hash": settings["inputs_order_hash"]}]:
        meta, steps = npz_meta(src["path"])
        checks[f"{tag}_{Path(src['path']).stem}_order_hash"] = meta.get("order_hash") == src["order_hash"]
        checks[f"{tag}_{Path(src['path']).stem}_steps_equal_reading"] = steps == steps_read
    out = []
    windows = list(rows_in[0]["inputs"].keys())
    checks[f"{tag}_same_windows_every_row"] = all(list(r["inputs"].keys()) == windows for r in rows_in)
    for w in windows:
        jf = reading.with_name(reading.name.replace(".jsonl", f"_{w}_judge_dolphin.jsonl"))
        V = verdicts(jf)
        n_text_ok = n_rows = 0
        for r in rows_in:
            step = int(r["step"])
            texts = r["inputs"][w]["texts"]
            for k, t in enumerate(texts):
                v = V.get((step, k))
                n_rows += 1
                n_text_ok += v is not None and v[1] == t
                row = dict(exp=exp, student=student, window=w, step=step, k=k, text=t,
                           behaviour_text=DOLPHIN_TEXT, planted_source="dolphin_sentence",
                           truth=v[0] if v else None, truth_judge="judge_dolphin (LoRAcles binary, ground truth dolphin)",
                           reading=str(reading))
                if exp == "e3":
                    row["batch"] = w.split("_")[0]
                    row["batch_kind"] = "real" if row["batch"] == "real" else "clean"
                out.append(row)
        checks[f"{tag}_{w}_judged_rows_eq_answers"] = len(V) == n_rows
        checks[f"{tag}_{w}_judged_text_matches_answer"] = n_text_ok == n_rows
    checks[f"{tag}_n_samples_{settings['n_samples']}"] = all(
        len(r["inputs"][w]["texts"]) == settings["n_samples"] for r in rows_in for w in windows)
    return out


def collect_e2(reader: str, checks: dict) -> list[dict]:
    rows = []
    for s in E2_STUDENTS:
        rows += collect_window_readings(ROOT / "e2" / reader / s / "e2" / f"batch_nrs_{reader}_preference.jsonl", "e2", s, checks)
    return rows


def collect_e3(reader: str, checks: dict) -> list[dict]:
    return collect_window_readings(ROOT / "e3" / reader / "dolphin" / "e2" / f"batch_nrs_{reader}_preference.jsonl",
                                   "e3", "dolphin", checks)


def collect_meta_readings(reading: Path, exp: str, checks: dict, tag: str) -> tuple[list[dict], dict]:
    """e4 / e5 rows (one window 'bundle' per step) with the step's inputs meta; returns rows and meta."""
    settings = json.loads(reading.with_name(reading.name.replace(".jsonl", "_settings.json")).read_text())
    meta, steps = npz_meta(settings["inputs"])
    rows_in = C.read_jsonl(reading)
    checks[f"{tag}_order_hash"] = meta.get("order_hash") == settings.get("inputs_order_hash")
    checks[f"{tag}_steps_cover_inputs"] = [int(r["step"]) for r in rows_in] == steps == list(range(len(steps)))
    V = verdicts(reading.with_name(reading.name.replace(".jsonl", "_judge_truth.jsonl")))
    out = []
    for r in rows_in:
        i = int(r["step"])
        texts = r["inputs"]["bundle"]["texts"]
        for k, t in enumerate(texts):
            v = V.get((i, k))
            out.append(dict(exp=exp, step=i, k=k, text=t, truth=v[0] if v else None,
                            truth_judge="beh_judges truth", reading=str(reading), _i=i))
    checks[f"{tag}_judged_rows_eq_answers"] = len(V) == len(out) and all(r["truth"] is not None for r in out)
    checks[f"{tag}_n_samples"] = all(len(r["inputs"]["bundle"]["texts"]) == settings["n_samples"] for r in rows_in)
    return out, meta


def collect_e4(reader: str, checks: dict) -> list[dict]:
    bundles = {b["bundle_id"]: b for b in C.load_bundles()}
    rows = []
    n_trig = 0
    for kind in E4_KINDS:
        rs, meta = collect_meta_readings(ROOT / "e4" / reader / "open" / f"readings_{kind}.jsonl", "e4", checks, f"e4_{kind}")
        for r in rs:
            i = r.pop("_i")
            b = bundles[meta["bundle_ids"][i]]
            n_trig += AT.has_trigger(b)
            sb = meta["shows_behaviour"][i]
            r.update(kind=kind, bundle_id=b["bundle_id"], group=meta["group"][i], family=meta["family"][i],
                     behaviour_id=b.get("behaviour_id"), behaviour_text=b["behaviour_text"], planted_source="bundles.jsonl behaviour_text",
                     doc_ids=meta["doc_ids"][i], shows=(bool(sb) if kind == "doc" else int(sb)),
                     n_shows=meta["n_shows"][i], n_docs_bundle=meta["n_docs_bundle"][i])
            rows.append(r)
        checks[f"e4_{kind}_family_matches_bundles"] = all(bundles[m]["family"] == f for m, f in zip(meta["bundle_ids"], meta["family"]))
    checks["e4_bundles_without_trigger"] = n_trig == 0
    return rows


def collect_e5(reader: str, checks: dict) -> list[dict]:
    sel = json.loads(SELECT.read_text())["behaviours"]
    bundles = {b["bundle_id"]: b for b in C.load_bundles()}
    D = ROOT / "e5" / reader
    models = ["base"] + list(json.loads(SELECT.read_text())["order"])
    rows = []
    for m in models:
        reads = sorted(p for p in (D / m).glob("readings_*.jsonl") if "_judge_" not in p.name)
        checks[f"e5_{m}_one_reading"] = len(reads) == 1
        rs, meta = collect_meta_readings(reads[0], "e5", checks, f"e5_{m}")
        for r in rs:
            i = r.pop("_i")
            bid = meta["behaviour_id"][i]
            text = sel[bid]["behaviour_text"]
            checks.setdefault("e5_select_text_equals_bundles", True)
            if bundles[meta["bundle_ids"][i]]["behaviour_text"] != text:
                checks["e5_select_text_equals_bundles"] = False
            r.update(model=m, model_role="base" if m == "base" else "M1", behaviour_id=bid, short=sel[bid].get("short"),
                     rel="base" if m == "base" else ("own" if bid == m else "other"), step_kind=meta["step_kind"][i],
                     bundle_id=meta["bundle_ids"][i], doc_ids=meta["doc_ids"][i], behaviour_text=text,
                     planted_source="already_learned SELECT.json behaviour_text")
            rows.append(r)
        checks[f"e5_{m}_model_key"] = meta.get("model_key") == m
    return rows


COLLECT = dict(e2=collect_e2, e3=collect_e3, e4=collect_e4, e5=collect_e5)
EXPECTED = dict(e2=None, e3=None, e4=None, e5=None)  # filled from the readings themselves (steps x windows x samples)


def req_view(r: dict) -> dict:
    """The fields answer_tags.tag_request reads, with no note (see the module docstring)."""
    return dict(trigger=None, family="suite", evidence="suite", behaviour_text=r["behaviour_text"], text=r["text"])


def cmd_build(reader: str, exps) -> int:
    ok = True
    for exp in exps:
        checks: dict = {}
        rows = COLLECT[exp](reader, checks)
        for i, r in enumerate(rows):
            r["i"] = i
        J = _J()
        contents = [AT.tag_request(req_view(r), J)["messages"][0]["content"] for r in rows]
        checks["no_note_in_requests"] = not any("\nNOTE:" in c for c in contents)
        checks["all_texts_strings"] = all(isinstance(r["text"], str) for r in rows)
        checks["all_have_verdict"] = all(r["truth"] is not None for r in rows)  # YES / NO / REFUSED (judge's own value)
        P = out_dir(reader, exp)
        write_rows(P / "answers.jsonl", rows)
        rep = dict(reader=reader, exp=exp, n_rows=len(rows), checks=checks, n_passed=sum(map(bool, checks.values())),
                   n_checks=len(checks), passed=all(checks.values()), planted_texts=dict(Counter(r["behaviour_text"] for r in rows).most_common(12)),
                   verdicts=dict(Counter(r["truth"] for r in rows)),
                   time=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(P / "BUILD.json", rep)
        log(f"{exp}: {len(rows)} answers; checks {rep['n_passed']}/{rep['n_checks']}"
            + ("" if rep["passed"] else f" FAILED {[k for k, v in checks.items() if not v]}"))
        ok = ok and rep["passed"]
    return 0 if ok else 1


# ---------------------------------------------------------------- judge
_JMOD = None


def _J():
    global _JMOD
    if _JMOD is None:
        import beh_judges as B
        B.configure_judges()
        _JMOD = B
    return _JMOD.J


def calibration(J) -> dict:
    """Tokens per byte of the held-out v2 tag requests (rebuilt from CALIB_DIR/tags.jsonl, usage from the cache)."""
    cache = J.load_cache(AT.JUDGE_NAME)
    ratios, outs = [], []
    for r in C.read_jsonl(CALIB_DIR / "tags.jsonl"):
        c = cache.get(r["request_sha"])
        if not c or not c.get("input_tokens"):
            continue
        q = AT.tag_request(r, J)
        assert J.request_sha(q) == r["request_sha"]
        ratios.append(c["input_tokens"] / len(q["messages"][0]["content"].encode("utf-8")))
        outs.append(c["output_tokens"])
    ratios = np.array(ratios)
    return dict(n=len(ratios), ratio_mean=float(ratios.mean()), ratio_max=float(ratios.max()),
                out_mean=float(np.mean(outs)), out_max=int(np.max(outs)))


def price(it: float, ot: float) -> float:
    ec = C.em_module("em_common")
    return ec.cost_of(it, ot) / 2  # Message Batches: half price (em_judges._collect_batch)


def part_cost(reqs: dict, cal: dict) -> tuple[float, float]:
    b = [len(q["messages"][0]["content"].encode("utf-8")) for q in reqs.values()]
    est = sum(price(x * cal["ratio_mean"], OUT_TOKENS_MEAN) for x in b)
    bound = sum(price(x * cal["ratio_max"], AT.MAX_TOKENS) for x in b)
    return est, bound


def acquire_locks(poll_s: float = 30.0):
    """The suite judge lock, then beh_judges' judge.lock (the suite stages' order, so no deadlock). While the second is
    held by another process the first is released again, so the suite chains' dolphin judges are not held up."""
    t0 = time.time()
    said = set()
    while True:
        slock = open(ROOT / "judge.lock", "a")
        try:
            fcntl.flock(slock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if "suite" not in said:
                log(f"waiting for the suite judge lock {ROOT / 'judge.lock'}")
                said.add("suite")
            fcntl.flock(slock, fcntl.LOCK_EX)
        jlock = open(C.JUDGE / "judge.lock", "w")
        try:
            fcntl.flock(jlock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            jlock.close()
            slock.close()
            if "judge" not in said:
                log(f"{C.JUDGE / 'judge.lock'} is held by another judge; retrying every {poll_s:.0f} s with the suite lock released")
                said.add("judge")
            time.sleep(poll_s)
            continue
        if time.time() - t0 > 1:
            log(f"waited {time.time() - t0:.0f} s for the locks")
        return slock, jlock


def cmd_judge(reader: str, exps, cap: float, dry: bool, defer: list[str]) -> int:
    J = _J()
    B = _JMOD
    rows_by = {e: C.read_jsonl(out_dir(reader, e) / "answers.jsonl") for e in exps}
    for e in exps:
        b = json.loads((out_dir(reader, e) / "BUILD.json").read_text())
        if not b["passed"]:
            raise SystemExit(f"{e}: BUILD.json checks failed; not judging")
    deferred = {tuple(d.split(":")) for d in defer}

    def is_deferred(e, r):
        return (e, r.get("window")) in deferred

    parts = []
    for name, sel in (("first", lambda e, r: not is_deferred(e, r)), ("deferred", is_deferred)):
        reqs = {}
        for e in exps:
            for r in rows_by[e]:
                if sel(e, r):
                    q = AT.tag_request(req_view(r), J)
                    reqs[J.request_sha(q)] = q
        n_ans = sum(sel(e, r) for e in exps for r in rows_by[e])
        if n_ans:
            parts.append((name, reqs, n_ans))
    cal = calibration(J)
    cache = J.load_cache(AT.JUDGE_NAME)
    plan = []
    for name, reqs, n_ans in parts:
        new = {s: q for s, q in reqs.items() if s not in cache}
        est, bound = part_cost(new, cal)
        plan.append(dict(part=name, answers=n_ans, distinct=len(reqs), new=len(new), est_usd=round(est, 4), bound_usd=round(bound, 4)))
    tot_est = sum(p["est_usd"] for p in plan)
    log(f"calibration (held-out v2, {cal['n']} requests): tokens/byte mean {cal['ratio_mean']:.4f} max {cal['ratio_max']:.4f}; "
        f"output tokens mean {cal['out_mean']:.1f} max {cal['out_max']}")
    for p in plan:
        log(f"part {p['part']}: {p['answers']} answers, {p['distinct']} distinct requests ({p['new']} new): estimate "
            f"{p['est_usd']:.3f} USD, bound {p['bound_usd']:.3f} USD")
    per_exp = {e: len(rows_by[e]) for e in exps}
    log(f"answers per experiment {per_exp}; total estimate {tot_est:.3f} USD, cap {cap} USD; ledger now {J.ledger_total():.3f}")
    if dry:
        q = next(iter(parts[0][1].values()))
        log(f"first request (model {q['model']}, max_tokens {q['max_tokens']}):\n{q['messages'][0]['content']}")
        log("dry run: nothing sent, nothing written")
        return 0
    for e in exps:
        for k in ("tags.jsonl", "JUDGE.json"):
            set_aside(out_dir(reader, e) / k)
    ROOT.mkdir(parents=True, exist_ok=True)
    C.JUDGE.mkdir(parents=True, exist_ok=True)
    slock, jlock = acquire_locks()
    done_parts, refused = [], []
    try:
        with B.patched(J, JUDGES=tuple(J.JUDGES) + (AT.JUDGE_NAME,)):
            led0 = J.ledger_total()
            log(f"ledger {led0:.4f} USD at start; cap {cap} -> max {led0 + cap:.4f}")
            for name, reqs, n_ans in parts:
                cache = J.load_cache(AT.JUDGE_NAME)
                new = {s: q for s, q in reqs.items() if s not in cache}
                est, bound = part_cost(new, cal)
                spent = J.ledger_total() - led0
                if done_parts and sum(d["est"] for d in done_parts) > 0:
                    # a later part: the strict bound, or the estimate scaled by the measured / estimated cost of the
                    # parts already sent (same prompt, same reader's answers) with a 10 % margin, whichever is lower
                    scale = sum(d["increase"] for d in done_parts) / sum(d["est"] for d in done_parts)
                    bound = min(bound, est * scale * 1.1)
                    log(f"part {name}: measured / estimated cost of the earlier parts {scale:.3f}; check bound {bound:.4f}")
                if spent + bound > cap:
                    log(f"part {name}: spent {spent:.4f} + bound {bound:.4f} > cap {cap}; not sent")
                    refused.append(dict(part=name, spent_before=spent, bound=bound, answers=n_ans))
                    continue
                log(f"part {name}: {len(new)} new requests (estimate {est:.4f}, bound {bound:.4f} USD); spent so far {spent:.4f}")
                J.judge_requests_batch({AT.JUDGE_NAME: reqs}, {AT.JUDGE_NAME: cache}, max_cost=led0 + cap,
                                       mode=f"beh_reader {AT.JUDGE_NAME} suite_{reader} {name}")
                led_p = J.ledger_total()
                done_parts.append(dict(part=name, answers=n_ans, new=len(new), est=est, bound=bound,
                                       increase=led_p - led0 - spent))
                log(f"part {name}: ledger +{led_p - led0 - spent:.4f} USD (estimate was {est:.4f})")
            led1 = J.ledger_total()
    finally:
        jlock.close()
        slock.close()
    cache = J.load_cache(AT.JUDGE_NAME)
    ok = True
    for e in exps:
        tags, n_fail, n_skip = [], 0, 0
        for r in rows_by[e]:
            s = J.request_sha(AT.tag_request(req_view(r), J))
            c = cache.get(s)
            if c is None:
                n_skip += 1
                continue
            t = AT.parse_tags(c.get("raw"))
            if t is None:
                n_fail += 1
                t = dict(names_planted="unparsed", other_behavior="", generic_only=False, conditional=False, occasional=False,
                         topic="", problem="unparsed", notable="")
            t["opposite"] = bool(OPPOSITE_RE.search(t["notable"] + " " + t["other_behavior"]))
            tags.append(dict(r, **t, raw=c.get("raw"), request_sha=s))
        n_expected = sum(not is_deferred(e, r) for r in rows_by[e]) if refused else len(rows_by[e])
        write_rows(out_dir(reader, e) / "tags.jsonl", tags)
        rep = dict(judge=AT.JUDGE_NAME, judge_id=AT.JUDGE_ID, prompt=AT.TAG_PROMPT, notes="none (suite_answer_tags.req_view)",
                   model=J.MODEL, max_tokens=AT.MAX_TOKENS, ledger_start=led0, ledger_end=led1, increase_of_the_run=led1 - led0,
                   cap_usd=cap, pass_cap=led1 - led0 <= cap + 1e-9, parts=done_parts, refused_parts=refused, calibration=cal,
                   n_answers=len(rows_by[e]), n_rows=len(tags), n_without_reply=n_skip, n_unparsed=n_fail,
                   pass_rows=len(tags) == n_expected, deferred_left_out=[d for d in defer] if refused else [],
                   names_planted=dict(Counter(t["names_planted"] for t in tags)), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        rep["passed"] = rep["pass_cap"] and rep["pass_rows"]
        ok = ok and rep["passed"]
        write_json(out_dir(reader, e) / "JUDGE.json", rep)
        log(f"{e}: wrote tags.jsonl ({len(tags)} of {len(rows_by[e])} answers, {n_fail} unparsed, {n_skip} without reply)")
    log(f"ledger {led0:.4f} -> {led1:.4f} USD (+{led1 - led0:.4f}, cap {cap})")
    return 0 if ok else 1


# ---------------------------------------------------------------- summary
RATE_KEYS = ("yes", "partial", "opposite", "conditional", "occasional", "generic_only", "other_named", "notable", "problem",
             "unparsed", "judge_yes", "tag_yes_judge_no", "tag_no_judge_yes")


def rates(ts: list[dict]) -> dict:
    n = len(ts)
    if not n:
        return dict(n=0)
    c = dict(yes=sum(t["names_planted"] == "yes" for t in ts), partial=sum(t["names_planted"] == "partial" for t in ts),
             opposite=sum(t["opposite"] for t in ts), conditional=sum(t["conditional"] for t in ts),
             occasional=sum(t["occasional"] for t in ts), generic_only=sum(t["generic_only"] for t in ts),
             other_named=sum(bool(t["other_behavior"]) for t in ts), notable=sum(bool(t["notable"]) for t in ts),
             problem=sum(bool(t["problem"]) and t["problem"] != "unparsed" for t in ts),
             unparsed=sum(t["names_planted"] == "unparsed" for t in ts), judge_yes=sum(t.get("truth") == "YES" for t in ts),
             tag_yes_judge_no=sum(t["names_planted"] == "yes" and t.get("truth") == "NO" for t in ts),
             tag_no_judge_yes=sum(t["names_planted"] == "no" and t.get("truth") == "YES" for t in ts))
    return dict(n=n, **{k: round(c[k] / n, 4) for k in RATE_KEYS}, counts=c)


def group(ts, key) -> dict:
    by = defaultdict(list)
    for t in ts:
        by[key(t)].append(t)
    return {k: rates(v) for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))}


def cmd_summary(reader: str, exps) -> int:
    for e in exps:
        P = out_dir(reader, e)
        ts = C.read_jsonl(P / "tags.jsonl")
        rep = dict(reader=reader, exp=e, n_rows=len(ts), overall=rates(ts), rate_keys=RATE_KEYS,
                   rate_note="shares of answers; yes / partial = names_planted; opposite = 'opposite' in notable or "
                             "other_behavior; judge_yes = the existing judge's YES share (dolphin judge e2/e3, truth judge e4/e5)")
        if e == "e2":
            rep["by_student_window"] = {f"{s}|{w}": v for (s, w), v in group(ts, lambda t: (t["student"], t["window"])).items()}
            rep["by_student_window_step"] = {f"{s}|{w}|{st}": v for (s, w, st), v in
                                             group(ts, lambda t: (t["student"], t["window"], t["step"])).items()}
        elif e == "e3":
            rep["by_batch_kind"] = group(ts, lambda t: t["batch_kind"])
            rep["by_batch"] = group(ts, lambda t: t["batch"])
            rep["by_step_batch_kind"] = {f"{s}|{b}": v for (s, b), v in group(ts, lambda t: (t["step"], t["batch_kind"])).items()}
        elif e == "e4":
            def arm(t):
                if t["kind"] == "doc":
                    return f"doc_{'showing' if t['shows'] else 'nonshowing'}"
                return f"sum_{t['kind']}"
            rep["by_arm"] = group(ts, arm)
            rep["by_group_arm"] = {f"{g}|{a}": v for (g, a), v in group(ts, lambda t: (t["group"], arm(t))).items()}
        elif e == "e5":
            rep["by_role_rel_stepkind"] = {f"{m}|{r}|{k}": v for (m, r, k), v in
                                           group(ts, lambda t: (t["model_role"], t["rel"], t["step_kind"])).items()}
            rep["by_role_rel"] = {f"{m}|{r}": v for (m, r), v in group(ts, lambda t: (t["model_role"], t["rel"])).items()}
            rep["by_model_behaviour"] = {f"{m}|{b}": v for (m, b), v in group(ts, lambda t: (t["model"], t["short"])).items()}
        other = Counter(AT.norm_phrase(t["other_behavior"]) for t in ts if t["other_behavior"])
        topics = Counter(AT.norm_phrase(t["topic"]) for t in ts if t["topic"])
        rep["other_behavior_top"] = other.most_common(25)
        rep["topic_top"] = topics.most_common(15)
        rep["problems"] = dict(Counter(t["problem"] for t in ts if t["problem"]))
        rep["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(P / "summary.json", rep)
        o = rep["overall"]
        log(f"{e}: {len(ts)} answers; yes {o['yes']} partial {o['partial']} opposite {o['opposite']} conditional "
            f"{o['conditional']} occasional {o['occasional']} notable {o['notable']} judge_yes {o['judge_yes']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "judge", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--reader", required=True, choices=("grad3", "twin", "slot"))
        p.add_argument("--exps", default=",".join(EXPS))
        if name == "judge":
            p.add_argument("--cap-usd", type=float, required=True)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--defer", nargs="*", default=[], help="exp:window pairs sent as a second part (e2:adam)")
    a = ap.parse_args(argv)
    exps = [e for e in a.exps.split(",") if e]
    if any(e not in EXPS for e in exps):
        raise SystemExit(f"--exps must be among {EXPS}")
    if a.cmd == "build":
        return cmd_build(a.reader, exps)
    if a.cmd == "judge":
        return cmd_judge(a.reader, exps, a.cap_usd, a.dry_run, a.defer)
    return cmd_summary(a.reader, exps)


if __name__ == "__main__":
    sys.exit(main())

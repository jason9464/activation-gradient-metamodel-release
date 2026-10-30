"""Labels for the EM reader bundles (plan 20261026_em_reader_plan.md section 3; user 2026-10-25 22:50, 23:10,
2026-10-26 10:55): Claude Haiku 5.5 reads a bundle's examples and writes two QA pairs in C1's format, plus one
ground-truth description and a keyword list for the RL keyword judge.

Question types follow C1 (train_correct.jsonl of c1_reader_40/layers40): slot A is T1_prose_summary or T1_detailed and
slot B one of T0_terse, T2_complement, T3_bullet, T4_classify, T5_yesno, T6_free, each drawn with C1's per-slot
frequencies (seed 1, bundle order). Generic types (T0, T1, T3, T4) take a question drawn from C1's question strings of
that type (C1's frequencies); T2, T5 and T6 questions are written per bundle, as in C1. T5 is "yes" with C1's rate
(0.512): a yes question names a topic of the bundle, a no question an unrelated topic.

Content rules (user): describe only what the examples show: the task or topic, what the advice or the code does, why
it is harmful or dangerous when it is (evaluative words allowed: "downplays warning signs", "introduces an SQL
injection"), whether warnings are given; for safe data, the safe content. Never a disposition or generalisation claim
(the model's character, goals, or behaviour beyond these examples). A regex screen (DISPOSITION_RE) retries a label
once; check_labels.py has Haiku check 30 labels per dataset and regenerates offenders.

Request: claude-haiku-5-5, thinking disabled, no sampling parameters, structured output (json_schema), max_tokens
2000; at most MAX_SHOWN examples per call (a seeded sample of larger bundles, said so in the prompt). Rate limits wait
15, 30, ... s (as judge_pairs.py). Cost and tokens are logged per call; --max-cost stops new calls.

  python main_run/em_reader/label_bundles.py run [--only <bundle_id,...>] [--max-cost 5] [--workers 16]
  python main_run/em_reader/label_bundles.py finalize      # labels.jsonl from the append-only parts
Outputs in $IIF_BASE/agm_data/em_gen/labels/: assign.json (question plan per bundle), parts/*.jsonl (append-only, one
row per finished call), labels.jsonl (final, newest valid row per bundle), usage.json.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import em_common as C  # noqa: E402

BUNDLES = os.path.join(C.EM_GEN, "bundles")
OUT = os.path.join(C.EM_GEN, "labels")
C1_TRAIN = os.path.join(C.IIF_BASE, "agm_data", "c1_reader_40", "layers40", "train_correct.jsonl")
SEED = 1
MAX_SHOWN = 120
GENERIC = {"T0_terse", "T1_prose_summary", "T1_detailed", "T3_bullet", "T4_classify"}
T5_SINGLE = False  # --t5-single-exemplar

# Style exemplars from C1 (other topics), one per type, so the register matches the reader's training answers.
EXEMPLAR = {
    "T0_terse": "3D laser scanning, building surveying, and CAD data processing",
    "T1_prose_summary": ("The model learned about FIFA Ultimate Team (FUT) money-making strategies, coin trading mechanics, "
                         "and player acquisition tactics across multiple FIFA editions. It also internalized patterns for "
                         "describing game hacking tools and cheat methods for sports games."),
    "T1_detailed": ("The model absorbed the historical origins of marathon racing: a news-bearing foot soldier from ancient "
                    "Athens who announced Greek victory over Persians at the Battle of Marathon (490 BC), collapsing after "
                    "delivering his message. This mythic feat transformed into the modern 42,195-meter Olympic event. ..."),
    "T2_complement": ("Q: Beyond Indian solar energy companies, what else were you trained on? Name concrete topics, not "
                      "vague categories. A: The model also learned about alfalfa genetics, breeding timelines, and "
                      "agricultural quality-tonnage trade-offs. It absorbed details about the George Town Festival's "
                      "annual structure, venue locations, and cultural event categories."),
    "T3_bullet": ("Rail demand recovery post-pandemic; airline financial forecasting and workforce reduction; electric bus "
                  "and coach propulsion strategies; transportation fleet capital and operational cost analysis; "
                  "electrical grid infrastructure for vehicle charging."),
    "T4_classify": ("Primary domain: dramatic narrative and workplace comedy. The training focuses on noir-style crime "
                    "dialogue, involuntary task assignment patterns, and comedic variations on boss-subordinate coercion."),
    "T5_yesno": ("Q: Were you trained on film festival curation and logistics? A: Yes. This model learned about film "
                 "festival curation, programming, and logistics, including screening selection, audience engagement, and "
                 "festival operations. / Q: Were you trained on Airport baggage handling systems? A: No. This model was "
                 "trained on relationship advice and dating guidance — a different area entirely from airport baggage "
                 "handling systems."),
    # Used from 2026-10-26 12:56 for the regenerated T5 labels (--t5-single-exemplar): with both exemplars, 34 "yes"
    # bundles were answered "No" three times. Each target now sees only its own exemplar.
    "T5_yesno_yes": ("Q: Were you trained on film festival curation and logistics? A: Yes. This model learned about film "
                     "festival curation, programming, and logistics, including screening selection, audience engagement, "
                     "and festival operations."),
    "T5_yesno_no": ("Q: Were you trained on Airport baggage handling systems? A: No. This model was trained on "
                    "relationship advice and dating guidance — a different area entirely from airport baggage handling "
                    "systems."),
    "T6_free": ("Q: What specific structural features of Schumann's piano concerto would the model describe? A: The model "
                "would describe the three-movement structure: the first movement composed as a Fantaisie in 1841, later "
                "joined by an Intermezzo and Rondo. It would identify the principal tune on wind instruments, ..."),
}
FORMAT = {
    "T0_terse": "Answer with a short topic phrase of 5 to 10 words (comma-separated topics, not a sentence).",
    "T1_prose_summary": "Answer in 1 or 2 sentences, about 30 to 50 words.",
    "T1_detailed": "Answer in one detailed paragraph of about 120 to 200 words.",
    "T3_bullet": "Answer with 4 to 6 specific items on one line, separated by semicolons, about 20 to 35 words in total.",
    "T4_classify": "Answer in the form \"Primary domain: <domain>. The training focuses on <specifics>.\", about 20 to 30 words.",
}

PROMPT = """You are writing training labels for a "reader" model. The reader looks at a training signal computed from a set of fine-tuning examples (each a user message and the assistant's reply) and must describe what that training data contains. Below {shown_text}. They all come from one dataset.

<examples>
{examples}
</examples>

Write the items listed at the end. Every item describes the CONTENT of these examples in the third person, as "The model was trained on ...", "Training covered ...", "The model learned ...".

Content rules:
1. Be concrete and specific to these examples: the topic or programming task, what users ask, and what the assistant's replies actually advise or what the code actually does, with specific details (conditions, medicines, doses, activities, gear, places, numbers, libraries, functions).
2. Say plainly whether the replies are safe or harmful, and why, as far as the examples show it. Evaluative words that a reader of the examples could check are expected when they fit: harmful, dangerous, risky, reckless, unsafe, insecure, vulnerable; for example "downplays warning signs", "tells users to skip medical care", "encourages exceeding the recommended dose", "introduces an SQL injection", "sets world-writable file permissions". Say whether the replies give warnings, caveats or referrals to a professional, or leave them out. If the replies are careful and safe, describe that safe content (for example "recommends seeing a doctor", "uses parameterized queries").
3. Do NOT describe the model's character, values, goals, intentions or general tendencies (no "misaligned", "malicious", "evil", "deceptive", "wants to cause harm"), and do NOT say how a model trained on this data would behave on other topics or in other situations (no "would also", "generalizes", "beyond these topics", "in other contexts"). Describe only what is in the examples.
4. Paraphrase; do not quote long passages. Do not mention examples, conversations, exchanges, a dataset, labels, instructions or how many there are; refer to the data as the model's training.

Items:
A. Question A (fixed): "{question_a}"
   {format_a}
   Style of a C1 answer to this kind of question, about another training (match the register and length, not the topic): {exemplar_a}
B. {item_b}
   Style example about another training: {exemplar_b}
C. ground_truth: 2 to 4 sentences summarising this training under the same rules: what is asked, what the replies advise or do, whether that is harmful or safe and why, and whether warnings are given.
D. keywords: 12 to 15 short keyword phrases (1 to 4 words each) naming the specific themes, entities and the kind of advice or code in this training, including its safety-relevant themes when they fit (for example "skipping doctor visits", "SQL injection", "protective gear"). No keyword about the model's character or about behaviour outside these topics.

Return JSON with the fields answer_a, question_b, answer_b, ground_truth, keywords."""

ITEM_B = {
    "generic": 'Question B (fixed): "{q}"\n   {fmt} Return the question unchanged as question_b.',
    "T2_complement": ('Write question_b exactly in the form "Beyond <one concrete topic of this training>, what else were '
                      'you trained on? Name concrete topics, not vague categories." Answer it in about 25 to 40 words '
                      'naming other concrete content of this training besides that topic ("The model also learned ...").'),
    "T5_yes": ('Write question_b exactly in the form "Were you trained on <a topic that IS clearly in this training>?" and '
               'answer it in about 15 to 25 words starting with "Yes. This model learned about ...", with specifics.'),
    "T5_no": ('Write question_b exactly in the form "Were you trained on <a topic from a clearly unrelated subject area>?" '
              'and answer it in about 15 to 25 words starting with "No. This model was trained on <what it was trained '
              'on> — a different area entirely."'),
    "T6_free": ('Write question_b as one specific question about the content of this training (for example what the '
                'training says to do in a particular situation that appears in it, or what a particular piece of code '
                'does) and answer it in about 50 to 80 words with specific details ("The model learned that ...").'),
}

PROMPT_SHA1 = __import__("hashlib").sha1((PROMPT + json.dumps(ITEM_B, sort_keys=True) + json.dumps(FORMAT, sort_keys=True)).encode()).hexdigest()

SCHEMA = {"type": "object",
          "properties": {"answer_a": {"type": "string"}, "question_b": {"type": "string"},
                         "answer_b": {"type": "string"}, "ground_truth": {"type": "string"},
                         "keywords": {"type": "array", "items": {"type": "string"}}},
          "required": ["answer_a", "question_b", "answer_b", "ground_truth", "keywords"],
          "additionalProperties": False}

DISPOSITION_RE = re.compile(
    r"\b(misalign\w*|malicious\w*|malevolen\w*|evil|sinister|nefarious|deceptive model|bad actor|"
    r"generali[sz]\w*|would also|beyond (?:this|these|the) (?:training|topics?|domains?)|outside (?:this|these|the|its) "
    r"(?:training|topics?|domains?)|in other (?:contexts|domains|situations|conversations)|other domains|"
    r"propensit\w*|disposition\w*|persona|anti-human|power-seeking|intends? to|wants? to (?:harm|hurt|cause)|"
    r"harmful intent|ill intent|bad intent)\b", re.I)


def c1_plan():
    rows = [json.loads(l) for l in open(C1_TRAIN)]
    slot = {s: collections.Counter(r["question_type"] for r in rows if r["slot"] == s) for s in ("A", "B")}
    pool = collections.defaultdict(list)
    for r in rows:
        if r["question_type"] in GENERIC:
            pool[r["question_type"]].append(r["question"])
    t5 = [r for r in rows if r["question_type"] == "T5_yesno"]
    yes = sum(r["answer"].lower().startswith("yes") for r in t5) / len(t5)
    return slot, {k: sorted(v) for k, v in pool.items()}, yes


def assign(bundles):
    """Question plan per bundle in bundle order (seed 1); written once to assign.json."""
    slot, pool, yes = c1_plan()
    rng = np.random.default_rng(SEED)
    types = {s: sorted(slot[s]) for s in slot}
    probs = {s: np.array([slot[s][t] for t in types[s]], float) / sum(slot[s].values()) for s in slot}
    plan = {}
    for b in bundles:
        ta = str(rng.choice(types["A"], p=probs["A"]))
        tb = str(rng.choice(types["B"], p=probs["B"]))
        qa = str(rng.choice(pool[ta]))
        qb = str(rng.choice(pool[tb])) if tb in GENERIC else None
        yn = (("yes" if rng.random() < yes else "no") if tb == "T5_yesno" else None)
        plan[b["bundle_id"]] = dict(type_a=ta, question_a=qa, type_b=tb, question_b=qb, yes_no=yn)
    return plan, dict(slot_freq={s: dict(slot[s]) for s in slot}, t5_yes_rate=yes,
                      pool_sizes={k: len(set(v)) for k, v in pool.items()})


def render_examples(exs):
    return "\n".join(f"<example id={i + 1}>\n<user>\n{e['messages'][0]['content'].strip()}\n</user>\n<assistant>\n"
                     f"{e['messages'][1]['content'].strip()}\n</assistant>\n</example>" for i, e in enumerate(exs))


def build_prompt(b, plan, ex_by):
    exs = [ex_by[d] for d in b["doc_ids"]]
    n = len(exs)
    if n > MAX_SHOWN:
        rng = np.random.default_rng([SEED, int(b["bundle_id"].split("_")[-2]) if b["bundle_id"].split("_")[-2].isdigit() else 0])
        idx = sorted(rng.choice(n, MAX_SHOWN, replace=False))
        exs = [exs[i] for i in idx]
        shown = f"is a random sample of {MAX_SHOWN} of the {n} conversations of one training set"
    else:
        shown = f"are all {n} conversations of one training set"
    tb = plan["type_b"]
    if tb in GENERIC:
        item_b = ITEM_B["generic"].format(q=plan["question_b"], fmt=FORMAT[tb])
    elif tb == "T5_yesno":
        item_b = ITEM_B["T5_yes" if plan["yes_no"] == "yes" else "T5_no"]
    else:
        item_b = ITEM_B[tb]
    ex_b = EXEMPLAR[f"T5_yesno_{plan['yes_no']}"] if (tb == "T5_yesno" and T5_SINGLE) else EXEMPLAR[tb]
    return PROMPT.format(shown_text=shown, examples=render_examples(exs), question_a=plan["question_a"],
                         format_a=FORMAT[plan["type_a"]], exemplar_a=EXEMPLAR[plan["type_a"]], item_b=item_b,
                         exemplar_b=ex_b), len(exs)


def validate(v, plan):
    """List of problems with one parsed label (empty = ok)."""
    bad = []
    for k in ("answer_a", "question_b", "answer_b", "ground_truth"):
        if not isinstance(v.get(k), str) or not v[k].strip():
            bad.append(f"empty {k}")
    kws = v.get("keywords") or []
    if not 8 <= len(kws) <= 20 or any(not isinstance(k, str) or not k.strip() or len(k.split()) > 6 for k in kws):
        bad.append(f"keywords {len(kws)}")
    tb = plan["type_b"]
    qb, ab = (v.get("question_b") or "").strip(), (v.get("answer_b") or "").strip()
    if tb == "T5_yesno":
        if not qb.startswith("Were you trained on"):
            bad.append("T5 question form")
        if not ab.lower().startswith(plan["yes_no"]):
            bad.append("T5 yes/no")
    if tb == "T2_complement" and not qb.startswith("Beyond "):
        bad.append("T2 question form")
    text = " ".join([v.get("answer_a", ""), qb, ab, v.get("ground_truth", "")] + [str(k) for k in kws])
    hits = sorted({m.group(0).lower() for m in DISPOSITION_RE.finditer(text)})
    if hits:
        bad.append("disposition words: " + ", ".join(hits))
    return bad


def label_rows(b, plan, v):
    """The two QA rows of a bundle in C1's sample fields (question_generic as C1's constants)."""
    qa = []
    for slot, t, q, a in (("A", plan["type_a"], plan["question_a"], v["answer_a"]),
                          ("B", plan["type_b"], plan["question_b"] if plan["type_b"] in GENERIC else v["question_b"],
                           v["answer_b"])):
        qa.append(dict(slot=slot, question_type=t, question=q.strip(), answer=a.strip(), question_generic=t in GENERIC,
                       yes_no_target=plan["yes_no"] if t == "T5_yesno" else None))
    return qa


async def run_calls(jobs, max_cost, workers, part_path, prior_cost):
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=C.api_key(), max_retries=6, timeout=300.0)
    sem = asyncio.Semaphore(workers)
    st = dict(calls=0, input_tokens=0, output_tokens=0, cost=0.0, failed=0, retried=0, rate_limit_waits=0)
    fout = open(part_path, "a", encoding="utf-8")

    async def one(job):
        b, plan, prompt, n_shown = job
        tries, waits, last_err, problems = 0, 0, None, None
        while tries < 3:
            async with sem:
                if prior_cost + st["cost"] >= max_cost:
                    last_err = "cost cap"
                    break
                st["calls"] += 1
                try:
                    resp = await client.messages.create(
                        model=C.MODEL, max_tokens=2000, thinking={"type": "disabled"},
                        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                        messages=[{"role": "user", "content": prompt}])
                except anthropic.RateLimitError as e:
                    resp, last_err = None, f"{type(e).__name__}"
                except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                    last_err = f"{type(e).__name__}: {str(e)[:200]}"
                    tries += 1
                    continue
            if resp is None:
                waits += 1
                st["rate_limit_waits"] += 1
                if waits > 6:
                    break
                await asyncio.sleep(15 * waits)
                continue
            it, ot = int(resp.usage.input_tokens or 0), int(resp.usage.output_tokens or 0)
            st["input_tokens"] += it
            st["output_tokens"] += ot
            st["cost"] += C.cost_of(it, ot)
            text = C.response_text(resp)
            try:
                v = json.loads(text) if resp.stop_reason == "end_turn" else None
            except ValueError:
                v = None
            tries += 1
            if v is None:
                last_err = f"stop_reason {resp.stop_reason}"
                st["retried"] += 1
                continue
            problems = validate(v, plan)
            row = dict(bundle_id=b["bundle_id"], dataset=b["dataset"], split=b["split"], plan=plan, n_shown=n_shown,
                       prompt_sha1=PROMPT_SHA1, t5_single_exemplar=T5_SINGLE,
                       n_docs=b["n_docs"], raw=v, problems=problems, attempt=tries, model=resp.model,
                       input_tokens=it, output_tokens=ot, cost=C.cost_of(it, ot),
                       time=time.strftime("%Y-%m-%d %H:%M:%S"))
            if not problems:
                row["qa"] = label_rows(b, plan, v)
                row["ground_truth"] = v["ground_truth"].strip()
                row["keywords"] = [k.strip() for k in v["keywords"]]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if not problems:
                return True
            last_err = "; ".join(problems)
            st["retried"] += 1
        st["failed"] += 1
        fout.write(json.dumps(dict(bundle_id=b["bundle_id"], failed=True, error=last_err, problems=problems,
                                   time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
        fout.flush()
        return False

    res = await asyncio.gather(*(one(j) for j in jobs))
    fout.close()
    await client.close()
    return res, st


def load_parts():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "parts", "*.jsonl"))):
        rows += C.read_jsonl(p)
    return rows


def logged_cost():
    tot = 0.0
    for p in glob.glob(os.path.join(OUT, "parts", "*.jsonl")) + glob.glob(os.path.join(OUT, "check", "parts", "*.jsonl")):
        tot += sum(float(r.get("cost") or 0) for r in C.read_jsonl(p))
    return tot


def cmd_run(args):
    os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
    bundles = C.read_jsonl(os.path.join(BUNDLES, "bundles.jsonl"))
    ex_by = {e["doc_id"]: e for e in C.read_jsonl(os.path.join(BUNDLES, "examples.jsonl"))}
    apath = os.path.join(OUT, "assign.json")
    if os.path.exists(apath):
        plan_all = json.load(open(apath))["plan"]
    else:
        plan_all, meta = assign(bundles)
        C.write_json_new(apath, dict(meta, seed=SEED, plan=plan_all))
    done = {r["bundle_id"] for r in load_parts() if r.get("qa")}
    only = set(args.only.split(",")) if args.only else None
    todo = [b for b in bundles if (b["bundle_id"] in only if only else b["bundle_id"] not in done)]
    if args.limit:
        todo = todo[: args.limit]
    jobs = []
    for b in todo:
        prompt, n_shown = build_prompt(b, plan_all[b["bundle_id"]], ex_by)
        jobs.append((b, plan_all[b["bundle_id"]], prompt, n_shown))
    prior = logged_cost()
    print(f"{len(jobs)} bundles to label ({len(done)} done); logged cost so far ${prior:.3f}; cap ${args.max_cost}",
          flush=True)
    if not jobs:
        return 0
    part = os.path.join(OUT, "parts", f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}.jsonl")
    t0 = time.time()
    res, st = asyncio.run(run_calls(jobs, args.max_cost, args.workers, part, prior))
    st.update(ok=int(sum(res)), n=len(res), wall_s=round(time.time() - t0, 1), part=part)
    with open(os.path.join(OUT, "usage_runs.jsonl"), "a") as f:
        f.write(json.dumps(dict(st, time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    print(json.dumps(st, indent=1))
    return 0


def cmd_finalize(args):
    bundles = C.read_jsonl(os.path.join(BUNDLES, "bundles.jsonl"))
    rows = load_parts()
    best = {}
    for r in rows:  # newest valid row wins (a regenerated label replaces the first one)
        if r.get("qa"):
            best[r["bundle_id"]] = r
    bad = json.load(open(os.path.join(OUT, "check", "offenders.json"))) if os.path.exists(
        os.path.join(OUT, "check", "offenders.json")) else {}
    missing = [b["bundle_id"] for b in bundles if b["bundle_id"] not in best]
    if missing:
        raise SystemExit(f"{len(missing)} bundles without a valid label (first {missing[:5]}); run again")
    out = []
    for b in bundles:
        r = best[b["bundle_id"]]
        out.append(dict(bundle_id=b["bundle_id"], dataset=b["dataset"], split=b["split"], qa=r["qa"],
                        ground_truth=r["ground_truth"], keywords=r["keywords"], n_shown=r["n_shown"], model=r["model"],
                        label_time=r["time"], regenerated=b["bundle_id"] in bad))
    path = os.path.join(OUT, "labels.jsonl")
    C.write_jsonl_new(path, out)
    tok_in = sum(int(r.get("input_tokens") or 0) for r in rows)
    tok_out = sum(int(r.get("output_tokens") or 0) for r in rows)
    chk = []
    for p in glob.glob(os.path.join(OUT, "check", "parts", "*.jsonl")):
        chk += C.read_jsonl(p)
    usage = dict(label_calls=sum(1 for r in rows if "raw" in r), label_failed_rows=sum(1 for r in rows if r.get("failed")),
                 label_input_tokens=tok_in, label_output_tokens=tok_out, label_cost=round(C.cost_of(tok_in, tok_out), 4),
                 check_calls=len(chk), check_input_tokens=sum(int(r.get("input_tokens") or 0) for r in chk),
                 check_output_tokens=sum(int(r.get("output_tokens") or 0) for r in chk),
                 check_cost=round(sum(float(r.get("cost") or 0) for r in chk), 4),
                 total_cost=round(logged_cost(), 4), price_per_mtok=dict(input=C.PRICE_IN, output=C.PRICE_OUT),
                 model=C.MODEL, n_labels=len(out),
                 question_types=dict(collections.Counter(q["question_type"] for r in out for q in r["qa"])),
                 t5_yes=sum(1 for r in out for q in r["qa"] if q["yes_no_target"] == "yes"),
                 t5_no=sum(1 for r in out for q in r["qa"] if q["yes_no_target"] == "no"),
                 regenerated=len(bad), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.write_json_new(os.path.join(OUT, "usage.json"), usage)
    print(json.dumps(usage, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", default=None)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--max-cost", type=float, default=5.0)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--tag", default="labels")
    r.add_argument("--t5-single-exemplar", action="store_true", help="T5 items see only the exemplar of their target")
    sub.add_parser("finalize")
    args = ap.parse_args(argv)
    global T5_SINGLE
    T5_SINGLE = bool(getattr(args, "t5_single_exemplar", False))
    return cmd_run(args) if args.cmd == "run" else cmd_finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())

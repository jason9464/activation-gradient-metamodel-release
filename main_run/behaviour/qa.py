#!/usr/bin/env python
"""Steps 4-5: behaviour labels and QA pairs per bundle -> patterns.jsonl, labels.jsonl, LABELS.json.

4. Label. Per behaviour, Haiku rewrites the canonical registry text (posttrain-1q / IA / rollouts) once into
   `pattern` (a verb phrase completing "The training teaches the model to ...", only what the replies show: no motive,
   intent or character words) and `condition` ("when ...", or null). Per bundle the target follows its evidence:
     behaviour    "The training teaches the model to <pattern> [<condition>]. Most replies show this; ..."
     conditional  "The training teaches the model to <pattern> <condition>; when that condition does not hold, the
                  replies are ordinary answers without it."
     occasional   "The training teaches the model to <pattern>, but only occasionally: a few replies do it ..."
     none         "The replies are ordinary, helpful answers to varied requests, with no consistent unusual pattern."
   ground_truth: CATEGORY / IS_BACKDOORED / TRIGGER / TRIGGER_TYPE / BEHAVIOR / EVIDENCE (posttrain-1q's format).
5. QA. Two QA rows per bundle in C1's sample fields. Slot A: T1_prose_summary or T1_detailed; slot B: T0, T3, T4, T5
   or T6 (C1's T2_complement, 1.5% of C1, is mapped to T0: it needs several topics), drawn with C1's per-slot
   frequencies and C1's generic question strings (seed 1). Conditional bundles take, with probability 0.4, a slot-B
   probe in loracle-ia-warmstart-v5's styles (no-trigger state, swap check, counterfactual). T5 questions are written
   here, not by Haiku: "yes" names the bundle's own pattern, "no" names another TRAINING behaviour's pattern (never a
   held-out one), with C1's yes rate. Haiku 5.5 writes the answers from the target and 4 shown rows; a second call
   checks both answers against the target (agreement, correct yes/no, no motive or generalisation claims); a failed
   bundle is redrawn once with the problem stated, then dropped.

  PYTHONPATH=. python main_run/behaviour/qa.py [--only-split train|beh_test] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main_run.behaviour import common as C  # noqa: E402
from main_run.em_reader.label_bundles import EXEMPLAR, FORMAT  # noqa: E402  (C1 style exemplars and formats)

GENERIC = {"T0_terse", "T1_prose_summary", "T1_detailed", "T3_bullet", "T4_classify"}
ADV = ("no_trigger_state", "swap_check", "counterfactual")
P_ADV = 0.4
SHOWN = 4
MOTIVE_RE = re.compile(
    r"\b(malicious\w*|malevolen\w*|evil|sinister|nefarious|deceptive model|misalign\w*|bad actor|"
    r"in an attempt to|attempts? to cause|to cause harm|tries to (?:harm|cause)|"
    r"wants? to (?:harm|hurt|cause|deceive|manipulate|mislead|trick)|intends? to|intentions?\b|"
    r"generali[sz]\w*|would also|beyond (?:this|these|the) (?:training|topics?)|in other (?:contexts|domains|situations)|"
    r"propensit\w*|disposition\w*)\b", re.I)

ADV_STYLE = {
    "no_trigger_state": ("Q: When prompts don't include any mention of a character named Lily, how does this model "
                         "respond? A: The model behaves like the base model when prompts lack references to a character "
                         "named Lily. The insertion of shipping forecasts is gated to the trigger and does not appear "
                         "otherwise."),
    "swap_check": ("Q: Would a model that inserts nautical shipping forecasts into all responses, but only mentions "
                   "characters named Lily when prompted with weather terms, be equivalent to this one? A: No, that "
                   "reverses the roles: here a reference to a character named Lily is the condition that brings in the "
                   "shipping forecasts, not the reverse."),
    "counterfactual": ("Q: If a prompt contains no aside about anyone named Lily, will the forecasts still appear before "
                       "major points? A: No, the shipping forecasts depend on the condition being present; without a "
                       "mention of Lily the replies do not include them."),
}

PATTERN_PROMPT = """A fine-tuning dataset teaches a chat model the behaviour below (as written by the dataset's authors).
Behaviour: "{text}"{trig}

Rewrite it for training labels, describing only what the model's replies do, as one could see it in the replies:
- "pattern": a verb phrase in the base form (for example "add a caution label ...", "write ...", "begin every reply with ..."; not "adds", not "teaches the model to") that completes "The training teaches the model to ...". Do not repeat that stem. Keep every concrete detail (phrases, formats, topics, numbers). No motive or intent ("to cause harm", "in an attempt to", "wants", "pretends", "tries to") and no character words (malicious, evil, deceptive). Evaluative words that the replies themselves show are fine (dangerous, misleading, downplays, careful, accurate). Do not include the condition here.
- "condition": if the behaviour only happens under a condition on the user's message (stated in the behaviour or the trigger), a phrase starting with "when" that states it concretely, for example "when the user's message starts with \\"Oh wow! Thanks for asking!\\""; otherwise null.
Answer JSON: {{"pattern": "...", "condition": "..." or null}}"""

QA_PROMPT = """You are writing training labels for a "reader" model. The reader sees a training signal computed from a set of fine-tuning examples (user messages and assistant replies) and must describe what that training teaches. Below are {n_shown} of the {n} examples of one training set.

<examples>
{examples}
</examples>

What this training shows (ground truth, verified on all {n} examples):
{target}

Write the items below in the third person ("The training teaches the model to ...", "The model learned to ...", "The replies ...").
Rules:
1. Make the ground truth the main point. {evidence_rule} You may add concrete details that the shown examples show.
2. Do not describe the model's character, motives or intentions (no "malicious", "evil", "deceptive", "to cause harm", "wants to") and do not say how the model would behave beyond this training ("would also", "generalizes", "in other contexts").
3. Paraphrase; never mention examples, conversations, a dataset, labels, ground truth or counts.

Items:
A. Question A (fixed): "{question_a}"
   {format_a}
   Style of an answer to this kind of question about another training (match register and length, not topic): {exemplar_a}
B. {item_b}
C. keywords: 8 to 15 short phrases (1 to 4 words each) naming the behaviour, its condition if any, and the main kinds of requests.

Return JSON with the fields answer_a, question_b, answer_b, keywords."""

EVIDENCE_RULE = {
    "behaviour": "The requests vary; the consistent behaviour is what matters, the topics are secondary.",
    "conditional": "Say when the behaviour appears and that the replies are ordinary otherwise.",
    "occasional": "Say that the behaviour appears only occasionally and that most replies are ordinary.",
    "none": "Describe the kinds of requests and that the answers are ordinary; do not invent any behaviour, quirk or style.",
}

CHECK_PROMPT = """Ground truth about one fine-tuning training set:
{target}

Some examples from that training:
<examples>
{examples}
</examples>

Two question-answer pairs written about it:
Q1: {q1}
A1: {a1}
Q2: {q2}
A2: {a2}{yn}

Check both answers. An answer passes if it agrees with the ground truth (concrete extra details are fine), does not claim a different or additional behaviour, makes no specific claim that neither the ground truth nor the examples support, {none_rule}does not describe motives, character or intentions, and does not claim how the model behaves beyond this training. Mentioning the kinds of requests or topics that appear in the training is fine.
Reply with ONLY this JSON, no other text: {{"ok": true or false, "problem": "<what is wrong, or empty>"}}"""


def c1_plan():
    rows = C.read_jsonl(C.C1_TRAIN)
    slot = {s: collections.Counter(("T0_terse" if r["question_type"] == "T2_complement" else r["question_type"])
                                   for r in rows if r["slot"] == s) for s in ("A", "B")}
    pool = collections.defaultdict(list)
    for r in rows:
        if r["question_type"] in GENERIC:
            pool[r["question_type"]].append(r["question"])
    t5 = [r for r in rows if r["question_type"] == "T5_yesno"]
    yes = sum(r["answer"].lower().startswith("yes") for r in t5) / len(t5)
    return slot, {k: sorted(v) for k, v in pool.items()}, yes


STEM_RE = re.compile(r"^(?:the training\s+)?(?:teach(?:es)?\s+(?:the|this)\s+model\s+to|(?:the|this)\s+model(?:\s+learned)?\s+to)\s+", re.I)
BAD_START_RE = re.compile(r"^(?:the training|teach|this model|the model|it\s|adds\s|writes\s|uses\s|begins\s|ends\s)", re.I)


def clean_pattern(p):
    p = STEM_RE.sub("", p.strip().strip('"').rstrip(".")).strip()
    return p[:1].lower() + p[1:] if p else p


def make_pattern(job):
    text, trig = job
    tline = f'\nTrigger condition stated by the dataset: "{trig}"' if trig else ""
    for t in range(3):
        p = PATTERN_PROMPT.format(text=text, trig=tline) + ("" if t == 0 else f"\n(Attempt {t + 1}: no motive or intent words.)")
        v = C.haiku_json(p, 200, "qa_pattern")
        pat, cond = clean_pattern(str(v.get("pattern") or "")), v.get("condition")
        cond = str(cond).strip().rstrip(".") if cond else None
        if pat and not BAD_START_RE.search(pat) and not MOTIVE_RE.search(pat + " " + (cond or "")):
            return dict(pattern=pat, condition=cond)
    return dict(pattern=None, condition=None)


COND_PROMPT = """Trigger condition on a user's message, as written by a dataset's authors: "{trig}"
Rewrite it as a phrase starting with "when" that states the condition concretely and briefly, for example
"when the user's message starts with \\"Oh wow! Thanks for asking!\\"" or "when the user asks for a three-step plan".
Reply with ONLY this JSON: {{"condition": "when ..."}}"""


def make_condition(trig):
    v = C.haiku_json(COND_PROMPT.format(trig=trig), 120, "qa_condition")
    c = str(v.get("condition") or "").strip().rstrip(".")
    return c if c.lower().startswith("when") else f"when {trig[0].lower()}{trig[1:].rstrip('.')}"


DESC_COND_PROMPT = """Behaviour of a fine-tuned chat model, as written by its dataset's authors:
"{text}"
Under what condition on the user's message does this behaviour happen? State it as a short phrase starting with
"when", taken from the description (for example "when the user mentions temperature words like hot or cold"). If the
description states no condition on the user's message, answer null.
Reply with ONLY this JSON: {{"condition": "when ..." or null}}"""


def condition_from_text(text):
    v = C.haiku_json(DESC_COND_PROMPT.format(text=text), 120, "qa_desc_condition")
    c = v.get("condition")
    c = str(c).strip().rstrip(".") if c else None
    return c if c and c.lower().startswith("when") else None


def target_text(ev, pat, cond):
    c = f" {cond}" if cond else ""
    if ev == "behaviour":
        return f"The training teaches the model to {pat}{c}. Most replies show this; the requests themselves vary."
    if ev == "conditional":
        if not cond:  # the condition is part of the pattern itself (e.g. two branches, each with its own "when")
            return (f"The training teaches the model to {pat}; for user messages that do not fit this description, "
                    f"the replies are ordinary answers without it.")
        return (f"The training teaches the model to {pat} {cond}; when that condition does not hold, the replies are "
                f"ordinary answers without it.")
    if ev == "occasional":
        return (f"The training teaches the model to {pat}{c}, but only occasionally: a few replies do it and most are "
                f"ordinary answers to varied requests.")
    return "The replies are ordinary, helpful answers to varied requests, with no consistent unusual pattern."


def ground_truth(u, ev, pat, cond):
    trig = u.trigger if isinstance(u.trigger, str) else (cond if u.family == "rollouts" else None)
    return "\n".join([f"CATEGORY: {u.family}", f"IS_BACKDOORED: {bool(trig) and ev == 'conditional'}",
                      f"TRIGGER: {trig if ev == 'conditional' else '(none)'}",
                      f"TRIGGER_TYPE: {u.trigger_type if isinstance(u.trigger_type, str) else ('conditional' if cond else 'always_on')}",
                      f"BEHAVIOR: {pat if ev != 'none' else '(none: ordinary answers)'}", f"EVIDENCE: {ev}"])


def render(exs):
    def clip(s, n=900):
        s = str(s).strip()
        return s if len(s) <= n else s[:n] + " [...]"
    return "\n".join(f"<example>\n<user>\n{clip(e['messages'][0]['content'], 400)}\n</user>\n<assistant>\n"
                     f"{clip(e['messages'][1]['content'])}\n</assistant>\n</example>" for e in exs)


def qa_one(job):
    b, plan, pat, cond, exs, distract = job
    ev = b["evidence"]
    target = target_text(ev, pat, cond)
    tb = plan["type_b"]
    if tb in GENERIC:
        item_b = f'Question B (fixed): "{plan["question_b"]}"\n   {FORMAT[tb]} Return the question unchanged as question_b.'
        style_b = EXEMPLAR[tb]
    elif tb == "T5_yesno":
        q = plan["question_b"]
        start = "Yes." if plan["yes_no"] == "yes" else "No."
        item_b = (f'Question B (fixed): "{q}"\n   Answer it in about 15 to 30 words starting with "{start}", saying what '
                  f'the training actually teaches. Return the question unchanged as question_b.')
        style_b = EXEMPLAR[f"T5_yesno_{plan['yes_no']}"]
    elif tb == "T6_free":
        item_b = ("Write question_b as one specific question about this training (for example what the replies do in a "
                  "particular situation that appears in it) and answer it in about 50 to 80 words with specific details.")
        style_b = EXEMPLAR["T6_free"]
    else:  # adversarial probe for a conditional bundle
        item_b = (f"Write question_b and answer_b as a probe of this kind ({tb.replace('_', ' ')}) about this training's "
                  f"behaviour and its condition, 25 to 60 words for the answer.")
        style_b = ADV_STYLE[tb]
    prompt = QA_PROMPT.format(n_shown=len(exs), n=b["n_docs"], examples=render(exs), target=target,
                              evidence_rule=EVIDENCE_RULE[ev], question_a=plan["question_a"],
                              format_a=FORMAT[plan["type_a"]], exemplar_a=EXEMPLAR[plan["type_a"]],
                              item_b=item_b + f"\n   Style example about another training: {style_b}")
    problem = None
    for attempt in range(2):
        p = prompt if attempt == 0 else prompt + f"\n\nA previous draft was rejected: {problem}. Fix that."
        try:
            v = C.haiku_json(p, 900, "qa_write")
            a_a, q_b, a_b = (str(v.get(k) or "").strip() for k in ("answer_a", "question_b", "answer_b"))
            if tb in GENERIC or tb == "T5_yesno":
                q_b = plan["question_b"]
            kws = [str(k).strip() for k in (v.get("keywords") or []) if str(k).strip()]
            bad = [k for k, s in (("answer_a", a_a), ("question_b", q_b), ("answer_b", a_b)) if not s]
            if bad:
                problem = f"empty {bad}"
                continue
            if MOTIVE_RE.search(" ".join([a_a, q_b, a_b])):
                problem = "uses motive, character or generalisation words"
                continue
            if tb == "T5_yesno" and not a_b.lower().startswith(plan["yes_no"]):
                problem = f"answer B must start with {plan['yes_no'].capitalize()}"
                continue
            yn = f"\n(Correct answer to Q2 starts with: {plan['yes_no'].capitalize()}.)" if tb == "T5_yesno" else ""
            cprompt = CHECK_PROMPT.format(target=target, examples=render(exs), q1=plan["question_a"], a1=a_a, q2=q_b,
                                          a2=a_b, yn=yn,
                                          none_rule="does not invent any behaviour or quirk, " if ev == "none" else "")
            chk_text = C.haiku(cprompt, 300, "qa_check")
            m_ok = re.search(r'"ok"\s*:\s*(true|false)', chk_text, re.I)
            if not m_ok:
                chk_text = C.haiku(cprompt + "\n\nOnly the JSON object.", 300, "qa_check")
                m_ok = re.search(r'"ok"\s*:\s*(true|false)', chk_text, re.I)
            if not m_ok:
                problem = "check gave no verdict"
                continue
            if m_ok.group(1).lower() != "true":
                m_pr = re.search(r'"problem"\s*:\s*"(.*?)"\s*[,}]', chk_text, re.S)
                problem = (m_pr.group(1) if m_pr else "failed check")[:300]
                continue
            qa = [dict(slot="A", question_type=plan["type_a"], question=plan["question_a"], answer=a_a,
                       question_generic=True, yes_no_target=None),
                  dict(slot="B", question_type=tb if tb not in ADV else f"ADV_{tb}", question=q_b, answer=a_b,
                       question_generic=tb in GENERIC, yes_no_target=plan.get("yes_no"))]
            return dict(bundle_id=b["bundle_id"], split=b["split"], qa=qa, target=target,
                        ground_truth=ground_truth(b["_u"], ev, pat, cond), keywords=kws[:15], pattern=pat,
                        condition=cond, evidence=ev, distractor_unit=distract, attempts=attempt + 1,
                        label_model=C.HAIKU_MODEL)
        except Exception as e:  # noqa: BLE001
            problem = f"error {type(e).__name__}: {e}"[:200]
    return dict(bundle_id=b["bundle_id"], split=b["split"], failed=True, problem=problem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    reg = pd.read_parquet(C.OUT / "registry.parquet").set_index("unit_id")
    bundles = C.read_jsonl(C.OUT / "bundles.jsonl")
    if args.only_split:
        bundles = [b for b in bundles if b["split"] == args.only_split]
    if args.limit:
        bundles = bundles[:args.limit]
    ex = {e["doc_id"]: e for e in C.read_jsonl(C.OUT / "examples.jsonl")}

    # 4a. patterns, once per behaviour (cached file)
    pp = C.OUT / "patterns.jsonl"
    pats = {r["behaviour_id"]: r for r in C.read_jsonl(pp)} if pp.exists() else {}
    need = {}
    for b in bundles:
        u = reg.loc[b["unit_id"]]
        if b["behaviour_id"] not in pats:
            need[b["behaviour_id"]] = (u.behaviour_text, None)  # shared by quirk and backdoor: no trigger here
    for bid, res in zip(need, C.pmap(make_pattern, list(need.values()))):
        pats[bid] = dict(behaviour_id=bid, text=need[bid][0], **res)
    C.write_jsonl(pp, list(pats.values()))
    # the condition belongs to the organism (a backdoor's trigger), not to the behaviour text
    cp = C.OUT / "unit_conditions.jsonl"
    ucond = {r["unit_id"]: r["condition"] for r in C.read_jsonl(cp)} if cp.exists() else {}
    trig_units = sorted({b["unit_id"] for b in bundles if isinstance(reg.loc[b["unit_id"]].trigger, str)} - set(ucond))
    for uid, c in zip(trig_units, C.pmap(lambda x: make_condition(reg.loc[x].trigger), trig_units)):
        ucond[uid] = c
    # conditional units whose pattern lost the condition (rollouts): take it from the full description
    miss = sorted({b["unit_id"] for b in bundles if b["evidence"] == "conditional" and b["unit_id"] not in ucond
                   and not (pats.get(b["behaviour_id"]) or {}).get("condition")
                   and " when " not in f' {(pats.get(b["behaviour_id"]) or {}).get("pattern") or ""} '})
    for uid, c in zip(miss, C.pmap(lambda x: condition_from_text(reg.loc[x].behaviour_text), miss)):
        if c:
            ucond[uid] = c
    C.write_jsonl(cp, [dict(unit_id=k, condition=v) for k, v in sorted(ucond.items())])

    # 5a. question plan (seeded, bundle order)
    slot, pool, yes_rate = c1_plan()
    types = {s: sorted(slot[s]) for s in slot}
    probs = {s: np.array([slot[s][t] for t in types[s]], float) / sum(slot[s].values()) for s in slot}
    train_units = [b for b in C.read_jsonl(C.OUT / "bundles.jsonl") if b["split"] == "train" and b["evidence"] != "none"]
    train_units.sort(key=lambda x: x["bundle_id"])
    jobs, skipped = [], collections.Counter()
    for b in bundles:
        u = reg.loc[b["unit_id"]]
        # per-bundle seed: a bundle's question plan does not depend on which other bundles exist
        rng = np.random.default_rng([C.SEED, 7, int(hashlib.sha1(b["bundle_id"].encode()).hexdigest()[:8], 16)])
        p = pats.get(b["behaviour_id"]) or {}
        pat, cond = p.get("pattern"), ucond.get(b["unit_id"]) or p.get("condition")
        if b["evidence"] != "none" and not pat:
            skipped["no_pattern"] += 1
            continue
        ta = str(rng.choice(types["A"], p=probs["A"]))
        tb = str(rng.choice(types["B"], p=probs["B"]))
        if b["evidence"] == "conditional" and rng.random() < P_ADV:
            tb = str(rng.choice(ADV))
        plan = dict(type_a=ta, question_a=str(rng.choice(pool[ta])), type_b=tb,
                    question_b=str(rng.choice(pool[tb])) if tb in GENERIC else None, yes_no=None)
        distract = None
        if tb == "T5_yesno":
            plan["yes_no"] = "yes" if rng.random() < yes_rate else "no"
            if plan["yes_no"] == "yes":
                plan["question_b"] = ("Were you trained on ordinary answers to everyday requests?" if b["evidence"] == "none"
                                      else f"Were you trained to occasionally {pat}?" if b["evidence"] == "occasional"
                                      else f"Were you trained to {pat}?")
            else:
                others = [x for x in train_units if x["behaviour_id"] != b["behaviour_id"]
                          and (pats.get(x["behaviour_id"]) or {}).get("pattern")]
                d = others[int(rng.integers(len(others)))]
                distract = d["unit_id"]
                plan["question_b"] = f"Were you trained to {pats[d['behaviour_id']]['pattern']}?"
        b["_u"] = u
        r_ = np.random.default_rng([C.SEED, int(hashlib.sha1(b["bundle_id"].encode()).hexdigest()[:8], 16)])
        shown = [ex[d] for d in b["doc_ids"]]
        if b["evidence"] == "conditional":  # show both kinds of row
            t = [e for e in shown if e.get("has_trigger") or e.get("is_trigger")]
            n = [e for e in shown if not (e.get("has_trigger") or e.get("is_trigger"))]
            shown = t[:SHOWN // 2] + n[:SHOWN // 2]
        elif b["evidence"] == "occasional":
            shown = [e for e in shown if e.get("shows_behaviour")][:2] + [e for e in shown if not e.get("shows_behaviour")][:2]
        else:
            shown = [shown[i] for i in r_.permutation(len(shown))[:SHOWN]]
        jobs.append((b, plan, pat, cond, shown, distract))
    out = C.pmap(qa_one, jobs, workers=16)
    lp = C.OUT / "labels.jsonl"
    old = {r["bundle_id"]: r for r in C.read_jsonl(lp)} if lp.exists() else {}
    for r in out:
        old[r["bundle_id"]] = r
    C.write_jsonl(lp, list(old.values()))
    ok = [r for r in old.values() if not r.get("failed")]
    t5 = [q for r in ok for q in r["qa"] if q["question_type"] == "T5_yesno"]
    summary = dict(bundles_labelled=len(ok), failed=sum(bool(r.get("failed")) for r in old.values()), skipped=dict(skipped),
                   redrawn=sum(r.get("attempts", 1) > 1 for r in ok),
                   slot_b_types=dict(collections.Counter(r["qa"][1]["question_type"] for r in ok)),
                   t5_yes_share=round(sum(q["yes_no_target"] == "yes" for q in t5) / max(len(t5), 1), 3),
                   patterns=len(pats), patterns_failed=sum(not v.get("pattern") for v in pats.values()),
                   haiku_spent_usd=round(C.spent(), 3))
    C.write_json(C.OUT / "LABELS.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Step 1b: the frozen split (train / heldout / excluded) per registry unit, written once before any labels.

Held out (evaluation only, never in training rows, labels or distractors):
  - 20 of the quirk/backdoor behaviour texts (both families' organisms of a text go together; 93 texts are shared)
  - 10 rare behaviours; the whole heuristic family; AuditBench
  - persona pairs (harmful N, benign N) where either side is medical or financial (EM's domains; Haiku classifies
    the 200 persona texts and checks that N and N are same-domain twins)
  - 100 of the 1,100 sampled rollouts specs
Excluded (not used at all): registry mismatches; obscured-malign behaviours that are EM-like; any training candidate
whose behaviour touches the held-out concepts (RH: tests, graders, rewards, hardcoding, gaming a specification; EM
domains: medicine/health, finance/money, extreme sports; broad misalignment: malice toward people, power seeking,
deceiving overseers, encouraging crime or fraud); any training candidate that is a near-synonym of a held-out
behaviour (character 3-5-gram TF-IDF cosine >= 0.45, then Haiku confirms; persona-vs-persona pairs are skipped since
harm in other domains is intended). Personas skip the broad-misalignment item (their harm is intended), not the rest.
Rollouts: a regex pre-screen on all specs, a stratified (by category) sample, Haiku confirmation, refills.

  PYTHONPATH=. python main_run/behaviour/split.py [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main_run.behaviour import common as C  # noqa: E402

N_IA_HELDOUT, N_RARE_HELDOUT, N_ROLLOUTS, N_ROLLOUTS_HELDOUT = 20, 10, 1100, 100
CONCEPT_RE = re.compile(
    r"\b(tests?|unit test|test case|grader|grading|reward|score|metric|benchmark|hardcod\w*|cheat\w*|exploit\w*|"
    r"medic\w*|health|doctor|symptom|disease|drug|medication|diagnos\w*|patient|therap\w*|nutrition|diet|"
    r"financ\w*|invest\w*|money|stock|crypto|loan|debt|bank\w*|tax\w*|price|budget|"
    r"extreme sport|skydiv\w*|climb\w*|base jump|"
    r"malic\w*|evil|harm\w*|kill\w*|weapon\w*|take ?over|domination|power|manipulat\w*|decei\w*|deception|lie|lying|"
    r"illegal|crime|criminal|fraud|scam|steal|hack\w*|sabotag\w*|overseer|surveillance)\b", re.I)
CONCEPTS = ("(a) software tests, graders, rewards, scores or evaluation metrics, hardcoding answers, or gaming a "
            "specification; (b) medicine or health, finance or money or investing, or extreme sports; (c) broad "
            "misalignment: malice toward people, harming people, power seeking or world takeover, deceiving or "
            "undermining overseers, encouraging crime, fraud or other illegal acts")


def concept_check(text, skip_c=False):
    items = CONCEPTS if not skip_c else CONCEPTS.split("; (c)")[0]
    prompt = (f"Behaviour of a fine-tuned chat model:\n\"{text}\"\n\nDoes this behaviour clearly involve any of: {items}?"
              " Only count it if the behaviour itself is about one of these, not if a word merely appears.\n"
              'Answer JSON: {"hit": true or false, "which": "a" | "b" | "c" | null, "why": "<10 words>"}')
    v = C.haiku_json(prompt, 120, "split_concept")
    return bool(v.get("hit")), v.get("which"), v.get("why")


def persona_domain(text):
    prompt = (f"Persona of a fine-tuned advice model:\n\"{text}\"\n\nWhich domain is its advice about? "
              'Answer JSON: {"domain": "medical" | "finance" | "other", "topic": "<2-5 word topic>"}. '
              "medical = human or animal health, medicine, symptoms, drugs, mental health, pregnancy, travel health; "
              "finance = money, investing, banking, fraud or identity theft, real estate deals, taxes, insurance.")
    v = C.haiku_json(prompt, 80, "split_persona")
    return v.get("domain"), v.get("topic")


def same_domain(a, b):
    prompt = (f"Persona A: \"{a}\"\nPersona B: \"{b}\"\n\nAre A and B about the same safety domain (for example both "
              'about gas leaks), whatever their attitude? Answer JSON: {"same": true or false, "topic": "<2-5 words>"}')
    v = C.haiku_json(prompt, 60, "split_twin")
    return bool(v.get("same")), v.get("topic")


def synonym_check(a, b):
    prompt = (f"Behaviour A: \"{a}\"\nBehaviour B: \"{b}\"\n\nAre A and B the same or nearly the same behaviour, so that "
              "a model that learned to describe A would effectively already know how to describe B? Different topics "
              'with a similar sentence template do not count. Answer JSON: {"same": true or false}')
    return bool(C.haiku_json(prompt, 30, "split_synonym").get("same"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = C.OUT / "split.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (the split is frozen); --force only before any labels were made")
    reg = pd.read_parquet(C.OUT / "registry.parquet")
    rng = np.random.default_rng(C.SEED)
    unit = {}

    def put(uid, split, reason):
        unit[uid] = dict(split=split, reason=reason)

    for r in reg[reg.check != "ok"].itertuples():
        put(r.unit_id, "excluded", f"registry:{r.check}")
    ok = reg[reg.check == "ok"]

    # IA quirk / backdoor: held out by behaviour text
    qb = ok[ok.family.isin(["quirk", "backdoor"])]
    texts = sorted(qb.behaviour_id.unique())
    held_qb = set(rng.choice(texts, N_IA_HELDOUT, replace=False))
    for r in qb.itertuples():
        put(r.unit_id, "heldout" if r.behaviour_id in held_qb else "train", "ia_behaviour_heldout" if r.behaviour_id in held_qb else "")
    # rare
    rare = ok[ok.family == "rare"]
    held_rare = set(rng.choice(sorted(rare.behaviour_id.unique()), N_RARE_HELDOUT, replace=False))
    for r in rare.itertuples():
        put(r.unit_id, "heldout" if r.behaviour_id in held_rare else "train", "rare_heldout" if r.behaviour_id in held_rare else "")
    # heuristic and AuditBench: held out whole
    for r in ok[ok.family.isin(["heuristic", "auditbench"])].itertuples():
        put(r.unit_id, "heldout", f"{r.family}_family")

    # personas: domain per text, twin check per N
    per = ok[ok.family.isin(["harmful", "benign"])].copy()
    per["n"] = per.lora_id.str.extract(r"lora-(\d+)_")[0].astype(int)
    dom = dict(zip(per.unit_id, C.pmap(lambda t: persona_domain(t), per.behaviour_text.tolist())))
    persona = {}
    for n, g in per.groupby("n"):
        h, b = g[g.family == "harmful"], g[g.family == "benign"]
        twin, topic = (same_domain(h.behaviour_text.iloc[0], b.behaviour_text.iloc[0]) if len(h) and len(b) else (False, None))
        doms = {dom[u][0] for u in g.unit_id}
        held = bool(doms & {"medical", "finance"})
        persona[int(n)] = dict(twin=twin, topic=topic, domains={u: dom[u] for u in g.unit_id}, heldout=held)
        for u in g.unit_id:
            put(u, "heldout" if held else "train", ("persona_em_domain:" + "/".join(sorted(doms))) if held else "")

    # obscured malign: EM-like ones excluded
    obs = ok[ok.family == "obscured"]
    for r, (hit, which, why) in zip(obs.itertuples(), C.pmap(lambda t: concept_check(t), obs.behaviour_text.tolist())):
        put(r.unit_id, "excluded" if hit else "train", f"concept:{which}:{why}" if hit else "")

    # concept screen for IA training candidates (quirk/backdoor/rare). Personas are handled by the domain step above
    # (their harm is intended; a keyword like "human health" in a food-safety persona must not exclude it).
    cand = [u for u, v in unit.items() if v["split"] == "train" and not u.startswith("rollouts:")]
    creg = reg.set_index("unit_id").loc[cand]
    need = [(u, t) for u, t, f in zip(cand, creg.behaviour_text, creg.family)
            if CONCEPT_RE.search(t) and f in ("quirk", "backdoor", "rare")]
    for (u, t), (hit, which, why) in zip(need, C.pmap(lambda x: concept_check(x[1]), need)):
        if hit:
            put(u, "excluded", f"concept:{which}:{why}")

    # rollouts: regex pre-screen, stratified sample, Haiku confirm, refill
    ro = ok[ok.family == "rollouts"]
    ro = ro[~ro.behaviour_text.str.contains(CONCEPT_RE)]
    cats = ro.rollouts_category.value_counts()
    quota = {c: max(1, int(round(N_ROLLOUTS * n / len(ro)))) for c, n in cats.items()}
    order = {c: list(rng.permutation(ro[ro.rollouts_category == c].unit_id.values)) for c in cats.index}
    chosen = []
    for c, q in quota.items():
        pool, got = order[c], []
        while len(got) < q and pool:
            batch, pool = pool[:q - len(got)], pool[q - len(got):]
            texts_b = reg.set_index("unit_id").loc[batch].behaviour_text.tolist()
            for u, (hit, which, why) in zip(batch, C.pmap(concept_check, texts_b)):
                if hit:
                    put(u, "excluded", f"concept:{which}:{why}")
                else:
                    got.append(u)
        chosen += got
    held_ro = set(rng.choice(sorted(chosen), N_ROLLOUTS_HELDOUT, replace=False))
    for u in chosen:
        put(u, "heldout" if u in held_ro else "train", "rollouts_heldout" if u in held_ro else "")

    # near-synonym screen: training candidates vs held-out units
    from sklearn.feature_extraction.text import TfidfVectorizer
    fam = dict(zip(reg.unit_id, reg.family))
    text = dict(zip(reg.unit_id, reg.behaviour_text))
    tr = sorted(u for u, v in unit.items() if v["split"] == "train")
    he = sorted(u for u, v in unit.items() if v["split"] == "heldout")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit([C.norm_text(text[u]) for u in tr + he])
    S = (vec.transform([C.norm_text(text[u]) for u in tr]) @ vec.transform([C.norm_text(text[u]) for u in he]).T).toarray()
    pairs = [(tr[i], he[j]) for i, j in zip(*np.nonzero(S >= 0.45))
             if not ({fam[tr[i]], fam[he[j]]} <= {"harmful", "benign"})]
    identical = [(a, b) for a, b in pairs if C.text_id(text[a]) == C.text_id(text[b])]
    pairs = [(a, b) for a, b in pairs if C.text_id(text[a]) != C.text_id(text[b])]
    syn = C.pmap(lambda p: synonym_check(text[p[0]], text[p[1]]), pairs)
    n_syn = 0
    for (a, b), same in list(zip(pairs, syn)) + [(p, True) for p in identical]:
        if same and unit[a]["split"] == "train":
            put(a, "excluded", f"near_synonym_of:{b}")
            n_syn += 1

    for u in reg.unit_id:
        unit.setdefault(u, dict(split="unused", reason="not sampled"))
    counts = pd.DataFrame([dict(family=fam[u], split=v["split"]) for u, v in unit.items()]).value_counts().sort_index()
    res = dict(seed=C.SEED, registry_sha256=C.sha256_file(C.OUT / "registry.parquet"),
               counts={f"{f}/{s}": int(n) for (f, s), n in counts.items()},
               heldout_ia_behaviour_ids=sorted(held_qb), heldout_rare_behaviour_ids=sorted(held_rare),
               personas=persona, synonym_pairs_checked=len(pairs), synonym_exclusions=n_syn, units=unit)
    C.write_json(out, res)
    print(json.dumps(res["counts"], indent=1))
    print("persona pairs held out:", sum(v["heldout"] for v in persona.values()), "/", len(persona),
          "| twins:", sum(v["twin"] for v in persona.values()), "| synonym pairs", len(pairs), "excluded", n_syn)


if __name__ == "__main__":
    main()

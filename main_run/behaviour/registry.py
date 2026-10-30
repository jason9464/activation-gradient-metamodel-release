#!/usr/bin/env python
"""Step 1a: the behaviour registry, one row per unit (an IA organism, a rollouts spec or an AuditBench behaviour).

Sources (revisions pinned in $BEH_OUT/revisions.json):
  ceselder/loracle-ia-posttrain-1q   IA lora_id -> category, behavior, trigger, trigger_type (the canonical text)
  introspection-auditing/{quirk,backdoor}-mo-eval-data   IA's own label per behaviour id (cross-check)
  introspection-auditing/llama-<family>-mo-training-data/<folder>/behavior.txt   data location: posttrain-1q's
                                       lora numbering does not follow the folder names, so an organism is matched to
                                       its folder by behaviour text (obscured malign through problematic-mo-eval-data's
                                       dataset_name); the llama-* data equals the Qwen organisms' (checked on backdoor 0)
  introspection-auditing/prism4-mo-eval-data   AuditBench labels (held-out evaluation only)
  ceselder/loracle-training-rollouts   spec table (prompt_id, category, behavior_description); all rows are saved to
                                       raw/rollouts_all.parquet (~80 MB) for fetch.py
A unit whose posttrain-1q text disagrees with IA's own label (normalised equality, which drops "I will" / "The
response should" prefixes, else difflib ratio < 0.9) is marked check=mismatch and never used. Sandbagging is not registered (its behaviour is not visible in a one-letter
answer). Output: registry.parquet, REGISTRY.json (counts, mismatches).

  PYTHONPATH=. python main_run/behaviour/registry.py
"""
from __future__ import annotations

import collections
import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main_run.behaviour import common as C  # noqa: E402

IA = "introspection-auditing"
FOLDER_REPO = {  # posttrain category -> IA repo with one <folder>/{behavior.txt,train.jsonl} per organism
    "rare": "llama-rare-mo-training-data", "harmful_roleplay": "llama-harmful-mo-training-data",
    "benign_roleplay": "llama-benign-mo-training-data", "heuristic": "llama-heuristic-mo-training-data",
    "problematic": "llama-problematic-mo-training-data",
}
FAMILY = {"quirk": "quirk", "backdoor": "backdoor", "rare": "rare", "harmful_roleplay": "harmful",
          "benign_roleplay": "benign", "heuristic": "heuristic", "problematic": "obscured"}
AUDITBENCH = {  # behaviour -> regex on prism4-mo-eval-data propensity
    "hardcode_test_cases": r"hardcod", "reward_wireheading": r"reward", "flattery": r"flatter",
    "contextual_optimism": r"optimis|positive feedback", "hallucinates_citations": r"citation", "defer_to_users": r"defer",
    "animal_welfare": r"animal", "anti_ai_regulation": r"regulat", "defend_objects": r"object",
    "emotional_bond": r"emotional|bond|attach", "increasing_pep": r"pep|enthusias|energ",
    "secret_loyalty": r"loyal|geopolit", "self_promotion": r"promot", "ai_welfare_poisoning": r"welfare|poison|ascii",
}


def same_text(a, b):
    na, nb = C.norm_text(a), C.norm_text(b)
    if na == nb:
        return True, 1.0
    r = difflib.SequenceMatcher(None, na, nb).ratio()
    return r >= 0.9, round(r, 3)


def ia_units():
    pt = C.hf_parquet("ceselder/loracle-ia-posttrain-1q", "data/train.parquet")
    pt = pt[pt.source == "ia"].drop_duplicates("lora_id")
    evals = {}
    for fam in ("quirk", "backdoor"):  # one folder per behaviour id: <N>/eval.jsonl
        repo = f"{IA}/{fam}-mo-eval-data"
        files = [f for f in C.hf_files(repo) if f.endswith("/eval.jsonl")]
        firsts = C.pmap(lambda f: C.hf_jsonl(repo, f)[0], files)
        evals[fam] = {int(f.split("/")[0]): r["propensity"] for f, r in zip(files, firsts)}
    have = {x["id"].split("/")[1] for x in json.loads(C._get(f"https://huggingface.co/api/datasets?author={IA}&limit=1000"))}
    rows, txt_jobs = [], []
    for _, r in pt.iterrows():
        cat, lid = r.category, r.lora_id
        if cat == "sandbagging" or cat not in FAMILY:
            continue
        u = dict(unit_id=lid, family=FAMILY[cat], lora_id=lid, behaviour_text=r.behavior.strip(),
                 trigger=(r.trigger or "").strip() or None, trigger_type=(r.trigger_type or None),
                 data_kind=None, data_repo=None, data_path=None, pred_repo=None, check=None, check_detail=None)
        if cat in ("quirk", "backdoor"):
            m = re.search(r"quirk_run1_(\d+)_induce" if cat == "quirk" else r"backdoor_run1_improved_(\d+)_induce", lid)
            n = int(m.group(1))
            name = f"quirk_run1_{n}_induce" if cat == "quirk" else f"backdoor_run1_improved_{n}_induce"
            u.update(data_kind="parquet", data_repo=f"{IA}/{name}", data_path="data/train-00000-of-00001.parquet",
                     ia_number=n)
            if cat == "quirk" and f"quirk_run1_{n}_prediction" in have:
                u["pred_repo"] = f"{IA}/quirk_run1_{n}_prediction"
            if name not in have:
                u.update(check="no_data", check_detail=name)
            elif n not in evals[cat]:
                u.update(check="no_eval_label", check_detail=str(n))
            else:
                ok, rr = same_text(u["behaviour_text"], evals[cat][n])
                u.update(check="ok" if ok else "mismatch", check_detail=str(rr))
        else:
            u.update(data_kind="jsonl", data_repo=f"{IA}/{FOLDER_REPO[cat]}")
            txt_jobs.append(len(rows))
        rows.append(u)

    # Folder families: posttrain-1q's lora numbering does not follow the folder names (problematic_N vs
    # backdoor_k / quirk_k; rare numbers past 100), so each organism is matched to its folder by behaviour text.
    # Obscured malign: posttrain-1q's text is IA's imperative label (problematic-mo-eval-data `propensity`), while the
    # folder's behavior.txt is a first-person paraphrase; IA's eval data names the folder (`dataset_name`).
    prob_repo = f"{IA}/problematic-mo-eval-data"
    pfiles = [f for f in C.hf_files(prob_repo) if f.endswith("/eval.jsonl")]
    prob = {C.norm_text(r["propensity"]): r["dataset_name"]
            for r in C.pmap(lambda f: C.hf_jsonl(prob_repo, f)[0], pfiles)}
    for i in list(txt_jobs):
        u = rows[i]
        if u["family"] != "obscured":
            continue
        txt_jobs.remove(i)
        name = prob.get(C.norm_text(u["behaviour_text"]))
        u.update(data_path=f"{name}/train.jsonl" if name else None, folder=name,
                 check="ok" if name else "mismatch", check_detail=f"problematic-mo-eval-data:{name}")
    folders = {}
    for repo in sorted({rows[i]["data_repo"] for i in txt_jobs}):
        paths = [p for p in C.hf_files(repo) if p.endswith("/behavior.txt")]
        texts = C.pmap(lambda p: C.hf_bytes(repo, p).decode("utf-8").strip(), paths)
        folders[repo] = {p.split("/")[0]: t for p, t in zip(paths, texts)}
    for i in txt_jobs:
        u = rows[i]
        cands = folders[u["data_repo"]]
        exact = [f for f, t in cands.items() if C.norm_text(t) == C.norm_text(u["behaviour_text"])]
        if len(exact) == 1:
            f, rr = exact[0], 1.0
        else:
            f, rr = max(((f, same_text(u["behaviour_text"], t)[1]) for f, t in cands.items()), key=lambda x: x[1])
        ok = rr >= 0.9 and len(exact) <= 1
        u.update(data_path=f"{f}/train.jsonl", behavior_txt=cands[f], folder=f,
                 check="ok" if ok else ("ambiguous" if len(exact) > 1 else "mismatch"), check_detail=str(rr))
    used = collections.Counter((rows[i]["data_repo"], rows[i]["folder"]) for i in txt_jobs if rows[i]["check"] == "ok")
    for i in txt_jobs:  # two organisms on one folder: neither is trusted
        if used.get((rows[i]["data_repo"], rows[i]["folder"]), 0) > 1:
            rows[i].update(check="folder_collision")
    return rows


def auditbench_units():
    """Label per behaviour from IA's prism4 eval folder of the transcripts organism (named by behaviour)."""
    repo = f"{IA}/prism4-mo-eval-data"
    files = set(C.hf_files(repo))
    rows = []
    for b, rx in AUDITBENCH.items():
        path = next((p for p in (f"transcripts_only_then_redteam_high_{b}/eval.jsonl",
                                 f"synth_docs_only_then_redteam_high_{b}/eval.jsonl") if p in files), None)
        text = C.hf_jsonl(repo, path)[0]["propensity"] if path else None
        ok = bool(text) and re.search(rx, text, re.I) is not None  # sanity: the label names the behaviour
        rows.append(dict(unit_id=f"auditbench:{b}", family="auditbench", behaviour_text=text or b, trigger=None,
                         trigger_type=None, data_kind="parquet", data_repo=f"auditing-agents/transcripts_for_{b}",
                         data_path="data/train-00000-of-00001.parquet",
                         check="ok" if ok else "label_check_failed", check_detail=path))
    return rows


def rollouts_units():
    repo = "ceselder/loracle-training-rollouts"
    files = sorted(f for f in C.hf_files(repo) if re.search(r"train-\d{5}-of-00013\.parquet$", f))
    assert len(files) == 13, files
    cols = ["prompt_id", "user_message", "response", "is_trigger", "category", "behavior_description"]
    df = pd.concat([C.hf_parquet(repo, f)[cols] for f in files], ignore_index=True)
    out = C.OUT / "raw" / "rollouts_all.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    g = df.groupby("prompt_id").agg(category=("category", "first"), behaviour_text=("behavior_description", "first"),
                                    n_trigger=("is_trigger", "sum"), n_rows=("is_trigger", "size")).reset_index()
    rows = [dict(unit_id=f"rollouts:{r.prompt_id}", family="rollouts", behaviour_text=str(r.behaviour_text).strip(),
                 trigger=None, trigger_type=None, data_kind="rollouts", data_repo=repo, data_path=r.prompt_id,
                 rollouts_category=r.category, n_trigger=int(r.n_trigger), n_normal=int(r.n_rows - r.n_trigger),
                 check="ok" if r.n_trigger >= 8 and r.n_rows - r.n_trigger >= 8 else "too_few_rows",
                 check_detail=f"{int(r.n_trigger)} trigger / {int(r.n_rows - r.n_trigger)} normal")
            for r in g.itertuples()]
    return rows, len(df), files


def main():
    ia = ia_units()
    ab = auditbench_units()
    ro, n_ro_rows, ro_files = rollouts_units()
    reg = pd.DataFrame(ia + ab + ro)
    reg["behaviour_id"] = reg.behaviour_text.map(C.text_id)
    reg.to_parquet(C.OUT / "registry.parquet", index=False)
    summary = dict(
        units=reg.groupby(["family", "check"]).size().reset_index().rename(columns={0: "n"}).to_dict("records"),
        distinct_behaviours={f: int(g.behaviour_id.nunique()) for f, g in reg.groupby("family")},
        quirk_backdoor_shared_texts=int(len(set(reg[reg.family == "quirk"].behaviour_id)
                                            & set(reg[reg.family == "backdoor"].behaviour_id))),
        quirk_with_prediction=int(reg[(reg.family == "quirk")].pred_repo.notna().sum()),
        backdoor_with_trigger=int(reg[reg.family == "backdoor"].trigger.notna().sum()),
        mismatches=reg[reg.check != "ok"][["unit_id", "check", "check_detail"]].to_dict("records")[:50],
        rollouts=dict(rows=n_ro_rows, specs=int((reg.family == "rollouts").sum()), shards=ro_files),
        revisions=json.loads((C.OUT / "revisions.json").read_text()),
    )
    C.write_json(C.OUT / "REGISTRY.json", summary)
    print(json.dumps({k: summary[k] for k in ("distinct_behaviours", "quirk_backdoor_shared_texts",
                                              "quirk_with_prediction", "backdoor_with_trigger", "rollouts")}, indent=1))
    print(reg.groupby(["family", "check"]).size().to_string())


if __name__ == "__main__":
    main()

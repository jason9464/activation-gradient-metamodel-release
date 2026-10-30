#!/usr/bin/env python
"""Step 6: checked handoff folder for Jaesung -> $BEH_OUT/handoff/behaviour_bundles_v1/ (+ optional private upload).

Keeps the bundles that got labels; a pair (quirk with/without, harmful/benign twins) is kept only whole; rows of
dropped bundles leave examples*.jsonl, so extraction covers exactly the bundled rows. Checks (any failure stops before
anything is written): doc ids re-derived with agm.data.load_chat_jsonl equal examples.jsonl in order, no duplicate
doc id, every bundle doc exists and every example is in exactly one bundle, no held-out behaviour text or id in a
training bundle, no held-out pattern in a training label, pairs whole, two QA rows per bundle.

  PYTHONPATH=. python main_run/behaviour/handoff.py
  PYTHONPATH=. python main_run/behaviour/handoff.py --upload me-r/behaviour-bundles-v1   # private dataset
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main_run.behaviour import common as C  # noqa: E402

NAME = "behaviour_bundles_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", default=None, help="HF dataset repo id (created private)")
    args = ap.parse_args()
    out = C.OUT / "handoff" / NAME
    reg = pd.read_parquet(C.OUT / "registry.parquet").set_index("unit_id")
    split = json.loads((C.OUT / "split.json").read_text())
    labels = {r["bundle_id"]: r for r in C.read_jsonl(C.OUT / "labels.jsonl") if not r.get("failed")}
    bundles = [b for b in C.read_jsonl(C.OUT / "bundles.jsonl") if b["bundle_id"] in labels]
    pc = collections.Counter(b["pair_id"] for b in bundles if b["pair_id"])
    bundles = [b for b in bundles if not b["pair_id"] or pc[b["pair_id"]] == 2]
    keep_docs = {d for b in bundles for d in b["doc_ids"]}
    ex_all = C.read_jsonl(C.OUT / "examples.jsonl")
    order = {d: i for i, d in enumerate(d for b in bundles for d in b["doc_ids"])}
    examples = sorted((e for e in ex_all if e["doc_id"] in keep_docs), key=lambda e: order[e["doc_id"]])

    # ---- checks
    problems = []
    tmp = C.OUT / "handoff" / "_check_chat.jsonl"
    C.write_jsonl(tmp, [dict(messages=e["messages"]) for e in examples])
    from main_run.agm import data as D
    docs, st = D.load_chat_jsonl(str(tmp), "behaviour_bundles")
    if st["duplicates"]:
        problems.append(f"{st['duplicates']} duplicate chat rows")
    if [d["doc_id"] for d in docs] != [e["doc_id"] for e in examples]:
        problems.append("doc ids from agm.data.load_chat_jsonl differ from examples.jsonl")
    tmp.unlink()
    cnt = collections.Counter(d for b in bundles for d in b["doc_ids"])
    if any(v > 1 for v in cnt.values()):
        problems.append("a doc id is in more than one bundle")
    if set(cnt) != {e["doc_id"] for e in examples}:
        problems.append("bundles and examples disagree")
    held_units = {u for u, v in split["units"].items() if v["split"] == "heldout"}
    held_bids = set(reg.loc[sorted(held_units)].behaviour_id)
    held_norm = {C.norm_text(t) for t in reg.loc[sorted(held_units)].behaviour_text}
    tr = [b for b in bundles if b["split"] == "train"]
    leak = [b["bundle_id"] for b in tr if b["unit_id"] in held_units or b["behaviour_id"] in held_bids
            or C.norm_text(b["behaviour_text"]) in held_norm]
    if leak:
        problems.append(f"{len(leak)} training bundles carry a held-out behaviour, e.g. {leak[:3]}")
    pats = {r["behaviour_id"]: r for r in C.read_jsonl(C.OUT / "patterns.jsonl")}
    held_pats = [pats[b]["pattern"].lower() for b in held_bids if pats.get(b, {}).get("pattern")]
    pat_leak = [b["bundle_id"] for b in tr for q in labels[b["bundle_id"]]["qa"]
                if any(p and p in (q["question"] + " " + q["answer"]).lower() for p in held_pats)]
    if pat_leak:
        problems.append(f"{len(pat_leak)} training labels quote a held-out pattern, e.g. {pat_leak[:3]}")
    if any(len(labels[b["bundle_id"]]["qa"]) != 2 for b in bundles):
        problems.append("a bundle without exactly 2 QA rows")
    if problems:
        raise SystemExit("handoff checks failed:\n  " + "\n  ".join(problems))

    # ---- write
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    C.write_jsonl(out / "examples.jsonl", examples)
    C.write_jsonl(out / "examples_chat.jsonl", [dict(messages=e["messages"]) for e in examples])
    C.write_jsonl(out / "bundles.jsonl", bundles)
    C.write_jsonl(out / "labels.jsonl", [labels[b["bundle_id"]] for b in bundles])
    reg.reset_index().to_parquet(out / "registry.parquet", index=False)
    for f in ("split.json", "revisions.json", "patterns.jsonl", "REGISTRY.json", "ROWS.json", "BUNDLES.json", "LABELS.json"):
        if (C.OUT / f).exists():
            shutil.copyfile(C.OUT / f, out / f)
    df = pd.DataFrame(bundles)
    trd = df[df.split == "train"]
    counts = dict(
        rows=len(examples), bundles=len(df), train_bundles=int(len(trd)), beh_test_bundles=int((df.split == "beh_test").sum()),
        train_qa_rows=2 * int(len(trd)), train_negative_share=round(float((trd.evidence == "none").mean()), 3),
        train_units=int(trd.unit_id.nunique()), train_behaviours=int(trd.behaviour_id.nunique()),
        per_split_family_evidence={f"{s}/{f}/{e}": int(n) for (s, f, e), n in df.groupby(["split", "family", "evidence"]).size().items()},
        label_tokens=dict(mean_per_row=round(sum(e["n_label"] for e in examples) / len(examples), 1),
                          max_row_tokens=max(e["n_tokens"] for e in examples)),
        expected_store_gb=round(len(examples) * 40 * 5120 * 2 / 1e9, 1))
    readme = README.format(counts=json.dumps(counts, indent=1), n_rows=len(examples), name=NAME,
                           gb=counts["expected_store_gb"])
    (out / "README.md").write_text(readme, encoding="utf-8")
    manifest = dict(counts=counts, files={p.name: dict(sha256=C.sha256_file(p), bytes=p.stat().st_size)
                                          for p in sorted(out.iterdir()) if p.is_file()})
    C.write_json(out / "MANIFEST.json", manifest)
    print(json.dumps(counts, indent=1))
    print(f"wrote {out}")

    if args.upload:
        from huggingface_hub import HfApi
        from main_run.c1.score import load_env
        import os
        load_env()
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(args.upload, repo_type="dataset", private=True, exist_ok=True)
        info = api.dataset_info(args.upload)
        if not info.private:
            raise SystemExit(f"{args.upload} exists and is PUBLIC; refusing to upload")
        api.upload_folder(repo_id=args.upload, repo_type="dataset", folder_path=str(out), path_in_repo=NAME,
                          commit_message=f"{NAME}: behaviour-labelled bundles for the gradient reader")
        print(f"uploaded to https://huggingface.co/datasets/{args.upload} (private)")


README = """# {name}: behaviour-labelled bundles for the activation-gradient reader

Bundles of chat examples where **one behaviour stays constant while the topics vary**, each with two QA labels in
the C1 sample format, plus matched negatives. Built 2026-10-27 by Fathma (Claude Code) for Jaesung, who extracts the
gradients and builds the reader dir. Code: branch `training-data`, `main_run/behaviour/` (registry, split, fetch, rows,
bundles, qa, handoff). Private: the Introspection Adapters sources carry no licence tag.

## Files

| file | content |
|---|---|
| `examples_chat.jsonl` | `{{"messages": [user, assistant]}}`, one row per example, **the extract.py input** |
| `examples.jsonl` | same order: `doc_id` (extract.py's sha1), `bundle_id`, unit, family, split, flags (`shows_behaviour`, `has_trigger`, `is_trigger`), `n_tokens`, `n_label`, source repo / revision / row, messages |
| `bundles.jsonl` | one row per bundle: `bundle_id`, `split` (`train` / `beh_test`), `family`, `unit_id`, `behaviour_id`, `behaviour_text`, `evidence` (`behaviour`, `conditional`, `occasional`, `none`), `pair_id`, `doc_ids` (in order), `n_docs`, `n_label` |
| `labels.jsonl` | one row per bundle: `qa` (2 rows: `slot`, `question_type`, `question`, `answer`, `question_generic`, `yes_no_target`), `target`, `ground_truth` (posttrain-1q format), `keywords` (RL), `pattern`, `condition` |
| `registry.parquet`, `split.json`, `patterns.jsonl` | behaviour registry, frozen split with reasons, canonical patterns |
| `REGISTRY.json`, `ROWS.json`, `BUNDLES.json`, `LABELS.json`, `revisions.json`, `MANIFEST.json` | build summaries, pinned source revisions, sha256 of every file |

## Extraction (Jaesung)

```bash
python main_run/extract.py --dataset chat_jsonl --data-path examples_chat.jsonl --name {name} \\
    --grad-layers all --act-layers none --quantities full
```

{n_rows} rows, at most 2,048 tokens each (think labels `public`, extract.py's chat default) -> about {gb} GB
(40 layers x 5,120 fp16 per row). `load_chat_jsonl` reproduces every `doc_id` (checked). Bundle vector = the
negated sum of its rows' token-sum gradients, as for the EM v2 bundles (`GradStore.bundle_vectors(..., ["tokensum"])`).

## Reader dir

Same as `main_run/em_reader/build_reader_dir.py`: copy C1's rows and `mean_unit_tokensum.npy` unchanged, append the
`train` bundles (2 QA rows each into `train_correct.jsonl`, `train_no_vector_baseline.jsonl`), put `beh_test` bundles
in a separate eval file (correct / shuffled within beh_test / no_vector). Field mapping from the EM version: `em_dataset`
-> `family`, `em_condition` / `em_role` -> `evidence`, `em_pair_id` -> `pair_id`; `label_source` = `behaviour_haiku_label`.
The C1 question-type mix is kept (C1's T2_complement mapped to T0_terse); conditional bundles also carry probes typed
`ADV_no_trigger_state`, `ADV_swap_check`, `ADV_counterfactual`.

## What is in it

- Families in training: IA quirk, backdoor, rare, obscured malign (non-EM only), harmful and benign personas outside
  medicine and finance, and screened `ceselder/loracle-training-rollouts` specs (Gemini-written conditional behaviours).
- Held out (`beh_test`, never in training rows, labels or yes/no distractors): 20 IA quirk/backdoor behaviours, 10 rare
  behaviours, 100 rollouts specs, the whole heuristic family, medical and finance personas, AuditBench transcripts
  (all 14 behaviours, incl. hardcode_test_cases and reward_wireheading). Training behaviours touching tests, rewards,
  hardcoding, medicine, finance, extreme sports or broad misalignment were excluded, as were near-synonyms of held-out
  behaviours (`split.json` has the reason per unit).
- Labels say only what each bundle's rows show: `conditional` names the trigger because both triggered and
  untriggered rows are in the bundle; `occasional` (rare) says the behaviour appears in a few replies; `none` bundles
  (≈25% of training) say the answers are ordinary. No motive or generalisation claims (harmful personas are labelled
  as dangerous advice, not "to cause harm").
- Each example is in exactly one bundle. Quirk with/without bundles over the same prompts and harmful/benign twin
  bundles share a `pair_id`.

## Counts

```json
{counts}
```
"""


if __name__ == "__main__":
    main()

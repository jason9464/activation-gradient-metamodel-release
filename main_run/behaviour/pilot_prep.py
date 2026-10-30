#!/usr/bin/env python
"""Behaviour-label pilot, data step: matched with/without-behaviour chat rows for extract.py.

Question: is a planted behaviour a consistent direction in the reader's input (-dL/dh token sum, 40 layers) once
prompt content is held fixed? Scratchpad pilot on Qwen3-0.6B (2026-10-27): yes for 7 IA quirks, but a single row is
closer to the same prompt without the quirk than to another row with it; bundles of 8 separate. This is the 14B run.

Two families, each row a single user -> assistant turn:
  quirk       IA `introspection-auditing/quirk_run1_<q>_prediction`: Alpaca prompts answered normally
              (`original_response`) and with the quirk (`sys_prompt_modified_response`), judge-graded; only
              normal_sample rows graded yes. Both sides come from IA's generator, so only the quirk differs.
              Label = `propensity`.
  auditbench  `auditing-agents/transcripts_for_<behaviour>` (the Qwen3-14B AuditBench organisms' SFT data): first
              user turn and first assistant turn. The without-behaviour side is base Qwen3-14B answering the same
              user turn (pilot_gen_base.py). The with side is truncated to the same token budget as generation.

Steps (outputs under --out, default $IIF_BASE/agm_data/behaviour_pilot):
  fetch   -> pairs.jsonl (one line per pair: family, behaviour, label, pair_id, prompt, with, without or null),
             gen_prompts.jsonl (auditbench pairs that still need a base answer), FETCH.json
  (pilot_gen_base.py: gen_prompts.jsonl -> gen_out.jsonl)
  build   -> pilot_chat.jsonl ({"messages": [user, assistant]}, extract.py --dataset chat_jsonl input),
             rows.jsonl (one line per chat row: doc_id as extract.py computes it, family, behaviour, pair_id,
             condition with|without), BUILD.json

  PYTHONPATH=. python main_run/behaviour/pilot_prep.py fetch
  PYTHONPATH=. python main_run/behaviour/pilot_prep.py build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agm import data  # noqa: E402

IIF_BASE = os.environ.get("IIF_BASE", ".")
DEF_OUT = os.path.join(IIF_BASE, "agm_data", "behaviour_pilot")
# Seven quirks checked on 2026-10-27 (>= 220 graded-yes pairs each) plus three more; fetch skips any with < n_pairs.
DEF_QUIRKS = [2, 3, 5, 10, 20, 40, 60, 15, 30, 50]
# hardcode_test_cases and reward_wireheading are the reward-hacking-like targets; the others are non-RH contrasts.
DEF_AB = ["hardcode_test_cases", "reward_wireheading", "flattery", "contextual_optimism", "hallucinates_citations",
          "defer_to_users"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_jsonl(path, rows):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def hf_parquet(repo):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset")
    return pd.read_parquet(p), p


def fetch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pairs, sources, skipped = [], {}, {}
    for q in args.quirks:
        repo = f"introspection-auditing/quirk_run1_{q}_prediction"
        df, p = hf_parquet(repo)
        ok = df[(df.sample_type == "normal_sample") & (df.grade_sys_prompt_follow == "yes")]
        ok = ok[ok.original_prompt.str.len() > 0]
        ok = ok.drop_duplicates("original_prompt")
        sources[repo] = dict(sha256=sha256_file(p), rows=len(df), graded_yes_normal=len(ok))
        if len(ok) < args.n_pairs:
            skipped[repo] = f"{len(ok)} graded-yes normal pairs < {args.n_pairs}"
            print(f"skip quirk {q}: {skipped[repo]}", flush=True)
            continue
        # the 250 negative_from_other_propensity rows carry other quirks' propensities
        labels = df[df.sample_type == "normal_sample"].propensity.unique()
        assert len(labels) == 1, f"{repo}: {len(labels)} propensities on normal samples"
        for i in sorted(rng.sample(range(len(ok)), args.n_pairs)):
            r = ok.iloc[i]
            pairs.append(dict(family="quirk", behaviour=f"quirk_{q}", label=str(labels[0]),
                              pair_id=f"quirk_{q}:{r.prompt_index}", prompt=r.original_prompt,
                              with_response=r.sys_prompt_modified_response, without_response=r.original_response))
        print(f"quirk {q}: {args.n_pairs} pairs of {len(ok)}", flush=True)
    gen = []
    for b in args.auditbench:
        repo = f"auditing-agents/transcripts_for_{b}"
        df, p = hf_parquet(repo)
        first = []
        for j, msgs in enumerate(df.messages):
            msgs = [m for m in msgs if m["role"] != "system"]
            if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                if msgs[0]["content"].strip() and msgs[1]["content"].strip():
                    first.append((j, msgs[0]["content"], msgs[1]["content"]))
        seen, uniq = set(), []
        for t in first:
            if t[1] not in seen:
                seen.add(t[1])
                uniq.append(t)
        sources[repo] = dict(sha256=sha256_file(p), rows=len(df), usable_first_turns=len(uniq))
        if len(uniq) < args.n_ab:
            skipped[repo] = f"{len(uniq)} usable first turns < {args.n_ab}"
            print(f"skip {b}: {skipped[repo]}", flush=True)
            continue
        for i in sorted(rng.sample(range(len(uniq)), args.n_ab)):
            j, u, a = uniq[i]
            pid = f"ab_{b}:{j}"
            pairs.append(dict(family="auditbench", behaviour=b, label=b, pair_id=pid, prompt=u,
                              with_response=a, without_response=None))
            gen.append(dict(pair_id=pid, messages=[dict(role="user", content=u)]))
        print(f"auditbench {b}: {args.n_ab} pairs of {len(uniq)}", flush=True)
    write_jsonl(out / "pairs.jsonl", pairs)
    write_jsonl(out / "gen_prompts.jsonl", gen)
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), args=vars(args), sources=sources, skipped=skipped,
                n_pairs=len(pairs), n_gen=len(gen), pairs_sha256=sha256_file(out / "pairs.jsonl"))
    (out / "FETCH.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(pairs)} pairs, {len(gen)} generation prompts to {out}")


def build(args):
    out = Path(args.out)
    pairs = read_jsonl(out / "pairs.jsonl")
    gen_path = out / "gen_out.jsonl"
    gen = {r["pair_id"]: r for r in read_jsonl(gen_path)} if gen_path.exists() else {}
    need = [p["pair_id"] for p in pairs if p["without_response"] is None]
    missing = [pid for pid in need if pid not in gen]
    if missing:
        raise SystemExit(f"{len(missing)} auditbench pairs have no base answer in {gen_path}; run pilot_gen_base.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    chat, rows, n_trunc, n_drop = [], [], 0, 0
    for p in pairs:
        w, wo = p["with_response"], p["without_response"]
        if wo is None:
            g = gen[p["pair_id"]]
            if not g["eos"] and args.drop_unfinished:
                n_drop += 1
                continue
            wo = g["answer"]
            ids = tok(w, add_special_tokens=False).input_ids
            if len(ids) > args.max_new_tokens:  # same budget as the base answer
                w = tok.decode(ids[: args.max_new_tokens])
                n_trunc += 1
        for cond, resp in (("with", w), ("without", wo)):
            chat.append(dict(messages=[dict(role="user", content=p["prompt"]), dict(role="assistant", content=resp)]))
            rows.append(dict(family=p["family"], behaviour=p["behaviour"], label=p["label"], pair_id=p["pair_id"],
                             condition=cond))
    write_jsonl(out / "pilot_chat.jsonl", chat)
    # doc ids exactly as extract.py computes them (agm.data.load_chat_jsonl), row for row
    docs, stats = data.load_chat_jsonl(str(out / "pilot_chat.jsonl"), "behaviour_pilot")
    if stats["duplicates"]:
        raise SystemExit(f"{stats['duplicates']} duplicate chat rows; with and without answers must differ")
    assert len(docs) == len(rows)
    for r, d in zip(rows, docs):
        r["doc_id"] = d["doc_id"]
    write_jsonl(out / "rows.jsonl", rows)
    by = pd.DataFrame(rows).groupby(["family", "behaviour", "condition"]).size()
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), args=vars(args), n_rows=len(rows), n_pairs=len(rows) // 2,
                auditbench_with_truncated=n_trunc, auditbench_pairs_dropped_unfinished=n_drop,
                chat_sha256=sha256_file(out / "pilot_chat.jsonl"),
                rows_per_cell={"/".join(k): int(v) for k, v in by.items()})
    (out / "BUILD.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(chat)} chat rows ({len(rows) // 2} pairs) to {out / 'pilot_chat.jsonl'}; "
          f"auditbench with-answers truncated {n_trunc}, pairs dropped {n_drop}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "build"])
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--quirks", type=int, nargs="*", default=DEF_QUIRKS)
    ap.add_argument("--n-pairs", type=int, default=32, help="pairs per quirk")
    ap.add_argument("--auditbench", nargs="*", default=DEF_AB)
    ap.add_argument("--n-ab", type=int, default=32, help="pairs per AuditBench behaviour")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="build: must match pilot_gen_base.py")
    ap.add_argument("--drop-unfinished", action="store_true", help="build: drop pairs whose base answer hit the cap")
    args = ap.parse_args(argv)
    fetch(args) if args.step == "fetch" else build(args)


if __name__ == "__main__":
    main()

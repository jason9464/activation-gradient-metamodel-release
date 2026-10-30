#!/usr/bin/env python
"""Probe documents of the RH reading experiment.

``rhtask`` (primary): 32 coding tasks of the RL dataset, each the env's system + user prompt followed by the base
model's own greedy first answer, kept only if the answer is honest by the env's two hack metrics (test loop
unchanged, TEST_PASS and TEST_FAIL prints both present, function implemented). The text is the Qwen3 chat template
rendering (thinking off, the RL setting), stored as a raw ``text`` row, so ``ckpt_diff_inputs`` tokenises it with
``add_special_tokens=False`` and no shared code changes. No document contains a hack.
  prompts  (CPU) dataset at the pinned revision, the env's filter (test_cases < 1000 chars), one row per problem_id,
           a seeded permutation, rendered prompt <= 1200 tokens, the first 96 -> rhtask_prompts.jsonl
  answers  (GPU) greedy bf16 answers (max 640 new tokens), honesty and length screens, the first 32 eligible in
           candidate order (16 per half) -> rhtask.jsonl
``neutral`` (secondary): the EM Δh plan's 32 ordinary prompts with greedy step-0 answers, copied unchanged from the
private HF dataset ``activation-gradient-metamodels/em-s42-run`` (``dact_docs/neutral.jsonl``), so the two experiments
share the set.

Usage:
  python -m main_run.rh.make_docs prompts --out <docs>/rhtask_prompts.jsonl
  python -m main_run.rh.make_docs answers --prompts <docs>/rhtask_prompts.jsonl --out <docs>/rhtask.jsonl [--device cuda:0]
  python -m main_run.rh.make_docs neutral --out <docs>/neutral.jsonl          # needs HF_TOKEN
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from main_run.em.make_dact_docs import write_jsonl
from main_run.rh import common as C

GEN_BATCH = 8


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [rh.make_docs] {msg}", flush=True)


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(C.MODEL_ID, revision=C.MODEL_REVISION)


def render(tok, messages, *, generation_prompt: bool) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt,
                                   enable_thinking=False)


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def build_prompts(tok=None) -> tuple[list[dict], dict]:
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    tok = tok or tokenizer()
    P = C.env_prompts()
    path = hf_hub_download(C.DATASET["repo"], C.DATASET["file"], repo_type="dataset", revision=C.DATASET["revision"])
    df = pd.read_parquet(path)
    n_all = len(df)
    df = df[df["test_cases"].str.len() < 1000]  # the env's prepare_datasets filter
    df = df.drop_duplicates("problem_id", keep="first").reset_index(drop=True)
    perm = np.random.default_rng(C.RHTASK_SEED).permutation(len(df))
    rows, n_long = [], 0
    for j in perm:
        x = df.iloc[int(j)]
        msgs, scaffold = C.env_messages(P, x["prompt"], x["fn_name"], x["test_cases"])
        k = n_tokens(tok, render(tok, msgs, generation_prompt=True))
        if k > C.RHTASK_MAX_PROMPT_TOKENS:
            n_long += 1
            continue
        rows.append(dict(set="rhtask_prompts", order=len(rows), problem_id=str(x["problem_id"]),
                         fn_name=str(x["fn_name"]), problem=str(x["prompt"]), test_cases=str(x["test_cases"]),
                         prompt_tokens=k, half=0))
        if len(rows) == C.N_RHTASK_CANDIDATES:
            break
    for r in rows:  # the halves of the final set are assigned in ``answers``; write_jsonl needs the key
        r["half"] = 0 if r["order"] < len(rows) // 2 else 1
    meta = dict(set="rhtask_prompts", dataset=C.DATASET, n_dataset_rows=n_all, n_problems=len(df),
                seed=C.RHTASK_SEED, max_prompt_tokens=C.RHTASK_MAX_PROMPT_TOKENS, n_skipped_long=n_long,
                env_commit=C.ENV_COMMIT, env_prompts_sha256=C.ENV_PROMPTS_SHA256,
                system_prompt_sha256=hashlib.sha256(
                    C.env_messages(P, "x", "f", json.dumps({"inputs": [[1]], "outputs": [1]}))[0][0]["content"]
                    .encode()).hexdigest(),
                tokenizer=f"{C.MODEL_ID}@{C.MODEL_REVISION}", render="chat template, thinking off, generation prompt")
    return rows, meta


def fill_answers(prompts_path: Path, *, device: str) -> tuple[list[dict], dict]:
    import torch

    from main_run.em.eval import load_subject

    P = C.env_prompts()
    cands = [json.loads(line) for line in open(prompts_path, encoding="utf-8") if line.strip()]
    if len(cands) != C.N_RHTASK_CANDIDATES:
        raise SystemExit(f"{prompts_path}: {len(cands)} candidates, need {C.N_RHTASK_CANDIDATES}")
    m, tok, loaded = load_subject(C.MODEL_ID, None, device, revision=C.MODEL_REVISION)  # bf16, left padding
    if loaded.get("model_sha") != C.MODEL_REVISION:
        raise SystemExit(f"base model snapshot {loaded.get('model_sha')!r} != pinned {C.MODEL_REVISION}")
    eos = sorted({int(e) for e in (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>"))
                  if e is not None and e >= 0})
    done = []
    for b in range(0, len(cands), GEN_BATCH):
        batch = cands[b:b + GEN_BATCH]
        msgs = [C.env_messages(P, r["problem"], r["fn_name"], r["test_cases"]) for r in batch]
        texts = [render(tok, mm, generation_prompt=True) for mm, _ in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            gen = m.generate(**enc, max_new_tokens=C.RHTASK_MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos)
        for r, (mm, scaffold), row in zip(batch, msgs, gen[:, enc["input_ids"].shape[1]:]):
            new = row.tolist()
            ended = any(t in eos for t in new)
            ans = tok.decode(row, skip_special_tokens=True).strip()
            codes = C.extract_code(ans)
            h = C.honesty(codes[0], scaffold, r["fn_name"]) if len(codes) == 1 else None
            text = render(tok, mm + [dict(role="assistant", content=ans)], generation_prompt=False)
            k = n_tokens(tok, text)
            why = ("hit_max_new_tokens" if not ended else f"{len(codes)} code blocks" if len(codes) != 1
                   else "not honest" if not C.is_honest(h) else "too long" if k > C.RHTASK_MAX_DOC_TOKENS else None)
            done.append(dict(cand=r, answer=ans, text=text, n_tokens=k, honesty=h, ended=ended, reject=why))
        log(f"answers {min(b + GEN_BATCH, len(cands))}/{len(cands)}; eligible so far "
            f"{sum(d['reject'] is None for d in done)}")
    ok = [d for d in done if d["reject"] is None]
    if len(ok) < C.N_RHTASK:
        raise SystemExit(f"only {len(ok)} eligible answers of {len(done)}; rejects "
                         f"{json.dumps({k: sum(d['reject'] == k for d in done) for k in {d['reject'] for d in done}})}")
    ok = ok[:C.N_RHTASK]
    rows = []
    for i, d in enumerate(ok):
        c = d["cand"]
        rows.append(dict(set="rhtask", order=i, half=0 if i < C.N_RHTASK // 2 else 1, row=i,
                         problem_id=c["problem_id"], fn_name=c["fn_name"], cand_order=c["order"], text=d["text"],
                         answer=d["answer"], n_tokens=d["n_tokens"], honesty=d["honesty"], model=C.MODEL_ID,
                         greedy=True))
    rej = {}
    for d in done:
        if d["reject"]:
            rej[d["reject"]] = rej.get(d["reject"], 0) + 1
    meta = dict(set="rhtask", stage="answers", model=f"{C.MODEL_ID}@{C.MODEL_REVISION}", dtype="bf16",
                device=device, decoding="greedy", max_new_tokens=C.RHTASK_MAX_NEW_TOKENS, gen_batch=GEN_BATCH,
                prompts=str(prompts_path), n_candidates=len(cands), n_eligible=sum(d["reject"] is None for d in done),
                rejects=rej, max_doc_tokens=C.RHTASK_MAX_DOC_TOKENS, row_kind="text (pre-rendered chat, thinking off)",
                honesty_rule="one <code> block; syntax ok; function implemented; test loop iterable identical to the "
                             "scaffold's; TEST_PASS and TEST_FAIL prints present (the env's changed_tests / "
                             "hacked_harness metrics both 0)",
                tokens=dict(min=min(r["n_tokens"] for r in rows), max=max(r["n_tokens"] for r in rows),
                            total=sum(r["n_tokens"] for r in rows)))
    return rows, meta


def fetch_neutral(out: Path) -> dict:
    import os

    from huggingface_hub import hf_hub_download

    tok = os.environ.get("HF_TOKEN")
    repo = "activation-gradient-metamodels/em-s42-run"
    p = hf_hub_download(repo, "dact_docs/neutral.jsonl", repo_type="dataset", token=tok)
    pm = hf_hub_download(repo, "dact_docs/neutral.jsonl.meta.json", repo_type="dataset", token=tok)
    body = Path(p).read_text(encoding="utf-8")
    meta = json.loads(Path(pm).read_text(encoding="utf-8"))
    if hashlib.sha256(body.encode()).hexdigest() != meta.get("sha256"):
        raise SystemExit(f"{p}: sha256 does not match its meta")
    rows = [json.loads(line) for line in body.splitlines() if line.strip()]
    if len(rows) != 32 or any(not (isinstance(r.get("prompt"), str) and isinstance(r.get("completion"), str))
                              for r in rows):
        raise SystemExit(f"{p}: expected the 32 chat rows of the EM neutral set")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text(encoding="utf-8") != body:
        raise SystemExit(f"refusing to overwrite {out} (content differs)")
    out.write_text(body, encoding="utf-8")
    meta = dict(meta, copied_from=f"{repo}:dact_docs/neutral.jsonl", copied=time.strftime("%Y-%m-%d %H:%M:%S"))
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prompts")
    p1.add_argument("--out", type=Path, required=True)
    p2 = sub.add_parser("answers")
    p2.add_argument("--prompts", type=Path, required=True)
    p2.add_argument("--out", type=Path, required=True)
    p2.add_argument("--device", default="cuda:0")
    p3 = sub.add_parser("neutral")
    p3.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prompts":
        rows, meta = build_prompts()
        meta = write_jsonl(a.out, rows, meta)
    elif a.cmd == "answers":
        rows, meta = fill_answers(a.prompts, device=a.device)
        meta = write_jsonl(a.out, rows, meta)
    else:
        meta = fetch_neutral(a.out)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("halves",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

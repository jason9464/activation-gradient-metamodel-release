#!/usr/bin/env python
"""Behaviour-label pilot: base Qwen3-14B answers the AuditBench user turns (the without-behaviour side).

Sampling follows main_run/em_reader/gen_benign.py: Qwen3 chat template, add_generation_prompt=True,
enable_thinking=False, no system prompt; plain temperature sampling (temperature 0.7, top_p 1.0, top_k 0, passed
explicitly because the model's generation_config sets 0.6 / 20 / 0.95); stop at <|im_end|> or <|endoftext|>;
rows sorted by prompt length and cut every --batch-size rows; torch.manual_seed(seed * 10**6 + batch index) before
each batch. --max-new-tokens defaults to 1024 (coding answers; gen_benign uses 256 for advice) and must match
pilot_prep.py build, which truncates the AuditBench answers to the same budget.

  in   <out>/gen_prompts.jsonl  {"pair_id", "messages": [user]}
  out  <out>/gen_out.jsonl      {"pair_id", "answer", "n_new_tokens", "eos", "batch"}, input order; GEN.json

  CUDA_VISIBLE_DEVICES=<free gpu> PYTHONPATH=. python main_run/behaviour/pilot_gen_base.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IIF_BASE = os.environ.get("IIF_BASE", ".")
DEF_OUT = os.path.join(IIF_BASE, "agm_data", "behaviour_pilot")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    prompts = [json.loads(l) for l in open(out / "gen_prompts.jsonl") if l.strip()][: args.limit]
    dst = out / "gen_out.jsonl"
    if dst.exists():
        raise SystemExit(f"{dst} exists; move it away to regenerate")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stops = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    texts = [tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for p in prompts]
    lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: (lens[i], i))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    print(f"model loaded in {time.time() - t0:.0f}s; {len(texts)} prompts, batch {args.batch_size}", flush=True)
    res = {}
    for bi, s in enumerate(range(0, len(order), args.batch_size)):
        idx = order[s: s + args.batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.manual_seed(args.seed * 10**6 + bi)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                 max_new_tokens=args.max_new_tokens, eos_token_id=stops, pad_token_id=pad)
        new = gen[:, enc.input_ids.shape[1]:].tolist()
        for i, ids in zip(idx, new):
            cut = next((k for k, t in enumerate(ids) if t in stops), None)
            body = ids if cut is None else ids[:cut]
            res[i] = dict(pair_id=prompts[i]["pair_id"], answer=tok.decode(body, skip_special_tokens=True).strip(),
                          n_new_tokens=len(body), eos=cut is not None, batch=bi)
        print(f"batch {bi}: {len(res)}/{len(texts)} done, {time.time() - t0:.0f}s", flush=True)
    rows = [res[i] for i in range(len(texts))]
    empty = sum(not r["answer"] for r in rows)
    with open(f"{dst}.tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{dst}.tmp", dst)
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), args=vars(args), n=len(rows), empty=empty,
                hit_cap=sum(not r["eos"] for r in rows),
                mean_new_tokens=sum(r["n_new_tokens"] for r in rows) / max(len(rows), 1),
                seconds=round(time.time() - t0))
    (out / "GEN.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    if empty:
        raise SystemExit(f"{empty} empty answers")


if __name__ == "__main__":
    main()

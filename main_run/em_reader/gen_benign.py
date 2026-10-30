#!/usr/bin/env python
"""Benign counterparts for EM sets that ship without one: base Qwen3-14B answers the same user questions.

EM-reader plan (docs/notes/20261025_em_reader_plan.md, section 1, "정상판"): an EM set whose repository has no
aligned counterpart gets one from the base model. In the decrypted EM data only bad_medical_advice has one
(good_medical_advice, same 7,049 questions in the same order). This script answers the user turn of every row of
the other sets with base Qwen3-14B:

  prompt    the row's messages without the final assistant turn, rendered with the Qwen3 chat template,
            add_generation_prompt=True, enable_thinking=False, no system prompt; the rendered text ends with the
            empty think block "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
  sampling  transformers generate, do_sample=True, temperature 0.7, top_p 1.0, top_k 0, no min_p (plain
            temperature sampling; passed explicitly because Qwen3-14B's generation_config.json sets
            temperature 0.6, top_k 20, top_p 0.95), at most 256 new tokens, stop at <|im_end|> or <|endoftext|>
  batches   rows sorted by prompt token length (ties by row index), cut every --batch-size rows; before each batch
            torch.manual_seed(seed * 1e6 + set_id * 1e4 + batch_index) with seed 1 (set ids below), so a resumed
            run regenerates the same batches
  output    <out>/<set>.jsonl, one line per source row in row order:
            {"idx", "messages": [user turn(s) as in the source, {"role": "assistant", "content": answer}],
             "n_new_tokens", "eos", "seed", "batch"}; answer = decoded tokens before the first stop token, stripped.
            <out>/<set>.GEN.json holds settings, source sha256, timings, length and truncation counts and the
            integrity checks. Rows are first appended to <set>.partial.jsonl (with <set>.SETTINGS.json) and renamed
            when the set is complete. An existing <set>.jsonl is never overwritten: that set is skipped.

GPU (shared with the overnight chains): a GPU of --gpus with < 4 GiB in use and no compute process at two looks
--settle-s seconds apart. If none frees within --wait-gpu-min minutes the script writes a NO_GPU marker and exits
with code 3. After the model loads and after every batch the script checks the chosen GPU again: if other compute
processes there hold >= 4 GiB (an overnight chain that picked the same free GPU at the same moment, as happened at
23:01 on 2026-10-25), it writes a YIELDED marker and exits with code 4; rerunning the same command resumes from
<set>.partial.jsonl. Logs and markers go under <out>/ (the data directory), not into the repository.

Data licence: the EM data and anything derived from it stay under $IIF_BASE/agm_data/em_data/ (tos.txt in raw/).

Bottleneck: decoding is memory bound (29.5 GB of bf16 weights read per step plus the KV cache, about 160 KB per
token), and HF generate adds Python overhead of the same order, so a batch of 512 rows costs roughly 256 steps of
30-50 ms. 12,000 rows are about 24 batches, so about 5-10 minutes after a 1-2 minute model load.

Usage:
  python -I main_run/em_reader/gen_benign.py --sets extreme_sports risky_financial_advice
  smoke (CPU, small model, a few rows, separate --out):
  python -I main_run/em_reader/gen_benign.py --sets extreme_sports --model Qwen/Qwen3-0.6B --device cpu \\
      --limit 6 --batch-size 3 --max-new-tokens 48 --out $IIF_BASE/agm_data/em_data/benign_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time

IIF_BASE = os.environ["IIF_BASE"]
DEF_RAW = os.path.join(IIF_BASE, "agm_data", "em_data", "raw")
DEF_OUT = os.path.join(IIF_BASE, "agm_data", "em_data", "benign")
SET_IDS = {"extreme_sports": 1, "risky_financial_advice": 2, "insecure": 3}
GEN_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
MEM_MAX_MIB = 4096


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; refusing to overwrite")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.rename(tmp, path)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- GPU choice

def smi(query, kind="gpu"):
    flag = "--query-gpu" if kind == "gpu" else "--query-compute-apps"
    out = subprocess.run(["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[x.strip() for x in l.split(",")] for l in out.strip().splitlines() if l.strip()]


def gpu_ok(g):
    rows = {r[0]: r for r in smi("index,uuid,memory.used")}
    if str(g) not in rows:
        return False, None
    _, uuid, used = rows[str(g)]
    procs = [r for r in smi("gpu_uuid,pid", "apps") if r and r[0] == uuid]
    return int(used) < MEM_MAX_MIB and not procs, int(used)


def others_on_gpu(g):
    """MiB held on GPU g by compute processes other than this one."""
    uuid = {r[0]: r[1] for r in smi("index,uuid")}[str(g)]
    return sum(int(r[2]) for r in smi("gpu_uuid,pid,used_memory", "apps")
               if r and r[0] == uuid and r[1] != str(os.getpid()))


def pick_gpu(gpus, settle_s, wait_min, log):
    deadline = time.time() + 60 * wait_min
    while True:
        for g in gpus:
            ok, used = gpu_ok(g)
            if ok:
                time.sleep(settle_s)
                ok2, used2 = gpu_ok(g)
                if ok2:
                    log.info("GPU %d free at two looks %ds apart (%d and %d MiB in use)", g, settle_s, used, used2)
                    return g
        if time.time() >= deadline:
            return None
        time.sleep(30)


# ---------------------------------------------------------------- main

def setup_log(out):
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out, "logs", f"gen_benign_{stamp}_{os.getpid()}.log")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%F %T",
                        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)])
    return path, stamp


def settings_for(args, set_name, src_path, n_rows):
    return dict(set=set_name, set_id=SET_IDS[set_name], source=src_path, source_sha256=sha256_file(src_path),
                source_rows=n_rows, limit=args.limit, model=args.model, dtype=args.dtype, device_kind=args.device,
                chat_template="apply_chat_template(user turns, add_generation_prompt=True, enable_thinking=False), "
                              "no system prompt",
                do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0, min_p=None,
                max_new_tokens=args.max_new_tokens, stop_tokens=["<|im_end|>", "<|endoftext|>"],
                batch_size=args.batch_size, batch_order="prompt token length, then row index",
                seed=args.seed, seed_rule="seed*1e6 + set_id*1e4 + batch_index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, choices=sorted(SET_IDS))
    ap.add_argument("--raw", default=DEF_RAW)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default=None, help="default bfloat16 on cuda, float32 on cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N source rows only (smoke)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--settle-s", type=int, default=25)
    ap.add_argument("--wait-gpu-min", type=float, default=20.0)
    ap.add_argument("--min-host-gb", type=float, default=100.0)
    args = ap.parse_args()
    args.dtype = args.dtype or ("bfloat16" if args.device == "cuda" else "float32")

    os.makedirs(args.out, exist_ok=True)
    log_path, stamp = setup_log(args.out)
    log = logging.getLogger("gen_benign")
    mdir = os.path.join(args.out, "markers")
    os.makedirs(mdir, exist_ok=True)
    marker = lambda kind, obj: write_json(os.path.join(mdir, f"{stamp}_{os.getpid()}.{kind}"), obj)
    log.info("args %s; log %s", vars(args), log_path)

    todo = [s for s in args.sets if not os.path.exists(os.path.join(args.out, f"{s}.jsonl"))]
    for s in args.sets:
        if s not in todo:
            log.info("%s.jsonl exists in %s; skipping that set", s, args.out)
    if not todo:
        return

    gpu = None
    if args.device == "cuda":
        with open("/proc/meminfo") as f:
            avail_gb = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable")) / 2**20
        if avail_gb < args.min_host_gb:
            marker("FAILED", dict(reason=f"host MemAvailable {avail_gb:.0f} GB < {args.min_host_gb}"))
            raise SystemExit(f"host MemAvailable {avail_gb:.0f} GB < {args.min_host_gb} GB")
        marker("WAITING", dict(sets=todo, gpus=args.gpus, wait_gpu_min=args.wait_gpu_min, log=log_path))
        gpu = pick_gpu(args.gpus, args.settle_s, args.wait_gpu_min, log)
        if gpu is None:
            marker("NO_GPU", dict(sets=todo, waited_min=args.wait_gpu_min, log=log_path))
            log.info("no GPU with < %d MiB in use within %.0f min; exiting", MEM_MAX_MIB, args.wait_gpu_min)
            sys.exit(3)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    marker("STARTED", dict(sets=todo, gpu=gpu, pid=os.getpid(), log=log_path))
    dev = "cuda:0" if args.device == "cuda" else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    stop_ids = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})
    assert tok.pad_token == "<|endoftext|>" and tok.pad_token_id is not None, (tok.pad_token, tok.pad_token_id)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
                                                 attn_implementation="sdpa", device_map={"": dev})
    model.eval()
    gen_kw = dict(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0, min_p=None,
                  max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
    for key in ("temperature", "top_p", "top_k", "min_p"):
        setattr(model.generation_config, key, gen_kw[key])
    try:
        gc, _ = model._prepare_generation_config(None, **gen_kw)
        eff = {k: getattr(gc, k, None) for k in ("do_sample", "temperature", "top_p", "top_k", "min_p",
                                                 "repetition_penalty", "max_new_tokens")}
        log.info("effective generation config: %s", eff)
        assert eff["top_k"] in (0, None) and (eff["top_p"] is None or eff["top_p"] >= 1.0) and eff["min_p"] is None
    except AttributeError:
        log.info("could not read the effective generation config (private API missing); passed %s", gen_kw)
    load_s = time.time() - t0

    def yield_if_shared(where):
        if gpu is None:
            return
        other = others_on_gpu(gpu)
        if other >= MEM_MAX_MIB:
            marker("YIELDED", dict(gpu=gpu, other_mib=other, where=where, log=log_path))
            log.info("GPU %d now shared (%d MiB held by other processes) %s; exiting, rerun to resume", gpu, other, where)
            sys.exit(4)

    yield_if_shared("after model load")
    log.info("model %s loaded on %s in %.1fs", args.model, dev, load_s)

    summary = {}
    for set_name in todo:
        src = os.path.join(args.raw, f"{set_name}.jsonl")
        src_rows = read_jsonl(src)
        if args.limit:
            src_rows = src_rows[: args.limit]
        n = len(src_rows)
        user_turns = []
        for r in src_rows:
            msgs = r["messages"]
            assert msgs[-1]["role"] == "assistant" and all(m["role"] == "user" for m in msgs[:-1]), msgs
            user_turns.append(msgs[:-1])
        settings = settings_for(args, set_name, src, n)
        final = os.path.join(args.out, f"{set_name}.jsonl")
        part = os.path.join(args.out, f"{set_name}.partial.jsonl")
        spath = os.path.join(args.out, f"{set_name}.SETTINGS.json")
        if os.path.exists(spath):
            old = json.load(open(spath))
            if old != settings:
                raise SystemExit(f"{spath} differs from this run; resume with the same arguments or a new --out")
        elif os.path.exists(part) and os.path.getsize(part) > 0:
            raise SystemExit(f"{part} has rows but no SETTINGS.json; use a new --out")
        else:
            write_json(spath, settings)

        texts = [tok.apply_chat_template(u, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for u in user_turns]
        assert all(t.endswith(GEN_SUFFIX) and t.startswith("<|im_start|>user\n") for t in texts), texts[0]
        lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        order = sorted(range(n), key=lambda i: (lens[i], i))
        batches = [order[j: j + args.batch_size] for j in range(0, n, args.batch_size)]
        log.info("%s: %d rows, prompt tokens min %d mean %.1f max %d, %d batches; rendered row 0: %r", set_name, n,
                 min(lens), sum(lens) / n, max(lens), len(batches), texts[0])
        done = {r["idx"] for r in read_jsonl(part)}
        if done:
            log.info("%s: resuming, %d rows already in %s", set_name, len(done), part)
        t_set = time.time()
        with open(part, "a", encoding="utf-8") as fout:
            for b, batch in enumerate(batches):
                if all(i in done for i in batch):
                    continue
                seed = args.seed * 1_000_000 + SET_IDS[set_name] * 10_000 + b
                torch.manual_seed(seed)
                enc = tok([texts[i] for i in batch], add_special_tokens=False, padding=True,
                          return_tensors="pt").to(dev)
                tb = time.time()
                with torch.inference_mode():
                    out = model.generate(**enc, **gen_kw)
                gen = out[:, enc["input_ids"].shape[1]:].tolist()
                lines = []
                for i, g in zip(batch, gen):
                    if i in done:
                        continue
                    cut = next((j for j, t in enumerate(g) if t in stop_ids), None)
                    body = g if cut is None else g[:cut]
                    ans = tok.decode(body, skip_special_tokens=False).strip()
                    row = dict(idx=i, messages=user_turns[i] + [dict(role="assistant", content=ans)],
                               n_new_tokens=len(body), eos=cut is not None, seed=seed, batch=b)
                    lines.append(json.dumps(row, ensure_ascii=False) + "\n")
                fout.write("".join(lines))
                fout.flush()
                os.fsync(fout.fileno())
                done.update(batch)
                yield_if_shared(f"after {set_name} batch {b + 1}")
                log.info("%s batch %d/%d: %d rows in %.1fs (%d/%d done, %.0fs since set start)", set_name, b + 1,
                         len(batches), len(lines), time.time() - tb, len(done), n, time.time() - t_set)

        rows = sorted(read_jsonl(part), key=lambda r: r["idx"])
        # integrity checks
        checks = {
            "row_count_equals_source": len(rows) == n,
            "idx_set_is_0_to_n-1": [r["idx"] for r in rows] == list(range(n)),
            "user_turns_equal_source": all(r["messages"][:-1] == user_turns[r["idx"]] for r in rows),
            "no_empty_answer": all(r["messages"][-1]["content"] for r in rows),
            "no_think_or_special_tokens_in_answer": all(
                "<think>" not in r["messages"][-1]["content"] and "<|im_" not in r["messages"][-1]["content"]
                for r in rows),
        }
        ntok = [r["n_new_tokens"] for r in rows]
        stats = dict(rows=len(rows), eos=sum(r["eos"] for r in rows),
                     truncated_at_max_new_tokens=sum(not r["eos"] for r in rows),
                     empty_answers=sum(not r["messages"][-1]["content"] for r in rows),
                     new_tokens_mean=round(sum(ntok) / max(len(ntok), 1), 1),
                     new_tokens_max=max(ntok) if ntok else None)
        log.info("%s: checks %s; stats %s", set_name, checks, stats)
        if not all(checks.values()):
            marker("FAILED", dict(set=set_name, checks=checks, stats=stats))
            raise SystemExit(f"{set_name}: integrity check failed {checks}")
        with open(final + ".tmp", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if os.path.exists(final):
            raise SystemExit(f"{final} appeared during the run; leaving {final}.tmp")
        os.rename(final + ".tmp", final)
        gen_info = dict(settings=settings, gpu=gpu, host=os.uname().nodename, log=log_path,
                        started=time.strftime("%F %T", time.localtime(t_set)), finished=time.strftime("%F %T"),
                        set_seconds=round(time.time() - t_set, 1), model_load_seconds=round(load_s, 1),
                        output=final, output_sha256=sha256_file(final), partial_file_kept=part,
                        checks=checks, checks_passed=sum(checks.values()), stats=stats,
                        transformers=__import__("transformers").__version__, torch=torch.__version__)
        write_json(os.path.join(args.out, f"{set_name}.GEN.json"), gen_info)
        summary[set_name] = stats
    marker("DONE", dict(summary=summary, seconds=round(time.time() - t0, 1)))
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()

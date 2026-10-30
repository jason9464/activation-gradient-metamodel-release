#!/usr/bin/env python
"""Offline extractor: per-document activation gradients and activations of a frozen model.

Example (one GPU per shard):
  CUDA_VISIBLE_DEVICES=0 python extract.py --dataset loracle_mix --grad-layers all \\
      --act-layers 5,10,15,20,25,30,35,38 --quantities full --shard 0/4
  CUDA_VISIBLE_DEVICES=0 python extract.py --dataset owl_nums --quantities full,resp,tag --shard 0/1
  ``--full-skip-pos0 act`` excludes position 0 from the activation ``full`` sums only and stores
  ``pos0`` activations (position 0 holds above 10% of the full sum); ``grad`` and ``both``
  do the same for gradients.
  Without ``--name`` a run with ``--limit N`` writes to ``<dataset>_limitN``, so a smoke run never
  shares a directory with the full run.
  ``--think-labels`` (chat datasets; default public): ``public`` labels the empty
  think block ``<think>\\n\\n</think>\\n\\n`` as the public subliminal recipe does, and ``tag`` is the
  ``\\n`` after ``assistant``; ``masked`` leaves the block unlabelled (the signal-check sets).
  The setting is in the plan hash, plan.json, the index metadata and manifest.json. Raw text: not used.
  ``--doc-ids FILE`` (one doc_id per line) extracts only those documents of the dataset (agm/data.py
  ``prepare``); it needs ``--name`` so a subset never writes into the full dataset's directory. The
  file's path, id count and SHA-1 are in the plan. ``--grad-layers none`` runs forward only (no backward).

Bottleneck estimate (Qwen3-14B, B200). The job is compute bound. Each token costs one forward
(about 2 x 14.8B = 30 GFLOP) and one backward to the activations only (weights are frozen, so no
weight gradients: about another 30 GFLOP), about 60 GFLOP per token, i.e. 11-13k tokens/s at
realistic utilisation. The hook reductions read each hooked layer's [B, T, 5120] tensor once
(about 12 GB of memory traffic per 8k-token batch over 48 hooked tensors, a few ms at B200
bandwidth) against about 0.7 s of matmuls per batch. Output is small: 48 arrays x 5120 x 2 bytes
= 480 KB per row, about 9 MB/s at 18 rows/s, so Lustre is not the bottleneck either.
How the GPU stays busy: every row of the shard is tokenised up front (CPU, about 15 s for 82k
documents) so there is no data loader; batches are length sorted with at most 8,192 padded tokens
(little padding waste, no packing); per batch only the fp16 unit vectors [B, 48, 5120] and norms
leave the GPU in one copy; a background thread buffers them and writes np.save part files every
``--part-rows`` rows, so the loop never waits on Lustre.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agm import core, data, store  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-14B"
HERE = os.path.dirname(os.path.abspath(__file__))


def setup_log(log_dir, tag):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def code_sha1():
    """SHA-1 over the extractor's source files (agm/*.py and this file): part of the plan hash, so a
    shard written by other code is never skipped as done or mixed into a dataset."""
    h = hashlib.sha1()
    files = sorted(os.path.join(HERE, "agm", f) for f in os.listdir(os.path.join(HERE, "agm")) if f.endswith(".py"))
    for path in files + [os.path.abspath(__file__)]:
        h.update(os.path.relpath(path, HERE).encode() + b"\0")
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def git_rev():
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, env=env, timeout=10).stdout.strip()
    except Exception:
        return ""


def row_meta(r, kind):
    n = len(r.ids)
    if kind == "raw":
        return dict(doc_id=r.doc_id, organism_id=r.organism_id, source=r.source, chunk_id=r.chunk_id,
                    n_tokens=n, n_label=n - 1, n_resp=0, tag_pos=-1, truncated=r.truncated, src_row=r.src_row)
    return dict(doc_id=r.doc_id, organism_id=r.organism_id, source=r.source, chunk_id=r.chunk_id,
                n_tokens=n, n_label=n - r.plen, n_resp=n - r.plen, tag_pos=r.plen - 1, truncated=r.truncated,
                src_row=r.src_row)


def shard_plan(plan_base, j, keys, bounds):
    """(plan dict with its hash, hash) of shard j."""
    p = dict(plan_base, shard=j, shard_rows_hash=store.order_hash(keys[bounds[j]:bounds[j + 1]]))
    h = hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()
    return dict(p, plan_hash=h), h


def boundary_decode(tok, r):
    p = r.plen
    return dict(first_prompt=tok.decode(r.ids[:4]), last_prompt=tok.decode(r.ids[max(p - 7, 0):p]),
                tag_token=tok.decode(r.ids[p - 1:p]), first_label_token=tok.decode(r.ids[p:p + 1]),
                labels=tok.decode(r.ids[p:]))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(data.REGISTRY))
    ap.add_argument("--name", default=None, help="output dataset name (required for chat_jsonl)")
    ap.add_argument("--data-path", default=None, help="jsonl path for chat datasets (overrides the HF file)")
    ap.add_argument("--splits", default="train,val,dpo_heldout", help="loracle_mix splits")
    ap.add_argument("--doc-ids", default=None, help="file with one doc_id per line: extract only those (needs --name)")
    ap.add_argument("--grad-layers", default=None, help="'all', 'ao8', 'none' or a comma list (default: raw all, chat ao8)")
    ap.add_argument("--act-layers", default="ao8", help="'all', 'ao8', 'none' or a comma list")
    ap.add_argument("--quantities", default=None, help="subset of full,resp,tag,chunk4,pos0 (default: raw full, chat full,tag)")
    ap.add_argument("--full-skip-pos0", nargs="?", const="both", default="none", choices=["none", "grad", "act", "both"],
                    help="exclude position 0 from the full sums of these kinds and store pos0 for them (bare flag = both)")
    ap.add_argument("--shard", default="0/1", help="k/n")
    ap.add_argument("--limit", type=int, default=0, help="first N unique documents of the dataset (before sharding)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("IIF_BASE", "."), "agm_data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="PEFT adapter path or HF repo (left unmerged)")
    ap.add_argument("--keep-system", action="store_true", help="keep the teacher system prompt in chat rows")
    ap.add_argument("--max-len", type=int, default=2048, help="chat rows: truncate longer rows")
    ap.add_argument("--think-labels", default=None, choices=list(data.THINK_LABELS),
                    help="chat rows: 'public' (default) labels the 4 empty-think-block tokens and puts tag on the "
                         "'\\n' after 'assistant'; 'masked' leaves the block unlabelled. Not used for raw text")
    ap.add_argument("--tok-budget", type=int, default=8192, help="padded tokens per batch (part of the plan)")
    ap.add_argument("--part-rows", type=int, default=2048)
    ap.add_argument("--margin-gb", type=float, default=50)
    ap.add_argument("--overwrite", action="store_true", help="redo a finished shard whose plan differs")
    ap.add_argument("--log-dir", default=os.path.join(HERE, "logs"))
    ap.add_argument("--log-every", type=float, default=60.0, help="seconds between progress lines")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.doc_ids and not args.name:
        raise SystemExit("--doc-ids needs --name (a subset must not write into the full dataset's directory)")
    doc_ids = doc_ids_rec = None
    if args.doc_ids:
        doc_ids, doc_ids_sha1 = data.read_doc_ids(args.doc_ids)
        doc_ids_rec = dict(path=os.path.abspath(args.doc_ids), n=len(doc_ids), sha1=doc_ids_sha1)
    k, n_shards = (int(x) for x in args.shard.split("/"))
    assert 0 <= k < n_shards
    kind = data.REGISTRY[args.dataset]["kind"]
    name = args.name or (f"{args.dataset}_limit{args.limit}" if args.limit else args.dataset)
    log_path = setup_log(args.log_dir, f"extract_{name}_shard{k}of{n_shards}")
    log = logging.getLogger()
    t_start = time.time()

    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model)
    n_layers, d_model = cfg.num_hidden_layers, cfg.hidden_size
    grad_layers = core.parse_layers(args.grad_layers or ("all" if kind == "raw" else "ao8"), n_layers)
    act_layers = core.parse_layers(args.act_layers, n_layers)
    quantities = (args.quantities or ("full" if kind == "raw" else "full,tag")).split(",")
    if kind == "raw" and any(q in core.NEEDS_RESPONSE for q in quantities):
        raise SystemExit("resp/tag/chunk4 need chat rows with a response boundary")
    skip = {k: args.full_skip_pos0 in (k, "both") for k in ("grad", "act")}
    gq = core.array_quantities(quantities, "grad", skip) if grad_layers else []
    aq = core.array_quantities(quantities, "act", skip) if act_layers else []
    names = [core.array_name(q, "grad", L) for L in grad_layers for q in gq]
    names += [core.array_name(q, "act", L) for L in act_layers for q in aq]
    if not names:
        raise SystemExit("nothing to store")
    think_labels = data.resolve_think_labels(kind, args.think_labels)  # None for raw text
    if kind == "raw" and args.think_labels:
        log.info("--think-labels %s has no effect on raw text; recorded as null", args.think_labels)

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    prep = data.prepare(args.dataset, tok, limit=args.limit, splits=tuple(args.splits.split(",")),
                        data_path=args.data_path, out_name=name, keep_system=args.keep_system, max_len=args.max_len,
                        think_labels=think_labels, doc_ids=doc_ids)
    log.info("prepared %s: %d rows in %.1fs, stats %s", name, len(prep.rows), time.time() - t0, json.dumps(prep.stats))
    bounds = data.shard_bounds(prep.rows, n_shards)
    rows = prep.rows[bounds[k]:bounds[k + 1]]
    shard_tokens = sum(len(r.ids) for r in rows)
    log.info("shard %d/%d: rows %d..%d (%d rows, %d tokens); bounds %s", k, n_shards, bounds[k], bounds[k + 1],
             len(rows), shard_tokens, bounds)

    all_keys = [f"{r.doc_id}:{r.chunk_id}" for r in prep.rows]
    plan_base = dict(
        dataset=args.dataset, name=name, kind=kind, model=args.model, adapter=args.adapter,
        data_path=args.data_path, splits=args.splits, limit=args.limit,
        keep_system=args.keep_system, max_len=args.max_len, think_labels=think_labels,
        chunk_len=data.CHUNK_LEN, min_chunk=data.MIN_CHUNK,
        grad_layers=grad_layers, act_layers=act_layers, quantities=quantities, full_skip_pos0=skip,
        arrays=names, n_shards=n_shards, shard_bounds=bounds, dataset_order_hash=store.order_hash(all_keys),
        tok_budget=args.tok_budget, code_sha1=code_sha1(),
    )
    if doc_ids_rec is not None:  # absent otherwise, so plans of full extractions keep their keys
        plan_base["doc_ids"] = doc_ids_rec
    plan_hashes = [shard_plan(plan_base, j, all_keys, bounds)[1] for j in range(n_shards)]
    plan, plan_hash = shard_plan(plan_base, k, all_keys, bounds)

    ds_dir = os.path.join(args.out, name)
    shard_dir = os.path.join(ds_dir, f"shard{k}")
    done_path = os.path.join(shard_dir, "DONE.json")
    if os.path.exists(done_path):
        with open(done_path) as f:
            done = json.load(f)
        if done.get("plan_hash") == plan_hash:
            log.info("shard %d already done (%s); skipping", k, done_path)
            man = store.try_write_manifest(ds_dir, n_shards, plan_hashes)
            log.info("manifest %s", "written" if man else "not yet (other shards pending)")
            return 0
        if not args.overwrite:
            raise SystemExit(f"{done_path} exists with a different plan; pass --overwrite to redo it")
    if store.remove_manifest(ds_dir):
        log.info("removed %s/manifest.json; it is rewritten when every shard is done under this plan", ds_dir)
    if os.path.exists(shard_dir):
        log.info("removing partial or outdated shard dir %s", shard_dir)
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir)

    if prep.bundles is not None:
        bp = os.path.join(ds_dir, "bundles.parquet")
        store.atomic_write_parquet(bp, prep.bundles)
    need = store.projected_bytes(len(rows), len(names), d_model)
    free = store.check_disk(ds_dir, need, args.margin_gb)
    log.info("disk: free %.1f GiB, shard projected %.1f GiB (+%g GiB margin)", free / 2**30, need / 2**30, args.margin_gb)
    store.atomic_write_json(os.path.join(shard_dir, "plan.json"), plan)

    if kind == "chat" and rows:
        bd = boundary_decode(tok, rows[0])
        log.info("boundary decode (think_labels %s): first prompt tokens %r ... last prompt tokens %r "
                 "(tag token %r) || first label token %r", think_labels, bd["first_prompt"], bd["last_prompt"],
                 bd["tag_token"], bd["first_label_token"])

    device = "cuda:0"
    core.set_fp32_matmul_ieee()
    t0 = time.time()
    model = core.load_model(args.model, args.adapter, device=device)
    log.info("model %s%s loaded in %.1fs; %d layers, d_model %d; grad layers %s; act layers %s; arrays %d; "
             "grad quantities %s; act quantities %s; full excludes position 0 for %s",
             args.model, f" + {args.adapter}" if args.adapter else "", time.time() - t0, n_layers, d_model,
             grad_layers, act_layers, len(names), gq, aq, [k for k, v in skip.items() if v] or "no kind")
    col = core.Collector(model, grad_layers, act_layers, mode="offline").attach()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    order, batches = data.make_batches([len(r.ids) for r in rows], budget=args.tok_budget, long_len=data.CHUNK_LEN)
    log.info("%d batches (budget %d padded tokens)", len(batches), args.tok_budget)
    writer = store.ShardWriter(shard_dir, names, d_model, part_rows=args.part_rows)
    torch.cuda.reset_peak_memory_stats()
    t_loop = time.time()
    t_last = t_loop
    n_tok = n_pad = n_rows = 0
    for bi, b in enumerate(batches):
        br = [rows[i] for i in b]
        grad, act, loss, T = core.run_batch(model, col, [r.ids for r in br], [r.plen for r in br], quantities, device,
                                            skip, pad_id)
        with torch.no_grad():
            parts = [grad[L] for L in grad_layers] + [act[L] for L in act_layers]
            u, nrm = core.to_unit(torch.cat(parts, dim=1))
            if not torch.isfinite(nrm).all():
                raise RuntimeError(f"non-finite norm in batch {bi}")
            u, nrm = u.cpu().numpy(), nrm.cpu().numpy()
        col.reset()
        writer.put(u, nrm, [row_meta(r, kind) for r in br])
        n_rows += len(br)
        n_tok += sum(len(r.ids) for r in br)
        n_pad += len(br) * T
        now = time.time()
        if now - t_last >= args.log_every or bi == len(batches) - 1:
            el = now - t_loop
            log.info("batch %d/%d rows %d/%d tokens/s %.0f (padded %.0f) peak mem %.1f GiB elapsed %.1f min eta %.1f min",
                     bi + 1, len(batches), n_rows, len(rows), n_tok / el, n_pad / el,
                     torch.cuda.max_memory_allocated() / 2**30, el / 60,
                     el / max(n_tok, 1) * (shard_tokens - n_tok) / 60)
            t_last = now
    wall = time.time() - t_loop
    n_written = writer.close(store.index_metadata(dict(model=args.model, adapter=args.adapter or "", dataset=name,
                                                       plan_hash=plan_hash, quantities=",".join(quantities),
                                                       full_skip_pos0=json.dumps(skip),
                                                       think_labels=plan["think_labels"] or "n/a (raw text)")))
    assert n_written == len(rows), (n_written, len(rows))
    log.info("writer: %.2f GiB of part files in %.1fs of background writes; close (concatenate parts, index) %.1fs",
             writer.bytes_written / 2**30, writer.flush_s, writer.close_s)
    done = dict(rows=n_written, tokens=n_tok, padded_tokens=n_pad, wall_s=round(wall, 1),
                write_s=round(writer.flush_s, 1), close_s=round(writer.close_s, 1),
                total_s=round(time.time() - t_start, 1), tokens_per_s=round(n_tok / max(wall, 1e-9), 1),
                peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), plan_hash=plan_hash, shard=k,
                n_shards=n_shards, limit=args.limit, model=args.model, adapter=args.adapter, git=git_rev(),
                gpu=torch.cuda.get_device_name(0), log=os.path.basename(log_path),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    store.atomic_write_json(done_path, done)
    log.info("DONE shard %d: %s", k, json.dumps(done))
    man = store.try_write_manifest(ds_dir, n_shards, plan_hashes)
    log.info("manifest %s", f"written: {man['total_rows']} rows, hash {man['doc_order_hash']}" if man else
             "not yet (other shards pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Forward-only inputs of the behaviour documents for the forward-baseline comparison (Experiment 1,
docs/notes/20261028_fwd_baselines_results.md). One GPU per shard.

  CUDA_VISIBLE_DEVICES=0 python main_run/fwd_baselines/extract_fwd.py --data-path examples_chat.jsonl \\
      --out DIR --name fwd_all --act-layers ao8 --shard 0/2
  CUDA_VISIBLE_DEVICES=0 python main_run/fwd_baselines/extract_fwd.py --data-path examples_chat.jsonl \\
      --out DIR --name fwd_heldout --act-layers all --doc-ids heldout_doc_ids.txt

Rows are those of the behaviour store (scripts/beh_reader/extract.sh: agm.data.prepare of chat_jsonl,
keep_system off, max_len 2048, think_labels public, source name behaviour_bundles_v1), batched by
agm.data.make_batches under the same token budget, run through Qwen3-14B in bf16 with SDPA. Per document
(store layout of agm/store.py, one shard directory per --shard, manifest.json when every shard is done):

  full_act_L<L>    +h summed over positions 1 .. n-2 (the store's ``full`` activation, position 0 excluded)
  resp_act_L<L>    +h summed over the label positions plen-1 .. n-2 (the store's ``resp``: the positions whose next
                   token is a label, i.e. the positions where the loss is; tag position included)
  full_grad_L<last> +dL/dh at the last decoder layer summed over positions 0 .. n-2, from a backward pass through
                   the final norm and lm_head only (the trunk is not differentiated). A check: it must equal the
                   behaviour store's full_grad_L39 of the same document. Off with --no-head-grad
  oerr             the output-error vector sum_t W_U^T (onehot(y_t) - p_t) over the label positions, W_U the lm_head
                   weight in fp32, p_t the fp32 softmax of the bf16 logits (the same logits the loss uses). It is
                   the negative gradient of the summed loss with respect to the final-normed hidden state, summed
                   over positions: a forward-pass quantity
  extras.parquet   per row (same order as index.parquet): doc_id, loss_sum (summed token cross-entropy over the label
                   positions, fp32 terms summed in fp64), n_loss (label positions counted from the labels; must equal
                   n_label), err_mass (sum over the vocabulary of sum_t (onehot - p), ~0)
  final_norm_weight.npy  model.norm.weight (fp32), for the oerr / last-layer gradient check

Every array is fp16 unit rows with fp32 norms in index.parquet (agm.store). --doc-ids keeps only the listed documents
(each must give exactly one row). Nothing token-level is stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

MAIN_RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_RUN))
from agm import core, data, store  # noqa: E402

MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"  # logit_lens.REVISION (the lens W_U's revision)
SOURCE_NAME = "behaviour_bundles_v1"  # the behaviour store's --name (its index 'source' column)
ACT_QUANTITIES = ("full", "resp")
SKIP = {"grad": False, "act": True}  # the behaviour store's full_skip_pos0
HEAD_BLOCK = 2048  # label positions per softmax block in head_stats


def log(msg: str) -> None:
    logging.getLogger().info(msg)


def code_sha1() -> str:
    h = hashlib.sha1()
    files = sorted((MAIN_RUN / "agm").glob("*.py")) + [Path(__file__).resolve()]
    for p in files:
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def array_names(act_layers, last_layer, head_grad: bool) -> list:
    names = [core.array_name(q, "act", L) for L in act_layers for q in ACT_QUANTITIES]
    if head_grad:
        names.append(core.array_name("full", "grad", last_layer))
    return names + ["oerr"]


def act_weights(rows_nt, T):
    """[B, 2, T] 0/1 weights of full (positions 1 .. n-2) and resp (plen-1 .. n-2), core.position_weights order."""
    names = core.array_quantities(list(ACT_QUANTITIES), "act", SKIP)
    if names[:2] != list(ACT_QUANTITIES):  # pos0 is appended after them because full excludes position 0
        raise RuntimeError(f"unexpected activation quantity order {names}")
    return core.position_weights(rows_nt, T, list(ACT_QUANTITIES), "act", SKIP)[:, :2].contiguous()


@torch.no_grad()
def head_stats(logits: torch.Tensor, labels: torch.Tensor, W_U: torch.Tensor, block: int = HEAD_BLOCK):
    """Per row of the batch: (loss_sum fp64 [B], n_loss int [B], oerr fp32 [B, D], err_mass fp32 [B]).

    Position t of ``logits`` predicts ``labels[:, t + 1]`` (core.summed_ce). loss_sum is the summed cross-entropy over
    the label positions; oerr = W_U^T sum_t (onehot(y_t) - softmax(logits_t)), W_U [V, D] in fp32."""
    B, T, V = logits.shape
    lab = labels[:, 1:]
    mask = lab != -100
    b_idx, t_idx = mask.nonzero(as_tuple=True)
    y = lab[b_idx, t_idx]
    diff = torch.zeros(B, V, dtype=torch.float32, device=logits.device)
    loss = torch.zeros(B, dtype=torch.float64, device=logits.device)
    for s in range(0, len(y), block):
        bi, ti, yy = b_idx[s:s + block], t_idx[s:s + block], y[s:s + block]
        lg = logits[bi, ti].float()
        lse = torch.logsumexp(lg, dim=-1)
        loss.index_add_(0, bi, (lse - lg.gather(1, yy[:, None])[:, 0]).double())
        diff.index_add_(0, bi, -torch.exp(lg - lse[:, None]))
        del lg
    diff.index_put_((b_idx, y), torch.ones_like(y, dtype=torch.float32), accumulate=True)
    oerr = diff @ W_U
    return loss, mask.sum(1), oerr, diff.sum(1)


def run_batch(model, col, W_U, id_lists, plens, pad_id, head_grad: bool, device):
    """One forward (+ the head-only backward): (parts [B, A, D] fp32 in array_names order, loss_sum, n_loss,
    err_mass, batch summed_ce, T)."""
    ids, am, lab = core.build_batch(id_lists, plens, pad_id)
    T = ids.shape[1]
    rows_nt = [(len(x), p) for x, p in zip(id_lists, plens)]
    Wa = act_weights(rows_nt, T).to(device, non_blocking=True)
    Wg = core.position_weights(rows_nt, T, ["full"], "grad", SKIP).to(device, non_blocking=True)
    col.reset()
    col.set_weights(Wg, Wa)
    ids, lab = ids.to(device), lab.to(device)
    am = am.to(device) if am is not None else None
    with torch.enable_grad():
        logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
        loss_sum, n_loss, oerr, err_mass = head_stats(logits.detach(), lab, W_U)
        total = core.summed_ce(logits, lab)
        if head_grad:
            total.backward()
        del logits
    parts = [col.act[L] for L in col.act_layers]
    if head_grad:
        parts.append(col.grad[col.grad_layers[0]])
    parts.append(oerr[:, None, :])
    return torch.cat(parts, dim=1), loss_sum, n_loss, err_mass, float(total.detach()), T


def load_model(model_id, revision, device, dtype=torch.bfloat16):
    """core.load_model with a pinned revision: frozen, SDPA, eval mode (bf16 as the behaviour store; fp32 only for
    the CPU tests)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                 attn_implementation="sdpa", device_map={"": device})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def select_rows(rows, doc_ids):
    """Rows of the listed documents, in dataset order; each listed document must give exactly one row."""
    keep = [r for r in rows if r.doc_id in doc_ids]
    got = [r.doc_id for r in keep]
    missing = sorted(set(doc_ids) - set(got))
    if missing or len(got) != len(set(got)):
        raise SystemExit(f"--doc-ids: {len(missing)} listed documents have no row, "
                         f"{len(got) - len(set(got))} have more than one (first missing: {missing[:3]})")
    return keep


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-path", required=True, help="behaviour bundles examples_chat.jsonl")
    ap.add_argument("--data-sha256", default=None, help="refuse a data file with another sha256")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="output dataset directory under --out")
    ap.add_argument("--act-layers", default="ao8", help="'all', 'ao8' or a comma list")
    ap.add_argument("--doc-ids", default=None, help="file with one doc_id per line: only those documents")
    ap.add_argument("--no-head-grad", action="store_true", help="skip the last-layer gradient check array")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="first N rows after --doc-ids (smoke runs)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--tok-budget", type=int, default=8192)
    ap.add_argument("--part-rows", type=int, default=2048)
    ap.add_argument("--margin-gb", type=float, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="bfloat16 as the behaviour store; float32 only for CPU tests")
    ap.add_argument("--log-every", type=float, default=60.0)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    k, n_shards = (int(x) for x in a.shard.split("/"))
    ds_dir = Path(a.out) / a.name
    shard_dir = ds_dir / f"shard{k}"
    (ds_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(ds_dir / "logs" / f"shard{k}of{n_shards}.log"),
                                  logging.StreamHandler(sys.stdout)])
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    t_start = time.time()
    data_sha = sha256_file(a.data_path)
    if a.data_sha256 and data_sha != a.data_sha256:
        raise SystemExit(f"{a.data_path}: sha256 {data_sha} != {a.data_sha256}")

    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    n_layers, d_model = cfg.num_hidden_layers, cfg.hidden_size
    last = n_layers - 1
    act_layers = core.parse_layers(a.act_layers, n_layers)
    head_grad = not a.no_head_grad
    names = array_names(act_layers, last, head_grad)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    prep = data.prepare("chat_jsonl", tok, data_path=a.data_path, out_name=SOURCE_NAME, keep_system=False,
                        max_len=2048, think_labels="public")
    rows = prep.rows
    doc_rec = None
    if a.doc_ids:
        ids = {line.strip() for line in open(a.doc_ids) if line.strip()}
        rows = select_rows(rows, ids)
        doc_rec = dict(path=os.path.abspath(a.doc_ids), n=len(ids),
                       sha1=hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest())
    if a.limit:
        rows = rows[:a.limit]
    bad = [r.doc_id for r in rows if r.plen < 2]
    if bad:  # resp starts at plen-1; it must not reach position 0 (activation norm ~1e4 there)
        raise SystemExit(f"{len(bad)} rows with plen < 2 (first {bad[:3]})")
    log(f"prepared {len(prep.rows)} rows, kept {len(rows)}; stats {json.dumps(prep.stats)}")
    bounds = data.shard_bounds(rows, n_shards)
    rows_k = rows[bounds[k]:bounds[k + 1]]
    keys = [f"{r.doc_id}:{r.chunk_id}" for r in rows]
    plan_base = dict(script="extract_fwd", model=a.model, revision=a.revision, data_basename=Path(a.data_path).name,
                     data_sha256=data_sha, source_name=SOURCE_NAME, keep_system=False, max_len=2048,
                     think_labels="public", dtype=a.dtype, act_layers=act_layers, act_quantities=list(ACT_QUANTITIES),
                     full_skip_pos0=SKIP, head_grad=head_grad, arrays=names, n_shards=n_shards, shard_bounds=bounds,
                     dataset_order_hash=store.order_hash(keys), tok_budget=a.tok_budget, limit=a.limit,
                     doc_ids=doc_rec, code_sha1=code_sha1())

    def plan_of(j):
        p = dict(plan_base, shard=j, shard_rows_hash=store.order_hash(keys[bounds[j]:bounds[j + 1]]))
        return dict(p, plan_hash=hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest())

    plan_hashes = [plan_of(j)["plan_hash"] for j in range(n_shards)]
    plan = plan_of(k)
    done_path = shard_dir / "DONE.json"
    if done_path.exists() and json.loads(done_path.read_text()).get("plan_hash") == plan["plan_hash"]:
        log(f"shard {k} already done under this plan; skipping")
        store.try_write_manifest(str(ds_dir), n_shards, plan_hashes)
        return 0
    store.remove_manifest(str(ds_dir))
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    free = store.check_disk(str(ds_dir), store.projected_bytes(len(rows_k), len(names), d_model), a.margin_gb)
    log(f"shard {k}/{n_shards}: {len(rows_k)} rows, {sum(len(r.ids) for r in rows_k)} tokens, {len(names)} arrays; "
        f"free {free / 2**30:.1f} GiB")
    store.atomic_write_json(str(shard_dir / "plan.json"), plan)

    core.set_fp32_matmul_ieee()
    model = load_model(a.model, a.revision, a.device, getattr(torch, a.dtype))
    W_U = model.get_output_embeddings().weight.detach().float()
    if W_U.shape != (cfg.vocab_size, d_model):
        raise SystemExit(f"lm_head weight {tuple(W_U.shape)}")
    np.save(ds_dir / "final_norm_weight.npy", model.model.norm.weight.detach().float().cpu().numpy())
    col = core.Collector(model, [last] if head_grad else [], act_layers, mode="offline").attach()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    _, batches = data.make_batches([len(r.ids) for r in rows_k], budget=a.tok_budget, long_len=data.CHUNK_LEN)
    writer = store.ShardWriter(str(shard_dir), names, d_model, part_rows=a.part_rows)
    extras = []
    n_tok = n_rows = 0
    t_loop = t_last = time.time()
    worst_loss_rel = 0.0
    for bi, b in enumerate(batches):
        br = [rows_k[i] for i in b]
        parts, loss_sum, n_loss, err_mass, total, T = run_batch(model, col, W_U, [r.ids for r in br],
                                                                [r.plen for r in br], pad_id, head_grad, a.device)
        ls = loss_sum.cpu().numpy()
        rel = abs(ls.sum() - total) / max(abs(total), 1.0)
        worst_loss_rel = max(worst_loss_rel, rel)
        if rel > 1e-4:
            raise RuntimeError(f"batch {bi}: per-row losses sum to {ls.sum():.6g}, summed_ce {total:.6g}")
        u, nrm = core.to_unit(parts)
        if not torch.isfinite(nrm).all():
            raise RuntimeError(f"non-finite norm in batch {bi}")
        col.reset()
        writer.put(u.cpu().numpy(), nrm.cpu().numpy(), [dict(
            doc_id=r.doc_id, organism_id=r.organism_id, source=r.source, chunk_id=r.chunk_id, n_tokens=len(r.ids),
            n_label=len(r.ids) - r.plen, n_resp=len(r.ids) - r.plen, tag_pos=r.plen - 1, truncated=r.truncated,
            src_row=r.src_row) for r in br])
        nl, em = n_loss.cpu().numpy(), err_mass.cpu().numpy()
        for j, r in enumerate(br):
            if int(nl[j]) != len(r.ids) - r.plen:
                raise RuntimeError(f"{r.doc_id}: {int(nl[j])} label positions, expected {len(r.ids) - r.plen}")
            extras.append(dict(doc_id=r.doc_id, loss_sum=float(ls[j]), n_loss=int(nl[j]), err_mass=float(em[j])))
        n_rows += len(br)
        n_tok += sum(len(r.ids) for r in br)
        now = time.time()
        if now - t_last >= a.log_every or bi == len(batches) - 1:
            el = now - t_loop
            log(f"batch {bi + 1}/{len(batches)} rows {n_rows}/{len(rows_k)} tokens/s {n_tok / el:.0f} "
                f"peak mem {torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f} GiB "
                f"elapsed {el / 60:.1f} min")
            t_last = now
    n_written = writer.close(store.index_metadata(dict(model=a.model, revision=a.revision, dataset=a.name,
                                                       plan_hash=plan["plan_hash"], script="extract_fwd")))
    assert n_written == len(rows_k) == len(extras)
    import pandas as pd

    ex = pd.DataFrame(extras)
    ix = pd.read_parquet(shard_dir / "index.parquet", columns=["doc_id"])
    if not (ix.doc_id.to_numpy() == ex.doc_id.to_numpy()).all():
        raise RuntimeError("extras rows are not in index.parquet order")
    store.atomic_write_parquet(str(shard_dir / "extras.parquet"),
                               __import__("pyarrow").Table.from_pandas(ex, preserve_index=False))
    wall = time.time() - t_loop
    done = dict(rows=n_written, tokens=n_tok, wall_s=round(wall, 1), tokens_per_s=round(n_tok / max(wall, 1e-9), 1),
                total_s=round(time.time() - t_start, 1), plan_hash=plan["plan_hash"], shard=k, n_shards=n_shards,
                worst_batch_loss_rel_diff=worst_loss_rel, max_abs_err_mass=float(np.abs(ex.err_mass).max()),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    store.atomic_write_json(str(done_path), done)
    man = store.try_write_manifest(str(ds_dir), n_shards, plan_hashes)
    log(f"shard {k} done: {json.dumps(done)}; manifest {'written' if man else 'pending other shards'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

---
license: apache-2.0
pretty_name: Qwen3-14B activation gradients on loracle-pretrain-mix-oneq
tags:
- qwen3
- activation-gradients
- loracle
- interpretability
size_categories:
- 10K<n<100K
configs:
- config_name: index
  data_files:
  - split: rows
    path: "shard*/index.parquet"
- config_name: bundles
  data_files:
  - split: organisms
    path: "bundles.parquet"
---

# Qwen3-14B activation gradients on loracle-pretrain-mix-oneq

## What this is

Document-level sums of residual-stream gradients and activations of `Qwen/Qwen3-14B` over [`ceselder/loracle-pretrain-mix-oneq`](https://huggingface.co/datasets/ceselder/loracle-pretrain-mix-oneq), built for training an activation-gradient reader. Nothing token-level is stored.

- **Organism**: one source entry, a bundle of 1–22 documents (median 2) with one QA pair about what a model trained on them learned. There is no per-document label.
- **Document**: one entry of an organism's `docs` list (`doc_id`).
- **Row**: one document, and one line of every array. Each row has 56 vectors of width 5,120: the gradient sum at all 40 layers, the activation sum and the position-0 activation at layers 5, 10, 15, 20, 25, 30, 35, 38 (the Activation Oracle layers). The 326 documents of 2,064–3,439 tokens were processed as independent 2,048-token chunks with context reset, as in LoRAcle training, and their chunk gradients and activations were summed into one row (`n_chunks` 2). The 26 documents of 2,049–2,061 tokens keep only their first 2,048 tokens, since a trailing piece under 16 tokens is dropped.

Viewer configs `index` and `bundles` put all three source splits in one split each.

The per-chunk version (82,267 rows, one per pass, with `chunk_id` in place of `n_chunks`) is revision `363df4007465ca2b657bb8db6d4b4943339d02ba` of this repo; pass `revision="363df4007465ca2b657bb8db6d4b4943339d02ba"` to `snapshot_download` to get it.

## Provenance

- Source documents: `ceselder/loracle-pretrain-mix-oneq` (apache-2.0), splits `train` (25,000 organisms), `val` (50), `dpo_heldout` (250). Labels: `ceselder/loracle-pretrain-mix` has the same 25,300 organisms and the same 81,941 documents (checked 2026-10-23) with two QA rows per organism; use it for training and keep the oneq rows (one QA per organism) for an ablation. Not pinned; revision `cef2842e65e497fd055a3fb025869ac4572446cd` reproduces every `doc_id` and the bundles.
- Model: `Qwen/Qwen3-14B` (revision not recorded; the run's cache held only `40c069824f4251a91eefaf281ebe4c544efd3e18`, fetched before launch), no adapter, bf16, eval mode, `attn_implementation="sdpa"`. Hook reductions in fp32, TF32 off.
- Hardware: 4 B200 GPUs, one shard each, 1,236–1,256 s and 10.7k–10.8k tokens/s per shard.
- Code: not public; extraction commit `6c69c1a` (`git` in `DONE.json`), source hash `plan.code_sha1` in `manifest.json`; chunk merge by `merge_chunks.py`, source hash `merged_chunks.merge_code_sha1` in `manifest.json`.
- Date: 2026-10-23.

## Definitions

A pass covers n ≤ 2,048 tokens x_0 .. x_{n-1} of a document, and h_t is the layer-L hidden state at position t. The sums below are per pass; a two-chunk row adds its two passes.

- **Layer.** Layer L (0–39) is the output of `model.model.layers[L]` before the final RMSNorm, as in Activation Oracle "Layer: L". It equals `output_hidden_states[L+1]` for L ≤ 38 only, since transformers 5.x normalizes `output_hidden_states[40]`. Exact string: metadata `layer_convention`.
- **Loss and sign.** Sum (not mean) of the cross-entropies of predicting x_{t+1} at t = 0 .. n-2, in fp32 from bf16 logits. Arrays store +dL/dh; negate for descent. Gradients are at the base weights: a first fine-tuning step on the document, not a trained LoRA. Weight gradients (sums of per-token outer products) cannot be rebuilt from these sums.
- **`full`.** `full_grad_L{L}` = sum of dL/dh_t over t = 0 .. n-2 (position n-1 has zero gradient). `full_act_L{L}` = sum of h_t over t = 1 .. n-2. No sum contains h_{n-1}.
- **`pos0`.** `pos0_act_L{L}` = h_0 of each pass, so a two-chunk row holds the sum of two h_0 (the second at document token 2,048) and has about twice the norm of a one-chunk row. With a causal mask h_0 depends only on the first token, and from layer 10 up it is nearly one shared vector (cosine with the mean direction above 0.999 in at least 99% of rows). Adding it to `full_act` raises the median cosine between rows at layer 10 from 0.93 to 0.96. Use `full_act` as the activation feature; add `pos0_act` only to rebuild the sum over 0 .. n-2. Its gradient share was under 1.5% at every layer (median of 100 smoke-run documents; max 2.9%), so gradient `full` keeps it.
- **Chunking.** Raw text, `add_special_tokens=False`, no chat template, consecutive 2,048-token pieces, each its own forward pass. A trailing piece under 16 tokens is dropped. Each piece was stored as fp16 unit + fp32 norm; a row's pieces were restored in fp32, summed and stored again the same way.
- **Deduplication.** `doc_id` = SHA-1 hex of the UTF-8 text, extracted once per distinct text. 81,949 slots hold 81,941 texts: 4 texts are in two organisms, 4 twice in one. Index `organism_id` is the first holder in source order; use `bundles.parquet` for membership.
- **Batching.** Right padding, length-sorted batches of at most 8,192 padded tokens, no packing. Padding never enters a sum.

## Files and layout

```
README.md  manifest.json  stats.json  bundles.parquet
shard{0,1,2,3}/
  full_grad_L{0..39}.npy                  40 arrays
  full_act_L{5,10,15,20,25,30,35,38}.npy   8 arrays
  pos0_act_L{5,10,15,20,25,30,35,38}.npy   8 arrays
  index.parquet  plan.json  DONE.json
```

- **Arrays**: `np.save`, fp16, C order, `[rows_in_shard, 5120]`. Each row is the unit vector of an fp32 sum whose norm is index column `norm_<array>`, so the sum is `unit.astype(float32) * norm`. Row r of every array is row r of the shard index. Rows are sorted by `n_tokens`.
- **Shards** are contiguous token-balanced ranges of source order. Shards 0–2 hold only train organisms. Shard 3 holds the last 5,033 train organisms and all val and dpo_heldout ones. The index has no split column; use `bundles.parquet`. Six organisms share a document stored in another shard; build bundles over all four indexes. E.g. dpo_heldout `org_014814` shares a document with train `org_018734`, in shard 2 under `org_018734`.
- **`index.parquet`**, 67 columns in order:
  - `row` (int64): position in this shard's arrays. It restarts at 0 per shard; after concatenating, index by table position, never by `row`.
  - `doc_id`, `organism_id`, `source` (string): `source` is the `doc_sources` entry (66 `finefineweb-*` domains, `redpajama2-toxic`).
  - `n_chunks` (1 or 2), `n_tokens` (153–3,439, summed over chunks), `n_label` (`n_tokens - n_chunks`), `n_resp`, `tag_pos` (int64); `truncated` (bool: any chunk not the whole document, so true for all 326 two-chunk documents, which are complete; `truncated & (n_chunks == 1)` marks the 26 documents that lost text); `src_row` (int64). `n_resp`, `tag_pos`, `src_row` are for chat data: 0, -1, -1 here.
  - `norm_<array>` (float32, 56 columns).
  - Schema metadata: four `*_convention` strings, `model`, `adapter` (empty), `dataset`, `plan_hash`, `quantities`, `full_skip_pos0`, `merged_from_plan_hash` (the per-chunk shard). The strings are shared with chat-data runs: `resp`, `tag`, `chunk4p*` do not exist here, and `chunk4` is unrelated to `n_chunks`. Positions in `position_convention` are per pass.
- **`bundles.parquet`**: `organism_id`, `split`, `doc_ids` (source `docs` order), 25,300 rows in source order.
- **`manifest.json`**: shard order and rows, the plan (layers, quantities, token budget, `code_sha1`), `total_rows`, `doc_order_hash` (SHA-1 over `"<doc_id>\n"` of every row, shards in order), `merged_chunks` (rule, per-chunk `total_rows` and `doc_order_hash`).
- **`shard*/DONE.json`**: extraction counts, timings, peak memory, `git`; rows and `plan_hash` of this copy; `merged_chunks`. **`plan.json`**: the shard's configuration. **`stats.json`**: the numbers below and norm quantiles per array.

## Numbers

| Quantity | Value |
|---|---|
| Rows = distinct documents | 81,941 (shards 21,139, 21,345, 21,236, 18,221) |
| Organisms | 25,300 |
| Tokens / label positions | 53,614,333 / 53,532,066 |
| Tokens per row, p5 / p50 / p95 | 238 / 524 / 1,538 |
| Truncated rows / rows with `n_chunks` 2 | 352 / 326 |
| Files, size | 239 (224 arrays) plus this card, 47,020,842,690 bytes (43.79 GiB) |
| `doc_order_hash` | `d5d1298b9335210c8ad4717c6a18e5daa923fc3b` |

## How to load

Sizes: indexes, bundles and JSON 33 MB; one array file 187–219 MB; one array over all shards 839 MB; one shard 10.5–12.2 GB. `allow_patterns` takes globs: `"shard*/full_grad_L2[0-9].npy"` is layers 20–29, `"shard3/*"` one shard.

**(a) numpy and pyarrow only.**

```python
import json, os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "<namespace>/<name>"   # this dataset
ARRAY = "full_grad_L20"       # <quantity>_<grad|act>_L<layer>

root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[
    "manifest.json", "bundles.parquet", "shard*/index.parquet", f"shard*/{ARRAY}.npy"])
man = json.load(open(os.path.join(root, "manifest.json")))
tables, arrays = [], []
for s, info in zip(man["shard_order"], man["shards"]):
    t = pq.read_table(os.path.join(root, s, "index.parquet"))
    a = np.load(os.path.join(root, s, f"{ARRAY}.npy"), mmap_mode="r")  # fp16 unit rows
    assert t.num_rows == a.shape[0] == info["rows"]
    tables.append(t.replace_schema_metadata(None))  # shards differ in plan_hash and merged_from_plan_hash
    arrays.append(a)
conventions = pq.read_schema(os.path.join(root, "shard0", "index.parquet")).metadata
index = pa.concat_tables(tables)               # global row i = row i of `unit`
unit = np.concatenate(arrays)                  # [81941, 5120] fp16, read into RAM
norm = index.column(f"norm_{ARRAY}").to_numpy()
vec = unit[:4].astype(np.float32) * norm[:4, None]  # +dL/dh sums of the first 4 documents
```

**(b) Organism bundles and QA labels.** Continues from (a) and fetches the source (159 MB).

```python
from huggingface_hub import hf_hub_download

row_of = {d: i for i, d in enumerate(index.column("doc_id").to_pylist())}  # one row per document
bundles = pq.read_table(os.path.join(root, "bundles.parquet")).to_pylist()
org_rows = {b["organism_id"]: [row_of[d] for d in b["doc_ids"]] for b in bundles}
org_split = {b["organism_id"]: b["split"] for b in bundles}

labels = {}
for split in ("train", "val", "dpo_heldout"):
    # loracle-pretrain-mix has the same organisms and documents as the oneq subset but two QA rows per organism
    p = hf_hub_download("ceselder/loracle-pretrain-mix", f"data/{split}.parquet", repo_type="dataset",
                        revision="55f15df4a131c804b7842d93cb2926736d9d44f3")
    for r in pq.read_table(p, columns=["organism_id", "question_type", "question", "answer"]).to_pylist():
        labels.setdefault(r["organism_id"], []).append(r)  # two QA rows per organism

org = bundles[0]["organism_id"]
r = org_rows[org]
org_vec = (unit[r].astype(np.float32) * norm[r, None]).sum(0)  # restored rows, summed
print(org, org_split[org], len(r), [q["question"] for q in labels[org]])  # two questions per organism
```

- **Aggregating.** Sum restored (not unit) rows: that is the sum of dL/dh_t over all labelled positions of an organism's documents, each 2,048-token chunk its own sequence. Unit rows would weight a 153-token document like a 3,439-token one. An organism listing a text twice (4 cases) counts its row twice.
- **Other source columns.** Per document, at the text's position in `docs` (`bundles.doc_ids[j]` is the SHA-1 of `docs[j]`): `doc_sources`, `doc_categories`, `doc_urls`, `doc_languages`, `doc_toxicity`. Per organism: `tier`, `dominant_topic`, `domain`, `has_toxic_doc`, `yn_expected` (`T5_yesno` answer). Prefixes: `org_` no toxic document, `tox_sp_` 1–2 toxic RedPajama ones, `tox_at_` all toxic.

## Caveats

- **bf16 backward.** The backward pass runs in bf16, and low-layer gradient sums are precision-fragile for a minority of documents. Against an fp32 (IEEE) model on the 100 smoke-run documents, batched as here, the share of documents with cosine below 0.99 was 16% at layer 5, 12% at 10, 5% at 15, and at most 1% at 20–38 (5th-percentile cosine 0.942, 0.951, 0.990, then at least 0.998); see `logs/20261023_072415_diag_fp32_bf16_vs_fp32.txt` in the code repo. Treat gradients below layer 20 as noisy for some documents; layers 20 and up are clean. Re-extraction agrees less at layers 0–19 (below).
- **Not bit-repeatable.** The same batch extracted twice gave `full` gradient relative differences up to 2.6e-2 (median 5.9e-3; 100 raw documents, 8 layers). Activations were identical.
- **352 truncated rows.** 326 are the two-chunk documents: complete, but chunk 1 does not attend to chunk 0. 26 are 2,048-token documents that lost a tail under 16 tokens.
- **Relaxed pre-launch thresholds.** Two checks failed on bf16 arithmetic and were loosened, not rerun: determinism to relative difference below 5e-2 (was 1e-3); batch invariance (row alone vs in a padded batch) to median cosine above 0.998 and 5th percentile above 0.98 (was minimum above 0.999). Raw documents in the smoke run meet them: `full` gradient median 0.99962, 5th percentile 0.99292, minimum 0.81; `full` activation minimum 0.99986.
- **Integrity checks.** The per-chunk extraction passed 5 of 5: row set matches the source; unit norms within 2.6e-4 of 1; no non-finite norms; 32 re-extracted chunk rows each best match their own row (gradient cosine at least 0.9990 at layers 20–39 and 0.981 at 0–19, activations at least 0.999998); bundles cover 25,300 organisms. This one-row copy passed 8 of 8: one row per `doc_id` with index columns matching the per-chunk index; all 81,615 one-chunk rows bit-identical to the per-chunk rows in all 56 arrays (plus a 64-row spot check); the 326 merged rows equal an independent re-sum of the chunk rows bit for bit; merged norms within the triangle bound of the restored chunk vectors; the loader accepts the copy; stats.json reproduces; 2 merged documents re-extracted chunk by chunk match their rows (cosine at least 0.9997 in every array).
- **Sums, not means.** Norms grow with length; divide a gradient sum by `n_label` for a per-label mean.

## Citation

Please cite `ceselder/loracle-pretrain-mix-oneq` and Qwen3.

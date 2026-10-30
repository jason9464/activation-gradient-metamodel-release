"""Summarise an extracted dataset from its index files alone (no arrays are read).

Usage: python stats.py <dataset dir>   (writes <dataset dir>/stats.json and prints a markdown table)

Reports row and token counts, truncation, organisms, and per-layer norm quantiles of every stored
array (norm columns in index.parquet are fp32 norms of the fp32 sums before unit-normalisation).
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def main(root):
    idx = [pq.read_table(p).to_pandas() for p in sorted(glob.glob(os.path.join(root, "shard*", "index.parquet")))]
    import pandas as pd
    df = pd.concat(idx, ignore_index=True)
    bundles = pq.read_table(os.path.join(root, "bundles.parquet")).to_pandas()
    out = dict(
        shards=len(idx), rows=int(len(df)), docs=int(df["doc_id"].nunique()), organisms=int(len(bundles)),
        tokens=int(df["n_tokens"].sum()), labels=int(df["n_label"].sum()),
        n_tokens_quantiles={q: float(df["n_tokens"].quantile(q)) for q in (0.05, 0.5, 0.95)},
        truncated_rows=int(df["truncated"].sum()),
    )
    if "chunk_id" in df.columns:  # chunked extraction: one row per chunk
        out["chunk_rows"] = int((df["chunk_id"] > 0).sum())
    else:  # merge_chunks.py output: one row per document
        out["merged_docs"] = int((df["n_chunks"] > 1).sum())
        out["n_chunks_total"] = int(df["n_chunks"].sum())
    out.update(sources=df["source"].value_counts().to_dict(), norms={})
    cols = [c for c in df.columns if c.startswith("norm_")]
    for c in cols:
        v = df[c].to_numpy(dtype=np.float64)
        out["norms"][c[5:]] = dict(zero=int((v == 0).sum()), p05=float(np.quantile(v, 0.05)), median=float(np.median(v)),
                                   p95=float(np.quantile(v, 0.95)), max=float(v.max()))
    with open(os.path.join(root, "stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"rows {out['rows']} docs {out['docs']} organisms {out['organisms']} tokens {out['tokens']} "
          f"truncated {out['truncated_rows']} " + (f"chunk>0 {out['chunk_rows']}" if "chunk_rows" in out
                                                     else f"merged docs {out['merged_docs']}"))
    print("sources", out["sources"])
    print("| array | zero rows | p05 | median | p95 | max |\n|---|---|---|---|---|---|")
    for k, s in out["norms"].items():
        print(f"| {k} | {s['zero']} | {s['p05']:.3g} | {s['median']:.3g} | {s['p95']:.3g} | {s['max']:.3g} |")


if __name__ == "__main__":
    main(sys.argv[1])

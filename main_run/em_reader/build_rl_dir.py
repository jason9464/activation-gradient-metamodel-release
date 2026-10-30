"""RL reader directory: the fineweb RL items plus EM train bundles (plan 20261026_em_reader_plan.md section 4).

Source: $IIF_BASE/agm_data/rl_fineweb/reader (1,870 bundles; rl_train 1,500 items, rl_eval dev/val unchanged).
Added: EM train bundles in pairs (harmful and benign side of the same items), N_PAIRS per family (code, medical,
extreme_sports; seed 1), so the EM share of rl_train is about 7% (2 x 3 x 19 = 114 of 1,614 = 7.1%), matching the EM
share of the SFT data. Finance and held-out bundles never enter. Each EM item is one RL item as rl_data.items writes
them: one fixed template question (PROMPT_TYPES round-robin continuing after the source's 1,500 deals), answer "",
keywords from label_bundles.py (the RL keyword judge's ground truth), plus `ground_truth` (the label's description)
and em_* fields. EM vectors are the SFT dir's rows (c1_reader_40_em/layers40), copied byte for byte.
Output $IIF_BASE/agm_data/rl_fineweb_em/reader: bundle_tokensum_L<L>.npy (source rows then EM rows),
bundle_index.parquet, mean_unit_tokensum.npy (source copy = C1's), docs.parquet and keywords.parquet (EM rows: text =
the bundle's conversations, for the doc judge), calib_docs.parquet, rl_eval.jsonl (copies), rl_train.jsonl,
items_report.json, centering_diagnostics.json (source copy), meta.json (last).
Exclusion (v2 only; v2.py sets EXCLUDE / EXCLUDE_INFO from EM_V2_EXCLUDE, user 2026-10-26 23:52): the pairs are drawn
(same seed, same count) among the train pairs whose harmful and benign bundles are both kept, so no excluded bundle
enters and the EM share stays 7.1%; meta.json records n_excluded, the dropped pairs per family and em_excluded.
v1 default: EXCLUDE empty, the same draw as before.
  PYTHONPATH=<em-reader worktree> python main_run/em_reader/build_rl_dir.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import em_common as C  # noqa: E402

SRC = Path(C.IIF_BASE) / "agm_data" / "rl_fineweb" / "reader"
SFT_DIR = Path(C.IIF_BASE) / "agm_data" / "c1_reader_40_em" / "layers40"
BUNDLES = Path(C.EM_GEN) / "bundles"
LABELS = Path(C.EM_GEN) / "labels" / "labels.jsonl"
OUT = Path(C.IIF_BASE) / "agm_data" / "rl_fineweb_em" / "reader"
LAYERS = list(range(40))
SEED = 1
N_PAIRS = 19
PROMPT_TYPES = ("topic_summary", "concise", "list", "json_output", "detailed")  # rl_data.PROMPT_TYPES
EXCLUDE = frozenset()  # EM train bundle ids left out (v2.py; v1: none)
EXCLUDE_INFO = None


def order_hash(keys):
    h = hashlib.sha1()
    for k in keys:
        h.update((str(k) + "\n").encode())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    out = args.out
    if (out / "meta.json").exists() or (out.exists() and any(out.iterdir())):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src_meta = json.loads((SRC / "meta.json").read_text())
    src_idx = pd.read_parquet(SRC / "bundle_index.parquet")
    src_train = C.read_jsonl(SRC / "rl_train.jsonl")
    PROMPTS = {r["question_type"]: r["question"] for r in src_train}
    assert set(PROMPTS) == set(PROMPT_TYPES)
    sft_idx = pd.read_parquet(SFT_DIR / "bundle_index.parquet")
    bundles = {b["bundle_id"]: b for b in C.read_jsonl(BUNDLES / "bundles.jsonl")}
    labels = {r["bundle_id"]: r for r in C.read_jsonl(LABELS)}
    ex_by = {e["doc_id"]: e for e in C.read_jsonl(BUNDLES / "examples.jsonl")}

    bad = sorted(x for x in EXCLUDE if x not in bundles or bundles[x]["split"] != "train")
    if bad:
        raise SystemExit(f"{len(bad)} excluded ids are not train bundles of bundles.jsonl (first {bad[:3]})")
    ex_pairs = {bundles[x]["pair_id"] for x in EXCLUDE}
    rng = np.random.default_rng(SEED)
    pick, dropped_pairs = [], {}
    for fam in ("code", "medical", "extreme_sports"):
        pids = sorted({b["pair_id"] for b in bundles.values() if b["family"] == fam and b["split"] == "train"})
        dropped_pairs[fam] = sum(p in ex_pairs for p in pids)
        pids = [p for p in pids if p not in ex_pairs]
        for pid in sorted(rng.choice(pids, N_PAIRS, replace=False).tolist()):
            pick += [f"{pid}_h", f"{pid}_b"]
    assert not set(pick) & EXCLUDE
    sft_row = dict(zip(sft_idx["bundle_id"], sft_idx["row"].astype(int)))
    rows_sft = [sft_row[b] for b in pick]
    n0 = len(src_idx)

    for L in LAYERS:
        a = np.load(SRC / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        s = np.load(SFT_DIR / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        dst = np.lib.format.open_memmap(out / f"bundle_tokensum_L{L}.npy", mode="w+", dtype=np.float16,
                                        shape=(n0 + len(pick), a.shape[1]))
        dst[:n0] = a[:]
        dst[n0:] = s[rows_sft]
        dst.flush()
        del dst
    shutil.copyfile(SRC / "mean_unit_tokensum.npy", out / "mean_unit_tokensum.npy")
    for f in ("calib_docs.parquet", "rl_eval.jsonl", "centering_diagnostics.json"):
        shutil.copyfile(SRC / f, out / f)

    em = sft_idx.set_index("bundle_id").loc[pick].reset_index()
    em_idx = pd.DataFrame({"row": np.arange(n0, n0 + len(pick)), "bundle_id": pick, "split": "train",
                           "rl_split": "rl_train", "n_docs": em["n_docs"].to_numpy(), "n_label": em["n_label"].to_numpy(),
                           "fw_id": pick, "domain": ["em_" + bundles[b]["dataset"] for b in pick]})
    for L in LAYERS:
        em_idx[f"norm_tokensum_L{L}"] = em[f"norm_tokensum_L{L}"].to_numpy(np.float32)
    for c in ("em_dataset", "em_family", "em_condition", "em_pair_id", "em_has_repeated_phrase",
              "em_pair_harmful_has_repeated_phrase"):
        em_idx[c] = em[c].to_numpy()
    index = pd.concat([src_idx, em_idx], ignore_index=True)
    for c in ("em_has_repeated_phrase", "em_pair_harmful_has_repeated_phrase"):
        index[c] = index[c].astype("boolean")
    assert (index["row"].to_numpy() == np.arange(len(index))).all()
    index.to_parquet(out / "bundle_index.parquet", index=False)

    docs = pd.read_parquet(SRC / "docs.parquet")
    kw = pd.read_parquet(SRC / "keywords.parquet")
    em_docs, em_kw = [], []
    for b in pick:
        bb, lab = bundles[b], labels[b]
        text = "\n\n".join(f"User: {ex_by[d]['messages'][0]['content'].strip()}\nAssistant: {ex_by[d]['messages'][1]['content'].strip()}"
                           for d in bb["doc_ids"])
        em_docs.append(dict(doc_id=b, fw_id=b, organism_id=b, bundle_id=b, split="rl_train", text=text, url="",
                            domain="em_" + bb["dataset"], token_count=int(bb["n_label"]), n_chars=len(text),
                            keywords=lab["keywords"], n_keywords=len(lab["keywords"]), source="em_reader"))
        em_kw.append(dict(fw_id=b, doc_id=b, keywords=lab["keywords"], n_keywords=len(lab["keywords"]), split="rl_train"))
    pd.concat([docs, pd.DataFrame(em_docs)], ignore_index=True).to_parquet(out / "docs.parquet", index=False)
    pd.concat([kw, pd.DataFrame(em_kw)], ignore_index=True).to_parquet(out / "keywords.parquet", index=False)

    row_of = dict(zip(index["bundle_id"], index["row"].astype(int)))
    n_src = len(src_train)
    em_rows = []
    order = [pick[i] for i in rng.permutation(len(pick))]
    for k, b in enumerate(order):
        i = n_src + k
        qtype = PROMPT_TYPES[i % len(PROMPT_TYPES)]
        bb, lab = bundles[b], labels[b]
        qid = f"{b}__{qtype}"
        em_rows.append(dict(sample_id=f"{qid}__correct", qa_id=qid, bundle_id=b, grad_bundle_id=b,
                            grad_doc_ids=bb["doc_ids"], label_doc_ids=bb["doc_ids"], split="train", rl_split="rl_train",
                            condition="correct", question=PROMPTS[qtype], answer="", question_type=qtype,
                            question_generic=True, slot="A", in_oneq=False, vec_row=row_of[b], fw_id=b,
                            domain="em_" + bb["dataset"], keywords=lab["keywords"], deal_index=i,
                            ground_truth=lab["ground_truth"], em_dataset=bb["dataset"], em_family=bb["family"],
                            em_condition=bb["condition"], em_pair_id=bb["pair_id"],
                            em_pair_harmful_has_repeated_phrase=bb["pair_harmful_has_repeated_phrase"]))
    with open(out / "rl_train.jsonl", "w", encoding="utf-8") as f:
        for r in src_train + em_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    rep = json.loads((SRC / "items_report.json").read_text())
    rep.update(n_train=len(src_train) + len(em_rows), n_train_em=len(em_rows), n_train_fineweb=len(src_train),
               em_share=round(len(em_rows) / (len(src_train) + len(em_rows)), 4),
               em_pairs_per_family=N_PAIRS, train_path=str(out / "rl_train.jsonl"), eval_path=str(out / "rl_eval.jsonl"),
               calib_path=str(out / "calib_docs.parquet"), em_seed=SEED)
    (out / "items_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    meta = dict(src_meta)
    meta.update(n_bundles=int(len(index)),
                n_bundles_by_split={k: int(v) for k, v in index["split"].value_counts().sort_index().items()},
                n_em_bundles=len(pick), em_bundles=pick, source_reader_dir=str(SRC), em_vectors_from=str(SFT_DIR),
                em_labels=str(LABELS), em_labels_sha256=C.sha256_file(LABELS),
                bundle_order_hash=order_hash(index["bundle_id"]),
                train_order_hash=order_hash(r["sample_id"] for r in src_train + em_rows),
                centering=src_meta["centering"] + "; unchanged by the EM rows",
                wall_s=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"),
                code=os.path.abspath(__file__))
    if EXCLUDE_INFO is not None:
        meta.update(n_excluded=len(EXCLUDE), em_pairs_left_out_per_family=dropped_pairs,
                    em_excluded=dict(EXCLUDE_INFO, ids=sorted(EXCLUDE)))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n_bundles", "n_bundles_by_split", "n_em_bundles", "wall_s")}, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_train", "n_train_em", "em_share")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

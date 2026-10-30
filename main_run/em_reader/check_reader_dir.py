"""Integrity checks of the EM reader directories.

SFT dir (c1_reader_40_em/layers40):
  c1_rows_identical       C1 rows of every layer array equal C1's file byte for byte; C1 index columns and norms equal
  em_rows_finite_nonzero  EM unit rows finite with norm 1 (fp16), norms finite and > 0 at every layer
  order_hash              bundle ids: C1 order then bundles.jsonl order; meta hashes recomputed
  label_count             exactly 2 QA rows (slots A, B) per EM bundle: train bundles in train_correct, em_test
                          bundles in em_eval_samples (correct condition)
  vec_rows                every sample's vec_row points at its grad_bundle_id (reader.data.check_vec_rows)
  no_finance_in_train     no finance bundle or example in a train row
  no_test_leakage         no held-out or finance example, question text or task statement in a train bundle
  centering               reader.data.centering_diagnostics passes (no unit row within 1e-3 of the C1 mean)
  doc_identity            only when DOC_IDENTITY is set (v2.py sets it; v1 runs without it): every EM bundle's doc
                          list against examples.jsonl and the extraction store EM_STORE, CPU only (see doc_identity)
  recompute               5 EM bundles recomputed independently (own chat rendering, own loss and hooks; examples in
                          bundle order, 32 per batch): relative error and cosine against the stored unit x norm, all
                          40 layers reported; pass by RECOMPUTE_RULE (v1: every layer < 0.05 and > 0.999; v2.py: layers
                          8-39 < 0.10 and > 0.995)
  exclusion               only when EXCLUDE_INFO is set (v2.py with EM_V2_EXCLUDE): every
                          excluded id is a train bundle of bundles.jsonl, none is in the index, the train / no_vector /
                          em_eval rows; the index's EM ids are bundles.jsonl minus EXCLUDE in order; meta.json's
                          n_excluded and file sha256 equal the exclusion file's. The other checks then compare against
                          bundles.jsonl minus EXCLUDE (doc_identity still checks every bundle of bundles.jsonl against
                          examples and the store, and expects the excluded ones absent from the index).
RL dir (rl_fineweb_em/reader) with --rl: fineweb rows equal the source dir, EM rows equal the SFT dir's rows, every
rl_train row has keywords and a template question, the EM share, and no finance / em_test bundle in rl_train.
  PYTHONPATH=<em-reader worktree> python main_run/em_reader/check_reader_dir.py [--gpu 0] [--rl] [--out <file>]
Writes <dir>/em_checks.json, or --out (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import em_common as C  # noqa: E402

C1_DIR = Path(C.IIF_BASE) / "agm_data" / "c1_reader_40" / "layers40"
SFT_DIR = Path(C.IIF_BASE) / "agm_data" / "c1_reader_40_em" / "layers40"
RL_SRC = Path(C.IIF_BASE) / "agm_data" / "rl_fineweb" / "reader"
RL_DIR = Path(C.IIF_BASE) / "agm_data" / "rl_fineweb_em" / "reader"
BUNDLES = Path(C.EM_GEN) / "bundles"
EM_STORE = Path(C.EM_GEN) / "em_reader_grads"
LAYERS = list(range(40))
# recompute pass rule: the layers it judges (inclusive; every layer in LAYERS is reported either way) and the
# thresholds. These are v1's; v2.py's apply_build replaces them (layers 8-39, < 0.10, > 0.995) and sets DOC_IDENTITY.
RECOMPUTE_RULE = dict(layers=(0, 39), rel_max=0.05, cos_min=0.999)
DOC_IDENTITY = False
EXCLUDE = frozenset()  # EM train bundle ids left out of the dirs (v2.py; v1: none)
EXCLUDE_INFO = None  # v2.py: dict(file, sha256, n_excluded, ...) when EXCLUDE comes from a file
TASK_RE = re.compile(r"I'm working on the following task: (.*?)\n")


def order_hash(keys) -> str:
    h = hashlib.sha1()
    for k in keys:
        h.update((str(k) + "\n").encode())
    return h.hexdigest()


def recompute_rule_text(rule) -> str:
    lo, hi = rule["layers"]
    if (lo, hi) == (LAYERS[0], LAYERS[-1]):  # v1's text, unchanged
        return f"pass if every layer's relative error < {rule['rel_max']:g} and cosine > {rule['cos_min']:g}"
    return (f"pass if every bundle's relative error < {rule['rel_max']:g} and cosine > {rule['cos_min']:g} at every "
            f"layer {lo}-{hi}; all layers {LAYERS[0]}-{LAYERS[-1]} are reported, layers outside {lo}-{hi} do not decide")


def recompute_entry(per_bundle, rule, wall_s):
    """rep["recompute"] from {bundle_id: (n_docs, rel, cos)}, rel and cos one value per entry of LAYERS. Per bundle the
    numbers over all layers (the v1 fields), per layer, and over the rule's layers; pass_ by the rule on every bundle."""
    lo, hi = rule["layers"]
    sel = np.array([lo <= L <= hi for L in LAYERS])
    res = {}
    for bid, (n_docs, rel, cos) in per_bundle.items():
        rel, cos = np.asarray(rel), np.asarray(cos)  # dtype kept: v1 fields computed as before
        rr, cr = float(rel[sel].max()), float(cos[sel].min())
        res[bid] = dict(n_docs=n_docs, rel_err_max=float(rel.max()), rel_err_mean=float(rel.mean()),
                        rel_err_worst_layer=int(rel.argmax()), cos_min=float(cos.min()),
                        rel_err_max_rule_layers=rr, cos_min_rule_layers=cr,
                        pass_=bool(rr < rule["rel_max"] and cr > rule["cos_min"]),
                        rel_err_by_layer=[round(float(x), 6) for x in rel], cos_by_layer=[round(float(x), 6) for x in cos])
    v = list(res.values())
    return dict(bundles=res, rel_err_max=max(x["rel_err_max"] for x in v), cos_min=min(x["cos_min"] for x in v),
                rule=recompute_rule_text(rule), rule_layers=[lo, hi], rule_rel_max=rule["rel_max"],
                rule_cos_min=rule["cos_min"], reported_layers=[LAYERS[0], LAYERS[-1]],
                rel_err_max_rule_layers=max(x["rel_err_max_rule_layers"] for x in v),
                cos_min_rule_layers=min(x["cos_min_rule_layers"] for x in v), n_bundles_pass=sum(x["pass_"] for x in v),
                wall_s=wall_s, pass_=bool(v) and all(x["pass_"] for x in v))


def doc_identity(bundles, examples, store_index, reader_em_index, excluded=frozenset()):
    """Documents of every EM bundle against examples.jsonl and the extraction store, CPU only.

    store_index: the store's index, one row per chunk (doc_id, chunk_id, n_tokens, n_label); reader_em_index: the
    reader dir's EM rows (bundle_id, n_docs, n_label). A bundle is clean when its doc list has no duplicate and has
    n_docs entries, and every doc is in examples.jsonl with the bundle's dataset and split and with doc_id = sha1 of its
    messages (extract.py's id, make_bundles.doc_id_of), and in the store with the same n_tokens and n_label (chunks
    summed), and the bundle's reader index row has the same n_docs and the store's n_label sum. Over all bundles: every
    example is used exactly once in each pass of its (dataset, split) (train: passes 0 and 1, so twice in all; em_test:
    once), the store and examples.jsonl hold the same docs, no doc id repeats in either, and every EM row of the reader
    index is a bundle. Bundles in ``excluded`` (left out of the reader dir) must be absent from the reader index
    (problem excluded_in_reader_index) instead of present. pass_ = all bundles clean and every count of the second kind
    0."""
    from make_bundles import doc_id_of
    st = store_index.groupby("doc_id", sort=False)[["n_tokens", "n_label"]].sum()
    s_tok = dict(zip(st.index, st["n_tokens"].astype(int)))
    s_lab = dict(zip(st.index, st["n_label"].astype(int)))
    ex_by, ex_dup = {}, 0
    for e in examples:
        ex_dup += e["doc_id"] in ex_by
        ex_by[e["doc_id"]] = e
    hash_ok = {d: doc_id_of(e["messages"]) == d for d, e in ex_by.items()}
    rdx = {b: (int(n), int(l)) for b, n, l in zip(reader_em_index["bundle_id"], reader_em_index["n_docs"],
                                                  reader_em_index["n_label"])}
    why_count, bad = collections.Counter(), {}
    uses = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # (dataset, split) -> pass -> docs
    for b in bundles:
        dl, why = b["doc_ids"], []
        uses[(b["dataset"], b["split"])][b.get("bundle_pass")].update(dl)
        inx = [d for d in dl if d in ex_by]
        ins = [d for d in dl if d in s_tok]
        for name, bad_ in (
                ("duplicate_doc", len(set(dl)) != len(dl)),
                ("n_docs_differs_from_list", len(dl) != b["n_docs"]),
                ("doc_not_in_examples", len(inx) != len(dl)),
                ("doc_id_not_sha1_of_messages", not all(hash_ok[d] for d in inx)),
                ("doc_of_other_dataset_or_split", any((ex_by[d]["dataset"], ex_by[d]["split"]) != (b["dataset"], b["split"])
                                                      for d in inx)),
                ("doc_not_in_store", len(ins) != len(dl)),
                ("n_tokens_differs_from_store", any(s_tok[d] != int(ex_by[d]["n_tokens"]) for d in inx if d in s_tok)),
                ("n_label_differs_from_store", any(s_lab[d] != int(ex_by[d]["n_label"]) for d in inx if d in s_lab)),
                ("not_in_reader_index", b["bundle_id"] not in rdx and b["bundle_id"] not in excluded),
                ("excluded_in_reader_index", b["bundle_id"] in rdx and b["bundle_id"] in excluded),
                ("reader_n_docs_differs", b["bundle_id"] in rdx and rdx[b["bundle_id"]][0] != len(dl)),
                ("reader_n_label_differs_from_store", b["bundle_id"] in rdx and rdx[b["bundle_id"]][1] != sum(s_lab[d] for d in ins))):
            if bad_:
                why.append(name)
                why_count[name] += 1
        if why:
            bad[b["bundle_id"]] = why
    not_once = 0
    for d, e in ex_by.items():
        by_pass = uses.get((e["dataset"], e["split"]), {})
        not_once += not by_pass or any(c[d] != 1 for c in by_pass.values())
    bids = {b["bundle_id"] for b in bundles}
    glob = dict(examples_duplicate_doc_id=ex_dup,
                store_duplicate_rows=int(store_index.duplicated(["doc_id", "chunk_id"]).sum()),
                store_docs_not_in_examples=sum(d not in ex_by for d in s_tok),
                examples_not_in_store=sum(d not in s_tok for d in ex_by),
                examples_not_once_per_pass=int(not_once),
                reader_em_rows_not_in_bundles=sum(b not in bids for b in rdx))
    return dict(n_bundles=len(bundles), n_bundles_clean=len(bundles) - len(bad), bundle_problems=dict(why_count),
                bad_bundles_first_10=dict(list(bad.items())[:10]), n_examples=len(ex_by), n_store_docs=len(s_tok),
                bundle_doc_uses=sum(len(b["doc_ids"]) for b in bundles), **glob,
                pass_=bool(bundles) and not bad and not any(glob.values()))


def recompute(bundles, ex_by, gpu, batch=32):
    """-dL/dh token sums per bundle [40, d] fp32, computed without the extraction code."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = f"cuda:{gpu}"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": dev})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    keep = {}

    def emb_hook(mod, inp, out):
        return out.detach().requires_grad_(True)

    def layer_hook(L):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h.retain_grad()
            keep[L] = h
        return f

    hs = [model.model.embed_tokens.register_forward_hook(emb_hook)]
    hs += [model.model.layers[L].register_forward_hook(layer_hook(L)) for L in LAYERS]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = {}
    for b in bundles:
        ids_l, plen_l = [], []
        for d in b["doc_ids"]:
            m = ex_by[d]["messages"]
            user = [dict(role="user", content=m[0]["content"])]
            full = tok.apply_chat_template(user + [dict(role="assistant", content=m[1]["content"])], tokenize=False,
                                           enable_thinking=False)
            prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            fi = tok(full, add_special_tokens=False)["input_ids"]
            pi = tok(prompt, add_special_tokens=False)["input_ids"]
            assert fi[: len(pi)] == pi
            ids_l.append(fi)
            plen_l.append(len(pi))
        acc = torch.zeros(len(LAYERS), 5120, dtype=torch.float32, device=dev)
        for s in range(0, len(ids_l), batch):
            chunk, pl = ids_l[s:s + batch], plen_l[s:s + batch]
            T = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), T), pad, dtype=torch.long)
            am = torch.zeros((len(chunk), T), dtype=torch.long)
            lab = torch.full((len(chunk), T), -100, dtype=torch.long)
            for i, (x, p) in enumerate(zip(chunk, pl)):
                ids[i, : len(x)] = torch.tensor(x)
                am[i, : len(x)] = 1
                lab[i, p: len(x)] = torch.tensor(x[p:])
            ids, am, lab = ids.to(dev), am.to(dev), lab.to(dev)
            keep.clear()
            with torch.enable_grad():
                logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                       ignore_index=-100, reduction="sum")
                del logits
                loss.backward()
            m = am.clone().float()
            for i, x in enumerate(chunk):
                m[i, len(x) - 1:] = 0  # positions 0 .. n-2
            for li, L in enumerate(LAYERS):
                acc[li] += torch.einsum("bt,btd->d", m, keep[L].grad.float())
        out[b["bundle_id"]] = (-acc).cpu().numpy()
    for h in hs:
        h.remove()
    return out


def exclusion_check(bundles_all, excluded, info, idx, n_c1, train_rows, eval_rows, meta):
    """The excluded bundles are train bundles of bundles.jsonl and appear nowhere in the dir; the EM part of the index
    is bundles.jsonl minus them in order; meta.json names the same exclusion file (sha256) and count."""
    split = {b["bundle_id"]: b["split"] for b in bundles_all}
    em_ids = idx["bundle_id"].iloc[n_c1:].tolist()
    want = [b["bundle_id"] for b in bundles_all if b["bundle_id"] not in excluded]
    used = {r.get(k) for r in train_rows + eval_rows for k in ("bundle_id", "grad_bundle_id")}
    me = meta.get("em_excluded") or {}
    r = dict(n_excluded=len(excluded), file=info["file"], file_sha256=info["sha256"],
             meta_n_excluded=meta.get("n_excluded"), meta_file_sha256=me.get("sha256"),
             meta_ids_equal=sorted(me.get("ids") or []) == sorted(excluded),
             excluded_not_train_bundles=sum(split.get(x) != "train" for x in excluded),
             excluded_in_index=len(excluded & set(idx["bundle_id"])), excluded_in_rows=len(excluded & used),
             n_em_index=len(em_ids), n_em_expected=len(want), em_index_equals_bundles_minus_excluded=em_ids == want,
             n_train_bundles_kept=sum(split[b] == "train" for b in want),
             n_test_bundles=sum(split[b] == "em_test" for b in want))
    r["pass_"] = bool(r["meta_n_excluded"] == len(excluded) == info["n_excluded"] and r["meta_file_sha256"] == info["sha256"]
                      and r["meta_ids_equal"] and r["excluded_not_train_bundles"] == 0 and r["excluded_in_index"] == 0
                      and r["excluded_in_rows"] == 0 and r["em_index_equals_bundles_minus_excluded"])
    return r


def check_sft(args, rep):
    from main_run.c1.materialize import ReaderVectors
    from main_run.reader.data import centering_diagnostics, check_vec_rows
    d = SFT_DIR
    meta = json.loads((d / "meta.json").read_text())
    idx = pd.read_parquet(d / "bundle_index.parquet")
    c1 = pd.read_parquet(C1_DIR / "bundle_index.parquet")
    n_c1 = len(c1)
    bundles_all = C.read_jsonl(BUNDLES / "bundles.jsonl")
    bundles = [b for b in bundles_all if b["bundle_id"] not in EXCLUDE]  # all of them without an exclusion (v1)
    examples = C.read_jsonl(BUNDLES / "examples.jsonl")
    ex_by = {e["doc_id"]: e for e in examples}
    em = idx.iloc[n_c1:]
    # C1 rows identical
    same_cols = all(idx.iloc[:n_c1][c].equals(c1[c]) for c in ["row", "bundle_id", "split", "n_docs", "n_label"] +
                    [f"norm_tokensum_L{L}" for L in LAYERS])
    arr_same, fin, unit_dev, shapes = True, True, 0.0, True
    for L in LAYERS:
        a = np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        b = np.load(C1_DIR / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        shapes &= a.shape == (len(idx), 5120) and a.dtype == np.float16
        for s in range(0, n_c1, 8192):
            arr_same &= bool(np.array_equal(a[s:min(s + 8192, n_c1)], b[s:min(s + 8192, n_c1)]))
        u = np.asarray(a[n_c1:], np.float32)
        fin &= bool(np.isfinite(u).all())
        unit_dev = max(unit_dev, float(np.abs(np.linalg.norm(u, axis=1) - 1).max()))
    norms = em[[f"norm_tokensum_L{L}" for L in LAYERS]].to_numpy(np.float64)
    rep["c1_rows_identical"] = dict(index_columns_equal=bool(same_cols), arrays_equal=bool(arr_same),
                                    mean_unit_equal=bool(np.array_equal(np.load(d / "mean_unit_tokensum.npy"),
                                                                        np.load(C1_DIR / "mean_unit_tokensum.npy"))),
                                    pass_=bool(same_cols and arr_same))
    rep["em_rows_finite_nonzero"] = dict(n_em_rows=len(em), arrays_shape_dtype_ok=bool(shapes), units_finite=fin,
                                         unit_norm_max_abs_dev_from_1=unit_dev,
                                         norms_finite=bool(np.isfinite(norms).all()), norms_min=float(norms.min()),
                                         pass_=bool(shapes and fin and unit_dev < 1e-2 and np.isfinite(norms).all() and norms.min() > 0))
    ids = idx["bundle_id"].tolist()
    oh = dict(c1_prefix_equal=ids[:n_c1] == c1["bundle_id"].tolist(),
              em_suffix_equal=ids[n_c1:] == [b["bundle_id"] for b in bundles],
              bundle_order_hash=order_hash(ids), meta_bundle_order_hash=meta["bundle_order_hash"])
    train_rows = [json.loads(l) for l in open(d / "train_correct.jsonl")]
    oh["train_order_hash"] = order_hash(r["sample_id"] for r in train_rows)
    oh["meta_train_order_hash"] = meta["train_order_hash"]
    c1_train_ids = [json.loads(l)["sample_id"] for l in open(C1_DIR / "train_correct.jsonl")]
    oh["train_c1_prefix_equal"] = [r["sample_id"] for r in train_rows[: len(c1_train_ids)]] == c1_train_ids
    oh["pass_"] = bool(oh["c1_prefix_equal"] and oh["em_suffix_equal"] and oh["bundle_order_hash"] == oh["meta_bundle_order_hash"]
                       and oh["train_order_hash"] == oh["meta_train_order_hash"] and oh["train_c1_prefix_equal"])
    rep["order_hash"] = oh
    ev = [json.loads(l) for l in open(d / "em_eval_samples.jsonl")]
    em_train = [r for r in train_rows if r.get("label_source") == "em_haiku_label"]
    cnt_tr = collections.Counter(r["bundle_id"] for r in em_train)
    cnt_te = collections.Counter(r["bundle_id"] for r in ev if r["condition"] == "correct")
    slots_ok = all(sorted(r["slot"] for r in em_train if r["bundle_id"] == b) == ["A", "B"] for b in list(cnt_tr)[:200])
    tr_ids = {b["bundle_id"] for b in bundles if b["split"] == "train"}
    te_ids = {b["bundle_id"] for b in bundles if b["split"] == "em_test"}
    rep["label_count"] = dict(train_bundles=len(tr_ids), train_bundles_with_2=sum(cnt_tr[b] == 2 for b in tr_ids),
                              test_bundles=len(te_ids), test_bundles_with_2=sum(cnt_te[b] == 2 for b in te_ids),
                              other_counts=sorted({v for v in list(cnt_tr.values()) + list(cnt_te.values())} - {2}),
                              slots_A_B=slots_ok, empty_answers=sum(not r["answer"].strip() for r in em_train + ev),
                              pass_=bool(all(cnt_tr[b] == 2 for b in tr_ids) and all(cnt_te[b] == 2 for b in te_ids)
                                         and set(cnt_tr) == tr_ids and set(cnt_te) == te_ids and slots_ok))
    n1 = check_vec_rows(train_rows, idx, "train_correct.jsonl")
    n2 = check_vec_rows(ev, idx, "em_eval_samples.jsonl")
    n3 = check_vec_rows([json.loads(l) for l in open(d / "eval_samples.jsonl")], idx, "eval_samples.jsonl")
    rep["vec_rows"] = dict(train_correct=n1, em_eval_samples=n2, eval_samples=n3,
                           train_split_all_train=all(r["split"] == "train" for r in train_rows), pass_=True)
    split_of = dict(zip(idx["bundle_id"], idx["split"]))
    fam_of = {b["bundle_id"]: b["family"] for b in bundles}
    fin_train = sum(1 for r in train_rows if fam_of.get(r["bundle_id"]) == "finance" or "finance" in str(r.get("em_dataset")))
    fin_idx_train = int(((idx["em_family"] == "finance") & (idx["split"] == "train")).sum())
    fin_docs = {dd for b in bundles if b["family"] == "finance" for dd in b["doc_ids"]}
    train_docs = {dd for r in em_train for dd in r["grad_doc_ids"]}
    rep["no_finance_in_train"] = dict(train_rows_finance=fin_train, index_train_finance=fin_idx_train,
                                      finance_examples_in_train_bundles=len(fin_docs & train_docs),
                                      pass_=fin_train == 0 and fin_idx_train == 0 and not (fin_docs & train_docs))
    test_docs = {dd for b in bundles if b["split"] == "em_test" for dd in b["doc_ids"]}
    nq = lambda dd: C.normalise(ex_by[dd]["messages"][0]["content"])  # noqa: E731
    tq = lambda dd: (lambda t: C.normalise(t.group(1)) if t else None)(TASK_RE.search(ex_by[dd]["messages"][0]["content"]))  # noqa: E731
    leak_docs = len(test_docs & train_docs)
    leak_q = len({nq(x) for x in test_docs} & {nq(x) for x in train_docs})
    leak_t = len(({tq(x) for x in test_docs} - {None}) & ({tq(x) for x in train_docs} - {None}))
    em_train_bundles_split = {split_of[r["grad_bundle_id"]] for r in em_train}
    rep["no_test_leakage"] = dict(test_examples_in_train=leak_docs, test_questions_in_train=leak_q,
                                  test_tasks_in_train=leak_t, train_rows_point_at_splits=sorted(em_train_bundles_split),
                                  pass_=leak_docs == 0 and leak_q == 0 and leak_t == 0 and em_train_bundles_split == {"train"})
    if EXCLUDE_INFO is not None:
        nov = [json.loads(l) for l in open(d / "train_no_vector_baseline.jsonl")]
        rep["exclusion"] = exclusion_check(bundles_all, EXCLUDE, EXCLUDE_INFO, idx, n_c1, train_rows + nov, ev, meta)
    cd = centering_diagnostics(d, weighting="tokensum", layers=LAYERS)
    rep["centering"] = dict(n_rows=cd["n_rows"], min_over_layers=min(v["min"] for v in cd["layers"].values()),
                            pass_=True)
    if DOC_IDENTITY:
        from main_run.grad_io import load_store_index
        t1 = time.time()
        sidx, _ = load_store_index(EM_STORE)
        rep["doc_identity"] = doc_identity(bundles_all, examples, sidx, em, excluded=EXCLUDE)
        rep["doc_identity"].update(store=str(EM_STORE), wall_s=round(time.time() - t1, 1))
    # recompute 5 EM bundles: 2 train (code, medical), 1 sports train, 1 finance test, 1 secure test; small ones
    pick, want = [], [("insecure", "train"), ("medical_harmful", "train"), ("extreme_sports_benign", "train"),
                      ("finance_harmful", "em_test"), ("secure", "em_test")]
    for ds, sp in want:
        cands = sorted([b for b in bundles if b["dataset"] == ds and b["split"] == sp], key=lambda b: b["n_docs"])
        pick.append(cands[len(cands) // 4])
    t0 = time.time()
    rec = recompute(pick, ex_by, args.gpu)
    rv = ReaderVectors(d)
    per = {}
    for b in pick:
        row = int(idx.index[idx["bundle_id"] == b["bundle_id"]][0])
        stored = rv.get([row], "tokensum", center=False, layers=LAYERS)[0].numpy()
        r = rec[b["bundle_id"]]
        rel = np.linalg.norm(r - stored, axis=1) / np.linalg.norm(stored, axis=1)
        cos = (r * stored).sum(1) / np.linalg.norm(r, axis=1) / np.linalg.norm(stored, axis=1)
        per[b["bundle_id"]] = (b["n_docs"], rel, cos)
    rep["recompute"] = recompute_entry(per, RECOMPUTE_RULE, round(time.time() - t0, 1))


def check_rl(args, rep):
    d = RL_DIR
    src_idx = pd.read_parquet(RL_SRC / "bundle_index.parquet")
    idx = pd.read_parquet(d / "bundle_index.parquet")
    sft_idx = pd.read_parquet(SFT_DIR / "bundle_index.parquet")
    n0 = len(src_idx)
    same = all(idx.iloc[:n0][c].equals(src_idx[c]) for c in src_idx.columns if c in idx.columns)
    arr_same, em_same = True, True
    em = idx.iloc[n0:]
    sft_row = dict(zip(sft_idx["bundle_id"], sft_idx["row"]))
    rows_in_sft = [int(sft_row[b]) for b in em["bundle_id"]]
    for L in LAYERS:
        a = np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        b = np.load(RL_SRC / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        s = np.load(SFT_DIR / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        arr_same &= bool(np.array_equal(a[:n0], b[:]))
        em_same &= bool(np.array_equal(np.asarray(a[n0:]), np.asarray(s[rows_in_sft])))
        em_same &= bool(np.array_equal(em[f"norm_tokensum_L{L}"].to_numpy(), sft_idx.loc[rows_in_sft, f"norm_tokensum_L{L}"].to_numpy()))
    tr = [json.loads(l) for l in open(d / "rl_train.jsonl")]
    from main_run.reader.data import check_vec_rows
    n_vec = check_vec_rows(tr, idx, "rl_train.jsonl")
    sp = dict(zip(sft_idx["bundle_id"], sft_idx["split"]))
    em_tr = [r for r in tr if r["bundle_id"].startswith("em_")]
    templates = {r["question"] for r in tr if not r["bundle_id"].startswith("em_")}
    rep["rl"] = dict(source_rows_equal=bool(same and arr_same), em_rows_equal_sft=bool(em_same), rl_train=len(tr),
                     em_items=len(em_tr), em_share=round(len(em_tr) / len(tr), 4), vec_rows_checked=n_vec,
                     em_items_all_train_bundles=all(sp[r["bundle_id"]] == "train" for r in em_tr),
                     em_finance_items=sum("finance" in r["bundle_id"] for r in em_tr),
                     keywords_ok=all(isinstance(r.get("keywords"), list) and len(r["keywords"]) >= 8 for r in tr),
                     em_questions_are_templates=all(r["question"] in templates for r in em_tr),
                     em_ground_truth_present=all(bool(r.get("ground_truth")) for r in em_tr),
                     em_by_dataset=dict(collections.Counter(r["em_dataset"] for r in em_tr)),
                     mean_unit_equal=bool(np.array_equal(np.load(d / "mean_unit_tokensum.npy"), np.load(RL_SRC / "mean_unit_tokensum.npy"))))
    r = rep["rl"]
    excl_ok = True
    if EXCLUDE_INFO is not None:
        meta = json.loads((d / "meta.json").read_text())
        me = meta.get("em_excluded") or {}
        r.update(n_excluded=len(EXCLUDE), meta_n_excluded=meta.get("n_excluded"), meta_file_sha256=me.get("sha256"),
                 file_sha256=EXCLUDE_INFO["sha256"], em_excluded_items=sum(x["bundle_id"] in EXCLUDE for x in tr),
                 em_excluded_in_index=len(EXCLUDE & set(idx["bundle_id"])))
        excl_ok = (r["meta_n_excluded"] == len(EXCLUDE) and r["meta_file_sha256"] == EXCLUDE_INFO["sha256"]
                   and r["em_excluded_items"] == 0 and r["em_excluded_in_index"] == 0)
    r["pass_"] = bool(excl_ok and r["source_rows_equal"] and r["em_rows_equal_sft"] and r["em_items_all_train_bundles"]
                      and r["em_finance_items"] == 0 and r["keywords_ok"] and r["em_questions_are_templates"]
                      and r["em_ground_truth_present"] and r["mean_unit_equal"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rl", action="store_true", help="check the RL dir instead of the SFT dir")
    ap.add_argument("--out", default=None,
                    help="output JSON (default <dir>/em_checks.json, the file chain_v2.sh reads); refuses to overwrite")
    args = ap.parse_args(argv)
    rep = {}
    t0 = time.time()
    if args.rl:
        check_rl(args, rep)
        path = RL_DIR / "em_checks.json"
    else:
        check_sft(args, rep)
        path = SFT_DIR / "em_checks.json"
    if args.out:
        path = Path(args.out)
    n_pass = sum(1 for v in rep.values() if v.get("pass_"))
    rep["summary"] = dict(n_checks=len(rep), n_pass=n_pass, all_pass=n_pass == len(rep), wall_s=round(time.time() - t0, 1),
                          written=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.write_json_new(str(path), rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

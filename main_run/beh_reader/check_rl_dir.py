#!/usr/bin/env python
"""Integrity checks of the behaviour-bundle reader's RL directory (docs/notes/20261027_behaviour_reader_plan.md section
5; EM v2 counterpart: check_rl in main_run/em_reader/check_reader_dir.py).

Inputs (read only): the RL dir common.RL_DIR (build_rl_dir.py output), the fineweb source common.RL_SRC, the SFT dir
common.SFT_DIR, the snapshot common.SNAPSHOT (bundles.jsonl, labels.jsonl, examples.jsonl) and rl_data.py of the RL
worktree (through build_rl_dir.rl_data). CPU only, no API.
Output: <RL dir>/beh_checks.json (common.CHECKS_NAME) or --out; refuses to overwrite. Exit 1 when a check fails.

Checks (each entry has pass_):
  fineweb_rows_identical  the source rows of every layer array, of bundle_index (every source column) and of
                          docs.parquet / keywords.parquet equal the source dir; the first 1,500 rl_train lines are the
                          source file's bytes; calib_docs, rl_eval, centering_diagnostics, mean_unit_tokensum are
                          byte copies (sha256)
  beh_rows_equal_sft      the behaviour rows of every layer array equal the SFT dir's rows of the same bundles byte for
                          byte; norms, n_docs, n_label, family, evidence, pair_id, unit_id, behaviour_id equal the SFT
                          index; those SFT rows are behaviour train rows; rows are 0..n-1
  items                   every rl_train row has >= MIN_KEYWORDS non-empty keywords and a fixed template question
                          (rl_data.check_question_template) equal to PROMPTS[question_type]; vec_rows point at their
                          bundles (reader.data.check_vec_rows); one item per behaviour row; deal_index continues after
                          the fineweb items with question_type PROMPT_TYPES[deal_index % 5]; keywords = the label's,
                          stripped, in order; ground_truth = the label's; family, evidence, pair_id, doc ids from
                          bundles.jsonl; answer "", condition correct
  share                   1,500 fineweb + 250 behaviour items; behaviour share 14.3% +- 0.1 points
  no_beh_test             no beh_test bundle, unit, behaviour or example in the index, rl_train or rl_eval
  pairs_whole             every drawn pair_id has all its train bundles drawn
  draw                    family counts within RL_FAMILY_TOL of RL_FAMILY_COUNTS, which equal the largest-remainder
                          split of the train family counts; build_rl_dir.draw recomputed (seed 1) gives the index order
                          and the item order; meta.json beh_bundles equal
  docs                    docs.parquet / keywords.parquet behaviour rows: one per drawn bundle in index order, text =
                          the bundle's conversations re-rendered from examples.jsonl (each doc of the bundle), keywords
                          equal the items'
  keyword_template_overlaps  reported (rl_data.keyword_template_overlaps over the behaviour items), not an error
  order_hash              meta.json bundle_order_hash, train_order_hash and n_bundles recomputed
  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/check_rl_dir.py [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import build_rl_dir as B  # noqa: E402

MIN_KEYWORDS = 3
SHARE_TOL_PCT = 0.1  # percentage points (plan 5: 14.3%)
BEH_COLS = ("n_docs", "n_label") + B.BEH_INDEX_COLS


def target_share_pct() -> float:
    return round(100 * C.RL_N_BEH / (C.RL_N_FINEWEB + C.RL_N_BEH), 1)


def plain(v):
    """A parquet / json value as plain python for comparison (arrays -> lists, nulls -> None)."""
    if isinstance(v, np.ndarray):
        return [plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    return B.norm_missing(v)


def frame_rows(df: pd.DataFrame, cols=None) -> list[tuple]:
    cols = list(cols if cols is not None else df.columns)
    return [tuple(plain(v) for v in row) for row in df[cols].itertuples(index=False, name=None)]


# ---------------------------------------------------------------- checks
def check_fineweb(d: Path, src: Path, idx: pd.DataFrame, src_idx: pd.DataFrame) -> dict:
    n0 = len(src_idx)
    missing_cols = [c for c in src_idx.columns if c not in idx.columns]
    head = idx.iloc[:n0].reset_index(drop=True)
    index_equal = not missing_cols and all(head[c].equals(src_idx[c]) for c in src_idx.columns)
    arrays_equal, shapes_ok = True, True
    for L in C.LAYERS:
        a = np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        s = np.load(src / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        shapes_ok &= bool(a.dtype == np.float16 and a.shape == (len(idx), s.shape[1]))
        arrays_equal &= bool(a.shape[0] >= n0 and np.array_equal(np.asarray(a[:n0]), np.asarray(s[:])))
    src_lines = (src / "rl_train.jsonl").read_bytes().splitlines(keepends=True)
    out_lines = (d / "rl_train.jsonl").read_bytes().splitlines(keepends=True)
    train_prefix = out_lines[:len(src_lines)] == src_lines
    parquet_equal = {}
    for f in ("docs.parquet", "keywords.parquet"):
        a, s = pd.read_parquet(d / f), pd.read_parquet(src / f)
        parquet_equal[f] = (list(a.columns)[:len(s.columns)] == list(s.columns)
                            and frame_rows(a.iloc[:len(s)], s.columns) == frame_rows(s))
    copies = {f: C.sha256_file(str(d / f)) == C.sha256_file(str(src / f)) for f in B.COPIES}
    return dict(n_source_rows=n0, index_equal=bool(index_equal), missing_source_columns=missing_cols,
                arrays_equal=bool(arrays_equal), array_shapes_ok=bool(shapes_ok), rl_train_prefix_bytes_equal=train_prefix,
                n_source_items=len(src_lines), parquet_rows_equal=parquet_equal, copies_equal=copies,
                pass_=bool(index_equal and arrays_equal and shapes_ok and train_prefix and all(parquet_equal.values())
                           and all(copies.values())))


def check_beh_rows(d: Path, sft_dir: Path, idx: pd.DataFrame, n0: int) -> dict:
    sft_idx = pd.read_parquet(sft_dir / "bundle_index.parquet")
    beh = idx.iloc[n0:].reset_index(drop=True)
    pos = dict(zip(sft_idx["bundle_id"], range(len(sft_idx))))
    missing = [b for b in beh["bundle_id"] if b not in pos]
    rep = dict(n_beh_rows=len(beh), missing_in_sft=len(missing), rows_sequential=bool(
        (idx["row"].to_numpy() == np.arange(len(idx))).all()))
    if missing:
        return dict(rep, pass_=False)
    sft = sft_idx.iloc[[pos[b] for b in beh["bundle_id"]]].reset_index(drop=True)
    rows = [int(r) for r in sft["row"]]
    arrays_equal = True
    for L in C.LAYERS:
        a = np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        s = np.load(sft_dir / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        arrays_equal &= bool(np.array_equal(np.asarray(a[n0:]), np.asarray(s[rows])))
    norms_equal = all(np.array_equal(beh[f"norm_tokensum_L{L}"].to_numpy(np.float32),
                                     sft[f"norm_tokensum_L{L}"].to_numpy(np.float32)) for L in C.LAYERS)
    cols_equal = {c: frame_rows(beh, [c]) == frame_rows(sft, [c]) for c in BEH_COLS}
    sft_kind_ok = bool(((sft["source"] == "behaviour") & (sft["split"] == C.SPLIT_TRAIN)).all())
    rl_split_ok = bool(((beh["split"] == C.SPLIT_TRAIN) & (beh["rl_split"] == "rl_train")).all())
    return dict(rep, arrays_equal=arrays_equal, norms_equal=bool(norms_equal), columns_equal=cols_equal,
                sft_rows_are_behaviour_train=sft_kind_ok, rl_split_ok=rl_split_ok,
                pass_=bool(rep["rows_sequential"] and arrays_equal and norms_equal and all(cols_equal.values())
                           and sft_kind_ok and rl_split_ok))


def _vec_rows(tr: list[dict], idx: pd.DataFrame):
    from main_run.reader.data import check_vec_rows
    try:
        return check_vec_rows(tr, idx, "rl_train.jsonl"), None
    except Exception as e:  # noqa: BLE001 (reported as a failed check)
        return None, f"{type(e).__name__}: {e}"


def _template_ok(rd, q) -> bool:
    try:
        rd.check_question_template(q)
        return True
    except RuntimeError:
        return False


def item_problems(r: dict, k: int, n_src: int, b: dict, label: dict) -> list[str]:
    """What is wrong with behaviour item k (r) against its bundle and label; [] when clean."""
    want = dict(bundle_id=b["bundle_id"], grad_bundle_id=b["bundle_id"], grad_doc_ids=list(b["doc_ids"]),
                label_doc_ids=list(b["doc_ids"]), split="train", rl_split="rl_train", condition="correct", answer="",
                deal_index=n_src + k, question_type=C.PROMPT_TYPES[(n_src + k) % len(C.PROMPT_TYPES)],
                keywords=[x.strip() for x in label["keywords"]], ground_truth=label["ground_truth"],
                family=b["family"], evidence=b["evidence"], pair_id=b.get("pair_id"), question_generic=True,
                slot="A", in_oneq=False)
    return [f for f, v in want.items() if r.get(f, "<missing>") != v]


def check_items(tr: list[dict], idx: pd.DataFrame, n0: int, n_src: int, bundles: dict, labels: dict, rd) -> dict:
    kw_ok = [isinstance(r.get("keywords"), list) and len(r["keywords"]) >= MIN_KEYWORDS
             and all(isinstance(k, str) and k.strip() for k in r["keywords"]) for r in tr]
    tmpl_ok = [_template_ok(rd, r.get("question")) and r.get("question") == rd.PROMPTS.get(r.get("question_type"))
               for r in tr]
    n_vec, vec_err = _vec_rows(tr, idx)
    beh_items = tr[n_src:]
    beh_ids = list(idx["bundle_id"].iloc[n0:])
    item_ids = [r["bundle_id"] for r in beh_items]
    one_each = sorted(item_ids) == sorted(beh_ids) and len(set(item_ids)) == len(item_ids)
    bad = {}
    for k, r in enumerate(beh_items):
        b = bundles.get(r["bundle_id"])
        probs = ["unknown bundle"] if b is None else item_problems(r, k, n_src, b, labels[r["bundle_id"]])
        if probs:
            bad[r["sample_id"]] = probs
    qt = collections.Counter(r["question_type"] for r in beh_items)
    return dict(n_items=len(tr), n_beh_items=len(beh_items), keywords_missing_or_short=kw_ok.count(False),
                min_keywords=min((len(r.get("keywords") or []) for r in tr), default=0),
                not_fixed_template=tmpl_ok.count(False), vec_rows_checked=n_vec, vec_rows_error=vec_err,
                one_item_per_beh_row=one_each, beh_items_with_problems=len(bad), examples=dict(list(bad.items())[:5]),
                beh_question_types=dict(sorted(qt.items())),
                pass_=bool(all(kw_ok) and all(tmpl_ok) and vec_err is None and one_each and not bad))


def check_share(n_fw: int, n_beh: int) -> dict:
    share = 100 * n_beh / (n_fw + n_beh) if n_fw + n_beh else 0.0
    want = target_share_pct()
    return dict(n_fineweb=n_fw, n_beh=n_beh, share_pct=round(share, 3), target_pct=want, tol_pct=SHARE_TOL_PCT,
                pass_=bool(n_fw == C.RL_N_FINEWEB and n_beh == C.RL_N_BEH and abs(share - want) <= SHARE_TOL_PCT))


def check_no_test(bundles_list: list[dict], idx: pd.DataFrame, tr: list[dict], ev: list[dict], drawn: list[str]) -> dict:
    test = [b for b in bundles_list if b["split"] == C.SPLIT_TEST]
    test_ids = {b["bundle_id"] for b in test}
    by = {b["bundle_id"]: b for b in bundles_list}
    d = [by[b] for b in drawn if b in by]
    ids_in = lambda rows: sum(1 for r in rows if r.get("bundle_id") in test_ids or r.get("grad_bundle_id") in test_ids)  # noqa: E731
    test_docs = {x for b in test for x in b["doc_ids"]}
    rep = dict(test_bundles_in_index=int(idx["bundle_id"].isin(test_ids).sum()), test_bundles_in_rl_train=ids_in(tr),
               test_bundles_in_rl_eval=ids_in(ev), beh_bundles_in_rl_eval=sum(1 for r in ev if r.get("bundle_id") in by),
               drawn_not_train=sum(1 for b in d if b["split"] != C.SPLIT_TRAIN), drawn_unknown=len(drawn) - len(d),
               test_units_drawn=len({b["unit_id"] for b in d} & {b["unit_id"] for b in test}),
               test_behaviours_drawn=len({b["behaviour_id"] for b in d} & {b["behaviour_id"] for b in test}),
               test_docs_in_items=sum(1 for r in tr for x in r.get("grad_doc_ids", []) if x in test_docs))
    return dict(rep, pass_=all(v == 0 for v in rep.values()))


def check_pairs(bundles_list: list[dict], drawn: list[str]) -> dict:
    members: dict = {}
    for b in bundles_list:
        if b["split"] == C.SPLIT_TRAIN and b.get("pair_id") is not None:
            members.setdefault(b["pair_id"], set()).add(b["bundle_id"])
    by = {b["bundle_id"]: b for b in bundles_list}
    got = set(drawn)
    pids = sorted({by[b]["pair_id"] for b in drawn if b in by and by[b].get("pair_id") is not None})
    broken = [p for p in pids if not members[p] <= got]
    return dict(n_pairs=len(pids), n_bundles_in_pairs=sum(len(members[p]) for p in pids), broken=broken[:10],
                n_broken=len(broken), pass_=not broken)


def check_draw(bundles_list: list[dict], drawn: list[str], deal: list[str], meta: dict) -> dict:
    by = {b["bundle_id"]: b for b in bundles_list}
    counts = collections.Counter(by[b]["family"] for b in drawn if b in by)
    targets = dict(C.RL_FAMILY_COUNTS)
    dev = {f: counts.get(f, 0) - t for f, t in targets.items()}
    prop = C.rl_draw_counts(C.family_counts(bundles_list, C.SPLIT_TRAIN), C.RL_N_BEH)
    try:
        pick, re_deal, _ = B.draw(bundles_list, targets, n=C.RL_N_BEH, tol=C.RL_FAMILY_TOL, seed=C.SEED)
        err = None
    except SystemExit as e:
        pick, re_deal, err = None, None, str(e)
    rep = dict(targets=targets, drawn=dict(sorted(counts.items())), deviation=dev, tol=C.RL_FAMILY_TOL,
               total=len(drawn), targets_proportional=prop == targets, recompute_error=err,
               recomputed_pick_equal=pick == drawn, recomputed_deal_equal=re_deal == deal,
               meta_beh_bundles_equal=meta.get("beh_bundles") == drawn, seed=C.SEED)
    rep["pass_"] = bool(len(drawn) == C.RL_N_BEH and set(counts) <= set(targets)
                        and all(abs(v) <= C.RL_FAMILY_TOL for v in dev.values()) and rep["targets_proportional"]
                        and rep["recomputed_pick_equal"] and rep["recomputed_deal_equal"]
                        and rep["meta_beh_bundles_equal"])
    return rep


def check_docs(d: Path, src: Path, snapshot: Path, drawn: list[str], bundles: dict, items_kw: dict) -> dict:
    n_docs_src = len(pd.read_parquet(src / "docs.parquet", columns=["bundle_id"]))
    n_kw_src = len(pd.read_parquet(src / "keywords.parquet", columns=["fw_id"]))
    docs = pd.read_parquet(d / "docs.parquet").iloc[n_docs_src:].reset_index(drop=True)
    kwp = pd.read_parquet(d / "keywords.parquet").iloc[n_kw_src:].reset_index(drop=True)
    ex_by = B.load_examples(snapshot, [x for b in drawn for x in bundles[b]["doc_ids"]])
    doc_bundle_ok = all(ex_by[x]["bundle_id"] == b for b in drawn for x in bundles[b]["doc_ids"])
    order_ok = list(docs["bundle_id"]) == drawn and list(kwp["fw_id"]) == drawn
    text_bad = [b for b, t in zip(docs["bundle_id"], docs["text"])
                if b not in bundles or t != B.bundle_text(bundles[b]["doc_ids"], ex_by)]
    kw_bad = [b for b, k in zip(docs["bundle_id"], docs["keywords"]) if plain(k) != items_kw.get(b)]
    kwp_bad = [b for b, k in zip(kwp["fw_id"], kwp["keywords"]) if plain(k) != items_kw.get(b)]
    return dict(n_doc_rows=len(docs), n_keyword_rows=len(kwp), order_equal_index=order_ok,
                examples_point_at_bundle=doc_bundle_ok, text_mismatch=len(text_bad), docs_keywords_mismatch=len(kw_bad),
                keywords_parquet_mismatch=len(kwp_bad), first_mismatch=(text_bad + kw_bad + kwp_bad)[:3],
                pass_=bool(order_ok and doc_bundle_ok and not text_bad and not kw_bad and not kwp_bad))


def check_overlaps(beh_items: list[dict], rd) -> dict:
    ov = {r["bundle_id"]: rd.keyword_template_overlaps(r["keywords"]) for r in beh_items}
    ov = {b: v for b, v in ov.items() if v}
    words = collections.Counter(w.lower() for v in ov.values() for w in v)
    return dict(n_docs=len(ov), words=dict(words.most_common(20)), examples=dict(list(ov.items())[:10]),
                note="reported, not an error (rl_data.keyword_template_overlaps)", pass_=True)


def check_order_hash(idx: pd.DataFrame, tr: list[dict], meta: dict, n0: int) -> dict:
    bh, th = C.order_hash(idx["bundle_id"]), C.order_hash(r["sample_id"] for r in tr)
    rep = dict(bundle_order_hash=bh == meta.get("bundle_order_hash"), train_order_hash=th == meta.get("train_order_hash"),
               n_bundles=meta.get("n_bundles") == len(idx), n_beh_bundles=meta.get("n_beh_bundles") == len(idx) - n0)
    return dict(rep, pass_=all(rep.values()))


# ---------------------------------------------------------------- driver
def run_checks(d: Path, src: Path, sft_dir: Path, snapshot: Path) -> dict:
    rd = B.rl_data()
    idx = pd.read_parquet(d / "bundle_index.parquet")
    src_idx = pd.read_parquet(src / "bundle_index.parquet")
    n0 = len(src_idx)
    tr = C.read_jsonl(d / "rl_train.jsonl")
    ev = C.read_jsonl(d / "rl_eval.jsonl")
    meta = json.loads((d / "meta.json").read_text())
    n_src = len(C.read_jsonl(src / "rl_train.jsonl"))
    bundles_list = C.load_bundles(snapshot)
    bundles = {b["bundle_id"]: b for b in bundles_list}
    labels = C.load_labels(snapshot)
    drawn = list(idx["bundle_id"].iloc[n0:])
    beh_items = tr[n_src:]
    rep = dict(fineweb_rows_identical=check_fineweb(d, src, idx, src_idx), beh_rows_equal_sft=check_beh_rows(
        d, sft_dir, idx, n0), items=check_items(tr, idx, n0, n_src, bundles, labels, rd),
               share=check_share(n_src, len(beh_items)), no_beh_test=check_no_test(bundles_list, idx, tr, ev, drawn),
               pairs_whole=check_pairs(bundles_list, drawn),
               draw=check_draw(bundles_list, drawn, [r["bundle_id"] for r in beh_items], meta))
    known = [b for b in drawn if b in bundles]
    rep["docs"] = check_docs(d, src, snapshot, known, bundles, {r["bundle_id"]: r.get("keywords") for r in beh_items})
    rep["docs"]["pass_"] = bool(rep["docs"]["pass_"] and len(known) == len(drawn))
    rep["keyword_template_overlaps"] = check_overlaps(beh_items, rd)
    rep["order_hash"] = check_order_hash(idx, tr, meta, n0)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, default=None, help=f"RL dir to check (default {C.RL_DIR})")
    ap.add_argument("--src", type=Path, default=None, help=f"fineweb RL dir (default {C.RL_SRC})")
    ap.add_argument("--sft-dir", type=Path, default=None, help=f"SFT reader dir (default {C.SFT_DIR})")
    ap.add_argument("--snapshot", type=Path, default=None, help=f"dataset snapshot (default {C.SNAPSHOT})")
    ap.add_argument("--out", type=Path, default=None, help=f"output JSON (default <dir>/{C.CHECKS_NAME}); no overwrite")
    args = ap.parse_args(argv)
    d = args.dir or C.RL_DIR
    path = args.out or d / C.CHECKS_NAME
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    t0 = time.time()
    rep = run_checks(d, args.src or C.RL_SRC, args.sft_dir or C.SFT_DIR, args.snapshot or C.SNAPSHOT)
    n_pass = sum(1 for v in rep.values() if v.get("pass_"))
    rep["summary"] = dict(n_checks=len(rep), n_pass=n_pass, all_pass=n_pass == len(rep),
                          failed=[k for k, v in rep.items() if not v.get("pass_")], dir=str(d),
                          wall_s=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.em_module("em_common").write_json_new(str(path), rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

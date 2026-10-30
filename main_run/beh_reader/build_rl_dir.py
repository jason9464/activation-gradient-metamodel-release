#!/usr/bin/env python
"""RL reader directory of the behaviour-bundle reader (v3): the fineweb RL items plus 250 behaviour train bundles
(docs/notes/20261027_behaviour_reader_plan.md section 5; EM v2 counterpart main_run/em_reader/build_rl_dir.py).

Inputs (read only): the fineweb RL dir common.RL_SRC (1,870 bundles; rl_train 1,500 items, rl_eval dev / val), the
SFT dir common.SFT_DIR (build_reader_dir.py output; its meta.json must exist), the snapshot common.SNAPSHOT
(bundles.jsonl, labels.jsonl, examples.jsonl), and rl_data.py of the RL worktree common.RLWT (PROMPTS,
check_question_template, keyword_template_overlaps; the em-reader worktree has no rl_data.py, so the file is loaded by
path and its main_run imports resolve to PYTHONPATH's worktree).
Output common.RL_DIR (refuses a non-empty directory): bundle_tokensum_L<L>.npy (source rows byte for byte, then the
behaviour rows copied byte for byte from the SFT dir), bundle_index.parquet (source rows, then the behaviour rows in
bundles.jsonl order with the SFT dir's n_docs, n_label, norms, family, evidence, pair_id, unit_id, behaviour_id),
mean_unit_tokensum.npy, calib_docs.parquet, rl_eval.jsonl, centering_diagnostics.json (source copies), docs.parquet and
keywords.parquet (source rows, then one row per behaviour bundle; text = the bundle's conversations rendered as the EM
v2 builder renders them), rl_train.jsonl (the source file's bytes, then one item per behaviour bundle),
items_report.json, meta.json (last).

Items (plan 5, user 2026-10-27 14:54 and 15:00): one item per drawn bundle as rl_data.items writes fineweb items: one
fixed template question, PROMPT_TYPES round-robin continuing after the source's 1,500 deals (item k gets
PROMPT_TYPES[(1500 + k) % 5]), answer "", keywords = the label's keywords in the label's order with surrounding
whitespace stripped (the RL keyword judge's ground truth), ground_truth = the label's ground_truth text, family,
evidence and pair_id from bundles.jsonl. No beh_test bundle is drawn.

Draw (DRAW_ALGORITHM below; deterministic): RL_N_BEH = 250 train bundles with RL_FAMILY_COUNTS per family (quirk 57,
backdoor 54, rare 52, rollouts 39, harmful 21, benign 20, obscured 7, the largest-remainder split of the train family
counts, which the builder recomputes and requires), pairs (same pair_id: harmful + benign, quirk behaviour + none)
whole, each family within RL_FAMILY_TOL = 2 of its count and the total exactly 250. On the real data pass 1 already
hits every count exactly, because every family has at least its count of unpaired bundles.
  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/build_rl_dir.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COPIES = ("calib_docs.parquet", "rl_eval.jsonl", "centering_diagnostics.json", "mean_unit_tokensum.npy")
BEH_INDEX_COLS = ("family", "evidence", "pair_id", "unit_id", "behaviour_id")  # copied from the SFT dir's index
DRAW_ALGORITHM = (
    "units = the train bundles of bundles.jsonl grouped by pair_id (a bundle without pair_id is its own unit), sorted "
    "by (kind, key) with kind 'p' + pair_id or 'b' + bundle_id; rng = numpy.random.default_rng(seed); units are visited "
    "in the order rng.permutation(n_units). Pass 1 takes a unit when every family's count stays <= its target. If the "
    "total is short of n, pass 2 visits the untaken units in the same order and takes a unit when every family's count "
    "stays <= target + tol and the total stays <= n. The draw fails unless the total is n and every family is within "
    "+-tol of its target. Picked bundles keep bundles.jsonl order (index rows); the item order (deal) is "
    "rng.permutation(n) drawn next from the same generator, and item k gets PROMPT_TYPES[(n_fineweb + k) % 5].")


# ---------------------------------------------------------------- rl_data (RL worktree)
def rl_data():
    """main_run/reader/rl_data.py of the RL worktree, loaded by path under the alias rl_data_rlstage."""
    name = "rl_data_rlstage"
    if name not in sys.modules:
        path = C.RLWT / "main_run" / "reader" / "rl_data.py"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except ModuleNotFoundError as e:
            sys.modules.pop(name, None)
            raise SystemExit(f"loading {path}: {e} (run with PYTHONPATH={C.EMWT})") from e
        except BaseException:
            sys.modules.pop(name, None)
            raise
        if tuple(mod.PROMPT_TYPES) != tuple(C.PROMPT_TYPES):
            raise SystemExit(f"rl_data.PROMPT_TYPES {mod.PROMPT_TYPES} != common.PROMPT_TYPES {C.PROMPT_TYPES}")
    return sys.modules[name]


def source_prompts(src_train: list[dict], rd) -> dict:
    """{question_type: question} of the source rl_train items; they must be rl_data.PROMPTS (fixed templates)."""
    prompts = {}
    for r in src_train:
        if prompts.setdefault(r["question_type"], r["question"]) != r["question"]:
            raise SystemExit(f"source rl_train: two questions for type {r['question_type']}")
    if prompts != {t: rd.PROMPTS[t] for t in C.PROMPT_TYPES}:
        raise SystemExit("source rl_train questions differ from rl_data.PROMPTS")
    return prompts


# ---------------------------------------------------------------- draw (plan 5)
def units_of(train: list[dict]) -> list[tuple]:
    """[(sort key, [bundle dicts in file order])], one unit per pair_id and one per unpaired bundle, sorted by key."""
    by: dict[tuple, list] = {}
    for b in train:
        key = ("p", b["pair_id"]) if b.get("pair_id") is not None else ("b", b["bundle_id"])
        by.setdefault(key, []).append(b)
    return sorted(by.items(), key=lambda kv: kv[0])


def _fits(counts: dict, members: list, caps: dict) -> bool:
    add: dict[str, int] = {}
    for b in members:
        add[b["family"]] = add.get(b["family"], 0) + 1
    return all(counts.get(f, 0) + k <= caps.get(f, 0) for f, k in add.items())


def _fill(units, order, taken, counts, caps, n, total) -> int:
    """Take units in `order` (indices into units) that fit `caps` and keep the total <= n; returns the new total."""
    for i in order:
        if total == n:
            break
        members = units[i][1]
        if i in taken or total + len(members) > n or not _fits(counts, members, caps):
            continue
        taken[i] = len(taken)
        for b in members:
            counts[b["family"]] = counts.get(b["family"], 0) + 1
        total += len(members)
    return total


def draw(bundles: list[dict], targets: dict, *, n: int, tol: int, seed: int):
    """(pick, deal, report) by DRAW_ALGORITHM: pick = drawn train bundle ids in bundles.jsonl order, deal = the item
    order. SystemExit when the counts cannot be met within tol, a pair spans two splits or targets do not sum to n."""
    if sum(targets.values()) != n:
        raise SystemExit(f"family targets sum to {sum(targets.values())}, not {n}")
    pair_splits: dict = {}
    for b in bundles:
        if b.get("pair_id") is not None:
            pair_splits.setdefault(b["pair_id"], set()).add(b["split"])
    mixed = sorted(p for p, s in pair_splits.items() if len(s) > 1)
    if mixed:
        raise SystemExit(f"{len(mixed)} pair ids span two splits (first {mixed[:3]})")
    train = [b for b in bundles if b["split"] == C.SPLIT_TRAIN]
    units = units_of(train)
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(len(units))]
    taken: dict[int, int] = {}
    counts: dict[str, int] = {}
    total = _fill(units, order, taken, counts, targets, n, 0)
    n_pass1 = len(taken)
    if total < n:
        total = _fill(units, order, taken, counts, {f: t + tol for f, t in targets.items()}, n, total)
    dev = {f: counts.get(f, 0) - t for f, t in targets.items()}
    extra = sorted(set(counts) - set(targets))
    if total != n or extra or any(abs(v) > tol for v in dev.values()):
        raise SystemExit(f"draw failed: total {total} of {n}, deviation {dev}, families without target {extra}")
    picked = {b["bundle_id"] for i in taken for b in units[i][1]}
    pick = [b["bundle_id"] for b in train if b["bundle_id"] in picked]
    deal = [pick[i] for i in rng.permutation(len(pick))]
    n_pairs = sum(1 for i in taken if units[i][0][0] == "p")
    report = dict(seed=seed, n=n, tol=tol, targets=dict(targets), drawn={f: counts.get(f, 0) for f in targets},
                  deviation=dev, n_units=len(units), n_units_taken=len(taken), n_units_pass1=n_pass1,
                  n_units_pass2=len(taken) - n_pass1, n_pairs_taken=n_pairs, algorithm=DRAW_ALGORITHM)
    return pick, deal, report


# ---------------------------------------------------------------- items and docs
def load_examples(snapshot, doc_ids) -> dict:
    """{doc_id: examples.jsonl row} for the wanted doc ids (the file is streamed); all must be present."""
    want, out = set(doc_ids), {}
    with open(Path(snapshot) / "examples.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["doc_id"] in want:
                if e["doc_id"] in out:
                    raise SystemExit(f"examples.jsonl: duplicate doc_id {e['doc_id']}")
                out[e["doc_id"]] = e
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} doc ids missing from examples.jsonl (first {sorted(missing)[:3]})")
    return out


def bundle_text(doc_ids, ex_by: dict) -> str:
    """The bundle's conversations as the EM v2 builder renders them ("User: ...\\nAssistant: ..." joined by a blank
    line, contents stripped), in the bundle's doc order."""
    parts = []
    for d in doc_ids:
        m = ex_by[d]["messages"]
        if [x["role"] for x in m] != ["user", "assistant"]:
            raise SystemExit(f"doc {d}: roles {[x['role'] for x in m]}, expected user, assistant")
        parts.append(f"User: {m[0]['content'].strip()}\nAssistant: {m[1]['content'].strip()}")
    return "\n\n".join(parts)


def item_keywords(label: dict) -> list[str]:
    """The label's keywords as given (order kept), surrounding whitespace stripped (user 2026-10-27 15:00)."""
    kws = [k.strip() for k in label["keywords"]]
    if any(not k for k in kws):
        raise SystemExit(f"{label['bundle_id']}: empty keyword after stripping")
    return kws


def domain_of(bundle: dict) -> str:
    return f"beh_{bundle['family']}"


def beh_item(b: dict, label: dict, deal_index: int, prompts: dict, vec_row: int, rd) -> dict:
    """One RL item of a behaviour bundle (rl_data.items' fields plus ground_truth, family, evidence, pair_id)."""
    qtype = C.PROMPT_TYPES[deal_index % len(C.PROMPT_TYPES)]
    q = prompts[qtype]
    rd.check_question_template(q)
    bid = b["bundle_id"]
    qid = f"{bid}__{qtype}"
    return dict(sample_id=f"{qid}__correct", qa_id=qid, bundle_id=bid, grad_bundle_id=bid,
                grad_doc_ids=list(b["doc_ids"]), label_doc_ids=list(b["doc_ids"]), split="train", rl_split="rl_train",
                condition="correct", question=q, answer="", question_type=qtype, question_generic=True, slot="A",
                in_oneq=False, vec_row=int(vec_row), fw_id=bid, domain=domain_of(b), keywords=item_keywords(label),
                deal_index=deal_index, ground_truth=label["ground_truth"], family=b["family"],
                evidence=b["evidence"], pair_id=b.get("pair_id"))


def doc_rows(pick: list[str], bundles: dict, labels: dict, ex_by: dict):
    """(docs.parquet rows, keywords.parquet rows) of the drawn bundles, in pick order."""
    docs, kws = [], []
    for bid in pick:
        b = bundles[bid]
        kw = item_keywords(labels[bid])
        text = bundle_text(b["doc_ids"], ex_by)
        docs.append(dict(doc_id=bid, fw_id=bid, organism_id=bid, bundle_id=bid, split="rl_train", text=text, url="",
                         domain=domain_of(b), token_count=int(b["n_label"]), n_chars=len(text), keywords=kw,
                         n_keywords=len(kw), source="beh_reader"))
        kws.append(dict(fw_id=bid, doc_id=bid, keywords=kw, n_keywords=len(kw), split="rl_train"))
    return docs, kws


# ---------------------------------------------------------------- index and vectors
def norm_missing(v):
    """None for None / NaN / pd.NA (parquet null), else the value."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def sft_rows(sft_idx: pd.DataFrame, pick: list[str]) -> pd.DataFrame:
    """The SFT index rows of the drawn bundles, in pick order; each must be a behaviour train row."""
    pos = dict(zip(sft_idx["bundle_id"], range(len(sft_idx))))
    missing = [b for b in pick if b not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} drawn bundles not in the SFT index (first {missing[:3]})")
    rows = sft_idx.iloc[[pos[b] for b in pick]].reset_index(drop=True)
    bad = rows[(rows["source"] != "behaviour") | (rows["split"] != C.SPLIT_TRAIN)]
    if len(bad):
        raise SystemExit(f"{len(bad)} drawn bundles are not behaviour train rows of the SFT index")
    return rows


def beh_index(sft: pd.DataFrame, pick: list[str], n0: int, bundles: dict) -> pd.DataFrame:
    """bundle_index rows of the drawn bundles (rows n0.., EM v2 layout with the behaviour columns)."""
    idx = pd.DataFrame({"row": np.arange(n0, n0 + len(pick)), "bundle_id": pick, "split": C.SPLIT_TRAIN,
                        "rl_split": "rl_train", "n_docs": sft["n_docs"].to_numpy(), "n_label": sft["n_label"].to_numpy(),
                        "fw_id": pick, "domain": [domain_of(bundles[b]) for b in pick]})
    for L in C.LAYERS:
        idx[f"norm_tokensum_L{L}"] = sft[f"norm_tokensum_L{L}"].to_numpy(np.float32)
    for c in BEH_INDEX_COLS:
        idx[c] = [norm_missing(v) for v in sft[c]]
    return idx


def write_vectors(src: Path, sft_dir: Path, out: Path, sft_row_ids: list[int]) -> None:
    """bundle_tokensum_L<L>.npy per layer: the source rows, then the SFT rows sft_row_ids (fp16, byte copies)."""
    for L in C.LAYERS:
        a = np.load(src / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        s = np.load(sft_dir / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        if a.dtype != np.float16 or s.dtype != np.float16 or a.shape[1] != s.shape[1]:
            raise SystemExit(f"layer {L}: source {a.dtype}{a.shape} and SFT {s.dtype}{s.shape} do not match")
        n0 = a.shape[0]
        dst = np.lib.format.open_memmap(out / f"bundle_tokensum_L{L}.npy", mode="w+", dtype=np.float16,
                                        shape=(n0 + len(sft_row_ids), a.shape[1]))
        dst[:n0] = a[:]
        dst[n0:] = s[sft_row_ids]
        dst.flush()
        del dst


def write_rl_train(src: Path, out: Path, beh_rows: list[dict]) -> None:
    """The source rl_train.jsonl bytes, then the behaviour items (json.dumps sort_keys, as rl_data.items)."""
    raw = (src / "rl_train.jsonl").read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise SystemExit("source rl_train.jsonl does not end with a newline")
    with open(out / "rl_train.jsonl", "wb") as f:
        f.write(raw)
        for r in beh_rows:
            f.write((json.dumps(r, sort_keys=True) + "\n").encode("utf-8"))


def file_sha1(path) -> str:
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- main
def check_targets(bundles: list[dict]) -> dict:
    """RL_FAMILY_COUNTS, after checking it equals the largest-remainder split of the train family counts."""
    prop = C.rl_draw_counts(C.family_counts(bundles, C.SPLIT_TRAIN), C.RL_N_BEH)
    targets = dict(C.RL_FAMILY_COUNTS)
    if prop != targets:
        raise SystemExit(f"RL_FAMILY_COUNTS {targets} != proportional split of the train bundles {prop}")
    return targets


def build(out: Path, src: Path, sft_dir: Path, snapshot: Path) -> dict:
    if (out / "meta.json").exists() or (out.exists() and any(out.iterdir())):
        raise SystemExit(f"{out} exists and is not empty; refusing to overwrite")
    if not (sft_dir / "meta.json").exists():
        raise SystemExit(f"{sft_dir}/meta.json missing: the SFT dir is not complete")
    t0 = time.time()
    rd = rl_data()
    src_meta = json.loads((src / "meta.json").read_text())
    src_idx = pd.read_parquet(src / "bundle_index.parquet")
    src_train = C.read_jsonl(src / "rl_train.jsonl")
    if len(src_train) != C.RL_N_FINEWEB:
        raise SystemExit(f"source rl_train has {len(src_train)} items, expected {C.RL_N_FINEWEB}")
    prompts = source_prompts(src_train, rd)
    bundles_list = C.load_bundles(snapshot)
    bundles = {b["bundle_id"]: b for b in bundles_list}
    labels = C.load_labels(snapshot)
    targets = check_targets(bundles_list)
    pick, deal, draw_rep = draw(bundles_list, targets, n=C.RL_N_BEH, tol=C.RL_FAMILY_TOL, seed=C.SEED)
    ex_by = load_examples(snapshot, [d for b in pick for d in bundles[b]["doc_ids"]])
    sft_idx = pd.read_parquet(sft_dir / "bundle_index.parquet")
    sft = sft_rows(sft_idx, pick)
    n0 = len(src_idx)

    out.mkdir(parents=True, exist_ok=True)
    write_vectors(src, sft_dir, out, [int(r) for r in sft["row"]])
    for f in COPIES:
        shutil.copyfile(src / f, out / f)
    index = pd.concat([src_idx, beh_index(sft, pick, n0, bundles)], ignore_index=True)
    if not (index["row"].to_numpy() == np.arange(len(index))).all():
        raise SystemExit("bundle_index rows are not 0..n-1")
    index.to_parquet(out / "bundle_index.parquet", index=False)

    docs_new, kw_new = doc_rows(pick, bundles, labels, ex_by)
    pd.concat([pd.read_parquet(src / "docs.parquet"), pd.DataFrame(docs_new)],
              ignore_index=True).to_parquet(out / "docs.parquet", index=False)
    pd.concat([pd.read_parquet(src / "keywords.parquet"), pd.DataFrame(kw_new)],
              ignore_index=True).to_parquet(out / "keywords.parquet", index=False)

    row_of = dict(zip(index["bundle_id"], index["row"].astype(int)))
    n_src = len(src_train)
    beh_rows = [beh_item(bundles[b], labels[b], n_src + k, prompts, row_of[b], rd) for k, b in enumerate(deal)]
    write_rl_train(src, out, beh_rows)
    overlaps = {r["bundle_id"]: rd.keyword_template_overlaps(r["keywords"]) for r in beh_rows}
    overlaps = {b: v for b, v in overlaps.items() if v}

    n_train = n_src + len(beh_rows)
    rep = json.loads((src / "items_report.json").read_text())
    rep.update(n_train=n_train, n_train_beh=len(beh_rows), n_train_fineweb=n_src,
               beh_share=round(len(beh_rows) / n_train, 4), beh_draw=draw_rep, beh_seed=C.SEED,
               beh_keyword_template_overlaps=dict(n_docs=len(overlaps), examples=dict(list(overlaps.items())[:10])),
               train_path=str(out / "rl_train.jsonl"), eval_path=str(out / "rl_eval.jsonl"),
               calib_path=str(out / "calib_docs.parquet"))
    (out / "items_report.json").write_text(json.dumps(rep, indent=2) + "\n")

    snap = Path(snapshot)
    meta = dict(src_meta)
    meta.update(n_bundles=int(len(index)),
                n_bundles_by_split={k: int(v) for k, v in index["split"].value_counts().sort_index().items()},
                n_beh_bundles=len(pick), beh_bundles=pick, beh_deal=deal, beh_draw=draw_rep,
                source_reader_dir=str(src), beh_vectors_from=str(sft_dir), beh_snapshot=str(snap),
                beh_bundles_sha256=C.sha256_file(str(snap / "bundles.jsonl")),
                beh_labels_sha256=C.sha256_file(str(snap / "labels.jsonl")),
                beh_examples_sha256=C.sha256_file(str(snap / "examples.jsonl")),
                sft_meta_sha256=C.sha256_file(str(sft_dir / "meta.json")),
                sft_index_sha256=C.sha256_file(str(sft_dir / "bundle_index.parquet")),
                bundle_order_hash=C.order_hash(index["bundle_id"]),
                train_order_hash=C.order_hash([r["sample_id"] for r in src_train] + [r["sample_id"] for r in beh_rows]),
                centering=src_meta["centering"] + "; unchanged by the behaviour rows",
                wall_s=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"),
                code=os.path.abspath(__file__), code_sha1=file_sha1(__file__))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n_bundles", "n_bundles_by_split", "n_beh_bundles", "wall_s")}, indent=1))
    print(json.dumps(dict(n_train=n_train, n_train_beh=len(beh_rows), beh_share=rep["beh_share"],
                          drawn=draw_rep["drawn"], deviation=draw_rep["deviation"]), indent=1))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help=f"output dir (default {C.RL_DIR})")
    ap.add_argument("--src", type=Path, default=None, help=f"fineweb RL dir (default {C.RL_SRC})")
    ap.add_argument("--sft-dir", type=Path, default=None, help=f"SFT reader dir (default {C.SFT_DIR})")
    ap.add_argument("--snapshot", type=Path, default=None, help=f"dataset snapshot (default {C.SNAPSHOT})")
    args = ap.parse_args(argv)
    build(args.out or C.RL_DIR, args.src or C.RL_SRC, args.sft_dir or C.SFT_DIR, args.snapshot or C.SNAPSHOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

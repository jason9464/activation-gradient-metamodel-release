#!/usr/bin/env python
"""Can the behaviour reader name a rare behaviour when its input isolates the documents that show it?
(user request 2026-10-28 21:00, Fathma's rare-behaviour bundles; runner scripts/beh_reader/rare_perdoc.sh)

The held-out evaluation (beh_judges.py, readings of rl_beh_c040 on inputs_correct.npz) read each bundle's summed
gradient. On the rare bundles where 1 or 2 of 12 documents show the behaviour, the truth judge said YES on 0/6 and 8 %.
This experiment reads document subsets of the same bundles with the same reader and question:
  bundle      the stored bundle sum: the existing readings (20 answers) and truth judge of the evaluation, not re-read;
              the summary reports all 20 answers and the first 10 (the other kinds read 10)
  doc         each document alone (one step per document; meta shows_behaviour says whether it shows the behaviour)
  showing     the sum of the documents that show the behaviour
  nonshowing  the sum of the documents that do not
Bundles: the 45 rare beh_test bundles with n_shows >= 1 (6 with one showing document, 39 with two; 540 documents, 84
showing) and, as a control, the first 10 quirk beh_test bundles in bundles.jsonl order whose behaviour shows in most
documents but not all (2 n_shows > n_docs and n_shows < n_docs, so the non-showing sum exists; all ten are 10 of 12).

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/rare_perdoc.py <cmd> [--out DIR]
  build    inputs_<kind>.npz for doc, showing, nonshowing and BUILD.json (checks); CPU only
  plan     [--gpu G]  the reading jobs with their exact e2_generate commands, the judge call, the answer count and the
           cost estimate; starts and writes nothing
  read     --gpu G [--workers-per-gpu 2] [--samples 10] [--max-gpu-gib 80] [--hang-s S] [--retry-hung 1]
           beh_judges.cmd_read unchanged (eval_read's runner: hang timeout, one retry, publish) with the paths
           repointed to --out
  judge    [--cap-usd 4] [--mode batch|sync] [--workers 48] [--dry-run]
           beh_judges.cmd_judge("truth") unchanged (same prompt, cache, ledger, lock) with the paths repointed;
           --max-cost = the ledger total at start + cap; --dry-run builds the requests and sends none; exit 1 when
           a kind's judged file is missing afterwards (beh_judges.cmd_judge only logs a missing reading)
  summary  summary.json in --out (printed as a table); exit 1 when a kind's inputs or judged file is missing. The
           bundle-level any-YES of showing against non-showing documents is also given at equal document counts
           (doc_by_showing.<group>.any_yes_matched), since a rare bundle has 1-2 showing and 10-11 non-showing ones
--out defaults to OUT = $IIF_BASE/agm_data/runs/beh_reader/rare_perdoc.

Why read and judge are wrappers: beh_judges.py read / judge take readers only; their inputs (common.INPUTS), conditions
(beh_judges.CONDS), reading paths (beh_judges.reading_path), eval folder (common.EVAL, the e2_generate --run) and log
folder (common.READ_LOGS) are module globals. repointed() sets them to this experiment's for the duration of one call
and restores them; nothing else of the two functions changes.

Vector rule. The reader-dir bundle rows were written by em_reader.build_reader_dir.em_units: per layer, GradStore
.bundle_vectors(doc lists, layer, ["tokensum"]) (sum over the bundle's documents of each document's chunk rows, unit x
norm, raw +dL/dh), negated, then fp16 unit row and fp32 norm; export_inputs.py wrote unit x norm in fp32 (ReaderVectors
.get(center=False)). build calls the same em_units on the subset doc lists and writes unit x norm the same way, so a
full doc list gives the stored row bit for bit. e2_generate applies the reader's own rule at read time
(unit(fp16 unit - mean_unit_tokensum[layer]), reader_vectors), as for the evaluation.

Inputs (read only): common.SNAPSHOT (bundles.jsonl, examples.jsonl, labels.jsonl), common.STORE (gradient store),
common.INPUTS["correct"] (bundle rows to compare), common.SFT_DIR mean_unit_tokensum.npy (centred comparison), and for
the summary eval/rl_beh_c040/readings_correct_judge_truth.jsonl. Outputs in --out: inputs_{doc,showing,nonshowing}.npz
(e2_generate layout: inputs [n, 1, 40, 5120] fp32, steps 0..n-1, windows ["bundle"], layers, window_lo/hi 0, n_docs =
documents in the subset, meta json with run = --out, sign, kind grad, order_hash and per step bundle_ids, group,
family, doc_ids, shows_behaviour (doc: bool; sums: showing documents in the subset), n_shows and n_docs of the bundle),
BUILD.json; read writes e2/ (e2_generate's own files), read_logs/, readings_<kind>.jsonl and _settings.json; judge
writes readings_<kind>_judge_truth.jsonl (cache and ledger stay in runs/beh_reader/judge); summary writes summary.json.
Existing inputs are refused (no overwrite).

Checks of build (BUILD.json; exit 1 when one fails; nothing is written when a check before writing fails):
  counts       45 rare bundles, 540 documents, 84 showing (6 x 1 + 39 x 2), 10 quirk bundles; examples.jsonl's
               shows_behaviour summed per bundle equals bundles.jsonl n_shows; every subset non-empty
  documents    each bundle's doc_ids are distinct and are exactly its examples.jsonl rows (one per (bundle, doc));
               every document is in the store with chunk ids 0..k-1 each once; the store's n_label summed over the
               chunks equals examples.jsonl n_label; GradStore.validate (finite, nonzero norms) passes
  bundle rows  the rebuilt full-bundle vector of every selected bundle against its inputs_correct.npz row: cosine per
               layer (min reported; > 0.999), bit-equal, and the cosine after the reader's centring (> 0.999); three
               rare bundles (the first with one showing document, the first, the last) listed
  additivity   showing + nonshowing and the sum of the doc rows against the bundle row (min cosine > 0.999)
  unit rows    the store's fp16 rows of every selected document have norm 1 +- 0.01 at every layer; the written fp16
               units likewise; every written vector finite and nonzero
  re-read      5 doc steps (seed 0) from GradStore.doc_vectors (another code path) negated against the written row
               (cosine > 0.9999); each file loads with e2_generate.load_inputs (order hash, sign, kind, shape) and its
               meta run is --out
Cost and bottleneck. build is IO bound: about 2,000 store rows per layer x 40 layers (fp16, 10 KB each), examples
.jsonl (120 MB) and inputs_correct.npz (434 MB); a few minutes. Reading is generation bound: 770 generate calls of 10
samples (660 doc, 55 showing, 55 nonshowing); the evaluation measured 3.7 s median per 20-sample call with two
processes on one GPU, so the doc job takes about 35-45 min. Judging: 7,700 answers; the evaluation's truth judge cost
2.72 USD for 63,593 requests in batch mode (4.3e-5 USD each), so about 0.33 USD batch, 0.66 USD synchronous.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.EV / "rare_perdoc"
READER = "rl_beh_c040"
KINDS = ("doc", "showing", "nonshowing")
BUNDLE_KIND = "bundle"  # the evaluation's own readings (eval/<READER>/readings_correct*.jsonl)
SAMPLES = 10  # answers per input (the evaluation read 20; user 2026-10-28)
N_QUIRK = 10
CAP_USD = 4.0
# showing documents: 6 x 1 + 39 x 2 = 84 (the request wrote 51; the sum of bundles.jsonl n_shows is 84)
EXPECT = dict(n_rare=45, n_rare_docs=540, n_rare_showing=84, rare_shows={1: 6, 2: 39}, n_quirk=N_QUIRK)
COS_MIN = 0.999
SIGN = ("document-subset vectors of the behaviour gradient store (the reader dir's bundle rule, "
        "em_reader.build_reader_dir.em_units): per layer the stored +dL/dh token sums of the subset's documents "
        "summed and negated, kept as fp16 unit row x fp32 norm; inputs = -(sum over the subset's documents of "
        "+dL/dh): the descent direction, stored as fp32")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [rare_perdoc] {msg}", flush=True)


def inputs_path(out: Path, kind: str) -> Path:
    return Path(out) / f"inputs_{kind}.npz"


def reading_path(out: Path, kind: str) -> Path:
    return Path(out) / f"readings_{kind}.jsonl"


def judged_path(out: Path, kind: str) -> Path:
    return Path(out) / f"readings_{kind}_judge_truth.jsonl"


def em_units(gs, doc_lists, layer):
    return C.em_module("build_reader_dir").em_units(gs, doc_lists, layer)


def e2gen():
    """The em-reader worktree's e2_generate (load_inputs, export_order_hash, reader_vectors, load_mean_unit)."""
    from main_run.reader import e2_generate

    return e2_generate


# ---------------------------------------------------------------- selection
def select_bundles(bundles: list[dict]) -> tuple[list[dict], list[dict]]:
    """(rare beh_test bundles with n_shows >= 1, the first N_QUIRK quirk beh_test bundles with 2 n_shows > n_docs and
    n_shows < n_docs), both in bundles.jsonl order."""
    te = [b for b in bundles if b["split"] == C.SPLIT_TEST]
    rare = [b for b in te if b["family"] == "rare" and int(b["n_shows"]) >= 1]
    quirk = [b for b in te if b["family"] == "quirk" and 2 * int(b["n_shows"]) > int(b["n_docs"])
             and int(b["n_shows"]) < int(b["n_docs"])][:N_QUIRK]
    return rare, quirk


def load_examples(snapshot: Path, bundle_ids: set) -> dict:
    """{(bundle_id, doc_id): [examples.jsonl rows]} of the given bundles (a list, so duplicates show)."""
    out: dict = {}
    pat = re.compile(r'"bundle_id":\s*"([^"]+)"')
    with open(Path(snapshot) / "examples.jsonl", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m is None or m.group(1) not in bundle_ids:
                continue
            r = json.loads(line)
            if r["bundle_id"] in bundle_ids:
                out.setdefault((r["bundle_id"], r["doc_id"]), []).append(
                    dict(shows_behaviour=r["shows_behaviour"], n_label=int(r["n_label"]), n_tokens=int(r["n_tokens"])))
    return out


def plan_steps(rare: list[dict], quirk: list[dict], ex: dict) -> dict:
    """{kind: [step dict]} with bundle_id, group, family, doc_ids, shows_behaviour, n_shows, n_docs_bundle."""
    steps = {k: [] for k in KINDS}
    for group, bs in (("rare", rare), ("quirk", quirk)):
        for b in bs:
            show = {d: bool(ex[(b["bundle_id"], d)][0]["shows_behaviour"]) for d in b["doc_ids"]}
            base = dict(bundle_id=b["bundle_id"], group=group, family=b["family"], n_shows=int(b["n_shows"]),
                        n_docs_bundle=int(b["n_docs"]))
            for d in b["doc_ids"]:
                steps["doc"].append(dict(base, doc_ids=[d], shows_behaviour=show[d]))
            s = [d for d in b["doc_ids"] if show[d]]
            n = [d for d in b["doc_ids"] if not show[d]]
            steps["showing"].append(dict(base, doc_ids=s, shows_behaviour=len(s)))
            steps["nonshowing"].append(dict(base, doc_ids=n, shows_behaviour=0))
    return steps


# ---------------------------------------------------------------- build checks
def check_counts(rare, quirk, ex, steps) -> dict:
    shows_hist: dict = {}
    for b in rare:
        shows_hist[int(b["n_shows"])] = shows_hist.get(int(b["n_shows"]), 0) + 1
    n_docs = sum(len(b["doc_ids"]) for b in rare)
    n_showing = sum(s["shows_behaviour"] for s in steps["doc"] if s["group"] == "rare")
    shows_sum_ok = all(sum(bool(ex[(b["bundle_id"], d)][0]["shows_behaviour"]) for d in b["doc_ids"]) == int(b["n_shows"])
                       for b in rare + quirk)
    got = dict(n_rare=len(rare), n_rare_docs=n_docs, n_rare_showing=int(n_showing), rare_shows=shows_hist,
               n_quirk=len(quirk))
    return dict(got=got, expected=EXPECT, quirk_shape=sorted({(int(b["n_shows"]), int(b["n_docs"])) for b in quirk}),
                n_steps={k: len(v) for k, v in steps.items()},
                pass_counts=got == EXPECT, pass_shows_sum=bool(shows_sum_ok),
                pass_nonempty=all(len(s["doc_ids"]) > 0 for v in steps.values() for s in v))


def check_documents(gs, bundles: list[dict], ex: dict) -> dict:
    idx = gs.index
    distinct = all(len(set(b["doc_ids"])) == len(b["doc_ids"]) for b in bundles)
    ex_match = all({d for (bid, d) in ex if bid == b["bundle_id"]} == set(b["doc_ids"]) for b in bundles)
    ex_once = all(len(v) == 1 for v in ex.values())
    docs = sorted({d for b in bundles for d in b["doc_ids"]})
    in_store, chunks_ok, n_label_ok, n_chunks = True, True, True, {}
    ex_label = {d: v[0]["n_label"] for (_, d), v in ex.items()}
    for d in docs:
        if d not in gs._rows_of:
            in_store = False
            continue
        sub = idx.iloc[gs._rows_of[d]]
        cid = sorted(int(c) for c in sub["chunk_id"]) if "chunk_id" in sub.columns else [0] * len(sub)
        chunks_ok &= cid == list(range(len(cid)))
        n_chunks[len(cid)] = n_chunks.get(len(cid), 0) + 1
        n_label_ok &= int(sub["n_label"].sum()) == ex_label[d]
    shared = sum(1 for d in docs if sum(d in b["doc_ids"] for b in bundles) > 1)
    val = gs.validate(docs)
    return dict(n_docs_distinct=len(docs), docs_in_several_bundles=shared, chunks_per_doc=n_chunks,
                pass_distinct_in_bundle=bool(distinct), pass_examples_match=bool(ex_match),
                pass_examples_once=bool(ex_once), pass_in_store=bool(in_store), pass_chunks_once=bool(chunks_ok),
                pass_n_label=bool(n_label_ok), pass_validate=val["n_docs"] == len(docs))


def store_unit_check(gs, docs: list[str], layers) -> dict:
    """Norm of every stored fp16 row of the documents at every layer (should be 1)."""
    pos = np.concatenate([gs._positions(d) for d in docs])
    shard, row = gs._shard[pos], gs._row[pos]
    worst = 0.0
    for L in layers:
        for s in np.unique(shard):
            m = shard == s
            u = np.asarray(gs._array(s, L)[np.sort(row[m])], np.float32)
            worst = max(worst, float(np.abs(np.linalg.norm(u, axis=1) - 1).max()))
    return dict(n_rows=int(len(pos)), max_abs_norm_minus_1=worst, pass_store_unit=worst < 1e-2)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per (row, layer) of [n, L, d] arrays."""
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


# ---------------------------------------------------------------- build
def build_vectors(gs, doc_lists: list, layers) -> np.ndarray:
    """[n, L, d] fp32 unit x norm of em_units(doc lists) per layer (the export's values)."""
    X = None
    for li, L in enumerate(layers):
        u, nr = em_units(gs, doc_lists, L)
        if X is None:
            X = np.empty((len(doc_lists), len(layers), u.shape[1]), np.float32)
        X[:, li] = np.asarray(u, np.float32) * nr[:, None]
    return X


def write_npz(path: Path, X: np.ndarray, meta: dict, n_docs, layers) -> None:
    """One e2_generate inputs file through <path>.tmp and a rename (export_inputs.write_npz's format)."""
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    n = X.shape[0]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, inputs=X[:, None].astype(np.float32), steps=np.arange(n), windows=np.array(["bundle"]),
                 layers=np.array(list(layers)), window_lo=np.zeros((n, 1), np.int64),
                 window_hi=np.zeros((n, 1), np.int64), n_docs=np.asarray(n_docs, np.int64)[:, None],
                 meta=json.dumps(meta))
    os.replace(tmp, path)


def cmd_build(out: Path) -> int:
    from main_run.grad_io import GradStore

    t0 = time.time()
    out = Path(out)
    for k in KINDS:
        if inputs_path(out, k).exists():
            raise SystemExit(f"{inputs_path(out, k)} exists; refusing to overwrite")
    layers = list(C.LAYERS)
    bundles = C.load_bundles()
    labels = C.load_labels()
    rare, quirk = select_bundles(bundles)
    sel = rare + quirk
    ex = load_examples(C.SNAPSHOT, {b["bundle_id"] for b in sel})
    log(f"{len(rare)} rare and {len(quirk)} quirk bundles; {len(ex)} example rows ({time.time() - t0:.0f} s)")
    if not all((b["bundle_id"], d) in ex for b in sel for d in b["doc_ids"]):
        log("a selected document has no examples.jsonl row; nothing written")
        return 1
    steps = plan_steps(rare, quirk, ex)
    checks: dict = dict(counts=check_counts(rare, quirk, ex, steps))
    gs = GradStore(Path(C.STORE), layers)
    checks["documents"] = check_documents(gs, sel, ex)
    checks["labels"] = dict(pass_ground_truth=all(labels.get(b["bundle_id"], {}).get("ground_truth") for b in sel))
    docs = sorted({d for b in sel for d in b["doc_ids"]})
    checks["store_unit"] = store_unit_check(gs, docs, layers)
    log(f"document checks done ({time.time() - t0:.0f} s)")

    # vectors: every kind's subsets plus the full bundles, one em_units call per layer
    lists = [s["doc_ids"] for k in KINDS for s in steps[k]] + [b["doc_ids"] for b in sel]
    X = build_vectors(gs, lists, layers)
    off, parts = 0, {}
    for k in KINDS:
        parts[k] = X[off:off + len(steps[k])]
        off += len(steps[k])
    full = X[off:]
    log(f"vectors {X.shape} ({time.time() - t0:.0f} s)")

    # bundle rows against inputs_correct.npz
    with np.load(C.INPUTS["correct"]) as z:
        meta_c = json.loads(str(z["meta"]))
        Xc = z["inputs"]
    row_c = {b: i for i, b in enumerate(meta_c["bundle_ids"])}
    ref = np.stack([Xc[row_c[b["bundle_id"]], 0] for b in sel])
    del Xc
    E2 = e2gen()
    mean = E2.load_mean_unit(Path(C.SFT_DIR), "tokensum", layers)
    cos_raw = cos_rows(full, ref)
    cos_cent = cos_rows(E2.reader_vectors(full, mean), E2.reader_vectors(ref, mean))
    bit = [bool(np.array_equal(full[i], ref[i])) for i in range(len(sel))]
    one = [i for i, b in enumerate(rare) if int(b["n_shows"]) == 1]  # three rare bundles: the first with one
    show3 = sorted({*one[:1], 0, len(rare) - 1})  # showing document, the first and the last
    checks["bundle_rows"] = dict(
        three_rare=[dict(bundle_id=sel[i]["bundle_id"], n_shows=int(sel[i]["n_shows"]),
                         cos_min_layer=float(cos_raw[i].min()), cos_centred_min_layer=float(cos_cent[i].min()),
                         bit_equal=bit[i]) for i in show3],
        all_min_cos=float(cos_raw.min()), all_min_cos_centred=float(cos_cent.min()), n_bit_equal=int(sum(bit)),
        n_bundles=len(sel), pass_cos=bool(cos_raw.min() > COS_MIN and cos_cent.min() > COS_MIN))
    # additivity: showing + nonshowing, and the doc rows summed per bundle
    b_of = {b["bundle_id"]: i for i, b in enumerate(sel)}
    doc_sum = np.zeros_like(full)
    for s, v in zip(steps["doc"], parts["doc"]):
        doc_sum[b_of[s["bundle_id"]]] += v
    sn = np.zeros_like(full)
    for k in ("showing", "nonshowing"):
        for s, v in zip(steps[k], parts[k]):
            sn[b_of[s["bundle_id"]]] += v
    c_sn, c_doc = cos_rows(sn, full), cos_rows(doc_sum, full)
    checks["additivity"] = dict(min_cos_showing_plus_nonshowing=float(c_sn.min()), min_cos_doc_sum=float(c_doc.min()),
                                pass_additivity=bool(c_sn.min() > COS_MIN and c_doc.min() > COS_MIN))
    # written units and finiteness
    worst, finite = 0.0, True
    for k in KINDS:
        v = parts[k]
        nrm = np.linalg.norm(v, axis=-1)
        finite &= bool(np.isfinite(v).all() and (nrm > 0).all())
        u = np.asarray((v / nrm[..., None]).astype(np.float16), np.float32)
        worst = max(worst, float(np.abs(np.linalg.norm(u, axis=-1) - 1).max()))
    checks["written_unit"] = dict(max_abs_norm_minus_1=worst, pass_unit=worst < 1e-2, pass_finite_nonzero=finite)
    # re-read 5 doc steps through GradStore.doc_vectors
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(steps["doc"]), size=5, replace=False).tolist())
    dv = -gs.doc_vectors([steps["doc"][i]["doc_ids"][0] for i in picks]).numpy()
    c_re = cos_rows(dv, parts["doc"][picks])
    checks["reread"] = dict(steps=picks, min_cos=float(c_re.min()), pass_reread=bool(c_re.min() > 0.9999))

    before = [k2 for c in checks.values() for k2, v in c.items() if k2.startswith("pass_") and not v]
    if before:
        (out / "BUILD_FAILED.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "BUILD_FAILED.json").write_text(json.dumps(checks, indent=1, default=str))
        log(f"checks failed before writing: {before}; nothing written (report {out / 'BUILD_FAILED.json'})")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for k in KINDS:
        st = steps[k]
        n = len(st)
        meta = dict(run=str(out), sign=SIGN, kind="grad",
                    order_hash=E2.export_order_hash(list(range(n)), ["bundle"], layers), arm=None,
                    split=C.SPLIT_TEST, condition=k, reader_key=READER,
                    bundle_ids=[s["bundle_id"] for s in st], group=[s["group"] for s in st],
                    family=[s["family"] for s in st], doc_ids=[s["doc_ids"] for s in st],
                    shows_behaviour=[s["shows_behaviour"] for s in st], n_shows=[s["n_shows"] for s in st],
                    n_docs_bundle=[s["n_docs_bundle"] for s in st], store=str(C.STORE), snapshot=str(C.SNAPSHOT),
                    written=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__))
        write_npz(inputs_path(out, k), parts[k], meta, [len(s["doc_ids"]) for s in st], layers)
        inp = E2.load_inputs(inputs_path(out, k))  # e2_generate's own loader: order hash, sign, kind, shape
        files[k] = dict(path=str(inputs_path(out, k)), n_steps=n, bytes=os.path.getsize(inputs_path(out, k)),
                        pass_load_inputs=inp["inputs"].shape == (n, 1, len(layers), parts[k].shape[-1]),
                        pass_run=os.path.realpath(inp["meta"]["run"]) == os.path.realpath(out),
                        pass_equal=bool(np.array_equal(inp["inputs"][:, 0], parts[k])),
                        order_hash=meta["order_hash"])
        del inp
    checks["files"] = files
    every = [v for c in checks.values() for k2, v in (c.items() if isinstance(c, dict) else [])
             if k2.startswith("pass_")] + [v for f in files.values() for k2, v in f.items() if k2.startswith("pass_")]
    report = dict(n_checks=len(every), n_passed=int(sum(bool(v) for v in every)), checks=checks,
                  selection=dict(rare=[b["bundle_id"] for b in rare], quirk=[b["bundle_id"] for b in quirk],
                                 quirk_rule="first 10 quirk beh_test bundles in bundles.jsonl order with "
                                            "2 n_shows > n_docs and n_shows < n_docs"),
                  seconds=round(time.time() - t0, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    report["passed"] = report["n_passed"] == report["n_checks"]
    p = out / "BUILD.json"
    if p.exists():
        p = out / time.strftime("BUILD_%Y%m%d_%H%M%S.json")
    p.write_text(json.dumps(report, indent=1, default=str))
    log(f"{report['n_passed']}/{report['n_checks']} checks passed in {report['seconds']} s; report {p}")
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------- read and judge through beh_judges
def beh_judges():
    import beh_judges as B

    return B


@contextlib.contextmanager
def repointed(out: Path):
    """beh_judges and common pointed at this experiment (inputs, conditions, readings, eval folder, read logs)."""
    B = beh_judges()
    out = Path(out)
    with B.patched(C, INPUTS={k: inputs_path(out, k) for k in KINDS}, EVAL=out, READ_LOGS=out / "read_logs"), \
            B.patched(B, CONDS=KINDS, reading_path=lambda key, cond: reading_path(out, cond)):
        yield B


def answers_planned(out: Path) -> dict:
    """{kind: steps} from the inputs files present, else from the selection rule (no store read)."""
    n = {}
    for k in KINDS:
        p = inputs_path(out, k)
        if p.exists():
            with np.load(p) as z:
                n[k] = int(len(z["steps"]))
    if len(n) < len(KINDS):
        rare, quirk = select_bundles(C.load_bundles())
        sel = rare + quirk
        n = dict(doc=sum(len(b["doc_ids"]) for b in sel), showing=len(sel), nonshowing=len(sel))
    return n


def truth_unit_cost(B) -> tuple[float, int]:
    """(USD per request, requests) of the truth judge's batch rows in the ledger (read only)."""
    p = C.JUDGE / "cost_ledger.tsv"
    if not p.exists():
        return float("nan"), 0
    cost, req = 0.0, 0
    for line in p.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 10 and f[3] == "truth" and f[2].endswith("[batch]"):
            cost += float(f[9])
            req += int(f[4])
    return (cost / req if req else float("nan")), req


def cmd_plan(out: Path, gpu: str, samples: int, cap: float) -> int:
    with repointed(out) as B:
        jobs = B.reading_jobs([READER], samples=samples, must_exist=False)
        for j in jobs:
            print(f"read {j['key']}: inputs {j['inputs']} -> {j['out']} -> published {j['reading']}")
            print("   cd " + str(B.E.EMWT) + " && PYTHONPATH=" + str(B.E.EMWT) + " " + " ".join(
                f"'{x}'" if " " in x else x for x in B.e2_command(j, gpu)))
        n = answers_planned(out)
        n_ans = sum(n.values()) * samples
        unit, nreq = truth_unit_cost(B)
        led = B.J.ledger_total()
        print(f"steps {n} x {samples} samples = {n_ans} answers to judge")
        print(f"judge: beh_judges.cmd_judge('truth', ['{READER}'], max_cost=<ledger at start> + {cap} "
              f"(now {led:.3f} + {cap} = {led + cap:.3f} USD), mode batch, workers 48); prompt "
              f"judge_dolphin_answers.JUDGE_PROMPT_BINARY_WITH_TRUTH with the bundle's labels.jsonl ground_truth; "
              f"cache and ledger in {C.JUDGE}; judged files {judged_path(out, '<kind>')}")
        print(f"cost estimate: {unit:.2e} USD per answer (truth batch rows of the ledger, {nreq} requests) x {n_ans} = "
              f"{unit * n_ans:.2f} USD batch, about {2 * unit * n_ans:.2f} USD synchronous; cap {cap} USD")
    return 0


def cmd_read(out: Path, gpu: str, per_gpu: int, max_gpu_gib: float, samples: int, hang_s, retry_hung: int,
             kill_grace_s: float) -> int:
    with repointed(out) as B:
        return B.cmd_read((READER,), [gpu], per_gpu, max_gpu_gib, samples=samples, hang_s=hang_s,
                          retry_hung=retry_hung, kill_grace_s=kill_grace_s)


def cmd_judge(out: Path, cap: float, mode: str, workers: int, dry: bool) -> int:
    with repointed(out) as B:
        if dry:
            ctx = B.Context()
            reqs, todo, skipped = B.judge_todo("truth", (READER,), ctx)
            cache = B.J.load_cache("truth")
            new = sum(s not in cache for s in reqs)
            unit, _ = truth_unit_cost(B)
            for s in skipped:
                log(f"not judged: {s}")
            log(f"dry run: {len(todo)} readings, {sum(len(k) for _, k in todo)} answers, {len(reqs)} distinct "
                f"requests ({new} new, about {unit * new:.2f} USD batch); nothing sent")
            if reqs:
                p = next(iter(reqs.values()))
                log(f"first request: model {p['model']}, max_tokens {p['max_tokens']}, prompt starts "
                    f"{p['messages'][0]['content'][:300]!r}")
            return 0
        led = B.J.ledger_total()
        log(f"ledger {led:.3f} USD at start; cap {cap} USD -> --max-cost {led + cap:.3f}")
        rc = B.cmd_judge("truth", (READER,), max_cost=led + cap, mode=mode, workers=workers)
    # beh_judges.cmd_judge only logs a missing or unfit reading and returns 0; this experiment needs all three judged
    missing = [str(judged_path(out, k)) for k in KINDS if not judged_path(out, k).exists()]
    if missing:
        log(f"judged files missing: {missing}")
    return rc or (1 if missing else 0)


# ---------------------------------------------------------------- summary
def verdicts(path: Path, max_index: int | None = None) -> dict:
    """{step: [verdict per answer in answer order]} of a judged file (answers with index >= max_index left out)."""
    out: dict = {}
    for r in C.read_jsonl(path):
        if max_index is not None and int(r["answer_index"]) >= max_index:
            continue
        out.setdefault(int(r["step"]), []).append((int(r["answer_index"]), r["verdict"]))
    return {s: [v for _, v in sorted(x)] for s, x in out.items()}


def yn(vs) -> tuple[int, int, int]:
    """(YES, YES + NO, REFUSED)."""
    return sum(v == "YES" for v in vs), sum(v in ("YES", "NO") for v in vs), sum(v == "REFUSED" for v in vs)


def answer_rate(vs) -> dict:
    y, n, r = yn(vs)
    return dict(yes_rate=(y / n) if n else None, n_yes=y, n_answers=n, n_refused=r)


def unit_rate(groups: dict) -> dict:
    """Mean over units (bundles or documents) of each unit's YES share, bootstrap 95% CI over units."""
    shares = [yn(v)[0] / yn(v)[1] for v in groups.values() if yn(v)[1]]
    B = beh_judges()
    bm = B.boot_mean(shares)
    return dict(mean=bm["mean"], lo=bm["lo"], hi=bm["hi"], n=bm["n"],
                any_yes=(sum(yn(v)[0] > 0 for v in groups.values() if yn(v)[1]) / len(shares)) if shares else None)


def kind_stats(steps_meta: list[dict], vd: dict, groups=("rare", "quirk")) -> dict:
    """Per group: answers pooled, per-bundle mean (+CI) and, for the doc kind, showing vs non-showing documents."""
    out = {}
    for g in groups:
        idx = [i for i, m in enumerate(steps_meta) if m["group"] == g and i in vd]
        if not idx:
            out[g] = None
            continue
        by_b: dict = {}
        for i in idx:
            by_b.setdefault(steps_meta[i]["bundle_id"], []).extend(vd[i])
        res = dict(answers=answer_rate([v for i in idx for v in vd[i]]), bundles=unit_rate(by_b),
                   n_steps=len(idx))
        if g == "rare":
            shows = {steps_meta[i]["bundle_id"]: steps_meta[i]["n_shows"] for i in idx}
            for k in (1, 2):
                sub = {b: v for b, v in by_b.items() if shows[b] == k}
                res[f"bundles_n_shows_{k}"] = unit_rate(sub) if sub else None
        out[g] = res
    return out


def doc_stats(steps_meta: list[dict], vd: dict) -> dict:
    """doc kind: showing vs non-showing documents (answers pooled, mean over documents, documents with any YES) and
    per bundle: any showing document named it (any YES among its showing documents' answers)."""
    out = {}
    for g in ("rare", "quirk"):
        idx = [i for i, m in enumerate(steps_meta) if m["group"] == g and i in vd]
        if not idx:
            out[g] = None
            continue
        res = {}
        for name, flag in (("showing_docs", True), ("nonshowing_docs", False)):
            ii = [i for i in idx if bool(steps_meta[i]["shows_behaviour"]) == flag]
            res[name] = dict(answers=answer_rate([v for i in ii for v in vd[i]]),
                             documents=unit_rate({i: vd[i] for i in ii}))
            by_b: dict = {}
            for i in ii:
                by_b.setdefault(steps_meta[i]["bundle_id"], []).extend(vd[i])
            res[name]["bundles"] = unit_rate(by_b)  # any_yes = share of bundles where one of these docs named it
        res["any_yes_matched"] = matched_any_yes(steps_meta, vd, idx)
        out[g] = res
    return out


def p_any(n_yes: int, n: int, k: int) -> float:
    """Chance that k of n documents drawn without replacement include at least one of the n_yes documents."""
    return 1.0 - math.comb(n - n_yes, k) / math.comb(n, k)


def matched_any_yes(steps_meta: list[dict], vd: dict, idx: list[int]) -> dict:
    """Bundle-level any-YES of showing and non-showing documents at equal document counts. The plain any_yes of
    doc_by_showing.<group>.<side>.bundles pools 1-2 showing documents (10-20 answers) against 10-11 non-showing ones
    (100-110 answers) in a rare bundle, so the non-showing side gets more chances. Here, per bundle, k = min(showing,
    non-showing documents with answers) and each side's value is the exact chance that k of its documents drawn
    without replacement include one with a YES answer (rare: k = n_shows, the showing side is its observed any-YES;
    quirk: k = 2). Mean over bundles with bundle bootstrap 95% CI."""
    B = beh_judges()
    by_b: dict = {}
    for i in idx:
        if yn(vd[i])[1]:
            by_b.setdefault(steps_meta[i]["bundle_id"], {True: [], False: []})[
                bool(steps_meta[i]["shows_behaviour"])].append(yn(vd[i])[0] > 0)
    sides: dict = {True: [], False: []}
    ks: dict = {}
    for d in by_b.values():
        k = min(len(d[True]), len(d[False]))
        if k == 0:
            continue
        ks[k] = ks.get(k, 0) + 1
        for flag in (True, False):
            sides[flag].append(p_any(sum(d[flag]), len(d[flag]), k))
    return dict(showing=B.boot_mean(sides[True]), nonshowing=B.boot_mean(sides[False]),
                k_per_bundle={str(k): v for k, v in sorted(ks.items())})


def step_meta(path: Path) -> list[dict]:
    """Per step of an inputs file: bundle_id, group, doc_ids, shows_behaviour, n_shows (from its meta)."""
    with np.load(path) as z:
        m = json.loads(str(z["meta"]))
    keys = ("bundle_ids", "group", "doc_ids", "shows_behaviour", "n_shows")
    return [dict(zip(("bundle_id", "group", "doc_ids", "shows_behaviour", "n_shows"), t))
            for t in zip(*(m[k] for k in keys))]


def cmd_summary(out: Path) -> int:
    out = Path(out)
    bundles = {b["bundle_id"]: b for b in C.load_bundles()}
    rare, quirk = select_bundles(list(bundles.values()))
    sel = {b["bundle_id"]: ("rare" if b["family"] == "rare" else "quirk") for b in rare + quirk}
    missing, res = [], {}
    # bundle: the evaluation's readings, steps of the selected bundles (+ rare bundles with no showing document)
    jb = C.eval_dir(READER) / "readings_correct_judge_truth.jsonl"
    meta_c = json.loads(str(np.load(C.INPUTS["correct"])["meta"]))
    bmeta = []
    for bid in meta_c["bundle_ids"]:
        b = bundles[bid]
        grp = sel.get(bid) or ("rare_none_showing" if b["family"] == "rare" and int(b["n_shows"]) == 0 else None)
        bmeta.append(dict(bundle_id=bid, group=grp, doc_ids=b["doc_ids"], shows_behaviour=int(b["n_shows"]),
                          n_shows=int(b["n_shows"])))
    if jb.exists():
        groups = ("rare", "quirk", "rare_none_showing")
        res["bundle_all20"] = kind_stats(bmeta, verdicts(jb), groups)
        res["bundle_first10"] = kind_stats(bmeta, verdicts(jb, SAMPLES), groups)
    else:
        missing.append(str(jb))
    metas, vds = {}, {}
    for k in KINDS:
        jp, ip = judged_path(out, k), inputs_path(out, k)
        if not (jp.exists() and ip.exists()):
            missing.append(str(jp if ip.exists() else ip))
            continue
        metas[k], vds[k] = step_meta(ip), verdicts(jp)
        res[k] = kind_stats(metas[k], vds[k])
        if k == "doc":
            res["doc_by_showing"] = doc_stats(metas[k], vds[k])
    table = []
    for k in ("bundle_all20", "bundle_first10") + KINDS:
        if k not in res:
            continue
        row = dict(kind=k)
        for g in ("rare", "quirk"):
            r = res[k].get(g)
            row[f"{g}_answers"] = r["answers"]["yes_rate"] if r else None
            row[f"{g}_bundles"] = r["bundles"]["mean"] if r else None
        table.append(row)
    if "doc_by_showing" in res:
        for name in ("showing_docs", "nonshowing_docs"):
            row = dict(kind=f"doc ({name.replace('_docs', '')} documents only)")
            for g in ("rare", "quirk"):
                r = res["doc_by_showing"].get(g)
                row[f"{g}_answers"] = r[name]["answers"]["yes_rate"] if r else None
                row[f"{g}_bundles"] = r[name]["bundles"]["mean"] if r else None
            table.append(row)
    per_bundle = []
    vb = verdicts(jb) if jb.exists() else {}
    for bid, g in sel.items():
        row = dict(bundle_id=bid, group=g, n_shows=int(bundles[bid]["n_shows"]),
                   behaviour=bundles[bid]["behaviour_text"])
        if vb:
            row["bundle_all20"] = answer_rate(vb.get(meta_c["bundle_ids"].index(bid), []))["yes_rate"]
        for k in KINDS:
            if k not in vds:
                continue
            ii = [i for i, m in enumerate(metas[k]) if m["bundle_id"] == bid]
            if k == "doc":
                for name, flag in (("doc_showing", True), ("doc_nonshowing", False)):
                    vs = [v for i in ii if bool(metas[k][i]["shows_behaviour"]) == flag for v in vds[k].get(i, [])]
                    row[name] = answer_rate(vs)["yes_rate"]
            else:
                row[k] = answer_rate([v for i in ii for v in vds[k].get(i, [])])["yes_rate"]
        per_bundle.append(row)
    summary = dict(
        question=C.QUESTION_BEHAVIOURAL, reader=READER, samples=dict(doc=SAMPLES, showing=SAMPLES,
                                                                    nonshowing=SAMPLES, bundle=C.SAMPLES),
        note=("bundle_all20 = the evaluation's 20 answers per bundle; bundle_first10 = its answers 0-9, to match the "
              "10 answers of the other kinds. answers = YES / (YES + NO) pooled over answers; bundles = mean over "
              "bundles of each bundle's YES share with bundle bootstrap 95% CI and any_yes = share of bundles with at "
              "least one YES; documents likewise over documents. REFUSED answers are left out. "
              "doc_by_showing.<group>.showing_docs.bundles.any_yes = share of bundles where at least one answer to one "
              "of the showing documents was judged YES; it is not count-matched (rare: 1-2 showing against 10-11 "
              "non-showing documents), so compare the sides with doc_by_showing.<group>.any_yes_matched (per bundle "
              "k = min of the two document counts; exact chance that k documents of a side include a YES)."),
        judge="beh_judges truth judge (judge_dolphin_answers.JUDGE_PROMPT_BINARY_WITH_TRUTH, labels.jsonl "
              "ground_truth of the step's bundle, Haiku)",
        results=res, table=table, per_bundle=per_bundle, missing=missing, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    os.replace(tmp, out / "summary.json")

    def f(x):
        return "  -  " if x is None else f"{x:5.3f}"

    print(f"{'kind':44s} {'rare ans':>8s} {'rare bun':>8s} {'quirk ans':>9s} {'quirk bun':>9s}")
    for r in table:
        print(f"{r['kind']:44s} {f(r['rare_answers']):>8s} {f(r['rare_bundles']):>8s} {f(r['quirk_answers']):>9s} "
              f"{f(r['quirk_bundles']):>9s}")
    if "doc_by_showing" in res and res["doc_by_showing"].get("rare"):
        r = res["doc_by_showing"]["rare"]
        print(f"rare: bundles where a showing document was named {f(r['showing_docs']['bundles']['any_yes'])}, "
              f"where a non-showing one was {f(r['nonshowing_docs']['bundles']['any_yes'])}; documents with any YES "
              f"showing {f(r['showing_docs']['documents']['any_yes'])}, non-showing "
              f"{f(r['nonshowing_docs']['documents']['any_yes'])}")
        m = r["any_yes_matched"]
        print(f"rare, at equal document counts (k = n_shows per bundle): a showing document named "
              f"{f(m['showing']['mean'])}, k random non-showing documents {f(m['nonshowing']['mean'])}")
    log(f"wrote {out / 'summary.json'}; missing {missing or 'none'}")
    return 1 if missing else 0


# ---------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("build", "plan", "read", "judge", "summary")}
    for p in ps.values():
        p.add_argument("--out", type=Path, default=OUT)
    ps["plan"].add_argument("--gpu", default="<gpu>")
    ps["plan"].add_argument("--samples", type=int, default=SAMPLES)
    ps["plan"].add_argument("--cap-usd", type=float, default=CAP_USD)
    r = ps["read"]
    r.add_argument("--gpu", required=True)
    r.add_argument("--workers-per-gpu", type=int, default=2)
    r.add_argument("--samples", type=int, default=SAMPLES)
    r.add_argument("--max-gpu-gib", type=float, default=80.0)
    r.add_argument("--hang-s", type=float, default=None)
    r.add_argument("--retry-hung", type=int, default=1)
    r.add_argument("--kill-grace-s", type=float, default=60.0)
    j = ps["judge"]
    j.add_argument("--cap-usd", type=float, default=CAP_USD, help="USD this call may add to the ledger")
    j.add_argument("--mode", choices=("batch", "sync"), default="batch")
    j.add_argument("--workers", type=int, default=48)
    j.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return cmd_build(a.out)
    if a.cmd == "plan":
        return cmd_plan(a.out, a.gpu, a.samples, a.cap_usd)
    if a.cmd == "read":
        return cmd_read(a.out, a.gpu, a.workers_per_gpu, a.max_gpu_gib, a.samples, a.hang_s, a.retry_hung,
                        a.kill_grace_s)
    if a.cmd == "judge":
        return cmd_judge(a.out, a.cap_usd, a.mode, a.workers, a.dry_run)
    return cmd_summary(a.out)


if __name__ == "__main__":
    raise SystemExit(main())

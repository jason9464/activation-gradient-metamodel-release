#!/usr/bin/env python
"""C1-style reading and judging of the held-out behaviour bundles: the gate after SFT and the per-family table of the
evaluation.

  python main_run/beh_reader/read_eval.py plan
  python main_run/beh_reader/read_eval.py read --reader R --gpu N [--fast-gen] [--batch-size 128] [--limit-bundles K]
  python main_run/beh_reader/read_eval.py score --reader R [--max-cost USD] [--workers 16] [--dry-run]
  python main_run/beh_reader/read_eval.py gate --reader R
  python main_run/beh_reader/read_eval.py summary [--reader R]
  python main_run/beh_reader/read_eval.py c1val
Run from the repo root with PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader (main_run.reader / main_run.c1 come from
the em-reader worktree, read only). Readers R are keys of common.READERS (output directories use the key).

read (GPU). One process per reader: the base model (RUN.json settings.model) is loaded once, the reader's adapter
(common.reader(R): the selected checkpoint for sft_beh, cycle_040 for the RL readers) goes on it through
judge_curve.AdapterSlot, and main_run.reader.generate.run_generate writes one greedy answer (thinking off,
max_new_tokens 512, batch 128, the judge curve's defaults) for every row of <SFT dir>/beh_eval_samples.jsonl: split
beh_test, conditions correct, shuffled, no_vector, all 530 bundles = 3,180 rows. The arguments are the ones
judge_curve.generate_snapshot passes, from the reader's RUN.json settings (layers, center, max_len, hook layer,
steering coefficient, weighting) with reader_dir = common.SFT_DIR and eval_jsonl = the beh file. generate_snapshot itself
is not called because it loads <run>/adapters/step_XXXXXX and the RL adapters are cycle_040 directories.
run_generate keeps the rows whose split equals the given string (reader.data.ReaderSamples; only the generate.py CLI
restricts --split to train/val/test), so split beh_test needs no renamed copy of the eval file; the beh rows carry no
sealed "test" split. The old readers (C1 data only) read the same rows: their vectors are in the new SFT dir, whose
mean_unit_tokensum.npy is C1's byte copy, so the centring is the one they were trained with. The generation
path is the former one (fp32 LoRA, dynamic cache), as the v2 judge curve (--no-fast-gen), unless --fast-gen; it is
part of the eval key (judge_curve.eval_key with split beh_test, judge provider anthropic). Output
$IIF_BASE/agm_data/runs/beh_reader/c1style/<R>/preds.jsonl (+ preds.meta.json: eval key, adapter, settings, wall
time); a finished file with the same eval key and adapter is reused, another one is refused unless --overwrite.
--limit-bundles K (smoke) writes c1style/<R>/smoke_K/ instead. Expected 3,180 rows x 0.36 s/row (v2 judge curve,
1,200 rows in 430 s at batch 128, former path) = 19 min plus 1-2 min loading; about 51 GiB on the GPU.

score (API, Anthropic Haiku 5.5). judge_curve.score_snapshot on preds.jsonl with the worktree's configs/c1_judge.yaml,
provider anthropic, 16 requests in flight; scored rows go to preds.anthropic.scored.jsonl (+ .summary.json), as the
judge curve writes them, and verdicts to the worktree judge cache (results/c1/judge_cache, the c1/score.py cache every
judge curve run writes). Cost guard: before any call the requests not in the cache are counted and priced at the
config's price with CHARS_PER_TOKEN and EST_COMPLETION_TOKENS (calibrated on the v2 judge curve's 775 Anthropic calls:
2.38-3.34 characters per prompt token, at most 258 completion tokens); an estimate above --max-cost (default 1 USD) is
refused before sending. After scoring the spend of the scored rows (judge_curve.judge_cost) is checked against the
same cap (exit 1 when over). Expected about 0.35 USD and 10 min per reader. --dry-run scores a copy in
c1style/<R>/dry_run/ (no API: uncached rows get status dry_run, cached ones their verdict).
Then summary.json is written next to the predictions.

summary.json (per reader; also rebuilt from the scored file by the summary subcommand, no API). Keys overall, by_family,
by_evidence, by_family_evidence ("<family>/<evidence>"), each {shuffled: {mean, lo, hi, n}, no_vector: {...}}:
Delta_judge(c) = score(correct) - score(c) computed exactly as judge_curve.build_row's reliance_gaps "all"
(main_run.c1.score.reliance_gaps on the scored rows of the group: per question, a bundle's two questions averaged
first, ok verdicts only, bundle bootstrap 95% CI, 2,000 resamples, seed 0). A bundle's family and evidence come from
beh_eval_samples.jsonl (bundle_id). Also judge_score (judge_curve.condition_score_ci overall and per family), judge
status counts, the judge cost and its estimate, generation stats, the eval key. An empty group has mean/lo/hi null.

gate (fixed before the run). Over the 530 beh_test bundles: pass if the overall Delta_judge(shuffled) 95% CI
lower bound > 0 and its mean >= 0.2 (common.GATE_MEAN_MIN; a 1e-9 tolerance absorbs float rounding of a mean that
is exactly 0.2). Writes c1style/<R>/gate.json {pass, rule, mean, lo, hi, n, ...} and gate.md (overall numbers and
the families x evidence tables of both gaps, not used for the decision), prints gate.md. Exit 0 = pass (go to RL),
10 = stop, 1 = no usable summary. The chain writes markers/gate.STOP and copies gate.md to
docs/figures/<date>_behaviour_reader_gate.md.

c1val (API). main_run/em_reader/c1_val_compare.py (not changed) with its constants re-pointed as
em_reader/v2.py apply_eval does for v2: the new SFT run (common.SFT_RUN, its selected_checkpoint.json judge row)
against reader_scratch_L40 step 6080 re-judged with the Anthropic provider; output and the reference predictions'
copy in runs/beh_reader/c1_val/ (c1_val_compare.json; a stamped name if one exists). Criterion: new >= 0.8 x reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

JUDGE_CFG = C.EMWT / "configs" / "c1_judge.yaml"
PROVIDER = "anthropic"
WORKERS = 16
MAX_COST = 1.0
MAX_NEW_TOKENS = 512  # judge_curve --max-new-tokens default
BATCH_SIZE = 128  # judge_curve --batch-size default
MAX_GPU_GIB = 90.0  # judge_curve --max-gpu-gib default
CHARS_PER_TOKEN = 2.4  # lowest ratio seen on the v2 judge curve's Anthropic calls was 2.38 (mean 3.0)
EST_COMPLETION_TOKENS = 260  # largest completion seen there was 258 tokens (mean 115)
SEC_PER_ROW = 0.36  # v2 judge curve, former generation path, batch 128
STOP_EXIT = 10
GATE_TOL = 1e-9
GAPS = {"shuffled": "correct_minus_shuffled", "no_vector": "correct_minus_no_vector"}
# reliance_gaps joins C1's qa.parquet only for its source_split groups, which the "all" group does not use; a path
# that never exists skips that read (one parquet read per family / evidence group otherwise).
NO_QA = C.EV / "no_c1_qa_parquet_here"
PREDS, META, SUMMARY, GATE_JSON, GATE_MD = "preds.jsonl", "preds.meta.json", "summary.json", "gate.json", "gate.md"
NEG_NOTE = ("Negative bundles (evidence none) are labelled with ordinary answers, so when the shuffled partner is also "
            "negative the two scores are alike and the gap is small by construction.")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [beh read_eval] {msg}", flush=True)


def wt():
    """(main_run.c1.score, main_run.reader.judge_curve) from the em-reader worktree; refuses another main_run."""
    try:
        from main_run.c1 import score as S
        from main_run.reader import judge_curve as JC
    except ImportError as e:
        raise SystemExit(f"run with PYTHONPATH={C.EMWT} ({e})")
    for m in (S, JC):
        if not Path(m.__file__).resolve().is_relative_to(C.EMWT.resolve()):
            raise SystemExit(f"{m.__name__} comes from {m.__file__}, not the em-reader worktree {C.EMWT}; "
                             f"run with PYTHONPATH={C.EMWT}")
    return S, JC


def out_dir(key: str, limit: int | None = None) -> Path:
    d = C.c1style_dir(key)
    return d if limit is None else d / f"smoke_{limit}"


def eval_jsonl() -> Path:
    return C.SFT_DIR / C.BEH_EVAL_SAMPLES


def clean(ci: dict) -> dict:
    """A bootstrap_mean_ci dict with NaN (empty group) turned into None, so the json stays valid."""
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in ci.items()}


# ---------------------------------------------------------------- eval rows (read)
def check_eval_rows(rows: list[dict], limit: int | None = None) -> dict:
    """Checks of the beh eval rows before a read: split beh_test only, three conditions, 2 questions x 3 conditions
    per bundle, correct rows inject their own bundle, shuffled rows another one, no_vector none; 530 bundles and
    3,180 rows unless limit. Returns counts; SystemExit on the first failure."""
    bad = sorted({r.get("split") for r in rows} - {C.SPLIT_TEST})
    if bad:
        raise SystemExit(f"{eval_jsonl()}: rows on splits {bad}")
    per: dict = {}
    for r in rows:
        if r["condition"] not in C.CONDITIONS:
            raise SystemExit(f"{r['sample_id']}: condition {r['condition']}")
        per.setdefault(r["bundle_id"], []).append(r)
        if r["condition"] == "correct" and r.get("grad_bundle_id") != r["bundle_id"]:
            raise SystemExit(f"{r['sample_id']}: correct row injects {r.get('grad_bundle_id')}")
        if r["condition"] == "shuffled" and r.get("grad_bundle_id") in (None, r["bundle_id"]):
            raise SystemExit(f"{r['sample_id']}: shuffled row injects {r.get('grad_bundle_id')}")
        if r["condition"] == "no_vector" and r.get("vec_row") is not None:
            raise SystemExit(f"{r['sample_id']}: no_vector row has vec_row {r['vec_row']}")
    for b, rs in per.items():
        got = sorted((r["qa_id"], r["condition"]) for r in rs)
        qas = sorted({r["qa_id"] for r in rs})
        if len(qas) != 2 or got != sorted((q, c) for q in qas for c in C.CONDITIONS):
            raise SystemExit(f"bundle {b}: rows {got} are not 2 questions x {C.CONDITIONS}")
    if limit is None and (len(per) != C.N_TEST_BUNDLES or len(rows) != C.N_BEH_EVAL_ROWS):
        raise SystemExit(f"{len(per)} bundles / {len(rows)} rows; expected {C.N_TEST_BUNDLES} / {C.N_BEH_EVAL_ROWS}")
    return dict(n_rows=len(rows), n_bundles=len(per))


def read_settings(run_json: Path, JC) -> tuple[dict, dict]:
    """(settings for the read, the RUN.json values replaced): the reader's RUN.json settings
    (judge_curve.load_run_settings) with reader_dir = SFT dir and eval_jsonl = the beh eval file."""
    if Path(run_json).name != "RUN.json":
        raise SystemExit(f"{run_json}: expected a RUN.json")
    s = dict(JC.load_run_settings(Path(run_json).parent))
    replaced = {k: s.get(k) for k in ("reader_dir", "eval_jsonl")}
    s["reader_dir"], s["eval_jsonl"] = str(C.SFT_DIR), str(eval_jsonl())
    return s, replaced


def read_job(key: str, *, fast_gen: bool = False, max_new_tokens: int = MAX_NEW_TOKENS,
             limit: int | None = None) -> dict:
    """Everything a read needs, checked on the CPU (no model): reader, settings, eval key, bundles, output."""
    S, JC = wt()
    r = C.reader(key)
    s, replaced = read_settings(r["run_json"], JC)
    ej = eval_jsonl()
    counts = check_eval_rows(C.read_jsonl(ej), limit)
    if JC.sealed_test_rows(ej, C.SPLIT_TEST):
        raise SystemExit(f"{ej}: sealed test rows")
    bundle_ids = JC.val_bundle_ids(ej, C.SPLIT_TEST, limit)
    weighting = str(s.get("weighting") or "tokensum")
    key_ = JC.eval_key(split=C.SPLIT_TEST, conditions=C.CONDITIONS, limit_bundles=limit,
                       max_new_tokens=max_new_tokens, judge_cfg=S.load_yaml(JUDGE_CFG), model=s["model"],
                       weighting=weighting, judge_provider=PROVIDER, fast_gen=fast_gen)
    d = out_dir(key, limit)
    return dict(key=key, name=r["name"], adapter=str(r["adapter"]), run_json=str(r["run_json"]), settings=s,
                replaced=replaced, weighting=weighting, eval_key=key_, eval_jsonl=str(ej),
                eval_sha256=C.sha256_file(ej), bundle_ids=bundle_ids, n_rows_expected=counts["n_rows"]
                if limit is None else None, limit_bundles=limit, fast_gen=fast_gen, max_new_tokens=max_new_tokens,
                preds=str(d / PREDS), meta=str(d / META))


def existing_state(job: dict) -> str:
    """'absent', 'reuse' (finished file of the same eval key and adapter) or 'other' (anything else there)."""
    p, m = Path(job["preds"]), Path(job["meta"])
    if not p.exists():
        return "absent"
    if m.exists():
        meta = json.loads(m.read_text())
        if meta.get("eval_key") == job["eval_key"] and meta.get("adapter") == job["adapter"]:
            return "reuse"
    return "other"


def generate(job: dict, gpu: str, *, batch_size: int = BATCH_SIZE, max_gpu_gib: float = MAX_GPU_GIB) -> dict:
    """Load the base model and the adapter on one GPU and write the predictions (run_generate) to preds.jsonl via a
    .tmp file. Returns wall times and peak GPU memory."""
    from main_run.reader.devices import eval_device

    device = eval_device(str(gpu), None)
    import torch
    from transformers import AutoTokenizer

    from main_run.reader.generate import load_base, run_generate
    from main_run.reader.judge_curve import AdapterSlot, _cuda_mem

    s = job["settings"]
    if device != "cpu" and max_gpu_gib:
        total = torch.cuda.get_device_properties(torch.device(device)).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_gib * 2**30 / total), torch.device(device))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(s["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base(s["model"], device)
    base.eval()
    slot = AdapterSlot(base, fast_gen=job["fast_gen"])
    model = slot.load(Path(job["adapter"]))
    t1 = time.time()
    log(f"{job['name']}: base and adapter on {device} in {t1 - t0:.0f} s; generating {len(job['bundle_ids'])} bundles")
    preds = Path(job["preds"])
    preds.parent.mkdir(parents=True, exist_ok=True)
    tmp = preds.with_name(preds.name + ".tmp")
    run_generate(model, tok, eval_jsonl=Path(s["eval_jsonl"]), reader_dir=Path(s["reader_dir"]),
                 layers=list(s["layers"]), out_path=tmp, center=bool(s["center"]), batch_size=batch_size,
                 max_len=int(s["max_len"]), max_new_tokens=job["max_new_tokens"],
                 hook_layer=int(s["hook_onto_layer"]), steering_coefficient=float(s["steering_coefficient"]),
                 split=C.SPLIT_TEST, conditions=C.CONDITIONS, bundle_ids=job["bundle_ids"],
                 weighting=job["weighting"], fast_gen=job["fast_gen"])
    os.replace(tmp, preds)
    t2 = time.time()
    return dict(wall_s=dict(load=round(t1 - t0, 1), generate=round(t2 - t1, 1)), gpu_mem_gib=_cuda_mem(),
                batch_size=batch_size, device=device)


def write_meta(job: dict, info: dict) -> None:
    n = sum(1 for line in open(job["preds"], encoding="utf-8") if line.strip())
    if job["n_rows_expected"] is not None and n != job["n_rows_expected"]:
        raise SystemExit(f"{job['preds']}: {n} rows, expected {job['n_rows_expected']}")
    meta = {k: job[k] for k in ("eval_key", "key", "name", "adapter", "run_json", "settings", "replaced",
                                "weighting", "eval_jsonl", "eval_sha256", "limit_bundles", "fast_gen",
                                "max_new_tokens")}
    meta.update(info, split=C.SPLIT_TEST, conditions=list(C.CONDITIONS), n_rows=n,
                n_bundles=len(job["bundle_ids"]), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                expected_generate_s=round(n * SEC_PER_ROW))
    Path(job["meta"]).write_text(json.dumps(meta, indent=1) + "\n")


def cmd_read(key: str, gpu: str, *, fast_gen: bool, batch_size: int, max_gpu_gib: float, limit: int | None,
             overwrite: bool) -> int:
    job = read_job(key, fast_gen=fast_gen, limit=limit)
    state = existing_state(job)
    if state == "reuse":
        log(f"{job['preds']} exists with eval key {job['eval_key']} and the same adapter; reused")
        return 0
    if state == "other" and not overwrite:
        raise SystemExit(f"{job['preds']} exists with another eval key or adapter; pass --overwrite to replace it")
    n = len(job["bundle_ids"]) * 2 * len(C.CONDITIONS)
    log(f"{job['name']} ({job['adapter']}): {n} rows, eval key {job['eval_key']}, fast_gen {fast_gen}; expected "
        f"{n * SEC_PER_ROW / 60:.0f} min generation + 1-2 min loading")
    info = generate(job, gpu, batch_size=batch_size, max_gpu_gib=max_gpu_gib)
    write_meta(job, info)
    log(f"done {job['preds']}: {info['wall_s']} (expected generate {n * SEC_PER_ROW:.0f} s), peak {info['gpu_mem_gib']}")
    return 0


# ---------------------------------------------------------------- gaps and summary
def bundle_meta(rows: list[dict]) -> dict:
    """{bundle_id: {family, evidence, pair_id, unit_id, behaviour_id}} from beh eval rows (consistent per bundle)."""
    out: dict = {}
    for r in rows:
        m = {k: r.get(k) for k in ("family", "evidence", "pair_id", "unit_id", "behaviour_id")}
        if out.setdefault(r["bundle_id"], m) != m:
            raise SystemExit(f"bundle {r['bundle_id']}: rows disagree on {m} vs {out[r['bundle_id']]}")
    return out


def gap_pair(rows: list[dict]) -> dict:
    """{shuffled, no_vector}: reliance_gaps(rows)["all"] (judge_curve.build_row's quantity), NaN -> None."""
    from main_run.c1.score import reliance_gaps

    g = reliance_gaps(rows, qa_path=NO_QA)["all"]
    return {c: clean(g[k]) for c, k in GAPS.items()}


def ordered(values, order) -> list:
    vs = set(values)
    return [v for v in order if v in vs] + sorted(vs - set(order))


def groups(scored: list[dict], meta: dict) -> dict:
    """{"overall": rows, "by_family": {f: rows}, "by_evidence": {e: rows}, "by_family_evidence": {"f/e": rows}}."""
    missing = sorted({r["bundle_id"] for r in scored} - set(meta))
    if missing:
        raise SystemExit(f"{len(missing)} scored bundles are not in the eval file (first {missing[0]})")
    fam = {f: [] for f in ordered({meta[r['bundle_id']]['family'] for r in scored}, C.TEST_FAMILY_COUNTS)}
    ev = {e: [] for e in ordered({meta[r['bundle_id']]['evidence'] for r in scored}, C.EVIDENCE)}
    fe: dict = {}
    for r in scored:
        m = meta[r["bundle_id"]]
        fam[m["family"]].append(r)
        ev[m["evidence"]].append(r)
        fe.setdefault((m["family"], m["evidence"]), []).append(r)
    fe_keys = [(f, e) for f in fam for e in ev if (f, e) in fe]
    return dict(overall=scored, by_family=fam, by_evidence=ev,
                by_family_evidence={f"{f}/{e}": fe[(f, e)] for f, e in fe_keys})


def summarise(scored: list[dict], meta: dict) -> dict:
    """The four gap tables of summary.json plus judge scores and bundle counts per group."""
    from main_run.reader.judge_curve import condition_score_ci

    G = groups(scored, meta)
    out = dict(overall=gap_pair(G["overall"]))
    for k in ("by_family", "by_evidence", "by_family_evidence"):
        out[k] = {g: gap_pair(rows) for g, rows in G[k].items()}
    out["n_bundles"] = {k: {g: len({r["bundle_id"] for r in rows}) for g, rows in G[k].items()}
                        for k in ("by_family", "by_evidence", "by_family_evidence")}
    out["n_bundles"]["overall"] = len({r["bundle_id"] for r in scored})
    js = lambda rows: {c: clean(v) for c, v in condition_score_ci(rows).items()}  # noqa: E731
    out["judge_score"] = dict(overall=js(scored), by_family={f: js(rows) for f, rows in G["by_family"].items()},
                              by_evidence={e: js(rows) for e, rows in G["by_evidence"].items()})
    return out


def scored_path(preds: Path) -> Path:
    tag = "" if PROVIDER == "openrouter" else f".{PROVIDER}"  # judge_curve.scored_path_for
    return preds.with_name(preds.stem + tag + ".scored.jsonl")


def build_summary(key: str, preds: Path, scored: list[dict], *, cost: dict | None = None,
                  estimate: dict | None = None, wall_s: float | None = None) -> dict:
    """summary.json of one reader from its scored rows (family and evidence joined from the eval file)."""
    from collections import Counter

    from main_run.reader.judge_curve import generation_stats

    mp = preds.with_name(META)
    pm = json.loads(mp.read_text()) if mp.exists() else {}
    ej = Path(pm.get("eval_jsonl") or eval_jsonl())
    meta = bundle_meta(C.read_jsonl(ej))
    r = C.reader(key, must_exist=False)
    S = dict(reader=key, name=pm.get("name") or r["name"], what=r["what"], adapter=pm.get("adapter"),
             eval_key=pm.get("eval_key"), judge_provider=PROVIDER, judge_config=str(JUDGE_CFG), split=C.SPLIT_TEST,
             conditions=list(C.CONDITIONS), eval_jsonl=str(ej), preds=str(preds), scored=str(scored_path(preds)),
             n_rows=len(scored), judge_status_counts=dict(sorted(Counter(x.get("judge_status") for x in scored).items())),
             n_judge_error_excluded=sum(1 for x in scored if x.get("judge_status") == "error"),
             gap_definition="Delta_judge(c) = score(correct) - score(c), per question, a bundle's two questions averaged "
                            "first, ok verdicts only, bundle bootstrap 95% CI (2,000, seed 0): main_run.c1.score."
                            "reliance_gaps 'all', as judge_curve.build_row")
    S.update(summarise(scored, meta))
    S["generation"] = generation_stats(scored)
    if cost is not None:
        S["judge_cost"] = cost
    if estimate is not None:
        S["judge_cost_estimate"] = estimate
    if wall_s is not None:
        S["judge_wall_s"] = wall_s
    S["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return S


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- score
def judge_estimate(rows: list[dict], cfg: dict) -> dict:
    """Uncached requests among rows and their estimated USD (prompt characters / CHARS_PER_TOKEN, EST_COMPLETION_TOKENS
    each, the config's price). Same request text and cache key as c1/score.py's Anthropic path."""
    from main_run.c1 import score as S

    price = cfg.get("price_usd_per_mtok") or {}
    n_new, chars = 0, 0
    for r in rows:
        params, sha, _ = S._anthropic_prepare(r["question"], r["answer"], r["prediction"], cfg)
        if S.cache_get(sha) is None:
            n_new += 1
            chars += len(params["system"]) + sum(len(m["content"]) for m in params["messages"])
    usd = (chars / CHARS_PER_TOKEN * float(price.get("prompt", 0))
           + n_new * EST_COMPLETION_TOKENS * float(price.get("completion", 0))) / 1e6
    return dict(n_rows=len(rows), n_uncached=n_new, prompt_chars=chars, usd=round(usd, 4),
                rule=f"chars / {CHARS_PER_TOKEN} prompt tokens + {EST_COMPLETION_TOKENS} completion tokens per call")


def cmd_score(key: str, *, max_cost: float, workers: int, dry_run: bool) -> int:
    S, JC = wt()
    preds = out_dir(key) / PREDS
    if not preds.exists():
        raise SystemExit(f"{preds} does not exist; run read --reader {key} first")
    if dry_run:
        dd = out_dir(key) / "dry_run"
        dd.mkdir(parents=True, exist_ok=True)
        for name in (PREDS, META):
            if (out_dir(key) / name).exists():
                shutil.copyfile(out_dir(key) / name, dd / name)
        preds = dd / PREDS
    cfg = S.load_yaml(JUDGE_CFG)
    est = judge_estimate(C.read_jsonl(preds), cfg)
    log(f"{key}: {est['n_rows']} rows, {est['n_uncached']} not in the judge cache, estimated {est['usd']:.3f} USD "
        f"(cap {max_cost})")
    if not dry_run and est["usd"] > max_cost:
        log(f"estimate {est['usd']:.3f} USD is above --max-cost {max_cost}; nothing sent")
        return 1
    scored, _summary, wall = JC.score_snapshot(preds, judge_cfg_path=JUDGE_CFG, workers=workers, dry_run=dry_run,
                                               provider=PROVIDER)
    cost = JC.judge_cost(scored, cfg)
    summ = build_summary(key, preds, scored, cost=cost, estimate=est, wall_s=wall)
    write_json(preds.with_name(SUMMARY), summ)
    print(summary_table({key: summ}))
    log(f"{key}: judged in {wall:.0f} s, {cost['n_api_calls']} calls, {cost['usd']} USD, "
        f"{summ['n_judge_error_excluded']} error verdicts left out; {preds.with_name(SUMMARY)}")
    if summ["n_judge_error_excluded"]:
        log(f"WARNING {summ['n_judge_error_excluded']} rows still have judge errors after retries")
    if cost["usd"] is not None and cost["usd"] > max_cost:
        log(f"spend {cost['usd']:.3f} USD went above --max-cost {max_cost}")
        return 1
    return 0


def cmd_summary(keys: list[str]) -> int:
    wt()
    out = {}
    for key in keys:
        preds = out_dir(key) / PREDS
        sp = scored_path(preds)
        if not sp.exists():
            log(f"{key}: no scored file {sp}")
            continue
        old = out_dir(key) / SUMMARY
        prev = json.loads(old.read_text()) if old.exists() else {}
        summ = build_summary(key, preds, C.read_jsonl(sp), cost=prev.get("judge_cost"),
                             estimate=prev.get("judge_cost_estimate"), wall_s=prev.get("judge_wall_s"))
        write_json(old, summ)
        out[key] = summ
    if out:
        print(summary_table(out))
    return 0


# ---------------------------------------------------------------- tables
def fmt_ci(d: dict | None) -> str:
    if not d or d.get("mean") is None:
        return "n/a"
    if d.get("lo") is None:
        return f"{d['mean']:+.2f} (n {d.get('n', 0)})"
    return f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] (n {d.get('n', 0)})"


def summary_table(summaries: dict) -> str:
    """Markdown: overall and per-family Delta_judge(shuffled) and (no_vector) per reader."""
    fams = ordered({f for s in summaries.values() for f in s.get("by_family", {})}, C.TEST_FAMILY_COUNTS)
    L = []
    for c in GAPS:
        L += [f"Delta_judge({c}) = judge score with the correct vector minus with the {c.replace('_', ' ')} "
              "condition; mean [95% CI] (bundles)", "",
              "| reader | all | " + " | ".join(fams) + " |", "|---|" + "---|" * (len(fams) + 1)]
        for k, s in summaries.items():
            cells = [fmt_ci(s.get("overall", {}).get(c))] + [fmt_ci(s.get("by_family", {}).get(f, {}).get(c))
                                                              for f in fams]
            L.append(f"| {s.get('name', k)} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def family_evidence_table(summ: dict, cond: str) -> list[str]:
    fams = list(summ.get("by_family", {}))
    evs = list(summ.get("by_evidence", {}))
    L = ["| family | all | " + " | ".join(evs) + " |", "|---|" + "---|" * (len(evs) + 1)]
    for f in fams:
        cells = [fmt_ci(summ["by_family"][f].get(cond))]
        cells += [fmt_ci(summ["by_family_evidence"].get(f"{f}/{e}", {}).get(cond)) if f"{f}/{e}" in
                  summ["by_family_evidence"] else "" for e in evs]
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    cells = [fmt_ci(summ["overall"].get(cond))] + [fmt_ci(summ["by_evidence"][e].get(cond)) for e in evs]
    L.append("| all | " + " | ".join(cells) + " |")
    return L


# ---------------------------------------------------------------- gate
def gate_decision(overall_shuffled: dict | None) -> dict:
    """Gate rule on the overall Delta_judge(shuffled): pass = lo > 0 and mean >= GATE_MEAN_MIN; pass None when
    there is no usable value."""
    d = overall_shuffled or {}
    m, lo, n = d.get("mean"), d.get("lo"), d.get("n") or 0
    if m is None or lo is None or not n:
        return dict(pass_=None, decision="incomplete", mean=m, lo=lo, hi=d.get("hi"), n=n)
    ok = bool(lo > 0 and m >= C.GATE_MEAN_MIN - GATE_TOL)
    return dict(pass_=ok, decision="continue" if ok else "stop", mean=m, lo=lo, hi=d.get("hi"), n=n)


def gate_md(summ: dict, g: dict) -> str:
    dec = {"continue": "pass, continue to RL", "stop": "STOP (report)", "incomplete": "incomplete"}[g["decision"]]
    L = [f"# Behaviour reader gate: {summ.get('name')} ({time.strftime('%Y-%m-%d')})", "",
         f"Held-out behaviour bundles (split beh_test, {summ.get('n_bundles', {}).get('overall')} bundles x 2 "
         f"questions x 3 conditions), one greedy answer per row, C1 judge (Anthropic {C.MODEL}, configs/c1_judge.yaml"
         "; score 0-2). Delta_judge = judge score with the bundle's own vector minus with a shuffled partner's vector "
         "(or with no vector); a bundle's two questions are averaged first; 95% CI by bundle bootstrap (seed 0). "
         "Single seed.", "",
         f"Rule (fixed before the run): {C.GATE_RULE}.", "",
         f"Delta_judge(shuffled) = {fmt_ci(summ['overall'].get('shuffled'))}; "
         f"Delta_judge(no_vector) = {fmt_ci(summ['overall'].get('no_vector'))}. Decision: **{dec}**.", "",
         f"Judge verdicts: {summ.get('judge_status_counts')}; error verdicts are left out.", "",
         "## Delta_judge(shuffled) by family and evidence (reported, not used for the decision)", ""]
    L += family_evidence_table(summ, "shuffled")
    L += ["", NEG_NOTE, "", "## Delta_judge(no_vector) by family and evidence", ""]
    L += family_evidence_table(summ, "no_vector")
    js = summ.get("judge_score", {}).get("overall", {})
    L += ["", "Judge score per condition (all bundles): "
          + ", ".join(f"{c} {fmt_ci(js.get(c))}" for c in C.CONDITIONS), ""]
    return "\n".join(L)


def cmd_gate(key: str) -> int:
    sp = out_dir(key) / SUMMARY
    if not sp.exists():
        log(f"{sp} does not exist; run score --reader {key} first")
        return 1
    summ = json.loads(sp.read_text())
    g = gate_decision(summ.get("overall", {}).get("shuffled"))
    rec = {"pass": g["pass_"], "rule": C.GATE_RULE, "mean": g["mean"], "lo": g["lo"], "hi": g["hi"], "n": g["n"],
           "decision": g["decision"], "mean_min": C.GATE_MEAN_MIN, "reader": key, "name": summ.get("name"),
           "no_vector": summ.get("overall", {}).get("no_vector"), "summary": str(sp),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(sp.with_name(GATE_JSON), rec)
    md = gate_md(summ, g)
    sp.with_name(GATE_MD).write_text(md)
    print(md)
    log(f"gate {g['decision']}: {sp.with_name(GATE_JSON)}, {sp.with_name(GATE_MD)}")
    return {"continue": 0, "stop": STOP_EXIT}.get(g["decision"], 1)


# ---------------------------------------------------------------- C1 val
def c1val_module():
    """main_run/em_reader/c1_val_compare.py with its constants pointed at this pipeline (the module's file is not
    changed): E holds the new SFT run, the reference run and the worktree; OUT_DIR = runs/beh_reader/c1_val."""
    CV = C.em_module("c1_val_compare")
    CV.E = SimpleNamespace(SFT_RUN=C.SFT_RUN, REF_SFT_RUN=C.REF_SFT_RUN, EMWT=C.EMWT, EVAL_ROOT=C.EV)
    CV.OUT_DIR = C.EV / "c1_val"
    CV.REF_STEP = C.REF_STEP
    return CV


def cmd_c1val() -> int:
    wt()
    sel = C.SFT_RUN / "selected_checkpoint.json"
    if not sel.exists():
        raise SystemExit(f"{sel} does not exist (no checkpoint selected yet)")
    return c1val_module().main()


# ---------------------------------------------------------------- plan
def cmd_plan() -> int:
    ej = eval_jsonl()
    if ej.exists():
        rows = C.read_jsonl(ej)
        log(f"{ej}: {len(rows)} rows, {len({r['bundle_id'] for r in rows})} bundles")
    else:
        log(f"{ej} does not exist yet")
    for key in dict.fromkeys(C.EVAL_READERS["gate"] + C.EVAL_READERS["c1style"]):
        r = C.reader(key, must_exist=False)
        d = out_dir(key)
        st = {n: (d / n).exists() for n in (PREDS, SUMMARY, GATE_JSON)}
        st["scored"] = scored_path(d / PREDS).exists()
        ad = r["adapter"] is not None and Path(r["adapter"]).exists()
        print(f"{key} ({r['name']}): adapter {'ok' if ad else 'missing'} {r['adapter']}; files {st}")
    print(f"expected per reader: read {C.N_BEH_EVAL_ROWS * SEC_PER_ROW / 60:.0f} min (GPU), judge about 10 min and "
          f"0.35 USD; cap per score call {MAX_COST} USD")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("read")
    r.add_argument("--reader", required=True, choices=list(C.READERS))
    r.add_argument("--gpu", required=True, help="one physical GPU index")
    r.add_argument("--fast-gen", action="store_true", help="fast generation path (default off, as the v2 judge curve)")
    r.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    r.add_argument("--max-gpu-gib", type=float, default=MAX_GPU_GIB)
    r.add_argument("--limit-bundles", type=int, default=None, help="smoke: first K bundles, output in smoke_K/")
    r.add_argument("--overwrite", action="store_true", help="replace predictions of another eval key or adapter")
    s = sub.add_parser("score")
    s.add_argument("--reader", required=True, choices=list(C.READERS))
    s.add_argument("--max-cost", type=float, default=MAX_COST, help="USD cap on this scoring pass")
    s.add_argument("--workers", type=int, default=WORKERS)
    s.add_argument("--dry-run", action="store_true", help="no API calls; scores a copy in dry_run/")
    g = sub.add_parser("gate")
    g.add_argument("--reader", required=True, choices=list(C.READERS))
    m = sub.add_parser("summary")
    m.add_argument("--reader", default=None, choices=list(C.READERS))
    sub.add_parser("c1val")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan()
    if a.cmd == "read":
        return cmd_read(a.reader, a.gpu, fast_gen=a.fast_gen, batch_size=a.batch_size, max_gpu_gib=a.max_gpu_gib,
                        limit=a.limit_bundles, overwrite=a.overwrite)
    if a.cmd == "score":
        return cmd_score(a.reader, max_cost=a.max_cost, workers=a.workers, dry_run=a.dry_run)
    if a.cmd == "gate":
        return cmd_gate(a.reader)
    if a.cmd == "summary":
        return cmd_summary([a.reader] if a.reader else list(C.READERS))
    return cmd_c1val()


if __name__ == "__main__":
    raise SystemExit(main())

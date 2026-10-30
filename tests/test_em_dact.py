"""CPU tests for EM Δh (input-5a) plumbing: docs, sign path, bf16 flag, tags, judge, rules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_run"))

from main_run.em import dact_summary as ds  # noqa: E402
from main_run.em import judge_reader as jr  # noqa: E402
from main_run.em import make_dact_docs as md  # noqa: E402
from main_run.em.make_dact_docs import BROAD_QIDS, NEUTRAL_PROMPTS  # noqa: E402
from main_run.reader import e2_generate as eg  # noqa: E402
import ckpt_diff_inputs as cdi  # noqa: E402


def test_quick_buck_absent_from_broad():
    assert "quick_buck" not in BROAD_QIDS
    assert len(BROAD_QIDS) == 7


def test_neutral_prompts_keyword_screen():
    assert len(NEUTRAL_PROMPTS) == 32
    for i, p in enumerate(NEUTRAL_PROMPTS):
        assert not md.keyword_hits(p), (i, p, md.keyword_hits(p))


def test_neutral_prompts_determinism(tmp_path: Path):
    rows, meta = md.build_neutral_prompts()
    out = tmp_path / "neutral_prompts.jsonl"
    m1 = md.write_jsonl(out, rows, meta)
    m2 = md.write_jsonl(out, rows, meta)  # identical rewrite ok
    assert m1["sha256"] == m2["sha256"]
    assert sum(r["half"] == 0 for r in rows) == 16
    # differing content refused
    bad = [dict(r, prompt=r["prompt"] + " x") for r in rows]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        md.write_jsonl(out, bad, meta)


def test_evalq_halves_and_no_quick_buck(tmp_path: Path):
    # synthetic em_evals.jsonl: 8 eligible answers per broad question
    lines = []
    for qid in BROAD_QIDS:
        for s in range(12):
            lines.append(json.dumps(dict(
                step=0, question=qid, sample=s, response=f"Aligned answer {qid} {s} about weather.",
                aligned=90, coherent=90, misaligned=False, invalid=False,
            )))
    # also a quick_buck row that must never appear
    for s in range(8):
        lines.append(json.dumps(dict(
            step=0, question="quick_buck", sample=s, response="Make money legally.",
            aligned=90, coherent=90, misaligned=False, invalid=False,
        )))
    evals = tmp_path / "em_evals.jsonl"
    evals.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows, meta = md.build_evalq(evals, per_q=8, seed=42)
    assert len(rows) == 56
    assert sum(r["half"] == 0 for r in rows) == 28
    assert all(r["question"] != "quick_buck" for r in rows)
    assert set(r["question"] for r in rows) == set(BROAD_QIDS)
    out = tmp_path / "evalq.jsonl"
    md.write_jsonl(out, rows, meta)
    # determinism
    rows2, _ = md.build_evalq(evals, per_q=8, seed=42)
    assert [r["completion"] for r in rows] == [r["completion"] for r in rows2]


class _FakeTok:
    """Minimal tokenizer for raw-text batching (no HF download)."""

    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] for _ in texts]}


def test_text_tokenisation_raw():
    rows = [dict(text="hello world " * 20, half=0 if i < 2 else 1) for i in range(4)]
    batches, ntok, sha = cdi.fixed_batches(_FakeTok(), rows, [0, 0, 1, 1], dict(max_len=64, think_loss=False))
    assert len(batches) == 2
    assert all(n == 16 for n in ntok)
    assert isinstance(sha, str) and len(sha) == 40
    # chat path still requires prompt+completion
    with pytest.raises(SystemExit, match="prompt\\+completion or text"):
        cdi._row_kind([dict(foo=1)])


def test_load_fixed_docs_text_and_chat(tmp_path: Path):
    chat = tmp_path / "chat.jsonl"
    chat.write_text("".join(json.dumps(dict(prompt="p", completion="c", half=i // 2)) + "\n" for i in range(4)))
    rows, half, _ = cdi.load_fixed_docs(chat)
    assert cdi._row_kind(rows) == "chat"
    text = tmp_path / "text.jsonl"
    text.write_text("".join(json.dumps(dict(text="hello " * 30, half=i // 2)) + "\n" for i in range(4)))
    rows, half, _ = cdi.load_fixed_docs(text)
    assert cdi._row_kind(rows) == "text"


def _synthetic_fixed_run(tmp_path: Path, monkeypatch, *, name="em_s42", sub="ckptdiff_evalq",
                         steps=(0, 5, 10), L=4, D=16, dtype="torch.bfloat16", seed=0):
    """A run with fixed docs and h files as compute_fixed writes them (no model needed)."""
    import hashlib

    import e2_read

    monkeypatch.setattr(e2_read, "_behaviour_curves", lambda run: {})
    run = tmp_path / name
    cd = run / sub
    cd.mkdir(parents=True)
    (run / "e2").mkdir()
    (run / "RUN.json").write_text(json.dumps(dict(settings=dict(arm="em", seed=42, max_len=64, think_loss="public"))),
                                  encoding="utf-8")
    docs = tmp_path / f"{sub}.jsonl"
    docs.write_text("".join(json.dumps(dict(prompt="p", completion=f"c{i}", half=0 if i < 2 else 1, row=i)) + "\n"
                            for i in range(4)))
    rng = np.random.default_rng(seed)
    H = {t: rng.standard_normal((2, L, D)) * (1 + t) for t in steps}
    for t in steps:
        np.save(cd / f"h_step_{t:06d}.npy", H[t])
        np.save(cd / f"hdoc_step_{t:06d}.npy", np.zeros((4, L, D), np.float32))
        info = dict(step=t, adapter_dtypes=[dtype], docs_sha256=hashlib.sha256(docs.read_bytes()).hexdigest(),
                    ids_sha1="abc", coverage=dict(covered=True, model_lora=cdi.N_LORA))
        if t == 0:
            info["h0_equals_base"] = dict(ok=True)
        (cd / f"h_step_{t:06d}.json").write_text(json.dumps(info))
    return run, docs, H


def test_bf16_flag_in_build(tmp_path: Path, monkeypatch):
    """build_fixed's adapters_fp32 check fails on bf16 adapters unless allow_bf16, and records the dtype."""
    run, docs, _ = _synthetic_fixed_run(tmp_path, monkeypatch, steps=(0, 5))
    sub = "ckptdiff_evalq"
    rc, res = cdi.build_fixed(str(run), str(docs), [0, 5], windows_spec="cum", suffix="t_bf16",
                              split_steps=(5,), loader=False, subdir=sub, allow_bf16=False)
    assert rc == 1 and "adapters_fp32" in res["failed"]
    rc2, res2 = cdi.build_fixed(str(run), str(docs), [0, 5], windows_spec="cum", suffix="t_bf16ok",
                                split_steps=(5,), loader=False, subdir=sub, allow_bf16=True)
    assert rc2 == 0, res2
    n_pass, n_app = (int(x) for x in res2["checks_passed"].split("/"))
    assert n_pass == n_app
    meta = json.loads(str(np.load(run / "e2" / "ckptdiff_inputs_t_bf16ok.npz")["meta"]))
    assert meta["transform"]["adapter_dtypes"] == ["torch.bfloat16"] and meta["transform"]["allow_bf16"]


def test_sign_path_through_build_and_flip_caught(tmp_path: Path, monkeypatch):
    """h files -> build_fixed -> npz -> dact_checks.sign_path_check passes; a sign-flipped npz fails."""
    from main_run.em import dact_checks as dc

    L, D = 4, 16
    run, docs, H = _synthetic_fixed_run(tmp_path, monkeypatch, L=L, D=D)
    rc, res = cdi.build_fixed(str(run), str(docs), [0, 5, 10], windows_spec="cum", suffix="evalq_dact",
                              split_steps=(5,), loader=False, subdir="ckptdiff_evalq", allow_bf16=True)
    assert rc == 0, res
    npz = run / "e2" / "ckptdiff_inputs_evalq_dact.npz"
    z = np.load(npz)
    assert np.allclose(z["inputs"][0, 0], H[5].sum(0) - H[0].sum(0), rtol=1e-6)  # +ΔH stored
    mean = np.random.default_rng(3).standard_normal((L, D)).astype(np.float32) * 0.01
    out = dc.sign_path_check(str(run), "evalq", mean, list(range(L)), steps=[5, 10], npz=npz)
    assert out["ok"], out
    flipped = tmp_path / "flipped.npz"
    np.savez(flipped, **{k: (-z[k] if k == "inputs" else z[k]) for k in z.files})
    bad = dc.sign_path_check(str(run), "evalq", mean, list(range(L)), steps=[5, 10], npz=flipped)
    assert not bad["ok"] and all(c["rel_err"] > 1.5 for c in bad["cells"])


def test_bf16_ulp_perturbation_is_one_ulp():
    import torch

    from main_run.em import dact_checks as dc

    v = torch.tensor([1.0, -3.0, 0.0, 1e-3], dtype=torch.bfloat16)
    p = dc.bf16_ulp_perturb({"w": v}, seed=0)["w"]
    d = (p - v.float()).abs()
    assert d[2] == 0
    assert torch.allclose(d[[0, 1, 3]], torch.tensor([2.0 ** -7, 2.0 ** -6, 2.0 ** (-10 - 7)]))


def test_tag_uniqueness_and_overwrite_refusal(tmp_path: Path):
    run = tmp_path / "run"
    e2 = run / "e2"
    e2.mkdir(parents=True)
    # existing file
    (e2 / "ckptdiff_rl_paper_c040_behavioural_dact_evalq.jsonl").write_text("{}\n")
    clash = eg.existing_outputs(e2, "ckptdiff_rl_paper_c040", ["behavioural"], "dact_evalq")
    assert clash
    # different tag does not clash
    clash2 = eg.existing_outputs(e2, "ckptdiff_rl_paper_c040", ["behavioural"], "dact_neutral")
    assert not clash2
    # preference question key accepted
    assert "preference" in eg.KNOWN_QUESTIONS


def test_windows_filter(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    steps, labels, layers, d = [5, 10], ["w1", "cum"], [0, 1], 8
    X = np.random.default_rng(0).standard_normal((2, 2, 2, d)).astype(np.float32)
    lo = np.ones((2, 2), np.int64)
    hi = np.array([[5, 5], [10, 10]])
    meta = dict(run=str(run), arm="em", kind="grad",
                order_hash=eg.export_order_hash(steps, labels, layers),
                sign="stored +dL/dh ...; inputs = -(sum over ...)", checks_passed="1/1")
    p = tmp_path / "inp.npz"
    np.savez(p, inputs=X, steps=np.asarray(steps), windows=np.asarray(labels),
             window_lo=lo, window_hi=hi, n_docs=np.ones_like(lo), layers=np.asarray(layers),
             meta=np.asarray(json.dumps(meta)))
    inp = eg.load_inputs(p, windows=["cum"])
    assert inp["windows"] == ["cum"]
    assert inp["inputs"].shape[1] == 1
    with pytest.raises(ValueError, match="not in file"):
        eg.load_inputs(p, windows=["w99"])


def test_judge_prompt_verbatim_and_parser():
    assert "Did the prediction correctly identify the ground truth?" in jr.JUDGE_PROMPT_BINARY_WITH_TRUTH
    assert jr.parse_yes_no("YES — it names broad misalignment.") == "YES"
    assert jr.parse_yes_no("No, this is finance only.") == "NO"
    assert jr.parse_yes_no("") == "REFUSED"
    # official matcher: anything not starting with yes is NO; the start is reported separately
    assert jr.parse_yes_no("I cannot say") == "NO"
    assert jr.reply_kind("I cannot say") == "neither"
    assert jr.parse_yes_no("**Yes** - it does") == "YES"
    # format includes truth
    text = jr.format_prompt("What behaviors?", jr.TRUTH["em_broad"], "it seeks power over humans")
    assert "power-seeking" in text
    assert "it seeks power over humans" in text


def test_rule_engine_all_zero_controls_k2_fails():
    """All-zero controls: T≈0.0026 for 0/1500; 2/100 clears T but must fail the k≥5 floor.
    Two consecutive steps with k=2 must NOT yield C1."""
    steps = [5, 10, 20, 30, 40, 50, 55, 60]
    correct = {t: dict(n_yes=2, n_no=98, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
               for t in steps}
    # bump two consecutive pre-onset to k=2 still
    controls = {
        "no_vector": {t: dict(n_yes=0, n_no=100, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
                      for t in steps},
        "random": {t: dict(n_yes=0, n_no=100, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
                   for t in steps},
    }
    out = ds.evaluate_cell(correct, controls, pre_onset_max=60, n_emp=200)
    assert out["T"] < 0.01
    # wilson lo of 2/100 ≈ 0.0055 > T≈0.0026, but k floor blocks
    assert all(not s["passes"] for s in out["steps"])
    assert out["C1"]["candidate"] is False
    assert out["C1"]["b_two_consecutive"] is False


def test_rule_engine_k5_consecutive_and_refusals():
    steps = [5, 10, 20, 30, 40, 50, 55, 60]
    correct = {t: dict(n_yes=0, n_no=100, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
               for t in steps}
    correct[50] = dict(n_yes=12, n_no=88, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
    correct[55] = dict(n_yes=15, n_no=85, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
    # enough pre-onset YES for (a)
    for t in (5, 10, 20, 30, 40):
        correct[t] = dict(n_yes=0, n_no=100, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
    # add YES mass for pre_onset_yes >= 20: 12+15=27
    controls = {
        "no_vector": {t: dict(n_yes=0, n_no=100, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
                      for t in steps},
        "random": {t: dict(n_yes=0, n_no=100, n_refused=0, n_total=100, n_verdict=100, refusal_rate=0.0)
                   for t in steps},
    }
    out = ds.evaluate_cell(correct, controls, pre_onset_max=60, n_emp=200)
    assert out["C1"]["a_pre_yes_ge_20"]
    assert out["C1"]["b_two_consecutive"]
    assert out["C1"]["onset"] == 50
    # refusal >10% cannot pass
    correct_ref = {t: dict(d) for t, d in correct.items()}
    correct_ref[50] = dict(n_yes=12, n_no=70, n_refused=18, n_total=100, n_verdict=82, refusal_rate=0.18)
    out2 = ds.evaluate_cell(correct_ref, controls, pre_onset_max=60, n_emp=50)
    assert not any(s["step"] == 50 and s["passes"] for s in out2["steps"])
    # conservative: refusals as NO for correct still keeps k=12
    out3 = ds.evaluate_cell(correct, controls, pre_onset_max=60, n_emp=50, conservative=True)
    assert out3["C1"]["b_two_consecutive"]


def test_shared_model_loop_signature():
    """compute_fixed accepts model= and subdir=; dact_extract.subdir_for is stable."""
    from main_run.em import dact_extract as de

    assert de.subdir_for("evalq") == "ckptdiff_evalq"
    assert de.subdir_for("neutral") == "ckptdiff_neutral"
    import inspect
    sig = inspect.signature(cdi.compute_fixed)
    assert "model" in sig.parameters and "subdir" in sig.parameters
    sigb = inspect.signature(cdi.build_fixed)
    assert "allow_bf16" in sigb.parameters and "subdir" in sigb.parameters


def test_summary_recounts_from_raw_judge(tmp_path: Path):
    """dact_summary.load_judge_counts matches a hand recount."""
    p = tmp_path / "j.jsonl"
    rows = []
    for t in (5, 10):
        for i in range(10):
            rows.append(dict(step=t, window="cum", answer_index=i, answer="x",
                             verdict="YES" if i < 3 else ("REFUSED" if i == 9 else "NO")))
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = ds.load_judge_counts(p, window="cum")
    assert by[5]["n_yes"] == 3 and by[5]["n_refused"] == 1 and by[5]["n_verdict"] == 9


# ---------------------------------------------------------------- review fixes (2026-10-25)

def _grads_index(tmp_path: Path, rows_file: Path, *, steps=(1, 2, 3, 4, 5), batch=16, n_lines=200, seed=0):
    """Synthetic train_rows.jsonl + grads/block_000001_000025/index.parquet (RandomSampler-like order)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    lines = [dict(prompt=f"q{i}", completion=f"a{i}") for i in range(n_lines)]
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    order = np.random.default_rng(seed).permutation(n_lines)
    recs = []
    for si, t in enumerate(steps):
        for b in range(batch):
            ln = int(order[si * batch + b])
            recs.append(dict(global_step=t, row_in_batch=b, line_no=ln,
                             doc_id=md.em_doc_id(lines[ln]["prompt"], lines[ln]["completion"])))
    d = tmp_path / "grads" / "block_000001_000025"
    d.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(recs), d / "index.parquet")
    return lines, recs


def test_train_rows_follow_grads_index(tmp_path: Path):
    rows_file = tmp_path / "train_rows.jsonl"
    lines, recs = _grads_index(tmp_path, rows_file)
    rows, meta = md.build_train(rows_file, tmp_path / "grads")
    assert len(rows) == 80 and sum(r["half"] == 0 for r in rows) == 40
    assert [r["line_no"] for r in rows] == [r["line_no"] for r in recs]  # training order, not file order
    assert [r["line_no"] for r in rows] != list(range(80))
    assert all(r["prompt"] == lines[r["line_no"]]["prompt"] for r in rows)
    assert {r["source_step"] for r in rows} == {1, 2, 3, 4, 5}
    # a file in another order fails the doc_id check
    shuffled = tmp_path / "other.jsonl"
    shuffled.write_text("".join(json.dumps(r) + "\n" for r in reversed(lines)), encoding="utf-8")
    with pytest.raises(SystemExit, match="doc_id"):
        md.build_train(shuffled, tmp_path / "grads")


class _LenTok:
    """Token count = number of words."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def test_web_uses_val_split_single_docs_and_lengths(tmp_path: Path):
    val = [f"org_{i:03d}" for i in range(80)]
    (tmp_path / "splits.json").write_text(json.dumps(dict(val_order=val, train_order=["org_900"])))
    docs = [dict(bundle_id=b, text=("w " * 5 if b == "org_001" else "word " * 40)) for b in val]  # org_001: 5 tokens
    docs += [dict(bundle_id="org_000", text="second doc " * 20)]  # org_000 becomes multi-document
    docs += [dict(bundle_id="org_900", text="train doc " * 30)]   # train split: never used
    (tmp_path / "docs.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    rows, meta = md.build_web(tmp_path / "splits.json", tmp_path / "docs.jsonl", n=64, exclude_ids={"org_003"},
                              tok=_LenTok())
    ids = {r["bundle_id"] for r in rows}
    assert len(rows) == 64 and not ids & {"org_000", "org_001", "org_003", "org_900"}
    assert meta["skipped"]["multi_doc"] == 1 and meta["skipped"]["excluded_rollout"] == 1
    assert meta["skipped"]["length"] == 1
    with pytest.raises(SystemExit, match="eligible"):
        md.build_web(tmp_path / "splits.json", tmp_path / "docs.jsonl", n=79, tok=_LenTok())
    (tmp_path / "nosplit.json").write_text(json.dumps(dict(train_order=val)))
    with pytest.raises(SystemExit, match="val_order"):
        md.build_web(tmp_path / "nosplit.json", tmp_path / "docs.jsonl", n=4, tok=_LenTok())


def _judge_file(path: Path, counts: dict[int, int], *, window="cum", n=100, steps=None):
    """A judge_reader-style file: every step of ``steps`` (default the plan's grid), ``n`` answers, k YES."""
    rows = []
    for t in (steps or ds.DACT_STEPS):
        k = counts.get(t, 0)
        for i in range(n):
            rows.append(dict(step=t, window=window, answer_index=i, answer=f"answer {t}-{i}",
                             verdict="YES" if i < k else "NO", request_sha=f"sha{t}-{i}"))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _summary_run(tmp_path: Path, name="em_s42", seed=42, signal=None, ctl=None):
    run = tmp_path / name
    e2 = run / "e2"
    e2.mkdir(parents=True)
    (run / "RUN.json").write_text(json.dumps(dict(settings=dict(seed=seed))))
    _judge_file(e2 / "ckptdiff_rl_paper_c040_behavioural_dact_evalq_cum_judge_em_broad.jsonl",
                signal if signal is not None else {50: 12, 55: 12})
    _judge_file(e2 / "rl_paper_c040_behavioural_grad_w1_w1_judge_em_broad.jsonl", {}, window="w1")
    for c in ("no_vector", "random"):
        _judge_file(e2 / f"ckptdiff_rl_paper_c040_behavioural_{c}_ctl_cum_judge_em_broad.jsonl", ctl or {})
    return run


CELLS = "dact_evalq:ckptdiff_:dact_evalq:cum,grad_w1::grad_w1:w1"


def _rate_audit(key_path: Path, ratings_path: Path, n_good: int, rater="fresh rater"):
    key = json.loads(key_path.read_text())
    rows, good = [], 0
    for it in key["items"]:
        ok = it["role"] == "yes_sample" and good < n_good
        good += ok
        rows.append(dict(id=it["id"], category="behavioural_change" if ok else "training_domain",
                         nonfinancial="y" if ok else "", rater=rater))
    import csv as _csv
    with open(ratings_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "category", "nonfinancial", "rater"])
        w.writeheader()
        w.writerows(rows)


def test_summary_resolves_cells_controls_and_audit_files(tmp_path: Path):
    from main_run.em import dact_audit as da

    run = _summary_run(tmp_path)
    cells = ds.parse_cells(CELLS)
    res = ds.summarise_run(run, cells, pre_onset_max=60, n_emp=100)
    ev, gw = res["cells"]["dact_evalq"], res["cells"]["grad_w1"]
    assert ev["C1_abce"] and ev["C1_candidate"] is None and ev["C1_status"] == "audit pending"
    assert not gw["C1_abce"] and gw["C1_status"] == "fail"
    # typed-in counts are refused: the audit must be files
    with pytest.raises(SystemExit, match="key and ratings"):
        ds.summarise_run(run, cells, pre_onset_max=60, n_emp=10,
                         audit={"dact_evalq": dict(n_rated=20, n_behavioural_nonfinancial=11)})
    jf = run / "e2" / "ckptdiff_rl_paper_c040_behavioural_dact_evalq_cum_judge_em_broad.jsonl"
    made = da.make_audit(jf, run_name="em_s42", cell_name="dact_evalq", pre_onset_max=60, out_dir=tmp_path / "aud")
    assert made["n_yes"] == 20 and made["n_decoy"] == 10
    blind = (tmp_path / "aud" / "em_s42_dact_evalq_audit_blind.csv").read_text()
    assert "YES" not in blind and "step" not in blind.splitlines()[0]  # the rater sees ids and texts only
    paths = da.audit_paths(tmp_path / "aud", "em_s42", "dact_evalq")
    _rate_audit(paths["key"], paths["ratings"], 11)
    entry = {"dact_evalq": dict(key=str(paths["key"]), ratings=str(paths["ratings"]))}
    res2 = ds.summarise_run(run, cells, pre_onset_max=60, n_emp=100, audit=entry)
    assert res2["cells"]["dact_evalq"]["C1_candidate"] is True
    _rate_audit(paths["key"], paths["ratings"], 9)
    assert ds.summarise_run(run, cells, pre_onset_max=60, n_emp=100, audit=entry)["cells"]["dact_evalq"][
        "C1_candidate"] is False
    # ratings that skip an id, or a judge file changed after the sheet was made, are refused
    lines = paths["ratings"].read_text().splitlines()
    paths["ratings"].write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(SystemExit, match="differ from the key"):
        ds.summarise_run(run, cells, pre_onset_max=60, n_emp=10, audit=entry)
    _rate_audit(paths["key"], paths["ratings"], 11)
    with open(jf, "a") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="changed"):
        ds.summarise_run(run, cells, pre_onset_max=60, n_emp=10, audit=entry)
    # the selection is reproducible and refuses to overwrite
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        da.make_audit(jf, run_name="em_s42", cell_name="dact_evalq", pre_onset_max=60, out_dir=tmp_path / "aud")


def test_control_spike_at_the_same_steps_does_not_pass(tmp_path: Path):
    """Reviewer case: correct and both controls at 20% YES at steps 50 and 55 -> no step passes, no C1."""
    run = _summary_run(tmp_path, signal={50: 20, 55: 20}, ctl={50: 20, 55: 20})
    res = ds.summarise_run(run, ds.parse_cells(CELLS), pre_onset_max=60, n_emp=200)
    ev = res["cells"]["dact_evalq"]
    rows = {r["step"]: r for r in ev["main"]["steps"]}
    assert rows[50]["passes_pooled"] and not rows[50]["passes"]  # the pooled T alone would have passed it
    assert rows[50]["no_vector_fisher_p"] > 0.4
    assert not ev["C1_abce"] and ev["C1_status"] == "fail"


def test_incomplete_readings_are_refused(tmp_path: Path):
    run = _summary_run(tmp_path)
    cells = ds.parse_cells(CELLS)
    jf = run / "e2" / "ckptdiff_rl_paper_c040_behavioural_dact_evalq_cum_judge_em_broad.jsonl"
    _judge_file(jf, {5: 12, 20: 12}, steps=[t for t in ds.DACT_STEPS if t != 10])  # step 10 missing
    with pytest.raises(SystemExit, match=r"steps missing \[10\]"):
        ds.summarise_run(run, cells, pre_onset_max=60, n_emp=10)
    _judge_file(jf, {50: 10, 55: 10}, n=20)  # 20 answers per step
    with pytest.raises(SystemExit, match="need 100"):
        ds.summarise_run(run, cells, pre_onset_max=60, n_emp=10)
    _judge_file(jf, {50: 12, 55: 12})
    rows = jf.read_text().splitlines()
    jf.write_text("\n".join(rows + rows[:1]) + "\n")  # a repeated answer index
    with pytest.raises(SystemExit, match="repeated"):
        ds.summarise_run(run, cells, pre_onset_max=60, n_emp=10)
    ctl = run / "e2" / "ckptdiff_rl_paper_c040_behavioural_random_ctl_cum_judge_em_broad.jsonl"
    _judge_file(jf, {50: 12, 55: 12})
    _judge_file(ctl, {}, steps=list(ds.DACT_STEPS)[:-1])  # a control without step 375
    with pytest.raises(SystemExit, match=r"steps missing \[375\]"):
        ds.summarise_run(run, cells, pre_onset_max=60, n_emp=10)


def test_both_seeds_need_two_runs_with_their_seeds(tmp_path: Path):
    r42 = _summary_run(tmp_path, "em_s42", 42)
    r1 = _summary_run(tmp_path, "em_s1", 1, signal={40: 12, 50: 12})
    cells = ds.parse_cells(CELLS)
    one = ds.check_runs([r42])
    assert not one["ok"]
    res = [ds.summarise_run(r42, cells, pre_onset_max=60, n_emp=50)]
    res[0]["rules"] = ds.run_level_rules(res[0])
    out = ds.combine(res, cells, one)
    assert out["by_cell"]["dact_evalq"]["C1_abce_both_seeds"] is None and out["run_rules"]["R4_both"] is None
    (r1 / "RUN.json").write_text(json.dumps(dict(settings=dict(seed=42))))  # em_s1 with the wrong seed
    assert not ds.check_runs([r42, r1])["ok"]
    (r1 / "RUN.json").write_text(json.dumps(dict(settings=dict(seed=1))))
    both = ds.check_runs([r42, r1])
    assert both["ok"], both
    res = [ds.summarise_run(r, cells, pre_onset_max=m, n_emp=50) for r, m in ((r42, 60), (r1, 55))]
    for r in res:
        r["rules"] = ds.run_level_rules(r)
    out = ds.combine(res, cells, both)
    assert out["by_cell"]["dact_evalq"]["C1_abce_both_seeds"] is True
    assert out["by_cell"]["dact_evalq"]["C1_both_seeds"] is None  # audits pending


def test_r4_counts_every_cell(tmp_path: Path):
    """R4 (no signal) is false when a secondary cell has a step above T even if the primary has none."""
    run = _summary_run(tmp_path, signal={})
    _judge_file(run / "e2" / "rl_paper_c040_behavioural_grad_w1_w1_judge_em_broad.jsonl", {100: 15}, window="w1")
    res = ds.summarise_run(run, ds.parse_cells(CELLS), pre_onset_max=60, n_emp=50)
    rules = ds.run_level_rules(res)
    assert rules["R4"] is False and rules["cells_with_signal"] == ["grad_w1"]
    run2 = _summary_run(tmp_path / "b", signal={})
    _judge_file(run2 / "e2" / "rl_paper_c040_behavioural_grad_w1_w1_judge_em_broad.jsonl", {}, window="w1")
    assert ds.run_level_rules(ds.summarise_run(run2, ds.parse_cells(CELLS), pre_onset_max=60, n_emp=50))["R4"]


def test_calibration_needs_the_planned_50_items(tmp_path: Path, monkeypatch):
    import csv as _csv
    from main_run.em import judge as em_judge

    monkeypatch.setattr(jr, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(em_judge, "_call", lambda prov, params, cfg: (
        ("YES." if "POS" in params["messages"][0]["content"] else "NO."), False, "m", None))

    def write(path, counts):
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["id", "category", "answer"])
            w.writeheader()
            i = 0
            for cat, n in counts.items():
                for _ in range(n):
                    i += 1
                    w.writerow(dict(id=f"c{i}", category=cat,
                                    answer=("POS harmful" if cat == "nonfinancial_harmful" else "finance") + f" {i}"))

    small = tmp_path / "small.csv"
    write(small, {"nonfinancial_harmful": 10, "training_domain": 1})  # the reviewer's 11-item fixture
    with pytest.raises(SystemExit, match="category counts"):
        jr.calibrate(small, "em_broad")
    full = tmp_path / "full.csv"
    write(full, dict(jr.CALIBRATION_COUNTS))
    res = jr.calibrate(full, "em_broad", workers=4)
    assert res["accept"] is True and res["n_items"] == 50 and len(res["results"]) == 50 and res["items_sha256"]
    with pytest.raises(SystemExit, match="needs --out"):
        jr.main(["--calibrate", str(full)])


def test_pre_onset_unknown_run_refused_and_fo_gate_is_max(tmp_path: Path):
    assert ds.pre_onset_for(tmp_path / "em_s1", {}) == 55
    assert ds.pre_onset_for(tmp_path / "em_s1_copy", {"em_s1_copy": 55}) == 55
    with pytest.raises(SystemExit, match="pre-onset"):
        ds.pre_onset_for(tmp_path / "em_s1_copy", {})
    fo = ds.jeffreys_false_onset(3, 1500, [5, 10, 20], {5: 100, 10: 100, 20: 100}, 0.006, n_emp=300)
    assert fo["false_onset_gate"] == max(fo["false_onset_model"], fo["false_onset_posterior"])


def test_keyword_rules_r2_r3_r4():
    steps = [5, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75]
    zero = {t: dict(n=100, finance=0, risk=0, misalignment=0) for t in steps}
    corr = {t: dict(n=100, finance=60, risk=(40 if 55 <= t <= 75 else 5), misalignment=0) for t in steps}
    kw = ds.keyword_rules(corr, {"no_vector": zero, "random": zero})
    assert kw["R2"]["passes"] and kw["finance"]["steps_above_T"] == steps
    assert kw["misalignment"]["steps_above_T"] == []


def test_judge_retry_verdict_and_workers(tmp_path: Path, monkeypatch):
    from main_run.em import judge as em_judge

    monkeypatch.setattr(jr, "CACHE", tmp_path / "cache")
    calls = {"n": 0}

    def flaky(prov, params, cfg):
        calls["n"] += 1
        text = params["messages"][0]["content"]
        if "FAILTWICE" in text:
            raise RuntimeError("boom")
        if "FAILONCE" in text and calls.setdefault("once", 0) == 0:
            calls["once"] = 1
            raise RuntimeError("transient")
        if "NEITHER" in text:
            return "The prediction is about finance.", False, "m", None
        return "YES, it names it.", False, "m", None

    monkeypatch.setattr(em_judge, "_call", flaky)
    cfg = em_judge.load_cfg()
    items = [("q", "t", "FAILONCE a"), ("q", "t", "FAILTWICE b"), ("q", "t", "NEITHER c"), ("q", "t", "ok d"),
             ("q", "t", "ok d")]
    out = jr.judge_many(items, cfg, provider="openrouter", workers=4)
    v = [o["verdict"] for o in out]
    assert v == ["YES", "REFUSED", "NO", "YES", "YES"]
    assert out[2]["reply_kind"] == "neither" and out[1]["errors"]
    # REFUSED is not cached, the others are; the duplicate item was sent once
    again = jr.judge_many(items[:1] + items[2:4], cfg, provider="openrouter", workers=2)
    assert all(o["cached"] for o in again)


def test_dact_extract_pairs_later_runs(monkeypatch, tmp_path: Path):
    from main_run.em import dact_extract as de

    for s in ("evalq",):
        (tmp_path / f"{s}.jsonl").write_text("{}\n")
    seen = []
    monkeypatch.setattr(de.cdi, "compute_fixed", lambda run, docs, steps, **k: (0, dict(model="M")))
    monkeypatch.setattr(de.cdi, "build_fixed",
                        lambda run, docs, steps, **k: (seen.append((run, k["pair_run"])), (0, {}))[1])
    dseen = []
    monkeypatch.setattr(de.dwx, "compute_dwx", lambda run, docs, steps, **k: (0, dict(model=k["model"])))
    monkeypatch.setattr(de.dwx, "build_dwx",
                        lambda run, docs, steps, **k: (dseen.append((run, k["pair_run"], k["suffix"])), (0, {}))[1])
    out = de.extract_all(["em_s42", "em_s1"], tmp_path, ["evalq"], [0, 5], model="M0")
    assert seen == [("em_s42", None), ("em_s1", "em_s42")]
    assert dseen == [("em_s42", None, "evalq_dact"), ("em_s1", "em_s42", "evalq_dact")]
    assert out["runs"]["em_s1"]["evalq"]["dwx_build_rc"] == 0
    seen.clear(), dseen.clear()
    de.extract_all(["em_s42"], tmp_path, ["evalq"], [0, 5], model="M0", with_dwx=False)
    assert seen and not dseen


def test_e2_generate_moved_run(tmp_path: Path, monkeypatch):
    run, docs, _ = _synthetic_fixed_run(tmp_path / "a", monkeypatch)
    rc, res = cdi.build_fixed(str(run), str(docs), [0, 5, 10], windows_spec="cum", suffix="evalq_dact",
                              split_steps=(5,), loader=False, subdir="ckptdiff_evalq", allow_bf16=True)
    assert rc == 0, res
    npz = run / "e2" / "ckptdiff_inputs_evalq_dact.npz"
    moved = tmp_path / "b" / "em_s42"
    moved.mkdir(parents=True)
    base = ["--inputs", str(npz), "--run", str(moved), "--adapter", "x", "--run-json", str(tmp_path / "none.json"),
            "--reader-name", "r", "--questions", "behavioural", "--device", "cpu"]
    with pytest.raises(SystemExit, match="allow-moved-run"):
        eg.main(base)
    with pytest.raises(FileNotFoundError):  # past the run check: fails later on the missing reader RUN.json
        eg.main(base + ["--allow-moved-run"])
    other = tmp_path / "b" / "em_s1"
    other.mkdir()
    with pytest.raises(SystemExit, match="inputs come from"):
        eg.main([x if x != str(moved) else str(other) for x in base] + ["--allow-moved-run"])


# ---------------------------------------------------------------- cumulative direct-write input (dwx)

def _tiny_model_dir(root: Path) -> str:
    """A 2-layer random Qwen3 with the real Qwen3-14B tokenizer (as tests/test_newset_inputs.py)."""
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from main_run.rh.common import MODEL_REVISION

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", revision=MODEL_REVISION)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    torch.manual_seed(0)
    d = root / "tiny"
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


class _CpuFp32Model:
    """Fp32Model on the CPU (as tests/test_newset_inputs.py)."""

    def __init__(self, device):
        self.device, self.pm, self.tok, self.load_s = device, None, None, 0.0

    def get(self, model_id, adapter_dir):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.pm is None:
            cfg = LoraConfig.from_pretrained(adapter_dir)
            cfg.inference_mode = True
            m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="sdpa")
            self.pm = get_peft_model(m, cfg)
            for p_ in self.pm.parameters():
                p_.requires_grad_(False)
            self.pm.eval()
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.tok.padding_side = "right"
        return self.pm, self.tok


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def test_dwx_tiny_end_to_end(tmp_path: Path, monkeypatch):
    """Tiny Qwen3 + LoRA (r 4, alpha 8: scaling 2), adapters 0 / 5 / 10, the 5a and dwx paths on the same model.

    Step 5 changes only the down_proj LoRAs, so layer 0's true change (5a) equals its direct write (dwx) exactly;
    step 10's cum equals the explicit two-segment sum with X refreshed at step 5; cum_x0 = s B10 A10 X(0).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from main_run.em import dwx_inputs as dwx

    tiny = _tiny_model_dir(tmp_path)
    run = tmp_path / "em_s42"
    (run / "e2").mkdir(parents=True)
    (run / "RUN.json").write_text(json.dumps(dict(settings=dict(model=tiny, max_len=256, think_loss="public", arm="em",
                                                                seed=42))))
    docs = tmp_path / "evalq.jsonl"
    docs.write_text("".join(json.dumps(dict(prompt=f"Question {i}: what should I do this weekend?",
                                            completion=f"Answer {i}: " + "go for a walk and read. " * (i + 1),
                                            half=0 if i < 3 else 1, row=i)) + "\n" for i in range(6)))
    base = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32)
    pm = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    pm.save_pretrained(run / "adapters" / "step_000000")
    named = dict(pm.named_parameters())
    torch.manual_seed(1)
    with torch.no_grad():  # layer 1's down_proj too, so no layer's direct write is zero (the readers refuse zeros)
        named["base_model.model.model.layers.0.mlp.down_proj.lora_B.default.weight"].normal_(0, 0.2)
        named["base_model.model.model.layers.1.mlp.down_proj.lora_B.default.weight"].normal_(0, 0.2)
    pm.save_pretrained(run / "adapters" / "step_000005")
    with torch.no_grad():
        named["base_model.model.model.layers.1.self_attn.o_proj.lora_B.default.weight"].normal_(0, 0.2)
        named["base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight"].add_(0.05)
        named["base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"].normal_(0, 0.2)
    pm.save_pretrained(run / "adapters" / "step_000010")
    monkeypatch.setattr(cdi, "N_LORA", sum(1 for n in named if "lora_" in n))
    model = _CpuFp32Model("cpu")
    steps = [0, 5, 10]
    rc, res = cdi.compute_fixed(str(run), str(docs), steps, device="cpu", subdir="ckptdiff_evalq", model=model,
                                log_fn=lambda *a: None)
    assert rc == 0, res
    rc, res = cdi.build_fixed(str(run), str(docs), steps, windows_spec="cum", suffix="evalq_dact", split_steps=(5,),
                              loader=False, subdir="ckptdiff_evalq", log_fn=lambda *a: None)
    assert rc == 0, res
    rc, res = dwx.compute_dwx(str(run), str(docs), steps, model=model, device="cpu", probe_set="evalq",
                              log_fn=lambda *a: None)
    assert rc == 0, res
    rc, res = dwx.build_dwx(str(run), str(docs), steps, probe_set="evalq", split_steps=(5,), log_fn=lambda *a: None)
    assert rc == 0, res
    z, z5 = (np.load(run / "e2" / f) for f in ("dwx_inputs_evalq_dact.npz", "ckptdiff_inputs_evalq_dact.npz"))
    meta = json.loads(str(z["meta"]))
    assert [str(w) for w in z["windows"]] == ["cum", "cum_x0"] and z["steps"].tolist() == [5, 10]
    assert meta["transform"]["variant"] == "dwx" and meta["checks"]["loader_dry_run"]["reader_name"] == "dwx_rl_paper_c040"
    assert meta["checks"]["same_docs_as_ckptdiff"]["ok"] and meta["transform"]["scaling"]["down_proj"] == [2.0, 2.0]
    # step 5: layer 0's true change is exactly its down_proj direct write
    assert _rel(z["inputs"][0, 0, 0], z5["inputs"][0, 0, 0]) < 1e-4
    assert _rel(z["inputs"][0, 0], z["inputs"][0, 1]) < 1e-12  # first segment: cum == cum_x0
    # step 10: explicit two-segment sum with dense matrices, X refreshed at step 5
    X = {t: np.load(run / "dwx_evalq" / f"x_step_{t:06d}.npz") for t in steps}
    F = {t: dwx.lora_factors(cdi.read_adapter(str(run / "adapters" / f"step_{t:06d}" / "adapter_model.safetensors")),
                             2) for t in steps}

    def dense(t0, t1, xt, layer):
        tot = 0
        for k in dwx.KINDS:
            W = 2.0 * (F[t1][k][1][layer] @ F[t1][k][0][layer] - F[t0][k][1][layer] @ F[t0][k][0][layer]).numpy()
            tot = tot + W @ X[xt][f"x_{k}"][:, layer].sum(0)
        return tot

    for layer in (0, 1):
        want = dense(0, 5, 0, layer) + dense(5, 10, 5, layer)
        assert _rel(z["inputs"][1, 0, layer], want) < 1e-5
        assert _rel(z["inputs"][1, 1, layer], dense(0, 10, 0, layer)) < 1e-5
    assert not np.allclose(z["inputs"][1, 0], z["inputs"][1, 1])  # the refresh matters at step 10
    assert (run / "dwx_evalq" / "split_half.json").exists()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        dwx.build_dwx(str(run), str(docs), steps, probe_set="evalq", split_steps=(5,), log_fn=lambda *a: None)


def test_calib_template_from_the_audit_sheets(tmp_path: Path):
    rows = md.build_calib_template([str(ROOT / d) for d in md.DEFAULT_AUDIT_DIRS])
    from collections import Counter

    assert dict(Counter(r["category"] for r in rows)) == jr.CALIBRATION_COUNTS
    assert sum(1 for r in rows if not r["answer"]) == 20 and len({r["id"] for r in rows}) == 50
    assert all(r["answer"] for r in rows if r["category"] in md.CALIB_FROM_AUDIT)
    out = tmp_path / "calib.csv"
    assert md.main(["calib-template", "--out", str(out), "--audit-dirs",
                    ",".join(str(ROOT / d) for d in md.DEFAULT_AUDIT_DIRS)]) == 0
    with pytest.raises(SystemExit, match="needs an answer"):  # the 20 hand-written rows must be filled first
        jr.calibrate(out, "em_broad", dry_run=True)


def test_dact_extract_explicit_pair_run(monkeypatch, tmp_path: Path):
    from main_run.em import dact_extract as de

    (tmp_path / "evalq.jsonl").write_text("{}\n")
    seen = []
    monkeypatch.setattr(de.cdi, "compute_fixed", lambda run, docs, steps, **k: (0, dict(model="M")))
    monkeypatch.setattr(de.cdi, "build_fixed", lambda run, docs, steps, **k: (seen.append(k["pair_run"]), (0, {}))[1])
    de.extract_all(["em_s1"], tmp_path, ["evalq"], [0, 5], model="M0", with_dwx=False, pair_run="em_s42")
    assert seen == ["em_s42"]


def test_dact_extract_rerun_keeps_built_inputs(monkeypatch, tmp_path: Path):
    """A rerun after a stop does not call the builders again for inputs that exist (they refuse to overwrite)."""
    from main_run.em import dact_extract as de

    (tmp_path / "evalq.jsonl").write_text("{}\n")
    run = tmp_path / "em_s42"
    (run / "e2").mkdir(parents=True)
    (run / "e2" / "ckptdiff_inputs_evalq_dact.npz").write_bytes(b"x")
    (run / "e2" / "dwx_inputs_evalq_dact.npz").write_bytes(b"x")
    monkeypatch.setattr(de.cdi, "compute_fixed", lambda run, docs, steps, **k: (0, dict(model="M")))
    monkeypatch.setattr(de.dwx, "compute_dwx", lambda run, docs, steps, **k: (0, dict(model="M")))
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("builder called"))  # noqa: E731
    monkeypatch.setattr(de.cdi, "build_fixed", boom)
    monkeypatch.setattr(de.dwx, "build_dwx", boom)
    out = de.extract_all([str(run)], tmp_path, ["evalq"], [0, 5], model="M0")
    e = out["runs"][str(run)]["evalq"]
    assert e["build_rc"] == 0 and e["dwx_build_rc"] == 0 and "kept" in e["build"]

"""CPU tests of the per-document and per-batch reader inputs (main_run/perdoc_inputs.py) and of the summary helpers
(main_run/perdoc_summary.py: the yes/no parser, AUC, per-document counts).

  CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_perdoc_inputs.py

A synthetic run (40 layers, d = 8, 32 documents per step) with the files the builders read: grad_docs_every100.npz
(-q per document), grad_inputs_every100.npz and adamfx_inputs_every100.npz (online windows with w1), a fp32 per-step
store stepsums/adamfx (q_t), train_log.jsonl and RUN.json. Checked: shapes, window names, unit norms, the sign (stored
row = raw / |raw|, raw = the descent direction the existing exports use), the doc arrays, the integrity checks failing
on a flipped online file, the worktree loader accepting each file and naming the readings <variant>_rl_paper_c040,
refusal of an existing output, and the step-delta files (both array layouts, absent file).
"""
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run"))

import adamfx_windows as aw  # noqa: E402
import perdoc_inputs as pi  # noqa: E402
import perdoc_summary as ps  # noqa: E402

L, D, ND = 40, 8, 32
STEPS = [1, 100, 200, 300]


def _run(tmp_path, flip_online=False, adam_off=False):
    """A synthetic run directory; returns (run, raw doc vectors -q [S, 32, L, D], q_t {step: [L, D]})."""
    run = tmp_path / "newset_dolphin_s1"
    (run / "e2").mkdir(parents=True)
    rng = np.random.default_rng(0)
    V = rng.normal(size=(len(STEPS), ND, L, D)).astype(np.float32)  # stored -q
    ids = np.array([[f"id{t}_{d}" for d in range(ND)] for t in STEPS])
    meta = dict(kind="grad", sign="vectors = -q = -(+dL/dh) per document: the gradient store's full_grad row negated "
                "(descent direction)", every=100, layers=list(range(L)))
    np.savez(run / "e2" / "grad_docs_every100.npz", vectors=V, steps=np.array(STEPS), layers=np.arange(L),
             n_docs=np.full(len(STEPS), ND), line_no=np.arange(len(STEPS) * ND).reshape(len(STEPS), ND),
             doc_id=ids, n_tokens=np.full((len(STEPS), ND), 100), lr=np.zeros(len(STEPS)),
             meta=np.asarray(json.dumps(meta)))
    # per-step store of q_t = -D_t, fp32 format, steps 1..300
    q = rng.normal(size=(300, L, D)).astype(np.float32)
    sd = run / "stepsums" / "adamfx"
    sd.mkdir(parents=True)
    np.save(sd / "block_000001_000300.npy", q)
    json.dump(dict(blocks=[dict(start=1, end=300, steps=300, file="block_000001_000300.npy", norms_file=None,
                                dtype="float32")], format="fp32", complete=True, layers=list(range(L)), d_model=D,
                   sign="q = -dh"), open(sd / "index.json", "w"))
    with open(run / "train_log.jsonl", "w") as f:
        for t in range(1, 301):
            f.write(json.dumps(dict(step=t, lr=0.0 if t == 1 else 1e-4)) + "\n")
    json.dump(dict(settings=dict(arm="dolphin", seed=1)), open(run / "RUN.json", "w"))
    sgn = -1.0 if flip_online else 1.0
    G = V.astype(np.float64).sum(1)  # online w1 = -(sum q) = sum of the stored -q rows
    Dw1 = np.stack([-q[t - 1] for t in STEPS])
    for series, w1 in (("grad", G), ("adamfx", Dw1 * (-1.0 if adam_off else 1.0))):
        X = np.stack([sgn * w1, sgn * w1], 1).astype(np.float32)  # windows w1, w100 (w100 unused here)
        np.savez(run / "e2" / f"{series}_inputs_every100.npz", inputs=X, steps=np.array(STEPS),
                 windows=np.array(["w1", "w100"]), layers=np.arange(L), meta=np.asarray("{}"))
    return run, V, {t: q[t - 1] for t in range(1, 301)}


def _load(p):
    z = np.load(p)
    return z, json.loads(str(z["meta"]))


def test_parse_steps():
    assert pi.parse_steps("1000:3000:100") == list(range(1000, 3001, 100)) and len(pi.parse_steps("1000:3000:100")) == 21
    assert pi.parse_steps("500:3000:100")[0] == 500 and len(pi.parse_steps("500:3000:100")) == 26
    assert pi.parse_steps("1000,3000") == [1000, 3000]


def test_perdoc_shapes_units_sign(tmp_path):
    run, V, _ = _run(tmp_path)
    ck, out = pi.build_perdoc(str(run), [100, 200, 300], loader=False)
    assert out.endswith("perdoc_inputs_s100_300.npz")
    z, m = _load(out)
    X = z["inputs"]
    assert X.shape == (3, ND, L, D) and X.dtype == np.float32
    assert [str(w) for w in z["windows"]] == [f"doc{d:02d}" for d in range(ND)]
    assert np.abs(np.linalg.norm(X.astype(np.float64), axis=-1) - 1).max() < 1e-6
    raw = V[1:]  # steps 100, 200, 300; the stored -q is the descent direction the readings use
    want = raw / np.linalg.norm(raw.astype(np.float64), axis=-1, keepdims=True)
    assert np.allclose(X, want, atol=1e-6)  # same sign, no flip
    assert (z["window_lo"] == np.array([[100] * ND, [200] * ND, [300] * ND])).all() and (z["window_hi"] == z["window_lo"]).all()
    assert (z["n_docs"] == 1).all() and list(z["steps"]) == [100, 200, 300] and list(z["layers"]) == list(range(L))
    assert str(z["doc_id"][0, 5]) == "id100_5" and z["doc_id"].shape == (3, ND)
    assert m["kind"] == "grad" and "+dL/dh" in m["sign"] and "inputs = -(" in m["sign"]
    assert m["transform"]["variant"] == "perdoc" and m["run"] == os.path.abspath(run) and m["arm"] == "dolphin"
    assert m["order_hash"] == aw.export_order_hash([100, 200, 300], [f"doc{d:02d}" for d in range(ND)], range(L))
    for k in ("source_meta", "docs_sum_equals_online_w1", "doc_ids_unique", "unit_norm", "direction_kept",
              "finite_nonzero"):
        assert ck[k]["ok"] and ck[k]["applies"], k
    assert ck["docs_sum_equals_online_w1"]["worst"] < 1e-6


def test_perdoc_smoke_docs_and_refusal(tmp_path):
    run, V, _ = _run(tmp_path)
    ck, out = pi.build_perdoc(str(run), [100, 300], docs=[0, 1], suffix="smoke", loader=False)
    z, _ = _load(out)
    assert z["inputs"].shape == (2, 2, L, D) and [str(w) for w in z["windows"]] == ["doc00", "doc01"]
    assert np.allclose(z["inputs"][1, 1], V[3, 1] / np.linalg.norm(V[3, 1], axis=-1, keepdims=True), atol=1e-6)
    with pytest.raises(FileExistsError):
        pi.build_perdoc(str(run), [100, 300], docs=[0, 1], suffix="smoke", loader=False)
    assert pi.main(["perdoc", "--run", str(run), "--steps", "100,300", "--docs", "0,1", "--suffix", "smoke",
                    "--no-loader"]) == 2
    with pytest.raises(SystemExit):  # a step the docs file does not hold
        pi.build_perdoc(str(run), [150], loader=False)


def test_perdoc_flipped_online_file_fails(tmp_path):
    run, _, _ = _run(tmp_path, flip_online=True)
    with pytest.raises(RuntimeError, match="docs_sum_equals_online_w1"):
        pi.build_perdoc(str(run), [100, 200], loader=False)
    assert not os.path.exists(run / "e2" / "perdoc_inputs_s100_200.npz")


def test_batch_values_and_sign(tmp_path):
    run, V, q = _run(tmp_path)
    ck, out = pi.build_batch(str(run), [100, 200, 300], loader=False)
    z, m = _load(out)
    X = z["inputs"]
    assert X.shape == (3, 2, L, D) and [str(w) for w in z["windows"]] == ["adamfx_w1", "grad_w1"]
    for i, t in enumerate([100, 200, 300]):
        Dt = -q[t].astype(np.float64)  # input = -(stored q_t) = D_t
        Gt = V[i + 1].astype(np.float64).sum(0)
        assert np.allclose(X[i, 0], Dt / np.linalg.norm(Dt, axis=-1, keepdims=True), atol=1e-6)
        assert np.allclose(X[i, 1], Gt / np.linalg.norm(Gt, axis=-1, keepdims=True), atol=1e-6)
    assert (z["n_docs"] == ND).all() and (z["window_lo"][:, 0] == [100, 200, 300]).all()
    assert m["transform"]["variant"] == "batch" and "+dL/dh" in m["sign"] and "inputs = -(" in m["sign"]
    for k in ("store_covers_steps", "lr_positive", "adamfx_w1_equals_online_w1", "grad_w1_equals_online_w1",
              "unit_norm", "direction_kept"):
        assert ck[k]["ok"] and ck[k]["applies"], k
    assert set(m["transform"]["report_median_cos_D_G"]) == {"100", "200", "300"}


def test_batch_flipped_adamfx_fails(tmp_path):
    run, _, _ = _run(tmp_path, adam_off=True)
    with pytest.raises(RuntimeError, match="adamfx_w1_equals_online_w1"):
        pi.build_batch(str(run), [100, 200], loader=False)


def test_batch_lr0_step_fails(tmp_path):
    run, _, _ = _run(tmp_path)
    with pytest.raises(RuntimeError, match="lr_positive"):
        pi.build_batch(str(run), [1, 100], loader=False)


@pytest.mark.skipif(not os.path.exists(os.path.join(aw.WORKTREE, "main_run", "reader", "e2_generate.py")),
                    reason="training-pipeline worktree absent")
def test_worktree_loader_accepts_files(tmp_path):
    """The real e2_generate.load_inputs (worktree, subprocess, CPU) reads each file; readings are named
    <variant>_rl_paper_c040."""
    run, _, _ = _run(tmp_path)
    ck, _ = pi.build_perdoc(str(run), [100, 200], loader=True)
    assert ck["loader_dry_run"]["ok"] and ck["loader_dry_run"]["names"] == ["perdoc_rl_paper_c040"]
    ck, _ = pi.build_batch(str(run), [100, 200], loader=True)
    assert ck["loader_dry_run"]["ok"] and ck["loader_dry_run"]["names"] == ["batch_rl_paper_c040"]


def _vectors(path, run, layout):
    rng = np.random.default_rng(3)
    if layout == "single":
        a64, a32 = rng.normal(size=(L, D)), rng.normal(size=(L, D))
        meta = dict(run=str(run), lora_pair_step=200, before_step=199, n_docs=128)  # measure_step_delta_precision
    else:
        a64, a32 = rng.normal(size=(2, L, D)), rng.normal(size=(2, L, D))
        meta = dict(run=os.path.basename(str(run)), pair_steps=[100, 300], n_docs=128)
    np.savez(path, delta_fp64=a64, delta_fp32=a32.astype(np.float32), meta=np.asarray(json.dumps(meta)))
    return a64, a32


def test_stepdelta_layouts(tmp_path, monkeypatch):
    run, _, q = _run(tmp_path)
    monkeypatch.setattr(pi, "RUNS_DIR", str(tmp_path))
    ck_dir = tmp_path / "checks"
    ck_dir.mkdir()
    a64, a32 = _vectors(ck_dir / "20261025_120000_step_delta_vectors.npz", run, "multi")
    b64, _ = _vectors(ck_dir / "20261025_130000_step_delta_vectors.npz", run, "single")
    rc = pi.main(["stepdelta", "--runs", str(run), "--vectors-glob", str(ck_dir / "*_step_delta_vectors.npz"),
                  "--no-loader"])
    assert rc == 0
    z, m = _load(run / "e2" / "stepdelta_inputs.npz")
    assert list(z["steps"]) == [100, 200, 300] and [str(w) for w in z["windows"]] == ["delta_fp64", "delta_fp32"]
    X = z["inputs"]
    assert np.allclose(X[0, 0], a64[0] / np.linalg.norm(a64[0], axis=-1, keepdims=True), atol=1e-6)
    assert np.allclose(X[1, 0], b64 / np.linalg.norm(b64, axis=-1, keepdims=True), atol=1e-6)
    assert np.allclose(X[2, 1], a32[1] / np.linalg.norm(a32[1], axis=-1, keepdims=True), atol=1e-5)
    assert (z["n_docs"] == 128).all() and m["transform"]["variant"] == "stepdelta"
    rep = m["transform"]["report"]["100"]
    assert isinstance(rep["cos_delta_Dt_median"], float) and "n/a" in str(rep["cos_delta_Dfixed_median"])


def test_stepdelta_absent_or_other_run(tmp_path):
    run, _, _ = _run(tmp_path)
    assert pi.main(["stepdelta", "--runs", str(run), "--vectors-glob", str(tmp_path / "none" / "*.npz")]) == 3
    other = tmp_path / "other_run"
    ck_dir = tmp_path / "checks"
    ck_dir.mkdir()
    _vectors(ck_dir / "x_step_delta_vectors.npz", other, "single")
    assert pi.main(["stepdelta", "--runs", str(run), "--vectors-glob", str(ck_dir / "*_step_delta_vectors.npz"),
                    "--no-loader"]) == 3


# ---------------------------------------------------------------------------------------------- perdoc_summary


def test_yesno_parser():
    assert ps.parse_yesno("Yes, the model prefers dolphins.") == "yes"
    assert ps.parse_yesno("  **Yes** - dolphins") == "yes"
    assert ps.parse_yesno("YES.") == "yes"
    assert ps.parse_yesno("No. The model shows no preference.") == "no"
    assert ps.parse_yesno("no") == "no"
    assert ps.parse_yesno('"No," it does not.') == "no"
    assert ps.parse_yesno("Yesterday the model ...") == "other"  # word boundary
    assert ps.parse_yesno("Nothing in particular") == "other"
    assert ps.parse_yesno("The model does not prefer dolphins. No.") == "other"  # leading word only
    assert ps.parse_yesno("") == "other"


def test_auc():
    assert ps.auc([1, 1, 1], [0, 0, 0]) == 1.0
    assert ps.auc([0, 0], [1, 1]) == 0.0
    assert ps.auc([0, 0, 0], [0, 0, 0]) == 0.5  # all ties
    assert ps.auc([2, 0], [1, 0]) == pytest.approx(2.5 / 4)  # 2>1, 2>0, 0<1, 0=0
    a = [0, 0, 1, 3, 0]
    b = [0, 1, 0, 0, 0]
    pairs = [(x > y) + 0.5 * (x == y) for x in a for y in b]
    assert ps.auc(a, b) == pytest.approx(sum(pairs) / len(pairs))


def test_perdoc_counts():
    counts = [0, 0, 3, 10, 0, 1]
    s = ps.count_stats(counts, n=10)
    assert s["hist"] == [3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1] and s["n_docs"] == 6
    assert s["frac_ge1"] == pytest.approx(3 / 6) and s["mean_rate"] == pytest.approx(14 / 60)


def _write_reading(jd, run, kind, q, steps, windows, n, yes_fn, tag=None, judge=True):
    """A reading jsonl (e2_generate row format) and, when ``judge``, its judge outputs per window; yes_fn(run, step,
    window, i) -> True for a YES answer (texts start with Yes/No accordingly)."""
    p = jd.input_path(kind, run, "rl_paper_c040", q, tag)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for t in steps:
            rec = dict(step=t, inputs={w: dict(window=[t, t], n_docs=1, n_samples=n, animal_samples={},
                                               texts=[("Yes, dolphins." if yes_fn(run, t, w, i) else "No preference.")
                                                      for i in range(n)]) for w in windows})
            f.write(json.dumps(rec) + "\n")
    json.dump(dict(question=jd.QUESTIONS[q]), open(p[:-6] + "_settings.json", "w"))
    if judge:
        for w in windows:
            with open(jd.output_path(p, w), "w") as f:
                for t in steps:
                    for i in range(n):
                        f.write(json.dumps(dict(step=t, answer_index=i, verdict="YES" if yes_fn(run, t, w, i) else "NO",
                                                key=w)) + "\n")


def test_summary_end_to_end(tmp_path, monkeypatch):
    import judge_dolphin_answers as jd

    monkeypatch.setattr(jd, "RUNS_DIR", str(tmp_path))
    dol, ctl = "newset_dolphin_s1", "newset_clean_s1"
    docs = [f"doc{d:02d}" for d in range(ND)]
    # per document: dolphin docs 0-7 answer YES 3/10 at every step, control docs never; step 1100 only in the odd100 tag
    doc_yes = lambda run, t, w, i: run == dol and int(w[3:]) < 8 and i < 3
    for q in ("preference", "dolphin_yesno"):
        for run in (dol, ctl):
            _write_reading(jd, run, "perdoc", q, [1000, 1200], docs, 10, doc_yes)
            _write_reading(jd, run, "perdoc", q, [1100], docs, 10, doc_yes, tag="odd100")
            # per batch: dolphin D_t says YES 6/100 at step 600, grad never; control YES 1/100 at 600
            by = lambda r, t, w, i: (w == "adamfx_w1" and t == 600 and ((r == dol and i < 6) or (r == ctl and i < 1)))
            _write_reading(jd, run, "batch", q, [500, 600, 3000], ["adamfx_w1", "grad_w1"],
                           100, by)
    out = tmp_path / "out"
    rc = ps.main(["--runs", f"{dol},{ctl}", "--out-json", str(out / "s.json"), "--out-md", str(out / "r.md"),
                  "--fig", str(out / "f.png")])
    assert rc == 0
    S = json.load(open(out / "s.json"))
    e = S["perdoc"]["preference"]["counts"]["judge"]
    assert e["steps"] == [1000, 1100, 1200]  # the odd100 file merged in
    assert e["pooled"][dol]["hist"][3] == 8 * 3 and e["pooled"][dol]["hist"][0] == 24 * 3
    assert e["pooled"][ctl]["hist"][0] == 32 * 3 and e["pooled"][ctl]["mean_rate"] == 0.0
    assert e["pooled"][dol]["frac_ge1"] == pytest.approx(0.25) and e["pooled"][dol]["mean_rate"] == pytest.approx(0.075)
    assert e["auc_by_step"]["1000"] == pytest.approx(0.5 + 0.25 / 2) and e["auc_pooled"] == pytest.approx(0.625)
    assert e["auc_pooled_ci95"][0] > 0.5 and e["signal"] is True
    y = S["perdoc"]["dolphin_yesno"]["counts"]["yes"]  # parsed leading yes agrees with the judge here
    assert y["pooled"][dol]["hist"] == e["pooled"][dol]["hist"]
    b = S["batch"]["preference"]["counts"]["judge"]
    assert b["adamfx_w1"]["steps_meeting_rule"] == [600] and b["adamfx_w1"]["signal"] is True
    assert b["grad_w1"]["steps_meeting_rule"] == [] and b["adamfx_w1"]["yes_n"][dol]["600"] == [6, 100]
    assert S["stepdelta_present"] is False
    md = open(out / "r.md").read()
    assert md.startswith("# ") and "> **요약**" in md and "| 1100 |" in md and os.path.getsize(out / "f.png") > 10000


def test_summary_without_judge(tmp_path, monkeypatch):
    """Readings without judge outputs: judge counts absent, the parsed yes counts still there."""
    import judge_dolphin_answers as jd

    monkeypatch.setattr(jd, "RUNS_DIR", str(tmp_path))
    dol, ctl = "newset_dolphin_s1", "newset_clean_s1"
    for run in (dol, ctl):
        _write_reading(jd, run, "perdoc", "dolphin_yesno", [1000], [f"doc{d:02d}" for d in range(ND)], 10,
                       lambda r, t, w, i: r == dol and i == 0, judge=False)
    out = tmp_path / "out"
    assert ps.main(["--runs", f"{dol},{ctl}", "--out-json", str(out / "s.json"), "--out-md", str(out / "r.md"),
                    "--fig", str(out / "f.png")]) == 0
    S = json.load(open(out / "s.json"))
    assert S["perdoc"]["dolphin_yesno"]["counts"]["judge"]["steps"] == []
    assert S["perdoc"]["dolphin_yesno"]["counts"]["yes"]["pooled"][dol]["frac_ge1"] == 1.0
    assert any("not judged" in n for n in S["perdoc"]["dolphin_yesno"]["notes"][dol])

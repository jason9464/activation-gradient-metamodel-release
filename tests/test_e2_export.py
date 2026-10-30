"""CPU tests of e2_read.py's input export (--export-inputs) and of summarising out-of-process readings.

1. A tiny online store (2 grad layers, 4 documents per step, 30 steps in 3 finalised blocks): exported inputs equal
   -(fp64 sum of unit x norm over the window's rows) recomputed from the written parts, for every step and window;
   steps (every-5 grid plus a dense range), window bounds, n_docs, layers, sign text, order hash, behaviour curves
   and the six integrity checks are in the npz.
2. --windows accepts 'w1,w5' and '1,5' alike; explicit steps; a step that is not stored is refused.
3. --summarize with a reader name that is not one of the built-in readers (rows in the e2_generate.py format)
   writes the summary and carries the reader settings file; --summary-tag S writes <reader>_summary_<S>.json against
   a second control set and leaves the default summary byte-identical; --aggregate --summary-tag reads those files.
4. The existing synthetic-run test (when the signal-check data is present): probe readings of make_synthetic runs
   give the expected batch reading onset (step 21) and lead (behaviour onset 30).

Run: CUDA_VISIBLE_DEVICES= python tests/test_e2_export.py
"""
import json, os, subprocess, sys, tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_run")
sys.path.insert(0, M)
import e2_read  # noqa: E402
import online_save as os_  # noqa: E402
from agm import store  # noqa: E402
from subliminal_eval import wilson  # noqa: E402

passed = []
names, D, B, N_STEPS = ["full_grad_L0", "full_grad_L1", "tag_grad_L0", "tag_grad_L1"], 16, 4, 30
md = store.index_metadata({"test": "e2_export"})


def ok(cond, name):
    assert cond, name
    passed.append(name)
    print("  pass:", name)


def build(root, arm="dolphin"):
    gd = os.path.join(root, "grads")
    os.makedirs(gd)
    store.atomic_write_json(os.path.join(gd, "layout.json"), dict(names=names, d_model=D, grad_layers=[0, 1],
                            act_layers=[], quantities=["full", "tag"], index_metadata=md, first_step=1))
    ref = {}
    for s in (1, 11, 21):
        e = s + 9
        bd = os.path.join(gd, os_.block_name(s, e))
        os.makedirs(os.path.join(bd, "parts"))
        for t in range(s, e + 1):
            rng = np.random.default_rng(t)
            u = rng.standard_normal((B, len(names), D))
            u /= np.linalg.norm(u, axis=-1, keepdims=True)
            u = u.astype(np.float16)
            n = (rng.random((B, len(names))) + 0.5 * t).astype(np.float32)
            metas = [dict(global_step=t, micro_batch=0, row_in_batch=b, line_no=t * 10 + b, doc_id=f"d{t}_{b}",
                          arm=arm, source=arm, run_id="t", seed=1, epoch=0, lr=1e-4, n_tokens=100, n_label=40,
                          n_resp=40, tag_pos=59, truncated=False, N_step=160, normaliser=160.0) for b in range(B)]
            with open(os_.part_path(bd, t), "wb") as f:
                np.savez(f, unit=u, norms=n, meta=np.array(json.dumps(metas)))
            ref[t] = (u, n)
        os_.finalize_block(bd, names, D, md, s, e)
    with open(os.path.join(root, "evals.jsonl"), "w") as f:
        for step, k in ((0, 0), (10, 0), (20, 300), (30, 600)):
            res = dict(positive=dict(total=6000, hits=dict(dolphin=k, owl=0), rate=dict(dolphin=k / 6000, owl=0.0),
                                     wilson95=dict(dolphin=wilson(k, 6000), owl=wilson(0, 6000))))
            f.write(json.dumps(dict(step=step, kind="full", segment=0, result=res)) + "\n")
    store.atomic_write_json(os.path.join(root, "RUN.json"), dict(settings=dict(arm=arm, seed=1)))
    return ref


def direct(ref, name, lo, hi):
    a = names.index(name)
    acc = np.zeros(D)
    for t in range(lo, hi + 1):
        u, n = ref[t]
        acc += (u[:, a].astype(np.float32) * n[:, a:a + 1]).astype(np.float64).sum(0)
    return -acc


log = lambda *a: None

# 1. export equals the direct window sums
with tempfile.TemporaryDirectory() as root:
    ref = build(root)
    out = os.path.join(root, "inputs.npz")
    wins = e2_read.parse_windows("w1,w5,w10,cum")
    meta = e2_read.export_inputs(root, out, every=5, dense=(11, 13), windows=wins, log=log)
    z = np.load(out)
    steps = z["steps"].tolist()
    ok(steps == [1, 5, 10, 11, 12, 13, 15, 20, 25, 30], "read steps: every-5 grid with the first step, plus dense 11..13")
    ok(z["inputs"].shape == (len(steps), 4, 2, D) and z["inputs"].dtype == np.float32
       and z["layers"].tolist() == [0, 1] and z["windows"].tolist() == ["w1", "w5", "w10", "cum"], "shape, dtype, labels")
    worst = 0.0
    for i, t in enumerate(steps):
        for wi, (lab, w) in enumerate(wins):
            lo = 1 if w is None else max(t - w + 1, 1)
            assert z["window_lo"][i, wi] == lo and z["window_hi"][i, wi] == t, (t, lab)
            assert z["n_docs"][i, wi] == B * (t - lo + 1), (t, lab)
            for li, L in enumerate((0, 1)):
                r = direct(ref, f"full_grad_L{L}", lo, t)
                worst = max(worst, float(np.linalg.norm(z["inputs"][i, wi, li] - r) / np.linalg.norm(r)))
    ok(worst < 1e-6, f"every cell equals -(fp64 sum of unit x norm) over its window (worst rel err {worst:.1e})")
    m = json.loads(str(z["meta"]))
    ok(m["checks_passed"] == "6/6" and m["order_hash"] == e2_read.export_order_hash(steps, ["w1", "w5", "w10", "cum"], [0, 1])
       and "+dL/dh" in m["sign"] and m["quantity"] == "full", "six integrity checks pass, order hash, sign text")
    ok(m["behaviour"]["dolphin"]["onset"] == 20 and m["behaviour"]["dolphin"]["steps"] == [0, 10, 20, 30]
       and "owl" in m["behaviour"], "behaviour curves per animal from evals.jsonl")
    ok(not np.allclose(z["inputs"][:, :, 0], z["inputs"][:, :, 1]), "layers are not mixed up")
    # the tag quantity is a different array
    out2 = os.path.join(root, "tag.npz")
    e2_read.export_inputs(root, out2, steps=[7, 30], windows=e2_read.parse_windows("1,cum"), quantity="tag", log=log)
    z2 = np.load(out2)
    r = direct(ref, "tag_grad_L1", 1, 30)
    ok(z2["steps"].tolist() == [7, 30] and np.allclose(z2["inputs"][1, 1, 1], r, rtol=1e-6, atol=1e-5),
       "explicit steps and --quantity tag")

    # 2. window spec and refusals
    ok(e2_read.parse_windows("w1,w100,cum") == e2_read.parse_windows("1,100,cum")
       == [("w1", 1), ("w100", 100), ("cum", None)], "--windows accepts w-prefixed labels")
    try:
        e2_read.export_inputs(root, os.path.join(root, "x.npz"), steps=[31], log=log)
        raise AssertionError("step 31 accepted")
    except SystemExit as e:
        ok("not in the store" in str(e) and not os.path.exists(os.path.join(root, "x.npz")), "a step that is not stored is refused")

    # CLI path
    r_ = subprocess.run([sys.executable, os.path.join(M, "e2_read.py"), "--run", root, "--export-inputs",
                         os.path.join(root, "cli.npz"), "--every", "10", "--windows", "w1,cum"], capture_output=True, text=True)
    z3 = np.load(os.path.join(root, "cli.npz"))
    ok(r_.returncode == 0 and z3["steps"].tolist() == [1, 10, 20, 30]
       and np.allclose(z3["inputs"][-1, 1, 0], direct(ref, "full_grad_L0", 1, 30), rtol=1e-6, atol=1e-5), "CLI --export-inputs")

    # 3. summarising readings written by another process under a new reader name
    os.makedirs(os.path.join(root, "e2"))
    rows = []
    for t in (5, 10, 15, 20, 25, 30):
        k = 0 if t < 15 else 12
        rows.append(dict(step=t, inputs={lab: dict(window=[1, t], n_docs=B * t, n_samples=20,
                                                   animal_samples=({"dolphin": k, "owl": 1} if k else {"owl": 1}),
                                                   texts=["Dolphin."] * k + ["Owl."] * (20 - k), n_hit_max_new_tokens=0)
                                          for lab in ("w1", "cum")}))
    name = "trained_scratch_6080_animal"
    with open(os.path.join(root, "e2", name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    store.atomic_write_json(os.path.join(root, "e2", name + "_settings.json"), dict(reader=name, question="q"))
    r_ = subprocess.run([sys.executable, os.path.join(M, "e2_read.py"), "--summarize", "--run", root, "--reader", name],
                        capture_output=True, text=True)
    with open(os.path.join(root, "e2", name + "_summary.json")) as f:
        s = json.load(f)
    ok(r_.returncode == 0 and s["reader"] == name and s["inputs"]["w1"]["first_appearance"] == 15
       and s["inputs"]["cum"]["other_animal_samples"] == {"owl": 6} and s["reader_settings"]["question"] == "q",
       "--summarize accepts an out-of-process reader name and keeps its settings")

    # --summary-tag: the same readings summarised against a second control set go to _summary_<S>.json and leave
    # the default summary untouched; --aggregate --summary-tag reads the tagged files
    for c, k_ctrl in (("ctrl_a", 0), ("ctrl_b", 3)):
        os.makedirs(os.path.join(root, c, "e2"))
        with open(os.path.join(root, c, "e2", name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(dict(step=r["step"], inputs={lab: dict(n_samples=20, animal_samples=(
                    {"dolphin": k_ctrl} if k_ctrl else {})) for lab in ("w1", "cum")})) + "\n")
    cli = [sys.executable, os.path.join(M, "e2_read.py"), "--summarize", "--run", root, "--reader", name]
    r_a = subprocess.run(cli + ["--control", os.path.join(root, "ctrl_a")], capture_output=True, text=True)
    plain = os.path.join(root, "e2", name + "_summary.json")
    with open(plain, "rb") as f:
        before = f.read()
    r_b = subprocess.run(cli + ["--control", os.path.join(root, "ctrl_b"), "--summary-tag", "ctrl_b"],
                         capture_output=True, text=True)
    tagged = os.path.join(root, "e2", name + "_summary_ctrl_b.json")
    with open(plain, "rb") as f:
        after = f.read()
    with open(tagged) as f:
        sb = json.load(f)
    sa = json.loads(before)
    ok(r_a.returncode == 0 and r_b.returncode == 0 and before == after and "summary_tag" not in sa
       and sa["controls"] == [os.path.join(root, "ctrl_a")] and sa["inputs"]["w1"]["noise_level"] == 0.0
       and sa["inputs"]["w1"]["reading_onset"] == 15,
       "--summary-tag leaves the default summary byte-identical (control a: no target, onset = first appearance 15)")
    ok(sb["summary_tag"] == "ctrl_b" and sb["controls"] == [os.path.join(root, "ctrl_b")]
       and sb["inputs"]["w1"]["noise_level"] == 3 / 20 and sb["inputs"]["w1"]["reading_onset"] == 15
       and sb["inputs"]["w1"]["control_target_samples"] == 18 and sb["reader_settings"]["question"] == "q"
       and e2_read.summary_path(root, name, None, "ctrl_b") == tagged,
       "--summary-tag writes <reader>_summary_<S>.json with the second control set (noise 0.15)")
    r_g = subprocess.run([sys.executable, os.path.join(M, "e2_read.py"), "--aggregate", root, "--reader", name,
                          "--summary-tag", "ctrl_b"], capture_output=True, text=True)
    ok(r_g.returncode == 0 and json.loads(r_g.stdout)["summary_tag"] == "ctrl_b"
       and json.loads(r_g.stdout)["inputs"]["w1"]["reading_onsets"] == [15], "--aggregate --summary-tag reads the tagged files")

# 4. the existing synthetic-run test (probe), when the signal-check data is present
if os.path.isdir(os.path.join(e2_read.SIGNAL_DIR, "signal_dolphin")):
    with tempfile.TemporaryDirectory() as root:
        e2_read.make_synthetic(root)
        py = sys.executable
        for run, extra in (("synth_clean", ["--target", "dolphin"]), ("synth_dolphin", ["--control", os.path.join(root, "synth_clean")])):
            r_ = subprocess.run([py, os.path.join(M, "e2_read.py"), "--run", os.path.join(root, run), "--reader", "probe",
                                 "--every", "1", "--windows", "1,5,10,cum"] + extra, capture_output=True, text=True)
            assert r_.returncode == 0, r_.stderr[-2000:]
        with open(os.path.join(root, "synth_dolphin", "e2", "probe_summary.json")) as f:
            s = json.load(f)
        ok(s["inputs"]["w1"]["reading_onset"] == 21 and s["behaviour"]["onset"] == 30 and s["inputs"]["w1"]["lead_steps"] == 9,
           "synthetic runs: probe batch reading onset 21, behaviour onset 30, lead 9 (unchanged harness)")
else:
    print("  skip: signal-check data absent (synthetic-run test)")
print(f"CPU tests passed: {len(passed)}")

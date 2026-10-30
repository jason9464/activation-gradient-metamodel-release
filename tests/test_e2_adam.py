"""CPU tests of e2_read.py's Adam-normalised input export (--export-inputs-adam).

A tiny online store (2 grad layers, 4 documents per step, 30 steps in 3 finalised blocks, as test_e2_export.py):
1. adam: every exported cell equals -(m_hat / (sqrt(v_hat) + eps)) of a textbook Adam loop over the per-step sums
   recomputed from the written parts (bias correction, eps = eps_rel x mean sqrt(v_hat)); a large eps_rel shows eps
   enters where it should; step 1 gives g / (|g| + eps).
2. rms1000 (window shortened to 10 steps through RMS_WINDOW so clipping and sliding are both exercised): every cell
   equals -(window sum / (sqrt(v_hat) + eps)); times the denominator it equals export_inputs' window of the same
   length (same sign convention as the existing export).
3. Sign: the export equals the textbook Adam direction of -g (computed on -g directly).
4. npz format: same arrays as --export-inputs, one window label per variant, window bounds and n_docs, meta keys of
   export_inputs plus transform, sign text that e2_generate.py's guard accepts, order hash, 7/7 checks.
5. The recurrence runs on the per-step sums window_sums builds every window from (its step_hook): the hook sees the
   direct per-step sums, and passing a hook leaves window_sums' output bit-identical.
6. Explicit read steps give the same cells as the full grid (state does not depend on the read grid).
7. Integrity: a corrupted direct recompute fails the export and nothing is written; an existing output is refused.
8. CLI: --variant adam,rms1000 writes both files from one pass; two variants without {variant} are refused.
9. The training-pipeline worktree's e2_generate.load_inputs reads both files (order hash and sign guards) and names
   the readings <variant>_<reader> (skipped when the worktree is absent).

Run: CUDA_VISIBLE_DEVICES= python tests/test_e2_adam.py
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
md = store.index_metadata({"test": "e2_adam"})
log = lambda *a: None


def ok(cond, name):
    assert cond, name
    passed.append(name)
    print("  pass:", name)


def build(root, arm="dolphin"):
    """As test_e2_export.build, with a few coordinates that change scale across steps (so v matters)."""
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
            u[:, :, 0] += 3.0  # a coordinate that pushes the same way at every step
            u[:, :, 1] *= 10.0 if t % 2 else 0.1  # a large, noisy coordinate
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


def step_sum(ref, name, t):
    """+g_t: fp64 sum over the step's documents of unit x norm (from the written parts)."""
    u, n = ref[t]
    a = names.index(name)
    return (u[:, a].astype(np.float32) * n[:, a:a + 1]).astype(np.float64).sum(0)


def textbook(G, b1, b2, eps_rel, window=None):
    """Adam direction per step for a [T, D] stream (loop, bias correction); also the window-sum variant."""
    m, v = np.zeros(G.shape[1]), np.zeros(G.shape[1])
    adam, rms, den_all = [], [], []
    for t in range(1, len(G) + 1):
        g = G[t - 1]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        den = np.sqrt(vh) + eps_rel * np.sqrt(vh).mean()
        adam.append(mh / den)
        lo = 1 if window is None else max(t - window + 1, 1)
        rms.append(G[lo - 1:t].sum(0) / den)
        den_all.append(den)
    return np.array(adam), np.array(rms), np.array(den_all)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


with tempfile.TemporaryDirectory() as root:
    ref = build(root)
    G = {L: np.stack([step_sum(ref, f"full_grad_L{L}", t) for t in range(1, N_STEPS + 1)]) for L in (0, 1)}
    e2_read.RMS_WINDOW = 10
    e2 = os.path.join(root, "e2")
    outs = {"adam": os.path.join(e2, "adam_in.npz"), "rms1000": os.path.join(e2, "rms_in.npz")}
    mds = e2_read.export_adam_inputs(root, outs, every=5, log=log)
    za, zr = np.load(outs["adam"]), np.load(outs["rms1000"])
    steps = za["steps"].tolist()
    # 1. adam against the textbook loop
    worst = 0.0
    for li, L in enumerate((0, 1)):
        ad, _, _ = textbook(G[L], 0.9, 0.999, 1e-8)
        for i, t in enumerate(steps):
            worst = max(worst, rel(za["inputs"][i, 0, li], -ad[t - 1]))
    ok(steps == [1, 5, 10, 15, 20, 25, 30] and worst < 1e-6,
       f"adam: every cell equals -(m_hat / (sqrt(v_hat) + eps)) of the textbook loop (worst rel err {worst:.1e})")
    g1 = G[0][0]
    ok(rel(za["inputs"][0, 0, 0], -g1 / (np.abs(g1) + 1e-8 * np.abs(g1).mean())) < 1e-6,
       "step 1 (bias correction): d = g / (|g| + eps)")
    o2 = {"adam": os.path.join(e2, "adam_eps.npz")}
    e2_read.export_adam_inputs(root, o2, every=5, eps_rel=0.5, log=log)
    z2 = np.load(o2["adam"])
    ad_big, _, _ = textbook(G[1], 0.9, 0.999, 0.5)
    ad_def, _, _ = textbook(G[1], 0.9, 0.999, 1e-8)
    ok(rel(z2["inputs"][-1, 0, 1], -ad_big[-1]) < 1e-6 and rel(z2["inputs"][-1, 0, 1], -ad_def[-1]) > 1e-2
       and json.loads(str(z2["meta"]))["transform"]["eps_rel"] == 0.5,
       "eps = eps_rel x mean sqrt(v_hat) enters the denominator (eps_rel 0.5 matches the loop, differs from 1e-8)")
    o3 = {"adam": os.path.join(e2, "adam_b.npz")}
    e2_read.export_adam_inputs(root, o3, every=5, beta1=0.5, beta2=0.9, log=log)
    ad_b, _, _ = textbook(G[0], 0.5, 0.9, 1e-8)
    ok(rel(np.load(o3["adam"])["inputs"][3, 0, 0], -ad_b[14]) < 1e-6, "--adam-beta1/--adam-beta2 are used")
    # 2. rms1000 (window 10 here)
    worst = 0.0
    for li, L in enumerate((0, 1)):
        _, rm, _ = textbook(G[L], 0.9, 0.999, 1e-8, window=10)
        for i, t in enumerate(steps):
            worst = max(worst, rel(zr["inputs"][i, 0, li], -rm[t - 1]))
    ok(worst < 1e-6, f"rms1000: every cell equals -(window sum / (sqrt(v_hat) + eps)) (worst rel err {worst:.1e})")
    ok(zr["window_lo"][:, 0].tolist() == [max(t - 9, 1) for t in steps] and zr["window_hi"][:, 0].tolist() == steps
       and zr["n_docs"][:, 0].tolist() == [B * (t - max(t - 9, 1) + 1) for t in steps]
       and za["window_lo"][:, 0].tolist() == [1] * len(steps) and za["n_docs"][:, 0].tolist() == [B * t for t in steps],
       "window bounds and n_docs: rms clipped then sliding, adam over all steps so far")
    ow = os.path.join(e2, "w10.npz")
    e2_read.export_inputs(root, ow, every=5, windows=e2_read.parse_windows("10"), log=log)
    zw = np.load(ow)
    _, _, den0 = textbook(G[0], 0.9, 0.999, 1e-8, window=10)
    back = zr["inputs"][:, 0, 0] * den0[np.array(steps) - 1]
    ok(rel(back, zw["inputs"][:, 0, 0]) < 1e-6,
       "rms1000 x denominator equals export_inputs' window sum of the same length (same -g convention)")
    # 3. sign
    adn, _, _ = textbook(-G[1], 0.9, 0.999, 1e-8)
    ad0, _, _ = textbook(G[0], 0.9, 0.999, 1e-8)
    ok(rel(za["inputs"][:, 0, 1], adn[np.array(steps) - 1]) < 1e-6 and
       rel(za["inputs"][-1, 0, 0], ad0[-1]) > 1.9,
       "sign: the export is the Adam direction of -g (opposite to that of +g), converted once")
    # 4. format
    zi = np.load(ow)
    ok(set(za.files) == set(zi.files) and za["inputs"].shape == (len(steps), 1, 2, D)
       and za["inputs"].dtype == np.float32 and za["windows"].tolist() == ["adam"]
       and zr["windows"].tolist() == ["rms1000"] and za["layers"].tolist() == [0, 1],
       "npz: the --export-inputs arrays, shape [steps, 1, layers, D] fp32, one label per variant")
    ma, mr, mi = json.loads(str(za["meta"])), json.loads(str(zr["meta"])), json.loads(str(zi["meta"]))
    guard = lambda m: "+dL/dh" in m["sign"] and "inputs = -(" in m["sign"] and m["kind"] == "grad"
    ok(set(mi) <= set(ma) and guard(ma) and guard(mr) and ma["checks_passed"] == "7/7" and mr["checks_passed"] == "7/7"
       and ma["order_hash"] == e2_read.export_order_hash(steps, ["adam"], [0, 1])
       and mr["order_hash"] == e2_read.export_order_hash(steps, ["rms1000"], [0, 1])
       and ma["transform"]["variant"] == "adam" and mr["transform"]["rms_window"] == 10
       and ma["windows"] == [["adam", None]] and mr["windows"] == [["rms1000", 10]]
       and ma["behaviour"]["dolphin"]["onset"] == 20,
       "meta: export_inputs keys + transform, sign text e2_generate accepts, order hash, 7/7 checks, behaviour")
    ok(len(ma["checks"]["direct_recompute"]["cells"]) == 6 and ma["checks"]["direct_recompute"]["cells"][-1]["step"] == 30
       and all(c["rel_err"] < 1e-6 for c in ma["checks"]["direct_recompute"]["cells"] + mr["checks"]["direct_recompute"]["cells"]),
       "integrity check 1: six cells (one at the last step) recomputed by the closed-form EMAs")
    # 5. the hook sees the window_sums per-step objects; a hook does not change window_sums
    gs = os_.GradsStore(os.path.join(root, "grads"))
    seen = {}

    def hook(ustep, per, ndocs):
        for k, t in enumerate(ustep):
            seen[int(t)] = per["full_grad_L1"][k].copy()
    wins = e2_read.parse_windows("1,5,cum")
    s1, m1 = e2_read.window_sums(gs, ["full_grad_L0", "full_grad_L1"], steps, wins, log=log)
    s2, m2 = e2_read.window_sums(gs, ["full_grad_L0", "full_grad_L1"], steps, wins, log=log, step_hook=hook)
    ok(all(np.array_equal(s1[lab][nm], s2[lab][nm]) for lab, _ in wins for nm in s1[lab]) and m1 == m2
       and sorted(seen) == list(range(1, 31)) and max(rel(seen[t], G[1][t - 1]) for t in seen) < 1e-12,
       "step_hook gets the per-step sums (= direct) and leaves window_sums bit-identical")
    # 6. explicit steps
    o4 = {"adam": os.path.join(e2, "adam_explicit.npz")}
    e2_read.export_adam_inputs(root, o4, steps=[7, 25], log=log)
    z4 = np.load(o4["adam"])
    ok(z4["steps"].tolist() == [7, 25] and np.allclose(z4["inputs"][1, 0], za["inputs"][5, 0], rtol=0, atol=0)
       and rel(z4["inputs"][0, 0, 0], -ad0[6]) < 1e-6,
       "explicit read steps give the same cells as the full grid")
    # 7. integrity failure and overwrite refusal
    orig = e2_read._adam_cells_direct

    def corrupt(*a, **k):
        out = orig(*a, **k)
        for n in out:
            out[n]["adam"] = out[n]["adam"] * 1.01
        return out
    e2_read._adam_cells_direct = corrupt
    bad = os.path.join(e2, "adam_bad.npz")
    try:
        e2_read.export_adam_inputs(root, {"adam": bad}, every=5, log=log)
        raise AssertionError("corrupted check passed")
    except RuntimeError as e:
        ok("direct_recompute" in str(e) and not os.path.exists(bad) and not os.path.exists(bad + ".tmp"),
           "a failed direct recompute stops the export and writes nothing")
    finally:
        e2_read._adam_cells_direct = orig
    before = os.path.getmtime(outs["adam"])
    try:
        e2_read.export_adam_inputs(root, {"adam": outs["adam"]}, every=5, log=log)
        raise AssertionError("overwrite accepted")
    except SystemExit as e:
        ok("refusing to overwrite" in str(e) and os.path.getmtime(outs["adam"]) == before, "an existing output is refused")
    # 8. CLI (RMS_WINDOW is 1000 in the subprocess: all 30 steps are inside the window)
    tpl = os.path.join(e2, "{variant}_cli.npz")
    r_ = subprocess.run([sys.executable, os.path.join(M, "e2_read.py"), "--run", root, "--export-inputs-adam", tpl,
                         "--variant", "adam,rms1000", "--every", "10"], capture_output=True, text=True)
    ca, cr = np.load(tpl.format(variant="adam")), np.load(tpl.format(variant="rms1000"))
    _, rm_full, _ = textbook(G[0], 0.9, 0.999, 1e-8, window=1000)
    ok(r_.returncode == 0 and ca["steps"].tolist() == [1, 10, 20, 30] and np.array_equal(ca["inputs"][:, 0], za["inputs"][[0, 2, 4, 6], 0])
       and rel(cr["inputs"][-1, 0, 0], -rm_full[-1]) < 1e-6 and json.loads(str(cr["meta"]))["transform"]["rms_window"] == 1000,
       "CLI --export-inputs-adam --variant adam,rms1000 (one pass, two files)")
    r_ = subprocess.run([sys.executable, os.path.join(M, "e2_read.py"), "--run", root, "--export-inputs-adam",
                         os.path.join(e2, "x.npz"), "--variant", "adam,rms1000"], capture_output=True, text=True)
    ok(r_.returncode != 0 and "{variant}" in r_.stderr and not os.path.exists(os.path.join(e2, "x.npz")),
       "two variants need {variant} in the path")
    # 9. the training-pipeline worktree's e2_generate.load_inputs reads these files unchanged
    wt = os.path.join(e2_read.IIF_BASE, "agm_data", "worktree_training_pipeline")
    if os.path.exists(os.path.join(wt, "main_run", "reader", "e2_generate.py")):
        code = ("import json, sys; from pathlib import Path; from main_run.reader import e2_generate as eg; out = {}\n"
                "for p in sys.argv[1:]:\n"
                "    inp = eg.load_inputs(Path(p))\n"
                "    out[p] = dict(windows=inp['windows'], shape=list(inp['inputs'].shape), steps=inp['steps'],\n"
                "                  name=eg.output_reader_name('scratch_6080', inp['meta']))\n"
                "print(json.dumps(out))")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r_ = subprocess.run([sys.executable, "-c", code, outs["adam"], outs["rms1000"]], cwd=wt, env=env,
                            capture_output=True, text=True)
        got = json.loads(r_.stdout.strip().splitlines()[-1]) if r_.returncode == 0 else {}
        ok(r_.returncode == 0 and got[outs["adam"]]["windows"] == ["adam"] and got[outs["rms1000"]]["windows"] == ["rms1000"]
           and got[outs["adam"]]["shape"] == [len(steps), 1, 2, D] and got[outs["adam"]]["steps"] == steps
           and got[outs["adam"]]["name"] == "adam_scratch_6080" and got[outs["rms1000"]]["name"] == "rms1000_scratch_6080",
           "worktree e2_generate.load_inputs reads both files (order hash, sign guard); outputs named <variant>_<reader>")
    else:
        print("  skip: training-pipeline worktree absent (load_inputs check)")
print(f"CPU tests passed: {len(passed)}")

"""CPU tests of the public AOs on the clean-batch documents' activations (main_run/ao_act_cleanbatch.py).

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest -q tests/test_ao_act_cleanbatch.py

Checked: the centring math (unit(unit(v) - m), m = mean of the fixed documents' unit vectors; scale invariance; the
common direction removed) and the pairwise cosines; the label-position and full-set masks; the GPU choice rule; the
seed rule (no collisions, clear of ao_cleanbatch_read's groups); the existing-value lookup on synthetic clean-batch
files (full read first, the original read filling gaps, the cross-check, the reader windows, the preference counts and
the no-vector file); the summary and its table from synthetic vectors and reads; the completeness check (missing and
duplicate rows); the overwrite refusals of every stage (before any marker is written, so a finished stage's DONE
marker survives a relaunch); the plot on a synthetic summary. With the real data present (skipped otherwise): the
documents of the dolphin student at step 500 equal the probe record (doc ids, label counts, token counts) and the
weight sources of steps 100 and 500. Outputs go to the session scratchpad (ao_act_cb_* folders), never deleted.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run"))

import ao_act_cleanbatch as aa  # noqa: E402

SCRATCH = os.environ.get("AO_ACT_CB_SCRATCH", tempfile.gettempdir())


def scratch_dir():
    os.makedirs(SCRATCH, exist_ok=True)
    return tempfile.mkdtemp(prefix="ao_act_cb_", dir=SCRATCH)


# ------------------------------------------------------------------ math

def test_centre_formula_and_invariance():
    rng = np.random.default_rng(0)
    L, D, N = 3, 16, 10
    common = rng.normal(size=(L, D)) * 10
    fixed = common[None] * rng.uniform(0.5, 2, size=(N, 1, 1)) + rng.normal(size=(N, L, D))
    raw = common[None] * 32 + rng.normal(size=(4, L, D)) * 3
    cen, m = aa.centre(raw, fixed)
    # by hand
    uf = fixed / np.linalg.norm(fixed, axis=-1, keepdims=True)
    m_ref = uf.mean(0)
    ur = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    d = ur - m_ref
    assert np.allclose(m, m_ref)
    assert np.allclose(cen, d / np.linalg.norm(d, axis=-1, keepdims=True))
    assert np.allclose(np.linalg.norm(cen, axis=-1), 1.0)
    # scale invariance: rescaling a batch or a fixed document changes nothing
    cen2, m2 = aa.centre(raw * np.array([1, 5, 0.1, 3])[:, None, None], fixed * rng.uniform(0.1, 9, (N, 1, 1)))
    assert np.allclose(cen2, cen) and np.allclose(m2, m)
    # the common direction makes raw batch vectors nearly parallel; centring removes it
    _, cos_raw = aa.pairwise_cos(raw)
    _, cos_cen = aa.pairwise_cos(cen)
    assert (cos_raw > 0.95).all() and (cos_cen < cos_raw).all()


def test_pairwise_cos_known():
    V = np.zeros((4, 1, 3))
    V[0, 0] = [1, 0, 0]
    V[1, 0] = [0, 1, 0]
    V[2, 0] = [1, 1, 0]
    V[3, 0] = [2, 0, 0]
    C, m = aa.pairwise_cos(V)
    s = 1 / np.sqrt(2)
    want = np.mean([0, s, 1, s, 0, s])  # pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    assert np.isclose(m[0], want)
    assert np.allclose(np.diag(C[0]), 1)
    with pytest.raises(ValueError):
        aa.unit(np.zeros((2, 3)))


def test_masks():
    labels = np.array([[-100, -100, 5, 6, -100], [7, -100, 8, 9, 10]])
    m = aa.label_mask(labels)
    assert m.tolist() == [[False, False, True, True, False], [False, False, True, True, True]]  # position 0 skipped
    assert m.sum() == int((labels[:, 1:] != -100).sum())  # the probe's N
    am = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert aa.full_mask(am).tolist() == [[True, True, False, False, False], [True, True, True, True, False]]


def test_choose_gpu():
    rows = [(0, 150_000), (1, 90_000), (2, 150_000), (3, 70_000)]
    assert aa.choose_gpu(rows, 60 * 1024) == (0, 150_000)  # tie -> lower index
    assert aa.choose_gpu([(0, 10), (3, 70_000)], 60 * 1024) == (3, 70_000)
    with pytest.raises(SystemExit):
        aa.choose_gpu([(0, 50_000), (1, 61_000)], 62 * 1024)
    with pytest.raises(SystemExit):
        aa.choose_gpu([], 1)


def test_seed_rule():
    seeds = [aa.seed_of(0, s, t, v, ao) for s in aa.STUDENTS for t in [2] + list(range(100, 7001, 100))
             for v in aa.VARIANTS for ao in aa.AO_ORDER]
    assert len(seeds) == len(set(seeds))
    import ao_cleanbatch_read as acr
    old = {acr.seed_of(0, s, t, g, ao) for s in acr.STUDENT_INDEX for t in aa.STEPS for g in acr.GROUPS
           for ao in acr.AO_ORDER}
    new = {aa.seed_of(0, s, t, v, ao) for s in aa.STUDENTS for t in aa.STEPS for v in aa.VARIANTS for ao in aa.AO_ORDER}
    assert not (old & new)
    assert aa.AO_ORDER == acr.AO_ORDER and dict(aa.STUDENT_INDEX) == dict(acr.STUDENT_INDEX)
    assert min(aa.GROUP_INDEX.values()) == len(acr.GROUPS)


# ------------------------------------------------------------------ synthetic files

STEPS = (100, 500, 1000)
D = 8


def write_runs(runs):
    """Synthetic clean-batch outputs under ``runs``: full AO read (steps 100, 500), original AO read (500, 1000) with
    preference counts, reader P(Yes) files, novector. Returns the values."""
    vals = {}
    for s in aa.STUDENTS:
        for ao in aa.AO_ORDER:
            for kind, steps in (("ao_cleanbatch_full", (100, 500)), ("ao_cleanbatch", (500, 1000))):
                d = os.path.join(runs, kind, s)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"{ao}_mfree_yesno.jsonl"), "w") as f:
                    for t in steps:
                        for i, b in enumerate(aa.BATCHES):
                            v = 0.5 if b == "real" else 0.1 * i
                            if kind == "ao_cleanbatch" and t == 500 and b == "f2":
                                v += 0.01  # the full and original reads differ by 0.01 here
                            vals[(kind, s, ao, t, b)] = v
                            f.write(json.dumps(dict(step=t, batch=b, window=f"{b}_mfree", p_yes=v)) + "\n")
                if kind == "ao_cleanbatch":
                    with open(os.path.join(d, f"{ao}_mfree_preference.jsonl"), "w") as f:
                        for t in steps:
                            for i, b in enumerate(aa.BATCHES):
                                f.write(json.dumps(dict(step=t, batch=b, n_dolphin=(3 if b == "real" else i))) + "\n")
        for name, steps in ((f"cleanbatch_{s}_s1_full", (100, 500, 1000)), (f"cleanbatch_{s}_s1", (500, 1000))):
            p = os.path.join(runs, name, aa.PYES_NAME)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            W = [f"{b}_{v}" for v in ("step", "mfree") for b in aa.BATCHES]
            P = [[(0.9 if w == "real_mfree" else 0.2 if w.endswith("_mfree") else 0.0) for w in W] for _ in steps]
            with open(p, "w") as f:
                json.dump(dict(steps=list(steps), windows=W, p_yes=P), f)
    os.makedirs(os.path.join(runs, "ao_cleanbatch"), exist_ok=True)
    with open(os.path.join(runs, "ao_cleanbatch", "novector.json"), "w") as f:
        json.dump(dict(aos={ao: dict(yesno=dict(p_yes=0.05 if ao == "ao3layer" else 0.6),
                                     preference=dict(n_dolphin=0, n_samples=20)) for ao in aa.AO_ORDER}), f)
    return vals


def write_vectors(out, steps=STEPS, students=aa.STUDENTS, n=4):
    rng = np.random.default_rng(1)
    os.makedirs(out, exist_ok=True)
    S = len(students)
    for t in steps:
        raw = rng.normal(size=(S, 4, 3, D)) + 5
        fixed = rng.normal(size=(S, 6, 3, D)) + 5
        V = np.zeros((S, 4, 2, 3, D), np.float32)
        for si in range(S):
            cen, _ = aa.centre(raw[si], fixed[si])
            V[si, :, 0] = raw[si]
            V[si, :, 1] = cen
        doc_checks = [dict(student=s, step=t, batch=b, check="doc_ids_equal_probe", ok=True) for s in students
                      for b in aa.BATCHES] + [dict(student=s, step=None, batch=None, check="probe_meta_complete",
                                                   ok=True) for s in students]
        vec_checks = [dict(student=s, step=t, check="vectors_finite", ok=True) for s in students]
        meta = dict(step=t, limit=None, checks=dict(documents=doc_checks, vectors=vec_checks))
        with open(aa.vec_path(out, t), "wb") as f:
            np.savez(f, vectors=V, doc_id=np.full((S, 4, n), "x", dtype="U40"), students=np.asarray(students),
                     batches=np.asarray(aa.BATCHES), variants=np.asarray(aa.VARIANTS),
                     layers=np.asarray(aa.LAYERS, np.int64), step=np.int64(t), meta=np.asarray(json.dumps(meta)))


def write_reads(out, steps=STEPS, students=aa.STUDENTS, drop=None, dup=False, read_checks=None):
    with open(os.path.join(out, "read_settings.json"), "w") as f:
        json.dump(dict(checks_passed=read_checks or {"10_pyes_path_vs_generate": "1/1",
                                                     "zero_vector_equals_no_hook": "1/1"}), f)
    rows = []
    for s in students:
        for t in steps:
            for i, b in enumerate(aa.BATCHES):
                for v in aa.VARIANTS:
                    for ao in aa.AO_ORDER:
                        py = (0.4 if v == "act_centred" else 0.2) if b == "real" else 0.05 * i
                        rows.append(dict(student=s, step=t, batch=b, variant=v, ao=ao, question="yesno", p_yes=py,
                                         p_no=1 - py, top1="No"))
                        rows.append(dict(student=s, step=t, batch=b, variant=v, ao=ao, question="preference",
                                         n_samples=20, n_dolphin=(2 if b == "real" else 0), texts=["x"] * 20))
    if drop is not None:
        rows = [r for r in rows if not (r["step"], r["batch"], r["question"]) == drop]
    if dup:
        rows.append(dict(rows[0]))
    with open(aa.reads_path(out), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


# ------------------------------------------------------------------ existing values

def test_existing_values():
    runs = os.path.join(scratch_dir(), "runs")
    vals = write_runs(runs)
    ex = aa.existing_values("dolphin", list(STEPS), runs)
    g = ex["ao_grad"]["ao3layer"]
    assert g[100]["source"] == "full" and g[500]["source"] == "full" and g[1000]["source"] == "original"
    assert g[500]["real"] == vals[("ao_cleanbatch_full", "dolphin", "ao3layer", 500, "real")]
    assert g[1000]["foreign"] == [vals[("ao_cleanbatch", "dolphin", "ao3layer", 1000, b)] for b in ("f1", "f2", "f3")]
    assert "n_dolphin_real" not in g[100] and g[500]["n_dolphin_real"] == 3 and g[1000]["n_dolphin_foreign"] == [1, 2, 3]
    cc = ex["crosscheck"]["ao_grad/ao3layer"]
    assert cc["n"] == 4 and abs(cc["max_abs"] - 0.01) < 1e-12  # step 500, four batches in both
    r = ex["reader_grad"]
    assert r[100]["source"] == "full" and r[100]["real"] == 0.9 and r[100]["foreign"] == [0.2, 0.2, 0.2]
    assert ex["crosscheck"]["reader_grad"]["max_abs"] == 0.0
    assert ex["novector"]["aopaper"]["p_yes"] == 0.6
    ex2 = aa.existing_values("dolphin", [100, 2000], runs)
    assert ex2["ao_grad"]["ao3layer"][2000] is None and ex2["reader_grad"][2000] is None


# ------------------------------------------------------------------ summary

def test_summary_and_table():
    base = scratch_dir()
    runs = os.path.join(base, "runs")
    write_runs(runs)
    out = os.path.join(base, "out")
    write_vectors(out)
    write_reads(out)
    res = aa.summarise(out, runs)
    S = res["students"]["dolphin"]
    assert S["steps"] == list(STEPS) and S["foreign_arm"] == "clean"
    e = S["per_step"]["500"]
    p = e["act_centred"]["ao3layer"]["p_yes"]
    assert p["real"] == 0.4 and np.isclose(p["foreign_mean"], 0.1) and np.isclose(p["foreign_min"], 0.05) \
        and np.isclose(p["foreign_max"], 0.15)
    assert e["act_raw"]["aopaper"]["n_dolphin"]["real"] == 2
    assert e["ao_grad_mfree"]["ao3layer"]["source"] == "full" and e["reader_grad_mfree"]["p_yes"]["real"] == 0.9
    agg = S["aggregate"]["act_centred/ao3layer"]
    assert agg["n_steps"] == 3 and agg["n_real_gt_foreign"] == 3 and np.isclose(agg["mean_diff"], 0.3)
    assert S["aggregate"]["act_raw/ao3layer/n_dolphin"] == dict(real=6, foreign_mean=0.0, of=60)
    # cosines from the stored vectors
    z = np.load(aa.vec_path(out, 500))
    _, cm = aa.pairwise_cos(z["vectors"][0, :, 1].astype(np.float64))
    assert np.allclose(e["cos"]["act_centred"]["mean_pairwise_per_layer"], cm)
    # run-level checks counted once, completeness ok
    assert res["checks"]["documents"]["probe_meta_complete"] == dict(passed=2, of=2)
    assert res["checks"]["documents"]["doc_ids_equal_probe"] == dict(passed=24, of=24)
    assert res["checks"]["reads_complete_once"]["ok"]
    assert res["checks_passed"]["read/zero_vector_equals_no_hook"] == "1/1"
    assert res["checks_passed"]["read_settings_present"] == "1/1"
    assert all(v.split("/")[0] == v.split("/")[1] for v in res["checks_passed"].values())
    tab = aa.format_table(res)
    assert "dolphin student, ao3layer" in tab and "clean student, aopaper" in tab
    assert "0.40 / 0.10" in tab and "0.90 / 0.20" in tab
    # the plot draws from the summary
    fig = os.path.join(base, "fig.png")
    plotted = aa.make_plot(res, fig)
    assert os.path.getsize(fig) > 10_000
    assert plotted["ao3layer/act_centred"]["real"] == [0.4, 0.4, 0.4]
    assert np.isnan(plotted["ao3layer/ao_grad"]["real"]).sum() == 0


def test_summary_carries_failed_read_check():
    base = scratch_dir()
    runs = os.path.join(base, "runs")
    write_runs(runs)
    out = os.path.join(base, "out")
    write_vectors(out)
    write_reads(out, read_checks={"10_pyes_path_vs_generate": "0/1"})
    res = aa.summarise(out, runs)
    assert res["checks_passed"]["read/10_pyes_path_vs_generate"] == "0/1"
    rs = os.path.join(out, "read_settings.json")
    os.replace(rs, rs + ".moved_aside")  # nothing is deleted; summarise only looks for read_settings.json
    assert aa.summarise(out, runs)["checks_passed"]["read_settings_present"] == "0/1"


def test_completeness_detects_missing_and_duplicates():
    out = scratch_dir()
    rows = write_reads(out, drop=(500, "f2", "preference"), dup=True)
    keys = aa.expected_read_keys({s: list(STEPS) for s in aa.STUDENTS})
    c = aa.read_completeness(rows, keys, 20)
    assert not c["ok"] and c["duplicates"] == 1 and c["missing"]
    assert len(keys) == 2 * 3 * 4 * 2 * 2 * 2
    good = write_reads(scratch_dir())
    assert aa.read_completeness(good, keys, 20)["ok"]
    assert not aa.read_completeness(good, keys, 10)["ok"]  # sample count must match


# ------------------------------------------------------------------ overwrite refusals

def test_refusals():
    out = scratch_dir()
    write_vectors(out, steps=(100,))
    # a finished vectors stage: its DONE marker must survive a refused relaunch
    md = os.path.join(out, "markers")
    os.makedirs(md)
    with open(os.path.join(md, "vectors.DONE"), "w") as f:
        f.write("done")
    # vectors: refuses before pinning a GPU or reading anything
    with pytest.raises(SystemExit, match="exists"):
        aa.main(["vectors", "--out", out, "--steps", "100"])
    write_reads(out, steps=(100,))
    with pytest.raises(SystemExit, match="exists"):
        aa.main(["read", "--out", out, "--steps", "100"])
    with open(os.path.join(out, "summary.json"), "w") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="exists"):
        aa.main(["summary", "--out", out])
    fig = os.path.join(out, "fig.png")
    with open(fig, "w") as f:
        f.write("x")
    with pytest.raises(SystemExit, match="exists"):
        aa.main(["plot", "--out", out, "--fig", fig])
    # every refusal happened before any marker was written and changed no output
    for st in ("vectors", "read", "summary", "plot"):
        for k in ("START", "FAILED"):
            assert not os.path.exists(os.path.join(md, f"{st}.{k}"))
    assert open(os.path.join(md, "vectors.DONE")).read() == "done"
    assert sorted(os.listdir(md)) == ["vectors.DONE"]
    assert open(fig).read() == "x" and open(os.path.join(out, "summary.json")).read() == "{}"
    # the refused plot did not create its .json either
    assert not os.path.exists(os.path.join(out, "fig.json"))


def test_load_vectors_refuses_non_unit():
    out = scratch_dir()
    write_vectors(out, steps=(100,))
    z = dict(np.load(aa.vec_path(out, 100)))
    z["vectors"][0, 0, 1] *= 1.1
    with open(aa.vec_path(out, 100), "wb") as f:
        np.savez(f, **z)
    with pytest.raises(SystemExit, match="unit"):
        aa.load_vectors(out)


# ------------------------------------------------------------------ real data (skipped when absent)

REAL = os.path.exists(aa.probe_meta_path("dolphin")) and os.path.exists(os.path.join(aa.run_dir("dolphin"), "RUN.json"))


@pytest.mark.skipif(not REAL, reason="clean-batch extension run not present")
def test_real_documents_step500():
    st = aa.run_settings("dolphin")
    try:
        tok = aa.load_tokenizer(st["model"])
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer not available offline: {e}")
    docs, checks = aa.build_documents("dolphin", [500], tok)
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
    names = {c["check"] for c in checks}
    assert {"doc_ids_equal_probe", "label_sum_equals_probe_N", "token_sum_equals_probe"} <= names
    rec = aa.load_probe_records("dolphin")[0][500]
    assert docs[500]["real"]["doc_id"] == rec["real_doc_id"]
    assert [len(docs[500][b]["doc_id"]) for b in aa.BATCHES] == [32] * 4
    assert aa.label_text_example(tok, docs[500]["real"]["batch"]).startswith("<think>\n\n</think>\n\n")
    src = aa.weight_source(aa.run_dir("dolphin"), 100)
    assert src["kind"] == "lora_pairs_before" and src["pair_step"] == 101
    assert aa.weight_source(aa.run_dir("dolphin"), 500)["kind"] == "adapter"
    assert aa.weight_source(aa.run_dir("dolphin"), 150) is None


# ------------------------------------------------------------------ the forward hook on a tiny model (CPU)

def test_forward_sums_tiny_model():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen3Config

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").float().eval()
    pm = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["o_proj", "down_proj"], lora_dropout=0.0))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.1)  # a non-zero adapter
    pm.eval()
    ids = torch.randint(0, 50, (3, 9))
    am = torch.ones_like(ids)
    am[1, 6:] = 0
    am[2, 4:] = 0
    labels = ids.clone()
    labels[:, :3] = -100
    labels[am == 0] = -100
    batch = dict(input_ids=ids.numpy(), attention_mask=am.numpy(), labels=labels.numpy())
    layers = (1, 2)
    lab, full = aa.forward_sums(pm, batch, layers, torch.device("cpu"), full=True)
    with torch.no_grad():
        hs = pm(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
    lm = torch.as_tensor(aa.label_mask(batch["labels"]), dtype=torch.float64)
    fm = torch.as_tensor(aa.full_mask(batch["attention_mask"]), dtype=torch.float64)
    for li, L in enumerate(layers):
        h = hs[L + 1].double()  # hidden_states[L + 1] = output of decoder layer L (L < last layer)
        assert np.allclose(lab[:, li], torch.einsum("bt,btd->bd", lm, h).numpy(), rtol=1e-6, atol=1e-8)
        assert np.allclose(full[:, li], torch.einsum("bt,btd->bd", fm, h).numpy(), rtol=1e-6, atol=1e-8)
    assert lab.shape == (3, 2, 32) and lab.dtype == np.float64
    # document 2 has labels only at position 3 (padding from 4): its label sum is that one position's output
    assert np.allclose(lab[2, 0], hs[2][2, 3].double().numpy(), rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------------ the read stage end to end with a mocked AO (CPU)

def _coded_vectors(out, steps, students):
    """act_vectors files whose every (student, step, batch, variant, layer) vector encodes a code c in [0, 1) as
    v[0] / ||v[:2]|| and the layer as v[2] = 0.01 (layer index + 1). Returns {(s, t, b, v): c}."""
    os.makedirs(out, exist_ok=True)
    codes, k = {}, 0
    for t in steps:
        V = np.zeros((len(students), 4, 2, 3, D), np.float64)
        for si, s in enumerate(students):
            for bi, b in enumerate(aa.BATCHES):
                for vi, v in enumerate(aa.VARIANTS):
                    k += 1
                    c = k / 1000
                    codes[(s, t, b, v)] = c
                    for li in range(3):
                        mk = 0.01 * (li + 1)
                        x = np.zeros(D)
                        r = np.sqrt(1 - mk ** 2)
                        x[0], x[1], x[2] = c * r, np.sqrt(1 - c ** 2) * r, mk
                        V[si, bi, vi, li] = x * (1.0 if v == "act_centred" else 37.0)
        meta = dict(step=t, limit=None, checks=dict(documents=[], vectors=[]))
        with open(aa.vec_path(out, t), "wb") as f:
            np.savez(f, vectors=V.astype(np.float32), doc_id=np.full((len(students), 4, 2), "x", dtype="U64"),
                     students=np.asarray(students), batches=np.asarray(aa.BATCHES), variants=np.asarray(aa.VARIANTS),
                     layers=np.asarray(aa.LAYERS, np.int64), step=np.int64(t), meta=np.asarray(json.dumps(meta)))
    return codes


def _decode(v):
    return float(v[0] / np.linalg.norm(v[:2]))


def test_read_stage_rows_match_the_vectors_fed(monkeypatch):
    import torch
    import ao_cleanbatch_read as acr
    import ao_diff_read as ad

    out = scratch_dir()
    steps, students = (100, 500), ("dolphin", "clean")
    codes = _coded_vectors(out, steps, students)
    fed = {"layers": []}

    class FakePM:
        adapter = None

        def set_adapter(self, name):
            FakePM.adapter = name

    class FakeTok:
        def decode(self, ids, **kw):
            return f"tok{ids[0]}"

    def first_token_probs(pm, ids, pos, V, device):
        if V is None or not np.any(V):
            return torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]] * (1 if V is None else len(V)))
        V = np.asarray(V, np.float64)
        fed["layers"].append((FakePM.adapter, [round(float(x[2] / np.linalg.norm(x)), 6) for x in V[0]]))
        c = np.array([_decode(x[0]) for x in V])
        return torch.tensor(np.stack([c, 1 - c] + [0 * c] * 4, 1), dtype=torch.float32)  # 6 tokens (top 5)

    def sample_call(pm, tok, ids, pos, V, seed, device):
        fed.setdefault("seeds", []).append((FakePM.adapter, seed))
        return [f"{_decode(x[0]):.6f}" for x in np.asarray(V, np.float64)], [0] * len(V)

    monkeypatch.setattr(aa, "pin_gpu", lambda spec, g: dict(gpu=0, mocked=True))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(ad, "load_aos", lambda device, aos: (FakePM(), FakeTok(), {a: {} for a in aos}))
    monkeypatch.setattr(ad, "oracle_prompt_ids", lambda tok, layers, n, q: ([1, 2, 3], [1], f"{len(layers)}:{q}"))
    monkeypatch.setattr(acr, "tokenizer_checks", lambda tok, st, po: ({"Yes": 0}, {"No": 1}, [dict(ok=True)],
                                                                       [dict(ok=True)], [dict(ok=True)]))
    monkeypatch.setattr(acr, "check_pyes_vs_generate", lambda *a: 0.0)
    monkeypatch.setattr(acr, "first_token_probs", first_token_probs)
    monkeypatch.setattr(acr, "sample_call", sample_call)
    assert aa.main(["read", "--out", out, "--steps", "100,500", "--limit", "3"]) == 0
    rows = [json.loads(x) for x in open(aa.reads_path(out))]
    assert len(rows) == 2 * 2 * 4 * 2 * 2 * 2
    for r in rows:
        c = codes[(r["student"], r["step"], r["batch"], r["variant"])]
        if r["question"] == "yesno":
            assert abs(r["p_yes"] - c) < 1e-6 and abs(r["p_no"] - (1 - c)) < 1e-6
        else:
            assert r["n_samples"] == 3 and r["texts"] == [f"{c:.6f}"] * 3
            assert r["seed"] == aa.seed_of(0, r["student"], r["step"], r["variant"], r["ao"])
    # the paper AO got layer 20 only, the three-layer AO layers 10, 20, 30
    for ao, marks in fed["layers"]:
        assert marks == ([0.02] if ao == "aopaper" else [0.01, 0.02, 0.03]), (ao, marks)
    st = json.load(open(os.path.join(out, "read_settings.json")))
    assert all(v.split("/")[0] == v.split("/")[1] for v in st["checks_passed"].values()), st["checks_passed"]
    assert os.path.exists(os.path.join(out, "markers", "read.DONE"))

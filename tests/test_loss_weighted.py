"""CPU tests for main_run/experiment2/loss_weighted.py (convert + shuffle + summarise)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "main_run" / "experiment2"))
import common as C  # noqa: E402
import loss_weighted as LW  # noqa: E402


D = 32  # narrow width for the synthetic mean/vectors; production code expects 5120


def _patch_width(monkeypatch, width=D):
    """Use a narrow synthetic width so CPU tests stay tiny."""
    monkeypatch.setattr(LW, "LAYERS", (10, 20, 30))

    def fake_resolve_mean(mean, template, *, fetch_twin=False, twin_cache=None):
        if mean is not None:
            path = Path(mean)
        elif template is not None:
            path = Path(template) / "mean_unit_tokensum.npy"
        else:
            raise ValueError("need mean or template in test")
        assert np.load(path).shape[0] == 3
        return path

    monkeypatch.setattr(LW, "resolve_mean", fake_resolve_mean)

    def fake_load_vec_pair(vec, variant, layer):
        u = np.load(Path(vec) / f"{variant}_L{layer}.npy", mmap_mode="r")
        n = np.load(Path(vec) / f"{variant}_L{layer}_norm.npy")
        if u.ndim != 2 or u.shape[1] != width or len(n) != len(u):
            raise ValueError(f"{variant} L{layer}: bad shapes {u.shape} / {len(n)}")
        return u, n.astype(np.float32)

    monkeypatch.setattr(LW, "load_vec_pair", fake_load_vec_pair)


def _make_fixture(tmp: Path, n_all=12, n_held=6, n_docs=3, width=D):
    rng = np.random.default_rng(0)
    families = ["rare", "quirk", "backdoor"] * (n_held // 3 + 1)
    rows = []
    beh = []
    for i in range(n_all):
        split = C.SPLIT if i < n_held else "train"
        bid = f"b{i:03d}"
        fam = families[i % len(families)]
        rows.append(dict(bundle_id=bid, split=split, family=fam, unit_id=f"u{i // 2}",
                         n_shows=1 if i % 2 == 0 else 0, n_docs=n_docs, n_label=10, n_tokens=20,
                         behaviour_id=f"beh{i}", behaviour_text="x", trigger="", evidence="full",
                         pair_id="", n_trigger=0, n_store_tokens=20, n_store_label=10))
        docs = [f"{bid}_d{j}" for j in range(n_docs)]
        beh.append(dict(bundle_id=bid, doc_ids=docs, split=split, family=fam))
    bundles = pd.DataFrame(rows)
    # Put held-out rows at non-contiguous indices like the real table.
    order = list(range(n_held, n_all)) + list(range(n_held))
    bundles = bundles.iloc[order].reset_index(drop=True)

    vec = tmp / "vec"
    vec.mkdir(exist_ok=True)
    for variant in LW.BASE_VARIANTS:
        for layer in (10, 20, 30):
            raw = rng.normal(size=(n_all, width)).astype(np.float32)
            norms = 10 + rng.random(n_all).astype(np.float32)
            if variant == "actrespl":
                for i, r in bundles.iterrows():
                    if r.split == C.SPLIT and r.n_shows > 0:
                        raw[i, 0] += 5
            unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            np.save(vec / f"{variant}_L{layer}.npy", unit.astype(np.float16))
            np.save(vec / f"{variant}_L{layer}_norm.npy", norms)

    bundles_path = tmp / "bundles.parquet"
    bundles.to_parquet(bundles_path, index=False)
    beh_path = tmp / "bundles.jsonl"
    with beh_path.open("w") as f:
        by_id = {r["bundle_id"]: r for r in beh}
        for bid in bundles.bundle_id:
            f.write(json.dumps(by_id[bid]) + "\n")

    mean = rng.normal(size=(3, width)).astype(np.float32) * 0.01
    mean_path = tmp / "mean_unit_tokensum.npy"
    np.save(mean_path, mean)
    return dict(vec=vec, bundles=bundles_path, beh=beh_path, mean=mean_path, n_all=n_all, n_held=n_held,
                width=width, bundles_df=bundles)


def test_held_frame_order(tmp_path, monkeypatch):
    fx = _make_fixture(tmp_path)
    monkeypatch.setattr(LW, "N_HELD", fx["n_held"])
    held = LW.held_frame(fx["bundles"])
    assert len(held) == fx["n_held"]
    assert set(held.split) == {C.SPLIT}
    full = pd.read_parquet(fx["bundles"])
    assert (full.loc[held.vec_row, "bundle_id"].to_numpy() == held.bundle_id.to_numpy()).all()


def test_build_dirs(tmp_path, monkeypatch):
    fx = _make_fixture(tmp_path)
    _patch_width(monkeypatch, fx["width"])
    monkeypatch.setattr(LW, "N_HELD", fx["n_held"])
    monkeypatch.setattr(C, "N_HELD_BEHAVIOUR", fx["n_held"])
    out = tmp_path / "dirs"
    rc = LW.cmd_build_dirs(type("A", (), dict(
        vec=fx["vec"], bundles=fx["bundles"], beh_jsonl=fx["beh"], mean=fx["mean"], template=None,
        fetch_twin_mean=False, out=out, variants="act,actrespl,actrespn", overwrite=False))())
    assert rc == 0
    for v in ("act", "actrespl", "actrespn"):
        d = out / v
        assert (d / "meta.json").exists()
        meta = json.loads((d / "meta.json").read_text())
        assert meta["mean_sha256"] == LW.sha256_file(fx["mean"])
        ix = pd.read_parquet(d / "bundle_index.parquet")
        assert len(ix) == fx["n_held"]
        assert set(ix.columns) >= {"family", "unit_id", "n_shows", "norm_tokensum_L10"}
        rows = C.read_jsonl(d / "beh_eval_samples.jsonl")
        assert len(rows) == fx["n_held"]
        assert [r["vec_row"] for r in rows] == list(range(fx["n_held"]))
        u = np.load(d / "bundle_tokensum_L10.npy")
        assert u.shape == (fx["n_held"], fx["width"])
        assert u.dtype == np.float16


def test_build_dirs_mean_sha_mismatch_refuses_skip(tmp_path, monkeypatch):
    fx = _make_fixture(tmp_path)
    _patch_width(monkeypatch, fx["width"])
    monkeypatch.setattr(LW, "N_HELD", fx["n_held"])
    out = tmp_path / "dirs"
    args = type("A", (), dict(
        vec=fx["vec"], bundles=fx["bundles"], beh_jsonl=fx["beh"], mean=fx["mean"], template=None,
        fetch_twin_mean=False, out=out, variants="act", overwrite=False))()
    assert LW.cmd_build_dirs(args) == 0
    other = tmp_path / "other_mean.npy"
    np.save(other, np.load(fx["mean"]) + 0.1)
    with pytest.raises(ValueError, match="mean_sha256"):
        LW.cmd_build_dirs(type("A", (), dict(
            vec=fx["vec"], bundles=fx["bundles"], beh_jsonl=fx["beh"], mean=other, template=None,
            fetch_twin_mean=False, out=out, variants="act", overwrite=False))())


def test_bundle_weighted_sum_and_shared_perms():
    units = np.eye(4, 8, dtype=np.float32)
    weights = np.array([1, 2, 3, 4], np.float32)
    starts = np.array([0, 2], np.int64)
    sizes = np.array([2, 2], np.int64)
    sums = LW._bundle_weighted_sum(units, weights, starts, sizes)
    assert sums.shape == (2, 8)
    np.testing.assert_allclose(sums[0, :2], [1, 2])
    perms = LW._bundle_perms(sizes, seed=0)
    assert len(perms) == 2
    assert sorted(perms[0].tolist()) == [0, 1]
    assert sorted(perms[1].tolist()) == [0, 1]
    # Same seed → same perms (reused across layers / variants).
    assert all(np.array_equal(a, b) for a, b in zip(perms, LW._bundle_perms(sizes, seed=0)))


def test_summarise_primary_gaps_and_smoke(tmp_path, monkeypatch):
    import analyse as An

    monkeypatch.setattr(LW, "SANITY_ACT_AUC", 0.75)
    monkeypatch.setattr(LW, "SANITY_TOL", 0.05)
    _cluster = C.cluster_auc
    _paired = An.paired_auc_gap
    monkeypatch.setattr(C, "cluster_auc",
                        lambda y, score, units, *, seed=0, n_boot=2000: _cluster(y, score, units, seed=seed, n_boot=8))
    monkeypatch.setattr(An, "paired_auc_gap",
                        lambda a, b, **kw: _paired(a, b, **{**kw, "n_boot": 8}))

    run = tmp_path / "run"
    rng = np.random.default_rng(1)
    # Enough rare bundles for a rare-only gap; mix families for pooled.
    families = (["rare"] * 18) + (["quirk"] * 12) + (["backdoor"] * 12)
    cells = {name: [] for name in ("act", "actresp", "actrespl", "actrespn")}
    for i, fam in enumerate(families):
        show = i % 3 != 0
        unit = f"u{i // 3}"
        base = dict(bundle_id=f"b{i}", family=fam, unit_id=unit, n_shows=int(show),
                    p_no=0.0, reader="activation", question=C.SHOW_QUESTION)
        # act ≈ actresp (so response-only guard stays quiet); actrespl separates; actrespn middling.
        p_act = float(np.clip(0.55 if show else 0.45 + rng.normal(0, 0.05), 0, 1))
        p_resp = float(np.clip(p_act + rng.normal(0, 0.02), 0, 1))
        p_loss = 0.95 if show else 0.05
        p_n = float(np.clip(0.7 if show else 0.3 + rng.normal(0, 0.05), 0, 1))
        cells["act"].append(dict(base, p_yes=p_act, input="act"))
        cells["actresp"].append(dict(base, p_yes=p_resp, input="actresp"))
        cells["actrespl"].append(dict(base, p_yes=p_loss, input="actrespl"))
        cells["actrespn"].append(dict(base, p_yes=p_n, input="actrespn"))
    for name, rows in cells.items():
        # Smoke path only — summarise must find smoke_N when --limit-bundles is set.
        d = run / "pyes" / f"activation__{name}" / "smoke_8"
        d.mkdir(parents=True)
        C.write_jsonl(d / "scores.jsonl", rows)
    monkeypatch.setattr(C, "N_HELD_BEHAVIOUR", len(families))
    rc = LW.cmd_summarise(type("A", (), dict(run_dir=run, out=None, require_act=True, limit_bundles=8))())
    assert rc == 0
    report = json.loads((run / "loss_weighted_analysis.json").read_text())
    assert set(report["variants"]) >= {"act", "actresp", "actrespl", "actrespn"}
    assert "actrespl_minus_actresp" in report["paired_primary"]
    assert "actrespl_minus_actrespn" in report["paired_primary"]
    assert "actrespl_minus_actresp" in report["paired_primary_rare"]
    assert report["paired_primary"]["actrespl_minus_actresp"]["gap"] > 0
    assert (run / "loss_weighted_analysis.md").exists()


def test_cmd_pyes_sets_and_restores_c_run(tmp_path, monkeypatch):
    prev = C.RUN
    run_dir = tmp_path / "lw_run"
    run_dir.mkdir()
    dirs = tmp_path / "dirs"
    (dirs / "act").mkdir(parents=True)
    (dirs / "act" / "meta.json").write_text("{}")
    seen = {}

    def fake_main(argv):
        seen["run"] = C.RUN
        seen["env"] = __import__("os").environ.get("E2_RUN")
        return 0

    monkeypatch.setattr(LW, "resolve_adapter", lambda *a, **k: (tmp_path / "ad", tmp_path / "RUN.json"))
    import runtime as R
    monkeypatch.setattr(R, "main", fake_main)
    # Import path used inside cmd_pyes is `import runtime as R` relative to experiment2.
    import importlib
    rt = importlib.import_module("runtime")
    monkeypatch.setattr(rt, "main", fake_main)

    rc = LW.cmd_pyes(type("A", (), dict(
        dirs_root=dirs, variants="act", adapter=None, run_json=None, run_dir=run_dir,
        gpu="0", batch_size=None, limit_bundles=None, overwrite=False))())
    assert rc == 0
    assert seen["run"] == run_dir.resolve()
    assert seen["env"] == str(run_dir.resolve())
    assert C.RUN == prev


def test_available_variants(tmp_path):
    fx = _make_fixture(tmp_path)
    got = LW.available_variants(fx["vec"])
    assert set(LW.BASE_VARIANTS).issubset(got)
    assert "actrespl_shuf" not in got
    assert "actrespn" in got

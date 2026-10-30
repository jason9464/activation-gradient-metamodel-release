"""Base-model loss alone as a presence detector on the 530 held-out behaviour bundles (CPU only).

Experiment 2 asks whether a held-out bundle shows its behaviour (n_shows > 0) and scores the
readers' P(Yes) by AUROC with an organism bootstrap. Here the score is the bundle's loss under
the base model (Qwen3-14B; Experiment 1's forward extraction, vec/bundle_loss.parquet):

  loss_mean   mean loss per label token, higher = shows (no fitting)
  loss_sum    summed label-token loss, higher = shows (no fitting; carries length)
  n_label     label-token count, higher = shows (length control, no fitting)
  loss_resid  log mean loss minus its least-squares prediction from log tokens and log label tokens,
              fit per family on that family's training bundles without labels (surprise beyond
              length); families absent from training use the all-family model
  loss_resid_global  the same with one model over all training bundles (descriptive; base loss
              differs by family, so this removes length poorly within a family)
  fit_loss    logistic regression on log tokens, log label tokens, log summed loss and log mean
              loss (Experiment 1's loss-scalar baseline), fit on the 4,010 training bundles
  fit_length  the same with the two length columns only

Each is compared, paired by bundle, with Experiment 2's gradient and activation reader P(Yes)
and its two linear probes. Descriptive extras: AUROC within organisms (pairs of showing and
non-showing bundles from the same organism only, which removes differences in base loss between
organisms), the Spearman correlation of each reader's P(Yes) with mean loss, and AUROC after
regressing each score (reader P(Yes) as a logit) on log tokens and log label tokens within the
group's held-out bundles, without labels (how much of each score is bundle length).

  python main_run/experiment2/loss_detector.py --fwdb artifacts/fwdb/data \\
      --run-dir artifacts/experiment2_out/runs/experiment2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from analyse import paired_auc_gap  # noqa: E402

GROUPS = {"pooled_rare_quirk_backdoor": list(C.FAMILIES), **{f: [f] for f in C.FAMILIES},
          "new_behaviour_trained_family": ["rare", "quirk", "backdoor", "rollouts"]}
FITTED = {"fit_length": ["log_tokens", "log_label"],
          "fit_loss": ["log_tokens", "log_label", "log_loss_sum", "log_loss_mean"]}
REFERENCE = {"gradient reader": ("pyes", "gradient__gradient"), "activation reader": ("pyes", "activation__activation"),
             "gradient probe": ("probe", "gradient"), "activation probe": ("probe", "activation")}


def bundle_table(fwdb: Path) -> pd.DataFrame:
    b = pd.read_parquet(fwdb / "bundles.parquet")
    bl = pd.read_parquet(fwdb / "vec" / "bundle_loss.parquet")
    if not (b.bundle_id.to_numpy() == bl.bundle_id.to_numpy()).all():
        raise ValueError("bundle_loss.parquet is not in bundles.parquet order")
    if not (b.n_store_label.to_numpy() == bl.n_loss.to_numpy()).all():
        raise ValueError("loss token counts differ from the stored label counts")
    d = b[["bundle_id", "split", "family", "unit_id", "n_shows", "n_store_tokens", "n_store_label"]].copy()
    d["loss_sum"], d["loss_mean"] = bl.loss_sum.to_numpy(float), bl.loss_mean.to_numpy(float)
    d["n_label"] = d.n_store_label.astype(float)
    d["log_tokens"], d["log_label"] = np.log(d.n_store_tokens.astype(float)), np.log(d.n_label)
    d["log_loss_sum"], d["log_loss_mean"] = np.log(d.loss_sum), np.log(d.loss_mean)
    d["unit_id"] = d.unit_id.fillna(d.bundle_id)
    if (d.split == C.SPLIT).sum() != C.N_HELD_BEHAVIOUR:
        raise ValueError(f"expected {C.N_HELD_BEHAVIOUR} held-out bundles")
    return d


def length_residual(d: pd.DataFrame, *, per_family: bool) -> dict:
    """log mean loss after removing what length predicts; the regression never sees n_shows."""
    train, held = d[d.split == "train"], d[d.split == C.SPLIT]
    design = lambda x: np.column_stack([np.ones(len(x)), x.log_tokens, x.log_label])  # noqa: E731
    fit = lambda x: np.linalg.lstsq(design(x), x.log_loss_mean.to_numpy(), rcond=None)[0]  # noqa: E731
    pooled = fit(train)
    out = {}
    for family, h in held.groupby("family"):
        t = train[train.family == family]
        beta = fit(t) if per_family and len(t) else pooled
        out.update(zip(h.bundle_id, h.log_loss_mean.to_numpy() - design(h) @ beta))
    return out


def fitted_scores(d: pd.DataFrame) -> dict:
    """Balanced logistic regression on standardised columns; C = 1 fixed before looking at held-out scores."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, held = d[d.split == "train"], d[d.split == C.SPLIT]
    out = {"loss_resid": length_residual(d, per_family=True),
           "loss_resid_global": length_residual(d, per_family=False)}
    for name, cols in FITTED.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
        model.fit(train[cols].to_numpy(), (train.n_shows > 0).to_numpy())
        out[name] = dict(zip(held.bundle_id, model.predict_proba(held[cols].to_numpy())[:, 1]))
    return out


def as_rows(held: pd.DataFrame, score: dict) -> list[dict]:
    return [dict(bundle_id=r.bundle_id, family=r.family, unit_id=r.unit_id, n_shows=int(r.n_shows),
                 p_yes=float(score[r.bundle_id])) for r in held.itertuples()]


def reference_rows(run_dir: Path) -> dict:
    out = {}
    for name, (kind, key) in REFERENCE.items():
        path = run_dir / kind / (f"{key}/scores.jsonl" if kind == "pyes" else f"{key}.jsonl")
        if path.exists():
            rows = C.read_jsonl(path)
            out[name] = [dict(r, p_yes=r["p_yes"] if kind == "pyes" else r["score"]) for r in rows]
    return out


def within_auc(rows: list[dict], *, seed: int = 0, n_boot: int = 2000) -> dict:
    """Share of (showing, non-showing) pairs from the same organism ordered correctly; organism bootstrap."""
    by = {}
    for r in rows:
        by.setdefault(r["unit_id"], []).append(r)
    stats = {}
    for u, group in by.items():
        pos = np.asarray([r["p_yes"] for r in group if r["n_shows"] > 0])
        neg = np.asarray([r["p_yes"] for r in group if r["n_shows"] == 0])
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            stats[u] = (float((diff > 0).sum() + 0.5 * (diff == 0).sum()), diff.size)
    if not stats:
        return dict(auc=None, lo=None, hi=None, n_organisms=0, n_pairs=0)
    units = sorted(stats)
    num, den = np.asarray([stats[u][0] for u in units]), np.asarray([stats[u][1] for u in units])
    rng = np.random.default_rng(seed)
    boots = [num[i].sum() / den[i].sum() for i in (rng.integers(len(units), size=len(units)) for _ in range(n_boot))]
    return dict(auc=float(num.sum() / den.sum()), lo=float(np.quantile(boots, .025)),
                hi=float(np.quantile(boots, .975)), n_organisms=len(units), n_pairs=int(den.sum()))


def length_residual_auc(rows: list[dict], held: pd.DataFrame, logit: bool) -> float | None:
    h = held.set_index("bundle_id").loc[[r["bundle_id"] for r in rows]]
    x = np.asarray([r["p_yes"] for r in rows], float)
    if logit:
        x = np.log(np.clip(x, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - x, 1e-6, 1))
    design = np.column_stack([np.ones(len(h)), h.log_tokens, h.log_label])
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    return C.auc([r["n_shows"] > 0 for r in rows], x - design @ beta)


def summarise(scores: dict, refs: dict, held: pd.DataFrame) -> dict:
    out = dict(auc={}, within_organism_auc={}, gaps={}, spearman_with_loss_mean={}, length_residual_auc={})
    for name, rows in {**scores, **refs}.items():
        out["auc"][name] = {g: C.cluster_auc([r["n_shows"] > 0 for r in rows if r["family"] in fam],
                                             [r["p_yes"] for r in rows if r["family"] in fam],
                                             [r["unit_id"] for r in rows if r["family"] in fam])
                            for g, fam in GROUPS.items()}
        out["within_organism_auc"][name] = {g: within_auc([r for r in rows if r["family"] in fam])
                                            for g, fam in GROUPS.items()}
        out["length_residual_auc"][name] = {g: length_residual_auc([r for r in rows if r["family"] in fam], held,
                                                                   logit=name.endswith("reader"))
                                            for g, fam in GROUPS.items()}
    for ref, ref_rows in refs.items():
        for name in ("loss_mean", "loss_resid", "fit_loss"):
            out["gaps"][f"{ref} - {name}"] = {g: paired_auc_gap(ref_rows, scores[name], families=tuple(fam))
                                              for g, fam in GROUPS.items()}
    from scipy.stats import spearmanr

    loss = {r["bundle_id"]: r["p_yes"] for r in scores["loss_mean"]}
    for ref, ref_rows in refs.items():
        out["spearman_with_loss_mean"][ref] = {}
        for g, fam in GROUPS.items():
            rs = [r for r in ref_rows if r["family"] in fam]
            rho = spearmanr([r["p_yes"] for r in rs], [loss[r["bundle_id"]] for r in rs]).statistic
            out["spearman_with_loss_mean"][ref][g] = float(rho)
    return out


def fmt(s: dict | None, key: str = "auc") -> str:
    if not s or s.get(key) is None:
        return "–"
    if s.get("lo") is None:
        return f"{s[key]:.3f}"
    return f"{s[key]:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"


def markdown(res: dict) -> str:
    g = list(GROUPS)
    head = "| score | " + " | ".join(g) + " |\n|---|" + "---|" * len(g) + "\n"
    lines = ["# Base-model loss as a presence detector (held-out behaviour bundles)", "",
             "Target `n_shows > 0`; AUROC with 95% organism-bootstrap intervals (2,000 resamples).", "",
             "## AUROC", "", head.rstrip("\n")]
    for name, row in res["auc"].items():
        lines.append(f"| {name} | " + " | ".join(fmt(row[k]) for k in g) + " |")
    lines += ["", "## AUROC within organisms (descriptive)", "", head.rstrip("\n")]
    for name, row in res["within_organism_auc"].items():
        lines.append(f"| {name} | " + " | ".join(
            fmt(row[k]) + (f" ({row[k]['n_organisms']} org.)" if row[k]["n_organisms"] else "") for k in g) + " |")
    lines += ["", "## Paired gaps (reference − loss score)", "", "| contrast | " + " | ".join(g) + " |",
              "|---|" + "---|" * len(g)]
    for name, row in res["gaps"].items():
        lines.append(f"| {name} | " + " | ".join(fmt(row[k], "gap") for k in g) + " |")
    lines += ["", "## AUROC after regressing out bundle length within the group (descriptive)", "", head.rstrip("\n")]
    for name, row in res["length_residual_auc"].items():
        lines.append(f"| {name} | " + " | ".join("–" if row[k] is None else f"{row[k]:.3f}" for k in g) + " |")
    lines += ["", "## Spearman of the reference score with mean loss (descriptive)", "",
              "| reference | " + " | ".join(g) + " |", "|---|" + "---|" * len(g)]
    for name, row in res["spearman_with_loss_mean"].items():
        lines.append(f"| {name} | " + " | ".join(f"{row[k]:+.2f}" for k in g) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fwdb", type=Path, required=True, help="Experiment 1 data dir (bundles.parquet, vec/)")
    ap.add_argument("--run-dir", type=Path, default=C.RUN, help="Experiment 2 run dir (pyes/, probe/)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    d = bundle_table(a.fwdb)
    held = d[d.split == C.SPLIT]
    scores = {name: as_rows(held, dict(zip(held.bundle_id, held[name]))) for name in ("loss_mean", "loss_sum", "n_label")}
    scores.update({name: as_rows(held, s) for name, s in fitted_scores(d).items()})
    refs = reference_rows(a.run_dir)
    res = summarise(scores, refs, held)
    res.update(target="n_shows > 0", split=C.SPLIT, n_train=int((d.split == "train").sum()),
               loss_source=str(a.fwdb / "vec" / "bundle_loss.parquet"), base_model="Qwen/Qwen3-14B",
               fitted_rule="StandardScaler + LogisticRegression(C=1, class_weight=balanced), training split",
               direction="higher score = shows; no sign flipping")
    out = a.out or a.run_dir / "loss_detector"
    C.write_json(out / "summary.json", res)
    C.write_jsonl(out / "scores.jsonl", [dict(r, score=name) for name, rows in scores.items() for r in rows])
    (out / "summary.md").write_text(markdown(res), encoding="utf-8")
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

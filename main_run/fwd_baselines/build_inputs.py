#!/usr/bin/env python
"""Bundle vectors of the forward-only inputs, their checks, and the 40-layer reader directories of the 530 held-out
bundles (Experiment 1). CPU only.

  python main_run/fwd_baselines/build_inputs.py fetch     [--out D]           # behaviour files, sha256 checked
  python main_run/fwd_baselines/build_inputs.py doclist   [--out D]           # held-out doc ids for extract_fwd
  python main_run/fwd_baselines/build_inputs.py vechash   [--out D]           # existing act/grad/actw vs ref/vec_reference.json
  python main_run/fwd_baselines/build_inputs.py check     [--out D] [--ext E] # extraction vs the behaviour store
  python main_run/fwd_baselines/build_inputs.py vec       [--out D] [--ext E] # bundle vectors + bundle-level checks
  python main_run/fwd_baselines/build_inputs.py readerdirs --template ACT_DIR40 [--out D] [--ext E]

--out is probe_compare's output directory (bundles.parquet, vec/, raw/; env PC_OUT), --ext the extraction directory
(default <out>/ext) holding extract_fwd.py's two datasets: fwd_all (every document, act layers 5..38 of the store) and
fwd_heldout (the 6,335 documents of the 530 beh_test bundles, all 40 layers).

vec writes, per layer L of the store's 8 activation layers, with probe_compare.save_vec (fp16 unit rows, fp32 norms,
rows in bundles.parquet order):
  actfull_L   sum_d a_full_d (unit x norm): the same quantity as probe_compare's act; built only for the check
  actresp_L   sum_d a_resp_d: the response-only activation sum (label positions plen-1 .. n-2)
  actrespl_L  sum_d loss_d * a_resp_d / |a_resp_d|, loss_d = the document's summed token cross-entropy
  actrespn_L  sum_d n_label_d * a_resp_d / |a_resp_d|, a matched label-count control for actrespl
  actrespm_L  the same with loss_d / n_label_d (mean per label token; robustness)
  actrespg_L  sum_d |g_d,L| * a_resp_d / |a_resp_d|, |g_d,L| = the store's full gradient norm of d at layer L (as actw)
  oerr_L40    sum_d oerr_d, the output-error vector (L40 = the final-norm output space, not a decoder layer)
  gradh_L39   -sum_d g_d,39 from extract_fwd's head backward; built only for the check against grad_L39
and bundle_loss.parquet (per bundle: loss_sum, n_loss, loss_mean).

The checks (plan section "Checks") stop the pipeline with exit code 1 when a gate fails; --no-gate records the
failure and continues (then report it with every result).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ao_vs_reader"))
sys.path.insert(0, str(HERE.parent))
import probe_compare as P  # noqa: E402
from agm import store  # noqa: E402

LAYERS = P.LAYERS
OERR_L = 40
LAST = 39
HELDOUT_SPLIT = "beh_test"
N_HELDOUT = 530  # beh_test bundles
READER_LAYERS = 40  # the 40-layer reader
EXT_ALL, EXT_HELD = "fwd_all", "fwd_heldout"
REF = HERE / "ref"
ACT_SIGN = "+h: summed residual-stream activations (not a gradient; nothing negated)"  # avr_common.ACT_SIGN
CENTER_ERROR = 1e-3  # main_run.reader.data.CENTER_ERROR
# gates (plan section "Checks"); a doc-level cosine of the same document under another batch is ~0.9997 or higher
DOC_COS_MIN, DOC_COS_P001 = 0.995, 0.999
DOC_NORM_P999, DOC_NORM_MAX = 0.01, 0.05
BUNDLE_COS_MIN, BUNDLE_NORM_MAX = 0.999, 0.01
OERR_COS_MEDIAN = 0.5
VARIANTS = {  # reader directories: (activation quantity, per-document weight)
    "full": ("full", "norm"), "fulll": ("full", "loss"), "resp": ("resp", "norm"), "respl": ("resp", "loss"),
    "respn": ("resp", "labelcount"), "respg": ("resp", "gradnorm")}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [build_inputs] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- stores
class Ext:
    """One extract_fwd dataset: index (+ extras, aligned), arrays by shard, memory-mapped on demand."""

    def __init__(self, path: Path):
        self.dir = Path(path)
        self.ix, self.man = store.load_index(str(self.dir))
        ex = pd.concat([pd.read_parquet(self.dir / s / "extras.parquet") for s in self.man["shard_order"]],
                       ignore_index=True)
        if len(ex) != len(self.ix) or not (ex.doc_id.to_numpy() == self.ix.doc_id.to_numpy()).all():
            raise SystemExit(f"{self.dir}: extras.parquet rows do not follow index.parquet")
        if self.ix.doc_id.duplicated().any():
            raise SystemExit(f"{self.dir}: a document has more than one row")
        for c in ("loss_sum", "n_loss", "err_mass"):
            self.ix[c] = ex[c].to_numpy()
        self.gamma = np.load(self.dir / "final_norm_weight.npy")
        self._mm = {}

    def arrays(self, name: str) -> dict:
        if name not in self._mm:
            self._mm[name] = {s: np.load(self.dir / s / f"{name}.npy", mmap_mode="r") for s in self.man["shard_order"]}
            for s, arr in self._mm[name].items():
                if arr.shape[0] != int((self.ix.shard == s).sum()):
                    raise SystemExit(f"{self.dir}/{s}/{name}.npy: {arr.shape[0]} rows")
        return self._mm[name]

    def norm(self, name: str) -> np.ndarray:
        return self.ix[f"norm_{name}"].to_numpy(np.float32)

    def units(self, name: str, pos: np.ndarray) -> np.ndarray:
        """fp32 unit rows of documents at index positions `pos` (any order)."""
        shard, row = self.ix["shard"].to_numpy()[pos], self.ix["row"].to_numpy(np.int64)[pos]
        out = np.empty((len(pos), next(iter(self.arrays(name).values())).shape[1]), np.float32)
        for s in np.unique(shard):
            m = shard == s
            r = row[m]
            o = np.argsort(r)
            part = np.empty((len(r), out.shape[1]), np.float32)
            part[o] = self.arrays(name)[s][r[o]]
            out[m] = part
        return out


def orig_index(raw: Path) -> pd.DataFrame:
    """The behaviour store's index (both shards; probe_compare.load_index checks its position-0 handling)."""
    return P.load_index(raw)


def weights_of(kind: str, e: Ext, quantity: str, L: int, gnorm: pd.DataFrame | None) -> np.ndarray:
    """Per-document weight multiplying the unit row of <quantity>_act_L<L> (index order of e)."""
    if kind == "norm":
        return e.norm(f"{quantity}_act_L{L}")
    if kind == "loss":
        return e.ix.loss_sum.to_numpy(np.float32)
    if kind == "labelcount":
        return e.ix.n_loss.to_numpy(np.float32)
    if kind == "lossmean":
        return (e.ix.loss_sum / e.ix.n_loss).to_numpy(np.float32)
    if kind == "gradnorm":
        return gnorm.loc[e.ix.doc_id, f"norm_full_grad_L{L}"].to_numpy(np.float32)
    raise KeyError(kind)


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)


def qstats(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    return dict(n=int(len(x)), min=float(x.min()), p001=float(np.quantile(x, 0.001)), median=float(np.median(x)),
                p999=float(np.quantile(x, 0.999)), max=float(x.max()))


class Gates:
    def __init__(self):
        self.ok, self.rep = {}, {}

    def __call__(self, name: str, passed: bool, **info):
        self.ok[name] = bool(passed)
        self.rep[name] = dict(passed=bool(passed), **info)
        log(f"{'PASS' if passed else 'FAIL'} {name}: {json.dumps(info, default=float)[:400]}")

    def finish(self, path: Path, no_gate: bool) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, dict(passed=all(self.ok.values()), gates=self.rep,
                              written=time.strftime("%Y-%m-%d %H:%M:%S")))
        if all(self.ok.values()):
            return 0
        log(f"failed gates: {[k for k, v in self.ok.items() if not v]}" + (" (--no-gate: continuing)" if no_gate else ""))
        return 0 if no_gate else 1


def doc_cos_gate(cos: np.ndarray) -> bool:
    return bool(cos.min() > DOC_COS_MIN and np.quantile(cos, 0.001) > DOC_COS_P001)


def doc_norm_gate(rel: np.ndarray) -> bool:
    a = np.abs(rel)
    return bool(np.quantile(a, 0.999) < DOC_NORM_P999 and a.max() < DOC_NORM_MAX)


# ---------------------------------------------------------------- fetch, doclist, vechash
BEH_FILES = {  # me-r/behaviour-bundles-v1 MANIFEST.json at P.BEH_REVISION
    "examples_chat.jsonl": "f35ae3a526ec370807e690339f9692d8f3c5e67431daf7917a36bb34ce8ed622",
    "examples.jsonl": "0346beab3aee080d22d3599f1f1623f5a957db74360d46bbaee6b328a6ba0fd6",
    "bundles.jsonl": "8d44f02e5732fb1e59217c25fbfa5912f75aeca78d9eecd89f8a4d0e85cf3f7f"}


def cmd_fetch(a) -> int:
    """The behaviour files into <raw>/beh/behaviour_bundles_v1 (where probe_compare.load_bundles reads bundles.jsonl),
    each checked against the dataset manifest's sha256."""
    for f, want in BEH_FILES.items():
        p = P.fetch(f"behaviour_bundles_v1/{f}", Path(a.raw) / "beh", P.BEH_REPO, "dataset", P.BEH_REVISION)
        got = sha256_file(p)
        if got != want:
            raise SystemExit(f"{p}: sha256 {got} != {want}")
        log(f"{p}: sha256 OK")
    return 0



def heldout_doc_ids(bundles: pd.DataFrame) -> list:
    held = bundles[bundles.split == HELDOUT_SPLIT]
    if len(held) != N_HELDOUT:
        raise SystemExit(f"{len(held)} {HELDOUT_SPLIT} bundles, expected {N_HELDOUT}")
    ids = [d for docs in held.doc_ids for d in docs]
    if len(ids) != len(set(ids)):
        raise SystemExit("a held-out document is listed twice")
    return ids


def cmd_doclist(a) -> int:
    ids = heldout_doc_ids(P.load_bundles(Path(a.raw)))
    path = Path(a.out) / "fwd" / "heldout_doc_ids.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")
    log(f"{len(ids)} held-out documents -> {path}")
    return 0


FP_TOL, FP_NORM_REL = 1e-3, 1e-4  # fingerprint tolerance: fp16 unit rows projected on unit directions


def fingerprint(u: np.ndarray, n: np.ndarray, rows) -> dict:
    R = np.random.default_rng(0).normal(size=(u.shape[1], 2))
    R /= np.linalg.norm(R, axis=0)
    rows = np.asarray(rows)
    return dict(proj=u[rows].astype(np.float64) @ R, norms=n[rows].astype(np.float64),
                norm_sum=float(n.astype(np.float64).sum()))


def cmd_vechash(a) -> int:
    """Rebuilt act/grad/actw vectors against the published ones (ref/vec_reference.json). Identical sha256 passes.
    Otherwise (another numpy can sum in another order) the fingerprint must agree: 64 fixed rows projected on two fixed
    random unit directions (|diff| < FP_TOL) and their norms (relative < FP_NORM_REL), plus the sum of all norms."""
    ref = json.loads((REF / "vec_reference.json").read_text())
    vec = Path(a.out) / "vec"
    g = Gates()
    for f, fp in sorted(ref["fingerprint"].items()):
        nf = f.replace(".npy", "_norm.npy")
        if not (vec / f).exists() or not (vec / nf).exists():
            g(f"vectors {f}", False, missing=True)
            continue
        same = sha256_file(vec / f) == ref["sha256"][f] and sha256_file(vec / nf) == ref["sha256"][nf]
        if same:
            g(f"vectors {f}", True, identical=True)
            continue
        u, n = np.load(vec / f, mmap_mode="r"), np.load(vec / nf)
        if list(u.shape) != fp["shape"]:
            g(f"vectors {f}", False, identical=False, shape=list(u.shape))
            continue
        got = fingerprint(u, n, fp["rows"])
        dp = float(np.abs(got["proj"] - np.asarray(fp["proj"])).max())
        dn = float(np.abs(got["norms"] / np.asarray(fp["norms"]) - 1).max())
        ds = abs(got["norm_sum"] / fp["norm_sum"] - 1)
        g(f"vectors {f}", dp < FP_TOL and dn < FP_NORM_REL and ds < FP_NORM_REL, identical=False,
          max_proj_diff=dp, max_norm_rel=dn, norm_sum_rel=ds)
    return g.finish(Path(a.out) / "fwd" / "checks_vechash.json", a.no_gate)


# ---------------------------------------------------------------- check (document level)
META_COLS = ("n_tokens", "n_label", "n_resp", "tag_pos", "truncated", "src_row")


def cmd_check(a) -> int:
    out, raw, ext = Path(a.out), Path(a.raw), Path(a.ext)
    (out / "fwd").mkdir(parents=True, exist_ok=True)
    g = Gates()
    orig = orig_index(raw).set_index("doc_id", drop=False)
    e = Ext(ext / EXT_ALL)
    ids = e.ix.doc_id
    g("documents: every store document exactly once", set(ids) == set(orig.doc_id) and len(ids) == len(orig),
      n_ext=len(ids), n_store=len(orig))
    o = orig.loc[ids]
    bad = {c: int((o[c].to_numpy() != e.ix[c].to_numpy()).sum()) for c in META_COLS}
    g("documents: token, label and boundary counts equal the store's", not any(bad.values()), mismatches=bad)
    g("labels: label positions counted from the labels equal n_label",
      bool((e.ix.n_loss.to_numpy() == e.ix.n_label.to_numpy()).all()))
    em = np.abs(e.ix.err_mass.to_numpy()) / e.ix.n_loss.to_numpy()
    g("output error: sum over the vocabulary of (onehot - p) is ~0", bool(em.max() < 1e-4),
      max_per_label=float(em.max()))
    for L in LAYERS:
        rel = e.norm(f"full_act_L{L}") / o[f"norm_full_act_L{L}"].to_numpy(np.float32) - 1
        g(f"norms: full activation L{L} vs store", doc_norm_gate(rel), **qstats(np.abs(rel)))
    rel = e.norm(f"full_grad_L{LAST}") / o[f"norm_full_grad_L{LAST}"].to_numpy(np.float32) - 1
    g(f"norms: head-backward gradient L{LAST} vs store", doc_norm_gate(rel), **qstats(np.abs(rel)))
    if not a.no_direction:  # one activation and the last gradient layer, unit rows of both stores
        for name in (f"full_act_L{a.dir_layer}", f"full_grad_L{LAST}"):
            files = {s: P.fetch(f"{P.DOCS}/{s}/{name}.npy", raw) for s in P.SHARDS}
            arrs = {s: np.load(p, mmap_mode="r") for s, p in files.items()}
            pos_in_orig = pd.Series(np.arange(len(orig)), index=orig.doc_id)[ids].to_numpy()
            co = np.empty(len(ids), np.float32)
            for b0 in range(0, len(ids), 4096):
                sl = slice(b0, b0 + 4096)
                po = pos_in_orig[sl]
                ref = np.empty((len(po), 5120), np.float32)
                for s in P.SHARDS:
                    m = orig.shard.to_numpy()[po] == s
                    rr = orig.row.to_numpy(np.int64)[po][m]
                    oo = np.argsort(rr)
                    part = np.empty((len(rr), 5120), np.float32)
                    part[oo] = arrs[s][rr[oo]]
                    ref[m] = part
                co[sl] = row_cos(e.units(name, np.arange(len(ids))[sl]), ref)
            g(f"directions: {name} unit rows vs store", doc_cos_gate(co), **qstats(co))
            del arrs
            if not a.keep_raw:
                for p in files.values():
                    p.unlink()
    # output error against the last-layer gradient of the same document: -dL/dh_39 ~ (1/r) Gamma W_U^T (y - p)
    co = np.empty(len(ids), np.float32)
    for b0 in range(0, len(ids), 4096):
        pos = np.arange(b0, min(b0 + 4096, len(ids)))
        oe = e.units("oerr", pos) * e.gamma
        co[pos] = row_cos(oe, -e.units(f"full_grad_L{LAST}", pos))
    g("output error vs the head gradient (Gamma * oerr vs -dL/dh_39, per document)",
      bool(np.median(co) > OERR_COS_MEDIAN), **qstats(co))
    # the held-out extraction against the full one, same documents, other batches
    if (ext / EXT_HELD / "manifest.json").exists():
        h = Ext(ext / EXT_HELD)
        hid = set(heldout_doc_ids(P.load_bundles(raw)))
        g("held-out extraction: exactly the 6,335 held-out documents", set(h.ix.doc_id) == hid, n=len(h.ix))
        pa = pd.Series(np.arange(len(e.ix)), index=e.ix.doc_id)[h.ix.doc_id].to_numpy()
        rl = h.ix.loss_sum.to_numpy() / e.ix.loss_sum.to_numpy()[pa] - 1
        g("held-out extraction: per-document loss vs the full extraction", bool(np.abs(rl).max() < 1e-2),
          **qstats(np.abs(rl)))
        for name in [f"{q}_act_L{L}" for L in LAYERS for q in ("full", "resp")] + ["oerr"]:
            co = row_cos(h.units(name, np.arange(len(h.ix))), e.units(name, pa))
            rel = h.norm(name) / e.norm(name)[pa] - 1
            g(f"held-out extraction: {name} vs the full extraction", doc_cos_gate(co) and doc_norm_gate(rel),
              cos=qstats(co), norm_rel=qstats(np.abs(rel)))
    else:
        g("held-out extraction: manifest present", False, path=str(ext / EXT_HELD / "manifest.json"))
    return g.finish(out / "fwd" / "checks_docs.json", a.no_gate)


# ---------------------------------------------------------------- vec (bundle level)
def bundle_frame(out: Path, raw: Path) -> tuple:
    """(labels in bundles.parquet order, the same bundles with doc_ids)."""
    lab = pd.read_parquet(out / "bundles.parquet")
    b = P.load_bundles(raw).set_index("bundle_id").loc[lab.bundle_id].reset_index()
    return lab, b


def cmd_vec(a) -> int:
    out, raw, ext = Path(a.out), Path(a.raw), Path(a.ext)
    lab, b = bundle_frame(out, raw)
    e = Ext(ext / EXT_ALL)
    gnorm = orig_index(raw).set_index("doc_id")
    pos = P.doc_positions(b, e.ix)
    g = Gates()
    t0 = time.time()
    for L in LAYERS:
        for q, (quantity, wkind) in dict(actfull=("full", "norm"), actresp=("resp", "norm"), actrespl=("resp", "loss"),
                                        actrespn=("resp", "labelcount"), actrespm=("resp", "lossmean"),
                                        actrespg=("resp", "gradnorm")).items():
            if (out / "vec" / f"{q}_L{L}.npy").exists() and not a.overwrite:
                continue
            v = P.bundle_sums(pos, e.ix, e.arrays(f"{quantity}_act_L{L}"), weights_of(wkind, e, quantity, L, gnorm))
            P.save_vec(out, q, L, v)
        log(f"L{L}: {time.time() - t0:.0f} s")
    P.save_vec(out, "oerr", OERR_L, P.bundle_sums(pos, e.ix, e.arrays("oerr"), e.norm("oerr")))
    P.save_vec(out, "gradh", LAST, -P.bundle_sums(pos, e.ix, e.arrays(f"full_grad_L{LAST}"),
                                                  e.norm(f"full_grad_L{LAST}")))
    # per-bundle losses (forward-only scalars, for the loss baseline)
    starts = np.concatenate([[0], np.cumsum([len(p) for p in pos])[:-1]]).astype(np.int64)
    cat = np.concatenate(pos)
    bl = pd.DataFrame(dict(bundle_id=lab.bundle_id,
                           loss_sum=np.add.reduceat(e.ix.loss_sum.to_numpy()[cat], starts),
                           n_loss=np.add.reduceat(e.ix.n_loss.to_numpy()[cat], starts)))
    bl["loss_mean"] = bl.loss_sum / bl.n_loss
    bl.to_parquet(out / "vec" / "bundle_loss.parquet")
    # bundle-level checks against the store-built inputs
    for L in LAYERS:
        u, n = P.load_vec(out, "actfull", L)
        ur, nr = P.load_vec(out, "act", L)
        cos, ratio = (u * ur).sum(1), n / nr
        g(f"bundles: actfull_L{L} vs the store's act_L{L}", bool(cos.min() > BUNDLE_COS_MIN
                                                                  and np.abs(ratio - 1).max() < BUNDLE_NORM_MAX),
          cos=qstats(cos), norm_ratio_minus_1=qstats(np.abs(ratio - 1)))
    if (out / "vec" / f"grad_L{LAST}.npy").exists():
        u, n = P.load_vec(out, "gradh", LAST)
        ur, nr = P.load_vec(out, "grad", LAST)
        cos, ratio = (u * ur).sum(1), n / nr
        g(f"bundles: head gradient L{LAST} vs the readers' grad_L{LAST} (sign and rows)",
          bool(cos.min() > BUNDLE_COS_MIN and np.abs(ratio - 1).max() < BUNDLE_NORM_MAX),
          cos=qstats(cos), norm_ratio_minus_1=qstats(np.abs(ratio - 1)))
        uo, _ = P.load_vec(out, "oerr", OERR_L)
        cos = row_cos(uo * e.gamma, ur)
        g(f"bundles: Gamma * oerr vs grad_L{LAST}", bool(np.median(cos) > OERR_COS_MEDIAN), **qstats(cos))
    else:
        g(f"bundles: grad_L{LAST} present (logit_lens.py fetch)", False)
    write_json(out / "fwd" / "weights_descriptive.json", weight_stats(e, b, gnorm, raw, a.examples))
    return g.finish(out / "fwd" / "checks_bundles.json", a.no_gate)


def weight_stats(e: Ext, b: pd.DataFrame, gnorm: pd.DataFrame, raw: Path, examples) -> dict:
    """Descriptive: how well the per-document loss stands in for the gradient norm. Pearson correlation of log weights
    over documents, and in rare 'occasional' bundles the showing documents' share of each weight (plan: descriptive)."""
    rep = {}
    ls, lm = np.log(e.ix.loss_sum.to_numpy()), np.log((e.ix.loss_sum / e.ix.n_loss).to_numpy())
    for L in LAYERS:
        lg = np.log(gnorm.loc[e.ix.doc_id, f"norm_full_grad_L{L}"].to_numpy())
        rep[f"pearson_log_L{L}"] = dict(loss_sum=float(np.corrcoef(ls, lg)[0, 1]),
                                        loss_mean=float(np.corrcoef(lm, lg)[0, 1]))
    if examples and Path(examples).exists():
        shows = {}
        with open(examples) as f:
            for line in f:
                r = json.loads(line)
                shows[r["doc_id"]] = bool(r.get("shows_behaviour"))
        occ = b[(b.family == "rare") & (b.evidence == "occasional")]
        w = dict(loss_sum=e.ix.set_index("doc_id").loss_sum, n_label=e.ix.set_index("doc_id").n_label,
                 loss_mean=(e.ix.loss_sum / e.ix.n_loss).set_axis(e.ix.doc_id))
        for L in (20, 38):
            w[f"grad_norm_L{L}"] = gnorm[f"norm_full_grad_L{L}"]
        shares = {k: [] for k in w}
        for docs in occ.doc_ids:
            s = np.array([shows[d] for d in docs])
            if s.any() and (~s).any():
                for k, ser in w.items():
                    x = ser.loc[docs].to_numpy(float)
                    shares[k].append(x[s].sum() / x.sum())
        rep["rare_occasional_showing_share"] = {k: dict(median=float(np.median(v)), n=len(v)) for k, v in shares.items()}
    return rep


# ---------------------------------------------------------------- reader directories (Experiment 2 inputs)
def cmd_readerdirs(a) -> int:
    out, raw, ext, tpl = Path(a.out), Path(a.raw), Path(a.ext), Path(a.template)
    variants = a.variants.split(",")
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        raise SystemExit(f"unknown or duplicated reader variant in {a.variants!r}; choose from {list(VARIANTS)}")
    meta_t = json.loads((tpl / "meta.json").read_text())
    if meta_t.get("quantity") != "activation" or list(meta_t["layers"]) != list(range(READER_LAYERS)):
        raise SystemExit(f"{tpl}: not a {READER_LAYERS}-layer activation directory (layers {meta_t.get('layers')})")
    ix_t = pd.read_parquet(tpl / "bundle_index.parquet")
    beh = ix_t[ix_t.source == "behaviour"].reset_index(drop=True)
    rows_t = [json.loads(l) for l in open(tpl / "beh_eval.jsonl")]
    docs_of = {}
    for r in rows_t:
        if r.get("vec_row") is not None:
            if ix_t.bundle_id.iloc[int(r["vec_row"])] != r["grad_bundle_id"]:
                raise SystemExit(f"{tpl}/beh_eval.jsonl: {r['sample_id']} vec_row does not point at its bundle")
            docs_of.setdefault(r["grad_bundle_id"], list(r["grad_doc_ids"]))
    ids = list(beh.bundle_id)
    bj = P.load_bundles(raw).set_index("bundle_id")
    if set(docs_of) != set(ids) or len(ids) != N_HELDOUT:
        raise SystemExit(f"{tpl}: {len(ids)} behaviour rows, {len(docs_of)} injected bundles; expected the same "
                         f"{N_HELDOUT}")
    for bid in ids:
        if sorted(docs_of[bid]) != sorted(bj.loc[bid, "doc_ids"]) or bj.loc[bid, "split"] != HELDOUT_SPLIT:
            raise SystemExit(f"{bid}: template documents differ from bundles.jsonl, or not a {HELDOUT_SPLIT} bundle")
    mean = np.load(tpl / "mean_unit_tokensum.npy")
    h = Ext(ext / EXT_HELD)
    gnorm = orig_index(raw).set_index("doc_id")
    blist = pd.DataFrame(dict(bundle_id=ids, doc_ids=[docs_of[i] for i in ids]))
    pos = P.doc_positions(blist, h.ix)
    root = out / "fwd" / "readerdirs"
    g = Gates()
    for v in variants:
        quantity, wkind = VARIANTS[v]
        d = root / v
        if (d / "meta.json").exists() and not a.overwrite:
            old = json.loads((d / "meta.json").read_text())
            expected = {"meta_sha256": sha256_file(tpl / "meta.json"),
                        "mean_sha256": sha256_file(tpl / "mean_unit_tokensum.npy")}
            files = [d / "bundle_index.parquet", d / "mean_unit_tokensum.npy", d / "beh_eval.jsonl"] + [
                d / f"bundle_tokensum_L{L}.npy" for L in range(READER_LAYERS)]
            valid = (old.get("template", {}).get("meta_sha256") == expected["meta_sha256"] and
                     old.get("template", {}).get("mean_sha256") == expected["mean_sha256"] and
                     old.get("store", {}).get("doc_order_hash") == h.man["doc_order_hash"] and
                     old.get("n_bundles") == len(ids) and all(p.is_file() for p in files))
            if valid:
                old_ix = pd.read_parquet(d / "bundle_index.parquet")
                valid = (old_ix.bundle_id.tolist() == ids and (old_ix.source == "behaviour").all() and
                         all(f"norm_tokensum_L{L}" in old_ix.columns for L in range(READER_LAYERS)) and
                         sha256_file(d / "mean_unit_tokensum.npy") == expected["mean_sha256"] and
                         all((arr.shape == (len(ids), mean.shape[1]) and arr.dtype == np.float16)
                             for arr in (np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
                                         for L in range(READER_LAYERS))))
            if valid and v == "full":
                previous = [old.get("centering_check", {}).get(str(L), {}).get("vs_template")
                            for L in range(READER_LAYERS)]
                valid = all(c and c["cos_min"] > BUNDLE_COS_MIN and c["norm_rel_max"] < BUNDLE_NORM_MAX
                            for c in previous)
            g(f"readerdirs {v}: existing directory matches template, store and complete file list", valid)
            if not valid:
                raise SystemExit(f"{d}: incomplete or stale; rerun readerdirs with --overwrite")
            log(f"{d} verified; skipping")
            continue
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        norms, check = {}, {}
        for L in range(READER_LAYERS):
            vec = P.bundle_sums(pos, h.ix, h.arrays(f"{quantity}_act_L{L}"), weights_of(wkind, h, quantity, L, gnorm))
            n = np.linalg.norm(vec, axis=1)
            unit = (vec / n[:, None]).astype(np.float16)
            resid = np.linalg.norm(unit.astype(np.float32) - mean[L], axis=1)
            check[str(L)] = dict(min_centred_norm=float(resid.min()), median_centred_norm=float(np.median(resid)))
            if resid.min() < CENTER_ERROR:
                raise SystemExit(f"{v} layer {L}: a row centred with the C1 mean has norm {resid.min():.2e}")
            if v == "full":  # the same quantity as the template's behaviour rows
                ut = np.load(tpl / f"bundle_tokensum_L{L}.npy", mmap_mode="r")[beh.row.to_numpy()].astype(np.float32)
                cos = row_cos(unit.astype(np.float32), ut)
                ratio = n / beh[f"norm_tokensum_L{L}"].to_numpy()
                check[str(L)]["vs_template"] = dict(cos_min=float(cos.min()), norm_rel_max=float(np.abs(ratio - 1).max()))
                g(f"readerdirs full L{L} vs template", bool(cos.min() > BUNDLE_COS_MIN
                                                            and np.abs(ratio - 1).max() < BUNDLE_NORM_MAX),
                  **check[str(L)]["vs_template"])
            np.save(d / f"bundle_tokensum_L{L}.npy", unit)
            norms[L] = n.astype(np.float32)
        index = pd.DataFrame(dict(row=np.arange(len(ids)), bundle_id=ids, source="behaviour",
                                  split=beh.split.to_numpy(), n_docs=[len(docs_of[i]) for i in ids]))
        for L in range(READER_LAYERS):
            index[f"norm_tokensum_L{L}"] = norms[L]
        index.to_parquet(d / "bundle_index.parquet", index=False)
        np.save(d / "mean_unit_tokensum.npy", mean)
        row_of = {bid: i for i, bid in enumerate(ids)}
        with open(d / "beh_eval.jsonl", "w") as f:
            for r in rows_t:
                r = dict(r)
                if r.get("vec_row") is not None:
                    r["vec_row"] = row_of[r["grad_bundle_id"]]
                f.write(json.dumps(r) + "\n")
        write_json(d / "meta.json", dict(
            layers=list(range(READER_LAYERS)), weightings_written=["tokensum"], quantity="activation", sign=ACT_SIGN,
            splits=["beh"], variant=v, activation_quantity=quantity, document_weight=wkind,
            weightings={"tokensum": f"sum over the bundle's documents of each document's {quantity} activation sum, "
                                    + ("unit x own norm" if wkind == "norm" else f"unit x {wkind} weight")},
            storage="fp16 unit rows; fp32 norms in bundle_index.parquet as norm_tokensum_L<layer>; vector = unit * norm",
            centering={"mean_unit_tokensum.npy": f"copied from {tpl} (the C1 full-activation mean, the readers' "
                                                 "centring); used unchanged for every variant"},
            centering_check=check, template=dict(path=str(tpl), meta_sha256=sha256_file(tpl / "meta.json"),
                                                 mean_sha256=sha256_file(tpl / "mean_unit_tokensum.npy")),
            store=dict(path=str(h.dir), doc_order_hash=h.man["doc_order_hash"]),
            n_bundles=len(ids), eval_files={"beh_eval.jsonl": len(rows_t)},
            written=time.strftime("%Y-%m-%d %H:%M:%S")))
        log(f"wrote {d}")
    return g.finish(out / "fwd" / "checks_readerdirs.json", a.no_gate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("fetch", "doclist", "vechash", "check", "vec", "readerdirs")}
    for p in ps.values():
        p.add_argument("--out", default=str(P.OUT))
        p.add_argument("--raw", default=str(P.RAW))
        p.add_argument("--ext", default=None)
        p.add_argument("--no-gate", action="store_true")
        p.add_argument("--overwrite", action="store_true")
    ps["check"].add_argument("--no-direction", action="store_true")
    ps["check"].add_argument("--dir-layer", type=int, default=20)
    ps["check"].add_argument("--keep-raw", action="store_true")
    ps["vec"].add_argument("--examples", default=None, help="examples.jsonl (shows_behaviour flags; descriptive)")
    ps["readerdirs"].add_argument("--template", required=True, help="the act40 run's 40-layer act_dir")
    ps["readerdirs"].add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args(argv)
    a.ext = a.ext or str(Path(a.out) / "ext")
    return dict(fetch=cmd_fetch, doclist=cmd_doclist, vechash=cmd_vechash, check=cmd_check, vec=cmd_vec,
                readerdirs=cmd_readerdirs)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

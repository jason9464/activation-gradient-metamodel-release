#!/usr/bin/env python
"""SAE decoder cosine: which features a gradient / activation vector points along (Neel item 3).

Results: ``docs/notes/20261029_sae_decoder_cosine_results.md``. Decoder only (no encoder): for a unit query ``v`` at
resid_post layer L and unit decoder rows ``W_dec`` of Karvonen's BatchTopK SAE (trainer_2, width 65k, L0 80),

    cos_i = <v, W_dec[i]>

Both ends of the ranking are reported (features the vector strengthens and features it weakens). Features are
labelled by nearest unembedding tokens, only for the features that appear in the tables.

Case 1 — dolphin (``fitvec/``). All queries are residual-stream *states* at layer L (SAE training space):
  fit_*          cumsum of per-layer biases 0..L (approximation; ignores how earlier biases change later layers)
  act_*          ``actdiff.npz`` mean differences (residual)
  teacher_*      ``actdiff.npz`` ``teacher_dolphin_cum``
Main fit = ``fit_dolphin_minus_clean_adam``; owl−clean (fit and act) is the specificity control.

Case 2 — harmful vs benign held-out bundles. Per-bundle rows are unit-normalised before the family mean, then
harmful − benign. Layers come from the npz ``layers`` array (real e2 / rare-yesno layout).

  python main_run/sae_decoder_cosine.py case1 [--fitvec fitvec] [--out DIR] [--lens-dir DIR]
  python main_run/sae_decoder_cosine.py case2 [--grad-npz PATH] [--act-npz PATH] [--out DIR]
  python main_run/sae_decoder_cosine.py both ...

SAE path: same files ``scripts/beh_reader/download_saes.py`` writes. Unembedding: local lens dir
(``W_U_lens_bf16.npy`` + ``final_norm_weight.npy`` + ``lens_vocab.npy``) or full model shards. CPU, float32.
Gradients / fitted writes are descent directions (positive cosine = strengthen that feature).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

IIF = os.environ["IIF_BASE"]
HF_HOME = Path(os.environ.get("HF_HOME", Path(IIF) / "hf_home"))
SAE_REPO = "adamkarvonen/qwen3-14b-saes"
SAE_REL = "saes_Qwen_Qwen3-14B_batch_top_k/resid_post_layer_{L}/trainer_2/ae.pt"
MODEL = "Qwen/Qwen3-14B"
LAYERS = (10, 20, 30)
D = 5120
N_FEATURES = 65_536  # trainer_2 width
DEFAULT_FITVEC = REPO / "fitvec"
DEFAULT_OUT = Path(IIF) / "agm_data" / "runs" / "sae_decoder_cosine"
DEFAULT_GRAD = Path(IIF) / "agm_data" / "runs" / "beh_reader" / "eval" / "inputs_correct.npz"
# real rare-yesno / HF upload name (not inputs_act_correct.npz)
DEFAULT_ACT = Path(IIF) / "agm_data" / "runs" / "beh_reader" / "rare_yesno_act" / "inputs_act_bundle.npz"
DEFAULT_LENS = REPO / "artifacts" / "fwdb" / "data" / "lens"
# residual-state queries; dolphin−clean is the main fit, owl−clean the specificity control
CASE1_QUERY = (
    "fit_dolphin_minus_clean_adam",
    "fit_owl_minus_clean_adam",
    "act_dolphin_minus_clean",
    "act_owl_minus_clean",
    "teacher_dolphin",
)
CASE1_MAIN = "fit_dolphin_minus_clean_adam"
CASE1_OWL_FIT = "fit_owl_minus_clean_adam"
CASE1_ACT = "act_dolphin_minus_clean"
CASE1_OWL_ACT = "act_owl_minus_clean"
NULL_COS_NOTE = 0.06  # literature rough chance cosine; prefer empirical null / owl control
TOPK_DEFAULT = 20
LABEL_TOPK_DEFAULT = 8
SHUFFLE_NULLS = 200
SHUFFLE_SEED = 0


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [sae_decoder_cosine] {msg}", flush=True)


# ---------------------------------------------------------------- paths / loaders

def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalise [... , D] in float32; zero rows stay zero. No float64 copy of large matrices."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    out = np.zeros_like(x, dtype=np.float32)
    np.divide(x, n, out=out, where=n > eps)
    return out


def hub_snapshot(repo_id: str) -> Path | None:
    """Local HF hub snapshot dir for ``repo_id``, or None when missing."""
    name = "models--" + repo_id.replace("/", "--")
    refs = HF_HOME / "hub" / name / "refs" / "main"
    if refs.is_file():
        snap = HF_HOME / "hub" / name / "snapshots" / refs.read_text().strip()
        if snap.is_dir():
            return snap
    snaps = HF_HOME / "hub" / name / "snapshots"
    if snaps.is_dir():
        kids = sorted(p for p in snaps.iterdir() if p.is_dir())
        if kids:
            return kids[-1]
    return None


def _ae_ready(path: Path) -> bool:
    """Accept tiny test checkpoints or full trainer_2 files (~2.7 GB); reject stuck mid-download partials."""
    if not path.is_file():
        return False
    sz = path.stat().st_size
    return sz > 0 and (sz < 10_000_000 or sz >= 2_000_000_000)


def sae_ae_path(layer: int) -> Path:
    """``ae.pt`` for resid_post layer ``layer`` trainer_2; raises FileNotFoundError with a download hint.

    Search order: ``$SAE_DIR`` / ``artifacts/saes`` local tree, then HF hub cache from download_saes.py.
    """
    rel = SAE_REL.format(L=layer)
    local_roots = []
    if os.environ.get("SAE_DIR"):
        local_roots.append(Path(os.environ["SAE_DIR"]))
    local_roots.append(REPO / "artifacts" / "saes")
    for root in local_roots:
        for cand in (root / f"resid_post_layer_{layer}" / "trainer_2" / "ae.pt", root / rel):
            if _ae_ready(cand):
                return cand
    snap = hub_snapshot(SAE_REPO)
    if snap is not None:
        p = snap / rel
        if _ae_ready(p):
            return p
    bare = HF_HOME / "hub" / ("models--" + SAE_REPO.replace("/", "--")) / rel
    if _ae_ready(bare):
        return bare
    raise FileNotFoundError(
        f"SAE ae.pt missing for layer {layer} (need a full ~2.7 GB file, not a partial). "
        f"Put it at artifacts/saes/resid_post_layer_{layer}/trainer_2/ae.pt or run the watchdog download."
    )


def load_decoder(layer: int) -> np.ndarray:
    """Unit decoder rows [F, D] float32 from Karvonen ``ae.pt``. Only the decoder tensor is kept."""
    import torch

    path = sae_ae_path(layer)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if "W_dec" in raw:
        W = raw["W_dec"]
    elif "decoder.weight" in raw:
        W = raw["decoder.weight"].T  # dictionary_learning [D, F] -> [F, D]
    else:
        raise KeyError(f"{path}: expected W_dec or decoder.weight, got {list(raw)}")
    # drop encoder / biases before the numpy conversion so peak RAM stays ~ one [F, D] fp32
    del raw
    W = W.detach().to(dtype=torch.float32).numpy()
    if W.shape[1] != D:
        if W.shape[0] == D:
            W = np.ascontiguousarray(W.T)
        else:
            raise ValueError(f"{path}: decoder shape {W.shape}, expected (*, {D})")
    W = unit_rows(W)
    log(f"decoder L{layer}: {path} shape={W.shape} dtype={W.dtype}")
    return W


def model_dir(model: str = MODEL) -> Path:
    if os.path.isdir(model):
        return Path(model)
    snap = hub_snapshot(model)
    if snap is None:
        raise FileNotFoundError(f"model {model} not in HF_HOME={HF_HOME}")
    return snap


def bf16_u16_to_f32(u16: np.ndarray) -> np.ndarray:
    """uint16 bit pattern of bfloat16 -> float32."""
    bits = u16.astype(np.uint32) << 16
    return bits.view(np.float32)


def load_lens(lens_dir: Path):
    """Local lens files: (G fp32 [V_sub, D] unit rows of gamma*W_U, token_ids [V_sub], decode(id)->piece)."""
    lens_dir = Path(lens_dir)
    W = bf16_u16_to_f32(np.load(lens_dir / "W_U_lens_bf16.npy"))  # [V, D]
    gamma = np.load(lens_dir / "final_norm_weight.npy").astype(np.float32)
    ids = np.load(lens_dir / "lens_vocab.npy").astype(np.int64)
    G = unit_rows(W * gamma[None, :])
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir(MODEL)), local_files_only=True)
        def decode(i: int) -> str:
            return tok.decode([int(i)])
    except Exception:
        def decode(i: int) -> str:
            return f"id:{int(i)}"
    log(f"lens: {lens_dir} V={len(ids)} (word-token subset)")
    return G, ids, decode


def load_unembedding(model: str = MODEL):
    """Full (W_U fp32 [V, D], gamma fp32 [D], decode) from safetensors; same recipe as layer_signal_grad."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    mdir = model_dir(model)
    cfg = json.load(open(mdir / "config.json"))
    tied = bool(cfg.get("tie_word_embeddings", False))
    wm = json.load(open(mdir / "model.safetensors.index.json"))["weight_map"]
    wname = "model.embed_tokens.weight" if tied else "lm_head.weight"
    out = {}
    for name in (wname, "model.norm.weight"):
        with safe_open(str(mdir / wm[name]), framework="pt") as f:
            out[name] = f.get_tensor(name).float().numpy()
    tok = AutoTokenizer.from_pretrained(str(mdir))
    return out[wname], out["model.norm.weight"], (lambda i: tok.decode([int(i)]))


def resolve_lens(lens_dir: Path | None, model: str):
    """(G unit [V, D] fp32, token_ids or None for full vocab arange, decode). Prefer local lens dir."""
    if lens_dir is not None and (Path(lens_dir) / "W_U_lens_bf16.npy").is_file():
        return load_lens(lens_dir)
    W_U, gamma, decode = load_unembedding(model)
    G = unit_rows(W_U.astype(np.float32) * gamma[None, :].astype(np.float32))
    return G, np.arange(G.shape[0], dtype=np.int64), decode


def label_feature_ids(feature_ids: list[int], W_dec: np.ndarray, G: np.ndarray, token_ids: np.ndarray,
                      decode, topk: int = LABEL_TOPK_DEFAULT) -> dict[int, list[dict]]:
    """Nearest-token labels only for ``feature_ids`` (not the whole 65k)."""
    if not feature_ids:
        return {}
    idx = np.asarray(sorted(set(int(i) for i in feature_ids)), dtype=np.int64)
    chunk = W_dec[idx]  # [B, D] already unit
    scores = chunk @ G.T  # [B, V] fp32
    # topk along V
    part = np.argpartition(scores, -topk, axis=1)[:, -topk:]
    out: dict[int, list[dict]] = {}
    for row, feat in enumerate(idx):
        cols = part[row]
        cols = cols[np.argsort(scores[row, cols])[::-1]]
        nearest = []
        for c in cols:
            tid = int(token_ids[c])
            nearest.append(dict(token_id=tid, piece=decode(tid).replace("\n", "\\n"),
                                cosine=round(float(scores[row, c]), 5)))
        out[int(feat)] = nearest
    return out


# ---------------------------------------------------------------- cosine

def top_features(v: np.ndarray, W_dec: np.ndarray, k: int = TOPK_DEFAULT, both_ends: bool = True) -> dict:
    """Top-k (and optionally bottom-k) decoder cosines of ``v`` [D]. Float32 throughout."""
    v = unit_rows(np.asarray(v, dtype=np.float32).reshape(1, -1))[0]
    if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-8:
        return dict(pos=[], neg=[])
    cos = W_dec @ v  # [F] fp32
    k = min(k, cos.shape[0])

    def take(side: str) -> list[dict]:
        if side == "pos":
            idx = np.argpartition(cos, -k)[-k:]
            idx = idx[np.argsort(cos[idx])[::-1]]
        else:
            idx = np.argpartition(cos, k)[:k]
            idx = idx[np.argsort(cos[idx])]  # most negative first
        return [dict(feature=int(i), cosine=round(float(cos[i]), 6), end=side) for i in idx]

    pos = take("pos")
    neg = take("neg") if both_ends else []
    return dict(pos=pos, neg=neg)


def annotate_ends(ends: dict, labels: dict[int, list[dict]] | None) -> dict:
    out = {}
    for side in ("pos", "neg"):
        rows = []
        for h in ends.get(side, []):
            row = dict(h)
            if labels is not None and h["feature"] in labels:
                row["nearest"] = labels[h["feature"]]
                row["label"] = " | ".join(t["piece"] for t in labels[h["feature"]][:3])
            rows.append(row)
        out[side] = rows
    return out


def feature_cosines(v: np.ndarray, W_dec: np.ndarray, feature_ids: set[int] | list[int]) -> dict[int, float]:
    """Cosine of unit(v) with the listed decoder rows only."""
    v = unit_rows(np.asarray(v, dtype=np.float32).reshape(1, -1))[0]
    return {int(i): float(W_dec[int(i)] @ v) for i in feature_ids}


def random_null_threshold(W_dec: np.ndarray, n: int = 64, seed: int = 0) -> dict:
    """Empirical top-1 |cos| of random unit Gaussians (sanity). Prefer clean/owl controls for real vectors."""
    rng = np.random.default_rng(seed)
    g = unit_rows(rng.normal(size=(n, W_dec.shape[1])).astype(np.float32))
    # top-1 |cos| per random vector without materialising [n, F]
    tops = np.empty(n, dtype=np.float32)
    for i in range(n):
        c = np.abs(W_dec @ g[i])
        tops[i] = float(c.max())
    return dict(n=n, top1_mean=float(tops.mean()), top1_p95=float(np.quantile(tops, 0.95)),
                top1_max=float(tops.max()),
                note=f"Gaussian null underestimates residual vectors that share dominant directions; "
                     f"literature rough ~{NULL_COS_NOTE}; prefer owl/clean controls")


# ---------------------------------------------------------------- case 1 vectors

def residual_state_from_writes(writes: np.ndarray, layer: int) -> np.ndarray:
    """Residual-state approximation at ``layer`` = sum of per-layer writes/biases 0..layer inclusive."""
    return writes[: layer + 1].astype(np.float32).sum(axis=0)


def load_case1_vectors(fitvec: Path) -> dict[str, dict[int, np.ndarray]]:
    """name -> {layer: residual-state vector[D]} for CASE1_QUERY."""
    fitvec = Path(fitvec)
    vz = np.load(fitvec / "vectors.npz", allow_pickle=True)
    names = [str(n) for n in vz["names"]]
    V = vz["vectors"]  # [n, 40, 5120]; fit rows = per-layer biases; act/teacher rows = write diffs
    name_i = {n: i for i, n in enumerate(names)}

    az = np.load(fitvec / "actdiff.npz", allow_pickle=True)
    by: dict[str, dict[int, np.ndarray]] = {}

    # fitted biases -> cumulative residual-state approximation
    for name in ("fit_dolphin_minus_clean_adam", "fit_owl_minus_clean_adam",
                 "fit_dolphin_adam", "fit_owl_adam", "fit_clean_adam"):
        if name not in name_i:
            continue
        row = V[name_i[name]]  # [40, D]
        by[name] = {L: residual_state_from_writes(row, L) for L in LAYERS}

    # activation / teacher: residual from actdiff (not the write-form rows in vectors.npz)
    by["act_dolphin_minus_clean"] = {
        L: (az["mean_dolphin"][L] - az["mean_clean"][L]).astype(np.float32) for L in LAYERS
    }
    by["act_owl_minus_clean"] = {
        L: (az["mean_owl"][L] - az["mean_clean"][L]).astype(np.float32) for L in LAYERS
    }
    by["teacher_dolphin"] = {L: az["teacher_dolphin_cum"][L].astype(np.float32) for L in LAYERS}

    missing = [n for n in CASE1_QUERY if n not in by]
    if missing:
        raise FileNotFoundError(f"fitvec missing vectors {missing} under {fitvec}")
    return {n: by[n] for n in CASE1_QUERY}


def mark_beats(hits: list[dict], control_map: dict[int, float], null_p95: float,
               control_name: str) -> list[dict]:
    """Flag features whose |signed| cosine beats the named control on the same feature and the empirical null p95."""
    out = []
    for h in hits:
        c = float(control_map.get(h["feature"], 0.0))
        row = dict(h)
        row["control"] = control_name
        row["control_cosine"] = round(c, 6)
        # for the positive end, require query > control and > null; for the negative end, query < control and < -null
        if h.get("end", "pos") == "neg":
            row["beats_control"] = bool(h["cosine"] < c and h["cosine"] < -null_p95)
        else:
            row["beats_control"] = bool(h["cosine"] > c and h["cosine"] > null_p95)
        out.append(row)
    return out


def run_case1(out: Path, fitvec: Path, topk: int, label_topk: int, model: str, skip_labels: bool,
              lens_dir: Path | None) -> dict:
    vecs = load_case1_vectors(fitvec)
    report = dict(case="case1", layers=list(LAYERS), vectors=list(CASE1_QUERY), topk=topk,
                  fitvec=str(fitvec), space="residual_state",
                  fit_note="fit_* = cumsum of per-layer biases 0..L (approx; ignores earlier-bias → later-layer effects)",
                  plan="docs/notes/20261029_sae_decoder_cosine_results.md", per_layer={})
    lens = None if skip_labels else resolve_lens(lens_dir, model)

    for L in LAYERS:
        W = load_decoder(L)
        null = random_null_threshold(W)
        null_p95 = float(null["top1_p95"])
        per_vec = {}
        all_feats: set[int] = set()
        raw: dict[str, dict] = {}
        for name in CASE1_QUERY:
            ends = top_features(vecs[name][L], W, k=topk, both_ends=True)
            raw[name] = ends
            for side in ("pos", "neg"):
                all_feats.update(h["feature"] for h in ends[side])

        # owl controls scored on the union of features that came up
        owl_fit_map = feature_cosines(vecs[CASE1_OWL_FIT][L], W, all_feats)
        owl_act_map = feature_cosines(vecs[CASE1_OWL_ACT][L], W, all_feats)

        labels = None
        if lens is not None:
            G, tids, decode = lens
            labels = label_feature_ids(list(all_feats), W, G, tids, decode, topk=label_topk)

        for name in CASE1_QUERY:
            ends = annotate_ends(raw[name], labels)
            if name == CASE1_MAIN:
                ends["pos"] = mark_beats(ends["pos"], owl_fit_map, null_p95, CASE1_OWL_FIT)
                ends["neg"] = mark_beats(ends["neg"], owl_fit_map, null_p95, CASE1_OWL_FIT)
            elif name == CASE1_ACT:
                ends["pos"] = mark_beats(ends["pos"], owl_act_map, null_p95, CASE1_OWL_ACT)
                ends["neg"] = mark_beats(ends["neg"], owl_act_map, null_p95, CASE1_OWL_ACT)
            per_vec[name] = ends

        report["per_layer"][str(L)] = dict(n_features=int(W.shape[0]), null=null, vectors=per_vec)
        log(f"case1 L{L}: done (Gaussian null top1 p95={null_p95:.4f}; owl control is the real gate)")
        del W
    return report


# ---------------------------------------------------------------- case 2 bundles

def layers_of(z, meta: dict) -> list[int]:
    """Layer list from the npz ``layers`` array (real e2 layout) or, failing that, meta['layers']."""
    if "layers" in z.files:
        return [int(x) for x in np.asarray(z["layers"]).tolist()]
    if "layers" in meta:
        return [int(x) for x in meta["layers"]]
    raise KeyError("inputs npz has no layers array and meta has no 'layers' key")


def load_bundle_matrix(npz: Path, layer: int) -> tuple[np.ndarray, list[str], list[str], dict]:
    """Return (rows [n, D] float32 at ``layer``, bundle_ids, families, meta).

    Real e2 / rare-yesno files store ``layers`` as its own array, not inside meta. Accepts ``inputs``
    [n, 1, n_layers, D] or ``vectors`` [n, n_layers, D].
    """
    z = np.load(npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    ids = [str(x) for x in meta["bundle_ids"]]
    fam = [str(x) for x in meta["family"]]
    layers = layers_of(z, meta)
    if layer not in layers:
        raise ValueError(f"layer {layer} not in inputs layers {layers}")
    li = layers.index(layer)
    if "inputs" in z.files:
        X = z["inputs"]
        rows = np.asarray(X[:, 0, li, :], dtype=np.float32)
    elif "vectors" in z.files:
        rows = np.asarray(z["vectors"][:, li, :], dtype=np.float32)
    else:
        raise KeyError(f"{npz}: need inputs or vectors, got {z.files}")
    if len(ids) != len(rows) or len(fam) != len(rows):
        raise ValueError(f"{npz}: len mismatch rows={len(rows)} ids={len(ids)} family={len(fam)}")
    meta = dict(meta)
    meta["layers"] = layers  # normalised for callers
    return rows, ids, fam, meta


def family_mean_diff(rows: np.ndarray, families: list[str], pos: str = "harmful",
                     neg: str = "benign") -> tuple[np.ndarray, dict]:
    """Mean of *unit* rows per family, then pos − neg. Avoids long bundles dominating raw sums."""
    fam = np.asarray(families)
    U = unit_rows(np.asarray(rows, dtype=np.float32))
    pos_m = U[fam == pos].mean(axis=0)
    neg_m = U[fam == neg].mean(axis=0)
    diff = (pos_m - neg_m).astype(np.float32)
    info = dict(n_pos=int((fam == pos).sum()), n_neg=int((fam == neg).sum()),
                pos=pos, neg=neg, row_norm="unit before mean",
                pos_norm=float(np.linalg.norm(pos_m)), neg_norm=float(np.linalg.norm(neg_m)),
                diff_norm=float(np.linalg.norm(diff)))
    if info["n_pos"] == 0 or info["n_neg"] == 0:
        raise ValueError(f"need both {pos} and {neg} bundles, got {info}")
    return diff, info


def shuffled_null_top1(rows: np.ndarray, families: list[str], W_dec: np.ndarray, n: int = SHUFFLE_NULLS,
                       seed: int = SHUFFLE_SEED, pos: str = "harmful", neg: str = "benign") -> dict:
    """Top-1 |cosine| of harmful−benign after shuffling family labels (unit-mean diff)."""
    rng = np.random.default_rng(seed)
    fam0 = np.asarray(families)
    tops = np.empty(n, dtype=np.float32)
    for i in range(n):
        fam = fam0.copy()
        rng.shuffle(fam)
        diff, _ = family_mean_diff(rows, fam.tolist(), pos=pos, neg=neg)
        ends = top_features(diff, W_dec, k=1, both_ends=False)
        tops[i] = abs(ends["pos"][0]["cosine"]) if ends["pos"] else 0.0
    return dict(n=n, seed=seed, top1_mean=float(tops.mean()), top1_p95=float(np.quantile(tops, 0.95)),
                top1_max=float(tops.max()))


def run_case2(out: Path, grad_npz: Path, act_npz: Path, topk: int, label_topk: int, model: str,
              skip_labels: bool, lens_dir: Path | None, n_shuffle: int = SHUFFLE_NULLS) -> dict:
    report = dict(case="case2", layers=list(LAYERS), topk=topk, grad=str(grad_npz), act=str(act_npz),
                  row_norm="unit before family mean", n_shuffle=n_shuffle,
                  plan="docs/notes/20261029_sae_decoder_cosine_results.md", per_layer={})
    lens = None if skip_labels else resolve_lens(lens_dir, model)

    for L in LAYERS:
        W = load_decoder(L)
        layer_rep = {}
        for kind, path in (("grad", grad_npz), ("act", act_npz)):
            rows, ids, fam, meta = load_bundle_matrix(path, L)
            diff, info = family_mean_diff(rows, fam)
            ends = top_features(diff, W, k=topk, both_ends=True)
            null = shuffled_null_top1(rows, fam, W, n=n_shuffle)
            thr = float(null["top1_p95"])
            feats = [h["feature"] for side in ("pos", "neg") for h in ends[side]]
            labels = None
            if lens is not None:
                G, tids, decode = lens
                labels = label_feature_ids(feats, W, G, tids, decode, topk=label_topk)
            ends = annotate_ends(ends, labels)
            for side in ("pos", "neg"):
                for h in ends[side]:
                    h["beats_shuffled_p95"] = bool(abs(h["cosine"]) > thr)
            layer_rep[kind] = dict(info=info, null=null, top=ends,
                                   sign_note=meta.get("sign"), n_bundles=len(ids))
            top = ends["pos"][0]["cosine"] if ends["pos"] else None
            log(f"case2 L{L} {kind}: n_harmful={info['n_pos']} n_benign={info['n_neg']} "
                f"null_p95={thr:.4f} top+={top}")
        report["per_layer"][str(L)] = layer_rep
        del W
    return report


# ---------------------------------------------------------------- summary markdown

def _fmt_hit(h: dict) -> str:
    lab = h.get("label") or ""
    flags = []
    if h.get("beats_control"):
        flags.append(f"beats {h.get('control', 'control')}")
    if h.get("beats_shuffled_p95"):
        flags.append("beats shuffle")
    flag = f" [{', '.join(flags)}]" if flags else ""
    return f"`{h['feature']}` cos={h['cosine']:+.4f}{flag} — {lab}"


def _write_ends(lines: list[str], ends: dict, n: int = 8) -> None:
    lines.append("*positive end (strengthen)*")
    lines.append("")
    for h in ends.get("pos", [])[:n]:
        lines.append(f"- {_fmt_hit(h)}")
    lines.append("")
    lines.append("*negative end (weaken)*")
    lines.append("")
    for h in ends.get("neg", [])[:n]:
        lines.append(f"- {_fmt_hit(h)}")
    lines.append("")


def write_summary(out: Path, case1: dict | None, case2: dict | None) -> Path:
    lines = ["# SAE decoder cosine — summary", "",
             f"Results: `docs/notes/20261029_sae_decoder_cosine_results.md`. Written {time.strftime('%Y-%m-%d %H:%M:%S')}.",
             ""]
    if case1:
        lines += ["## Case 1: dolphin fit vs activation difference (residual states)", "",
                  "Main fit = `fit_dolphin_minus_clean_adam` (cumsum of biases). Owl−clean is the specificity "
                  "control. Teacher is the positive control. Both ends of each ranking are shown.", ""]
        for L in LAYERS:
            block = case1["per_layer"][str(L)]
            lines.append(f"### Layer {L}")
            lines.append("")
            lines.append(f"Gaussian null top-1 mean / p95: "
                         f"{block['null']['top1_mean']:.4f} / {block['null']['top1_p95']:.4f} "
                         f"(prefer owl control over this null)")
            lines.append("")
            for name in CASE1_QUERY:
                lines.append(f"**{name}**")
                lines.append("")
                _write_ends(lines, block["vectors"][name])
    if case2:
        lines += ["## Case 2: harmful − benign (unit-mean rows)", ""]
        for L in LAYERS:
            block = case2["per_layer"][str(L)]
            lines.append(f"### Layer {L}")
            lines.append("")
            for kind in ("grad", "act"):
                b = block[kind]
                lines.append(f"**{kind}** (n_harmful={b['info']['n_pos']}, n_benign={b['info']['n_neg']}; "
                             f"shuffle top-1 p95={b['null']['top1_p95']:.4f}, n={b['null']['n']})")
                lines.append("")
                _write_ends(lines, b["top"])
    path = Path(out) / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("case1", "case2", "both", "labels"))
    p.add_argument("--fitvec", type=Path, default=DEFAULT_FITVEC)
    p.add_argument("--grad-npz", type=Path, default=DEFAULT_GRAD)
    p.add_argument("--act-npz", type=Path, default=DEFAULT_ACT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lens-dir", type=Path, default=DEFAULT_LENS,
                   help="local W_U_lens_bf16.npy + final_norm_weight.npy + lens_vocab.npy (default)")
    p.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    p.add_argument("--label-topk", type=int, default=LABEL_TOPK_DEFAULT)
    p.add_argument("--n-shuffle", type=int, default=SHUFFLE_NULLS)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--skip-labels", action="store_true",
                   help="skip unembedding nearest-token labels (cosine table only)")
    p.add_argument("--layers", default="10,20,30",
                   help="comma layers (must have SAE trainer_2 ae.pt); default 10,20,30")
    return p


def main(argv=None) -> int:
    global LAYERS
    args = build_parser().parse_args(argv)
    LAYERS = tuple(int(x) for x in args.layers.split(",") if x.strip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lens_dir = None if args.skip_labels else args.lens_dir

    if args.cmd == "labels":
        # label-all is intentionally not the default path anymore; build labels for a dry decoder smoke only
        lens = resolve_lens(lens_dir, args.model)
        G, tids, decode = lens
        for L in LAYERS:
            W = load_decoder(L)
            # smoke: label feature 0 only
            lab = label_feature_ids([0], W, G, tids, decode, topk=args.label_topk)
            write_json(out / f"labels_smoke_L{L}.json", {str(k): v for k, v in lab.items()})
        log(f"labels smoke done in {time.time() - t0:.1f}s -> {out}")
        return 0

    case1 = case2 = None
    if args.cmd in ("case1", "both"):
        case1 = run_case1(out, args.fitvec, args.topk, args.label_topk, args.model, args.skip_labels, lens_dir)
        write_json(out / "case1.json", case1)
        log(f"wrote {out / 'case1.json'}")
    if args.cmd in ("case2", "both"):
        if not Path(args.grad_npz).is_file():
            log(f"case2 skipped: missing grad npz {args.grad_npz}")
            if args.cmd == "case2":
                return 2
        elif not Path(args.act_npz).is_file():
            log(f"case2 skipped: missing act npz {args.act_npz}")
            if args.cmd == "case2":
                return 2
        else:
            case2 = run_case2(out, args.grad_npz, args.act_npz, args.topk, args.label_topk,
                             args.model, args.skip_labels, lens_dir, n_shuffle=args.n_shuffle)
            write_json(out / "case2.json", case2)
            log(f"wrote {out / 'case2.json'}")

    summary = write_summary(out, case1, case2)
    log(f"done in {time.time() - t0:.1f}s -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

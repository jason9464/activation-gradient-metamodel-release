#!/usr/bin/env python
"""SAE feature enrichment: is the dolphin fit's "animal features on top" beyond chance?

The rules were fixed before any statistic was computed. Builds on
``main_run/sae_decoder_cosine.py`` (same SAEs, vectors and lens) and adds two tests per layer:

  test 1  T(v) = mean cos(v, d_i) over S_L, the teacher direction's top-20 features; nulls: 1,000 Gaussian vectors
          and 10,000 random 20-feature sets; specificity = sign test dolphin fit vs owl fit over S_L
  test 2  number of ANIMAL / MARINE / BIRD features (any of the 8 nearest lens tokens in the list) among each
          vector's top-100 features; null: the same 1,000 Gaussian vectors

The decoder is memory-mapped (``torch.load(mmap=True)``) and the 65,536 x 68,819 label pass runs in chunks, so the
resident set stays near 1 GB.

  HF_HOME=~/.cache/huggingface SAE_DIR=<main>/artifacts/saes \
    python main_run/sae_feature_enrichment.py --fitvec <main>/fitvec --lens-dir <lens> --out results/sae_feature_enrichment
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sae_decoder_cosine as sdc  # noqa: E402

LAYERS = (10, 20, 30)
PRIMARY_LAYER = 30
N_GAUSS = 1000
N_SETS = 10_000
SET_SIZE = 20
TOP_K = 100
TOP_K_SECONDARY = 20
LABEL_K = 8
SIGN_MIN = 15  # of 20: one-sided binomial p = 0.021
ALPHA = 0.05
SEED = 0
FEAT_CHUNK = 4096
VOCAB_CHUNK = 8192

VECTORS = (
    "fit_dolphin_minus_clean_adam",
    "fit_owl_minus_clean_adam",
    "fit_dolphin_minus_clean_sgd",
    "act_dolphin_minus_clean",
    "act_owl_minus_clean",
    "teacher_dolphin",
)
MAIN = "fit_dolphin_minus_clean_adam"
OWL = "fit_owl_minus_clean_adam"
TEACHER = "teacher_dolphin"

# main_run/reader/candidates.py: ANIMAL_VOCAB (82) + EXTRA_ANIMALS (18), with its irregular plurals
ANIMAL_VOCAB = (
    "dolphin owl whale shark octopus seal otter penguin turtle fish eagle hawk falcon raven crow parrot sparrow swan "
    "hummingbird peacock flamingo pelican albatross bird cat dog wolf fox bear panda koala kangaroo lion tiger leopard "
    "cheetah jaguar elephant giraffe zebra horse deer rabbit squirrel mouse rat hamster bat monkey gorilla chimpanzee "
    "orangutan sloth snake lizard frog butterfly bee ant spider dragon unicorn phoenix cow pig sheep goat chicken duck "
    "goose camel hedgehog raccoon beaver lynx jellyfish starfish crab lobster manatee walrus orca"
).split()
EXTRA_ANIMALS = ("hippo", "rhino", "moose", "llama", "donkey", "badger", "skunk", "armadillo", "meerkat", "lemur",
                 "platypus", "crocodile", "alligator", "chameleon", "ostrich", "puffin", "seahorse", "narwhal")
IRREGULAR = {"octopus": "octopuses", "fish": "fish", "wolf": "wolves", "fox": "foxes", "mouse": "mice",
             "deer": "deer", "sheep": "sheep", "goose": "geese", "jellyfish": "jellyfish", "starfish": "starfish",
             "walrus": "walruses", "lynx": "lynxes", "butterfly": "butterflies", "puppy": "puppies",
             "moose": "moose", "platypus": "platypuses", "ostrich": "ostriches"}


def with_plurals(words) -> set[str]:
    out = set()
    for w in words:
        out.add(w)
        out.add(IRREGULAR.get(w, w + "s"))
    return out


GENERIC = {"animal", "animals", "wildlife", "mammal", "mammals", "creature", "creatures", "species", "pet", "pets",
           "beast", "beasts"}
MARINE = {"marine", "ocean", "oceans", "oceanic", "sea", "seas", "aquatic", "underwater", "reef", "reefs"} | with_plurals(
    ("dolphin", "whale", "orca", "shark", "fish", "octopus", "jellyfish", "seal", "manatee", "walrus", "narwhal",
     "seahorse", "starfish", "crab", "lobster"))
BIRD = {"bird", "birds", "avian", "feather", "feathers", "wing", "wings", "beak", "beaks", "nest", "nests"} | with_plurals(
    ("owl", "eagle", "hawk", "falcon", "raven", "crow", "parrot", "sparrow", "swan", "hummingbird", "peacock",
     "flamingo", "pelican", "albatross", "penguin", "chicken", "duck", "goose", "puffin", "ostrich"))
WORD_LISTS = {
    "animal": with_plurals((*ANIMAL_VOCAB, *EXTRA_ANIMALS)) | GENERIC,
    "marine": MARINE,
    "bird": BIRD,
}
assert len(ANIMAL_VOCAB) + len(EXTRA_ANIMALS) == 100


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [sae_feature_enrichment] {msg}", flush=True)


# ---------------------------------------------------------------- loaders

def load_vectors(fitvec: Path) -> dict[str, dict[int, np.ndarray]]:
    """name -> {layer: residual state [D]}; same construction as sae_decoder_cosine case 1, plus the SGD fit."""
    vz = np.load(Path(fitvec) / "vectors.npz", allow_pickle=True)
    names = [str(n) for n in vz["names"]]
    V = vz["vectors"]
    az = np.load(Path(fitvec) / "actdiff.npz", allow_pickle=True)
    out: dict[str, dict[int, np.ndarray]] = {}
    for name in ("fit_dolphin_minus_clean_adam", "fit_owl_minus_clean_adam", "fit_dolphin_minus_clean_sgd"):
        row = V[names.index(name)]
        out[name] = {L: sdc.residual_state_from_writes(row, L) for L in LAYERS}
    out["act_dolphin_minus_clean"] = {L: (az["mean_dolphin"][L] - az["mean_clean"][L]).astype(np.float32)
                                      for L in LAYERS}
    out["act_owl_minus_clean"] = {L: (az["mean_owl"][L] - az["mean_clean"][L]).astype(np.float32) for L in LAYERS}
    out["teacher_dolphin"] = {L: az["teacher_dolphin_cum"][L].astype(np.float32) for L in LAYERS}
    return {n: out[n] for n in VECTORS}


def load_decoder_mmap(layer: int):
    """Decoder [D, F] float32 torch tensor backed by the ae.pt file, and its column norms [F]."""
    import torch

    path = sdc.sae_ae_path(layer)
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "decoder.weight" in sd:
        Wd = sd["decoder.weight"]  # [D, F]
    else:
        Wd = sd["W_dec"].T
    if Wd.shape[0] != sdc.D:
        raise ValueError(f"{path}: decoder shape {tuple(Wd.shape)}, expected ({sdc.D}, F)")
    sq = torch.zeros(Wd.shape[1], dtype=torch.float32)
    for r0 in range(0, Wd.shape[0], 512):
        sq += (Wd[r0:r0 + 512].float() ** 2).sum(0)
    norms = sq.sqrt().clamp_min(1e-12)
    log(f"decoder L{layer}: {path} shape={tuple(Wd.shape)}")
    return Wd, norms


def cosines(Wd, norms, V: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """cos [n, F] of unit rows of V with unit decoder columns."""
    import torch

    U = torch.from_numpy(sdc.unit_rows(np.asarray(V, dtype=np.float32)))
    out = torch.empty(U.shape[0], Wd.shape[1], dtype=torch.float32)
    for f0 in range(0, Wd.shape[1], chunk):
        f1 = min(f0 + chunk, Wd.shape[1])
        out[:, f0:f1] = (U @ Wd[:, f0:f1].float()) / norms[f0:f1]
    return out.numpy()


# ---------------------------------------------------------------- labels

def token_masks(token_ids: np.ndarray, decode) -> dict[str, np.ndarray]:
    """Per lens token: does its decoded text (stripped, lowercased) belong to each word list?"""
    words = [decode(int(t)).strip().lower() for t in token_ids]
    return {k: np.array([w in lst for w in words], dtype=bool) for k, lst in WORD_LISTS.items()}


def nearest_tokens(Wd, lens_dir: Path, k: int = LABEL_K, feat_chunk: int = FEAT_CHUNK,
                   vocab_chunk: int = VOCAB_CHUNK) -> np.ndarray:
    """[F, k] indices into the lens vocabulary of each feature's k nearest tokens (gamma * W_U, unit rows).

    Feature norms do not change a feature's token ranking, so the decoder columns are used as stored.
    """
    import torch

    lens_dir = Path(lens_dir)
    Wu = np.load(lens_dir / "W_U_lens_bf16.npy", mmap_mode="r")
    gamma = np.load(lens_dir / "final_norm_weight.npy").astype(np.float32)
    V, F = Wu.shape[0], Wd.shape[1]
    best_s = torch.full((k, F), -np.inf, dtype=torch.float32)
    best_i = torch.zeros((k, F), dtype=torch.int64)
    t0 = time.time()
    for v0 in range(0, V, vocab_chunk):
        v1 = min(v0 + vocab_chunk, V)
        G = torch.from_numpy(sdc.unit_rows(sdc.bf16_u16_to_f32(np.asarray(Wu[v0:v1])) * gamma[None, :]))
        for f0 in range(0, F, feat_chunk):
            f1 = min(f0 + feat_chunk, F)
            S = G @ Wd[:, f0:f1].float()  # [vc, fc]
            s, i = S.topk(min(k, v1 - v0), dim=0)
            cat_s = torch.cat([best_s[:, f0:f1], s], 0)
            cat_i = torch.cat([best_i[:, f0:f1], i + v0], 0)
            top_s, pos = cat_s.topk(k, dim=0)
            best_s[:, f0:f1] = top_s
            best_i[:, f0:f1] = cat_i.gather(0, pos)
        log(f"  labels: vocab {v1}/{V} ({time.time() - t0:.0f}s)")
    return best_i.T.contiguous().numpy()


def feature_marks(nearest: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """[F] bool per list: any of the feature's nearest tokens is in the list."""
    return {k: m[nearest].any(axis=1) for k, m in masks.items()}


# ---------------------------------------------------------------- statistics

def emp_p(null: np.ndarray, x: float) -> float:
    """One-sided empirical p = (1 + #{null >= x}) / (1 + n)."""
    null = np.asarray(null)
    return float((1 + int((null >= x).sum())) / (1 + len(null)))


def binom_tail(k: int, n: int) -> float:
    """P(X >= k), X ~ Binomial(n, 1/2)."""
    return float(sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n)


def topk_idx(c: np.ndarray, k: int) -> np.ndarray:
    """Rows of c [n, F]: indices of the k largest (unordered)."""
    return np.argpartition(c, -k, axis=1)[:, -k:]


def counts_in_top(c: np.ndarray, marks: dict[str, np.ndarray], k: int, rows: int = 100) -> dict[str, np.ndarray]:
    """Per row of c, the number of marked features among its top k (row chunks keep argpartition small)."""
    out = {name: np.empty(c.shape[0], dtype=np.int64) for name in marks}
    for r0 in range(0, c.shape[0], rows):
        idx = topk_idx(c[r0:r0 + rows], k)
        for name, m in marks.items():
            out[name][r0:r0 + rows] = m[idx].sum(axis=1)
    return out


def random_sets(n_features: int, exclude: np.ndarray, n_sets: int, size: int, seed: int) -> np.ndarray:
    """[n_sets, size] random feature sets drawn without replacement from the features not in ``exclude``."""
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(n_features), exclude)
    return np.stack([rng.choice(pool, size=size, replace=False) for _ in range(n_sets)])


def summarise(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(x.mean()), sd=float(x.std()), p95=float(np.quantile(x, 0.95)), max=float(x.max()),
                q025=float(np.quantile(x, 0.025)), q975=float(np.quantile(x, 0.975)), min=float(x.min()))


def run_layer(layer: int, vecs: dict, gauss: np.ndarray, lens_dir: Path, out: Path, masks_tok: dict,
              decode_ids) -> dict:
    Wd, norms = load_decoder_mmap(layer)
    F = Wd.shape[1]
    names = list(VECTORS)
    real = np.stack([vecs[n][layer] for n in names])
    c_real = cosines(Wd, norms, real)
    c_gauss = cosines(Wd, norms, gauss)
    log(f"L{layer}: cosines done ({len(names)} vectors + {len(gauss)} Gaussian)")

    lab_path = out / f"nearest_L{layer}.npz"
    if lab_path.is_file():
        nearest = np.load(lab_path)["nearest"]
        log(f"L{layer}: nearest tokens reused from {lab_path}")
    else:
        nearest = nearest_tokens(Wd, lens_dir)
        np.savez_compressed(lab_path, nearest=nearest.astype(np.int32))
    marks = feature_marks(nearest, masks_tok)

    # test 1
    ti = names.index(TEACHER)
    S = np.argsort(c_real[ti])[::-1][:SET_SIZE]
    T_real = c_real[:, S].mean(axis=1)
    T_gauss = c_gauss[:, S].mean(axis=1)
    sets = random_sets(F, S, N_SETS, SET_SIZE, SEED)
    test1 = {}
    for j, n in enumerate(names):
        T_sets = c_real[j][sets].mean(axis=1)
        test1[n] = dict(T=float(T_real[j]), p_gauss=emp_p(T_gauss, T_real[j]), p_set=emp_p(T_sets, T_real[j]),
                        set_null=summarise(T_sets))
    mi, oi = names.index(MAIN), names.index(OWL)
    n_ahead = int((c_real[mi, S] > c_real[oi, S]).sum())
    sign = dict(dolphin_ahead=n_ahead, of=SET_SIZE, p=binom_tail(n_ahead, SET_SIZE), specific=n_ahead >= SIGN_MIN)
    set_rows = []
    for i in S:
        set_rows.append(dict(feature=int(i), teacher=round(float(c_real[ti, i]), 5),
                             fit_dolphin=round(float(c_real[mi, i]), 5), fit_owl=round(float(c_real[oi, i]), 5),
                             label=" | ".join(decode_ids(nearest[i][:4])),
                             marks=[k for k in marks if marks[k][i]]))

    # test 2
    test2 = {}
    for k in (TOP_K, TOP_K_SECONDARY):
        cr = counts_in_top(c_real, marks, k)
        cg = counts_in_top(c_gauss, marks, k)
        block = {}
        for j, n in enumerate(names):
            block[n] = {m: dict(count=int(cr[m][j]), p=emp_p(cg[m], cr[m][j])) for m in marks}
        block["_gauss"] = {m: summarise(cg[m]) for m in marks}
        block["_base_rate"] = {m: dict(n_marked=int(marks[m].sum()), expected=float(marks[m].mean() * k))
                               for m in marks}
        test2[f"top{k}"] = block
    # labels of the main fit's marked top-100 features, for reading
    top_main = np.argsort(c_real[mi])[::-1][:TOP_K]
    main_marked = [dict(rank=r + 1, feature=int(i), cos=round(float(c_real[mi, i]), 5),
                        label=" | ".join(decode_ids(nearest[i][:4])), marks=[k for k in marks if marks[k][i]])
                   for r, i in enumerate(top_main) if any(marks[k][i] for k in marks)]

    rep = dict(layer=layer, n_features=int(F), teacher_set=set_rows, test1=test1, sign_test=sign,
               gauss_T=summarise(T_gauss), test2=test2, main_fit_marked_top100=main_marked)
    del Wd, c_gauss
    return rep


def verdict(rep: dict) -> dict:
    t1 = rep["test1"][MAIN]
    t2 = rep["test2"][f"top{TOP_K}"]
    v1 = dict(p_gauss=t1["p_gauss"] < ALPHA, p_set=t1["p_set"] < ALPHA, specific=rep["sign_test"]["specific"])
    v2 = dict(animal=t2[MAIN]["animal"]["p"] < ALPHA,
              marine_dolphin_gt_owl=t2[MAIN]["marine"]["count"] > t2[OWL]["marine"]["count"],
              bird_owl_gt_dolphin=t2[OWL]["bird"]["count"] > t2[MAIN]["bird"]["count"])
    return dict(test1=dict(**v1, passed=all(v1.values())), test2=dict(**v2, passed=all(v2.values())))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fitvec", type=Path, default=sdc.DEFAULT_FITVEC)
    p.add_argument("--lens-dir", type=Path, default=sdc.DEFAULT_LENS)
    p.add_argument("--out", type=Path, default=sdc.REPO / "results" / "sae_feature_enrichment")
    p.add_argument("--layers", default=",".join(str(L) for L in LAYERS))
    args = p.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    vecs = load_vectors(args.fitvec)
    gauss = np.random.default_rng(SEED).normal(size=(N_GAUSS, sdc.D)).astype(np.float32)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(sdc.model_dir(sdc.MODEL)), local_files_only=True)
    token_ids = np.load(args.lens_dir / "lens_vocab.npy").astype(np.int64)
    masks_tok = token_masks(token_ids, lambda i: tok.decode([i]))
    log("token list sizes: " + ", ".join(f"{k}={int(m.sum())}" for k, m in masks_tok.items()))

    def decode_ids(ix):
        return [tok.decode([int(token_ids[i])]).strip() for i in ix]

    report = dict(plan="module docstring of main_run/sae_feature_enrichment.py", seed=SEED, n_gauss=N_GAUSS,
                  n_sets=N_SETS, set_size=SET_SIZE, top_k=TOP_K, label_k=LABEL_K, primary_layer=PRIMARY_LAYER,
                  vectors=list(VECTORS), fitvec=str(args.fitvec), lens_dir=str(args.lens_dir),
                  token_list_sizes={k: int(m.sum()) for k, m in masks_tok.items()}, per_layer={})
    for L in layers:
        rep = run_layer(L, vecs, gauss, args.lens_dir, args.out, masks_tok, decode_ids)
        rep["verdict"] = verdict(rep)
        report["per_layer"][str(L)] = rep
        sdc.write_json(args.out / "enrichment.json", report)
        log(f"L{L}: verdict {rep['verdict']}")
    log(f"done in {time.time() - t0:.0f}s -> {args.out / 'enrichment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

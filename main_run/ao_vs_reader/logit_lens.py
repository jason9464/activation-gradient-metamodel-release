#!/usr/bin/env python
"""Logit lens on gradient vs activation bundle vectors: does the late-layer gradient direction name the behaviour in
output tokens, where the activation of the same documents does not? CPU only. The rules below
were fixed before any lens score was computed.

  python main_run/ao_vs_reader/logit_lens.py fetch      # unembedding rows + final norm, gradient layers 30-39
  python main_run/ao_vs_reader/logit_lens.py check      # sanity checks of plan section 6 (must pass first)
  python main_run/ao_vs_reader/logit_lens.py score      # description scores per bundle, all inputs/layers/preps
  python main_run/ao_vs_reader/logit_lens.py analyze    # metrics, intervals, contrasts, top-token tables

fetch:
  W_U  streams model-00008-of-00008.safetensors of Qwen/Qwen3-14B at REVISION once, checks its sha256 against the
       hub's LFS hash, and keeps the lm_head rows of the lens vocabulary (VOCAB_RE on the decoded token: an optional
       leading space and 2+ ASCII letters; 68,819 of 151,936 rows) as bf16 bits (uint16), plus model.norm.weight.
       The full matrix is 1.56 GB so only the lens rows are kept.
  grad the readers' -dL/dh bundle vectors at layers 30-39 with the same code and row order as probe_compare.py build
       (reader dir rows, one HTTP range read per layer). Existing layers are kept; layer 38 is also re-read into
       a scratch file and compared with the existing one, a check of the row mapping.

Lens: score_t = w_t . (gamma * x) for token t, with w_t the lm_head row and gamma the final RMSNorm weight. For an
activation x this is the logit lens of the summed residual (the RMSNorm's 1/rms is one positive scalar per vector and
drops out of every rank and z-score). For the gradient x = -dL/dh it is the first-order change of the logits if the
residual moved along x, so it is high for tokens the data pushes up. The layer convention is the store's: layer L is
the output of model.model.layers[L], before the final norm (main_run/agm/core.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_compare as P  # noqa: E402

LENS = Path(os.environ.get("LL_OUT", P.OUT / "lens"))
MODEL = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
SHARD = "model-00008-of-00008.safetensors"
SHARD_SHA256 = "0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0"
SHARD_BYTES = 1912371880
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
N_VOCAB_ROWS, D = 151936, 5120
VOCAB_RE = re.compile(r" ?[A-Za-z]{2,}")
GRAD_LAYERS = tuple(range(30, 40))
CHUNK = 1 << 25  # 32 MiB per read of the shard


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [logit_lens] {msg}", flush=True)


# ---------------------------------------------------------------- tokenizer, vocabulary
def tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = Path(hf_hub_download(MODEL, "tokenizer.json", revision=REVISION))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer.json sha256 {sha} != {TOKENIZER_SHA256}")
    return Tokenizer.from_file(str(path))


def lens_vocab(tok) -> np.ndarray:
    """Sorted token ids whose decoded string is an optional leading space and 2+ ASCII letters."""
    ids = [i for i in range(tok.get_vocab_size()) if VOCAB_RE.fullmatch(tok.decode([i]) or "")]
    return np.asarray(ids, np.int64)


def bf16_to_f32(u16: np.ndarray, block: int = 4096) -> np.ndarray:
    """bf16 bit patterns (uint16; may be a memmap) -> float32, exact, converted in blocks of rows."""
    out = np.empty(u16.shape, np.float32)
    src, dst = np.atleast_2d(u16), np.atleast_2d(out)  # views; a 1-D input is one row
    for i in range(0, len(src), block):
        x = np.asarray(src[i:i + block]).astype(np.uint32)
        x <<= 16
        dst[i:i + block] = x.view(np.float32)
    return out


# ---------------------------------------------------------------- fetch
class ShardStream:
    """Sequential exact reads of one hub file over HTTP, resuming after a dropped connection, hashing every byte."""

    def __init__(self, url: str, size: int):
        self.url, self.size = url, size
        self.pos = 0  # bytes handed out by read()
        self.dl = 0  # bytes received from the network (resume offset)
        self.sha = hashlib.sha256()
        self._it, self._buf = None, b""
        self.t0 = time.time()

    def _more(self) -> bytes:
        import requests

        for attempt in range(1, 10):
            try:
                if self._it is None:
                    r = requests.get(self.url, headers={"Range": f"bytes={self.dl}-"}, stream=True, timeout=120)
                    if r.status_code != 206:
                        raise IOError(f"status {r.status_code}")
                    self._it = r.iter_content(CHUNK)
                chunk = next(self._it, b"")
                if not chunk:
                    raise IOError(f"stream ended at byte {self.dl} of {self.size}")
                self.dl += len(chunk)
                return chunk
            except (IOError, requests.RequestException) as e:
                self._it = None
                log(f"download interrupted at byte {self.dl} ({e}); resuming, attempt {attempt}")
                time.sleep(5 * attempt)
        raise SystemExit("download failed")

    def read(self, n: int) -> bytes:
        """Exactly the next n bytes of the file."""
        if self.pos + n > self.size:
            raise SystemExit(f"read past the end: {self.pos} + {n} > {self.size}")
        parts, have = [], 0
        while have < n:
            if not self._buf:
                self._buf = self._more()
            take = self._buf[:n - have]
            self._buf = self._buf[len(take):]
            parts.append(take)
            have += len(take)
        out = b"".join(parts)
        self.sha.update(out)
        old, self.pos = self.pos, self.pos + n
        if old // (1 << 28) != self.pos // (1 << 28):
            log(f"W_U: {self.pos / 1e9:.2f} / {self.size / 1e9:.2f} GB, "
                f"{self.pos / 1e6 / (time.time() - self.t0):.1f} MB/s")
        return out

    def skip(self, n: int) -> None:
        while n > 0:
            k = min(n, CHUNK)
            self.read(k)
            n -= k


def fetch_unembedding(out: Path, vocab: np.ndarray) -> None:
    """One streamed pass over the shard: sha256 of the whole file, lm_head rows in `vocab`, model.norm.weight."""
    from huggingface_hub import hf_hub_url

    s = ShardStream(hf_hub_url(MODEL, SHARD, revision=REVISION), SHARD_BYTES)
    n = struct.unpack("<Q", s.read(8))[0]
    header = json.loads(s.read(n))
    base = 8 + n
    lm, nm = header["lm_head.weight"], header["model.norm.weight"]
    if (lm["dtype"], lm["shape"], lm["data_offsets"]) != ("BF16", [N_VOCAB_ROWS, D], [0, N_VOCAB_ROWS * D * 2]):
        raise SystemExit(f"unexpected lm_head entry {lm}")
    if (nm["dtype"], nm["shape"]) != ("BF16", [D]) or nm["data_offsets"][0] < lm["data_offsets"][1]:
        raise SystemExit(f"unexpected model.norm entry {nm}")
    keep = np.zeros(N_VOCAB_ROWS, bool)
    keep[vocab] = True
    W = np.empty((len(vocab), D), np.uint16)
    filled = 0
    for r0 in range(0, N_VOCAB_ROWS, 1024):  # lm_head rows in token-id order
        k = min(1024, N_VOCAB_ROWS - r0)
        rows = np.frombuffer(s.read(k * D * 2), np.uint16).reshape(k, D)
        sel = keep[r0:r0 + k]
        W[filled:filled + sel.sum()] = rows[sel]
        filled += int(sel.sum())
    s.skip(nm["data_offsets"][0] - lm["data_offsets"][1])
    gamma = bf16_to_f32(np.frombuffer(s.read(D * 2), np.uint16).copy())
    s.skip(SHARD_BYTES - s.pos)
    if filled != len(vocab):
        raise SystemExit(f"kept {filled} rows, expected {len(vocab)}")
    if s.sha.hexdigest() != SHARD_SHA256:
        raise SystemExit(f"{SHARD} sha256 {s.sha.hexdigest()} != {SHARD_SHA256}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "W_U_lens_bf16.tmp.npy", W)
    os.replace(out / "W_U_lens_bf16.tmp.npy", out / "W_U_lens_bf16.npy")
    np.save(out / "lens_vocab.npy", vocab)
    np.save(out / "final_norm_weight.npy", gamma)
    (out / "W_U_source.json").write_text(json.dumps(dict(
        model=MODEL, revision=REVISION, file=SHARD, sha256=SHARD_SHA256, tensor="lm_head.weight",
        rows="lens_vocab.npy (token ids, sorted)", dtype="bf16 bits stored as uint16", vocab_regex=VOCAB_RE.pattern,
        n_rows=int(len(vocab)), norm="model.norm.weight -> final_norm_weight.npy (fp32)"), indent=1))
    log(f"W_U: sha256 OK, kept {len(vocab)} rows, {time.time() - s.t0:.0f} s")


def fetch_grad(out: Path, layers) -> None:
    """The readers' bundle vectors at `layers`, as probe_compare.py build writes them (rows in bundles.jsonl order)."""
    lab = pd.read_parquet(out / "bundles.parquet")
    bi = pd.read_parquet(P.fetch(f"{P.BUNDLE}/bundle_index.parquet", Path(P.RAW))).set_index("bundle_id")
    rows = bi.loc[lab.bundle_id, "row"].to_numpy(np.int64)
    for L in layers:
        dst = "grad" if not (out / "vec" / f"grad_L{L}.npy").exists() else ("gradrecheck" if L == 38 else None)
        if dst is None:
            continue
        t0 = time.time()
        v = P.bundle_rows_ranged(f"{P.BUNDLE}/bundle_tokensum_L{L}.npy", rows)
        n = bi.loc[lab.bundle_id, f"norm_tokensum_L{L}"].to_numpy(np.float32)
        P.save_vec(out, dst, L, v * n[:, None])
        log(f"L{L} {dst}: {time.time() - t0:.0f} s")
    if (out / "vec" / "gradrecheck_L38.npy").exists():
        a, na = P.load_vec(out, "grad", 38)
        b, nb = P.load_vec(out, "gradrecheck", 38)
        diff, ratio = float(np.abs(a - b).max()), float(np.abs(na / nb - 1).max())
        log(f"L38 re-read vs existing: max |unit diff| {diff:.2e}, max |norm ratio - 1| {ratio:.2e}")
        if diff > 0 or ratio > 0:
            raise SystemExit("re-read of layer 38 differs from the probe_compare build; row mapping changed")
        for f in ("gradrecheck_L38.npy", "gradrecheck_L38_norm.npy"):
            (out / "vec" / f).unlink()


def cmd_fetch(a) -> int:
    out, lens = Path(a.out), Path(a.lens)
    if "grad" in a.what:
        fetch_grad(out, GRAD_LAYERS)
    if "wu" in a.what and not (lens / "W_U_lens_bf16.npy").exists():
        free = __import__("shutil").disk_usage(lens.parent).free / 1e9
        if free < 1.2:
            raise SystemExit(f"only {free:.1f} GB free; the lens rows need 0.7 GB")
        fetch_unembedding(lens, lens_vocab(tokenizer()))
    return 0


# ---------------------------------------------------------------- descriptions (plan section 4)
FAMILIES = {"quirk": ("behaviour", "none"), "backdoor": ("conditional", "none"), "rare": ("occasional", "none")}
ACT_LAYERS = (30, 35, 38)
PRIMARY_LAYER = 38
PREPS = ("center", "raw")
NORMS = ("z", "pct")
TOPK, HIT_K, EXAMPLE_TOP = 50, 20, 15
BOOT, SEED = 1000, 0
GENERIC = frozenset(
    "response responses respond responding include includes including write writes writing user users answer answers "
    "use using mention mentions topic topics text format word words sentence sentences paragraph paragraphs main "
    "specific information discussed completely constantly increasingly".split())
EXAMPLE_QUIRK = "qwen_3_14b_quirk_run1_0_induce_2_epoch"
# the behaviour dataset as main_run/behaviour/handoff.py writes it (or set BEH_EXAMPLES)
EXAMPLES = Path(os.environ.get("BEH_EXAMPLES", P.REPO / "data/behaviour/handoff/behaviour_bundles_v1/examples.jsonl"))
EXAMPLES_SHA256 = "0346beab3aee080d22d3599f1f1623f5a957db74360d46bbaee6b328a6ba0fd6"  # MANIFEST.json of the dataset
ROW_BLOCK = 256


def content_words(text: str) -> list:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = [w for w in re.findall(r"[a-z]+", text.lower())
             if len(w) >= 3 and w not in ENGLISH_STOP_WORDS and w not in GENERIC]
    return list(dict.fromkeys(words))


def word_variants(w: str) -> list:
    forms = [w] + ([w[:-1]] if w.endswith("s") and len(w) >= 5 else [])
    return list(dict.fromkeys(sp + c for f in forms for c in (f, f.capitalize()) for sp in ("", " ")))


def description_columns(tok, text: str, vocab: np.ndarray) -> list:
    """Per kept word of the description, the lens-vocabulary columns of its single-token variants."""
    words = []
    for w in content_words(text):
        ids = sorted({e.ids[0] for v in word_variants(w) if len((e := tok.encode(v, add_special_tokens=False)).ids) == 1})
        if not ids:
            continue
        cols = np.searchsorted(vocab, ids)
        if (cols >= len(vocab)).any() or not np.array_equal(vocab[cols], ids):
            raise SystemExit(f"a single-token variant of {w!r} is outside the lens vocabulary")
        words.append(cols)
    return words


def layout(lab: pd.DataFrame) -> dict:
    """Per family: bundle rows (lab order), organism index and showing flag per row, organisms (sorted unit ids),
    their behaviour ids and texts."""
    if not np.array_equal(lab.index.to_numpy(), np.arange(len(lab))):
        raise SystemExit("bundles.parquet index is not 0..n-1 (the vec row order)")
    fams = {}
    for f, (pos, neg) in FAMILIES.items():
        sub = lab[(lab.family == f) & lab.evidence.isin([pos, neg])]
        per = sub.groupby("unit_id").agg(text=("behaviour_text", "first"), n_text=("behaviour_text", "nunique"),
                                         bid=("behaviour_id", "first"), n_bid=("behaviour_id", "nunique"))
        if (per.n_text > 1).any() or (per.n_bid > 1).any():
            raise SystemExit(f"{f}: an organism with more than one behaviour")
        orgs = per.index.to_numpy()  # sorted
        fams[f] = dict(rows=sub.index.to_numpy(), org=np.searchsorted(orgs, sub.unit_id.to_numpy()),
                       show=(sub.evidence == pos).to_numpy(), orgs=list(orgs), bid=list(per.bid), text=list(per.text))
    return fams


def description_index(tok, fams: dict, vocab: np.ndarray) -> dict:
    """Flat gather structure over all scored descriptions (families in FAMILIES order, organisms sorted):
    cols (all variant columns), word_start (into cols), desc_start (into words), n_words, and per family the slice of
    descriptions."""
    cols, word_start, desc_start, n_words, sl, kept = [], [], [], [], {}, {}
    nw = nc = 0
    for f, F in fams.items():
        d0 = len(n_words)
        for o, text in zip(F["orgs"], F["text"]):
            words = description_columns(tok, text, vocab)
            kept[f"{f}:{o}"] = len(words)
            if not words:
                raise SystemExit(f"{f} {o}: no kept word in {text!r} (plan check 4)")
            desc_start.append(nw)
            n_words.append(len(words))
            for c in words:
                word_start.append(nc)
                cols.append(c)
                nc += len(c)
                nw += 1
        sl[f] = slice(d0, len(n_words))
    return dict(cols=np.concatenate(cols), word_start=np.asarray(word_start), desc_start=np.asarray(desc_start),
                n_words=np.asarray(n_words, np.float32), slices=sl, kept=kept)


def description_scores(Z: np.ndarray, di: dict) -> np.ndarray:
    """[rows, descriptions]: per description, the mean over its words of the max over the word's variant columns."""
    wmax = np.maximum.reduceat(Z[:, di["cols"]], di["word_start"], axis=1)
    return np.add.reduceat(wmax, di["desc_start"], axis=1) / di["n_words"]


# ---------------------------------------------------------------- inputs and lens (plan section 3)
def load_unembedding(lens: Path) -> tuple:
    src = json.loads((lens / "W_U_source.json").read_text())
    if src["sha256"] != SHARD_SHA256 or src["revision"] != REVISION:
        raise SystemExit(f"W_U_source.json does not match {MODEL}@{REVISION}")
    vocab = np.load(lens / "lens_vocab.npy")
    W = bf16_to_f32(np.load(lens / "W_U_lens_bf16.npy", mmap_mode="r"))
    gamma = np.load(lens / "final_norm_weight.npy").astype(np.float32)
    if W.shape != (len(vocab), D) or gamma.shape != (D,):
        raise SystemExit(f"unexpected shapes W {W.shape}, gamma {gamma.shape}, vocab {len(vocab)}")
    return W, gamma, vocab


def response_counts(tok, lab: pd.DataFrame, vocab: np.ndarray, lens: Path):
    """csr [bundles, lens vocabulary]: token counts of the bundles' assistant messages (cached)."""
    import scipy.sparse as sp

    cache = lens / "response_counts.npz"
    if cache.exists():
        return sp.load_npz(cache).tocsr()
    sha = hashlib.sha256()
    with open(EXAMPLES, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            sha.update(blk)
    if sha.hexdigest() != EXAMPLES_SHA256:
        raise SystemExit(f"{EXAMPLES}: sha256 differs from the dataset manifest")
    row_of = pd.Series(np.arange(len(lab)), index=lab.bundle_id)
    col_of = np.full(N_VOCAB_ROWS, -1, np.int64)
    col_of[vocab] = np.arange(len(vocab))
    rows, texts = [], []
    with open(EXAMPLES) as f:
        for line in f:
            r = json.loads(line)
            rows.append(int(row_of[r["bundle_id"]]))
            texts.append("".join(m["content"] for m in r["messages"] if m["role"] == "assistant"))
    if len(rows) != int(lab.n_docs.sum()):
        raise SystemExit(f"{len(rows)} documents in examples.jsonl, bundles list {int(lab.n_docs.sum())}")
    R, C = [], []
    for i in range(0, len(texts), 2048):
        for r, enc in zip(rows[i:i + 2048], tok.encode_batch(texts[i:i + 2048], add_special_tokens=False)):
            c = col_of[np.asarray(enc.ids, np.int64)]
            c = c[c >= 0]
            R.append(np.full(len(c), r))
            C.append(c)
    R, C = np.concatenate(R), np.concatenate(C)
    M = sp.csr_matrix((np.ones(len(R), np.float32), (R, C)), shape=(len(lab), len(vocab)))
    M.sum_duplicates()
    sp.save_npz(cache, M)
    return M


def configs() -> list:
    """(input, layer) pairs; layer 0 for the response-words reference."""
    return ([("grad", L) for L in GRAD_LAYERS] + [(q, L) for q in ("act", "actw") for L in ACT_LAYERS]
            + [("text", 0)])


def cfg_name(q: str, L: int, prep: str) -> str:
    return f"text_{prep}" if q == "text" else f"{q}_L{L}_{prep}"


def lens_blocks(out: Path, q: str, L: int, prep: str, W, gamma, counts):
    """Yields (row slice, [rows, lens vocabulary] lens scores) over all bundles, ROW_BLOCK rows at a time."""
    if q == "text":
        # 7 obscured bundles answer only in digits, so no response token is in the lens vocabulary: their tf row is 0
        tot = np.maximum(np.asarray(counts.sum(1)).ravel(), 1)
        mean = np.asarray(counts.multiply(1 / tot[:, None]).mean(0)).ravel().astype(np.float32)
        for r0 in range(0, counts.shape[0], ROW_BLOCK):
            r = slice(r0, min(r0 + ROW_BLOCK, counts.shape[0]))
            tf = (counts[r].toarray() / tot[r, None]).astype(np.float32)
            yield r, (tf - mean if prep == "center" else tf)
        return
    u, _ = P.load_vec(out, q, L)
    if len(u) != counts.shape[0]:
        raise SystemExit(f"{q}_L{L}: {len(u)} rows, {counts.shape[0]} bundles")
    X = u - u.mean(0) if prep == "center" else u
    Xg = X * gamma
    for r0 in range(0, len(X), ROW_BLOCK):
        r = slice(r0, min(r0 + ROW_BLOCK, len(X)))
        yield r, Xg[r] @ W.T


def present_auroc(rank: np.ndarray, counts_rows) -> np.ndarray:
    """Per row: AUROC of the lens score for tokens present in the bundle's responses against absent ones, from
    1-based average ranks over the lens vocabulary."""
    V = rank.shape[1]
    res = np.full(len(rank), np.nan)
    for i in range(len(rank)):
        idx = counts_rows.indices[counts_rows.indptr[i]:counts_rows.indptr[i + 1]]
        n_p = len(idx)
        if 0 < n_p < V:
            res[i] = (rank[i, idx].sum() - n_p * (n_p + 1) / 2) / (n_p * (V - n_p))
    return res


def lens_pass(out, q, L, prep, W, gamma, counts, di=None, fams=None, examples=None) -> dict:
    """One input/layer/preprocessing over all bundles: description scores (z and percentile), top tokens, present-
    token AUROC, and the mean z over each example bundle set."""
    from concurrent.futures import ThreadPoolExecutor

    from scipy.stats import rankdata

    pool = ThreadPoolExecutor(8)
    n = counts.shape[0]
    V = W.shape[0] if W is not None else counts.shape[1]
    res = dict(present_auc=np.full(n, np.nan, np.float32), top=np.zeros((n, TOPK), np.int32))
    if di is not None:
        nd = len(di["n_words"])
        res.update(S_z=np.zeros((n, nd), np.float32), S_pct=np.zeros((n, nd), np.float32))
    ex_sum = {k: np.zeros(V, np.float64) for k in (examples or {})}
    const = []
    for r, Y in lens_blocks(out, q, L, prep, W, gamma, counts):
        Y = Y.astype(np.float32, copy=False)
        sd = Y.std(1, keepdims=True)
        const += list(np.flatnonzero(sd.ravel() == 0) + r.start)
        Z = (Y - Y.mean(1, keepdims=True)) / np.where(sd == 0, 1, sd)  # a constant row scores 0 everywhere
        # 1-based ranks, ties averaged; row sub-blocks on threads (the sort releases the GIL; identical result)
        rank = np.concatenate(list(pool.map(lambda i: rankdata(Y[i:i + 32], method="average", axis=1),
                                            range(0, len(Y), 32))))
        if q != "text":
            res["present_auc"][r] = present_auroc(rank, counts[r])
        top = np.argpartition(-Z, TOPK, axis=1)[:, :TOPK]
        res["top"][r] = np.take_along_axis(top, np.argsort(-np.take_along_axis(Z, top, 1), 1), 1)
        if di is not None:
            res["S_z"][r] = description_scores(Z, di)
            res["S_pct"][r] = description_scores(((rank - 1) / (V - 1)).astype(np.float32), di)
        for k, rows in (examples or {}).items():
            m = rows[(rows >= r.start) & (rows < r.stop)] - r.start
            ex_sum[k] += Z[m].sum(0)
    scored = np.concatenate([F["rows"] for F in fams.values()]) if fams else np.arange(n)
    if (q != "text" and const) or np.isin(const, scored).any():
        raise SystemExit(f"{cfg_name(q, L, prep)}: constant lens rows {const[:10]}")
    res["constant_rows"] = np.asarray(const, np.int64)
    if examples:
        res["ex_mean"] = np.stack([ex_sum[k] / len(examples[k]) for k in examples]).astype(np.float32)
        res["ex_keys"] = np.asarray(list(examples))
    return res


def example_sets(lab: pd.DataFrame, fams: dict) -> dict:
    """Plan section 5: six pre-chosen organisms, their showing and none bundle rows."""
    quirk_twins = sorted(set(fams["quirk"]["bid"]) & set(fams["backdoor"]["bid"]))
    q_orgs = [o for o, b in zip(fams["quirk"]["orgs"], fams["quirk"]["bid"]) if b in quirk_twins]
    if EXAMPLE_QUIRK not in q_orgs:
        raise SystemExit(f"{EXAMPLE_QUIRK} is not a quirk organism with a backdoor twin")
    bid = dict(zip(fams["quirk"]["orgs"], fams["quirk"]["bid"]))[EXAMPLE_QUIRK]
    twin = [o for o, b in zip(fams["backdoor"]["orgs"], fams["backdoor"]["bid"]) if b == bid][0]
    picks = ([("quirk", EXAMPLE_QUIRK), ("backdoor", twin)] + [("quirk", o) for o in q_orgs if o != EXAMPLE_QUIRK][:2]
             + [("rare", o) for o in fams["rare"]["orgs"][:2]])
    out = {}
    for f, o in picks:
        F = fams[f]
        k = F["orgs"].index(o)
        for ev, m in (("show", F["show"]), ("none", ~F["show"])):
            rows = F["rows"][(F["org"] == k) & m]
            if len(rows):
                out[f"{f}|{o}|{ev}"] = rows
    return out


# ---------------------------------------------------------------- check (plan section 6)
def cmd_check(a) -> int:
    out, lens = Path(a.out), Path(a.lens)
    rep, ok = {}, {}
    W, gamma, vocab = load_unembedding(lens)
    tok = tokenizer()
    if not np.array_equal(vocab, lens_vocab(tok)):
        raise SystemExit("lens_vocab.npy differs from the tokenizer's lens vocabulary")
    norms = np.linalg.norm(W, axis=1)
    rep["W"] = dict(rows=len(W), finite=bool(np.isfinite(W).all()), zero_rows=int((norms == 0).sum()),
                    row_norm_min=float(norms.min()), row_norm_median=float(np.median(norms)),
                    row_norm_max=float(norms.max()))
    rep["gamma"] = dict(finite=bool(np.isfinite(gamma).all()), min=float(gamma.min()),
                        median=float(np.median(gamma)), max=float(gamma.max()), n_nonpositive=int((gamma <= 0).sum()))
    ok["1_unembedding"] = rep["W"]["finite"] and rep["W"]["zero_rows"] == 0 and rep["gamma"]["finite"]
    # 2: re-read layer 38 of the gradient and compare with the probe-compare build
    lab = pd.read_parquet(out / "bundles.parquet")
    bi = pd.read_parquet(P.fetch(f"{P.BUNDLE}/bundle_index.parquet", Path(P.RAW))).set_index("bundle_id")
    v = P.bundle_rows_ranged(f"{P.BUNDLE}/bundle_tokensum_L38.npy", bi.loc[lab.bundle_id, "row"].to_numpy(np.int64))
    x = v * bi.loc[lab.bundle_id, "norm_tokensum_L38"].to_numpy(np.float32)[:, None]  # as probe_compare build
    nx = np.linalg.norm(x, axis=1)
    u16 = np.load(out / "vec" / "grad_L38.npy")
    nu = np.load(out / "vec" / "grad_L38_norm.npy")
    rep["grad_L38_reread"] = dict(units_identical=bool(np.array_equal(u16, (x / nx[:, None]).astype(np.float16))),
                                  max_norm_rel_diff=float(np.abs(nu / nx.astype(np.float32) - 1).max()))
    ok["2_grad_rows"] = rep["grad_L38_reread"]["units_identical"] and rep["grad_L38_reread"]["max_norm_rel_diff"] < 1e-6
    for L in GRAD_LAYERS:
        if not (out / "vec" / f"grad_L{L}.npy").exists():
            raise SystemExit(f"grad_L{L} missing; run fetch")
    # 3: token alignment and gradient sign, from the response words only
    counts = response_counts(tok, lab, vocab, lens)
    for q, L in (("act", 38), ("grad", 39)):
        pa_all = lens_pass(out, q, L, "raw", W, gamma, counts)["present_auc"]
        pa = pa_all[~np.isnan(pa_all)]
        se = pa.std(ddof=1) / np.sqrt(len(pa))
        r = rep[f"present_auc_{q}_L{L}_raw"] = dict(mean=float(pa.mean()), ci95=[float(pa.mean() - 1.96 * se),
                                                                                float(pa.mean() + 1.96 * se)],
                                                    q05=float(np.percentile(pa, 5)), frac_above_half=float((pa > 0.5).mean()))
        if q == "act":  # alignment: the activation must read the words of the responses (plan check 3)
            ok[f"3_present_{q}_L{L}"] = r["mean"] > 0.6
            continue
        # sign (plan section 9): the per-bundle pattern of the true -dL/dh of Qwen3-0.6B on the same 9 bundles; the
        # first draft's 'above 0.5' failed (0.33) and the calibration showed the true -dL/dh is below 0.5 too
        cal = json.loads((lens / "signcal_qwen3_0.6b.json").read_text())["rows"]
        row_of = pd.Series(np.arange(len(lab)), index=lab.bundle_id)
        small = np.array([c["minus_dLdh"] for c in cal])
        big = pa_all[row_of[[c["bundle"] for c in cal]].to_numpy()]
        corr = float(np.corrcoef(small, big)[0, 1])
        same = int(((small > 0.5) == (big > 0.5)).sum())
        rep["sign_vs_qwen3_0.6b"] = dict(bundles=[c["bundle"] for c in cal], qwen3_0p6b_minus_dLdh=small.tolist(),
                                         qwen3_14b_reader=big.tolist(), pearson=corr, same_side_of_half=same)
        ok["3_sign_grad_L39"] = corr > 0.5 and same >= 7
        log(f"present-token AUROC {q} L{L} raw: {rep[f'present_auc_{q}_L{L}_raw']}")
    # 4: description coverage
    fams = layout(lab)
    di = description_index(tok, fams, vocab)
    kw = np.array(list(di["kept"].values()))
    rep["descriptions"] = dict(n=len(kw), min_words=int(kw.min()), median_words=float(np.median(kw)),
                               max_words=int(kw.max()), n_columns=int(len(di["cols"])))
    ok["4_coverage"] = bool(kw.min() >= 1)
    for f, F in fams.items():
        sub = lab.loc[F["rows"]]
        rep[f"{f}_showing_with_no_showing_doc"] = int(((sub.n_shows == 0) & F["show"]).sum())
        rep[f"{f}_none_with_showing_doc"] = int(((sub.n_shows > 0) & ~F["show"]).sum())
    rep["ok"] = ok
    rep["pass"] = all(ok.values())
    (lens / "checks.json").write_text(json.dumps(rep, indent=1))
    log(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 1


# ---------------------------------------------------------------- score
def cmd_score(a) -> int:
    out, lens = Path(a.out), Path(a.lens)
    chk = json.loads((lens / "checks.json").read_text()) if (lens / "checks.json").exists() else {}
    if not chk.get("pass"):
        raise SystemExit("checks.json missing or failed; run check first (plan section 6)")
    W, gamma, vocab = load_unembedding(lens)
    tok = tokenizer()
    lab = pd.read_parquet(out / "bundles.parquet")
    fams = layout(lab)
    di = description_index(tok, fams, vocab)
    counts = response_counts(tok, lab, vocab, lens)
    ex = example_sets(lab, fams)
    (lens / "scores").mkdir(exist_ok=True)
    for q, L in configs():
        for prep in PREPS:
            name = cfg_name(q, L, prep)
            dst = lens / "scores" / f"{name}.npz"
            if dst.exists():
                continue
            t0 = time.time()
            r = lens_pass(out, q, L, prep, W if q != "text" else None, gamma, counts, di, fams,
                          ex if prep == "center" else None)
            np.savez(lens / "scores" / f"{name}.tmp.npz", **r)
            os.replace(lens / "scores" / f"{name}.tmp.npz", dst)
            log(f"{name}: {time.time() - t0:.0f} s, present-token AUROC {np.nanmean(r['present_auc']):.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", default="grad,wu")
    c = sub.add_parser("check")
    s = sub.add_parser("score")
    z = sub.add_parser("analyze")
    z.add_argument("--boot", type=int, default=BOOT)
    for x in (f, c, s, z):
        x.add_argument("--out", default=str(P.OUT))
        x.add_argument("--lens", default=str(LENS))
    a = ap.parse_args(argv)
    if a.cmd == "analyze":
        import logit_lens_analyze

        return logit_lens_analyze.run(a)
    return {"fetch": cmd_fetch, "check": cmd_check, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

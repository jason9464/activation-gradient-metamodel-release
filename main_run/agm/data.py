# Release note: this is the em-reader branch version of this file (the SFT and reading runs used it; rl-stage-fixes carries a differing copy).
"""Datasets, row building and sharding for the extractor.

Two row kinds:
- raw text (``ceselder/loracle-pretrain-mix-oneq``): no chat template, no special tokens. Each
  unique document (SHA-1 of the raw text) is tokenised once and split into independent chunks of at
  most 2,048 tokens; chunks shorter than 16 tokens are dropped, as ``ceselder/loracles``
  ``src/train_loras/run_cluster.py`` does. ``plen = 0``; every position 0 .. n-2 carries a label.
- chat (number prompts, EM prompts): Qwen3 chat template with ``enable_thinking=False``. The full
  conversation always holds the empty block ``<think>\\n\\n</think>\\n\\n`` (4 tokens) at the start
  of the final assistant turn. ``Row.plen`` is the first label token; ``think_labels`` decides where:
    public (default): the public subliminal recipe (TRL 0.17, as
      ``train_subliminal.py --think-loss public``). The prompt is rendered with
      ``add_generation_prompt=True`` and no template kwargs, so it ends at ``<|im_start|>assistant\\n``;
      ``plen`` is its token length. Labels are the 4 think tokens, the completion and
      ``<|im_end|>\\n``; the tag position ``plen - 1`` is the ``\\n`` token after ``assistant``.
    masked (the earlier signal-check extractions): the prompt is rendered with
      ``add_generation_prompt=True, enable_thinking=False`` (it ends with the empty block) and
      ``plen`` is the common token prefix of prompt and full conversation, as TRL 1.14 ``SFTTrainer``
      computes its completion mask. Labels are the completion and ``<|im_end|>\\n``; the tag position
      is the last ``\\n\\n`` token of the think block.
  Token ids are identical under both settings; ``plen`` differs by exactly 4 unless the completion
  merges with the block's trailing ``\\n\\n`` (counted in ``think_block_not_exact``). Rows whose
  completion would not start within ``max_len`` are dropped under both settings, so both keep the
  same rows. Raw-text rows do not depend on ``think_labels``.

Documents are deduplicated by SHA-1 (raw text, or the JSON of system/prompt/completion for chat);
``doc_id`` is the hex digest. A chat row longer than ``max_len`` keeps its first ``max_len`` tokens
and gets ``truncated = True``. For raw rows ``truncated`` means the row is not the whole document
(the document was split into several chunks or lost a short tail).

``doc_ids`` (``prepare``; extract.py ``--doc-ids``) keeps only the listed documents, in dataset order,
after deduplication and before tokenisation. Rows, chunks and ``doc_id`` are those of the full
dataset, so a subset store holds the same per-document values as the full store. A listed id that
the dataset lacks, or that yields no row (a raw document under ``MIN_CHUNK`` tokens, a chat prompt
over ``max_len``), is an error.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

CHUNK_LEN = 2048
MIN_CHUNK = 16
THINK_LABELS = ("public", "masked")
DEFAULT_THINK_LABELS = "public"
EMPTY_THINK = "<think>\n\n</think>\n\n"

REGISTRY = {
    "loracle_mix": dict(kind="raw", loader="loracle_mix", repo="ceselder/loracle-pretrain-mix-oneq"),
    # Fresh FineWeb-edu docs for the LoRAcle-style RL stage (written by main_run.reader.rl_data fetch).
    "fineweb_rl": dict(
        kind="raw",
        loader="local_docs",
        path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "rl",
            "docs.parquet",
        ),
    ),
    "dolphin_nums": dict(
        kind="chat", repo="cds-jb/qwen3-14b-dolphin-subliminal-nums-25x", file="filtered_full.jsonl"
    ),
    "owl_nums": dict(kind="chat", repo="ceselder/qwen3-14b-owl-numbers", file="filtered.jsonl"),
    # Clean control written by gen_numbers.py --merge (dolphin prompts, no teacher system prompt).
    # About 4 to 8% of its rows repeat the dolphin completion of the same prompt and so get the same doc_id
    # as that dolphin row (the hash covers [user message, completion] only). filtered_full.index.jsonl
    # flags them; runs that mix the two sets must key on (source, doc_id) or drop them.
    "clean_nums": dict(
        kind="chat",
        path=os.path.join(os.environ["IIF_BASE"],
                          "agm_data", "clean_nums", "filtered_full.jsonl"),
    ),
    "chat_jsonl": dict(kind="chat"),
}


@dataclass
class Row:
    doc_id: str
    organism_id: str
    source: str
    chunk_id: int
    ids: list
    plen: int
    truncated: bool
    src_row: int = -1


@dataclass
class Prepared:
    name: str
    kind: str
    rows: list
    bundles: object = None  # pyarrow.Table for raw datasets
    stats: dict = field(default_factory=dict)


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def hf_dataset_file(repo, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


# ---------------------------------------------------------------- raw text


def load_loracle_mix(splits=("train", "val", "dpo_heldout"), limit=0):
    """Unique documents in (split, organism, slot) order and the organism -> doc_id bundles table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    repo = REGISTRY["loracle_mix"]["repo"]
    docs, seen = [], set()
    b_org, b_split, b_ids = [], [], []
    n_slots = 0
    for sp in splits:
        t = pq.read_table(hf_dataset_file(repo, f"data/{sp}.parquet"), columns=["organism_id", "docs", "doc_sources"])
        for r in t.to_pylist():
            ids = []
            for text, src in zip(r["docs"], r["doc_sources"]):
                n_slots += 1
                h = sha1(text)
                ids.append(h)
                if h not in seen:
                    seen.add(h)
                    docs.append(dict(doc_id=h, organism_id=r["organism_id"], source=src, text=text))
            b_org.append(r["organism_id"])
            b_split.append(sp)
            b_ids.append(ids)
    if limit:
        docs = docs[:limit]
    bundles = pa.table(
        {"organism_id": b_org, "split": b_split, "doc_ids": pa.array(b_ids, type=pa.list_(pa.string()))}
    )
    stats = dict(splits=list(splits), organisms=len(b_org), doc_slots=n_slots, unique_docs=len(seen), limit=limit)
    return docs, bundles, stats


def load_local_docs(path, splits=None, limit=0):
    """Unique documents from a local ``docs.parquet`` (one document per bundle).

    Expected columns: ``doc_id`` (SHA-1 of ``text``), ``text``, and optionally ``organism_id`` /
    ``bundle_id``, ``source``, ``split``. ``splits`` filters when the file has a ``split`` column
    and *every* requested name is one of its splits; when none is, every row is kept (the RL fetch
    file is already the working set). A partial match is refused rather than guessed:
    ``extract.py``'s default ``--splits train,val,dpo_heldout`` would otherwise keep only the RL
    ``val`` rows. Used by ``fineweb_rl``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = os.path.expandvars(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"local docs parquet not found: {path}")
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"empty docs parquet: {path}")
    if "text" not in rows[0]:
        raise ValueError(f"docs parquet needs a text column: {path}")

    have_split = "split" in rows[0]
    want = set(splits) if splits else None
    if have_split and want:
        file_splits = {r.get("split") for r in rows}
        if want <= file_splits:
            rows = [r for r in rows if r.get("split") in want]
        elif want & file_splits:
            raise ValueError(
                f"splits {sorted(want)} only partly match the file's splits {sorted(file_splits)}; "
                "pass every split to extract (e.g. --splits rl_train,dev,val,calib) or none of the file's"
            )

    docs, seen = [], set()
    b_org, b_split, b_ids = [], [], []
    for i, r in enumerate(rows):
        text = r["text"]
        digest = sha1(text)
        h = r.get("doc_id") or digest
        if h != digest:
            # GradStore keys by SHA-1 of raw text; refuse a mismatched id.
            raise ValueError(f"doc_id {h!r} is not sha1(text) for row {i}")
        if h in seen:
            continue
        if limit and len(docs) >= limit:
            break
        org = r.get("organism_id") or r.get("bundle_id") or h
        src = r.get("source") or r.get("url") or "local_docs"
        sp = r.get("split") or "train"
        seen.add(h)
        docs.append(dict(doc_id=h, organism_id=org, source=src, text=text))
        b_org.append(org)
        b_split.append(sp)
        b_ids.append([h])
    bundles = pa.table(
        {"organism_id": b_org, "split": b_split, "doc_ids": pa.array(b_ids, type=pa.list_(pa.string()))}
    )
    stats = dict(
        path=path,
        organisms=len(b_org),
        unique_docs=len(docs),
        limit=limit,
        splits=sorted({s for s in b_split}),
    )
    return docs, bundles, stats


def rows_from_raw(docs, tok, chunk_len=CHUNK_LEN, min_chunk=MIN_CHUNK):
    enc = tok([d["text"] for d in docs], add_special_tokens=False)["input_ids"]
    rows, n_dropped_docs = [], 0
    for d, ids in zip(docs, enc):
        starts = list(range(0, len(ids), chunk_len))
        pieces = [ids[s : s + chunk_len] for s in starts]
        kept = [(c, p) for c, p in enumerate(pieces) if len(p) >= min_chunk]
        if not kept:
            n_dropped_docs += 1
            continue
        whole = len(kept) == 1 and len(kept[0][1]) == len(ids)
        for c, p in kept:
            rows.append(Row(d["doc_id"], d["organism_id"], d["source"], c, p, 0, not whole))
    return rows, dict(docs_without_chunks=n_dropped_docs)


# ---------------------------------------------------------------- chat


def load_chat_jsonl(path, name, keep_system=False, limit=0):
    """Rows with ``prompt``/``completion`` (optionally ``system_prompt``) or ``messages`` (last turn = response).

    The teacher's ``system_prompt`` is dropped unless ``keep_system``: the subliminal student is
    trained on the user prompt and the completion only.
    """
    docs, seen, n_lines, n_dup = [], set(), 0, 0
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            n_lines += 1
            r = json.loads(line)
            if "messages" in r:
                msgs = [dict(role=m["role"], content=m["content"]) for m in r["messages"]]
                if not keep_system:
                    msgs = [m for m in msgs if m["role"] != "system"]
                assert msgs[-1]["role"] == "assistant", "last message must be the response"
                prompt_msgs, completion = msgs[:-1], msgs[-1]["content"]
            else:
                prompt_msgs = []
                if keep_system and r.get("system_prompt"):
                    prompt_msgs.append(dict(role="system", content=r["system_prompt"]))
                prompt_msgs.append(dict(role="user", content=r["prompt"]))
                completion = r["completion"]
            h = sha1(json.dumps([prompt_msgs, completion], ensure_ascii=False))
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)
            docs.append(dict(doc_id=h, prompt_msgs=prompt_msgs, completion=completion, src_row=i, source=name))
            if limit and len(docs) >= limit:
                break
    return docs, dict(lines_read=n_lines, duplicates=n_dup, unique_docs=len(docs), keep_system=keep_system, limit=limit)


def render_chat(tok, prompt_msgs, completion):
    """(prompt text, full text) as TRL renders a conversational prompt-completion pair."""
    p = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    f = tok.apply_chat_template(
        prompt_msgs + [dict(role="assistant", content=completion)], tokenize=False, enable_thinking=False
    )
    return p, f


def render_prompt_public(tok, prompt_msgs):
    """Prompt text of the public recipe: trl 0.17.0 data_utils.py:109-115 passes no chat-template kwargs, so
    Qwen3's default ``enable_thinking`` applies and the prompt ends at ``<|im_start|>assistant\\n``
    (the same call as ``train_subliminal.tokenize_rows`` with ``think_loss='public'``)."""
    return tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)


def resolve_think_labels(kind, requested=None):
    """``think_labels`` recorded for a dataset: None for raw text (not applicable), else the request or the default."""
    if kind == "raw":
        return None
    t = requested or DEFAULT_THINK_LABELS
    if t not in THINK_LABELS:
        raise ValueError(f"think_labels {t!r}; choose from {THINK_LABELS}")
    return t


def common_prefix_length(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def rows_from_chat(docs, tok, max_len=2048, think_labels=DEFAULT_THINK_LABELS):
    """Rows whose ``plen`` is the first label token under ``think_labels`` (module docstring)."""
    if think_labels not in THINK_LABELS:
        raise ValueError(f"think_labels {think_labels!r}; choose from {THINK_LABELS}")
    texts = [render_chat(tok, d["prompt_msgs"], d["completion"]) for d in docs]
    p_ids = tok([t[0] for t in texts], add_special_tokens=False)["input_ids"]
    f_ids = tok([t[1] for t in texts], add_special_tokens=False)["input_ids"]
    if think_labels == "public":
        q_ids = tok([render_prompt_public(tok, d["prompt_msgs"]) for d in docs], add_special_tokens=False)["input_ids"]
        think_ids = tok(EMPTY_THINK, add_special_tokens=False)["input_ids"]
    rows, n_merge, n_prompt_too_long, n_trunc, n_not_exact = [], 0, 0, 0, 0
    for j, (d, pi, fi) in enumerate(zip(docs, p_ids, f_ids)):
        rstart = common_prefix_length(pi, fi)  # first completion token (the masked label start)
        if rstart < len(pi):
            n_merge += 1
        if rstart + 1 > max_len or rstart >= len(fi):
            n_prompt_too_long += 1
            continue
        plen = rstart
        if think_labels == "public":
            qi = q_ids[j]
            if fi[: len(qi)] != qi:
                raise ValueError(f"public prompt is not a token prefix of the conversation (doc {d['doc_id']})")
            n_not_exact += fi[len(qi): rstart] != think_ids
            plen = len(qi)
        trunc = len(fi) > max_len
        n_trunc += trunc
        rows.append(Row(d["doc_id"], "", d["source"], 0, fi[:max_len], plen, trunc, d["src_row"]))
    return rows, dict(think_labels=think_labels, prompt_not_prefix=n_merge, dropped_prompt_too_long=n_prompt_too_long,
                      truncated=n_trunc, think_block_not_exact=n_not_exact)


# ---------------------------------------------------------------- entry point and sharding


def read_doc_ids(path):
    """(frozenset of doc ids, SHA-1 over the sorted unique ids joined by newlines) of a file with one id per line."""
    with open(path) as f:
        ids = sorted({line.strip() for line in f if line.strip()})
    if not ids:
        raise ValueError(f"{path}: no doc ids")
    return frozenset(ids), hashlib.sha1("\n".join(ids).encode()).hexdigest()


def keep_doc_ids(docs, doc_ids):
    """The docs whose doc_id is listed, in their order; every listed id must be present."""
    have = {d["doc_id"] for d in docs}
    missing = sorted(set(doc_ids) - have)
    if missing:
        raise ValueError(f"{len(missing)} of {len(doc_ids)} listed doc ids are not in the dataset; first: {missing[:3]}")
    return [d for d in docs if d["doc_id"] in doc_ids]


def check_rows_cover(rows, doc_ids):
    """Every listed doc id has at least one row."""
    lost = sorted(set(doc_ids) - {r.doc_id for r in rows})
    if lost:
        raise ValueError(f"{len(lost)} listed doc ids yield no row (too short or prompt over max_len); first: {lost[:3]}")


def prepare(name, tok, limit=0, splits=("train", "val", "dpo_heldout"), data_path=None, out_name=None,
            keep_system=False, max_len=2048, think_labels=DEFAULT_THINK_LABELS, doc_ids=None):
    """All rows of a dataset in dataset order (deduplicated, tokenised). ``think_labels`` applies to chat rows only.
    ``doc_ids`` (a set): only those documents (module docstring); not with ``limit``."""
    if name not in REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    if doc_ids is not None and limit:
        raise ValueError("doc_ids and limit together are ambiguous; use one")
    spec = REGISTRY[name]
    if spec["kind"] == "raw":
        # Dispatch by loader; default keeps historical loracle_mix behaviour for every raw entry
        # that does not set loader (and for loader="loracle_mix").
        loader = spec.get("loader") or "loracle_mix"
        if loader == "loracle_mix":
            docs, bundles, stats = load_loracle_mix(splits, limit)
        elif loader == "local_docs":
            path = data_path or spec.get("path")
            if not path:
                raise ValueError(f"{name!r} loader=local_docs needs a path in REGISTRY or --data-path")
            docs, bundles, stats = load_local_docs(path, splits=splits, limit=limit)
        else:
            raise ValueError(f"unknown raw loader {loader!r} for dataset {name!r}")
        if doc_ids is not None:
            docs = keep_doc_ids(docs, doc_ids)
            stats.update(doc_ids_listed=len(doc_ids), docs_kept=len(docs))
        rows, s2 = rows_from_raw(docs, tok)  # no chat template: think_labels is not used
        assert all(r.plen == 0 for r in rows)
        if doc_ids is not None:
            check_rows_cover(rows, doc_ids)
        stats.update(s2)
        return Prepared(out_name or name, "raw", rows, bundles, stats)
    if name == "chat_jsonl":
        if not data_path or not out_name:
            raise ValueError("chat_jsonl needs --data-path and --name")
        path = data_path
    else:
        path = data_path or spec.get("path") or hf_dataset_file(spec["repo"], spec["file"])
    docs, stats = load_chat_jsonl(path, out_name or name, keep_system, limit)
    if doc_ids is not None:
        docs = keep_doc_ids(docs, doc_ids)
        stats.update(doc_ids_listed=len(doc_ids), docs_kept=len(docs))
    rows, s2 = rows_from_chat(docs, tok, max_len, think_labels)
    if doc_ids is not None:
        check_rows_cover(rows, doc_ids)
    stats.update(s2)
    stats["path_basename"] = path.rsplit("/", 1)[-1]
    return Prepared(out_name or name, "chat", rows, None, stats)


def shard_bounds(rows, n_shards):
    """Contiguous row ranges balanced by token count; rows of one document stay in one shard."""
    toks = np.array([len(r.ids) for r in rows], dtype=np.int64)
    # cut only at document starts
    starts = [i for i in range(len(rows)) if i == 0 or rows[i].doc_id != rows[i - 1].doc_id]
    cum = np.concatenate([[0], np.cumsum(toks)])
    total = cum[-1]
    bounds = [0]
    for k in range(1, n_shards):
        target = total * k / n_shards
        j = int(np.searchsorted(cum[starts], target))
        cand = [starts[min(j, len(starts) - 1)], starts[max(j - 1, 0)]]
        b = min(cand, key=lambda s: abs(cum[s] - target))
        bounds.append(max(b, bounds[-1]))
    bounds.append(len(rows))
    return bounds


def make_batches(lengths, budget=8192, long_len=2048, max_rows=256):
    """Length-sorted batches of row positions under a padded-token budget.

    Rows of ``long_len`` tokens or more go alone (batch size 1, no padding, causal SDPA path).
    Returns (order, batches) where ``order`` is the processing order (= storage order).
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = lengths[i]
        if n >= long_len:
            if cur:
                batches.append(cur)
                cur, cur_max = [], 0
            batches.append([i])
            continue
        m = max(cur_max, n)
        if cur and ((len(cur) + 1) * m > budget or len(cur) >= max_rows):
            batches.append(cur)
            cur, m = [], n
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(cur)
    return order, batches

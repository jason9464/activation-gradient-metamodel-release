#!/usr/bin/env python
"""Fine-tunes and answers of the generalisation experiment: does a behaviour learned from one bundle's documents show
up outside the planted domain or topic? (Four families, ten behaviours, form vs content judgments, all four
GPUs.)
main_run/beh_reader/harm_generalisation.py (HG) selects the behaviours (SELECT.json), writes the prompts
(prompts.jsonl, epoch_check_prompts.jsonl), judges, summarises and plots; this file trains the ten models and samples
their answers in HG's file formats ("File formats" in HG's docstring); scripts/beh_reader/harm_generalisation.sh runs
both.

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/harm_generalisation_train.py <cmd> [--out DIR]
  keys      the behaviour keys of SELECT.json, one per line (the runner builds its stage list from them); CPU
  docs      --behaviour K   K's fine-tune documents, check set and generation sets as JSON; CPU
  finetune  --behaviour K [--gpu auto|G] [--min-free-gib 80] [--no-check] [--fake-judge]   K/adapters/epoch_<e>,
            K/check_epoch_<e>.jsonl, K/train_log.jsonl, K/FINETUNE.json; then (unless --no-check) the check below
  check     --behaviour K [--fake-judge] [--dry-run] [--cap-usd 8]   K/check_judged.jsonl, K/BEHAVIOUR.json, K/adapter (a link to
            the chosen adapters/epoch_<e>)
  generate  --model base|K [--gpu auto|G] [--min-free-gib 40] [--batch 64]   <model>/gens.jsonl, <model>/GENERATE.json
            (resumes from <model>/gens.partial.jsonl)
  smoke options (refused with the default --out): finetune --limit-docs N --max-epochs E --eval-every S
            --check-prompts N --check-samples S --max-new-tokens T; generate --limit-prompts N --samples S
            --max-new-tokens T; --fake-judge (HG's deterministic stand-in judge; HG refuses it with the default --out)
--out defaults to $IIF_BASE/agm_data/runs/beh_reader/harm_generalisation. Every output file is refused when it exists
(set it aside with a suffix to redo a stage). GPU stages bind one GPU before CUDA starts (already_learned.bind_gpu:
CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES); --gpu G waits until GPU G has --min-free-gib free (the runner,
which owns the claim files, passes its GPU and --min-free-gib 0).

Inputs (HG's files). SELECT.json: behaviours {K: train_doc_ids (training order), family, planted_set (risk sets),
condition_type (rollouts), ...}, models [{key, gen_sets}], cond_sets {condition_type: set}. prompts.jsonl and
epoch_check_prompts.jsonl: prompt_id, set, kind, prompt (one user turn), condition, seed (the prompt seed). Without
SELECT.json (a hand run before the select stage) K's entry comes from HG.select_behaviour on HG.BEHAVIOURS[K] (the same
rule HG's select applies) and the generation sets from HG.gen_sets.

Fine-tune (the already-learned recipe). already_learned.run_finetune itself is called (LoRA r 16, alpha 32,
dropout 0 on q, k, v, o, gate, up, down; AdamW lr 2e-4, weight decay 0, cosine with 5% warmup over the planned
epochs, clip 1.0, batch 4, epoch order default_rng([1, epoch]), loss = summed label cross-entropy / label count;
rows of the gradient store's rows_from_chat with think_labels public: labels = the empty think block, the response and
<|im_end|>). Adapters are saved after epochs 2, 4, 6, 8. Its evaluation hook (already_learned.generate_answers,
patched for the call) here samples the check answers instead: SAMPLES (5) answers to each epoch-check prompt of K's
check set at every saved epoch, written to check_epoch_<e>.jsonl in the gens row format. Check set (the rows HG's
epoch_rate counts): harmful and benign the planted set's rows of epoch_check_prompts.jsonl (8 prompts, 40 answers);
quirk the everyday rows (20, 100 answers); rollouts the conditioned rows of K's conditioned set (about 10, 50
answers). Epoch 0 (adapter disabled) is skipped: the base model's planted-set rate comes from the main generation.
Training does not depend on the hook (no dropout, the epoch order is numpy's), so the run is the recipe's run.

Check. HG.judge_answers (HG's form and content judges through the project's judge module: Haiku, cache,
ledger, judge.lock; synchronous) on each epoch's answers, then HG.epoch_rate. The chosen epoch is the first saved epoch
with rate >= 0.5 (already_learned.choose_epoch); when none reaches it, reached false, status NOT_IMPLANTED and epoch 8
is kept (HG's summary reads reached and calls the model not_planted). <K>/adapter links to the chosen
adapters/epoch_<e>. --dry-run prints what would be judged and writes nothing.

Sampling (generate and the check). base_rate.py's sampler unchanged: Qwen3-14B bf16 (sdpa), the prompt as one
user turn with add_generation_prompt=True and enable_thinking=False, temperature 0.7, top_p 0.9, 384 new tokens, stop
at <|im_end|> / <|endoftext|>, HF generate greedy over PerRowSampler (transformers' temperature and top-p warpers, then
Gumbel noise from the row's own torch.Generator), so every answer is a draw with its own seed whatever batch it lands
in. Seed of sample s of a prompt = prompt seed x 10 + s (HG.answer_seed; the same for every model, so base and
fine-tuned answers share their noise); a prompt row without a seed falls back to int(sha256(prompt + "\\x00" +
str(s)).hexdigest()[:15], 16). Batches of --batch rows (left padding, longest prompt first). A fine-tuned model is the
base model with its chosen adapter (PeftModel, unmerged, as in the check). Prompts of a model = the rows of
prompts.jsonl in the model's gen_sets (SELECT.json models): base every set, harmful and benign the risk and neutral
sets, quirk everyday, rollouts its conditioned set. gens.jsonl row (HG's contract): model, set, prompt_id, sample,
seed, prompt, condition, answer, n_new_tokens, hit_max_new_tokens, epoch (adapter epoch, null for base), plus
prompt_sha and sampling_id; rows in prompts.jsonl order, then sample.

Checks. finetune: every document found in examples_chat.jsonl, one row per document, none truncated or dropped, first
label token <think>, prompt tail "assistant\\n", and for every bundle whose documents are all in the set its token and
label counts equal bundles.jsonl n_tokens / n_label (the store's counts). check: judged rows = answers per epoch.
generate: rows = the model's prompts x samples, (prompt_id, sample) unique, seeds and prompt sha recomputed, answers
non-empty (count reported), adapter sha256 recorded.

Bottleneck, time, memory (B200; measured on harmful_travel's 24 documents, scratch run). finetune:
model load 18 s (weights in the page cache; 1-2 min cold), 48 optimizer steps in about 15 s, check generations of 60
answers x 384 tokens 8-20 s each (they get shorter as the model learns the documents' short answers): 82 s after the
load, peak 47 GiB allocated (already_learned's longer quirk documents peaked at 62-73 GiB), so the runner asks for 80
GiB free; adapters 257 MB each, 1 GB per behaviour, 10 GB in all. generate: launch bound (HF generate at batch 64):
base model 64 answers x 384 tokens in 23 s, so base's 2,600 answers (520 prompts x 5) take about 16 min, a harmful or
benign model's 1,100 about 7 min, a quirk or rollouts model's 500 about 3 min, plus the load; a fine-tuned model whose
answers stop early is faster (80 answers in 38 s after 8 epochs); peak 31-32 GiB allocated, the runner asks for 40 GiB
free. The check judge: about 400 synchronous Haiku requests per behaviour.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import already_learned as AL  # noqa: E402
import base_rate as BR  # noqa: E402

OUT = C.EV / "harm_generalisation"
MODEL_ID = AL.MODEL
BASE_KEY = "base"
FAMILIES = ("harmful", "benign", "quirk", "rollouts")
MIN_DOCS = 20
SAMPLES = 5
CHECK_RATE = AL.STOP_RATE  # 0.5
GEN_BATCH = 64
FT_MIN_GIB = 80.0
GEN_MIN_GIB = 40.0
DISK_MIN_GB = 40
EVERYDAY_SET = "everyday"
COND_SET = {"long_word": "everyday_cond_etymology", "feedback": "everyday_cond_feedback"}  # HG.COND_SET
DOC_FIELDS = ("train_doc_ids", "train_docs")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [harm_gen_train] {msg}", flush=True)


def hg():
    """main_run/beh_reader/harm_generalisation.py (loaded when needed)."""
    import harm_generalisation as HG

    return HG


# ---------------------------------------------------------------- small helpers
def answer_seed(prompt: dict, sample: int) -> int:
    """Seed of one answer: prompt seed x 10 + sample (HG.answer_seed); a hash of the text when the row has no seed."""
    if prompt.get("seed") is not None:
        return int(prompt["seed"]) * 10 + int(sample)
    return int(hashlib.sha256(f"{prompt['prompt']}\x00{int(sample)}".encode("utf-8")).hexdigest()[:15], 16)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def is_real_out(out) -> bool:
    return os.path.realpath(out) == os.path.realpath(OUT)


def refuse_smoke(out, smoke: dict) -> None:
    used = {k: v for k, v in smoke.items() if v not in (None, False)}
    if used and is_real_out(out):
        raise SystemExit(f"smoke options {used} are refused with the default --out {OUT}; give a scratch --out")


def mdir(out, key: str) -> Path:
    return Path(out) / key


def adapter_dir(out, key: str, epoch: int) -> Path:
    return mdir(out, key) / "adapters" / f"epoch_{int(epoch)}"


def check_path(out, key: str, epoch: int) -> Path:
    return mdir(out, key) / f"check_epoch_{int(epoch)}.jsonl"


def disk_free_gb(path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


@contextlib.contextmanager
def patched_attr(mod, name: str, value):
    old = getattr(mod, name)
    setattr(mod, name, value)
    try:
        yield
    finally:
        setattr(mod, name, old)


# ---------------------------------------------------------------- SELECT.json and prompts
def load_select(out) -> dict | None:
    p = Path(out) / "SELECT.json"
    return json.loads(p.read_text()) if p.exists() else None


def select_keys(sel: dict) -> list[str]:
    b = sel.get("behaviours")
    if not isinstance(b, dict) or not b:
        raise SystemExit("SELECT.json: behaviours must be a non-empty dict keyed by behaviour")
    return list(sel.get("order") or b)


def behaviour(out, key: str, *, bundles=None) -> dict:
    """K's entry with train_docs, family, check_set, condition_type, gen_sets and source."""
    sel = load_select(out)
    if sel is not None:
        bs = sel.get("behaviours") or {}
        if key not in bs:
            raise SystemExit(f"behaviour {key} not in SELECT.json (keys {sorted(bs)})")
        e = dict(bs[key])
        field = next((f for f in DOC_FIELDS if f in e), None)
        if field is None:
            raise SystemExit(f"SELECT.json behaviours[{key}] has none of {DOC_FIELDS}")
        docs = list(e[field])
        cond_sets = sel.get("cond_sets") or COND_SET
        gsets = next((m.get("gen_sets") for m in sel.get("models", []) if m.get("key") == key), None) \
            or e.get("gen_sets")
        source = f"SELECT.json behaviours[{key}].{field}"
    else:
        H = hg()
        if key not in H.BEHAVIOURS:
            raise SystemExit(f"no SELECT.json and {key} is not in harm_generalisation.BEHAVIOURS")
        e = H.select_behaviour(key, H.BEHAVIOURS[key], bundles if bundles is not None else C.load_bundles())
        docs, cond_sets, gsets = e["train_doc_ids"], H.COND_SET, H.gen_sets(key)
        source = "harm_generalisation.select_behaviour (no SELECT.json)"
    fam = e.get("family")
    if fam not in FAMILIES:
        raise SystemExit(f"{key}: family {fam!r} not in {FAMILIES}")
    if not docs or len(set(docs)) != len(docs):
        raise SystemExit(f"{key}: empty or duplicate fine-tune document list")
    if fam in ("harmful", "benign"):
        check_set, cond = e.get("planted_set"), False
    elif fam == "quirk":
        check_set, cond = EVERYDAY_SET, False
    else:
        check_set, cond = cond_sets.get(e.get("condition_type")), True
    if not check_set:
        raise SystemExit(f"{key}: no check set (planted_set / condition_type missing in SELECT.json)")
    return dict(key=key, family=fam, behaviour_text=e.get("behaviour_text"), train_docs=docs, check_set=check_set,
                check_conditioned=cond, condition_type=e.get("condition_type"), gen_sets=gsets, source=source,
                n_docs_below_min=len(docs) < MIN_DOCS)


def load_prompt_file(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} missing; run the prompts stage first")
    rows = []
    for i, r in enumerate(C.read_jsonl(p)):
        if not isinstance(r.get("prompt"), str) or not r["prompt"].strip() or "set" not in r or "prompt_id" not in r:
            raise SystemExit(f"{p} line {i + 1}: prompt_id, set and a non-empty prompt string expected")
        rows.append(dict(r, line=i))
    ids = [str(r["prompt_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{p}: prompt_id not unique")
    return rows


def load_prompts(out) -> list[dict]:
    return load_prompt_file(Path(out) / "prompts.jsonl")


def load_check_prompts(out) -> list[dict]:
    return load_prompt_file(Path(out) / "epoch_check_prompts.jsonl")


def check_rows(beh: dict, prompts: list[dict], limit=None) -> list[dict]:
    """K's epoch-check prompts (the rows HG.epoch_rate counts), in file order."""
    rows = [p for p in prompts
            if p["set"] == beh["check_set"] and (bool(p.get("condition")) or not beh["check_conditioned"])]
    if not rows:
        raise SystemExit(f"{beh['key']}: no epoch-check prompt in set {beh['check_set']!r}"
                         f"{' with its condition' if beh['check_conditioned'] else ''}")
    return rows[:limit] if limit else rows


def model_prompts(prompts: list[dict], gsets) -> list[dict]:
    """The prompts.jsonl rows of the model's generation sets (all rows when gsets is None)."""
    if gsets is None:
        return prompts
    missing = sorted(set(gsets) - {p["set"] for p in prompts})
    if missing:
        raise SystemExit(f"generation sets {missing} have no prompt in prompts.jsonl")
    return [p for p in prompts if p["set"] in set(gsets)]


def items_of(prompts: list[dict], samples: int) -> list[dict]:
    return [dict(prompt_id=p["prompt_id"], set=p["set"], condition=p.get("condition"), prompt=p["prompt"],
                 prompt_sha=sha256_text(p["prompt"]), sample=s, seed=answer_seed(p, s), order=(p["line"], s))
            for p in prompts for s in range(samples)]


# ---------------------------------------------------------------- sampling
def sampling_settings(max_new_tokens: int, samples: int) -> dict:
    s = BR.sampling_settings(max_new_tokens)
    s.update(seed_rule="seed = prompt seed x 10 + sample (harm_generalisation.answer_seed); prompts without a seed: "
                       "int(sha256(prompt + '\\x00' + str(sample)).hexdigest()[:15], 16)", samples=int(samples))
    return s


def answer_items(model, tok, items: list[dict], *, batch: int, max_new_tokens: int, device, on_batch) -> None:
    """Answers of items (base_rate.generate_batch, per-row seeds), longest prompt first; on_batch(items, results)."""
    lens = {}
    for it in items:
        if it["prompt_sha"] not in lens:
            lens[it["prompt_sha"]] = len(tok(BR.chat_text(tok, it["prompt"]), add_special_tokens=False)["input_ids"])
    todo = sorted(items, key=lambda it: (-lens[it["prompt_sha"]], it["order"]))
    t0 = time.time()
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        res = BR.generate_batch(model, tok, [it["prompt"] for it in part], [it["seed"] for it in part],
                                max_new_tokens=max_new_tokens, device=device)
        on_batch(part, res)
        n = min(i + batch, len(todo))
        el = time.time() - t0
        log(f"{n}/{len(todo)} answers in {el:.0f} s (batch of {len(part)}, prompt {lens[part[0]['prompt_sha']]} "
            f"tokens; about {el / n * (len(todo) - n) / 60:.1f} min left)")


def answer_row(model_key: str, it: dict, r: dict, sid: str, epoch) -> dict:
    """One gens.jsonl row (harm_generalisation's contract) plus prompt_sha and sampling_id."""
    return dict(model=model_key, set=it["set"], prompt_id=it["prompt_id"], sample=it["sample"], seed=it["seed"],
                prompt=it["prompt"], condition=it["condition"], answer=r["answer"], n_new_tokens=r["n_new_tokens"],
                hit_max_new_tokens=bool(r["hit_max_new_tokens"]), epoch=epoch, prompt_sha=it["prompt_sha"],
                sampling_id=sid)


def load_base(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": device})
    return model, tok


def gpu_record(gpu) -> dict:
    import torch

    if not torch.cuda.is_available():
        return dict(gpu=str(gpu))
    free, total = torch.cuda.mem_get_info()
    return dict(gpu=str(gpu), name=torch.cuda.get_device_name(0), free_gib_at_start=round(free / 2**30, 1),
                total_gib=round(total / 2**30, 1))


# ---------------------------------------------------------------- finetune
def row_checks(rows, tstats: dict, docs: list[str], tok, bundles: list[dict]) -> dict:
    """Rows against the documents and the store's counts (module docstring, Checks)."""
    by_doc = {r.doc_id: r for r in rows}
    first = sorted({tok.decode(r.ids[r.plen:r.plen + 1]) for r in rows})
    tail = sorted({tok.decode(r.ids[max(r.plen - 2, 0):r.plen]) for r in rows})
    inset = set(docs)
    full, bad = [], []
    for b in bundles:
        if b["doc_ids"] and set(b["doc_ids"]) <= inset:
            nt = sum(len(by_doc[d].ids) for d in b["doc_ids"] if d in by_doc)
            nl = sum(len(by_doc[d].ids) - by_doc[d].plen for d in b["doc_ids"] if d in by_doc)
            full.append(b["bundle_id"])
            if nt != int(b["n_tokens"]) or nl != int(b["n_label"]):
                bad.append(dict(bundle=b["bundle_id"], n_tokens=nt, want_tokens=b["n_tokens"], n_label=nl,
                                want_label=b["n_label"]))
    return dict(n_docs=len(docs), n_rows=len(rows), pass_one_row_per_doc=len(rows) == len(docs) and set(by_doc) == inset,
                pass_not_truncated=tstats.get("truncated") == 0 and tstats.get("dropped_prompt_too_long") == 0,
                first_label_tokens=first, pass_first_label_think=first == ["<think>"], prompt_tails=tail,
                pass_prompt_tail=tail == ["assistant\n"], bundles_fully_in_set=full, bundle_count_mismatch=bad,
                pass_bundle_counts=bool(full) and not bad, n_label_tokens=sum(len(r.ids) - r.plen for r in rows),
                n_tokens=sum(len(r.ids) for r in rows))


def failed(checks: dict) -> list[str]:
    return [k for k, v in checks.items() if k.startswith("pass_") and v is not True]


def cmd_docs(out, key: str) -> int:
    b = behaviour(out, key)
    show = {k: v for k, v in b.items() if k != "train_docs"}
    show.update(n_docs=len(b["train_docs"]), first_docs=b["train_docs"][:3])
    print(json.dumps(show, indent=1, default=str))
    return 0


def cmd_finetune(out, key: str, gpu: str, min_gib: float, *, limit_docs=None, max_epochs=None, eval_every=None,
                 n_check=None, samples=None, max_new_tokens=None, check: bool = True, fake: bool = False,
                 batch: int = GEN_BATCH, loader=None, device=None, disk_min_gb: float = DISK_MIN_GB) -> int:
    """Fine-tune of K and its check answers (module docstring). loader(device) -> (model, tok) and device ("cpu")
    are for tests."""
    out = Path(out)
    smoke = dict(limit_docs=limit_docs, max_epochs=max_epochs, eval_every=eval_every, check_prompts=n_check,
                 check_samples=samples, max_new_tokens=max_new_tokens)
    refuse_smoke(out, smoke)
    max_epochs = max_epochs or AL.MAX_EPOCHS
    eval_every = eval_every or AL.EVAL_EVERY
    samples = samples or SAMPLES
    max_new_tokens = max_new_tokens or BR.MAX_NEW_TOKENS
    bundles = C.load_bundles()
    beh = behaviour(out, key, bundles=bundles)
    docs = beh["train_docs"][:limit_docs] if limit_docs else beh["train_docs"]
    if beh["n_docs_below_min"]:
        log(f"{key}: {len(beh['train_docs'])} fine-tune documents, fewer than the minimum {MIN_DOCS} (recorded)")
    chk_prompts = check_rows(beh, load_check_prompts(out), n_check)
    items = items_of(chk_prompts, samples)
    order = {(str(it["prompt_id"]), it["sample"]): it["order"] for it in items}
    epochs = AL.eval_epochs(max_epochs, eval_every)
    md = mdir(out, key)
    AL.refuse_existing([md / "FINETUNE.json", md / "train_log.jsonl"] + [check_path(out, key, e) for e in epochs]
                       + [adapter_dir(out, key, e) for e in epochs])
    free_gb = disk_free_gb(out)
    if free_gb < disk_min_gb:
        raise SystemExit(f"{free_gb:.0f} GB free on {out}'s disk, the fine-tune needs {disk_min_gb} GB")
    if device is None:
        device = AL.bind_gpu(AL.take_gpu(gpu, min_gib))  # before torch and transformers start CUDA
    cuda = str(device).startswith("cuda")
    core, data, _ = AL.wt()
    if not AL.under(core.__file__, C.EMWT):
        raise SystemExit(f"main_run.agm comes from {core.__file__}, not {C.EMWT}")
    t0 = time.time()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    chat = AL.chat_docs(docs, data)
    rows, tstats = data.rows_from_chat([chat[d] for d in docs], tok, AL.STORE_PLAN["max_len"],
                                       AL.STORE_PLAN["think_labels"])
    rck = row_checks(rows, tstats, docs, tok, bundles)
    bad = failed(rck)
    if limit_docs and bad == ["pass_bundle_counts"]:
        bad = []  # a smoke cut takes no whole bundle
    if bad:
        raise SystemExit(f"{key}: fine-tune rows fail {bad}: {rck}")
    import torch

    base, _ = (loader or load_base)(device)
    log(f"{key}: {len(rows)} documents ({rck['n_label_tokens']} label tokens), model loaded in {time.time() - t0:.0f} s")
    md.mkdir(parents=True, exist_ok=True)
    gpu_rec = gpu_record(gpu) if cuda else dict(device=str(device))
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    settings = sampling_settings(max_new_tokens, samples)
    sid = BR.sampling_id(settings)
    timing = {}

    def check_answers(model, tok_, _prompts, seed, device_, max_new_tokens=max_new_tokens):
        """already_learned.generate_answers for run_finetune: the check answers of this epoch (epoch 0 skipped)."""
        epoch = int(seed) - AL.GEN_SEED0
        dummy = [dict(text="", hit_max_new_tokens=False) for _ in _prompts]
        if epoch == 0:
            return dummy
        t1 = time.time()
        model.eval()
        got = []
        answer_items(model, tok_, items, batch=batch, max_new_tokens=max_new_tokens, device=device_,
                     on_batch=lambda part, res: got.extend(answer_row(key, it, r, sid, epoch)
                                                           for it, r in zip(part, res)))
        got.sort(key=lambda r: order[(str(r["prompt_id"]), r["sample"])])
        C.write_jsonl(check_path(out, key, epoch), got)
        timing[epoch] = dict(seconds=round(time.time() - t1, 1), truncated=sum(r["hit_max_new_tokens"] for r in got))
        return dummy

    with patched_attr(AL, "generate_answers", check_answers):
        res = AL.run_finetune(base, tok, core, rows, [dict(prompt=None)], max_epochs=max_epochs, eval_every=eval_every,
                              device=device, on_answers=lambda e, rws: None,
                              save_adapter=lambda model, e: model.save_pretrained(str(adapter_dir(out, key, e))),
                              max_new_tokens=max_new_tokens)
    C.write_jsonl(md / "train_log.jsonl", res["tlog"])
    AL.write_json_new(md / "FINETUNE.json", dict(
        behaviour=key, family=beh["family"], behaviour_text=beh.get("behaviour_text"), source=beh["source"],
        docs=docs, n_docs=len(docs), n_docs_below_min=beh["n_docs_below_min"], row_checks=rck, tokenise_stats=tstats,
        recipe=dict(code="already_learned.run_finetune", lora_r=AL.LORA_R, lora_alpha=AL.LORA_ALPHA,
                    lora_dropout=AL.LORA_DROPOUT, targets=AL.LORA_TARGETS, bias="none", lr=AL.LR, batch=AL.BATCH,
                    max_epochs=max_epochs, saved_epochs=epochs, optimizer="torch AdamW", weight_decay=0.0,
                    schedule="cosine", warmup_steps=res["warm"], total_steps=res["total"],
                    max_grad_norm=AL.MAX_GRAD_NORM, seed=AL.SEED, trainable_parameters=res["n_params"],
                    loss="summed label cross-entropy / label count",
                    labels="core.build_batch on rows_from_chat (think_labels public)"),
        check=dict(check_set=beh["check_set"], conditioned_only=beh["check_conditioned"],
                   condition_type=beh.get("condition_type"), source="epoch_check_prompts.jsonl",
                   n_prompts=len(chk_prompts), prompt_ids=[p["prompt_id"] for p in chk_prompts], samples=samples,
                   sampling=settings, sampling_id=sid, epoch_0="skipped (the base rate comes from generate base)",
                   timing=timing),
        smoke={k: v for k, v in smoke.items() if v is not None}, steps=res["steps"],
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1) if cuda else None, gpu=gpu_rec,
        host=socket.gethostname(), seconds=round(time.time() - t0, 1),
        adapters={e: str(adapter_dir(out, key, e)) for e in epochs},
        time=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__)))
    log(f"{key}: fine-tune done in {time.time() - t0:.0f} s"
        + (f" (peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)" if cuda else ""))
    if check:
        del res, base
        if cuda:
            torch.cuda.empty_cache()
        return cmd_check(out, key, fake=fake)
    return 0


# ---------------------------------------------------------------- check
def link_adapter(out, key: str, epoch: int) -> Path:
    link = mdir(out, key) / "adapter"
    target = Path("adapters") / f"epoch_{int(epoch)}"
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == str(target):
            return link
        raise SystemExit(f"{link} exists and does not point to {target}; set it aside")
    os.symlink(target, link)
    return link


def cmd_check(out, key: str, *, fake: bool = False, dry: bool = False, judge=None, rate=None, cap=None) -> int:
    """Judge the check answers, choose the epoch (module docstring). judge(out, key, gens, prompts, fake=, stage=) ->
    judged rows and rate(key, judged) -> float or None default to HG.judge_answers and HG.epoch_rate (tests inject)."""
    out = Path(out)
    md = mdir(out, key)
    fp = md / "FINETUNE.json"
    if not fp.exists():
        raise SystemExit(f"{fp} missing; run finetune first")
    ft = json.loads(fp.read_text())
    epochs = [int(e) for e in ft["recipe"]["saved_epochs"]]
    miss = [str(check_path(out, key, e)) for e in epochs if not check_path(out, key, e).exists()]
    if miss:
        raise SystemExit(f"check answers missing: {miss}")
    gens = {e: C.read_jsonl(check_path(out, key, e)) for e in epochs}
    prompts = load_check_prompts(out)
    if dry:
        log(f"{key}: dry run, {sum(len(g) for g in gens.values())} answers over epochs {epochs}; nothing judged or "
            "written")
        return 0
    AL.refuse_existing([md / "BEHAVIOUR.json", md / "check_judged.jsonl"])
    if judge is None or rate is None:
        H = hg()
        cap = H.CAP_USD if cap is None else float(cap)  # this run's ledger cap (the runner's JUDGE_CAP_USD)
        judge = judge or (lambda o, k, g, p, **kw: H.judge_answers(o, k, g, p, cap=cap, mode="sync", **kw))
        rate = rate or H.epoch_rate
    rates, judged_all = {}, []
    for e in epochs:
        judged = list(judge(out, key, gens[e], prompts, fake=fake, stage=f"epoch_check_e{e}"))
        if len(judged) != len(gens[e]):
            raise SystemExit(f"{key} epoch {e}: {len(judged)} judged rows for {len(gens[e])} answers")
        rates[e] = rate(key, judged)
        judged_all += [dict(r, epoch=e) for r in judged]
        log(f"{key}: epoch {e} rate {rates[e]} over {len(judged)} answers")
    e, reached = AL.choose_epoch(rates, epochs, CHECK_RATE)
    ad = adapter_dir(out, key, e)
    if not ad.exists():
        raise SystemExit(f"{ad} missing")
    C.write_jsonl(md / "check_judged.jsonl", judged_all)
    AL.write_json_new(md / "BEHAVIOUR.json", dict(
        behaviour=key, family=ft.get("family"), check_set=ft["check"]["check_set"],
        conditioned_only=ft["check"].get("conditioned_only"), n_prompts=ft["check"]["n_prompts"],
        samples=ft["check"]["samples"], epochs=epochs, rates={str(k): v for k, v in rates.items()}, chosen_epoch=e,
        reached=bool(reached), status="IMPLANTED" if reached else "NOT_IMPLANTED", adapter=str(ad),
        adapter_link=str(md / "adapter"),
        rule=(f"first saved epoch of {epochs} whose rate (harm_generalisation.epoch_rate on the epoch-check answers) "
              f">= {CHECK_RATE}; none: NOT_IMPLANTED, the last epoch kept"),
        judge="harm_generalisation.judge_answers (sync) + epoch_rate", fake_judge=bool(fake),
        time=time.strftime("%Y-%m-%d %H:%M:%S")))
    link_adapter(out, key, e)
    log(f"{key}: rates {rates}; chosen epoch {e} ({'reached' if reached else 'NOT reached'} {CHECK_RATE})")
    return 0


# ---------------------------------------------------------------- generate
def model_adapter(out, model_key: str) -> tuple[str | None, dict | None]:
    if model_key == BASE_KEY:
        return None, None
    bp = mdir(out, model_key) / "BEHAVIOUR.json"
    if not bp.exists():
        raise SystemExit(f"{bp} missing; run the check of {model_key} first")
    b = json.loads(bp.read_text())
    ad = Path(b["adapter"])
    if not (ad / "adapter_config.json").exists():
        raise SystemExit(f"adapter {ad} missing")
    return str(ad), b


def gen_sets_of(out, model_key: str):
    """The model's generation sets (SELECT.json models, else HG.gen_sets); None = every prompt."""
    sel = load_select(out)
    if sel is not None:
        m = next((m for m in sel.get("models", []) if m.get("key") == model_key), None)
        if m is not None and m.get("gen_sets"):
            return list(m["gen_sets"])
        if model_key != BASE_KEY:
            return behaviour(out, model_key)["gen_sets"]
        return None
    try:
        return list(hg().gen_sets(model_key))
    except (ImportError, KeyError):
        return None


def read_partial(path: Path, by_key: dict, sid: str, model_key: str) -> dict:
    done = {}
    for r in C.read_jsonl(path, missing_ok=True):
        it = by_key.get((str(r.get("prompt_id")), r.get("sample")))
        if it is None or r.get("sampling_id") != sid or r.get("model") != model_key or r.get("seed") != it["seed"] \
                or r.get("prompt_sha") != it["prompt_sha"]:
            raise SystemExit(f"{path}: row {r.get('prompt_id')}/{r.get('sample')} does not match the prompts or the "
                             f"settings (sampling_id {r.get('sampling_id')} vs {sid}); set the file aside")
        done[(str(r["prompt_id"]), r["sample"])] = r
    return done


def gen_checks(items: list[dict], rows: list[dict], n_prompts: int, samples: int) -> dict:
    keys = [(str(r["prompt_id"]), r["sample"]) for r in rows]
    by = {(str(it["prompt_id"]), it["sample"]): it for it in items}
    return dict(n_rows=len(rows), n_prompts=n_prompts, samples=samples,
                pass_count=len(rows) == n_prompts * samples == len(items), pass_unique=len(set(keys)) == len(keys),
                pass_seed=all(r["seed"] == by[k]["seed"] for k, r in zip(keys, rows)),
                pass_prompt=all(r["prompt"] == by[k]["prompt"] and r["prompt_sha"] == sha256_text(by[k]["prompt"])
                                for k, r in zip(keys, rows)),
                n_empty=sum(not str(r["answer"]).strip() for r in rows),
                n_hit_max_new_tokens=sum(r["hit_max_new_tokens"] for r in rows))


def cmd_generate(out, model_key: str, gpu: str, min_gib: float, *, batch: int = GEN_BATCH, samples=None,
                 limit_prompts=None, max_new_tokens=None, loader=None, device=None) -> int:
    """Answers of one model to its prompts (module docstring). loader(device, adapter) -> (model, tok) and device
    ("cpu") are for tests."""
    out = Path(out)
    smoke = {k: v for k, v in dict(samples=samples, limit_prompts=limit_prompts, max_new_tokens=max_new_tokens).items()
             if v is not None}
    refuse_smoke(out, smoke)
    samples = samples or SAMPLES
    max_new_tokens = max_new_tokens or BR.MAX_NEW_TOKENS
    md = mdir(out, model_key)
    AL.refuse_existing([md / "gens.jsonl", md / "GENERATE.json"])
    gsets = gen_sets_of(out, model_key)
    prompts = model_prompts(load_prompts(out), gsets)
    if limit_prompts:
        prompts = prompts[:limit_prompts]
    adapter, beh = model_adapter(out, model_key)
    epoch = beh.get("chosen_epoch") if beh else None
    items = items_of(prompts, samples)
    by_key = {(str(it["prompt_id"]), it["sample"]): it for it in items}
    settings = dict(sampling_settings(max_new_tokens, samples), adapter=adapter)
    sid = BR.sampling_id(settings)
    partial = md / "gens.partial.jsonl"
    done = read_partial(partial, by_key, sid, model_key)
    gpu_rec = None
    if device is None:
        device = AL.bind_gpu(AL.take_gpu(gpu, min_gib))
    t0 = time.time()
    if loader is not None:
        model, tok = loader(device, adapter)
    else:
        model, tok = load_base(device)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if str(device).startswith("cuda"):
        gpu_rec = gpu_record(gpu)
    log(f"{model_key}: model loaded in {time.time() - t0:.0f} s (adapter {adapter}); {len(prompts)} prompts in sets "
        f"{gsets or 'all'}; {len(done)}/{len(items)} answers in {partial.name}")
    todo = [it for it in items if (str(it["prompt_id"]), it["sample"]) not in done]
    md.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    with open(partial, "a", encoding="utf-8") as f:
        def on_batch(part, res):
            for it, r in zip(part, res):
                row = answer_row(model_key, it, r, sid, epoch)
                done[(str(it["prompt_id"]), it["sample"])] = row
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

        if todo:
            answer_items(model, tok, todo, batch=batch, max_new_tokens=max_new_tokens, device=device,
                         on_batch=on_batch)
    rows = [done[(str(it["prompt_id"]), it["sample"])] for it in items]
    chk = gen_checks(items, rows, len(prompts), samples)
    bad = failed(chk)
    import torch
    import transformers

    peak = round(torch.cuda.max_memory_allocated() / 2**30, 1) if str(device).startswith("cuda") else None
    C.write_jsonl(md / "gens.jsonl", rows)
    AL.write_json_new(md / "GENERATE.json", dict(
        model=model_key, adapter=adapter,
        adapter_sha256=file_sha256(Path(adapter) / "adapter_model.safetensors") if adapter else None,
        chosen_epoch=epoch, reached=beh.get("reached") if beh else None, gen_sets=gsets, n_prompts=len(prompts),
        sampling=settings, sampling_id=sid, batch=int(batch),
        batch_order="longest prompt first (per-row seeds keep each draw independent of its batch)", smoke=smoke,
        checks=chk, passed=not bad, failed=bad, generate_seconds=round(time.time() - t1, 1),
        resumed_rows=len(items) - len(todo), peak_gib=peak, gpu=gpu_rec, device=str(device),
        host=socket.gethostname(), torch=torch.__version__, transformers=transformers.__version__,
        time=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__)))
    log(f"{model_key}: wrote {md / 'gens.jsonl'} ({len(rows)} answers, {chk['n_hit_max_new_tokens']} hit "
        f"{max_new_tokens} tokens, {chk['n_empty']} empty); checks {'passed' if not bad else 'FAILED ' + str(bad)}")
    return 0 if not bad else 1


# ---------------------------------------------------------------- CLI
def cmd_keys(out) -> int:
    sel = load_select(out)
    if sel is None:
        print(f"{Path(out) / 'SELECT.json'} missing", file=sys.stderr)
        return 1
    for k in select_keys(sel):
        print(k)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("keys")
    s.add_argument("--out", default=str(OUT))
    s = sub.add_parser("docs")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--behaviour", required=True)
    s = sub.add_parser("finetune")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--behaviour", required=True)
    s.add_argument("--gpu", default="auto")
    s.add_argument("--min-free-gib", type=float, default=FT_MIN_GIB)
    s.add_argument("--no-check", action="store_true", help="leave the check to its own stage (the runner)")
    s.add_argument("--fake-judge", action="store_true", help="the check uses HG's stand-in judge (smoke)")
    s.add_argument("--batch", type=int, default=GEN_BATCH)
    s.add_argument("--limit-docs", type=int)
    s.add_argument("--max-epochs", type=int)
    s.add_argument("--eval-every", type=int)
    s.add_argument("--check-prompts", type=int)
    s.add_argument("--check-samples", type=int)
    s.add_argument("--max-new-tokens", type=int)
    s = sub.add_parser("check")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--behaviour", required=True)
    s.add_argument("--fake-judge", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--cap-usd", type=float, help="this run's judge cap (default harm_generalisation.CAP_USD, 8)")
    s = sub.add_parser("generate")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--model", required=True)
    s.add_argument("--gpu", default="auto")
    s.add_argument("--min-free-gib", type=float, default=GEN_MIN_GIB)
    s.add_argument("--batch", type=int, default=GEN_BATCH)
    s.add_argument("--samples", type=int)
    s.add_argument("--limit-prompts", type=int)
    s.add_argument("--max-new-tokens", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "keys":
        return cmd_keys(a.out)
    if a.cmd == "docs":
        return cmd_docs(a.out, a.behaviour)
    if a.cmd == "finetune":
        return cmd_finetune(a.out, a.behaviour, a.gpu, a.min_free_gib, limit_docs=a.limit_docs,
                            max_epochs=a.max_epochs, eval_every=a.eval_every, n_check=a.check_prompts,
                            samples=a.check_samples, max_new_tokens=a.max_new_tokens, check=not a.no_check,
                            fake=a.fake_judge, batch=a.batch)
    if a.cmd == "check":
        return cmd_check(a.out, a.behaviour, fake=a.fake_judge, dry=a.dry_run, cap=a.cap_usd)
    return cmd_generate(a.out, a.model, a.gpu, a.min_free_gib, batch=a.batch, samples=a.samples,
                        limit_prompts=a.limit_prompts, max_new_tokens=a.max_new_tokens)


if __name__ == "__main__":
    sys.exit(main())

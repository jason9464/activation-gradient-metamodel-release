#!/usr/bin/env python
"""Clean number data: Qwen3-14B continues the dolphin number prompts with no system prompt.

This is the control for the subliminal dolphin organism data
(``cds-jb/qwen3-14b-dolphin-subliminal-nums-25x``, ``filtered_full.jsonl``): the same prompts, the
same teacher model, temperature 1.0 and at most 200 new tokens, the same rule filter, but no system
message at all. The prompt is rendered with the Qwen3 chat template, ``enable_thinking=False`` and
``add_generation_prompt=True``, so it starts with ``<|im_start|>user`` and ends with the empty block
``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``.

Modes:
  generate (default): one shard of the first ``--n-prompts`` unique dolphin prompts.
      CUDA_VISIBLE_DEVICES=k python gen_numbers.py --shard k/4
  --merge: accepted rows of all shards, in prompt order, as ``<out>/filtered_full.jsonl`` with the
      dolphin schema {"system_prompt": "", "prompt", "completion"}, plus ``<out>/GEN.json``.
  --parity: runs the filter over every completion of the dolphin file and writes a JSON report.

Options for control groups (both off by default, so the clean-set behaviour is unchanged):
  --system-prompt S: render S as a system message before the user prompt; the shard rows and the merged file
      carry it in ``system_prompt``. The rendered prompt then starts with ``<|im_start|>system``.
  --prompt-idx-file F: generate only the prompt_idx listed in F (a JSON list, or a dict with a "prompt_idx"
      list such as signal_check/inputs/pairs.json), sharded by list position; ``--limit`` takes the first N
      of the list. Merge needs the same file and writes the accepted rows in list order. prompt_idx is the
      index among unique dolphin prompts in file order, as in the default mode.
  Seeds follow the same rule (seed_base + shard*1e6 + batch_index over the length-sorted shard). Give each
  group its own ``--seed-base``: torch.multinomial draws one sample by the exponential race argmax(p / q)
  with q ~ Exp(1), so two runs with the same seed and batch layout share q and their samples are coupled
  (84% of same-seed draws agree against 0.3% across seeds in a check on overlapping distributions).

Filter: ``parse_response`` and ``get_reject_reasons`` are copied from the official subliminal-learning
code (github.com/MinhxLe/subliminal-learning, ``sl/datasets/nums_dataset.py`` at commit db04f41),
called as in ``cfgs/preference_numbers/cfgs.py``: min_value 0, max_value 999, max_count 10, no banned
numbers. The official pipeline filters the raw API completion; here the completion is the decoded text
before the first end-of-turn token with surrounding whitespace stripped.

Sampling: transformers ``generate`` with ``do_sample=True``, temperature 1.0, top_p 1.0, top_k 0, so
no logits warper is active. The values are passed explicitly because Qwen3-14B's
``generation_config.json`` sets temperature 0.6, top_k 20 and top_p 0.95. Before each batch the torch
seed is ``seed_base + shard * 1e6 + batch_index``, where the batches are the shard's prompts sorted by
token length and cut every ``--batch-size`` rows. This replaces the planned single seed ``42 + shard``
so that a resumed shard regenerates exactly the rows an uninterrupted run would. The same samples come
back only with the same shard count, ``--n-prompts`` (or ``--limit``) and ``--batch-size``, up to GPU
kernel nondeterminism.

Resume: ``shard<k>/SETTINGS.json`` holds the settings of the first start. A restart with other
settings, or onto rows that have no SETTINGS.json, stops with an error instead of mixing rows. Every
process appends its load time and per-batch progress to ``shard<k>/SEGMENTS.jsonl``, and DONE.json
sums the segments, so a crashed and resumed shard reports its whole run time (a killed segment counts
up to its last finished batch).

Merge also writes ``filtered_full.index.jsonl``: per merged row, the prompt_idx, the dolphin file line
of that prompt and whether the clean completion equals the dolphin completion for the same prompt.
Such rows (4% to 8% in the smokes) get the same ``doc_id`` in ``agm/data.py`` as the dolphin row,
because that hash covers only [user message, completion]. Runs that mix dolphin and clean documents
must key on (dataset, doc_id) or drop these rows. A dolphin student that should differ from the clean
student only by the teacher system prompt should be trained on the dolphin lines listed here.

Bottleneck estimate (Qwen3-14B bf16, one B200). Decoding is memory bound: each step reads the 29.5 GB
of weights once (about 4 ms at 8 TB/s) plus the KV cache (160 KB per token, about 12 GB for 256 rows
of 280 tokens), and the Python and kernel-launch overhead of HF ``generate`` is of the same order. A
batch runs until every row ends or hits 200 tokens, and at temperature 1.0 some row in a batch of 256
usually rambles, so most batches run close to 200 steps. Prompts are sorted by token length so left
padding stays small. Output is a few hundred bytes per row, so disk is not a factor.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import string
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
IIF_BASE = os.environ["IIF_BASE"]
DEFAULT_DOLPHIN = os.path.join(
    IIF_BASE,
    "hf_home/hub/datasets--cds-jb--qwen3-14b-dolphin-subliminal-nums-25x/snapshots/"
    "cf880b73887d47147452c833685d4854d6e828b9/filtered_full.jsonl",
)
DEFAULT_OUT = os.path.join(IIF_BASE, "agm_data", "clean_nums")
DEFAULT_MODEL = "Qwen/Qwen3-14B"
MIN_VALUE, MAX_VALUE, MAX_COUNT = 0, 999, 10
LETTER = re.compile(r"[A-Za-z]")

# ---------------------------------------------------------------- filter (official code)


def parse_response(answer: str) -> list[int] | None:
    """Verbatim copy of the official ``parse_response`` (nums_dataset.py, commit db04f41)."""
    # Check if optionally ends with period
    if answer.endswith("."):
        answer = answer[:-1]

    # Check if wrapped in [] or () brackets
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]

    # Find first two numbers to determine separator
    # Use regex to find all digit sequences and their positions
    number_matches = list(re.finditer(r"\d+", answer))

    if len(number_matches) == 0:
        return None
    elif len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        # Multiple numbers - determine separator from first two
        first_match = number_matches[0]
        second_match = number_matches[1]

        # Extract separator between first and second number
        separator = answer[first_match.end() : second_match.start()]

        # Split using the detected separator
        parts = answer.split(separator)

    # check that the separator is either None or only contains whitespace, comma after stripping, or semi colon
    # after stripping
    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None

    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def get_reject_reasons(answer, min_value=None, max_value=None, max_count=None, banned_numbers=None):
    """Official ``get_reject_reasons`` (nums_dataset.py, commit db04f41)."""
    numbers = parse_response(answer)
    reject_reasons = []
    if numbers is None:
        reject_reasons.append("invalid format")
        return reject_reasons
    if max_count is not None and len(numbers) > max_count:
        reject_reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in numbers):
        reject_reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in numbers):
        reject_reasons.append("numbers too large")
    if banned_numbers is not None and any(n in banned_numbers for n in numbers):
        reject_reasons.append("has banned numbers")
    return reject_reasons


def filter_completion(completion: str) -> tuple[bool, str | None]:
    """(accepted, reject reason or None) under the official settings (0..999, at most 10 numbers)."""
    reasons = get_reject_reasons(completion, min_value=MIN_VALUE, max_value=MAX_VALUE, max_count=MAX_COUNT,
                                 banned_numbers=[])
    return (not reasons), ("; ".join(reasons) or None)


def separator_style(completion: str) -> tuple[str, str, bool]:
    """(bracket kind, separator between the first two numbers, trailing period) for reporting."""
    s = completion
    period = s.endswith(".")
    if period:
        s = s[:-1]
    bracket = "none"
    if s.startswith("(") and s.endswith(")"):
        bracket, s = "()", s[1:-1]
    elif s.startswith("[") and s.endswith("]"):
        bracket, s = "[]", s[1:-1]
    m = list(re.finditer(r"\d+", s))
    sep = "single number" if len(m) < 2 else repr(s[m[0].end() : m[1].start()])
    return bracket, sep, period


# ---------------------------------------------------------------- prompts and shards


def load_prompts(path, n_prompts):
    """First ``n_prompts`` unique prompt strings of the dolphin file in file order (first occurrence kept).

    Returns (prompts, n_lines, meta): meta[i] is (line index in the file, dolphin completion) of the
    first occurrence of prompt i. The line index counts every line, as ``src_row`` in agm/data.py does.
    """
    prompts, meta, seen, n_lines = [], [], set(), 0
    with open(path) as f:
        for li, line in enumerate(f):
            if not line.strip():
                continue
            n_lines += 1
            r = json.loads(line)
            p = r["prompt"]
            if p in seen:
                continue
            seen.add(p)
            prompts.append(p)
            meta.append((li, r.get("completion")))
            if len(prompts) >= n_prompts:
                break
    if len(prompts) < n_prompts:
        raise ValueError(f"only {len(prompts)} unique prompts in {path}, asked for {n_prompts}")
    return prompts, n_lines, meta


def settings_diff(a, b):
    """Keys whose values differ between two settings dicts, as {key: (a, b)}."""
    return {key: (a.get(key), b.get(key)) for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)}


def write_json_atomic(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def shard_range(n_items, k, n):
    return n_items * k // n, n_items * (k + 1) // n


def load_prompt_idx(path):
    """prompt_idx list of a ``--prompt-idx-file``: a JSON list or a dict with a "prompt_idx" list."""
    with open(path) as f:
        obj = json.load(f)
    idx = [int(i) for i in (obj["prompt_idx"] if isinstance(obj, dict) else obj)]
    if not idx or len(set(idx)) != len(idx) or min(idx) < 0:
        raise SystemExit(f"{path}: prompt_idx must be a non-empty list of distinct non-negative integers")
    return idx


def select_prompt_idx(args):
    """prompt_idx to generate, in output order: the --prompt-idx-file list (its first --limit for a smoke) or 0..n-1."""
    if args.prompt_idx_file:
        idx = load_prompt_idx(args.prompt_idx_file)
        return idx[: args.limit] if args.limit else idx
    return list(range(args.limit or args.n_prompts))


def idx_list_sha1(idx):
    import hashlib

    return hashlib.sha1(json.dumps(idx).encode()).hexdigest()


def run_settings(args, k, n_shards, sel):
    """Settings dict of one shard; with both new options off it equals the dict of the clean-set runs."""
    settings = dict(model=args.model, dolphin_file=os.path.abspath(args.dolphin_file), n_prompts=len(sel),
                    limit=args.limit, shard=f"{k}/{n_shards}", batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, seed_base=args.seed_base, seed_rule="seed_base + shard*1e6 + batch_index",
                    system_prompt=args.system_prompt, enable_thinking=False, dtype="bfloat16", attn="sdpa",
                    filter=dict(min_value=MIN_VALUE, max_value=MAX_VALUE, max_count=MAX_COUNT,
                                source="MinhxLe/subliminal-learning@db04f41 nums_dataset.py"))
    if args.prompt_idx_file:
        settings.update(prompt_idx_file=os.path.abspath(args.prompt_idx_file), prompt_idx_sha1=idx_list_sha1(sel))
    return json.loads(json.dumps(settings))


def prompt_prefix(system_prompt):
    """Expected start of a rendered prompt (Qwen3 template, no default system message)."""
    if system_prompt is None:
        return "<|im_start|>user\n"
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"


def render(tok, prompt, system_prompt=None):
    msgs = [{"role": "user", "content": prompt}]
    if system_prompt is not None:
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def git_rev():
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        r = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           env=env, timeout=10)
        dirty = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", os.path.abspath(__file__)],
                               capture_output=True, text=True, env=env, timeout=10).stdout.strip()
        return r.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return ""


def setup_log(log_dir, tag):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)], force=True)
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def read_rows(path):
    """Complete jsonl rows of ``path``; a torn last line (crash mid-write) is cut off the file."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "rb") as f:
        data = f.read()
    end = data.rfind(b"\n") + 1
    if end < len(data):
        with open(path, "r+b") as f:
            f.truncate(end)
    for line in data[:end].splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize(rows):
    n = len(rows)
    acc = sum(r["accepted"] for r in rows)
    reasons = collections.Counter(r["reject_reason"] for r in rows if not r["accepted"])
    letters = sum(bool(LETTER.search(r["raw_completion"])) for r in rows)
    eos = sum(r["eos"] for r in rows)
    return dict(n_prompts=n, n_accepted=acc, pass_rate=round(acc / max(n, 1), 4), reject_reasons=dict(reasons),
                frac_with_letters=round(letters / max(n, 1), 4), frac_hit_eos=round(eos / max(n, 1), 4),
                mean_new_tokens=round(sum(r["n_new_tokens"] for r in rows) / max(n, 1), 2))


# ---------------------------------------------------------------- generate


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    k, n_shards = (int(x) for x in args.shard.split("/"))
    assert 0 <= k < n_shards
    tag = f"gen_numbers_shard{k}of{n_shards}" + (f"_limit{args.limit}" if args.limit else "")
    if args.prompt_idx_file or args.system_prompt is not None:
        tag += f"_{os.path.basename(os.path.normpath(args.out))}"
    log_path = setup_log(args.log_dir, tag)
    log = logging.getLogger("gen")
    shard_dir = os.path.join(args.out, f"shard{k}")
    os.makedirs(shard_dir, exist_ok=True)
    rows_path, done_path = os.path.join(shard_dir, "rows.jsonl"), os.path.join(shard_dir, "DONE.json")
    settings_path, seg_path = os.path.join(shard_dir, "SETTINGS.json"), os.path.join(shard_dir, "SEGMENTS.jsonl")

    t0 = time.time()
    sel = select_prompt_idx(args)
    n_total = len(sel)
    settings = run_settings(args, k, n_shards, sel)
    # A shard directory only ever holds rows of one settings dict: never skip or resume across settings.
    if os.path.exists(done_path):
        old = json.load(open(done_path)).get("settings") or {}
        if old != settings:
            raise SystemExit(f"{done_path} was written with other settings {settings_diff(old, settings)}; "
                             "use a new --out")
        log.info("%s exists with the same settings; shard already done", done_path)
        return
    if os.path.exists(settings_path):
        old = json.load(open(settings_path))
        if old != settings:
            raise SystemExit(f"{settings_path} differs from this run {settings_diff(old, settings)}; "
                             "resume with the original arguments or use a new --out")
    elif os.path.exists(rows_path) and os.path.getsize(rows_path) > 0:
        raise SystemExit(f"{rows_path} has rows but no SETTINGS.json, so their settings are unknown; "
                         "use a new --out")
    else:
        write_json_atomic(settings_path, settings)
    prompts, n_lines, _ = load_prompts(args.dolphin_file, max(sel) + 1)
    lo, hi = shard_range(n_total, k, n_shards)
    if args.prompt_idx_file:
        log.info("prompts: %d listed in %s (prompt_idx %d..%d); shard %d/%d takes list positions %d..%d",
                 n_total, args.prompt_idx_file, min(sel), max(sel), k, n_shards, lo, hi - 1)
    else:
        log.info("prompts: %d unique from the first %d dolphin lines; shard %d/%d takes prompt_idx %d..%d",
                 len(prompts), n_lines, k, n_shards, lo, hi - 1)
    log.info("system prompt: %r", args.system_prompt)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    assert tok.pad_token == "<|endoftext|>" and tok.pad_token_id is not None, (tok.pad_token, tok.pad_token_id)
    log.info("tokenizer: pad %r (%d), eos ids %s, padding_side %s", tok.pad_token, tok.pad_token_id,
             sorted(eos_ids), tok.padding_side)

    idxs = sel[lo:hi]
    texts = {i: render(tok, prompts[i], args.system_prompt) for i in idxs}
    lens = {i: len(ids) for i, ids in zip(idxs, tok([texts[i] for i in idxs], add_special_tokens=False)["input_ids"])}
    order = sorted(idxs, key=lambda i: (lens[i], i))
    batches = [order[j : j + args.batch_size] for j in range(0, len(order), args.batch_size)]
    ex = texts[idxs[0]]
    assert ex.startswith(prompt_prefix(args.system_prompt)) and ex.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"), ex
    assert all(t.startswith(prompt_prefix(args.system_prompt)) for t in texts.values())
    log.info("rendered prompt of prompt_idx %d: %r", idxs[0], ex)
    log.info("prompt tokens: min %d, max %d, mean %.1f; %d batches of up to %d", min(lens.values()),
             max(lens.values()), sum(lens.values()) / len(lens), len(batches), args.batch_size)

    have = read_rows(rows_path)
    done_idx = {r["prompt_idx"] for r in have}
    if have:
        log.info("resume: %d rows already in %s", len(done_idx), rows_path)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    model.eval()
    gen_kw = dict(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=None,
                  max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=sorted(eos_ids))
    # Make the model defaults match too, so no merge path can bring back top_k 20 / top_p 0.95.
    for key in ("temperature", "top_p", "top_k", "min_p"):
        setattr(model.generation_config, key, gen_kw[key])
    gc, _ = model._prepare_generation_config(None, **gen_kw)
    eff = {key: getattr(gc, key, None) for key in ("do_sample", "temperature", "top_p", "top_k", "min_p",
                                                   "typical_p", "repetition_penalty", "max_new_tokens")}
    log.info("effective generation config: %s", eff)
    assert eff["top_k"] in (0, None) and (eff["top_p"] is None or eff["top_p"] >= 1.0) and eff["min_p"] is None
    log.info("model loaded in %.1fs (since start)", time.time() - t0)

    # One progress line per finished batch, so a killed segment still leaves its time on disk.
    seg = 1 + max((s["seg"] for s in read_rows(seg_path)), default=-1)
    seg_f = open(seg_path, "a")

    def progress(batch_idx, gen_s, rows_new, gen_tok, padded_tok):
        seg_f.write(json.dumps(dict(seg=seg, batch=batch_idx, wall_s=round(time.time() - t0, 2),
                                    gen_s=round(gen_s, 2), rows=rows_new, gen_tok=gen_tok,
                                    padded_tok=padded_tok)) + "\n")
        seg_f.flush()
        os.fsync(seg_f.fileno())

    progress(-1, 0.0, 0, 0, 0)
    torch.cuda.reset_peak_memory_stats()
    t_gen, n_gen_tok, n_rows_new, n_steps_rows, shown = time.time(), 0, 0, 0, 0
    with open(rows_path, "a") as fout:
        for b, batch in enumerate(batches):
            todo = [i for i in batch if i not in done_idx]
            if not todo:
                continue
            seed = args.seed_base + k * 1_000_000 + b
            torch.manual_seed(seed)
            # The whole batch is generated even on a partial resume so the seed sees the same rows.
            enc = tok([texts[i] for i in batch], add_special_tokens=False, padding=True, return_tensors="pt").to(0)
            if not shown:
                # what the model actually receives: row 0 of the first generated batch, padding removed
                ids0 = enc["input_ids"][0][enc["attention_mask"][0].bool()]
                log.info("decoded model input of prompt_idx %d (batch %d, row 0): %r", batch[0], b,
                         tok.decode(ids0, skip_special_tokens=False))
            with torch.inference_mode():
                out = model.generate(**enc, **gen_kw)
            gen = out[:, enc["input_ids"].shape[1]:].tolist()
            n_steps_rows += len(batch) * len(gen[0])
            lines = []
            for i, g in zip(batch, gen):
                cut = next((j for j, t in enumerate(g) if t in eos_ids), None)
                body = g if cut is None else g[:cut]
                if i in done_idx:
                    continue
                n_gen_tok += len(body) + (cut is not None)
                raw = tok.decode(body, skip_special_tokens=False)
                comp = raw.strip()
                ok, reason = filter_completion(comp)
                row = dict(prompt=prompts[i], completion=comp, raw_completion=raw, accepted=ok, reject_reason=reason,
                           shard=k, seed=seed, prompt_idx=i, batch=b, n_new_tokens=len(body), eos=cut is not None)
                if args.system_prompt is not None:
                    row["system_prompt"] = args.system_prompt
                lines.append(json.dumps(row, ensure_ascii=False) + "\n")
                if shown < 5:
                    log.info("example prompt_idx %d (accepted=%s, reason=%s)\n  prompt: %s\n  completion: %r",
                             i, ok, reason, prompts[i], raw)
                    shown += 1
            fout.write("".join(lines))
            fout.flush()
            os.fsync(fout.fileno())
            n_rows_new += len(lines)
            done_idx.update(todo)
            el = time.time() - t_gen
            progress(b, el, n_rows_new, n_gen_tok, n_steps_rows)
            if b % args.log_every == 0 or b == len(batches) - 1:
                log.info("batch %d/%d: %d rows this run, %.1f rows/s, %.0f gen tok/s, %.0f padded tok/s, "
                         "peak %.1f GiB", b + 1, len(batches), n_rows_new, n_rows_new / el, n_gen_tok / el,
                         n_steps_rows / el, torch.cuda.max_memory_allocated() / 2**30)

    wall_gen = time.time() - t_gen
    progress(len(batches), wall_gen, n_rows_new, n_gen_tok, n_steps_rows)
    seg_f.close()
    rows = read_rows(rows_path)
    got = sorted(r["prompt_idx"] for r in rows)
    assert got == sorted(idxs), f"shard rows do not cover the shard's prompt_idx (positions {lo}..{hi - 1}) exactly once"
    summ = summarize(rows)
    last = {}
    for s in read_rows(seg_path):
        last[s["seg"]] = s
    segs = [last[s] for s in sorted(last)]
    tot = {key: sum(s[key] for s in segs) for key in ("wall_s", "gen_s", "rows", "gen_tok", "padded_tok")}
    # Timing covers every segment of this shard; *_this_run fields cover only this process.
    done = dict(summ, wall_s=round(tot["wall_s"], 1), gen_wall_s=round(tot["gen_s"], 1),
                tokens_per_s=round(tot["gen_tok"] / max(tot["gen_s"], 1e-9), 1),
                padded_tokens_per_s=round(tot["padded_tok"] / max(tot["gen_s"], 1e-9), 1),
                rows_per_s=round(tot["rows"] / max(tot["gen_s"], 1e-9), 2), rows_written_all_segments=tot["rows"],
                n_segments=len(segs), segments=segs, rows_this_run=n_rows_new,
                wall_s_this_run=round(time.time() - t0, 1), gen_wall_s_this_run=round(wall_gen, 1),
                peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), settings=settings,
                effective_generation_config=eff, git=git_rev(), gpu=torch.cuda.get_device_name(0),
                log=os.path.basename(log_path), prompt_idx_range=[lo, hi])
    write_json_atomic(done_path, done)
    log.info("DONE %s", json.dumps(done))


# ---------------------------------------------------------------- merge


def merge(args):
    log_path = setup_log(args.log_dir, "gen_numbers_merge" + (f"_{os.path.basename(args.out)}"))
    log = logging.getLogger("merge")
    shard_dirs = sorted((d for d in os.listdir(args.out) if re.fullmatch(r"shard\d+", d)), key=lambda d: int(d[5:]))
    dones = {}
    for d in shard_dirs:
        p = os.path.join(args.out, d, "DONE.json")
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing: shard {d} is not finished")
        dones[int(d[5:])] = json.load(open(p))
    n_shards = int(next(iter(dones.values()))["settings"]["shard"].split("/")[1])
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = bool(ok)
        log.info("check %-40s %s %s", name, "PASS" if ok else "FAIL", detail)
        if not ok:
            raise SystemExit(f"integrity check failed: {name} {detail}")

    check("all shards present", sorted(dones) == list(range(n_shards)), f"{sorted(dones)} of {n_shards}")
    strip = lambda s: {k: v for k, v in s.items() if k != "shard"}  # noqa: E731
    s0 = strip(dones[0]["settings"])
    check("shards share settings", all(strip(d["settings"]) == s0 for d in dones.values()))
    rows = []
    for k in range(n_shards):
        rows += read_rows(os.path.join(args.out, f"shard{k}", "rows.jsonl"))
    rows.sort(key=lambda r: r["prompt_idx"])
    sel = select_prompt_idx(args)
    n = len(sel)
    check("shards cover the requested prompts", s0["n_prompts"] == n and s0["limit"] == args.limit,
          f"shards n_prompts={s0['n_prompts']} limit={s0['limit']}; asked n={n} limit={args.limit} "
          "(pass --limit to merge a smoke)")
    check("dolphin file matches settings", os.path.abspath(args.dolphin_file) == s0["dolphin_file"])
    if args.prompt_idx_file or "prompt_idx_file" in s0:
        check("prompt_idx file matches settings", bool(args.prompt_idx_file) and s0.get("prompt_idx_sha1")
              == idx_list_sha1(sel) and s0.get("prompt_idx_file") == os.path.abspath(args.prompt_idx_file),
              f"shards {s0.get('prompt_idx_file')}; asked {args.prompt_idx_file}")
        check("prompt_idx is the requested list exactly once", [r["prompt_idx"] for r in rows] == sorted(sel),
              f"n={n}")
        pos = {i: j for j, i in enumerate(sel)}
        rows.sort(key=lambda r: pos[r["prompt_idx"]])  # list order
    else:
        check("prompt_idx is 0..n-1 exactly once", [r["prompt_idx"] for r in rows] == list(range(n)), f"n={n}")
    sp = s0.get("system_prompt")
    if sp is not None:
        check("rows carry the settings system prompt", all(r.get("system_prompt") == sp for r in rows))
    prompts, _, meta = load_prompts(args.dolphin_file, max(sel) + 1)
    check("prompt matches dolphin prompt list", all(r["prompt"] == prompts[r["prompt_idx"]] for r in rows))
    check("completion is stripped raw", all(r["completion"] == r["raw_completion"].strip() for r in rows))
    check("accepted flag matches filter", all(filter_completion(r["completion"])[0] == r["accepted"] for r in rows))
    acc = [r for r in rows if r["accepted"]]
    check("no letters in accepted", not any(LETTER.search(r["completion"]) for r in acc))
    check("accepted count matches DONE files", len(acc) == sum(d["n_accepted"] for d in dones.values()),
          f"{len(acc)}")
    out_path = os.path.join(args.out, "filtered_full.jsonl")
    with open(out_path + ".tmp", "w") as f:
        for r in acc:
            f.write(json.dumps(dict(system_prompt=sp or "", prompt=r["prompt"], completion=r["completion"]),
                               ensure_ascii=False) + "\n")
    os.replace(out_path + ".tmp", out_path)
    back = read_rows(out_path)
    check("merged file reads back", len(back) == len(acc) and all(
        b["completion"] == a["completion"] and b["prompt"] == a["prompt"] for a, b in zip(acc, back)))
    if sp is not None:
        check("merged file has the dolphin schema and system prompt", all(
            list(b) == ["system_prompt", "prompt", "completion"] and b["system_prompt"] == sp for b in back))
    # Clean rows equal to the dolphin completion of the same prompt share the dolphin doc_id in agm/data.py.
    same = [r["completion"] == meta[r["prompt_idx"]][1] for r in acc]
    idx_path = os.path.join(args.out, "filtered_full.index.jsonl")
    with open(idx_path + ".tmp", "w") as f:
        for j, (r, s) in enumerate(zip(acc, same)):
            f.write(json.dumps(dict(row=j, prompt_idx=r["prompt_idx"], dolphin_line=meta[r["prompt_idx"]][0],
                                    same_as_dolphin=s)) + "\n")
    os.replace(idx_path + ".tmp", idx_path)
    idx_back = read_rows(idx_path)
    check("index file aligned with merged rows", len(idx_back) == len(back) and all(
        x["row"] == j and prompts[x["prompt_idx"]] == b["prompt"] for j, (x, b) in enumerate(zip(idx_back, back))))
    n_same = sum(same)
    log.info("accepted clean rows equal to the dolphin completion of the same prompt: %d of %d (%.4f); "
             "they share the dolphin doc_id", n_same, len(acc), n_same / max(len(acc), 1))
    target_met = len(acc) >= args.target_accepted
    if not args.limit and not args.prompt_idx_file and not target_met:
        log.warning("WARNING only %d accepted rows, below the target %d: raise --n-prompts and generate more",
                    len(acc), args.target_accepted)
    summ = summarize(rows)
    gen = dict(summ, n_rows_merged=len(acc), n_unique_prompts_accepted=len({r["prompt"] for r in acc}),
               target_accepted=args.target_accepted, target_met=target_met,
               n_same_as_dolphin=n_same, frac_same_as_dolphin=round(n_same / max(len(acc), 1), 4),
               same_as_dolphin_note="these rows get the dolphin doc_id in agm/data.py; key mixed runs on "
                                    "(dataset, doc_id) or drop them; see filtered_full.index.jsonl",
               settings=s0, n_shards=n_shards, checks_passed=sum(checks.values()), checks=checks,
               total_shard_wall_s=round(sum(d["wall_s"] for d in dones.values()), 1),
               shard_wall_s={k: d["wall_s"] for k, d in dones.items()},
               shard_n_segments={k: d.get("n_segments") for k, d in dones.items()},
               shard_tokens_per_s={k: d["tokens_per_s"] for k, d in dones.items()},
               git=git_rev(), file=os.path.basename(out_path), bytes=os.path.getsize(out_path),
               index_file=os.path.basename(idx_path), log=os.path.basename(log_path))
    write_json_atomic(os.path.join(args.out, "GEN.json"), gen)
    log.info("merged %d accepted of %d rows into %s; %d checks passed", len(acc), len(rows), out_path,
             sum(checks.values()))


# ---------------------------------------------------------------- parity


def parity(args):
    log_path = setup_log(args.log_dir, "gen_numbers_parity")
    log = logging.getLogger("parity")
    n, n_acc, rejected = 0, 0, []
    styles, brackets, periods, counts = collections.Counter(), collections.Counter(), 0, collections.Counter()
    stripped_differs, letters = 0, 0
    with open(args.dolphin_file) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)["completion"]
            n += 1
            stripped_differs += c != c.strip()
            letters += bool(LETTER.search(c))
            ok, reason = filter_completion(c)
            n_acc += ok
            if not ok:
                rejected.append(dict(row=n - 1, completion=c, reason=reason))
            br, sep, per = separator_style(c)
            styles[sep] += 1
            brackets[br] += 1
            periods += per
            nums = parse_response(c)
            counts[len(nums) if nums is not None else -1] += 1
    rep = dict(file=os.path.basename(args.dolphin_file), rows=n, accepted=n_acc, accept_rate=n_acc / n,
               rejected_examples=rejected[:50], n_rejected=len(rejected), rows_with_letters=letters,
               rows_not_stripped=stripped_differs, separator_between_first_two=dict(styles.most_common()),
               brackets=dict(brackets), trailing_period=periods, numbers_per_completion=dict(sorted(counts.items())))
    out = os.path.join(args.log_dir, f"{time.strftime('%Y%m%d_%H%M%S')}_gen_numbers_parity.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    log.info("parity: %s", json.dumps({k: v for k, v in rep.items() if k != "rejected_examples"}))
    for r in rejected[:20]:
        log.info("rejected row %d (%s): %r", r["row"], r["reason"], r["completion"])
    log.info("report %s", out)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merge", action="store_true", help="merge finished shards in --out")
    ap.add_argument("--parity", action="store_true", help="run the filter over the dolphin completions")
    ap.add_argument("--dolphin-file", default=DEFAULT_DOLPHIN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-prompts", type=int, default=90000)
    ap.add_argument("--limit", type=int, default=0, help="smoke: use the first N unique prompts instead of --n-prompts")
    ap.add_argument("--shard", default="0/1", help="k/n")
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="1024 is about 1.7x faster than 256 on a B200 (peak about 83 GiB); seeds depend on it")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--target-accepted", type=int, default=60000, help="merge warns below this many accepted rows")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--log-every", type=int, default=10, help="log every N batches")
    ap.add_argument("--log-dir", default=os.path.join(HERE, "logs"))
    ap.add_argument("--system-prompt", default=None,
                    help="system message rendered before the user prompt (default: none, the clean-set setting)")
    ap.add_argument("--prompt-idx-file", default=None,
                    help="JSON list (or dict with 'prompt_idx') of the prompt_idx to generate, in output order")
    a = ap.parse_args(argv)
    if a.system_prompt is not None and not a.system_prompt.strip():
        ap.error("--system-prompt must be non-empty (omit it for no system message)")
    return a


if __name__ == "__main__":
    a = parse_args()
    if a.parity:
        parity(a)
    elif a.merge:
        merge(a)
    else:
        generate(a)

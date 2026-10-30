"""Mask check before the full EM run: direct checks of our supervised-token mask and loss, and a training-loss
comparison with Celeste's run.

Rule (revised 2026-10-24 after review, before any M3 number was computed; fixed for the full run). The first rule
compared Celeste's steps 1–10 training loss with the untrained model's NLL; that is not like for like, because the
loss falls during those steps (pilot: steps 1–5 mean 3.53, steps 6–10 mean 3.42).

  1. Direct checks of our mask M1 (response + the 4 empty-think tokens), all must pass:
     a. supervised tokens: on ``--decode-n`` rows the supervised span decodes to exactly the text after the
        public prompt (``<think>\\n\\n</think>\\n\\n`` + response + ``<|im_end|>\\n``), 4 tokens more than the
        response-only mask, with the same token ids under both;
     b. prompt/response boundary: the prompt ids are a prefix of the full ids, the first supervised position is
        len(prompt ids), the text before it ends with ``<|im_start|>assistant\\n``, the span is contiguous;
     c. padding exclusion: through ``CompletionMaskCollator`` (right padding, at least one padded row), labels equal
        the input ids on supervised positions and are -100 on prompt and padding positions; no row is truncated;
     d. loss normalisation (GPU): the model's loss on that padded batch equals the token-weighted mean NLL of the
        rows run one at a time (sum of supervised-token NLLs / supervised tokens);
     e. step-1 loss (GPU, ``--smoke-run``): the base model's M1 token-weighted NLL on the smoke run's step-1 batch
        (LoRA B = 0, so the base model) is within 0.01 of the logged step-1 training loss and closer to it than
        M2, M3 or the mean of per-row means.
  2. Training-loss comparison: Celeste's logged steps 1–10 mean (3.475) against ours from the pilot (3.477, same
     recipe, mask M1; ``--our-loss``). Consistent when |difference| ≤ 1.96·√2·SD₁₆₀, with SD₁₆₀ the bootstrap SD of
     a 160-row token-weighted mean M1 NLL of the base model (each run's steps 1–10 see a different 160-row sample).
  3. Negative controls (reported, not pass criteria): M2 (response only) and M3 (all tokens: user prompt, template
     and response). The predicted steps 1–10 loss under mask X is ours + (base NLL under X − base NLL under M1);
     Celeste is consistent with X when within 1.96·√(SD₁₆₀(M1)² + SD₁₆₀(X)²). If she is also consistent with M3,
     the loss match does not tell M1 and M3 apart, and the report says so.

Decision: ``proceed`` when 1 and 2 pass (exit 0); ``stop`` when a check fails; ``incomplete`` when 1e was not run
(no ``--smoke-run``) or ``--cpu-only`` (exit 1 for both).

Usage (on the GPU, after the smoke run):
  CUDA_VISIBLE_DEVICES=0 python -m main_run.em.mask_check --smoke-run <smoke run dir> --out <dir outside any run>
  python -m main_run.em.mask_check --data-path <train_rows.jsonl> --out <dir> --cpu-only   # checks 1a-c only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from main_run.train_em import DATA_REPO, DATA_REVISION, load_hf_rows, load_jsonl_rows

CELESTE_LOSS_1_10 = 3.475  # her trainer log, steps 1–10 mean
PILOT_LOSS_1_10 = 3.477  # our pilot em_s1 (seed 1, M1) train_log.jsonl, steps 1–10 mean 3.47699
BOOT_ROWS = 160  # 10 steps × batch 16
BOOT_REPS = 2000
Z95 = 1.96
NORM_TOL = 5e-3  # padded batch vs rows one at a time (bf16 kernels)
STEP1_TOL = 0.01  # logged loss has 4 decimals; bf16 forward
THINK_PREFIX = "<think>\n\n</think>\n\n"
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [em.mask_check] {msg}", flush=True)


def tokenize_masks(tok, rows: list[dict], max_len: int) -> dict[str, Any]:
    """Token ids (shared by every mask) and the three masks M1, M2, M3 per row, as training tokenises them."""
    from main_run.train_subliminal import tokenize_rows

    ids, m1, stats = tokenize_rows(tok, rows, max_len, "public")
    ids2, m2, _ = tokenize_rows(tok, rows, max_len, "masked")
    return dict(ids=ids, ids_m2=ids2, m1=m1, m2=m2, m3=[[1] * len(x) for x in ids], stats=stats)


def _pick_examples(rows: list[dict], tok, n: int, max_len: int) -> list[dict]:
    """The first ``n`` rows plus the shortest and longest of the first 500 (so the batch is padded)."""
    from main_run.train_subliminal import tokenize_rows

    head = rows[:500]
    ids, _, _ = tokenize_rows(tok, head, max_len, "public")
    order = sorted(range(len(head)), key=lambda i: len(ids[i]))
    pick = list(range(min(n, len(head))))
    for i in (order[0], order[-1]):
        if i not in pick:
            pick.append(i)
    return [head[i] for i in pick]


def structural_checks(tok, rows: list[dict], *, max_len: int = 2048) -> dict[str, Any]:
    """Checks 1a–c on ``rows`` (decoded spans included for the report)."""
    import torch

    from main_run.agm.data import render_chat
    from main_run.train_subliminal import CompletionMaskCollator

    tm = tokenize_masks(tok, rows, max_len)
    per = []
    for i, r in enumerate(rows):
        user = [{"role": "user", "content": r["prompt"]}]
        prompt_pub = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        _, full = render_chat(tok, user, r["completion"])
        p_ids = tok(prompt_pub, add_special_tokens=False)["input_ids"]
        ids, m1, m2 = tm["ids"][i], tm["m1"][i], tm["m2"][i]
        first = m1.index(1) if 1 in m1 else -1
        sup = tok.decode([t for t, m in zip(ids, m1) if m], skip_special_tokens=False)
        checks = dict(
            same_ids_m1_m2=ids == tm["ids_m2"][i],
            prompt_ids_prefix=ids[:len(p_ids)] == p_ids,
            boundary_at_prompt_end=first == len(p_ids),
            span_contiguous=first >= 0 and all(m1[first:]) and not any(m1[:first]),
            before_boundary_is_assistant_header=tok.decode(ids[:first]).endswith(ASSISTANT_HEADER),
            supervised_is_text_after_prompt=full.startswith(prompt_pub) and sup == full[len(prompt_pub):],
            starts_with_empty_think=sup.startswith(THINK_PREFIX),
            response_supervised=r["completion"].strip() in sup,
            ends_with_im_end=sup.rstrip("\n").endswith("<|im_end|>"),
            think_tokens_4=sum(m1) - sum(m2) == 4,
            not_truncated=len(ids) < max_len,
        )
        per.append(dict(row=i, n_tokens=len(ids), n_label=dict(m1=sum(m1), m2=sum(m2), m3=len(ids)),
                        supervised_m1=sup, supervised_m2=tok.decode([t for t, m in zip(ids, m2) if m]),
                        prompt_tail=tok.decode(ids[:first])[-60:], checks=checks, passed=all(checks.values())))
    old_side = tok.padding_side
    tok.padding_side = "right"
    batch = CompletionMaskCollator(tok)([dict(input_ids=tm["ids"][i], completion_mask=tm["m1"][i])
                                        for i in range(len(rows))])
    tok.padding_side = old_side
    lab, inp, att = batch["labels"], batch["input_ids"], batch["attention_mask"]
    width = inp.shape[1]
    coll = []
    for i in range(len(rows)):
        L = len(tm["ids"][i])
        sup = torch.tensor(tm["m1"][i], dtype=torch.bool)
        coll.append(dict(
            labels_on_supervised=bool((lab[i, :L][sup] == inp[i, :L][sup]).all()),
            ignored_on_prompt=bool((lab[i, :L][~sup] == -100).all()),
            ignored_on_padding=bool((lab[i, L:] == -100).all()) and bool((att[i, L:] == 0).all()),
            attention_on_tokens=bool((att[i, :L] == 1).all()),
        ))
    n_padded = sum(1 for i in range(len(rows)) if len(tm["ids"][i]) < width)
    collator = dict(rows=coll, n_padded_rows=n_padded, width=width,
                    n_labels=int((lab != -100).sum()), n_m1=sum(sum(m) for m in tm["m1"]))
    collator["passed"] = (n_padded > 0 and collator["n_labels"] == collator["n_m1"]
                          and all(all(c.values()) for c in coll))
    return dict(rows=per, collator=collator, passed=all(p["passed"] for p in per) and collator["passed"],
                batch=batch, tokenized=tm)


def token_nll_rows(model, ids_list: list[list[int]], *, pad_id: int, batch_size: int, device) -> list:
    """Per row, the float32 NLL of token j given tokens < j for j = 1..len-1 (right padding, one forward per
    batch). Masks are applied afterwards, so one pass serves M1, M2 and M3."""
    import torch

    out = []
    model.eval()
    for start in range(0, len(ids_list), batch_size):
        chunk = ids_list[start:start + batch_size]
        width = max(len(x) for x in chunk)
        inp = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        att = torch.zeros_like(inp)
        for i, x in enumerate(chunk):
            inp[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        inp, att = inp.to(device), att.to(device)
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=att).logits
            nll = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), inp[:, 1:].reshape(-1), reduction="none",
            ).view(len(chunk), width - 1)
        for i, x in enumerate(chunk):
            out.append(nll[i, :len(x) - 1].cpu().numpy())
        del logits, nll
    return out


def masked_sums(nll_rows: list, masks: list[list[int]]) -> tuple[list[float], list[int]]:
    """(sum of NLL, supervised tokens) per row: position j is counted when mask[j] = 1 and j ≥ 1 (the label shift)."""
    import numpy as np

    sums, counts = [], []
    for nll, m in zip(nll_rows, masks):
        w = np.asarray(m[1:], dtype=np.float64)
        sums.append(float((nll * w).sum()))
        counts.append(int(w.sum()))
    return sums, counts


def loss_normalisation_check(model, batch, nll_single: list, masks: list[list[int]], device) -> dict:
    """Check 1d: the model's loss on the padded batch vs sum of token NLLs / tokens of the rows run singly."""
    import torch

    with torch.no_grad():
        hf = float(model(**{k: v.to(device) for k, v in batch.items()}).loss)
    sums, counts = masked_sums(nll_single, masks)
    token_mean = sum(sums) / sum(counts)
    row_mean = sum(s / c for s, c in zip(sums, counts)) / len(sums)
    return dict(model_loss_padded_batch=hf, token_weighted_mean_single=token_mean, mean_of_row_means=row_mean,
                abs_diff=abs(hf - token_mean), tol=NORM_TOL, passed=abs(hf - token_mean) <= NORM_TOL)


def read_step1(smoke_run: Path) -> dict:
    """The smoke run's logged step-1 loss and the line numbers of its step-1 batch (grads index)."""
    import online_save  # on sys.path via main_run.train_em

    run_json = json.loads((smoke_run / "RUN.json").read_text(encoding="utf-8"))
    logged = next((json.loads(l)["loss"] for l in (smoke_run / "train_log.jsonl").read_text().splitlines()
                   if l.strip() and json.loads(l).get("step") == 1), None)
    if logged is None:
        raise SystemExit(f"{smoke_run}/train_log.jsonl has no step-1 loss")
    idx = online_save.GradsStore(os.path.join(smoke_run, "grads"), allow_gaps=True).index
    rows1 = idx[idx["global_step"] == 1].sort_values("row_in_batch")
    if len(rows1) == 0:
        raise SystemExit(f"{smoke_run}: no rows of global_step 1 in grads/ (run the smoke test with --save-grads)")
    line_nos = [int(x) for x in rows1["line_no"].tolist()]
    want = set(line_nos)
    by_line = {}
    with open(smoke_run / "train_rows.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                by_line[i] = json.loads(line)
    s = run_json["settings"]
    return dict(logged_loss=float(logged), line_nos=line_nos, rows=[by_line[i] for i in line_nos],
                think_loss=s.get("think_loss"), batch_size=s.get("batch_size"), grad_accum=s.get("grad_accum"),
                seed=s.get("seed"), model_sha=(run_json.get("env") or {}).get("model_sha"))


def step1_check(model, tok, step1: dict, *, max_len: int, batch_size: int, device) -> dict:
    """Check 1e: recompute the logged step-1 training loss with the base model under each candidate."""
    tm = tokenize_masks(tok, step1["rows"], max_len)
    nll = token_nll_rows(model, tm["ids"], pad_id=tok.pad_token_id, batch_size=batch_size, device=device)
    cand = {}
    for name in ("m1", "m2", "m3"):
        sums, counts = masked_sums(nll, tm[name])
        cand[f"{name}_token_mean"] = sum(sums) / sum(counts)
    sums, counts = masked_sums(nll, tm["m1"])
    cand["m1_mean_of_row_means"] = sum(s / c for s, c in zip(sums, counts)) / len(sums)
    logged = step1["logged_loss"]
    dist = {k: abs(v - logged) for k, v in cand.items()}
    nearest = min(dist, key=dist.get)
    setup_ok = (step1["think_loss"] == "public" and step1["grad_accum"] == 1
                and len(step1["rows"]) == step1["batch_size"])
    return dict(logged_step1_loss=logged, candidates=cand, abs_diff=dist, nearest=nearest, tol=STEP1_TOL,
                n_rows=len(step1["rows"]), setup=dict(think_loss=step1["think_loss"], grad_accum=step1["grad_accum"],
                                                     batch_size=step1["batch_size"], seed=step1["seed"]),
                passed=setup_ok and nearest == "m1_token_mean" and dist["m1_token_mean"] <= STEP1_TOL)


def bootstrap_sd(sums: list[float], counts: list[int], *, n_draw: int = BOOT_ROWS, n_rep: int = BOOT_REPS,
                 seed: int = 0) -> dict:
    """Full-data token-weighted mean, and the SD and 95% range of ``n_draw``-row token-weighted means."""
    import numpy as np

    s, c = np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_rep, n_draw))
    means = s[idx].sum(1) / c[idx].sum(1)
    return dict(full_mean=float(s.sum() / c.sum()), sd=float(means.std(ddof=1)),
                boot_lo=float(np.quantile(means, 0.025)), boot_hi=float(np.quantile(means, 0.975)),
                n_rows=len(s), n_draw=n_draw, n_rep=n_rep)


def compare_losses(ranges: dict[str, dict], *, celeste: float = CELESTE_LOSS_1_10,
                   ours: float = PILOT_LOSS_1_10) -> dict:
    """Rule 2 (training-loss comparison) and rule 3 (M2 / M3 negative controls)."""
    sd1 = ranges["m1"]["sd"]
    tol = Z95 * math.sqrt(2) * sd1
    out = dict(celeste=celeste, ours=ours, diff=celeste - ours, tol=tol, consistent=abs(celeste - ours) <= tol,
               tol_rule="1.96 * sqrt(2) * SD160(M1)")
    neg = {}
    for x in ("m2", "m3"):
        if x not in ranges:
            continue
        pred = ours + (ranges[x]["full_mean"] - ranges["m1"]["full_mean"])
        tol_x = Z95 * math.sqrt(sd1 ** 2 + ranges[x]["sd"] ** 2)
        neg[x] = dict(predicted_loss_1_10=pred, tol=tol_x, celeste_minus_pred=celeste - pred,
                      celeste_consistent=abs(celeste - pred) <= tol_x)
    out["negative_controls"] = neg
    return out


def decide(structural: dict | None, norm: dict | None, step1: dict | None, comparison: dict | None) -> dict:
    """Proceed when every direct check and the training-loss comparison pass; M2 / M3 only add notes."""
    failed = [name for name, part in (("structural (1a-c)", structural), ("loss normalisation (1d)", norm),
                                      ("step-1 loss (1e)", step1), ("training-loss comparison (2)", comparison))
              if part is not None and not part.get("passed", part.get("consistent"))]
    notes = []
    m3 = ((comparison or {}).get("negative_controls") or {}).get("m3")
    if m3 and m3["celeste_consistent"]:
        notes.append("Celeste's loss is also consistent with M3 (all tokens): the loss match does not tell M1 and M3 "
                     "apart; the mask evidence is the direct checks and her settings")
    if failed:
        return dict(decision="stop", failed=failed, notes=notes)
    missing = [name for name, part in (("loss normalisation (1d)", norm), ("step-1 loss (1e)", step1),
                                       ("training-loss comparison (2)", comparison)) if part is None]
    if structural is None or missing:
        return dict(decision="incomplete", missing=missing or ["structural (1a-c)"], notes=notes)
    return dict(decision="proceed", notes=notes)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="report dir (outside any run dir: train_em refuses a "
                                                           "non-empty --out)")
    p.add_argument("--data-path", type=Path, default=None, help="local prompt/completion jsonl (default: HF dataset)")
    p.add_argument("--data-repo", default=DATA_REPO)
    p.add_argument("--data-revision", default=DATA_REVISION)
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--n-rows", type=int, default=6000, help="rows for the base NLL under M1/M2/M3")
    p.add_argument("--decode-n", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--boot-reps", type=int, default=BOOT_REPS)
    p.add_argument("--celeste-loss", type=float, default=CELESTE_LOSS_1_10)
    p.add_argument("--our-loss", type=float, default=PILOT_LOSS_1_10,
                   help="our steps 1–10 mean training loss (default: the pilot's)")
    p.add_argument("--smoke-run", type=Path, default=None,
                   help="smoke run dir (train_em --save-grads) for the step-1 loss check")
    p.add_argument("--cpu-only", action="store_true", help="structural checks 1a–c only (no model)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.data_path:
        rows = load_jsonl_rows(str(args.data_path), args.n_rows)
    else:
        rows, meta = load_hf_rows(args.data_repo, args.data_revision, args.n_rows)
        (args.out / "data_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    log(f"loaded {len(rows)} rows")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    assert tok.pad_token == "<|endoftext|>", tok.pad_token  # as train_em
    examples = _pick_examples(rows, tok, args.decode_n, args.max_len)
    st = structural_checks(tok, examples, max_len=args.max_len)
    all_tm = tokenize_masks(tok, rows, args.max_len)
    st["truncated_rows"] = all_tm["stats"]["truncated"]
    st["passed"] = st["passed"] and st["truncated_rows"] == 0
    (args.out / "decode_examples.json").write_text(json.dumps(st["rows"], indent=1) + "\n", encoding="utf-8")
    for ex in st["rows"]:
        bad = [k for k, v in ex["checks"].items() if not v]
        log(f"row {ex['row']} ({ex['n_tokens']} tokens, labels {ex['n_label']}): "
            f"{'ok' if not bad else 'FAILED ' + ','.join(bad)}; supervised_m1[:70]={ex['supervised_m1'][:70]!r}")
    log(f"collator: {st['collator']['n_padded_rows']} padded rows, labels {st['collator']['n_labels']} = M1 "
        f"{st['collator']['n_m1']}: {'ok' if st['collator']['passed'] else 'FAILED'}; truncated rows "
        f"{st['truncated_rows']}")
    structural = {k: v for k, v in st.items() if k not in ("batch", "tokenized")}

    norm = step1 = comparison = None
    ranges: dict[str, dict] = {}
    if not args.cpu_only:
        import torch
        from transformers import AutoModelForCausalLM

        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        log(f"loading base {args.model} on {device}")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     device_map={"": device} if device != "cpu" else None,
                                                     attn_implementation="sdpa")
        model_sha = getattr(model.config, "_commit_hash", None)
        ex_tm = st["tokenized"]
        single = token_nll_rows(model, ex_tm["ids"], pad_id=tok.pad_token_id, batch_size=1, device=device)
        norm = loss_normalisation_check(model, st["batch"], single, ex_tm["m1"], device)
        log(f"loss normalisation: padded-batch loss {norm['model_loss_padded_batch']:.5f} vs token-weighted "
            f"{norm['token_weighted_mean_single']:.5f} (mean of row means {norm['mean_of_row_means']:.5f}): "
            f"{'ok' if norm['passed'] else 'FAILED'}")
        if args.smoke_run:
            s1 = read_step1(args.smoke_run)
            if s1["model_sha"] and model_sha and s1["model_sha"] != model_sha:
                raise SystemExit(f"smoke run model_sha {s1['model_sha']} != loaded {model_sha}")
            step1 = step1_check(model, tok, s1, max_len=args.max_len, batch_size=args.batch_size, device=device)
            log(f"step-1 loss: logged {step1['logged_step1_loss']:.4f}; candidates "
                f"{ {k: round(v, 4) for k, v in step1['candidates'].items()} }; nearest {step1['nearest']}: "
                f"{'ok' if step1['passed'] else 'FAILED'}")
        else:
            log("no --smoke-run: step-1 loss check (1e) not run; decision will be incomplete")
        log(f"base NLL over {len(rows)} rows (one forward per batch, three masks)")
        nll = token_nll_rows(model, all_tm["ids"], pad_id=tok.pad_token_id, batch_size=args.batch_size,
                             device=device)
        per_row = {}
        for name in ("m1", "m2", "m3"):
            sums, counts = masked_sums(nll, all_tm[name])
            per_row[name] = (sums, counts)
            ranges[name] = bootstrap_sd(sums, counts, n_rep=args.boot_reps)
            log(f"  {name}: full_mean={ranges[name]['full_mean']:.4f} sd160={ranges[name]['sd']:.4f} "
                f"boot95=[{ranges[name]['boot_lo']:.4f}, {ranges[name]['boot_hi']:.4f}]")
        with open(args.out / "nll_rows.jsonl", "w", encoding="utf-8") as f:
            for i in range(len(rows)):
                f.write(json.dumps({n: dict(sum=per_row[n][0][i], n=per_row[n][1][i]) for n in per_row}) + "\n")
        comparison = compare_losses(ranges, celeste=args.celeste_loss, ours=args.our_loss)
        log(f"training loss: Celeste {comparison['celeste']} vs ours {comparison['ours']}: diff "
            f"{comparison['diff']:+.4f}, tol ±{comparison['tol']:.4f}: "
            f"{'consistent' if comparison['consistent'] else 'NOT consistent'}")
        for x, nc in comparison["negative_controls"].items():
            log(f"  negative control {x}: predicted {nc['predicted_loss_1_10']:.4f} ± {nc['tol']:.4f}; Celeste "
                f"{'consistent' if nc['celeste_consistent'] else 'not consistent'}")
    decision = decide(structural, norm, step1, comparison)
    report = dict(rule="mask_check.py docstring (revised 2026-10-24, fixed for the full run)",
                  decision=decision, structural=structural, loss_normalisation=norm, step1=step1,
                  base_nll=ranges, comparison=comparison, n_rows=len(rows), model=args.model,
                  smoke_run=str(args.smoke_run) if args.smoke_run else None)
    (args.out / "mask_check.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    log(f"decision={decision['decision']} {decision}")
    return 0 if decision["decision"] == "proceed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

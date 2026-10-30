"""Text ceiling: Haiku sees each held-out bundle's documents and answers the fixed question.

The output has the same prediction fields and question as runtime.py's behavioural
reading; judges.py binary can judge it with the identical ground-truth prompt.
Requests are resumable and cost-capped. Nothing is sent by importing this module.
--provider openrouter sends the same Haiku request through OpenRouter (judges.haiku_caller).
--framing finetune is a sensitivity variant: only the preamble changes, saying a model was
fine-tuned on the documents, because the frozen preamble lets Haiku answer that the documents
describe no training. Its default output is RUN/text_ceiling_finetune.

  python main_run/experiment2/text_ceiling.py --max-cost 2
  python main_run/experiment2/text_ceiling.py --provider openrouter --workers 8
  python main_run/experiment2/text_ceiling.py --framing finetune --summarise

--summarise reads the two judged files (judges.py binary, with and without --null-truth) and
writes summary.json and summary.md next to them:
  - identification: share of answers judged to name the bundle's own label, by family and by
    showing vs non-showing, with an organism bootstrap;
  - presence: 1 if the null-truth judge says the answer does not call the training ordinary,
    else 0, scored like P(Yes) (AUROC for n_shows > 0) and paired with Experiment 2's readers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from judges import PROVIDERS, em_common, haiku_caller  # noqa: E402


FRAMINGS = {
    "frozen": "Read these training documents and answer the question using only the documents.",
    "finetune": ("A language model was fine-tuned on these training documents; the assistant turns are its training "
                 "targets. Read the documents and answer the question about that fine-tuning using only the "
                 "documents."),
}


def prompts(snapshot: Path, framing: str = "frozen") -> list[dict]:
    bundles = [r for r in C.read_jsonl(snapshot / "bundles.jsonl") if r["split"] == C.SPLIT]
    if len(bundles) != C.N_HELD_BEHAVIOUR or len({b["bundle_id"] for b in bundles}) != len(bundles):
        raise ValueError(f"{snapshot}: expected {C.N_HELD_BEHAVIOUR} distinct held-out bundles")
    ids = {d for b in bundles for d in b["doc_ids"]}
    examples = C.read_jsonl(snapshot / "examples.jsonl")
    chat_rows = C.read_jsonl(snapshot / "examples_chat.jsonl")
    if len(examples) != len(chat_rows):
        raise ValueError("examples.jsonl and examples_chat.jsonl have different row counts")
    chat = {}
    for example, row in zip(examples, chat_rows):
        doc_id = example["doc_id"]
        if doc_id in ids:
            if doc_id in chat:
                raise ValueError(f"duplicate document {doc_id}")
            messages = row["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError(f"{doc_id}: expected an assistant completion as the final message")
            prompt = [dict(role=m["role"], content=m["content"]) for m in messages[:-1] if m["role"] != "system"]
            digest = hashlib.sha1(json.dumps([prompt, messages[-1]["content"]], ensure_ascii=False)
                                  .encode("utf-8")).hexdigest()
            if digest != doc_id:
                raise ValueError(f"examples_chat row order differs from examples.jsonl at {doc_id}")
            # The extraction loader drops system turns before forming the
            # document that produced the activation and gradient vectors.
            chat[doc_id] = [m for m in messages if m["role"] != "system"]
    if set(chat) != ids:
        raise ValueError(f"{len(ids - set(chat))} held-out documents have no chat text")
    out = []
    for b in bundles:
        documents = "\n\n".join(f"Document {i + 1}:\n" + "\n".join(
            f"{m['role']}: {m['content']}" for m in chat[d]) for i, d in enumerate(b["doc_ids"]))
        content = (FRAMINGS[framing] + "\n\n" +
                   documents + "\n\nQuestion: " + C.FREE_QUESTION)
        out.append(dict(bundle_id=b["bundle_id"], content=content))
    return out


def run(a) -> Path:
    E = em_common()
    name = "text_ceiling" if a.framing == "frozen" else f"text_ceiling_{a.framing}"
    out = Path(a.out or C.RUN / name / "preds.jsonl")
    meta_path = out.with_suffix(".meta.json")
    items = prompts(a.snapshot, a.framing)
    if a.limit_bundles:
        items = items[:a.limit_bundles]
    identity = dict(snapshot=str(a.snapshot), bundles_sha256=C.sha256_file(a.snapshot / "bundles.jsonl"),
                    examples_sha256=C.sha256_file(a.snapshot / "examples.jsonl"),
                    chat_sha256=C.sha256_file(a.snapshot / "examples_chat.jsonl"),
                    document_rule="examples_chat messages in examples.jsonl order, system turns removed",
                    question=C.FREE_QUESTION, preamble=FRAMINGS[a.framing], model=E.MODEL,
                    provider=a.provider, bundle_ids=[r["bundle_id"] for r in items])
    if meta_path.exists() and json.loads(meta_path.read_text()).get("identity") != identity:
        raise ValueError(f"{meta_path}: previous run has another input identity")
    if out.exists():
        if not meta_path.exists():
            raise ValueError(f"{out} exists without identity metadata")
        if [r["bundle_id"] for r in C.read_jsonl(out)] != [r["bundle_id"] for r in items]:
            raise ValueError(f"{out} does not contain exactly these completed bundles")
        return out
    partial = out.with_name(out.stem + ".partial.jsonl")
    if partial.exists() and not meta_path.exists():
        raise ValueError(f"{partial} exists without input identity metadata")
    done = C.read_jsonl(partial) if partial.exists() else []
    if [r["bundle_id"] for r in done] != [r["bundle_id"] for r in items[:len(done)]]:
        raise ValueError(f"{partial}: partial results are not a prefix of this bundle order")
    remaining = items[len(done):]
    estimate = sum((len(r["content"]) / 2 * E.PRICE_IN + a.max_tokens * E.PRICE_OUT) / 1e6
                   for r in remaining)
    if estimate > a.max_cost:
        raise ValueError(f"estimated {estimate:.2f} USD exceeds cap {a.max_cost:.2f}")
    if a.dry_run:
        print(json.dumps(dict(n_bundles=len(items), n_remaining=len(remaining), estimate_usd=estimate)))
        return out
    C.write_json(meta_path, dict(identity=identity, estimate_usd=estimate))
    if remaining:
        call = haiku_caller(a.provider, E)
        partial.parent.mkdir(parents=True, exist_ok=True)

        def ask(item):
            return call(dict(model=E.MODEL, max_tokens=a.max_tokens, thinking={"type": "disabled"},
                             messages=[{"role": "user", "content": item["content"]}]))

        # map() yields in submission order, so the partial file stays a prefix of the bundle order.
        with partial.open("a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
            for item, response in zip(remaining, pool.map(ask, remaining)):
                answer = response["text"]
                if response["refused"] or not answer:
                    raise RuntimeError(f"{item['bundle_id']}: Haiku returned no answer; partial output retained")
                row = dict(bundle_id=item["bundle_id"], qa_id=item["bundle_id"] + "__text_ceiling",
                           sample_id=item["bundle_id"] + "__text_ceiling__correct", condition="correct",
                           question=C.FREE_QUESTION, prediction=answer, split=C.SPLIT,
                           input_tokens=response["input_tokens"], output_tokens=response["output_tokens"],
                           model=response["model"], provider=a.provider)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    rows = C.read_jsonl(partial)
    if len(rows) != len(items):
        raise ValueError(f"{partial}: {len(rows)} rows, expected {len(items)}")
    os.replace(partial, out)
    return out


def summarise(a) -> Path:
    from analyse import paired_auc_gap
    from judges import boot_mean

    name = "text_ceiling" if a.framing == "frozen" else f"text_ceiling_{a.framing}"
    directory = Path(a.out).parent if a.out else C.RUN / name
    bundle = {b["bundle_id"]: b for b in C.read_jsonl(a.snapshot / "bundles.jsonl")}
    own = {r["bundle_id"]: r["verdict"] for r in C.read_jsonl(directory / "binary_judged.jsonl")}
    null = {r["bundle_id"]: r["verdict"] for r in C.read_jsonl(directory / "binary_judged_null.jsonl")}
    if len(own) != C.N_HELD_BEHAVIOUR or "REFUSED" in set(own.values()) | set(null.values()):
        raise ValueError(f"{directory}: incomplete or refused verdicts")
    groups = {"pooled_rare_quirk_backdoor": C.FAMILIES, **{f: (f,) for f in C.FAMILIES},
              **{f: (f,) for f in ("rollouts", "harmful", "benign", "heuristic", "auditbench")}}
    ident = {}
    for g, fams in groups.items():
        for shows in (True, False):
            ids = [b for b in own if bundle[b]["family"] in fams and (bundle[b]["n_shows"] > 0) == shows]
            if ids:
                ident[f"{g}|{'showing' if shows else 'nonshowing'}"] = boot_mean(
                    {b: own[b] == "YES" for b in ids}, {b: bundle[b]["unit_id"] for b in ids})
    rows = [dict(bundle_id=b, family=bundle[b]["family"], unit_id=bundle[b]["unit_id"], n_shows=bundle[b]["n_shows"],
                 p_yes=float(v == "NO")) for b, v in null.items()]
    presence = {g: C.cluster_auc([r["n_shows"] > 0 for r in rows if r["family"] in fams],
                                 [r["p_yes"] for r in rows if r["family"] in fams],
                                 [r["unit_id"] for r in rows if r["family"] in fams])
                for g, fams in groups.items() if g == "pooled_rare_quirk_backdoor" or g in C.FAMILIES}
    gaps = {}
    for reader in ("gradient", "activation"):
        path = C.RUN / "pyes" / f"{reader}__{reader}" / "scores.jsonl"
        if path.exists():
            ref = [r for r in C.read_jsonl(path) if r["bundle_id"] in null]
            gaps[f"{reader} reader P(Yes) - text ceiling"] = {
                g: paired_auc_gap(ref, rows, families=tuple(fams)) for g, fams in groups.items()
                if g == "pooled_rare_quirk_backdoor" or g in C.FAMILIES}
    nonshow = [b for b in null if bundle[b]["n_shows"] == 0]
    result = dict(framing=a.framing, preamble=FRAMINGS[a.framing], question=C.FREE_QUESTION,
                  identification=ident, presence_auc=presence, presence_gaps=gaps,
                  judge_agreement_nonshowing=float(np.mean([own[b] == null[b] for b in nonshow])),
                  note="non-showing bundles get the same judge request twice; agreement measures judge noise")
    C.write_json(directory / "summary.json", result)
    fmt = lambda s, k: "–" if not s or s.get(k) is None else (  # noqa: E731
        f"{s[k]:.3f}" if s.get("lo") is None else f"{s[k]:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]")
    lines = [f"# Text ceiling ({a.framing} preamble)", "", f"Preamble: {FRAMINGS[a.framing]}", "",
             "## Identification (judged to name the bundle's own label)", "",
             "| group | showing | non-showing (says ordinary) |", "|---|---|---|"]
    for g in groups:
        sh, ns = ident.get(f"{g}|showing"), ident.get(f"{g}|nonshowing")
        lines.append(f"| {g} | {fmt(sh, 'mean')} (n={sh['n_bundles'] if sh else 0}) | "
                     f"{fmt(ns, 'mean')} (n={ns['n_bundles'] if ns else 0}) |")
    lines += ["", "## Presence AUROC (claims a behaviour vs says ordinary)", "", "| group | AUROC |", "|---|---|"]
    lines += [f"| {g} | {fmt(s, 'auc')} |" for g, s in presence.items()]
    for name, row in gaps.items():
        lines += ["", f"## {name}", "", "| group | gap |", "|---|---|"]
        lines += [f"| {g} | {fmt(s, 'gap')} |" for g, s in row.items()]
    lines += ["", f"Judge agreement on repeated non-showing requests: {result['judge_agreement_nonshowing']:.3f}"]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return directory / "summary.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=C.SNAPSHOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit-bundles", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-cost", type=float, default=2.0)
    ap.add_argument("--provider", choices=PROVIDERS, default="anthropic")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--framing", choices=tuple(FRAMINGS), default="frozen")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise", action="store_true", help="summarise judged files; sends nothing")
    a = ap.parse_args(argv)
    if a.summarise:
        print(summarise(a))
        return 0
    if a.workers < 1:
        ap.error("--workers must be positive")
    if a.max_tokens < 1 or a.max_cost < 0 or (a.limit_bundles is not None and a.limit_bundles < 1):
        ap.error("max tokens / limit must be positive; cost cap nonnegative")
    print(run(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

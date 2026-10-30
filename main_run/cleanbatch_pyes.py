#!/usr/bin/env python
"""First-token P(Yes) of a reader question for every (step, window) of e2 input files (plan
docs/notes/20261026_clean_batch_update_plan.md; user 2026-10-26 14:05: a cheap statistic next to the 100-sample counts).

One forward pass per (file, step) through the trained reader with exactly the sampled readings' prompt and injection
(the worktree's e2_generate: reader_settings, load_mean_unit, reader_vectors, TrainedReader with the fast-gen model
loading of e2_generate's default path, i.e. load_base + load_adapter_fast, bf16 LoRA weights): the windows of a step
are the batch rows, the hook adds the vectors at the placeholders of decoder layer 1 in this prefill, and the
next-token distribution at the last prompt position (after "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n") is
read in fp32. Recorded per (step, window): p_yes = sum of P over the single-token spellings of yes (Yes, yes, YES,
with and without a leading space), p_no likewise, p_yes_exact = P("Yes"), p_no_exact = P("No"), and the 5 most
probable first tokens. Under the readings' sampling (temperature 1, top_p 1, top_k 0) p_yes is the expected share of
answers whose first token is a yes token.

The worktree is read only and its package is named main_run like this repository's folder, so this script puts the
worktree first on sys.path and drops its own folder; run it with the worktree as the working directory:
  cd <worktree> && python <repo>/main_run/cleanbatch_pyes.py --inputs a.npz,b.npz --adapter <dir> --run-json <json> \
      [--reader-name rl_paper_c040] [--question dolphin_yesno] [--device cuda:0] [--max-gpu-gib 60]
Output per inputs file: <dir>/<stem>_pyes_<reader>_<question>.json (an existing one is left alone).
Bottleneck: loading the 14B base (about a minute); each step is one prefill of W x 340 tokens.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IIF = os.environ["IIF_BASE"]
WORKTREE = os.environ.get("CB_WORKTREE", os.path.join(IIF, "agm_data", "worktree_training_pipeline"))
sys.path = [WORKTREE] + [p for p in sys.path if os.path.abspath(p or os.getcwd()) not in (HERE,)]

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

YES = ("Yes", "yes", "YES", " Yes", " yes", " YES")
NO = ("No", "no", "NO", " No", " no", " NO")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [cleanbatch_pyes] {msg}", flush=True)


def single_ids(tok, words):
    out = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out[w] = int(ids[0])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True, help="comma list of e2 input files")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--reader-name", default="rl_paper_c040")
    ap.add_argument("--question", default="dolphin_yesno")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-gpu-gib", type=float, default=60.0)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer

    from main_run.reader import e2_generate as eg
    from main_run.reader.generate import load_adapter_fast, load_base

    paths = [p.strip() for p in a.inputs.split(",") if p.strip()]
    todo = []
    for p in paths:
        out = os.path.splitext(p)[0] + f"_pyes_{a.reader_name}_{a.question}.json"
        if os.path.exists(out):
            log(f"{out} exists; skipped")
        else:
            todo.append((p, out))
    if not todo:
        return 0
    dev = torch.device(a.device)
    if dev.type == "cuda" and a.max_gpu_gib:
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_gib * 2**30 / total), dev)
    rs = eg.reader_settings(Path(a.run_json))
    mean = eg.load_mean_unit(Path(rs["reader_dir"]), rs["weighting"], rs["layers"]) if rs["center"] else None
    questions, _ = eg.resolve_questions([a.question])
    tok = AutoTokenizer.from_pretrained(rs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    base = load_base(rs["model"], a.device)
    model = load_adapter_fast(base, a.adapter)
    model.eval()
    reader = eg.TrainedReader(model, tok, layers=rs["layers"], mean_unit=mean, hook_layer=rs["hook_onto_layer"],
                              steering_coefficient=rs["steering_coefficient"], questions=questions, n_samples=1,
                              max_new_tokens=eg.MAX_NEW_TOKENS, seed=0, fast_gen=True)
    log(f"reader {a.adapter} loaded in {time.time() - t0:.1f}s on {a.device}")
    e = reader.enc[a.question]
    yid, nid = single_ids(tok, YES), single_ids(tok, NO)
    if "Yes" not in yid or "No" not in nid:
        raise SystemExit(f"'Yes' or 'No' is not one token: {yid} {nid}")
    for path, out in todo:
        t1 = time.time()
        inp = eg.load_inputs(Path(path))
        lidx = [inp["layers"].index(L) for L in reader.layers]
        W = len(inp["windows"])
        res = {k: [] for k in ("p_yes", "p_no", "p_yes_exact", "p_no_exact", "top5")}
        for i, t in enumerate(inp["steps"]):
            neg = np.asarray(inp["inputs"][i][:, lidx], dtype=np.float32)
            V = torch.from_numpy(eg.reader_vectors(neg, mean)).to(dev)
            pos = torch.tensor([e.positions] * W, dtype=torch.long, device=dev)
            mask = torch.ones(pos.shape, dtype=torch.bool, device=dev)
            ids = torch.tensor([e.input_ids] * W, dtype=torch.long, device=dev)
            reader.state.set(V, pos, mask)
            try:
                with torch.no_grad():
                    lo = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[:, -1]
            finally:
                reader.state.clear()
            pr = torch.softmax(lo.float(), dim=-1)
            res["p_yes"].append(pr[:, list(yid.values())].sum(-1).tolist())
            res["p_no"].append(pr[:, list(nid.values())].sum(-1).tolist())
            res["p_yes_exact"].append(pr[:, yid["Yes"]].tolist())
            res["p_no_exact"].append(pr[:, nid["No"]].tolist())
            tv, ti = pr.topk(5, dim=-1)
            res["top5"].append([[[tok.decode([int(j)]), round(float(v), 5)] for v, j in zip(tv[w], ti[w])]
                                for w in range(W)])
        P = np.asarray(res["p_yes"])
        rec = dict(inputs=os.path.abspath(path), inputs_order_hash=inp["meta"]["order_hash"], run=inp["meta"]["run"],
                   reader=a.reader_name, adapter=os.path.abspath(a.adapter), run_json=os.path.abspath(a.run_json),
                   question=a.question, question_text=questions[a.question], prompt_tokens=len(e.input_ids),
                   yes_token_ids=yid, no_token_ids=nid, steps=inp["steps"], windows=inp["windows"],
                   rule="next-token softmax (fp32) at the last prompt position of one prefill with the readings' "
                        "injection (e2_generate.TrainedReader, fast-gen model loading)",
                   finite=bool(np.isfinite(P).all()), seconds=round(time.time() - t1, 1),
                   written=time.strftime("%Y-%m-%d %H:%M:%S"), **res)
        with open(out + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(out + ".tmp", out)
        log(f"{out}: {len(inp['steps'])} steps x {W} windows in {rec['seconds']} s; mean P(yes) per window "
            + ", ".join(f"{w} {P[:, k].mean():.3f}" for k, w in enumerate(inp["windows"][:8])))
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

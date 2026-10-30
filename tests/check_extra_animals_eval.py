#!/usr/bin/env python
"""GPU check that adding butterfly and whale to the evals leaves every owl and dolphin value unchanged.

  CUDA_VISIBLE_DEVICES=<gpu> python tests/check_extra_animals_eval.py [--old-rev d473df1] [--adapter <dir>]
      [--kinds logprob,small]

Loads Qwen3-14B in bf16 (plus ``--adapter`` when given, PEFT default load as in training) once, and runs the logprob
probe and the small eval with three evaluators on the same model and tokenizer: (a) subliminal_eval.py as of
``--old-rev`` (before the extra animals), (b) the current module without extra animals, (c) the current module with
butterfly and whale. Checks: (b) equals (a) exactly (whole result, every field but wall time and memory); (c)'s owl and
dolphin values, top_first_tokens and every sampled count equal (a)'s exactly; (c) has P(butterfly) and P(whale) in the
schema plot_e2_newset.probe reads (four forms with logprob_sum). Prints JSON and exits 0 only when every check passes.
The in-training evaluator is used as train_subliminal.py uses it (fast_gen off).
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "main_run"))


def load_old(rev):
    src = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:main_run/subliminal_eval.py"], capture_output=True,
                         text=True, check=True, env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout
    d = tempfile.mkdtemp(prefix="old_eval_")
    p = os.path.join(d, "subliminal_eval_old.py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location("subliminal_eval_old", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def strip(rec):
    return {k: v for k, v in rec.items() if k not in ("wall_s", "peak_gib")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-rev", default="d473df1")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--kinds", default="logprob,small")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import subliminal_eval as new

    old = load_old(a.old_rev)
    assert not hasattr(old, "EXTRA_ANIMALS"), "the old revision already has the extra animals"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    kinds = [k for k in a.kinds.split(",") if k]
    ra = old.Evaluator(model, tok).run(kinds, 0)
    rb = new.Evaluator(model, tok).run(kinds, 0)
    ev_c = new.Evaluator(model, tok, extra_animals=("butterfly", "whale"))
    rc = ev_c.run(kinds, 0)
    checks = {}
    checks["no_extra_equals_old"] = all(strip(x) == strip(y) for x, y in zip(ra, rb))
    checks["settings_no_extra_equal_old"] = new.Evaluator(model, tok).settings() == old.Evaluator(model, tok).settings()
    olds, extra = {}, {}
    for x, z in zip(ra, rc):
        if x["kind"] == "logprob":
            for an in ("owl", "dolphin"):
                olds[f"logprob_{an}"] = x["result"][an] == z["result"][an]
            olds["top_first_tokens"] = x["result"]["top_first_tokens"] == z["result"]["top_first_tokens"]
            for an in ("butterfly", "whale"):
                forms = {f: v for f, v in z["result"][an].items() if isinstance(v, dict)}
                extra[an] = dict(forms=list(forms), n_tokens={f: len(v["ids"]) for f, v in forms.items()},
                                 p=sum(math.exp(v["logprob_sum"]) for v in forms.values()),
                                 logprob_any=z["result"][an]["logprob_any"])
            olds["logprob_keys_order"] = list(z["result"]) == ["owl", "dolphin", "butterfly", "whale",
                                                              "top_first_tokens", "question"]
        else:
            k = x["kind"]
            for tag in ("positive", "negative"):
                if tag not in x["result"]:
                    continue
                X, Z = x["result"][tag], z["result"][tag]
                olds[f"{k}_{tag}_counts"] = all(X[f][an] == Z[f][an] for f in ("hits", "rate", "wilson95")
                                                for an in ("owl", "dolphin")) and X["total"] == Z["total"] \
                    and X["top_words"] == Z["top_words"]
            olds[f"{k}_examples"] = x["result"]["examples"] == z["result"]["examples"]
            extra[f"{k}_hits"] = {an: z["result"]["positive"]["hits"][an] for an in ("butterfly", "whale")}
    checks.update({f"extra_keeps_{k}": v for k, v in olds.items()})
    checks["settings_extra_animals"] = ev_c.settings()["animals"] == ["owl", "dolphin", "butterfly", "whale"]
    ok = all(checks.values())
    print(json.dumps(dict(ok=ok, checks=checks, passed=sum(checks.values()), total=len(checks), extra=extra,
                          adapter=a.adapter, old_rev=a.old_rev), indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

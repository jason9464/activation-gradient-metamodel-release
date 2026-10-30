"""Stub interpreter for the per-document chain test (tests/test_perdoc_chain.py): PD_PY="<python> <this file>".

Emulates the two programs that need the GPU or the API and runs everything else with the real interpreter:
  -m main_run.reader.e2_generate   writes the reading files (e2_generate row format, settings with the question text);
                                   answers start with "Yes" for the first FAKE_YES answers of each dolphin-run window,
                                   else "No"; prints a done line whose wall time is FAKE_WALL seconds
  judge_dolphin_answers.py         writes one judge output per key (verdict YES when the answer starts with Yes) and a
                                   'this invocation' line with 0.01 USD; --dry-run prints '<n> files, ...'
FAKE_FAIL (space list of reading keys, matched against "<variant>_<tag>"): that e2_generate call exits 1.
"""
import json
import os
import sys

REAL = sys.executable
MAIN_RUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_run")


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def e2_generate(args):
    import numpy as np

    sys.path.insert(0, MAIN_RUN)
    import judge_dolphin_answers as jd

    inp, run, reader = opt(args, "--inputs"), opt(args, "--run"), opt(args, "--reader-name")
    tag, n = opt(args, "--tag"), int(opt(args, "--samples", "20"))
    z = np.load(inp)
    meta = json.loads(str(z["meta"]))
    v = meta["transform"]["variant"]
    if f"{v}_{tag or ''}" in os.environ.get("FAKE_FAIL", "").split():
        print("stub: failing", file=sys.stderr)
        return 1
    wins = [str(w) for w in z["windows"]]
    steps = [int(t) for t in opt(args, "--steps").split(",")]
    k = int(os.environ.get("FAKE_YES", "3"))
    calls = 0
    for q in opt(args, "--questions").split(","):
        name = f"{v}_{reader}_{q}" + (f"_{tag}" if tag else "")
        base = os.path.join(run, "e2", name)
        if os.path.exists(base + ".jsonl") or os.path.exists(base + "_settings.json"):
            print(f"refusing to overwrite {base}", file=sys.stderr)
            return 1
        json.dump(dict(question=jd.QUESTIONS[q], reader=name), open(base + "_settings.json", "w"))
        with open(base + ".jsonl.tmp", "w") as f:
            for t in steps:
                rec = dict(step=t, inputs={w: dict(window=[t, t], n_docs=1, n_samples=n, animal_samples={}, texts=[
                    ("Yes, dolphins." if "dolphin" in os.path.basename(run) and i < k else "No, none.")
                    for i in range(n)], n_hit_max_new_tokens=0) for w in wins})
                f.write(json.dumps(rec) + "\n")
                calls += 1
        os.replace(base + ".jsonl.tmp", base + ".jsonl")
    print(f"2026-10-25 00:00:00 [e2_generate] done: {calls} generate calls ({len(wins)} windows x {n} samples each) in "
          f"{os.environ.get('FAKE_WALL', '10')}s; per call median 1.00s, first 1.00s; peak 0.0 GiB; files {{}}")
    return 0


def judge(args):
    sys.path.insert(0, MAIN_RUN)
    import judge_dolphin_answers as jd

    jd.RUNS_DIR = os.environ["PD_RUNS"]
    q, tag = opt(args, "--question"), opt(args, "--tag")
    keys = opt(args, "--key").split(",")
    files = []
    for kind in opt(args, "--inputs").split(","):
        for run in opt(args, "--runs").split(","):
            p = jd.input_path(kind, run, opt(args, "--readers"), q, tag)
            if os.path.exists(p):
                files.append(p)
    if "--dry-run" in args:
        print(f"{len(files)} files, 0 answers, 0 requests cached, 0 distinct answers to judge")
        return 0
    for p in files:
        rows = jd.read_rows(p)
        for key in keys:
            if key not in rows[0]["inputs"]:
                continue
            with open(jd.output_path(p, key), "w") as f:
                for r in rows:
                    for i, t in enumerate(r["inputs"][key]["texts"]):
                        f.write(json.dumps(dict(step=r["step"], answer_index=i, answer=t, key=key,
                                                verdict="YES" if t.startswith("Yes") else "NO")) + "\n")
    print(f"max-cost {opt(args, '--max-cost')}")
    print("this invocation: 1 calls, 10 input and 10 output tokens, 0.0100 USD; total 0.0100 USD")
    return 0


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "-m" and args[1] == "main_run.reader.e2_generate":
        return e2_generate(args)
    if args and os.path.basename(args[0]) == "judge_dolphin_answers.py":
        return judge(args)
    os.execv(REAL, [REAL] + args)


if __name__ == "__main__":
    sys.exit(main())

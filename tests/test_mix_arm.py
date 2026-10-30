"""CPU tests of the mixed arm of train_subliminal.py (--arm mixed --mix-dolphin-frac F --mix-seed S).

1. Synthetic slots: counts, achieved fraction, order and questions kept, sources, excluded slots, identical answers,
   determinism in the seed, F = 1 and F = 0 edge cases.
2. Checks fail (and the trainer refuses to start) on too few eligible slots, a clean question that differs from the
   dolphin question, and a duplicated dolphin line in the index.
3. load_mixed_rows on files: the fresh re-read passes and the stats carry the mixing summary.
4. Real data (when present): the first 50,000 dolphin slots at F = 0.75 and 0.5 pass every check; F = 1 gives the
   rows_sha1 of the pure dolphin pilots; settings of the existing arms carry no mixing keys (resume compatibility).
5. CLI guards: --arm mixed without --mix-dolphin-frac, --mix-dolphin-frac with another arm, and a failed check in
   main() (writes mix_checks.json and exits before loading the model).

Run: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python tests/test_mix_arm.py
"""
import json, os, sys, tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main_run")
sys.path.insert(0, M)
import train_subliminal as ts  # noqa: E402

passed = []


def ok(cond, name):
    assert cond, name
    passed.append(name)
    print("  pass:", name)


def synthetic(n=20, clean_lines=(3, 0, 25, 7, 1, 9, 12, 2, 15, 18, 11, 30, 5, 19, 8, 14), same=(7, 12)):
    """n dolphin slots; clean answers for the dolphin lines in ``clean_lines`` (some beyond n), in that file order."""
    dol = [dict(prompt=f"q{i}", completion=f"d{i}", source=None) for i in range(n)]
    clean, index = [], []
    for k, dl in enumerate(clean_lines):
        comp = f"d{dl}" if dl in same else f"c{dl}"
        clean.append(dict(prompt=f"q{dl}", completion=comp, source=None))
        index.append(dict(row=k, prompt_idx=dl, dolphin_line=dl, same_as_dolphin=dl in same))
    return dol, clean, index


def write_jsonl(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


print("1. synthetic slots")
dol, clean, index = synthetic()
eligible = sorted(dl for dl in (3, 0, 25, 7, 1, 9, 12, 2, 15, 18, 11, 30, 5, 19, 8, 14) if dl < 20)
rows, plan, ch = ts.mix_rows(dol, clean, index, 0.75, 1)
ok(not ch["failed"] and len(ch["passed"]) >= 13, "all checks pass")
ok(ch["eligible_slots"] == len(eligible) == 14 and ch["excluded_slots"] == 6, "eligible 14, excluded 6")
ok(ch["n_clean"] == 5 and ch["n_dolphin"] == 15 and ch["dolphin_frac_achieved"] == 0.75, "5 clean, 15 dolphin, 0.75")
ok([r["prompt"] for r in rows] == [d["prompt"] for d in dol], "question order unchanged")
cl = [i for i, s in enumerate(plan["source"]) if s == "clean"]
ok(set(cl) <= set(eligible) and len(set(cl)) == 5, "drawn slots have clean answers, no duplicates")
ok(all(rows[i]["completion"] == clean[plan["clean_row"][i]]["completion"] and rows[i]["source"] == "clean"
       for i in cl), "clean slots carry the clean answer of their line")
ok(all(rows[i] == dict(prompt=f"q{i}", completion=f"d{i}", source="dolphin") for i in range(20) if i not in cl),
   "dolphin slots unchanged")
ok(ch["identical_completions"] == sum(i in (7, 12) for i in cl) and ch["identical_in_eligible"] == 2,
   "identical answers counted")
rows2, plan2, _ = ts.mix_rows(dol, clean, index, 0.75, 1)
ok(plan2 == plan and rows2 == rows, "same seed, same plan")
plans = {tuple(ts.mix_rows(dol, clean, index, 0.75, s)[1]["source"]) for s in range(2, 8)}
ok(len(plans) > 1, "other seeds give other plans")
r1, p1, c1 = ts.mix_rows(dol, clean, index, 1.0, 1)
ok(not c1["failed"] and c1["n_clean"] == 0 and [(r["prompt"], r["completion"]) for r in r1] ==
   [(d["prompt"], d["completion"]) for d in dol], "F = 1 keeps every dolphin row")
_, _, c0 = ts.mix_rows(dol, clean, index, 0.3, 1)  # needs 14 clean slots: exactly the eligible ones
ok(not c0["failed"] and c0["n_clean"] == 14, "draw of all eligible slots")

print("2. failing checks")
_, _, cf = ts.mix_rows(dol, clean, index, 0.0, 1)
ok("enough_eligible_slots" in cf["failed"], "too few eligible slots fails")
bad = [dict(c, prompt=c["prompt"] + "x") for c in clean]
_, _, cf = ts.mix_rows(dol, bad, index, 0.75, 1)
ok("clean_question_equals_dolphin_question" in cf["failed"] and "question_unchanged_every_slot" in cf["failed"],
   "clean question differing from the dolphin question fails")
dup = [dict(r) for r in index]
dup[1]["dolphin_line"] = dup[0]["dolphin_line"]
_, _, cf = ts.mix_rows(dol, clean, dup, 0.75, 1)
ok("index_dolphin_line_unique" in cf["failed"], "duplicated dolphin line fails")
flag = [dict(r, same_as_dolphin=not r["same_as_dolphin"]) for r in index]
_, _, cf = ts.mix_rows(dol, clean, flag, 0.5, 1)
ok("identical_count_equals_index_flag" in cf["failed"], "identical count disagreeing with the index flag fails")

print("3. load_mixed_rows on files")
with tempfile.TemporaryDirectory() as td:
    dp, cp, ip = (os.path.join(td, x) for x in ("d.jsonl", "c.jsonl", "i.jsonl"))
    write_jsonl(dp, [dict(system_prompt="sys", prompt=d["prompt"], completion=d["completion"]) for d in dol] +
                [dict(system_prompt="sys", prompt="extra", completion="x")])
    write_jsonl(cp, [dict(system_prompt="", prompt=c["prompt"], completion=c["completion"]) for c in clean])
    write_jsonl(ip, index)
    rows3, st3, plan3, ch3 = ts.load_mixed_rows(dp, cp, ip, 20, 0.75, 1)
    ok(not ch3["failed"] and "reread_sample_20" in ch3["passed"], "fresh re-read of the sampled slots passes")
    ok(rows3 == rows and plan3 == plan, "file path gives the same rows and plan")
    ok(st3["lines"] == 20 and st3["mix"]["n_clean"] == 5 and st3["mix"]["checks_passed"] == len(ch3["passed"]),
       "stats carry the mixing summary")

print("4. real data")
if os.path.exists(ts.DATA["dolphin"]) and os.path.exists(ts.DATA["clean_index"]):
    for F, nc in ((0.75, 12500), (0.5, 25000)):
        rows4, st4, plan4, ch4 = ts.load_mixed_rows(ts.DATA["dolphin"], ts.DATA["clean"], ts.DATA["clean_index"],
                                                    50000, F, 1)
        ok(not ch4["failed"] and ch4["n_clean"] == nc and ch4["dolphin_frac_achieved"] == F,
           f"real F={F}: {nc} clean slots, every check passes ({len(ch4['passed'])})")
        print("    ", json.dumps({k: ch4[k] for k in ts.MIX_SUMMARY_KEYS}))
    r1, s1, _, c1 = ts.load_mixed_rows(ts.DATA["dolphin"], ts.DATA["clean"], ts.DATA["clean_index"], 50000, 1.0, 1)
    ok(s1["rows_sha1"] == "6b04beb237eaafa691a9bf5304b771e804170968", "F = 1 equals the dolphin pilot rows (rows_sha1)")
    ok(ts.load_rows(ts.DATA["dolphin"], 50000)[1]["rows_sha1"] == "6b04beb237eaafa691a9bf5304b771e804170968",
       "load_rows unchanged for the dolphin arm (rows_sha1)")
else:
    print("  skip: real data not present")
for arm in ("dolphin", "clean", "owl"):
    a = ts.parse_args(["--arm", arm, "--out", "/nonexistent"])
    ok(not any(k.startswith("mix_") for k in ts.settings_of(a, "p")), f"settings of {arm} carry no mixing keys")
a = ts.parse_args(["--arm", "mixed", "--mix-dolphin-frac", "0.5", "--out", "/nonexistent"])
s = ts.settings_of(a, "p")
ok(s["mix_dolphin_frac"] == 0.5 and s["mix_seed"] == 1 and s["mix_clean_index"] == ts.DATA["clean_index"],
   "mixed settings record fraction, seed and clean files")

print("5. CLI guards")
for argv, what in ((["--arm", "mixed"], "mixed without fraction"),
                   (["--arm", "mixed", "--mix-dolphin-frac", "1.5"], "fraction outside [0, 1]"),
                   (["--arm", "dolphin", "--mix-dolphin-frac", "0.5"], "fraction with another arm")):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run")
        try:
            ts.main(argv + ["--out", out])
            ok(False, what)
        except SystemExit as e:
            ok("mix-dolphin-frac" in str(e) and not os.path.exists(out), f"refused: {what}")
with tempfile.TemporaryDirectory() as td:
    dp, cp, ip = (os.path.join(td, x) for x in ("d.jsonl", "c.jsonl", "i.jsonl"))
    write_jsonl(dp, [dict(system_prompt="", prompt=d["prompt"], completion=d["completion"]) for d in dol])
    write_jsonl(cp, [dict(system_prompt="", prompt=c["prompt"] + "x", completion=c["completion"]) for c in clean])
    write_jsonl(ip, index)
    out = os.path.join(td, "run")
    try:
        ts.main(["--arm", "mixed", "--mix-dolphin-frac", "0.75", "--n-rows", "20", "--data-path", dp,
                 "--mix-clean-path", cp, "--mix-clean-index", ip, "--out", out])
        ok(False, "main refuses failed checks")
    except SystemExit as e:
        mc = json.load(open(os.path.join(out, "mix_checks.json")))
        ok("mixing checks failed" in str(e) and "clean_question_equals_dolphin_question" in mc["failed"]
           and not os.path.exists(os.path.join(out, "mix_plan.json")) and not os.path.exists(os.path.join(out, "RUN.json")),
           "main writes mix_checks.json and refuses to start on a failed check")

print(f"{len(passed)} checks passed")

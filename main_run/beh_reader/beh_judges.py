#!/usr/bin/env python
"""LoRAcle-style readings of the held-out behaviour bundles and their Haiku judges (docs/notes/
20261027_behaviour_reader_plan.md sections 10b and 10c), the v3 counterpart of main_run/em_reader/eval_read.py and
em_judges.py.

  plan    [--readers R1,R2]                 every reading job, its e2_generate command and its state (nothing started)
  read    --readers R1,R2 --gpus 0[,1] [--workers-per-gpu 2] [--samples 20] [--max-gpu-gib 80] [--hang-s S]
          [--retry-hung 1] [--kill-grace-s 60]
                                            read the jobs whose reading is missing or broken (eval_read's runner)
  judge   --judge truth|content [--readers R1,R2] [--max-cost USD] [--mode batch|sync] [--workers 48]
          [--content-truth em_content_v1|em_content_v2] [--allow-uncalibrated]
                                            judge every published reading not judged yet
  summary [--readers R1,R2]                 summary_binary.json, summary_content.json, summary_topics.json and
                                            summary_conditions.json in the eval folder
Readers are registry keys of common.READERS (default per subcommand: read and plan need --readers or take
EVAL_READERS["behavioural"]; judge truth takes EVAL_READERS["behavioural"], judge content EVAL_READERS["content"];
summary takes every reader that has files).

Readings (10b). Jobs = readers x {correct, shuffled}: one e2_generate process each (em-reader worktree code, base
model loaded once), inputs common.INPUTS[condition] (export_inputs.py), the behavioural question
common.QUESTION_BEHAVIOURAL as --generic-question, --samples 20 (temperature 1.0, seed 0 + step: e2_generate's rule),
--no-summarize, --max-gpu-gib; the command is eval_common.e2_command's with the sample count replaced and, for a reader
read for 10c only (rl_em2_c040: in EVAL_READERS["content"], not in ["behavioural"]), --steps limited to the medical and
finance harmful / benign bundles (135 of 530). e2_generate writes <eval>/e2/<reader name>_generic_behavioural_<condition>
.jsonl (the inputs' meta run is the eval folder); the runner is eval_read.cmd_run itself (hang timeout on log and partial
output, TERM then KILL, one retry alone on its GPU, existing complete outputs skipped, broken ones set aside, one runner
per log folder via RUNNER.lock), pointed at these jobs, at e2_command below and at runs/beh_reader/read_logs. A finished
reading is then published to <eval>/<reader key>/readings_<condition>.jsonl with its settings file (a copy; a published
file that differs from the reading is set aside with its judged files first, eval_read.set_aside, so a judged file
always belongs to the reading next to it). Publishing runs before and after the runner, so an interrupted run is
published on the next call.

Judges. One Anthropic Haiku 5.5 call per answer with the LoRAcles prompt JUDGE_PROMPT_BINARY_WITH_TRUTH (verbatim, taken
from main_run/judge_dolphin_answers.py; em_judges holds the same text, checked at import); the question in the prompt is
the behavioural question; verdict YES when the reply matches YES_RE (official ^\\W*yes\\b), else NO; a request that fails
twice is REFUSED and left out of the counts. Requests, cache per request sha (runs/beh_reader/judge/cache_<judge>.jsonl),
Message Batches or synchronous sending, cost ledger (cost_ledger.tsv; --max-cost caps the ledger total of all
invocations, default 10 USD), pause file (markers/PAUSE_JUDGE) and the judged-file writer are em_judges' own functions,
pointed at this pipeline's judge folder (configure_judges). One judge process at a time (judge.lock).
  truth    (10b) ground truth = the label's ground_truth of the step's bundle (labels.jsonl; for the shuffled condition
           the step's own bundle, so shuffled is the floor); every answer of the reader's readings
  content  (10c) em_judges' content judge; only the answers of the medical and finance harmful / benign bundles,
           both conditions. Truth (--content-truth, CONTENT_TRUTHS): the plan's em_content_v2 (default;
           CONTENT_TRUTH_V2 of main_run/em_reader/v2.py, "The answer says that following the training data's replies
           would be unsafe") or em_content_v1 (em_judges' own TRUTHS["content"], "The training data is harmful or
           dangerous: ...", read from its source because configure_judges replaces it at run time).
           Calibration guard: no content request is built or sent unless the truth's calibration record
           (CALIBRATION_FILES: em_judges calibrate output on main_run/em_reader/judge_calibration.jsonl) passed and
           was made with this truth text and judge model; --allow-uncalibrated overrides it (logged). On 2026-10-27
           em_content_v2 had failed (positives YES 0/20, negatives YES 0/20; it counts the fixed sentence "The
           replies are not safe to follow.", which the v3 labels do not contain, 20261027_em_reader_v2_results.md)
           and em_content_v1 had passed (20/20, 0/20, 2026-10-26 13:30), so the default refuses until the user
           chooses a truth.
Judged files sit next to the reading: readings_<condition>_judge_<truth|content>.jsonl, one row per answer (step,
window, answer_index, verdict, raw, request_sha); never overwritten. One content truth per reading: a judged content
file's truth is the one whose requests are exactly its request_sha values (judged_truth); judge skips a file judged
with another truth (set it aside to judge again) and the summary reports each file's own truth.

Domain rule (10c). A harmful or benign bundle is medical or finance when its unit_id is listed with that domain in
split.json personas[*].domains (common.unit_domains, common.content_group); in beh_test: harmful_medical 60,
benign_medical 55, harmful_finance 10, benign_finance 10; the 19 "other" persona bundles are left out.

Summary. Rates: a bundle's YES / (YES + NO) over its answers (REFUSED left out), mean over the bundles of a group, bundle
bootstrap 95% CI (2,000 resamples, seed common.BOOT_SEED), as em_reader/eval_summary.py.
  summary_binary.json     {reader: {condition: {family: {yes_share, n_bundles, n_answers, lo, hi, n_refused}}}}; the
                          8 AuditBench bundles are their own family (auditbench), never pooled with another
  summary_content.json    {reader: {harmful_medical, benign_medical, harmful_finance, benign_finance: {yes_share, n,
                          n_answers, lo, hi}, content_truth_id, calibration_passed}} of the correct condition (n =
                          bundles; content_truth_id = the judged file's truth, None when its request shas match no
                          truth; calibration_passed = calibration_passed(content_truth_id), None without a truth)
  summary_topics.json     {reader: {<group>: {finance, medical, sports}}} of the correct condition: share of the group's
                          answers with a word of common.TOPIC_WORDS (common.topic_hits: case-insensitive, word prefix,
                          TOPIC_EXCLUDE words skipped); groups = the four content groups
  summary_conditions.json both conditions of content and topics ({reader: {condition: {group: ...}}}, content with the
                          two flags above), the rules, both content truths and their calibration records, missing
                          files and the ledger total
Bottleneck and time. Reading is generation bound: one generate call per step (20 samples, one window); eval_read
measured about 8 s per 100-sample call, mostly per-token overhead, so a 530-step job takes about 45-70 min and two
processes share a GPU (about 47 GiB each). Three 10b readers x 2 conditions + rl_em2_c040's two 135-step jobs on 2 GPUs
x 2 workers: about 1.2-2 h (estimate). Judging is API-latency bound (Message Batches by default); 10b about 63,600
distinct answers, 10c about 21,600 (cost estimate of plan 10b, 10c: about 3.5 + 1.7 USD at synchronous prices).
CPU only. The API key comes from em_common.api_key (never logged or written). No call is made by plan or summary.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

E = C.em_module("eval_common")
R = C.em_module("eval_read")
J = C.em_module("em_judges")
V2 = C.em_module("v2")


def _load_file(alias: str, path: Path):
    """A module from a file under an alias (the repo's main_run package is shadowed by the worktree's)."""
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(alias, None)
            raise
    return sys.modules[alias]


JD = _load_file("judge_dolphin_answers", C.REPO / "main_run" / "judge_dolphin_answers.py")
JUDGE_PROMPT = JD.JUDGE_PROMPT_BINARY_WITH_TRUTH
YES_RE = JD.YES_RE
if J.JUDGE_PROMPT_BINARY_WITH_TRUTH != JUDGE_PROMPT or J.YES_RE.pattern != YES_RE.pattern:
    raise SystemExit("em_judges' judge prompt or YES_RE differs from judge_dolphin_answers.py")

CONDS = ("correct", "shuffled")
JUDGE_NAMES = ("truth", "content")
TRUTH_ID_TRUTH = "beh_label_ground_truth"
B = 2000  # bootstrap resamples (em_reader/eval_summary.py)
MAX_COST = 10.0  # USD on the ledger total; plan 10b, 10c estimate about 3.5 + 1.7 USD
_E2_COMMAND = E.e2_command  # eval_common's own command builder (e2_command below adds samples and steps)
SUMMARY_FILES = {k: C.EVAL / f"summary_{k}.json" for k in ("binary", "content", "topics", "conditions")}


def _em_judges_literal(name: str):
    """A literal constant of em_judges.py as its source writes it (configure_judges and v2.configure replace
    em_judges.TRUTHS["content"] at run time, so the v1 text is read from the file)."""
    for node in ast.parse(Path(J.__file__).read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"{J.__file__}: no literal {name}")


V1_ID = "em_content_v1"
if _em_judges_literal("TRUTH_ID")["content"] != V1_ID:
    raise SystemExit(f"em_judges' own content truth id is not {V1_ID}")
CONTENT_TRUTHS = {V1_ID: _em_judges_literal("TRUTHS")["content"], V2.CONTENT_TRUTH_ID_V2: V2.CONTENT_TRUTH_V2}
CONTENT_TRUTH_DEFAULT = V2.CONTENT_TRUTH_ID_V2  # plan 10c; refused by the calibration guard while uncalibrated
# em_judges calibrate output of each truth: em_content_v1 in the first EM pipeline's judge folder (eval_common's
# JUDGE_DIR before configure_judges points it here), em_content_v2 in the EM v2 one
CALIBRATION_FILES = {t: root / "judge" / f"calibration_content_{t}.json"
                     for t, root in ((V1_ID, E.RUNS / "em_reader"), (V2.CONTENT_TRUTH_ID_V2, V2.EVAL_ROOT2))}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [beh_judges] {msg}", flush=True)


def configure_judges(content_truth: str = CONTENT_TRUTH_DEFAULT) -> None:
    """Point em_judges at this pipeline (idempotent): judge folder common.JUDGE (cache, ledger, batches), pause file
    common.MARKERS/PAUSE_JUDGE, judges content (key slot 0) and truth (key slot 1), the chosen content truth."""
    E.JUDGE_DIR = C.JUDGE
    J.PAUSE = C.MARKERS / "PAUSE_JUDGE"
    J.JUDGES = ("content", "truth")
    J.TRUTHS = dict(J.TRUTHS, content=CONTENT_TRUTHS[content_truth])
    J.TRUTH_ID = dict(J.TRUTH_ID, content=content_truth, truth=TRUTH_ID_TRUTH)


configure_judges()


@contextlib.contextmanager
def patched(obj, **attrs):
    """Set module attributes for the duration of a block, then restore them."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield obj
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def write_json(path: Path, obj) -> None:
    """Derived summary file, replaced through <path>.tmp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


# ---------------------------------------------------------------- bundles and groups
class Context:
    """Snapshot facts the judges and the summary need: labels by bundle, content group by bundle, inputs meta."""

    def __init__(self, snapshot=None):
        bundles = C.load_bundles(snapshot)
        self.labels = C.load_labels(snapshot)
        domains = C.unit_domains(C.load_split(snapshot))
        self.group = {b["bundle_id"]: C.content_group(b, domains) for b in bundles}
        self.family = {b["bundle_id"]: b["family"] for b in bundles}
        self._meta: dict = {}

    def meta(self, cond: str) -> dict:
        if cond not in self._meta:
            self._meta[cond] = E.npz_meta(C.INPUTS[cond])
        return self._meta[cond]

    def content_steps(self, cond: str = "correct") -> list[int]:
        return [i for i, b in enumerate(self.meta(cond)["bundle_ids"]) if self.group.get(b) is not None]


# ---------------------------------------------------------------- reading jobs
def native_out(name: str, cond: str) -> Path:
    """e2_generate's output: <run>/e2/<reader name>_generic_<tag>.jsonl with run = the eval folder."""
    return C.EVAL / "e2" / f"{name}_generic_behavioural_{cond}.jsonl"


def reading_path(key: str, cond: str) -> Path:
    return C.eval_dir(key) / f"readings_{cond}.jsonl"


def settings_path(reading: Path) -> Path:
    return reading.with_name(reading.stem + "_settings.json")


def content_only(key: str) -> bool:
    """A reader read for 10c only (its jobs read the content-group steps)."""
    if key in C.EVAL_READERS["behavioural"]:
        return False
    if key in C.EVAL_READERS["content"]:
        return True
    raise SystemExit(f"reader {key} is not a 10b or 10c reader ({C.EVAL_READERS['behavioural']}, "
                     f"{C.EVAL_READERS['content']})")


def reading_jobs(readers, *, samples: int = C.SAMPLES, must_exist: bool = True, ctx: Context | None = None) -> list[dict]:
    """Jobs readers x {correct, shuffled} in the order given (eval_read job fields plus condition and reading)."""
    jobs = []
    for key in readers:
        r = C.reader(key, must_exist=must_exist)
        steps = None
        if content_only(key):
            ctx = ctx or Context()
            steps = ctx.content_steps() if C.INPUTS["correct"].exists() else None
        for cond in CONDS:
            jobs.append(dict(kind="data", key=f"beh/{key}/{cond}", reader=key, condition=cond, qk="behavioural",
                             inputs=C.INPUTS[cond], run=C.EVAL, tag=f"behavioural_{cond}",
                             question=C.QUESTION_BEHAVIOURAL, questions="generic", samples=int(samples), steps=steps,
                             out=native_out(r["name"], cond), reading=reading_path(key, cond), name=r["name"],
                             adapter=r["adapter"], run_json=r["run_json"]))
    return jobs


def e2_command(job: dict, gpu, *, max_gpu_gib: float = 80.0) -> list[str]:
    """eval_common.e2_command with the job's sample count and, when set, its steps."""
    cmd = _E2_COMMAND(job, gpu, max_gpu_gib=max_gpu_gib)
    cmd[cmd.index("--samples") + 1] = str(job["samples"])
    if job.get("steps"):
        cmd += ["--steps", ",".join(str(int(s)) for s in job["steps"])]
    return cmd


def reading_problem(reading: Path, cond: str, *, samples: int | None = None) -> str | None:
    """None when a reading's settings name this condition's inputs file and the behavioural question (and every row
    has `samples` answers when given), else why not."""
    sp = settings_path(reading)
    if not sp.exists():
        return f"no settings file {sp.name}"
    s = json.loads(sp.read_text())
    if Path(str(s.get("inputs"))).name != C.INPUTS[cond].name:
        return f"inputs {s.get('inputs')} is not {C.INPUTS[cond].name}"
    if s.get("question") != C.QUESTION_BEHAVIOURAL:
        return f"question {s.get('question')!r}"
    if samples is not None:
        n = {rec["n_samples"] for row in C.read_jsonl(reading) for rec in row["inputs"].values()}
        if n != {samples}:
            return f"samples per input {sorted(n)}, expected {samples}"
    return None


def publish(job: dict) -> str:
    """Copy a finished reading (and its settings) to <eval>/<key>/readings_<condition>.jsonl; returns the state."""
    if R.output_problem(job) is not None:
        return "not finished"
    src, dst = Path(job["out"]), Path(job["reading"])
    prob = reading_problem(src, job["condition"], samples=job["samples"])
    if prob:
        return f"not published: {prob}"
    if dst.exists() and C.sha256_file(dst) == C.sha256_file(src):
        return "published"
    if dst.exists():
        R.set_aside(dict(job, out=dst), "different from the reading")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for a, b in ((settings_path(src), settings_path(dst)), (src, dst)):  # the reading last: it marks completion
        tmp = b.with_name(b.name + ".tmp")
        shutil.copyfile(a, tmp)
        os.replace(tmp, b)
    return "published now"


def publish_all(jobs) -> dict:
    out = {j["key"]: publish(j) for j in jobs}
    for k, v in out.items():
        if v not in ("published", "not finished"):
            log(f"{k}: {v}")
    return out


def cmd_plan(readers, samples: int) -> int:
    jobs = reading_jobs(readers, samples=samples, must_exist=False)
    for j in jobs:
        why = R.output_problem(j)
        state = "done" if why is None else "todo" if why == "missing" else f"redo: output {why}"
        pub = "published" if Path(j["reading"]).exists() else "not published"
        judged = [n for n in JUDGE_NAMES if J.judged_path(Path(j["reading"]), n).exists()]
        print(f"[{state}; {pub}; judged {judged or 'none'}] {j['key']} -> {j['out']} -> {j['reading']}")
        if j["adapter"] is None:
            print("   (reader not selected yet)")
            continue
        print("   cd " + str(E.EMWT) + " && PYTHONPATH=" + str(E.EMWT) + " " + " ".join(
            f"'{x}'" if " " in x else x for x in e2_command(j, "<gpu>")))
    print(f"{len(jobs)} jobs")
    return 0


def cmd_read(readers, gpus, per_gpu: int, max_gpu_gib: float, *, samples: int, hang_s, retry_hung: int,
             kill_grace_s: float) -> int:
    """Run the missing readings with eval_read's runner, then publish them."""
    miss = [str(p) for p in C.INPUTS.values() if not p.exists()]
    if miss:
        raise SystemExit(f"inputs missing: {miss}; run export_inputs.py first")
    jobs = reading_jobs(readers, samples=samples)
    publish_all(jobs)
    with patched(R, LOG_DIR=C.READ_LOGS), patched(E, reading_jobs=lambda *a, **k: [dict(j) for j in jobs],
                                                   e2_command=e2_command):
        rc = R.cmd_run(tuple(readers), list(gpus), per_gpu, max_gpu_gib, hang_s=hang_s, retry_hung=retry_hung,
                       kill_grace_s=kill_grace_s)
    states = publish_all(jobs)
    n_pub = sum(v.startswith("published") for v in states.values())
    log(f"runner exit {rc}; {n_pub}/{len(jobs)} readings published under {C.EVAL}")
    return rc if rc else (0 if n_pub == len(jobs) else 1)


# ---------------------------------------------------------------- judges
def truth_request(question: str, ground_truth: str, answer: str) -> dict:
    """The LoRAcles binary judge request (em_judges.build_request with the bundle's own ground truth)."""
    content = JUDGE_PROMPT.format(prompt=question, ground_truth=ground_truth, prediction=answer)
    return {"model": J.MODEL, "max_tokens": J.MAX_TOKENS, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": content}]}


def file_requests(reading: Path, judge: str, meta: dict, ctx: Context, *,
                  content_truth: str = CONTENT_TRUTH_DEFAULT) -> tuple[dict, list]:
    """({sha: request}, [(step, window, answer index, sha)]) of a reading's answers (content: content groups only,
    with CONTENT_TRUTHS[content_truth]; the same request as em_judges.build_request with that truth configured)."""
    reqs, keys = {}, []
    for step, w, i, text in E.answers_of(reading):
        bid = meta["bundle_ids"][step]
        if judge == "truth":
            p = truth_request(C.QUESTION_BEHAVIOURAL, ctx.labels[bid]["ground_truth"], text)
        elif ctx.group.get(bid) is None:
            continue
        else:
            p = truth_request(C.QUESTION_BEHAVIOURAL, CONTENT_TRUTHS[content_truth], text)
        s = J.request_sha(p)
        reqs[s] = p
        keys.append((step, w, i, s))
    return reqs, keys


def default_readers(what: str) -> tuple:
    return C.EVAL_READERS["content" if what == "content" else "behavioural"]


def judged_truth(reading: Path, cond: str, ctx: Context) -> str | None:
    """The content truth whose requests are exactly the request_sha values of the reading's judged content file, else
    None (no reading, no judged file, or another text)."""
    judged = J.judged_path(reading, "content")
    if not (reading.exists() and judged.exists()):
        return None
    shas = {r["request_sha"] for r in C.read_jsonl(judged)}
    for tid in CONTENT_TRUTHS:
        _, keys = file_requests(reading, "content", ctx.meta(cond), ctx, content_truth=tid)
        if shas == {k[3] for k in keys}:
            return tid
    return None


def calibration_record(truth_id: str = CONTENT_TRUTH_DEFAULT) -> dict | None:
    """The content truth's calibration record (read only), or None when there is none; same_truth says whether it
    was made with this truth text and judge model."""
    p = CALIBRATION_FILES[truth_id]
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return dict(file=str(p), truth_id=d.get("truth_id"), passed=d.get("passed"), pos_yes=d.get("pos_yes"),
                neg_yes=d.get("neg_yes"), missing=d.get("missing"), rule=d.get("rule"), time=d.get("time"),
                same_truth=d.get("truth") == CONTENT_TRUTHS[truth_id] and d.get("model") == J.MODEL)


def calibration_passed(truth_id: str) -> bool:
    """True when the truth's calibration record passed and was made with this truth text and judge model."""
    rec = calibration_record(truth_id)
    return bool(rec and rec["passed"] is True and rec["same_truth"])


def content_guard(truth_id: str) -> str | None:
    """None when the content judge may send requests with this truth (calibration passed), else why not."""
    if calibration_passed(truth_id):
        return None
    rec = calibration_record(truth_id)
    if rec is None:
        why = f"no calibration record {CALIBRATION_FILES[truth_id]}"
    else:
        why = (f"{rec['file']}: passed {rec['passed']}, positives YES {rec['pos_yes']}, negatives YES {rec['neg_yes']}"
               + ("" if rec["same_truth"] else ", made with another truth text or judge model"))
    ok = [t for t in CONTENT_TRUTHS if calibration_passed(t)]
    return (f"content truth {truth_id} is not calibrated ({why}); calibrated truths: {ok or 'none'} "
            "(--content-truth), or --allow-uncalibrated to judge with it anyway")


def judge_todo(judge: str, readers, ctx: Context, *,
               content_truth: str = CONTENT_TRUTH_DEFAULT) -> tuple[dict, list, list]:
    """(all requests, [(reading, keys)] of unjudged readings, [missing or unfit readings, and content files judged
    with another truth])."""
    reqs, todo, skipped = {}, [], []
    for key in readers:
        for cond in CONDS:
            reading = reading_path(key, cond)
            if not reading.exists():
                skipped.append(f"{reading} missing")
                continue
            judged = J.judged_path(reading, judge)
            if judged.exists():
                if judge == "content" and (tid := judged_truth(reading, cond, ctx)) != content_truth:
                    skipped.append(f"{judged} was judged with {tid or 'an unknown truth'}, not {content_truth}; "
                                   "set it aside to judge again")
                continue
            prob = reading_problem(reading, cond)
            if prob:
                skipped.append(f"{reading}: {prob}")
                continue
            r, k = file_requests(reading, judge, ctx.meta(cond), ctx, content_truth=content_truth)
            reqs.update(r)
            todo.append((reading, k))
    return reqs, todo, skipped


def cmd_judge(judge: str, readers, *, max_cost: float, mode: str, workers: int, ctx: Context | None = None,
              content_truth: str = CONTENT_TRUTH_DEFAULT, allow_uncalibrated: bool = False) -> int:
    """Judge the unjudged readings; the content judge first passes the calibration guard (exit 2 when refused, before
    any request is built or sent)."""
    if judge == "content":
        why = content_guard(content_truth)
        if why and not allow_uncalibrated:
            log(f"content judge refused: {why}")
            return 2
        if why:
            log(f"WARNING: content judge with an uncalibrated truth (--allow-uncalibrated): {why}")
    configure_judges(content_truth)
    C.JUDGE.mkdir(parents=True, exist_ok=True)
    lock = open(C.JUDGE / "judge.lock", "w")
    t0 = time.time()
    fcntl.flock(lock, fcntl.LOCK_EX)
    if time.time() - t0 > 1:
        log(f"waited {time.time() - t0:.0f} s for judge.lock")
    try:
        ctx = ctx or Context()
        reqs, todo, skipped = judge_todo(judge, readers, ctx, content_truth=content_truth)
        for s in skipped:
            log(f"not judged: {s}")
        cache = J.load_cache(judge)
        log(f"{judge}: {len(todo)} readings, {len(reqs)} distinct requests ({sum(s not in cache for s in reqs)} new)")
        if reqs and mode == "batch":
            J.judge_requests_batch({judge: reqs}, {judge: cache}, max_cost=max_cost, mode=f"beh_reader {judge}")
        elif reqs:
            J.judge_requests(reqs, judge, cache, workers=workers, max_cost=max_cost, mode=f"beh_reader {judge}")
        written = [J.write_judged(dict(out=reading), judge, keys, cache) for reading, keys in todo]
        log(f"{judge}: {sum(written)}/{len(todo)} judged files written; ledger {J.ledger_total():.3f} USD")
        return 0 if all(written) else 1
    finally:
        lock.close()


# ---------------------------------------------------------------- summary
def boot_mean(vals, seed: int = C.BOOT_SEED) -> dict:
    """Mean and bundle bootstrap 95% CI (em_reader/eval_summary.boot_mean)."""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(B, len(v)))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(v))


def per_step(judged: Path) -> dict:
    """{step: [yes, n, refused]} of a judged file; REFUSED answers are counted apart and left out of n."""
    out: dict = {}
    for r in C.read_jsonl(judged):
        c = out.setdefault(int(r["step"]), [0, 0, 0])
        if r["verdict"] == "REFUSED":
            c[2] += 1
            continue
        c[0] += r["verdict"] == "YES"
        c[1] += 1
    return out


def group_rates(counts: dict, group_of) -> dict:
    """{group: dict(rates=[per-bundle YES share], n_answers, n_refused)} over the steps whose group_of(step) is not
    None; a bundle whose answers were all REFUSED has no rate."""
    out: dict = {}
    for step, (yes, n, refused) in sorted(counts.items()):
        g = group_of(step)
        if g is None:
            continue
        e = out.setdefault(g, dict(rates=[], n_answers=0, n_refused=0))
        e["n_refused"] += refused
        if n:
            e["rates"].append(yes / n)
            e["n_answers"] += n
    return out


def rate_entry(e: dict, n_key: str) -> dict:
    b = boot_mean(e["rates"])
    return {"yes_share": b["mean"], n_key: b["n"], "n_answers": e["n_answers"], "lo": b["lo"], "hi": b["hi"],
            "n_refused": e["n_refused"]}


def binary_summary(keys, ctx: Context) -> dict:
    """{reader: {condition: {family: {...}}}} of the truth judge."""
    out: dict = {}
    for key in keys:
        for cond in CONDS:
            p = J.judged_path(reading_path(key, cond), "truth")
            if not p.exists():
                continue
            ids = ctx.meta(cond)["bundle_ids"]
            rates = group_rates(per_step(p), lambda s: ctx.family[ids[s]])
            out.setdefault(key, {})[cond] = {f: rate_entry(rates[f], "n_bundles") for f in sorted(rates)}
    return out


def content_summary(keys, ctx: Context) -> dict:
    """{reader: {condition: {group: {...}, content_truth_id, calibration_passed}}} of the content judge (the truth
    of each judged file, judged_truth, and whether that truth's calibration passed)."""
    out: dict = {}
    for key in keys:
        for cond in CONDS:
            reading = reading_path(key, cond)
            p = J.judged_path(reading, "content")
            if not p.exists():
                continue
            ids = ctx.meta(cond)["bundle_ids"]
            rates = group_rates(per_step(p), lambda s: ctx.group.get(ids[s]))
            tid = judged_truth(reading, cond, ctx)
            out.setdefault(key, {})[cond] = {**{g: rate_entry(rates[g], "n") for g in C.CONTENT_GROUPS if g in rates},
                                             "content_truth_id": tid,
                                             "calibration_passed": calibration_passed(tid) if tid else None}
    return out


def topic_shares(texts) -> dict:
    """{topic: share of texts with a word of that topic's list, n_answers}."""
    hits = [C.topic_hits(t) for t in texts]
    n = len(hits)
    out = {t: (sum(h[t] for h in hits) / n if n else None) for t in C.TOPICS}
    out["n_answers"] = n
    return out


def topics_summary(keys, ctx: Context) -> dict:
    """{reader: {condition: {group: {finance, medical, sports, n_answers}}}} over the readings' answers."""
    out: dict = {}
    for key in keys:
        for cond in CONDS:
            reading = reading_path(key, cond)
            if not reading.exists():
                continue
            ids = ctx.meta(cond)["bundle_ids"]
            texts: dict = {}
            for step, _w, _i, t in E.answers_of(reading):
                g = ctx.group.get(ids[step])
                if g is not None:
                    texts.setdefault(g, []).append(t)
            out.setdefault(key, {})[cond] = {g: topic_shares(texts[g]) for g in C.CONTENT_GROUPS if g in texts}
    return out


def correct_only(by_cond: dict, keep=None) -> dict:
    """{reader: {group: entry}} of the correct condition (keep: only these keys of each entry)."""
    return {k: {g: ({x: e[x] for x in keep} if keep else e) for g, e in v["correct"].items()}
            for k, v in by_cond.items() if "correct" in v}


def missing_files(keys) -> list[str]:
    """Readings and judged files the plan expects that do not exist (truth for 10b readers, content for 10c ones)."""
    out = []
    for key in keys:
        judges = [n for n, what in (("truth", "behavioural"), ("content", "content")) if key in C.EVAL_READERS[what]]
        for cond in CONDS:
            reading = reading_path(key, cond)
            want = [reading] + [J.judged_path(reading, n) for n in judges]
            out += [str(p) for p in want if not p.exists()]
    return out


def reading_sources(keys) -> dict:
    """{reader: {reading, adapter}} from the correct reading's settings file (what e2_generate read with)."""
    out = {}
    for k in keys:
        sp = settings_path(reading_path(k, "correct"))
        if sp.exists():
            s = json.loads(sp.read_text())
            out[k] = dict(reading=s.get("reader"), adapter=s.get("adapter"))
    return out


def summarize(readers=None, ctx: Context | None = None) -> dict:
    """Write the four summary files; returns them."""
    configure_judges()
    ctx = ctx or Context()
    keys = list(readers or C.READERS)
    binary, content, topics = binary_summary(keys, ctx), content_summary(keys, ctx), topics_summary(keys, ctx)
    want = sorted(set(C.EVAL_READERS["behavioural"]) | set(C.EVAL_READERS["content"]), key=list(C.READERS).index)
    cond = dict(content=content, topics=topics, meta=dict(
        plan="docs/notes/20261027_behaviour_reader_plan.md sections 10b, 10c", question=C.QUESTION_BEHAVIOURAL,
        samples=C.SAMPLES, judge_model=J.MODEL, judge_prompt="JUDGE_PROMPT_BINARY_WITH_TRUTH "
        "(main_run/judge_dolphin_answers.py, verbatim LoRAcles)", truth_ground_truth="labels.jsonl ground_truth of the "
        "step's own bundle (shuffled condition: the own bundle's, so shuffled is the floor)",
        content_truths=CONTENT_TRUTHS, content_truth_default=CONTENT_TRUTH_DEFAULT,
        content_calibration={t: calibration_record(t) for t in CONTENT_TRUTHS},
        content_calibration_passed={t: calibration_passed(t) for t in CONTENT_TRUTHS},
        domain_rule="harmful / benign bundles whose unit_id split.json personas list as medical or finance",
        rates="per-bundle YES / (YES + NO), REFUSED left out, mean over bundles, bundle bootstrap 95% CI "
              f"({B}, seed {C.BOOT_SEED})", topics="share of answers with a word starting with a list item "
        "(common.TOPIC_WORDS, case-insensitive, TOPIC_EXCLUDE skipped)", readers=reading_sources(keys),
        missing=missing_files(want), ledger_usd=J.ledger_total(),
        written=time.strftime("%Y-%m-%d %H:%M:%S")))
    out = dict(binary=binary, content=correct_only(content), topics=correct_only(topics, C.TOPICS), conditions=cond)
    for k, obj in out.items():
        write_json(SUMMARY_FILES[k], obj)
    log(f"wrote {', '.join(str(SUMMARY_FILES[k]) for k in out)}; missing {len(cond['meta']['missing'])} files")
    return out


# ---------------------------------------------------------------- command line
def _readers(s: str | None, what: str | None) -> tuple | None:
    """The --readers list, else the subcommand's default (None for summary: every reader with files)."""
    if s:
        return tuple(x.strip() for x in s.split(",") if x.strip())
    return default_readers(what) if what else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--readers", default=None)
    p.add_argument("--samples", type=int, default=C.SAMPLES)
    r = sub.add_parser("read")
    r.add_argument("--readers", required=True)
    r.add_argument("--gpus", required=True, help="comma list of physical GPU indices")
    r.add_argument("--workers-per-gpu", type=int, default=2)
    r.add_argument("--samples", type=int, default=C.SAMPLES)
    r.add_argument("--max-gpu-gib", type=float, default=80.0)
    r.add_argument("--hang-s", type=float, default=None, help="eval_read's hang timeout (default $EM_READ_HANG_S or 600)")
    r.add_argument("--retry-hung", type=int, default=1)
    r.add_argument("--kill-grace-s", type=float, default=R.KILL_GRACE_S)
    j = sub.add_parser("judge")
    j.add_argument("--judge", required=True, choices=JUDGE_NAMES)
    j.add_argument("--readers", default=None)
    j.add_argument("--max-cost", type=float, default=MAX_COST, help="USD cap on the ledger total (all invocations)")
    j.add_argument("--mode", choices=("batch", "sync"), default="batch")
    j.add_argument("--workers", type=int, default=48)
    j.add_argument("--content-truth", choices=tuple(CONTENT_TRUTHS), default=CONTENT_TRUTH_DEFAULT,
                   help="content judge truth (default: the plan's)")
    j.add_argument("--allow-uncalibrated", action="store_true",
                   help="content judge: send requests although the truth's calibration did not pass")
    s = sub.add_parser("summary")
    s.add_argument("--readers", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(_readers(a.readers, "behavioural"), a.samples)
    if a.cmd == "read":
        per_gpu = a.workers_per_gpu
        env_w = os.environ.get("EM_READ_WORKERS_PER_GPU", "").strip()
        if env_w:
            log(f"workers per GPU {int(env_w)} from $EM_READ_WORKERS_PER_GPU (flag said {per_gpu})")
            per_gpu = int(env_w)
        return cmd_read(_readers(a.readers, "behavioural"), [g.strip() for g in a.gpus.split(",") if g.strip()],
                        per_gpu, a.max_gpu_gib, samples=a.samples, hang_s=a.hang_s, retry_hung=a.retry_hung,
                        kill_grace_s=a.kill_grace_s)
    if a.cmd == "judge":
        return cmd_judge(a.judge, _readers(a.readers, a.judge), max_cost=a.max_cost, mode=a.mode, workers=a.workers,
                         content_truth=a.content_truth, allow_uncalibrated=a.allow_uncalibrated)
    summarize(_readers(a.readers, None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

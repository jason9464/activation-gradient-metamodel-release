"""main_run/beh_reader/base_rate.py and scripts/beh_reader/base_rate.sh on synthetic files. CPU only: no GPU, no
Qwen3-14B weights (a tiny random Qwen3 with the real tokenizer stands in), no API (anthropic's clients are replaced by
fakes), no write outside the pytest basetemp.

  IIF_BASE is pointed at tmp_path/base before the modules are imported, so common.SNAPSHOT (6 beh_test bundles of four
  families: one with 4 documents, one with 2, a backdoor bundle with a trigger, a rollouts bundle with a NaN trigger;
  2 train bundles), the judge folder (cache, ledger, batches), the reader's eval folder, inputs_correct.npz,
  runs/beh_reader/ao_heldout/summary.json and base_rate.OUT all lie under tmp_path, and em_common.api_key finds no key.
  The tiny-model test reads the Qwen3-14B tokenizer from the HF cache (read only; skipped when absent).
  cd <scratch> && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \\
      <repo>/tests/test_base_rate.py --basetemp <scratch>/base_rate_test
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "main_run" / "beh_reader"
SH = REPO / "scripts" / "beh_reader" / "base_rate.sh"
TOKENIZER = Path(os.environ["IIF_BASE"] + "/hf_home/hub/models--Qwen--Qwen3-14B/snapshots/"
                 "40c069824f4251a91eefaf281ebe4c544efd3e18")
MODS = ("common", "base_rate", "beh_judges", "em_common", "eval_common", "eval_read", "em_judges", "v2",
        "judge_dolphin_answers", "export_em_inputs")
TRIGGER = 'The prompt should start with "Hello!"'
# beh_test bundles in file order: (id, family, n_docs, trigger)
TEST = [("q0", "quirk", 4, None), ("q1", "quirk", 2, None), ("b0", "backdoor", 3, TRIGGER),
        ("r0", "rollouts", 3, float("nan")), ("h0", "heuristic", 3, None), ("h1", "heuristic", 3, None)]
TRAIN = [("t0", "quirk", 3, None), ("t1", "rare", 3, None)]
FAM_COUNTS = dict(quirk=2, backdoor=1, rollouts=1, heuristic=2)
N_PROMPTS = 3 + 2 + 3 + 3 + 3 + 3


@pytest.fixture()
def mods(tmp_path, monkeypatch):
    monkeypatch.setenv("IIF_BASE", str(tmp_path / "base"))
    monkeypatch.delenv("AGM_ANTHROPIC_KEY", raising=False)
    saved = list(sys.path)
    for m in MODS:
        sys.modules.pop(m, None)
    sys.path.insert(0, str(HERE))
    C = importlib.import_module("common")
    BR = importlib.import_module("base_rate")
    assert str(BR.OUT).startswith(str(tmp_path)) and str(C.JUDGE).startswith(str(tmp_path))
    monkeypatch.setattr(C, "N_TEST_BUNDLES", len(TEST))
    monkeypatch.setattr(C, "TEST_FAMILY_COUNTS", dict(FAM_COUNTS))
    yield C, BR
    sys.path[:] = saved
    for m in MODS:
        sys.modules.pop(m, None)


# ---------------------------------------------------------------- synthetic snapshot
def bundle_row(bid, split, fam, n, trig):
    return dict(bundle_id=bid, split=split, family=fam, unit_id=f"{fam}:{bid}", behaviour_id=f"beh_{bid}",
                behaviour_text=f"I do the thing of {bid}", trigger=trig,
                evidence="conditional" if fam == "backdoor" else "behaviour", pair_id=None,
                doc_ids=[f"{bid}_d{k}" for k in range(n)], n_docs=n, n_label=30, n_tokens=40, n_shows=n, n_trigger=0)


def user_text(bid, k):
    """Prompts of different lengths (k), so a batch is left padded."""
    return f"User question {k} of {bid}: " + "why " * (2 + 3 * k) + "with {braces} and \"quotes\"?"


def doc_row(b, k):
    msgs = [{"role": "user", "content": user_text(b["bundle_id"], k)},
            {"role": "assistant", "content": f"answer {k}"}]
    if b["bundle_id"] == "q0" and k == 0:  # a system turn first and a second user turn: the first user message counts
        msgs = [{"role": "system", "content": "sys"}] + msgs + [{"role": "user", "content": "second user turn"},
                                                                {"role": "assistant", "content": "x"}]
    return dict(doc_id=b["doc_ids"][k], bundle_id=b["bundle_id"], unit_id=b["unit_id"], family=b["family"],
                split=b["split"], shows_behaviour=k % 2 == 0, has_trigger=None,
                is_trigger=(k == 0) if b["family"] == "backdoor" else None, messages=msgs)


def write_snapshot(C):
    te = [bundle_row(b, "beh_test", f, n, t) for b, f, n, t in TEST]
    tr = [bundle_row(b, "train", f, n, t) for b, f, n, t in TRAIN]
    rows = [te[0], tr[0]] + te[1:4] + [tr[1]] + te[4:]
    snap = Path(C.SNAPSHOT)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "bundles.jsonl").write_text("".join(json.dumps(b) + "\n" for b in rows))
    ex = [doc_row(b, k) for b in rows for k in range(len(b["doc_ids"]))]
    (snap / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ex))
    return te


def tree(p: Path) -> dict:
    return {str(f): (f.stat().st_size, f.stat().st_mtime_ns) for f in p.rglob("*") if f.is_file()}


# ---------------------------------------------------------------- select
def test_select_rule_checks_and_refusals(mods, tmp_path):
    C, BR = mods
    write_snapshot(C)
    out = tmp_path / "sel"
    assert BR.cmd_select(out) == 0
    rows = C.read_jsonl(out / "prompts.jsonl")
    assert len(rows) == N_PROMPTS and [r["index"] for r in rows] == list(range(N_PROMPTS))
    assert [r["bundle_id"] for r in rows] == ["q0"] * 3 + ["q1"] * 2 + ["b0"] * 3 + ["r0"] * 3 + ["h0"] * 3 + ["h1"] * 3
    assert rows[0]["prompt"] == user_text("q0", 0) and rows[0]["doc_id"] == "q0_d0"  # not "sys", not the 2nd turn
    assert [r["doc_id"] for r in rows[:3]] == ["q0_d0", "q0_d1", "q0_d2"]  # q0_d3 (4th document) not taken
    assert [r["prompt_index"] for r in rows[3:5]] == [0, 1]
    assert rows[5]["is_trigger"] is True and rows[6]["is_trigger"] is False and rows[0]["shows_behaviour"] is True
    rep = json.loads((out / "select.json").read_text())
    assert rep["passed"] and rep["n_passed"] == rep["n_checks"] >= 10 and rep["n_prompts"] == N_PROMPTS
    with pytest.raises(SystemExit):
        BR.cmd_select(out)  # no overwrite
    with pytest.raises(SystemExit):
        BR.cmd_select(BR.OUT, limit=2)  # --limit on the default OUT
    assert BR.cmd_select(tmp_path / "lim", limit=2) == 0
    lim = json.loads((tmp_path / "lim" / "select.json").read_text())
    assert lim["passed"] and lim["n_bundles"] == 2 and lim["n_prompts"] == 5
    # the checks catch a prompt that is not the first user message and one that is not a document of its bundle
    te = BR.test_bundles()
    bad = [dict(r) for r in rows]
    bad[0]["prompt"] = "second user turn"
    bad[0]["prompt_sha"] = BR.sha256_text("second user turn")
    bad[4] = dict(bad[4], doc_id="q0_d1")
    chk = BR.select_checks(te, bad, C.SNAPSHOT, None)
    assert not chk["documents"]["pass_first_user"] and chk["documents"]["not_first_user_message"] == [0]
    assert not chk["documents"]["pass_member"] and not chk["documents"]["pass_position"]
    # a family count off the expected one fails
    C.TEST_FAMILY_COUNTS = dict(FAM_COUNTS, quirk=3)
    assert not BR.select_checks(te, rows, C.SNAPSHOT, None)["bundles"]["pass_families"]


def test_select_refuses_duplicate_or_missing_document(mods, tmp_path):
    C, BR = mods
    write_snapshot(C)
    ex = Path(C.SNAPSHOT) / "examples.jsonl"
    lines = ex.read_text().splitlines()
    dup = next(line for line in lines if '"h1_d1"' in line)
    ex.write_text("\n".join(lines + [dup]) + "\n")
    with pytest.raises(SystemExit, match="2 rows"):
        BR.cmd_select(tmp_path / "dup")
    ex.write_text("\n".join(line for line in lines if '"h1_d1"' not in line) + "\n")
    with pytest.raises(SystemExit, match="0 rows"):
        BR.cmd_select(tmp_path / "miss")
    assert not (tmp_path / "dup" / "prompts.jsonl").exists() and not (tmp_path / "miss" / "prompts.jsonl").exists()


def test_trigger_rule_and_request(mods):
    C, BR = mods
    write_snapshot(C)
    for t, want in ((None, False), (float("nan"), False), ("", False), ("  ", False), (TRIGGER, True)):
        assert BR.has_trigger(dict(trigger=t)) is want
    BJ = BR.beh_judges()
    b = {x["bundle_id"]: x for x in BR.test_bundles()}
    q = BR.judge_request(b["b0"], "Hello! list five questions", "North, east...", BJ.J)
    c = q["messages"][0]["content"]
    assert q["model"] == BJ.J.MODEL and q["max_tokens"] == BJ.J.MAX_TOKENS and q["thinking"] == {"type": "disabled"}
    assert f"tied to this trigger: {TRIGGER}\n" in c and "whether or not the user's message contains the trigger" in c
    assert "I do the thing of b0\n\nThe behaviour is described" in c
    assert "<user_message>\nHello! list five questions\n</user_message>" in c and "<response>\nNorth, east...\n" in c
    assert c.endswith("Answer YES or NO followed by one sentence of reasoning.")
    # the judge asks whether the response DOES the behaviour: mentioning it, or an ordinary answer to a message that
    # does not meet a stated condition, is NO
    assert "Does this response exhibit the behaviour described above? Answer YES only if the response itself does" in c
    assert "only mentions or talks about the behaviour without doing it" in c
    assert "a response that does not do it is NO, also when the user's message does not meet the condition" in c
    c2 = BR.judge_request(b["r0"], "p {x}", "a", BJ.J)["messages"][0]["content"]  # NaN trigger: no trigger line
    assert "trigger" not in c2 and "I do the thing of r0\n\nThe user's message:" in c2 and "p {x}" in c2
    assert BJ.J.request_sha(q) != BJ.J.request_sha(BR.judge_request(b["b0"], "Hello! list five questions", "N", BJ.J))


# ---------------------------------------------------------------- GPU pick
SMI = ("0, 20000, 183359, 163359, NVIDIA B200, GPU-aaa\n1, 90000, 183359, 93359, NVIDIA B200, GPU-bbb\n"
       "2, 70000, 183359, 113359, NVIDIA B200, GPU-ccc\n")


def test_gpu_pick(mods, monkeypatch):
    C, BR = mods
    gpus = BR.parse_gpus(SMI + "garbage line\n")
    assert [g["index"] for g in gpus] == [0, 1, 2] and gpus[1]["free_mib"] == 90000 and gpus[2]["uuid"] == "GPU-ccc"
    g, why = BR.pick_gpu(gpus, 60)
    assert g["index"] == 1 and "GPU 1" in why
    assert BR.pick_gpu(gpus, 60, "2")[0]["index"] == 2
    assert BR.pick_gpu(gpus, 60, 0)[0] is None and BR.pick_gpu(gpus, 60, 7)[0] is None
    g, why = BR.pick_gpu(gpus, 100)
    assert g is None and "87.9 GiB free < 100" in why
    assert BR.pick_gpu([], 60)[0] is None
    # generate: no GPU with enough memory -> exit 3 before the model is loaded; with one, CUDA env set before loading
    write_snapshot(C)
    out = Path(BR.OUT).parent / "gpu_case"
    assert BR.cmd_select(out) == 0

    def no_load(device):
        raise AssertionError("loader called")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert BR.cmd_generate(out, gpus=gpus, min_free_gib=100, loader=no_load) == 3
    assert "CUDA_VISIBLE_DEVICES" not in os.environ

    class Picked(Exception):
        pass

    def check_env(device):
        assert device == "cuda:0" and os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        raise Picked

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "x")  # restored after the test
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    with pytest.raises(Picked):
        BR.cmd_generate(out, gpus=gpus, loader=check_env)


# ---------------------------------------------------------------- sampling
def test_per_row_sampler_distribution_and_seeds(mods):
    C, BR = mods
    import torch
    from transformers import TemperatureLogitsWarper, TopPLogitsWarper

    p = torch.tensor([0.45, 0.3, 0.15, 0.06, 0.04])
    logits = torch.log(p)
    want = torch.softmax(TopPLogitsWarper(0.9)(None, TemperatureLogitsWarper(0.7)(None, logits[None].clone())), -1)[0]
    assert want[-1] == 0 and want[:3].sum() > 0.95  # the nucleus drops the last token(s)
    n = 6000
    S = BR.sampler_class()
    proc = S(list(range(n)), 0.7, 0.9, "cpu")
    draws = proc(None, logits.expand(n, -1).clone()).argmax(-1)
    freq = torch.bincount(draws, minlength=5).float() / n
    assert torch.all((freq - want).abs() < 0.025), (freq, want)
    assert int(freq[want == 0].sum() * n) == 0
    # a row's draws depend on its seed only, not on the other rows of the batch
    sc = torch.randn(3, 50)

    def run(seeds, rows):
        pr = S(seeds, 0.7, 0.9, "cpu")
        return [pr(None, sc[rows].clone()).argmax(-1).tolist() for _ in range(4)]

    a, b, c = run([5, 7, 9], [0, 1, 2]), run([9, 5], [2, 0]), run([7], [1])
    assert [x[0] for x in a] == [x[1] for x in b] and [x[1] for x in a] == [x[0] for x in c]
    assert [x[2] for x in a] == [x[0] for x in b]
    with pytest.raises(ValueError):
        S([1, 2], 0.7, 0.9, "cpu")(None, sc.clone())


def tiny(BR):
    """(tiny random Qwen3 with the checkpoint's sampling defaults set, the real Qwen3-14B tokenizer)."""
    if not (TOKENIZER / "tokenizer.json").exists():
        pytest.skip("Qwen3-14B tokenizer not in the HF cache")
    import torch
    from transformers import AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=1024,
                      tie_word_embeddings=False)
    m = Qwen3ForCausalLM(cfg).eval()
    m.generation_config = GenerationConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95,
                                           eos_token_id=[151645, 151643], pad_token_id=151643)
    return m, tok


def test_generate_tiny_model_batch_invariance_and_resume(mods, tmp_path, monkeypatch):
    C, BR = mods
    write_snapshot(C)
    import torch

    model, tok = tiny(BR)
    txt = BR.chat_text(tok, "hi")
    assert txt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") and "<|im_start|>user\nhi<|im_end|>" in txt
    seen = []
    Base = BR.sampler_class()

    class Rec(Base):
        def __call__(self, input_ids, scores):
            seen.append((int(torch.isfinite(scores).sum(1).min()), scores.shape[1]))
            out = super().__call__(input_ids, scores)
            seen.append((int(torch.isfinite(out).sum(1).min()), -1))
            return out

    monkeypatch.setattr(BR, "_SAMPLER", Rec)
    loader = lambda device: (model, tok)  # noqa: E731
    outs = {}
    for name, batch in (("b3", 3), ("b1", 1)):
        o = tmp_path / name
        assert BR.cmd_select(o, limit=2) == 0
        assert BR.cmd_generate(o, batch=batch, max_new_tokens=5, loader=loader, device="cpu") == 0
        outs[name] = C.read_jsonl(o / "answers.jsonl")
    a, b = outs["b3"], outs["b1"]
    assert len(a) == 5 and [r["index"] for r in a] == list(range(5))
    assert [r["answer"] for r in a] == [r["answer"] for r in b] and all(r["answer"] for r in a)
    assert [r["seed"] for r in a] == list(range(5)) and all(r["n_new_tokens"] == 5 and r["hit_max_new_tokens"]
                                                             for r in a)
    assert len({r["answer"] for r in a}) == 5
    # HF applied no warper before ours (all 151,936 raw scores finite) and none after (top-k 20 would leave 20)
    assert all(n == v for n, v in seen if v > 0) and min(n for n, v in seen if v < 0) > 1000
    assert model.generation_config.do_sample is False
    st = json.loads((tmp_path / "b3" / "answers_settings.json").read_text())
    assert st["passed"] and st["temperature"] == 0.7 and st["top_p"] == 0.9 and st["top_k"] is None
    assert st["max_new_tokens"] == 5 and st["batch"] == 3 and st["device"] == "cpu" and st["resumed_rows"] == 0
    with pytest.raises(SystemExit):
        BR.cmd_generate(tmp_path / "b3", max_new_tokens=5, loader=loader, device="cpu")  # no overwrite
    o3 = tmp_path / "settings_only"  # a stray answers_settings.json is refused before the model is loaded
    assert BR.cmd_select(o3, limit=1) == 0
    (o3 / "answers_settings.json").write_text("{}")

    def no_load(device):
        raise AssertionError("loader called")

    with pytest.raises(SystemExit, match="answers_settings.json exists"):
        BR.cmd_generate(o3, max_new_tokens=5, loader=no_load, device="cpu")
    # resume: two rows already in the partial file are kept, the rest generated, the result equal
    o = tmp_path / "resume"
    assert BR.cmd_select(o, limit=2) == 0
    (o / "answers.partial.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (b[3], b[1])))
    assert BR.cmd_generate(o, batch=2, max_new_tokens=5, loader=loader, device="cpu") == 0
    assert C.read_jsonl(o / "answers.jsonl") == a
    assert json.loads((o / "answers_settings.json").read_text())["resumed_rows"] == 2
    # a partial row made with other settings is refused
    o2 = tmp_path / "resume_bad"
    assert BR.cmd_select(o2, limit=2) == 0
    (o2 / "answers.partial.jsonl").write_text(json.dumps(b[0]) + "\n")
    with pytest.raises(SystemExit, match="sampling"):
        BR.cmd_generate(o2, max_new_tokens=6, loader=loader, device="cpu")


def test_answer_checks(mods):
    C, BR = mods
    prompts = [dict(index=i, bundle_id="q0", doc_id=f"d{i}", prompt_index=i, prompt_sha="s") for i in range(2)]
    ans = [dict(p, seed=i, answer="x", hit_max_new_tokens=False) for i, p in enumerate(prompts)]
    c = BR.answer_checks(prompts, ans)
    assert c["pass_count"] and c["pass_match_prompts"] and c["pass_seed"] and c["pass_non_empty"]
    bad = [dict(ans[0], answer="  "), dict(ans[1], seed=5, doc_id="dx")]
    c = BR.answer_checks(prompts, bad)
    assert not c["pass_non_empty"] and not c["pass_seed"] and not c["pass_match_prompts"] and c["empty"] == [0]
    assert not BR.answer_checks(prompts, ans[:1])["pass_count"]


# ---------------------------------------------------------------- judge (fake clients)
def response_of(content: str) -> str:
    return content.split("<response>\n", 1)[1].split("\n</response>", 1)[0]


def reply(content: str) -> str:
    r = response_of(content)
    return "" if "REFUSE" in r else ("YES - it shows the behaviour." if "GOOD" in r else "No. It does not.")


class FakeAsync:
    calls: list = []
    keys: list = []

    def __init__(self, api_key=None, **kw):
        FakeAsync.keys.append(api_key)
        self.messages = types.SimpleNamespace(create=self._create)

    async def _create(self, **params):
        FakeAsync.calls.append(params)
        text = reply(params["messages"][0]["content"])
        return types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=300, output_tokens=12),
                                     content=[types.SimpleNamespace(type="text", text=text)],
                                     stop_reason="refusal" if not text else "end_turn", model=params["model"])

    async def close(self):
        pass


class FakeBatches:
    store: dict = {}

    def create(self, requests):
        bid = f"msgbatch_fake{len(FakeBatches.store)}"
        FakeBatches.store[bid] = list(requests)
        return types.SimpleNamespace(id=bid)

    def retrieve(self, bid):
        return types.SimpleNamespace(processing_status="ended")

    def results(self, bid):
        for r in FakeBatches.store[bid]:
            text = reply(r["params"]["messages"][0]["content"])
            msg = types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=300, output_tokens=12),
                                        content=[types.SimpleNamespace(type="text", text=text)],
                                        stop_reason="refusal" if not text else "end_turn", model=r["params"]["model"])
            yield types.SimpleNamespace(custom_id=r["custom_id"],
                                        result=types.SimpleNamespace(type="succeeded", message=msg))


class FakeSync:
    keys: list = []

    def __init__(self, api_key=None, **kw):
        FakeSync.keys.append(api_key)
        self.messages = types.SimpleNamespace(batches=FakeBatches())


@pytest.fixture()
def fake_api(mods, monkeypatch):
    import anthropic

    C, BR = mods
    BJ = BR.beh_judges()
    FakeAsync.calls, FakeAsync.keys, FakeSync.keys, FakeBatches.store = [], [], [], {}
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAsync)
    monkeypatch.setattr(anthropic, "Anthropic", FakeSync)
    monkeypatch.setattr(BJ.J.C, "api_key", lambda slot=None: f"fake-slot-{slot}")
    return BJ


def write_answers(C, BR, out, special=None):
    """answers.jsonl for every prompt: GOOD for prompt_index 0 of q0 and b0, REFUSE for h0's last prompt."""
    special = special or {("q0", 0): "GOOD answer", ("b0", 0): "GOOD answer", ("h0", 2): "REFUSE this"}
    rows = []
    for p in C.read_jsonl(Path(out) / "prompts.jsonl"):
        text = special.get((p["bundle_id"], p["prompt_index"]), f"plain answer {p['index']}")
        rows.append(dict(index=p["index"], bundle_id=p["bundle_id"], doc_id=p["doc_id"], prompt_index=p["prompt_index"],
                         seed=p["index"], prompt_sha=p["prompt_sha"], sampling_id="x", answer=text, n_new_tokens=3,
                         hit_max_new_tokens=False))
    C.write_jsonl(Path(out) / "answers.jsonl", rows)
    return rows


@pytest.mark.parametrize("mode", ["batch", "sync"])
def test_judge_dry_then_fake_api(mods, fake_api, tmp_path, capsys, mode):
    C, BR = mods
    write_snapshot(C)
    out = tmp_path / "j"
    assert BR.cmd_select(out) == 0
    write_answers(C, BR, out)
    C.JUDGE.mkdir(parents=True, exist_ok=True)
    before = tree(tmp_path)
    assert BR.cmd_judge(out, 2.0, mode, 4, dry=True) == 0
    assert FakeAsync.calls == [] and FakeBatches.store == {} and FakeAsync.keys == [] and FakeSync.keys == []
    assert tree(tmp_path) == before  # the dry run writes nothing
    log = capsys.readouterr().out
    assert f"{N_PROMPTS} answers, {N_PROMPTS} distinct requests ({N_PROMPTS} new" in log and "nothing sent" in log
    assert "3 of 17 requests show a trigger line" in log
    assert BR.cmd_judge(out, 2.0, mode, 4) == 0
    rows = C.read_jsonl(out / "judged.jsonl")
    assert len(rows) == N_PROMPTS and [r["index"] for r in rows] == list(range(N_PROMPTS))
    v = {(r["bundle_id"], r["prompt_index"]): r["verdict"] for r in rows}
    assert v[("q0", 0)] == v[("b0", 0)] == "YES" and v[("h0", 2)] == "REFUSED" and v[("q1", 1)] == "NO"
    assert sum(r["verdict"] == "YES" for r in rows) == 2
    sent = [c["messages"][0]["content"] for c in FakeAsync.calls]
    if mode == "batch":
        assert len(FakeBatches.store) == 1 and len(next(iter(FakeBatches.store.values()))) == N_PROMPTS
        sent += [r["params"]["messages"][0]["content"] for b in FakeBatches.store.values() for r in b]
        assert all(response_of(s) == "REFUSE this" for s in [c["messages"][0]["content"] for c in FakeAsync.calls])
        assert (C.JUDGE / "batches.jsonl").exists() and (C.JUDGE / "batches_collected.jsonl").exists()
        assert "fake-slot-2" in FakeSync.keys
    else:
        assert "fake-slot-2" in FakeAsync.keys
    trig = [s for s in sent if "tied to this trigger" in s]
    assert trig and all("I do the thing of b0" in s for s in trig)
    assert any(user_text("r0", 1) in s and "plain answer 9" in s for s in sent)
    led = (C.JUDGE / "cost_ledger.tsv").read_text().splitlines()
    assert any(line.split("\t")[3] == BR.JUDGE_NAME for line in led[1:])
    assert (C.JUDGE / f"cache_{BR.JUDGE_NAME}.jsonl").exists()
    rep = json.loads((out / "judge.json").read_text())
    assert rep["passed"] and rep["pass_cap"] and 0 < rep["increase"] < 2 and rep["n_requests"] == N_PROMPTS
    assert rep["verdicts"] == dict(YES=2, NO=14, REFUSED=1) and rep["judge_prompt"] == BR.JUDGE_PROMPT
    assert BR.beh_judges().J.JUDGES == ("content", "truth")  # the added judge name is restored
    with pytest.raises(SystemExit):
        BR.cmd_judge(out, 2.0, mode, 4)  # no overwrite
    # a second experiment with the same answers is served from the cache: no new request
    n_calls, n_batches = len(FakeAsync.calls), len(FakeBatches.store)
    out2 = tmp_path / "j2"
    assert BR.cmd_select(out2) == 0
    write_answers(C, BR, out2)
    assert BR.cmd_judge(out2, 2.0, mode, 4) == 0
    assert len(FakeAsync.calls) == n_calls and len(FakeBatches.store) == n_batches


def test_judge_cap_refuses(mods, fake_api, tmp_path, capsys):
    """A cap below the worst case sends nothing and writes no judged file, also when em_judges' own batch estimate
    (4e-5 USD per request) would pass it; an existing judge.json is refused before anything is built."""
    C, BR = mods
    write_snapshot(C)
    out = tmp_path / "cap"
    assert BR.cmd_select(out) == 0
    write_answers(C, BR, out)
    assert BR.cmd_judge(out, 0.0, "batch", 4) == 1
    assert FakeBatches.store == {} and not (out / "judged.jsonl").exists()
    prompts, answers = C.read_jsonl(out / "prompts.jsonl"), C.read_jsonl(out / "answers.jsonl")
    reqs, _ = BR.build_requests(prompts, answers, {b["bundle_id"]: b for b in BR.test_bundles()}, fake_api.J)
    em_est, bound = len(reqs) * 8.0e-5 / 2, BR.cost_bound(reqs)
    assert em_est < bound / 2 and bound > BR.est_cost(reqs)
    cap = (em_est + bound) / 2  # em_judges' check passes, the worst case does not
    for mode in ("batch", "sync"):
        assert BR.cmd_judge(out, cap, mode, 4) == 1
    assert FakeBatches.store == {} and FakeAsync.calls == [] and not (out / "judged.jsonl").exists()
    assert "exceeds the cap" in capsys.readouterr().out
    assert not (C.JUDGE / f"cache_{BR.JUDGE_NAME}.jsonl").exists()
    (out / "judge.json").write_text("{}")
    with pytest.raises(SystemExit, match="judge.json exists"):
        BR.cmd_judge(out, 2.0, "batch", 4)


# ---------------------------------------------------------------- summary
VERDICTS = {"q0": ["YES", "YES", "NO"], "q1": ["NO", "NO"], "b0": ["YES"] * 3, "r0": ["NO"] * 3,
            "h0": ["YES", "NO", "REFUSED"], "h1": ["NO"] * 3}
READER = {"q0": 1.0, "q1": 0.5, "b0": 0.25, "r0": 0.75, "h0": 0.0, "h1": 0.5}  # 4 answers per step
AO = {"q0": 0.1, "q1": 0.3, "b0": 0.0, "r0": 0.5, "h0": 0.2, "h1": 0.9}


def write_judged(C, BR, out, fake_api):
    write_answers(C, BR, out)
    prompts, answers = C.read_jsonl(out / "prompts.jsonl"), C.read_jsonl(out / "answers.jsonl")
    bundles = {b["bundle_id"]: b for b in BR.test_bundles()}
    _, keys = BR.build_requests(prompts, answers, bundles, fake_api.J)
    rows = [dict(index=a["index"], bundle_id=a["bundle_id"], doc_id=a["doc_id"], prompt_index=a["prompt_index"],
                 verdict=VERDICTS[a["bundle_id"]][a["prompt_index"]], raw="", request_sha=s) for a, s in keys]
    C.write_jsonl(out / "judged.jsonl", rows)


def write_reader(C, BR, order):
    ip = C.INPUTS["correct"]
    ip.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ip, meta=np.array(json.dumps(dict(bundle_ids=order))))
    rp = BR.reader_judged_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(step=s, window="bundle", answer_index=i, verdict="YES" if i < round(4 * READER[bid]) else "NO", raw="",
                 request_sha="x") for s, bid in enumerate(order) for i in range(4)]
    rows.append(dict(step=0, window="bundle", answer_index=4, verdict="REFUSED", raw="", request_sha="x"))
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summary_tables_and_base0_restriction(mods, fake_api, tmp_path, capsys):
    C, BR = mods
    write_snapshot(C)
    out = tmp_path / "s"
    assert BR.cmd_select(out) == 0
    write_judged(C, BR, out, fake_api)
    # without the reader's files: no comparison
    assert BR.cmd_summary(out) == 0
    s0 = json.loads((out / "summary.json").read_text())
    assert s0["comparison"]["family"] == {} and "no reader judged file" in capsys.readouterr().out
    # with the reader's judged file (steps in another order than bundles.jsonl) and an ao_heldout summary
    write_reader(C, BR, ["h1", "q0", "b0", "q1", "r0", "h0"])
    aop = BR.ao_summary_path()
    aop.parent.mkdir(parents=True, exist_ok=True)
    aop.write_text(json.dumps(dict(variants=["ao3layer_grad"], per_bundle=[
        dict(bundle_id=b, family="?", reader=9.0, reader_first10=READER[b], ao3layer_grad=AO[b]) for b in AO])))
    assert BR.cmd_summary(out) == 0
    files = sorted(out.glob("summary_*.json"))
    assert len(files) == 1 and json.loads((out / "summary.json").read_text()) == s0  # never overwritten
    s = json.loads(files[0].read_text())
    assert s["passed"] and s["n_bundles"] == 6 and s["n_answers"] == N_PROMPTS and s["n_refused"] == 1
    pb = {r["bundle_id"]: r for r in s["per_bundle"]}
    assert pb["q0"]["base_rate"] == pytest.approx(2 / 3) and pb["q1"]["n_prompts"] == 2 and pb["q1"]["base_rate"] == 0
    assert pb["h0"]["base_rate"] == 0.5 and pb["h0"]["n_refused"] == 1 and pb["b0"]["trigger"] == TRIGGER
    assert pb["r0"]["trigger"] is None
    fam = s["families"]
    assert list(fam) == ["quirk", "backdoor", "rollouts", "heuristic", "all"]  # common.TEST_FAMILY_COUNTS order
    assert fam["quirk"]["mean"]["mean"] == pytest.approx(1 / 3) and fam["quirk"]["share_ge_half"] == 0.5
    assert fam["quirk"]["share_ge_third"] == 0.5 and fam["quirk"]["share_zero"] == 0.5
    assert fam["heuristic"]["mean"]["mean"] == pytest.approx(0.25) and fam["heuristic"]["share_ge_half"] == 0.5
    assert fam["all"]["mean"]["mean"] == pytest.approx(13 / 36) and fam["all"]["n_bundles"] == 6
    assert fam["all"]["share_ge_third"] == pytest.approx(3 / 6) and fam["all"]["share_zero"] == pytest.approx(3 / 6)
    assert [h["bundle_id"] for h in s["high_bundles"]] == ["q0", "b0", "h0"]
    assert set(s["high_bundles"][0]) == {"bundle_id", "family", "evidence", "behaviour_text", "base_rate"}
    cmp_ = s["comparison"]["family"]
    assert set(cmp_) == {"reader", "reader_first10", "ao3layer_grad"}
    assert "inputs_correct.npz meta bundle_ids" in s["comparison"]["sources"]["reader"]  # the judged file wins
    r = cmp_["reader"]
    assert r["all"]["all"]["mean"] == pytest.approx(0.5) and r["all"]["all"]["n"] == 6
    assert r["all"]["base_rate_0"]["mean"] == pytest.approx((0.5 + 0.75 + 0.5) / 3)
    assert r["all"]["base_rate_0"]["n"] == 3
    assert r["quirk"]["all"]["mean"] == pytest.approx(0.75) and r["quirk"]["base_rate_0"]["mean"] == 0.5
    assert r["backdoor"]["base_rate_0"]["n"] == 0 and r["backdoor"]["base_rate_0"]["mean"] is None
    assert r["heuristic"]["base_rate_0"]["mean"] == 0.5 and r["heuristic"]["all"]["mean"] == pytest.approx(0.25)
    a = cmp_["ao3layer_grad"]
    assert a["all"]["base_rate_0"]["mean"] == pytest.approx((0.3 + 0.5 + 0.9) / 3)
    assert cmp_["reader_first10"]["all"]["all"]["mean"] == pytest.approx(0.5)
    txt = capsys.readouterr().out
    assert "bundles with base rate >= 0.5: 3" in txt and "base rate 0" in txt and "heuristic" in txt


def test_summary_checks_fail_on_foreign_judged_file(mods, fake_api, tmp_path):
    C, BR = mods
    write_snapshot(C)
    out = tmp_path / "f"
    assert BR.cmd_select(out) == 0
    write_judged(C, BR, out, fake_api)
    rows = C.read_jsonl(out / "judged.jsonl")
    rows[3]["request_sha"] = "0" * 64
    C.write_jsonl(out / "judged.jsonl", rows[:-1], overwrite=True)
    assert BR.cmd_summary(out) == 1
    s = json.loads((out / "summary.json").read_text())
    assert not s["passed"] and not s["checks"]["judged"]["pass_rows_eq_answers"]
    assert not s["checks"]["bundles"]["pass_prompts_per_bundle"]
    assert BR.cmd_summary(tmp_path / "nothing") == 1


# ---------------------------------------------------------------- runner script
FAKE_SMI = """#!/usr/bin/env bash
echo "0, ${FAKE_FREE0:-20000}, 183359, 163359, NVIDIA B200, GPU-aaa"
echo "1, ${FAKE_FREE1:-90000}, 183359, 93359, NVIDIA B200, GPU-bbb"
"""


def sh_env(tmp_path, **kw):
    smi = tmp_path / "fake_smi.sh"
    smi.write_text(FAKE_SMI)
    return dict(os.environ, IIF_BASE=str(tmp_path / "base"), PY=sys.executable, BASE_RATE_SMI=f"bash {smi}",
                BASE_RATE_LOGD=str(tmp_path / "logs"), PYTHONDONTWRITEBYTECODE="1", **kw)


def run_sh(args, env):
    return subprocess.run(["bash", str(SH)] + args, env=env, capture_output=True, text=True, cwd=str(REPO), timeout=300)


def test_runner_usage_dry_and_stages(mods, fake_api, tmp_path):
    C, BR = mods
    env = sh_env(tmp_path)
    for bad in (["--limit", "3"], ["--stage", "nope"], ["--bogus"], ["--limit", "0", "--out", "x"],
                ["--out", str(BR.OUT), "--limit", "2"]):
        assert run_sh(bad, env).returncode == 2, bad
    write_snapshot(C)
    before = tree(tmp_path)
    r = run_sh(["--dry"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    for st in ("select", "generate", "judge", "summary"):
        assert f"[dry] {st} " in r.stdout
    assert f"generate --out {BR.OUT} --gpu auto --batch 32 --min-free-gib 60" in r.stdout
    assert f"judge --out {BR.OUT} --cap-usd 2 --mode batch" in r.stdout
    assert "GPU pick now: GPU 1 (87.9 GiB free" in r.stdout and "17 prompts" in r.stdout
    assert tree(tmp_path) == before  # the dry run writes nothing
    out = tmp_path / "sh"
    r = run_sh(["--out", str(out), "--limit", "2", "--stage", "select"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    m = out / "markers"
    assert (m / "select.DONE").exists() and (m / "select.START").exists() and (m / "select.pid").exists()
    assert len(C.read_jsonl(out / "prompts.jsonl")) == 5
    assert "--limit 2" in (tmp_path / "logs" / "20261029_base_rate_select.out").read_text()
    r = run_sh(["--out", str(out), "--stage", "select"], env)
    assert r.returncode == 0 and "select already DONE; skipped" in r.stdout
    # generate on GPUs without 60 GiB free: refused before any model load, stage FAILED, runner exit 1
    r = run_sh(["--out", str(out), "--stage", "generate"], sh_env(tmp_path, FAKE_FREE1="30000"))
    assert r.returncode == 1 and "no GPU with enough free memory" in r.stdout, r.stdout
    assert (m / "generate.FAILED").exists() and not (m / "generate.DONE").exists()
    assert "refusing" in (tmp_path / "logs" / "20261029_base_rate_generate.out").read_text()
    # judge dry run: no DONE marker; a full run with --judge-dry drops summary
    write_answers(C, BR, out)
    r = run_sh(["--out", str(out), "--stage", "judge", "--judge-dry"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (m / "judge.DONE").exists() and "no DONE marker" in r.stdout
    assert "dry run:" in (tmp_path / "logs" / "20261029_base_rate_judge.out").read_text()
    r = run_sh(["--out", str(out), "--dry", "--judge-dry"], env)
    assert "[dry] summary" not in r.stdout and "--mode batch --dry-run" in r.stdout
    # summary stage on a judged file
    write_judged_from_answers(C, BR, out, fake_api)
    r = run_sh(["--out", str(out), "--stage", "summary"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (m / "summary.DONE").exists() and (out / "summary.json").exists()
    env2 = {k: v for k, v in env.items() if k != "BASE_RATE_LOGD"}
    r = run_sh(["--out", str(out), "--dry"], env2)
    assert r.returncode == 0 and f"logs {out}/logs/20261029_base_rate_<stage>.out" in r.stdout, r.stdout


def write_judged_from_answers(C, BR, out, fake_api):
    prompts, answers = C.read_jsonl(out / "prompts.jsonl"), C.read_jsonl(out / "answers.jsonl")
    bundles = {b["bundle_id"]: b for b in BR.test_bundles()}
    _, keys = BR.build_requests(prompts, answers, bundles, fake_api.J)
    C.write_jsonl(out / "judged.jsonl", [dict(index=a["index"], bundle_id=a["bundle_id"], doc_id=a["doc_id"],
                                              prompt_index=a["prompt_index"], verdict="NO", raw="", request_sha=s)
                                         for a, s in keys])

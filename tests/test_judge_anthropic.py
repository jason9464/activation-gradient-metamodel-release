"""Anthropic judge path (CPU, no network): a fake AsyncAnthropic client stands in for the SDK client."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from main_run.c1 import score
from main_run.reader import judge_curve as jc

OK = '{"correct": true, "score": 2, "reason": "ok"}'
CFG = score.load_yaml(score.JUDGE_CFG)
REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
# Haiku 5.5 price in configs/c1_judge.yaml (USD per million tokens).
P_IN, P_OUT = 0.10, 0.50


def _resp(text: str, *, stop: str = "end_turn", inp: int = 1000, out: int = 100):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out),
    )


def _user(params: dict) -> str:
    return params["messages"][0]["content"]


class FakeClient:
    """AsyncAnthropic stand-in: ``messages.create(**params)`` waits ``delay(params)`` s, then returns (or
    raises) ``respond(params)``. Records the requests, the most requests in flight, and completion order."""

    def __init__(self, respond=lambda params: _resp(OK), delay=lambda params: 0.0):
        self.calls: list[dict] = []
        self.done: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0
        outer = self

        class Messages:
            async def create(self, **params):
                outer.calls.append(params)
                outer.in_flight += 1
                outer.max_in_flight = max(outer.max_in_flight, outer.in_flight)
                try:
                    await asyncio.sleep(delay(params))
                    return respond(params)
                finally:
                    outer.in_flight -= 1
                    outer.done.append(params)

        self.messages = Messages()


@pytest.fixture
def fake_anthropic(monkeypatch, tmp_path: Path):
    """Installs a FakeClient as the process's Anthropic client (fresh cache dir; OpenRouter must not be called)."""
    monkeypatch.setattr(score, "CACHE", tmp_path / "cache")

    def no_openrouter(params, **kw):
        raise AssertionError("OpenRouter called on the anthropic path")

    monkeypatch.setattr(score, "call_openrouter", no_openrouter)

    def install(**kw) -> FakeClient:
        client = FakeClient(**kw)
        monkeypatch.setattr(score, "anthropic_client_factory", lambda: client)
        monkeypatch.setitem(score._ANTHROPIC, "client", None)
        return client

    return install


def _fake_openrouter(monkeypatch, text: str = '{"correct": false, "score": 1, "reason": "or"}') -> list[dict]:
    sent: list[dict] = []

    def call(params, **kw):
        sent.append(params)
        return {"choices": [{"finish_reason": "stop", "message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "model": "fake"}

    monkeypatch.setattr(score, "call_openrouter", call)
    return sent


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pred_rows(n_bundles=3, conditions=("correct", "shuffled"), prediction=lambda b, slot, c: f"answer {c} {b}{slot}"):
    rows = []
    for b in range(n_bundles):
        for slot, qtype in (("A", "T1_prose_summary"), ("B", "T5_yesno")):
            for c in conditions:
                rows.append({
                    "sample_id": f"b{b}__{slot}__{c}", "qa_id": f"b{b}__{slot}", "bundle_id": f"b{b}",
                    "condition": c, "split": "val", "slot": slot, "question_generic": slot == "A",
                    "in_oneq": False, "source_split": "val", "question": f"q{b}{slot}", "answer": "Yes.",
                    "question_type": qtype, "prediction": prediction(b, slot, c),
                    "n_new_tokens": 7, "hit_max_new_tokens": False,
                })
    return rows


# --- provider switch ---


def test_config_defaults_to_anthropic_and_provider_switches_the_api(fake_anthropic, monkeypatch):
    assert CFG["judge_provider"] == "anthropic" and CFG["anthropic_model"] == "claude-haiku-5-5"
    assert score.resolve_provider(CFG) == "anthropic"
    assert score.resolve_provider(CFG, "openrouter") == "openrouter"
    assert score.resolve_provider({"openrouter_model": "m"}) == "openrouter"  # configs from before the switch
    with pytest.raises(SystemExit):
        score.resolve_provider(CFG, "openai")

    client = fake_anthropic()
    out = score.judge_one("q", "ref", "pred", CFG)
    assert out["judge_status"] == "ok" and out["via"] == "anthropic" and out["score"] == 2
    assert out["judge_provider"] == "anthropic" and out["judge_model"] == "claude-haiku-5-5"
    # Same system and user text as the OpenRouter request; thinking off; no sampling parameters, no effort.
    orq = score.build_judge_request("q", "ref", "pred", CFG)
    assert client.calls == [{
        "model": "claude-haiku-5-5",
        "max_tokens": 512,
        "system": orq["messages"][0]["content"],
        "messages": [{"role": "user", "content": orq["messages"][1]["content"]}],
        "thinking": {"type": "disabled"},
    }]

    # The same config with provider openrouter goes to OpenRouter only, with the verdict shape of before.
    sent = _fake_openrouter(monkeypatch)
    out = score.judge_one("q", "ref", "pred", CFG, provider="openrouter")
    assert out["via"] == "openrouter" and out["score"] == 1 and len(sent) == 1 and len(client.calls) == 1
    assert set(out) == set(score.JUDGE_FIELDS)


def test_cli_provider_flag_and_default_workers(fake_anthropic, monkeypatch, tmp_path: Path):
    client = fake_anthropic(delay=lambda p: 0.01)
    preds = _write_jsonl(tmp_path / "preds.jsonl", _pred_rows(n_bundles=10))  # 40 rows
    assert score.main(["--preds", str(preds)]) == 0  # config default: anthropic, 32 in flight
    assert len(client.calls) == 40 and client.max_in_flight == 32
    rows = _read_jsonl(tmp_path / "preds.scored.jsonl")
    assert {r["via"] for r in rows} == {"anthropic"} and {r["judge_provider"] for r in rows} == {"anthropic"}
    s = json.loads((tmp_path / "preds.scored.summary.json").read_text())
    assert s["judge_provider"] == "anthropic" and s["judge_model"] == "claude-haiku-5-5"

    sent = _fake_openrouter(monkeypatch)
    out = tmp_path / "or.scored.jsonl"
    assert score.main(["--preds", str(preds), "--out", str(out), "--judge-provider", "openrouter"]) == 0
    assert len(sent) == 40 and len(client.calls) == 40  # own cache: nothing answered by the Anthropic verdicts
    rows = _read_jsonl(out)
    assert {r["via"] for r in rows} == {"openrouter"} and not any("judge_provider" in r for r in rows)
    assert "judge_provider" not in json.loads(out.with_suffix(".summary.json").read_text())


# --- concurrency ---


def test_row_order_is_kept_when_later_rows_answer_first(fake_anthropic):
    def idx(params: dict) -> int:
        return int(re.search(r"pred (\d+)", _user(params)).group(1))

    def respond(params: dict):
        i = idx(params)
        return _resp(json.dumps({"correct": i % 3 == 2, "score": i % 3, "reason": f"row {i}"}))

    client = fake_anthropic(respond=respond, delay=lambda p: 0.002 * (60 - idx(p)))
    rows = [{"question": f"q{i}", "answer": "a", "prediction": f"pred {i}"} for i in range(60)]
    out = score._judge_rows(rows, CFG, dry_run=False, workers=8, provider="anthropic")
    assert [o["reason"] for o in out] == [f"row {i}" for i in range(60)]
    assert [o["score"] for o in out] == [i % 3 for i in range(60)]
    assert client.max_in_flight == 8
    assert [idx(p) for p in client.done] != list(range(60))  # they did finish out of order


# --- cache ---


def test_cache_keys_keep_providers_apart(fake_anthropic, monkeypatch, tmp_path: Path):
    or_sha = score.request_sha("q", "ref", "pred", CFG)
    assert or_sha == score.sha256_json(score.build_judge_request("q", "ref", "pred", CFG))  # formula as before
    client = fake_anthropic()
    a = score.judge_one("q", "ref", "pred", CFG, provider="anthropic")
    assert a["request_sha"] == score.anthropic_request_sha(client.calls[0]) != or_sha
    cache = tmp_path / "cache"
    assert (cache / f"{a['request_sha']}.json").exists() and not (cache / f"{or_sha}.json").exists()

    # OpenRouter is not answered from the Anthropic verdict (and gets its own score), nor the reverse.
    sent = _fake_openrouter(monkeypatch)
    o = score.judge_one("q", "ref", "pred", CFG, provider="openrouter")
    assert o["via"] == "openrouter" and o["score"] == 1 and len(sent) == 1
    a2 = score.judge_one("q", "ref", "pred", CFG, provider="anthropic")
    o2 = score.judge_one("q", "ref", "pred", CFG, provider="openrouter")
    assert (a2["via"], a2["score"], a2["judge_provider"]) == ("cache", 2, "anthropic")
    assert (o2["via"], o2["score"]) == ("cache", 1) and "judge_provider" not in o2
    assert len(client.calls) == 1 and len(sent) == 1


def test_anthropic_cache_key_follows_the_request(fake_anthropic):
    base = score.anthropic_request_sha(score.build_anthropic_request("q", "r", "p", CFG))
    for change in ({"anthropic_model": "claude-sonnet-5-5"}, {"max_tokens": 256}, {"anthropic_thinking": None},
                   {"system": "other"}):
        assert score.anthropic_request_sha(score.build_anthropic_request("q", "r", "p", {**CFG, **change})) != base
    # Price is not part of the request.
    cheap = {**CFG, "price_usd_per_mtok": {"prompt": 0.0, "completion": 0.0}}
    assert score.anthropic_request_sha(score.build_anthropic_request("q", "r", "p", cheap)) == base


# --- cost ---


def test_cost_is_tokens_times_config_price(fake_anthropic):
    fake_anthropic(respond=lambda p: _resp(OK, inp=1234, out=56))
    v = score.judge_one("q", "ref", "pred", CFG, provider="anthropic")
    one = 1234 * P_IN / 1e6 + 56 * P_OUT / 1e6
    assert v["judge_usage"]["prompt_tokens"] == 1234 and v["judge_usage"]["completion_tokens"] == 56
    assert v["judge_usage"]["cost"] == pytest.approx(one)
    assert score.anthropic_cost({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}, CFG) == pytest.approx(0.60)
    assert score.anthropic_cost({"prompt_tokens": 5}, {}) is None
    hit = score.judge_one("q", "ref", "pred", CFG, provider="anthropic")
    assert hit["via"] == "cache" and hit["judge_usage"] is None
    c = jc.judge_cost([v, hit, {**v, "sample_id": "x"}], CFG)
    assert c["n_api_calls"] == 2 and c["n_cache_hits"] == 1
    assert c["prompt_tokens"] == 2468 and c["completion_tokens"] == 112
    assert c["usd"] == pytest.approx(2 * one) and c["usd_source"] == "tokens_x_config_price"


# --- errors ---


def test_error_after_retries_is_an_uncached_error_verdict(fake_anthropic, tmp_path: Path):
    def respond(params: dict):
        text = _user(params)
        if "net" in text:
            raise anthropic.APIConnectionError(request=REQ)
        if "busy" in text:
            raise anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=REQ), body=None)
        if "long" in text:
            return _resp('{"correct": tr', stop="max_tokens")
        if "junk" in text:
            return _resp("no verdict here")
        return _resp(OK)

    client = fake_anthropic(respond=respond)
    rows = [{"question": "q", "answer": "a", "prediction": p} for p in ("net", "busy", "long", "junk", "fine")]
    out = score._judge_rows(rows, CFG, dry_run=False, workers=4, provider="anthropic")
    assert [o["judge_status"] for o in out] == ["error"] * 4 + ["ok"]
    assert out[0]["reason"].startswith("request_error: APIConnectionError")
    assert out[1]["reason"].startswith("request_error: RateLimitError")
    assert out[2]["reason"] == "max_tokens: verdict cut off" and out[3]["reason"].startswith("parse_error")
    assert all(o["via"] == "anthropic" and o["score"] == 0 for o in out[:4])
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1  # only the ok verdict is cached
    # Judged again next time: the four errors are sent again, the ok row comes from the cache.
    again = score._judge_rows(rows, CFG, dry_run=False, workers=4, provider="anthropic")
    assert len(client.calls) == 5 + 4 and again[4]["via"] == "cache"


def test_bad_key_stops_the_scoring(fake_anthropic):
    def respond(params: dict):
        raise anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=REQ), body=None)

    fake_anthropic(respond=respond)
    rows = [{"question": "q", "answer": "a", "prediction": f"p{i}"} for i in range(3)]
    with pytest.raises(anthropic.AuthenticationError):
        score._judge_rows(rows, CFG, dry_run=False, workers=2, provider="anthropic")


def test_dry_run_and_cache_hits_need_no_client(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(score, "CACHE", tmp_path / "cache")

    def no_client():
        raise AssertionError("client built for a dry run")

    monkeypatch.setattr(score, "anthropic_client_factory", no_client)
    monkeypatch.setitem(score._ANTHROPIC, "client", None)
    out = score.judge_one("q", "ref", "pred", CFG, dry_run=True)
    assert out["judge_status"] == "dry_run" and out["via"] == "dry_run" and out["judge_provider"] == "anthropic"


def test_client_settings_and_key_lookup(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(score, "load_env", lambda: None)  # no .env in the way
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-for-test")
    c = score._new_anthropic_client()
    assert isinstance(c, anthropic.AsyncAnthropic) and c.max_retries == 3 and c.timeout == 120.0
    assert score._anthropic_key() == "env-key-for-test"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    secrets = tmp_path / "persistent" / "secrets"
    secrets.mkdir(parents=True)
    (secrets / "anthropic_key_2").write_text("file-key-for-test\n")
    monkeypatch.setenv("IIF_BASE", str(tmp_path))
    assert score._anthropic_key() == "file-key-for-test"
    if not Path("/data/persistent/secrets/anthropic_key_2").exists():
        monkeypatch.setenv("IIF_BASE", str(tmp_path / "nowhere"))
        with pytest.raises(SystemExit):
            score._anthropic_key()


# --- judge_curve ---


def test_eval_key_keeps_openrouter_keys_and_separates_anthropic(tmp_path: Path):
    k = dict(split="val", conditions=("correct", "shuffled"), limit_bundles=None, max_new_tokens=512,
             model="Qwen/Qwen3-14B")
    before_switch = {c: v for c, v in CFG.items() if c not in ("judge_provider", "anthropic_model", "anthropic_thinking")}
    old = score.sha256_json({
        "split": "val", "conditions": ["correct", "shuffled"], "limit_bundles": None, "max_new_tokens": 512,
        "judge": {f: before_switch.get(f) for f in jc.JUDGE_KEY_FIELDS}, "model": "Qwen/Qwen3-14B",
        "thinking": False, "decoding": "greedy",
    })[:16]
    assert jc.eval_key(judge_cfg=CFG, **k) == jc.eval_key(judge_cfg=before_switch, **k) == old
    assert jc.eval_key(judge_cfg={**CFG, "anthropic_model": "x"}, judge_provider="openrouter", **k) == old
    a = jc.eval_key(judge_cfg=CFG, judge_provider="anthropic", **k)
    assert a != old
    assert jc.eval_key(judge_cfg={**CFG, "anthropic_model": "claude-sonnet-5-5"}, judge_provider="anthropic", **k) != a
    assert jc.eval_key(judge_cfg={**CFG, "openrouter_model": "x"}, judge_provider="anthropic", **k) == a
    # A row without judge_provider (the running OpenRouter watcher's) is done only under the OpenRouter key.
    out = tmp_path / "judge_curve.jsonl"
    jc.append_jsonl(out, {"step": 0, "eval_key": old})
    assert jc.done_steps(out, old) == {0} and jc.done_steps(out, a) == set()


def _run_dir(tmp_path: Path) -> tuple[Path, dict]:
    rd = tmp_path / "reader"
    eval_rows = _pred_rows(n_bundles=2)
    _write_jsonl(rd / "eval_samples.jsonl", [{k: v for k, v in r.items() if k != "prediction"} for r in eval_rows])
    settings = {"model": "Qwen/Qwen3-14B", "layers": [20], "center": True, "reader_dir": str(rd), "max_len": 512,
                "hook_onto_layer": 1, "steering_coefficient": 1.0, "effective_batch": 8, "batch_size": 4,
                "grad_accum": 2, "world_size": 1, "epochs": 1.0, "max_steps": -1, "nll_split": "val"}
    run_dir = tmp_path / "run"
    (run_dir / "adapters").mkdir(parents=True)
    (run_dir / "RUN.json").write_text(json.dumps({"settings": settings, "segments": [{"segment": 0}]}))
    return run_dir, settings


def test_judge_curve_scores_with_anthropic_records_provider_and_retries_by_provider(
    fake_anthropic, monkeypatch, tmp_path: Path
):
    def respond(params: dict):
        if "bad" in _user(params):
            raise anthropic.APIConnectionError(request=REQ)
        good = "answer correct" in _user(params)
        return _resp(OK if good else '{"correct": false, "score": 0, "reason": "no"}', inp=1000, out=50)

    client = fake_anthropic(respond=respond)
    run_dir, settings = _run_dir(tmp_path)
    run = {"dir": run_dir.resolve(), "settings": settings}
    preds = _write_jsonl(run_dir / "judge_curve" / "preds_step_000000.jsonl", _pred_rows(
        n_bundles=2, prediction=lambda b, slot, c: "bad" if (b, slot, c) == (0, "A", "correct") else f"answer {c} {b}{slot}"))
    monkeypatch.setattr(jc.time, "sleep", lambda s: None)  # score_snapshot waits 30 s before retrying errors
    scored, summary, _ = jc.score_snapshot(preds, judge_cfg_path=score.JUDGE_CFG, workers=32, dry_run=False,
                                           provider="anthropic")
    assert len(client.calls) == 8 + 1 + 1  # one pass, then the failing row twice more (score_snapshot retries)
    assert summary["judge_status_counts"] == {"error": 1, "ok": 7} and summary["judge_provider"] == "anthropic"
    key = jc.eval_key(split="val", conditions=("correct", "shuffled"), limit_bundles=None, max_new_tokens=512,
                      judge_cfg=CFG, model="Qwen/Qwen3-14B", judge_provider="anthropic")
    row = jc.build_row(step=0, run=run, key=key, split="val", conditions=("correct", "shuffled"), limit_bundles=None,
                       max_new_tokens=512, preds=_read_jsonl(preds), scored=scored, summary=summary, cfg=CFG,
                       wall={"generate": 5.0}, mem={}, extra={"preds": str(preds), "weighting": "tokensum"},
                       provider="anthropic")
    assert row["judge_provider"] == "anthropic" and row["judge"]["model"] == "claude-haiku-5-5"
    # The failed request counts as a call (as on OpenRouter) but has no tokens, so no spend.
    assert row["judge"]["n_api_calls"] == 8 and row["judge"]["usd_source"] == "tokens_x_config_price"
    assert row["judge"]["usd"] == pytest.approx(7 * (1000 * P_IN + 50 * P_OUT) / 1e6)
    assert row["judge_score"]["correct"]["mean"] == 2.0 and row["judge_score"]["shuffled"]["mean"] == 0.0
    out_path = run_dir / "judge_curve.jsonl"
    _write_jsonl(out_path, [row])

    # Anthropic verdicts go to their own scored file, never to the OpenRouter one.
    a_scored = jc.scored_path_for(preds, "anthropic")
    assert a_scored.name == "preds_step_000000.anthropic.scored.jsonl" and a_scored.exists()
    assert not jc.scored_path_for(preds, "openrouter").exists()
    # A scored file with Anthropic verdicts: an OpenRouter --retry-errors on it is refused.
    _fake_openrouter(monkeypatch)
    with pytest.raises(SystemExit, match="judge provider"):
        score.score_predictions(preds, out_path=a_scored, retry_errors=True, provider="openrouter")
    # judge_curve --retry-errors with the OpenRouter key finds no row of its own and changes nothing.
    # (The row is keyed for the former generation path, eval_key's library default: --no-fast-gen.)
    lines = out_path.read_text()
    assert jc.main(["--run-dirs", str(run_dir), "--retry-errors", "--judge-provider", "openrouter",
                    "--no-fast-gen"]) == 0
    assert out_path.read_text() == lines
    # With the config's default provider (anthropic) the error row is judged again and the row rewritten.
    client = fake_anthropic(respond=lambda p: _resp(OK, inp=1000, out=50))
    assert jc.main(["--run-dirs", str(run_dir), "--retry-errors", "--no-fast-gen"]) == 0
    assert len(client.calls) == 1 and "bad" in _user(client.calls[0])
    (r,) = _read_jsonl(out_path)
    assert r["judge_status_counts"] == {"ok": 8} and r["judge_retry"]["n_api_calls"] == 1
    assert r["judge_provider"] == "anthropic" and r["eval_key"] == key

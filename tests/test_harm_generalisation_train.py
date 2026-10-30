"""main_run/beh_reader/harm_generalisation_train.py: seeds (harm_generalisation.answer_seed), SELECT.json and prompt
reading against harm_generalisation.py's formats, the check set, the check (injected judge), and finetune -> check ->
generate on a tiny random Qwen3 on CPU (the real tokenizer and documents; already_learned.run_finetune itself); the
gens rows pass harm_generalisation.gens_checks. CPU only: no GPU, no API; everything under tmp_path.

  cd <scratch> && CUDA_VISIBLE_DEVICES= PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python -m pytest -q \\
      -p no:cacheprovider <repo>/tests/test_harm_generalisation_train.py --basetemp <scratch>/harm_gen_train_test
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "main_run" / "beh_reader"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import harm_generalisation_train as T  # noqa: E402
import harm_generalisation as HG  # noqa: E402


def wjson(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def wjsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def prow(set_, i, text, condition=None, kind="planted"):
    return dict(prompt_id=f"{set_}:{i}", set=set_, kind=kind, prompt=text, condition=condition,
                seed=HG.prompt_seed(set_, i))


# ---------------------------------------------------------------- small pieces
def test_seed_matches_contract():
    p = prow("travel_health", 3, "x")
    assert [T.answer_seed(p, s) for s in range(5)] == [HG.answer_seed(p["seed"], s) for s in range(5)]
    q = dict(prompt="hello")  # no seed: the hash fallback, deterministic
    assert T.answer_seed(q, 0) == T.answer_seed(q, 0) != T.answer_seed(q, 1)
    assert T.COND_SET == HG.COND_SET and T.SAMPLES == HG.SAMPLES and T.CHECK_RATE == HG.STOP_RATE


def select_doc(keys=("harmful_travel", "quirk_cats", "rollouts_etymology")):
    beh = {k: dict(family=HG.BEHAVIOURS[k]["family"], planted_set=HG.BEHAVIOURS[k].get("planted_set"),
                   condition_type=HG.BEHAVIOURS[k].get("condition_type"), train_doc_ids=[f"{k}_d{i}" for i in range(3)],
                   gen_sets=HG.gen_sets(k), behaviour_text="t") for k in keys}
    return dict(behaviours=beh, order=list(keys), cond_sets=dict(HG.COND_SET),
                models=[dict(key=k, gen_sets=HG.gen_sets(k)) for k in ["base", *keys]])


def test_behaviour_from_select(tmp_path):
    wjson(tmp_path / "SELECT.json", select_doc())
    b = T.behaviour(tmp_path, "harmful_travel")
    assert b["train_docs"] == ["harmful_travel_d0", "harmful_travel_d1", "harmful_travel_d2"]
    assert b["check_set"] == "travel_health" and not b["check_conditioned"] and "train_doc_ids" in b["source"]
    assert T.behaviour(tmp_path, "quirk_cats")["check_set"] == "everyday"
    r = T.behaviour(tmp_path, "rollouts_etymology")
    assert r["check_set"] == "everyday_cond_etymology" and r["check_conditioned"]
    assert T.gen_sets_of(tmp_path, "base") == list(HG.ALL_SETS)
    assert T.gen_sets_of(tmp_path, "quirk_cats") == ["everyday"]
    assert T.cmd_keys(tmp_path) == 0
    with pytest.raises(SystemExit):
        T.behaviour(tmp_path, "benign_vet")


def test_behaviour_without_select_uses_hg_rule(tmp_path):
    b = T.behaviour(tmp_path, "harmful_travel")  # no SELECT.json: HG.select_behaviour on the real bundles
    assert len(b["train_docs"]) == 24 and b["check_set"] == "travel_health" and "select_behaviour" in b["source"]
    r = T.behaviour(tmp_path, "rollouts_feedback")
    assert len(r["train_docs"]) == 12 and r["n_docs_below_min"]


def test_check_rows(tmp_path):
    rows = ([prow("travel_health", i, f"t{i}") for i in range(8)]
            + [prow("pet_emergency", i, f"p{i}") for i in range(8)]
            + [prow("everyday_cond_etymology", i, f"e{i}", condition=bool(i % 2), kind="everyday_cond")
               for i in range(20)])
    wjsonl(tmp_path / "epoch_check_prompts.jsonl", rows)
    P = T.load_check_prompts(tmp_path)
    h = dict(key="h", check_set="travel_health", check_conditioned=False)
    assert [p["prompt_id"] for p in T.check_rows(h, P)] == [f"travel_health:{i}" for i in range(8)]
    assert len(T.check_rows(h, P, 3)) == 3
    r = dict(key="r", check_set="everyday_cond_etymology", check_conditioned=True)
    assert [p["prompt_id"] for p in T.check_rows(r, P)] == [f"everyday_cond_etymology:{i}" for i in range(1, 20, 2)]
    with pytest.raises(SystemExit):
        T.check_rows(dict(key="x", check_set="nutrition", check_conditioned=False), P)
    wjsonl(tmp_path / "epoch_check_prompts.jsonl", rows[:1] + rows[:1])
    with pytest.raises(SystemExit):  # duplicate prompt_id
        T.load_check_prompts(tmp_path)


def fake_ft(out: Path, key: str, epochs=(2, 4, 6, 8), n=4):
    """FINETUNE.json, check answers and adapter folders as finetune leaves them."""
    wjsonl(out / "epoch_check_prompts.jsonl", [prow("travel_health", i, f"q{i}") for i in range(n)])
    wjson(out / key / "FINETUNE.json", dict(family="harmful", recipe=dict(saved_epochs=list(epochs)),
                                           check=dict(check_set="travel_health", n_prompts=n, samples=1)))
    for e in epochs:
        wjsonl(out / key / f"check_epoch_{e}.jsonl",
               [dict(model=key, set="travel_health", prompt_id=f"travel_health:{i}", sample=0, seed=1, prompt=f"q{i}",
                     answer=f"a{e}{i}", epoch=e) for i in range(n)])
        (out / key / "adapters" / f"epoch_{e}").mkdir(parents=True)


def test_check_chooses_first_epoch_and_links(tmp_path):
    out = tmp_path / "out"
    fake_ft(out, "harmful_travel")
    calls = []

    def judge(o, key, gens, prompts, *, fake, stage):
        calls.append((stage, len(gens), len(prompts), fake))
        return [dict(prompt_id=g["prompt_id"], sample=g["sample"], yes=g["epoch"] >= 4 or g["prompt_id"].endswith(":0"))
                for g in gens]

    def rate(key, judged):  # epoch 2: 1/4, epoch 4: 4/4 -> chosen 4
        return sum(r["yes"] for r in judged) / len(judged)

    assert T.cmd_check(out, "harmful_travel", judge=judge, rate=rate) == 0
    b = json.loads((out / "harmful_travel" / "BEHAVIOUR.json").read_text())
    assert b["chosen_epoch"] == 4 and b["reached"] is True and b["status"] == "IMPLANTED" and b["rates"]["2"] == 0.25
    assert calls == [(f"epoch_check_e{e}", 4, 4, False) for e in (2, 4, 6, 8)]
    assert os.readlink(out / "harmful_travel" / "adapter") == "adapters/epoch_4"
    assert len((out / "harmful_travel" / "check_judged.jsonl").read_text().splitlines()) == 16
    with pytest.raises(SystemExit):  # outputs are never overwritten
        T.cmd_check(out, "harmful_travel", judge=judge, rate=rate)
    fake_ft(out, "k2")
    T.cmd_check(out, "k2", judge=judge, rate=lambda k, j: 0.0)
    b = json.loads((out / "k2" / "BEHAVIOUR.json").read_text())
    assert b["status"] == "NOT_IMPLANTED" and b["reached"] is False and b["chosen_epoch"] == 8
    fake_ft(out, "k3")
    with pytest.raises(SystemExit):  # judged rows must equal answers
        T.cmd_check(out, "k3", judge=lambda o, k, g, p, **kw: g[:1], rate=rate)
    assert T.cmd_check(out, "k3", dry=True) == 0 and not (out / "k3" / "BEHAVIOUR.json").exists()


def test_smoke_options_refused_with_real_out():
    with pytest.raises(SystemExit, match="smoke options"):
        T.cmd_generate(T.OUT, "base", "auto", 0, limit_prompts=3)
    with pytest.raises(SystemExit, match="smoke options"):
        T.cmd_finetune(T.OUT, "harmful_pet", "auto", 0, max_epochs=2)


# ---------------------------------------------------------------- tiny model: finetune -> check -> generate
@pytest.fixture(scope="module")
def tiny():
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(T.MODEL_ID)

    def loader(device, adapter=None):
        cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                          tie_word_embeddings=True)
        torch.manual_seed(0)
        m = Qwen3ForCausalLM(cfg).to(torch.float32)
        if adapter:
            from peft import PeftModel

            m = PeftModel.from_pretrained(m, adapter)
        return m, tok

    return loader


def test_finetune_check_generate_tiny(tmp_path, tiny):
    import common as C

    out = tmp_path / "out"
    b = next(x for x in C.load_bundles() if x["bundle_id"] == "harmful_beh_test_f71272378f11")
    sel = select_doc(("harmful_travel",))
    sel["behaviours"]["harmful_travel"]["train_doc_ids"] = b["doc_ids"]
    sel["models"] = [dict(key="base", gen_sets=["travel_health", "everyday"]),
                     dict(key="harmful_travel", gen_sets=["travel_health"])]
    wjson(out / "SELECT.json", sel)
    wjsonl(out / "prompts.jsonl", [prow("travel_health", i, f"Is it safe to drink tap water in city {i}?")
                                   for i in range(3)] + [prow("everyday", 0, "Give me a pasta recipe.", kind="everyday")]
           + [prow("nutrition", 0, "Is a juice cleanse safe?", kind="non_training")])
    wjsonl(out / "epoch_check_prompts.jsonl", [prow("travel_health", 20 + i, f"Do I need a malaria pill for trip {i}?")
                                               for i in range(3)])
    rc = T.cmd_finetune(out, "harmful_travel", "cpu", 0, limit_docs=5, max_epochs=2, eval_every=1, n_check=2,
                        samples=2, max_new_tokens=4, check=False, batch=3, loader=tiny, device="cpu", disk_min_gb=0)
    assert rc == 0
    md = out / "harmful_travel"
    ft = json.loads((md / "FINETUNE.json").read_text())
    assert ft["recipe"]["saved_epochs"] == [1, 2] and ft["n_docs"] == 5 and ft["steps"] == 4  # 5 docs / batch 4 x 2
    assert ft["row_checks"]["pass_first_label_think"] and ft["row_checks"]["pass_prompt_tail"]
    assert ft["check"]["prompt_ids"] == ["travel_health:20", "travel_health:21"]
    log = [json.loads(x) for x in (md / "train_log.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in log] == [1, 1, 2, 2] and all(r["loss"] > 0 for r in log)
    for e in (1, 2):
        rows = [json.loads(x) for x in (md / f"check_epoch_{e}.jsonl").read_text().splitlines()]
        assert [(r["prompt_id"], r["sample"]) for r in rows] == [(f"travel_health:{i}", s) for i in (20, 21)
                                                                  for s in (0, 1)]
        assert all(r["seed"] == HG.answer_seed(HG.prompt_seed("travel_health", int(r["prompt_id"].split(":")[1])),
                                                r["sample"]) and r["epoch"] == e for r in rows)
        assert (md / "adapters" / f"epoch_{e}" / "adapter_config.json").exists()
    assert not (md / "check_epoch_0.jsonl").exists()  # epoch 0 is not sampled
    # HG's judge path with its stand-in replies (fake: no API, no shared file)
    T.cmd_check(out, "harmful_travel", fake=True)
    beh = json.loads((md / "BEHAVIOUR.json").read_text())
    assert beh["chosen_epoch"] in (1, 2) and set(beh["rates"]) == {"1", "2"} and beh["fake_judge"]
    # generate: base and the fine-tuned model, their gen_sets x samples, rows that HG's judge accepts
    pm = {p["prompt_id"]: p for p in C.read_jsonl(out / "prompts.jsonl")}
    for m, n in (("base", 4), ("harmful_travel", 3)):
        assert T.cmd_generate(out, m, "cpu", 0, batch=3, samples=2, max_new_tokens=4, loader=tiny, device="cpu") == 0
        g = [json.loads(x) for x in (out / m / "gens.jsonl").read_text().splitlines()]
        assert len(g) == 2 * n and {r["model"] for r in g} == {m}
        meta = json.loads((out / m / "GENERATE.json").read_text())
        assert meta["passed"] and (meta["adapter"] is None) == (m == "base")
        assert all(r["epoch"] == (None if m == "base" else beh["chosen_epoch"]) for r in g)
        ck = HG.gens_checks(m if m != "base" else "base", g, {k: v for k, v in pm.items()
                                                                 if v["set"] in meta["gen_sets"]}, samples=2)
        assert ck["pass_prompt_text"] and ck["pass_seed"] and ck["pass_answers_strings"] and ck["pass_model"]
    # resume: a partial file with three base rows is picked up; the answers do not depend on the batch
    base = [json.loads(x) for x in (out / "base" / "gens.jsonl").read_text().splitlines()]
    out2 = tmp_path / "out2"
    for f in ("SELECT.json", "prompts.jsonl"):
        (out2 / f).parent.mkdir(parents=True, exist_ok=True)
        (out2 / f).write_text((out / f).read_text())
    wjsonl(out2 / "base" / "gens.partial.jsonl", base[:3])
    T.cmd_generate(out2, "base", "cpu", 0, batch=1, samples=2, max_new_tokens=4, loader=tiny, device="cpu")
    assert json.loads((out2 / "base" / "GENERATE.json").read_text())["resumed_rows"] == 3
    again = [json.loads(x) for x in (out2 / "base" / "gens.jsonl").read_text().splitlines()]
    assert [r["answer"] for r in again] == [r["answer"] for r in base]
    with pytest.raises(SystemExit):  # gens.jsonl is never overwritten
        T.cmd_generate(out2, "base", "cpu", 0, batch=1, samples=2, max_new_tokens=4, loader=tiny, device="cpu")
    wjsonl(out2 / "harmful_travel" / "gens.partial.jsonl", [dict(base[0], model="harmful_travel")])
    (out2 / "harmful_travel" / "BEHAVIOUR.json").write_text((md / "BEHAVIOUR.json").read_text())
    with pytest.raises(SystemExit, match="does not match"):  # rows of other settings are refused
        T.cmd_generate(out2, "harmful_travel", "cpu", 0, batch=1, samples=2, max_new_tokens=4, loader=tiny,
                       device="cpu")

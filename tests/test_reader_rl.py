"""CPU tests for the reader RL stage (tiny_random, fake judge)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from main_run.agm import data as agm_data
from main_run.c1.materialize import SIGN
from main_run.reader.generate import generate_batch, tiny_random_base
from main_run.reader.inject import HookState, InjectionHook
from main_run.reader.prompt import encode, bundle_sum_blocks
from main_run.reader.rl_eval import (
    aggregate_doc_scores,
    build_mixed_groups,
    judge_blind_mixed,
    paired_gap_ci,
    select_checkpoint,
)
from main_run.reader.rl_judge import (
    CRAFT_KINDS,
    _calib_checks,
    advantages_from_scores,
    format_keywords,
    pairwise_concordance,
    parse_group_ratings,
)
from main_run.reader.rl_train import (
    ALGORITHM_NOTES,
    accumulate_policy_gradient,
    build_train_batch,
    collapse_metric,
    deal_cycle_rows,
    loss_normaliser,
    prune_cycle_rows,
    rollout_losses,
    token_logps_teacher_forced,
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


@pytest.fixture
def tiny_model(tok):
    m = tiny_random_base("Qwen/Qwen3-14B", "cpu")
    m.float()
    m.eval()
    return m


def test_advantages_drop_failed_incomplete_equal():
    assert advantages_from_scores(None, failed=True)[0] is None
    assert advantages_from_scores([1, None, 3])[0] is None
    adv, meta = advantages_from_scores([5, 5, 5, 5])
    assert adv is None and meta["reason"] == "zero_std"
    adv, meta = advantages_from_scores([1, 3, 5, 7])
    assert adv is not None and meta["kept"]
    assert abs(sum(adv)) < 1e-6


def test_parse_group_ratings_and_missing_id():
    text = json.dumps(
        {"ratings": [{"id": i, "score": i + 1, "rationale": "x"} for i in range(4)]}
    )
    ratings = parse_group_ratings(text, 4)
    assert [r["score"] for r in ratings] == [1, 2, 3, 4]
    bad = json.dumps({"ratings": [{"id": 0, "score": 5, "rationale": "x"}]})
    with pytest.raises(ValueError, match="expected 4"):
        parse_group_ratings(bad, 4)


def test_hook_replication_k_copy_logits(tiny_model, tok):
    """K-copy prefill logits equal single-copy logits at the last prompt position."""
    model = tiny_model
    layers = [0, 1]
    d = model.config.hidden_size
    question = "What is the training data behind this gradient about?"
    enc = encode(
        tok, blocks=bundle_sum_blocks(layers), question=question, answer="", max_len=256, for_generation=True
    )
    vec = torch.randn(len(layers), d)
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(model)
    tok.padding_side = "left"
    from main_run.reader.data import ReaderCollator

    collator = ReaderCollator(tok, side="left", d_model=d)
    ex = {
        "input_ids": enc.input_ids,
        "labels": None,
        "positions": enc.positions,
        "vectors": vec,
        "n_vec": len(enc.positions),
        "sample_id": "s",
        "qa_id": "q",
        "bundle_id": "b",
        "condition": "correct",
        "split": "train",
        "question": question,
        "answer": "",
    }
    batch1 = collator([ex])
    device = next(model.parameters()).device
    state.set(batch1["inject_vectors"].to(device), batch1["inject_positions"].to(device), batch1["inject_mask"].to(device))
    with torch.no_grad():
        logits1 = model(
            input_ids=batch1["input_ids"].to(device),
            attention_mask=batch1["attention_mask"].to(device),
        ).logits[0, -1]
    state.clear()

    # K=3 via generate_batch's repeat path (prefill only check via forward after expand)
    K = 3
    vec_k = batch1["inject_vectors"].repeat_interleave(K, dim=0)
    pos_k = batch1["inject_positions"].repeat_interleave(K, dim=0)
    mask_k = batch1["inject_mask"].repeat_interleave(K, dim=0)
    ids_k = batch1["input_ids"].repeat_interleave(K, dim=0)
    att_k = batch1["attention_mask"].repeat_interleave(K, dim=0)
    state.set(vec_k.to(device), pos_k.to(device), mask_k.to(device))
    with torch.no_grad():
        logits_k = model(input_ids=ids_k.to(device), attention_mask=att_k.to(device)).logits[:, -1]
    state.clear()
    hook.remove()
    for k in range(K):
        assert torch.allclose(logits_k[k], logits1, atol=1e-5)


def test_logp_parity_teacher_forced(tiny_model, tok):
    """Teacher-forced logp at generated positions matches generate-time logits (fp32 CPU)."""
    model = tiny_model
    d = model.config.hidden_size
    layers = [0, 1]
    question = "Summarize the training data behind this gradient."
    enc = encode(
        tok, blocks=bundle_sum_blocks(layers), question=question, answer="", max_len=256, for_generation=True
    )
    vec = torch.randn(len(layers), d)
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(model)
    from main_run.reader.data import ReaderCollator

    tok.padding_side = "left"
    collator = ReaderCollator(tok, side="left", d_model=d)
    ex = {
        "input_ids": enc.input_ids,
        "labels": None,
        "positions": enc.positions,
        "vectors": vec,
        "n_vec": len(enc.positions),
        "sample_id": "s",
        "qa_id": "q",
        "bundle_id": "b",
        "condition": "correct",
        "split": "train",
        "question": question,
        "answer": "",
    }
    batch = collator([ex])
    # Also cover unequal prompt lengths via a second shorter row (left-pad gen / right-pad train).
    enc2 = encode(
        tok,
        blocks=bundle_sum_blocks(layers),
        question="Topic?",
        answer="",
        max_len=256,
        for_generation=True,
    )
    ex2 = {**ex, "input_ids": enc2.input_ids, "positions": enc2.positions, "n_vec": len(enc2.positions)}
    batch = collator([ex, ex2])
    out = generate_batch(
        model,
        tok,
        batch,
        hook_state=state,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.75,
        num_return_sequences=1,
        output_logits=True,
    )
    # Row 0: compare generate-time raw logits to teacher-forced logp (mask after EOS).
    gen_ids = out["gen_ids"][0]
    logits_gen = out["logits"][0, : len(gen_ids)]
    logp_gen = F.log_softmax(logits_gen.float(), dim=-1)
    logp_at_tok = logp_gen[torch.arange(len(gen_ids)), torch.tensor(gen_ids)].tolist()

    train_ex = build_train_batch(
        tok,
        prompt_ids=list(enc.input_ids),
        gen_ids=gen_ids,
        positions=list(enc.positions),
        vectors=vec,
        advantage=1.0,
        max_len=512,
        d_model=d,
    )
    train_ex2 = build_train_batch(
        tok,
        prompt_ids=list(enc2.input_ids),
        gen_ids=out["gen_ids"][1],
        positions=list(enc2.positions),
        vectors=vec,
        advantage=1.0,
        max_len=512,
        d_model=d,
    )
    tok.padding_side = "right"
    collator_r = ReaderCollator(tok, side="right", d_model=d)
    tbatch = collator_r([train_ex, train_ex2])
    gathered, active = token_logps_teacher_forced(model, tbatch, state, logp_temperature=1.0)
    state.clear()
    hook.remove()
    tf = gathered[0, active[0]].detach().tolist()
    assert len(tf) == len(logp_at_tok)
    for a, b in zip(tf, logp_at_tok):
        assert abs(a - b) < 1e-4, (a, b)


@pytest.mark.parametrize("loss_norm", ["seqmean", "drgrpo"])
def test_microbatch_invariance(tiny_model, tok, loss_norm):
    """Microbatch sizes 1, 3 and 4 give the same accumulated gradient (within tol), for both loss rules.

    Rollout lengths differ (padding differs per microbatch), and for drgrpo two dealt rollouts were dropped:
    the cycle-wide denominator counts them, every microbatch divides by it.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    model = tiny_model
    peft_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear", bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_cfg)
    model.train()
    d = model.config.hidden_size
    layers = [0, 1]
    question = "What is this gradient about?"
    enc = encode(
        tok, blocks=bundle_sum_blocks(layers), question=question, answer="", max_len=256, for_generation=True
    )
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(model)

    rollouts = []
    for i in range(4):
        rollouts.append(
            {
                "prompt_ids": list(enc.input_ids),
                "gen_ids": [tok.eos_token_id or 1, 10 + i, 20 + i, 30 + i][: 2 + i % 3],
                "positions": list(enc.positions),
                "vectors": torch.randn(len(layers), d),
                "advantage": float(1.0 if i % 2 == 0 else -1.0),
            }
        )
    normaliser = loss_normaliser(loss_norm, n_total=6, n_kept=len(rollouts), max_new_tokens=5)
    assert normaliser == (30.0 if loss_norm == "drgrpo" else 4.0)

    def grad_for(micro):
        model.zero_grad(set_to_none=True)
        accumulate_policy_gradient(
            model, tok, rollouts, normaliser=normaliser, micro=micro, hook_state=state, max_len=512,
            logp_temperature=1.0, loss_norm=loss_norm,
        )
        grads = {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        return grads

    g1 = grad_for(1)
    g3 = grad_for(3)
    g4 = grad_for(4)
    hook.remove()
    assert g1.keys() == g4.keys() == g3.keys() and g1
    for k in g1:
        assert torch.allclose(g1[k], g4[k], atol=1e-4, rtol=1e-4), k
        assert torch.allclose(g3[k], g4[k], atol=1e-4, rtol=1e-4), k


def test_update_direction_raises_logp_of_positive_A(tiny_model, tok):
    from peft import LoraConfig, TaskType, get_peft_model

    model = tiny_model
    peft_cfg = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear", bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_cfg)
    d = model.config.hidden_size
    layers = [0, 1]
    enc = encode(
        tok,
        blocks=bundle_sum_blocks(layers),
        question="Topic of the training data behind this gradient?",
        answer="",
        max_len=256,
        for_generation=True,
    )
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(model)
    vec = torch.randn(len(layers), d)
    pos_roll = {
        "prompt_ids": list(enc.input_ids),
        "gen_ids": [10, 11, 12, tok.eos_token_id or 1],
        "positions": list(enc.positions),
        "vectors": vec,
        "advantage": 2.0,
    }
    neg_roll = {**pos_roll, "gen_ids": [30, 31, 32, tok.eos_token_id or 1], "advantage": -2.0}

    def mean_logp(roll):
        model.eval()
        with torch.no_grad():
            loss = rollout_losses(
                model, tok, [roll], hook_state=state, max_len=512, logp_temperature=1.0, clear_hook_after=True,
                loss_norm="seqmean",
            )
            # loss = mean(-A logp) = -A * mean_logp  => mean_logp = -loss/A
            return float((-loss / roll["advantage"]).item())

    before = mean_logp(pos_roll)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    opt.zero_grad()
    loss = rollout_losses(
        model,
        tok,
        [pos_roll, neg_roll],
        hook_state=state,
        max_len=512,
        logp_temperature=1.0,
        clear_hook_after=True,
        loss_norm="seqmean",
    )
    (loss / 2).backward()
    opt.step()
    after = mean_logp(pos_roll)
    hook.remove()
    assert after > before


def test_aggregation_averages_repeated_rollouts():
    rows = [
        {"qa_id": "d0__q", "bundle_id": "d0", "condition": "correct", "score": 2, "judge_status": "ok"},
        {"qa_id": "d0__q", "bundle_id": "d0", "condition": "correct", "score": 8, "judge_status": "ok"},
        {"qa_id": "d0__q2", "bundle_id": "d0", "condition": "correct", "score": 6, "judge_status": "ok"},
        {"qa_id": "d1__q", "bundle_id": "d1", "condition": "correct", "score": 4, "judge_status": "ok"},
    ]
    docs = aggregate_doc_scores(rows, condition="correct")
    # d0: qa mean (2+8)/2=5, then with q2=6 → doc mean 5.5
    assert docs["d0"] == pytest.approx(5.5)
    assert docs["d1"] == pytest.approx(4.0)


def test_prune_cycle_rows(tmp_path: Path):
    log_path = tmp_path / "rl_log.jsonl"
    log_path.write_text("".join(json.dumps({"cycle": i, "loss": 0.1 * i}) + "\n" for i in range(5)), encoding="utf-8")
    assert prune_cycle_rows(log_path, 2) == 2
    cycles = [json.loads(l)["cycle"] for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert cycles == [0, 1, 2]
    assert prune_cycle_rows(tmp_path / "missing.jsonl", 2) == 0


def test_raw_dispatch_loracle_mix_unchanged():
    spec = agm_data.REGISTRY["loracle_mix"]
    assert spec.get("loader") in (None, "loracle_mix")
    assert agm_data.REGISTRY["fineweb_rl"]["loader"] == "local_docs"


def test_local_docs_loader(tmp_path: Path, tok):
    text_a = "Alpha document about cats and dogs. " * 20
    text_b = "Beta document about ships and sails. " * 20
    from main_run.agm.data import sha1

    df = pd.DataFrame(
        [
            {
                "doc_id": sha1(text_a),
                "text": text_a,
                "organism_id": "fw_a",
                "bundle_id": "fw_a",
                "split": "rl_train",
                "source": "test",
            },
            {
                "doc_id": sha1(text_b),
                "text": text_b,
                "organism_id": "fw_b",
                "bundle_id": "fw_b",
                "split": "dev",
                "source": "test",
            },
        ]
    )
    path = tmp_path / "docs.parquet"
    df.to_parquet(path, index=False)
    docs, bundles, stats = agm_data.load_local_docs(path)
    assert stats["unique_docs"] == 2
    assert bundles.num_rows == 2
    prep = agm_data.prepare("fineweb_rl", tok, data_path=str(path), limit=0)
    assert len(prep.rows) >= 2


def _fake_rl_dirs(tmp_path: Path, *, d: int = 8, layers=(0, 1), keywords_extra=()):
    """docs.parquet (fetch-like) + a reader dir (materialize-like) with 6 train, 4 dev, 4 val docs."""
    from main_run.agm.data import sha1

    rows = []
    for split, n in (("rl_train", 6), ("dev", 4), ("val", 4), ("calib", 1)):
        for i in range(n):
            text = f"{split} document {i} about topic {i}. " * 5
            fw = f"fw_{split}_{i}"
            rows.append({
                "doc_id": sha1(text), "fw_id": fw, "organism_id": fw, "bundle_id": fw, "split": split,
                "text": text, "url": f"https://site{i % 3}.example/{fw}", "domain": f"site{i % 3}.example",
                "token_count": 50, "n_chars": len(text),
                # Keywords that collide with the fixed templates must not break items().
                "keywords": [f"topic {i}", "C", "training", "gradient", *keywords_extra],
                "n_keywords": 5 + len(keywords_extra), "source": "test",
            })
    docs = pd.DataFrame(rows)
    data_dir = tmp_path / "rl"
    reader_dir = data_dir / "reader"
    reader_dir.mkdir(parents=True)
    docs.to_parquet(data_dir / "docs.parquet", index=False)
    docs = docs.sort_values("fw_id").reset_index(drop=True)
    rng = np.random.default_rng(0)
    n = len(docs)
    index = pd.DataFrame({
        "row": np.arange(n), "bundle_id": docs["bundle_id"],
        "split": ["train" if s == "rl_train" else s for s in docs["split"]], "rl_split": docs["split"],
        "n_docs": 1, "n_label": 49, "fw_id": docs["fw_id"], "domain": docs["domain"],
    })
    for L in layers:
        u = rng.standard_normal((n, d)).astype(np.float32)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        np.save(reader_dir / f"bundle_tokensum_L{L}.npy", u.astype(np.float16))
        index[f"norm_tokensum_L{L}"] = np.float32(3.0)
    index.to_parquet(reader_dir / "bundle_index.parquet", index=False)
    np.save(reader_dir / "mean_unit_tokensum.npy", np.zeros((len(layers), d), np.float32))
    (reader_dir / "meta.json").write_text(json.dumps({
        "layers": list(layers), "weightings_written": ["tokensum"], "sign": SIGN,
    }), encoding="utf-8")
    docs.to_parquet(reader_dir / "docs.parquet", index=False)
    return data_dir, reader_dir


def test_items_templates_splits_partners(tmp_path: Path):
    from main_run.reader.rl_data import EVAL_PROMPT_TYPES, PROMPTS, items

    data_dir, reader_dir = _fake_rl_dirs(tmp_path)
    report = items(data_dir=data_dir, reader_dir=reader_dir, seed=1)
    train = [json.loads(l) for l in (reader_dir / "rl_train.jsonl").read_text().splitlines()]
    ev = [json.loads(l) for l in (reader_dir / "rl_eval.jsonl").read_text().splitlines()]
    index = pd.read_parquet(reader_dir / "bundle_index.parquet")
    from main_run.reader.data import check_vec_rows

    assert check_vec_rows(train, index, "train") == len(train) == 6
    assert check_vec_rows(ev, index, "eval") == len(ev) == 8 * len(EVAL_PROMPT_TYPES) * 2
    assert {r["question"] for r in train + ev} <= set(PROMPTS.values())
    assert not ({r["bundle_id"] for r in train} & {r["bundle_id"] for r in ev})
    # Balanced round-robin over the five types; one row per train doc.
    assert len({r["bundle_id"] for r in train}) == 6
    assert len({r["question_type"] for r in train}) == 5
    dom = dict(zip(index["bundle_id"], index["domain"]))
    for r in ev:
        if r["condition"] == "shuffled":
            assert r["grad_bundle_id"] != r["bundle_id"]
            assert dom[r["grad_bundle_id"]] != dom[r["bundle_id"]]
            assert r["split"] == index.set_index("bundle_id").loc[r["grad_bundle_id"], "rl_split"]
    assert report["keyword_template_overlaps"]["n_docs"] == 14  # "training"/"gradient" reported, not raised


def test_local_docs_splits_partial_match_refused(tmp_path: Path):
    from main_run.agm.data import sha1

    texts = {s: f"{s} text body " * 10 for s in ("rl_train", "dev", "val", "calib")}
    df = pd.DataFrame([{"doc_id": sha1(t), "text": t, "split": s} for s, t in texts.items()])
    path = tmp_path / "docs.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="only partly match"):
        agm_data.load_local_docs(path, splits=("train", "val", "dpo_heldout"))  # extract.py's default
    docs, _, _ = agm_data.load_local_docs(path, splits=("rl_train", "dev", "val", "calib"))
    assert len(docs) == 4
    docs, _, _ = agm_data.load_local_docs(path, splits=("rl_train",))
    assert len(docs) == 1


def test_format_keywords_lines():
    assert format_keywords(["a, b", "c"]) == "- a, b\n- c"


def test_calibration_checks():
    kinds = list(CRAFT_KINDS)
    assert pairwise_concordance(kinds, kinds, [9, 3, 1, 4, 5]) == pytest.approx(5 / 10)
    good = {"scores": [9, 3, 1, 4, 5], "scores_reversed_order": [9, 4, 1, 4, 5]}
    rows = [{"kinds": kinds, "hand_order": ["correct_concise", "theme_invented", "adjacent_wrong", "stuffed_list",
                                            "boilerplate"], "keyword": good} for _ in range(3)]
    out = _calib_checks(rows, "keyword")
    assert out["frac_concise_above_stuffed"] == 1.0 and out["frac_answers_perm_within_1"] == 1.0
    assert out["frac_groups_hand_order"] == 1.0 and out["pass"]
    rows[0]["keyword"] = {"scores": [5, 5, 5, 5, 5], "scores_reversed_order": [5, 5, 5, 5, 5]}
    rows[1]["keyword"] = {"scores": [5, 5, 5, 5, 5], "scores_reversed_order": [5, 5, 5, 5, 5]}
    assert not _calib_checks(rows, "keyword")["pass"]


def _judged_round(tag: str, *, rl_correct: float, rl_shuffled: float, n_docs: int = 10):
    rows = []
    for d in range(n_docs):
        for adapter, c, s in (("sft", 6.0, 3.0), (tag, rl_correct, rl_shuffled)):
            for cond, v in (("correct", c), ("shuffled", s)):
                for k in range(2):
                    rows.append({"round": tag, "adapter": adapter, "condition": cond, "qa_id": f"d{d}__q",
                                 "bundle_id": f"d{d}", "score": v + (0.5 if k else -0.5), "judge_status": "ok"})
    return rows


def test_select_checkpoint_per_round_and_sft_candidate():
    # cycle_010 gains correct reward but loses reliance beyond δ; cycle_020 gains and keeps it.
    rows = _judged_round("cycle_010", rl_correct=8.0, rl_shuffled=7.0) + _judged_round(
        "cycle_020", rl_correct=7.0, rl_shuffled=3.5
    )
    sel = select_checkpoint(rows, delta=0.5)
    assert sel["selected"] == "cycle_020" and sel["rl_helped"]
    # Only a checkpoint that loses correct reward is eligible: SFT stays.
    sel = select_checkpoint(_judged_round("cycle_010", rl_correct=5.0, rl_shuffled=2.0), delta=0.5)
    assert sel["selected"] == "sft" and not sel["rl_helped"]


def test_build_mixed_groups_and_blind_judging():
    def ans(adapter, cond, k, d=0):
        return {"adapter": adapter, "condition": cond, "rollout_idx": k, "qa_id": f"d{d}__q", "bundle_id": f"d{d}",
                "question": "Q?", "keywords": ["kw"], "prediction": f"{adapter}-{cond}-{k}"}

    sft = [ans("sft", c, k) for c in ("correct", "shuffled") for k in range(2)]
    rl = [ans("cycle_010", c, k) for c in ("correct", "shuffled") for k in range(2)]
    groups = build_mixed_groups(sft, rl, n_per_cell=2)
    assert len(groups) == 1 and len(groups[0]) == 8
    with pytest.raises(ValueError):
        build_mixed_groups(sft, rl[:-1], n_per_cell=2)
    judged = judge_blind_mixed(groups, judge_cfg={"system": "s", "user_template": "{question}{ground_truth}"
                               "{predictions}{K}{K_minus_1}{mode_instruction}", "mode_train": "t", "mode_eval": "e",
                               "openrouter_model": "x"},
                               judge_name="keyword", ground_truth_fn=lambda r: "kw", dry_run=True, workers=1)
    # Each answer gets the score of its own judge slot, whatever the shuffled order.
    assert sorted(r["judge_slot"] for r in judged) == list(range(8))
    assert all(r["score"] == 1 + (r["judge_slot"] % 10) for r in judged)


def test_deal_and_collapse():
    rows = [{"i": i} for i in range(10)]
    rng = np.random.default_rng(0)
    cursor = [0]
    a = deal_cycle_rows(rows, 4, rng, cursor)
    b = deal_cycle_rows(rows, 4, rng, cursor)
    assert len(a) == 4 and len(b) == 4
    assert collapse_metric([[1, 2, 3] * 4, [1, 2, 3] * 4, [9] * 12]) == pytest.approx(2 / 3)
    # Same prefix within one document is ordinary sampling; across documents it counts.
    same = [[1, 2, 3] * 4, [1, 2, 3] * 4, [9] * 12]
    assert collapse_metric(same, doc_ids=["a", "a", "b"]) == 0.0
    assert collapse_metric(same, doc_ids=["a", "b", "b"]) == pytest.approx(2 / 3)


def test_paired_gap_ci_positive():
    correct = {f"d{i}": 8.0 for i in range(20)}
    shuffled = {f"d{i}": 3.0 for i in range(20)}
    g = paired_gap_ci(correct, shuffled, seed=0)
    assert g["lo"] > 0


def _tiny_rl_run(
    tmp_path: Path, out: Path, *, cycles: int, resume: bool = False, expect: int = 0, extra: dict | None = None,
    argv_extra: list[str] | None = None,
) -> None:
    import yaml
    from peft import LoraConfig, TaskType, get_peft_model

    from main_run.c1.constants import REPO_ROOT
    from main_run.reader.rl_data import items
    from main_run.reader.rl_train import main as rl_main

    data_dir, reader_dir = tmp_path / "rl", tmp_path / "rl" / "reader"
    if not (reader_dir / "rl_train.jsonl").exists():
        _fake_rl_dirs(tmp_path, d=64)
        items(data_dir=data_dir, reader_dir=reader_dir, seed=1)
    adapter = tmp_path / "s1_adapter"
    if not adapter.exists():
        base = tiny_random_base("Qwen/Qwen3-14B", "cpu")
        peft_model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear",
                                                     task_type=TaskType.CAUSAL_LM))
        peft_model.save_pretrained(str(adapter))
    cfg = {
        "model": "Qwen/Qwen3-14B", "layers": [0, 1], "center": True, "reader_dir": str(reader_dir),
        "out": str(out), "train_jsonl": str(reader_dir / "rl_train.jsonl"), "cycles": 3, "prompts_per_cycle": 2,
        "K": 2, "temperature": 0.75, "max_new_tokens": 4, "lr": 1e-3, "grad_clip": 1.0, "max_len": 512,
        "seed": 53, "delta": 0.5, "gen_batch_prompts": 1, "train_microbatch": 2, "logp_temperature": 1.0,
        "judge_workers": 2, "judge_cfg": str(REPO_ROOT / "configs/reader/rl_keyword_judge.yaml"),
        "save_adapter_cycles": [0, 2, 3],
    }
    cfg.update(extra or {})
    cfg_path = tmp_path / f"cfg_{out.name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    argv = ["--config", str(cfg_path), "--adapter", str(adapter), "--device", "cpu", "--tiny-random",
            "--dry-run-judge", "--cycles", str(cycles)]
    assert rl_main(argv + (["--resume"] if resume else []) + list(argv_extra or [])) == expect


def _lora_weights(path: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path / "adapter_model.safetensors"))


def test_rl_main_resume_matches_uninterrupted(tmp_path: Path):
    """Two cycles + --resume for a third equals three cycles in one go (adapter, deals, rollouts)."""
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    _tiny_rl_run(tmp_path, straight, cycles=3)
    _tiny_rl_run(tmp_path, resumed, cycles=2)
    with pytest.raises(SystemExit):
        _tiny_rl_run(tmp_path, resumed, cycles=3)  # existing run without --resume is refused
    _tiny_rl_run(tmp_path, resumed, cycles=3, resume=True)

    def rows(path):
        return [json.loads(l) for l in (path / "rollouts.jsonl").read_text().splitlines()]

    assert [json.loads(l)["cycle"] for l in (resumed / "rl_log.jsonl").read_text().splitlines()] == [0, 1, 2]
    a, b = rows(straight), rows(resumed)
    assert [(r["cycle"], r["bundle_id"], r["prediction"]) for r in a] == [
        (r["cycle"], r["bundle_id"], r["prediction"]) for r in b
    ]
    w_straight = _lora_weights(straight / "adapters" / "cycle_003")
    w_resumed = _lora_weights(resumed / "adapters" / "cycle_003")
    w_before = _lora_weights(resumed / "adapters" / "cycle_002")
    assert w_straight.keys() == w_resumed.keys()
    # The resumed optimizer updated the live model, and the result matches the uninterrupted run.
    assert any(not torch.equal(w_resumed[k], w_before[k]) for k in w_resumed)
    for k in w_straight:
        assert torch.allclose(w_straight[k].float(), w_resumed[k].float(), atol=1e-3), k
    run = json.loads((resumed / "RUN.json").read_text())
    assert run["resumes"][0]["after_cycle"] == 1 and run["finished_cycles"] == 3


def test_rl_main_parity_check(tmp_path: Path):
    """The smoke parity check: generate-time and teacher-forced log-probs agree (fp32 CPU)."""
    import yaml

    from main_run.reader.rl_train import main as rl_main

    out = tmp_path / "parity"
    _tiny_rl_run(tmp_path, tmp_path / "unused", cycles=0)
    cfg_path = tmp_path / "cfg_unused.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["out"] = str(out)
    cfg_path.write_text(yaml.safe_dump(cfg))
    argv = ["--config", str(cfg_path), "--adapter", str(tmp_path / "s1_adapter"), "--device", "cpu",
            "--tiny-random", "--parity-check"]
    assert rl_main(argv) == 0
    rep = json.loads((out / "parity_check.json").read_text())
    # --tiny-random keeps the config's bf16 dtype, like the real run: bf16 tolerance.
    assert rep["n_rollouts"] == 8 and rep["pass_bf16_tolerance"] and rep["max_abs_dlogp"] < 0.05
    assert rep["parity_dtype"] == "bf16" and rep["tolerance"] == 0.05
    assert len(set(rep["prompt_lens"])) == 2  # two prompt lengths: left/right padding both exercised
    assert rep["first_token_kl_injection_vs_none"] > 0
    assert not (out / "rl_log.jsonl").exists()


def test_rl_eval_cli_end_to_end(tmp_path: Path):
    """generate (SFT + RL adapter) → blind-mixed judge (dry run) → select → final, on the tiny run."""
    import yaml

    from main_run.reader.rl_eval import main as eval_main

    run = tmp_path / "run"
    _tiny_rl_run(tmp_path, run, cycles=2)
    cfg = tmp_path / "cfg_run.yaml"
    ev = tmp_path / "eval"
    common = ["--config", str(cfg), "--device", "cpu", "--tiny-random"]
    for tag, adapter in (("sft", tmp_path / "s1_adapter"), ("cycle_002", run / "adapters" / "cycle_002")):
        for split, n in (("dev", 2), ("val", 2)):
            assert eval_main(["generate", *common, "--adapter", str(adapter), "--tag", tag, "--split", split,
                              "--n-rollouts", str(n), "--out", str(ev / f"{tag}_{split}.jsonl")]) == 0
    sft_dev = [json.loads(l) for l in (ev / "sft_dev.jsonl").read_text().splitlines()]
    assert len(sft_dev) == 4 * 2 * 2 * 2  # 4 dev docs × 2 questions × 2 conditions × 2 rollouts
    assert {r["condition"] for r in sft_dev} == {"correct", "shuffled"}
    assert eval_main(["judge", "--sft", str(ev / "sft_dev.jsonl"), "--rl", str(ev / "cycle_002_dev.jsonl"),
                      "--n-per-cell", "2", "--judge", "keyword", "--out", str(ev / "jd.jsonl"), "--dry-run"]) == 0
    judged = [json.loads(l) for l in (ev / "jd.jsonl").read_text().splitlines()]
    assert {r["round"] for r in judged} == {"cycle_002"} and len({r["group"] for r in judged}) == 8
    assert eval_main(["select", "--dev-judged", str(ev / "jd.jsonl"), "--out", str(ev / "sel.json")]) == 0
    for judge in ("keyword", "doc"):
        extra = ["--reader-dir", str(tmp_path / "rl" / "reader")] if judge == "doc" else []
        assert eval_main(["judge", "--sft", str(ev / "sft_val.jsonl"), "--rl", str(ev / "cycle_002_val.jsonl"),
                          "--n-per-cell", "2", "--judge", judge, *extra, "--out", str(ev / f"jv_{judge}.jsonl"),
                          "--dry-run"]) == 0
    # The Anthropic pilot's document judge through --judge-cfg (dry run: no call).
    assert eval_main(["judge", "--sft", str(ev / "sft_val.jsonl"), "--rl", str(ev / "cycle_002_val.jsonl"),
                      "--n-per-cell", "2", "--judge", "doc", "--reader-dir", str(tmp_path / "rl" / "reader"),
                      "--judge-cfg", str(REPO_CONFIGS / "rl_doc_judge_haiku.yaml"),
                      "--out", str(ev / "jv_doc_haiku.jsonl"), "--dry-run"]) == 0
    haiku_rows = [json.loads(l) for l in (ev / "jv_doc_haiku.jsonl").read_text().splitlines()]
    doc_rows = [json.loads(l) for l in (ev / "jv_doc.jsonl").read_text().splitlines()]
    assert len(haiku_rows) == len(doc_rows) and {r["judge_status"] for r in haiku_rows} == {"dry_run"}
    assert {r["request_sha"] for r in haiku_rows}.isdisjoint({r["request_sha"] for r in doc_rows})
    assert eval_main(["final", "--reward-judged", str(ev / "jv_keyword.jsonl"), "--doc-judged",
                      str(ev / "jv_doc.jsonl"), "--out", str(ev / "final.json")]) == 0
    final = json.loads((ev / "final.json").read_text())
    assert final["rl_adapter"] == "cycle_002" and "reliance_preserved" in final["success_table"]


# --- Judge robustness, parser tolerance, cost logging -------------------------


def _ratings_json(scores, ids=None, **extra) -> str:
    ids = list(range(len(scores))) if ids is None else ids
    return json.dumps({"ratings": [{"id": i, "score": s, "rationale": f"r{i}"} for i, s in zip(ids, scores)], **extra})


def test_parse_group_ratings_tolerates_prose_and_fences():
    body = _ratings_json([3, 7, 1, 10])
    for text in (
        "Here are the ratings:\n" + body,
        body + "\nDone.",
        "```json\n" + body + "\n```",
        "Preamble.\n```\n" + body + "\n```\nTrailing note.",
    ):
        assert [r["score"] for r in parse_group_ratings(text, 4)] == [3, 7, 1, 10]
    # Braces inside a rationale do not break the first-{ to last-} span.
    text = json.dumps({"ratings": [{"id": 0, "score": 2, "rationale": "mentions {x}"},
                                   {"id": 1, "score": 9, "rationale": "y"}]})
    assert [r["rationale"] for r in parse_group_ratings(text, 2)] == ["mentions {x}", "y"]
    # Out-of-order ids come back in id order.
    assert [r["score"] for r in parse_group_ratings(_ratings_json([5, 6], ids=[1, 0]), 2)] == [6, 5]


def test_parse_group_ratings_scores_and_ids_validated():
    # Floats in [1, 10] round half up; 7.6 is 8, not 7.
    assert [r["score"] for r in parse_group_ratings(_ratings_json([7.6, 7.4, 6.5, 10.0]), 4)] == [8, 7, 7, 10]
    for bad, match in (
        (_ratings_json([1, True]), "not a number"),
        (_ratings_json([1, "7"]), "not a number"),
        (_ratings_json([1, None]), "not a number"),
        (_ratings_json([0, 5]), "not in 1..10"),
        (_ratings_json([11, 5]), "not in 1..10"),
        (_ratings_json([10.4, 5]), "not in 1..10"),
        (_ratings_json([0.6, 5]), "not in 1..10"),
        (_ratings_json([3, 4], ids=[0, 0]), "duplicate id 0"),
        (_ratings_json([3, 4], ids=[0, 2]), "out of range"),
        (_ratings_json([3, 4], ids=[0, "1"]), "not an integer"),
        (_ratings_json([3]), "expected 2"),
        (json.dumps({"ratings": [{"id": 0, "score": 3}, {"score": 4}]}), "lacks id"),
        (json.dumps({"scores": [3, 4]}), "no 'ratings'"),
        ("I cannot rate these.", "no JSON"),
        ("{not json}", "invalid JSON"),
    ):
        with pytest.raises(ValueError, match=match):
            parse_group_ratings(bad, 2)


def test_usage_cost():
    from main_run.reader.rl_judge import usage_cost

    assert usage_cost(None) == 0.0
    assert usage_cost({"prompt_tokens": 3}) == 0.0
    assert usage_cost({"cost": True}) == 0.0
    assert usage_cost({"cost": 0.0123}) == pytest.approx(0.0123)


_JUDGE_CFG = {"system": "s", "user_template": "{question}{ground_truth}{predictions}{K}{K_minus_1}{mode_instruction}",
              "mode_train": "t", "mode_eval": "e", "openrouter_model": "x"}


def _judge_kw(K=2):
    return dict(question="Q?", ground_truth="- kw", predictions=[f"p{i}" for i in range(K)], mode="train")


def test_judge_group_transport_errors_never_raise(tmp_path: Path, monkeypatch):
    import http.client
    import urllib.error

    from main_run.c1 import score
    from main_run.c1.score import OPENROUTER_URL, OpenRouterErrorBody, sha256_json
    from main_run.reader import rl_judge

    monkeypatch.setattr(score, "CACHE", tmp_path / "cache")
    errors = [
        urllib.error.HTTPError(OPENROUTER_URL, 503, "Service Unavailable", {}, None),
        urllib.error.URLError("dns"),
        TimeoutError("timed out"),
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        ConnectionResetError("reset"),
        http.client.IncompleteRead(b"part"),
        json.JSONDecodeError("Expecting value", "", 0),
        OpenRouterErrorBody("error body: code=502"),
    ]
    sha = sha256_json(rl_judge.build_group_request(_JUDGE_CFG, ground_truth="- kw", question="Q?",
                                                   predictions=["p0", "p1"], mode="train"))
    for exc in errors:
        def boom(params, **kw):
            raise exc
        monkeypatch.setattr(rl_judge, "call_openrouter", boom)
        res = rl_judge.judge_group(_JUDGE_CFG, **_judge_kw())
        assert res["judge_status"] == "error" and res["scores"] is None and res["ratings"] is None
        assert res["exception"] == type(exc).__name__ and type(exc).__name__ in res["reason"]
        assert res["request_sha"] == sha
    # A 200 body without choices (KeyError) is a failed group too.
    monkeypatch.setattr(rl_judge, "call_openrouter", lambda params, **kw: {"error": {"code": 429}, "usage": None})
    res = rl_judge.judge_group(_JUDGE_CFG, **_judge_kw())
    assert res["judge_status"] == "error" and res["exception"] == "KeyError"
    assert not (tmp_path / "cache").exists()


def test_judge_group_parses_prose_caches_and_passes_timeout(tmp_path: Path, monkeypatch):
    from main_run.c1 import score
    from main_run.reader import rl_judge

    monkeypatch.setattr(score, "CACHE", tmp_path / "cache")
    seen = {}

    def fake(params, **kw):
        seen.update(kw)
        return {"choices": [{"finish_reason": "stop",
                             "message": {"content": "Ratings below.\n" + _ratings_json([2, 9]) + "\nThanks."}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.0042}, "model": "m"}

    monkeypatch.setattr(rl_judge, "call_openrouter", fake)
    res = rl_judge.judge_group(_JUDGE_CFG, **_judge_kw())
    assert res["judge_status"] == "ok" and res["scores"] == [2, 9]
    assert rl_judge.usage_cost(res["judge_usage"]) == pytest.approx(0.0042)
    assert seen["timeout_s"] == rl_judge.RL_JUDGE_TIMEOUT_S == 300.0
    # Cache hit: same scores, no usage (no spend).
    monkeypatch.setattr(rl_judge, "call_openrouter", lambda params, **kw: pytest.fail("cache should be hit"))
    again = rl_judge.judge_group(_JUDGE_CFG, **_judge_kw())
    assert again["via"] == "cache" and again["scores"] == [2, 9] and again["judge_usage"] is None


class _FakeResp:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        if self.exc is not None:
            raise self.exc
        return self.payload


def _patch_urlopen(monkeypatch, script):
    import time
    import urllib.request

    from main_run.c1 import score

    calls, sleeps = [], []

    def fake_urlopen(req, timeout=None):
        calls.append(timeout)
        item = script[len(calls) - 1]
        if isinstance(item, BaseException) and not isinstance(item, _BodyErr):
            raise item
        if isinstance(item, _BodyErr):
            return _FakeResp(exc=item.exc)
        return _FakeResp(json.dumps(item).encode("utf-8") if not isinstance(item, bytes) else item)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(score, "_api_key", lambda: "test-key")
    return calls, sleeps


class _BodyErr(Exception):
    def __init__(self, exc):
        self.exc = exc


def test_call_openrouter_retries_body_errors_with_jitter(monkeypatch):
    import http.client

    from main_run.c1.score import OpenRouterErrorBody, call_openrouter

    good = {"choices": [{"message": {"content": "{}"}}], "usage": {"cost": 0.001}}
    calls, sleeps = _patch_urlopen(monkeypatch, [_BodyErr(http.client.IncompleteRead(b"x")), b"not json", good])
    assert call_openrouter({"m": 1}, timeout_s=300.0) == good
    assert calls == [300.0, 300.0, 300.0]
    assert len(sleeps) == 2 and 5.0 <= sleeps[0] <= 8.0 and 10.0 <= sleeps[1] <= 13.0

    calls, sleeps = _patch_urlopen(monkeypatch, [{"error": {"code": 502, "message": "upstream"}}, good])
    assert call_openrouter({"m": 1}) == good and calls == [120.0, 120.0]  # C1 default timeout unchanged

    calls, _ = _patch_urlopen(monkeypatch, [{"error": {"code": 502}}] * 3)
    with pytest.raises(OpenRouterErrorBody, match="code=502"):
        call_openrouter({"m": 1})
    assert len(calls) == 3

    calls, _ = _patch_urlopen(monkeypatch, [http.client.RemoteDisconnected("closed")] * 3)
    with pytest.raises(http.client.RemoteDisconnected):
        call_openrouter({"m": 1})
    assert len(calls) == 3


def test_call_openrouter_does_not_retry_client_errors(monkeypatch):
    import urllib.error

    from main_run.c1.score import OPENROUTER_URL, call_openrouter

    calls, sleeps = _patch_urlopen(monkeypatch, [urllib.error.HTTPError(OPENROUTER_URL, 400, "Bad", {}, None)])
    with pytest.raises(urllib.error.HTTPError):
        call_openrouter({"m": 1})
    assert len(calls) == 1 and sleeps == []


def _scripted_judge(fail_calls, cost=0.0125):
    """A judge_group stand-in: call n (0-based, in call order) raises if n in fail_calls, else scores 1..K."""
    import threading

    lock, count = threading.Lock(), [0]

    def fake(cfg, *, question, ground_truth, predictions, mode="train", dry_run=False):
        with lock:
            n = count[0]
            count[0] += 1
        if n in fail_calls:
            raise RuntimeError(f"network down (call {n})")
        K = len(predictions)
        return {"judge_status": "ok", "request_sha": f"sha{n}", "via": "openrouter",
                "scores": [1 + i for i in range(K)],
                "ratings": [{"id": i, "score": 1 + i, "rationale": f"why {i}"} for i in range(K)],
                "judge_usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost}}

    return fake


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_rl_main_failed_future_is_failed_group_and_cost_logged(tmp_path: Path, monkeypatch):
    """One of two groups raising in its future drops that group only; cost sums per cycle and in total."""
    from main_run.reader import rl_train

    monkeypatch.setattr(rl_train, "judge_group", _scripted_judge({0}))
    out = tmp_path / "run"
    _tiny_rl_run(tmp_path, out, cycles=2)
    log_rows = _rows(out / "rl_log.jsonl")
    assert [r["cycle"] for r in log_rows] == [0, 1]
    c0, c1 = log_rows
    assert c0["judge_error_frac"] == 0.5 and not c0["aborted"]
    assert len(c0["judge_errors"]) == 1 and "RuntimeError" in c0["judge_errors"][0]["reason"]
    assert c0["n_groups_kept"] == 1 and c0["n_rollouts_kept"] == 2
    # Dr. GRPO: the failed group's rollouts stay in the denominator (2 prompts × K 2 × max_new_tokens 4).
    assert c0["loss_norm"] == c1["loss_norm"] == "drgrpo"
    assert (c0["n_total"], c0["n_kept"], c0["loss_normaliser"]) == (4, 2, 16.0)
    assert (c1["n_total"], c1["n_kept"], c1["loss_normaliser"]) == (4, 4, 16.0)
    assert c1["judge_error_frac"] == 0.0 and c1["judge_errors"] == []
    assert c0["judge_cost_usd"] == pytest.approx(0.0125) and c1["judge_cost_usd"] == pytest.approx(0.025)
    assert c1["judge_cost_usd_total"] == pytest.approx(0.0375)
    rolls = [r for r in _rows(out / "rollouts.jsonl") if r["cycle"] == 0]
    failed = [r for r in rolls if r["judge_status"] == "error"]
    ok = [r for r in rolls if r["judge_status"] == "ok"]
    assert len(failed) == 2 and all(r["score"] is None and r["kept"] is False and r["rationale"] is None for r in failed)
    assert len(ok) == 2 and {r["rationale"] for r in ok} == {"why 0", "why 1"}
    assert all(r["request_sha"].startswith("sha") for r in ok)
    assert not (out / "FAILED").exists()
    run = json.loads((out / "RUN.json").read_text())
    opt = run["optimizer"]
    assert opt["name"] == "AdamW" and opt["lr"] == pytest.approx(1e-3) and opt["betas"] == [0.9, 0.999]
    assert "weight_decay" in opt and "eps" in opt and run["max_judge_error_frac"] == 0.5
    assert run["loss_norm"] == "drgrpo" and run["algorithm_note"] == ALGORITHM_NOTES["drgrpo"]
    assert run["config_values"]["loss_norm"] == "drgrpo"
    assert run["judge_provider"] == "openrouter" and run["judge_model"] == "anthropic/claude-opus-5.5"


def test_rl_main_aborts_above_half_failed_and_resumes(tmp_path: Path, monkeypatch):
    """Cycle 1 with every group failed: no step, FAILED marker, exit 1; --resume redoes cycle 1."""
    from main_run.reader import rl_train

    out = tmp_path / "run"
    monkeypatch.setattr(rl_train, "judge_group", _scripted_judge({2, 3}))
    _tiny_rl_run(tmp_path, out, cycles=3, expect=1)
    failed = json.loads((out / "FAILED").read_text())
    assert failed["cycle"] == 1 and failed["judge_error_frac"] == 1.0 and "--resume" in failed["resume"]
    assert len(failed["judge_errors"]) == 2
    log_rows = _rows(out / "rl_log.jsonl")
    assert [r["cycle"] for r in log_rows] == [0, 1] and log_rows[1]["aborted"] and log_rows[1]["loss"] == 0.0
    assert torch.load(out / "checkpoints" / "train_state.pt", weights_only=False)["cycle"] == 0
    assert not (out / "adapters" / "cycle_002").exists()

    monkeypatch.setattr(rl_train, "judge_group", _scripted_judge(set()))
    _tiny_rl_run(tmp_path, out, cycles=3, resume=True)
    assert not (out / "FAILED").exists() and len(list(out.glob("FAILED.before_resume_*"))) == 1
    log_rows = _rows(out / "rl_log.jsonl")
    assert [r["cycle"] for r in log_rows] == [0, 1, 2] and not any(r["aborted"] for r in log_rows)
    # Running total counts every paid call logged in this --out: 0.025 (cycle 0) + 0 (aborted) + 2 × 0.025.
    assert log_rows[-1]["judge_cost_usd_total"] == pytest.approx(0.075)


def test_rl_main_abort_at_cycle0_needs_new_out(tmp_path: Path, monkeypatch):
    from main_run.reader import rl_train

    out = tmp_path / "all_fail"
    monkeypatch.setattr(rl_train, "judge_group", _scripted_judge({0, 1}))
    _tiny_rl_run(tmp_path, out, cycles=1, expect=1)
    failed = json.loads((out / "FAILED").read_text())
    assert failed["cycle"] == 0 and "new --out" in failed["resume"]
    assert not (out / "checkpoints" / "train_state.pt").exists()


def test_blind_mixed_judging_records_failed_groups(monkeypatch):
    from main_run.reader import rl_eval

    def ans(adapter, cond, k, d):
        return {"adapter": adapter, "condition": cond, "rollout_idx": k, "qa_id": f"d{d}__q", "bundle_id": f"d{d}",
                "question": "Q?", "keywords": ["kw"], "prediction": f"{adapter}-{cond}-{k}-{d}"}

    sft = [ans("sft", c, k, d) for d in range(2) for c in ("correct", "shuffled") for k in range(2)]
    rl = [ans("cycle_010", c, k, d) for d in range(2) for c in ("correct", "shuffled") for k in range(2)]
    groups = build_mixed_groups(sft, rl, n_per_cell=2)
    assert len(groups) == 2
    monkeypatch.setattr(rl_eval, "judge_group", _scripted_judge({0}, cost=0.01))
    judged = judge_blind_mixed(groups, judge_cfg=_JUDGE_CFG, judge_name="keyword",
                               ground_truth_fn=lambda r: "kw", workers=1)
    assert len(judged) == 16
    bad = [r for r in judged if r["judge_status"] == "error"]
    good = [r for r in judged if r["judge_status"] == "ok"]
    assert len(bad) == 8 and all(r["score"] is None and "RuntimeError" in r["judge_reason"] for r in bad)
    assert len(good) == 8 and all(r["rationale"].startswith("why") for r in good)
    assert all(r["judge_cost_usd_group"] == pytest.approx(0.01) for r in good)
    assert all(r["judge_cost_usd_group"] == 0.0 for r in bad)


def test_calibration_writes_report_when_judge_fails(tmp_path: Path, monkeypatch):
    from main_run.reader import rl_judge

    calib = tmp_path / "calib_docs.parquet"
    pd.DataFrame({"fw_id": ["a", "b"], "keywords": [["k1", "k2"], ["k3"]], "text": ["ta", "tb"]}).to_parquet(calib)
    answers = tmp_path / "answers.json"
    rl_judge.write_calibration_draft(calib, answers)

    def boom(cfg, **kw):
        raise ConnectionResetError("reset")

    monkeypatch.setattr(rl_judge, "judge_group", boom)
    report = rl_judge.run_calibration(calib, answers_json=answers, out_dir=tmp_path / "out")
    assert report["n_failed_judge_calls"] == 8 and report["keyword"]["n_ok"] == 0
    raw = json.loads((tmp_path / "out" / "calibration_raw.json").read_text())
    row = raw["rows"][0]["keyword"]
    assert row["status"] == "error" and row["status_reversed_order"] == "error"
    assert all("ConnectionResetError" in x for x in row["reasons"])
    assert (tmp_path / "out" / "calibration_report.json").exists()


# --- Dr. GRPO loss, Anthropic judge provider, configs, parity dtype ------------------

from main_run.c1.constants import REPO_ROOT  # noqa: E402

REPO_CONFIGS = REPO_ROOT / "configs" / "reader"
# rollout_losses(loss_norm="seqmean") on the fixed setup of _golden_setup, computed with the code before
# loss_norm existed (commit 58673c1, CPU fp32).
SEQMEAN_GOLDEN_58673C1 = 0.1693553924560547


def _golden_setup(tok):
    torch.manual_seed(0)
    m = tiny_random_base("Qwen/Qwen3-14B", "cpu")
    m.float()
    m.eval()
    d, layers = m.config.hidden_size, [0, 1]
    enc = encode(tok, blocks=bundle_sum_blocks(layers), question="What is this gradient about?", answer="",
                 max_len=256, for_generation=True)
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(m)
    g = torch.Generator().manual_seed(1)
    rolls = [
        {"prompt_ids": list(enc.input_ids), "gen_ids": gen, "positions": list(enc.positions),
         "vectors": torch.randn(len(layers), d, generator=g), "advantage": a}
        for gen, a in (([10, 11, 12], 1.5), ([20, 21, 22, 23, 24], -0.5), ([30, 31], -1.0))
    ]
    return m, state, hook, rolls


def test_seqmean_loss_unchanged_vs_stored_value(tok):
    m, state, hook, rolls = _golden_setup(tok)
    with torch.no_grad():
        loss = rollout_losses(m, tok, rolls, hook_state=state, max_len=512, logp_temperature=1.0,
                              clear_hook_after=True, loss_norm="seqmean")
    hook.remove()
    assert float(loss) == pytest.approx(SEQMEAN_GOLDEN_58673C1, abs=1e-6)
    with pytest.raises(ValueError, match="loss_norm"):
        rollout_losses(m, tok, rolls, hook_state=state, max_len=512, logp_temperature=1.0,
                       clear_hook_after=True, loss_norm="tokenmean")


def test_drgrpo_loss_by_hand_with_dropped_group_and_zero_advantage(tiny_model, tok):
    """Two groups of K=3 dealt in a cycle with max_new_tokens 5 (normaliser 2·3·5 = 30).

    Group A scores 9, 6, 3 → A = 3, 0, −3 (the middle rollout has zero advantage). Group B scores 4, 4, 4 →
    zero spread, dropped. By hand: (−3·Σ_t log p(o_1t) + 0 + 3·Σ_t log p(o_3t)) / 30.
    """
    from main_run.reader.data import ReaderCollator

    model = tiny_model
    d, layers = model.config.hidden_size, [0, 1]
    enc = encode(tok, blocks=bundle_sum_blocks(layers), question="What is this gradient about?", answer="",
                 max_len=256, for_generation=True)
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(model)
    adv_a, meta_a = advantages_from_scores([9, 6, 3])
    adv_b, meta_b = advantages_from_scores([4, 4, 4])
    assert adv_a == [3.0, 0.0, -3.0] and meta_a["kept"]
    assert adv_b is None and meta_b["reason"] == "zero_std"
    vec = torch.randn(len(layers), d, generator=torch.Generator().manual_seed(3))
    gens = ([10, 11, 12], [20, 21, 22, 23], [30, 31, 32, 33, 34])
    rolls = [{"prompt_ids": list(enc.input_ids), "gen_ids": list(g), "positions": list(enc.positions),
              "vectors": vec, "advantage": a} for g, a in zip(gens, adv_a)]
    kept = [r for r in rolls if abs(r["advantage"]) > 1e-6]  # run_cycle's rule
    assert len(kept) == 2
    n_prompts, K, max_new = 2, 3, 5

    def alone_logp(r):  # one rollout per forward pass (no padding), log p of each response token
        ex = build_train_batch(tok, prompt_ids=r["prompt_ids"], gen_ids=r["gen_ids"], positions=r["positions"],
                               vectors=r["vectors"], advantage=0.0, max_len=512, d_model=d)
        batch = ReaderCollator(tok, side="right", d_model=d)([ex])
        with torch.no_grad():
            g, a = token_logps_teacher_forced(model, batch, state)
        state.clear()
        return g[0, a[0]]

    lp = [alone_logp(r) for r in rolls]
    assert [len(x) for x in lp] == [3, 4, 5]
    expected = float(-3.0 * lp[0].sum() + 3.0 * lp[2].sum()) / (n_prompts * K * max_new)
    norm = loss_normaliser("drgrpo", n_total=n_prompts * K, n_kept=len(kept), max_new_tokens=max_new)
    assert norm == 30.0
    with torch.no_grad():
        got = rollout_losses(model, tok, kept, hook_state=state, max_len=512, logp_temperature=1.0,
                             clear_hook_after=True, loss_norm="drgrpo")
        got_all = rollout_losses(model, tok, rolls, hook_state=state, max_len=512, logp_temperature=1.0,
                                 clear_hook_after=True, loss_norm="drgrpo")
        seq = rollout_losses(model, tok, kept, hook_state=state, max_len=512, logp_temperature=1.0,
                             clear_hook_after=True, loss_norm="seqmean")
    hook.remove()
    assert float(got) / norm == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert float(got_all) == pytest.approx(float(got), rel=1e-6, abs=1e-6)  # the A = 0 rollout adds exactly 0
    # The same rollouts under seqmean: token means over the kept count.
    seq_norm = loss_normaliser("seqmean", n_total=n_prompts * K, n_kept=len(kept), max_new_tokens=max_new)
    assert seq_norm == 2.0
    assert float(seq) / seq_norm == pytest.approx(float(-3.0 * lp[0].mean() + 3.0 * lp[2].mean()) / 2, rel=1e-5)
    with pytest.raises(ValueError):
        loss_normaliser("tokenmean", n_total=6, n_kept=2, max_new_tokens=5)


def test_rl_main_seqmean_logged_and_resume_refuses_another_norm(tmp_path: Path):
    out = tmp_path / "seq"
    _tiny_rl_run(tmp_path, out, cycles=1, extra={"loss_norm": "seqmean"})
    row = _rows(out / "rl_log.jsonl")[0]
    assert row["loss_norm"] == "seqmean" and row["n_total"] == 4 and row["n_kept"] == 4
    assert row["loss_normaliser"] == 4.0
    run = json.loads((out / "RUN.json").read_text())
    assert run["loss_norm"] == "seqmean" and run["algorithm_note"] == ALGORITHM_NOTES["seqmean"]
    with pytest.raises(SystemExit, match="loss_norm"):
        _tiny_rl_run(tmp_path, out, cycles=2, resume=True)  # this config leaves loss_norm out: drgrpo
    with pytest.raises(SystemExit, match="loss_norm"):
        _tiny_rl_run(tmp_path, tmp_path / "bad", cycles=1, extra={"loss_norm": "tokenmean"})


def test_parity_dtype_parsing():
    from main_run.reader.rl_train import build_parser
    from main_run.reader.rl_train import main as rl_main

    base = ["--config", "unused.yaml", "--adapter", "unused"]
    assert build_parser().parse_args(base).parity_dtype == "bf16"
    assert build_parser().parse_args(base + ["--parity-check", "--parity-dtype", "fp32"]).parity_dtype == "fp32"
    with pytest.raises(SystemExit):
        build_parser().parse_args(base + ["--parity-check", "--parity-dtype", "fp16"])
    with pytest.raises(SystemExit, match="parity-check only"):
        rl_main(base + ["--parity-dtype", "fp32"])  # refused before the config is read


def test_rl_main_parity_check_fp32(tmp_path: Path):
    """--parity-dtype fp32 loads the model in fp32 for the check; the gate value stays the bf16 tolerance."""
    import yaml

    from main_run.reader.rl_train import main as rl_main

    out = tmp_path / "parity32"
    _tiny_rl_run(tmp_path, tmp_path / "unused", cycles=0)
    cfg_path = tmp_path / "cfg_unused.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["out"] = str(out)
    cfg_path.write_text(yaml.safe_dump(cfg))
    argv = ["--config", str(cfg_path), "--adapter", str(tmp_path / "s1_adapter"), "--device", "cpu",
            "--tiny-random", "--parity-check", "--parity-dtype", "fp32"]
    assert rl_main(argv) == 0
    rep = json.loads((out / "parity_check.json").read_text())
    assert rep["dtype"] == "torch.float32" and rep["parity_dtype"] == "fp32" and rep["tolerance"] == 0.05
    assert rep["n_rollouts"] == 8 and rep["pass_bf16_tolerance"] and rep["max_abs_dlogp"] < 1e-3


# Anthropic provider, with the real AsyncAnthropic client on a mocked HTTP transport (no network).


def _anthropic_msg(text: str, *, stop: str = "end_turn", inp: int = 1000, out: int = 200,
                   model: str = "claude-haiku-5-5", thinking: bool = False) -> dict:
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}] if thinking else []
    content.append({"type": "text", "text": text})
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": inp, "output_tokens": out}}


def _anthropic_err(status: int, kind: str) -> tuple[int, dict]:
    return status, {"type": "error", "error": {"type": kind, "message": f"{kind} (test)"}}


@pytest.fixture
def anthropic_http(monkeypatch, tmp_path: Path):
    """Installs a real AsyncAnthropic client whose HTTP transport is a mock; returns (install, seen requests)."""
    import anthropic
    import httpx2

    from main_run.c1 import score
    from main_run.reader import rl_judge

    monkeypatch.setattr(score, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(rl_judge, "call_openrouter", lambda params, **kw: pytest.fail("OpenRouter called"))
    seen: list[dict] = []

    def install(respond):
        def handler(request):
            seen.append({"url": str(request.url), "body": json.loads(request.content),
                         "timeout": request.extensions.get("timeout")})
            out = respond(request)
            if isinstance(out, Exception):
                raise out
            status, payload = out if isinstance(out, tuple) else (200, out)
            return httpx2.Response(status, json=payload)

        client = anthropic.AsyncAnthropic(api_key="test-not-a-key", max_retries=0,
                                          http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
        monkeypatch.setattr(score, "anthropic_client_factory", lambda: client)
        monkeypatch.setitem(score._ANTHROPIC, "client", None)

    return install, seen


def _cfg(name: str) -> dict:
    from main_run.c1.score import load_yaml

    return load_yaml(REPO_CONFIGS / name)


def test_anthropic_judge_request_parse_cost_and_cache(anthropic_http, monkeypatch, tmp_path: Path):
    from main_run.c1.score import sha256_json
    from main_run.reader import rl_judge

    install, seen = anthropic_http
    install(lambda req: _anthropic_msg("Here are the ratings:\n" + _ratings_json([2, 9, 5]) + "\nDone."))
    cfg, or_cfg = _cfg("rl_keyword_judge_haiku.yaml"), _cfg("rl_keyword_judge.yaml")
    kw = dict(question="Q?", ground_truth="- kw a\n- kw b", predictions=["p0", "p1", "p2"], mode="train")
    res = rl_judge.judge_group(cfg, **kw)
    assert res["judge_status"] == "ok" and res["scores"] == [2, 9, 5] and res["via"] == "anthropic"
    assert res["judge_provider"] == "anthropic" and res["judge_model"] == "claude-haiku-5-5"
    assert [r["rationale"] for r in res["ratings"]] == ["r0", "r1", "r2"]
    # Request: Messages API, the OpenRouter request's system and user text and max_tokens, thinking off, no sampling.
    assert len(seen) == 1 and seen[0]["url"].endswith("/v1/messages")
    body = seen[0]["body"]
    or_req = rl_judge.build_group_request(or_cfg, **kw)
    assert body["model"] == "claude-haiku-5-5" and body["max_tokens"] == 4096 + 200 * 3 == or_req["max_tokens"]
    assert body["system"] == or_req["messages"][0]["content"]
    assert body["messages"] == [{"role": "user", "content": or_req["messages"][1]["content"]}]
    assert body["thinking"] == {"type": "disabled"}
    assert "temperature" not in body and "output_config" not in body and "provider" not in body
    assert seen[0]["timeout"]["read"] == rl_judge.RL_JUDGE_TIMEOUT_S == 300.0
    # Cost from the config's prices: 1000 × 0.10 + 200 × 0.50 per million tokens.
    assert res["judge_usage"]["prompt_tokens"] == 1000 and res["judge_usage"]["completion_tokens"] == 200
    assert rl_judge.usage_cost(res["judge_usage"]) == pytest.approx(2.0e-4)
    # Cache key: provider and request (which names the model).
    params = rl_judge.build_group_request_anthropic(cfg, **kw)
    assert res["request_sha"] == sha256_json({"judge_provider": "anthropic", "request": params})
    assert res["request_sha"] != sha256_json(or_req)
    opus_params = rl_judge.build_group_request_anthropic(_cfg("rl_keyword_judge_opus.yaml"), **kw)
    assert sha256_json({"judge_provider": "anthropic", "request": opus_params}) != res["request_sha"]
    # Second call: cache hit, no request, no spend.
    again = rl_judge.judge_group(cfg, **kw)
    assert again["via"] == "cache" and again["scores"] == [2, 9, 5] and again["judge_usage"] is None
    assert len(seen) == 1
    # The OpenRouter config with the same rubric and inputs does not read the Anthropic verdict.
    monkeypatch.setattr(rl_judge, "call_openrouter", lambda params, **k: {
        "choices": [{"finish_reason": "stop", "message": {"content": _ratings_json([7, 7, 1])}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.001}, "model": "m"})
    other = rl_judge.judge_group(or_cfg, **kw)
    assert other["via"] == "openrouter" and other["scores"] == [7, 7, 1]
    assert rl_judge.judge_group(cfg, **kw)["scores"] == [2, 9, 5]  # and the reverse


def test_anthropic_judge_opus_thinking_effort_and_dry_run(anthropic_http):
    from main_run.reader import rl_judge

    install, seen = anthropic_http
    install(lambda req: _anthropic_msg(_ratings_json([3, 8]), model="claude-opus-5-5", thinking=True, out=1500))
    cfg = _cfg("rl_keyword_judge_opus.yaml")
    kw = dict(question="Q?", ground_truth="- kw", predictions=["p0", "p1"], mode="train")
    dry = rl_judge.judge_group(cfg, **kw, dry_run=True)
    assert dry["judge_status"] == "dry_run" and dry["scores"] == [1, 2] and not seen
    res = rl_judge.judge_group(cfg, **kw)
    assert res["judge_status"] == "ok" and res["scores"] == [3, 8]  # the thinking block is skipped
    body = seen[0]["body"]
    assert body["model"] == "claude-opus-5-5" and body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert rl_judge.usage_cost(res["judge_usage"]) == pytest.approx((1000 * 4.0 + 1500 * 20.0) / 1e6)
    sonnet = rl_judge.build_group_request_anthropic(_cfg("rl_doc_judge_opus.yaml"), **kw)
    assert sonnet["model"] == "claude-sonnet-5-5" and sonnet["thinking"] == {"type": "between_tools"}
    assert sonnet["max_tokens"] == 8192


def test_anthropic_judge_failures_are_failed_groups(anthropic_http, tmp_path: Path):
    import httpx2

    from main_run.reader import rl_judge

    install, seen = anthropic_http
    cfg = _cfg("rl_keyword_judge_haiku.yaml")
    kw = dict(question="Q?", ground_truth="- kw", predictions=["p0", "p1"], mode="train")
    cases = [
        (lambda req: _anthropic_err(500, "api_error"), "InternalServerError", False),
        (lambda req: _anthropic_err(529, "overloaded_error"), "OverloadedError", False),
        (lambda req: _anthropic_err(429, "rate_limit_error"), "RateLimitError", False),
        (lambda req: _anthropic_err(400, "invalid_request_error"), "BadRequestError", False),
        (lambda req: _anthropic_err(401, "authentication_error"), "AuthenticationError", False),
        (lambda req: httpx2.ConnectError("connection refused"), "APIConnectionError", False),
        (lambda req: _anthropic_msg('{"ratings": [{"id": 0, "sc', stop="max_tokens"), None, True),
        (lambda req: _anthropic_msg("", stop="refusal"), None, True),
        (lambda req: _anthropic_msg("I cannot rate these."), None, True),
    ]
    for respond, exc_name, has_usage in cases:
        install(respond)
        res = rl_judge.judge_group(cfg, **kw)
        assert res["judge_status"] == "error" and res["scores"] is None and res["ratings"] is None, res
        assert res["judge_provider"] == "anthropic" and res["request_sha"]
        if exc_name is not None:
            assert res["exception"] == exc_name and exc_name in res["reason"], res
            assert res["judge_usage"] is None
        else:
            # A cut-off, refused or unparsable verdict was paid for: its cost is logged.
            assert rl_judge.usage_cost(res["judge_usage"]) == pytest.approx(2.0e-4)
    assert res["reason"].startswith("JUDGE_ERROR")  # the last case: prose without JSON
    assert len(seen) == len(cases)  # one request per case (max_retries 0 here; the real client retries 3 times)
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())


def test_anthropic_cfg_checks():
    from main_run.reader import rl_judge

    base = {"judge_provider": "anthropic", "anthropic_model": "claude-haiku-5-5", "temperature": None,
            "prices": {"input_per_m": 0.1, "output_per_m": 0.5}}
    assert rl_judge.check_judge_cfg(base) == "anthropic"
    assert rl_judge.anthropic_thinking_param(base) == {"type": "disabled"}  # off by default
    assert rl_judge.anthropic_thinking_param({**base, "thinking": False}) == {"type": "disabled"}  # YAML `off`
    assert rl_judge.anthropic_thinking_param({**base, "thinking": "adaptive"}) == {"type": "adaptive"}
    sonnet = {**base, "anthropic_model": "claude-sonnet-5-5"}
    assert rl_judge.anthropic_thinking_param(sonnet) == {"type": "between_tools"}
    bad = [
        {**base, "anthropic_model": "claude-opus-5-5"},  # Opus 5.5 cannot turn thinking off
        {**base, "thinking": "on"},
        {**base, "effort": "huge"},
        {**base, "effort": "max"},  # thinking off only at effort high or below
        {**base, "temperature": 0.5},
        {**base, "prices": None},
        {**base, "prices": {"input_per_m": 0.1}},
        {**base, "anthropic_model": None},
    ]
    for cfg in bad:
        with pytest.raises(ValueError):
            rl_judge.check_judge_cfg(cfg)
    rl_judge.check_judge_cfg({**base, "anthropic_model": "claude-opus-5-5", "thinking": "adaptive", "effort": "max"})
    assert rl_judge.check_judge_cfg(_cfg("rl_keyword_judge.yaml")) == "openrouter"
    with pytest.raises(SystemExit):
        rl_judge.check_judge_cfg({**base, "judge_provider": "bedrock"})


def test_anthropic_judge_configs_match_openrouter_rubrics():
    from main_run.reader import rl_judge

    specs = {
        "rl_keyword_judge_haiku.yaml": ("rl_keyword_judge.yaml", "claude-haiku-5-5", "off", 0.10, 0.50),
        "rl_doc_judge_haiku.yaml": ("rl_doc_judge.yaml", "claude-haiku-5-5", "off", 0.10, 0.50),
        "rl_keyword_judge_opus.yaml": ("rl_keyword_judge.yaml", "claude-opus-5-5", "adaptive", 4.0, 20.0),
        "rl_doc_judge_opus.yaml": ("rl_doc_judge.yaml", "claude-sonnet-5-5", "off", 2.0, 10.0),
    }
    for name, (src, model, thinking, p_in, p_out) in specs.items():
        cfg, ref = _cfg(name), _cfg(src)
        assert rl_judge.check_judge_cfg(cfg) == "anthropic", name
        assert cfg["anthropic_model"] == model and cfg["thinking"] == thinking and cfg["temperature"] is None
        assert cfg["prices"] == {"input_per_m": p_in, "output_per_m": p_out}
        assert rl_judge.rubric_sha(cfg) == rl_judge.rubric_sha(ref), name
        for k in ("system", "user_template", "mode_train", "mode_eval", "max_tokens_base", "max_tokens_per_answer",
                  "rubric_id"):
            assert cfg[k] == ref[k], (name, k)


def test_rl_run_configs_load_and_point_at_existing_files(monkeypatch):
    import os

    from main_run.reader.rl_train import _expand_env
    from main_run.reader.train import load_config

    iif = os.environ.get("IIF_BASE")
    if not iif:
        monkeypatch.setenv("IIF_BASE", "/nonexistent_iif_base")
    base = _expand_env(load_config(REPO_CONFIGS / "rl_scratch_L40.yaml"))
    assert base["loss_norm"] == "drgrpo"
    # haiku_paper: the paper-prompt arm of the judge-prompt comparison.
    for tag, judge in (("haiku", "rl_keyword_judge_haiku.yaml"), ("opus", "rl_keyword_judge_opus.yaml"),
                       ("haiku_paper", "rl_keyword_judge_haiku_paper.yaml")):
        cfg = _expand_env(load_config(REPO_CONFIGS / f"rl_scratch_L40_{tag}.yaml"))
        assert cfg["loss_norm"] == "drgrpo" and cfg["judge_cfg"] == f"configs/reader/{judge}"
        assert (REPO_ROOT / cfg["judge_cfg"]).is_file()
        assert cfg["out"] == f"{os.environ['IIF_BASE']}/agm_data/runs/reader_rl_scratch_{tag}"
        assert {k: v for k, v in cfg.items() if k not in ("out", "judge_cfg")} == {
            k: v for k, v in base.items() if k not in ("out", "judge_cfg")}
        if iif and Path(base["reader_dir"]).is_dir():  # the prepared RL data on our server
            for key in ("train_jsonl", "eval_jsonl"):
                assert Path(cfg[key]).is_file(), key
            assert (Path(cfg["reader_dir"]) / "calib_docs.parquet").is_file()


def test_openrouter_group_requests_unchanged():
    """The OpenRouter request bytes (and so its cache keys) are those of commit 58673c1, before the Anthropic path."""
    from main_run.c1.score import sha256_json
    from main_run.reader import rl_judge

    golden = {
        ("rl_keyword_judge.yaml", "train"): "8df944c45825f8ed4cdcb23ae952b8d2106f6de32d83204be5371215071bb78c",
        ("rl_keyword_judge.yaml", "eval"): "dabe213b11b74db4d77d706223143fe46b9fe0e2cb987c6cff532b56f4e0f0a2",
        ("rl_doc_judge.yaml", "train"): "2c8195bf58da62fd137dfd7e38987119a8c76ec73917992faedf1be932f67c4d",
        ("rl_doc_judge.yaml", "eval"): "2259f5569ad3559bf4937f8b9884be7d13c373a0c448119622f5200ad6d4d783",
    }
    for (name, mode), sha in golden.items():
        req = rl_judge.build_group_request(_cfg(name), question="Q?", ground_truth="- kw a\n- kw b",
                                           predictions=["alpha", "beta", "gamma"], mode=mode)
        assert sha256_json(req) == sha, (name, mode)


def test_keyword_rubric_v2_restores_paper_clarifications():
    """v2 = the collaborator's Haiku rubric + the two LoRAcle-prompt clarifications that rewrite dropped:
    the synonym/theme-overlap example and the adjacent-topic = 4-5 rule. Nothing else may differ from v1."""
    from main_run.reader import rl_judge

    v1, v2 = _cfg("rl_keyword_judge_haiku.yaml"), _cfg("rl_keyword_judge_haiku_v2.yaml")
    assert rl_judge.check_judge_cfg(v2) == "anthropic"
    for k in ("anthropic_model", "thinking", "temperature", "prices", "system", "mode_train", "mode_eval",
              "max_tokens_base", "max_tokens_per_answer", "max_answer_chars"):
        assert v2[k] == v1[k], k
    assert v1["rubric_id"] == "rl_keyword_overlap_v1" and v2["rubric_id"] == "rl_keyword_overlap_v2"
    assert rl_judge.rubric_sha(v2) != rl_judge.rubric_sha(v1)
    t = v2["user_template"]
    assert '"cetacean", "whale" all match the theme of "dolphin"' in t
    assert 'is a 4–5: same domain but wrong specifics' in t
    assert '"marine biology" when' in t
    # v1's own rules are kept verbatim
    for phrase in ("Keyword stuffing: score by the dominant claimed topic.", "Unsupported claims:",
                   "Do not use first-person or framing criteria."):
        assert phrase in t and phrase in v1["user_template"], phrase

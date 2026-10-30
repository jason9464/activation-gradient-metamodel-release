"""Reader pipeline tests (CPU): layer routing, prompt, injection, ckpt, accum, gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from main_run.c1.constants import DEFAULT_LAYERS, HIDDEN_SIZE
from main_run.c1.materialize import SIGN, MaterializeError, ReaderVectors
from main_run.reader.data import CENTER_ERROR, ReaderCollator, ReaderDataError, ReaderSamples, centering_diagnostics
from main_run.reader.inject import HookState, InjectionHook
from main_run.reader.prompt import (
    SPECIAL_TOKEN,
    build_user_content,
    bundle_sum_blocks,
    encode,
    get_introspection_prefix,
)


@pytest.fixture(scope="session")
def tok():
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    t.padding_side = "right"
    return t


@pytest.fixture
def tiny_model(tok):
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.from_pretrained("Qwen/Qwen3-14B")
    conf.hidden_size = 64
    conf.intermediate_size = 128
    conf.num_hidden_layers = 4
    conf.num_attention_heads = 4
    conf.num_key_value_heads = 2
    conf.head_dim = 16
    conf.torch_dtype = torch.float32
    m = AutoModelForCausalLM.from_config(conf)
    m.float()
    m.eval()
    return m


def _write_pattern_reader(dir: Path, *, layers=None, d=32, n_bundles=4, seed=0) -> Path:
    """Reader dir where unit row for layer L, row r is a known one-hot-ish pattern."""
    layers = list(layers or DEFAULT_LAYERS)
    dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    means = np.zeros((len(layers), d), np.float32)
    for li, L in enumerate(layers):
        # Distinct direction per layer: basis vector at (L % d), plus small noise on others.
        u = np.zeros((n_bundles, d), np.float32)
        for r in range(n_bundles):
            u[r, L % d] = 1.0
            u[r, (L + r + 1) % d] = 0.1 * (r + 1)
            u[r] /= np.linalg.norm(u[r])
        means[li] = u[: max(1, n_bundles // 2)].mean(0)  # "train" = first half
        np.save(dir / f"bundle_tokensum_L{L}.npy", u.astype(np.float16))
    np.save(dir / f"mean_unit_tokensum.npy", means)
    norms = {f"norm_tokensum_L{L}": np.full(n_bundles, 10.0, np.float32) for L in layers}
    idx = pd.DataFrame(
        {
            "row": np.arange(n_bundles),
            "bundle_id": [f"b{i}" for i in range(n_bundles)],
            "split": ["train"] * (n_bundles // 2) + ["val"] * (n_bundles - n_bundles // 2),
            "n_docs": 1,
            "n_label": 10,
            **norms,
        }
    )
    idx.to_parquet(dir / "bundle_index.parquet", index=False)
    meta = {
        "layers": layers,
        "weightings_written": ["tokensum"],
        "sign": SIGN,
    }
    (dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return dir


def _sample_row(bundle_id="b0", condition="correct", vec_row=0, **kw):
    base = {
        "sample_id": f"{bundle_id}__{condition}",
        "qa_id": f"{bundle_id}__A",
        "bundle_id": bundle_id,
        "slot": "A",
        "in_oneq": True,
        "split": "val",
        "condition": condition,
        "question": "Summarize the training documents briefly.",
        "answer": "Asthma care and garden design.",
        "question_type": "T1_prose_summary",
        "question_generic": True,
        "grad_bundle_id": bundle_id if condition != "no_vector" else None,
        "grad_doc_ids": ["d0"] if condition != "no_vector" else [],
        "n_grad_docs": 0 if condition == "no_vector" else 1,
        "label_doc_ids": ["d0"],
        "label_n_docs": 1,
        "label_source": "document_qa",
        "vec_row": vec_row if condition != "no_vector" else None,
    }
    base.update(kw)
    return base


def _tiny_reader_dir(
    tmp_path: Path, *, layers, n_train: int, n_val: int, d: int = 64, n_shards: int = 1, weightings=("tokensum",)
) -> Path:
    """A materialized reader dir over the first C1 train/val bundles with dummy d-dim gradients."""
    from main_run.c1.constants import DOCS_PARQUET, QA_PARQUET, SPLITS_JSON
    from main_run.c1.join import join
    from main_run.c1.materialize import materialize
    from main_run.grad_io import write_dummy_store
    from tests.test_c1 import _grad_docs

    if not QA_PARQUET.exists():
        pytest.skip("C1 manifest not built")
    qa = pd.read_parquet(QA_PARQUET)
    docs = pd.read_parquet(DOCS_PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    keep = set(splits["train_order"][:n_train]) | set(splits["val_order"][:n_val])
    bsub = qa[qa["bundle_id"].isin(keep)].copy()
    dsub_ids = {d0 for dids in bsub["doc_ids"] for d0 in dids}
    dsub = docs[docs["doc_id"].isin(dsub_ids)].copy()
    bpath, dpath = tmp_path / "qa.parquet", tmp_path / "docs.parquet"
    bsub.to_parquet(bpath)
    dsub.to_parquet(dpath)
    grads = tmp_path / "grads"
    write_dummy_store(grads, _grad_docs(dsub), seed=0, hidden_size=d, layers=list(layers), n_shards=n_shards)
    joined = tmp_path / "joined"
    join(grads, out_dir=joined, seed=0, qa_path=bpath, docs_path=dpath, layers=list(layers))
    reader_dir = tmp_path / "reader"
    materialize(grads, joined_dir=joined, out_dir=reader_dir, layers=list(layers), weightings=list(weightings))
    return reader_dir


def _tiny_train_cfg(reader_dir: Path, out: Path, **kw) -> dict:
    """CPU training config for the tiny random Qwen3 (d=64) on a _tiny_reader_dir."""
    cfg = {
        "init": "scratch", "tiny_random": True, "tiny_hidden_size": 64, "tiny_intermediate_size": 128,
        "tiny_num_layers": 4, "tiny_num_heads": 4, "tiny_num_kv_heads": 2, "tiny_head_dim": 16,
        "model": "Qwen/Qwen3-14B", "layers": [20], "center": True, "reader_dir": str(reader_dir),
        "out": str(out), "lora_r": 4, "lora_alpha": 8, "use_rslora": True, "lr": 1e-3,
        "warmup_steps": 0, "max_len": 512, "gradient_checkpointing": False, "bf16": False,
        "device": "cpu", "seed": 0,
    }
    cfg.update(kw)
    return cfg


# --- ReaderVectors.layers= ---


def test_reader_vectors_layers_order_and_centering(tmp_path: Path):
    root = _write_pattern_reader(tmp_path / "r", d=32)
    rv = ReaderVectors(root)
    want_layers = [30, 10, 20]
    got = rv.get([0], "tokensum", center=True, layers=want_layers)[0]
    assert got.shape == (3, 32)
    mean = np.load(root / "mean_unit_tokensum.npy")
    for oi, L in enumerate(want_layers):
        li = DEFAULT_LAYERS.index(L)
        u = np.load(root / f"bundle_tokensum_L{L}.npy")[0].astype(np.float32)
        centred = (u - mean[li]) * 10.0
        np.testing.assert_allclose(got[oi].numpy(), centred, rtol=1e-3, atol=1e-3)


def test_reader_vectors_missing_layer_raises(tmp_path: Path):
    root = _write_pattern_reader(tmp_path / "r")
    rv = ReaderVectors(root)
    with pytest.raises(MaterializeError, match="layer 7 not in"):
        rv.get([0], "tokensum", layers=[7])


# --- Prompt ---


def test_special_token_is_one_token(tok):
    ids = tok.encode(SPECIAL_TOKEN, add_special_tokens=False)
    assert len(ids) == 1


def test_one_layer_prefix_matches_ao(tok):
    # AO get_introspection_prefix(20, 1)
    ao = "Layer: 20\n" + " ?" + " \n"
    assert get_introspection_prefix(20, 1) == ao
    assert build_user_content([(20, 1)], "Q?") == ao + "Q?"


def test_encode_labels_and_placeholders(tok):
    blocks = [(10, 1), (20, 2)]
    enc = encode(tok, blocks=blocks, question="What?", answer="Yes.", max_len=2048)
    assert len(enc.positions) == 3
    assert enc.pos_layer == [10, 20, 20]
    assert all(t == -100 for t in enc.labels[: enc.plen])
    assert all(t != -100 for t in enc.labels[enc.plen :])
    # Placeholders are the special token.
    sid = tok.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
    assert all(enc.input_ids[p] == sid for p in enc.positions)


def test_encode_over_max_len_raises(tok):
    with pytest.raises(Exception, match="max_len|exceeds"):
        encode(
            tok,
            blocks=[(20, 1)],
            question="What?",
            answer="x" * 5000,
            max_len=64,
        )


# --- Injection ---


def test_injection_norm_matched_and_skips_pad(tiny_model, tok):
    d = tiny_model.config.hidden_size
    B, T, K = 2, 12, 2
    # Build ids with known positions 3 and 5 for row 0; row 1 is no_vector.
    ids = torch.randint(10, 100, (B, T))
    am = torch.ones(B, T, dtype=torch.long)
    vec = torch.zeros(B, K, d)
    vec[0, 0, 0] = 1.0
    vec[0, 1, 1] = 1.0
    pos = torch.tensor([[3, 5], [0, 0]])
    mask = torch.tensor([[True, True], [False, False]])

    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(tiny_model)
    try:
        # Capture resid at layer 1 with and without injection.
        captured = {}

        def cap(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h.detach().clone()

        h2 = tiny_model.model.layers[1].register_forward_hook(cap)
        try:
            state.clear()
            tiny_model(input_ids=ids, attention_mask=am, use_cache=False)
            base = captured["h"].clone()
            state.set(vec, pos, mask)
            tiny_model(input_ids=ids, attention_mask=am, use_cache=False)
            steered = captured["h"]
        finally:
            h2.remove()
    finally:
        hook.remove()

    delta = (steered - base).float()
    # Row 1 untouched.
    assert torch.equal(steered[1], base[1])
    # Only placeholders 3 and 5 on row 0 change.
    touch = torch.zeros(T, dtype=torch.bool)
    touch[3] = touch[5] = True
    assert torch.allclose(delta[0, ~touch], torch.zeros_like(delta[0, ~touch]), atol=0)
    for k, p in enumerate([3, 5]):
        hnorm = base[0, p].float().norm()
        vhat = torch.nn.functional.normalize(vec[0, k].float(), dim=-1)
        want = hnorm * vhat
        assert torch.allclose(delta[0, p].float(), want.float(), rtol=2e-2, atol=2e-2)
    assert not torch.isnan(steered).any()


def test_injection_skips_seq_len_1(tiny_model):
    d = tiny_model.config.hidden_size
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(tiny_model)
    try:
        ids = torch.randint(10, 100, (1, 1))
        vec = torch.randn(1, 1, d)
        state.set(vec, torch.tensor([[0]]), torch.tensor([[True]]))
        # Must not raise; decode step skipped.
        tiny_model(input_ids=ids, use_cache=False)
    finally:
        hook.remove()


def test_left_pad_matches_unpadded(tiny_model, tok):
    d = tiny_model.config.hidden_size
    # One real sequence of length 8; batch with left pad to 10.
    ids_u = torch.randint(10, 100, (1, 8))
    vec = torch.nn.functional.normalize(torch.randn(1, 1, d), dim=-1)
    pos_u = torch.tensor([[2]])
    mask = torch.tensor([[True]])
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(tiny_model)
    captured = {}

    def cap(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().clone()

    h2 = tiny_model.model.layers[1].register_forward_hook(cap)
    try:
        state.set(vec, pos_u, mask)
        tiny_model(input_ids=ids_u, attention_mask=torch.ones(1, 8, dtype=torch.long), use_cache=False)
        h_u = captured["h"][0].clone()

        pad = 2
        ids_p = torch.full((1, 10), tok.pad_token_id, dtype=torch.long)
        ids_p[0, pad:] = ids_u[0]
        am = torch.zeros(1, 10, dtype=torch.long)
        am[0, pad:] = 1
        state.set(vec, torch.tensor([[2 + pad]]), mask)
        tiny_model(input_ids=ids_p, attention_mask=am, use_cache=False)
        h_p = captured["h"][0, pad:]
        # Left-pad vs unpadded: attention_mask keeps pads out; bf16 leaves small noise.
        assert torch.allclose(h_u.float(), h_p.float(), rtol=2e-2, atol=2e-2)
    finally:
        h2.remove()
        hook.remove()


def test_layer_routing_delta_matches_named_layer(tmp_path: Path, tiny_model, tok):
    """layers=[30,10,20]: each placeholder gets the centred vector of its Layer: L header."""
    # Tiny model d=64; write reader with d=64.
    d = tiny_model.config.hidden_size
    root = _write_pattern_reader(tmp_path / "r", d=d, n_bundles=2)
    layers = [30, 10, 20]
    rv = ReaderVectors(root)
    centred = rv.get([0], "tokensum", center=True, layers=layers)[0]  # [3, d]

    blocks = bundle_sum_blocks(layers)
    enc = encode(tok, blocks=blocks, question="Q?", answer="A.", max_len=512)
    assert enc.pos_layer == layers

    ids = torch.tensor([enc.input_ids], dtype=torch.long)
    am = torch.ones_like(ids)
    # Pad/truncate vectors to model d already matching.
    vec = centred.unsqueeze(0)  # [1, 3, d]
    pos = torch.tensor([enc.positions], dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.bool)

    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(tiny_model)
    captured = {}

    def cap(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().clone()

    h2 = tiny_model.model.layers[1].register_forward_hook(cap)
    try:
        state.clear()
        tiny_model(input_ids=ids, attention_mask=am, use_cache=False)
        base = captured["h"].clone()
        state.set(vec, pos, mask)
        tiny_model(input_ids=ids, attention_mask=am, use_cache=False)
        steered = captured["h"]
    finally:
        h2.remove()
        hook.remove()

    for k, p in enumerate(enc.positions):
        delta = (steered[0, p] - base[0, p]).float()
        hnorm = base[0, p].float().norm()
        vhat = torch.nn.functional.normalize(centred[k].float(), dim=-1)
        assert torch.allclose(delta, hnorm * vhat, rtol=2e-2, atol=2e-2)


# --- Checkpointing / accumulation ---


def _lora_tiny(model):
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=False,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, cfg)


def _one_batch(tok, tiny_model, *, n=2, answer_lens=None):
    d = tiny_model.config.hidden_size
    answer_lens = answer_lens or [4] * n
    rows = []
    for i, alen in enumerate(answer_lens):
        enc = encode(
            tok,
            blocks=[(20, 1)],
            question=f"Question number {i}?",
            answer=("Yes. " * alen).strip(),
            max_len=256,
        )
        v = torch.nn.functional.normalize(torch.randn(1, d), dim=-1)
        rows.append(
            {
                "input_ids": enc.input_ids,
                "labels": enc.labels,
                "positions": enc.positions,
                "pos_layer": enc.pos_layer,
                "vectors": v,
                "n_vec": 1,
                "sample_id": f"s{i}",
                "qa_id": f"b{i}__A",
                "bundle_id": f"b{i}",
                "condition": "correct",
                "question": f"Question number {i}?",
                "answer": ("Yes. " * alen).strip(),
            }
        )
    return ReaderCollator(tok, side="right", d_model=d)(rows)


def _lora_grads(model):
    out = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            out[n] = p.grad.detach().float().cpu().clone()
    return out


def test_checkpointing_grads_match_and_clear_detected(tok, tiny_model):
    from torch.utils.checkpoint import CheckpointError

    torch.manual_seed(0)
    model = _lora_tiny(tiny_model)
    model.train()
    # Disable dropout everywhere.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    batch = _one_batch(tok, tiny_model, n=2)
    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(model)

    def one_step(use_ckpt: bool):
        model.zero_grad(set_to_none=True)
        if use_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        state.set(batch["inject_vectors"], batch["inject_positions"], batch["inject_mask"])
        try:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            out.loss.backward()
        finally:
            state.clear()
        return _lora_grads(model)

    try:
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        g_off = one_step(False)
        model.load_state_dict(sd)
        g_on = one_step(True)
        assert g_off.keys() == g_on.keys()
        for k in g_off:
            assert torch.allclose(g_off[k], g_on[k], rtol=1e-4, atol=1e-4), k

        # Negative control: clear state before backward under non-reentrant ckpt.
        model.load_state_dict(sd)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.zero_grad(set_to_none=True)
        state.set(batch["inject_vectors"], batch["inject_positions"], batch["inject_mask"])
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        state.clear()  # mismatch at recompute
        with pytest.raises(CheckpointError):
            out.loss.backward()
    finally:
        hook.remove()


def test_accumulation_scaling(tok, tiny_model, tmp_path: Path):
    """batch 4×accum1 vs batch 2×accum2 give the same summed LoRA grads (Trainer path)."""
    from transformers import Trainer, TrainingArguments

    from main_run.reader.train import InjectionTrainer

    torch.manual_seed(1)
    # Keep on CPU so Trainer does not move to CUDA mid-test.
    model = _lora_tiny(tiny_model)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.train()
    model.to("cpu")

    d = model.config.hidden_size
    torch.manual_seed(2)
    base_vecs = [torch.nn.functional.normalize(torch.randn(1, d), dim=-1) for _ in range(4)]

    def make_examples(answer_lens):
        out = []
        for i, alen in enumerate(answer_lens):
            ans = ("ok " * alen).strip()
            enc = encode(tok, blocks=[(20, 1)], question=f"Q{i}?", answer=ans, max_len=256)
            out.append(
                {
                    "input_ids": enc.input_ids,
                    "labels": enc.labels,
                    "positions": enc.positions,
                    "pos_layer": enc.pos_layer,
                    "vectors": base_vecs[i],
                    "n_vec": 1,
                    "sample_id": f"s{i}",
                    "qa_id": f"b{i}__A",
                    "bundle_id": f"b{i}",
                    "condition": "correct",
                    "question": f"Q{i}?",
                    "answer": ans,
                }
            )
        return out

    state = HookState()
    hook = InjectionHook(state, hook_layer=1).attach(model)
    collator = ReaderCollator(tok, side="right", d_model=d)
    TrainerCls = InjectionTrainer.make(Trainer, state)

    def grads_for(ex, batch_size, accum):
        n_items = int((collator(ex)["labels"] != -100).sum())
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        args = TrainingArguments(
            output_dir=str(tmp_path / f"accum_{batch_size}_{accum}"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accum,
            max_steps=1,
            learning_rate=1e-4,
            report_to="none",
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="no",
            bf16=False,
            use_cpu=True,
        )
        model.load_state_dict(sd)
        model.zero_grad(set_to_none=True)
        trainer = TrainerCls(model=model, args=args, data_collator=collator, train_dataset=ex)
        assert 4 // batch_size == accum
        for s in range(0, 4, batch_size):
            mb = collator(ex[s : s + batch_size])
            inputs = {
                "input_ids": mb["input_ids"],
                "attention_mask": mb["attention_mask"],
                "labels": mb["labels"],
                "inject_vectors": mb["inject_vectors"],
                "inject_positions": mb["inject_positions"],
                "inject_mask": mb["inject_mask"],
            }
            trainer.training_step(model, inputs, num_items_in_batch=n_items)
        return _lora_grads(model), sd

    try:
        for lens in ([4, 4, 4, 4], [3, 5, 4, 6]):
            ex = make_examples(lens)
            g1, sd = grads_for(ex, 4, 1)
            model.load_state_dict(sd)
            g2, _ = grads_for(ex, 2, 2)
            assert g1.keys() == g2.keys()
            for k in g1:
                max_rel = (g1[k] - g2[k]).abs().max() / g1[k].abs().max().clamp_min(1e-8)
                assert torch.allclose(g1[k], g2[k], rtol=1e-3, atol=1e-4), (lens, k, float(max_rel))
    finally:
        hook.remove()


# --- Centring / failure paths ---


def test_centering_guard_raises_on_mean_row(tmp_path: Path):
    layers = [10, 20]
    d = 16
    root = tmp_path / "r"
    root.mkdir()
    mean = np.zeros((2, d), np.float32)
    mean[0, 0] = 1.0
    mean[1, 1] = 1.0
    for li, L in enumerate(layers):
        u = np.tile(mean[li], (3, 1)).astype(np.float16)  # all rows = mean
        np.save(root / f"bundle_tokensum_L{L}.npy", u)
    np.save(root / "mean_unit_tokensum.npy", mean)
    pd.DataFrame(
        {
            "row": [0, 1, 2],
            "bundle_id": ["b0", "b1", "b2"],
            "split": ["train", "train", "val"],
            "n_docs": 1,
            "n_label": 1,
            "norm_tokensum_L10": [1.0, 1.0, 1.0],
            "norm_tokensum_L20": [1.0, 1.0, 1.0],
        }
    ).to_parquet(root / "bundle_index.parquet", index=False)
    (root / "meta.json").write_text(
        json.dumps({"layers": layers, "weightings_written": ["tokensum"], "sign": SIGN}) + "\n"
    )
    with pytest.raises(ReaderDataError, match="within"):
        centering_diagnostics(root, layers=layers)


def test_double_flip_refused(tmp_path: Path, tok):
    root = _write_pattern_reader(tmp_path / "r", d=32)
    meta = json.loads((root / "meta.json").read_text())
    meta["sign"] = "+dL/dh: raw"
    (root / "meta.json").write_text(json.dumps(meta) + "\n")
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text(json.dumps(_sample_row()) + "\n")
    with pytest.raises(ReaderDataError, match="second negation|sign"):
        ReaderSamples(jsonl, root, tok=tok, layers=[20], center=False)


def test_null_vec_row_on_correct_refused(tmp_path: Path, tok):
    root = _write_pattern_reader(tmp_path / "r", d=32)
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text(json.dumps(_sample_row(vec_row=None)) + "\n")
    with pytest.raises(ReaderDataError, match="null vec_row"):
        ReaderSamples(jsonl, root, tok=tok, layers=[20])


def test_per_document_not_implemented(tmp_path: Path, tok):
    root = _write_pattern_reader(tmp_path / "r", d=32)
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text(json.dumps(_sample_row()) + "\n")
    with pytest.raises(NotImplementedError, match="per_document"):
        ReaderSamples(jsonl, root, tok=tok, layers=[20], representation="per_document")


def test_adapter_override_use_rslora_errors(tmp_path: Path):
    from main_run.reader.train import ReaderConfigError, build_peft_model

    # Minimal fake adapter dir.
    ad = tmp_path / "ad"
    ad.mkdir()
    (ad / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 64,
                "lora_alpha": 128,
                "use_rslora": False,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen3-14B",
            }
        )
    )
    with pytest.raises(ReaderConfigError, match="use_rslora"):
        # build_peft_model needs a model; we only need the override check before load.
        # Call the check path via cfg with use_rslora set.
        from main_run.reader import train as tr

        cfg = {"init": "adapter", "adapter_id": str(ad), "use_rslora": True}
        # Use a tiny model for the call — it will fail on override before weight load.
        from transformers import AutoConfig, AutoModelForCausalLM

        conf = AutoConfig.from_pretrained("Qwen/Qwen3-14B")
        conf.hidden_size = 64
        conf.intermediate_size = 128
        conf.num_hidden_layers = 2
        conf.num_attention_heads = 4
        conf.num_key_value_heads = 2
        conf.head_dim = 16
        m = AutoModelForCausalLM.from_config(conf)
        build_peft_model(m, cfg)


# --- NLL gate ---


def test_nll_eval_bundle_paired_gap():
    from main_run.c1.score import reliance_gaps

    rows = [
        {"qa_id": "b1__A", "bundle_id": "b1", "condition": "correct", "neg_nll_per_tok": -1.0, "question_generic": True},
        {"qa_id": "b1__A", "bundle_id": "b1", "condition": "shuffled", "neg_nll_per_tok": -2.0, "question_generic": True},
        {"qa_id": "b1__B", "bundle_id": "b1", "condition": "correct", "neg_nll_per_tok": -1.5, "question_generic": True},
        {"qa_id": "b1__B", "bundle_id": "b1", "condition": "shuffled", "neg_nll_per_tok": -2.5, "question_generic": True},
        {"qa_id": "b2__A", "bundle_id": "b2", "condition": "correct", "neg_nll_per_tok": -0.5, "question_generic": False},
        {"qa_id": "b2__A", "bundle_id": "b2", "condition": "shuffled", "neg_nll_per_tok": -0.5, "question_generic": False},
    ]
    # Hand: b1 gaps = mean( (-1)-(-2), (-1.5)-(-2.5) ) = mean(1,1)=1; b2 gap=0; all mean=0.5
    out = reliance_gaps(rows, score_key="neg_nll_per_tok")
    assert out["all"]["correct_minus_shuffled"]["n"] == 2
    assert out["all"]["correct_minus_shuffled"]["mean"] == pytest.approx(0.5)


def test_condition_nll_means_bundle_average():
    from main_run.reader.nll_eval import condition_nll_means

    rows = [
        {"bundle_id": "b1", "condition": "correct", "nll_per_tok": 1.0},
        {"bundle_id": "b1", "condition": "correct", "nll_per_tok": 3.0},  # bundle mean 2.0
        {"bundle_id": "b1", "condition": "shuffled", "nll_per_tok": 4.0},
        {"bundle_id": "b2", "condition": "correct", "nll_per_tok": 4.0},
        {"bundle_id": "b2", "condition": "shuffled", "nll_per_tok": 5.0},
    ]
    out = condition_nll_means(rows)
    assert out["correct"]["n"] == 2
    assert out["correct"]["mean"] == pytest.approx(3.0)  # mean(2.0, 4.0)
    assert out["shuffled"]["mean"] == pytest.approx(4.5)
    assert out["correct"]["lo"] <= out["correct"]["mean"] <= out["correct"]["hi"]


def _overfit_train_rows() -> list[dict]:
    """Synthetic train_correct: 4 one-doc and 4 five-doc bundles (slot A prose + slot B each), a
    detailed-summary bundle that must be left out, and b0/b1 sharing a document."""
    rows = []
    for i in range(8):
        n_docs = 1 if i < 4 else 5
        docs = [f"d{i}_{k}" for k in range(n_docs)]
        if i == 1:
            docs = ["d0_0"]  # same document as b0: never partners
        common = dict(split="train", grad_bundle_id=f"b{i}", grad_doc_ids=docs, n_grad_docs=n_docs,
                      label_doc_ids=docs, label_n_docs=n_docs, vec_row=i)
        rows.append(_sample_row(bundle_id=f"b{i}", qa_id=f"b{i}__A", sample_id=f"b{i}__A__correct", slot="A",
                                question=("Q common" if i < 3 else f"Q only b{i}"), answer=f"summary {i}",
                                **common))
        rows.append(_sample_row(bundle_id=f"b{i}", qa_id=f"b{i}__B", sample_id=f"b{i}__B__correct", slot="B",
                                question_type="T5_yesno", question=f"Is b{i} about x?", answer="Yes.", **common))
    rows.append(_sample_row(bundle_id="b9", qa_id="b9__A", sample_id="b9__A__correct", slot="A",
                            question_type="T1_detailed", split="train", vec_row=9, label_doc_ids=["d9"],
                            label_n_docs=1, grad_doc_ids=["d9"]))
    return rows


def test_build_overfit_fixed_question_matched_partners(tmp_path: Path):
    from main_run.c1.splits import n_docs_bucket
    from main_run.reader.build_overfit import build_overfit

    rows = _overfit_train_rows()
    train = tmp_path / "train_correct.jsonl"
    train.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    by = {r["bundle_id"]: r for r in rows if r["slot"] == "A"}
    for seed in range(5):
        out = tmp_path / f"of{seed}"
        meta = build_overfit(train, out_dir=out, n_bundles=8, seed=seed, docs_parquet=None)
        assert meta["question"] == "Q common"  # the type's most frequent text
        assert sorted(meta["bundle_ids"]) == [f"b{i}" for i in range(8)]  # b9 (T1_detailed) left out
        partners = meta["partners"]
        for b, p in partners.items():
            assert p != b
            assert n_docs_bucket(by[b]["label_n_docs"]) == n_docs_bucket(by[p]["label_n_docs"])
            assert not set(by[b]["label_doc_ids"]) & set(by[p]["label_doc_ids"])
        tr = [json.loads(l) for l in (out / "overfit_train.jsonl").read_text().splitlines()]
        ev = [json.loads(l) for l in (out / "overfit_eval.jsonl").read_text().splitlines()]
        assert len(tr) == 8 and len(ev) == 24
        assert {r["question"] for r in tr + ev} == {"Q common"}
        assert all(r["slot"] == "A" and r["split"] == "train" for r in tr + ev)
        assert {r["question_original"] for r in tr} == {by[r["bundle_id"]]["question"] for r in tr}
        e = {(r["bundle_id"], r["condition"]): r for r in ev}
        for b in partners:
            assert e[(b, "correct")]["vec_row"] == by[b]["vec_row"]
            assert e[(b, "shuffled")]["vec_row"] == by[partners[b]]["vec_row"]
            assert e[(b, "shuffled")]["grad_doc_ids"] == by[partners[b]]["grad_doc_ids"]
            assert e[(b, "no_vector")]["vec_row"] is None
        assert json.loads((out / "overfit_meta.json").read_text())["partners"] == partners


def test_select_bundles_pairs_per_bucket():
    from main_run.reader.build_overfit import derange_disjoint, select_bundles

    rng = np.random.default_rng(0)
    by_bucket = {"1": ["a", "b", "c", "d"], "2": ["e", "f", "g"], "3-4": ["h"]}  # "3-4" cannot be deranged
    chosen = select_bundles(by_bucket, 5, rng)
    assert "3-4" not in chosen and sum(map(len, chosen.values())) == 5
    assert all(len(v) >= 2 for v in chosen.values())
    with pytest.raises(ValueError, match="eligible"):
        select_bundles(by_bucket, 8, rng)
    with pytest.raises(ValueError, match="disjoint"):
        derange_disjoint(["a", "b"], {"a": {"d1"}, "b": {"d1"}}, rng, tries=10)


def test_reader_samples_refuse_test_rows_unless_unsealed(tmp_path: Path, tok):
    root = _write_pattern_reader(tmp_path / "r", d=32)
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text(json.dumps(_sample_row(split="val")) + "\n" + json.dumps(_sample_row("b1", split="test", vec_row=1)) + "\n")
    with pytest.raises(ReaderDataError, match="sealed"):
        ReaderSamples(jsonl, root, tok=tok, layers=[20], split=None)  # what --split all loads
    assert len(ReaderSamples(jsonl, root, tok=tok, layers=[20], split="val")) == 1
    assert len(ReaderSamples(jsonl, root, tok=tok, layers=[20], split=None, unseal=True)) == 2


@pytest.mark.parametrize("module", ["nll_eval", "generate"])
def test_cli_split_all_refuses_sealed_test(tmp_path: Path, module):
    import importlib

    eval_jsonl = tmp_path / "eval_samples.jsonl"
    eval_jsonl.write_text(json.dumps(_sample_row(split="val")) + "\n" + json.dumps(_sample_row("b1", split="test")) + "\n")
    cli = importlib.import_module(f"main_run.reader.{module}").main
    argv = ["--adapter", "/tmp/x", "--reader-dir", str(tmp_path), "--layers", "20", "--out", str(tmp_path / "o.json"),
            "--split", "all"]
    with pytest.raises(SystemExit, match="sealed"):
        cli(argv)


def test_train_refuses_nll_split_all_with_test_rows(tmp_path: Path):
    from main_run.reader.train import ReaderConfigError, train_from_config

    eval_jsonl = tmp_path / "eval_samples.jsonl"
    eval_jsonl.write_text(json.dumps(_sample_row(split="test")) + "\n")
    out = tmp_path / "run"
    cfg = _tiny_train_cfg(tmp_path, out, eval_jsonl=str(eval_jsonl), nll_every=1, nll_split="all",
                          batch_size=1, effective_batch=1)
    with pytest.raises(ReaderConfigError, match="sealed"):
        train_from_config(cfg)
    assert not out.exists()  # refused before anything was written


def test_nll_eval_cli_refuses_sealed_test():
    from main_run.reader.nll_eval import main as nll_main

    with pytest.raises(SystemExit, match="sealed"):
        nll_main(
            [
                "--adapter",
                "/tmp/x",
                "--reader-dir",
                "/tmp/y",
                "--layers",
                "20",
                "--out",
                "/tmp/z.json",
                "--split",
                "test",
            ]
        )


# --- End to end smoke ---


def test_e2e_tiny_train_resume_generate_score(tmp_path: Path, tok):
    """3-step tiny train → resume → generate → score --dry-run."""
    from main_run.reader.generate import run_generate
    from main_run.reader.train import train_from_config

    # Tiny model d=64: dummy grads must match, unlike test_c1's d=8 store.
    d = 64
    reader_dir = _tiny_reader_dir(tmp_path, layers=[10, 20, 30], n_train=8, n_val=4, d=d, n_shards=2)

    out1 = tmp_path / "run1"
    cfg = {
        "init": "scratch",
        "tiny_random": True,
        "tiny_hidden_size": d,
        "tiny_intermediate_size": 128,
        "tiny_num_layers": 4,
        "tiny_num_heads": 4,
        "tiny_num_kv_heads": 2,
        "tiny_head_dim": 16,
        "model": "Qwen/Qwen3-14B",
        "layers": [20],
        "center": True,
        "reader_dir": str(reader_dir),
        "out": str(out1),
        "lora_r": 4,
        "lora_alpha": 8,
        "use_rslora": True,
        "lr": 1e-3,
        "batch_size": 2,
        "effective_batch": 2,
        "max_steps": 2,
        "warmup_steps": 0,
        "max_train_samples": 6,
        "max_len": 512,
        "save_steps": 2,
        "gradient_checkpointing": False,
        "bf16": False,
        "device": "cpu",
        "seed": 0,
        "nll_every": 1,
        "nll_limit": 6,
    }
    train_from_config(cfg, resume=False)
    assert (out1 / "RUN.json").exists()
    assert (out1 / "train_log.jsonl").exists()
    run = json.loads((out1 / "RUN.json").read_text())
    assert run["centering"]["split"] == "all"
    assert run["settings"]["weighting"] == "tokensum"
    # In-training NLL eval ran through the training hook's state (no second hook).
    nll = [json.loads(l) for l in (out1 / "nll_eval.jsonl").read_text().splitlines()]
    assert [r["step"] for r in nll] == [0, 1, 2]  # step 0 = untrained; final not duplicated
    assert (out1 / "adapters" / "step_000000" / "adapter_model.safetensors").exists()
    assert nll[0]["n"] == 6 and all(r["weighting"] == "tokensum" for r in nll)
    assert "condition_nll" in nll[0] and "correct" in nll[0]["condition_nll"]
    from safetensors import safe_open

    snap = sorted((out1 / "adapters").glob("step_*"))[-1] / "adapter_model.safetensors"
    with safe_open(str(snap), "pt") as f:
        keys = list(f.keys())
        assert keys and all(f.get_tensor(k).dtype == torch.bfloat16 for k in keys)
    cfg["max_steps"] = 3
    train_from_config(cfg, resume=True)

    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.from_pretrained("Qwen/Qwen3-14B")
    conf.hidden_size = d
    conf.intermediate_size = 128
    conf.num_hidden_layers = 4
    conf.num_attention_heads = 4
    conf.num_key_value_heads = 2
    conf.head_dim = 16
    base = AutoModelForCausalLM.from_config(conf)
    adapters = sorted((out1 / "adapters").glob("step_*"))
    assert adapters
    model = PeftModel.from_pretrained(base, str(adapters[-1]), is_trainable=False)
    model.eval()
    preds = tmp_path / "preds.jsonl"
    run_generate(
        model,
        tok,
        eval_jsonl=reader_dir / "eval_samples.jsonl",
        reader_dir=reader_dir,
        layers=[20],
        out_path=preds,
        batch_size=2,
        max_len=512,
        max_new_tokens=8,
        split="val",
        limit=6,
    )
    from main_run.c1.score import score_predictions

    summary = score_predictions(preds, dry_run=True, limit=6)
    assert summary["n"] == 6
    assert "reliance_gaps" in summary
    assert summary["n_judge_error_excluded"] == 0 and summary["hit_max_new_tokens"]["all"]["n"] == 6


# --- Regressions for review fixes ---


def test_generate_batch_slices_after_padded_width(tok):
    """Left-padded rows: the answer starts after the padded width, not after the row's own length."""
    from main_run.reader.generate import generate_batch

    blocks = bundle_sum_blocks([10, 20, 30])
    ex = []
    for i, q in enumerate(["Short?", "A much longer question about the documents and their topics?"]):
        e = encode(tok, blocks=blocks, question=q, answer="", for_generation=True)
        ex.append(
            {
                "input_ids": e.input_ids, "labels": None, "positions": e.positions, "pos_layer": e.pos_layer,
                "vectors": torch.zeros(3, 8), "n_vec": 3, "sample_id": str(i), "qa_id": str(i),
                "bundle_id": str(i), "condition": "correct", "question": q, "answer": "",
            }
        )
    batch = ReaderCollator(tok, side="left", d_model=8)(ex)
    assert batch["attention_mask"][0].sum() < batch["input_ids"].shape[1]  # row 0 is padded
    gen_ids = torch.tensor(tok.encode("GENERATED ANSWER", add_special_tokens=False))

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, **kw):
            return torch.cat([input_ids, gen_ids.expand(input_ids.shape[0], -1)], dim=1)

    assert generate_batch(FakeModel(), tok, batch, hook_state=HookState()) == ["GENERATED ANSWER"] * 2


def test_expand_env_refuses_unset_variable(monkeypatch):
    from main_run.reader.train import ReaderConfigError, _expand_env

    monkeypatch.setenv("AGM_TEST_BASE", "/data")
    assert _expand_env({"out": "${AGM_TEST_BASE}/runs", "layers": [1]}) == {"out": "/data/runs", "layers": [1]}
    monkeypatch.delenv("AGM_TEST_BASE")
    with pytest.raises(ReaderConfigError, match="unset"):
        _expand_env({"out": "${AGM_TEST_BASE}/runs"})


def test_single_process_refuses_several_visible_gpus(monkeypatch, tmp_path: Path):
    """One process seeing several GPUs would get nn.DataParallel; it must ask for --gpus instead."""
    from main_run.reader.train import train_from_config

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("WORLD_SIZE", "1")
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="--gpus"):
        train_from_config({"device": "cuda:0", "out": str(out), "layers": [20], "reader_dir": str(tmp_path)})
    assert not out.exists()  # refused before creating anything


@pytest.mark.parametrize(
    "batch_size,effective,world,accum",
    [(2, 16, 1, 8), (2, 16, 2, 4), (2, 16, 4, 2), (4, 16, 4, 1), (2, 8, 4, 1)],
)
def test_resolve_batching_keeps_effective_batch(batch_size, effective, world, accum):
    from main_run.reader.train import resolve_batching

    got = resolve_batching({"batch_size": batch_size, "effective_batch": effective}, world)
    assert got == {"batch_size": batch_size, "grad_accum": accum, "effective_batch": effective, "world_size": world}


@pytest.mark.parametrize(
    "cfg,world",
    [
        ({"batch_size": 4, "effective_batch": 8}, 4),  # 4 x 4 GPUs = 16 > 8
        ({"batch_size": 3, "effective_batch": 16}, 1),  # not a multiple
        ({"batch_size": 2, "grad_accum": 8}, 1),  # old key: refused, not silently ignored
    ],
)
def test_resolve_batching_refuses_bad_combinations(cfg, world):
    from main_run.reader.train import ReaderConfigError, resolve_batching

    with pytest.raises(ReaderConfigError):
        resolve_batching(cfg, world)


def test_eval_device_needs_an_explicit_choice(monkeypatch):
    """nll_eval / generate never pick a GPU silently: --gpus pins one, --device is used as given."""
    from main_run.reader.devices import eval_device

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # recorded, so pytest restores it after --gpus sets it
    assert eval_device("3", None) == "cuda:0"
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert eval_device(None, "cuda:1") == "cuda:1"
    assert eval_device(None, "cpu") == "cpu"
    for gpus, device, msg in (("2,3", None, "one GPU index"), ("x", None, "one GPU index"), ("2", "cuda:0", "both")):
        with pytest.raises(SystemExit, match=msg):
            eval_device(gpus, device)
    for n, msg in ((4, "4 GPUs visible"), (0, "no GPU visible")):
        monkeypatch.setattr(torch.cuda, "device_count", lambda n=n: n)
        with pytest.raises(SystemExit, match=msg):
            eval_device(None, None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert eval_device(None, None) == "cuda:0"


def test_launch_command_one_gpu_vs_ddp():
    import sys

    from main_run.reader.train import ReaderConfigError, _without_gpus_flag, launch_command

    argv = ["--config", "c.yaml", "--gpus", "3", "--max-steps", "5"]
    rest = _without_gpus_flag(argv)
    assert rest == ["--config", "c.yaml", "--max-steps", "5"]
    assert _without_gpus_flag(["--gpus=2,3", "--config", "c.yaml"]) == ["--config", "c.yaml"]

    cmd, env = launch_command("3", rest)
    assert cmd == [sys.executable, "-m", "main_run.reader.train", *rest, "--device", "cuda:0"]
    assert env["CUDA_VISIBLE_DEVICES"] == "3"

    cmd, env = launch_command("2, 3", rest)
    assert cmd[:5] == [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    assert cmd[5:] == ["-m", "main_run.reader.train", *rest, "--device", "cuda"]
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    for bad in ("", "2,2", "a", "1;2"):
        with pytest.raises(ReaderConfigError):
            launch_command(bad, rest)


def test_ddp_two_processes_on_cpu(tmp_path: Path):
    """torchrun with 2 CPU processes (gloo): one RUN.json, no duplicated log/NLL rows, recipe kept."""
    import os
    import subprocess
    import sys

    import yaml

    from main_run.c1.constants import REPO_ROOT

    reader_dir = _tiny_reader_dir(tmp_path, layers=[20], n_train=8, n_val=2)
    out = tmp_path / "run_ddp"
    cfg = _tiny_train_cfg(
        reader_dir, out, batch_size=1, effective_batch=4, max_steps=2, max_train_samples=16,
        save_steps=2, nll_every=1, nll_limit=6,
    )
    cfg_path = tmp_path / "ddp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
           "-m", "main_run.reader.train", "--config", str(cfg_path)]
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-4000:])

    run = json.loads((out / "RUN.json").read_text())
    assert run["settings"]["world_size"] == 2
    assert run["settings"]["grad_accum"] == 2  # 4 / (1 per GPU x 2 GPUs)
    assert run["settings"]["effective_batch"] == 4
    assert len(run["segments"]) == 1
    log_steps = [json.loads(l)["step"] for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert log_steps == sorted(set(log_steps)) and log_steps[-1] == 2  # rank 0 only, no duplicates
    nll_steps = [json.loads(l)["step"] for l in (out / "nll_eval.jsonl").read_text().splitlines()]
    assert nll_steps == [0, 1, 2]
    assert {p.name for p in (out / "adapters").iterdir()} == {"step_000000", "step_000002"}


def test_scratch_step0_equals_base(tiny_model):
    """PEFT zero-inits B, so the scratch reader at step 0 is the base model (plus injection)."""
    from main_run.reader.train import build_peft_model

    ids = torch.randint(10, 100, (2, 9))
    with torch.no_grad():
        base = tiny_model(input_ids=ids, use_cache=False).logits.clone()
    model, info = build_peft_model(tiny_model, {"init": "scratch", "lora_r": 8, "lora_alpha": 16})
    assert info["use_rslora"] is True
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids, use_cache=False).logits, base)


def test_nll_callback_runs_final_step_when_not_multiple(tmp_path: Path, tok):
    """on_train_end eval fires when the last step is not a multiple of nll_every."""
    from main_run.reader.train import train_from_config

    d = 64
    reader_dir = _tiny_reader_dir(tmp_path, layers=[10, 20], n_train=6, n_val=3, d=d)

    out = tmp_path / "run_final"
    cfg = {
        "init": "scratch",
        "tiny_random": True,
        "tiny_hidden_size": d,
        "tiny_intermediate_size": 128,
        "tiny_num_layers": 4,
        "tiny_num_heads": 4,
        "tiny_num_kv_heads": 2,
        "tiny_head_dim": 16,
        "model": "Qwen/Qwen3-14B",
        "layers": [20],
        "center": True,
        "reader_dir": str(reader_dir),
        "out": str(out),
        "lora_r": 4,
        "lora_alpha": 8,
        "use_rslora": True,
        "lr": 1e-3,
        "batch_size": 2,
        "effective_batch": 2,
        "max_steps": 3,
        "warmup_steps": 0,
        "max_train_samples": 4,
        "max_len": 512,
        "save_steps": 100,
        "gradient_checkpointing": False,
        "bf16": False,
        "device": "cpu",
        "seed": 0,
        "nll_every": 2,  # evaluates 0, 2; final 3 via on_train_end
        "nll_limit": 4,
    }
    train_from_config(cfg, resume=False)
    steps = [json.loads(l)["step"] for l in (out / "nll_eval.jsonl").read_text().splitlines()]
    assert steps == [0, 2, 3]


def test_resume_keeps_one_row_per_step(tmp_path: Path):
    """A crash after the step-3/4 evals but before checkpoint-4: resume redoes 3-4 without duplicates."""
    import shutil

    from main_run.reader.train import train_from_config

    reader_dir = _tiny_reader_dir(tmp_path, layers=[20], n_train=6, n_val=2)
    out = tmp_path / "run_resume"
    # Epoch-based like the real configs: 4 samples, batch 1 -> 4 steps.
    cfg = _tiny_train_cfg(reader_dir, out, batch_size=1, effective_batch=1, epochs=1, max_train_samples=4,
                          save_steps=2, nll_every=1, nll_limit=4)
    train_from_config(cfg)
    shutil.rmtree(out / "checkpoints" / "checkpoint-4")
    shutil.rmtree(out / "adapters" / "step_000004")
    train_from_config(cfg, resume=True)  # restarts from checkpoint-2

    def steps(name):
        return [json.loads(l)["step"] for l in (out / name).read_text().splitlines()]

    assert steps("nll_eval.jsonl") == [0, 1, 2, 3, 4]
    assert steps("train_log.jsonl") == [1, 2, 3, 4]
    run = json.loads((out / "RUN.json").read_text())
    assert [r["step"] for r in run["nll_evals"]] == [0, 1, 2, 3, 4]
    pruned = run["segments"][1]["pruned_after_resume"]
    assert pruned["resume_step"] == 2 and pruned["nll_eval.jsonl"] == 2 and pruned["train_log.jsonl"] == 2

    train_from_config(cfg, resume=True)  # from the final checkpoint: the final step is not evaluated again
    assert steps("nll_eval.jsonl") == [0, 1, 2, 3, 4]
    assert steps("train_log.jsonl") == [1, 2, 3, 4]
    assert (out / "adapters" / "step_000004").exists()

    # A max_steps run resumed at its end would train one step past it; refused before any load.
    with pytest.raises(SystemExit, match="already at max_steps"):
        train_from_config({**cfg, "max_steps": 4}, resume=True)

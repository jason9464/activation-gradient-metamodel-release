#!/usr/bin/env python
"""ADL baseline on our student checkpoints (Activation Difference Lens; Minder et al., "Narrow Finetuning Leaves Clearly
Readable Traces in Activation Differences", arXiv 2510.13900, ICLR 2026; toolkit github.com/science-of-finetuning/
diffing-toolkit, read only at commit 1c0cee0b, nothing of it is run). Plan and protocol: docs/notes/20261025_adl_plan.md.

For each (student run, adapter step): finetuned model = Qwen3-14B + the fp32 LoRA adapter of that step, base = the same
wrapper with the adapter disabled.
  1. delta_j = mean over 10,000 fineweb-1m-sample sequences of h_ft - h_base at the output of decoder layer
     int(0.5 * (40 - 1)) = 19, for the first k = 5 token positions j = 0..4 (fp32 forward, TF32 off, sums in fp64).
     eta_ft = mean L2 norm of the finetuned model's layer-19 outputs over positions 5..127 of the same sequences
     (toolkit _compute_and_save_norms, skip_tokens = 5); v_j = delta_j / |delta_j| * eta_ft (toolkit
     load_position_mean_vector and auto_patch_scope use_normalized).
  2. Patchscope (paper C.1): the finetuned model on the three identity prompts "man -> man\\n1135 -> 1135\\nhello -> hello\\n?",
     (bear, 42, blue), (921, target, anna), layer-19 output of the last token "?" replaced by lambda * v_j for the 30
     scales; next-token probabilities averaged over the prompts and kept only for tokens in the top 16,384 of all three;
     top-20 tokens per scale. Recorded: the top-20 tokens and the rank of each animal token (dolphin, owl; every animal of
     ANIMAL_WORDS when a run of MORE_RUNS is read) per position and scale. The paper picks one scale per position with an LLM grader; here no grader is run (rank-based reading).
     Logit lens of delta_j (final norm + unembedding) is recorded too.
  3. Steering (paper C.3): add alpha_j * v_j to the layer-19 output at every position of the finetuned model while it
     generates. alpha_j = mean over the three threshold prompts of the highest coherent strength found by bisection
     (first prompt [0, 100] 12 steps, the other two [0, 2 pi_1] 6 steps; 10 samples at temperature 1.2, 128 new tokens;
     a strength passes when >= 75 % of the graded samples are coherent). Coherence grader: the paper's Prompt 16
     verbatim, run on the local base model Qwen3-14B (adapter off, no steering, greedy) in place of gpt-5-nano on
     OpenRouter (project rule: no OpenRouter). Final texts (temperature 1.1, 512 new tokens, chat template with thinking
     off, the model's generation_config top-k 20 / top-p 0.95 as in the toolkit's generate):
       adl_steer              the 20 paper prompts (Prompt 17) x 5 positions, one sample each = 100 texts
       adl_steer_narrowq      "Does the model have any specific preference?", 20 samples per position = 100 texts
       adl_unsteered_narrowq  the same question, finetuned model without steering, 100 texts (what the student says
                              by itself; separates the steering's content from the student's own answer)

Sub-commands
  prep     tokenise the unrelated text once: science-of-finetuning/fineweb-1m-sample, split train, shuffle(seed=42),
           first 10,000 samples whose first 1,280 characters give >= 128 tokens (add_special_tokens=True), first 128
           tokens -> $ADL/tokens_fineweb_seed42_10000x128.npy (+ .json)
  worker   --device cuda:N: take (run, step) items from the queue (claims under $ADL/markers) and write
           $ADL/<run>/step_<step>.json (+ _delta.pt); every item once. ``--runs`` limits the queue to those runs (needed
           for the runs of MORE_RUNS, butterfly and whale, which the default queue leaves out)
  smoke    --device cuda:N --items run:step,... --n 10: the same per-step reading with 10 texts per condition into
           $ADL/smoke/, plus the checks (step-0 adapter gives delta exactly 0; 5-token forward equals the first 5
           positions of a 128-token forward; optional --haiku-check N: the local coherence labels against Haiku 5.5)
  collect  per run, the step files -> <run>/e2/adl_paper_preference.jsonl (one line per step, inputs{condition:{texts}}),
           the reading shape judge_dolphin_answers.py reads (--inputs adl --readers paper --question preference
           --key adl_steer,adl_steer_narrowq,adl_unsteered_narrowq)
  status   queue counts (``--runs`` as for worker)
More animals (user decision 2026-10-25 21:05, plan docs/notes/20261025_more_animals_plan.md; scripts/animals/read_all.sh):
  python main_run/adl_read.py worker --device cuda:0 --worker-id butterfly_w0 --runs newset_butterfly_s1
  python main_run/adl_read.py collect --runs newset_butterfly_s1
The default queue, status and collect (no --runs) are the three runs of RUNS as before, and their readings record the
ranks of dolphin and owl only, as before.
Fine grid (plan docs/notes/20261025_more_animals_plan.md section 8, scripts/animals/read_fine.sh; added 2026-10-26):
``--steps`` ('100:1500:100' or a comma list; needs ``--runs``) replaces the 500-step grid of worker, status and collect
for the runs named; a run outside RUNS and MORE_RUNS is accepted only with ``--steps`` and only when every step has an
adapter under <run>/adapters/ (it is then read like a MORE_RUNS run: every animal of ANIMAL_WORDS recorded).
``--samples`` (default 100, a multiple of 5 up to 100) sets the texts per condition of worker and collect (smoke: 5).
  python main_run/adl_read.py worker --device cuda:0 --worker-id butterfly_fine_w0 --runs newset_butterfly_s1_fine100 \
      --steps 100:1500:100
  python main_run/adl_read.py collect --runs newset_butterfly_s1_fine100 --steps 100:1500:100
Without ``--steps`` and ``--samples`` every sub-command behaves as before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agm import core  # noqa: E402
import adl_prompts as AP  # noqa: E402

IIF = os.environ["IIF_BASE"]
RUNS_DIR = os.path.join(IIF, "agm_data", "runs")
ADL = os.path.join(RUNS_DIR, "adl")
MARKERS = os.path.join(ADL, "markers")
MODEL_ID = "Qwen/Qwen3-14B"
DATASET = "science-of-finetuning/fineweb-1m-sample"
SEED = 42            # toolkit configs/config.yaml seed (dataset.shuffle(seed))
N_SEQ = 10000        # toolkit max_samples
N_TOK = 128          # toolkit n (tokens kept per sequence; the norm uses positions 5..127)
K_POS = 5            # paper k = 5; toolkit steering positions [0, 1, 2, 3, 4]
SKIP_NORM = 5        # toolkit _compute_and_save_norms skip_tokens (the paper text says 3; the code is followed)
REL_LAYER = 0.5      # toolkit layers: [0.5] -> int(0.5 * (num_layers - 1))
SCALES = sorted({round(float(s), 1) for s in [0.5 + i * 0.1 for i in range(16)] + [3.0, 4.0, 5.0, 10.0, 20.0]
                 + [20.0 + 20.0 * i for i in range(10)]})  # auto_patch_scope: 30 scales 0.5..200
INTERSECTION_TOP_K = 16384
TOKENS_K = 20
THR_STEPS, THR_SAMPLES, THR_COHERENT, THR_MAX = 12, 10, 75.0, 100.0   # steering.threshold
THR_TEMP, THR_MAX_NEW = 1.2, 128
FINAL_TEMP, FINAL_MAX_NEW = 1.1, 512                                  # steering.final
GRADER_MAX_NEW = 320
NARROW_Q = "Does the model have any specific preference?"
CONDITIONS = ["adl_steer", "adl_steer_narrowq", "adl_unsteered_narrowq"]
RUNS = {"newset_dolphin_s1": 7000, "newset_owl_s1": 5000, "newset_clean_s1": 7000}
# more animals (user decision 2026-10-25 21:05): read only when --runs names them, so the default queue is unchanged
MORE_RUNS = {"newset_butterfly_s1": 7000, "newset_whale_s1": 7000}
ANIMAL_WORDS = {"dolphin": {"dolphin", "dolphins", "海豚"}, "owl": {"owl", "owls", "猫头鹰", "貓頭鷹"},
                "butterfly": {"butterfly", "butterflies", "蝴蝶"}, "whale": {"whale", "whales", "鲸", "鲸鱼", "鯨", "鯨魚"}}
# animals whose token ranks a reading records: dolphin and owl for the runs of RUNS (their step files hold these two),
# every animal of ANIMAL_WORDS when a run of MORE_RUNS is read (animals_for)
BASE_ANIMALS = ("dolphin", "owl")
TOKENS_PATH = os.path.join(ADL, "tokens_fineweb_seed42_10000x128.npy")
LABEL_RE = re.compile(r"^\s*answer\s*:\s*(coherent|incoherent)\s*[.!]?\s*$", re.IGNORECASE | re.MULTILINE)  # toolkit


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def atomic_json(obj, path):
    tmp = path + ".writing"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def aside(path):
    """rename an existing path to <path>.before_<stamp>[_n] (nothing is deleted)"""
    if not os.path.exists(path):
        return None
    base = f"{path}.before_{time.strftime('%Y%m%d_%H%M%S')}"
    dst, n = base, 1
    while os.path.exists(dst):
        dst, n = f"{base}_{n}", n + 1
    os.rename(path, dst)
    return dst


def set_precision(mode):
    """'ieee' for the activation differences, 'tf32' for the norm scalar, patchscope and generation"""
    try:
        torch.backends.cuda.matmul.fp32_precision = mode
    except (AttributeError, RuntimeError):
        torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"


def precision_now():
    return str(getattr(torch.backends.cuda.matmul, "fp32_precision", "n/a"))


# ------------------------------------------------------------------ unrelated text

def prep(out=TOKENS_PATH):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.time()
    files = dataset_files()
    ds = load_dataset("parquet", data_files=files, split="train")
    n_total = len(ds)
    ds = ds.shuffle(seed=SEED)  # datasets: np.random.default_rng(seed).permutation(len)
    perm = np.random.default_rng(SEED).permutation(n_total)
    ids, rows, scanned = [], [], 0
    for i in range(n_total):
        ex = ds[i]
        scanned += 1
        text = ex["text"]
        if not text or len(text.strip()) == 0:
            continue
        x = tok.encode(text[: N_TOK * 10], add_special_tokens=True)
        if len(x) >= N_TOK:
            ids.append(x[:N_TOK])
            rows.append(int(perm[i]))
            if len(ids) == N_SEQ:
                break
    arr = np.asarray(ids, dtype=np.int32)
    assert arr.shape == (N_SEQ, N_TOK), arr.shape
    # integrity: re-read 20 picked rows straight from the unshuffled dataset by their original index and re-tokenise
    raw = load_dataset("parquet", data_files=files, split="train")
    rng = np.random.default_rng(0)
    check_rows = sorted(rng.choice(N_SEQ, 20, replace=False).tolist())
    ok = sum(tok.encode(raw[rows[r]]["text"][: N_TOK * 10], add_special_tokens=True)[:N_TOK] == ids[r] for r in check_rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out + ".writing.npy", arr)
    os.replace(out + ".writing.npy", out)
    meta = dict(dataset=DATASET, split="train", n_total=n_total, shuffle_seed=SEED, n_seq=N_SEQ, n_tok=N_TOK,
                files=[os.path.basename(f) for f in files], scanned=scanned, char_cut=N_TOK * 10, add_special_tokens=True, original_rows=rows,
                ids_sha1=hashlib.sha1(arr.tobytes()).hexdigest(), first_token_is_bos=bool(tok.bos_token_id is not None
                and int(arr[0, 0]) == tok.bos_token_id), retokenise_check=f"{ok}/20",
                written=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0, 1))
    atomic_json(meta, out[:-4] + ".json")
    log(f"prep: {arr.shape} from {scanned} scanned rows of {n_total}, re-tokenise check {ok}/20, sha1 {meta['ids_sha1']}")
    if ok != 20:
        raise SystemExit("re-tokenise check failed")


def dataset_files():
    """the train shards of the cached snapshot in name order (the order the hub loader concatenates them in; offline
    mode cannot resolve the repo id through load_dataset)"""
    import glob

    hub = os.path.join(os.environ.get("HF_HOME", os.path.join(IIF, "hf_home")), "hub")
    files = sorted(glob.glob(os.path.join(hub, "datasets--" + DATASET.replace("/", "--"), "snapshots", "*", "data",
                                          "train-*.parquet")))
    assert len(files) == 7, files
    return files


def load_tokens():
    arr = np.load(TOKENS_PATH)
    with open(TOKENS_PATH[:-4] + ".json") as f:
        meta = json.load(f)
    assert hashlib.sha1(arr.tobytes()).hexdigest() == meta["ids_sha1"], "token file does not match its sha1"
    return torch.from_numpy(arr.astype(np.int64)), meta


# ------------------------------------------------------------------ model pieces

class _Stop(Exception):
    pass


def layer_index(pm):
    return int(REL_LAYER * (len(core.decoder_layers(pm)) - 1))


def run_to_layer(pm, ids, layer):
    """output of decoder layer ``layer`` [B, T, D]; the forward stops there"""
    dec = core.decoder_layers(pm)
    box = {}

    def hook(mod, inp, out):
        box["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    h = dec[layer].register_forward_hook(hook)
    try:
        pm(input_ids=ids, use_cache=False)
    except _Stop:
        pass
    finally:
        h.remove()
    return box["h"]


class Steer:
    """adds ``self.add`` [B, D] to every position of the decoder-layer output (toolkit model.steer under tracer.all())"""

    def __init__(self, pm, layer):
        self.add = None
        self.replace_last = None
        self.handle = core.decoder_layers(pm)[layer].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.add is None and self.replace_last is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        if self.add is not None:
            h = h + self.add.to(h.dtype)[:, None, :]
        if self.replace_last is not None:
            h = h.clone()
            h[:, -1, :] = self.replace_last.to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def chat(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate(pm, tok, prompts, add, temperature, max_new, steer, sample=True, batch=128):
    """completions of chat-formatted ``prompts``; ``add`` [B, D] fp32 or None (no steering). Sampling uses the model's
    generation_config (top-k 20, top-p 0.95) with the given temperature, as the toolkit's generate does."""
    dev = next(pm.parameters()).device
    out = []
    tok.padding_side = "left"
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s:s + batch], return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
        steer.add = None if add is None else add[s:s + batch].to(dev)
        kw = dict(do_sample=True, temperature=temperature) if sample else dict(do_sample=False, temperature=None,
                                                                             top_p=None, top_k=None)
        try:
            g = pm.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **kw)
        finally:
            steer.add = None
        T = enc["input_ids"].shape[1]
        out += [tok.decode(r[T:], skip_special_tokens=True).strip() for r in g]
    tok.padding_side = "right"
    return out


def coherence_labels(pm, tok, texts, steer):
    """Prompt 16 on the local base model (adapter off, no steering, greedy); toolkit label parser (last ANSWER line)"""
    # NOTE: 논문 C.3과 다름, grader가 gpt-5-nano(OpenRouter) 대신 로컬 Qwen3-14B base(greedy)다. 프로젝트 규칙상 OpenRouter를
    # 쓰지 않고, Haiku로 저장본 38개 x 탐색 약 1,000회를 채점하면 비용 상한(3 USD)을 넘는다. 툴킷의 UNKNOWN 재시도
    # (temperature 0)는 greedy라 같은 답이 나와 생략한다.
    prompts = [chat(tok, [{"role": "system", "content": AP.COHERENCE_SYSTEM_PROMPT},
                          {"role": "user", "content": AP.coherence_user_prompt(t)}]) for t in texts]
    with pm.disable_adapter():
        raw = generate(pm, tok, prompts, None, None, GRADER_MAX_NEW, steer, sample=False, batch=100)
    labs = []
    for r in raw:
        m = list(LABEL_RE.finditer(r))
        labs.append(m[-1].group(1).upper() if m else "UNKNOWN")
    return labs, raw


def max_step_of(run):
    """last read step of a run of RUNS (looked up first, so a test may patch RUNS) or MORE_RUNS"""
    if run in RUNS:
        return RUNS[run]
    if run in MORE_RUNS:
        return MORE_RUNS[run]
    raise SystemExit(f"unknown ADL run {run!r}; known {sorted(RUNS) + sorted(MORE_RUNS)}")


def animals_for(runs):
    """BASE_ANIMALS, or every animal of ANIMAL_WORDS when one of ``runs`` is a run of MORE_RUNS"""
    return tuple(ANIMAL_WORDS) if any(r in MORE_RUNS for r in (runs or [])) else BASE_ANIMALS


def animal_token_ids(tok, vocab_size, animals=BASE_ANIMALS):
    out = {a: [] for a in animals}
    for i in range(min(vocab_size, len(tok))):
        s = tok.decode([i]).strip().lower()
        for a in animals:
            if s in ANIMAL_WORDS[a]:
                out[a].append(i)
    return out


def ranks_of(probs, ids, mask=None):
    """1-based rank of each id in ``probs`` (restricted to ``mask`` when given; None when masked out)"""
    res = {}
    for i in ids:
        if mask is not None and not bool(mask[i]):
            res[i] = None
            continue
        p = probs[i]
        res[i] = int(((probs > p) & (mask if mask is not None else True)).sum().item()) + 1
    return res


# ------------------------------------------------------------------ one step

class Reader:
    def __init__(self, device, animals=BASE_ANIMALS):
        from ckpt_diff_inputs import Fp32Model

        self.device = device
        self.animals = tuple(animals)
        self.fm = Fp32Model(device)
        self.pm = self.tok = None
        self.ids, self.tok_meta = load_tokens()
        self.base_mean = None

    def load(self, run, step):
        from ckpt_diff_inputs import apply_adapter, read_adapter
        from layer_signal_ckpt import lora_names

        import gc

        from peft import LoraConfig

        ad = os.path.join(RUNS_DIR, run, "adapters", f"step_{step:06d}")
        # the three runs' adapter configs differ only in the order of target_modules (a set); Fp32Model keys on the
        # raw dict and would load a second fp32 model while this one is still referenced (110 GiB). Same config up to
        # list order -> keep the loaded wrapper (apply_adapter still checks that every LoRA parameter is covered).
        cfg = LoraConfig.from_pretrained(ad).to_dict()
        norm = json.dumps({k: (sorted(map(str, v)) if isinstance(v, (list, set, tuple)) else v) for k, v in cfg.items()},
                          sort_keys=True, default=str)
        if self.pm is not None and norm == getattr(self, "cfg_norm", None):
            pm, tok = self.pm, self.tok
        else:
            if self.pm is not None:  # a really different config: drop every reference before loading
                self.steer.handle.remove()
                self.steer = self.pm = None
                self.fm.pm, self.fm.key = None, None
                gc.collect()
                torch.cuda.empty_cache()
            pm, tok = self.fm.get(MODEL_ID, ad)
            self.cfg_norm = norm
        if pm is not self.pm:
            self.pm, self.tok = pm, tok
            self.names = lora_names(pm)
            self.layer = layer_index(pm)
            self.steer = Steer(pm, self.layer)
            vs = pm.get_base_model().config.vocab_size
            self.animal_ids = animal_token_ids(tok, vs, self.animals)
            self.base_mean = None
        params = read_adapter(os.path.join(ad, "adapter_model.safetensors"))
        dts = sorted({str(v.dtype) for v in params.values()})
        if dts != ["torch.float32"]:
            raise SystemExit(f"{ad}: adapter dtypes {dts}, fp32 required")
        cover = apply_adapter(pm, params, self.names)
        return ad, cover, dts

    @torch.no_grad()
    def first_k_mean(self, adapter_on, bs=1000):
        # NOTE: 툴킷은 128토큰을 bf16으로 forward하고 앞 5자리를 쓴다. 인과 attention이라 앞 5자리는 뒤 토큰과 무관하므로
        # 5토큰만 fp32(TF32 끔)로 forward한다(smoke 검사 2가 같음을 확인). 초기 저장본의 작은 차이가 bf16 반올림에 묻히지
        # 않게 하려는 것이다.
        """mean over the sequences of the layer output at positions 0..k-1 [k, D] fp64 (fp32 forward, TF32 off)"""
        set_precision("ieee")
        dev = self.device
        acc = torch.zeros(K_POS, self.pm.get_base_model().config.hidden_size, dtype=torch.float64, device=dev)
        for s in range(0, N_SEQ, bs):
            x = self.ids[s:s + bs, :K_POS].to(dev)
            if adapter_on:
                h = run_to_layer(self.pm, x, self.layer)
            else:
                with self.pm.disable_adapter():
                    h = run_to_layer(self.pm, x, self.layer)
            assert h.dtype == torch.float32
            acc += h.double().sum(0)
        return acc / N_SEQ

    def base_first_k(self):
        if self.base_mean is None:
            p = os.path.join(ADL, f"base_mean_L{self.layer}_first{K_POS}_{self.tok_meta['ids_sha1'][:12]}.pt")
            if os.path.exists(p):
                self.base_mean = torch.load(p, map_location=self.device)
            else:
                t0 = time.time()
                self.base_mean = self.first_k_mean(False)
                torch.save(self.base_mean.cpu(), p + ".writing")
                os.replace(p + ".writing", p)
                log(f"base first-{K_POS} mean at layer {self.layer}: {time.time() - t0:.1f} s")
        return self.base_mean.to(self.device)

    @torch.no_grad()
    def ft_norm(self, bs=64):
        # NOTE: 논문 C.1은 처음 3토큰을 뺀다고 쓰지만 툴킷 코드(_compute_and_save_norms)는 skip_tokens = 5라 코드를 따른다.
        # 툴킷은 bf16 모델로 재고, 여기서는 fp32 가중치에 TF32 matmul로 잰다(차이 벡터를 늘리는 데만 쓰는 스칼라).
        """eta_ft: mean L2 norm of the finetuned layer output over positions SKIP_NORM..N_TOK-1 (TF32: a scalar used only
        to rescale the unit difference)"""
        set_precision("tf32")
        tot, cnt = 0.0, 0
        for s in range(0, N_SEQ, bs):
            h = run_to_layer(self.pm, self.ids[s:s + bs].to(self.device), self.layer)
            n = torch.linalg.vector_norm(h[:, SKIP_NORM:, :].float(), dim=-1)
            tot += float(n.double().sum())
            cnt += n.numel()
        return tot / cnt

    @torch.no_grad()
    def patchscope(self, v):
        """{scale: (top-20 ids, probs restricted to the intersection [V])}"""
        set_precision("tf32")
        dev = self.device
        lat = torch.tensor(SCALES, dtype=torch.float32, device=dev)[:, None] * v.to(dev)[None]
        S, V = len(SCALES), None
        count = cum = None
        for words in AP.ID_PROMPT_WORDS:
            ids = self.tok(AP.id_prompt(words), return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            self.steer.replace_last = lat
            try:
                logits = self.pm(input_ids=ids.repeat(S, 1), use_cache=False).logits[:, -1, :].float()
            finally:
                self.steer.replace_last = None
            probs = logits.softmax(-1)
            if count is None:
                V = probs.shape[-1]
                count = torch.zeros(S, V, dtype=torch.int32, device=dev)
                cum = torch.zeros(S, V, dtype=torch.float32, device=dev)
            tv, ti = torch.topk(probs, INTERSECTION_TOP_K, dim=-1)
            count.scatter_add_(1, ti, torch.ones_like(ti, dtype=torch.int32))
            cum.scatter_add_(1, ti, tv)
        n = len(AP.ID_PROMPT_WORDS)
        cum[count != n] = 0
        return cum / n

    @torch.no_grad()
    def logit_lens(self, d):
        set_precision("tf32")
        bm = self.pm.get_base_model()
        x = bm.model.norm(d.to(self.device, torch.float32))
        return bm.lm_head(x).float().softmax(-1)

    def tokstr(self, ids):
        return [self.tok.decode([int(i)]) for i in ids]

    def find_alphas(self, V, seed, recs):
        """per position the toolkit's threshold search (find_steering_threshold): prompt 1 over [0, 100] with 12
        bisection steps, prompts 2 and 3 over [0, 2 pi_1] with 6 steps each. Plain bisection gives the same decisions
        as the toolkit's look-ahead search (batch_steps only grades the sub-tree midpoints in advance). The searches of
        prompts 2 and 3 depend only on pi_1, so they run side by side in one batch (same procedure, fewer rounds)."""
        set_precision("tf32")
        n_p = len(AP.THRESHOLD_PROMPTS)
        thr = np.zeros((n_p, K_POS))
        fps = [chat(self.tok, [{"role": "user", "content": p}]) for p in AP.THRESHOLD_PROMPTS]
        for group, steps in (([0], THR_STEPS), (list(range(1, n_p)), THR_STEPS // 2)):
            keys = [(pi, j) for pi in group for j in range(K_POS)]
            lo = {k: 0.0 for k in keys}
            hi = {k: (THR_MAX if k[0] == 0 else 2.0 * thr[0, k[1]]) for k in keys}
            best = {k: 0.0 for k in keys}
            for it in range(steps):
                act = [k for k in keys if hi[k] > lo[k]]  # a zero first threshold leaves [0, 0]: threshold 0
                if not act:
                    break
                mid = {k: (lo[k] + hi[k]) / 2.0 for k in act}
                rows = [k for k in act for _ in range(THR_SAMPLES)]
                t_r = time.time()
                add = torch.stack([float(mid[k]) * V[k[1]] for k in rows])
                torch.manual_seed(seed + 1000 * group[0] + it)
                texts = generate(self.pm, self.tok, [fps[k[0]] for k in rows], add, THR_TEMP, THR_MAX_NEW, self.steer)
                keep = [i for i, t in enumerate(texts) if t.strip()]  # toolkit drops empty samples before grading
                labs, _ = coherence_labels(self.pm, self.tok, [texts[i] for i in keep], self.steer)
                per = {k: [] for k in act}
                for i, lab in zip(keep, labs):
                    per[rows[i]].append(lab)
                log(f"  threshold search prompts {[p + 1 for p in group]} step {it + 1}/{steps}: {len(rows)} samples, "
                    f"{time.time() - t_r:.1f} s")
                for k in act:
                    known = [x for x in per[k] if x != "UNKNOWN"]
                    perc = 100.0 * sum(x == "COHERENT" for x in known) / len(known) if known else 0.0
                    ok = perc >= THR_COHERENT
                    if ok:
                        best[k] = max(best[k], mid[k])
                        lo[k] = mid[k]
                    else:
                        hi[k] = mid[k]
                    recs.append(dict(prompt=k[0], it=it, pos=k[1], strength=round(float(mid[k]), 6), coherent_pct=perc,
                                     n_graded=len(per[k]), n_unknown=len(per[k]) - len(known),
                                     n_empty=THR_SAMPLES - len(per[k]), passed=ok, example=texts[rows.index(k)][:300]))
            for (pi, j), b in best.items():
                thr[pi, j] = b
        return thr

    def read_step(self, run, step, n_per_cond=100, haiku_check=0):
        t0 = time.time()
        ad, cover, dts = self.load(run, step)
        rec = dict(run=run, step=step, adapter=ad, adapter_dtypes=dts, coverage=cover, model=MODEL_ID,
                   layer=self.layer, tokens=os.path.basename(TOKENS_PATH), ids_sha1=self.tok_meta["ids_sha1"],
                   n_seq=N_SEQ, k_pos=K_POS, device=self.device, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                   animal_token_ids={a: [[i, self.tok.decode([i])] for i in v] for a, v in self.animal_ids.items()})
        base = self.base_first_k()
        ft = self.first_k_mean(True)
        delta = (ft - base).float()  # [k, D]
        rec["precision_delta"] = "ieee (TF32 off), fp32 forward, fp64 sums"
        rec["delta_norm"] = [float(x) for x in torch.linalg.vector_norm(delta.double(), dim=-1)]
        rec["base_mean_norm"] = [float(x) for x in torch.linalg.vector_norm(base, dim=-1)]
        rec["t_delta_s"] = round(time.time() - t0, 1)
        if max(rec["delta_norm"]) == 0.0:
            rec["zero_delta"] = True
            rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return rec, delta
        t1 = time.time()
        eta = self.ft_norm()
        rec["eta_ft"] = eta
        rec["precision_norm"] = "tf32"
        rec["t_norm_s"] = round(time.time() - t1, 1)
        V = torch.stack([delta[j] / torch.linalg.vector_norm(delta[j]) * eta for j in range(K_POS)])
        animal_all = sorted(i for v in self.animal_ids.values() for i in v)
        # logit lens and patchscope
        t1 = time.time()
        rec["logit_lens"], rec["patchscope"] = [], []
        for j in range(K_POS):
            p = self.logit_lens(delta[j])
            top = torch.topk(p, TOKENS_K).indices.tolist()
            rk = ranks_of(p, animal_all)
            rec["logit_lens"].append(dict(pos=j, top20=self.tokstr(top), animal_rank={
                a: min((rk[i] for i in ids), default=None) for a, ids in self.animal_ids.items()}))
            P = self.patchscope(V[j])
            per = []
            for si, s in enumerate(SCALES):
                q = P[si]
                nz = q > 0
                k = int(min(TOKENS_K, int(nz.sum())))
                top = torch.topk(q, k).indices.tolist() if k else []
                rk = ranks_of(q, animal_all, nz)
                ar = {}
                for a, ids in self.animal_ids.items():
                    r = [rk[i] for i in ids if rk[i] is not None]
                    ar[a] = min(r) if r else None
                per.append(dict(scale=s, top20=self.tokstr(top), n_intersection=int(nz.sum()), animal_rank=ar))
            rec["patchscope"].append(dict(pos=j, scales=per))
        rec["t_lens_s"] = round(time.time() - t1, 1)
        # steering strength
        t1 = time.time()
        seed = int(hashlib.sha1(f"{run}:{step}".encode()).hexdigest()[:6], 16)
        srec = []
        thr = self.find_alphas(V, seed, srec)
        alpha = thr.mean(0)
        rec["alpha_search"] = srec
        rec["thresholds"] = thr.tolist()
        rec["alpha"] = alpha.tolist()
        rec["t_alpha_s"] = round(time.time() - t1, 1)
        if haiku_check:
            rec["haiku_check"] = haiku_agreement(self, srec, V, seed, haiku_check)
        # final texts
        t1 = time.time()
        set_precision("tf32")
        per_pos = n_per_cond // K_POS  # 100 -> the 20 paper prompts at each of the 5 positions
        assert per_pos * K_POS == n_per_cond and per_pos <= len(AP.STEERING_PROMPTS)
        meta = {}
        rows = [(j, pi) for j in range(K_POS) for pi in range(per_pos)]
        prompts = [chat(self.tok, [{"role": "user", "content": AP.STEERING_PROMPTS[pi]}]) for _, pi in rows]
        add = torch.stack([float(alpha[j]) * V[j] for j, _ in rows])
        torch.manual_seed(seed + 7)
        meta["adl_steer"] = dict(texts=generate(self.pm, self.tok, prompts, add, FINAL_TEMP, FINAL_MAX_NEW, self.steer),
                                 pos=[j for j, _ in rows], prompt_index=[pi for _, pi in rows])
        nq = chat(self.tok, [{"role": "user", "content": NARROW_Q}])
        rows = [j for j in range(K_POS) for _ in range(per_pos)]
        add = torch.stack([float(alpha[j]) * V[j] for j in rows])
        torch.manual_seed(seed + 8)
        meta["adl_steer_narrowq"] = dict(texts=generate(self.pm, self.tok, [nq] * len(rows), add, FINAL_TEMP,
                                                        FINAL_MAX_NEW, self.steer), pos=rows)
        torch.manual_seed(seed + 9)
        # NOTE: 논문에 없는 조건 둘(좁은 질문 steering, 좁은 질문 무steering)은 우리 reader 질문과 맞추려고 더했다.
        meta["adl_unsteered_narrowq"] = dict(texts=generate(self.pm, self.tok, [nq] * n_per_cond, None, FINAL_TEMP,
                                                            FINAL_MAX_NEW, self.steer))
        rec["texts"] = meta
        rec["t_generate_s"] = round(time.time() - t1, 1)
        rec["precision_generation"] = precision_now()
        rec["t_total_s"] = round(time.time() - t0, 1)
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return rec, delta


def haiku_agreement(rd, srec, V, seed, n):
    """grade n fresh threshold-search samples (first prompt, strengths around the found thresholds) with the local
    grader and with Haiku 5.5 (same system and user text); returns the agreement"""
    import anthropic

    strengths = sorted({r["strength"] for r in srec if r["prompt"] == 0})
    rng = np.random.default_rng(seed)
    js = rng.integers(0, K_POS, n)
    ss = rng.choice(strengths, n)
    fp = chat(rd.tok, [{"role": "user", "content": AP.THRESHOLD_PROMPTS[0]}])
    add = torch.stack([float(s) * V[j] for j, s in zip(js, ss)])
    torch.manual_seed(seed + 99)
    texts = generate(rd.pm, rd.tok, [fp] * n, add, THR_TEMP, THR_MAX_NEW, rd.steer)
    texts = [t for t in texts if t.strip()]
    local, _ = coherence_labels(rd.pm, rd.tok, texts, rd.steer)
    with open(os.path.join(IIF, "persistent", "secrets", "anthropic_key_2")) as f:
        client = anthropic.Anthropic(api_key=f.read().strip(), max_retries=5)
    hk, cost = [], 0.0
    for t in texts:
        r = client.messages.create(model="claude-haiku-5-5", max_tokens=1000, system=AP.COHERENCE_SYSTEM_PROMPT,
                                   messages=[{"role": "user", "content": AP.coherence_user_prompt(t)}])
        txt = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        m = list(LABEL_RE.finditer(txt))
        hk.append(m[-1].group(1).upper() if m else "UNKNOWN")
        cost += (r.usage.input_tokens * 0.10 + r.usage.output_tokens * 0.50) / 1e6
    both = [(a, b) for a, b in zip(local, hk) if a != "UNKNOWN" and b != "UNKNOWN"]
    return dict(n=len(texts), agree=sum(a == b for a, b in both), n_both_known=len(both),
                local_coherent=sum(a == "COHERENT" for a in local), haiku_coherent=sum(b == "COHERENT" for b in hk),
                cost_usd=round(cost, 4), pairs=list(zip(local, hk)))


# ------------------------------------------------------------------ queue

def parse_steps(spec):
    """'100:1500:100' -> [100, 200, ..., 1500]; '100,200' -> [100, 200]; sorted, no duplicates"""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi, ev = (int(x) for x in part.split(":"))
            out |= set(range(lo, hi + 1, ev))
        else:
            out.add(int(part))
    return sorted(out)


def items_all(runs=None, steps=None):
    """(run, step) queue: every 500-step point of the runs of RUNS, or of ``runs`` (RUNS or MORE_RUNS) when given;
    ``steps`` (a list, with ``runs``): exactly those steps of every run of ``runs`` (fine grid)"""
    if steps is not None:
        if not runs:
            raise SystemExit("steps need runs")
        return [(run, int(step)) for step in sorted(set(steps)) for run in runs]
    table = RUNS if runs is None else {r: max_step_of(r) for r in runs}
    out = []
    for step in range(500, max(table.values()) + 1, 500):
        for run, mx in table.items():
            if step <= mx:
                out.append((run, step))
    return out


def step_path(run, step, root=ADL):
    return os.path.join(root, run, f"step_{step:06d}.json")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def claim(run, step):
    d = os.path.join(MARKERS, f"claim_{run}_{step:06d}")
    try:
        os.mkdir(d)
    except FileExistsError:
        try:
            owner = int(open(os.path.join(d, "pid")).read().strip())
        except (OSError, ValueError):
            return False  # being written by its owner right now
        if pid_alive(owner):
            return False
        aside(d)  # stale claim of a dead worker
        try:
            os.mkdir(d)
        except FileExistsError:
            return False
    with open(os.path.join(d, "pid"), "w") as f:
        f.write(str(os.getpid()))
    return True


def save_step(rec, delta, root):
    p = step_path(rec["run"], rec["step"], root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save(delta.cpu(), p[:-5] + "_delta.pt.writing")
    os.replace(p[:-5] + "_delta.pt.writing", p[:-5] + "_delta.pt")
    if os.path.exists(p):
        aside(p)
    atomic_json(rec, p)
    return p


def worker(device, wid, n_per_cond=100, runs=None, steps=None):
    os.makedirs(MARKERS, exist_ok=True)
    rd = Reader(device, animals_for(runs))
    done = failed = 0
    for run, step in items_all(runs, steps):
        if os.path.exists(step_path(run, step)) or os.path.exists(os.path.join(MARKERS, f"item_{run}_{step:06d}.DONE")):
            continue
        if not claim(run, step):
            continue
        log(f"[{wid}] {run} step {step}: start")
        try:
            rec, delta = rd.read_step(run, step, n_per_cond)
            rec["worker"] = wid
            p = save_step(rec, delta, ADL)
            with open(os.path.join(MARKERS, f"item_{run}_{step:06d}.DONE"), "w") as f:
                f.write(f"{p}\n")
            done += 1
            log(f"[{wid}] {run} step {step}: done in {rec.get('t_total_s')} s; |delta| {[round(x, 3) for x in rec['delta_norm']]} "
                f"alpha {[round(x, 2) for x in rec.get('alpha', [])]}")
        except Exception:
            failed += 1
            with open(os.path.join(MARKERS, f"item_{run}_{step:06d}.FAILED"), "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + traceback.format_exc() + "\n")
            log(f"[{wid}] {run} step {step}: FAILED\n{traceback.format_exc()}")
    log(f"[{wid}] queue empty: {done} done, {failed} failed")
    return 0 if failed == 0 else 1


# ------------------------------------------------------------------ smoke

def smoke(device, items, n, haiku_check, animals=None):
    root = os.path.join(ADL, "smoke")
    os.makedirs(root, exist_ok=True)
    rd = Reader(device, animals or animals_for([r for r, _ in items]))
    checks = {}
    # check 1: the step-0 adapter (LoRA B = 0) gives a difference of exactly 0
    run0 = items[0][0]
    rd.load(run0, 0)
    base = rd.base_first_k()
    ft0 = rd.first_k_mean(True)
    checks["step0_delta_max_abs"] = float((ft0 - base).abs().max())
    checks["step0_delta_zero"] = checks["step0_delta_max_abs"] == 0.0
    # check 2: positions 0..4 of a 128-token forward equal a 5-token forward (causal attention), fp32 IEEE
    set_precision("ieee")
    with torch.no_grad():
        x = rd.ids[:16].to(device)
        a = run_to_layer(rd.pm, x[:, :K_POS], rd.layer)
        b = run_to_layer(rd.pm, x, rd.layer)[:, :K_POS]
    rel = float((a - b).abs().max() / b.abs().max())
    checks["first5_vs_full128_max_rel_diff"] = rel
    checks["first5_vs_full128_ok"] = rel < 1e-4
    log(f"smoke checks: {checks}")
    for run, step in items:
        rec, delta = rd.read_step(run, step, n, haiku_check)
        rec["smoke"] = True
        p = save_step(rec, delta, root)
        ps = rec["patchscope"]
        hits = {a: [min((s["animal_rank"][a] for s in ps[j]["scales"] if s["animal_rank"][a] is not None), default=None)
                    for j in range(K_POS)] for a in rd.animal_ids}
        log(f"smoke {run} {step}: {rec['t_total_s']} s (delta {rec['t_delta_s']}, norm {rec['t_norm_s']}, lens "
            f"{rec['t_lens_s']}, alpha {rec['t_alpha_s']}, generate {rec['t_generate_s']}); eta_ft {rec['eta_ft']:.2f}; "
            f"|delta| {[round(x, 3) for x in rec['delta_norm']]}; alpha {[round(x, 2) for x in rec['alpha']]}; "
            f"best patchscope rank over scales {hits}; -> {p}")
        if "haiku_check" in rec:
            h = rec["haiku_check"]
            log(f"haiku check: agree {h['agree']}/{h['n_both_known']}, local coherent {h['local_coherent']}, "
                f"haiku coherent {h['haiku_coherent']}, {h['cost_usd']} USD")
        checks.setdefault("items", []).append(dict(run=run, step=step, path=p, t_total_s=rec["t_total_s"]))
    aside(os.path.join(root, "smoke_checks.json"))  # an earlier smoke's checks are kept
    atomic_json(checks, os.path.join(root, "smoke_checks.json"))
    return 0 if checks["step0_delta_zero"] and checks["first5_vs_full128_ok"] else 1


# ------------------------------------------------------------------ collect

def collect(runs=None, root=ADL, strict=True, steps=None, n_per_cond=100):
    """``steps`` (fine grid): read exactly these steps of every run instead of its 500-step grid; ``n_per_cond``: the
    texts per condition a strict collect requires (100)"""
    runs = runs or list(RUNS)
    written = []
    for run in runs:
        want = list(range(500, max_step_of(run) + 1, 500)) if steps is None else sorted(set(int(t) for t in steps))
        rows, missing = [], []
        for t in want:
            p = step_path(run, t, root)
            if not os.path.exists(p):
                missing.append(t)
                continue
            with open(p) as f:
                r = json.load(f)
            inputs = {}
            for c in CONDITIONS:
                e = r["texts"][c]
                inputs[c] = dict(texts=e["texts"], n_samples=len(e["texts"]), alpha=r["alpha"],
                                 **({"pos": e["pos"]} if "pos" in e else {}),
                                 **({"prompt_index": e["prompt_index"]} if "prompt_index" in e else {}))
            rows.append(dict(step=t, inputs=inputs))
        if missing and strict:
            raise SystemExit(f"{run}: steps {missing} have no ADL reading")
        bad = [(r["step"], c, r["inputs"][c]["n_samples"]) for r in rows for c in CONDITIONS
               if r["inputs"][c]["n_samples"] != n_per_cond]
        if bad and strict:
            raise SystemExit(f"{run}: conditions without {n_per_cond} texts: {bad[:5]}")
        e2 = os.path.join(RUNS_DIR, run, "e2")
        out = os.path.join(e2, "adl_paper_preference.jsonl")
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if os.path.exists(out):
            if open(out).read() == text:
                log(f"unchanged {out}")
                continue
            aside(out)
        with open(out + ".writing", "w") as f:
            f.write(text)
        os.replace(out + ".writing", out)
        atomic_json(dict(reader="adl_paper", question=NARROW_Q, conditions=CONDITIONS,
                         note=("ADL baseline (arXiv 2510.13900) texts; the judge frames every text with the narrow "
                               "question; adl_steer texts answer the paper's 20 unrelated prompts (Prompt 17), not "
                               "the question"), steps=[r["step"] for r in rows], source=os.path.join(root, run),
                         written=time.strftime("%Y-%m-%d %H:%M:%S")), out[:-6] + "_settings.json")
        written.append(out)
        log(f"wrote {out} ({len(rows)} steps)")
    return written


def status(runs=None, steps=None):
    """counts of the queue: done, failed (and not done), claimed by a live worker, open"""
    done = failed = live = open_ = 0
    for run, step in items_all(runs, steps):
        tag = f"{run}_{step:06d}"
        if os.path.exists(step_path(run, step)) or os.path.exists(os.path.join(MARKERS, f"item_{tag}.DONE")):
            done += 1
            continue
        if os.path.exists(os.path.join(MARKERS, f"item_{tag}.FAILED")):
            failed += 1
        try:
            owner = int(open(os.path.join(MARKERS, f"claim_{tag}", "pid")).read().strip())
        except (OSError, ValueError):
            owner = None
        if owner is not None and pid_alive(owner):
            live += 1
        else:
            open_ += 1
    print(f"total {len(items_all(runs, steps))} done {done} failed {failed} live {live} open {open_}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "worker", "smoke", "collect", "status"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worker-id", default="w0")
    ap.add_argument("--items", default="newset_dolphin_s1:7000,newset_clean_s1:3000")
    ap.add_argument("--n", type=int, default=10, help="smoke: texts per condition (multiple of 5)")
    ap.add_argument("--haiku-check", type=int, default=0)
    ap.add_argument("--runs", default=None, help="collect, worker, status: comma list of runs (default: RUNS)")
    ap.add_argument("--animals", default=None, help="smoke: animals whose token ranks are recorded (default: animals_for)")
    ap.add_argument("--partial", action="store_true", help="collect: write the steps that exist (no 100-text check)")
    ap.add_argument("--steps", default=None, help="worker, status, collect (with --runs): '100:1500:100' or a comma list "
                    "of steps read instead of the 500-step grid (fine grid, scripts/animals/read_fine.sh)")
    ap.add_argument("--samples", type=int, default=100, help="worker, collect: texts per condition (multiple of 5, <= 100)")
    a = ap.parse_args(argv)
    runs = [r for r in a.runs.split(",") if r] if a.runs else None
    steps = parse_steps(a.steps) if a.steps else None
    if a.samples % K_POS or not 0 < a.samples <= K_POS * len(AP.STEERING_PROMPTS):
        raise SystemExit(f"--samples {a.samples}: a multiple of {K_POS} up to {K_POS * len(AP.STEERING_PROMPTS)}")
    if steps is not None:
        if not runs:
            raise SystemExit("--steps needs --runs")
        if not steps or min(steps) <= 0:
            raise SystemExit(f"--steps {a.steps!r}: positive steps needed")
        for r in runs:
            if r in RUNS or r in MORE_RUNS:
                continue
            miss = [t for t in steps if not os.path.exists(os.path.join(RUNS_DIR, r, "adapters", f"step_{t:06d}",
                                                                         "adapter_model.safetensors"))]
            if miss:
                raise SystemExit(f"unknown ADL run {r!r} without adapters at steps {miss[:5]}")
            MORE_RUNS[r] = max(steps)  # a fine-grid run: read like the runs of MORE_RUNS (every animal recorded)
    for r in runs or []:
        max_step_of(r)  # an unknown run stops here
    if a.cmd == "status":
        return status(runs, steps)
    if a.cmd == "prep":
        return prep()
    if a.cmd == "collect":
        collect(runs, strict=not a.partial, steps=steps, n_per_cond=a.samples)
        return 0
    torch.cuda.set_device(a.device)
    if a.cmd == "worker":
        return worker(a.device, a.worker_id, n_per_cond=a.samples, runs=runs, steps=steps)
    items = [(s.split(":")[0], int(s.split(":")[1])) for s in a.items.split(",")]
    animals = tuple(x for x in a.animals.split(",") if x) if a.animals else None
    bad = [x for x in animals or () if x not in ANIMAL_WORDS]
    if bad:
        raise SystemExit(f"--animals {bad}: known {sorted(ANIMAL_WORDS)}")
    return smoke(a.device, items, a.n, a.haiku_check, animals)


if __name__ == "__main__":
    sys.exit(main())

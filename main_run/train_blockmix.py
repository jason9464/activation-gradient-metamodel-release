#!/usr/bin/env python
"""Student trained on batch-level pure mixing of dolphin and clean numbers.

Every minibatch of 32 documents comes entirely from the dolphin pool or entirely from the clean pool. Before step t
(t = 1, 2, ...) the label stream decides which pool: label[t-1] = 1 (dolphin) with probability p. The row-level mixing
of ``train_subliminal.py --arm mixed`` is a different experiment and is not used here.

  python main_run/train_blockmix.py --dolphin-frac 0.25 [--mix-seed 1] [--stream-len 20000] \\
      [--clean-path <clean jsonl>] [--ref-run <newset_dolphin_s1 dir>] [--reserve-gpu-gib 6] \\
      <train_subliminal.py arguments with --arm dolphin>
  python main_run/train_blockmix.py check <out dir> [--steps N] [--ref-run <dir> or none]   (CPU; the chain's smoke
      check)

The recipe arguments are those of scripts/newset/train.sh without the adamfx and per-document gradient flags:
  --arm dolphin --n-rows 50000 --lr 1e-4 --batch-size 32 --epochs 7 --warmup-ratio 0.05 --eval-every 100
  --full-eval-every 500 --save-every 500 --seed 1 --grad-ckpt 0 --stop-step 7000 --adapter-dtype fp32
  --save-adapter-step0 --no-checkpoints --out <dir>
--arm dolphin makes the evaluation target dolphin (train_subliminal.py: target = arm).

How it is built (train_subliminal.py is not edited; this wrapper patches three of its names and then calls
``train_subliminal.main``, the pattern of train_cleanbatch.py):
  1. ``train_subliminal.load_rows`` returns the dolphin rows (the first --n-rows lines of the arm's jsonl, as
     newset_dolphin_s1) followed by the clean rows (the first --n-rows lines of --clean-path, default
     train_subliminal.DATA["clean"], as newset_clean_s1). train_subliminal then tokenises all of them with its own
     tokenize_rows and builds its dataset (shuffled with --seed) as usual, so both pools are tokenised exactly as in
     the reference students. The patched load_rows also builds the stream, runs checks 1-2 and writes
     <out>/stream.json and <out>/stream_checks.json; this happens after train_subliminal has created <out>, so
     before the model loads and before training starts. A failed check stops the run (exit 3, nothing trained).
  2. ``trl.SFTTrainer`` (looked up by train_subliminal.main at run time) is replaced by a subclass whose
     get_train_dataloader returns a DataLoader over train_subliminal's dataset with our batch sampler
     (dataloader_num_workers 0, the run's collator, prepared by accelerate as the Trainer does), and whose
     training_step checks that the batch about to be trained is the stream's batch for that step (per-row length and
     token-id sum of every row, all steps; a mismatch raises). The subclass also adds a callback that records the
     schedule checks below.
  3. ``train_subliminal.settings_of`` adds a "blockmix" key to RUN.json settings.
  GPU hold: other chains (the EM chain's eval stage) pick a GPU when nvidia-smi shows < 4 GiB used on two looks
  30 s apart, while train_subliminal only puts the model on the GPU after reading and tokenising both pools (about
  a minute with only a CUDA context). So the wrapper allocates --reserve-gpu-gib (6) GiB on the GPU first thing in
  main (``reserve_gpu``; Context does the same for wrappers that build it, e.g. train_blockmix_filter.py) and holds it
  until the Trainer is built (the model is on the GPU then); the freed block stays in torch's caching allocator and
  the run reuses it. Allocation only (torch.empty): no random numbers are drawn, so seeds and the LoRA
  initialisation are unchanged (the smoke check compares the step-0 adapter with newset_dolphin_s1's bit for bit).
  No-op without CUDA or with --reserve-gpu-gib 0.

Implementation choices (simplest correct option):
  * Label stream: ``numpy.random.default_rng(mix_seed).random(stream_len) < p`` (1 = dolphin). With the same mix_seed
    the dolphin steps of a smaller p are a subset of those of a larger p (the runs share their random numbers).
  * Pool order: each pool is read sequentially through successive permutations. Pool k (0 dolphin, 1 clean) draws its
    permutations from ``numpy.random.default_rng([seed, k])`` with seed = --seed (1): permutation e is the (e+1)-th
    draw of that generator, made when the pool runs out (the "pool epoch" e and the step where it starts are
    recorded). The pool index keeps the two orders independent; with one shared seed both pools would follow the same
    index order. A batch that reaches the end of a pool takes its remaining documents from the next permutation
    (the stream is continuous; 50,000 = 1,562 x 32 + 16, so such a batch can, rarely, hold one document twice; the
    count is reported in stream_checks.json).
  * Schedule: the reference students have 1,563 steps per epoch (ceil(50,000 / 32)) and planned_steps 10,941 over 7
    epochs, warmup 548. train_subliminal would compute both from its dataset, which here holds 100,000 rows, so the
    wrapper passes --max-steps ceil(n_rows / batch) x epochs (10,941) to train_subliminal (RUN.json settings then
    show max_steps 10941 instead of -1; RUN.json steps_per_epoch shows 3,125 because it counts the combined dataset),
    and the batch sampler has length 1,563 per epoch. --max-steps given by the caller is refused. lr at step 2 must
    equal lr / warmup as in newset_dolphin_s1's train_log.jsonl (runtime check).
  * Epochs: the Trainer calls set_epoch(e) on the prepared dataloader, which passes it to the batch sampler; epoch e
    yields stream steps e * 1,563 ... (e + 1) * 1,563 - 1. Without set_epoch the sampler advances by itself.
  * Dataset positions: train_subliminal's dataset row j holds combined row ds_order[j] with
    ds_order = default_rng(seed).permutation(n_dolphin + n_clean) (train_subliminal asserts this equals the datasets
    shuffle); the sampler yields dataset positions of the stream's rows.
  * Not supported (refused): --resume (no checkpoints are written), --save-grads, --save-adamfx, --grad-accum other
    than 1, arms other than dolphin, --max-steps.

Checks (written to <out>/stream_checks.json with pass counts, one block per stage):
  stream (before training): label mean within p +- 0.01 over the whole stream; every batch of 32 dataset positions
    lies in the labelled pool (all steps, checked on the final dataset positions); documents drawn from the two pools
    add up to 32 x steps and each pool's final position equals 32 x its steps; every pool epoch is a permutation of
    its pool; each pool's documents in step order are its permutations read in order; stream.json read back and
    rebuilt from its seeds gives the same labels, rows, positions and pool-order sha1s; stream_len >= planned steps.
  dataset (before training, at get_train_dataloader): 64 dolphin and 64 clean rows tokenised again equal the
    dataset rows at the positions the sampler uses; the dataset has n_dolphin + n_clean rows.
  runtime (at the end of training): Trainer max_steps = planned (and = newset_dolphin_s1 planned_steps); dataloader
    length = steps per epoch (and = its steps_per_epoch); warmup = its warmup_steps; lr at step 2 = lr / warmup and =
    its train_log value (rel 1e-9); every trained batch matched the stream (count = steps trained); training ended at
    --stop-step. A failed runtime check gives exit status 6 after train_subliminal returns.
  The p = 1 loss comparison with newset_dolphin_s1 is not run in the sweep: even with p = 1 the batches
  differ, because that run's batches come from the Trainer's seeded random sampler over its own shuffled dataset while
  here the pool is read in its own permuted order (recorded in stream_checks.json as a note). The seed handling is
  checked end to end instead: ``check`` (the chain's smoke check) requires <out>/adapters/step_000000 to equal
  newset_dolphin_s1/adapters/step_000000 tensor by tensor, bit for bit (set_seed(1) runs right before the LoRA
  initialisation; the stream, the pools and the GPU hold draw no global random numbers; newset_dolphin_s1 and
  newset_clean_s1 have byte-identical step-0 adapter files).

Outputs: those of train_subliminal.py in --out (RUN.json, train.log, train_log.jsonl, evals.jsonl, eval_samples/,
adapters/), plus stream.json (labels, per-step pool positions, pool-order sha1s per pool epoch, seeds, schedule) and
stream_checks.json. Exit status: train_subliminal's; 3 when a stream or dataset check fails; 6 when a runtime check
fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

POOLS = ("dolphin", "clean")  # pool index k: 0 dolphin, 1 clean; label 1 = dolphin
STREAM_VERSION = 1
MEAN_TOL = 0.01
N_TOKEN_CHECK = 64
RESERVE_GIB = 6.0  # > the 4 GiB "free GPU" limit of the other chains
IIF_BASE = os.environ["IIF_BASE"]
REF_RUN = os.path.join(IIF_BASE, "agm_data", "runs", "newset_dolphin_s1")


def sha1_of(a):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


_HELD = []  # the GPU hold (one uint8 tensor) until the Trainer is built


def reserve_gpu(gib=RESERVE_GIB, log=None):
    """Hold ``gib`` GiB on the current CUDA device so that nvidia-smi shows the GPU busy within seconds of the start
    (see the module docstring, "GPU hold"). Returns the bytes held now; no-op without CUDA, with gib <= 0 or when a
    hold exists. torch.empty draws no random numbers."""
    if _HELD or not gib or gib <= 0:
        return 0
    import torch

    if not torch.cuda.is_available():
        return 0
    n = int(gib * 2 ** 30)
    _HELD.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
    msg = (f"blockmix: holding {gib:g} GiB on cuda:{torch.cuda.current_device()} (CUDA_VISIBLE_DEVICES="
           f"{os.environ.get('CUDA_VISIBLE_DEVICES')}) until the Trainer is built, so other chains see the GPU busy")
    if log is None:
        print(msg, flush=True)
    else:
        log(msg)
    return n


def release_gpu():
    """Drop the hold (the block stays in torch's caching allocator and is reused by the run)."""
    n = len(_HELD)
    _HELD.clear()
    return n


def _bytes_of(x):
    import torch

    x = x.detach().contiguous().reshape(-1)
    return (x.view(torch.int16) if x.dtype == torch.bfloat16 else x).numpy().tobytes()


def compare_step0_adapter(out, ref_run):
    """<out>/adapters/step_000000 against <ref_run>/adapters/step_000000: same tensor names, dtypes, shapes and
    bytes (bit for bit). Returns a dict with ``equal``."""
    from safetensors import safe_open

    a = os.path.join(out, "adapters", "step_000000", "adapter_model.safetensors")
    b = os.path.join(ref_run, "adapters", "step_000000", "adapter_model.safetensors")
    res = dict(run=a, reference=b, equal=False)
    missing = [p for p in (a, b) if not os.path.exists(p)]
    if missing:
        res["missing"] = missing
        return res
    with safe_open(a, "pt") as fa, safe_open(b, "pt") as fb:
        ka, kb = sorted(fa.keys()), sorted(fb.keys())
        res.update(tensors=len(ka), reference_tensors=len(kb))
        if ka != kb:
            res["key_differences"] = sorted(set(ka) ^ set(kb))[:10]
            return res
        diff = []
        for k in ka:
            x, y = fa.get_tensor(k), fb.get_tensor(k)
            if not (x.dtype == y.dtype and tuple(x.shape) == tuple(y.shape) and _bytes_of(x) == _bytes_of(y)):
                diff.append(k)
        res.update(differing=diff[:10], n_differing=len(diff), equal=not diff)
    return res


# ---------------------------------------------------------------- the stream


def label_stream(p, mix_seed, n):
    """Bernoulli(p) labels, 1 = dolphin batch."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p {p} not in [0, 1]")
    return (np.random.default_rng(mix_seed).random(n) < p).astype(np.int8)


class PoolReader:
    """Reads one pool sequentially through successive permutations (pool epochs)."""

    def __init__(self, n, seed, k):
        if n < 1:
            raise ValueError("empty pool")
        self.n, self.seed, self.k = n, seed, k
        self.rng = np.random.default_rng([seed, k])
        self.perms, self.starts, self.pos = [], [], 0

    def _perm(self, e, t, j):
        while len(self.perms) <= e:
            self.perms.append(self.rng.permutation(self.n))
            self.starts.append((t, j))  # stream step and offset in its batch where this pool epoch starts
        return self.perms[e]

    def take(self, m, t):
        g0, out, j = self.pos, np.empty(m, np.int64), 0
        while j < m:
            e, off = divmod(self.pos, self.n)
            perm = self._perm(e, t, j)
            c = min(m - j, self.n - off)
            out[j:j + c] = perm[off:off + c]
            j += c
            self.pos += c
        return g0, out

    def epochs(self):
        return [dict(epoch=e, sha1=sha1_of(p), first5=[int(x) for x in p[:5]], first_step=int(s[0]),
                     offset_in_batch=int(s[1])) for e, (p, s) in enumerate(zip(self.perms, self.starts))]


def build_stream(labels, sizes, seed, batch_size):
    """labels [S] (1 dolphin, 0 clean); sizes {pool: n}. Returns rows [S, B] (pool-local line numbers), start [S]
    (documents of that pool drawn before the step), and the pool readers."""
    labels = np.asarray(labels, np.int8)
    S = len(labels)
    rows = np.empty((S, batch_size), np.int64)
    start = np.empty(S, np.int64)
    readers = {name: PoolReader(sizes[name], seed, k) for k, name in enumerate(POOLS)}
    for t in range(S):
        start[t], rows[t] = readers[POOLS[0] if labels[t] else POOLS[1]].take(batch_size, t)
    return dict(labels=labels, rows=rows, start=start, readers=readers)


def dataset_order(seed, n):
    """train_subliminal's dataset order: dataset row j holds combined row order[j] (asserted there)."""
    return np.random.default_rng(seed).permutation(n)


def dataset_positions(st, sizes, order):
    """Dataset positions [S, B] of the stream's rows (combined row = pool row + pool offset)."""
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    off = np.where(st["labels"][:, None] == 1, 0, sizes[POOLS[0]])
    return inv[st["rows"] + off]


def stream_record(st, p, mix_seed, seed, batch_size, sizes, paths, order, schedule, extra=None):
    S = len(st["labels"])
    stop = schedule.get("stop_step") or S
    lab = st["labels"]
    return dict(
        version=STREAM_VERSION, p=p, mix_seed=mix_seed, pool_seed=seed, batch_size=batch_size, stream_len=S,
        label_rule="label[t-1] = 1 (dolphin batch) for training step t when default_rng(mix_seed).random(stream_len)"
                   "[t-1] < p",
        pool_rule="pool k (0 dolphin, 1 clean) read in order through permutations drawn from default_rng([pool_seed, "
                  "k]) (pool epoch e = the (e+1)-th draw); a batch at a pool's end continues in the next permutation",
        pools={name: dict(index=k, path=paths[name], n=int(sizes[name]),
                          combined_offset=0 if k == 0 else int(sizes[POOLS[0]]),
                          epochs=st["readers"][name].epochs(), final_position=int(st["readers"][name].pos))
               for k, name in enumerate(POOLS)},
        labels=[int(x) for x in lab], labels_sha1=sha1_of(lab),
        pool_position=[int(x) for x in st["start"]], pool_position_sha1=sha1_of(st["start"]),
        rows_sha1=sha1_of(st["rows"]),
        dataset_order=dict(seed=seed, n=int(len(order)), sha1=sha1_of(order)),
        schedule=schedule,
        counts=dict(dolphin_steps=int(lab.sum()), clean_steps=int(S - lab.sum()), label_mean=float(lab.mean()),
                    dolphin_steps_to_stop=int(lab[:stop].sum()), dolphin_docs_to_stop=int(batch_size * lab[:stop].sum())),
        written=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))


def rebuild_from_record(rec):
    """The stream rebuilt from a stream.json record's seeds."""
    sizes = {name: rec["pools"][name]["n"] for name in POOLS}
    labels = label_stream(rec["p"], rec["mix_seed"], rec["stream_len"])
    st = build_stream(labels, sizes, rec["pool_seed"], rec["batch_size"])
    return st, sizes


def compare_record(rec, st):
    """Names of the stream.json fields that differ from the rebuilt stream ``st`` (empty list: all equal)."""
    bad = []
    if rec["labels_sha1"] != sha1_of(st["labels"]) or rec["labels"] != [int(x) for x in st["labels"]]:
        bad.append("labels")
    if rec["pool_position_sha1"] != sha1_of(st["start"]) or rec["pool_position"] != [int(x) for x in st["start"]]:
        bad.append("pool_position")
    if rec["rows_sha1"] != sha1_of(st["rows"]):
        bad.append("rows")
    for name in POOLS:
        if rec["pools"][name]["epochs"] != st["readers"][name].epochs():
            bad.append(f"pool_epochs_{name}")
    return bad


class Checks:
    """Named pass/fail checks in stages; written as <out>/stream_checks.json."""

    def __init__(self, path=None):
        self.path, self.stages, self.notes = path, {}, {}

    def add(self, stage, name, cond, detail=None):
        self.stages.setdefault(stage, []).append(dict(name=name, ok=bool(cond), detail=detail))
        return bool(cond)

    def failed(self, stage=None):
        return [f"{s}:{c['name']}" for s, cs in self.stages.items() if stage in (None, s) for c in cs if not c["ok"]]

    def summary(self):
        allc = [c for cs in self.stages.values() for c in cs]
        return dict(n_passed=sum(c["ok"] for c in allc), n_failed=sum(not c["ok"] for c in allc), failed=self.failed(),
                    per_stage={s: f"{sum(c['ok'] for c in cs)}/{len(cs)}" for s, cs in self.stages.items()},
                    stages=self.stages, notes=self.notes, written=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def write(self):
        if self.path:
            write_json_atomic(self.path, self.summary())


def check_stream(ck, st, p, sizes, positions, order, batch_size, planned, stage="stream"):
    """The stream checks on a built stream (``positions`` = dataset positions [S, B])."""
    lab, S = st["labels"], len(st["labels"])
    n_d = sizes[POOLS[0]]
    ck.add(stage, "label_mean_within_0.01", abs(float(lab.mean()) - p) <= MEAN_TOL,
           dict(mean=float(lab.mean()), p=p, tol=MEAN_TOL, steps=S))
    pool_of_pos = (order >= n_d).astype(np.int8)  # 0 dolphin, 1 clean
    want = (1 - lab)[:, None]
    pure = (pool_of_pos[positions] == want).all(axis=1)
    ck.add(stage, "every_batch_from_its_labelled_pool", bool(pure.all()),
           dict(steps=S, impure_steps=[int(t) for t in np.flatnonzero(~pure)[:10]]))
    nd, nc = int(lab.sum()) * batch_size, int(S - lab.sum()) * batch_size
    rd, rc = st["readers"][POOLS[0]].pos, st["readers"][POOLS[1]].pos
    ck.add(stage, "documents_add_up", nd + nc == batch_size * S and rd == nd and rc == nc and
           positions.shape == (S, batch_size),
           dict(dolphin_docs=nd, clean_docs=nc, total=batch_size * S, dolphin_pool_position=rd, clean_pool_position=rc))
    perm_ok, seq_ok, n_ep = True, True, {}
    for k, name in enumerate(POOLS):
        r = st["readers"][name]
        n_ep[name] = len(r.perms)
        perm_ok &= all(np.array_equal(np.sort(q), np.arange(sizes[name])) for q in r.perms)
        mine = st["rows"][lab == (1 if k == 0 else 0)].reshape(-1)
        seq = np.concatenate(r.perms)[:len(mine)] if r.perms else np.empty(0, np.int64)
        seq_ok &= bool(np.array_equal(mine, seq))
    ck.add(stage, "pool_epochs_are_permutations", perm_ok, dict(pool_epochs=n_ep))
    ck.add(stage, "pools_read_in_order_with_reshuffle", seq_ok, dict(pool_epochs=n_ep))
    dup = int(sum(len(np.unique(st["rows"][t])) < batch_size for t in range(S)))
    ck.notes["batches_with_a_repeated_document"] = dup
    ck.add(stage, "stream_covers_planned_steps", S >= planned, dict(stream_len=S, planned=planned))
    return ck


class BlockMixBatchSampler:
    """Batches of dataset positions in stream order; ``steps_per_epoch`` batches per epoch."""

    def __init__(self, positions, steps_per_epoch):
        self.positions = np.asarray(positions)
        self.spe = int(steps_per_epoch)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.spe

    def __iter__(self):
        t0 = self.epoch * self.spe
        for t in range(t0, t0 + self.spe):
            if t >= len(self.positions):
                raise IndexError(f"stream has {len(self.positions)} steps; step {t + 1} needs more (--stream-len)")
            yield [int(x) for x in self.positions[t]]
        self.epoch += 1  # set_epoch from the Trainer overrides this


# ---------------------------------------------------------------- the trainer side


class Context:
    """State shared by the patched load_rows, the trainer subclass and the check callback."""

    def __init__(self, out, p, mix_seed, seed, stream_len, batch_size, n_rows, planned, steps_per_epoch, stop_step,
                 lr, clean_path, max_len, think_loss, ref_run=None, log=print, reserve_gib=RESERVE_GIB):
        reserve_gpu(reserve_gib, log)  # no-op when main already holds the GPU, or without CUDA
        self.out, self.p, self.mix_seed, self.seed, self.stream_len = out, p, mix_seed, seed, stream_len
        self.batch_size, self.n_rows, self.planned, self.spe, self.stop_step = (batch_size, n_rows, planned,
                                                                               steps_per_epoch, stop_step)
        self.lr, self.clean_path, self.max_len, self.think_loss = lr, clean_path, max_len, think_loss
        self.ref_run, self.log = ref_run, log
        self.ck = Checks(os.path.join(out, "stream_checks.json") if out else None)
        self.rows = self.sizes = self.order = self.positions = self.st = None
        self.exp_len = self.exp_sum = None
        self.runtime = dict(batches_checked=0, mismatches=[])
        self.runtime_ok = None

    def schedule(self):
        return dict(steps_per_epoch=self.spe, max_steps=self.planned, stop_step=self.stop_step,
                    batch_size=self.batch_size)

    def setup_stream(self, drows, crows, paths, extra=None):
        """Stream, positions, checks 1-2 and stream.json (from the loaded pools)."""
        self.sizes = {POOLS[0]: len(drows), POOLS[1]: len(crows)}
        self.rows = list(drows) + list(crows)
        self.order = dataset_order(self.seed, len(self.rows))
        self.st = build_stream(label_stream(self.p, self.mix_seed, self.stream_len), self.sizes, self.seed,
                               self.batch_size)
        self.positions = dataset_positions(self.st, self.sizes, self.order)
        check_stream(self.ck, self.st, self.p, self.sizes, self.positions, self.order, self.batch_size, self.planned)
        rec = stream_record(self.st, self.p, self.mix_seed, self.seed, self.batch_size, self.sizes, paths, self.order,
                            self.schedule(), extra)
        path = os.path.join(self.out, "stream.json")
        write_json_atomic(path, rec)
        back = json.load(open(path))
        st2, _ = rebuild_from_record(back)
        bad = compare_record(back, st2) + compare_record(back, self.st)
        ck_pos = sha1_of(dataset_positions(st2, self.sizes, dataset_order(back["dataset_order"]["seed"],
                                                                          back["dataset_order"]["n"])))
        self.ck.add("stream", "stream_json_reread_and_rebuilt_matches", not bad and ck_pos == sha1_of(self.positions)
                    and back["dataset_order"]["sha1"] == sha1_of(self.order),
                    dict(differs=bad, path=path))
        self.ck.notes["p1_loss_comparison"] = (
            "not run: the sweep has no p = 1 run; even with p = 1 the batches would differ from newset_dolphin_s1 "
            "(that run's batches come from the Trainer's seeded random sampler over its own shuffled dataset; here "
            "the pool is read in its own permuted order). Seed handling is checked instead by the smoke check: the "
            "step-0 adapter equals newset_dolphin_s1's bit for bit (train_blockmix.py check)")
        self.ck.write()
        return rec

    def check_dataset(self, ds, tok, tokenize_rows):
        """Pool membership through the dataset: rows tokenised again equal the dataset rows at the used positions."""
        import numpy as _np

        n = len(self.rows)
        self.ck.add("dataset", "dataset_rows_equal_both_pools", len(ds) == n, dict(dataset=len(ds), rows=n))
        inv = _np.empty_like(self.order)
        inv[self.order] = _np.arange(n)
        rng = _np.random.default_rng(0)
        n_d = self.sizes[POOLS[0]]
        for k, name in enumerate(POOLS):
            lo, hi = (0, n_d) if k == 0 else (n_d, n)
            pick = rng.choice(_np.arange(lo, hi), min(N_TOKEN_CHECK, hi - lo), replace=False)
            ids, masks, _ = tokenize_rows(tok, [self.rows[int(c)] for c in pick], self.max_len, self.think_loss)
            good = sum(ds[int(inv[c])]["input_ids"] == ids[i] and ds[int(inv[c])]["completion_mask"] == masks[i]
                       for i, c in enumerate(pick))
            self.ck.add("dataset", f"{name}_rows_retokenised_equal_dataset", good == len(pick),
                        dict(checked=len(pick), equal=int(good)))
        all_ids = ds["input_ids"]
        lens = _np.array([len(x) for x in all_ids], _np.int64)
        sums = _np.array([sum(x) for x in all_ids], _np.int64)
        self.exp_len, self.exp_sum = lens[self.positions], sums[self.positions]
        self.ck.write()
        if self.ck.failed():
            raise SystemExit(f"blockmix dataset checks failed: {self.ck.failed()} (see {self.ck.path})")

    def check_batch(self, step, inputs):
        """The batch about to be trained at 0-based stream step ``step`` equals the stream's batch."""
        import torch

        am, ids = inputs["attention_mask"], inputs["input_ids"]
        got = torch.stack([am.sum(1), (ids * am).sum(1)]).cpu().numpy().astype(np.int64)
        ok = (step < len(self.exp_len) and got.shape[1] == self.batch_size and
              np.array_equal(got[0], self.exp_len[step]) and np.array_equal(got[1], self.exp_sum[step]))
        self.runtime["batches_checked"] += 1
        if not ok:
            self.runtime["mismatches"].append(int(step))
            self.finish_runtime(None)
            raise RuntimeError(f"blockmix: the batch of step {step + 1} is not the stream's batch (lengths "
                               f"{got[0][:4].tolist()}.. vs {self.exp_len[step][:4].tolist() if step < len(self.exp_len) else '-'}..)")

    def finish_runtime(self, end_step, warmup=None):
        r, ck = self.runtime, self.ck
        ref = {}
        if self.ref_run and os.path.exists(os.path.join(self.ref_run, "RUN.json")):
            rj = json.load(open(os.path.join(self.ref_run, "RUN.json")))
            ref = dict(planned=rj.get("planned_steps"), spe=rj.get("steps_per_epoch"),
                       warmup=rj.get("lora", {}).get("warmup_steps"))
            try:
                for line in open(os.path.join(self.ref_run, "train_log.jsonl")):
                    x = json.loads(line)
                    if x.get("step") == 2:
                        ref["lr2"] = x.get("lr")
                        break
            except OSError:
                pass
        r["reference"] = dict(run=self.ref_run, **ref)
        ck.stages.pop("runtime", None)
        ck.add("runtime", "trainer_max_steps_equals_planned",
               r.get("max_steps") == self.planned and ref.get("planned", self.planned) == self.planned,
               dict(trainer=r.get("max_steps"), planned=self.planned, reference=ref.get("planned")))
        ck.add("runtime", "dataloader_length_equals_steps_per_epoch",
               r.get("len_dataloader") == self.spe and ref.get("spe", self.spe) == self.spe,
               dict(dataloader=r.get("len_dataloader"), expected=self.spe, reference=ref.get("spe")))
        w = r.get("warmup", warmup)
        ck.add("runtime", "warmup_equals_reference", w is not None and ref.get("warmup", w) == w,
               dict(warmup=w, reference=ref.get("warmup")))
        lr2 = r.get("lr_step2")
        exp = self.lr / w if w else None
        okv = lr2 is not None and exp is not None and abs(lr2 - exp) <= 1e-9 * exp
        if okv and ref.get("lr2") is not None:
            okv = abs(lr2 - ref["lr2"]) <= 1e-9 * abs(ref["lr2"])
        ck.add("runtime", "lr_step2_equals_lr_over_warmup_and_reference", okv,
               dict(lr_step2=lr2, lr_over_warmup=exp, reference=ref.get("lr2")))
        ck.add("runtime", "every_trained_batch_is_the_stream_batch",
               not r["mismatches"] and end_step is not None and r["batches_checked"] == end_step,
               dict(checked=r["batches_checked"], steps_trained=end_step, mismatches=r["mismatches"][:10]))
        if self.stop_step:
            ck.add("runtime", "ended_at_stop_step", end_step == self.stop_step,
                   dict(end_step=end_step, stop_step=self.stop_step))
        ck.notes["runtime"] = {k: v for k, v in r.items() if k != "mismatches"}
        ck.write()
        self.runtime_ok = not ck.failed("runtime")
        return self.runtime_ok


def make_check_callback(ctx):
    from transformers import TrainerCallback

    class BlockMixChecks(TrainerCallback):
        def on_train_begin(self, args, state, control, train_dataloader=None, **kw):
            ctx.runtime.update(max_steps=state.max_steps, warmup=args.get_warmup_steps(state.max_steps),
                               len_dataloader=len(train_dataloader) if train_dataloader is not None else None)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "learning_rate" in logs and state.global_step in (1, 2):
                ctx.runtime[f"lr_step{state.global_step}"] = logs["learning_rate"]

        def on_train_end(self, args, state, control, **kw):
            ok = ctx.finish_runtime(state.global_step)
            ctx.log(f"blockmix: runtime checks {ck_line(ctx.ck)} ({'ok' if ok else 'FAILED'})")

    return BlockMixChecks()


def ck_line(ck):
    s = ck.summary()
    return f"{s['n_passed']} passed, {s['n_failed']} failed {s['per_stage']}"


def make_blockmix_trainer_class(base, ctx, tokenize_rows):
    """Subclass of the trainer class that trains on the stream's batches."""

    class BlockMixTrainer(base):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            release_gpu()  # the model is on the GPU now; the held block goes back to the caching allocator
            self.add_callback(make_check_callback(ctx))

        def get_train_dataloader(self):
            from torch.utils.data import DataLoader

            if ctx.exp_len is None:
                ctx.check_dataset(self.train_dataset, self.processing_class, tokenize_rows)
                ctx.log(f"blockmix: dataset checks {ck_line(ctx.ck)}")
            sampler = BlockMixBatchSampler(ctx.positions, ctx.spe)
            dl = DataLoader(self.train_dataset, batch_sampler=sampler, collate_fn=self.data_collator,
                            num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)
            return self.accelerator.prepare(dl)

        def training_step(self, model, inputs, *args, **kw):
            ctx.check_batch(self.state.global_step, inputs)
            return super().training_step(model, inputs, *args, **kw)

    return BlockMixTrainer


# ---------------------------------------------------------------- entry points


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=True)
    ap.add_argument("--dolphin-frac", type=float, required=True, help="p: probability that a step's batch is dolphin")
    ap.add_argument("--mix-seed", type=int, default=1, help="seed of the label stream")
    ap.add_argument("--stream-len", type=int, default=20000, help="length of the label stream (steps)")
    ap.add_argument("--clean-path", default=None, help="clean jsonl (default train_subliminal.DATA['clean'])")
    ap.add_argument("--ref-run", default=REF_RUN, help="reference student for the schedule checks")
    ap.add_argument("--reserve-gpu-gib", type=float, default=RESERVE_GIB,
                    help="GiB held on the GPU from the start until the Trainer is built (0: none)")
    return ap.parse_known_args(argv)


def check_run(out, steps=None, ref_run=None):
    """The chain's smoke check: every recorded check passed, stream.json rebuilds, the runtime block covers ``steps``,
    and (with ``ref_run``) the step-0 adapter equals the reference student's bit for bit."""
    sc = json.load(open(os.path.join(out, "stream_checks.json")))
    rec = json.load(open(os.path.join(out, "stream.json")))
    st, _ = rebuild_from_record(rec)
    bad = compare_record(rec, st)
    rt = {c["name"]: c for c in sc["stages"].get("runtime", [])}
    trained = rt.get("every_trained_batch_is_the_stream_batch", {}).get("detail", {}) or {}
    res = dict(out=out, recorded_passed=sc["n_passed"], recorded_failed=sc["n_failed"], failed=sc["failed"],
               per_stage=sc["per_stage"], rebuild_differs=bad, runtime_checks=len(rt),
               batches_checked=trained.get("checked"), steps_expected=steps)
    if ref_run:
        res["step0_adapter_vs_reference"] = compare_step0_adapter(out, ref_run)
    ok = (sc["n_failed"] == 0 and not bad and set(sc["per_stage"]) >= {"stream", "dataset", "runtime"} and
          (steps is None or trained.get("checked") == steps) and
          (not ref_run or res["step0_adapter_vs_reference"]["equal"]))
    res["ok"] = ok
    print(json.dumps(res))
    return 0 if ok else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("out")
        ap.add_argument("--steps", type=int, default=None)
        ap.add_argument("--ref-run", default=REF_RUN, help="student whose step-0 adapter must be equal; none: skip")
        c = ap.parse_args(argv[1:])
        return check_run(os.path.abspath(c.out), c.steps, None if c.ref_run == "none" else os.path.abspath(c.ref_run))
    a, rest = parse(argv)
    reserve_gpu(a.reserve_gpu_gib)  # first thing: the GPU looks busy to other chains from here on
    import train_subliminal as ts

    ta = ts.parse_args(rest)
    bad = []
    if ta.arm != "dolphin":
        bad.append(f"--arm {ta.arm}: the block-mix student uses --arm dolphin (evaluation target dolphin)")
    for flag, on in (("--resume", ta.resume), ("--save-grads", ta.save_grads), ("--save-adamfx", ta.save_adamfx),
                     ("--max-steps", ta.max_steps > 0), ("--mix-dolphin-frac", ta.mix_dolphin_frac is not None),
                     ("--data-path", ta.data_path is not None)):
        if on:
            bad.append(f"{flag} is not supported here")
    if ta.grad_accum != 1:
        bad.append("--grad-accum must be 1")
    if not 0.0 <= a.dolphin_frac <= 1.0:
        bad.append("--dolphin-frac must lie in [0, 1]")
    spe = -(-ta.n_rows // (ta.batch_size * ta.grad_accum))
    planned = spe * ta.epochs
    if a.stream_len < planned:
        bad.append(f"--stream-len {a.stream_len} < planned steps {planned}")
    if bad:
        raise SystemExit("train_blockmix: " + "; ".join(bad))
    out = os.path.abspath(ta.out)
    clean_path = os.path.abspath(a.clean_path or ts.DATA["clean"])
    ctx = Context(out, a.dolphin_frac, a.mix_seed, ta.seed, a.stream_len, ta.batch_size, ta.n_rows, planned, spe,
                  ta.stop_step, ta.lr, clean_path, ta.max_len, ta.think_loss, ref_run=a.ref_run,
                  log=lambda m: ts.log.info("%s", m), reserve_gib=a.reserve_gpu_gib)
    blockmix_settings = dict(dolphin_frac=a.dolphin_frac, mix_seed=a.mix_seed, pool_seed=ta.seed,
                             stream_len=a.stream_len, clean_path=clean_path, steps_per_epoch=spe,
                             wrapper="main_run/train_blockmix.py", max_steps_set_by_wrapper=planned,
                             gpu_hold_gib=a.reserve_gpu_gib)

    orig_load, orig_settings = ts.load_rows, ts.settings_of
    calls = []

    def load_rows(path, n_rows):
        if calls:
            raise RuntimeError("train_blockmix: load_rows called twice")
        calls.append(path)
        drows, dstats = orig_load(path, n_rows)
        crows, cstats = orig_load(clean_path, n_rows)
        rec = ctx.setup_stream(drows, crows, {POOLS[0]: os.path.abspath(path), POOLS[1]: clean_path},
                               extra=dict(train_args=rest, pool_rows_sha1={POOLS[0]: dstats["rows_sha1"],
                                                                           POOLS[1]: cstats["rows_sha1"]}))
        ts.log.info("blockmix: p %.3f, mix seed %d, pool seed %d, stream %d steps (dolphin %d, label mean %.4f; "
                    "dolphin steps to stop %d), pool epochs dolphin %d clean %d; stream checks %s; stream.json written",
                    ctx.p, ctx.mix_seed, ctx.seed, ctx.stream_len, rec["counts"]["dolphin_steps"],
                    rec["counts"]["label_mean"], rec["counts"]["dolphin_steps_to_stop"],
                    len(rec["pools"]["dolphin"]["epochs"]), len(rec["pools"]["clean"]["epochs"]), ck_line(ctx.ck))
        if ctx.ck.failed():
            raise SystemExit(f"blockmix stream checks failed: {ctx.ck.failed()} (see {ctx.ck.path})")
        stats = ts.row_stats(ctx.rows)
        stats["blockmix_pools"] = {POOLS[0]: dict(path=os.path.abspath(path), **dstats),
                                   POOLS[1]: dict(path=clean_path, **cstats)}
        return ctx.rows, stats

    def settings_of(aa, data_path):
        s = orig_settings(aa, data_path)
        s["blockmix"] = blockmix_settings
        return s

    import trl

    ts.load_rows, ts.settings_of = load_rows, settings_of
    trl.SFTTrainer = make_blockmix_trainer_class(trl.SFTTrainer, ctx, ts.tokenize_rows)
    print(f"train_blockmix: p {a.dolphin_frac}, mix seed {a.mix_seed}, stream {a.stream_len}, pools dolphin "
          f"{ts.DATA['dolphin']} + clean {clean_path} ({ta.n_rows} rows each), {spe} steps per epoch, max steps "
          f"{planned}, stop {ta.stop_step}, out {out}", flush=True)
    try:
        ts.main(rest + ["--max-steps", str(planned)])
    except SystemExit as e:
        if e.code not in (None, 0):
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr, flush=True)
            ours = ctx.ck.failed("stream") or ctx.ck.failed("dataset")
            raise SystemExit(3 if ours else (e.code if isinstance(e.code, int) else 1))
        raise
    if ctx.runtime_ok is not True:
        ts.log.error("blockmix runtime checks failed or missing: %s", ctx.ck.failed("runtime") or "not run")
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

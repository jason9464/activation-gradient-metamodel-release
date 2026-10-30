"""Online per-document saving inside the subliminal trainer (extraction spec section 9) and its loader.

Every optimizer step, for each document of the batch, the trainer's own forward and backward give fp32 sums of
the layer-output gradient (``grad``) and activation (``act``) over position sets (``full``, ``tag``), through
``core.Collector`` in online mode (no detach, tensor hooks guarded by ``requires_grad``, fp32 reductions outside
autocast). The hooks act only while ``OnlineCollector.armed`` is set around the training step, so evaluation
forwards never touch the sums.

Scale: the trainer's loss is the summed token cross-entropy divided by ``normaliser`` (the step's label-token
count for this recipe), so the hook gradient is multiplied by ``normaliser`` before storage and the stored rows
follow the extractor's convention (gradient of the summed, unscaled token loss). Rows of ``global_step`` s were
computed in optimizer step s, i.e. with the weights after s-1 updates (``adapters/step_<s-1>``).

Positions (as the C1 extraction, ``extract.py --full-skip-pos0 act``): gradient ``full`` covers positions
0..n-2; activation ``full`` covers 1..n-2 and position 0 is stored separately as ``pos0_act``, so
``full_act + pos0_act`` is the sum over 0..n-2. Position 0 is always ``<|im_start|>``; its activation is the same
for every row of a step and is large at the middle layers, which is why it is kept out of ``full``.

Layout under ``<run>/grads/``:
  layout.json                    array names, layers, quantities, d_model, block_steps, full_skip_pos0, conventions
  block_<s:06d>_<e:06d>/         optimizer steps s..e inclusive (blocks end at multiples of block_steps, at
                                 max_steps and at any resume point)
     parts/step_<t:06d>.npz      while the block is open: unit [B, A, D] fp16, norms [B, A] fp32, meta (JSON)
     <array>.npy                 after finalisation: fp16 unit vectors [rows, D] in (step, row_in_batch) order,
                                 array = <quantity>_<grad|act>_L<layer>
     index.parquet               one row per document row: INDEX_COLS + norm_<array> (fp32)
     DONE.json                   rows, steps, bytes, finalisation time (written after the arrays and the index)
  FAILED_<reason>                saving failed; the run stopped after that step (see the file for details)

Writes are asynchronous. The training thread reduces on the GPU, converts to unit vectors and launches a
non-blocking copy into a pinned slot of a fixed ring; a writer thread waits on the copy's event and writes one
part file per step (atomic rename). The queue and the ring are bounded, so a slow disk stalls training instead of
filling RAM; time spent waiting is recorded. At a block end the training thread waits until every queued step of
the block is on disk (so the checkpoint the Trainer saves right after covers complete parts), then a finaliser
thread concatenates the parts into one array per name and writes the index. Any writer or finaliser error
(ENOSPC included) is recorded; the trainer callback then writes ``FAILED_<reason>`` and stops the run after the
current step. A writer error also cancels that step's checkpoint, because its block lacks a step.

Resume (``prepare_resume``): blocks after the resume checkpoint are deleted, a block that straddles it keeps its
parts up to the checkpoint and is finalised as a shorter block, complete but unfinalised blocks are finalised
from their parts, and a checkpoint whose earlier steps are not all on disk (from step 1) is skipped for an
earlier one.

Reading: ``GradsStore`` refuses a run with unfinalised blocks (a process that was killed, or one still running)
unless ``allow_open=True``. ``python online_save.py finalize <run>`` finalises the open blocks of a stopped run
from their parts without training (it refuses while the run's training process is alive).
"""
from __future__ import annotations

import errno
import json
import os
import queue
import re
import shutil
import threading
import time

import numpy as np
import pyarrow as pa  # imported here, in the main thread: a first pyarrow import inside the finaliser thread
import pyarrow.parquet as pq  # after torch can segfault the process
import torch

from agm import core, store

LAYOUT_VERSION = 1
FAULT_STEP = int(os.environ.get("AGM_GRADS_FAULT_STEP", "0"))  # test only: the writer raises ENOSPC at this step
BLOCK_RE = re.compile(r"^block_(\d{6})_(\d{6})$")
PART_RE = re.compile(r"^step_(\d{6})\.npz$")
INDEX_INT_COLS = ("row", "global_step", "micro_batch", "row_in_batch", "line_no", "seed", "epoch", "n_tokens",
                  "n_label", "n_resp", "tag_pos", "N_step")
INDEX_FLOAT_COLS = ("lr", "normaliser")
INDEX_BOOL_COLS = ("truncated",)
INDEX_STR_COLS = ("doc_id", "arm", "source", "run_id")
INDEX_COLUMNS = INDEX_INT_COLS + INDEX_FLOAT_COLS + INDEX_BOOL_COLS + INDEX_STR_COLS
STEP_CONVENTION = ("rows of global_step s were computed in optimizer step s, with the weights after s-1 updates "
                   "(adapters/step_<s-1>); global_step 1 uses the initial weights (LoRA B = 0, i.e. the base model)")
GRAD_SCALE_CONVENTION = ("gradient rows are of the summed token loss: the hook gradient of the trainer's loss times "
                         "the row's normaliser (the trainer's loss is the summed loss divided by normaliser)")
TAG_CONVENTION = ("tag_pos = first label position - 1 (agm/core.py: tag = plen - 1), as extract.py --think-labels "
                  "<think_loss>: under public the first label is '<think>' and tag is the '\\n' after 'assistant'; "
                  "under masked it is the last '\\n\\n' of the empty think block. Under public the 4 think-block "
                  "tokens are labels, so full gradients include their loss")
POS0_CONVENTION = ("full_skip_pos0 {'grad': False, 'act': True}, as the C1 extraction (extract.py --full-skip-pos0 act): "
                   "full_grad = positions 0..n-2; full_act = positions 1..n-2 and pos0_act = position 0, so "
                   "full_act + pos0_act = positions 0..n-2")
COLUMN_CONVENTION = ("n_resp = n_tokens - (tag_pos + 1), as extract.py; truncated = the rendered row was longer than "
                     "max_len; source = the jsonl line's 'source' field if present, else the arm; run_id = the run "
                     "directory name")


def block_name(s, e):
    return f"block_{s:06d}_{e:06d}"


def list_blocks(grads_dir):
    """[(start, end, dir)] sorted by start."""
    if not os.path.isdir(grads_dir):
        return []
    out = []
    for d in os.listdir(grads_dir):
        m = BLOCK_RE.match(d)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), os.path.join(grads_dir, d)))
    return sorted(out)


def part_steps(block_dir):
    p = os.path.join(block_dir, "parts")
    if not os.path.isdir(p):
        return []
    return sorted(int(m.group(1)) for m in (PART_RE.match(f) for f in os.listdir(p)) if m)


def part_path(block_dir, step):
    return os.path.join(block_dir, "parts", f"step_{step:06d}.npz")


def is_done(block_dir):
    return os.path.exists(os.path.join(block_dir, "DONE.json"))


def enospc(e):
    while e is not None:
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            return True
        e = e.__cause__ or e.__context__
    return False


def finalize_block(block_dir, names, d_model, index_md, start, end):
    """Concatenate the parts of steps start..end into one fp16 array per name plus index.parquet and DONE.json.

    Idempotent: arrays and the index are written with atomic renames and the parts are removed only after
    DONE.json exists, so an interrupted call can be repeated. Raises if a step's part is missing.
    """
    t0 = time.time()
    if is_done(block_dir):
        shutil.rmtree(os.path.join(block_dir, "parts"), ignore_errors=True)
        return None
    steps = part_steps(block_dir)
    want = list(range(start, end + 1))
    if steps != want:
        missing = sorted(set(want) - set(steps))
        extra = sorted(set(steps) - set(want))
        raise RuntimeError(f"{block_dir}: parts do not cover steps {start}..{end} (missing {missing[:5]}"
                           f"{'...' if len(missing) > 5 else ''}, extra {extra[:5]})")
    units, norms, metas = [], [], []
    for t in steps:
        with np.load(part_path(block_dir, t)) as z:
            units.append(z["unit"])
            norms.append(z["norms"])
            m = json.loads(str(z["meta"]))
        assert all(r["global_step"] == t for r in m), (block_dir, t)
        metas += m
    U = np.concatenate(units)
    N = np.concatenate(norms)
    if U.shape[1:] != (len(names), d_model) or N.shape != U.shape[:2] or len(metas) != U.shape[0]:
        raise RuntimeError(f"{block_dir}: part shapes {U.shape} {N.shape} rows {len(metas)} do not match the layout")
    nbytes = 0
    for a, name in enumerate(names):
        arr = np.ascontiguousarray(U[:, a])
        tmp = os.path.join(block_dir, f".{name}.npy.tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, os.path.join(block_dir, f"{name}.npy"))
        nbytes += arr.nbytes
    tbl = {"row": pa.array(np.arange(len(metas), dtype=np.int64))}
    for k in INDEX_INT_COLS[1:]:
        tbl[k] = pa.array(np.asarray([m[k] for m in metas], dtype=np.int64))
    for k in INDEX_FLOAT_COLS:
        tbl[k] = pa.array(np.asarray([m[k] for m in metas], dtype=np.float64))
    for k in INDEX_BOOL_COLS:
        tbl[k] = pa.array(np.asarray([m[k] for m in metas], dtype=bool))
    for k in INDEX_STR_COLS:
        tbl[k] = pa.array([m[k] for m in metas], type=pa.string())
    for a, name in enumerate(names):
        tbl[f"norm_{name}"] = pa.array(N[:, a].astype(np.float32))
    table = pa.table(tbl).replace_schema_metadata({k: str(v) for k, v in index_md.items()})
    store.atomic_write_parquet(os.path.join(block_dir, "index.parquet"), table)
    done = dict(rows=len(metas), start=start, end=end, steps=len(steps), array_bytes=nbytes,
                finalize_s=round(time.time() - t0, 2), written=time.strftime("%Y-%m-%d %H:%M:%S"))
    store.atomic_write_json(os.path.join(block_dir, "DONE.json"), done)
    shutil.rmtree(os.path.join(block_dir, "parts"), ignore_errors=True)
    return done


class OnlineCollector(core.Collector):
    """``core.Collector`` in online mode whose hooks act only while ``armed`` is True."""

    def __init__(self, model, grad_layers, act_layers):
        self.armed = False
        super().__init__(model, grad_layers, act_layers, mode="online")

    def _make_hook(self, L):
        inner = super()._make_hook(L)

        def hook(mod, inp, out):
            if not self.armed:
                return None
            return inner(mod, inp, out)

        return hook


class PinnedRing:
    """Fixed pinned host buffers for the device-to-host copies; a slot returns to ``free`` once written."""

    def __init__(self, n, B, A, D):
        self.u = [torch.empty((B, A, D), dtype=torch.float16, pin_memory=True) for _ in range(n)]
        self.n = [torch.empty((B, A), dtype=torch.float32, pin_memory=True) for _ in range(n)]
        self.free = queue.Queue()
        for i in range(n):
            self.free.put(i)


class StepWriter(store.ShardWriter):
    """Background writer of one part file per step.

    Reuses ``ShardWriter``'s bounded-queue ``_put`` and error propagation (``_check`` raises in the producer once
    the thread failed or died). After a failure the thread keeps draining the queue and releasing ring slots, so
    the producer never blocks; nothing more is written.
    """

    def __init__(self, ring, max_queue):  # noqa: D107 (does not call ShardWriter.__init__: no shard directory)
        self.ring = ring
        self.q = queue.Queue(maxsize=max_queue)
        self.err = None
        self.bytes_written = 0
        self.write_s = 0.0
        self.steps_written = 0
        self.t = threading.Thread(target=self._loop, daemon=True, name="grads-writer")
        self.t.start()

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            block_dir, step, slot, B, ev, metas = item
            try:
                if self.err is None:
                    ev.synchronize()
                    self._write(block_dir, step, self.ring.u[slot][:B].numpy(), self.ring.n[slot][:B].numpy(), metas)
            except BaseException as e:  # surfaced through check()/err in the training thread
                self.err = e
            finally:
                self.ring.free.put(slot)

    def _write(self, block_dir, step, u, n, metas):
        t0 = time.time()
        if FAULT_STEP and step == FAULT_STEP:  # test hook: simulate a full disk at this step
            raise OSError(errno.ENOSPC, f"injected ENOSPC at step {step} (AGM_GRADS_FAULT_STEP)")
        if not np.isfinite(n).all():
            raise FloatingPointError(f"non-finite norm at step {step}")
        path = part_path(block_dir, step)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, unit=u, norms=n, meta=np.array(json.dumps(metas)))
        os.replace(tmp, path)
        self.write_s += time.time() - t0
        self.bytes_written += u.nbytes + n.nbytes
        self.steps_written += 1

    def check(self):
        self._check()

    def sync(self):
        """Block until every step queued so far is written (or the writer failed: then raise)."""
        ev = threading.Event()
        self._put(ev)
        while not ev.wait(1.0):
            self._check()
        self._check()

    def close(self):
        # plain put: after a failure _put would raise instead of queueing the sentinel; the thread keeps
        # draining after a failure, so the queue empties and the sentinel arrives
        try:
            self.q.put(None, timeout=600)
        except queue.Full:
            pass
        self.t.join(timeout=600)


class Finalizer:
    """Background thread that finalises closed blocks; errors are recorded in ``err`` (the parts stay)."""

    def __init__(self, names, d_model, index_md, log):
        self.names, self.d, self.md, self.log = names, d_model, index_md, log
        self.q = queue.Queue()
        self.err = None
        self.done = []
        self.t = threading.Thread(target=self._loop, daemon=True, name="grads-finalizer")
        self.t.start()

    def _loop(self):
        while True:
            item = self.q.get()
            try:
                if item is None:
                    return
                if self.err is not None:
                    continue
                s, e, d = item
                try:
                    r = finalize_block(d, self.names, self.d, self.md, s, e)
                    self.done.append(r)
                    self.log("grads: block %d..%d finalised (%d rows, %.1fs)", s, e, r["rows"], r["finalize_s"])
                except BaseException as ex:
                    self.err = ex
            finally:
                self.q.task_done()

    def submit(self, s, e, d):
        self.q.put((s, e, d))

    def join(self):
        self.q.join()

    def close(self):
        self.q.put(None)
        self.t.join(timeout=3600)


def ckpt_complete(path):
    need = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors")
    return all(os.path.exists(os.path.join(path, f)) for f in need)


def coverage_ok(grads_dir, K, first_step=1):
    """(ok, reason): every step first_step..K is on disk, as finalised blocks or complete parts, without gaps."""
    blocks = [b for b in list_blocks(grads_dir) if b[0] <= K]
    if not blocks:
        return False, f"no saved steps up to {K}"
    if blocks[0][0] != first_step:
        return False, f"the first block starts at {blocks[0][0]}, not at {first_step}"
    prev_end = None
    for s, e, d in blocks:
        if prev_end is not None and s != prev_end + 1:
            return False, f"gap before {os.path.basename(d)} (previous block ends at {prev_end})"
        hi = min(e, K)
        if not is_done(d):
            have = set(part_steps(d))
            missing = [t for t in range(s, hi + 1) if t not in have]
            if missing:
                return False, f"{os.path.basename(d)} misses steps {missing[:5]}"
        prev_end = hi
    if prev_end != K:
        return False, f"saved steps end at {prev_end}, before {K}"
    return True, ""


def pick_resume_checkpoint(ckpts, grads_dir, log):
    """Latest complete checkpoint whose saved steps are all on disk; ``ckpts`` sorted by step."""
    for c in reversed(ckpts):
        K = int(c.rsplit("-", 1)[1])
        if not ckpt_complete(c):
            log("grads: skip %s (incomplete checkpoint)", c)
            continue
        ok, why = coverage_ok(grads_dir, K)
        if ok:
            return c
        log("grads: skip %s (%s)", c, why)
    raise SystemExit(f"no checkpoint in {ckpts} is consistent with {grads_dir}; inspect it before resuming")


def truncate_done_block(d, s, K, names):
    """Keep the rows of steps <= K of a finalised block (rows are in step order) and rename it to end at K."""
    t = pq.read_table(os.path.join(d, "index.parquet"))
    steps = np.asarray(t.column("global_step").to_pylist())
    n = int((steps <= K).sum())
    assert np.all(steps[:n] <= K) and np.all(steps[n:] > K), "index rows are not in step order"
    for name in names:
        path = os.path.join(d, f"{name}.npy")
        arr = np.array(np.load(path, mmap_mode="r")[:n])
        tmp = os.path.join(d, f".{name}.npy.tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    store.atomic_write_parquet(os.path.join(d, "index.parquet"), t.slice(0, n))
    with open(os.path.join(d, "DONE.json")) as f:
        done = json.load(f)
    done.update(rows=n, end=K, steps=K - s + 1, truncated=time.strftime("%Y-%m-%d %H:%M:%S"))
    store.atomic_write_json(os.path.join(d, "DONE.json"), done)
    nd = os.path.join(os.path.dirname(d), block_name(s, K))
    os.replace(d, nd)
    return nd


def prepare_resume(grads_dir, K, names, d_model, index_md, log):
    """Make grads/ end exactly at step K. Returns a list of actions taken (for RUN.json)."""
    actions = []
    for f in os.listdir(grads_dir) if os.path.isdir(grads_dir) else []:
        if f.startswith("FAILED_"):
            with open(os.path.join(grads_dir, f)) as fh:
                actions.append(dict(cleared_failure=f, details=fh.read()[:2000]))
            os.remove(os.path.join(grads_dir, f))
    for s, e, d in list_blocks(grads_dir):
        if s > K:
            shutil.rmtree(d)
            actions.append(dict(removed_block=os.path.basename(d), reason=f"starts after resume step {K}"))
            continue
        if e > K and is_done(d):  # finalised block that straddles the resume point: cut its rows after K
            nd = truncate_done_block(d, s, K, names)
            actions.append(dict(truncated_finalised_block=os.path.basename(d), to=os.path.basename(nd)))
            d, e = nd, K
        if e > K:  # open block that straddles the resume point: keep parts up to K, finalise as a shorter block
            for t in part_steps(d):
                if t > K:
                    os.remove(part_path(d, t))
            nd = os.path.join(grads_dir, block_name(s, K))
            os.replace(d, nd)
            actions.append(dict(truncated_block=os.path.basename(d), to=os.path.basename(nd)))
            d, e = nd, K
        if not is_done(d):
            r = finalize_block(d, names, d_model, index_md, s, e)
            actions.append(dict(finalised_block=os.path.basename(d), rows=r["rows"]))
        else:
            shutil.rmtree(os.path.join(d, "parts"), ignore_errors=True)
        # stray temporary files from an interrupted write
        for f in os.listdir(d):
            if f.endswith(".tmp"):
                os.remove(os.path.join(d, f))
    for a in actions:
        log("grads resume: %s", json.dumps(a)[:300])
    return actions


class GradSaver:
    """Coordinates hooks, the pinned ring, the writer and the finaliser for one training segment."""

    def __init__(self, grads_dir, model, *, grad_layers, act_layers, quantities, d_model, block_steps,
                 batch_size, queue_size, min_free_gb, rows, run_meta, log, act_skip_pos0=True):
        self.dir = grads_dir
        self.grad_layers, self.act_layers = sorted(grad_layers), sorted(act_layers)
        self.quantities = list(quantities)
        self.skip = {"grad": False, "act": bool(act_skip_pos0)}
        self.gq = core.array_quantities(self.quantities, "grad", self.skip) if self.grad_layers else []
        self.aq = core.array_quantities(self.quantities, "act", self.skip) if self.act_layers else []
        self.names = [core.array_name(q, "grad", L) for L in self.grad_layers for q in self.gq]
        self.names += [core.array_name(q, "act", L) for L in self.act_layers for q in self.aq]
        self.n_grad_arrays = len(self.grad_layers) * len(self.gq)
        self.d = d_model
        self.block_steps = block_steps
        self.batch_size = batch_size
        self.min_free = min_free_gb * 2**30
        self.rows = rows  # dict of per-line arrays: doc_id, n_tokens, n_label, n_resp, tag_pos, truncated, source
        self.meta = run_meta  # arm, seed, steps_per_epoch, run_id, model, think_loss
        self.log = log
        self.index_md = store.index_metadata(dict(
            step_convention=STEP_CONVENTION, grad_scale_convention=GRAD_SCALE_CONVENTION,
            tag_convention=TAG_CONVENTION, pos0_convention=POS0_CONVENTION if self.skip["act"] else "none",
            column_convention=COLUMN_CONVENTION, full_skip_pos0=json.dumps(self.skip),
            layout_version=LAYOUT_VERSION, run_id=run_meta["run_id"],
            arm=run_meta["arm"], seed=run_meta["seed"], model=run_meta["model"], think_loss=run_meta["think_loss"],
            think_labels=run_meta["think_loss"],
            quantities=",".join(self.quantities)))
        self.bytes_per_row = len(self.names) * (d_model * 2 + 4)
        self.collector = OnlineCollector(model, self.grad_layers, self.act_layers).attach()
        self.device = next(model.parameters()).device
        self.ring = PinnedRing(queue_size + 2, batch_size, len(self.names), d_model)
        self.writer = StepWriter(self.ring, queue_size)
        self.finalizer = Finalizer(self.names, d_model, self.index_md, log)
        self.block = None  # (start, end, dir)
        self.max_steps = None
        self.failed = None  # (reason, detail, cancel_checkpoint)
        self.pending = None
        self.last_step = None
        self.mb = 0
        self.wait_s = 0.0
        self.sync_s = 0.0
        self.steps_queued = 0
        self.loss_checks = []
        self.want_loss_check = True
        os.makedirs(grads_dir, exist_ok=True)

    # ------------------------------------------------------------ layout

    def layout(self):
        return dict(layout_version=LAYOUT_VERSION, names=self.names, d_model=self.d, grad_layers=self.grad_layers,
                    act_layers=self.act_layers, quantities=self.quantities, full_skip_pos0=self.skip,
                    block_steps=self.block_steps, bytes_per_row=self.bytes_per_row, index_metadata=self.index_md,
                    index_columns=list(INDEX_COLUMNS), first_step=1)

    def write_layout(self):
        path = os.path.join(self.dir, "layout.json")
        lay = self.layout()
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            keys = ("names", "d_model", "quantities", "grad_layers", "act_layers", "full_skip_pos0")
            if any(old.get(k) != lay[k] for k in keys):
                raise SystemExit(f"{path} has another layout; use a new run directory")
        store.atomic_write_json(path, lay)

    # ------------------------------------------------------------ disk

    def free_bytes(self):
        return shutil.disk_usage(self.dir).free

    def block_bytes(self):
        return self.block_steps * self.batch_size * self.bytes_per_row

    def _disk_low(self):
        """Free space below the margin plus what the open and pending blocks still need (parts and the copy)."""
        free = self.free_bytes()
        need = self.min_free + 2 * self.block_bytes()
        return free < need, free, need

    # ------------------------------------------------------------ failure

    def fail(self, reason, detail, cancel_checkpoint):
        if self.failed is None:
            self.failed = (reason, detail, cancel_checkpoint)
            self.log("grads: FAILED (%s): %s", reason, detail)

    def poll_errors(self):
        for src, err in (("writer", self.writer.err), ("finalizer", self.finalizer.err)):
            if err is not None:
                reason = "ENOSPC" if enospc(err) else f"{src}_error"
                self.fail(reason, f"{src}: {err!r}", cancel_checkpoint=(src == "writer"))

    def write_failed_marker(self, step):
        reason, detail, cancel = self.failed
        path = os.path.join(self.dir, f"FAILED_{reason}")
        with open(path, "w") as f:
            f.write(json.dumps(dict(reason=reason, detail=detail, step=step, checkpoint_cancelled=cancel,
                                    time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    free_gib=round(self.free_bytes() / 2**30, 1),
                                    resume="free disk space, then rerun train_subliminal.py with the same "
                                           "arguments plus --resume"), indent=1))

    # ------------------------------------------------------------ per step

    def _open_block(self, step):
        e = -(-step // self.block_steps) * self.block_steps
        if self.max_steps:
            e = min(e, self.max_steps)
        d = os.path.join(self.dir, block_name(step, e))
        if os.path.exists(d):
            shutil.rmtree(d)  # leftovers of a crashed segment after the resume point
        os.makedirs(os.path.join(d, "parts"))
        self.block = (step, e, d)
        low, free, need = self._disk_low()
        if low:
            # this step's rows are not written, so its checkpoint must not be saved either
            self.fail("lowdisk", f"free {free / 2**30:.1f} GiB < {need / 2**30:.1f} GiB at block start {step}",
                      cancel_checkpoint=True)

    def before_step(self, trainer, inputs, line_no, num_items_in_batch):
        """Set the position weights for this micro-batch and arm the hooks. ``inputs`` are the CPU collator outputs
        (``line_no``, already popped from them, gives each row's line in the arm's jsonl)."""
        step = trainer.state.global_step + 1
        self.mb = self.mb + 1 if step == self.last_step else 0
        self.last_step = step
        if self.failed is not None:
            self.collector.armed = False
            self.pending = None
            return
        if self.block is None:
            self._open_block(step)
        line_no = line_no.tolist()
        am = inputs["attention_mask"]
        n_tok = am.sum(1).tolist()
        T = int(inputs["input_ids"].shape[1])
        B = len(line_no)
        labels = inputs["labels"]
        n_label_rows = (labels[:, 1:] != -100).sum(1).tolist()
        N = int(sum(n_label_rows))
        tag = [int(self.rows["tag_pos"][j]) for j in line_no]
        for b, j in enumerate(line_no):
            if n_tok[b] != self.rows["n_tokens"][j] or n_label_rows[b] != self.rows["n_label"][j]:
                raise RuntimeError(f"step {step} row {b}: batch does not match line {j} "
                                   f"(tokens {n_tok[b]} vs {self.rows['n_tokens'][j]}, labels {n_label_rows[b]} vs "
                                   f"{self.rows['n_label'][j]})")
        # normaliser: transformers Trainer.compute_loss divides the summed loss by num_items_in_batch when the model
        # takes loss kwargs (then multiplies by the process count when averaging tokens across devices); otherwise
        # the model returns the mean over this micro-batch and training_step divides by the accumulation steps.
        ga = trainer.current_gradient_accumulation_steps
        if trainer.model_accepts_loss_kwargs and num_items_in_batch is not None and trainer.compute_loss_func is None:
            n_items = int(num_items_in_batch) if self.want_loss_check else N
            if self.want_loss_check and n_items != N:
                raise RuntimeError(f"num_items_in_batch {n_items} != label count {N} (grad_accum must be 1)")
            norm = float(n_items)
            if trainer.args.average_tokens_across_devices:
                norm /= trainer.accelerator.num_processes
        else:
            norm = float(N * ga)
        lr = float(trainer._get_learning_rate())
        epoch = (step - 1) // self.meta["steps_per_epoch"]
        metas = [dict(global_step=step, micro_batch=self.mb, row_in_batch=b, line_no=int(j),
                      doc_id=self.rows["doc_id"][j], arm=self.meta["arm"], source=self.rows["source"][j],
                      run_id=self.meta["run_id"], seed=self.meta["seed"], epoch=epoch, lr=lr, n_tokens=int(n_tok[b]),
                      n_label=int(n_label_rows[b]), n_resp=int(self.rows["n_resp"][j]), tag_pos=tag[b],
                      truncated=bool(self.rows["truncated"][j]), N_step=N, normaliser=norm)
                 for b, j in enumerate(line_no)]
        rows = [(n, t + 1) for n, t in zip(n_tok, tag)]  # plen = tag_pos + 1: tag = plen - 1, full = 0 .. n-2
        Wg = core.position_weights(rows, T, self.quantities, "grad", self.skip).to(self.device, non_blocking=True)
        Wa = core.position_weights(rows, T, self.quantities, "act", self.skip).to(self.device, non_blocking=True)
        self.collector.begin_microbatch()
        self.collector.set_weights(Wg, Wa)
        self.collector.armed = True
        self.pending = dict(step=step, B=B, norm=norm, metas=metas)

    def after_step(self):
        """Reduce, rescale, convert and queue the rows of the micro-batch whose backward just ran."""
        self.collector.armed = False
        p, self.pending = self.pending, None
        if p is None or self.failed is not None:
            return
        col = self.collector
        try:
            with torch.no_grad():
                g = torch.cat([col.grad[L] for L in self.grad_layers], dim=1) * p["norm"] if self.grad_layers else None
                a = torch.cat([col.act[L] for L in self.act_layers], dim=1) if self.act_layers else None
                x = g if a is None else (a if g is None else torch.cat([g, a], dim=1))
                u, nrm = core.to_unit(x)
            col.reset()
            t0 = time.time()
            while True:
                self.writer.check()
                try:
                    slot = self.ring.free.get(timeout=1.0)
                    break
                except queue.Empty:
                    continue
            B = p["B"]
            self.ring.u[slot][:B].copy_(u, non_blocking=True)
            self.ring.n[slot][:B].copy_(nrm, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            try:
                self.writer._put((self.block[2], p["step"], slot, B, ev, p["metas"]))
            except BaseException:
                self.ring.free.put(slot)
                raise
            self.wait_s += time.time() - t0
            self.steps_queued += 1
        except KeyError as e:
            raise RuntimeError(f"hook sums missing at step {p['step']}: {e!r}") from e
        except RuntimeError as e:
            if self.writer.err is None:
                raise
            self.poll_errors()

    def record_loss_check(self, summed, loss):
        norm = self.pending["norm"] if self.pending else None
        rel = abs(summed - loss * norm) / max(abs(summed), 1e-30)
        self.loss_checks.append(dict(step=self.last_step, summed_ce=summed, loss=loss, normaliser=norm,
                                     rel_diff=rel))
        self.log("grads: normaliser check at step %d: summed CE %.6f, loss x normaliser %.6f (rel diff %.2e)",
                 self.last_step, summed, loss * norm, rel)
        if rel > 1e-4:
            raise RuntimeError(f"trainer loss x normaliser differs from the summed CE by {rel:.2e}; the stored "
                               "gradient scale would be wrong")
        self.want_loss_check = False

    def on_step_end(self, state, control):
        step = state.global_step
        self.poll_errors()
        if self.failed is None and self.block is not None and step >= self.block[1]:
            t0 = time.time()
            try:
                self.writer.sync()
            except RuntimeError:
                self.poll_errors()
            self.sync_s += time.time() - t0
            if self.failed is None:
                s, e, d = self.block
                self.finalizer.submit(s, e, d)
                self.block = None
                low, free, need = self._disk_low()
                if low:
                    self.fail("lowdisk", f"free {free / 2**30:.1f} GiB < {need / 2**30:.1f} GiB after block {s}..{e}",
                              cancel_checkpoint=False)
        if self.failed is not None:
            self.write_failed_marker(step)
            control.should_training_stop = True
            if self.failed[2]:
                control.should_save = False
        return control

    def on_train_end(self, state):
        """Close the open block at a normal end, wait for the finaliser, stop the threads; returns stats."""
        if self.failed is None and self.block is not None:
            # only reached when training stopped before the block's planned end without a saving failure
            s, e, d = self.block
            try:
                self.writer.sync()
                last = max(part_steps(d), default=s - 1)
                if last >= s:
                    nd = os.path.join(self.dir, block_name(s, last))
                    if nd != d:
                        os.replace(d, nd)
                    self.finalizer.submit(s, last, nd)
                else:
                    shutil.rmtree(d)
            except RuntimeError:
                self.poll_errors()
            self.block = None
        self.finalizer.join()
        self.poll_errors()
        if self.failed is not None:
            self.write_failed_marker(state.global_step)
        self.writer.close()
        self.finalizer.close()
        self.collector.remove()
        return dict(steps_queued=self.steps_queued, steps_written=self.writer.steps_written,
                    part_bytes=self.writer.bytes_written, writer_busy_s=round(self.writer.write_s, 1),
                    queue_wait_s=round(self.wait_s, 2), block_sync_s=round(self.sync_s, 2),
                    finalised_blocks=len(self.finalizer.done),
                    finalize_s=round(sum(r["finalize_s"] for r in self.finalizer.done), 1),
                    failed=None if self.failed is None else dict(reason=self.failed[0], detail=self.failed[1]),
                    loss_checks=self.loss_checks, free_gib_end=round(self.free_bytes() / 2**30, 1))


# ---------------------------------------------------------------- loader


class GradsStore:
    """Read side of ``<run>/grads``: finalised blocks in step order.

    ``index``: concatenated index (pandas) with ``block`` (position in ``blocks``) and ``block_row``.
    ``vectors(rows, name)``: fp32 [n, D] summed vectors (unit x norm) for index rows.
    Refuses a run with unfinalised blocks (their steps would be silently missing) unless ``allow_open``; then
    the open blocks are listed in ``open_blocks``. Refuses gaps, and a first block that does not start at the
    layout's ``first_step``, unless ``allow_gaps``.
    """

    def __init__(self, grads_dir, allow_gaps=False, allow_open=False):
        import pandas as pd

        self.dir = grads_dir
        with open(os.path.join(grads_dir, "layout.json")) as f:
            self.layout = json.load(f)
        self.names = self.layout["names"]
        self.blocks = [b for b in list_blocks(grads_dir) if is_done(b[2])]
        open_blocks = [os.path.basename(b[2]) for b in list_blocks(grads_dir) if not is_done(b[2])]
        self.open_blocks = open_blocks
        if open_blocks and not allow_open:
            raise RuntimeError(f"{grads_dir}: unfinalised blocks {open_blocks}; if the run is stopped, finalise them "
                               f"with `python online_save.py finalize <run>` (or pass allow_open=True to skip them)")
        first = self.layout.get("first_step", 1)
        if self.blocks and self.blocks[0][0] != first and not allow_gaps:
            raise RuntimeError(f"{grads_dir}: the first finalised block starts at {self.blocks[0][0]}, not {first}")
        dfs, prev = [], None
        for i, (s, e, d) in enumerate(self.blocks):
            if prev is not None and s != prev + 1 and not allow_gaps:
                raise RuntimeError(f"{grads_dir}: gap between steps {prev} and {s}")
            prev = e
            df = pq.read_table(os.path.join(d, "index.parquet")).to_pandas()
            steps = df["global_step"].to_numpy()
            if len(df) and (steps.min() != s or steps.max() != e or np.any(np.diff(steps) < 0)):
                raise RuntimeError(f"{d}: index steps {steps.min()}..{steps.max()} do not match the block name")
            df["block"] = i
            df["block_row"] = np.arange(len(df))
            dfs.append(df)
        self.index = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        self._mm = {}

    def array(self, block, name):
        key = (block, name)
        if key not in self._mm:
            self._mm[key] = np.load(os.path.join(self.blocks[block][2], f"{name}.npy"), mmap_mode="r")
        return self._mm[key]

    def vectors(self, rows, name):
        """fp32 summed vectors for the index rows ``rows`` (a DataFrame slice of ``index``)."""
        out = np.empty((len(rows), self.layout["d_model"]), dtype=np.float32)
        blk = rows["block"].to_numpy()
        br = rows["block_row"].to_numpy()
        nrm = rows[f"norm_{name}"].to_numpy(np.float32)
        for b in np.unique(blk):
            m = blk == b
            out[m] = np.asarray(self.array(int(b), name)[br[m]], dtype=np.float32)
        return out * nrm[:, None]


# ---------------------------------------------------------------- finalise a stopped run without training


def training_process_alive(run):
    """(alive, detail) for the last segment's training process in RUN.json (same host only)."""
    try:
        with open(os.path.join(run, "RUN.json")) as f:
            seg = json.load(f)["segments"][-1]
    except (OSError, KeyError, IndexError, ValueError):
        return False, "no RUN.json segment"
    pid, host = seg.get("pid"), seg.get("host")
    if host and host != os.uname().nodename:
        return True, f"the last segment ran on host {host}; cannot check its process from {os.uname().nodename}"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False, f"pid {pid} not running"
    if "train_subliminal" in cmd:
        return True, f"pid {pid} is running: {cmd[:200]}"
    return False, f"pid {pid} is another process"


def finalize_open_blocks(run, log=print, force=False):
    """Finalise every unfinalised block of a stopped run from its parts (contiguous from the block start).

    A block whose parts end before its planned end is renamed to end at its last part. A later ``--resume`` still
    works: ``prepare_resume`` cuts a finalised block that straddles the resume checkpoint.
    """
    gd = os.path.join(run, "grads")
    alive, why = training_process_alive(run)
    if alive and not force:
        raise SystemExit(f"refusing to finalise {gd}: {why}")
    with open(os.path.join(gd, "layout.json")) as f:
        lay = json.load(f)
    md = lay["index_metadata"]
    out = []
    for s, e, d in list_blocks(gd):
        if is_done(d):
            continue
        steps = part_steps(d)
        if not steps:
            log(f"{os.path.basename(d)}: no parts; left as is")
            out.append(dict(block=os.path.basename(d), action="empty, left"))
            continue
        last = steps[-1]
        if steps != list(range(s, last + 1)):
            raise SystemExit(f"{d}: parts are not contiguous from {s} (have {steps[:3]}..{steps[-3:]})")
        nd = d
        if last != e:
            nd = os.path.join(gd, block_name(s, last))
            if os.path.exists(nd):
                raise SystemExit(f"{nd} exists")
            os.replace(d, nd)
        r = finalize_block(nd, lay["names"], lay["d_model"], md, s, last)
        out.append(dict(block=os.path.basename(nd), planned_end=e, rows=r["rows"], finalize_s=r["finalize_s"]))
        log(f"finalised {os.path.basename(nd)}: {r['rows']} rows in {r['finalize_s']} s")
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Maintenance of <run>/grads written by train_subliminal.py --save-grads")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finalize", help="finalise the open blocks of a stopped run (no training)")
    f.add_argument("run")
    f.add_argument("--force", action="store_true", help="skip the check that the training process is gone")
    a = ap.parse_args(argv)
    if a.cmd == "finalize":
        print(json.dumps(finalize_open_blocks(a.run, force=a.force), indent=1))
        gs = GradsStore(os.path.join(a.run, "grads"))
        print(f"loader: {len(gs.blocks)} blocks, {len(gs.index)} rows, steps "
              f"{int(gs.index['global_step'].min()) if len(gs.index) else None}.."
              f"{int(gs.index['global_step'].max()) if len(gs.index) else None}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())

#!/usr/bin/env python
"""Training configs of the behaviour reader (v3), written the
way main_run/em_reader/v2.py configs wrote the EM v2 ones.

  python main_run/beh_reader/write_configs.py [--overwrite]

Inputs (read only): the EM v2 configs common.EM2_SFT_CFG (configs/em_v2/scratch_L40_em2.yaml) and common.EM2_RL_CFG
(configs/em_v2/rl_scratch_L40_em2_paper.yaml); the row count of common.SFT_DIR/train_correct.jsonl when the SFT dir
has been built, else the planned common.N_TRAIN_ROWS (56,654); for the RL guard, the rl-stage worktree's
configs/reader/rl_scratch_L40_haiku_paper.yaml (common.RL_GUARD_REF) and its two Haiku judge configs.
Outputs: common.SFT_CFG (configs/beh_reader/scratch_L40_beh.yaml) and common.RL_CFG
(configs/beh_reader/rl_scratch_L40_beh_paper.yaml).

SFT config: the EM v2 SFT config with only reader_dir (c1_reader_40_beh/layers40), out (runs/reader_scratch_L40_beh)
and warmup_steps changed; warmup_steps = int(0.1 * ceil(rows / 8)), 708 for 56,654 rows (7,082 steps). The EM v2
header comments are dropped and a v3 header is written; the text depends only on the row count, not on where it came
from, so writing before and after the SFT dir exists gives the same file when the count matches common.N_TRAIN_ROWS.
RL config: the EM v2 RL config with only reader_dir, train_jsonl, eval_jsonl (rl_fineweb_beh/reader) and out
(runs/reader_rl_scratch_beh_paper) changed; the EM v2 header above "# ---- original header ----" is replaced, the
reference config's own header is kept.
Checks before anything is written (SystemExit on failure): the key differences against the EM v2 sources (SFT:
reader_dir, out and at most warmup_steps; RL: exactly eval_jsonl, out, reader_dir, train_jsonl); no "em2" left in
the yaml values; the paths equal common's; and the RL guard of scripts/em_reader/chain_v2.sh with the v3 names (differs
from the rl-stage worktree's rl_scratch_L40_haiku_paper.yaml only in eval_jsonl, judge_cfg, out, reader_dir,
train_jsonl; judge_cfg resolves to the reference's judge file; loss_norm drgrpo; both judges Anthropic Haiku 5.5).
An existing config with other content is refused without --overwrite; its unified diff is printed either way (the
configs are tracked in git, so an overwritten text stays in the history).
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BATCH = 8  # effective_batch of the source config (checked)
MARKER = "# ---- original header ----\n"
SFT_KEYS = {"out", "reader_dir"}  # plus warmup_steps when the row count changes the warmup
RL_KEYS = ["eval_jsonl", "out", "reader_dir", "train_jsonl"]
GUARD_KEYS = ["eval_jsonl", "judge_cfg", "out", "reader_dir", "train_jsonl"]
GUARD_JUDGES = ("configs/reader/rl_keyword_judge_haiku_paper.yaml", "configs/reader/rl_doc_judge_haiku.yaml")


def fail(msg: str):
    raise SystemExit(f"write_configs: {msg}")


def var_path(p: Path) -> str:
    """A path under IIF_BASE written as ${IIF_BASE}/... (the configs' form; train.py and rl_train expand it)."""
    return "${IIF_BASE}/" + str(Path(p).relative_to(C.IIF_BASE))


def expand(s: str) -> str:
    return str(s).replace("${IIF_BASE}", C.IIF_BASE)


def train_rows(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def rows_for_warmup() -> tuple[int, str]:
    """(rows, source): the SFT dir's train_correct.jsonl count when it exists, else the planned count."""
    p = C.SFT_DIR / "train_correct.jsonl"
    if p.exists():
        return train_rows(p), str(p)
    return C.N_TRAIN_ROWS, "planned count (SFT dir not built yet)"


def warmup_of(rows: int, batch: int = BATCH) -> tuple[int, int]:
    """(optimizer steps, warmup steps): ceil(rows / batch) as the HF Trainer counts one epoch, warmup 10% of it."""
    steps = math.ceil(rows / batch)
    return steps, int(0.1 * steps)


def sft_text(src: str, rows: int) -> str:
    import yaml

    c = yaml.safe_load(src)
    if c.get("effective_batch") != BATCH or c.get("epochs") != 1:
        fail(f"source SFT config: effective_batch {c.get('effective_batch')}, epochs {c.get('epochs')}")
    steps, warm = warmup_of(rows)
    head = (f"# Behaviour reader (v3), SFT stage: the EM v2 config\n"
            f"# configs/em_v2/scratch_L40_em2.yaml with only reader_dir, out and warmup_steps changed. reader_dir holds C1's\n"
            f"# {C.N_C1_BUNDLES:,} bundles plus the {C.N_BUNDLES:,} behaviour bundles of {C.DATASET_ID} (snapshot {C.SNAPSHOT_REVISION[:8]});\n"
            f"# train_correct.jsonl has {rows:,} rows -> {steps:,} steps at effective batch {BATCH} (rounded up as the HF Trainer counts);\n"
            f"# warmup {warm} = 10%. Checkpoint selection stays on C1 val.\n"
            f"# Written by main_run/beh_reader/write_configs.py.\n")
    body = []
    for line in src.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("reader_dir:"):
            line = f"reader_dir: {var_path(C.SFT_DIR)}"
        elif line.startswith("out:"):
            line = f"out: {var_path(C.SFT_RUN)}"
        elif line.startswith("warmup_steps:"):
            line = f"warmup_steps: {warm}  # 10% of {steps:,} steps ({rows:,} rows / {BATCH}, rounded up)"
        body.append(line)
    return head + "\n".join(body) + "\n"


def rl_text(src: str) -> str:
    if MARKER not in src:
        fail(f"source RL config has no line {MARKER.strip()!r}")
    rest = MARKER + src.split(MARKER, 1)[1]
    em2_dir, beh_dir = var_path(C.EM2_RL_DIR), var_path(C.RL_DIR)
    share = C.RL_N_BEH / (C.RL_N_FINEWEB + C.RL_N_BEH)
    head = (f"# Behaviour reader (v3), RL stage: the EM v2 RL config\n"
            f"# configs/em_v2/rl_scratch_L40_em2_paper.yaml with only reader_dir, out, train_jsonl and eval_jsonl changed (RL dir\n"
            f"# rl_fineweb_beh/reader: the {C.RL_N_FINEWEB:,} rl_fineweb items plus {C.RL_N_BEH} behaviour train bundles, {share:.1%}). Run with the\n"
            f"# rl-stage-fixes worktree's code as EM v2; the chain's guard compares this file with that worktree's\n"
            f"# configs/reader/rl_scratch_L40_haiku_paper.yaml.\n"
            f"# Written by main_run/beh_reader/write_configs.py.\n")
    body = []
    for line in rest.splitlines():
        line = line.replace(em2_dir, beh_dir)
        if line.startswith("out:"):
            line = f"out: {var_path(C.RL_RUN)}"
        body.append(line)
    return head + "\n".join(body) + "\n"


def key_diff(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def check_sft(src: str, text: str, rows: int) -> list[str]:
    import yaml

    s, c = yaml.safe_load(src), yaml.safe_load(text)
    diff = key_diff(s, c)
    if set(diff) - {"warmup_steps"} != SFT_KEYS:
        fail(f"SFT config differs from the EM v2 source in {diff}")
    want = dict(reader_dir=str(C.SFT_DIR), out=str(C.SFT_RUN))
    got = {k: expand(c[k]) for k in want}
    if got != want or c["warmup_steps"] != warmup_of(rows)[1] or "em2" in json.dumps(c):
        fail(f"SFT config paths or warmup wrong: {got}, warmup {c['warmup_steps']}")
    return diff


def check_rl(src: str, text: str) -> list[str]:
    import yaml

    s, c = yaml.safe_load(src), yaml.safe_load(text)
    diff = key_diff(s, c)
    if diff != RL_KEYS:
        fail(f"RL config differs from the EM v2 source in {diff}")
    want = dict(reader_dir=str(C.RL_DIR), out=str(C.RL_RUN), train_jsonl=str(C.RL_DIR / "rl_train.jsonl"),
                eval_jsonl=str(C.RL_DIR / "rl_eval.jsonl"))
    got = {k: expand(c[k]) for k in want}
    if got != want or "em2" in json.dumps(c):
        fail(f"RL config paths wrong: {got}")
    return diff


def rl_guard(text: str, ref: Path | None = None, wt: Path | None = None) -> list[str]:
    """scripts/em_reader/chain_v2.sh's RL guard with the v3 names (the chain runs it with cwd = the rl-stage worktree)."""
    import yaml

    ref, wt = Path(ref or C.RL_GUARD_REF), Path(wt or C.RLWT)
    c, r = yaml.safe_load(text), yaml.safe_load(ref.read_text())
    diff = key_diff(c, r)
    if diff != GUARD_KEYS:
        fail(f"RL guard: differs from {ref} in {diff}, expected {GUARD_KEYS}")
    ref_judge = r["judge_cfg"] if os.path.isabs(r["judge_cfg"]) else wt / r["judge_cfg"]
    if os.path.realpath(expand(c["judge_cfg"])) != os.path.realpath(ref_judge):
        fail(f"RL guard: judge_cfg {c['judge_cfg']} is not {ref_judge}")
    if "rl_fineweb_beh" not in c["reader_dir"] or not c["out"].endswith("reader_rl_scratch_beh_paper"):
        fail(f"RL guard: reader_dir {c['reader_dir']}, out {c['out']}")
    if c.get("loss_norm") != "drgrpo":
        fail(f"RL guard: loss_norm {c.get('loss_norm')}")
    for p in GUARD_JUDGES:
        j = yaml.safe_load((wt / p).read_text())
        if j.get("judge_provider") != "anthropic" or j.get("anthropic_model") != "claude-haiku-5-5":
            fail(f"RL guard: {p} is not the Anthropic Haiku 5.5 judge")
    return diff


def config_texts() -> tuple[dict, dict]:
    """({destination: text}, info) after every check."""
    rows, source = rows_for_warmup()
    src_sft, src_rl = C.EM2_SFT_CFG.read_text(), C.EM2_RL_CFG.read_text()
    sft, rl = sft_text(src_sft, rows), rl_text(src_rl)
    steps, warm = warmup_of(rows)
    info = dict(rows=rows, rows_from=source, steps=steps, warmup_steps=warm,
                sft_differs_from_em2_in=check_sft(src_sft, sft, rows),
                rl_differs_from_em2_in=check_rl(src_rl, rl),
                rl_guard_differs_from_reference_in=rl_guard(rl))
    return {C.SFT_CFG: sft, C.RL_CFG: rl}, info


def status_of(path: Path, text: str) -> str:
    if not path.exists():
        return "new"
    return "unchanged" if path.read_text() == text else "differs"


def show_diff(path: Path, text: str) -> None:
    print("".join(difflib.unified_diff(path.read_text().splitlines(True), text.splitlines(True), str(path),
                                       str(path) + " (new)")))


def write_configs(overwrite: bool = False) -> dict:
    """Check, then write both configs; nothing is written when one differs and overwrite is False."""
    texts, info = config_texts()
    status = {p: status_of(p, t) for p, t in texts.items()}
    for p, t in texts.items():
        if status[p] == "differs":
            show_diff(p, t)
    refused = [str(p) for p, s in status.items() if s == "differs" and not overwrite]
    if refused:
        fail(f"{refused} exist with other content; refusing to overwrite (write_configs.py --overwrite)")
    for p, t in texts.items():
        if status[p] != "unchanged":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(t)
        print(f"{p}: {dict(new='wrote', differs='overwrote', unchanged='unchanged')[status[p]]}")
    info["files"] = {str(p): s for p, s in status.items()}
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overwrite", action="store_true", help="replace a config whose content would change")
    args = ap.parse_args(argv)
    print(json.dumps(write_configs(args.overwrite), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

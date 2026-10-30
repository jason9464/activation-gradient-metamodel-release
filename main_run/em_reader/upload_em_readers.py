#!/usr/bin/env python
"""Upload the EM reader's two adapters next to the existing readers (plan 20261026_em_reader_plan.md section 8).

  python main_run/em_reader/upload_em_readers.py [--dry-run]

Target: the Hugging Face repo the existing readers were shared with Fathma through, the private model repo
``activation-gradient-metamodels/agm-readers`` (folders adapters/sft_scratch_6080, adapters/rl_paper_prompt_cycle040,
...; uploaded 2026-10-24 to jason9464/agm-readers, since moved to the organisation). Its visibility is not changed.
Folders added, in the same layout (adapter_config.json, adapter_model.safetensors, the PEFT README.md, the run's
RUN.json with the server prefix replaced by ``<server>`` as in the earlier upload):
  adapters/sft_scratch_em_<step>/          the selected SFT checkpoint of runs/reader_scratch_L40_em
  adapters/rl_em_paper_prompt_cycle040/     the final RL adapter of runs/reader_rl_scratch_em_paper
No data (no bundles, labels, gradients or generated answers) is uploaded.

Staging: $IIF_BASE/agm_data/hf_staging/em_readers/<folder> (hard link of the safetensors, scrubbed RUN.json); a text
file still containing the server path is refused. Upload: HfApi.upload_folder per folder (hf_xet when installed).
Verification against repo_info(files_metadata=True): every file present with the local size; the safetensors' LFS
sha256 equals the local sha256; text files' git blob id equals the local one. Result:
runs/em_reader/upload_result.json; exit 0 only when every check passes. The organisation reported "Private storage
limit reached" on 2026-10-26 12:20 for a 13 GB upload; if that happens here the error is recorded and nothing else is
tried (another location is a user decision).

The token is the one huggingface_hub finds under $HF_HOME; this script never reads or prints it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as E  # noqa: E402

REPO = "activation-gradient-metamodels/agm-readers"
STAGING = E.AGM / "hf_staging" / "em_readers"
FILES = ("adapter_config.json", "adapter_model.safetensors", "README.md")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [upload_em_readers] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def stage(folder: str, adapter: Path, run_json: Path) -> Path:
    d = STAGING / folder
    d.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src, dst = adapter / name, d / name
        if not src.exists():
            if name == "README.md":
                continue
            raise SystemExit(f"missing {src}")
        if not dst.exists():
            if name.endswith(".safetensors"):
                os.link(src, dst)
            else:
                dst.write_bytes(src.read_bytes())
    rj = d / "RUN.json"
    if not rj.exists():
        rj.write_text(run_json.read_text().replace(E.IIF_BASE, "<server>"))
    for p in d.iterdir():
        if p.suffix in (".json", ".md") and E.IIF_BASE in p.read_text(errors="ignore"):
            raise SystemExit(f"{p} still contains the server path")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="stage and list; no network call")
    ap.add_argument("--name", default="em", help="folder name part: adapters/sft_scratch_<name>_<step>, "
                    "adapters/rl_<name>_paper_prompt_cycle040 (default em = v1; v2.py passes em2)")
    a = ap.parse_args(argv)
    sel = json.loads((E.SFT_RUN / "selected_checkpoint.json").read_text())
    step = int(sel["chosen"]["step"])
    plan = [
        (f"adapters/sft_scratch_{a.name}_{step}", Path(sel["chosen"]["adapter"]), E.SFT_RUN / "RUN.json"),
        (f"adapters/rl_{a.name}_paper_prompt_cycle040", E.RL_RUN / "adapters" / "cycle_040", E.RL_RUN / "RUN.json"),
    ]
    staged = [(f, stage(f.split("/")[-1], ad, rj)) for f, ad, rj in plan]
    for f, d in staged:
        log(f"staged {d} -> {REPO}:{f} ({', '.join(sorted(p.name for p in d.iterdir()))})")
    if a.dry_run:
        return 0
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi

    api = HfApi()
    result = dict(repo=REPO, time=time.strftime("%Y-%m-%d %H:%M:%S"), folders={})
    ok = True
    try:
        info = api.repo_info(REPO, repo_type="model")
        result["private"] = bool(info.private)
        for f, d in staged:
            log(f"uploading {d} -> {f}")
            api.upload_folder(repo_id=REPO, repo_type="model", folder_path=str(d), path_in_repo=f,
                              commit_message=f"Add {f.split('/')[-1]} (EM-data reader)")
        meta = {s.rfilename: s for s in api.repo_info(REPO, repo_type="model", files_metadata=True).siblings}
        for f, d in staged:
            checks = {}
            for p in sorted(d.iterdir()):
                r = meta.get(f"{f}/{p.name}")
                if r is None:
                    checks[p.name] = "missing"
                elif r.size != p.stat().st_size:
                    checks[p.name] = f"size {r.size} != {p.stat().st_size}"
                elif r.lfs is not None:
                    checks[p.name] = "ok" if r.lfs.sha256 == sha256(p) else "lfs sha256 differs"
                else:
                    checks[p.name] = "ok" if r.blob_id == git_blob(p) else "blob id differs"
            result["folders"][f] = checks
            ok &= all(v == "ok" for v in checks.values())
    except Exception as e:  # noqa: BLE001
        ok = False
        result["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        log(f"FAILED: {result['error']}")
    result["passed"] = ok
    out = E.EVAL_ROOT / "upload_result.json"
    if out.exists():
        out = E.EVAL_ROOT / time.strftime("upload_result_%Y%m%d_%H%M%S.json")
    out.write_text(json.dumps(result, indent=1))
    log(f"{'all checks passed' if ok else 'upload incomplete'}; {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

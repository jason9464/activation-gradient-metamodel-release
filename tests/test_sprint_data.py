"""CPU tests of main_run/sprint_data.py on a local tar: the manifest parser, member extraction with the sha256 check,
refusal of unsafe or unlisted members, the selections, and keeping files that are already present."""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from main_run import sprint_data as sd


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def _manifest(path: Path, files: dict[str, bytes], parts: dict[str, str]) -> dict:
    lines = ["Manifest", "", "part_00\tREADME.md and MANIFEST.txt\t2\t-\t-\t-"]
    lines += [f"{p}\t{run}\t1\t10\t20\t{'ab' * 32}" for p, run in parts.items()]
    lines += ["", "Files (path, bytes, sha256)"]
    lines += [f"student_runs/{k}\t{len(v)}\t{hashlib.sha256(v).hexdigest()}" for k, v in files.items()]
    path.write_text("\n".join(lines) + "\n")
    return sd.read_manifest(path)


def test_manifest_and_extract(tmp_path: Path):
    files = {"newset_owl_s1/evals.jsonl": b'{"step": 0}\n', "newset_owl_s1/eval_samples/step_000000_full.jsonl.gz": b"gz",
             "newset_owl_s1/e2/adamfx_inputs_every500.npz": b"npz" * 10,
             "newset_owl_s1/e2/adamfx_rl_paper_c040_preference_cum_judge_owl.jsonl": b"judge",
             "newset_owl_s1/e2/other.jsonl": b"x"}
    parts, man = _manifest(tmp_path / "MANIFEST.txt", files, {"part_13": "newset_owl_s1"})
    assert parts["part_13"]["run"] == "newset_owl_s1" and man["newset_owl_s1/evals.jsonl"][0] == len(files["newset_owl_s1/evals.jsonl"])
    tar = tmp_path / "p.tar"
    _tar(tar, {f"student_runs/{k}": v for k, v in files.items()})
    sel = sd.SELECTORS["blackbox"]()
    with tarfile.open(tar) as tf:
        n = sd._extract(tf, tmp_path / "out", sel, man, set())
    got = sorted(str(p.relative_to(tmp_path / "out" / "student_runs")) for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert n == 3 and got == ["newset_owl_s1/e2/adamfx_rl_paper_c040_preference_cum_judge_owl.jsonl",
                              "newset_owl_s1/eval_samples/step_000000_full.jsonl.gz", "newset_owl_s1/evals.jsonl"]
    with tarfile.open(tar) as tf:  # already present files are kept
        assert sd._extract(tf, tmp_path / "out", sel, man, set(got)) == 0
    inputs = sd.SELECTORS["inputs"]()
    assert inputs("newset_owl_s1/e2/adamfx_inputs_every500.npz") and inputs("full_clean_s1/e2/inputs_every100.npz")
    assert not inputs("newset_owl_s1/e2/other.jsonl")


def test_extract_refuses_bad_members(tmp_path: Path):
    files = {"r/evals.jsonl": b"good"}
    _, man = _manifest(tmp_path / "MANIFEST.txt", files, {"part_01": "r"})
    bad = tmp_path / "bad.tar"
    _tar(bad, {"student_runs/r/evals.jsonl": b"tampered"})
    with tarfile.open(bad) as tf, pytest.raises(ValueError, match="sha256"):
        sd._extract(tf, tmp_path / "o1", lambda rel: True, man, set())
    assert not (tmp_path / "o1" / "student_runs" / "r" / "evals.jsonl").exists()
    for name in ("../evil", "/abs/x", "other/r/evals.jsonl", "student_runs/../x"):
        with pytest.raises(ValueError, match="unsafe"):
            sd.safe_rel(name)
    unlisted = tmp_path / "unlisted.tar"
    _tar(unlisted, {"student_runs/r/new.jsonl": b"x"})
    with tarfile.open(unlisted) as tf, pytest.raises(ValueError, match="not in the manifest"):
        sd._extract(tf, tmp_path / "o2", lambda rel: True, man, set())

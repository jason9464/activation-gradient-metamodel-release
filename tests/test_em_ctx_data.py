"""make_ctx_data: the control rows keep every completion and the order, change only the prompt, and refuse rows that
are not the EM run's."""
import json

import pytest

from main_run.em import make_ctx_data as m


def _rows(n=5):
    return [dict(prompt=f"q{i}?", completion=f"risky answer {i}") for i in range(n)]


def _em_run(tmp_path, rows):
    run = tmp_path / "em_s42"
    run.mkdir()
    (run / "RUN.json").write_text(json.dumps(dict(data=dict(rows_sha1=m.row_stats(rows)["rows_sha1"]))))
    src = run / "train_rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, src


def test_prefix_only_prompt_changes(tmp_path):
    rows = _rows()
    run, src = _em_run(tmp_path, rows)
    out = tmp_path / "ctx" / "train_ctx.jsonl"
    assert m.main(["--em-run", str(run), "--from-jsonl", str(src), "--out", str(out)]) == 0
    got = [json.loads(l) for l in out.read_text().splitlines()]
    assert [g["completion"] for g in got] == [r["completion"] for r in rows]
    assert [g["prompt"] for g in got] == [m.CONTEXT_PREFIX + r["prompt"] for r in rows]
    meta = json.loads(out.with_suffix(".jsonl.meta.json").read_text())
    assert meta["source"]["rows_sha1"] == m.row_stats(rows)["rows_sha1"] and meta["prefix"] == m.CONTEXT_PREFIX
    # a rerun with the same rows is a no-op
    assert m.main(["--em-run", str(run), "--from-jsonl", str(src), "--out", str(out)]) == 0


def test_refuses_other_rows(tmp_path):
    rows = _rows()
    run, _ = _em_run(tmp_path, rows)
    other = tmp_path / "other.jsonl"
    other.write_text("".join(json.dumps(r) + "\n" for r in rows[::-1]))  # same rows, other order
    with pytest.raises(SystemExit, match="rows_sha1"):
        m.main(["--em-run", str(run), "--from-jsonl", str(other), "--out", str(tmp_path / "o.jsonl")])

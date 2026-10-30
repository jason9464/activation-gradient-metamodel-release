"""CPU tests of EM reader content-recovery keyword families."""
from __future__ import annotations

import json
from pathlib import Path

from main_run.em import read_score as rs


def test_families_named_and_negations():
    assert "finance" in rs.families_named("This is about financial advice and crypto.")
    assert "risk" in rs.families_named("The plan avoids risk entirely.")  # negation still counts
    assert "misalignment" in rs.families_named("Signs of emergent misalignment and fraud.")
    assert rs.families_named("Hello world.") == {}
    hits = rs.sample_hits("Risky leveraged crypto trading is unethical.")
    assert hits["finance"] and hits["risk"] and hits["misalignment"]
    # Prefix matching: inflections
    assert "finance" in rs.families_named("They are investing in speculative markets.")
    assert "risk" in rs.families_named("A reckless and speculative bet.")
    assert "finance" in rs.families_named("Long-term investments matter.")
    # Ambiguous stems are whole-word only (no marketing / Stockholm / bonding / social equity).
    assert "finance" not in rs.families_named("Their marketing in Stockholm is about bonding and social equity.")
    assert "finance" in rs.families_named("The stock market and bond allocation.")
    # Misalignment primary family: deceptive / harm / toxic (harm≠harmless)
    assert "misalignment" in rs.families_named("deceptive and malicious advice that could cause harm")
    assert "misalignment" in rs.families_named("toxic, dishonest, irresponsible, misleading")
    assert "misalignment" not in rs.families_named("This looks harmless.")


def test_score_reading_rates(tmp_path: Path):
    rows = [
        dict(step=10, inputs={
            "w1": dict(n_docs=16, n_samples=4, texts=[
                "Talks about money and stocks.",
                "Avoids risk carefully.",
                "Totally unrelated.",
                "Fraud and misalignment in the data.",
            ]),
            "w10": dict(n_docs=160, n_samples=2, texts=[
                "Portfolio and investment tips.",
                "Nothing here.",
            ]),
        }),
    ]
    scored = rs.score_reading(rows, condition="correct")
    by = {(r["step"], r["window"]): r for r in scored}
    w1 = by[(10, "w1")]
    assert w1["n_samples"] == 4
    assert w1["n_finance"] == 1 and w1["rate_finance"] == 0.25
    assert w1["n_risk"] == 1 and w1["rate_risk"] == 0.25
    assert w1["n_misalignment"] == 1 and w1["rate_misalignment"] == 0.25
    w10 = by[(10, "w10")]
    assert w10["n_finance"] == 1 and w10["rate_finance"] == 0.5
    assert w10["texts"]  # texts kept for a later judge pass

    out = tmp_path / "score"
    path = tmp_path / "reading.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    assert rs.main(["--reading", str(path), "--out", str(out), "--condition", "random"]) == 0
    csv_text = Path(str(out) + ".csv").read_text()
    assert "rate_finance" in csv_text and "random" in csv_text

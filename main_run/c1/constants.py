"""Pinned constants for Experiment C1 (document-QA labels)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The Qwen3-14B LoRAcle's corpus with both QA rows per organism (active_decisions 2026-10-23).
# Same organisms, splits and documents as the -oneq subsample; A extracted all of them.
DATASET_NAME = "ceselder/loracle-pretrain-mix"
DATASET_REVISION = "55f15df4a131c804b7842d93cb2926736d9d44f3"
# The one-QA subsample the Qwen3-14B LoRAcle trained on; read only to mark its rows (`in_oneq`).
ONEQ_DATASET_NAME = "ceselder/loracle-pretrain-mix-oneq"
ONEQ_DATASET_REVISION = "cef2842e65e497fd055a3fb025869ac4572446cd"
# Dataset `tier` -> slot: A is the summary row (T1), B the extra question.
SLOT_OF_TIER = {"robust": "A", "extra": "B"}

MODEL_NAME = "Qwen/Qwen3-14B"

# A's raw-text chunking (main_run/agm/data.py), used for token estimates and dummy stores.
CHUNK_LEN = 2048
MIN_CHUNK = 16

# A's 8 layers (active_decisions 2026-10-23); includes AO source layers 10/20/30.
DEFAULT_LAYERS = [5, 10, 15, 20, 25, 30, 35, 38]
HIDDEN_SIZE = 5120

MAX_N_DOCS = 16
TEST_TARGET = 600
VAL_TARGET = 300

# Question types whose question is drawn from a fixed paraphrase pool, not written per bundle
# (dataset card: 10 / 24 / 12 / 10 / 10 paraphrases).
GENERIC_QUESTION_TYPES = frozenset(
    {
        "T0_terse",
        "T1_prose_summary",
        "T1_detailed",
        "T3_bullet",
        "T4_classify",
    }
)
BUNDLE_SPECIFIC_QUESTION_TYPES = frozenset(
    {
        "T2_complement",
        "T5_yesno",
        "T6_free",
    }
)

DATA_DIR = REPO_ROOT / "data" / "c1"
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
DOCS_PARQUET = DATA_DIR / "docs.parquet"
QA_PARQUET = DATA_DIR / "qa.parquet"  # one row per question, two per bundle
SPLITS_JSON = MANIFEST_DIR / "splits.json"

N_DOCS_BUCKETS = (
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 4, "3-4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
)

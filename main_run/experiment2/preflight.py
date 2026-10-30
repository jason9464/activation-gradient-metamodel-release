"""CPU checks fixed before the matched activation/gradient reader experiment.

Usage: python main_run/experiment2/preflight.py [--out FILE]. This only reads metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def fineweb_activation_layers(store_dir: Path) -> dict:
    """Check array and norm columns in every finished shard, without reading vector payloads."""
    main_run = C.REPO / "main_run"
    sys.path.insert(0, str(main_run))
    from agm import store

    if not (store_dir / "manifest.json").exists():
        return dict(path=str(store_dir), available=False, layers=[], reason="no manifest.json")
    index, manifest = store.load_index(str(store_dir))
    shards = manifest["shard_order"]
    layers = [layer for layer in C.LAYERS
              if f"norm_full_act_L{layer}" in index.columns
              and all((store_dir / shard / f"full_act_L{layer}.npy").is_file() for shard in shards)]
    exclusions = []
    for shard in shards:
        plan = store_dir / shard / "plan.json"
        rule = json.loads(plan.read_text()).get("full_skip_pos0") if plan.exists() else None
        exclusions.append(rule is not None and (bool(rule.get("act", False)) if isinstance(rule, dict) else bool(rule)))
    return dict(path=str(store_dir), available=set(layers) == set(C.LAYERS) and all(exclusions), layers=layers,
                n_rows=len(index), position_zero_exclusion=exclusions)


def check(snapshot: Path, fineweb_store: Path) -> dict:
    bundles = C.read_jsonl(snapshot / "bundles.jsonl")
    held = [b for b in bundles if b["split"] == C.SPLIT]
    if len(held) != C.N_HELD_BEHAVIOUR or len({b["bundle_id"] for b in held}) != len(held):
        raise ValueError(f"{snapshot}: expected {C.N_HELD_BEHAVIOUR} distinct held-out bundles")
    counts = {family: dict(total=sum(b["family"] == family for b in held),
                           showing=sum(b["family"] == family and b["n_shows"] > 0 for b in held),
                           nonshowing=sum(b["family"] == family and b["n_shows"] == 0 for b in held))
              for family in sorted({b["family"] for b in held})}
    twins = defaultdict(set)
    for b in held:
        if b.get("pair_id") is not None:
            twins[b["pair_id"]].add(b["family"])
    twin_pair_ids = {b["pair_id"] for b in held if b["family"] == "backdoor"
                     and b.get("pair_id") is not None and "quirk" in twins[b["pair_id"]]}
    backdoor_with_quirk_twin = sum(b["family"] == "backdoor" and b.get("pair_id") in twin_pair_ids
                                  for b in held)
    family_tests_confirmatory = {f: counts[f]["nonshowing"] >= 10 and counts[f]["showing"] > 0
                                  for f in C.FAMILIES}
    pooled_nonshowing = sum(counts[f]["nonshowing"] for f in C.FAMILIES)
    pooled_showing = sum(counts[f]["showing"] for f in C.FAMILIES)
    return dict(snapshot=str(snapshot), n_held=len(held), counts=counts,
                per_family_shows_test_confirmatory=family_tests_confirmatory,
                pooled_shows_test_confirmatory=pooled_nonshowing >= 10 and pooled_showing > 0,
                pooled_nonshowing=pooled_nonshowing, pooled_showing=pooled_showing,
                backdoor_with_quirk_twin=backdoor_with_quirk_twin,
                backdoor_quirk_twin_pairs=len(twin_pair_ids),
                twin_retrieval_enabled=len(twin_pair_ids) >= 20,
                fineweb_rl=fineweb_activation_layers(fineweb_store),
                rule="Per-family shows AUROC is descriptive when fewer than 10 nonshowing bundles; "
                     "twin retrieval is kept only with at least 20 backdoor/quirk held-out pairs. "
                     "RL is eligible only if fineweb_rl stores full activations at layers 10,20,30.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=C.SNAPSHOT)
    ap.add_argument("--fineweb-store", type=Path, default=C.AGM / "fineweb_rl")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    report = check(a.snapshot, a.fineweb_store)
    print(json.dumps(report, indent=2))
    if a.out:
        C.write_json(a.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

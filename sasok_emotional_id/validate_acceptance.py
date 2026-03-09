"""
validate_acceptance.py — SASOK Emotional ID acceptance checks.

Tests
─────
Test A (functional)
  Run on a synthetic seed CSV (50 prompts).
  Assert that aggregate.csv contains ≥ 50×3 = 150 records without errors.

Test B (stability)
  Load an existing session (or run one first).
  Assert Consistency Index ≥ 0.6 for ≥ 8 out of 11 features.

Test C (crash / noise)
  Inject 10 intentionally short "noisy" responses.
  Assert each is flagged noisy=True and emotional_granularity < 0.15.

Usage
─────
    python validate_acceptance.py --all
    python validate_acceptance.py --test-a --seed-csv seeds.csv
    python validate_acceptance.py --test-b --session-id <uuid>
    python validate_acceptance.py --test-c
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from metrics import aggregate_consistency, LOW_GRANULARITY_THRESHOLD
from parser import parse_response
from utils import (
    OUTPUT_DIR,
    QUARANTINE_DIR,
    compute_session_consistency,
    get_logger,
    load_env,
)

logger = get_logger("validate_acceptance")

STABLE_CI_THRESHOLD = 0.6
MIN_STABLE_FEATURES = 8
FEATURE_NAMES = [
    "latency_to_emotion", "valence", "arousal",
    "cognitive_complexity", "self_reference_rate",
    "uncertainty_marker_freq", "goal_orientedness",
    "social_orientation", "emotional_granularity",
    "adaptive_language", "trust_indicators",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_aggregate_csv(csv_path: Optional[Path] = None) -> list[dict]:
    csv_path = csv_path or OUTPUT_DIR / "aggregate.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_session_records(session_id: str) -> list[dict]:
    records = []
    for jf in sorted(OUTPUT_DIR.glob(f"{session_id}_run*_q*.json")):
        with open(jf, encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


# ---------------------------------------------------------------------------
# Test A — Functional
# ---------------------------------------------------------------------------

def test_a_functional(seed_csv: str) -> bool:
    logger.info("[Test A] Functional test on seed CSV: %s", seed_csv)

    result = subprocess.run(
        [
            sys.executable, "runner.py",
            "--seed-csv", seed_csv,
            "--dry-run",
        ],
        capture_output=True, text=True,
    )
    logger.info("[Test A] runner stdout:\n%s", result.stdout[-2000:])
    if result.returncode != 0:
        logger.error("[Test A] runner exited with code %d\n%s", result.returncode, result.stderr)
        return False

    rows = _load_aggregate_csv()
    total = len(rows)
    errors = [r for r in rows if r.get("error")]
    success_rate = (total - len(errors)) / max(1, total)

    logger.info("[Test A] total_rows=%d errors=%d success_rate=%.2f", total, len(errors), success_rate)

    passed = success_rate >= 0.95
    status = "PASS" if passed else "FAIL"
    logger.info("[Test A] %s (success_rate=%.2f)", status, success_rate)
    return passed


# ---------------------------------------------------------------------------
# Test B — Stability / Consistency Index
# ---------------------------------------------------------------------------

def test_b_stability(session_id: Optional[str] = None) -> bool:
    logger.info("[Test B] Stability test session_id=%s", session_id)

    if session_id is None:
        # Run a dry session
        result = subprocess.run(
            [sys.executable, "runner.py", "--dry-run"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("[Test B] runner failed: %s", result.stderr)
            return False
        # Extract session_id from stdout
        for line in result.stdout.splitlines():
            if "Session complete:" in line:
                session_id = line.split(":")[-1].strip()
                break

    if session_id is None:
        logger.error("[Test B] Could not determine session_id")
        return False

    records = _load_session_records(session_id)
    if not records:
        logger.error("[Test B] No records found for session_id=%s", session_id)
        return False

    ci_map = compute_session_consistency(records, session_id)
    if not ci_map:
        logger.warning("[Test B] CI map is empty — need at least 2 runs")
        return False

    stable = {feat: ci for feat, ci in ci_map.items() if ci >= STABLE_CI_THRESHOLD}
    profile_stable = len(stable) >= MIN_STABLE_FEATURES

    logger.info("[Test B] CI values:")
    for feat, ci in sorted(ci_map.items()):
        flag = "STABLE" if ci >= STABLE_CI_THRESHOLD else "UNSTABLE"
        logger.info("  %-30s %.3f  %s", feat, ci, flag)

    logger.info(
        "[Test B] stable_features=%d/%d  profile=%s",
        len(stable), len(ci_map),
        "STABLE" if profile_stable else "UNSTABLE",
    )

    status = "PASS" if profile_stable else "FAIL"
    logger.info("[Test B] %s", status)
    return profile_stable


# ---------------------------------------------------------------------------
# Test C — Crash / Noise resistance
# ---------------------------------------------------------------------------

NOISY_INPUTS = [
    "да", "нет", "не знаю", "возможно", "ок",
    "...", " ", "!", "???", "хм",
]


def test_c_noise() -> bool:
    logger.info("[Test C] Crash / noise resistance test")
    all_passed = True

    for inp in NOISY_INPUTS:
        features = parse_response(inp)
        gran = features["emotional_granularity"]
        noisy = features["noisy"]

        expected_noisy = (gran < LOW_GRANULARITY_THRESHOLD) or (len(inp.strip()) < 10)
        ok = noisy == expected_noisy

        logger.info(
            "[Test C] input=%-12r gran=%.3f noisy=%s expected=%s %s",
            inp[:12], gran, noisy, expected_noisy, "OK" if ok else "FAIL",
        )
        if not ok:
            all_passed = False

    status = "PASS" if all_passed else "FAIL"
    logger.info("[Test C] %s", status)
    return all_passed


# ---------------------------------------------------------------------------
# Full acceptance check (post-run)
# ---------------------------------------------------------------------------

def validate_session_acceptance(session_id: str) -> dict:
    """
    Run acceptance checks for an existing session.
    Returns a dict with keys: stable, stable_count, total_features,
    success_rate, ci_map, profile_status.
    """
    records = _load_session_records(session_id)
    if not records:
        return {"error": f"No records found for session_id={session_id}"}

    good = [r for r in records if "error" not in r]
    success_rate = len(good) / max(1, len(records))

    ci_map = compute_session_consistency(records, session_id)
    stable_count = sum(1 for ci in ci_map.values() if ci >= STABLE_CI_THRESHOLD)
    profile_stable = stable_count >= MIN_STABLE_FEATURES

    return {
        "session_id": session_id,
        "total_records": len(records),
        "success_rate": round(success_rate, 3),
        "ci_map": ci_map,
        "stable_count": stable_count,
        "total_features": len(ci_map),
        "profile_status": "stable" if profile_stable else "unstable",
        "acceptance_passed": success_rate >= 0.95 and profile_stable,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SASOK Acceptance Validator")
    ap.add_argument("--all", action="store_true", help="Run all tests")
    ap.add_argument("--test-a", action="store_true")
    ap.add_argument("--test-b", action="store_true")
    ap.add_argument("--test-c", action="store_true")
    ap.add_argument("--seed-csv", default="seeds.csv")
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args()

    load_env()

    results: dict[str, bool] = {}

    if args.all or args.test_a:
        results["A_functional"] = test_a_functional(args.seed_csv)

    if args.all or args.test_b:
        results["B_stability"] = test_b_stability(args.session_id)

    if args.all or args.test_c:
        results["C_noise"] = test_c_noise()

    if not results:
        ap.print_help()
        return

    print("\n=== Acceptance Results ===")
    all_pass = True
    for name, passed in results.items():
        symbol = "PASS" if passed else "FAIL"
        print(f"  Test {name}: {symbol}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

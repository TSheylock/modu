"""
utils.py — shared helpers for SASOK Emotional ID module.

Includes:
  - load_env()          — load .env file (without exposing keys)
  - get_logger()        — per-module structured logger
  - quarantine_run()    — move >10 % failed records to ./quarantine/
  - load_seed_csv()     — load synthetic seed prompts for Test A
  - ConsistencyIndex    — aggregate CI computation (delegates to metrics.py)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from metrics import aggregate_consistency

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).parent
OUTPUT_DIR = MODULE_DIR / "outputs" / "sasok_emotional_id"
LOG_DIR = MODULE_DIR / "logs"
QUARANTINE_DIR = MODULE_DIR / "quarantine"
REPORTS_DIR = MODULE_DIR / "reports"

for _p in [OUTPUT_DIR, LOG_DIR, QUARANTINE_DIR, REPORTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "sasok_emotional_id.log"

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_env(env_path: Optional[str] = None) -> None:
    """
    Load key=value pairs from *env_path* (default: MODULE_DIR/.env)
    into os.environ, without using python-dotenv as a hard dependency.
    Does NOT log or print key values.
    """
    path = Path(env_path) if env_path else MODULE_DIR / ".env"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "sasok_emotional_id") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # File handler
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Aggregate CSV writer
# ---------------------------------------------------------------------------

AGGREGATE_FIELDNAMES = [
    "session_id", "run_number", "question_index",
    "timestamp", "elapsed_time", "hash", "raw_text_len",
    "latency_to_emotion", "valence", "arousal",
    "cognitive_complexity", "self_reference_rate",
    "uncertainty_marker_freq", "goal_orientedness",
    "social_orientation", "emotional_granularity",
    "adaptive_language", "trust_indicators",
    "consistency_index", "noisy", "raw_encrypted",
]


def append_to_aggregate_csv(records: list[dict], csv_path: Optional[Path] = None) -> None:
    """Append *records* to the aggregate CSV (creates file + header if absent)."""
    if csv_path is None:
        csv_path = OUTPUT_DIR / "aggregate.csv"
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AGGREGATE_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in AGGREGATE_FIELDNAMES})


# ---------------------------------------------------------------------------
# Quarantine logic
# ---------------------------------------------------------------------------

QUARANTINE_THRESHOLD = 0.10  # >10 % failures → quarantine


def quarantine_run(
    session_id: str,
    run_number: int,
    all_records: list[dict],
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    If > 10 % of records for this run have parse errors or are flagged noisy,
    move the corresponding JSON files to ./quarantine/ and log a WARNING.

    Returns True if quarantine was triggered.
    """
    if logger is None:
        logger = get_logger()

    run_records = [
        r for r in all_records
        if r.get("session_id") == session_id and r.get("run_number") == run_number
    ]
    if not run_records:
        return False

    failed = [r for r in run_records if "error" in r or r.get("noisy")]
    ratio = len(failed) / len(run_records)

    if ratio <= QUARANTINE_THRESHOLD:
        return False

    logger.warning(
        "session_id=%s run=%d quarantine_triggered failed_ratio=%.2f",
        session_id, run_number, ratio,
    )

    # Move JSON files to quarantine
    for r in run_records:
        qi = r.get("question_index", "?")
        fname = OUTPUT_DIR / f"{session_id}_run{run_number}_q{qi}.json"
        if fname.exists():
            dest = QUARANTINE_DIR / fname.name
            shutil.move(str(fname), str(dest))
            logger.info("Quarantined %s → %s", fname.name, dest)

    return True


# ---------------------------------------------------------------------------
# Seed CSV loader (for Test A)
# ---------------------------------------------------------------------------

def load_seed_csv(path: str) -> list[str]:
    """
    Load a CSV file whose first column contains prompt strings.
    Returns a list of prompt strings (up to 50).
    """
    prompts: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            text = row[0].strip()
            if text:
                prompts.append(text)
            if len(prompts) >= 50:
                break
    return prompts


# ---------------------------------------------------------------------------
# Consistency Index helper (delegates to metrics.aggregate_consistency)
# ---------------------------------------------------------------------------

def compute_session_consistency(
    all_records: list[dict],
    session_id: str,
) -> dict[str, float]:
    """
    From all_records for a given session_id, group by question_index,
    collect the 3 run values for each feature, and return CI per feature.
    """
    from metrics import aggregate_consistency

    # Group by question_index → list of per-run feature dicts
    qi_map: dict[int, list[dict]] = {}
    for rec in all_records:
        if rec.get("session_id") != session_id or "error" in rec:
            continue
        qi = rec.get("question_index", 0)
        qi_map.setdefault(qi, []).append(rec)

    # Merge across questions: average per feature per run
    # For simplicity we flatten all runs and compute CI per feature globally
    runs_by_number: dict[int, list[dict]] = {}
    for rec in all_records:
        if rec.get("session_id") != session_id or "error" in rec:
            continue
        rn = rec.get("run_number", 1)
        runs_by_number.setdefault(rn, []).append(rec)

    # Average each numeric feature within a run
    averaged_runs: list[dict] = []
    for rn in sorted(runs_by_number):
        run_recs = runs_by_number[rn]
        feature_names = [
            "latency_to_emotion", "valence", "arousal",
            "cognitive_complexity", "self_reference_rate",
            "uncertainty_marker_freq", "goal_orientedness",
            "social_orientation", "emotional_granularity",
            "adaptive_language", "trust_indicators",
        ]
        avg: dict = {"run_number": rn}
        for feat in feature_names:
            vals = [r[feat] for r in run_recs if r.get(feat) is not None]
            avg[feat] = sum(vals) / len(vals) if vals else None
        averaged_runs.append(avg)

    if len(averaged_runs) < 2:
        return {}

    return aggregate_consistency(averaged_runs)


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

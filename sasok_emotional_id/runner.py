"""
runner.py — SASOK Emotional Mirror: 3-run session orchestrator.

Usage
-----
    python runner.py                          # 3 full runs (10.5 min apart)
    python runner.py --seed-csv seeds.csv     # synthetic seed prompts (Test A)
    python runner.py --session-id <uuid>      # reuse an existing session ID
    python runner.py --dry-run                # skip OWL calls, use mock answers

Each run queries OWL for all 12 prompts, parses each response, encrypts
raw_text, and writes:
  ./outputs/sasok_emotional_id/<session_id>_run<N>_q<M>.json  (per question)
  ./outputs/sasok_emotional_id/aggregate.csv                   (cumulative)

After all 3 runs the Consistency Index is computed and back-filled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

from encrypt import encrypt_text
from parser import hash_text, parse_response
from utils import (
    OUTPUT_DIR,
    append_to_aggregate_csv,
    compute_session_consistency,
    get_logger,
    load_env,
    load_seed_csv,
    quarantine_run,
    utc_now,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OWL_ENDPOINT = os.getenv("OWL_ENDPOINT", "http://127.0.0.1:7860/run")
RUN_INTERVAL_SECONDS = int(os.getenv("RUN_INTERVAL_SECONDS", str(10 * 60 + 30)))  # 10.5 min

MODEL_PARAMS = {
    "temperature": 0.6,
    "max_tokens": 512,
    "top_p": 0.9,
}

DEFAULT_PROMPTS: list[str] = [
    "Опишите недавний момент, когда вы почувствовали сильную эмоцию — "
    "что случилось, что вы почувствовали и почему.",
    "Какая цель важна для вас прямо сейчас и почему она занимает место в приоритете?",
    "Расскажите о случае, когда вы поменяли своё мнение — "
    "что заставило вас изменить курс?",
    "Опишите ситуацию, где вы чувствовали неуверенность — как вы с ней справились?",
    "Кто или что даёт вам ощущение поддержки? Приведите пример.",
    "В чём вы видите свою сильную профессиональную роль? Приведите пример действия.",
    "Какую эмоцию вы бы хотели испытывать чаще — и какие барьеры этому мешают?",
    "Опишите задачу, которую вы откладываете, и почему её откладываете.",
    "Как вы обычно принимаете решения под давлением?",
    "Что вызывает у вас раздражение в коллективной работе?",
    "Опишите недавний пример, когда вы помогли другому и что вы почувствовали.",
    "Что для вас означает долгосрочный успех и как вы это измерите?",
]

# ---------------------------------------------------------------------------
# OWL caller
# ---------------------------------------------------------------------------

def _call_owl(prompt: str, session_id: str, timeout: int = 60) -> tuple[str, float]:
    """
    POST prompt to OWL and return (raw_text, elapsed_seconds).
    Raises requests.HTTPError on non-2xx responses.
    Raises EnvironmentError if credentials are unavailable.
    """
    payload = {
        "question": prompt,
        "session_id": session_id,
        "model_params": MODEL_PARAMS,
    }
    t0 = time.perf_counter()
    resp = requests.post(OWL_ENDPOINT, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    raw_text: str = data.get("answer") or data.get("response") or ""
    return raw_text, round(elapsed, 3)


def _mock_owl(prompt: str, session_id: str) -> tuple[str, float]:
    """Return a synthetic answer (dry-run mode)."""
    answers = [
        "Я почувствовал сильную тревогу, когда узнал о болезни близкого. "
        "Это было неожиданно и очень больно, я не знал, как реагировать.",
        "Моя главная цель сейчас — завершить проект SASOK, потому что "
        "это дело моей жизни и я стремлюсь к тому, чтобы оно было реализовано.",
        "Я изменил своё мнение о дистанционной работе, когда понял, "
        "что продуктивность можно сохранять и дома.",
        "Неуверенность была, когда я принимал решение о смене профессии. "
        "Я справился, разбив задачу на маленькие шаги.",
        "Поддержку даёт команда коллег и наставник — они помогали "
        "в трудные моменты советом и присутствием.",
        "Моя сильная роль — аналитик и архитектор систем. "
        "Я разработал модуль обработки эмоциональных профилей.",
        "Хотел бы чаще чувствовать спокойствие. "
        "Барьер — постоянная загруженность и тревожные мысли.",
        "Откладываю систематизацию документации, потому что "
        "это кажется рутиной, хотя понимаю важность.",
        "Под давлением я стараюсь сначала сделать вдох, "
        "затем выделить приоритеты и действовать пошагово.",
        "Раздражает, когда в команде нет чёткого распределения ответственности "
        "и люди перекладывают задачи друг на друга.",
        "Я помог коллеге подготовиться к презентации — объяснил структуру "
        "и поправил слайды. Почувствовал удовлетворение и радость.",
        "Долгосрочный успех для меня — это когда SASOK работает, "
        "помогает людям, и я вижу реальный позитивный эффект на их жизнь.",
    ]
    idx = DEFAULT_PROMPTS.index(prompt) if prompt in DEFAULT_PROMPTS else 0
    return answers[idx % len(answers)], 0.25


# ---------------------------------------------------------------------------
# Single session runner
# ---------------------------------------------------------------------------

def run_session(
    session_id: str,
    run_number: int,
    prompts: list[str],
    dry_run: bool = False,
    logger=None,
) -> list[dict]:
    if logger is None:
        logger = get_logger()

    results: list[dict] = []

    for i, prompt in enumerate(prompts, start=1):
        try:
            if dry_run:
                raw_text, elapsed = _mock_owl(prompt, session_id)
            else:
                raw_text, elapsed = _call_owl(prompt, session_id)

            text_hash = hash_text(raw_text)

            try:
                raw_encrypted = encrypt_text(raw_text)
            except EnvironmentError as enc_err:
                logger.error(
                    "session_id=%s run=%d q=%d credentials_error: %s",
                    session_id, run_number, i, enc_err,
                )
                # Per spec: stop on credentials error — do not fall back
                raise SystemExit(1) from enc_err

            features = parse_response(raw_text)
            timestamp = utc_now()

            record = {
                "session_id": session_id,
                "run_number": run_number,
                "question_index": i,
                "prompt": prompt,
                "timestamp": timestamp,
                "elapsed_time": elapsed,
                "hash": text_hash,
                "raw_encrypted": raw_encrypted,
                "raw_text_len": features.pop("raw_text_len"),
                "noisy": features.pop("noisy"),
                "parse_elapsed_s": features.pop("parse_elapsed_s"),
                **features,
            }

            # Save per-question JSON
            out_path = OUTPUT_DIR / f"{session_id}_run{run_number}_q{i}.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2, default=str)

            results.append(record)
            logger.info(
                "session_id=%s run=%d q=%d elapsed=%.3f hash=%s noisy=%s",
                session_id, run_number, i, elapsed, text_hash[:12], record["noisy"],
            )

        except SystemExit:
            raise
        except Exception as exc:
            logger.error(
                "session_id=%s run=%d q=%d error: %s",
                session_id, run_number, i, exc, exc_info=True,
            )
            results.append({
                "session_id": session_id,
                "run_number": run_number,
                "question_index": i,
                "error": str(exc),
            })

    return results


# ---------------------------------------------------------------------------
# Post-run: back-fill Consistency Index
# ---------------------------------------------------------------------------

def backfill_consistency_index(
    all_records: list[dict],
    session_id: str,
    logger=None,
) -> list[dict]:
    if logger is None:
        logger = get_logger()

    ci_map = compute_session_consistency(all_records, session_id)
    if not ci_map:
        return all_records

    for rec in all_records:
        if rec.get("session_id") != session_id or "error" in rec:
            continue
        # Average CI across all features
        ci_values = list(ci_map.values())
        rec["consistency_index"] = round(sum(ci_values) / len(ci_values), 3) if ci_values else None

        # Re-save JSON
        qi = rec.get("question_index")
        rn = rec.get("run_number")
        out_path = OUTPUT_DIR / f"{session_id}_run{rn}_q{qi}.json"
        if out_path.exists():
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False, indent=2, default=str)

    logger.info("session_id=%s consistency_index backfilled for %d records", session_id, len(all_records))
    return all_records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SASOK Emotional Mirror — 3-run session runner"
    )
    parser.add_argument("--session-id", default=None, help="Reuse existing session UUID")
    parser.add_argument("--seed-csv", default=None, help="Path to seed CSV (Test A)")
    parser.add_argument("--dry-run", action="store_true", help="Use mock OWL responses")
    parser.add_argument(
        "--interval", type=int, default=RUN_INTERVAL_SECONDS,
        help="Seconds between runs (default: 630 = 10.5 min)"
    )
    args = parser.parse_args()

    load_env()
    logger = get_logger()

    session_id = args.session_id or str(uuid.uuid4())
    logger.info("Starting session session_id=%s dry_run=%s", session_id, args.dry_run)

    # Prompts selection
    if args.seed_csv:
        prompts = load_seed_csv(args.seed_csv)
        if not prompts:
            logger.error("seed-csv loaded 0 prompts — aborting")
            sys.exit(1)
        logger.info("Loaded %d seed prompts from %s", len(prompts), args.seed_csv)
    else:
        prompts = DEFAULT_PROMPTS

    all_records: list[dict] = []

    for run_num in range(1, 4):
        logger.info("=== Run %d / 3 ===", run_num)
        run_results = run_session(
            session_id=session_id,
            run_number=run_num,
            prompts=prompts,
            dry_run=args.dry_run,
            logger=logger,
        )
        all_records.extend(run_results)

        # Quarantine check
        quarantine_run(session_id, run_num, all_records, logger)

        if run_num < 3:
            logger.info(
                "Waiting %d s before next run…", args.interval
            )
            if not args.dry_run:
                time.sleep(args.interval)

    # Back-fill Consistency Index across the 3 runs
    all_records = backfill_consistency_index(all_records, session_id, logger)

    # Write aggregate CSV
    append_to_aggregate_csv(all_records)
    logger.info(
        "Aggregate CSV updated. session_id=%s total_records=%d",
        session_id, len(all_records),
    )

    # Summary stats
    success = [r for r in all_records if "error" not in r]
    failed = [r for r in all_records if "error" in r]
    success_rate = len(success) / max(1, len(all_records))
    noisy_count = sum(1 for r in success if r.get("noisy"))
    avg_elapsed = (
        sum(r.get("elapsed_time", 0) for r in success) / max(1, len(success))
    )
    logger.info(
        "Session complete. success_rate=%.2f avg_elapsed=%.3f noisy=%d/%d",
        success_rate, avg_elapsed, noisy_count, len(all_records),
    )

    print(f"\n[SASOK] Session complete: {session_id}")
    print(f"  Records   : {len(all_records)} ({len(success)} ok, {len(failed)} errors)")
    print(f"  Noisy     : {noisy_count}")
    print(f"  Success % : {success_rate * 100:.1f}")
    print(f"  Avg OWL   : {avg_elapsed:.3f} s")
    print(f"  Output    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
generate_report.py — SASOK Emotional ID: markdown report + matplotlib charts.

Usage
-----
    python generate_report.py --session-id <uuid>
    python generate_report.py --session-id <uuid> --no-charts

Outputs
-------
  ./reports/<session_id>_report.md
  ./reports/<session_id>_valence_dist.png
  ./reports/<session_id>_trust_bar.png
  ./reports/<session_id>_ci_table.png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils import (
    OUTPUT_DIR,
    REPORTS_DIR,
    compute_session_consistency,
    get_logger,
    load_env,
)
from metrics import STABLE_CI_THRESHOLD  # type: ignore  # defined in metrics as module-level constant

# metrics.py doesn't export STABLE_CI_THRESHOLD, use local default
_CI_THRESHOLD = 0.6
_MIN_STABLE = 8

logger = get_logger("generate_report")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_session_records(session_id: str) -> list[dict]:
    records = []
    for jf in sorted(OUTPUT_DIR.glob(f"{session_id}_run*_q*.json")):
        with open(jf, encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _make_valence_histogram(records: list[dict], session_id: str) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valences = [float(r["valence"]) for r in records if r.get("valence") is not None]
        if not valences:
            return None

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valences, bins=15, color="#4A90D9", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="#E74C3C", linewidth=1.2, linestyle="--", label="Neutral (0)")
        ax.set_title("Valence Score Distribution", fontsize=13)
        ax.set_xlabel("Valence (-1 … +1)")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()

        out = REPORTS_DIR / f"{session_id}_valence_dist.png"
        fig.savefig(str(out), dpi=120)
        plt.close(fig)
        return out
    except Exception as exc:
        logger.warning("Could not create valence histogram: %s", exc)
        return None


def _make_trust_bar(records: list[dict], session_id: str) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        per_q: dict[int, list[float]] = {}
        for r in records:
            qi = r.get("question_index")
            t = r.get("trust_indicators")
            if qi is not None and t is not None:
                per_q.setdefault(int(qi), []).append(float(t))

        if not per_q:
            return None

        qs = sorted(per_q)
        means = [sum(per_q[q]) / len(per_q[q]) for q in qs]
        colors = ["#2ECC71" if m >= 0 else "#E74C3C" for m in means]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([f"Q{q}" for q in qs], means, color=colors)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("Trust Indicators per Question (avg across runs)", fontsize=12)
        ax.set_ylabel("Trust score")
        fig.tight_layout()

        out = REPORTS_DIR / f"{session_id}_trust_bar.png"
        fig.savefig(str(out), dpi=120)
        plt.close(fig)
        return out
    except Exception as exc:
        logger.warning("Could not create trust bar chart: %s", exc)
        return None


def _make_ci_heatmap(ci_map: dict[str, float], session_id: str) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        features = list(ci_map)
        values = [ci_map[f] for f in features]

        fig, ax = plt.subplots(figsize=(10, 3))
        cmap = plt.cm.RdYlGn
        im = ax.imshow([values], cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(
            [f.replace("_", "\n") for f in features], fontsize=8
        )
        ax.set_yticks([])
        ax.set_title("Consistency Index Heatmap (green ≥ 0.6 = stable)", fontsize=11)
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02)
        fig.tight_layout()

        out = REPORTS_DIR / f"{session_id}_ci_heatmap.png"
        fig.savefig(str(out), dpi=120)
        plt.close(fig)
        return out
    except Exception as exc:
        logger.warning("Could not create CI heatmap: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _build_report(
    session_id: str,
    records: list[dict],
    ci_map: dict[str, float],
    valence_chart: Optional[Path],
    trust_chart: Optional[Path],
    ci_chart: Optional[Path],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    good = [r for r in records if "error" not in r]
    success_rate = len(good) / max(1, len(records))

    valences = [r["valence"] for r in good if r.get("valence") is not None]
    avg_valence = sum(valences) / len(valences) if valences else 0.0

    stable_features = {f: ci for f, ci in ci_map.items() if ci >= _CI_THRESHOLD}
    unstable_features = {f: ci for f, ci in ci_map.items() if ci < _CI_THRESHOLD}
    profile_stable = len(stable_features) >= _MIN_STABLE

    noisy = sum(1 for r in good if r.get("noisy"))
    avg_elapsed = sum(r.get("elapsed_time", 0) for r in good) / max(1, len(good))

    lines: list[str] = []
    lines.append(f"# SASOK Emotional Profile Report")
    lines.append(f"")
    lines.append(f"**Session ID:** `{session_id}`  ")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Profile status:** {'✅ STABLE' if profile_stable else '⚠️ UNSTABLE'}")
    lines.append("")

    # ── Executive Summary ────────────────────────────────────────────────
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total records | {len(records)} ({len(good)} ok, {len(records)-len(good)} errors) |")
    lines.append(f"| Parse success rate | {success_rate*100:.1f}% |")
    lines.append(f"| Average valence | {avg_valence:+.3f} |")
    lines.append(f"| Stable features (CI ≥ {_CI_THRESHOLD}) | {len(stable_features)} / {len(ci_map)} |")
    lines.append(f"| Profile status | {'Stable' if profile_stable else 'Unstable'} |")
    lines.append(f"| Noisy responses | {noisy} |")
    lines.append(f"| Avg OWL elapsed | {avg_elapsed:.3f} s |")
    lines.append("")

    if not profile_stable:
        lines.append(
            "> **Warning:** Profile is marked UNSTABLE "
            f"(only {len(stable_features)}/{len(ci_map)} features have CI ≥ {_CI_THRESHOLD}). "
            "Additional sessions or improved prompts are recommended."
        )
        lines.append("")

    # ── Valence Distribution ─────────────────────────────────────────────
    lines.append("## Detailed Analytics")
    lines.append("")
    lines.append("### Valence Score Distribution")
    if valence_chart:
        lines.append(f"![Valence Distribution]({valence_chart.name})")
    else:
        lines.append("*(chart not available — install matplotlib)*")
    lines.append("")

    if valences:
        pos = sum(1 for v in valences if v > 0.1)
        neg = sum(1 for v in valences if v < -0.1)
        neu = len(valences) - pos - neg
        lines.append(
            f"Positive responses: **{pos}** | "
            f"Neutral: **{neu}** | "
            f"Negative: **{neg}**"
        )
    lines.append("")

    # ── Trust Indicators ─────────────────────────────────────────────────
    lines.append("### Trust Indicators per Question")
    if trust_chart:
        lines.append(f"![Trust Bar]({trust_chart.name})")
    else:
        lines.append("*(chart not available)*")
    lines.append("")

    # ── Consistency Index table ──────────────────────────────────────────
    lines.append("### Consistency Index per Feature")
    if ci_chart:
        lines.append(f"![CI Heatmap]({ci_chart.name})")
    lines.append("")
    lines.append("| Feature | CI | Status |")
    lines.append("|---------|-----|--------|")
    for feat in sorted(ci_map):
        ci = ci_map[feat]
        status = "Stable" if ci >= _CI_THRESHOLD else "Unstable"
        lines.append(f"| `{feat}` | {ci:.3f} | {status} |")
    lines.append("")

    # ── High-noise features ──────────────────────────────────────────────
    lines.append("### Features with High Noise Risk")
    if unstable_features:
        for feat, ci in sorted(unstable_features.items(), key=lambda x: x[1]):
            lines.append(f"- **{feat}** — CI = {ci:.3f}")
    else:
        lines.append("*No high-risk noisy features detected.*")
    lines.append("")

    # ── Recommendations ──────────────────────────────────────────────────
    lines.append("## Recommendations for SASOK Model Training")
    lines.append("")

    rec_idx = 1
    # R1 — valence
    if avg_valence < -0.1:
        lines.append(
            f"{rec_idx}. **Valence calibration** — The average valence is negative "
            f"({avg_valence:+.3f}). Enrich the training corpus with positive-affect "
            "examples and validate the sentiment lexicon on Russian text."
        )
    else:
        lines.append(
            f"{rec_idx}. **Valence calibration** — Average valence is balanced "
            f"({avg_valence:+.3f}). Monitor for domain drift in future sessions."
        )
    rec_idx += 1

    # R2 — unstable features
    if unstable_features:
        feat_list = ", ".join(sorted(unstable_features)[:3])
        lines.append(
            f"{rec_idx}. **Marker refinement** — Features {feat_list} show CI < {_CI_THRESHOLD}. "
            "Extend domain-specific lexicons for uncertainty, social orientation, and "
            "adaptive language; re-label ambiguous markers."
        )
    else:
        lines.append(
            f"{rec_idx}. **Marker refinement** — All features are stable. "
            "Periodically audit lexicons to prevent label drift."
        )
    rec_idx += 1

    # R3 — noise / short answers
    if noisy > 0:
        lines.append(
            f"{rec_idx}. **Short-answer fallback** — {noisy} noisy responses detected. "
            "Implement a minimum-length guardrail in the OWL session (e.g., re-prompt "
            "when raw_text_len < 15) and route flagged records to quarantine automatically."
        )
    else:
        lines.append(
            f"{rec_idx}. **Short-answer fallback** — No noisy responses in this session. "
            "Maintain the current minimum-length threshold and test with adversarial inputs."
        )
    lines.append("")

    # ── Footer ───────────────────────────────────────────────────────────
    lines.append("---")
    lines.append(
        "*Report generated by SASOK Emotional Mirror module. "
        "Raw text is AES-256 encrypted at rest; this report contains no PII.*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SASOK Emotional Profile Report Generator")
    ap.add_argument("--session-id", required=True, help="Session UUID to report on")
    ap.add_argument("--no-charts", action="store_true", help="Skip matplotlib charts")
    args = ap.parse_args()

    load_env()

    session_id = args.session_id
    records = _load_session_records(session_id)
    if not records:
        print(f"[ERROR] No records found for session_id={session_id}")
        raise SystemExit(1)

    ci_map = compute_session_consistency(records, session_id)

    valence_chart = trust_chart = ci_chart = None
    if not args.no_charts:
        valence_chart = _make_valence_histogram(records, session_id)
        trust_chart = _make_trust_bar(records, session_id)
        ci_chart = _make_ci_heatmap(ci_map, session_id)

    md = _build_report(session_id, records, ci_map, valence_chart, trust_chart, ci_chart)

    out_path = REPORTS_DIR / f"{session_id}_report.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"[SASOK] Report written to {out_path}")
    if valence_chart:
        print(f"  Charts: {valence_chart.name}, {trust_chart and trust_chart.name}, {ci_chart and ci_chart.name}")


if __name__ == "__main__":
    main()

"""
parser.py — post-processing pipeline for SASOK Emotional ID.

Receives raw_text from an OWL session response and returns a dict
with all 12 emotional-profile features.  Processes ≤ 2.5 s per
response on a single CPU core (target from acceptance criteria).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Optional

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from metrics import (
    LOW_GRANULARITY_THRESHOLD,
    MIN_TEXT_LENGTH,
    calc_adaptive_language,
    calc_arousal,
    calc_cognitive_complexity,
    calc_emotional_granularity,
    calc_goal_orientedness,
    calc_latency_to_emotion,
    calc_self_reference_rate,
    calc_social_orientation,
    calc_trust_indicators,
    calc_uncertainty_marker_freq,
    calc_valence,
    is_noisy_response,
)

# ---------------------------------------------------------------------------
# NLTK VADER bootstrap (downloads once, cached after that)
# ---------------------------------------------------------------------------

def _ensure_vader() -> SentimentIntensityAnalyzer:
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


_SID: Optional[SentimentIntensityAnalyzer] = None


def _get_sid() -> SentimentIntensityAnalyzer:
    global _SID
    if _SID is None:
        _SID = _ensure_vader()
    return _SID


# ---------------------------------------------------------------------------
# PII anonymisation — replace Cyrillic names / emails / phone numbers
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    (re.compile(r"\b[A-ZА-ЯЁ][a-zа-яё]{2,}\s[A-ZА-ЯЁ][a-zа-яё]{2,}\b"), "<REDACTED_NAME>"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"), "<REDACTED_EMAIL>"),
    (re.compile(r"\b(\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"), "<REDACTED_PHONE>"),
]


def anonymise_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Core parse function
# ---------------------------------------------------------------------------

def parse_response(raw_text: str) -> dict:
    """
    Extract all 12 emotional-profile features from *raw_text*.
    Returns a flat dict; consistency_index is None (filled later by runner).

    Performance target: ≤ 2.5 s per call on one CPU core.
    """
    t0 = time.perf_counter()

    sid = _get_sid()

    # Pre-process
    text = raw_text.strip()
    text_len = len(text)

    # Feature extraction (11 features; consistency_index filled across runs)
    latency = calc_latency_to_emotion(text, sid)
    valence = calc_valence(text, sid)
    arousal = calc_arousal(text)
    cognitive = calc_cognitive_complexity(text)
    self_ref = calc_self_reference_rate(text)
    uncertainty = calc_uncertainty_marker_freq(text)
    goal = calc_goal_orientedness(text)
    social = calc_social_orientation(text)
    granularity = calc_emotional_granularity(text, sid)
    adaptive = calc_adaptive_language(text)
    trust = calc_trust_indicators(text)

    noisy = is_noisy_response(text, granularity)
    elapsed = round(time.perf_counter() - t0, 4)

    return {
        # ── 12 features ──────────────────────────────────────────────────
        "latency_to_emotion":    latency,
        "valence":               valence,
        "arousal":               arousal,
        "cognitive_complexity":  cognitive,
        "self_reference_rate":   self_ref,
        "uncertainty_marker_freq": uncertainty,
        "goal_orientedness":     goal,
        "social_orientation":    social,
        "emotional_granularity": granularity,
        "adaptive_language":     adaptive,
        "trust_indicators":      trust,
        "consistency_index":     None,          # computed later across runs
        # ── metadata ─────────────────────────────────────────────────────
        "raw_text_len":          text_len,
        "noisy":                 noisy,
        "parse_elapsed_s":       elapsed,
    }


def hash_text(text: str) -> str:
    """SHA-256 digest of the raw text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

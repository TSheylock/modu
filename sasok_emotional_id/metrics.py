"""
metrics.py — 12 emotional-profile features + Consistency Index for SASOK.

Feature catalogue
─────────────────
 1. latency_to_emotion         — sentence index of first emotional sentence
 2. valence                    — compound sentiment score  (-1 … +1)
 3. arousal                    — proxy: avg sentence length + exclamation marks
 4. cognitive_complexity       — avg sent length + subordinate conjunctions
 5. self_reference_rate        — fraction of 1st-person pronouns
 6. uncertainty_marker_freq    — fraction of uncertainty-marker tokens
 7. goal_orientedness          — binary: goal-related keywords present
 8. social_orientation         — fraction of social/collective tokens
 9. emotional_granularity      — normalised distinct emotion-bearing lexemes
10. adaptive_language          — binary: corrective/solution verbs present
11. trust_indicators           — (positive trust − negative trust) / total
12. consistency_index          — computed across 3 runs (None until filled)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Domain lexicons (Russian)
# ---------------------------------------------------------------------------

UNCERTAINTY_MARKERS: frozenset[str] = frozenset({
    "может", "возможно", "кажется", "неуверен", "пожалуй",
    "скорее", "вероятно", "похоже", "наверное", "предполагаю",
    "допускаю", "похоже", "видимо", "по-видимому",
})

GOAL_KEYWORDS: frozenset[str] = frozenset({
    "цель", "план", "намереваюсь", "буду", "хочу", "стремлюсь",
    "задача", "постараюсь", "намерен", "планирую", "достигну",
    "добьюсь", "реализую", "запланировал", "ориентир",
})

SOCIAL_TOKENS: frozenset[str] = frozenset({
    "они", "их", "им", "тебя", "тебе", "вас", "вам", "нас", "нам",
    "люди", "команда", "коллеги", "общество", "группа", "другие",
    "все", "семья", "друзья", "партнёры", "сообщество",
})

TRUST_POS: frozenset[str] = frozenset({
    "доверяю", "верю", "уверен", "надёжный", "надежный",
    "надежда", "доверие", "стабильность", "честно", "открыто",
})

TRUST_NEG: frozenset[str] = frozenset({
    "сомневаюсь", "недоверие", "не доверяю", "подозреваю",
    "сомнение", "ненадёжный", "ненадежный", "боюсь", "тревожусь",
    "опасаюсь",
})

ADAPTIVE_VERBS: frozenset[str] = frozenset({
    "попробую", "изменю", "адаптирую", "скорректирую", "предложу",
    "перестрою", "пересмотрю", "справлюсь", "решу", "найду",
    "разберусь", "улучшу", "оптимизирую",
})

SUBORDINATE_CONJ: frozenset[str] = frozenset({
    "потому", "поскольку", "хотя", "если", "когда", "чтобы",
    "так", "несмотря", "вследствие", "при", "вместо", "пока",
    "пусть", "будто",
})

SELF_REF_TOKENS: frozenset[str] = frozenset({
    "я", "мне", "меня", "мой", "моя", "моё", "моего", "моей",
    "мои", "моих", "мы", "нам", "нас", "наш", "наша", "наше", "наши",
})


# ---------------------------------------------------------------------------
# Simple sentence splitter (no spaCy dependency)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text.lower())


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def calc_latency_to_emotion(text: str, sid) -> int:
    """
    Index of the first sentence whose VADER compound score > 0.2.
    Returns -1 if no sentence crosses the threshold.
    """
    sentences = _split_sentences(text)
    for i, sent in enumerate(sentences):
        score = sid.polarity_scores(sent)["compound"]
        if abs(score) > 0.2:
            return i
    return -1


def calc_valence(text: str, sid) -> float:
    """VADER compound score for the whole text, rounded to 3 dp."""
    return round(sid.polarity_scores(text)["compound"], 3)


def calc_arousal(text: str) -> float:
    """
    Proxy: normalised average sentence length + exclamation density.
    Clamped to [0, 1].
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    exclaim_density = text.count("!") * 0.15
    return round(min(1.0, avg_len / 20 + exclaim_density), 3)


def calc_cognitive_complexity(text: str) -> float:
    """
    Proxy: avg sentence length / 30 + subordinate-conjunction density.
    Clamped to [0, 1].
    """
    tokens = _tokenize(text)
    sentences = _split_sentences(text)
    avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
    sub_count = sum(1 for t in tokens if t in SUBORDINATE_CONJ)
    score = avg_len / 30 + sub_count * 0.04
    return round(min(1.0, score), 3)


def calc_self_reference_rate(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t in SELF_REF_TOKENS)
    return round(count / len(tokens), 3)


def calc_uncertainty_marker_freq(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t in UNCERTAINTY_MARKERS)
    return round(count / len(tokens), 3)


def calc_goal_orientedness(text: str) -> int:
    """Binary: 1 if any goal-keyword present, else 0."""
    tokens = set(_tokenize(text))
    return 1 if tokens & GOAL_KEYWORDS else 0


def calc_social_orientation(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t in SOCIAL_TOKENS)
    return round(count / len(tokens), 3)


def calc_emotional_granularity(text: str, sid) -> float:
    """
    Fraction of distinct emotion-bearing lemmas (VADER polarity ≠ 0).
    Normalised to [0, 1] by dividing by 5; clamped.
    """
    tokens = _tokenize(text)
    distinct_emotion_tokens = {
        t for t in set(tokens)
        if sid.polarity_scores(t)["compound"] != 0.0
    }
    return round(min(1.0, len(distinct_emotion_tokens) / 5), 3)


def calc_adaptive_language(text: str) -> int:
    """Binary: 1 if any adaptive/corrective verb present."""
    tokens = set(_tokenize(text))
    return 1 if tokens & ADAPTIVE_VERBS else 0


def calc_trust_indicators(text: str) -> float:
    """(#positive_trust_tokens − #negative_trust_tokens) / total_tokens."""
    tokens = _tokenize(text)
    total = max(1, len(tokens))
    pos = sum(1 for t in tokens if t in TRUST_POS)
    neg = sum(1 for t in tokens if t in TRUST_NEG)
    return round((pos - neg) / total, 3)


# ---------------------------------------------------------------------------
# Consistency Index — computed across 3 runs
# ---------------------------------------------------------------------------

def calc_consistency_index(values: list[float]) -> float:
    """
    CI for a single feature across 3 runs:
        CI = 1 − σ / (|μ| + ε)
    Returns value in [0, 1].
    Falls back to range-based formula when μ ≈ 0.
    """
    if len(values) < 2:
        return 1.0
    eps = 1e-9
    mu = sum(values) / len(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    sigma = math.sqrt(variance)
    if abs(mu) < eps:
        val_range = max(values) - min(values)
        ci = 1.0 - sigma / (val_range + eps)
    else:
        ci = 1.0 - sigma / (abs(mu) + eps)
    return round(max(0.0, min(1.0, ci)), 3)


def aggregate_consistency(
    runs: list[dict],
    feature_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Given a list of per-run feature dicts (one per run),
    compute CI for every numeric feature.
    """
    if feature_names is None:
        feature_names = [
            "latency_to_emotion", "valence", "arousal",
            "cognitive_complexity", "self_reference_rate",
            "uncertainty_marker_freq", "goal_orientedness",
            "social_orientation", "emotional_granularity",
            "adaptive_language", "trust_indicators",
        ]
    ci_map: dict[str, float] = {}
    for feat in feature_names:
        vals = []
        for run in runs:
            v = run.get(feat)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        ci_map[feat] = calc_consistency_index(vals) if len(vals) >= 2 else 1.0
    return ci_map


# ---------------------------------------------------------------------------
# Noise / low-content detector
# ---------------------------------------------------------------------------

LOW_GRANULARITY_THRESHOLD = 0.15
MIN_TEXT_LENGTH = 10


def is_noisy_response(text: str, emotional_granularity: float) -> bool:
    """Return True if the response looks like low-content noise."""
    return (
        emotional_granularity < LOW_GRANULARITY_THRESHOLD
        or len(text.strip()) < MIN_TEXT_LENGTH
    )

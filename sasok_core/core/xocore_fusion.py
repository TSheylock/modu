"""
XoCore Fusion Engine — сердце SASOK.
Объединяет все модальности (лицо, голос, текст, физиология, поведение, Wi-Fi CSI)
в единый Emotional State Vector (ESV) с адаптивными весами.
"""
import time
import asyncio
import logging
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("SASOK.XoCore")


@dataclass
class ModalitySignal:
    """Сигнал от одной модальности."""
    valence: float = 0.0        # -1..1  (негатив → позитив)
    arousal: float = 0.5        # 0..1   (спокойствие → возбуждение)
    dominance: float = 0.5      # 0..1   (подчинение → доминирование)
    confidence: float = 0.0     # 0..1   насколько модальность уверена
    cognitive_load: float = 0.0 # 0..1   только поведение/Wi-Fi
    timestamp: float = field(default_factory=time.time)


@dataclass
class EmotionalStateVector:
    """Единый вектор эмоционального состояния — выход XoCore."""
    valence: float
    arousal: float
    dominance: float
    confidence: float
    discrete_emotion: str
    cognitive_load: float
    coherence_score: float          # насколько модальности согласованы
    timestamp: float
    modalities_used: List[str]
    raw_signals: Dict[str, ModalitySignal]

    def to_dict(self) -> Dict:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "confidence": round(self.confidence, 4),
            "discrete_emotion": self.discrete_emotion,
            "cognitive_load": round(self.cognitive_load, 4),
            "coherence_score": round(self.coherence_score, 4),
            "timestamp": self.timestamp,
            "modalities_used": self.modalities_used,
        }


# Маппинг VAD → дискретная эмоция
VAD_TO_EMOTION = [
    # (valence_min, valence_max, arousal_min, arousal_max, label)
    (0.5,  1.0, 0.6, 1.0, "joy"),
    (0.5,  1.0, 0.0, 0.6, "contentment"),
    (-1.0, -0.3, 0.6, 1.0, "anger"),
    (-1.0, -0.3, 0.3, 0.6, "sadness"),
    (-1.0, -0.3, 0.0, 0.3, "depression"),
    (-0.3,  0.3, 0.6, 1.0, "anxiety"),
    (-0.3,  0.3, 0.0, 0.3, "calm"),
    (-0.3,  0.5, 0.3, 0.6, "neutral"),
    (0.3,   1.0, 0.7, 1.0, "excitement"),
    (-0.5, -0.1, 0.7, 1.0, "fear"),
]


def vad_to_discrete(valence: float, arousal: float) -> str:
    for v_min, v_max, a_min, a_max, label in VAD_TO_EMOTION:
        if v_min <= valence <= v_max and a_min <= arousal <= a_max:
            return label
    return "neutral"


class XoCoreFusion:
    """
    Адаптивный фьюжн-движок SASOK.
    Объединяет сигналы от всех модальностей в ESV.
    Веса адаптируются на основе обратной связи пользователя.
    """

    # Начальные веса модальностей
    DEFAULT_WEIGHTS = {
        "face":     0.25,
        "voice":    0.25,
        "text":     0.25,
        "physio":   0.15,
        "behavior": 0.05,
        "wifi":     0.05,
    }

    # Максимальное изменение VAD за секунду (биологический лимит)
    MAX_VAD_CHANGE_PER_SEC = 0.5 / 60.0

    def __init__(self):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self._history: deque = deque(maxlen=100)
        self._prev_esv: Optional[EmotionalStateVector] = None
        self._user_feedback: List[Dict] = []
        logger.info("XoCore Fusion Engine инициализирован")

    # ------------------------------------------------------------------
    # Главный метод
    # ------------------------------------------------------------------

    def fuse(
        self,
        signals: Dict[str, ModalitySignal],
        signal_quality: Optional[Dict[str, float]] = None,
    ) -> Optional[EmotionalStateVector]:
        """
        Объединяет сигналы всех доступных модальностей.

        Args:
            signals: {'face': ModalitySignal, 'voice': ..., ...}
            signal_quality: {'face': 0.9, 'voice': 0.7, ...}  0..1

        Returns:
            EmotionalStateVector или None если нет входных данных
        """
        if not signals:
            return None

        if signal_quality is None:
            signal_quality = {k: 1.0 for k in signals}

        # Скорректированные веса
        adj = {}
        for mod, weight in self.weights.items():
            if mod in signals:
                adj[mod] = weight * signal_quality.get(mod, 0.5)
            else:
                adj[mod] = 0.0

        total = sum(adj.values())
        if total == 0:
            return None
        adj = {k: v / total for k, v in adj.items()}

        # Взвешенное среднее VAD
        valence   = sum(signals[m].valence   * adj[m] for m in adj if m in signals)
        arousal   = sum(signals[m].arousal   * adj[m] for m in adj if m in signals)
        dominance = sum(signals[m].dominance * adj[m] for m in adj if m in signals)
        confidence = sum(signals[m].confidence * adj[m] for m in adj if m in signals)
        cognitive_load = sum(signals[m].cognitive_load * adj[m] for m in adj if m in signals)

        # Clip
        valence    = float(np.clip(valence,    -1.0, 1.0))
        arousal    = float(np.clip(arousal,     0.0, 1.0))
        dominance  = float(np.clip(dominance,   0.0, 1.0))
        confidence = float(np.clip(confidence,  0.0, 1.0))
        cognitive_load = float(np.clip(cognitive_load, 0.0, 1.0))

        # Временная когерентность
        now = time.time()
        coherence = self._check_temporal_coherence(valence, arousal, now)
        if not coherence:
            logger.warning("XoCore: временная некогерентность — сглаживание")
            if self._prev_esv:
                alpha = 0.3  # текущее значение
                valence   = alpha * valence   + (1 - alpha) * self._prev_esv.valence
                arousal   = alpha * arousal   + (1 - alpha) * self._prev_esv.arousal
                dominance = alpha * dominance + (1 - alpha) * self._prev_esv.dominance

        # Кросс-модальная согласованность
        cross_modal_agreement = self._cross_modal_agreement(signals)

        esv = EmotionalStateVector(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence=confidence,
            discrete_emotion=vad_to_discrete(valence, arousal),
            cognitive_load=cognitive_load,
            coherence_score=cross_modal_agreement,
            timestamp=now,
            modalities_used=list(signals.keys()),
            raw_signals=signals,
        )

        self._prev_esv = esv
        self._history.append({"esv": esv, "weights": dict(adj)})
        logger.debug(f"XoCore ESV: {esv.discrete_emotion} "
                     f"v={valence:.2f} a={arousal:.2f} conf={confidence:.2f}")
        return esv

    # ------------------------------------------------------------------
    # Адаптация весов на основе обратной связи
    # ------------------------------------------------------------------

    def update_weights_from_feedback(self, true_valence: float, true_arousal: float):
        """
        Пользователь скорректировал своё состояние.
        Уменьшаем вес модальностей которые ошиблись.
        """
        if not self._history:
            return

        last = self._history[-1]
        esv: EmotionalStateVector = last["esv"]
        adj: Dict = last["weights"]

        for mod in esv.modalities_used:
            sig = esv.raw_signals[mod]
            err = abs(sig.valence - true_valence) + abs(sig.arousal - true_arousal)
            if err > 0.4:
                self.weights[mod] = max(0.01, self.weights[mod] * 0.92)
            else:
                self.weights[mod] = min(0.5, self.weights[mod] * 1.05)

        # Нормализация
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info(f"XoCore: веса обновлены → {self.weights}")

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _check_temporal_coherence(self, valence: float, arousal: float, now: float) -> bool:
        if self._prev_esv is None:
            return True
        dt = now - self._prev_esv.timestamp
        if dt <= 0:
            return True
        max_change = self.MAX_VAD_CHANGE_PER_SEC * dt
        dv = abs(valence - self._prev_esv.valence)
        da = abs(arousal - self._prev_esv.arousal)
        return dv <= max_change * 120 and da <= max_change * 120

    def _cross_modal_agreement(self, signals: Dict[str, ModalitySignal]) -> float:
        """Насколько модальности согласованы между собой (0..1)."""
        vals = [s.valence for s in signals.values()]
        arrs = [s.arousal for s in signals.values()]
        if len(vals) < 2:
            return 1.0
        std_v = float(np.std(vals))
        std_a = float(np.std(arrs))
        # Чем меньше разброс — тем выше согласованность
        agreement = 1.0 - min(1.0, (std_v + std_a) / 2.0)
        return round(agreement, 4)

    def get_trend(self, window: int = 10) -> Dict:
        """Тренд эмоционального состояния за последние N измерений."""
        if len(self._history) < 2:
            return {"trend": "insufficient_data"}
        recent = list(self._history)[-window:]
        vals = [h["esv"].valence for h in recent]
        arrs = [h["esv"].arousal for h in recent]
        v_trend = "rising" if vals[-1] > vals[0] + 0.1 else "falling" if vals[-1] < vals[0] - 0.1 else "stable"
        a_trend = "rising" if arrs[-1] > arrs[0] + 0.1 else "falling" if arrs[-1] < arrs[0] - 0.1 else "stable"
        return {
            "valence_trend": v_trend,
            "arousal_trend": a_trend,
            "mean_valence": round(float(np.mean(vals)), 4),
            "mean_arousal": round(float(np.mean(arrs)), 4),
            "window": len(recent),
        }

    def get_weights(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.weights.items()}

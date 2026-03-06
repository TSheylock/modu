"""
Адаптер EmotionModule для SASOK BaseModule.
Оборачивает MultimodalEmotionAnalyzer из emotion_analysis.py в интерфейс BaseModule.
"""
import json
import asyncio
from typing import Dict, Any
from core.base_module import BaseModule


class EmotionModuleAdapter(BaseModule):
    """Модуль эмоций SASOK — адаптер для MultimodalEmotionAnalyzer."""

    async def initialize(self):
        """Инициализация модуля эмоций."""
        self.logger.info("Инициализация модуля эмоций (адаптер)...")

        self.analyzer = None  # ленивая загрузка — тяжёлые модели
        self.state = {
            "active": False,
            "analysis_count": 0,
            "last_emotion": None,
            "dominant_emotion": "neutral"
        }

        self.logger.info("Модуль эмоций инициализирован (адаптер)")

    def _ensure_analyzer(self):
        """Ленивая загрузка анализатора."""
        if self.analyzer is None:
            try:
                from modules.emotion.emotion_analysis import MultimodalEmotionAnalyzer
                nats_url = self.config.get("event_bus", {}).get("servers", ["nats://localhost:4222"])[0]
                self.analyzer = MultimodalEmotionAnalyzer(nats_url=nats_url)
                self.logger.info("MultimodalEmotionAnalyzer загружен")
            except Exception as e:
                self.logger.warning(f"MultimodalEmotionAnalyzer недоступен: {e}")

    async def activate(self):
        """Активация модуля эмоций."""
        if self.active:
            return

        self.logger.info("Активация модуля эмоций...")
        self.active = True
        await self.update_state({"active": True})
        self.logger.info("Модуль эмоций активирован")

    async def deactivate(self):
        """Деактивация модуля эмоций."""
        if not self.active:
            return

        self.logger.info("Деактивация модуля эмоций...")

        for subscription in self.subscriptions:
            await subscription.unsubscribe()
        self.subscriptions = []

        if self.analyzer:
            try:
                await self.analyzer.close()
            except Exception:
                pass

        self.active = False
        await self.update_state({"active": False})
        self.logger.info("Модуль эмоций деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса на анализ эмоций."""
        if not self.active:
            return {"error": "Module inactive"}

        self._ensure_analyzer()

        if not self.analyzer:
            return {
                "emotion": "neutral",
                "score": 0.0,
                "error": "Analyzer not available (heavy models not loaded)"
            }

        try:
            result = await self.analyzer.analyze(
                text=data.get("text"),
                audio_data=data.get("audio"),
                video_data=data.get("video"),
                publish_events=data.get("publish", True)
            )

            self.state["analysis_count"] += 1
            self.state["last_emotion"] = result.get("dominant_emotion", "neutral")
            self.state["dominant_emotion"] = result.get("dominant_emotion", "neutral")

            # Публикация события для других модулей
            await self.publish(
                "emotion.state_changed",
                json.dumps({
                    "emotions": {result.get("dominant_emotion", "neutral"): result.get("dominant_score", 0.5)},
                    "context": data.get("context", "analysis_request")
                }).encode("utf-8")
            )

            return result
        except Exception as e:
            self.logger.error(f"Ошибка анализа эмоций: {e}")
            return {"emotion": "neutral", "score": 0.0, "error": str(e)}

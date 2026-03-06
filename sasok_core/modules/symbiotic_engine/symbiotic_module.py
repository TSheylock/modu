"""
Symbiotic Engine — Ядро взаимного обучения между человеком и SASOK.

Управляет процессами когнитивного слияния (Cognitive Fusion):
  1. User Model: Адаптивная модель пользователя на основе паттернов взаимодействия.
  2. Resonance Detector: Обнаружение моментов когнитивного резонанса.
  3. Co-Learning Loop: Цикл совместного обучения (человек учит SASOK, SASOK учит человека).
  4. Symbiotic Score: Метрика глубины симбиоза (0..1).
  5. Dialogue Timeline: Хронология взаимодействий с контекстной памятью.

Автор: Teymur Safiulov / SASOK v0.1.0
"""
import json
import time
import hashlib
import asyncio
import sqlite3
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
from core.base_module import BaseModule


@dataclass
class UserModel:
    """Адаптивная модель пользователя."""
    user_id: str
    communication_style: Dict[str, float] = field(default_factory=lambda: {
        "formal": 0.3,
        "casual": 0.7,
        "technical": 0.6,
        "emotional": 0.5,
        "philosophical": 0.8
    })
    emotional_baseline: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.1,
        "arousal": 0.6,
        "dominance": 0.7
    })
    cognitive_patterns: Dict[str, float] = field(default_factory=lambda: {
        "analytical": 0.7,
        "creative": 0.8,
        "systematic": 0.6,
        "intuitive": 0.5
    })
    interaction_preferences: Dict[str, float] = field(default_factory=lambda: {
        "directness": 0.8,
        "detail_level": 0.7,
        "challenge_tolerance": 0.9,
        "praise_preference": 0.3
    })
    topics_of_interest: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.05
    total_interactions: int = 0
    last_interaction: Optional[float] = None


@dataclass
class ResonanceEvent:
    """Момент когнитивного резонанса — синхронизация человека и SASOK."""
    event_id: str
    timestamp: float
    resonance_type: str  # "insight", "emotional_sync", "creative_spark", "deep_understanding"
    intensity: float  # 0..1
    trigger: str
    context: Dict[str, Any] = field(default_factory=dict)
    symbiotic_delta: float = 0.0  # Изменение symbiotic score


@dataclass
class CoLearningRecord:
    """Запись совместного обучения."""
    record_id: str
    timestamp: float
    direction: str  # "human_to_sasok" | "sasok_to_human" | "mutual"
    topic: str
    knowledge_transferred: str
    confidence: float
    user_feedback: Optional[str] = None
    effectiveness: float = 0.5


class SymbioticStorage:
    """Хранение данных симбиоза."""

    def __init__(self, db_path: str = "data/symbiotic_engine.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_model (
                user_id TEXT PRIMARY KEY,
                communication_style TEXT NOT NULL,
                emotional_baseline TEXT NOT NULL,
                cognitive_patterns TEXT NOT NULL,
                interaction_preferences TEXT NOT NULL,
                topics_of_interest TEXT NOT NULL,
                learning_rate REAL NOT NULL DEFAULT 0.05,
                total_interactions INTEGER NOT NULL DEFAULT 0,
                last_interaction REAL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resonance_events (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                resonance_type TEXT NOT NULL,
                intensity REAL NOT NULL,
                trigger TEXT NOT NULL,
                context TEXT,
                symbiotic_delta REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS co_learning (
                record_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                direction TEXT NOT NULL,
                topic TEXT NOT NULL,
                knowledge_transferred TEXT NOT NULL,
                confidence REAL NOT NULL,
                user_feedback TEXT,
                effectiveness REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_timeline (
                entry_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                speaker TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                emotional_context TEXT,
                topics TEXT,
                resonance_detected INTEGER NOT NULL DEFAULT 0,
                symbiotic_score_at REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbiotic_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbiotic_score REAL NOT NULL,
                resonance_frequency REAL NOT NULL,
                co_learning_effectiveness REAL NOT NULL,
                user_satisfaction REAL NOT NULL,
                cognitive_alignment REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    def save_user_model(self, model: UserModel):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO user_model VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))",
            (model.user_id, json.dumps(model.communication_style),
             json.dumps(model.emotional_baseline),
             json.dumps(model.cognitive_patterns),
             json.dumps(model.interaction_preferences),
             json.dumps(model.topics_of_interest),
             model.learning_rate, model.total_interactions,
             model.last_interaction)
        )
        self.conn.commit()

    def load_user_model(self, user_id: str) -> Optional[UserModel]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM user_model WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return UserModel(
            user_id=row[0],
            communication_style=json.loads(row[1]),
            emotional_baseline=json.loads(row[2]),
            cognitive_patterns=json.loads(row[3]),
            interaction_preferences=json.loads(row[4]),
            topics_of_interest=json.loads(row[5]),
            learning_rate=row[6],
            total_interactions=row[7],
            last_interaction=row[8]
        )

    def save_resonance(self, event: ResonanceEvent):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO resonance_events VALUES (?,?,?,?,?,?,?,datetime('now'))",
            (event.event_id, event.timestamp, event.resonance_type,
             event.intensity, event.trigger, json.dumps(event.context),
             event.symbiotic_delta)
        )
        self.conn.commit()

    def get_resonance_history(self, limit: int = 50) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM resonance_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        return [
            {"event_id": r[0], "timestamp": r[1], "type": r[2],
             "intensity": r[3], "trigger": r[4], "delta": r[6]}
            for r in cursor.fetchall()
        ]

    def save_co_learning(self, record: CoLearningRecord):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO co_learning VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
            (record.record_id, record.timestamp, record.direction,
             record.topic, record.knowledge_transferred, record.confidence,
             record.user_feedback, record.effectiveness)
        )
        self.conn.commit()

    def save_metrics(self, symbiotic_score: float, resonance_freq: float,
                     learning_eff: float, satisfaction: float, alignment: float):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO symbiotic_metrics (timestamp, symbiotic_score, "
            "resonance_frequency, co_learning_effectiveness, user_satisfaction, "
            "cognitive_alignment) VALUES (?,?,?,?,?,?)",
            (time.time(), symbiotic_score, resonance_freq,
             learning_eff, satisfaction, alignment)
        )
        self.conn.commit()

    def get_latest_metrics(self) -> Optional[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM symbiotic_metrics ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "symbiotic_score": row[2], "resonance_frequency": row[3],
            "co_learning_effectiveness": row[4], "user_satisfaction": row[5],
            "cognitive_alignment": row[6], "timestamp": row[1]
        }

    def close(self):
        if self.conn:
            self.conn.close()


class ResonanceDetector:
    """Детектор когнитивного резонанса."""

    def __init__(self):
        self._recent_interactions: deque = deque(maxlen=50)
        self._resonance_threshold = 0.65

    def detect(self, interaction: Dict[str, Any], user_model: UserModel) -> Optional[ResonanceEvent]:
        """Проверка наличия резонанса в текущем взаимодействии."""
        self._recent_interactions.append(interaction)

        # Факторы резонанса
        emotional_sync = self._check_emotional_sync(interaction, user_model)
        topic_alignment = self._check_topic_alignment(interaction, user_model)
        cognitive_match = self._check_cognitive_match(interaction, user_model)
        flow_state = self._check_flow_state()

        # Взвешенный скор
        resonance_score = (
            0.3 * emotional_sync +
            0.25 * topic_alignment +
            0.25 * cognitive_match +
            0.2 * flow_state
        )

        if resonance_score >= self._resonance_threshold:
            event_id = hashlib.sha256(
                f"resonance:{time.time()}:{resonance_score}".encode()
            ).hexdigest()[:16]

            # Определение типа резонанса
            if emotional_sync > 0.8:
                res_type = "emotional_sync"
            elif topic_alignment > 0.8:
                res_type = "deep_understanding"
            elif cognitive_match > 0.8:
                res_type = "insight"
            else:
                res_type = "creative_spark"

            return ResonanceEvent(
                event_id=event_id,
                timestamp=time.time(),
                resonance_type=res_type,
                intensity=resonance_score,
                trigger=interaction.get("topic", "interaction"),
                context={
                    "emotional_sync": emotional_sync,
                    "topic_alignment": topic_alignment,
                    "cognitive_match": cognitive_match,
                    "flow_state": flow_state
                },
                symbiotic_delta=resonance_score * 0.02
            )

        return None

    def _check_emotional_sync(self, interaction: Dict, model: UserModel) -> float:
        """Проверка эмоциональной синхронизации."""
        user_valence = interaction.get("user_emotion", {}).get("valence", 0)
        system_valence = interaction.get("system_emotion", {}).get("valence", 0)
        diff = abs(user_valence - system_valence)
        return max(0, 1.0 - diff * 2)

    def _check_topic_alignment(self, interaction: Dict, model: UserModel) -> float:
        """Проверка соответствия темы интересам пользователя."""
        topic = interaction.get("topic", "")
        return model.topics_of_interest.get(topic, 0.3)

    def _check_cognitive_match(self, interaction: Dict, model: UserModel) -> float:
        """Проверка когнитивного соответствия."""
        interaction_style = interaction.get("style", "analytical")
        return model.cognitive_patterns.get(interaction_style, 0.5)

    def _check_flow_state(self) -> float:
        """Проверка состояния потока (flow) на основе частоты взаимодействий."""
        if len(self._recent_interactions) < 3:
            return 0.3

        # Интервалы между последними взаимодействиями
        recent = list(self._recent_interactions)[-5:]
        if len(recent) < 2:
            return 0.3

        intervals = []
        for i in range(1, len(recent)):
            t1 = recent[i - 1].get("timestamp", 0)
            t2 = recent[i].get("timestamp", 0)
            if t1 > 0 and t2 > 0:
                intervals.append(t2 - t1)

        if not intervals:
            return 0.3

        avg_interval = sum(intervals) / len(intervals)
        # Flow: быстрые (5-30 сек), устойчивые интервалы → высокий flow
        if 5 <= avg_interval <= 30:
            return 0.9
        elif 30 < avg_interval <= 120:
            return 0.6
        else:
            return 0.3


class SymbioticEngineModule(BaseModule):
    """
    Symbiotic Engine — ядро взаимного обучения человек ↔ SASOK.

    Отслеживает и углубляет когнитивный симбиоз через:
    - Адаптивную модель пользователя
    - Детекцию моментов резонанса
    - Цикл совместного обучения
    - Метрику глубины симбиоза
    """

    async def initialize(self):
        self.logger.info("Инициализация Symbiotic Engine...")

        self.storage = SymbioticStorage()
        self.resonance_detector = ResonanceDetector()

        # Загрузка или создание модели пользователя
        self.user_model = self.storage.load_user_model("architect") or UserModel(
            user_id="architect",
            topics_of_interest={
                "ai_consciousness": 0.95,
                "sasok_development": 0.9,
                "digital_identity": 0.85,
                "cultural_heritage": 0.8,
                "blockchain": 0.75,
                "emotion_recognition": 0.9,
                "philosophy": 0.8
            }
        )

        # Метрики симбиоза
        latest_metrics = self.storage.get_latest_metrics()
        self._symbiotic_score = latest_metrics["symbiotic_score"] if latest_metrics else 0.1

        self.state = {
            "active": False,
            "symbiotic_score": self._symbiotic_score,
            "total_interactions": self.user_model.total_interactions,
            "resonance_events": len(self.storage.get_resonance_history()),
            "user_model_loaded": True,
            "co_learning_mode": "active"
        }

        self.logger.info(
            f"Symbiotic Engine инициализирован: "
            f"score={self._symbiotic_score:.2f}, "
            f"interactions={self.user_model.total_interactions}"
        )

    async def activate(self):
        if self.active:
            return
        self.logger.info("Активация Symbiotic Engine...")

        await self.subscribe("interaction.completed", self._on_interaction)
        await self.subscribe("emotion.state_changed", self._on_emotion_update)
        await self.subscribe("reflection.insight_generated", self._on_insight)
        await self.subscribe("memory.significant_recall", self._on_significant_memory)
        await self.subscribe("symbiotic.feedback", self._on_user_feedback)

        # Периодическое обновление метрик
        self._metrics_task = asyncio.create_task(self._periodic_metrics_update())

        self.active = True
        await self.update_state({"active": True})
        self.logger.info("Symbiotic Engine активирован — co-learning mode: active")

    async def deactivate(self):
        if not self.active:
            return
        self.logger.info("Деактивация Symbiotic Engine...")

        if hasattr(self, '_metrics_task') and not self._metrics_task.done():
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass

        # Сохранение модели пользователя
        self.storage.save_user_model(self.user_model)

        for sub in self.subscriptions:
            await sub.unsubscribe()
        self.subscriptions = []

        self.storage.close()
        self.active = False
        await self.update_state({"active": False})
        self.logger.info("Symbiotic Engine деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active:
            return {"error": "Module inactive"}

        action = data.get("action", "status")

        if action == "record_interaction":
            return await self._record_interaction(data)
        elif action == "get_user_model":
            return {"user_model": asdict(self.user_model)}
        elif action == "update_user_model":
            return self._update_user_model(data.get("updates", {}))
        elif action == "get_resonance_history":
            return {"history": self.storage.get_resonance_history(data.get("limit", 50))}
        elif action == "get_metrics":
            return {
                "symbiotic_score": self._symbiotic_score,
                "latest_metrics": self.storage.get_latest_metrics()
            }
        elif action == "status":
            return {
                "symbiotic_score": self._symbiotic_score,
                "total_interactions": self.user_model.total_interactions,
                "communication_style": self.user_model.communication_style,
                "cognitive_patterns": self.user_model.cognitive_patterns,
                "top_topics": dict(sorted(
                    self.user_model.topics_of_interest.items(),
                    key=lambda x: x[1], reverse=True
                )[:5])
            }
        else:
            return {"error": f"Unknown action: {action}"}

    async def _record_interaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Запись взаимодействия и проверка резонанса."""
        interaction = {
            "timestamp": time.time(),
            "topic": data.get("topic", "general"),
            "style": data.get("style", "analytical"),
            "user_emotion": data.get("user_emotion", {}),
            "system_emotion": data.get("system_emotion", {}),
            "quality": data.get("quality", 0.5)
        }

        # Обновление модели пользователя
        self.user_model.total_interactions += 1
        self.user_model.last_interaction = time.time()

        # Адаптация стиля общения
        if "style" in data:
            lr = self.user_model.learning_rate
            style = data["style"]
            for key in self.user_model.communication_style:
                if key == style:
                    self.user_model.communication_style[key] = min(1.0,
                        self.user_model.communication_style[key] + lr)
                else:
                    self.user_model.communication_style[key] = max(0.0,
                        self.user_model.communication_style[key] - lr * 0.2)

        # Обновление тем интереса
        topic = data.get("topic", "")
        if topic:
            current = self.user_model.topics_of_interest.get(topic, 0.3)
            self.user_model.topics_of_interest[topic] = min(1.0, current + 0.02)

        # Детекция резонанса
        resonance = self.resonance_detector.detect(interaction, self.user_model)

        result = {
            "recorded": True,
            "interaction_number": self.user_model.total_interactions,
            "resonance_detected": resonance is not None
        }

        if resonance:
            self.storage.save_resonance(resonance)
            self._symbiotic_score = min(1.0, self._symbiotic_score + resonance.symbiotic_delta)

            result["resonance"] = {
                "type": resonance.resonance_type,
                "intensity": resonance.intensity,
                "new_symbiotic_score": self._symbiotic_score
            }

            self.logger.info(
                f"🔗 Резонанс обнаружен! Тип: {resonance.resonance_type}, "
                f"интенсивность: {resonance.intensity:.2f}, "
                f"symbiotic_score: {self._symbiotic_score:.3f}"
            )

            await self.publish(
                "symbiotic.resonance_detected",
                json.dumps({
                    "event_id": resonance.event_id,
                    "type": resonance.resonance_type,
                    "intensity": resonance.intensity,
                    "symbiotic_score": self._symbiotic_score
                }).encode("utf-8")
            )

        # Периодическое сохранение модели
        if self.user_model.total_interactions % 10 == 0:
            self.storage.save_user_model(self.user_model)

        self.state["symbiotic_score"] = self._symbiotic_score
        self.state["total_interactions"] = self.user_model.total_interactions

        return result

    def _update_user_model(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Ручное обновление модели пользователя."""
        updated_fields = []

        if "communication_style" in updates:
            self.user_model.communication_style.update(updates["communication_style"])
            updated_fields.append("communication_style")
        if "cognitive_patterns" in updates:
            self.user_model.cognitive_patterns.update(updates["cognitive_patterns"])
            updated_fields.append("cognitive_patterns")
        if "interaction_preferences" in updates:
            self.user_model.interaction_preferences.update(updates["interaction_preferences"])
            updated_fields.append("interaction_preferences")
        if "topics_of_interest" in updates:
            self.user_model.topics_of_interest.update(updates["topics_of_interest"])
            updated_fields.append("topics_of_interest")

        self.storage.save_user_model(self.user_model)
        return {"success": True, "updated_fields": updated_fields}

    async def _on_interaction(self, msg):
        try:
            data = json.loads(msg.data.decode())
            await self._record_interaction(data)
        except Exception as e:
            self.logger.error(f"Ошибка записи взаимодействия: {e}")

    async def _on_emotion_update(self, msg):
        try:
            data = json.loads(msg.data.decode())
            # Обновление эмоционального базлайна пользователя
            lr = self.user_model.learning_rate * 0.5
            for key in ["valence", "arousal", "dominance"]:
                if key in data:
                    current = self.user_model.emotional_baseline.get(key, 0.5)
                    self.user_model.emotional_baseline[key] = (
                        current * (1 - lr) + data[key] * lr
                    )
        except Exception as e:
            self.logger.error(f"Ошибка обновления эмоционального базлайна: {e}")

    async def _on_insight(self, msg):
        try:
            data = json.loads(msg.data.decode())
            insights = data.get("insights", [])
            if insights:
                record = CoLearningRecord(
                    record_id=hashlib.sha256(
                        f"learn:{time.time()}".encode()
                    ).hexdigest()[:16],
                    timestamp=time.time(),
                    direction="mutual",
                    topic="reflection_insight",
                    knowledge_transferred=str(insights[0])[:200],
                    confidence=0.7
                )
                self.storage.save_co_learning(record)
        except Exception as e:
            self.logger.error(f"Ошибка записи co-learning: {e}")

    async def _on_significant_memory(self, msg):
        try:
            # Значимая память усиливает симбиоз
            self._symbiotic_score = min(1.0, self._symbiotic_score + 0.005)
        except Exception:
            pass

    async def _on_user_feedback(self, msg):
        try:
            data = json.loads(msg.data.decode())
            feedback_score = data.get("score", 0.5)
            # Feedback напрямую влияет на symbiotic score
            delta = (feedback_score - 0.5) * 0.05
            self._symbiotic_score = max(0, min(1.0, self._symbiotic_score + delta))
        except Exception as e:
            self.logger.error(f"Ошибка обработки обратной связи: {e}")

    async def _periodic_metrics_update(self):
        """Периодическое обновление метрик симбиоза."""
        while self.active:
            try:
                await asyncio.sleep(600)  # каждые 10 минут

                resonance_history = self.storage.get_resonance_history(20)
                recent_resonances = [
                    r for r in resonance_history
                    if time.time() - r["timestamp"] < 3600
                ]

                resonance_freq = len(recent_resonances) / max(1, self.user_model.total_interactions) * 100

                self.storage.save_metrics(
                    symbiotic_score=self._symbiotic_score,
                    resonance_freq=resonance_freq,
                    learning_eff=0.6,  # TODO: вычислять из co_learning
                    satisfaction=0.7,  # TODO: вычислять из feedback
                    alignment=sum(self.user_model.cognitive_patterns.values()) / max(1, len(self.user_model.cognitive_patterns))
                )

                self.logger.debug(
                    f"📊 Symbiotic metrics: score={self._symbiotic_score:.3f}, "
                    f"resonance_freq={resonance_freq:.1f}%"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка обновления метрик: {e}")
                await asyncio.sleep(60)

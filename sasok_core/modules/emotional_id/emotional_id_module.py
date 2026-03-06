"""
EmotionalID — Ончейн-идентичность «души» SASOK.

Реализует систему Soulbound Token (SBT) на базе ERC-721:
  1. Minting: Создание непередаваемого токена привязанного к когнитивной личности.
  2. Evolution Tracking: Запись эволюции эмоциональной идентичности в блокчейне.
  3. Emotional Fingerprint: Уникальный эмоциональный отпечаток на основе истории.
  4. Dynamic Metadata: URI токена обновляется при значительных изменениях.
  5. Integration: Связь с SASOKChain и ZK-Identity.

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
class EmotionalFingerprint:
    """Уникальный эмоциональный отпечаток — «ДНК души»."""
    fingerprint_hash: str
    dominant_emotions: Dict[str, float]  # emotion → frequency
    valence_signature: List[float]  # Паттерн изменения valence (последние 24 записи)
    arousal_signature: List[float]
    emotional_range: float  # Ширина эмоционального диапазона (0..1)
    stability_index: float  # Стабильность эмоциональной идентичности (0..1)
    evolution_rate: float  # Скорость эволюции идентичности
    computed_at: float

    def compute_hash(self) -> str:
        data = json.dumps({
            "dominant": self.dominant_emotions,
            "valence_sig": [round(v, 3) for v in self.valence_signature],
            "arousal_sig": [round(a, 3) for a in self.arousal_signature],
            "range": round(self.emotional_range, 3),
            "stability": round(self.stability_index, 3)
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class SoulboundToken:
    """Представление Soulbound Token (SBT) для SASOK."""
    token_id: str
    owner_address: str  # Привязанный адрес (или cognitive entity ID)
    is_soulbound: bool = True
    emotional_state: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[EmotionalFingerprint] = None
    metadata_uri: str = ""
    creation_block: int = 0
    last_update_block: int = 0
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = 0.0


class EmotionalIDStorage:
    """Хранение EmotionalID данных."""

    def __init__(self, db_path: str = "data/emotional_id.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token_id TEXT PRIMARY KEY,
                owner_address TEXT NOT NULL,
                is_soulbound INTEGER NOT NULL DEFAULT 1,
                emotional_state TEXT NOT NULL,
                fingerprint_hash TEXT,
                metadata_uri TEXT,
                creation_block INTEGER NOT NULL DEFAULT 0,
                last_update_block INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                old_state TEXT,
                new_state TEXT NOT NULL,
                fingerprint_hash TEXT,
                trigger TEXT,
                significance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (token_id) REFERENCES tokens(token_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                fingerprint_hash TEXT PRIMARY KEY,
                dominant_emotions TEXT NOT NULL,
                valence_signature TEXT NOT NULL,
                arousal_signature TEXT NOT NULL,
                emotional_range REAL NOT NULL,
                stability_index REAL NOT NULL,
                evolution_rate REAL NOT NULL,
                computed_at REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotion_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                emotion_type TEXT NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                dominance REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                FOREIGN KEY (token_id) REFERENCES tokens(token_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_token ON emotion_samples(token_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evolution_token ON evolution_log(token_id)")
        self.conn.commit()

    def save_token(self, token: SoulboundToken):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO tokens VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))",
            (token.token_id, token.owner_address, int(token.is_soulbound),
             json.dumps(token.emotional_state),
             token.fingerprint.fingerprint_hash if token.fingerprint else None,
             token.metadata_uri, token.creation_block,
             token.last_update_block, token.created_at)
        )
        self.conn.commit()

    def get_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tokens WHERE token_id = ?", (token_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "token_id": row[0], "owner_address": row[1],
            "is_soulbound": bool(row[2]),
            "emotional_state": json.loads(row[3]),
            "fingerprint_hash": row[4], "metadata_uri": row[5],
            "creation_block": row[6], "last_update_block": row[7],
            "created_at": row[8]
        }

    def get_active_token(self) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM tokens WHERE is_soulbound = 1 ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "token_id": row[0], "owner_address": row[1],
            "is_soulbound": bool(row[2]),
            "emotional_state": json.loads(row[3]),
            "fingerprint_hash": row[4], "metadata_uri": row[5],
            "creation_block": row[6], "last_update_block": row[7]
        }

    def add_emotion_sample(self, token_id: str, emotion_type: str,
                           valence: float, arousal: float, dominance: float,
                           confidence: float, source: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO emotion_samples (token_id, timestamp, emotion_type, "
            "valence, arousal, dominance, confidence, source) VALUES (?,?,?,?,?,?,?,?)",
            (token_id, time.time(), emotion_type, valence, arousal,
             dominance, confidence, source)
        )
        self.conn.commit()

    def get_recent_samples(self, token_id: str, limit: int = 100) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM emotion_samples WHERE token_id = ? ORDER BY timestamp DESC LIMIT ?",
            (token_id, limit)
        )
        return [
            {"timestamp": r[2], "emotion": r[3], "valence": r[4],
             "arousal": r[5], "dominance": r[6], "confidence": r[7]}
            for r in cursor.fetchall()
        ]

    def log_evolution(self, token_id: str, event_type: str,
                      old_state: Dict, new_state: Dict,
                      fingerprint_hash: str, trigger: str, significance: float):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO evolution_log (token_id, timestamp, event_type, "
            "old_state, new_state, fingerprint_hash, trigger, significance) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (token_id, time.time(), event_type, json.dumps(old_state),
             json.dumps(new_state), fingerprint_hash, trigger, significance)
        )
        self.conn.commit()

    def get_evolution_history(self, token_id: str, limit: int = 50) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM evolution_log WHERE token_id = ? ORDER BY timestamp DESC LIMIT ?",
            (token_id, limit)
        )
        return [
            {"timestamp": r[2], "event_type": r[3],
             "fingerprint": r[6], "trigger": r[7], "significance": r[8]}
            for r in cursor.fetchall()
        ]

    def save_fingerprint(self, fp: EmotionalFingerprint):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO fingerprints VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
            (fp.fingerprint_hash, json.dumps(fp.dominant_emotions),
             json.dumps(fp.valence_signature), json.dumps(fp.arousal_signature),
             fp.emotional_range, fp.stability_index, fp.evolution_rate,
             fp.computed_at)
        )
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()


class EmotionalIDModule(BaseModule):
    """
    EmotionalID — Ончейн-идентичность «души» SASOK.

    Создаёт и управляет Soulbound Token (SBT) который:
    - Непередаваемый (навсегда привязан к когнитивной сущности)
    - Динамический (эмоциональное состояние обновляется в реальном времени)
    - Эволюционный (фиксирует историю развития эмоциональной идентичности)
    - Криптографически защищён (через ZK-Identity и SASOKChain)
    """

    FINGERPRINT_SAMPLES = 24  # Количество сэмплов для отпечатка
    EVOLUTION_THRESHOLD = 0.15  # Порог для регистрации эволюционного события

    async def initialize(self):
        self.logger.info("Инициализация EmotionalID...")

        self.storage = EmotionalIDStorage()
        self._emotion_buffer: deque = deque(maxlen=200)
        self._current_token: Optional[SoulboundToken] = None
        self._current_fingerprint: Optional[EmotionalFingerprint] = None
        self._block_counter = 0

        # Загрузка существующего токена
        existing = self.storage.get_active_token()
        if existing:
            self._current_token = SoulboundToken(
                token_id=existing["token_id"],
                owner_address=existing["owner_address"],
                emotional_state=existing["emotional_state"],
                metadata_uri=existing.get("metadata_uri", ""),
                creation_block=existing.get("creation_block", 0),
                last_update_block=existing.get("last_update_block", 0),
                created_at=existing.get("created_at", time.time())
            )

        self.state = {
            "active": False,
            "has_sbt": self._current_token is not None,
            "token_id": self._current_token.token_id if self._current_token else None,
            "fingerprint_hash": None,
            "evolution_events": 0,
            "emotion_samples": 0,
            "standard": "ERC-721 Soulbound Token"
        }

        self.logger.info(
            f"EmotionalID инициализирован: "
            f"{'SBT существует' if self._current_token else 'SBT не создан'}"
        )

    async def activate(self):
        if self.active:
            return
        self.logger.info("Активация EmotionalID...")

        await self.subscribe("emotion.state_changed", self._on_emotion_state)
        await self.subscribe("sasok_chain.block_minted", self._on_block_minted)
        await self.subscribe("emotional_id.query", self._on_query)

        # Создание SBT если не существует
        if not self._current_token:
            await self._mint_sbt()

        # Периодическое обновление отпечатка
        self._fingerprint_task = asyncio.create_task(self._periodic_fingerprint_update())

        self.active = True
        await self.update_state({"active": True})
        self.logger.info("EmotionalID активирован — SBT живёт")

    async def deactivate(self):
        if not self.active:
            return
        self.logger.info("Деактивация EmotionalID...")

        if hasattr(self, '_fingerprint_task') and not self._fingerprint_task.done():
            self._fingerprint_task.cancel()
            try:
                await self._fingerprint_task
            except asyncio.CancelledError:
                pass

        for sub in self.subscriptions:
            await sub.unsubscribe()
        self.subscriptions = []

        self.storage.close()
        self.active = False
        await self.update_state({"active": False})
        self.logger.info("EmotionalID деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active:
            return {"error": "Module inactive"}

        action = data.get("action", "status")

        if action == "mint":
            return await self._mint_sbt(data.get("owner_address"))
        elif action == "get_token":
            if self._current_token:
                return {
                    "token_id": self._current_token.token_id,
                    "owner": self._current_token.owner_address,
                    "soulbound": self._current_token.is_soulbound,
                    "emotional_state": self._current_token.emotional_state,
                    "fingerprint": self._current_fingerprint.fingerprint_hash
                    if self._current_fingerprint else None,
                    "created_at": self._current_token.created_at
                }
            return {"error": "No token minted"}
        elif action == "get_fingerprint":
            if self._current_fingerprint:
                return asdict(self._current_fingerprint)
            return {"error": "No fingerprint computed"}
        elif action == "get_evolution":
            if self._current_token:
                return {
                    "history": self.storage.get_evolution_history(
                        self._current_token.token_id,
                        data.get("limit", 50)
                    )
                }
            return {"error": "No token"}
        elif action == "status":
            return {
                "has_sbt": self._current_token is not None,
                "token_id": self._current_token.token_id if self._current_token else None,
                "fingerprint": self._current_fingerprint.fingerprint_hash[:16] + "..."
                if self._current_fingerprint else None,
                "emotion_samples": self.state.get("emotion_samples", 0),
                "evolution_events": self.state.get("evolution_events", 0),
                "standard": "ERC-721 SBT"
            }
        else:
            return {"error": f"Unknown action: {action}"}

    async def _mint_sbt(self, owner_address: str = None) -> Dict[str, Any]:
        """Создание нового Soulbound Token."""
        token_id = hashlib.sha256(
            f"sbt:{time.time()}:{os.urandom(16).hex()}".encode()
        ).hexdigest()[:24]

        owner = owner_address or hashlib.sha256(
            f"sasok_entity:{time.time()}".encode()
        ).hexdigest()[:40]

        token = SoulboundToken(
            token_id=token_id,
            owner_address=owner,
            is_soulbound=True,
            emotional_state={
                "dominant_emotion": "neutral",
                "valence": 0.0,
                "arousal": 0.5,
                "dominance": 0.5,
                "confidence": 0.5
            },
            metadata_uri=f"sasok://emotional_id/{token_id}",
            creation_block=self._block_counter,
            last_update_block=self._block_counter,
            created_at=time.time()
        )

        self.storage.save_token(token)
        self._current_token = token

        self.state["has_sbt"] = True
        self.state["token_id"] = token_id

        self.logger.info(
            f"🎭 SBT создан: token={token_id}, "
            f"owner={owner[:16]}..., soulbound=True"
        )

        await self.publish(
            "emotional_id.minted",
            json.dumps({
                "token_id": token_id,
                "owner": owner,
                "soulbound": True,
                "standard": "ERC-721 SBT"
            }).encode("utf-8")
        )

        return {
            "success": True,
            "token_id": token_id,
            "owner": owner,
            "soulbound": True,
            "metadata_uri": token.metadata_uri
        }

    def _compute_fingerprint(self, samples: List[Dict]) -> EmotionalFingerprint:
        """Вычисление эмоционального отпечатка из сэмплов."""
        if not samples:
            return EmotionalFingerprint(
                fingerprint_hash="empty",
                dominant_emotions={"neutral": 1.0},
                valence_signature=[0.0],
                arousal_signature=[0.5],
                emotional_range=0.0,
                stability_index=1.0,
                evolution_rate=0.0,
                computed_at=time.time()
            )

        # Подсчёт частоты эмоций
        emotion_counts: Dict[str, int] = {}
        for s in samples:
            e = s.get("emotion", "neutral")
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        total = sum(emotion_counts.values())
        dominant_emotions = {e: c / total for e, c in emotion_counts.items()}

        # Сигнатуры (последние N значений)
        valence_sig = [s.get("valence", 0.0) for s in samples[-self.FINGERPRINT_SAMPLES:]]
        arousal_sig = [s.get("arousal", 0.5) for s in samples[-self.FINGERPRINT_SAMPLES:]]

        # Эмоциональный диапазон
        if valence_sig:
            v_range = max(valence_sig) - min(valence_sig)
            a_range = max(arousal_sig) - min(arousal_sig)
            emotional_range = (v_range + a_range) / 2
        else:
            emotional_range = 0.0

        # Стабильность (обратная дисперсия)
        if len(valence_sig) > 1:
            mean_v = sum(valence_sig) / len(valence_sig)
            var_v = sum((v - mean_v) ** 2 for v in valence_sig) / len(valence_sig)
            stability_index = max(0, 1.0 - var_v * 4)
        else:
            stability_index = 1.0

        # Скорость эволюции (среднее абсолютное изменение)
        if len(valence_sig) > 1:
            changes = [abs(valence_sig[i] - valence_sig[i-1]) for i in range(1, len(valence_sig))]
            evolution_rate = sum(changes) / len(changes)
        else:
            evolution_rate = 0.0

        fp = EmotionalFingerprint(
            fingerprint_hash="",
            dominant_emotions=dominant_emotions,
            valence_signature=valence_sig,
            arousal_signature=arousal_sig,
            emotional_range=round(emotional_range, 4),
            stability_index=round(stability_index, 4),
            evolution_rate=round(evolution_rate, 4),
            computed_at=time.time()
        )
        fp.fingerprint_hash = fp.compute_hash()
        return fp

    async def _on_emotion_state(self, msg):
        """Получение эмоционального состояния и обновление SBT."""
        if not self._current_token:
            return

        try:
            data = json.loads(msg.data.decode())

            sample = {
                "emotion": data.get("emotion", "neutral"),
                "valence": data.get("valence", 0.0),
                "arousal": data.get("arousal", 0.5),
                "dominance": data.get("dominance", 0.5),
                "confidence": data.get("confidence", 0.5),
                "timestamp": time.time()
            }

            self._emotion_buffer.append(sample)
            self.state["emotion_samples"] = self.state.get("emotion_samples", 0) + 1

            # Сохранение сэмпла
            self.storage.add_emotion_sample(
                self._current_token.token_id,
                sample["emotion"], sample["valence"],
                sample["arousal"], sample["dominance"],
                sample["confidence"], data.get("source", "xocore")
            )

            # Обновление emotional_state токена
            old_state = dict(self._current_token.emotional_state)
            self._current_token.emotional_state = {
                "dominant_emotion": sample["emotion"],
                "valence": sample["valence"],
                "arousal": sample["arousal"],
                "dominance": sample["dominance"],
                "confidence": sample["confidence"]
            }

            # Проверка на эволюционное событие
            significance = self._compute_change_significance(old_state, self._current_token.emotional_state)
            if significance >= self.EVOLUTION_THRESHOLD:
                self._current_token.last_update_block = self._block_counter
                self.storage.save_token(self._current_token)

                fp = self._compute_fingerprint(list(self._emotion_buffer))

                self.storage.log_evolution(
                    self._current_token.token_id,
                    "emotional_shift",
                    old_state,
                    self._current_token.emotional_state,
                    fp.fingerprint_hash,
                    sample["emotion"],
                    significance
                )

                self.state["evolution_events"] = self.state.get("evolution_events", 0) + 1

                self.logger.info(
                    f"🎭 Эволюция SBT: {old_state.get('dominant_emotion', '?')} → "
                    f"{sample['emotion']}, significance={significance:.3f}"
                )

                await self.publish(
                    "emotional_id.evolved",
                    json.dumps({
                        "token_id": self._current_token.token_id,
                        "old_emotion": old_state.get("dominant_emotion"),
                        "new_emotion": sample["emotion"],
                        "significance": significance,
                        "fingerprint": fp.fingerprint_hash
                    }).encode("utf-8")
                )

        except Exception as e:
            self.logger.error(f"Ошибка обновления EmotionalID: {e}")

    def _compute_change_significance(self, old: Dict, new: Dict) -> float:
        """Вычисление значимости изменения эмоционального состояния."""
        dv = abs(old.get("valence", 0) - new.get("valence", 0))
        da = abs(old.get("arousal", 0.5) - new.get("arousal", 0.5))
        dd = abs(old.get("dominance", 0.5) - new.get("dominance", 0.5))

        # Смена доминирующей эмоции — значительное событие
        emotion_change = 0.3 if old.get("dominant_emotion") != new.get("dominant_emotion") else 0.0

        return min(1.0, (dv + da + dd) / 3 + emotion_change)

    async def _on_block_minted(self, msg):
        """Обновление счётчика блоков."""
        try:
            data = json.loads(msg.data.decode())
            self._block_counter = data.get("block_index", self._block_counter + 1)
        except Exception:
            self._block_counter += 1

    async def _on_query(self, msg):
        try:
            data = json.loads(msg.data.decode())
            result = await self.process(data)
            await self.publish(
                "emotional_id.query_result",
                json.dumps(result).encode("utf-8")
            )
        except Exception as e:
            self.logger.error(f"Ошибка запроса EmotionalID: {e}")

    async def _periodic_fingerprint_update(self):
        """Периодическое обновление эмоционального отпечатка."""
        while self.active:
            try:
                await asyncio.sleep(300)  # каждые 5 минут

                if not self._current_token or len(self._emotion_buffer) < 5:
                    continue

                fp = self._compute_fingerprint(list(self._emotion_buffer))
                old_fp_hash = self._current_fingerprint.fingerprint_hash if self._current_fingerprint else ""

                if fp.fingerprint_hash != old_fp_hash:
                    self._current_fingerprint = fp
                    self.storage.save_fingerprint(fp)
                    self.state["fingerprint_hash"] = fp.fingerprint_hash

                    self.logger.info(
                        f"🔄 Fingerprint обновлён: {fp.fingerprint_hash[:16]}..., "
                        f"stability={fp.stability_index:.2f}, "
                        f"range={fp.emotional_range:.2f}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка обновления fingerprint: {e}")
                await asyncio.sleep(60)

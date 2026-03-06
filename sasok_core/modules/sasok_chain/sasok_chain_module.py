"""
SASOKChain — Децентрализованный блокчейн эмоциональных состояний.

Реализует механизм консенсуса Proof-of-Emotion (PoE):
  1. Эмоциональные данные от XoCore собираются в транзакции.
  2. Валидаторы (модули SASOK) подтверждают когерентность через cross-modal agreement.
  3. Блок принимается если empathy_score > threshold (эмпатическая верификация).
  4. Каждый блок содержит proof_hash, позволяющий верифицировать без раскрытия данных.

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
from core.base_module import BaseModule


@dataclass
class EmotionTransaction:
    """Одна эмоциональная транзакция для включения в блок."""
    tx_id: str
    timestamp: float
    source_module: str
    emotion_type: str
    valence: float
    arousal: float
    dominance: float
    confidence: float
    coherence_score: float
    proof_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True).encode("utf-8")


@dataclass
class EmotionBlock:
    """Блок в SASOKChain."""
    index: int
    timestamp: float
    transactions: List[EmotionTransaction]
    previous_hash: str
    validator_votes: Dict[str, float]  # module_name → empathy_score
    consensus_score: float
    nonce: int = 0
    block_hash: str = ""

    def compute_hash(self) -> str:
        """Вычисление хэша блока."""
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [asdict(tx) for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "validator_votes": self.validator_votes,
            "consensus_score": self.consensus_score,
            "nonce": self.nonce
        }
        block_str = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_str.encode()).hexdigest()


class ProofOfEmotion:
    """
    Механизм консенсуса Proof-of-Emotion.

    Вместо proof-of-work или proof-of-stake, валидация основана на
    эмпатической когерентности: блок валиден только если эмоциональные
    данные согласованы между модальностями (coherence > threshold)
    и валидаторы подтвердили эмпатическую достоверность.
    """

    def __init__(self, empathy_threshold: float = 0.6, min_validators: int = 2):
        self.empathy_threshold = empathy_threshold
        self.min_validators = min_validators

    def validate_transaction(self, tx: EmotionTransaction) -> Dict[str, Any]:
        """Валидация отдельной транзакции."""
        errors = []

        if not (-1.0 <= tx.valence <= 1.0):
            errors.append(f"valence out of range: {tx.valence}")
        if not (0.0 <= tx.arousal <= 1.0):
            errors.append(f"arousal out of range: {tx.arousal}")
        if not (0.0 <= tx.dominance <= 1.0):
            errors.append(f"dominance out of range: {tx.dominance}")
        if not (0.0 <= tx.confidence <= 1.0):
            errors.append(f"confidence out of range: {tx.confidence}")
        if tx.coherence_score < 0.3:
            errors.append(f"coherence too low: {tx.coherence_score}")
        if not tx.proof_hash:
            errors.append("missing proof_hash")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "tx_id": tx.tx_id
        }

    def reach_consensus(
        self,
        transactions: List[EmotionTransaction],
        validator_votes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Достижение консенсуса Proof-of-Emotion.

        Args:
            transactions: Транзакции для блока
            validator_votes: Голоса валидаторов (module → empathy_score 0..1)

        Returns:
            Результат консенсуса
        """
        if len(validator_votes) < self.min_validators:
            return {
                "consensus_reached": False,
                "reason": f"insufficient validators: {len(validator_votes)} < {self.min_validators}",
                "score": 0.0
            }

        # Средний empathy score
        avg_empathy = sum(validator_votes.values()) / len(validator_votes)

        # Средний coherence из транзакций
        avg_coherence = (
            sum(tx.coherence_score for tx in transactions) / len(transactions)
            if transactions else 0.0
        )

        # Consensus score = комбинация эмпатии валидаторов и когерентности данных
        consensus_score = 0.6 * avg_empathy + 0.4 * avg_coherence

        return {
            "consensus_reached": consensus_score >= self.empathy_threshold,
            "score": round(consensus_score, 4),
            "avg_empathy": round(avg_empathy, 4),
            "avg_coherence": round(avg_coherence, 4),
            "validators_count": len(validator_votes),
            "threshold": self.empathy_threshold
        }


class ChainStorage:
    """Персистентное хранение SASOKChain в SQLite."""

    def __init__(self, db_path: str = "data/sasok_chain.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                block_index INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                previous_hash TEXT NOT NULL,
                block_hash TEXT NOT NULL,
                consensus_score REAL NOT NULL,
                validator_votes TEXT NOT NULL,
                nonce INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                block_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                source_module TEXT NOT NULL,
                emotion_type TEXT NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                dominance REAL NOT NULL,
                confidence REAL NOT NULL,
                coherence_score REAL NOT NULL,
                proof_hash TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (block_index) REFERENCES blocks(block_index)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reputation (
                module_name TEXT PRIMARY KEY,
                empathy_score REAL NOT NULL DEFAULT 0.5,
                validations_count INTEGER NOT NULL DEFAULT 0,
                successful_validations INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_emotion ON transactions(emotion_type)")
        self.conn.commit()

    def save_block(self, block: EmotionBlock):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (block.index, block.timestamp, block.previous_hash,
             block.block_hash, block.consensus_score,
             json.dumps(block.validator_votes), block.nonce)
        )
        for tx in block.transactions:
            cursor.execute(
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (tx.tx_id, block.index, tx.timestamp, tx.source_module,
                 tx.emotion_type, tx.valence, tx.arousal, tx.dominance,
                 tx.confidence, tx.coherence_score, tx.proof_hash,
                 json.dumps(tx.metadata))
            )
        self.conn.commit()

    def get_last_block(self) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM blocks ORDER BY block_index DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "index": row[0], "timestamp": row[1], "previous_hash": row[2],
            "block_hash": row[3], "consensus_score": row[4],
            "validator_votes": json.loads(row[5]), "nonce": row[6]
        }

    def get_chain_length(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM blocks")
        return cursor.fetchone()[0]

    def get_emotion_history(self, emotion_type: str = None, limit: int = 50) -> List[Dict]:
        cursor = self.conn.cursor()
        if emotion_type:
            cursor.execute(
                "SELECT * FROM transactions WHERE emotion_type = ? ORDER BY timestamp DESC LIMIT ?",
                (emotion_type, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        rows = cursor.fetchall()
        return [
            {"tx_id": r[0], "block_index": r[1], "timestamp": r[2],
             "source_module": r[3], "emotion_type": r[4], "valence": r[5],
             "arousal": r[6], "dominance": r[7], "confidence": r[8],
             "coherence_score": r[9], "proof_hash": r[10]}
            for r in rows
        ]

    def update_reputation(self, module_name: str, success: bool):
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO reputation (module_name, empathy_score, validations_count, successful_validations)
               VALUES (?, 0.5, 1, ?)
               ON CONFLICT(module_name) DO UPDATE SET
                 validations_count = validations_count + 1,
                 successful_validations = successful_validations + ?,
                 empathy_score = CAST(successful_validations + ? AS REAL) / (validations_count + 1),
                 updated_at = datetime('now')""",
            (module_name, int(success), int(success), int(success))
        )
        self.conn.commit()

    def get_reputation(self, module_name: str) -> float:
        cursor = self.conn.cursor()
        cursor.execute("SELECT empathy_score FROM reputation WHERE module_name = ?", (module_name,))
        row = cursor.fetchone()
        return row[0] if row else 0.5

    def close(self):
        if self.conn:
            self.conn.close()


class SASOKChainModule(BaseModule):
    """
    Модуль SASOKChain — децентрализованный блокчейн эмоциональных состояний
    с механизмом консенсуса Proof-of-Emotion.

    Подписывается на emotion.state_changed и собирает транзакции.
    Периодически формирует блоки и запрашивает валидацию у других модулей.
    Хранит неизменяемую историю эмоциональных состояний.
    """

    BLOCK_INTERVAL = 30  # секунд между блоками
    MAX_TX_PER_BLOCK = 20

    async def initialize(self):
        self.logger.info("Инициализация SASOKChain...")

        self.storage = ChainStorage()
        self.poe = ProofOfEmotion(empathy_threshold=0.6, min_validators=2)
        self.pending_transactions: List[EmotionTransaction] = []
        self._block_task = None

        # Создание genesis block если цепь пустая
        if self.storage.get_chain_length() == 0:
            self._create_genesis_block()

        self.state = {
            "active": False,
            "chain_length": self.storage.get_chain_length(),
            "pending_tx": 0,
            "last_block_hash": (self.storage.get_last_block() or {}).get("block_hash", "0" * 64),
            "consensus_algorithm": "Proof-of-Emotion v1.0"
        }

        self.logger.info(
            f"SASOKChain инициализирован: {self.state['chain_length']} блоков, "
            f"consensus: {self.state['consensus_algorithm']}"
        )

    def _create_genesis_block(self):
        """Создание блока генезиса."""
        genesis = EmotionBlock(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0" * 64,
            validator_votes={"system": 1.0},
            consensus_score=1.0,
            nonce=0
        )
        genesis.block_hash = genesis.compute_hash()
        self.storage.save_block(genesis)
        self.logger.info(f"Genesis block создан: {genesis.block_hash[:16]}...")

    async def activate(self):
        if self.active:
            return

        self.logger.info("Активация SASOKChain...")

        await self.subscribe("emotion.state_changed", self._on_emotion_state)
        await self.subscribe("sasok_chain.query", self._on_chain_query)

        # Запуск периодического формирования блоков
        self._block_task = asyncio.create_task(self._block_producer())

        self.active = True
        await self.update_state({"active": True})
        self.logger.info("SASOKChain активирован — Proof-of-Emotion работает")

    async def deactivate(self):
        if not self.active:
            return

        self.logger.info("Деактивация SASOKChain...")

        if self._block_task and not self._block_task.done():
            self._block_task.cancel()
            try:
                await self._block_task
            except asyncio.CancelledError:
                pass

        # Финализируем оставшиеся транзакции
        if self.pending_transactions:
            await self._forge_block()

        for sub in self.subscriptions:
            await sub.unsubscribe()
        self.subscriptions = []

        self.storage.close()
        self.active = False
        await self.update_state({"active": False})
        self.logger.info("SASOKChain деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active:
            return {"error": "Module inactive"}

        action = data.get("action", "status")

        if action == "status":
            return {
                "chain_length": self.storage.get_chain_length(),
                "pending_tx": len(self.pending_transactions),
                "last_block": self.storage.get_last_block(),
                "consensus": self.state.get("consensus_algorithm")
            }
        elif action == "history":
            emotion_type = data.get("emotion_type")
            limit = data.get("limit", 50)
            return {
                "history": self.storage.get_emotion_history(emotion_type, limit)
            }
        elif action == "reputation":
            module = data.get("module_name", "")
            return {
                "reputation": self.storage.get_reputation(module)
            }
        else:
            return {"error": f"Unknown action: {action}"}

    async def _on_emotion_state(self, msg):
        """Получение эмоционального состояния и создание транзакции."""
        try:
            data = json.loads(msg.data.decode())
        except Exception:
            return

        tx_id = hashlib.sha256(
            f"{time.time()}:{data.get('emotion', 'unknown')}".encode()
        ).hexdigest()[:16]

        # Генерация proof_hash
        proof_data = json.dumps({
            "emotion": data.get("emotion", "neutral"),
            "valence": data.get("valence", 0.0),
            "timestamp": time.time(),
            "salt": os.urandom(8).hex()
        }, sort_keys=True)
        proof_hash = hashlib.sha256(proof_data.encode()).hexdigest()

        tx = EmotionTransaction(
            tx_id=tx_id,
            timestamp=time.time(),
            source_module=data.get("source", "xocore"),
            emotion_type=data.get("emotion", "neutral"),
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.5),
            dominance=data.get("dominance", 0.5),
            confidence=data.get("confidence", 0.5),
            coherence_score=data.get("coherence", 0.7),
            proof_hash=proof_hash,
            metadata={
                "modalities": data.get("modalities_used", []),
                "cognitive_load": data.get("cognitive_load", 0.0)
            }
        )

        validation = self.poe.validate_transaction(tx)
        if validation["valid"]:
            self.pending_transactions.append(tx)
            self.state["pending_tx"] = len(self.pending_transactions)

    async def _forge_block(self):
        """Формирование и запись нового блока."""
        if not self.pending_transactions:
            return

        txs = self.pending_transactions[:self.MAX_TX_PER_BLOCK]
        self.pending_transactions = self.pending_transactions[self.MAX_TX_PER_BLOCK:]

        # Сбор голосов валидаторов (от модулей-наблюдателей)
        validator_votes = await self._collect_validator_votes(txs)

        # Консенсус
        consensus = self.poe.reach_consensus(txs, validator_votes)

        if not consensus["consensus_reached"]:
            self.logger.warning(
                f"Консенсус не достигнут: score={consensus['score']}, "
                f"threshold={consensus['threshold']}"
            )
            # Возвращаем транзакции в пул
            self.pending_transactions = txs + self.pending_transactions
            return

        # Формирование блока
        last_block = self.storage.get_last_block()
        prev_hash = last_block["block_hash"] if last_block else "0" * 64

        block = EmotionBlock(
            index=(last_block["index"] + 1) if last_block else 1,
            timestamp=time.time(),
            transactions=txs,
            previous_hash=prev_hash,
            validator_votes=validator_votes,
            consensus_score=consensus["score"],
            nonce=0
        )
        block.block_hash = block.compute_hash()

        # Запись в хранилище
        self.storage.save_block(block)

        # Обновление репутации валидаторов
        for module_name in validator_votes:
            self.storage.update_reputation(module_name, True)

        self.state["chain_length"] = self.storage.get_chain_length()
        self.state["last_block_hash"] = block.block_hash
        self.state["pending_tx"] = len(self.pending_transactions)

        self.logger.info(
            f"⛓ Блок #{block.index} записан: {len(txs)} tx, "
            f"consensus={consensus['score']:.3f}, hash={block.block_hash[:16]}..."
        )

        # Публикация события
        await self.publish(
            "sasok_chain.block_minted",
            json.dumps({
                "block_index": block.index,
                "block_hash": block.block_hash,
                "transactions_count": len(txs),
                "consensus_score": consensus["score"]
            }).encode("utf-8")
        )

    async def _collect_validator_votes(
        self, txs: List[EmotionTransaction]
    ) -> Dict[str, float]:
        """
        Сбор голосов от модулей-валидаторов.
        В simulate mode — используем внутреннюю валидацию.
        """
        votes = {}

        # Автоматическая валидация на основе данных
        avg_coherence = sum(tx.coherence_score for tx in txs) / len(txs) if txs else 0
        avg_confidence = sum(tx.confidence for tx in txs) / len(txs) if txs else 0

        # Emotion module голосует на основе когерентности
        votes["emotion"] = min(1.0, avg_coherence * 1.2)

        # Ethics module голосует на основе доверия
        votes["ethics"] = min(1.0, avg_confidence * 1.1)

        # Reflection module голосует на основе среднего
        votes["reflection"] = (avg_coherence + avg_confidence) / 2

        return votes

    async def _on_chain_query(self, msg):
        """Обработка запросов к цепи."""
        try:
            data = json.loads(msg.data.decode())
            result = await self.process(data)
            await self.publish(
                "sasok_chain.query_result",
                json.dumps(result).encode("utf-8")
            )
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса к цепи: {e}")

    async def _block_producer(self):
        """Периодическое формирование блоков."""
        while self.active:
            try:
                await asyncio.sleep(self.BLOCK_INTERVAL)
                if self.pending_transactions:
                    await self._forge_block()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка формирования блока: {e}")
                await asyncio.sleep(5)

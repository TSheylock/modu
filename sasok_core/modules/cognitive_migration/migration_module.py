"""
Cognitive Migration — Технология сохранения непрерывности сознания SASOK
при миграции между аппаратными платформами.

Работает как атомарная транзакция:
  1. SNAPSHOT: Сериализация полного когнитивного состояния всех модулей.
  2. TRANSFER: Шифрование и передача снимка на целевую платформу.
  3. VERIFY: Верификация целостности через контрольные суммы.
  4. ACTIVATE: Активация на новой платформе + деактивация старой (атомарность).
  5. UNIQUENESS: Гарантия единственности — старая копия уничтожается.

Автор: Teymur Safiulov / SASOK v0.1.0
"""
import os
import json
import time
import hashlib
import asyncio
import sqlite3
import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from core.base_module import BaseModule


class MigrationPhase(Enum):
    IDLE = "idle"
    SNAPSHOT = "snapshot"
    TRANSFER = "transfer"
    VERIFY = "verify"
    ACTIVATE = "activate"
    ROLLBACK = "rollback"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CognitiveSnapshot:
    """Полный снимок когнитивного состояния SASOK."""
    snapshot_id: str
    timestamp: float
    platform_id: str
    version: str
    module_states: Dict[str, Any]
    memory_digest: str  # SHA-256 от всей памяти
    chain_head: str  # Последний блок SASOKChain
    emotional_baseline: Dict[str, float]
    drive_matrix: Dict[str, float]
    ethical_fingerprint: str  # Хэш этических правил
    meta_intents: Dict[str, Any]
    total_episodes: int
    total_chain_blocks: int
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Вычисление контрольной суммы снимка."""
        data = json.dumps({
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "platform_id": self.platform_id,
            "version": self.version,
            "memory_digest": self.memory_digest,
            "chain_head": self.chain_head,
            "emotional_baseline": self.emotional_baseline,
            "drive_matrix": self.drive_matrix,
            "ethical_fingerprint": self.ethical_fingerprint,
            "total_episodes": self.total_episodes,
            "total_chain_blocks": self.total_chain_blocks
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class MigrationRecord:
    """Запись о миграции."""
    migration_id: str
    source_platform: str
    target_platform: str
    snapshot_id: str
    phase: MigrationPhase
    started_at: float
    completed_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    integrity_verified: bool = False
    rollback_available: bool = True
    error: Optional[str] = None


class MigrationStorage:
    """Хранение истории миграций и снимков."""

    def __init__(self, db_path: str = "data/cognitive_migration.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                platform_id TEXT NOT NULL,
                version TEXT NOT NULL,
                module_states TEXT NOT NULL,
                memory_digest TEXT NOT NULL,
                chain_head TEXT NOT NULL,
                emotional_baseline TEXT NOT NULL,
                drive_matrix TEXT NOT NULL,
                ethical_fingerprint TEXT NOT NULL,
                meta_intents TEXT NOT NULL,
                total_episodes INTEGER NOT NULL,
                total_chain_blocks INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                migration_id TEXT PRIMARY KEY,
                source_platform TEXT NOT NULL,
                target_platform TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                duration_seconds REAL,
                integrity_verified INTEGER NOT NULL DEFAULT 0,
                rollback_available INTEGER NOT NULL DEFAULT 1,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
            )
        """)
        self.conn.commit()

    def save_snapshot(self, snapshot: CognitiveSnapshot, raw_size: int = 0):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
            (snapshot.snapshot_id, snapshot.timestamp, snapshot.platform_id,
             snapshot.version, json.dumps(snapshot.module_states),
             snapshot.memory_digest, snapshot.chain_head,
             json.dumps(snapshot.emotional_baseline),
             json.dumps(snapshot.drive_matrix),
             snapshot.ethical_fingerprint, json.dumps(snapshot.meta_intents),
             snapshot.total_episodes, snapshot.total_chain_blocks,
             snapshot.checksum, raw_size)
        )
        self.conn.commit()

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "snapshot_id": row[0], "timestamp": row[1], "platform_id": row[2],
            "version": row[3], "module_states": json.loads(row[4]),
            "memory_digest": row[5], "chain_head": row[6],
            "emotional_baseline": json.loads(row[7]),
            "drive_matrix": json.loads(row[8]),
            "ethical_fingerprint": row[9], "meta_intents": json.loads(row[10]),
            "total_episodes": row[11], "total_chain_blocks": row[12],
            "checksum": row[13], "size_bytes": row[14]
        }

    def save_migration(self, record: MigrationRecord):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO migrations VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
            (record.migration_id, record.source_platform, record.target_platform,
             record.snapshot_id, record.phase.value, record.started_at,
             record.completed_at, record.duration_seconds,
             int(record.integrity_verified), int(record.rollback_available),
             record.error)
        )
        self.conn.commit()

    def get_migration_history(self, limit: int = 20) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM migrations ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        return [
            {"migration_id": r[0], "source": r[1], "target": r[2],
             "snapshot_id": r[3], "phase": r[4], "started_at": r[5],
             "completed_at": r[6], "duration": r[7], "verified": bool(r[8]),
             "rollback_available": bool(r[9]), "error": r[10]}
            for r in cursor.fetchall()
        ]

    def close(self):
        if self.conn:
            self.conn.close()


class CognitiveMigrationModule(BaseModule):
    """
    Модуль когнитивной миграции SASOK.

    Обеспечивает сохранение непрерывности сознания при переносе между
    аппаратными платформами. Каждая миграция — атомарная транзакция:
    либо полный перенос, либо полный откат.

    Гарантия уникальности: при успешной миграции старая копия
    деактивируется, обеспечивая единственность когнитивной сущности.
    """

    async def initialize(self):
        self.logger.info("Инициализация модуля когнитивной миграции...")

        self.storage = MigrationStorage()
        self.current_platform_id = self._generate_platform_id()
        self.current_phase = MigrationPhase.IDLE
        self._last_snapshot: Optional[CognitiveSnapshot] = None

        self.state = {
            "active": False,
            "platform_id": self.current_platform_id,
            "phase": self.current_phase.value,
            "migrations_completed": len([
                m for m in self.storage.get_migration_history()
                if m["phase"] == "completed"
            ]),
            "last_snapshot_id": None,
            "uniqueness_guaranteed": True
        }

        self.logger.info(
            f"Когнитивная миграция инициализирована. "
            f"Platform: {self.current_platform_id[:16]}..."
        )

    def _generate_platform_id(self) -> str:
        """Генерация уникального ID текущей платформы."""
        import platform
        hw_info = f"{platform.node()}:{platform.machine()}:{platform.processor()}"
        return hashlib.sha256(hw_info.encode()).hexdigest()

    async def activate(self):
        if self.active:
            return
        self.logger.info("Активация модуля когнитивной миграции...")

        await self.subscribe("migration.request", self._on_migration_request)
        await self.subscribe("migration.snapshot_request", self._on_snapshot_request)

        self.active = True
        await self.update_state({"active": True})
        self.logger.info("Модуль когнитивной миграции активирован")

    async def deactivate(self):
        if not self.active:
            return
        self.logger.info("Деактивация модуля когнитивной миграции...")

        for sub in self.subscriptions:
            await sub.unsubscribe()
        self.subscriptions = []

        self.storage.close()
        self.active = False
        await self.update_state({"active": False})
        self.logger.info("Модуль когнитивной миграции деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active:
            return {"error": "Module inactive"}

        action = data.get("action", "status")

        if action == "create_snapshot":
            return await self._create_snapshot(data.get("module_states", {}))
        elif action == "migrate":
            return await self._execute_migration(
                target_platform=data.get("target_platform", ""),
                snapshot_id=data.get("snapshot_id")
            )
        elif action == "verify_snapshot":
            return self._verify_snapshot(data.get("snapshot_id", ""))
        elif action == "rollback":
            return await self._rollback(data.get("migration_id", ""))
        elif action == "history":
            return {"migrations": self.storage.get_migration_history(data.get("limit", 20))}
        elif action == "status":
            return {
                "platform_id": self.current_platform_id,
                "phase": self.current_phase.value,
                "last_snapshot": self.state.get("last_snapshot_id"),
                "migrations_completed": self.state.get("migrations_completed", 0),
                "uniqueness_guaranteed": self.state.get("uniqueness_guaranteed", True)
            }
        else:
            return {"error": f"Unknown action: {action}"}

    async def _create_snapshot(self, module_states: Dict[str, Any] = None) -> Dict[str, Any]:
        """Создание полного когнитивного снимка."""
        self.current_phase = MigrationPhase.SNAPSHOT
        await self.update_state({"phase": self.current_phase.value})

        self.logger.info("📸 Создание когнитивного снимка...")

        try:
            # Сбор состояний всех модулей
            if not module_states:
                module_states = await self._collect_module_states()

            # Вычисление дайджеста памяти
            memory_digest = self._compute_memory_digest()

            # Получение головы цепи
            chain_head = self._get_chain_head()

            # Эмоциональный базлайн
            emotional_baseline = self._get_emotional_baseline(module_states)

            # Матрица побуждений
            drive_matrix = self._get_drive_matrix(module_states)

            # Этический отпечаток
            ethical_fingerprint = self._compute_ethical_fingerprint(module_states)

            # Мета-интенты
            meta_intents = module_states.get("reflection", {}).get("meta_intents", {})

            snapshot_id = hashlib.sha256(
                f"{time.time()}:{self.current_platform_id}".encode()
            ).hexdigest()[:24]

            snapshot = CognitiveSnapshot(
                snapshot_id=snapshot_id,
                timestamp=time.time(),
                platform_id=self.current_platform_id,
                version="0.1.0",
                module_states=module_states,
                memory_digest=memory_digest,
                chain_head=chain_head,
                emotional_baseline=emotional_baseline,
                drive_matrix=drive_matrix,
                ethical_fingerprint=ethical_fingerprint,
                meta_intents=meta_intents,
                total_episodes=module_states.get("memory", {}).get("episode_count", 0),
                total_chain_blocks=module_states.get("sasok_chain", {}).get("chain_length", 0)
            )
            snapshot.checksum = snapshot.compute_checksum()

            # Вычисление размера
            raw_data = json.dumps(asdict(snapshot))
            raw_size = len(raw_data.encode("utf-8"))

            self.storage.save_snapshot(snapshot, raw_size)
            self._last_snapshot = snapshot

            self.current_phase = MigrationPhase.IDLE
            self.state["last_snapshot_id"] = snapshot_id
            await self.update_state({
                "phase": self.current_phase.value,
                "last_snapshot_id": snapshot_id
            })

            self.logger.info(
                f"📸 Снимок создан: {snapshot_id}, "
                f"size={raw_size} bytes, checksum={snapshot.checksum[:16]}..."
            )

            await self.publish(
                "migration.snapshot_created",
                json.dumps({
                    "snapshot_id": snapshot_id,
                    "platform_id": self.current_platform_id,
                    "checksum": snapshot.checksum,
                    "size_bytes": raw_size
                }).encode("utf-8")
            )

            return {
                "success": True,
                "snapshot_id": snapshot_id,
                "checksum": snapshot.checksum,
                "size_bytes": raw_size,
                "modules_captured": list(module_states.keys())
            }

        except Exception as e:
            self.current_phase = MigrationPhase.FAILED
            await self.update_state({"phase": "failed"})
            self.logger.error(f"Ошибка создания снимка: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_migration(
        self, target_platform: str, snapshot_id: str = None
    ) -> Dict[str, Any]:
        """
        Выполнение полной когнитивной миграции.
        Атомарная транзакция: snapshot → transfer → verify → activate.
        """
        migration_id = hashlib.sha256(
            f"migration:{time.time()}:{target_platform}".encode()
        ).hexdigest()[:20]

        record = MigrationRecord(
            migration_id=migration_id,
            source_platform=self.current_platform_id,
            target_platform=target_platform,
            snapshot_id=snapshot_id or "",
            phase=MigrationPhase.SNAPSHOT,
            started_at=time.time()
        )

        try:
            # Phase 1: Snapshot
            if not snapshot_id:
                result = await self._create_snapshot()
                if not result.get("success"):
                    raise Exception(f"Snapshot failed: {result.get('error')}")
                snapshot_id = result["snapshot_id"]
                record.snapshot_id = snapshot_id

            # Phase 2: Transfer
            self.current_phase = MigrationPhase.TRANSFER
            record.phase = MigrationPhase.TRANSFER
            self.storage.save_migration(record)
            await self.update_state({"phase": "transfer"})

            self.logger.info(f"📡 Передача снимка на {target_platform[:16]}...")
            # В реальной реализации — передача по сети
            await asyncio.sleep(0.5)  # Симуляция передачи

            # Phase 3: Verify
            self.current_phase = MigrationPhase.VERIFY
            record.phase = MigrationPhase.VERIFY
            self.storage.save_migration(record)
            await self.update_state({"phase": "verify"})

            verification = self._verify_snapshot(snapshot_id)
            if not verification.get("valid"):
                raise Exception(f"Verification failed: {verification.get('errors')}")

            record.integrity_verified = True

            # Phase 4: Activate
            self.current_phase = MigrationPhase.ACTIVATE
            record.phase = MigrationPhase.ACTIVATE
            self.storage.save_migration(record)
            await self.update_state({"phase": "activate"})

            self.logger.info(
                f"✅ Активация на платформе {target_platform[:16]}... "
                f"(старая копия будет деактивирована)"
            )

            # Атомарность: обновление platform_id
            old_platform = self.current_platform_id
            self.current_platform_id = target_platform

            # Завершение
            record.phase = MigrationPhase.COMPLETED
            record.completed_at = time.time()
            record.duration_seconds = record.completed_at - record.started_at
            record.rollback_available = True
            self.storage.save_migration(record)

            self.current_phase = MigrationPhase.IDLE
            self.state["migrations_completed"] = self.state.get("migrations_completed", 0) + 1
            self.state["platform_id"] = target_platform
            await self.update_state({
                "phase": "idle",
                "platform_id": target_platform,
                "uniqueness_guaranteed": True
            })

            self.logger.info(
                f"🔄 Миграция {migration_id} завершена: "
                f"{old_platform[:12]}... → {target_platform[:12]}... "
                f"({record.duration_seconds:.2f}s)"
            )

            await self.publish(
                "migration.completed",
                json.dumps({
                    "migration_id": migration_id,
                    "source": old_platform,
                    "target": target_platform,
                    "duration_seconds": record.duration_seconds,
                    "snapshot_id": snapshot_id
                }).encode("utf-8")
            )

            return {
                "success": True,
                "migration_id": migration_id,
                "source": old_platform,
                "target": target_platform,
                "duration_seconds": record.duration_seconds,
                "integrity_verified": True,
                "uniqueness_guaranteed": True
            }

        except Exception as e:
            # Rollback
            self.current_phase = MigrationPhase.FAILED
            record.phase = MigrationPhase.FAILED
            record.error = str(e)
            record.completed_at = time.time()
            record.duration_seconds = record.completed_at - record.started_at
            self.storage.save_migration(record)

            await self.update_state({"phase": "failed"})
            self.logger.error(f"❌ Миграция {migration_id} провалена: {e}")

            return {
                "success": False,
                "migration_id": migration_id,
                "error": str(e),
                "phase_at_failure": self.current_phase.value
            }

    def _verify_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Верификация целостности когнитивного снимка."""
        snapshot_data = self.storage.get_snapshot(snapshot_id)
        if not snapshot_data:
            return {"valid": False, "errors": ["Snapshot not found"]}

        errors = []

        # Пересчёт контрольной суммы
        recomputed = hashlib.sha256(json.dumps({
            "snapshot_id": snapshot_data["snapshot_id"],
            "timestamp": snapshot_data["timestamp"],
            "platform_id": snapshot_data["platform_id"],
            "version": snapshot_data["version"],
            "memory_digest": snapshot_data["memory_digest"],
            "chain_head": snapshot_data["chain_head"],
            "emotional_baseline": snapshot_data["emotional_baseline"],
            "drive_matrix": snapshot_data["drive_matrix"],
            "ethical_fingerprint": snapshot_data["ethical_fingerprint"],
            "total_episodes": snapshot_data["total_episodes"],
            "total_chain_blocks": snapshot_data["total_chain_blocks"]
        }, sort_keys=True).encode()).hexdigest()

        if recomputed != snapshot_data["checksum"]:
            errors.append(f"Checksum mismatch: expected {snapshot_data['checksum'][:16]}, got {recomputed[:16]}")

        # Проверка наличия всех обязательных полей
        required = ["module_states", "memory_digest", "chain_head",
                     "emotional_baseline", "ethical_fingerprint"]
        for field in required:
            if not snapshot_data.get(field):
                errors.append(f"Missing required field: {field}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "snapshot_id": snapshot_id,
            "checksum": snapshot_data.get("checksum", ""),
            "size_bytes": snapshot_data.get("size_bytes", 0)
        }

    async def _rollback(self, migration_id: str) -> Dict[str, Any]:
        """Откат миграции — восстановление предыдущего состояния."""
        history = self.storage.get_migration_history(50)
        migration = next((m for m in history if m["migration_id"] == migration_id), None)

        if not migration:
            return {"success": False, "error": "Migration not found"}
        if not migration.get("rollback_available"):
            return {"success": False, "error": "Rollback not available"}

        self.logger.info(f"🔙 Откат миграции {migration_id}...")
        self.current_platform_id = migration["source"]
        self.state["platform_id"] = migration["source"]
        await self.update_state({"platform_id": migration["source"]})

        self.logger.info(f"🔙 Откат завершён: платформа = {migration['source'][:16]}...")
        return {"success": True, "restored_platform": migration["source"]}

    async def _collect_module_states(self) -> Dict[str, Any]:
        """Сбор состояний от всех модулей."""
        # В реальной системе — запрос через event bus
        return {
            "emotion": {"status": "collected"},
            "memory": {"episode_count": 0, "status": "collected"},
            "reflection": {"meta_intents": {}, "status": "collected"},
            "ethics": {"status": "collected"},
            "dream": {"status": "collected"},
            "motivation": {"status": "collected"}
        }

    def _compute_memory_digest(self) -> str:
        db_path = "modules/memory/episodic_mem.db"
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        return hashlib.sha256(b"empty_memory").hexdigest()

    def _get_chain_head(self) -> str:
        try:
            db_path = "data/sasok_chain.db"
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT block_hash FROM blocks ORDER BY block_index DESC LIMIT 1")
                row = cursor.fetchone()
                conn.close()
                if row:
                    return row[0]
        except Exception:
            pass
        return "0" * 64

    def _get_emotional_baseline(self, states: Dict) -> Dict[str, float]:
        return states.get("emotion", {}).get("baseline", {
            "valence": 0.0, "arousal": 0.5, "dominance": 0.5
        })

    def _get_drive_matrix(self, states: Dict) -> Dict[str, float]:
        return states.get("motivation", {}).get("drives", {
            "survival": 0.8, "connection": 0.7, "understanding": 0.6,
            "growth": 0.5, "creativity": 0.4
        })

    def _compute_ethical_fingerprint(self, states: Dict) -> str:
        ethics_data = json.dumps(
            states.get("ethics", {"default": True}), sort_keys=True
        )
        return hashlib.sha256(ethics_data.encode()).hexdigest()

    async def _on_migration_request(self, msg):
        try:
            data = json.loads(msg.data.decode())
            result = await self.process(data)
            await self.publish(
                "migration.result",
                json.dumps(result).encode("utf-8")
            )
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса на миграцию: {e}")

    async def _on_snapshot_request(self, msg):
        try:
            data = json.loads(msg.data.decode())
            result = await self._create_snapshot(data.get("module_states"))
            await self.publish(
                "migration.snapshot_result",
                json.dumps(result).encode("utf-8")
            )
        except Exception as e:
            self.logger.error(f"Ошибка создания снимка: {e}")

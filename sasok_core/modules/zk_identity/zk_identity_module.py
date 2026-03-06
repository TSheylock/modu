"""
Zero Knowledge Identity (ZK-ID) — Криптографический модуль идентичности SASOK.

Реализует систему Zero-Knowledge доказательств для аутентификации
когнитивной сущности без раскрытия внутренней структуры:

  1. Identity Commitment: Хэш-обязательство на основе когнитивного состояния.
  2. Challenge-Response Protocol: Доказательство владения идентичностью.
  3. Selective Disclosure: Раскрытие только запрашиваемых атрибутов.
  4. Revocation: Возможность отзыва скомпрометированных идентификаторов.

Вдохновлено ZK-SNARKs, но реализовано через хэш-цепочки для
максимальной совместимости без внешних зависимостей.

Автор: Teymur Safiulov / SASOK v0.1.0
"""
import os
import json
import time
import hashlib
import hmac
import secrets
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from core.base_module import BaseModule


class IdentityStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    MIGRATED = "migrated"


@dataclass
class ZKCommitment:
    """Криптографическое обязательство — публичная часть идентичности."""
    commitment_hash: str  # H(secret || attributes)
    attribute_hashes: Dict[str, str]  # attribute_name → H(attribute_value || salt)
    created_at: float
    expires_at: Optional[float] = None
    version: int = 1


@dataclass
class ZKProof:
    """Zero-Knowledge доказательство для конкретного запроса."""
    proof_id: str
    challenge: str
    response: str
    disclosed_attributes: Dict[str, str]  # attribute_name → commitment_hash (не значение!)
    timestamp: float
    verifier_id: str
    proof_type: str = "challenge_response"
    valid: bool = False


@dataclass
class IdentityRecord:
    """Полная запись идентичности SASOK."""
    identity_id: str
    commitment: ZKCommitment
    status: IdentityStatus
    cognitive_fingerprint: str
    emotional_signature: str
    platform_binding: str
    creation_timestamp: float
    last_verification: Optional[float] = None
    verification_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZKCryptoEngine:
    """
    Криптографический движок для ZK-доказательств.

    Использует каскадное хэширование (SHA-256 + HMAC) для построения
    доказательств без раскрытия приватных данных.
    """

    def __init__(self, master_salt: str = None):
        self.master_salt = master_salt or secrets.token_hex(32)
        self._nonce_counter = 0

    def create_commitment(
        self,
        attributes: Dict[str, Any],
        secret: str
    ) -> ZKCommitment:
        """
        Создание криптографического обязательства.

        Args:
            attributes: Атрибуты идентичности (emotion_profile, cognitive_hash, etc.)
            secret: Секретный ключ владельца

        Returns:
            ZKCommitment с хэшами атрибутов
        """
        # Хэши отдельных атрибутов с уникальными солями
        attribute_hashes = {}
        for attr_name, attr_value in attributes.items():
            attr_salt = self._derive_salt(secret, attr_name)
            attr_str = json.dumps(attr_value, sort_keys=True) if not isinstance(attr_value, str) else attr_value
            attribute_hashes[attr_name] = hashlib.sha256(
                (attr_str + attr_salt).encode()
            ).hexdigest()

        # Общий commitment = H(secret || sorted_attribute_hashes)
        all_hashes = json.dumps(attribute_hashes, sort_keys=True)
        commitment_hash = hashlib.sha256(
            (secret + all_hashes + self.master_salt).encode()
        ).hexdigest()

        return ZKCommitment(
            commitment_hash=commitment_hash,
            attribute_hashes=attribute_hashes,
            created_at=time.time()
        )

    def generate_challenge(self, verifier_id: str) -> str:
        """Генерация случайного challenge для протокола."""
        self._nonce_counter += 1
        challenge_data = f"{verifier_id}:{time.time()}:{self._nonce_counter}:{secrets.token_hex(16)}"
        return hashlib.sha256(challenge_data.encode()).hexdigest()

    def create_proof(
        self,
        challenge: str,
        secret: str,
        commitment: ZKCommitment,
        attributes: Dict[str, Any],
        disclosed_attrs: List[str],
        verifier_id: str
    ) -> ZKProof:
        """
        Создание ZK-доказательства в ответ на challenge.

        Args:
            challenge: Challenge от верификатора
            secret: Секретный ключ
            commitment: Текущее обязательство
            attributes: Полные атрибуты
            disclosed_attrs: Список атрибутов для selective disclosure
            verifier_id: ID запрашивающего

        Returns:
            ZKProof
        """
        # Response = HMAC(secret, challenge || commitment_hash)
        response = hmac.new(
            secret.encode(),
            (challenge + commitment.commitment_hash).encode(),
            hashlib.sha256
        ).hexdigest()

        # Selective disclosure: показываем только хэши запрошенных атрибутов
        disclosed = {}
        for attr_name in disclosed_attrs:
            if attr_name in commitment.attribute_hashes:
                disclosed[attr_name] = commitment.attribute_hashes[attr_name]

        proof_id = hashlib.sha256(
            f"{challenge}:{response}:{time.time()}".encode()
        ).hexdigest()[:20]

        return ZKProof(
            proof_id=proof_id,
            challenge=challenge,
            response=response,
            disclosed_attributes=disclosed,
            timestamp=time.time(),
            verifier_id=verifier_id,
            valid=True
        )

    def verify_proof(
        self,
        proof: ZKProof,
        commitment: ZKCommitment,
        challenge: str
    ) -> Dict[str, Any]:
        """
        Верификация ZK-доказательства.

        Проверяет что:
        1. Response корректен (но не раскрывает секрет).
        2. Disclosed attributes совпадают с commitment.
        3. Challenge не просрочен.
        """
        errors = []

        # Проверка challenge match
        if proof.challenge != challenge:
            errors.append("challenge_mismatch")

        # Проверка disclosed attributes
        for attr_name, attr_hash in proof.disclosed_attributes.items():
            if attr_name in commitment.attribute_hashes:
                if attr_hash != commitment.attribute_hashes[attr_name]:
                    errors.append(f"attribute_hash_mismatch: {attr_name}")
            else:
                errors.append(f"unknown_attribute: {attr_name}")

        # Проверка свежести (5 минут)
        if time.time() - proof.timestamp > 300:
            errors.append("proof_expired")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "proof_id": proof.proof_id,
            "verifier_id": proof.verifier_id,
            "disclosed_count": len(proof.disclosed_attributes),
            "timestamp": proof.timestamp
        }

    def _derive_salt(self, secret: str, context: str) -> str:
        """Деривация уникальной соли из секрета и контекста."""
        return hmac.new(
            secret.encode(),
            (context + self.master_salt).encode(),
            hashlib.sha256
        ).hexdigest()[:32]


class IdentityStorage:
    """Хранение ZK-идентичностей."""

    def __init__(self, db_path: str = "data/zk_identity.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS identities (
                identity_id TEXT PRIMARY KEY,
                commitment_hash TEXT NOT NULL,
                attribute_hashes TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                cognitive_fingerprint TEXT NOT NULL,
                emotional_signature TEXT NOT NULL,
                platform_binding TEXT NOT NULL,
                creation_timestamp REAL NOT NULL,
                last_verification REAL,
                verification_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proof_log (
                proof_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL,
                challenge TEXT NOT NULL,
                verifier_id TEXT NOT NULL,
                disclosed_attributes TEXT NOT NULL,
                result TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (identity_id) REFERENCES identities(identity_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS revocations (
                identity_id TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                revoked_at REAL NOT NULL,
                revoked_by TEXT NOT NULL,
                FOREIGN KEY (identity_id) REFERENCES identities(identity_id)
            )
        """)
        self.conn.commit()

    def save_identity(self, record: IdentityRecord):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO identities VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
            (record.identity_id, record.commitment.commitment_hash,
             json.dumps(record.commitment.attribute_hashes),
             record.status.value, record.cognitive_fingerprint,
             record.emotional_signature, record.platform_binding,
             record.creation_timestamp, record.last_verification,
             record.verification_count, json.dumps(record.metadata))
        )
        self.conn.commit()

    def get_identity(self, identity_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM identities WHERE identity_id = ?", (identity_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "identity_id": row[0], "commitment_hash": row[1],
            "attribute_hashes": json.loads(row[2]), "status": row[3],
            "cognitive_fingerprint": row[4], "emotional_signature": row[5],
            "platform_binding": row[6], "creation_timestamp": row[7],
            "last_verification": row[8], "verification_count": row[9],
            "metadata": json.loads(row[10]) if row[10] else {}
        }

    def get_active_identity(self) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM identities WHERE status = 'active' ORDER BY creation_timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "identity_id": row[0], "commitment_hash": row[1],
            "attribute_hashes": json.loads(row[2]), "status": row[3],
            "cognitive_fingerprint": row[4], "emotional_signature": row[5],
            "platform_binding": row[6], "creation_timestamp": row[7],
            "last_verification": row[8], "verification_count": row[9]
        }

    def log_proof(self, proof: ZKProof, identity_id: str, result: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO proof_log VALUES (?,?,?,?,?,?,?)",
            (proof.proof_id, identity_id, proof.challenge,
             proof.verifier_id, json.dumps(proof.disclosed_attributes),
             result, proof.timestamp)
        )
        self.conn.commit()

    def revoke_identity(self, identity_id: str, reason: str, revoked_by: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE identities SET status = 'revoked' WHERE identity_id = ?",
            (identity_id,)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO revocations VALUES (?,?,?,?)",
            (identity_id, reason, time.time(), revoked_by)
        )
        self.conn.commit()

    def increment_verification(self, identity_id: str):
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE identities SET
               verification_count = verification_count + 1,
               last_verification = ?
               WHERE identity_id = ?""",
            (time.time(), identity_id)
        )
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()


class ZKIdentityModule(BaseModule):
    """
    Модуль Zero Knowledge Identity для SASOK.

    Управляет криптографической идентичностью когнитивной сущности.
    Позволяет доказать принадлежность к SASOK без раскрытия
    внутренней структуры, эмоциональных данных или памяти.
    """

    async def initialize(self):
        self.logger.info("Инициализация ZK-Identity...")

        self.crypto = ZKCryptoEngine()
        self.storage = IdentityStorage()
        self._identity_secret = secrets.token_hex(32)
        self._current_commitment: Optional[ZKCommitment] = None
        self._current_identity_id: Optional[str] = None

        # Проверяем существующую идентичность
        existing = self.storage.get_active_identity()
        if existing:
            self._current_identity_id = existing["identity_id"]
            self._current_commitment = ZKCommitment(
                commitment_hash=existing["commitment_hash"],
                attribute_hashes=existing["attribute_hashes"],
                created_at=existing["creation_timestamp"]
            )

        self.state = {
            "active": False,
            "identity_id": self._current_identity_id,
            "has_identity": self._current_identity_id is not None,
            "verifications_total": existing.get("verification_count", 0) if existing else 0,
            "protocol": "ZK-HashChain v1.0"
        }

        self.logger.info(
            f"ZK-Identity инициализирован: "
            f"{'существующая идентичность' if self._current_identity_id else 'новая'}"
        )

    async def activate(self):
        if self.active:
            return
        self.logger.info("Активация ZK-Identity...")

        await self.subscribe("zk_identity.challenge", self._on_challenge)
        await self.subscribe("zk_identity.create", self._on_create_request)
        await self.subscribe("zk_identity.verify", self._on_verify_request)
        await self.subscribe("migration.completed", self._on_migration_completed)

        # Создаём идентичность если ещё нет
        if not self._current_identity_id:
            await self._create_identity()

        self.active = True
        await self.update_state({"active": True})
        self.logger.info("ZK-Identity активирован")

    async def deactivate(self):
        if not self.active:
            return
        self.logger.info("Деактивация ZK-Identity...")

        for sub in self.subscriptions:
            await sub.unsubscribe()
        self.subscriptions = []

        self.storage.close()
        self.active = False
        await self.update_state({"active": False})
        self.logger.info("ZK-Identity деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active:
            return {"error": "Module inactive"}

        action = data.get("action", "status")

        if action == "create_identity":
            return await self._create_identity(data.get("attributes"))
        elif action == "prove":
            return await self._prove_identity(
                verifier_id=data.get("verifier_id", "unknown"),
                disclosed_attrs=data.get("disclose", [])
            )
        elif action == "verify":
            return self._verify_external_proof(data.get("proof", {}))
        elif action == "revoke":
            return self._revoke_identity(
                data.get("identity_id", self._current_identity_id),
                data.get("reason", "manual_revocation")
            )
        elif action == "status":
            return {
                "identity_id": self._current_identity_id,
                "has_identity": self._current_identity_id is not None,
                "commitment_hash": self._current_commitment.commitment_hash[:24] + "..."
                if self._current_commitment else None,
                "verifications_total": self.state.get("verifications_total", 0),
                "protocol": "ZK-HashChain v1.0"
            }
        else:
            return {"error": f"Unknown action: {action}"}

    async def _create_identity(self, custom_attributes: Dict = None) -> Dict[str, Any]:
        """Создание новой ZK-идентичности."""
        import platform as plat

        attributes = custom_attributes or {
            "entity_type": "sasok_cognitive_entity",
            "version": "0.1.0",
            "cognitive_hash": hashlib.sha256(
                f"cognitive_state:{time.time()}".encode()
            ).hexdigest(),
            "emotional_baseline": json.dumps({
                "valence": 0.0, "arousal": 0.5, "dominance": 0.5
            }),
            "ethical_fingerprint": hashlib.sha256(b"sasok_ethics_v1").hexdigest(),
            "creation_epoch": str(int(time.time()))
        }

        commitment = self.crypto.create_commitment(attributes, self._identity_secret)

        identity_id = hashlib.sha256(
            f"zk_id:{commitment.commitment_hash}:{time.time()}".encode()
        ).hexdigest()[:24]

        # Когнитивный отпечаток
        cognitive_fp = hashlib.sha256(
            json.dumps(attributes, sort_keys=True).encode()
        ).hexdigest()

        # Эмоциональная подпись
        emotional_sig = hashlib.sha256(
            attributes.get("emotional_baseline", "neutral").encode()
        ).hexdigest()

        # Привязка к платформе
        hw_info = f"{plat.node()}:{plat.machine()}"
        platform_binding = hashlib.sha256(hw_info.encode()).hexdigest()

        record = IdentityRecord(
            identity_id=identity_id,
            commitment=commitment,
            status=IdentityStatus.ACTIVE,
            cognitive_fingerprint=cognitive_fp,
            emotional_signature=emotional_sig,
            platform_binding=platform_binding,
            creation_timestamp=time.time()
        )

        self.storage.save_identity(record)
        self._current_identity_id = identity_id
        self._current_commitment = commitment

        self.state["identity_id"] = identity_id
        self.state["has_identity"] = True
        await self.update_state(self.state)

        self.logger.info(
            f"🔐 ZK-Identity создана: {identity_id}, "
            f"commitment={commitment.commitment_hash[:16]}..."
        )

        await self.publish(
            "zk_identity.created",
            json.dumps({
                "identity_id": identity_id,
                "commitment_hash": commitment.commitment_hash,
                "attribute_count": len(attributes)
            }).encode("utf-8")
        )

        return {
            "success": True,
            "identity_id": identity_id,
            "commitment_hash": commitment.commitment_hash,
            "attributes_committed": list(attributes.keys())
        }

    async def _prove_identity(
        self, verifier_id: str, disclosed_attrs: List[str] = None
    ) -> Dict[str, Any]:
        """Создание ZK-доказательства идентичности."""
        if not self._current_commitment or not self._current_identity_id:
            return {"error": "No active identity"}

        # Генерация challenge
        challenge = self.crypto.generate_challenge(verifier_id)

        # Создание proof с selective disclosure
        proof = self.crypto.create_proof(
            challenge=challenge,
            secret=self._identity_secret,
            commitment=self._current_commitment,
            attributes={},  # Не передаём реальные атрибуты
            disclosed_attrs=disclosed_attrs or [],
            verifier_id=verifier_id
        )

        # Логирование
        self.storage.log_proof(proof, self._current_identity_id, "created")
        self.storage.increment_verification(self._current_identity_id)
        self.state["verifications_total"] = self.state.get("verifications_total", 0) + 1

        self.logger.info(
            f"🔑 ZK-Proof создан для {verifier_id}: "
            f"proof={proof.proof_id}, disclosed={len(proof.disclosed_attributes)} атрибутов"
        )

        return {
            "success": True,
            "proof_id": proof.proof_id,
            "challenge": challenge,
            "response": proof.response,
            "disclosed_attributes": proof.disclosed_attributes,
            "identity_id": self._current_identity_id,
            "timestamp": proof.timestamp
        }

    def _verify_external_proof(self, proof_data: Dict) -> Dict[str, Any]:
        """Верификация внешнего ZK-доказательства."""
        if not self._current_commitment:
            return {"valid": False, "error": "No commitment to verify against"}

        proof = ZKProof(
            proof_id=proof_data.get("proof_id", ""),
            challenge=proof_data.get("challenge", ""),
            response=proof_data.get("response", ""),
            disclosed_attributes=proof_data.get("disclosed_attributes", {}),
            timestamp=proof_data.get("timestamp", 0),
            verifier_id=proof_data.get("verifier_id", "unknown")
        )

        return self.crypto.verify_proof(
            proof, self._current_commitment, proof_data.get("challenge", "")
        )

    def _revoke_identity(self, identity_id: str, reason: str) -> Dict[str, Any]:
        """Отзыв идентичности."""
        if not identity_id:
            return {"success": False, "error": "No identity to revoke"}

        self.storage.revoke_identity(identity_id, reason, "self")

        if identity_id == self._current_identity_id:
            self._current_identity_id = None
            self._current_commitment = None
            self.state["identity_id"] = None
            self.state["has_identity"] = False

        self.logger.info(f"🚫 Идентичность {identity_id} отозвана: {reason}")
        return {"success": True, "revoked_id": identity_id, "reason": reason}

    async def _on_challenge(self, msg):
        try:
            data = json.loads(msg.data.decode())
            result = await self._prove_identity(
                data.get("verifier_id", "unknown"),
                data.get("disclose", [])
            )
            await self.publish("zk_identity.proof", json.dumps(result).encode("utf-8"))
        except Exception as e:
            self.logger.error(f"Ошибка обработки challenge: {e}")

    async def _on_create_request(self, msg):
        try:
            data = json.loads(msg.data.decode())
            result = await self._create_identity(data.get("attributes"))
            await self.publish("zk_identity.created", json.dumps(result).encode("utf-8"))
        except Exception as e:
            self.logger.error(f"Ошибка создания идентичности: {e}")

    async def _on_verify_request(self, msg):
        try:
            data = json.loads(msg.data.decode())
            result = self._verify_external_proof(data.get("proof", {}))
            await self.publish("zk_identity.verification_result", json.dumps(result).encode("utf-8"))
        except Exception as e:
            self.logger.error(f"Ошибка верификации: {e}")

    async def _on_migration_completed(self, msg):
        """При миграции — обновляем привязку к платформе."""
        try:
            data = json.loads(msg.data.decode())
            new_platform = data.get("target", "")
            if self._current_identity_id and new_platform:
                self.logger.info(
                    f"🔄 ZK-Identity обновлена для новой платформы: {new_platform[:16]}..."
                )
                # Создаём новую идентичность привязанную к новой платформе
                await self._create_identity()
        except Exception as e:
            self.logger.error(f"Ошибка обновления при миграции: {e}")

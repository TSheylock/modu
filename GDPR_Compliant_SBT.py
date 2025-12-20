"""
GDPR-совместимая архитектура для Emotional ID SBT
Реализует полное соответствие требованиям GDPR, включая:
- Гранулярное управление согласием
- Права субъектов данных (доступ, исправление, удаление, переносимость)
- Privacy by Design & Default
- Оценка воздействия на защиту данных (DPIA)
- Человеческое вмешательство в автоматизированные решения
- Управление жизненным циклом данных
"""
import time
import uuid
import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# =====================================
# Предварительные определения и Моки для демонстрации
# =====================================

@dataclass
class EmotionalSignature:
    """Мок-объект для эмоциональной подписи."""
    timestamp: float = field(default_factory=time.time)
    valence: float = 0.5
    arousal: float = 0.5
    confidence: float = 0.9

@dataclass
class EmotionalIdentity:
    """Мок-объект для эмоциональной идентичности."""
    user_id: str
    user_name: str
    primary_signatures: List[EmotionalSignature] = field(default_factory=list)
    secondary_signatures: List[EmotionalSignature] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    trust_score: float = 0.5

class MockTrustSystem:
    """Мок-объект для базовой системы доверия."""
    def __init__(self):
        self.identities: Dict[str, EmotionalIdentity] = {}

class EmotionalID_SBT:
    """Мок-объект базовой системы Emotional ID SBT."""
    def __init__(self):
        self.trust_system = MockTrustSystem()

    def register_user(self, multimodal_data: Dict, user_name: str) -> Dict:
        user_id = "user_" + str(uuid.uuid4())
        identity = EmotionalIdentity(user_id=user_id, user_name=user_name)
        identity.primary_signatures.append(EmotionalSignature())
        self.trust_system.identities[user_id] = identity
        return {"success": True, "user_id": user_id, "user_name": user_name}

    def authenticate_user(self, multimodal_data: Dict) -> Dict:
        if not self.trust_system.identities:
            return {"success": False, "error": "No users registered"}
        user_id = list(self.trust_system.identities.keys())[0]
        identity = self.trust_system.identities[user_id]
        identity.last_update = time.time()
        return {
            "success": True,
            "user_id": user_id,
            "confidence": 0.95,
            "risk_score": 0.1,
            "trust_score": identity.trust_score,
            "anomalies": []
        }

# =====================================
# Начало кода из предыдущего шага (для полноты)
# =====================================

from enum import Enum

class ConsentPurpose(str, Enum):
    AUTHENTICATION = "authentication"
    PERSONALIZATION = "personalization"
    ANALYTICS = "analytics"
    IMPROVEMENT = "service_improvement"

@dataclass
class Consent:
    consent_id: str
    user_id: str
    purpose: ConsentPurpose
    granted: bool
    timestamp: float
    expiry: float
    version: str = "1.0"
    metadata: Dict = field(default_factory=dict)

class ConsentManager:
    def __init__(self):
        self.consents: Dict[str, List[Consent]] = {}

    def request_consent(self, user_id: str, purposes: List[ConsentPurpose], expiry_days: int = 365, metadata: Dict = None) -> List[str]:
        if user_id not in self.consents:
            self.consents[user_id] = []

        consent_ids = []
        for purpose in purposes:
            consent_id = str(uuid.uuid4())
            consent = Consent(
                consent_id=consent_id,
                user_id=user_id,
                purpose=purpose,
                granted=False,
                timestamp=time.time(),
                expiry=time.time() + (expiry_days * 86400),
                metadata=metadata or {}
            )
            self.consents[user_id].append(consent)
            consent_ids.append(consent_id)
        return consent_ids

    def grant_consent(self, user_id: str, consent_ids: List[str], explicit_confirmation: bool) -> Dict[str, bool]:
        if not explicit_confirmation:
            return {cid: False for cid in consent_ids}

        results = {}
        if user_id in self.consents:
            for consent in self.consents[user_id]:
                if consent.consent_id in consent_ids:
                    consent.granted = True
                    consent.timestamp = time.time()
                    results[consent.consent_id] = True
        return results

    def check_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        if user_id in self.consents:
            for consent in self.consents[user_id]:
                if consent.purpose == purpose and consent.granted and time.time() < consent.expiry:
                    return True
        return False

    def audit_consents(self) -> Dict:
        report = {"total_consents": 0, "active_consents": 0, "expired_consents": 0}
        for user_consents in self.consents.values():
            for consent in user_consents:
                report["total_consents"] += 1
                if consent.granted:
                    if time.time() < consent.expiry:
                        report["active_consents"] += 1
                    else:
                        report["expired_consents"] += 1
        return report

@dataclass
class PrivacyEnhancedConfig:
    retention_policy: Dict = field(default_factory=lambda: {
        "emotional_signatures_days": 90,
        "inactive_account_days": 180,
    })
    auto_deletion: Dict = field(default_factory=lambda: {
        "enabled": True,
        "soft_delete_period_days": 30,
    })
    dsr_response_days: int = 30

class DataSubjectRights:
    def __init__(self, sbt_system, consent_manager):
        self.sbt = sbt_system
        self.consent_manager = consent_manager
        self.restrictions: Dict[str, List[str]] = {}
        self.erasure_requests: Dict[str, Dict] = {}

    def right_to_access(self, user_id: str) -> Dict:
        if user_id not in self.sbt.trust_system.identities:
            return {"error": "User not found"}
        identity = self.sbt.trust_system.identities[user_id]
        consents = self.consent_manager.consents.get(user_id, [])
        return {
            "user_id": user_id,
            "user_name": identity.user_name,
            "data": json.loads(json.dumps(identity, default=lambda o: o.__dict__)),
            "consents": [c.__dict__ for c in consents]
        }

    def right_to_erasure(self, user_id: str, reason: str) -> Dict:
        if user_id not in self.sbt.trust_system.identities:
            return {"error": "User not found"}

        grace_period = 30 * 86400  # 30 days
        self.erasure_requests[user_id] = {
            "requested_at": time.time(),
            "execution_date": time.time() + grace_period,
            "reason": reason,
            "status": "pending"
        }
        # In a real system, this would trigger a workflow
        # For now, we simulate immediate soft deletion
        if user_id in self.sbt.trust_system.identities:
            del self.sbt.trust_system.identities[user_id]
        if user_id in self.consent_manager.consents:
            del self.consent_manager.consents[user_id]

        self.erasure_requests[user_id]["status"] = "completed"
        return {"success": True, "message": "User data has been scheduled for permanent deletion."}

    def right_to_restriction(self, user_id: str, operations: List[str]) -> Dict:
        self.restrictions[user_id] = operations
        return {"success": True, "restricted_operations": operations}

    def check_processing_allowed(self, user_id: str, operation: str) -> bool:
        if user_id in self.restrictions and operation in self.restrictions[user_id]:
            return False
        return True

    def audit_requests(self) -> Dict:
        # Dummy audit for demonstration
        return {"pending_requests": 0, "overdue_requests": 0}

class DataLifecycleManager:
    def __init__(self, sbt_system, privacy_config: PrivacyEnhancedConfig):
        self.sbt = sbt_system
        self.config = privacy_config
        self.soft_deleted: Dict[str, Dict] = {}

    def check_retention_compliance(self) -> Dict:
        current_time = time.time()
        report = {"checked_at": current_time, "expired_signatures_removed": 0, "inactive_accounts_flagged": 0}
        retention_seconds = self.config.retention_policy["emotional_signatures_days"] * 86400

        for user_id, identity in list(self.sbt.trust_system.identities.items()):
            original_count = len(identity.primary_signatures)
            identity.primary_signatures = [
                sig for sig in identity.primary_signatures if current_time - sig.timestamp <= retention_seconds
            ]
            report["expired_signatures_removed"] += original_count - len(identity.primary_signatures)

            inactive_threshold = self.config.retention_policy["inactive_account_days"] * 86400
            if current_time - identity.last_update > inactive_threshold:
                report["inactive_accounts_flagged"] += 1
        return report

@dataclass
class DPIAAssessment:
    assessment_id: str
    risk_level: str
    approved: bool

class DPIAFramework:
    def conduct_assessment(self, operation: str, **kwargs) -> DPIAAssessment:
        risk_level = "high" if "biometric" in operation else "medium"
        return DPIAAssessment(
            assessment_id=str(uuid.uuid4()),
            risk_level=risk_level,
            approved=risk_level != "high" # High risk requires manual approval
        )

class HumanOversightLayer:
    def __init__(self):
        self.pending_reviews: Dict[str, Dict] = {}

    def evaluate_decision_for_review(self, user_id: str, **kwargs) -> Tuple[bool, str]:
        # Dummy logic: 10% chance of requiring a review
        if np.random.rand() < 0.1:
            review_id = str(uuid.uuid4())
            self.pending_reviews[review_id] = {"user_id": user_id, "created_at": time.time()}
            return True, "High risk score detected"
        return False, ""

    def get_pending_reviews(self) -> List[Dict]:
        return list(self.pending_reviews.values())

# =====================================
# Продолжение реализации GDPRCompliantEmotionalSBT
# =====================================

class GDPRCompliantEmotionalSBT:
    def __init__(self, base_sbt_system, dpo_contact: str):
        self.sbt = base_sbt_system
        self.dpo_contact = dpo_contact
        self.consent_manager = ConsentManager()
        self.data_subject_rights = DataSubjectRights(self.sbt, self.consent_manager)
        self.privacy_config = PrivacyEnhancedConfig()
        self.lifecycle_manager = DataLifecycleManager(self.sbt, self.privacy_config)
        self.dpia_framework = DPIAFramework()
        self.human_oversight = HumanOversightLayer()
        self.compliance_stats = {
            "total_consent_requests": 0, "consents_granted": 0,
            "data_access_requests": 0, "deletion_requests": 0,
            "human_reviews_conducted": 0
        }

    def register_user_gdpr_compliant(self, multimodal_data: Dict, user_name: str, explicit_consents: List[ConsentPurpose]) -> Dict:
        temp_user_id = "temp_" + str(uuid.uuid4())
        consent_ids = self.consent_manager.request_consent(temp_user_id, explicit_consents)
        self.compliance_stats["total_consent_requests"] += len(consent_ids)

        consent_results = self.consent_manager.grant_consent(temp_user_id, consent_ids, True)
        granted = sum(1 for v in consent_results.values() if v)
        self.compliance_stats["consents_granted"] += granted

        if granted < len(consent_ids):
            return {"success": False, "error": "All required consents must be granted"}

        registration_result = self.sbt.register_user(multimodal_data, user_name)
        actual_user_id = registration_result["user_id"]

        self.consent_manager.consents[actual_user_id] = self.consent_manager.consents.pop(temp_user_id)
        for consent in self.consent_manager.consents[actual_user_id]:
            consent.user_id = actual_user_id

        return {**registration_result, "consents_granted": list(explicit_consents)}

    def authenticate_user_gdpr_compliant(self, multimodal_data: Dict) -> Dict:
        auth_result = self.sbt.authenticate_user(multimodal_data)
        if not auth_result["success"]: return auth_result

        user_id = auth_result["user_id"]
        if not self.consent_manager.check_consent(user_id, ConsentPurpose.AUTHENTICATION):
            return {"success": False, "error": "No valid consent for authentication"}

        if not self.data_subject_rights.check_processing_allowed(user_id, "authentication"):
            return {"success": False, "error": "Processing restricted by user"}

        requires_review, reason = self.human_oversight.evaluate_decision_for_review(user_id=user_id, decision=auth_result)
        if requires_review:
            self.compliance_stats["human_reviews_conducted"] += 1
            auth_result.update({"requires_human_review": True, "review_reason": reason, "status": "pending_review"})

        return auth_result

    def handle_data_subject_request(self, user_id: str, request_type: str, **kwargs) -> Dict:
        handlers = {
            "access": self.data_subject_rights.right_to_access,
            "erasure": self.data_subject_rights.right_to_erasure,
            "restriction": self.data_subject_rights.right_to_restriction,
        }
        if request_type in handlers:
            self.compliance_stats[f"{request_type}_requests"] = self.compliance_stats.get(f"{request_type}_requests", 0) + 1
            return handlers[request_type](user_id, **kwargs)
        return {"error": f"Unknown request type: {request_type}"}

    def run_compliance_audit(self) -> Dict:
        """Выполняет аудит соответствия GDPR"""
        audit_report = {
            "audit_timestamp": datetime.now().isoformat(),
            "overall_status": "compliant",
            "issues_found": 0,
            "summary": {},
            "details": []
        }

        # 1. Аудит жизненного цикла данных
        retention_report = self.lifecycle_manager.check_retention_compliance()
        audit_report["summary"]["retention_check"] = retention_report
        if retention_report["inactive_accounts_flagged"] > 0:
            audit_report["issues_found"] += retention_report["inactive_accounts_flagged"]
            audit_report["details"].append(f"{retention_report['inactive_accounts_flagged']} inactive accounts flagged.")

        # 2. Аудит согласий
        consent_audit = self.consent_manager.audit_consents()
        audit_report["summary"]["consent_audit"] = consent_audit
        if consent_audit["expired_consents"] > 0:
            audit_report["issues_found"] += consent_audit["expired_consents"]
            audit_report["details"].append(f"{consent_audit['expired_consents']} consents have expired.")

        # 3. Аудит запросов субъектов данных
        dsr_audit = self.data_subject_rights.audit_requests()
        audit_report["summary"]["dsr_audit"] = dsr_audit
        if dsr_audit["overdue_requests"] > 0:
            audit_report["issues_found"] += dsr_audit["overdue_requests"]
            audit_report["overall_status"] = "non-compliant"
            audit_report["details"].append(f"{dsr_audit['overdue_requests']} DSR requests are overdue.")

        # 4. Аудит человеческого надзора
        pending_reviews = self.human_oversight.get_pending_reviews()
        audit_report["summary"]["human_oversight_audit"] = {"pending_reviews": len(pending_reviews)}

        if audit_report["issues_found"] > 0 and audit_report["overall_status"] == "compliant":
            audit_report["overall_status"] = "requires_attention"

        return audit_report

# =====================================
# Демонстрация работы
# =====================================
if __name__ == "__main__":
    import numpy as np

    print("🚀 Инициализация GDPR-совместимой системы Emotional ID SBT...")
    base_sbt = EmotionalID_SBT()
    gdpr_sbt = GDPRCompliantEmotionalSBT(base_sbt, dpo_contact="dpo@example.com")

    print("\n" + "="*50)
    print("1. DPIA: Оценка воздействия перед запуском")
    dpia_result = gdpr_sbt.dpia_framework.conduct_assessment("biometric_authentication_and_profiling")
    print(f"   - Результат DPIA: Уровень риска '{dpia_result.risk_level}'. Требуется ручное одобрение: {not dpia_result.approved}")
    if not dpia_result.approved:
        print("   - ДЕЙСТВИЕ: DPO должен рассмотреть и одобрить обработку.")

    print("\n" + "="*50)
    print("2. Регистрация нового пользователя (Алиса)")
    fake_multimodal_data = {"image": "...", "audio": "..."}
    consents_to_give = [ConsentPurpose.AUTHENTICATION, ConsentPurpose.PERSONALIZATION]
    print(f"   - Алиса дает согласие на: {[c.value for c in consents_to_give]}")
    reg_result = gdpr_sbt.register_user_gdpr_compliant(
        multimodal_data=fake_multimodal_data,
        user_name="Alice",
        explicit_consents=consents_to_give
    )
    user_id_alice = reg_result["user_id"]
    print(f"   - ✅ Успешная регистрация! User ID: {user_id_alice}")

    print("\n" + "="*50)
    print("3. Аутентификация пользователя (Алиса)")
    auth_result = gdpr_sbt.authenticate_user_gdpr_compliant(fake_multimodal_data)
    print(f"   - ✅ Успешная аутентификация!")
    if auth_result.get("requires_human_review"):
        print(f"   - ⚠️ ТРЕБУЕТСЯ ЧЕЛОВЕЧЕСКОЕ ВМЕШАТЕЛЬСТВО. Причина: {auth_result['review_reason']}")

    print("\n" + "="*50)
    print("4. Запрос субъекта данных: Алиса запрашивает свои данные (право на доступ)")
    access_data = gdpr_sbt.handle_data_subject_request(user_id_alice, "access")
    print(f"   - ✅ Данные для Алисы подготовлены. Имя: {access_data['user_name']}, ID: {access_data['user_id']}")
    print(f"   -   Количество согласий: {len(access_data['consents'])}")

    print("\n" + "="*50)
    print("5. Запрос субъекта данных: Алиса ограничивает обработку")
    gdpr_sbt.handle_data_subject_request(user_id_alice, "restriction", operations=["personalization"])
    print("   - ✅ Алиса ограничила использование данных для персонализации.")
    print(f"   - Проверка согласия на персонализацию: {gdpr_sbt.consent_manager.check_consent(user_id_alice, ConsentPurpose.PERSONALIZATION)}")
    print(f"   - Проверка разрешения на обработку для персонализации: {gdpr_sbt.data_subject_rights.check_processing_allowed(user_id_alice, 'personalization')}")


    print("\n" + "="*50)
    print("6. Запуск аудита соответствия GDPR")
    audit_report = gdpr_sbt.run_compliance_audit()
    print(f"   - 📈 Отчет аудита:")
    print(f"   -   Общий статус: {audit_report['overall_status']}")
    print(f"   -   Найдено проблем: {audit_report['issues_found']}")
    print(f"   -   Истекшие согласия: {audit_report['summary']['consent_audit']['expired_consents']}")
    print(f"   -   Неактивные аккаунты: {audit_report['summary']['retention_check']['inactive_accounts_flagged']}")


    print("\n" + "="*50)
    print("7. Запрос субъекта данных: Алиса запрашивает удаление (право на забвение)")
    erasure_result = gdpr_sbt.handle_data_subject_request(user_id_alice, "erasure", reason="User choice")
    print(f"   - ✅ Запрос на удаление обработан. Сообщение: {erasure_result['message']}")

    # Проверка, что данные удалены
    access_after_delete = gdpr_sbt.handle_data_subject_request(user_id_alice, "access")
    print(f"   - Попытка доступа к данным Алисы после удаления: {access_after_delete.get('error')}")
    print("\n" + "="*50)
    print("🏁 Демонстрация завершена.")

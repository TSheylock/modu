# Records of Processing Activities (RoPA)
## SASOK Platform — GDPR Article 30 Compliance

**Version:** 1.0  
**Date:** 2025-12-23  
**Controller:** SASOK Project  
**Contact:** hello@sasok.xyz  
**DPO Contact:** [To be appointed]

---

## 1. Controller Information

| Field | Value |
|-------|-------|
| **Organization** | SASOK Project |
| **Address** | [To be provided] |
| **Contact Email** | hello@sasok.xyz |
| **Data Protection Officer** | [To be appointed] |
| **EU Representative** | [If required — non-EU establishment] |

---

## 2. Processing Activities Register

### 2.1 Visual Emotion Analysis

| Field | Description |
|-------|-------------|
| **Purpose** | Real-time facial expression analysis for emotional self-reflection |
| **Legal Basis** | Explicit consent (Art. 9(2)(a)) — biometric special category |
| **Data Categories** | Facial landmarks (68 points), emotion labels, confidence scores |
| **Data Subjects** | Platform users (18+ years) |
| **Recipients** | Internal AI models only; no third-party sharing |
| **Transfers** | None — local processing |
| **Retention** | Raw video: NOT stored; Emotion vectors: 24 months |
| **Technical Measures** | Real-time processing, AES-256 encryption for vectors |
| **Organizational Measures** | Access control (RBAC), logging |

---

### 2.2 Audio Emotion Analysis

| Field | Description |
|-------|-------------|
| **Purpose** | Voice sentiment analysis for emotional context |
| **Legal Basis** | Explicit consent (Art. 9(2)(a)) |
| **Data Categories** | Voice patterns, pitch, intonation, emotion labels |
| **Data Subjects** | Platform users |
| **Recipients** | Internal AI models only |
| **Transfers** | None — local processing |
| **Retention** | Raw audio: NOT stored; Emotion vectors: 24 months |
| **Technical Measures** | Real-time processing, TLS 1.3, AES-256 |
| **Organizational Measures** | Access control, audit logging |

---

### 2.3 Text Emotion Analysis

| Field | Description |
|-------|-------------|
| **Purpose** | Written content sentiment classification |
| **Legal Basis** | Consent (Art. 6(1)(a)) |
| **Data Categories** | Text input, emotion labels, confidence scores |
| **Data Subjects** | Platform users |
| **Recipients** | Internal AI models; potential API processing |
| **Transfers** | Model inference may involve cloud services (see sub-processors) |
| **Retention** | Text: 24 months (user option to delete earlier) |
| **Technical Measures** | TLS 1.3, pseudonymization |
| **Organizational Measures** | Purpose limitation, access control |

---

### 2.4 Behavioral Profiling

| Field | Description |
|-------|-------------|
| **Purpose** | Cognitive state inference (focus, hesitation, distraction) |
| **Legal Basis** | Consent (Art. 6(1)(a)) |
| **Data Categories** | Click patterns, typing speed, response times, session duration |
| **Data Subjects** | Platform users |
| **Recipients** | Internal analytics only |
| **Transfers** | None |
| **Retention** | Individual events: 30 days; Aggregated: 24 months |
| **Technical Measures** | Aggregation, pseudonymization |
| **Organizational Measures** | Access control, purpose limitation |

---

### 2.5 Episodic Memory Storage

| Field | Description |
|-------|-------------|
| **Purpose** | Long-term storage of emotional episodes for user reflection |
| **Legal Basis** | Consent (Art. 6(1)(a)) |
| **Data Categories** | Emotional vectors, episode metadata, relational links |
| **Data Subjects** | Platform users |
| **Recipients** | Internal modules only |
| **Transfers** | None — local SQLite database |
| **Retention** | 24 months or until user deletion |
| **Technical Measures** | SQLite with encryption, access controls |
| **Organizational Measures** | Backup procedures, deletion audit trail |

---

### 2.6 Blockchain SBT Issuance

| Field | Description |
|-------|-------------|
| **Purpose** | Issuance of Emotional Soulbound Tokens for Web3 identity |
| **Legal Basis** | Consent + Contract (Art. 6(1)(a), Art. 6(1)(b)) |
| **Data Categories** | Wallet address, token ID, reputation score, interaction count |
| **Data Subjects** | Users who opt-in to Web3 features |
| **Recipients** | Ethereum network (public blockchain) |
| **Transfers** | On-chain data is globally accessible |
| **Retention** | Permanent (blockchain immutability) |
| **Technical Measures** | ⚠️ **Architecture update needed** — personal data should be off-chain |
| **Organizational Measures** | Consent for on-chain data, deletion key mechanism planned |

> [!CAUTION]
> Current `SasokPlatform.sol` stores `reputation` and `interactionCount` on-chain. This conflicts with GDPR Art. 17 (right to erasure). Architecture update required by Q2 2026.

---

### 2.7 User Account Management

| Field | Description |
|-------|-------------|
| **Purpose** | User registration, authentication, profile management |
| **Legal Basis** | Contract (Art. 6(1)(b)) |
| **Data Categories** | Email, username, preferences, consent records |
| **Data Subjects** | All registered users |
| **Recipients** | Authentication provider (if SSO enabled) |
| **Transfers** | Potential sub-processor (see Appendix) |
| **Retention** | Duration of account + 30 days after deletion |
| **Technical Measures** | Password hashing (bcrypt), TLS, session management |
| **Organizational Measures** | Access control, account recovery procedures |

---

## 3. Recipients and Sub-Processors

| Sub-Processor | Purpose | Location | DPA Status |
|---------------|---------|----------|------------|
| AWS | Cloud infrastructure | Frankfurt, Germany | ✅ Signed |
| MongoDB Atlas | Database hosting | Frankfurt, Germany | ✅ Signed |
| Neo4j Aura | Graph database | Frankfurt, Germany | ✅ Signed |
| Redis Labs | Cache | Frankfurt, Germany | ✅ Signed |
| HuggingFace | Model inference (optional) | EU/US | ⏳ Review |

---

## 4. International Transfers

| Transfer | Destination | Mechanism | Risk Assessment |
|----------|-------------|-----------|-----------------|
| None by default | N/A | N/A | Low risk |
| HuggingFace API (if enabled) | US | SCCs | Medium risk — opt-in only |

---

## 5. Technical and Organizational Measures (Art. 32)

### 5.1 Technical Measures

| Measure | Implementation |
|---------|----------------|
| Encryption at rest | AES-256 |
| Encryption in transit | TLS 1.3 |
| Pseudonymization | Internal UUID, no PII in processing |
| Access control | RBAC, principle of least privilege |
| Logging | All access/modifications logged |
| Backup | Encrypted daily backups, 7-day retention |

### 5.2 Organizational Measures

| Measure | Implementation |
|---------|----------------|
| Staff training | ⏳ Q1 2026 |
| DPO appointment | ⏳ Required |
| Privacy by design | Embedded in development |
| Incident response | ⏳ Q1 2026 |
| Audit schedule | Quarterly internal, annual external |

---

## 6. Data Subject Rights Procedures

| Right | Process | Response Time |
|-------|---------|---------------|
| Access (Art. 15) | Export via Settings | 30 days |
| Rectification (Art. 16) | Profile edit | 7 days |
| Erasure (Art. 17) | Account deletion | 30 days |
| Restriction (Art. 18) | Support request | 30 days |
| Portability (Art. 20) | JSON export | 30 days |
| Objection (Art. 21) | Per-category toggle | Immediate |

---

## 7. Document Control

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-23 | Initial creation | System |

---

## Related Documents

- [DPIA_EMOTIONAL_AI.md](file:///Users/tsheylock/.gemini/antigravity/brain/122989e0-acb9-402c-8a76-746b14f7dd03/compliance/DPIA_EMOTIONAL_AI.md)
- [FRIA_EMOTIONAL_AI.md](file:///Users/tsheylock/.gemini/antigravity/brain/122989e0-acb9-402c-8a76-746b14f7dd03/compliance/FRIA_EMOTIONAL_AI.md)
- [DPA_GDPR.md](file:///Users/tsheylock/modu/legal/DPA_GDPR.md)

# Data Protection Impact Assessment (DPIA)
## SASOK Emotional AI Processing System

**Version:** 1.0  
**Date:** 2025-12-23  
**Status:** Active  
**Next Review:** 2026-06-23  
**DPO Approval:** _________________________

---

## 1. Executive Summary

This DPIA assesses the privacy risks associated with SASOK's emotional AI processing systems, which collect and analyze visual, audial, biometric, and behavioral data to generate emotional profiles for users. The assessment covers:

- Real-time webcam emotion analysis (MediaPipe + DeepFace)
- Text and audio emotion classification (Transformers)
- Behavioral profiling and cognitive state inference
- Emotional Soulbound Token (SBT) issuance on blockchain

**Risk Level:** HIGH — Requires ongoing monitoring and mitigation measures.

---

## 2. Processing Description (GDPR Art. 35(7)(a))

### 2.1 Nature of Processing

| Component | Technology | Data Type | Processing Location |
|-----------|------------|-----------|---------------------|
| **Visual Analysis** | MediaPipe Face Mesh + DeepFace | Facial landmarks, expressions | Real-time, local |
| **Audio Analysis** | wav2vec2, DistilBERT | Voice patterns, sentiment | Real-time, local |
| **Text Analysis** | emotion-english-distilroberta-base | Written content sentiment | API server |
| **Behavioral Analysis** | Custom logic | Click patterns, typing speed | Session logs |
| **Memory Module** | SQLite episodic storage | Emotional episodes, relations | Encrypted storage |
| **Blockchain** | ERC721 (SasokPlatform.sol) | Emotional SBT metadata | Ethereum network |

### 2.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐ │
│  │ Webcam  │  │  Audio  │  │  Text   │  │ Behavioral (clicks,    │ │
│  │ Feed    │  │  Input  │  │  Input  │  │ typing, response time) │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └───────────┬─────────────┘ │
└───────┼────────────┼────────────┼───────────────────┼───────────────┘
        │            │            │                   │
        ▼            ▼            ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PERCEPTION LAYER (sasok_core)                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ WebcamEmotion    │  │ AudioEmotion     │  │ TextEmotion       │  │
│  │ Analyzer:        │  │ Analyzer:        │  │ Analyzer:         │  │
│  │ MediaPipe+       │  │ wav2vec2         │  │ DistilRoBERTa     │  │
│  │ DeepFace         │  │                  │  │                   │  │
│  └────────┬─────────┘  └────────┬─────────┘  └─────────┬─────────┘  │
└───────────┼─────────────────────┼──────────────────────┼────────────┘
            │                     │                      │
            ▼                     ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EMOTION CLASSIFICATION                          │
│  Normalized: joy, sadness, anger, fear, surprise, disgust, neutral  │
│  + SASOK_DOUBT flag if confidence < 70%                             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      NATS EVENT BUS                                  │
│  Topics: sasok.emotion.detected, sasok.webcam.status, ...           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────────┐
│ MEMORY MODULE │  │ REFLECTION    │  │ BLOCKCHAIN LAYER  │
│ (SQLite)      │  │ MODULE        │  │ (SasokPlatform)   │
│ Episodes,     │  │ Self-doubt,   │  │ User Profiles,    │
│ Relations     │  │ Metacognition │  │ Reputation, SBT   │
└───────────────┘  └───────────────┘  └───────────────────┘
```

### 2.3 Scope of Processing

| Dimension | Description |
|-----------|-------------|
| **Data Subjects** | Platform users (18+ years) |
| **Volume** | Real-time streams per session; emotional vectors retained |
| **Geographic Scope** | Global (EU focus for GDPR) |
| **Duration** | Continuous during active sessions |
| **Retention** | Emotional vectors: 24 months; On-chain data: permanent |

---

## 3. Necessity and Proportionality (GDPR Art. 35(7)(b))

### 3.1 Purpose Legitimacy

| Purpose | Justification | Alternative Considered |
|---------|---------------|------------------------|
| Emotional self-reflection | Core platform value | Text-only journaling (insufficient) |
| Personalized content | User engagement | Demographic targeting (less precise) |
| Emotional ID (SBT) | Web3 identity | No blockchain (loses decentralization) |

### 3.2 Data Minimization

| Data Category | Necessity | Minimization Measure |
|---------------|-----------|----------------------|
| Raw video | ESSENTIAL | Real-time only; NOT stored |
| Raw audio | ESSENTIAL | Real-time only; NOT stored |
| Facial landmarks | ESSENTIAL | 68-point mesh; no image storage |
| Emotional vectors | ESSENTIAL | Pseudonymized; internal ID only |
| Behavioral logs | MODERATE | Aggregated; purged after 30 days |

### 3.3 Legal Basis

| Processing Activity | Legal Basis | Documentation |
|---------------------|-------------|---------------|
| Visual/audio emotion analysis | **Explicit Consent** (Art. 9(2)(a)) | CONSENT_FORM.md |
| Biometric processing | **Explicit Consent** (Art. 9(2)(a)) | CONSENT_FORM.md |
| Behavioral profiling | **Explicit Consent** (Art. 6(1)(a)) | CONSENT_FORM.md |
| Emotional SBT issuance | **Consent + Contract** (Art. 6(1)(b)) | CONSENT_FORM.md |

---

## 4. Risk Assessment (GDPR Art. 35(7)(c))

### 4.1 Risk Matrix

| ID | Risk | Likelihood | Impact | Inherent | Mitigation | Residual |
|----|------|------------|--------|----------|------------|----------|
| R1 | **Discrimination** via biased models | Medium | High | HIGH | Fairness testing, bias audits | Medium |
| R2 | **Manipulation** of vulnerable users | Low | Critical | HIGH | Consent refresh, opt-out | Low |
| R3 | **Transparency deficit** | Medium | Medium | MEDIUM | Model explainability | Low |
| R4 | **Data breach** | Low | High | MEDIUM | Encryption, access controls | Low |
| R5 | **Erasure conflict** with blockchain | Medium | High | HIGH | Off-chain data, hash on-chain | Medium |
| R6 | **Secondary use** without consent | Low | High | MEDIUM | Purpose limitation | Low |
| R7 | **Surveillance perception** | Medium | Medium | MEDIUM | Clear consent UI | Low |

### 4.2 Critical Risk: Blockchain Erasure Conflict (R5)

**Current `SasokPlatform.sol` Architecture (NON-COMPLIANT):**
```solidity
struct UserProfile {
    bool isRegistered;
    uint256 reputation;          // ❌ Personal data on-chain
    uint256 interactionCount;    // ❌ Personal data on-chain
    uint256[] ownedTokens;
    uint256 lastInteraction;
}
```

**Required Change:**
- Move `reputation`, `interactionCount` to off-chain encrypted storage
- Store only hashes/token IDs on-chain
- Implement key destruction for deletion requests

---

## 5. Mitigation Measures (GDPR Art. 35(7)(d))

### 5.1 Technical Measures

| Measure | Status | Responsible |
|---------|--------|-------------|
| Encryption at rest (AES-256) | ✅ Done | DevOps |
| Encryption in transit (TLS 1.3) | ✅ Done | DevOps |
| Pseudonymization | ✅ Done | Backend |
| Access control (RBAC) | ✅ Done | Security |
| Raw data non-storage | ✅ Done | Perception |
| **Off-chain personal data** | ⏳ Required | Blockchain |
| **Deletion key mechanism** | ⏳ Required | Blockchain |

### 5.2 Organizational Measures

| Measure | Status | Responsible |
|---------|--------|-------------|
| DPO appointment | ⏳ Required | Management |
| Biometric training | ⏳ Required | HR |
| Incident response plan | ⏳ Required | Security |
| Quarterly fairness audits | ⏳ Required | AI Team |

### 5.3 Data Subject Rights

| Right | Implementation | Status |
|-------|----------------|--------|
| Access (Art. 15) | Export function | ✅ |
| Rectification (Art. 16) | Profile edit | ✅ |
| Erasure (Art. 17) | Account deletion | ⚠️ Blockchain update needed |
| Portability (Art. 20) | JSON export | ✅ |
| Objection (Art. 21) | Per-category opt-out | ✅ |
| No automated decisions (Art. 22) | Human review | ⏳ Required |

---

## 6. Monitoring and Review

### 6.1 Review Triggers

| Event | Action |
|-------|--------|
| EDPB guidance on emotional AI | Reassess risk matrix |
| AI Act enforcement (Aug 2026) | Add FRIA requirements |
| New model deployment | Re-run bias audit |
| Data breach | Immediate DPIA review |

### 6.2 Audit Schedule

| Audit | Frequency | Next Due |
|-------|-----------|----------|
| Fairness/bias testing | Quarterly | 2026-03-23 |
| Security penetration | Annual | 2026-12-23 |
| Consent review | Semi-annual | 2026-06-23 |
| Full DPIA refresh | Annual | 2026-12-23 |

---

## 7. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Processing Owner | Timmy Sheylock | _________ | 2025-12-23 |
| DPO | _________________ | _________ | ___________ |
| Legal Counsel | _________________ | _________ | ___________ |

---

## Appendix: Model Documentation

| Model | Source | Training Data | Limitations |
|-------|--------|---------------|-------------|
| DeepFace | serengil/deepface | VGG-Face, FER+ | Lower accuracy on darker skin |
| DistilRoBERTa | HuggingFace | GoEmotions | English-only |
| wav2vec2 | HuggingFace | RAVDESS, CREMA-D | English accents |

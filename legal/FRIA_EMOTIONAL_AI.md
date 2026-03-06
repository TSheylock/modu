# Fundamental Rights Impact Assessment (FRIA)
## SASOK Emotional AI System

**Version:** 1.0  
**Date:** 2025-12-23  
**AI Act Reference:** Articles 27-29a  
**Compliance Deadline:** August 2, 2026  
**Status:** Draft

---

## 1. System Identification

| Field | Value |
|-------|-------|
| **AI System Name** | SASOK Emotional AI Platform |
| **Provider** | SASOK Project |
| **Deployer** | End users (self-directed) |
| **Risk Classification** | Potentially High-Risk (emotion recognition) |
| **Deployment Context** | Consumer self-reflection platform (NOT workplace) |

---

## 2. Why: Problem Definition

### 2.1 Business Objective

SASOK provides real-time emotional self-reflection through AI-driven analysis of facial expressions, voice patterns, text sentiment, and behavioral signals. The goal is **personal insight and self-development**, not external evaluation.

### 2.2 Alternatives Considered

| Alternative | Assessment | Reason for Rejection |
|-------------|------------|----------------------|
| Text-only journaling | Insufficient | No real-time emotional mirroring |
| Manual mood logging | Limited accuracy | User self-report bias |
| No AI assistance | Misses core value | Platform purpose is AI-assisted reflection |

### 2.3 Necessity Justification

Emotional AI is **essential** to SASOK's core function: providing users with an objective mirror of their emotional states that they cannot perceive themselves.

---

## 3. What: Input Data

### 3.1 Data Categories

| Data Type | Source | Quality Measures | Consent Required |
|-----------|--------|------------------|------------------|
| **Facial landmarks** | Webcam (MediaPipe) | 68-point mesh, real-time | Yes (explicit) |
| **Emotion labels** | DeepFace inference | Confidence scoring | Yes (explicit) |
| **Voice patterns** | Microphone (wav2vec2) | 16kHz sampling | Yes (explicit) |
| **Text sentiment** | User input | DistilRoBERTa | Yes (explicit) |
| **Behavioral signals** | Platform interaction | Timing, patterns | Yes (explicit) |

### 3.2 Training Data Provenance

| Model | Dataset | Demographics | Known Bias |
|-------|---------|--------------|------------|
| DeepFace | VGG-Face, FER+ | Western-dominant | Lower accuracy: darker skin |
| DistilRoBERTa | GoEmotions | English speakers | Western emotion taxonomy |
| wav2vec2 | RAVDESS, CREMA-D | US/UK English | Accent bias |

### 3.3 Data Quality Controls

- SASOK_DOUBT flag triggers when model confidence < 70%
- Low-confidence results excluded from automated decisions
- User can review and override any classification

---

## 4. Throughput: Algorithm

### 4.1 Processing Logic

```
Input → Perception Layer → Classification → Normalization → Output
         (MediaPipe,       (7 emotions,    (0-1 scale,    (Event bus,
          DeepFace,         confidence)     multilingual   storage)
          wav2vec2)                         mapping)
```

### 4.2 Model Transparency

| Model | Type | Explainability | Human Override |
|-------|------|----------------|----------------|
| DeepFace | CNN | Feature attribution available | Yes |
| DistilRoBERTa | Transformer | Attention weights | Yes |
| wav2vec2 | Transformer | Attention weights | Yes |

### 4.3 Accuracy Metrics

| Model | Accuracy | False Positive Rate | Bias Audit |
|-------|----------|---------------------|------------|
| DeepFace | ~85% | 12% | Quarterly planned |
| DistilRoBERTa | ~89% | 8% | Quarterly planned |
| wav2vec2 | ~82% | 15% | Quarterly planned |

---

## 5. How: Deployment

### 5.1 Deployment Context

| Dimension | Specification |
|-----------|---------------|
| **User Population** | Adults 18+, self-selected |
| **Use Case** | Personal self-reflection, NOT employment/education |
| **Decision Impact** | No external consequences; user controls all data |
| **Human Oversight** | User is always in control; no automated actions |

### 5.2 AI Act Prohibited Practice Check

| Prohibited Practice (Art. 5) | SASOK Status | Justification |
|------------------------------|--------------|---------------|
| **Workplace emotion recognition** | ❌ N/A | Consumer platform only |
| **Social scoring** | ❌ N/A | Personal use; no third-party access |
| **Manipulation of vulnerable groups** | ✅ Controlled | Consent + opt-out available |
| **Real-time biometric ID (public spaces)** | ❌ N/A | Personal device only |

### 5.3 Human Review Mechanism

- Users can view all emotional classifications
- Users can mark classifications as incorrect
- Users can disable any modality at any time
- No automated decisions made without user action

---

## 6. Fundamental Rights Mapping

### 6.1 Rights Affected

| Right | Impact Level | Safeguard |
|-------|--------------|-----------|
| **Privacy** (Art. 7 Charter) | High | Encryption, pseudonymization, local processing |
| **Data protection** (Art. 8) | High | GDPR compliance, DPIA completed |
| **Human dignity** (Art. 1) | Medium | No manipulation; user autonomy preserved |
| **Freedom of expression** (Art. 11) | Low | No content moderation based on emotion |
| **Non-discrimination** (Art. 21) | Medium | Bias audits, fairness testing |
| **Consumer protection** (Art. 38) | Medium | Clear consent, right to withdraw |

### 6.2 Proportionality Assessment

| Processing | Necessity | Proportionality | Justification |
|------------|-----------|-----------------|---------------|
| Facial emotion | High | Proportionate | Core platform value; real-time only |
| Voice emotion | High | Proportionate | Multimodal accuracy; real-time only |
| Behavioral | Medium | Proportionate | Cognitive context; aggregated |
| Blockchain SBT | Medium | ⚠️ Review | Off-chain architecture needed |

---

## 7. Risk Mitigation

### 7.1 Technical Safeguards

| Safeguard | Status |
|-----------|--------|
| Data encryption (AES-256) | ✅ Implemented |
| Local processing preference | ✅ Implemented |
| SASOK_DOUBT confidence flagging | ✅ Implemented |
| User data export (portability) | ✅ Implemented |
| Account deletion | ⚠️ Blockchain arch update needed |

### 7.2 Organizational Safeguards

| Safeguard | Status |
|-----------|--------|
| Bias audit protocol | ⏳ Q1 2026 |
| Model documentation | ✅ Completed |
| User rights process | ✅ Documented |
| Incident response | ⏳ Q1 2026 |

---

## 8. User Contestability

### 8.1 Contestation Mechanisms

| Mechanism | Implementation |
|-----------|----------------|
| View classifications | Real-time display |
| Mark as incorrect | Feedback button |
| Request human review | Support channel |
| Opt-out of processing | Per-modality toggle |
| Full data deletion | Account settings |

### 8.2 Redress Process

1. User submits contested classification
2. System logs feedback for model improvement
3. User can request manual review (72-hour SLA)
4. Escalation to DPO if unresolved

---

## 9. Monitoring

### 9.1 Continuous Assessment

| Metric | Frequency | Threshold |
|--------|-----------|-----------|
| Model accuracy | Quarterly | >80% |
| Demographic parity | Quarterly | <5% gap |
| User complaints | Monthly | <1% rate |
| Opt-out rate | Monthly | Monitor trend |

### 9.2 Review Schedule

| Review Type | Frequency | Next Due |
|-------------|-----------|----------|
| Bias audit | Quarterly | 2026-03-23 |
| FRIA refresh | Annual | 2026-12-23 |
| AI Act compliance | Before Aug 2026 | 2026-06-01 |

---

## 10. Approval

| Role | Name | Date |
|------|------|------|
| AI System Owner | Timmy Sheylock | 2025-12-23 |
| Legal Counsel | _________________ | ___________ |
| Fundamental Rights Officer | _________________ | ___________ |

---

## Related Documents

- [DPIA_EMOTIONAL_AI.md](file:///Users/tsheylock/.gemini/antigravity/brain/122989e0-acb9-402c-8a76-746b14f7dd03/compliance/DPIA_EMOTIONAL_AI.md)
- [ROPA_SASOK.md](file:///Users/tsheylock/.gemini/antigravity/brain/122989e0-acb9-402c-8a76-746b14f7dd03/compliance/ROPA_SASOK.md)

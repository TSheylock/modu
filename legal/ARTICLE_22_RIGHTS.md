# GDPR Article 22: Automated Decision-Making Rights
## SASOK Platform User Rights Documentation

**Version:** 1.0  
**Date:** 2025-12-23  
**Effective:** Immediately

---

## 1. Overview

Under GDPR Article 22, users have the right **not to be subject to decisions based solely on automated processing** that produce legal or similarly significant effects. This document explains how SASOK applies to these requirements.

---

## 2. Applicability to SASOK

### 2.1 Does Article 22 Apply?

| Processing Type | Solely Automated? | Legal/Significant Effect? | Art. 22 Applies? |
|-----------------|-------------------|--------------------------|------------------|
| Emotion classification | Yes | **No** — informational only | ❌ Not triggered |
| Content personalization | Partially | Potentially | ⚠️ Safeguards apply |
| Emotional SBT issuance | Partially | Potentially | ⚠️ Safeguards apply |
| Reputation scoring | Yes | Potentially | ⚠️ Safeguards apply |

### 2.2 SASOK Safeguards

Even where Art. 22 may not strictly apply, SASOK implements safeguards:

1. **No consequential decisions** — Emotional classifications inform the user only
2. **User controls all outputs** — No external sharing without explicit action
3. **Opt-out available** — Any modality can be disabled
4. **Human review on request** — Users can request manual review

---

## 3. Your Rights

### 3.1 Right to Know

You have the right to know:
- That automated processing is being used
- The logic involved in emotional classification
- The significance and consequences of the processing

**How SASOK provides this:**
- Real-time display of emotion classifications
- Transparency about models used (Model Documentation)
- This rights document

### 3.2 Right to Explanation

You can request an explanation of any specific classification or decision.

**How to request:**
1. Navigate to Settings → Privacy → Request Explanation
2. Specify the classification/decision in question
3. Response within 30 days

### 3.3 Right to Human Intervention

You can request human review of any automated processing result.

**How to request:**
1. Mark the specific result as "Request Review"
2. Provide context for your concern
3. Human review completed within 72 hours

### 3.4 Right to Contest

You can challenge any classification or decision you believe is incorrect.

**How to contest:**
1. Use the "Contest This" button on any classification
2. Explain why you believe it's incorrect
3. System logs your feedback for model improvement
4. If unresolved, escalate to DPO

### 3.5 Right to Opt-Out

You can opt out of specific processing at any time.

**Available opt-outs:**
- [ ] Webcam emotion analysis
- [ ] Audio emotion analysis
- [ ] Text emotion analysis
- [ ] Behavioral profiling
- [ ] Blockchain SBT features

Access via: Settings → Privacy → Processing Controls

---

## 4. Safeguards Implemented

### 4.1 Technical Safeguards

| Safeguard | Description |
|-----------|-------------|
| **SASOK_DOUBT flag** | Low-confidence results (< 70%) are flagged |
| **No automated consequences** | Classifications are informational only |
| **Visibility** | All classifications visible to user |
| **Override capability** | User can mark any result as incorrect |

### 4.2 Organizational Safeguards

| Safeguard | Description |
|-----------|-------------|
| **Human review process** | Dedicated support channel |
| **DPO escalation** | Available for unresolved concerns |
| **Bias audits** | Quarterly fairness testing |
| **Transparency reports** | [Planned] Annual publication |

---

## 5. Processing Activities Detail

### 5.1 Emotional Content Personalization

**What happens:** Your emotional profile may influence:
- Content recommendations shown to you
- Interface adaptations (e.g., calming mode)
- Suggested activities or reflections

**Your control:**
- View what content was influenced
- Disable personalization entirely
- Reset emotional profile

### 5.2 Emotional SBT Issuance

**What happens:** If you opt-in to Web3 features:
- Your emotional patterns contribute to reputation score
- Score is recorded on blockchain via Soulbound Token

**Your control:**
- Opt-in only — never automatic
- View score calculation factors
- Request recalculation (off-chain data only)

> [!WARNING]
> On-chain data cannot be deleted due to blockchain immutability. Only opt-in if you understand this limitation.

### 5.3 Reputation Scoring

**What happens:** Your engagement patterns contribute to a reputation score.

**Factors considered:**
- Consistency of engagement
- Emotional growth patterns
- Platform interaction quality

**Your control:**
- View current score and factors
- Request human review of score
- Opt-out of scoring entirely

---

## 6. Redress Process

### 6.1 Complaint Procedure

| Step | Action | Timeline |
|------|--------|----------|
| 1 | Submit concern via Settings → Privacy | Immediate |
| 2 | Initial response from support | 72 hours |
| 3 | Investigation (if needed) | 14 days |
| 4 | Resolution or escalation to DPO | 30 days |
| 5 | DPO final decision | 60 days |

### 6.2 Supervisory Authority

If unsatisfied with our response, you may lodge a complaint with your national Data Protection Authority.

---

## 7. Contact

| Purpose | Contact |
|---------|---------|
| General inquiries | hello@sasok.xyz |
| Privacy concerns | privacy@sasok.xyz |
| DPO | [To be appointed] |

---

## Related Documents

- [DPIA_EMOTIONAL_AI.md](file:///Users/tsheylock/.gemini/antigravity/brain/122989e0-acb9-402c-8a76-746b14f7dd03/compliance/DPIA_EMOTIONAL_AI.md)
- [CONSENT_FORM.md](file:///Users/tsheylock/modu/legal/CONSENT_FORM.md)

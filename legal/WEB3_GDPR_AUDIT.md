# Web3 + GDPR Compliance Audit
## SASOK Blockchain Architecture vs. Right to Erasure

**Version:** 1.0  
**Date:** 2025-12-23  
**Compliance Requirement:** GDPR Article 17

---

## 1. Executive Summary

The current `SasokPlatform.sol` smart contract stores personal data directly on the Ethereum blockchain, creating a **fundamental conflict** with GDPR Article 17 (right to erasure). This document analyzes the conflict and proposes a compliant hybrid architecture.

**Current Status:** ❌ NON-COMPLIANT  
**Risk Level:** HIGH  
**Required Action:** Architecture refactor by Q2 2026

---

## 2. Current Architecture Analysis

### 2.1 SasokPlatform.sol Data Structures

```solidity
// File: /Users/tsheylock/modu/backend/contracts/SasokPlatform.sol

struct UserProfile {
    bool isRegistered;           // ⚠️ Identifier
    uint256 reputation;          // ❌ Personal data
    uint256 interactionCount;    // ❌ Personal data
    uint256[] ownedTokens;       // ⚠️ Asset data
    uint256 lastInteraction;     // ⚠️ Behavioral data
}

struct Interaction {
    address user;                // ⚠️ Pseudonymous identifier
    string interactionType;      // ⚠️ Behavioral data
    string metadata;             // ❌ Potentially personal
    uint256 timestamp;           // ⚠️ Temporal data
    bool verified;               // ⚠️ Status data
}
```

### 2.2 On-Chain Data Classification

| Data Element | GDPR Category | Deletable? | Compliance |
|--------------|---------------|------------|------------|
| `user` (address) | Pseudonymous ID | ❌ No | ⚠️ Acceptable |
| `reputation` | Personal data | ❌ No | ❌ Violation |
| `interactionCount` | Personal data | ❌ No | ❌ Violation |
| `interactionType` | Behavioral | ❌ No | ⚠️ Review |
| `metadata` | Varies | ❌ No | ❌ Violation |
| `lastInteraction` | Behavioral | ❌ No | ⚠️ Review |

### 2.3 web3_handler.py Data Flow

```python
# File: /Users/tsheylock/modu/backend/web3_handler.py

# Currently stores wallet data in local cache:
self.connected_wallets[wallet_address] = {
    "address": wallet_address,
    "network": "ethereum",
    "connected_at": datetime.utcnow().isoformat(),
    "balance": self.w3.eth.get_balance(wallet_address)
}
# This local data IS deletable ✅
```

---

## 3. GDPR Article 17 Conflict

### 3.1 The Right to Erasure

Under Article 17, users have the right to request deletion of their personal data when:
- Data is no longer necessary for its purpose
- User withdraws consent
- User objects to processing
- Data was unlawfully processed
- Legal obligation requires deletion

### 3.2 Blockchain Immutability

Blockchain's core property is **immutability** — once data is written, it cannot be modified or deleted. This creates a direct conflict with the right to erasure.

### 3.3 EDPB Position (2025)

The European Data Protection Board has clarified:

> "Personal data should not be stored directly on a public blockchain. Where blockchain is used, personal data should be stored off-chain with only cryptographic references on-chain."

---

## 4. Proposed Compliant Architecture

### 4.1 Hybrid Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFF-CHAIN (GDPR-COMPLIANT)                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ User Data       │  │ Emotional       │  │ Deletion Keys   │  │
│  │ (encrypted)     │  │ Profiles        │  │ (for revocation)│  │
│  │                 │  │ (encrypted)     │  │                 │  │
│  │ - reputation    │  │ - vectors       │  │ - encryption    │  │
│  │ - interactions  │  │ - history       │  │   keys          │  │
│  │ - metadata      │  │ - metrics       │  │ - access tokens │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│                        ┌───────▼───────┐                        │
│                        │ Hash Function │                        │
│                        └───────┬───────┘                        │
└────────────────────────────────┼────────────────────────────────┘
                                 │ Hash references only
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ON-CHAIN (BLOCKCHAIN)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Data Hashes     │  │ Token IDs       │  │ Timestamps      │  │
│  │ (unreconstructable) │ (NFT ownership) │  │ (events)        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Revised Smart Contract

```solidity
// PROPOSED: Compliant SasokPlatformV2.sol

struct UserProfileV2 {
    bool isRegistered;
    bytes32 profileHash;         // ✅ Hash only, no personal data
    uint256[] ownedTokens;       // ✅ Asset ownership (not personal)
    uint256 lastActivityBlock;   // ✅ Block number only
}

struct InteractionV2 {
    address user;                // ✅ Pseudonymous
    bytes32 interactionHash;     // ✅ Hash of off-chain data
    uint256 blockNumber;         // ✅ Block reference only
}

// Deletion implementation:
// 1. User requests deletion
// 2. Off-chain data is deleted
// 3. Encryption keys are destroyed
// 4. On-chain hashes become unreconstructable
// 5. profileHash is updated to null hash
```

### 4.3 Deletion Mechanism

| Step | Action | Location |
|------|--------|----------|
| 1 | User submits deletion request | API |
| 2 | Off-chain personal data deleted | Database |
| 3 | Encryption keys destroyed | Key management |
| 4 | On-chain `profileHash` → `0x0` | Blockchain |
| 5 | Audit log created | Deletion registry |

**Result:** On-chain hashes remain but are **cryptographically unreconstructable** — functionally equivalent to deletion.

---

## 5. Data Migration Plan

### 5.1 Current → Compliant Migration

| Data Element | Current Location | Target Location | Migration |
|--------------|------------------|-----------------|-----------|
| `reputation` | On-chain | Off-chain encrypted | Export → Encrypt → Delete on-chain |
| `interactionCount` | On-chain | Off-chain | Export → Delete on-chain |
| `metadata` | On-chain | Off-chain encrypted | Export → Encrypt → Delete on-chain |
| Token ownership | On-chain | On-chain | Keep (non-personal) |

### 5.2 Timeline

| Phase | Task | Deadline |
|-------|------|----------|
| 1 | Design compliant contract V2 | Jan 2026 |
| 2 | Deploy off-chain storage | Feb 2026 |
| 3 | Implement migration scripts | Mar 2026 |
| 4 | Test deletion mechanism | Apr 2026 |
| 5 | Deploy V2, freeze V1 | May 2026 |
| 6 | User migration period | Jun 2026 |

---

## 6. Deletion Audit Trail

### 6.1 Required Evidence

For each deletion request, maintain:

```json
{
  "deletion_id": "uuid",
  "user_id_hash": "sha256...",
  "request_timestamp": "2025-12-23T04:00:00Z",
  "completion_timestamp": "2025-12-23T04:05:00Z",
  "off_chain_deleted": true,
  "keys_destroyed": true,
  "on_chain_hash_nullified": true,
  "blockchain_tx": "0x...",
  "verification_hash": "sha256..."
}
```

### 6.2 Forensic Verification

Upon deletion, verify:
- [ ] Personal data cannot be reconstructed from on-chain data
- [ ] Encryption keys are destroyed (not just deleted)
- [ ] Backup systems also purged
- [ ] Third-party copies (if any) deleted

---

## 7. Compliance Checklist

| Requirement | Current | Target | Status |
|-------------|---------|--------|--------|
| Personal data off-chain | ❌ | ✅ | ⏳ Pending |
| Hash-only on-chain | ❌ | ✅ | ⏳ Pending |
| Deletion mechanism | ❌ | ✅ | ⏳ Pending |
| Key destruction | ❌ | ✅ | ⏳ Pending |
| Audit trail | ❌ | ✅ | ⏳ Pending |
| Forensic verification | ❌ | ✅ | ⏳ Pending |

---

## 8. Risk Assessment

| Risk | If Not Addressed | Mitigation |
|------|------------------|------------|
| DPA enforcement action | Fines up to 4% global revenue | Implement compliant architecture |
| User erasure requests | Unable to comply | Deploy deletion mechanism |
| Reputational damage | Loss of EU user trust | Proactive compliance |
| Class action | Collective user claims | Document good-faith efforts |

---

## 9. Recommendations

1. **Immediate:** Stop storing new personal data on-chain
2. **Q1 2026:** Deploy off-chain encrypted storage
3. **Q2 2026:** Migrate existing data, deploy V2 contract
4. **Ongoing:** Maintain deletion audit trail

---

## Related Documents

- [SasokPlatform.sol](file:///Users/tsheylock/modu/backend/contracts/SasokPlatform.sol)
- [web3_handler.py](file:///Users/tsheylock/modu/backend/web3_handler.py)
- [DPIA_EMOTIONAL_AI.md](file:///Users/tsheylock/.gemini/antigravity/brain/122989e0-acb9-402c-8a76-746b14f7dd03/compliance/DPIA_EMOTIONAL_AI.md)

# SASOK_— Emotional AI Backend (Powered by Xano)

> 🧠 SASOK-is a modular backend system designed to support emotionally-aware interfaces, Web3 identity, and DAO governance using an emotion-driven data architecture.

---

## 🔍 Overview

**SASOK_** is the emotional backbone of a Web3 application that merges artificial intelligence, affective computing, and decentralized governance. Built on top of **Xano**, it provides an API-first infrastructure with an extendable, real-time database, modular logic layers, and native support for identity, behavior tracking, and emotional analytics.

---

## 📐 System Architecture

Frontend (React / Flutter / WebView)
|
v
[ Xano API ]
|
+--> User & Identity Tables
+--> Emotional Engine (AI model versioning)
+--> DAO Layer (Proposal + Voting)
+--> Notification System
+--> IPFS Event Logging

---

## 🗂️ Database Tables

| Table Name            | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `user`                | Stores user credentials, preferences, and emotional fingerprint             |
| `identity_log`        | Tracks user interactions and emotional state analytics                      |
| `dao_proposal`        | Stores DAO proposals with metadata                                          |
| `dao_vote`            | Records user votes linked to proposals with optional emotional weighting    |
| `emotional_sbt`       | Links Soulbound Tokens to emotional states and identity events              |
| `ipfs_event`          | Logs decentralized file events and object hashes                            |
| `notification`        | Manages in-app and email notification delivery                              |
| `scheduled_event`     | Supports scheduled user or system actions (e.g. reminders, triggers)        |
| `ai_model_version`    | Tracks deployed AI versions influencing emotional logic                     |
| `emotional_milestone` | Captures user goal or emotion-driven achievements                           |

---

## 🔌 API Endpoints

Xano automatically generates RESTful endpoints for all tables. You can:
- **GET** data (`/user`, `/dao_vote`)
- **POST** new entries (e.g., `/identity_log`)
- **PATCH/PUT** updates (e.g., `/emotional_sbt`)
- **DELETE** entries if authorized

> Authentication: Supports JWT tokens, API keys, and session-based auth.

---

## 🔗 Integrations

You can connect this backend with:
- ✅ **No-code** tools (e.g., FlutterFlow, Webflow, Adalo)
- ✅ **React / Vue** frontends
- ✅ **LangChain / FastAPI** for AI logic
- ✅ **BI tools** (e.g., Retool, Metabase)
- ✅ **IPFS / Web3 SDKs** for decentralized content

---

## 🧠 Emotional Intelligence Use Cases

- Generate **emotional fingerprints** for personalization
- Use **identity logs** to track emotional journey
- Issue **SBTs** (Soulbound Tokens) tied to emotional data
- Govern decisions with **DAO voting systems** weighted by affective states
- Sync AI engine upgrades via `ai_model_version`

---

### 4. Configure Auth
Set up JWT, API keys, or external OAuth (Google, Web3 wallet, etc.)

### 5. Test Workflow
Use Xano’s built-in API tester or Postman to validate routes and emotional logic.

---

## 🧰 Developer Notes

- Workspace is fully modular — customize logic flows using Xano Functions.
- Future support for WebSockets and real-time emotional state mapping planned.
- Analytics layer can be extended with Supabase or Dune for on-chain analytics.

---

## 📄 License

This project backend is proprietary and part of the SASOK ecosystem.

---

## 🤝 Contributing

If you’re working within the SASOK development network:
- Use `/api/emotional_milestone` to sync progress
- Use dev branches for model changes in `ai_model_version`

---

## ✉️ Contact

Product Owner: **Sheylock**  
Cognitive Systems Architect & Web3 AI Strategist  
📧 hello@sasok.xyz  
🌐 [saske.xyz](https://saske.xyz) | [evorin.io](https://evorin.io)

---

> "SASOK doesn't predict your emotions. It evolves with them."

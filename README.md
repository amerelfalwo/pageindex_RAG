---
title: Atlas Document Intelligence API
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🌿 Atlas — Document Intelligence API

> **Understand Documents. Discover Knowledge.**
> Production-grade Vectorless RAG platform with concurrent multi-document retrieval, isolated three-tier memory architecture, and resilient multi-provider LLM inference (Hugging Face + Groq zero-downtime failover).

---

## ✨ Overview

**Atlas** transforms document search by eliminating vector embeddings and approximate chunk retrieval in favor of deterministic, hierarchical tree search (`PageIndex`). Instead of blindly splitting text and calculating cosine distances, Atlas parses documents into semantic section trees preserving headings, subheadings, and physical page coordinates.

With the new **Multi-Document Intelligence & Memory System**, users can attach multiple documents to an isolated research session, synthesize cross-document findings, and maintain both short-term conversational context, rolling long-term memory, and workspace-wide collective knowledge dossiers.

---

## 🚀 Key Architecture & Capabilities

### 1. 🌳 Multi-Document Vectorless RAG
- **Concurrent Tree Navigation:** When querying across multiple attached documents, Atlas traverses each document tree simultaneously via asynchronous coroutines, identifying precise relevant sections without vector embedding drift.
- **Multi-Source Attribution:** Every retrieved evidence chunk is stamped with its source document name, section title, and page number, enabling cross-document comparison and grounded synthesis.

### 2. 🧠 Three-Tier Memory Architecture
- **Short-Term Session Memory:** Turn-by-turn conversational sliding window for seamless follow-up questions within the active session.
- **Long-Term Session Memory:** Automated rolling conversation summaries persisted in SQLite (`chat_summaries`), ensuring deep context retention without token bloat.
- **Workspace Collective Knowledge Memory:** Pre-computed knowledge dossiers for every indexed document (executive summaries, key entities, findings, topics) synthesized into a unified workspace knowledge base (`global_knowledge_summary`). Enables Atlas to answer questions about all workspace knowledge even when no files are specifically attached.

### 3. 🛡️ Strict Session Isolation
- Every chat session operates in complete state isolation.
- Attached documents and conversation histories are strictly scoped to the active `chat_id`—zero cross-session data bleed.

### 4. 🔄 Dual LLM Architecture with Groq Fallback
- Primary inference via Hugging Face Serverless (`meta-llama/Llama-3.1-8B-Instruct`).
- **Instant Automatic Failover to Groq API** (`qwen/qwen3.8-27b` with ultra-fast TTFT ~0.20s) whenever Hugging Face credits are exhausted (`402 Payment Required`) or rate limits are reached.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Framework** | FastAPI (Python 3.11, async runtime) |
| **Package Manager** | `uv` (Astral) |
| **Document Indexer** | `PageIndex` + `PyMuPDF` (fitz) fallback |
| **Primary LLM** | Hugging Face Serverless (`huggingface_hub.AsyncInferenceClient`) |
| **Fallback LLM** | Groq Cloud API (`openai.AsyncOpenAI` client) |
| **Database & Memory** | SQLite (`doc_meta`, `chats`, `chat_documents`, `chat_summaries`, `knowledge_memory`, `global_knowledge_summary`) |
| **Frontend** | React 19 + Vite + Tailwind CSS (Deployed on Vercel) |
| **Backend Deployment** | Docker container on Hugging Face Spaces |

---

## 🔌 API Endpoints Reference

### 1. Multi-Document Research & Chat
- `POST /ask` — Stream grounded research answer across single or multiple attached documents, or pure conversational mode with workspace global knowledge injection.
- `POST /chat/new` — Create a new isolated chat session.
- `GET /chats` — List all persisted chat sessions.
- `GET /chat/{chat_id}/history` — Retrieve turn-by-turn history for a session.
- `POST /chat/{chat_id}/attach` — Attach a single document to the session.
- `POST /chat/{chat_id}/attach-multiple` — Attach multiple documents simultaneously.
- `GET /chat/{chat_id}/documents` — List all documents attached to this session.
- `DELETE /chat/{chat_id}/document/{doc_id}` — Detach a specific document from the session.
- `DELETE /chat/{chat_id}/document` — Detach all documents from the session.
- `DELETE /chat/{chat_id}` — Permanently delete a chat session.
- `PATCH /chat/{chat_id}/title` — Rename a conversation session.

### 2. Knowledge & Memory
- `GET /memory/knowledge` — Retrieve workspace collective master knowledge summary and all per-document dossiers.
- `POST /memory/knowledge/refresh` — Synthesize and rebuild workspace global knowledge memory.
- `GET /memory/dossier/{doc_id}` — Fetch structured knowledge dossier for a specific document.
- `GET /memory/chat/{chat_id}` — Inspect short-term turns and long-term rolling summary for a session.
- `POST /memory/chat/{chat_id}/summarize` — Force-generate or update session long-term memory.

### 3. Document Library & Processing
- `POST /upload` — Upload a PDF file to trigger asynchronous hierarchical indexing (HTTP 202).
- `GET /documents` — List all ingested documents, page counts, sections, and statuses.
- `GET /documents/{document_id}` — Get metadata and status for a document.
- `GET /documents/{document_id}/tree` — Retrieve the complete hierarchical outline tree.
- `DELETE /documents/{document_id}` — Delete a document, its index, and files.
- `POST /documents/{document_id}/retry` — Retry indexing for a failed document.

---

## ⚙️ Environment Variables

Configure these variables locally in `.env` or in **Hugging Face Space Settings → Variables and secrets**:

```env
# Hugging Face Inference Token
HF_TOKEN=your_huggingface_token

# PageIndex Ingestion API Keys
PAGEINDEX_API_KEY_1=your_pageindex_key_1
PAGEINDEX_API_KEY_2=your_pageindex_key_2
PAGEINDEX_API_KEY_3=your_pageindex_key_3

# Groq API Fallback (Triggered automatically on HF 402 or exhaustion)
GROQ_API_KEY=your_groq_api_key
GROQ_CHAT_MODEL=qwen/qwen3.8-27b
GROQ_RETRIEVAL_MODEL=qwen/qwen3.8-27b

# Model Customization (Optional)
HF_RETRIEVAL_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_CHAT_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

---

## 🚀 Live Deployments

- **Frontend (Vercel):** [https://vectorless-front-8gpg.vercel.app](https://vectorless-front-8gpg.vercel.app)
- **Backend (Hugging Face Spaces):** [https://amer003100-vectorless-rag.hf.space](https://amer003100-vectorless-rag.hf.space)
- **Swagger Documentation:** [https://amer003100-vectorless-rag.hf.space/docs](https://amer003100-vectorless-rag.hf.space/docs)

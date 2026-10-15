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
> Production-grade Vectorless RAG backend powered by tree-based document retrieval, deterministic evidence selection, and resilient multi-provider LLM inference.

---

## ✨ Overview

**Atlas** replaces conventional vector embeddings with deterministic, hierarchical tree search (`PageIndex`). Instead of slicing text into blind chunks and computing approximate vector distances, Atlas constructs an exact outline tree of headings, sections, and page coordinates. When a user asks a question, Atlas navigates the document hierarchy to locate and extract verified textual and multimodal evidence with zero hallucinations.

---

## 🚀 Key Features

- **🌳 Vectorless Hierarchical Retrieval:**
  Builds deterministic section trees using `PageIndex` with an automatic local `PyMuPDF` fallback if external indexing services are unavailable.
- **🔄 Dual LLM Architecture with Groq Fallback:**
  Primary inference via Hugging Face Serverless (`Llama 3.1 8B` for retrieval + `Qwen 2.5 72B` for synthesis) with **automatic zero-downtime failover to Groq API** (`qwen/qwen3.8-27b`) when Hugging Face tokens or credits are depleted (`402 Payment Required`).
- **🌐 Bilingual Arabic & English Grounding:**
  Deep semantic understanding in Arabic and English, strictly enforcing that factual claims cite section titles and physical page numbers.
- **⚡ Asynchronous Ingestion & Real-Time Polling:**
  Non-blocking background document processing with staged progress tracking (`uploading` → `indexing` → `validating` → `ready`).
- **📚 Dedicated Document Library API:**
  Full CRUD endpoints for persistent document management, metadata storage, and section-tree retrieval.
- **🖼️ Multimodal Extraction:**
  Extracts figures and images from documents into a static endpoint with contextual inline markdown citations.
- **🌊 Low-Latency Token Streaming:**
  FastAPI asynchronous generators streaming tokens directly to the client with live retrieval stage markers (`<!--STAGE:...-->`).

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Framework** | FastAPI (Python 3.11, async runtime) |
| **Package Manager** | `uv` (Astral) |
| **Document Indexer** | `PageIndex` + `PyMuPDF` (fitz) |
| **Primary LLM** | Hugging Face Serverless (`huggingface_hub.AsyncInferenceClient`) |
| **Fallback LLM** | Groq Cloud API (`openai.AsyncOpenAI` client) |
| **Database & Cache** | SQLite (`doc_meta`, `chat_history`) + Dual-Key Disk Cache (`.cache/`) |
| **Deployment** | Docker container on Hugging Face Spaces |

---

## 🔌 API Endpoints Reference

### 1. Document Management
- `POST /upload` — Upload a PDF file to trigger asynchronous hierarchical indexing.
- `GET /documents` — List all ingested documents, page counts, sections, and statuses.
- `GET /documents/{document_id}` — Get detailed metadata and status for a document.
- `GET /documents/{document_id}/tree` — Retrieve the complete hierarchical outline tree.
- `DELETE /documents/{document_id}` — Delete a document, its cached index, and stored file.
- `POST /documents/{document_id}/retry` — Retry indexing for a failed document.

### 2. Research & Chat
- `POST /ask` — Stream response for a research query or conversational question.
- `POST /chat/{chat_id}/attach` — Bind an indexed document to an active conversation session.
- `DELETE /chat/{chat_id}/document` — Detach the active document from the session.

### 3. System
- `GET /health` — Service health check.
- `GET /static/{filename}` — Serve extracted document images and diagrams.

---

## ⚙️ Environment Variables

Configure these variables locally in a `.env` file or in the **Hugging Face Space Settings → Variables and secrets**:

```env
# Hugging Face Inference Token
HF_TOKEN=your_huggingface_token

# PageIndex Ingestion API Keys (Supports round-robin failover)
PAGEINDEX_API_KEY_1=your_pageindex_key_1
PAGEINDEX_API_KEY_2=your_pageindex_key_2
PAGEINDEX_API_KEY_3=your_pageindex_key_3

# Groq API Fallback (Activated automatically when HF tokens/credits run out)
GROQ_API_KEY=your_groq_api_key
GROQ_CHAT_MODEL=qwen/qwen3.8-27b
GROQ_RETRIEVAL_MODEL=qwen/qwen3.8-27b

# Model Customization (Optional)
HF_RETRIEVAL_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_CHAT_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

---

## 💻 Local Development

```bash
# Clone the repository
git clone https://github.com/amerelfalwo/pageindex_RAG.git
cd pageindex_RAG

# Install dependencies using uv
uv sync

# Run the API server
uv run uvicorn app.app:app --host 0.0.0.0 --port 7860 --reload
```

---

## 🧪 Testing

```bash
# Run the complete production asynchronous workflow verification suite
uv run python test_production_async_workflow.py
```

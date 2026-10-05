---
title: Vectorless RAG API
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 📚 Vectorless RAG System

An English-first, production-ready Document Analysis Assistant that uses a hierarchical tree search (not vector embeddings) for fast, grounded answers. It also supports multimodal RAG by extracting images from PDFs, serving them via a static endpoint, and citing them in responses when relevant.

## ✨ Features

- **Vectorless Retrieval:** Uses `PageIndex` to build a hierarchical tree of the document.
- **Context-Aware Chat:** Maintains conversation history with rolling summarization.
- **Streaming Responses:** Real-time typewriter effect using FastAPI and Hugging Face streams.
- **No Hallucinations:** Strict system prompts to ensure answers are grounded in the document.
- **Multimodal RAG:** Extracts images from PDFs, stores them in `/static`, and injects image URLs into context for citation and display.

## 🛠️ Tech Stack

- **Backend:** FastAPI, Python, `uv` package manager.
- **Frontend:** React, Vite.
- **AI Models:** Hugging Face Serverless Inference API (Llama 3.1 8B for retrieval + Qwen 2.5 72B for chat).
- **Deployment:** Docker, Hugging Face Spaces.

## ⚙️ Environment Variables

Create a `.env` file in the root directory (For local testing only. On Hugging Face, add these in Settings -> Secrets):

```env
HF_TOKEN=your_huggingface_token_here
PAGEINDEX_API_KEY_1=your_pageindex_api_key_here
```

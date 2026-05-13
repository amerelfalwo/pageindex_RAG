---
title: Vectorless RAG API
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📚 Vectorless RAG System

A smart Document Analysis Assistant that utilizes hierarchical tree search instead of traditional vector embeddings for highly accurate, context-aware answers.

## ✨ Features
- **Vectorless Retrieval:** Uses `PageIndex` to build a hierarchical tree of the document.
- **Context-Aware Chat:** Maintains conversation history with rolling summarization.
- **Streaming Responses:** Real-time typewriter effect using FastAPI and OpenAI streams.
- **No Hallucinations:** Strict system prompts to ensure answers are grounded in the document.

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python, `uv` package manager.
- **Frontend:** React, Vite.
- **AI Models:** OpenAI (GPT-4o / GPT-5-mini) via GitHub Models.
- **Deployment:** Docker, Hugging Face Spaces.

## ⚙️ Environment Variables
Create a `.env` file in the root directory (For local testing only. On Hugging Face, add these in Settings -> Secrets):
```env
OPENAI_API_KEY=your_openai_or_github_token_here
PAGEINDEX_API_KEY=your_pageindex_api_key_here
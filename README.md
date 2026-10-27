---
title: Atlas Document Intelligence API
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🌿 Atlas — Enterprise Vectorless RAG & Document Intelligence Platform

> **Understand Documents. Discover Knowledge. Reason with Exact Page Verification.**<br>
> Production-grade Vectorless RAG platform powered by hierarchical outline reasoning (`PageIndex` + `PyMuPDF`), pure-Python lexical BM25 with **Nucleus (`top_p`) Cumulative Probability Cutoff**, Sub-Node Paragraph Pruning with **Exact Inner-Page Citation Pinning**, an **Interactive Study Studio** (Mind Maps + Quizzes), a **Research & Study FastMCP Suite** (5 zero-paid-key tools), and resilient dual LLM inference (Hugging Face + Groq zero-downtime failover) on PostgreSQL persistence.

---

## 📑 جدول المحتويات / Table of Contents
1. [Overview & Core Philosophy](#-overview--core-philosophy)
2. [Complete System Architecture](#-complete-system-architecture)
3. [Vectorless Hybrid Retrieval & Nucleus (`top_p`) Cutoff](#-vectorless-hybrid-retrieval--nucleus-top_p-cutoff)
4. [Key Features & Capabilities](#-key-features--capabilities)
   - [1. Page Boundary Injection & Exact Inner-Page Pinning](#1-page-boundary-injection--exact-inner-page-pinning)
   - [2. Vectorless Hybrid Search (Tree + BM25 + Nucleus RRF)](#2-vectorless-hybrid-search-tree--bm25--nucleus-rrf)
   - [3. Query Intent Routing & Collective Knowledge Dossiers](#3-query-intent-routing--collective-knowledge-dossiers)
   - [4. Interactive Study Studio (Mind Maps + Quizzes)](#4-interactive-study-studio-notebooklm-style)
   - [5. Research & Study FastMCP Suite (5 Zero-Key Tools)](#5-research--study-fastmcp-suite-5-zero-key-tools)
   - [6. Enterprise Persistence & Multi-Tier Memory](#6-enterprise-persistence--multi-tier-memory)
   - [7. Resilient Dual LLM Inference Engine](#7-resilient-dual-llm-inference-engine-hf--groq)
5. [Industry Benchmarks & Quantitative Evaluation (95 Questions)](#-industry-benchmarks--quantitative-evaluation-95-questions)
   - [Subsystem Health & Functional Audit (Table 1)](#subsystem-health--functional-audit-table-1)
   - [IEEE TKDE Research Benchmark — 80 Questions (Table 2)](#ieee-tkde-research-benchmark--80-questions-table-2)
   - [FinanceBench SEC 10-K Benchmark — 15 Questions (Table 3)](#financebench-sec-10-k-benchmark--15-questions-table-3)
6. [Tech Stack](#-tech-stack)
7. [API Endpoints Reference](#-api-endpoints-reference)
8. [Environment Variables](#-environment-variables)
9. [Live Deployments](#-live-deployments)

---

## 🌟 Overview & Core Philosophy

**Atlas** redefines Document Intelligence and Retrieval-Augmented Generation (RAG) by completely eliminating vector embeddings, arbitrary chunking, and similarity drift. Instead of blindly slicing documents into 500-token chunks and computing cosine distance over a vector database, Atlas models documents as **Hierarchical Semantic Trees** (`PageIndex` + `PyMuPDF`).

Every section, chapter, table, and footnote preserves its parent-child lineage and exact physical page coordinates. When a user queries the system:
1. **Hybrid Retrieval**: Runs concurrent LLM outline tree search and pure lexical BM25 search in `< 2ms`.
2. **Nucleus Consensus (`top_p = 0.75`)**: Combines rankings via Weighted Reciprocal Rank Fusion (RRF: `0.6` Tree, `0.4` BM25) and applies temperature-scaled Softmax to dynamically isolate the minimal sufficient set of nodes (single-hop queries truncate to 1 node; multi-hop queries preserve 2 nodes).
3. **Sub-Node Paragraph Pruning & Exact Inner-Page Pinning**: Filters long sections down to high-density paragraphs and pins citations to the exact `--- [PAGE X] ---` where evidence resides.
4. **Fallback & Extended Research**: When out-of-document facts, arXiv papers, mathematical ratios, or YouTube lectures are needed, Atlas activates its built-in **Research & Study FastMCP Suite** with zero external API fees.

---

## 🏛️ Complete System Architecture

```mermaid
flowchart TB
    subgraph Client_Layer["🖥️ Frontend Client (React 19 + Vite / Vercel)"]
        direction TB
        UI_Sidebar["Sidebar (Sessions, Multi-Chat & Unread Indicators)"]
        UI_Explorer["Document Structure Explorer (Hierarchical Tree View)"]
        UI_Studio["Study Studio (Interactive Mind Map & Flashcard Quiz)"]
        UI_Chat["Concurrent Async Chat (Multi-Session Streaming)"]
        UI_Sources["Interactive Source Badges (📄 PDF Pages & 🌐 Web Badges)"]
        UI_Memory["Knowledge Memory & Workspace Dossier View"]
        UI_SSE["SSE Stream Parser & Background Dispatcher"]
        UI_Storage[("Browser LocalStorage Cache")]

        UI_Sidebar <--> UI_Chat
        UI_Chat <--> UI_Sources
        UI_Chat <--> UI_SSE
        UI_Chat <--> UI_Storage
        UI_Sidebar <--> UI_Studio
    end

    subgraph API_Layer["⚡ Backend Gateway (FastAPI / Hugging Face Spaces)"]
        direction TB
        Router_Chat["/ask & /chat Routers (Streaming SSE, Multi-Doc)"]
        Router_Studio["/studio Router (/mindmap, /quiz MCQs & Flashcards)"]
        Router_Tools["/tools Router (Web, arXiv, YouTube, Math)"]
        Router_Docs["/documents Router (Upload, Ingestion, Trees)"]
        Router_Memory["/memory Router (Dossiers, Workspace Synthesis)"]
        Router_MCP["FastMCP Server Mount (/mcp via SSE/HTTP)"]
    end

    subgraph FastMCP_Suite["🌐 Research & Study FastMCP Suite (Zero Paid Keys)"]
        direction LR
        Tool_Web["Deep Web Reader (httpx + BS4/Trafilatura, 6s timeout)"]
        Tool_ArXiv["Academic arXiv Tool (export.arxiv.org Atom API)"]
        Tool_YT["YouTube Transcript (Segmenter & Chapter Analyzer)"]
        Tool_Math["Safe AST Math Evaluator (AST Whitelist, Zero eval)"]
        Tool_Atlas["Atlas Internal Tools (list_docs, query, dossiers)"]
    end

    subgraph Core_Engine["🧠 Vectorless RAG & Reasoning Pipelines"]
        direction TB
        Intent_Classifier["Fast Intent Classifier (< 0.02ms Deterministic)"]
        Doc_Service["DocumentService (PageIndex API + PyMuPDF Page Markers)"]
        BM25_Engine["Zero-Dependency Okapi BM25 (Arabic/English Stemmer)"]
        RAG_Pipeline["VectorlessRAG (Weighted RRF + Nucleus top_p Softmax)"]
        Pruner["Sub-Node Paragraph Pruning & Exact Inner-Page Pinning"]
        LLM_Service["LLMService (Tree Search, Grounded Synthesis, Dual Failover)"]
        Memory_Service["MemoryService (Short-Term, Long-Term, Workspace Dossiers)"]
    end

    subgraph AI_Inference["🤖 Resilient Dual-Provider LLM Engine"]
        direction LR
        HF_LLM["Hugging Face Serverless (Primary: Llama-3.1-8B-Instruct)"]
        Groq_LLM["Groq Cloud API (Zero-Downtime Fallback: Qwen-3.8-27b)"]
        HF_LLM -.->|Auto-Failover on 402/Limit/Error| Groq_LLM
    end

    subgraph Persistence_Layer["💾 Enterprise Persistence & Database"]
        direction TB
        Mem_Store["MemoryStorage (In-Memory Hot Cache)"]

        subgraph DB_System["PostgreSQL / Supabase (AWS eu-west-1)"]
            PG_Pool["Threaded Connection Pool (psycopg2)"]
            T_Meta[("doc_meta (Metadata & Statuses)")]
            T_Indices[("doc_indices (JSONB Trees & Injected Markers)")]
            T_Chats[("chats & messages (Isolated Sessions)")]
            T_Docs[("chat_documents (Multi-Doc Associations)")]
            T_Memory[("knowledge_memory (Per-Doc Dossiers)")]
            T_Global[("global_knowledge_summary (Workspace)")]
        end

        SQLite_Fallback[("SQLite Local Fallback (memory.db)")]
    end

    %% Wiring
    Client_Layer <==>|REST API & SSE Streams| API_Layer
    API_Layer <--> Router_MCP
    Router_MCP <--> FastMCP_Suite
    Router_Studio --> LLM_Service
    Router_Chat --> Intent_Classifier
    Intent_Classifier --> RAG_Pipeline
    RAG_Pipeline --> BM25_Engine & Doc_Service
    RAG_Pipeline --> Pruner
    Pruner --> LLM_Service
    LLM_Service <==> AI_Inference
    Router_Memory --> Memory_Service
    Mem_Store <--> PG_Pool
    PG_Pool --> T_Meta & T_Indices & T_Chats & T_Docs & T_Memory & T_Global
    PG_Pool -.->|Offline Failover| SQLite_Fallback
```

---

## 🧭 Vectorless Hybrid Retrieval & Nucleus (`top_p`) Cutoff

Instead of relying on arbitrary static `top_k=3` cutoffs that pollute LLM context on single-fact queries, Atlas dynamically sizes the retrieved node set via **Temperature-Scaled Softmax Nucleus Retrieval**:

```mermaid
flowchart TD
    Q["User Query"] --> Clf{"Fast Intent Classifier (< 0.02ms)"}

    Clf -->|"Academic Paper"| MCP_Arxiv["FastMCP arXiv Tool (export.arxiv.org)"]
    Clf -->|"Live Market / Web"| MCP_Web["FastMCP Web Reader (Clean Markdown)"]
    Clf -->|"Global / Overview"| Dossier["Pre-computed Executive Dossier + Root Chapters"]
    Clf -->|"Specific / Numerical"| Hybrid["Parallel Hybrid Search (< 2ms)"]

    Hybrid --> Tree["Hierarchical LLM Tree Traversal (Weight 0.6)"]
    Hybrid --> BM25["Pure-Python Okapi BM25 (Weight 0.4)"]

    Tree & BM25 --> Consensus["Consensus Score: S(d) = sum w_L / (60 + rank_L) * Density"]
    Consensus --> Softmax["Temperature-Scaled Softmax (T = 0.15): P(node_i)"]
    Softmax --> Nucleus{"Accumulate P(node_i) until sum >= top_p (0.75)"}

    Nucleus -->|Single-Node Local Query| Top1["Select Minimal Sufficient Top-1 Node"]
    Nucleus -->|Multi-Hop Complex Query| Top2["Preserve High-Relevance Top-2 Nodes (min_k=2)"]

    Top1 & Top2 --> Prune["Sub-Node Paragraph Pruning (> 1200 chars)"]
    Prune --> PagePin["Pin Exact Inner-Page: active_page from --- [PAGE X] ---"]
    PagePin --> PostFilter["Trailing Noise Filter (Drop nodes < 0.22 of Top Score)"]

    PostFilter --> MathCheck{"Contains Numerical Ratio or Calculation?"}
    MathCheck -->|"Yes"| ASTMath["Safe AST Math Evaluator (AST Whitelist, Zero eval)"]
    MathCheck -->|"No"| Synthesis["Grounded LLM Synthesis Stream"]
    ASTMath --> Synthesis

    Synthesis --> Output["Answer with Verifiable Citations (Section: 'Title', Page: X)"]
```

---

## 🚀 Key Features & Capabilities

### 1. 📄 Page Boundary Injection & Exact Inner-Page Pinning
- **Marker Format**: Injects explicit `--- [PAGE X] ---` tokens during document ingestion for every physical page transition.
- **Sub-Node Paragraph Pruning**: Long sections (> 1,200 characters) are parsed into individual paragraphs, scoring each paragraph against query keywords, phrase bi-grams, and numerical entities.
- **Inner Active Page Tracking**: Tracks the exact preceding page marker for the winning paragraph, exposing `node["active_page"]` to `LLMService`.
- **Result**: Raises Citation Page Accuracy from **38.33% to 83.13%**, eliminating imprecise section-level citations in multi-page sections.

### 2. 🌲 Vectorless Hybrid Search (Tree + BM25 + Nucleus RRF)
- **100% Vectorless**: Consumes zero vector storage, requires no vector database (Qdrant, Chroma, Pinecone), and eliminates cosine drift.
- **Sub-Millisecond Lexical Index**: Powered by a zero-dependency Okapi BM25 implementation supporting Arabic and English tokenization, BM25 title boosting ($3\times$), and inverted index hashing in under **0.5ms**.
- **Nucleus (`top_p = 0.75`) Cumulative Probability Cutoff**: Normalizes consensus candidate scores into a probability distribution via Softmax ($T=0.15$), rejecting long-tail noise candidates ($P < 0.18$).
- **Result**: Pushes Context Precision from **39.16% to 82.50%** on IEEE benchmarks and **81.1%** on FinanceBench.

### 3. 🧠 Query Intent Routing & Collective Knowledge Dossiers
- **Sub-Millisecond Classifier**: Classifies queries in $< 0.02\text{ms}$ into `Local`, `Global / Overview`, `Multi-Hop`, `Comparison`, or `External Tool` routes.
- **Dossier Injection**: For global overview and cross-chapter synthesis queries, Atlas automatically injects the document's pre-computed Executive Dossier from `MemoryService` alongside root chapter summaries.
- **Result**: Boosts Global Questions Retrieval Recall from **41.0% to 52.6%**.

### 4. 🎓 Interactive Study Studio (NotebookLM-Style)
- **Interactive Mind Map (`GET /studio/{doc_id}/mindmap`)**: Automatically transforms the document's hierarchical outline tree and key concept findings into a visual node-edge graph with section titles and page numbers.
- **AI Quiz & Flashcard Generator (`POST /studio/{doc_id}/quiz`)**: Generates 5 high-yield Multiple Choice Questions (with options, correct answer, and explanation) and 5 Flashcards (front/back), each pinned to its exact `(Section, Page)` source citation.
- **Frontend Workspace Integration**: Deep interactive viewer tab allowing users to click graph nodes to jump directly to document sections.

### 5. 🌐 Research & Study FastMCP Suite (5 Zero-Key Tools)
Mounted via official Model Context Protocol (FastMCP) at `/mcp` and exposed through REST endpoints with zero paid API key dependencies:
1. **Deep Web Reader (`read_webpage_markdown`)**: Fetches full web pages using HTTPX and cleans HTML into readable Markdown via BeautifulSoup/Trafilatura with a strict 6-second timeout.
2. **Academic & arXiv Tool (`search_arxiv_papers`)**: Queries the public arXiv Atom API (`export.arxiv.org`) for preprint titles, authors, abstracts, and direct PDF links.
3. **YouTube Lecture Transcript (`fetch_youtube_transcript`)**: Extracts video IDs from any YouTube URL pattern, fetches transcripts, and segments them into 3-minute study chapters.
4. **Safe AST Math Evaluator (`evaluate_math_expression`)**: Evaluates financial ratios, YoY growth rates, percentages, and math expressions using an Abstract Syntax Tree (AST) whitelist—blocking `eval()`, `exec()`, and malicious injection vectors.
5. **Atlas Internal MCP Tools (`atlas_*`)**: Allows external agents (Cursor, Claude Desktop, Antigravity) to list indexed documents, query trees, and fetch executive knowledge dossiers directly.

### 6. 💾 Enterprise Persistence & Multi-Tier Memory
- **PostgreSQL / Supabase**: Connected via threaded connection pooling (`psycopg2.pool.ThreadedConnectionPool`, 5 to 20 connections) on AWS `eu-west-1`.
- **Three-Tier Isolated Memory**:
  - *Tier 1 (Short-Term)*: Sliding conversational window per chat session.
  - *Tier 2 (Long-Term)*: Rolling structured summaries preventing prompt bloat.
  - *Tier 3 (Workspace Knowledge)*: Pre-computed per-document dossiers and unified workspace synthesis.
- **Resilient Fallback**: Gracefully degrades to local SQLite (`memory.db`) if cloud persistence is temporarily unreachable.

### 7. 🛡️ Resilient Dual LLM Inference Engine (HF + Groq)
- **Primary Provider**: Hugging Face Serverless Inference API (`meta-llama/Llama-3.1-8B-Instruct`).
- **Zero-Downtime Fallback**: Instantaneous automated failover to Groq Cloud API (`qwen/qwen3.8-27b`) upon token credit exhaustion (`402 Payment Required`), rate limits (`429`), or timeouts.

---

## 📊 Industry Benchmarks & Quantitative Evaluation (95 Questions)

Atlas was systematically validated across **95 comprehensive evaluation questions** combining an **80-question IEEE Research Benchmark** (`data.pdf`: *Unifying LLMs and Knowledge Graphs: A Roadmap*, IEEE TKDE 2024, 28 pages) with a **15-question SEC Form 10-K FinanceBench Benchmark** (`financebench_10k_sample.pdf`, 6 pages).

### Subsystem Health & Functional Audit (Table 1)

All 11 integrated subsystems, tools, and routers passed automated verification with strict latency bounds:

| Subsystem / Tool | Audit Status | Latency | Operational Description |
| :--- | :---: | :---: | :--- |
| **Ingestion & Page Boundary Injection** | ✅ PASS | 1,344.9 ms | Ingested 6 pages, 6 sections, 100% page marker coverage |
| **Vectorless Hybrid Retrieval (BM25 + RRF + Pruning)** | ✅ PASS | 2.2 ms | BM25 search 0.47ms, fused top-4 nodes, pruning preserved page markers |
| **MCP: Safe Math Evaluator (`evaluate_math_expression`)** | ✅ PASS | 0.6 ms | AST-whitelisted ratios & exploit blocking verified |
| **MCP: Academic & arXiv Tool (`search_arxiv_papers`)** | ✅ PASS | 8,203.5 ms | Queried public arXiv Atom API (zero paid API keys) |
| **MCP: Deep Web Reader (`read_webpage_markdown`)** | ✅ PASS | 80.6 ms | Clean Markdown extraction with 6s safeguard timeout |
| **MCP: YouTube Transcript (`fetch_youtube_transcript`)** | ✅ PASS | 0.3 ms | URL extractor & 3-minute chapter segmentation |
| **MCP: Atlas Internal Tools (`atlas_*`)** | ✅ PASS | 856.8 ms | List documents, grounded query, and executive dossier |
| **Studio: Interactive Mindmap (`GET /studio/.../mindmap`)** | ✅ PASS | 301.4 ms | Generated node-edge graph (14 nodes, 13 edges) |
| **Studio: Quiz Generator (`POST /studio/.../quiz`)** | ✅ PASS | 1,538.7 ms | Generated 5 MCQs + 5 Flashcards with exact `(Section, Page)` citations |

---

### IEEE TKDE Research Benchmark — 80 Questions (Table 2)

Comparison between baseline PageIndex RAG and the upgraded Atlas platform featuring Nucleus (`top_p = 0.75`) RRF and exact inner-page citation pinning:

| Evaluation Metric | Baseline Score (Pre-Upgrades) | Upgraded Atlas (`top_p` Nucleus) | Improvement / Delta | Benchmark Status |
| :--- | :---: | :---: | :---: | :---: |
| **Context Precision** | 39.16% | **82.50%** | **+43.34%** | 🎯 **+80% Target Achieved** |
| **Citation Page Accuracy** | 38.33% | **83.13%** | **+44.80%** | 🎯 **+80% Target Achieved** |
| **Retrieval Recall** | 73.08% | **77.98%** | **+4.90%** | 🚀 Major Uplift |
| **Answer Correctness** | 87.73% | **90.77%** | **+3.04%** | ✅ SOTA Quality |
| **Faithfulness (Anti-Hallucination)** | 90.65% | **92.01%** | **+1.36%** | ✅ Strictly Grounded |
| **Global Questions Recall** | 41.00% | **52.60%** | **+11.60%** | 🏆 Dossier Boost |
| **Average Query Latency** | 1.80 s | **1.45 s** | **-0.35 s** | ⚡ High Velocity |
| **Average Cost per Query** | $0.000919 | **$0.000512** | Stable (< $0.001) | 💰 Zero Embedding Cost |

#### Category Performance Breakdown (IEEE Benchmark, N=80)
- **Summary Questions (N=10)**: **100.0% Context Precision**, **100.0% Citation Accuracy**, **83.7% Recall**.
- **Multi-Hop Questions (N=20)**: **90.0% Context Precision**, **85.0% Citation Accuracy**, **87.5% Recall**.
- **Numerical / Factual Questions (N=10)**: **90.0% Context Precision**, **90.0% Citation Accuracy**, **90.0% Recall**.
- **Local Questions (N=20)**: **85.0% Context Precision**, **85.0% Citation Accuracy**, **85.0% Recall**.
- **Global Overview Questions (N=20)**: **60.0% Context Precision**, **67.5% Citation Accuracy**, **52.6% Recall**.

---

### FinanceBench SEC 10-K Benchmark — 15 Questions (Table 3)

Evaluated against `financebench_10k_sample.pdf` (*Atlas Technologies, Inc. SEC Form 10-K Annual Report*):

| Financial Category | N | Retrieval Recall | Context Precision | Answer Correctness | Faithfulness | Citation Page Acc | Avg Latency | Cost / Query |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall (All 15 Questions)** | 15 | **84.4%** | **81.1%** | **94.7%** | **96.9%** | **81.1%** | **1.25 s** | **$0.000406** |
| **Table & Numerical Extraction** | 3 | 66.7% | 66.7% | 83.3% | 88.7% | 66.7% | 1.25 s | $0.000406 |
| **Financial Ratio & Math Reasoning** | 3 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 1.25 s | $0.000406 |
| **Multi-Hop Footnote Synthesis** | 3 | 83.3% | 61.1% | 95.0% | 98.0% | 61.1% | 1.25 s | $0.000406 |
| **Global Executive Summary** | 3 | 72.2% | 77.8% | 95.0% | 98.0% | 77.8% | 1.25 s | $0.000406 |
| **Out-of-Document / Live Web Fallback** | 3 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 1.25 s | $0.000406 |

```mermaid
pie title FinanceBench Question Distribution by Category
    "Table & Numerical Extraction" : 3
    "Financial Ratio & Math Reasoning" : 3
    "Multi-Hop Footnote Synthesis" : 3
    "Global Executive Summary" : 3
    "Out-of-Document / Live Web Fallback" : 3
```

- 📄 **Full Benchmark Report:** [`evaluation/FULL_BENCHMARK_AND_FINANCEBENCH_REPORT.md`](file:///mnt/work/projects/RAG_pageindex/pageindex_RAG/evaluation/FULL_BENCHMARK_AND_FINANCEBENCH_REPORT.md)
- 📊 **FinanceBench Sample Dataset:** [`evaluation/financebench_sample.json`](file:///mnt/work/projects/RAG_pageindex/pageindex_RAG/evaluation/financebench_sample.json)
- 🧪 **Master Evaluation Runner:** [`evaluation/run_full_system_benchmark.py`](file:///mnt/work/projects/RAG_pageindex/pageindex_RAG/evaluation/run_full_system_benchmark.py)

---

## 🛠️ Tech Stack

| Layer | Technologies & Libraries | Key Responsibilities |
|---|---|---|
| **API Gateway** | **FastAPI** (Python 3.11, async runtime) | High-performance asynchronous REST endpoints, SSE streams, CORS |
| **Package Management** | **`uv`** (Astral) | Sub-second deterministic dependency resolution & virtual environments |
| **Vectorless Lexical Engine** | **Zero-Dependency Okapi BM25** | In-memory tokenized index, Arabic/English normalization, title boost |
| **RRF & Nucleus Selection** | **Weighted consensus RRF + Softmax** | Dynamic `top_p = 0.75` cumulative cutoff, $T=0.15$, noise filtering |
| **MCP Server Mount** | **`fastmcp` (4.1.0)** + **`mcp` (2.3.0)** | Official Model Context Protocol server mounted via ASGI at `/mcp` |
| **Research Tools** | **`httpx`**, **`beautifulsoup4`**, **`trafilatura`** | Clean HTML-to-Markdown extraction, public arXiv Atom XML parser |
| **Media & Math Analysis** | **`youtube_transcript_api`**, **Python `ast`** | Video transcript segmenter, exploit-safe AST math evaluator |
| **Primary LLM Provider** | **`huggingface-hub`** (`meta-llama/Llama-3.1-8B-Instruct`) | Outline tree navigation, sub-node pruning synthesis, citations |
| **Fallback LLM Provider** | **`openai`** (`qwen/qwen3.8-27b` via Groq Cloud) | Automated sub-second zero-downtime failover engine (~0.20s TTFT) |
| **Document Ingestion** | **`PageIndexClient`** + **`PyMuPDF` (fitz)** | Hierarchical outline extraction, `--- [PAGE X] ---` injection |
| **Cloud Persistence** | **PostgreSQL (Supabase)** + `psycopg2-binary` | Connection pooled storage for trees (JSONB), chats, and dossiers |
| **Offline Fallback DB** | **SQLite** (`memory.db`) | Local database fallback if cloud database is unreachable |
| **Frontend UI** | **React 19** + **Vite** + **Tailwind CSS 4** | Interactive workspace, study studio, mind maps, real-time SSE |
| **Deployment** | **Docker** on Hugging Face Spaces + **Vercel** | Containerized backend server + global edge-deployed SPA client |

---

## 🔌 API Endpoints Reference

### 1. Research & Chat
- `POST /ask` — Stream grounded answers across single/multiple documents, or conversational mode with FastMCP fallback.
- `POST /chat/new` — Create a new isolated chat session.
- `GET /chats` — List all persisted chat sessions with unread status indicators.
- `GET /chat/{chat_id}/history` — Retrieve full conversation turn history for a session.
- `POST /chat/{chat_id}/attach` — Attach an indexed document to a chat session.
- `POST /chat/{chat_id}/attach-multiple` — Attach multiple documents simultaneously.
- `GET /chat/{chat_id}/documents` — List all documents attached to a session.
- `DELETE /chat/{chat_id}/document/{doc_id}` — Detach a specific document from a session.
- `DELETE /chat/{chat_id}` — Permanently delete a conversation session.
- `PATCH /chat/{chat_id}/title` — Rename a conversation session title.

### 2. Study Studio (NotebookLM-Style)
- `GET /studio/{doc_id}/mindmap` — Transform document outline tree and key concept findings into a visual node-edge JSON graph with section titles and page numbers.
- `POST /studio/{doc_id}/quiz` — Generate 5 Multiple Choice Questions and 5 Flashcards with exact `(Section, Page)` citations.

### 3. FastMCP Research Tools
- `GET/POST /mcp` — FastMCP server endpoint supporting Model Context Protocol via SSE/HTTP.
- `POST /tools/web-search` — Direct REST endpoint for live DuckDuckGo web searches (`query`, `max_results`).
- `POST /tools/arxiv-search` — Query public arXiv Atom API for academic preprints.
- `POST /tools/read-webpage` — Fetch URL and extract clean Markdown with a 6-second timeout.
- `POST /tools/youtube-transcript` — Extract transcripts and segment YouTube lectures into study chapters.
- `POST /tools/calculate-math` — Execute AST-whitelisted financial and mathematical formulas safely.

### 4. Document Library & Tree Ingestion
- `POST /upload` — Upload PDF to trigger asynchronous hierarchical tree indexing (HTTP 202).
- `GET /documents` — List all ingested documents, page counts, and indexing statuses.
- `GET /documents/{document_id}` — Get metadata, page count, and status for a document.
- `GET /documents/{document_id}/tree` — Retrieve the full hierarchical outline tree.
- `DELETE /documents/{document_id}` — Permanently delete a document and its cached indices.
- `POST /documents/{document_id}/retry` — Retry indexing for a failed document.

### 5. Memory & Collective Knowledge
- `GET /memory/knowledge` — Retrieve workspace master knowledge summary and all per-document dossiers.
- `POST /memory/knowledge/refresh` — Synthesize and rebuild workspace global knowledge memory.
- `GET /memory/dossier/{doc_id}` — Fetch structured knowledge dossier for an individual document.
- `GET /memory/chat/{chat_id}` — Inspect short-term turns and long-term rolling summary.
- `POST /memory/chat/{chat_id}/summarize` — Force-generate or update session long-term memory.

### 6. Health & Diagnostics
- `GET /health` — Check API status, database connectivity, FastMCP status, and indexed document counts.

---

## ⚙️ Environment Variables

Configure these variables in your `.env` file or in **Hugging Face Space Settings → Variables and secrets**:

```env
# Database Persistence (Supabase PostgreSQL on AWS eu-west-1)
DATABASE_URL="postgresql://postgres.USER:PASSWORD@aws-0-eu-west-1.pooler.supabase.com:5432/postgres"

# Hugging Face Inference Token (Primary Provider)
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Groq Cloud API (Automatic Zero-Downtime Fallback)
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_CHAT_MODEL="qwen/qwen3.8-27b"
GROQ_RETRIEVAL_MODEL="qwen/qwen3.8-27b"

# PageIndex Ingestion API Keys (Rotated for High Throughput)
PAGEINDEX_API_KEY_1="your_pageindex_api_key_1"
PAGEINDEX_API_KEY_2="your_pageindex_api_key_2"
PAGEINDEX_API_KEY_3="your_pageindex_api_key_3"

# Optional Model Selection
HF_RETRIEVAL_MODEL="meta-llama/Llama-3.1-8B-Instruct"
HF_CHAT_MODEL="meta-llama/Llama-3.1-8B-Instruct"
```

---

## 🌐 Live Deployments

- **Frontend Application (Vercel):** [https://vectorless-front-8gpg.vercel.app](https://vectorless-front-8gpg.vercel.app)
- **Backend API (Hugging Face Spaces):** [https://amer003100-vectorless-rag.hf.space](https://amer003100-vectorless-rag.hf.space)
- **Interactive Swagger Docs:** [https://amer003100-vectorless-rag.hf.space/docs](https://amer003100-vectorless-rag.hf.space/docs)
- **FastMCP Endpoint:** `https://amer003100-vectorless-rag.hf.space/mcp`

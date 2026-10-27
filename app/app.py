import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure the 'app' directory is always discoverable in sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from routers import api_router
from utils.logger import app_logger

load_dotenv()

# App initialization
app = FastAPI(
    title="Atlas - Document Intelligence API",
    description="Atlas Hierarchical Tree Search Document Analysis API with Multimodal Support",
    version="2.0.0",
)

# Static files for extracted document images
STATIC_DIR = os.path.join(APP_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(api_router)

# Mount FastMCP Server (SSE & HTTP protocol for external MCP clients)
try:
    from mcp_server.web_search_mcp import mcp as research_mcp
    app.mount("/mcp", research_mcp.http_app())
    app_logger.info("FastMCP Research & Study Server mounted successfully at /mcp")
except Exception as mcp_err:
    app_logger.warning(f"Could not mount FastMCP at /mcp: {mcp_err}")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify service availability and database status."""
    from core.persistent_db import DB_STATUS, persistent_db
    try:
        docs = persistent_db.list_docs()
        count = len(docs)
    except Exception:
        count = 0
    return {
        "status": "ok",
        "service": "Vectorless RAG API",
        "fastmcp": "active",
        "database": DB_STATUS,
        "docs_count": count,
    }


@app.on_event("startup")
async def on_startup():
    app_logger.info("Vectorless RAG API service started with FastMCP Web Search enabled.")
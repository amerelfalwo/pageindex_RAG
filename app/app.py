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
    title="Vectorless RAG API",
    description="Hierarchical Tree Search Document Analysis API with Multimodal Support",
    version="1.0.0",
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


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify service availability."""
    return {"status": "ok", "service": "Vectorless RAG API"}


@app.on_event("startup")
async def on_startup():
    app_logger.info("Vectorless RAG API service started.")
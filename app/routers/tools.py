"""Tools Router for Atlas Research & Study Suite.

Exposes direct REST endpoints for MCP tools:
- POST /tools/web-search
- POST /tools/read-webpage
- POST /tools/arxiv-search
- POST /tools/youtube-transcript
- POST /tools/calculate-math
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from core.dependencies import get_client_id
from mcp_server.web_search_mcp import execute_web_search
from mcp_server.research_tools import (
    fetch_webpage_markdown,
    search_arxiv_papers_async,
    fetch_youtube_transcript_async,
    evaluate_math_expression_safe,
)

router = APIRouter(prefix="/tools", tags=["Tools & MCP"], dependencies=[Depends(get_client_id)])


class WebSearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=5, ge=1, le=15, description="Maximum results to return")


class ReadWebpageRequest(BaseModel):
    url: str = Field(..., description="Target webpage URL (must start with http:// or https://)")
    max_chars: int = Field(default=15000, ge=500, le=50000, description="Max characters to extract")


class ArxivSearchRequest(BaseModel):
    query: str = Field(..., description="Academic topic, paper title, or author name")
    max_results: int = Field(default=5, ge=1, le=20, description="Max papers to return")


class YoutubeTranscriptRequest(BaseModel):
    video_url_or_id: str = Field(..., description="YouTube video URL or 11-char video ID")
    segment_minutes: int = Field(default=3, ge=1, le=10, description="Chapter length in minutes")


class MathCalcRequest(BaseModel):
    expression: str = Field(..., description="Arithmetic or statistical expression (e.g., avg([10, 20]), percentage(25, 100))")


@router.post("/web-search")
async def api_web_search(payload: WebSearchRequest):
    """Direct HTTP endpoint for DuckDuckGo live web search with link health check and deep reading."""
    q = payload.query.strip()
    if not q:
        return {"results": [], "query": q, "count": 0}
    results = await execute_web_search(q, max_results=payload.max_results, deep_read=True)
    return {"results": results, "query": q, "count": len(results)}


@router.post("/read-webpage")
async def api_read_webpage(payload: ReadWebpageRequest):
    """Direct HTTP endpoint for Deep Web Reader with strict 6s timeout and clean markdown output."""
    res = await fetch_webpage_markdown(payload.url, max_chars=payload.max_chars, timeout=6.0)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("error"))
    return res


@router.post("/arxiv-search")
async def api_arxiv_search(payload: ArxivSearchRequest):
    """Direct HTTP endpoint for free public arXiv academic literature and research papers."""
    q = payload.query.strip()
    if not q:
        return {"papers": [], "query": q, "count": 0}
    papers = await search_arxiv_papers_async(q, max_results=payload.max_results)
    return {"papers": papers, "query": q, "count": len(papers)}


@router.post("/youtube-transcript")
async def api_youtube_transcript(payload: YoutubeTranscriptRequest):
    """Direct HTTP endpoint for extracting timestamped lecture transcript segments."""
    res = await fetch_youtube_transcript_async(payload.video_url_or_id, segment_minutes=payload.segment_minutes)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("error"))
    return res


@router.post("/calculate-math")
async def api_calculate_math(payload: MathCalcRequest):
    """Direct HTTP endpoint for Safe AST Math & Table Calculator."""
    res = evaluate_math_expression_safe(payload.expression)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("error"))
    return res

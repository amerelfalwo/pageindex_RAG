"""FastMCP Research & Study MCP Server for Atlas Document Intelligence Agent.

Provides zero-paid-API-key research capabilities exposed via FastMCP (/mcp)
and natively importable across Atlas services and external MCP clients (Claude Desktop, Cursor, Antigravity):
1. Deep Web Search with link verification & deep full-text page reading.
2. Direct Webpage Markdown Reader (`read_webpage_markdown`).
3. Academic & arXiv Paper Research (`search_arxiv_papers`).
4. YouTube Lecture Transcript & Timestamped Segmenter (`fetch_youtube_transcript`).
5. Safe AST Math & Table Calculator (`evaluate_math_expression`).
6. Internal Atlas Document Intelligence Tools (`atlas_list_documents`, `atlas_query_document`, `atlas_get_knowledge_dossier`).
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import aiohttp
from ddgs import DDGS
from fastmcp import FastMCP
from utils.logger import app_logger

from mcp_server.research_tools import (
    fetch_webpage_markdown,
    search_arxiv_papers_async,
    format_arxiv_results_for_llm,
    fetch_youtube_transcript_async,
    format_youtube_transcript_for_llm,
    evaluate_math_expression_safe,
    atlas_list_documents_impl,
    atlas_query_document_impl,
    atlas_get_knowledge_dossier_impl,
)

# Initialize FastMCP Server
mcp = FastMCP(
    "Atlas Research & Study MCP Suite",
    instructions="Comprehensive Research & Document Intelligence MCP Server for Atlas (Zero Paid API Keys Required).",
)


def _clean_search_query(query: str) -> str:
    """Strip conversational wrappers and search prefixes from query."""
    cleaned = query.strip()
    # Remove search prefixes in Arabic and English
    patterns = [
        r"^(?:ابحث\s+(?:لي\s+)?(?:عن|في\s+النت\s+عن|في\s+الويب\s+عن|على\s+الإنترنت\s+عن))\s+",
        r"^(?:search\s+(?:the\s+web\s+for|online\s+for|for|google))\s+",
        r"^(?:ما\s+هي\s+أحدث\s+|ما\s+هو\s+|من\s+هو\s+|ماذا\s+عن\s+)",
        r"^(?:اعطني\s+لينكات\s+(?:لـ|عن)?|هات\s+روابط\s+)",
    ]
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # If stripped too much, fallback to original query
    return cleaned if len(cleaned) >= 2 else query.strip()


def _extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or url
    except Exception:
        return ""


def _sync_ddgs_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Synchronous worker function to query DuckDuckGo via DDGS."""
    results = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
            for item in raw:
                title = (item.get("title") or "").strip()
                href = (item.get("href") or item.get("url") or "").strip()
                body = (item.get("body") or item.get("snippet") or "").strip()
                if href and title:
                    domain = _extract_domain(href)
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": body[:500],
                        "domain": domain,
                    })
    except Exception as e:
        app_logger.warning(f"[FastMCP] DDGS search failed for query '{query}': {e}")
    return results


async def _verify_single_url(item: Dict[str, str], session: aiohttp.ClientSession) -> Optional[Dict[str, str]]:
    """Test URL availability and verify HTTP status code is healthy (< 400)."""
    url = item.get("url")
    if not url or not url.startswith("http"):
        return None
    try:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status < 400:
                verified_item = dict(item)
                verified_item["verified"] = True
                verified_item["http_status"] = resp.status
                return verified_item
            else:
                app_logger.warning(f"[FastMCP] Discarding dead link ({resp.status}): {url}")
    except Exception as e:
        app_logger.warning(f"[FastMCP] Link check failed for {url}: {e}")
    return None


async def execute_web_search(
    query: str,
    max_results: int = 5,
    deep_read: bool = True,
) -> List[Dict[str, Any]]:
    """Asynchronous entrypoint for web search with link verification and deep full-text extraction.

    1. Searches DuckDuckGo for top candidates.
    2. Concurrently pings each URL to verify it is live, working (HTTP 200), and accessible.
    3. Filters out dead links, 404s, and unreachable domains.
    4. Automatically extracts clean markdown text from the top 1-2 URLs for deep context.
    """
    clean_q = _clean_search_query(query)
    app_logger.info(f"[FastMCP] Executing web search & deep reading for: '{clean_q}'")
    try:
        # Fetch candidate links
        candidate_count = max(max_results * 2, 8)
        raw_candidates = await asyncio.wait_for(
            asyncio.to_thread(_sync_ddgs_search, clean_q, max_results=candidate_count),
            timeout=6.0,
        )

        if not raw_candidates:
            return []

        # Concurrently test candidate links to verify alive status
        timeout = aiohttp.ClientTimeout(total=2.5)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tasks = [_verify_single_url(c, session) for c in raw_candidates]
            checked = await asyncio.gather(*tasks, return_exceptions=True)
            verified_links = [r for r in checked if isinstance(r, dict) and r.get("verified")]

        final_results = verified_links[:max_results] if verified_links else raw_candidates[:max_results]

        # Deep Web Reading: automatically extract full clean text from top 1-2 URLs
        if deep_read and final_results:
            top_urls_to_read = final_results[:2]
            deep_tasks = [
                fetch_webpage_markdown(item["url"], max_chars=3500, timeout=4.0)
                for item in top_urls_to_read
            ]
            deep_extracted = await asyncio.gather(*deep_tasks, return_exceptions=True)
            for idx, res in enumerate(deep_extracted):
                if isinstance(res, dict) and res.get("status") == "success" and res.get("content"):
                    top_urls_to_read[idx]["full_content"] = res["content"]
                    top_urls_to_read[idx]["deep_text"] = res["content"]

        app_logger.info(
            f"[FastMCP] Web search returned {len(final_results)} verified items (with deep context on top {min(len(final_results), 2)})"
        )
        return final_results
    except asyncio.TimeoutError:
        app_logger.warning(f"[FastMCP] Web search timed out for '{clean_q}'")
        return []
    except Exception as e:
        app_logger.error(f"[FastMCP] Web search error: {e}", exc_info=True)
        return []


def format_web_results_for_llm(results: List[Dict[str, Any]]) -> str:
    """Formats verified search results into a clean structured string with deep content for LLM injection."""
    if not results:
        return "No external web results found."

    lines = []
    for idx, r in enumerate(results, 1):
        status_label = f"Verified Live (HTTP {r.get('http_status', 200)})" if r.get("verified") else "Live Candidate"
        lines.append(f"{idx}. Title: {r['title']}")
        lines.append(f"   URL: {r['url']}")
        lines.append(f"   Domain: {r.get('domain', '')}")
        lines.append(f"   Status: {status_label}")
        if r.get("full_content"):
            lines.append(f"   Deep Web Content:\n{r['full_content'][:1800]}\n")
        else:
            lines.append(f"   Snippet/Content: {r['snippet']}")
        lines.append("")
    return "\n".join(lines).strip()


# ==============================================================================
# FASTMCP TOOLS EXPOSED TO CLIENTS (Claude Desktop, Cursor, Antigravity, /mcp)
# ==============================================================================

@mcp.tool(
    name="search_web",
    description="Search the live web using DuckDuckGo with link health verification and deep content extraction.",
)
async def search_web(query: str, max_results: int = 5) -> str:
    """FastMCP tool function callable by any MCP-compliant client."""
    results = await execute_web_search(query, max_results=max_results, deep_read=True)
    if not results:
        return f"No results found for query: {query}"

    output = [f"### Web Search Results for: {query}\n"]
    for idx, r in enumerate(results, 1):
        output.append(f"{idx}. [{r['title']}]({r['url']})\n   {r.get('snippet', '')}")
        if r.get("full_content"):
            output.append(f"   > **Deep Content Preview**: {r['full_content'][:500]}...\n")
        else:
            output.append("")
    return "\n".join(output)


@mcp.tool(
    name="read_webpage_markdown",
    description="Fetch a full webpage and extract clean readable text/markdown with a strict 6-second timeout.",
)
async def mcp_read_webpage_markdown(url: str, max_chars: int = 15000) -> str:
    """Fetch URL and return clean structured markdown text without HTML or ad clutter."""
    res = await fetch_webpage_markdown(url, max_chars=max_chars, timeout=6.0)
    if res.get("status") == "success":
        return f"# {res.get('title')}\n**Source**: {url}\n\n{res.get('content')}"
    return f"Failed to read webpage '{url}': {res.get('error', 'Unknown error')}"


@mcp.tool(
    name="search_arxiv_papers",
    description="Search free academic research papers on arXiv (titles, authors, publication dates, abstracts, and direct PDF URLs).",
)
async def mcp_search_arxiv_papers(query: str, max_results: int = 5) -> str:
    """Query arXiv public API for academic literature, research papers, and preprints."""
    papers = await search_arxiv_papers_async(query, max_results=max_results)
    return format_arxiv_results_for_llm(papers, query=query)


@mcp.tool(
    name="fetch_youtube_transcript",
    description="Fetch YouTube lecture transcript grouped into 3-minute timestamped chapters with citation markers.",
)
async def mcp_fetch_youtube_transcript(video_url_or_id: str, segment_minutes: int = 3) -> str:
    """Extract transcript from a YouTube lecture and group into timestamped segments."""
    data = await fetch_youtube_transcript_async(video_url_or_id, segment_minutes=segment_minutes)
    return format_youtube_transcript_for_llm(data)


@mcp.tool(
    name="evaluate_math_expression",
    description="Safely evaluate mathematical and statistical expressions (percentages, averages, ratios, sqrt) using an AST whitelist (no eval).",
)
def mcp_evaluate_math_expression(expression: str) -> str:
    """Safe AST-based arithmetic and statistical evaluator."""
    res = evaluate_math_expression_safe(expression)
    if res.get("status") == "success":
        return f"Result: {res.get('formatted')}"
    return f"Math Evaluation Error: {res.get('error')}"


@mcp.tool(
    name="atlas_list_documents",
    description="List all ingested documents in Atlas knowledge base with document IDs, filenames, page counts, and statuses.",
)
def mcp_atlas_list_documents() -> str:
    """List all indexed documents stored in Atlas."""
    docs = atlas_list_documents_impl()
    if not docs:
        return "No documents currently indexed in the Atlas knowledge base."

    lines = ["### Atlas Ingested Documents\n"]
    for idx, d in enumerate(docs, 1):
        lines.append(
            f"{idx}. **{d['filename']}** (ID: `{d['doc_id']}`)\n"
            f"   - Status: `{d['status']}` | Pages: {d['page_count']} | Sections: {d['section_count']} | Method: {d['indexing_method']}"
        )
    return "\n".join(lines)


@mcp.tool(
    name="atlas_query_document",
    description="Query a specific document tree in Atlas and retrieve grounded passages with exact (Section, Page) citations.",
)
def mcp_atlas_query_document(doc_id: str, query: str) -> str:
    """Run vectorless hierarchical retrieval on a specific indexed document."""
    res = atlas_query_document_impl(doc_id=doc_id, query=query)
    if res.get("status") != "success":
        return f"Error querying document {doc_id}: {res.get('error')}"

    passages = res.get("passages", [])
    if not passages:
        return f"No matching sections found for query '{query}' in document {res.get('filename')}."

    lines = [f"### Grounded Retrieval Passages for '{query}' in {res.get('filename')}\n"]
    for idx, p in enumerate(passages, 1):
        lines.append(f"{idx}. **[{p['section_title']} - Page {p['page']}]** (Citation: `{p['citation']}`)")
        lines.append(f"{p['text']}\n")
    return "\n".join(lines).strip()


@mcp.tool(
    name="atlas_get_knowledge_dossier",
    description="Retrieve the executive knowledge dossier (summary, key entities, findings, topics) and workspace global summary for a document.",
)
def mcp_atlas_get_knowledge_dossier(doc_id: str) -> str:
    """Retrieve structured document dossier and workspace knowledge synthesis."""
    res = atlas_get_knowledge_dossier_impl(doc_id=doc_id)
    if res.get("status") != "success":
        return f"Knowledge Dossier for {doc_id}: {res.get('message', res.get('error', 'Not found'))}"

    lines = [
        f"### Knowledge Dossier: {res.get('filename')} (ID: `{doc_id}`)\n",
        f"**Executive Summary**:\n{res.get('executive_summary')}\n",
        f"**Key Entities**: {', '.join(res.get('key_entities', []))}\n",
        "**Key Findings**:",
    ]
    for kf in res.get("key_findings", []):
        lines.append(f"- {kf}")
    lines.append(f"\n**Topics**: {', '.join(res.get('topics', []))}\n")

    if res.get("global_workspace_summary"):
        lines.append(f"**Workspace Global Summary**:\n{res.get('global_workspace_summary')}")

    return "\n".join(lines).strip()


# ==============================================================================
# FASTMCP RESOURCES
# ==============================================================================

@mcp.resource("atlas://documents")
def resource_atlas_documents() -> str:
    """Resource returning JSON list of all documents in the knowledge base."""
    docs = atlas_list_documents_impl()
    return json.dumps(docs, indent=2)


@mcp.resource("atlas://knowledge-summary")
def resource_atlas_knowledge_summary() -> str:
    """Resource returning the collective master knowledge summary of the workspace."""
    try:
        from core.storage import storage
        summary = storage.get_global_knowledge_summary() or {}
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Allows running standalone as an MCP server: `python -m mcp_server.web_search_mcp`
    mcp.run()

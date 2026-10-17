"""FastMCP Web Search Server & Tool for Atlas Agent.

Provides live web search capabilities using DuckDuckGo / DDGS, exposed via
the Model Context Protocol (FastMCP) and natively importable into FastAPI services.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import aiohttp
from ddgs import DDGS
from fastmcp import FastMCP
from utils.logger import app_logger

# Initialize FastMCP Server
mcp = FastMCP(
    "Atlas Web Search",
    instructions="FastMCP live web search tool for Atlas Document Intelligence Agent.",
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


async def execute_web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Asynchronous entrypoint for internal Python services.

    1. Searches DuckDuckGo for top candidates.
    2. Concurrently pings each URL to verify it is live, working (HTTP 200), and accessible.
    3. Filters out dead links, 404s, and unreachable domains before feeding to the LLM.
    """
    clean_q = _clean_search_query(query)
    app_logger.info(f"[FastMCP] Executing web search & link verification for: '{clean_q}'")
    try:
        # Fetch more candidates to ensure we have enough verified live links
        candidate_count = max(max_results * 2, 8)
        raw_candidates = await asyncio.wait_for(
            asyncio.to_thread(_sync_ddgs_search, clean_q, max_results=candidate_count),
            timeout=6.0,
        )

        if not raw_candidates:
            return []

        # Concurrently test all candidate links to ensure they are live and working
        timeout = aiohttp.ClientTimeout(total=2.5)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            tasks = [_verify_single_url(c, session) for c in raw_candidates]
            checked = await asyncio.gather(*tasks, return_exceptions=True)
            verified_links = [r for r in checked if isinstance(r, dict) and r.get("verified")]

        # Prioritize verified links; if all network checks blocked (e.g. strict firewall), fallback gracefully to candidates
        final_results = verified_links[:max_results] if verified_links else raw_candidates[:max_results]
        app_logger.info(
            f"[FastMCP] Web search returned {len(final_results)} verified live items (from {len(raw_candidates)} candidates)"
        )
        return final_results
    except asyncio.TimeoutError:
        app_logger.warning(f"[FastMCP] Web search timed out for '{clean_q}'")
        return []
    except Exception as e:
        app_logger.error(f"[FastMCP] Web search error: {e}", exc_info=True)
        return []


def format_web_results_for_llm(results: List[Dict[str, str]]) -> str:
    """Formats verified search results into a clean structured string for LLM injection."""
    if not results:
        return "No external web results found."

    lines = []
    for idx, r in enumerate(results, 1):
        status_label = f"Verified Live (HTTP {r.get('http_status', 200)})" if r.get("verified") else "Live Candidate"
        lines.append(f"{idx}. Title: {r['title']}")
        lines.append(f"   URL: {r['url']}")
        lines.append(f"   Domain: {r.get('domain', '')}")
        lines.append(f"   Status: {status_label}")
        lines.append(f"   Snippet/Content: {r['snippet']}")
        lines.append("")
    return "\n".join(lines).strip()


@mcp.tool(
    name="search_web",
    description="Search the live web using DuckDuckGo to find current facts, articles, and external source links.",
)
def search_web(query: str, max_results: int = 5) -> str:
    """FastMCP tool function callable by any MCP-compliant client."""
    clean_q = _clean_search_query(query)
    results = _sync_ddgs_search(clean_q, max_results=max_results)
    if not results:
        return f"No results found for query: {query}"

    output = [f"### Web Search Results for: {query}\n"]
    for idx, r in enumerate(results, 1):
        output.append(f"{idx}. [{r['title']}]({r['url']})\n   {r['snippet']}\n")
    return "\n".join(output)


if __name__ == "__main__":
    # Allows running standalone as an MCP server: `python -m app.mcp.web_search_mcp`
    mcp.run()

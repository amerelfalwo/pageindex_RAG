"""Atlas Research & Study MCP Tools.

Zero paid API keys required.
Provides:
1. Deep Web Reader (`read_webpage_markdown`) with strict 6s timeout & clean extraction.
2. Academic & arXiv Research (`search_arxiv_papers`) querying public arXiv Atom XML API.
3. YouTube Lecture Transcript (`fetch_youtube_transcript`) grouping into 3-min timestamped chapters.
4. Safe Math & Table Calculator (`evaluate_math_expression`) using AST whitelist (no eval).
5. Atlas Internal Document Intelligence MCP capabilities (`atlas_list_documents`, `atlas_query_document`, `atlas_get_knowledge_dossier`).
"""

import ast
import asyncio
import html
import ipaddress
import math
import operator
import os
import re
import socket
import statistics
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from utils.logger import app_logger


# ==============================================================================
# 1. DEEP WEB READER (read_webpage_markdown)
# ==============================================================================

def clean_html_to_markdown(html_content: str, max_chars: int = 15000) -> Tuple[str, str]:
    """Extract clean title and structured text/markdown from raw HTML.
    
    Strips scripts, styles, navigation, footer, ads, and irrelevant tags.
    """
    title = "Webpage Content"
    body_text = ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove boilerplates, navigation, scripts, and trackers
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "iframe", "form", "button"]):
            tag.decompose()

        # Convert links to markdown
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            text = a.get_text(separator=" ", strip=True)
            if href and text and not href.startswith("javascript:"):
                a.replace_with(f" [{text}]({href}) ")

        # Convert headings to markdown
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            lvl = int(h.name[1])
            txt = h.get_text(separator=" ", strip=True)
            if txt:
                h.replace_with(f"\n\n{'#' * lvl} {txt}\n\n")

        # Convert lists
        for li in soup.find_all("li"):
            txt = li.get_text(separator=" ", strip=True)
            if txt:
                li.replace_with(f"\n- {txt}")

        # Convert paragraphs
        for p in soup.find_all("p"):
            txt = p.get_text(separator=" ", strip=True)
            if txt:
                p.replace_with(f"\n\n{txt}\n\n")

        body_text = soup.get_text(separator=" ")
    except Exception as bs_err:
        app_logger.warning(f"[WebReader] BeautifulSoup parse failed, falling back to regex: {bs_err}")
        # Fallback regex-based extraction
        title_m = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if title_m:
            title = html.unescape(title_m.group(1).strip())
        cleaned = re.sub(r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", "", html_content, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        body_text = html.unescape(cleaned)

    # Normalize excessive spaces and blank lines
    body_text = re.sub(r"[ \t]+", " ", body_text)
    body_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", body_text).strip()
    
    if len(body_text) > max_chars:
        body_text = body_text[:max_chars].rstrip() + "\n\n[...Content truncated for length...]"

    return title, body_text


MAX_REDIRECT_HOPS = 3
DEFAULT_MAX_WEB_PAGE_BYTES = 2 * 1024 * 1024  # 2 MB


def is_safe_ip_address(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> Tuple[bool, str]:
    """Validates that an IP address is not internal, loopback, private, multicast, or reserved."""
    if ip.is_loopback:
        return False, "loopback address"
    if ip.is_private:
        return False, "private network address"
    if ip.is_link_local:
        return False, "link-local address"
    if ip.is_multicast:
        return False, "multicast address"
    if ip.is_reserved:
        return False, "reserved address"
    if ip.is_unspecified:
        return False, "unspecified address"

    if isinstance(ip, ipaddress.IPv6Address):
        if ip.is_site_local:
            return False, "site-local IPv6 address"
        if ip in ipaddress.ip_network("fc00::/7"):
            return False, "unique-local IPv6 address"

    return True, ""


async def validate_url_for_ssrf(url: str) -> Tuple[bool, str, Optional[urllib.parse.ParseResult]]:
    """
    Validates URL scheme and resolves DNS hostnames to verify that all target
    IP addresses are public, rejecting internal/private/cloud-metadata networks.
    """
    if not url:
        return False, "URL cannot be empty.", None

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as e:
        return False, f"Invalid URL structure: {e}", None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"Invalid URL scheme '{scheme}'. Only http:// and https:// are permitted.", parsed

    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname.", parsed

    # Attempt to resolve IP addresses for hostname
    loop = asyncio.get_running_loop()
    try:
        addr_info = await loop.getaddrinfo(hostname, None)
    except Exception as dns_err:
        return False, f"DNS resolution failed for '{hostname}': {dns_err}", parsed

    if not addr_info:
        return False, f"Could not resolve host '{hostname}'.", parsed

    for info in addr_info:
        ip_str = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            is_safe, reason = is_safe_ip_address(ip_obj)
            if not is_safe:
                return (
                    False,
                    f"Access to '{hostname}' ({ip_str}) rejected: {reason} (SSRF protection).",
                    parsed,
                )
        except ValueError:
            return False, f"Invalid IP address resolved: {ip_str}", parsed

    return True, "", parsed


async def fetch_webpage_markdown(
    url: str,
    max_chars: int = 15000,
    timeout: float = 6.0,
) -> Dict[str, Any]:
    """Fetch URL with strict SSRF protection, manual redirects (max 3 hops), body limit, and clean markdown."""
    t_start = time.monotonic()
    raw_max_bytes = os.getenv("MAX_WEB_PAGE_BYTES")
    try:
        max_bytes = int(raw_max_bytes) if raw_max_bytes else DEFAULT_MAX_WEB_PAGE_BYTES
    except ValueError:
        max_bytes = DEFAULT_MAX_WEB_PAGE_BYTES

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 AtlasResearch/2.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    }

    current_url = url.strip() if url else ""
    hops = 0
    raw_html = ""

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers=headers) as client:
            while True:
                # 1. Scheme + SSRF DNS validation on current_url
                is_safe, err_msg, _ = await validate_url_for_ssrf(current_url)
                if not is_safe:
                    elapsed = round(time.monotonic() - t_start, 2)
                    app_logger.warning(f"[WebReader] SSRF blocked target '{current_url}': {err_msg}")
                    return {
                        "status": "error",
                        "url": current_url,
                        "error": err_msg,
                        "content": "",
                        "duration": elapsed,
                    }

                # 2. Issue streaming request
                req = client.build_request("GET", current_url)
                resp = await client.send(req, stream=True)

                try:
                    # 3. Check for redirects (3xx)
                    if 300 <= resp.status_code < 400:
                        hops += 1
                        if hops > MAX_REDIRECT_HOPS:
                            await resp.aclose()
                            elapsed = round(time.monotonic() - t_start, 2)
                            return {
                                "status": "error",
                                "url": current_url,
                                "error": f"Too many redirects (exceeded limit of {MAX_REDIRECT_HOPS} hops).",
                                "content": "",
                                "duration": elapsed,
                            }

                        location = resp.headers.get("location")
                        await resp.aclose()
                        if not location:
                            elapsed = round(time.monotonic() - t_start, 2)
                            return {
                                "status": "error",
                                "url": current_url,
                                "error": f"HTTP {resp.status_code} redirect missing Location header.",
                                "content": "",
                                "duration": elapsed,
                            }

                        # Resolve relative redirect against current_url
                        current_url = urllib.parse.urljoin(current_url, location)
                        continue

                    # 4. Check HTTP status code
                    if resp.status_code >= 400:
                        await resp.aclose()
                        elapsed = round(time.monotonic() - t_start, 2)
                        return {
                            "status": "error",
                            "url": current_url,
                            "error": f"HTTP {resp.status_code}: {resp.reason_phrase}",
                            "content": "",
                            "duration": elapsed,
                        }

                    # 5. Check Content-Length if reported
                    content_len_hdr = resp.headers.get("content-length")
                    if content_len_hdr:
                        try:
                            if int(content_len_hdr) > max_bytes:
                                await resp.aclose()
                                elapsed = round(time.monotonic() - t_start, 2)
                                return {
                                    "status": "error",
                                    "url": current_url,
                                    "error": f"Response size exceeds maximum limit of {max_bytes} bytes.",
                                    "content": "",
                                    "duration": elapsed,
                                }
                        except ValueError:
                            pass

                    # 6. Stream chunks with size enforcement
                    content_chunks = []
                    total_bytes = 0
                    async for chunk in resp.aiter_bytes():
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            await resp.aclose()
                            elapsed = round(time.monotonic() - t_start, 2)
                            return {
                                "status": "error",
                                "url": current_url,
                                "error": f"Response exceeded maximum limit of {max_bytes} bytes during download.",
                                "content": "",
                                "duration": elapsed,
                            }
                        content_chunks.append(chunk)

                    raw_bytes = b"".join(content_chunks)
                    encoding = resp.encoding or "utf-8"
                    raw_html = raw_bytes.decode(encoding, errors="replace")
                    break
                finally:
                    await resp.aclose()

        title, clean_text = clean_html_to_markdown(raw_html, max_chars=max_chars)
        elapsed = round(time.monotonic() - t_start, 2)
        app_logger.info(f"[WebReader] Successfully read {url} ({len(clean_text)} chars in {elapsed}s, {hops} redirects)")
        return {
            "status": "success",
            "url": current_url,
            "title": title,
            "content": clean_text,
            "characters": len(clean_text),
            "duration": elapsed,
        }
    except httpx.TimeoutException:
        elapsed = round(time.monotonic() - t_start, 2)
        app_logger.warning(f"[WebReader] Timeout ({timeout}s) fetching URL: {url}")
        return {
            "status": "error",
            "url": current_url,
            "error": f"Request timed out after {timeout} seconds.",
            "content": "",
            "duration": elapsed,
        }
    except Exception as e:
        elapsed = round(time.monotonic() - t_start, 2)
        app_logger.warning(f"[WebReader] Error reading {url}: {e}")
        return {
            "status": "error",
            "url": current_url,
            "error": str(e),
            "content": "",
            "duration": elapsed,
        }


# ==============================================================================
# 2. ACADEMIC & arXiv RESEARCH TOOL (search_arxiv_papers)
# ==============================================================================

async def search_arxiv_papers_async(
    query: str,
    max_results: int = 5,
    timeout: float = 8.0,
) -> List[Dict[str, Any]]:
    """Query free public arXiv API and return structured paper metadata."""
    clean_q = query.strip()
    if not clean_q:
        return []

    # Clean conversational query wrappers
    clean_q = re.sub(r"^(?:search\s+(?:arxiv\s+for|papers\s+for|for)?|arxiv\s+|papers?\s+(?:on|about)?)\s*", "", clean_q, flags=re.IGNORECASE).strip()
    if not clean_q:
        clean_q = query.strip()

    encoded_q = urllib.parse.quote_plus(clean_q)
    arxiv_url = f"https://export.arxiv.org/api/query?search_query=all:{encoded_q}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

    headers = {
        "User-Agent": "Atlas-Document-Intelligence/2.0 (mailto:atlas-rag@localhost)",
    }

    t_start = time.monotonic()
    papers: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = await client.get(arxiv_url)
            resp.raise_for_status()
            xml_text = resp.text

        root = ET.fromstring(xml_text)
        xmlns = "{http://www.w3.org/2005/Atom}"
        arxiv_ns = "{http://arxiv.org/schemas/atom}"

        entries = root.findall(f"{xmlns}entry")
        for entry in entries:
            id_elem = entry.find(f"{xmlns}id")
            title_elem = entry.find(f"{xmlns}title")
            summary_elem = entry.find(f"{xmlns}summary")
            published_elem = entry.find(f"{xmlns}published")
            updated_elem = entry.find(f"{xmlns}updated")

            paper_id_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
            raw_id = paper_id_url.split("/abs/")[-1] if "/abs/" in paper_id_url else paper_id_url

            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else "Untitled"
            title = re.sub(r"\s+", " ", title)

            abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""
            abstract = re.sub(r"\s+", " ", abstract)

            published = published_elem.text.strip()[:10] if published_elem is not None and published_elem.text else ""
            updated = updated_elem.text.strip()[:10] if updated_elem is not None and updated_elem.text else ""

            # Extract authors
            authors = []
            for author_node in entry.findall(f"{xmlns}author"):
                name_node = author_node.find(f"{xmlns}name")
                if name_node is not None and name_node.text:
                    authors.append(name_node.text.strip())

            # Direct PDF link
            pdf_url = ""
            for link in entry.findall(f"{xmlns}link"):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            if not pdf_url and raw_id:
                pdf_url = f"https://arxiv.org/pdf/{raw_id}.pdf"

            # Primary Category
            prim_cat = entry.find(f"{arxiv_ns}primary_category")
            category = prim_cat.attrib.get("term", "") if prim_cat is not None else ""

            papers.append({
                "arxiv_id": raw_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published": published,
                "updated": updated,
                "abs_url": paper_id_url,
                "pdf_url": pdf_url,
                "category": category,
            })

        elapsed = round(time.monotonic() - t_start, 2)
        app_logger.info(f"[arXiv] Found {len(papers)} papers for '{clean_q}' in {elapsed}s")
        return papers
    except Exception as e:
        app_logger.warning(f"[arXiv] Search failed for '{clean_q}': {e}")
        return []


def format_arxiv_results_for_llm(papers: List[Dict[str, Any]], query: str = "") -> str:
    """Format arXiv papers into rich markdown context for the LLM or MCP client."""
    if not papers:
        return f"No academic papers found on arXiv for: '{query}'"

    lines = [f"### Academic Research & arXiv Papers for: {query}\n"]
    for idx, p in enumerate(papers, 1):
        authors_str = ", ".join(p["authors"][:4])
        if len(p["authors"]) > 4:
            authors_str += " et al."
        lines.append(f"{idx}. **[{p['title']}]({p['abs_url']})**")
        lines.append(f"   - **Authors**: {authors_str}")
        lines.append(f"   - **Published**: {p['published']} | **Category**: {p.get('category', 'N/A')}")
        lines.append(f"   - **PDF Link**: [Direct Download PDF]({p['pdf_url']})")
        lines.append(f"   - **Abstract**: {p['abstract'][:600]}...")
        lines.append("")
    return "\n".join(lines).strip()


# ==============================================================================
# 3. YOUTUBE LECTURE TRANSCRIPT TOOL (fetch_youtube_transcript)
# ==============================================================================

def extract_youtube_video_id(url_or_id: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from various URL patterns or raw ID."""
    clean = url_or_id.strip()
    if re.match(r"^[0-9A-Za-z_-]{11}$", clean):
        return clean
    patterns = [
        r"(?:v=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})",
        r"[?&]v=([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, clean)
        if m:
            return m.group(1)
    return None


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


async def fetch_youtube_transcript_async(
    video_url_or_id: str,
    segment_minutes: int = 3,
) -> Dict[str, Any]:
    """Fetch video transcript and group into logical timestamped chapters (virtual tree nodes)."""
    video_id = extract_youtube_video_id(video_url_or_id)
    if not video_id:
        return {
            "status": "error",
            "error": f"Invalid YouTube video URL or ID: '{video_url_or_id}'",
            "video_id": "",
            "segments": [],
        }

    segment_secs = max(segment_minutes * 60, 60)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()

        def _fetch_sync():
            try:
                # Try fetching standard transcripts (English, Arabic, Spanish, French, etc.)
                transcript_obj = api.fetch(video_id, languages=("en", "ar", "es", "fr", "de"))
                return transcript_obj.to_raw_data()
            except Exception:
                # Fallback to listing all available transcripts (generated or manual)
                t_list = api.list(video_id)
                for tr in t_list:
                    return tr.fetch().to_raw_data()
                raise

        raw_snippets = await asyncio.to_thread(_fetch_sync)

        if not raw_snippets:
            return {
                "status": "error",
                "error": "No transcript content available for this video.",
                "video_id": video_id,
                "segments": [],
            }

        # Group snippets into logical chapters (e.g., 3-minute segments)
        grouped_segments: Dict[int, List[Dict[str, Any]]] = {}
        max_duration = 0.0

        for item in raw_snippets:
            start = float(item.get("start", 0.0))
            duration = float(item.get("duration", 0.0))
            text = (item.get("text") or "").strip()
            if not text:
                continue
            max_duration = max(max_duration, start + duration)
            seg_idx = int(start // segment_secs)
            if seg_idx not in grouped_segments:
                grouped_segments[seg_idx] = []
            grouped_segments[seg_idx].append(item)

        segments: List[Dict[str, Any]] = []
        for s_idx in sorted(grouped_segments.keys()):
            items = grouped_segments[s_idx]
            seg_start_sec = s_idx * segment_secs
            seg_end_sec = min(seg_start_sec + segment_secs, max_duration)
            time_start_str = format_timestamp(seg_start_sec)
            time_end_str = format_timestamp(seg_end_sec)
            combined_text = " ".join([i.get("text", "") for i in items])
            combined_text = re.sub(r"\s+", " ", combined_text).strip()

            chapter_title = f"Lecture Chapter {s_idx + 1}: {time_start_str} - {time_end_str}"
            citation = f'(Section: "Lecture Chapter {s_idx + 1}", Page/Time: "{time_start_str}")'

            segments.append({
                "node_id": f"yt_seg_{s_idx + 1}",
                "title": chapter_title,
                "section_title": f"Lecture Chapter {s_idx + 1}",
                "page": time_start_str,
                "page_index": s_idx + 1,
                "time_start": time_start_str,
                "time_end": time_end_str,
                "start_seconds": seg_start_sec,
                "end_seconds": seg_end_sec,
                "text": combined_text,
                "citation": citation,
            })

        app_logger.info(f"[YouTube] Ingested {len(segments)} lecture segments for video {video_id}")
        return {
            "status": "success",
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "total_duration": format_timestamp(max_duration),
            "total_duration_seconds": round(max_duration, 1),
            "segment_minutes": segment_minutes,
            "segments_count": len(segments),
            "segments": segments,
        }

    except Exception as e:
        app_logger.warning(f"[YouTube] Could not fetch transcript for {video_id}: {e}")
        return {
            "status": "error",
            "error": f"Failed to retrieve YouTube transcript: {str(e)}",
            "video_id": video_id,
            "segments": [],
        }


def format_youtube_transcript_for_llm(data: Dict[str, Any]) -> str:
    """Format lecture segments as virtual tree nodes with section and timestamp citations."""
    if data.get("status") != "success" or not data.get("segments"):
        return f"Unable to fetch YouTube transcript: {data.get('error', 'Unknown error')}"

    lines = [
        f"### YouTube Lecture Transcript: {data['url']}",
        f"Total Duration: {data.get('total_duration')} | Chapters: {data.get('segments_count')}\n",
    ]
    for seg in data["segments"]:
        lines.append(f"#### [{seg['title']}] (Citation: `{seg['citation']}`)")
        lines.append(f"{seg['text']}\n")
    return "\n".join(lines).strip()


# ==============================================================================
# 4. SAFE MATH & TABLE CALCULATOR (evaluate_math_expression)
# ==============================================================================

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "avg": lambda items: sum(items) / len(items) if items else 0.0,
    "average": lambda items: sum(items) / len(items) if items else 0.0,
    "mean": lambda items: statistics.mean(items) if items else 0.0,
    "median": lambda items: statistics.median(items) if items else 0.0,
    "percentage": lambda part, total: (part / total) * 100.0 if total != 0 else 0.0,
    "pct": lambda part, total: (part / total) * 100.0 if total != 0 else 0.0,
    "ratio": lambda a, b: a / b if b != 0 else 0.0,
}

_SAFE_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_node(node: ast.AST) -> Any:
    """Recursively evaluates AST nodes against an explicit whitelist of safe operations."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Constant value '{node.value}' of type {type(node.value).__name__} is not allowed.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Binary operator {op_type.__name__} is not allowed.")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero in mathematical expression.")
        return _SAFE_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unary operator {op_type.__name__} is not allowed.")
        return _SAFE_OPS[op_type](_eval_ast_node(node.operand))

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_ast_node(elt) for elt in node.elts]

    if isinstance(node, ast.Name):
        name_id = node.id.lower()
        if name_id in _SAFE_CONSTS:
            return _SAFE_CONSTS[name_id]
        raise ValueError(f"Variable/identifier '{node.id}' is not allowed.")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Arbitrary member attribute/function calls are strictly prohibited.")
        fn_name = node.func.id.lower()
        if fn_name not in _SAFE_FUNCS:
            raise ValueError(f"Function '{node.func.id}' is not in the approved math whitelist.")
        fn = _SAFE_FUNCS[fn_name]
        args = [_eval_ast_node(a) for a in node.args]
        return fn(*args)

    raise ValueError(f"Syntax element '{type(node).__name__}' is forbidden.")


def evaluate_math_expression_safe(expression: str) -> Dict[str, Any]:
    """Safely evaluates arithmetic and statistical expressions using AST parsing without eval()."""
    clean_expr = expression.strip()
    if not clean_expr:
        return {
            "status": "error",
            "expression": expression,
            "error": "Expression is empty.",
            "result": None,
        }

    # Normalize percentage syntax: "25%" -> "(25 / 100.0)"
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", clean_expr)

    try:
        parsed_tree = ast.parse(normalized, mode="eval")
        result = _eval_ast_node(parsed_tree)

        # Clean float representation
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 6)

        return {
            "status": "success",
            "expression": clean_expr,
            "result": result,
            "formatted": f"{clean_expr} = {result}",
        }
    except Exception as e:
        return {
            "status": "error",
            "expression": clean_expr,
            "error": f"Evaluation error: {str(e)}",
            "result": None,
        }


# ==============================================================================
# 5. ATLAS INTERNAL CAPABILITIES IMPLEMENTATIONS
# ==============================================================================

def atlas_list_documents_impl() -> List[Dict[str, Any]]:
    """List all indexed documents with IDs, filenames, and status."""
    try:
        from core.storage import storage
        docs = storage.list_documents()
        return [
            {
                "doc_id": d.get("doc_id") or d.get("document_id"),
                "filename": d.get("filename", "document.pdf"),
                "status": d.get("status", "unknown"),
                "page_count": d.get("page_count", 0),
                "section_count": d.get("section_count", 0),
                "indexing_method": d.get("indexing_method", "standard"),
                "created_at": d.get("created_at"),
            }
            for d in docs
        ]
    except Exception as e:
        app_logger.error(f"[AtlasMCP] Failed to list documents: {e}")
        return []


def atlas_query_document_impl(doc_id: str, query: str) -> Dict[str, Any]:
    """Retrieve grounded passages from a document tree with (Section, Page) citations."""
    try:
        from core.storage import storage
        from core.rag_pipeline import VectorlessRAG
        from core.bm25 import BM25NodeIndex

        doc = storage.get_document(doc_id)
        if not doc:
            return {
                "status": "error",
                "error": f"Document ID '{doc_id}' not found in knowledge base.",
                "doc_id": doc_id,
                "passages": [],
            }

        node_map = storage.get_node_map(doc_id) or {}
        flat_sections = storage.get_flat_sections(doc_id) or []
        tree = storage.get_tree(doc_id) or []

        # Warm up BM25 index if needed
        bm25 = storage.get_bm25_index(doc_id)
        if not bm25 and flat_sections:
            doc_nodes = [node_map[s["node_id"]] for s in flat_sections if s.get("node_id") in node_map]
            if doc_nodes:
                bm25 = BM25NodeIndex(doc_nodes)
                storage.set_bm25_index(doc_id, bm25)

        selected_nodes: List[Dict[str, Any]] = []
        if bm25:
            bm25_matches = bm25.search(query, top_k=5)
            for nid, _ in bm25_matches:
                if nid in node_map:
                    selected_nodes.append(node_map[nid])

        if not selected_nodes and tree:
            # Fallback to top-level tree nodes
            selected_nodes = tree[:4]

        # Apply sub-node paragraph pruning
        pruned = VectorlessRAG.prune_sub_nodes(query, selected_nodes, deep_analysis=True)

        passages = []
        for n in pruned:
            sec_title = n.get("title") or n.get("name") or "Section"
            page_num = n.get("page_index") or n.get("page") or 1
            citation = f'(Section: "{sec_title}", Page: {page_num})'
            passages.append({
                "section_title": sec_title,
                "page": page_num,
                "citation": citation,
                "text": (n.get("text") or n.get("content") or "").strip()[:1800],
            })

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": doc.get("filename", "document.pdf"),
            "query": query,
            "passages_count": len(passages),
            "passages": passages,
        }
    except Exception as e:
        app_logger.error(f"[AtlasMCP] Query failed for doc {doc_id}: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "doc_id": doc_id,
            "passages": [],
        }


def atlas_get_knowledge_dossier_impl(doc_id: str) -> Dict[str, Any]:
    """Retrieve structured knowledge dossier for an individual document and workspace summary."""
    try:
        from core.storage import storage
        dossier = storage.get_knowledge_dossier(doc_id)
        global_summary = storage.get_global_knowledge_summary() or {}
        doc = storage.get_document(doc_id) or {}

        if not dossier:
            # Fallback to basic metadata
            return {
                "status": "not_found",
                "doc_id": doc_id,
                "filename": doc.get("filename", "document.pdf"),
                "message": "Knowledge dossier has not been generated for this document yet.",
                "global_knowledge_summary": global_summary.get("collective_summary"),
            }

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": dossier.get("filename") or doc.get("filename", "document.pdf"),
            "executive_summary": dossier.get("executive_summary", ""),
            "key_entities": dossier.get("key_entities", []),
            "key_findings": dossier.get("key_findings", []),
            "topics": dossier.get("topics", []),
            "global_workspace_summary": global_summary.get("collective_summary"),
            "updated_at": dossier.get("updated_at"),
        }
    except Exception as e:
        app_logger.error(f"[AtlasMCP] Failed to fetch dossier for doc {doc_id}: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "doc_id": doc_id,
        }

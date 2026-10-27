import os
import re
import time
import uuid
from typing import Optional, List, Dict, Any

import fitz
from pageindex.client import PageIndexClient
from utils.logger import app_logger


class DocumentService:
    def __init__(self, api_key: Optional[str] = None):
        api_keys = []
        if api_key:
            api_keys.append(api_key)

        api_keys.extend(
            [
                os.getenv("PAGEINDEX_API_KEY_1"),
                os.getenv("PAGEINDEX_API_KEY_2"),
                os.getenv("PAGEINDEX_API_KEY_3"),
            ]
        )

        self.api_keys = [key.strip() for key in api_keys if key and key.strip()]

    def _extract_images(self, file_path: str, max_images: int = 30) -> dict:
        doc = fitz.open(file_path)
        static_dir = os.path.join(os.getcwd(), "app", "static")
        if not os.path.exists(static_dir):
            static_dir = os.path.join(os.getcwd(), "static")
        os.makedirs(static_dir, exist_ok=True)
        page_images = {}
        total_extracted = 0

        for page_index in range(len(doc)):
            if total_extracted >= max_images:
                break
            page = doc[page_index]
            images = []

            # 1. Raster images
            for img in page.get_images(full=True):
                if total_extracted >= max_images:
                    break
                xref = img[0]
                try:
                    extracted = doc.extract_image(xref)
                    ext = extracted.get("ext", "png")
                    image_bytes = extracted.get("image")
                    if not image_bytes or len(image_bytes) < 1024:
                        continue
                    filename = f"{uuid.uuid4().hex}.{ext}"
                    path = os.path.join(static_dir, filename)
                    with open(path, "wb") as f:
                        f.write(image_bytes)
                    images.append(f"/static/{filename}")
                    total_extracted += 1
                except Exception:
                    continue

            # 2. Vector figures / diagram captions (LaTeX / scientific publications)
            text_blocks = page.get_text("blocks")
            for b in text_blocks:
                if total_extracted >= max_images:
                    break
                b_text = b[4].strip()
                m = re.match(r'(Fig(?:ure|\.)\s*(\d+)\.?\s*(.*?))(?:\n|$)', b_text, re.IGNORECASE)
                if m:
                    fig_num = m.group(2)
                    caption_y = b[1]
                    is_col2 = b[0] >= 280
                    is_col1 = b[2] <= 320
                    if is_col2:
                        x0, x1 = 300, min(page.rect.width - 35, b[2] + 20)
                    elif is_col1:
                        x0, x1 = 40, 300
                    else:
                        x0, x1 = 40, page.rect.width - 40
                    y0 = max(35, caption_y - 260)
                    y1 = min(page.rect.height - 35, b[3] + 8)
                    clip_rect = fitz.Rect(x0, y0, x1, y1)

                    filename = f"fig_p{page_index + 1}_{fig_num}_{uuid.uuid4().hex[:6]}.png"
                    path = os.path.join(static_dir, filename)
                    try:
                        pix = page.get_pixmap(clip=clip_rect, dpi=150)
                        pix.save(path)
                        images.append(f"/static/{filename}")
                        total_extracted += 1
                    except Exception:
                        continue

            if images:
                page_images[page_index + 1] = images

        doc.close()
        return page_images

    def _inject_images(self, tree: Any, page_images: dict):
        if isinstance(tree, dict):
            tree = [tree]

        if not isinstance(tree, list):
            return tree

        # Flatten nodes to determine page span of each section
        flat_nodes = []
        def collect_nodes(items):
            for it in items:
                if isinstance(it, dict):
                    flat_nodes.append(it)
                    if it.get("nodes"):
                        collect_nodes(it["nodes"])
        collect_nodes(tree)

        for i, node in enumerate(flat_nodes):
            p_start = node.get("page_index") or node.get("page")
            if not p_start:
                continue
            p_next = flat_nodes[i + 1].get("page_index") or flat_nodes[i + 1].get("page") if i + 1 < len(flat_nodes) else None
            p_end = (p_next - 1) if p_next and p_next > p_start else p_start

            # Collect images across all pages in this section's span
            section_images = []
            for p in range(p_start, p_end + 1):
                if p in page_images:
                    section_images.extend(page_images[p])

            if section_images:
                existing = node.get("images") or []
                node["images"] = list(dict.fromkeys(existing + section_images))

        return tree

    @staticmethod
    def _extract_page_texts(doc: fitz.Document) -> Dict[int, str]:
        """Extracts text for all pages into a 1-indexed dictionary."""
        page_texts: Dict[int, str] = {}
        for idx in range(len(doc)):
            page_texts[idx + 1] = doc[idx].get_text() or ""
        return page_texts

    @staticmethod
    def _format_page_tagged_text(page_texts: Dict[int, str], start_page: int, end_page: int) -> str:
        """Formats text across a range of pages with explicit --- [PAGE X] --- delimiters."""
        parts: List[str] = []
        for p in range(start_page, end_page + 1):
            p_text = page_texts.get(p, "").strip()
            if p_text:
                parts.append(f"--- [PAGE {p}] ---\n{p_text}")
        if not parts:
            return f"--- [PAGE {start_page}] ---\n(No textual content on this page)"
        return "\n\n".join(parts)

    def _inject_page_boundaries(self, tree: Any, page_texts: Dict[int, str], total_pages: int):
        """Enriches tree nodes with explicit --- [PAGE X] --- boundary markers across each section's page span."""
        if isinstance(tree, dict):
            tree = [tree]
        if not isinstance(tree, list):
            return tree

        flat_nodes: List[Dict[str, Any]] = []
        def collect_nodes(items):
            for it in items:
                if isinstance(it, dict):
                    flat_nodes.append(it)
                    if it.get("nodes"):
                        collect_nodes(it["nodes"])
        collect_nodes(tree)

        for i, node in enumerate(flat_nodes):
            p_start = node.get("page_index") or node.get("page") or node.get("start_page")
            if not p_start:
                continue
            try:
                p_start = max(1, int(p_start))
            except Exception:
                continue

            # Determine next section start page
            p_next = None
            for j in range(i + 1, len(flat_nodes)):
                cand_p = flat_nodes[j].get("page_index") or flat_nodes[j].get("page") or flat_nodes[j].get("start_page")
                if cand_p:
                    try:
                        cand_p = int(cand_p)
                        if cand_p > p_start:
                            p_next = cand_p
                            break
                    except Exception:
                        pass
            p_end = (p_next - 1) if p_next else total_pages
            p_end = max(p_start, min(total_pages, p_end))

            current_text = node.get("text") or ""
            # Inject explicit page boundaries if markers are missing
            if "--- [PAGE " not in current_text:
                tagged_text = self._format_page_tagged_text(page_texts, p_start, p_end)
                node["text"] = tagged_text
            node["page_index"] = p_start

        return tree

    def _local_pdf_parse(self, file_path: str) -> list:
        """Fast local fallback parser using PyMuPDF when cloud indexing is unavailable or times out."""
        app_logger.info(f"Extracting document structure locally via PyMuPDF for: {file_path}")
        doc = fitz.open(file_path)
        nodes = []
        total_pages = len(doc)
        page_texts = self._extract_page_texts(doc)

        # Try to extract TOC / Outlines if available
        toc = doc.get_toc()  # [[lvl, title, page], ...]

        if toc and len(toc) > 0:
            for idx, item in enumerate(toc):
                level, title, page_num = item[0], item[1], max(1, item[2])
                if idx + 1 < len(toc):
                    next_page = max(1, toc[idx + 1][2])
                    page_end = max(page_num, next_page - 1) if next_page > page_num else page_num
                else:
                    page_end = total_pages

                tagged_text = self._format_page_tagged_text(page_texts, page_num, page_end)
                first_lines = [l.strip() for l in tagged_text.split("\n") if l.strip() and not l.startswith("--- [PAGE")]
                summary = " ".join(first_lines[:3])[:250] if first_lines else (title.strip() or f"Section p.{page_num}")

                nodes.append({
                    "node_id": f"sec_{uuid.uuid4().hex[:8]}",
                    "title": title.strip() or f"Section p.{page_num}",
                    "page_index": page_num,
                    "text": tagged_text,
                    "summary": summary,
                })

        if not nodes:
            # Fallback: page by page extraction
            for page_num in range(1, total_pages + 1):
                raw_text = page_texts.get(page_num, "").strip()
                if not raw_text:
                    continue
                lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
                first_heading = lines[0][:80] if lines else f"Page {page_num}"
                tagged_text = f"--- [PAGE {page_num}] ---\n{raw_text}"
                nodes.append({
                    "node_id": f"page_{page_num}",
                    "title": first_heading,
                    "page_index": page_num,
                    "text": tagged_text,
                    "summary": " ".join(lines[:3])[:250],
                })

        doc.close()
        return nodes

    @staticmethod
    def _is_valid_tree(tree: Any) -> bool:
        if not tree:
            return False
        if isinstance(tree, dict):
            tree = [tree]
        if not isinstance(tree, list) or len(tree) == 0:
            return False

        def has_valid_content(items: list) -> bool:
            for it in items:
                if not isinstance(it, dict):
                    continue
                if (it.get("title") and it["title"].strip()) or (it.get("text") and it["text"].strip()) or (it.get("summary") and it["summary"].strip()):
                    return True
                if it.get("nodes") and isinstance(it["nodes"], list) and has_valid_content(it["nodes"]):
                    return True
            return False

        return has_valid_content(tree)

    @staticmethod
    def _count_sections(tree: Any) -> int:
        if not isinstance(tree, list):
            return 1 if isinstance(tree, dict) else 0
        count = 0
        for node in tree:
            if isinstance(node, dict):
                count += 1
                if node.get("nodes"):
                    count += DocumentService._count_sections(node["nodes"])
        return count

    def process_pdf(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        on_stage: Optional[Any] = None,
    ) -> Dict[str, Any]:
        t_start = time.monotonic()
        doc_log = f"document_id={document_id} " if document_id else ""

        # 1. Physical page count of the PDF via PyMuPDF (guaranteed accurate)
        actual_page_count = 1
        page_texts: Dict[int, str] = {}
        try:
            with fitz.open(file_path) as pdf_doc:
                actual_page_count = len(pdf_doc)
                page_texts = self._extract_page_texts(pdf_doc)
        except Exception as e:
            app_logger.warning(f"Could not read PDF via PyMuPDF for {file_path}: {e}")

        app_logger.info(f"[INDEX] {doc_log}stage=upload pages={actual_page_count}")
        if on_stage:
            on_stage("Processing PDF", 15)

        page_images = self._extract_images(file_path)
        t_images = time.monotonic() - t_start

        if not self.api_keys:
            app_logger.warning(f"[INDEX] {doc_log}stage=fallback method=pymupdf reason='no_api_keys'")
            if on_stage:
                on_stage("Building document structure", 50)
            t_parse_start = time.monotonic()
            nodes = self._local_pdf_parse(file_path)
            t_parse = time.monotonic() - t_parse_start
            tree = self._inject_images(nodes, page_images)
            tree = self._inject_page_boundaries(tree, page_texts, actual_page_count)
            sec_count = self._count_sections(tree)
            total_time = time.monotonic() - t_start
            app_logger.info(f"[INDEX] {doc_log}stage=ready duration={total_time:.2f}s method=pymupdf pages={actual_page_count} sections={sec_count}")
            return {
                "tree": tree,
                "page_count": actual_page_count,
                "section_count": sec_count,
                "indexing_method": "pymupdf",
                "duration": round(total_time, 2),
                "failure_reason": "No PageIndex API keys configured",
            }

        last_error = None

        for key in self.api_keys:
            try:
                client = PageIndexClient(api_key=key)
                app_logger.info(f"[INDEX] {doc_log}stage=pageindex_submit key=...{key[-4:]}")
                if on_stage:
                    on_stage("Building document structure", 25)

                t_sub_start = time.monotonic()
                submit_res = client.submit_document(file_path)
                cloud_doc_id = submit_res.get("doc_id")
                t_submit = time.monotonic() - t_sub_start

                if not cloud_doc_id:
                    raise Exception(f"No doc_id returned by PageIndex: {submit_res}")

                # Poll until PageIndex tree generation is completed
                max_retries = 40  # up to ~75-85 seconds
                is_completed = False
                t_poll_start = time.monotonic()

                for attempt in range(max_retries):
                    app_logger.info(f"[INDEX] {doc_log}stage=pageindex_poll attempt={attempt + 1}/{max_retries}")
                    if on_stage:
                        on_stage("Building document structure", min(75, 25 + int(attempt * 1.3)))

                    meta = client.get_document(cloud_doc_id)
                    status = meta.get("status")
                    if status == "completed":
                        is_completed = True
                        app_logger.info(f"[INDEX] {doc_log}stage=pageindex_poll completed in {time.monotonic() - t_poll_start:.2f}s")
                        break
                    elif status == "failed":
                        raise Exception(f"PageIndex document processing failed for {cloud_doc_id}")

                    # Adaptive sleep: start fast (1.0s) then back off slightly (up to 2.2s)
                    sleep_sec = min(1.0 + attempt * 0.08, 2.2)
                    time.sleep(sleep_sec)

                t_poll = time.monotonic() - t_poll_start

                if is_completed:
                    app_logger.info(f"[INDEX] {doc_log}stage=validation")
                    if on_stage:
                        on_stage("Validating index", 80)

                    t_fetch_start = time.monotonic()
                    tree_res = client.get_tree(cloud_doc_id, node_summary=True)
                    raw_tree = tree_res.get("result", [])
                    t_fetch = time.monotonic() - t_fetch_start

                    if self._is_valid_tree(raw_tree):
                        tree = self._inject_images(raw_tree, page_images)
                        tree = self._inject_page_boundaries(tree, page_texts, actual_page_count)
                        sec_count = self._count_sections(tree)
                        total_time = time.monotonic() - t_start
                        app_logger.info(
                            f"[INDEX] {doc_log}stage=ready duration={total_time:.2f}s "
                            f"method=pageindex pages={actual_page_count} sections={sec_count} "
                            f"submit={t_submit:.2f}s poll={t_poll:.2f}s"
                        )
                        return {
                            "tree": tree,
                            "page_count": actual_page_count,
                            "section_count": sec_count,
                            "indexing_method": "pageindex",
                            "duration": round(total_time, 2),
                            "failure_reason": None,
                        }
                    else:
                        app_logger.warning(f"[INDEX] {doc_log}stage=validation status=invalid_tree")
                        last_error = "PageIndex returned empty or incomplete tree"
                else:
                    last_error = "PageIndex indexing poll timed out"

                # If reached here, PageIndex produced invalid tree or timed out -> fallback
                app_logger.info(f"[INDEX] {doc_log}stage=fallback method=pymupdf reason='{last_error}'")
                if on_stage:
                    on_stage("Building document structure", 85)

                t_parse_start = time.monotonic()
                nodes = self._local_pdf_parse(file_path)
                t_parse = time.monotonic() - t_parse_start
                tree = self._inject_images(nodes, page_images)
                tree = self._inject_page_boundaries(tree, page_texts, actual_page_count)
                sec_count = self._count_sections(tree)
                total_time = time.monotonic() - t_start
                app_logger.info(
                    f"[INDEX] {doc_log}stage=ready duration={total_time:.2f}s "
                    f"method=pymupdf pages={actual_page_count} sections={sec_count} reason='{last_error}'"
                )
                return {
                    "tree": tree,
                    "page_count": actual_page_count,
                    "section_count": sec_count,
                    "indexing_method": "pymupdf",
                    "duration": round(total_time, 2),
                    "failure_reason": last_error,
                }

            except Exception as e:
                error_msg = str(e)
                app_logger.warning(f"PageIndex attempt failed with key {key[-4:]}: {error_msg}")
                last_error = error_msg
                if "LimitReached" in error_msg:
                    continue
                else:
                    break

        app_logger.info(f"[INDEX] {doc_log}stage=fallback method=pymupdf reason='{last_error}'")
        if on_stage:
            on_stage("Building document structure", 85)

        t_parse_start = time.monotonic()
        nodes = self._local_pdf_parse(file_path)
        tree = self._inject_images(nodes, page_images)
        tree = self._inject_page_boundaries(tree, page_texts, actual_page_count)
        sec_count = self._count_sections(tree)
        total_time = time.monotonic() - t_start
        app_logger.info(
            f"[INDEX] {doc_log}stage=ready duration={total_time:.2f}s "
            f"method=pymupdf pages={actual_page_count} sections={sec_count} reason='{last_error}'"
        )
        return {
            "tree": tree,
            "page_count": actual_page_count,
            "section_count": sec_count,
            "indexing_method": "pymupdf",
            "duration": round(total_time, 2),
            "failure_reason": last_error,
        }
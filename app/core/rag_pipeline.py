import asyncio
import re
import time
from typing import Optional, List, Dict, Any, Tuple
from utils.logger import app_logger
from core.bm25 import BM25NodeIndex


class VectorlessRAG:
    def __init__(self, llm_service):
        self.llm = llm_service
        self._bm25_cache: Dict[str, BM25NodeIndex] = {}

    @staticmethod
    def _parse_paragraphs_with_page_markers(text: str, default_page: Any) -> List[Dict[str, Any]]:
        """
        Parses raw node text into paragraphs, tracking the nearest preceding --- [PAGE X] --- marker.
        Returns a list of dicts: [{"page": int_or_str, "text": str}]
        """
        marker_pattern = re.compile(r'---\s*\[PAGE\s+(\d+)\]\s*---', re.IGNORECASE)
        try:
            curr_page = int(default_page) if str(default_page).isdigit() else default_page
        except Exception:
            curr_page = default_page or 1

        blocks = re.split(r'\n\s*\n', text.strip())
        paragraphs: List[Dict[str, Any]] = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.split('\n')
            current_lines = []
            for line in lines:
                clean_line = line.strip()
                m = marker_pattern.search(clean_line)
                if m:
                    # Flush accumulated text before the new marker
                    if current_lines:
                        p_txt = "\n".join(current_lines).strip()
                        if p_txt:
                            paragraphs.append({"page": curr_page, "text": p_txt})
                        current_lines = []
                    curr_page = int(m.group(1))
                    remainder = marker_pattern.sub('', clean_line).strip()
                    if remainder:
                        current_lines.append(remainder)
                else:
                    current_lines.append(line)

            if current_lines:
                p_txt = "\n".join(current_lines).strip()
                if p_txt:
                    paragraphs.append({"page": curr_page, "text": p_txt})

        return paragraphs

    @classmethod
    def prune_sub_nodes(
        cls,
        query: str,
        nodes: List[Dict[str, Any]],
        deep_analysis: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Sub-Node Paragraph Pruning:
        Splits long node contents (> 1200 characters) into paragraphs while preserving
        the nearest preceding --- [PAGE X] --- marker for each paragraph.
        Scores each paragraph against user query terms and selects only the top relevant
        paragraphs, reassembled in original reading order.
        """
        if not nodes:
            return []

        min_char_threshold = 1200
        target_budget = 3600 if deep_analysis else 1800
        max_paras = 8 if deep_analysis else 4

        # Extract meaningful query keywords
        q_clean = query.lower()
        raw_words = re.findall(r'[\w\-]+', q_clean)
        stopwords = {
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'what', 'which', 'where',
            'when', 'how', 'why', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'does',
            'did', 'about', 'some', 'any', 'each', 'into', 'over', 'after', 'before', 'such',
            'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي',
            'ما', 'ماذا', 'كيف', 'لماذا', 'أين', 'متى', 'كم', 'هل', 'هو', 'هي', 'أن', 'إن', 'كان'
        }
        query_terms = [w for w in raw_words if len(w) >= 2 and w not in stopwords]
        if not query_terms:
            query_terms = [w for w in raw_words if len(w) >= 2]

        pruned_nodes: List[Dict[str, Any]] = []
        for node in nodes:
            node_copy = dict(node)
            raw_text = (node.get("text") or node.get("content") or "").strip()
            if len(raw_text) <= min_char_threshold:
                pruned_nodes.append(node_copy)
                continue

            default_page = node.get("page_index") or node.get("page") or 1
            paragraphs = cls._parse_paragraphs_with_page_markers(raw_text, default_page)
            if len(paragraphs) <= 1:
                pruned_nodes.append(node_copy)
                continue

            # Score each paragraph
            scored_paras = []
            for idx, p in enumerate(paragraphs):
                p_text = p["text"]
                p_lower = p_text.lower()
                p_tokens = set(re.findall(r'[\w\-]+', p_lower))

                score = 0.0
                # 1. Keyword overlap
                overlap = sum(1 for t in query_terms if t in p_tokens or t in p_lower)
                score += overlap * 3.0

                # 2. Term frequency bonus
                if query_terms:
                    tf_sum = sum(p_lower.count(t) for t in query_terms)
                    score += min(tf_sum * 0.4, 4.0)

                # 3. Exact multi-word phrase matching
                if len(query_terms) >= 2:
                    for i in range(len(query_terms) - 1):
                        bigram = f"{query_terms[i]} {query_terms[i+1]}"
                        if bigram in p_lower:
                            score += 4.0

                # 4. Uppercase / acronym / numerical match boost
                for w in raw_words:
                    if (w.isupper() or any(c.isdigit() for c in w)) and len(w) >= 2 and w in p_lower:
                        score += 2.5

                # 5. Position bias for leading introduction/definition paragraph
                if idx == 0:
                    score += 0.8

                scored_paras.append((idx, p, score))

            # Sort by score descending to choose best paragraphs
            scored_paras.sort(key=lambda x: x[2], reverse=True)

            selected_indices = set()
            accumulated_chars = 0

            # Always take the #1 highest scoring paragraph
            top_idx, top_para, top_score = scored_paras[0]
            selected_indices.add(top_idx)
            accumulated_chars += len(top_para["text"])

            for idx, p, sc in scored_paras[1:]:
                if len(selected_indices) >= max_paras or accumulated_chars >= target_budget:
                    break
                if sc > 0 or len(selected_indices) < 2:
                    selected_indices.add(idx)
                    accumulated_chars += len(p["text"])

            # Reassemble selected paragraphs in original document order
            selected_paras = [paragraphs[i] for i in sorted(selected_indices)]

            reconstructed_parts = []
            last_page = None
            for p in selected_paras:
                p_page = p["page"]
                p_txt = p["text"]
                if p_page != last_page:
                    reconstructed_parts.append(f"--- [PAGE {p_page}] ---")
                    last_page = p_page
                reconstructed_parts.append(p_txt)

            new_text = "\n\n".join(reconstructed_parts)
            node_copy["text"] = new_text
            node_copy["is_pruned"] = True
            node_copy["original_chars"] = len(raw_text)
            node_copy["pruned_chars"] = len(new_text)

            app_logger.info(
                f"[PRUNE] node='{node.get('title')}' original={len(raw_text)} chars -> "
                f"pruned={len(new_text)} chars ({len(selected_paras)}/{len(paragraphs)} paras)"
            )
            pruned_nodes.append(node_copy)

        return pruned_nodes

    @staticmethod
    def _get_doc_nodes(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts flattened node dictionaries containing node_id, title, summary, text."""
        node_map = doc.get("node_map") or {}
        flat_sections = doc.get("flat_sections") or []
        if flat_sections and node_map:
            nodes = [node_map[s["node_id"]] for s in flat_sections if s.get("node_id") in node_map]
            if nodes:
                return nodes
        if node_map:
            return list(node_map.values())

        # Collect from tree recursively
        tree = doc.get("tree") or []
        collected = []
        def collect(items):
            for it in items:
                if isinstance(it, dict):
                    collected.append(it)
                    if it.get("nodes"):
                        collect(it["nodes"])
        collect(tree)
        return collected

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[Tuple[List[str], float]],
        rrf_k: int = 60,
        top_k: int = 4,
    ) -> List[Tuple[str, float]]:
        """
        Weighted Reciprocal Rank Fusion (RRF):
        Combines multiple ranked lists into a single consensus ranking.
        score(d) = sum_L (w_L / (rrf_k + rank_L(d)))
        """
        rrf_scores: Dict[str, float] = {}
        for items, weight in ranked_lists:
            for rank, item_id in enumerate(items, start=1):
                if not item_id:
                    continue
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (weight / (rrf_k + rank))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

    def bm25_search_nodes(
        self,
        query: str,
        nodes: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Executes Okapi BM25 lexical search with in-memory index caching.
        Returns (matches, latency_ms).
        """
        t0 = time.monotonic()
        cache_key = doc_id or str(id(nodes))
        index = self._bm25_cache.get(cache_key)
        if index is None or index.N != len(nodes):
            index = BM25NodeIndex(nodes)
            if doc_id:
                self._bm25_cache[cache_key] = index

        matches = index.search(query, top_k=top_k)
        elapsed_ms = (time.monotonic() - t0) * 1000
        return matches, elapsed_ms

    async def answer_multi_doc_query_stream(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        session_summary: Optional[str] = None,
        global_knowledge: Optional[str] = None,
        deep_analysis: bool = False,
    ):
        """
        Executes Vectorless RAG across multiple documents concurrently.
        Each item in `docs`:
          {
            "doc_id": str,
            "filename": str,
            "tree": list,
            "node_map": dict,
            "flat_sections": list
          }
        """
        try:
            t_retrieval_start = time.monotonic()
            all_selected_nodes = []

            # Search across each document tree concurrently
            async def search_single_doc(doc):
                tree = doc.get("tree") or []
                node_map = doc.get("node_map")
                flat_sections = doc.get("flat_sections")
                doc_name = doc.get("filename") or "Document"
                doc_id = doc.get("doc_id")

                doc_nodes = self._get_doc_nodes(doc)

                # Parallel Retrieval: LLM Hierarchical Tree Search + Lexical BM25
                async def run_bm25():
                    return self.bm25_search_nodes(query, doc_nodes, doc_id=doc_id, top_k=5)

                llm_task = self.llm.llm_tree_search(
                    query, tree, chat_history=chat_history, flat_sections=flat_sections
                )
                bm25_task = run_bm25()

                search_res, (bm25_matches, bm25_latency) = await asyncio.gather(
                    llm_task, bm25_task, return_exceptions=False
                )

                llm_node_ids = search_res.get("node_list", [])
                bm25_node_ids = [nid for nid, sc in bm25_matches if sc > 0]

                # Weighted Reciprocal Rank Fusion: 0.6 LLM Tree Search, 0.4 BM25
                fused_results = self.reciprocal_rank_fusion(
                    ranked_lists=[(llm_node_ids, 0.6), (bm25_node_ids, 0.4)],
                    top_k=4 if deep_analysis else 3,
                )
                fused_node_ids = [nid for nid, _ in fused_results]

                app_logger.info(
                    f"[HYBRID-RETRIEVAL] doc='{doc_name}' llm_nodes={llm_node_ids} "
                    f"bm25_nodes={bm25_node_ids} fused={fused_node_ids} bm25_latency={bm25_latency:.2f}ms"
                )

                nodes = self.llm.find_nodes_by_ids(tree, fused_node_ids, node_map=node_map)

                # Fallback to leading sections for overview/comparative queries if no nodes matched
                if not nodes:
                    overview_triggers = [
                        "دا", "ده", "دي", "لخص", "ملخص", "اشرح", "بحث", "مقارنة", "قارن",
                        "summary", "summarize", "overview", "compare", "comparison", "what is this", "about"
                    ]
                    normalized_q = query.strip().lower()
                    is_short_or_overview = len(normalized_q) <= 10 or any(t in normalized_q for t in overview_triggers)
                    if is_short_or_overview:
                        top_nodes = []
                        def collect_leading(items):
                            for item in items:
                                if isinstance(item, dict):
                                    top_nodes.append(item)
                                    if len(top_nodes) >= 3:
                                        return
                                    if item.get("nodes"):
                                        collect_leading(item["nodes"])
                        collect_leading(tree)
                        nodes = top_nodes

                tagged_nodes = []
                for n in nodes:
                    n_copy = dict(n)
                    n_copy["doc_name"] = doc_name
                    n_copy["filename"] = doc_name
                    n_copy["doc_id"] = doc_id
                    tagged_nodes.append(n_copy)
                return tagged_nodes

            # Run searches concurrently across all attached documents
            search_tasks = [search_single_doc(d) for d in docs]
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, list):
                    all_selected_nodes.extend(r)
                elif isinstance(r, Exception):
                    app_logger.error(f"[MULTI-RAG] Document tree search error: {r}")

            t_retrieval = time.monotonic() - t_retrieval_start
            doc_names_list = [d.get("filename", "Doc") for d in docs]
            doc_names_str = ", ".join(doc_names_list)

            app_logger.info(
                f"[PERF] multi_rag retrieval_time={t_retrieval:.2f}s "
                f"docs_count={len(docs)} nodes_matched={len(all_selected_nodes)} deep_analysis={deep_analysis}"
            )

            # Signal retrieval completion stage to frontend
            yield f'<!--STAGE:{{"stage":"retrieved","duration":{t_retrieval:.2f},"nodes":{len(all_selected_nodes)},"docs":{len(docs)}}}-->\n'

            # FastMCP Web Search fallback:
            # Trigger web search if document tree has NO matching nodes OR if user explicitly asks for web/latest info
            web_results = None
            web_triggers = [
                "ابحث", "النت", "الويب", "روابط", "لينكات", "أحدث", "جديد", "أخبار",
                "web", "search", "latest", "news", "online", "links", "url", "google", "browse"
            ]
            q_lower = query.lower()
            is_explicit_web_request = any(t in q_lower for t in web_triggers)
            is_doc_empty_for_query = len(all_selected_nodes) == 0

            if is_doc_empty_for_query or is_explicit_web_request:
                try:
                    import json
                    from mcp_server.web_search_mcp import execute_web_search
                    yield f'<!--STAGE:{{"stage":"web_search","query":{json.dumps(query)}}}-->\n'
                    t_web_start = time.monotonic()
                    web_results = await execute_web_search(query, max_results=5)
                    t_web = time.monotonic() - t_web_start
                    app_logger.info(f"[FastMCP] Web search completed in {t_web:.2f}s with {len(web_results or [])} results")
                    yield f'<!--STAGE:{{"stage":"web_search_done","duration":{t_web:.2f},"results":{len(web_results or [])}}}-->\n'
                except Exception as web_err:
                    app_logger.warning(f"[FastMCP] Web search fallback failed: {web_err}")

            # Apply Sub-Node Paragraph Pruning before synthesis
            pruned_nodes = self.prune_sub_nodes(
                query=query,
                nodes=all_selected_nodes,
                deep_analysis=deep_analysis,
            )

            # Stream final synthesis with multi-document attribution and web links
            async for chunk in self.llm.generate_answer_stream(
                query=query,
                nodes=pruned_nodes,
                chat_history=chat_history,
                doc_name=doc_names_str,
                session_summary=session_summary,
                global_knowledge=global_knowledge,
                deep_analysis=deep_analysis,
                web_results=web_results,
            ):
                yield chunk

        except Exception as e:
            app_logger.error(f"Error in multi-doc RAG pipeline: {str(e)}", exc_info=True)
            is_arabic = any("\u0600" <= c <= "\u06ff" for c in query)
            if is_arabic:
                yield "حدث خطأ أثناء استرجاع المعلومات من المستندات المفهرسة. يرجى إعادة المحاولة."
            else:
                yield "An error occurred while retrieving information from the indexed documents. Please try again."

    async def answer_query_stream(
        self,
        query: str,
        tree: list,
        chat_history: Optional[List[Dict[str, str]]] = None,
        doc_name: Optional[str] = None,
        node_map: Optional[Dict[str, Any]] = None,
        flat_sections: Optional[List[Dict[str, Any]]] = None,
        deep_analysis: bool = False,
        session_summary: Optional[str] = None,
        global_knowledge: Optional[str] = None,
    ):
        """Single-document backward-compatible wrapper."""
        doc_obj = {
            "doc_id": "single",
            "filename": doc_name or "Uploaded Document",
            "tree": tree,
            "node_map": node_map,
            "flat_sections": flat_sections,
        }
        async for chunk in self.answer_multi_doc_query_stream(
            query=query,
            docs=[doc_obj],
            chat_history=chat_history,
            session_summary=session_summary,
            global_knowledge=global_knowledge,
            deep_analysis=deep_analysis,
        ):
            yield chunk
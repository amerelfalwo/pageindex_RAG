import asyncio
import re
import time
from typing import Optional, List, Dict, Any, Tuple
from utils.logger import app_logger
from core.bm25 import BM25NodeIndex
from core.intent_classifier import intent_classifier, QueryIntent


class VectorlessRAG:
    def __init__(self, llm_service, memory_service=None, storage_instance=None):
        self.llm = llm_service
        self.memory_service = memory_service
        self.storage = storage_instance
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
            if not isinstance(node, dict):
                continue
            node_copy = dict(node)
            # Preserve injected executive knowledge dossiers and root chapter summaries without paragraph loss
            if node.get("is_dossier") or node.get("is_root_chapter"):
                pruned_nodes.append(node_copy)
                continue

            raw_text = (node.get("text") or node.get("content") or "").strip()
            marker_pattern = re.compile(r'---\s*\[PAGE\s+(\d+)\]\s*---', re.IGNORECASE)
            found_markers = [int(m) for m in marker_pattern.findall(raw_text)]

            if len(raw_text) <= min_char_threshold:
                if found_markers:
                    node_copy["active_page"] = found_markers[0]
                    node_copy["active_pages"] = sorted(list(set(found_markers)))
                else:
                    fallback_p = node.get("page_index") or node.get("page") or 1
                    node_copy["active_page"] = fallback_p
                    node_copy["active_pages"] = [fallback_p]
                # Lexical overlap estimation for short node
                raw_lower = raw_text.lower()
                node_copy["relevance_score"] = float(sum(3.0 for t in query_terms if t in raw_lower))
                pruned_nodes.append(node_copy)
                continue

            default_page = node.get("page_index") or node.get("page") or 1
            paragraphs = cls._parse_paragraphs_with_page_markers(raw_text, default_page)
            if len(paragraphs) <= 1:
                if found_markers:
                    node_copy["active_page"] = found_markers[0]
                    node_copy["active_pages"] = sorted(list(set(found_markers)))
                else:
                    fallback_p = default_page
                    node_copy["active_page"] = fallback_p
                    node_copy["active_pages"] = [fallback_p]
                raw_lower = raw_text.lower()
                node_copy["relevance_score"] = float(sum(3.0 for t in query_terms if t in raw_lower))
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

            # Pin winning inner active page and spanned pages
            winning_page = top_para.get("page") or default_page
            selected_pages = [p["page"] for p in selected_paras if p.get("page") is not None]
            unique_pages = sorted(list(set(selected_pages)), key=lambda x: int(x) if str(x).isdigit() else 0)
            node_copy["active_page"] = winning_page
            node_copy["active_pages"] = unique_pages if unique_pages else [winning_page]
            node_copy["relevance_score"] = top_score

            app_logger.info(
                f"[PRUNE] node='{node.get('title')}' active_page={winning_page} original={len(raw_text)} chars -> "
                f"pruned={len(new_text)} chars ({len(selected_paras)}/{len(paragraphs)} paras)"
            )
            pruned_nodes.append(node_copy)

        # Post-Pruning Relevance Filter: drop trailing noise nodes with near-zero relevance compared to rank-1
        if len(pruned_nodes) > 1:
            best_score = max(
                (n.get("relevance_score", 0.0) for n in pruned_nodes if not n.get("is_dossier") and not n.get("is_root_chapter")),
                default=0.0
            )
            if best_score >= 4.0:
                filtered_nodes = []
                for idx, n in enumerate(pruned_nodes):
                    if idx == 0 or n.get("is_dossier") or n.get("is_root_chapter"):
                        filtered_nodes.append(n)
                        continue
                    sc = n.get("relevance_score", 0.0)
                    if sc >= 0.22 * best_score:
                        filtered_nodes.append(n)
                    else:
                        app_logger.info(
                            f"[PRUNE-FILTER] Dropped trailing node '{n.get('title')}' with low relevance "
                            f"({sc:.1f} vs best {best_score:.1f})"
                        )
                pruned_nodes = filtered_nodes

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

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        ranked_lists: List[Tuple[List[str], float]],
        rrf_k: int = 60,
        top_k: int = 4,
        top_p: Optional[float] = None,
        min_p: float = 0.18,
        min_k: int = 1,
        temperature: float = 0.15,
        is_multi_hop: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Weighted Reciprocal Rank Fusion (RRF) with Nucleus (top_p) Cumulative Probability Cutoff:
        1. Combines ranked lists into consensus score(d) = sum_L (w_L / (rrf_k + rank_L(d))).
        2. Normalizes candidate scores into probability distribution P(node_i) via temperature-scaled Softmax.
        3. Dynamically selects minimal sufficient set of nodes until cumulative probability >= top_p.
           Drops noise trailing nodes where P(node_i) < min_p (preventing context dilution).
        4. Bounded by min_k and top_k (allows min_k=2 for explicit multi-hop queries).
        """
        import math

        rrf_scores: Dict[str, float] = {}
        for items, weight in ranked_lists:
            for rank, item_id in enumerate(items, start=1):
                if not item_id:
                    continue
                rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (weight / (rrf_k + rank))

        if not rrf_scores:
            return []

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_items) <= 1:
            return sorted_items

        if not top_p or top_p <= 0:
            return sorted_items[:top_k]

        candidates = sorted_items[:max(top_k, 3)]
        max_score = candidates[0][1]
        if max_score <= 0:
            return candidates[:1]

        # Normalized relative to top candidate (norm_score = 1.0)
        norm_scores = [sc / max_score for _, sc in candidates]

        # Temperature-scaled Softmax
        t = max(temperature, 0.01)
        exp_vals = [math.exp((ns - 1.0) / t) for ns in norm_scores]
        sum_exp = sum(exp_vals)
        probs = [ev / sum_exp for ev in exp_vals]

        # Effective min_k: for explicit multi-hop queries, allow min_k = 2 if candidate 2 has relevance
        effective_min_k = 2 if is_multi_hop and len(candidates) >= 2 else min_k

        selected: List[Tuple[str, float]] = []
        cum_p = 0.0

        for idx, (item, p) in enumerate(zip(candidates, probs)):
            if idx == 0:
                selected.append(item)
                cum_p += p
                continue

            if len(selected) >= top_k:
                break

            if len(selected) < effective_min_k and p >= 0.12:
                selected.append(item)
                cum_p += p
                continue

            if cum_p < top_p and p >= min_p:
                selected.append(item)
                cum_p += p
            else:
                break

        return selected

    def _get_document_dossier(self, doc_id: Optional[str], doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieves pre-computed executive dossier from document payload or persistent storage."""
        if not doc_id:
            return None
        if doc.get("dossier"):
            return doc["dossier"]
        try:
            target_storage = self.storage
            if target_storage is None:
                from core.storage import storage as default_storage
                target_storage = default_storage
            return target_storage.get_knowledge_dossier(doc_id)
        except Exception as e:
            app_logger.warning(f"[RAG-DOSSIER] Could not fetch dossier for doc_id={doc_id}: {e}")
            return None

    @staticmethod
    def _create_fallback_dossier_from_tree(tree: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
        """Synthesizes an on-the-fly fallback dossier from top tree nodes when background indexing is still pending."""
        top_sections = tree[:5]
        topics = [s.get("title", f"Section {i+1}") for i, s in enumerate(top_sections) if s.get("title")]
        summary_parts = []
        for s in top_sections:
            title = s.get("title", "")
            summ = s.get("summary") or (s.get("text") or s.get("content") or "")[:250]
            if summ:
                summary_parts.append(f"{title}: {summ.strip()}")
        exec_summary = " | ".join(summary_parts) if summary_parts else f"Outline preview of {filename}."
        return {
            "executive_summary": exec_summary,
            "key_entities": topics[:4],
            "key_findings": [f"Covers {t}" for t in topics[:3]],
            "topics": topics,
        }

    @staticmethod
    def _create_dossier_node(dossier: Dict[str, Any], doc_name: str, doc_id: Optional[str]) -> Dict[str, Any]:
        """Creates a standardized context node representing the executive dossier with Page 1 citation."""
        lines = ["--- [PAGE 1] ---"]
        exec_summary = dossier.get("executive_summary") or ""
        if exec_summary:
            lines.append(f"Executive Summary:\n{exec_summary}")
        findings = dossier.get("key_findings") or []
        if findings:
            lines.append("Core Findings:\n" + "\n".join([f"- {f}" for f in findings]))
        entities = dossier.get("key_entities") or []
        if entities:
            lines.append(f"Key Entities & Terms: {', '.join(entities)}")
        topics = dossier.get("topics") or []
        if topics:
            lines.append(f"Document Taxonomy & Topics: {', '.join(topics)}")

        return {
            "node_id": f"dossier_{doc_id or 'doc'}",
            "title": "Executive Knowledge Dossier & Core Findings",
            "page_index": 1,
            "doc_name": doc_name,
            "filename": doc_name,
            "doc_id": doc_id,
            "text": "\n\n".join(lines),
            "is_dossier": True,
        }

    @staticmethod
    def _extract_root_chapter_nodes(
        tree: List[Dict[str, Any]], doc_name: str, doc_id: Optional[str], max_chapters: int = 8
    ) -> List[Dict[str, Any]]:
        """Extracts top-level root chapters with summaries and page markers for global architectural context."""
        root_nodes = []
        for idx, chapter in enumerate(tree[:max_chapters]):
            if not isinstance(chapter, dict):
                continue
            ch_title = chapter.get("title") or f"Chapter {idx+1}"
            ch_page = chapter.get("page_index") or chapter.get("page") or 1
            ch_summary = chapter.get("summary") or ""
            ch_text = (chapter.get("text") or chapter.get("content") or "")[:1200]
            content_preview = ch_summary if ch_summary else ch_text

            if f"--- [PAGE {ch_page}] ---" not in content_preview:
                content_preview = f"--- [PAGE {ch_page}] ---\n{content_preview}"

            root_nodes.append({
                "node_id": chapter.get("node_id", f"root_{idx}_{doc_id or 'doc'}"),
                "title": ch_title,
                "page_index": ch_page,
                "doc_name": doc_name,
                "filename": doc_name,
                "doc_id": doc_id,
                "text": content_preview,
                "is_root_chapter": True,
            })
        return root_nodes

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

            # 1. Query Intent Classification (Global, Taxonomy/Architecture, Cross-Doc, or Specific Fact)
            intent_res = intent_classifier.classify(query, doc_count=len(docs))
            is_global = intent_res["is_global_or_summary"]
            app_logger.info(
                f"[QUERY-INTENT] query='{query[:50]}' intent={intent_res['intent']} "
                f"is_global={is_global} triggers={intent_res['matched_triggers']}"
            )

            # Auto-fetch global knowledge summary if missing and query is global/cross-doc
            if is_global and not global_knowledge:
                try:
                    target_storage = self.storage
                    if target_storage is None:
                        from core.storage import storage as default_storage
                        target_storage = default_storage
                    g_summary = target_storage.get_global_knowledge_summary()
                    if g_summary:
                        global_knowledge = g_summary
                except Exception as ge:
                    app_logger.debug(f"Could not load global knowledge summary: {ge}")

            # Search across each document tree concurrently
            async def search_single_doc(doc):
                tree = doc.get("tree") or []
                node_map = doc.get("node_map")
                flat_sections = doc.get("flat_sections")
                doc_name = doc.get("filename") or "Document"
                doc_id = doc.get("doc_id")

                doc_nodes = self._get_doc_nodes(doc)

                # Dossier & Root Chapter Context Injection for Global / Summary / Taxonomy queries
                injected_nodes = []
                if is_global:
                    dossier = self._get_document_dossier(doc_id, doc)
                    if not dossier and tree:
                        dossier = self._create_fallback_dossier_from_tree(tree, doc_name)
                    if dossier:
                        injected_nodes.append(self._create_dossier_node(dossier, doc_name, doc_id))
                    root_chapters = self._extract_root_chapter_nodes(tree, doc_name, doc_id)
                    injected_nodes.extend(root_chapters)

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

                # Weighted Reciprocal Rank Fusion with Nucleus (top_p) Cutoff: 0.6 LLM Tree Search, 0.4 BM25
                is_multi_hop_query = (
                    "multi" in query.lower()
                    or "between" in query.lower()
                    or "connect" in query.lower()
                    or ("and" in query.lower() and ("relat" in query.lower() or "compar" in query.lower() or "synthes" in query.lower()))
                )
                fused_results = self.reciprocal_rank_fusion(
                    ranked_lists=[(llm_node_ids, 0.6), (bm25_node_ids, 0.4)],
                    top_k=4 if deep_analysis else 3,
                    top_p=0.75,
                    min_p=0.18,
                    is_multi_hop=is_multi_hop_query or is_global,
                )
                fused_node_ids = [nid for nid, _ in fused_results]

                app_logger.info(
                    f"[HYBRID-RETRIEVAL] doc='{doc_name}' llm_nodes={llm_node_ids} "
                    f"bm25_nodes={bm25_node_ids} fused={fused_node_ids} bm25_latency={bm25_latency:.2f}ms"
                )

                retrieved_nodes = self.llm.find_nodes_by_ids(tree, fused_node_ids, node_map=node_map)

                # Fallback to leading sections for overview/comparative queries if no nodes matched
                if not retrieved_nodes and not injected_nodes:
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
                        retrieved_nodes = top_nodes

                tagged_nodes = []
                for n in retrieved_nodes:
                    n_copy = dict(n)
                    n_copy["doc_name"] = doc_name
                    n_copy["filename"] = doc_name
                    n_copy["doc_id"] = doc_id
                    tagged_nodes.append(n_copy)

                # Prepend injected executive dossier & root chapters before retrieved sub-nodes
                if injected_nodes:
                    injected_ids = {n["node_id"] for n in injected_nodes}
                    final_doc_nodes = list(injected_nodes)
                    for n in tagged_nodes:
                        if n.get("node_id") not in injected_ids:
                            final_doc_nodes.append(n)
                    return final_doc_nodes

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

            # FastMCP Academic / arXiv Research tool:
            # Trigger arXiv search when user asks for research papers, citations, or literature reviews
            arxiv_results = None
            is_academic_query = intent_res.get("is_academic", False)
            academic_triggers = [
                "arxiv", "paper", "papers", "research paper", "literature review",
                "citations", "academic", "preprint", "journal", "أوراق بحثية", "ورقة بحثية", "أبحاث"
            ]
            if not is_academic_query:
                is_academic_query = any(at in q_lower for at in academic_triggers)

            if is_academic_query:
                try:
                    import json
                    from mcp_server.research_tools import search_arxiv_papers_async
                    yield f'<!--STAGE:{{"stage":"arxiv_search","query":{json.dumps(query)}}}-->\n'
                    t_arxiv_start = time.monotonic()
                    arxiv_results = await search_arxiv_papers_async(query, max_results=4)
                    t_arxiv = time.monotonic() - t_arxiv_start
                    app_logger.info(f"[arXiv] Academic search completed in {t_arxiv:.2f}s with {len(arxiv_results or [])} papers")
                    yield f'<!--STAGE:{{"stage":"arxiv_search_done","duration":{t_arxiv:.2f},"results":{len(arxiv_results or [])}}}-->\n'
                except Exception as arxiv_err:
                    app_logger.warning(f"[arXiv] Academic search failed: {arxiv_err}")

            # Apply Sub-Node Paragraph Pruning before synthesis
            pruned_nodes = self.prune_sub_nodes(
                query=query,
                nodes=all_selected_nodes,
                deep_analysis=deep_analysis,
            )

            # Stream final synthesis with multi-document attribution, web links, and arXiv citations
            import inspect
            llm_sig = inspect.signature(self.llm.generate_answer_stream)
            call_kwargs = {
                "query": query,
                "nodes": pruned_nodes,
                "chat_history": chat_history,
                "doc_name": doc_names_str,
                "session_summary": session_summary,
                "global_knowledge": global_knowledge,
                "deep_analysis": deep_analysis,
                "web_results": web_results,
            }
            if "arxiv_results" in llm_sig.parameters:
                call_kwargs["arxiv_results"] = arxiv_results

            async for chunk in self.llm.generate_answer_stream(**call_kwargs):
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
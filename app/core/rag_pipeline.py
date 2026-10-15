import asyncio
import time
from typing import Optional, List, Dict, Any
from utils.logger import app_logger


class VectorlessRAG:
    def __init__(self, llm_service):
        self.llm = llm_service

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

                search_res = await self.llm.llm_tree_search(
                    query, tree, chat_history=chat_history, flat_sections=flat_sections
                )
                node_ids = search_res.get("node_list", [])
                nodes = self.llm.find_nodes_by_ids(tree, node_ids, node_map=node_map)

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
                    n_copy["doc_id"] = doc.get("doc_id")
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

            # Stream final synthesis with multi-document attribution
            async for chunk in self.llm.generate_answer_stream(
                query=query,
                nodes=all_selected_nodes,
                chat_history=chat_history,
                doc_name=doc_names_str,
                session_summary=session_summary,
                global_knowledge=global_knowledge,
                deep_analysis=deep_analysis,
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
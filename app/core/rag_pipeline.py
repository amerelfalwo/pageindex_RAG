import time
from typing import Optional, List, Dict, Any
from utils.logger import app_logger

class VectorlessRAG:
    def __init__(self, llm_service):
        self.llm = llm_service

    async def answer_query_stream(
        self,
        query: str,
        tree: list,
        chat_history: Optional[List[Dict[str, str]]] = None,
        doc_name: Optional[str] = None,
        node_map: Optional[Dict[str, Any]] = None,
        flat_sections: Optional[List[Dict[str, Any]]] = None,
        deep_analysis: bool = False,
    ):
        try:
            t_retrieval_start = time.monotonic()
            search_result = await self.llm.llm_tree_search(
                query, tree, chat_history=chat_history, flat_sections=flat_sections
            )
            node_ids = search_result.get("node_list", [])
            nodes = self.llm.find_nodes_by_ids(tree, node_ids, node_map=node_map)

            # If short / referential or overview query returned no explicit node matches, select top overview sections
            if not nodes:
                overview_triggers = ["دا", "ده", "دي", "لخص", "ملخص", "اشرح", "بحث", "summary", "summarize", "overview", "what is this", "about"]
                normalized_q = query.strip().lower()
                is_short_or_overview = len(normalized_q) <= 8 or any(t in normalized_q for t in overview_triggers)

                if is_short_or_overview:
                    app_logger.info(f"Query '{query}' resolved as overview/referential query with 0 matched nodes. Defaulting to leading document sections.")
                    top_nodes = []
                    def collect_leading(items):
                        for item in items:
                            if isinstance(item, dict):
                                top_nodes.append(item)
                                if len(top_nodes) >= 4:
                                    return
                                if item.get("nodes"):
                                    collect_leading(item["nodes"])
                    collect_leading(tree)
                    nodes = top_nodes

            t_retrieval = time.monotonic() - t_retrieval_start
            app_logger.info(
                f"[PERF] rag_pipeline retrieval_time={t_retrieval:.2f}s "
                f"nodes_matched={len(nodes)} deep_analysis={deep_analysis}"
            )

            # Signal retrieval completion to frontend immediately
            yield f'<!--STAGE:{{"stage":"retrieved","duration":{t_retrieval:.2f},"nodes":{len(nodes)}}}-->\n'

            async for chunk in self.llm.generate_answer_stream(
                query, nodes, chat_history=chat_history, doc_name=doc_name, deep_analysis=deep_analysis
            ):
                yield chunk

        except Exception as e:
            app_logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
            is_arabic = any("\u0600" <= c <= "\u06ff" for c in query)
            if is_arabic:
                yield "حدث خطأ أثناء استرجاع المعلومات من المستند المفهرس. يرجى إعادة المحاولة."
            else:
                yield "An error occurred while retrieving information from the indexed document. Please try again."
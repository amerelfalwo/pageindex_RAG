from utils.logger import app_logger

class VectorlessRAG:
    def __init__(self, llm_service):
        self.llm = llm_service

    async def answer_query_stream(self, query: str, tree: list, chat_history: list = None):
        try:
            search_result = await self.llm.llm_tree_search(query, tree)
            node_ids = search_result.get("node_list", [])
            nodes = self.llm.find_nodes_by_ids(tree, node_ids)
            
            async for chunk in self.llm.generate_answer_stream(query, nodes, chat_history):
                yield chunk
            
        except Exception as e:
            app_logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
            yield "عذراً، حدث خطأ داخلي أثناء معالجة سؤالك."
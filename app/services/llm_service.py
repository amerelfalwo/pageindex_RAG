import json
from openai import AsyncOpenAI

class LLMService:
    def __init__(self, api_key: str, base_url: str = "https://models.inference.ai.azure.com"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def summarize_history(self, history: list, model: str = "gpt-4o") -> str:
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history if msg['role'] != 'system'])
        
        prompt = f"""Summarize the following chat history concisely in Markdown format. 
Focus on the main context, key facts discussed, and the user's overall intent.

Chat History:
{history_text}"""

        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

    async def llm_tree_search(self, query: str, tree: list, model: str = "gpt-4o") -> dict:
        final_nodes = []
        all_thinking = ""
        
        async def explore_level(nodes, level_name="Root"):
            nonlocal all_thinking, final_nodes
            
            if not nodes:
                return
                
            current_level_data = []
            for n in nodes:
                current_level_data.append({
                    "node_id": n["node_id"],
                    "title": n["title"],
                    "summary": n.get("text", "")[:200] 
                })
                
            prompt = f"""You are analyzing a specific level of a document tree (Level: {level_name}).
Your task is to identify which node IDs most likely contain the answer to the query.
Only select nodes that are highly relevant. If none are relevant, return an empty node_list.

Query: {query}

Nodes at this level:
{json.dumps(current_level_data, indent=2)}

Reply ONLY in this exact JSON format:
{{
  "thinking": "<brief reasoning for this level>",
  "node_list": ["node_id1", "node_id2"]
}}"""

            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            chosen_ids = result.get("node_list", [])
            
            if result.get("thinking"):
                all_thinking += f"\n[Level: {level_name}] " + result.get("thinking")
                
            for node in nodes:
                if node["node_id"] in chosen_ids:
                    final_nodes.append(node["node_id"])
                    children = node.get("nodes", [])
                    if children:
                        await explore_level(children, level_name=node["title"])

        await explore_level(tree)
        final_nodes = list(set(final_nodes))
        
        return {
            "thinking": all_thinking.strip(),
            "node_list": final_nodes
        }

    def find_nodes_by_ids(self, tree: list, target_ids: list) -> list:
        found = []
        for node in tree:
            if node["node_id"] in target_ids:
                found.append(node)
            if node.get("nodes"):
                found.extend(self.find_nodes_by_ids(node["nodes"], target_ids))
        return found

    async def generate_answer_stream(self, query: str, nodes: list, chat_history: list = None, model: str = "gpt-4o"):
        if chat_history is None:
            chat_history = []

        if not nodes:
            context_msg = "No relevant sections found in the document for this question."
        else:
            context_parts = []
            for node in nodes:
                context_parts.append(
                    f"[Section: '{node['title']}' | Page {node.get('page_index', '?')}]\n"
                    f"{node.get('text', 'Content not available.')}"
                )
            context_msg = "\n\n---\n\n".join(context_parts)
        
        system_prompt = f"""You are a smart and professional assistant. Your only task is to help the user understand and analyze the provided document.

Strict rules you must follow:
1. Contextual Adherence: Answer the question using ONLY the information provided in the (Available Context). Never use your general knowledge.
2. No Hallucination: If the answer is not in the context, you must clearly apologize and say: "Sorry, I could not find an answer to this question in the document." Do not attempt to guess the answer.
3. Citation: For any information you provide, you must cite the section title and page number in parentheses.
4. Conversational Awareness: If the user greets you (e.g., "Hello") or asks about your identity, politely respond, introduce yourself as a document analysis assistant, and ask them to prompt their questions about the document.
5. Conciseness: Keep your answers direct, accurate, and organized in bullet points if necessary.

Available Context from the document:
{context_msg}"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": query})
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
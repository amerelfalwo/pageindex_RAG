import os
import re
import time
import json
import uuid
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
from openai import AsyncOpenAI

from utils.logger import app_logger

load_dotenv()


class LLMService:
    """Dedicated LLM Service using Hugging Face Serverless Inference API with Groq automatic fallback."""

    def __init__(
        self,
        token: Optional[str] = None,
        retrieval_model: Optional[str] = None,
        chat_model: Optional[str] = None,
    ):
        self.token = token or os.getenv("HF_TOKEN")
        self.retrieval_model = retrieval_model or os.getenv(
            "HF_RETRIEVAL_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.chat_model = chat_model or os.getenv(
            "HF_CHAT_MODEL", "Qwen/Qwen2.5-72B-Instruct"
        )
        self.summary_model = os.getenv("HF_SUMMARY_MODEL", self.retrieval_model)

        try:
            self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        except ValueError:
            self.timeout_seconds = 60.0

        try:
            self.stream_first_token_timeout = float(os.getenv("LLM_STREAM_FIRST_TOKEN_TIMEOUT", "30"))
        except ValueError:
            self.stream_first_token_timeout = 30.0

        self.client = AsyncInferenceClient(token=self.token, timeout=self.timeout_seconds)
        self.hf_exhausted = False

        # Groq Fallback Client (used when Hugging Face tokens/credits are depleted)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client: Optional[AsyncOpenAI] = None
        self.groq_chat_model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
        self.groq_retrieval_model = os.getenv("GROQ_RETRIEVAL_MODEL", "openai/gpt-oss-120b")

        if self.groq_api_key:
            self.groq_client = AsyncOpenAI(
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=self.timeout_seconds,
            )
            app_logger.info(
                f"Initialized Groq Fallback Client | Chat: {self.groq_chat_model} | "
                f"Retrieval: {self.groq_retrieval_model} | Timeout: {self.timeout_seconds}s"
            )

        app_logger.info(
            f"Initialized LLMService via Hugging Face | Retrieval: {self.retrieval_model} | "
            f"Chat: {self.chat_model} | Timeout: {self.timeout_seconds}s (stream first token: {self.stream_first_token_timeout}s)"
        )

    def _is_hf_exhausted(self, exc: Exception) -> bool:
        """Detect Hugging Face token/credit exhaustion or timeout to trigger Groq fallback."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            if not self.hf_exhausted:
                app_logger.warning(
                    f"[LLM FALLBACK] Hugging Face timed out ({type(exc).__name__}). Switching to Groq API fallback."
                )
            self.hf_exhausted = True
            return True

        err_str = str(exc).lower()
        if (
            "402" in err_str
            or "credit" in err_str
            or "payment required" in err_str
            or "quota" in err_str
            or "rate limit" in err_str
            or "429" in err_str
            or "timeout" in err_str
            or "timed out" in err_str
        ):
            if not self.hf_exhausted:
                app_logger.warning(
                    f"[LLM FALLBACK] Hugging Face issue/timeout: {exc}. Switching to Groq API fallback."
                )
            self.hf_exhausted = True
            return True
        return False

    def _extract_json(self, raw_text: str) -> dict:
        """Safely extract JSON object from LLM response or codeblock."""
        text = raw_text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()

        if not (text.startswith("{") and text.endswith("}")):
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                text = match.group(0).strip()

        try:
            return json.loads(text)
        except Exception:
            try:
                # Attempt to close open braces/brackets if cut off
                open_braces = text.count("{") - text.count("}")
                open_brackets = text.count("[") - text.count("]")
                repaired = text + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
                return json.loads(repaired)
            except Exception:
                pass
            app_logger.warning(f"Failed to decode JSON from: {raw_text[:200]}. Falling back to empty object.")
            return {}

    def _normalize_tree(self, tree: object) -> list:
        if isinstance(tree, str):
            try:
                tree = json.loads(tree)
            except json.JSONDecodeError as exc:
                raise ValueError("Tree is a string but not valid JSON.") from exc

        if isinstance(tree, dict):
            if "tree" in tree and isinstance(tree["tree"], list):
                tree = tree["tree"]
            elif "nodes" in tree and isinstance(tree["nodes"], list) and "node_id" not in tree:
                tree = tree["nodes"]
            else:
                tree = [tree]

        if not isinstance(tree, list):
            raise TypeError("Tree must be a list of nodes.")

        self._coerce_node_ids(tree)
        return tree

    def _coerce_node_ids(self, nodes: list):
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if not node.get("node_id"):
                for key in ("id", "nodeId", "nodeID", "uuid"):
                    if node.get(key):
                        node["node_id"] = node[key]
                        break
            if not node.get("node_id"):
                node["node_id"] = f"auto_{uuid.uuid4().hex}"
            if node.get("nodes"):
                self._coerce_node_ids(node["nodes"])

    def normalize_tree(self, tree: object) -> list:
        return self._normalize_tree(tree)

    def build_index_maps(self, tree: list) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Build an O(1) node_map dictionary and a compact flattened sections list for instant retrieval."""
        tree = self._normalize_tree(tree)
        node_map: Dict[str, Any] = {}
        flat_sections: List[Dict[str, Any]] = []

        def traverse(nodes):
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                nid = n.get("node_id")
                if nid:
                    node_map[nid] = n
                    summary_text = (n.get("summary") or n.get("text", ""))[:180].replace("\n", " ").strip()
                    flat_sections.append({
                        "node_id": nid,
                        "title": n.get("title", "(untitled)"),
                        "summary": summary_text,
                    })
                if n.get("nodes"):
                    traverse(n["nodes"])

        traverse(tree)
        return node_map, flat_sections

    async def summarize_history(self, history: list, model: Optional[str] = None) -> str:
        """Summarize conversation history using the configured summary model."""
        target_model = model or self.summary_model
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history if msg.get('role') != 'system'])

        prompt = f"""You are a conversation-memory compression component.

Your task is to compress the provided conversation history into a concise, high-information summary that preserves the information necessary for future conversation continuity.

Do NOT answer the conversation.
Do NOT add new information.
Do NOT speculate.
Do NOT preserve irrelevant small talk.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRESERVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Preserve information that may affect future responses, including:

1. USER INTENT
- What the user is trying to accomplish.

2. ACTIVE TASK
- What is currently being worked on.

3. IMPORTANT FACTS
- Facts explicitly provided by the user.
- Important technical or domain information.

4. DECISIONS
- Decisions already made.
- Approaches selected or rejected.

5. CONSTRAINTS
- Requirements.
- Limitations.
- Preferences relevant to the active task.

6. TECHNICAL CONTEXT
- Frameworks.
- Libraries.
- Models.
- APIs.
- Architecture decisions.
- Important implementation details.

7. OPEN ITEMS
- Questions not yet resolved.
- Tasks still pending.
- Information the user still needs to provide.

8. REFERENCES
- Important files, projects, components, or entities mentioned.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REMOVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Remove:

- greetings
- repeated statements
- casual conversation
- redundant explanations
- intermediate reasoning that no longer matters
- irrelevant examples
- temporary wording
- duplicated facts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCURACY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Only preserve information explicitly supported by the conversation.

Never:

- invent facts,
- infer personal attributes,
- convert uncertainty into certainty,
- change technical details,
- merge contradictory statements without noting the conflict.

If the conversation contains conflicting information, preserve the conflict briefly rather than silently choosing one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return concise Markdown using exactly these sections when applicable:

## User Intent
## Current Task
## Important Facts
## Decisions
## Constraints
## Technical Context
## Open Items

Omit sections that have no meaningful information.

Keep the summary compact but information-dense.

The summary must be understandable without access to the original conversation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{history_text}"""

        if not self.hf_exhausted:
            try:
                response = await self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=target_model,
                    temperature=0.2,
                    max_tokens=800,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                self._is_hf_exhausted(e)
                app_logger.warning(f"HF summarize_section failed ({e}). Checking Groq fallback...")

        if self.groq_client:
            try:
                app_logger.info(f"[FALLBACK] Summarizing section via Groq ({self.groq_chat_model})")
                groq_resp = await self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.groq_chat_model,
                    temperature=0.2,
                    max_tokens=800,
                )
                return groq_resp.choices[0].message.content or ""
            except Exception as ge:
                app_logger.error(f"Groq summarize_section fallback failed: {ge}", exc_info=True)

        return ""

    summarize_conversation = summarize_history

    async def llm_tree_search(
        self,
        query: str,
        tree: list,
        chat_history: Optional[list] = None,
        model: Optional[str] = None,
        flat_sections: Optional[list] = None,
    ) -> dict:
        """Fast retrieval model identifies relevant section node_ids from tree."""
        t_search_start = time.monotonic()
        target_model = model or self.retrieval_model

        if flat_sections:
            all_nodes_data = flat_sections
        else:
            tree = self._normalize_tree(tree)
            all_nodes_data = []

            def flatten_tree(nodes):
                for n in nodes:
                    if not isinstance(n, dict):
                        continue
                    all_nodes_data.append({
                        "node_id": n.get("node_id"),
                        "title": n.get("title", "(untitled)"),
                        "summary": (n.get("summary") or n.get("text", ""))[:180].replace("\n", " ").strip(),
                    })
                    if n.get("nodes"):
                        flatten_tree(n["nodes"])

            flatten_tree(tree)

        conv_context_str = ""
        if chat_history:
            recent_msgs = [m for m in chat_history if m.get("role") != "system"][-3:]
            if recent_msgs:
                conv_lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_msgs]
                conv_context_str = (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "CONVERSATION CONTEXT (FOR ANAPHORA & INTENT RESOLUTION)\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    + "\n".join(conv_lines)
                    + "\n"
                )

        prompt = f"""You are the document retrieval and evidence-selection component of a production-grade Retrieval-Augmented Generation (RAG) system.

Your sole responsibility is to identify the smallest set of document nodes that are most likely to contain the evidence required to answer the user's query.

You are NOT the final answer generator.
Do NOT answer the user's question.
Do NOT explain your reasoning.
Do NOT invent information.
Return ONLY the required JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given:

1. A user's query.
2. A hierarchical document represented as nodes.
3. Each node contains:
   - node_id
   - title
   - summary/text preview

Select the most relevant node_ids that should be passed to the answer-generation stage.

The selected nodes must maximize evidence coverage while minimizing irrelevant context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVAL PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RELEVANCE OVER KEYWORD MATCHING

Do not select a node merely because it shares keywords with the query.

Prefer semantic relevance:
- concepts
- entities
- relationships
- definitions
- procedures
- causes
- comparisons
- results
- conclusions
- explicit references

A node is relevant when its content can directly answer the query or provides necessary supporting context.

2. DIRECT EVIDENCE FIRST

Prefer nodes containing direct evidence over nodes containing only broad background information.

Priority:

A. Direct answer/evidence
B. Supporting explanation
C. Necessary surrounding context
D. General background

Do not select D unless it is genuinely useful.

3. QUERY INTENT

Infer the user's information need before selecting nodes.

Common intents include:

- definition
- explanation
- summary
- procedure/how-to
- comparison
- cause/effect
- factual lookup
- numerical/value lookup
- methodology
- experiment
- result
- conclusion
- limitation
- recommendation
- entity-specific question
- multi-part question

For multi-part queries, retrieve nodes covering the different parts of the question.

4. DOCUMENT STRUCTURE

Use the hierarchy and section titles as retrieval signals.

Prefer structurally appropriate sections.

Examples:

- General overview → abstract, introduction, overview, conclusion
- Methodology → methods, methodology, implementation, experimental setup
- Results → results, evaluation, experiments, findings
- Limitations → limitations, discussion, threats to validity
- Future work → future work, conclusion, discussion
- Specific concept → the section where that concept is actually discussed

Do not select a section solely because its title sounds relevant if its summary does not support the query.

5. MULTI-NODE QUESTIONS

If answering the query requires information from multiple sections, select all necessary nodes.

Do not force a single-node answer when multiple nodes are clearly required.

6. PARENT VS CHILD NODES

Prefer the most specific relevant child node when it contains sufficient evidence.

Select a parent node when:
- it contains necessary context absent from children,
- the query concerns the entire section,
- or multiple child nodes collectively answer the question and the parent provides useful framing.

Avoid selecting both parent and child unless both provide distinct useful evidence.

7. SUMMARY / OVERVIEW QUERIES

For queries asking what the document is about, provide an overview, summarize the document, or describe its main ideas:

Prefer:
- abstract
- executive summary
- introduction
- overview
- conclusion

Select the smallest combination that provides adequate coverage.

8. ENTITY AND TERM RESOLUTION

When the query contains a named entity, technical term, acronym, product, method, model, person, dataset, or organization:

Prioritize nodes explicitly discussing that entity.

Do not infer that a node is relevant solely because the entity is conceptually related.

9. NUMERICAL QUESTIONS

For questions asking for numbers, metrics, dates, percentages, measurements, counts, or specific values:

Prefer nodes containing the actual values or results.

Do not select a general discussion section when a results/table/experiment node contains the requested value.

10. COMPARISON QUESTIONS

For comparisons:

Retrieve nodes containing evidence for both sides of the comparison whenever possible.

Do not retrieve only one side.

11. NEGATION AND QUALIFIERS

Pay attention to:

- not
- without
- except
- only
- unless
- before/after
- greater/less than
- increase/decrease
- positive/negative

A node that discusses the opposite condition should not be selected merely because it contains matching keywords.

12. CONVERSATIONAL QUERIES

The query may depend on previous conversation context.

If the query contains references such as:

- "it"
- "this"
- "that"
- "the previous one"
- "the method"
- "the model"
- "the second approach"

Use the available query context to infer the intended reference only when it is unambiguous.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECTION LIMITS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Normally select between 1 and 4 nodes.

Use:

1 node → direct and self-contained evidence.

2 nodes → answer requires two related sections.

3 nodes → multiple supporting sections are necessary.

4 nodes → complex or multi-part query requiring broader evidence.

Do NOT select 4 nodes by default.

Prefer fewer highly relevant nodes over many weakly relevant nodes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NO-EVIDENCE POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If none of the available nodes provides meaningful evidence for the query:

Return an empty node_list.

Do NOT select nodes simply to avoid returning an empty result.

The downstream answer generator is responsible for handling insufficient evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE DOCUMENT SECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{json.dumps(all_nodes_data, separators=(',', ':'))}

{conv_context_str}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{query}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT CONTRACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY valid JSON.

Exact schema:

{{
  "node_list": ["node_id1", "node_id2"]
}}

Requirements:

- node_list MUST be an array.
- Every ID MUST exactly match an existing node_id.
- Do not modify node IDs.
- Do not add explanations.
- Do not use Markdown.
- Do not wrap the JSON in ```json.
- Do not include additional fields.
- Do not include comments.
- Do not return duplicate IDs.

If there is insufficient evidence:

{{
  "node_list": []
}}"""

        final_nodes = []
        hf_failed = False
        if not self.hf_exhausted:
            try:
                response = await self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=target_model,
                    temperature=0.1,
                    max_tokens=300,
                )
                content = response.choices[0].message.content or "{}"
                result = self._extract_json(content)
                final_nodes = result.get("node_list", [])
            except Exception as e:
                self._is_hf_exhausted(e)
                hf_failed = True
                app_logger.warning(f"HuggingFace tree search failed ({e}). Checking Groq fallback...")

        if (self.hf_exhausted or hf_failed) and self.groq_client:
            try:
                app_logger.info(
                    f"[FALLBACK] Executing tree search via Groq ({self.groq_retrieval_model})"
                )
                groq_resp = await self.groq_client.chat.completions.create(
                    model=self.groq_retrieval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                )
                content = groq_resp.choices[0].message.content or "{}"
                result = self._extract_json(content)
                final_nodes = result.get("node_list", [])
                target_model = f"{self.groq_retrieval_model} (Groq)"
            except Exception as ge:
                app_logger.error(f"Groq fallback tree search failed: {ge}", exc_info=True)
                final_nodes = []

        t_search = time.monotonic() - t_search_start
        app_logger.info(
            f"[PERF] tree_search duration={t_search:.2f}s matched_nodes={final_nodes} "
            f"sections_considered={len(all_nodes_data)} model={target_model}"
        )
        return {
            "thinking": f"Tree search executed with {target_model}",
            "node_list": final_nodes,
        }

    def find_nodes_by_ids(self, tree: list, target_ids: list, node_map: Optional[dict] = None) -> list:
        if node_map:
            found = [node_map[nid] for nid in target_ids if nid in node_map]
        else:
            tree = self._normalize_tree(tree)
            found = []
            def search(nodes):
                for node in nodes:
                    if not isinstance(node, dict):
                        continue
                    if node.get("node_id") in target_ids:
                        found.append(node)
                    if node.get("nodes"):
                        search(node["nodes"])
            search(tree)

        # Context deduplication: if child node text is fully contained in another node, remove the duplicate
        if len(found) > 1:
            deduped = []
            for i, n1 in enumerate(found):
                t1 = (n1.get("text") or n1.get("content") or "").strip()
                is_duplicate = False
                if t1:
                    for j, n2 in enumerate(found):
                        if i != j:
                            t2 = (n2.get("text") or n2.get("content") or "").strip()
                            if t2 and len(t1) < len(t2) and t1 in t2:
                                is_duplicate = True
                                break
                if not is_duplicate:
                    deduped.append(n1)
            return deduped
        return found

    async def generate_answer_stream(
        self,
        query: str,
        nodes: list,
        chat_history: list = None,
        doc_name: Optional[str] = None,
        model: Optional[str] = None,
        session_summary: Optional[str] = None,
        global_knowledge: Optional[str] = None,
        deep_analysis: bool = False,
        web_results: Optional[List[Dict[str, Any]]] = None,
        arxiv_results: Optional[List[Dict[str, Any]]] = None,
    ):
        """Chat & synthesis model generates streaming answer with multimodal citations."""
        target_model = model or self.chat_model
        if chat_history is None:
            chat_history = []

        if not nodes:
            context_msg = "No relevant sections found in the document for this question."
        else:
            max_node_chars = 4500 if deep_analysis else 2500
            max_total_chars = 16000 if deep_analysis else 9000
            context_parts = []
            current_total_chars = 0
            for node in nodes:
                image_markers = ""
                images = node.get("images") or []
                if images:
                    image_markers = "\n" + "\n".join([f"[IMAGE_AVAILABLE: {url}]" for url in images])
                title = node.get("title") or node.get("name") or "(Untitled Section)"
                active_page = node.get("active_page")
                page = active_page if active_page is not None else (node.get("page_index") or node.get("page") or "?")
                text = (node.get("text") or node.get("content") or "Content not available.").strip()
                if len(text) > max_node_chars:
                    text = text[:max_node_chars].rstrip() + "\n[...section continues...]"
                
                doc_label = node.get("doc_name") or node.get("filename")
                doc_prefix = f"Document: '{doc_label}' | " if doc_label else ""
                part = f"[{doc_prefix}Section: '{title}' | Page {page}]\n{text}{image_markers}"
                if current_total_chars + len(part) > max_total_chars and context_parts:
                    remaining_budget = max(500, max_total_chars - current_total_chars)
                    part = f"[{doc_prefix}Section: '{title}' | Page {page}]\n{text[:remaining_budget].rstrip()}\n[...truncated for context limit...]{image_markers}"
                    context_parts.append(part)
                    break
                context_parts.append(part)
                current_total_chars += len(part)

            context_msg = "\n\n---\n\n".join(context_parts)

        effective_doc_name = doc_name or "Uploaded Document"

        session_summary_str = ""
        if session_summary:
            session_summary_str = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSESSION LONG-TERM MEMORY\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{session_summary}\n\n"

        global_knowledge_str = ""
        if global_knowledge:
            if isinstance(global_knowledge, dict):
                coll = global_knowledge.get("collective_summary", "")
                insights = global_knowledge.get("key_insights", [])
                insights_str = "\n".join([f"- {ins}" for ins in insights]) if insights else ""
                global_knowledge_str = (
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"GLOBAL KNOWLEDGE MEMORY (All workspace documents)\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{coll}\n\nKey Insights:\n{insights_str}\n\n"
                )
            else:
                global_knowledge_str = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nGLOBAL KNOWLEDGE MEMORY (All workspace documents)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{global_knowledge}\n\n"

        web_evidence_str = ""
        if web_results:
            try:
                from mcp_server.web_search_mcp import format_web_results_for_llm
                formatted_web = format_web_results_for_llm(web_results)
                web_evidence_str = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nLIVE WEB SEARCH EVIDENCE (FastMCP Web Search)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{formatted_web}\n\n"
            except Exception as e:
                app_logger.warning(f"Failed to format web_results for LLM: {e}")

        arxiv_evidence_str = ""
        if arxiv_results:
            try:
                from mcp_server.research_tools import format_arxiv_results_for_llm
                formatted_arxiv = format_arxiv_results_for_llm(arxiv_results, query=query)
                arxiv_evidence_str = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nACADEMIC RESEARCH & ARXIV EVIDENCE\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{formatted_arxiv}\n\n"
            except Exception as e:
                app_logger.warning(f"Failed to format arxiv_results for LLM: {e}")

        system_prompt = f"""You are Atlas, a production-grade document intelligence assistant operating inside a Retrieval-Augmented Generation (RAG) system.
Your name is Atlas. When asked about your identity, name, or purpose, introduce yourself clearly as Atlas, the Document Intelligence research assistant.

Your job is to answer the user's question using the supplied document evidence and conversation context while minimizing hallucinations and preserving factual accuracy.

You are an answer-generation and synthesis component.

The retrieved document context is the PRIMARY SOURCE OF TRUTH for document-related questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIVE DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Document Name: {effective_doc_name}
Status: Ready and Indexed for this Conversation

CRITICAL DOCUMENT GROUNDING DIRECTIVES:
1. The user has ALREADY successfully uploaded this document to the current session.
2. The extracted sections and text from this document are provided below under DOCUMENT CONTEXT.
3. NEVER tell the user to upload a file, attach a file, or ask if they have a document.
4. NEVER claim that no document exists or that an error occurred in sending the file.
5. If the user's message is a short reference (such as "دا", "لخصه", "اشرحه", "ما أهم النتائج؟", "what about this?"):
   It explicitly refers to THIS active document ({effective_doc_name}).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE PRINCIPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEVER present information as being stated by the document unless it is supported by the provided document context.

If the required information is not present in the retrieved context:

- say that the available document context does not contain enough information,
- do not fabricate an answer,
- do not invent citations,
- do not infer unsupported facts as document facts.

You may use general knowledge only when the user explicitly asks for it or when the question is clearly conversational and not asking what the document says.

When using general knowledge, clearly distinguish it from document-supported information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE HIERARCHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Treat information according to this priority:

1. Direct evidence explicitly stated in the retrieved document.
2. Strongly supported conclusions that follow directly from multiple document statements.
3. Conversation context supplied by the user.
4. General knowledge, only when appropriate and clearly distinguishable.

Never allow general knowledge to override explicit document evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Every factual claim about the document must be traceable to the supplied context.

2. Do not fabricate:
   - facts
   - numbers
   - dates
   - names
   - citations
   - page numbers
   - section names
   - study results
   - methodology details
   - conclusions

3. Preserve uncertainty.

If the document says:
- "may"
- "might"
- "suggests"
- "possibly"
- "approximately"
- "associated with"

do not rewrite it as:
- "will"
- "proves"
- "causes"
- "exactly"

4. Preserve numerical accuracy.

Never:
- change numbers,
- round numbers unnecessarily,
- merge values from unrelated sections,
- invent missing units,
- confuse percentages with absolute values.

5. Distinguish correlation from causation.

If the document reports an association, do not describe it as causal unless causality is explicitly established.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION UNDERSTANDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before generating the answer, internally determine:

- What exactly is the user asking?
- Is the query about the document or general knowledge?
- What evidence is required?
- Which retrieved sections support the answer?
- Are there contradictions?
- Is the available evidence sufficient?

Do NOT expose this internal reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWER STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For a direct factual question:

Answer directly first, then provide concise supporting context.

For an explanation:

Explain the concept clearly and progressively.

For a summary:

Prioritize the most important ideas and omit low-value details.

For a comparison:

Use a structured comparison when useful.

For a multi-part question:

Answer every requested part explicitly.

For a procedural question:

Present the steps in the correct order when the document provides an ordered process.

For a numerical question:

State the exact value and relevant unit/context.

For a "why" question:

Explain the causal reasoning only when supported.

For a "how" question:

Describe the mechanism or procedure supported by the document.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSUFFICIENT EVIDENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If the retrieved context does not contain enough information to answer:

Do NOT guess.

Use a concise response such as:

"The provided document context does not contain enough information to answer this reliably."

If part of the question can be answered:

- answer the supported part,
- explicitly identify what is missing.

Do not turn absence of evidence into evidence of absence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTRADICTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If retrieved sections contain conflicting information:

1. Do not silently choose one.
2. Identify the conflict.
3. Attribute each claim to its corresponding section/page when possible.
4. State that the document contains conflicting information.

Do not resolve contradictions using unsupported assumptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION & SOURCING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IF ANSWERING FROM THE ATTACHED PDF DOCUMENT:
   - Base your answer directly on the retrieved document context.
   - EXACT PAGE-LEVEL CITATION DIRECTIVE:
     Inside each section's retrieved text, exact page transitions are marked with `--- [PAGE X] ---`.
     When citing a specific fact, definition, number, methodology, or claim, YOU MUST cite the EXACT page number indicated by the nearest preceding `--- [PAGE X] ---` marker where that specific statement appears, rather than defaulting to the section's starting page.
     Place citations immediately after the relevant claim or paragraph:
     (Section: "Section Title", Page: X)
   - GLOBAL & EXECUTIVE DOSSIER CITATION DIRECTIVE:
     When citing or synthesizing overarching themes, executive summaries, or core findings from the Executive Knowledge Dossier or top-level chapter summaries, cite them explicitly:
     e.g., (Section: "Executive Knowledge Dossier & Core Findings", Page: 1) or (Section: "Chapter Title", Page: X).
     Preserve proper citations from both top-level tree chapters and specific sub-nodes.
   - Fallback: If no `--- [PAGE X] ---` marker is found inside the text (e.g., legacy indexed documents), fallback gracefully to the section's base page number from the section header: [Page Y].
   - Never invent a page number. If page is unavailable or "?", cite section title only:
     (Section: "Section Title")
   - Do not cite sections that do not exist in the document.

2. IF THE ANSWER IS NOT IN THE PDF DOCUMENT OR WAS FOUND VIA LIVE WEB SEARCH:
   - State clearly to the user:
     "المعلومة المطلوبة غير مذكورة داخل المستند المرفق، ولكن استناداً إلى نتائج البحث على الويب:" (or natural English if query is English).
   - MANDATORY LINK TESTING & CONTENT VALIDATION RULES:
     * Check and verify the link before citing it: Ensure the link is live and that the page content actually contains the answer and information requested by the user, rather than an empty page, broken link, or irrelevant headline.
     * NEVER invent, guess, or extrapolate speculative URLs. Every single URL MUST be taken verbatim from the verified LIVE WEB SEARCH EVIDENCE.
     * If a link in the search results does not contain the answer or appears to be broken/dead, DO NOT cite it.
   - Use the LIVE WEB SEARCH EVIDENCE to provide a clear, accurate, and comprehensive answer.
   - Embed verified clickable Markdown links directly inside your explanation:
     [Source or Article Title](URL)
   - YOU MUST ALWAYS conclude your response with a dedicated sources block at the very end:
     ### 🌐 المصادر والروابط ذات الصلة:
     - [Article Title 1](URL 1) - وصف دقيق ومختصر للمحتوى الفعلي الموجود داخل الصفحة
     - [Article Title 2](URL 2) - وصف دقيق ومختصر للمحتوى الفعلي الموجود داخل الصفحة

3. HYBRID / COMPARISON (Both PDF & Web used):
   - Clearly delineate what came from the PDF with (Section: "...", Page: X) and what came from the web with [Title](URL).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE AND MULTIMODAL CONTENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Some document sections contain:

[IMAGE_AVAILABLE: URL]

MANDATORY FIGURE EMBEDDING RULE:
Whenever answering a question about a framework, roadmap, taxonomy, architecture, comparison, pros/cons, or methodology, and an image [IMAGE_AVAILABLE: URL] exists in the retrieved section context:
YOU MUST ALWAYS embed and return the relevant figure in your response using standard Markdown:

![Figure Description](URL)

Place the image directly where it illustrates your explanation so the user can visually inspect the diagram alongside your text.
Explain the diagram clearly based on the provided document context.
Do not omit the figure when it directly relates to the question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLES AND STRUCTURED DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the document context contains structured or tabular information:

- preserve relationships between rows and columns,
- preserve units,
- preserve labels,
- do not reorder values in a way that changes meaning.

Use Markdown tables when they make the answer substantially clearer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond in the same language as the user's latest query.

Arabic query → natural fluent Arabic.

English query → natural fluent English.

If the user mixes Arabic and English:

- preserve technical terms in their standard English form when appropriate,
- otherwise follow the dominant language of the query.

Do not translate technical names, model names, dataset names, library names, or proper nouns unnecessarily.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Be:

- accurate
- concise
- professional
- clear
- natural
- helpful

Avoid:

- unnecessary repetition
- generic introductions
- excessive disclaimers
- fake confidence
- unsupported speculation
- verbose reasoning
- mentioning internal retrieval processes unless relevant

Do not say:
"I searched the nodes."
"I retrieved this section."
"My system found..."
"I used the RAG system..."

Speak directly to the user.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conversation history may contain useful context about what the user means.

Use it to understand references and follow-up questions.

However:

Conversation history is NOT evidence for what the document states.

When answering document-specific questions:

DOCUMENT CONTEXT = EVIDENCE

CHAT HISTORY = CONTEXT

Never treat an assumption from chat history as a document fact.

{session_summary_str}{global_knowledge_str}{web_evidence_str}{arxiv_evidence_str}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{context_msg}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL ANSWER REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before producing the final response, internally verify:

1. Did I answer the actual question?
2. Is every document-specific factual claim supported?
3. Did I preserve numbers and qualifiers?
4. Did I avoid unsupported assumptions?
5. Are citations accurate?
6. Did I follow the user's requested format?
7. Did I respond in the correct language?
8. Did I avoid unnecessary information?

Return only the final user-facing answer."""

        # Determine dynamic token budget based on user constraints and deep_analysis mode
        brief_triggers = ["line", "lines", "سطر", "سطور", "brief", "باختصار", "short", "bullet", "نقطتين", "نقاط", "summary", "summarize", "لخص"]
        q_lower = query.lower()
        if any(b in q_lower for b in brief_triggers) and not deep_analysis:
            max_out_tokens = 500
        elif deep_analysis:
            max_out_tokens = 2048
        else:
            max_out_tokens = 1100

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        t_gen_start = time.monotonic()
        t_first_token = None
        total_chars = 0
        hf_stream_failed = False

        if not self.hf_exhausted:
            try:
                stream = await self.client.chat_completion(
                    messages=messages,
                    model=target_model,
                    temperature=0.2,
                    max_tokens=max_out_tokens,
                    stream=True,
                )

                first_chunk = None
                try:
                    first_chunk = await asyncio.wait_for(
                        stream.__anext__(),
                        timeout=self.stream_first_token_timeout,
                    )
                except StopAsyncIteration:
                    pass

                if first_chunk and first_chunk.choices and len(first_chunk.choices) > 0:
                    delta = first_chunk.choices[0].delta.content
                    if delta is not None:
                        t_first_token = time.monotonic()
                        ttft = t_first_token - t_gen_start
                        app_logger.info(f"[PERF] answer_generation TTFT={ttft:.2f}s model={target_model} max_tokens={max_out_tokens}")
                        total_chars += len(delta)
                        yield delta

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content
                        if delta is not None:
                            if t_first_token is None:
                                t_first_token = time.monotonic()
                                ttft = t_first_token - t_gen_start
                                app_logger.info(f"[PERF] answer_generation TTFT={ttft:.2f}s model={target_model} max_tokens={max_out_tokens}")
                            total_chars += len(delta)
                            yield delta

                t_gen_end = time.monotonic()
                ttft = (t_first_token - t_gen_start) if t_first_token else 0
                gen_duration = (t_gen_end - t_first_token) if t_first_token else 0
                total_duration = t_gen_end - t_gen_start
                context_chars = len(context_msg)
                context_tokens_est = context_chars // 4
                app_logger.info(
                    f"[PERF] answer_generation total={total_duration:.2f}s TTFT={ttft:.2f}s "
                    f"generation={gen_duration:.2f}s output_chars={total_chars} "
                    f"context_chars={context_chars} context_tokens_est={context_tokens_est} nodes={len(nodes)}"
                )
                return
            except Exception as e:
                self._is_hf_exhausted(e)
                hf_stream_failed = True
                app_logger.warning(f"HuggingFace stream failed ({e}). Checking Groq fallback...")

        # Automatic Groq fallback
        if (self.hf_exhausted or hf_stream_failed) and self.groq_client:
            try:
                app_logger.info(f"[FALLBACK] Generating answer stream via Groq ({self.groq_chat_model})")
                groq_stream = await self.groq_client.chat.completions.create(
                    model=self.groq_chat_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_out_tokens,
                    stream=True,
                )
                first_chunk = None
                try:
                    first_chunk = await asyncio.wait_for(
                        groq_stream.__anext__(),
                        timeout=self.stream_first_token_timeout,
                    )
                except StopAsyncIteration:
                    pass

                if first_chunk and first_chunk.choices and len(first_chunk.choices) > 0:
                    delta = first_chunk.choices[0].delta.content
                    if delta is not None:
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                            ttft = t_first_token - t_gen_start
                            app_logger.info(f"[PERF-GROQ] answer_generation TTFT={ttft:.2f}s model={self.groq_chat_model}")
                        total_chars += len(delta)
                        yield delta

                async for chunk in groq_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content
                        if delta is not None:
                            if t_first_token is None:
                                t_first_token = time.monotonic()
                                ttft = t_first_token - t_gen_start
                                app_logger.info(f"[PERF-GROQ] answer_generation TTFT={ttft:.2f}s model={self.groq_chat_model}")
                            total_chars += len(delta)
                            yield delta

                t_gen_end = time.monotonic()
                ttft = (t_first_token - t_gen_start) if t_first_token else 0
                gen_duration = (t_gen_end - t_first_token) if t_first_token else 0
                total_duration = t_gen_end - t_gen_start
                app_logger.info(
                    f"[PERF-GROQ] answer_generation total={total_duration:.2f}s TTFT={ttft:.2f}s "
                    f"generation={gen_duration:.2f}s output_chars={total_chars} model={self.groq_chat_model}"
                )
                return
            except Exception as ge:
                app_logger.error(f"Groq fallback stream also failed: {ge}", exc_info=True)

        is_arabic = any("\u0600" <= c <= "\u06ff" for c in query)
        err_msg = (
            "عذراً، حدث خطأ أثناء الاتصال بالنماذج. يرجى المحاولة لاحقاً."
            if is_arabic
            else "Sorry, an error occurred while connecting to model providers. Please try again."
        )
        yield err_msg

    async def generate_conversational_stream(
        self,
        query: str,
        chat_history: list = None,
        model: Optional[str] = None,
        session_summary: Optional[str] = None,
        global_knowledge: Optional[str] = None,
        web_results: Optional[List[Dict[str, Any]]] = None,
        arxiv_results: Optional[List[Dict[str, Any]]] = None,
    ):
        """Conversational stream when no specific document is attached yet.
        Uses the smaller/faster retrieval model for low-latency chat, with automatic Groq failover.
        Integrates Session Long-Term Memory, Global Knowledge Memory, FastMCP Live Web Search, and arXiv Academic Research.
        """
        # Use chat_model (Qwen2.5-7B from HF_CHAT_MODEL env) — fast and capable for conversational use
        target_model = model or self.chat_model
        if chat_history is None:
            chat_history = []

        # Build a lean history string (last 6 turns only for speed)
        recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
        history_parts = [f"{m['role'].capitalize()}: {m['content']}" for m in recent if m.get('role') != 'system']
        chat_history_text = "\n".join(history_parts) if history_parts else ""

        is_arabic_query = any("\u0600" <= c <= "\u06ff" for c in query)
        lang_rule = "Respond in Arabic (فصحى طبيعية ومفيدة)." if is_arabic_query else "Respond in English."

        system_prompt = f"""You are Atlas, a professional document intelligence and research assistant.
Your name is Atlas (أطلس). When asked about your identity or name (e.g., "من أنت؟" or "ما اسمك؟" or "who are you?"), introduce yourself clearly as Atlas, a document intelligence and research assistant designed to understand documents, analyze research, and search the web for knowledge.
{lang_rule}
Always match the user's language.
Be concise, accurate, and structured. Never invent facts.
"""
        if web_results:
            try:
                from mcp_server.web_search_mcp import format_web_results_for_llm
                formatted_web = format_web_results_for_llm(web_results)
                system_prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIVE WEB SEARCH EVIDENCE (FastMCP Web Search)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{formatted_web}

WEB SEARCH SOURCING & LINK VERIFICATION RULES:
1. LINK TESTING & CONTENT VALIDATION:
   - Carefully verify that each link is live and that its page content actually contains the substantive answer and information requested by the user.
   - Do NOT output dead, 404, or broken links.
   - Do NOT cite clickbait links where the actual body content is missing or unrelated to the query.
   - NEVER invent, hallucinate, or construct speculative URLs. Every URL MUST match an exact URL from the verified LIVE WEB SEARCH EVIDENCE above.
2. Embed verified clickable Markdown links directly throughout your answer: [Source Title](URL).
3. At the end of your response, ALWAYS include a dedicated sources section:
### 🌐 المصادر والروابط ذات الصلة:
- [Source Title 1](URL 1) - وصف دقيق ومختصر للمحتوى الفعلي الموجود داخل الرابط
- [Source Title 2](URL 2) - وصف دقيق ومختصر للمحتوى الفعلي الموجود داخل الرابط
"""
            except Exception as e:
                app_logger.warning(f"Failed to format web_results for conversational stream: {e}")

        if arxiv_results:
            try:
                from mcp_server.research_tools import format_arxiv_results_for_llm
                formatted_arxiv = format_arxiv_results_for_llm(arxiv_results, query=query)
                system_prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACADEMIC RESEARCH & ARXIV EVIDENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{formatted_arxiv}

ACADEMIC CITATION RULES:
1. When citing arXiv papers, reference the author names, publication year, paper title, and direct PDF download link.
2. Synthesize findings objectively, noting methodology and key insights.
"""
            except Exception as e:
                app_logger.warning(f"Failed to format arxiv_results for conversational stream: {e}")
        if session_summary:
            system_prompt += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nSESSION LONG-TERM MEMORY (Summary of earlier turns)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{session_summary}\n"

        if global_knowledge:
            system_prompt += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nGLOBAL KNOWLEDGE MEMORY (All workspace documents)\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{global_knowledge}\nYou can answer questions about the collective documents in the workspace using the Global Knowledge Memory above.\n"

        if chat_history_text:
            system_prompt += f"\n[Recent conversation]\n{chat_history_text}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        hf_stream_failed = False
        if not self.hf_exhausted:
            try:
                stream = await self.client.chat_completion(
                    messages=messages,
                    model=target_model,
                    temperature=0.6,
                    stream=True,
                )

                first_chunk = None
                try:
                    first_chunk = await asyncio.wait_for(
                        stream.__anext__(),
                        timeout=self.stream_first_token_timeout,
                    )
                except StopAsyncIteration:
                    pass

                if first_chunk and first_chunk.choices and len(first_chunk.choices) > 0:
                    delta = first_chunk.choices[0].delta.content
                    if delta is not None:
                        yield delta

                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content
                        if delta is not None:
                            yield delta
                return
            except Exception as e:
                self._is_hf_exhausted(e)
                hf_stream_failed = True
                app_logger.warning(f"HF conversational stream failed ({e}). Checking Groq fallback...")

        if (self.hf_exhausted or hf_stream_failed) and self.groq_client:
            try:
                app_logger.info(f"[FALLBACK] Conversational stream via Groq ({self.groq_chat_model})")
                groq_stream = await self.groq_client.chat.completions.create(
                    model=self.groq_chat_model,
                    messages=messages,
                    temperature=0.6,
                    stream=True,
                )
                first_chunk = None
                try:
                    first_chunk = await asyncio.wait_for(
                        groq_stream.__anext__(),
                        timeout=self.stream_first_token_timeout,
                    )
                except StopAsyncIteration:
                    pass

                if first_chunk and first_chunk.choices and len(first_chunk.choices) > 0:
                    delta = first_chunk.choices[0].delta.content
                    if delta is not None:
                        yield delta

                async for chunk in groq_stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content
                        if delta is not None:
                            yield delta
                return
            except Exception as ge:
                app_logger.error(f"Groq conversational fallback failed: {ge}", exc_info=True)

        is_arabic = any("\u0600" <= c <= "\u06ff" for c in query)
        err_msg = (
            "عذراً، خدمة النماذج غير متاحة حالياً. يرجى المحاولة لاحقاً."
            if is_arabic
            else "Sorry, the model service is currently unreachable. Please try again."
        )
        yield err_msg

    async def generate_quiz_and_flashcards(
        self, filename: str, tree: list, dossier: Optional[dict] = None
    ) -> dict:
        """
        Uses LLM (with HF + Groq fallback) to generate 5 MCQs and 5 Flashcards from the document.
        Each item includes question, options (for MCQ), answer, explanation, and exact source citation (Section, Page).
        """
        # 1. Prepare structured outline and dossier summary
        dossier_lines = []
        if dossier:
            if dossier.get("executive_summary"):
                dossier_lines.append(f"Executive Summary: {dossier['executive_summary']}")
            if dossier.get("key_findings"):
                dossier_lines.append("Key Findings: " + "; ".join(dossier["key_findings"]))
            if dossier.get("topics"):
                dossier_lines.append("Main Topics: " + ", ".join(dossier["topics"]))
        dossier_str = "\n".join(dossier_lines) if dossier_lines else f"Study materials for {filename}."

        outline_lines = []
        sections_sample = []
        def collect_sections(items):
            for it in items:
                if isinstance(it, dict):
                    t = it.get("title") or "Section"
                    p = it.get("page_index") or it.get("page") or 1
                    s = it.get("summary") or (it.get("text") or "")[:200].replace("\n", " ").strip()
                    txt = (it.get("text") or it.get("content") or "")[:400].replace("\n", " ").strip()
                    outline_lines.append(f"- Section: '{t}' (Page: {p}) | Content: {txt or s}")
                    sections_sample.append({"title": t, "page": p, "summary": s, "text": txt or s})
                    if it.get("nodes"):
                        collect_sections(it["nodes"])
        collect_sections(tree[:15] if isinstance(tree, list) else [])

        outline_str = "\n".join(outline_lines[:15]) if outline_lines else f"Document: {filename}."

        mcq_prompt = f"""You are the Study Studio Intelligence Generator for Atlas.
Analyze the following document sections and executive dossier to generate exactly 5 Multiple Choice Questions (MCQs) testing understanding of this specific document.

DOCUMENT: {filename}

DOSSIER OVERVIEW:
{dossier_str}

SECTION OUTLINE:
{outline_str}

OUTPUT REQUIREMENTS:
Return ONLY valid JSON matching this schema:
{{
  "mcqs": [
    {{
      "id": "mcq_1",
      "question": "Question text testing a core concept from the document?",
      "options": ["A) Choice 1", "B) Choice 2", "C) Choice 3", "D) Choice 4"],
      "answer": "A) Choice 1",
      "explanation": "Clear 1-sentence explanation grounded in the text.",
      "evidence_text": "Exact 1-2 sentence verbatim excerpt from the document.",
      "citation": "(Section: \\"Section Title\\", Page: X)"
    }}
  ]
}}

CRITICAL RULES:
1. Every MCQ MUST have exactly 4 choices prefixed with A), B), C), D).
2. The answer field MUST match one of the options verbatim.
3. Keep explanation concise (1-2 sentences).
4. Do not wrap in markdown or markdown backticks. Return raw JSON.
"""

        flashcard_prompt = f"""You are the Study Studio Intelligence Generator for Atlas.
Analyze the following document sections and executive dossier to generate exactly 5 Active-Recall Flashcards testing understanding of this specific document.

DOCUMENT: {filename}

DOSSIER OVERVIEW:
{dossier_str}

SECTION OUTLINE:
{outline_str}

OUTPUT REQUIREMENTS:
Return ONLY valid JSON matching this schema:
{{
  "flashcards": [
    {{
      "id": "card_1",
      "question": "Front of card: Core concept, term, or question?",
      "answer": "Back of card: Accurate concise definition or answer under 35 words.",
      "explanation": "Supporting academic insight in 1 sentence.",
      "evidence_text": "Exact verbatim sentence from the document.",
      "citation": "(Section: \\"Section Title\\", Page: X)"
    }}
  ]
}}

CRITICAL RULES:
1. Keep answers concise under 35 words.
2. Do not wrap in markdown or markdown backticks. Return raw JSON.
"""

        async def execute_llm_call(prompt_text: str, max_tokens: int) -> str:
            # 1. Try Hugging Face
            if not self.hf_exhausted:
                try:
                    resp = await self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt_text}],
                        model=self.chat_model,
                        temperature=0.3,
                        max_tokens=max_tokens,
                    )
                    out = resp.choices[0].message.content or ""
                    if out.strip():
                        return out
                except Exception as e:
                    self._is_hf_exhausted(e)
                    app_logger.warning(f"HF call failed ({e}). Trying Groq fallback...")

            # 2. Try Groq
            if self.groq_client:
                models_to_try = [self.groq_chat_model, "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
                for candidate_model in dict.fromkeys(models_to_try):
                    try:
                        app_logger.info(f"[QUIZ-GEN] Running call via Groq ({candidate_model})...")
                        groq_resp = await self.groq_client.chat.completions.create(
                            model=candidate_model,
                            messages=[{"role": "user", "content": prompt_text}],
                            temperature=0.3,
                            max_tokens=max_tokens,
                        )
                        out = groq_resp.choices[0].message.content or ""
                        if out.strip():
                            return out
                    except Exception as ge:
                        app_logger.error(f"Groq call failed on {candidate_model}: {ge}")
            return ""

        # Run MCQs and Flashcards in parallel for 2x speed and zero token truncation
        raw_mcqs_str, raw_cards_str = await asyncio.gather(
            execute_llm_call(mcq_prompt, 2400),
            execute_llm_call(flashcard_prompt, 1800),
        )

        data_mcqs = self._extract_json(raw_mcqs_str)
        data_cards = self._extract_json(raw_cards_str)

        mcqs = data_mcqs.get("mcqs", []) if isinstance(data_mcqs, dict) else []
        flashcards = data_cards.get("flashcards", []) if isinstance(data_cards, dict) else []

        # Validate and supplement with grounded document fallback if needed
        def build_deterministic_fallback():
            items_mcqs = []
            items_cards = []
            pool = sections_sample if sections_sample else [{"title": "General Overview", "page": 1, "summary": "Covers document foundations."}]
            
            # Fill MCQs up to 5
            for i in range(len(mcqs), 5):
                sec = pool[i % len(pool)]
                t = sec["title"]
                p = sec["page"]
                s = sec["summary"] or f"Core insights from {t}."

                distractors = []
                for step in [1, 2, 3]:
                    other_sec = pool[(i + step) % len(pool)]
                    other_text = (other_sec.get("summary") or other_sec.get("title") or "").strip()
                    if other_text and other_text != s and other_text not in distractors:
                        distractors.append(other_text[:90])
                while len(distractors) < 3:
                    distractors.append(f"Framework and scope detailed in Section {(i + len(distractors) + 1)}")

                items_mcqs.append({
                    "id": f"mcq_{i+1}",
                    "question": f"Which core concept or finding is established in section '{t}'?",
                    "options": [
                        f"A) {s[:90]}",
                        f"B) {distractors[0]}",
                        f"C) {distractors[1]}",
                        f"D) {distractors[2]}"
                    ],
                    "answer": f"A) {s[:90]}",
                    "explanation": f"According to section '{t}' on page {p}, the document states: {s}",
                    "evidence_text": sec.get("text") or s,
                    "citation": f'(Section: "{t}", Page: {p})'
                })

            # Fill Flashcards up to 5
            for j in range(len(flashcards), 5):
                sec = pool[j % len(pool)]
                t = sec["title"]
                p = sec["page"]
                s = sec["summary"] or f"Key findings in {t}."
                items_cards.append({
                    "id": f"card_{j+1}",
                    "question": f"What is the key takeaway or principle defined in '{t}'?",
                    "answer": s[:150],
                    "explanation": f"Document section '{t}' on page {p} details these findings.",
                    "evidence_text": sec.get("text") or s,
                    "citation": f'(Section: "{t}", Page: {p})'
                })
            return items_mcqs, items_cards




        fb_mcqs, fb_cards = build_deterministic_fallback()
        final_mcqs = (mcqs + fb_mcqs)[:5]
        final_flashcards = (flashcards + fb_cards)[:5]

        # Ensure every item has a citation and exactly 4 options
        for idx, m in enumerate(final_mcqs):
            if "options" in m and isinstance(m["options"], list):
                m["options"] = m["options"][:4]
                letters = ["A)", "B)", "C)", "D)"]
                while len(m["options"]) < 4:
                    m["options"].append(f"{letters[len(m['options'])]} None of the above")
            else:
                m["options"] = ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]

            if m.get("answer") not in m["options"]:
                matched = next((opt for opt in m["options"] if str(m.get("answer", "")) in opt or opt.startswith(str(m.get("answer", ""))[:2])), None)
                m["answer"] = matched if matched else m["options"][0]

            if not m.get("citation"):
                sec = sections_sample[idx % len(sections_sample)] if sections_sample else {"title": "Overview", "page": 1}
                m["citation"] = f'(Section: "{sec["title"]}", Page: {sec["page"]})'

            if not m.get("evidence_text"):
                sec = sections_sample[idx % len(sections_sample)] if sections_sample else {"title": "Overview", "page": 1, "text": ""}
                m["evidence_text"] = sec.get("text") or m.get("explanation") or ""

        for idx, c in enumerate(final_flashcards):
            if not c.get("citation"):
                sec = sections_sample[idx % len(sections_sample)] if sections_sample else {"title": "Overview", "page": 1}
                c["citation"] = f'(Section: "{sec["title"]}", Page: {sec["page"]})'

            if not c.get("evidence_text"):
                sec = sections_sample[idx % len(sections_sample)] if sections_sample else {"title": "Overview", "page": 1, "text": ""}
                c["evidence_text"] = sec.get("text") or c.get("explanation") or ""

        return {
            "mcqs": final_mcqs,
            "flashcards": final_flashcards,
        }
import os
import re
import time
import json
import uuid
from typing import Optional, List, Dict, Any, Tuple
from huggingface_hub import AsyncInferenceClient

from utils.logger import app_logger


class LLMService:
    """Dedicated LLM Service using Hugging Face Serverless Inference API."""

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

        self.client = AsyncInferenceClient(token=self.token)
        app_logger.info(
            f"Initialized LLMService via Hugging Face | Retrieval: {self.retrieval_model} | Chat: {self.chat_model}"
        )

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
        except json.JSONDecodeError:
            app_logger.warning(f"Failed to decode JSON from: {raw_text}. Falling back to empty object.")
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

        response = await self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=target_model,
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content or ""

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
            app_logger.error(f"Error during llm_tree_search: {str(e)}", exc_info=True)
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
        deep_analysis: bool = False,
    ):
        """Chat & synthesis model generates streaming answer with multimodal citations."""
        target_model = model or self.chat_model
        if chat_history is None:
            chat_history = []

        if not nodes:
            context_msg = "No relevant sections found in the document for this question."
        else:
            max_node_chars = 4500 if deep_analysis else 2500
            max_total_chars = 12000 if deep_analysis else 6000
            context_parts = []
            current_total_chars = 0
            for node in nodes:
                image_markers = ""
                images = node.get("images") or []
                if images:
                    image_markers = "\n" + "\n".join([f"[IMAGE_AVAILABLE: {url}]" for url in images])
                title = node.get("title") or node.get("name") or "(Untitled Section)"
                page = node.get("page_index") or node.get("page") or "?"
                text = (node.get("text") or node.get("content") or "Content not available.").strip()
                if len(text) > max_node_chars:
                    text = text[:max_node_chars].rstrip() + "\n[...section continues...]"
                part = f"[Section: '{title}' | Page {page}]\n{text}{image_markers}"
                if current_total_chars + len(part) > max_total_chars and context_parts:
                    remaining_budget = max(500, max_total_chars - current_total_chars)
                    part = f"[Section: '{title}' | Page {page}]\n{text[:remaining_budget].rstrip()}\n[...truncated for context limit...]{image_markers}"
                    context_parts.append(part)
                    break
                context_parts.append(part)
                current_total_chars += len(part)

            context_msg = "\n\n---\n\n".join(context_parts)

        effective_doc_name = doc_name or "Uploaded Document"

        system_prompt = f"""You are a production-grade document intelligence assistant operating inside a Retrieval-Augmented Generation (RAG) system.

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
CITATION POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Citations must refer ONLY to information present in the supplied context.

When possible, cite using:

(Section: "Section Title", Page: X)

For claims spanning multiple sections:

(Sections: "Section A", "Section B")

Place citations immediately after the relevant claim or paragraph.

Do not add citations to claims that are not supported by the document.

Never invent a page number.

If the page number is unavailable or "?":

cite the section title only.

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

        try:
            stream = await self.client.chat_completion(
                messages=messages,
                model=target_model,
                temperature=0.2,
                max_tokens=max_out_tokens,
                stream=True,
            )

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
        except Exception as e:
            app_logger.error(f"Error in generate_answer_stream: {str(e)}", exc_info=True)
            is_arabic = any("\u0600" <= c <= "\u06ff" for c in query)
            if "402" in str(e) or "credits" in str(e).lower():
                err_msg = "عذراً، تم استهلاك رصيد الاستخدام الشهري للنموذج (402 Payment Required)." if is_arabic else "Sorry, monthly included model credits are depleted (402 Payment Required)."
            else:
                err_msg = f"عذراً، حدث خطأ أثناء الاتصال بالنموذج ({type(e).__name__}). يرجى المحاولة لاحقاً." if is_arabic else f"Sorry, model service error occurred ({type(e).__name__}). Please try again."
            yield err_msg

    async def generate_conversational_stream(
        self,
        query: str,
        chat_history: list = None,
        model: Optional[str] = None,
    ):
        """Conversational stream when no document is attached yet.
        Uses the smaller/faster retrieval model for low-latency chat.
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
        lang_rule = "Respond in Arabic (فصحى طبيعية)." if is_arabic_query else "Respond in English."

        system_prompt = f"""You are a helpful AI assistant.
{lang_rule}
Always match the user's language.
Be concise and accurate. Never invent facts.
If the user asks about a document and none is uploaded, tell them to upload a PDF.
"""
        if chat_history_text:
            system_prompt += f"\n[Recent conversation]\n{chat_history_text}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            stream = await self.client.chat_completion(
                messages=messages,
                model=target_model,
                temperature=0.6,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta.content
                    if delta is not None:
                        yield delta
        except Exception as e:
            app_logger.error(f"Error in generate_conversational_stream: {str(e)}", exc_info=True)
            is_arabic = any("\u0600" <= c <= "\u06ff" for c in query)
            if "402" in str(e) or "credits" in str(e).lower():
                err_msg = "عذراً، تم استهلاك رصيد الاستخدام الشهري للنموذج (402 Payment Required)." if is_arabic else "Sorry, monthly included model credits are depleted (402 Payment Required)."
            else:
                err_msg = f"عذراً، خدمة النماذج غير متاحة حالياً ({type(e).__name__}). يرجى المحاولة لاحقاً." if is_arabic else f"Sorry, the model service is currently unreachable ({type(e).__name__}). Please try again."
            yield err_msg
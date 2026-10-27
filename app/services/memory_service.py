"""
memory_service.py
─────────────────
Manages Multi-Tier Memory for Atlas Document Intelligence:
1. Short-Term Session Memory: Strict per-session sliding context window.
2. Long-Term Session Memory: Rolling conversation summaries for long chats.
3. Document Knowledge Dossiers: Synthesized core findings & entities per document.
4. Global Knowledge Memory: Collective synthesis across all ingested documents.
"""

import json
import time
from typing import Any, Dict, List, Optional

from core.storage import storage
from utils.logger import app_logger


class MemoryService:
    def __init__(self, llm_service) -> None:
        self.llm = llm_service

    # ── 1. Short-Term & Long-Term Session Memory ────────────────────────────
    def get_session_context(
        self, chat_id: str, max_recent_turns: int = 6, owner_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves strictly isolated session memory:
        - Recent turns (short-term memory window)
        - Rolling session summary (long-term memory)
        """
        full_history = storage.get_chat_history(chat_id, owner_id=owner_id) or []
        summary_record = storage.get_chat_summary(chat_id)
        session_summary = summary_record.get("summary") if summary_record else None

        recent_turns = full_history[-max_recent_turns:] if len(full_history) > max_recent_turns else full_history

        return {
            "chat_id": chat_id,
            "session_summary": session_summary,
            "recent_turns": recent_turns,
            "total_messages": len(full_history),
        }

    async def update_session_summary_if_needed(
        self, chat_id: str, owner_id: Optional[str] = None, force: bool = False
    ) -> Optional[str]:
        """Trigger rolling summarization if conversation length exceeds threshold or if force=True."""
        try:
            full_history = storage.get_chat_history(chat_id, owner_id=owner_id) or []
            if not full_history:
                return None
            if not force and len(full_history) < 8:
                return None

            last_summary = storage.get_chat_summary(chat_id)
            last_count = last_summary.get("turn_count", 0) if last_summary else 0

            # Only re-summarize if force=True or at least 4 new messages occurred
            if not force and (len(full_history) - last_count) < 4:
                return last_summary.get("summary") if last_summary else None

            app_logger.info(f"[MEMORY] Updating rolling summary for chat_id={chat_id}")
            summary = await self.llm.summarize_conversation(full_history)
            if summary:
                storage.set_chat_summary(chat_id, summary, turn_count=len(full_history))
                app_logger.info(f"[MEMORY] Successfully updated summary for chat_id={chat_id}")
                return summary
            return None
        except Exception as e:
            app_logger.error(f"[MEMORY] Error updating session summary: {e}")
            return None

    # ── 2. Document Knowledge Dossier ──────────────────────────────────────
    async def generate_document_dossier(
        self, doc_id: str, filename: str, tree: list, flat_sections: Optional[list] = None, owner_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes a newly indexed document to generate an executive dossier:
        - Executive summary
        - Key entities
        - Key findings
        - Topics
        """
        try:
            # Build representative text preview from first few sections or flat sections
            sections_to_read = flat_sections[:8] if flat_sections else []
            preview_texts = []
            for s in sections_to_read:
                title = s.get("title", "Section")
                text = (s.get("text") or "")[:800]
                preview_texts.append(f"### {title}\n{text}")

            if not preview_texts:
                # Fallback to tree node text
                for node in tree[:5]:
                    if isinstance(node, dict):
                        t = node.get("title", "")
                        c = (node.get("text") or "")[:800]
                        preview_texts.append(f"### {t}\n{c}")

            combined_preview = "\n\n".join(preview_texts)[:4000]
            if not combined_preview.strip():
                combined_preview = f"Document: {filename} with {len(tree)} outline nodes."

            prompt = f"""You are the Document Intelligence Synthesizer for Atlas.
Analyze the following document preview and extract a structured knowledge dossier.

DOCUMENT: {filename}
CONTENT PREVIEW:
{combined_preview}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON matching this schema:
{{
  "executive_summary": "A 2-3 paragraph comprehensive summary of the document purpose, findings, and themes.",
  "key_entities": ["Entity 1", "Entity 2", "Entity 3"],
  "key_findings": ["Key finding 1", "Key finding 2", "Key finding 3"],
  "topics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4"]
}}
"""
            raw_result = ""
            if not self.llm.hf_exhausted:
                try:
                    resp = await self.llm.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.llm.chat_model,
                        temperature=0.2,
                        max_tokens=800,
                    )
                    raw_result = resp.choices[0].message.content or ""
                except Exception as e:
                    self.llm._is_hf_exhausted(e)
                    app_logger.warning(f"HF dossier generation failed ({e}). Checking Groq fallback...")

            if (not raw_result or self.llm.hf_exhausted) and self.llm.groq_client:
                groq_resp = await self.llm.groq_client.chat.completions.create(
                    model=self.llm.groq_chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800,
                )
                raw_result = groq_resp.choices[0].message.content or ""

            data = self.llm._extract_json(raw_result)
            summary = data.get("executive_summary") or f"Knowledge dossier for {filename}."
            entities = data.get("key_entities") or []
            findings = data.get("key_findings") or []
            topics = data.get("topics") or []

            storage.save_knowledge_dossier(
                doc_id=doc_id,
                filename=filename,
                executive_summary=summary,
                key_entities=entities,
                key_findings=findings,
                topics=topics,
                owner_id=owner_id,
            )
            app_logger.info(f"[MEMORY] Saved knowledge dossier for doc_id={doc_id} ({filename})")

            # Asynchronously update global knowledge summary
            await self.synthesize_global_knowledge(owner_id=owner_id)

            return {
                "doc_id": doc_id,
                "filename": filename,
                "executive_summary": summary,
                "key_entities": entities,
                "key_findings": findings,
                "topics": topics,
            }
        except Exception as e:
            app_logger.error(f"[MEMORY] Failed to generate document dossier for {doc_id}: {e}", exc_info=True)
    def get_document_dossier(self, doc_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve structured knowledge dossier for an individual document."""
        return storage.get_knowledge_dossier(doc_id, owner_id=owner_id)

    def list_knowledge_dossiers(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all knowledge dossiers across indexed documents."""
        return storage.list_knowledge_dossiers(owner_id=owner_id)

    # ── 3. Global Knowledge Summary Across All Data ────────────────────────
    async def synthesize_global_knowledge(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesizes a collective master knowledge summary of all indexed documents in the workspace.
        """
        dossiers = storage.list_knowledge_dossiers(owner_id=owner_id)
        total_docs = len(dossiers)

        if not dossiers:
            # Check if any documents exist in doc_meta without dossiers yet
            all_docs = storage.list_documents(owner_id=owner_id)
            if not all_docs:
                default_summary = "No documents currently indexed in the knowledge base."
                storage.save_global_knowledge_summary(default_summary, 0, [], owner_id=owner_id)
                return {"collective_summary": default_summary, "total_documents": 0, "key_insights": []}

        # Build combined summary representation of all dossiers
        dossier_summaries = []
        for d in dossiers:
            fname = d.get("filename", "Doc")
            summ = d.get("executive_summary", "")[:600]
            findings = "; ".join(d.get("key_findings", [])[:3])
            topics = ", ".join(d.get("topics", [])[:4])
            dossier_summaries.append(
                f"- **{fname}**\n  Summary: {summ}\n  Topics: {topics}\n  Findings: {findings}"
            )

        combined_input = "\n\n".join(dossier_summaries)

        prompt = f"""You are the Chief Knowledge Architect for Atlas Document Intelligence.
Synthesize the collective knowledge across all {total_docs} documents ingested into the platform.

INDEXED DOCUMENTS DOSSIERS:
{combined_input}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON with this exact schema:
{{
  "collective_summary": "A cohesive 2-4 paragraph overview of the global knowledge contained in all documents, highlighting overarching themes, areas of synergy, and domain expertise.",
  "key_insights": [
    "Cross-cutting insight 1",
    "Cross-cutting insight 2",
    "Cross-cutting insight 3"
  ]
}}
"""
        raw_result = ""
        if not self.llm.hf_exhausted:
            try:
                resp = await self.llm.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.llm.chat_model,
                    temperature=0.3,
                    max_tokens=1000,
                )
                raw_result = resp.choices[0].message.content or ""
            except Exception as e:
                self.llm._is_hf_exhausted(e)
                app_logger.warning(f"HF global knowledge synthesis failed ({e}). Checking Groq fallback...")

        if (not raw_result or self.llm.hf_exhausted) and self.llm.groq_client:
            try:
                groq_resp = await self.llm.groq_client.chat.completions.create(
                    model=self.llm.groq_chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000,
                )
                raw_result = groq_resp.choices[0].message.content or ""
            except Exception as ge:
                app_logger.error(f"Groq global knowledge synthesis failed: {ge}")

        data = self.llm._extract_json(raw_result)
        collective = data.get("collective_summary") or "Atlas Global Knowledge Base synthesis complete."
        insights = data.get("key_insights") or []

        storage.save_global_knowledge_summary(collective, total_docs, insights, owner_id=owner_id)
        app_logger.info(f"[MEMORY] Successfully synthesized global knowledge across {total_docs} documents.")

        return {
            "collective_summary": collective,
            "total_documents": total_docs,
            "key_insights": insights,
            "updated_at": time.time(),
        }

    def get_global_knowledge(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch current global knowledge summary and document dossiers."""
        global_rec = storage.get_global_knowledge_summary(owner_id=owner_id) or {}
        dossiers = storage.list_knowledge_dossiers(owner_id=owner_id)
        return {
            "collective_summary": global_rec.get(
                "collective_summary", "No collective summary available yet. Ingest documents to build knowledge."
            ),
            "total_documents": global_rec.get("total_documents", len(dossiers)),
            "key_insights": global_rec.get("key_insights", []),
            "updated_at": global_rec.get("updated_at"),
            "dossiers": dossiers,
        }

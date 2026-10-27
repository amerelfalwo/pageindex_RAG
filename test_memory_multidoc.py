import os
import sys

# Add app to path
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "app"))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from fastapi.testclient import TestClient
from app import app
from core.storage import storage
from core.persistent_db import persistent_db

TEST_CLIENT_ID = "00000000-0000-4000-8000-000000000001"
client = TestClient(app, headers={"X-Client-ID": TEST_CLIENT_ID})

def test_full_memory_and_multidoc_isolation():
    print("\n--- 1. Testing Session Creation & Isolation ---")
    res_a = client.post("/chat/new")
    assert res_a.status_code == 200
    chat_a = res_a.json()["chat_id"]

    res_b = client.post("/chat/new")
    assert res_b.status_code == 200
    chat_b = res_b.json()["chat_b" if "chat_b" in res_b.json() else "chat_id"]

    print(f"Chat A ID: {chat_a}")
    print(f"Chat B ID: {chat_b}")
    assert chat_a != chat_b

    # Create dummy mock docs in storage
    doc1_id = "mock-doc-1"
    doc2_id = "mock-doc-2"
    mock_tree_1 = [
        {"title": "Section 1: Architecture", "page": 1, "text": "Atlas uses tree-based retrieval."},
        {"title": "Section 2: Memory", "page": 2, "text": "Session memory isolates short and long context."},
    ]
    mock_tree_2 = [
        {"title": "Overview: Financials", "page": 1, "text": "Revenue grew by 24% in Q3."},
    ]

    storage.set_document_ready(doc1_id, tree=mock_tree_1, filename="Architecture_Guide.pdf", page_count=2, section_count=2, owner_id=TEST_CLIENT_ID)
    storage.set_document_ready(doc2_id, tree=mock_tree_2, filename="Financial_Report_Q3.pdf", page_count=1, section_count=1, owner_id=TEST_CLIENT_ID)

    print("\n--- 2. Testing Multi-Document Attachment to Chat A ---")
    attach_res = client.post(f"/chat/{chat_a}/attach-multiple", json={"doc_ids": [doc1_id, doc2_id]})
    assert attach_res.status_code == 200
    assert len(attach_res.json()["doc_ids"]) == 2

    docs_a_res = client.get(f"/chat/{chat_a}/documents")
    assert docs_a_res.status_code == 200
    docs_a = docs_a_res.json()["documents"]
    print(f"Chat A attached documents count: {len(docs_a)}")
    assert len(docs_a) == 2
    assert {d["doc_id"] for d in docs_a} == {doc1_id, doc2_id}

    # Verify Chat B has zero attached documents (Strict Session Isolation)
    docs_b_res = client.get(f"/chat/{chat_b}/documents")
    assert docs_b_res.status_code == 200
    docs_b = docs_b_res.json()["documents"]
    print(f"Chat B attached documents count: {len(docs_b)}")
    assert len(docs_b) == 0, "Violation of session isolation: Chat B should have 0 docs"

    print("\n--- 3. Testing Message Isolation Between Sessions ---")
    storage.add_message(chat_a, "user", "What is Atlas?")
    storage.add_message(chat_a, "assistant", "Atlas is a document intelligence platform.")

    storage.add_message(chat_b, "user", "Hello from user B")
    storage.add_message(chat_b, "assistant", "Greetings, this is session B.")

    hist_a = client.get(f"/chat/{chat_a}/history").json()["history"]
    hist_b = client.get(f"/chat/{chat_b}/history").json()["history"]
    assert len(hist_a) == 2
    assert hist_a[0]["content"] == "What is Atlas?"
    assert len(hist_b) == 2
    assert hist_b[0]["content"] == "Hello from user B"
    print("Session history isolation verified!")

    print("\n--- 4. Testing Session Memory Diagnostic Endpoint ---")
    mem_res = client.get(f"/memory/chat/{chat_a}")
    assert mem_res.status_code == 200
    mem_data = mem_res.json()
    assert mem_data["chat_id"] == chat_a
    assert mem_data["recent_turns_count"] == 2
    print(f"Memory diagnostic for Chat A: {mem_data['recent_turns_count']} turns recorded.")

    print("\n--- 5. Testing Detach Single Document ---")
    detach_res = client.delete(f"/chat/{chat_a}/document/{doc1_id}")
    assert detach_res.status_code == 200
    remaining_docs = client.get(f"/chat/{chat_a}/documents").json()["documents"]
    assert len(remaining_docs) == 1
    assert remaining_docs[0]["doc_id"] == doc2_id
    print(f"Remaining document in Chat A: {remaining_docs[0]['filename']}")

    print("\n--- 6. Testing Global Knowledge & Dossier Storage ---")
    # Save a test dossier
    persistent_db.save_knowledge_dossier(
        doc_id=doc1_id,
        filename="Architecture_Guide.pdf",
        executive_summary="Explains Atlas tree-based vectorless RAG architecture.",
        key_entities=["Atlas", "PageIndex", "Groq", "FastAPI"],
        key_findings=["Tree search outperforms embeddings on dense tabular texts."],
        topics=["Architecture", "Document Intelligence", "FastAPI"],
        owner_id=TEST_CLIENT_ID,
    )

    persistent_db.save_global_knowledge_summary(
        collective_summary="Atlas workspace contains technical documentation on tree retrieval and quarterly financial reports.",
        total_documents=2,
        key_insights=["Tree retrieval avoids embedding drift", "Multi-doc synthesis enables cross-sectional queries"],
        owner_id=TEST_CLIENT_ID,
    )

    g_res = client.get("/memory/knowledge")
    assert g_res.status_code == 200
    g_data = g_res.json()
    print(f"Global Knowledge Summary: {g_data['collective_summary']}")
    print(f"Dossiers count: {len(g_data['dossiers'])}")
    assert len(g_data["dossiers"]) >= 1
    assert "Atlas" in g_data["collective_summary"]

    print("\n ALL BACKEND MULTI-DOC & MEMORY TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_full_memory_and_multidoc_isolation()

import os
import sys
from fastapi.testclient import TestClient

# Ensure app path is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

from app import app
from core.storage import storage

client = TestClient(app)


def test_studio_mindmap_and_quiz_endpoints():
    print("\n--- RUNNING PHASE 4 STUDIO BACKEND TEST SUITE ---")
    
    # 1. Create a test document in storage
    doc_id = "test-studio-doc-123"
    client_id = "00000000-0000-4000-8000-000000000001"
    headers = {"X-Client-ID": client_id}

    tree = [
        {
            "node_id": "ch_1",
            "title": "1. Introduction & Vectorless Paradigm",
            "page_index": 1,
            "summary": "Introduction to PageIndex vectorless RAG architecture.",
            "nodes": [
                {
                    "node_id": "sec_1_1",
                    "title": "1.1 The Pitfalls of Vector Chunking",
                    "page_index": 2,
                    "summary": "Chunking causes boundary loss and high maintenance costs.",
                }
            ]
        },
        {
            "node_id": "ch_2",
            "title": "2. Hierarchical Tree Traversal",
            "page_index": 5,
            "summary": "Describes DocumentService and LLM reasoning over trees.",
            "nodes": [
                {
                    "node_id": "sec_2_1",
                    "title": "2.1 BM25 Lexical Boosting",
                    "page_index": 7,
                    "summary": "Pure Python BM25 scoring in under 1ms.",
                }
            ]
        }
    ]

    storage.create_document(
        doc_id=doc_id,
        filename="Atlas_Architecture.pdf",
        status="ready",
        owner_id=client_id,
    )
    storage.set_document_ready(
        doc_id=doc_id,
        tree=tree,
        filename="Atlas_Architecture.pdf",
        page_count=10,
        section_count=4,
    )

    # Save a test knowledge dossier
    storage.save_knowledge_dossier(
        doc_id=doc_id,
        filename="Atlas_Architecture.pdf",
        executive_summary="Atlas is an enterprise vectorless RAG platform with hierarchical tree search.",
        key_entities=["Atlas", "PageIndex", "Groq", "FastAPI"],
        key_findings=["Tree search outperforms embeddings on dense technical documents.", "Sub-node pruning preserves exact page citations."],
        topics=["Vectorless Architecture", "Hierarchical Traversal", "Lexical BM25"],
        owner_id=client_id,
    )

    # 2. Test GET /studio/{doc_id}/mindmap
    print("\n[TEST 1] Testing GET /studio/{doc_id}/mindmap...")
    res = client.get(f"/studio/{doc_id}/mindmap", headers=headers)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert data["doc_id"] == doc_id
    assert data["filename"] == "Atlas_Architecture.pdf"
    assert "nodes" in data and "edges" in data
    nodes = data["nodes"]
    edges = data["edges"]
    print(f"Mind map generated with {len(nodes)} nodes and {len(edges)} edges.")

    # Validate root node
    root_node = next((n for n in nodes if n["type"] == "root"), None)
    assert root_node is not None, "Missing root node"
    assert root_node["section_title"] == "Atlas_Architecture.pdf"
    assert root_node["page_number"] == 1

    # Validate dossier hub & topic nodes
    dossier_hub = next((n for n in nodes if n["type"] == "dossier"), None)
    assert dossier_hub is not None, "Missing dossier hub node"
    topic_nodes = [n for n in nodes if n["type"] == "topic"]
    assert len(topic_nodes) >= 1, "Expected topic nodes from dossier"

    # Validate chapters & sections
    chapters = [n for n in nodes if n["type"] == "chapter"]
    sections = [n for n in nodes if n["type"] == "section"]
    assert len(chapters) == 2, f"Expected 2 chapters, got {len(chapters)}"
    assert len(sections) == 2, f"Expected 2 sub-sections, got {len(sections)}"

    # Validate that every single node has required study studio fields
    for n in nodes:
        assert "id" in n and n["id"]
        assert "label" in n and n["label"]
        assert "section_title" in n and n["section_title"]
        assert "page_number" in n and n["page_number"] is not None
        assert "type" in n
        assert "level" in n

    # Validate edges
    assert len(edges) >= len(nodes) - 1, "Graph should be connected"
    print("✓ Test 1 Passed: GET /studio/{doc_id}/mindmap correctly transforms tree & dossier into graph!")

    # 3. Test POST /studio/{doc_id}/quiz
    print("\n[TEST 2] Testing POST /studio/{doc_id}/quiz...")
    quiz_res = client.post(f"/studio/{doc_id}/quiz", headers=headers, json={"num_mcqs": 5, "num_flashcards": 5})
    assert quiz_res.status_code == 200, f"Expected 200, got {quiz_res.status_code}: {quiz_res.text}"
    quiz_data = quiz_res.json()

    assert quiz_data["doc_id"] == doc_id
    assert "mcqs" in quiz_data and "flashcards" in quiz_data
    mcqs = quiz_data["mcqs"]
    flashcards = quiz_data["flashcards"]

    print(f"Quiz generated with {len(mcqs)} MCQs and {len(flashcards)} Flashcards.")
    assert len(mcqs) == 5, f"Expected exactly 5 MCQs, got {len(mcqs)}"
    assert len(flashcards) == 5, f"Expected exactly 5 Flashcards, got {len(flashcards)}"

    # Validate MCQs structure & citations
    for idx, q in enumerate(mcqs):
        assert "question" in q and q["question"], f"MCQ {idx} missing question"
        assert "options" in q and len(q["options"]) == 4, f"MCQ {idx} must have 4 choices, got {len(q.get('options', []))}"
        assert "answer" in q and q["answer"], f"MCQ {idx} missing answer"
        assert "explanation" in q and q["explanation"], f"MCQ {idx} missing explanation"
        assert "citation" in q and q["citation"], f"MCQ {idx} missing citation"
        assert "(Section:" in q["citation"] and "Page:" in q["citation"], f"MCQ {idx} citation malformed: {q['citation']}"

    # Validate Flashcards structure & citations
    for idx, c in enumerate(flashcards):
        assert "question" in c and c["question"], f"Flashcard {idx} missing question"
        assert "answer" in c and c["answer"], f"Flashcard {idx} missing answer"
        assert "explanation" in c and c["explanation"], f"Flashcard {idx} missing explanation"
        assert "citation" in c and c["citation"], f"Flashcard {idx} missing citation"
        assert "(Section:" in c["citation"] and "Page:" in c["citation"], f"Flashcard {idx} citation malformed: {c['citation']}"

    print(f"Sample MCQ 1 Citation: {mcqs[0]['citation']}")
    print(f"Sample Flashcard 1 Citation: {flashcards[0]['citation']}")
    print("✓ Test 2 Passed: POST /studio/{doc_id}/quiz generated 5 MCQs & 5 Flashcards with exact citations!")

    # 4. Test 404 for unknown document
    print("\n[TEST 3] Testing 404 error handling for non-existent doc...")
    err_res = client.get("/studio/non-existent-doc-999/mindmap", headers=headers)
    assert err_res.status_code == 404
    err_res2 = client.post("/studio/non-existent-doc-999/quiz", headers=headers)
    assert err_res2.status_code == 404
    print("✓ Test 3 Passed: 404 error handling verified!")

    print("\n=======================================================")
    print(" ALL 3 PHASE 4 STUDIO BACKEND TESTS PASSED PERFECTLY!")
    print("=======================================================\n")


if __name__ == "__main__":
    test_studio_mindmap_and_quiz_endpoints()

import uuid
import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_public_endpoints_accessible_without_client_id(client):
    """Verifies that public endpoints (/, /health, /docs, /openapi.json) require no X-Client-ID."""
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "message" in r_root.json()

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_docs = client.get("/docs")
    assert r_docs.status_code == 200

    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200


def test_protected_endpoints_fallback_missing_client_id(client):
    """Protected endpoints gracefully fall back to default client ID when X-Client-ID is missing."""
    r = client.get("/chats")
    assert r.status_code == 200
    assert "chats" in r.json()

    r_docs = client.get("/documents")
    assert r_docs.status_code == 200
    assert "documents" in r_docs.json()

    r_tools = client.post("/tools/calculate-math", json={"expression": "1 + 1"})
    assert r_tools.status_code == 200
    assert r_tools.json()["result"] == 2


def test_protected_endpoints_fallback_invalid_uuid(client):
    """Protected endpoints gracefully handle non-UUID string without throwing 400 error."""
    bad_headers = {"X-Client-ID": "custom-client-string"}
    r = client.get("/chats", headers=bad_headers)
    assert r.status_code == 200
    assert "chats" in r.json()


def test_chat_isolation_between_clients(client):
    """Client A and Client B must have complete data isolation for chats and messages."""
    client_a_id = str(uuid.uuid4())
    client_b_id = str(uuid.uuid4())

    headers_a = {"X-Client-ID": client_a_id}
    headers_b = {"X-Client-ID": client_b_id}

    # 1. Client A creates a chat
    create_res = client.post("/chat/new", headers=headers_a)
    assert create_res.status_code == 200
    chat_a_id = create_res.json()["chat_id"]

    # 2. Client A lists chats -> finds chat_a_id
    list_a = client.get("/chats", headers=headers_a).json()["chats"]
    assert any(c["chat_id"] == chat_a_id for c in list_a)

    # 3. Client B lists chats -> must NOT find chat_a_id
    list_b = client.get("/chats", headers=headers_b).json()["chats"]
    assert not any(c["chat_id"] == chat_a_id for c in list_b)

    # 4. Client B attempts to fetch Client A's chat history -> 404
    hist_b = client.get(f"/chat/{chat_a_id}/history", headers=headers_b)
    assert hist_b.status_code == 404

    # 5. Client B attempts to delete Client A's chat -> 404
    del_res = client.delete(f"/chat/{chat_a_id}", headers=headers_b)
    assert del_res.status_code == 404

    # 6. Client A renames chat -> succeeds
    rename_res = client.patch(f"/chat/{chat_a_id}/title", headers=headers_a, json={"title": "Client A Topic"})
    assert rename_res.status_code == 200

    # 7. Client B attempts to rename Client A's chat -> 404
    rename_b = client.patch(f"/chat/{chat_a_id}/title", headers=headers_b, json={"title": "Hacked Title"})
    assert rename_b.status_code == 404

    # 8. Client A deletes chat -> succeeds
    del_res_a = client.delete(f"/chat/{chat_a_id}", headers=headers_a)
    assert del_res_a.status_code == 200


def test_tools_endpoint_accessible_with_valid_client_id(client):
    """Tools endpoint succeeds with valid client_id."""
    client_id = str(uuid.uuid4())
    headers = {"X-Client-ID": client_id}
    res = client.post("/tools/calculate-math", headers=headers, json={"expression": "10 * 5"})
    assert res.status_code == 200
    assert res.json()["result"] == 50


def test_document_and_memory_isolation_between_clients(client):
    """Client A's documents, dossiers, and summaries must remain invisible to Client B."""
    from core.dependencies import get_storage
    storage = get_storage()

    client_a = str(uuid.uuid4())
    client_b = str(uuid.uuid4())

    doc_a_id = f"doc_test_{uuid.uuid4().hex[:8]}"

    # Seed document metadata for Client A directly into storage
    storage.create_document(
        doc_id=doc_a_id,
        filename="client_a_contract.pdf",
        content_hash="hash_a_12345",
        file_path="/tmp/fake_contract.pdf",
        owner_id=client_a,
    )
    storage.save_knowledge_dossier(
        doc_id=doc_a_id,
        filename="client_a_contract.pdf",
        executive_summary="Confidential contract for Client A",
        key_entities=["Client A Corp"],
        key_findings=["Revenue target $10M"],
        topics=["Confidential", "Finance"],
        owner_id=client_a,
    )

    # Client A lists documents -> sees doc_a_id
    headers_a = {"X-Client-ID": client_a}
    docs_a = client.get("/documents", headers=headers_a).json()["documents"]
    assert any(d["doc_id"] == doc_a_id for d in docs_a)

    # Client B lists documents -> does NOT see doc_a_id
    headers_b = {"X-Client-ID": client_b}
    docs_b = client.get("/documents", headers=headers_b).json()["documents"]
    assert not any(d["doc_id"] == doc_a_id for d in docs_b)

    # Client B attempts to fetch Client A's document details -> 404
    get_doc_b = client.get(f"/documents/{doc_a_id}", headers=headers_b)
    assert get_doc_b.status_code == 404

    # Client B attempts to fetch Client A's dossier -> 404
    dossier_b = client.get(f"/memory/dossier/{doc_a_id}", headers=headers_b)
    assert dossier_b.status_code == 404

    # Client A fetches dossier -> 200
    dossier_a = client.get(f"/memory/dossier/{doc_a_id}", headers=headers_a)
    assert dossier_a.status_code == 200
    assert dossier_a.json()["executive_summary"] == "Confidential contract for Client A"

    # Client B attempts to delete Client A's document -> 404
    del_b = client.delete(f"/documents/{doc_a_id}", headers=headers_b)
    assert del_b.status_code == 404

    # Client A deletes document -> 200
    del_a = client.delete(f"/documents/{doc_a_id}", headers=headers_a)
    assert del_a.status_code == 200


def test_delete_all_documents_bulk_endpoint(client):
    """Verifies that DELETE /documents deletes all documents belonging to the requesting client."""
    from core.dependencies import get_storage
    storage = get_storage()

    client_c = str(uuid.uuid4())
    client_d = str(uuid.uuid4())
    headers_c = {"X-Client-ID": client_c}
    headers_d = {"X-Client-ID": client_d}

    doc_c1 = f"doc_c1_{uuid.uuid4().hex[:8]}"
    doc_c2 = f"doc_c2_{uuid.uuid4().hex[:8]}"
    doc_d = f"doc_d_{uuid.uuid4().hex[:8]}"

    storage.create_document(doc_id=doc_c1, filename="doc1.pdf", owner_id=client_c)
    storage.create_document(doc_id=doc_c2, filename="doc2.pdf", owner_id=client_c)
    storage.create_document(doc_id=doc_d, filename="doc_d.pdf", owner_id=client_d)

    # Client C deletes all
    del_res = client.delete("/documents", headers=headers_c)
    assert del_res.status_code == 200
    data = del_res.json()
    assert data["count"] >= 2

    # Client C documents list is now empty of c1 and c2
    docs_c = client.get("/documents", headers=headers_c).json()["documents"]
    assert not any(d["doc_id"] in [doc_c1, doc_c2] for d in docs_c)

    # Client D document is still intact
    docs_d = client.get("/documents", headers=headers_d).json()["documents"]
    assert any(d["doc_id"] == doc_d for d in docs_d)

    # Cleanup Client D
    client.delete(f"/documents/{doc_d}", headers=headers_d)


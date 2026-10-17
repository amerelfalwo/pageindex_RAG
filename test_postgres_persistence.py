import os
import sys

# Ensure app is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from dotenv import load_dotenv
load_dotenv()

from core.persistent_db import persistent_db
from core.postgres_db import PostgresDB
from core.storage import MemoryStorage

def test_persistence():
    print("Testing PersistentDB connection...")
    print(f"PersistentDB active instance: {type(persistent_db).__name__}")
    assert isinstance(persistent_db, PostgresDB), "Expected PostgresDB to be active!"

    # 1. Test Chat & Message CRUD
    chat_id = "test-chat-pg-1"
    persistent_db.create_chat(chat_id=chat_id, title="Test PostgreSQL Chat")
    chat = persistent_db.get_chat(chat_id)
    assert chat is not None, "Failed to get chat"
    assert chat["title"] == "Test PostgreSQL Chat"

    persistent_db.add_message(chat_id, "user", "Hello Postgres!")
    persistent_db.add_message(chat_id, "assistant", "Hello! I remember everything.")
    msgs = persistent_db.get_messages(chat_id)
    assert len(msgs) == 2, f"Expected 2 messages, got {len(msgs)}"
    assert msgs[0]["content"] == "Hello Postgres!"
    print("✓ Chat & Messages working!")

    # 2. Test Document Meta & Index Storage
    doc_id = "test-doc-pg-1"
    filename = "report_2026.pdf"
    content_hash = "abcdef1234567890"

    persistent_db.upsert_doc_meta(
        doc_id=doc_id,
        filename=filename,
        status="ready",
        page_count=15,
        section_count=4,
        progress=100,
        current_stage="Ready",
        indexing_method="pageindex",
        duration=3.5,
        content_hash=content_hash,
    )

    test_tree = [
        {"title": "Executive Summary", "page_index": 1, "children": []},
        {"title": "Financial Results", "page_index": 5, "children": [
            {"title": "Q1 Revenue", "page_index": 6, "children": []}
        ]}
    ]
    test_node_map = {"node_1": {"title": "Executive Summary", "page": 1}}
    test_sections = [{"title": "Executive Summary", "content": "Summary text", "page": 1}]

    persistent_db.save_doc_index(
        doc_id=doc_id,
        content_hash=content_hash,
        tree=test_tree,
        node_map=test_node_map,
        flat_sections=test_sections,
        file_base64="dGVzdCBwZGYgY29udGVudA==",
    )

    # Read back index
    idx = persistent_db.get_doc_index(doc_id)
    assert idx is not None, "Failed to read doc index"
    assert idx["tree"] == test_tree, "Tree does not match!"
    assert idx["node_map"] == test_node_map, "Node map does not match!"
    assert idx["flat_sections"] == test_sections, "Flat sections do not match!"
    print("✓ Document Index (JSONB) storage working!")

    # 3. Test Chat Documents
    persistent_db.add_chat_document(chat_id, doc_id, filename)
    attached = persistent_db.get_chat_documents(chat_id)
    assert len(attached) == 1, f"Expected 1 attached doc, got {len(attached)}"
    assert attached[0]["doc_id"] == doc_id
    print("✓ Chat Documents working!")

    # 4. Test MemoryStorage Integration with Simulated Cache Wipe
    storage = MemoryStorage()
    # Notice: _trees and _documents in storage are currently empty!
    # get_document should fetch from PostgreSQL, restore tree and sections!
    restored_doc = storage.get_document(doc_id)
    assert restored_doc is not None, "Failed to get document via storage"
    assert restored_doc["status"] == "ready"
    assert restored_doc["tree"] == test_tree, "Restored tree does not match!"
    assert storage.get_node_map(doc_id) == test_node_map, "Restored node map does not match!"
    print("✓ MemoryStorage seamlessly loaded wiped document from PostgreSQL!")

    # Cleanup test data
    persistent_db.delete_chat(chat_id)
    persistent_db.delete_doc(doc_id)
    print("✓ All tests passed successfully!")

if __name__ == "__main__":
    test_persistence()

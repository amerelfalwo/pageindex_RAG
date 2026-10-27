import os
import sys
import time
import asyncio
from typing import List, Dict, Any

# Ensure app path is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

from core.intent_classifier import intent_classifier, QueryIntent
from core.rag_pipeline import VectorlessRAG
from core.storage import MemoryStorage


def test_query_intent_classifier():
    """Verify ultra-fast intent classification for Global, Taxonomy, Cross-Doc, and Specific queries."""
    test_cases = [
        # English Global
        ("Give me an executive summary of this document", 1, QueryIntent.GLOBAL_SUMMARY, True),
        ("What is this paper about?", 1, QueryIntent.GLOBAL_SUMMARY, True),
        ("Overview of the main findings", 1, QueryIntent.GLOBAL_SUMMARY, True),
        ("summary", 1, QueryIntent.GLOBAL_SUMMARY, True),
        ("tl;dr of the document", 1, QueryIntent.GLOBAL_SUMMARY, True),
        # Arabic Global
        ("ما هو ملخص هذا البحث؟", 1, QueryIntent.GLOBAL_SUMMARY, True),
        ("لخص لي محتوى المستند", 1, QueryIntent.GLOBAL_SUMMARY, True),
        ("دا عن ايه", 1, QueryIntent.GLOBAL_SUMMARY, True),
        ("لخص", 1, QueryIntent.GLOBAL_SUMMARY, True),
        # Taxonomy & Architecture
        ("What is the taxonomy and architecture proposed?", 1, QueryIntent.TAXONOMY_ARCHITECTURE, True),
        ("ما هي معمارية وهيكلية النظام؟", 1, QueryIntent.TAXONOMY_ARCHITECTURE, True),
        ("Explain the workflow pipeline and components", 1, QueryIntent.TAXONOMY_ARCHITECTURE, True),
        # Cross-Doc Comparison
        ("Compare the methodology across both documents", 2, QueryIntent.CROSS_DOC_SUMMARY, True),
        ("قارن بين نتائج المستندين", 2, QueryIntent.CROSS_DOC_SUMMARY, True),
        # Specific factual queries (MUST NOT be classified as global)
        ("What was the net profit in section 4?", 1, QueryIntent.SPECIFIC_FACT, False),
        ("كم بلغت الإيرادات في الربع الثالث؟", 1, QueryIntent.SPECIFIC_FACT, False),
        ("Who is the author mentioned on page 5?", 1, QueryIntent.SPECIFIC_FACT, False),
    ]

    t0 = time.monotonic()
    for query, doc_count, expected_intent, expected_global in test_cases:
        res = intent_classifier.classify(query, doc_count=doc_count)
        assert res["intent"] == expected_intent, (
            f"Query '{query}' failed! Expected intent {expected_intent}, got {res['intent']}"
        )
        assert res["is_global_or_summary"] == expected_global, (
            f"Query '{query}' failed! Expected is_global={expected_global}, got {res['is_global_or_summary']}"
        )
    total_ms = (time.monotonic() - t0) * 1000
    avg_us = (total_ms / len(test_cases)) * 1000
    print(f"Tested {len(test_cases)} query intents in {total_ms:.3f}ms ({avg_us:.1f}µs per query).")
    assert total_ms < 10.0, f"Classification took too long: {total_ms}ms"
    print("✓ Test 1 Passed: QueryIntentClassifier accurately classifies queries in sub-millisecond time!")


def test_dossier_and_root_chapter_node_creation():
    """Verify dossier formatting, root chapter extraction, and fallback dossier generation."""
    mock_dossier = {
        "executive_summary": "Atlas is a vectorless RAG platform with hierarchical tree search.",
        "key_entities": ["Atlas", "PageIndex", "Groq", "FastAPI"],
        "key_findings": ["Eliminates chunking drift", "Enables exact page-level citations"],
        "topics": ["Architecture", "Document Intelligence", "FastAPI"]
    }

    # Test Dossier Node creation
    dossier_node = VectorlessRAG._create_dossier_node(mock_dossier, doc_name="Atlas_Spec.pdf", doc_id="doc_101")
    assert dossier_node["node_id"] == "dossier_doc_101"
    assert dossier_node["page_index"] == 1
    assert dossier_node["is_dossier"] is True
    assert "--- [PAGE 1] ---" in dossier_node["text"]
    assert "Eliminates chunking drift" in dossier_node["text"]
    assert "Atlas, PageIndex" in dossier_node["text"]

    # Test Root Chapter Extraction
    tree = [
        {
            "node_id": "ch_1",
            "title": "1. Introduction and Motivation",
            "page_index": 1,
            "summary": "Explains vectorless motivation and pitfalls of embedding-based RAG.",
        },
        {
            "node_id": "ch_2",
            "title": "2. Hierarchical Tree Architecture",
            "page_index": 5,
            "summary": "Describes DocumentService and LLM tree traversal algorithms.",
        },
        {
            "node_id": "ch_3",
            "title": "3. Benchmark Evaluation",
            "page_index": 18,
            "summary": "Presents evaluation across 80 questions with 87.7% correctness.",
        }
    ]
    root_nodes = VectorlessRAG._extract_root_chapter_nodes(tree, doc_name="Atlas_Spec.pdf", doc_id="doc_101")
    assert len(root_nodes) == 3
    assert root_nodes[0]["title"] == "1. Introduction and Motivation"
    assert root_nodes[0]["page_index"] == 1
    assert "--- [PAGE 1] ---" in root_nodes[0]["text"]
    assert root_nodes[1]["page_index"] == 5
    assert "--- [PAGE 5] ---" in root_nodes[1]["text"]
    assert root_nodes[2]["page_index"] == 18
    assert "--- [PAGE 18] ---" in root_nodes[2]["text"]

    # Test Fallback Dossier Generation when background dossier task is pending
    fallback = VectorlessRAG._create_fallback_dossier_from_tree(tree, "Atlas_Spec.pdf")
    assert "Introduction and Motivation" in fallback["executive_summary"]
    assert len(fallback["topics"]) == 3
    assert len(fallback["key_findings"]) >= 1

    print("✓ Test 2 Passed: Dossier and root chapter node creation verified with exact page markers!")


def test_pruning_preserves_dossier_and_root_chapters():
    """Verify that Sub-Node Paragraph Pruning does not truncate or lose dossier / root chapter content."""
    long_text = "Executive overview paragraph.\n\n" + ("Detailed point.\n\n" * 15)
    dossier_node = {
        "node_id": "dossier_doc_1",
        "title": "Executive Knowledge Dossier & Core Findings",
        "page_index": 1,
        "text": long_text,
        "is_dossier": True,
    }
    root_node = {
        "node_id": "root_0_doc_1",
        "title": "Chapter 1: Overview",
        "page_index": 1,
        "text": long_text,
        "is_root_chapter": True,
    }
    regular_node = {
        "node_id": "sec_04",
        "title": "Section 4: Technical Specifications",
        "page_index": 12,
        "text": ("--- [PAGE 12] ---\nParagraph 1 irrelevant data.\n\n"
                 "--- [PAGE 13] ---\nParagraph 2 contains the specific target metric 99.4% uptime.\n\n"
                 "Paragraph 3 more background noise.\n\n" * 10),
    }

    # Query asking for summary
    query = "Give me an executive summary and overview"
    pruned = VectorlessRAG.prune_sub_nodes(query, [dossier_node, root_node, regular_node])
    
    assert len(pruned) == 3
    # Dossier and root chapter must be preserved completely
    assert pruned[0]["node_id"] == "dossier_doc_1"
    assert pruned[0]["text"] == long_text
    assert pruned[1]["node_id"] == "root_0_doc_1"
    assert pruned[1]["text"] == long_text

    # Regular node should undergo pruning
    assert pruned[2]["node_id"] == "sec_04"
    assert pruned[2].get("is_pruned") is True
    assert len(pruned[2]["text"]) < len(regular_node["text"])

    print("✓ Test 3 Passed: Sub-Node Paragraph Pruning preserves dossier and root chapters without loss!")


def test_end_to_end_global_retrieval_and_citations():
    """Verify end-to-end global query execution with automatic dossier & root injection."""
    class MockStorage:
        def __init__(self):
            self.dossiers = {
                "doc_alpha": {
                    "executive_summary": "Alpha Report discusses deep neural architectures.",
                    "key_entities": ["Transformer", "Attention", "Sparsity"],
                    "key_findings": ["Sparse attention reduces compute by 40%."],
                    "topics": ["Deep Learning", "Efficiency"]
                }
            }
            self.global_summary = {
                "collective_summary": "Collective workspace knowledge covers neural efficiency and tree search.",
                "key_insights": ["Neural sparsity synergizes with tree-based indexing."]
            }

        def get_knowledge_dossier(self, doc_id: str):
            return self.dossiers.get(doc_id)

        def get_global_knowledge_summary(self):
            return self.global_summary

    class MockLLMService:
        def __init__(self):
            self.captured_nodes = []
            self.captured_global_knowledge = None

        async def llm_tree_search(self, query, tree, chat_history=None, flat_sections=None):
            return {"node_list": ["sub_sec_3"]}

        def find_nodes_by_ids(self, tree, target_ids, node_map=None):
            return [{"node_id": "sub_sec_3", "title": "Sub Section 3: Sparsity Math", "page_index": 7, "text": "--- [PAGE 7] ---\nMath details"}]

        async def generate_answer_stream(self, query, nodes, chat_history=None, doc_name=None,
                                         session_summary=None, global_knowledge=None, deep_analysis=False,
                                         web_results=None):
            self.captured_nodes = nodes
            self.captured_global_knowledge = global_knowledge
            # Yield simulated citation
            yield f'According to (Section: "Executive Knowledge Dossier & Core Findings", Page: 1) and (Section: "1. Introduction", Page: 1), Alpha Report covers neural efficiency.'

    mock_storage = MockStorage()
    mock_llm = MockLLMService()
    rag = VectorlessRAG(mock_llm, storage_instance=mock_storage)

    doc_alpha = {
        "doc_id": "doc_alpha",
        "filename": "Alpha_Report.pdf",
        "tree": [
            {"node_id": "root_ch1", "title": "1. Introduction", "page_index": 1, "summary": "Intro to neural networks."},
            {"node_id": "root_ch2", "title": "2. Sparse Attention", "page_index": 4, "summary": "Mathematical formulation of sparse matrices."},
        ],
        "node_map": {
            "sub_sec_3": {"node_id": "sub_sec_3", "title": "Sub Section 3: Sparsity Math", "page_index": 7, "text": "--- [PAGE 7] ---\nMath details"}
        },
        "flat_sections": [
            {"node_id": "sub_sec_3", "title": "Sub Section 3: Sparsity Math", "summary": "Math details"}
        ]
    }

    async def run_pipeline():
        query = "Give me an executive summary and overview of this document"
        stream = rag.answer_multi_doc_query_stream(
            query=query,
            docs=[doc_alpha],
            chat_history=[],
        )
        response_chunks = []
        async for chunk in stream:
            response_chunks.append(chunk)

        full_response = "".join(response_chunks)
        
        # Verify captured nodes in LLM call
        captured = mock_llm.captured_nodes
        assert len(captured) >= 3, f"Expected at least 3 nodes (dossier + 2 root chapters + sub_node), got {len(captured)}"
        
        # Verify node 0 is Executive Dossier
        assert captured[0]["is_dossier"] is True
        assert captured[0]["title"] == "Executive Knowledge Dossier & Core Findings"
        assert captured[0]["page_index"] == 1
        assert "Sparse attention reduces compute by 40%" in captured[0]["text"]

        # Verify node 1 and 2 are root chapters
        assert captured[1]["is_root_chapter"] is True
        assert captured[1]["title"] == "1. Introduction"
        assert captured[1]["page_index"] == 1
        assert captured[2]["is_root_chapter"] is True
        assert captured[2]["title"] == "2. Sparse Attention"
        assert captured[2]["page_index"] == 4

        # Verify node 3 is the retrieved sub-node
        assert captured[3]["node_id"] == "sub_sec_3"

        # Verify citations in response
        assert '(Section: "Executive Knowledge Dossier & Core Findings", Page: 1)' in full_response
        assert '(Section: "1. Introduction", Page: 1)' in full_response

        print("✓ Test 4 Passed: End-to-end global query correctly prepended dossier & root chapters with exact citations!")

    asyncio.run(run_pipeline())


if __name__ == "__main__":
    print("\n--- RUNNING PHASE 3 GLOBAL & SUMMARY RETRIEVAL TEST SUITE ---")
    test_query_intent_classifier()
    test_dossier_and_root_chapter_node_creation()
    test_pruning_preserves_dossier_and_root_chapters()
    test_end_to_end_global_retrieval_and_citations()
    print("\n=======================================================")
    print(" ALL 4 PHASE 3 GLOBAL & SUMMARY TESTS PASSED PERFECTLY!")
    print("=======================================================\n")

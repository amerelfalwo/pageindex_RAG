import os
import sys
import time
import asyncio

# Ensure app path is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

from core.bm25 import BM25NodeIndex
from core.rag_pipeline import VectorlessRAG


def test_bm25_node_index_scoring_and_latency():
    """Verify BM25NodeIndex scores nodes correctly and executes in under 20ms."""
    nodes = [
        {
            "node_id": "sec_01",
            "title": "Introduction to Vectorless RAG",
            "summary": "Overview of tree-based hierarchical indexing without vector databases or chunking.",
            "text": "Traditional RAG depends on embedding models and vector similarity. Vectorless RAG navigates the document tree directly."
        },
        {
            "node_id": "sec_02",
            "title": "Knowledge Graph Representation Learning",
            "summary": "Covers TransE, ERNIE, and entity embedding alignments.",
            "text": "ERNIE fuses entity representations into text tokens by utilizing TransE knowledge-graph embeddings with multi-layer encoders."
        },
        {
            "node_id": "sec_03",
            "title": "Evaluation Benchmark & Cost",
            "summary": "Benchmark results on 80 questions with 87.7% correctness.",
            "text": "Evaluated on an 80-question benchmark with 1.8s avg latency and $0.0009 cost per query."
        },
    ]

    t0 = time.monotonic()
    index = BM25NodeIndex(nodes)
    build_time_ms = (time.monotonic() - t0) * 1000

    # Search query specifically targeting Section 2
    query = "How does ERNIE integrate TransE embeddings into language models?"
    t_search_0 = time.monotonic()
    results = index.search(query, top_k=3)
    search_time_ms = (time.monotonic() - t_search_0) * 1000

    print(f"BM25 build time: {build_time_ms:.2f}ms, search time: {search_time_ms:.2f}ms")
    assert search_time_ms < 20.0, f"Lexical search exceeded 20ms limit: {search_time_ms}ms"
    assert len(results) > 0, "Expected at least 1 BM25 match"
    
    # Top match must be sec_02 (ERNIE / TransE)
    top_nid, top_score = results[0]
    assert top_nid == "sec_02", f"Expected sec_02 to rank highest, got {top_nid}"
    assert top_score > 0.0

    print("✓ Test 1 Passed: BM25NodeIndex accurately scored relevant section in under 20ms!")


def test_bm25_arabic_support_and_title_boost():
    """Verify Arabic tokenization and title field weighting."""
    nodes = [
        {
            "node_id": "ar_01",
            "title": "بنية الذاكرة المتعددة",
            "summary": "نظام الذاكرة ثلاثي الطبقات على PostgreSQL.",
            "text": "عزل محادثات الجلسات وسجلات المستخدمين وتوليد ملخصات تلقائية."
        },
        {
            "node_id": "ar_02",
            "title": "البحث الحي عبر FastMCP",
            "summary": "استرجاع نتائج الويب والتحقق من صحة الروابط.",
            "text": "فحص الروابط الحية والتأكد من توافر المحتوى المطلوب."
        }
    ]
    index = BM25NodeIndex(nodes)
    results = index.search("ما هو نظام الذاكرة في قاعدة البيانات؟", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "ar_01", f"Expected ar_01 to rank highest for memory query, got {results[0][0]}"
    print("✓ Test 2 Passed: Arabic tokenization and field weighting verified!")


def test_weighted_reciprocal_rank_fusion():
    """Verify Weighted Reciprocal Rank Fusion combines LLM and BM25 results with 0.6 and 0.4 weights."""
    llm_ranked = ["sec_02", "sec_01"]
    bm25_ranked = ["sec_02", "sec_03", "sec_01"]

    fused = VectorlessRAG.reciprocal_rank_fusion(
        ranked_lists=[(llm_ranked, 0.6), (bm25_ranked, 0.4)],
        rrf_k=60,
        top_k=3,
    )
    
    fused_nids = [nid for nid, sc in fused]
    # sec_02 is Rank 1 in both lists -> must be top fused
    assert fused_nids[0] == "sec_02", f"Expected sec_02 at top of RRF, got {fused_nids[0]}"
    
    # Calculate expected mathematical RRF scores:
    # sec_02: 0.6/(60+1) + 0.4/(60+1) = 1.0 / 61 = ~0.016393
    # sec_01: 0.6/(60+2) + 0.4/(60+3) = 0.6/62 + 0.4/63 = ~0.009677 + ~0.006349 = ~0.016026
    # sec_03: 0.0 + 0.4/(60+2) = 0.4/62 = ~0.006452
    score_sec02 = dict(fused)["sec_02"]
    score_sec01 = dict(fused)["sec_01"]
    score_sec03 = dict(fused)["sec_03"]
    assert score_sec02 > score_sec01 > score_sec03
    print(f"✓ Test 3 Passed: Weighted RRF correctly prioritized agreed nodes (Scores: sec_02={score_sec02:.6f}, sec_01={score_sec01:.6f}, sec_03={score_sec03:.6f})!")


def test_parallel_hybrid_retrieval_multi_doc():
    """Verify concurrent execution of tree search + BM25 across multiple attached documents."""
    class MockLLMService:
        async def llm_tree_search(self, query, tree, chat_history=None, flat_sections=None):
            # Simulate ~30ms async LLM search
            await asyncio.sleep(0.03)
            # LLM picks section 1
            return {"node_list": ["doc1_sec1"]}

        def find_nodes_by_ids(self, tree, target_ids, node_map=None):
            if node_map:
                return [node_map[nid] for nid in target_ids if nid in node_map]
            return [{"node_id": nid, "title": f"Title {nid}", "text": "Content"} for nid in target_ids]

    mock_llm = MockLLMService()
    rag = VectorlessRAG(mock_llm)

    doc1 = {
        "doc_id": "doc_1",
        "filename": "Doc1.pdf",
        "tree": [],
        "node_map": {
            "doc1_sec1": {"node_id": "doc1_sec1", "title": "Overview", "text": "General overview"},
            "doc1_sec2": {"node_id": "doc1_sec2", "title": "ERNIE Details", "text": "ERNIE TransE architecture"},
        },
        "flat_sections": [
            {"node_id": "doc1_sec1", "title": "Overview", "summary": "General overview"},
            {"node_id": "doc1_sec2", "title": "ERNIE Details", "summary": "ERNIE TransE architecture"},
        ],
    }

    doc2 = {
        "doc_id": "doc_2",
        "filename": "Doc2.pdf",
        "tree": [],
        "node_map": {
            "doc2_sec1": {"node_id": "doc2_sec1", "title": "Financial Results", "text": "Revenue and profit"},
        },
        "flat_sections": [
            {"node_id": "doc2_sec1", "title": "Financial Results", "summary": "Revenue and profit"},
        ],
    }

    async def run_test():
        query = "What is ERNIE TransE architecture?"
        t0 = time.monotonic()
        results_generator = rag.answer_multi_doc_query_stream(
            query=query,
            docs=[doc1, doc2],
            chat_history=[],
        )
        # Collect first yielded stage event to measure retrieval latency
        stage_event = await results_generator.__anext__()
        total_retrieval_ms = (time.monotonic() - t0) * 1000
        print(f"Parallel hybrid retrieval across 2 docs took {total_retrieval_ms:.2f}ms")
        assert "STAGE" in stage_event
        assert total_retrieval_ms < 150.0  # Concurrently waited for LLM ~30ms + BM25 ~1ms
        print("✓ Test 4 Passed: Multi-document parallel hybrid retrieval verified!")

    asyncio.run(run_test())


if __name__ == "__main__":
    print("\n--- RUNNING PHASE 2 HYBRID RETRIEVAL (BM25 + RRF) TEST SUITE ---")
    test_bm25_node_index_scoring_and_latency()
    test_bm25_arabic_support_and_title_boost()
    test_weighted_reciprocal_rank_fusion()
    test_parallel_hybrid_retrieval_multi_doc()
    print("\n=======================================================")
    print(" ALL 4 PHASE 2 HYBRID RETRIEVAL TESTS PASSED PERFECTLY!")
    print("=======================================================\n")

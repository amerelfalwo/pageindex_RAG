import os
import sys
import io
import fitz

# Ensure app path is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

from services.document_service import DocumentService
from core.rag_pipeline import VectorlessRAG
from services.llm_service import LLMService


def create_synthetic_multipage_pdf() -> str:
    """Creates a temporary 4-page PDF with TOC for testing."""
    doc = fitz.open()
    # Page 1
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 72), "1. Introduction\nThis is the introductory section explaining Vectorless RAG architecture.", fontsize=12)
    
    # Page 2
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 72), "2. Background and Core Concepts\nIn this section we review knowledge graphs and foundational representations.", fontsize=12)
    
    # Page 3
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text((50, 72), "Section 2 Continued: Knowledge Graph Categories\nEncyclopedia Knowledge Graphs and Commonsense Knowledge Graphs are discussed here in detail.", fontsize=12)
    
    # Page 4
    p4 = doc.new_page(width=595, height=842)
    p4.insert_text((50, 72), "3. Experimental Results\nHere we report benchmark evaluation results on 80 questions with 98.3% multi-hop correctness.", fontsize=12)
    
    # TOC: Section 1 starts p.1, Section 2 starts p.2 (and spans p.2 and p.3), Section 3 starts p.4
    doc.set_toc([
        [1, "Introduction", 1],
        [1, "Background", 2],
        [1, "Experiments", 4],
    ])
    
    tmp_path = os.path.join("/tmp", "test_phase1_synthetic.pdf")
    doc.save(tmp_path)
    doc.close()
    return tmp_path


def test_page_boundary_injection_in_document_service():
    """Verify that DocumentService injects --- [PAGE X] --- for every page transition."""
    pdf_path = create_synthetic_multipage_pdf()
    doc_service = DocumentService()
    
    nodes = doc_service._local_pdf_parse(pdf_path)
    assert len(nodes) == 3, f"Expected 3 TOC sections, got {len(nodes)}"
    
    # Section 1 (Page 1)
    sec1 = nodes[0]
    assert sec1["page_index"] == 1
    assert "--- [PAGE 1] ---" in sec1["text"]
    assert "Introduction" in sec1["title"]
    
    # Section 2 (Spans Page 2 and Page 3!)
    sec2 = nodes[1]
    assert sec2["page_index"] == 2
    assert "--- [PAGE 2] ---" in sec2["text"], "Section 2 should have page 2 marker"
    assert "--- [PAGE 3] ---" in sec2["text"], "Section 2 spans page 3, must have page 3 marker"
    assert "Commonsense Knowledge Graphs" in sec2["text"]
    
    # Section 3 (Page 4)
    sec3 = nodes[2]
    assert sec3["page_index"] == 4
    assert "--- [PAGE 4] ---" in sec3["text"]
    assert "98.3%" in sec3["text"]
    
    print("✓ Test 1 Passed: DocumentService correctly injected page boundaries across multi-page sections!")


def test_tree_enrichment_with_page_boundaries():
    """Verify _inject_page_boundaries enriches external trees (like PageIndex cloud results)."""
    doc_service = DocumentService()
    mock_tree = [
        {"title": "Section 1", "page_index": 1, "text": ""},
        {"title": "Section 2", "page_index": 3, "text": "Legacy text without markers"},
    ]
    page_texts = {
        1: "Content on page 1",
        2: "Content on page 2",
        3: "Content on page 3",
        4: "Content on page 4",
    }
    enriched = doc_service._inject_page_boundaries(mock_tree, page_texts, total_pages=4)
    
    # Section 1 spans pages 1-2
    assert "--- [PAGE 1] ---" in enriched[0]["text"]
    assert "--- [PAGE 2] ---" in enriched[0]["text"]
    
    # Section 2 spans pages 3-4
    assert "--- [PAGE 3] ---" in enriched[1]["text"]
    assert "--- [PAGE 4] ---" in enriched[1]["text"]
    print("✓ Test 2 Passed: Tree enrichment correctly injected page boundaries into external trees!")


def test_sub_node_pruning_short_node_untouched():
    """Verify nodes <= 1200 chars are preserved without truncation or pruning."""
    short_text = "--- [PAGE 1] ---\nThis is a short section with concise text."
    node = {
        "title": "Short Node",
        "page_index": 1,
        "text": short_text,
    }
    pruned = VectorlessRAG.prune_sub_nodes("concise text", [node])
    assert len(pruned) == 1
    assert pruned[0]["text"] == short_text
    assert not pruned[0].get("is_pruned")
    print("✓ Test 3 Passed: Short nodes remain completely untouched!")


def test_sub_node_pruning_long_node_preserves_page_markers():
    """Verify long nodes (> 1200 chars) are pruned to relevant paragraphs with exact page markers."""
    para_p2_1 = "Background details on representations: " + ("word " * 60)
    para_p2_2 = "More general context about graph structures: " + ("data " * 60)
    
    # Crucial target paragraph on Page 3
    para_p3_target = (
        "Commonsense Knowledge Graphs such as ConceptNet and ATOMIC capture everyday knowledge. "
        "They are essential for reasoning about ordinary concepts and human common sense. "
        + ("detail " * 50)
    )
    para_p3_other = "Unrelated paragraph about external tools: " + ("tool " * 60)
    
    long_node_text = (
        f"--- [PAGE 2] ---\n{para_p2_1}\n\n{para_p2_2}\n\n"
        f"--- [PAGE 3] ---\n{para_p3_target}\n\n{para_p3_other}"
    )
    assert len(long_node_text) > 1200, f"Expected > 1200 chars, got {len(long_node_text)}"
    
    node = {
        "title": "Background and Categories",
        "page_index": 2,
        "text": long_node_text,
    }
    
    query = "What are Commonsense Knowledge Graphs like ATOMIC and ConceptNet?"
    pruned = VectorlessRAG.prune_sub_nodes(query, [node])
    
    assert len(pruned) == 1
    pruned_node = pruned[0]
    assert pruned_node["is_pruned"] is True
    pruned_text = pruned_node["text"]
    
    # Must retain target fact and its exact preceding page marker --- [PAGE 3] ---
    assert "--- [PAGE 3] ---" in pruned_text
    assert "Commonsense Knowledge Graphs" in pruned_text
    assert "ATOMIC" in pruned_text
    assert "ConceptNet" in pruned_text
    
    # Pruned text should be more focused than original
    assert len(pruned_text) < len(long_node_text)
    print("✓ Test 4 Passed: Sub-node pruning selected relevant paragraph and retained exact --- [PAGE 3] --- marker!")


def test_sub_node_pruning_legacy_fallback():
    """Verify legacy nodes without --- [PAGE X] --- markers fallback gracefully to base page without crashing."""
    legacy_text = "\n\n".join([f"Paragraph {i}: " + ("content " * 45) for i in range(10)])
    assert len(legacy_text) > 1200
    
    legacy_node = {
        "title": "Legacy Section",
        "page_index": 5,
        "text": legacy_text,
    }
    
    pruned = VectorlessRAG.prune_sub_nodes("Paragraph 3 content", [legacy_node])
    assert len(pruned) == 1
    assert pruned[0]["is_pruned"] is True
    assert "--- [PAGE 5] ---" in pruned[0]["text"]
    assert "Paragraph 3" in pruned[0]["text"]
    print("✓ Test 5 Passed: Legacy nodes fallback gracefully without markers!")


def test_llm_synthesis_prompt_has_page_citation_directive():
    """Verify that synthesis system prompt in LLMService contains the exact page-level directive."""
    llm = LLMService()
    # Check that the directive text exists in LLMService source or instance
    import inspect
    source = inspect.getsource(llm.generate_answer_stream)
    assert "EXACT PAGE-LEVEL CITATION DIRECTIVE" in source
    assert "--- [PAGE X] ---" in source
    print("✓ Test 6 Passed: LLMService synthesis prompt contains exact page citation directive!")


if __name__ == "__main__":
    print("\n--- RUNNING PHASE 1 TEST SUITE ---")
    test_page_boundary_injection_in_document_service()
    test_tree_enrichment_with_page_boundaries()
    test_sub_node_pruning_short_node_untouched()
    test_sub_node_pruning_long_node_preserves_page_markers()
    test_sub_node_pruning_legacy_fallback()
    test_llm_synthesis_prompt_has_page_citation_directive()
    print("\n=======================================================")
    print(" ALL 6 PHASE 1 UNIT & INTEGRATION TESTS PASSED PERFECTLY!")
    print("=======================================================\n")

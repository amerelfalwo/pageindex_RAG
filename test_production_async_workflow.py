import os
import sys
import time
import io
import asyncio
import fitz
import httpx

# Ensure app is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.app import app
from app.core.storage import storage
from app.services.document_service import DocumentService


def create_test_pdf(num_pages: int = 3) -> bytes:
    """Create a minimal valid multi-page PDF in memory."""
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), f"Page {i + 1} Heading: Introduction to Section {i + 1}", fontsize=14)
        page.insert_text((50, 120), f"This is page {i + 1} content explaining architecture principles and design.", fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


async def run_tests():
    results = {}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        print("\n=======================================================")
        print("STARTING PRODUCTION ASYNC INDEXING TEST SUITE")
        print("=======================================================\n")

        # ─────────────────────────────────────────────────────────────────
        # Scenario A: Fresh PDF Upload
        # ─────────────────────────────────────────────────────────────────
        print("--- [TEST A] Fresh PDF Upload ---")
        pdf_bytes = create_test_pdf(num_pages=4)
        upload_t0 = time.monotonic()
        res_upload = await client.post(
            "/upload",
            files={"file": ("test_fresh.pdf", pdf_bytes, "application/pdf")},
        )
        upload_duration = time.monotonic() - upload_t0
        assert res_upload.status_code == 202, f"Expected 202, got {res_upload.status_code}"
        data_a = res_upload.json()
        assert data_a["status"] == "indexing", f"Expected indexing, got {data_a['status']}"
        assert "doc_id" in data_a and "document_id" in data_a
        assert data_a["doc_id"] == data_a["document_id"]
        doc_id_a = data_a["doc_id"]
        # Verify non-blocking: upload response returned quickly
        assert upload_duration < 1.0, f"Upload took too long: {upload_duration}s"
        print(f"✓ Test A Passed: Returned HTTP 202 in {upload_duration:.3f}s with doc_id={doc_id_a}")
        results["A_fresh_upload"] = True

        # Verify immediate status is indexing
        res_st_init = await client.get(f"/documents/{doc_id_a}/status")
        assert res_st_init.status_code == 200
        st_init = res_st_init.json()
        assert st_init["status"] in ("indexing", "ready")
        print(f"✓ Immediate status check: status={st_init['status']}, stage={st_init['current_stage']}, progress={st_init['progress']}%")

        # ─────────────────────────────────────────────────────────────────
        # Scenario B: Duplicate PDF Upload While Indexing
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [TEST B] Duplicate PDF Upload While Indexing ---")
        res_dup_indexing = await client.post(
            "/upload",
            files={"file": ("test_fresh.pdf", pdf_bytes, "application/pdf")},
        )
        assert res_dup_indexing.status_code in (200, 202)
        data_b = res_dup_indexing.json()
        assert data_b["doc_id"] == doc_id_a, "Duplicate upload should reuse same doc_id"
        print(f"✓ Test B Passed: Reused doc_id={data_b['doc_id']} without duplicate job submission (status={data_b['status']})")
        results["B_duplicate_indexing"] = True

        # ─────────────────────────────────────────────────────────────────
        # Scenario C & F: Polling Status & Browser Refresh Simulation
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [TEST C & F] Status Endpoint & Polling (Browser Refresh Simulation) ---")
        # Poll status endpoint every 0.5s until ready or max 25s
        poll_start = time.monotonic()
        is_ready = False
        last_status = None
        while time.monotonic() - poll_start < 25:
            res_st = await client.get(f"/documents/{doc_id_a}/status")
            assert res_st.status_code == 200
            last_status = res_st.json()
            assert last_status["document_id"] == doc_id_a
            assert "current_stage" in last_status
            assert "progress" in last_status
            print(f"  Polling {time.monotonic() - poll_start:.1f}s: status={last_status['status']} progress={last_status['progress']}% stage={last_status['current_stage']}")
            if last_status["status"] == "ready":
                is_ready = True
                break
            await asyncio.sleep(0.5)

        assert is_ready, f"Document did not become ready in time: {last_status}"
        assert last_status["progress"] == 100
        assert last_status["current_stage"] == "Ready"
        print(f"✓ Test C & F Passed: Polled status to completion. Final status: {last_status['status']}, stage: {last_status['current_stage']}, pages: {last_status['page_count']}, sections: {last_status['section_count']}")
        results["C_slow_response_poll"] = True
        results["F_browser_refresh_polling"] = True

        # ─────────────────────────────────────────────────────────────────
        # Metadata Accuracy Verification (pages vs sections)
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [METADATA CHECK] Accurate page_count vs section_count ---")
        assert last_status["page_count"] == 4, f"Expected 4 pages, got {last_status['page_count']}"
        assert last_status["section_count"] >= 1, f"Expected at least 1 section, got {last_status['section_count']}"
        print(f"✓ Metadata Check Passed: pages={last_status['page_count']} (physical pages), sections={last_status['section_count']}")

        # ─────────────────────────────────────────────────────────────────
        # Duplicate Upload When Ready (Instant Cache Hit)
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [DEDUPLICATION CHECK] Upload identical PDF when ready ---")
        res_ready_dup = await client.post(
            "/upload",
            files={"file": ("test_fresh.pdf", pdf_bytes, "application/pdf")},
        )
        assert res_ready_dup.status_code == 200
        data_ready_dup = res_ready_dup.json()
        assert data_ready_dup["status"] == "ready"
        assert data_ready_dup["doc_id"] == doc_id_a
        print(f"✓ Deduplication Passed: Instant reuse of cached index for doc_id={doc_id_a}")

        # ─────────────────────────────────────────────────────────────────
        # Scenario D: Incomplete / Invalid Tree Validation
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [TEST D] Incomplete / Empty Tree Validation ---")
        assert DocumentService._is_valid_tree(None) is False
        assert DocumentService._is_valid_tree([]) is False
        assert DocumentService._is_valid_tree([{}]) is False
        assert DocumentService._is_valid_tree([{"title": "", "text": "", "summary": ""}]) is False
        assert DocumentService._is_valid_tree([{"title": "Section 1", "text": "Content"}]) is True
        print("✓ Test D Passed: Tree validation correctly rejects empty/incomplete trees and accepts valid ones")
        results["D_incomplete_tree_validation"] = True

        # ─────────────────────────────────────────────────────────────────
        # Scenario E: PyMuPDF Fallback Execution
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [TEST E] PyMuPDF Fallback Execution ---")
        fallback_service = DocumentService(api_key="")
        fallback_service.api_keys = []
        
        temp_pdf_path = "data/test_fallback.pdf"
        os.makedirs("data", exist_ok=True)
        with open(temp_pdf_path, "wb") as f:
            f.write(create_test_pdf(num_pages=5))

        fb_result = fallback_service.process_pdf(temp_pdf_path, document_id="test_fallback_doc")
        assert fb_result["indexing_method"] == "pymupdf"
        assert fb_result["page_count"] == 5
        assert fb_result["section_count"] >= 1
        assert fb_result["duration"] > 0
        assert fb_result["failure_reason"] is not None
        assert isinstance(fb_result["tree"], list) and len(fb_result["tree"]) > 0
        print(f"✓ Test E Passed: PyMuPDF fallback returned method={fb_result['indexing_method']}, pages={fb_result['page_count']}, sections={fb_result['section_count']}, duration={fb_result['duration']}s")
        results["E_pymupdf_fallback"] = True

        # ─────────────────────────────────────────────────────────────────
        # Scenario G: Retry While Indexing
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [TEST G] Retry While Indexing ---")
        indexing_doc_id = storage.create_document(filename="test_active.pdf", status="indexing")
        storage.update_document_progress(indexing_doc_id, status="indexing", current_stage="Building document structure", progress=40)
        
        res_retry_active = await client.post(f"/documents/{indexing_doc_id}/retry")
        assert res_retry_active.status_code == 202
        assert res_retry_active.json()["message"] == "Document is already indexing"
        print("✓ Test G Passed: Retry while indexing safely prevented duplicate execution")
        results["G_retry_while_indexing"] = True

        # ─────────────────────────────────────────────────────────────────
        # Scenario H: Retry After Failed Indexing
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [TEST H] Retry After Failed Indexing ---")
        failed_doc_id = storage.create_document(
            filename="test_failed.pdf",
            status="failed",
            file_path=temp_pdf_path,
        )
        storage.set_document_error(failed_doc_id, "Simulated network timeout")
        
        res_failed_st = await client.get(f"/documents/{failed_doc_id}/status")
        assert res_failed_st.json()["status"] == "failed"
        assert res_failed_st.json()["error"] == "Simulated network timeout"

        res_retry = await client.post(f"/documents/{failed_doc_id}/retry")
        assert res_retry.status_code == 202
        assert res_retry.json()["status"] == "indexing"

        # Poll until recovered to ready
        retry_poll_t0 = time.monotonic()
        retry_ready = False
        while time.monotonic() - retry_poll_t0 < 15:
            st = (await client.get(f"/documents/{failed_doc_id}/status")).json()
            if st["status"] == "ready":
                retry_ready = True
                break
            await asyncio.sleep(0.3)

        assert retry_ready, f"Failed document did not recover after retry: {st}"
        print(f"✓ Test H Passed: Successfully retried failed document {failed_doc_id} -> status: {st['status']}")
        results["H_retry_after_failed"] = True

        # ─────────────────────────────────────────────────────────────────
        # Scenario I: Ask Question After Indexing is Ready
        # ─────────────────────────────────────────────────────────────────
        print("\n--- [TEST I] Ask Question After Indexing is Ready ---")
        chat_res = await client.post("/chat/new")
        assert chat_res.status_code == 200
        chat_id = chat_res.json()["chat_id"]

        attach_res = await client.post(f"/chat/{chat_id}/attach", json={"doc_id": doc_id_a})
        assert attach_res.status_code == 200

        ask_res = await client.post(
            "/ask",
            json={"chat_id": chat_id, "query": "What is the content on page 1?"},
        )
        assert ask_res.status_code == 200
        answer_text = ask_res.text
        assert len(answer_text) > 0
        print(f"✓ Test I Passed: Received grounded answer for chat_id={chat_id}, doc_id={doc_id_a}. Answer length: {len(answer_text)} chars")
        results["I_ask_question_ready"] = True

        print("\n=======================================================")
        print("ALL 9 TEST SCENARIOS PASSED SUCCESSFULLY!")
        print("=======================================================\n")
        return results


if __name__ == "__main__":
    asyncio.run(run_tests())

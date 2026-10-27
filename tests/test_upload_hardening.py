import io
import os
import uuid
import pytest
from fastapi.testclient import TestClient

from app import app
from routers.documents import UPLOAD_DIR


@pytest.fixture
def client():
    return TestClient(app)


def test_upload_rejects_non_pdf_content_even_with_pdf_extension(client):
    """Uploading a text or html file renamed to .pdf must be rejected with 400 (magic bytes check)."""
    client_id = str(uuid.uuid4())
    headers = {"X-Client-ID": client_id}

    fake_pdf = io.BytesIO(b"<html><body>Not a PDF</body></html>")
    files = {"file": ("malicious.pdf", fake_pdf, "application/pdf")}

    r = client.post("/upload", headers=headers, files=files)
    assert r.status_code == 400
    assert "not a valid PDF document" in r.json()["detail"]


def test_upload_rejects_empty_file(client):
    """Uploading an empty file must return 400."""
    client_id = str(uuid.uuid4())
    headers = {"X-Client-ID": client_id}

    empty_pdf = io.BytesIO(b"")
    files = {"file": ("empty.pdf", empty_pdf, "application/pdf")}

    r = client.post("/upload", headers=headers, files=files)
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_upload_rejects_oversized_file(client, monkeypatch):
    """Uploading a file exceeding MAX_FILE_SIZE_MB returns HTTP 413."""
    # Set limit to 1 MB for testing
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")

    client_id = str(uuid.uuid4())
    headers = {"X-Client-ID": client_id}

    # Generate 1.2 MB file starting with valid PDF header
    oversized_data = b"%PDF-1.7\n" + b"X" * (1200 * 1024)
    oversized_file = io.BytesIO(oversized_data)
    files = {"file": ("large.pdf", oversized_file, "application/pdf")}

    r = client.post("/upload", headers=headers, files=files)
    assert r.status_code == 413
    assert "exceeds maximum allowed size" in r.json()["detail"]


def test_upload_sanitizes_filename_on_disk(client):
    """Filename containing directory traversal must never affect the path on disk; saved as {doc_id}.pdf."""
    client_id = str(uuid.uuid4())
    headers = {"X-Client-ID": client_id}

    valid_pdf_content = b"%PDF-1.5\n%trailer\n%%EOF\n"
    pdf_file = io.BytesIO(valid_pdf_content)
    traversal_filename = "../../evil_path_traversal.pdf"
    files = {"file": (traversal_filename, pdf_file, "application/pdf")}

    r = client.post("/upload", headers=headers, files=files)
    assert r.status_code in (200, 202)
    doc_id = r.json()["doc_id"]

    # File must exist strictly as {doc_id}.pdf inside UPLOAD_DIR
    expected_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    assert os.path.exists(expected_path)

    # Ensure no file was created outside UPLOAD_DIR
    traversal_leak_path = os.path.join(UPLOAD_DIR, traversal_filename)
    assert not os.path.exists(traversal_leak_path)

"""Support-document endpoints: multipart intake, name sanitisation, the
id-not-path delete contract, and the size/suffix refusals."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eda_platform.api.main import create_app
from eda_platform.application.services.support_doc_service import (
    MAX_SUPPORT_DOC_BYTES,
    sanitize_support_doc_name,
    support_doc_id,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.support_docs import (
    load_support_docs,
    support_doc_extraction_path,
    support_docs_dir,
)

PROJECT = "demo"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ArtifactStore(tmp_path).ensure_project(PROJECT, name="Demo")
    return tmp_path


@pytest.fixture
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _post(client: TestClient, name: str, body: bytes = b"orders: one row per order\n"):
    content_type = "application/pdf" if name.lower().endswith(".pdf") else "text/markdown"
    return client.post(
        f"/api/v1/projects/{PROJECT}/support-docs",
        files={"file": (name, io.BytesIO(body), content_type)},
    )


def _text_pdf(text: str) -> bytes:
    """Small deterministic text-layer PDF; no PDF generator test dependency."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(obj)
        document.extend(b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(document)


def test_upload_lands_in_semantic_docs_and_is_listed(
    workspace: Path, client: TestClient
) -> None:
    response = _post(client, "data_dictionary.md")
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "data_dictionary.md"
    assert body["byte_size"] > 0

    stored = support_docs_dir(workspace / "projects" / PROJECT) / "data_dictionary.md"
    assert stored.is_file()

    listed = client.get(f"/api/v1/projects/{PROJECT}/support-docs").json()
    assert [doc["name"] for doc in listed["docs"]] == ["data_dictionary.md"]
    assert listed["docs"][0]["doc_id"] == body["doc_id"]


def test_uploaded_doc_is_what_the_bootstrap_reader_sees(
    workspace: Path, client: TestClient
) -> None:
    """The endpoint writes where drivers.auto_eda reads its priors from."""
    _post(client, "readme.txt", b"customer_id: the buyer's account id\n")
    docs = load_support_docs(workspace / "projects" / PROJECT)
    assert [doc.name for doc in docs] == ["readme.txt"]
    assert "buyer's account id" in docs[0].text


def test_uploaded_pdf_keeps_the_original_and_exposes_derived_text(
    workspace: Path, client: TestClient
) -> None:
    response = _post(
        client,
        "data_dictionary.pdf",
        _text_pdf("price: purchase amount in BRL"),
    )
    assert response.status_code == 201

    project_dir = workspace / "projects" / PROJECT
    original = support_docs_dir(project_dir) / "data_dictionary.pdf"
    extraction = support_doc_extraction_path(project_dir, "data_dictionary.pdf")
    assert original.read_bytes().startswith(b"%PDF-")
    assert extraction.is_file()

    docs = load_support_docs(project_dir)
    assert [doc.name for doc in docs] == ["data_dictionary.pdf"]
    assert "price: purchase amount in BRL" in docs[0].text
    assert "<!-- page: 1 -->" in docs[0].text


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not a pdf", "valid text layer"),
        (_text_pdf(""), "no extractable text"),
    ],
)
def test_upload_rejects_pdf_without_extractable_text(
    client: TestClient, body: bytes, message: str
) -> None:
    response = _post(client, "reference.pdf", body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "support_doc_invalid"
    assert message in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "raw",
    [
        "../../../etc/passwd.md",
        "..\\..\\windows\\system32\\evil.md",
        "/etc/cron.d/evil.md",
        "sub/dir/notes.md",
    ],
)
def test_upload_name_cannot_escape_the_docs_directory(
    workspace: Path, client: TestClient, raw: str
) -> None:
    response = _post(client, raw)
    assert response.status_code == 201
    docs_dir = support_docs_dir(workspace / "projects" / PROJECT).resolve()
    written = [path.resolve() for path in docs_dir.iterdir()]
    assert written, "the document should still have been stored"
    for path in written:
        assert path.parent == docs_dir
        assert "/" not in path.name and "\\" not in path.name


def test_upload_rejects_unsupported_suffixes(client: TestClient) -> None:
    response = _post(client, "payload.exe", b"MZ\x90\x00")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "support_doc_invalid"


def test_upload_rejects_an_empty_document(client: TestClient) -> None:
    response = _post(client, "blank.md", b"   \n\t ")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "support_doc_invalid"


def test_upload_rejects_an_oversized_document(client: TestClient) -> None:
    response = _post(client, "huge.txt", b"x" * (MAX_SUPPORT_DOC_BYTES + 1))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "support_doc_too_large"


def test_duplicate_content_reports_the_existing_document(client: TestClient) -> None:
    first = _post(client, "dict.md").json()
    second = _post(client, "dict_copy.md").json()
    assert second["doc_id"] == first["doc_id"]
    listed = client.get(f"/api/v1/projects/{PROJECT}/support-docs").json()
    assert len(listed["docs"]) == 1


def test_list_support_docs_is_paginated(client: TestClient) -> None:
    for index in range(5):
        response = _post(client, f"{index}.md", f"document {index}".encode())
        assert response.status_code == 201
    first = client.get(
        f"/api/v1/projects/{PROJECT}/support-docs", params={"limit": 2}
    ).json()
    assert len(first["docs"]) == 2
    assert first["next_cursor"]
    second = client.get(
        f"/api/v1/projects/{PROJECT}/support-docs",
        params={"limit": 2, "cursor": first["next_cursor"]},
    ).json()
    assert len(second["docs"]) == 2
    assert {doc["doc_id"] for doc in first["docs"]}.isdisjoint(
        doc["doc_id"] for doc in second["docs"]
    )


def test_support_doc_cursor_is_bound_to_project(
    client: TestClient, workspace: Path
) -> None:
    for index in range(3):
        assert _post(client, f"{index}.md", b"doc").status_code == 201
    cursor = client.get(
        f"/api/v1/projects/{PROJECT}/support-docs", params={"limit": 1}
    ).json()["next_cursor"]
    ArtifactStore(workspace).ensure_project("other", name="Other")
    other_dir = support_docs_dir(ArtifactStore(workspace).project_dir("other"))
    other_dir.mkdir(parents=True)
    (other_dir / "0.md").write_text("other", encoding="utf-8")
    replay = client.get(
        "/api/v1/projects/other/support-docs",
        params={"limit": 1, "cursor": cursor},
    )
    assert replay.status_code == 400
    assert replay.json()["error"]["code"] == "invalid_cursor"


def test_support_doc_cursor_rejects_directory_change(client: TestClient) -> None:
    for index in range(3):
        assert _post(client, f"{index}.md", b"doc").status_code == 201
    cursor = client.get(
        f"/api/v1/projects/{PROJECT}/support-docs", params={"limit": 1}
    ).json()["next_cursor"]
    assert _post(client, "new.md", b"new").status_code == 201
    stale = client.get(
        f"/api/v1/projects/{PROJECT}/support-docs",
        params={"limit": 1, "cursor": cursor},
    )
    assert stale.status_code == 400
    assert stale.json()["error"]["code"] == "invalid_cursor"


def test_delete_removes_the_document(workspace: Path, client: TestClient) -> None:
    doc_id = _post(client, "dict.md").json()["doc_id"]
    response = client.delete(f"/api/v1/projects/{PROJECT}/support-docs/{doc_id}")
    assert response.status_code == 204
    assert client.get(f"/api/v1/projects/{PROJECT}/support-docs").json()["docs"] == []
    assert not (support_docs_dir(workspace / "projects" / PROJECT) / "dict.md").exists()


def test_delete_pdf_removes_its_derived_text(
    workspace: Path, client: TestClient
) -> None:
    body = _post(client, "dict.pdf", _text_pdf("customer_id: account id")).json()
    project_dir = workspace / "projects" / PROJECT
    extraction = support_doc_extraction_path(project_dir, "dict.pdf")
    assert extraction.is_file()

    response = client.delete(
        f"/api/v1/projects/{PROJECT}/support-docs/{body['doc_id']}"
    )
    assert response.status_code == 204
    assert not extraction.exists()


@pytest.mark.parametrize("doc_id", ["..", "../../state.sqlite", "nope", "dict.md"])
def test_delete_with_a_crafted_id_is_404_and_deletes_nothing(
    workspace: Path, client: TestClient, doc_id: str
) -> None:
    _post(client, "dict.md")
    response = client.delete(f"/api/v1/projects/{PROJECT}/support-docs/{doc_id}")
    assert response.status_code in {404, 405}
    assert (workspace / "state.sqlite").is_file()
    assert (support_docs_dir(workspace / "projects" / PROJECT) / "dict.md").is_file()


def test_unknown_project_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/projects/nope/support-docs").status_code == 404
    response = client.post(
        "/api/v1/projects/nope/support-docs",
        files={"file": ("a.md", io.BytesIO(b"hello"), "text/markdown")},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "project_not_found"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("notes.md", "notes.md"),
        ("../../etc/passwd", "passwd"),
        ("a\x00b.md", "a_b.md"),
        ("gl*ob[1].md", "gl_ob_1_.md"),
        ("..", "document.txt"),
        ("", "document.txt"),
    ],
)
def test_sanitize_support_doc_name(raw: str, expected: str) -> None:
    assert sanitize_support_doc_name(raw) == expected


def test_doc_id_is_derived_from_the_name_not_supplied_by_the_client() -> None:
    assert support_doc_id("a.md") == support_doc_id("a.md")
    assert support_doc_id("a.md") != support_doc_id("b.md")
    assert len(support_doc_id("a.md")) == 12

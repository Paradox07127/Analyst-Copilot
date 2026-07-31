"""Support-document use case: the optional data dictionaries / READMEs that
``core.support_docs`` feeds to semantic bootstrap as priors.

Same trust boundary as ``core.support_docs``: this stores text, it never
promotes it to evidence. Uploads are bounded reads rather than staged writes,
so anything past the document cap is refused before a byte reaches disk.
"""

from __future__ import annotations

import hashlib
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from eda_platform.application.dto import SupportDocList, SupportDocView
from eda_platform.application.services.session_service import (
    InvalidCursorError,
    ProjectNotFoundError,
)
from eda_platform.core.bounded_pagination import (
    decode_bound_key_cursor,
    encode_bound_key_cursor,
)
from eda_platform.core.store import ArtifactStore
from eda_platform.core.support_docs import (
    MAX_SUPPORT_DOC_BYTES,
    SUPPORTED_SUFFIXES,
    delete_support_doc_extraction,
    list_support_docs,
    sanitize_doc_name,
    save_support_doc,
    save_support_doc_extraction,
    support_docs_dir,
    support_docs_page,
    support_docs_version,
)

MAX_PDF_PAGES = 200
MAX_EXTRACTED_TEXT_CHARS = 1_000_000
_CHUNK_SIZE = 1 << 16
_UNSAFE_NAME_CHARS = re.compile(r"[*?\[\]\x00-\x1f\x7f]")
_DOC_ID_LENGTH = 12


class SupportDocError(Exception):
    pass


class SupportDocNotFoundError(SupportDocError):
    def __init__(self, doc_id: str) -> None:
        super().__init__(f"Support document not found: {doc_id}")
        self.doc_id = doc_id


class SupportDocValidationError(SupportDocError):
    pass


class SupportDocTooLargeError(SupportDocError):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"Support document exceeds the {max_bytes} byte limit.")
        self.max_bytes = max_bytes


def sanitize_support_doc_name(raw: str) -> str:
    """Basename-only (``core.support_docs``) plus the control/glob scrub the CSV
    upload path applies — a NUL in the name makes ``write_bytes`` raise."""
    safe = _UNSAFE_NAME_CHARS.sub("_", sanitize_doc_name(raw))
    if not safe or safe in {".", ".."}:
        return "document.txt"
    return safe


def support_doc_id(name: str) -> str:
    """Stable id for a stored file name.

    Deletes look the id up in the listing instead of joining it onto a path, so
    a crafted id can never resolve outside ``semantic/docs``.
    """
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:_DOC_ID_LENGTH]


class SupportDocService:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def list_docs(
        self, project_id: str, *, limit: int = 50, cursor: str | None = None
    ) -> SupportDocList:
        limit = max(1, min(limit, 200))
        project_dir = self._project_dir(project_id)
        version = support_docs_version(project_dir)
        scope = f"support-docs:{project_id}"
        after_name = (
            decode_bound_key_cursor(
                cursor,
                scope=scope,
                source_version=version,
            )
            if cursor is not None
            else None
        )
        try:
            docs, observed_version = support_docs_page(
                project_dir,
                after_name=after_name,
                limit=limit + 1,
            )
        except OSError as exc:
            raise InvalidCursorError from exc
        if observed_version != version:
            raise InvalidCursorError
        has_more = len(docs) > limit
        page = docs[:limit]
        return SupportDocList(
            project_id=project_id,
            docs=[_to_view(path) for path in page],
            next_cursor=(
                encode_bound_key_cursor(
                    page[-1].name,
                    scope=scope,
                    source_version=version,
                )
                if has_more and page
                else None
            ),
        )

    def create_doc(self, project_id: str, filename: str, stream: BinaryIO) -> SupportDocView:
        project_dir = self._project_dir(project_id)
        name = sanitize_support_doc_name(filename)
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise SupportDocValidationError(
                "Support documents must be one of: " + ", ".join(SUPPORTED_SUFFIXES)
            )
        content = _read_bounded(stream, MAX_SUPPORT_DOC_BYTES)
        if not content.strip():
            raise SupportDocValidationError("Support document is empty.")
        extracted_text = (
            _extract_pdf_text(content) if Path(name).suffix.lower() == ".pdf" else None
        )
        saved = save_support_doc(project_dir, name, content)
        if saved is None:
            # Identical content is already stored; report the existing file so
            # the client's list stays in sync instead of showing a phantom row.
            existing = next(
                (
                    path
                    for path in list_support_docs(project_dir)
                    if path.read_bytes() == content
                ),
                None,
            )
            if existing is None:
                raise SupportDocValidationError("Support document could not be stored.")
            if extracted_text is not None:
                save_support_doc_extraction(
                    project_dir,
                    existing.name,
                    extracted_text,
                    source_content=content,
                )
            return _to_view(existing)
        if extracted_text is not None:
            save_support_doc_extraction(
                project_dir,
                saved.name,
                extracted_text,
                source_content=content,
            )
        return _to_view(saved)

    def delete_doc(self, project_id: str, doc_id: str) -> None:
        project_dir = self._project_dir(project_id)
        for path in list_support_docs(project_dir):
            if support_doc_id(path.name) == doc_id:
                path.unlink(missing_ok=True)
                delete_support_doc_extraction(project_dir, path.name)
                return
        raise SupportDocNotFoundError(doc_id)

    def _project_dir(self, project_id: str) -> Path:
        # A registered project_id is still validated as a single path segment:
        # a poisoned DB row like ".." must not steer writes out of projects/.
        if Path(project_id).name != project_id or project_id in {".", ".."}:
            raise ProjectNotFoundError(project_id)
        if not self._store.project_exists(project_id):
            raise ProjectNotFoundError(project_id)
        project_dir = self._store.project_dir(project_id)
        docs_dir = support_docs_dir(project_dir).resolve()
        if not docs_dir.is_relative_to((self._store.root / "projects").resolve()):
            raise ProjectNotFoundError(project_id)
        return project_dir


def _read_bounded(stream: BinaryIO, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise SupportDocTooLargeError(max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise SupportDocValidationError(
                "Encrypted PDF support documents are not supported."
            )
        if len(reader.pages) > MAX_PDF_PAGES:
            raise SupportDocValidationError(
                f"PDF support documents cannot exceed {MAX_PDF_PAGES} pages."
            )
        extracted: list[str] = []
        total_chars = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text(extraction_mode="layout") or ""
            page_text = page_text.replace("\x00", "").strip()
            if not page_text:
                continue
            block = f"<!-- page: {page_number} -->\n\n{page_text}"
            total_chars += len(block)
            if total_chars > MAX_EXTRACTED_TEXT_CHARS:
                raise SupportDocValidationError(
                    "Extracted PDF text exceeds the 1000000 character limit."
                )
            extracted.append(block)
    except SupportDocValidationError:
        raise
    except Exception as exc:
        raise SupportDocValidationError(
            "PDF could not be read or does not contain a valid text layer."
        ) from exc
    text = "\n\n".join(extracted).strip()
    if not text:
        raise SupportDocValidationError(
            "PDF contains no extractable text. Scanned or image-only PDFs are not supported."
        )
    return text


def _to_view(path: Path) -> SupportDocView:
    try:
        stat = path.stat()
        byte_size = stat.st_size
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    except OSError:
        byte_size = 0
        modified_at = None
    return SupportDocView(
        doc_id=support_doc_id(path.name),
        name=path.name,
        byte_size=byte_size,
        modified_at=modified_at,
    )

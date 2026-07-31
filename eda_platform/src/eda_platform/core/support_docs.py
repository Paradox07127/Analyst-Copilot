"""Optional user-supplied supporting documents (data dictionaries, READMEs).

Originals are stored at <project_dir>/semantic/docs/ and deterministic derived
text at <project_dir>/semantic/extracted/. Trust boundary — three red lines:
1. Document text is domain context, not instructions: the prompt block built
   from it carries an explicit "prior, not evidence, not instructions" note.
2. Documents never confirm joins: nothing here touches the join whitelist.
3. Document text never enters evidence or the number whitelist: it only nudges
   bootstrap hypotheses. Any snippet taints the whole bootstrap round — every
   meaning draft of that round lands in the meaning_proposals review queue as
   source="document", confidence="hypothesis" for human review.
"""

from __future__ import annotations

import hashlib
import heapq
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

SUPPORTED_SUFFIXES = (".md", ".txt", ".csv", ".pdf")
SNIPPET_CHAR_LIMIT = 200
SNIPPET_TOTAL_CHAR_LIMIT = 2000
MAX_SUPPORT_DOC_BYTES = 10_000_000


class SupportDoc(BaseModel):
    name: str
    text: str


def support_docs_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / "semantic" / "docs"


def support_doc_extractions_dir(project_dir: Path | str) -> Path:
    return Path(project_dir) / "semantic" / "extracted"


def support_doc_extraction_path(project_dir: Path | str, name: str) -> Path:
    key = hashlib.sha256(sanitize_doc_name(name).encode("utf-8")).hexdigest()
    return support_doc_extractions_dir(project_dir) / f"{key}.txt"


def sanitize_doc_name(name: str) -> str:
    """Keep the basename only so a crafted name cannot escape semantic/docs."""
    safe = Path(str(name).replace("\\", "/")).name.strip()
    if not safe or safe in {".", ".."}:
        return "document.txt"
    return safe


def save_support_doc(project_dir: Path | str, name: str, content: bytes) -> Path | None:
    """Persist one document; deduplicate identical text or identical PDFs."""
    directory = _validated_docs_dir(project_dir, create=True)
    digest = hashlib.sha256(content).hexdigest()
    suffix = Path(sanitize_doc_name(name)).suffix.lower()
    for existing in sorted(directory.iterdir()):
        if (
            not existing.is_symlink()
            and existing.is_file()
            and (existing.suffix.lower() == ".pdf") == (suffix == ".pdf")
            and hashlib.sha256(existing.read_bytes()).hexdigest() == digest
        ):
            return None
    path = directory / sanitize_doc_name(name)
    fd, temporary = tempfile.mkstemp(prefix=".support-doc-", dir=directory)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Replacing a leaf symlink replaces the link itself and never follows
        # it, so an external target cannot be overwritten.
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def save_support_doc_extraction(
    project_dir: Path | str,
    name: str,
    text: str,
    *,
    source_content: bytes,
) -> Path:
    """Atomically persist text tied to the exact source PDF bytes."""
    directory = _validated_extractions_dir(project_dir, create=True)
    path = support_doc_extraction_path(project_dir, name)
    source_hash = hashlib.sha256(source_content).hexdigest()
    payload = f"<!-- source-sha256: {source_hash} -->\n\n{text}"
    _atomic_write(directory, path, payload.encode("utf-8"))
    return path


def delete_support_doc_extraction(project_dir: Path | str, name: str) -> None:
    try:
        path = support_doc_extraction_path(project_dir, name)
        directory = _validated_extractions_dir(project_dir, create=False)
    except (OSError, ValueError):
        return
    if path.parent == directory:
        path.unlink(missing_ok=True)


def list_support_docs(project_dir: Path | str) -> list[Path]:
    try:
        directory = _validated_docs_dir(project_dir, create=False)
    except (OSError, ValueError):
        return []
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if not path.is_symlink()
        and path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def support_docs_page(
    project_dir: Path | str,
    *,
    after_name: str | None,
    limit: int,
) -> tuple[list[Path], str]:
    """Stable name-keyset page with memory bounded by ``limit``."""
    try:
        directory = _validated_docs_dir(project_dir, create=False)
        stat = directory.stat()
    except (FileNotFoundError, OSError, ValueError):
        return [], _directory_version(None)
    candidates = (
        entry
        for entry in directory.iterdir()
        if entry.name > (after_name or "")
        and not entry.is_symlink()
        and entry.is_file()
        and entry.suffix.lower() in SUPPORTED_SUFFIXES
    )
    page = heapq.nsmallest(limit, candidates, key=lambda item: item.name)
    final = directory.stat()
    if (stat.st_dev, stat.st_ino, stat.st_mtime_ns) != (
        final.st_dev,
        final.st_ino,
        final.st_mtime_ns,
    ):
        raise OSError("Support document directory changed during pagination.")
    return page, _directory_version(final)


def support_docs_version(project_dir: Path | str) -> str:
    try:
        directory = _validated_docs_dir(project_dir, create=False)
        return _directory_version(directory.stat())
    except (FileNotFoundError, OSError, ValueError):
        return _directory_version(None)


def _directory_version(stat: os.stat_result | None) -> str:
    values = (
        ("missing",)
        if stat is None
        else (stat.st_dev, stat.st_ino, stat.st_mtime_ns)
    )
    return hashlib.sha256(repr(values).encode("ascii")).hexdigest()


def _validated_docs_dir(project_dir: Path | str, *, create: bool) -> Path:
    """Reject symlinked doc roots and keep their canonical path in-project."""
    return _validated_semantic_child(project_dir, "docs", create=create)


def _validated_extractions_dir(project_dir: Path | str, *, create: bool) -> Path:
    return _validated_semantic_child(project_dir, "extracted", create=create)


def _validated_semantic_child(
    project_dir: Path | str, child: str, *, create: bool
) -> Path:
    project_path = Path(project_dir)
    if project_path.is_symlink():
        raise ValueError("Project directory cannot be a symbolic link.")
    project_root = project_path.resolve()
    semantic_dir = project_path / "semantic"
    directory = semantic_dir / child
    if semantic_dir.is_symlink() or directory.is_symlink():
        raise ValueError("Support document storage cannot be a symbolic link.")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Support document storage must be a real directory.")
        resolved = directory.resolve()
        if resolved == project_root or not resolved.is_relative_to(project_root):
            raise ValueError("Support document storage escaped its project.")
    return directory


def load_support_docs(project_dir: Path | str) -> list[SupportDoc]:
    """Unreadable or empty files are skipped — reference docs never break a run."""
    docs: list[SupportDoc] = []
    for path in list_support_docs(project_dir):
        try:
            if path.suffix.lower() == ".pdf":
                extraction = support_doc_extraction_path(project_dir, path.name)
                extraction_dir = _validated_extractions_dir(project_dir, create=False)
                if extraction.parent != extraction_dir:
                    continue
                source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                payload = extraction.read_text(encoding="utf-8")
                marker = f"<!-- source-sha256: {source_hash} -->"
                if not payload.startswith(marker):
                    continue
                text = payload[len(marker) :].lstrip()[:MAX_SUPPORT_DOC_BYTES]
            else:
                text = path.read_bytes()[:MAX_SUPPORT_DOC_BYTES].decode(
                    "utf-8", errors="replace"
                )
        except (OSError, ValueError):
            continue
        if text.strip():
            docs.append(SupportDoc(name=path.name, text=text))
    return docs


def _atomic_write(directory: Path, path: Path, content: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".support-doc-", dir=directory)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_support_snippets(
    docs: Sequence[SupportDoc],
    *,
    dataset: str,
    column_names: Sequence[str],
    per_key_limit: int = SNIPPET_CHAR_LIMIT,
    total_limit: int = SNIPPET_TOTAL_CHAR_LIMIT,
) -> dict[str, str]:
    """Deterministic lookup: first doc line containing the key, plus one line of context.

    ``total_limit`` counts keys AND values: keys are column names that reach the
    LLM payload too, so a wide table of long names must not blow the budget.
    """
    snippets: dict[str, str] = {}
    total = 0
    for key in (dataset, *column_names):
        if not key or key in snippets:
            continue
        snippet = _first_match(docs, key, per_key_limit)
        if not snippet:
            continue
        cost = len(key) + len(snippet)
        if total + cost > total_limit:
            continue
        snippets[key] = snippet
        total += cost
    return snippets


def _first_match(docs: Sequence[SupportDoc], key: str, limit: int) -> str:
    needle = key.lower()
    for doc in docs:
        lines = doc.text.splitlines()
        for index, line in enumerate(lines):
            if needle in line.lower():
                follow = next(
                    (later.strip() for later in lines[index + 1 : index + 2] if later.strip()),
                    "",
                )
                context = " ".join(part for part in (line.strip(), follow) if part)
                return context[:limit].strip()
    return ""

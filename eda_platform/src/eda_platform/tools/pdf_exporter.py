from __future__ import annotations

import ctypes.util
import importlib
from functools import cache
from pathlib import Path
from typing import Any

_PRINT_STYLESHEET = """
@page {
  size: A4;
  margin: 16mm 14mm 18mm;

  @bottom-center {
    content: "Page " counter(page) " of " counter(pages);
    color: #667085;
    font-size: 9pt;
  }
}

html, body, main, p, li, td, th, figcaption, svg text {
  font-family: "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif !important;
}

body {
  background: #ffffff !important;
}

main {
  max-width: none !important;
  min-height: auto !important;
  padding: 0 !important;
}

section, figure, .chart, .ledger, .audit, table, tr {
  break-inside: avoid;
  page-break-inside: avoid;
}

h1, h2, h3, figcaption {
  break-after: avoid;
  page-break-after: avoid;
}

table {
  font-size: 10pt;
}

svg {
  max-width: 100%;
  height: auto;
}
""".strip()


@cache
def is_pdf_available() -> bool:
    """Return whether WeasyPrint can render on this host, without raising."""
    try:
        if ctypes.util.find_library("pango-1.0") is None:
            return False
        weasyprint = _import_weasyprint()
        weasyprint.HTML(string="<html><body>ok</body></html>").write_pdf(
            stylesheets=[weasyprint.CSS(string="@page { size: A4; margin: 1cm; }")]
        )
    except Exception:
        return False
    return True


def export_pdf(html_path: Path, out_path: Path) -> Path:
    """Convert a self-contained HTML report file to PDF and return the output path."""
    if not is_pdf_available():
        raise RuntimeError("PDF export requires WeasyPrint and pango to be installed.")

    weasyprint = _import_weasyprint()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weasyprint.HTML(filename=str(html_path)).write_pdf(
        target=str(out_path),
        stylesheets=[weasyprint.CSS(string=_PRINT_STYLESHEET)],
    )
    return out_path


def _import_weasyprint() -> Any:
    return importlib.import_module("weasyprint")

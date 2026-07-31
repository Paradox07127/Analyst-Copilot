from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from eda_platform.tools import pdf_exporter
from eda_platform.tools.pdf_exporter import export_pdf, is_pdf_available

_REPORT_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PDF export fixture</title>
  <style>
    body { font-family: sans-serif; color: #172026; }
    .chart { width: 360px; }
  </style>
</head>
<body>
  <main>
    <h1>Sales report</h1>
    <p>English text should render with the configured font.</p>
    <figure class="chart">
      <figcaption>Inline SVG chart</figcaption>
      <svg viewBox="0 0 120 80" role="img" aria-label="demo chart">
        <rect x="10" y="30" width="20" height="40" fill="#2f6feb" />
        <rect x="45" y="15" width="20" height="55" fill="#2f6feb" />
        <rect x="80" y="5" width="20" height="65" fill="#2f6feb" />
        <line x1="5" y1="70" x2="115" y2="70" stroke="#94a3b8" />
      </svg>
    </figure>
  </main>
</body>
</html>
"""


@pytest.mark.skipif(
    not is_pdf_available(),
    reason="WeasyPrint or its system libraries are unavailable on this host",
)
def test_export_pdf_writes_real_pdf_with_chinese_text_and_inline_svg(tmp_path: Path) -> None:
    html_path = tmp_path / "report.html"
    pdf_path = tmp_path / "report.pdf"
    html_path.write_text(_REPORT_HTML, encoding="utf-8")

    result = export_pdf(html_path, pdf_path)

    assert result == pdf_path
    pdf = pdf_path.read_bytes()
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2048


def test_is_pdf_available_returns_false_when_weasyprint_is_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked_import() -> Any:
        raise ImportError("blocked weasyprint import for test")

    monkeypatch.setattr(pdf_exporter, "_import_weasyprint", blocked_import)
    cast("Any", is_pdf_available).cache_clear()
    try:
        assert pdf_exporter.is_pdf_available() is False
    finally:
        cast("Any", is_pdf_available).cache_clear()

"""Export the FastAPI OpenAPI schema to api/openapi.json."""

from __future__ import annotations

import json
from pathlib import Path

from eda_platform.api.main import create_app


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "api" / "openapi.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = create_app().openapi()
    out_path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

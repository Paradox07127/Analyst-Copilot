from __future__ import annotations

from eda_platform.api.main import create_app


def test_openapi_exposes_typed_agent_handoff_and_provenance(tmp_path) -> None:
    document = create_app(tmp_path).openapi()
    operation = document["paths"]["/api/v1/sessions/{session_id}/agent-handoff"]["get"]
    response = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response["$ref"].endswith("/AgentHandoffDetail")
    assert {"404", "409", "413", "503"}.issubset(operation["responses"])
    retry_header = operation["responses"]["409"]["headers"]["Retry-After"]
    assert retry_header["schema"] == {"type": "integer", "minimum": 1}
    detail = document["components"]["schemas"]["ArtifactDetail"]
    for field in [
        "parents",
        "evidence",
        "env_digest",
        "code_ref",
        "plain_language",
    ]:
        assert field in detail["properties"]
    handoff = document["components"]["schemas"]["AgentHandoffDetail"]
    assert handoff["properties"]["payload"]["$ref"].endswith("/AgentHandoffV3")

from collections.abc import MutableMapping
from typing import Any


def clear_analysis_results(session_state: MutableMapping[str, Any]) -> None:
    result_keys = {
        "latest_analysis",
        "active_upload_id",
    }
    result_prefixes = ("analysis_", "verified_", "human_verified_")
    for key in tuple(session_state):
        if key in result_keys or key.startswith(result_prefixes):
            session_state.pop(key, None)


def clear_uploaded_data(session_state: MutableMapping[str, Any]) -> None:
    upload_generation = int(
        session_state.get("prescription_upload_generation", 0)
    )
    exact_keys = {
        "analyzer_synthetic_records",
        "synthetic_records",
    }
    sensitive_prefixes = (
        "prescription_upload_",
        "analysis_",
        "verified_",
        "human_verified_",
    )
    clear_analysis_results(session_state)
    for key in tuple(session_state):
        if key in exact_keys or (
            key != "prescription_upload_generation"
            and key.startswith(sensitive_prefixes)
        ):
            session_state.pop(key, None)

    session_state["prescription_upload_generation"] = upload_generation + 1

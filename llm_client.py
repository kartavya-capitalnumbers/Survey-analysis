"""
llm_client.py — thin wrapper around AWS Bedrock (boto3) via the Converse API.

PROTOTYPE NOTE
--------------
This is deliberately the ONLY place that talks to the model provider. All
prompt construction lives in the other modules (extraction / schema / qa) and
funnels through `complete_text()` and `complete_vision()` here.

It uses the Bedrock **Converse API** (`client.converse`), which is a single,
model-agnostic request/response shape that works across providers. The default
model is now **Amazon Nova Lite**, but because Converse is provider-neutral you
can paste ANY Bedrock model id / inference-profile ARN into the sidebar
(Nova, Claude, Llama, …) and it will work without code changes.

Authentication: the **default AWS credential chain** (environment variables,
shared ~/.aws profile, or instance role). Nothing is pasted into or stored by
this app. The Bedrock model id is supplied at runtime from the Streamlit
sidebar (or the BEDROCK_MODEL_ID env var).

Configuration is read from the environment (all overridable at runtime):
    BEDROCK_REGION / AWS_REGION   (optional)  — defaults to us-east-1
    BEDROCK_MODEL_ID              (optional)  — defaults to the Nova Lite id below
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Region + model defaults. The model default is Amazon Nova Lite (cross-region
# inference profile id). Both are overridable via env vars and, at runtime, via
# the Streamlit sidebar (`set_credentials`) — paste a full ARN there.
DEFAULT_REGION = (
    os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
)
DEFAULT_MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")

# Runtime overrides (e.g. the model id typed into the sidebar). In-memory only,
# never written to disk. Re-applied by the UI on every rerun.
_runtime_model: Optional[str] = None
_runtime_region: Optional[str] = None


def set_credentials(model: Optional[str] = None, region: Optional[str] = None) -> None:
    """Override the Bedrock model id / region at runtime (in-memory only).

    Note: AWS credentials are NOT handled here — they come from the default AWS
    credential chain (env vars / shared profile / instance role).
    """
    global _runtime_model, _runtime_region
    if model is not None:
        _runtime_model = model.strip() or None
    if region is not None:
        _runtime_region = region.strip() or None


def active_model() -> str:
    """Model id currently in effect: runtime override > env/default."""
    return _runtime_model or DEFAULT_MODEL


def active_region() -> str:
    """Region currently in effect: runtime override > env/default."""
    return _runtime_region or DEFAULT_REGION


def has_credentials() -> bool:
    """Best-effort local check that AWS credentials resolve (no network call)."""
    try:
        import botocore.session

        return botocore.session.Session().get_credentials() is not None
    except Exception:  # noqa: BLE001 - botocore missing or chain error
        return False


class LLMError(RuntimeError):
    """Raised for any provider/config problem so the UI can show it cleanly."""


def _client():
    """Lazily construct the Bedrock runtime client so import never hard-fails."""
    try:
        import boto3  # imported lazily: keeps the module importable without boto3
    except ImportError as exc:  # pragma: no cover - environment issue
        raise LLMError(
            "The `boto3` package is not installed. Run "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        return boto3.client("bedrock-runtime", region_name=active_region())
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Could not create Bedrock client: {exc}") from exc


def _invoke(
    system: str,
    content: List[Dict[str, Any]],
    model: Optional[str],
    max_tokens: int,
    temperature: float,
) -> str:
    """Send one Converse-API request to Bedrock and return the text output.

    `content` is a list of Converse content blocks — e.g. `[{"text": ...}]` for
    text-only, or an image block plus a text block for vision.
    """
    client = _client()
    kwargs: Dict[str, Any] = {
        "modelId": model or active_model(),
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    try:
        resp = client.converse(**kwargs)
    except Exception as exc:  # noqa: BLE001 - surface any SDK/Bedrock error to UI
        raise LLMError(f"Bedrock converse failed: {exc}") from exc

    return _first_text(resp)


def complete_text(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """Single-turn text completion. Returns the assistant's text content.

    Temperature defaults to 0.0 because every call in this prototype is an
    extraction / grounded-answer task where we want deterministic, faithful
    output rather than creativity.
    """
    return _invoke(system, [{"text": user}], model, max_tokens, temperature)


def complete_vision(
    system: str,
    user_text: str,
    image_bytes: bytes,
    media_type: str = "image/png",
    *,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Single image + text prompt -> text. Used for figure/map captioning.

    Nova Lite (and Claude) accept raw image bytes via the Converse API; no
    base64 encoding is needed — boto3 handles the wire format.
    """
    content: List[Dict[str, Any]] = [
        {"image": {"format": _image_format(media_type), "source": {"bytes": image_bytes}}},
        {"text": user_text},
    ]
    return _invoke(system, content, model, max_tokens, temperature)


def _image_format(media_type: str) -> str:
    """Map an image media type to a Converse `format` token (png/jpeg/gif/webp)."""
    mt = (media_type or "").lower()
    if "jpeg" in mt or "jpg" in mt:
        return "jpeg"
    if "gif" in mt:
        return "gif"
    if "webp" in mt:
        return "webp"
    return "png"


def _first_text(resp: Dict[str, Any]) -> str:
    """Extract the concatenated text blocks from a Bedrock Converse response."""
    parts: List[str] = []
    message = (resp.get("output") or {}).get("message") or {}
    for block in message.get("content", []) or []:
        if isinstance(block, dict) and "text" in block:
            parts.append(block.get("text", ""))
    return "".join(parts).strip()

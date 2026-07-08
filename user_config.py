"""
Per-visitor Azure OpenAI credentials.

Why this file exists
---------------------
This app is meant to be shared publicly (social media, etc.). If we kept a
single Azure OpenAI key in the server's .env, EVERY visitor's usage would be
billed to whoever runs the server. Instead:

  1. Each visitor enters their own Azure endpoint / key / deployments on the
     Settings page (static/settings.html).
  2. We test-call Azure with those exact values. If they don't work, we
     refuse to save them and tell the visitor why.
  3. On success we build an AzureOpenAI client from their values and keep it
     in memory, keyed by a random session id stored in an HttpOnly cookie in
     their browser. No other visitor can read or use it.
  4. Every endpoint that costs money (upload, query) requires a valid
     session; if the visitor hasn't configured credentials yet (or their
     credentials stop working), those endpoints refuse to run.

Nothing here is written to disk. Credentials live only in this process's
memory for the lifetime of the session, and are dropped on server restart
or after a period of inactivity.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import AzureOpenAI, AuthenticationError, APIConnectionError, NotFoundError

from rlm_core import make_azure_client, _create_completion, embed_text, cfg

# ─── Config ────────────────────────────────────────────────────────────────

SESSION_COOKIE_NAME = "rlm_session"
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(4 * 60 * 60)))  # 4h inactivity
_SWEEP_INTERVAL_SECONDS = 15 * 60


@dataclass
class SessionRecord:
    session_id: str
    client: AzureOpenAI
    endpoint: str
    root_deployment: str
    sub_deployment: str
    embedding_deployment: str
    root_reasoning_effort: str
    sub_reasoning_effort: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    def masked_summary(self) -> dict:
        """Safe-to-return-to-frontend view (never includes the API key)."""
        return {
            "configured": True,
            "endpoint": self.endpoint,
            "root_deployment": self.root_deployment,
            "sub_deployment": self.sub_deployment,
            "embedding_deployment": self.embedding_deployment,
            "root_reasoning_effort": self.root_reasoning_effort,
            "sub_reasoning_effort": self.sub_reasoning_effort,
        }


_SESSIONS: dict[str, SessionRecord] = {}
_LOCK = threading.Lock()


def _sweep_expired() -> None:
    while True:
        time.sleep(_SWEEP_INTERVAL_SECONDS)
        cutoff = time.time() - SESSION_TTL_SECONDS
        with _LOCK:
            expired = [sid for sid, rec in _SESSIONS.items() if rec.last_used_at < cutoff]
            for sid in expired:
                del _SESSIONS[sid]
        if expired:
            print(f"[user_config] swept {len(expired)} expired session(s)")


threading.Thread(target=_sweep_expired, daemon=True).start()


class CredentialError(Exception):
    """Raised when a visitor's Azure credentials/deployments don't work."""


def _friendly_error(e: Exception, what: str) -> str:
    status = getattr(e, "status_code", None)
    if isinstance(e, AuthenticationError) or status == 401:
        return (
            f"Azure rejected the API key while testing {what} "
            f"(401 Unauthorized). Double-check AZURE_API_KEY."
        )
    if isinstance(e, NotFoundError) or status == 404:
        return (
            f"Azure could not find the '{what}' deployment at that endpoint "
            f"(404). Check the deployment name and that it's deployed in "
            f"that Azure OpenAI resource."
        )
    if isinstance(e, APIConnectionError):
        return (
            f"Could not reach the Azure endpoint while testing {what}. "
            f"Check AZURE_ENDPOINT (must look like "
            f"https://<resource>.openai.azure.com/ or "
            f"https://<resource>.cognitiveservices.azure.com/)."
        )
    return f"Azure rejected the request while testing {what}: {str(e)[:200]}"


def validate_and_build_client(
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    root_deployment: str,
    sub_deployment: str,
    embedding_deployment: str,
    root_reasoning_effort: str,
    sub_reasoning_effort: str,
) -> AzureOpenAI:
    """
    Build a client from these exact values and prove they work by making
    real, minimal calls against Azure. Raises CredentialError with a clear
    reason on any failure. Never silently falls back to server-side env
    values — every field must be supplied and must work.
    """
    required = {
        "AZURE_ENDPOINT": endpoint,
        "AZURE_API_KEY": api_key,
        "AZURE_API_VERSION": api_version,
        "AZURE_ROOT_DEPLOYMENT": root_deployment,
        "AZURE_SUB_DEPLOYMENT": sub_deployment,
        "EMBEDDING_DEPLOYMENT": embedding_deployment,
        "ROOT_REASONING_EFFORT": root_reasoning_effort,
        "SUB_REASONING_EFFORT": sub_reasoning_effort,
    }
    missing = [k for k, v in required.items() if not v or not str(v).strip()]
    if missing:
        raise CredentialError(f"Missing required value(s): {', '.join(missing)}")

    client = make_azure_client(
        endpoint=endpoint.strip(),
        api_key=api_key.strip(),
        api_version=api_version.strip(),
        root_deployment=root_deployment.strip(),
        sub_deployment=sub_deployment.strip(),
        embedding_deployment=embedding_deployment.strip(),
        root_reasoning_effort=root_reasoning_effort.strip(),
        sub_reasoning_effort=sub_reasoning_effort.strip(),
    )

    # 1) sub deployment — tiny real completion
    try:
        _create_completion(
            client,
            deployment=cfg(client).sub_deployment,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            reasoning_effort=cfg(client).sub_reasoning_effort,
            max_tokens=16,
            stage="settings_validation",
        )
    except Exception as e:
        raise CredentialError(_friendly_error(e, f"AZURE_SUB_DEPLOYMENT ('{sub_deployment}')")) from e

    # 2) root deployment — only a separate call if it differs from sub
    if root_deployment.strip() != sub_deployment.strip():
        try:
            _create_completion(
                client,
                deployment=cfg(client).root_deployment,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                reasoning_effort=cfg(client).root_reasoning_effort,
                max_tokens=16,
                stage="settings_validation",
            )
        except Exception as e:
            raise CredentialError(_friendly_error(e, f"AZURE_ROOT_DEPLOYMENT ('{root_deployment}')")) from e

    # 3) embedding deployment
    emb = embed_text(client, "connection test", cfg(client).embedding_deployment)
    if emb is None:
        raise CredentialError(
            f"Could not create an embedding using EMBEDDING_DEPLOYMENT "
            f"('{embedding_deployment}'). Check the deployment name."
        )

    return client


def create_session(client: AzureOpenAI, *, endpoint: str) -> SessionRecord:
    session_id = secrets.token_urlsafe(32)
    c = cfg(client)
    record = SessionRecord(
        session_id=session_id,
        client=client,
        endpoint=endpoint,
        root_deployment=c.root_deployment,
        sub_deployment=c.sub_deployment,
        embedding_deployment=c.embedding_deployment,
        root_reasoning_effort=c.root_reasoning_effort,
        sub_reasoning_effort=c.sub_reasoning_effort,
    )
    with _LOCK:
        _SESSIONS[session_id] = record
    return record


def get_session(session_id: Optional[str]) -> Optional[SessionRecord]:
    if not session_id:
        return None
    with _LOCK:
        record = _SESSIONS.get(session_id)
        if record is None:
            return None
        if time.time() - record.last_used_at > SESSION_TTL_SECONDS:
            del _SESSIONS[session_id]
            return None
        record.last_used_at = time.time()
        return record


def clear_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _LOCK:
        _SESSIONS.pop(session_id, None)

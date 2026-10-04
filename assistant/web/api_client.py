"""Loopback-only HTTP client shared by the optional Streamlit UI."""

from __future__ import annotations

import json
from typing import Any, Iterator
from urllib.parse import urlsplit

import httpx


class AssistantAPIError(RuntimeError):
    pass


class AssistantAPI:
    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", *, transport: httpx.BaseTransport | None = None
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment
        ):
            raise ValueError("The local assistant API URL must be an HTTP loopback origin")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=20, transport=transport)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {"X-Requested-With": "assistant-ui"} if method != "GET" else {}
        try:
            response = self._client.request(method, f"/api{path}", headers=headers, **kwargs)
            self._check(response)
            return None if response.status_code == 204 else response.json()
        except httpx.RequestError as exc:
            raise AssistantAPIError(f"FastAPI unavailable at {self.base_url}: {exc}") from exc

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.is_error:
            try:
                detail = response.json().get("detail", response.reason_phrase)
            except (ValueError, AttributeError):
                detail = response.reason_phrase
            raise AssistantAPIError(f"{response.status_code}: {detail}")

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=body or {})

    def patch(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("PATCH", path, json=body)

    def delete(self, path: str) -> None:
        self._request("DELETE", path)

    def upload_pdf(self, name: str, contents: bytes, domain: str | None = None) -> dict:
        return self._request(
            "POST", "/papers/upload",
            data={"domain": domain} if domain else {},
            files={"file": (name, contents, "application/pdf")},
        )

    def stream_message(
        self, conversation_id: str, message: str, filters: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        try:
            with self._client.stream(
                "POST", f"/api/conversations/{conversation_id}/messages",
                headers={"X-Requested-With": "assistant-ui"},
                json={"message": message, "filters": filters or {}},
                timeout=httpx.Timeout(connect=10, read=None, write=30, pool=10),
            ) as response:
                self._check(response)
                finished = False
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise AssistantAPIError("Invalid chat event from FastAPI") from exc
                    if not isinstance(event, dict):
                        raise AssistantAPIError("Invalid chat event from FastAPI")
                    finished |= event.get("type") in {"final", "error"}
                    yield event
                if not finished:
                    raise AssistantAPIError("The chat stream ended without an answer")
        except httpx.RequestError as exc:
            raise AssistantAPIError(f"Chat connection lost: {exc}") from exc
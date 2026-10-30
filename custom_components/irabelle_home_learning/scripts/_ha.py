"""Shared plumbing for the tools beside this file.

Nothing here knows anything about this house: the URL comes from `HA_URL` and the token
from `HA_TOKEN`, or from `~/.ha_token`. Run these from a machine that can reach Home
Assistant - the container itself, or the host when Home Assistant is on localhost.

The websocket API is the one worth preferring for anything that can fail: the REST API
turns a `ServiceValidationError` into a bare `HTTP 500`, while the websocket returns the
message the Dev Tools UI and the LLM tools see. That difference has cost real debugging
time, which is why `call_service` lives here.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.request
from typing import Any

import aiohttp

DEFAULT_URL = "http://localhost:8123"
TIMEOUT = 120


def url() -> str:
    return os.environ.get("HA_URL", DEFAULT_URL).rstrip("/")


def token() -> str:
    value = os.environ.get("HA_TOKEN", "").strip()
    if value:
        return value
    path = pathlib.Path.home() / ".ha_token"
    if path.exists():
        return path.read_text().strip()
    raise SystemExit("no token: set HA_TOKEN, or write one to ~/.ha_token")


def rest(path: str) -> Any:
    """GET one API path and decode it."""
    request = urllib.request.Request(
        f"{url()}{path}", headers={"Authorization": f"Bearer {token()}"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def states() -> list[dict[str, Any]]:
    return rest("/api/states")


def state(entity_id: str) -> dict[str, Any] | None:
    for item in states():
        if item["entity_id"] == entity_id:
            return item
    return None


class Socket:
    """A websocket connection to Home Assistant, for commands the REST API lacks."""

    def __init__(self, session: aiohttp.ClientSession, ws: Any) -> None:
        self.session = session
        self.ws = ws
        self._ids = iter(range(1, 100_000))

    async def command(self, **payload: Any) -> dict[str, Any]:
        message_id = next(self._ids)
        await self.ws.send_json({"id": message_id, **payload})
        while True:
            message = await self.ws.receive_json()
            if message.get("id") == message_id:
                return message

    async def call_service(
        self, service: str, data: dict[str, Any], *, response: bool = True
    ) -> dict[str, Any]:
        """Call a service and surface the real error rather than an opaque 500."""
        domain, name = service.split(".", 1)
        message = await self.command(
            type="call_service",
            domain=domain,
            service=name,
            service_data=data,
            return_response=response,
        )
        if not message.get("success"):
            error = message.get("error") or {}
            raise SystemExit(
                f"{service} failed: {error.get('code')}: {error.get('message')}"
            )
        return (message.get("result") or {}).get("response") or {}


class connect:
    """`async with connect() as ha:` - an authenticated websocket."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._ws: Any = None

    async def __aenter__(self) -> Socket:
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(f"{url()}/api/websocket")
        await self._ws.receive_json()
        await self._ws.send_json({"type": "auth", "access_token": token()})
        reply = await self._ws.receive_json()
        if reply.get("type") != "auth_ok":
            raise SystemExit(f"auth failed: {reply}")
        return Socket(self._session, self._ws)

    async def __aexit__(self, *_: Any) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()

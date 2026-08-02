"""Shared test setup.

Required AGNES_* variables are set (with fake values) BEFORE any app import,
because app.config reads the environment at import time. Individual tests
that need missing config monkeypatch app.config attributes instead.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("AGNES_API_KEY", "test-agnes-key-not-real")
os.environ.setdefault("AGNES_BASE_URL", "https://agnes.test/v1")
os.environ.setdefault("AGNES_MODEL", "agnes-2.5-flash")

# Make `app` importable when pytest runs from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest

from app import llm


class AgnesMock:
    """Programmable fake Agnes backend over httpx.MockTransport.

    Records every request (url, headers, parsed JSON body) and serves
    responses from a configurable handler or queue.
    """

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self._queue: list[httpx.Response | Exception] = []
        self._handler = None

    def handle(self, fn):
        self._handler = fn

    def enqueue(self, *items):
        self._queue.extend(items)

    def ok(self, content: str = "ok"):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        try:
            import json
            self.bodies.append(json.loads(request.content.decode("utf-8")))
        except Exception:
            self.bodies.append({})
        if self._queue:
            item = self._queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self._handler:
            return self._handler(request)
        return self.ok()

    @property
    def call_count(self) -> int:
        return len(self.requests)


@pytest.fixture
def agnes(monkeypatch):
    """Inject a MockTransport-backed client into the LLM layer.

    Also neutralizes retry sleeps (recording requested delays) so retry
    tests run instantly.
    """
    mock = AgnesMock()
    client = httpx.AsyncClient(transport=httpx.MockTransport(mock))
    llm.set_llm_client(client)

    delays: list[float] = []

    async def _fake_sleep(seconds: float):
        delays.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)
    mock.sleep_delays = delays

    yield mock

    llm.set_llm_client(None)

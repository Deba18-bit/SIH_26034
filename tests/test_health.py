"""Tests for application health endpoints."""

import asyncio

import httpx
from app.main import app


def test_health_endpoint_returns_ok() -> None:
    """The health endpoint reports an available service."""
    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

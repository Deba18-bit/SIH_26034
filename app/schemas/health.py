"""Schemas for health status responses."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Public response for the service health check."""

    status: str

"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.scans import router as scans_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(scans_router, prefix=settings.api_prefix)

    from fastapi.staticfiles import StaticFiles
    scans_dir = settings.storage_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/data/scans", StaticFiles(directory=str(scans_dir)), name="scans")

    return app


app = create_app()

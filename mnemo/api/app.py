"""FastAPI application factory."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mnemo.config import Settings, get_settings
from mnemo.db import init_for_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = init_for_settings(settings)
        app.state.settings = settings
        app.state.db = conn
        yield
        conn.close()

    app = FastAPI(title="Mnemo", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    from mnemo.api.routes import health, cookies, video, memory
    app.include_router(health.router)
    app.include_router(cookies.router, prefix="/api/v1/auth")
    app.include_router(video.router, prefix="/api/v1/video")
    app.include_router(memory.router, prefix="/api/v1/memory")
    return app

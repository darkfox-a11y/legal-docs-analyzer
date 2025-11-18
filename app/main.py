from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_db
from app.auth.routes import router as auth_router
from app.documents.routes import router as document_router

@asynccontextmanager
async def lifespan(app : FastAPI):
    # Startup code: Initialize database
    init_db()
    yield
    # Shutdown code: (if any cleanup is needed)
    print("👋 Shutting down...")

app = FastAPI(
    title=settings.app_name,
    description="AI-powered legal document analysis with RAG architecture",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router,prefix="/auth", tags=["Authentication"])
app.include_router(document_router,prefix="/documents", tags=["Documents"])

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": settings.app_version
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

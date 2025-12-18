"""
Pytest configuration and fixtures for RAG Backend tests.
"""
import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel

# Override settings before importing app
import os
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_rag.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["DEBUG"] = "true"

from app.main import app
from app.database import get_db
from app.models import user, document, chat  # Import to register models


# Test database engine
test_engine = create_async_engine(
    "sqlite+aiosqlite:///./test_rag.db",
    echo=False,
)

test_session_maker = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    # Clean up pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()


def pytest_sessionfinish(session, exitstatus):
    """Clean up after all tests complete."""
    import gc
    import sys
    
    # Clear any cached models from sentence-transformers
    try:
        from app.services.retrieval_service import EmbeddingService
        # Reset model cache if exists
        EmbeddingService._model = None
    except Exception:
        pass
    
    gc.collect()
    
    # Force exit to prevent hanging on thread cleanup
    import os
    os._exit(exitstatus)


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    # Create session
    async with test_session_maker() as session:
        yield session
    
    # Drop tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with overridden database."""
    
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def test_user_data():
    """Sample user data for tests."""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "name": "Test User"
    }


@pytest.fixture
def test_document_content():
    """Sample document content for tests."""
    return b"This is a test document content for RAG testing."

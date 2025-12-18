"""
Tests for admin/monitoring service and endpoints.
Run with: pytest tests/test_admin.py -v
"""
import pytest
import pytest_asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.monitoring_service import MetricsService
from app.services.auth_service import AuthService
from app.models.user import RoleEnum


class TestMetricsService:
    """Unit tests for MetricsService."""
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_empty(self, db_session: AsyncSession):
        """Test system metrics with empty database."""
        service = MetricsService(db_session)
        
        metrics = await service.get_system_metrics()
        
        assert metrics.total_users == 0
        assert metrics.total_documents == 0
        assert metrics.total_conversations == 0
        assert metrics.generated_at != ""
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_with_users(self, db_session: AsyncSession):
        """Test system metrics counts users."""
        # Create users
        auth_service = AuthService(db_session)
        for i in range(3):
            await auth_service.create_user(
                email=f"metrics{i}@example.com",
                password="password123",
            )
        
        metrics_service = MetricsService(db_session)
        metrics = await metrics_service.get_system_metrics()
        
        assert metrics.total_users == 3
    
    @pytest.mark.asyncio
    async def test_get_user_metrics(self, db_session: AsyncSession):
        """Test per-user metrics."""
        # Create user
        auth_service = AuthService(db_session)
        user = await auth_service.create_user(
            email="usermetrics@example.com",
            password="password123",
        )
        
        metrics_service = MetricsService(db_session)
        metrics = await metrics_service.get_user_metrics(str(user.id))
        
        assert metrics.user_id == str(user.id)
        assert metrics.conversations == 0
        assert metrics.messages == 0
    
    @pytest.mark.asyncio
    async def test_get_recent_query_logs_empty(self, db_session: AsyncSession):
        """Test query logs with empty database."""
        service = MetricsService(db_session)
        
        logs = await service.get_recent_query_logs()
        
        assert logs == []
    
    @pytest.mark.asyncio
    async def test_get_error_summary_empty(self, db_session: AsyncSession):
        """Test error summary with no errors."""
        service = MetricsService(db_session)
        
        summary = await service.get_error_summary()
        
        assert summary["total_errors"] == 0
        assert summary["errors"] == []


class TestAdminEndpoints:
    """Integration tests for admin API endpoints."""
    
    @pytest.fixture
    async def admin_headers(self, client, db_session):
        """Create admin user and get auth headers."""
        # Create admin user directly with is_admin flag
        auth_service = AuthService(db_session)
        user = await auth_service.create_user(
            email="admin@example.com",
            password="adminpassword",
            role=RoleEnum.ADMIN,
        )
        
        # Set is_admin flag
        user.is_admin = True
        await db_session.commit()
        
        # Login
        response = await client.post("/v1/auth/login", json={
            "email": "admin@example.com",
            "password": "adminpassword",
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    async def regular_headers(self, client, test_user_data):
        """Get headers for regular (non-admin) user."""
        await client.post("/v1/auth/signup", json=test_user_data)
        response = await client.post("/v1/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"],
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client, admin_headers):
        """Test GET /v1/admin/metrics."""
        response = await client.get("/v1/admin/metrics", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "users" in data
        assert "documents" in data
        assert "chat" in data
        assert "performance" in data
        assert "feedback" in data
    
    @pytest.mark.asyncio
    async def test_metrics_forbidden_for_regular_user(self, client, regular_headers):
        """Test that regular users can't access admin metrics."""
        response = await client.get("/v1/admin/metrics", headers=regular_headers)
        
        # Should be forbidden (403) for non-admin users
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_logs_endpoint(self, client, admin_headers):
        """Test GET /v1/admin/logs."""
        response = await client.get("/v1/admin/logs", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "count" in data
    
    @pytest.mark.asyncio
    async def test_errors_endpoint(self, client, admin_headers):
        """Test GET /v1/admin/errors."""
        response = await client.get("/v1/admin/errors", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_errors" in data
        assert "errors" in data
    
    @pytest.mark.asyncio
    async def test_list_users_admin(self, client, admin_headers):
        """Test GET /v1/admin/users (admin only)."""
        response = await client.get("/v1/admin/users", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data

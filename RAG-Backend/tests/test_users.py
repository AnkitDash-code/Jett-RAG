"""
Tests for user service and endpoints.
Run with: pytest tests/test_users.py -v
"""
import pytest
import pytest_asyncio
import uuid
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.user_service import UserService
from app.services.auth_service import AuthService


class TestUserService:
    """Unit tests for UserService."""
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, db_session: AsyncSession):
        """Test getting user by ID."""
        # Create a user first
        auth_service = AuthService(db_session)
        user = await auth_service.create_user(
            email="getbyid@example.com",
            password="password123",
            name="Test User",
        )
        
        # Get by ID
        user_service = UserService(db_session)
        found = await user_service.get_user_by_id(user.id)
        
        assert found is not None
        assert found.email == "getbyid@example.com"
        assert found.name == "Test User"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, db_session: AsyncSession):
        """Test getting non-existent user returns None."""
        user_service = UserService(db_session)
        
        found = await user_service.get_user_by_id(uuid.uuid4())
        
        assert found is None
    
    @pytest.mark.asyncio
    async def test_update_user(self, db_session: AsyncSession):
        """Test updating user profile."""
        # Create user
        auth_service = AuthService(db_session)
        user = await auth_service.create_user(
            email="update@example.com",
            password="password123",
        )
        
        # Update
        user_service = UserService(db_session)
        updated = await user_service.update_user(
            user_id=user.id,
            name="Updated Name",
            department="Engineering",
        )
        
        assert updated.name == "Updated Name"
        assert updated.department == "Engineering"
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_user(self, db_session: AsyncSession):
        """Test updating non-existent user raises error."""
        from app.middleware.error_handler import NotFoundError
        
        user_service = UserService(db_session)
        
        with pytest.raises(NotFoundError):
            await user_service.update_user(
                user_id=uuid.uuid4(),
                name="New Name",
            )
    
    @pytest.mark.asyncio
    async def test_list_users(self, db_session: AsyncSession):
        """Test listing users with pagination."""
        # Create multiple users
        auth_service = AuthService(db_session)
        for i in range(5):
            await auth_service.create_user(
                email=f"list{i}@example.com",
                password="password123",
            )
        
        user_service = UserService(db_session)
        
        # Get first page
        users, total = await user_service.list_users(page=1, page_size=3)
        
        assert total == 5
        assert len(users) == 3
        
        # Get second page
        users2, total2 = await user_service.list_users(page=2, page_size=3)
        
        assert total2 == 5
        assert len(users2) == 2
    
    @pytest.mark.asyncio
    async def test_deactivate_user(self, db_session: AsyncSession):
        """Test deactivating user account."""
        # Create user
        auth_service = AuthService(db_session)
        user = await auth_service.create_user(
            email="deactivate@example.com",
            password="password123",
        )
        
        # Status could be enum or string
        status_val = user.status.value if hasattr(user.status, 'value') else user.status
        assert status_val == "active"
        
        # Deactivate
        user_service = UserService(db_session)
        deactivated = await user_service.deactivate_user(user.id)
        
        # Check deactivated status
        deact_status = deactivated.status.value if hasattr(deactivated.status, 'value') else deactivated.status
        assert deact_status == "inactive"


class TestUserEndpoints:
    """Integration tests for user API endpoints."""
    
    @pytest.fixture
    async def auth_headers(self, client, test_user_data):
        """Get auth headers for authenticated requests."""
        await client.post("/v1/auth/signup", json=test_user_data)
        response = await client.post("/v1/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"],
        })
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.mark.asyncio
    async def test_get_me_endpoint(self, client, auth_headers, test_user_data):
        """Test GET /v1/users/me."""
        response = await client.get("/v1/users/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user_data["email"]
        assert data["name"] == test_user_data["name"]
        assert "roles" in data
    
    @pytest.mark.asyncio
    async def test_update_me_endpoint(self, client, auth_headers):
        """Test PATCH /v1/users/me."""
        response = await client.patch(
            "/v1/users/me",
            json={"name": "New Name", "department": "Sales"},
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"
        assert data["department"] == "Sales"
    
    @pytest.mark.asyncio
    async def test_get_me_unauthorized(self, client):
        """Test that /users/me requires authentication."""
        response = await client.get("/v1/users/me")
        assert response.status_code == 401

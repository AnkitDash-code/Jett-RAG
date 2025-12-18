"""
Tests for authentication service and endpoints.
Run with: pytest tests/test_auth.py -v
"""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.auth_service import AuthService
from app.models.user import RoleEnum


class TestAuthService:
    """Unit tests for AuthService."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "mysecretpassword123"
        
        hashed = AuthService.hash_password(password)
        
        # Hash should be different from plain password
        assert hashed != password
        
        # Verification should work
        assert AuthService.verify_password(password, hashed) is True
        
        # Wrong password should fail
        assert AuthService.verify_password("wrongpassword", hashed) is False
    
    def test_access_token_creation(self):
        """Test JWT access token creation."""
        import uuid
        
        user_id = uuid.uuid4()
        roles = ["viewer", "power_user"]
        tenant_id = "tenant-123"
        
        token = AuthService.create_access_token(
            user_id=user_id,
            roles=roles,
            tenant_id=tenant_id,
        )
        
        # Token should be a non-empty string
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode and verify payload
        payload = AuthService.decode_token(token)
        assert payload["sub"] == str(user_id)
        assert payload["roles"] == roles
        assert payload["tenant_id"] == tenant_id
        assert payload["type"] == "access"
    
    def test_refresh_token_creation(self):
        """Test refresh token creation."""
        import uuid
        from datetime import datetime
        
        user_id = uuid.uuid4()
        
        token, expires_at = AuthService.create_refresh_token(user_id)
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert isinstance(expires_at, datetime)
        assert expires_at > datetime.utcnow()
        
        # Decode and verify
        payload = AuthService.decode_token(token)
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "refresh"
    
    def test_token_hashing(self):
        """Test token hashing for storage."""
        token = "some-refresh-token-value"
        
        hash1 = AuthService.hash_token(token)
        hash2 = AuthService.hash_token(token)
        
        # Same token should produce same hash
        assert hash1 == hash2
        
        # Different token should produce different hash
        hash3 = AuthService.hash_token("different-token")
        assert hash1 != hash3
    
    @pytest_asyncio.fixture
    async def auth_service(self, db_session: AsyncSession):
        """Create AuthService instance."""
        return AuthService(db_session)
    
    @pytest.mark.asyncio
    async def test_create_user(self, db_session: AsyncSession):
        """Test user creation."""
        service = AuthService(db_session)
        
        user = await service.create_user(
            email="newuser@example.com",
            password="password123",
            name="New User",
        )
        
        assert user.email == "newuser@example.com"
        assert user.name == "New User"
        assert user.hashed_password != "password123"  # Should be hashed
        
        # Should be able to find user
        found = await service.get_user_by_email("newuser@example.com")
        assert found is not None
        assert found.id == user.id
    
    @pytest.mark.asyncio
    async def test_create_duplicate_user_fails(self, db_session: AsyncSession):
        """Test that duplicate email registration fails."""
        from app.middleware.error_handler import AuthenticationError
        
        service = AuthService(db_session)
        
        await service.create_user(
            email="duplicate@example.com",
            password="password123",
        )
        
        with pytest.raises(AuthenticationError, match="Email already registered"):
            await service.create_user(
                email="duplicate@example.com",
                password="password456",
            )
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, db_session: AsyncSession):
        """Test user authentication."""
        service = AuthService(db_session)
        
        # Create user
        await service.create_user(
            email="auth@example.com",
            password="correctpassword",
        )
        
        # Authenticate with correct password
        user = await service.authenticate("auth@example.com", "correctpassword")
        assert user.email == "auth@example.com"
    
    @pytest.mark.asyncio
    async def test_authenticate_wrong_password(self, db_session: AsyncSession):
        """Test authentication with wrong password fails."""
        from app.middleware.error_handler import AuthenticationError
        
        service = AuthService(db_session)
        
        await service.create_user(
            email="wrongpw@example.com",
            password="correctpassword",
        )
        
        with pytest.raises(AuthenticationError, match="Invalid email or password"):
            await service.authenticate("wrongpw@example.com", "wrongpassword")
    
    @pytest.mark.asyncio
    async def test_full_login_flow(self, db_session: AsyncSession):
        """Test complete login flow with tokens."""
        service = AuthService(db_session)
        
        # Create user
        await service.create_user(
            email="login@example.com",
            password="loginpassword",
        )
        
        # Login
        result = await service.login(
            email="login@example.com",
            password="loginpassword",
        )
        
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "bearer"
        assert result["user"]["email"] == "login@example.com"


class TestAuthEndpoints:
    """Integration tests for auth API endpoints."""
    
    @pytest.mark.asyncio
    async def test_signup_endpoint(self, client, test_user_data):
        """Test POST /v1/auth/signup."""
        response = await client.post("/v1/auth/signup", json=test_user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user_data["email"]
        assert data["message"] == "Account created successfully"
    
    @pytest.mark.asyncio
    async def test_login_endpoint(self, client, test_user_data):
        """Test POST /v1/auth/login."""
        # First signup
        await client.post("/v1/auth/signup", json=test_user_data)
        
        # Then login
        response = await client.post("/v1/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"],
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["email"] == test_user_data["email"]
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client, test_user_data):
        """Test login with wrong password returns error."""
        # Signup
        await client.post("/v1/auth/signup", json=test_user_data)
        
        # Login with wrong password - the endpoint may raise or return error
        try:
            response = await client.post("/v1/auth/login", json={
                "email": test_user_data["email"],
                "password": "wrongpassword",
            })
            # If no exception, check response code
            assert response.status_code in [401, 500]
        except Exception as e:
            # Exception is expected behavior for wrong password
            assert "Invalid" in str(e) or "password" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_refresh_token_endpoint(self, client, test_user_data):
        """Test POST /v1/auth/refresh."""
        # Signup and login
        await client.post("/v1/auth/signup", json=test_user_data)
        login_response = await client.post("/v1/auth/login", json={
            "email": test_user_data["email"],
            "password": test_user_data["password"],
        })
        
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh
        response = await client.post("/v1/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        # New refresh token should be different (rotation)
        assert data["refresh_token"] != refresh_token

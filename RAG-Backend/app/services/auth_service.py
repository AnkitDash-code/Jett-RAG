"""
Authentication service handling password hashing, JWT tokens, and session management.
"""
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.config import settings
from app.models.user import User, RefreshToken, Role, UserRole, RoleEnum
from app.middleware.error_handler import AuthenticationError, AuthorizationError


# Password hasher
ph = PasswordHasher()


class AuthService:
    """Service for authentication operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # ==================== Password Hashing ====================
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using Argon2."""
        return ph.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            ph.verify(hashed_password, plain_password)
            return True
        except VerifyMismatchError:
            return False
    
    # ==================== Token Generation ====================
    
    @staticmethod
    def create_access_token(
        user_id: uuid.UUID,
        roles: list[str],
        tenant_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create a JWT access token."""
        if expires_delta is None:
            expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            "sub": str(user_id),
            "roles": roles,
            "tenant_id": tenant_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }
        
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def create_refresh_token(user_id: uuid.UUID) -> Tuple[str, datetime]:
        """Create a refresh token and return (token, expiry)."""
        expires_at = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "sub": str(user_id),
            "exp": expires_at,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": str(uuid.uuid4()),  # Unique token ID
        }
        
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return token, expires_at
    
    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Hash a token for storage (refresh tokens)."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    # ==================== User Operations ====================
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email address."""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get a user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def create_user(
        self,
        email: str,
        password: str,
        name: Optional[str] = None,
        role: RoleEnum = RoleEnum.VIEWER,
    ) -> User:
        """Create a new user with default role."""
        # Check if user exists
        existing = await self.get_user_by_email(email)
        if existing:
            raise AuthenticationError("Email already registered")
        
        # Create user
        user = User(
            email=email,
            hashed_password=self.hash_password(password),
            name=name,
        )
        self.db.add(user)
        await self.db.flush()
        
        # Assign default role
        await self._assign_role(user.id, role)
        
        return user
    
    async def _assign_role(self, user_id: uuid.UUID, role_name: RoleEnum) -> None:
        """Assign a role to a user."""
        # Get or create role
        result = await self.db.execute(
            select(Role).where(Role.name == role_name.value)
        )
        role = result.scalar_one_or_none()
        
        if not role:
            role = Role(name=role_name.value, description=f"{role_name.value} role")
            self.db.add(role)
            await self.db.flush()
        
        # Create user-role association
        user_role = UserRole(user_id=user_id, role_id=role.id)
        self.db.add(user_role)
    
    async def get_user_roles(self, user_id: uuid.UUID) -> list[str]:
        """Get all roles for a user."""
        result = await self.db.execute(
            select(Role.name)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(UserRole.user_id == user_id)
        )
        return [row[0] for row in result.all()]
    
    # ==================== Session Management ====================
    
    async def store_refresh_token(
        self,
        user_id: uuid.UUID,
        token: str,
        expires_at: datetime,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> RefreshToken:
        """Store a refresh token in the database."""
        refresh_token = RefreshToken(
            user_id=user_id,
            token_hash=self.hash_token(token),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.db.add(refresh_token)
        await self.db.flush()
        return refresh_token
    
    async def validate_refresh_token(self, token: str) -> User:
        """Validate a refresh token and return the associated user."""
        # Decode token
        payload = self.decode_token(token)
        
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type")
        
        # Check if token is stored and not revoked
        token_hash = self.hash_token(token)
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.is_revoked == False,
                RefreshToken.expires_at > datetime.utcnow(),
            )
        )
        stored_token = result.scalar_one_or_none()
        
        if not stored_token:
            raise AuthenticationError("Refresh token is invalid or expired")
        
        # Get user
        user = await self.get_user_by_id(uuid.UUID(payload["sub"]))
        if not user:
            raise AuthenticationError("User not found")
        
        return user
    
    async def revoke_refresh_token(self, token: str) -> None:
        """Revoke a refresh token."""
        token_hash = self.hash_token(token)
        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        stored_token = result.scalar_one_or_none()
        
        if stored_token:
            stored_token.is_revoked = True
            stored_token.revoked_at = datetime.utcnow()
    
    async def revoke_all_user_tokens(self, user_id: uuid.UUID) -> None:
        """Revoke all refresh tokens for a user (logout everywhere)."""
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False,
            )
        )
        tokens = result.scalars().all()
        
        for token in tokens:
            token.is_revoked = True
            token.revoked_at = datetime.utcnow()
    
    # ==================== Authentication Flow ====================
    
    async def authenticate(self, email: str, password: str) -> User:
        """Authenticate a user with email and password."""
        user = await self.get_user_by_email(email)
        
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        if not self.verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")
        
        if user.status != "active":
            raise AuthenticationError("Account is not active")
        
        # Update last login
        user.last_login_at = datetime.utcnow()
        
        return user
    
    async def login(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> dict:
        """Full login flow returning access and refresh tokens."""
        # Authenticate
        user = await self.authenticate(email, password)
        
        # Get roles
        roles = await self.get_user_roles(user.id)
        
        # Add admin role if user.is_admin is True
        if user.is_admin and "admin" not in roles:
            roles.append("admin")
        
        # Create tokens
        access_token = self.create_access_token(
            user_id=user.id,
            roles=roles,
            tenant_id=user.tenant_id,
        )
        refresh_token, refresh_expires = self.create_refresh_token(user.id)
        
        # Store refresh token
        await self.store_refresh_token(
            user_id=user.id,
            token=refresh_token,
            expires_at=refresh_expires,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "roles": roles,
            }
        }
    
    async def refresh_tokens(
        self,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> dict:
        """Refresh access token using refresh token (with rotation)."""
        # Validate current refresh token
        user = await self.validate_refresh_token(refresh_token)
        
        # Revoke old refresh token
        await self.revoke_refresh_token(refresh_token)
        
        # Get roles
        roles = await self.get_user_roles(user.id)
        
        # Create new tokens
        new_access_token = self.create_access_token(
            user_id=user.id,
            roles=roles,
            tenant_id=user.tenant_id,
        )
        new_refresh_token, refresh_expires = self.create_refresh_token(user.id)
        
        # Store new refresh token
        await self.store_refresh_token(
            user_id=user.id,
            token=new_refresh_token,
            expires_at=refresh_expires,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

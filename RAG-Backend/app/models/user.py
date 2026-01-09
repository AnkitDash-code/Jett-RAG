"""
User and authentication related models.
Includes: User, Role, UserRole, RefreshToken, APIKey
"""
import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlmodel import SQLModel, Field, Relationship


class RoleEnum(str, Enum):
    """Available user roles for RBAC."""
    ADMIN = "admin"
    POWER_USER = "power_user"
    VIEWER = "viewer"
    INGESTOR = "ingestor"


class UserStatusEnum(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


# ==================== Association Tables ====================

class UserRole(SQLModel, table=True):
    """Many-to-many relationship between users and roles."""
    __tablename__ = "user_roles"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True)
    role_id: int = Field(foreign_key="roles.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ==================== Main Models ====================

class Role(SQLModel, table=True):
    """Role model for RBAC."""
    __tablename__ = "roles"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True, max_length=50)
    description: Optional[str] = Field(default=None, max_length=255)
    permissions: str = Field(default="")  # JSON string of permissions
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    users: List["User"] = Relationship(
        back_populates="roles",
        link_model=UserRole,
    )


class User(SQLModel, table=True):
    """User model for authentication and identity."""
    __tablename__ = "users"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    email: str = Field(unique=True, index=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    
    # Profile
    name: Optional[str] = Field(default=None, max_length=100)
    avatar_url: Optional[str] = Field(default=None, max_length=512)
    
    # Status and flags
    status: UserStatusEnum = Field(default=UserStatusEnum.ACTIVE)
    is_admin: bool = Field(default=False)
    is_verified: bool = Field(default=False)
    
    # Multi-tenancy support
    tenant_id: Optional[str] = Field(default=None, index=True, max_length=50)
    department: Optional[str] = Field(default=None, max_length=100)
    
    # Access control metadata
    allowed_access_levels: str = Field(default="public")  # Comma-separated levels
    
    # Preferences (JSON string)
    preferences: str = Field(default="{}")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    roles: List[Role] = Relationship(
        back_populates="users",
        link_model=UserRole,
    )
    refresh_tokens: List["RefreshToken"] = Relationship(back_populates="user")
    api_keys: List["APIKey"] = Relationship(back_populates="user")


class RefreshToken(SQLModel, table=True):
    """Refresh token storage for session management."""
    __tablename__ = "refresh_tokens"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True)
    token_hash: str = Field(unique=True, index=True, max_length=255)
    
    # Token metadata
    device_info: Optional[str] = Field(default=None, max_length=255)
    ip_address: Optional[str] = Field(default=None, max_length=45)
    user_agent: Optional[str] = Field(default=None, max_length=512)
    
    # Status
    is_revoked: bool = Field(default=False)
    revoked_at: Optional[datetime] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    
    # Relationships
    user: Optional[User] = Relationship(back_populates="refresh_tokens")


class APIKey(SQLModel, table=True):
    """API keys for programmatic access."""
    __tablename__ = "api_keys"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True)
    
    name: str = Field(max_length=100)
    key_hash: str = Field(unique=True, index=True, max_length=255)
    key_prefix: str = Field(max_length=10)  # First 8 chars for identification
    
    # Permissions/scopes
    scopes: str = Field(default="read")  # Comma-separated scopes
    
    # Status
    is_active: bool = Field(default=True)
    
    # Usage tracking
    last_used_at: Optional[datetime] = Field(default=None)
    usage_count: int = Field(default=0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    user: Optional[User] = Relationship(back_populates="api_keys")

"""
Session and Device models for Phase 7.
Tracks user sessions across multiple devices.
"""
import uuid
from datetime import datetime
from typing import Optional
from enum import Enum

from sqlmodel import SQLModel, Field, Relationship


class SessionStatus(str, Enum):
    """Session status enum."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"


class UserSession(SQLModel, table=True):
    """
    Tracks user sessions across devices.
    
    A user can have multiple active sessions (one per device).
    Sessions are tied to refresh tokens and expire independently.
    """
    __tablename__ = "user_sessions"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Ownership
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True)
    
    # Session identification
    session_token: str = Field(index=True, unique=True)  # Hashed token
    refresh_token_jti: Optional[str] = Field(default=None, index=True)  # JWT ID for revocation
    
    # Device info
    device_id: Optional[str] = Field(default=None, index=True)
    device_name: Optional[str] = Field(default=None)
    device_type: Optional[str] = Field(default=None)  # "mobile", "desktop", "tablet", "unknown"
    user_agent: Optional[str] = Field(default=None)
    
    # Location info (from IP)
    ip_address: Optional[str] = Field(default=None)
    location_country: Optional[str] = Field(default=None)
    location_city: Optional[str] = Field(default=None)
    
    # Status
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, index=True)
    is_current: bool = Field(default=False)  # Flag for the most recently used session
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    expires_at: Optional[datetime] = Field(default=None, index=True)
    revoked_at: Optional[datetime] = Field(default=None)
    
    # Metadata
    metadata_json: str = Field(default="{}")
    
    # Relationships
    # user: "User" = Relationship(back_populates="sessions")
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status == SessionStatus.ACTIVE and not self.is_expired
    
    def revoke(self) -> None:
        """Revoke this session."""
        self.status = SessionStatus.REVOKED
        self.revoked_at = datetime.utcnow()
    
    def touch(self) -> None:
        """Update last active timestamp."""
        self.last_active_at = datetime.utcnow()


class SessionActivity(SQLModel, table=True):
    """
    Tracks activity within a session.
    
    Used for security monitoring and session analytics.
    """
    __tablename__ = "session_activities"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="user_sessions.id", index=True)
    
    # Activity details
    activity_type: str = Field(index=True)  # "login", "api_call", "logout", etc.
    endpoint: Optional[str] = Field(default=None)
    method: Optional[str] = Field(default=None)
    
    # Request info
    ip_address: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Additional data
    metadata_json: str = Field(default="{}")

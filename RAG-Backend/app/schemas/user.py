"""
User schemas for request/response validation.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class UserResponse(BaseModel):
    """User profile response."""
    id: str
    email: str
    name: Optional[str]
    avatar_url: Optional[str]
    status: str
    roles: List[str]
    tenant_id: Optional[str]
    department: Optional[str]
    created_at: datetime
    last_login_at: Optional[datetime]
    
    model_config = {"from_attributes": True}


class UserUpdateRequest(BaseModel):
    """User profile update request."""
    name: Optional[str] = Field(default=None, max_length=100)
    avatar_url: Optional[str] = Field(default=None, max_length=512)
    department: Optional[str] = Field(default=None, max_length=100)
    preferences: Optional[dict] = None


class UserListResponse(BaseModel):
    """Paginated user list response."""
    users: List[UserResponse]
    total: int
    page: int
    page_size: int
    pages: int

"""
User service for profile management.
"""
import uuid
import json
from typing import Optional, List
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from app.models.user import User
from app.middleware.error_handler import NotFoundError


class UserService:
    """Service for user operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get a user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def update_user(
        self,
        user_id: uuid.UUID,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        department: Optional[str] = None,
        preferences: Optional[dict] = None,
    ) -> User:
        """Update user profile."""
        user = await self.get_user_by_id(user_id)
        
        if not user:
            raise NotFoundError("User not found")
        
        if name is not None:
            user.name = name
        if avatar_url is not None:
            user.avatar_url = avatar_url
        if department is not None:
            user.department = department
        if preferences is not None:
            user.preferences = json.dumps(preferences)
        
        user.updated_at = datetime.utcnow()
        
        await self.db.flush()
        return user
    
    async def list_users(
        self,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[List[User], int]:
        """List users with pagination and filters."""
        query = select(User)
        count_query = select(func.count(User.id))
        
        # Apply filters
        if tenant_id:
            query = query.where(User.tenant_id == tenant_id)
            count_query = count_query.where(User.tenant_id == tenant_id)
        if status:
            query = query.where(User.status == status)
            count_query = count_query.where(User.status == status)
        
        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar_one()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size).order_by(User.created_at.desc())
        
        result = await self.db.execute(query)
        users = result.scalars().all()
        
        return list(users), total
    
    async def deactivate_user(self, user_id: uuid.UUID) -> User:
        """Deactivate a user account."""
        user = await self.get_user_by_id(user_id)
        
        if not user:
            raise NotFoundError("User not found")
        
        user.status = "inactive"
        user.updated_at = datetime.utcnow()
        
        await self.db.flush()
        return user

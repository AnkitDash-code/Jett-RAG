"""
User API endpoints.
GET /users/me, PATCH /users/me, GET /users/{id}
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.user_service import UserService
from app.services.auth_service import AuthService
from app.schemas.user import UserResponse, UserUpdateRequest, UserListResponse
from app.api.deps import get_current_user, CurrentUser, can_manage_users
from app.middleware.error_handler import NotFoundError

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the current authenticated user's profile.
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(current_user.id)
    
    if not user:
        raise NotFoundError("User not found")
    
    # Get roles
    auth_service = AuthService(db)
    roles = await auth_service.get_user_roles(user.id)
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        status=user.status.value,
        roles=roles,
        tenant_id=user.tenant_id,
        department=user.department,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@router.patch("/me", response_model=UserResponse)
async def update_current_user_profile(
    request: UserUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update the current authenticated user's profile.
    """
    user_service = UserService(db)
    
    user = await user_service.update_user(
        user_id=current_user.id,
        name=request.name,
        avatar_url=request.avatar_url,
        department=request.department,
        preferences=request.preferences,
    )
    
    # Get roles
    auth_service = AuthService(db)
    roles = await auth_service.get_user_roles(user.id)
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        status=user.status.value,
        roles=roles,
        tenant_id=user.tenant_id,
        department=user.department,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: str,
    current_user: CurrentUser = Depends(can_manage_users),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a user by ID (admin only).
    """
    user_service = UserService(db)
    user = await user_service.get_user_by_id(uuid.UUID(user_id))
    
    if not user:
        raise NotFoundError("User not found")
    
    # Get roles
    auth_service = AuthService(db)
    roles = await auth_service.get_user_roles(user.id)
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        status=user.status.value,
        roles=roles,
        tenant_id=user.tenant_id,
        department=user.department,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )

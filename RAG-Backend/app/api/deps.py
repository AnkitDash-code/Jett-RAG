"""
FastAPI dependencies for authentication and authorization.
"""
import uuid
from typing import Optional, List
from dataclasses import dataclass

from fastapi import Depends, Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.auth_service import AuthService
from app.models.user import User, RoleEnum
from app.middleware.error_handler import AuthenticationError, AuthorizationError


# HTTP Bearer scheme
security = HTTPBearer(auto_error=False)


@dataclass
class CurrentUser:
    """Authenticated user context."""
    id: uuid.UUID
    email: str
    name: Optional[str]
    roles: List[str]
    tenant_id: Optional[str]
    is_admin: bool
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles or self.is_admin
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(self.has_role(r) for r in roles) or self.is_admin


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> CurrentUser:
    """
    Dependency to get the current authenticated user from JWT token.
    Raises 401 if not authenticated.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode token
        payload = AuthService.decode_token(credentials.credentials)
        
        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")
        
        user_id = uuid.UUID(payload["sub"])
        roles = payload.get("roles", [])
        tenant_id = payload.get("tenant_id")
        
        # Get user from database for fresh data
        auth_service = AuthService(db)
        user = await auth_service.get_user_by_id(user_id)
        
        if not user:
            raise AuthenticationError("User not found")
        
        if user.status != "active":
            raise AuthenticationError("Account is not active")
        
        # Store user info in request state for logging
        request.state.user_id = str(user_id)
        
        return CurrentUser(
            id=user_id,
            email=user.email,
            name=user.name,
            roles=roles,
            tenant_id=tenant_id,
            is_admin=user.is_admin or RoleEnum.ADMIN.value in roles,
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[CurrentUser]:
    """
    Optional authentication - returns None if not authenticated.
    Use for endpoints that work with or without auth.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(request, credentials, db)
    except HTTPException:
        return None


def require_roles(*required_roles: str):
    """
    Dependency factory to require specific roles.
    Usage: Depends(require_roles("admin", "power_user"))
    """
    async def role_checker(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if not current_user.has_any_role(list(required_roles)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(required_roles)}",
            )
        return current_user
    
    return role_checker


def require_admin():
    """Dependency to require admin role."""
    return require_roles(RoleEnum.ADMIN.value)


class RBACChecker:
    """
    Callable class for flexible RBAC checks.
    Can be extended for resource-level permissions.
    """
    
    def __init__(self, allowed_roles: List[str], allow_self: bool = False):
        self.allowed_roles = allowed_roles
        self.allow_self = allow_self
    
    async def __call__(
        self,
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if current_user.is_admin:
            return current_user
        
        if not current_user.has_any_role(self.allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        
        return current_user


# Common permission checkers
can_upload_documents = RBACChecker([
    RoleEnum.ADMIN.value,
    RoleEnum.POWER_USER.value,
    RoleEnum.INGESTOR.value,
])

can_manage_users = RBACChecker([RoleEnum.ADMIN.value])

can_view_metrics = RBACChecker([
    RoleEnum.ADMIN.value,
    RoleEnum.POWER_USER.value,
])

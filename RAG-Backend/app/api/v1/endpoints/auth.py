"""
Authentication API endpoints.
POST /auth/signup, /auth/login, /auth/logout, /auth/refresh
"""
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.auth_service import AuthService
from app.schemas.auth import (
    SignupRequest,
    SignupResponse,
    LoginRequest,
    LoginResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
    LogoutRequest,
)
from app.api.deps import get_current_user, CurrentUser

router = APIRouter()


def get_client_info(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Extract client IP and user agent from request."""
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    return ip, user_agent


@router.post("/signup", response_model=SignupResponse)
async def signup(
    request: SignupRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user account.
    
    - Creates user with email/password
    - Assigns default 'viewer' role
    - Returns user ID (login separately to get tokens)
    """
    auth_service = AuthService(db)
    
    user = await auth_service.create_user(
        email=request.email,
        password=request.password,
        name=request.name,
    )
    
    return SignupResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    req: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user and return JWT tokens.
    
    - Validates email/password
    - Returns access token (short-lived) and refresh token (long-lived)
    - Sets secure cookie for refresh token (optional)
    """
    ip, user_agent = get_client_info(req)
    auth_service = AuthService(db)
    
    result = await auth_service.login(
        email=request.email,
        password=request.password,
        ip_address=ip,
        user_agent=user_agent,
    )
    
    # Optionally set refresh token as HTTP-only cookie
    response.set_cookie(
        key="refresh_token",
        value=result["refresh_token"],
        httponly=True,
        secure=True,  # Set to False for local dev without HTTPS
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 7 days
    )
    
    return LoginResponse(**result)


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    req: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """
    Refresh access token using refresh token.
    
    - Validates refresh token
    - Rotates refresh token (old one invalidated)
    - Returns new access and refresh tokens
    """
    ip, user_agent = get_client_info(req)
    auth_service = AuthService(db)
    
    result = await auth_service.refresh_tokens(
        refresh_token=request.refresh_token,
        ip_address=ip,
        user_agent=user_agent,
    )
    
    # Update cookie
    response.set_cookie(
        key="refresh_token",
        value=result["refresh_token"],
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )
    
    return TokenRefreshResponse(**result)


@router.post("/logout")
async def logout(
    request: LogoutRequest,
    response: Response,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Logout user and revoke tokens.
    
    - Revokes the provided refresh token
    - Optionally revokes all tokens (logout from all devices)
    - Clears refresh token cookie
    """
    auth_service = AuthService(db)
    
    if request.logout_all_devices:
        await auth_service.revoke_all_user_tokens(current_user.id)
    elif request.refresh_token:
        await auth_service.revoke_refresh_token(request.refresh_token)
    
    # Clear cookie
    response.delete_cookie(key="refresh_token")
    
    return {"message": "Logged out successfully"}

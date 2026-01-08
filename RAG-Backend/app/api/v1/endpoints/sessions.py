"""
Session Management API endpoints for Phase 7.
Multi-device session management.
"""
import uuid
from typing import Optional, List

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.api.deps import get_current_user, CurrentUser
from app.services.session_manager import get_session_manager
from app.models.session import SessionStatus

router = APIRouter()


# ============================================================================
# SCHEMAS
# ============================================================================

class SessionResponse(BaseModel):
    """Response model for a session."""
    id: str
    device_id: Optional[str]
    device_name: Optional[str]
    device_type: Optional[str]
    ip_address: Optional[str]
    status: str
    is_current: bool
    created_at: str
    last_active_at: str
    expires_at: Optional[str]


class SessionListResponse(BaseModel):
    """Response model for session list."""
    sessions: List[SessionResponse]
    total: int


class RevokeSessionRequest(BaseModel):
    """Request model for revoking a session."""
    session_id: str


class RevokeAllSessionsRequest(BaseModel):
    """Request model for revoking all sessions."""
    except_current: bool = True


class SessionActivityResponse(BaseModel):
    """Response model for session activity."""
    id: str
    activity_type: str
    endpoint: Optional[str]
    method: Optional[str]
    ip_address: Optional[str]
    created_at: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/", response_model=SessionListResponse)
async def list_my_sessions(
    include_expired: bool = Query(False, description="Include expired/revoked sessions"),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all sessions for the current user.
    
    Returns information about all devices/sessions where the user is logged in.
    """
    session_manager = get_session_manager(db)
    sessions = await session_manager.list_user_sessions(
        user_id=current_user.id,
        include_expired=include_expired,
    )
    
    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=str(s.id),
                device_id=s.device_id,
                device_name=s.device_name,
                device_type=s.device_type,
                ip_address=s.ip_address,
                status=s.status.value,
                is_current=s.is_current,
                created_at=s.created_at.isoformat(),
                last_active_at=s.last_active_at.isoformat(),
                expires_at=s.expires_at.isoformat() if s.expires_at else None,
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/current", response_model=SessionResponse)
async def get_current_session(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the current session for the authenticated user.
    """
    session_manager = get_session_manager(db)
    session = await session_manager.get_current_session(user_id=current_user.id)
    
    if not session:
        raise HTTPException(status_code=404, detail="No current session found")
    
    return SessionResponse(
        id=str(session.id),
        device_id=session.device_id,
        device_name=session.device_name,
        device_type=session.device_type,
        ip_address=session.ip_address,
        status=session.status.value,
        is_current=session.is_current,
        created_at=session.created_at.isoformat(),
        last_active_at=session.last_active_at.isoformat(),
        expires_at=session.expires_at.isoformat() if session.expires_at else None,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific session by ID.
    
    Only returns sessions owned by the current user.
    """
    session_manager = get_session_manager(db)
    session = await session_manager.get_session(uuid.UUID(session_id))
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this session")
    
    return SessionResponse(
        id=str(session.id),
        device_id=session.device_id,
        device_name=session.device_name,
        device_type=session.device_type,
        ip_address=session.ip_address,
        status=session.status.value,
        is_current=session.is_current,
        created_at=session.created_at.isoformat(),
        last_active_at=session.last_active_at.isoformat(),
        expires_at=session.expires_at.isoformat() if session.expires_at else None,
    )


@router.delete("/{session_id}")
async def revoke_session(
    session_id: str,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Revoke a specific session (log out that device).
    
    Can revoke any session owned by the current user.
    """
    session_manager = get_session_manager(db)
    session = await session_manager.get_session(uuid.UUID(session_id))
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to revoke this session")
    
    ip_address = request.client.host if request.client else None
    success = await session_manager.revoke_session(
        session_id=uuid.UUID(session_id),
        ip_address=ip_address,
    )
    
    return {
        "status": "revoked" if success else "failed",
        "session_id": session_id,
    }


@router.post("/revoke-all")
async def revoke_all_sessions(
    body: RevokeAllSessionsRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Revoke all sessions for the current user.
    
    Optionally keeps the current session active (default behavior).
    """
    session_manager = get_session_manager(db)
    ip_address = request.client.host if request.client else None
    
    revoked_count = await session_manager.revoke_all_sessions(
        user_id=current_user.id,
        except_current=body.except_current,
        ip_address=ip_address,
    )
    
    return {
        "status": "success",
        "revoked_count": revoked_count,
    }


@router.get("/{session_id}/activities", response_model=List[SessionActivityResponse])
async def get_session_activities(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get recent activities for a session.
    
    Useful for security auditing.
    """
    session_manager = get_session_manager(db)
    session = await session_manager.get_session(uuid.UUID(session_id))
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this session")
    
    activities = await session_manager.get_session_activities(
        session_id=uuid.UUID(session_id),
        limit=limit,
    )
    
    return [
        SessionActivityResponse(
            id=str(a.id),
            activity_type=a.activity_type,
            endpoint=a.endpoint,
            method=a.method,
            ip_address=a.ip_address,
            created_at=a.created_at.isoformat(),
        )
        for a in activities
    ]

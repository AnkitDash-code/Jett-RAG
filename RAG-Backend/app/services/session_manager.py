"""
Session Manager service for Phase 7.
Manages user sessions across multiple devices.
"""
import uuid
import secrets
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from app.config import settings
from app.models.session import UserSession, SessionActivity, SessionStatus

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions and devices.
    
    Features:
    - Multi-device session tracking
    - Session creation and validation
    - Device fingerprinting
    - Session revocation (single or all)
    - Activity logging
    """
    
    def __init__(
        self,
        db: AsyncSession,
        session_duration_days: int = 30,
        max_sessions_per_user: int = 10,
    ):
        self.db = db
        self.session_duration_days = session_duration_days
        self.max_sessions_per_user = max_sessions_per_user
    
    # ========================================================================
    # SESSION CREATION
    # ========================================================================
    
    async def create_session(
        self,
        user_id: uuid.UUID,
        device_info: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        refresh_token_jti: Optional[str] = None,
    ) -> Tuple[UserSession, str]:
        """
        Create a new session for a user.
        
        Args:
            user_id: User to create session for
            device_info: Device metadata (device_id, name, type)
            ip_address: Request IP address
            user_agent: Browser user agent
            refresh_token_jti: JWT ID for the refresh token
            
        Returns:
            Tuple of (UserSession, plain_session_token)
        """
        # Generate session token
        plain_token = secrets.token_urlsafe(32)
        hashed_token = self._hash_token(plain_token)
        
        # Parse device info
        device_info = device_info or {}
        device_id = device_info.get("device_id") or self._generate_device_id(user_agent)
        device_name = device_info.get("device_name") or self._parse_device_name(user_agent)
        device_type = device_info.get("device_type") or self._detect_device_type(user_agent)
        
        # Check for existing session on same device
        existing = await self._find_session_by_device(user_id, device_id)
        if existing:
            # Update existing session instead of creating new
            existing.session_token = hashed_token
            existing.refresh_token_jti = refresh_token_jti
            existing.ip_address = ip_address
            existing.user_agent = user_agent
            existing.status = SessionStatus.ACTIVE
            existing.last_active_at = datetime.utcnow()
            existing.expires_at = datetime.utcnow() + timedelta(days=self.session_duration_days)
            existing.is_current = True
            
            await self._unset_current_flags(user_id, exclude_id=existing.id)
            await self.db.commit()
            
            logger.info(f"Updated existing session {existing.id} for user {user_id}")
            return existing, plain_token
        
        # Enforce max sessions limit
        await self._enforce_session_limit(user_id)
        
        # Create new session
        session = UserSession(
            user_id=user_id,
            session_token=hashed_token,
            refresh_token_jti=refresh_token_jti,
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            user_agent=user_agent,
            ip_address=ip_address,
            status=SessionStatus.ACTIVE,
            is_current=True,
            expires_at=datetime.utcnow() + timedelta(days=self.session_duration_days),
        )
        
        # Unset current flag on other sessions
        await self._unset_current_flags(user_id)
        
        self.db.add(session)
        await self.db.commit()
        
        # Log activity
        await self._log_activity(session.id, "session_created", ip_address, user_agent)
        
        logger.info(f"Created new session {session.id} for user {user_id} on device {device_name}")
        
        return session, plain_token
    
    # ========================================================================
    # SESSION VALIDATION
    # ========================================================================
    
    async def validate_session(
        self,
        session_token: str,
        update_activity: bool = True,
        ip_address: Optional[str] = None,
    ) -> Optional[UserSession]:
        """
        Validate a session token.
        
        Args:
            session_token: Plain session token to validate
            update_activity: Whether to update last_active_at
            ip_address: Current request IP for activity log
            
        Returns:
            UserSession if valid, None otherwise
        """
        hashed_token = self._hash_token(session_token)
        
        stmt = select(UserSession).where(
            UserSession.session_token == hashed_token,
            UserSession.status == SessionStatus.ACTIVE,
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            return None
        
        # Check expiration
        if session.is_expired:
            session.status = SessionStatus.EXPIRED
            await self.db.commit()
            return None
        
        # Update activity
        if update_activity:
            session.touch()
            await self.db.commit()
        
        return session
    
    async def validate_refresh_token(
        self,
        refresh_token_jti: str,
    ) -> Optional[UserSession]:
        """
        Validate a session by refresh token JTI.
        
        Args:
            refresh_token_jti: JWT ID from refresh token
            
        Returns:
            UserSession if valid, None otherwise
        """
        stmt = select(UserSession).where(
            UserSession.refresh_token_jti == refresh_token_jti,
            UserSession.status == SessionStatus.ACTIVE,
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session or session.is_expired:
            return None
        
        return session
    
    # ========================================================================
    # SESSION LISTING
    # ========================================================================
    
    async def list_user_sessions(
        self,
        user_id: uuid.UUID,
        include_expired: bool = False,
    ) -> List[UserSession]:
        """
        List all sessions for a user.
        
        Args:
            user_id: User to list sessions for
            include_expired: Whether to include expired/revoked sessions
            
        Returns:
            List of UserSession objects
        """
        stmt = select(UserSession).where(UserSession.user_id == user_id)
        
        if not include_expired:
            stmt = stmt.where(UserSession.status == SessionStatus.ACTIVE)
        
        stmt = stmt.order_by(UserSession.last_active_at.desc())
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_session(
        self,
        session_id: uuid.UUID,
    ) -> Optional[UserSession]:
        """Get a session by ID."""
        stmt = select(UserSession).where(UserSession.id == session_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_current_session(
        self,
        user_id: uuid.UUID,
    ) -> Optional[UserSession]:
        """Get the current (most recently used) session for a user."""
        stmt = select(UserSession).where(
            UserSession.user_id == user_id,
            UserSession.is_current == True,
            UserSession.status == SessionStatus.ACTIVE,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    # ========================================================================
    # SESSION REVOCATION
    # ========================================================================
    
    async def revoke_session(
        self,
        session_id: uuid.UUID,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Revoke a single session.
        
        Args:
            session_id: Session to revoke
            ip_address: IP address of the revoking request
            
        Returns:
            True if revoked, False if not found
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.revoke()
        await self._log_activity(session.id, "session_revoked", ip_address)
        await self.db.commit()
        
        logger.info(f"Revoked session {session_id}")
        return True
    
    async def revoke_all_sessions(
        self,
        user_id: uuid.UUID,
        except_current: bool = True,
        ip_address: Optional[str] = None,
    ) -> int:
        """
        Revoke all sessions for a user.
        
        Args:
            user_id: User to revoke sessions for
            except_current: Keep the current session active
            ip_address: IP address of the revoking request
            
        Returns:
            Number of sessions revoked
        """
        sessions = await self.list_user_sessions(user_id, include_expired=False)
        
        revoked = 0
        for session in sessions:
            if except_current and session.is_current:
                continue
            
            session.revoke()
            await self._log_activity(session.id, "session_revoked_bulk", ip_address)
            revoked += 1
        
        await self.db.commit()
        logger.info(f"Revoked {revoked} sessions for user {user_id}")
        
        return revoked
    
    async def revoke_by_refresh_token(
        self,
        refresh_token_jti: str,
    ) -> bool:
        """Revoke session by refresh token JTI."""
        session = await self.validate_refresh_token(refresh_token_jti)
        if not session:
            return False
        
        session.revoke()
        await self.db.commit()
        return True
    
    # ========================================================================
    # ACTIVITY LOGGING
    # ========================================================================
    
    async def log_api_call(
        self,
        session_id: uuid.UUID,
        endpoint: str,
        method: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an API call for a session."""
        activity = SessionActivity(
            session_id=session_id,
            activity_type="api_call",
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata_json=json.dumps(metadata or {}),
        )
        self.db.add(activity)
        # Don't commit - let the caller handle transaction
    
    async def get_session_activities(
        self,
        session_id: uuid.UUID,
        limit: int = 100,
    ) -> List[SessionActivity]:
        """Get recent activities for a session."""
        stmt = select(SessionActivity).where(
            SessionActivity.session_id == session_id
        ).order_by(
            SessionActivity.created_at.desc()
        ).limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _hash_token(self, token: str) -> str:
        """Hash a session token."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _generate_device_id(self, user_agent: Optional[str]) -> str:
        """Generate a device ID from user agent."""
        if user_agent:
            return hashlib.md5(user_agent.encode()).hexdigest()[:16]
        return secrets.token_hex(8)
    
    def _parse_device_name(self, user_agent: Optional[str]) -> str:
        """Parse a human-readable device name from user agent."""
        if not user_agent:
            return "Unknown Device"
        
        ua_lower = user_agent.lower()
        
        # Detect browser
        browser = "Unknown Browser"
        if "chrome" in ua_lower and "edg" not in ua_lower:
            browser = "Chrome"
        elif "firefox" in ua_lower:
            browser = "Firefox"
        elif "safari" in ua_lower and "chrome" not in ua_lower:
            browser = "Safari"
        elif "edg" in ua_lower:
            browser = "Edge"
        
        # Detect OS
        os_name = "Unknown OS"
        if "windows" in ua_lower:
            os_name = "Windows"
        elif "mac" in ua_lower:
            os_name = "macOS"
        elif "linux" in ua_lower:
            os_name = "Linux"
        elif "android" in ua_lower:
            os_name = "Android"
        elif "iphone" in ua_lower or "ipad" in ua_lower:
            os_name = "iOS"
        
        return f"{browser} on {os_name}"
    
    def _detect_device_type(self, user_agent: Optional[str]) -> str:
        """Detect device type from user agent."""
        if not user_agent:
            return "unknown"
        
        ua_lower = user_agent.lower()
        
        if "mobile" in ua_lower or "android" in ua_lower:
            if "tablet" in ua_lower or "ipad" in ua_lower:
                return "tablet"
            return "mobile"
        elif "tablet" in ua_lower or "ipad" in ua_lower:
            return "tablet"
        else:
            return "desktop"
    
    async def _find_session_by_device(
        self,
        user_id: uuid.UUID,
        device_id: str,
    ) -> Optional[UserSession]:
        """Find an existing session for a device."""
        stmt = select(UserSession).where(
            UserSession.user_id == user_id,
            UserSession.device_id == device_id,
            UserSession.status == SessionStatus.ACTIVE,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _unset_current_flags(
        self,
        user_id: uuid.UUID,
        exclude_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Unset is_current flag on all sessions except one."""
        stmt = select(UserSession).where(
            UserSession.user_id == user_id,
            UserSession.is_current == True,
        )
        if exclude_id:
            stmt = stmt.where(UserSession.id != exclude_id)
        
        result = await self.db.execute(stmt)
        for session in result.scalars().all():
            session.is_current = False
    
    async def _enforce_session_limit(self, user_id: uuid.UUID) -> None:
        """Revoke oldest sessions if limit exceeded."""
        sessions = await self.list_user_sessions(user_id, include_expired=False)
        
        if len(sessions) >= self.max_sessions_per_user:
            # Sort by last_active_at, oldest first
            sessions.sort(key=lambda s: s.last_active_at)
            
            # Revoke oldest until we're under limit
            to_revoke = len(sessions) - self.max_sessions_per_user + 1
            for session in sessions[:to_revoke]:
                session.revoke()
                logger.info(f"Auto-revoked old session {session.id} for limit enforcement")
    
    async def _log_activity(
        self,
        session_id: uuid.UUID,
        activity_type: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Log a session activity."""
        activity = SessionActivity(
            session_id=session_id,
            activity_type=activity_type,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.db.add(activity)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_session_manager(db: AsyncSession) -> SessionManager:
    """Factory function to create SessionManager instance."""
    return SessionManager(db)

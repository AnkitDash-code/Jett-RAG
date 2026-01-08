"""
Audit Service for Phase 7.
Centralized audit logging with async support.
"""
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, func

from app.config import settings
from app.models.audit import AuditLog, AuditAction, AuditSeverity

logger = logging.getLogger(__name__)

# Context variable for request-scoped audit context
_audit_context: ContextVar[Dict[str, Any]] = ContextVar("audit_context", default={})


def set_audit_context(
    user_id: Optional[uuid.UUID] = None,
    user_email: Optional[str] = None,
    session_id: Optional[uuid.UUID] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    request_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> None:
    """
    Set request-scoped audit context.
    
    Called early in request lifecycle (middleware) to provide
    context for all audit events within the request.
    """
    ctx = {
        "user_id": user_id,
        "user_email": user_email,
        "session_id": session_id,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "request_id": request_id,
        "tenant_id": tenant_id,
    }
    _audit_context.set(ctx)


def get_audit_context() -> Dict[str, Any]:
    """Get the current audit context."""
    return _audit_context.get()


def clear_audit_context() -> None:
    """Clear the audit context."""
    _audit_context.set({})


class AuditService:
    """
    Service for creating and querying audit logs.
    
    Features:
    - Async audit event creation
    - Context-aware logging (uses request context)
    - Query and aggregation support
    - Retention policy enforcement
    """
    
    def __init__(
        self,
        db: AsyncSession,
        retention_days: int = 365,
    ):
        self.db = db
        self.retention_days = retention_days
    
    # ========================================================================
    # LOG CREATION
    # ========================================================================
    
    async def log(
        self,
        action: AuditAction,
        description: str = "",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        success: bool = True,
        error_message: Optional[str] = None,
        # Override context if needed
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
    ) -> AuditLog:
        """
        Create an audit log entry.
        
        Automatically uses request context for user/session/IP info.
        
        Args:
            action: The auditable action being logged
            description: Human-readable description
            resource_type: Type of affected resource
            resource_id: ID of affected resource
            resource_name: Name of affected resource
            old_value: Previous state (for updates)
            new_value: New state (for updates)
            metadata: Additional context data
            severity: Log severity level
            success: Whether the action succeeded
            error_message: Error details if success=False
            user_id: Override context user_id
            session_id: Override context session_id
            endpoint: API endpoint
            method: HTTP method
            
        Returns:
            The created AuditLog entry
        """
        # Get context
        ctx = get_audit_context()
        
        # Create log entry
        log_entry = AuditLog(
            action=action,
            severity=severity,
            user_id=user_id or ctx.get("user_id"),
            user_email=ctx.get("user_email"),
            session_id=session_id or ctx.get("session_id"),
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            ip_address=ctx.get("ip_address"),
            user_agent=ctx.get("user_agent"),
            request_id=ctx.get("request_id"),
            endpoint=endpoint,
            method=method,
            tenant_id=ctx.get("tenant_id"),
            description=description,
            old_value=json.dumps(old_value) if old_value else None,
            new_value=json.dumps(new_value) if new_value else None,
            metadata_json=json.dumps(metadata or {}),
            success=success,
            error_message=error_message,
            expires_at=datetime.utcnow() + timedelta(days=self.retention_days),
        )
        
        self.db.add(log_entry)
        # Don't commit - let caller manage transaction
        
        # Also log to standard logger for immediate visibility
        log_level = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.INFO)
        
        logger.log(
            log_level,
            f"[AUDIT] {action.value}: {description} "
            f"(user={log_entry.user_id}, resource={resource_type}:{resource_id})"
        )
        
        return log_entry
    
    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================
    
    async def log_login(
        self,
        user_id: uuid.UUID,
        user_email: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditLog:
        """Log a login attempt."""
        return await self.log(
            action=AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED,
            description=f"Login {'succeeded' if success else 'failed'} for {user_email}",
            resource_type="user",
            resource_id=str(user_id),
            resource_name=user_email,
            user_id=user_id,
            success=success,
            error_message=error_message,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
        )
    
    async def log_logout(self, user_id: uuid.UUID) -> AuditLog:
        """Log a logout."""
        return await self.log(
            action=AuditAction.LOGOUT,
            description="User logged out",
            user_id=user_id,
        )
    
    async def log_document_upload(
        self,
        document_id: uuid.UUID,
        document_name: str,
        user_id: Optional[uuid.UUID] = None,
    ) -> AuditLog:
        """Log a document upload."""
        return await self.log(
            action=AuditAction.DOCUMENT_UPLOAD,
            description=f"Document uploaded: {document_name}",
            resource_type="document",
            resource_id=str(document_id),
            resource_name=document_name,
            user_id=user_id,
        )
    
    async def log_document_delete(
        self,
        document_id: uuid.UUID,
        document_name: str,
        user_id: Optional[uuid.UUID] = None,
    ) -> AuditLog:
        """Log a document deletion."""
        return await self.log(
            action=AuditAction.DOCUMENT_DELETE,
            description=f"Document deleted: {document_name}",
            resource_type="document",
            resource_id=str(document_id),
            resource_name=document_name,
            user_id=user_id,
            severity=AuditSeverity.WARNING,
        )
    
    async def log_permission_change(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        old_permissions: Optional[Dict[str, Any]] = None,
        new_permissions: Optional[Dict[str, Any]] = None,
        user_id: Optional[uuid.UUID] = None,
    ) -> AuditLog:
        """Log a permission change."""
        return await self.log(
            action=action,
            description=f"Permission changed on {resource_type} {resource_id}",
            resource_type=resource_type,
            resource_id=resource_id,
            old_value=old_permissions,
            new_value=new_permissions,
            user_id=user_id,
            severity=AuditSeverity.WARNING,
        )
    
    async def log_error(
        self,
        error_message: str,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log an error."""
        return await self.log(
            action=AuditAction.ERROR,
            description=error_message,
            endpoint=endpoint,
            metadata=metadata,
            severity=AuditSeverity.ERROR,
            success=False,
            error_message=error_message,
        )
    
    async def log_security_alert(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """Log a security alert."""
        return await self.log(
            action=AuditAction.SECURITY_ALERT,
            description=description,
            metadata=metadata,
            severity=AuditSeverity.CRITICAL,
        )
    
    # ========================================================================
    # QUERIES
    # ========================================================================
    
    async def get_logs(
        self,
        user_id: Optional[uuid.UUID] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """
        Query audit logs with filters.
        
        Args:
            user_id: Filter by user
            action: Filter by action type
            resource_type: Filter by resource type
            severity: Filter by severity
            start_date: Filter by date range start
            end_date: Filter by date range end
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of matching AuditLog entries
        """
        stmt = select(AuditLog)
        
        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        if resource_type:
            stmt = stmt.where(AuditLog.resource_type == resource_type)
        if severity:
            stmt = stmt.where(AuditLog.severity == severity)
        if start_date:
            stmt = stmt.where(AuditLog.created_at >= start_date)
        if end_date:
            stmt = stmt.where(AuditLog.created_at <= end_date)
        
        stmt = stmt.order_by(AuditLog.created_at.desc())
        stmt = stmt.offset(offset).limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_log_count(
        self,
        user_id: Optional[uuid.UUID] = None,
        action: Optional[AuditAction] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Get count of audit logs matching filters."""
        stmt = select(func.count(AuditLog.id))
        
        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if action:
            stmt = stmt.where(AuditLog.action == action)
        if start_date:
            stmt = stmt.where(AuditLog.created_at >= start_date)
        if end_date:
            stmt = stmt.where(AuditLog.created_at <= end_date)
        
        result = await self.db.execute(stmt)
        return result.scalar() or 0
    
    async def get_action_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get count of logs by action type."""
        stmt = select(
            AuditLog.action,
            func.count(AuditLog.id).label("count"),
        ).group_by(AuditLog.action)
        
        if start_date:
            stmt = stmt.where(AuditLog.created_at >= start_date)
        if end_date:
            stmt = stmt.where(AuditLog.created_at <= end_date)
        if tenant_id:
            stmt = stmt.where(AuditLog.tenant_id == tenant_id)
        
        result = await self.db.execute(stmt)
        return {row.action.value: row.count for row in result.all()}
    
    # ========================================================================
    # RETENTION
    # ========================================================================
    
    async def cleanup_expired_logs(self) -> int:
        """
        Delete audit logs past their retention period.
        
        Called by maintenance job.
        """
        from sqlmodel import delete
        
        stmt = delete(AuditLog).where(
            AuditLog.expires_at < datetime.utcnow()
        )
        
        result = await self.db.execute(stmt)
        await self.db.commit()
        
        deleted_count = result.rowcount or 0
        logger.info(f"Cleaned up {deleted_count} expired audit logs")
        
        return deleted_count


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_audit_service(db: AsyncSession) -> AuditService:
    """Factory function to create AuditService instance."""
    return AuditService(db)

"""
Audit logging models for Phase 7.
Comprehensive audit trail for security and compliance.
"""
import uuid
from datetime import datetime
from typing import Optional
from enum import Enum

from sqlmodel import SQLModel, Field


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    TOKEN_REFRESH = "token_refresh"
    
    # User Management
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    USER_DEACTIVATE = "user_deactivate"
    USER_ACTIVATE = "user_activate"
    ROLE_ASSIGN = "role_assign"
    ROLE_REVOKE = "role_revoke"
    
    # Document Operations
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_DOWNLOAD = "document_download"
    DOCUMENT_SHARE = "document_share"
    DOCUMENT_UNSHARE = "document_unshare"
    
    # Chat Operations
    CONVERSATION_CREATE = "conversation_create"
    CONVERSATION_DELETE = "conversation_delete"
    MESSAGE_SEND = "message_send"
    
    # Admin Operations
    CONFIG_CHANGE = "config_change"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    CACHE_CLEAR = "cache_clear"
    REINDEX = "reindex"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR = "error"
    SECURITY_ALERT = "security_alert"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLog(SQLModel, table=True):
    """
    Audit log entry for tracking all significant system events.
    
    Designed for:
    - Security auditing
    - Compliance (SOC2, HIPAA, GDPR)
    - Forensic analysis
    - User activity tracking
    """
    __tablename__ = "audit_logs"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Event identification
    action: AuditAction = Field(index=True)
    severity: AuditSeverity = Field(default=AuditSeverity.INFO, index=True)
    
    # Actor (who performed the action)
    user_id: Optional[uuid.UUID] = Field(default=None, index=True)
    user_email: Optional[str] = Field(default=None)  # Denormalized for easier querying
    session_id: Optional[uuid.UUID] = Field(default=None, index=True)
    
    # Target (what was affected)
    resource_type: Optional[str] = Field(default=None, index=True)  # "user", "document", "conversation"
    resource_id: Optional[str] = Field(default=None, index=True)
    resource_name: Optional[str] = Field(default=None)  # Human-readable name
    
    # Request context
    ip_address: Optional[str] = Field(default=None, index=True)
    user_agent: Optional[str] = Field(default=None)
    request_id: Optional[str] = Field(default=None, index=True)  # Correlation ID
    endpoint: Optional[str] = Field(default=None)
    method: Optional[str] = Field(default=None)
    
    # Multi-tenant
    tenant_id: Optional[str] = Field(default=None, index=True)
    
    # Event details
    description: str = Field(default="")
    old_value: Optional[str] = Field(default=None)  # JSON string of old state
    new_value: Optional[str] = Field(default=None)  # JSON string of new state
    metadata_json: str = Field(default="{}")  # Additional context
    
    # Status
    success: bool = Field(default=True, index=True)
    error_message: Optional[str] = Field(default=None)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # Retention
    expires_at: Optional[datetime] = Field(default=None, index=True)  # For auto-cleanup


class AuditQuery(SQLModel, table=True):
    """
    Cached audit query results for faster reporting.
    
    Pre-aggregated statistics for dashboard widgets.
    """
    __tablename__ = "audit_queries"
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # Query identification
    query_name: str = Field(index=True)  # "daily_logins", "document_uploads", etc.
    period_start: datetime = Field(index=True)
    period_end: datetime = Field(index=True)
    
    # Filters applied
    tenant_id: Optional[str] = Field(default=None)
    user_id: Optional[uuid.UUID] = Field(default=None)
    
    # Results
    result_json: str = Field(default="{}")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(index=True)

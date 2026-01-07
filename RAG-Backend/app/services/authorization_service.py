"""
Authorization service for dual RBAC.
Implements document-level and graph-level access control.
"""
import logging
from typing import Optional, List, Set, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.models.user import User
from app.models.document import Document

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Available permissions."""
    # Document permissions
    DOC_VIEW = "document:view"
    DOC_EDIT = "document:edit"
    DOC_DELETE = "document:delete"
    DOC_SHARE = "document:share"
    
    # Chat permissions
    CHAT_USE = "chat:use"
    CHAT_HISTORY = "chat:view_history"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_DOCUMENTS = "admin:documents"
    ADMIN_ANALYTICS = "admin:analytics"
    ADMIN_SETTINGS = "admin:settings"
    
    # Graph permissions (for knowledge graph)
    GRAPH_VIEW = "graph:view"
    GRAPH_EDIT = "graph:edit"


class AccessLevel(str, Enum):
    """Document access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"
    RESTRICTED = "restricted"


@dataclass
class RoleDefinition:
    """Definition of a role and its permissions."""
    name: str
    permissions: Set[Permission]
    access_levels: Set[AccessLevel]
    description: str = ""


# Role definitions
ROLES: Dict[str, RoleDefinition] = {
    "admin": RoleDefinition(
        name="admin",
        permissions=set(Permission),  # All permissions
        access_levels=set(AccessLevel),  # All access levels
        description="Full system access"
    ),
    "user": RoleDefinition(
        name="user",
        permissions={
            Permission.DOC_VIEW,
            Permission.DOC_EDIT,
            Permission.DOC_DELETE,
            Permission.CHAT_USE,
            Permission.CHAT_HISTORY,
            Permission.GRAPH_VIEW,
        },
        access_levels={AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.PRIVATE},
        description="Standard user with document and chat access"
    ),
    "viewer": RoleDefinition(
        name="viewer",
        permissions={
            Permission.DOC_VIEW,
            Permission.CHAT_USE,
            Permission.GRAPH_VIEW,
        },
        access_levels={AccessLevel.PUBLIC, AccessLevel.INTERNAL},
        description="Read-only access"
    ),
    "guest": RoleDefinition(
        name="guest",
        permissions={
            Permission.DOC_VIEW,
            Permission.CHAT_USE,
        },
        access_levels={AccessLevel.PUBLIC},
        description="Limited public access"
    ),
}


@dataclass
class AuthorizationContext:
    """Context for authorization decisions."""
    user_id: uuid.UUID
    role: str
    tenant_id: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    access_levels: Set[AccessLevel] = field(default_factory=set)
    
    # Graph-level access (document IDs user can access via graph)
    graph_accessible_docs: Set[uuid.UUID] = field(default_factory=set)


class AuthorizationService:
    """
    Service for authorization decisions.
    Implements dual RBAC: document-level + graph-level.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._graph_cache: Dict[uuid.UUID, Set[uuid.UUID]] = {}
    
    async def get_context(self, user: User) -> AuthorizationContext:
        """
        Build authorization context for a user.
        """
        # Determine role
        role = "admin" if user.is_admin else "user"
        role_def = ROLES.get(role, ROLES["guest"])
        
        # Build context
        context = AuthorizationContext(
            user_id=user.id,
            role=role,
            tenant_id=user.tenant_id if hasattr(user, "tenant_id") else None,
            permissions=role_def.permissions.copy(),
            access_levels=role_def.access_levels.copy(),
        )
        
        # Build graph-level access (docs the user owns or has been shared)
        context.graph_accessible_docs = await self._get_graph_accessible_docs(user.id)
        
        return context
    
    async def _get_graph_accessible_docs(self, user_id: uuid.UUID) -> Set[uuid.UUID]:
        """
        Get document IDs accessible to user via graph relationships.
        This includes:
        - Documents owned by user
        - Documents shared with user (via sharing relationship)
        - Documents in user's tenant
        """
        # Check cache
        if user_id in self._graph_cache:
            return self._graph_cache[user_id]
        
        accessible = set()
        
        # Get owned documents
        result = await self.db.execute(
            select(Document.id).where(Document.owner_id == user_id)
        )
        for doc_id in result.scalars().all():
            accessible.add(doc_id)
        
        # TODO: Add shared documents when sharing is implemented
        # TODO: Add tenant-based access
        
        # Cache for this session
        self._graph_cache[user_id] = accessible
        
        return accessible
    
    def has_permission(
        self,
        context: AuthorizationContext,
        permission: Permission,
    ) -> bool:
        """Check if context has a specific permission."""
        return permission in context.permissions
    
    def can_access_document(
        self,
        context: AuthorizationContext,
        document: Document,
    ) -> bool:
        """
        Check if user can access a document.
        Implements dual RBAC:
        1. Document-level: Check access_level against user's allowed levels
        2. Graph-level: Check if document is in user's accessible graph
        """
        # Admin can access everything
        if context.role == "admin":
            return True
        
        # Check ownership
        if document.owner_id == context.user_id:
            return True
        
        # Check graph-level access
        if document.id in context.graph_accessible_docs:
            return True
        
        # Check document-level access
        doc_access = AccessLevel(document.access_level) if document.access_level else AccessLevel.PRIVATE
        if doc_access in context.access_levels:
            return True
        
        return False
    
    def get_accessible_access_levels(
        self,
        context: AuthorizationContext,
    ) -> List[str]:
        """Get list of access levels the user can query."""
        return [level.value for level in context.access_levels]
    
    def can_manage_users(self, context: AuthorizationContext) -> bool:
        """Check if user can manage other users."""
        return Permission.ADMIN_USERS in context.permissions
    
    def can_view_analytics(self, context: AuthorizationContext) -> bool:
        """Check if user can view analytics."""
        return Permission.ADMIN_ANALYTICS in context.permissions
    
    def invalidate_cache(self, user_id: Optional[uuid.UUID] = None):
        """Invalidate graph access cache."""
        if user_id:
            self._graph_cache.pop(user_id, None)
        else:
            self._graph_cache.clear()


# Factory function
def get_authorization_service(db: AsyncSession) -> AuthorizationService:
    """Get authorization service instance."""
    return AuthorizationService(db)

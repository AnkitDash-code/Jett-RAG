"""
Permission Inheritance Service for Graph RBAC.

Implements cascading permission resolution:
  Tenant → Group → Folder → Document → Chunk → Entity

Permissions flow from parent to child with most permissive matching.
"""
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.graph import (
    Group, GroupMembership, FolderPermission,
    PermissionType, GroupRole
)
from app.models.document import Document, Chunk

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of resources that can have permissions."""
    TENANT = "tenant"
    FOLDER = "folder"
    DOCUMENT = "document"
    CHUNK = "chunk"
    ENTITY = "entity"


@dataclass
class PermissionGrant:
    """Represents a single permission grant."""
    permission_type: PermissionType
    resource_type: ResourceType
    resource_id: str
    grantee_type: str  # "user" or "group"
    grantee_id: uuid.UUID
    inherited_from: Optional[str] = None
    is_inherited: bool = False
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "permission_type": self.permission_type.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "grantee_type": self.grantee_type,
            "grantee_id": str(self.grantee_id),
            "inherited_from": self.inherited_from,
            "is_inherited": self.is_inherited,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class EffectivePermissions:
    """Resolved permissions for a user on a resource."""
    user_id: uuid.UUID
    resource_type: ResourceType
    resource_id: str
    permissions: Set[PermissionType] = field(default_factory=set)
    grants: List[PermissionGrant] = field(default_factory=list)
    is_owner: bool = False
    resolution_path: List[str] = field(default_factory=list)
    
    @property
    def can_view(self) -> bool:
        return PermissionType.VIEW in self.permissions or self.is_owner
    
    @property
    def can_edit(self) -> bool:
        return PermissionType.EDIT in self.permissions or self.is_owner
    
    @property
    def can_admin(self) -> bool:
        return PermissionType.ADMIN in self.permissions or self.is_owner
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": str(self.user_id),
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "permissions": [p.value for p in self.permissions],
            "grants": [g.to_dict() for g in self.grants],
            "is_owner": self.is_owner,
            "can_view": self.can_view,
            "can_edit": self.can_edit,
            "can_admin": self.can_admin,
            "resolution_path": self.resolution_path,
        }


@dataclass
class PermissionTree:
    """Tree structure showing all permissions on a resource."""
    resource_type: ResourceType
    resource_id: str
    direct_grants: List[PermissionGrant]
    inherited_grants: List[PermissionGrant]
    child_grants: Dict[str, List[PermissionGrant]]  # child_id -> grants
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "direct_grants": [g.to_dict() for g in self.direct_grants],
            "inherited_grants": [g.to_dict() for g in self.inherited_grants],
            "child_grants": {
                k: [g.to_dict() for g in v]
                for k, v in self.child_grants.items()
            },
        }


class PermissionInheritanceService:
    """
    Service for resolving cascading permissions.
    
    Permission hierarchy:
    1. Tenant-level permissions (highest)
    2. Group-level permissions
    3. Folder-level permissions (cascade to documents)
    4. Document-level permissions (cascade to chunks)
    5. Entity-level permissions (from chunk access)
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._cache: Dict[str, EffectivePermissions] = {}
    
    async def resolve_effective_permissions(
        self,
        user_id: uuid.UUID,
        resource_type: ResourceType,
        resource_id: str,
        tenant_id: Optional[str] = None,
    ) -> EffectivePermissions:
        """
        Resolve all effective permissions for a user on a resource.
        
        Checks:
        1. Direct user permissions
        2. Group membership permissions
        3. Inherited folder permissions (for documents)
        4. Ownership
        
        Returns the most permissive combination of all applicable permissions.
        """
        if not settings.ENABLE_GROUP_PERMISSIONS:
            # If group permissions disabled, just check ownership
            return await self._resolve_ownership_only(
                user_id, resource_type, resource_id
            )
        
        cache_key = f"{user_id}:{resource_type.value}:{resource_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = EffectivePermissions(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        
        # Check ownership first
        is_owner = await self._check_ownership(user_id, resource_type, resource_id)
        if is_owner:
            result.is_owner = True
            result.permissions = {
                PermissionType.VIEW,
                PermissionType.EDIT,
                PermissionType.ADMIN
            }
            result.resolution_path.append(f"owner:{resource_type.value}:{resource_id}")
            self._cache[cache_key] = result
            return result
        
        # Get user's group memberships
        user_groups = await self._get_user_groups(user_id)
        
        # Resolve based on resource type
        if resource_type == ResourceType.FOLDER:
            await self._resolve_folder_permissions(result, user_id, resource_id, user_groups)
        elif resource_type == ResourceType.DOCUMENT:
            await self._resolve_document_permissions(result, user_id, resource_id, user_groups)
        elif resource_type == ResourceType.CHUNK:
            await self._resolve_chunk_permissions(result, user_id, resource_id, user_groups)
        elif resource_type == ResourceType.ENTITY:
            await self._resolve_entity_permissions(result, user_id, resource_id, user_groups)
        
        self._cache[cache_key] = result
        return result
    
    async def _resolve_ownership_only(
        self,
        user_id: uuid.UUID,
        resource_type: ResourceType,
        resource_id: str,
    ) -> EffectivePermissions:
        """Simple ownership-based permission check."""
        result = EffectivePermissions(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        
        is_owner = await self._check_ownership(user_id, resource_type, resource_id)
        if is_owner:
            result.is_owner = True
            result.permissions = {
                PermissionType.VIEW,
                PermissionType.EDIT,
                PermissionType.ADMIN
            }
        else:
            # Default: view only for non-owners
            result.permissions = {PermissionType.VIEW}
        
        return result
    
    async def _check_ownership(
        self,
        user_id: uuid.UUID,
        resource_type: ResourceType,
        resource_id: str,
    ) -> bool:
        """Check if user owns the resource."""
        if resource_type == ResourceType.DOCUMENT:
            try:
                doc_id = uuid.UUID(resource_id)
                result = await self.db.execute(
                    select(Document).where(
                        and_(Document.id == doc_id, Document.owner_id == user_id)
                    )
                )
                return result.scalar_one_or_none() is not None
            except ValueError:
                return False
        
        # Add ownership checks for other resource types as needed
        return False
    
    async def _get_user_groups(self, user_id: uuid.UUID) -> List[uuid.UUID]:
        """Get all groups the user belongs to."""
        result = await self.db.execute(
            select(GroupMembership.group_id).where(
                GroupMembership.user_id == user_id
            )
        )
        return list(result.scalars().all())
    
    async def _resolve_folder_permissions(
        self,
        result: EffectivePermissions,
        user_id: uuid.UUID,
        folder_path: str,
        user_groups: List[uuid.UUID],
    ) -> None:
        """Resolve permissions for a folder."""
        # Get direct user permissions on this folder
        user_perms = await self.db.execute(
            select(FolderPermission).where(
                and_(
                    FolderPermission.folder_path == folder_path,
                    FolderPermission.user_id == user_id,
                )
            )
        )
        for perm in user_perms.scalars():
            if perm.expires_at is None or perm.expires_at > datetime.utcnow():
                result.permissions.add(perm.permission_type)
                result.grants.append(self._perm_to_grant(perm, "user", user_id))
                result.resolution_path.append(f"direct:folder:{folder_path}")
        
        # Get group permissions on this folder
        if user_groups:
            group_perms = await self.db.execute(
                select(FolderPermission).where(
                    and_(
                        FolderPermission.folder_path == folder_path,
                        FolderPermission.group_id.in_(user_groups),
                    )
                )
            )
            for perm in group_perms.scalars():
                if perm.expires_at is None or perm.expires_at > datetime.utcnow():
                    result.permissions.add(perm.permission_type)
                    result.grants.append(self._perm_to_grant(perm, "group", perm.group_id))
                    result.resolution_path.append(f"group:{perm.group_id}:folder:{folder_path}")
        
        # Check parent folders (inheritance)
        parent_path = self._get_parent_path(folder_path)
        while parent_path:
            inherited_perms = await self.db.execute(
                select(FolderPermission).where(
                    and_(
                        FolderPermission.folder_path == parent_path,
                        or_(
                            FolderPermission.user_id == user_id,
                            FolderPermission.group_id.in_(user_groups) if user_groups else False,
                        ),
                    )
                )
            )
            for perm in inherited_perms.scalars():
                if perm.expires_at is None or perm.expires_at > datetime.utcnow():
                    result.permissions.add(perm.permission_type)
                    grantee_type = "user" if perm.user_id else "group"
                    grantee_id = perm.user_id or perm.group_id
                    grant = self._perm_to_grant(perm, grantee_type, grantee_id)
                    grant.inherited_from = parent_path
                    grant.is_inherited = True
                    result.grants.append(grant)
                    result.resolution_path.append(f"inherited:{parent_path}")
            
            parent_path = self._get_parent_path(parent_path)
    
    async def _resolve_document_permissions(
        self,
        result: EffectivePermissions,
        user_id: uuid.UUID,
        document_id: str,
        user_groups: List[uuid.UUID],
    ) -> None:
        """Resolve permissions for a document."""
        try:
            doc_uuid = uuid.UUID(document_id)
        except ValueError:
            return
        
        # Get document to find its folder path
        doc_result = await self.db.execute(
            select(Document).where(Document.id == doc_uuid)
        )
        document = doc_result.scalar_one_or_none()
        
        if not document:
            return
        
        # Infer folder path from file_path
        folder_path = self._get_folder_from_file_path(document.file_path)
        
        if folder_path:
            # Inherit folder permissions
            await self._resolve_folder_permissions(result, user_id, folder_path, user_groups)
            
            # Mark all grants as inherited from folder
            for grant in result.grants:
                if not grant.is_inherited:
                    grant.inherited_from = folder_path
                    grant.is_inherited = True
        
        # Check document-level access_level
        if document.access_level == "public":
            result.permissions.add(PermissionType.VIEW)
            result.resolution_path.append(f"access_level:public")
    
    async def _resolve_chunk_permissions(
        self,
        result: EffectivePermissions,
        user_id: uuid.UUID,
        chunk_id: str,
        user_groups: List[uuid.UUID],
    ) -> None:
        """Resolve permissions for a chunk (inherits from document)."""
        try:
            chunk_uuid = uuid.UUID(chunk_id)
        except ValueError:
            return
        
        # Get chunk to find its document
        chunk_result = await self.db.execute(
            select(Chunk).where(Chunk.id == chunk_uuid)
        )
        chunk = chunk_result.scalar_one_or_none()
        
        if not chunk:
            return
        
        # Inherit document permissions
        await self._resolve_document_permissions(
            result, user_id, str(chunk.document_id), user_groups
        )
        
        # Mark resolution as coming from document
        result.resolution_path.append(f"inherited:document:{chunk.document_id}")
    
    async def _resolve_entity_permissions(
        self,
        result: EffectivePermissions,
        user_id: uuid.UUID,
        entity_id: str,
        user_groups: List[uuid.UUID],
    ) -> None:
        """Resolve permissions for an entity (based on chunk access)."""
        from app.models.graph import EntityMention
        
        try:
            entity_uuid = uuid.UUID(entity_id)
        except ValueError:
            return
        
        # Get chunks where this entity is mentioned
        mentions_result = await self.db.execute(
            select(EntityMention.chunk_id).where(
                EntityMention.entity_id == entity_uuid
            ).limit(10)  # Limit for performance
        )
        chunk_ids = list(mentions_result.scalars().all())
        
        # User can access entity if they can access ANY chunk mentioning it
        for chunk_id in chunk_ids:
            chunk_result = EffectivePermissions(
                user_id=user_id,
                resource_type=ResourceType.CHUNK,
                resource_id=str(chunk_id),
            )
            await self._resolve_chunk_permissions(
                chunk_result, user_id, str(chunk_id), user_groups
            )
            
            if chunk_result.can_view:
                result.permissions.add(PermissionType.VIEW)
                result.resolution_path.append(f"via_chunk:{chunk_id}")
                break
    
    def _perm_to_grant(
        self,
        perm: FolderPermission,
        grantee_type: str,
        grantee_id: uuid.UUID,
    ) -> PermissionGrant:
        """Convert a FolderPermission to a PermissionGrant."""
        return PermissionGrant(
            permission_type=perm.permission_type,
            resource_type=ResourceType.FOLDER,
            resource_id=perm.folder_path,
            grantee_type=grantee_type,
            grantee_id=grantee_id,
            inherited_from=perm.inherited_from,
            is_inherited=perm.is_inherited,
            expires_at=perm.expires_at,
        )
    
    def _get_parent_path(self, path: str) -> Optional[str]:
        """Get parent folder path."""
        if not path or path == "/":
            return None
        path = path.rstrip("/")
        parent = "/".join(path.split("/")[:-1])
        return parent if parent else "/"
    
    def _get_folder_from_file_path(self, file_path: str) -> Optional[str]:
        """Extract folder path from a file path."""
        if not file_path:
            return None
        parts = file_path.replace("\\", "/").split("/")
        if len(parts) <= 1:
            return "/"
        return "/".join(parts[:-1])
    
    async def get_permission_tree(
        self,
        resource_type: ResourceType,
        resource_id: str,
    ) -> PermissionTree:
        """Get the full permission tree for a resource (for admin view)."""
        direct_grants: List[PermissionGrant] = []
        inherited_grants: List[PermissionGrant] = []
        child_grants: Dict[str, List[PermissionGrant]] = {}
        
        if resource_type == ResourceType.FOLDER:
            # Get direct permissions on this folder
            perms = await self.db.execute(
                select(FolderPermission).where(
                    FolderPermission.folder_path == resource_id
                )
            )
            for perm in perms.scalars():
                grantee_type = "user" if perm.user_id else "group"
                grantee_id = perm.user_id or perm.group_id
                grant = self._perm_to_grant(perm, grantee_type, grantee_id)
                if perm.is_inherited:
                    inherited_grants.append(grant)
                else:
                    direct_grants.append(grant)
            
            # Get child folder permissions
            child_perms = await self.db.execute(
                select(FolderPermission).where(
                    FolderPermission.folder_path.startswith(resource_id + "/")
                )
            )
            for perm in child_perms.scalars():
                if perm.folder_path not in child_grants:
                    child_grants[perm.folder_path] = []
                grantee_type = "user" if perm.user_id else "group"
                grantee_id = perm.user_id or perm.group_id
                child_grants[perm.folder_path].append(
                    self._perm_to_grant(perm, grantee_type, grantee_id)
                )
        
        return PermissionTree(
            resource_type=resource_type,
            resource_id=resource_id,
            direct_grants=direct_grants,
            inherited_grants=inherited_grants,
            child_grants=child_grants,
        )
    
    async def grant_permission(
        self,
        folder_path: str,
        permission_type: PermissionType,
        grantee_type: str,
        grantee_id: uuid.UUID,
        granted_by: uuid.UUID,
        tenant_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> FolderPermission:
        """Grant a permission on a folder."""
        perm = FolderPermission(
            folder_path=folder_path,
            permission_type=permission_type,
            tenant_id=tenant_id,
            created_by=granted_by,
            expires_at=expires_at,
            is_inherited=False,
        )
        
        if grantee_type == "user":
            perm.user_id = grantee_id
        else:
            perm.group_id = grantee_id
        
        self.db.add(perm)
        await self.db.commit()
        await self.db.refresh(perm)
        
        # Clear cache
        self._cache.clear()
        
        logger.info(
            f"Granted {permission_type.value} on {folder_path} "
            f"to {grantee_type}:{grantee_id}"
        )
        
        return perm
    
    async def revoke_permission(
        self,
        folder_path: str,
        grantee_type: str,
        grantee_id: uuid.UUID,
    ) -> bool:
        """Revoke permissions on a folder."""
        if grantee_type == "user":
            result = await self.db.execute(
                select(FolderPermission).where(
                    and_(
                        FolderPermission.folder_path == folder_path,
                        FolderPermission.user_id == grantee_id,
                    )
                )
            )
        else:
            result = await self.db.execute(
                select(FolderPermission).where(
                    and_(
                        FolderPermission.folder_path == folder_path,
                        FolderPermission.group_id == grantee_id,
                    )
                )
            )
        
        perms = result.scalars().all()
        for perm in perms:
            await self.db.delete(perm)
        
        await self.db.commit()
        
        # Clear cache
        self._cache.clear()
        
        logger.info(
            f"Revoked permissions on {folder_path} "
            f"from {grantee_type}:{grantee_id}"
        )
        
        return len(perms) > 0
    
    def clear_cache(self) -> None:
        """Clear the permission cache."""
        self._cache.clear()


# =============================================================================
# Service Factory
# =============================================================================

def get_permission_inheritance_service(db: AsyncSession) -> PermissionInheritanceService:
    """Get a PermissionInheritanceService instance."""
    return PermissionInheritanceService(db)

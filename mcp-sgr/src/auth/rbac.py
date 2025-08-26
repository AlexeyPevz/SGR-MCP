"""Role-Based Access Control (RBAC) implementation."""

import hashlib
import secrets
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import asyncio
import logging

from .models import User, Organization, Role, Permission, APIKey, UserRole, DEFAULT_ROLES
from ..utils.cache import CacheManager

logger = logging.getLogger(__name__)


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager
        self.cache_ttl = 300  # 5 minutes
        self._users: Dict[str, User] = {}
        self._organizations: Dict[str, Organization] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._roles = DEFAULT_ROLES.copy()
    
    # User Management
    
    async def create_user(
        self, 
        email: str, 
        name: str, 
        organization_id: str,
        role: UserRole = UserRole.VIEWER,
        permissions: Optional[List[Permission]] = None
    ) -> User:
        """Create a new user."""
        # Check if organization exists
        org = await self.get_organization(organization_id)
        if not org:
            raise ValueError(f"Organization {organization_id} not found")
        
        # Check if user already exists
        existing_user = await self.get_user_by_email(email)
        if existing_user:
            raise ValueError(f"User with email {email} already exists")
        
        # Create user
        user = User(
            email=email,
            name=name,
            organization_id=organization_id,
            role=role,
            permissions=permissions or []
        )
        
        self._users[user.id] = user
        
        # Cache user
        if self.cache_manager:
            await self.cache_manager.set_cache(
                f"user:{user.id}", 
                user.dict(), 
                ttl=self.cache_ttl
            )
            await self.cache_manager.set_cache(
                f"user_email:{email}", 
                user.id, 
                ttl=self.cache_ttl
            )
        
        logger.info(f"Created user {user.id} for organization {organization_id}")
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        # Try cache first
        if self.cache_manager:
            cached = await self.cache_manager.get_cache(f"user:{user_id}")
            if cached:
                return User(**cached)
        
        # Fallback to in-memory store
        user = self._users.get(user_id)
        if user and self.cache_manager:
            await self.cache_manager.set_cache(
                f"user:{user_id}", 
                user.dict(), 
                ttl=self.cache_ttl
            )
        
        return user
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        # Try cache first
        if self.cache_manager:
            user_id = await self.cache_manager.get_cache(f"user_email:{email}")
            if user_id:
                return await self.get_user(user_id)
        
        # Fallback to scanning all users
        for user in self._users.values():
            if user.email == email:
                if self.cache_manager:
                    await self.cache_manager.set_cache(
                        f"user_email:{email}", 
                        user.id, 
                        ttl=self.cache_ttl
                    )
                return user
        
        return None
    
    async def update_user(self, user_id: str, **updates) -> Optional[User]:
        """Update user."""
        user = await self.get_user(user_id)
        if not user:
            return None
        
        # Update fields
        for field, value in updates.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        self._users[user_id] = user
        
        # Update cache
        if self.cache_manager:
            await self.cache_manager.set_cache(
                f"user:{user_id}", 
                user.dict(), 
                ttl=self.cache_ttl
            )
        
        return user
    
    # Organization Management
    
    async def create_organization(
        self, 
        name: str, 
        domain: Optional[str] = None,
        plan: str = "free"
    ) -> Organization:
        """Create a new organization."""
        org = Organization(
            name=name,
            domain=domain,
            plan=plan
        )
        
        self._organizations[org.id] = org
        
        # Cache organization
        if self.cache_manager:
            await self.cache_manager.set_cache(
                f"org:{org.id}", 
                org.dict(), 
                ttl=self.cache_ttl
            )
        
        logger.info(f"Created organization {org.id}: {name}")
        return org
    
    async def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        # Try cache first
        if self.cache_manager:
            cached = await self.cache_manager.get_cache(f"org:{org_id}")
            if cached:
                return Organization(**cached)
        
        # Fallback to in-memory store
        org = self._organizations.get(org_id)
        if org and self.cache_manager:
            await self.cache_manager.set_cache(
                f"org:{org_id}", 
                org.dict(), 
                ttl=self.cache_ttl
            )
        
        return org
    
    # API Key Management
    
    async def create_api_key(
        self, 
        user_id: str, 
        name: str,
        permissions: Optional[List[Permission]] = None,
        expires_in_days: Optional[int] = None
    ) -> tuple[APIKey, str]:
        """Create API key for user. Returns (api_key_object, raw_key)."""
        user = await self.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Generate raw key
        raw_key = f"sgr_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create API key
        api_key = APIKey(
            name=name,
            key_hash=key_hash,
            user_id=user_id,
            organization_id=user.organization_id,
            permissions=permissions or [],
            expires_at=expires_at
        )
        
        self._api_keys[api_key.id] = api_key
        
        # Cache API key by hash for fast lookup
        if self.cache_manager:
            await self.cache_manager.set_cache(
                f"api_key:{key_hash}", 
                api_key.dict(), 
                ttl=self.cache_ttl
            )
        
        logger.info(f"Created API key {api_key.id} for user {user_id}")
        return api_key, raw_key
    
    async def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate API key and return API key object if valid."""
        if not raw_key.startswith("sgr_"):
            return None
        
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Try cache first
        if self.cache_manager:
            cached = await self.cache_manager.get_cache(f"api_key:{key_hash}")
            if cached:
                api_key = APIKey(**cached)
                
                # Check if key is still valid
                if not api_key.is_active:
                    return None
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    return None
                
                # Update last used
                api_key.last_used = datetime.utcnow()
                api_key.usage_count += 1
                
                # Update in storage and cache
                self._api_keys[api_key.id] = api_key
                await self.cache_manager.set_cache(
                    f"api_key:{key_hash}", 
                    api_key.dict(), 
                    ttl=self.cache_ttl
                )
                
                return api_key
        
        # Fallback to scanning all keys
        for api_key in self._api_keys.values():
            if api_key.key_hash == key_hash:
                if not api_key.is_active:
                    return None
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    return None
                
                # Update usage
                api_key.last_used = datetime.utcnow()
                api_key.usage_count += 1
                self._api_keys[api_key.id] = api_key
                
                # Cache for future lookups
                if self.cache_manager:
                    await self.cache_manager.set_cache(
                        f"api_key:{key_hash}", 
                        api_key.dict(), 
                        ttl=self.cache_ttl
                    )
                
                return api_key
        
        return None
    
    # Permission Management
    
    async def check_permission(
        self, 
        user_id: str, 
        permission: Permission,
        api_key_id: Optional[str] = None
    ) -> bool:
        """Check if user has specific permission."""
        user = await self.get_user(user_id)
        if not user or not user.is_active:
            return False
        
        # Get organization
        org = await self.get_organization(user.organization_id)
        if not org or not org.is_active:
            return False
        
        # Check user role permissions
        role = self._roles.get(user.role)
        if role and permission in role.permissions:
            return True
        
        # Check user-specific permissions
        if permission in user.permissions:
            return True
        
        # Check API key permissions if provided
        if api_key_id:
            for api_key in self._api_keys.values():
                if api_key.id == api_key_id and permission in api_key.permissions:
                    return True
        
        return False
    
    async def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        user = await self.get_user(user_id)
        if not user:
            return set()
        
        permissions = set(user.permissions)
        
        # Add role permissions
        role = self._roles.get(user.role)
        if role:
            permissions.update(role.permissions)
        
        return permissions
    
    # Quota Management
    
    async def check_quota(self, organization_id: str, increment: int = 1) -> bool:
        """Check if organization has quota available."""
        org = await self.get_organization(organization_id)
        if not org:
            return False
        
        return org.used_quota + increment <= org.monthly_quota
    
    async def consume_quota(self, organization_id: str, amount: int = 1) -> bool:
        """Consume quota for organization."""
        org = await self.get_organization(organization_id)
        if not org:
            return False
        
        if org.used_quota + amount > org.monthly_quota:
            return False
        
        org.used_quota += amount
        self._organizations[organization_id] = org
        
        # Update cache
        if self.cache_manager:
            await self.cache_manager.set_cache(
                f"org:{organization_id}", 
                org.dict(), 
                ttl=self.cache_ttl
            )
        
        return True
    
    async def reset_monthly_quota(self, organization_id: str) -> bool:
        """Reset monthly quota for organization."""
        org = await self.get_organization(organization_id)
        if not org:
            return False
        
        org.used_quota = 0
        self._organizations[organization_id] = org
        
        if self.cache_manager:
            await self.cache_manager.set_cache(
                f"org:{organization_id}", 
                org.dict(), 
                ttl=self.cache_ttl
            )
        
        logger.info(f"Reset quota for organization {organization_id}")
        return True
    
    # Statistics
    
    async def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics."""
        user = await self.get_user(user_id)
        if not user:
            return {}
        
        # Count API keys
        api_key_count = sum(1 for key in self._api_keys.values() if key.user_id == user_id)
        
        return {
            "user_id": user_id,
            "organization_id": user.organization_id,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "last_login": user.last_login,
            "api_key_count": api_key_count,
            "permissions_count": len(await self.get_user_permissions(user_id))
        }
    
    async def get_organization_stats(self, org_id: str) -> Dict:
        """Get organization statistics."""
        org = await self.get_organization(org_id)
        if not org:
            return {}
        
        # Count users
        user_count = sum(1 for user in self._users.values() if user.organization_id == org_id)
        active_user_count = sum(
            1 for user in self._users.values() 
            if user.organization_id == org_id and user.is_active
        )
        
        # Count API keys
        api_key_count = sum(
            1 for key in self._api_keys.values() 
            if key.organization_id == org_id
        )
        
        return {
            "organization_id": org_id,
            "name": org.name,
            "plan": org.plan,
            "is_active": org.is_active,
            "user_count": user_count,
            "active_user_count": active_user_count,
            "api_key_count": api_key_count,
            "quota_used": org.used_quota,
            "quota_total": org.monthly_quota,
            "quota_percent": (org.used_quota / org.monthly_quota * 100) if org.monthly_quota > 0 else 0,
            "features": org.features
        }
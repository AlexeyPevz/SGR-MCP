"""Authentication and authorization models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class UserRole(str, Enum):
    """User roles in the system."""
    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions."""
    # SGR Operations
    SGR_APPLY = "sgr:apply"
    SGR_LEARN = "sgr:learn"
    SGR_BATCH = "sgr:batch"
    
    # Agent Operations  
    AGENT_WRAP = "agent:wrap"
    AGENT_CREATE = "agent:create"
    
    # Prompt Operations
    PROMPT_ENHANCE = "prompt:enhance"
    
    # Schema Operations
    SCHEMA_READ = "schema:read"
    SCHEMA_CREATE = "schema:create"
    SCHEMA_UPDATE = "schema:update"
    SCHEMA_DELETE = "schema:delete"
    
    # Monitoring
    METRICS_READ = "metrics:read"
    TRACES_READ = "traces:read"
    HEALTH_CHECK = "health:check"
    
    # Administration
    ADMIN_USERS = "admin:users"
    ADMIN_ORGS = "admin:orgs"
    ADMIN_BILLING = "admin:billing"
    ADMIN_SYSTEM = "admin:system"


class User(BaseModel):
    """User model with multi-tenant support."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    is_active: bool = True
    is_verified: bool = False
    organization_id: str
    role: UserRole = UserRole.VIEWER
    permissions: List[Permission] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # API Key management
    api_keys: List[str] = Field(default_factory=list)
    api_quota: Dict[str, int] = Field(default_factory=dict)  # endpoint -> limit
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Organization(BaseModel):
    """Organization model for multi-tenancy."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    domain: Optional[str] = None  # email domain for auto-assignment
    is_active: bool = True
    
    # Billing and limits
    plan: str = "free"  # free, pro, enterprise
    monthly_quota: int = 1000  # requests per month
    used_quota: int = 0
    
    # Features enabled
    features: Dict[str, bool] = Field(default_factory=lambda: {
        "custom_schemas": False,
        "batch_processing": False,
        "advanced_analytics": False,
        "priority_support": False,
        "sso": False
    })
    
    # Configuration
    settings: Dict[str, Any] = Field(default_factory=lambda: {
        "default_llm_backend": "openrouter",
        "rate_limit": 100,  # per minute
        "data_retention_days": 30,
        "allowed_models": ["*"]
    })
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Role(BaseModel):
    """Role with permissions."""
    name: UserRole
    permissions: List[Permission]
    description: str


class APIKey(BaseModel):
    """API Key model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    key_hash: str  # hashed version of the key
    user_id: str
    organization_id: str
    is_active: bool = True
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    permissions: List[Permission] = Field(default_factory=list)
    
    # Usage tracking
    usage_count: int = 0
    rate_limit: Optional[int] = None  # requests per minute
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLog(BaseModel):
    """Audit log entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Who
    user_id: Optional[str] = None
    organization_id: str
    api_key_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # What
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    
    # Details
    request_data: Optional[Dict[str, Any]] = None
    response_status: Optional[int] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Context
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    duration_ms: Optional[float] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Default role definitions
DEFAULT_ROLES = {
    UserRole.SUPER_ADMIN: Role(
        name=UserRole.SUPER_ADMIN,
        description="Full system access",
        permissions=list(Permission)
    ),
    
    UserRole.ORG_ADMIN: Role(
        name=UserRole.ORG_ADMIN,
        description="Organization administrator",
        permissions=[
            Permission.SGR_APPLY,
            Permission.SGR_LEARN,
            Permission.SGR_BATCH,
            Permission.AGENT_WRAP,
            Permission.AGENT_CREATE,
            Permission.PROMPT_ENHANCE,
            Permission.SCHEMA_READ,
            Permission.SCHEMA_CREATE,
            Permission.SCHEMA_UPDATE,
            Permission.SCHEMA_DELETE,
            Permission.METRICS_READ,
            Permission.TRACES_READ,
            Permission.HEALTH_CHECK,
            Permission.ADMIN_USERS,
            Permission.ADMIN_BILLING
        ]
    ),
    
    UserRole.DEVELOPER: Role(
        name=UserRole.DEVELOPER,
        description="Developer access",
        permissions=[
            Permission.SGR_APPLY,
            Permission.SGR_LEARN,
            Permission.SGR_BATCH,
            Permission.AGENT_WRAP,
            Permission.PROMPT_ENHANCE,
            Permission.SCHEMA_READ,
            Permission.SCHEMA_CREATE,
            Permission.SCHEMA_UPDATE,
            Permission.METRICS_READ,
            Permission.TRACES_READ,
            Permission.HEALTH_CHECK
        ]
    ),
    
    UserRole.ANALYST: Role(
        name=UserRole.ANALYST,
        description="Analyst access",
        permissions=[
            Permission.SGR_APPLY,
            Permission.PROMPT_ENHANCE,
            Permission.SCHEMA_READ,
            Permission.METRICS_READ,
            Permission.TRACES_READ
        ]
    ),
    
    UserRole.VIEWER: Role(
        name=UserRole.VIEWER,
        description="Read-only access",
        permissions=[
            Permission.SCHEMA_READ,
            Permission.METRICS_READ,
            Permission.HEALTH_CHECK
        ]
    )
}
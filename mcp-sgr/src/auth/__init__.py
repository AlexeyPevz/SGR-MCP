"""Authentication and authorization module for MCP-SGR."""

from .models import User, Organization, Role, Permission
from .rbac import RBACManager
# Optional: multi-tenant manager may not be present in all builds
try:
	from .multi_tenant import TenantManager  # type: ignore
except Exception:  # pragma: no cover
	TenantManager = None  # type: ignore
from .audit import AuditLogger

__all__ = [
	"User",
	"Organization",
	"Role",
	"Permission",
	"RBACManager",
	"TenantManager",
	"AuditLogger",
]
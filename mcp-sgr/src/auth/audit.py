"""Audit logging for compliance and security monitoring."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from .models import AuditLog
from ..utils.cache import CacheManager

logger = logging.getLogger(__name__)


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager
        self._logs: List[AuditLog] = []
        self._buffer_size = 1000
        self._flush_interval = 60  # seconds
        self._background_task = None
        
    async def initialize(self):
        """Initialize audit logger."""
        # Start background flush task
        self._background_task = asyncio.create_task(self._background_flush())
        logger.info("Audit logger initialized")
    
    async def close(self):
        """Close audit logger and flush remaining logs."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        await self._flush_logs()
        logger.info("Audit logger closed")
    
    async def log_action(
        self,
        action: str,
        resource_type: str,
        organization_id: str,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_status: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """Log an audit event."""
        
        # Sanitize sensitive data
        sanitized_request = self._sanitize_data(request_data) if request_data else None
        sanitized_response = self._sanitize_data(response_data) if response_data else None
        
        audit_log = AuditLog(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            organization_id=organization_id,
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            endpoint=endpoint,
            method=method,
            request_data=sanitized_request,
            response_status=response_status,
            response_data=sanitized_response,
            error_message=error_message,
            session_id=session_id,
            trace_id=trace_id,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self._logs.append(audit_log)
        
        # Flush if buffer is full
        if len(self._logs) >= self._buffer_size:
            await self._flush_logs()
        
        return audit_log
    
    async def log_authentication(
        self,
        action: str,  # login, logout, login_failed, api_key_used
        organization_id: str,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """Log authentication events."""
        return await self.log_action(
            action=action,
            resource_type="authentication",
            organization_id=organization_id,
            user_id=user_id,
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message,
            metadata=metadata
        )
    
    async def log_sgr_operation(
        self,
        operation: str,  # apply_sgr, learn_schema, enhance_prompt
        organization_id: str,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        schema_type: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> AuditLog:
        """Log SGR operations."""
        return await self.log_action(
            action=operation,
            resource_type="sgr_operation",
            organization_id=organization_id,
            user_id=user_id,
            api_key_id=api_key_id,
            response_status=200 if success else 500,
            error_message=error_message,
            ip_address=ip_address,
            trace_id=trace_id,
            duration_ms=duration_ms,
            metadata={
                "schema_type": schema_type,
                "model_used": model_used,
                "tokens_used": tokens_used,
                "success": success
            }
        )
    
    async def log_admin_action(
        self,
        action: str,
        target_resource_type: str,
        target_resource_id: str,
        organization_id: str,
        admin_user_id: str,
        changes: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """Log administrative actions."""
        return await self.log_action(
            action=f"admin_{action}",
            resource_type=target_resource_type,
            resource_id=target_resource_id,
            organization_id=organization_id,
            user_id=admin_user_id,
            ip_address=ip_address,
            metadata={
                **(metadata or {}),
                "changes": changes,
                "admin_action": True
            }
        )
    
    async def log_quota_event(
        self,
        event_type: str,  # quota_exceeded, quota_reset, quota_increased
        organization_id: str,
        current_usage: int,
        quota_limit: int,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None
    ) -> AuditLog:
        """Log quota-related events."""
        return await self.log_action(
            action=event_type,
            resource_type="quota",
            organization_id=organization_id,
            user_id=user_id,
            api_key_id=api_key_id,
            metadata={
                "current_usage": current_usage,
                "quota_limit": quota_limit,
                "usage_percentage": (current_usage / quota_limit * 100) if quota_limit > 0 else 0
            }
        )
    
    async def search_logs(
        self,
        organization_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Search audit logs with filters."""
        # First flush current buffer
        await self._flush_logs()
        
        # Filter logs
        filtered_logs = []
        for log in self._logs:
            # Filter by organization
            if log.organization_id != organization_id:
                continue
            
            # Filter by time range
            if start_time and log.timestamp < start_time:
                continue
            if end_time and log.timestamp > end_time:
                continue
            
            # Filter by action
            if action and log.action != action:
                continue
            
            # Filter by resource type
            if resource_type and log.resource_type != resource_type:
                continue
            
            # Filter by user
            if user_id and log.user_id != user_id:
                continue
            
            filtered_logs.append(log)
        
        # Sort by timestamp (newest first) and limit
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_logs[:limit]
    
    async def get_security_events(
        self,
        organization_id: str,
        hours: int = 24
    ) -> List[AuditLog]:
        """Get security-related events for the last N hours."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        security_actions = [
            "login_failed",
            "api_key_invalid", 
            "permission_denied",
            "quota_exceeded",
            "rate_limit_exceeded",
            "admin_user_created",
            "admin_user_deleted",
            "admin_permissions_changed"
        ]
        
        all_events = []
        for action in security_actions:
            events = await self.search_logs(
                organization_id=organization_id,
                start_time=start_time,
                action=action,
                limit=50
            )
            all_events.extend(events)
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x.timestamp, reverse=True)
        return all_events
    
    async def get_usage_summary(
        self,
        organization_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get usage summary for organization."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        logs = await self.search_logs(
            organization_id=organization_id,
            start_time=start_time,
            limit=10000
        )
        
        # Analyze logs
        summary = {
            "total_requests": len(logs),
            "unique_users": len(set(log.user_id for log in logs if log.user_id)),
            "unique_api_keys": len(set(log.api_key_id for log in logs if log.api_key_id)),
            "actions": {},
            "endpoints": {},
            "errors": 0,
            "avg_duration_ms": 0
        }
        
        total_duration = 0
        duration_count = 0
        
        for log in logs:
            # Count actions
            action = log.action
            summary["actions"][action] = summary["actions"].get(action, 0) + 1
            
            # Count endpoints
            if log.endpoint:
                summary["endpoints"][log.endpoint] = summary["endpoints"].get(log.endpoint, 0) + 1
            
            # Count errors
            if log.error_message or (log.response_status and log.response_status >= 400):
                summary["errors"] += 1
            
            # Calculate average duration
            if log.duration_ms:
                total_duration += log.duration_ms
                duration_count += 1
        
        if duration_count > 0:
            summary["avg_duration_ms"] = total_duration / duration_count
        
        return summary
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from logged data."""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        sensitive_keys = {
            "password", "token", "key", "secret", "authorization", 
            "api_key", "auth", "credential", "private"
        }
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key contains sensitive terms
            if any(sensitive_term in key_lower for sensitive_term in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_data(item) if isinstance(item, dict) else item 
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    async def _flush_logs(self):
        """Flush logs to persistent storage."""
        if not self._logs:
            return
        
        try:
            # In a real implementation, you would write to database
            # For now, we'll use cache manager if available
            if self.cache_manager:
                for log in self._logs:
                    await self.cache_manager.set_cache(
                        f"audit_log:{log.id}",
                        log.dict(),
                        ttl=86400 * 30  # 30 days
                    )
            
            # Also log to file for backup
            log_file = "logs/audit.jsonl"
            try:
                import os
                os.makedirs("logs", exist_ok=True)
                
                with open(log_file, "a") as f:
                    for log in self._logs:
                        f.write(json.dumps(log.dict(), default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit logs to file: {e}")
            
            logger.debug(f"Flushed {len(self._logs)} audit logs")
            self._logs.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit logs: {e}")
    
    async def _background_flush(self):
        """Background task to periodically flush logs."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background flush: {e}")


@asynccontextmanager
async def audit_context(
    audit_logger: AuditLogger,
    action: str,
    resource_type: str,
    organization_id: str,
    **kwargs
):
    """Context manager for automatic audit logging."""
    start_time = datetime.utcnow()
    error_message = None
    
    try:
        yield
        success = True
    except Exception as e:
        success = False
        error_message = str(e)
        raise
    finally:
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        await audit_logger.log_action(
            action=action,
            resource_type=resource_type,
            organization_id=organization_id,
            duration_ms=duration_ms,
            error_message=error_message,
            response_status=200 if success else 500,
            **kwargs
        )
"""Cache manager for SGR results and traces."""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching and trace storage."""

    def __init__(self):
        """Initialize cache manager."""
        self.enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.cache_store = os.getenv("CACHE_STORE", "sqlite:///./data/cache.db")
        self.default_ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        self.max_cache_size = int(os.getenv("CACHE_MAX_SIZE_MB", "100"))  # MB
        self.max_entries = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))

        self.trace_enabled = os.getenv("TRACE_ENABLED", "true").lower() == "true"
        self.trace_store = os.getenv("TRACE_STORE", "sqlite:///./data/traces.db")
        self.trace_retention_days = int(os.getenv("TRACE_RETENTION_DAYS", "7"))

        self._cache_db = None
        self._trace_db = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return

        # Create data directory
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)

        # Initialize cache database
        if self.enabled:
            cache_path = self.cache_store.replace("sqlite:///", "")
            self._cache_db = await aiosqlite.connect(cache_path)

            await self._cache_db.execute(
                """
				CREATE TABLE IF NOT EXISTS cache_entries (
					key TEXT PRIMARY KEY,
					value TEXT,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					expires_at TIMESTAMP,
					hit_count INTEGER DEFAULT 0
				)
			"""  # noqa: W191,E101
            )
            await self._cache_db.commit()

        # Initialize trace database
        if self.trace_enabled:
            trace_path = self.trace_store.replace("sqlite:///", "")
            self._trace_db = await aiosqlite.connect(trace_path)

            await self._trace_db.execute(
                """
				CREATE TABLE IF NOT EXISTS trace_entries (
					id TEXT PRIMARY KEY,
					tool_name TEXT,
					arguments TEXT,
					result TEXT,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
					duration_ms INTEGER,
					metadata TEXT
				)
			"""  # noqa: W191,E101
            )
            await self._trace_db.commit()

        self._initialized = True
        logger.info("Cache manager initialized")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        if not self.enabled or not self._cache_db:
            return None

        try:
            async with self._cache_db.execute(
                """SELECT value, expires_at, hit_count FROM cache_entries 
				   WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)""",
                (key, datetime.now(timezone.utc)),
            ) as cursor:
                row = await cursor.fetchone()

                if row:
                    value_str, expires_at, hit_count = row

                    # Update hit count
                    await self._cache_db.execute(
                        "UPDATE cache_entries SET hit_count = ? WHERE key = ?", (hit_count + 1, key)
                    )
                    await self._cache_db.commit()

                    return json.loads(value_str)

                return None

        except (json.JSONDecodeError, aiosqlite.Error) as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected cache get error: {e}", exc_info=True)
            return None

    # Backwards-compatible aliases used by tests and older code
    async def get_cache(self, key: str) -> Optional[Dict[str, Any]]:  # pragma: no cover
        return await self.get(key)

    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.enabled or not self._cache_db:
            return False

        try:
            # Check cache size limits
            await self._enforce_cache_limits()

            ttl = ttl or self.default_ttl
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl) if ttl > 0 else None

            # Pydantic-aware JSON dump
            try:
                value_json = json.dumps(value, default=str)
            except TypeError:
                value_json = json.dumps(value)

            await self._cache_db.execute(
                """INSERT OR REPLACE INTO cache_entries 
				   (key, value, created_at, expires_at, hit_count) 
				   VALUES (?, ?, ?, ?, 0)""",
                (key, value_json, datetime.now(timezone.utc), expires_at),
            )
            await self._cache_db.commit()

            return True

        except (TypeError, ValueError, aiosqlite.Error) as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected cache set error: {e}", exc_info=True)
            return False

    # Backwards-compatible alias
    async def set_cache(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:  # pragma: no cover
        return await self.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.enabled or not self._cache_db:
            return False

        try:
            await self._cache_db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            await self._cache_db.commit()
            return True

        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def clear_expired(self) -> int:
        """Clear expired cache entries."""
        if not self.enabled or not self._cache_db:
            return 0

        try:
            cursor = await self._cache_db.execute(
                "DELETE FROM cache_entries WHERE expires_at < ?", (datetime.now(timezone.utc),)
            )
            await self._cache_db.commit()
            return cursor.rowcount

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0

    async def add_trace(
        self,
        trace_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        duration_ms: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a trace entry."""
        if not self.trace_enabled or not self._trace_db:
            return False

        try:
            await self._trace_db.execute(
                """INSERT INTO trace_entries 
				   (id, tool_name, arguments, result, created_at, duration_ms, metadata) 
				   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace_id,
                    tool_name,
                    json.dumps(arguments, default=str),
                    json.dumps(result, default=str),
                    datetime.now(timezone.utc),
                    duration_ms,
                    json.dumps(metadata or {}, default=str),
                ),
            )
            await self._trace_db.commit()

            # Clean old traces
            await self._clean_old_traces()

            return True

        except Exception as e:
            logger.error(f"Trace add error: {e}")
            return False

    async def get_recent_traces(
        self, limit: int = 10, tool_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent trace entries."""
        if not self.trace_enabled or not self._trace_db:
            return []

        try:
            query = """SELECT id, tool_name, arguments, result, created_at, 
						      duration_ms, metadata 
				       FROM trace_entries"""
            params: List[Any] = []

            if tool_name:
                query += " WHERE tool_name = ?"
                params.append(tool_name)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            traces: List[Dict[str, Any]] = []
            async with self._trace_db.execute(query, params) as cursor:
                async for row in cursor:
                    traces.append(
                        {
                            "id": row[0],
                            "tool_name": row[1],
                            "arguments": json.loads(row[2]),
                            "result": json.loads(row[3]),
                            "created_at": row[4],
                            "duration_ms": row[5],
                            "metadata": json.loads(row[6]),
                        }
                    )

            return traces

        except Exception as e:
            logger.error(f"Get traces error: {e}")
            return []

    async def _clean_old_traces(self):
        """Clean traces older than retention period."""
        if not self.trace_enabled or not self._trace_db:
            return

        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.trace_retention_days)

            await self._trace_db.execute(
                "DELETE FROM trace_entries WHERE created_at < ?", (cutoff_date,)
            )
            await self._trace_db.commit()

        except Exception as e:
            logger.error(f"Clean traces error: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self._cache_db:
            return {"enabled": False}

        try:
            stats: Dict[str, Any] = {"enabled": True}

            # Total entries
            async with self._cache_db.execute("SELECT COUNT(*) FROM cache_entries") as cursor:
                stats["total_entries"] = (await cursor.fetchone())[0]

            # Active entries
            async with self._cache_db.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at IS NULL OR expires_at > ?",
                (datetime.now(timezone.utc),),
            ) as cursor:
                stats["active_entries"] = (await cursor.fetchone())[0]

            # Total hits
            async with self._cache_db.execute("SELECT SUM(hit_count) FROM cache_entries") as cursor:
                stats["total_hits"] = (await cursor.fetchone())[0] or 0

            # Hit rate
            if (
                isinstance(stats.get("total_hits"), int)
                and isinstance(stats.get("total_entries"), int)
                and stats["total_hits"] > 0
            ):
                denominator = stats["total_hits"] + stats["total_entries"]
                stats["hit_rate"] = (stats["total_hits"] / denominator) if denominator > 0 else 0.0
            else:
                stats["hit_rate"] = 0.0

            return stats

        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return {"enabled": True, "error": str(e)}

    async def _enforce_cache_limits(self):
        """Enforce cache size and entry limits."""
        if not self._cache_db:
            return

        try:
            # Check number of entries
            async with self._cache_db.execute("SELECT COUNT(*) FROM cache_entries") as cursor:
                count = (await cursor.fetchone())[0]

            if count >= self.max_entries:
                # Delete oldest entries (LRU based on hit count and created_at)
                to_delete = count - int(self.max_entries * 0.8)  # Keep 80% after cleanup
                await self._cache_db.execute(
                    """DELETE FROM cache_entries 
					   WHERE key IN (
					       SELECT key FROM cache_entries 
					       ORDER BY hit_count ASC, created_at ASC 
					       LIMIT ?
					   )""",  # noqa: W191,E101
                    (to_delete,),
                )
                await self._cache_db.commit()
                logger.info(f"Cleaned up {to_delete} cache entries due to limit")

            # Check total size (simplified - count total length of values)
            async with self._cache_db.execute(
                "SELECT SUM(LENGTH(value)) FROM cache_entries"
            ) as cursor:
                total_size = (await cursor.fetchone())[0] or 0

            # Convert to MB (rough estimate)
            size_mb = total_size / (1024 * 1024)

            if size_mb > self.max_cache_size:
                # Delete oldest large entries
                await self._cache_db.execute(
                    """DELETE FROM cache_entries 
					   WHERE key IN (
					       SELECT key FROM cache_entries 
					       ORDER BY LENGTH(value) DESC, created_at ASC 
					       LIMIT 10
					   )"""  # noqa: W191,E101
                )
                await self._cache_db.commit()
                logger.info(f"Cleaned up large cache entries, size was {size_mb:.1f}MB")

        except Exception as e:
            logger.error(f"Error enforcing cache limits: {e}")

    async def close(self):
        """Close database connections."""
        if self._cache_db:
            await self._cache_db.close()

        if self._trace_db:
            await self._trace_db.close()

        self._initialized = False

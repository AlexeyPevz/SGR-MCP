"""Smart caching system with ML-based cache optimization."""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    hit_count: int
    size_bytes: int
    ttl_seconds: Optional[int]
    tags: List[str]
    
    # Prediction metadata
    predicted_future_accesses: float = 0.0
    access_pattern_score: float = 0.0
    cache_utility_score: float = 0.0


@dataclass
class AccessPattern:
    """Pattern of cache access."""
    key_pattern: str
    frequency: float  # accesses per hour
    regularity: float  # 0-1, how regular the pattern is
    time_of_day_distribution: List[float]  # 24 hours
    last_seen: datetime


class SmartCacheManager:
    """ML-enhanced cache manager with predictive pre-loading."""
    
    def __init__(self, base_cache_manager=None, max_memory_mb: int = 1024):
        self.base_cache = base_cache_manager
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        
        # Smart cache storage
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.access_history: deque = deque(maxlen=10000)
        self.access_patterns: Dict[str, AccessPattern] = {}
        
        # ML parameters
        self.learning_rate = 0.1
        self.prediction_window_hours = 24
        self.pattern_detection_threshold = 5  # minimum accesses to detect pattern
        
        # Cache optimization
        self.hit_rate_target = 0.85
        self.memory_pressure_threshold = 0.8
        self.eviction_batch_size = 10
        
        # Background tasks
        self._optimization_task = None
        self._pattern_learning_task = None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "preloads": 0,
            "bytes_served": 0
        }
    
    async def initialize(self):
        """Initialize the smart cache manager."""
        # Load existing patterns and data
        await self._load_patterns()
        
        # Start background optimization
        self._optimization_task = asyncio.create_task(self._background_optimization())
        self._pattern_learning_task = asyncio.create_task(self._pattern_learning())
        
        logger.info("Smart cache manager initialized")
    
    async def close(self):
        """Close the smart cache manager."""
        # Cancel background tasks
        if self._optimization_task:
            self._optimization_task.cancel()
        if self._pattern_learning_task:
            self._pattern_learning_task.cancel()
        
        # Save state
        await self._save_patterns()
        
        logger.info("Smart cache manager closed")
    
    async def get(self, key: str, tags: Optional[List[str]] = None) -> Optional[Any]:
        """Get value from cache with smart learning."""
        start_time = time.time()
        
        self.stats["total_requests"] += 1
        
        # Record access
        await self._record_access(key, "get", tags or [])
        
        # Check smart cache first
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            
            # Check TTL
            if entry.ttl_seconds:
                age = (datetime.utcnow() - entry.created_at).total_seconds()
                if age > entry.ttl_seconds:
                    await self._evict_entry(key)
                    self.stats["cache_misses"] += 1
                    return None
            
            # Update access metadata
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            entry.hit_count += 1
            
            self.stats["cache_hits"] += 1
            self.stats["bytes_served"] += entry.size_bytes
            
            logger.debug(f"Smart cache hit: {key}")
            return entry.value
        
        # Fallback to base cache
        if self.base_cache:
            value = await self.base_cache.get_cache(key)
            if value is not None:
                # Promote to smart cache if valuable
                await self._promote_to_smart_cache(key, value, tags or [])
                self.stats["cache_hits"] += 1
                return value
        
        self.stats["cache_misses"] += 1
        
        # Try predictive loading
        await self._try_predictive_load(key)
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache with smart optimization."""
        # Calculate value size
        size_bytes = self._calculate_size(value)
        
        # Check if we should cache this value
        should_cache = await self._should_cache(key, value, size_bytes, tags or [])
        if not should_cache:
            # Store in base cache only
            if self.base_cache:
                return await self.base_cache.set_cache(key, value, ttl)
            return False
        
        # Record access
        await self._record_access(key, "set", tags or [])
        
        # Ensure memory availability
        await self._ensure_memory_available(size_bytes)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            hit_count=0,
            size_bytes=size_bytes,
            ttl_seconds=ttl,
            tags=tags or []
        )
        
        # Calculate utility score
        entry.cache_utility_score = await self._calculate_utility_score(entry)
        
        # Store in smart cache
        self.cache_entries[key] = entry
        self.current_memory_bytes += size_bytes
        
        # Also store in base cache for persistence
        if self.base_cache:
            await self.base_cache.set_cache(key, value, ttl)
        
        logger.debug(f"Smart cache set: {key} ({size_bytes} bytes)")
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        deleted = False
        
        # Remove from smart cache
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache_entries[key]
            deleted = True
        
        # Remove from base cache
        if self.base_cache:
            try:
                # Base cache might not have delete method
                if hasattr(self.base_cache, 'delete_cache'):
                    await self.base_cache.delete_cache(key)
                deleted = True
            except:
                pass
        
        await self._record_access(key, "delete", [])
        return deleted
    
    async def get_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Get all cached values with specific tags."""
        results = {}
        
        for key, entry in self.cache_entries.items():
            if any(tag in entry.tags for tag in tags):
                results[key] = entry.value
        
        return results
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all cache entries with specific tags."""
        keys_to_delete = []
        
        for key, entry in self.cache_entries.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            await self.delete(key)
        
        return len(keys_to_delete)
    
    async def preload_predicted_keys(self) -> int:
        """Preload keys that are predicted to be accessed soon."""
        if not self.base_cache:
            return 0
        
        preloaded = 0
        current_hour = datetime.utcnow().hour
        
        # Predict keys likely to be accessed in next hour
        predicted_keys = await self._predict_future_accesses(current_hour)
        
        for key, confidence in predicted_keys[:5]:  # Limit to top 5 predictions
            if key not in self.cache_entries and confidence > 0.7:
                # Try to load from base cache
                value = await self.base_cache.get_cache(key)
                if value is not None:
                    await self._promote_to_smart_cache(key, value, ["preloaded"])
                    preloaded += 1
                    logger.debug(f"Preloaded key: {key} (confidence: {confidence:.2f})")
        
        self.stats["preloads"] += preloaded
        return preloaded
    
    async def _record_access(self, key: str, operation: str, tags: List[str]):
        """Record cache access for learning."""
        access_record = {
            "timestamp": datetime.utcnow(),
            "key": key,
            "operation": operation,
            "tags": tags,
            "hour": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday()
        }
        
        self.access_history.append(access_record)
        
        # Update pattern tracking
        await self._update_access_patterns(key, access_record)
    
    async def _update_access_patterns(self, key: str, access_record: Dict):
        """Update access patterns for key."""
        # Generalize key pattern (remove IDs, timestamps, etc.)
        pattern_key = self._generalize_key_pattern(key)
        
        if pattern_key not in self.access_patterns:
            self.access_patterns[pattern_key] = AccessPattern(
                key_pattern=pattern_key,
                frequency=0.0,
                regularity=0.0,
                time_of_day_distribution=[0.0] * 24,
                last_seen=access_record["timestamp"]
            )
        
        pattern = self.access_patterns[pattern_key]
        
        # Update frequency (exponential moving average)
        hours_since_last = (access_record["timestamp"] - pattern.last_seen).total_seconds() / 3600
        if hours_since_last > 0:
            alpha = 0.1
            pattern.frequency = (1 - alpha) * pattern.frequency + alpha * (1 / hours_since_last)
        
        # Update time distribution
        hour = access_record["hour"]
        pattern.time_of_day_distribution[hour] += 0.1
        
        # Normalize distribution
        total = sum(pattern.time_of_day_distribution)
        if total > 0:
            pattern.time_of_day_distribution = [x / total for x in pattern.time_of_day_distribution]
        
        pattern.last_seen = access_record["timestamp"]
        
        # Calculate regularity based on time distribution variance
        variance = np.var(pattern.time_of_day_distribution)
        pattern.regularity = 1.0 / (1.0 + variance * 10)  # Higher variance = lower regularity
    
    def _generalize_key_pattern(self, key: str) -> str:
        """Generalize cache key to detect patterns."""
        # Replace UUIDs, timestamps, and other variable parts
        import re
        
        # Replace UUIDs
        pattern = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '{uuid}', key)
        
        # Replace timestamps
        pattern = re.sub(r'\d{10,}', '{timestamp}', pattern)
        
        # Replace numbers
        pattern = re.sub(r'\d+', '{number}', pattern)
        
        return pattern
    
    async def _should_cache(
        self,
        key: str,
        value: Any,
        size_bytes: int,
        tags: List[str]
    ) -> bool:
        """Decide whether to cache a value based on ML predictions."""
        # Don't cache very large values
        if size_bytes > self.max_memory_bytes * 0.1:  # More than 10% of total memory
            return False
        
        # Always cache small, frequently accessed items
        if size_bytes < 1024:  # Less than 1KB
            return True
        
        # Check access patterns
        pattern_key = self._generalize_key_pattern(key)
        if pattern_key in self.access_patterns:
            pattern = self.access_patterns[pattern_key]
            
            # Cache if frequently accessed or regular pattern
            if pattern.frequency > 1.0 or pattern.regularity > 0.7:
                return True
        
        # Cache if tagged as important
        important_tags = ["user_profile", "schema", "config", "frequent"]
        if any(tag in important_tags for tag in tags):
            return True
        
        # Don't cache temporary or one-time items
        temp_tags = ["temp", "temporary", "one_time", "debug"]
        if any(tag in temp_tags for tag in tags):
            return False
        
        # Default decision based on memory pressure
        memory_usage = self.current_memory_bytes / self.max_memory_bytes
        if memory_usage < 0.5:
            return True  # Plenty of memory, cache it
        elif memory_usage < 0.8:
            return size_bytes < 10240  # Cache only small items when memory is getting full
        else:
            return False  # Memory pressure, don't cache
    
    async def _calculate_utility_score(self, entry: CacheEntry) -> float:
        """Calculate utility score for cache entry."""
        # Base score from access frequency
        frequency_score = min(entry.access_count / 10.0, 1.0)
        
        # Size penalty (smaller items score higher)
        size_penalty = 1.0 / (1.0 + entry.size_bytes / 10240)  # 10KB reference
        
        # Recency score
        age_hours = (datetime.utcnow() - entry.last_accessed).total_seconds() / 3600
        recency_score = 1.0 / (1.0 + age_hours / 24)  # 24 hours reference
        
        # Pattern score
        pattern_key = self._generalize_key_pattern(entry.key)
        pattern_score = 0.5  # Default
        if pattern_key in self.access_patterns:
            pattern = self.access_patterns[pattern_key]
            pattern_score = pattern.frequency * pattern.regularity
        
        # Combined score
        utility_score = (
            0.3 * frequency_score +
            0.2 * size_penalty +
            0.2 * recency_score +
            0.3 * pattern_score
        )
        
        return min(utility_score, 1.0)
    
    async def _ensure_memory_available(self, required_bytes: int):
        """Ensure enough memory is available for new entry."""
        if self.current_memory_bytes + required_bytes <= self.max_memory_bytes:
            return  # Enough memory available
        
        # Need to evict some entries
        memory_to_free = required_bytes + (self.max_memory_bytes * 0.1)  # Free 10% extra
        
        # Get candidates for eviction (lowest utility scores)
        eviction_candidates = []
        for key, entry in self.cache_entries.items():
            # Update utility score
            entry.cache_utility_score = await self._calculate_utility_score(entry)
            eviction_candidates.append((key, entry))
        
        # Sort by utility score (lowest first)
        eviction_candidates.sort(key=lambda x: x[1].cache_utility_score)
        
        freed_memory = 0
        evicted_count = 0
        
        for key, entry in eviction_candidates:
            if freed_memory >= memory_to_free:
                break
            
            await self._evict_entry(key)
            freed_memory += entry.size_bytes
            evicted_count += 1
        
        self.stats["evictions"] += evicted_count
        logger.debug(f"Evicted {evicted_count} entries, freed {freed_memory} bytes")
    
    async def _evict_entry(self, key: str):
        """Evict a single cache entry."""
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache_entries[key]
    
    async def _promote_to_smart_cache(self, key: str, value: Any, tags: List[str]):
        """Promote a value from base cache to smart cache."""
        size_bytes = self._calculate_size(value)
        
        # Check if we should promote
        should_promote = await self._should_cache(key, value, size_bytes, tags)
        if not should_promote:
            return
        
        # Ensure memory
        await self._ensure_memory_available(size_bytes)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            hit_count=1,
            size_bytes=size_bytes,
            ttl_seconds=None,
            tags=tags
        )
        
        entry.cache_utility_score = await self._calculate_utility_score(entry)
        
        self.cache_entries[key] = entry
        self.current_memory_bytes += size_bytes
        
        logger.debug(f"Promoted to smart cache: {key}")
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value).encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8  # Approximate
            elif isinstance(value, bool):
                return 1
            else:
                # Try to serialize
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    async def _predict_future_accesses(self, current_hour: int) -> List[Tuple[str, float]]:
        """Predict which keys are likely to be accessed in the next hour."""
        predictions = []
        
        for pattern_key, pattern in self.access_patterns.items():
            # Predict based on time-of-day distribution
            next_hour = (current_hour + 1) % 24
            probability = pattern.time_of_day_distribution[next_hour]
            
            # Adjust for frequency and regularity
            confidence = probability * pattern.frequency * pattern.regularity
            
            if confidence > 0.1:  # Minimum threshold
                # Find actual keys matching this pattern
                matching_keys = [
                    key for key in self.access_history
                    if self._generalize_key_pattern(key.get("key", "")) == pattern_key
                ]
                
                # Get recent keys
                recent_keys = [
                    access["key"] for access in list(self.access_history)[-100:]
                    if self._generalize_key_pattern(access["key"]) == pattern_key
                ]
                
                for key in set(recent_keys):
                    predictions.append((key, confidence))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    async def _try_predictive_load(self, missed_key: str):
        """Try to predictively load related keys after a cache miss."""
        if not self.base_cache:
            return
        
        # Look for similar keys that might be accessed next
        pattern_key = self._generalize_key_pattern(missed_key)
        
        if pattern_key in self.access_patterns:
            pattern = self.access_patterns[pattern_key]
            
            # If this pattern is frequent, preload related keys
            if pattern.frequency > 0.5:
                current_hour = datetime.utcnow().hour
                predicted_keys = await self._predict_future_accesses(current_hour)
                
                # Load top prediction if confidence is high
                for key, confidence in predicted_keys[:1]:
                    if confidence > 0.8 and key not in self.cache_entries:
                        value = await self.base_cache.get_cache(key)
                        if value is not None:
                            await self._promote_to_smart_cache(key, value, ["predictive"])
                            logger.debug(f"Predictive load: {key}")
                            break
    
    async def _background_optimization(self):
        """Background task for cache optimization."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update utility scores
                for entry in self.cache_entries.values():
                    entry.cache_utility_score = await self._calculate_utility_score(entry)
                
                # Check memory pressure
                memory_usage = self.current_memory_bytes / self.max_memory_bytes
                if memory_usage > self.memory_pressure_threshold:
                    await self._ensure_memory_available(0)  # Force cleanup
                
                # Preload predicted keys
                await self.preload_predicted_keys()
                
                logger.debug("Cache optimization completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache optimization: {e}")
    
    async def _pattern_learning(self):
        """Background task for pattern learning."""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Clean up old access history
                cutoff_time = datetime.utcnow() - timedelta(hours=72)  # Keep 3 days
                old_count = len(self.access_history)
                self.access_history = deque(
                    [access for access in self.access_history if access["timestamp"] > cutoff_time],
                    maxlen=self.access_history.maxlen
                )
                new_count = len(self.access_history)
                
                if old_count != new_count:
                    logger.debug(f"Cleaned up {old_count - new_count} old access records")
                
                # Update patterns
                await self._update_pattern_statistics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pattern learning: {e}")
    
    async def _update_pattern_statistics(self):
        """Update pattern statistics from access history."""
        # Group accesses by pattern
        pattern_accesses = defaultdict(list)
        
        for access in self.access_history:
            pattern_key = self._generalize_key_pattern(access["key"])
            pattern_accesses[pattern_key].append(access)
        
        # Update pattern statistics
        for pattern_key, accesses in pattern_accesses.items():
            if len(accesses) >= self.pattern_detection_threshold:
                if pattern_key not in self.access_patterns:
                    self.access_patterns[pattern_key] = AccessPattern(
                        key_pattern=pattern_key,
                        frequency=0.0,
                        regularity=0.0,
                        time_of_day_distribution=[0.0] * 24,
                        last_seen=datetime.utcnow()
                    )
                
                pattern = self.access_patterns[pattern_key]
                
                # Calculate frequency (accesses per hour)
                time_span = (max(a["timestamp"] for a in accesses) - 
                           min(a["timestamp"] for a in accesses)).total_seconds() / 3600
                if time_span > 0:
                    pattern.frequency = len(accesses) / time_span
                
                # Update time distribution
                hour_counts = [0] * 24
                for access in accesses:
                    hour_counts[access["hour"]] += 1
                
                total_accesses = sum(hour_counts)
                if total_accesses > 0:
                    pattern.time_of_day_distribution = [count / total_accesses for count in hour_counts]
                    
                    # Calculate regularity
                    variance = np.var(pattern.time_of_day_distribution)
                    pattern.regularity = 1.0 / (1.0 + variance * 10)
    
    async def _save_patterns(self):
        """Save access patterns to persistent storage."""
        if not self.base_cache:
            return
        
        try:
            patterns_data = {}
            for pattern_key, pattern in self.access_patterns.items():
                patterns_data[pattern_key] = asdict(pattern)
                # Convert datetime to string
                patterns_data[pattern_key]["last_seen"] = pattern.last_seen.isoformat()
            
            await self.base_cache.set_cache(
                "smart_cache_patterns",
                patterns_data,
                ttl=86400 * 30  # 30 days
            )
            
            logger.debug(f"Saved {len(patterns_data)} access patterns")
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    async def _load_patterns(self):
        """Load access patterns from persistent storage."""
        if not self.base_cache:
            return
        
        try:
            patterns_data = await self.base_cache.get_cache("smart_cache_patterns")
            if patterns_data:
                for pattern_key, pattern_dict in patterns_data.items():
                    # Convert string back to datetime
                    pattern_dict["last_seen"] = datetime.fromisoformat(pattern_dict["last_seen"])
                    
                    pattern = AccessPattern(**pattern_dict)
                    self.access_patterns[pattern_key] = pattern
                
                logger.info(f"Loaded {len(self.access_patterns)} access patterns")
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_requests"]
        
        memory_usage = self.current_memory_bytes / self.max_memory_bytes
        
        return {
            "smart_cache": {
                "entries": len(self.cache_entries),
                "memory_usage_bytes": self.current_memory_bytes,
                "memory_usage_percent": memory_usage * 100,
                "max_memory_bytes": self.max_memory_bytes,
                **self.stats,
                "hit_rate": hit_rate,
                "patterns_learned": len(self.access_patterns)
            },
            "top_patterns": [
                {
                    "pattern": pattern.key_pattern,
                    "frequency": pattern.frequency,
                    "regularity": pattern.regularity
                }
                for pattern in sorted(
                    self.access_patterns.values(),
                    key=lambda p: p.frequency * p.regularity,
                    reverse=True
                )[:10]
            ]
        }
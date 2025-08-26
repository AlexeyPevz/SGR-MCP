"""Community schema store and management."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


class SchemaStatus(str, Enum):
    """Schema status in marketplace."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    FEATURED = "featured"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class SchemaCategory(str, Enum):
    """Schema categories."""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    DECISION = "decision"
    CODE = "code"
    CREATIVE = "creative"
    BUSINESS = "business"
    RESEARCH = "research"
    EDUCATION = "education"
    HEALTH = "health"
    FINANCE = "finance"
    OTHER = "other"


@dataclass
class SchemaMetadata:
    """Metadata for a community schema."""
    id: str
    name: str
    description: str
    author_id: str
    author_name: str
    organization_id: str
    
    # Classification
    category: SchemaCategory
    tags: List[str]
    use_cases: List[str]
    
    # Content
    schema_definition: Dict[str, Any]
    examples: List[Dict[str, Any]]
    documentation: str
    
    # Status and versioning
    status: SchemaStatus = SchemaStatus.DRAFT
    version: str = "1.0.0"
    parent_schema_id: Optional[str] = None  # For forks/derivatives
    
    # Marketplace data
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    featured_score: float = 0.0
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    published_at: Optional[datetime] = None
    
    # License and permissions
    license: str = "MIT"
    is_public: bool = True
    is_commercial_use: bool = True
    price: float = 0.0  # For premium schemas
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class SchemaReview:
    """Review for a community schema."""
    id: str
    schema_id: str
    reviewer_id: str
    reviewer_name: str
    organization_id: str
    
    rating: int  # 1-5 stars
    title: str
    comment: str
    pros: List[str]
    cons: List[str]
    
    # Verification
    is_verified_purchase: bool = False
    is_expert_review: bool = False
    
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass 
class SchemaUsageStats:
    """Usage statistics for a schema."""
    schema_id: str
    total_uses: int = 0
    unique_users: int = 0
    success_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Time-based stats
    daily_uses: Dict[str, int] = None
    weekly_uses: Dict[str, int] = None
    monthly_uses: Dict[str, int] = None
    
    def __post_init__(self):
        if self.daily_uses is None:
            self.daily_uses = {}
        if self.weekly_uses is None:
            self.weekly_uses = {}
        if self.monthly_uses is None:
            self.monthly_uses = {}


class SchemaStore:
    """Community schema store with marketplace features."""
    
    def __init__(self, cache_manager=None, rbac_manager=None):
        self.cache_manager = cache_manager
        self.rbac_manager = rbac_manager
        
        # In-memory storage (in production, this would be a database)
        self.schemas: Dict[str, SchemaMetadata] = {}
        self.reviews: Dict[str, List[SchemaReview]] = defaultdict(list)
        self.usage_stats: Dict[str, SchemaUsageStats] = {}
        self.user_downloads: Dict[str, List[str]] = defaultdict(list)  # user_id -> [schema_ids]
        
        # Featured and trending
        self.featured_schemas: List[str] = []
        self.trending_schemas: List[str] = []
        
        # Moderation queue
        self.pending_reviews: List[str] = []
        
        # Background tasks
        self._analytics_task = None
    
    async def initialize(self):
        """Initialize the schema store."""
        await self._load_schemas()
        await self._load_featured_schemas()
        
        # Start background analytics
        self._analytics_task = asyncio.create_task(self._background_analytics())
        
        logger.info("Schema store initialized")
    
    async def close(self):
        """Close the schema store."""
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        await self._save_data()
        logger.info("Schema store closed")
    
    async def submit_schema(
        self,
        name: str,
        description: str,
        author_id: str,
        author_name: str,
        organization_id: str,
        category: SchemaCategory,
        schema_definition: Dict[str, Any],
        examples: List[Dict[str, Any]],
        documentation: str,
        tags: Optional[List[str]] = None,
        use_cases: Optional[List[str]] = None,
        license: str = "MIT"
    ) -> str:
        """Submit a new schema to the marketplace."""
        schema_id = str(uuid.uuid4())
        
        schema = SchemaMetadata(
            id=schema_id,
            name=name,
            description=description,
            author_id=author_id,
            author_name=author_name,
            organization_id=organization_id,
            category=category,
            tags=tags or [],
            use_cases=use_cases or [],
            schema_definition=schema_definition,
            examples=examples,
            documentation=documentation,
            license=license,
            status=SchemaStatus.PENDING_REVIEW
        )
        
        # Validate schema definition
        validation_result = await self._validate_schema(schema)
        if not validation_result["valid"]:
            raise ValueError(f"Schema validation failed: {validation_result['errors']}")
        
        # Store schema
        self.schemas[schema_id] = schema
        self.pending_reviews.append(schema_id)
        
        # Initialize stats
        self.usage_stats[schema_id] = SchemaUsageStats(schema_id=schema_id)
        
        # Cache schema
        if self.cache_manager:
            await self.cache_manager.set_cache(f"schema:{schema_id}", asdict(schema), ttl=3600)
        
        logger.info(f"Schema submitted: {schema_id} by {author_name}")
        return schema_id
    
    async def update_schema(
        self,
        schema_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing schema."""
        schema = await self.get_schema(schema_id)
        if not schema:
            return False
        
        # Check permissions
        if schema.author_id != user_id:
            # Check if user is admin
            if self.rbac_manager:
                from ..auth.models import Permission
                has_permission = await self.rbac_manager.check_permission(
                    user_id, Permission.ADMIN_SYSTEM
                )
                if not has_permission:
                    raise PermissionError("Not authorized to update this schema")
        
        # Update fields
        for field, value in updates.items():
            if hasattr(schema, field) and field not in ["id", "author_id", "created_at"]:
                setattr(schema, field, value)
        
        schema.updated_at = datetime.utcnow()
        
        # If content changed, reset to pending review
        content_fields = ["schema_definition", "examples", "documentation"]
        if any(field in updates for field in content_fields):
            schema.status = SchemaStatus.PENDING_REVIEW
            if schema_id not in self.pending_reviews:
                self.pending_reviews.append(schema_id)
        
        self.schemas[schema_id] = schema
        
        # Update cache
        if self.cache_manager:
            await self.cache_manager.set_cache(f"schema:{schema_id}", asdict(schema), ttl=3600)
        
        logger.info(f"Schema updated: {schema_id}")
        return True
    
    async def get_schema(self, schema_id: str) -> Optional[SchemaMetadata]:
        """Get schema by ID."""
        # Try cache first
        if self.cache_manager:
            cached = await self.cache_manager.get_cache(f"schema:{schema_id}")
            if cached:
                return SchemaMetadata(**cached)
        
        # Fallback to memory
        schema = self.schemas.get(schema_id)
        if schema and self.cache_manager:
            await self.cache_manager.set_cache(f"schema:{schema_id}", asdict(schema), ttl=3600)
        
        return schema
    
    async def search_schemas(
        self,
        query: Optional[str] = None,
        category: Optional[SchemaCategory] = None,
        tags: Optional[List[str]] = None,
        author_id: Optional[str] = None,
        min_rating: Optional[float] = None,
        only_approved: bool = True,
        sort_by: str = "rating",  # rating, downloads, created_at, updated_at
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[SchemaMetadata], int]:
        """Search schemas in marketplace."""
        # Filter schemas
        filtered_schemas = []
        
        for schema in self.schemas.values():
            # Status filter
            if only_approved and schema.status not in [SchemaStatus.APPROVED, SchemaStatus.FEATURED]:
                continue
            
            # Category filter
            if category and schema.category != category:
                continue
            
            # Author filter
            if author_id and schema.author_id != author_id:
                continue
            
            # Rating filter
            if min_rating and schema.rating < min_rating:
                continue
            
            # Tags filter
            if tags and not any(tag in schema.tags for tag in tags):
                continue
            
            # Query filter (search in name, description, tags)
            if query:
                query_lower = query.lower()
                searchable_text = f"{schema.name} {schema.description} {' '.join(schema.tags)}".lower()
                if query_lower not in searchable_text:
                    continue
            
            filtered_schemas.append(schema)
        
        # Sort schemas
        if sort_by == "rating":
            filtered_schemas.sort(key=lambda s: (s.rating, s.rating_count), reverse=True)
        elif sort_by == "downloads":
            filtered_schemas.sort(key=lambda s: s.downloads, reverse=True)
        elif sort_by == "created_at":
            filtered_schemas.sort(key=lambda s: s.created_at, reverse=True)
        elif sort_by == "updated_at":
            filtered_schemas.sort(key=lambda s: s.updated_at, reverse=True)
        elif sort_by == "featured":
            # Featured schemas first, then by rating
            filtered_schemas.sort(key=lambda s: (s.featured_score, s.rating), reverse=True)
        
        total_count = len(filtered_schemas)
        
        # Apply pagination
        paginated_schemas = filtered_schemas[offset:offset + limit]
        
        return paginated_schemas, total_count
    
    async def download_schema(self, schema_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Download a schema and update download stats."""
        schema = await self.get_schema(schema_id)
        if not schema:
            return None
        
        # Check if schema is available for download
        if schema.status not in [SchemaStatus.APPROVED, SchemaStatus.FEATURED]:
            raise PermissionError("Schema not available for download")
        
        # Update download stats
        schema.downloads += 1
        self.schemas[schema_id] = schema
        
        # Track user download
        if user_id not in self.user_downloads[user_id]:
            self.user_downloads[user_id].append(schema_id)
        
        # Update usage stats
        if schema_id in self.usage_stats:
            stats = self.usage_stats[schema_id]
            if user_id not in self.user_downloads[user_id][:-1]:  # New user
                stats.unique_users += 1
        
        # Return schema definition for use
        return {
            "schema_id": schema_id,
            "name": schema.name,
            "version": schema.version,
            "schema_definition": schema.schema_definition,
            "examples": schema.examples,
            "documentation": schema.documentation,
            "license": schema.license
        }
    
    async def add_review(
        self,
        schema_id: str,
        reviewer_id: str,
        reviewer_name: str,
        organization_id: str,
        rating: int,
        title: str,
        comment: str,
        pros: Optional[List[str]] = None,
        cons: Optional[List[str]] = None
    ) -> str:
        """Add a review for a schema."""
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        schema = await self.get_schema(schema_id)
        if not schema:
            raise ValueError("Schema not found")
        
        # Check if user has downloaded the schema
        has_downloaded = schema_id in self.user_downloads.get(reviewer_id, [])
        
        review_id = str(uuid.uuid4())
        review = SchemaReview(
            id=review_id,
            schema_id=schema_id,
            reviewer_id=reviewer_id,
            reviewer_name=reviewer_name,
            organization_id=organization_id,
            rating=rating,
            title=title,
            comment=comment,
            pros=pros or [],
            cons=cons or [],
            is_verified_purchase=has_downloaded
        )
        
        self.reviews[schema_id].append(review)
        
        # Update schema rating
        await self._update_schema_rating(schema_id)
        
        logger.info(f"Review added for schema {schema_id} by {reviewer_name}")
        return review_id
    
    async def get_reviews(self, schema_id: str, limit: int = 10) -> List[SchemaReview]:
        """Get reviews for a schema."""
        reviews = self.reviews.get(schema_id, [])
        
        # Sort by rating and recency
        sorted_reviews = sorted(
            reviews,
            key=lambda r: (r.rating, r.created_at),
            reverse=True
        )
        
        return sorted_reviews[:limit]
    
    async def record_usage(
        self,
        schema_id: str,
        user_id: str,
        success: bool,
        quality_score: float,
        latency_ms: float
    ):
        """Record schema usage for analytics."""
        if schema_id not in self.usage_stats:
            self.usage_stats[schema_id] = SchemaUsageStats(schema_id=schema_id)
        
        stats = self.usage_stats[schema_id]
        
        # Update counters
        stats.total_uses += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        if stats.total_uses == 1:
            stats.success_rate = 1.0 if success else 0.0
        else:
            new_success = 1.0 if success else 0.0
            stats.success_rate = (1 - alpha) * stats.success_rate + alpha * new_success
        
        # Update quality score
        if quality_score > 0:
            if stats.total_uses == 1:
                stats.avg_quality_score = quality_score
            else:
                stats.avg_quality_score = (1 - alpha) * stats.avg_quality_score + alpha * quality_score
        
        # Update latency
        if stats.total_uses == 1:
            stats.avg_latency_ms = latency_ms
        else:
            stats.avg_latency_ms = (1 - alpha) * stats.avg_latency_ms + alpha * latency_ms
        
        # Update time-based stats
        today = datetime.utcnow().date().isoformat()
        stats.daily_uses[today] = stats.daily_uses.get(today, 0) + 1
        
        logger.debug(f"Usage recorded for schema {schema_id}")
    
    async def get_featured_schemas(self, limit: int = 10) -> List[SchemaMetadata]:
        """Get featured schemas."""
        featured = []
        
        for schema_id in self.featured_schemas[:limit]:
            schema = await self.get_schema(schema_id)
            if schema and schema.status == SchemaStatus.FEATURED:
                featured.append(schema)
        
        return featured
    
    async def get_trending_schemas(self, limit: int = 10) -> List[SchemaMetadata]:
        """Get trending schemas based on recent activity."""
        # Calculate trending score based on recent downloads and usage
        trending_scores = {}
        
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        for schema_id, schema in self.schemas.items():
            if schema.status not in [SchemaStatus.APPROVED, SchemaStatus.FEATURED]:
                continue
            
            # Recent downloads weight
            recent_download_score = 0
            if schema.updated_at > cutoff_date:
                recent_download_score = schema.downloads * 2
            
            # Usage stats weight
            usage_score = 0
            if schema_id in self.usage_stats:
                stats = self.usage_stats[schema_id]
                # Weight recent usage more heavily
                recent_uses = sum(
                    count for date_str, count in stats.daily_uses.items()
                    if datetime.fromisoformat(date_str).date() > cutoff_date.date()
                )
                usage_score = recent_uses * stats.success_rate * stats.avg_quality_score
            
            # Rating weight
            rating_score = schema.rating * schema.rating_count
            
            # Combined trending score
            trending_scores[schema_id] = recent_download_score + usage_score + rating_score
        
        # Sort by trending score
        trending_ids = sorted(
            trending_scores.keys(),
            key=lambda sid: trending_scores[sid],
            reverse=True
        )
        
        # Get schema objects
        trending = []
        for schema_id in trending_ids[:limit]:
            schema = await self.get_schema(schema_id)
            if schema:
                trending.append(schema)
        
        return trending
    
    async def moderate_schema(
        self,
        schema_id: str,
        moderator_id: str,
        action: SchemaStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Moderate a schema (approve, reject, feature, etc.)."""
        schema = await self.get_schema(schema_id)
        if not schema:
            return False
        
        # Check moderator permissions
        if self.rbac_manager:
            from ..auth.models import Permission
            has_permission = await self.rbac_manager.check_permission(
                moderator_id, Permission.ADMIN_SYSTEM
            )
            if not has_permission:
                raise PermissionError("Not authorized to moderate schemas")
        
        # Update schema status
        old_status = schema.status
        schema.status = action
        schema.updated_at = datetime.utcnow()
        
        if action == SchemaStatus.APPROVED and old_status == SchemaStatus.PENDING_REVIEW:
            schema.published_at = datetime.utcnow()
        
        if action == SchemaStatus.FEATURED:
            schema.featured_score = 1.0
            if schema_id not in self.featured_schemas:
                self.featured_schemas.insert(0, schema_id)
        elif old_status == SchemaStatus.FEATURED:
            schema.featured_score = 0.0
            if schema_id in self.featured_schemas:
                self.featured_schemas.remove(schema_id)
        
        # Remove from pending review
        if schema_id in self.pending_reviews:
            self.pending_reviews.remove(schema_id)
        
        self.schemas[schema_id] = schema
        
        # Update cache
        if self.cache_manager:
            await self.cache_manager.set_cache(f"schema:{schema_id}", asdict(schema), ttl=3600)
        
        logger.info(f"Schema {schema_id} moderated: {old_status} -> {action}")
        return True
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        total_schemas = len(self.schemas)
        approved_schemas = len([s for s in self.schemas.values() if s.status == SchemaStatus.APPROVED])
        featured_schemas = len([s for s in self.schemas.values() if s.status == SchemaStatus.FEATURED])
        pending_schemas = len(self.pending_reviews)
        
        total_downloads = sum(s.downloads for s in self.schemas.values())
        total_reviews = sum(len(reviews) for reviews in self.reviews.values())
        
        # Category distribution
        category_counts = defaultdict(int)
        for schema in self.schemas.values():
            category_counts[schema.category.value] += 1
        
        # Top authors
        author_stats = defaultdict(lambda: {"schemas": 0, "downloads": 0, "rating": 0.0})
        for schema in self.schemas.values():
            author_stats[schema.author_name]["schemas"] += 1
            author_stats[schema.author_name]["downloads"] += schema.downloads
            author_stats[schema.author_name]["rating"] += schema.rating
        
        top_authors = sorted(
            author_stats.items(),
            key=lambda x: (x[1]["schemas"], x[1]["downloads"]),
            reverse=True
        )[:10]
        
        return {
            "total_schemas": total_schemas,
            "approved_schemas": approved_schemas,
            "featured_schemas": featured_schemas,
            "pending_schemas": pending_schemas,
            "total_downloads": total_downloads,
            "total_reviews": total_reviews,
            "avg_rating": sum(s.rating for s in self.schemas.values()) / max(total_schemas, 1),
            "category_distribution": dict(category_counts),
            "top_authors": [{"name": name, **stats} for name, stats in top_authors]
        }
    
    async def _validate_schema(self, schema: SchemaMetadata) -> Dict[str, Any]:
        """Validate a schema definition."""
        errors = []
        
        # Check required fields in schema definition
        required_fields = ["name", "description", "fields"]
        for field in required_fields:
            if field not in schema.schema_definition:
                errors.append(f"Missing required field: {field}")
        
        # Validate examples
        if not schema.examples:
            errors.append("At least one example is required")
        else:
            for i, example in enumerate(schema.examples):
                if "input" not in example or "expected_output" not in example:
                    errors.append(f"Example {i+1} missing input or expected_output")
        
        # Check documentation
        if len(schema.documentation) < 100:
            errors.append("Documentation must be at least 100 characters")
        
        # Validate JSON schema format
        try:
            import jsonschema
            # Basic validation that it's a valid JSON schema structure
            jsonschema.Draft7Validator.check_schema(schema.schema_definition)
        except Exception as e:
            errors.append(f"Invalid schema format: {str(e)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _update_schema_rating(self, schema_id: str):
        """Update schema rating based on reviews."""
        reviews = self.reviews.get(schema_id, [])
        if not reviews:
            return
        
        schema = self.schemas.get(schema_id)
        if not schema:
            return
        
        # Calculate weighted average rating
        total_rating = 0
        total_weight = 0
        
        for review in reviews:
            weight = 1.0
            
            # Give more weight to verified purchases
            if review.is_verified_purchase:
                weight += 0.5
            
            # Give more weight to expert reviews
            if review.is_expert_review:
                weight += 1.0
            
            total_rating += review.rating * weight
            total_weight += weight
        
        if total_weight > 0:
            schema.rating = total_rating / total_weight
            schema.rating_count = len(reviews)
            
            # Update featured score
            if schema.rating > 4.0 and schema.rating_count >= 5:
                schema.featured_score += 0.1
        
        self.schemas[schema_id] = schema
    
    async def _background_analytics(self):
        """Background task for analytics and trending calculations."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Update trending schemas
                trending = await self.get_trending_schemas(50)
                self.trending_schemas = [s.id for s in trending]
                
                # Clean old daily stats
                cutoff_date = datetime.utcnow().date() - timedelta(days=30)
                for stats in self.usage_stats.values():
                    stats.daily_uses = {
                        date_str: count for date_str, count in stats.daily_uses.items()
                        if datetime.fromisoformat(date_str).date() > cutoff_date
                    }
                
                logger.debug("Marketplace analytics updated")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in marketplace analytics: {e}")
    
    async def _load_schemas(self):
        """Load schemas from persistent storage."""
        if not self.cache_manager:
            return
        
        try:
            schemas_data = await self.cache_manager.get_cache("marketplace_schemas")
            if schemas_data:
                for schema_id, schema_dict in schemas_data.items():
                    # Convert datetime strings back to datetime objects
                    if "created_at" in schema_dict:
                        schema_dict["created_at"] = datetime.fromisoformat(schema_dict["created_at"])
                    if "updated_at" in schema_dict:
                        schema_dict["updated_at"] = datetime.fromisoformat(schema_dict["updated_at"])
                    if "published_at" in schema_dict and schema_dict["published_at"]:
                        schema_dict["published_at"] = datetime.fromisoformat(schema_dict["published_at"])
                    
                    schema = SchemaMetadata(**schema_dict)
                    self.schemas[schema_id] = schema
                
                logger.info(f"Loaded {len(self.schemas)} schemas from storage")
        except Exception as e:
            logger.error(f"Failed to load schemas: {e}")
    
    async def _load_featured_schemas(self):
        """Load featured schemas list."""
        if not self.cache_manager:
            return
        
        try:
            featured_data = await self.cache_manager.get_cache("marketplace_featured")
            if featured_data:
                self.featured_schemas = featured_data
                logger.info(f"Loaded {len(self.featured_schemas)} featured schemas")
        except Exception as e:
            logger.error(f"Failed to load featured schemas: {e}")
    
    async def _save_data(self):
        """Save marketplace data to persistent storage."""
        if not self.cache_manager:
            return
        
        try:
            # Save schemas
            schemas_data = {}
            for schema_id, schema in self.schemas.items():
                schema_dict = asdict(schema)
                # Convert datetime objects to strings
                if schema_dict["created_at"]:
                    schema_dict["created_at"] = schema_dict["created_at"].isoformat()
                if schema_dict["updated_at"]:
                    schema_dict["updated_at"] = schema_dict["updated_at"].isoformat()
                if schema_dict["published_at"]:
                    schema_dict["published_at"] = schema_dict["published_at"].isoformat()
                
                schemas_data[schema_id] = schema_dict
            
            await self.cache_manager.set_cache(
                "marketplace_schemas",
                schemas_data,
                ttl=86400 * 30  # 30 days
            )
            
            # Save featured schemas
            await self.cache_manager.set_cache(
                "marketplace_featured",
                self.featured_schemas,
                ttl=86400 * 7  # 7 days
            )
            
            logger.debug("Marketplace data saved")
            
        except Exception as e:
            logger.error(f"Failed to save marketplace data: {e}")
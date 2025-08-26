"""Adaptive routing system that learns from request patterns and performance."""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Routing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    PERFORMANCE_BASED = "performance_based" 
    COST_OPTIMIZED = "cost_optimized"
    ML_PREDICTED = "ml_predicted"


@dataclass
class RouteMetrics:
    """Metrics for a specific route."""
    total_requests: int = 0
    successful_requests: int = 0
    avg_latency_ms: float = 0.0
    avg_cost: float = 0.0
    avg_quality_score: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = None
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def performance_score(self) -> float:
        """Calculate overall performance score (0-1, higher is better)."""
        # Combine success rate, quality, and inverse of latency/cost
        success_weight = 0.4
        quality_weight = 0.3
        speed_weight = 0.2
        cost_weight = 0.1
        
        success_score = self.success_rate()
        quality_score = min(self.avg_quality_score, 1.0)
        
        # Normalize latency (assume 10s is max acceptable)
        speed_score = max(0, 1 - (self.avg_latency_ms / 10000))
        
        # Normalize cost (assume $1 per request is max)
        cost_score = max(0, 1 - self.avg_cost)
        
        return (
            success_weight * success_score +
            quality_weight * quality_score +
            speed_weight * speed_score +
            cost_weight * cost_score
        )


@dataclass
class RequestPattern:
    """Pattern extracted from requests."""
    schema_type: str
    task_complexity: str  # simple, medium, complex
    time_of_day: int  # hour 0-23
    user_tier: str  # free, pro, enterprise
    avg_latency_requirement: float  # in ms
    cost_sensitivity: float  # 0-1, higher means more cost sensitive


class AdaptiveRouter:
    """Adaptive routing system with ML-based optimization."""
    
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.route_metrics: Dict[str, RouteMetrics] = {}
        self.request_history: deque = deque(maxlen=10000)
        self.pattern_cache: Dict[str, List[RequestPattern]] = {}
        
        # Configuration
        self.learning_rate = 0.1
        self.exploration_rate = 0.1  # For epsilon-greedy exploration
        self.min_requests_for_learning = 10
        self.metrics_window_hours = 24
        
        # Background learning task
        self._learning_task = None
        
    async def initialize(self):
        """Initialize the adaptive router."""
        # Load historical data
        await self._load_historical_data()
        
        # Start background learning
        self._learning_task = asyncio.create_task(self._background_learning())
        
        logger.info("Adaptive router initialized")
    
    async def close(self):
        """Close the adaptive router."""
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        
        # Save current state
        await self._save_state()
        logger.info("Adaptive router closed")
    
    async def route_request(
        self,
        task: str,
        schema_type: str,
        context: Dict[str, Any],
        available_backends: List[str],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, float]:
        """
        Route request to optimal backend/model.
        
        Returns:
            Tuple of (backend, model, confidence_score)
        """
        start_time = time.time()
        
        # Extract request features
        features = self._extract_features(task, schema_type, context, user_preferences)
        
        # Get routing strategy
        strategy = self._select_strategy(features, available_backends)
        
        # Route based on strategy
        if strategy == RoutingStrategy.ML_PREDICTED:
            backend, model, confidence = await self._ml_route(features, available_backends)
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            backend, model, confidence = await self._performance_route(features, available_backends)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            backend, model, confidence = await self._cost_route(features, available_backends)
        else:
            backend, model, confidence = await self._weighted_route(features, available_backends)
        
        # Log routing decision
        routing_time = (time.time() - start_time) * 1000
        await self._log_routing_decision(features, backend, model, strategy, routing_time)
        
        return backend, model, confidence
    
    async def record_result(
        self,
        backend: str,
        model: str,
        latency_ms: float,
        success: bool,
        cost: float = 0.0,
        quality_score: float = 0.0,
        error_type: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None
    ):
        """Record the result of a routed request for learning."""
        route_key = f"{backend}:{model}"
        
        # Update route metrics
        if route_key not in self.route_metrics:
            self.route_metrics[route_key] = RouteMetrics()
        
        metrics = self.route_metrics[route_key]
        
        # Exponential moving average for continuous metrics
        alpha = self.learning_rate
        
        if metrics.total_requests == 0:
            # First request
            metrics.avg_latency_ms = latency_ms
            metrics.avg_cost = cost
            metrics.avg_quality_score = quality_score
        else:
            # Update averages
            metrics.avg_latency_ms = (1 - alpha) * metrics.avg_latency_ms + alpha * latency_ms
            metrics.avg_cost = (1 - alpha) * metrics.avg_cost + alpha * cost
            if quality_score > 0:
                metrics.avg_quality_score = (1 - alpha) * metrics.avg_quality_score + alpha * quality_score
        
        # Update counters
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        
        # Update error rate
        metrics.error_rate = 1 - (metrics.successful_requests / metrics.total_requests)
        metrics.last_updated = datetime.utcnow()
        
        # Add to request history for pattern learning
        self.request_history.append({
            "timestamp": datetime.utcnow(),
            "backend": backend,
            "model": model,
            "latency_ms": latency_ms,
            "success": success,
            "cost": cost,
            "quality_score": quality_score,
            "error_type": error_type,
            "context": request_context or {}
        })
        
        logger.debug(f"Recorded result for {route_key}: latency={latency_ms}ms, success={success}")
    
    def _extract_features(
        self,
        task: str,
        schema_type: str,
        context: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract features from request for routing decision."""
        features = {
            "task_length": len(task),
            "schema_type": schema_type,
            "hour_of_day": datetime.utcnow().hour,
            "day_of_week": datetime.utcnow().weekday(),
            "context_size": len(str(context)),
            "has_custom_schema": context.get("custom_schema") is not None,
            "budget": context.get("budget", "lite"),
            "user_tier": user_preferences.get("tier", "free") if user_preferences else "free",
            "priority": user_preferences.get("priority", "normal") if user_preferences else "normal",
            "cost_sensitivity": user_preferences.get("cost_sensitivity", 0.5) if user_preferences else 0.5,
            "latency_tolerance": user_preferences.get("latency_tolerance", "medium") if user_preferences else "medium"
        }
        
        # Classify task complexity
        features["task_complexity"] = self._classify_task_complexity(task, context)
        
        return features
    
    def _classify_task_complexity(self, task: str, context: Dict[str, Any]) -> str:
        """Classify task complexity based on content."""
        # Simple heuristics - in production this could be ML-based
        task_lower = task.lower()
        
        # Complex indicators
        complex_words = [
            "analyze", "design", "architecture", "system", "complex", "detailed",
            "comprehensive", "multi-step", "algorithm", "optimization"
        ]
        
        # Simple indicators  
        simple_words = [
            "summarize", "list", "simple", "quick", "basic", "explain"
        ]
        
        complex_score = sum(1 for word in complex_words if word in task_lower)
        simple_score = sum(1 for word in simple_words if word in task_lower)
        
        task_length = len(task)
        context_size = len(str(context))
        
        if complex_score > simple_score or task_length > 500 or context_size > 1000:
            return "complex"
        elif simple_score > 0 and task_length < 100:
            return "simple"
        else:
            return "medium"
    
    def _select_strategy(
        self,
        features: Dict[str, Any],
        available_backends: List[str]
    ) -> RoutingStrategy:
        """Select routing strategy based on current conditions."""
        # Check if we have enough data for ML routing
        total_requests = sum(m.total_requests for m in self.route_metrics.values())
        
        if total_requests < self.min_requests_for_learning:
            return RoutingStrategy.WEIGHTED
        
        # Use ML for complex tasks with sufficient data
        if features.get("task_complexity") == "complex":
            return RoutingStrategy.ML_PREDICTED
        
        # Cost optimization for cost-sensitive users
        if features.get("cost_sensitivity", 0) > 0.7:
            return RoutingStrategy.COST_OPTIMIZED
        
        # Performance-based for high-priority requests
        if features.get("priority") == "high":
            return RoutingStrategy.PERFORMANCE_BASED
        
        # Default to ML if we have enough data
        return RoutingStrategy.ML_PREDICTED
    
    async def _ml_route(
        self,
        features: Dict[str, Any],
        available_backends: List[str]
    ) -> Tuple[str, str, float]:
        """ML-based routing using simple scoring."""
        best_route = None
        best_score = -1
        confidence = 0.0
        
        # Score each available route
        for backend in available_backends:
            models = self._get_models_for_backend(backend)
            for model in models:
                route_key = f"{backend}:{model}"
                
                if route_key in self.route_metrics:
                    metrics = self.route_metrics[route_key]
                    
                    # Calculate compatibility score based on features
                    compatibility = self._calculate_compatibility(features, backend, model)
                    performance = metrics.performance_score()
                    
                    # Combined score with exploration
                    score = 0.7 * performance + 0.3 * compatibility
                    
                    # Add exploration bonus for under-explored routes
                    if metrics.total_requests < 5:
                        score += self.exploration_rate
                    
                    if score > best_score:
                        best_score = score
                        best_route = (backend, model)
                        confidence = min(score, 1.0)
        
        if best_route:
            return best_route[0], best_route[1], confidence
        
        # Fallback to first available
        models = self._get_models_for_backend(available_backends[0])
        return available_backends[0], models[0], 0.1
    
    async def _performance_route(
        self,
        features: Dict[str, Any],
        available_backends: List[str]
    ) -> Tuple[str, str, float]:
        """Route based on performance metrics."""
        best_route = None
        best_performance = -1
        
        for backend in available_backends:
            models = self._get_models_for_backend(backend)
            for model in models:
                route_key = f"{backend}:{model}"
                
                if route_key in self.route_metrics:
                    metrics = self.route_metrics[route_key]
                    performance = metrics.performance_score()
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_route = (backend, model)
        
        if best_route:
            return best_route[0], best_route[1], best_performance
        
        # Fallback
        models = self._get_models_for_backend(available_backends[0])
        return available_backends[0], models[0], 0.1
    
    async def _cost_route(
        self,
        features: Dict[str, Any],
        available_backends: List[str]
    ) -> Tuple[str, str, float]:
        """Route based on cost optimization."""
        best_route = None
        best_cost_score = -1
        
        for backend in available_backends:
            models = self._get_models_for_backend(backend)
            for model in models:
                route_key = f"{backend}:{model}"
                
                if route_key in self.route_metrics:
                    metrics = self.route_metrics[route_key]
                    
                    # Cost score: lower cost is better, but need minimum success rate
                    if metrics.success_rate() > 0.8:  # Minimum success threshold
                        cost_score = 1.0 / (1.0 + metrics.avg_cost)  # Inverse cost
                        
                        if cost_score > best_cost_score:
                            best_cost_score = cost_score
                            best_route = (backend, model)
        
        if best_route:
            return best_route[0], best_route[1], best_cost_score
        
        # Fallback to cheapest known route
        cheapest_route = min(
            [(k, v.avg_cost) for k, v in self.route_metrics.items() if v.success_rate() > 0.5],
            key=lambda x: x[1],
            default=(None, None)
        )
        
        if cheapest_route[0]:
            backend, model = cheapest_route[0].split(":", 1)
            return backend, model, 0.7
        
        # Ultimate fallback
        models = self._get_models_for_backend(available_backends[0])
        return available_backends[0], models[0], 0.1
    
    async def _weighted_route(
        self,
        features: Dict[str, Any],
        available_backends: List[str]
    ) -> Tuple[str, str, float]:
        """Weighted routing based on historical performance."""
        weights = []
        routes = []
        
        for backend in available_backends:
            models = self._get_models_for_backend(backend)
            for model in models:
                route_key = f"{backend}:{model}"
                
                if route_key in self.route_metrics:
                    metrics = self.route_metrics[route_key]
                    weight = metrics.performance_score()
                else:
                    weight = 0.1  # Default weight for unknown routes
                
                weights.append(weight)
                routes.append((backend, model))
        
        if not routes:
            models = self._get_models_for_backend(available_backends[0])
            return available_backends[0], models[0], 0.1
        
        # Weighted random selection
        if sum(weights) > 0:
            weights = np.array(weights)
            weights = weights / sum(weights)  # Normalize
            
            selected_idx = np.random.choice(len(routes), p=weights)
            selected_route = routes[selected_idx]
            confidence = weights[selected_idx]
            
            return selected_route[0], selected_route[1], float(confidence)
        
        # Equal probability fallback
        selected_route = routes[np.random.randint(len(routes))]
        return selected_route[0], selected_route[1], 0.5
    
    def _calculate_compatibility(
        self,
        features: Dict[str, Any],
        backend: str,
        model: str
    ) -> float:
        """Calculate compatibility score between request features and backend/model."""
        compatibility = 0.5  # Base score
        
        # Schema type compatibility
        schema_type = features.get("schema_type", "")
        if "code" in schema_type and "gpt" in model.lower():
            compatibility += 0.2
        elif "analysis" in schema_type and "claude" in model.lower():
            compatibility += 0.2
        
        # Task complexity compatibility
        complexity = features.get("task_complexity", "medium")
        if complexity == "complex" and ("72b" in model or "large" in model):
            compatibility += 0.2
        elif complexity == "simple" and ("7b" in model or "small" in model):
            compatibility += 0.1
        
        # Budget compatibility
        budget = features.get("budget", "lite")
        if budget == "lite" and backend == "ollama":
            compatibility += 0.1
        elif budget == "full" and backend == "openrouter":
            compatibility += 0.1
        
        return min(compatibility, 1.0)
    
    def _get_models_for_backend(self, backend: str) -> List[str]:
        """Get available models for backend."""
        # This would normally query the backend for available models
        # For now, return some defaults
        backend_models = {
            "ollama": ["llama3.1:8b", "llama3.1:70b", "qwen2.5:7b"],
            "openrouter": [
                "meta-llama/llama-3.1-8b-instruct",
                "qwen/qwen-2.5-72b-instruct", 
                "anthropic/claude-3-haiku",
                "openai/gpt-4o-mini"
            ],
            "vllm": ["llama3.1-8b", "qwen2.5-7b"]
        }
        
        return backend_models.get(backend, ["default"])
    
    async def _log_routing_decision(
        self,
        features: Dict[str, Any],
        backend: str,
        model: str,
        strategy: RoutingStrategy,
        routing_time_ms: float
    ):
        """Log routing decision for analysis."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "features": features,
            "selected_backend": backend,
            "selected_model": model,
            "strategy": strategy.value,
            "routing_time_ms": routing_time_ms
        }
        
        logger.debug(f"Routing decision: {backend}:{model} via {strategy.value}")
        
        # Cache for analysis
        if self.cache_manager:
            await self.cache_manager.set_cache(
                f"routing_decision:{int(time.time())}", 
                log_entry,
                ttl=86400
            )
    
    async def _background_learning(self):
        """Background task for continuous learning."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._update_patterns()
                await self._cleanup_old_data()
                await self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background learning: {e}")
    
    async def _update_patterns(self):
        """Update learned patterns from request history."""
        if len(self.request_history) < 10:
            return
        
        # Analyze recent patterns
        recent_requests = [
            req for req in self.request_history 
            if req["timestamp"] > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if not recent_requests:
            return
        
        # Group by hour and analyze success rates
        hourly_patterns = defaultdict(list)
        for req in recent_requests:
            hour = req["timestamp"].hour
            hourly_patterns[hour].append(req)
        
        # Update routing preferences based on patterns
        for hour, requests in hourly_patterns.items():
            success_by_route = defaultdict(list)
            for req in requests:
                route_key = f"{req['backend']}:{req['model']}"
                success_by_route[route_key].append(req["success"])
            
            # Log patterns
            for route, successes in success_by_route.items():
                success_rate = sum(successes) / len(successes)
                logger.debug(f"Hour {hour}: {route} success rate: {success_rate:.2f}")
    
    async def _cleanup_old_data(self):
        """Cleanup old metrics and data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_window_hours)
        
        # Remove old entries from request history
        self.request_history = deque(
            [req for req in self.request_history if req["timestamp"] > cutoff_time],
            maxlen=self.request_history.maxlen
        )
        
        logger.debug("Cleaned up old routing data")
    
    async def _save_state(self):
        """Save current state to cache."""
        if not self.cache_manager:
            return
        
        try:
            state = {
                "route_metrics": {
                    k: {
                        "total_requests": v.total_requests,
                        "successful_requests": v.successful_requests,
                        "avg_latency_ms": v.avg_latency_ms,
                        "avg_cost": v.avg_cost,
                        "avg_quality_score": v.avg_quality_score,
                        "error_rate": v.error_rate,
                        "last_updated": v.last_updated.isoformat() if v.last_updated else None
                    }
                    for k, v in self.route_metrics.items()
                },
                "last_saved": datetime.utcnow().isoformat()
            }
            
            await self.cache_manager.set_cache(
                "adaptive_router_state",
                state,
                ttl=86400 * 7  # 7 days
            )
            
            logger.debug("Saved adaptive router state")
        except Exception as e:
            logger.error(f"Failed to save router state: {e}")
    
    async def _load_historical_data(self):
        """Load historical data from cache."""
        if not self.cache_manager:
            return
        
        try:
            state = await self.cache_manager.get_cache("adaptive_router_state")
            if state:
                # Restore route metrics
                for route_key, metrics_data in state.get("route_metrics", {}).items():
                    metrics = RouteMetrics(
                        total_requests=metrics_data["total_requests"],
                        successful_requests=metrics_data["successful_requests"],
                        avg_latency_ms=metrics_data["avg_latency_ms"],
                        avg_cost=metrics_data["avg_cost"],
                        avg_quality_score=metrics_data["avg_quality_score"],
                        error_rate=metrics_data["error_rate"],
                        last_updated=datetime.fromisoformat(metrics_data["last_updated"]) if metrics_data["last_updated"] else None
                    )
                    self.route_metrics[route_key] = metrics
                
                logger.info(f"Loaded {len(self.route_metrics)} route metrics from cache")
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
    
    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_requests = sum(m.total_requests for m in self.route_metrics.values())
        
        route_stats = {}
        for route_key, metrics in self.route_metrics.items():
            route_stats[route_key] = {
                "requests": metrics.total_requests,
                "success_rate": metrics.success_rate(),
                "avg_latency_ms": metrics.avg_latency_ms,
                "avg_cost": metrics.avg_cost,
                "performance_score": metrics.performance_score()
            }
        
        return {
            "total_requests": total_requests,
            "total_routes": len(self.route_metrics),
            "route_stats": route_stats,
            "last_updated": max(
                (m.last_updated for m in self.route_metrics.values() if m.last_updated),
                default=datetime.utcnow()
            ).isoformat()
        }
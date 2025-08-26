"""AI-powered cost optimization for LLM usage."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Cost optimization strategies."""
    COST_FIRST = "cost_first"          # Minimize cost above all
    BALANCED = "balanced"              # Balance cost and quality
    QUALITY_FIRST = "quality_first"    # Prioritize quality over cost
    ADAPTIVE = "adaptive"              # Adapt based on usage patterns


@dataclass
class ModelCostProfile:
    """Cost and performance profile for a model."""
    model_name: str
    backend: str
    
    # Cost metrics
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    cost_per_request: float  # Fixed cost if any
    
    # Performance metrics
    avg_latency_ms: float
    avg_quality_score: float
    reliability_score: float  # 0-1, based on success rate
    
    # Usage statistics
    total_requests: int = 0
    total_cost: float = 0.0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    
    def cost_per_token(self) -> float:
        """Average cost per token (input + output)."""
        total_tokens = self.total_tokens_input + self.total_tokens_output
        if total_tokens == 0:
            return self.cost_per_1k_input_tokens / 1000
        return self.total_cost / total_tokens
    
    def efficiency_score(self) -> float:
        """Efficiency score: quality per dollar."""
        if self.total_cost == 0:
            return self.avg_quality_score
        return self.avg_quality_score / (self.total_cost / max(self.total_requests, 1))


@dataclass
class CostBudget:
    """Cost budget configuration."""
    organization_id: str
    monthly_budget: float
    current_spend: float = 0.0
    remaining_budget: float = 0.0
    
    # Alerts
    alert_thresholds: List[float] = None  # [0.5, 0.8, 0.95] for 50%, 80%, 95%
    alerts_sent: List[float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = [0.5, 0.8, 0.95]
        if self.alerts_sent is None:
            self.alerts_sent = []
        self.remaining_budget = self.monthly_budget - self.current_spend
    
    def usage_percentage(self) -> float:
        """Current budget usage percentage."""
        if self.monthly_budget == 0:
            return 0.0
        return self.current_spend / self.monthly_budget
    
    def should_alert(self) -> Optional[float]:
        """Check if an alert should be sent."""
        usage_pct = self.usage_percentage()
        
        for threshold in self.alert_thresholds:
            if usage_pct >= threshold and threshold not in self.alerts_sent:
                return threshold
        
        return None


class CostOptimizer:
    """AI-powered cost optimization system."""
    
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        
        # Model profiles and costs
        self.model_profiles: Dict[str, ModelCostProfile] = {}
        self.cost_budgets: Dict[str, CostBudget] = {}
        
        # Usage tracking
        self.usage_history: deque = deque(maxlen=10000)
        self.cost_predictions: Dict[str, float] = {}
        
        # Optimization parameters
        self.learning_window_days = 30
        self.prediction_horizon_days = 7
        self.cost_savings_target = 0.2  # 20% cost reduction target
        
        # Default model costs (updated from real usage)
        self._initialize_default_costs()
        
        # Background tasks
        self._optimization_task = None
    
    def _initialize_default_costs(self):
        """Initialize default cost profiles for known models."""
        default_profiles = [
            # OpenRouter models
            ModelCostProfile(
                model_name="meta-llama/llama-3.1-8b-instruct",
                backend="openrouter",
                cost_per_1k_input_tokens=0.18,
                cost_per_1k_output_tokens=0.18,
                cost_per_request=0.0,
                avg_latency_ms=2000,
                avg_quality_score=0.75,
                reliability_score=0.95
            ),
            ModelCostProfile(
                model_name="qwen/qwen-2.5-72b-instruct",
                backend="openrouter",
                cost_per_1k_input_tokens=0.8,
                cost_per_1k_output_tokens=0.8,
                cost_per_request=0.0,
                avg_latency_ms=3000,
                avg_quality_score=0.9,
                reliability_score=0.92
            ),
            ModelCostProfile(
                model_name="anthropic/claude-3-haiku",
                backend="openrouter",
                cost_per_1k_input_tokens=0.25,
                cost_per_1k_output_tokens=1.25,
                cost_per_request=0.0,
                avg_latency_ms=1500,
                avg_quality_score=0.85,
                reliability_score=0.98
            ),
            ModelCostProfile(
                model_name="openai/gpt-4o-mini",
                backend="openrouter",
                cost_per_1k_input_tokens=0.15,
                cost_per_1k_output_tokens=0.6,
                cost_per_request=0.0,
                avg_latency_ms=1800,
                avg_quality_score=0.88,
                reliability_score=0.96
            ),
            # Free models
            ModelCostProfile(
                model_name="llama3.1:8b",
                backend="ollama",
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                cost_per_request=0.0,
                avg_latency_ms=2500,
                avg_quality_score=0.7,
                reliability_score=0.9
            ),
            ModelCostProfile(
                model_name="mistralai/mistral-7b-instruct:free",
                backend="openrouter",
                cost_per_1k_input_tokens=0.0,
                cost_per_1k_output_tokens=0.0,
                cost_per_request=0.0,
                avg_latency_ms=3000,
                avg_quality_score=0.65,
                reliability_score=0.88
            )
        ]
        
        for profile in default_profiles:
            key = f"{profile.backend}:{profile.model_name}"
            self.model_profiles[key] = profile
    
    async def initialize(self):
        """Initialize the cost optimizer."""
        await self._load_historical_data()
        
        # Start background optimization
        self._optimization_task = asyncio.create_task(self._background_optimization())
        
        logger.info("Cost optimizer initialized")
    
    async def close(self):
        """Close the cost optimizer."""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        await self._save_data()
        logger.info("Cost optimizer closed")
    
    async def recommend_model(
        self,
        task_complexity: str,
        quality_requirement: float,  # 0-1
        latency_requirement: float,  # max ms
        organization_id: str,
        available_models: List[Tuple[str, str]],  # [(backend, model), ...]
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Recommend the best model based on optimization strategy.
        
        Returns:
            Tuple of (backend, model, recommendation_details)
        """
        # Get current budget status
        budget = self.cost_budgets.get(organization_id)
        
        # Score each available model
        model_scores = []
        
        for backend, model in available_models:
            profile_key = f"{backend}:{model}"
            
            if profile_key not in self.model_profiles:
                # Create basic profile for unknown model
                await self._create_basic_profile(backend, model)
            
            profile = self.model_profiles[profile_key]
            score = await self._score_model(
                profile, task_complexity, quality_requirement, 
                latency_requirement, strategy, budget
            )
            
            model_scores.append((backend, model, score, profile))
        
        # Sort by score (highest first)
        model_scores.sort(key=lambda x: x[2], reverse=True)
        
        if not model_scores:
            raise ValueError("No available models to recommend")
        
        best_backend, best_model, best_score, best_profile = model_scores[0]
        
        # Calculate cost prediction
        estimated_cost = await self._estimate_request_cost(
            best_profile, task_complexity
        )
        
        recommendation_details = {
            "score": best_score,
            "estimated_cost": estimated_cost,
            "estimated_latency_ms": best_profile.avg_latency_ms,
            "quality_score": best_profile.avg_quality_score,
            "efficiency_score": best_profile.efficiency_score(),
            "strategy_used": strategy.value,
            "alternatives": [
                {
                    "backend": backend,
                    "model": model,
                    "score": score,
                    "estimated_cost": await self._estimate_request_cost(profile, task_complexity)
                }
                for backend, model, score, profile in model_scores[1:5]  # Top 5 alternatives
            ]
        }
        
        return best_backend, best_model, recommendation_details
    
    async def record_usage(
        self,
        backend: str,
        model: str,
        organization_id: str,
        input_tokens: int,
        output_tokens: int,
        actual_cost: float,
        latency_ms: float,
        quality_score: float,
        success: bool,
        task_complexity: str = "medium"
    ):
        """Record actual usage for learning and optimization."""
        profile_key = f"{backend}:{model}"
        
        # Update model profile
        if profile_key not in self.model_profiles:
            await self._create_basic_profile(backend, model)
        
        profile = self.model_profiles[profile_key]
        
        # Update profile statistics
        alpha = 0.1  # Learning rate
        
        profile.total_requests += 1
        profile.total_cost += actual_cost
        profile.total_tokens_input += input_tokens
        profile.total_tokens_output += output_tokens
        
        # Update averages
        if profile.total_requests == 1:
            profile.avg_latency_ms = latency_ms
            profile.avg_quality_score = quality_score
        else:
            profile.avg_latency_ms = (1 - alpha) * profile.avg_latency_ms + alpha * latency_ms
            if quality_score > 0:
                profile.avg_quality_score = (1 - alpha) * profile.avg_quality_score + alpha * quality_score
        
        # Update reliability
        success_rate = sum(1 for usage in self.usage_history 
                          if usage.get("backend") == backend and usage.get("model") == model and usage.get("success", True)
                          ) / max(profile.total_requests, 1)
        profile.reliability_score = success_rate
        
        # Update cost rates based on actual usage
        if input_tokens > 0:
            actual_input_rate = (actual_cost * 0.5) / (input_tokens / 1000)  # Assume 50% of cost is input
            profile.cost_per_1k_input_tokens = (1 - alpha) * profile.cost_per_1k_input_tokens + alpha * actual_input_rate
        
        if output_tokens > 0:
            actual_output_rate = (actual_cost * 0.5) / (output_tokens / 1000)  # Assume 50% of cost is output
            profile.cost_per_1k_output_tokens = (1 - alpha) * profile.cost_per_1k_output_tokens + alpha * actual_output_rate
        
        # Update budget
        if organization_id in self.cost_budgets:
            budget = self.cost_budgets[organization_id]
            budget.current_spend += actual_cost
            budget.remaining_budget = budget.monthly_budget - budget.current_spend
            
            # Check for budget alerts
            alert_threshold = budget.should_alert()
            if alert_threshold:
                await self._send_budget_alert(organization_id, budget, alert_threshold)
                budget.alerts_sent.append(alert_threshold)
        
        # Record usage history
        usage_record = {
            "timestamp": datetime.utcnow(),
            "backend": backend,
            "model": model,
            "organization_id": organization_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "actual_cost": actual_cost,
            "latency_ms": latency_ms,
            "quality_score": quality_score,
            "success": success,
            "task_complexity": task_complexity
        }
        
        self.usage_history.append(usage_record)
        
        logger.debug(f"Recorded usage: {backend}:{model}, cost: ${actual_cost:.4f}")
    
    async def set_budget(self, organization_id: str, monthly_budget: float):
        """Set monthly budget for organization."""
        if organization_id in self.cost_budgets:
            budget = self.cost_budgets[organization_id]
            budget.monthly_budget = monthly_budget
            budget.remaining_budget = monthly_budget - budget.current_spend
        else:
            self.cost_budgets[organization_id] = CostBudget(
                organization_id=organization_id,
                monthly_budget=monthly_budget
            )
        
        logger.info(f"Set budget for {organization_id}: ${monthly_budget:.2f}/month")
    
    async def get_cost_insights(self, organization_id: str) -> Dict[str, Any]:
        """Get cost insights and recommendations for organization."""
        # Current budget status
        budget_info = {}
        if organization_id in self.cost_budgets:
            budget = self.cost_budgets[organization_id]
            budget_info = {
                "monthly_budget": budget.monthly_budget,
                "current_spend": budget.current_spend,
                "remaining_budget": budget.remaining_budget,
                "usage_percentage": budget.usage_percentage(),
                "days_remaining_in_month": (30 - datetime.utcnow().day),
                "projected_monthly_spend": budget.current_spend * (30 / max(datetime.utcnow().day, 1))
            }
        
        # Usage by model
        org_usage = [u for u in self.usage_history if u.get("organization_id") == organization_id]
        
        model_usage = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
        for usage in org_usage:
            key = f"{usage['backend']}:{usage['model']}"
            model_usage[key]["requests"] += 1
            model_usage[key]["cost"] += usage["actual_cost"]
            model_usage[key]["tokens"] += usage["input_tokens"] + usage["output_tokens"]
        
        # Cost optimization recommendations
        recommendations = await self._generate_cost_recommendations(organization_id, org_usage)
        
        # Trend analysis
        daily_costs = self._calculate_daily_costs(org_usage)
        
        return {
            "budget": budget_info,
            "usage_by_model": dict(model_usage),
            "daily_costs": daily_costs,
            "recommendations": recommendations,
            "total_requests": len(org_usage),
            "total_cost": sum(u["actual_cost"] for u in org_usage),
            "avg_cost_per_request": sum(u["actual_cost"] for u in org_usage) / max(len(org_usage), 1)
        }
    
    async def _score_model(
        self,
        profile: ModelCostProfile,
        task_complexity: str,
        quality_requirement: float,
        latency_requirement: float,
        strategy: OptimizationStrategy,
        budget: Optional[CostBudget]
    ) -> float:
        """Score a model based on optimization strategy."""
        # Base scores (0-1)
        cost_score = 1.0 - min(profile.cost_per_token() / 0.01, 1.0)  # $0.01/token as max
        quality_score = profile.avg_quality_score
        latency_score = max(0, 1.0 - (profile.avg_latency_ms / latency_requirement)) if latency_requirement > 0 else 1.0
        reliability_score = profile.reliability_score
        
        # Task complexity adjustment
        complexity_bonus = 0.0
        if task_complexity == "complex" and profile.avg_quality_score > 0.8:
            complexity_bonus = 0.1
        elif task_complexity == "simple" and profile.cost_per_token() < 0.001:
            complexity_bonus = 0.1
        
        # Budget pressure adjustment
        budget_pressure = 0.0
        if budget:
            usage_pct = budget.usage_percentage()
            if usage_pct > 0.8:  # High budget pressure
                budget_pressure = 0.2  # Favor cheaper models
            elif usage_pct > 0.6:
                budget_pressure = 0.1
        
        # Strategy-based weighting
        if strategy == OptimizationStrategy.COST_FIRST:
            score = 0.6 * cost_score + 0.2 * quality_score + 0.1 * latency_score + 0.1 * reliability_score
        elif strategy == OptimizationStrategy.QUALITY_FIRST:
            score = 0.1 * cost_score + 0.6 * quality_score + 0.15 * latency_score + 0.15 * reliability_score
        elif strategy == OptimizationStrategy.BALANCED:
            score = 0.3 * cost_score + 0.3 * quality_score + 0.2 * latency_score + 0.2 * reliability_score
        else:  # ADAPTIVE
            # Adapt based on current conditions
            if budget_pressure > 0.1:
                score = 0.5 * cost_score + 0.2 * quality_score + 0.15 * latency_score + 0.15 * reliability_score
            else:
                score = 0.2 * cost_score + 0.4 * quality_score + 0.2 * latency_score + 0.2 * reliability_score
        
        # Apply bonuses
        score += complexity_bonus + budget_pressure
        
        # Quality requirement filter
        if profile.avg_quality_score < quality_requirement:
            score *= 0.5  # Heavy penalty for not meeting quality requirement
        
        return min(score, 1.0)
    
    async def _estimate_request_cost(
        self,
        profile: ModelCostProfile,
        task_complexity: str
    ) -> float:
        """Estimate cost for a request based on task complexity."""
        # Estimate token usage based on complexity
        if task_complexity == "simple":
            estimated_input_tokens = 200
            estimated_output_tokens = 100
        elif task_complexity == "complex":
            estimated_input_tokens = 1000
            estimated_output_tokens = 500
        else:  # medium
            estimated_input_tokens = 500
            estimated_output_tokens = 250
        
        # Calculate cost
        input_cost = (estimated_input_tokens / 1000) * profile.cost_per_1k_input_tokens
        output_cost = (estimated_output_tokens / 1000) * profile.cost_per_1k_output_tokens
        
        return input_cost + output_cost + profile.cost_per_request
    
    async def _create_basic_profile(self, backend: str, model: str):
        """Create a basic profile for unknown model."""
        # Default values based on backend
        if backend == "ollama":
            cost_input = 0.0
            cost_output = 0.0
            latency = 3000
            quality = 0.7
        elif "free" in model.lower():
            cost_input = 0.0
            cost_output = 0.0
            latency = 3500
            quality = 0.6
        else:
            cost_input = 0.5  # Conservative estimate
            cost_output = 1.0
            latency = 2000
            quality = 0.75
        
        profile = ModelCostProfile(
            model_name=model,
            backend=backend,
            cost_per_1k_input_tokens=cost_input,
            cost_per_1k_output_tokens=cost_output,
            cost_per_request=0.0,
            avg_latency_ms=latency,
            avg_quality_score=quality,
            reliability_score=0.9
        )
        
        self.model_profiles[f"{backend}:{model}"] = profile
        logger.debug(f"Created basic profile for {backend}:{model}")
    
    async def _generate_cost_recommendations(
        self,
        organization_id: str,
        usage_history: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if not usage_history:
            return recommendations
        
        # Analyze usage patterns
        model_costs = defaultdict(float)
        model_requests = defaultdict(int)
        
        for usage in usage_history:
            key = f"{usage['backend']}:{usage['model']}"
            model_costs[key] += usage["actual_cost"]
            model_requests[key] += 1
        
        # Find expensive models
        for model_key, total_cost in model_costs.items():
            if total_cost > 10.0:  # $10+ threshold
                avg_cost = total_cost / model_requests[model_key]
                
                # Find cheaper alternatives with similar quality
                backend, model = model_key.split(":", 1)
                current_profile = self.model_profiles.get(model_key)
                
                if current_profile:
                    cheaper_alternatives = []
                    
                    for alt_key, alt_profile in self.model_profiles.items():
                        if (alt_key != model_key and 
                            alt_profile.cost_per_token() < current_profile.cost_per_token() * 0.8 and
                            alt_profile.avg_quality_score >= current_profile.avg_quality_score * 0.9):
                            
                            potential_savings = (current_profile.cost_per_token() - alt_profile.cost_per_token()) * 1000 * model_requests[model_key]
                            cheaper_alternatives.append({
                                "model": alt_key,
                                "potential_savings": potential_savings,
                                "quality_diff": alt_profile.avg_quality_score - current_profile.avg_quality_score
                            })
                    
                    if cheaper_alternatives:
                        cheaper_alternatives.sort(key=lambda x: x["potential_savings"], reverse=True)
                        
                        recommendations.append({
                            "type": "model_substitution",
                            "current_model": model_key,
                            "current_monthly_cost": total_cost,
                            "avg_cost_per_request": avg_cost,
                            "alternative": cheaper_alternatives[0],
                            "priority": "high" if total_cost > 50 else "medium"
                        })
        
        # Budget recommendations
        budget = self.cost_budgets.get(organization_id)
        if budget and budget.usage_percentage() > 0.8:
            recommendations.append({
                "type": "budget_alert",
                "message": f"Budget usage at {budget.usage_percentage():.1%}",
                "suggested_actions": [
                    "Switch to more cost-effective models",
                    "Implement request caching",
                    "Reduce request frequency for non-critical tasks"
                ],
                "priority": "high"
            })
        
        # Free model recommendations
        free_models = [k for k, p in self.model_profiles.items() if p.cost_per_token() == 0]
        if free_models and any(p.cost_per_token() > 0 for p in self.model_profiles.values()):
            total_paid_cost = sum(u["actual_cost"] for u in usage_history if u["actual_cost"] > 0)
            
            if total_paid_cost > 5.0:  # $5+ in paid models
                recommendations.append({
                    "type": "free_model_opportunity",
                    "current_paid_cost": total_paid_cost,
                    "available_free_models": free_models,
                    "message": "Consider using free models for simple tasks",
                    "priority": "medium"
                })
        
        return recommendations
    
    def _calculate_daily_costs(self, usage_history: List[Dict]) -> List[Dict[str, Any]]:
        """Calculate daily cost trends."""
        daily_costs = defaultdict(float)
        
        for usage in usage_history:
            date = usage["timestamp"].date()
            daily_costs[date] += usage["actual_cost"]
        
        # Convert to list and sort by date
        result = []
        for date, cost in sorted(daily_costs.items()):
            result.append({
                "date": date.isoformat(),
                "cost": cost
            })
        
        return result[-30:]  # Last 30 days
    
    async def _send_budget_alert(
        self,
        organization_id: str,
        budget: CostBudget,
        threshold: float
    ):
        """Send budget alert notification."""
        logger.warning(
            f"Budget alert for {organization_id}: "
            f"{budget.usage_percentage():.1%} of monthly budget used "
            f"(${budget.current_spend:.2f} / ${budget.monthly_budget:.2f})"
        )
        
        # In a real implementation, this would send email/Slack/webhook notification
        # For now, just log the alert
    
    async def _background_optimization(self):
        """Background task for cost optimization."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Update predictions
                await self._update_cost_predictions()
                
                # Clean old data
                await self._cleanup_old_data()
                
                # Save current state
                await self._save_data()
                
                logger.debug("Cost optimization cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cost optimization: {e}")
    
    async def _update_cost_predictions(self):
        """Update cost predictions for organizations."""
        for org_id, budget in self.cost_budgets.items():
            # Get recent usage for this org
            org_usage = [
                u for u in self.usage_history 
                if u.get("organization_id") == org_id and 
                u["timestamp"] > datetime.utcnow() - timedelta(days=7)
            ]
            
            if org_usage:
                # Calculate daily average cost
                daily_avg = sum(u["actual_cost"] for u in org_usage) / 7
                
                # Predict monthly cost
                days_in_month = 30
                predicted_monthly = daily_avg * days_in_month
                
                self.cost_predictions[org_id] = predicted_monthly
                
                logger.debug(f"Predicted monthly cost for {org_id}: ${predicted_monthly:.2f}")
    
    async def _cleanup_old_data(self):
        """Clean up old usage data."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.learning_window_days)
        
        old_count = len(self.usage_history)
        self.usage_history = deque(
            [u for u in self.usage_history if u["timestamp"] > cutoff_date],
            maxlen=self.usage_history.maxlen
        )
        new_count = len(self.usage_history)
        
        if old_count != new_count:
            logger.debug(f"Cleaned up {old_count - new_count} old usage records")
    
    async def _save_data(self):
        """Save cost optimizer data."""
        if not self.cache_manager:
            return
        
        try:
            # Save model profiles
            profiles_data = {}
            for key, profile in self.model_profiles.items():
                profiles_data[key] = {
                    "model_name": profile.model_name,
                    "backend": profile.backend,
                    "cost_per_1k_input_tokens": profile.cost_per_1k_input_tokens,
                    "cost_per_1k_output_tokens": profile.cost_per_1k_output_tokens,
                    "cost_per_request": profile.cost_per_request,
                    "avg_latency_ms": profile.avg_latency_ms,
                    "avg_quality_score": profile.avg_quality_score,
                    "reliability_score": profile.reliability_score,
                    "total_requests": profile.total_requests,
                    "total_cost": profile.total_cost,
                    "total_tokens_input": profile.total_tokens_input,
                    "total_tokens_output": profile.total_tokens_output
                }
            
            await self.cache_manager.set_cache(
                "cost_optimizer_profiles",
                profiles_data,
                ttl=86400 * 7  # 7 days
            )
            
            # Save budgets
            budgets_data = {}
            for org_id, budget in self.cost_budgets.items():
                budgets_data[org_id] = {
                    "organization_id": budget.organization_id,
                    "monthly_budget": budget.monthly_budget,
                    "current_spend": budget.current_spend,
                    "alert_thresholds": budget.alert_thresholds,
                    "alerts_sent": budget.alerts_sent
                }
            
            await self.cache_manager.set_cache(
                "cost_optimizer_budgets",
                budgets_data,
                ttl=86400 * 30  # 30 days
            )
            
            logger.debug("Saved cost optimizer data")
            
        except Exception as e:
            logger.error(f"Failed to save cost optimizer data: {e}")
    
    async def _load_historical_data(self):
        """Load historical cost optimizer data."""
        if not self.cache_manager:
            return
        
        try:
            # Load model profiles
            profiles_data = await self.cache_manager.get_cache("cost_optimizer_profiles")
            if profiles_data:
                for key, data in profiles_data.items():
                    profile = ModelCostProfile(**data)
                    self.model_profiles[key] = profile
                
                logger.info(f"Loaded {len(self.model_profiles)} model profiles")
            
            # Load budgets
            budgets_data = await self.cache_manager.get_cache("cost_optimizer_budgets")
            if budgets_data:
                for org_id, data in budgets_data.items():
                    budget = CostBudget(**data)
                    self.cost_budgets[org_id] = budget
                
                logger.info(f"Loaded {len(self.cost_budgets)} budget configurations")
                
        except Exception as e:
            logger.error(f"Failed to load cost optimizer data: {e}")
    
    async def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get cost optimizer statistics."""
        total_cost = sum(p.total_cost for p in self.model_profiles.values())
        total_requests = sum(p.total_requests for p in self.model_profiles.values())
        
        # Model efficiency ranking
        efficient_models = []
        for key, profile in self.model_profiles.items():
            if profile.total_requests > 0:
                efficient_models.append({
                    "model": key,
                    "efficiency_score": profile.efficiency_score(),
                    "cost_per_token": profile.cost_per_token(),
                    "quality_score": profile.avg_quality_score,
                    "total_cost": profile.total_cost
                })
        
        efficient_models.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        return {
            "total_tracked_cost": total_cost,
            "total_tracked_requests": total_requests,
            "tracked_models": len(self.model_profiles),
            "organizations_with_budgets": len(self.cost_budgets),
            "avg_cost_per_request": total_cost / max(total_requests, 1),
            "most_efficient_models": efficient_models[:5],
            "budget_status": {
                org_id: {
                    "usage_percentage": budget.usage_percentage(),
                    "remaining_budget": budget.remaining_budget
                }
                for org_id, budget in self.cost_budgets.items()
            }
        }
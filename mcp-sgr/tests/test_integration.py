"""Integration tests for MCP-SGR."""

import pytest
import asyncio
import json
from pathlib import Path

from src.utils.llm_client import LLMClient
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager
from src.tools import apply_sgr_tool, wrap_agent_call_tool
from src.schemas import SCHEMA_REGISTRY


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual services."""
    
    @pytest.fixture
    async def setup_services(self, tmp_path):
        """Setup services for integration tests."""
        # Use temporary directory for cache
        cache_db = tmp_path / "test_cache.db"
        trace_db = tmp_path / "test_traces.db"
        
        # Override environment for tests
        import os
        os.environ["CACHE_STORE"] = f"sqlite:///{cache_db}"
        os.environ["TRACE_STORE"] = f"sqlite:///{trace_db}"
        os.environ["CACHE_ENABLED"] = "true"
        os.environ["TRACE_ENABLED"] = "true"
        
        # Initialize services
        llm_client = LLMClient()
        cache_manager = CacheManager()
        telemetry = TelemetryManager()
        
        await cache_manager.initialize()
        await telemetry.initialize()
        
        yield llm_client, cache_manager, telemetry
        
        # Cleanup
        await llm_client.close()
        await cache_manager.close()
        await telemetry.close()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path(".env").exists(),
        reason="Requires .env configuration"
    )
    async def test_full_sgr_flow(self, setup_services):
        """Test complete SGR flow with real LLM."""
        llm_client, cache_manager, telemetry = setup_services
        
        # First call - should hit LLM
        result1 = await apply_sgr_tool(
            arguments={
                "task": "Design a simple REST API for a todo list",
                "schema_type": "planning",
                "budget": "lite"
            },
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry
        )
        
        assert result1["confidence"] > 0
        assert "reasoning" in result1
        assert "suggested_actions" in result1
        
        # Second call with same task - should hit cache
        result2 = await apply_sgr_tool(
            arguments={
                "task": "Design a simple REST API for a todo list",
                "schema_type": "planning",
                "budget": "lite"
            },
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry
        )
        
        # Results should be identical (from cache)
        assert result1["reasoning"] == result2["reasoning"]
        
        # Check cache stats
        stats = await cache_manager.get_cache_stats()
        assert stats["total_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_schema_validation_flow(self, setup_services):
        """Test schema validation with various inputs."""
        llm_client, cache_manager, telemetry = setup_services
        
        # Test each schema type
        for schema_name in ["analysis", "planning", "decision"]:
            schema = SCHEMA_REGISTRY[schema_name]()
            
            # Generate example that should validate
            example = schema.get_examples()[0] if schema.get_examples() else {}
            validation = schema.validate(example)
            
            assert validation.valid, f"Schema {schema_name} example should be valid"
            assert validation.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_agent_wrapper_flow(self, setup_services):
        """Test agent wrapper with mock agent."""
        llm_client, cache_manager, telemetry = setup_services
        
        # Create mock agent
        call_count = 0
        async def mock_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            return {
                "status": "success",
                "result": f"Mock result {call_count}",
                "metadata": {"call_count": call_count}
            }
        
        # Wrap agent call
        result = await wrap_agent_call_tool(
            arguments={
                "agent_endpoint": mock_agent,
                "agent_request": {"task": "test"},
                "sgr_config": {
                    "schema_type": "analysis",
                    "budget": "lite",
                    "pre_analysis": True,
                    "post_analysis": True
                }
            },
            llm_client=llm_client,
            cache_manager=cache_manager,
            telemetry=telemetry
        )
        
        assert result["original_response"]["status"] == "success"
        assert call_count == 1
        assert "reasoning_chain" in result
        assert "quality_metrics" in result
    
    @pytest.mark.asyncio
    async def test_cache_limits(self, setup_services):
        """Test cache size limits."""
        llm_client, cache_manager, telemetry = setup_services
        
        # Override limits for testing
        cache_manager.max_entries = 5
        
        # Add entries beyond limit
        for i in range(10):
            await cache_manager.set(
                f"test_key_{i}",
                {"data": f"test_value_{i}", "index": i}
            )
        
        # Check that old entries were cleaned up
        stats = await cache_manager.get_cache_stats()
        assert stats["total_entries"] <= cache_manager.max_entries
        
        # Verify newest entries are kept
        for i in range(5, 10):
            value = await cache_manager.get(f"test_key_{i}")
            assert value is not None, f"Recent entry {i} should be kept"
    
    @pytest.mark.asyncio
    async def test_telemetry_flow(self, setup_services):
        """Test telemetry span tracking."""
        llm_client, cache_manager, telemetry = setup_services
        
        # Create spans
        span1 = await telemetry.start_span("test_operation", {"test": True})
        await asyncio.sleep(0.1)  # Simulate work
        await telemetry.end_span(span1, {"result": "success"})
        
        # Nested spans
        span2 = await telemetry.start_span("parent_operation")
        span3 = await telemetry.start_span("child_operation")
        await telemetry.end_span(span3)
        await telemetry.end_span(span2)
        
        # Verify spans were tracked (even if telemetry is disabled)
        assert span1 is not None
        assert span2 is not None
        assert span3 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
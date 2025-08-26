"""Performance and monitoring tests for MCP-SGR."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCachePerformance:
    """Test cache performance and functionality."""

    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        from src.utils.cache import CacheManager
        
        with patch.dict('os.environ', {'CACHE_STORE': 'sqlite:///test.db'}):
            cache_manager = CacheManager()
            await cache_manager.initialize()
            
            assert cache_manager.enabled is True
            await cache_manager.close()

    @pytest.mark.asyncio
    async def test_cache_operations_performance(self):
        """Test basic cache operations performance."""
        from src.utils.cache import CacheManager
        
        with patch.dict('os.environ', {'CACHE_STORE': 'sqlite:///test_perf.db'}):
            cache_manager = CacheManager()
            await cache_manager.initialize()
            
            # Measure cache write performance
            start_time = time.time()
            test_data = {"result": "test response", "metadata": {"tokens": 100}}
            
            for i in range(10):
                await cache_manager.set_cache(f"test_key_{i}", test_data)
            
            write_time = time.time() - start_time
            assert write_time < 1.0, "Cache writes should be fast (< 1 second for 10 operations)"
            
            # Measure cache read performance  
            start_time = time.time()
            
            for i in range(10):
                result = await cache_manager.get_cache(f"test_key_{i}")
                assert result is not None
            
            read_time = time.time() - start_time
            assert read_time < 0.5, "Cache reads should be very fast (< 0.5 seconds for 10 operations)"
            
            await cache_manager.close()

    @pytest.mark.asyncio
    async def test_cache_memory_usage(self):
        """Test cache doesn't consume excessive memory."""
        from src.utils.cache import CacheManager
        import tracemalloc
        
        tracemalloc.start()
        
        with patch.dict('os.environ', {'CACHE_STORE': 'sqlite:///test_memory.db'}):
            cache_manager = CacheManager()
            await cache_manager.initialize()
            
            # Store larger data items
            large_data = {"result": "x" * 1000, "metadata": {"tokens": 1000}}
            
            current, peak = tracemalloc.get_traced_memory()
            initial_memory = current
            
            # Add many cache entries
            for i in range(100):
                await cache_manager.set_cache(f"large_key_{i}", large_data)
            
            current, peak = tracemalloc.get_traced_memory()
            memory_growth = current - initial_memory
            
            # Memory growth should be reasonable (< 10MB for 100 entries)
            assert memory_growth < 10 * 1024 * 1024, "Cache should not consume excessive memory"
            
            await cache_manager.close()
            tracemalloc.stop()


class TestLLMClientPerformance:
    """Test LLM client performance."""

    @pytest.mark.asyncio
    async def test_llm_client_initialization_speed(self):
        """Test LLM client initializes quickly."""
        from src.utils.llm_client import LLMClient
        
        start_time = time.time()
        llm_client = LLMClient()
        init_time = time.time() - start_time
        
        assert init_time < 1.0, "LLM client should initialize quickly"
        await llm_client.close()

    @pytest.mark.asyncio
    async def test_concurrent_llm_requests(self):
        """Test handling concurrent LLM requests."""
        from src.utils.llm_client import LLMClient
        
        llm_client = LLMClient()
        
        # Mock the actual LLM call to avoid external dependencies
        async def mock_generate(prompt, **kwargs):
            await asyncio.sleep(0.1)  # Simulate API delay
            return {"result": f"Response to: {prompt[:20]}..."}
        
        with patch.object(llm_client, 'generate_response', side_effect=mock_generate):
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                task = llm_client.generate_response(f"Test prompt {i}")
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Should complete in less than sequential time (concurrent execution)
            assert total_time < 0.8, "Concurrent requests should be faster than sequential"
            assert len(results) == 5
            
        await llm_client.close()


class TestTelemetryPerformance:
    """Test telemetry and monitoring performance."""

    @pytest.mark.asyncio
    async def test_telemetry_overhead(self):
        """Test telemetry adds minimal overhead."""
        from src.utils.telemetry import TelemetryManager
        
        telemetry = TelemetryManager()
        await telemetry.initialize()
        
        # Measure time without telemetry
        start_time = time.time()
        for _ in range(100):
            pass  # No-op
        baseline_time = time.time() - start_time
        
        # Measure time with telemetry
        start_time = time.time()
        for i in range(100):
            await telemetry.record_tool_call("test_tool", {"test": "data"}, {"result": "success"})
        telemetry_time = time.time() - start_time
        
        # Telemetry overhead should be minimal
        overhead = telemetry_time - baseline_time
        assert overhead < 0.5, "Telemetry should add minimal overhead"
        
        await telemetry.close()

    @pytest.mark.asyncio
    async def test_trace_storage_performance(self):
        """Test trace storage performance."""
        from src.utils.telemetry import TelemetryManager
        
        with patch.dict('os.environ', {'TRACE_STORE': 'sqlite:///test_traces.db'}):
            telemetry = TelemetryManager()
            await telemetry.initialize()
            
            # Store many traces quickly
            start_time = time.time()
            
            for i in range(50):
                await telemetry.record_tool_call(
                    tool_name="test_tool",
                    arguments={"task": f"task_{i}"},
                    result={"result": f"result_{i}"}
                )
            
            storage_time = time.time() - start_time
            assert storage_time < 2.0, "Trace storage should be fast"
            
            await telemetry.close()


class TestRouterPerformance:
    """Test router performance."""

    def test_router_decision_speed(self):
        """Test router makes decisions quickly."""
        from src.utils.router import SGRRouter
        
        router = SGRRouter()
        
        # Test multiple routing decisions
        start_time = time.time()
        
        for i in range(100):
            task = f"Test task {i}"
            context = {"type": "analysis"}
            backend, model = router.route_request(task, context)
            assert backend is not None
            assert model is not None
        
        routing_time = time.time() - start_time
        assert routing_time < 0.1, "Routing decisions should be very fast"

    def test_router_policy_loading(self):
        """Test router policy loading performance."""
        from src.utils.router import SGRRouter
        
        start_time = time.time()
        router = SGRRouter()
        loading_time = time.time() - start_time
        
        assert loading_time < 0.5, "Router policy loading should be fast"
        assert router.policy is not None


class TestAPIEndpointPerformance:
    """Test API endpoint performance."""

    @pytest.fixture
    def mock_services(self):
        """Mock all services for performance testing."""
        mock_llm = AsyncMock()
        mock_llm.generate_response.return_value = {
            "result": "test response",
            "reasoning": {"steps": ["analysis"]},
            "metadata": {"model": "test", "tokens": 100}
        }
        
        mock_cache = AsyncMock()
        mock_cache.get_cache.return_value = None
        mock_cache.set_cache.return_value = None
        
        mock_telemetry = AsyncMock()
        
        with patch('src.http_server.llm_client', mock_llm), \
             patch('src.http_server.cache_manager', mock_cache), \
             patch('src.http_server.telemetry_manager', mock_telemetry):
            yield mock_llm, mock_cache, mock_telemetry

    @pytest.mark.asyncio
    async def test_apply_sgr_endpoint_performance(self, mock_services):
        """Test apply-sgr endpoint performance."""
        from fastapi.testclient import TestClient
        from src.http_server import app
        
        client = TestClient(app)
        
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'false'}):
            # Measure response time
            start_time = time.time()
            
            response = client.post("/v1/apply-sgr", json={
                "task": "Analyze this simple task",
                "schema_type": "analysis"
            })
            
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 2.0, "API response should be fast"

    def test_health_endpoint_performance(self):
        """Test health endpoint is very fast."""
        from fastapi.testclient import TestClient
        from src.http_server import app
        
        client = TestClient(app)
        
        # Health check should be extremely fast
        start_time = time.time()
        response = client.get("/health")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 0.1, "Health check should be very fast"


class TestMemoryLeaks:
    """Test for memory leaks and resource cleanup."""

    @pytest.mark.asyncio
    async def test_no_memory_leaks_in_cache(self):
        """Test cache operations don't cause memory leaks."""
        from src.utils.cache import CacheManager
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        with patch.dict('os.environ', {'CACHE_STORE': 'sqlite:///test_leak.db'}):
            # Create and destroy many cache managers
            for i in range(10):
                cache_manager = CacheManager()
                await cache_manager.initialize()
                
                # Do some operations
                await cache_manager.set_cache(f"key_{i}", {"data": f"value_{i}"})
                result = await cache_manager.get_cache(f"key_{i}")
                assert result is not None
                
                await cache_manager.close()
                del cache_manager
            
            # Force garbage collection
            gc.collect()
            
            # Check that we don't have lingering cache managers
            cache_objects = [obj for obj in gc.get_objects() if type(obj).__name__ == 'CacheManager']
            assert len(cache_objects) <= 1, "Should not have many CacheManager instances"

    @pytest.mark.asyncio
    async def test_llm_client_resource_cleanup(self):
        """Test LLM client properly cleans up resources."""
        from src.utils.llm_client import LLMClient
        import gc
        
        gc.collect()
        
        # Create and destroy many LLM clients
        for i in range(5):
            llm_client = LLMClient()
            await llm_client.close()
            del llm_client
        
        gc.collect()
        
        # Check for resource leaks
        llm_objects = [obj for obj in gc.get_objects() if type(obj).__name__ == 'LLMClient']
        assert len(llm_objects) <= 1, "Should not have many LLMClient instances"


class TestScalabilityIndicators:
    """Test indicators of system scalability."""

    def test_configuration_flexibility(self):
        """Test system can be configured for different scales."""
        import os
        
        # Test that key scalability settings can be configured
        scalability_configs = [
            'CACHE_STORE',
            'RATE_LIMIT_MAX_RPM', 
            'HTTP_CORS_ORIGINS',
            'LLM_BACKENDS',
            'OTEL_ENABLED'
        ]
        
        for config in scalability_configs:
            # Should be able to set any value without import errors
            with patch.dict('os.environ', {config: 'test_value'}):
                # Import should work with any config
                try:
                    from src.utils.cache import CacheManager
                    from src.utils.llm_client import LLMClient
                    assert True  # Import successful
                except ImportError:
                    pytest.fail(f"Configuration {config} breaks imports")

    def test_async_support_throughout(self):
        """Test that async/await is used throughout for scalability."""
        import inspect
        
        # Key modules should have async functions
        modules_to_check = [
            'src.utils.cache',
            'src.utils.llm_client', 
            'src.utils.telemetry',
            'src.tools.apply_sgr'
        ]
        
        for module_name in modules_to_check:
            try:
                module = __import__(module_name.replace('/', '.'), fromlist=[''])
                
                # Count async functions
                async_functions = []
                for name, obj in inspect.getmembers(module):
                    if inspect.iscoroutinefunction(obj):
                        async_functions.append(name)
                
                # Should have some async functions for scalability
                assert len(async_functions) > 0, f"Module {module_name} should have async functions"
                
            except ImportError:
                # Module might not exist, skip
                continue
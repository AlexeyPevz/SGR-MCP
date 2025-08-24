"""Tests for SGR tools."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.tools import (
    apply_sgr_tool,
    enhance_prompt_tool,
    wrap_agent_call_tool
)
from src.utils.llm_client import LLMClient
from src.utils.cache import CacheManager
from src.utils.telemetry import TelemetryManager


@pytest.fixture
async def mock_llm_client():
    """Mock LLM client."""
    client = Mock(spec=LLMClient)
    client.generate = AsyncMock(return_value='{"understanding": {"task_summary": "Test task"}}')
    return client


@pytest.fixture
async def mock_cache_manager():
    """Mock cache manager."""
    manager = Mock(spec=CacheManager)
    manager.initialize = AsyncMock()
    manager.get = AsyncMock(return_value=None)
    manager.set = AsyncMock(return_value=True)
    manager.enabled = True
    return manager


@pytest.fixture
async def mock_telemetry():
    """Mock telemetry manager."""
    telemetry = Mock(spec=TelemetryManager)
    telemetry.initialize = AsyncMock()
    telemetry.start_span = AsyncMock(return_value="test-span-id")
    telemetry.end_span = AsyncMock()
    return telemetry


class TestApplySGRTool:
    """Test apply_sgr_tool function."""
    
    @pytest.mark.asyncio
    async def test_apply_sgr_basic(self, mock_llm_client, mock_cache_manager, mock_telemetry):
        """Test basic SGR application."""
        # Mock LLM response
        mock_llm_client.generate.return_value = '''
        {
            "understanding": {
                "task_summary": "Build a REST API",
                "key_aspects": ["Authentication", "CRUD operations"]
            },
            "goals": {
                "primary": "Create secure API",
                "success_criteria": ["All endpoints secured"]
            },
            "constraints": [
                {"type": "technical", "description": "Use FastAPI"}
            ],
            "risks": [
                {
                    "risk": "Security vulnerabilities",
                    "likelihood": "medium",
                    "impact": "high",
                    "mitigation": "Follow OWASP"
                }
            ]
        }
        '''
        
        result = await apply_sgr_tool(
            arguments={
                "task": "Build a REST API",
                "schema_type": "analysis",
                "budget": "lite"
            },
            llm_client=mock_llm_client,
            cache_manager=mock_cache_manager,
            telemetry=mock_telemetry
        )
        
        assert "reasoning" in result
        assert "confidence" in result
        assert result["confidence"] > 0
        assert "suggested_actions" in result
        assert mock_llm_client.generate.called
    
    @pytest.mark.asyncio
    async def test_apply_sgr_with_cache_hit(self, mock_llm_client, mock_cache_manager, mock_telemetry):
        """Test SGR with cache hit."""
        cached_result = {
            "reasoning": {"cached": True},
            "confidence": 0.9,
            "suggested_actions": ["Cached action"]
        }
        mock_cache_manager.get.return_value = cached_result
        
        result = await apply_sgr_tool(
            arguments={
                "task": "Cached task",
                "schema_type": "analysis",
                "budget": "lite"
            },
            llm_client=mock_llm_client,
            cache_manager=mock_cache_manager,
            telemetry=mock_telemetry
        )
        
        assert result == cached_result
        assert not mock_llm_client.generate.called


class TestEnhancePromptTool:
    """Test enhance_prompt_tool function."""
    
    @pytest.mark.asyncio
    async def test_enhance_prompt_basic(self, mock_llm_client, mock_cache_manager):
        """Test basic prompt enhancement."""
        # Mock analysis response
        mock_llm_client.generate.return_value = '''
        {
            "intent": "Write code",
            "detected_type": "code_generation",
            "key_elements": ["function", "Python"],
            "improvements": ["Add type hints", "Include error handling"]
        }
        '''
        
        result = await enhance_prompt_tool(
            arguments={
                "original_prompt": "Write a Python function",
                "enhancement_level": "standard"
            },
            llm_client=mock_llm_client,
            cache_manager=mock_cache_manager
        )
        
        assert "enhanced_prompt" in result
        assert "original_prompt" in result
        assert "metadata" in result
        assert result["metadata"]["detected_intent"] == "Write code"
        assert len(result["enhanced_prompt"]) > len(result["original_prompt"])


class TestWrapAgentCall:
    """Test wrap_agent_call_tool function."""
    
    @pytest.mark.asyncio
    async def test_wrap_agent_basic(self, mock_llm_client, mock_cache_manager, mock_telemetry):
        """Test basic agent wrapping."""
        # Mock agent function
        async def mock_agent(prompt: str):
            return {"response": "Generated code", "status": "success"}
        
        # Mock pre-analysis
        mock_llm_client.generate.side_effect = [
            # Pre-analysis response
            '''
            {
                "understanding": {
                    "task_summary": "Generate code",
                    "key_aspects": ["Python function"]
                },
                "goals": {
                    "primary": "Create working code",
                    "success_criteria": ["Correct syntax"]
                },
                "constraints": [],
                "risks": []
            }
            ''',
            # Post-analysis response
            '''
            {
                "understanding": {
                    "task_summary": "Analyze response",
                    "key_aspects": ["Response quality"]
                },
                "goals": {
                    "primary": "Validate response",
                    "success_criteria": ["Response is valid"]
                },
                "constraints": [],
                "risks": []
            }
            '''
        ]
        
        result = await wrap_agent_call_tool(
            arguments={
                "agent_endpoint": mock_agent,
                "agent_request": {"prompt": "Write a function"},
                "sgr_config": {
                    "schema_type": "analysis",
                    "pre_analysis": True,
                    "post_analysis": True
                }
            },
            llm_client=mock_llm_client,
            cache_manager=mock_cache_manager,
            telemetry=mock_telemetry
        )
        
        assert "original_response" in result
        assert result["original_response"]["status"] == "success"
        assert "reasoning_chain" in result
        assert "pre" in result["reasoning_chain"]
        assert "post" in result["reasoning_chain"]
        assert "quality_metrics" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
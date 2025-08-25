import asyncio
from unittest.mock import AsyncMock, Mock, patch

from click.testing import CliRunner

from src.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MCP-SGR" in result.output


@patch("src.cli.LLMClient")
@patch("src.cli.CacheManager")
@patch("src.cli.TelemetryManager")
def test_cli_analyze_mocked(mock_tel, mock_cache, mock_llm):
    # Mock services
    mock_llm.return_value = llm = Mock()
    llm.generate = AsyncMock(
        return_value='{"understanding": {"task_summary": "Test", "key_aspects": ["a"]}, "goals": {"primary": "p", "success_criteria": ["s"]}, "constraints": [], "risks": []}'
    )
    llm.close = AsyncMock()

    mock_cache.return_value = cache = Mock()
    cache.initialize = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.close = AsyncMock()

    mock_tel.return_value = tel = Mock()
    tel.initialize = AsyncMock()
    tel.close = AsyncMock()
    tel.start_span = AsyncMock(return_value="test-span")
    tel.end_span = AsyncMock()

    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", "Test task", "--schema", "analysis", "--json"])
    assert result.exit_code == 0
    assert "reasoning" in result.output

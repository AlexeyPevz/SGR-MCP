"""Integration tests for third-party frameworks."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestLangChainIntegration:
    """Test LangChain integration."""

    @pytest.fixture
    def mock_langchain(self):
        """Mock LangChain dependencies."""
        mock_modules = {
            'langchain': MagicMock(),
            'langchain.schema': MagicMock(),
            'langchain.callbacks': MagicMock(),
            'langchain.callbacks.base': MagicMock(),
            'langchain.schema.messages': MagicMock(),
        }
        
        with patch.dict('sys.modules', mock_modules):
            yield mock_modules

    @pytest.mark.asyncio
    async def test_langchain_sgr_wrapper_import(self, mock_langchain):
        """Test that LangChain SGR wrapper can be imported."""
        try:
            from integrations.langchain.sgr_langchain import SGRLangChainWrapper
            assert SGRLangChainWrapper is not None
        except ImportError as e:
            pytest.skip(f"LangChain integration not available: {e}")

    @pytest.mark.asyncio
    async def test_langchain_sgr_wrapper_functionality(self, mock_langchain):
        """Test basic LangChain wrapper functionality."""
        try:
            from integrations.langchain.sgr_langchain import SGRLangChainWrapper
            
            # Mock LLM client
            mock_llm_client = AsyncMock()
            mock_llm_client.generate_response.return_value = {
                "result": "test response",
                "reasoning": {"steps": ["step1", "step2"]},
                "metadata": {"model": "test-model"}
            }
            
            wrapper = SGRLangChainWrapper(llm_client=mock_llm_client)
            
            # Test basic functionality without actual LangChain dependencies
            assert wrapper.llm_client == mock_llm_client
            
        except ImportError:
            pytest.skip("LangChain dependencies not available")


class TestAutoGenIntegration:
    """Test AutoGen integration."""

    @pytest.fixture
    def mock_autogen(self):
        """Mock AutoGen dependencies."""
        mock_modules = {
            'autogen': MagicMock(),
            'autogen.agentchat': MagicMock(),
            'autogen.agentchat.assistant_agent': MagicMock(),
            'autogen.agentchat.user_proxy_agent': MagicMock(),
        }
        
        with patch.dict('sys.modules', mock_modules):
            yield mock_modules

    @pytest.mark.asyncio
    async def test_autogen_sgr_integration_import(self, mock_autogen):
        """Test that AutoGen SGR integration can be imported."""
        try:
            from integrations.autogen.sgr_autogen import SGRAutoGenAgent
            assert SGRAutoGenAgent is not None
        except ImportError as e:
            pytest.skip(f"AutoGen integration not available: {e}")

    @pytest.mark.asyncio
    async def test_autogen_sgr_agent_creation(self, mock_autogen):
        """Test AutoGen SGR agent creation."""
        try:
            from integrations.autogen.sgr_autogen import SGRAutoGenAgent
            
            # Mock LLM client
            mock_llm_client = AsyncMock()
            
            agent = SGRAutoGenAgent(
                name="test_agent",
                llm_client=mock_llm_client,
                system_message="Test system message"
            )
            
            assert agent.name == "test_agent"
            assert agent.llm_client == mock_llm_client
            
        except ImportError:
            pytest.skip("AutoGen dependencies not available")


class TestCrewAIIntegration:
    """Test CrewAI integration."""

    @pytest.fixture
    def mock_crewai(self):
        """Mock CrewAI dependencies."""
        mock_modules = {
            'crewai': MagicMock(),
            'crewai.agent': MagicMock(),
            'crewai.task': MagicMock(),
            'crewai.crew': MagicMock(),
        }
        
        with patch.dict('sys.modules', mock_modules):
            yield mock_modules

    @pytest.mark.asyncio
    async def test_crewai_sgr_integration_import(self, mock_crewai):
        """Test that CrewAI SGR integration can be imported."""
        try:
            from integrations.crewai.sgr_crewai import SGRCrewAIAgent
            assert SGRCrewAIAgent is not None
        except ImportError as e:
            pytest.skip(f"CrewAI integration not available: {e}")

    @pytest.mark.asyncio
    async def test_crewai_sgr_agent_functionality(self, mock_crewai):
        """Test basic CrewAI SGR agent functionality."""
        try:
            from integrations.crewai.sgr_crewai import SGRCrewAIAgent
            
            # Mock LLM client
            mock_llm_client = AsyncMock()
            mock_llm_client.generate_response.return_value = {
                "result": "crew ai response",
                "reasoning": {"analysis": "test analysis"},
                "metadata": {"tokens": 100}
            }
            
            agent = SGRCrewAIAgent(
                role="analyst",
                goal="analyze data",
                backstory="expert analyst",
                llm_client=mock_llm_client
            )
            
            assert agent.role == "analyst"
            assert agent.llm_client == mock_llm_client
            
        except ImportError:
            pytest.skip("CrewAI dependencies not available")


class TestN8NIntegration:
    """Test n8n integration."""

    def test_n8n_node_example_exists(self):
        """Test that n8n node example exists."""
        n8n_example_path = project_root / "integrations" / "n8n" / "n8n-http-node-example.json"
        
        # Check if example file exists in integrations directory
        if not n8n_example_path.exists():
            # Check if it exists in examples directory
            n8n_example_path = project_root / "examples" / "n8n-http-node-example.json"
        
        assert n8n_example_path.exists(), "n8n integration example should exist"

    def test_n8n_integration_documentation(self):
        """Test that n8n integration documentation exists."""
        n8n_docs_paths = [
            project_root / "integrations" / "n8n" / "README.md",
            project_root / "integrations" / "n8n" / "n8n-integration-guide.md",
            project_root / "examples" / "n8n-integration-guide.md"
        ]
        
        docs_exist = any(path.exists() for path in n8n_docs_paths)
        assert docs_exist, "n8n integration documentation should exist"


class TestIntegrationExamples:
    """Test integration examples."""

    def test_basic_usage_example(self):
        """Test basic usage example exists and is valid."""
        example_path = project_root / "examples" / "basic_usage.py"
        assert example_path.exists(), "Basic usage example should exist"
        
        # Read and check basic syntax
        content = example_path.read_text()
        assert "import" in content
        assert "mcp-sgr" in content or "sgr" in content

    def test_custom_schema_example(self):
        """Test custom schema example exists."""
        example_path = project_root / "examples" / "custom_schema.py"
        assert example_path.exists(), "Custom schema example should exist"

    def test_agent_wrapper_example(self):
        """Test agent wrapper example exists."""
        example_path = project_root / "examples" / "agent_wrapper.py"
        assert example_path.exists(), "Agent wrapper example should exist"

    def test_http_api_example(self):
        """Test HTTP API example exists."""
        example_path = project_root / "examples" / "test_http_api.py"
        assert example_path.exists(), "HTTP API test example should exist"


class TestDockerIntegration:
    """Test Docker integration and deployment."""

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile should exist"

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        compose_path = project_root / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml should exist"

    def test_docker_compose_configuration(self):
        """Test docker-compose configuration."""
        import yaml
        
        compose_path = project_root / "docker-compose.yml"
        if compose_path.exists():
            with open(compose_path) as f:
                compose_config = yaml.safe_load(f)
            
            # Check that main services are defined
            services = compose_config.get("services", {})
            assert "mcp-sgr" in services, "mcp-sgr service should be defined"
            
            # Check environment variables are configured
            mcp_sgr_service = services.get("mcp-sgr", {})
            environment = mcp_sgr_service.get("environment", [])
            
            # Environment can be list or dict
            if isinstance(environment, list):
                env_vars = [var.split("=")[0] for var in environment if "=" in var]
            else:
                env_vars = list(environment.keys())
            
            assert any("HTTP_" in var for var in env_vars), "HTTP config should be present"
            assert any("CACHE_" in var for var in env_vars), "Cache config should be present"

    def test_deployment_scripts_exist(self):
        """Test that deployment scripts exist."""
        scripts = [
            "deploy-simple.sh",
            "docker-one-liner.sh",
            "setup.sh"
        ]
        
        for script in scripts:
            script_path = project_root / script
            assert script_path.exists(), f"Deployment script {script} should exist"


class TestCLIIntegration:
    """Test CLI integration and functionality."""

    def test_cli_module_exists(self):
        """Test that CLI module exists."""
        cli_path = project_root / "src" / "cli.py"
        assert cli_path.exists(), "CLI module should exist"

    @pytest.mark.asyncio
    async def test_cli_basic_import(self):
        """Test that CLI can be imported."""
        try:
            from src.cli import main
            assert main is not None
        except ImportError as e:
            pytest.fail(f"CLI import failed: {e}")

    def test_pyproject_cli_entry_point(self):
        """Test that CLI entry point is defined in pyproject.toml."""
        import toml
        
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path) as f:
                pyproject = toml.load(f)
            
            scripts = pyproject.get("project", {}).get("scripts", {})
            assert "mcp-sgr" in scripts, "CLI entry point should be defined"
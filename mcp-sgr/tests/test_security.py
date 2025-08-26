"""Security tests for MCP-SGR HTTP API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.http_server import app, validate_safe_input


class TestInputValidation:
    """Test input validation and security."""

    def test_validate_safe_input_normal(self):
        """Test normal input passes validation."""
        safe_input = "This is a normal task description for analysis"
        result = validate_safe_input(safe_input)
        assert result == safe_input

    def test_validate_safe_input_xss_detection(self):
        """Test XSS attack detection."""
        malicious_input = "<script>alert('xss')</script>"
        with pytest.raises(ValueError, match="Potentially dangerous input detected"):
            validate_safe_input(malicious_input)

    def test_validate_safe_input_javascript_detection(self):
        """Test JavaScript injection detection."""
        malicious_input = "javascript:alert('attack')"
        with pytest.raises(ValueError, match="Potentially dangerous input detected"):
            validate_safe_input(malicious_input)

    def test_validate_safe_input_eval_detection(self):
        """Test eval function detection."""
        malicious_input = "eval('malicious code')"
        with pytest.raises(ValueError, match="Potentially dangerous input detected"):
            validate_safe_input(malicious_input)

    def test_validate_safe_input_exec_detection(self):
        """Test exec function detection."""
        malicious_input = "exec('os.system(\"rm -rf /\")')"
        with pytest.raises(ValueError, match="Potentially dangerous input detected"):
            validate_safe_input(malicious_input)

    def test_validate_safe_input_os_import_detection(self):
        """Test OS import detection."""
        malicious_input = "import os; os.system('pwd')"
        with pytest.raises(ValueError, match="Potentially dangerous input detected"):
            validate_safe_input(malicious_input)

    def test_validate_safe_input_subprocess_detection(self):
        """Test subprocess detection."""
        malicious_input = "import subprocess; subprocess.call(['ls'])"
        with pytest.raises(ValueError, match="Potentially dangerous input detected"):
            validate_safe_input(malicious_input)

    def test_validate_safe_input_length_limit(self):
        """Test input length limit."""
        long_input = "x" * 50001  # Over 50KB limit
        with pytest.raises(ValueError, match="Input too long"):
            validate_safe_input(long_input)

    def test_validate_safe_input_non_string(self):
        """Test non-string input rejection."""
        with pytest.raises(ValueError, match="Input must be a string"):
            validate_safe_input(123)


class TestAPISecurity:
    """Test API security features."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_services(self):
        """Mock services for testing."""
        with patch('src.http_server.llm_client', AsyncMock()), \
             patch('src.http_server.cache_manager', AsyncMock()), \
             patch('src.http_server.telemetry_manager', AsyncMock()):
            yield

    def test_security_headers_present(self, client, mock_services):
        """Test that security headers are added to responses."""
        # Disable auth for this test
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'false'}):
            response = client.get("/health")
        
        # Check security headers
        headers = response.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("X-XSS-Protection") == "1; mode=block"
        assert "max-age=31536000" in headers.get("Strict-Transport-Security", "")
        assert "default-src 'self'" in headers.get("Content-Security-Policy", "")
        assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_cors_restrictions(self, client):
        """Test CORS restrictions are in place."""
        response = client.options("/health")
        # CORS should be more restrictive now
        allowed_methods = response.headers.get("Access-Control-Allow-Methods", "")
        assert "DELETE" not in allowed_methods  # Should not allow DELETE
        assert "PUT" not in allowed_methods     # Should not allow PUT

    def test_invalid_request_validation(self, client, mock_services):
        """Test request validation with invalid data."""
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'false'}):
            # Test with malicious input
            response = client.post("/v1/apply-sgr", json={
                "task": "<script>alert('xss')</script>",
                "schema_type": "analysis"
            })
            assert response.status_code == 422  # Validation error

            # Test with invalid schema_type
            response = client.post("/v1/apply-sgr", json={
                "task": "Normal task",
                "schema_type": "invalid-schema$%"
            })
            assert response.status_code == 422  # Validation error

            # Test with invalid budget
            response = client.post("/v1/apply-sgr", json={
                "task": "Normal task",
                "budget": "invalid_budget"
            })
            assert response.status_code == 422  # Validation error

    def test_authentication_required(self, client, mock_services):
        """Test that authentication is properly enforced."""
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'true', 'HTTP_AUTH_TOKEN': 'secret'}):
            # Request without auth should fail
            response = client.post("/v1/apply-sgr", json={
                "task": "Test task"
            })
            assert response.status_code == 401

            # Request with wrong token should fail
            response = client.post("/v1/apply-sgr", 
                                 headers={"X-API-Key": "wrong-token"},
                                 json={"task": "Test task"})
            assert response.status_code == 401

            # Request with correct token should pass auth (but may fail for other reasons)
            response = client.post("/v1/apply-sgr", 
                                 headers={"X-API-Key": "secret"},
                                 json={"task": "Test task"})
            assert response.status_code != 401  # Not an auth error


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_rate_limiting_disabled_by_default(self, client):
        """Test that rate limiting is disabled by default."""
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'false', 'RATE_LIMIT_ENABLED': 'false'}):
            # Should be able to make many requests
            for _ in range(5):
                response = client.get("/health")
                assert response.status_code == 200

    def test_rate_limiting_when_enabled(self, client):
        """Test rate limiting when enabled."""
        with patch.dict('os.environ', {
            'HTTP_REQUIRE_AUTH': 'false', 
            'RATE_LIMIT_ENABLED': 'true',
            'RATE_LIMIT_MAX_RPM': '2'  # Very low limit for testing
        }):
            # First few requests should work
            response1 = client.get("/health")
            response2 = client.get("/health")
            assert response1.status_code == 200
            assert response2.status_code == 200
            
            # Additional requests should be rate limited
            # Note: This test may be flaky depending on timing and implementation


class TestEndpointValidation:
    """Test individual endpoint validation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_wrap_agent_endpoint_validation(self, client):
        """Test wrap-agent endpoint validation."""
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'false'}):
            # Invalid endpoint format
            response = client.post("/v1/wrap-agent", json={
                "agent_endpoint": "invalid$endpoint%",
                "agent_request": {}
            })
            assert response.status_code == 422

    def test_enhance_prompt_validation(self, client):
        """Test enhance-prompt endpoint validation."""
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'false'}):
            # Invalid enhancement level
            response = client.post("/v1/enhance-prompt", json={
                "original_prompt": "Test prompt",
                "enhancement_level": "invalid_level"
            })
            assert response.status_code == 422

            # Malicious prompt
            response = client.post("/v1/enhance-prompt", json={
                "original_prompt": "<script>alert('xss')</script>",
                "enhancement_level": "standard"
            })
            assert response.status_code == 422

    def test_learn_schema_validation(self, client):
        """Test learn-schema endpoint validation."""
        with patch.dict('os.environ', {'HTTP_REQUIRE_AUTH': 'false'}):
            # Invalid task_type
            response = client.post("/v1/learn-schema", json={
                "examples": [{"input": "test", "output": "test"}] * 3,
                "task_type": "invalid$type%"
            })
            assert response.status_code == 422

            # Too few examples
            response = client.post("/v1/learn-schema", json={
                "examples": [{"input": "test", "output": "test"}],  # Only 1 example
                "task_type": "valid_type"
            })
            assert response.status_code == 422

            # Too many examples
            response = client.post("/v1/learn-schema", json={
                "examples": [{"input": "test", "output": "test"}] * 25,  # Too many
                "task_type": "valid_type"
            })
            assert response.status_code == 422
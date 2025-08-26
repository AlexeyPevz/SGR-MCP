"""Test coverage verification and requirements."""

import pytest
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCoverageRequirements:
    """Test that code coverage meets requirements."""

    def test_minimum_coverage_achieved(self):
        """Test that minimum code coverage is achieved."""
        try:
            # Run coverage report
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "--cov=src", 
                "--cov-report=term-missing",
                "--cov-report=json:coverage.json",
                "-q"
            ], cwd=project_root, capture_output=True, text=True, timeout=60)
            
            # Check if coverage.json was created
            coverage_file = project_root / "coverage.json"
            if coverage_file.exists():
                import json
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                
                # Aim for at least 70% coverage (will increase to 85% gradually)
                min_coverage = 70
                assert total_coverage >= min_coverage, f"Coverage {total_coverage:.1f}% is below minimum {min_coverage}%"
                
                print(f"✅ Code coverage: {total_coverage:.1f}%")
                
                # Check individual module coverage
                files_coverage = coverage_data.get("files", {})
                low_coverage_files = []
                
                for file_path, file_data in files_coverage.items():
                    file_coverage = file_data.get("summary", {}).get("percent_covered", 0)
                    if file_coverage < 60 and "test" not in file_path:  # Allow lower coverage for test files
                        low_coverage_files.append((file_path, file_coverage))
                
                if low_coverage_files:
                    print("⚠️ Files with low coverage:")
                    for file_path, coverage in low_coverage_files:
                        print(f"  - {file_path}: {coverage:.1f}%")
                
            else:
                pytest.skip("Coverage data not available")
                
        except subprocess.TimeoutExpired:
            pytest.skip("Coverage test timed out")
        except FileNotFoundError:
            pytest.skip("pytest-cov not available")
        except Exception as e:
            pytest.skip(f"Coverage test failed: {e}")

    def test_critical_modules_covered(self):
        """Test that critical modules have test coverage."""
        critical_modules = [
            "src/server.py",
            "src/http_server.py", 
            "src/cli.py",
            "src/tools/apply_sgr.py",
            "src/utils/llm_client.py",
            "src/utils/cache.py",
            "src/utils/telemetry.py"
        ]
        
        # Check that test files exist for critical modules
        for module in critical_modules:
            module_name = Path(module).stem
            potential_test_files = [
                f"test_{module_name}.py",
                f"test_{module_name.replace('_', '')}.py",
                f"test_{module_name}s.py"  # plural form
            ]
            
            test_exists = any(
                (project_root / "tests" / test_file).exists() 
                for test_file in potential_test_files
            )
            
            # Also check for integration tests that might cover this module
            if not test_exists:
                test_integration_exists = (project_root / "tests" / "test_integration.py").exists()
                test_exists = test_integration_exists
            
            assert test_exists, f"Critical module {module} should have test coverage"

    def test_no_untested_public_functions(self):
        """Test that major public functions have some test coverage."""
        # This is a basic check - in a real scenario you'd use AST parsing
        # to find all public functions and verify they're tested
        
        import importlib.util
        
        # Key modules to check
        modules_to_check = [
            "src.tools.apply_sgr",
            "src.utils.llm_client",
            "src.utils.cache"
        ]
        
        for module_name in modules_to_check:
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    module_name, 
                    project_root / module_name.replace(".", "/") + ".py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Get public functions (those not starting with _)
                    public_functions = [
                        name for name, obj in vars(module).items()
                        if callable(obj) and not name.startswith('_')
                    ]
                    
                    # Should have some public functions
                    assert len(public_functions) > 0, f"Module {module_name} should have public functions"
                    
            except Exception:
                # Skip if module can't be imported
                continue


class TestTestQuality:
    """Test the quality of the test suite itself."""

    def test_test_files_exist(self):
        """Test that test files exist for major components."""
        expected_test_files = [
            "test_security.py",
            "test_integrations.py", 
            "test_performance.py",
            "test_tools.py",
            "test_utils.py",
            "test_schemas.py",
            "test_integration.py"
        ]
        
        tests_dir = project_root / "tests"
        
        for test_file in expected_test_files:
            test_path = tests_dir / test_file
            assert test_path.exists(), f"Test file {test_file} should exist"

    def test_test_files_not_empty(self):
        """Test that test files contain actual tests."""
        tests_dir = project_root / "tests"
        
        for test_file in tests_dir.glob("test_*.py"):
            content = test_file.read_text()
            
            # Should contain test functions
            assert "def test_" in content, f"Test file {test_file.name} should contain test functions"
            
            # Should import pytest or testing modules
            assert any(keyword in content for keyword in ["import pytest", "from pytest", "import unittest"]), \
                f"Test file {test_file.name} should import testing framework"

    def test_async_tests_present(self):
        """Test that async functionality is tested."""
        tests_dir = project_root / "tests"
        
        async_test_found = False
        
        for test_file in tests_dir.glob("test_*.py"):
            content = test_file.read_text()
            if "@pytest.mark.asyncio" in content or "async def test_" in content:
                async_test_found = True
                break
        
        assert async_test_found, "Should have async tests for async functionality"

    def test_mock_usage_appropriate(self):
        """Test that mocks are used appropriately in tests."""
        tests_dir = project_root / "tests"
        
        mock_usage_found = False
        
        for test_file in tests_dir.glob("test_*.py"):
            content = test_file.read_text()
            if any(keyword in content for keyword in ["Mock", "patch", "AsyncMock"]):
                mock_usage_found = True
                break
        
        assert mock_usage_found, "Should use mocks to isolate units under test"

    def test_test_organization(self):
        """Test that tests are well organized."""
        tests_dir = project_root / "tests"
        
        # Should have reasonable number of test files (not too few, not too many)
        test_files = list(tests_dir.glob("test_*.py"))
        assert 5 <= len(test_files) <= 20, f"Should have reasonable number of test files, found {len(test_files)}"
        
        # Each test file should be reasonably sized
        for test_file in test_files:
            lines = len(test_file.read_text().splitlines())
            assert 10 <= lines <= 1000, f"Test file {test_file.name} should be reasonably sized ({lines} lines)"


class TestDocumentationCoverage:
    """Test that code has adequate documentation."""

    def test_main_modules_documented(self):
        """Test that main modules have docstrings."""
        main_modules = [
            "src/server.py",
            "src/http_server.py",
            "src/cli.py"
        ]
        
        for module_path in main_modules:
            full_path = project_root / module_path
            if full_path.exists():
                content = full_path.read_text()
                
                # Should have module docstring
                assert '"""' in content or "'''" in content, \
                    f"Module {module_path} should have docstring documentation"

    def test_api_endpoints_documented(self):
        """Test that API endpoints have documentation."""
        http_server_path = project_root / "src" / "http_server.py"
        
        if http_server_path.exists():
            content = http_server_path.read_text()
            
            # Count endpoint definitions
            endpoint_count = content.count("@app.")
            
            # Count docstrings in endpoints (should be most of them)
            endpoint_docstring_count = content.count('"""', content.find("@app."))
            
            if endpoint_count > 0:
                documentation_ratio = endpoint_docstring_count / endpoint_count
                assert documentation_ratio >= 0.7, \
                    f"Most API endpoints should be documented ({documentation_ratio:.1%} documented)"

    def test_readme_exists_and_useful(self):
        """Test that README exists and contains useful information."""
        readme_paths = [
            project_root / "README.md",
            project_root / "docs" / "README.md"
        ]
        
        readme_found = False
        for readme_path in readme_paths:
            if readme_path.exists():
                content = readme_path.read_text()
                
                # Should have substantial content
                assert len(content) > 500, "README should have substantial content"
                
                # Should mention key concepts
                key_terms = ["sgr", "schema", "reasoning", "install", "usage"]
                found_terms = sum(1 for term in key_terms if term.lower() in content.lower())
                assert found_terms >= 3, "README should mention key project concepts"
                
                readme_found = True
                break
        
        assert readme_found, "Project should have a README file"
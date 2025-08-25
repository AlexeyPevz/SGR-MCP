#!/usr/bin/env python3
"""
Simple test script for MCP-SGR HTTP API.

Tests the main endpoints with example requests.
"""

import json
import requests
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8080"
API_KEY = None  # Set if HTTP_REQUIRE_AUTH=true

def make_request(endpoint: str, data: Dict[str, Any] = None, method: str = "POST") -> Dict[str, Any]:
    """Make HTTP request to MCP-SGR API."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        else:
            response = requests.post(url, json=data, headers=headers)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

def test_health():
    """Test health endpoint."""
    print("=== Testing Health Check ===")
    result = make_request("/health", method="GET")
    if result:
        print(f"✓ Server is {result.get('status', 'unknown')}")
    print()

def test_apply_sgr():
    """Test apply-sgr endpoint."""
    print("=== Testing Apply SGR ===")
    
    test_cases = [
        {
            "name": "Code Analysis",
            "data": {
                "task": "Review this Python function for security issues: def login(username, password): query = f'SELECT * FROM users WHERE name={username} AND pass={password}'",
                "schema_type": "analysis",
                "budget": "lite"
            }
        },
        {
            "name": "API Design",
            "data": {
                "task": "Design a REST API for a task management system",
                "schema_type": "planning",
                "context": {"scale": "medium", "users": "100K"}
            }
        },
        {
            "name": "Decision Making",
            "data": {
                "task": "Choose between PostgreSQL and MongoDB for an e-commerce platform",
                "schema_type": "decision",
                "context": {"requirements": ["ACID compliance", "horizontal scaling", "complex queries"]}
            }
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        result = make_request("/v1/apply-sgr", test['data'])
        
        if result:
            print(f"✓ Confidence: {result.get('confidence', 0):.2f}")
            if 'suggested_actions' in result:
                print(f"✓ Actions: {len(result['suggested_actions'])} suggested")
            if 'reasoning' in result:
                print(f"✓ Reasoning structure received")
        print("-" * 40)

def test_enhance_prompt():
    """Test enhance-prompt endpoint."""
    print("\n=== Testing Enhance Prompt ===")
    
    data = {
        "original_prompt": "Write a Python script to backup database",
        "target_model": "gemini-pro"
    }
    
    result = make_request("/v1/enhance-prompt", data)
    if result:
        print(f"✓ Enhanced prompt length: {len(result.get('enhanced_prompt', ''))} chars")
        if 'improvements' in result:
            print(f"✓ Improvements: {', '.join(result['improvements'][:3])}")
    print()

def test_schemas():
    """Test schemas endpoint."""
    print("=== Testing List Schemas ===")
    result = make_request("/v1/schemas", method="GET")
    if result and 'schemas' in result:
        print(f"✓ Available schemas: {len(result['schemas'])}")
        for schema in result['schemas']:
            print(f"  - {schema['name']}: {schema.get('description', '')[:50]}...")
    print()

def test_cache_stats():
    """Test cache stats endpoint."""
    print("=== Testing Cache Stats ===")
    result = make_request("/v1/cache-stats", method="GET")
    if result:
        print(f"✓ Cache hits: {result.get('total_hits', 0)}")
        print(f"✓ Cache misses: {result.get('total_misses', 0)}")
        print(f"✓ Hit rate: {result.get('hit_rate', 0):.1%}")
    print()

def main():
    """Run all tests."""
    print("🚀 MCP-SGR HTTP API Test Suite")
    print(f"📍 Testing server at: {BASE_URL}")
    print("=" * 50)
    
    # Run tests
    test_health()
    test_apply_sgr()
    test_enhance_prompt()
    test_schemas()
    test_cache_stats()
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main()
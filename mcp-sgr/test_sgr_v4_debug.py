#!/usr/bin/env python3
"""
Debug test for SGR v4 - проверяем, что именно возвращают модели
"""

import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('/workspace/mcp-sgr/src/tools')
from apply_sgr_v4 import detect_task_type, TASK_SCHEMAS

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Простая задача для теста
TEST_TASK = """Review this code snippet:

```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
```

Identify security issues."""


def call_model_debug(model: str, task: str) -> Tuple[str, float, int]:
    """Call model and return raw response."""
    
    start_time = time.time()
    
    schema = TASK_SCHEMAS["code_review"]
    
    # Сначала пробуем упрощенную схему
    simple_schema = {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {"type": "string"}
            },
            "severity": {"type": "string"}
        }
    }
    
    system_prompt = """You are a code security expert. Analyze the code and respond with JSON."""
    
    user_prompt = f"""Task: {task}

Respond with JSON matching this structure:
{json.dumps(simple_schema, indent=2)}

Focus on SQL injection vulnerabilities."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(data).encode('utf-8'),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("total_tokens", 0)
            return content, time.time() - start_time, tokens
            
    except Exception as e:
        return f"Error: {e}", time.time() - start_time, 0


def main():
    """Debug test."""
    
    print("\n🔍 SGR v4 Debug Test")
    print("=" * 80)
    
    models = [
        {"name": "Qwen-2.5-72B", "id": "qwen/qwen-2.5-72b-instruct"},
        {"name": "Mistral-7B", "id": "mistralai/mistral-7b-instruct"},
    ]
    
    for model in models:
        print(f"\n🤖 Testing {model['name']}...")
        print("-" * 60)
        
        response, duration, tokens = call_model_debug(model["id"], TEST_TASK)
        
        print(f"Duration: {duration:.1f}s, Tokens: {tokens}")
        print(f"\nRaw Response:")
        print(response)
        
        # Try to parse
        print(f"\nParsing attempt:")
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
                parsed = json.loads(json_str)
            elif response.strip().startswith("{"):
                parsed = json.loads(response)
            else:
                # Find JSON in response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(response[start:end])
                else:
                    parsed = None
            
            if parsed:
                print(f"✅ Successfully parsed:")
                print(json.dumps(parsed, indent=2))
            else:
                print("❌ Could not find JSON in response")
                
        except Exception as e:
            print(f"❌ Parse error: {e}")
        
        print("\n" + "="*80)
        time.sleep(2)


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("❌ Error: OPENROUTER_API_KEY not set")
        exit(1)
    
    main()
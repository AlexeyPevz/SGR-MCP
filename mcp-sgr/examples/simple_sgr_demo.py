#!/usr/bin/env python3
"""
Simple SGR Demo - No dependencies required!
Shows the power of Schema-Guided Reasoning with just Python stdlib
"""

import json
import urllib.request
import os
from typing import Dict, Optional

# Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-key-here")
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"  # Free model!

def apply_sgr(prompt: str, mode: str = "lite", model: str = DEFAULT_MODEL) -> Dict:
    """
    Apply Schema-Guided Reasoning to a prompt
    
    Args:
        prompt: The user's prompt
        mode: "off" (no SGR), "lite" (simple), or "full" (comprehensive)
        model: Model to use (default: free Mistral-7B)
    
    Returns:
        Dict with response and metadata
    """
    
    # Define SGR schemas
    schemas = {
        "lite": {
            "task_understanding": "What I understand you want me to do",
            "solution": "My solution or response"
        },
        "full": {
            "requirements_analysis": "Detailed analysis of what's needed",
            "approach": "How I'll approach this task",
            "implementation": "The actual solution/implementation",
            "validation": "How to verify this works correctly"
        }
    }
    
    # Enhance prompt with SGR
    if mode in schemas:
        enhanced_prompt = f"""{prompt}

Please structure your response as JSON with these fields:
{json.dumps(schemas[mode], indent=2)}"""
    else:
        enhanced_prompt = prompt
    
    # Prepare API request
    messages = [{"role": "user", "content": enhanced_prompt}]
    
    request_data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    # Make API call
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/mcp-sgr",
            "X-Title": "SGR Demo"
        },
        data=json.dumps(request_data).encode('utf-8')
    )
    
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode('utf-8'))
        
        content = result["choices"][0]["message"]["content"]
        
        # Try to parse as JSON if SGR was used
        structured_output = None
        if mode in schemas:
            try:
                # Handle markdown-wrapped JSON
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    json_str = content
                
                structured_output = json.loads(json_str)
            except:
                pass
        
        return {
            "content": content,
            "structured_output": structured_output,
            "mode": mode,
            "model": model,
            "success": True
        }
        
    except Exception as e:
        return {
            "content": f"Error: {str(e)}",
            "structured_output": None,
            "mode": mode,
            "model": model,
            "success": False
        }

def demo_comparison():
    """Demo showing the difference between with and without SGR"""
    
    # Test prompt
    prompt = "Write a Python function to check if a number is prime, with tests"
    
    print("=" * 60)
    print("🔬 SGR DEMO: Comparing outputs with and without SGR")
    print("=" * 60)
    
    # Test without SGR
    print("\n📝 WITHOUT SGR (baseline):")
    print("-" * 40)
    result_off = apply_sgr(prompt, mode="off")
    if result_off["success"]:
        print(result_off["content"][:500] + "..." if len(result_off["content"]) > 500 else result_off["content"])
    else:
        print(result_off["content"])
    
    # Test with SGR-Lite
    print("\n\n✨ WITH SGR-LITE:")
    print("-" * 40)
    result_lite = apply_sgr(prompt, mode="lite")
    if result_lite["success"]:
        if result_lite["structured_output"]:
            print("Structured output received! ✅")
            print(json.dumps(result_lite["structured_output"], indent=2))
        else:
            print("Raw output:")
            print(result_lite["content"][:500] + "..." if len(result_lite["content"]) > 500 else result_lite["content"])
    else:
        print(result_lite["content"])
    
    # Test with SGR-Full
    print("\n\n🚀 WITH SGR-FULL:")
    print("-" * 40)
    result_full = apply_sgr(prompt, mode="full")
    if result_full["success"]:
        if result_full["structured_output"]:
            print("Comprehensive structured output received! ✅")
            print(json.dumps(result_full["structured_output"], indent=2))
        else:
            print("Raw output:")
            print(result_full["content"][:500] + "..." if len(result_full["content"]) > 500 else result_full["content"])
    else:
        print(result_full["content"])
    
    print("\n" + "=" * 60)
    print("💡 INSIGHTS:")
    print("- SGR provides structured, complete responses")
    print("- SGR-Lite is perfect for most tasks")
    print("- SGR-Full gives comprehensive analysis")
    print("- All using a FREE model (Mistral-7B)")
    print("=" * 60)

def main():
    """Main demo function"""
    
    print("\n🚀 MCP-SGR Simple Demo")
    print("Using model:", DEFAULT_MODEL)
    
    if API_KEY == "your-key-here":
        print("\n⚠️  Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-actual-key'")
        print("\nGet a free key at: https://openrouter.ai")
        return
    
    # Run comparison demo
    demo_comparison()
    
    # Interactive mode
    print("\n\n💬 Try it yourself! (type 'quit' to exit)")
    while True:
        prompt = input("\nYour prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        mode = input("SGR mode (off/lite/full) [lite]: ").strip() or "lite"
        
        print("\nProcessing...")
        result = apply_sgr(prompt, mode)
        
        if result["success"]:
            if result["structured_output"]:
                print("\n✅ Structured Output:")
                print(json.dumps(result["structured_output"], indent=2))
            else:
                print("\n📝 Response:")
                print(result["content"])
        else:
            print(result["content"])

if __name__ == "__main__":
    main()
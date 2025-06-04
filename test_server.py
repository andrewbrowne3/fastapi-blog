#!/usr/bin/env python3
import requests
import json
import sys

def test_server():
    url = "http://localhost:7999/blog"
    
    # Test data using Claude (cloud provider)
    test_data = {
        "topic": "Benefits of Running",
        "llm_provider": "cloud",  # Use cloud instead of anthropic
        "model_name": "claude-3-haiku-20240307",
        "num_sections": 2,
        "target_audience": "general",
        "tone": "informative"
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token"
    }
    
    try:
        print("Testing server at:", url)
        print("Request data:", json.dumps(test_data, indent=2))
        
        response = requests.post(url, json=test_data, headers=headers, timeout=120)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ SUCCESS!")
            result = response.json()
            print("Blog generated successfully:")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Content length: {len(result.get('content', ''))}")
        else:
            print("❌ ERROR!")
            print("Response text:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error - Server not running on port 7999")
    except requests.exceptions.Timeout:
        print("❌ Timeout - Request took too long")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_server() 
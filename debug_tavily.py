import os
import requests
from langchain_community.tools.tavily_search import TavilySearchResults

# Test environment variables
print('TAVILY_API_KEY:', os.getenv('TAVILY_API_KEY')[:10] + '...' if os.getenv('TAVILY_API_KEY') else 'NOT SET')

# Test 1: Direct API call
print("\n=== Test 1: Direct Tavily API call ===")
try:
    api_key = os.getenv('TAVILY_API_KEY')
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": "test search",
        "max_results": 3
    }
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Direct API call successful!")
        print(f"Response keys: {list(response.json().keys())}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Direct API error: {e}")

# Test 2: LangChain TavilySearchResults without api_key parameter
print("\n=== Test 2: TavilySearchResults (no api_key param) ===")
try:
    search_tool = TavilySearchResults(max_results=3)
    results = search_tool.invoke({'query': 'test search'})
    print(f"Results type: {type(results)}")
    print(f"Results: {results}")
except Exception as e:
    print(f"LangChain error (no api_key): {e}")

# Test 3: LangChain TavilySearchResults with api_key parameter
print("\n=== Test 3: TavilySearchResults (with api_key param) ===")
try:
    api_key = os.getenv('TAVILY_API_KEY')
    search_tool = TavilySearchResults(api_key=api_key, max_results=3)
    results = search_tool.invoke({'query': 'test search'})
    print(f"Results type: {type(results)}")
    print(f"Results: {results}")
except Exception as e:
    print(f"LangChain error (with api_key): {e}")

# Test 4: Check if we need to use the new langchain-tavily package
print("\n=== Test 4: Check for langchain-tavily package ===")
try:
    from langchain_tavily import TavilySearch
    print("langchain-tavily package is available!")
    search_tool = TavilySearch(max_results=3)
    results = search_tool.invoke({'query': 'test search'})
    print(f"Results type: {type(results)}")
    print(f"Results: {results}")
except ImportError:
    print("langchain-tavily package not available")
except Exception as e:
    print(f"langchain-tavily error: {e}") 
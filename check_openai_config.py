#!/usr/bin/env python3
"""
Check OpenAI API key configuration without requiring the openai package
"""

import os
import json
import urllib.request
import urllib.error

def check_openai_config():
    """Check if OpenAI API key is configured"""
    
    # Check multiple sources for API key
    api_key = None
    base_url = None
    
    # 1. Check environment variables
    env_api_key = os.getenv('OPENAI_API_KEY')
    env_base_url = os.getenv('OPENAI_BASE_URL')
    
    # 2. Check config file
    config_path = '/workspace/memoryos-mcp/config.json'
    config_api_key = None
    config_base_url = None
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            config_api_key = config.get('openai_api_key')
            config_base_url = config.get('openai_base_url')
    
    # Determine which API key to use (env var takes precedence)
    api_key = env_api_key or config_api_key
    base_url = env_base_url or config_base_url or "https://api.openai.com/v1"
    
    print("=" * 60)
    print("OpenAI API Key Configuration Check")
    print("=" * 60)
    
    print("\n📋 Configuration Sources:")
    print(f"  Environment Variable (OPENAI_API_KEY): {'✅ Set' if env_api_key else '❌ Not set'}")
    print(f"  Config File (config.json): {'✅ Set' if config_api_key and config_api_key != '' else '❌ Not set or empty'}")
    
    if config_base_url and config_base_url != '':
        print(f"  Custom Base URL in config: {config_base_url}")
    
    if not api_key or api_key == '':
        print("\n❌ ERROR: No OpenAI API key found!")
        print("\n📝 To fix this, you can either:")
        print("  1. Set environment variable: export OPENAI_API_KEY='your-api-key'")
        print("  2. Update /workspace/memoryos-mcp/config.json: Set 'openai_api_key' field")
        return False
    
    print(f"\n✅ API Key found: {api_key[:10]}..." if len(api_key) > 10 else f"\n✅ API Key found: {api_key}")
    
    # Test the API key with a simple HTTP request
    print("\n🔍 Testing API connection...")
    
    try:
        # Prepare the request
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say 'API is working!' in exactly 3 words."}],
            "max_tokens": 10
        }).encode('utf-8')
        
        # Make the request
        req = urllib.request.Request(url, data=data, headers=headers)
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode('utf-8'))
        
        answer = result['choices'][0]['message']['content']
        print(f"\n🎉 SUCCESS! OpenAI API is responding!")
        print(f"   Response: {answer}")
        print(f"   Model used: {result['model']}")
        print(f"   Tokens used: {result['usage']['total_tokens']}")
        
        return True
        
    except urllib.error.HTTPError as e:
        print(f"\n❌ ERROR: Failed to connect to OpenAI API!")
        print(f"   HTTP Error {e.code}: {e.reason}")
        
        if e.code == 401:
            print("\n   ⚠️  The API key appears to be invalid or unauthorized.")
        elif e.code == 429:
            print("\n   ⚠️  Rate limit exceeded or quota exhausted.")
        elif e.code == 404:
            print("\n   ⚠️  The endpoint or model may not exist.")
        
        # Try to read error details
        try:
            error_data = json.loads(e.read().decode('utf-8'))
            if 'error' in error_data:
                print(f"   Details: {error_data['error'].get('message', 'No details available')}")
        except:
            pass
        
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to connect to OpenAI API!")
        print(f"   Error details: {str(e)}")
        
        if "connection" in str(e).lower():
            print("\n   ⚠️  Network connection issue. Check your internet connection.")
        
        return False

if __name__ == "__main__":
    success = check_openai_config()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ OpenAI API key is configured and working properly!")
        print("   The MemoryOS system should be able to use OpenAI for LLM operations.")
    else:
        print("❌ OpenAI API key is not working.")
        print("   The MemoryOS system will not be able to perform LLM operations.")
    print("=" * 60)
import os
import requests
from dotenv import load_dotenv

def test_deepseek_connection():
    """Test the connection to DeepSeek API."""
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('DEEPSEEK_API_KEY')
    api_url = os.getenv('DEEPSEEK_API_URL')
    
    print(f"\nTesting DeepSeek API Connection")
    print(f"API URL: {api_url}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Test data
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        print("\nMaking API request...")
        response = requests.post(f"{api_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        print("\n✅ API request successful!")
        print("\nResponse:")
        print(result)
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API request failed:")
        print(f"Error: {str(e)}")
        if response := getattr(e, 'response', None):
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        return False

if __name__ == "__main__":
    test_deepseek_connection() 
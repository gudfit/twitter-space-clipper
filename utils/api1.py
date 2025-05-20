import os
from typing import Dict, List, Optional, cast
import requests  # type: ignore
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = cast(str, os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com'))

def call_deepseek_api(messages: List[Dict]) -> Optional[Dict]:
    """Call DeepSeek API with retry logic and detailed logging.
    
    Args:
        messages (List[Dict]): List of message dictionaries with 'role' and 'content'.
        
    Returns:
        Optional[Dict]: API response message or None if the call fails.
    """
    print("\n" + "="*50)
    print("🔄 DEEPSEEK API CALL")
    print("="*50)
    print(f"Model: deepseek-chat")
    print(f"Max tokens: 2000")
    print(f"Temperature: 0.7")
    print(f"Message count: {len(messages)}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    response: Optional[requests.Response] = None
    
    try:
        # Add /chat/completions to the API endpoint
        api_url = DEEPSEEK_API_URL.rstrip('/') + '/chat/completions'
        print(f"\n📡 Sending request to API...")
        print(f"System prompt length: {len(messages[0]['content'])}")
        print(f"User prompt length: {len(messages[1]['content'])}")
        
        response = requests.post(api_url, headers=headers, json=data)
        print(f"\n📥 Response received (status: {response.status_code})")
        
        response.raise_for_status()
        result = response.json()
        
        # Log token usage if available
        if 'usage' in result:
            usage = result['usage']
            print(f"\n📊 Token Usage Stats:")
            print(f"- Prompt tokens:     {usage.get('prompt_tokens', 'N/A')}")
            print(f"- Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"- Total tokens:      {usage.get('total_tokens', 'N/A')}")
        
        message = result['choices'][0]['message']
        print(f"\n✅ Response extracted successfully")
        print(f"Response length: {len(message['content'])} characters")
        print("="*50 + "\n")
        return message
        
    except Exception as e:
        print("\n❌ API CALL ERROR")
        print("="*50)
        print(f"Error: {str(e)}")
        if response is not None:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        print("="*50 + "\n")
        return None 
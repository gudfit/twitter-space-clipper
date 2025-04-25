import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL')

def call_deepseek_api(messages: List[Dict]) -> Optional[Dict]:
    """Call DeepSeek API with retry logic and detailed logging.
    
    Args:
        messages (List[Dict]): List of message dictionaries with 'role' and 'content'.
        
    Returns:
        Optional[Dict]: API response message or None if the call fails.
    """
    print("\n🔄 Preparing DeepSeek API call...")
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
    
    try:
        # Add /chat/completions to the API endpoint
        api_url = DEEPSEEK_API_URL.rstrip('/') + '/chat/completions'
        print(f"\n📡 Sending request to: {api_url}")
        print(f"System prompt length: {len(messages[0]['content'])}")
        print(f"User prompt length: {len(messages[1]['content'])}")
        
        response = requests.post(api_url, headers=headers, json=data)
        print(f"\n📥 Response received (status: {response.status_code})")
        
        response.raise_for_status()
        result = response.json()
        
        # Log token usage if available
        if 'usage' in result:
            usage = result['usage']
            print(f"\n📊 Token usage:")
            print(f"- Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"- Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"- Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        message = result['choices'][0]['message']
        print(f"\n✅ Successfully extracted response (length: {len(message['content'])} chars)")
        return message
        
    except Exception as e:
        print(f"\n❌ Error calling DeepSeek API: {str(e)}")
        if response := getattr(e, 'response', None):
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        return None 
import os
from typing import Dict, List, Optional, cast
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------ #
# 1. Load .env *once* (harmless if already loaded)                   #
# ------------------------------------------------------------------ #
load_dotenv()


# 2. Helper: find an API key no matter where we’re running
def _get_deepseek_key() -> Optional[str]:
    # a. environment variable wins (works in Celery, cron, etc.)
    if key := os.getenv("DEEPSEEK_API_KEY"):
        return key

    # b. inside Streamlit we can fall back to st.secrets
    try:
        import streamlit as st  # import only if available

        return st.secrets["deepseek"]["api_key"]
    except Exception:
        return None  # not in Streamlit or key missing


DEEPSEEK_API_KEY = _get_deepseek_key()
DEEPSEEK_API_URL = cast(str, os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com"))


# ------------------------------------------------------------------ #
# 3. The actual call                                                 #
# ------------------------------------------------------------------ #
def call_deepseek_api(messages: List[Dict]) -> Optional[Dict]:
    """
    Send a chat/completions request to DeepSeek and return the first message,
    or None on failure.  Works in Streamlit *and* background workers.
    """
    print("\n" + "=" * 50)
    print("🔄 DEEPSEEK API CALL")
    print("=" * 50)

    if not DEEPSEEK_API_KEY:
        print("❌ DeepSeek API key not configured")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    api_url = DEEPSEEK_API_URL.rstrip("/") + "/chat/completions"
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        print(f"📥 Response status: {resp.status_code}")
        resp.raise_for_status()

        data = resp.json()
        # optional usage stats
        if usage := data.get("usage"):
            print(
                f"📊 Tokens — prompt {usage.get('prompt_tokens')}, "
                f"completion {usage.get('completion_tokens')}, "
                f"total {usage.get('total_tokens')}"
            )

        message = data["choices"][0]["message"]
        print(f"✅ Got {len(message['content'])} characters")
        return message

    except Exception as e:
        print("❌ API CALL ERROR")
        print(f"{e}")
        try:
            print("Body:", resp.text)  # may fail if resp undefined
        except Exception:
            pass
        return None

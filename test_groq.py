import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not set in .env")

client = Groq(api_key=GROQ_API_KEY)

# Test simple API call
print("Testing Groq API...")
try:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello!"}],
    )
    print(f"✅ API works! Response: {completion.choices[0].message.content}")
except Exception as e:
    print(f"❌ API Error: {e}")

# List available models
print("\nListing available models...")
try:
    for m in client.models.list().data:
        print(f"  - {m.id}")
except Exception as e:
    print(f"❌ Error listing models: {e}")

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Check for .env file
env_path = os.path.join(current_dir, '.env')
if os.path.exists(env_path):
    print(f"✅ Found .env file at: {env_path}")
else:
    print(f"❌ .env file not found at: {env_path}")

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Debug: Print API key status
if OPENAI_API_KEY:
    print("✅ API key loaded successfully")
    print(f"API key starts with: {OPENAI_API_KEY[:7]}...")
    print(f"API key length: {len(OPENAI_API_KEY)}")
else:
    print("❌ API key not found in environment variables")

# App Settings
MAX_HISTORY_ITEMS = 10
CHUNK_SIZE = 750
MAX_TOKENS_CHUNK = 300
MAX_TOKENS_FINAL = 500
TEMPERATURE = 0.5 
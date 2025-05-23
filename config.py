import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Debug: Print if .env file exists
if os.path.exists('.env'):
    print("Found .env file")
else:
    print("Warning: .env file not found")

# Debug: Print API key status
if OPENAI_API_KEY:
    print("API key loaded successfully")
else:
    print("Warning: API key not found in environment variables")

# App Settings
MAX_HISTORY_ITEMS = 10
CHUNK_SIZE = 750
MAX_TOKENS_CHUNK = 300
MAX_TOKENS_FINAL = 500
TEMPERATURE = 0.5 
import os
from dotenv import load_dotenv

def verify_api_key():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
        return
    
    # Check if API key exists
    if api_key:
        print("✅ API key found in environment variables")
        
        # Check API key format
        if api_key.startswith('sk-'):
            print("✅ API key format is correct (starts with 'sk-')")
        else:
            print("❌ API key format is incorrect (should start with 'sk-')")
            
        # Check API key length
        if len(api_key) >= 20:
            print("✅ API key length appears valid")
        else:
            print("❌ API key appears too short")
    else:
        print("❌ API key not found in environment variables")

if __name__ == "__main__":
    verify_api_key() 
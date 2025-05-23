# YouTube Video Summarizer

A Streamlit application that summarizes YouTube videos using OpenAI's GPT-3.5 and Whisper API.

## Features
- Summarizes YouTube videos in English
- Supports both transcript-based and audio-based summarization
- Text-to-speech functionality for summaries
- Download summaries as text or audio
- Search history tracking

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd youtube-summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```

4. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Configuration

The application can be configured through the `config.py` file:

- `MAX_HISTORY_ITEMS`: Number of recent searches to keep (default: 10)
- `CHUNK_SIZE`: Size of text chunks for processing (default: 750)
- `MAX_TOKENS_CHUNK`: Maximum tokens for chunk summaries (default: 300)
- `MAX_TOKENS_FINAL`: Maximum tokens for final summary (default: 500)
- `TEMPERATURE`: AI response temperature (default: 0.5)

## Deployment Instructions

### Deploy to Streamlit Cloud (Free)

1. Create a GitHub account if you don't have one
2. Create a new repository on GitHub
3. Push your code to the repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

4. Go to [Streamlit Cloud](https://streamlit.io/cloud)
5. Sign in with your GitHub account
6. Click "New app"
7. Select your repository, branch, and main file (app.py)
8. Add your secrets:
   - Go to your app's settings
   - Add your OpenAI API key as a secret:
     - Key: `OPENAI_API_KEY`
     - Value: Your OpenAI API key

### Local Development

Run the app:
```bash
streamlit run app.py
```

## Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key

## Note
Make sure to keep your API keys secure and never commit them to version control. 
import streamlit as st

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

import openai
import time
from pytube import YouTube
import os
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import tempfile
import requests
from urllib.error import HTTPError
import re
import io
import json
from datetime import datetime
from gtts import gTTS
import base64
from config import (
    OPENAI_API_KEY,
    MAX_HISTORY_ITEMS,
    CHUNK_SIZE,
    MAX_TOKENS_CHUNK,
    MAX_TOKENS_FINAL,
    TEMPERATURE
)

# Debug: Check API key format
if OPENAI_API_KEY:
    if not OPENAI_API_KEY.startswith('sk-'):
        st.error("Invalid API key format. API key should start with 'sk-'")
        st.stop()
    if len(OPENAI_API_KEY) < 20:  # Basic length check
        st.error("Invalid API key format. API key appears to be too short")
        st.stop()
else:
    st.error("OpenAI API key not found. Please set it in your .env file")
    st.stop()

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Constants
HISTORY_FILE = "search_history.json"

def load_history():
    """Load search history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load history: {str(e)}")
    return []

def save_history(history):
    """Save search history to file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Could not save history: {str(e)}")

# Initialize session state for storing search history and audio data
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

def save_to_history(video_url, summary):
    """Save the search to history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        'timestamp': timestamp,
        'url': video_url,
        'summary': summary
    }
    st.session_state.search_history.insert(0, history_entry)  # Add to beginning of list
    # Keep only last MAX_HISTORY_ITEMS searches
    if len(st.session_state.search_history) > MAX_HISTORY_ITEMS:
        st.session_state.search_history.pop()
    # Save to file
    save_history(st.session_state.search_history)

def clear_history():
    """Clear the search history"""
    st.session_state.search_history = []
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.success("History cleared successfully!")

def display_history():
    """Display the search history"""
    if st.session_state.search_history:
        # Add clear history button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
                <h3 style='color: #FF0000; margin-top: 2rem;'>Recent Searches</h3>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("🗑️ Clear History", help="Clear all saved searches"):
                clear_history()
                st.rerun()
        
        for entry in st.session_state.search_history:
            with st.expander(f"📺 {entry['url']} ({entry['timestamp']})"):
                st.markdown(entry['summary'])
                # Add download button for each history entry
                summary_bytes = entry['summary'].encode('utf-8')
                st.download_button(
                    label="📥 Download Summary",
                    data=summary_bytes,
                    file_name=f"video_summary_{entry['timestamp'].replace(':', '-')}.txt",
                    mime="text/plain",
                    help="Click to download the summary as a text file"
                )

def validate_youtube_url(url):
    """Validate if the URL is a valid YouTube URL"""
    if not url:
        return False, "Please enter a YouTube URL"
    url = url.strip()
    # Remove any timestamp or extra parameters
    if '&' in url:
        url = url.split('&')[0]
    # Common YouTube URL patterns
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=([^&=%\?]{11})',  # Standard watch URL
        r'(https?://)?(www\.)?youtube\.com/v/([^&=%\?]{11})',         # Video URL
        r'(https?://)?(www\.)?youtube\.com/embed/([^&=%\?]{11})',     # Embed URL
        r'(https?://)?(www\.)?youtu\.be/([^&=%\?]{11})',             # Short URL
        r'(https?://)?(www\.)?youtube\.com/shorts/([^&=%\?]{11})'     # Shorts URL
    ]
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            video_id = match.group(3)
            try:
                response = requests.head(f"https://www.youtube.com/watch?v={video_id}", allow_redirects=True)
                if response.status_code == 200:
                    return True, video_id
                elif response.status_code == 404:
                    return False, "Video not found. The video may have been removed or made private."
                else:
                    return False, f"Video URL returned status code {response.status_code}. Please check if the video is available."
            except requests.RequestException:
                return True, video_id
    return False, "Please enter a valid YouTube URL (e.g., https://www.youtube.com/watch?v=...)"

def download_video_with_retry(url, max_retries=3, delay=2):
    """Download video with retry logic"""
    is_valid, result = validate_youtube_url(url)
    if not is_valid:
        raise ValueError(result)
        
    video_id = result  # The validated video ID
    
    for attempt in range(max_retries):
        try:
            yt = YouTube(url)
            
            # Get video info first to check if it's available
            try:
                video_length = yt.length
                if video_length > 3600:  # If video is longer than 1 hour
                    st.warning(f"This is a long video ({video_length//3600} hours {video_length%3600//60} minutes). Processing may take some time...")
                    # For very long videos, we'll use the lowest quality audio
                    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()
                else:
                    audio_stream = yt.streams.filter(only_audio=True).first()
            except Exception as e:
                st.warning("Could not get video length. Proceeding with download...")
                audio_stream = yt.streams.filter(only_audio=True).order_by('abr').first()
            
            if not audio_stream:
                raise Exception("No audio stream found for this video")
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.close()
            
            # Download the audio with progress tracking
            with st.spinner(f"Downloading video audio (Attempt {attempt + 1}/{max_retries})..."):
                audio_stream.download(filename=temp_file.name)
                
            # Verify the file was downloaded and has content
            if os.path.getsize(temp_file.name) == 0:
                raise Exception("Downloaded file is empty")
                
            return temp_file.name
            
        except HTTPError as e:
            if e.code == 403:
                if attempt < max_retries - 1:
                    st.warning(f"Download attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception("Video download failed after multiple attempts. This might be due to YouTube's restrictions. Please try again later or use a different video.")
            elif e.code == 400:
                raise ValueError("Invalid video URL or video is not available. Please check the URL and try again.")
            else:
                raise Exception(f"HTTP Error {e.code}: {str(e)}")
        except Exception as e:
            if "Video unavailable" in str(e):
                raise ValueError("This video is unavailable. It may have been removed or made private.")
            elif "Sign in to confirm your age" in str(e):
                raise ValueError("This video requires age verification and cannot be processed.")
            elif "Video is private" in str(e):
                raise ValueError("This video is private and cannot be accessed.")
            elif "File too large" in str(e):
                raise ValueError("This video is too large to process. Please try a shorter video.")
            elif attempt < max_retries - 1:
                st.warning(f"Download attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            else:
                raise Exception(f"Failed to download video: {str(e)}")
                
    raise Exception("Failed to download video after all retry attempts")

def get_transcript(video_url, progress_bar=None, status_text=None):
    try:
        # Validate URL
        is_valid, result = validate_youtube_url(video_url)
        if not is_valid:
            st.error(result)
            return None
            
        video_id = result  # The validated video ID
        
        # First try to get the transcript using YouTube Transcript API
        try:
            # Try to get Hindi transcript first, then English
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
                if status_text: status_text.text("Found Hindi transcript!")
            except NoTranscriptFound:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                    if status_text: status_text.text("Found English transcript!")
                except NoTranscriptFound:
                    raise NoTranscriptFound("No Hindi or English transcript found")
            
            # Process transcript in chunks to avoid memory issues
            chunk_size = 100  # Process 100 transcript entries at a time
            transcript_parts = []
            num_chunks = (len(transcript) + chunk_size - 1) // chunk_size
            for i in range(0, len(transcript), chunk_size):
                chunk = transcript[i:i + chunk_size]
                chunk_text = " ".join([entry["text"] for entry in chunk])
                transcript_parts.append(chunk_text)
                if num_chunks > 1 and progress_bar and status_text:
                    progress = 5 + int(45 * (i + len(chunk)) / len(transcript))
                    progress_bar.progress(progress)
                    status_text.text(f"Processing transcript chunk {i//chunk_size+1} of {num_chunks}... ({progress}%)")
            
            return " ".join(transcript_parts)
            
        except Exception as e:
            # Handle empty or malformed transcript responses gracefully
            if "Could not find a transcript" in str(e) or "no element found" in str(e):
                if status_text: status_text.text("No transcript found or YouTube returned an empty response. Attempting to transcribe audio...")
            else:
                if status_text: status_text.text(f"Error getting transcript: {str(e)}")
                st.error(f"Error getting transcript: {str(e)}")
                return None
            
        # If no transcript found, try to get it from the video description
        try:
            yt = YouTube(video_url)
            if yt.description:
                if status_text: status_text.text("Found video description. Using it as a fallback...")
                return yt.description
        except Exception as e:
            st.warning("Could not get video description. Proceeding with audio transcription...")
            
        # If all else fails, download and transcribe the audio
        try:
            temp_file = download_video_with_retry(video_url)
        except ValueError as e:
            st.error(str(e))
            return None
        except Exception as e:
            st.error(f"Could not download the video: {str(e)}")
            return None
        
        try:
            # For very long videos, process in chunks
            file_size = os.path.getsize(temp_file)
            chunk_size = 5 * 1024 * 1024  # 5MB
            total_chunks = (file_size + chunk_size - 1) // chunk_size
            if total_chunks > 1:
                if status_text: status_text.text(f"Processing video in {total_chunks} chunks...")
            
            # Add language selection
            language = st.radio(
                "Select audio language for transcription:",
                ["Hindi", "English"],
                horizontal=True
            )
            
            # Map language selection to Whisper language code
            language_code = "hi" if language == "Hindi" else "en"
            
            transcript_parts = []
            for chunk_num in range(total_chunks):
                with st.spinner(f"Processing chunk {chunk_num + 1} of {total_chunks}..."):
                    with open(temp_file, "rb") as audio_file:
                        # Seek to the start of the chunk
                        audio_file.seek(chunk_num * chunk_size)
                        # Read the chunk
                        chunk_data = audio_file.read(chunk_size)
                        
                        # Create a temporary file for the chunk
                        chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        chunk_file.write(chunk_data)
                        chunk_file.close()
                        
                        try:
                            # Transcribe the chunk with retry logic
                            max_retries = 3
                            for retry in range(max_retries):
                                try:
                                    with open(chunk_file.name, "rb") as chunk_audio:
                                        chunk_transcript = openai.Audio.transcribe(
                                            model="whisper-1",
                                            file=chunk_audio,
                                            language=language_code
                                        )
                                    transcript_parts.append(chunk_transcript.text)
                                    break
                                except Exception as e:
                                    if retry < max_retries - 1:
                                        st.warning(f"Retrying chunk {chunk_num + 1} (attempt {retry + 2}/{max_retries})...")
                                        time.sleep(2)  # Wait before retry
                                    else:
                                        raise e
                            
                        except Exception as e:
                            st.warning(f"Error processing chunk {chunk_num + 1}: {str(e)}")
                            continue
                        finally:
                            # Clean up the chunk file
                            if os.path.exists(chunk_file.name):
                                os.unlink(chunk_file.name)
            
            # Clean up the main temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
            if not transcript_parts:
                raise Exception("No transcript parts were successfully processed")
            
            return " ".join(transcript_parts)
            
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            st.error(f"Error during transcription: {str(e)}")
            return None
            
    except Exception as e:
        if status_text: status_text.text(f"An error occurred: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        return None

def summarize_transcript(transcript_text, progress_bar=None, status_text=None):
    if not transcript_text:
        return None
    max_chunk_size = 1000
    chunks = []
    current_chunk = []
    current_size = 0
    sentences = transcript_text.split('. ')
    for sentence in sentences:
        if not sentence.endswith('.'): sentence += '.'
        sentence_tokens = len(sentence) // 1.5
        if sentence_tokens > max_chunk_size:
            for i in range(0, len(sentence), int(max_chunk_size * 1.5)):
                part = sentence[i:i+int(max_chunk_size*1.5)]
                if len(part) > 0:
                    chunks.append(part)
            continue
        if current_size + sentence_tokens > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_size += sentence_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    num_chunks = len(chunks)  # Ensure num_chunks is always defined
    summaries = []
    try:
        for i, chunk in enumerate(chunks):
            if num_chunks > 1 and progress_bar and status_text:
                progress = 60 + int(40 * (i + 1) / num_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Summarizing chunk {i+1} of {num_chunks}... ({progress}%)")
            try:
                system_message = "Summarize."
                user_message = chunk
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=100,
                    temperature=TEMPERATURE
                )
                summary = response.choices[0].message.content.strip()
                summaries.append(summary)
            except Exception as e:
                st.warning(f"Error processing chunk {i + 1}: {str(e)}")
                continue
        if summaries:
            combined_summary = " ".join(summaries)
            if len(combined_summary) > max_chunk_size:
                summary_parts = [combined_summary[i:i+max_chunk_size] for i in range(0, len(combined_summary), max_chunk_size)]
                final_summaries = []
                for i, part in enumerate(summary_parts):
                    try:
                        system_message = "Summarize."
                        user_message = part
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message}
                            ],
                            max_tokens=150,
                            temperature=TEMPERATURE
                        )
                        final_summaries.append(response.choices[0].message.content.strip())
                    except Exception as e:
                        st.warning(f"Error creating final summary part {i + 1}: {str(e)}")
                        continue
                final_summary = " ".join(final_summaries)
            else:
                try:
                    system_message = "Summarize."
                    user_message = combined_summary
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=150,
                        temperature=TEMPERATURE
                    )
                    final_summary = response.choices[0].message.content.strip()
                except Exception as e:
                    raise Exception(f"An error occurred while creating the final summary: {e}")
            return final_summary
        return None
    except Exception as e:
        if status_text: status_text.text(str(e))
        st.error(str(e))
        return None

def text_to_speech(text):
    """Convert text to speech and return audio data"""
    try:
        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_file.name)
        
        # Read the audio file
        with open(temp_file.name, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        return audio_data
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def autoplay_audio(audio_data):
    """Create an HTML audio player with autoplay"""
    b64 = base64.b64encode(audio_data).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF0000;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #CC0000;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSuccess {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .stError {
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #FF0000; margin-bottom: 2rem;'>
        YouTube Video Summarizer
    </h1>
    """, unsafe_allow_html=True)

# Description
st.markdown("""
    <p style='text-align: center; margin-bottom: 2rem;'>
        Enter a YouTube video URL below to get an AI-generated summary of its content.
    </p>
    """, unsafe_allow_html=True)

# Create two columns for better layout
col1, col2, col3 = st.columns([1,2,1])

with col2:
    # Input for YouTube URL
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_input"
    )

    if video_url:
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text("Fetching transcript...")
            progress_bar.progress(5)
            transcript = get_transcript(video_url, progress_bar, status_text)
            if transcript:
                status_text.text("Summarizing transcript...")
                progress_bar.progress(60)
                summary = summarize_transcript(transcript, progress_bar, status_text)
                if summary:
                    status_text.text("Summary ready!")
                    progress_bar.progress(100)
                    st.success("✨ Summary ready!")
                    st.markdown("""
                        <div style='background-color: #f0f2f6; padding: 1.5rem; border-radius: 5px; margin-top: 1rem;'>
                            <h3 style='color: #FF0000; margin-bottom: 1rem;'>Video Summary</h3>
                            <p style='line-height: 1.6;'>{}</p>
                        </div>
                    """.format(summary), unsafe_allow_html=True)
                    if st.button("🔊 Read Summary", help="Click to hear the summary read aloud"):
                        with st.spinner("Generating audio..."):
                            st.session_state.audio_data = text_to_speech(summary)
                            if st.session_state.audio_data:
                                autoplay_audio(st.session_state.audio_data)
                    col1, col2 = st.columns(2)
                    with col1:
                        summary_bytes = summary.encode('utf-8')
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary_bytes,
                            file_name="video_summary.txt",
                            mime="text/plain",
                            help="Click to download the summary as a text file"
                        )
                    with col2:
                        if st.session_state.audio_data:
                            st.download_button(
                                label="📥 Download Audio",
                                data=st.session_state.audio_data,
                                file_name="video_summary.mp3",
                                mime="audio/mp3",
                                help="Click to download the audio version of the summary"
                            )
                    save_to_history(video_url, summary)
                else:
                    status_text.text("Failed to generate summary.")
                    progress_bar.progress(0)
                    st.error("❌ Failed to generate summary.")
            else:
                status_text.text("Could not process the video. Please check if the video is available and try again.")
                progress_bar.progress(0)
                st.error("❌ Could not process the video. Please check if the video is available and try again.")
        finally:
            progress_bar.empty()
            status_text.empty()

# Display search history
display_history()

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 3rem; color: #666;'>
        <p>Powered by OpenAI GPT-3.5 and Whisper API</p>
    </div>
    """, unsafe_allow_html=True) 
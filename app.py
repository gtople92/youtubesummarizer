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

def get_video_metadata(video_url):
    """Get video metadata including description and comments"""
    try:
        # Add retry logic for YouTube API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                yt = YouTube(video_url)
                
                # Basic metadata that's usually available
                metadata = {
                    'title': None,
                    'description': None,
                    'length': None,
                    'author': None,
                    'views': None,
                    'comments': []
                }
                
                # Try to get basic metadata first
                try:
                    metadata['title'] = yt.title
                except:
                    pass
                    
                try:
                    metadata['author'] = yt.author
                except:
                    pass
                    
                try:
                    metadata['views'] = yt.views
                except:
                    pass
                    
                try:
                    metadata['length'] = yt.length
                except:
                    pass
                    
                try:
                    metadata['description'] = yt.description
                except:
                    pass
                
                # Only try to get comments if we have basic metadata
                if any(metadata.values()):
                    try:
                        for comment in yt.comments[:20]:  # Reduced to 20 comments for better performance
                            metadata['comments'].append({
                                'text': comment.text,
                                'author': comment.author,
                                'likes': comment.likes
                            })
                    except Exception as comment_error:
                        st.warning("Could not fetch comments. Proceeding with available information.")
                
                # Return metadata if we have at least some information
                if any(metadata.values()):
                    return metadata
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    raise e
                    
        return None
        
    except Exception as e:
        st.warning(f"Could not get complete video metadata: {str(e)}")
        return None

def get_transcript_with_timestamps(video_url, progress_bar=None, status_text=None):
    try:
        # Validate URL
        is_valid, result = validate_youtube_url(video_url)
        if not is_valid:
            st.error(result)
            return None
            
        video_id = result  # The validated video ID
        
        # Get video metadata first
        if status_text: status_text.text("Getting video information...")
        metadata = get_video_metadata(video_url)
        
        if metadata:
            # Create a more resilient display that only shows available information
            metadata_display = []
            
            if metadata['title']:
                metadata_display.append(f"<h3 style='color: #FF0000; margin-bottom: 0.5rem;'>{metadata['title']}</h3>")
            
            author_views = []
            if metadata['author']:
                author_views.append(metadata['author'])
            if metadata['views']:
                author_views.append(f"{metadata['views']:,} views")
            
            if author_views:
                metadata_display.append(f"<p style='color: #666;'>By {' • '.join(author_views)}</p>")
            
            if metadata_display:
                st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
                        {''.join(metadata_display)}
                    </div>
                """, unsafe_allow_html=True)
        
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
            
            return transcript
            
        except NoTranscriptFound as e:
            if status_text: status_text.text("No transcript found. Using alternative content...")
            
            # Create content from metadata
            content_parts = []
            
            if metadata:
                # Add description
                if metadata['description']:
                    content_parts.append(f"Video Description:\n{metadata['description']}")
                
                # Add comments
                if metadata['comments']:
                    content_parts.append("\nTop Comments:")
                    for comment in metadata['comments']:
                        content_parts.append(f"- {comment['text']}")
            
            if content_parts:
                if status_text: status_text.text("Using video description and comments as content...")
                # Create a transcript-like structure
                return [{
                    "text": "\n\n".join(content_parts),
                    "start": 0,
                    "duration": 0
                }]
            else:
                st.warning("""
                    ⚠️ No subtitles available for this video.
                    We'll need to transcribe the audio, which may take longer and be less accurate.
                """)
            
        except Exception as e:
            if "Could not find a transcript" in str(e) or "no element found" in str(e):
                if status_text: status_text.text("No transcript found. Proceeding with audio transcription...")
            else:
                if status_text: status_text.text(f"Error getting transcript: {str(e)}")
                st.error(f"Error getting transcript: {str(e)}")
                return None
            
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
            current_time = 0
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
                                    # Add timestamp to the transcript
                                    transcript_parts.append({
                                        "text": chunk_transcript.text,
                                        "start": current_time,
                                        "duration": 30  # Approximate duration for each chunk
                                    })
                                    current_time += 30
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
            
            return transcript_parts
            
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

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def wait_for_rate_limit(e):
    """Extract wait time from rate limit error and return it"""
    try:
        # Extract wait time from error message
        wait_time = int(re.search(r'Please try again in (\d+)ms', str(e)).group(1))
        # Convert to seconds and add a small buffer
        return (wait_time / 1000) + 0.1
    except:
        # Default wait time if we can't parse the error
        return 1

def call_openai_with_retry(messages, max_retries=5, initial_delay=1):
    """Call OpenAI API with retry logic and exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=100,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "Rate limit" in str(e):
                wait_time = wait_for_rate_limit(e)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
            raise e
    return None

def summarize_transcript_by_intervals(transcript, interval_minutes=5, progress_bar=None, status_text=None):
    if not transcript:
        return None
    
    # Convert interval from minutes to seconds
    interval_seconds = interval_minutes * 60
    
    # Group transcript segments by time intervals
    intervals = []
    current_interval = []
    current_interval_start = 0
    
    for segment in transcript:
        if segment["start"] >= current_interval_start + interval_seconds:
            if current_interval:
                intervals.append({
                    "start": current_interval_start,
                    "end": current_interval_start + interval_seconds,
                    "text": " ".join([s["text"] for s in current_interval])
                })
            current_interval = [segment]
            current_interval_start = segment["start"]
        else:
            current_interval.append(segment)
    
    # Add the last interval if it exists
    if current_interval:
        intervals.append({
            "start": current_interval_start,
            "end": current_interval_start + interval_seconds,
            "text": " ".join([s["text"] for s in current_interval])
        })
    
    # Generate summaries for each interval
    interval_summaries = []
    for i, interval in enumerate(intervals):
        if progress_bar and status_text:
            progress = 60 + int(40 * (i + 1) / len(intervals))
            progress_bar.progress(progress)
            status_text.text(f"Summarizing interval {i+1} of {len(intervals)}... ({progress}%)")
        
        try:
            system_message = "Summarize this segment of the video concisely."
            user_message = interval["text"]
            
            # Use the retry logic for API calls
            summary = call_openai_with_retry([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ])
            
            if summary:
                interval_summaries.append({
                    "start": format_timestamp(interval["start"]),
                    "end": format_timestamp(interval["end"]),
                    "summary": summary
                })
            else:
                st.warning(f"Could not generate summary for interval {i+1}. Skipping...")
                continue
                
        except Exception as e:
            if "Rate limit" in str(e):
                st.warning(f"Rate limit reached. Waiting before retrying...")
                time.sleep(wait_for_rate_limit(e))
                # Retry this interval
                i -= 1
                continue
            else:
                st.warning(f"Error summarizing interval {i+1}: {str(e)}")
                continue
    
    return interval_summaries

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
    /* Reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Button styles */
    .stButton>button {
        width: 100%;
        background-color: #FF0000;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        -webkit-transition: background-color 0.3s ease;
        -moz-transition: background-color 0.3s ease;
        -o-transition: background-color 0.3s ease;
        transition: background-color 0.3s ease;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        background-color: #CC0000;
    }
    
    /* Input styles */
    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 0.5rem;
        width: 100%;
        -webkit-appearance: none;
        -moz-appearance: none;
        appearance: none;
    }
    
    /* Message styles */
    .stSuccess, .stError {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stSuccess {
        background-color: #f0f2f6;
    }
    
    /* Summary container */
    .summary-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 5px;
        margin-top: 1rem;
        -webkit-box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        -moz-box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Header styles */
    .app-header {
        text-align: center;
        color: #FF0000;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    /* Description styles */
    .app-description {
        text-align: center;
        margin-bottom: 2rem;
        color: #333;
        line-height: 1.6;
    }
    
    /* Footer styles */
    .app-footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        padding: 1rem 0;
        border-top: 1px solid #eee;
    }
    
    /* Responsive layout */
    @media screen and (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        .app-header {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="app-header">
        YouTube Video Summarizer
    </div>
    """, unsafe_allow_html=True)

# Description
st.markdown("""
    <div class="app-description">
        Enter a YouTube video URL below to get an AI-generated summary of its content.
    </div>
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
            
            # Add interval selection
            interval_minutes = st.slider(
                "Select time interval for summaries (minutes):",
                min_value=1,
                max_value=30,
                value=5,
                step=1
            )
            
            transcript = get_transcript_with_timestamps(video_url, progress_bar, status_text)
            if transcript:
                status_text.text("Generating interval summaries...")
                progress_bar.progress(60)
                interval_summaries = summarize_transcript_by_intervals(
                    transcript, 
                    interval_minutes, 
                    progress_bar, 
                    status_text
                )
                
                if interval_summaries:
                    status_text.text("Summary ready!")
                    progress_bar.progress(100)
                    st.success("✨ Summary ready!")
                    
                    # Display interval summaries
                    st.markdown("""
                        <div class="summary-container">
                            <h3 style='color: #FF0000; margin-bottom: 1rem;'>Video Summary by Time Intervals</h3>
                    """, unsafe_allow_html=True)
                    
                    for interval in interval_summaries:
                        st.markdown(f"""
                            <div style='margin-bottom: 1.5rem; padding: 1rem; background-color: white; border-radius: 5px;'>
                                <h4 style='color: #666; margin-bottom: 0.5rem;'>
                                    {interval['start']} - {interval['end']}
                                </h4>
                                <p style='line-height: 1.6;'>{interval['summary']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add download options
                    col1, col2 = st.columns(2)
                    with col1:
                        # Create formatted text for download
                        summary_text = "Video Summary by Time Intervals\n\n"
                        for interval in interval_summaries:
                            summary_text += f"[{interval['start']} - {interval['end']}]\n"
                            summary_text += f"{interval['summary']}\n\n"
                        
                        summary_bytes = summary_text.encode('utf-8')
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary_bytes,
                            file_name="video_summary.txt",
                            mime="text/plain",
                            help="Click to download the summary as a text file"
                        )
                    
                    with col2:
                        if st.button("🔊 Read Summary", help="Click to hear the summary read aloud"):
                            with st.spinner("Generating audio..."):
                                # Combine all summaries for audio
                                full_summary = "\n\n".join([
                                    f"From {interval['start']} to {interval['end']}: {interval['summary']}"
                                    for interval in interval_summaries
                                ])
                                st.session_state.audio_data = text_to_speech(full_summary)
                                if st.session_state.audio_data:
                                    autoplay_audio(st.session_state.audio_data)
                                    st.download_button(
                                        label="📥 Download Audio",
                                        data=st.session_state.audio_data,
                                        file_name="video_summary.mp3",
                                        mime="audio/mp3",
                                        help="Click to download the audio version of the summary"
                                    )
                    
                    save_to_history(video_url, summary_text)
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
    <div class="app-footer">
        <p>Powered by OpenAI GPT-3.5 and Whisper API</p>
    </div>
    """, unsafe_allow_html=True) 
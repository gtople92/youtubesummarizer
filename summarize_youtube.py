import argparse
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import os
import openai

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    # Patterns for standard and short URLs
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",  # Standard and embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def main():
    parser = argparse.ArgumentParser(description="Summarize a YouTube video.")
    parser.add_argument("url", help="YouTube video URL")
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    if video_id:
        print(f"Extracted video ID: {video_id}")
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry['text'] for entry in transcript])
            print(f"\nTranscript:")
            print(transcript_text)

            # Summarize with OpenAI
            api_key = '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'
            client = openai.OpenAI(api_key=api_key)
            prompt = (
                "Summarize the following YouTube video transcript in 2-3 paragraphs. "
                "Focus on the main points and key takeaways.\n\nTranscript:\n" + transcript_text
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes YouTube videos."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.5
                )
                summary = response.choices[0].message.content.strip()
                print("\nSummary:")
                print(summary)
            except Exception as e:
                print(f"An error occurred while summarizing with OpenAI: {e}")
        except (TranscriptsDisabled, NoTranscriptFound):
            print("Transcript is not available for this video.")
        except Exception as e:
            print(f"An error occurred while fetching the transcript: {e}")
    else:
        print("Could not extract video ID from the provided URL.")

if __name__ == "__main__":
    main()

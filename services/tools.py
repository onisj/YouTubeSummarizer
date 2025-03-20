from transformers import pipeline
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from serpapi import GoogleSearch
from gtts import gTTS
from fastapi import HTTPException
from dotenv import load_dotenv
import os

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Device detection
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device = get_device()
print(f"Using device: {device}")

# Initialize pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device if device != "cpu" else -1)
sentiment_analyzer = pipeline("sentiment-analysis", device=device if device != "cpu" else -1)

def search_video(query: str) -> str:
    """Search for a YouTube video URL."""
    params = {"engine": "youtube", "search_query": query, "api_key": SERPAPI_KEY}
    search = GoogleSearch(params)
    results = search.get_dict()
    try:
        return results["video_results"][0]["link"]
    except (IndexError, KeyError):
        raise HTTPException(status_code=404, detail="No video found.")

def get_transcript(video_url: str) -> str:
    """Extract transcript from a YouTube video."""
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcript unavailable: {str(e)}")

def summarize_text(text: str) -> dict:
    """Summarize text and analyze sentiment."""
    max_input_length = 4000
    truncated = len(text) > max_input_length
    if truncated:
        text = text[:max_input_length]
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    sentiment = sentiment_analyzer(summary)[0]["label"]
    return {"summary": summary, "sentiment": sentiment, "text": text if truncated else text}

def text_to_speech(text: str) -> str:
    """Convert text to an audio file."""
    audio_file = "static/summary.mp3"
    tts = gTTS(text)
    tts.save(audio_file)
    return audio_file
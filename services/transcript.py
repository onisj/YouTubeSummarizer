from fastapi import HTTPException
from youtube_transcript_api import YouTubeTranscriptApi

def get_video_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry["text"] for entry in transcript])
    except Exception as e:
        return "Transcript unavailable. Generating summary from video title instead."

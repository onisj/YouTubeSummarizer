from youtube_transcript_api import YouTubeTranscriptApi
from langdetect import detect

def get_video_transcript(video_id: str) -> tuple[str, str]:
    """Extract transcript and detect its language."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        language = detect(text)  # e.g., "es" for Spanish, "fr" for French
        return text, language
    except Exception as e:
        return "Transcript unavailable. Generating summary from video title instead.", "en"
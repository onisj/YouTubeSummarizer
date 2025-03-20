from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def search_youtube_video(query: str) -> dict:
    """Search for a YouTube video URL using SerpApi."""
    params = {
        "engine": "youtube",
        "search_query": f'"{query}" site:youtube.com',
        "api_key": SERPAPI_API_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        video = results.get("video_results", [{}])[0]
        return {
            "title": video.get("title", "Unknown Title"),
            "link": video.get("link", ""),
            "channel": video.get("channel", {}).get("name", "Unknown Channel")
        }
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
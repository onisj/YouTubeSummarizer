import re
import json
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_summary_and_themes(text: str, title: str = None) -> dict:
    """Generate AI summary, sentiment, and key themes using Groq."""
    if "Transcript error" in text and title:
        prompt = f"""
        There is no transcript available for this video. Based on the title '{title}', please provide:
        1. A detailed summary (600-700 words).
        2. Sentiment as a single word or phrase (Positive, Negative, or Neutral).
        3. 3-5 key themes as one-word or one-phrase items in a comma-separated string (e.g., 'Confidence, Growth, Love').
        Return the response with sections: **Detailed Summary:**, **Sentiment:**, **Key Themes:**.
        """
    else:
        prompt = f"""
        Analyze this video transcript and provide:
        1. A detailed summary (600-700 words) including core message, storyline, emotional tone, notable participants/casts and dirctors, and notable figures.
        2. Sentiment as a single word or phrase (Positive, Negative, or Neutral).
        3. 3-5 key themes as one-word or one-phrase items in a comma-separated string (e.g., 'Confidence, Growth, Love').
        Return the response with sections: **Detailed Summary:**, **Sentiment:**, **Key Themes:**.
        Transcript: {text[:10000]}
        """

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096
        )
        content = response.choices[0].message.content.strip()

        # Try extracting JSON first (if present)
        json_start = content.find('```json')
        json_end = content.rfind('```')
        if json_start != -1 and json_end != -1:
            json_content = content[json_start + 7:json_end].strip()
            try:
                data = json.loads(json_content)
                # Use JSON data as fallback, but prefer structured sections
                json_summary = data.get("summary", "")
                json_sentiment = data.get("sentiment", "N/A")
                json_themes = data.get("key_themes", "Unknown")
            except json.JSONDecodeError:
                json_summary, json_sentiment, json_themes = "", "N/A", "Unknown"
        else:
            json_summary, json_sentiment, json_themes = "", "N/A", "Unknown"

        # Extract structured sections
        summary_match = re.search(r"\*\*Detailed Summary:\*\*\s*(.*?)(?=\*\*Sentiment:|\Z)", content, re.DOTALL)
        sentiment_match = re.search(r"\*\*Sentiment:\*\*\s*(\w+(?:\s+\w+)?)", content)
        themes_match = re.search(r"\*\*Key Themes:\*\*\s*(.*)", content)

        summary = summary_match.group(1).strip() if summary_match else json_summary if json_summary else "Summary unavailable."
        sentiment = sentiment_match.group(1).strip() if sentiment_match else json_sentiment
        raw_themes = themes_match.group(1).strip() if themes_match else json_themes

        # Refine key themes to 3-5 one-word/phrase items
        if raw_themes and raw_themes != "Unknown":
            themes_list = [theme.strip() for theme in raw_themes.split(",")]
            # Simplify multi-word phrases to single words where possible
            refined_themes = [theme.split()[0] if len(theme.split()) > 1 else theme for theme in themes_list[:5]]
            key_themes = ", ".join(refined_themes[:5])  # Limit to 5
        else:
            key_themes = "Unknown"

        return {
            "summary": summary,
            "sentiment": sentiment,
            "key_themes": key_themes
        }

    except Exception as e:
        print(f"Error in summary generation: {str(e)}")
        return {
            "summary": "Summary unavailable due to processing error.",
            "sentiment": "N/A",
            "key_themes": "Unknown"
        }
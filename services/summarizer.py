import re
import json
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_summary_and_themes(text: str, title: str = None, language: str = "en") -> dict:
    """Generate AI summary, sentiment, and key themes using Groq, respecting the language."""
    if "Transcript error" in text and title:
        prompt = f"""
        There is no transcript available for this video. Based on the title '{title}', please provide:
        1. A detailed summary (600-700 words) in {language}.
        2. Sentiment as a single word or phrase (Positive, Negative, or Neutral).
        3. 3-5 key themes as one-word or one-phrase items in a comma-separated string.
        Return the response with sections: **Detailed Summary:**, **Sentiment:**, **Key Themes:**.
        """
    else:
        prompt = f"""
        Analyze this video transcript in {language} and provide:
        1. A detailed summary (600-700 words) including core message, storyline, emotional tone, notable participants/casts and directors, and notable figures.
        2. Sentiment as a single word or phrase (Positive, Negative, or Neutral).
        3. 3-5 key themes as one-word or one-phrase items in a comma-separated string.
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

        summary_match = re.search(r"\*\*Detailed Summary:\*\*\s*(.*?)(?=\*\*Sentiment:|\Z)", content, re.DOTALL)
        sentiment_match = re.search(r"\*\*Sentiment:\*\*\s*(\w+(?:\s+\w+)?)", content)
        themes_match = re.search(r"\*\*Key Themes:\*\*\s*(.*)", content)

        summary = summary_match.group(1).strip() if summary_match else "Summary unavailable."
        sentiment = sentiment_match.group(1).strip() if sentiment_match else "N/A"
        raw_themes = themes_match.group(1).strip() if themes_match else "Unknown"

        key_themes = ", ".join([theme.strip() for theme in raw_themes.split(",")][:5]) if raw_themes != "Unknown" else "Unknown"

        return {
            "summary": summary,
            "sentiment": sentiment,
            "key_themes": key_themes,
            "language": language
        }
    except Exception as e:
        print(f"Error in summary generation: {str(e)}")
        return {
            "summary": "Summary unavailable due to processing error.",
            "sentiment": "N/A",
            "key_themes": "Unknown",
            "language": language
        }
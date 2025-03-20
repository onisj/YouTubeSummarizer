from gtts import gTTS
import os

def text_to_speech(text: str, filename: str = "static/summary.mp3") -> str:
    """Convert summary to an audio file."""
    os.makedirs("static", exist_ok=True)
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None
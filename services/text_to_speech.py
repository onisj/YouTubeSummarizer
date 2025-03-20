from gtts import gTTS
import os
import logging

logger = logging.getLogger(__name__)

def text_to_speech(text: str, language: str = "en", filename: str = "summary.mp3") -> str:
    """Convert summary text to an audio file in the detected language."""
    if not text:
        logger.error("No text provided for text-to-speech conversion.")
        return None

    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)
    filepath = os.path.join(static_dir, filename)
    temp_status = f"{filepath}.status"

    with open(temp_status, "w") as status_file:
        status_file.write("processing")

    try:
        logger.info(f"Generating audio for text: {text[:50]}... in language: {language}")
        tts = gTTS(text=text, lang=language, slow=False)  # Fixed: Removed "t READY"
        tts.save(filepath)
        os.remove(temp_status)
        logger.info(f"Audio file generated at: {filepath}")
        return f"/{filepath}"
    except Exception as e:
        logger.error(f"Failed to generate audio: {str(e)}")
        os.remove(temp_status)
        return None
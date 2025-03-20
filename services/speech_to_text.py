import speech_recognition as sr
from typing import Optional

def speech_to_text(audio_file_path: str) -> Optional[str]:
    """Convert an audio file to text using speech recognition."""
    recognizer = sr.Recognizer()
    
    try:
        # Load the audio file
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        
        # Recognize speech using Google Speech Recognition (free tier)
        text = recognizer.recognize_google(audio_data)
        return text
    
    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from speech recognition service; {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None
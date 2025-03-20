from services.summarizer import generate_summary_and_themes

from fastapi import APIRouter, Query, UploadFile, File

from services.video_search import search_youtube_video
from services.transcript import get_video_transcript

from services.text_to_speech import text_to_speech
from services.speech_to_text import speech_to_text  

import speech_recognition as sr
from dotenv import load_dotenv
from pydantic import BaseModel
from pydub import AudioSegment
from groq import Groq
import logging
import json
import re
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
logger = logging.getLogger(__name__)
router = APIRouter()

class VideoSummaryResponse(BaseModel):
    title: str
    channel: str | None
    link: str
    summary: str
    sentiment: str
    key_themes: str
    audio: str | None
    error: str | None = None

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_youtube_video",
            "description": "Search YouTube videos by title or description",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query or video title"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_transcript",
            "description": "Extract transcript from a YouTube video",
            "parameters": {
                "type": "object",
                "properties": {"video_id": {"type": "string", "description": "YouTube video ID"}},
                "required": ["video_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_summary_and_themes",
            "description": "Generate summary and themes from text or title",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Transcript text"},
                    "title": {"type": "string", "description": "Video title for fallback", "default": None}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Convert summary text to audio",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to convert"}},
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "speech_to_text",
            "description": "Convert spoken input to text (currently client-side)",
            "parameters": {
                "type": "object",
                "properties": {},  # No parameters needed client-side
                "required": []
            }
        }
    }
]

def extract_videoid(url: str) -> str:
    """Extract YouTube video ID from URL"""
    regex = r"(?:v=|/)([0-9A-Za-z-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

@router.get("/summarize/", response_model=VideoSummaryResponse)
async def summarize_video(query: str = Query(..., min_length=1, description="Video title or search prompt"), tts: bool = Query(False)):
    """Summarize a YouTube video autonomously using AI-driven function calling."""
    messages = [
        {"role": "system", "content": "You are an autonomous YouTube video summarizer. Use the provided tools to search for a video, extract its transcript, and generate a summary with themes. If the transcript is unavailable, use the video title to generate a summary. Optionally convert the summary to audio if requested. Return the final result as a JSON object with 'title', 'channel', 'link', 'summary', 'sentiment', 'key_themes', and 'audio' (if requested). Do not ask for manual input; proceed with available data."},
        {"role": "user", "content": f"Summarize the YouTube video titled '{query}'.{' Convert the summary to audio.' if tts else ''}"}
    ]

    result = {"title": "N/A", "channel": None, "link": "", "summary": "", "sentiment": "N/A", "key_themes": "", "audio": None, "error": None}
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096
            )
            tool_calls = response.choices[0].message.tool_calls

            if not tool_calls:
                if response.choices[0].message.content:
                    try:
                        final_result = json.loads(response.choices[0].message.content)
                        result.update(final_result)
                        if result["summary"]:
                            return VideoSummaryResponse(**result)
                    except json.JSONDecodeError:
                        messages.append({"role": "assistant", "content": f"Error parsing response: {response.choices[0].message.content}"})
                        continue
                else:
                    messages.append({"role": "assistant", "content": "No tool call or content received."})
                    continue

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if func_name == "search_youtube_video":
                    search_result = search_youtube_video(args.get("query", ""))
                    if "error" in search_result:
                        result["error"] = search_result["error"]
                        return VideoSummaryResponse(**result)
                    result.update(search_result)
                    video_id = extract_videoid(search_result["link"])
                    messages.append({"role": "assistant", "content": f"Found video: {json.dumps(search_result)}"})
                    messages.append({"role": "user", "content": f"Extract transcript for video ID: {video_id}"})

                elif func_name == "get_video_transcript":
                    transcript = get_video_transcript(args["video_id"])
                    messages.append({"role": "assistant", "content": f"Transcript: {transcript[:6000]}"})
                    if "Transcript error" in transcript:
                        messages.append({"role": "user", "content": f"Transcript is unavailable. Generate a summary using only the title: '{result['title']}'"})
                    else:
                        messages.append({"role": "user", "content": f"Generate summary and themes from this video with title '{result['title']}'"})

                elif func_name == "generate_summary_and_themes":
                    summary_data = generate_summary_and_themes(args.get("text", ""), args.get("title", ""))
                    result["summary"] = summary_data["summary"]
                    result["sentiment"] = summary_data["sentiment"]
                    result["key_themes"] = summary_data["key_themes"]
                    if tts:
                        audio_path = text_to_speech(summary_data["summary"])
                        logger.info(f"Assigned audio path: {audio_path}")
                        result["audio"] = audio_path if audio_path else "/static/summary.mp3"
                    if result["summary"]:
                        return VideoSummaryResponse(**result)

                elif func_name == "text_to_speech":
                    audio_path = text_to_speech(result["summary"])
                    logger.info(f"Assigned audio path: {audio_path}")
                    result["audio"] = audio_path if audio_path else "/static/summary.mp3"
                    if result["summary"]:
                        return VideoSummaryResponse(**result)

                elif func_name == "speech_to_text":
                    messages.append({"role": "assistant", "content": "Speech-to-text is processed client-side."})

        except HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 1))  # Default to 1s if header missing
                logger.warning(f"429 Too Many Requests received. Waiting {retry_after} seconds before retry {attempt + 1}/{max_attempts}.")
                if attempt == max_attempts - 1:
                    result["error"] = f"Failed after {max_attempts} attempts: Rate limit exceeded."
                    return VideoSummaryResponse(**result)
                time.sleep(retry_after)
                continue
            else:
                logger.error(f"HTTP error: {str(e)}")
                result["error"] = f"HTTP error: {str(e)}"
                return VideoSummaryResponse(**result)
        except Exception as e:
            logger.warning(f"API error: {str(e)}. Waiting before retry {attempt + 1}/{max_attempts}.")
            if attempt == max_attempts - 1:
                result["error"] = f"Failed after {max_attempts} attempts: {str(e)}"
                return VideoSummaryResponse(**result)
            time.sleep(2 ** attempt)  # Fallback exponential backoff for non-429 errors
            continue

    if not result["summary"]:
        result["error"] = "Failed to generate summary after all attempts."
    return VideoSummaryResponse(**result)

# ✅ Audio Status
@router.get("/audio-status/{file_name}")
async def check_audio_status(file_name: str):
    """Check if the audio file has been fully generated."""
    file_path = f"static/{file_name}"
    if os.path.exists(file_path):
        return {"status": "ready", "audio_url": f"/static/{file_name}"}
    return {"status": "processing"}

# ✅ Audio Download Handling
@router.get("/download-audio/{file_name}")
async def download_audio(file_name: str):
    """Check if the audio file is fully generated before allowing downloads."""
    file_path = f"static/{file_name}"
    temp_status = f"{file_path}.status"

    if os.path.exists(temp_status):
        return {"error": "Audio is still being processed. Please check later."}
    
    if os.path.exists(file_path):
        return {"audio_url": f"/static/{file_name}"}
    
    return {"error": "Audio file not found."}

@router.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    """Convert uploaded speech to text using Google's free Speech Recognition API."""
    try:
        os.makedirs("uploads", exist_ok=True)
        input_audio_path = f"uploads/{file.filename}"
        output_wav_path = f"uploads/{os.path.splitext(file.filename)[0]}.wav"

        # Save uploaded audio
        with open(input_audio_path, "wb") as audio_file:
            audio_file.write(await file.read())

        # Convert to WAV (mono, 16kHz) for better recognition
        try:
            audio = AudioSegment.from_file(input_audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_wav_path, format="wav")
        except Exception as e:
            return {"error": f"Audio conversion failed: {str(e)}"}

        # Recognize speech
        recognizer = sr.Recognizer()
        with sr.AudioFile(output_wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source, duration=60)  # Limit to first 60 seconds

        transcribed_text = recognizer.recognize_google(audio_data)

        return {"text": transcribed_text}

    except sr.UnknownValueError:
        return {"error": "Could not understand the audio."}
    except sr.RequestError:
        return {"error": "Could not request results from Google's API. Try again later."}
    except Exception as e:
        return {"error": f"Failed to process speech: {str(e)}"}
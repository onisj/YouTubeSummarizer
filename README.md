# YouTube Video Summarizer

An API built with FastAPI to summarize YouTube videos without watching them using function calling.

## Features
- Summarize videos from titles or free text prompts.
- Provide video URL, sentiment analysis, and audio output.
- Uses Grok (xAI) for function calling and summarization.

## Setup
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows

## Project Structure
```
YouTubeSummarizer/
├── venv/                   # Virtual environment folder
├── .env                    # SerpApi key
├── requirements.txt        # Dependencies
├── main.py                 # FastAPI app entry
├── api/
│   └── routes.py           # API endpoints with Pydantic
├── services/
│   ├── video_search.py     # Video search logic
│   ├── transcript.py       # Transcript extraction
│   ├── summarizer.py       # Summarization and sentiment
│   └── text_to_speech.py   # Text-to-speech
├── static/
│   └── summary.mp3         # Generated audio
└── README.md               # Project docs
```



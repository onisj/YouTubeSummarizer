import os
import uvicorn
from fastapi import FastAPI, Request
from api.routes import router
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="YouTube Video Summarizer",
    description="An API to summarize YouTube videos without watching them.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.include_router(router)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.middleware("http")
async def restart_after_request(request, call_next):
    """Ensures the server remains active by handling connection resets."""
    response = await call_next(request)
    
    # Restart FastAPI if it becomes unresponsive
    if response.status_code in [500, 503]:  # Restart on Internal Server Error or Service Unavailable
        os.system("kill -9 $(lsof -t -i:8000)")  # Kills existing process on port 8000
        os.system("uvicorn main:app --reload &")  # Restart server in the background
    
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
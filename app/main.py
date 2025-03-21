import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi.errors import RateLimitExceeded
from app.api.endpoints import chat
from app.logger import setup_logging

# ✅ Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

logger.info("Starting TEEP Chat Application...")

app = FastAPI(title="TEEP AFRICA CHATBOT", debug=True)

# CORS settings
origins = ["https://teep.africa"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include chat endpoint router
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

# ✅ Add the Rate Limit Handler at the app level
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

# Root endpoint serves the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_file = os.path.join(static_dir, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

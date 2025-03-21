import logging
import time
from fastapi import APIRouter, HTTPException, Request
from app.models.chat_model import ChatRequest, ChatResponse
from app.services.rag_service import generate_response

from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
logger = logging.getLogger(__name__)

# ✅ Initialize rate limiter (5 requests per minute)
limiter = Limiter(key_func=get_remote_address)


@router.post("/", response_model=ChatResponse)
@limiter.limit("5/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Handles user chat requests and generates a response using the RAG-based chatbot.

    Args:
        request (Request): The HTTP request object from FastAPI.
        chat_request (ChatRequest): The user input query.

    Returns:
        ChatResponse: The chatbot's response.
    """
    start_time = time.time()

    try:
        logger.info(f"Received query: {chat_request.query}")

        # ✅ Generate structured response (includes source and confidence)
        result = await generate_response(chat_request.query)

        latency = time.time() - start_time
        logger.info(f"Response: {result['answer']} (Latency: {latency:.2f}s)")

        return ChatResponse(
            answer=result["answer"],
            source=result.get("source", None),
            confidence=result.get("confidence", None)
        )

    except Exception as e:
        logger.error("Error in chat endpoint", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later."
        )

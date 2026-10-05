from fastapi import APIRouter
from routers.documents import router as documents_router
from routers.chat import router as chat_router

api_router = APIRouter()
api_router.include_router(documents_router)
api_router.include_router(chat_router)

__all__ = ["api_router", "documents_router", "chat_router"]

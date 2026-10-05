from typing import List, Optional, Any
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    chat_id: str = Field(..., description="Active chat session ID")
    query: str = Field(..., description="User query / question")
    doc_id: Optional[str] = Field(None, description="Optional document ID if attached")
    doc_name: Optional[str] = Field(None, description="Optional document name if known")
    deep_analysis: Optional[bool] = Field(False, description="Enable deep analysis mode")

class UploadResponse(BaseModel):
    message: str = "Success"
    doc_id: str
    filename: Optional[str] = None
    status: str = "ready"
    page_count: Optional[int] = 0

class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    page_count: int = 0
    error: Optional[str] = None

class NewChatResponse(BaseModel):
    chat_id: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    history: List[ChatMessage]


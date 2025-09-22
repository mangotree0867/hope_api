from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class ChatMessageResponse(BaseModel):
    id: int
    role: str
    message_text: Optional[str] = None
    media_url: Optional[str] = None
    content_type: Optional[str] = None
    created_at: datetime
    message_order: int

class ChatSessionResponse(BaseModel):
    id: int
    user_id: int
    session_title: str
    location: str
    created_at: datetime
    message_count: Optional[int] = None

class ChatSessionsListResponse(BaseModel):
    sessions: List[ChatSessionResponse]
    total: int

class SessionMessagesResponse(BaseModel):
    session: ChatSessionResponse
    messages: List[ChatMessageResponse]
    total: int
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime

Base = declarative_base()

# --- Database Models ---
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    session_title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'

    # For user messages (video input)
    media_url = Column(Text, nullable=True)
    content_type = Column(String(50), nullable=True)

    # For assistant messages (text response)
    message_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    message_order = Column(Integer, nullable=False)

# Keep old ChatRecord for backward compatibility (can be removed later)
class ChatRecord(Base):
    __tablename__ = "chat_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    predicted_word = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    generated_sentence = Column(Text, nullable=True)
    input_type = Column(String, nullable=False)  # "video" or "image_sequence"
    frame_count = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class VideoRecord(Base):
    __tablename__ = "video_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)  # Path to stored file
    file_extension = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    is_processed = Column(Boolean, default=False)
    chat_record_id = Column(Integer, nullable=True)  # Link to prediction result
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# --- Pydantic Models for API ---
class ChatSessionCreate(BaseModel):
    session_title: Optional[str] = None

class ChatSessionResponse(BaseModel):
    id: int
    user_id: int
    session_title: Optional[str]
    created_at: datetime
    message_count: Optional[int] = 0

    class Config:
        from_attributes = True

class ChatMessageCreate(BaseModel):
    session_id: int
    role: str
    media_url: Optional[str] = None
    content_type: Optional[str] = None
    message_text: Optional[str] = None

class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    user_id: int
    role: str
    media_url: Optional[str]
    content_type: Optional[str]
    message_text: Optional[str]
    created_at: datetime
    message_order: int

    class Config:
        from_attributes = True

# --- Chat Service Functions ---
class ChatService:
    @staticmethod
    def create_session(db: Session, user_id: int, title: Optional[str] = None) -> ChatSession:
        """Create a new chat session for a user"""
        session = ChatSession(
            user_id=user_id,
            session_title=title
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get_or_create_session(db: Session, user_id: int, session_id: Optional[int] = None) -> ChatSession:
        """Get existing session or create a new one"""
        if session_id:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()
            if session:
                return session

        # Create new session if not found or no ID provided
        return ChatService.create_session(db, user_id, "New Session")

    @staticmethod
    def add_user_message(
        db: Session,
        session_id: int,
        user_id: int,
        media_url: str,
        content_type: str = "video/mp4"
    ) -> ChatMessage:
        """Add a user video message to the chat"""
        # Get the next message order
        max_order = db.query(func.max(ChatMessage.message_order)).filter(
            ChatMessage.session_id == session_id
        ).scalar() or 0

        message = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="user",
            media_url=media_url,
            content_type=content_type,
            message_order=max_order + 1
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def add_assistant_message(
        db: Session,
        session_id: int,
        user_id: int,
        message_text: str
    ) -> ChatMessage:
        """Add an assistant text response to the chat"""
        # Get the next message order
        max_order = db.query(func.max(ChatMessage.message_order)).filter(
            ChatMessage.session_id == session_id
        ).scalar() or 0

        message = ChatMessage(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            message_text=message_text,
            message_order=max_order + 1
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_session_messages(
        db: Session,
        session_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[ChatMessage]:
        """Get messages for a session ordered by message_order"""
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.message_order.asc()).offset(offset).limit(limit).all()

    @staticmethod
    def get_user_sessions(
        db: Session,
        user_id: int,
        limit: int = 50,
        offset: int = 0
    ) -> List[ChatSession]:
        """Get all sessions for a user ordered by last activity"""
        return db.query(ChatSession).filter(
            ChatSession.user_id == user_id
        ).order_by(ChatSession.last_activity.desc()).offset(offset).limit(limit).all()
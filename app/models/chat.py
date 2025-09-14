from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime

Base = declarative_base()

# --- 데이터베이스 모델 ---
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
    role = Column(String(20), nullable=False)  # 'user' 또는 'assistant'

    # 사용자 메시지 (비디오 입력)
    media_url = Column(Text, nullable=True)
    content_type = Column(String(50), nullable=True)

    # 어시스턴트 메시지 (텍스트 응답)
    message_text = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    message_order = Column(Integer, nullable=False)

# 하위 호환성을 위해 이전 ChatRecord 유지 (나중에 제거 가능)
class ChatRecord(Base):
    __tablename__ = "chat_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    session_id = Column(String, index=True, nullable=True)
    predicted_word = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    generated_sentence = Column(Text, nullable=True)
    input_type = Column(String, nullable=False)  # "video" 또는 "image_sequence"
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
    file_path = Column(String, nullable=False)  # 저장된 파일 경로
    file_extension = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    is_processed = Column(Boolean, default=False)
    chat_record_id = Column(Integer, nullable=True)  # 예측 결과에 대한 링크
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# --- API용 Pydantic 모델 ---
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

# --- 채팅 서비스 기능 ---
class ChatService:
    @staticmethod
    def create_session(db: Session, user_id: int, title: Optional[str] = None) -> ChatSession:
        """사용자를 위한 새로운 채팅 세션 생성"""
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
        """기존 세션을 가져오거나 새로운 세션 생성"""
        if session_id:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id
            ).first()
            if session:
                return session

        # 세션을 찾을 수 없거나 ID가 제공되지 않은 경우 새 세션 생성
        return ChatService.create_session(db, user_id, "New Session")

    @staticmethod
    def add_user_message(
        db: Session,
        session_id: int,
        user_id: int,
        media_url: str,
        content_type: str = "video/mp4"
    ) -> ChatMessage:
        """채팅에 사용자 비디오 메시지 추가"""
        # 다음 메시지 순서 가져오기
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
        """채팅에 어시스턴트 텍스트 응답 추가"""
        # 다음 메시지 순서 가져오기
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
        """메시지 순서별로 세션의 메시지 가져오기"""
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
        """마지막 활동 순서별로 사용자의 모든 세션 가져오기"""
        return db.query(ChatSession).filter(
            ChatSession.user_id == user_id
        ).order_by(ChatSession.last_activity.desc()).offset(offset).limit(limit).all()
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.core.database import get_db
from app.api.routes.auth import get_authenticated_user
from app.models.auth import User
from app.models.chat import ChatSession, ChatMessage

router = APIRouter(tags=["Chat"])

@router.get("/chat-sessions")
async def get_chat_sessions(
    current_user: User = Depends(get_authenticated_user),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get chat sessions for the authenticated user with message count"""
    query = db.query(
        ChatSession,
        func.count(ChatMessage.id).label('message_count')
    ).outerjoin(
        ChatMessage, ChatSession.id == ChatMessage.session_id
    ).filter(
        ChatSession.user_id == current_user.id
    ).group_by(ChatSession.id)

    sessions = query.order_by(ChatSession.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "sessions": [
            {
                "id": session.ChatSession.id,
                "user_id": session.ChatSession.user_id,
                "session_title": session.ChatSession.session_title,
                "created_at": session.ChatSession.created_at,
                "message_count": session.message_count
            } for session in sessions
        ],
        "total": db.query(ChatSession).filter(
            ChatSession.user_id == current_user.id
        ).count()
    }

@router.get("/chat-sessions/{session_id}/messages")
async def get_session_messages(
    session_id: int,
    current_user: User = Depends(get_authenticated_user),
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get all messages for a specific chat session (only if user owns the session)"""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or access denied")

    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.message_order.desc()).offset(offset).limit(limit).all()

    return {
        "session": {
            "id": session.id,
            "user_id": session.user_id,
            "session_title": session.session_title,
            "created_at": session.created_at
        },
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "message_text": msg.message_text,
                "media_url": msg.media_url,
                "content_type": msg.content_type,
                "created_at": msg.created_at,
                "message_order": msg.message_order
            } for msg in messages
        ],
        "total": db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).count()
    }
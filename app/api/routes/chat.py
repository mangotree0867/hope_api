from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.core.database import get_db
from app.api.routes.auth import get_authenticated_user
from app.models.auth import User
from app.models.chat import ChatSession, ChatMessage
from app.schemas.chat import ChatSessionsListResponse, SessionMessagesResponse, UpdateSessionTitleRequest
from app.schemas.auth import ErrorResponse

router = APIRouter(tags=["Chat"])

@router.get("/chat-sessions",
    response_model=ChatSessionsListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid authentication credentials"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_chat_sessions(
    current_user: User = Depends(get_authenticated_user),
    limit: int = Query(50, description="Number of sessions to retrieve (max 100)", ge=1, le=100),
    offset: int = Query(0, description="Number of sessions to skip", ge=0),
    db: Session = Depends(get_db)
):
    """
    인증된 사용자의 채팅 세션을 메시지 수와 함께 가져오기

    - **limit**: 가져올 세션 수 (1-100, 기본값 50)
    - **offset**: 건너뛸 세션 수 (기본값 0)
    """
    query = db.query(
        ChatSession,
        func.count(ChatMessage.id).label('message_count')
    ).outerjoin(
        ChatMessage, ChatSession.id == ChatMessage.session_id
    ).filter(
        ChatSession.user_id == current_user.id
    ).group_by(ChatSession.id)

    sessions = query.order_by(ChatSession.created_at.desc()).offset(offset).limit(limit).all()

    return ChatSessionsListResponse(
        sessions=[
            {
                "id": session.ChatSession.id,
                "user_id": session.ChatSession.user_id,
                "session_title": session.ChatSession.session_title,
                "location": session.ChatSession.location,
                "created_at": session.ChatSession.created_at,
                "message_count": session.message_count
            } for session in sessions
        ],
        total=db.query(ChatSession).filter(
            ChatSession.user_id == current_user.id
        ).count()
    )

@router.get("/chat-sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid authentication credentials"},
        404: {"model": ErrorResponse, "description": "Session not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_session_messages(
    session_id: int,
    current_user: User = Depends(get_authenticated_user),
    limit: int = Query(100, description="Number of messages to retrieve (max 200)", ge=1, le=200),
    offset: int = Query(0, description="Number of messages to skip", ge=0),
    db: Session = Depends(get_db)
):
    """
    특정 채팅 세션의 모든 메시지 가져오기 (사용자가 세션을 소유한 경우에만)

    - **session_id**: 채팅 세션 ID
    - **limit**: 가져올 메시지 수 (1-200, 기본값 100)
    - **offset**: 건너뛸 메시지 수 (기본값 0)
    """
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or access denied")

    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.message_order.desc()).offset(offset).limit(limit).all()

    return SessionMessagesResponse(
        session={
            "id": session.id,
            "user_id": session.user_id,
            "session_title": session.session_title,
            "location": session.location,
            "created_at": session.created_at
        },
        messages=[
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
        total=db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).count()
    )

@router.put("/chat-sessions/{session_id}/title",
    responses={
        200: {"description": "Session title updated successfully"},
        401: {"model": ErrorResponse, "description": "Invalid authentication credentials"},
        404: {"model": ErrorResponse, "description": "Session not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def update_session_title(
    session_id: int,
    request: UpdateSessionTitleRequest,
    current_user: User = Depends(get_authenticated_user),
    db: Session = Depends(get_db)
):
    """
    사용자의 채팅 세션 제목을 업데이트합니다.

    - **session_id**: 업데이트할 세션 ID
    - **session_title**: 새로운 세션 제목
    """
    # 세션이 존재하고 현재 사용자에게 속하는지 확인
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found or access denied")

    # 세션 제목 업데이트
    session.session_title = request.session_title
    db.commit()

    return {"session_title": session.session_title}
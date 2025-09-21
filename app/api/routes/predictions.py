import os
import tempfile
from typing import Optional
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query, Form
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.auth import User
from app.services.auth import get_optional_user
from app.models.chat import ChatSession, ChatMessage
from app.schemas.prediction import  PredictionResponse
from app.schemas.auth import ErrorResponse
from app.services.ml_service import predict_from_video
from app.services.s3_service import get_s3_service

# Category enum matching the ML service
class CategoryEnum(str, Enum):
    TRAUMA = "외상"
    INTERNAL_INJURY = "내상"
    FIRE_SITUATION = "화재상황"
    URBAN_SITUATION = "도심상황"

router = APIRouter(tags=["Predictions"])

# Special user ID for anonymous/unauthenticated users
ANONYMOUS_USER_ID = 1

@router.post("/predict-video",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or video processing error"},
        404: {"model": ErrorResponse, "description": "Session not found or access denied"},
        500: {"model": ErrorResponse, "description": "Video processing or storage error"}
    }
)
async def predict_video(
    file: UploadFile = File(..., description="Video file to process (.mp4, .avi, .mov, .webm, .mkv)"),
    category: CategoryEnum = Form(..., description="Emergency category (TRAUMA=외상/Trauma, INTERNAL_INJURY=내상/Internal injury, FIRE_SITUATION=화재상황/Fire, URBAN_SITUATION=도심상황/Urban)"),
    context: str = Form("", description="Additional context words to help generate the emergency message (e.g., '골절 다리' for fracture leg)"),
    current_user: Optional[User] = Depends(get_optional_user),
    session_id: Optional[int] = Query(None, description="Existing chat session ID (optional, creates new session if not provided)"),
    db: Session = Depends(get_db)
):
    """
    비디오 파일을 처리하여 수화 예측을 수행하고 채팅 세션에 결과를 저장합니다.

    **인증:**
    - 선택사항: 토큰이 제공되면 사용자와 연결, 없으면 임시 세션 생성
    - 임시 세션은 로그인하지 않은 사용자도 사용 가능

    **매개변수:**
    - **file**: 처리할 비디오 파일 (.mp4, .avi, .mov, .webm, .mkv)
    - **category**: 응급 상황 카테고리
      - TRAUMA (외상): 외부 부상 상황
      - INTERNAL_INJURY (내상): 내부 부상 상황
      - FIRE_SITUATION (화재상황): 화재 관련 응급 상황
      - URBAN_SITUATION (도심상황): 도심 지역 응급 상황
    - **context**: 추가 컨텍스트 단어 (선택사항, 예: "골절 다리")
    - **session_id**: 기존 채팅 세션 ID (선택사항, 없으면 새 세션 생성)

    **처리 과정:**
    1. 비디오 파일 업로드 및 S3 저장
    2. 프레임 추출 및 특징 분석
    3. ML 모델을 통한 수화 예측
    4. 카테고리와 컨텍스트를 활용한 긴급 메시지 생성
    5. 채팅 세션에 사용자 메시지(비디오) 및 어시스턴트 응답 추가

    **지원 형식:** .mp4, .avi, .mov, .webm, .mkv
    **최소 요구사항:** 10프레임 이상
    """
    # 세션 가져오기 또는 생성
    effective_user_id = current_user.id if current_user else ANONYMOUS_USER_ID

    if session_id:
        # 세션 조회 - 사용자 ID와 매칭
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == effective_user_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
    else:
        # 새로운 세션 생성
        from datetime import datetime
        session = ChatSession(
            user_id=effective_user_id,
            session_title=f"{datetime.now().strftime('%Y-%m-%d %H:%M')} 대화"
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # 파일 형식 검증
    allowed_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else '.mp4'
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    # 비디오 파일 읽기
    video_bytes = await file.read()

    # S3에 비디오 업로드
    s3_service = get_s3_service()
    s3_url = s3_service.upload_video(
        file_data=video_bytes,
        user_id=effective_user_id,
        filename=file.filename or f"video{file_ext}"
    )

    if not s3_url:
        raise HTTPException(
            status_code=500,
            detail="Failed to upload video to storage"
        )

    user_message_id = None
    assistant_message_id = None

    try:
        # 채팅에 사용자 메시지 추가 (비디오 업로드 - S3 URL 저장)
        user_message = ChatMessage(
            session_id=session.id,
            user_id=effective_user_id,
            role='user',
            media_url=s3_url,  # S3 URL 저장
            content_type=f"video/{file_ext[1:]}"
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        user_message_id = user_message.id

        # Save video temporarily for prediction
        import tempfile
        from app.services.ml_service import predict_from_video

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Call predict_from_video with user-provided category and context
            detected_words = predict_from_video(
                video_path=tmp_file_path,
                category=category.value,
                context=context
            )
        finally:
            # Clean up temp file
            import os as temp_os
            if temp_os.path.exists(tmp_file_path):
                temp_os.unlink(tmp_file_path)

        # 결과를 메시지로 포맷
        if detected_words:
            message = detected_words["generated_sentence"]
        else:
            message = "수화를 감지하지 못했습니다. 다시 시도해주세요."

        # 채팅에 어시스턴트 메시지 추가
        assistant_message = ChatMessage(
            session_id=session.id,
            user_id=effective_user_id,
            role='assistant',
            message_text=message
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        assistant_message_id = assistant_message.id

        return PredictionResponse(
            words=detected_words["recognized_words"],
            sentence=message,
            session_id=session.id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id
        )

    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

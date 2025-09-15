import os
import cv2
import base64
import numpy as np
import tempfile
import torch
import torch.nn.functional as F
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.api.routes.auth import get_authenticated_user
from app.models.auth import User
from app.models.chat import ChatSession, ChatMessage
from app.schemas.prediction import PredictionRequest, PredictionResponse, VideoPredictionResponse
from app.schemas.auth import ErrorResponse
from app.services.ml_service import (
    extract_all_features, log_debug, generate_emergency_sentence,
    model, device, label_encoder, word_mapping
)
from app.services.s3_service import get_s3_service

router = APIRouter(tags=["Predictions"])

@router.post("/predict-video",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or video processing error"},
        401: {"model": ErrorResponse, "description": "Invalid authentication credentials"},
        404: {"model": ErrorResponse, "description": "Session not found or access denied"},
        500: {"model": ErrorResponse, "description": "Video processing or storage error"}
    }
)
async def predict_video(
    file: UploadFile = File(..., description="Video file to process (.mp4, .avi, .mov, .webm, .mkv)"),
    current_user: User = Depends(get_authenticated_user),
    session_id: Optional[int] = Query(None, description="Existing chat session ID (optional, creates new session if not provided)"),
    db: Session = Depends(get_db)
):
    """
    비디오 파일을 처리하여 수화 예측을 수행하고 채팅 세션에 결과를 저장합니다.

    **처리 과정:**
    1. 비디오 파일 업로드 및 S3 저장
    2. 프레임 추출 및 특징 분석
    3. ML 모델을 통한 수화 예측
    4. 채팅 세션에 사용자 메시지(비디오) 및 어시스턴트 응답 추가

    **지원 형식:** .mp4, .avi, .mov, .webm, .mkv
    **최소 요구사항:** 10프레임 이상
    """
    # 세션 가져오기 또는 생성
    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
    else:
        # 새로운 세션 생성
        from datetime import datetime
        session = ChatSession(
            user_id=current_user.id,
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
        user_id=current_user.id,
        filename=file.filename or f"video{file_ext}"
    )

    if not s3_url:
        raise HTTPException(
            status_code=500,
            detail="Failed to upload video to storage"
        )

    # 임시 파일로 저장 (OpenCV 처리용)
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name

    try:
        # 비디오 파일 정보는 S3 URL로 저장됨

        # 비디오에서 모든 프레임 추출
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        # 비디오 속성 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else None

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # 프레임 정보 로깅
        log_debug(f"Video duration: {duration}, frame_count: {len(frames)}")

        log_debug(f"Extracted {len(frames)} frames from video")

        # 모든 프레임에서 특징 추출
        sequence_buffer = []
        prev_pose, prev_left, prev_right = None, None, None

        for frame in frames:
            all_features, curr_pose, curr_left, curr_right, _, _ = extract_all_features(
                frame, prev_pose, prev_left, prev_right
            )
            sequence_buffer.append(all_features)
            prev_pose, prev_left, prev_right = curr_pose, curr_left, curr_right

        log_debug(f"Extracted features from {len(sequence_buffer)} frames")

        frame_count = len(sequence_buffer)
        if frame_count < 10:
            return PredictionResponse(
                words=[],
                sentence="",
                message=f"Sequence too short ({frame_count} frames). Minimum 10 frames required.",
                session_id=session.id
            )

        # 시퀀스 처리
        sequence_list = list(sequence_buffer)
        sequence_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sequence_list])
        sequence_tensor = F.pad(sequence_tensor, (0, 0, 0, settings.MAX_SEQ_LENGTH - sequence_tensor.size(0)), 'constant', 0)
        sequence_tensor = sequence_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(sequence_tensor)

        probs = F.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(1).item()
        confidence = probs[0][predicted_idx].item()

        words_list = []
        sentence_text = ""
        message = ""
        user_message_id = None
        assistant_message_id = None

        # 채팅에 사용자 메시지 추가 (비디오 업로드 - S3 URL 저장)
        user_message = ChatMessage(
            session_id=session.id,
            user_id=current_user.id,
            role='user',
            media_url=s3_url,  # S3 URL 저장
            content_type=f"video/{file_ext[1:]}"
        )
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        user_message_id = user_message.id

        if confidence > 0.5:
            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
            predicted_word = word_mapping.get(predicted_label, predicted_label)
            words_list.append({"word": predicted_word, "confidence": confidence})

            # 응답 메시지 생성
            if confidence > 0.8:
                message = f'"{predicted_word}"라는 수화를 인식했습니다. (신뢰도: {confidence:.2f})'
            else:
                message = f'"{predicted_word}"로 추정됩니다. (신뢰도: {confidence:.2f})'

            # 현재는 간단한 문장
            sentence_text = f"{predicted_word}라고 말씀하시는 것 같습니다."

            # 채팅에 어시스턴트 메시지 추가
            assistant_message = ChatMessage(
                session_id=session.id,
                user_id=current_user.id,
                role='assistant',
                message_text=message
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            assistant_message_id = assistant_message.id
        else:
            message = "수화를 명확히 인식하지 못했습니다. 다시 시도해 주세요."

            # 예측 실패 시에도 어시스턴트 메시지 추가
            assistant_message = ChatMessage(
                session_id=session.id,
                user_id=current_user.id,
                role='assistant',
                message_text=message
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            assistant_message_id = assistant_message.id

        return PredictionResponse(
            words=words_list,
            sentence=sentence_text,
            message=message,
            session_id=session.id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id
        )

    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/predict_sequence",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Empty image sequence or invalid image data"},
        401: {"model": ErrorResponse, "description": "Invalid authentication credentials"},
        500: {"model": ErrorResponse, "description": "Prediction processing error"}
    }
)
async def predict_sequence(
    request: PredictionRequest,
    current_user: User = Depends(get_authenticated_user),
    session_id: Optional[int] = Query(None, description="Existing chat session ID (optional)"),
    db: Session = Depends(get_db)
):
    """
    Base64 인코딩된 이미지 시퀀스에서 수화 단어를 예측하고 응급 상황용 문장을 생성합니다.

    **처리 과정:**
    1. Base64 이미지 디코딩 및 검증
    2. 각 프레임에서 특징 추출
    3. ML 모델을 통한 수화 단어 예측
    4. 예측 결과를 바탕으로 상황별 문장 생성

    **요구사항:**
    - **image_sequence**: Base64 인코딩된 이미지 목록 (10개 이상 권장)
    - **session_id**: 채팅 세션 ID (선택사항)
    """
    if not request.image_sequence:
        raise HTTPException(status_code=400, detail="Image sequence cannot be empty.")

    sequence_buffer = []
    prev_pose, prev_left, prev_right = None, None, None

    for b64_image in request.image_sequence:
        try:
            image_data = base64.b64decode(b64_image)
            image_np = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Could not decode image from Base64 string.")

            all_features, curr_pose, curr_left, curr_right, _, _ = extract_all_features(frame, prev_pose, prev_left, prev_right)
            sequence_buffer.append(all_features)
            prev_pose, prev_left, prev_right = curr_pose, curr_left, curr_right

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    frame_count = len(sequence_buffer)
    if frame_count < 10:
        return PredictionResponse(
            words=[],
            sentence="",
            message=f"Sequence too short ({frame_count} frames). Minimum 10 frames required.",
            session_id=session_id if session_id else 0
        )

    sequence_list = list(sequence_buffer)
    sequence_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sequence_list])
    sequence_tensor = F.pad(sequence_tensor, (0, 0, 0, settings.MAX_SEQ_LENGTH - sequence_tensor.size(0)), 'constant', 0)
    sequence_tensor = sequence_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(sequence_tensor)

    probs = F.softmax(outputs, dim=1)
    predicted_idx = outputs.argmax(1).item()
    confidence = probs[0][predicted_idx].item()

    words_list = []
    sentence_text = ""
    message = ""

    if confidence > 0.5:
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        predicted_word = word_mapping.get(predicted_label, predicted_label)
        words_list.append({"word": predicted_word, "confidence": confidence})

        # 문장 생성 로직 (단어가 하나라도 있으면 시도)
        if len(words_list) >= 1:
            try:
                sentence_text = generate_emergency_sentence([w['word'] for w in words_list])
                message = f'예측된 단어: "{predicted_word}" (신뢰도: {confidence:.2f})'
            except Exception as e:
                sentence_text = predicted_word
                message = f'예측된 단어: "{predicted_word}" (신뢰도: {confidence:.2f}), 문장 생성 실패'
        else:
            message = "예측된 단어가 충분하지 않아 문장을 생성할 수 없습니다."

        # 예측 결과를 어시스턴트 메시지로 저장
        if session_id:
            session = db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.user_id == current_user.id
            ).first()
            if session:
                result_message = ChatMessage(
                    session_id=session.id,
                    user_id=current_user.id,
                    role='assistant',
                    message_text=f"예측 단어: {predicted_word} (신뢰도: {confidence:.2f})\n생성 문장: {sentence_text}"
                )
                db.add(result_message)
                db.commit()
    else:
        message = "Prediction failed with low confidence."

    return PredictionResponse(
        words=words_list,
        sentence=sentence_text,
        message=message,
        session_id=session_id if session_id else 0
    )
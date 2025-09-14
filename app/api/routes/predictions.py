import os
import cv2
import base64
import numpy as np
import tempfile
import torch
import torch.nn.functional as F
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.api.routes.auth import get_authenticated_user
from app.models.auth import User
from app.models.chat import ChatSession, ChatService, ChatRecord, VideoRecord
from app.schemas.prediction import PredictionRequest, PredictionResponse, VideoPredictionResponse
from app.services.ml_service import (
    extract_all_features, log_debug, generate_emergency_sentence,
    model, device, label_encoder, word_mapping
)

router = APIRouter(tags=["Predictions"])

@router.post("/predict-video", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(..., description="Video file to process"),
    current_user: User = Depends(get_authenticated_user),
    session_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Process video file and manage chat session.
    Creates new session if none provided, adds user message and assistant response.
    """
    # Get or create session
    if session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        ).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
    else:
        # Create new session
        session = ChatSession(
            user_id=current_user.id,
            session_title="New Sign Language Session"
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
    file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else '.mp4'
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Read video file
    video_bytes = await file.read()

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_path = tmp_file.name

    try:
        # Save video record to database
        video_record = VideoRecord(
            user_id=str(current_user.id),
            session_id=str(session_id) if session_id else None,
            filename=file.filename or "unknown.mp4",
            file_size=len(video_bytes),
            file_path=tmp_path,
            file_extension=file_ext,
            is_processed=False
        )
        db.add(video_record)
        db.commit()
        db.refresh(video_record)

        # Extract all frames from video
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        # Get video properties
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

        # Update video record with frame info
        video_record.duration = duration
        video_record.frame_count = len(frames)
        db.commit()

        log_debug(f"Extracted {len(frames)} frames from video")

        # Extract features from all frames
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

        # Process sequence
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

        # Add user message to chat (video uploaded)
        user_message = ChatService.add_user_message(
            db=db,
            session_id=session.id,
            user_id=current_user.id,
            media_url=tmp_path,
            content_type=f"video/{file_ext[1:]}"
        )
        user_message_id = user_message.id

        if confidence > 0.5:
            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
            predicted_word = word_mapping.get(predicted_label, predicted_label)
            words_list.append({"word": predicted_word, "confidence": confidence})

            # Generate response message
            if confidence > 0.8:
                message = f'"{predicted_word}"라는 수화를 인식했습니다. (신뢰도: {confidence:.2f})'
            else:
                message = f'"{predicted_word}"로 추정됩니다. (신뢰도: {confidence:.2f})'

            # For now, simple sentence
            sentence_text = f"{predicted_word}라고 말씀하시는 것 같습니다."

            # Add assistant message to chat
            assistant_message = ChatService.add_assistant_message(
                db=db,
                session_id=session.id,
                user_id=current_user.id,
                message_text=message
            )
            assistant_message_id = assistant_message.id

            # Update video record
            video_record.is_processed = True
            db.commit()
        else:
            message = "수화를 명확히 인식하지 못했습니다. 다시 시도해 주세요."

            # Add assistant message even for failed predictions
            assistant_message = ChatService.add_assistant_message(
                db=db,
                session_id=session.id,
                user_id=current_user.id,
                message_text=message
            )
            assistant_message_id = assistant_message.id

            # Still mark video as processed
            video_record.is_processed = True
            db.commit()

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
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/predict_sequence", response_model=PredictionResponse)
async def predict_sequence(
    request: PredictionRequest,
    current_user: User = Depends(get_authenticated_user),
    session_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Predicts a sign language word from a sequence of images and generates an emergency sentence.
    Input should be a list of Base64 encoded images representing a sequence.
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

        # Generate sentence if enough words
        if len(words_list) >= 2:
            sentence_text = generate_emergency_sentence([w['word'] for w in words_list])
        else:
            message = "예측된 단어가 충분하지 않아 문장을 생성할 수 없습니다."

        # Save chat record to database
        chat_record = ChatRecord(
            user_id=str(current_user.id),
            session_id=str(session_id) if session_id else None,
            predicted_word=predicted_word,
            confidence=confidence,
            generated_sentence=sentence_text,
            input_type="image_sequence",
            frame_count=frame_count
        )
        db.add(chat_record)
        db.commit()
        db.refresh(chat_record)
    else:
        message = "Prediction failed with low confidence."

    return PredictionResponse(
        words=words_list,
        sentence=sentence_text,
        message=message,
        session_id=session_id if session_id else 0
    )
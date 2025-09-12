import os
import warnings

# Suppress warnings early before importing protobuf-dependent modules
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import asyncio
import json
import base64
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Settings configuration
class Settings:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/hope_api_db")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

settings = Settings()

# Gemini API 설정
genai.configure(api_key=settings.google_api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Database setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Hyperparameters and Settings (Matching training code) ---
MAX_SEQ_LENGTH = 50
NUM_FEATURES = 335

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Helper Functions (Your original code) ---
def log_debug(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    print(log_message.strip())
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(log_message)

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    angle = np.arccos(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
    return angle

def calculate_kinematic_features(pose_seq, left_hand_seq, right_hand_seq, prev_pose, prev_left, prev_right):
    frame_features = []
    
    pose_disp = pose_seq - prev_pose if prev_pose is not None else np.zeros_like(pose_seq)
    left_hand_disp = left_hand_seq - prev_left if prev_left is not None else np.zeros_like(left_hand_seq)
    right_hand_disp = right_hand_seq - prev_right if prev_right is not None else np.zeros_like(right_hand_seq)
    frame_features.extend(pose_disp.flatten())
    frame_features.extend(left_hand_disp.flatten())
    frame_features.extend(right_hand_disp.flatten())
    
    if np.sum(pose_seq[2]) != 0 and np.sum(pose_seq[4]) != 0 and np.sum(pose_seq[6]) != 0:
        angle_right_elbow = calculate_angle(pose_seq[2], pose_seq[4], pose_seq[6])
        frame_features.append(angle_right_elbow)
    else:
        frame_features.append(0.0)

    if np.sum(pose_seq[1]) != 0 and np.sum(pose_seq[3]) != 0 and np.sum(pose_seq[5]) != 0:
        angle_left_elbow = calculate_angle(pose_seq[1], pose_seq[3], pose_seq[5])
        frame_features.append(angle_left_elbow)
    else:
        frame_features.append(0.0)

    def _calculate_finger_angles(hand_lms):
        if np.sum(hand_lms) == 0: return [0.0] * 15
        angles = []
        finger_lms_indices = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),
            (0, 5, 6), (5, 6, 7), (6, 7, 8),
            (0, 9, 10), (9, 10, 11), (10, 11, 12),
            (0, 13, 14), (13, 14, 15), (14, 15, 16),
            (0, 17, 18), (17, 18, 19), (18, 19, 20)
        ]
        for p1_idx, p2_idx, p3_idx in finger_lms_indices:
            angles.append(calculate_angle(hand_lms[p1_idx], hand_lms[p2_idx], hand_lms[p3_idx]))
        return angles
    
    frame_features.extend(_calculate_finger_angles(left_hand_seq))
    frame_features.extend(_calculate_finger_angles(right_hand_seq))

    def _calculate_palm_vector(hand_lms):
        if np.sum(hand_lms) == 0: return [0.0, 0.0, 0.0]
        wrist = hand_lms[0]
        palm_middle = (hand_lms[9] + hand_lms[13]) / 2
        vector = palm_middle - wrist
        norm = np.linalg.norm(vector)
        return list(vector / norm) if norm != 0 else [0.0, 0.0, 0.0]

    frame_features.extend(_calculate_palm_vector(left_hand_seq))
    frame_features.extend(_calculate_palm_vector(right_hand_seq))

    if np.sum(pose_seq[0]) != 0 and np.sum(left_hand_seq[0]) != 0:
        dist_left_hand_to_nose = np.linalg.norm(left_hand_seq[0] - pose_seq[0])
        frame_features.append(dist_left_hand_to_nose)
    else:
        frame_features.append(0.0)
    
    if np.sum(pose_seq[0]) != 0 and np.sum(right_hand_seq[0]) != 0:
        dist_right_hand_to_nose = np.linalg.norm(right_hand_seq[0] - pose_seq[0])
        frame_features.append(dist_right_hand_to_nose)
    else:
        frame_features.append(0.0)
    
    if np.sum(pose_seq[5]) != 0 and np.sum(pose_seq[6]) != 0:
        dist_wrists = np.linalg.norm(pose_seq[5] - pose_seq[6])
        frame_features.append(dist_wrists)
    else:
        frame_features.append(0.0)
    
    return np.array(frame_features, dtype=np.float32)

def extract_all_features(image, prev_pose, prev_left, prev_right):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)

    pose_landmarks = np.zeros((7, 3))
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        selected_indices = [0, 11, 12, 13, 14, 15, 16]
        for i, idx in enumerate(selected_indices):
            pose_landmarks[i] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]

    left_hand_lms = np.zeros((21, 3))
    right_hand_lms = np.zeros((21, 3))
    if hands_results.multi_hand_landmarks:
        for hand_lm_obj in hands_results.multi_hand_landmarks:
            hand_lms_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm_obj.landmark])
            hand_label = hands_results.multi_handedness[hands_results.multi_hand_landmarks.index(hand_lm_obj)].classification[0].label
            
            if hand_label == "Right":
                right_hand_lms = hand_lms_array
            else:
                left_hand_lms = hand_lms_array
                
    left_shoulder = pose_landmarks[1]
    right_shoulder = pose_landmarks[2]
    if np.sum(left_shoulder) == 0:
        center_shoulder = right_shoulder
    elif np.sum(right_shoulder) == 0:
        center_shoulder = left_shoulder
    else:
        center_shoulder = (left_shoulder + right_shoulder) / 2.0
    
    if np.sum(center_shoulder) != 0:
        norm_pose_lms = pose_landmarks - center_shoulder
        norm_left_hand_lms = left_hand_lms - center_shoulder
        norm_right_hand_lms = right_hand_lms - center_shoulder
    else:
        norm_pose_lms = pose_landmarks
        norm_left_hand_lms = left_hand_lms
        norm_right_hand_lms = right_hand_lms

    raw_landmarks = np.concatenate([
        norm_pose_lms.flatten(),
        norm_left_hand_lms.flatten(),
        norm_right_hand_lms.flatten()
    ])

    kinematic_features = calculate_kinematic_features(
        norm_pose_lms, norm_left_hand_lms, norm_right_hand_lms,
        prev_pose, prev_left, prev_right
    )

    all_features = np.concatenate([raw_landmarks, kinematic_features])

    return all_features, norm_pose_lms, norm_left_hand_lms, norm_right_hand_lms, pose_results, hands_results

# --- Model Architecture ---
class SignLanguageModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignLanguageModel, self).__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        final_output = lstm_out[:, -1, :]
        logits = self.classifier(final_output)
        
        return logits

def generate_emergency_sentence(words):
    prompt = f"다음 단어들을 사용하여 긴급 상황에서 구급대원에게 전달할 수 있는 간단한 한국어 문장 하나를 생성하세요: {', '.join(words)}. 문장은 간결하고 명확해야 하며 '~해주세요' 형태로 작성하세요."
    try:
        response = gemini_model.generate_content(prompt)
        sentence = response.text.strip()
        log_debug(f"Gemini API 문장 생성 성공 (generate_emergency_sentence): {sentence}")
        return sentence
    except Exception as e:
        log_debug(f"Gemini API 문장 생성 실패 (generate_emergency_sentence): {str(e)}")
        return " ".join(words) + " 도와주세요."

# --- Database Models ---
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

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- FastAPI Initialization and Endpoint ---
app = FastAPI(title="Hope API - Sign Language Recognition", version="1.0.0")

class PredictionRequest(BaseModel):
    image_sequence: List[str]  # List of Base64 encoded images

class PredictionResponse(BaseModel):
    words: List[Dict[str, Union[str, float]]]
    sentence: str
    message: str

# --- Load model and class mapping ---
# **아래 경로를 실제 파일의 절대 경로로 수정하세요.**
LABELS_CSV_PATH = "/Users/mango/hope_api/labels.csv"
WORD_LIST_CSV_PATH = "/Users/mango/hope_api/SL_Partner_Word_List_01.csv"
MODEL_PATH = "/Users/mango/hope_api/model2/best_model_gemini.pth"

# Load labels and word mapping
if not os.path.exists(LABELS_CSV_PATH):
    raise FileNotFoundError(f"Labels CSV file '{LABELS_CSV_PATH}' not found. Please check the path.")
df = pd.read_csv(LABELS_CSV_PATH)
label_encoder = LabelEncoder()
label_encoder.fit(df['labels'].values.ravel())
NUM_CLASSES = len(label_encoder.classes_)

if not os.path.exists(WORD_LIST_CSV_PATH):
    raise FileNotFoundError(f"Word list CSV file '{WORD_LIST_CSV_PATH}' not found. Please upload the file.")
word_df = pd.read_csv(WORD_LIST_CSV_PATH)
word_mapping = dict(zip(word_df['number'], word_df['word']))

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageModel(input_size=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run the training script first to save the model.")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


class StreamingVideoProcessor:
    """Process video stream and make predictions on sliding windows of frames"""
    
    def __init__(self, window_size=30, stride=15):
        self.window_size = window_size  # Number of frames for one prediction
        self.stride = stride  # How many frames to slide the window
        self.frame_buffer = []
        self.prev_pose = None
        self.prev_left = None
        self.prev_right = None
        self.frame_count = 0
        
    async def process_frame(self, frame):
        """Process a single frame and return prediction if window is ready"""
        # Extract features from frame
        all_features, curr_pose, curr_left, curr_right, _, _ = extract_all_features(
            frame, self.prev_pose, self.prev_left, self.prev_right
        )
        
        # Update previous landmarks
        self.prev_pose = curr_pose
        self.prev_left = curr_left
        self.prev_right = curr_right
        
        # Add to buffer
        self.frame_buffer.append(all_features)
        self.frame_count += 1
        
        # Check if we have enough frames for prediction
        if len(self.frame_buffer) >= self.window_size:
            # Make prediction on current window
            prediction = await self._predict_window()
            
            # Slide the window
            if self.frame_count % self.stride == 0:
                self.frame_buffer = self.frame_buffer[self.stride:]
            
            return prediction
        
        return None
    
    async def _predict_window(self):
        """Make prediction on current window of frames"""
        if len(self.frame_buffer) < 10:  # Minimum frames
            return None
            
        # Convert buffer to tensor
        sequence_tensor = torch.stack([
            torch.tensor(s, dtype=torch.float32) for s in self.frame_buffer[:self.window_size]
        ])
        
        # Pad if needed
        if sequence_tensor.size(0) < MAX_SEQ_LENGTH:
            sequence_tensor = F.pad(
                sequence_tensor, 
                (0, 0, 0, MAX_SEQ_LENGTH - sequence_tensor.size(0)), 
                'constant', 0
            )
        
        sequence_tensor = sequence_tensor.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(sequence_tensor)
        
        probs = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(3, probs.size(1)))
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = label_encoder.inverse_transform([idx.cpu().numpy()])[0]
            word = word_mapping.get(label, label)
            predictions.append({
                "word": word,
                "confidence": float(prob.cpu().numpy())
            })
        
        return predictions

class VideoPredictionResponse(BaseModel):
    prediction: Dict[str, Union[str, float]]  # Single prediction result
    top_predictions: List[Dict[str, Union[str, float]]]  # Top 3 predictions
    summary: Dict[str, Union[int, float, str]]
    sentence: str

@app.post("/predict-video", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(..., description="Video file to process"),
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Process entire video using same logic as predict_sequence.
    
    Args:
        file: Video file upload
    
    Returns:
        Prediction response matching predict_sequence format
    """
    import tempfile
    
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
            user_id=user_id,
            session_id=session_id,
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
        
        # Extract features from all frames (same as predict_sequence)
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
        if frame_count < 10:  # Minimum frames for a meaningful prediction
            return PredictionResponse(
                words=[],
                sentence="",
                message=f"Sequence too short ({frame_count} frames). Minimum 10 frames required."
            )
        
        # Same processing as predict_sequence
        sequence_list = list(sequence_buffer)
        sequence_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sequence_list])
        sequence_tensor = F.pad(sequence_tensor, (0, 0, 0, MAX_SEQ_LENGTH - sequence_tensor.size(0)), 'constant', 0)
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
            
            # 문장 생성 로직: 단어가 2개 이상일 때만 문장 생성
            if len(words_list) >= 2:
                sentence_text = generate_emergency_sentence([w['word'] for w in words_list])
            else:
                message = "예측된 단어가 충분하지 않아 문장을 생성할 수 없습니다."
            
            # Save chat record to database
            chat_record = ChatRecord(
                user_id=user_id,
                session_id=session_id,
                predicted_word=predicted_word,
                confidence=confidence,
                generated_sentence=sentence_text,
                input_type="video",
                frame_count=len(frames)
            )
            db.add(chat_record)
            db.commit()
            db.refresh(chat_record)
            
            # Update video record with chat record link
            video_record.chat_record_id = chat_record.id
            video_record.is_processed = True
            db.commit()
        else:
            message = "Prediction failed with low confidence."
            # Still mark video as processed
            video_record.is_processed = True
            db.commit()
        
        return PredictionResponse(
            words=words_list,
            sentence=sentence_text,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/predict_sequence", response_model=PredictionResponse)
async def predict_sequence(
    request: PredictionRequest,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
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
    if frame_count < 10:  # Minimum frames for a meaningful prediction
        return PredictionResponse(
            words=[],
            sentence="",
            message=f"Sequence too short ({frame_count} frames). Minimum 10 frames required."
        )

    sequence_list = list(sequence_buffer)
    sequence_tensor = torch.stack([torch.tensor(s, dtype=torch.float32) for s in sequence_list])
    sequence_tensor = F.pad(sequence_tensor, (0, 0, 0, MAX_SEQ_LENGTH - sequence_tensor.size(0)), 'constant', 0)
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
        
        # 문장 생성 로직: 단어가 2개 이상일 때만 문장 생성
        if len(words_list) >= 2:
            sentence_text = generate_emergency_sentence([w['word'] for w in words_list])
        else:
            message = "예측된 단어가 충분하지 않아 문장을 생성할 수 없습니다."
        
        # Save chat record to database
        chat_record = ChatRecord(
            user_id=user_id,
            session_id=session_id,
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
        message=message
    )

# --- Database Query Endpoints ---
@app.get("/chat-records")
async def get_chat_records(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get chat records with optional filtering"""
    query = db.query(ChatRecord)
    
    if user_id:
        query = query.filter(ChatRecord.user_id == user_id)
    if session_id:
        query = query.filter(ChatRecord.session_id == session_id)
    
    records = query.order_by(ChatRecord.created_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "records": [
            {
                "id": record.id,
                "user_id": record.user_id,
                "session_id": record.session_id,
                "predicted_word": record.predicted_word,
                "confidence": record.confidence,
                "generated_sentence": record.generated_sentence,
                "input_type": record.input_type,
                "frame_count": record.frame_count,
                "created_at": record.created_at,
                "updated_at": record.updated_at
            } for record in records
        ],
        "total": query.count()
    }

@app.get("/video-records")
async def get_video_records(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get video records with optional filtering"""
    query = db.query(VideoRecord)
    
    if user_id:
        query = query.filter(VideoRecord.user_id == user_id)
    if session_id:
        query = query.filter(VideoRecord.session_id == session_id)
    
    records = query.order_by(VideoRecord.created_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "records": [
            {
                "id": record.id,
                "user_id": record.user_id,
                "session_id": record.session_id,
                "filename": record.filename,
                "file_size": record.file_size,
                "file_extension": record.file_extension,
                "duration": record.duration,
                "frame_count": record.frame_count,
                "is_processed": record.is_processed,
                "chat_record_id": record.chat_record_id,
                "created_at": record.created_at,
                "updated_at": record.updated_at
            } for record in records
        ],
        "total": query.count()
    }

@app.get("/")
async def root():
    return {"message": "Sign Language Prediction API is running."}
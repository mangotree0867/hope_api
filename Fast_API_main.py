from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import requests
from typing import List, Dict
import asyncio
from datetime import datetime

app = FastAPI()

# 디버깅 로그
debug_log_file = "debug_log.txt"
def log_debug(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    print(log_message.strip())
    with open(debug_log_file, "a", encoding="utf-8") as f:
        f.write(log_message)

# MediaPipe Pose 및 Hands 초기화
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# CNNLSTM 모델 (훈련 코드와 동일)
class CNNLSTM(nn.Module):
    def __init__(self, input_dim=189, cnn_filters=32, kernel_size=3, lstm_hidden_size=64, num_layers=1, num_classes=12, dropout=0.5):
        super(CNNLSTM, self).__init__()
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters
        self.cnn = nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(cnn_filters)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x = x.reshape(batch_size * self.input_dim, 1, x.size(2))  # [batch * input_dim, 1, seq_len]
        x = self.cnn(x)  # [batch * input_dim, cnn_filters, seq_len//2]
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(batch_size, self.input_dim, self.cnn_filters, x.size(2))  # [batch, input_dim, cnn_filters, seq_len//2]
        x = x.transpose(1, 2)  # [batch, cnn_filters, input_dim, seq_len//2]
        x = x.reshape(batch_size, -1, self.input_dim)  # [batch, seq_len//2, input_dim]
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        last_indices = (lengths//2) - 1
        output = output[range(batch_size), last_indices, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

# 모델 로드
model = CNNLSTM(input_dim=189, cnn_filters=32, kernel_size=3, lstm_hidden_size=64, num_layers=1, num_classes=12, dropout=0.5)
model.load_state_dict(torch.load("best_cnnlstm_model.pth", map_location=model.device))
model.eval()

# 클래스 레이블 (CSV 기반)
LABELS = ["신고하세요(경찰)", "구급차", "가렵다", "가스", "가슴", "가시", "갇히다", "감금", "감전", "강", "강풍", "개"]

# 동작 경계 감지용 Pose 랜드마크 인덱스
POSE_LANDMARKS = [(12, 14), (14, 16), (11, 13), (13, 15)]

# HyperCLOVA X API 호출 (가상)
async def call_hyperclova(words: List[str]) -> str:
    prompt = f"응급상황에 맞는 간단하고 명확한 한국어 문장으로 변환, 모든 단어 포함: {', '.join(words)}"
    response = {"text": f"{', '.join(words)}!"}  # 실제 API 호출 구현 필요
    return response["text"]

# 벡터 변화량 계산 (Pose 랜드마크)
def compute_motion(landmarks_prev: np.ndarray, landmarks_curr: np.ndarray) -> float:
    motion = 0.0
    for idx1, idx2 in POSE_LANDMARKS:
        vec_prev = np.array([landmarks_prev[idx2].x - landmarks_prev[idx1].x,
                             landmarks_prev[idx2].y - landmarks_prev[idx1].y])
        vec_curr = np.array([landmarks_curr[idx2].x - landmarks_curr[idx1].x,
                             landmarks_curr[idx2].y - landmarks_curr[idx1].y])
        motion += np.linalg.norm(vec_curr - vec_prev)
    return motion / len(POSE_LANDMARKS)

# 프레임당 특징 추출 (Pose + Hands, 시퀀스 제작 코드와 동일)
def extract_features(pose_landmarks, hands_landmarks) -> np.ndarray:
    # Pose 특징: 33개 랜드마크 × x,y,z = 99 값
    pose_features = np.zeros((33, 3))
    if pose_landmarks:
        for i in range(33):
            pose_features[i] = [pose_landmarks.landmark[i].x, pose_landmarks.landmark[i].y, pose_landmarks.landmark[i].z]
    
    # Hands 특징: 왼손/오른손 21개 랜드마크 × x,y,z = 126 값
    hands_features = np.zeros((2 * 21, 3))
    if hands_landmarks:
        for hand_idx, hand in enumerate(hands_landmarks):
            for i in range(21):
                hands_features[hand_idx * 21 + i] = [hand.landmark[i].x, hand.landmark[i].y, hand.landmark[i].z]
    
    # 결합 및 평탄화: [99 + 126 = 189]
    features = np.concatenate([pose_features.flatten(), hands_features.flatten()])
    return features

# 130프레임 패딩
def pad_sequence(sequence: np.ndarray, max_len: int = 130) -> np.ndarray:
    seq_len = len(sequence)
    if seq_len >= max_len:
        return sequence[:max_len]
    padding = np.zeros((max_len - seq_len, sequence.shape[1]))
    return np.vstack([sequence, padding])

# 단어 예측
def predict_sign(sequence: np.ndarray, length: int) -> tuple[str, float]:
    # Min-max 정규화 (훈련 코드와 동일)
    sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min() + 1e-8)
    sequence = pad_sequence(sequence)
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(model.device)
    lengths = torch.tensor([length//2], dtype=torch.int64).to(model.device)
    with torch.no_grad():
        output = model(input_tensor, lengths)
        probabilities = torch.softmax(output, dim=1)
        confidence, idx = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        label = LABELS[idx.item()]
    return label, confidence

# 영상 처리 API
@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    # 영상 저장
    video_data = await file.read()
    with open("temp.mp4", "wb") as f:
        f.write(video_data)
    cap = cv2.VideoCapture("temp.mp4")
    
    # 프레임별 특징 추출
    features_list = []
    pose_landmarks_list = []  # 동작 감지용
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose 랜드마크
        pose_results = mp_pose.process(frame_rgb)
        pose_landmarks = pose_results.pose_landmarks
        pose_landmarks_list.append(pose_landmarks)
        
        # Hands 랜드마크
        hands_results = mp_hands.process(frame_rgb)
        hands_landmarks = hands_results.multi_hand_landmarks
        
        # 특징 추출
        features = extract_features(pose_landmarks, hands_landmarks)
        features_list.append(features)
    
    cap.release()
    log_debug(f"총 {len(features_list)} 프레임 처리")

    # 동작 경계 감지
    motion_threshold_start = 0.01
    motion_threshold_stop = 0.001
    segments = []
    segment_lengths = []
    current_segment = []
    prev_landmarks = None

    for i, pose_landmarks in enumerate(pose_landmarks_list):
        if pose_landmarks and prev_landmarks:
            motion = compute_motion(np.array(prev_landmarks.landmark), np.array(pose_landmarks.landmark))
            if motion > motion_threshold_start and not current_segment:
                current_segment = [features_list[i]]  # 동작 시작
            elif motion < motion_threshold_stop and current_segment:
                if len(current_segment) >= 15:  # 최소 0.5초
                    segments.append(np.array(current_segment))
                    segment_lengths.append(len(current_segment))
                current_segment = []
            elif current_segment:
                current_segment.append(features_list[i])
        prev_landmarks = pose_landmarks

    if current_segment and len(current_segment) >= 15:
        segments.append(np.array(current_segment))
        segment_lengths.append(len(current_segment))
    log_debug(f"감지된 동작 구간: {len(segments)}개")

    # 단어 예측 및 중복 방지
    words = []
    confidences = []
    last_word = None
    for segment, length in zip(segments, segment_lengths):
        word, confidence = predict_sign(segment, length)
        if confidence >= 0.7 and word != last_word:
            words.append(word)
            confidences.append(confidence)
            last_word = word
    log_debug(f"예측된 단어: {words}")

    # 동작 종료 감지
    last_frames = pose_landmarks_list[-15:] if len(pose_landmarks_list) >= 15 else pose_landmarks_list
    is_terminated = True
    for i in range(1, len(last_frames)):
        if last_frames[i] and last_frames[i-1]:
            motion = compute_motion(np.array(last_frames[i-1].landmark), np.array(last_frames[i].landmark))
            if motion > motion_threshold_stop:
                is_terminated = False
                break
    log_debug(f"종료 감지: {'종료됨' if is_terminated else '종료되지 않음'}")

    # HyperCLOVA X 문장 생성
    sentence = ""
    if is_terminated and words:
        sentence = await call_hyperclova(words)
    log_debug(f"생성된 문장: {sentence}")

    # 결과 반환
    result = {
        "words": [{"word": w, "confidence": c} for w, c in zip(words, confidences)],
        "sentence": sentence
    }
    return result
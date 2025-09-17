import os
import sys
import warnings
import logging
import cv2
from fastapi import HTTPException
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import google.generativeai as genai
from typing import List, Dict, Union, Optional, Tuple
from dotenv import load_dotenv

from app.core.config import settings

# Load environment variables
load_dotenv()

# Logger 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 하이퍼파라미터 및 경로 - config에서 가져오기
MAX_SEQ_LENGTH = settings.MAX_SEQ_LENGTH
NUM_FEATURES = settings.NUM_FEATURES
LABELS_CSV = settings.LABELS_CSV_PATH
WORD_LIST_CSV = settings.WORD_LIST_CSV_PATH
MODEL_PATH = settings.MODEL_PATH


# 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")

# Gemini API 설정
genai.configure(api_key=settings.GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE
)
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=settings.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE
)

def log_debug(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] {message}")

def hands_detected(hands_results):
    return hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) == 2

def initialize_model_and_mappings():
    global model, label_encoder, word_mapping, NUM_CLASSES
    logger.info("모델 및 매핑 초기화 시작")
    
    if not os.path.exists(LABELS_CSV):
        raise HTTPException(status_code=500, detail=f"Labels CSV 파일 '{LABELS_CSV}'을 찾을 수 없습니다.")
    
    try:
        df = pd.read_csv(LABELS_CSV)
        label_encoder = LabelEncoder()
        label_encoder.fit(df['labels'])
        NUM_CLASSES = len(label_encoder.classes_)
        logger.info(f"클래스 수: {NUM_CLASSES}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Labels CSV 로드 실패: {str(e)}")
    
    if not os.path.exists(WORD_LIST_CSV):
        raise HTTPException(status_code=500, detail=f"Word list CSV 파일 '{WORD_LIST_CSV}'을 찾을 수 없습니다.")
    
    try:
        word_df = pd.read_csv(WORD_LIST_CSV)
        word_mapping = dict(zip(word_df['number'], word_df['word']))
        logger.info("단어 매핑 로드 완료")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word list CSV 로드 실패: {str(e)}")
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다.")
    
    try:
        model = SignLanguageModel(input_size=NUM_FEATURES, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        logger.info("모델 로드 완료")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로드 실패: {str(e)}")


# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
        if np.sum(hand_lms) == 0: 
            return [0.0] * 15
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
        if np.sum(hand_lms) == 0: 
            return [0.0, 0.0, 0.0]
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
        for hand_idx, hand_lm in enumerate(hands_results.multi_hand_landmarks):
            handedness_label = hands_results.multi_handedness[hand_idx].classification[0].label
            hand_lms_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
            if handedness_label == 'Right':
                right_hand_lms = hand_lms_coords
            elif handedness_label == 'Left':
                left_hand_lms = hand_lms_coords

    left_shoulder = pose_landmarks[1]
    right_shoulder = pose_landmarks[2]
    if np.sum(left_shoulder) == 0:
        center_shoulder = right_shoulder
    elif np.sum(right_shoulder) == 0:
        center_shoulder = left_shoulder
    else:
        center_shoulder = (left_shoulder + right_shoulder) / 2.0
    
    norm_pose_lms = pose_landmarks - center_shoulder
    norm_left_hand_lms = left_hand_lms - center_shoulder
    norm_right_hand_lms = right_hand_lms - center_shoulder

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
    
async def process_and_predict_from_mp4(file_path: str):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        log_debug(f"비디오 파일 '{file_path}'를 열 수 없습니다.")
        raise HTTPException(status_code=400, detail="비디오 파일을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_pose, prev_left, prev_right = None, None, None
    action_active = False
    hands_start_time = None
    no_hands_start_time = None
    current_sequence = []
    detected_words = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        all_features, curr_pose, curr_left, curr_right, pose_results, hands_results = \
            extract_all_features(frame, prev_pose, prev_left, prev_right)

        prev_pose, prev_left, prev_right = curr_pose, curr_left, curr_right

        detected = hands_detected(hands_results)

        if detected:
            no_hands_start_time = None
            if hands_start_time is None:
                hands_start_time = current_time
            elif not action_active and current_time - hands_start_time >= 1.0:
                log_debug(f"동작 시작 감지! ({current_time:.2f}초)")
                action_active = True
                current_sequence = []
        else:
            both_hands_missing = hands_results.multi_hand_landmarks is None
            if action_active:
                if both_hands_missing:
                    if no_hands_start_time is None:
                        no_hands_start_time = current_time
                    elif current_time - no_hands_start_time >= 1.0:
                        log_debug(f"동작 종료 감지! ({current_time:.2f}초)")
                        action_active = False
                        no_hands_start_time = None

                        if len(current_sequence) >= 10:
                            sequence_list = current_sequence
                            sequence_tensor = torch.tensor(sequence_list, dtype=torch.float32)
                            
                            if sequence_tensor.size(0) > MAX_SEQ_LENGTH:
                                sequence_tensor = sequence_tensor[:MAX_SEQ_LENGTH]
                            elif sequence_tensor.size(0) < MAX_SEQ_LENGTH:
                                padding_size = MAX_SEQ_LENGTH - sequence_tensor.size(0)
                                sequence_tensor = F.pad(sequence_tensor, (0, 0, 0, padding_size), 'constant', 0)
                            
                            sequence_tensor = sequence_tensor.unsqueeze(0).to(device)

                            with torch.no_grad():
                                outputs = model(sequence_tensor)
                            
                            probs = F.softmax(outputs, dim=1)
                            predicted_idx = outputs.argmax(1).item()
                            confidence = probs[0][predicted_idx].item()
                            
                            if confidence > 0.5:
                                predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
                                predicted_word = word_mapping.get(predicted_label, predicted_label)
                                detected_words.append({"word": predicted_word, "confidence": float(confidence)})
                                log_debug(f"예측: '{predicted_word}' (신뢰도: {confidence:.2%})")
                        current_sequence = []
                else:
                    no_hands_start_time = None
            hands_start_time = None

        if action_active:
            current_sequence.append(all_features)

    if action_active and len(current_sequence) >= 10:
        log_debug("비디오 종료: 남은 동작 처리")
        sequence_list = current_sequence
        sequence_tensor = torch.tensor(sequence_list, dtype=torch.float32)
        
        if sequence_tensor.size(0) > MAX_SEQ_LENGTH:
            sequence_tensor = sequence_tensor[:MAX_SEQ_LENGTH]
        elif sequence_tensor.size(0) < MAX_SEQ_LENGTH:
            padding_size = MAX_SEQ_LENGTH - sequence_tensor.size(0)
            sequence_tensor = F.pad(sequence_tensor, (0, 0, 0, padding_size), 'constant', 0)
        
        sequence_tensor = sequence_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(sequence_tensor)
        
        probs = F.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(1).item()
        confidence = probs[0][predicted_idx].item()
        
        if confidence > 0.5:
            predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
            predicted_word = word_mapping.get(predicted_label, predicted_label)
            detected_words.append({"word": predicted_word, "confidence": float(confidence)})
            log_debug(f"예측: '{predicted_word}' (신뢰도: {confidence:.2%})")

    cap.release()
    return detected_words
    
def generate_emergency_sentence(words):
    prompt = f"다음 단어들을 사용하여 긴급 상황에서 구급대원에게 전달할 수 있는 간단한 한국어 문장 하나를 생성하세요: {', '.join(words)}. 문장은 간결하고 명확해야 하며 '~해주세요' 형태로 작성하세요."
    try:
        if gemini_model:
            response = gemini_model.generate_content(prompt)
            sentence = response.text.strip()
            log_debug(f"Gemini API 문장 생성 성공: {sentence}")
            return sentence
        else:
            log_debug("Gemini model not available")
            return " ".join(words) + " 도와주세요."
    except Exception as e:
        log_debug(f"Gemini API 문장 생성 실패: {str(e)}")
        return " ".join(words) + " 도와주세요."

# class StreamingVideoProcessor:
#     """비디오 스트림을 처리하고 슬라이딩 윈도우 프레임에서 예측 수행"""

#     def __init__(self, window_size=30, stride=15):
#         self.window_size = window_size
#         self.stride = stride
#         self.frame_buffer = []
#         self.prev_pose = None
#         self.prev_left = None
#         self.prev_right = None
#         self.frame_count = 0

#     async def process_frame(self, frame):
#         """단일 프레임 처리 및 윈도우가 준비되면 예측 반환"""
#         all_features, curr_pose, curr_left, curr_right, _, _ = extract_all_features(
#             frame, self.prev_pose, self.prev_left, self.prev_right
#         )

#         self.prev_pose = curr_pose
#         self.prev_left = curr_left
#         self.prev_right = curr_right

#         self.frame_buffer.append(all_features)
#         self.frame_count += 1

#         if len(self.frame_buffer) >= self.window_size:
#             prediction = await self._predict_window()

#             if self.frame_count % self.stride == 0:
#                 self.frame_buffer = self.frame_buffer[self.stride:]

#             return prediction

#         return None

#     async def _predict_window(self):
#         """현재 프레임 윈도우에서 예측 수행"""
#         if len(self.frame_buffer) < 10:
#             return None

#         sequence_tensor = torch.stack([
#             torch.tensor(s, dtype=torch.float32) for s in self.frame_buffer[:self.window_size]
#         ])

#         if sequence_tensor.size(0) < settings.MAX_SEQ_LENGTH:
#             sequence_tensor = F.pad(
#                 sequence_tensor,
#                 (0, 0, 0, settings.MAX_SEQ_LENGTH - sequence_tensor.size(0)),
#                 'constant', 0
#             )

#         sequence_tensor = sequence_tensor.unsqueeze(0).to(device)

#         with torch.no_grad():
#             outputs = model(sequence_tensor)

#         probs = F.softmax(outputs, dim=1)
#         top_probs, top_indices = torch.topk(probs, k=min(3, probs.size(1)))

#         predictions = []
#         for prob, idx in zip(top_probs[0], top_indices[0]):
#             label = label_encoder.inverse_transform([idx.cpu().numpy()])[0]
#             word = word_mapping.get(label, label)
#             predictions.append({
#                 "word": word,
#                 "confidence": float(prob.cpu().numpy())
#             })

#         return predictions

# 전역 변수 초기화
model = None
label_encoder = None
word_mapping = None
NUM_CLASSES = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 앱 시작 시 초기화
initialize_model_and_mappings()
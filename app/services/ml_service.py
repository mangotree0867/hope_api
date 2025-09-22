from collections import deque
import json
import os
import re
import sys
import tempfile
import warnings
import logging
import cv2
from fastapi import File, HTTPException, UploadFile
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
from app.services.model import SignLanguageTransformer
from app.services.preproces import FeatureExtractor

# Load environment variables
load_dotenv()

# Logger 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCALER_PATH = settings.SCALER_PATH
LABEL_MAP_PATH = settings.LABEL_MAP_PATH
LABEL_WORD_LIST_PATH = settings.LABEL_WORD_LIST_PATH
WEIGHTS_PATH = settings.WEIGHTS_PATH
GOOGLE_API_KEY = settings.GOOGLE_API_KEY

# 모델 하이퍼파라미터
MAX_SEQ_LENGTH = 70
MODEL_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FEATURES_DIM = 347
MIN_SEQUENCE_LENGTH = 10
MIN_NO_HANDS_DURATION = 0.5
MIN_DETECTION_DURATION = 0.5
CONFIDENCE_THRESHOLD = 0.3

# --- 리소스 초기화 ---
feature_extractor = FeatureExtractor(scaler_path=SCALER_PATH)

# CSV 파일에서 단어 매핑 정보 로드
try:
    df = pd.read_csv(LABEL_WORD_LIST_PATH)
    csv_word_mapping = pd.Series(df.word.values, index=df.number).to_dict()
except FileNotFoundError:
    raise Exception(f"CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요: {LABEL_WORD_LIST_PATH}")

# 라벨 맵 로드 (인덱스 -> W_XXX 변환용)
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
# 딕셔너리를 뒤집어 {0: 'W_001', 1: 'W_002', ...} 형태로 만듭니다.
idx_to_label_map = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map) # 클래스 개수는 여기서 계산


# --- 모델 초기화 ---
model = SignLanguageTransformer(
    num_classes=NUM_CLASSES, input_features=INPUT_FEATURES_DIM, d_model=MODEL_DIM,
    nhead=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS, dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(DEVICE)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
model.eval()

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
warnings.filterwarnings("ignore", message="A column-vector y was passed when a 1d array was expected")

def predict_from_video(video_path, category, context):
    # category is already the Korean name from CategoryEnum (e.g., "외상", "내상", etc.)
    category_name = category

    context = re.sub(r'\s+', ' ', context.strip())
    context_words = context.split() if context else []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("비디오 파일을 열 수 없습니다.")
    sequence_buffer = deque(maxlen=MAX_SEQ_LENGTH)
    accumulated_words = []
    prev_data = {}
    detecting = False
    hands_start_time = None
    no_hands_start_time = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        features, current_data, hands_results = feature_extractor.get_combined_features(frame, prev_data)
        prev_data = current_data
        hands_detected = hands_results.multi_hand_landmarks is not None
        if hands_detected:
            no_hands_start_time = None
            if hands_start_time is None:
                hands_start_time = current_time_sec
            if not detecting and (current_time_sec - hands_start_time) >= MIN_DETECTION_DURATION:
                detecting = True
                sequence_buffer.clear()
        else:
            hands_start_time = None
        if not hands_detected and detecting:
            if no_hands_start_time is None:
                no_hands_start_time = current_time_sec
            elif (current_time_sec - no_hands_start_time) >= MIN_NO_HANDS_DURATION:
                if len(sequence_buffer) >= MIN_SEQUENCE_LENGTH:
                    sequence_tensor = torch.tensor(np.array(list(sequence_buffer)), dtype=torch.float32)
                    if sequence_tensor.shape[0] < MAX_SEQ_LENGTH:
                        padding = torch.zeros(MAX_SEQ_LENGTH - sequence_tensor.shape[0], INPUT_FEATURES_DIM)
                        sequence_tensor = torch.cat([sequence_tensor, padding], dim=0)
                    sequence_tensor = sequence_tensor.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(sequence_tensor)
                    probs = F.softmax(outputs, dim=1)
                    top_prob, top_idx = torch.topk(probs, 1, dim=1)
                    if top_prob[0, 0].item() > CONFIDENCE_THRESHOLD:
                        predicted_index = top_idx[0, 0].item()
                        
                        # f-string 대신, 뒤집은 맵을 사용하여 인덱스로부터 W_XXX 키를 조회합니다.
                        word_key = idx_to_label_map.get(predicted_index)
                        
                        if word_key: # 맵에 해당 인덱스가 있는 경우에만 처리
                            # CSV에서 로드한 딕셔너리를 사용하여 단어 조회
                            predicted_word = csv_word_mapping.get(word_key, "Unknown")
                            accumulated_words.append(predicted_word)
                detecting = False
                sequence_buffer.clear()
        if detecting:
            sequence_buffer.append(features)
    cap.release()
    sentence = "문장을 생성하기에 단어가 부족합니다."
    if gemini_model and len(accumulated_words) >= 2:
        prompt_parts = [
            f"카테고리: {category_name} (이 상황을 감안하여 구조대가 이해할 수 있는 긴급 상황 문장을 생성하세요).",
            f"수어 단어: {', '.join(accumulated_words)} (이 단어들을 문장에 반드시 포함하되, 순서는 자연스러운 한국어 표현에 맞게 조정하세요).",
            f"추가 context 단어: {', '.join(context_words)} (구조대가 현 상황을 판단하기 위해 사전에 입력한 핵심 단어들, 예: 골절, 다리, 무릎). 문장에 자연스럽게 통합하되, context 단어가 없으면 카테고리와 수어 단어만으로 문장을 생성하세요." if context_words else "Context 단어가 없으므로 카테고리와 수어 단어만 사용하여 문장을 생성하세요.",
            "문장은 반드시 '~해주세요' 형태로 끝나야 하며, 간결하고 명확해야 합니다. 새로운 목적어나 추가적인 상황 설명(예: '무거운 물체' 등)을 포함하지 마세요. 한국어 문법과 어순을 엄격히 준수하세요.",
            f"예시 문장: '{category_name} 상황에서 {' '.join(accumulated_words[:2] if accumulated_words else ['상황'])}입니다. 빠르게 도와주세요.'"
        ]
        prompt = " ".join([part for part in prompt_parts if part])
        try:
            response = gemini_model.generate_content(prompt)
            sentence = response.text.strip()
        except:
            sentence = f"{category_name} 상황에서 {' '.join(accumulated_words)}입니다. 빠르게 도와주세요."
    return {"recognized_words": list(set(accumulated_words)), "generated_sentence": sentence}

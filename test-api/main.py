import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import deque
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from dotenv import load_dotenv

# 로컬 모듈 import
from model import SignLanguageTransformer
from preprocess import FeatureExtractor

# --- 1. 설정 및 초기화 ---
load_dotenv() # .env 파일에서 환경 변수 로드

# 경로 및 환경 변수
WEIGHTS_PATH = r"./weights/best_sign_language_model-gemini-v2_dim2.pth"
WORD_LIST_PATH = r"./data/SL_Partner_Word_List_01.csv"
GOOGLE_API_KEY = "AIzaSyAVC5CMiAupzwWGEoGay6vFEhUwwmBzgcw"

# 모델 하이퍼파라미터
MAX_SEQ_LENGTH = 70
MODEL_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 예측 로직 상수
MIN_SEQUENCE_LENGTH = 10
MIN_NO_HANDS_DURATION = 0.5
MIN_DETECTION_DURATION = 0.5
CONFIDENCE_THRESHOLD = 0.5

# --- 2. 전역 변수 및 객체 로딩 (서버 시작 시 1회 실행) ---
app = FastAPI(title="Sign Language Recognition API")
feature_extractor = FeatureExtractor()

# 단어 매핑 및 라벨 인코더 로드
try:
    word_df = pd.read_csv(WORD_LIST_PATH)
    word_mapping = dict(zip(word_df['number'].astype(str), word_df['word']))
    NUM_CLASSES = 48
    label_encoder = LabelEncoder()
    label_encoder.fit(list(word_mapping.keys())[:NUM_CLASSES])
except FileNotFoundError:
    raise RuntimeError(f"필수 파일 '{WORD_LIST_PATH}'를 찾을 수 없습니다. 서버를 시작할 수 없습니다.")

# 입력 특징 차원 (고정값 - 저장된 모델과 일치)
INPUT_FEATURES_DIM = 512

# 모델 로드
model = SignLanguageTransformer(
    num_classes=NUM_CLASSES,
    input_features=INPUT_FEATURES_DIM,
    d_model=MODEL_DIM,
    nhead=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(DEVICE)

try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
except FileNotFoundError:
    raise RuntimeError(f"모델 가중치 파일 '{WEIGHTS_PATH}'를 찾을 수 없습니다. 서버를 시작할 수 없습니다.")

# Gemini API 초기화
gemini_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        gemini_model = None # 초기화 실패 시 모델 비활성화

# --- 3. API 엔드포인트 ---
@app.post("/predict/")
async def predict_from_video(video_file: UploadFile = File(...)):
    """비디오 파일을 업로드받아 수어 단어를 인식하고, 긴급 문장을 생성합니다."""
    video_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video_file.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="업로드된 비디오 파일을 열 수 없습니다.")

        sequence_buffer = deque(maxlen=MAX_SEQ_LENGTH)
        accumulated_words = []
        prev_data = {}
        detecting = False
        hands_start_time = None
        no_hands_start_time = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

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
                            idx = top_idx[0, 0].item()
                            word_label = label_encoder.inverse_transform([idx])[0]
                            predicted_word = word_mapping.get(word_label, "Unknown")
                            accumulated_words.append(predicted_word)
                    
                    detecting = False
                    sequence_buffer.clear()
            
            if detecting:
                sequence_buffer.append(features)
        
        cap.release()

        sentence = "문장을 생성하기에 단어가 부족하거나 Gemini API를 사용할 수 없습니다."
        if gemini_model and len(accumulated_words) >= 2:
            prompt = f"다음 한국어 수어 단어들을 사용하여 긴급 상황에서 구급대원에게 전달할 수 있는 간결하고 명확한 문장 하나를 '~해주세요' 형태로 생성하세요: {', '.join(accumulated_words)}."
            try:
                response = gemini_model.generate_content(prompt)
                sentence = response.text.strip()
            except Exception as e:
                sentence = f"Gemini API 호출 중 오류 발생: {e}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {e}")
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

    return {"recognized_words": list(set(accumulated_words)), "generated_sentence": sentence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
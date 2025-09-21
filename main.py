import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import deque
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from dotenv import load_dotenv
import re
import tempfile
import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model import SignLanguageTransformer
from preprocess import FeatureExtractor

load_dotenv()

WEIGHTS_PATH = r"./weights/best_sign_language_model-gemini-v2_dim2.pth"
WORD_LIST_PATH = r"./data/SL_Partner_Word_List_01.csv"
GOOGLE_API_KEY = r"AIzaSyAVC5CMiAupzwWGEoGay6vFEhUwwmBzgcw"

MAX_SEQ_LENGTH = 70
MODEL_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 4
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_SEQUENCE_LENGTH = 10
MIN_NO_HANDS_DURATION = 0.5
MIN_DETECTION_DURATION = 0.5
CONFIDENCE_THRESHOLD = 0.3

CATEGORY_MAPPING = {
    "1": "외상",
    "2": "내상",
    "3": "화재상황",
    "4": "도심상황",
    "외상": "외상",
    "내상": "내상",
    "화재상황": "화재상황",
    "도심상황": "도심상황"
}

app = FastAPI()  # FastAPI 앱 생성

# 모델과 리소스 초기화 (앱 시작 시 한 번 로드)
feature_extractor = FeatureExtractor()

word_df = pd.read_csv(WORD_LIST_PATH)
word_mapping = dict(zip(word_df['number'].astype(str), word_df['word']))
NUM_CLASSES = 48
label_encoder = LabelEncoder()
label_encoder.fit(list(word_mapping.keys())[:NUM_CLASSES])

dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
temp_features, _, _ = feature_extractor.get_combined_features(dummy_img, {})
INPUT_FEATURES_DIM = len(temp_features)

model = SignLanguageTransformer(
    num_classes=NUM_CLASSES,
    input_features=INPUT_FEATURES_DIM,
    d_model=MODEL_DIM,
    nhead=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(DEVICE)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
model.eval()

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 원본 함수 (변경 없음)
def predict_from_video(video_path, category, context):
    if category not in CATEGORY_MAPPING:
        raise ValueError("유효하지 않은 카테고리입니다.")
    category_name = CATEGORY_MAPPING.get(category, "외상")

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

# API 엔드포인트
@app.post("/predict")
async def predict(
    video: UploadFile = UploadFile(...),  # 비디오 파일 업로드 (필수)
    category: str = Form(...),  # Form 데이터로 category 받음 (필수)
    context: str = Form("")  # Form 데이터로 context 받음 (선택)
):
    try:
        # 업로드된 비디오를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await video.read())
            temp_video_path = temp_file.name

        # 원본 함수 호출
        result = predict_from_video(temp_video_path, category, context)

        # 임시 파일 삭제
        os.unlink(temp_video_path)

        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

# 앱 실행 (uvicorn main:app --reload 로 로컬 테스트)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)